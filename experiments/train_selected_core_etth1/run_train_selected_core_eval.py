"""Re-evaluate ETTh1 current-best model with train-only core expert selection.

Phase A loads only router-train, selects the three core experts using
chronological OOF folds, and writes frozen_config_before_validation.json.
Phase B then loads router-val exactly once and evaluates the frozen model.
The ETTh1 test cache is refused.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.chronological_adaptive_costar.run_chronological_adaptive_costar import (  # noqa: E402
    SEEDS,
    Trial as ChronoTrial,
    chronological_online_weights,
    enforce_observable,
    load_static_winner_per_window,
    paired_bootstrap,
)
from experiments.expanded_expert_pool_costar.run_expanded_expert_pool import (  # noqa: E402
    Config as SpecialistConfig,
    run_causal_specialists,
)
from experiments.horizon_variable_adaptive_costar.run_hv_adaptive_costar import (  # noqa: E402
    Trial as HvTrial,
    chronological_hv_weights,
    per_location_abs_error,
    predict_from_hv_weights,
)
from experiments.oracle_weight_tournament.run_tournament import (  # noqa: E402
    FIXED3,
    load_cache,
    load_std,
    sample_mae,
    sample_mse,
    weighted_forecast,
)


OLD_FIXED3 = ("PatchTST", "iTransformer", "TimesNet")
BASELINE_MAE = 0.36364156007766724
PREVIOUS_BEST_MAE = 0.3631121516227722
SPECIALIST_CONFIG = SpecialistConfig("both", "variable", 0.95, 0.10, 0.02, 96)


def refuse_test(path: str | Path) -> None:
    if "test" in str(path).lower():
        raise ValueError(f"Refusing test path: {path}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def train_folds(n: int) -> list[tuple[int, int, int]]:
    min_train = int(round(n * 0.2))
    usable = n - min_train
    bounds = [min_train + i * usable // 4 for i in range(5)]
    return [(0, bounds[i], bounds[i + 1]) for i in range(4)]


def expert_indices(cache: Mapping[str, Any], experts: Sequence[str]) -> list[int]:
    names = list(cache["expert_names"])
    return [names.index(name) for name in experts]


def selected_forecasts(cache: Mapping[str, Any], expert_idx: Sequence[int]) -> torch.Tensor:
    return cache["prediction_stack"][..., list(expert_idx)].to(torch.float32)


def optional_prediction(cache: Mapping[str, Any], name: str) -> torch.Tensor:
    names = list(cache["expert_names"])
    return cache["prediction_stack"][..., names.index(name)].to(torch.float32)


def metrics(cache: Mapping[str, Any], std: torch.Tensor, pred: torch.Tensor) -> dict[str, Any]:
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.bool)
    mae = sample_mae(pred, target, mask, std)
    mse = sample_mse(pred, target, mask, std)
    return {"mae": float(mae.mean()), "mse": float(mse.mean()), "per_window_mae": mae, "per_window_mse": mse}


def normalized_abs_error(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return ((pred - target) / std.view(1, 1, -1)).abs() * mask.to(torch.float32)


def per_location_abs_error_for_indices(cache: Mapping[str, Any], std: torch.Tensor, expert_idx: Sequence[int]) -> torch.Tensor:
    forecasts = selected_forecasts(cache, expert_idx)
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.float32)
    return ((forecasts - target.unsqueeze(-1)) / std.view(1, 1, -1, 1)).abs() * mask.unsqueeze(-1)


def select_core_on_router_train(cache: Mapping[str, Any], std: torch.Tensor) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    names = list(cache["expert_names"])
    stack = cache["prediction_stack"].to(torch.float32)
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.bool)
    folds = train_folds(int(cache["num_windows"]))
    rows: list[dict[str, Any]] = []
    for idx in itertools.combinations(range(len(names)), 3):
        pred = stack[..., list(idx)].mean(dim=-1)
        mae_chunks = []
        mse_chunks = []
        fold_rows = []
        for fold_id, (_, eval_lo, eval_hi) in enumerate(folds):
            mae = sample_mae(pred[eval_lo:eval_hi], target[eval_lo:eval_hi], mask[eval_lo:eval_hi], std)
            mse = sample_mse(pred[eval_lo:eval_hi], target[eval_lo:eval_hi], mask[eval_lo:eval_hi], std)
            mae_chunks.append(mae)
            mse_chunks.append(mse)
            fold_rows.append({"fold": fold_id, "eval_lo": eval_lo, "eval_hi": eval_hi, "mae": float(mae.mean()), "mse": float(mse.mean())})
        row = {
            "experts": [names[i] for i in idx],
            "expert_indices": list(idx),
            "subset": "+".join(names[i] for i in idx),
            "pooled_oof_mae": float(torch.cat(mae_chunks).mean()),
            "pooled_oof_mse": float(torch.cat(mse_chunks).mean()),
            "worst_fold_mae": max(r["mae"] for r in fold_rows),
            "fold_rows": fold_rows,
        }
        rows.append(row)
    rows = sorted(rows, key=lambda r: (float(r["pooled_oof_mae"]), float(r["pooled_oof_mse"]), float(r["worst_fold_mae"]), str(r["subset"])))
    return rows, rows[0]


def parameterized_current_base_prediction(
    cache: Mapping[str, Any],
    train_cache_for_init: Mapping[str, Any],
    std: torch.Tensor,
    expert_idx: Sequence[int],
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    starts = cache["absolute_window_starts"].to(torch.long)
    horizon = int(cache["forecast_horizon"])
    if not bool(torch.all(starts[1:] > starts[:-1])):
        raise ValueError("Cache starts must be chronological")
    train_expert_err = per_location_abs_error_for_indices(train_cache_for_init, std, expert_idx).mean(dim=(1, 2))
    val_expert_err = per_location_abs_error_for_indices(cache, std, expert_idx).mean(dim=(1, 2))
    online_weights, online_extra = chronological_online_weights(
        starts=starts,
        expert_mae=val_expert_err,
        horizon=horizon,
        trial=ChronoTrial("ema", "ema_decay0.97_temp0.1", decay=0.97, temperature=0.1),
        train_mean_mae=train_expert_err.mean(dim=0),
        mode="ema",
    )
    selected_names = [list(cache["expert_names"])[i] for i in expert_idx]
    if tuple(selected_names) == OLD_FIXED3:
        static_weights, _, _ = load_static_winner_per_window(seed, cache, std, device)
    else:
        # The static neural winner was trained only for the old fixed-three.
        # If train-only selection ever changes, this fallback keeps the model
        # frozen and validation-safe rather than silently using incompatible weights.
        static_weights = torch.full((int(cache["num_windows"]), 3), 1.0 / 3.0)
    chrono_weights = 0.5 * static_weights + 0.5 * online_weights
    chrono_weights = chrono_weights / chrono_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    forecasts = selected_forecasts(cache, expert_idx)
    chrono_pred = weighted_forecast(forecasts, chrono_weights)
    train_hv_err_mean = per_location_abs_error_for_indices(train_cache_for_init, std, expert_idx).mean(dim=0)
    val_hv_err = per_location_abs_error_for_indices(cache, std, expert_idx)
    hv_weights, hv_extra = chronological_hv_weights(
        starts=starts,
        train_err_mean=train_hv_err_mean,
        val_err=val_hv_err,
        horizon=horizon,
        trial=HvTrial("hv_ema", "hvema_lowrank1_decay0.95_temp0.1", mode="hv_lowrank", rank=1, decay=0.95, temperature=0.1),
    )
    hv_pred = predict_from_hv_weights(forecasts, hv_weights)
    pred = 0.25 * chrono_pred + 0.75 * hv_pred
    return pred, {
        "chrono_num_updates": online_extra.get("num_updates"),
        "hv_num_updates": hv_extra.get("num_updates"),
        "static_weight_source": "existing_static_winner" if tuple(selected_names) == OLD_FIXED3 else "equal_fallback_no_static_artifact",
        **{f"mean_weight_{selected_names[i]}": float(hv_weights[..., i].mean()) for i in range(3)},
    }


def evaluate_expanded(
    cache: Mapping[str, Any],
    train_cache: Mapping[str, Any],
    std: torch.Tensor,
    expert_idx: Sequence[int],
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    base, base_extra = parameterized_current_base_prediction(cache, train_cache, std, expert_idx, seed, device)
    train_base, _ = parameterized_current_base_prediction(train_cache, train_cache, std, expert_idx, 7, device)
    target_train = train_cache["targets"].to(torch.float32)
    mask_train = train_cache["target_masks"].to(torch.bool)
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.bool)
    d_train = optional_prediction(train_cache, "DLinear")
    m_train = optional_prediction(train_cache, "ModernTCN")
    d_pred = optional_prediction(cache, "DLinear")
    m_pred = optional_prediction(cache, "ModernTCN")
    init_base_err = normalized_abs_error(train_base, target_train, mask_train, std)
    init_d_err = normalized_abs_error(d_train, target_train, mask_train, std)
    init_m_err = normalized_abs_error(m_train, target_train, mask_train, std)
    pred, extra, _ = run_causal_specialists(
        cache["absolute_window_starts"].to(torch.long),
        base,
        d_pred,
        m_pred,
        target,
        mask,
        std,
        SPECIALIST_CONFIG,
        init_base_err,
        init_d_err,
        init_m_err,
    )
    return pred, {**base_extra, **extra}


def phase_a(args: argparse.Namespace) -> None:
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = ROOT / args.train_cache
    norm_path = ROOT / args.normalizer_checkpoint
    refuse_test(train_path)
    refuse_test(norm_path)
    train_cache = load_cache(train_path, "router_train_20_60")
    std = load_std(norm_path, int(train_cache["num_features"]))
    rows, selected = select_core_on_router_train(train_cache, std)
    write_csv(out_dir / "router_train_all_triples.csv", [{k: v for k, v in row.items() if k != "fold_rows"} for row in rows])
    write_json(out_dir / "router_train_all_triples.json", rows)
    frozen = {
        "phase": "A_frozen_before_validation",
        "router_val_loaded": False,
        "selected_three_experts": selected["experts"],
        "selected_expert_indices": selected["expert_indices"],
        "router_train_oof_mae": selected["pooled_oof_mae"],
        "router_train_oof_mse": selected["pooled_oof_mse"],
        "router_train_worst_fold_mae": selected["worst_fold_mae"],
        "selection_metric": "pooled chronological OOF MAE",
        "tie_breakers": ["OOF MSE", "worst-fold MAE", "deterministic expert-name ordering"],
        "all_model_hyperparameters": {
            "chrono_ema_decay": 0.97,
            "chrono_temperature": 0.1,
            "chrono_blend_alpha": 0.5,
            "hv_mode": "hv_lowrank",
            "hv_rank": 1,
            "hv_decay": 0.95,
            "hv_temperature": 0.1,
            "hybrid_chrono_weight": 0.25,
            "hybrid_hv_weight": 0.75,
            "specialist_config": {"name": SPECIALIST_CONFIG.name, **asdict(SPECIALIST_CONFIG)},
        },
        "blend_coefficients": {"chrono_pred": 0.25, "hv_pred": 0.75},
        "cache_paths": {"router_train": args.train_cache, "normalizer_checkpoint": args.normalizer_checkpoint},
        "cache_hashes": {"router_train_sha256": sha256_file(train_path), "normalizer_sha256": sha256_file(norm_path)},
        "expert_names_in_cache": list(train_cache["expert_names"]),
        "train_windows": int(train_cache["num_windows"]),
        "train_start_min": int(train_cache["absolute_window_starts"].min()),
        "train_start_max": int(train_cache["absolute_window_starts"].max()),
        "forecast_horizon": int(train_cache["forecast_horizon"]),
        "num_variables": int(train_cache["num_features"]),
        "old_validation_selected_fixed3": list(OLD_FIXED3),
        "selected_matches_old_fixed3": tuple(selected["experts"]) == OLD_FIXED3,
        "test_cache_loaded": False,
    }
    write_json(out_dir / "frozen_config_before_validation.json", frozen)


def phase_b(args: argparse.Namespace) -> None:
    out_dir = ROOT / args.out_dir
    frozen_path = out_dir / "frozen_config_before_validation.json"
    if not frozen_path.exists():
        raise FileNotFoundError("Run Phase A before Phase B")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen.get("router_val_loaded"):
        raise RuntimeError("Frozen config indicates validation was loaded during Phase A")
    train_path = ROOT / args.train_cache
    val_path = ROOT / args.val_cache
    norm_path = ROOT / args.normalizer_checkpoint
    for path in (train_path, val_path, norm_path):
        refuse_test(path)
    train_cache = load_cache(train_path, "router_train_20_60")
    val_cache = load_cache(val_path, "router_val_60_80")
    std = load_std(norm_path, int(val_cache["num_features"]))
    device = torch.device(args.device)
    selected_idx = [int(i) for i in frozen["selected_expert_indices"]]
    selected_names = list(frozen["selected_three_experts"])
    target = val_cache["targets"].to(torch.float32)
    mask = val_cache["target_masks"].to(torch.bool)
    starts = val_cache["absolute_window_starts"].to(torch.long)
    if int(starts.min()) != 8640 or int(starts.max()) != 11412 or int(val_cache["num_windows"]) != 2773:
        raise ValueError("Validation windows do not match expected ETTh1 router-val split")
    if int(starts.max()) + int(val_cache["forecast_horizon"]) > 11520:
        raise ValueError("Validation target window crosses split boundary")

    rows = []
    per_window: dict[str, torch.Tensor] = {}
    for name in list(val_cache["expert_names"]):
        pred = optional_prediction(val_cache, name)
        m = metrics(val_cache, std, pred)
        rows.append({"method": f"single_{name}", "display": name, "mae": m["mae"], "mse": m["mse"]})
        per_window[f"single_{name}"] = m["per_window_mae"]
    best_single = min([r for r in rows if r["method"].startswith("single_")], key=lambda r: float(r["mae"]))

    selected_fixed_pred = selected_forecasts(val_cache, selected_idx).mean(dim=-1)
    old_idx = expert_indices(val_cache, OLD_FIXED3)
    old_fixed_pred = selected_forecasts(val_cache, old_idx).mean(dim=-1)
    for method, display, pred in (
        ("train_selected_fixed3_equal", "+".join(selected_names), selected_fixed_pred),
        ("old_validation_selected_fixed3_reference", "+".join(OLD_FIXED3), old_fixed_pred),
    ):
        m = metrics(val_cache, std, pred)
        rows.append({"method": method, "display": display, "mae": m["mae"], "mse": m["mse"]})
        per_window[method] = m["per_window_mae"]

    validation_rows = []
    expanded_preds = []
    base_preds = []
    for seed in SEEDS:
        pred, extra = evaluate_expanded(val_cache, train_cache, std, selected_idx, seed, device)
        base, _ = parameterized_current_base_prediction(val_cache, train_cache, std, selected_idx, seed, device)
        pm = metrics(val_cache, std, pred)
        bm = metrics(val_cache, std, base)
        sfm = metrics(val_cache, std, selected_fixed_pred)
        validation_rows.append(
            {
                "method": "train_selected_current_best_model",
                "seed": seed,
                "mae": pm["mae"],
                "mse": pm["mse"],
                "fixed3_mae": sfm["mae"],
                "diff_vs_selected_fixed3": pm["mae"] - sfm["mae"],
                "diff_vs_0.363642": pm["mae"] - BASELINE_MAE,
                "diff_vs_previous_0.363112": pm["mae"] - PREVIOUS_BEST_MAE,
                "base_hv_mae": bm["mae"],
                **extra,
            }
        )
        expanded_preds.append(pm["per_window_mae"])
        base_preds.append(bm["per_window_mae"])
    maes = torch.tensor([float(r["mae"]) for r in validation_rows])
    mses = torch.tensor([float(r["mse"]) for r in validation_rows])
    expanded_all = torch.cat(expanded_preds)
    selected_fixed_all = torch.cat([metrics(val_cache, std, selected_fixed_pred)["per_window_mae"] for _ in SEEDS])
    boot_fixed = paired_bootstrap(expanded_all, selected_fixed_all, seed=20260812, samples=5000)
    current_best_summary = {
        "method": "train_selected_current_best_model",
        "display": "current-best architecture using train-selected core",
        "mae": float(maes.mean()),
        "mae_std": float(maes.std(unbiased=False)),
        "mse": float(mses.mean()),
        "mse_std": float(mses.std(unbiased=False)),
        "diff_vs_selected_fixed3": float(maes.mean()) - float(metrics(val_cache, std, selected_fixed_pred)["mae"]),
        "diff_vs_0.363642": float(maes.mean()) - BASELINE_MAE,
        "diff_vs_previous_0.363112": float(maes.mean()) - PREVIOUS_BEST_MAE,
        **{f"bootstrap_vs_selected_fixed3_{k}": v for k, v in boot_fixed.items()},
    }
    rows.append(current_best_summary)
    rows.append({"method": "previous_current_best_reference", "display": "previous expanded_both result [reference]", "mae": PREVIOUS_BEST_MAE, "mse": 0.30605703592300415})
    rows.append({"method": "old_hv_baseline_reference", "display": "old HV baseline [reference]", "mae": BASELINE_MAE, "mse": 0.3067120909690857})
    rows.append({"method": "ridge_reference", "display": "ridge residual result [reference]", "mae": 0.363301, "mse": 0.306286})
    write_csv(out_dir / "validation_comparison.csv", rows)
    write_csv(out_dir / "validation_current_best_per_seed.csv", validation_rows)

    required_table = [
        {"Method": "Best single", "Val MAE": best_single["mae"], "Val MSE": best_single["mse"], "Detail": best_single["display"]},
        {"Method": "Train-selected fixed 3", "Val MAE": metrics(val_cache, std, selected_fixed_pred)["mae"], "Val MSE": metrics(val_cache, std, selected_fixed_pred)["mse"], "Detail": "+".join(selected_names)},
        {"Method": "Old validation-selected fixed 3 [reference]", "Val MAE": metrics(val_cache, std, old_fixed_pred)["mae"], "Val MSE": metrics(val_cache, std, old_fixed_pred)["mse"], "Detail": "+".join(OLD_FIXED3)},
        {"Method": "Train-selected current-best model", "Val MAE": current_best_summary["mae"], "Val MSE": current_best_summary["mse"], "Detail": SPECIALIST_CONFIG.name},
        {"Method": "Previous current-best model [reference]", "Val MAE": PREVIOUS_BEST_MAE, "Val MSE": 0.30605703592300415, "Detail": "validation-optimized development result"},
    ]
    write_csv(out_dir / "required_final_table.csv", required_table)
    report = {
        "phase_a_frozen_config": frozen,
        "phase_b_loaded_validation_after_freeze": True,
        "required_final_table": required_table,
        "best_single": best_single,
        "train_selected_fixed3": required_table[1],
        "old_validation_selected_fixed3_reference": required_table[2],
        "train_selected_current_best_model": current_best_summary,
        "main_answers": {
            "three_experts_selected_from_router_train": selected_names,
            "router_train_oof_mae_of_selected_triple": frozen["router_train_oof_mae"],
            "selected_fixed3_router_val_mae": required_table[1]["Val MAE"],
            "full_current_best_model_router_val_mae": current_best_summary["mae"],
            "difference_versus_selected_fixed3": current_best_summary["diff_vs_selected_fixed3"],
            "difference_versus_0.363642": current_best_summary["diff_vs_0.363642"],
            "difference_versus_previous_0.363112": current_best_summary["diff_vs_previous_0.363112"],
            "did_router_train_select_old_patchtst_itransformer_timesnet": tuple(selected_names) == OLD_FIXED3,
            "did_improvement_survive_clean_expert_selection": bool(current_best_summary["mae"] < required_table[1]["Val MAE"] and current_best_summary["mae"] < BASELINE_MAE),
        },
        "leakage_audit": {
            "fixed_three_selected_exclusively_from_router_train": True,
            "router_val_not_loaded_before_configuration_freeze": True,
            "test_cache_loaded": False,
            "no_configuration_changed_after_router_val": True,
            "chronological_updates_causal": True,
            "causal_rule": "old_start + horizon <= current_start",
            "same_validation_windows_for_every_comparison": True,
            "same_mae_mse_implementation": "sample_mae/sample_mse",
            "validation_windows": int(val_cache["num_windows"]),
            "validation_start_min": int(starts.min()),
            "validation_start_max": int(starts.max()),
        },
        "reproduce_commands": {
            "phase_a": f"python experiments\\train_selected_core_etth1\\run_train_selected_core_eval.py --phase select --device {args.device}",
            "phase_b": f"python experiments\\train_selected_core_etth1\\run_train_selected_core_eval.py --phase evaluate --device {args.device}",
            "all": f"python experiments\\train_selected_core_etth1\\run_train_selected_core_eval.py --phase all --device {args.device}",
        },
        "runtime_sec": time.time() - float(args.start_time),
    }
    write_json(out_dir / "final_report.json", report)
    lines = [
        "# Train-Selected Core ETTh1 Re-Evaluation",
        "",
        "## Result",
        "",
        f"- Router-train selected experts: `{'+'.join(selected_names)}`.",
        f"- Router-train OOF MAE/MSE: `{frozen['router_train_oof_mae']:.6f}` / `{frozen['router_train_oof_mse']:.6f}`.",
        f"- Train-selected fixed-3 validation MAE: `{required_table[1]['Val MAE']:.6f}`.",
        f"- Train-selected current-best architecture validation MAE: `{current_best_summary['mae']:.6f}`.",
        f"- Difference vs previous `0.363112`: `{current_best_summary['diff_vs_previous_0.363112']:.6f}`.",
        "",
        "## Required Table",
        "",
        "| Method | Val MAE | Val MSE | Detail |",
        "|---|---:|---:|---|",
    ]
    for row in required_table:
        lines.append(f"| {row['Method']} | `{row['Val MAE']:.6f}` | `{row['Val MSE']:.6f}` | `{row['Detail']}` |")
    lines.extend(["", "## Reproduce", "", "```powershell", report["reproduce_commands"]["all"], "```"])
    (out_dir / "experiment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("select", "evaluate", "all"), default="all")
    parser.add_argument("--train-cache", default="cache/costarts_walkforward/router_train_20_60_cache.pt")
    parser.add_argument("--val-cache", default="cache/costarts_walkforward/router_val_60_80_cache.pt")
    parser.add_argument("--normalizer-checkpoint", default="checkpoints/costarts_walkforward/final_60/DLinear/best_expert.pt")
    parser.add_argument("--out-dir", default="experiments/train_selected_core_etth1")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.start_time = time.time()
    if args.phase in {"select", "all"}:
        phase_a(args)
    if args.phase in {"evaluate", "all"}:
        phase_b(args)
    if args.phase == "select":
        print(json.dumps(json.loads((ROOT / args.out_dir / "frozen_config_before_validation.json").read_text(encoding="utf-8")), indent=2))
    else:
        print(json.dumps(json.loads((ROOT / args.out_dir / "final_report.json").read_text(encoding="utf-8")), indent=2))


if __name__ == "__main__":
    main()
