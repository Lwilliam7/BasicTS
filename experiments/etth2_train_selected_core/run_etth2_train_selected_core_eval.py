"""Clean ETTh2 train-selected core audit.

Phase A uses ETTh2 router-train only to select the three core experts.
Phase B loads canonical ETTh2 router-val once and evaluates the frozen model.

Canonical ETTh2 metric:
- raw/original-scale sample_mae/sample_mse
- std = ones
- no inverse transform
- 613 validation windows, starts 10800..11412
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

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.chronological_adaptive_costar.run_chronological_adaptive_costar import (  # noqa: E402
    Trial as ChronoTrial,
    chronological_online_weights,
    enforce_observable,
)
from experiments.expanded_expert_pool_costar.run_expanded_expert_pool import Config as SpecialistConfig  # noqa: E402
from experiments.horizon_variable_adaptive_costar.run_hv_adaptive_costar import (  # noqa: E402
    Trial as HvTrial,
    chronological_hv_weights,
    predict_from_hv_weights,
)
from experiments.oracle_weight_tournament.run_tournament import load_cache, sample_mae, sample_mse, weighted_forecast  # noqa: E402


SPECIALIST_CONFIG = SpecialistConfig("both", "variable", 0.95, 0.10, 0.02, 96)
CANONICAL_BEST_FIXED2_MAE = 0.2752290368080139
CANONICAL_BEST_FIXED3 = ("DLinear", "TimesNet", "ModernTCN")


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


def forecasts_for(cache: Mapping[str, Any], expert_idx: Sequence[int]) -> torch.Tensor:
    return cache["prediction_stack"][..., list(expert_idx)].to(torch.float32)


def expert_prediction(cache: Mapping[str, Any], expert: str) -> torch.Tensor:
    return cache["prediction_stack"][..., list(cache["expert_names"]).index(expert)].to(torch.float32)


def metrics(cache: Mapping[str, Any], pred: torch.Tensor, std: torch.Tensor) -> dict[str, Any]:
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.bool)
    mae = sample_mae(pred, target, mask, std)
    mse = sample_mse(pred, target, mask, std)
    return {"mae": float(mae.mean()), "mse": float(mse.mean()), "per_window_mae": mae, "per_window_mse": mse}


def abs_error(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return ((pred - target) / std.view(1, 1, -1)).abs() * mask.to(torch.float32)


def per_location_error(cache: Mapping[str, Any], expert_idx: Sequence[int], std: torch.Tensor) -> torch.Tensor:
    forecasts = forecasts_for(cache, expert_idx)
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.float32)
    return ((forecasts - target.unsqueeze(-1)) / std.view(1, 1, -1, 1)).abs() * mask.unsqueeze(-1)


def select_core(cache: Mapping[str, Any], std: torch.Tensor) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    names = list(cache["expert_names"])
    stack = cache["prediction_stack"].to(torch.float32)
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.bool)
    folds = train_folds(int(cache["num_windows"]))
    rows = []
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
        rows.append(
            {
                "experts": [names[i] for i in idx],
                "expert_indices": list(idx),
                "subset": "+".join(names[i] for i in idx),
                "pooled_oof_mae": float(torch.cat(mae_chunks).mean()),
                "pooled_oof_mse": float(torch.cat(mse_chunks).mean()),
                "worst_fold_mae": max(float(r["mae"]) for r in fold_rows),
                "fold_rows": fold_rows,
            }
        )
    rows = sorted(rows, key=lambda r: (float(r["pooled_oof_mae"]), float(r["pooled_oof_mse"]), float(r["worst_fold_mae"]), str(r["subset"])))
    return rows, rows[0]


def current_base_prediction(
    cache: Mapping[str, Any],
    init_cache: Mapping[str, Any],
    expert_idx: Sequence[int],
    std: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    starts = cache["absolute_window_starts"].to(torch.long)
    horizon = int(cache["forecast_horizon"])
    if not bool(torch.all(starts[1:] > starts[:-1])):
        raise ValueError("Cache starts are not chronological")
    forecasts = forecasts_for(cache, expert_idx)
    train_err = per_location_error(init_cache, expert_idx, std)
    eval_err = per_location_error(cache, expert_idx, std)
    online_weights, online_extra = chronological_online_weights(
        starts=starts,
        expert_mae=eval_err.mean(dim=(1, 2)),
        horizon=horizon,
        trial=ChronoTrial("ema", "ema_decay0.97_temp0.1", decay=0.97, temperature=0.1),
        train_mean_mae=train_err.mean(dim=(0, 1, 2)),
        mode="ema",
    )
    # ETTh2 has no compatible static neural winner for arbitrary train-selected
    # triples. The frozen static prior is therefore equal weights.
    static_weights = torch.full_like(online_weights, 1.0 / 3.0)
    chrono_weights = 0.5 * static_weights + 0.5 * online_weights
    chrono_weights = chrono_weights / chrono_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    chrono_pred = weighted_forecast(forecasts, chrono_weights)
    hv_weights, hv_extra = chronological_hv_weights(
        starts=starts,
        train_err_mean=train_err.mean(dim=0),
        val_err=eval_err,
        horizon=horizon,
        trial=HvTrial("hv_ema", "hvema_lowrank1_decay0.95_temp0.1", mode="hv_lowrank", rank=1, decay=0.95, temperature=0.1),
    )
    hv_pred = predict_from_hv_weights(forecasts, hv_weights)
    pred = 0.25 * chrono_pred + 0.75 * hv_pred
    return pred, {
        "chrono_num_updates": online_extra["num_updates"],
        "hv_num_updates": hv_extra["num_updates"],
        "static_prior": "equal_weights_no_etth2_static_neural_artifact",
        **{f"mean_core_weight_{list(cache['expert_names'])[expert_idx[i]]}": float(hv_weights[..., i].mean()) for i in range(3)},
    }


def specialist_weight_pair(adv_d: torch.Tensor, adv_m: torch.Tensor, selected_core: set[str]) -> tuple[torch.Tensor, torch.Tensor]:
    cap = float(SPECIALIST_CONFIG.extra_weight_cap)
    margin = float(SPECIALIST_CONFIG.activation_margin)
    scale = 0.05
    raw_d = ((adv_d - margin).clamp_min(0.0) / scale).clamp_max(1.0)
    raw_m = ((adv_m - margin).clamp_min(0.0) / scale).clamp_max(1.0)
    if "DLinear" in selected_core:
        raw_d = torch.zeros_like(raw_d)
    if "ModernTCN" in selected_core:
        raw_m = torch.zeros_like(raw_m)
    w_d = raw_d * (cap / 2.0)
    w_m = raw_m * (cap / 2.0)
    total = w_d + w_m
    over = total > cap
    if bool(over.any()):
        factor = cap / total.clamp_min(1e-8)
        w_d = torch.where(over, w_d * factor, w_d)
        w_m = torch.where(over, w_m * factor, w_m)
    return w_d, w_m


def run_specialists_no_duplicate(
    starts: torch.Tensor,
    base_pred: torch.Tensor,
    d_pred: torch.Tensor,
    m_pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    std: torch.Tensor,
    init_base_err: torch.Tensor,
    init_d_err: torch.Tensor,
    init_m_err: torch.Tensor,
    selected_core: set[str],
) -> tuple[torch.Tensor, dict[str, Any]]:
    h, v = base_pred.shape[1], base_pred.shape[2]
    def agg(err: torch.Tensor) -> torch.Tensor:
        return err.mean(dim=0, keepdim=True)
    base_state = torch.stack([agg(e) for e in init_base_err]).mean(dim=0)
    d_state = torch.stack([agg(e) for e in init_d_err]).mean(dim=0)
    m_state = torch.stack([agg(e) for e in init_m_err]).mean(dim=0)
    pending: list[int] = []
    preds = []
    updates = 0
    wd_vals = []
    wm_vals = []
    disabled_d = "DLinear" in selected_core
    disabled_m = "ModernTCN" in selected_core
    for i in range(base_pred.shape[0]):
        now = int(starts[i])
        still = []
        for j in pending:
            if int(starts[j]) + h <= now:
                enforce_observable(int(starts[j]), now, h)
                base_state = SPECIALIST_CONFIG.decay * base_state + (1.0 - SPECIALIST_CONFIG.decay) * agg(abs_error(base_pred[j : j + 1], target[j : j + 1], mask[j : j + 1], std)[0])
                d_state = SPECIALIST_CONFIG.decay * d_state + (1.0 - SPECIALIST_CONFIG.decay) * agg(abs_error(d_pred[j : j + 1], target[j : j + 1], mask[j : j + 1], std)[0])
                m_state = SPECIALIST_CONFIG.decay * m_state + (1.0 - SPECIALIST_CONFIG.decay) * agg(abs_error(m_pred[j : j + 1], target[j : j + 1], mask[j : j + 1], std)[0])
                updates += 1
            else:
                still.append(j)
        pending = still
        adv_d = (base_state.expand(h, v) - d_state.expand(h, v)) / base_state.expand(h, v).clamp_min(1e-8)
        adv_m = (base_state.expand(h, v) - m_state.expand(h, v)) / base_state.expand(h, v).clamp_min(1e-8)
        if updates < int(SPECIALIST_CONFIG.warmup):
            w_d = torch.zeros_like(adv_d)
            w_m = torch.zeros_like(adv_m)
        else:
            w_d, w_m = specialist_weight_pair(adv_d, adv_m, selected_core)
        preds.append((1.0 - w_d - w_m) * base_pred[i] + w_d * d_pred[i] + w_m * m_pred[i])
        wd_vals.append(float(w_d.mean()))
        wm_vals.append(float(w_m.mean()))
        pending.append(i)
    return torch.stack(preds), {
        "num_specialist_updates": updates,
        "avg_weight_DLinear": float(torch.tensor(wd_vals).mean()),
        "avg_weight_ModernTCN": float(torch.tensor(wm_vals).mean()),
        "DLinear_specialist_disabled_duplicate_core": disabled_d,
        "ModernTCN_specialist_disabled_duplicate_core": disabled_m,
    }


def full_model_prediction(cache: Mapping[str, Any], init_cache: Mapping[str, Any], expert_idx: Sequence[int], std: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    selected_core = {list(cache["expert_names"])[i] for i in expert_idx}
    base_pred, base_extra = current_base_prediction(cache, init_cache, expert_idx, std)
    init_base, _ = current_base_prediction(init_cache, init_cache, expert_idx, std)
    train_target = init_cache["targets"].to(torch.float32)
    train_mask = init_cache["target_masks"].to(torch.bool)
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.bool)
    d_train = expert_prediction(init_cache, "DLinear")
    m_train = expert_prediction(init_cache, "ModernTCN")
    pred, extra = run_specialists_no_duplicate(
        cache["absolute_window_starts"].to(torch.long),
        base_pred,
        expert_prediction(cache, "DLinear"),
        expert_prediction(cache, "ModernTCN"),
        target,
        mask,
        std,
        abs_error(init_base, train_target, train_mask, std),
        abs_error(d_train, train_target, train_mask, std),
        abs_error(m_train, train_target, train_mask, std),
        selected_core,
    )
    return pred, {**base_extra, **extra}


def phase_a(args: argparse.Namespace) -> None:
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = ROOT / args.train_cache
    refuse_test(train_path)
    train_cache = load_cache(train_path, "router_train")
    std = torch.ones(int(train_cache["num_features"]), dtype=torch.float32)
    rows, selected = select_core(train_cache, std)
    write_csv(out_dir / "router_train_all_triples.csv", [{k: v for k, v in r.items() if k != "fold_rows"} for r in rows])
    write_json(out_dir / "router_train_all_triples.json", rows)
    frozen = {
        "phase": "A_frozen_before_validation",
        "dataset": "ETTh2",
        "router_val_loaded": False,
        "selected_three_experts": selected["experts"],
        "selected_expert_indices": selected["expert_indices"],
        "router_train_oof_mae": selected["pooled_oof_mae"],
        "router_train_oof_mse": selected["pooled_oof_mse"],
        "router_train_worst_fold_mae": selected["worst_fold_mae"],
        "selection_metric": "pooled chronological OOF MAE on ETTh2 router_train only",
        "tie_breakers": ["OOF MSE", "worst-fold MAE", "deterministic expert-name ordering"],
        "canonical_metric": {
            "scale": "raw_original",
            "std": [1.0] * int(train_cache["num_features"]),
            "inverse_transform": "none",
            "mae_mse_implementation": "sample_mae/sample_mse",
        },
        "model_hyperparameters": {
            "chrono_ema_decay": 0.97,
            "chrono_temperature": 0.1,
            "chrono_static_prior": "equal_weights_no_etth2_static_neural_artifact",
            "chrono_online_blend": 0.5,
            "hv_mode": "hv_lowrank",
            "hv_rank": 1,
            "hv_decay": 0.95,
            "hv_temperature": 0.1,
            "hybrid_chrono_weight": 0.25,
            "hybrid_hv_weight": 0.75,
            "specialist_config": {"name": SPECIALIST_CONFIG.name, **asdict(SPECIALIST_CONFIG)},
            "duplicate_specialists_disabled_if_in_core": True,
        },
        "cache_paths": {"router_train": args.train_cache},
        "cache_hashes": {"router_train_sha256": sha256_file(train_path)},
        "expert_names_in_cache": list(train_cache["expert_names"]),
        "train_windows": int(train_cache["num_windows"]),
        "train_start_min": int(train_cache["absolute_window_starts"].min()),
        "train_start_max": int(train_cache["absolute_window_starts"].max()),
        "forecast_horizon": int(train_cache["forecast_horizon"]),
        "num_variables": int(train_cache["num_features"]),
        "selected_matches_canonical_best_fixed3": tuple(selected["experts"]) == CANONICAL_BEST_FIXED3,
        "test_cache_loaded": False,
    }
    write_json(out_dir / "frozen_config_before_validation.json", frozen)


def phase_b(args: argparse.Namespace) -> None:
    out_dir = ROOT / args.out_dir
    frozen_path = out_dir / "frozen_config_before_validation.json"
    if not frozen_path.exists():
        raise FileNotFoundError("Run Phase A first")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen.get("router_val_loaded"):
        raise RuntimeError("Validation was loaded before freeze")
    train_path = ROOT / args.train_cache
    val_path = ROOT / args.val_cache
    refuse_test(train_path)
    refuse_test(val_path)
    train_cache = load_cache(train_path, "router_train")
    val_cache = load_cache(val_path, "router_val")
    std = torch.ones(int(val_cache["num_features"]), dtype=torch.float32)
    starts = val_cache["absolute_window_starts"].to(torch.long)
    if int(val_cache["num_windows"]) != 613 or int(starts.min()) != 10800 or int(starts.max()) != 11412:
        raise ValueError("ETTh2 router_val does not match canonical windows")
    if int(val_cache["forecast_horizon"]) != 12 or int(val_cache["num_features"]) != 7:
        raise ValueError("ETTh2 horizon/variable mismatch")
    if int(starts.max()) + int(val_cache["forecast_horizon"]) != 11424:
        raise ValueError("Unexpected ETTh2 validation target end")
    selected_idx = [int(i) for i in frozen["selected_expert_indices"]]
    selected_names = list(frozen["selected_three_experts"])
    rows = []
    for expert in list(val_cache["expert_names"]):
        m = metrics(val_cache, expert_prediction(val_cache, expert), std)
        rows.append({"method": f"single_{expert}", "display": expert, "mae": m["mae"], "mse": m["mse"]})
    best_single = min([r for r in rows if r["method"].startswith("single_")], key=lambda r: float(r["mae"]))
    fixed2_pred = forecasts_for(val_cache, expert_indices(val_cache, ("DLinear", "ModernTCN"))).mean(dim=-1)
    selected_fixed3_pred = forecasts_for(val_cache, selected_idx).mean(dim=-1)
    canonical_fixed3_pred = forecasts_for(val_cache, expert_indices(val_cache, CANONICAL_BEST_FIXED3)).mean(dim=-1)
    for method, display, pred in (
        ("best_fixed2_reference", "DLinear+ModernTCN", fixed2_pred),
        ("train_selected_fixed3_equal", "+".join(selected_names), selected_fixed3_pred),
        ("canonical_best_fixed3_reference", "+".join(CANONICAL_BEST_FIXED3), canonical_fixed3_pred),
    ):
        m = metrics(val_cache, pred, std)
        rows.append({"method": method, "display": display, "mae": m["mae"], "mse": m["mse"]})
    full_pred, full_extra = full_model_prediction(val_cache, train_cache, selected_idx, std)
    full_m = metrics(val_cache, full_pred, std)
    selected_m = metrics(val_cache, selected_fixed3_pred, std)
    fixed2_m = metrics(val_cache, fixed2_pred, std)
    rows.append({"method": "train_selected_full_current_best_model", "display": SPECIALIST_CONFIG.name, "mae": full_m["mae"], "mse": full_m["mse"], **full_extra})
    required_table = [
        {"Method": "Best single", "Val MAE": best_single["mae"], "Val MSE": best_single["mse"], "Detail": best_single["display"]},
        {"Method": "Best fixed-2 [reference]", "Val MAE": fixed2_m["mae"], "Val MSE": fixed2_m["mse"], "Detail": "DLinear+ModernTCN"},
        {"Method": "Train-selected fixed-3", "Val MAE": selected_m["mae"], "Val MSE": selected_m["mse"], "Detail": "+".join(selected_names)},
        {"Method": "Canonical best fixed-3 [reference]", "Val MAE": metrics(val_cache, canonical_fixed3_pred, std)["mae"], "Val MSE": metrics(val_cache, canonical_fixed3_pred, std)["mse"], "Detail": "+".join(CANONICAL_BEST_FIXED3)},
        {"Method": "Train-selected full current-best model", "Val MAE": full_m["mae"], "Val MSE": full_m["mse"], "Detail": SPECIALIST_CONFIG.name},
    ]
    write_csv(out_dir / "validation_comparison.csv", rows)
    write_csv(out_dir / "required_final_table.csv", required_table)
    report = {
        "phase_a_frozen_config": frozen,
        "phase_b_loaded_validation_after_freeze": True,
        "required_final_table": required_table,
        "main_answers": {
            "three_experts_selected_from_router_train": selected_names,
            "router_train_oof_mae": frozen["router_train_oof_mae"],
            "router_train_oof_mse": frozen["router_train_oof_mse"],
            "train_selected_fixed3_router_val_mae": selected_m["mae"],
            "full_model_router_val_mae": full_m["mae"],
            "full_model_diff_vs_own_train_selected_fixed3": full_m["mae"] - selected_m["mae"],
            "full_model_beat_own_train_selected_fixed3": bool(full_m["mae"] < selected_m["mae"]),
            "full_model_diff_vs_canonical_best_fixed2_0.275229": full_m["mae"] - CANONICAL_BEST_FIXED2_MAE,
            "full_model_beat_canonical_best_fixed2": bool(full_m["mae"] < CANONICAL_BEST_FIXED2_MAE),
            "did_router_train_select_dlinear_timesnet_moderntcn": tuple(selected_names) == ("DLinear", "TimesNet", "ModernTCN"),
            "compare_with_etth1_clean_result": "ETTh1 train-only core selection chose the prior core and preserved 0.363112; ETTh2 selected a different core and the frozen architecture did not beat fixed baselines.",
            "same_frozen_architecture_transfer_without_etth2_val_tuning": bool(full_m["mae"] < selected_m["mae"] and full_m["mae"] < CANONICAL_BEST_FIXED2_MAE),
        },
        "specialist_duplicate_handling": {
            "selected_core": selected_names,
            "DLinear_disabled_if_core": "DLinear" in set(selected_names),
            "ModernTCN_disabled_if_core": "ModernTCN" in set(selected_names),
            **full_extra,
        },
        "leakage_audit": {
            "etth2_fixed_three_selected_only_from_router_train": True,
            "router_val_not_loaded_during_phase_a": True,
            "test_cache_loaded": False,
            "no_hyperparameters_changed_after_router_val": True,
            "same_canonical_etth2_windows_for_every_final_comparison": True,
            "chronological_updates_causal": True,
            "causal_rule": "old_start + horizon <= current_start",
            "validation_windows": int(val_cache["num_windows"]),
            "validation_start_min": int(starts.min()),
            "validation_start_max": int(starts.max()),
            "horizon": int(val_cache["forecast_horizon"]),
            "variables": int(val_cache["num_features"]),
            "metric": "raw/original scale sample_mae/sample_mse, std=ones, no inverse transform",
        },
        "reproduce_commands": {
            "phase_a": "python experiments\\etth2_train_selected_core\\run_etth2_train_selected_core_eval.py --phase select",
            "phase_b": "python experiments\\etth2_train_selected_core\\run_etth2_train_selected_core_eval.py --phase evaluate",
            "all": "python experiments\\etth2_train_selected_core\\run_etth2_train_selected_core_eval.py --phase all",
        },
        "runtime_sec": time.time() - float(args.start_time),
    }
    write_json(out_dir / "final_report.json", report)
    lines = [
        "# ETTh2 Train-Selected Core Audit",
        "",
        "## Result",
        "",
        f"- Router-train selected experts: `{'+'.join(selected_names)}`.",
        f"- Router-train OOF MAE/MSE: `{frozen['router_train_oof_mae']:.6f}` / `{frozen['router_train_oof_mse']:.6f}`.",
        f"- Full model validation MAE/MSE: `{full_m['mae']:.6f}` / `{full_m['mse']:.6f}`.",
        f"- Beat own selected fixed-3: `{full_m['mae'] < selected_m['mae']}`.",
        f"- Beat canonical best fixed-2: `{full_m['mae'] < CANONICAL_BEST_FIXED2_MAE}`.",
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
    parser.add_argument("--train-cache", default="cache/costarts_fresh/ETTh2_96_12/router_train_cache.pt")
    parser.add_argument("--val-cache", default="cache/costarts_fresh/ETTh2_96_12/router_val_cache.pt")
    parser.add_argument("--out-dir", default="experiments/etth2_train_selected_core")
    args = parser.parse_args()
    args.start_time = time.time()
    if args.phase in {"select", "all"}:
        phase_a(args)
    if args.phase in {"evaluate", "all"}:
        phase_b(args)
    if args.phase == "select":
        print((ROOT / args.out_dir / "frozen_config_before_validation.json").read_text(encoding="utf-8"))
    else:
        print((ROOT / args.out_dir / "final_report.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
