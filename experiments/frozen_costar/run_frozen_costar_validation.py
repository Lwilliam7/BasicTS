"""Validation-only frozen COSTAR comparison.

This experiment isolates the effect of validation-time sequential adaptation.
It keeps the selected core, frozen expert forecasts, static prior, and
hyperparameters from the existing COSTAR path, but freezes all validation
weights at their router-train initialized values.

No test cache is loaded by this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.etth2_train_selected_core.run_etth2_train_selected_core_eval as etth2  # noqa: E402
import experiments.train_selected_core_etth1.run_train_selected_core_eval as etth1  # noqa: E402
from experiments.chronological_adaptive_costar.run_chronological_adaptive_costar import (  # noqa: E402
    CURRENT_WINNER,
    SEEDS,
    softmax_neg,
)
from experiments.expanded_expert_pool_costar.run_expanded_expert_pool import (  # noqa: E402
    expand_group,
    init_state_from_errors,
    weights_from_advantage,
)
from experiments.horizon_variable_adaptive_costar.run_hv_adaptive_costar import (  # noqa: E402
    Trial as HvTrial,
    errors_to_weights,
    predict_from_hv_weights,
)
from experiments.oracle_weight_tournament.run_tournament import (  # noqa: E402
    WeightStudent,
    args_global_weights,
    load_cache,
    load_std,
    sample_mae,
    sample_mse,
    weighted_forecast,
)


OUT_DIR = ROOT / "experiments/frozen_costar"
ETTH1_FROZEN = ROOT / "experiments/train_selected_core_etth1/frozen_config_before_validation.json"
ETTH2_FROZEN = ROOT / "experiments/etth2_train_selected_core/frozen_config_before_validation.json"
ETTH1_EXPECTED_ONLINE_MAE = 0.3631121516227722
ETTH2_EXPECTED_ONLINE_MAE = 0.27683165669441223
HV_TRIAL = HvTrial(
    "hv_ema",
    "hvema_lowrank1_decay0.95_temp0.1",
    mode="hv_lowrank",
    rank=1,
    decay=0.95,
    temperature=0.1,
)


def refuse_test(path: str | Path) -> None:
    if "test" in str(path).lower():
        raise ValueError(f"Refusing test path: {path}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_frozen_core(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    names = data.get("selected_three_experts")
    if not isinstance(names, list) or len(names) != 3:
        raise ValueError(f"Missing selected_three_experts in {path}")
    return [str(x) for x in names]


def metric_values(cache: Mapping[str, Any], pred: torch.Tensor, std: torch.Tensor) -> dict[str, float]:
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.bool)
    return {
        "mae": float(sample_mae(pred, target, mask, std).mean()),
        "mse": float(sample_mse(pred, target, mask, std).mean()),
    }


def fixed_core_prediction(cache: Mapping[str, Any], expert_idx: Sequence[int]) -> torch.Tensor:
    return cache["prediction_stack"][..., list(expert_idx)].to(torch.float32).mean(dim=-1)


def tensor_digest(cache: Mapping[str, Any]) -> dict[str, str]:
    keys = ["histories", "prediction_stack", "absolute_window_starts"]
    out: dict[str, str] = {}
    for key in keys:
        t = cache[key].detach().cpu().contiguous()
        out[key] = hashlib.sha256(t.numpy().tobytes()).hexdigest()
    return out


def cloned_with_random_targets(cache: Mapping[str, Any], randomize_masks: bool = False) -> dict[str, Any]:
    cloned = dict(cache)
    gen = torch.Generator().manual_seed(12345 if not randomize_masks else 54321)
    cloned["targets"] = torch.randn(cache["targets"].shape, generator=gen, dtype=torch.float32)
    if randomize_masks:
        cloned["target_masks"] = torch.rand(cache["target_masks"].shape, generator=gen) > 0.37
    return cloned


def frozen_hv_weights(num_windows: int, train_err_mean: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    one = errors_to_weights(train_err_mean, HV_TRIAL)
    return one.unsqueeze(0).expand(num_windows, -1, -1, -1).clone(), {
        "hv_num_updates": 0,
        "hv_weight_source": "router_train_initialized_repeated",
    }


def load_static_winner_weights_no_targets(
    seed: int,
    cache: Mapping[str, Any],
    expert_idx: Sequence[int],
    device: torch.device,
) -> torch.Tensor:
    ckpt_path = ROOT / "experiments/oracle_weight_tournament/checkpoints" / f"{CURRENT_WINNER}_seed{seed}" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    trial = ckpt["trial"]
    model = WeightStudent(
        args_global_weights(),
        int(cache["input_len"]),
        int(cache["forecast_horizon"]),
        int(cache["num_features"]),
        mode="prototype_residual",
        num_prototypes=int(trial["num_prototypes"]),
        residual_scale=float(trial["residual_scale"]),
        feature_mix=trial.get("feature_mix", "full"),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    histories = cache["histories"].to(torch.float32)
    forecasts = cache["prediction_stack"][..., list(expert_idx)].to(torch.float32)
    weights: list[torch.Tensor] = []
    with torch.no_grad():
        for lo in range(0, int(histories.shape[0]), 1024):
            hi = min(lo + 1024, int(histories.shape[0]))
            out = model(
                histories[lo:hi].to(device),
                forecasts[lo:hi].to(device),
                prototypes=ckpt["prototypes"],
            )
            weights.append(out["weights"].detach().cpu())
    return torch.cat(weights, dim=0)


def frozen_costar_base_prediction(
    dataset: str,
    cache: Mapping[str, Any],
    train_cache_for_init: Mapping[str, Any],
    std: torch.Tensor,
    expert_idx: Sequence[int],
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Frozen equivalent of the current base COSTAR prediction path."""
    n = int(cache["num_windows"])
    forecasts = cache["prediction_stack"][..., list(expert_idx)].to(torch.float32)
    if dataset == "ETTh1":
        train_err = etth1.per_location_abs_error_for_indices(train_cache_for_init, std, expert_idx)
        train_expert_mae = train_err.mean(dim=(0, 1, 2))
        selected_names = [list(cache["expert_names"])[i] for i in expert_idx]
        if tuple(selected_names) == etth1.OLD_FIXED3:
            static_weights = load_static_winner_weights_no_targets(seed, cache, expert_idx, device)
            static_source = "existing_static_winner_no_target_metrics"
        else:
            static_weights = torch.full((n, 3), 1.0 / 3.0)
            static_source = "equal_fallback_no_static_artifact"
    elif dataset == "ETTh2":
        train_err = etth2.per_location_error(train_cache_for_init, expert_idx, std)
        train_expert_mae = train_err.mean(dim=(0, 1, 2))
        static_weights = torch.full((n, 3), 1.0 / 3.0)
        static_source = "equal_weights_no_etth2_static_neural_artifact"
    else:
        raise ValueError(dataset)

    online_init = softmax_neg(train_expert_mae, 0.1)
    online_weights = online_init.view(1, 3).expand(n, -1).clone()
    chrono_weights = 0.5 * static_weights + 0.5 * online_weights
    chrono_weights = chrono_weights / chrono_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    chrono_pred = weighted_forecast(forecasts, chrono_weights)

    hv_weights, hv_extra = frozen_hv_weights(n, train_err.mean(dim=0))
    hv_pred = predict_from_hv_weights(forecasts, hv_weights)
    pred = 0.25 * chrono_pred + 0.75 * hv_pred
    names = [list(cache["expert_names"])[i] for i in expert_idx]
    return pred, {
        "chrono_num_updates": 0,
        "hv_num_updates": hv_extra["hv_num_updates"],
        "static_weight_source": static_source,
        "general_weight_source": "router_train_initialized_repeated",
        "horizon_variable_weight_source": "router_train_initialized_repeated",
        **{f"mean_weight_{names[i]}": float(hv_weights[..., i].mean()) for i in range(3)},
    }


def frozen_etth1_specialists(
    base_pred: torch.Tensor,
    d_pred: torch.Tensor,
    m_pred: torch.Tensor,
    init_base_err: torch.Tensor,
    init_d_err: torch.Tensor,
    init_m_err: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    h, v = base_pred.shape[1], base_pred.shape[2]
    base_state = init_state_from_errors(init_base_err, etth1.SPECIALIST_CONFIG.structure)
    d_state = init_state_from_errors(init_d_err, etth1.SPECIALIST_CONFIG.structure)
    m_state = init_state_from_errors(init_m_err, etth1.SPECIALIST_CONFIG.structure)
    count = int(init_base_err.shape[0])
    base_s = expand_group(base_state, etth1.SPECIALIST_CONFIG.structure, h, v)
    d_s = expand_group(d_state, etth1.SPECIALIST_CONFIG.structure, h, v)
    m_s = expand_group(m_state, etth1.SPECIALIST_CONFIG.structure, h, v)
    adv_d = (base_s - d_s) / base_s.clamp_min(1e-8)
    adv_m = (base_s - m_s) / base_s.clamp_min(1e-8)
    if count < int(etth1.SPECIALIST_CONFIG.warmup):
        w_d = torch.zeros_like(adv_d)
        w_m = torch.zeros_like(adv_m)
    else:
        w_d, w_m = weights_from_advantage(adv_d, adv_m, etth1.SPECIALIST_CONFIG)
    pred = (1.0 - w_d - w_m).unsqueeze(0) * base_pred + w_d.unsqueeze(0) * d_pred + w_m.unsqueeze(0) * m_pred
    return pred, {
        "num_specialist_updates": 0,
        "specialist_weight_source": "router_train_initialized_repeated",
        "avg_weight_DLinear": float(w_d.mean()),
        "avg_weight_ModernTCN": float(w_m.mean()),
        "max_window_weight_DLinear": float(w_d.mean()),
        "max_window_weight_ModernTCN": float(w_m.mean()),
        "activation_rate_DLinear": float((w_d > 0).to(torch.float32).mean()),
        "activation_rate_ModernTCN": float((w_m > 0).to(torch.float32).mean()),
    }


def frozen_etth2_specialists(
    base_pred: torch.Tensor,
    d_pred: torch.Tensor,
    m_pred: torch.Tensor,
    init_base_err: torch.Tensor,
    init_d_err: torch.Tensor,
    init_m_err: torch.Tensor,
    selected_core: set[str],
) -> tuple[torch.Tensor, dict[str, Any]]:
    h, _ = base_pred.shape[1], base_pred.shape[2]

    def agg(err: torch.Tensor) -> torch.Tensor:
        return err.mean(dim=0, keepdim=True)

    base_state = torch.stack([agg(e) for e in init_base_err]).mean(dim=0)
    d_state = torch.stack([agg(e) for e in init_d_err]).mean(dim=0)
    m_state = torch.stack([agg(e) for e in init_m_err]).mean(dim=0)
    adv_d = (base_state.expand_as(base_pred[0]) - d_state.expand_as(base_pred[0])) / base_state.expand_as(base_pred[0]).clamp_min(1e-8)
    adv_m = (base_state.expand_as(base_pred[0]) - m_state.expand_as(base_pred[0])) / base_state.expand_as(base_pred[0]).clamp_min(1e-8)
    if int(init_base_err.shape[0]) < int(etth2.SPECIALIST_CONFIG.warmup):
        w_d = torch.zeros_like(adv_d)
        w_m = torch.zeros_like(adv_m)
    else:
        w_d, w_m = etth2.specialist_weight_pair(adv_d, adv_m, selected_core)
    pred = (1.0 - w_d - w_m).unsqueeze(0) * base_pred + w_d.unsqueeze(0) * d_pred + w_m.unsqueeze(0) * m_pred
    return pred, {
        "num_specialist_updates": 0,
        "specialist_weight_source": "router_train_initialized_repeated",
        "avg_weight_DLinear": float(w_d.mean()),
        "avg_weight_ModernTCN": float(w_m.mean()),
        "DLinear_specialist_disabled_duplicate_core": "DLinear" in selected_core,
        "ModernTCN_specialist_disabled_duplicate_core": "ModernTCN" in selected_core,
        "effective_horizon": h,
    }


def frozen_costar_prediction(
    dataset: str,
    cache: Mapping[str, Any],
    train_cache: Mapping[str, Any],
    std: torch.Tensor,
    expert_idx: Sequence[int],
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    base, base_extra = frozen_costar_base_prediction(dataset, cache, train_cache, std, expert_idx, seed, device)
    if dataset == "ETTh1":
        train_base, _ = etth1.parameterized_current_base_prediction(train_cache, train_cache, std, expert_idx, 7, device)
        target_train = train_cache["targets"].to(torch.float32)
        mask_train = train_cache["target_masks"].to(torch.bool)
        pred, extra = frozen_etth1_specialists(
            base,
            etth1.optional_prediction(cache, "DLinear"),
            etth1.optional_prediction(cache, "ModernTCN"),
            etth1.normalized_abs_error(train_base, target_train, mask_train, std),
            etth1.normalized_abs_error(etth1.optional_prediction(train_cache, "DLinear"), target_train, mask_train, std),
            etth1.normalized_abs_error(etth1.optional_prediction(train_cache, "ModernTCN"), target_train, mask_train, std),
        )
    elif dataset == "ETTh2":
        selected_core = {list(cache["expert_names"])[i] for i in expert_idx}
        init_base, _ = etth2.current_base_prediction(train_cache, train_cache, expert_idx, std)
        train_target = train_cache["targets"].to(torch.float32)
        train_mask = train_cache["target_masks"].to(torch.bool)
        pred, extra = frozen_etth2_specialists(
            base,
            etth2.expert_prediction(cache, "DLinear"),
            etth2.expert_prediction(cache, "ModernTCN"),
            etth2.abs_error(init_base, train_target, train_mask, std),
            etth2.abs_error(etth2.expert_prediction(train_cache, "DLinear"), train_target, train_mask, std),
            etth2.abs_error(etth2.expert_prediction(train_cache, "ModernTCN"), train_target, train_mask, std),
            selected_core,
        )
    else:
        raise ValueError(dataset)
    return pred, {**base_extra, **extra}


def online_prediction(
    dataset: str,
    cache: Mapping[str, Any],
    train_cache: Mapping[str, Any],
    std: torch.Tensor,
    expert_idx: Sequence[int],
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if dataset == "ETTh1":
        return etth1.evaluate_expanded(cache, train_cache, std, expert_idx, seed, device)
    if dataset == "ETTh2":
        return etth2.full_model_prediction(cache, train_cache, expert_idx, std)
    raise ValueError(dataset)


def evaluate_dataset(
    dataset: str,
    train_cache: Mapping[str, Any],
    val_cache: Mapping[str, Any],
    std: torch.Tensor,
    expert_names: Sequence[str],
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if dataset == "ETTh1":
        expert_idx = etth1.expert_indices(val_cache, expert_names)
        seeds = list(SEEDS)
    else:
        expert_idx = etth2.expert_indices(val_cache, expert_names)
        seeds = [7]

    fixed_pred = fixed_core_prediction(val_cache, expert_idx)
    fixed_metrics = metric_values(val_cache, fixed_pred, std)
    rows = [
        {
            "dataset": dataset,
            "method": "equal_fixed_three",
            "expert_set": "+".join(expert_names),
            "uses_earlier_validation_targets": False,
            "seeds": "",
            **fixed_metrics,
        }
    ]

    frozen_preds: list[torch.Tensor] = []
    frozen_seed_rows: list[dict[str, Any]] = []
    online_preds: list[torch.Tensor] = []
    online_seed_rows: list[dict[str, Any]] = []
    frozen_extra_last: dict[str, Any] = {}
    online_extra_last: dict[str, Any] = {}
    for seed in seeds:
        fp, fextra = frozen_costar_prediction(dataset, val_cache, train_cache, std, expert_idx, seed, device)
        op, oextra = online_prediction(dataset, val_cache, train_cache, std, expert_idx, seed, device)
        frozen_preds.append(fp)
        online_preds.append(op)
        fm = metric_values(val_cache, fp, std)
        om = metric_values(val_cache, op, std)
        frozen_seed_rows.append({"dataset": dataset, "method": "frozen_costar", "seed": seed, **fm})
        online_seed_rows.append({"dataset": dataset, "method": "online_costar", "seed": seed, **om})
        frozen_extra_last = fextra
        online_extra_last = oextra

    frozen_mean = torch.stack(frozen_preds).mean(dim=0)
    online_mean = torch.stack(online_preds).mean(dim=0)
    frozen_metrics = metric_values(val_cache, frozen_mean, std)
    online_metrics = metric_values(val_cache, online_mean, std)
    rows.append(
        {
            "dataset": dataset,
            "method": "frozen_costar",
            "expert_set": "+".join(expert_names),
            "uses_earlier_validation_targets": False,
            "seeds": ",".join(str(s) for s in seeds),
            "seed_mae_mean": float(torch.tensor([r["mae"] for r in frozen_seed_rows]).mean()),
            "seed_mae_std": float(torch.tensor([r["mae"] for r in frozen_seed_rows]).std(unbiased=False)),
            **frozen_metrics,
            **frozen_extra_last,
        }
    )
    rows.append(
        {
            "dataset": dataset,
            "method": "online_costar",
            "expert_set": "+".join(expert_names),
            "uses_earlier_validation_targets": True,
            "seeds": ",".join(str(s) for s in seeds),
            "seed_mae_mean": float(torch.tensor([r["mae"] for r in online_seed_rows]).mean()),
            "seed_mae_std": float(torch.tensor([r["mae"] for r in online_seed_rows]).std(unbiased=False)),
            **online_metrics,
            **online_extra_last,
        }
    )
    diagnostics = {
        "expert_indices": list(expert_idx),
        "seed_rows": frozen_seed_rows + online_seed_rows,
        "frozen_first_window_equals_online": bool(torch.allclose(frozen_mean[0], online_mean[0], atol=1e-5, rtol=0.0)),
        "frozen_first_window_max_abs_diff": float((frozen_mean[0] - online_mean[0]).abs().max()),
        "online_expected_mae": ETTH1_EXPECTED_ONLINE_MAE if dataset == "ETTh1" else ETTH2_EXPECTED_ONLINE_MAE,
        "online_expected_mae_abs_diff": abs(
            online_metrics["mae"] - (ETTH1_EXPECTED_ONLINE_MAE if dataset == "ETTh1" else ETTH2_EXPECTED_ONLINE_MAE)
        ),
    }
    return rows, diagnostics


def run_leakage_checks(
    dataset: str,
    train_cache: Mapping[str, Any],
    val_cache: Mapping[str, Any],
    std: torch.Tensor,
    expert_names: Sequence[str],
    device: torch.device,
) -> dict[str, Any]:
    if dataset == "ETTh1":
        expert_idx = etth1.expert_indices(val_cache, expert_names)
        seed = 7
    else:
        expert_idx = etth2.expert_indices(val_cache, expert_names)
        seed = 7
    before = tensor_digest(val_cache)
    pred, _ = frozen_costar_prediction(dataset, val_cache, train_cache, std, expert_idx, seed, device)
    after = tensor_digest(val_cache)
    if before != after:
        raise AssertionError(f"{dataset} frozen prediction mutated validation cache tensors")
    target_mut = cloned_with_random_targets(val_cache, randomize_masks=False)
    mask_mut = cloned_with_random_targets(val_cache, randomize_masks=True)
    pred_target, _ = frozen_costar_prediction(dataset, target_mut, train_cache, std, expert_idx, seed, device)
    pred_mask, _ = frozen_costar_prediction(dataset, mask_mut, train_cache, std, expert_idx, seed, device)
    if not torch.equal(pred, pred_target):
        raise AssertionError(f"{dataset} frozen predictions changed after replacing validation targets")
    if not torch.equal(pred, pred_mask):
        raise AssertionError(f"{dataset} frozen predictions changed after replacing validation masks")
    online_pred, _ = online_prediction(dataset, val_cache, train_cache, std, expert_idx, seed, device)
    online_mut, _ = online_prediction(dataset, target_mut, train_cache, std, expert_idx, seed, device)
    online_changed = not torch.equal(online_pred, online_mut)
    if not online_changed:
        raise AssertionError(f"{dataset} online COSTAR did not respond to target replacement; check test setup")
    first_diff = float((pred[0] - online_pred[0]).abs().max())
    first_equal = first_diff <= 1e-5
    if not first_equal:
        raise AssertionError(f"{dataset} frozen and online predictions do not share first-window initialization")
    return {
        "validation_target_replacement_unchanged": True,
        "validation_mask_replacement_unchanged": True,
        "validation_cache_prediction_tensors_unchanged": True,
        "online_costar_target_replacement_changed_predictions": online_changed,
        "frozen_and_online_first_window_equal": first_equal,
        "first_window_max_abs_diff": first_diff,
    }


def make_report(report: Mapping[str, Any]) -> None:
    lines = [
        "# Frozen COSTAR Validation Comparison",
        "",
        "This validation-only experiment isolates sequential validation-target feedback.",
        "The frozen path repeats router-train initialized general and horizon-variable weights across all validation windows.",
        "",
        "## Target-Feedback Trace",
        "",
        "- `parameterized_current_base_prediction()` / ETTh2 `current_base_prediction()` read validation errors through `per_location_abs_error_for_indices(cache, ...)` or `per_location_error(cache, ...)`.",
        "- `chronological_online_weights()` updates EMA state from validation expert MAE after `old_start + horizon <= current_start`.",
        "- `chronological_hv_weights()` updates horizon-variable EMA state from validation per-location expert errors after the same causal delay.",
        "- `run_causal_specialists()` and ETTh2 `run_specialists_no_duplicate()` update specialist states from validation base/DLinear/ModernTCN absolute errors.",
        "- Static ETTh1 neural weights are produced from current history and forecasts only in the frozen path; target-based MAE/MSE reporting from the legacy loader is not used.",
        "",
        "## Results",
        "",
        "| Dataset | Equal fixed-three | Frozen COSTAR | Online COSTAR |",
        "|---|---:|---:|---:|",
    ]
    rows = report["results"]
    for dataset in ["ETTh1", "ETTh2"]:
        by_method = {row["method"]: row for row in rows if row["dataset"] == dataset}
        lines.append(
            "| {dataset} | `{fixed:.6f}` / `{fixed_mse:.6f}` | `{frozen:.6f}` / `{frozen_mse:.6f}` | `{online:.6f}` / `{online_mse:.6f}` |".format(
                dataset=dataset,
                fixed=by_method["equal_fixed_three"]["mae"],
                fixed_mse=by_method["equal_fixed_three"]["mse"],
                frozen=by_method["frozen_costar"]["mae"],
                frozen_mse=by_method["frozen_costar"]["mse"],
                online=by_method["online_costar"]["mae"],
                online_mse=by_method["online_costar"]["mse"],
            )
        )
    lines.extend(
        [
            "",
            "## Configuration",
            "",
            f"- ETTh1 core: `{'+'.join(report['datasets']['ETTh1']['core'])}`.",
            f"- ETTh2 core: `{'+'.join(report['datasets']['ETTh2']['core'])}`.",
            "- Base mixture: `0.25` chronological branch, `0.75` horizon-variable branch.",
            "- Chronological branch: `0.5` static prior, `0.5` router-train EMA initialization.",
            "- Horizon-variable branch: low-rank rank `1`, decay `0.95`, temperature `0.1`, frozen at router-train initialization.",
            "- Specialist config: `both_variable_decay0.95_cap0.1_marginbp200_warm96`, frozen at router-train initialization.",
            "",
            "## Leakage Checks",
            "",
        ]
    )
    for dataset, checks in report["leakage_checks"].items():
        lines.append(f"- {dataset}: all frozen target/mask mutation and cache immutability checks passed.")
        lines.append(f"  Online target replacement changed predictions: `{checks['online_costar_target_replacement_changed_predictions']}`.")
    lines.extend(["", "## Reproduce", "", "```powershell", report["command"], "```"])
    (OUT_DIR / "frozen_costar_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    global OUT_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    OUT_DIR = Path(args.out_dir)
    start_time = time.time()
    device = torch.device(args.device)

    paths = {
        "ETTh1": {
            "train_cache": ROOT / "cache/costarts_walkforward/router_train_20_60_cache.pt",
            "val_cache": ROOT / "cache/costarts_walkforward/router_val_60_80_cache.pt",
            "normalizer": ROOT / "checkpoints/costarts_walkforward/final_60/DLinear/best_expert.pt",
            "frozen_config": ETTH1_FROZEN,
        },
        "ETTh2": {
            "train_cache": ROOT / "cache/costarts_fresh/ETTh2_96_12/router_train_cache.pt",
            "val_cache": ROOT / "cache/costarts_fresh/ETTh2_96_12/router_val_cache.pt",
            "normalizer": None,
            "frozen_config": ETTH2_FROZEN,
        },
    }
    for dataset_paths in paths.values():
        for key in ["train_cache", "val_cache", "frozen_config"]:
            refuse_test(dataset_paths[key])

    report: dict[str, Any] = {
        "experiment": "frozen_costar",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "test_loaded": False,
        "test_evaluated": False,
        "command": f"python experiments\\frozen_costar\\run_frozen_costar_validation.py --device {args.device}",
        "results": [],
        "datasets": {},
        "leakage_checks": {},
        "seed_details": [],
    }

    for dataset in ["ETTh1", "ETTh2"]:
        p = paths[dataset]
        train_cache = load_cache(p["train_cache"], "router_train_20_60" if dataset == "ETTh1" else "router_train")
        val_cache = load_cache(p["val_cache"], "router_val_60_80" if dataset == "ETTh1" else "router_val")
        if dataset == "ETTh1":
            std = load_std(p["normalizer"], int(val_cache["num_features"]))
        else:
            std = torch.ones(int(val_cache["num_features"]), dtype=torch.float32)
        core = load_frozen_core(p["frozen_config"])
        rows, diagnostics = evaluate_dataset(dataset, train_cache, val_cache, std, core, device)
        checks = run_leakage_checks(dataset, train_cache, val_cache, std, core, device)
        report["results"].extend(rows)
        report["seed_details"].extend(diagnostics.pop("seed_rows"))
        report["leakage_checks"][dataset] = checks
        report["datasets"][dataset] = {
            "core": core,
            "train_cache": str(p["train_cache"].relative_to(ROOT)),
            "val_cache": str(p["val_cache"].relative_to(ROOT)),
            "normalizer": str(p["normalizer"].relative_to(ROOT)) if p["normalizer"] is not None else "std_ones",
            "frozen_config": str(p["frozen_config"].relative_to(ROOT)),
            "cache_hashes": {
                "train_cache_sha256": sha256_file(p["train_cache"]),
                "val_cache_sha256": sha256_file(p["val_cache"]),
            },
            "diagnostics": diagnostics,
        }
        expected_diff = diagnostics["online_expected_mae_abs_diff"]
        if expected_diff > 5e-5:
            raise AssertionError(f"{dataset} online COSTAR MAE drifted by {expected_diff}")

    report["runtime_sec"] = time.time() - start_time
    write_json(OUT_DIR / "frozen_costar_validation_results.json", report)
    write_csv(OUT_DIR / "frozen_costar_validation_results.csv", report["results"])
    write_csv(OUT_DIR / "frozen_costar_seed_results.csv", report["seed_details"])
    write_json(
        OUT_DIR / "frozen_costar_config.json",
        {
            "method": "frozen_costar",
            "online_method_label": "online_costar",
            "selection_rule": "reuse existing frozen train-selected cores",
            "no_validation_target_feedback": True,
            "no_test_cache_loaded": True,
            "base_mixture": {"chronological": 0.25, "horizon_variable": 0.75},
            "chrono_branch": {"static_prior": 0.5, "train_initialized_online": 0.5},
            "hv_trial": asdict(HV_TRIAL),
            "specialist_config": asdict(etth1.SPECIALIST_CONFIG),
        },
    )
    make_report(report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
