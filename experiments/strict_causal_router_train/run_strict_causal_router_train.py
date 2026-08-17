"""Strictly causal router-train rebuild for the active ETTh1 COSTAR path.

This runner exists to remove a subtle training-time chronology mismatch from the
historical COSTAR experiments: when the adaptive COSTAR predictor was evaluated
on ``router_train`` itself, its initial EMA/HxV states could be computed from the
complete router-train cache.  That is legal for *future validation* initialization
because router-train is then fully historical, but it is not a prequential
prediction of the router-train windows themselves.

The strict path below makes every adaptive router-train prediction using only
information observable before that prediction origin:

* global EMA state starts label-free (equal weights),
* horizon-variable EMA state starts label-free (equal weights),
* a router-train target can update state only when
  ``old_start + forecast_horizon <= current_start``,
* core and specialist hyperparameters are selected using router-train only,
* validation is loaded only after the strict router-train configuration is
  frozen,
* no test cache is loaded by this script.

Historical result artifacts are intentionally left untouched.  This produces a
new strict-causal candidate that must be reported separately until replicated.
"""

from __future__ import annotations

import argparse
import json
import math
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
    enforce_observable,
)
from experiments.expanded_expert_pool_costar import run_expanded_expert_pool as specialist_impl  # noqa: E402
from experiments.oracle_weight_tournament.run_tournament import load_cache, load_std, sample_mae, sample_mse  # noqa: E402
from experiments.train_selected_core_etth1 import run_train_selected_core_eval as main_impl  # noqa: E402


CHRONO_DECAY = 0.97
CHRONO_TEMPERATURE = 0.1
CHRONO_STATIC_BLEND = 0.5
HV_DECAY = 0.95
HV_TEMPERATURE = 0.1
HV_RANK = 1
FINAL_CHRONO_BLEND = 0.25
FINAL_HV_BLEND = 0.75


def refuse_test(path: str | Path) -> None:
    if "test" in str(path).lower():
        raise ValueError(f"Refusing test path: {path}")


def write_json(path: Path, payload: Mapping[str, Any] | Sequence[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def softmax_neg(errors: torch.Tensor, temperature: float) -> torch.Tensor:
    return torch.softmax(-errors / max(float(temperature), 1e-8), dim=-1)


def low_rank_matrix(x: torch.Tensor, rank: int) -> torch.Tensor:
    """Rank-r approximation of each expert's HxV error surface."""
    if rank <= 0:
        return x.mean(dim=(0, 1), keepdim=True).expand_as(x)
    pieces = []
    for expert in range(x.shape[-1]):
        mat = x[..., expert]
        u, s, vh = torch.linalg.svd(mat, full_matrices=False)
        r = min(int(rank), int(s.numel()))
        pieces.append((u[:, :r] * s[:r]) @ vh[:r])
    return torch.stack(pieces, dim=-1)


def equal_prior_hv_weights(error_hve: torch.Tensor, rank: int, temperature: float) -> torch.Tensor:
    """Convert HxVxE errors to weights with an equal, identity-agnostic prior."""
    err = low_rank_matrix(error_hve, rank)
    centered = err - err.mean(dim=-1, keepdim=True)
    return torch.softmax(-centered / max(float(temperature), 1e-8), dim=-1)


def strict_global_weights(
    starts: torch.Tensor,
    expert_mae: torch.Tensor,
    horizon: int,
    decay: float = CHRONO_DECAY,
    temperature: float = CHRONO_TEMPERATURE,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Prequential expert weights with no target-derived initialization."""
    if expert_mae.ndim != 2:
        raise ValueError(f"expert_mae must be [N,E], got {tuple(expert_mae.shape)}")
    if not bool(torch.all(starts[1:] > starts[:-1])):
        raise ValueError("Router-train starts must be strictly chronological")

    ema = torch.zeros(expert_mae.shape[1], dtype=torch.float32)
    pending: list[int] = []
    weights: list[torch.Tensor] = []
    updates = 0
    first_update_at: int | None = None

    for i in range(expert_mae.shape[0]):
        now = int(starts[i])
        still_pending: list[int] = []
        for j in pending:
            if int(starts[j]) + horizon <= now:
                enforce_observable(int(starts[j]), now, horizon)
                ema = float(decay) * ema + (1.0 - float(decay)) * expert_mae[j]
                updates += 1
                if first_update_at is None:
                    first_update_at = i
            else:
                still_pending.append(j)
        pending = still_pending
        weights.append(softmax_neg(ema, temperature))
        pending.append(i)

    return torch.stack(weights), {
        "num_updates": updates,
        "first_update_row": first_update_at,
        "label_derived_initial_state": False,
    }


def strict_hv_weights(
    starts: torch.Tensor,
    error_hve: torch.Tensor,
    horizon: int,
    decay: float = HV_DECAY,
    temperature: float = HV_TEMPERATURE,
    rank: int = HV_RANK,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Prequential horizon-variable weights with equal label-free initialization."""
    if error_hve.ndim != 4:
        raise ValueError(f"error_hve must be [N,H,V,E], got {tuple(error_hve.shape)}")
    if not bool(torch.all(starts[1:] > starts[:-1])):
        raise ValueError("Router-train starts must be strictly chronological")

    ema = torch.zeros_like(error_hve[0], dtype=torch.float32)
    pending: list[int] = []
    weights: list[torch.Tensor] = []
    updates = 0
    first_update_at: int | None = None

    for i in range(error_hve.shape[0]):
        now = int(starts[i])
        still_pending: list[int] = []
        for j in pending:
            if int(starts[j]) + horizon <= now:
                enforce_observable(int(starts[j]), now, horizon)
                ema = float(decay) * ema + (1.0 - float(decay)) * error_hve[j]
                updates += 1
                if first_update_at is None:
                    first_update_at = i
            else:
                still_pending.append(j)
        pending = still_pending
        weights.append(equal_prior_hv_weights(ema, rank, temperature))
        pending.append(i)

    return torch.stack(weights), {
        "num_updates": updates,
        "first_update_row": first_update_at,
        "label_derived_initial_state": False,
    }


def strict_router_train_base_prediction(
    cache: Mapping[str, Any],
    std: torch.Tensor,
    expert_idx: Sequence[int],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Generate COSTAR router-train predictions strictly prequentially.

    Crucially, this function has no ``train_cache_for_init`` argument.  The
    router-train cache may provide the error of an old prediction only after that
    prediction's full horizon is observable.
    """
    role = cache.get("cache_role", cache.get("split_role"))
    if role != "router_train_20_60":
        raise ValueError(f"Strict training prediction requires router_train_20_60, got {role!r}")

    starts = cache["absolute_window_starts"].to(torch.long)
    horizon = int(cache["forecast_horizon"])
    forecasts = main_impl.selected_forecasts(cache, expert_idx)
    e = len(expert_idx)
    if e != 3:
        raise ValueError(f"Active COSTAR path expects exactly three core experts, got {e}")

    expert_err = main_impl.per_location_abs_error_for_indices(cache, std, expert_idx).mean(dim=(1, 2))
    online_w, online_extra = strict_global_weights(starts, expert_err, horizon)
    static_w = torch.full_like(online_w, 1.0 / float(e))
    chrono_w = (1.0 - CHRONO_STATIC_BLEND) * static_w + CHRONO_STATIC_BLEND * online_w
    chrono_w = chrono_w / chrono_w.sum(dim=1, keepdim=True).clamp_min(1e-8)
    chrono_pred = main_impl.weighted_forecast(forecasts, chrono_w)

    hv_err = main_impl.per_location_abs_error_for_indices(cache, std, expert_idx)
    hv_w, hv_extra = strict_hv_weights(starts, hv_err, horizon)
    hv_pred = (forecasts * hv_w).sum(dim=-1)

    pred = FINAL_CHRONO_BLEND * chrono_pred + FINAL_HV_BLEND * hv_pred
    return pred, {
        "strict_prequential_router_train": True,
        "initial_global_weights": [1.0 / e] * e,
        "initial_hv_weights": [1.0 / e] * e,
        "chrono_num_updates": online_extra["num_updates"],
        "hv_num_updates": hv_extra["num_updates"],
        "chrono_first_update_row": online_extra["first_update_row"],
        "hv_first_update_row": hv_extra["first_update_row"],
    }


def prefix_causality_probe(
    cache: Mapping[str, Any],
    std: torch.Tensor,
    expert_idx: Sequence[int],
    probe_windows: int = 256,
) -> dict[str, Any]:
    """Machine-check that future router-train targets cannot alter earlier predictions."""
    n = min(int(cache["num_windows"]), int(probe_windows))
    if n < 32:
        raise ValueError("Need at least 32 windows for causality probe")

    keys_to_slice = (
        "histories",
        "targets",
        "target_masks",
        "prediction_stack",
        "absolute_window_starts",
    )
    small = dict(cache)
    for key in keys_to_slice:
        small[key] = cache[key][:n].clone()
    small["num_windows"] = n

    cut = n // 2
    pred_a, _ = strict_router_train_base_prediction(small, std, expert_idx)
    perturbed = dict(small)
    perturbed["targets"] = small["targets"].clone()
    perturbed["targets"][cut:] = perturbed["targets"][cut:] + 1000.0
    pred_b, _ = strict_router_train_base_prediction(perturbed, std, expert_idx)

    starts = small["absolute_window_starts"].to(torch.long)
    horizon = int(small["forecast_horizon"])
    changed_target_start = int(starts[cut])
    first_affected = n
    for i in range(cut, n):
        if changed_target_start + horizon <= int(starts[i]):
            first_affected = i
            break

    max_prefix_diff = float((pred_a[:first_affected] - pred_b[:first_affected]).abs().max()) if first_affected > 0 else 0.0
    passed = bool(torch.equal(pred_a[:first_affected], pred_b[:first_affected]))
    if not passed:
        raise AssertionError(f"Future-target perturbation changed causal prefix; max diff={max_prefix_diff}")
    return {
        "passed": passed,
        "probe_windows": n,
        "perturbed_from_row": cut,
        "first_prediction_allowed_to_change": first_affected,
        "max_unaffected_prefix_abs_diff": max_prefix_diff,
    }


def select_strict_config(
    train_cache: Mapping[str, Any],
    std: torch.Tensor,
    expert_idx: Sequence[int],
) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
    """Select specialist config using only strict-causal router-train predictions."""
    train_base, base_extra = strict_router_train_base_prediction(train_cache, std, expert_idx)
    folds = specialist_impl.train_folds(int(train_cache["num_windows"]))
    configs = specialist_impl.grid()
    configs_by_name = {cfg.name: cfg for cfg in configs}
    leaderboard, _ = specialist_impl.grid_eval_cached(train_cache, std, train_base, configs, folds)
    selected_all = specialist_impl.select_with_one_se(leaderboard, configs_by_name)
    selected_both = selected_all["both"]["selected"]
    return train_base, {
        "all_scenarios": selected_all,
        "selected_both": selected_both,
        "folds": [{"fold": i, "train_lo": lo, "eval_lo": evlo, "eval_hi": evhi} for i, (lo, evlo, evhi) in enumerate(folds)],
    }, base_extra


def config_from_row(row: Mapping[str, Any]) -> specialist_impl.Config:
    return specialist_impl.Config(
        scenario=str(row["scenario"]),
        structure=str(row["structure"]),
        decay=float(row["decay"]),
        extra_weight_cap=float(row["extra_weight_cap"]),
        activation_margin=float(row["activation_margin"]),
        warmup=int(row["warmup"]),
    )


def metrics(cache: Mapping[str, Any], std: torch.Tensor, pred: torch.Tensor) -> dict[str, float]:
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.bool)
    return {
        "mae": float(sample_mae(pred, target, mask, std).mean()),
        "mse": float(sample_mse(pred, target, mask, std).mean()),
    }


def evaluate_validation(
    train_cache: Mapping[str, Any],
    val_cache: Mapping[str, Any],
    std: torch.Tensor,
    expert_idx: Sequence[int],
    train_base_strict: torch.Tensor,
    specialist_config: specialist_impl.Config,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Evaluate frozen strict-selected configuration on validation.

    Validation initialization may use all router-train labels because every one
    of those labels is historical by the validation origin.  The specialist
    baseline errors, however, are computed from the strict prequential
    router-train base predictions.
    """
    base, base_extra = main_impl.parameterized_current_base_prediction(
        val_cache,
        train_cache,
        std,
        expert_idx,
        7,
        device,
    )

    train_target = train_cache["targets"].to(torch.float32)
    train_mask = train_cache["target_masks"].to(torch.bool)
    val_target = val_cache["targets"].to(torch.float32)
    val_mask = val_cache["target_masks"].to(torch.bool)
    d_train = main_impl.optional_prediction(train_cache, "DLinear")
    m_train = main_impl.optional_prediction(train_cache, "ModernTCN")
    d_val = main_impl.optional_prediction(val_cache, "DLinear")
    m_val = main_impl.optional_prediction(val_cache, "ModernTCN")

    init_base_err = main_impl.normalized_abs_error(train_base_strict, train_target, train_mask, std)
    init_d_err = main_impl.normalized_abs_error(d_train, train_target, train_mask, std)
    init_m_err = main_impl.normalized_abs_error(m_train, train_target, train_mask, std)

    pred, specialist_extra, _ = specialist_impl.run_causal_specialists(
        val_cache["absolute_window_starts"].to(torch.long),
        base,
        d_val,
        m_val,
        val_target,
        val_mask,
        std,
        specialist_config,
        init_base_err,
        init_d_err,
        init_m_err,
    )
    return pred, {**base_extra, **specialist_extra}


def phase_select(args: argparse.Namespace) -> None:
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = ROOT / args.train_cache
    norm_path = ROOT / args.normalizer_checkpoint
    refuse_test(train_path)
    refuse_test(norm_path)

    train_cache = load_cache(train_path, "router_train_20_60")
    std = load_std(norm_path, int(train_cache["num_features"]))
    _, core = main_impl.select_core_on_router_train(train_cache, std)
    expert_idx = [int(i) for i in core["expert_indices"]]

    probe = prefix_causality_probe(train_cache, std, expert_idx)
    train_base, selected_specialists, base_extra = select_strict_config(train_cache, std, expert_idx)
    selected_both = selected_specialists["selected_both"]

    payload = {
        "phase": "strict_router_train_frozen_before_validation",
        "router_val_loaded": False,
        "test_cache_loaded": False,
        "selected_three_experts": list(core["experts"]),
        "selected_expert_indices": expert_idx,
        "core_selection_metric": "router-train chronological OOF MAE",
        "router_train_oof_mae": float(core["pooled_oof_mae"]),
        "strict_router_train": {
            "enabled": True,
            "initialization": "label-free equal state",
            "causal_rule": "old_start + forecast_horizon <= current_start",
            "same_cache_full_target_initialization": False,
            "base_extra": base_extra,
            "causality_probe": probe,
            "strict_train_base_mae": metrics(train_cache, std, train_base)["mae"],
        },
        "selected_specialist_config": selected_both,
        "specialist_selection": selected_specialists,
        "fixed_adaptive_hyperparameters": {
            "chrono_decay": CHRONO_DECAY,
            "chrono_temperature": CHRONO_TEMPERATURE,
            "chrono_static_blend": CHRONO_STATIC_BLEND,
            "hv_decay": HV_DECAY,
            "hv_temperature": HV_TEMPERATURE,
            "hv_rank": HV_RANK,
            "final_chrono_blend": FINAL_CHRONO_BLEND,
            "final_hv_blend": FINAL_HV_BLEND,
        },
    }
    write_json(out_dir / "frozen_config_before_validation.json", payload)


def phase_evaluate(args: argparse.Namespace) -> None:
    out_dir = ROOT / args.out_dir
    frozen_path = out_dir / "frozen_config_before_validation.json"
    if not frozen_path.exists():
        raise FileNotFoundError("Run --phase select before --phase evaluate")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen.get("router_val_loaded"):
        raise RuntimeError("Strict config was not frozen before validation")

    train_path = ROOT / args.train_cache
    val_path = ROOT / args.val_cache
    norm_path = ROOT / args.normalizer_checkpoint
    for path in (train_path, val_path, norm_path):
        refuse_test(path)

    train_cache = load_cache(train_path, "router_train_20_60")
    val_cache = load_cache(val_path, "router_val_60_80")
    std = load_std(norm_path, int(val_cache["num_features"]))
    expert_idx = [int(i) for i in frozen["selected_expert_indices"]]
    config = config_from_row(frozen["selected_specialist_config"])
    train_base, _ = strict_router_train_base_prediction(train_cache, std, expert_idx)

    pred, extra = evaluate_validation(
        train_cache,
        val_cache,
        std,
        expert_idx,
        train_base,
        config,
        torch.device(args.device),
    )
    result = metrics(val_cache, std, pred)
    fixed_pred = main_impl.selected_forecasts(val_cache, expert_idx).mean(dim=-1)
    fixed_result = metrics(val_cache, std, fixed_pred)

    report = {
        "label": "strict_causal_router_train_validation",
        "configuration_frozen_before_validation": True,
        "test_cache_loaded": False,
        "selected_three_experts": frozen["selected_three_experts"],
        "selected_specialist_config": {"name": config.name, **asdict(config)},
        "validation": result,
        "fixed_core_validation": fixed_result,
        "delta_mae_vs_fixed_core": result["mae"] - fixed_result["mae"],
        "router_train_causality_probe": frozen["strict_router_train"]["causality_probe"],
        "prediction_extra": extra,
        "interpretation": "New strict-causal router-train candidate; do not rewrite historical preregistered or after-test audit rows.",
    }
    write_json(out_dir / "validation_report.json", report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("select", "evaluate", "all"), default="all")
    parser.add_argument("--train-cache", default="cache/costarts_walkforward/router_train_20_60_cache.pt")
    parser.add_argument("--val-cache", default="cache/costarts_walkforward/router_val_60_80_cache.pt")
    parser.add_argument("--normalizer-checkpoint", default="checkpoints/costarts_walkforward/final_60/DLinear/best_expert.pt")
    parser.add_argument("--out-dir", default="experiments/strict_causal_router_train/results")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    started = time.time()
    if args.phase in {"select", "all"}:
        phase_select(args)
    if args.phase in {"evaluate", "all"}:
        phase_evaluate(args)
    print(f"strict causal router-train run completed in {time.time() - started:.2f}s")


if __name__ == "__main__":
    main()
