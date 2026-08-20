"""Dual-timescale (short-term / long-term) memory COSTAR.

Adds a second, slower EMA memory alongside the existing fast EMA memory used
by the global chronological branch (`chronological_online_weights`) and the
horizon-variable branch (`chronological_hv_weights` / `errors_to_weights`).
Both memories are built causally from router_train only, frozen before
router_val for the Frozen path, and updated causally during router_val
(after `old_start + horizon <= current_start`) for the Online path.

HARD RULE: this script never loads, inspects, or scores against any cache or
file whose path contains "test". `refuse_test()` guards every cache/config
path argument. Only `router_train` (model/hyperparameter selection) and a
single frozen `router_val` evaluation are used.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.etth2_train_selected_core.run_etth2_train_selected_core_eval as etth2  # noqa: E402
import experiments.train_selected_core_etth1.run_train_selected_core_eval as etth1  # noqa: E402
from experiments.chronological_adaptive_costar.run_chronological_adaptive_costar import (  # noqa: E402
    enforce_observable,
    softmax_neg,
)
from experiments.expanded_expert_pool_costar.run_expanded_expert_pool import run_causal_specialists  # noqa: E402
from experiments.frozen_costar.run_frozen_costar_validation import (  # noqa: E402
    ETTH1_FROZEN,
    ETTH2_FROZEN,
    cloned_with_random_targets,
    frozen_etth1_specialists,
    frozen_etth2_specialists,
    load_frozen_core,
    metric_values,
    online_prediction as original_online_prediction,
    frozen_costar_prediction as original_frozen_prediction,
    tensor_digest,
)
from experiments.horizon_variable_adaptive_costar.run_hv_adaptive_costar import (  # noqa: E402
    Trial as HvTrial,
    errors_to_weights,
    predict_from_hv_weights,
)
from experiments.oracle_weight_tournament.run_tournament import (  # noqa: E402
    load_cache,
    load_std,
    sample_mae,
    sample_mse,
    weighted_forecast,
)


OUT_DIR = ROOT / "experiments/dual_timescale_memory_costar"
SHORT_DECAY_GRID = (0.80, 0.90, 0.95)
LONG_DECAY_GRID = (0.97, 0.99, 0.995)
MIX_GRID = (0.1, 0.3, 0.5, 0.7, 0.9)
SCHEDULE_GRID = ((1.0, 0.0), (0.9, 0.1), (0.8, 0.2), (0.7, 0.3), (0.6, 0.4), (0.5, 0.5))
CHRONO_TEMPERATURE = 0.1
DUAL_HV_TRIAL = HvTrial("hv_ema", "dual_hv", mode="hv_lowrank", rank=1, decay=0.95, temperature=0.1)


def refuse_test(path: str | Path) -> None:
    """Hard guard: never touch a cache/config path that mentions "test"."""
    if "test" in str(path).lower():
        raise ValueError(f"Test access forbidden during dual-memory development: {path}")


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


# ---------------------------------------------------------------------------
# Core dual-memory primitives
# ---------------------------------------------------------------------------


def train_folds(n: int) -> list[tuple[int, int, int]]:
    min_train = int(round(n * 0.2))
    usable = n - min_train
    bounds = [min_train + i * usable // 4 for i in range(5)]
    return [(0, bounds[i], bounds[i + 1]) for i in range(4)]


def constant_schedule(value: float) -> Callable[[int], torch.Tensor]:
    return lambda horizon: torch.full((horizon,), float(value))


def linspace_schedule(start: float, end: float) -> Callable[[int], torch.Tensor]:
    return lambda horizon: torch.linspace(float(start), float(end), horizon)


def build_causal_dual_memory(
    starts: torch.Tensor,
    per_window_error: torch.Tensor,
    horizon: int,
    short_decay: float,
    long_decay: float,
    seed_value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int, list[dict[str, Any]]]:
    """Chronologically build final short/long EMA states from realized errors.

    Both states are seeded from the single earliest realized error (not a
    pooled mean over the whole range), so their divergence comes purely from
    differing decay speeds applied over the same causal update sequence.
    """
    n = int(per_window_error.shape[0])
    short = seed_value.clone()
    long_ = seed_value.clone()
    pending: list[int] = []
    audit: list[dict[str, Any]] = []
    updates = 0
    for i in range(n):
        now = int(starts[i])
        still: list[int] = []
        for j in pending:
            due = int(starts[j]) + horizon
            if due <= now:
                enforce_observable(int(starts[j]), now, horizon)
                loss = per_window_error[j]
                short = short_decay * short + (1.0 - short_decay) * loss
                long_ = long_decay * long_ + (1.0 - long_decay) * loss
                updates += 1
                audit.append(
                    {
                        "forecast_start": int(starts[j]),
                        "target_end": due,
                        "current_start": now,
                        "memory_updated": True,
                        "short_memory_updated": True,
                        "long_memory_updated": True,
                    }
                )
            else:
                still.append(j)
        pending = still
        pending.append(i)
    return short, long_, updates, audit


def causal_dual_weight_sequence(
    starts: torch.Tensor,
    per_window_error: torch.Tensor,
    horizon: int,
    short_decay: float,
    long_decay: float,
    short_init: torch.Tensor,
    long_init: torch.Tensor,
    weight_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, int, list[dict[str, Any]]]:
    """Causal walk-forward: emit a weight per window from dual EMA state that
    only reflects updates whose full target horizon is already observable."""
    n = int(per_window_error.shape[0])
    short = short_init.clone()
    long_ = long_init.clone()
    pending: list[int] = []
    weights = []
    audit: list[dict[str, Any]] = []
    updates = 0
    for i in range(n):
        now = int(starts[i])
        still: list[int] = []
        for j in pending:
            due = int(starts[j]) + horizon
            if due <= now:
                enforce_observable(int(starts[j]), now, horizon)
                loss = per_window_error[j]
                short = short_decay * short + (1.0 - short_decay) * loss
                long_ = long_decay * long_ + (1.0 - long_decay) * loss
                updates += 1
                audit.append(
                    {
                        "forecast_start": int(starts[j]),
                        "target_end": due,
                        "current_start": now,
                        "memory_updated": True,
                        "short_memory_updated": True,
                        "long_memory_updated": True,
                    }
                )
            else:
                still.append(j)
        pending = still
        weights.append(weight_fn(short, long_))
        pending.append(i)
    return torch.stack(weights), updates, audit


def causal_dual_state_walk(
    starts: torch.Tensor,
    per_window_error: torch.Tensor,
    horizon: int,
    short_decay: float,
    long_decay: float,
    short_init: torch.Tensor,
    long_init: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Same causal walk as `causal_dual_weight_sequence`, but records the raw
    short/long EMA state seen at each window (before that window's own error
    is folded in) instead of collapsing it through a weight function. Lets a
    single walk serve as the source for the short-only view, the long-only
    view, and any mixed view of the same underlying dual memory."""
    n = int(per_window_error.shape[0])
    short = short_init.clone()
    long_ = long_init.clone()
    pending: list[int] = []
    short_states = []
    long_states = []
    for i in range(n):
        now = int(starts[i])
        still: list[int] = []
        for j in pending:
            due = int(starts[j]) + horizon
            if due <= now:
                enforce_observable(int(starts[j]), now, horizon)
                loss = per_window_error[j]
                short = short_decay * short + (1.0 - short_decay) * loss
                long_ = long_decay * long_ + (1.0 - long_decay) * loss
            else:
                still.append(j)
        pending = still
        short_states.append(short.clone())
        long_states.append(long_.clone())
        pending.append(i)
    return torch.stack(short_states), torch.stack(long_states)


def paired_block_bootstrap(
    candidate: torch.Tensor, baseline: torch.Tensor, block: int = 48, seed: int = 777, samples: int = 5000
) -> dict[str, Any]:
    diff = candidate - baseline
    n = diff.numel()
    block = max(1, min(block, n))
    n_blocks = max(1, -(-n // block))
    gen = torch.Generator().manual_seed(seed)
    vals = []
    for _ in range(samples):
        starts_idx = torch.randint(0, n - block + 1, (n_blocks,), generator=gen)
        idx = torch.cat([torch.arange(s, s + block) for s in starts_idx.tolist()])[:n]
        vals.append(float(diff[idx].mean()))
    t = torch.tensor(vals)
    return {
        "mean_diff_candidate_minus_baseline": float(diff.mean()),
        "block_size": block,
        "num_blocks": n_blocks,
        "ci95_low": float(torch.quantile(t, 0.025)),
        "ci95_high": float(torch.quantile(t, 0.975)),
        "ci_excludes_zero": bool(torch.quantile(t, 0.975) < 0 or torch.quantile(t, 0.025) > 0),
    }


# ---------------------------------------------------------------------------
# Dataset-agnostic helpers
# ---------------------------------------------------------------------------


def dataset_per_location_error(dataset: str, cache: Mapping[str, Any], expert_idx: Sequence[int], std: torch.Tensor) -> torch.Tensor:
    if dataset == "ETTh1":
        return etth1.per_location_abs_error_for_indices(cache, std, expert_idx)
    if dataset == "ETTh2":
        return etth2.per_location_error(cache, expert_idx, std)
    raise ValueError(dataset)


def dataset_forecasts(cache: Mapping[str, Any], expert_idx: Sequence[int]) -> torch.Tensor:
    return cache["prediction_stack"][..., list(expert_idx)].to(torch.float32)


def build_train_dual_memory(
    dataset: str, train_cache: Mapping[str, Any], expert_idx: Sequence[int], std: torch.Tensor, short_decay: float, long_decay: float
) -> dict[str, Any]:
    starts = train_cache["absolute_window_starts"].to(torch.long)
    horizon = int(train_cache["forecast_horizon"])
    err_hve = dataset_per_location_error(dataset, train_cache, expert_idx, std)
    err_e = err_hve.mean(dim=(1, 2))
    chrono_short, chrono_long, chrono_updates, chrono_audit = build_causal_dual_memory(starts, err_e, horizon, short_decay, long_decay, err_e[0])
    hv_short, hv_long, hv_updates, hv_audit = build_causal_dual_memory(starts, err_hve, horizon, short_decay, long_decay, err_hve[0])
    return {
        "chrono_short": chrono_short,
        "chrono_long": chrono_long,
        "chrono_updates": chrono_updates,
        "hv_short": hv_short,
        "hv_long": hv_long,
        "hv_updates": hv_updates,
        "chrono_audit": chrono_audit,
        "hv_audit": hv_audit,
        "short_decay": short_decay,
        "long_decay": long_decay,
        "chrono_short_recent_minus_long_persistent_l1": float((chrono_short - chrono_long).abs().mean()),
        "hv_short_recent_minus_long_persistent_l1": float((hv_short - hv_long).abs().mean()),
    }


def dual_base_prediction(
    dataset: str,
    cache: Mapping[str, Any],
    expert_idx: Sequence[int],
    std: torch.Tensor,
    memory: Mapping[str, Any],
    short_decay: float,
    long_decay: float,
    chrono_mix: float,
    hv_schedule_fn: Callable[[int], torch.Tensor],
    online: bool,
) -> tuple[torch.Tensor, dict[str, Any]]:
    n = int(cache["num_windows"])
    horizon = int(cache["forecast_horizon"])
    forecasts = dataset_forecasts(cache, expert_idx)
    static_weights = torch.full((n, 3), 1.0 / 3.0)
    err_hve = dataset_per_location_error(dataset, cache, expert_idx, std)
    err_e = err_hve.mean(dim=(1, 2))
    schedule = hv_schedule_fn(horizon)

    if online:
        starts = cache["absolute_window_starts"].to(torch.long)

        def chrono_wf(s: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
            return softmax_neg(chrono_mix * s + (1.0 - chrono_mix) * l, CHRONO_TEMPERATURE)

        online_w, chrono_updates, chrono_audit = causal_dual_weight_sequence(
            starts, err_e, horizon, short_decay, long_decay, memory["chrono_short"], memory["chrono_long"], chrono_wf
        )

        def hv_wf(s: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
            mixed = schedule.view(-1, 1, 1) * s + (1.0 - schedule.view(-1, 1, 1)) * l
            return errors_to_weights(mixed, DUAL_HV_TRIAL)

        hv_w, hv_updates, hv_audit = causal_dual_weight_sequence(
            starts, err_hve, horizon, short_decay, long_decay, memory["hv_short"], memory["hv_long"], hv_wf
        )
        memory_source = "router_train_then_online"
    else:
        mixed_chrono = chrono_mix * memory["chrono_short"] + (1.0 - chrono_mix) * memory["chrono_long"]
        online_w = softmax_neg(mixed_chrono, CHRONO_TEMPERATURE).view(1, 3).expand(n, -1).clone()
        mixed_hv = schedule.view(-1, 1, 1) * memory["hv_short"] + (1.0 - schedule.view(-1, 1, 1)) * memory["hv_long"]
        hv_w = errors_to_weights(mixed_hv, DUAL_HV_TRIAL).unsqueeze(0).expand(n, -1, -1, -1).clone()
        chrono_updates, hv_updates, chrono_audit, hv_audit = 0, 0, [], []
        memory_source = "router_train_only"

    chrono_w = 0.5 * static_weights + 0.5 * online_w
    chrono_w = chrono_w / chrono_w.sum(dim=1, keepdim=True).clamp_min(1e-8)
    chrono_pred = weighted_forecast(forecasts, chrono_w)
    hv_pred = predict_from_hv_weights(forecasts, hv_w)
    pred = 0.25 * chrono_pred + 0.75 * hv_pred
    names = [list(cache["expert_names"])[i] for i in expert_idx]
    return pred, {
        "chrono_num_updates": chrono_updates,
        "hv_num_updates": hv_updates,
        "short_eval_updates": chrono_updates + hv_updates,
        "long_eval_updates": chrono_updates + hv_updates,
        "memory_source": memory_source,
        "causality_audit_rows": len(chrono_audit) + len(hv_audit),
        "_chrono_audit": chrono_audit,
        "_hv_audit": hv_audit,
        **{f"mean_weight_{names[i]}": float(hv_w[..., i].mean()) for i in range(3)},
    }


def layer_specialists(
    dataset: str,
    cache: Mapping[str, Any],
    train_cache: Mapping[str, Any],
    std: torch.Tensor,
    expert_idx: Sequence[int],
    base: torch.Tensor,
    train_base: torch.Tensor,
    online: bool,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Layer the unchanged specialist logic (DLinear/ModernTCN advantage weighting)
    on top of a base prediction. The specialist EMA is independent of the dual
    short/long branch memory, so any base prediction (single-memory, dual-memory,
    short-only, long-only, ...) can be passed through unmodified."""
    if dataset == "ETTh1":
        target_train = train_cache["targets"].to(torch.float32)
        mask_train = train_cache["target_masks"].to(torch.bool)
        d_train = etth1.optional_prediction(train_cache, "DLinear")
        m_train = etth1.optional_prediction(train_cache, "ModernTCN")
        init_base_err = etth1.normalized_abs_error(train_base, target_train, mask_train, std)
        init_d_err = etth1.normalized_abs_error(d_train, target_train, mask_train, std)
        init_m_err = etth1.normalized_abs_error(m_train, target_train, mask_train, std)
        d_pred = etth1.optional_prediction(cache, "DLinear")
        m_pred = etth1.optional_prediction(cache, "ModernTCN")
        if online:
            pred, extra, _ = run_causal_specialists(
                cache["absolute_window_starts"].to(torch.long),
                base,
                d_pred,
                m_pred,
                cache["targets"].to(torch.float32),
                cache["target_masks"].to(torch.bool),
                std,
                etth1.SPECIALIST_CONFIG,
                init_base_err,
                init_d_err,
                init_m_err,
            )
        else:
            pred, extra = frozen_etth1_specialists(base, d_pred, m_pred, init_base_err, init_d_err, init_m_err)
    elif dataset == "ETTh2":
        selected_core = {list(cache["expert_names"])[i] for i in expert_idx}
        target_train = train_cache["targets"].to(torch.float32)
        mask_train = train_cache["target_masks"].to(torch.bool)
        d_train = etth2.expert_prediction(train_cache, "DLinear")
        m_train = etth2.expert_prediction(train_cache, "ModernTCN")
        init_base_err = etth2.abs_error(train_base, target_train, mask_train, std)
        init_d_err = etth2.abs_error(d_train, target_train, mask_train, std)
        init_m_err = etth2.abs_error(m_train, target_train, mask_train, std)
        d_pred = etth2.expert_prediction(cache, "DLinear")
        m_pred = etth2.expert_prediction(cache, "ModernTCN")
        if online:
            pred, extra = etth2.run_specialists_no_duplicate(
                cache["absolute_window_starts"].to(torch.long),
                base,
                d_pred,
                m_pred,
                cache["targets"].to(torch.float32),
                cache["target_masks"].to(torch.bool),
                std,
                init_base_err,
                init_d_err,
                init_m_err,
                selected_core,
            )
        else:
            pred, extra = frozen_etth2_specialists(base, d_pred, m_pred, init_base_err, init_d_err, init_m_err, selected_core)
    else:
        raise ValueError(dataset)
    return pred, extra


def dual_costar_prediction(
    dataset: str,
    cache: Mapping[str, Any],
    train_cache: Mapping[str, Any],
    std: torch.Tensor,
    expert_idx: Sequence[int],
    memory: Mapping[str, Any],
    short_decay: float,
    long_decay: float,
    chrono_mix: float,
    hv_schedule_fn: Callable[[int], torch.Tensor],
    online: bool,
) -> tuple[torch.Tensor, dict[str, Any]]:
    base, base_extra = dual_base_prediction(dataset, cache, expert_idx, std, memory, short_decay, long_decay, chrono_mix, hv_schedule_fn, online)
    train_base, _ = dual_base_prediction(dataset, train_cache, expert_idx, std, memory, short_decay, long_decay, chrono_mix, hv_schedule_fn, online=False)
    pred, extra = layer_specialists(dataset, cache, train_cache, std, expert_idx, base, train_base, online)
    return pred, {**base_extra, **extra}


# ---------------------------------------------------------------------------
# Router-train fold selection (short_decay, long_decay, mixing strategy)
# ---------------------------------------------------------------------------


def fold_base_mae(
    dataset: str,
    train_cache: Mapping[str, Any],
    expert_idx: Sequence[int],
    std: torch.Tensor,
    folds: Sequence[tuple[int, int, int]],
    short_decay: float,
    long_decay: float,
    chrono_mix: float,
    hv_schedule_fn: Callable[[int], torch.Tensor],
) -> tuple[float, list[float]]:
    starts = train_cache["absolute_window_starts"].to(torch.long)
    horizon = int(train_cache["forecast_horizon"])
    forecasts_all = dataset_forecasts(train_cache, expert_idx)
    target = train_cache["targets"].to(torch.float32)
    mask = train_cache["target_masks"].to(torch.bool)
    err_hve_all = dataset_per_location_error(dataset, train_cache, expert_idx, std)
    err_e_all = err_hve_all.mean(dim=(1, 2))
    schedule = hv_schedule_fn(horizon)
    fold_maes = []
    for train_lo, eval_lo, eval_hi in folds:
        seed_chrono = err_e_all[train_lo:eval_lo].mean(dim=0) if eval_lo > train_lo else err_e_all[0]
        seed_hv = err_hve_all[train_lo:eval_lo].mean(dim=0) if eval_lo > train_lo else err_hve_all[0]

        def chrono_wf(s: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
            return softmax_neg(chrono_mix * s + (1.0 - chrono_mix) * l, CHRONO_TEMPERATURE)

        w_chrono, _, _ = causal_dual_weight_sequence(
            starts[eval_lo:eval_hi], err_e_all[eval_lo:eval_hi], horizon, short_decay, long_decay, seed_chrono, seed_chrono, chrono_wf
        )

        def hv_wf(s: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
            mixed = schedule.view(-1, 1, 1) * s + (1.0 - schedule.view(-1, 1, 1)) * l
            return errors_to_weights(mixed, DUAL_HV_TRIAL)

        w_hv, _, _ = causal_dual_weight_sequence(
            starts[eval_lo:eval_hi], err_hve_all[eval_lo:eval_hi], horizon, short_decay, long_decay, seed_hv, seed_hv, hv_wf
        )
        static_w = torch.full((eval_hi - eval_lo, 3), 1.0 / 3.0)
        chrono_w = 0.5 * static_w + 0.5 * w_chrono
        chrono_w = chrono_w / chrono_w.sum(dim=1, keepdim=True).clamp_min(1e-8)
        chrono_pred = weighted_forecast(forecasts_all[eval_lo:eval_hi], chrono_w)
        hv_pred = predict_from_hv_weights(forecasts_all[eval_lo:eval_hi], w_hv)
        pred = 0.25 * chrono_pred + 0.75 * hv_pred
        mae = sample_mae(pred, target[eval_lo:eval_hi], mask[eval_lo:eval_hi], std)
        fold_maes.append(mae)
    cat = torch.cat(fold_maes)
    return float(cat.mean()), [float(m.mean()) for m in fold_maes]


def select_decays_and_mixing(
    dataset: str, train_cache: Mapping[str, Any], expert_idx: Sequence[int], std: torch.Tensor
) -> dict[str, Any]:
    folds = train_folds(int(train_cache["num_windows"]))

    stage_a_rows = []
    for s in SHORT_DECAY_GRID:
        for l in LONG_DECAY_GRID:
            if not (s < l):
                continue
            mae, fold_list = fold_base_mae(dataset, train_cache, expert_idx, std, folds, s, l, 0.5, constant_schedule(0.5))
            stage_a_rows.append({"short_decay": s, "long_decay": l, "fold_mae_mean": mae, "fold_maes": fold_list})
    stage_a_rows.sort(key=lambda r: (r["fold_mae_mean"], -r["short_decay"], r["long_decay"]))
    best_a = stage_a_rows[0]
    sel_short, sel_long = best_a["short_decay"], best_a["long_decay"]

    stage_b_chrono_rows = []
    for w in MIX_GRID:
        mae, fold_list = fold_base_mae(dataset, train_cache, expert_idx, std, folds, sel_short, sel_long, w, constant_schedule(0.5))
        stage_b_chrono_rows.append({"chrono_mix": w, "fold_mae_mean": mae, "fold_maes": fold_list})
    stage_b_chrono_rows.sort(key=lambda r: r["fold_mae_mean"])
    sel_chrono_mix = stage_b_chrono_rows[0]["chrono_mix"]

    stage_b_hv_rows = []
    for w in MIX_GRID:
        mae, fold_list = fold_base_mae(dataset, train_cache, expert_idx, std, folds, sel_short, sel_long, sel_chrono_mix, constant_schedule(w))
        stage_b_hv_rows.append({"hv_mix": w, "fold_mae_mean": mae, "fold_maes": fold_list})
    stage_b_hv_rows.sort(key=lambda r: r["fold_mae_mean"])
    sel_hv_mix_fixed = stage_b_hv_rows[0]["hv_mix"]

    stage_c_rows = []
    for start, end in SCHEDULE_GRID:
        mae, fold_list = fold_base_mae(
            dataset, train_cache, expert_idx, std, folds, sel_short, sel_long, sel_chrono_mix, linspace_schedule(start, end)
        )
        stage_c_rows.append({"schedule_start": start, "schedule_end": end, "fold_mae_mean": mae, "fold_maes": fold_list})
    stage_c_rows.sort(key=lambda r: r["fold_mae_mean"])
    sel_schedule = (stage_c_rows[0]["schedule_start"], stage_c_rows[0]["schedule_end"])

    return {
        "dataset": dataset,
        "folds": folds,
        "stage_a_decay_grid": stage_a_rows,
        "stage_b_chrono_mix_grid": stage_b_chrono_rows,
        "stage_b_hv_mix_grid": stage_b_hv_rows,
        "stage_c_schedule_grid": stage_c_rows,
        "selected_short_decay": sel_short,
        "selected_long_decay": sel_long,
        "selected_chrono_mix": sel_chrono_mix,
        "selected_hv_mix_fixed": sel_hv_mix_fixed,
        "selected_schedule_start": sel_schedule[0],
        "selected_schedule_end": sel_schedule[1],
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def per_horizon_mae(cache: Mapping[str, Any], std: torch.Tensor, pred: torch.Tensor) -> list[dict[str, Any]]:
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.float32)
    stdv = std.view(1, 1, -1)
    abs_err = ((pred - target) / stdv).abs() * mask
    rows = []
    for h in range(pred.shape[1]):
        denom = mask[:, h].sum().clamp_min(1)
        rows.append({"horizon_index": h, "mae": float(abs_err[:, h].sum() / denom)})
    return rows


def variant_configs(selection: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        "short_only": {"chrono_mix": 1.0, "hv_schedule_fn": constant_schedule(1.0)},
        "long_only": {"chrono_mix": 0.0, "hv_schedule_fn": constant_schedule(0.0)},
        "fifty_fifty": {"chrono_mix": 0.5, "hv_schedule_fn": constant_schedule(0.5)},
        "train_selected_fixed": {
            "chrono_mix": selection["selected_chrono_mix"],
            "hv_schedule_fn": constant_schedule(selection["selected_hv_mix_fixed"]),
        },
        "dual_memory_horizon_schedule": {
            "chrono_mix": selection["selected_chrono_mix"],
            "hv_schedule_fn": linspace_schedule(selection["selected_schedule_start"], selection["selected_schedule_end"]),
        },
    }


def causality_check(audit_rows: Sequence[Mapping[str, Any]], horizon: int) -> dict[str, Any]:
    violations = [r for r in audit_rows if not (int(r["forecast_start"]) + horizon <= int(r["current_start"]))]
    return {"rows_checked": len(audit_rows), "violations": len(violations), "result": "PASS" if not violations else "FAIL"}


# ---------------------------------------------------------------------------
# Short-vs-long weight disagreement analysis (router_val only)
# ---------------------------------------------------------------------------


def pearson_corr_axis0(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-expert correlation between two [window, expert] weight series, taken across windows."""
    ac = a - a.mean(dim=0, keepdim=True)
    bc = b - b.mean(dim=0, keepdim=True)
    num = (ac * bc).sum(dim=0)
    den = (ac.pow(2).sum(dim=0).sqrt() * bc.pow(2).sum(dim=0).sqrt()).clamp_min(1e-8)
    return num / den


def pearson_corr_flat(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.flatten() - a.mean()
    bf = b.flatten() - b.mean()
    den = (af.pow(2).sum().sqrt() * bf.pow(2).sum().sqrt()).clamp_min(1e-8)
    return float((af * bf).sum() / den)


def pearson_corr_axis1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-window correlation between short/long weight vectors across the expert axis.

    With only 3 experts each per-window correlation is a noisy 3-point statistic;
    it is reported per-spec alongside the more stable across-window correlations
    from `pearson_corr_axis0`/`pearson_corr_flat`.
    """
    ac = a - a.mean(dim=-1, keepdim=True)
    bc = b - b.mean(dim=-1, keepdim=True)
    num = (ac * bc).sum(dim=-1)
    den = (ac.pow(2).sum(dim=-1).sqrt() * bc.pow(2).sum(dim=-1).sqrt()).clamp_min(1e-8)
    return num / den


def disagreement_bucket_mae(l1: torch.Tensor, pred_mae: torch.Tensor, quantile: float = 0.25) -> dict[str, Any]:
    n = l1.numel()
    k = max(1, int(round(n * quantile)))
    order = torch.argsort(l1)
    low_idx = order[:k]
    high_idx = order[-k:]
    return {
        "high_disagreement_count": int(high_idx.numel()),
        "high_disagreement_mae": float(pred_mae[high_idx].mean()),
        "low_disagreement_count": int(low_idx.numel()),
        "low_disagreement_mae": float(pred_mae[low_idx].mean()),
        "all_windows_mae": float(pred_mae.mean()),
        "high_minus_low_mae": float(pred_mae[high_idx].mean() - pred_mae[low_idx].mean()),
        "quantile": quantile,
    }


def weight_disagreement_analysis(
    dataset: str,
    val_cache: Mapping[str, Any],
    std: torch.Tensor,
    expert_idx: Sequence[int],
    memory: Mapping[str, Any],
    selection: Mapping[str, Any],
    dual_online_pred: torch.Tensor,
    original_online_pred: torch.Tensor,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """For every router_val window, compare the short-only and long-only readings
    of the same dual memory: L1 distance, top-expert agreement, correlation, and
    MAE on the highest-disagreement windows. Both branches are covered; the
    horizon-variable branch's per-(H,V) weights are mean-pooled to one vector
    per window so it is directly comparable to the chronological branch."""
    starts = val_cache["absolute_window_starts"].to(torch.long)
    horizon = int(val_cache["forecast_horizon"])
    names = [list(val_cache["expert_names"])[i] for i in expert_idx]
    short_decay, long_decay = selection["selected_short_decay"], selection["selected_long_decay"]

    err_hve = dataset_per_location_error(dataset, val_cache, expert_idx, std)
    err_e = err_hve.mean(dim=(1, 2))

    chrono_short_states, chrono_long_states = causal_dual_state_walk(
        starts, err_e, horizon, short_decay, long_decay, memory["chrono_short"], memory["chrono_long"]
    )
    w_short_chrono = softmax_neg(chrono_short_states, CHRONO_TEMPERATURE)
    w_long_chrono = softmax_neg(chrono_long_states, CHRONO_TEMPERATURE)

    hv_short_states, hv_long_states = causal_dual_state_walk(
        starts, err_hve, horizon, short_decay, long_decay, memory["hv_short"], memory["hv_long"]
    )
    w_short_hv = torch.stack([errors_to_weights(hv_short_states[i], DUAL_HV_TRIAL).mean(dim=(0, 1)) for i in range(hv_short_states.shape[0])])
    w_long_hv = torch.stack([errors_to_weights(hv_long_states[i], DUAL_HV_TRIAL).mean(dim=(0, 1)) for i in range(hv_long_states.shape[0])])

    chrono_l1 = (w_short_chrono - w_long_chrono).abs().sum(dim=-1)
    hv_l1 = (w_short_hv - w_long_hv).abs().sum(dim=-1)
    chrono_top_short = w_short_chrono.argmax(dim=-1)
    chrono_top_long = w_long_chrono.argmax(dim=-1)
    hv_top_short = w_short_hv.argmax(dim=-1)
    hv_top_long = w_long_hv.argmax(dim=-1)
    chrono_top_mismatch = chrono_top_short != chrono_top_long
    hv_top_mismatch = hv_top_short != hv_top_long
    chrono_window_corr = pearson_corr_axis1(w_short_chrono, w_long_chrono)
    hv_window_corr = pearson_corr_axis1(w_short_hv, w_long_hv)

    target = val_cache["targets"].to(torch.float32)
    mask = val_cache["target_masks"].to(torch.bool)
    dual_pred_mae = sample_mae(dual_online_pred, target, mask, std)
    original_pred_mae = sample_mae(original_online_pred, target, mask, std)

    per_window_rows = [
        {
            "dataset": dataset,
            "window_index": i,
            "absolute_window_start": int(starts[i]),
            "chrono_l1_distance": float(chrono_l1[i]),
            "chrono_top_expert_short": names[int(chrono_top_short[i])],
            "chrono_top_expert_long": names[int(chrono_top_long[i])],
            "chrono_top_expert_mismatch": bool(chrono_top_mismatch[i]),
            "chrono_per_window_correlation": float(chrono_window_corr[i]),
            "hv_l1_distance_mean_over_hv": float(hv_l1[i]),
            "hv_top_expert_short": names[int(hv_top_short[i])],
            "hv_top_expert_long": names[int(hv_top_long[i])],
            "hv_top_expert_mismatch": bool(hv_top_mismatch[i]),
            "hv_per_window_correlation": float(hv_window_corr[i]),
            "dual_memory_online_mae": float(dual_pred_mae[i]),
            "original_single_memory_online_mae": float(original_pred_mae[i]),
        }
        for i in range(err_e.shape[0])
    ]

    summary = {
        "dataset": dataset,
        "num_windows": int(err_e.shape[0]),
        "chrono_mean_l1_distance": float(chrono_l1.mean()),
        "chrono_top_expert_mismatch_rate": float(chrono_top_mismatch.to(torch.float32).mean()),
        "chrono_per_expert_correlation": {names[e]: float(pearson_corr_axis0(w_short_chrono, w_long_chrono)[e]) for e in range(len(names))},
        "chrono_overall_correlation": pearson_corr_flat(w_short_chrono, w_long_chrono),
        "chrono_mean_per_window_correlation": float(chrono_window_corr.mean()),
        "hv_mean_l1_distance": float(hv_l1.mean()),
        "hv_top_expert_mismatch_rate": float(hv_top_mismatch.to(torch.float32).mean()),
        "hv_per_expert_correlation": {names[e]: float(pearson_corr_axis0(w_short_hv, w_long_hv)[e]) for e in range(len(names))},
        "hv_overall_correlation": pearson_corr_flat(w_short_hv, w_long_hv),
        "hv_mean_per_window_correlation": float(hv_window_corr.mean()),
        "mae_by_disagreement_split_by_chrono_l1": {
            "dual_memory_online": disagreement_bucket_mae(chrono_l1, dual_pred_mae),
            "original_single_memory_online": disagreement_bucket_mae(chrono_l1, original_pred_mae),
        },
        "mae_by_disagreement_split_by_hv_l1": {
            "dual_memory_online": disagreement_bucket_mae(hv_l1, dual_pred_mae),
            "original_single_memory_online": disagreement_bucket_mae(hv_l1, original_pred_mae),
        },
    }
    return summary, per_window_rows


# ---------------------------------------------------------------------------
# Short-vs-long winner analysis (router_val only, diagnostic -- no tuning)
# ---------------------------------------------------------------------------


def short_vs_long_winner_analysis(
    dataset: str,
    val_cache: Mapping[str, Any],
    train_cache: Mapping[str, Any],
    std: torch.Tensor,
    expert_idx: Sequence[int],
    memory: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """For every router_val window, build the full online dual-memory prediction
    reading purely through the short EMA and purely through the long EMA, then
    ask which one produced the lower-error forecast. Both readings are derived
    from the same `causal_dual_state_walk` call per branch (one walk each for
    the chronological and horizon-variable branches), so the short and long
    predictions can never drift out of causal alignment. This is diagnostic
    only: nothing is selected or tuned on router_val here.
    """
    starts = val_cache["absolute_window_starts"].to(torch.long)
    horizon = int(val_cache["forecast_horizon"])
    n = int(val_cache["num_windows"])
    forecasts = dataset_forecasts(val_cache, expert_idx)
    short_decay, long_decay = selection["selected_short_decay"], selection["selected_long_decay"]

    err_hve = dataset_per_location_error(dataset, val_cache, expert_idx, std)
    err_e = err_hve.mean(dim=(1, 2))

    chrono_short_states, chrono_long_states = causal_dual_state_walk(
        starts, err_e, horizon, short_decay, long_decay, memory["chrono_short"], memory["chrono_long"]
    )
    hv_short_states, hv_long_states = causal_dual_state_walk(
        starts, err_hve, horizon, short_decay, long_decay, memory["hv_short"], memory["hv_long"]
    )

    w_short_chrono = softmax_neg(chrono_short_states, CHRONO_TEMPERATURE)
    w_long_chrono = softmax_neg(chrono_long_states, CHRONO_TEMPERATURE)
    w_short_hv = torch.stack([errors_to_weights(hv_short_states[i], DUAL_HV_TRIAL) for i in range(n)])
    w_long_hv = torch.stack([errors_to_weights(hv_long_states[i], DUAL_HV_TRIAL) for i in range(n)])
    w_short_hv_pooled = w_short_hv.mean(dim=(1, 2))
    w_long_hv_pooled = w_long_hv.mean(dim=(1, 2))

    chrono_l1 = (w_short_chrono - w_long_chrono).abs().sum(dim=-1)
    hv_l1 = (w_short_hv_pooled - w_long_hv_pooled).abs().sum(dim=-1)
    chrono_top_mismatch = w_short_chrono.argmax(dim=-1) != w_long_chrono.argmax(dim=-1)
    hv_top_mismatch = w_short_hv_pooled.argmax(dim=-1) != w_long_hv_pooled.argmax(dim=-1)

    static_weights = torch.full((n, 3), 1.0 / 3.0)

    def base_from(w_chrono: torch.Tensor, w_hv: torch.Tensor) -> torch.Tensor:
        chrono_w = 0.5 * static_weights + 0.5 * w_chrono
        chrono_w = chrono_w / chrono_w.sum(dim=1, keepdim=True).clamp_min(1e-8)
        chrono_pred = weighted_forecast(forecasts, chrono_w)
        hv_pred = predict_from_hv_weights(forecasts, w_hv)
        return 0.25 * chrono_pred + 0.75 * hv_pred

    base_short = base_from(w_short_chrono, w_short_hv)
    base_long = base_from(w_long_chrono, w_long_hv)

    # Frozen router-train-only base, reused as the specialist init reference,
    # exactly as the existing short_only/long_only variants already do.
    train_base_short, _ = dual_base_prediction(
        dataset, train_cache, expert_idx, std, memory, short_decay, long_decay, 1.0, constant_schedule(1.0), online=False
    )
    train_base_long, _ = dual_base_prediction(
        dataset, train_cache, expert_idx, std, memory, short_decay, long_decay, 0.0, constant_schedule(0.0), online=False
    )
    pred_short, _ = layer_specialists(dataset, val_cache, train_cache, std, expert_idx, base_short, train_base_short, online=True)
    pred_long, _ = layer_specialists(dataset, val_cache, train_cache, std, expert_idx, base_long, train_base_long, online=True)

    target = val_cache["targets"].to(torch.float32)
    mask = val_cache["target_masks"].to(torch.bool)
    short_mae = sample_mae(pred_short, target, mask, std)
    long_mae = sample_mae(pred_long, target, mask, std)
    delta = short_mae - long_mae
    eps = 1e-9
    winner = ["short" if float(d) < -eps else ("long" if float(d) > eps else "tie") for d in delta]

    per_window_rows = [
        {
            "dataset": dataset,
            "window_index": i,
            "absolute_window_start": int(starts[i]),
            "short_only_mae": float(short_mae[i]),
            "long_only_mae": float(long_mae[i]),
            "short_minus_long_mae": float(delta[i]),
            "winner": winner[i],
            "chrono_l1_distance": float(chrono_l1[i]),
            "chrono_top_expert_mismatch": bool(chrono_top_mismatch[i]),
            "hv_l1_distance": float(hv_l1[i]),
            "hv_top_expert_mismatch": bool(hv_top_mismatch[i]),
        }
        for i in range(n)
    ]

    def rate_stats(sel: torch.Tensor) -> dict[str, Any]:
        idx = sel.nonzero(as_tuple=True)[0]
        count = int(idx.numel())
        if count == 0:
            return {"count": 0, "short_win_rate": None, "long_win_rate": None, "tie_rate": None, "mean_short_minus_long_mae": None}
        d = delta[idx]
        short_wins = int((d < -eps).sum())
        long_wins = int((d > eps).sum())
        ties = count - short_wins - long_wins
        return {
            "count": count,
            "short_win_rate": short_wins / count,
            "long_win_rate": long_wins / count,
            "tie_rate": ties / count,
            "mean_short_minus_long_mae": float(d.mean()),
        }

    def top_quartile_mask(l1: torch.Tensor, quantile: float = 0.25) -> torch.Tensor:
        k = max(1, int(round(n * quantile)))
        order = torch.argsort(l1, descending=True)
        sel = torch.zeros(n, dtype=torch.bool)
        sel[order[:k]] = True
        return sel

    overall = rate_stats(torch.ones(n, dtype=torch.bool))
    overall["mean_mae_short_only"] = float(short_mae.mean())
    overall["mean_mae_long_only"] = float(long_mae.mean())

    short_win_idx = (delta < -eps).nonzero(as_tuple=True)[0]
    long_win_idx = (delta > eps).nonzero(as_tuple=True)[0]

    summary = {
        "dataset": dataset,
        "num_windows": n,
        "overall": overall,
        "chrono_top_expert_mismatch_windows": rate_stats(chrono_top_mismatch),
        "chrono_top_quartile_l1_windows": rate_stats(top_quartile_mask(chrono_l1)),
        "hv_top_expert_mismatch_windows": rate_stats(hv_top_mismatch),
        "hv_top_quartile_l1_windows": rate_stats(top_quartile_mask(hv_l1)),
        "avg_margin_when_short_wins": float((-delta[short_win_idx]).mean()) if short_win_idx.numel() else None,
        "avg_margin_when_long_wins": float(delta[long_win_idx].mean()) if long_win_idx.numel() else None,
    }
    return summary, per_window_rows


def evaluate_dataset(
    dataset: str,
    train_cache: Mapping[str, Any],
    val_cache: Mapping[str, Any],
    std: torch.Tensor,
    core: Sequence[str],
    device: torch.device,
) -> dict[str, Any]:
    expert_idx = etth1.expert_indices(val_cache, core) if dataset == "ETTh1" else etth2.expert_indices(val_cache, core)
    horizon = int(val_cache["forecast_horizon"])

    selection = select_decays_and_mixing(dataset, train_cache, expert_idx, std)
    memory = build_train_dual_memory(dataset, train_cache, expert_idx, std, selection["selected_short_decay"], selection["selected_long_decay"])

    # --- original single-memory COSTAR (reused unmodified) ---
    orig_frozen_pred, orig_frozen_extra = original_frozen_prediction(dataset, val_cache, train_cache, std, expert_idx, 7, device)
    orig_online_pred, orig_online_extra = original_online_prediction(dataset, val_cache, train_cache, std, expert_idx, 7, device)
    orig_frozen_metrics = metric_values(val_cache, orig_frozen_pred, std)
    orig_online_metrics = metric_values(val_cache, orig_online_pred, std)

    variants = variant_configs(selection)
    result_rows: list[dict[str, Any]] = []
    per_horizon_rows: list[dict[str, Any]] = []
    causality_rows: list[dict[str, Any]] = []
    headline: dict[str, dict[str, Any]] = {}

    def add_row(method: str, mixing: str, online: bool, pred: torch.Tensor, extra: Mapping[str, Any], baseline_metrics: Mapping[str, float], baseline_pred: torch.Tensor) -> None:
        m = metric_values(val_cache, pred, std)
        boot = paired_block_bootstrap(
            sample_mae(pred, val_cache["targets"].to(torch.float32), val_cache["target_masks"].to(torch.bool), std),
            sample_mae(baseline_pred, val_cache["targets"].to(torch.float32), val_cache["target_masks"].to(torch.bool), std),
        )
        row = {
            "dataset": dataset,
            "method": method,
            "mixing_strategy": mixing,
            "mode": "online" if online else "frozen",
            "mae": m["mae"],
            "mse": m["mse"],
            "delta_mae_vs_original": m["mae"] - baseline_metrics["mae"],
            "delta_mse_vs_original": m["mse"] - baseline_metrics["mse"],
            "selected_short_decay": selection["selected_short_decay"],
            "selected_long_decay": selection["selected_long_decay"],
            **{k: v for k, v in extra.items() if not k.startswith("_")},
            **{f"boot_{k}": v for k, v in boot.items()},
        }
        result_rows.append(row)
        per_horizon_rows.extend(
            [{"dataset": dataset, "method": method, "mode": row["mode"], **r} for r in per_horizon_mae(val_cache, std, pred)]
        )

    result_rows.append(
        {"dataset": dataset, "method": "original_single_memory", "mixing_strategy": "single_ema", "mode": "frozen", **orig_frozen_metrics, "delta_mae_vs_original": 0.0, "delta_mse_vs_original": 0.0}
    )
    result_rows.append(
        {"dataset": dataset, "method": "original_single_memory", "mixing_strategy": "single_ema", "mode": "online", **orig_online_metrics, "delta_mae_vs_original": 0.0, "delta_mse_vs_original": 0.0}
    )
    per_horizon_rows.extend([{"dataset": dataset, "method": "original_single_memory", "mode": "frozen", **r} for r in per_horizon_mae(val_cache, std, orig_frozen_pred)])
    per_horizon_rows.extend([{"dataset": dataset, "method": "original_single_memory", "mode": "online", **r} for r in per_horizon_mae(val_cache, std, orig_online_pred)])

    for mixing, cfg in variants.items():
        for online in (False, True):
            pred, extra = dual_costar_prediction(
                dataset, val_cache, train_cache, std, expert_idx, memory,
                selection["selected_short_decay"], selection["selected_long_decay"],
                cfg["chrono_mix"], cfg["hv_schedule_fn"], online=online,
            )
            baseline_metrics = orig_online_metrics if online else orig_frozen_metrics
            baseline_pred = orig_online_pred if online else orig_frozen_pred
            method = "dual_memory" if mixing == "dual_memory_horizon_schedule" else f"dual_memory_{mixing}"
            add_row(method, mixing, online, pred, extra, baseline_metrics, baseline_pred)
            if mixing == "dual_memory_horizon_schedule":
                headline[("online" if online else "frozen")] = {"pred": pred, "extra": extra}
                audit = list(extra.get("_chrono_audit", [])) + list(extra.get("_hv_audit", []))
                causality_rows.append({"dataset": dataset, "mode": "online" if online else "frozen", **causality_check(audit, horizon)})

    # --- invariant checks: frozen predictions must not react to val targets/masks ---
    before = tensor_digest(val_cache)
    frozen_pred_again, _ = dual_costar_prediction(
        dataset, val_cache, train_cache, std, expert_idx, memory,
        selection["selected_short_decay"], selection["selected_long_decay"],
        variants["dual_memory_horizon_schedule"]["chrono_mix"], variants["dual_memory_horizon_schedule"]["hv_schedule_fn"], online=False,
    )
    after = tensor_digest(val_cache)
    if before != after:
        raise AssertionError(f"{dataset} dual frozen prediction mutated validation cache tensors")
    target_mut = cloned_with_random_targets(val_cache, randomize_masks=False)
    mask_mut = cloned_with_random_targets(val_cache, randomize_masks=True)
    frozen_pred_target_mut, _ = dual_costar_prediction(
        dataset, target_mut, train_cache, std, expert_idx, memory,
        selection["selected_short_decay"], selection["selected_long_decay"],
        variants["dual_memory_horizon_schedule"]["chrono_mix"], variants["dual_memory_horizon_schedule"]["hv_schedule_fn"], online=False,
    )
    frozen_pred_mask_mut, _ = dual_costar_prediction(
        dataset, mask_mut, train_cache, std, expert_idx, memory,
        selection["selected_short_decay"], selection["selected_long_decay"],
        variants["dual_memory_horizon_schedule"]["chrono_mix"], variants["dual_memory_horizon_schedule"]["hv_schedule_fn"], online=False,
    )
    online_pred_baseline, _ = dual_costar_prediction(
        dataset, val_cache, train_cache, std, expert_idx, memory,
        selection["selected_short_decay"], selection["selected_long_decay"],
        variants["dual_memory_horizon_schedule"]["chrono_mix"], variants["dual_memory_horizon_schedule"]["hv_schedule_fn"], online=True,
    )
    online_pred_target_mut, _ = dual_costar_prediction(
        dataset, target_mut, train_cache, std, expert_idx, memory,
        selection["selected_short_decay"], selection["selected_long_decay"],
        variants["dual_memory_horizon_schedule"]["chrono_mix"], variants["dual_memory_horizon_schedule"]["hv_schedule_fn"], online=True,
    )
    invariant = {
        "frozen_unchanged_after_target_randomization": bool(torch.equal(headline["frozen"]["pred"], frozen_pred_target_mut)),
        "frozen_unchanged_after_mask_randomization": bool(torch.equal(headline["frozen"]["pred"], frozen_pred_mask_mut)),
        "frozen_repeat_call_identical": bool(torch.equal(headline["frozen"]["pred"], frozen_pred_again)),
        "online_changed_after_target_randomization": bool(not torch.equal(online_pred_baseline, online_pred_target_mut)),
    }
    if not (invariant["frozen_unchanged_after_target_randomization"] and invariant["frozen_unchanged_after_mask_randomization"]):
        raise AssertionError(f"{dataset} dual-memory frozen predictions changed after validation target/mask replacement")
    if not invariant["online_changed_after_target_randomization"]:
        raise AssertionError(f"{dataset} dual-memory online predictions did not react to validation target replacement")

    disagreement_summary, disagreement_rows = weight_disagreement_analysis(
        dataset, val_cache, std, expert_idx, memory, selection, headline["online"]["pred"], orig_online_pred
    )
    short_vs_long_summary, short_vs_long_rows = short_vs_long_winner_analysis(
        dataset, val_cache, train_cache, std, expert_idx, memory, selection
    )

    return {
        "dataset": dataset,
        "expert_indices": list(expert_idx),
        "core": list(core),
        "selection": selection,
        "results": result_rows,
        "per_horizon": per_horizon_rows,
        "causality": causality_rows,
        "invariant_checks": invariant,
        "disagreement": disagreement_summary,
        "disagreement_per_window": disagreement_rows,
        "short_vs_long_winner": short_vs_long_summary,
        "short_vs_long_winner_per_window": short_vs_long_rows,
        "memory_summary": {
            "short_decay": memory["short_decay"],
            "long_decay": memory["long_decay"],
            "chrono_train_updates": memory["chrono_updates"],
            "hv_train_updates": memory["hv_updates"],
            "chrono_short_minus_long_l1": memory["chrono_short_recent_minus_long_persistent_l1"],
            "hv_short_minus_long_l1": memory["hv_short_recent_minus_long_persistent_l1"],
        },
    }


def make_report(out_dir: Path, report: Mapping[str, Any]) -> None:
    lines = [
        "# Dual-Timescale (Short/Long) Memory COSTAR",
        "",
        "Adds a second, slower EMA memory to both the global chronological branch and",
        "the horizon-variable branch. Both memories are built causally from `router_train`",
        "only, frozen before `router_val` for Frozen COSTAR, and updated causally during",
        "`router_val` (only after `old_start + horizon <= current_start`) for Online COSTAR.",
        "",
        "## Selection (router_train only)",
        "",
        "| Dataset | short_decay | long_decay | chrono mix (train-selected fixed) | hv mix (train-selected fixed) | hv schedule (start,end) |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for dataset, d in report["datasets"].items():
        sel = d["selection"]
        lines.append(
            f"| {dataset} | {sel['selected_short_decay']:g} | {sel['selected_long_decay']:g} | "
            f"{sel['selected_chrono_mix']:g} | {sel['selected_hv_mix_fixed']:g} | "
            f"({sel['selected_schedule_start']:g}, {sel['selected_schedule_end']:g}) |"
        )
    lines += ["", "## Validation results", ""]
    for dataset, d in report["datasets"].items():
        lines.append(f"### {dataset}")
        lines.append("")
        lines.append("| Method | Mode | MAE | MSE | Delta MAE vs original |")
        lines.append("|---|---|---:|---:|---:|")
        for row in d["results"]:
            if row["method"] in ("original_single_memory", "dual_memory", "dual_memory_short_only", "dual_memory_long_only", "dual_memory_fifty_fifty", "dual_memory_train_selected_fixed"):
                lines.append(f"| `{row['method']}` | {row['mode']} | `{row['mae']:.6f}` | `{row['mse']:.6f}` | `{row['delta_mae_vs_original']:+.6f}` |")
        lines.append("")
        lines.append(f"- Causality audit: {d['causality']}")
        lines.append(f"- Invariant checks: {d['invariant_checks']}")
        lines.append("")
    lines += ["## Short-vs-long weight disagreement (router_val, per window)", ""]
    lines.append(
        "Every router_val window is scored under the *same* dual-memory state read two ways: "
        "purely through the short EMA and purely through the long EMA. Disagreement between the "
        "two readings is measured by L1 distance, top-expert mismatch rate, correlation, and MAE "
        "on the highest-disagreement windows (top/bottom quartile by L1 distance)."
    )
    lines.append("")
    lines.append(
        "| Dataset | Branch | Mean L1 | Top-expert mismatch rate | Per-expert corr. (across windows) | Overall corr. | Mean per-window corr. (n=3, noisy) |"
    )
    lines.append("|---|---|---:|---:|---|---:|---:|")
    for dataset, d in report["datasets"].items():
        dis = d["disagreement"]
        for branch, prefix in (("chrono", "chrono"), ("horizon-variable (mean-pooled)", "hv")):
            per_expert = ", ".join(f"{k}={v:+.3f}" for k, v in dis[f"{prefix}_per_expert_correlation"].items())
            lines.append(
                f"| {dataset} | {branch} | `{dis[f'{prefix}_mean_l1_distance']:.4f}` | "
                f"`{dis[f'{prefix}_top_expert_mismatch_rate']:.4f}` | {per_expert} | "
                f"`{dis[f'{prefix}_overall_correlation']:+.4f}` | `{dis[f'{prefix}_mean_per_window_correlation']:+.4f}` |"
            )
    lines.append("")
    lines.append("MAE on high- vs low-disagreement windows (top/bottom quartile by chronological-branch L1 distance):")
    lines.append("")
    lines.append("| Dataset | Method | High-disagreement MAE | Low-disagreement MAE | All-windows MAE | High minus low |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for dataset, d in report["datasets"].items():
        split = d["disagreement"]["mae_by_disagreement_split_by_chrono_l1"]
        for method, key in (("dual_memory (online)", "dual_memory_online"), ("original_single_memory (online)", "original_single_memory_online")):
            b = split[key]
            lines.append(
                f"| {dataset} | {method} | `{b['high_disagreement_mae']:.6f}` | `{b['low_disagreement_mae']:.6f}` | "
                f"`{b['all_windows_mae']:.6f}` | `{b['high_minus_low_mae']:+.6f}` |"
            )
    lines.append("")
    lines += ["## Short vs long: which memory actually produces the better forecast?", ""]
    lines.append(
        "Diagnostic only, computed once on router_val with nothing tuned against it. For every "
        "window, the full online dual-memory pipeline (base branches + specialists) is built twice "
        "from the *same* `causal_dual_state_walk` per branch: once reading only the short EMA, once "
        "reading only the long EMA. `delta = short_only_mae - long_only_mae`; `delta < 0` means short "
        "memory won that window, `delta > 0` means long memory won."
    )
    lines.append("")

    def fmt_rate(v: Any) -> str:
        return f"{v:.3f}" if v is not None else "n/a"

    def fmt_mae(v: Any) -> str:
        return f"{v:+.6f}" if v is not None else "n/a"

    lines.append("| Dataset | Condition | Windows | Short win rate | Long win rate | Tie rate | Mean (short-long) MAE |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    condition_labels = [
        ("overall", "All router_val windows"),
        ("chrono_top_expert_mismatch_windows", "Chrono top-expert differs"),
        ("chrono_top_quartile_l1_windows", "Chrono top-25% L1 disagreement"),
        ("hv_top_expert_mismatch_windows", "HV (mean-pooled) top-expert differs"),
        ("hv_top_quartile_l1_windows", "HV (mean-pooled) top-25% L1 disagreement"),
    ]
    for dataset, d in report["datasets"].items():
        s = d["short_vs_long_winner"]
        for key, label in condition_labels:
            c = s[key]
            lines.append(
                f"| {dataset} | {label} | {c['count']} | {fmt_rate(c['short_win_rate'])} | "
                f"{fmt_rate(c['long_win_rate'])} | {fmt_rate(c['tie_rate'])} | {fmt_mae(c['mean_short_minus_long_mae'])} |"
            )
    lines.append("")
    lines.append("| Dataset | Mean MAE short-only | Mean MAE long-only | Avg margin when short wins | Avg margin when long wins |")
    lines.append("|---|---:|---:|---:|---:|")
    for dataset, d in report["datasets"].items():
        s = d["short_vs_long_winner"]
        lines.append(
            f"| {dataset} | `{s['overall']['mean_mae_short_only']:.6f}` | `{s['overall']['mean_mae_long_only']:.6f}` | "
            f"{fmt_mae(s['avg_margin_when_short_wins'])} | {fmt_mae(s['avg_margin_when_long_wins'])} |"
        )
    lines.append("")
    lines.append("### Does short-vs-long disagreement tell us which memory should be trusted?")
    lines.append("")
    for dataset, d in report["datasets"].items():
        s = d["short_vs_long_winner"]
        chrono_hi = s["chrono_top_quartile_l1_windows"]
        hv_hi = s["hv_top_quartile_l1_windows"]
        rates = [r["short_win_rate"] for r in (chrono_hi, hv_hi) if r["short_win_rate"] is not None]
        close_to_half = all(abs(r - 0.5) <= 0.10 for r in rates) if rates else False
        if close_to_half:
            verdict = (
                f"On the highest-disagreement windows, short wins {fmt_rate(chrono_hi['short_win_rate'])} of the time "
                f"(chrono split) and {fmt_rate(hv_hi['short_win_rate'])} of the time (HV split) -- close to 50/50. "
                "Disagreement alone does not identify which memory to trust here, so **dynamic mixing is not "
                "justified by this signal alone** for this dataset."
            )
        else:
            leader = "short" if (sum(rates) / len(rates)) > 0.5 else "long"
            verdict = (
                f"On the highest-disagreement windows, {leader} memory wins clearly and consistently "
                f"(chrono split short-win-rate={fmt_rate(chrono_hi['short_win_rate'])}, HV split short-win-rate={fmt_rate(hv_hi['short_win_rate'])}). "
                f"High short-vs-long disagreement is a **candidate signal for a future dynamic mixer** on {dataset}: "
                f"lean toward {leader}-only when disagreement is high."
            )
        lines.append(f"- **{dataset}**: {verdict}")
    lines.append("")
    lines += [
        "## Hard rule compliance",
        "",
        "```text",
        "TEST SET ACCESSED: NO",
        "TEST CACHE LOADED: NO",
        "TEST METRICS COMPUTED: NO",
        "```",
        "",
        "## Reproduce",
        "",
        "```powershell",
        report["command"],
        "```",
    ]
    (out_dir / "dual_timescale_memory_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    global OUT_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    OUT_DIR = Path(args.out_dir)
    device = torch.device(args.device)
    start = time.time()

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
        if dataset_paths["normalizer"] is not None:
            refuse_test(dataset_paths["normalizer"])
    refuse_test(args.out_dir)

    report: dict[str, Any] = {
        "experiment": "dual_timescale_memory_costar",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": f"python experiments\\dual_timescale_memory_costar\\run_dual_timescale_memory_costar.py --device {args.device}",
        "hard_rule": "router_train -> selection; router_val -> one frozen evaluation; test set never touched",
        "datasets": {},
    }
    all_results: list[dict[str, Any]] = []
    all_per_horizon: list[dict[str, Any]] = []
    all_selection_stage_a: list[dict[str, Any]] = []
    all_selection_stage_b_chrono: list[dict[str, Any]] = []
    all_selection_stage_b_hv: list[dict[str, Any]] = []
    all_selection_stage_c: list[dict[str, Any]] = []
    all_disagreement_per_window: list[dict[str, Any]] = []
    all_short_vs_long_per_window: list[dict[str, Any]] = []

    for dataset in ["ETTh1", "ETTh2"]:
        p = paths[dataset]
        train_cache = load_cache(p["train_cache"], "router_train_20_60" if dataset == "ETTh1" else "router_train")
        val_cache = load_cache(p["val_cache"], "router_val_60_80" if dataset == "ETTh1" else "router_val")
        if dataset == "ETTh1":
            std = load_std(p["normalizer"], int(val_cache["num_features"]))
        else:
            std = torch.ones(int(val_cache["num_features"]), dtype=torch.float32)
        core = load_frozen_core(p["frozen_config"])
        print(f"[dual-memory] {dataset}: selecting decays/mixing on router_train...", flush=True)
        result = evaluate_dataset(dataset, train_cache, val_cache, std, core, device)
        report["datasets"][dataset] = {
            k: v for k, v in result.items() if k not in ("results", "per_horizon", "disagreement_per_window", "short_vs_long_winner_per_window")
        }
        report["datasets"][dataset]["results"] = result["results"]
        all_results.extend(result["results"])
        all_per_horizon.extend(result["per_horizon"])
        all_disagreement_per_window.extend(result["disagreement_per_window"])
        all_short_vs_long_per_window.extend(result["short_vs_long_winner_per_window"])
        all_selection_stage_a.extend([{"dataset": dataset, **r} for r in result["selection"]["stage_a_decay_grid"]])
        all_selection_stage_b_chrono.extend([{"dataset": dataset, **r} for r in result["selection"]["stage_b_chrono_mix_grid"]])
        all_selection_stage_b_hv.extend([{"dataset": dataset, **r} for r in result["selection"]["stage_b_hv_mix_grid"]])
        all_selection_stage_c.extend([{"dataset": dataset, **r} for r in result["selection"]["stage_c_schedule_grid"]])
        print(f"[dual-memory] {dataset}: done. selected short={result['selection']['selected_short_decay']} long={result['selection']['selected_long_decay']}", flush=True)
        print(f"[dual-memory] {dataset}: disagreement mean_l1(chrono)={result['disagreement']['chrono_mean_l1_distance']:.4f} top_mismatch_rate={result['disagreement']['chrono_top_expert_mismatch_rate']:.4f}", flush=True)
        sv = result["short_vs_long_winner"]["overall"]
        print(f"[dual-memory] {dataset}: short_win_rate={sv['short_win_rate']:.3f} long_win_rate={sv['long_win_rate']:.3f} tie_rate={sv['tie_rate']:.3f}", flush=True)

    report["runtime_sec"] = time.time() - start
    report["test_set_accessed"] = False
    report["test_cache_loaded"] = False
    report["test_metrics_computed"] = False

    write_json(OUT_DIR / "dual_timescale_memory_results.json", report)
    write_csv(OUT_DIR / "dual_timescale_memory_results.csv", all_results)
    write_csv(OUT_DIR / "dual_timescale_memory_per_horizon.csv", all_per_horizon)
    write_csv(OUT_DIR / "dual_timescale_memory_weight_disagreement_per_window.csv", all_disagreement_per_window)
    write_csv(OUT_DIR / "dual_timescale_memory_short_vs_long_winner_per_window.csv", all_short_vs_long_per_window)
    write_csv(OUT_DIR / "router_train_stage_a_decay_grid.csv", all_selection_stage_a)
    write_csv(OUT_DIR / "router_train_stage_b_chrono_mix_grid.csv", all_selection_stage_b_chrono)
    write_csv(OUT_DIR / "router_train_stage_b_hv_mix_grid.csv", all_selection_stage_b_hv)
    write_csv(OUT_DIR / "router_train_stage_c_schedule_grid.csv", all_selection_stage_c)
    make_report(OUT_DIR, report)

    print("TEST SET ACCESSED: NO")
    print("TEST CACHE LOADED: NO")
    print("TEST METRICS COMPUTED: NO")
    print(json.dumps({k: v for k, v in report.items() if k != "datasets"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
