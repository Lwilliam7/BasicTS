"""Shared, dataset-generic building blocks for the frozen multi-dataset COSTAR
evaluation. Every function here operates directly on the standardized
walk-forward cache schema produced by `scripts/build_costarts_walkforward_cache.py`
(histories/targets/target_masks/prediction_stack/expert_names/
absolute_window_starts/forecast_horizon), which is identical across ETTh1,
ETTm1, Weather, and Electricity -- so nothing dataset-specific is invented
here; this simply generalizes the same per-dataset logic already used for
ETTh1/ETTh2 (`select_core_on_router_train`, `chronological_online_weights`,
`chronological_hv_weights`) to any cache with that schema.

FROZEN ROUTER: Full horizon x variable (HxV) causal EMA only. No separate
global branch, no global+HxV blend, no low-rank approximation, no
dual-timescale memory, no specialists, no Ridge/MLP residual correction.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.chronological_adaptive_costar.run_chronological_adaptive_costar import (  # noqa: E402
    enforce_observable,
    paired_bootstrap,
)
from experiments.horizon_variable_adaptive_costar.run_hv_adaptive_costar import (  # noqa: E402
    Trial as HvTrial,
    chronological_hv_weights,
    predict_from_hv_weights,
)
from experiments.oracle_weight_tournament.run_tournament import (  # noqa: E402
    load_cache,
    load_std,
    sample_mae,
    sample_mse,
    weighted_forecast,
)


# Canonical, frozen settings -- identical to costar_router_ablation and the
# production ETTh1/ETTh2 HxV branch. Not tuned per dataset.
CANONICAL_DECAY = 0.95
CANONICAL_TEMPERATURE = 0.1
CORE_SIZE = 3
EXPERT_ORDER = ("DLinear", "PatchTST", "iTransformer", "TimesNet", "ModernTCN")


def refuse_test(path: str | Path) -> None:
    if "test" in str(path).lower():
        raise ValueError(f"Test access forbidden before freeze: {path}")


def expert_indices(cache: Mapping[str, Any], experts: Sequence[str]) -> list[int]:
    names = list(cache["expert_names"])
    return [names.index(name) for name in experts]


def forecasts_for(cache: Mapping[str, Any], expert_idx: Sequence[int]) -> torch.Tensor:
    return cache["prediction_stack"][..., list(expert_idx)].to(torch.float32)


def per_location_error(cache: Mapping[str, Any], expert_idx: Sequence[int], std: torch.Tensor) -> torch.Tensor:
    forecasts = forecasts_for(cache, expert_idx)
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.float32)
    return ((forecasts - target.unsqueeze(-1)) / std.view(1, 1, -1, 1)).abs() * mask.unsqueeze(-1)


def metric_values(cache: Mapping[str, Any], pred: torch.Tensor, std: torch.Tensor) -> dict[str, Any]:
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.bool)
    mae = sample_mae(pred, target, mask, std)
    mse = sample_mse(pred, target, mask, std)
    return {"mae": float(mae.mean()), "mse": float(mse.mean()), "per_window_mae": mae, "per_window_mse": mse}


# ---------------------------------------------------------------------------
# Expert-selection protocol (unchanged): pooled chronological OOF MAE over
# router_train folds, generalized from `select_core_on_router_train` in the
# ETTh1/ETTh2 modules so it works on any dataset's own router_train cache.
# ---------------------------------------------------------------------------


def train_folds(n: int) -> list[tuple[int, int, int]]:
    min_train = int(round(n * 0.2))
    usable = n - min_train
    bounds = [min_train + i * usable // 4 for i in range(5)]
    return [(0, bounds[i], bounds[i + 1]) for i in range(4)]


def select_core_on_router_train(cache: Mapping[str, Any], std: torch.Tensor, core_size: int = CORE_SIZE) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    names = list(cache["expert_names"])
    stack = cache["prediction_stack"].to(torch.float32)
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.bool)
    folds = train_folds(int(cache["num_windows"]))
    rows: list[dict[str, Any]] = []
    for idx in itertools.combinations(range(len(names)), core_size):
        pred = weighted_forecast(stack[..., list(idx)], torch.full((int(cache["num_windows"]), core_size), 1.0 / core_size))
        mae_chunks, mse_chunks, fold_rows = [], [], []
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
                "worst_fold_mae": max(r["mae"] for r in fold_rows),
                "fold_rows": fold_rows,
            }
        )
    rows = sorted(rows, key=lambda r: (float(r["pooled_oof_mae"]), float(r["pooled_oof_mse"]), float(r["worst_fold_mae"]), str(r["subset"])))
    return rows, rows[0]


def select_best_single_expert(cache: Mapping[str, Any], std: torch.Tensor, expert_idx: Sequence[int]) -> tuple[int, str, float]:
    """Best single expert *within the selected core*, by pooled router_train MAE."""
    names = list(cache["expert_names"])
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.bool)
    best_idx, best_name, best_mae = None, None, float("inf")
    for i in expert_idx:
        pred = cache["prediction_stack"][..., i].to(torch.float32)
        mae = float(sample_mae(pred, target, mask, std).mean())
        if mae < best_mae:
            best_idx, best_name, best_mae = i, names[i], mae
    return best_idx, best_name, best_mae


# ---------------------------------------------------------------------------
# The five primary methods. Global / variable-only / full-HxV all reuse the
# SAME unified formula (`errors_to_weights` via `chronological_hv_weights`),
# varying only the aggregation granularity via `trial.mode` -- this isolates
# granularity as the only variable, exactly as in costar_router_ablation.
# No blending with a static prior, no low-rank approximation, no specialists.
# ---------------------------------------------------------------------------


def granularity_trial(mode: str) -> HvTrial:
    return HvTrial("hv_ema", f"frozen_{mode}", mode=mode, rank=1, decay=CANONICAL_DECAY, temperature=CANONICAL_TEMPERATURE)


def equal_fixed_prediction(cache: Mapping[str, Any], expert_idx: Sequence[int]) -> tuple[torch.Tensor, dict[str, Any]]:
    pred = forecasts_for(cache, expert_idx).mean(dim=-1)
    return pred, {"num_causal_updates": 0, "decay": None, "temperature": None}


def best_single_expert_prediction(cache: Mapping[str, Any], expert_col: int) -> tuple[torch.Tensor, dict[str, Any]]:
    pred = cache["prediction_stack"][..., expert_col].to(torch.float32)
    return pred, {"num_causal_updates": 0, "decay": None, "temperature": None}


def granularity_ema_prediction(
    cache: Mapping[str, Any], train_cache: Mapping[str, Any], expert_idx: Sequence[int], std: torch.Tensor, mode: str
) -> tuple[torch.Tensor, dict[str, Any]]:
    starts = cache["absolute_window_starts"].to(torch.long)
    horizon = int(cache["forecast_horizon"])
    if not bool(torch.all(starts[1:] > starts[:-1])):
        raise AssertionError("Cache absolute_window_starts are not strictly chronological")
    forecasts = forecasts_for(cache, expert_idx)
    train_err = per_location_error(train_cache, expert_idx, std)
    val_err = per_location_error(cache, expert_idx, std)
    trial = granularity_trial(mode)
    weights, extra = chronological_hv_weights(starts, train_err.mean(dim=0), val_err, horizon, trial)
    pred = predict_from_hv_weights(forecasts, weights)
    return pred, {"num_causal_updates": extra["num_updates"], "decay": CANONICAL_DECAY, "temperature": CANONICAL_TEMPERATURE, "mode": mode}


METHOD_ORDER = [
    ("best_single_expert", "Best single expert"),
    ("equal_fixed", "Equal fixed ensemble"),
    ("global_causal", "Global causal EMA"),
    ("variable_only", "Variable-only causal EMA"),
    ("hxv_causal", "Full HxV causal EMA (FROZEN CANDIDATE)"),
]
MODE_BY_METHOD = {"global_causal": "global", "variable_only": "variable", "hxv_causal": "hv"}


def predict_method(
    method: str, cache: Mapping[str, Any], train_cache: Mapping[str, Any], expert_idx: Sequence[int], std: torch.Tensor, best_expert_col: int
) -> tuple[torch.Tensor, dict[str, Any]]:
    if method == "equal_fixed":
        return equal_fixed_prediction(cache, expert_idx)
    if method == "best_single_expert":
        return best_single_expert_prediction(cache, best_expert_col)
    return granularity_ema_prediction(cache, train_cache, expert_idx, std, MODE_BY_METHOD[method])


# ---------------------------------------------------------------------------
# Causality perturbation check: mutate only the tail of a validation/test
# cache and verify earlier-window predictions are bit-identical.
# ---------------------------------------------------------------------------


def perturb_tail_targets(cache: Mapping[str, Any], suffix_start: int, seed: int = 20260821) -> dict[str, Any]:
    cloned = dict(cache)
    gen = torch.Generator().manual_seed(seed)
    targets = cache["targets"].clone()
    noise = torch.randn(targets[suffix_start:].shape, generator=gen, dtype=torch.float32)
    targets[suffix_start:] = noise
    cloned["targets"] = targets
    return cloned


def causality_perturbation_check(
    method: str, cache: Mapping[str, Any], train_cache: Mapping[str, Any], expert_idx: Sequence[int], std: torch.Tensor, best_expert_col: int
) -> dict[str, Any]:
    n = int(cache["num_windows"])
    suffix_start = int(round(n * 0.75))
    starts = cache["absolute_window_starts"].to(torch.long)
    starts_chronological = bool(torch.all(starts[1:] > starts[:-1]))
    base_pred, _ = predict_method(method, cache, train_cache, expert_idx, std, best_expert_col)
    mutated = perturb_tail_targets(cache, suffix_start)
    mut_pred, _ = predict_method(method, mutated, train_cache, expert_idx, std, best_expert_col)
    prefix_equal = bool(torch.equal(base_pred[:suffix_start], mut_pred[:suffix_start]))
    tail_differs = not bool(torch.equal(base_pred[suffix_start:], mut_pred[suffix_start:]))
    return {
        "method": method,
        "starts_chronological": starts_chronological,
        "suffix_start_window": suffix_start,
        "num_windows": n,
        "earlier_windows_unchanged": prefix_equal,
        "tail_predictions_reacted": tail_differs,
        "result": "PASS" if (starts_chronological and prefix_equal) else "FAIL",
    }


def verify_router_train_out_of_sample(router_train_cache: Mapping[str, Any]) -> dict[str, Any]:
    """Check the router_train cache's own provenance: its forecasts must come
    from experts trained strictly before the windows they predict (the
    walk-forward block_b_oos / block_c_oos scheme), never from experts that
    were trained on the same range they are scored on."""
    provenance = router_train_cache.get("provenance", {})
    source_caches = provenance.get("source_caches", {})
    ok = {"block_b_oos", "block_c_oos"}.issubset(source_caches)
    return {
        "cache_role": router_train_cache.get("cache_role"),
        "has_block_b_oos_source": "block_b_oos" in source_caches,
        "has_block_c_oos_source": "block_c_oos" in source_caches,
        "protocol_note": provenance.get("protocol"),
        "result": "PASS" if ok else "FAIL",
    }


# ---------------------------------------------------------------------------
# Dependence-aware statistics (block bootstrap, every-12th phase bootstrap) --
# same methodology as costar_router_ablation's dependence-aware retest.
# ---------------------------------------------------------------------------


def block_bootstrap_with_prob(
    candidate: torch.Tensor, baseline: torch.Tensor, block: int, seed: int = 20260821, samples: int = 10000, chunk: int = 500
) -> dict[str, Any]:
    """Vectorized paired moving/block bootstrap: resamples contiguous runs of
    `block` windows with replacement to reconstruct a same-length pseudo-series,
    `samples` times. Processed in chunks to bound peak memory for large
    validation sets. Fully equivalent to (but ~1000x faster than) a plain
    Python loop doing the same resampling one sample at a time."""
    diff = candidate - baseline
    n = diff.numel()
    block = max(1, min(block, n))
    n_blocks = max(1, -(-n // block))
    max_start = n - block + 1
    offsets = torch.arange(block)
    gen = torch.Generator().manual_seed(seed)
    means = torch.empty(samples, dtype=torch.float32)
    for lo in range(0, samples, chunk):
        hi = min(lo + chunk, samples)
        batch = hi - lo
        starts_idx = torch.randint(0, max_start, (batch, n_blocks), generator=gen)
        idx = (starts_idx.unsqueeze(-1) + offsets.view(1, 1, -1)).reshape(batch, n_blocks * block)[:, :n]
        means[lo:hi] = diff[idx].mean(dim=1)
    return {
        "block_size": block,
        "samples": samples,
        "mean_delta": float(diff.mean()),
        "ci95_low": float(torch.quantile(means, 0.025)),
        "ci95_high": float(torch.quantile(means, 0.975)),
        "ci_excludes_zero": bool(torch.quantile(means, 0.975) < 0 or torch.quantile(means, 0.025) > 0),
        "prob_delta_negative": float((means < 0).to(torch.float32).mean()),
    }


def every_kth_phase_bootstrap(diff: torch.Tensor, k: int = 12, seed: int = 20260821, samples: int = 10000) -> dict[str, Any]:
    n = diff.numel()
    phase_means, phase_counts = [], []
    for offset in range(k):
        idx = torch.arange(offset, n, k)
        if idx.numel() == 0:
            continue
        phase_means.append(float(diff[idx].mean()))
        phase_counts.append(int(idx.numel()))
    t = torch.tensor(phase_means)
    num_phases = t.numel()
    gen = torch.Generator().manual_seed(seed)
    vals = []
    for _ in range(samples):
        idx = torch.randint(0, num_phases, (num_phases,), generator=gen)
        vals.append(float(t[idx].mean()))
    boot = torch.tensor(vals)
    return {
        "k": k,
        "num_phases": num_phases,
        "windows_per_phase_min": min(phase_counts),
        "windows_per_phase_max": max(phase_counts),
        "mean_delta": float(t.mean()),
        "ci95_low": float(torch.quantile(boot, 0.025)),
        "ci95_high": float(torch.quantile(boot, 0.975)),
        "ci_excludes_zero": bool(torch.quantile(boot, 0.975) < 0 or torch.quantile(boot, 0.025) > 0),
        "prob_delta_negative": float((boot < 0).to(torch.float32).mean()),
    }
