"""Residual correction experiments on the frozen COSTAR-TS validation baseline.

The runner performs two validation-safe follow-ups to the current best ETTh1
walk-forward predictor:

1. causal EMA residual-bias correction over global/horizon/variable/HxV groups
2. conservative residual correction with chronological ridge cross-fitting

All hyperparameters are selected on chronological folds inside router-train.
Validation is evaluated once after selection.  The test cache is refused.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


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
from experiments.horizon_variable_adaptive_costar.run_hv_adaptive_costar import (  # noqa: E402
    Trial as HvTrial,
    chronological_hv_weights,
    fixed3_forecasts,
    per_location_abs_error,
    predict_from_hv_weights,
)
from experiments.oracle_weight_tournament.run_tournament import (  # noqa: E402
    FIXED3,
    args_global_weights,
    fixed3_indices,
    load_cache,
    load_std,
    sample_mae,
    sample_mse,
    weighted_forecast,
)


BASELINE_MAE = 0.36364156007766724
BASELINE_MSE = 0.3067120909690857
STRONG_TARGET = 0.3619
EXCEPTIONAL_TARGET = 0.3600
SCALES = (96, 192, 336, 720)


@dataclass(frozen=True)
class BiasConfig:
    structure: str
    decay: float
    alpha: float
    clip_multiple: float | None
    min_count: int

    @property
    def name(self) -> str:
        clip = "unclipped" if self.clip_multiple is None else f"clip{self.clip_multiple:g}"
        return f"{self.structure}_decay{self.decay:g}_alpha{self.alpha:g}_{clip}_warm{self.min_count}"


@dataclass(frozen=True)
class RidgeConfig:
    ridge: float
    alpha: float
    clip_multiple: float | None
    feature_set: str = "full"

    @property
    def name(self) -> str:
        clip = "unclipped" if self.clip_multiple is None else f"clip{self.clip_multiple:g}"
        return f"ridge{self.ridge:g}_alpha{self.alpha:g}_{clip}_{self.feature_set}"


@dataclass(frozen=True)
class MlpConfig:
    seed: int
    hidden: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-2
    alpha: float = 0.1
    clip_multiple: float | None = 0.5
    epochs: int = 40
    patience: int = 6

    @property
    def name(self) -> str:
        clip = "unclipped" if self.clip_multiple is None else f"clip{self.clip_multiple:g}"
        return f"mlp_seed{self.seed}_h{self.hidden}_alpha{self.alpha:g}_{clip}"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def refuse_test_path(path: str | Path) -> None:
    if "test" in str(path).lower():
        raise ValueError(f"Refusing test path: {path}")


def metrics(cache: Mapping[str, Any], std: torch.Tensor, pred: torch.Tensor) -> dict[str, Any]:
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.bool)
    mae = sample_mae(pred, target, mask, std)
    mse = sample_mse(pred, target, mask, std)
    return {"mae": float(mae.mean()), "mse": float(mse.mean()), "per_window_mae": mae, "per_window_mse": mse}


def location_mae(cache: Mapping[str, Any], std: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.float32)
    stdv = std.view(1, 1, -1)
    return ((pred - target) / stdv).abs() * mask


def per_axis_rows(cache: Mapping[str, Any], std: torch.Tensor, pred: torch.Tensor, label: str) -> list[dict[str, Any]]:
    err = location_mae(cache, std, pred)
    mask = cache["target_masks"].to(torch.float32)
    rows: list[dict[str, Any]] = []
    for h in range(err.shape[1]):
        rows.append({"method": label, "axis": "horizon", "index": h, "mae": float(err[:, h].sum() / mask[:, h].sum().clamp_min(1))})
    for v in range(err.shape[2]):
        rows.append({"method": label, "axis": "variable", "index": v, "mae": float(err[:, :, v].sum() / mask[:, :, v].sum().clamp_min(1))})
    return rows


def per_location_rows(cache: Mapping[str, Any], std: torch.Tensor, pred: torch.Tensor, baseline: torch.Tensor, label: str) -> list[dict[str, Any]]:
    cand = location_mae(cache, std, pred)
    base = location_mae(cache, std, baseline)
    mask = cache["target_masks"].to(torch.float32)
    rows: list[dict[str, Any]] = []
    for h in range(cand.shape[1]):
        for v in range(cand.shape[2]):
            denom = mask[:, h, v].sum().clamp_min(1)
            cand_mae = float(cand[:, h, v].sum() / denom)
            base_mae = float(base[:, h, v].sum() / denom)
            rows.append(
                {
                    "method": label,
                    "horizon": h,
                    "variable": v,
                    "mae": cand_mae,
                    "baseline_mae": base_mae,
                    "delta_vs_baseline": cand_mae - base_mae,
                }
            )
    return rows


def train_folds(n: int, min_train_fraction: float = 0.2, folds: int = 4) -> list[tuple[int, int, int]]:
    min_train = max(1, int(round(n * min_train_fraction)))
    usable = n - min_train
    bounds = [min_train + i * usable // folds for i in range(folds + 1)]
    return [(0, bounds[i], bounds[i + 1]) for i in range(folds)]


def residual_std_by_structure(residual: torch.Tensor, structure: str) -> torch.Tensor:
    if structure == "global":
        return residual.flatten().std(unbiased=False).clamp_min(1e-6).view(1, 1)
    if structure == "horizon":
        return residual.std(dim=(0, 2), unbiased=False).clamp_min(1e-6).view(-1, 1)
    if structure == "variable":
        return residual.std(dim=(0, 1), unbiased=False).clamp_min(1e-6).view(1, -1)
    if structure == "hv":
        return residual.std(dim=0, unbiased=False).clamp_min(1e-6)
    raise ValueError(structure)


def aggregate_residual(residual: torch.Tensor, structure: str) -> torch.Tensor:
    if structure == "global":
        return residual.mean().view(1, 1)
    if structure == "horizon":
        return residual.mean(dim=1, keepdim=True)
    if structure == "variable":
        return residual.mean(dim=0, keepdim=True)
    if structure == "hv":
        return residual
    raise ValueError(structure)


def expand_group(x: torch.Tensor, structure: str, h: int, v: int) -> torch.Tensor:
    if structure == "global":
        return x.expand(h, v)
    if structure == "horizon":
        return x.expand(h, v)
    if structure == "variable":
        return x.expand(h, v)
    if structure == "hv":
        return x
    raise ValueError(structure)


def causal_bias_correct(
    starts: torch.Tensor,
    baseline: torch.Tensor,
    target: torch.Tensor,
    horizon: int,
    config: BiasConfig,
    init_residuals: torch.Tensor | None,
    clip_std: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    h, v = baseline.shape[1], baseline.shape[2]
    if init_residuals is None or init_residuals.numel() == 0:
        state = aggregate_residual(torch.zeros(h, v), config.structure)
        count = 0
    else:
        grouped = torch.stack([aggregate_residual(r, config.structure) for r in init_residuals])
        state = grouped.mean(dim=0)
        count = int(init_residuals.shape[0])
    pending: list[int] = []
    preds: list[torch.Tensor] = []
    correction_abs = []
    clipped_count = 0
    total_count = 0
    updates = 0
    for i in range(baseline.shape[0]):
        now = int(starts[i])
        still: list[int] = []
        for j in pending:
            if int(starts[j]) + horizon <= now:
                enforce_observable(int(starts[j]), now, horizon)
                grouped = aggregate_residual(target[j] - baseline[j], config.structure)
                state = float(config.decay) * state + (1.0 - float(config.decay)) * grouped
                count += 1
                updates += 1
            else:
                still.append(j)
        pending = still
        bias = state
        if config.clip_multiple is not None:
            limit = float(config.clip_multiple) * clip_std
            clipped = bias.clamp(-limit, limit)
            clipped_count += int((clipped != bias).sum().item())
            bias = clipped
        total_count += int(bias.numel())
        correction = torch.zeros((h, v), dtype=torch.float32)
        if count >= int(config.min_count):
            correction = float(config.alpha) * expand_group(bias, config.structure, h, v)
        correction_abs.append(float(correction.abs().mean()))
        preds.append(baseline[i] + correction)
        pending.append(i)
    return torch.stack(preds), {
        "num_updates": updates,
        "final_count": count,
        "mean_abs_correction": float(np.mean(correction_abs)) if correction_abs else 0.0,
        "max_abs_correction": float(max(correction_abs)) if correction_abs else 0.0,
        "clip_frequency": float(clipped_count / max(total_count, 1)),
    }


def build_bias_grid() -> list[BiasConfig]:
    configs: list[BiasConfig] = []
    for structure in ("global", "horizon", "variable", "hv"):
        for decay in (0.90, 0.95, 0.97, 0.98, 0.99):
            for alpha in (0.10, 0.25, 0.50, 0.75, 1.00):
                for clip in (0.5, 1.0, 2.0, 3.0, None):
                    for warm in (0, 12, 24, 48, 96):
                        configs.append(BiasConfig(structure, decay, alpha, clip, warm))
    return configs


def evaluate_bias_on_train_folds(
    starts: torch.Tensor,
    baseline: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    std: torch.Tensor,
    config: BiasConfig,
    folds: Sequence[tuple[int, int, int]],
) -> dict[str, Any]:
    h = int(baseline.shape[1])
    fold_rows = []
    cand_all = []
    base_all = []
    residual = target - baseline
    for fold_id, (train_lo, eval_lo, eval_hi) in enumerate(folds):
        train_residual = residual[train_lo:eval_lo]
        clip_std = residual_std_by_structure(train_residual, config.structure)
        pred, extra = causal_bias_correct(
            starts[eval_lo:eval_hi],
            baseline[eval_lo:eval_hi],
            target[eval_lo:eval_hi],
            h,
            config,
            init_residuals=train_residual,
            clip_std=clip_std,
        )
        cand = sample_mae(pred, target[eval_lo:eval_hi], mask[eval_lo:eval_hi], std)
        base = sample_mae(baseline[eval_lo:eval_hi], target[eval_lo:eval_hi], mask[eval_lo:eval_hi], std)
        fold_rows.append({"fold": fold_id, "mae": float(cand.mean()), "baseline_mae": float(base.mean()), "delta": float(cand.mean() - base.mean()), **extra})
        cand_all.append(cand)
        base_all.append(base)
    cand_t = torch.cat(cand_all)
    base_t = torch.cat(base_all)
    return {
        "name": config.name,
        **asdict(config),
        "fold_mae": float(cand_t.mean()),
        "fold_baseline_mae": float(base_t.mean()),
        "fold_delta": float(cand_t.mean() - base_t.mean()),
        "fold_wins": int(sum(r["delta"] < 0 for r in fold_rows)),
        "fold_rows": fold_rows,
    }


def causal_bias_state_sequence(
    starts: torch.Tensor,
    residual: torch.Tensor,
    horizon: int,
    structure: str,
    decay: float,
    init_residuals: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    h, v = residual.shape[1], residual.shape[2]
    grouped_init = torch.stack([aggregate_residual(r, structure) for r in init_residuals])
    state = grouped_init.mean(dim=0)
    count = int(init_residuals.shape[0])
    pending: list[int] = []
    states: list[torch.Tensor] = []
    counts: list[int] = []
    for i in range(residual.shape[0]):
        now = int(starts[i])
        still: list[int] = []
        for j in pending:
            if int(starts[j]) + horizon <= now:
                enforce_observable(int(starts[j]), now, horizon)
                grouped = aggregate_residual(residual[j], structure)
                state = float(decay) * state + (1.0 - float(decay)) * grouped
                count += 1
            else:
                still.append(j)
        pending = still
        states.append(expand_group(state, structure, h, v).clone())
        counts.append(count)
        pending.append(i)
    return torch.stack(states), torch.tensor(counts, dtype=torch.long)


def evaluate_bias_grid_cached(
    starts: torch.Tensor,
    baseline: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    std: torch.Tensor,
    configs: Sequence[BiasConfig],
    folds: Sequence[tuple[int, int, int]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    residual = target - baseline
    state_cache: dict[tuple[str, float, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for fold_id, (train_lo, eval_lo, eval_hi) in enumerate(folds):
        init_residuals = residual[train_lo:eval_lo]
        for structure in ("global", "horizon", "variable", "hv"):
            clip_std = expand_group(residual_std_by_structure(init_residuals, structure), structure, baseline.shape[1], baseline.shape[2])
            for decay in (0.90, 0.95, 0.97, 0.98, 0.99):
                state_seq, count_seq = causal_bias_state_sequence(
                    starts[eval_lo:eval_hi],
                    residual[eval_lo:eval_hi],
                    int(baseline.shape[1]),
                    structure,
                    decay,
                    init_residuals,
                )
                state_cache[(structure, decay, fold_id)] = (state_seq, count_seq, clip_std)

    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for config in configs:
        cand_all = []
        base_all = []
        fold_rows = []
        total_clipped = 0
        total_values = 0
        abs_corr = []
        for fold_id, (_, eval_lo, eval_hi) in enumerate(folds):
            state_seq, count_seq, clip_std = state_cache[(config.structure, config.decay, fold_id)]
            bias = state_seq
            if config.clip_multiple is not None:
                limit = float(config.clip_multiple) * clip_std.view(1, *clip_std.shape)
                clipped = bias.clamp(-limit, limit)
                total_clipped += int((clipped != bias).sum().item())
                bias = clipped
            total_values += int(bias.numel())
            warm = (count_seq >= int(config.min_count)).to(torch.float32).view(-1, 1, 1)
            correction = float(config.alpha) * bias * warm
            pred = baseline[eval_lo:eval_hi] + correction
            cand = sample_mae(pred, target[eval_lo:eval_hi], mask[eval_lo:eval_hi], std)
            base = sample_mae(baseline[eval_lo:eval_hi], target[eval_lo:eval_hi], mask[eval_lo:eval_hi], std)
            cand_all.append(cand)
            base_all.append(base)
            abs_corr.append(float(correction.abs().mean()))
            fold_rows.append(
                {
                    "fold": fold_id,
                    "mae": float(cand.mean()),
                    "baseline_mae": float(base.mean()),
                    "delta": float(cand.mean() - base.mean()),
                }
            )
        cand_t = torch.cat(cand_all)
        base_t = torch.cat(base_all)
        row = {
            "name": config.name,
            **asdict(config),
            "fold_mae": float(cand_t.mean()),
            "fold_baseline_mae": float(base_t.mean()),
            "fold_delta": float(cand_t.mean() - base_t.mean()),
            "fold_wins": int(sum(r["delta"] < 0 for r in fold_rows)),
            "mean_abs_correction": float(np.mean(abs_corr)),
            "clip_frequency": float(total_clipped / max(total_values, 1)),
            "fold_rows": fold_rows,
        }
        rows.append({k: v for k, v in row.items() if k != "fold_rows"})
        if best is None or float(row["fold_mae"]) < float(best["fold_mae"]):
            best = row
    assert best is not None
    return rows, best


def load_series_prefix(dataset_dir: Path) -> torch.Tensor:
    train_path = dataset_dir / "train_data.npy"
    val_path = dataset_dir / "val_data.npy"
    refuse_test_path(train_path)
    refuse_test_path(val_path)
    train = torch.from_numpy(np.load(train_path)).to(torch.float32)
    val = torch.from_numpy(np.load(val_path)).to(torch.float32)
    return torch.cat((train, val), dim=0)


def assert_history_available(series: torch.Tensor, start: int, scale: int) -> None:
    if start - scale < 0:
        raise RuntimeError(f"Insufficient history: start={start}, scale={scale}")
    if start > series.shape[0]:
        raise RuntimeError(f"Forecast start {start} exceeds loaded non-test series length {series.shape[0]}")


def history_summary_for_window(series: torch.Tensor, start: int, variable: int, scale: int) -> list[float]:
    assert_history_available(series, start, scale)
    hist = series[start - scale : start, variable].to(torch.float32)
    first = hist[: max(1, scale // 4)].mean()
    last = hist[-max(1, scale // 4) :].mean()
    mean = hist.mean()
    std = hist.std(unbiased=False)
    return [
        float(mean),
        float(std),
        float(hist[-1]),
        float(last - first),
        float(hist[-1] - mean),
        float(last - mean),
    ]


def causal_residual_stats(
    starts: torch.Tensor,
    residual_norm: torch.Tensor,
    horizon: int,
    init_residuals_norm: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    h, v = residual_norm.shape[1], residual_norm.shape[2]
    if init_residuals_norm is None or init_residuals_norm.numel() == 0:
        mean = torch.zeros(h, v)
        var = torch.ones(h, v)
        count = 0
    else:
        mean = init_residuals_norm.mean(dim=0)
        var = init_residuals_norm.var(dim=0, unbiased=False).clamp_min(1e-6)
        count = int(init_residuals_norm.shape[0])
    pending: list[int] = []
    out: list[torch.Tensor] = []
    updates = 0
    for i in range(residual_norm.shape[0]):
        now = int(starts[i])
        still: list[int] = []
        for j in pending:
            if int(starts[j]) + horizon <= now:
                enforce_observable(int(starts[j]), now, horizon)
                x = residual_norm[j]
                count += 1
                delta = x - mean
                mean = mean + delta / count
                var = ((count - 2) / max(count - 1, 1)) * var + delta.square() / count if count > 1 else torch.zeros_like(var)
                var = var.clamp_min(1e-6)
                updates += 1
            else:
                still.append(j)
        pending = still
        out.append(torch.stack((mean, var.sqrt()), dim=-1))
        pending.append(i)
    return torch.stack(out), {"num_residual_stat_updates": updates}


def build_feature_tensor(
    cache: Mapping[str, Any],
    starts: torch.Tensor,
    baseline: torch.Tensor,
    std: torch.Tensor,
    series: torch.Tensor,
    init_residuals_norm: torch.Tensor | None,
) -> tuple[torch.Tensor, list[str], dict[str, Any]]:
    forecasts = fixed3_forecasts(cache)
    target = cache["targets"].to(torch.float32)
    residual_norm = (target - baseline) / std.view(1, 1, -1)
    h, v = baseline.shape[1], baseline.shape[2]
    stats, stat_extra = causal_residual_stats(starts, residual_norm, h, init_residuals_norm)
    rows: list[list[float]] = []
    names: list[str] | None = None
    global_history_cache: dict[tuple[int, int], list[float]] = {}
    for i in range(baseline.shape[0]):
        start = int(starts[i])
        for hh in range(h):
            h_onehot = [1.0 if hh == k else 0.0 for k in range(h)]
            for vv in range(v):
                expert_vals = forecasts[i, hh, vv].tolist()
                pairwise = [
                    expert_vals[0] - expert_vals[1],
                    expert_vals[0] - expert_vals[2],
                    expert_vals[1] - expert_vals[2],
                    abs(expert_vals[0] - expert_vals[1]),
                    abs(expert_vals[0] - expert_vals[2]),
                    abs(expert_vals[1] - expert_vals[2]),
                ]
                disp = [
                    float(forecasts[i, hh, vv].mean()),
                    float(forecasts[i, hh, vv].std(unbiased=False)),
                    float(forecasts[i, hh, vv].max() - forecasts[i, hh, vv].min()),
                ]
                history_bits: list[float] = []
                history_names: list[str] = []
                for scale in SCALES:
                    key = (start, vv, scale)
                    if key not in global_history_cache:
                        global_history_cache[key] = history_summary_for_window(series, start, vv, scale)
                    vals = global_history_cache[key]
                    history_bits.extend(vals)
                    history_names.extend([f"hist_s{scale}_{name}" for name in ("mean", "std", "last", "trend", "last_minus_mean", "tail_minus_mean")])
                row = (
                    [float(baseline[i, hh, vv])]
                    + [float(x) for x in expert_vals]
                    + pairwise
                    + disp
                    + [float(stats[i, hh, vv, 0]), float(stats[i, hh, vv, 1])]
                    + [float(hh) / max(h - 1, 1), float(vv) / max(v - 1, 1)]
                    + h_onehot
                    + [1.0 if vv == k else 0.0 for k in range(v)]
                    + history_bits
                )
                if names is None:
                    names = (
                        ["baseline_pred", *[f"expert_{name}" for name in FIXED3]]
                        + ["diff_p_i", "diff_p_t", "diff_i_t", "absdiff_p_i", "absdiff_p_t", "absdiff_i_t"]
                        + ["expert_mean", "expert_std", "expert_range"]
                        + ["causal_residual_mean", "causal_residual_std", "horizon_scaled", "variable_scaled"]
                        + [f"horizon_{k}" for k in range(h)]
                        + [f"variable_{k}" for k in range(v)]
                        + history_names
                    )
                rows.append(row)
    assert names is not None
    return torch.tensor(rows, dtype=torch.float32), names, stat_extra


def flattened_targets(cache: Mapping[str, Any], baseline: torch.Tensor, std: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.bool)
    y = ((target - baseline) / std.view(1, 1, -1)).reshape(-1)
    return y, mask.reshape(-1)


def fit_scaler(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = x.mean(dim=0, keepdim=True)
    scale = x.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    return mean, scale


def apply_scaler(x: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (x - mean) / scale


def fit_ridge(x: torch.Tensor, y: torch.Tensor, ridge: float) -> torch.Tensor:
    ones = torch.ones((x.shape[0], 1), dtype=x.dtype)
    xb = torch.cat((ones, x), dim=1)
    eye = torch.eye(xb.shape[1], dtype=x.dtype)
    eye[0, 0] = 0.0
    a = xb.T @ xb + float(ridge) * eye
    b = xb.T @ y
    return torch.linalg.solve(a, b)


def predict_linear(x: torch.Tensor, coef: torch.Tensor) -> torch.Tensor:
    return coef[0] + x @ coef[1:]


def apply_residual_delta(
    baseline: torch.Tensor,
    delta_norm_flat: torch.Tensor,
    std: torch.Tensor,
    alpha: float,
    clip_multiple: float | None,
    residual_train_std_norm: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    n, h, v = baseline.shape
    delta = delta_norm_flat.view(n, h, v)
    if clip_multiple is not None:
        limit = float(clip_multiple) * residual_train_std_norm.view(1, h, v)
        clipped = delta.clamp(-limit, limit)
    else:
        clipped = delta
    raw = float(alpha) * clipped * std.view(1, 1, -1)
    return baseline + raw, {
        "mean_abs_delta_norm": float(delta.abs().mean()),
        "mean_abs_applied_delta_norm": float((float(alpha) * clipped).abs().mean()),
        "clip_frequency": float((clipped != delta).to(torch.float32).mean()),
    }


def evaluate_ridge_on_folds(
    cache: Mapping[str, Any],
    starts: torch.Tensor,
    baseline: torch.Tensor,
    std: torch.Tensor,
    series: torch.Tensor,
    config: RidgeConfig,
    folds: Sequence[tuple[int, int, int]],
) -> dict[str, Any]:
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.bool)
    residual_norm = (target - baseline) / std.view(1, 1, -1)
    fold_rows = []
    cand_all = []
    base_all = []
    for fold_id, (train_lo, eval_lo, eval_hi) in enumerate(folds):
        train_slice = {
            **cache,
            "targets": target[:eval_lo],
            "target_masks": mask[:eval_lo],
            "prediction_stack": cache["prediction_stack"][:eval_lo],
        }
        x_train, _, _ = build_feature_tensor(
            train_slice,
            starts[:eval_lo],
            baseline[:eval_lo],
            std,
            series,
            init_residuals_norm=None,
        )
        y_train, m_train = flattened_targets(train_slice, baseline[:eval_lo], std)
        x_train = x_train[m_train]
        y_train = y_train[m_train]
        mean, scale = fit_scaler(x_train)
        coef = fit_ridge(apply_scaler(x_train, mean, scale), y_train, config.ridge)
        eval_slice = {
            **cache,
            "targets": target[eval_lo:eval_hi],
            "target_masks": mask[eval_lo:eval_hi],
            "prediction_stack": cache["prediction_stack"][eval_lo:eval_hi],
        }
        x_eval, _, _ = build_feature_tensor(
            eval_slice,
            starts[eval_lo:eval_hi],
            baseline[eval_lo:eval_hi],
            std,
            series,
            init_residuals_norm=residual_norm[:eval_lo],
        )
        delta = predict_linear(apply_scaler(x_eval, mean, scale), coef)
        train_resid_std = residual_norm[:eval_lo].std(dim=0, unbiased=False).clamp_min(1e-6)
        pred, extra = apply_residual_delta(baseline[eval_lo:eval_hi], delta, std, config.alpha, config.clip_multiple, train_resid_std)
        cand = sample_mae(pred, target[eval_lo:eval_hi], mask[eval_lo:eval_hi], std)
        base = sample_mae(baseline[eval_lo:eval_hi], target[eval_lo:eval_hi], mask[eval_lo:eval_hi], std)
        fold_rows.append({"fold": fold_id, "mae": float(cand.mean()), "baseline_mae": float(base.mean()), "delta": float(cand.mean() - base.mean()), **extra})
        cand_all.append(cand)
        base_all.append(base)
    cand_t = torch.cat(cand_all)
    base_t = torch.cat(base_all)
    return {
        "name": config.name,
        **asdict(config),
        "fold_mae": float(cand_t.mean()),
        "fold_baseline_mae": float(base_t.mean()),
        "fold_delta": float(cand_t.mean() - base_t.mean()),
        "fold_wins": int(sum(r["delta"] < 0 for r in fold_rows)),
        "fold_rows": fold_rows,
    }


def prepare_ridge_folds(
    cache: Mapping[str, Any],
    starts: torch.Tensor,
    baseline: torch.Tensor,
    std: torch.Tensor,
    series: torch.Tensor,
    folds: Sequence[tuple[int, int, int]],
) -> list[dict[str, Any]]:
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.bool)
    residual_norm = (target - baseline) / std.view(1, 1, -1)
    prepared = []
    for fold_id, (_, eval_lo, eval_hi) in enumerate(folds):
        train_slice = {
            **cache,
            "targets": target[:eval_lo],
            "target_masks": mask[:eval_lo],
            "prediction_stack": cache["prediction_stack"][:eval_lo],
        }
        x_train, _, _ = build_feature_tensor(train_slice, starts[:eval_lo], baseline[:eval_lo], std, series, init_residuals_norm=None)
        y_train, m_train = flattened_targets(train_slice, baseline[:eval_lo], std)
        x_train = x_train[m_train]
        y_train = y_train[m_train]
        mean, scale = fit_scaler(x_train)
        x_train = apply_scaler(x_train, mean, scale)
        eval_slice = {
            **cache,
            "targets": target[eval_lo:eval_hi],
            "target_masks": mask[eval_lo:eval_hi],
            "prediction_stack": cache["prediction_stack"][eval_lo:eval_hi],
        }
        x_eval, _, _ = build_feature_tensor(
            eval_slice,
            starts[eval_lo:eval_hi],
            baseline[eval_lo:eval_hi],
            std,
            series,
            init_residuals_norm=residual_norm[:eval_lo],
        )
        prepared.append(
            {
                "fold": fold_id,
                "eval_lo": eval_lo,
                "eval_hi": eval_hi,
                "x_train": x_train,
                "y_train": y_train,
                "x_eval": apply_scaler(x_eval, mean, scale),
                "baseline_eval": baseline[eval_lo:eval_hi],
                "target_eval": target[eval_lo:eval_hi],
                "mask_eval": mask[eval_lo:eval_hi],
                "base_mae": sample_mae(baseline[eval_lo:eval_hi], target[eval_lo:eval_hi], mask[eval_lo:eval_hi], std),
                "train_resid_std": residual_norm[:eval_lo].std(dim=0, unbiased=False).clamp_min(1e-6),
            }
        )
    return prepared


def evaluate_ridge_grid_cached(
    prepared_folds: Sequence[Mapping[str, Any]],
    std: torch.Tensor,
    configs: Sequence[RidgeConfig],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ridge_values = sorted({float(c.ridge) for c in configs})
    delta_cache: dict[tuple[int, float], torch.Tensor] = {}
    for fold in prepared_folds:
        for ridge in ridge_values:
            coef = fit_ridge(fold["x_train"], fold["y_train"], ridge)
            delta_cache[(int(fold["fold"]), ridge)] = predict_linear(fold["x_eval"], coef)

    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for config in configs:
        cand_all = []
        base_all = []
        fold_rows = []
        clips = []
        mags = []
        for fold in prepared_folds:
            delta = delta_cache[(int(fold["fold"]), float(config.ridge))]
            pred, extra = apply_residual_delta(
                fold["baseline_eval"],
                delta,
                std,
                config.alpha,
                config.clip_multiple,
                fold["train_resid_std"],
            )
            cand = sample_mae(pred, fold["target_eval"], fold["mask_eval"], std)
            base = fold["base_mae"]
            cand_all.append(cand)
            base_all.append(base)
            clips.append(float(extra["clip_frequency"]))
            mags.append(float(extra["mean_abs_applied_delta_norm"]))
            fold_rows.append({"fold": int(fold["fold"]), "mae": float(cand.mean()), "baseline_mae": float(base.mean()), "delta": float(cand.mean() - base.mean())})
        cand_t = torch.cat(cand_all)
        base_t = torch.cat(base_all)
        row = {
            "name": config.name,
            **asdict(config),
            "fold_mae": float(cand_t.mean()),
            "fold_baseline_mae": float(base_t.mean()),
            "fold_delta": float(cand_t.mean() - base_t.mean()),
            "fold_wins": int(sum(r["delta"] < 0 for r in fold_rows)),
            "mean_abs_applied_delta_norm": float(np.mean(mags)),
            "clip_frequency": float(np.mean(clips)),
            "fold_rows": fold_rows,
        }
        rows.append({k: v for k, v in row.items() if k != "fold_rows"})
        if best is None or float(row["fold_mae"]) < float(best["fold_mae"]):
            best = row
    assert best is not None
    return rows, best


def build_ridge_grid() -> list[RidgeConfig]:
    return [
        RidgeConfig(ridge, alpha, clip)
        for ridge in (1.0, 10.0, 100.0, 1000.0, 10000.0)
        for alpha in (0.05, 0.10, 0.25, 0.50)
        for clip in (0.25, 0.5, 1.0, 2.0, None)
    ]


class TinyResidualMlp(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def train_mlp_final(
    train_cache: Mapping[str, Any],
    val_cache: Mapping[str, Any],
    train_baseline: torch.Tensor,
    val_baseline: torch.Tensor,
    std: torch.Tensor,
    series: torch.Tensor,
    config: MlpConfig,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    set_seed(config.seed)
    train_starts = train_cache["absolute_window_starts"].to(torch.long)
    val_starts = val_cache["absolute_window_starts"].to(torch.long)
    residual_norm = (train_cache["targets"].to(torch.float32) - train_baseline) / std.view(1, 1, -1)
    x_all, names, _ = build_feature_tensor(train_cache, train_starts, train_baseline, std, series, init_residuals_norm=None)
    y_all, m_all = flattened_targets(train_cache, train_baseline, std)
    x_all = x_all[m_all]
    y_all = y_all[m_all]
    split = int(0.85 * x_all.shape[0])
    x_fit, y_fit = x_all[:split], y_all[:split]
    x_es, y_es = x_all[split:], y_all[split:]
    mean, scale = fit_scaler(x_fit)
    x_fit = apply_scaler(x_fit, mean, scale)
    x_es = apply_scaler(x_es, mean, scale)
    model = TinyResidualMlp(x_fit.shape[1], config.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loader = DataLoader(TensorDataset(x_fit, y_fit), batch_size=4096, shuffle=True)
    best_state = None
    best_loss = float("inf")
    bad = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = torch.nn.functional.smooth_l1_loss(pred, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            es_loss = torch.nn.functional.smooth_l1_loss(model(x_es.to(device)), y_es.to(device)).item()
        if es_loss < best_loss - 1e-6:
            best_loss = es_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= config.patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    x_val, _, _ = build_feature_tensor(val_cache, val_starts, val_baseline, std, series, init_residuals_norm=residual_norm)
    model.eval()
    outs = []
    with torch.no_grad():
        x_val = apply_scaler(x_val, mean, scale)
        for i in range(0, x_val.shape[0], 16384):
            outs.append(model(x_val[i : i + 16384].to(device)).cpu())
    delta = torch.cat(outs)
    pred, extra = apply_residual_delta(
        val_baseline,
        delta,
        std,
        config.alpha,
        config.clip_multiple,
        residual_norm.std(dim=0, unbiased=False).clamp_min(1e-6),
    )
    return pred, {"early_stop_loss": best_loss, "features": len(names), **extra}


def fixed_current_best_prediction(
    cache: Mapping[str, Any],
    train_cache_for_init: Mapping[str, Any],
    std: torch.Tensor,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    starts = cache["absolute_window_starts"].to(torch.long)
    if not bool(torch.all(starts[1:] > starts[:-1])):
        raise ValueError("Cache starts must be chronological")
    horizon = int(cache["forecast_horizon"])
    train_expert_err = per_location_abs_error(train_cache_for_init, std).mean(dim=(1, 2))
    val_expert_err = per_location_abs_error(cache, std).mean(dim=(1, 2))
    online_weights, online_extra = chronological_online_weights(
        starts=starts,
        expert_mae=val_expert_err,
        horizon=horizon,
        trial=ChronoTrial("ema", "ema_decay0.97_temp0.1", decay=0.97, temperature=0.1),
        train_mean_mae=train_expert_err.mean(dim=0),
        mode="ema",
    )
    static_weights, _, _ = load_static_winner_per_window(seed, cache, std, device)
    chrono_weights = 0.5 * static_weights + 0.5 * online_weights
    chrono_weights = chrono_weights / chrono_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    chrono_pred = weighted_forecast(fixed3_forecasts(cache), chrono_weights)

    train_hv_err_mean = per_location_abs_error(train_cache_for_init, std).mean(dim=0)
    val_hv_err = per_location_abs_error(cache, std)
    hv_weights, hv_extra = chronological_hv_weights(
        starts=starts,
        train_err_mean=train_hv_err_mean,
        val_err=val_hv_err,
        horizon=horizon,
        trial=HvTrial("hv_ema", "hvema_lowrank1_decay0.95_temp0.1", mode="hv_lowrank", rank=1, decay=0.95, temperature=0.1),
    )
    hv_pred = predict_from_hv_weights(fixed3_forecasts(cache), hv_weights)
    pred = 0.25 * chrono_pred + 0.75 * hv_pred
    return pred, {
        "chrono_num_updates": online_extra.get("num_updates"),
        "hv_num_updates": hv_extra.get("num_updates"),
        **{f"mean_weight_{FIXED3[i]}": float(hv_weights[..., i].mean()) for i in range(3)},
    }


def summarize_method(
    cache: Mapping[str, Any],
    std: torch.Tensor,
    pred: torch.Tensor,
    baseline: torch.Tensor,
    name: str,
    seed: int,
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    m = metrics(cache, std, pred)
    b = metrics(cache, std, baseline)
    boot = paired_bootstrap(m["per_window_mae"], b["per_window_mae"], seed=seed, samples=5000)
    return {
        "method": name,
        "seed": seed,
        "mae": m["mae"],
        "mse": m["mse"],
        "baseline_mae": b["mae"],
        "baseline_mse": b["mse"],
        "absolute_improvement_mae": b["mae"] - m["mae"],
        "percent_improvement_mae": 100.0 * (b["mae"] - m["mae"]) / b["mae"],
        **boot,
        **extra,
    }


def aggregate_seed_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    out = []
    for name, group in grouped.items():
        maes = torch.tensor([float(r["mae"]) for r in group])
        mses = torch.tensor([float(r["mse"]) for r in group])
        out.append(
            {
                "method": name,
                "seeds": len(group),
                "mae_mean": float(maes.mean()),
                "mae_std": float(maes.std(unbiased=False)) if len(group) > 1 else 0.0,
                "mse_mean": float(mses.mean()),
                "mse_std": float(mses.std(unbiased=False)) if len(group) > 1 else 0.0,
                "improvement_vs_0.363642": BASELINE_MAE - float(maes.mean()),
                "percent_improvement_vs_0.363642": 100.0 * (BASELINE_MAE - float(maes.mean())) / BASELINE_MAE,
            }
        )
    return sorted(out, key=lambda r: float(r["mae_mean"]))


def aggregate_postrun_audit(out_dir: Path, validation_rows: Sequence[Mapping[str, Any]], location_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    methods = sorted({str(r["method"]) for r in validation_rows})
    seeds = sorted({int(r["seed"]) for r in validation_rows})
    aggregate_ci = []
    for method in methods:
        if method == "baseline_current_best":
            cand = torch.cat([torch.load(out_dir / "per_window" / f"baseline_seed{seed}.pt", map_location="cpu", weights_only=False) for seed in seeds])
            base = cand.clone()
        else:
            paths = [out_dir / "per_window" / f"{method}_seed{seed}.pt" for seed in seeds]
            if not all(path.exists() for path in paths):
                continue
            cand = torch.cat([torch.load(path, map_location="cpu", weights_only=False) for path in paths])
            base = torch.cat([torch.load(out_dir / "per_window" / f"baseline_seed{seed}.pt", map_location="cpu", weights_only=False) for seed in seeds])
        aggregate_ci.append({"method": method, "mean_per_window_mae": float(cand.mean()), "baseline_mean_per_window_mae": float(base.mean()), **paired_bootstrap(cand, base, seed=20260812, samples=10000)})

    hv_groups: dict[tuple[str, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in location_rows:
        method = str(row["method"]).rsplit("_seed", 1)[0]
        hv_groups[(method, int(row["horizon"]), int(row["variable"]))].append(row)
    hv_agg = []
    for (method, horizon, variable), group in sorted(hv_groups.items()):
        hv_agg.append(
            {
                "method": method,
                "horizon": horizon,
                "variable": variable,
                "mae_mean": float(np.mean([float(r["mae"]) for r in group])),
                "baseline_mae_mean": float(np.mean([float(r["baseline_mae"]) for r in group])),
                "delta_vs_baseline_mean": float(np.mean([float(r["delta_vs_baseline"]) for r in group])),
                "seeds": len(group),
            }
        )
    write_csv(out_dir / "per_horizon_variable_mae_aggregate.csv", hv_agg)

    axis_rows = []
    for method in sorted({r["method"] for r in hv_agg}):
        vals = [r for r in hv_agg if r["method"] == method]
        for axis, key in (("horizon", "horizon"), ("variable", "variable")):
            for idx in sorted({int(r[key]) for r in vals}):
                sub = [r for r in vals if int(r[key]) == idx]
                axis_rows.append(
                    {
                        "method": method,
                        "axis": axis,
                        "index": idx,
                        "mae_mean": float(np.mean([float(r["mae_mean"]) for r in sub])),
                        "baseline_mae_mean": float(np.mean([float(r["baseline_mae_mean"]) for r in sub])),
                        "delta_vs_baseline_mean": float(np.mean([float(r["delta_vs_baseline_mean"]) for r in sub])),
                    }
                )
    write_csv(out_dir / "per_axis_mae_aggregate.csv", axis_rows)
    worst = []
    for method in sorted({r["method"] for r in hv_agg}):
        worst.append(max([r for r in hv_agg if r["method"] == method], key=lambda r: float(r["delta_vs_baseline_mean"])))

    correction = []
    for method in sorted(m for m in methods if m != "baseline_current_best"):
        vals = [r for r in validation_rows if r["method"] == method]

        def mean_field(name: str) -> float | None:
            xs = [float(v[name]) for v in vals if name in v and v[name] not in ("", None)]
            return float(np.mean(xs)) if xs else None

        correction.append(
            {
                "method": method,
                "mean_abs_correction": mean_field("mean_abs_correction"),
                "mean_abs_applied_delta_norm": mean_field("mean_abs_applied_delta_norm"),
                "clip_frequency": mean_field("clip_frequency"),
            }
        )
    write_csv(out_dir / "aggregate_bootstrap_ci.csv", aggregate_ci)
    return {
        "aggregate_bootstrap_ci": aggregate_ci,
        "worst_average_horizon_variable_regression_by_method": worst,
        "correction_magnitude_summary": correction,
    }


def make_report(out_dir: Path, report: Mapping[str, Any]) -> None:
    winner = report["best_validation_method"]
    lines = [
        "# Causal Residual Correction COSTAR-TS",
        "",
        "## Protocol",
        "",
        "- Dataset: ETTh1 router-train `20-60%`, validation `60-80%`.",
        "- Test cache was not loaded or evaluated.",
        "- Frozen baseline: `hybrid_chrono_hvema_lowrank1_decay0.95_temp0.1_alpha0.75`.",
        "- Correction hyperparameters were selected on chronological router-train folds only.",
        "- Online residual updates use only windows satisfying `old_start + horizon <= current_start`.",
        "",
        "## Results",
        "",
        f"- Baseline reproduction mean MAE: `{report['baseline_reproduction']['mae_mean']:.6f}`.",
        f"- Best validation method: `{winner['method']}`.",
        f"- Best MAE / MSE: `{winner['mae_mean']:.6f}` / `{winner['mse_mean']:.6f}`.",
        f"- Improvement vs `0.363642`: `{winner['improvement_vs_0.363642']:.6f}` MAE ({winner['percent_improvement_vs_0.363642']:.3f}%).",
        f"- Strong target `<= 0.3619`: `{bool(winner['mae_mean'] <= STRONG_TARGET)}`.",
        f"- Exceptional target `<= 0.3600`: `{bool(winner['mae_mean'] <= EXCEPTIONAL_TARGET)}`.",
        f"- Experiment 1 selected config: `{report['experiment1']['selected_config']['name']}`.",
        f"- Experiment 2 ridge selected config: `{report['experiment2']['ridge_selected_config']['name']}`.",
        f"- MLP run: `{report['experiment2']['mlp_ran']}`.",
        f"- Aggregate paired bootstrap CI for winner: `[{report['best_validation_method']['aggregate_ci95_low']:.6f}, {report['best_validation_method']['aggregate_ci95_high']:.6f}]`.",
        "",
        "## Leakage Checks",
        "",
        f"- Causal update assertions passed: `{report['leakage_checks']['causal_assertions_passed']}`.",
        f"- Test cache loaded: `{report['leakage_checks']['test_cache_loaded']}`.",
        f"- Long-history summaries end before forecast start: `{report['leakage_checks']['history_summaries_causal']}`.",
        "",
        "## Reproduce",
        "",
        "```powershell",
        report["reproduce_command"],
        "```",
    ]
    (out_dir / "experiment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", default="cache/costarts_walkforward/router_train_20_60_cache.pt")
    parser.add_argument("--val-cache", default="cache/costarts_walkforward/router_val_60_80_cache.pt")
    parser.add_argument("--normalizer-checkpoint", default="checkpoints/costarts_walkforward/final_60/DLinear/best_expert.pt")
    parser.add_argument("--dataset-dir", default="datasets/ETTh1")
    parser.add_argument("--out-dir", default="experiments/residual_correction_costar")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--force-mlp", action="store_true")
    args = parser.parse_args()

    start_time = time.time()
    for path in (args.train_cache, args.val_cache, args.normalizer_checkpoint):
        refuse_test_path(path)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_window").mkdir(exist_ok=True)

    train_cache = load_cache(ROOT / args.train_cache, "router_train_20_60")
    val_cache = load_cache(ROOT / args.val_cache, "router_val_60_80")
    std = load_std(ROOT / args.normalizer_checkpoint, int(val_cache["num_features"]))
    series = load_series_prefix(ROOT / args.dataset_dir)
    device = torch.device(args.device)

    train_starts = train_cache["absolute_window_starts"].to(torch.long)
    val_starts = val_cache["absolute_window_starts"].to(torch.long)
    if int(val_starts[-1]) + int(val_cache["forecast_horizon"]) > 11520:
        raise ValueError("Validation target window exceeds 60-80 split end")
    for start in (int(train_starts[0]), int(val_starts[0]), int(val_starts[-1])):
        for scale in SCALES:
            assert_history_available(series, start, scale)

    folds = train_folds(int(train_cache["num_windows"]))
    fold_rows_json = [{"fold": i, "train_lo": lo, "eval_lo": evlo, "eval_hi": evhi} for i, (lo, evlo, evhi) in enumerate(folds)]

    baseline_by_seed: dict[int, torch.Tensor] = {}
    val_baseline_rows = []
    for seed in SEEDS:
        pred, extra = fixed_current_best_prediction(val_cache, train_cache, std, seed, device)
        baseline_by_seed[seed] = pred
        row = summarize_method(val_cache, std, pred, pred, "baseline_current_best", seed, extra)
        val_baseline_rows.append(row)
        torch.save(metrics(val_cache, std, pred)["per_window_mae"], out_dir / "per_window" / f"baseline_seed{seed}.pt")

    baseline_repro = aggregate_seed_rows(val_baseline_rows)[0]
    # Use seed 7 baseline on router-train for deterministic model selection.
    train_baseline, _ = fixed_current_best_prediction(train_cache, train_cache, std, 7, device)
    val_baseline = baseline_by_seed[7]
    train_target = train_cache["targets"].to(torch.float32)
    train_mask = train_cache["target_masks"].to(torch.bool)
    val_target = val_cache["targets"].to(torch.float32)
    horizon = int(train_cache["forecast_horizon"])

    bias_rows, best_bias = evaluate_bias_grid_cached(
        train_starts,
        train_baseline,
        train_target,
        train_mask,
        std,
        build_bias_grid(),
        folds,
    )
    write_csv(out_dir / "experiment1_bias_fold_leaderboard.csv", sorted(bias_rows, key=lambda r: float(r["fold_mae"])))
    write_json(out_dir / "experiment1_bias_selected_folds.json", best_bias)

    best_bias_config = BiasConfig(
        structure=str(best_bias["structure"]),
        decay=float(best_bias["decay"]),
        alpha=float(best_bias["alpha"]),
        clip_multiple=None if best_bias["clip_multiple"] is None else float(best_bias["clip_multiple"]),
        min_count=int(best_bias["min_count"]),
    )

    validation_rows = list(val_baseline_rows)
    axis_rows = []
    location_rows = []
    bias_seed_preds: dict[int, torch.Tensor] = {}
    for seed in SEEDS:
        baseline = baseline_by_seed[seed]
        residual_train = train_cache["targets"].to(torch.float32) - train_baseline
        clip_std = residual_std_by_structure(residual_train, best_bias_config.structure)
        pred, extra = causal_bias_correct(
            val_starts,
            baseline,
            val_target,
            int(val_cache["forecast_horizon"]),
            best_bias_config,
            init_residuals=residual_train,
            clip_std=clip_std,
        )
        bias_seed_preds[seed] = pred
        name = "experiment1_bias"
        validation_rows.append(summarize_method(val_cache, std, pred, baseline, name, seed, extra))
        axis_rows.extend(per_axis_rows(val_cache, std, pred, f"{name}_seed{seed}"))
        location_rows.extend(per_location_rows(val_cache, std, pred, baseline, f"{name}_seed{seed}"))
        torch.save(metrics(val_cache, std, pred)["per_window_mae"], out_dir / "per_window" / f"{name}_seed{seed}.pt")

    prepared_ridge_folds = prepare_ridge_folds(train_cache, train_starts, train_baseline, std, series, folds)
    ridge_rows, best_ridge = evaluate_ridge_grid_cached(prepared_ridge_folds, std, build_ridge_grid())
    write_csv(out_dir / "experiment2_ridge_fold_leaderboard.csv", sorted(ridge_rows, key=lambda r: float(r["fold_mae"])))
    write_json(out_dir / "experiment2_ridge_selected_folds.json", best_ridge)

    best_ridge_config = RidgeConfig(
        ridge=float(best_ridge["ridge"]),
        alpha=float(best_ridge["alpha"]),
        clip_multiple=None if best_ridge["clip_multiple"] is None else float(best_ridge["clip_multiple"]),
    )
    train_resid_norm = (train_cache["targets"].to(torch.float32) - train_baseline) / std.view(1, 1, -1)
    x_train, feature_names, stat_extra = build_feature_tensor(train_cache, train_starts, train_baseline, std, series, init_residuals_norm=None)
    y_train, m_train = flattened_targets(train_cache, train_baseline, std)
    x_fit = x_train[m_train]
    y_fit = y_train[m_train]
    feat_mean, feat_scale = fit_scaler(x_fit)
    coef = fit_ridge(apply_scaler(x_fit, feat_mean, feat_scale), y_fit, best_ridge_config.ridge)
    write_json(
        out_dir / "experiment2_feature_manifest.json",
        {"num_features": len(feature_names), "feature_names": feature_names, "residual_stat_extra": stat_extra},
    )
    ridge_seed_preds: dict[int, torch.Tensor] = {}
    for seed in SEEDS:
        baseline = baseline_by_seed[seed]
        x_val, _, val_stat_extra = build_feature_tensor(val_cache, val_starts, baseline, std, series, init_residuals_norm=train_resid_norm)
        delta = predict_linear(apply_scaler(x_val, feat_mean, feat_scale), coef)
        pred, extra = apply_residual_delta(
            baseline,
            delta,
            std,
            best_ridge_config.alpha,
            best_ridge_config.clip_multiple,
            train_resid_norm.std(dim=0, unbiased=False).clamp_min(1e-6),
        )
        ridge_seed_preds[seed] = pred
        name = "experiment2_ridge"
        validation_rows.append(summarize_method(val_cache, std, pred, baseline, name, seed, {**extra, **val_stat_extra}))
        axis_rows.extend(per_axis_rows(val_cache, std, pred, f"{name}_seed{seed}"))
        location_rows.extend(per_location_rows(val_cache, std, pred, baseline, f"{name}_seed{seed}"))
        torch.save(metrics(val_cache, std, pred)["per_window_mae"], out_dir / "per_window" / f"{name}_seed{seed}.pt")

    ridge_useful = bool(float(best_ridge["fold_delta"]) < -1e-5 and int(best_ridge["fold_wins"]) >= 3)
    mlp_ran = bool(args.force_mlp or ridge_useful)
    mlp_rows = []
    if mlp_ran:
        for seed in SEEDS:
            config = MlpConfig(seed=seed, alpha=best_ridge_config.alpha, clip_multiple=best_ridge_config.clip_multiple)
            pred, extra = train_mlp_final(train_cache, val_cache, train_baseline, baseline_by_seed[seed], std, series, config, device)
            name = "experiment2_mlp"
            validation_rows.append(summarize_method(val_cache, std, pred, baseline_by_seed[seed], name, seed, extra))
            axis_rows.extend(per_axis_rows(val_cache, std, pred, f"{name}_seed{seed}"))
            location_rows.extend(per_location_rows(val_cache, std, pred, baseline_by_seed[seed], f"{name}_seed{seed}"))
            mlp_rows.append({"seed": seed, **asdict(config), **extra})
            torch.save(metrics(val_cache, std, pred)["per_window_mae"], out_dir / "per_window" / f"{name}_seed{seed}.pt")
    if mlp_rows:
        write_csv(out_dir / "experiment2_mlp_runs.csv", mlp_rows)

    seed_summary = aggregate_seed_rows(validation_rows)
    audit = aggregate_postrun_audit(out_dir, validation_rows, location_rows)
    ci_by_method = {row["method"]: row for row in audit["aggregate_bootstrap_ci"]}
    for row in seed_summary:
        ci = ci_by_method.get(row["method"])
        if ci is not None:
            row.update(
                {
                    "aggregate_mean_diff_candidate_minus_baseline": ci["mean_diff_candidate_minus_baseline"],
                    "aggregate_ci95_low": ci["ci95_low"],
                    "aggregate_ci95_high": ci["ci95_high"],
                    "aggregate_ci_excludes_zero": ci["ci_excludes_zero"],
                }
            )
    write_csv(out_dir / "validation_per_seed_results.csv", validation_rows)
    write_csv(out_dir / "validation_seed_summary.csv", seed_summary)
    write_csv(out_dir / "per_axis_mae.csv", axis_rows)
    write_csv(out_dir / "per_horizon_variable_mae.csv", location_rows)

    best_method = seed_summary[0]
    worst_regressions = sorted(location_rows, key=lambda r: float(r["delta_vs_baseline"]), reverse=True)[:10]
    report = {
        "baseline_reference": {"mae": BASELINE_MAE, "mse": BASELINE_MSE, "name": "hybrid_chrono_hvema_lowrank1_decay0.95_temp0.1_alpha0.75"},
        "baseline_reproduction": baseline_repro,
        "folds": fold_rows_json,
        "experiment1": {"selected_config": {"name": best_bias_config.name, **asdict(best_bias_config)}, "train_fold_selection": best_bias},
        "experiment2": {
            "ridge_selected_config": {"name": best_ridge_config.name, **asdict(best_ridge_config)},
            "ridge_train_fold_selection": best_ridge,
            "ridge_useful_oof_signal": ridge_useful,
            "mlp_ran": mlp_ran,
        },
        "validation_seed_summary": seed_summary,
        "best_validation_method": best_method,
        "worst_horizon_variable_regressions": worst_regressions,
        **audit,
        "leakage_checks": {
            "causal_assertions_passed": True,
            "test_cache_loaded": False,
            "history_summaries_causal": True,
            "validation_start": int(val_starts[0]),
            "validation_end_exclusive": 11520,
        },
        "target_reached": bool(float(best_method["mae_mean"]) <= STRONG_TARGET),
        "exceptional_target_reached": bool(float(best_method["mae_mean"]) <= EXCEPTIONAL_TARGET),
        "runtime_sec": time.time() - start_time,
        "reproduce_command": f"python experiments\\residual_correction_costar\\run_residual_correction_experiments.py --device {args.device}",
    }
    write_json(out_dir / "final_report.json", report)
    make_report(out_dir, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
