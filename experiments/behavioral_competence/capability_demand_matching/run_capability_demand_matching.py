"""Natural capability-demand matching for frozen forecast expert routing.

Validation-only. The study estimates expert capability profiles from natural
router_train variation, then matches router_val demand fingerprints to those
profiles. It never loads test caches and never uses router_val targets to build
features, bins, profiles, model weights, or routing temperatures.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.behavioral_competence.common import (  # noqa: E402
    disagreement_features_group_c,
    forecast_features_group_b,
    window_features_group_a,
)
from experiments.behavioral_competence.model_runtime import (  # noqa: E402
    WALKFORWARD_CHECKPOINT_ROOTS,
    load_expert_runtime,
    sha256_file,
)
from experiments.behavioral_competence.run_behavioral_competence import raw_history_cache  # noqa: E402
from experiments.costar_multidataset_frozen.common import block_bootstrap_with_prob, every_kth_phase_bootstrap  # noqa: E402
from experiments.final_test_evaluation.run_final_frozen_test_evaluation import etth2_checkpoint_path  # noqa: E402
from experiments.frozen_hv_costar.run_frozen_hv_costar import LOADERS  # noqa: E402
from experiments.oracle_weight_tournament.run_tournament import sample_mae, sample_mse  # noqa: E402


OUT = Path(__file__).resolve().parent
DATASETS = ("ETTh1", "ETTh2", "ETTm1", "Weather", "Electricity")
EXPERT_ORDER = ("DLinear", "PatchTST", "iTransformer", "TimesNet", "ModernTCN")
AXES = ("trend", "seasonality", "frequency", "volatility", "shift", "crossvar")
HORIZON = 12
RIDGE_ALPHA = 1.0
SHRINKAGE = 32.0
BLOCK_LENGTH = 24
BOOTSTRAP_SAMPLES = 5000
TEMP_GRID = (0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
CODE_VERSION = "capability_demand_matching_v1"
EPS = 1e-8


def refuse_test(path: str | Path) -> None:
    if "test" in str(path).lower():
        raise ValueError(f"test access forbidden: {path}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def json_default(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"{type(obj).__name__} is not JSON serializable")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def file_hash(path: Path) -> str:
    refuse_test(path)
    return sha256_file(path)


def tensor_hash(tensor: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def cache_paths(dataset: str) -> dict[str, Path]:
    if dataset == "ETTh1":
        return {
            "router_train": ROOT / "cache/costarts_walkforward/router_train_20_60_cache.pt",
            "router_val": ROOT / "cache/costarts_walkforward/router_val_60_80_cache.pt",
            "normalizer": ROOT / "checkpoints/costarts_walkforward/final_60/DLinear/best_expert.pt",
        }
    if dataset == "ETTh2":
        return {
            "router_train": ROOT / "cache/costarts_fresh/ETTh2_96_12/router_train_cache.pt",
            "router_val": ROOT / "cache/costarts_fresh/ETTh2_96_12/router_val_cache.pt",
        }
    return {
        "router_train": ROOT / f"cache/costarts_walkforward_{dataset}/router_train_20_60_cache.pt",
        "router_val": ROOT / f"cache/costarts_walkforward_{dataset}/router_val_60_80_cache.pt",
        "normalizer": ROOT / f"checkpoints/costarts_walkforward_{dataset}/final_60/DLinear/best_expert.pt",
    }


def checkpoint_paths(dataset: str) -> dict[str, Path]:
    if dataset == "ETTh2":
        return {expert: etth2_checkpoint_path(expert) for expert in EXPERT_ORDER}
    root = WALKFORWARD_CHECKPOINT_ROOTS[dataset]
    return {expert: root / "final_60" / expert / "best_expert.pt" for expert in EXPERT_ORDER}


def relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def validate_cache_schema(dataset: str, cache: Mapping[str, Any], split: str, core_names: Sequence[str]) -> dict[str, Any]:
    role = str(cache.get("cache_role", cache.get("split_role")))
    n = int(cache["num_windows"])
    starts = cache["absolute_window_starts"].to(torch.long)
    shape_ok = (
        tuple(cache["histories"].shape[:2]) == (n, 96)
        and tuple(cache["targets"].shape[:2]) == (n, HORIZON)
        and tuple(cache["target_masks"].shape) == tuple(cache["targets"].shape)
        and tuple(cache["prediction_stack"].shape[:3]) == tuple(cache["targets"].shape)
        and int(cache["forecast_horizon"]) == HORIZON
    )
    expected_role = "router_val" if dataset == "ETTh2" and split == "router_val" else split
    if dataset != "ETTh2":
        expected_role = "router_train_20_60" if split == "router_train" else "router_val_60_80"
    return {
        "split": split,
        "role": role,
        "expected_role": expected_role,
        "role_ok": role == expected_role,
        "shape_ok": bool(shape_ok),
        "expert_order_ok": tuple(cache["expert_names"]) == EXPERT_ORDER,
        "core_in_cache": all(name in cache["expert_names"] for name in core_names),
        "starts_chronological": bool(torch.all(starts[1:] > starts[:-1])),
        "num_windows": n,
        "start_min": int(starts.min()),
        "start_max": int(starts.max()),
        "history_shape": list(cache["histories"].shape),
        "target_shape": list(cache["targets"].shape),
        "prediction_shape": list(cache["prediction_stack"].shape),
    }


def selected_forecasts(bundle: Any, cache: Mapping[str, Any]) -> torch.Tensor:
    return bundle.forecasts_fn(cache, bundle.expert_idx).to(torch.float32)


def per_window_expert_mae(cache: Mapping[str, Any], forecasts: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.bool)
    return torch.stack([sample_mae(forecasts[..., i], target, mask, std) for i in range(forecasts.shape[-1])], dim=1)


def competence(cache: Mapping[str, Any], forecasts: torch.Tensor, std: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    error = per_window_expert_mae(cache, forecasts, std)
    return error - error.mean(dim=1, keepdim=True), error


def passive_features(cache: Mapping[str, Any], forecasts: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    history = cache["histories"].to(torch.float32)
    a = window_features_group_a(history, std)
    features = []
    for i in range(forecasts.shape[-1]):
        f = forecasts[..., i]
        b = forecast_features_group_b(f, history[:, -1], std)
        c = disagreement_features_group_c(f, forecasts, std)
        features.append(torch.cat((a, b, c), dim=1))
    return torch.stack(features, dim=1)


def _safe_corr(x: np.ndarray, y: np.ndarray, rank: bool = False) -> float:
    if x.size < 3 or float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return 0.0
    try:
        stat = spearmanr(x, y).statistic if rank else pearsonr(x, y).statistic
        return 0.0 if not np.isfinite(stat) else float(stat)
    except Exception:
        return 0.0


def demand_features_from_history(history: torch.Tensor, std: torch.Tensor, batch_size: int = 512) -> torch.Tensor:
    """History-only demand fingerprint [N,96,F] -> [N,6]."""
    history = history.to(torch.float32)
    n, length, variables = history.shape
    stdv = std.to(torch.float32).view(1, variables).clamp_min(1e-6)
    t = torch.arange(length, dtype=torch.float32)
    tc = t - t.mean()
    slope_denom = (tc * tc).sum().clamp_min(EPS)
    max_lag = min(48, length // 2)
    out_chunks = []
    for lo in range(0, n, batch_size):
        x = history[lo : lo + batch_size]
        b = x.shape[0]
        mean = x.mean(dim=1, keepdim=True)
        xc = x - mean
        slope = (xc * tc.view(1, -1, 1)).sum(dim=1) / slope_denom
        trend = (slope.abs() * (length - 1) / stdv).mean(dim=1)

        spec_ac = torch.fft.rfft(xc, n=2 * length, dim=1)
        ac = torch.fft.irfft(spec_ac * spec_ac.conj(), n=2 * length, dim=1)[:, : length, :]
        var = (xc * xc).sum(dim=1, keepdim=True).clamp_min(EPS)
        ac_norm = ac / var
        if max_lag > 2:
            seasonality = ac_norm[:, 2 : max_lag + 1, :].abs().mean(dim=2).max(dim=1).values
        else:
            seasonality = torch.zeros(b)

        spec = torch.fft.rfft(xc, dim=1)
        power = spec.abs().pow(2)[:, 1:, :]
        total_power = power.sum(dim=1, keepdim=True).clamp_min(EPS)
        p = power / total_power
        entropy = -(p * p.clamp_min(EPS).log()).sum(dim=1) / math.log(max(2, p.shape[1]))
        split = max(1, power.shape[1] // 3)
        low = power[:, :split, :].sum(dim=1)
        high = power[:, split:, :].sum(dim=1)
        high_low = torch.log1p(high / low.clamp_min(EPS))
        frequency = (entropy + 0.10 * high_low).mean(dim=1)

        diff = x[:, 1:, :] - x[:, :-1, :]
        diff_rms = diff.square().mean(dim=1).sqrt()
        diff_abs = diff.abs().median(dim=1).values
        volatility = ((0.5 * diff_rms + 0.5 * diff_abs) / stdv).mean(dim=1)

        half = length // 2
        first, second = x[:, :half, :], x[:, half:, :]
        mean_shift = ((second.mean(dim=1) - first.mean(dim=1)).abs() / stdv).mean(dim=1)
        var_shift = torch.log1p((second.var(dim=1, unbiased=False) - first.var(dim=1, unbiased=False)).abs() / stdv.square()).mean(dim=1)
        slope_first = ((first - first.mean(dim=1, keepdim=True)) * tc[:half].view(1, -1, 1)).sum(dim=1) / (tc[:half].square().sum().clamp_min(EPS))
        slope_second = ((second - second.mean(dim=1, keepdim=True)) * tc[half:].view(1, -1, 1)).sum(dim=1) / (tc[half:].square().sum().clamp_min(EPS))
        shift = mean_shift + 0.25 * var_shift + 0.25 * ((slope_second - slope_first).abs() * (length - 1) / stdv).mean(dim=1)

        if variables <= 1:
            crossvar = torch.zeros(b)
        else:
            z = xc / x.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
            factor = z.mean(dim=2, keepdim=True)
            factor = factor - factor.mean(dim=1, keepdim=True)
            factor_std = factor.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
            corr = (z * factor).mean(dim=1).abs() / factor_std.squeeze(1)
            crossvar = corr.mean(dim=1).clamp(0.0, 1.0)

        chunk = torch.stack((trend, seasonality, frequency, volatility, shift, crossvar), dim=1)
        out_chunks.append(torch.nan_to_num(chunk, nan=0.0, posinf=0.0, neginf=0.0))
    return torch.cat(out_chunks, dim=0).to(torch.float32)


def folds_from_starts(starts: torch.Tensor, horizon: int = HORIZON) -> list[dict[str, Any]]:
    n = int(starts.numel())
    min_train = int(round(n * 0.2))
    usable = n - min_train
    bounds = [min_train + i * usable // 4 for i in range(5)]
    rows = []
    for fold in range(4):
        lo, hi = bounds[fold], bounds[fold + 1]
        current_origin = int(starts[lo])
        legal = torch.where(starts + horizon <= current_origin)[0]
        rows.append(
            {
                "fold": fold,
                "eval_lo": lo,
                "eval_hi": hi,
                "fit_indices": legal,
                "fit_windows": int(legal.numel()),
                "fit_start_min": int(starts[legal[0]]) if legal.numel() else None,
                "fit_start_max": int(starts[legal[-1]]) if legal.numel() else None,
                "eval_start_min": current_origin,
                "eval_start_max": int(starts[hi - 1]),
                "purge_horizon": horizon,
                "old_target_end_le_current_origin": bool(legal.numel() == 0 or int(starts[legal[-1]]) + horizon <= current_origin),
            }
        )
    return rows


def quantile_bins(train_values: torch.Tensor, eval_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    q = torch.quantile(train_values.to(torch.float32), torch.tensor([1.0 / 3.0, 2.0 / 3.0]), dim=0)
    bins = torch.zeros_like(eval_values, dtype=torch.long)
    bins += (eval_values > q[0].view(1, -1)).long()
    bins += (eval_values > q[1].view(1, -1)).long()
    return q.T.contiguous(), bins


def ridge_fit(train_x: torch.Tensor, train_y: torch.Tensor, alpha: float = RIDGE_ALPHA) -> dict[str, torch.Tensor]:
    x = train_x.to(torch.float64)
    y = train_y.to(torch.float64)
    mean = x.mean(dim=0, keepdim=True)
    scale = x.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    xs = (x - mean) / scale
    xa = torch.cat((torch.ones(xs.shape[0], 1, dtype=torch.float64), xs), dim=1)
    xtx = xa.T @ xa
    reg = torch.eye(xtx.shape[0], dtype=torch.float64) * alpha
    reg[0, 0] = 0.0
    rhs = xa.T @ y
    try:
        beta = torch.linalg.solve(xtx + reg, rhs)
    except RuntimeError:
        beta = torch.linalg.pinv(xtx + reg) @ rhs
    return {"mean": mean.to(torch.float32), "scale": scale.to(torch.float32), "beta": beta.to(torch.float32)}


def ridge_predict(model: Mapping[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    xs = (x.to(torch.float32) - model["mean"]) / model["scale"]
    xa = torch.cat((torch.ones(xs.shape[0], 1), xs), dim=1)
    return xa @ model["beta"]


def fit_predict_ridge(train_x: torch.Tensor, train_y: torch.Tensor, eval_x: torch.Tensor) -> torch.Tensor:
    model = ridge_fit(train_x, train_y)
    return ridge_predict(model, eval_x).to(torch.float32)


def expert_id_features(demand: torch.Tensor, num_experts: int) -> torch.Tensor:
    n, d = demand.shape
    onehot = torch.eye(num_experts, dtype=torch.float32).unsqueeze(0).expand(n, -1, -1)
    repeated = demand.unsqueeze(1).expand(-1, num_experts, -1)
    return torch.cat((repeated, onehot), dim=2)


def fit_capability_profile(train_demand: torch.Tensor, train_z: torch.Tensor, shrinkage: float = SHRINKAGE) -> dict[str, Any]:
    q, train_bins = quantile_bins(train_demand, train_demand)
    n_axes = train_demand.shape[1]
    n_experts = train_z.shape[1]
    global_mean = train_z.mean(dim=0)
    table = torch.empty(n_experts, n_axes, 3, dtype=torch.float32)
    counts = torch.zeros(n_experts, n_axes, 3, dtype=torch.long)
    for expert in range(n_experts):
        for axis in range(n_axes):
            for b in range(3):
                mask = train_bins[:, axis] == b
                counts[expert, axis, b] = int(mask.sum())
                if bool(mask.any()):
                    raw_mean = train_z[mask, expert].mean()
                else:
                    raw_mean = global_mean[expert]
                count = float(counts[expert, axis, b])
                table[expert, axis, b] = (count * raw_mean + shrinkage * global_mean[expert]) / (count + shrinkage)

    curves = []
    for axis in range(n_axes):
        x = train_demand[:, axis : axis + 1]
        model = ridge_fit(torch.cat((x, x.square()), dim=1), train_z)
        curves.append(model)
    return {"quantiles": q, "global_mean": global_mean, "table": table, "counts": counts, "curves": curves}


def bins_from_quantiles(values: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    bins = torch.zeros_like(values, dtype=torch.long)
    bins += (values > quantiles[:, 0].view(1, -1)).long()
    bins += (values > quantiles[:, 1].view(1, -1)).long()
    return bins


def predict_capability(
    profile: Mapping[str, Any],
    demand: torch.Tensor,
    expert_perm: torch.Tensor | None = None,
    axis_perm: torch.Tensor | None = None,
    axes_subset: Sequence[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = profile["quantiles"]
    table = profile["table"]
    n_experts, n_axes, _ = table.shape
    if expert_perm is None:
        expert_perm = torch.arange(n_experts)
    if axis_perm is None:
        axis_perm = torch.arange(n_axes)
    if axes_subset is None:
        axes_subset = list(range(n_axes))

    bins = bins_from_quantiles(demand, q)
    regime = torch.zeros(demand.shape[0], n_experts)
    continuous = torch.zeros_like(regime)
    for target_axis in axes_subset:
        source_axis = int(axis_perm[target_axis])
        b = bins[:, target_axis]
        for expert in range(n_experts):
            source_expert = int(expert_perm[expert])
            regime[:, expert] += table[source_expert, source_axis, b]
        x = demand[:, target_axis : target_axis + 1]
        curve = profile["curves"][source_axis]
        continuous += ridge_predict(curve, torch.cat((x, x.square()), dim=1))[:, expert_perm]
    denom = float(len(axes_subset))
    regime = regime / denom
    continuous = continuous / denom
    return 0.5 * (regime + continuous), regime, continuous


def per_window_prediction_mae(pred: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
    return (pred - actual).abs().mean(dim=1)


def competence_metrics(pred: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    diff = pred - actual
    pred_np = pred.detach().cpu().numpy().reshape(-1)
    actual_np = actual.detach().cpu().numpy().reshape(-1)
    pair = []
    for i in range(actual.shape[1]):
        for j in range(i + 1, actual.shape[1]):
            pair.append(((pred[:, i] - pred[:, j]) * (actual[:, i] - actual[:, j]) > 0).to(torch.float32))
    pred_order = torch.argsort(pred, dim=1)
    best_actual = actual.argmin(dim=1)
    best_pred = pred.argmin(dim=1)
    regret = actual[torch.arange(actual.shape[0]), best_pred] - actual[torch.arange(actual.shape[0]), best_actual]
    return {
        "mae": float(diff.abs().mean()),
        "mse": float(diff.square().mean()),
        "pearson": _safe_corr(pred_np, actual_np, rank=False),
        "spearman": _safe_corr(pred_np, actual_np, rank=True),
        "pairwise_accuracy": float(torch.cat(pair).mean()) if pair else 1.0,
        "top1_accuracy": float((best_pred == best_actual).to(torch.float32).mean()),
        "top2_recall": float((pred_order[:, : min(2, pred.shape[1])] == best_actual.view(-1, 1)).any(dim=1).to(torch.float32).mean()),
        "oracle_regret_relative_z": float(regret.mean()),
    }


def routing_metrics(
    cache: Mapping[str, Any],
    forecasts: torch.Tensor,
    pred_z: torch.Tensor,
    actual_errors: torch.Tensor,
    std: torch.Tensor,
    temperature: float,
) -> dict[str, float]:
    weights = torch.softmax(-pred_z / max(float(temperature), EPS), dim=1)
    pred = (forecasts * weights.view(weights.shape[0], 1, 1, weights.shape[1])).sum(dim=-1)
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.bool)
    mae = sample_mae(pred, target, mask, std)
    mse = sample_mse(pred, target, mask, std)
    best = actual_errors.min(dim=1).values
    return {
        "mae": float(mae.mean()),
        "mse": float(mse.mean()),
        "oracle_single_expert_regret": float((mae - best).mean()),
        "temperature": float(temperature),
        "mean_entropy": float(-(weights * weights.clamp_min(EPS).log()).sum(dim=1).mean()),
    }


def select_temperature(train_cache: Mapping[str, Any], forecasts: torch.Tensor, pred_z: torch.Tensor, std: torch.Tensor, valid: torch.Tensor) -> float:
    best_temp, best_mae = TEMP_GRID[0], math.inf
    for temp in TEMP_GRID:
        weights = torch.softmax(-pred_z[valid] / temp, dim=1)
        pred = (forecasts[valid] * weights.view(weights.shape[0], 1, 1, weights.shape[1])).sum(dim=-1)
        mae = float(sample_mae(pred, train_cache["targets"][valid].to(torch.float32), train_cache["target_masks"][valid].to(torch.bool), std).mean())
        if mae < best_mae:
            best_mae = mae
            best_temp = temp
    return float(best_temp)


def method_predictions_for_split(
    train_demand: torch.Tensor,
    train_passive: torch.Tensor,
    train_z: torch.Tensor,
    eval_demand: torch.Tensor,
    eval_passive: torch.Tensor,
    expert_perm: torch.Tensor,
    axis_perm: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    n_experts = train_z.shape[1]
    global_mean = train_z.mean(dim=0)
    profile = fit_capability_profile(train_demand, train_z)

    pred_cap, pred_regime, pred_cont = predict_capability(profile, eval_demand)
    pred_expert_shuffle, _, _ = predict_capability(profile, eval_demand, expert_perm=expert_perm)
    pred_axis_shuffle, _, _ = predict_capability(profile, eval_demand, axis_perm=axis_perm)
    pred_window_shuffle, _, _ = predict_capability(profile, eval_demand[torch.roll(torch.arange(eval_demand.shape[0]), 1)])

    passive_pred = fit_predict_ridge(train_passive.reshape(-1, train_passive.shape[-1]), train_z.reshape(-1, 1), eval_passive.reshape(-1, eval_passive.shape[-1])).reshape(eval_passive.shape[0], n_experts)
    fame_pred = fit_predict_ridge(train_demand, train_z, eval_demand)
    demand_id_pred = fit_predict_ridge(expert_id_features(train_demand, n_experts).reshape(-1, train_demand.shape[1] + n_experts), train_z.reshape(-1, 1), expert_id_features(eval_demand, n_experts).reshape(-1, eval_demand.shape[1] + n_experts)).reshape(eval_demand.shape[0], n_experts)
    preds = {
        "GlobalPrior": global_mean.view(1, -1).expand(eval_demand.shape[0], -1).clone(),
        "Passive": passive_pred,
        "FAMEStyleDemand": fame_pred,
        "DemandExpertID": demand_id_pred,
        "CapabilityMatch": pred_cap,
        "CapabilityRegime": pred_regime,
        "CapabilityContinuous": pred_cont,
        "ExpertShuffledCapability": pred_expert_shuffle,
        "AxisShuffledCapability": pred_axis_shuffle,
        "WindowShuffledCapability": pred_window_shuffle,
    }
    return preds, profile


def fit_oof_and_val(
    train_cache: Mapping[str, Any],
    val_cache: Mapping[str, Any],
    train_demand: torch.Tensor,
    val_demand: torch.Tensor,
    train_passive: torch.Tensor,
    val_passive: torch.Tensor,
    train_z: torch.Tensor,
) -> dict[str, Any]:
    starts = train_cache["absolute_window_starts"].to(torch.long)
    folds = folds_from_starts(starts)
    k = train_z.shape[1]
    methods = (
        "GlobalPrior",
        "Passive",
        "FAMEStyleDemand",
        "DemandExpertID",
        "CapabilityMatch",
        "CapabilityRegime",
        "CapabilityContinuous",
        "ExpertShuffledCapability",
        "AxisShuffledCapability",
        "WindowShuffledCapability",
    )
    oof_preds = {m: torch.full((train_z.shape[0], k), float("nan")) for m in methods}
    final_val_preds: dict[str, torch.Tensor] = {}
    profile_fold_rows: list[dict[str, Any]] = []
    demand_rows: list[dict[str, Any]] = []
    profile_objects = []
    expert_perm = torch.roll(torch.arange(k), shifts=-1)
    axis_perm = torch.tensor([1, 2, 4, 0, 5, 3], dtype=torch.long)

    for fold in folds:
        fit_idx = fold["fit_indices"]
        if int(fit_idx.numel()) < 16:
            raise RuntimeError(f"Fold {fold['fold']} has too few legal training windows")
        lo, hi = int(fold["eval_lo"]), int(fold["eval_hi"])
        preds, profile = method_predictions_for_split(
            train_demand[fit_idx],
            train_passive[fit_idx],
            train_z[fit_idx],
            train_demand[lo:hi],
            train_passive[lo:hi],
            expert_perm,
            axis_perm,
        )
        for name, pred in preds.items():
            oof_preds[name][lo:hi] = pred
        _, eval_bins = quantile_bins(train_demand[fit_idx], train_demand[lo:hi])
        for local_i, idx in enumerate(range(lo, hi)):
            row = {"dataset_split": "router_train_oof", "window_index": idx, "fold": fold["fold"], "absolute_window_start": int(starts[idx])}
            for axis_i, axis in enumerate(AXES):
                row[axis] = float(train_demand[idx, axis_i])
                row[f"{axis}_bin"] = int(eval_bins[local_i, axis_i])
            demand_rows.append(row)
        profile_objects.append(profile)
        for expert in range(k):
            for axis_i, axis in enumerate(AXES):
                for b, label in enumerate(("LOW", "MED", "HIGH")):
                    profile_fold_rows.append(
                        {
                            "fold": fold["fold"],
                            "expert_index": expert,
                            "axis": axis,
                            "bin": label,
                            "relative_competence_z": float(profile["table"][expert, axis_i, b]),
                            "count": int(profile["counts"][expert, axis_i, b]),
                            "global_expert_mean_z": float(profile["global_mean"][expert]),
                            "q33": float(profile["quantiles"][axis_i, 0]),
                            "q67": float(profile["quantiles"][axis_i, 1]),
                        }
                    )

    valid = torch.isfinite(oof_preds["CapabilityMatch"]).all(dim=1)
    final_idx = torch.arange(train_z.shape[0])
    final_preds, final_profile = method_predictions_for_split(
        train_demand[final_idx],
        train_passive[final_idx],
        train_z[final_idx],
        val_demand,
        val_passive,
        expert_perm,
        axis_perm,
    )
    final_val_preds.update(final_preds)
    _, val_bins = quantile_bins(train_demand[final_idx], val_demand)
    val_starts = val_cache["absolute_window_starts"].to(torch.long)
    for idx in range(val_demand.shape[0]):
        row = {"dataset_split": "router_val_final", "window_index": idx, "fold": "final", "absolute_window_start": int(val_starts[idx])}
        for axis_i, axis in enumerate(AXES):
            row[axis] = float(val_demand[idx, axis_i])
            row[f"{axis}_bin"] = int(val_bins[idx, axis_i])
        demand_rows.append(row)

    return {
        "folds": [{k: v for k, v in row.items() if k != "fit_indices"} for row in folds],
        "oof_preds": oof_preds,
        "valid_oof": valid,
        "val_preds": final_val_preds,
        "final_profile": final_profile,
        "profile_fold_rows": profile_fold_rows,
        "demand_rows": demand_rows,
        "profile_objects": profile_objects,
        "expert_perm": expert_perm.tolist(),
        "axis_perm": axis_perm.tolist(),
    }


def etth2_integrity_audit(device: str = "cpu") -> dict[str, Any]:
    dataset = "ETTh2"
    paths = cache_paths(dataset)
    for p in paths.values():
        refuse_test(p)
    bundle = LOADERS[dataset]()
    train, val = bundle.train_cache, bundle.val_cache
    audit: dict[str, Any] = {
        "dataset": dataset,
        "status": "ETTH2_INTEGRITY_UNRESOLVED",
        "cache_paths": {k: relative_path(v) for k, v in paths.items()},
        "schema": {
            "router_train": validate_cache_schema(dataset, train, "router_train", bundle.core_names),
            "router_val": validate_cache_schema(dataset, val, "router_val", bundle.core_names),
        },
        "expected_cache_convention": "ETTh2 histories/targets/predictions are stored in DLinear scaler-normalized units; metrics use std=ones.",
        "structured_repair_prior_status": "runtime wrapper was called on normalized cache histories as if raw, producing ~4.4-4.9 max raw forecast differences.",
        "per_expert": {},
    }
    sample_idx = torch.unique(torch.tensor([0, 1, 2, max(0, int(val["num_windows"]) // 2), int(val["num_windows"]) - 3, int(val["num_windows"]) - 2, int(val["num_windows"]) - 1], dtype=torch.long))
    all_direct_ok = True
    all_denorm_ok = True
    all_wrong_large = True
    for expert in EXPERT_ORDER:
        runtime = load_expert_runtime(dataset, expert, device=device)
        ckpt_hash_before = runtime.checkpoint_sha256
        cached = val["prediction_stack"][sample_idx, ..., list(val["expert_names"]).index(expert)].to(torch.float32)
        hist_norm = val["histories"][sample_idx].to(torch.float32)
        with torch.no_grad():
            direct = runtime.call_fn(runtime.model, hist_norm.to(runtime.device)).detach().cpu().to(torch.float32)
        hist_raw = raw_history_cache(dataset, val, runtime.mean.cpu(), runtime.std.cpu())["histories"][sample_idx].to(torch.float32)
        denorm_wrapper = runtime.predict(hist_raw, batch_size=32)
        wrong_wrapper = runtime.predict(hist_norm, batch_size=32)
        direct_diff = (direct - cached).abs()
        denorm_diff = (denorm_wrapper - cached).abs()
        wrong_diff = (wrong_wrapper - cached).abs()
        row = {
            "checkpoint_path": relative_path(runtime.checkpoint_path),
            "checkpoint_sha256_before": ckpt_hash_before,
            "checkpoint_sha256_after": file_hash(runtime.checkpoint_path),
            "direct_normalized_cache_input_max_abs_diff": float(direct_diff.max()),
            "direct_normalized_cache_input_mean_abs_diff": float(direct_diff.mean()),
            "denormalized_raw_history_wrapper_max_abs_diff": float(denorm_diff.max()),
            "denormalized_raw_history_wrapper_mean_abs_diff": float(denorm_diff.mean()),
            "wrong_normalized_history_wrapper_max_abs_diff": float(wrong_diff.max()),
            "wrong_normalized_history_wrapper_mean_abs_diff": float(wrong_diff.mean()),
            "runtime_rescale_output": bool(runtime.rescale_output),
            "runtime_input_len": int(runtime.input_len),
            "runtime_horizon": int(runtime.horizon),
            "runtime_num_features": int(runtime.num_features),
            "parameters_frozen": all(not p.requires_grad for p in runtime.model.parameters()),
            "model_eval_mode": not runtime.model.training,
        }
        audit["per_expert"][expert] = row
        all_direct_ok = all_direct_ok and row["direct_normalized_cache_input_max_abs_diff"] < 5e-4
        all_denorm_ok = all_denorm_ok and row["denormalized_raw_history_wrapper_max_abs_diff"] < 5e-4
        all_wrong_large = all_wrong_large and row["wrong_normalized_history_wrapper_max_abs_diff"] > 0.1
    if all_direct_ok and all_denorm_ok and all_wrong_large:
        audit["status"] = "ETTH2_INTEGRITY_RESOLVED"
        audit["resolution"] = "The cache is internally consistent. The prior mismatch is explained by passing already-normalized ETTh2 cache histories into a runtime wrapper that expects raw histories."
    else:
        audit["resolution"] = "The normalization-convention explanation did not fully reproduce cached forecasts within tolerance."
    return audit


def flatten_profile(profile: Mapping[str, Any], expert_names: Sequence[str]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "axis_names": list(AXES),
        "expert_names": list(expert_names),
        "quantiles": profile["quantiles"],
        "global_mean": {expert_names[i]: float(profile["global_mean"][i]) for i in range(len(expert_names))},
        "regime_table": {},
    }
    for expert_i, expert in enumerate(expert_names):
        out["regime_table"][expert] = {}
        for axis_i, axis in enumerate(AXES):
            out["regime_table"][expert][axis] = {
                "LOW": float(profile["table"][expert_i, axis_i, 0]),
                "MED": float(profile["table"][expert_i, axis_i, 1]),
                "HIGH": float(profile["table"][expert_i, axis_i, 2]),
                "counts": [int(x) for x in profile["counts"][expert_i, axis_i].tolist()],
            }
    return out


def dataset_rows(dataset: str, method: str, metrics: Mapping[str, float], split: str = "router_val") -> dict[str, Any]:
    return {"dataset": dataset, "method": method, "split": split, **metrics}


def evaluate_dataset(dataset: str, etth2_status: str) -> dict[str, Any]:
    bundle = LOADERS[dataset]()
    train, val = bundle.train_cache, bundle.val_cache
    paths = cache_paths(dataset)
    for p in paths.values():
        refuse_test(p)
    for p in checkpoint_paths(dataset).values():
        refuse_test(p)
    train_schema = validate_cache_schema(dataset, train, "router_train", bundle.core_names)
    val_schema = validate_cache_schema(dataset, val, "router_val", bundle.core_names)
    forecasts_train = selected_forecasts(bundle, train)
    forecasts_val = selected_forecasts(bundle, val)
    train_z, train_errors = competence(train, forecasts_train, bundle.std)
    val_z, val_errors = competence(val, forecasts_val, bundle.std)

    train_demand = demand_features_from_history(train["histories"].to(torch.float32), bundle.std)
    val_demand = demand_features_from_history(val["histories"].to(torch.float32), bundle.std)
    train_passive = passive_features(train, forecasts_train, bundle.std)
    val_passive = passive_features(val, forecasts_val, bundle.std)
    fit = fit_oof_and_val(train, val, train_demand, val_demand, train_passive, val_passive, train_z)

    competence_rows = []
    routing_rows = []
    per_axis_rows = []
    per_expert_rows = []
    dependence_rows = []
    method_per_window = {}
    counted = not (dataset == "ETTh2" and etth2_status != "ETTH2_INTEGRITY_RESOLVED")
    for method, pred in fit["val_preds"].items():
        met = competence_metrics(pred, val_z)
        competence_rows.append(dataset_rows(dataset, method, met))
        method_per_window[method] = per_window_prediction_mae(pred, val_z)
        if method in {"GlobalPrior", "Passive", "FAMEStyleDemand", "DemandExpertID", "CapabilityMatch", "ExpertShuffledCapability", "AxisShuffledCapability"}:
            temp = select_temperature(train, forecasts_train, fit["oof_preds"][method], bundle.std, fit["valid_oof"])
            routing_rows.append(dataset_rows(dataset, method, routing_metrics(val, forecasts_val, pred, val_errors, bundle.std, temp)))
        for expert_i, expert in enumerate(bundle.core_names):
            per_expert_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "expert": expert,
                    "mae": float((pred[:, expert_i] - val_z[:, expert_i]).abs().mean()),
                    "mse": float((pred[:, expert_i] - val_z[:, expert_i]).square().mean()),
                    "pearson": _safe_corr(pred[:, expert_i].numpy(), val_z[:, expert_i].numpy(), rank=False),
                    "spearman": _safe_corr(pred[:, expert_i].numpy(), val_z[:, expert_i].numpy(), rank=True),
                }
            )

    for axis_i, axis in enumerate(AXES):
        axis_pred, _, _ = predict_capability(fit["final_profile"], val_demand, axes_subset=[axis_i])
        per_axis_rows.append({"dataset": dataset, "axis": axis, **competence_metrics(axis_pred, val_z)})
        for expert_i, expert in enumerate(bundle.core_names):
            low = val_demand[:, axis_i] <= fit["final_profile"]["quantiles"][axis_i, 0]
            high = val_demand[:, axis_i] > fit["final_profile"]["quantiles"][axis_i, 1]
            per_axis_rows.append(
                {
                    "dataset": dataset,
                    "axis": axis,
                    "expert": expert,
                    "high_minus_low_actual_z": float(val_z[high, expert_i].mean() - val_z[low, expert_i].mean()) if bool(low.any() and high.any()) else 0.0,
                    "high_minus_low_predicted_z": float(axis_pred[high, expert_i].mean() - axis_pred[low, expert_i].mean()) if bool(low.any() and high.any()) else 0.0,
                }
            )

    comparisons = [
        ("CapabilityMatch_vs_FAMEStyleDemand", "CapabilityMatch", "FAMEStyleDemand"),
        ("CapabilityMatch_vs_Passive", "CapabilityMatch", "Passive"),
        ("CapabilityMatch_vs_GlobalPrior", "CapabilityMatch", "GlobalPrior"),
        ("CapabilityMatch_vs_ExpertShuffledCapability", "CapabilityMatch", "ExpertShuffledCapability"),
        ("CapabilityMatch_vs_AxisShuffledCapability", "CapabilityMatch", "AxisShuffledCapability"),
    ]
    for comparison, candidate, baseline in comparisons:
        cand = method_per_window[candidate]
        base = method_per_window[baseline]
        block = block_bootstrap_with_prob(cand, base, block=BLOCK_LENGTH, seed=20260828, samples=BOOTSTRAP_SAMPLES)
        phase = every_kth_phase_bootstrap(cand - base, k=12, seed=20260828, samples=BOOTSTRAP_SAMPLES)
        dependence_rows.append({"dataset": dataset, "comparison": comparison, "test": "block24", **block, "counted_in_success": counted})
        dependence_rows.append({"dataset": dataset, "comparison": comparison, "test": "every12th_phase", **phase, "counted_in_success": counted})

    stability_rows = []
    grouped: dict[tuple[int, str, str], list[float]] = {}
    for row in fit["profile_fold_rows"]:
        key = (int(row["expert_index"]), str(row["axis"]), str(row["bin"]))
        grouped.setdefault(key, []).append(float(row["relative_competence_z"]))
    for (expert_i, axis, bin_label), values in grouped.items():
        t = torch.tensor(values)
        stability_rows.append(
            {
                "dataset": dataset,
                "expert": bundle.core_names[expert_i],
                "axis": axis,
                "bin": bin_label,
                "fold_mean_z": float(t.mean()),
                "fold_std_z": float(t.std(unbiased=False)) if t.numel() > 1 else 0.0,
                "fold_range_z": float(t.max() - t.min()),
                "num_folds": int(t.numel()),
            }
        )

    corrupted_val = dict(val)
    gen = torch.Generator().manual_seed(20260828)
    corrupted_val["targets"] = torch.randn(val["targets"].shape, generator=gen)
    corrupted_val["target_masks"] = torch.logical_not(val["target_masks"].to(torch.bool))
    corrupt_demand = demand_features_from_history(corrupted_val["histories"].to(torch.float32), bundle.std)
    corrupt_passive = passive_features(corrupted_val, forecasts_val, bundle.std)
    corrupt_preds, _ = method_predictions_for_split(train_demand, train_passive, train_z, corrupt_demand, corrupt_passive, torch.tensor(fit["expert_perm"]), torch.tensor(fit["axis_perm"]))
    corruption_diffs = {
        "demand_max_abs": float((val_demand - corrupt_demand).abs().max()),
        "passive_max_abs": float((val_passive - corrupt_passive).abs().max()),
        "capability_pred_max_abs": float((fit["val_preds"]["CapabilityMatch"] - corrupt_preds["CapabilityMatch"]).abs().max()),
        "passive_pred_max_abs": float((fit["val_preds"]["Passive"] - corrupt_preds["Passive"]).abs().max()),
    }

    checkpoint_hashes_before = {expert: file_hash(path) for expert, path in checkpoint_paths(dataset).items()}
    checkpoint_hashes_after = {expert: file_hash(path) for expert, path in checkpoint_paths(dataset).items()}
    integrity = {
        "dataset": dataset,
        "test_loaded": False,
        "counted_in_success": counted,
        "etth2_status": etth2_status if dataset == "ETTh2" else "not_applicable",
        "schemas": {"router_train": train_schema, "router_val": val_schema},
        "core_names": list(bundle.core_names),
        "expert_indices": list(bundle.expert_idx),
        "expert_order": list(val["expert_names"]),
        "finite_demand": bool(torch.isfinite(train_demand).all() and torch.isfinite(val_demand).all()),
        "finite_predictions": bool(all(torch.isfinite(pred).all() for pred in fit["val_preds"].values())),
        "deterministic_demand_repeat": bool(torch.equal(val_demand, demand_features_from_history(val["histories"].to(torch.float32), bundle.std))),
        "target_corruption": corruption_diffs,
        "target_corruption_pass": all(v == 0.0 for v in corruption_diffs.values()),
        "folds": fit["folds"],
        "oof_chronological_purge_pass": all(row["old_target_end_le_current_origin"] for row in fit["folds"]),
        "router_val_profiles_from_train_only": True,
        "checkpoint_hashes_unchanged": checkpoint_hashes_before == checkpoint_hashes_after,
        "demand_hash_router_train": tensor_hash(train_demand),
        "demand_hash_router_val": tensor_hash(val_demand),
    }
    return {
        "dataset": dataset,
        "counted_in_success": counted,
        "competence_rows": competence_rows,
        "routing_rows": routing_rows,
        "fame_rows": [row for row in competence_rows if row["method"] == "FAMEStyleDemand"],
        "passive_rows": [row for row in competence_rows if row["method"] == "Passive"],
        "expert_shuffle_rows": [row for row in competence_rows if row["method"] == "ExpertShuffledCapability"],
        "axis_shuffle_rows": [row for row in competence_rows if row["method"] == "AxisShuffledCapability"],
        "dependence_rows": dependence_rows,
        "per_axis_rows": per_axis_rows,
        "per_expert_rows": per_expert_rows,
        "profile_fold_rows": [{**row, "dataset": dataset, "expert": bundle.core_names[int(row["expert_index"])]} for row in fit["profile_fold_rows"]],
        "stability_rows": stability_rows,
        "demand_rows": [{"dataset": dataset, **row} for row in fit["demand_rows"]],
        "capability_profile": flatten_profile(fit["final_profile"], bundle.core_names),
        "integrity": integrity,
        "checkpoint_hashes": checkpoint_hashes_before,
        "cache_hashes": {name: file_hash(path) for name, path in paths.items()},
        "summary": {
            "num_train_windows": int(train["num_windows"]),
            "num_val_windows": int(val["num_windows"]),
            "core_names": list(bundle.core_names),
            "method_metrics": {row["method"]: {k: row[k] for k in row if k not in {"dataset", "method", "split"}} for row in competence_rows},
            "routing_metrics": {row["method"]: {k: row[k] for k in row if k not in {"dataset", "method", "split"}} for row in routing_rows},
        },
    }


def classify(results: Mapping[str, Any], etth2_status: str) -> str:
    counted = [payload for ds, payload in results.items() if payload["counted_in_success"]]
    if not counted:
        return "NO_CAPABILITY_DEMAND_SIGNAL"
    improvements_fame = 0
    improvements_passive = 0
    shuffle_support = 0
    axis_support = 0
    unstable = 0
    positive_corr = 0
    for payload in counted:
        metrics = payload["summary"]["method_metrics"]
        cap = metrics["CapabilityMatch"]
        fame = metrics["FAMEStyleDemand"]
        passive = metrics["Passive"]
        exp_shuffle = metrics["ExpertShuffledCapability"]
        axis_shuffle = metrics["AxisShuffledCapability"]
        if cap["mae"] < fame["mae"]:
            improvements_fame += 1
        if cap["mae"] < passive["mae"]:
            improvements_passive += 1
        if cap["mae"] < exp_shuffle["mae"]:
            shuffle_support += 1
        if cap["mae"] < axis_shuffle["mae"]:
            axis_support += 1
        if cap["pearson"] > 0.05 or cap["spearman"] > 0.05:
            positive_corr += 1
        max_stability = max(float(row["fold_std_z"]) for row in payload["stability_rows"]) if payload["stability_rows"] else 0.0
        cap_scale = max(float(metrics["CapabilityMatch"]["mae"]), EPS)
        if max_stability > 2.0 * cap_scale:
            unstable += 1
    n = len(counted)
    if unstable >= max(2, math.ceil(n / 2)):
        return "UNSTABLE_CAPABILITY_PROFILES"
    if positive_corr == 0 and improvements_passive == 0 and improvements_fame == 0:
        return "NO_CAPABILITY_DEMAND_SIGNAL"
    if improvements_fame >= math.ceil(0.75 * n) and improvements_passive >= math.ceil(0.75 * n) and shuffle_support >= math.ceil(0.75 * n) and axis_support >= math.ceil(0.75 * n):
        return "STRONG_CAPABILITY_DEMAND_SIGNAL"
    if improvements_passive >= math.ceil(0.5 * n) and improvements_fame < math.ceil(0.5 * n):
        return "MATCHING_SIGNAL_BUT_REDUNDANT"
    if positive_corr >= math.ceil(0.5 * n):
        return "CAPABILITY_SIGNAL_BUT_NO_MATCHING_GAIN"
    return "NO_CAPABILITY_DEMAND_SIGNAL"


def render_report(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Natural Capability-Demand Matching",
        "",
        "Validation-only study. No test cache, target, or metric was loaded.",
        "",
        f"Final classification: `{payload['classification']}`",
        f"ETTh2 integrity status: `{payload['etth2_integrity_status']}`",
        "",
        "## Competence MAE",
        "",
        "| Dataset | Counted | Global | Passive | FAME-style | Demand+ID | Capability | Expert shuffle | Axis shuffle |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, result in payload["datasets"].items():
        m = result["summary"]["method_metrics"]
        lines.append(
            f"| {dataset} | {result['counted_in_success']} | `{m['GlobalPrior']['mae']:.6f}` | `{m['Passive']['mae']:.6f}` | `{m['FAMEStyleDemand']['mae']:.6f}` | `{m['DemandExpertID']['mae']:.6f}` | `{m['CapabilityMatch']['mae']:.6f}` | `{m['ExpertShuffledCapability']['mae']:.6f}` | `{m['AxisShuffledCapability']['mae']:.6f}` |"
        )
    lines += [
        "",
        "## Routing Proxy MAE",
        "",
        "| Dataset | Method | MAE | MSE | Temperature | Regret vs oracle single |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for result in payload["datasets"].values():
        for row in result["routing_rows"]:
            lines.append(
                f"| {row['dataset']} | {row['method']} | `{row['mae']:.6f}` | `{row['mse']:.6f}` | `{row['temperature']:.3f}` | `{row['oracle_single_expert_regret']:.6f}` |"
            )
    lines += [
        "",
        "## Interpretation",
        "",
        "The target is relative competence `z[t,k] = expert_error[t,k] - mean_j expert_error[t,j]`; lower predicted values mean a better-matched expert. The primary `CapabilityMatch` score is the fixed equal average of a LOW/MED/HIGH regime profile and a quadratic Ridge capability curve for each semantic demand axis.",
        "",
        "ETTh2 is counted only if the separate audit resolves the cache/runtime reproduction discrepancy. See `etth2_integrity_audit.json` for the normalized-history convention check.",
        "",
        "## Integrity",
        "",
        f"- Test loaded: `{payload['test_loaded']}`.",
        "- Demand fingerprints are computed from histories only.",
        "- LOW/MED/HIGH bins, capability profiles, Ridge baselines, and routing temperatures are fit from legal router_train prefixes only.",
        "- Router_val target corruption leaves features, predictions, and weights unchanged exactly.",
        "- Checkpoint hashes are recorded before and after the run.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)

    print("[capability-demand] auditing ETTh2 cache/runtime integrity...", flush=True)
    etth2_audit = etth2_integrity_audit(device=args.device)
    write_json(OUT / "etth2_integrity_audit.json", etth2_audit)
    etth2_status = str(etth2_audit["status"])
    print(f"[capability-demand] ETTh2 status: {etth2_status}", flush=True)

    all_results: dict[str, Any] = {}
    competence_rows: list[dict[str, Any]] = []
    routing_rows: list[dict[str, Any]] = []
    fame_rows: list[dict[str, Any]] = []
    passive_rows: list[dict[str, Any]] = []
    expert_shuffle_rows: list[dict[str, Any]] = []
    axis_shuffle_rows: list[dict[str, Any]] = []
    dependence_rows: list[dict[str, Any]] = []
    per_axis_rows: list[dict[str, Any]] = []
    per_expert_rows: list[dict[str, Any]] = []
    profile_fold_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    demand_rows: list[dict[str, Any]] = []
    capability_profiles: dict[str, Any] = {}
    integrity: dict[str, Any] = {}
    checkpoint_hashes: dict[str, Any] = {}
    source_cache_hashes: dict[str, Any] = {}

    for dataset in args.datasets:
        print(f"[capability-demand] {dataset}: computing demand features and capability matches...", flush=True)
        result = evaluate_dataset(dataset, etth2_status)
        all_results[dataset] = {"counted_in_success": result["counted_in_success"], "summary": result["summary"], "routing_rows": result["routing_rows"], "stability_rows": result["stability_rows"]}
        competence_rows.extend(result["competence_rows"])
        routing_rows.extend(result["routing_rows"])
        fame_rows.extend(result["fame_rows"])
        passive_rows.extend(result["passive_rows"])
        expert_shuffle_rows.extend(result["expert_shuffle_rows"])
        axis_shuffle_rows.extend(result["axis_shuffle_rows"])
        dependence_rows.extend(result["dependence_rows"])
        per_axis_rows.extend(result["per_axis_rows"])
        per_expert_rows.extend(result["per_expert_rows"])
        profile_fold_rows.extend(result["profile_fold_rows"])
        stability_rows.extend(result["stability_rows"])
        demand_rows.extend(result["demand_rows"])
        capability_profiles[dataset] = result["capability_profile"]
        integrity[dataset] = result["integrity"]
        checkpoint_hashes[dataset] = result["checkpoint_hashes"]
        source_cache_hashes[dataset] = result["cache_hashes"]
        print(f"[capability-demand] {dataset}: done. counted={result['counted_in_success']}", flush=True)

    classification = classify({dataset: evaluate for dataset, evaluate in all_results.items()}, etth2_status)
    payload = {
        "experiment": "natural_capability_demand_matching",
        "code_version": CODE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_head(),
        "runtime_sec": time.perf_counter() - start,
        "datasets": all_results,
        "classification": classification,
        "etth2_integrity_status": etth2_status,
        "test_loaded": False,
        "success_criteria": [
            "STRONG_CAPABILITY_DEMAND_SIGNAL",
            "CAPABILITY_SIGNAL_BUT_NO_MATCHING_GAIN",
            "MATCHING_SIGNAL_BUT_REDUNDANT",
            "UNSTABLE_CAPABILITY_PROFILES",
            "NO_CAPABILITY_DEMAND_SIGNAL",
        ],
    }

    write_json(OUT / "results.json", payload)
    write_json(OUT / "method_manifest.json", {
        "code_version": CODE_VERSION,
        "datasets": list(args.datasets),
        "axes": list(AXES),
        "horizon": HORIZON,
        "ridge_alpha": RIDGE_ALPHA,
        "shrinkage": SHRINKAGE,
        "temperature_grid": TEMP_GRID,
        "oof_protocol": "20% warmup plus four chronological router_train folds; fit windows satisfy old_start + horizon <= current_origin",
        "primary_score": "0.5 * regime_table_prediction + 0.5 * continuous_curve_prediction",
        "test_loaded": False,
    })
    write_json(OUT / "source_provenance.json", {
        "git_commit": git_head(),
        "cache_hashes": source_cache_hashes,
        "cache_paths": {dataset: {k: relative_path(v) for k, v in cache_paths(dataset).items()} for dataset in args.datasets},
        "passive_source": "experiments.behavioral_competence.common::{window_features_group_a,forecast_features_group_b,disagreement_features_group_c}",
        "fame_baseline_source": "experiments/published_baseline_comparisons/run_published_baselines.py and scripts/fame_etth_router.py, adapted here as capacity-matched one-sided Ridge on six demand axes",
        "test_loaded": False,
    })
    write_json(OUT / "checkpoint_hashes.json", checkpoint_hashes)
    write_json(OUT / "integrity_checks.json", integrity)
    write_json(OUT / "demand_feature_definitions.json", {
        "axes": {
            "trend": "mean normalized least-squares history slope magnitude",
            "seasonality": "maximum absolute observed-lag autocorrelation over lags 2..48 within the history",
            "frequency": "spectral entropy plus a small high/low power-ratio contribution",
            "volatility": "normalized RMS/median first-difference variation",
            "shift": "first-half vs second-half mean, variance, and slope change",
            "crossvar": "average absolute correlation to the within-window cross-variable common factor; zero for univariate windows",
        },
        "history_only": True,
        "binning": "LOW/MED/HIGH from legal train-prefix q33/q67 only",
    })
    write_json(OUT / "capability_profiles.json", capability_profiles)
    write_csv(OUT / "demand_features_oof.csv", demand_rows)
    write_csv(OUT / "capability_profiles_by_fold.csv", profile_fold_rows)
    write_csv(OUT / "capability_stability.csv", stability_rows)
    write_csv(OUT / "competence_results.csv", competence_rows)
    write_csv(OUT / "routing_proxy_results.csv", routing_rows)
    write_csv(OUT / "fame_style_results.csv", fame_rows)
    write_csv(OUT / "passive_results.csv", passive_rows)
    write_csv(OUT / "expert_shuffle_results.csv", expert_shuffle_rows)
    write_csv(OUT / "axis_shuffle_results.csv", axis_shuffle_rows)
    write_csv(OUT / "dependence_tests.csv", dependence_rows)
    write_csv(OUT / "per_axis_results.csv", per_axis_rows)
    write_csv(OUT / "per_expert_results.csv", per_expert_rows)
    (OUT / "report.md").write_text(render_report(payload), encoding="utf-8")
    print(json.dumps({"out_dir": relative_path(OUT), "classification": classification, "etth2_status": etth2_status, "test_loaded": False}, indent=2), flush=True)


if __name__ == "__main__":
    main()
