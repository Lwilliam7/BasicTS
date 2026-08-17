"""Causal residual-covariance weighting for COSTAR-TS.

This experiment keeps the frozen PatchTST + iTransformer + TimesNet experts
and the existing horizon-variable adaptive prediction fixed.  Hyperparameters
are selected only on chronological folds inside router-train; validation is
evaluated once after selection.  Test caches are refused.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.chronological_adaptive_costar.run_chronological_adaptive_costar import (  # noqa: E402
    SEEDS,
    enforce_observable,
    paired_bootstrap,
)
from experiments.horizon_variable_adaptive_costar.run_hv_adaptive_costar import fixed3_forecasts  # noqa: E402
from experiments.oracle_weight_tournament.run_tournament import (  # noqa: E402
    FIXED3,
    load_cache,
    load_std,
    sample_mae,
    sample_mse,
    weighted_forecast,
)
from experiments.residual_correction_costar.run_residual_correction_experiments import (  # noqa: E402
    BASELINE_MAE,
    BASELINE_MSE,
    fixed_current_best_prediction,
)


BASELINE_NAME = "hybrid_chrono_hvema_lowrank1_decay0.95_temp0.1_alpha0.75"
STRONG_TARGET = 0.3619
EPS = 1e-8


@dataclass(frozen=True)
class CovConfig:
    family: str
    structure: str
    decay: float
    ridge: float
    shrink_diag: float
    shrink_global: float
    bias_weight: float
    hybrid_alpha: float
    min_count: int = 96

    @property
    def base_key(self) -> tuple[Any, ...]:
        return (
            self.family,
            self.structure,
            self.decay,
            self.ridge,
            self.shrink_diag,
            self.shrink_global,
            self.bias_weight,
            self.min_count,
        )

    @property
    def name(self) -> str:
        return (
            f"{self.family}_{self.structure}_decay{self.decay:g}_ridge{self.ridge:g}"
            f"_sd{self.shrink_diag:g}_sg{self.shrink_global:g}_bias{self.bias_weight:g}"
            f"_alpha{self.hybrid_alpha:g}_warm{self.min_count}"
        )


def refuse_test(path: str | Path) -> None:
    if "test" in str(path).lower():
        raise ValueError(f"Refusing test path: {path}")


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


def metrics(cache: Mapping[str, Any], std: torch.Tensor, pred: torch.Tensor) -> dict[str, Any]:
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.bool)
    mae = sample_mae(pred, target, mask, std)
    mse = sample_mse(pred, target, mask, std)
    return {"mae": float(mae.mean()), "mse": float(mse.mean()), "per_window_mae": mae, "per_window_mse": mse}


def normalized_abs_error(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return ((pred - target) / std.view(1, 1, -1)).abs() * mask.to(torch.float32)


def per_axis_rows(cache: Mapping[str, Any], std: torch.Tensor, pred: torch.Tensor, base: torch.Tensor, method: str) -> list[dict[str, Any]]:
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.float32)
    cand = normalized_abs_error(pred, target, mask.to(torch.bool), std)
    ref = normalized_abs_error(base, target, mask.to(torch.bool), std)
    rows: list[dict[str, Any]] = []
    for h in range(cand.shape[1]):
        denom = mask[:, h].sum().clamp_min(1)
        mae = float(cand[:, h].sum() / denom)
        bmae = float(ref[:, h].sum() / denom)
        rows.append({"method": method, "axis": "horizon", "index": h, "mae": mae, "baseline_mae": bmae, "delta_vs_baseline": mae - bmae})
    for v in range(cand.shape[2]):
        denom = mask[:, :, v].sum().clamp_min(1)
        mae = float(cand[:, :, v].sum() / denom)
        bmae = float(ref[:, :, v].sum() / denom)
        rows.append({"method": method, "axis": "variable", "index": v, "mae": mae, "baseline_mae": bmae, "delta_vs_baseline": mae - bmae})
    return rows


def per_hv_rows(cache: Mapping[str, Any], std: torch.Tensor, pred: torch.Tensor, base: torch.Tensor, method: str) -> list[dict[str, Any]]:
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.float32)
    cand = normalized_abs_error(pred, target, mask.to(torch.bool), std)
    ref = normalized_abs_error(base, target, mask.to(torch.bool), std)
    rows: list[dict[str, Any]] = []
    for h in range(cand.shape[1]):
        for v in range(cand.shape[2]):
            denom = mask[:, h, v].sum().clamp_min(1)
            mae = float(cand[:, h, v].sum() / denom)
            bmae = float(ref[:, h, v].sum() / denom)
            rows.append({"method": method, "horizon": h, "variable": v, "mae": mae, "baseline_mae": bmae, "delta_vs_baseline": mae - bmae})
    return rows


def train_folds(n: int) -> list[tuple[int, int, int]]:
    min_train = int(round(n * 0.2))
    usable = n - min_train
    bounds = [min_train + i * usable // 4 for i in range(5)]
    return [(0, bounds[i], bounds[i + 1]) for i in range(4)]


def aggregate_residual(resid: torch.Tensor, structure: str) -> torch.Tensor:
    if structure == "global":
        return resid.mean(dim=(0, 1), keepdim=False).view(1, 3)
    if structure == "horizon":
        return resid.mean(dim=1)
    if structure == "variable":
        return resid.mean(dim=0)
    if structure == "hv":
        return resid.reshape(-1, 3)
    raise ValueError(structure)


def group_shape(horizon: int, variables: int, structure: str) -> tuple[int, ...]:
    if structure == "global":
        return (1,)
    if structure == "horizon":
        return (horizon,)
    if structure == "variable":
        return (variables,)
    if structure == "hv":
        return (horizon, variables)
    raise ValueError(structure)


def expand_weights(group_weights: torch.Tensor, structure: str, horizon: int, variables: int) -> torch.Tensor:
    if structure == "global":
        return group_weights.view(1, 1, 3).expand(horizon, variables, 3)
    if structure == "horizon":
        return group_weights.view(horizon, 1, 3).expand(horizon, variables, 3)
    if structure == "variable":
        return group_weights.view(1, variables, 3).expand(horizon, variables, 3)
    if structure == "hv":
        return group_weights.view(horizon, variables, 3)
    raise ValueError(structure)


def simplex_grid(step: float = 0.05) -> torch.Tensor:
    units = int(round(1.0 / step))
    rows = []
    for a in range(units + 1):
        for b in range(units + 1 - a):
            rows.append((a / units, b / units, (units - a - b) / units))
    return torch.tensor(rows, dtype=torch.float32)


def init_cov_state(residuals_norm: torch.Tensor, structure: str) -> tuple[torch.Tensor, torch.Tensor, int]:
    grouped = torch.stack([aggregate_residual(r, structure) for r in residuals_norm])
    mean = grouped.mean(dim=0)
    second = torch.einsum("nge,ngf->gef", grouped, grouped) / max(grouped.shape[0], 1)
    return mean, second, int(grouped.shape[0])


def group_abs_error(err: torch.Tensor, structure: str) -> torch.Tensor:
    if structure == "global":
        return err.mean(dim=(0, 1), keepdim=False).view(1, 3)
    if structure == "horizon":
        return err.mean(dim=1)
    if structure == "variable":
        return err.mean(dim=0)
    if structure == "hv":
        return err.reshape(-1, 3)
    raise ValueError(structure)


def init_abs_state(abs_err: torch.Tensor, structure: str) -> tuple[torch.Tensor, int]:
    grouped = torch.stack([group_abs_error(e, structure) for e in abs_err])
    return grouped.mean(dim=0), int(grouped.shape[0])


def optimal_cov_weights(
    mean: torch.Tensor,
    second: torch.Tensor,
    global_mean: torch.Tensor,
    global_second: torch.Tensor,
    config: CovConfig,
    grid: torch.Tensor,
    fallback: torch.Tensor,
    count: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    groups = mean.shape[0]
    eye = torch.eye(3, dtype=mean.dtype).view(1, 3, 3)
    cov = second - torch.einsum("ge,gf->gef", mean, mean)
    cov = 0.5 * (cov + cov.transpose(1, 2))
    diag = torch.diag_embed(torch.diagonal(cov, dim1=1, dim2=2).clamp_min(1e-8))
    if config.family == "diagonal_variance":
        cov = diag
    elif config.family == "full_covariance":
        cov = (1.0 - float(config.shrink_diag)) * cov + float(config.shrink_diag) * diag
        if float(config.shrink_global) > 0.0:
            gcov = global_second - torch.einsum("ge,gf->gef", global_mean, global_mean)
            gcov = 0.5 * (gcov + gcov.transpose(1, 2))
            gcov = gcov.mean(dim=0, keepdim=True).expand(groups, 3, 3)
            cov = (1.0 - float(config.shrink_global)) * cov + float(config.shrink_global) * gcov
    else:
        raise ValueError(config.family)
    cov = cov + float(config.ridge) * eye
    eig = torch.linalg.eigvalsh(cov)
    cond = eig[:, -1] / eig[:, 0].clamp_min(1e-12)
    good = torch.isfinite(cond) & (eig[:, 0] > 0) & (cond < 1e5) & (count >= int(config.min_count))
    score = torch.einsum("ke,gef,kf->gk", grid, cov, grid)
    if float(config.bias_weight) > 0.0:
        bias = mean @ grid.T
        score = score + float(config.bias_weight) * bias.square()
    idx = score.argmin(dim=1)
    weights = grid[idx].clone()
    weights[~good] = fallback.view(1, 3)
    if torch.any(weights < -1e-7) or torch.any((weights.sum(dim=1) - 1.0).abs() > 1e-5):
        raise AssertionError("Covariance weights are not convex simplex weights")
    return weights, {
        "fallback_groups": int((~good).sum()),
        "mean_condition": float(cond[torch.isfinite(cond)].mean()) if bool(torch.isfinite(cond).any()) else float("inf"),
        "max_condition": float(cond[torch.isfinite(cond)].max()) if bool(torch.isfinite(cond).any()) else float("inf"),
        "mean_min_eig": float(eig[:, 0].mean()),
    }


def inverse_error_weights(abs_state: torch.Tensor, count: int, config: CovConfig, fallback: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    if count < int(config.min_count):
        w = fallback.view(1, 3).expand(abs_state.shape[0], 3).clone()
        return w, {"fallback_groups": int(abs_state.shape[0]), "mean_condition": 1.0, "max_condition": 1.0, "mean_min_eig": 0.0}
    inv = 1.0 / abs_state.clamp_min(1e-6)
    w = inv / inv.sum(dim=1, keepdim=True).clamp_min(EPS)
    return w, {"fallback_groups": 0, "mean_condition": 1.0, "max_condition": 1.0, "mean_min_eig": 0.0}


def run_causal_weighting(
    starts: torch.Tensor,
    forecasts: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    std: torch.Tensor,
    hv_baseline: torch.Tensor,
    config: CovConfig,
    init_forecasts: torch.Tensor,
    init_target: torch.Tensor,
    init_mask: torch.Tensor,
    grid: torch.Tensor,
    trace_prefix: Mapping[str, Any] | None = None,
) -> tuple[torch.Tensor, dict[str, Any], list[dict[str, Any]]]:
    horizon, variables = forecasts.shape[1], forecasts.shape[2]
    fallback = torch.full((3,), 1.0 / 3.0)
    resid_norm = (target.unsqueeze(-1) - forecasts) / std.view(1, 1, -1, 1)
    init_resid_norm = (init_target.unsqueeze(-1) - init_forecasts) / std.view(1, 1, -1, 1)
    init_abs = (init_resid_norm.abs() * init_mask.to(torch.float32).unsqueeze(-1))
    global_mean, global_second, _ = init_cov_state(init_resid_norm, "global")
    if config.family == "inverse_error":
        abs_state, count = init_abs_state(init_abs, config.structure)
        mean = second = None
    else:
        mean, second, count = init_cov_state(init_resid_norm, config.structure)
        abs_state = None
    pending: list[int] = []
    preds: list[torch.Tensor] = []
    traces: list[dict[str, Any]] = []
    updates = 0
    fallback_groups = 0
    condition_vals = []
    min_eigs = []
    weight_chunks = []
    for i in range(forecasts.shape[0]):
        now = int(starts[i])
        still: list[int] = []
        for j in pending:
            if int(starts[j]) + horizon <= now:
                enforce_observable(int(starts[j]), now, horizon)
                x_global = aggregate_residual(resid_norm[j], "global")
                global_mean = float(config.decay) * global_mean + (1.0 - float(config.decay)) * x_global
                global_second = float(config.decay) * global_second + (1.0 - float(config.decay)) * torch.einsum("ge,gf->gef", x_global, x_global)
                if config.family == "inverse_error":
                    assert abs_state is not None
                    x_abs = group_abs_error(resid_norm[j].abs() * mask[j].to(torch.float32).unsqueeze(-1), config.structure)
                    abs_state = float(config.decay) * abs_state + (1.0 - float(config.decay)) * x_abs
                else:
                    assert mean is not None and second is not None
                    x = aggregate_residual(resid_norm[j], config.structure)
                    mean = float(config.decay) * mean + (1.0 - float(config.decay)) * x
                    second = float(config.decay) * second + (1.0 - float(config.decay)) * torch.einsum("ge,gf->gef", x, x)
                count += 1
                updates += 1
            else:
                still.append(j)
        pending = still
        if config.family == "inverse_error":
            assert abs_state is not None
            group_w, diag = inverse_error_weights(abs_state, count, config, fallback)
        else:
            assert mean is not None and second is not None
            group_w, diag = optimal_cov_weights(mean, second, global_mean, global_second, config, grid, fallback, count)
        w = expand_weights(group_w, config.structure, horizon, variables)
        cov_pred = (forecasts[i] * w).sum(dim=-1)
        pred = (1.0 - float(config.hybrid_alpha)) * hv_baseline[i] + float(config.hybrid_alpha) * cov_pred
        preds.append(pred)
        fallback_groups += int(diag["fallback_groups"])
        condition_vals.append(float(diag["mean_condition"]))
        min_eigs.append(float(diag["mean_min_eig"]))
        weight_chunks.append(w)
        traces.append(
            {
                **dict(trace_prefix or {}),
                "row": i,
                "start": now,
                "completed_count": count,
                "fallback_groups": int(diag["fallback_groups"]),
                "mean_condition": float(diag["mean_condition"]),
                "max_condition": float(diag["max_condition"]),
                "mean_min_eig": float(diag["mean_min_eig"]),
                **{f"mean_weight_{FIXED3[e]}": float(w[..., e].mean()) for e in range(3)},
                "mean_abs_delta_vs_hv": float((pred - hv_baseline[i]).abs().mean()),
            }
        )
        pending.append(i)
    pred_t = torch.stack(preds)
    weights_t = torch.stack(weight_chunks)
    if torch.any(weights_t < -1e-7) or torch.any((weights_t.sum(dim=-1) - 1.0).abs() > 1e-5):
        raise AssertionError("Expanded covariance weights violate simplex constraints")
    return pred_t, {
        "num_updates": updates,
        "fallback_group_rate": float(fallback_groups / max(weights_t.shape[0] * weights_t.shape[1] * weights_t.shape[2], 1)),
        "mean_condition": float(np.mean(condition_vals)) if condition_vals else 0.0,
        "mean_min_eig": float(np.mean(min_eigs)) if min_eigs else 0.0,
        "mean_abs_delta_vs_hv": float((pred_t - hv_baseline).abs().mean()),
        **{f"avg_weight_{FIXED3[e]}": float(weights_t[..., e].mean()) for e in range(3)},
    }, traces


def config_grid() -> list[CovConfig]:
    configs: list[CovConfig] = []
    for structure in ("global", "horizon", "variable", "hv"):
        for decay in (0.95, 0.97, 0.98, 0.99):
            for alpha in (0.25, 0.50, 0.75, 1.00):
                configs.append(CovConfig("inverse_error", structure, decay, 0.0, 1.0, 0.0, 0.0, alpha))
            for ridge in (1e-4, 1e-3):
                for bias in (0.0, 1.0):
                    for alpha in (0.25, 0.50, 0.75, 1.00):
                        configs.append(CovConfig("diagonal_variance", structure, decay, ridge, 1.0, 0.0, bias, alpha))
            for ridge in (1e-4, 1e-3):
                for shrink_diag, shrink_global in ((0.25, 0.0), (0.50, 0.0), (0.50, 0.25), (0.75, 0.25)):
                    for bias in (0.0, 1.0):
                        for alpha in (0.25, 0.50, 0.75, 1.00):
                            configs.append(CovConfig("full_covariance", structure, decay, ridge, shrink_diag, shrink_global, bias, alpha))
    return configs


def simplicity_key(config: CovConfig) -> tuple[Any, ...]:
    family_rank = {"inverse_error": 0, "diagonal_variance": 1, "full_covariance": 2}[config.family]
    structure_rank = {"global": 0, "horizon": 1, "variable": 2, "hv": 3}[config.structure]
    return (family_rank, structure_rank, config.hybrid_alpha, config.ridge, config.shrink_global, config.shrink_diag, config.bias_weight, -config.decay)


def select_one_se(rows: Sequence[Mapping[str, Any]], configs: Mapping[str, CovConfig]) -> Mapping[str, Any]:
    best = min(rows, key=lambda r: float(r["fold_mae_mean"]))
    threshold = float(best["fold_mae_mean"]) + float(best["fold_mae_se"])
    eligible = [r for r in rows if float(r["fold_mae_mean"]) <= threshold]
    selected = sorted(eligible, key=lambda r: simplicity_key(configs[str(r["name"])]))[0]
    return {"best": best, "selected": selected, "one_se_threshold": threshold}


def evaluate_folds(
    cache: Mapping[str, Any],
    std: torch.Tensor,
    hv_baseline: torch.Tensor,
    configs: Sequence[CovConfig],
    folds: Sequence[tuple[int, int, int]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    starts = cache["absolute_window_starts"].to(torch.long)
    forecasts = fixed3_forecasts(cache)
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.bool)
    grid = simplex_grid()
    rows: list[dict[str, Any]] = []
    fold_details: list[dict[str, Any]] = []
    by_base: dict[tuple[Any, ...], list[CovConfig]] = defaultdict(list)
    for config in configs:
        by_base[config.base_key].append(config)
    total = len(by_base)
    for base_idx, family_configs in enumerate(by_base.values(), start=1):
        base_config = family_configs[0]
        path_config = CovConfig(**(asdict(base_config) | {"hybrid_alpha": 1.0}))
        if base_idx == 1 or base_idx % 25 == 0 or base_idx == total:
            print(f"[fold-select] causal path {base_idx}/{total}: {path_config.name}", flush=True)
        path_by_fold: list[tuple[torch.Tensor, torch.Tensor, Mapping[str, Any]]] = []
        for fold_id, (train_lo, eval_lo, eval_hi) in enumerate(folds):
            cov_pred, extra, _ = run_causal_weighting(
                starts[eval_lo:eval_hi],
                forecasts[eval_lo:eval_hi],
                target[eval_lo:eval_hi],
                mask[eval_lo:eval_hi],
                std,
                hv_baseline[eval_lo:eval_hi],
                path_config,
                forecasts[train_lo:eval_lo],
                target[train_lo:eval_lo],
                mask[train_lo:eval_lo],
                grid,
                trace_prefix={"fold": fold_id, "config": path_config.name},
            )
            path_by_fold.append((cov_pred, hv_baseline[eval_lo:eval_hi], extra))
        for config in family_configs:
            cand_all = []
            base_all = []
            fold_rows = []
            diag_rows = []
            for fold_id, (_, eval_lo, eval_hi) in enumerate(folds):
                cov_pred, hv_pred, extra = path_by_fold[fold_id]
                pred = (1.0 - float(config.hybrid_alpha)) * hv_pred + float(config.hybrid_alpha) * cov_pred
                cm = sample_mae(pred, target[eval_lo:eval_hi], mask[eval_lo:eval_hi], std)
                bm = sample_mae(hv_pred, target[eval_lo:eval_hi], mask[eval_lo:eval_hi], std)
                cand_all.append(cm)
                base_all.append(bm)
                scaled_extra = dict(extra)
                scaled_extra["mean_abs_delta_vs_hv"] = float(extra["mean_abs_delta_vs_hv"]) * float(config.hybrid_alpha)
                frow = {"name": config.name, "fold": fold_id, "mae": float(cm.mean()), "baseline_mae": float(bm.mean()), "delta": float(cm.mean() - bm.mean()), **scaled_extra}
                fold_rows.append(frow)
                fold_details.append({**asdict(config), **frow})
                diag_rows.append(scaled_extra)
            cand = torch.cat(cand_all)
            base = torch.cat(base_all)
            deltas = torch.tensor([float(r["delta"]) for r in fold_rows])
            rows.append(
                {
                    "name": config.name,
                    **asdict(config),
                    "fold_mae_mean": float(cand.mean()),
                    "fold_baseline_mae_mean": float(base.mean()),
                    "fold_delta_mean": float(cand.mean() - base.mean()),
                    "fold_delta_std": float(deltas.std(unbiased=False)),
                    "fold_mae_se": float(deltas.std(unbiased=True) / math.sqrt(len(fold_rows))),
                    "fold_wins": int(sum(float(r["delta"]) < 0 for r in fold_rows)),
                    "fallback_group_rate": float(np.mean([float(r["fallback_group_rate"]) for r in diag_rows])),
                    "mean_condition": float(np.mean([float(r["mean_condition"]) for r in diag_rows])),
                    "mean_abs_delta_vs_hv": float(np.mean([float(r["mean_abs_delta_vs_hv"]) for r in diag_rows])),
                    **{f"avg_weight_{FIXED3[e]}": float(np.mean([float(r[f'avg_weight_{FIXED3[e]}']) for r in diag_rows])) for e in range(3)},
                }
            )
    return rows, fold_details


def aggregate_seed_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    out = []
    for method, group in grouped.items():
        maes = torch.tensor([float(r["mae"]) for r in group])
        mses = torch.tensor([float(r["mse"]) for r in group])
        out.append(
            {
                "method": method,
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


def make_report(out_dir: Path, report: Mapping[str, Any]) -> None:
    selected = report["selection"]["selected"]
    best_val = report["best_validation_method"]
    ci = report["aggregate_bootstrap_ci"].get(best_val["method"], {})
    lines = [
        "# Residual-Covariance Weighting COSTAR-TS",
        "",
        "## Protocol",
        "",
        f"- Frozen baseline: `{BASELINE_NAME}`.",
        "- Core experts: PatchTST, iTransformer, TimesNet.",
        "- Hyperparameters selected on chronological router-train folds only.",
        "- Validation was evaluated once after selection.",
        "- Test cache was not loaded.",
        "",
        "## Selection",
        "",
        f"- Selected config: `{selected['name']}`.",
        f"- Best fold config: `{report['selection']['best']['name']}`.",
        f"- One-SE threshold: `{report['selection']['one_se_threshold']:.6f}`.",
        "",
        "## Validation",
        "",
        f"- Best validation method: `{best_val['method']}`.",
        f"- MAE / MSE: `{best_val['mae_mean']:.6f}` / `{best_val['mse_mean']:.6f}`.",
        f"- Improvement vs `0.363642`: `{best_val['improvement_vs_0.363642']:.6f}` ({best_val['percent_improvement_vs_0.363642']:.3f}%).",
        f"- Aggregate paired CI: `[{ci.get('ci95_low', 0.0):.6f}, {ci.get('ci95_high', 0.0):.6f}]`.",
        f"- Strong target `<= 0.3619`: `{bool(best_val['mae_mean'] <= STRONG_TARGET)}`.",
        "",
        "## Diagnostics",
        "",
        f"- Mean fallback group rate: `{selected['fallback_group_rate']:.6f}`.",
        f"- Mean condition: `{selected['mean_condition']:.3f}`.",
        f"- Mean absolute delta vs HV baseline: `{selected['mean_abs_delta_vs_hv']:.6f}`.",
        "",
        "## Reproduce",
        "",
        "```powershell",
        report["reproduce_command"],
        "```",
    ]
    (out_dir / "covariance_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", default="cache/costarts_walkforward/router_train_20_60_cache.pt")
    parser.add_argument("--val-cache", default="cache/costarts_walkforward/router_val_60_80_cache.pt")
    parser.add_argument("--normalizer-checkpoint", default="checkpoints/costarts_walkforward/final_60/DLinear/best_expert.pt")
    parser.add_argument("--out-dir", default="experiments/residual_covariance_weighting_costar")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    args = parser.parse_args()
    t0 = time.time()
    for path in (args.train_cache, args.val_cache, args.normalizer_checkpoint):
        refuse_test(path)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_window").mkdir(exist_ok=True)

    train_cache = load_cache(ROOT / args.train_cache, "router_train_20_60")
    val_cache = load_cache(ROOT / args.val_cache, "router_val_60_80")
    std = load_std(ROOT / args.normalizer_checkpoint, int(val_cache["num_features"]))
    device = torch.device(args.device)
    train_starts = train_cache["absolute_window_starts"].to(torch.long)
    val_starts = val_cache["absolute_window_starts"].to(torch.long)
    if int(val_starts[-1]) + int(val_cache["forecast_horizon"]) > 11520:
        raise ValueError("Validation target window crosses held-out test boundary")
    if not bool(torch.all(train_starts[1:] > train_starts[:-1])) or not bool(torch.all(val_starts[1:] > val_starts[:-1])):
        raise ValueError("Caches must be chronological")

    train_base, _ = fixed_current_best_prediction(train_cache, train_cache, std, 7, device)
    folds = train_folds(int(train_cache["num_windows"]))
    configs = config_grid()
    cfg_by_name = {c.name: c for c in configs}
    fold_rows, fold_details = evaluate_folds(train_cache, std, train_base, configs, folds)
    fold_rows_sorted = sorted(fold_rows, key=lambda r: float(r["fold_mae_mean"]))
    write_csv(out_dir / "router_train_fold_leaderboard.csv", fold_rows_sorted)
    write_csv(out_dir / "router_train_fold_details.csv", fold_details)
    selection = select_one_se(fold_rows_sorted, cfg_by_name)
    write_json(out_dir / "selected_config.json", selection)
    selected_cfg = cfg_by_name[str(selection["selected"]["name"])]

    validation_rows = []
    trace_rows = []
    axis_rows = []
    hv_rows = []
    per_method: dict[str, list[torch.Tensor]] = defaultdict(list)
    per_base_by_seed: dict[int, torch.Tensor] = {}
    val_forecasts = fixed3_forecasts(val_cache)
    val_target = val_cache["targets"].to(torch.float32)
    val_mask = val_cache["target_masks"].to(torch.bool)
    train_forecasts = fixed3_forecasts(train_cache)
    train_target = train_cache["targets"].to(torch.float32)
    train_mask = train_cache["target_masks"].to(torch.bool)
    grid = simplex_grid()

    equal_pred = weighted_forecast(val_forecasts, torch.full((int(val_cache["num_windows"]), 3), 1.0 / 3.0))
    equal_metrics = metrics(val_cache, std, equal_pred)
    validation_rows.append({"method": "equal_fixed3_reference", "seed": -1, "mae": equal_metrics["mae"], "mse": equal_metrics["mse"], "diff_vs_0.363642": equal_metrics["mae"] - BASELINE_MAE})

    for seed in SEEDS:
        base_val, base_extra = fixed_current_best_prediction(val_cache, train_cache, std, seed, device)
        bm = metrics(val_cache, std, base_val)
        per_base_by_seed[seed] = bm["per_window_mae"]
        per_method["baseline_fixed3_hv"].append(bm["per_window_mae"])
        validation_rows.append({"method": "baseline_fixed3_hv", "seed": seed, "mae": bm["mae"], "mse": bm["mse"], "diff_vs_0.363642": bm["mae"] - BASELINE_MAE, **base_extra})
        torch.save(bm["per_window_mae"], out_dir / "per_window" / f"baseline_seed{seed}.pt")

        pred, extra, traces = run_causal_weighting(
            val_starts,
            val_forecasts,
            val_target,
            val_mask,
            std,
            base_val,
            selected_cfg,
            train_forecasts,
            train_target,
            train_mask,
            grid,
            trace_prefix={"method": "residual_covariance_selected", "seed": seed, "config": selected_cfg.name},
        )
        mm = metrics(val_cache, std, pred)
        boot = paired_bootstrap(mm["per_window_mae"], bm["per_window_mae"], seed=seed, samples=args.bootstrap_samples)
        method = "residual_covariance_selected"
        validation_rows.append({"method": method, "seed": seed, "mae": mm["mae"], "mse": mm["mse"], "diff_vs_0.363642": mm["mae"] - BASELINE_MAE, **boot, **extra})
        trace_rows.extend(traces)
        axis_rows.extend(per_axis_rows(val_cache, std, pred, base_val, f"{method}_seed{seed}"))
        hv_rows.extend(per_hv_rows(val_cache, std, pred, base_val, f"{method}_seed{seed}"))
        per_method[method].append(mm["per_window_mae"])
        torch.save(mm["per_window_mae"], out_dir / "per_window" / f"{method}_seed{seed}.pt")

    seed_summary = aggregate_seed_summary([r for r in validation_rows if int(r["seed"]) >= 0])
    agg_boot: dict[str, dict[str, Any]] = {}
    base_all = torch.cat([per_base_by_seed[s] for s in SEEDS])
    for method, chunks in per_method.items():
        cand = torch.cat(chunks)
        agg_boot[method] = paired_bootstrap(cand, base_all, seed=20260812, samples=args.bootstrap_samples)

    hv_group: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    for row in hv_rows:
        method = row["method"].rsplit("_seed", 1)[0]
        hv_group[(method, int(row["horizon"]), int(row["variable"]))].append(float(row["delta_vs_baseline"]))
    worst = []
    for method in sorted({key[0] for key in hv_group}):
        candidates = [{"method": m, "horizon": h, "variable": v, "delta_vs_baseline_mean": float(np.mean(vals))} for (m, h, v), vals in hv_group.items() if m == method]
        worst.append(max(candidates, key=lambda r: float(r["delta_vs_baseline_mean"])))

    write_csv(out_dir / "validation_per_seed_results.csv", validation_rows)
    write_csv(out_dir / "validation_seed_summary.csv", seed_summary)
    write_csv(out_dir / "validation_weight_covariance_traces.csv", trace_rows)
    write_csv(out_dir / "per_axis_mae.csv", axis_rows)
    write_csv(out_dir / "per_horizon_variable_mae.csv", hv_rows)
    write_json(out_dir / "aggregate_bootstrap_ci.json", agg_boot)

    best = min(seed_summary, key=lambda r: float(r["mae_mean"]))
    selected_fold = selection["selected"]
    selected_boot = agg_boot.get("residual_covariance_selected", {})
    promote = bool(
        best["method"] == "residual_covariance_selected"
        and float(best["mae_mean"]) < BASELINE_MAE
        and selected_boot.get("ci_excludes_zero", False)
        and float(selected_boot.get("ci95_high", 1.0)) < 0.0
        and int(selected_fold["fold_wins"]) >= 3
        and all(float(r["delta_vs_baseline_mean"]) <= 0.001 for r in worst)
    )
    report = {
        "baseline_reference": {"name": BASELINE_NAME, "mae": BASELINE_MAE, "mse": BASELINE_MSE},
        "folds": [{"fold": i, "train_lo": lo, "eval_lo": evlo, "eval_hi": evhi} for i, (lo, evlo, evhi) in enumerate(folds)],
        "selection": selection,
        "validation_seed_summary": seed_summary,
        "aggregate_bootstrap_ci": agg_boot,
        "worst_horizon_variable_regression": worst,
        "best_validation_method": best,
        "promotion_decision": "Promote residual-covariance weighting." if promote else "Do not promote residual-covariance weighting.",
        "promotion_checks": {
            "beats_0.363642": float(best["mae_mean"]) < BASELINE_MAE,
            "selected_fold_wins": int(selected_fold["fold_wins"]),
            "ci_excludes_zero": selected_boot.get("ci_excludes_zero", False),
            "no_major_regression": all(float(r["delta_vs_baseline_mean"]) <= 0.001 for r in worst),
        },
        "safety": {
            "test_cache_loaded": False,
            "validation_used_for_selection": False,
            "causality_rule": "old_start + prediction_length <= current_start",
            "simplex_assertions_passed": True,
        },
        "runtime_sec": time.time() - t0,
        "device": str(device),
        "reproduce_command": f"python experiments\\residual_covariance_weighting_costar\\run_residual_covariance_weighting.py --device {args.device}",
    }
    write_json(out_dir / "final_report.json", report)
    make_report(out_dir, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
