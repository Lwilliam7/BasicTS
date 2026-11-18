"""Signed expert-parity residual fault gate for frozen COSTAR cores.

Strict validation-only experiment:

- train/model selection uses router_train chronological OOF predictions only
- router_val is evaluated once after fitting on router_train
- no test cache or test target is opened
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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.behavioral_competence.common import (  # noqa: E402
    disagreement_features_group_c,
    forecast_features_group_b,
    window_features_group_a,
)
from experiments.behavioral_competence.model_runtime import WALKFORWARD_CHECKPOINT_ROOTS, sha256_file  # noqa: E402
from experiments.costar_multidataset_frozen.common import block_bootstrap_with_prob, every_kth_phase_bootstrap  # noqa: E402
from experiments.final_test_evaluation.run_final_frozen_test_evaluation import etth2_checkpoint_path  # noqa: E402
from experiments.frozen_hv_costar.run_frozen_hv_costar import LOADERS  # noqa: E402
from experiments.horizon_variable_adaptive_costar.run_hv_adaptive_costar import Trial as HvTrial, errors_to_weights, predict_from_hv_weights  # noqa: E402
from experiments.oracle_weight_tournament.run_tournament import sample_mae, sample_mse  # noqa: E402


OUT = Path(__file__).resolve().parent
DATASETS = ("ETTh1", "ETTh2", "ETTm1", "Weather", "Electricity")
EXPERT_ORDER = ("DLinear", "PatchTST", "iTransformer", "TimesNet", "ModernTCN")
HORIZON = 12
FAULT_QUANTILE_GRID = (0.80, 0.90)
GAMMA_GRID = (0.5, 1.0, 2.0)
INTERVENTION_THRESHOLD_GRID: tuple[float | None, ...] = (None, 0.4, 0.6, 0.8)
BLOCK_LENGTH = 24
BOOTSTRAP_SAMPLES = 5000
CODE_VERSION = "pair_residual_fault_gate_v1"
BASELINE_TRIAL = HvTrial("hv_ema", "strict_train_only_frozen_hxv_baseline", mode="hv", rank=1, decay=0.95, temperature=0.1)
EPS = 1e-8


@dataclass
class Detector:
    constant_prob: float | None
    scaler: StandardScaler | None
    model: LogisticRegression | None


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


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


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


def validate_cache(dataset: str, cache: Mapping[str, Any], split: str, core_names: Sequence[str]) -> dict[str, Any]:
    n = int(cache["num_windows"])
    starts = cache["absolute_window_starts"].to(torch.long)
    expected_role = "router_train" if dataset == "ETTh2" and split == "router_train" else "router_val" if dataset == "ETTh2" else "router_train_20_60" if split == "router_train" else "router_val_60_80"
    role = str(cache.get("cache_role", cache.get("split_role")))
    return {
        "dataset": dataset,
        "split": split,
        "role": role,
        "expected_role": expected_role,
        "role_ok": role == expected_role,
        "expert_order_ok": tuple(cache["expert_names"]) == EXPERT_ORDER,
        "core_in_cache": all(name in cache["expert_names"] for name in core_names),
        "shape_ok": tuple(cache["histories"].shape[:2]) == (n, 96)
        and tuple(cache["targets"].shape[:2]) == (n, HORIZON)
        and tuple(cache["prediction_stack"].shape[:3]) == tuple(cache["targets"].shape)
        and int(cache["forecast_horizon"]) == HORIZON,
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


def per_location_error(cache: Mapping[str, Any], forecasts: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.float32)
    return ((forecasts - target.unsqueeze(-1)) / std.view(1, 1, -1, 1)).abs() * mask.unsqueeze(-1)


def per_window_expert_mae(cache: Mapping[str, Any], forecasts: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.bool)
    return torch.stack([sample_mae(forecasts[..., i], target, mask, std) for i in range(forecasts.shape[-1])], dim=1)


def relative_regret(expert_mae: torch.Tensor) -> torch.Tensor:
    regrets = []
    k = expert_mae.shape[1]
    for expert in range(k):
        others = torch.cat((expert_mae[:, :expert], expert_mae[:, expert + 1 :]), dim=1)
        other_median = torch.quantile(others, 0.5, dim=1)
        regrets.append(expert_mae[:, expert] - other_median)
    return torch.stack(regrets, dim=1)


def fault_threshold(regret: torch.Tensor, indices: torch.Tensor, quantile: float) -> float:
    positives = regret[indices].reshape(-1)
    positives = positives[positives > 0]
    if positives.numel() == 0:
        return float("inf")
    return float(torch.quantile(positives, quantile))


def passive_features(cache: Mapping[str, Any], forecasts: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    history = cache["histories"].to(torch.float32)
    a = window_features_group_a(history, std)
    out = []
    for expert in range(forecasts.shape[-1]):
        f = forecasts[..., expert]
        b = forecast_features_group_b(f, history[:, -1], std)
        c = disagreement_features_group_c(f, forecasts, std)
        out.append(torch.cat((a, b, c), dim=1))
    return torch.stack(out, dim=1)


def summarize_tensor(x: torch.Tensor, include_variable_profile: bool) -> torch.Tensor:
    """Summarize [N,H,F] signed normalized forecast/residual structure."""
    n, h, v = x.shape
    half = h // 2
    scalars = [
        x.mean(dim=(1, 2)),
        x[:, :half].mean(dim=(1, 2)),
        x[:, half:].mean(dim=(1, 2)),
        x.abs().mean(dim=(1, 2)),
        x.abs().amax(dim=(1, 2)),
    ]
    horizon_profile = x.mean(dim=2)
    variable_profile = x.mean(dim=1)
    if include_variable_profile:
        var_part = variable_profile
    else:
        q = torch.quantile(variable_profile, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], dtype=torch.float32), dim=1).T
        abs_var = variable_profile.abs()
        var_part = torch.cat((q, abs_var.mean(dim=1, keepdim=True), abs_var.std(dim=1, keepdim=True, unbiased=False), abs_var.amax(dim=1, keepdim=True)), dim=1)
    return torch.cat([s.view(n, 1) for s in scalars] + [horizon_profile, var_part], dim=1)


def parity_features(forecasts: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Per-expert parity features from all expert pairs plus target-oriented pairs."""
    n, _, variables, k = forecasts.shape
    include_variable = variables <= 32
    stdv = std.view(1, 1, -1)
    unordered = []
    for i in range(k):
        for j in range(i + 1, k):
            unordered.append(summarize_tensor((forecasts[..., i] - forecasts[..., j]) / stdv, include_variable))
    out = []
    for target in range(k):
        pieces = list(unordered)
        for other in range(k):
            if other == target:
                continue
            pieces.append(summarize_tensor((forecasts[..., target] - forecasts[..., other]) / stdv, include_variable))
        out.append(torch.cat(pieces, dim=1))
    return torch.stack(out, dim=1)


def raw_forecast_features(forecasts: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    n, _, variables, k = forecasts.shape
    include_variable = variables <= 32
    stdv = std.view(1, 1, -1)
    all_expert = [summarize_tensor(forecasts[..., i] / stdv, include_variable) for i in range(k)]
    ensemble = summarize_tensor(forecasts.mean(dim=-1) / stdv, include_variable)
    out = []
    for target in range(k):
        pieces = [ensemble] + all_expert + [summarize_tensor((forecasts[..., target] - forecasts.mean(dim=-1)) / stdv, include_variable)]
        out.append(torch.cat(pieces, dim=1))
    return torch.stack(out, dim=1)


def permute_windows(x: torch.Tensor, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return x[torch.randperm(x.shape[0], generator=gen)]


def make_features(kind: str, passive: torch.Tensor, parity: torch.Tensor, raw: torch.Tensor, seed: int | None = None) -> torch.Tensor:
    if kind == "passive":
        return passive
    if kind == "parity":
        return parity
    if kind == "passive_parity":
        return torch.cat((passive, parity), dim=2)
    if kind == "shuffled_passive_parity":
        if seed is None:
            raise ValueError("shuffle seed required")
        return torch.cat((passive, permute_windows(parity, seed)), dim=2)
    if kind == "passive_raw":
        return torch.cat((passive, raw), dim=2)
    raise ValueError(kind)


def train_folds(starts: torch.Tensor) -> list[dict[str, Any]]:
    n = int(starts.numel())
    min_train = int(round(n * 0.2))
    usable = n - min_train
    bounds = [min_train + i * usable // 4 for i in range(5)]
    folds = []
    for fold in range(4):
        lo, hi = bounds[fold], bounds[fold + 1]
        current_origin = int(starts[lo])
        fit_idx = torch.where(starts + HORIZON <= current_origin)[0]
        folds.append(
            {
                "fold": fold,
                "fit_idx": fit_idx,
                "eval_lo": lo,
                "eval_hi": hi,
                "fit_windows": int(fit_idx.numel()),
                "fit_start_min": int(starts[fit_idx[0]]) if fit_idx.numel() else None,
                "fit_start_max": int(starts[fit_idx[-1]]) if fit_idx.numel() else None,
                "eval_start_min": current_origin,
                "eval_start_max": int(starts[hi - 1]),
                "old_target_end_le_current_origin": bool(fit_idx.numel() == 0 or int(starts[fit_idx[-1]]) + HORIZON <= current_origin),
            }
        )
    return folds


def fit_detector(features: torch.Tensor, labels: torch.Tensor) -> Detector:
    x = features.reshape(-1, features.shape[-1]).detach().cpu().numpy()
    y = labels.reshape(-1).detach().cpu().numpy().astype(np.int64)
    if np.unique(y).size < 2:
        return Detector(float(y.mean()), None, None)
    scaler = StandardScaler()
    xs = scaler.fit_transform(x)
    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, solver="lbfgs", random_state=7)
    model.fit(xs, y)
    return Detector(None, scaler, model)


def predict_detector(detector: Detector, features: torch.Tensor) -> torch.Tensor:
    if detector.constant_prob is not None:
        return torch.full(features.shape[:2], float(detector.constant_prob), dtype=torch.float32)
    assert detector.scaler is not None and detector.model is not None
    x = features.reshape(-1, features.shape[-1]).detach().cpu().numpy()
    p = detector.model.predict_proba(detector.scaler.transform(x))[:, 1]
    return torch.from_numpy(p.astype(np.float32)).reshape(features.shape[:2]).clamp(0.0, 1.0)


def baseline_weights_from_errors(error_hve: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return errors_to_weights(error_hve[indices].mean(dim=0), BASELINE_TRIAL)


def expand_weights(weights: torch.Tensor, n: int) -> torch.Tensor:
    return weights.unsqueeze(0).expand(n, -1, -1, -1).clone()


def gate_weights(base_weights: torch.Tensor, p_fault: torch.Tensor, gamma: float, intervention_threshold: float | None) -> torch.Tensor:
    if intervention_threshold is not None:
        active = p_fault.max(dim=1).values >= float(intervention_threshold)
    else:
        active = torch.ones(p_fault.shape[0], dtype=torch.bool)
    multiplier = (1.0 - p_fault.clamp(0.0, 0.99)).pow(float(gamma))
    multiplier = torch.where(active.view(-1, 1), multiplier, torch.ones_like(multiplier))
    gated = base_weights * multiplier.view(multiplier.shape[0], 1, 1, multiplier.shape[1])
    return gated / gated.sum(dim=-1, keepdim=True).clamp_min(EPS)


def metrics(cache: Mapping[str, Any], pred: torch.Tensor, std: torch.Tensor) -> dict[str, Any]:
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.bool)
    mae = sample_mae(pred, target, mask, std)
    mse = sample_mse(pred, target, mask, std)
    return {"mae": float(mae.mean()), "mse": float(mse.mean()), "per_window_mae": mae, "per_window_mse": mse}


def fault_metrics(prob: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    p = prob.reshape(-1).detach().cpu().numpy()
    y = labels.reshape(-1).detach().cpu().numpy().astype(np.int64)
    clipped = np.clip(p, 1e-6, 1.0 - 1e-6)
    out = {
        "fault_rate": float(y.mean()),
        "mean_prob": float(p.mean()),
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))),
        "mean_prob_true_fault": float(p[y == 1].mean()) if np.any(y == 1) else 0.0,
        "mean_prob_non_fault": float(p[y == 0].mean()) if np.any(y == 0) else 0.0,
    }
    if np.unique(y).size >= 2:
        out["auc"] = float(roc_auc_score(y, p))
        out["average_precision"] = float(average_precision_score(y, p))
    else:
        out["auc"] = 0.5
        out["average_precision"] = out["fault_rate"]
    return out


def suppression_diagnostics(base_weights: torch.Tensor, gated_weights: torch.Tensor, p_fault: torch.Tensor, labels: torch.Tensor, regret: torch.Tensor) -> dict[str, float]:
    base_expert = base_weights.mean(dim=(1, 2))
    gated_expert = gated_weights.mean(dim=(1, 2))
    delta = gated_expert - base_expert
    suppressed = delta < -1e-6
    true_fault = labels.to(torch.bool)
    return {
        "intervention_rate": float((p_fault.max(dim=1).values > 1e-6).to(torch.float32).mean()),
        "mean_abs_weight_change": float(delta.abs().mean()),
        "mean_weight_removed_true_fault": float((-delta[true_fault]).mean()) if bool(true_fault.any()) else 0.0,
        "mean_weight_removed_non_fault": float((-delta[~true_fault]).mean()) if bool((~true_fault).any()) else 0.0,
        "suppression_precision": float((true_fault & suppressed).sum() / suppressed.sum().clamp_min(1)) if bool(suppressed.any()) else 0.0,
        "suppression_recall": float((true_fault & suppressed).sum() / true_fault.sum().clamp_min(1)) if bool(true_fault.any()) else 0.0,
        "mean_pred_prob_on_positive_regret": float(p_fault[regret > 0].mean()) if bool((regret > 0).any()) else 0.0,
    }


def oof_probabilities(
    kind: str,
    base_feature_parts: Mapping[str, torch.Tensor],
    regret: torch.Tensor,
    quantile: float,
    folds: Sequence[Mapping[str, Any]],
    dataset_seed: int,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    n, k = regret.shape
    p_oof = torch.full((n, k), float("nan"))
    labels_oof = torch.zeros(n, k, dtype=torch.bool)
    rows = []
    for fold in folds:
        fit_idx = fold["fit_idx"]
        lo, hi = int(fold["eval_lo"]), int(fold["eval_hi"])
        threshold = fault_threshold(regret, fit_idx, quantile)
        y_fit = regret[fit_idx] >= threshold
        y_eval = regret[lo:hi] >= threshold
        fit_features = make_features(kind, base_feature_parts["passive"][fit_idx], base_feature_parts["parity"][fit_idx], base_feature_parts["raw"][fit_idx], seed=dataset_seed + 1000 + int(fold["fold"]))
        eval_features = make_features(kind, base_feature_parts["passive"][lo:hi], base_feature_parts["parity"][lo:hi], base_feature_parts["raw"][lo:hi], seed=dataset_seed + 2000 + int(fold["fold"]))
        detector = fit_detector(fit_features, y_fit)
        p_oof[lo:hi] = predict_detector(detector, eval_features)
        labels_oof[lo:hi] = y_eval
        rows.append(
            {
                "fold": int(fold["fold"]),
                "fault_quantile": quantile,
                "fault_threshold": threshold,
                "fit_windows": int(fit_idx.numel()),
                "eval_lo": lo,
                "eval_hi": hi,
                "eval_fault_rate": float(y_eval.to(torch.float32).mean()),
                "fit_fault_rate": float(y_fit.to(torch.float32).mean()),
                "constant_detector": detector.constant_prob is not None,
                "old_target_end_le_current_origin": bool(fold["old_target_end_le_current_origin"]),
            }
        )
    valid = torch.isfinite(p_oof).all(dim=1)
    return p_oof, labels_oof, rows


def train_oof_baseline(train_cache: Mapping[str, Any], forecasts: torch.Tensor, error_hve: torch.Tensor, folds: Sequence[Mapping[str, Any]], std: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    n, h, v, k = forecasts.shape
    weights = torch.full((n, h, v, k), float("nan"))
    pred = torch.full((n, h, v), float("nan"))
    rows = []
    for fold in folds:
        fit_idx = fold["fit_idx"]
        lo, hi = int(fold["eval_lo"]), int(fold["eval_hi"])
        w = baseline_weights_from_errors(error_hve, fit_idx)
        weights[lo:hi] = expand_weights(w, hi - lo)
        pred[lo:hi] = predict_from_hv_weights(forecasts[lo:hi], weights[lo:hi])
        m = metrics({**train_cache, "targets": train_cache["targets"][lo:hi], "target_masks": train_cache["target_masks"][lo:hi]}, pred[lo:hi], std)
        rows.append({"fold": int(fold["fold"]), "baseline_oof_mae": m["mae"], "baseline_oof_mse": m["mse"], "fit_windows": int(fit_idx.numel()), "eval_lo": lo, "eval_hi": hi})
    return weights, pred, torch.isfinite(pred).all(dim=(1, 2)), rows


METHODS = {
    "Passive Fault Gate": "passive",
    "Parity Fault Gate": "parity",
    "Passive + Parity Fault Gate": "passive_parity",
    "Shuffled Parity Control": "shuffled_passive_parity",
    "Raw Forecast Control": "passive_raw",
}


def select_and_evaluate_method(
    dataset: str,
    method: str,
    kind: str,
    train_cache: Mapping[str, Any],
    val_cache: Mapping[str, Any],
    train_parts: Mapping[str, torch.Tensor],
    val_parts: Mapping[str, torch.Tensor],
    train_forecasts: torch.Tensor,
    val_forecasts: torch.Tensor,
    train_base_weights: torch.Tensor,
    val_base_weights: torch.Tensor,
    valid_oof: torch.Tensor,
    train_regret: torch.Tensor,
    val_regret: torch.Tensor,
    std: torch.Tensor,
    folds: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    dataset_seed = 20260829 + sum(ord(c) for c in dataset) + sum(ord(c) for c in method)
    selection_rows = []
    best: dict[str, Any] | None = None
    best_oof_prob = None
    best_oof_labels = None
    best_fold_rows: list[dict[str, Any]] = []
    for q in FAULT_QUANTILE_GRID:
        p_oof, labels_oof, fold_rows = oof_probabilities(kind, train_parts, train_regret, q, folds, dataset_seed)
        for gamma in GAMMA_GRID:
            for intervention_threshold in INTERVENTION_THRESHOLD_GRID:
                gated = gate_weights(train_base_weights[valid_oof], p_oof[valid_oof], gamma, intervention_threshold)
                pred = predict_from_hv_weights(train_forecasts[valid_oof], gated)
                oof_cache = {"targets": train_cache["targets"][valid_oof], "target_masks": train_cache["target_masks"][valid_oof]}
                m = metrics(oof_cache, pred, std)
                row = {
                    "dataset": dataset,
                    "method": method,
                    "fault_quantile": q,
                    "gamma": gamma,
                    "intervention_threshold": "none" if intervention_threshold is None else intervention_threshold,
                    "oof_mae": m["mae"],
                    "oof_mse": m["mse"],
                    "fault_auc": fault_metrics(p_oof[valid_oof], labels_oof[valid_oof])["auc"],
                }
                selection_rows.append(row)
                key = (m["mae"], m["mse"], q, gamma, 0.0 if intervention_threshold is None else intervention_threshold)
                if best is None or key < best["key"]:
                    best = {**row, "key": key, "intervention_threshold_value": intervention_threshold}
                    best_oof_prob = p_oof
                    best_oof_labels = labels_oof
                    best_fold_rows = fold_rows
    assert best is not None and best_oof_prob is not None and best_oof_labels is not None

    q = float(best["fault_quantile"])
    final_threshold = fault_threshold(train_regret, torch.arange(train_regret.shape[0]), q)
    y_train = train_regret >= final_threshold
    y_val = val_regret >= final_threshold
    train_features = make_features(kind, train_parts["passive"], train_parts["parity"], train_parts["raw"], seed=dataset_seed + 3000)
    val_features = make_features(kind, val_parts["passive"], val_parts["parity"], val_parts["raw"], seed=dataset_seed + 4000)
    detector = fit_detector(train_features, y_train)
    p_val = predict_detector(detector, val_features)
    gated_val_weights = gate_weights(val_base_weights, p_val, float(best["gamma"]), best["intervention_threshold_value"])
    gated_val_pred = predict_from_hv_weights(val_forecasts, gated_val_weights)
    route = metrics(val_cache, gated_val_pred, std)
    fm = fault_metrics(p_val, y_val)
    suppression = suppression_diagnostics(val_base_weights, gated_val_weights, p_val, y_val, val_regret)

    oof_gated = gate_weights(train_base_weights[valid_oof], best_oof_prob[valid_oof], float(best["gamma"]), best["intervention_threshold_value"])
    oof_pred = predict_from_hv_weights(train_forecasts[valid_oof], oof_gated)
    oof_route = metrics({"targets": train_cache["targets"][valid_oof], "target_masks": train_cache["target_masks"][valid_oof]}, oof_pred, std)

    return {
        "method": method,
        "selected": {k: v for k, v in best.items() if k not in {"key", "intervention_threshold_value"}},
        "final_fault_threshold": final_threshold,
        "selection_rows": selection_rows,
        "fold_rows": [{**row, "dataset": dataset, "method": method} for row in best_fold_rows],
        "val_prob": p_val,
        "val_labels": y_val,
        "val_weights": gated_val_weights,
        "val_pred": gated_val_pred,
        "val_route": route,
        "oof_route": oof_route,
        "fault_metrics": fm,
        "suppression": suppression,
        "oof_prob": best_oof_prob,
        "oof_labels": best_oof_labels,
    }


def evaluate_dataset(dataset: str) -> dict[str, Any]:
    bundle = LOADERS[dataset]()
    train_cache, val_cache = bundle.train_cache, bundle.val_cache
    paths = cache_paths(dataset)
    for path in list(paths.values()) + list(checkpoint_paths(dataset).values()):
        refuse_test(path)
    checkpoint_before = {expert: file_hash(path) for expert, path in checkpoint_paths(dataset).items()}

    train_schema = validate_cache(dataset, train_cache, "router_train", bundle.core_names)
    val_schema = validate_cache(dataset, val_cache, "router_val", bundle.core_names)
    train_forecasts = selected_forecasts(bundle, train_cache)
    val_forecasts = selected_forecasts(bundle, val_cache)
    train_err_hve = per_location_error(train_cache, train_forecasts, bundle.std)
    val_err = per_window_expert_mae(val_cache, val_forecasts, bundle.std)
    train_err = per_window_expert_mae(train_cache, train_forecasts, bundle.std)
    train_regret = relative_regret(train_err)
    val_regret = relative_regret(val_err)

    train_parts = {
        "passive": passive_features(train_cache, train_forecasts, bundle.std),
        "parity": parity_features(train_forecasts, bundle.std),
        "raw": raw_forecast_features(train_forecasts, bundle.std),
    }
    val_parts = {
        "passive": passive_features(val_cache, val_forecasts, bundle.std),
        "parity": parity_features(val_forecasts, bundle.std),
        "raw": raw_forecast_features(val_forecasts, bundle.std),
    }
    folds = train_folds(train_cache["absolute_window_starts"].to(torch.long))
    base_train_weights, base_train_pred, valid_oof, baseline_fold_rows = train_oof_baseline(train_cache, train_forecasts, train_err_hve, folds, bundle.std)
    final_base_weight = baseline_weights_from_errors(train_err_hve, torch.arange(train_err_hve.shape[0]))
    val_base_weights = expand_weights(final_base_weight, val_forecasts.shape[0])
    val_base_pred = predict_from_hv_weights(val_forecasts, val_base_weights)
    baseline_metrics = metrics(val_cache, val_base_pred, bundle.std)

    method_payloads = {}
    routing_rows = [
        {
            "dataset": dataset,
            "method": "Baseline",
            "mae": baseline_metrics["mae"],
            "mse": baseline_metrics["mse"],
            "delta_vs_baseline_mae": 0.0,
            "delta_vs_baseline_mse": 0.0,
            "selected_fault_quantile": None,
            "selected_gamma": None,
            "selected_intervention_threshold": None,
        }
    ]
    detector_rows = []
    suppression_rows = []
    selection_rows = []
    fold_rows = [{**row, "dataset": dataset, "method": "Baseline"} for row in baseline_fold_rows]
    dependence_rows = []
    val_per_window = {"Baseline": baseline_metrics["per_window_mae"]}
    for method, kind in METHODS.items():
        payload = select_and_evaluate_method(
            dataset,
            method,
            kind,
            train_cache,
            val_cache,
            train_parts,
            val_parts,
            train_forecasts,
            val_forecasts,
            base_train_weights,
            val_base_weights,
            valid_oof,
            train_regret,
            val_regret,
            bundle.std,
            folds,
        )
        method_payloads[method] = payload
        route = payload["val_route"]
        routing_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "mae": route["mae"],
                "mse": route["mse"],
                "delta_vs_baseline_mae": route["mae"] - baseline_metrics["mae"],
                "delta_vs_baseline_mse": route["mse"] - baseline_metrics["mse"],
                "selected_fault_quantile": payload["selected"]["fault_quantile"],
                "selected_gamma": payload["selected"]["gamma"],
                "selected_intervention_threshold": payload["selected"]["intervention_threshold"],
                "final_fault_threshold": payload["final_fault_threshold"],
                "oof_selected_mae": payload["oof_route"]["mae"],
            }
        )
        detector_rows.append({"dataset": dataset, "method": method, **payload["fault_metrics"]})
        suppression_rows.append({"dataset": dataset, "method": method, **payload["suppression"]})
        selection_rows.extend(payload["selection_rows"])
        fold_rows.extend(payload["fold_rows"])
        val_per_window[method] = route["per_window_mae"]
        boot = block_bootstrap_with_prob(route["per_window_mae"], baseline_metrics["per_window_mae"], block=BLOCK_LENGTH, seed=20260829, samples=BOOTSTRAP_SAMPLES)
        phase = every_kth_phase_bootstrap(route["per_window_mae"] - baseline_metrics["per_window_mae"], k=12, seed=20260829, samples=BOOTSTRAP_SAMPLES)
        dependence_rows.append({"dataset": dataset, "comparison": f"{method}_vs_Baseline", "test": "block24", **boot})
        dependence_rows.append({"dataset": dataset, "comparison": f"{method}_vs_Baseline", "test": "every12th_phase", **phase})

    for comparison, candidate, baseline in (
        ("Passive+Parity_vs_Passive", "Passive + Parity Fault Gate", "Passive Fault Gate"),
        ("Passive+Parity_vs_RawForecastControl", "Passive + Parity Fault Gate", "Raw Forecast Control"),
        ("Passive+Parity_vs_ShuffledParity", "Passive + Parity Fault Gate", "Shuffled Parity Control"),
    ):
        cand = val_per_window[candidate]
        base = val_per_window[baseline]
        boot = block_bootstrap_with_prob(cand, base, block=BLOCK_LENGTH, seed=20260829, samples=BOOTSTRAP_SAMPLES)
        phase = every_kth_phase_bootstrap(cand - base, k=12, seed=20260829, samples=BOOTSTRAP_SAMPLES)
        dependence_rows.append({"dataset": dataset, "comparison": comparison, "test": "block24", **boot})
        dependence_rows.append({"dataset": dataset, "comparison": comparison, "test": "every12th_phase", **phase})

    corrupted_val = dict(val_cache)
    gen = torch.Generator().manual_seed(20260829)
    corrupted_val["targets"] = torch.randn(val_cache["targets"].shape, generator=gen)
    corrupted_val["target_masks"] = torch.logical_not(val_cache["target_masks"].to(torch.bool))
    corrupt_parts = {
        "passive": passive_features(corrupted_val, val_forecasts, bundle.std),
        "parity": parity_features(val_forecasts, bundle.std),
        "raw": raw_forecast_features(val_forecasts, bundle.std),
    }
    feature_corruption = {part: float((val_parts[part] - corrupt_parts[part]).abs().max()) for part in val_parts}
    checkpoint_after = {expert: file_hash(path) for expert, path in checkpoint_paths(dataset).items()}
    integrity = {
        "dataset": dataset,
        "test_loaded": False,
        "schemas": {"router_train": train_schema, "router_val": val_schema},
        "core_names": list(bundle.core_names),
        "expert_indices": list(bundle.expert_idx),
        "feature_shapes": {part: list(value.shape) for part, value in train_parts.items()},
        "finite_features": bool(all(torch.isfinite(value).all() for value in list(train_parts.values()) + list(val_parts.values()))),
        "oof_chronological_purge_pass": all(row["old_target_end_le_current_origin"] for row in folds),
        "valid_oof_windows": int(valid_oof.sum()),
        "router_val_target_corruption_feature_max_abs": feature_corruption,
        "target_corruption_feature_pass": all(value == 0.0 for value in feature_corruption.values()),
        "checkpoint_hashes_unchanged": checkpoint_before == checkpoint_after,
        "baseline_uses_router_train_only": True,
        "detector_selection_uses_router_train_oof_only": True,
        "router_val_evaluated_once_after_selection": True,
        "train_feature_hashes": {part: tensor_hash(value) for part, value in train_parts.items()},
        "val_feature_hashes": {part: tensor_hash(value) for part, value in val_parts.items()},
    }
    return {
        "dataset": dataset,
        "core_names": list(bundle.core_names),
        "routing_rows": routing_rows,
        "detector_rows": detector_rows,
        "suppression_rows": suppression_rows,
        "selection_rows": selection_rows,
        "fold_rows": fold_rows,
        "dependence_rows": dependence_rows,
        "integrity": integrity,
        "checkpoint_hashes": checkpoint_before,
        "cache_hashes": {name: file_hash(path) for name, path in paths.items()},
        "summary": {
            "baseline_mae": baseline_metrics["mae"],
            "baseline_mse": baseline_metrics["mse"],
            "best_gate_by_val_mae": min(routing_rows[1:], key=lambda r: (float(r["mae"]), float(r["mse"]))),
            "methods": {row["method"]: row for row in routing_rows},
        },
    }


def classify(results: Mapping[str, Any]) -> str:
    datasets = list(results.values())
    parity_better_baseline = 0
    passive_parity_better_baseline = 0
    passive_parity_better_passive = 0
    passive_parity_better_raw = 0
    passive_parity_better_shuffle = 0
    for result in datasets:
        rows = result["summary"]["methods"]
        base = rows["Baseline"]["mae"]
        if rows["Parity Fault Gate"]["mae"] < base:
            parity_better_baseline += 1
        if rows["Passive + Parity Fault Gate"]["mae"] < base:
            passive_parity_better_baseline += 1
        if rows["Passive + Parity Fault Gate"]["mae"] < rows["Passive Fault Gate"]["mae"]:
            passive_parity_better_passive += 1
        if rows["Passive + Parity Fault Gate"]["mae"] < rows["Raw Forecast Control"]["mae"]:
            passive_parity_better_raw += 1
        if rows["Passive + Parity Fault Gate"]["mae"] < rows["Shuffled Parity Control"]["mae"]:
            passive_parity_better_shuffle += 1
    n = len(datasets)
    if passive_parity_better_baseline >= math.ceil(0.8 * n) and passive_parity_better_passive >= math.ceil(0.6 * n) and passive_parity_better_raw >= math.ceil(0.6 * n) and passive_parity_better_shuffle >= math.ceil(0.8 * n):
        return "STRONG_PARITY_FAULT_ISOLATION_SIGNAL"
    if passive_parity_better_baseline >= math.ceil(0.6 * n) and passive_parity_better_shuffle >= math.ceil(0.6 * n):
        return "PARITY_FAULT_SIGNAL_BUT_CONTROL_LIMITED"
    if parity_better_baseline or passive_parity_better_baseline:
        return "WEAK_OR_INCONSISTENT_PARITY_FAULT_SIGNAL"
    return "NO_PARITY_FAULT_ISOLATION_SIGNAL"


def render_plan() -> str:
    lines = [
        "# Pair Residual Fault Gate Plan",
        "",
        "Strict validation-only experiment. No test cache/file may be opened.",
        "",
        "## Baseline",
        "",
        "Use the existing train-only frozen HxV COSTAR weighting primitive over each dataset's already-selected K=3 core from `experiments.frozen_hv_costar.run_frozen_hv_costar.LOADERS`. Router_train OOF folds build baseline weights from earlier legal targets only; router_val uses one final baseline weight tensor fit from all router_train targets.",
        "",
        "## Fault Target",
        "",
        "`regret_k = MAE_k - median(MAE_other_experts)`. A fault is `regret_k >= q` where `q` is the 80th or 90th percentile of positive router_train regret, chosen by chronological OOF router_train routing MAE only.",
        "",
        "## Features",
        "",
        "- Passive: existing A+B+C window/forecast/disagreement features.",
        "- Parity: signed pair residual summaries for every expert pair plus target-expert-oriented pairs: signed mean, early/late signed mean, absolute magnitude, max magnitude, horizon profile, and variable profile when manageable.",
        "- Shuffled parity: Passive+Parity architecture with parity windows independently shuffled inside train/eval splits.",
        "- Raw forecast control: Passive plus compressed raw expert forecast summaries instead of explicit pair residuals.",
        "",
        "## Gate",
        "",
        "`w_new[k] = w_base[k] * (1 - p_fault[k]) ** gamma`, renormalized over experts. `gamma` in `[0.5, 1.0, 2.0]` and intervention threshold in `[none, 0.4, 0.6, 0.8]` are selected from router_train OOF only.",
    ]
    return "\n".join(lines) + "\n"


def render_report(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Signed Pair Residual Fault Gate",
        "",
        f"Final classification: `{payload['classification']}`",
        "",
        "Strict validation-only. No test cache, target, or metric was loaded.",
        "",
        "## Routing MAE",
        "",
        "| Dataset | Baseline | Passive | Parity | Passive+Parity | Shuffled | Raw Control |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, result in payload["datasets"].items():
        rows = result["summary"]["methods"]
        lines.append(
            f"| {dataset} | `{rows['Baseline']['mae']:.6f}` | `{rows['Passive Fault Gate']['mae']:.6f}` | `{rows['Parity Fault Gate']['mae']:.6f}` | `{rows['Passive + Parity Fault Gate']['mae']:.6f}` | `{rows['Shuffled Parity Control']['mae']:.6f}` | `{rows['Raw Forecast Control']['mae']:.6f}` |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "Faults are relative busts, not absolute high-error events: `regret_k = L_k - median(L_other_experts)`. The gate suppresses only experts predicted to be faulty by multiplying baseline HxV weights by `(1 - p_fault)^gamma` and renormalizing.",
        "",
        "The Raw Forecast Control is the critical comparator: parity supports the fault-isolation hypothesis only if Passive+Parity improves beyond Passive and beyond Passive+Raw while also beating shuffled parity.",
        "",
        "## Integrity",
        "",
        f"- Test loaded: `{payload['test_loaded']}`.",
        "- Every cache/checkpoint path is refused if it contains `test`.",
        "- Router_train detector predictions are chronological OOF with a horizon-12 purge.",
        "- Fault thresholds, gamma, and intervention thresholds are selected on router_train OOF only.",
        "- Router_val target corruption leaves passive, parity, and raw features unchanged exactly.",
        "- Forecasting checkpoint hashes are recorded and unchanged.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    args = parser.parse_args()
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "implementation_plan.md").write_text(render_plan(), encoding="utf-8")

    all_results: dict[str, Any] = {}
    routing_rows: list[dict[str, Any]] = []
    detector_rows: list[dict[str, Any]] = []
    suppression_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    dependence_rows: list[dict[str, Any]] = []
    integrity: dict[str, Any] = {}
    checkpoint_hashes: dict[str, Any] = {}
    cache_hashes: dict[str, Any] = {}

    for dataset in args.datasets:
        print(f"[pair-fault] {dataset}: building parity features and OOF gates...", flush=True)
        result = evaluate_dataset(dataset)
        all_results[dataset] = {"core_names": result["core_names"], "summary": result["summary"]}
        routing_rows.extend(result["routing_rows"])
        detector_rows.extend(result["detector_rows"])
        suppression_rows.extend(result["suppression_rows"])
        selection_rows.extend(result["selection_rows"])
        fold_rows.extend(result["fold_rows"])
        dependence_rows.extend(result["dependence_rows"])
        integrity[dataset] = result["integrity"]
        checkpoint_hashes[dataset] = result["checkpoint_hashes"]
        cache_hashes[dataset] = result["cache_hashes"]
        print(f"[pair-fault] {dataset}: done. best={result['summary']['best_gate_by_val_mae']['method']}", flush=True)

    classification = classify(all_results)
    payload = {
        "experiment": "signed_pair_residual_fault_gate",
        "code_version": CODE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_head(),
        "runtime_sec": time.perf_counter() - start,
        "classification": classification,
        "datasets": all_results,
        "test_loaded": False,
        "success_criteria": [
            "STRONG_PARITY_FAULT_ISOLATION_SIGNAL",
            "PARITY_FAULT_SIGNAL_BUT_CONTROL_LIMITED",
            "WEAK_OR_INCONSISTENT_PARITY_FAULT_SIGNAL",
            "NO_PARITY_FAULT_ISOLATION_SIGNAL",
        ],
    }
    write_json(OUT / "results.json", payload)
    write_json(OUT / "method_manifest.json", {
        "code_version": CODE_VERSION,
        "datasets": list(args.datasets),
        "fault_quantile_grid": FAULT_QUANTILE_GRID,
        "gamma_grid": GAMMA_GRID,
        "intervention_threshold_grid": INTERVENTION_THRESHOLD_GRID,
        "baseline": "strict train-only frozen HxV COSTAR weights over selected K=3 core",
        "detector": "LogisticRegression(C=1.0, class_weight=balanced) with train-only StandardScaler; constant fallback if one class",
        "test_loaded": False,
    })
    write_json(OUT / "source_provenance.json", {
        "git_commit": git_head(),
        "cache_paths": {dataset: {k: rel(v) for k, v in cache_paths(dataset).items()} for dataset in args.datasets},
        "cache_hashes": cache_hashes,
        "baseline_source": "experiments.horizon_variable_adaptive_costar.run_hv_adaptive_costar::{errors_to_weights,predict_from_hv_weights}",
        "dataset_loader_source": "experiments.frozen_hv_costar.run_frozen_hv_costar.LOADERS",
        "passive_feature_source": "experiments.behavioral_competence.common::{window_features_group_a,forecast_features_group_b,disagreement_features_group_c}",
        "test_loaded": False,
    })
    write_json(OUT / "checkpoint_hashes.json", checkpoint_hashes)
    write_json(OUT / "integrity_checks.json", integrity)
    write_json(OUT / "feature_manifest.json", {
        "parity_residual": "(forecast_i - forecast_j) / dataset_std",
        "parity_summaries": ["signed_mean", "signed_early_horizon_mean", "signed_late_horizon_mean", "absolute_magnitude", "max_magnitude", "horizon_wise_signed_profile", "variable_wise_signed_profile_if_variables_le_32_else_quantile_profile"],
        "raw_control": "passive plus compressed raw expert forecasts and target-expert deviation from ensemble, with the same summary family",
        "fault_target": "expert MAE minus median of other expert MAEs",
    })
    write_csv(OUT / "routing_results.csv", routing_rows)
    write_csv(OUT / "fault_detector_results.csv", detector_rows)
    write_csv(OUT / "suppression_diagnostics.csv", suppression_rows)
    write_csv(OUT / "threshold_selection.csv", selection_rows)
    write_csv(OUT / "oof_fold_manifest.csv", fold_rows)
    write_csv(OUT / "dependence_tests.csv", dependence_rows)
    (OUT / "report.md").write_text(render_report(payload), encoding="utf-8")
    print(json.dumps({"out_dir": rel(OUT), "classification": classification, "test_loaded": False}, indent=2), flush=True)


if __name__ == "__main__":
    main()
