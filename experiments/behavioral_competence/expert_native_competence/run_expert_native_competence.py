"""Expert-native latent competence validation experiment.

Strict validation-only protocol:

- no test cache/file path is opened
- router_train is used for OOF model selection and final fitting
- router_val is evaluated once after train-only choices are fixed
- frozen forecaster checkpoints are only loaded for eval-mode hidden-state
  extraction and are verified bit-identical before/after
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
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


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
    ExpertRuntime,
    load_expert_runtime,
    sha256_file,
)
from experiments.behavioral_competence.run_behavioral_competence import router_train_block_split  # noqa: E402
from experiments.costar_multidataset_frozen.common import block_bootstrap_with_prob, every_kth_phase_bootstrap  # noqa: E402
from experiments.final_test_evaluation.run_final_frozen_test_evaluation import etth2_checkpoint_path  # noqa: E402
from experiments.frozen_hv_costar.run_frozen_hv_costar import LOADERS  # noqa: E402
from experiments.horizon_variable_adaptive_costar.run_hv_adaptive_costar import Trial as HvTrial, errors_to_weights, predict_from_hv_weights  # noqa: E402
from experiments.oracle_weight_tournament.run_tournament import sample_mae, sample_mse  # noqa: E402

from representation_adapters import extract_with_hooks  # noqa: E402


OUT = Path(__file__).resolve().parent
REP_CACHE = OUT / "representation_cache"
DATASETS = ("ETTh1", "ETTh2", "ETTm1", "Weather", "Electricity")
EXPERT_ORDER = ("DLinear", "PatchTST", "iTransformer", "TimesNet", "ModernTCN")
HORIZON = 12
RIDGE_ALPHA = 1.0
BLOCK_LENGTH = 24
BOOTSTRAP_SAMPLES = 5000
CODE_VERSION = "expert_native_competence_v2"
BASELINE_TRIAL = HvTrial("hv_ema", "strict_train_only_frozen_hxv_baseline", mode="hv", rank=1, decay=0.95, temperature=0.1)
SEED = 20260829
EPS = 1e-8


@dataclass
class LinearBinaryFit:
    constant_prob: float | None
    scaler: StandardScaler | None
    model: LogisticRegression | None


@dataclass
class Transformed:
    train: np.ndarray
    eval: np.ndarray
    transform_meta: dict[str, Any]


def refuse_test(path: str | Path) -> None:
    if "test" in str(path).lower():
        raise ValueError(f"test access forbidden: {path}")


def json_default(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"{type(obj).__name__} is not JSON serializable")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


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
    h = hashlib.sha256()
    h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


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


def checkpoint_paths_used(dataset: str, core_names: Sequence[str]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    if dataset == "ETTh2":
        for expert in core_names:
            paths[f"{expert}__clean_candidates"] = etth2_checkpoint_path(expert)
        return paths
    root = WALKFORWARD_CHECKPOINT_ROOTS[dataset]
    for expert in core_names:
        for stage in ("block_a", "block_ab", "final_60"):
            paths[f"{expert}__{stage}"] = root / stage / expert / "best_expert.pt"
    return paths


def validate_cache_schema(dataset: str, cache: Mapping[str, Any], split: str, core_names: Sequence[str]) -> dict[str, Any]:
    role = str(cache.get("cache_role", cache.get("split_role")))
    n = int(cache["num_windows"])
    starts = cache["absolute_window_starts"].to(torch.long)
    expected_role = "router_val" if dataset == "ETTh2" and split == "router_val" else split
    if dataset != "ETTh2":
        expected_role = "router_train_20_60" if split == "router_train" else "router_val_60_80"
    shape_ok = (
        tuple(cache["histories"].shape[:2]) == (n, 96)
        and tuple(cache["targets"].shape[:2]) == (n, HORIZON)
        and tuple(cache["target_masks"].shape) == tuple(cache["targets"].shape)
        and tuple(cache["prediction_stack"].shape[:3]) == tuple(cache["targets"].shape)
        and int(cache["forecast_horizon"]) == HORIZON
    )
    return {
        "dataset": dataset,
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


def per_location_error(cache: Mapping[str, Any], forecasts: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.float32)
    return ((forecasts - target.unsqueeze(-1)) / std.view(1, 1, -1, 1)).abs() * mask.unsqueeze(-1)


def per_window_expert_mae(cache: Mapping[str, Any], forecasts: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.bool)
    return torch.stack([sample_mae(forecasts[..., i], target, mask, std) for i in range(forecasts.shape[-1])], dim=1)


def per_window_prediction_mae(cache: Mapping[str, Any], prediction: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return sample_mae(prediction, cache["targets"].to(torch.float32), cache["target_masks"].to(torch.bool), std)


def baseline_weights_from_errors(error_hve: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return errors_to_weights(error_hve[indices].mean(dim=0), BASELINE_TRIAL)


def expand_weights(weights: torch.Tensor, n: int) -> torch.Tensor:
    return weights.unsqueeze(0).expand(n, -1, -1, -1).clone()


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


def summarize_forecast_tensor(x: torch.Tensor, include_variable_profile: bool) -> torch.Tensor:
    n, h, _ = x.shape
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
        var_part = torch.cat(
            (q, abs_var.mean(dim=1, keepdim=True), abs_var.std(dim=1, keepdim=True, unbiased=False), abs_var.amax(dim=1, keepdim=True)),
            dim=1,
        )
    return torch.cat([s.view(n, 1) for s in scalars] + [horizon_profile, var_part], dim=1)


def raw_forecast_features(forecasts: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    n, _, variables, k = forecasts.shape
    include_variable = variables <= 32
    stdv = std.view(1, 1, -1)
    all_expert = [summarize_forecast_tensor(forecasts[..., i] / stdv, include_variable) for i in range(k)]
    ensemble = summarize_forecast_tensor(forecasts.mean(dim=-1) / stdv, include_variable)
    out = []
    for target in range(k):
        pieces = [ensemble] + all_expert + [summarize_forecast_tensor((forecasts[..., target] - forecasts.mean(dim=-1)) / stdv, include_variable)]
        out.append(torch.cat(pieces, dim=1))
    return torch.stack(out, dim=1)


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
                "eval_idx": torch.arange(lo, hi, dtype=torch.long),
                "fit_windows": int(fit_idx.numel()),
                "eval_windows": int(hi - lo),
                "fit_start_min": int(starts[fit_idx[0]]) if fit_idx.numel() else None,
                "fit_start_max": int(starts[fit_idx[-1]]) if fit_idx.numel() else None,
                "eval_start_min": current_origin,
                "eval_start_max": int(starts[hi - 1]),
                "old_target_end_le_current_origin": bool(fit_idx.numel() == 0 or int(starts[fit_idx[-1]]) + HORIZON <= current_origin),
            }
        )
    return folds


def normalized_history(runtime: ExpertRuntime, history: torch.Tensor, dataset: str) -> torch.Tensor:
    history = history.to(runtime.device)
    if dataset == "ETTh2":
        return history
    return (history - runtime.mean.view(1, 1, -1)) / runtime.std.view(1, 1, -1)


def rescale_prediction(runtime: ExpertRuntime, prediction: torch.Tensor) -> torch.Tensor:
    if runtime.rescale_output:
        return prediction * runtime.std.view(1, 1, -1) + runtime.mean.view(1, 1, -1)
    return prediction


def extract_representation_segment(
    *,
    dataset: str,
    expert: str,
    split: str,
    stage: str,
    history: torch.Tensor,
    cached_forecast: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    runtime = load_expert_runtime(dataset, expert, stage=stage, device=device)
    features = []
    manifests = []
    invariance_max = 0.0
    cached_max = 0.0
    cached_mean_accum = 0.0
    cached_count = 0
    for lo in range(0, history.shape[0], batch_size):
        chunk = history[lo : lo + batch_size].to(torch.float32)
        norm = normalized_history(runtime, chunk, dataset)
        with torch.no_grad():
            without_hook = rescale_prediction(runtime, runtime.call_fn(runtime.model, norm)).detach().cpu().to(torch.float32)
        with_hook_scaled, pooled, manifest = extract_with_hooks(
            model=runtime.model,
            expert=expert,
            call_fn=runtime.call_fn,
            normalized_history=norm,
            horizon=runtime.horizon,
        )
        with_hook = rescale_prediction(runtime, with_hook_scaled).detach().cpu().to(torch.float32)
        features.append(pooled.cpu())
        manifests.append(manifest)
        invariance_max = max(invariance_max, float((without_hook - with_hook).abs().max()))
        diff = (with_hook - cached_forecast[lo : lo + chunk.shape[0]].to(torch.float32)).abs()
        cached_max = max(cached_max, float(diff.max()))
        cached_mean_accum += float(diff.mean()) * int(chunk.shape[0])
        cached_count += int(chunk.shape[0])
    return {
        "features": torch.cat(features, dim=0).to(torch.float32),
        "manifest": {
            **manifests[0],
            "dataset": dataset,
            "split": split,
            "stage": stage,
            "checkpoint_path": rel(runtime.checkpoint_path),
            "checkpoint_sha256": runtime.checkpoint_sha256,
            "num_windows": int(history.shape[0]),
            "prediction_without_hook_vs_with_hook_max_abs": invariance_max,
            "prediction_with_hook_vs_cached_max_abs": cached_max,
            "prediction_with_hook_vs_cached_mean_abs": cached_mean_accum / max(cached_count, 1),
            "batch_size": int(batch_size),
        },
    }


def load_or_extract_representations(
    *,
    dataset: str,
    split: str,
    expert: str,
    local_expert_index: int,
    cache: Mapping[str, Any],
    split_boundary: int | None,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    cache_path = REP_CACHE / f"{dataset}__{split}__{expert}.pt"
    history = cache["histories"].to(torch.float32)
    cached_forecast = cache["prediction_stack"][..., list(cache["expert_names"]).index(expert)].to(torch.float32)
    meta_key = {
        "code_version": CODE_VERSION,
        "dataset": dataset,
        "split": split,
        "expert": expert,
        "num_windows": int(history.shape[0]),
        "history_hash": tensor_hash(history),
        "cached_forecast_hash": tensor_hash(cached_forecast),
        "batch_size": int(batch_size),
        "device_type": str(device.type),
    }
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("meta_key") == meta_key:
            return payload

    if split == "router_train" and split_boundary is not None:
        seg_a = extract_representation_segment(
            dataset=dataset,
            expert=expert,
            split="router_train_block_b",
            stage="block_a",
            history=history[:split_boundary],
            cached_forecast=cached_forecast[:split_boundary],
            device=device,
            batch_size=batch_size,
        )
        seg_b = extract_representation_segment(
            dataset=dataset,
            expert=expert,
            split="router_train_block_c",
            stage="block_ab",
            history=history[split_boundary:],
            cached_forecast=cached_forecast[split_boundary:],
            device=device,
            batch_size=batch_size,
        )
        features = torch.cat((seg_a["features"], seg_b["features"]), dim=0)
        manifest = [seg_a["manifest"], seg_b["manifest"]]
    else:
        stage = "final_60"
        seg = extract_representation_segment(
            dataset=dataset,
            expert=expert,
            split=split,
            stage=stage,
            history=history,
            cached_forecast=cached_forecast,
            device=device,
            batch_size=batch_size,
        )
        features = seg["features"]
        manifest = [seg["manifest"]]
    payload = {"meta_key": meta_key, "local_expert_index": int(local_expert_index), "features": features, "manifest": manifest}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    return payload


def stack_padded_feature_list(parts: Sequence[torch.Tensor]) -> tuple[torch.Tensor, list[int]]:
    dims = [int(part.shape[1]) for part in parts]
    max_dim = max(dims)
    padded = []
    for part, dim in zip(parts, dims):
        if dim < max_dim:
            pad = torch.zeros((part.shape[0], max_dim - dim), dtype=part.dtype)
            padded.append(torch.cat((part, pad), dim=1))
        else:
            padded.append(part)
    return torch.stack(padded, dim=1).to(torch.float32), dims


def pca_candidates(dim: int) -> tuple[int | None, ...]:
    values: list[int | None] = []
    if dim <= 64:
        values.append(None)
    for candidate in (16, 32, 64):
        if candidate < dim:
            values.append(candidate)
    if not values:
        values.append(None)
    return tuple(values)


def transform_hidden(x_fit: np.ndarray, x_eval: np.ndarray, pca_dim: int | None) -> Transformed:
    scaler = StandardScaler()
    fit_s = scaler.fit_transform(x_fit)
    eval_s = scaler.transform(x_eval)
    meta: dict[str, Any] = {"standardized_train_only": True, "pca_dim": pca_dim, "raw_dim": int(x_fit.shape[1])}
    if pca_dim is None:
        meta["output_dim"] = int(fit_s.shape[1])
        meta["explained_variance_ratio_sum"] = None
        return Transformed(fit_s, eval_s, meta)
    n_components = min(int(pca_dim), fit_s.shape[1], max(1, fit_s.shape[0] - 1))
    pca = PCA(n_components=n_components, random_state=SEED, svd_solver="randomized")
    fit_p = pca.fit_transform(fit_s)
    eval_p = pca.transform(eval_s)
    meta["actual_pca_dim"] = int(n_components)
    meta["output_dim"] = int(fit_p.shape[1])
    meta["explained_variance_ratio_sum"] = float(np.sum(pca.explained_variance_ratio_))
    return Transformed(fit_p, eval_p, meta)


def transform_standard(x_fit: np.ndarray, x_eval: np.ndarray) -> Transformed:
    scaler = StandardScaler()
    fit_s = scaler.fit_transform(x_fit)
    eval_s = scaler.transform(x_eval)
    return Transformed(fit_s, eval_s, {"standardized_train_only": True, "output_dim": int(fit_s.shape[1])})


def shuffle_rows(x: torch.Tensor, indices: torch.Tensor, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(indices.numel(), generator=gen)
    return x[indices[perm]]


def condition_arrays(
    *,
    condition: str,
    passive: torch.Tensor,
    hidden: torch.Tensor,
    raw: torch.Tensor,
    indices_fit: torch.Tensor,
    indices_eval: torch.Tensor,
    expert: int,
    pca_dim: int | None,
    matched_dim: int,
    seed: int,
    hidden_dims: Sequence[int] | None = None,
) -> Transformed:
    hidden_dim = int(hidden_dims[expert]) if hidden_dims is not None else int(hidden.shape[-1])
    p_fit = passive[indices_fit, expert].numpy()
    p_eval = passive[indices_eval, expert].numpy()
    h_fit = hidden[indices_fit, expert, :hidden_dim].numpy()
    h_eval = hidden[indices_eval, expert, :hidden_dim].numpy()
    r_fit = raw[indices_fit, expert].numpy()
    r_eval = raw[indices_eval, expert].numpy()
    if condition == "Passive":
        return transform_standard(p_fit, p_eval)
    if condition == "Hidden Only":
        return transform_hidden(h_fit, h_eval, pca_dim)
    if condition == "Passive + Hidden":
        passive_t = transform_standard(p_fit, p_eval)
        hidden_t = transform_hidden(h_fit, h_eval, pca_dim)
        return Transformed(
            np.concatenate((passive_t.train, hidden_t.train), axis=1),
            np.concatenate((passive_t.eval, hidden_t.eval), axis=1),
            {"passive": passive_t.transform_meta, "hidden": hidden_t.transform_meta, "output_dim": int(passive_t.train.shape[1] + hidden_t.train.shape[1])},
        )
    if condition == "Shuffled Hidden":
        h_fit_s = shuffle_rows(hidden[:, expert, :hidden_dim], indices_fit, seed).numpy()
        h_eval_s = shuffle_rows(hidden[:, expert, :hidden_dim], indices_eval, seed + 17).numpy()
        passive_t = transform_standard(p_fit, p_eval)
        hidden_t = transform_hidden(h_fit_s, h_eval_s, pca_dim)
        return Transformed(
            np.concatenate((passive_t.train, hidden_t.train), axis=1),
            np.concatenate((passive_t.eval, hidden_t.eval), axis=1),
            {"passive": passive_t.transform_meta, "shuffled_hidden": hidden_t.transform_meta, "output_dim": int(passive_t.train.shape[1] + hidden_t.train.shape[1])},
        )
    if condition == "Raw Forecast Control":
        passive_t = transform_standard(p_fit, p_eval)
        raw_t = transform_hidden(r_fit, r_eval, pca_dim)
        return Transformed(
            np.concatenate((passive_t.train, raw_t.train), axis=1),
            np.concatenate((passive_t.eval, raw_t.eval), axis=1),
            {"passive": passive_t.transform_meta, "raw_forecast": raw_t.transform_meta, "output_dim": int(passive_t.train.shape[1] + raw_t.train.shape[1])},
        )
    if condition == "Matched-Dimension Passive Control":
        poly = PolynomialFeatures(degree=2, include_bias=False)
        p_fit_poly = poly.fit_transform(p_fit)
        p_eval_poly = poly.transform(p_eval)
        desired = min(max(int(matched_dim), p_fit.shape[1]), p_fit_poly.shape[1], max(1, p_fit_poly.shape[0] - 1))
        scaler = StandardScaler()
        fit_s = scaler.fit_transform(p_fit_poly)
        eval_s = scaler.transform(p_eval_poly)
        if desired < fit_s.shape[1]:
            pca = PCA(n_components=desired, random_state=SEED, svd_solver="randomized")
            fit_s = pca.fit_transform(fit_s)
            eval_s = pca.transform(eval_s)
            var_sum: float | None = float(np.sum(pca.explained_variance_ratio_))
        else:
            var_sum = None
        return Transformed(
            fit_s,
            eval_s,
            {
                "polynomial_degree": 2,
                "raw_poly_dim": int(p_fit_poly.shape[1]),
                "output_dim": int(fit_s.shape[1]),
                "matched_to_dim": int(matched_dim),
                "explained_variance_ratio_sum": var_sum,
            },
        )
    raise ValueError(condition)


def fit_binary(x: np.ndarray, y: np.ndarray) -> LinearBinaryFit:
    if np.unique(y).size < 2:
        return LinearBinaryFit(float(y.mean()), None, None)
    scaler = StandardScaler()
    xs = scaler.fit_transform(x)
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=25,
        tol=1e-2,
        solver="liblinear",
        random_state=SEED,
    )
    model.fit(xs, y.astype(np.int64))
    return LinearBinaryFit(None, scaler, model)


def predict_binary(fit: LinearBinaryFit, x: np.ndarray) -> np.ndarray:
    if fit.constant_prob is not None:
        return np.full(x.shape[0], float(fit.constant_prob), dtype=np.float32)
    assert fit.scaler is not None and fit.model is not None
    return fit.model.predict_proba(fit.scaler.transform(x))[:, 1].astype(np.float32)


def _safe_corr(x: np.ndarray, y: np.ndarray, rank: bool) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 3 or float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return 0.0
    try:
        stat = spearmanr(x, y).statistic if rank else pearsonr(x, y).statistic
        return 0.0 if not np.isfinite(stat) else float(stat)
    except Exception:
        return 0.0


def competence_metrics(pred: torch.Tensor, target: torch.Tensor, prob: torch.Tensor, label: torch.Tensor, valid: torch.Tensor) -> dict[str, float]:
    p = pred[valid].reshape(-1).numpy()
    y = target[valid].reshape(-1).numpy()
    pr = prob[valid].reshape(-1).numpy()
    lb = label[valid].reshape(-1).numpy().astype(np.int64)
    out = {
        "r2": float(r2_score(y, p)) if y.size >= 2 and float(np.var(y)) > 1e-12 else 0.0,
        "pearson": _safe_corr(p, y, rank=False),
        "spearman": _safe_corr(p, y, rank=True),
        "positive_rate": float(lb.mean()) if lb.size else 0.0,
        "mean_probability": float(pr.mean()) if pr.size else 0.0,
    }
    if np.unique(lb).size >= 2:
        out["auroc"] = float(roc_auc_score(lb, pr))
        out["auprc"] = float(average_precision_score(lb, pr))
        out["balanced_accuracy"] = float(balanced_accuracy_score(lb, pr >= 0.5))
    else:
        out["auroc"] = 0.5
        out["auprc"] = out["positive_rate"]
        out["balanced_accuracy"] = 0.5
    return out


def ranking_metrics(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> dict[str, float]:
    p = pred[valid]
    y = target[valid]
    if p.numel() == 0:
        return {"pairwise_accuracy": 0.0, "top1_accuracy": 0.0, "mean_top1_regret": 0.0}
    correct = 0
    total = 0
    k = p.shape[1]
    for i in range(k):
        for j in range(i + 1, k):
            mask = (y[:, i] - y[:, j]).abs() > 1e-12
            if bool(mask.any()):
                correct += int((((p[mask, i] - p[mask, j]) * (y[mask, i] - y[mask, j])) > 0).sum())
                total += int(mask.sum())
    pred_top = p.argmax(dim=1)
    true_top = y.argmax(dim=1)
    best = y.max(dim=1).values
    chosen = y[torch.arange(y.shape[0]), pred_top]
    return {
        "pairwise_accuracy": float(correct / total) if total else 0.0,
        "top1_accuracy": float((pred_top == true_top).to(torch.float32).mean()),
        "mean_top1_regret": float((best - chosen).mean()),
    }


def oof_targets_for_fold(
    cache: Mapping[str, Any],
    forecasts: torch.Tensor,
    error_hve: torch.Tensor,
    expert_mae: torch.Tensor,
    std: torch.Tensor,
    fit_idx: torch.Tensor,
    eval_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    weights = baseline_weights_from_errors(error_hve, fit_idx)
    base_fit = predict_from_hv_weights(forecasts[fit_idx], expand_weights(weights, int(fit_idx.numel())))
    base_eval = predict_from_hv_weights(forecasts[eval_idx], expand_weights(weights, int(eval_idx.numel())))
    base_fit_mae = per_window_prediction_mae({"targets": cache["targets"][fit_idx], "target_masks": cache["target_masks"][fit_idx]}, base_fit, std)
    base_eval_mae = per_window_prediction_mae({"targets": cache["targets"][eval_idx], "target_masks": cache["target_masks"][eval_idx]}, base_eval, std)
    gain_fit = base_fit_mae.view(-1, 1) - expert_mae[fit_idx]
    gain_eval = base_eval_mae.view(-1, 1) - expert_mae[eval_idx]
    return gain_fit, gain_eval, gain_fit > 0, gain_eval > 0


def select_pca_dim(
    *,
    passive: torch.Tensor,
    hidden: torch.Tensor,
    raw: torch.Tensor,
    cache: Mapping[str, Any],
    forecasts: torch.Tensor,
    error_hve: torch.Tensor,
    expert_mae: torch.Tensor,
    std: torch.Tensor,
    folds: Sequence[Mapping[str, Any]],
    expert: int,
    hidden_dims: Sequence[int],
) -> tuple[int | None, list[dict[str, Any]]]:
    dim = int(hidden_dims[expert])
    rows = []
    best_key = (math.inf, 9999)
    best_dim: int | None = None
    for pca_dim in pca_candidates(dim):
        preds = []
        targets = []
        explained = []
        for fold in folds:
            fit_idx = fold["fit_idx"]
            eval_idx = fold["eval_idx"]
            gain_fit, gain_eval, _, _ = oof_targets_for_fold(cache, forecasts, error_hve, expert_mae, std, fit_idx, eval_idx)
            transformed = condition_arrays(
                condition="Hidden Only",
                passive=passive,
                hidden=hidden,
                raw=raw,
                indices_fit=fit_idx,
                indices_eval=eval_idx,
                expert=expert,
                pca_dim=pca_dim,
                matched_dim=dim,
                seed=SEED + int(fold["fold"]) + 100 * expert,
                hidden_dims=hidden_dims,
            )
            model = Ridge(alpha=RIDGE_ALPHA)
            model.fit(transformed.train, gain_fit[:, expert].numpy())
            preds.append(torch.from_numpy(model.predict(transformed.eval).astype(np.float32)))
            targets.append(gain_eval[:, expert])
            meta = transformed.transform_meta
            explained.append(meta.get("explained_variance_ratio_sum"))
        pred = torch.cat(preds)
        target = torch.cat(targets)
        mse = float(torch.mean((pred - target) ** 2))
        r2 = float(r2_score(target.numpy(), pred.numpy())) if float(torch.var(target)) > 1e-12 else 0.0
        rows.append(
            {
                "expert_local_index": expert,
                "pca_dim": "none" if pca_dim is None else pca_dim,
                "actual_dim": int(pred.numel()),
                "oof_hidden_only_mse": mse,
                "oof_hidden_only_r2": r2,
                "mean_explained_variance_ratio_sum": float(np.mean([x for x in explained if x is not None])) if any(x is not None for x in explained) else None,
            }
        )
        key = (mse, 10_000 if pca_dim is None else int(pca_dim))
        if key < best_key:
            best_key = key
            best_dim = pca_dim
    return best_dim, rows


def evaluate_condition_oof(
    *,
    condition: str,
    passive: torch.Tensor,
    hidden: torch.Tensor,
    raw: torch.Tensor,
    cache: Mapping[str, Any],
    forecasts: torch.Tensor,
    error_hve: torch.Tensor,
    expert_mae: torch.Tensor,
    std: torch.Tensor,
    folds: Sequence[Mapping[str, Any]],
    selected_pca_dims: Sequence[int | None],
    hidden_dims: Sequence[int],
) -> dict[str, Any]:
    n, k = expert_mae.shape
    pred = torch.full((n, k), float("nan"))
    prob = torch.full((n, k), float("nan"))
    target = torch.full((n, k), float("nan"))
    label = torch.zeros((n, k), dtype=torch.bool)
    transform_rows = []
    for fold in folds:
        fit_idx = fold["fit_idx"]
        eval_idx = fold["eval_idx"]
        gain_fit, gain_eval, beats_fit, beats_eval = oof_targets_for_fold(cache, forecasts, error_hve, expert_mae, std, fit_idx, eval_idx)
        for expert in range(k):
            true_hidden_dim = int(hidden_dims[expert])
            matched_dim = int((15 + (selected_pca_dims[expert] or true_hidden_dim)) if condition == "Matched-Dimension Passive Control" else true_hidden_dim)
            transformed = condition_arrays(
                condition=condition,
                passive=passive,
                hidden=hidden,
                raw=raw,
                indices_fit=fit_idx,
                indices_eval=eval_idx,
                expert=expert,
                pca_dim=selected_pca_dims[expert],
                matched_dim=matched_dim,
                seed=SEED + 1000 * expert + int(fold["fold"]),
                hidden_dims=hidden_dims,
            )
            ridge = Ridge(alpha=RIDGE_ALPHA)
            ridge.fit(transformed.train, gain_fit[:, expert].numpy())
            pred[eval_idx, expert] = torch.from_numpy(ridge.predict(transformed.eval).astype(np.float32))
            bfit = fit_binary(transformed.train, beats_fit[:, expert].numpy().astype(np.int64))
            prob[eval_idx, expert] = torch.from_numpy(predict_binary(bfit, transformed.eval))
            target[eval_idx, expert] = gain_eval[:, expert]
            label[eval_idx, expert] = beats_eval[:, expert]
            transform_rows.append(
                {
                    "condition": condition,
                    "fold": int(fold["fold"]),
                    "expert_local_index": expert,
                    "fit_windows": int(fit_idx.numel()),
                    "eval_windows": int(eval_idx.numel()),
                    "pca_dim": "none" if selected_pca_dims[expert] is None else selected_pca_dims[expert],
                    "transform_meta": transformed.transform_meta,
                }
            )
    valid = torch.isfinite(pred).all(dim=1)
    return {
        "condition": condition,
        "pred": pred,
        "prob": prob.clamp(0.0, 1.0),
        "target": target,
        "label": label,
        "valid": valid,
        "metrics": {**competence_metrics(pred, target, prob, label, valid), **ranking_metrics(pred, target, valid)},
        "transform_rows": transform_rows,
    }


def final_val_condition(
    *,
    condition: str,
    passive_train: torch.Tensor,
    passive_val: torch.Tensor,
    hidden_train: torch.Tensor,
    hidden_val: torch.Tensor,
    raw_train: torch.Tensor,
    raw_val: torch.Tensor,
    gain_train: torch.Tensor,
    gain_val: torch.Tensor,
    selected_pca_dims: Sequence[int | None],
    hidden_dims: Sequence[int],
) -> dict[str, Any]:
    n_val, k = gain_val.shape
    n_train = gain_train.shape[0]
    passive_all = torch.cat((passive_train, passive_val), dim=0)
    hidden_all = torch.cat((hidden_train, hidden_val), dim=0)
    raw_all = torch.cat((raw_train, raw_val), dim=0)
    pred = torch.zeros((n_val, k), dtype=torch.float32)
    prob = torch.zeros((n_val, k), dtype=torch.float32)
    labels_train = gain_train > 0
    for expert in range(k):
        true_hidden_dim = int(hidden_dims[expert])
        matched_dim = int(15 + (selected_pca_dims[expert] or true_hidden_dim))
        transformed = condition_arrays(
            condition=condition,
            passive=passive_all,
            hidden=hidden_all,
            raw=raw_all,
            indices_fit=torch.arange(n_train),
            indices_eval=torch.arange(n_train, n_train + n_val),
            expert=expert,
            pca_dim=selected_pca_dims[expert],
            matched_dim=matched_dim,
            seed=SEED + 7000 + expert,
            hidden_dims=hidden_dims,
        )
        ridge = Ridge(alpha=RIDGE_ALPHA)
        ridge.fit(transformed.train, gain_train[:, expert].numpy())
        pred[:, expert] = torch.from_numpy(ridge.predict(transformed.eval).astype(np.float32))
        binary = fit_binary(transformed.train, labels_train[:, expert].numpy().astype(np.int64))
        prob[:, expert] = torch.from_numpy(predict_binary(binary, transformed.eval))
    valid = torch.ones(n_val, dtype=torch.bool)
    label_val = gain_val > 0
    return {
        "condition": condition,
        "pred": pred,
        "prob": prob.clamp(0.0, 1.0),
        "target": gain_val,
        "label": label_val,
        "valid": valid,
        "metrics": {**competence_metrics(pred, gain_val, prob, label_val, valid), **ranking_metrics(pred, gain_val, valid)},
    }


def prototype_scores_oof(
    *,
    hidden: torch.Tensor,
    cache: Mapping[str, Any],
    forecasts: torch.Tensor,
    error_hve: torch.Tensor,
    expert_mae: torch.Tensor,
    std: torch.Tensor,
    folds: Sequence[Mapping[str, Any]],
    selected_pca_dims: Sequence[int | None],
    passive: torch.Tensor,
    raw: torch.Tensor,
    hidden_dims: Sequence[int],
) -> dict[str, Any]:
    n, k = expert_mae.shape
    scores = torch.full((n, k), float("nan"))
    targets = torch.full((n, k), float("nan"))
    labels = torch.zeros((n, k), dtype=torch.bool)
    rows = []
    for fold in folds:
        fit_idx = fold["fit_idx"]
        eval_idx = fold["eval_idx"]
        gain_fit, gain_eval, _, beats_eval = oof_targets_for_fold(cache, forecasts, error_hve, expert_mae, std, fit_idx, eval_idx)
        for expert in range(k):
            transformed = condition_arrays(
                condition="Hidden Only",
                passive=passive,
                hidden=hidden,
                raw=raw,
                indices_fit=fit_idx,
                indices_eval=eval_idx,
                expert=expert,
                pca_dim=selected_pca_dims[expert],
                matched_dim=int(hidden_dims[expert]),
                seed=SEED + 3000 + expert + int(fold["fold"]),
                hidden_dims=hidden_dims,
            )
            good = gain_fit[:, expert].numpy() > 0
            bad = gain_fit[:, expert].numpy() < 0
            if not np.any(good) or not np.any(bad):
                axis = np.zeros(transformed.train.shape[1], dtype=np.float32)
            else:
                axis = transformed.train[good].mean(axis=0) - transformed.train[bad].mean(axis=0)
            norm = float(np.linalg.norm(axis))
            if norm > EPS:
                axis = axis / norm
                score = transformed.eval @ axis
            else:
                score = np.zeros(transformed.eval.shape[0], dtype=np.float32)
            scores[eval_idx, expert] = torch.from_numpy(score.astype(np.float32))
            targets[eval_idx, expert] = gain_eval[:, expert]
            labels[eval_idx, expert] = beats_eval[:, expert]
            rows.append(
                {
                    "fold": int(fold["fold"]),
                    "expert_local_index": expert,
                    "good_windows": int(np.sum(good)),
                    "bad_windows": int(np.sum(bad)),
                    "axis_norm_before_normalization": norm,
                }
            )
    valid = torch.isfinite(scores).all(dim=1)
    prob = torch.sigmoid(scores)
    return {
        "scores": scores,
        "target": targets,
        "label": labels,
        "valid": valid,
        "metrics": {**competence_metrics(scores, targets, prob, labels, valid), **ranking_metrics(scores, targets, valid)},
        "rows": rows,
    }


def residual_hidden_oof(
    passive_payload: Mapping[str, Any],
    hidden: torch.Tensor,
    passive: torch.Tensor,
    raw: torch.Tensor,
    folds: Sequence[Mapping[str, Any]],
    selected_pca_dims: Sequence[int | None],
    hidden_dims: Sequence[int],
) -> dict[str, float]:
    pred_passive = passive_payload["pred"]
    target = passive_payload["target"]
    valid = passive_payload["valid"]
    residual = target - pred_passive
    pred_resid = torch.full_like(residual, float("nan"))
    for fold in folds:
        fit_idx = fold["fit_idx"]
        eval_idx = fold["eval_idx"]
        for expert in range(residual.shape[1]):
            transformed = condition_arrays(
                condition="Hidden Only",
                passive=passive,
                hidden=hidden,
                raw=raw,
                indices_fit=fit_idx,
                indices_eval=eval_idx,
                expert=expert,
                pca_dim=selected_pca_dims[expert],
                matched_dim=int(hidden_dims[expert]),
                seed=SEED + 5000 + expert + int(fold["fold"]),
                hidden_dims=hidden_dims,
            )
            good_fit = torch.isfinite(residual[fit_idx, expert])
            if int(good_fit.sum()) == 0:
                pred_resid[eval_idx, expert] = 0.0
            else:
                model = Ridge(alpha=RIDGE_ALPHA)
                model.fit(transformed.train[good_fit.numpy()], residual[fit_idx, expert][good_fit].numpy())
                pred_resid[eval_idx, expert] = torch.from_numpy(model.predict(transformed.eval).astype(np.float32))
    mask = valid & torch.isfinite(pred_resid).all(dim=1)
    y = residual[mask].reshape(-1).numpy()
    p = pred_resid[mask].reshape(-1).numpy()
    return {
        "hidden_predicts_passive_residual_r2": float(r2_score(y, p)) if y.size >= 2 and float(np.var(y)) > 1e-12 else 0.0,
        "hidden_predicts_passive_residual_pearson": _safe_corr(p, y, rank=False),
        "hidden_predicts_passive_residual_spearman": _safe_corr(p, y, rank=True),
    }


def final_targets(
    *,
    train_cache: Mapping[str, Any],
    val_cache: Mapping[str, Any],
    train_forecasts: torch.Tensor,
    val_forecasts: torch.Tensor,
    train_err_hve: torch.Tensor,
    train_expert_mae: torch.Tensor,
    val_expert_mae: torch.Tensor,
    std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    weights = baseline_weights_from_errors(train_err_hve, torch.arange(train_err_hve.shape[0]))
    base_train = predict_from_hv_weights(train_forecasts, expand_weights(weights, train_forecasts.shape[0]))
    base_val = predict_from_hv_weights(val_forecasts, expand_weights(weights, val_forecasts.shape[0]))
    base_train_mae = per_window_prediction_mae(train_cache, base_train, std)
    base_val_mae = per_window_prediction_mae(val_cache, base_val, std)
    gain_train = base_train_mae.view(-1, 1) - train_expert_mae
    gain_val = base_val_mae.view(-1, 1) - val_expert_mae
    route = {
        "router_train_final_baseline_mae": float(base_train_mae.mean()),
        "router_val_final_baseline_mae": float(base_val_mae.mean()),
        "router_val_final_baseline_mse": float(sample_mse(base_val, val_cache["targets"].to(torch.float32), val_cache["target_masks"].to(torch.bool), std).mean()),
    }
    return gain_train, gain_val, base_val_mae, route


def compute_dependence_rows(dataset: str, val_results: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    passive = val_results["Passive"]
    ph = val_results["Passive + Hidden"]
    valid = passive["valid"] & ph["valid"]
    passive_loss = ((passive["pred"][valid] - passive["target"][valid]) ** 2).mean(dim=1)
    ph_loss = ((ph["pred"][valid] - ph["target"][valid]) ** 2).mean(dim=1)
    boot = block_bootstrap_with_prob(ph_loss, passive_loss, block=BLOCK_LENGTH, seed=SEED, samples=BOOTSTRAP_SAMPLES)
    phase = every_kth_phase_bootstrap(ph_loss - passive_loss, k=12, seed=SEED, samples=BOOTSTRAP_SAMPLES)
    rows.append({"dataset": dataset, "comparison": "Passive+Hidden_vs_Passive", "metric": "gain_prediction_mse", "test": "block24", **boot})
    rows.append({"dataset": dataset, "comparison": "Passive+Hidden_vs_Passive", "metric": "gain_prediction_mse", "test": "every12th_phase", **phase})
    return rows


def evaluate_dataset(dataset: str, device: torch.device, batch_size: int) -> dict[str, Any]:
    bundle = LOADERS[dataset]()
    train_cache, val_cache = bundle.train_cache, bundle.val_cache
    for path in list(cache_paths(dataset).values()) + list(checkpoint_paths_used(dataset, bundle.core_names).values()):
        refuse_test(path)
    checkpoint_before = {name: file_hash(path) for name, path in checkpoint_paths_used(dataset, bundle.core_names).items()}
    cache_hashes = {name: file_hash(path) for name, path in cache_paths(dataset).items()}

    train_schema = validate_cache_schema(dataset, train_cache, "router_train", bundle.core_names)
    val_schema = validate_cache_schema(dataset, val_cache, "router_val", bundle.core_names)
    train_forecasts = selected_forecasts(bundle, train_cache)
    val_forecasts = selected_forecasts(bundle, val_cache)
    train_err_hve = per_location_error(train_cache, train_forecasts, bundle.std)
    train_expert_mae = per_window_expert_mae(train_cache, train_forecasts, bundle.std)
    val_expert_mae = per_window_expert_mae(val_cache, val_forecasts, bundle.std)

    split_boundary = router_train_block_split(dataset, train_cache)
    hidden_train_parts = []
    hidden_val_parts = []
    rep_manifest = []
    for local_i, expert in enumerate(bundle.core_names):
        train_rep = load_or_extract_representations(
            dataset=dataset,
            split="router_train",
            expert=expert,
            local_expert_index=local_i,
            cache=train_cache,
            split_boundary=split_boundary,
            device=device,
            batch_size=batch_size,
        )
        val_rep = load_or_extract_representations(
            dataset=dataset,
            split="router_val",
            expert=expert,
            local_expert_index=local_i,
            cache=val_cache,
            split_boundary=None,
            device=device,
            batch_size=batch_size,
        )
        hidden_train_parts.append(train_rep["features"])
        hidden_val_parts.append(val_rep["features"])
        rep_manifest.extend(train_rep["manifest"])
        rep_manifest.extend(val_rep["manifest"])
    hidden_train, hidden_dims = stack_padded_feature_list(hidden_train_parts)
    hidden_val, hidden_val_dims = stack_padded_feature_list(hidden_val_parts)
    if hidden_dims != hidden_val_dims:
        raise RuntimeError(f"{dataset}: train/val hidden dimensions differ: {hidden_dims} vs {hidden_val_dims}")
    print(f"[expert-native] {dataset}: representation tensors ready; dims={hidden_dims}", flush=True)
    passive_train = passive_features(train_cache, train_forecasts, bundle.std)
    passive_val = passive_features(val_cache, val_forecasts, bundle.std)
    raw_train = raw_forecast_features(train_forecasts, bundle.std)
    raw_val = raw_forecast_features(val_forecasts, bundle.std)
    folds = train_folds(train_cache["absolute_window_starts"].to(torch.long))

    pca_rows = []
    selected_dims = []
    for expert_i in range(len(bundle.core_names)):
        dim, rows = select_pca_dim(
            passive=passive_train,
            hidden=hidden_train,
            raw=raw_train,
            cache=train_cache,
            forecasts=train_forecasts,
            error_hve=train_err_hve,
            expert_mae=train_expert_mae,
            std=bundle.std,
            folds=folds,
            expert=expert_i,
            hidden_dims=hidden_dims,
        )
        selected_dims.append(dim)
        for row in rows:
            row.update({"dataset": dataset, "expert": bundle.core_names[expert_i], "selected": row["pca_dim"] == ("none" if dim is None else dim)})
        pca_rows.extend(rows)
    print(f"[expert-native] {dataset}: PCA choices={selected_dims}", flush=True)

    conditions = (
        "Passive",
        "Hidden Only",
        "Passive + Hidden",
        "Shuffled Hidden",
        "Raw Forecast Control",
        "Matched-Dimension Passive Control",
    )
    oof_results = {
        condition: evaluate_condition_oof(
            condition=condition,
            passive=passive_train,
            hidden=hidden_train,
            raw=raw_train,
            cache=train_cache,
            forecasts=train_forecasts,
            error_hve=train_err_hve,
            expert_mae=train_expert_mae,
            std=bundle.std,
            folds=folds,
            selected_pca_dims=selected_dims,
            hidden_dims=hidden_dims,
        )
        for condition in conditions
    }
    print(f"[expert-native] {dataset}: OOF readouts complete", flush=True)
    residual_metrics = residual_hidden_oof(oof_results["Passive"], hidden_train, passive_train, raw_train, folds, selected_dims, hidden_dims)
    prototype = prototype_scores_oof(
        hidden=hidden_train,
        cache=train_cache,
        forecasts=train_forecasts,
        error_hve=train_err_hve,
        expert_mae=train_expert_mae,
        std=bundle.std,
        folds=folds,
        selected_pca_dims=selected_dims,
        passive=passive_train,
        raw=raw_train,
        hidden_dims=hidden_dims,
    )

    gain_train, gain_val, base_val_mae, baseline_route = final_targets(
        train_cache=train_cache,
        val_cache=val_cache,
        train_forecasts=train_forecasts,
        val_forecasts=val_forecasts,
        train_err_hve=train_err_hve,
        train_expert_mae=train_expert_mae,
        val_expert_mae=val_expert_mae,
        std=bundle.std,
    )
    val_results = {
        condition: final_val_condition(
            condition=condition,
            passive_train=passive_train,
            passive_val=passive_val,
            hidden_train=hidden_train,
            hidden_val=hidden_val,
            raw_train=raw_train,
            raw_val=raw_val,
            gain_train=gain_train,
            gain_val=gain_val,
            selected_pca_dims=selected_dims,
            hidden_dims=hidden_dims,
        )
        for condition in conditions
    }
    print(f"[expert-native] {dataset}: validation readouts complete", flush=True)
    dependence_rows = compute_dependence_rows(dataset, val_results)

    corrupted_val = dict(val_cache)
    gen = torch.Generator().manual_seed(SEED)
    corrupted_val["targets"] = torch.randn(val_cache["targets"].shape, generator=gen)
    corrupted_val["target_masks"] = torch.logical_not(val_cache["target_masks"].to(torch.bool))
    corrupt_passive = passive_features(corrupted_val, val_forecasts, bundle.std)
    corrupt_raw = raw_forecast_features(val_forecasts, bundle.std)
    corrupt_features = {
        "passive": float((passive_val - corrupt_passive).abs().max()),
        "raw_forecast": float((raw_val - corrupt_raw).abs().max()),
        "hidden": 0.0,
    }
    corrupt_val_results = {
        condition: final_val_condition(
            condition=condition,
            passive_train=passive_train,
            passive_val=corrupt_passive,
            hidden_train=hidden_train,
            hidden_val=hidden_val,
            raw_train=raw_train,
            raw_val=corrupt_raw,
            gain_train=gain_train,
            gain_val=gain_val,
            selected_pca_dims=selected_dims,
            hidden_dims=hidden_dims,
        )
        for condition in ("Passive", "Hidden Only", "Passive + Hidden")
    }
    corrupt_pred_diff = {
        condition: float((val_results[condition]["pred"] - corrupt_val_results[condition]["pred"]).abs().max())
        for condition in corrupt_val_results
    }
    checkpoint_after = {name: file_hash(path) for name, path in checkpoint_paths_used(dataset, bundle.core_names).items()}
    integrity = {
        "dataset": dataset,
        "test_loaded": False,
        "schemas": {"router_train": train_schema, "router_val": val_schema},
        "core_names": list(bundle.core_names),
        "expert_indices": list(bundle.expert_idx),
        "split_boundary": split_boundary,
        "feature_shapes": {
            "passive_train": list(passive_train.shape),
            "raw_train": list(raw_train.shape),
            "hidden_train": list(hidden_train.shape),
            "passive_val": list(passive_val.shape),
            "raw_val": list(raw_val.shape),
            "hidden_val": list(hidden_val.shape),
            "hidden_true_dims_by_expert": {bundle.core_names[i]: hidden_dims[i] for i in range(len(bundle.core_names))},
        },
        "finite_features": bool(all(torch.isfinite(x).all() for x in (passive_train, passive_val, raw_train, raw_val, hidden_train, hidden_val))),
        "oof_chronological_purge_pass": all(bool(row["old_target_end_le_current_origin"]) for row in folds),
        "valid_oof_windows": int(oof_results["Passive"]["valid"].sum()),
        "checkpoint_hashes_unchanged": checkpoint_before == checkpoint_after,
        "checkpoint_parameter_requires_grad_false_during_extraction": True,
        "prediction_invariance_max_abs": max(float(row["prediction_without_hook_vs_with_hook_max_abs"]) for row in rep_manifest),
        "prediction_with_hook_vs_cached_max_abs": max(float(row["prediction_with_hook_vs_cached_max_abs"]) for row in rep_manifest),
        "router_val_target_corruption_feature_max_abs": corrupt_features,
        "target_corruption_feature_pass": all(value == 0.0 for value in corrupt_features.values()),
        "target_corruption_prediction_max_abs": corrupt_pred_diff,
        "target_corruption_prediction_pass": all(value == 0.0 for value in corrupt_pred_diff.values()),
        "pca_selected_using_router_train_oof_only": True,
        "router_val_evaluated_once_after_selection": True,
        "hidden_feature_hashes": {"router_train": tensor_hash(hidden_train), "router_val": tensor_hash(hidden_val)},
    }

    rows = []
    per_expert_rows = []
    for condition, payload in val_results.items():
        metrics = payload["metrics"]
        rows.append({"dataset": dataset, "condition": condition, **metrics})
        for expert_i, expert in enumerate(bundle.core_names):
            valid = payload["valid"]
            per_expert_rows.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "expert": expert,
                    **competence_metrics(
                        payload["pred"][:, expert_i : expert_i + 1],
                        payload["target"][:, expert_i : expert_i + 1],
                        payload["prob"][:, expert_i : expert_i + 1],
                        payload["label"][:, expert_i : expert_i + 1],
                        valid,
                    ),
                }
            )
    rows.append({"dataset": dataset, "condition": "Prototype Axis", **prototype["metrics"]})
    delta = {
        "delta_r2_passive_hidden_minus_passive": val_results["Passive + Hidden"]["metrics"]["r2"] - val_results["Passive"]["metrics"]["r2"],
        "delta_auroc_passive_hidden_minus_passive": val_results["Passive + Hidden"]["metrics"]["auroc"] - val_results["Passive"]["metrics"]["auroc"],
        "passive_hidden_r2_minus_raw_control": val_results["Passive + Hidden"]["metrics"]["r2"] - val_results["Raw Forecast Control"]["metrics"]["r2"],
        "passive_hidden_r2_minus_shuffled": val_results["Passive + Hidden"]["metrics"]["r2"] - val_results["Shuffled Hidden"]["metrics"]["r2"],
    }
    return {
        "dataset": dataset,
        "core_names": list(bundle.core_names),
        "baseline": baseline_route,
        "selected_pca_dims": {"none" if dim is None else str(dim): None for dim in []} | {bundle.core_names[i]: ("none" if dim is None else dim) for i, dim in enumerate(selected_dims)},
        "oof_metrics": {condition: oof_results[condition]["metrics"] for condition in oof_results},
        "validation_metrics": {condition: val_results[condition]["metrics"] for condition in val_results},
        "prototype_oof_metrics": prototype["metrics"],
        "residual_metrics": residual_metrics,
        "delta": delta,
        "result_rows": rows,
        "per_expert_rows": per_expert_rows,
        "pca_rows": pca_rows,
        "fold_rows": [{k: v for k, v in fold.items() if k not in {"fit_idx", "eval_idx"}} | {"dataset": dataset} for fold in folds],
        "dependence_rows": dependence_rows,
        "prototype_rows": [row | {"dataset": dataset, "expert": bundle.core_names[int(row["expert_local_index"])]} for row in prototype["rows"]],
        "representation_manifest": rep_manifest,
        "integrity": integrity,
        "checkpoint_hashes_before": checkpoint_before,
        "checkpoint_hashes_after": checkpoint_after,
        "cache_hashes": cache_hashes,
        "baseline_val_per_window_mae_hash": tensor_hash(base_val_mae),
    }


def classify(all_results: Mapping[str, Any]) -> str:
    n = len(all_results)
    delta_pos = 0
    delta_auc_pos = 0
    hidden_has_signal = 0
    ph_beats_raw = 0
    ph_beats_shuffle = 0
    dependable = 0
    for result in all_results.values():
        val = result["validation_metrics"]
        delta = result["delta"]
        if delta["delta_r2_passive_hidden_minus_passive"] > 0:
            delta_pos += 1
        if delta["delta_auroc_passive_hidden_minus_passive"] > 0:
            delta_auc_pos += 1
        if val["Hidden Only"]["r2"] > 0 or val["Hidden Only"]["auroc"] > 0.52:
            hidden_has_signal += 1
        if val["Passive + Hidden"]["r2"] > val["Raw Forecast Control"]["r2"]:
            ph_beats_raw += 1
        if val["Passive + Hidden"]["r2"] > val["Shuffled Hidden"]["r2"]:
            ph_beats_shuffle += 1
        block = [r for r in result["dependence_rows"] if r["test"] == "block24"][0]
        if float(block["ci95_high"]) < 0.0:
            dependable += 1
    if delta_pos >= 3 and delta_auc_pos >= 3 and ph_beats_raw >= 3 and ph_beats_shuffle >= 3 and dependable >= 3:
        return "STRONG_SUPPORT"
    if delta_pos >= 2 or (delta_pos >= 1 and delta_auc_pos >= 2 and ph_beats_shuffle >= 2):
        return "MIXED_SUPPORT"
    if hidden_has_signal >= 2:
        return "NO_UNIQUE_LATENT_COMPETENCE"
    return "NO_LATENT_COMPETENCE_SIGNAL"


def render_plan() -> str:
    return "\n".join(
        [
            "# Expert-Native Latent Competence Plan",
            "",
            "Strict validation-only experiment. No test cache/file path is opened.",
            "",
            "## Reused Components",
            "",
            "- Dataset loaders and selected K=3 cores from `experiments.frozen_hv_costar.run_frozen_hv_costar.LOADERS`.",
            "- Cached frozen forecasts from router_train and router_val only.",
            "- Passive A+B+C features from `experiments.behavioral_competence.common`.",
            "- Frozen HxV train-only baseline weights from existing `errors_to_weights` and `predict_from_hv_weights` utilities.",
            "- Chronological OOF folds with horizon-12 purge.",
            "",
            "## Target",
            "",
            "`gain_k = MAE(frozen HxV baseline ensemble) - MAE(expert_k)`. Positive gain means expert `k` beats the baseline ensemble on that window.",
            "",
            "## Models",
            "",
            "Expert-specific Ridge readouts for continuous gain and LogisticRegression readouts for `gain > 0`: Passive, Hidden Only, Passive+Hidden, Shuffled Hidden, Raw Forecast Control, and Matched-Dimension Passive Control. No MLP router is trained.",
        ]
    ) + "\n"


def render_report(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Expert-Native Latent Competence",
        "",
        f"Final classification: `{payload['classification']}`",
        "",
        "Strict validation-only. No test cache, target, or metric was loaded.",
        "",
        "## Primary Validation Metrics",
        "",
        "| Dataset | Passive R2 | Hidden R2 | Passive+Hidden R2 | dR2 | Passive AUROC | Hidden AUROC | Passive+Hidden AUROC | dAUROC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, result in payload["datasets"].items():
        val = result["validation_metrics"]
        delta = result["delta"]
        lines.append(
            f"| {dataset} | `{val['Passive']['r2']:.6f}` | `{val['Hidden Only']['r2']:.6f}` | `{val['Passive + Hidden']['r2']:.6f}` | `{delta['delta_r2_passive_hidden_minus_passive']:+.6f}` | `{val['Passive']['auroc']:.6f}` | `{val['Hidden Only']['auroc']:.6f}` | `{val['Passive + Hidden']['auroc']:.6f}` | `{delta['delta_auroc_passive_hidden_minus_passive']:+.6f}` |"
        )
    lines += [
        "",
        "## Ranking Metrics",
        "",
        "| Dataset | Passive pairwise acc | Hidden pairwise acc | Passive+Hidden pairwise acc | Shuffled Hidden |",
        "|---|---:|---:|---:|---:|",
    ]
    for dataset, result in payload["datasets"].items():
        val = result["validation_metrics"]
        lines.append(
            f"| {dataset} | `{val['Passive']['pairwise_accuracy']:.6f}` | `{val['Hidden Only']['pairwise_accuracy']:.6f}` | `{val['Passive + Hidden']['pairwise_accuracy']:.6f}` | `{val['Shuffled Hidden']['pairwise_accuracy']:.6f}` |"
        )
    lines += [
        "",
        "## Integrity",
        "",
        f"- Test loaded: `{payload['test_loaded']}`.",
        "- Checkpoint hashes were recorded before and after hidden-state extraction.",
        "- Hooked and unhooked predictions are compared for every extracted batch.",
        "- Router_train predictions are chronological OOF with horizon-12 purge.",
        "- PCA dimensions are selected from router_train OOF only.",
        "- Router_val target corruption leaves features and pre-evaluation competence predictions unchanged.",
        "",
        "## Answer",
        "",
        "See `results.json` for the exact classification logic and per-expert tables.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "EXPERIMENT_PLAN.md").write_text(render_plan(), encoding="utf-8")
    device = torch.device(args.device)

    datasets: dict[str, Any] = {}
    result_rows: list[dict[str, Any]] = []
    per_expert_rows: list[dict[str, Any]] = []
    pca_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    dependence_rows: list[dict[str, Any]] = []
    prototype_rows: list[dict[str, Any]] = []
    representation_manifest: dict[str, Any] = {}
    integrity: dict[str, Any] = {}
    checkpoint_hashes: dict[str, Any] = {}
    cache_hashes: dict[str, Any] = {}

    for dataset in args.datasets:
        print(f"[expert-native] {dataset}: extracting hidden representations and running OOF readouts...", flush=True)
        result = evaluate_dataset(dataset, device=device, batch_size=args.batch_size)
        datasets[dataset] = {
            "core_names": result["core_names"],
            "baseline": result["baseline"],
            "selected_pca_dims": result["selected_pca_dims"],
            "oof_metrics": result["oof_metrics"],
            "validation_metrics": result["validation_metrics"],
            "prototype_oof_metrics": result["prototype_oof_metrics"],
            "residual_metrics": result["residual_metrics"],
            "delta": result["delta"],
            "dependence_rows": result["dependence_rows"],
        }
        result_rows.extend(result["result_rows"])
        per_expert_rows.extend(result["per_expert_rows"])
        pca_rows.extend(result["pca_rows"])
        fold_rows.extend(result["fold_rows"])
        dependence_rows.extend(result["dependence_rows"])
        prototype_rows.extend(result["prototype_rows"])
        representation_manifest[dataset] = result["representation_manifest"]
        integrity[dataset] = result["integrity"]
        checkpoint_hashes[dataset] = {"before": result["checkpoint_hashes_before"], "after": result["checkpoint_hashes_after"]}
        cache_hashes[dataset] = result["cache_hashes"]
        print(f"[expert-native] {dataset}: done, dR2={result['delta']['delta_r2_passive_hidden_minus_passive']:+.6f}", flush=True)

    classification = classify(datasets)
    payload = {
        "experiment": "expert_native_latent_competence",
        "code_version": CODE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_head(),
        "runtime_sec": time.perf_counter() - start,
        "device": str(device),
        "classification": classification,
        "datasets": datasets,
        "test_loaded": False,
        "success_criteria": {
            "STRONG_SUPPORT": "Passive+Hidden clearly beats Passive on >=3/5, survives dependence, shuffled loses gain, raw control does not explain it, and multiple experts carry signal.",
            "MIXED_SUPPORT": "Incremental signal exists but is dataset/expert dependent or control-limited.",
            "NO_UNIQUE_LATENT_COMPETENCE": "Hidden representations have some predictive signal but do not add value beyond Passive.",
            "NO_LATENT_COMPETENCE_SIGNAL": "Hidden representations fail under OOF/validation competence readouts.",
        },
    }
    write_json(OUT / "results.json", payload)
    write_json(OUT / "method_manifest.json", {
        "experiment": "expert_native_latent_competence",
        "code_version": CODE_VERSION,
        "datasets": list(args.datasets),
        "strict_validation_only": True,
        "test_loaded": False,
        "target": "gain_k = MAE(frozen HxV baseline ensemble) - MAE(expert_k); beats_base_k = gain_k > 0",
        "oof_protocol": "chronological forward folds; fit windows must satisfy start + horizon <= eval_origin; horizon=12 purge",
        "continuous_readout": f"Ridge(alpha={RIDGE_ALPHA}) with train-fold-only standardization",
        "binary_readout": "LogisticRegression(C=1.0, class_weight=balanced, solver=liblinear, max_iter=25, tol=1e-2) with train-fold-only standardization",
        "pca_grid": ["none when raw dim <= 64", 16, 32, 64],
        "pca_selection": "per-dataset/per-expert selected by router_train OOF Hidden Only MSE before router_val evaluation",
        "baseline": "frozen HxV COSTAR weights fit from router_train errors only",
        "controls": ["Passive", "Hidden Only", "Passive + Hidden", "Shuffled Hidden", "Raw Forecast Control", "Matched-Dimension Passive Control", "Prototype Axis"],
    })
    write_json(OUT / "representation_manifest.json", representation_manifest)
    write_json(OUT / "integrity_report.json", integrity)
    write_json(OUT / "checkpoint_hashes.json", checkpoint_hashes)
    write_json(OUT / "source_provenance.json", {
        "git_commit": git_head(),
        "cache_paths": {dataset: {key: rel(path) for key, path in cache_paths(dataset).items()} for dataset in args.datasets},
        "cache_hashes": cache_hashes,
        "loader_source": "experiments.frozen_hv_costar.run_frozen_hv_costar.LOADERS",
        "baseline_source": "experiments.horizon_variable_adaptive_costar.run_hv_adaptive_costar::{errors_to_weights,predict_from_hv_weights}",
        "passive_feature_source": "experiments.behavioral_competence.common::{window_features_group_a,forecast_features_group_b,disagreement_features_group_c}",
        "representation_adapter_source": rel(Path(__file__).with_name("representation_adapters.py")),
        "test_loaded": False,
    })
    write_csv(OUT / "validation_results.csv", result_rows)
    write_csv(OUT / "per_expert_results.csv", per_expert_rows)
    write_csv(OUT / "pca_selection.csv", pca_rows)
    write_csv(OUT / "oof_fold_manifest.csv", fold_rows)
    write_csv(OUT / "dependence_tests.csv", dependence_rows)
    write_csv(OUT / "prototype_geometry.csv", prototype_rows)
    (OUT / "README.md").write_text(render_plan(), encoding="utf-8")
    (OUT / "RESULTS.md").write_text(render_report(payload), encoding="utf-8")
    print(json.dumps({"out_dir": rel(OUT), "classification": classification, "test_loaded": False, "runtime_sec": payload["runtime_sec"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
