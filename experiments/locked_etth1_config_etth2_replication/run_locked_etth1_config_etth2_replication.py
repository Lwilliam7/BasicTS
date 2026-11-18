"""Locked ETTh1-config replications of missing ETTh2 COSTAR methods.

These runs port selected ETTh1 methods to the canonical ETTh2 protocol without
ETTh2 validation tuning:

- MLP residual corrector
- Ridge residual corrector
- Oracle prototype residual
- Dynamic fixed-three, seed 7

The test cache is loaded only after `manifest_before_test.json` is written.
This is an after-final-test replication audit, not a pre-test preregistration.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.chronological_adaptive_costar.run_chronological_adaptive_costar import (  # noqa: E402
    Trial as ChronoTrial,
    chronological_online_weights,
    enforce_observable,
)
from experiments.etth2_train_selected_core.run_etth2_train_selected_core_eval import (  # noqa: E402
    current_base_prediction,
    expert_indices,
    forecasts_for,
    full_model_prediction,
)
from experiments.horizon_variable_adaptive_costar.run_hv_adaptive_costar import (  # noqa: E402
    Trial as HvTrial,
    chronological_hv_weights,
    predict_from_hv_weights,
)
from experiments.oracle_weight_tournament.run_tournament import (  # noqa: E402
    Fixed3WindowDataset as OracleWindowDataset,
    TrialConfig as OracleTrialConfig,
    WeightStudent,
    args_global_weights,
    kmeans,
    oracle_weights_grid,
    sample_mae,
    sample_mse,
    weighted_forecast,
)


OUT_DIR = ROOT / "experiments" / "locked_etth1_config_etth2_replication"
FROZEN_RESULTS_DIR = ROOT / "experiments" / "frozen_model_test_results"
ALL_RESULTS_DIR = ROOT / "experiments" / "all_results_summary"
TRAIN_CACHE = ROOT / "cache" / "costarts_fresh" / "ETTh2_96_12" / "router_train_cache.pt"
VAL_CACHE = ROOT / "cache" / "costarts_fresh" / "ETTh2_96_12" / "router_val_cache.pt"
TEST_CACHE = ROOT / "experiments" / "final_test_evaluation" / "generated" / "caches" / "ETTh2" / "locked_test_cache_v2.pt"
DATASET_DIR = ROOT / "datasets" / "ETTh2"

LABEL = "locked_etth1_config_etth2_replication"
CORE = ("DLinear", "PatchTST", "ModernTCN")
SEEDS = (7, 11, 13, 17, 19)
SCALES = (96, 192, 336, 720)
SINGLE_DLINEAR_VAL = {"mae": 0.28095653653144836, "mse": 0.17149297893047333}
FULL_ADAPTIVE_VAL = {"mae": 0.27683213353157043, "mse": 0.16727977991104126}
SINGLE_DLINEAR_TEST_MAE = 0.30170753598213196
FULL_ADAPTIVE_TEST_MAE = 0.29780814051628113
FIXED_CORE_VAL = {"mae": 0.2808783948421478, "mse": 0.17193281650543213}


@dataclass(frozen=True)
class RidgeConfig:
    ridge: float = 1.0
    alpha: float = 0.1
    clip_multiple: float | None = 0.25
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
    clip_multiple: float | None = 0.25
    epochs: int = 40
    patience: int = 6

    @property
    def name(self) -> str:
        clip = "unclipped" if self.clip_multiple is None else f"clip{self.clip_multiple:g}"
        return f"mlp_seed{self.seed}_h{self.hidden}_alpha{self.alpha:g}_{clip}"


@dataclass(frozen=True)
class DynamicConfig:
    seed: int = 7
    batch_size: int = 512
    epochs: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    entropy_weight: float = 0.0
    embedding_dim: int = 64
    hidden_dim: int = 64
    ablation: str = "full"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_tensor(tensor: torch.Tensor) -> str:
    arr = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def refuse_test_path(path: str | Path) -> None:
    if "test" in str(path).lower():
        raise ValueError(f"Refusing test path before manifest: {path}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_cache(path: Path, expected_role: str, allow_test: bool = False) -> dict[str, Any]:
    if not allow_test:
        refuse_test_path(path)
    cache = torch.load(path, map_location="cpu", weights_only=False)
    role = cache.get("cache_role", cache.get("split_role"))
    if role != expected_role:
        raise ValueError(f"{path}: role={role!r}, expected {expected_role!r}")
    if not allow_test and "test" in str(role).lower():
        raise ValueError(f"Refusing test role before manifest: {role}")
    return cache


def load_series_prefix() -> torch.Tensor:
    train_path = DATASET_DIR / "train_data.npy"
    val_path = DATASET_DIR / "val_data.npy"
    refuse_test_path(train_path)
    refuse_test_path(val_path)
    train = torch.from_numpy(np.load(train_path)).to(torch.float32)
    val = torch.from_numpy(np.load(val_path)).to(torch.float32)
    return torch.cat((train, val), dim=0)


def validate_cache_shape(cache: Mapping[str, Any], role: str) -> None:
    starts = cache["absolute_window_starts"].to(torch.long)
    if not bool(torch.all(starts[1:] > starts[:-1])):
        raise ValueError(f"{role} starts are not chronological")
    if int(cache["forecast_horizon"]) != 12 or int(cache["input_len"]) != 96 or int(cache["num_features"]) != 7:
        raise ValueError(f"{role} shape mismatch")
    expected = {
        "router_train": (2053, 8640, 10692),
        "router_val": (613, 10800, 11412),
        "locked_test": (2773, 11520, 14292),
    }[role]
    if (int(cache["num_windows"]), int(starts.min()), int(starts.max())) != expected:
        raise ValueError(f"{role} windows {int(cache['num_windows'])}, {int(starts.min())}, {int(starts.max())} != {expected}")
    if list(cache["expert_names"]) != ["DLinear", "PatchTST", "iTransformer", "TimesNet", "ModernTCN"]:
        raise ValueError("Unexpected expert order")


def metrics(cache: Mapping[str, Any], pred: torch.Tensor, std: torch.Tensor) -> dict[str, Any]:
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.bool)
    mae = sample_mae(pred, target, mask, std)
    mse = sample_mse(pred, target, mask, std)
    return {"mae": float(mae.mean()), "mse": float(mse.mean()), "per_window_mae": mae, "per_window_mse": mse}


def metric_row(method: str, split: str, cache: Mapping[str, Any], pred: torch.Tensor, std: torch.Tensor, expert_set: Sequence[str], seed: int | str = "mean") -> dict[str, Any]:
    m = metrics(cache, pred, std)
    return {
        "method": method,
        "split": split,
        "seed": seed,
        "expert_set": "+".join(expert_set),
        "mae": m["mae"],
        "mse": m["mse"],
        "prediction_sha256": sha256_tensor(pred),
    }


def assert_history_available(series: torch.Tensor, start: int, scale: int, allow_test_history: bool) -> None:
    if start - scale < 0:
        raise RuntimeError(f"Insufficient history: start={start}, scale={scale}")
    if not allow_test_history and start > 11520:
        raise RuntimeError(f"Unexpected pre-test feature construction for start={start}")
    if start > series.shape[0]:
        raise RuntimeError(f"Forecast start {start} exceeds loaded series prefix length {series.shape[0]}")


def history_summary_for_window(series: torch.Tensor, start: int, variable: int, scale: int, allow_test_history: bool) -> list[float]:
    assert_history_available(series, start, scale, allow_test_history)
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


def causal_residual_stats(starts: torch.Tensor, residual_norm: torch.Tensor, horizon: int, init_residuals_norm: torch.Tensor | None) -> tuple[torch.Tensor, dict[str, Any]]:
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
    allow_test_history: bool,
) -> tuple[torch.Tensor, list[str], dict[str, Any]]:
    forecasts = forecasts_for(cache, expert_indices(cache, CORE))
    target = cache["targets"].to(torch.float32)
    residual_norm = (target - baseline) / std.view(1, 1, -1)
    h, v = baseline.shape[1], baseline.shape[2]
    stats, stat_extra = causal_residual_stats(starts, residual_norm, h, init_residuals_norm)
    rows: list[list[float]] = []
    names: list[str] | None = None
    history_cache: dict[tuple[int, int, int], list[float]] = {}
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
                    if key not in history_cache:
                        history_cache[key] = history_summary_for_window(series, start, vv, scale, allow_test_history)
                    vals = history_cache[key]
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
                        ["baseline_pred", *[f"expert_{name}" for name in CORE]]
                        + ["diff_0_1", "diff_0_2", "diff_1_2", "absdiff_0_1", "absdiff_0_2", "absdiff_1_2"]
                        + ["expert_mean", "expert_std", "expert_range"]
                        + ["causal_residual_mean", "causal_residual_std", "horizon_scaled", "variable_scaled"]
                        + [f"horizon_{k}" for k in range(h)]
                        + [f"variable_{k}" for k in range(v)]
                        + history_names
                    )
                rows.append(row)
    assert names is not None
    x = torch.tensor(rows, dtype=torch.float32)
    expected_rows = int(cache["num_windows"]) * int(cache["forecast_horizon"]) * int(cache["num_features"])
    if x.shape[0] != expected_rows:
        raise RuntimeError(f"Feature rows {x.shape[0]} != {expected_rows}")
    return x, names, stat_extra


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
    return torch.linalg.solve(xb.T @ xb + float(ridge) * eye, xb.T @ y)


def predict_linear(x: torch.Tensor, coef: torch.Tensor) -> torch.Tensor:
    return coef[0] + x @ coef[1:]


def apply_residual_delta(baseline: torch.Tensor, delta_norm_flat: torch.Tensor, std: torch.Tensor, alpha: float, clip_multiple: float | None, residual_train_std_norm: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
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


def build_train_baseline(train_cache: Mapping[str, Any], std: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    idx = expert_indices(train_cache, CORE)
    pred, extra = current_base_prediction(train_cache, train_cache, idx, std)
    return pred, extra


def build_eval_baseline(cache: Mapping[str, Any], train_cache: Mapping[str, Any], std: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    idx = expert_indices(cache, CORE)
    pred, extra = full_model_prediction(cache, train_cache, idx, std)
    return pred, extra


def train_ridge_artifact(train_cache: Mapping[str, Any], train_baseline: torch.Tensor, std: torch.Tensor, series: torch.Tensor) -> dict[str, Any]:
    config = RidgeConfig()
    path = OUT_DIR / "artifacts" / "ridge" / "ridge_artifact.pt"
    if path.exists():
        artifact = load_artifact(path)
        if artifact.get("config") == asdict(config):
            return {"path": path, "config": config, "artifact": artifact}
    starts = train_cache["absolute_window_starts"].to(torch.long)
    x_all, feature_names, stat_extra = build_feature_tensor(train_cache, starts, train_baseline, std, series, init_residuals_norm=None, allow_test_history=False)
    y_all, m_all = flattened_targets(train_cache, train_baseline, std)
    x_fit = x_all[m_all]
    y_fit = y_all[m_all]
    mean, scale = fit_scaler(x_fit)
    coef = fit_ridge(apply_scaler(x_fit, mean, scale), y_fit, config.ridge)
    residual_norm = (train_cache["targets"].to(torch.float32) - train_baseline) / std.view(1, 1, -1)
    artifact = {
        "config": asdict(config),
        "feature_names": feature_names,
        "feature_mean": mean,
        "feature_scale": scale,
        "coef": coef,
        "residual_train_std_norm": residual_norm.std(dim=0, unbiased=False).clamp_min(1e-6),
        "train_residual_norm": residual_norm,
        "stat_extra": stat_extra,
        "x_shape": list(x_all.shape),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, path)
    return {"path": path, "config": config, "artifact": artifact}


def predict_ridge(cache: Mapping[str, Any], baseline: torch.Tensor, train_artifact: Mapping[str, Any], std: torch.Tensor, series: torch.Tensor, allow_test_history: bool) -> tuple[torch.Tensor, dict[str, Any]]:
    starts = cache["absolute_window_starts"].to(torch.long)
    art = train_artifact["artifact"]
    x, names, stat_extra = build_feature_tensor(cache, starts, baseline, std, series, init_residuals_norm=art["train_residual_norm"], allow_test_history=allow_test_history)
    if names != art["feature_names"]:
        raise RuntimeError("Ridge feature names changed")
    delta = predict_linear(apply_scaler(x, art["feature_mean"], art["feature_scale"]), art["coef"])
    pred, extra = apply_residual_delta(baseline, delta, std, float(art["config"]["alpha"]), art["config"]["clip_multiple"], art["residual_train_std_norm"])
    return pred, {**extra, **stat_extra}


def train_mlp_artifact(config: MlpConfig, train_cache: Mapping[str, Any], train_baseline: torch.Tensor, std: torch.Tensor, series: torch.Tensor, device: torch.device) -> dict[str, Any]:
    root = OUT_DIR / "artifacts" / "mlp" / f"seed_{config.seed}"
    path = root / "mlp_artifact.pt"
    if path.exists():
        artifact = load_artifact(path)
        if artifact.get("config") == asdict(config):
            return {"path": path, "config": config, "artifact": artifact}
    set_seed(config.seed)
    starts = train_cache["absolute_window_starts"].to(torch.long)
    residual_norm = (train_cache["targets"].to(torch.float32) - train_baseline) / std.view(1, 1, -1)
    x_all, feature_names, stat_extra = build_feature_tensor(train_cache, starts, train_baseline, std, series, init_residuals_norm=None, allow_test_history=False)
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
    best_epoch = -1
    bad = 0
    curve = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = F.smooth_l1_loss(pred, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            es_loss = F.smooth_l1_loss(model(x_es.to(device)), y_es.to(device)).item()
        curve.append({"epoch": epoch, "train_loss": float(statistics.mean(losses)), "early_stop_loss": es_loss})
        if es_loss < best_loss - 1e-6:
            best_loss = es_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_epoch = epoch
            bad = 0
        else:
            bad += 1
            if bad >= config.patience:
                break
    assert best_state is not None
    artifact = {
        "config": asdict(config),
        "feature_names": feature_names,
        "feature_mean": mean,
        "feature_scale": scale,
        "state_dict": best_state,
        "residual_train_std_norm": residual_norm.std(dim=0, unbiased=False).clamp_min(1e-6),
        "train_residual_norm": residual_norm,
        "best_epoch": best_epoch,
        "early_stop_loss": best_loss,
        "stat_extra": stat_extra,
        "x_shape": list(x_all.shape),
    }
    root.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, path)
    write_csv(root / "training_curve.csv", curve)
    return {"path": path, "config": config, "artifact": artifact}


def predict_mlp(cache: Mapping[str, Any], baseline: torch.Tensor, train_artifact: Mapping[str, Any], std: torch.Tensor, series: torch.Tensor, device: torch.device, allow_test_history: bool) -> tuple[torch.Tensor, dict[str, Any]]:
    art = train_artifact["artifact"]
    x, names, stat_extra = build_feature_tensor(cache, cache["absolute_window_starts"].to(torch.long), baseline, std, series, init_residuals_norm=art["train_residual_norm"], allow_test_history=allow_test_history)
    if names != art["feature_names"]:
        raise RuntimeError("MLP feature names changed")
    model = TinyResidualMlp(len(names), int(art["config"]["hidden"])).to(device)
    model.load_state_dict(art["state_dict"])
    model.eval()
    x = apply_scaler(x, art["feature_mean"], art["feature_scale"])
    outs = []
    with torch.no_grad():
        for i in range(0, x.shape[0], 16384):
            outs.append(model(x[i : i + 16384].to(device)).cpu())
    delta = torch.cat(outs)
    pred, extra = apply_residual_delta(baseline, delta, std, float(art["config"]["alpha"]), art["config"]["clip_multiple"], art["residual_train_std_norm"])
    return pred, {**extra, **stat_extra, "best_epoch": art["best_epoch"], "early_stop_loss": art["early_stop_loss"]}


def forecast_scalars(forecasts: torch.Tensor) -> torch.Tensor:
    a, b, c = forecasts[..., 0], forecasts[..., 1], forecasts[..., 2]
    tensors = [a - b, a - c, b - c, (a - b).abs(), (a - c).abs(), (b - c).abs(), forecasts.var(dim=-1, unbiased=False), forecasts.max(dim=-1).values - forecasts.min(dim=-1).values]
    vals = []
    for x in tensors:
        vals.append(x.mean(dim=(1, 2)))
        vals.append(x.abs().amax(dim=(1, 2)))
    return torch.stack(vals[:12], dim=1)


class DynamicDataset(Dataset):
    def __init__(self, cache: Mapping[str, Any]) -> None:
        self.histories = cache["histories"].to(torch.float32)
        self.forecasts = forecasts_for(cache, expert_indices(cache, CORE))
        self.targets = cache["targets"].to(torch.float32)
        self.masks = cache["target_masks"].to(torch.bool)
        self.starts = cache["absolute_window_starts"].to(torch.long)

    def __len__(self) -> int:
        return int(self.histories.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "history": self.histories[idx],
            "forecasts": self.forecasts[idx],
            "targets": self.targets[idx],
            "target_masks": self.masks[idx],
            "absolute_window_start": self.starts[idx],
            "index": torch.tensor(idx, dtype=torch.long),
        }


class Fixed3DynamicWeightRouter(nn.Module):
    def __init__(self, input_len: int = 96, horizon: int = 12, num_features: int = 7, num_experts: int = 3, embedding_dim: int = 64, hidden_dim: int = 64) -> None:
        super().__init__()
        self.input_len = int(input_len)
        self.horizon = int(horizon)
        self.num_features = int(num_features)
        self.num_experts = int(num_experts)
        self.history_encoder = nn.Sequential(
            nn.Conv1d(num_features, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.GroupNorm(1, hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=4, dilation=2),
            nn.GELU(),
            nn.GroupNorm(1, hidden_dim),
            nn.AdaptiveAvgPool1d(1),
        )
        self.history_projection = nn.Sequential(nn.Linear(hidden_dim, embedding_dim), nn.GELU(), nn.LayerNorm(embedding_dim))
        flat_dim = horizon * num_features
        self.forecast_encoder = nn.Sequential(nn.Linear(flat_dim * 4, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, embedding_dim), nn.GELU(), nn.LayerNorm(embedding_dim))
        self.scalar_encoder = nn.Sequential(nn.Linear(12, embedding_dim), nn.GELU(), nn.LayerNorm(embedding_dim))
        self.head = nn.Sequential(nn.Linear(embedding_dim * 3, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, num_experts))

    def forward(self, history: torch.Tensor, fixed3_forecasts: torch.Tensor, ablation: str = "full") -> dict[str, torch.Tensor]:
        history_rep = self.history_encoder(history.transpose(1, 2)).squeeze(-1)
        history_rep = self.history_projection(history_rep)
        mean_forecast = fixed3_forecasts.mean(dim=-1)
        flattened = torch.cat([fixed3_forecasts[..., index].flatten(1) for index in range(self.num_experts)] + [mean_forecast.flatten(1)], dim=1)
        scalar = forecast_scalars(fixed3_forecasts)
        if ablation == "history_only":
            flattened = torch.zeros_like(flattened)
            scalar = torch.zeros_like(scalar)
        elif ablation == "history_forecasts":
            scalar = torch.zeros_like(scalar)
        elif ablation == "history_disagreement":
            flattened = torch.zeros_like(flattened)
        elif ablation != "full":
            raise ValueError(ablation)
        forecast_rep = self.forecast_encoder(flattened)
        scalar_rep = self.scalar_encoder(scalar)
        logits = self.head(torch.cat((history_rep, forecast_rep, scalar_rep), dim=1))
        return {"logits": logits, "weights": torch.softmax(logits, dim=1)}


def train_dynamic_artifact(train_cache: Mapping[str, Any], std: torch.Tensor, device: torch.device) -> dict[str, Any]:
    config = DynamicConfig()
    root = OUT_DIR / "artifacts" / "dynamic_fixed3_seed7"
    path = root / "dynamic_seed7_artifact.pt"
    if path.exists():
        artifact = load_artifact(path)
        if artifact.get("config") == asdict(config):
            return {"path": path, "config": config, "artifact": artifact}
    set_seed(config.seed)
    model = Fixed3DynamicWeightRouter(
        input_len=int(train_cache["input_len"]),
        horizon=int(train_cache["forecast_horizon"]),
        num_features=int(train_cache["num_features"]),
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loader = DataLoader(DynamicDataset(train_cache), batch_size=config.batch_size, shuffle=True)
    curves = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        for batch in loader:
            hist = batch["history"].to(device)
            forecasts = batch["forecasts"].to(device)
            target = batch["targets"].to(device)
            mask = batch["target_masks"].to(device)
            out = model(hist, forecasts, config.ablation)
            pred = weighted_forecast(forecasts, out["weights"])
            loss = sample_mae(pred, target, mask, std.to(device)).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        train_eval = eval_dynamic(model, train_cache, std, device, config)
        curves.append({"epoch": epoch, "train_loss": float(statistics.mean(losses)), "train_mae": train_eval["mae"], "train_mse": train_eval["mse"]})
    artifact = {"config": asdict(config), "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "training_curves": curves}
    root.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, path)
    write_csv(root / "training_curves.csv", curves)
    return {"path": path, "config": config, "artifact": artifact}


@torch.no_grad()
def eval_dynamic(model: Fixed3DynamicWeightRouter, cache: Mapping[str, Any], std: torch.Tensor, device: torch.device, config: DynamicConfig) -> dict[str, Any]:
    model.eval()
    loader = DataLoader(DynamicDataset(cache), batch_size=config.batch_size, shuffle=False)
    preds = []
    weights = []
    for batch in loader:
        hist = batch["history"].to(device)
        forecasts = batch["forecasts"].to(device)
        out = model(hist, forecasts, config.ablation)
        preds.append(weighted_forecast(forecasts, out["weights"]).cpu())
        weights.append(out["weights"].cpu())
    pred = torch.cat(preds)
    met = metrics(cache, pred, std)
    w = torch.cat(weights)
    met["prediction"] = pred
    met["mean_weights"] = {CORE[i]: float(w[:, i].mean()) for i in range(3)}
    met["weight_std"] = {CORE[i]: float(w[:, i].std(unbiased=False)) for i in range(3)}
    return met


def predict_dynamic(cache: Mapping[str, Any], artifact: Mapping[str, Any], std: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, dict[str, Any]]:
    config = DynamicConfig(**artifact["artifact"]["config"])
    model = Fixed3DynamicWeightRouter(embedding_dim=config.embedding_dim, hidden_dim=config.hidden_dim).to(device)
    model.load_state_dict(artifact["artifact"]["state_dict"])
    out = eval_dynamic(model, cache, std, device, config)
    return out["prediction"], {"mean_weights": out["mean_weights"], "weight_std": out["weight_std"]}


def train_oracle_artifact(config: OracleTrialConfig, train_cache: Mapping[str, Any], std: torch.Tensor, device: torch.device) -> dict[str, Any]:
    root = OUT_DIR / "artifacts" / "oracle_prototype_residual" / f"seed_{config.seed}"
    path = root / "oracle_protores_artifact.pt"
    if path.exists():
        artifact = load_artifact(path)
        if artifact.get("config") == asdict(config):
            return {"path": path, "config": config, "artifact": artifact}
    set_seed(config.seed)
    core_idx = expert_indices(train_cache, CORE)
    forecasts = forecasts_for(train_cache, core_idx)
    targets = train_cache["targets"].to(torch.float32)
    masks = train_cache["target_masks"].to(torch.bool)
    global_weights = torch.tensor(args_global_weights(), dtype=torch.float32)
    teacher, teacher_mae = oracle_weights_grid(forecasts, targets, masks, std, global_weights, config.teacher_lambda, step=0.02)
    prototypes, proto_labels = kmeans(teacher, config.num_prototypes, config.seed)
    ds = OracleWindowDataset({**train_cache, "prediction_stack": train_cache["prediction_stack"]}, core_idx)
    model = WeightStudent(
        global_weights,
        int(train_cache["input_len"]),
        int(train_cache["forecast_horizon"]),
        int(train_cache["num_features"]),
        mode="prototype_residual",
        num_prototypes=config.num_prototypes,
        rank=config.rank,
        residual_scale=config.residual_scale,
        feature_mix=config.feature_mix,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr)
    loader = DataLoader(ds, batch_size=config.batch_size, shuffle=True)
    curves = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        for batch in loader:
            hist = batch["history"].to(device)
            fcast = batch["forecasts"].to(device)
            target = batch["target"].to(device)
            mask = batch["mask"].to(device)
            idx = batch["index"]
            out = model(hist, fcast, prototypes=prototypes)
            weights = out["weights"]
            teach = teacher[idx].to(device)
            teacher_loss = F.smooth_l1_loss(weights, teach) + F.cross_entropy(out["logits"], proto_labels[idx].to(device))
            pred = weighted_forecast(fcast, weights)
            forecast_loss = sample_mae(pred, target, mask, std.to(device)).mean()
            residual_loss = (weights - global_weights.to(device).view(1, 3)).square().mean()
            loss = config.forecast_weight * forecast_loss + config.teacher_weight * teacher_loss + config.residual_weight * residual_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        train_eval = eval_oracle_model(model, train_cache, core_idx, prototypes, std, device)
        curves.append({"epoch": epoch, "train_loss": float(statistics.mean(losses)), "train_mae": train_eval["mae"], "train_mse": train_eval["mse"]})
    artifact = {
        "config": asdict(config),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "prototypes": prototypes,
        "teacher_sha256": sha256_tensor(teacher),
        "teacher_train_mae_mean": float(teacher_mae.mean()),
        "training_curves": curves,
    }
    root.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, path)
    write_csv(root / "training_curves.csv", curves)
    return {"path": path, "config": config, "artifact": artifact}


@torch.no_grad()
def eval_oracle_model(model: WeightStudent, cache: Mapping[str, Any], core_idx: Sequence[int], prototypes: torch.Tensor, std: torch.Tensor, device: torch.device) -> dict[str, Any]:
    model.eval()
    ds = OracleWindowDataset(cache, core_idx)
    loader = DataLoader(ds, batch_size=1024, shuffle=False)
    preds = []
    weights = []
    for batch in loader:
        out = model(batch["history"].to(device), batch["forecasts"].to(device), prototypes=prototypes)
        preds.append(weighted_forecast(batch["forecasts"].to(device), out["weights"]).cpu())
        weights.append(out["weights"].cpu())
    pred = torch.cat(preds)
    met = metrics(cache, pred, std)
    w = torch.cat(weights)
    met["prediction"] = pred
    met["mean_weights"] = {CORE[i]: float(w[:, i].mean()) for i in range(3)}
    met["weight_std"] = {CORE[i]: float(w[:, i].std(unbiased=False)) for i in range(3)}
    return met


def predict_oracle(cache: Mapping[str, Any], train_cache: Mapping[str, Any], artifact: Mapping[str, Any], std: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, dict[str, Any]]:
    config = OracleTrialConfig(**artifact["artifact"]["config"])
    core_idx = expert_indices(cache, CORE)
    model = WeightStudent(
        torch.tensor(args_global_weights(), dtype=torch.float32),
        int(train_cache["input_len"]),
        int(train_cache["forecast_horizon"]),
        int(train_cache["num_features"]),
        mode="prototype_residual",
        num_prototypes=config.num_prototypes,
        rank=config.rank,
        residual_scale=config.residual_scale,
        feature_mix=config.feature_mix,
    ).to(device)
    model.load_state_dict(artifact["artifact"]["state_dict"])
    out = eval_oracle_model(model, cache, core_idx, artifact["artifact"]["prototypes"], std, device)
    return out["prediction"], {"mean_weights": out["mean_weights"], "weight_std": out["weight_std"], "teacher_sha256": artifact["artifact"]["teacher_sha256"]}


def build_chrono_oof_baseline(train_cache: Mapping[str, Any], std: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    starts = train_cache["absolute_window_starts"].to(torch.long)
    h = int(train_cache["forecast_horizon"])
    idx = expert_indices(train_cache, CORE)
    forecasts = forecasts_for(train_cache, idx)
    target = train_cache["targets"].to(torch.float32)
    mask = train_cache["target_masks"].to(torch.float32)
    err = ((forecasts - target.unsqueeze(-1)) / std.view(1, 1, -1, 1)).abs() * mask.unsqueeze(-1)
    train_mean = err.mean(dim=(0, 1, 2))
    online, online_extra = chronological_online_weights(
        starts,
        err.mean(dim=(1, 2)),
        h,
        ChronoTrial("ema", "ema_decay0.97_temp0.1", decay=0.97, temperature=0.1),
        train_mean,
        mode="ema",
    )
    static = torch.full_like(online, 1.0 / 3.0)
    chrono_weights = 0.5 * static + 0.5 * online
    chrono_weights = chrono_weights / chrono_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    chrono_pred = weighted_forecast(forecasts, chrono_weights)
    hv_weights, hv_extra = chronological_hv_weights(
        starts=starts,
        train_err_mean=err.mean(dim=0),
        val_err=err,
        horizon=h,
        trial=HvTrial("hv_ema", "hvema_lowrank1_decay0.95_temp0.1", mode="hv_lowrank", rank=1, decay=0.95, temperature=0.1),
    )
    hv_pred = predict_from_hv_weights(forecasts, hv_weights)
    return 0.25 * chrono_pred + 0.75 * hv_pred, {"chrono_oof_online_updates": online_extra["num_updates"], "chrono_oof_hv_updates": hv_extra["num_updates"]}


def aggregate_seed_predictions(preds: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.stack(list(preds)).mean(dim=0)


def aggregate_seed_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["method"]), []).append(row)
    out = []
    for method, group in grouped.items():
        maes = [float(r["mae"]) for r in group]
        mses = [float(r["mse"]) for r in group]
        out.append({
            "method": method,
            "seeds": len(group),
            "mae_mean": float(statistics.mean(maes)),
            "mae_std": float(statistics.pstdev(maes)) if len(maes) > 1 else 0.0,
            "mse_mean": float(statistics.mean(mses)),
            "mse_std": float(statistics.pstdev(mses)) if len(mses) > 1 else 0.0,
        })
    return sorted(out, key=lambda r: float(r["mae_mean"]))


def train_validate(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_cache = load_cache(TRAIN_CACHE, "router_train", allow_test=False)
    val_cache = load_cache(VAL_CACHE, "router_val", allow_test=False)
    validate_cache_shape(train_cache, "router_train")
    validate_cache_shape(val_cache, "router_val")
    std = torch.ones(int(train_cache["num_features"]), dtype=torch.float32)
    device = torch.device(args.device)
    series = load_series_prefix()
    if series.shape[0] != 11520:
        raise RuntimeError("Pre-test ETTh2 series prefix must stop at validation end 11520")
    train_baseline, train_base_extra = build_train_baseline(train_cache, std)
    val_baseline, val_base_extra = build_eval_baseline(val_cache, train_cache, std)
    train_oof_baseline, oof_extra = build_chrono_oof_baseline(train_cache, std)
    ridge_art = train_ridge_artifact(train_cache, train_oof_baseline, std, series)
    ridge_val_pred, ridge_extra = predict_ridge(val_cache, val_baseline, ridge_art, std, series, allow_test_history=False)
    mlp_artifacts = [train_mlp_artifact(MlpConfig(seed=seed), train_cache, train_oof_baseline, std, series, device) for seed in SEEDS]
    mlp_val_preds = []
    mlp_val_rows = []
    for art in mlp_artifacts:
        pred, extra = predict_mlp(val_cache, val_baseline, art, std, series, device, allow_test_history=False)
        mlp_val_preds.append(pred)
        row = metric_row("MLP residual corrector", "router_val", val_cache, pred, std, CORE, seed=art["config"].seed)
        row.update(extra)
        mlp_val_rows.append(row)
    oracle_configs = [
        OracleTrialConfig(
            family="prototype_residual",
            name=f"final_phase2_protores_lam0.01_k16_scale0.3_rw0.001_seed{seed}",
            seed=seed,
            num_prototypes=16,
            teacher_lambda=0.01,
            residual_scale=0.30,
            residual_weight=0.001,
            epochs=10,
        )
        for seed in SEEDS
    ]
    oracle_artifacts = [train_oracle_artifact(config, train_cache, std, device) for config in oracle_configs]
    oracle_val_preds = []
    oracle_val_rows = []
    for art in oracle_artifacts:
        pred, extra = predict_oracle(val_cache, train_cache, art, std, device)
        oracle_val_preds.append(pred)
        row = metric_row("Oracle prototype residual", "router_val", val_cache, pred, std, CORE, seed=art["config"].seed)
        row.update(extra)
        oracle_val_rows.append(row)
    dynamic_art = train_dynamic_artifact(train_cache, std, device)
    dynamic_val_pred, dynamic_extra = predict_dynamic(val_cache, dynamic_art, std, device)

    val_rows = [
        metric_row("Full adaptive model", "router_val", val_cache, val_baseline, std, CORE),
        metric_row("Ridge residual corrector", "router_val", val_cache, ridge_val_pred, std, CORE),
        metric_row("MLP residual corrector", "router_val", val_cache, aggregate_seed_predictions(mlp_val_preds), std, CORE),
        metric_row("Oracle prototype residual", "router_val", val_cache, aggregate_seed_predictions(oracle_val_preds), std, CORE),
        metric_row("Dynamic fixed-three, seed 7", "router_val", val_cache, dynamic_val_pred, std, CORE, seed=7),
    ]
    val_rows[1].update(ridge_extra)
    val_rows[-1].update(dynamic_extra)
    write_csv(OUT_DIR / "validation_results.csv", val_rows)
    write_csv(OUT_DIR / "validation_per_seed_results.csv", mlp_val_rows + oracle_val_rows)
    write_csv(OUT_DIR / "validation_seed_summary.csv", aggregate_seed_rows(mlp_val_rows + oracle_val_rows))

    manifest = {
        "label": LABEL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": "ETTh2 locked replication of ETTh1 method configs after the earlier final test evaluation; not claimed as pre-test preregistration.",
        "test_loaded_before_manifest": False,
        "test_metrics_seen_before_manifest": True,
        "dataset": "ETTh2",
        "core": list(CORE),
        "splits": {
            "router_train": [8640, 10800],
            "router_val": [10800, 11520],
            "test": [11520, 14400],
        },
        "canonical_protocol": {
            "input_len": 96,
            "horizon": 12,
            "variables": 7,
            "metric": "sample_mae/sample_mse",
            "std": [1.0] * 7,
            "inverse_transform": "none",
        },
        "cache_paths": {
            "router_train": str(TRAIN_CACHE),
            "router_val": str(VAL_CACHE),
            "test_planned_after_manifest": str(TEST_CACHE),
        },
        "cache_hashes": {
            "router_train": sha256_file(TRAIN_CACHE),
            "router_val": sha256_file(VAL_CACHE),
        },
        "baseline_artifacts": {
            "train_base_extra": train_base_extra,
            "val_base_extra": val_base_extra,
            "oof_baseline": oof_extra,
        },
        "methods": {
            "ridge_residual_corrector": {
                "source": "experiments/residual_correction_costar/run_residual_correction_experiments.py",
                "locked_etth1_config": asdict(RidgeConfig()),
                "artifact_path": str(ridge_art["path"]),
                "artifact_sha256": sha256_file(ridge_art["path"]),
            },
            "mlp_residual_corrector": {
                "source": "experiments/residual_correction_costar/run_residual_correction_experiments.py",
                "locked_etth1_config": asdict(MlpConfig(seed=7)),
                "seeds": list(SEEDS),
                "artifact_paths": [str(a["path"]) for a in mlp_artifacts],
                "artifact_sha256": {str(a["config"].seed): sha256_file(a["path"]) for a in mlp_artifacts},
            },
            "oracle_prototype_residual": {
                "source": "experiments/oracle_weight_tournament/run_tournament.py",
                "locked_etth1_config": {
                    "family": "prototype_residual",
                    "teacher_lambda": 0.01,
                    "num_prototypes": 16,
                    "residual_scale": 0.30,
                    "residual_weight": 0.001,
                    "epochs": 10,
                    "global_weights": list(args_global_weights()),
                },
                "seeds": list(SEEDS),
                "artifact_paths": [str(a["path"]) for a in oracle_artifacts],
                "artifact_sha256": {str(a["config"].seed): sha256_file(a["path"]) for a in oracle_artifacts},
            },
            "dynamic_fixed_three_seed7": {
                "source": "scripts/train_costarts_fixed3_dynamic_weighting.py",
                "locked_etth1_config": asdict(DynamicConfig()),
                "etth1_selected_epoch_reused": 2,
                "artifact_path": str(dynamic_art["path"]),
                "artifact_sha256": sha256_file(dynamic_art["path"]),
            },
        },
        "validation_results": val_rows,
        "validation_per_seed": mlp_val_rows + oracle_val_rows,
        "checks": {
            "router_train_only_fitting": True,
            "router_val_not_used_for_training_or_checkpoint_selection": True,
            "test_cache_not_loaded_before_manifest": True,
            "causal_residual_rule": "old_start + horizon <= current_start",
            "long_history_features_pre_test_from_train_plus_val_only": True,
            "tensor_shapes_checked": True,
            "config_and_artifact_hashes_recorded": True,
        },
        "runtime_sec_train_validate": time.perf_counter() - started,
    }
    write_json(OUT_DIR / "manifest_before_test.json", manifest)
    print(json.dumps({"phase": "train_validate", "manifest": str(OUT_DIR / "manifest_before_test.json"), "validation_results": val_rows}, indent=2))


def load_artifact(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def evaluate_test(args: argparse.Namespace) -> None:
    manifest_path = OUT_DIR / "manifest_before_test.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Run train_validate before test evaluation")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("test_loaded_before_manifest") is not False:
        raise RuntimeError("Manifest does not show pre-test no-load state")
    started = time.perf_counter()
    train_cache = load_cache(TRAIN_CACHE, "router_train", allow_test=False)
    val_cache = load_cache(VAL_CACHE, "router_val", allow_test=False)
    test_cache = load_cache(TEST_CACHE, "locked_test", allow_test=True)
    validate_cache_shape(train_cache, "router_train")
    validate_cache_shape(val_cache, "router_val")
    validate_cache_shape(test_cache, "locked_test")
    std = torch.ones(int(test_cache["num_features"]), dtype=torch.float32)
    device = torch.device(args.device)
    train_val_series = load_series_prefix()
    test_series = torch.cat((train_val_series, torch.from_numpy(np.load(DATASET_DIR / "test_data.npy")).to(torch.float32)), dim=0)
    test_baseline, _ = build_eval_baseline(test_cache, train_cache, std)

    ridge_art = {"artifact": load_artifact(Path(manifest["methods"]["ridge_residual_corrector"]["artifact_path"]))}
    ridge_pred, ridge_extra = predict_ridge(test_cache, test_baseline, ridge_art, std, test_series, allow_test_history=True)
    mlp_preds = []
    mlp_seed_rows = []
    for path in manifest["methods"]["mlp_residual_corrector"]["artifact_paths"]:
        art = {"artifact": load_artifact(Path(path))}
        pred, extra = predict_mlp(test_cache, test_baseline, art, std, test_series, device, allow_test_history=True)
        mlp_preds.append(pred)
        seed = int(art["artifact"]["config"]["seed"])
        row = metric_row("MLP residual corrector", "locked_test", test_cache, pred, std, CORE, seed=seed)
        row.update(extra)
        mlp_seed_rows.append(row)
    oracle_preds = []
    oracle_seed_rows = []
    for path in manifest["methods"]["oracle_prototype_residual"]["artifact_paths"]:
        art = {"artifact": load_artifact(Path(path))}
        pred, extra = predict_oracle(test_cache, train_cache, art, std, device)
        oracle_preds.append(pred)
        seed = int(art["artifact"]["config"]["seed"])
        row = metric_row("Oracle prototype residual", "locked_test", test_cache, pred, std, CORE, seed=seed)
        row.update(extra)
        oracle_seed_rows.append(row)
    dynamic_art = {"artifact": load_artifact(Path(manifest["methods"]["dynamic_fixed_three_seed7"]["artifact_path"]))}
    dynamic_pred, dynamic_extra = predict_dynamic(test_cache, dynamic_art, std, device)

    test_rows = [
        metric_row("Ridge residual corrector", "locked_test", test_cache, ridge_pred, std, CORE),
        metric_row("MLP residual corrector", "locked_test", test_cache, aggregate_seed_predictions(mlp_preds), std, CORE),
        metric_row("Oracle prototype residual", "locked_test", test_cache, aggregate_seed_predictions(oracle_preds), std, CORE),
        metric_row("Dynamic fixed-three, seed 7", "locked_test", test_cache, dynamic_pred, std, CORE, seed=7),
    ]
    test_rows[0].update(ridge_extra)
    test_rows[-1].update(dynamic_extra)
    val_lookup = {row["method"]: row for row in manifest["validation_results"]}
    for row in test_rows:
        val = val_lookup[row["method"]]
        row["validation_mae"] = val["mae"]
        row["validation_mse"] = val["mse"]
        row["test_minus_validation_mae"] = row["mae"] - row["validation_mae"]
        row["improvement_vs_single_DLinear_test_mae"] = SINGLE_DLINEAR_TEST_MAE - row["mae"]
        row["improvement_vs_ETTh2_full_adaptive_test_mae"] = FULL_ADAPTIVE_TEST_MAE - row["mae"]
        row["status"] = LABEL
        row["selection_protocol"] = "ETTh1 locked configuration ported to ETTh2; fitted on ETTh2 router_train only; test evaluated once after manifest"
    write_csv(OUT_DIR / "test_results.csv", test_rows)
    write_csv(OUT_DIR / "test_per_seed_results.csv", mlp_seed_rows + oracle_seed_rows)
    write_csv(OUT_DIR / "test_seed_summary.csv", aggregate_seed_rows(mlp_seed_rows + oracle_seed_rows))

    final_payload = {
        **manifest,
        "test_evaluation_complete": True,
        "test_cache_loaded_after_manifest": True,
        "test_cache_hash": sha256_file(TEST_CACHE),
        "test_results": test_rows,
        "test_per_seed": mlp_seed_rows + oracle_seed_rows,
        "runtime_sec_test": time.perf_counter() - started,
    }
    write_json(OUT_DIR / "final_report.json", final_payload)
    write_comparison_report(final_payload)
    update_matched_results(final_payload)
    update_all_results_summary(final_payload)
    update_project_memory(final_payload)
    print(json.dumps({"phase": "test", "test_results": test_rows, "report": str(OUT_DIR / "LOCKED_ETTH1_CONFIG_ETTH2_REPLICATION_REPORT.md")}, indent=2))


def fmt(x: Any) -> str:
    return "" if x is None or x == "" else f"{float(x):.6f}"


def write_comparison_report(payload: Mapping[str, Any]) -> None:
    rows = payload["test_results"]
    val_rows = payload["validation_results"]
    etth1 = {
        "MLP residual corrector": (0.32604682445526123, 0.2673218250274658, 0.3633176386356354),
        "Ridge residual corrector": (0.32644808292388916, 0.2674521803855896, 0.36330097913742065),
        "Oracle prototype residual": (0.3268287479877472, 0.2673642635345459, 0.3660282492637634),
        "Dynamic fixed-three, seed 7": (0.32924923300743103, 0.27206283807754517, 0.36598527431488037),
    }
    lines = [
        "# Locked ETTh1-Config ETTh2 Replication",
        "",
        "Label: `locked_etth1_config_etth2_replication`.",
        "",
        "These ETTh2 artifacts were created after the earlier final ETTh2 test evaluation. They port the locked ETTh1 method configurations, fit on ETTh2 `router_train` only, save a manifest, and then evaluate ETTh2 test once. They are not described as frozen before the already-completed ETTh2 final test evaluation.",
        "",
        "## ETTh2 Results",
        "",
        "| Method | Val MAE | Val MSE | Test MAE | Test MSE | Gain vs DLinear test | Gain vs ETTh2 full adaptive test |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['validation_mae']:.6f} | {row['validation_mse']:.6f} | {row['mae']:.6f} | {row['mse']:.6f} | {row['improvement_vs_single_DLinear_test_mae']:+.6f} | {row['improvement_vs_ETTh2_full_adaptive_test_mae']:+.6f} |"
        )
    lines.extend(["", "## Matched ETTh1 vs ETTh2", "", "| Method | ETTh1 Test MAE | ETTh1 Val MAE | ETTh2 Test MAE | ETTh2 Val MAE |", "|---|---:|---:|---:|---:|"])
    for row in rows:
        e = etth1[row["method"]]
        lines.append(f"| {row['method']} | {e[0]:.6f} | {e[2]:.6f} | {row['mae']:.6f} | {row['validation_mae']:.6f} |")
    lines.extend(
        [
            "",
            "## Checks",
            "",
            "- Router-train fitting only: passed.",
            "- ETTh2 `router_val` was not used for training, feature selection, or checkpoint selection.",
            "- Test cache loaded only after `manifest_before_test.json` was written.",
            "- Causal residual updates enforce `old_start + horizon <= current_start`.",
            "- Long-history summaries use data before each forecast start; pre-test manifest used train+val prefix only.",
            "- Checkpoint/config hashes are recorded in the manifest.",
            "",
            "## Reproduce",
            "",
            "```powershell",
            "python experiments\\locked_etth1_config_etth2_replication\\run_locked_etth1_config_etth2_replication.py --phase all --device cuda",
            "```",
        ]
    )
    (OUT_DIR / "LOCKED_ETTH1_CONFIG_ETTH2_REPLICATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_matched_results(payload: Mapping[str, Any]) -> None:
    path = FROZEN_RESULTS_DIR / "matched_etth1_etth2_results.csv"
    if not path.exists():
        return
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    by_method = {row["method"]: row for row in payload["test_results"]}
    for row in rows:
        if row["method"] in by_method:
            src = by_method[row["method"]]
            row["etth2_test_mae"] = src["mae"]
            row["etth2_test_mse"] = src["mse"]
            row["etth2_validation_mae"] = src["validation_mae"]
            row["etth2_expert_set"] = "+".join(CORE)
            row["etth2_status"] = LABEL
            row["etth2_note"] = "Locked ETTh1 configuration replicated on ETTh2; fitted on router_train only; evaluated after manifest."
    write_csv(path, rows)
    md_path = FROZEN_RESULTS_DIR / "MATCHED_ETTH1_ETTH2_RESULTS.md"
    lines = [
        "# Matched ETTh1/ETTh2 Results",
        "",
        "This table includes ETTh2 locked ETTh1-config replications where valid artifacts now exist. Rows labeled `locked_etth1_config_etth2_replication` were generated after the earlier final ETTh2 test evaluation and are not pre-test preregistrations.",
        "",
        "| Method | ETTh1 Test MAE | ETTh2 Test MAE | ETTh2 Status | ETTh2 Note |",
        "|---|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(f"| {row['method']} | {fmt(row['etth1_test_mae'])} | {fmt(row['etth2_test_mae'])} | {row['etth2_status']} | {row['etth2_note']} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_all_results_summary(payload: Mapping[str, Any]) -> None:
    path = ALL_RESULTS_DIR / "all_costar_results.csv"
    if not path.exists():
        return
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    existing = {(r.get("dataset"), r.get("method"), r.get("source_file")) for r in rows}
    for row in payload["test_results"]:
        out = {
            "dataset": "ETTh2",
            "split": "test",
            "method": row["method"],
            "mae": row["mae"],
            "mse": row["mse"],
            "seeds": 5 if row["method"] in {"MLP residual corrector", "Oracle prototype residual"} else 1,
            "status": LABEL,
            "selection_protocol": row["selection_protocol"],
            "source_file": str(OUT_DIR / "test_results.csv"),
        }
        key = (out["dataset"], out["method"], out["source_file"])
        if key not in existing:
            rows.append(out)
    write_csv(path, rows)


def update_project_memory(payload: Mapping[str, Any]) -> None:
    # Keep this lightweight and append-only; do not rewrite the whole memory by hand.
    log = ROOT / "project_memory" / "experiments" / "2026-08-13_locked_etth1_config_etth2_replication.md"
    lines = [
        "# Locked ETTh1-Config ETTh2 Replication",
        "",
        "Created ETTh2 counterparts for ETTh1-only residual/prototype/dynamic methods using canonical ETTh2 splits and the train-selected core `DLinear+PatchTST+ModernTCN`.",
        "",
        "Important interpretation: these runs happened after the earlier final ETTh2 test evaluation and are labeled `locked_etth1_config_etth2_replication`, not pre-test preregistered final results.",
        "",
        "| Method | Val MAE | Test MAE | Test MSE | Gain vs ETTh2 full adaptive test |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in payload["test_results"]:
        lines.append(f"| {row['method']} | `{row['validation_mae']:.6f}` | `{row['mae']:.6f}` | `{row['mse']:.6f}` | `{row['improvement_vs_ETTh2_full_adaptive_test_mae']:+.6f}` |")
    lines.extend(["", "Artifacts:", "", f"- `{OUT_DIR.relative_to(ROOT) / 'manifest_before_test.json'}`", f"- `{OUT_DIR.relative_to(ROOT) / 'test_results.csv'}`", f"- `{OUT_DIR.relative_to(ROOT) / 'LOCKED_ETTH1_CONFIG_ETTH2_REPLICATION_REPORT.md'}`"])
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    exp = ROOT / "project_memory" / "EXPERIMENTS.md"
    text = exp.read_text(encoding="utf-8")
    marker = "| 2026-08-13 | ETTh2 | Locked ETTh1-config replication |"
    if marker not in text:
        row_bits = []
        best = min(payload["test_results"], key=lambda r: float(r["mae"]))
        row_bits.append(
            f"| 2026-08-13 | ETTh2 | Locked ETTh1-config replication | Port ETTh1-only residual/prototype/dynamic methods to ETTh2 without ETTh2 validation tuning | MLP/prototype 5; dynamic 1 | best `{best['mae']:.6f}` ({best['method']}) | best `{best['mse']:.6f}` | n/a | ETTh2 full adaptive `{FULL_ADAPTIVE_TEST_MAE:.6f}` test | `{FULL_ADAPTIVE_TEST_MAE - best['mae']:+.6f}` vs full adaptive | Additional after-final-test replication; not pre-test preregistered | `experiments/locked_etth1_config_etth2_replication/final_report.json`; `project_memory/experiments/2026-08-13_locked_etth1_config_etth2_replication.md` |"
        )
        text = text.replace("\nUnverified or partial:", "\n" + row_bits[0] + "\n\nUnverified or partial:")
        exp.write_text(text, encoding="utf-8")

    current = ROOT / "project_memory" / "CURRENT_STATE.md"
    cur_text = current.read_text(encoding="utf-8")
    block = """
## Locked ETTh1-Config ETTh2 Replication

ADDITIONAL AFTER-FINAL-TEST REPLICATION:

ETTh2 artifacts now exist for the previously ETTh1-only MLP residual, ridge residual, oracle prototype residual, and dynamic fixed-three seed7 rows. These were fit on ETTh2 router-train only and evaluated after writing a manifest, but they were run after the earlier final ETTh2 test evaluation. Do not treat them as pre-test preregistered final competitors.

Artifacts:

- `experiments/locked_etth1_config_etth2_replication/final_report.json`
- `experiments/locked_etth1_config_etth2_replication/LOCKED_ETTH1_CONFIG_ETTH2_REPLICATION_REPORT.md`
- `experiments/frozen_model_test_results/matched_etth1_etth2_results.csv`

"""
    if "## Locked ETTh1-Config ETTh2 Replication" not in cur_text:
        cur_text = cur_text.replace("## Final Pre-Test Freeze", block + "## Final Pre-Test Freeze")
        current.write_text(cur_text, encoding="utf-8")

    dec = ROOT / "project_memory" / "DECISIONS.md"
    dec_text = dec.read_text(encoding="utf-8")
    if "## Locked ETTh1-Config ETTh2 Replications Are Audit Rows" not in dec_text:
        insert = """
## Locked ETTh1-Config ETTh2 Replications Are Audit Rows

Status: Additional after-final-test replication

Evidence:

- `experiments/locked_etth1_config_etth2_replication/final_report.json`
- `experiments/frozen_model_test_results/matched_etth1_etth2_results.csv`

Decision:

Use the new ETTh2 MLP residual, ridge residual, oracle prototype residual, and dynamic fixed-three seed7 rows only as locked ETTh1-config ETTh2 replication audits. They were fit without ETTh2 validation tuning but were run after the earlier final ETTh2 test evaluation, so they do not supersede the final frozen ETTh2 result.

Reason:

This preserves the distinction between the original confirmatory frozen test evaluation and later matched-table completeness work.

"""
        dec_text = dec_text.replace("## Final Models Are Frozen Before Test", insert + "## Final Models Are Frozen Before Test")
        dec.write_text(dec_text, encoding="utf-8")

    todo = ROOT / "project_memory" / "TODO.md"
    todo_text = todo.read_text(encoding="utf-8")
    if "## Completed - Locked ETTh1-Config ETTh2 Replication" not in todo_text:
        insert = """
## Completed - Locked ETTh1-Config ETTh2 Replication

Closed 2026-08-13.

Result:

- Created ETTh2 counterparts for MLP residual, ridge residual, oracle prototype residual, and dynamic fixed-three seed7 using locked ETTh1 method configs and ETTh2 router-train-only fitting.
- Saved `experiments/locked_etth1_config_etth2_replication/manifest_before_test.json` before loading the ETTh2 test cache.
- Updated `experiments/frozen_model_test_results/matched_etth1_etth2_results.csv`.

Decision:

Treat these as additional after-final-test replication audit rows, not as pre-test preregistered final model results.

Evidence:

- `experiments/locked_etth1_config_etth2_replication/final_report.json`
- `project_memory/experiments/2026-08-13_locked_etth1_config_etth2_replication.md`

"""
        todo_text = todo_text.replace("## Completed - Matched ETTh1/ETTh2 Frozen-Model Results Table", insert + "## Completed - Matched ETTh1/ETTh2 Frozen-Model Results Table")
        todo.write_text(todo_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("train_validate", "test", "all"), default="all")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.phase in {"train_validate", "all"}:
        train_validate(args)
    if args.phase in {"test", "all"}:
        evaluate_test(args)


if __name__ == "__main__":
    main()
