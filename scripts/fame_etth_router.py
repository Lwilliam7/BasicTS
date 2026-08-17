"""FAME-style ETTh router baseline utilities.

This is an ETTh adaptation of FAME's forecastability-aware sparse routing idea:
history-only fingerprints, soft expert-suitability targets from frozen expert
losses, and sparse Top-r forecast fusion. It is not an exact reproduction of
the retail/industrial FAME implementation.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


EPSILON = 1e-8


def assert_no_test_path(path: str | Path) -> None:
    normalized_parts = [part.lower() for part in Path(path).parts]
    if any("test" in part for part in normalized_parts):
        raise ValueError(f"Refusing to load test-related path: {path}")


def load_router_cache(path: str | Path, expected_split_role: str) -> dict[str, Any]:
    assert_no_test_path(path)
    cache = torch.load(path, map_location="cpu", weights_only=False)
    if cache.get("split_role") != expected_split_role:
        raise ValueError(f"Expected split_role={expected_split_role}, got {cache.get('split_role')}")
    return cache


def validate_cache_pair(train_cache: Mapping[str, Any], val_cache: Mapping[str, Any]) -> None:
    if train_cache["split_role"] != "router_train":
        raise ValueError("Training cache must have split_role='router_train'")
    if val_cache["split_role"] != "router_val":
        raise ValueError("Validation cache must have split_role='router_val'")
    if tuple(train_cache["expert_names"]) != tuple(val_cache["expert_names"]):
        raise ValueError("Expert ordering mismatch between train and validation caches")
    train_hist_shape = tuple(train_cache["histories"].shape[1:])
    val_hist_shape = tuple(val_cache["histories"].shape[1:])
    if train_hist_shape != val_hist_shape:
        raise ValueError(f"History dimensions mismatch: {train_hist_shape} != {val_hist_shape}")
    train_pred_shape = tuple(train_cache["prediction_stack"].shape[1:])
    val_pred_shape = tuple(val_cache["prediction_stack"].shape[1:])
    if train_pred_shape != val_pred_shape:
        raise ValueError(f"Forecast dimensions mismatch: {train_pred_shape} != {val_pred_shape}")


@dataclass
class FingerprintScaler:
    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def fit(cls, features: torch.Tensor) -> "FingerprintScaler":
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True, unbiased=False).clamp_min(EPSILON)
        return cls(mean=mean, std=std)

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        return (features - self.mean) / self.std


def _quantile(values: torch.Tensor, q: float) -> torch.Tensor:
    return torch.quantile(values, q, dim=1)


def _rolling_variance_mean(history: torch.Tensor, window: int = 12) -> torch.Tensor:
    if history.shape[1] < window:
        return history.var(dim=1, unbiased=False)
    windows = history.unfold(dimension=1, size=window, step=1)
    return windows.var(dim=-1, unbiased=False).mean(dim=1)


def _linear_trend_features(history: torch.Tensor) -> tuple[torch.Tensor, ...]:
    batch_size, input_len, _ = history.shape
    time = torch.arange(input_len, device=history.device, dtype=history.dtype)
    time = time - time.mean()
    time = time.view(1, input_len, 1)
    centered = history - history.mean(dim=1, keepdim=True)
    denom = (time.square().sum(dim=1) + EPSILON)
    slope = (centered * time).sum(dim=1) / denom
    intercept = history.mean(dim=1)
    fit = intercept[:, None, :] + slope[:, None, :] * time
    total_var = history.var(dim=1, unbiased=False).clamp_min(EPSILON)
    trend_strength = fit.var(dim=1, unbiased=False) / total_var
    normalized_slope = slope / (history.std(dim=1, unbiased=False).clamp_min(EPSILON))
    drift = history[:, -1, :] - history[:, 0, :]
    edge = min(12, input_len)
    moving_average_change = history[:, -edge:, :].mean(dim=1) - history[:, :edge, :].mean(dim=1)
    return slope, normalized_slope, trend_strength, drift, moving_average_change


def _autocorrelation(history: torch.Tensor, lag: int) -> torch.Tensor:
    if history.shape[1] <= lag:
        return torch.zeros(history.shape[0], history.shape[2], device=history.device, dtype=history.dtype)
    left = history[:, :-lag, :]
    right = history[:, lag:, :]
    left = left - left.mean(dim=1, keepdim=True)
    right = right - right.mean(dim=1, keepdim=True)
    denom = left.square().sum(dim=1).sqrt() * right.square().sum(dim=1).sqrt()
    return (left * right).sum(dim=1) / denom.clamp_min(EPSILON)


def _seasonal_strength(history: torch.Tensor, period: int = 24) -> torch.Tensor:
    batch_size, input_len, num_features = history.shape
    if input_len < period * 2:
        return torch.zeros(batch_size, num_features, device=history.device, dtype=history.dtype)
    phase_means = []
    for phase in range(period):
        phase_values = history[:, phase::period, :]
        phase_means.append(phase_values.mean(dim=1))
    seasonal = torch.stack(phase_means, dim=1)
    seasonal_var = seasonal.var(dim=1, unbiased=False)
    total_var = history.var(dim=1, unbiased=False).clamp_min(EPSILON)
    return seasonal_var / total_var


def _spectral_features(history: torch.Tensor) -> tuple[torch.Tensor, ...]:
    centered = history - history.mean(dim=1, keepdim=True)
    spectrum = torch.fft.rfft(centered, dim=1)
    power = spectrum.abs().square()
    if power.shape[1] > 1:
        power = power[:, 1:, :]
    total = power.sum(dim=1).clamp_min(EPSILON)
    probabilities = power / total[:, None, :]
    entropy = -(probabilities * torch.log(probabilities.clamp_min(EPSILON))).sum(dim=1)
    entropy = entropy / math.log(max(power.shape[1], 2))
    dominant_frequency = power.argmax(dim=1).to(history.dtype) / max(power.shape[1] - 1, 1)

    bins = power.shape[1]
    low_end = max(1, bins // 3)
    mid_end = max(low_end + 1, (2 * bins) // 3)
    low_energy = power[:, :low_end, :].sum(dim=1) / total
    mid_energy = power[:, low_end:mid_end, :].sum(dim=1) / total
    high_energy = power[:, mid_end:, :].sum(dim=1) / total
    return entropy, dominant_frequency, low_energy, mid_energy, high_energy


def _cross_channel_features(history: torch.Tensor) -> torch.Tensor:
    channel_means = history.mean(dim=1)
    channel_stds = history.std(dim=1, unbiased=False)
    dispersion = history.std(dim=2, unbiased=False)
    centered = history - history.mean(dim=1, keepdim=True)
    covariance = torch.einsum("btc,btd->bcd", centered, centered) / max(history.shape[1] - 1, 1)
    std = centered.square().sum(dim=1).div(max(history.shape[1] - 1, 1)).sqrt().clamp_min(EPSILON)
    corr = covariance / (std[:, :, None] * std[:, None, :])
    num_channels = history.shape[2]
    offdiag_mask = ~torch.eye(num_channels, device=history.device, dtype=torch.bool)
    offdiag_corr = corr[:, offdiag_mask]
    return torch.stack(
        (
            channel_means.mean(dim=1),
            channel_stds.mean(dim=1),
            channel_stds.std(dim=1, unbiased=False),
            dispersion.mean(dim=1),
            dispersion.std(dim=1, unbiased=False),
            offdiag_corr.mean(dim=1),
            offdiag_corr.abs().mean(dim=1),
        ),
        dim=1,
    )


def extract_fame_etth_fingerprint(histories: torch.Tensor) -> torch.Tensor:
    """Extract history-only ETTh forecastability fingerprints.

    Args:
        histories: Tensor with shape [num_windows, input_len, num_features].

    Returns:
        Tensor with shape [num_windows, fingerprint_dim].
    """

    histories = histories.to(torch.float32)
    q25 = _quantile(histories, 0.25)
    q75 = _quantile(histories, 0.75)
    diffs = histories[:, 1:, :] - histories[:, :-1, :]
    mean = histories.mean(dim=1)
    std = histories.std(dim=1, unbiased=False)
    outlier_ratio = (torch.abs(histories - mean[:, None, :]) > 3.0 * std[:, None, :].clamp_min(EPSILON)).float().mean(dim=1)
    trend_parts = _linear_trend_features(histories)
    spectral_parts = _spectral_features(histories)

    per_channel_features = [
        mean,
        std,
        histories.median(dim=1).values,
        q75 - q25,
        histories.amax(dim=1) - histories.amin(dim=1),
        diffs.std(dim=1, unbiased=False),
        diffs.abs().mean(dim=1),
        _rolling_variance_mean(histories),
        outlier_ratio,
        *trend_parts,
        _autocorrelation(histories, 1),
        _autocorrelation(histories, 12),
        _autocorrelation(histories, 24),
        _seasonal_strength(histories, 24),
        *spectral_parts,
    ]
    flattened = [feature.reshape(feature.shape[0], -1) for feature in per_channel_features]
    features = torch.cat([*flattened, _cross_channel_features(histories)], dim=1)
    return torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def soft_expert_targets(error_matrix: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    if tau <= 0:
        raise ValueError("tau must be positive")
    return torch.softmax(-error_matrix.to(torch.float32) / tau, dim=1)


class FameETThRouter(nn.Module):
    def __init__(self, input_dim: int, num_experts: int, hidden_size: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_experts),
        )

    def forward(self, fingerprints: torch.Tensor) -> torch.Tensor:
        return self.net(fingerprints)

    def probabilities(self, fingerprints: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(fingerprints), dim=1)


@dataclass
class FameTrainingConfig:
    seed: int = 7
    tau: float = 0.1
    target_metric: str = "mae"
    hidden_size: int = 128
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 256
    epochs: int = 200
    prediction_loss_weight: float = 0.0
    load_balance_weight: float = 0.0
    expert_cost_weight: float = 0.0


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def _masked_metric(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[float, float]:
    mask = mask.to(torch.float32)
    denom = mask.sum().clamp_min(EPSILON)
    diff = prediction - target
    mae = (diff.abs() * mask).sum() / denom
    mse = (diff.square() * mask).sum() / denom
    return float(mae), float(mse)


def dense_weighted_prediction(prediction_stack: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.einsum("nhfe,ne->nhf", prediction_stack.to(torch.float32), weights.to(torch.float32))


def train_fame_router(
    train_cache: Mapping[str, Any],
    fingerprints: torch.Tensor,
    config: FameTrainingConfig,
) -> tuple[FameETThRouter, dict[str, float]]:
    set_deterministic_seed(config.seed)
    errors = train_cache["error_matrix"] if config.target_metric == "mae" else train_cache["mse_matrix"]
    targets = soft_expert_targets(errors, tau=config.tau)
    model = FameETThRouter(
        input_dim=fingerprints.shape[1],
        num_experts=targets.shape[1],
        hidden_size=config.hidden_size,
        dropout=config.dropout,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    num_samples = fingerprints.shape[0]

    prediction_stack = train_cache["prediction_stack"].to(torch.float32)
    true_targets = train_cache["targets"].to(torch.float32)
    target_masks = train_cache["target_masks"].to(torch.float32)

    for _ in range(config.epochs):
        permutation = torch.randperm(num_samples)
        for start in range(0, num_samples, config.batch_size):
            batch_index = permutation[start : start + config.batch_size]
            batch_features = fingerprints[batch_index]
            batch_targets = targets[batch_index]
            logits = model(batch_features)
            log_probs = F.log_softmax(logits, dim=1)
            loss = F.kl_div(log_probs, batch_targets, reduction="batchmean")

            probs = torch.softmax(logits, dim=1)
            if config.prediction_loss_weight:
                pred = dense_weighted_prediction(prediction_stack[batch_index], probs)
                mae = (torch.abs(pred - true_targets[batch_index]) * target_masks[batch_index]).sum()
                mae = mae / target_masks[batch_index].sum().clamp_min(EPSILON)
                loss = loss + config.prediction_loss_weight * mae
            if config.load_balance_weight:
                usage = probs.mean(dim=0)
                target_usage = torch.full_like(usage, 1.0 / usage.numel())
                loss = loss + config.load_balance_weight * F.mse_loss(usage, target_usage)
            if config.expert_cost_weight:
                equal_cost = torch.ones(probs.shape[1], dtype=probs.dtype, device=probs.device)
                loss = loss + config.expert_cost_weight * (probs * equal_cost).sum(dim=1).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        logits = model(fingerprints)
        final_loss = F.kl_div(F.log_softmax(logits, dim=1), targets, reduction="batchmean")
    return model, {"train_kl": float(final_loss)}


def sparse_top_r_weights(probabilities: torch.Tensor, top_r: int, delta: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    if top_r < 1:
        raise ValueError("top_r must be >= 1")
    top_r = min(top_r, probabilities.shape[1])
    top_values, top_indices = torch.topk(probabilities, k=top_r, dim=1)
    active = torch.zeros_like(probabilities, dtype=torch.bool)
    active.scatter_(1, top_indices, True)
    if delta is not None:
        threshold = (top_values[:, :1] - float(delta)).clamp_min(0.0)
        keep_top = top_values >= threshold
        keep_top[:, 0] = True
        active = torch.zeros_like(probabilities, dtype=torch.bool)
        active.scatter_(1, top_indices, keep_top)
    weights = torch.where(active, probabilities, torch.zeros_like(probabilities))
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(EPSILON)
    return active, weights


def evaluate_sparse_router(
    probabilities: torch.Tensor,
    cache: Mapping[str, Any],
    top_r: int,
    delta: float | None = None,
) -> dict[str, Any]:
    active, weights = sparse_top_r_weights(probabilities, top_r=top_r, delta=delta)
    prediction = dense_weighted_prediction(cache["prediction_stack"], weights)
    mae, mse = _masked_metric(prediction, cache["targets"], cache["target_masks"])
    best_expert = cache["best_expert"].to(torch.long)
    top1 = probabilities.argmax(dim=1)
    entropy = -(probabilities * probabilities.clamp_min(EPSILON).log()).sum(dim=1)
    usage = active.to(torch.float32).mean(dim=0)
    return {
        "mae": mae,
        "mse": mse,
        "average_experts_used": float(active.sum(dim=1).to(torch.float32).mean()),
        "top1_accuracy": float((top1 == best_expert).to(torch.float32).mean() * 100.0),
        "top_r_oracle_coverage": float(active.gather(1, best_expert[:, None]).to(torch.float32).mean() * 100.0),
        "routing_entropy": float(entropy.mean()),
        "expert_usage": {name: float(usage[i] * 100.0) for i, name in enumerate(cache["expert_names"])},
        "weights_sum_mean": float(weights.sum(dim=1).mean()),
    }


def evaluate_weighted_average(cache: Mapping[str, Any]) -> dict[str, float]:
    num_experts = len(cache["expert_names"])
    weights = torch.full((cache["prediction_stack"].shape[0], num_experts), 1.0 / num_experts)
    pred = dense_weighted_prediction(cache["prediction_stack"], weights)
    mae, mse = _masked_metric(pred, cache["targets"], cache["target_masks"])
    return {"mae": mae, "mse": mse, "average_experts_used": float(num_experts)}


def evaluate_best_fixed_expert(cache: Mapping[str, Any]) -> dict[str, Any]:
    errors = cache["error_matrix"].to(torch.float32).mean(dim=0)
    best_index = int(errors.argmin())
    pred = cache["prediction_stack"][..., best_index]
    mae, mse = _masked_metric(pred, cache["targets"], cache["target_masks"])
    return {
        "mae": mae,
        "mse": mse,
        "average_experts_used": 1.0,
        "expert": cache["expert_names"][best_index],
    }


def evaluate_oracle_best_single(cache: Mapping[str, Any]) -> dict[str, Any]:
    best_index = cache["error_matrix"].to(torch.float32).argmin(dim=1)
    prediction = cache["prediction_stack"][torch.arange(best_index.numel()), :, :, best_index]
    mae, mse = _masked_metric(prediction, cache["targets"], cache["target_masks"])
    return {"mae": mae, "mse": mse, "average_experts_used": 1.0}


def parameter_count(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())

