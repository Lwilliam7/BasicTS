"""Sequential COSTAR-TS router variants with explicit history fingerprints."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch
from torch import nn


FingerprintMode = Literal["embedding_only", "fingerprint_only", "embedding_fingerprint"]


@dataclass(frozen=True)
class FingerprintRouterConfig:
    num_experts: int
    max_subset_size: int
    input_len: int = 96
    forecast_horizon: int = 12
    num_features: int = 7
    embedding_dim: int = 64
    hidden_dim: int = 64
    fingerprint_mode: FingerprintMode = "embedding_only"
    fingerprint_lags: tuple[int, ...] = (1, 2, 4, 8)
    recent_fraction: float = 0.25


def _safe_std(values: torch.Tensor, dim: int) -> torch.Tensor:
    if values.shape[dim] <= 1:
        return torch.zeros_like(values.mean(dim=dim))
    return values.std(dim=dim, unbiased=False)


def compute_history_fingerprints(
    history: torch.Tensor,
    *,
    lags: tuple[int, ...] = (1, 2, 4, 8),
    recent_fraction: float = 0.25,
) -> torch.Tensor:
    """Compute deterministic per-variable fingerprints from input history only."""

    batch, length, features = history.shape
    dtype = history.dtype
    device = history.device
    recent_len = max(4, int(round(length * float(recent_fraction))))
    recent = history[:, -recent_len:, :]
    x = torch.arange(length, device=device, dtype=dtype)
    x = (x - x.mean()) / x.std(unbiased=False).clamp_min(1e-6)
    recent_x = torch.arange(recent_len, device=device, dtype=dtype)
    recent_x = (recent_x - recent_x.mean()) / recent_x.std(unbiased=False).clamp_min(1e-6)

    mean = history.mean(dim=1)
    std = _safe_std(history, dim=1).clamp_min(1e-6)
    minimum = history.min(dim=1).values
    maximum = history.max(dim=1).values
    centered = history - mean[:, None, :]
    slope = (centered * x[None, :, None]).mean(dim=1) / x.square().mean().clamp_min(1e-6)
    recent_mean = recent.mean(dim=1)
    recent_std = _safe_std(recent, dim=1)
    recent_slope = ((recent - recent_mean[:, None, :]) * recent_x[None, :, None]).mean(dim=1) / recent_x.square().mean().clamp_min(1e-6)
    diffs = history[:, 1:, :] - history[:, :-1, :]
    diff_mean = diffs.mean(dim=1)
    diff_std = _safe_std(diffs, dim=1)
    acfs = []
    for lag in lags:
        if lag >= length:
            acfs.append(torch.zeros(batch, features, device=device, dtype=dtype))
            continue
        left = history[:, :-lag, :] - history[:, :-lag, :].mean(dim=1, keepdim=True)
        right = history[:, lag:, :] - history[:, lag:, :].mean(dim=1, keepdim=True)
        denom = (left.square().mean(dim=1).sqrt() * right.square().mean(dim=1).sqrt()).clamp_min(1e-6)
        acfs.append((left * right).mean(dim=1) / denom)

    trend_strength = slope.abs() / std
    spectrum = torch.fft.rfft(centered, dim=1)
    power = spectrum.abs().square()
    if power.shape[1] > 1:
        nonzero = power[:, 1:, :]
        dominant = nonzero.argmax(dim=1).to(dtype) / max(nonzero.shape[1], 1)
        split = max(1, nonzero.shape[1] // 3)
        low_energy = nonzero[:, :split, :].sum(dim=1)
        high_energy = nonzero[:, -split:, :].sum(dim=1)
        total_energy = nonzero.sum(dim=1).clamp_min(1e-6)
        low_ratio = low_energy / total_energy
        high_ratio = high_energy / total_energy
    else:
        dominant = torch.zeros(batch, features, device=device, dtype=dtype)
        low_ratio = torch.zeros_like(dominant)
        high_ratio = torch.zeros_like(dominant)

    features_per_variable = [
        mean,
        std,
        minimum,
        maximum,
        slope,
        recent_slope,
        diff_mean,
        diff_std,
        *acfs,
        recent_mean - mean,
        recent_std - std,
        trend_strength,
        dominant,
        low_ratio,
        high_ratio,
    ]
    return torch.cat(features_per_variable, dim=1)


class SequentialCOSTARTSFingerprintRouter(nn.Module):
    """Sequential COSTAR router with controlled fingerprint representation modes."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        config = FingerprintRouterConfig(**kwargs)
        self.config = config
        self.num_experts = int(config.num_experts)
        self.max_subset_size = int(config.max_subset_size)
        self.input_len = int(config.input_len)
        self.forecast_horizon = int(config.forecast_horizon)
        self.num_features = int(config.num_features)
        self.embedding_dim = int(config.embedding_dim)
        self.hidden_dim = int(config.hidden_dim)
        self.fingerprint_mode = config.fingerprint_mode
        self.fingerprint_lags = tuple(config.fingerprint_lags)
        self.recent_fraction = float(config.recent_fraction)

        self.history_encoder = None
        self.history_projection = None
        if self.fingerprint_mode in ("embedding_only", "embedding_fingerprint"):
            self.history_encoder = nn.Sequential(
                nn.Conv1d(config.num_features, config.hidden_dim, kernel_size=5, padding=2),
                nn.GELU(),
                nn.GroupNorm(1, config.hidden_dim),
                nn.Conv1d(config.hidden_dim, config.hidden_dim, kernel_size=5, padding=4, dilation=2),
                nn.GELU(),
                nn.GroupNorm(1, config.hidden_dim),
                nn.AdaptiveAvgPool1d(1),
            )
            self.history_projection = nn.Sequential(
                nn.Linear(config.hidden_dim, config.embedding_dim),
                nn.GELU(),
                nn.LayerNorm(config.embedding_dim),
            )

        fingerprint_dim = (14 + len(config.fingerprint_lags)) * config.num_features
        self.fingerprint_projection = None
        self.history_fingerprint_fusion = None
        if self.fingerprint_mode in ("fingerprint_only", "embedding_fingerprint"):
            self.fingerprint_projection = nn.Sequential(
                nn.Linear(fingerprint_dim, config.embedding_dim),
                nn.GELU(),
                nn.LayerNorm(config.embedding_dim),
            )
        if self.fingerprint_mode == "embedding_fingerprint":
            self.history_fingerprint_fusion = nn.Sequential(
                nn.Linear(config.embedding_dim * 2, config.embedding_dim),
                nn.GELU(),
                nn.LayerNorm(config.embedding_dim),
            )

        self.mask_encoder = nn.Sequential(nn.Linear(config.num_experts, config.embedding_dim), nn.GELU(), nn.LayerNorm(config.embedding_dim))
        self.queried_forecast_encoder = nn.Sequential(
            nn.Linear(config.forecast_horizon * config.num_features, config.embedding_dim),
            nn.GELU(),
            nn.LayerNorm(config.embedding_dim),
        )
        self.current_average_encoder = nn.Sequential(
            nn.Linear(config.forecast_horizon * config.num_features, config.embedding_dim),
            nn.GELU(),
            nn.LayerNorm(config.embedding_dim),
        )
        self.scalar_encoder = nn.Sequential(nn.Linear(3, config.embedding_dim), nn.GELU(), nn.LayerNorm(config.embedding_dim))
        self.expert_embeddings = nn.Embedding(config.num_experts, config.embedding_dim)
        self.fusion = nn.Sequential(
            nn.Linear(config.embedding_dim * 5, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.embedding_dim),
            nn.GELU(),
            nn.LayerNorm(config.embedding_dim),
        )
        self.utility_head = nn.Linear(config.embedding_dim, config.num_experts)

    def config_dict(self) -> dict[str, Any]:
        return asdict(self.config)

    @property
    def fingerprint_dim(self) -> int:
        return (14 + len(self.fingerprint_lags)) * self.num_features

    def encode_history(self, history: torch.Tensor) -> torch.Tensor:
        parts = []
        if self.fingerprint_mode in ("embedding_only", "embedding_fingerprint"):
            assert self.history_encoder is not None and self.history_projection is not None
            history_representation = self.history_encoder(history.transpose(1, 2)).squeeze(-1)
            parts.append(self.history_projection(history_representation))
        if self.fingerprint_mode in ("fingerprint_only", "embedding_fingerprint"):
            assert self.fingerprint_projection is not None
            fingerprint = compute_history_fingerprints(
                history,
                lags=self.fingerprint_lags,
                recent_fraction=self.recent_fraction,
            )
            parts.append(self.fingerprint_projection(fingerprint))
        if self.fingerprint_mode == "embedding_fingerprint":
            assert self.history_fingerprint_fusion is not None
            return self.history_fingerprint_fusion(torch.cat(parts, dim=-1))
        return parts[0]

    def _default_scalar_features(
        self,
        queried_expert_ids: torch.Tensor,
        queried_expert_forecasts: torch.Tensor,
        current_average_forecast: torch.Tensor,
    ) -> torch.Tensor:
        valid_slots = queried_expert_ids >= 0
        counts = valid_slots.sum(dim=1).to(queried_expert_forecasts.dtype)
        pairwise_mean = []
        pairwise_max = []
        for row in range(queried_expert_forecasts.shape[0]):
            valid = valid_slots[row]
            if int(valid.sum()) < 2:
                pairwise_mean.append(torch.zeros((), device=queried_expert_forecasts.device, dtype=queried_expert_forecasts.dtype))
                pairwise_max.append(torch.zeros((), device=queried_expert_forecasts.device, dtype=queried_expert_forecasts.dtype))
                continue
            forecasts = queried_expert_forecasts[row, valid]
            diffs = []
            for left in range(forecasts.shape[0]):
                for right in range(left + 1, forecasts.shape[0]):
                    diffs.append(torch.mean(torch.abs(forecasts[left] - forecasts[right])))
            pairwise = torch.stack(diffs)
            pairwise_mean.append(pairwise.mean())
            pairwise_max.append(pairwise.max())
        return torch.stack((torch.stack(pairwise_mean), torch.stack(pairwise_max), counts / max(self.num_experts, 1)), dim=1)

    def encode(
        self,
        history: torch.Tensor,
        queried_mask: torch.Tensor,
        queried_expert_ids: torch.Tensor,
        queried_expert_forecasts: torch.Tensor,
        current_average_forecast: torch.Tensor | None = None,
        scalar_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = history.shape[0]
        valid_slots = queried_expert_ids >= 0
        denominator = valid_slots.sum(dim=1, keepdim=True).clamp_min(1).to(history.dtype)

        history_representation = self.encode_history(history)
        mask_representation = self.mask_encoder(queried_mask.to(history.dtype))
        forecast_flat = queried_expert_forecasts.reshape(batch_size, self.max_subset_size, self.forecast_horizon * self.num_features)
        safe_ids = queried_expert_ids.clamp_min(0)
        forecast_representation = self.queried_forecast_encoder(forecast_flat)
        forecast_representation = forecast_representation + self.expert_embeddings(safe_ids)
        forecast_representation = forecast_representation * valid_slots.unsqueeze(-1).to(history.dtype)
        queried_representation = forecast_representation.sum(dim=1) / denominator
        if current_average_forecast is None:
            masked_forecasts = queried_expert_forecasts * valid_slots[:, :, None, None].to(history.dtype)
            current_average_forecast = masked_forecasts.sum(dim=1) / denominator[:, :, None]
        average_representation = self.current_average_encoder(current_average_forecast.reshape(batch_size, self.forecast_horizon * self.num_features))
        if scalar_features is None:
            scalar_features = self._default_scalar_features(queried_expert_ids, queried_expert_forecasts, current_average_forecast)
        scalar_representation = self.scalar_encoder(scalar_features.to(history.dtype))
        return self.fusion(torch.cat((history_representation, mask_representation, queried_representation, average_representation, scalar_representation), dim=-1))

    def forward(
        self,
        history: torch.Tensor,
        queried_mask: torch.Tensor,
        queried_expert_ids: torch.Tensor,
        queried_expert_forecasts: torch.Tensor,
        current_average_forecast: torch.Tensor | None = None,
        scalar_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        representation = self.encode(
            history,
            queried_mask,
            queried_expert_ids,
            queried_expert_forecasts,
            current_average_forecast=current_average_forecast,
            scalar_features=scalar_features,
        )
        return {"representation": representation, "utility_prediction": self.utility_head(representation)}
