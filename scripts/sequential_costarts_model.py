"""Model code for the validation-best sequential COSTAR-TS router checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn


class SequentialCOSTARTSRouter(nn.Module):
    """Forecast-state sequential COSTAR-TS router.

    This class matches the per-seed checkpoints in
    checkpoints/costarts_sequential/seed_*/best_sequential_costarts_router.pt.
    It is the model architecture used by the best validation result family, not
    the old history-only COSTARTSRouter.
    """
#constructor
    def __init__(
        self,
        num_experts: int,
        max_subset_size: int,
        input_len: int = 96,
        forecast_horizon: int = 12,
        num_features: int = 7,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        **_: Any,
    ) -> None:
        super().__init__()
        self.num_experts = int(num_experts)
        self.max_subset_size = int(max_subset_size)
        self.input_len = int(input_len)
        self.forecast_horizon = int(forecast_horizon)
        self.num_features = int(num_features)
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)

        self.history_encoder = nn.Sequential(
            nn.Conv1d(num_features, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.GroupNorm(1, hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=4, dilation=2),
            nn.GELU(),
            nn.GroupNorm(1, hidden_dim),
            nn.AdaptiveAvgPool1d(1),
        )
        self.history_projection = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
             nn.GELU(),
            nn.LayerNorm(embedding_dim),
        )
        self.mask_encoder = nn.Sequential(
            nn.Linear(num_experts, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
        )
        self.queried_forecast_encoder = nn.Sequential(
            nn.Linear(forecast_horizon * num_features, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
        )
        self.current_average_encoder = nn.Sequential(
            nn.Linear(forecast_horizon * num_features, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
        )
        self.scalar_encoder = nn.Sequential(
            nn.Linear(3, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
        )
        self.expert_embeddings = nn.Embedding(num_experts, embedding_dim)
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 5, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
        )
        self.utility_head = nn.Linear(embedding_dim, num_experts)

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
                pairwise_mean.append(
                    torch.zeros((), device=queried_expert_forecasts.device, dtype=queried_expert_forecasts.dtype)
                )
                pairwise_max.append(
                    torch.zeros((), device=queried_expert_forecasts.device, dtype=queried_expert_forecasts.dtype)
                )
                continue
            forecasts = queried_expert_forecasts[row, valid]
            diffs = []
            for left in range(forecasts.shape[0]):
                for right in range(left + 1, forecasts.shape[0]):
                    diffs.append(torch.mean(torch.abs(forecasts[left] - forecasts[right])))
            pairwise = torch.stack(diffs)
            pairwise_mean.append(pairwise.mean())
            pairwise_max.append(pairwise.max())
        return torch.stack(
            (
                torch.stack(pairwise_mean),
                torch.stack(pairwise_max),
                counts / max(self.num_experts, 1),
            ),
            dim=1,
        )

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

        history_representation = self.history_encoder(history.transpose(1, 2)).squeeze(-1)
        history_representation = self.history_projection(history_representation)
        mask_representation = self.mask_encoder(queried_mask.to(history.dtype))

        forecast_flat = queried_expert_forecasts.reshape(
            batch_size,
            self.max_subset_size,
            self.forecast_horizon * self.num_features,
        )
        safe_ids = queried_expert_ids.clamp_min(0)
        forecast_representation = self.queried_forecast_encoder(forecast_flat)
        forecast_representation = forecast_representation + self.expert_embeddings(safe_ids)
        forecast_representation = forecast_representation * valid_slots.unsqueeze(-1).to(history.dtype)
        queried_representation = forecast_representation.sum(dim=1) / denominator

        if current_average_forecast is None:
            masked_forecasts = queried_expert_forecasts * valid_slots[:, :, None, None].to(history.dtype)
            current_average_forecast = masked_forecasts.sum(dim=1) / denominator[:, :, None]
        average_representation = self.current_average_encoder(
            current_average_forecast.reshape(batch_size, self.forecast_horizon * self.num_features)
        )

        if scalar_features is None:
            scalar_features = self._default_scalar_features(
                queried_expert_ids,
                queried_expert_forecasts,
                current_average_forecast,
            )
        scalar_representation = self.scalar_encoder(scalar_features.to(history.dtype))

        return self.fusion(
            torch.cat(
                (
                    history_representation,
                    mask_representation,
                    queried_representation,
                    average_representation,
                    scalar_representation,
                ),
                dim=-1,
            )
        )

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
        return {
            "representation": representation,
            "utility_prediction": self.utility_head(representation),
        }

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path) -> "SequentialCOSTARTSRouter":
#loads a checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
#creates a model
        model = cls(**checkpoint["router_config"])
#load weights
        model.load_state_dict(checkpoint["router_state_dict"], strict=True)
        return model

#wsave space
def load_sequential_costarts_router(checkpoint_path: str | Path) -> SequentialCOSTARTSRouter:
    return SequentialCOSTARTSRouter.from_checkpoint(checkpoint_path)
