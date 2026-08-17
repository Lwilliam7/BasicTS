"""Q/K and Q/K/V Sequential COSTAR-TS router ablations."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch
from torch import nn


AttentionRouterMode = Literal["qk", "qkv"]


@dataclass(frozen=True)
class AttentionRouterConfig:
    num_experts: int
    max_subset_size: int
    input_len: int = 96
    forecast_horizon: int = 12
    num_features: int = 7
    embedding_dim: int = 64
    hidden_dim: int = 64
    attention_dim: int | None = None
    attention_mode: AttentionRouterMode = "qk"


class SequentialCOSTARSAttentionRouter(nn.Module):
    """Sequential COSTAR router with explicit learned expert Q/K(/V) scoring."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        config = AttentionRouterConfig(**kwargs)
        self.config = config
        self.num_experts = int(config.num_experts)
        self.max_subset_size = int(config.max_subset_size)
        self.input_len = int(config.input_len)
        self.forecast_horizon = int(config.forecast_horizon)
        self.num_features = int(config.num_features)
        self.embedding_dim = int(config.embedding_dim)
        self.hidden_dim = int(config.hidden_dim)
        self.attention_dim = int(config.attention_dim or config.embedding_dim)
        self.attention_mode = config.attention_mode
        if self.attention_mode not in ("qk", "qkv"):
            raise ValueError(f"Unknown attention_mode={self.attention_mode!r}")

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

        self.query_projection = nn.Linear(config.embedding_dim, self.attention_dim, bias=False)
        self.key_projection = nn.Linear(config.embedding_dim, self.attention_dim, bias=False)
        self.value_projection = None
        self.context_projection = None
        self.post_context_query_projection = None
        if self.attention_mode == "qkv":
            self.value_projection = nn.Linear(config.embedding_dim, self.attention_dim, bias=False)
            self.context_projection = nn.Sequential(
                nn.Linear(self.attention_dim, config.embedding_dim),
                nn.GELU(),
                nn.LayerNorm(config.embedding_dim),
            )
            self.post_context_query_projection = nn.Linear(config.embedding_dim, self.attention_dim, bias=False)

    def config_dict(self) -> dict[str, Any]:
        return asdict(self.config)

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

    def encode_state(
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

    def _qk_scores(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = self.query_projection(state)
        keys = self.key_projection(self.expert_embeddings.weight)
        scores = query @ keys.T / math.sqrt(float(self.attention_dim))
        return query, keys, scores

    def attention_scores(self, state: torch.Tensor, queried_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        query, keys, scores = self._qk_scores(state)
        masked_scores = scores.masked_fill(queried_mask.to(torch.bool), -1e9)
        probabilities = torch.softmax(masked_scores, dim=-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
        if self.attention_mode == "qk":
            return {
                "query": query,
                "keys": keys,
                "attention_scores": scores,
                "masked_attention_scores": masked_scores,
                "attention_probabilities": probabilities,
                "attention_entropy": entropy,
                "fused_state": state,
                "utility_prediction": scores,
            }

        assert self.value_projection is not None
        assert self.context_projection is not None
        assert self.post_context_query_projection is not None
        values = self.value_projection(self.expert_embeddings.weight)
        context = probabilities @ values
        fused_state = torch.layer_norm(state + self.context_projection(context), (state.shape[-1],))
        fused_query = self.post_context_query_projection(fused_state)
        final_scores = fused_query @ keys.T / math.sqrt(float(self.attention_dim))
        return {
            "query": query,
            "keys": keys,
            "values": values,
            "context": context,
            "attention_scores": scores,
            "masked_attention_scores": masked_scores,
            "attention_probabilities": probabilities,
            "attention_entropy": entropy,
            "fused_state": fused_state,
            "utility_prediction": final_scores,
        }

    def forward(
        self,
        history: torch.Tensor,
        queried_mask: torch.Tensor,
        queried_expert_ids: torch.Tensor,
        queried_expert_forecasts: torch.Tensor,
        current_average_forecast: torch.Tensor | None = None,
        scalar_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        state = self.encode_state(
            history,
            queried_mask,
            queried_expert_ids,
            queried_expert_forecasts,
            current_average_forecast=current_average_forecast,
            scalar_features=scalar_features,
        )
        outputs = self.attention_scores(state, queried_mask)
        outputs["representation"] = outputs["fused_state"]
        return outputs
