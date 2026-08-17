"""Full-sequence Transformer router variants for Sequential COSTAR-TS."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class SequentialCOSTARTSTransformerRouter(nn.Module):
    """Sequential COSTAR-TS router that preserves temporal tokens.

    The class intentionally keeps the same forward signature as
    ``SequentialCOSTARTSRouterFull`` so existing training/evaluation code can
    isolate only the encoder change.
    """

    def __init__(
        self,
        num_experts: int,
        max_subset_size: int,
        input_len: int = 96,
        forecast_horizon: int = 12,
        num_features: int = 7,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        feedforward_dim: int = 128,
        dropout: float = 0.1,
        state_mode: str = "history_only",
        pooling: str = "cls",
        **_: Any,
    ) -> None:
        super().__init__()
        if state_mode not in {"history_only", "history_ensemble", "full"}:
            raise ValueError(f"Unknown state_mode: {state_mode}")
        if pooling not in {"cls", "mean", "attention"}:
            raise ValueError(f"Unknown pooling: {pooling}")
        self.num_experts = int(num_experts)
        self.max_subset_size = int(max_subset_size)
        self.input_len = int(input_len)
        self.forecast_horizon = int(forecast_horizon)
        self.num_features = int(num_features)
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.state_mode = state_mode
        self.pooling = pooling

        max_tokens = 1 + self.input_len + self.forecast_horizon + self.max_subset_size * self.forecast_horizon
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, max_tokens, embedding_dim))
        self.type_embedding = nn.Embedding(3, embedding_dim)
        self.expert_embedding = nn.Embedding(self.num_experts, embedding_dim)
        self.history_projection = nn.Linear(num_features, embedding_dim)
        self.forecast_projection = nn.Linear(num_features, embedding_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.attention_pool = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.mask_encoder = nn.Sequential(
            nn.Linear(num_experts, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
        )
        self.scalar_encoder = nn.Sequential(
            nn.Linear(3, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
        )
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
        )
        self.utility_head = nn.Linear(embedding_dim, num_experts)
        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

    def _scalar_features(
        self,
        queried_expert_ids: torch.Tensor,
        queried_expert_forecasts: torch.Tensor,
    ) -> torch.Tensor:
        valid = queried_expert_ids >= 0
        counts = valid.sum(dim=1).to(queried_expert_forecasts.dtype)
        pair_values = []
        pair_masks = []
        for left in range(queried_expert_forecasts.shape[1]):
            for right in range(left + 1, queried_expert_forecasts.shape[1]):
                pair_values.append(
                    torch.mean(
                        torch.abs(queried_expert_forecasts[:, left] - queried_expert_forecasts[:, right]),
                        dim=(1, 2),
                    )
                )
                pair_masks.append(valid[:, left] & valid[:, right])
        if pair_values:
            values = torch.stack(pair_values, dim=1)
            masks = torch.stack(pair_masks, dim=1)
            pairwise_mean = (values * masks.to(values.dtype)).sum(dim=1) / masks.sum(dim=1).clamp_min(1).to(values.dtype)
            pairwise_max = values.masked_fill(~masks, 0.0).max(dim=1).values
        else:
            pairwise_mean = torch.zeros_like(counts)
            pairwise_max = torch.zeros_like(counts)
        return torch.stack((pairwise_mean, pairwise_max, counts / max(self.num_experts, 1)), dim=1)

    def _append_tokens(
        self,
        tokens: list[torch.Tensor],
        masks: list[torch.Tensor],
        values: torch.Tensor,
        type_id: int,
        valid_mask: torch.Tensor | None = None,
        expert_ids: torch.Tensor | None = None,
    ) -> None:
        projected = self.forecast_projection(values)
        projected = projected + self.type_embedding.weight[type_id].view(1, 1, -1)
        if expert_ids is not None:
            projected = projected + self.expert_embedding(expert_ids).unsqueeze(2)
            projected = projected.reshape(values.shape[0], -1, self.embedding_dim)
        if valid_mask is None:
            mask = torch.zeros(projected.shape[:2], dtype=torch.bool, device=projected.device)
        else:
            mask = ~valid_mask
            if mask.ndim == 2:
                mask = mask.unsqueeze(-1).expand(-1, -1, self.forecast_horizon).reshape(values.shape[0], -1)
        tokens.append(projected)
        masks.append(mask)

    def encode(
        self,
        history: torch.Tensor,
        queried_mask: torch.Tensor,
        queried_expert_ids: torch.Tensor,
        queried_expert_forecasts: torch.Tensor,
        current_average_forecast: torch.Tensor | None = None,
        scalar_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = history.shape[0]
        tokens = [self.cls_token.expand(batch, -1, -1)]
        masks = [torch.zeros((batch, 1), dtype=torch.bool, device=history.device)]

        history_tokens = self.history_projection(history) + self.type_embedding.weight[0].view(1, 1, -1)
        tokens.append(history_tokens)
        masks.append(torch.zeros((batch, self.input_len), dtype=torch.bool, device=history.device))

        if self.state_mode in {"history_ensemble", "full"}:
            if current_average_forecast is None:
                current_average_forecast = torch.zeros(
                    (batch, self.forecast_horizon, self.num_features),
                    dtype=history.dtype,
                    device=history.device,
                )
            self._append_tokens(tokens, masks, current_average_forecast, type_id=1)

        if self.state_mode == "full":
            valid_slots = queried_expert_ids >= 0
            safe_ids = queried_expert_ids.clamp_min(0)
            self._append_tokens(
                tokens,
                masks,
                queried_expert_forecasts,
                type_id=2,
                valid_mask=valid_slots,
                expert_ids=safe_ids,
            )

        sequence = torch.cat(tokens, dim=1)
        padding_mask = torch.cat(masks, dim=1)
        sequence = sequence + self.position_embedding[:, : sequence.shape[1], :]
        encoded = self.transformer(sequence, src_key_padding_mask=padding_mask)

        if self.pooling == "cls":
            sequence_representation = encoded[:, 0]
        elif self.pooling == "mean":
            valid = (~padding_mask).to(encoded.dtype)
            sequence_representation = (encoded * valid[:, :, None]).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)[:, None]
        else:
            scores = self.attention_pool(encoded).squeeze(-1).masked_fill(padding_mask, -1e9)
            weights = torch.softmax(scores, dim=1)
            sequence_representation = (encoded * weights[:, :, None]).sum(dim=1)

        if scalar_features is None:
            scalar_features = self._scalar_features(queried_expert_ids, queried_expert_forecasts)
        mask_representation = self.mask_encoder(queried_mask.to(history.dtype))
        scalar_representation = self.scalar_encoder(scalar_features.to(history.dtype))
        return self.fusion(torch.cat((sequence_representation, mask_representation, scalar_representation), dim=1))

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
