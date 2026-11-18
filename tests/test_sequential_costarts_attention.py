from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.sequential_costarts_attention_model import SequentialCOSTARSAttentionRouter


def _dummy_state() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    history = torch.randn(2, 96, 7)
    queried_mask = torch.zeros(2, 5)
    queried_mask[:, 1] = 1.0
    queried_expert_ids = torch.full((2, 5), -1, dtype=torch.long)
    queried_expert_ids[:, 0] = 1
    queried_forecasts = torch.randn(2, 5, 12, 7)
    current_average = queried_forecasts[:, 0]
    return history, queried_mask, queried_expert_ids, queried_forecasts, current_average


def test_qk_router_shapes_and_masking() -> None:
    model = SequentialCOSTARSAttentionRouter(
        num_experts=5,
        max_subset_size=5,
        input_len=96,
        forecast_horizon=12,
        num_features=7,
        embedding_dim=16,
        hidden_dim=16,
        attention_dim=8,
        attention_mode="qk",
    )
    outputs = model(*_dummy_state())
    assert outputs["representation"].shape == (2, 16)
    assert outputs["query"].shape == (2, 8)
    assert outputs["keys"].shape == (5, 8)
    assert outputs["attention_scores"].shape == (2, 5)
    assert outputs["attention_probabilities"].shape == (2, 5)
    assert outputs["utility_prediction"].shape == (2, 5)
    assert torch.all(outputs["masked_attention_scores"][:, 1] < -1e8)
    assert torch.allclose(outputs["attention_probabilities"][:, 1], torch.zeros(2), atol=1e-7)


def test_qkv_router_shapes_and_masked_expert_cannot_be_selected_after_eval_mask() -> None:
    model = SequentialCOSTARSAttentionRouter(
        num_experts=5,
        max_subset_size=5,
        input_len=96,
        forecast_horizon=12,
        num_features=7,
        embedding_dim=16,
        hidden_dim=16,
        attention_dim=8,
        attention_mode="qkv",
    )
    history, queried_mask, queried_expert_ids, queried_forecasts, current_average = _dummy_state()
    outputs = model(history, queried_mask, queried_expert_ids, queried_forecasts, current_average)
    masked_utilities = outputs["utility_prediction"].masked_fill(queried_mask.to(torch.bool), -1e9)
    assert outputs["values"].shape == (5, 8)
    assert outputs["context"].shape == (2, 8)
    assert outputs["representation"].shape == (2, 16)
    assert not torch.any(masked_utilities.argmax(dim=1) == 1)


if __name__ == "__main__":
    test_qk_router_shapes_and_masking()
    test_qkv_router_shapes_and_masked_expert_cannot_be_selected_after_eval_mask()
    print("attention router tests passed")
