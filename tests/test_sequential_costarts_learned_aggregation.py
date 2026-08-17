from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_sequential_costarts_learned_aggregation import QueriedForecastCombiner


def test_queried_forecast_combiner_masks_unqueried_slots() -> None:
    combiner = QueriedForecastCombiner(num_experts=5, max_queries=5, horizon=12, num_features=7, state_dim=16, hidden_dim=8)
    state = torch.randn(2, 16)
    queried_ids = torch.tensor([[2, -1, -1, -1, -1], [1, 3, -1, -1, -1]], dtype=torch.long)
    queried_forecasts = torch.randn(2, 5, 12, 7)
    outputs = combiner(state, queried_ids, queried_forecasts)
    assert outputs["prediction"].shape == (2, 12, 7)
    assert outputs["weights"].shape == (2, 5)
    assert torch.allclose(outputs["weights"][0, 1:], torch.zeros(4), atol=1e-7)
    assert torch.allclose(outputs["weights"][1, 2:], torch.zeros(3), atol=1e-7)
    assert torch.allclose(outputs["weights"].sum(dim=1), torch.ones(2), atol=1e-6)


if __name__ == "__main__":
    test_queried_forecast_combiner_masks_unqueried_slots()
    print("learned aggregation tests passed")
