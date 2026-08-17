from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.sequential_costarts_fingerprint_model import SequentialCOSTARTSFingerprintRouter, compute_history_fingerprints


def test_history_fingerprint_shape_matches_router_dimension() -> None:
    history = torch.randn(3, 96, 7)
    fingerprints = compute_history_fingerprints(history, lags=(1, 2, 4, 8))
    model = SequentialCOSTARTSFingerprintRouter(
        num_experts=5,
        max_subset_size=5,
        input_len=96,
        forecast_horizon=12,
        num_features=7,
        embedding_dim=16,
        hidden_dim=16,
        fingerprint_mode="fingerprint_only",
    )
    assert fingerprints.shape == (3, model.fingerprint_dim)
    assert model.fingerprint_dim == 18 * 7


def test_fingerprint_router_modes_forward() -> None:
    history = torch.randn(2, 96, 7)
    queried_mask = torch.zeros(2, 5)
    queried_expert_ids = torch.full((2, 5), -1, dtype=torch.long)
    queried_forecasts = torch.zeros(2, 5, 12, 7)
    current_average = torch.zeros(2, 12, 7)

    for mode in ("embedding_only", "fingerprint_only", "embedding_fingerprint"):
        model = SequentialCOSTARTSFingerprintRouter(
            num_experts=5,
            max_subset_size=5,
            input_len=96,
            forecast_horizon=12,
            num_features=7,
            embedding_dim=16,
            hidden_dim=16,
            fingerprint_mode=mode,
        )
        outputs = model(history, queried_mask, queried_expert_ids, queried_forecasts, current_average_forecast=current_average)
        assert outputs["representation"].shape == (2, 16)
        assert outputs["utility_prediction"].shape == (2, 5)


if __name__ == "__main__":
    test_history_fingerprint_shape_matches_router_dimension()
    test_fingerprint_router_modes_forward()
    print("fingerprint router tests passed")
