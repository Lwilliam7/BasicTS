import pytest
import torch

from experiments.chronological_adaptive_costar.run_chronological_adaptive_costar import (
    Trial,
    chronological_online_weights,
    enforce_observable,
)


def test_observable_assertion_respects_full_horizon():
    with pytest.raises(RuntimeError):
        enforce_observable(due_start=0, current_start=11, horizon=12)
    enforce_observable(due_start=0, current_start=12, horizon=12)


def test_online_weights_do_not_update_before_full_horizon():
    starts = torch.tensor([0, 5, 10, 12], dtype=torch.long)
    expert_mae = torch.tensor(
        [
            [0.0, 10.0, 10.0],
            [10.0, 0.0, 10.0],
            [10.0, 10.0, 0.0],
            [10.0, 10.0, 0.0],
        ],
        dtype=torch.float32,
    )
    train_mean = torch.ones(3, dtype=torch.float32)
    weights, extra = chronological_online_weights(
        starts=starts,
        expert_mae=expert_mae,
        horizon=12,
        trial=Trial("ema", "synthetic", decay=0.0, temperature=0.1),
        train_mean_mae=train_mean,
        mode="ema",
    )
    assert extra["num_updates"] == 1
    assert torch.allclose(weights[0], torch.full((3,), 1.0 / 3.0), atol=1e-6)
    assert torch.allclose(weights[1], torch.full((3,), 1.0 / 3.0), atol=1e-6)
    assert torch.allclose(weights[2], torch.full((3,), 1.0 / 3.0), atol=1e-6)
    assert int(weights[3].argmax()) == 0
