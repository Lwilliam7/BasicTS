from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_sequential_costarts_forecast_state import (
    ForecastStateStopRouter,
    ablated_inputs,
    marginal_utilities_for_state,
    state_supervision,
)


def test_marginal_utility_depends_on_partial_subset() -> None:
    prediction_stack = torch.tensor([[[[0.0, 2.0, 4.0]]]], dtype=torch.float32)
    targets = torch.ones(1, 1, 1)
    masks = torch.ones_like(targets, dtype=torch.bool)
    metric_std = torch.ones(1)

    after_low_expert = marginal_utilities_for_state(
        prediction_stack,
        targets,
        masks,
        torch.tensor([[0, -1, -1]]),
        metric_std,
    )
    after_high_expert = marginal_utilities_for_state(
        prediction_stack,
        targets,
        masks,
        torch.tensor([[2, -1, -1]]),
        metric_std,
    )

    assert after_low_expert[0, 1] > 0
    assert after_high_expert[0, 1] > 0
    assert not torch.allclose(after_low_expert, after_high_expert)


def test_stop_label_turns_on_when_no_remaining_expert_beats_cost() -> None:
    prediction_stack = torch.tensor([[[[0.0, 0.2, 3.0]]]], dtype=torch.float32)
    targets = torch.zeros(1, 1, 1)
    masks = torch.ones_like(targets, dtype=torch.bool)
    metric_std = torch.ones(1)

    keep_going = state_supervision(
        prediction_stack,
        targets,
        masks,
        torch.tensor([[2, -1, -1]]),
        metric_std,
        lambda_cost=0.0,
        num_experts=3,
    )
    stop = state_supervision(
        prediction_stack,
        targets,
        masks,
        torch.tensor([[0, -1, -1]]),
        metric_std,
        lambda_cost=0.0,
        num_experts=3,
    )

    assert keep_going.stop_target.item() == 0.0
    assert stop.stop_target.item() == 1.0


def test_initial_state_cannot_be_stop_label_even_if_cost_is_large() -> None:
    prediction_stack = torch.tensor([[[[1.0, 2.0, 3.0]]]], dtype=torch.float32)
    targets = torch.zeros(1, 1, 1)
    masks = torch.ones_like(targets, dtype=torch.bool)
    metric_std = torch.ones(1)

    state = state_supervision(
        prediction_stack,
        targets,
        masks,
        torch.tensor([[-1, -1, -1]]),
        metric_std,
        lambda_cost=999.0,
        num_experts=3,
    )

    assert state.stop_target.item() == 0.0


def test_ablation_inputs_do_not_leak_forecast_state_for_history_only() -> None:
    model = ForecastStateStopRouter(
        num_experts=3,
        max_subset_size=3,
        input_len=4,
        forecast_horizon=1,
        num_features=1,
        embedding_dim=8,
        hidden_dim=8,
    )
    prediction_stack = torch.randn(2, 1, 1, 3)
    targets = torch.zeros(2, 1, 1)
    masks = torch.ones_like(targets, dtype=torch.bool)
    state = state_supervision(
        prediction_stack,
        targets,
        masks,
        torch.tensor([[0, -1, -1], [1, 2, -1]]),
        torch.ones(1),
        lambda_cost=0.0,
        num_experts=3,
    )
    history = torch.randn(2, 4, 1)
    mask, ids, forecasts, average, scalar = ablated_inputs(model, history, state, "history_only")

    assert torch.count_nonzero(mask).item() == 0
    assert torch.all(ids == -1)
    assert torch.count_nonzero(forecasts).item() == 0
    assert torch.count_nonzero(average).item() == 0
    assert torch.count_nonzero(scalar).item() == 0

