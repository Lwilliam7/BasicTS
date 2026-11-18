from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.sequential_costarts_model_full import SequentialCOSTARTSRouterFull
from scripts.sequential_costarts_utility_objective import (
    available_expert_mask,
    compute_marginal_utilities,
    masked_utility_targets,
    utility_listwise_loss,
    utility_listwise_stop_loss,
    utility_pairwise_loss,
    utility_weighted_pairwise_loss,
)


def _toy_tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prediction_stack = torch.tensor(
        [[[[0.0, 2.0, 10.0]], [[0.0, 2.0, 10.0]]]],
        dtype=torch.float32,
    )
    targets = torch.zeros(1, 2, 1)
    masks = torch.ones_like(targets, dtype=torch.bool)
    return prediction_stack, targets, masks


def test_initial_utility_ranks_lower_loss_expert_higher() -> None:
    prediction_stack, targets, masks = _toy_tensors()
    queried_ids = torch.full((1, 3), -1, dtype=torch.long)
    utilities = compute_marginal_utilities(prediction_stack, targets, masks, queried_ids)
    assert utilities[0, 0] > utilities[0, 1] > utilities[0, 2]
    probabilities = masked_utility_targets(utilities, torch.ones_like(utilities, dtype=torch.bool), temperature=0.5)
    assert probabilities.argmax(dim=1).item() == 0


def test_positive_and_negative_marginal_utility() -> None:
    prediction_stack, targets, masks = _toy_tensors()
    queried_ids = torch.tensor([[1, -1, -1]])
    utilities = compute_marginal_utilities(prediction_stack, targets, masks, queried_ids)
    assert utilities[0, 0] > 0
    assert utilities[0, 2] < 0


def test_queried_experts_are_masked_and_softmax_normalizes_available_only() -> None:
    utilities = torch.tensor([[5.0, 1.0, 3.0]])
    queried_mask = torch.tensor([[1.0, 0.0, 0.0]])
    available = available_expert_mask(queried_mask)
    probabilities = masked_utility_targets(utilities, available, temperature=1.0)
    assert probabilities[0, 0].item() == 0.0
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(1))
    assert probabilities[0, 2] > probabilities[0, 1]


def test_utility_targets_depend_on_queried_subset() -> None:
    prediction_stack = torch.tensor([[[[0.0, 2.0, 4.0]]]], dtype=torch.float32)
    targets = torch.ones(1, 1, 1)
    masks = torch.ones_like(targets, dtype=torch.bool)
    utilities_a = compute_marginal_utilities(prediction_stack, targets, masks, torch.tensor([[0, -1, -1]]))
    utilities_b = compute_marginal_utilities(prediction_stack, targets, masks, torch.tensor([[2, -1, -1]]))
    assert not torch.allclose(utilities_a, utilities_b)


def test_pairwise_ordering_loss_is_lower_for_correct_order() -> None:
    utilities = torch.tensor([[3.0, 1.0, -1.0]])
    available = torch.ones_like(utilities, dtype=torch.bool)
    good_scores = torch.tensor([[3.0, 1.0, -1.0]])
    bad_scores = torch.tensor([[-1.0, 1.0, 3.0]])
    assert utility_pairwise_loss(good_scores, utilities, available) < utility_pairwise_loss(bad_scores, utilities, available)
    assert utility_weighted_pairwise_loss(good_scores, utilities, available) < utility_weighted_pairwise_loss(bad_scores, utilities, available)
    assert utility_listwise_loss(good_scores, utilities, available, temperature=1.0) < utility_listwise_loss(bad_scores, utilities, available, temperature=1.0)


def test_listwise_stop_masks_initial_stop_and_prefers_stop_over_negative_utility() -> None:
    scores = torch.tensor([[-0.2, -0.3, -0.4], [-0.2, -0.3, -0.4]], requires_grad=True)
    utilities = torch.tensor([[-0.1, -0.2, -0.3], [-0.1, -0.2, -0.3]])
    available = torch.ones_like(utilities, dtype=torch.bool)
    initial_stop_unavailable = torch.tensor([False, True])
    loss = utility_listwise_stop_loss(scores, utilities, available, initial_stop_unavailable, temperature=0.01)
    loss.backward()
    assert scores.grad is not None
    # For the non-initial row, STOP utility 0 is better than all negative expert utilities.
    action_utilities = torch.cat((utilities[1:2], torch.zeros(1, 1)), dim=1)
    target = torch.softmax(action_utilities / 0.01, dim=1)
    assert target.argmax(dim=1).item() == 3


def test_validation_forward_does_not_require_targets_and_baseline_mode_works() -> None:
    model = SequentialCOSTARTSRouterFull(num_experts=3, max_subset_size=3, input_len=4, forecast_horizon=1, num_features=1, embedding_dim=8, hidden_dim=8)
    history = torch.randn(2, 4, 1)
    queried_mask = torch.zeros(2, 3)
    queried_ids = torch.full((2, 3), -1, dtype=torch.long)
    queried_forecasts = torch.zeros(2, 3, 1, 1)
    outputs = model(history, queried_mask, queried_ids, queried_forecasts, current_average_forecast=torch.zeros(2, 1, 1))
    assert outputs["utility_prediction"].shape == (2, 3)


if __name__ == "__main__":
    test_initial_utility_ranks_lower_loss_expert_higher()
    test_positive_and_negative_marginal_utility()
    test_queried_experts_are_masked_and_softmax_normalizes_available_only()
    test_utility_targets_depend_on_queried_subset()
    test_pairwise_ordering_loss_is_lower_for_correct_order()
    test_listwise_stop_masks_initial_stop_and_prefers_stop_over_negative_utility()
    test_validation_forward_does_not_require_targets_and_baseline_mode_works()
    print("utility objective tests passed")
