import pytest
import torch

from scripts.costars import compare_sequential_costarts_aggregation as compare
from tests.test_sequential_costarts import ConstantUtilityRouter, _rollout_cache


def test_weighted_average_ignores_unqueried_forecasts():
    batch = {
        "queried_expert_ids": torch.tensor([[0, 2, -1]]),
        "queried_expert_forecasts": torch.tensor([[[[1.0]], [[3.0]], [[1000.0]]]]),
    }
    weights = torch.tensor([0.25, 100.0, 0.75])

    forecast = compare.weighted_average_forecast_from_state(batch, weights)

    assert forecast[0, 0, 0].item() == pytest.approx(2.5)


def test_uniform_weights_match_equal_average():
    batch = {
        "queried_expert_ids": torch.tensor([[0, 1, 2]]),
        "queried_expert_forecasts": torch.tensor([[[[1.0]], [[5.0]], [[9.0]]]]),
    }

    forecast = compare.weighted_average_forecast_from_state(batch, torch.ones(3))

    assert forecast[0, 0, 0].item() == pytest.approx(5.0)


def test_train_error_weights_reject_non_train_split():
    source = {
        "split_role": "router_val",
        "error_matrix": torch.ones(4, 3),
    }

    with pytest.raises(ValueError, match="router_train"):
        compare.train_error_softmax_weights(source, temperature=0.1)


def test_fit_global_convex_weights_rejects_non_train_split():
    source = {
        "split_role": "router_val",
        "prediction_stack": torch.zeros(2, 1, 1, 3),
        "targets": torch.zeros(2, 1, 1),
        "target_masks": torch.ones(2, 1, 1, dtype=torch.bool),
    }

    with pytest.raises(ValueError, match="router_train"):
        compare.fit_global_convex_weights(source, steps=1, learning_rate=0.1, device=torch.device("cpu"))


def test_route_final_state_indices_rejects_non_validation_split():
    cache = _rollout_cache()
    cache["split_role"] = "router_train"
    router = ConstantUtilityRouter([1.0, 1.0, 1.0])

    with pytest.raises(ValueError, match="router_val"):
        compare.route_final_state_indices(
            router,
            cache,
            fixed_first_expert=0,
            threshold=0.0,
            max_query_count=3,
            batch_size=8,
            device=torch.device("cpu"),
        )
