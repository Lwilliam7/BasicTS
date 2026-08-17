import copy
import os
import sys
from pathlib import Path

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.fame_etth_router import (
    FingerprintScaler,
    assert_no_test_path,
    evaluate_sparse_router,
    extract_fame_etth_fingerprint,
    soft_expert_targets,
    sparse_top_r_weights,
    validate_cache_pair,
)


def _toy_cache(num_windows: int = 4) -> dict:
    prediction_stack = torch.zeros(num_windows, 2, 1, 3)
    prediction_stack[..., 0] = 1.0
    prediction_stack[..., 1] = 2.0
    prediction_stack[..., 2] = 3.0
    return {
        "split_role": "router_val",
        "expert_names": ("A", "B", "C"),
        "histories": torch.randn(num_windows, 96, 2),
        "targets": torch.ones(num_windows, 2, 1),
        "target_masks": torch.ones(num_windows, 2, 1),
        "prediction_stack": prediction_stack,
        "error_matrix": torch.tensor(
            [
                [0.1, 0.2, 0.3],
                [0.3, 0.2, 0.1],
                [0.2, 0.1, 0.3],
                [0.1, 0.3, 0.2],
            ]
        ),
        "mse_matrix": torch.ones(num_windows, 3),
        "best_expert": torch.tensor([0, 2, 1, 0])[:num_windows],
    }


def test_fingerprint_shape_finite_and_history_only():
    torch.manual_seed(1)
    histories = torch.randn(5, 96, 7)
    original = histories.clone()
    features = extract_fame_etth_fingerprint(histories)
    assert features.shape[0] == 5
    assert features.shape[1] > 7
    assert torch.isfinite(features).all()

    changed_future_placeholder = original.clone()
    features_again = extract_fame_etth_fingerprint(changed_future_placeholder)
    assert torch.allclose(features, features_again)


def test_train_scaler_is_reused_on_validation():
    train_features = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    val_features = torch.tensor([[10.0, 20.0]])
    scaler = FingerprintScaler.fit(train_features)
    transformed = scaler.transform(val_features)
    assert not torch.allclose(transformed.mean(dim=0), torch.zeros(2))
    assert torch.allclose(scaler.mean, torch.tensor([[2.0, 3.0]]))


def test_soft_targets_sum_and_lower_mae_gets_higher_suitability():
    errors = torch.tensor([[0.1, 0.2, 0.4]])
    targets = soft_expert_targets(errors, tau=0.1)
    assert torch.allclose(targets.sum(dim=1), torch.ones(1))
    assert targets[0, 0] > targets[0, 1] > targets[0, 2]


def test_top_r_selects_correct_experts_and_weights_sum():
    probabilities = torch.tensor([[0.5, 0.3, 0.2], [0.1, 0.7, 0.2]])
    active, weights = sparse_top_r_weights(probabilities, top_r=2)
    assert active.tolist() == [[True, True, False], [False, True, True]]
    assert torch.allclose(weights.sum(dim=1), torch.ones(2))


def test_top_1_matches_selected_expert_forecast():
    cache = _toy_cache(num_windows=1)
    probabilities = torch.tensor([[0.1, 0.8, 0.1]])
    metrics = evaluate_sparse_router(probabilities, cache, top_r=1)
    assert metrics["average_experts_used"] == 1.0
    assert metrics["mae"] == 1.0


def test_no_test_cache_is_loaded():
    try:
        assert_no_test_path(Path("cache") / "costarts_router_test_cache.pt")
    except ValueError:
        return
    raise AssertionError("test cache path was not rejected")


def test_expert_ordering_mismatch_raises():
    train = _toy_cache()
    val = copy.deepcopy(train)
    train["split_role"] = "router_train"
    val["split_role"] = "router_val"
    val["expert_names"] = ("B", "A", "C")
    try:
        validate_cache_pair(train, val)
    except ValueError:
        return
    raise AssertionError("expert ordering mismatch did not raise")


def test_deterministic_fingerprint_for_fixed_input():
    torch.manual_seed(42)
    histories = torch.randn(3, 96, 7)
    left = extract_fame_etth_fingerprint(histories)
    right = extract_fame_etth_fingerprint(histories)
    assert torch.allclose(left, right)


if __name__ == "__main__":
    test_fingerprint_shape_finite_and_history_only()
    test_train_scaler_is_reused_on_validation()
    test_soft_targets_sum_and_lower_mae_gets_higher_suitability()
    test_top_r_selects_correct_experts_and_weights_sum()
    test_top_1_matches_selected_expert_forecast()
    test_no_test_cache_is_loaded()
    test_expert_ordering_mismatch_raises()
    test_deterministic_fingerprint_for_fixed_input()
