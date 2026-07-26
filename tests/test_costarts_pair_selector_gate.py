import torch

from scripts.evaluate_costarts_pair_selector_gate import (
    confidence_targets_from_pair_losses,
    gated_prediction_for_threshold,
    select_confidence_threshold,
    select_fixed_pair_class,
)
from scripts.train_costarts_pair_selector import build_pair_index


def _gate_cache() -> dict:
    histories = torch.zeros(3, 96, 7)
    targets = torch.tensor([[[0.0]], [[10.0]], [[4.0]]])
    target_masks = torch.ones(3, 1, 1, dtype=torch.bool)
    prediction_stack = torch.tensor(
        [
            [[[0.0, 2.0, 8.0]]],
            [[[8.0, 12.0, 20.0]]],
            [[[2.0, 6.0, 8.0]]],
        ],
        dtype=torch.float32,
    )
    error_matrix = (prediction_stack.squeeze(1).squeeze(1) - targets.view(3, 1)).abs()
    return {
        "split_role": "router_val",
        "histories": histories,
        "targets": targets,
        "target_masks": target_masks,
        "prediction_stack": prediction_stack,
        "error_matrix": error_matrix,
        "mse_matrix": error_matrix.pow(2),
        "best_expert": torch.argmin(error_matrix, dim=1),
        "sample_indices": torch.arange(3),
        "expert_names": ("a", "b", "c"),
        "num_windows": 3,
        "input_len": 96,
        "forecast_horizon": 1,
        "num_features": 1,
    }


def test_confidence_target_construction_labels_predicted_pair_wins():
    pair_mae = torch.tensor(
        [
            [1.0, 4.0, 5.0],
            [1.0, 4.0, 6.0],
            [3.0, 1.0, 2.0],
        ]
    )
    targets = confidence_targets_from_pair_losses(
        predicted_pair_class=torch.tensor([0, 2, 1]),
        fixed_pair_class=0,
        pair_mae=pair_mae,
    )

    assert torch.equal(targets["will_beat_fixed"], torch.tensor([False, False, True]))
    assert torch.allclose(targets["improvement"], torch.tensor([0.0, -5.0, 2.0]))


def test_threshold_selection_uses_supplied_validation_losses_only():
    cache = _gate_cache()
    pair_index = build_pair_index(3)
    pair_mae = torch.tensor(
        [
            [1.0, 4.0, 5.0],
            [1.0, 4.0, 6.0],
            [3.0, 1.0, 2.0],
        ]
    )
    selected = select_confidence_threshold(
        cache=cache,
        pair_index=pair_index,
        predicted_pair_class=torch.tensor([0, 2, 1]),
        fixed_pair_class=0,
        score=torch.tensor([0.1, 0.2, 0.9]),
        score_name="unit_score",
        pair_mae=pair_mae,
        steps=5,
    )

    assert selected["score_name"] == "unit_score"
    assert selected["policy"] in {"always_fixed_pair", "always_predicted_pair", "confidence_gated"}
    assert selected["threshold"] in [row["threshold"] for row in selected["threshold_rows"]]


def test_threshold_selection_respects_switch_rate_constraints():
    cache = _gate_cache()
    pair_index = build_pair_index(3)
    selected = select_confidence_threshold(
        cache=cache,
        pair_index=pair_index,
        predicted_pair_class=torch.tensor([2, 2, 1]),
        fixed_pair_class=0,
        score=torch.tensor([0.1, 0.2, 0.9]),
        score_name="unit_score",
        pair_mae=torch.tensor(
            [
                [5.0, 4.0, 1.0],
                [5.0, 4.0, 1.0],
                [5.0, 1.0, 4.0],
            ]
        ),
        steps=5,
        min_switch_rate=0.2,
        max_switch_rate=0.7,
    )

    assert selected["constraint_eligible"] is True
    assert 0.2 <= selected["switch_rate"] <= 0.7
    assert selected["constraints"]["fallback_to_unconstrained"] is False


def test_threshold_selection_reports_when_constraints_have_no_candidate():
    cache = _gate_cache()
    pair_index = build_pair_index(3)
    selected = select_confidence_threshold(
        cache=cache,
        pair_index=pair_index,
        predicted_pair_class=torch.tensor([2, 2, 1]),
        fixed_pair_class=0,
        score=torch.tensor([0.1, 0.2, 0.9]),
        score_name="unit_score",
        pair_mae=torch.ones(3, 3),
        steps=5,
        min_switch_rate=0.4,
        max_switch_rate=0.6,
    )

    assert selected["constraints"]["eligible_threshold_count"] == 0
    assert selected["constraints"]["fallback_to_unconstrained"] is True


def test_fixed_pair_fallback_when_threshold_is_infinite():
    cache = _gate_cache()
    pair_index = build_pair_index(3)
    gated = gated_prediction_for_threshold(
        cache=cache,
        pair_index=pair_index,
        predicted_pair_class=torch.tensor([2, 2, 2]),
        fixed_pair_class=0,
        score=torch.tensor([0.0, 1.0, 2.0]),
        threshold=float("inf"),
    )

    assert gated["switch_rate"] == 0.0
    assert gated["selected_pair_indices"].tolist() == [[0, 1], [0, 1], [0, 1]]


def test_switching_behavior_uses_predicted_pair_above_threshold():
    cache = _gate_cache()
    pair_index = build_pair_index(3)
    gated = gated_prediction_for_threshold(
        cache=cache,
        pair_index=pair_index,
        predicted_pair_class=torch.tensor([2, 2, 1]),
        fixed_pair_class=0,
        score=torch.tensor([0.0, 0.8, 0.9]),
        threshold=0.5,
    )

    assert gated["switch_mask"].tolist() == [False, True, True]
    assert gated["selected_pair_indices"].tolist() == [[0, 1], [1, 2], [0, 2]]


def test_fixed_pair_selection_can_use_router_validation_pair():
    train_pair_mae = torch.tensor([[1.0, 5.0, 6.0], [1.0, 5.0, 6.0]])
    val_pair_mae = torch.tensor([[3.0, 1.0, 4.0], [3.0, 1.0, 4.0]])

    assert select_fixed_pair_class(train_pair_mae, val_pair_mae, selection="router_train") == 0
    assert select_fixed_pair_class(train_pair_mae, val_pair_mae, selection="router_val") == 1
