import inspect
import json
from pathlib import Path

import torch

from scripts import train_costarts_subset_utility_router as costarts_train
from scripts import evaluate_costarts_final_comparison as final_comparison
from scripts.evaluate_costarts_cost_sweep import _finalize_predictions
from scripts.train_costarts_subset_utility_router import (
    build_cost_aware_targets,
    load_and_normalize_expert_costs,
    subset_utility_losses,
)


def _batch():
    num_experts = 3
    batch_size = 4
    queried_mask = torch.tensor(
        [
            [False, False, False],
            [True, False, False],
            [True, False, False],
            [True, True, False],
        ]
    )
    remaining_mask = ~queried_mask
    valid_action_mask = torch.cat((remaining_mask, queried_mask.any(dim=1, keepdim=True)), dim=1)
    expert_errors = torch.tensor(
        [
            [0.30, 0.10, 0.20],
            [0.20, 0.14, 0.19],
            [0.20, 0.19, 0.21],
            [0.30, 0.20, 0.19],
        ],
        dtype=torch.float32,
    )
    marginal_gain = torch.tensor(
        [
            [float("nan"), float("nan"), float("nan")],
            [float("-inf"), 0.06, 0.01],
            [float("-inf"), 0.01, -0.01],
            [float("-inf"), float("-inf"), 0.01],
        ],
        dtype=torch.float32,
    )
    max_subset_size = num_experts
    return {
        "history": torch.zeros(batch_size, 96, 7),
        "queried_mask": queried_mask,
        "remaining_mask": remaining_mask,
        "queried_expert_ids": torch.tensor(
            [
                [-1, -1, -1],
                [0, -1, -1],
                [0, -1, -1],
                [0, 1, -1],
            ]
        ),
        "queried_expert_forecasts": torch.zeros(batch_size, max_subset_size, 12, 7),
        "true_targets": torch.zeros(batch_size, 12, 7),
        "target_mask": torch.ones(batch_size, 12, 7, dtype=torch.bool),
        "true_expert_error_vector": expert_errors,
        "marginal_gain_best_queried_oracle": marginal_gain,
        "marginal_gain_equal_queried_average": marginal_gain.clone(),
        "cost_adjusted_utility": marginal_gain.clone(),
        "optimal_next_action": torch.tensor([1, 1, 1, 2]),
        "valid_action_mask": valid_action_mask,
        "pairwise_labels_queried": torch.zeros(batch_size, num_experts, num_experts, dtype=torch.int8),
        "pairwise_labels_remaining": torch.zeros(batch_size, num_experts, num_experts, dtype=torch.int8),
        "subset_size": queried_mask.sum(dim=1),
    }


def test_increasing_lambda_decreases_non_empty_query_utility():
    batch = _batch()
    costs = torch.tensor([1.0, 2.0, 4.0])
    utility_zero, _ = build_cost_aware_targets(batch, cost_coefficient=0.0, normalized_expert_costs=costs)
    utility_high, _ = build_cost_aware_targets(batch, cost_coefficient=0.5, normalized_expert_costs=costs)
    row = 1
    assert torch.isclose(utility_zero[row, 1] - utility_high[row, 1], torch.tensor(1.0))
    assert torch.isclose(utility_zero[row, 2] - utility_high[row, 2], torch.tensor(2.0))


def test_invalid_and_already_queried_experts_are_masked():
    utility, _ = build_cost_aware_targets(
        _batch(),
        cost_coefficient=0.0,
        normalized_expert_costs=torch.ones(3),
    )
    assert torch.isneginf(utility[1, 0])
    assert torch.isneginf(utility[3, 0])
    assert torch.isneginf(utility[3, 1])


def test_stop_optimal_when_remaining_utility_non_positive():
    _, actions = build_cost_aware_targets(
        _batch(),
        cost_coefficient=0.1,
        normalized_expert_costs=torch.ones(3),
    )
    assert int(actions[2]) == 3


def test_stop_invalid_for_empty_subset():
    batch = _batch()
    _, actions = build_cost_aware_targets(
        batch,
        cost_coefficient=100.0,
        normalized_expert_costs=torch.ones(3),
    )
    assert int(actions[0]) != 3
    assert bool(batch["valid_action_mask"][0, 3]) is False


def test_lambda_zero_reproduces_zero_cost_targets():
    batch = _batch()
    utility, actions = build_cost_aware_targets(
        batch,
        cost_coefficient=0.0,
        normalized_expert_costs=torch.tensor([9.0, 2.0, 5.0]),
    )
    assert torch.isclose(utility[0, 1], -batch["true_expert_error_vector"][0, 1])
    assert torch.isclose(utility[1, 1], batch["marginal_gain_equal_queried_average"][1, 1])
    assert int(actions[0]) == 1
    assert int(actions[1]) == 1


def test_cost_aware_targets_use_equal_average_marginal_gain():
    batch = _batch()
    batch["marginal_gain_best_queried_oracle"] = torch.tensor(
        [
            [float("nan"), float("nan"), float("nan")],
            [float("-inf"), 1.00, 0.01],
            [float("-inf"), 0.01, -0.01],
            [float("-inf"), float("-inf"), 0.01],
        ],
        dtype=torch.float32,
    )
    batch["marginal_gain_equal_queried_average"] = torch.tensor(
        [
            [float("nan"), float("nan"), float("nan")],
            [float("-inf"), -0.01, 0.20],
            [float("-inf"), 0.01, -0.01],
            [float("-inf"), float("-inf"), 0.01],
        ],
        dtype=torch.float32,
    )

    utility, actions = build_cost_aware_targets(
        batch,
        cost_coefficient=0.0,
        normalized_expert_costs=torch.ones(3),
    )

    assert torch.isclose(utility[1, 1], torch.tensor(-0.01))
    assert torch.isclose(utility[1, 2], torch.tensor(0.20))
    assert int(actions[1]) == 2


def test_two_lambdas_can_produce_different_labels():
    batch = _batch()
    low_actions = build_cost_aware_targets(
        batch,
        cost_coefficient=0.0,
        normalized_expert_costs=torch.ones(3),
    )[1]
    high_actions = build_cost_aware_targets(
        batch,
        cost_coefficient=0.1,
        normalized_expert_costs=torch.ones(3),
    )[1]
    assert int(low_actions[1]) == 1
    assert int(high_actions[1]) == 3


def test_changing_lambda_changes_training_loss_on_same_batch():
    batch = _batch()
    outputs = {
        "action_logits": torch.zeros(4, 4),
        "utility_prediction": torch.zeros(4, 3),
        "expert_score": torch.zeros(4, 3),
        "mix_logits": torch.zeros(4, 3),
    }
    weights = {"action": 1.0, "utility": 1.0, "pairwise": 0.0, "mix": 0.0}
    loss_zero, _ = subset_utility_losses(
        outputs,
        batch,
        weights,
        cost_coefficient=0.0,
        normalized_expert_costs=torch.ones(3),
    )
    loss_high, _ = subset_utility_losses(
        outputs,
        batch,
        weights,
        cost_coefficient=0.1,
        normalized_expert_costs=torch.ones(3),
    )
    assert not torch.isclose(loss_zero, loss_high)


def test_cost_normalization_is_identical_for_train_and_val_ordering():
    expert_names = ("DLinear", "PatchTST", "ModernTCN")
    raw_a, norm_a, _ = load_and_normalize_expert_costs(expert_names, cost_mode="equal")
    raw_b, norm_b, _ = load_and_normalize_expert_costs(expert_names, cost_mode="equal")
    assert torch.equal(raw_a, raw_b)
    assert torch.equal(norm_a, norm_b)
    assert torch.allclose(norm_a, torch.ones(3))


def test_training_module_does_not_load_test_data_or_frozen_experts():
    source = inspect.getsource(costarts_train.train_subset_utility_costarts_router)
    assert "router_val" in source
    assert "router_test" not in source
    assert "load_selected_experts" not in source
    assert "experts_loaded" in source
    assert "False" in source


def test_cost_sweep_equal_average_finalizer_uses_queried_forecast_mean():
    class _Router:
        def __call__(self, history, queried_mask, queried_expert_ids, queried_expert_forecasts):
            batch_size, num_experts = queried_mask.shape
            return {
                "expert_score": torch.zeros(batch_size, num_experts),
                "mix_logits": torch.zeros(batch_size, num_experts),
            }

    cache = {
        "num_experts": 3,
        "history": torch.zeros(2, 96, 7),
        "queried_mask": torch.tensor([[True, False, True], [False, True, False]]),
        "queried_expert_ids": torch.tensor([[0, 2, -1], [1, -1, -1]]),
        "queried_expert_forecasts": torch.tensor(
            [
                [[[2.0]], [[6.0]], [[0.0]]],
                [[[10.0]], [[0.0]], [[0.0]]],
            ]
        ),
        "true_targets": torch.zeros(2, 1, 1),
        "target_mask": torch.ones(2, 1, 1, dtype=torch.bool),
        "true_expert_error_vector": torch.tensor([[2.0, 5.0, 6.0], [3.0, 10.0, 1.0]]),
        "remaining_mask": torch.tensor([[False, True, False], [True, False, True]]),
        "marginal_gain_best_queried_oracle": torch.zeros(2, 3),
        "marginal_gain_equal_queried_average": torch.zeros(2, 3),
        "valid_action_mask": torch.ones(2, 4, dtype=torch.bool),
        "subset_size": torch.tensor([2, 1]),
    }
    prediction, targets, masks, selected = _finalize_predictions(
        router=_Router(),
        cache=cache,
        final_state_indices=[0, 1],
        batch_size=2,
        device=torch.device("cpu"),
        finalizer="equal_average",
    )
    assert torch.equal(prediction, torch.tensor([[[4.0]], [[10.0]]]))
    assert torch.equal(targets, torch.zeros(2, 1, 1))
    assert torch.equal(masks, torch.ones(2, 1, 1, dtype=torch.bool))
    assert torch.equal(selected, torch.tensor([0, 1]))


def test_deployable_validation_mae_uses_equal_average_finalizer():
    class _Router:
        def eval(self):
            return self

        def __call__(self, history, queried_mask, queried_expert_ids, queried_expert_forecasts):
            batch_size, num_experts = queried_mask.shape
            action_logits = torch.full((batch_size, num_experts + 1), -10.0)
            for index in range(batch_size):
                if not bool(queried_mask[index, 0]):
                    action_logits[index, 0] = 10.0
                elif not bool(queried_mask[index, 1]):
                    action_logits[index, 1] = 10.0
                else:
                    action_logits[index, 2] = 10.0
            return {
                "action_logits": action_logits,
                "utility_prediction": torch.zeros(batch_size, num_experts),
                "expert_score": torch.tensor([[10.0, -10.0]]).expand(batch_size, -1),
                "mix_logits": torch.zeros(batch_size, num_experts),
            }

    queried_mask = torch.tensor(
        [
            [False, False],
            [True, False],
            [False, True],
            [True, True],
        ]
    )
    queried_ids = torch.tensor([[-1, -1], [0, -1], [1, -1], [0, 1]])
    queried_forecasts = torch.zeros(4, 2, 1, 1)
    queried_forecasts[1, 0, 0, 0] = 0.0
    queried_forecasts[2, 0, 0, 0] = 4.0
    queried_forecasts[3, 0, 0, 0] = 0.0
    queried_forecasts[3, 1, 0, 0] = 4.0
    valid_action_mask = torch.tensor(
        [
            [True, True, False],
            [False, True, True],
            [True, False, True],
            [False, False, True],
        ]
    )
    cache = {
        "split_role": "router_val",
        "source_split_role": "router_val",
        "num_source_windows": 1,
        "num_states": 4,
        "num_experts": 2,
        "max_subset_size": 2,
        "forecast_horizon": 1,
        "num_features": 1,
        "subset_sampling_mode": "exhaustive",
        "stop_action_index": 2,
        "source_sample_indices_contiguous": True,
        "state_id": torch.arange(4),
        "sample_index": torch.zeros(4, dtype=torch.long),
        "source_row": torch.zeros(4, dtype=torch.long),
        "subset_size": queried_mask.sum(dim=1),
        "queried_mask": queried_mask,
        "remaining_mask": ~queried_mask,
        "queried_expert_ids": queried_ids,
        "queried_expert_forecasts": queried_forecasts,
        "history": torch.zeros(4, 96, 1),
        "true_targets": torch.full((4, 1, 1), 2.0),
        "target_mask": torch.ones(4, 1, 1, dtype=torch.bool),
        "true_expert_error_vector": torch.full((4, 2), 2.0),
        "current_loss_best_queried_oracle": torch.full((4,), float("nan")),
        "current_loss_equal_queried_average": torch.full((4,), float("nan")),
        "current_loss_deployable_reranker": torch.full((4,), float("nan")),
        "marginal_gain_best_queried_oracle": torch.zeros(4, 2),
        "marginal_gain_equal_queried_average": torch.zeros(4, 2),
        "cost_adjusted_utility": torch.zeros(4, 2),
        "optimal_next_action": torch.tensor([0, 1, 0, 2]),
        "valid_action_mask": valid_action_mask,
        "pairwise_labels_queried": torch.zeros(4, 2, 2, dtype=torch.int8),
        "pairwise_labels_remaining": torch.zeros(4, 2, 2, dtype=torch.int8),
    }

    metrics = costarts_train.evaluate_deployable_inference(
        _Router(),
        cache,
        batch_size=1,
        device=torch.device("cpu"),
    )

    assert metrics["validation_mae"] == 0.0
    assert metrics["validation_mse"] == 0.0


def test_final_comparison_fixed_and_weighted_baselines_use_train_split(tmp_path, monkeypatch):
    expert_names = ("A", "B", "C")
    num_windows = 2
    train_errors = torch.tensor([[5.0, 1.0, 2.0], [5.0, 1.0, 2.0]])
    val_errors = torch.tensor([[1.0, 9.0, 9.0], [1.0, 9.0, 9.0]])

    def make_base_cache(split_role, errors):
        predictions = errors.view(num_windows, 1, 1, len(expert_names)).expand(
            num_windows,
            12,
            7,
            len(expert_names),
        )
        return {
            "split_role": split_role,
            "histories": torch.zeros(num_windows, 96, 7),
            "targets": torch.zeros(num_windows, 12, 7),
            "target_masks": torch.ones(num_windows, 12, 7, dtype=torch.bool),
            "prediction_stack": predictions,
            "error_matrix": errors,
            "mse_matrix": errors.pow(2),
            "best_expert": torch.argmin(errors, dim=1),
            "sample_indices": torch.arange(num_windows),
            "num_windows": num_windows,
            "expert_names": expert_names,
            "input_len": 96,
            "forecast_horizon": 12,
            "num_features": 7,
        }

    train_cache = make_base_cache("router_train", train_errors)
    val_cache = make_base_cache("router_val", val_errors)
    subset_cache = {
        "expert_names": expert_names,
        "subset_size": torch.zeros(num_windows, dtype=torch.long),
        "sample_index": torch.arange(num_windows),
    }

    def fake_load(path):
        name = path.name
        if name == "train.pt":
            return train_cache
        if name == "val.pt":
            return val_cache
        if name == "subset_val.pt":
            return subset_cache
        raise AssertionError(f"Unexpected load path: {name}")

    monkeypatch.setattr(final_comparison, "_load_torch", fake_load)
    monkeypatch.setattr(final_comparison, "validate_costarts_subset_states", lambda cache: None)

    payload = final_comparison.evaluate_final_comparison(
        train_cache_path=tmp_path / "train.pt",
        val_cache_path=tmp_path / "val.pt",
        subset_val_cache_path=tmp_path / "subset_val.pt",
        old_costarts_checkpoint=tmp_path / "missing_old.pt",
        subset_checkpoint=tmp_path / "missing_subset.pt",
        routerdc_no_contrastive_checkpoint=tmp_path / "missing_no_contrastive.pt",
        routerdc_contrastive_checkpoint=tmp_path / "missing_contrastive.pt",
        output_dir=tmp_path,
        batch_size=2,
        device=torch.device("cpu"),
        seed=7,
        ridge=1e-4,
    )

    rows = {row["method"]: row for row in payload["rows"]}
    assert "validation_weighted_average" not in rows
    assert rows["best_fixed_expert"]["selection_split"] == "router_train"
    assert rows["best_fixed_expert"]["first_query_oracle_match"] == 0.0
    assert rows["best_fixed_expert"]["note"] == "Best fixed expert selected on router_train: B."
    assert rows["train_weighted_average"]["selection_split"] == "router_train"

    train_mean = train_errors.mean(dim=0)
    train_weights = (1.0 / train_mean) / (1.0 / train_mean).sum()
    expected_train_mae = float((val_errors[0] * train_weights).sum())
    assert torch.isclose(
        torch.tensor(rows["train_weighted_average"]["mae"]),
        torch.tensor(expected_train_mae),
    )

    val_mean = val_errors.mean(dim=0)
    val_weights = (1.0 / val_mean) / (1.0 / val_mean).sum()
    leaked_val_mae = float((val_errors[0] * val_weights).sum())
    assert not torch.isclose(
        torch.tensor(rows["train_weighted_average"]["mae"]),
        torch.tensor(leaked_val_mae),
    )


def test_tracked_final_comparison_artifacts_do_not_report_validation_fit_baselines():
    root = Path(__file__).resolve().parents[1]
    artifact_paths = (
        root / "results/router_summary/costarts_subset_utility/final_comparison.json",
        root / "results/router_summary/costarts_subset_utility/paper_package/tables/final_comparison.json",
    )
    for artifact_path in artifact_paths:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        rows = {row["method"]: row for row in payload["rows"]}
        assert "validation_weighted_average" not in rows
        assert rows["best_fixed_expert"]["selection_split"] == "router_train"
        for row in payload["rows"]:
            assert row["selection_split"] != "router_val_reference"
