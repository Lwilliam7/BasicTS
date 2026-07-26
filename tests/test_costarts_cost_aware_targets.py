import inspect
from pathlib import Path

import torch

from scripts import train_costarts_subset_utility_router as costarts_train
from scripts.train_costarts_subset_utility_router import (
    RouterDCFirstQueryModule,
    SubsetUtilityCOSTARTSRouter,
    build_cost_aware_targets,
    _equal_average_queried_forecasts,
    first_query_regret_loss,
    first_query_soft_targets,
    inspect_routerdc_first_query_checkpoint,
    load_routerdc_first_query_weights,
    load_and_normalize_expert_costs,
    subset_utility_losses,
)


def _scratch_path(filename):
    directory = Path(".codex_tmp_tests")
    directory.mkdir(exist_ok=True)
    return directory / filename


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
    assert torch.isclose(utility[1, 1], batch["marginal_gain_best_queried_oracle"][1, 1])
    assert int(actions[0]) == 1
    assert int(actions[1]) == 1


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


def _ensemble_batch():
    batch = _batch()
    batch["true_expert_error_vector"] = torch.tensor(
        [
            [0.10, 0.20, 0.30],
            [0.10, 0.20, 0.30],
            [0.10, 0.20, 0.30],
            [0.10, 0.20, 0.30],
        ],
        dtype=torch.float32,
    )
    batch["marginal_gain_best_queried_oracle"] = torch.tensor(
        [
            [float("nan"), float("nan"), float("nan")],
            [float("-inf"), 0.0, 0.0],
            [float("-inf"), 0.0, -0.01],
            [float("-inf"), float("-inf"), -0.01],
        ],
        dtype=torch.float32,
    )
    batch["marginal_gain_equal_queried_average"] = torch.tensor(
        [
            [float("nan"), float("nan"), float("nan")],
            [float("-inf"), 0.04, 0.01],
            [float("-inf"), -0.01, -0.02],
            [float("-inf"), float("-inf"), -0.01],
        ],
        dtype=torch.float32,
    )
    return batch


def test_equal_average_candidate_can_be_useful_when_standalone_is_not_better():
    batch = _ensemble_batch()
    best_utility, best_actions = build_cost_aware_targets(
        batch,
        cost_coefficient=0.0,
        normalized_expert_costs=torch.ones(3),
        utility_finalizer="best_single",
    )
    equal_utility, equal_actions = build_cost_aware_targets(
        batch,
        cost_coefficient=0.0,
        normalized_expert_costs=torch.ones(3),
        utility_finalizer="equal_average",
    )
    assert best_utility[1, 1] <= 0
    assert equal_utility[1, 1] > 0
    assert int(best_actions[1]) == 3
    assert int(equal_actions[1]) == 1


def test_equal_average_stop_when_all_ensemble_utilities_non_positive():
    _, actions = build_cost_aware_targets(
        _ensemble_batch(),
        cost_coefficient=0.0,
        normalized_expert_costs=torch.ones(3),
        utility_finalizer="equal_average",
    )
    assert int(actions[2]) == 3


def test_equal_average_queried_experts_remain_masked():
    utility, _ = build_cost_aware_targets(
        _ensemble_batch(),
        cost_coefficient=0.0,
        normalized_expert_costs=torch.ones(3),
        utility_finalizer="equal_average",
    )
    assert torch.isneginf(utility[1, 0])
    assert torch.isneginf(utility[3, 0])
    assert torch.isneginf(utility[3, 1])


def test_equal_average_stop_invalid_for_empty_subset():
    batch = _ensemble_batch()
    _, actions = build_cost_aware_targets(
        batch,
        cost_coefficient=100.0,
        normalized_expert_costs=torch.ones(3),
        utility_finalizer="equal_average",
    )
    assert int(actions[0]) != 3
    assert bool(batch["valid_action_mask"][0, 3]) is False


def test_equal_costs_do_not_change_expert_ordering():
    batch = _ensemble_batch()
    low, _ = build_cost_aware_targets(
        batch,
        cost_coefficient=0.0,
        normalized_expert_costs=torch.ones(3),
        utility_finalizer="equal_average",
    )
    high, _ = build_cost_aware_targets(
        batch,
        cost_coefficient=0.1,
        normalized_expert_costs=torch.ones(3),
        utility_finalizer="equal_average",
    )
    row = 1
    valid = batch["valid_action_mask"][row, :3]
    assert torch.equal(torch.argsort(low[row, valid]), torch.argsort(high[row, valid]))


def test_deployment_equal_average_uses_only_queried_forecasts():
    batch = _batch()
    forecasts = torch.zeros(2, 3, 12, 7)
    forecasts[0, 0] = 1.0
    forecasts[0, 1] = 3.0
    forecasts[0, 2] = 1000.0
    forecasts[1, 0] = 2.0
    forecasts[1, 1] = 8.0
    forecasts[1, 2] = 1000.0
    batch = {
        "queried_expert_ids": torch.tensor([[0, 1, -1], [0, 2, -1]]),
        "queried_expert_forecasts": forecasts,
    }
    before = _equal_average_queried_forecasts(batch)
    batch["queried_expert_forecasts"][0, 2] = -9999.0
    after = _equal_average_queried_forecasts(batch)
    assert torch.allclose(before, after)
    assert torch.allclose(before[0], torch.full((12, 7), 2.0))
    assert torch.allclose(before[1], torch.full((12, 7), 5.0))


def test_validation_loader_is_not_used_for_training_source():
    source = inspect.getsource(costarts_train.train_subset_utility_costarts_router)
    assert "train_dataset = CostartsSubsetStateDataset(train_cache)" in source
    assert "DataLoader(\n        train_dataset" in source
    assert "CostartsSubsetStateDataset(val_cache)" not in source


def test_router_forward_inputs_exclude_oracle_label_tensors():
    signature = inspect.signature(costarts_train.SubsetUtilityCOSTARTSRouter.forward)
    assert list(signature.parameters) == [
        "self",
        "history",
        "queried_mask",
        "queried_expert_ids",
        "queried_expert_forecasts",
    ]


def test_first_query_soft_targets_sum_to_one():
    targets = first_query_soft_targets(torch.tensor([[0.3, 0.1, 0.2]]), temperature=0.02)
    assert torch.allclose(targets.sum(dim=-1), torch.ones(1))


def test_lower_error_gets_larger_first_query_probability():
    targets = first_query_soft_targets(torch.tensor([[0.3, 0.1, 0.2]]), temperature=0.05)
    assert targets[0, 1] > targets[0, 2] > targets[0, 0]


def test_equal_errors_get_equal_first_query_probability():
    targets = first_query_soft_targets(torch.tensor([[0.2, 0.2, 0.2]]), temperature=0.05)
    assert torch.allclose(targets, torch.full((1, 3), 1 / 3))


def test_lower_temperature_sharpens_first_query_distribution():
    errors = torch.tensor([[0.3, 0.1, 0.2]])
    warm = first_query_soft_targets(errors, temperature=0.2)
    cold = first_query_soft_targets(errors, temperature=0.02)
    assert cold.max() > warm.max()


def test_first_query_regret_loss_near_zero_for_oracle_probability():
    logits = torch.tensor([[-100.0, 100.0, -100.0]])
    errors = torch.tensor([[0.3, 0.1, 0.2]])
    assert first_query_regret_loss(logits, errors) < 1e-6


def test_first_query_regret_loss_positive_for_worse_expert():
    logits = torch.tensor([[100.0, -100.0, -100.0]])
    errors = torch.tensor([[0.3, 0.1, 0.2]])
    assert first_query_regret_loss(logits, errors) > 0


def test_separate_first_query_head_outputs_expert_logits_only():
    router = SubsetUtilityCOSTARTSRouter(
        num_experts=3,
        max_subset_size=3,
        first_query_head_type="separate",
    )
    batch = _batch()
    outputs = router(
        batch["history"],
        batch["queried_mask"],
        batch["queried_expert_ids"],
        batch["queried_expert_forecasts"],
    )
    assert tuple(outputs["first_query_logits"].shape) == (4, 3)
    assert tuple(outputs["action_logits"].shape) == (4, 4)


def test_empty_first_query_ignores_invalid_queried_forecast_slots():
    router = SubsetUtilityCOSTARTSRouter(
        num_experts=3,
        max_subset_size=3,
        first_query_head_type="separate",
    )
    batch = _batch()
    history = batch["history"][:1]
    queried_mask = batch["queried_mask"][:1]
    queried_ids = batch["queried_expert_ids"][:1]
    forecasts_a = batch["queried_expert_forecasts"][:1].clone()
    forecasts_b = forecasts_a.clone()
    forecasts_b[:, 0] = 999.0
    first_a = router(history, queried_mask, queried_ids, forecasts_a)["first_query_logits"]
    first_b = router(history, queried_mask, queried_ids, forecasts_b)["first_query_logits"]
    assert torch.allclose(first_a, first_b)


def test_continuation_action_logits_exist_for_non_empty_states_with_separate_first_head():
    router = SubsetUtilityCOSTARTSRouter(
        num_experts=3,
        max_subset_size=3,
        first_query_head_type="separate",
    )
    batch = _batch()
    outputs = router(
        batch["history"][1:],
        batch["queried_mask"][1:],
        batch["queried_expert_ids"][1:],
        batch["queried_expert_forecasts"][1:],
    )
    assert tuple(outputs["action_logits"].shape) == (3, 4)


def _write_routerdc_checkpoint(
    path,
    *,
    expert_names=("A", "B", "C"),
    embedding_dim=8,
):
    module = RouterDCFirstQueryModule(
        input_len=96,
        num_features=7,
        num_experts=len(expert_names),
        embedding_dim=embedding_dim,
        hidden_dim=8,
        dropout=0.0,
        router_temperature=1.0,
    )
    torch.save(
        {
            "epoch": 3,
            "selected_expert_names": list(expert_names),
            "router_config": module.config_dict(),
            "router_state_dict": module.state_dict(),
        },
        path,
    )


def test_routerdc_checkpoint_compatibility_loads_matching_order_and_shapes():
    path = _scratch_path("routerdc_compatible.pt")
    _write_routerdc_checkpoint(path)
    config, report = inspect_routerdc_first_query_checkpoint(
        path,
        expert_names=("A", "B", "C"),
        input_len=96,
        num_features=7,
        num_experts=3,
        embedding_dim=8,
    )
    assert config["embedding_dim"] == 8
    assert report["expert_embeddings_shape"] == [3, 8]
    assert report["compatible_with_costarts_expert_order"] is True


def test_routerdc_checkpoint_rejects_mismatched_expert_order():
    path = _scratch_path("routerdc_bad_order.pt")
    _write_routerdc_checkpoint(path, expert_names=("A", "B", "C"))
    try:
        inspect_routerdc_first_query_checkpoint(
            path,
            expert_names=("B", "A", "C"),
            input_len=96,
            num_features=7,
            num_experts=3,
            embedding_dim=8,
        )
    except ValueError as exc:
        assert "expert order mismatch" in str(exc)
    else:
        raise AssertionError("expected expert order mismatch")


def test_routerdc_checkpoint_rejects_mismatched_embedding_dimension():
    path = _scratch_path("routerdc_bad_dim.pt")
    _write_routerdc_checkpoint(path, embedding_dim=8)
    try:
        inspect_routerdc_first_query_checkpoint(
            path,
            expert_names=("A", "B", "C"),
            input_len=96,
            num_features=7,
            num_experts=3,
            embedding_dim=16,
        )
    except ValueError as exc:
        assert "embedding_dim mismatch" in str(exc)
    else:
        raise AssertionError("expected embedding dimension mismatch")


def test_routerdc_frozen_first_query_receives_no_gradients():
    path = _scratch_path("routerdc_frozen.pt")
    _write_routerdc_checkpoint(path)
    config, _ = inspect_routerdc_first_query_checkpoint(
        path,
        expert_names=("A", "B", "C"),
        input_len=96,
        num_features=7,
        num_experts=3,
        embedding_dim=8,
    )
    router = SubsetUtilityCOSTARTSRouter(
        num_experts=3,
        max_subset_size=3,
        embedding_dim=8,
        hidden_dim=8,
        first_query_initialization="routerdc_frozen",
        routerdc_config=config,
    )
    load_routerdc_first_query_weights(router, path)
    batch = _batch()
    outputs = router(
        batch["history"],
        batch["queried_mask"],
        batch["queried_expert_ids"],
        batch["queried_expert_forecasts"],
    )
    loss, _ = subset_utility_losses(
        outputs,
        batch,
        {"action": 1.0, "utility": 1.0, "pairwise": 0.0, "mix": 0.0},
        cost_coefficient=0.0,
        normalized_expert_costs=torch.ones(3),
        first_query_target="soft",
        first_query_loss_weight=1.0,
    )
    loss.backward()
    assert all(parameter.grad is None for parameter in router.routerdc_first_query.parameters())


def test_routerdc_finetune_first_query_receives_gradients_from_empty_states():
    path = _scratch_path("routerdc_finetune.pt")
    _write_routerdc_checkpoint(path)
    config, _ = inspect_routerdc_first_query_checkpoint(
        path,
        expert_names=("A", "B", "C"),
        input_len=96,
        num_features=7,
        num_experts=3,
        embedding_dim=8,
    )
    router = SubsetUtilityCOSTARTSRouter(
        num_experts=3,
        max_subset_size=3,
        embedding_dim=8,
        hidden_dim=8,
        first_query_initialization="routerdc_finetune",
        routerdc_config=config,
    )
    load_routerdc_first_query_weights(router, path)
    batch = _batch()
    outputs = router(
        batch["history"],
        batch["queried_mask"],
        batch["queried_expert_ids"],
        batch["queried_expert_forecasts"],
    )
    loss, _ = subset_utility_losses(
        outputs,
        batch,
        {"action": 0.0, "utility": 0.0, "pairwise": 0.0, "mix": 0.0},
        cost_coefficient=0.0,
        normalized_expert_costs=torch.ones(3),
        first_query_target="soft",
        first_query_loss_weight=1.0,
    )
    loss.backward()
    assert any(parameter.grad is not None for parameter in router.routerdc_first_query.parameters())


def test_routerdc_first_query_does_not_replace_continuation_state_inputs():
    router = SubsetUtilityCOSTARTSRouter(
        num_experts=3,
        max_subset_size=3,
        first_query_initialization="routerdc_frozen",
        routerdc_config={
            "input_len": 96,
            "num_features": 7,
            "num_experts": 3,
            "embedding_dim": 64,
            "hidden_dim": 64,
            "dropout": 0.0,
            "router_temperature": 1.0,
            "router_type": "routerdc_hard",
        },
    )
    signature = inspect.signature(router.encode)
    assert list(signature.parameters) == [
        "history",
        "queried_mask",
        "queried_expert_ids",
        "queried_expert_forecasts",
    ]
    batch = _batch()
    outputs = router(
        batch["history"][1:],
        batch["queried_mask"][1:],
        batch["queried_expert_ids"][1:],
        batch["queried_expert_forecasts"][1:],
    )
    assert tuple(outputs["action_logits"].shape) == (3, 4)
