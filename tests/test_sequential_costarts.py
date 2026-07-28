import pytest
import torch

from scripts.costars import train_sequential_costarts as sequential


EXPERTS = ("A", "B", "C")


def _subset_cache(split="router_train", sampling_mode="exhaustive", num_experts=3):
    num_states = 2
    horizon = 1
    features = 1
    valid_action_mask = torch.ones(num_states, num_experts + 1, dtype=torch.bool)
    return {
        "cache_type": "costarts_subset_states",
        "split_role": split,
        "source_split_role": split,
        "state_id": torch.arange(num_states),
        "sample_index": torch.arange(num_states),
        "source_row": torch.arange(num_states),
        "subset_size": torch.ones(num_states, dtype=torch.long),
        "queried_mask": torch.zeros(num_states, num_experts, dtype=torch.bool),
        "remaining_mask": torch.ones(num_states, num_experts, dtype=torch.bool),
        "queried_expert_ids": torch.full((num_states, num_experts), -1, dtype=torch.long),
        "queried_expert_forecasts": torch.zeros(num_states, num_experts, horizon, features),
        "history": torch.zeros(num_states, 96, features),
        "true_targets": torch.zeros(num_states, horizon, features),
        "target_mask": torch.ones(num_states, horizon, features, dtype=torch.bool),
        "true_expert_error_vector": torch.ones(num_states, num_experts),
        "current_loss_best_queried_oracle": torch.zeros(num_states),
        "current_loss_equal_queried_average": torch.ones(num_states),
        "current_loss_deployable_reranker": torch.zeros(num_states),
        "marginal_gain_best_queried_oracle": torch.zeros(num_states, num_experts),
        "marginal_gain_equal_queried_average": torch.zeros(num_states, num_experts),
        "cost_adjusted_utility": torch.zeros(num_states, num_experts),
        "optimal_next_action": torch.full((num_states,), num_experts, dtype=torch.long),
        "valid_action_mask": valid_action_mask,
        "pairwise_labels_queried": torch.zeros(num_states, num_experts, num_experts, dtype=torch.int8),
        "pairwise_labels_remaining": torch.zeros(num_states, num_experts, num_experts, dtype=torch.int8),
        "source_sample_indices_contiguous": True,
        "subset_sampling_mode": sampling_mode,
        "num_states": num_states,
        "num_source_windows": num_states,
        "num_experts": num_experts,
        "max_subset_size": num_experts,
        "forecast_horizon": horizon,
        "num_features": features,
        "expert_names": EXPERTS[:num_experts],
        "stop_action_index": num_experts,
        "cost_schedule_by_expert": {},
    }


def _rollout_cache(num_experts=3, num_windows=2):
    rows = []
    masks = []
    ids = []
    forecasts = []
    histories = []
    targets = []
    target_masks = []
    sample_indices = []
    true_errors = []
    gains = []
    valid_actions = []
    subset_sizes = []
    expert_values = torch.tensor([10.0, 0.0, 20.0])[:num_experts]
    for row in range(num_windows):
        for mask in range(1, 1 << num_experts):
            selected = [expert for expert in range(num_experts) if mask & (1 << expert)]
            selected_values = expert_values[selected]
            current_error = selected_values.mean().abs()
            state_gains = torch.full((num_experts,), float("-inf"))
            valid = torch.zeros(num_experts + 1, dtype=torch.bool)
            for expert in range(num_experts):
                if expert in selected:
                    continue
                new_error = torch.cat((selected_values, expert_values[expert : expert + 1])).mean().abs()
                state_gains[expert] = current_error - new_error
                valid[expert] = True
            valid[num_experts] = True
            state_ids = torch.full((num_experts,), -1, dtype=torch.long)
            state_forecasts = torch.zeros(num_experts, 1, 1)
            for slot, expert in enumerate(selected):
                state_ids[slot] = expert
                state_forecasts[slot, 0, 0] = expert_values[expert]
            rows.append(row)
            masks.append([(mask & (1 << expert)) != 0 for expert in range(num_experts)])
            ids.append(state_ids)
            forecasts.append(state_forecasts)
            histories.append(torch.zeros(96, 1))
            targets.append(torch.zeros(1, 1))
            target_masks.append(torch.ones(1, 1, dtype=torch.bool))
            sample_indices.append(row)
            true_errors.append(expert_values.abs())
            gains.append(state_gains)
            valid_actions.append(valid)
            subset_sizes.append(len(selected))
    return {
        "cache_type": "costarts_subset_states",
        "split_role": "router_val",
        "source_split_role": "router_val",
        "state_id": torch.arange(len(rows)),
        "sample_index": torch.tensor(sample_indices, dtype=torch.long),
        "source_row": torch.tensor(rows, dtype=torch.long),
        "subset_size": torch.tensor(subset_sizes, dtype=torch.long),
        "queried_mask": torch.tensor(masks, dtype=torch.bool),
        "remaining_mask": ~torch.tensor(masks, dtype=torch.bool),
        "queried_expert_ids": torch.stack(ids),
        "queried_expert_forecasts": torch.stack(forecasts),
        "history": torch.stack(histories),
        "true_targets": torch.stack(targets),
        "target_mask": torch.stack(target_masks),
        "true_expert_error_vector": torch.stack(true_errors),
        "marginal_gain_equal_queried_average": torch.stack(gains),
        "valid_action_mask": torch.stack(valid_actions),
        "source_sample_indices_contiguous": True,
        "subset_sampling_mode": "exhaustive",
        "num_states": len(rows),
        "num_source_windows": num_windows,
        "num_experts": num_experts,
        "max_subset_size": num_experts,
        "forecast_horizon": 1,
        "num_features": 1,
        "expert_names": EXPERTS[:num_experts],
        "stop_action_index": num_experts,
    }


class ConstantUtilityRouter(torch.nn.Module):
    def __init__(self, scores):
        super().__init__()
        self.scores = torch.tensor(scores, dtype=torch.float32)

    def forward(self, history, queried_mask, queried_expert_ids, queried_expert_forecasts):
        return self.scores.to(history.device).repeat(history.shape[0], 1)


def test_pair_improvement_targets_have_expected_signs():
    cache = _subset_cache()
    cache["marginal_gain_equal_queried_average"][0] = torch.tensor([0.0, 0.5, -0.25])
    cache["queried_mask"][0, 0] = True
    cache["valid_action_mask"][0, 0] = False

    targets = sequential.marginal_utility_targets(cache)

    assert targets[0, 0].item() == 0.0
    assert targets[0, 1].item() > 0.0
    assert targets[0, 2].item() < 0.0


def test_masking_prevents_repeated_expert_queries():
    scores = torch.tensor([[10.0, 9.0, 8.0]])
    queried = torch.tensor([[True, False, False]])

    masked = sequential.masked_utility_scores(scores, queried)

    assert masked.argmax(dim=1).item() == 1
    assert masked[0, 0].item() == pytest.approx(-1e9)


def test_dataset_does_not_expose_future_targets_or_unqueried_forecasts():
    cache = _subset_cache()
    cache["queried_mask"][0, 0] = True
    cache["queried_expert_ids"][0, 0] = 0
    cache["queried_expert_forecasts"][0, 0, 0, 0] = 3.0
    cache["valid_action_mask"][0, 0] = False
    cache["valid_action_mask"][1, :3] = False
    dataset = sequential.SequentialStateDataset(cache)

    item = dataset[0]

    assert "true_targets" not in item
    assert "candidate_loss_after_equal_average" not in item
    assert torch.equal(item["queried_expert_forecasts"][1:], torch.zeros_like(item["queried_expert_forecasts"][1:]))


def test_dataset_rejects_final_test_split():
    cache = _subset_cache("test")

    with pytest.raises(ValueError, match="router_train/router_val"):
        sequential.SequentialStateDataset(cache)


def test_equal_average_uses_all_and_only_queried_forecasts():
    batch = {
        "queried_expert_ids": torch.tensor([[0, 2, -1]]),
        "queried_expert_forecasts": torch.tensor([[[[1.0]], [[3.0]], [[100.0]]]]),
    }

    forecast = sequential.equal_average_forecast_from_state(batch)

    assert forecast[0, 0, 0].item() == pytest.approx(2.0)


def test_rollout_stops_when_best_score_is_below_threshold(monkeypatch):
    cache = _rollout_cache()
    router = ConstantUtilityRouter([-1.0, -1.0, -1.0])
    monkeypatch.setattr(sequential, "validate_costarts_subset_states", lambda cache: None)

    metrics = sequential.rollout_policy(
        router,
        cache,
        fixed_first_expert=0,
        threshold=0.0,
        max_query_count=3,
        batch_size=8,
        device=torch.device("cpu"),
    )

    assert metrics["average_experts_queried"] == pytest.approx(1.0)
    assert metrics["query_count_distribution"] == {"1": 2}


def test_rollout_enforces_max_query_count_and_never_repeats(monkeypatch):
    cache = _rollout_cache()
    router = ConstantUtilityRouter([100.0, 50.0, 25.0])
    monkeypatch.setattr(sequential, "validate_costarts_subset_states", lambda cache: None)

    metrics = sequential.rollout_policy(
        router,
        cache,
        fixed_first_expert=0,
        threshold=0.0,
        max_query_count=2,
        batch_size=8,
        device=torch.device("cpu"),
    )

    assert metrics["average_experts_queried"] == pytest.approx(2.0)
    assert metrics["query_count_distribution"] == {"2": 2}
    for row in metrics["per_window"]:
        queried = row["queried_experts"].split()
        assert len(queried) == len(set(queried))


def test_state_lookup_keeps_pair_indexing_consistent(monkeypatch):
    cache = _rollout_cache(num_windows=1)
    monkeypatch.setattr(sequential, "validate_costarts_subset_states", lambda cache: None)

    lookup = sequential.build_state_lookup(cache)

    assert lookup[0][1] != lookup[0][3]
    assert int(cache["queried_mask"][lookup[0][3]].sum().item()) == 2


def test_threshold_selection_uses_validation_cache_only(monkeypatch):
    cache = _rollout_cache()
    router = ConstantUtilityRouter([0.1, 0.2, 0.3])
    seen_splits = []

    def fake_rollout(router, cache, **kwargs):
        seen_splits.append(cache["split_role"])
        return {
            "validation_mae": 1.0,
            "validation_mse": 1.0,
            "average_experts_queried": 1.0,
            "useful_query_precision": 0.0,
            "harmful_query_rate": 0.0,
        }

    monkeypatch.setattr(sequential, "validate_costarts_subset_states", lambda cache: None)
    monkeypatch.setattr(sequential, "rollout_policy", fake_rollout)

    sequential.select_threshold(
        router,
        cache,
        fixed_first_expert=0,
        max_query_count=3,
        batch_size=8,
        device=torch.device("cpu"),
    )

    assert seen_splits
    assert set(seen_splits) == {"router_val"}


def test_cache_validation_rejects_test_split(monkeypatch):
    train = _subset_cache("test")
    val = _subset_cache("router_val")
    monkeypatch.setattr(sequential, "validate_costarts_subset_states", lambda cache: None)

    with pytest.raises(ValueError, match="router_train"):
        sequential.validate_sequential_caches(train, val)


def test_cache_validation_rejects_non_exhaustive_rollout_cache(monkeypatch):
    train = _subset_cache("router_train", sampling_mode="random")
    val = _subset_cache("router_val")
    monkeypatch.setattr(sequential, "validate_costarts_subset_states", lambda cache: None)

    with pytest.raises(ValueError, match="exhaustive"):
        sequential.validate_sequential_caches(train, val)


def test_checkpoint_metadata_names_sequential_costarts():
    router = sequential.SequentialCOSTARTSRouter(num_experts=3, max_subset_size=3, forecast_horizon=1, num_features=1)
    optimizer = torch.optim.AdamW(router.parameters(), lr=1e-3)
    path = sequential.ROOT / "results" / "router_summary" / "costarts_sequential" / "unit_test_checkpoint.pt"
    config = sequential.SequentialCOSTARTSConfig(output_dir=str(path.parent), results_dir=str(path.parent))
    try:
        sequential.save_checkpoint(
            path,
            router,
            optimizer,
            epoch=1,
            metrics={"validation_mae": 1.0},
            config=config,
            expert_names=EXPERTS,
            threshold=0.0,
            fixed_first_expert=0,
        )
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        assert checkpoint["router_type"] == "sequential_costarts"
        assert checkpoint["model_name"] == "Sequential COSTARTS"
        assert checkpoint["target"] == "marginal_gain_equal_queried_average"
        assert checkpoint["test_set_used"] is False
        assert checkpoint["experts_updated"] is False
    finally:
        path.unlink(missing_ok=True)
