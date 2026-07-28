import pytest
import torch

from scripts.costars import train_old_costarts_pair_selector as old_selector


def _toy_cache(split="router_train", n=4):
    targets = torch.zeros(n, 12, 7)
    mask = torch.ones_like(targets, dtype=torch.bool)
    stack = torch.stack(
        [torch.full_like(targets, float(value)) for value in (1, 2, 3, 4, 5)],
        dim=-1,
    )
    mae, mse = old_selector.pair_error_matrices({
        "prediction_stack": stack,
        "targets": targets,
        "target_masks": mask,
    })
    expert_mae = torch.stack([torch.full((n,), float(value)) for value in (1, 2, 3, 4, 5)], dim=1)
    return {
        "split_role": split,
        "expert_names": old_selector.EXPECTED_EXPERTS,
        "input_len": 96,
        "forecast_horizon": 12,
        "num_features": 7,
        "num_windows": n,
        "histories": torch.zeros(n, 96, 7),
        "targets": targets,
        "target_masks": mask,
        "prediction_stack": stack,
        "error_matrix": expert_mae,
        "mse_matrix": expert_mae.square(),
        "sample_indices": torch.arange(n),
        "absolute_window_starts": torch.arange(n),
    }


def test_old_dataset_is_history_only_and_train_only():
    cache = _toy_cache("router_train")
    pair_mae, _ = old_selector.pair_error_matrices(cache)
    dataset = old_selector.OldCostartsPairSelectorDataset(cache, pair_mae, "soft", 0.01)
    item = dataset[0]

    assert set(item) == {"history", "hard_target", "source_index", "soft_target"}
    assert item["history"].shape == (96, 7)
    assert "targets" not in item
    assert "prediction_stack" not in item

    val = _toy_cache("router_val")
    with pytest.raises(ValueError, match="router_train"):
        old_selector.OldCostartsPairSelectorDataset(val, pair_mae, "soft", 0.01)


def test_old_cache_schema_rejects_wrong_expert_order(monkeypatch):
    cache = _toy_cache("router_train", n=2053)
    bad = dict(cache)
    bad["expert_names"] = tuple(reversed(old_selector.EXPECTED_EXPERTS))

    with pytest.raises(ValueError, match="Expert ordering"):
        old_selector.validate_old_costarts_cache(bad, "router_train")


def test_old_cache_schema_rejects_noncontiguous_sample_indices():
    cache = _toy_cache("router_val", n=613)
    cache["sample_indices"] = torch.arange(613) + 1

    with pytest.raises(ValueError, match="sample_indices"):
        old_selector.validate_old_costarts_cache(cache, "router_val")


def test_old_split_local_sample_ids_are_allowed_but_roles_must_differ():
    train = _toy_cache("router_train", n=2053)
    val = _toy_cache("router_val", n=613)
    old_selector.validate_cache_pair(train, val)

    with pytest.raises(ValueError, match="same split"):
        old_selector.validate_cache_pair(train, dict(train))


def test_reference_baselines_include_equal_average_all_experts():
    val = _toy_cache("router_val")
    pairs = old_selector.pair_class_order()
    train_pair_mae, _ = old_selector.pair_error_matrices(_toy_cache("router_train"), pairs)
    val_pair_mae, val_pair_mse = old_selector.pair_error_matrices(val, pairs)
    baselines = old_selector.build_baselines(train_pair_mae, val_pair_mae, val_pair_mse, val, pairs, {})

    assert baselines["equal_average_all_experts"]["average_experts_used"] == 5.0
    assert baselines["fixed_pair"]["pair"] == old_selector.FIXED_PAIR_NAME
    assert baselines["old_costarts_reference"]["source"].endswith("final_comparison.csv")
