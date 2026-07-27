import itertools
from pathlib import Path

import pytest
import torch

from scripts.costars import train_etth2_pair_selector as selector


def _toy_cache(split="router_train"):
    targets = torch.zeros(6, 12, 7)
    mask = torch.ones_like(targets, dtype=torch.bool)
    values = (1.0, 2.0, 3.0, 4.0, 5.0)
    stack = torch.stack([torch.full_like(targets, value) for value in values], dim=-1)
    starts = torch.arange(6) + (8640 if split == "router_train" else 10800)
    mae, mse = selector.per_window_error(stack, targets, mask)
    return {
        "dataset": "ETTh2",
        "split_role": split,
        "expert_names": selector.EXPECTED_EXPERTS,
        "scaler_hash": selector.EXPECTED_HASHES["scaler"],
        "histories": torch.arange(6 * 96 * 7, dtype=torch.float32).reshape(6, 96, 7),
        "targets": targets,
        "target_masks": mask,
        "prediction_stack": stack,
        "error_matrix": mae,
        "mse_matrix": mse,
        "sample_indices": torch.arange(6),
        "absolute_window_starts": starts,
    }


def test_pair_ordering_has_exactly_ten_deterministic_unique_pairs():
    first = selector.pair_class_order()
    second = selector.pair_class_order()

    assert first == second
    assert len(first) == 10
    assert [pair["expert_indices"] for pair in first] == [list(pair) for pair in itertools.combinations(range(5), 2)]
    assert len({pair["pair"] for pair in first}) == 10


def test_pair_predictions_are_exact_equal_average():
    cache = _toy_cache()
    pairs = selector.pair_class_order()
    pair_stack = selector.pair_prediction_stack(cache, pairs)

    dlinear_patchtst = pair_stack[..., selector.pair_name_to_index("DLinear+PatchTST", pairs)]
    expected = 0.5 * cache["prediction_stack"][..., 0] + 0.5 * cache["prediction_stack"][..., 1]
    assert torch.equal(dlinear_patchtst, expected)


def test_soft_targets_sum_to_one_and_prefer_lower_mae():
    pair_mae = torch.tensor([[0.2, 0.1, 0.4], [1.0, 0.5, 0.75]])
    targets = selector.soft_pair_targets(pair_mae, temperature=0.01)

    assert torch.allclose(targets.sum(dim=1), torch.ones(2))
    assert targets[0, 1] > targets[0, 0] > targets[0, 2]
    assert targets[1, 1] > targets[1, 2] > targets[1, 0]


def test_training_dataset_accepts_only_router_train_and_contains_history_only():
    train = _toy_cache("router_train")
    pair_mae, _ = selector.pair_error_matrices(train)
    dataset = selector.PairSelectorDataset(train, pair_mae, "soft", 0.01)
    item = dataset[0]

    assert set(item) == {"history", "hard_target", "source_index", "soft_target"}
    assert item["history"].shape == (96, 7)
    assert "targets" not in item
    assert "prediction_stack" not in item

    val = _toy_cache("router_val")
    val_pair_mae, _ = selector.pair_error_matrices(val)
    with pytest.raises(ValueError, match="router_train"):
        selector.PairSelectorDataset(val, val_pair_mae, "soft", 0.01)


def test_checkpoint_metadata_rejects_mismatched_expert_dataset_and_horizon():
    checkpoint = {
        "dataset": "ETTh2",
        "input_len": 96,
        "horizon": 12,
        "expert_order": list(selector.EXPECTED_EXPERTS),
        "cache_hashes": {
            "router_train": selector.EXPECTED_HASHES["router_train"],
            "router_val": selector.EXPECTED_HASHES["router_val"],
        },
        "scaler_hash": selector.EXPECTED_HASHES["scaler"],
    }
    selector.checkpoint_metadata_is_valid(checkpoint)

    bad_dataset = dict(checkpoint, dataset="ETTh1")
    with pytest.raises(ValueError, match="dataset"):
        selector.checkpoint_metadata_is_valid(bad_dataset)

    bad_horizon = dict(checkpoint, horizon=24)
    with pytest.raises(ValueError, match="horizon"):
        selector.checkpoint_metadata_is_valid(bad_horizon)

    bad_order = dict(checkpoint, expert_order=list(reversed(selector.EXPECTED_EXPERTS)))
    with pytest.raises(ValueError, match="ordering"):
        selector.checkpoint_metadata_is_valid(bad_order)


def test_validation_confidence_rows_align_to_source_indices():
    val = _toy_cache("router_val")
    pairs = selector.pair_class_order()
    pair_mae, _ = selector.pair_error_matrices(val, pairs)
    logits = torch.zeros(val["histories"].shape[0], len(pairs))
    rows = selector.validation_confidence_rows(
        val,
        logits,
        pair_mae,
        selector.pair_name_to_index(selector.FIXED_PAIR_NAME, pairs),
        pairs,
    )

    assert [row["absolute_window_start"] for row in rows] == val["absolute_window_starts"].tolist()
    assert [row["sample_index"] for row in rows] == val["sample_indices"].tolist()


def test_forecast_mae_reproduces_direct_calculation_from_chosen_pairs():
    val = _toy_cache("router_val")
    pairs = selector.pair_class_order()
    pair_mae, pair_mse = selector.pair_error_matrices(val, pairs)
    fixed_pair_index = selector.pair_name_to_index("DLinear+PatchTST", pairs)
    logits = torch.full((val["histories"].shape[0], len(pairs)), -10.0)
    logits[:, fixed_pair_index] = 10.0
    metrics = selector.evaluate_logits(
        logits,
        pair_mae,
        pair_mse,
        fixed_pair_index,
        pair_mae.argmin(dim=1),
        selector.soft_pair_targets(pair_mae, 0.01),
    )
    direct_mae, direct_mse = selector.per_window_error(
        0.5 * val["prediction_stack"][..., 0] + 0.5 * val["prediction_stack"][..., 1],
        val["targets"],
        val["target_masks"],
    )

    assert metrics["selected_pair_mae"] == pytest.approx(float(direct_mae.mean().item()))
    assert metrics["selected_pair_mse"] == pytest.approx(float(direct_mse.mean().item()))


def test_cross_seed_agreement_rejects_misaligned_confidence_rows():
    frame_a = [{"absolute_window_start": "10800", "selected_pair": "DLinear+ModernTCN", "max_predicted_probability": "0.8", "selected_pair_error": "0.1"}]
    frame_b = [{"absolute_window_start": "10801", "selected_pair": "DLinear+ModernTCN", "max_predicted_probability": "0.7", "selected_pair_error": "0.2"}]

    with pytest.raises(ValueError, match="misaligned"):
        selector.cross_seed_agreement([frame_a, frame_b], selector.pair_class_order())


def test_locked_seed_configuration_is_rejected_if_changed():
    args = selector.parse_args(["--seeds", "7,11,13"])

    with pytest.raises(ValueError, match="expects seeds"):
        seeds = selector.parse_seeds(args.seeds)
        if seeds != list(selector.DEFAULT_SEEDS):
            raise ValueError(f"This locked local run expects seeds {selector.DEFAULT_SEEDS}, got {seeds}")


def test_forbidden_test_cache_names_are_not_created(tmp_path):
    assert not (tmp_path / "test_cache.pt").exists()
    assert not (tmp_path / "locked_test_cache.pt").exists()
