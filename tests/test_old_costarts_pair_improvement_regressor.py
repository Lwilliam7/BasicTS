import pytest
import torch

from scripts.costars import train_old_costarts_pair_improvement_regressor as regressor


def _toy_cache(split="router_train", n=4):
    targets = torch.zeros(n, 12, 7)
    mask = torch.ones_like(targets, dtype=torch.bool)
    # Pair (0, 1) error = 1.5, pair (0, 2) error = 2.0, pair (3, 4) error = 4.5.
    stack = torch.stack(
        [torch.full_like(targets, float(value)) for value in (1, 2, 3, 4, 5)],
        dim=-1,
    )
    return {
        "split_role": split,
        "expert_names": regressor.EXPECTED_EXPERTS,
        "input_len": 96,
        "forecast_horizon": 12,
        "num_features": 7,
        "num_windows": n,
        "histories": torch.zeros(n, 96, 7),
        "targets": targets,
        "target_masks": mask,
        "prediction_stack": stack,
        "sample_indices": torch.arange(n),
        "absolute_window_starts": torch.arange(n),
    }


def test_pair_improvement_targets_have_correct_signs():
    cache = _toy_cache()
    pairs = regressor.pair_class_order()
    pair_mae, _ = regressor.pair_error_matrices(cache, pairs)
    fixed_index = regressor.pair_name_to_index("DLinear+iTransformer", pairs)
    better_index = regressor.pair_name_to_index("DLinear+PatchTST", pairs)
    worse_index = regressor.pair_name_to_index("TimesNet+ModernTCN", pairs)
    targets = regressor.pair_improvement_targets(pair_mae, fixed_index)

    assert torch.allclose(targets[:, fixed_index], torch.zeros(cache["num_windows"]))
    assert torch.all(targets[:, better_index] > 0)
    assert torch.all(targets[:, worse_index] < 0)


def test_dataset_is_history_only_and_train_only():
    cache = _toy_cache("router_train")
    pairs = regressor.pair_class_order()
    pair_mae, _ = regressor.pair_error_matrices(cache, pairs)
    dataset = regressor.PairImprovementDataset(cache, pair_mae, fixed_pair_index=0)
    item = dataset[0]

    assert set(item) == {"history", "improvement_target", "source_index"}
    assert item["history"].shape == (96, 7)
    assert "targets" not in item
    assert "prediction_stack" not in item

    val = _toy_cache("router_val")
    with pytest.raises(ValueError, match="router_train"):
        regressor.PairImprovementDataset(val, pair_mae, fixed_pair_index=0)


def test_threshold_selection_uses_supplied_validation_errors_only():
    cache = _toy_cache("router_val")
    pairs = regressor.pair_class_order()
    pair_mae, pair_mse = regressor.pair_error_matrices(cache, pairs)
    fixed_index = regressor.pair_name_to_index("DLinear+iTransformer", pairs)
    better_index = regressor.pair_name_to_index("DLinear+PatchTST", pairs)
    scores = torch.full_like(pair_mae, -1.0)
    scores[:, better_index] = 0.5

    threshold, rows, metrics = regressor.select_threshold_on_validation(scores, pair_mae, pair_mse, fixed_index)

    assert rows
    assert threshold < 0.5
    assert metrics["selected_pair_mae"] < float(pair_mae[:, fixed_index].mean().item())
    assert metrics["switch_rate"] == pytest.approx(100.0)


def test_pair_indexing_is_consistent_between_targets_and_inference():
    cache = _toy_cache("router_val")
    pairs = regressor.pair_class_order()
    pair_mae, pair_mse = regressor.pair_error_matrices(cache, pairs)
    fixed_index = regressor.pair_name_to_index("DLinear+iTransformer", pairs)
    best_index = regressor.pair_name_to_index("DLinear+PatchTST", pairs)
    targets = regressor.pair_improvement_targets(pair_mae, fixed_index)
    scores = torch.zeros_like(targets)
    scores[:, best_index] = targets[:, best_index] + 1.0
    metrics = regressor.evaluate_scores(scores, pair_mae, pair_mse, fixed_index, threshold=0.0)

    assert torch.equal(metrics["selected_pair_indices"], torch.full((cache["num_windows"],), best_index))
    assert metrics["improvement_over_fixed_pair"] > 0


def test_no_final_test_cache_paths_are_created(tmp_path):
    assert not (tmp_path / "costarts_router_test_cache.pt").exists()
    assert not (tmp_path / "costarts_locked_test_cache.pt").exists()


def test_validation_fixed_pair_selection_chooses_lowest_validation_mae():
    cache = _toy_cache("router_val")
    pairs = regressor.pair_class_order()
    pair_mae, _ = regressor.pair_error_matrices(cache, pairs)
    index, name = regressor.select_validation_fixed_pair(pair_mae, pairs)

    assert index == regressor.pair_name_to_index("DLinear+PatchTST", pairs)
    assert name == "DLinear+PatchTST"
