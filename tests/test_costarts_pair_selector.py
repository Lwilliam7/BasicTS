import inspect

import torch

from scripts.train_costarts_pair_selector import (
    CostartsPairDataset,
    CostartsPairSelector,
    build_pair_index,
    evaluate_pair_selector,
    masked_pair_mae_mse,
    pair_errors_to_soft_targets,
    pair_to_class_index,
)


def _tiny_cache() -> dict:
    histories = torch.zeros(2, 96, 7)
    targets = torch.tensor([[[0.0]], [[10.0]]])
    target_masks = torch.ones(2, 1, 1, dtype=torch.bool)
    prediction_stack = torch.tensor(
        [
            [[[0.0, 2.0, 8.0]]],
            [[[8.0, 12.0, 20.0]]],
        ],
        dtype=torch.float32,
    )
    error_matrix = (prediction_stack.squeeze(1).squeeze(1) - targets.view(2, 1)).abs()
    return {
        "split_role": "router_val",
        "histories": histories,
        "targets": targets,
        "target_masks": target_masks,
        "prediction_stack": prediction_stack,
        "error_matrix": error_matrix,
        "mse_matrix": error_matrix.pow(2),
        "best_expert": torch.argmin(error_matrix, dim=1),
        "sample_indices": torch.arange(2),
        "expert_names": ("a", "b", "c"),
        "num_windows": 2,
        "input_len": 96,
        "forecast_horizon": 1,
        "num_features": 1,
    }


def test_pair_index_mapping_is_stable_and_unordered():
    pair_index = build_pair_index(4)
    assert pair_index.tolist() == [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    mapping = pair_to_class_index(pair_index)
    assert mapping[(0, 1)] == 0
    assert mapping[(1, 3)] == 4
    assert mapping[(2, 3)] == 5


def test_pair_label_construction_uses_equal_average_pair_error():
    cache = _tiny_cache()
    pair_index = build_pair_index(3)
    pair_mae, pair_mse = masked_pair_mae_mse(cache, pair_index)

    assert torch.equal(pair_index, torch.tensor([[0, 1], [0, 2], [1, 2]]))
    assert torch.allclose(pair_mae[0], torch.tensor([1.0, 4.0, 5.0]))
    assert torch.allclose(pair_mse[0], torch.tensor([1.0, 16.0, 25.0]))
    assert int(torch.argmin(pair_mae[0])) == 0
    assert int(torch.argmin(pair_mae[1])) == 0

    soft_targets = pair_errors_to_soft_targets(pair_mae, temperature=0.5)
    assert torch.allclose(soft_targets.sum(dim=1), torch.ones(2))
    assert int(torch.argmax(soft_targets[0])) == 0
    assert int(torch.argmax(soft_targets[1])) == 0
    assert float(soft_targets[0, 0]) > float(soft_targets[0, 1]) > float(soft_targets[0, 2])


def test_dataset_and_model_use_causal_history_only_as_input():
    cache = _tiny_cache()
    pair_index = build_pair_index(3)
    dataset = CostartsPairDataset(cache, pair_index, target_temperature=0.5)
    item = dataset[0]

    assert set(item.keys()) == {"history", "soft_targets", "best_pair"}
    assert tuple(item["history"].shape) == (96, 7)

    signature = inspect.signature(CostartsPairSelector.forward)
    assert list(signature.parameters) == ["self", "history"]


def test_inference_selects_one_pair_and_equal_averages_its_forecasts():
    class DummyPairSelector(torch.nn.Module):
        def forward(self, history):
            logits = torch.zeros(history.shape[0], 3)
            logits[:, 2] = 10.0
            return logits

    cache = _tiny_cache()
    pair_index = build_pair_index(3)
    metrics = evaluate_pair_selector(
        DummyPairSelector(),
        cache,
        pair_index,
        batch_size=2,
        device=torch.device("cpu"),
    )

    assert metrics["average_experts_queried"] == 2.0
    assert metrics["selected_pair_class"].tolist() == [2, 2]
    assert metrics["selected_pair_indices"].tolist() == [[1, 2], [1, 2]]
    assert metrics["mae"] == 5.5
    assert metrics["best_pair_accuracy"] == 0.0
    assert metrics["best_individual_expert_top2_coverage"] == 0.0
