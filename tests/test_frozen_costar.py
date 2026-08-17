from __future__ import annotations

import torch

from experiments.frozen_costar.run_frozen_costar_validation import (
    frozen_costar_prediction,
    frozen_hv_weights,
    online_prediction,
    tensor_digest,
)


def _cache(num_windows: int, offset: int = 0) -> dict[str, torch.Tensor | int | list[str]]:
    gen = torch.Generator().manual_seed(1000 + offset)
    horizon = 2
    variables = 2
    experts = ["DLinear", "PatchTST", "ModernTCN"]
    return {
        "histories": torch.randn(num_windows, 4, variables, generator=gen),
        "targets": torch.randn(num_windows, horizon, variables, generator=gen),
        "target_masks": torch.ones(num_windows, horizon, variables, dtype=torch.bool),
        "prediction_stack": torch.randn(num_windows, horizon, variables, len(experts), generator=gen),
        "absolute_window_starts": torch.arange(offset, offset + num_windows),
        "expert_names": experts,
        "num_windows": num_windows,
        "forecast_horizon": horizon,
        "input_len": 4,
        "num_features": variables,
    }


def test_frozen_hv_weights_repeat_train_initialized_weights() -> None:
    train_err_mean = torch.tensor(
        [
            [[0.4, 0.5, 0.6], [0.8, 0.7, 0.9]],
            [[0.3, 0.35, 0.45], [0.6, 0.55, 0.5]],
        ],
        dtype=torch.float32,
    )
    weights, extra = frozen_hv_weights(5, train_err_mean)
    assert weights.shape == (5, 2, 2, 3)
    assert extra["hv_num_updates"] == 0
    for i in range(1, 5):
        assert torch.equal(weights[0], weights[i])


def test_frozen_etth2_prediction_ignores_eval_targets_and_masks() -> None:
    train_cache = _cache(7, 0)
    val_cache = _cache(8, 100)
    std = torch.ones(2)
    idx = [0, 1, 2]
    pred, _ = frozen_costar_prediction("ETTh2", val_cache, train_cache, std, idx, 7, torch.device("cpu"))

    target_mut = dict(val_cache)
    target_mut["targets"] = torch.randn_like(val_cache["targets"])
    pred_target, _ = frozen_costar_prediction("ETTh2", target_mut, train_cache, std, idx, 7, torch.device("cpu"))
    assert torch.equal(pred, pred_target)

    mask_mut = dict(val_cache)
    mask_mut["target_masks"] = torch.zeros_like(val_cache["target_masks"], dtype=torch.bool)
    pred_mask, _ = frozen_costar_prediction("ETTh2", mask_mut, train_cache, std, idx, 7, torch.device("cpu"))
    assert torch.equal(pred, pred_mask)


def test_frozen_etth2_prediction_does_not_mutate_cache_tensors() -> None:
    train_cache = _cache(7, 0)
    val_cache = _cache(8, 100)
    before = tensor_digest(val_cache)
    _pred, _ = frozen_costar_prediction("ETTh2", val_cache, train_cache, torch.ones(2), [0, 1, 2], 7, torch.device("cpu"))
    after = tensor_digest(val_cache)
    assert before == after


def test_frozen_and_online_begin_with_same_first_window() -> None:
    train_cache = _cache(7, 0)
    val_cache = _cache(8, 100)
    std = torch.ones(2)
    idx = [0, 1, 2]
    frozen, _ = frozen_costar_prediction("ETTh2", val_cache, train_cache, std, idx, 7, torch.device("cpu"))
    online, _ = online_prediction("ETTh2", val_cache, train_cache, std, idx, 7, torch.device("cpu"))
    assert torch.equal(frozen[0], online[0])


def test_online_etth2_path_still_uses_eval_targets_after_delay() -> None:
    train_cache = _cache(7, 0)
    val_cache = _cache(16, 100)
    std = torch.ones(2)
    idx = [0, 1, 2]
    online, _ = online_prediction("ETTh2", val_cache, train_cache, std, idx, 7, torch.device("cpu"))
    target_mut = dict(val_cache)
    target_mut["targets"] = val_cache["targets"] + 10.0
    online_mut, _ = online_prediction("ETTh2", target_mut, train_cache, std, idx, 7, torch.device("cpu"))
    assert not torch.equal(online, online_mut)
