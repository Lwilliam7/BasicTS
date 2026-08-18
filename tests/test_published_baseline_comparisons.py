from __future__ import annotations

from pathlib import Path

import torch

from experiments.published_baseline_comparisons import run_published_baselines as runner


def _toy_cache(num_windows: int = 24, offset: int = 100) -> dict:
    gen = torch.Generator().manual_seed(123 + offset)
    return {
        "cache_role": "router_train_20_60",
        "split_role": "router_train_20_60",
        "histories": torch.randn(num_windows, 96, 7, generator=gen),
        "targets": torch.randn(num_windows, 12, 7, generator=gen),
        "target_masks": torch.ones(num_windows, 12, 7, dtype=torch.bool),
        "prediction_stack": torch.randn(num_windows, 12, 7, 5, generator=gen),
        "expert_names": runner.EXPERTS,
        "absolute_window_starts": torch.arange(offset, offset + num_windows),
        "forecast_horizon": 12,
        "input_len": 96,
        "num_features": 7,
        "num_windows": num_windows,
    }


def test_refuse_test_path_blocks_test_names() -> None:
    try:
        runner.refuse_test(Path("cache") / "router_test_cache.pt")
    except ValueError:
        return
    raise AssertionError("test path was not rejected")


def test_validate_cache_accepts_expected_costar_shape() -> None:
    runner.validate_cache(_toy_cache(), "router_train_20_60", "toy")


def test_validate_cache_rejects_wrong_expert_order() -> None:
    cache = _toy_cache()
    cache["expert_names"] = ("PatchTST", "DLinear", "iTransformer", "TimesNet", "ModernTCN")
    try:
        runner.validate_cache(cache, "router_train_20_60", "toy")
    except ValueError:
        return
    raise AssertionError("expert ordering mismatch did not raise")


def test_granger_ramanathan_global_prediction_shape() -> None:
    cache = _toy_cache()
    std = torch.ones(7)
    cfg = runner.LinearConfig("Granger-Ramanathan", "global", 0.0, "canonical_ols")
    model = runner.fit_gr(cache, std, cfg)
    pred = runner.predict_gr(model, cache)
    assert pred.shape == cache["targets"].shape


def test_bates_granger_weights_sum_to_one() -> None:
    cache = _toy_cache()
    std = torch.ones(7)
    cfg = runner.BatesConfig("global", "covariance", 0.25)
    model = runner.fit_bates(cache, std, cfg)
    assert torch.allclose(model["weights"].sum(), torch.tensor(1.0), atol=1e-5)


def test_onenet_delayed_updates_obey_horizon_rule() -> None:
    cache = _toy_cache(num_windows=20)
    std = torch.ones(7)
    cfg = runner.OneNetConfig(eta=0.1, decay=0.97)
    init = torch.ones(2)
    _pred, extra = runner.onenet_predict(cache, std, cfg, init, runner.ONENET_BRANCHES)
    assert extra["num_updates"] == 8
