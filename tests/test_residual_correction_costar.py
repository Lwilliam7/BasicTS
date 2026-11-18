import pytest
import torch

from experiments.residual_correction_costar.run_residual_correction_experiments import (
    BiasConfig,
    assert_history_available,
    causal_bias_correct,
    history_summary_for_window,
)


def test_bias_corrector_waits_for_complete_target_window() -> None:
    starts = torch.tensor([0, 5, 10, 12], dtype=torch.long)
    baseline = torch.zeros(4, 2, 1)
    target = torch.ones(4, 2, 1)
    config = BiasConfig(structure="hv", decay=0.0, alpha=1.0, clip_multiple=None, min_count=1)
    pred, extra = causal_bias_correct(
        starts=starts,
        baseline=baseline,
        target=target,
        horizon=12,
        config=config,
        init_residuals=None,
        clip_std=torch.ones(2, 1),
    )
    assert extra["num_updates"] == 1
    assert torch.allclose(pred[0], torch.zeros(2, 1))
    assert torch.allclose(pred[1], torch.zeros(2, 1))
    assert torch.allclose(pred[2], torch.zeros(2, 1))
    assert torch.allclose(pred[3], torch.ones(2, 1))


def test_bias_corrector_warmup_blocks_initial_correction() -> None:
    starts = torch.tensor([0, 12], dtype=torch.long)
    baseline = torch.zeros(2, 1, 1)
    target = torch.ones(2, 1, 1)
    config = BiasConfig(structure="global", decay=0.0, alpha=1.0, clip_multiple=None, min_count=2)
    pred, extra = causal_bias_correct(
        starts=starts,
        baseline=baseline,
        target=target,
        horizon=12,
        config=config,
        init_residuals=None,
        clip_std=torch.ones(1, 1),
    )
    assert extra["num_updates"] == 1
    assert torch.allclose(pred, torch.zeros_like(pred))


def test_history_summary_requires_past_only_window() -> None:
    series = torch.arange(20, dtype=torch.float32).view(20, 1)
    with pytest.raises(RuntimeError):
        assert_history_available(series, start=5, scale=6)
    vals = history_summary_for_window(series, start=10, variable=0, scale=4)
    assert vals[2] == 9.0
    assert vals[0] == pytest.approx(7.5)
