import pytest
import torch

from experiments.chronological_adaptive_costar.run_chronological_adaptive_costar import enforce_observable
from experiments.residual_covariance_weighting_costar.run_residual_covariance_weighting import (
    CovConfig,
    expand_weights,
    optimal_cov_weights,
    simplex_grid,
)


def test_horizon_delayed_release_rejects_forbidden_update() -> None:
    with pytest.raises(RuntimeError):
        enforce_observable(due_start=100, current_start=111, horizon=12)
    enforce_observable(due_start=100, current_start=112, horizon=12)


def test_covariance_weights_are_convex_and_fallback_when_under_warmup() -> None:
    cfg = CovConfig(
        family="full_covariance",
        structure="hv",
        decay=0.97,
        ridge=1e-4,
        shrink_diag=0.5,
        shrink_global=0.25,
        bias_weight=1.0,
        hybrid_alpha=0.5,
        min_count=96,
    )
    mean = torch.zeros(4, 3)
    second = torch.eye(3).repeat(4, 1, 1)
    fallback = torch.tensor([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    weights, diag = optimal_cov_weights(
        mean=mean,
        second=second,
        global_mean=mean[:1],
        global_second=second[:1],
        config=cfg,
        grid=simplex_grid(0.05),
        fallback=fallback,
        count=0,
    )
    assert diag["fallback_groups"] == 4
    assert torch.allclose(weights, fallback.view(1, 3).expand_as(weights))
    expanded = expand_weights(weights, "hv", horizon=2, variables=2)
    assert expanded.shape == (2, 2, 3)
    assert bool(torch.all(expanded >= 0))
    assert torch.allclose(expanded.sum(dim=-1), torch.ones(2, 2))
