import pytest
import torch

from experiments.chronological_adaptive_costar.run_chronological_adaptive_costar import enforce_observable
from experiments.expanded_expert_pool_costar.run_expanded_expert_pool import (
    Config,
    run_causal_specialists,
    weights_from_advantage,
)


def test_observable_rule_rejects_early_update() -> None:
    with pytest.raises(RuntimeError):
        enforce_observable(due_start=0, current_start=11, horizon=12)
    enforce_observable(due_start=0, current_start=12, horizon=12)


def test_optional_weights_are_convex_and_capped() -> None:
    cfg = Config("both", "hv", decay=0.97, extra_weight_cap=0.05, activation_margin=0.01, warmup=24)
    adv_d = torch.full((12, 7), 0.50)
    adv_m = torch.full((12, 7), 0.50)
    w_d, w_m = weights_from_advantage(adv_d, adv_m, cfg)
    assert torch.all(w_d >= 0)
    assert torch.all(w_m >= 0)
    assert torch.all(w_d + w_m <= 0.050001)


def test_causal_specialist_does_not_update_before_full_horizon() -> None:
    starts = torch.tensor([0, 5, 10, 12], dtype=torch.long)
    base = torch.ones(4, 2, 1)
    dlin = torch.zeros(4, 2, 1)
    modern = torch.ones(4, 2, 1)
    target = torch.zeros(4, 2, 1)
    mask = torch.ones(4, 2, 1, dtype=torch.bool)
    std = torch.ones(1)
    init_bad = torch.full((1, 2, 1), 10.0)
    init_good = torch.full((1, 2, 1), 0.1)
    cfg = Config("dlinear_only", "hv", decay=0.0, extra_weight_cap=0.10, activation_margin=0.0, warmup=1)
    pred, extra, traces = run_causal_specialists(
        starts,
        base,
        dlin,
        modern,
        target,
        mask,
        std,
        cfg,
        init_bad,
        init_good,
        init_bad,
    )
    assert extra["num_updates"] == 1
    assert traces[0]["completed_count"] == 1
    assert traces[1]["completed_count"] == 1
    assert traces[2]["completed_count"] == 1
    assert traces[3]["completed_count"] == 2
    assert torch.all(pred <= base)
