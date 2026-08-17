from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.sequential_costarts_end_to_end import (
    EXPERT_ORDER,
    EndToEndCOSTARTSConfig,
    FullEndToEndCOSTARTS,
    end_to_end_costarts_loss,
    gradient_report,
)


def test_full_end_to_end_costarts_gradients_reach_all_experts_and_router() -> None:
    torch.manual_seed(7)
    config = EndToEndCOSTARTSConfig(
        input_len=96,
        forecast_horizon=12,
        num_features=7,
        expert_hidden_size=8,
        router_embedding_dim=16,
        router_hidden_dim=16,
        max_queries=3,
        route_temperature=2.0,
        gumbel_routing=False,
    )
    model = FullEndToEndCOSTARTS(config)
    history = torch.randn(2, 96, 7)
    target = torch.randn(2, 12, 7)
    mask = torch.ones_like(target, dtype=torch.bool)
    outputs = model.forward_soft(history, temperature=2.0)
    losses = end_to_end_costarts_loss(
        outputs,
        target,
        mask,
        alpha_expert=0.5,
        lambda_query=0.01,
        lambda_balance=0.05,
        lambda_stop=0.1,
        query_cost=0.0,
    )
    losses["total_loss"].backward()
    report = gradient_report(model)
    for name in (*EXPERT_ORDER, "COSTAR_router"):
        assert report[name] > 0.0, f"{name} did not receive a nonzero gradient: {report}"


if __name__ == "__main__":
    test_full_end_to_end_costarts_gradients_reach_all_experts_and_router()
    print("end-to-end gradient test passed")
