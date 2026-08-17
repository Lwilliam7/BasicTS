import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.old.train_costarts_adaptive_router import (
    COSTARTSAdaptiveRouter,
    _equal_average_from_state,
    masked_action_logits,
    rollout_adaptive_router,
)


def _tiny_router(variant: str = "forecast_disagreement") -> COSTARTSAdaptiveRouter:
    torch.manual_seed(3)
    return COSTARTSAdaptiveRouter(
        num_experts=3,
        input_len=4,
        forecast_horizon=2,
        num_features=2,
        embedding_dim=8,
        hidden_dim=8,
        variant=variant,
    )


def _state_inputs(num_experts: int = 3):
    history = torch.randn(2, 4, 2)
    queried_mask = torch.tensor([[True, False, False], [True, True, False]])
    queried_ids = torch.tensor([[0, -1, -1], [0, 1, -1]])
    forecasts = torch.randn(2, num_experts, 2, 2)
    return history, queried_mask, queried_ids, forecasts


def _tiny_cache() -> dict:
    num_windows = 2
    num_experts = 3
    histories = torch.randn(num_windows, 4, 2)
    targets = torch.zeros(num_windows, 2, 2)
    expert_forecasts = torch.randn(num_windows, num_experts, 2, 2)
    rows = []
    for source_row in range(num_windows):
        for mask_value in range(1 << num_experts):
            queried = [idx for idx in range(num_experts) if mask_value & (1 << idx)]
            queried_ids = torch.full((num_experts,), -1, dtype=torch.long)
            queried_forecasts = torch.zeros(num_experts, 2, 2)
            queried_mask = torch.zeros(num_experts, dtype=torch.bool)
            for slot, expert_index in enumerate(queried):
                queried_ids[slot] = expert_index
                queried_forecasts[slot] = expert_forecasts[source_row, expert_index]
                queried_mask[expert_index] = True
            valid_action_mask = torch.cat((~queried_mask, torch.tensor([len(queried) > 0])))
            rows.append(
                {
                    "source_row": source_row,
                    "history": histories[source_row],
                    "queried_mask": queried_mask,
                    "queried_expert_ids": queried_ids,
                    "queried_expert_forecasts": queried_forecasts,
                    "true_targets": targets[source_row],
                    "target_mask": torch.ones(2, 2, dtype=torch.bool),
                    "true_expert_error_vector": torch.rand(num_experts),
                    "remaining_mask": ~queried_mask,
                    "marginal_gain_best_queried_oracle": torch.zeros(num_experts),
                    "marginal_gain_equal_queried_average": torch.zeros(num_experts),
                    "valid_action_mask": valid_action_mask,
                    "subset_size": torch.tensor(len(queried), dtype=torch.long),
                }
            )
    return {
        "num_source_windows": num_windows,
        "num_states": len(rows),
        "num_experts": num_experts,
        "expert_names": ("A", "B", "C"),
        **{
            key: torch.stack([row[key] for row in rows])
            for key in rows[0]
            if key != "source_row"
        },
        "source_row": torch.tensor([row["source_row"] for row in rows], dtype=torch.long),
    }


def test_queried_experts_are_masked_from_future_selection():
    logits = torch.tensor([[10.0, 9.0, 1.0, 0.0]])
    valid = torch.tensor([[False, True, True, True]])
    masked = masked_action_logits(logits, valid)
    assert masked[0, 0] < -1e8
    assert int(torch.argmax(masked, dim=1)) == 1


def test_stop_action_can_remain_available():
    logits = torch.zeros(1, 4)
    valid = torch.tensor([[False, True, True, True]])
    masked = masked_action_logits(logits, valid)
    assert masked[0, 3] == 0


def test_router_state_changes_after_forecast_is_added():
    router = _tiny_router()
    history, queried_mask, queried_ids, forecasts = _state_inputs()
    before = router(history[:1], queried_mask[:1], queried_ids[:1], forecasts[:1])["action_logits"]
    queried_mask_after = torch.tensor([[True, True, False]])
    queried_ids_after = torch.tensor([[0, 1, -1]])
    after = router(history[:1], queried_mask_after, queried_ids_after, forecasts[:1])["action_logits"]
    assert not torch.allclose(before, after)


def test_unqueried_forecasts_do_not_influence_logits():
    router = _tiny_router()
    history, queried_mask, queried_ids, forecasts = _state_inputs()
    changed = forecasts.clone()
    changed[:, 1:] = torch.randn_like(changed[:, 1:]) * 100.0
    original_logits = router(history[:1], queried_mask[:1], queried_ids[:1], forecasts[:1])["action_logits"]
    changed_logits = router(history[:1], queried_mask[:1], queried_ids[:1], changed[:1])["action_logits"]
    assert torch.allclose(original_logits, changed_logits, atol=1e-5)


def test_equal_aggregation_uses_exactly_queried_experts():
    batch = {
        "queried_expert_ids": torch.tensor([[0, 2, -1]]),
        "queried_expert_forecasts": torch.tensor([[[[1.0]], [[3.0]], [[100.0]]]]),
    }
    assert torch.equal(_equal_average_from_state(batch), torch.tensor([[[2.0]]]))


def test_rollout_terminates_and_respects_cap():
    router = _tiny_router("mask_only")
    cache = _tiny_cache()
    metrics = rollout_adaptive_router(
        router,
        cache,
        batch_size=4,
        device=torch.device("cpu"),
        max_queries=2,
        forced_budget=2,
    )
    assert metrics["average_experts_queried"] <= 2.0
    assert metrics["all_queries_unique"]


def test_optimizer_contains_only_router_parameters():
    router = _tiny_router()
    optimizer = torch.optim.AdamW(router.parameters(), lr=1e-3)
    router_params = {id(parameter) for parameter in router.parameters()}
    opt_params = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert opt_params == router_params
