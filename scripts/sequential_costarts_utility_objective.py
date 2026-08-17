"""Utility-target helpers for Sequential COSTAR-TS routing experiments."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from scripts.train_sequential_costarts_full_walkforward import current_average_from_ids, sample_mae


def compute_marginal_utilities(
    prediction_stack: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    queried_ids: torch.Tensor,
) -> torch.Tensor:
    """Return state-dependent marginal utility for each expert.

    For non-empty queried sets, utility is the reduction in MAE from adding the
    expert to the current equal-average ensemble. For the initial empty state,
    utility is ``-MAE(expert_j)``, which ranks first-query candidates by their
    standalone forecast loss without inventing an arbitrary zero forecast.
    """

    batch, _, _, num_experts = prediction_stack.shape
    current_count = (queried_ids >= 0).sum(dim=1)
    has_current = current_count > 0
    current_prediction = current_average_from_ids(prediction_stack, queried_ids)
    current_loss = sample_mae(current_prediction, targets, masks)
    utilities = []
    for expert_id in range(num_experts):
        next_ids = queried_ids.clone()
        insert_slot = current_count.clamp(max=queried_ids.shape[1] - 1)
        next_ids[torch.arange(batch, device=prediction_stack.device), insert_slot] = expert_id
        candidate_prediction = current_average_from_ids(prediction_stack, next_ids)
        candidate_loss = sample_mae(candidate_prediction, targets, masks)
        utility = torch.where(has_current, current_loss - candidate_loss, -candidate_loss)
        utilities.append(utility)
    return torch.stack(utilities, dim=1).detach()


def available_expert_mask(queried_mask: torch.Tensor) -> torch.Tensor:
    return ~queried_mask.to(torch.bool)


def masked_utility_targets(
    utilities: torch.Tensor,
    available_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    masked = utilities.masked_fill(~available_mask, -1e9)
    return torch.softmax(masked / float(temperature), dim=1).detach()


def utility_listwise_loss(
    scores: torch.Tensor,
    utilities: torch.Tensor,
    available_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    targets = masked_utility_targets(utilities, available_mask, temperature)
    log_probs = F.log_softmax(scores.masked_fill(~available_mask, -1e9), dim=1)
    return -(targets * log_probs).sum(dim=1).mean()


def utility_listwise_stop_loss(
    scores: torch.Tensor,
    utilities: torch.Tensor,
    available_mask: torch.Tensor,
    stop_available: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Listwise utility loss over remaining experts plus a STOP action.

    STOP has fixed utility 0 and fixed logit 0. It is normally unavailable for
    the initial state, where the first expert query remains forced.
    """

    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    stop_available = stop_available.to(torch.bool).view(-1, 1)
    expert_utilities = utilities.masked_fill(~available_mask, -1e9)
    expert_scores = scores.masked_fill(~available_mask, -1e9)
    stop_utility = torch.zeros((scores.shape[0], 1), dtype=utilities.dtype, device=utilities.device)
    stop_score = torch.zeros((scores.shape[0], 1), dtype=scores.dtype, device=scores.device)
    stop_utility = stop_utility.masked_fill(~stop_available, -1e9)
    stop_score = stop_score.masked_fill(~stop_available, -1e9)
    action_utilities = torch.cat((expert_utilities, stop_utility), dim=1)
    action_scores = torch.cat((expert_scores, stop_score), dim=1)
    targets = torch.softmax(action_utilities / float(temperature), dim=1).detach()
    log_probs = F.log_softmax(action_scores, dim=1)
    return -(targets * log_probs).sum(dim=1).mean()


def utility_pairwise_loss(
    scores: torch.Tensor,
    utilities: torch.Tensor,
    available_mask: torch.Tensor,
    min_utility_diff: float = 0.0,
) -> torch.Tensor:
    score_diff = scores[:, :, None] - scores[:, None, :]
    utility_diff = utilities[:, :, None] - utilities[:, None, :]
    pair_mask = available_mask[:, :, None] & available_mask[:, None, :]
    pair_mask = pair_mask & (utility_diff > float(min_utility_diff))
    if not bool(pair_mask.any()):
        return scores.sum() * 0.0
    losses = F.softplus(-score_diff[pair_mask])
    return losses.mean()


def utility_weighted_pairwise_loss(
    scores: torch.Tensor,
    utilities: torch.Tensor,
    available_mask: torch.Tensor,
    min_utility_diff: float = 0.0,
) -> torch.Tensor:
    score_diff = scores[:, :, None] - scores[:, None, :]
    utility_diff = utilities[:, :, None] - utilities[:, None, :]
    pair_mask = available_mask[:, :, None] & available_mask[:, None, :]
    pair_mask = pair_mask & (utility_diff > float(min_utility_diff))
    if not bool(pair_mask.any()):
        return scores.sum() * 0.0
    weights = utility_diff[pair_mask].detach()
    losses = F.softplus(-score_diff[pair_mask]) * weights
    return losses.sum() / weights.sum().clamp_min(1e-12)


def stop_calibration_loss(
    scores: torch.Tensor,
    utilities: torch.Tensor,
    available_mask: torch.Tensor,
) -> torch.Tensor:
    """Preserve old stop semantics by anchoring scores to utility values."""

    return F.smooth_l1_loss(
        scores.masked_select(available_mask),
        utilities.masked_select(available_mask),
    )


def utility_regret(
    scores: torch.Tensor,
    utilities: torch.Tensor,
    available_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    masked_scores = scores.masked_fill(~available_mask, -1e9)
    masked_utilities = utilities.masked_fill(~available_mask, -1e9)
    selected = masked_scores.argmax(dim=1)
    selected_utility = masked_utilities.gather(1, selected[:, None]).squeeze(1)
    best_values, best_ids = masked_utilities.max(dim=1)
    top2_ids = torch.topk(masked_scores, k=min(2, scores.shape[1]), dim=1).indices
    top2_hit = (top2_ids == best_ids[:, None]).any(dim=1)
    regret = best_values - selected_utility
    return {
        "selected": selected,
        "best": best_ids,
        "selected_utility": selected_utility,
        "best_utility": best_values,
        "regret": regret,
        "top1_hit": selected == best_ids,
        "top2_hit": top2_hit,
        "positive_selected": selected_utility > 0,
        "any_positive": (masked_utilities > 0).any(dim=1),
    }


def utility_statistics(values: torch.Tensor, near_zero_epsilon: float = 1e-3) -> dict[str, Any]:
    flat = values.detach().flatten().to(torch.float64)
    if flat.numel() == 0:
        return {}
    quantiles = torch.quantile(flat, torch.tensor([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95], dtype=torch.float64))
    return {
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "median": float(quantiles[3].item()),
        "p05": float(quantiles[0].item()),
        "p10": float(quantiles[1].item()),
        "p25": float(quantiles[2].item()),
        "p75": float(quantiles[4].item()),
        "p90": float(quantiles[5].item()),
        "p95": float(quantiles[6].item()),
        "fraction_positive": float((flat > 0).to(torch.float64).mean().item() * 100.0),
        "fraction_near_zero": float((flat.abs() <= near_zero_epsilon).to(torch.float64).mean().item() * 100.0),
    }
