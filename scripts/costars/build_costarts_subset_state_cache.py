"""Build exhaustive subset-state supervision for COSTARTS-style routers.

This phase-one cache expands each cached router window into many reachable
states. A state is defined by the subset of experts that has already been
queried. No forecasting experts are run here; the script reuses the offline
prediction stack stored by ``scripts/train_costarts_router.py``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import torch

try:
    from scripts.router_experiment_config import load_router_experiment_config
except ImportError:
    from router_experiment_config import load_router_experiment_config


DEFAULT_SOURCE_CACHE = "cache/costarts_router_train_cache.pt"
DEFAULT_OUTPUT_CACHE = "cache/costarts_subset_state_train_cache.pt"
STOP_ACTION_NAME = "__STOP__"


@dataclass(frozen=True)
class SubsetStateBuildConfig:
    source_cache_path: Union[str, Path] = DEFAULT_SOURCE_CACHE
    output_cache_path: Union[str, Path] = DEFAULT_OUTPUT_CACHE
    subset_cap: Optional[int] = None
    action_temperature: float = 0.1
    stop_improvement_threshold: float = 0.0
    force_stop_at_cap: bool = True
    force_rebuild: bool = False
    debug: bool = False


def _load_torch_checkpoint(path: Union[str, Path]) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _cost_tensor(expert_names: Sequence[str], cost_weights: Mapping[str, float]) -> torch.Tensor:
    return torch.tensor(
        [float(cost_weights.get(name, 0.0)) for name in expert_names],
        dtype=torch.float32,
    )


def _subset_masks(num_experts: int, subset_cap: int) -> torch.Tensor:
    masks = []
    for subset_size in range(0, subset_cap + 1):
        for indices in combinations(range(num_experts), subset_size):
            mask = torch.zeros(num_experts, dtype=torch.bool)
            if indices:
                mask[list(indices)] = True
            masks.append(mask)
    return torch.stack(masks, dim=0)


def validate_costarts_subset_state_cache(cache: Mapping[str, object]) -> None:
    """Validate tensor contracts for a generated subset-state cache."""

    expert_names = tuple(cache["expert_names"])
    num_experts = len(expert_names)
    stop_action_index = int(cache["stop_action_index"])
    num_states = int(cache["num_states"])
    horizon = int(cache["forecast_horizon"])
    num_features = int(cache["num_features"])

    assert stop_action_index == num_experts
    assert tuple(cache["state_sample_indices"].shape) == (num_states,)
    assert tuple(cache["subset_sizes"].shape) == (num_states,)
    assert tuple(cache["queried_masks"].shape) == (num_states, num_experts)
    assert tuple(cache["queryable_expert_masks"].shape) == (num_states, num_experts)
    assert tuple(cache["valid_action_masks"].shape) == (num_states, num_experts + 1)
    assert tuple(cache["candidate_utilities"].shape) == (num_states, num_experts + 1)
    assert tuple(cache["action_probabilities"].shape) == (num_states, num_experts + 1)
    assert tuple(cache["optimal_action"].shape) == (num_states,)
    assert tuple(cache["current_best_error"].shape) == (num_states,)
    assert tuple(cache["current_best_expert"].shape) == (num_states,)
    assert tuple(cache["queried_forecasts"].shape) == (
        num_states,
        horizon,
        num_features,
        num_experts,
    )
    assert tuple(cache["candidate_pairwise_ordering_targets"].shape) == (
        num_states,
        num_experts,
        num_experts,
    )
    assert tuple(cache["queried_pairwise_ordering_targets"].shape) == (
        num_states,
        num_experts,
        num_experts,
    )
    if not torch.all(cache["valid_action_masks"].any(dim=-1)):
        raise AssertionError("Every subset state must have at least one valid action")
    if not torch.all(cache["optimal_action"].ge(0) & cache["optimal_action"].le(num_experts)):
        raise AssertionError("Optimal actions must be expert indices or STOP")
    chosen_valid = cache["valid_action_masks"].gather(
        1,
        cache["optimal_action"].view(-1, 1),
    )
    if not torch.all(chosen_valid):
        raise AssertionError("Every optimal action must be valid in its state")
    prob_sums = cache["action_probabilities"].sum(dim=-1)
    if not torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5):
        raise AssertionError("Action probabilities must sum to 1")


def build_costarts_subset_state_cache(
    source_cache: Mapping[str, object],
    *,
    source_cache_path: Union[str, Path],
    output_cache_path: Union[str, Path],
    subset_cap: Optional[int] = None,
    action_temperature: float = 0.1,
    stop_improvement_threshold: float = 0.0,
    cost_weights: Optional[Mapping[str, float]] = None,
    force_stop_at_cap: bool = True,
    debug: bool = False,
) -> dict:
    """Expand a COSTARTS expert cache into exhaustive queried-subset states."""

    if action_temperature <= 0:
        raise ValueError("action_temperature must be positive")
    if stop_improvement_threshold < 0:
        raise ValueError("stop_improvement_threshold must be non-negative")

    expert_names = tuple(source_cache["expert_names"])
    num_experts = len(expert_names)
    if num_experts <= 0:
        raise ValueError("source cache must contain at least one expert")

    subset_cap = num_experts if subset_cap is None else int(subset_cap)
    if subset_cap < 0 or subset_cap > num_experts:
        raise ValueError(f"subset_cap must be between 0 and {num_experts}")

    prediction_stack = source_cache["prediction_stack"].to(torch.float32)
    error_matrix = source_cache["error_matrix"].to(torch.float32)
    num_windows, forecast_horizon, num_features, prediction_experts = prediction_stack.shape
    if prediction_experts != num_experts:
        raise AssertionError("prediction_stack expert dimension does not match expert_names")
    if tuple(error_matrix.shape) != (num_windows, num_experts):
        raise AssertionError("error_matrix shape does not match prediction_stack")

    expert_costs = _cost_tensor(expert_names, cost_weights or {})
    stop_action_index = num_experts
    subset_masks = _subset_masks(num_experts, subset_cap)
    num_subsets = subset_masks.shape[0]
    num_states = num_windows * num_subsets

    state_sample_indices = []
    subset_sizes = []
    queried_masks = []
    queryable_expert_masks = []
    valid_action_masks = []
    candidate_utilities = []
    action_probabilities = []
    optimal_actions = []
    current_best_errors = []
    current_best_experts = []
    queried_forecasts = []
    candidate_pairwise_targets = []
    queried_pairwise_targets = []

    source_indices = torch.arange(num_windows, dtype=torch.long)
    finite_large = torch.finfo(torch.float32).max / 4
    neg_large = -finite_large

    for subset_index, subset_mask in enumerate(subset_masks):
        subset_size = int(subset_mask.sum().item())
        has_queried = subset_size > 0
        can_query_more = subset_size < subset_cap

        queried_error_matrix = error_matrix.masked_fill(~subset_mask.view(1, -1), finite_large)
        if has_queried:
            current_best_error, current_best_expert = queried_error_matrix.min(dim=-1)
        else:
            current_best_error = torch.full((num_windows,), finite_large, dtype=torch.float32)
            current_best_expert = torch.full((num_windows,), -1, dtype=torch.long)

        queryable_mask = (~subset_mask).clone()
        if force_stop_at_cap and not can_query_more:
            queryable_mask[:] = False

        valid_actions = torch.zeros(num_windows, num_experts + 1, dtype=torch.bool)
        valid_actions[:, :num_experts] = queryable_mask.view(1, -1)
        valid_actions[:, stop_action_index] = has_queried

        result_error = torch.minimum(current_best_error.view(-1, 1), error_matrix)
        expert_utilities = -result_error - expert_costs.view(1, -1)
        expert_utilities = expert_utilities.masked_fill(~queryable_mask.view(1, -1), neg_large)
        stop_utility = torch.full((num_windows,), neg_large, dtype=torch.float32)
        if has_queried:
            stop_utility = -current_best_error

        utilities = torch.cat((expert_utilities, stop_utility.view(-1, 1)), dim=-1)

        best_query_utility, best_query_action = expert_utilities.max(dim=-1)
        stop_allowed = valid_actions[:, stop_action_index]
        stop_close_enough = stop_utility >= (
            best_query_utility - float(stop_improvement_threshold)
        )
        optimal_action = torch.where(
            stop_allowed & stop_close_enough,
            torch.full_like(best_query_action, stop_action_index),
            best_query_action,
        )
        no_query_available = ~valid_actions[:, :num_experts].any(dim=-1)
        optimal_action = torch.where(
            no_query_available & stop_allowed,
            torch.full_like(optimal_action, stop_action_index),
            optimal_action,
        )

        masked_for_softmax = utilities.masked_fill(~valid_actions, neg_large)
        probabilities = torch.softmax(masked_for_softmax / float(action_temperature), dim=-1)
        probabilities = probabilities.masked_fill(~valid_actions, 0.0)
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        expert_diff = expert_utilities.unsqueeze(2) - expert_utilities.unsqueeze(1)
        candidate_pairwise = torch.sign(expert_diff).to(torch.int8)
        candidate_pairwise_valid = queryable_mask.view(1, -1, 1) & queryable_mask.view(1, 1, -1)
        candidate_pairwise = candidate_pairwise.masked_fill(~candidate_pairwise_valid, 0)

        queried_scores = -error_matrix
        queried_diff = queried_scores.unsqueeze(2) - queried_scores.unsqueeze(1)
        queried_pairwise = torch.sign(queried_diff).to(torch.int8)
        queried_pairwise_valid = subset_mask.view(1, -1, 1) & subset_mask.view(1, 1, -1)
        queried_pairwise = queried_pairwise.masked_fill(~queried_pairwise_valid, 0)

        state_sample_indices.append(source_indices)
        subset_sizes.append(torch.full((num_windows,), subset_size, dtype=torch.long))
        queried_masks.append(subset_mask.view(1, -1).expand(num_windows, -1))
        queryable_expert_masks.append(queryable_mask.view(1, -1).expand(num_windows, -1))
        valid_action_masks.append(valid_actions)
        candidate_utilities.append(utilities)
        action_probabilities.append(probabilities)
        optimal_actions.append(optimal_action)
        current_best_errors.append(current_best_error)
        current_best_experts.append(current_best_expert)
        queried_forecasts.append(prediction_stack * subset_mask.view(1, 1, 1, -1))
        candidate_pairwise_targets.append(candidate_pairwise)
        queried_pairwise_targets.append(queried_pairwise)

        if debug:
            print(
                f"subset {subset_index + 1:03d}/{num_subsets}: "
                f"size={subset_size}, states={num_windows}, "
                f"queryable={queryable_mask.tolist()}"
            )

    cache = {
        "cache_type": "costarts_subset_state",
        "source_cache_path": str(source_cache_path),
        "source_split_role": source_cache.get("split_role", "unknown"),
        "output_cache_path": str(output_cache_path),
        "expert_names": expert_names,
        "action_names": tuple(expert_names) + (STOP_ACTION_NAME,),
        "num_source_windows": int(num_windows),
        "num_states": int(num_states),
        "num_subsets": int(num_subsets),
        "subset_cap": int(subset_cap),
        "forecast_horizon": int(forecast_horizon),
        "num_features": int(num_features),
        "num_experts": int(num_experts),
        "stop_action_index": int(stop_action_index),
        "action_temperature": float(action_temperature),
        "stop_improvement_threshold": float(stop_improvement_threshold),
        "force_stop_at_cap": bool(force_stop_at_cap),
        "cost_weights": dict(cost_weights or {}),
        "state_sample_indices": torch.cat(state_sample_indices, dim=0).to(torch.long),
        "subset_sizes": torch.cat(subset_sizes, dim=0).to(torch.long),
        "queried_masks": torch.cat(queried_masks, dim=0).to(torch.bool),
        "queryable_expert_masks": torch.cat(queryable_expert_masks, dim=0).to(torch.bool),
        "valid_action_masks": torch.cat(valid_action_masks, dim=0).to(torch.bool),
        "candidate_utilities": torch.cat(candidate_utilities, dim=0).to(torch.float32),
        "action_probabilities": torch.cat(action_probabilities, dim=0).to(torch.float32),
        "optimal_action": torch.cat(optimal_actions, dim=0).to(torch.long),
        "current_best_error": torch.cat(current_best_errors, dim=0).to(torch.float32),
        "current_best_expert": torch.cat(current_best_experts, dim=0).to(torch.long),
        "queried_forecasts": torch.cat(queried_forecasts, dim=0).to(torch.float32),
        "candidate_pairwise_ordering_targets": torch.cat(
            candidate_pairwise_targets,
            dim=0,
        ).to(torch.int8),
        "queried_pairwise_ordering_targets": torch.cat(
            queried_pairwise_targets,
            dim=0,
        ).to(torch.int8),
    }
    validate_costarts_subset_state_cache(cache)
    return cache


def build_and_save_costarts_subset_state_cache(config: SubsetStateBuildConfig) -> dict:
    source_cache_path = Path(config.source_cache_path)
    output_cache_path = Path(config.output_cache_path)
    if not source_cache_path.exists():
        raise FileNotFoundError(f"Missing source COSTARTS cache: {source_cache_path}")
    if output_cache_path.exists() and not config.force_rebuild:
        cache = _load_torch_checkpoint(output_cache_path)
        validate_costarts_subset_state_cache(cache)
        print(f"Using existing subset-state cache: {output_cache_path}")
        return cache

    router_config = load_router_experiment_config()
    source_cache = _load_torch_checkpoint(source_cache_path)
    cache = build_costarts_subset_state_cache(
        source_cache,
        source_cache_path=source_cache_path,
        output_cache_path=output_cache_path,
        subset_cap=config.subset_cap,
        action_temperature=config.action_temperature,
        stop_improvement_threshold=config.stop_improvement_threshold,
        cost_weights=router_config.cost_weights,
        force_stop_at_cap=config.force_stop_at_cap,
        debug=config.debug,
    )
    output_cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output_cache_path)
    print(f"Saved COSTARTS subset-state cache: {output_cache_path}")
    print(f"  source windows: {cache['num_source_windows']}")
    print(f"  subset cap: {cache['subset_cap']}")
    print(f"  subsets per window: {cache['num_subsets']}")
    print(f"  total states: {cache['num_states']}")
    print(f"  action names: {cache['action_names']}")
    return cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build exhaustive subset-state training data from a COSTARTS expert cache.",
    )
    parser.add_argument("--source-cache", default=DEFAULT_SOURCE_CACHE)
    parser.add_argument("--output-cache", default=DEFAULT_OUTPUT_CACHE)
    parser.add_argument(
        "--subset-cap",
        type=int,
        default=None,
        help="Maximum queried-subset size to materialize. Defaults to all experts.",
    )
    parser.add_argument("--action-temperature", type=float, default=0.1)
    parser.add_argument("--stop-improvement-threshold", type=float, default=0.0)
    parser.add_argument(
        "--allow-query-at-cap",
        action="store_true",
        help="Do not force STOP when subset size reaches subset_cap.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output cache.")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_and_save_costarts_subset_state_cache(
        SubsetStateBuildConfig(
            source_cache_path=args.source_cache,
            output_cache_path=args.output_cache,
            subset_cap=args.subset_cap,
            action_temperature=args.action_temperature,
            stop_improvement_threshold=args.stop_improvement_threshold,
            force_stop_at_cap=not args.allow_query_at_cap,
            force_rebuild=args.force,
            debug=args.debug,
        )
    )


if __name__ == "__main__":
    main()
