"""Build exhaustive subset-state datasets for frozen-expert sequential routing.

The builder consumes the offline COSTARTS expert caches produced by
``scripts/train_costarts_router.py``. It never re-runs forecasting experts and
does not use the test split.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import torch

try:
    from scripts.old.router_experiment_config import load_router_experiment_config
except ImportError:
    from scripts.old.router_experiment_config import load_router_experiment_config


DEFAULT_TRAIN_SOURCE = "cache/costarts_router_train_cache.pt"
DEFAULT_VAL_SOURCE = "cache/costarts_router_val_cache.pt"
DEFAULT_TRAIN_OUTPUT = "cache/costarts_subset_states_train.pt"
DEFAULT_VAL_OUTPUT = "cache/costarts_subset_states_val.pt"
SUPPORTED_SAMPLING_MODES = (
    "exhaustive",
    "random",
    "oracle_path_only",
    "model_induced_states",
)


@dataclass(frozen=True)
class CostartsSubsetStateConfig:
    train_source_cache: Union[str, Path] = DEFAULT_TRAIN_SOURCE
    val_source_cache: Union[str, Path] = DEFAULT_VAL_SOURCE
    train_output_cache: Union[str, Path] = DEFAULT_TRAIN_OUTPUT
    val_output_cache: Union[str, Path] = DEFAULT_VAL_OUTPUT
    max_subset_size: Optional[int] = None
    include_empty_set: bool = True
    utility_cost_coefficient: float = 1.0
    cost_schedule_by_expert: Mapping[str, float] = None
    subset_sampling_mode: str = "exhaustive"
    random_states_per_sample: int = 32
    seed: int = 7
    force_rebuild: bool = False
    print_examples: int = 3


def _load_torch(path: Union[str, Path]) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _subset_size_counts(sizes: torch.Tensor) -> dict[str, int]:
    counts = torch.bincount(sizes.to(torch.long))
    return {str(index): int(value) for index, value in enumerate(counts.tolist()) if value}


def _expert_costs(expert_names: Sequence[str], costs: Mapping[str, float]) -> torch.Tensor:
    return torch.tensor(
        [float(costs.get(name, costs.get(f"Candidate_{name}", 0.0))) for name in expert_names],
        dtype=torch.float32,
    )


def _all_subset_masks(
    num_experts: int,
    max_subset_size: int,
    include_empty_set: bool,
) -> torch.Tensor:
    masks = []
    start_size = 0 if include_empty_set else 1
    for subset_size in range(start_size, max_subset_size + 1):
        for subset in combinations(range(num_experts), subset_size):
            mask = torch.zeros(num_experts, dtype=torch.bool)
            if subset:
                mask[list(subset)] = True
            masks.append(mask)
    if not masks:
        raise ValueError("No subsets selected; enable include_empty_set or use max_subset_size >= 1")
    return torch.stack(masks, dim=0)


def _build_state_index(
    error_matrix: torch.Tensor,
    *,
    max_subset_size: int,
    include_empty_set: bool,
    sampling_mode: str,
    random_states_per_sample: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_windows, num_experts = error_matrix.shape
    if sampling_mode == "model_induced_states":
        raise NotImplementedError(
            "model_induced_states requires saved model rollouts; this offline builder "
            "supports exhaustive, random, and oracle_path_only from cache alone."
        )

    if sampling_mode == "exhaustive":
        subset_masks = _all_subset_masks(num_experts, max_subset_size, include_empty_set)
        source_rows = torch.arange(num_windows, dtype=torch.long).repeat_interleave(
            subset_masks.shape[0]
        )
        masks = subset_masks.repeat(num_windows, 1)
        return source_rows, masks

    rows = []
    masks = []
    rng = random.Random(seed)
    start_size = 0 if include_empty_set else 1
    if sampling_mode == "oracle_path_only":
        oracle_order = torch.argsort(error_matrix, dim=1)
        for sample_index in range(num_windows):
            for subset_size in range(start_size, max_subset_size + 1):
                mask = torch.zeros(num_experts, dtype=torch.bool)
                if subset_size:
                    mask[oracle_order[sample_index, :subset_size]] = True
                rows.append(sample_index)
                masks.append(mask)
    elif sampling_mode == "random":
        all_masks = _all_subset_masks(num_experts, max_subset_size, include_empty_set)
        for sample_index in range(num_windows):
            chosen = [
                all_masks[index]
                for index in rng.sample(
                    range(all_masks.shape[0]),
                    k=min(int(random_states_per_sample), all_masks.shape[0]),
                )
            ]
            for mask in chosen:
                rows.append(sample_index)
                masks.append(mask.clone())
    else:
        raise ValueError(f"Unknown subset_sampling_mode: {sampling_mode}")
    return torch.tensor(rows, dtype=torch.long), torch.stack(masks, dim=0)


def _masked_mae_mse(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    target_masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = target_masks.to(torch.float32)
    denominator = mask.sum(dim=(1, 2)).clamp_min(1.0)
    abs_error = torch.abs(predictions - targets) * mask
    squared_error = (predictions - targets).pow(2) * mask
    return (
        abs_error.sum(dim=(1, 2)) / denominator,
        squared_error.sum(dim=(1, 2)) / denominator,
    )


def _pairwise_labels(errors: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
    # 1 means row expert is better than column expert, -1 worse, 0 invalid/tie.
    diff = errors.unsqueeze(2) - errors.unsqueeze(1)
    labels = torch.sign(-diff).to(torch.int8)
    return labels.masked_fill(~pair_mask, 0)


def validate_costarts_subset_states(cache: Mapping[str, Any]) -> None:
    num_states = int(cache["num_states"])
    num_experts = int(cache["num_experts"])
    max_subset_size = int(cache["max_subset_size"])
    horizon = int(cache["forecast_horizon"])
    num_features = int(cache["num_features"])

    assert cache["split_role"] in {"router_train", "router_val"}
    assert cache["source_split_role"] == cache["split_role"]
    assert tuple(cache["state_id"].shape) == (num_states,)
    assert tuple(cache["sample_index"].shape) == (num_states,)
    assert tuple(cache["source_row"].shape) == (num_states,)
    assert tuple(cache["queried_mask"].shape) == (num_states, num_experts)
    assert tuple(cache["remaining_mask"].shape) == (num_states, num_experts)
    assert tuple(cache["queried_expert_ids"].shape) == (num_states, max_subset_size)
    assert tuple(cache["queried_expert_forecasts"].shape) == (
        num_states,
        max_subset_size,
        horizon,
        num_features,
    )
    assert tuple(cache["history"].shape) == (num_states, 96, num_features)
    assert tuple(cache["true_targets"].shape) == (num_states, horizon, num_features)
    assert tuple(cache["target_mask"].shape) == (num_states, horizon, num_features)
    assert tuple(cache["true_expert_error_vector"].shape) == (num_states, num_experts)
    assert tuple(cache["current_loss_best_queried_oracle"].shape) == (num_states,)
    assert tuple(cache["current_loss_equal_queried_average"].shape) == (num_states,)
    assert tuple(cache["current_loss_deployable_reranker"].shape) == (num_states,)
    assert tuple(cache["marginal_gain_best_queried_oracle"].shape) == (
        num_states,
        num_experts,
    )
    assert tuple(cache["marginal_gain_equal_queried_average"].shape) == (
        num_states,
        num_experts,
    )
    if "base_expert_costs" in cache:
        assert tuple(cache["base_expert_costs"].shape) == (num_experts,)
    assert tuple(cache["cost_adjusted_utility"].shape) == (num_states, num_experts)
    assert tuple(cache["optimal_next_action"].shape) == (num_states,)
    assert tuple(cache["valid_action_mask"].shape) == (num_states, num_experts + 1)
    assert tuple(cache["pairwise_labels_queried"].shape) == (
        num_states,
        num_experts,
        num_experts,
    )
    assert tuple(cache["pairwise_labels_remaining"].shape) == (
        num_states,
        num_experts,
        num_experts,
    )
    if cache["source_sample_indices_contiguous"] is not True:
        raise AssertionError("source sample indices must be contiguous")
    if cache["split_role"] == "test":
        raise AssertionError("subset-state cache must never use test split")
    chosen_valid = cache["valid_action_mask"].gather(
        1,
        cache["optimal_next_action"].view(-1, 1),
    )
    if not torch.all(chosen_valid):
        raise AssertionError("Optimal next action must always be valid")


def _assert_queried_forecast_integrity(
    cache: Mapping[str, Any],
    source_prediction_stack: torch.Tensor,
) -> None:
    source_rows = cache["source_row"]
    ids = cache["queried_expert_ids"]
    forecasts = cache["queried_expert_forecasts"]
    num_states, max_subset_size = ids.shape
    for slot in range(max_subset_size):
        valid = ids[:, slot] >= 0
        if not torch.any(valid):
            continue
        expected = source_prediction_stack[source_rows[valid], :, :, ids[valid, slot]]
        actual = forecasts[valid, slot]
        if not torch.allclose(actual, expected, atol=0.0, rtol=0.0):
            raise AssertionError(f"queried forecasts do not match source predictions at slot {slot}")
    padded = ids < 0
    if torch.any(padded):
        expanded = padded[:, :, None, None].expand_as(forecasts)
        if not torch.equal(forecasts.masked_select(expanded), torch.zeros_like(forecasts.masked_select(expanded))):
            raise AssertionError("padded queried forecast slots must be zero")


def build_subset_state_cache_from_costarts_cache(
    source_cache: Mapping[str, Any],
    *,
    output_cache_path: Union[str, Path],
    max_subset_size: Optional[int],
    include_empty_set: bool,
    utility_cost_coefficient: float,
    cost_schedule_by_expert: Mapping[str, float],
    subset_sampling_mode: str,
    random_states_per_sample: int,
    seed: int,
) -> dict:
    split_role = str(source_cache["split_role"])
    if split_role not in {"router_train", "router_val"}:
        raise ValueError(f"Subset-state generation only supports router_train/router_val, got {split_role}")
    if subset_sampling_mode not in SUPPORTED_SAMPLING_MODES:
        raise ValueError(f"subset_sampling_mode must be one of {SUPPORTED_SAMPLING_MODES}")
    if utility_cost_coefficient < 0:
        raise ValueError("utility_cost_coefficient must be non-negative")

    expert_names = tuple(source_cache["expert_names"])
    num_experts = len(expert_names)
    max_subset_size = num_experts if max_subset_size is None else int(max_subset_size)
    if max_subset_size < 0 or max_subset_size > num_experts:
        raise ValueError(f"max_subset_size must be between 0 and {num_experts}")

    num_windows = int(source_cache["num_windows"])
    histories = source_cache["histories"].to(torch.float32)
    targets = source_cache["targets"].to(torch.float32)
    target_masks = source_cache["target_masks"].to(torch.bool)
    prediction_stack = source_cache["prediction_stack"].to(torch.float32)
    error_matrix = source_cache["error_matrix"].to(torch.float32)
    sample_indices = source_cache["sample_indices"].to(torch.long)
    assert tuple(histories.shape) == (num_windows, 96, int(source_cache["num_features"]))
    assert tuple(targets.shape) == (
        num_windows,
        int(source_cache["forecast_horizon"]),
        int(source_cache["num_features"]),
    )
    assert tuple(prediction_stack.shape) == (
        num_windows,
        int(source_cache["forecast_horizon"]),
        int(source_cache["num_features"]),
        num_experts,
    )
    if not torch.equal(sample_indices, torch.arange(num_windows, dtype=sample_indices.dtype)):
        raise AssertionError("source cache sample_indices must be contiguous")

    source_rows, queried_mask = _build_state_index(
        error_matrix,
        max_subset_size=max_subset_size,
        include_empty_set=include_empty_set,
        sampling_mode=subset_sampling_mode,
        random_states_per_sample=random_states_per_sample,
        seed=seed,
    )
    num_states = int(source_rows.shape[0])
    state_errors = error_matrix[source_rows]
    state_prediction_stack = prediction_stack[source_rows]
    state_targets = targets[source_rows]
    state_masks = target_masks[source_rows]
    subset_sizes = queried_mask.sum(dim=1).to(torch.long)
    remaining_mask = ~queried_mask
    capped_remaining_mask = remaining_mask & (subset_sizes[:, None] < max_subset_size)
    has_queried = subset_sizes > 0

    inf = torch.tensor(float("inf"), dtype=torch.float32)
    nan = torch.tensor(float("nan"), dtype=torch.float32)

    masked_errors = state_errors.masked_fill(~queried_mask, float("inf"))
    current_loss_best = masked_errors.min(dim=1).values
    current_loss_best = torch.where(has_queried, current_loss_best, nan)

    count = subset_sizes.to(torch.float32).clamp_min(1.0)
    summed_predictions = (state_prediction_stack * queried_mask[:, None, None, :]).sum(dim=-1)
    equal_predictions = summed_predictions / count[:, None, None]
    current_loss_equal, _ = _masked_mae_mse(equal_predictions, state_targets, state_masks)
    current_loss_equal = torch.where(has_queried, current_loss_equal, nan)

    candidate_loss_best = torch.minimum(
        torch.where(has_queried, current_loss_best, inf).view(-1, 1),
        state_errors,
    )
    candidate_loss_best = torch.where(has_queried.view(-1, 1), candidate_loss_best, state_errors)
    marginal_best = current_loss_best.view(-1, 1) - candidate_loss_best
    marginal_best = torch.where(has_queried.view(-1, 1), marginal_best, nan)
    marginal_best = marginal_best.masked_fill(~capped_remaining_mask, float("-inf"))

    candidate_equal_predictions = (
        summed_predictions[:, :, :, None] + state_prediction_stack
    ) / (subset_sizes.to(torch.float32).view(-1, 1, 1, 1) + 1.0)
    mask_float = state_masks.to(torch.float32)
    denominator = mask_float.sum(dim=(1, 2)).clamp_min(1.0).view(-1, 1)
    candidate_equal_loss = (
        torch.abs(candidate_equal_predictions - state_targets[:, :, :, None])
        * mask_float[:, :, :, None]
    ).sum(dim=(1, 2)) / denominator
    marginal_equal = current_loss_equal.view(-1, 1) - candidate_equal_loss
    marginal_equal = torch.where(has_queried.view(-1, 1), marginal_equal, nan)
    marginal_equal = marginal_equal.masked_fill(~capped_remaining_mask, float("-inf"))

    expert_costs = _expert_costs(expert_names, cost_schedule_by_expert)
    if torch.any(expert_costs < 0):
        raise ValueError("expert costs must be non-negative")
    empty_state_utility = -state_errors - utility_cost_coefficient * expert_costs.view(1, -1)
    non_empty_utility = marginal_equal - utility_cost_coefficient * expert_costs.view(1, -1)
    utility = torch.where(has_queried.view(-1, 1), non_empty_utility, empty_state_utility)
    utility = utility.masked_fill(~capped_remaining_mask, float("-inf"))

    stop_action_index = num_experts
    valid_action_mask = torch.zeros(num_states, num_experts + 1, dtype=torch.bool)
    valid_action_mask[:, :num_experts] = capped_remaining_mask
    valid_action_mask[:, stop_action_index] = has_queried
    best_utility, best_expert_action = utility.max(dim=1)
    stop_is_best = has_queried & ((best_utility <= 0.0) | ~capped_remaining_mask.any(dim=1))
    optimal_next_action = torch.where(
        stop_is_best,
        torch.full_like(best_expert_action, stop_action_index),
        best_expert_action,
    )
    if torch.any(~valid_action_mask.gather(1, optimal_next_action.view(-1, 1)).squeeze(1)):
        raise AssertionError("computed an invalid optimal action")

    queried_ids = torch.full((num_states, max_subset_size), -1, dtype=torch.long)
    for state_index in range(num_states):
        ids = torch.nonzero(queried_mask[state_index], as_tuple=False).flatten()
        queried_ids[state_index, : ids.numel()] = ids
    gather_ids = queried_ids.clamp_min(0)
    queried_forecasts = state_prediction_stack.gather(
        dim=-1,
        index=gather_ids[:, None, None, :].expand(
            num_states,
            int(source_cache["forecast_horizon"]),
            int(source_cache["num_features"]),
            max_subset_size,
        ),
    ).permute(0, 3, 1, 2)
    queried_forecasts = queried_forecasts.masked_fill(
        (queried_ids < 0)[:, :, None, None],
        0.0,
    )

    pairwise_queried_mask = queried_mask[:, :, None] & queried_mask[:, None, :]
    pairwise_remaining_mask = remaining_mask[:, :, None] & remaining_mask[:, None, :]
    pairwise_labels_queried = _pairwise_labels(state_errors, pairwise_queried_mask)
    pairwise_labels_remaining = _pairwise_labels(state_errors, pairwise_remaining_mask)

    source_sample_indices = sample_indices[source_rows]
    cache = {
        "cache_type": "costarts_subset_states",
        "split_role": split_role,
        "source_split_role": split_role,
        "source_cache_path": str(source_cache.get("cache_path", "")),
        "output_cache_path": str(output_cache_path),
        "expert_names": expert_names,
        "num_source_windows": num_windows,
        "num_states": num_states,
        "num_experts": num_experts,
        "max_subset_size": max_subset_size,
        "include_empty_set": bool(include_empty_set),
        "subset_sampling_mode": subset_sampling_mode,
        "random_states_per_sample": int(random_states_per_sample),
        "seed": int(seed),
        "forecast_horizon": int(source_cache["forecast_horizon"]),
        "num_features": int(source_cache["num_features"]),
        "stop_action_index": stop_action_index,
        "utility_finalizer": "equal_queried_average",
        "utility_cost_coefficient": float(utility_cost_coefficient),
        "cost_schedule_by_expert": dict(cost_schedule_by_expert),
        "base_expert_costs": expert_costs.to(torch.float32),
        "empty_state_utility_definition": "-L({e}) - lambda*c_e because L(empty) is undefined",
        "source_sample_indices_contiguous": bool(
            torch.equal(sample_indices, torch.arange(num_windows, dtype=sample_indices.dtype))
        ),
        "state_id": torch.arange(num_states, dtype=torch.long),
        "sample_index": source_sample_indices.to(torch.long),
        "source_row": source_rows.to(torch.long),
        "subset_size": subset_sizes.to(torch.long),
        "queried_mask": queried_mask.to(torch.bool),
        "remaining_mask": remaining_mask.to(torch.bool),
        "queried_expert_ids": queried_ids.to(torch.long),
        "queried_expert_forecasts": queried_forecasts.to(torch.float32),
        "history": histories[source_rows].to(torch.float32),
        "true_targets": state_targets.to(torch.float32),
        "target_mask": state_masks.to(torch.bool),
        "true_expert_error_vector": state_errors.to(torch.float32),
        "current_loss_best_queried_oracle": current_loss_best.to(torch.float32),
        "current_loss_equal_queried_average": current_loss_equal.to(torch.float32),
        "current_loss_deployable_reranker": torch.full((num_states,), float("nan")),
        "candidate_loss_after_best_queried_oracle": candidate_loss_best.masked_fill(
            ~remaining_mask,
            float("nan"),
        ).to(torch.float32),
        "candidate_loss_after_equal_average": candidate_equal_loss.masked_fill(
            ~remaining_mask,
            float("nan"),
        ).to(torch.float32),
        "marginal_gain_best_queried_oracle": marginal_best.to(torch.float32),
        "marginal_gain_equal_queried_average": marginal_equal.to(torch.float32),
        "marginal_gain_deployable_reranker": torch.full(
            (num_states, num_experts),
            float("nan"),
        ),
        "cost_adjusted_utility": utility.to(torch.float32),
        "optimal_next_action": optimal_next_action.to(torch.long),
        "valid_action_mask": valid_action_mask.to(torch.bool),
        "pairwise_labels_queried": pairwise_labels_queried.to(torch.int8),
        "pairwise_labels_remaining": pairwise_labels_remaining.to(torch.int8),
        "pairwise_mask_queried": pairwise_queried_mask.to(torch.bool),
        "pairwise_mask_remaining": pairwise_remaining_mask.to(torch.bool),
        "state_counts_by_subset_size": _subset_size_counts(subset_sizes),
    }
    validate_costarts_subset_states(cache)
    _assert_queried_forecast_integrity(cache, prediction_stack)
    return cache


def build_and_save_subset_states(
    source_path: Union[str, Path],
    output_path: Union[str, Path],
    *,
    max_subset_size: Optional[int],
    include_empty_set: bool,
    utility_cost_coefficient: float,
    cost_schedule_by_expert: Mapping[str, float],
    subset_sampling_mode: str,
    random_states_per_sample: int,
    seed: int,
    force_rebuild: bool,
    print_examples: int,
) -> dict:
    source_path = Path(source_path)
    output_path = Path(output_path)
    if output_path.exists() and not force_rebuild:
        cache = _load_torch(output_path)
        validate_costarts_subset_states(cache)
        print(f"Using existing subset-state cache: {output_path}")
        return cache
    if not source_path.exists():
        raise FileNotFoundError(f"Missing COSTARTS source cache: {source_path}")
    source_cache = _load_torch(source_path)
    source_cache = dict(source_cache)
    source_cache["cache_path"] = str(source_path)
    cache = build_subset_state_cache_from_costarts_cache(
        source_cache,
        output_cache_path=output_path,
        max_subset_size=max_subset_size,
        include_empty_set=include_empty_set,
        utility_cost_coefficient=utility_cost_coefficient,
        cost_schedule_by_expert=cost_schedule_by_expert,
        subset_sampling_mode=subset_sampling_mode,
        random_states_per_sample=random_states_per_sample,
        seed=seed,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output_path)
    print(f"Saved: {output_path}")
    print(f"  split_role: {cache['split_role']}")
    print(f"  source windows: {cache['num_source_windows']}")
    print(f"  total states: {cache['num_states']}")
    print(f"  state counts by subset size: {cache['state_counts_by_subset_size']}")
    if print_examples > 0:
        print_example_states(cache, limit=print_examples)
    return cache


def print_example_states(cache: Mapping[str, Any], limit: int = 3) -> None:
    expert_names = tuple(cache["expert_names"])
    stop_index = int(cache["stop_action_index"])
    print("  example states:")
    for index in range(min(limit, int(cache["num_states"]))):
        ids = [
            expert_names[int(expert_id)]
            for expert_id in cache["queried_expert_ids"][index].tolist()
            if expert_id >= 0
        ]
        action = int(cache["optimal_next_action"][index])
        action_name = "STOP" if action == stop_index else expert_names[action]
        utilities = cache["cost_adjusted_utility"][index].tolist()
        finite_utilities = {
            expert_names[expert_index]: round(float(value), 6)
            for expert_index, value in enumerate(utilities)
            if math.isfinite(float(value))
        }
        print(
            f"    state_id={index}, sample={int(cache['sample_index'][index])}, "
            f"subset={ids}, action={action_name}, utilities={finite_utilities}"
        )


def _config_from_repo() -> tuple[dict[str, Any], dict[str, str]]:
    repo_config = load_router_experiment_config()
    return (
        {
            "max_subset_size": repo_config.costarts_subset_max_size,
            "include_empty_set": repo_config.costarts_subset_include_empty,
            "utility_cost_coefficient": repo_config.costarts_subset_utility_cost_coefficient,
            "cost_schedule_by_expert": dict(repo_config.costarts_subset_cost_schedule),
            "subset_sampling_mode": repo_config.costarts_subset_sampling_mode,
            "seed": repo_config.random_seed,
        },
        dict(repo_config.cache_paths),
    )


def parse_args() -> argparse.Namespace:
    defaults, cache_paths = _config_from_repo()
    parser = argparse.ArgumentParser(
        description="Build COSTARTS subset-state train/val caches from existing offline expert caches.",
    )
    parser.add_argument("--train-source-cache", default=cache_paths.get("costarts_train", DEFAULT_TRAIN_SOURCE))
    parser.add_argument("--val-source-cache", default=cache_paths.get("costarts_val", DEFAULT_VAL_SOURCE))
    parser.add_argument(
        "--train-output-cache",
        default=cache_paths.get("costarts_subset_states_train", DEFAULT_TRAIN_OUTPUT),
    )
    parser.add_argument(
        "--val-output-cache",
        default=cache_paths.get("costarts_subset_states_val", DEFAULT_VAL_OUTPUT),
    )
    parser.add_argument("--split", choices=("both", "train", "val"), default="both")
    parser.add_argument("--max-subset-size", type=int, default=defaults["max_subset_size"])
    parser.add_argument("--exclude-empty-set", action="store_true")
    parser.add_argument(
        "--utility-cost-coefficient",
        type=float,
        default=defaults["utility_cost_coefficient"],
    )
    parser.add_argument(
        "--cost-schedule-json",
        default=json.dumps(defaults["cost_schedule_by_expert"]),
        help="JSON object mapping expert name to non-negative query cost.",
    )
    parser.add_argument(
        "--subset-sampling-mode",
        choices=SUPPORTED_SAMPLING_MODES,
        default=defaults["subset_sampling_mode"],
    )
    parser.add_argument("--random-states-per-sample", type=int, default=32)
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--print-examples", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cost_schedule = json.loads(args.cost_schedule_json)
    include_empty = not args.exclude_empty_set
    outputs = []
    if args.split in {"both", "train"}:
        outputs.append(
            build_and_save_subset_states(
                args.train_source_cache,
                args.train_output_cache,
                max_subset_size=args.max_subset_size,
                include_empty_set=include_empty,
                utility_cost_coefficient=args.utility_cost_coefficient,
                cost_schedule_by_expert=cost_schedule,
                subset_sampling_mode=args.subset_sampling_mode,
                random_states_per_sample=args.random_states_per_sample,
                seed=args.seed,
                force_rebuild=args.force,
                print_examples=args.print_examples,
            )
        )
    if args.split in {"both", "val"}:
        outputs.append(
            build_and_save_subset_states(
                args.val_source_cache,
                args.val_output_cache,
                max_subset_size=args.max_subset_size,
                include_empty_set=include_empty,
                utility_cost_coefficient=args.utility_cost_coefficient,
                cost_schedule_by_expert=cost_schedule,
                subset_sampling_mode=args.subset_sampling_mode,
                random_states_per_sample=args.random_states_per_sample,
                seed=args.seed,
                force_rebuild=args.force,
                print_examples=args.print_examples,
            )
        )
    if len(outputs) == 2:
        train_names = tuple(outputs[0]["expert_names"])
        val_names = tuple(outputs[1]["expert_names"])
        if train_names != val_names:
            raise AssertionError(f"Train/val expert ordering mismatch: {train_names} != {val_names}")
    print("COSTARTS subset-state generation complete.")


if __name__ == "__main__":
    main()
