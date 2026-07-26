"""Sequential rollout evaluation for SubsetUtilityCOSTARTSRouter.

The evaluator starts from the empty queried set, selects QUERY or STOP actions,
reveals only the selected expert forecast from the offline frozen-expert cache,
and updates the subset state before the next action. It supports greedy,
temperature-sampled, and forced-budget rollouts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F

try:
    from scripts.build_costarts_subset_states import validate_costarts_subset_states
    from scripts.train_costarts_subset_utility_router import (
        SubsetUtilityCOSTARTSRouter,
        _build_state_lookup,
        _masked_action_logits,
        _state_batch,
        set_reproducible_seed,
    )
except ImportError:
    from build_costarts_subset_states import validate_costarts_subset_states
    from train_costarts_subset_utility_router import (
        SubsetUtilityCOSTARTSRouter,
        _build_state_lookup,
        _masked_action_logits,
        _state_batch,
        set_reproducible_seed,
    )


DEFAULT_CACHE = "cache/costarts_subset_states_val.pt"
DEFAULT_CHECKPOINT = "checkpoints/costarts_subset_utility/best_subset_utility_costarts_router.pt"
DEFAULT_OUTPUT_DIR = "results/router_summary/costarts_subset_utility/rollouts"


def _load_torch(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return str(value)
    return value


def _load_router(checkpoint_path: Path, device: torch.device) -> tuple[SubsetUtilityCOSTARTSRouter, dict]:
    checkpoint = _load_torch(checkpoint_path)
    router = SubsetUtilityCOSTARTSRouter(**checkpoint["router_config"]).to(device)
    router.load_state_dict(checkpoint["router_state_dict"])
    router.eval()
    return router, checkpoint


def _bit_indices(mask_int: int, num_experts: int) -> list[int]:
    return [index for index in range(num_experts) if mask_int & (1 << index)]


def _masked_softmax_sample(logits: torch.Tensor, valid_mask: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    masked = logits.masked_fill(~valid_mask.to(torch.bool), -1e9)
    probabilities = torch.softmax(masked / temperature, dim=-1)
    probabilities = probabilities.masked_fill(~valid_mask.to(torch.bool), 0.0)
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.multinomial(probabilities, num_samples=1).squeeze(-1)


def _masked_argmax_expert(logits: torch.Tensor, valid_mask: torch.Tensor, num_experts: int) -> torch.Tensor:
    expert_valid = valid_mask[:, :num_experts].to(torch.bool)
    return torch.argmax(logits[:, :num_experts].masked_fill(~expert_valid, -1e9), dim=-1)


def _mae_mse(prediction: torch.Tensor, target: torch.Tensor, target_mask: torch.Tensor) -> tuple[float, float]:
    mask = target_mask.to(torch.float32)
    denominator = mask.sum().clamp_min(1.0)
    mae = (torch.abs(prediction - target) * mask).sum() / denominator
    mse = ((prediction - target).pow(2) * mask).sum() / denominator
    return float(mae), float(mse)


@torch.no_grad()
def evaluate_rollouts(
    *,
    router: SubsetUtilityCOSTARTSRouter,
    cache: Mapping[str, Any],
    mode: str,
    finalizer: str,
    force_k: Optional[int],
    temperature: float,
    max_queries: Optional[int],
    batch_size: int,
    device: torch.device,
    seed: int,
    detailed_limit: int,
) -> dict[str, Any]:
    validate_costarts_subset_states(cache)
    if cache["split_role"] != "router_val":
        raise ValueError("Sequential rollout evaluation must use router_val subset-state cache")
    if cache["subset_sampling_mode"] != "exhaustive":
        raise ValueError("Sequential rollouts require exhaustive subset-state cache")
    if finalizer not in {"equal_average", "reranker", "sparse_mixture"}:
        raise ValueError("finalizer must be equal_average, reranker, or sparse_mixture")
    if mode not in {"greedy", "sampled", "forced"}:
        raise ValueError("mode must be greedy, sampled, or forced")

    set_reproducible_seed(seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    router.eval()

    lookup = _build_state_lookup(cache)
    num_windows = int(cache["num_source_windows"])
    num_experts = int(cache["num_experts"])
    cache_max_subset = int(cache["max_subset_size"])
    stop_index = int(cache["stop_action_index"])
    max_queries = cache_max_subset if max_queries is None else min(int(max_queries), cache_max_subset)
    if max_queries <= 0:
        raise ValueError("max_queries must be positive")
    if force_k is not None and (force_k < 1 or force_k > max_queries):
        raise ValueError("force_k must be between 1 and max_queries")

    masks = [0 for _ in range(num_windows)]
    done = [False for _ in range(num_windows)]
    query_sequences: list[list[int]] = [[] for _ in range(num_windows)]
    action_sequences: list[list[str]] = [[] for _ in range(num_windows)]
    action_prob_sequences: list[list[dict[str, float]]] = [[] for _ in range(num_windows)]
    predicted_utility_sequences: list[list[float]] = [[] for _ in range(num_windows)]
    query_event_count = 0
    state_changed_count = 0
    duplicate_query_count = 0
    stop_step_counts = torch.zeros(max_queries + 1, dtype=torch.long)
    latency_start = time.perf_counter()

    for step in range(max_queries):
        active = [index for index in range(num_windows) if not done[index]]
        if not active:
            break
        for offset in range(0, len(active), batch_size):
            rows = active[offset : offset + batch_size]
            state_indices = [lookup[row][masks[row]] for row in rows]
            batch = _state_batch(cache, state_indices, device)
            outputs = router(
                batch["history"],
                batch["queried_mask"],
                batch["queried_expert_ids"],
                batch["queried_expert_forecasts"],
            )
            valid_action_mask = batch["valid_action_mask"].to(torch.bool)
            masked_logits = _masked_action_logits(outputs["action_logits"], valid_action_mask)
            action_probabilities = torch.softmax(masked_logits, dim=-1).detach().cpu()

            if mode == "greedy":
                actions = torch.argmax(masked_logits, dim=-1).detach().cpu()
            elif mode == "sampled":
                # torch.multinomial with an explicit CPU generator is more portable.
                probabilities = torch.softmax(masked_logits.detach().cpu() / temperature, dim=-1)
                probabilities = probabilities.masked_fill(~valid_action_mask.detach().cpu(), 0.0)
                probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                actions = torch.multinomial(probabilities, num_samples=1, generator=torch.Generator().manual_seed(seed + step + offset)).squeeze(-1)
            else:
                actions = _masked_argmax_expert(
                    outputs["action_logits"],
                    valid_action_mask,
                    num_experts,
                ).detach().cpu()

            utilities = outputs["utility_prediction"].detach().cpu()
            for local_index, sample_index in enumerate(rows):
                action = int(actions[local_index])
                if mode == "forced" and len(query_sequences[sample_index]) >= int(force_k):
                    action = stop_index
                if action == stop_index:
                    if not done[sample_index]:
                        done[sample_index] = True
                        stop_step = max(1, len(query_sequences[sample_index]))
                        stop_step_counts[min(stop_step, max_queries)] += 1
                        action_sequences[sample_index].append("STOP")
                    continue

                if action in query_sequences[sample_index]:
                    duplicate_query_count += 1
                    done[sample_index] = True
                    action_sequences[sample_index].append("DUPLICATE_BLOCKED")
                    continue

                previous_mask = masks[sample_index]
                masks[sample_index] |= 1 << action
                if masks[sample_index] != previous_mask:
                    state_changed_count += 1
                query_event_count += 1
                query_sequences[sample_index].append(action)
                action_sequences[sample_index].append(f"QUERY:{action}")
                predicted_utility_sequences[sample_index].append(float(utilities[local_index, action]))
                action_prob_sequences[sample_index].append(
                    {
                        str(action_index): float(action_probabilities[local_index, action_index])
                        for action_index in range(num_experts + 1)
                    }
                )

                if len(query_sequences[sample_index]) >= max_queries:
                    done[sample_index] = True
                    stop_step_counts[len(query_sequences[sample_index])] += 1
                elif mode == "forced" and len(query_sequences[sample_index]) >= int(force_k):
                    done[sample_index] = True
                    stop_step_counts[len(query_sequences[sample_index])] += 1
                    action_sequences[sample_index].append("STOP_FORCED_BUDGET")

    for sample_index in range(num_windows):
        if not query_sequences[sample_index]:
            # STOP is invalid at the empty state, but this guard ensures metrics
            # stay defined if a malformed checkpoint/logit path ever produces it.
            state_index = lookup[sample_index][masks[sample_index]]
            batch = _state_batch(cache, [state_index], device)
            outputs = router(
                batch["history"],
                batch["queried_mask"],
                batch["queried_expert_ids"],
                batch["queried_expert_forecasts"],
            )
            valid_action_mask = batch["valid_action_mask"].to(torch.bool)
            fallback = int(_masked_argmax_expert(outputs["action_logits"], valid_action_mask, num_experts)[0].cpu())
            masks[sample_index] |= 1 << fallback
            query_sequences[sample_index].append(fallback)
            action_sequences[sample_index].append(f"QUERY_FALLBACK:{fallback}")
            query_event_count += 1
            state_changed_count += 1
            stop_step_counts[1] += 1

    final_state_indices = [lookup[row][masks[row]] for row in range(num_windows)]
    selected_experts = torch.empty(num_windows, dtype=torch.long)
    mix_weights_full = torch.zeros(num_windows, num_experts, dtype=torch.float32)
    predictions = []
    target_rows = []
    mask_rows = []

    for offset in range(0, num_windows, batch_size):
        rows = list(range(offset, min(offset + batch_size, num_windows)))
        state_indices = final_state_indices[offset : offset + len(rows)]
        batch = _state_batch(cache, state_indices, device)
        outputs = router(
            batch["history"],
            batch["queried_mask"],
            batch["queried_expert_ids"],
            batch["queried_expert_forecasts"],
        )
        queried_mask = batch["queried_mask"].detach().cpu()
        queried_ids = batch["queried_expert_ids"].detach().cpu()
        queried_forecasts = batch["queried_expert_forecasts"].detach().cpu()
        true_targets = batch["true_targets"].detach().cpu()
        target_mask = batch["target_mask"].detach().cpu()
        target_rows.append(true_targets)
        mask_rows.append(target_mask)

        if finalizer == "equal_average":
            valid_slots = queried_ids >= 0
            slot_weights = valid_slots.to(queried_forecasts.dtype)
            slot_weights = slot_weights / slot_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
            pred = (queried_forecasts * slot_weights[:, :, None, None]).sum(dim=1)
            predictions.append(pred)
            for local_index in range(len(rows)):
                valid_experts = queried_ids[local_index][valid_slots[local_index]]
                selected_experts[offset + local_index] = valid_experts[0].to(torch.long)
                weight = 1.0 / max(int(valid_experts.numel()), 1)
                for expert_index in valid_experts.tolist():
                    mix_weights_full[offset + local_index, int(expert_index)] = weight
        elif finalizer == "reranker":
            scores = outputs["expert_score"].detach().cpu().masked_fill(~queried_mask, -1e9)
            selected = torch.argmax(scores, dim=-1)
            selected_experts[offset : offset + len(rows)] = selected
            chosen_positions = (queried_ids == selected[:, None]).to(torch.float32).argmax(dim=1)
            pred = queried_forecasts[torch.arange(len(rows)), chosen_positions]
            for local_index, expert_index in enumerate(selected.tolist()):
                mix_weights_full[offset + local_index, expert_index] = 1.0
            predictions.append(pred)
        else:
            valid_slots = queried_ids >= 0
            gathered_logits = outputs["mix_logits"].detach().cpu().gather(1, queried_ids.clamp_min(0))
            gathered_logits = gathered_logits.masked_fill(~valid_slots, -1e9)
            slot_weights = torch.softmax(gathered_logits, dim=1).masked_fill(~valid_slots, 0.0)
            slot_weights = slot_weights / slot_weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
            pred = (queried_forecasts * slot_weights[:, :, None, None]).sum(dim=1)
            predictions.append(pred)
            for local_index in range(len(rows)):
                for slot_index, expert_index in enumerate(queried_ids[local_index].tolist()):
                    if expert_index >= 0:
                        mix_weights_full[offset + local_index, int(expert_index)] = slot_weights[local_index, slot_index]
            selected_experts[offset : offset + len(rows)] = torch.argmax(mix_weights_full[offset : offset + len(rows)], dim=1)

    prediction = torch.cat(predictions, dim=0)
    targets = torch.cat(target_rows, dim=0)
    target_masks = torch.cat(mask_rows, dim=0)
    mae, mse = _mae_mse(prediction, targets, target_masks)

    source_errors = torch.empty(num_windows, num_experts, dtype=torch.float32)
    for row in range(num_windows):
        source_errors[row] = cache["true_expert_error_vector"][lookup[row][0]]
    oracle_best = torch.argmin(source_errors, dim=1)
    oracle_mae = float(source_errors.min(dim=1).values.mean())
    selected_error = source_errors.gather(1, selected_experts.view(-1, 1)).squeeze(1)
    avg_queried = sum(len(seq) for seq in query_sequences) / max(num_windows, 1)
    latency_seconds = time.perf_counter() - latency_start

    detailed = []
    expert_names = tuple(cache["expert_names"])
    for sample_index in range(min(detailed_limit, num_windows)):
        queried_names = [expert_names[index] for index in query_sequences[sample_index]]
        selected_name = expert_names[int(selected_experts[sample_index])]
        detailed.append(
            {
                "sample_index": int(cache["sample_index"][lookup[sample_index][0]]),
                "query_sequence_indices": query_sequences[sample_index],
                "query_sequence_names": queried_names,
                "action_sequence": action_sequences[sample_index],
                "stop_step": len(query_sequences[sample_index]),
                "final_selected_expert": selected_name,
                "mix_weights": {
                    expert_names[index]: float(mix_weights_full[sample_index, index])
                    for index in range(num_experts)
                    if float(mix_weights_full[sample_index, index]) > 0.0
                },
                "oracle_best_expert": expert_names[int(oracle_best[sample_index])],
                "selected_expert_error": float(selected_error[sample_index]),
                "oracle_error": float(source_errors[sample_index].min()),
                "predicted_utilities": predicted_utility_sequences[sample_index],
                "action_probabilities": action_prob_sequences[sample_index],
                "state_changed_each_query": True,
                "queries_unique": len(query_sequences[sample_index]) == len(set(query_sequences[sample_index])),
            }
        )

    return {
        "metadata": {
            "mode": mode,
            "finalizer": finalizer,
            "force_k": force_k,
            "temperature": temperature,
            "max_queries": max_queries,
            "num_windows": num_windows,
            "num_experts": num_experts,
            "expert_names": expert_names,
            "test_data_used": False,
            "frozen_expert_calls_simulated_from_cache": query_event_count,
            "router_state_changes": state_changed_count,
            "duplicate_query_count": duplicate_query_count,
        },
        "metrics": {
            "mae": mae,
            "mse": mse,
            "oracle_mae": oracle_mae,
            "selected_expert_mae_mean": float(selected_error.mean()),
            "regret_to_oracle": mae - oracle_mae,
            "oracle_match_rate": float((selected_experts == oracle_best).to(torch.float32).mean()),
            "average_experts_queried": float(avg_queried),
            "latency_seconds": float(latency_seconds),
            "latency_ms_per_sample": float(latency_seconds * 1000.0 / max(num_windows, 1)),
            "stop_step_distribution": {
                str(step): int(count)
                for step, count in enumerate(stop_step_counts.tolist())
                if step > 0 and count
            },
            "selection_counts": {
                expert_names[index]: int(count)
                for index, count in enumerate(torch.bincount(selected_experts, minlength=num_experts).tolist())
            },
            "query_utilization_counts": {
                expert_names[index]: int(count)
                for index, count in enumerate(
                    torch.bincount(
                        torch.tensor([idx for seq in query_sequences for idx in seq], dtype=torch.long),
                        minlength=num_experts,
                    ).tolist()
                )
            },
            "all_queries_unique": duplicate_query_count == 0,
            "state_changed_for_every_query": state_changed_count == query_event_count,
            "stop_step_one_fraction": float(stop_step_counts[1] / max(num_windows, 1)) if len(stop_step_counts) > 1 else 0.0,
            "selected_expert_diagnostic": (
                "first_queried_expert_when_finalizer_is_equal_average"
                if finalizer == "equal_average"
                else "model_selected_expert"
            ),
        },
        "rollouts": detailed,
    }


def _write_outputs(payloads: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "rollout_summary.json"
    details_path = output_dir / "detailed_rollouts.json"
    csv_path = output_dir / "rollout_summary.csv"
    summary_payload = {
        "runs": [
            {
                "metadata": payload["metadata"],
                "metrics": payload["metrics"],
            }
            for payload in payloads
        ]
    }
    summary_path.write_text(json.dumps(_jsonable(summary_payload), indent=2), encoding="utf-8")
    details_path.write_text(json.dumps(_jsonable(payloads), indent=2), encoding="utf-8")
    fields = [
        "mode",
        "finalizer",
        "force_k",
        "temperature",
        "max_queries",
        "mae",
        "mse",
        "oracle_mae",
        "selected_expert_mae_mean",
        "regret_to_oracle",
        "oracle_match_rate",
        "average_experts_queried",
        "latency_seconds",
        "latency_ms_per_sample",
        "stop_step_distribution",
        "selection_counts",
        "query_utilization_counts",
        "all_queries_unique",
        "state_changed_for_every_query",
        "stop_step_one_fraction",
        "selected_expert_diagnostic",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for payload in payloads:
            row = {
                "mode": payload["metadata"]["mode"],
                "finalizer": payload["metadata"]["finalizer"],
                "force_k": payload["metadata"]["force_k"],
                "temperature": payload["metadata"]["temperature"],
                "max_queries": payload["metadata"]["max_queries"],
                **payload["metrics"],
            }
            row["stop_step_distribution"] = json.dumps(row["stop_step_distribution"], sort_keys=True)
            row["selection_counts"] = json.dumps(row["selection_counts"], sort_keys=True)
            row["query_utilization_counts"] = json.dumps(row["query_utilization_counts"], sort_keys=True)
            writer.writerow(row)
    print(f"Saved: {csv_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {details_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SubsetUtilityCOSTARTS sequential rollouts.")
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mode", choices=("greedy", "sampled", "forced", "all"), default="all")
    parser.add_argument(
        "--finalizer",
        choices=("equal_average", "reranker", "sparse_mixture", "all"),
        default="equal_average",
    )
    parser.add_argument("--force-k", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--detailed-limit", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    cache = _load_torch(Path(args.cache))
    router, checkpoint = _load_router(Path(args.checkpoint), device)
    modes = ("greedy", "sampled", "forced") if args.mode == "all" else (args.mode,)
    finalizers = ("equal_average", "reranker", "sparse_mixture") if args.finalizer == "all" else (args.finalizer,)
    payloads = []
    for mode in modes:
        if mode == "forced":
            force_values = (
                range(1, int(cache["max_subset_size"]) + 1)
                if args.force_k is None
                else (args.force_k,)
            )
        else:
            force_values = (args.force_k,)
        for force_k in force_values:
            for finalizer in finalizers:
                payload = evaluate_rollouts(
                    router=router,
                    cache=cache,
                    mode=mode,
                    finalizer=finalizer,
                    force_k=force_k,
                    temperature=args.temperature,
                    max_queries=args.max_queries,
                    batch_size=args.batch_size,
                    device=device,
                    seed=args.seed,
                    detailed_limit=args.detailed_limit,
                )
                payload["metadata"]["checkpoint"] = str(args.checkpoint)
                payload["metadata"]["checkpoint_epoch"] = int(checkpoint.get("epoch", -1))
                payloads.append(payload)
                metrics = payload["metrics"]
                print(
                    f"{mode} | final={finalizer} | K={force_k} | "
                    f"MAE={metrics['mae']:.6f} MSE={metrics['mse']:.6f} "
                    f"avg_q={metrics['average_experts_queried']:.3f} "
                    f"stop@1={metrics['stop_step_one_fraction']:.3f} "
                    f"unique={metrics['all_queries_unique']} "
                    f"state_updates={metrics['state_changed_for_every_query']}"
                )
    _write_outputs(payloads, Path(args.output_dir))


if __name__ == "__main__":
    main()
