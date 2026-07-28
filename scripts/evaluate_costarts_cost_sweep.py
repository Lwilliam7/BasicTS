"""Cost sweep and stopping calibration for subset-utility COSTARTS.

The evaluator keeps the trained router fixed and applies query costs at
decision time. It starts from the empty state, queries while the best predicted
cost-adjusted utility is positive, and stops once no remaining query appears
worth its cost.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch

try:
    from scripts.costars.build_costarts_subset_states import validate_costarts_subset_states
    from scripts.train_costarts_subset_utility_router import (
        SubsetUtilityCOSTARTSRouter,
        _build_state_lookup,
        _state_batch,
        set_reproducible_seed,
    )
except ImportError:
    from scripts.costars.build_costarts_subset_states import validate_costarts_subset_states
    from train_costarts_subset_utility_router import (
        SubsetUtilityCOSTARTSRouter,
        _build_state_lookup,
        _state_batch,
        set_reproducible_seed,
    )


DEFAULT_CACHE = "cache/costarts_subset_states_val.pt"
DEFAULT_CHECKPOINT = "checkpoints/costarts_subset_utility/best_subset_utility_costarts_router.pt"
DEFAULT_OUTPUT_DIR = "results/router_summary/costarts_subset_utility"
DEFAULT_LAMBDAS = "0,0.0005,0.001,0.0025,0.005,0.01,0.02,0.05,0.1,0.2,0.5"


def _load_torch(path: Path) -> dict[str, Any]:
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


def _load_router(checkpoint_path: Path, device: torch.device) -> tuple[SubsetUtilityCOSTARTSRouter, dict[str, Any]]:
    checkpoint = _load_torch(checkpoint_path)
    router = SubsetUtilityCOSTARTSRouter(**checkpoint["router_config"]).to(device)
    router.load_state_dict(checkpoint["router_state_dict"])
    router.eval()
    return router, checkpoint


def _parse_lambdas(value: str) -> tuple[float, ...]:
    lambdas = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not lambdas:
        raise ValueError("At least one lambda value is required.")
    if any(item < 0 for item in lambdas):
        raise ValueError("Lambda values must be non-negative.")
    if 0.0 not in lambdas:
        lambdas = (0.0,) + lambdas
    return tuple(sorted(set(lambdas)))


def _parse_costs(value: Optional[str], expert_names: Sequence[str]) -> torch.Tensor:
    if not value:
        return torch.ones(len(expert_names), dtype=torch.float32)
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("--expert-costs-json must be a JSON object.")
    costs = []
    for name in expert_names:
        cost = float(payload.get(name, payload.get(f"Candidate_{name}", 1.0)))
        if cost < 0:
            raise ValueError(f"Expert cost for {name} must be non-negative.")
        costs.append(cost)
    return torch.tensor(costs, dtype=torch.float32)


def _mae_mse(prediction: torch.Tensor, target: torch.Tensor, target_mask: torch.Tensor) -> tuple[float, float]:
    mask = target_mask.to(torch.float32)
    denominator = mask.sum().clamp_min(1.0)
    mae = (torch.abs(prediction - target) * mask).sum() / denominator
    mse = ((prediction - target).pow(2) * mask).sum() / denominator
    return float(mae), float(mse)


def _bit_indices(mask_int: int, num_experts: int) -> list[int]:
    return [index for index in range(num_experts) if mask_int & (1 << index)]


def _calibration_metrics(counts: Mapping[str, int]) -> dict[str, float]:
    tp = float(counts["true_stop_pred_stop"])
    fp = float(counts["true_continue_pred_stop"])
    tn = float(counts["true_continue_pred_continue"])
    fn = float(counts["true_stop_pred_continue"])
    return {
        "false_stop_rate": fp / max(fp + tn, 1.0),
        "false_continue_rate": fn / max(fn + tp, 1.0),
        "stop_precision": tp / max(tp + fp, 1.0),
        "stop_recall": tp / max(tp + fn, 1.0),
    }


def _finalize_predictions(
    *,
    router: SubsetUtilityCOSTARTSRouter,
    cache: Mapping[str, Any],
    final_state_indices: Sequence[int],
    batch_size: int,
    device: torch.device,
    finalizer: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_windows = len(final_state_indices)
    num_experts = int(cache["num_experts"])
    if finalizer not in {"reranker", "sparse_mixture", "oracle_best_queried"}:
        raise ValueError("finalizer must be reranker, sparse_mixture, or oracle_best_queried")

    selected_experts = torch.empty(num_windows, dtype=torch.long)
    predictions = []
    targets = []
    masks = []
    for offset in range(0, num_windows, batch_size):
        rows = list(range(offset, min(offset + batch_size, num_windows)))
        batch = _state_batch(cache, final_state_indices[offset : offset + len(rows)], device)
        outputs = router(
            batch["history"],
            batch["queried_mask"],
            batch["queried_expert_ids"],
            batch["queried_expert_forecasts"],
        )
        queried_mask = batch["queried_mask"].detach().cpu()
        queried_ids = batch["queried_expert_ids"].detach().cpu()
        queried_forecasts = batch["queried_expert_forecasts"].detach().cpu()
        targets.append(batch["true_targets"].detach().cpu())
        masks.append(batch["target_mask"].detach().cpu())

        if finalizer == "oracle_best_queried":
            errors = batch["true_expert_error_vector"].detach().cpu().masked_fill(~queried_mask, float("inf"))
            selected = torch.argmin(errors, dim=-1)
            positions = (queried_ids == selected[:, None]).to(torch.float32).argmax(dim=1)
            prediction = queried_forecasts[torch.arange(len(rows)), positions]
        elif finalizer == "reranker":
            scores = outputs["expert_score"].detach().cpu().masked_fill(~queried_mask, -1e9)
            selected = torch.argmax(scores, dim=-1)
            positions = (queried_ids == selected[:, None]).to(torch.float32).argmax(dim=1)
            prediction = queried_forecasts[torch.arange(len(rows)), positions]
        else:
            valid_slots = queried_ids >= 0
            logits = outputs["mix_logits"].detach().cpu().gather(1, queried_ids.clamp_min(0))
            logits = logits.masked_fill(~valid_slots, -1e9)
            slot_weights = torch.softmax(logits, dim=1).masked_fill(~valid_slots, 0.0)
            slot_weights = slot_weights / slot_weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
            prediction = (queried_forecasts * slot_weights[:, :, None, None]).sum(dim=1)
            full_weights = torch.zeros(len(rows), num_experts)
            for row_index in range(len(rows)):
                for slot_index, expert_index in enumerate(queried_ids[row_index].tolist()):
                    if expert_index >= 0:
                        full_weights[row_index, int(expert_index)] = slot_weights[row_index, slot_index]
            selected = torch.argmax(full_weights, dim=-1)

        selected_experts[offset : offset + len(rows)] = selected.to(torch.long)
        predictions.append(prediction)

    return (
        torch.cat(predictions, dim=0),
        torch.cat(targets, dim=0),
        torch.cat(masks, dim=0),
        selected_experts,
    )


@torch.no_grad()
def evaluate_cost_lambda(
    *,
    router: SubsetUtilityCOSTARTSRouter,
    cache: Mapping[str, Any],
    query_lambda: float,
    expert_costs: torch.Tensor,
    batch_size: int,
    device: torch.device,
    finalizer: str,
    max_queries: Optional[int],
) -> dict[str, Any]:
    validate_costarts_subset_states(cache)
    if cache["split_role"] != "router_val":
        raise ValueError("Cost sweep must use router_val subset-state cache.")
    if cache["subset_sampling_mode"] != "exhaustive":
        raise ValueError("Cost sweep requires exhaustive subset-state cache.")

    router.eval()
    lookup = _build_state_lookup(cache)
    num_windows = int(cache["num_source_windows"])
    num_experts = int(cache["num_experts"])
    cache_max_subset = int(cache["max_subset_size"])
    max_queries = cache_max_subset if max_queries is None else min(int(max_queries), cache_max_subset)
    if max_queries <= 0:
        raise ValueError("max_queries must be positive.")

    costs_cpu = expert_costs.detach().cpu().to(torch.float32)
    costs_device = costs_cpu.to(device)
    masks = [0 for _ in range(num_windows)]
    done = [False for _ in range(num_windows)]
    query_sequences: list[list[int]] = [[] for _ in range(num_windows)]
    stop_step_counts = torch.zeros(max_queries + 1, dtype=torch.long)
    calibration_counts = {
        "true_stop_pred_stop": 0,
        "true_stop_pred_continue": 0,
        "true_continue_pred_stop": 0,
        "true_continue_pred_continue": 0,
    }

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
            queried_mask = batch["queried_mask"].to(torch.bool)
            remaining_mask = ~queried_mask
            has_queried = queried_mask.any(dim=1)
            if step + 1 >= max_queries:
                remaining_mask = torch.zeros_like(remaining_mask)
            predicted_utility = outputs["utility_prediction"] - float(query_lambda) * costs_device.view(1, -1)
            predicted_utility = predicted_utility.masked_fill(~remaining_mask, -1e9)
            best_predicted_utility, best_actions = predicted_utility.max(dim=1)
            predicted_stop = has_queried & ((best_predicted_utility <= 0.0) | ~remaining_mask.any(dim=1))

            true_marginal = batch["marginal_gain_best_queried_oracle"] if "marginal_gain_best_queried_oracle" in batch else None
            if true_marginal is None:
                state_index_tensor = torch.tensor(state_indices, dtype=torch.long)
                true_marginal = cache["marginal_gain_best_queried_oracle"][state_index_tensor].to(device)
            true_utility = true_marginal - float(query_lambda) * costs_device.view(1, -1)
            true_utility = true_utility.masked_fill(~remaining_mask, -1e9)
            best_true_utility = true_utility.max(dim=1).values
            oracle_stop = has_queried & ((best_true_utility <= 0.0) | ~remaining_mask.any(dim=1))

            for local_index, sample_index in enumerate(rows):
                if bool(has_queried[local_index]):
                    pred_stop = bool(predicted_stop[local_index])
                    truth_stop = bool(oracle_stop[local_index])
                    if truth_stop and pred_stop:
                        calibration_counts["true_stop_pred_stop"] += 1
                    elif truth_stop and not pred_stop:
                        calibration_counts["true_stop_pred_continue"] += 1
                    elif not truth_stop and pred_stop:
                        calibration_counts["true_continue_pred_stop"] += 1
                    else:
                        calibration_counts["true_continue_pred_continue"] += 1

                action = int(best_actions[local_index].detach().cpu())
                if bool(predicted_stop[local_index]):
                    done[sample_index] = True
                    stop_count_step = max(1, len(query_sequences[sample_index]))
                    stop_step_counts[min(stop_count_step, max_queries)] += 1
                    continue
                if action in query_sequences[sample_index]:
                    done[sample_index] = True
                    stop_count_step = max(1, len(query_sequences[sample_index]))
                    stop_step_counts[min(stop_count_step, max_queries)] += 1
                    continue
                query_sequences[sample_index].append(action)
                masks[sample_index] |= 1 << action
                if len(query_sequences[sample_index]) >= max_queries:
                    done[sample_index] = True
                    stop_step_counts[len(query_sequences[sample_index])] += 1

    for sample_index in range(num_windows):
        if not query_sequences[sample_index]:
            state_index = lookup[sample_index][0]
            errors = cache["true_expert_error_vector"][state_index]
            action = int(torch.argmin(errors).item())
            query_sequences[sample_index].append(action)
            masks[sample_index] |= 1 << action
            stop_step_counts[1] += 1

    final_state_indices = [lookup[row][masks[row]] for row in range(num_windows)]
    prediction, targets, target_masks, selected_experts = _finalize_predictions(
        router=router,
        cache=cache,
        final_state_indices=final_state_indices,
        batch_size=batch_size,
        device=device,
        finalizer=finalizer,
    )
    mae, mse = _mae_mse(prediction, targets, target_masks)

    source_errors = torch.empty(num_windows, num_experts, dtype=torch.float32)
    for row in range(num_windows):
        source_errors[row] = cache["true_expert_error_vector"][lookup[row][0]]
    oracle_best = torch.argmin(source_errors, dim=1)
    oracle_mae = float(source_errors.min(dim=1).values.mean())
    selected_error = source_errors.gather(1, selected_experts.view(-1, 1)).squeeze(1)
    avg_queries = sum(len(sequence) for sequence in query_sequences) / max(num_windows, 1)
    calibration = _calibration_metrics(calibration_counts)
    selection_counts = torch.bincount(selected_experts, minlength=num_experts)
    query_counts = torch.bincount(
        torch.tensor([index for sequence in query_sequences for index in sequence], dtype=torch.long),
        minlength=num_experts,
    )
    expert_names = tuple(cache["expert_names"])
    return {
        "lambda": float(query_lambda),
        "mae": mae,
        "mse": mse,
        "oracle_mae": oracle_mae,
        "regret_to_oracle": mae - oracle_mae,
        "selected_expert_mae_mean": float(selected_error.mean()),
        "oracle_match_rate": float((selected_experts == oracle_best).to(torch.float32).mean()),
        "average_experts_queried": float(avg_queries),
        "windows_with_more_than_one_expert": int(sum(1 for sequence in query_sequences if len(sequence) > 1)),
        "fraction_windows_with_more_than_one_expert": float(
            sum(1 for sequence in query_sequences if len(sequence) > 1) / max(num_windows, 1)
        ),
        "stop_step_distribution": {
            str(step): int(count)
            for step, count in enumerate(stop_step_counts.tolist())
            if step > 0 and count
        },
        "selection_counts": {
            expert_names[index]: int(count)
            for index, count in enumerate(selection_counts.tolist())
        },
        "query_utilization_counts": {
            expert_names[index]: int(count)
            for index, count in enumerate(query_counts.tolist())
        },
        "calibration_counts": calibration_counts,
        **calibration,
    }


def write_outputs(rows: list[dict[str, Any]], *, output_dir: Path, metadata: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "cost_sweep.csv"
    pareto_path = output_dir / "pareto_curve.json"
    fieldnames = [
        "lambda",
        "mae",
        "mse",
        "oracle_mae",
        "regret_to_oracle",
        "selected_expert_mae_mean",
        "oracle_match_rate",
        "average_experts_queried",
        "windows_with_more_than_one_expert",
        "fraction_windows_with_more_than_one_expert",
        "false_stop_rate",
        "false_continue_rate",
        "stop_precision",
        "stop_recall",
        "stop_step_distribution",
        "selection_counts",
        "query_utilization_counts",
        "calibration_counts",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writable = dict(row)
            for key in (
                "stop_step_distribution",
                "selection_counts",
                "query_utilization_counts",
                "calibration_counts",
            ):
                writable[key] = json.dumps(writable[key], sort_keys=True)
            writer.writerow(writable)
    pareto = {
        "metadata": dict(metadata),
        "points": [
            {
                "lambda": row["lambda"],
                "mae": row["mae"],
                "mse": row["mse"],
                "average_experts_queried": row["average_experts_queried"],
                "false_stop_rate": row["false_stop_rate"],
                "false_continue_rate": row["false_continue_rate"],
                "stop_precision": row["stop_precision"],
                "stop_recall": row["stop_recall"],
                "fraction_windows_with_more_than_one_expert": row[
                    "fraction_windows_with_more_than_one_expert"
                ],
            }
            for row in rows
        ],
    }
    pareto_path.write_text(json.dumps(_jsonable(pareto), indent=2), encoding="utf-8")
    print(f"Saved: {csv_path}")
    print(f"Saved: {pareto_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate COSTARTS subset router cost sweeps.")
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--lambdas", default=DEFAULT_LAMBDAS)
    parser.add_argument(
        "--expert-costs-json",
        default=None,
        help='Optional JSON object, e.g. {"DLinear":1.0,"PatchTST":1.5}. Defaults to uniform cost 1.',
    )
    parser.add_argument("--finalizer", choices=("reranker", "sparse_mixture", "oracle_best_queried"), default="reranker")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_reproducible_seed(args.seed)
    device = torch.device(args.device)
    cache = _load_torch(Path(args.cache))
    router, checkpoint = _load_router(Path(args.checkpoint), device)
    validate_costarts_subset_states(cache)
    expert_names = tuple(cache["expert_names"])
    costs = _parse_costs(args.expert_costs_json, expert_names)
    lambdas = _parse_lambdas(args.lambdas)
    print("COSTARTS calibrated query-cost sweep")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  checkpoint_epoch: {checkpoint.get('epoch', -1)}")
    print(f"  cache: {args.cache}")
    print(f"  finalizer: {args.finalizer}")
    print(f"  lambdas: {lambdas}")
    print(f"  expert_costs: {dict(zip(expert_names, costs.tolist()))}")

    rows = []
    for value in lambdas:
        row = evaluate_cost_lambda(
            router=router,
            cache=cache,
            query_lambda=value,
            expert_costs=costs,
            batch_size=args.batch_size,
            device=device,
            finalizer=args.finalizer,
            max_queries=args.max_queries,
        )
        rows.append(row)
        print(
            f"lambda={value:.6g} "
            f"MAE={row['mae']:.6f} "
            f"avg_q={row['average_experts_queried']:.3f} "
            f">1={row['fraction_windows_with_more_than_one_expert']:.3f} "
            f"false_stop={row['false_stop_rate']:.3f} "
            f"false_continue={row['false_continue_rate']:.3f}"
        )

    avg_queries = [row["average_experts_queried"] for row in rows]
    smooth_nonincreasing = all(
        later <= earlier + 1e-9
        for earlier, later in zip(avg_queries, avg_queries[1:])
    )
    if rows[0]["windows_with_more_than_one_expert"] <= 0:
        print("WARNING: lambda=0 did not query more than one expert on any validation window.")
    if not smooth_nonincreasing:
        print("WARNING: average queries were not monotonic across lambda values.")

    write_outputs(
        rows,
        output_dir=Path(args.output_dir),
        metadata={
            "checkpoint": args.checkpoint,
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "cache": args.cache,
            "finalizer": args.finalizer,
            "expert_names": expert_names,
            "expert_costs": dict(zip(expert_names, costs.tolist())),
            "zero_cost_queries_more_than_one_window": rows[0]["windows_with_more_than_one_expert"] > 0,
            "average_queries_nonincreasing": smooth_nonincreasing,
            "test_data_used": False,
        },
    )


if __name__ == "__main__":
    main()
