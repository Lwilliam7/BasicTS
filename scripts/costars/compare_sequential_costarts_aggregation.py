"""Compare sequential COSTARTS final aggregation rules on router validation.

The sequential router and its validation-selected stop threshold are held fixed.
This script changes only the final forecast aggregation over the experts that
the router already queried.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.costars.train_costarts_subset_utility_router import _load_torch
from scripts.costars.train_sequential_costarts import (
    DEFAULT_SOURCE_TRAIN_CACHE,
    DEFAULT_TRAIN_CACHE,
    DEFAULT_VAL_CACHE,
    SequentialCOSTARTSRouter,
    build_state_lookup,
    mae_mse_per_window,
    masked_utility_scores,
    state_batch,
    validate_sequential_caches,
)


DEFAULT_CHECKPOINT_ROOT = "checkpoints/costarts_sequential"
DEFAULT_RESULTS_DIR = "results/router_summary/costarts_sequential/aggregation_compare"
DEFAULT_SEEDS = (7, 11, 13, 17, 19)


def json_default(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"Cannot JSON encode {type(value)!r}")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_seeds(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def load_router_from_checkpoint(path: Path, device: torch.device) -> tuple[SequentialCOSTARTSRouter, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("router_type") != "sequential_costarts":
        raise ValueError(f"{path} is not a sequential_costarts checkpoint")
    config = checkpoint["router_config"]
    router = SequentialCOSTARTSRouter(
        num_experts=int(config["num_experts"]),
        max_subset_size=int(config["max_subset_size"]),
        input_len=int(config["input_len"]),
        forecast_horizon=int(config["forecast_horizon"]),
        num_features=int(config["num_features"]),
        hidden_dim=int(config["hidden_dim"]),
        embedding_dim=int(config["embedding_dim"]),
    ).to(device)
    router.load_state_dict(checkpoint["router_state_dict"])
    router.eval()
    if checkpoint.get("test_set_used") is not False:
        raise ValueError(f"{path} reports test_set_used={checkpoint.get('test_set_used')!r}")
    return router, checkpoint


@torch.no_grad()
def route_final_state_indices(
    router: SequentialCOSTARTSRouter,
    cache: Mapping[str, Any],
    *,
    fixed_first_expert: int,
    threshold: float,
    max_query_count: int,
    batch_size: int,
    device: torch.device,
) -> tuple[list[int], list[list[int]]]:
    if cache["split_role"] != "router_val":
        raise ValueError("Aggregation comparison may only route router_val")
    lookup = build_state_lookup(cache)
    num_windows = int(cache["num_source_windows"])
    num_experts = int(cache["num_experts"])
    masks = [1 << fixed_first_expert for _ in range(num_windows)]
    histories = [[fixed_first_expert] for _ in range(num_windows)]
    done = [False for _ in range(num_windows)]
    for _ in range(1, min(max_query_count, num_experts)):
        active = [row for row in range(num_windows) if not done[row]]
        if not active:
            break
        for offset in range(0, len(active), batch_size):
            rows = active[offset : offset + batch_size]
            state_indices = [lookup[row][masks[row]] for row in rows]
            batch = state_batch(cache, state_indices, device)
            scores = router(
                batch["history"],
                batch["queried_mask"],
                batch["queried_expert_ids"],
                batch["queried_expert_forecasts"],
            )
            scores = masked_utility_scores(scores, batch["queried_mask"])
            best_scores, actions = scores.max(dim=1)
            for local, row in enumerate(rows):
                action = int(actions[local].detach().cpu().item())
                score = float(best_scores[local].detach().cpu().item())
                if score <= threshold or len(histories[row]) >= max_query_count:
                    done[row] = True
                    continue
                histories[row].append(action)
                masks[row] |= 1 << action
                if len(histories[row]) >= max_query_count:
                    done[row] = True
    return [lookup[row][masks[row]] for row in range(num_windows)], histories


def weighted_average_forecast_from_state(batch: Mapping[str, torch.Tensor], global_weights: torch.Tensor) -> torch.Tensor:
    ids = batch["queried_expert_ids"]
    forecasts = batch["queried_expert_forecasts"]
    valid = ids >= 0
    safe_ids = ids.clamp_min(0)
    weights = global_weights.to(forecasts.device, forecasts.dtype)[safe_ids] * valid.to(forecasts.dtype)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return (forecasts * weights[:, :, None, None]).sum(dim=1)


def evaluate_final_states(
    cache: Mapping[str, Any],
    final_state_indices: Sequence[int],
    *,
    global_weights: Optional[torch.Tensor],
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    maes = []
    mses = []
    for offset in range(0, len(final_state_indices), batch_size):
        state_indices = final_state_indices[offset : offset + batch_size]
        batch = state_batch(cache, state_indices, device)
        if global_weights is None:
            prediction = weighted_average_forecast_from_state(
                batch,
                torch.ones(int(cache["num_experts"]), dtype=torch.float32, device=device),
            )
        else:
            prediction = weighted_average_forecast_from_state(batch, global_weights.to(device))
        mae, mse = mae_mse_per_window(prediction, batch["true_targets"], batch["target_mask"])
        maes.append(mae.cpu())
        mses.append(mse.cpu())
    mae_all = torch.cat(maes)
    mse_all = torch.cat(mses)
    return {
        "validation_mae": float(mae_all.mean().item()),
        "validation_mse": float(mse_all.mean().item()),
    }


def fit_global_convex_weights(
    source_train: Mapping[str, Any],
    *,
    steps: int,
    learning_rate: float,
    device: torch.device,
) -> torch.Tensor:
    if source_train["split_role"] != "router_train":
        raise ValueError("Global aggregation weights must be learned on router_train")
    predictions = source_train["prediction_stack"].to(device=device, dtype=torch.float32)
    targets = source_train["targets"].to(device=device, dtype=torch.float32)
    masks = source_train["target_masks"].to(device=device, dtype=torch.bool)
    num_experts = int(predictions.shape[-1])
    logits = torch.zeros(num_experts, dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=learning_rate)
    mask_float = masks.to(torch.float32)
    denom = mask_float.sum().clamp_min(1.0)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        weights = torch.softmax(logits, dim=0)
        forecast = torch.einsum("nhfe,e->nhf", predictions, weights)
        loss = ((forecast - targets).abs() * mask_float).sum() / denom
        loss.backward()
        optimizer.step()
    return torch.softmax(logits.detach().cpu(), dim=0)


def train_error_softmax_weights(source_train: Mapping[str, Any], temperature: float) -> torch.Tensor:
    if source_train["split_role"] != "router_train":
        raise ValueError("Train-error weights require router_train")
    expert_mae = source_train["error_matrix"].to(torch.float32).mean(dim=0)
    if temperature <= 0:
        index = int(expert_mae.argmin().item())
        weights = torch.zeros_like(expert_mae)
        weights[index] = 1.0
        return weights
    return torch.softmax(-expert_mae / float(temperature), dim=0)


def query_count_stats(histories: Sequence[Sequence[int]], num_experts: int) -> dict[str, Any]:
    counts = torch.tensor([len(history) for history in histories], dtype=torch.long)
    return {
        "average_experts_queried": float(counts.to(torch.float32).mean().item()),
        "query_count_distribution": {
            str(index): int((counts == index).sum().item())
            for index in range(1, num_experts + 1)
            if int((counts == index).sum().item())
        },
    }


def compare_seed(
    seed: int,
    checkpoint_root: Path,
    val_cache: Mapping[str, Any],
    source_train: Mapping[str, Any],
    learned_weights: torch.Tensor,
    *,
    temperatures: Sequence[float],
    batch_size: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checkpoint_path = checkpoint_root / f"seed_{seed}" / "best_sequential_costarts_router.pt"
    router, checkpoint = load_router_from_checkpoint(checkpoint_path, device)
    threshold = float(checkpoint["selected_stop_threshold"])
    fixed_first = int(checkpoint["fixed_first_expert"])
    max_query_count = int(checkpoint["router_config"]["max_subset_size"])
    final_state_indices, histories = route_final_state_indices(
        router,
        val_cache,
        fixed_first_expert=fixed_first,
        threshold=threshold,
        max_query_count=max_query_count,
        batch_size=batch_size,
        device=device,
    )
    query_stats = query_count_stats(histories, int(val_cache["num_experts"]))
    rows = []
    equal = evaluate_final_states(val_cache, final_state_indices, global_weights=None, batch_size=batch_size, device=device)
    rows.append({"seed": seed, "aggregation": "equal_average", "selected_temperature": "", **equal, **query_stats})
    learned = evaluate_final_states(val_cache, final_state_indices, global_weights=learned_weights, batch_size=batch_size, device=device)
    rows.append({"seed": seed, "aggregation": "learned_global_convex_train", "selected_temperature": "", **learned, **query_stats})

    temperature_rows = []
    best_temp_row = None
    for temperature in temperatures:
        weights = train_error_softmax_weights(source_train, temperature)
        metrics = evaluate_final_states(val_cache, final_state_indices, global_weights=weights, batch_size=batch_size, device=device)
        row = {
            "seed": seed,
            "temperature": temperature,
            "validation_mae": metrics["validation_mae"],
            "validation_mse": metrics["validation_mse"],
            "weights": weights.tolist(),
        }
        temperature_rows.append(row)
        if best_temp_row is None or row["validation_mae"] < best_temp_row["validation_mae"] - 1e-12:
            best_temp_row = row
    assert best_temp_row is not None
    rows.append({
        "seed": seed,
        "aggregation": "validation_selected_train_error_softmax",
        "selected_temperature": best_temp_row["temperature"],
        "validation_mae": best_temp_row["validation_mae"],
        "validation_mse": best_temp_row["validation_mse"],
        **query_stats,
    })
    details = {
        "seed": seed,
        "checkpoint": str(checkpoint_path),
        "threshold": threshold,
        "fixed_first_expert": checkpoint["fixed_first_expert_name"],
        "query_stats": query_stats,
        "temperature_search": temperature_rows,
    }
    return rows, details


def aggregate(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    names = sorted({str(row["aggregation"]) for row in rows})
    aggregate_rows = []
    for name in names:
        subset = [row for row in rows if row["aggregation"] == name]
        for metric in ("validation_mae", "validation_mse", "average_experts_queried"):
            values = np.array([float(row[metric]) for row in subset], dtype=float)
            aggregate_rows.append({
                "aggregation": name,
                "metric": metric,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "min": float(values.min()),
                "max": float(values.max()),
            })
    return aggregate_rows


def write_report(path: Path, rows: Sequence[Mapping[str, Any]], aggregate_rows: Sequence[Mapping[str, Any]]) -> None:
    mae_rows = {row["aggregation"]: row for row in aggregate_rows if row["metric"] == "validation_mae"}
    best = min(mae_rows.items(), key=lambda item: item[1]["mean"])
    lines = [
        "# Sequential COSTARTS Aggregation Comparison",
        "",
        "The router decisions, thresholds, and selected expert subsets are fixed from the existing sequential COSTARTS checkpoints. Only the final aggregation over queried forecasts changes.",
        "",
        "## Mean Validation MAE",
        "",
    ]
    for name, row in sorted(mae_rows.items()):
        lines.append(f"- `{name}`: `{row['mean']:.6f}` +/- `{row['std']:.6f}`")
    lines.extend([
        "",
        "## Winner",
        "",
        f"`{best[0]}` has the lowest mean validation MAE.",
        "",
        "## Leakage Note",
        "",
        "No final test cache is loaded. Learned convex weights are fit on `router_train`; train-error softmax temperatures are selected on `router_val`, so that row is a validation diagnostic and should not be treated as locked-test evidence.",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", default=DEFAULT_TRAIN_CACHE)
    parser.add_argument("--val-cache", default=DEFAULT_VAL_CACHE)
    parser.add_argument("--source-train-cache", default=DEFAULT_SOURCE_TRAIN_CACHE)
    parser.add_argument("--checkpoint-root", default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--weight-steps", type=int, default=750)
    parser.add_argument("--weight-learning-rate", type=float, default=0.05)
    parser.add_argument("--temperatures", default="0,0.0025,0.005,0.01,0.025,0.05,0.1,0.25,0.5,1.0")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> dict[str, Any]:
    args = parse_args(argv)
    device = torch.device(args.device)
    train_cache = _load_torch(args.train_cache)
    val_cache = _load_torch(args.val_cache)
    source_train = _load_torch(args.source_train_cache)
    validate_sequential_caches(train_cache, val_cache)
    if val_cache["split_role"] != "router_val" or source_train["split_role"] != "router_train":
        raise ValueError("Aggregation comparison is restricted to router_train/router_val")
    learned_weights = fit_global_convex_weights(
        source_train,
        steps=args.weight_steps,
        learning_rate=args.weight_learning_rate,
        device=device,
    )
    temperatures = [float(item.strip()) for item in args.temperatures.split(",") if item.strip()]
    all_rows = []
    details = {
        "learned_global_convex_train_weights": learned_weights.tolist(),
        "expert_names": list(source_train["expert_names"]),
        "seeds": parse_seeds(args.seeds),
        "test_set_used": False,
        "note": "Router decisions are fixed; only final aggregation changes.",
        "per_seed": [],
    }
    for seed in parse_seeds(args.seeds):
        rows, seed_details = compare_seed(
            seed,
            Path(args.checkpoint_root),
            val_cache,
            source_train,
            learned_weights,
            temperatures=temperatures,
            batch_size=args.batch_size,
            device=device,
        )
        all_rows.extend(rows)
        details["per_seed"].append(seed_details)
    aggregate_rows = aggregate(all_rows)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    write_csv(results_dir / "per_seed_aggregation_results.csv", all_rows)
    write_csv(results_dir / "aggregate_aggregation_results.csv", aggregate_rows)
    summary = {
        "per_seed": all_rows,
        "aggregate": aggregate_rows,
        "details": details,
    }
    (results_dir / "aggregation_comparison_summary.json").write_text(
        json.dumps(summary, indent=2, default=json_default),
        encoding="utf-8",
    )
    write_report(results_dir / "aggregation_comparison_report.md", all_rows, aggregate_rows)
    return summary


if __name__ == "__main__":
    main()
