"""Train full Sequential COSTAR-TS on walk-forward OOS router caches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_costarts_walkforward_cache import EXPERT_ORDER, sha256_file, validate_walkforward_cache
from scripts.sequential_costarts_model_full import SequentialCOSTARTSRouterFull


@dataclass
class WalkForwardRouterTrainingConfig:
    train_cache: str = "cache/costarts_walkforward/router_train_20_60_cache.pt"
    val_cache: str = "cache/costarts_walkforward/router_val_60_80_cache.pt"
    checkpoint_root: str = "checkpoints/costarts_walkforward/sequential_full"
    results_root: str = "results/router_summary/costarts_walkforward/sequential_full"
    seeds: tuple[int, ...] = (7, 11, 13, 17, 19)
    batch_size: int = 256
    max_epochs: int = 50
    patience: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    embedding_dim: int = 64
    hidden_dim: int = 64
    max_queries: int = 5
    query_cost: float = 0.0
    query_threshold: float = 0.0
    device: str = "cpu"


class CacheWindowDataset(Dataset):
    def __init__(self, cache: Mapping[str, Any]) -> None:
        self.cache = cache
        self.num_windows = int(cache["num_windows"])

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "history": self.cache["histories"][index],
            "targets": self.cache["targets"][index],
            "target_masks": self.cache["target_masks"][index],
            "prediction_stack": self.cache["prediction_stack"][index],
            "absolute_window_start": self.cache["absolute_window_starts"][index],
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_mae(prediction: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    mask = masks.to(prediction.dtype)
    denom = mask.flatten(1).sum(dim=1).clamp_min(1.0)
    return ((prediction - targets).abs() * mask).flatten(1).sum(dim=1) / denom


def sample_mse(prediction: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    mask = masks.to(prediction.dtype)
    denom = mask.flatten(1).sum(dim=1).clamp_min(1.0)
    return ((prediction - targets).square() * mask).flatten(1).sum(dim=1) / denom


def current_average_from_ids(prediction_stack: torch.Tensor, queried_ids: torch.Tensor) -> torch.Tensor:
    batch, horizon, features, _ = prediction_stack.shape
    valid = queried_ids >= 0
    if not bool(valid.any()):
        return torch.zeros((batch, horizon, features), dtype=prediction_stack.dtype, device=prediction_stack.device)
    gathered = torch.zeros(
        (batch, queried_ids.shape[1], horizon, features),
        dtype=prediction_stack.dtype,
        device=prediction_stack.device,
    )
    for slot in range(queried_ids.shape[1]):
        ids = queried_ids[:, slot].clamp_min(0)
        gathered[:, slot] = prediction_stack[torch.arange(batch, device=prediction_stack.device), :, :, ids]
    gathered = gathered * valid[:, :, None, None].to(prediction_stack.dtype)
    denom = valid.sum(dim=1).clamp_min(1).to(prediction_stack.dtype)
    return gathered.sum(dim=1) / denom[:, None, None]


def make_state(
    prediction_stack: torch.Tensor,
    queried_ids: torch.Tensor,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, horizon, features, _ = prediction_stack.shape
    device = prediction_stack.device
    queried_mask = torch.zeros((batch, num_experts), dtype=prediction_stack.dtype, device=device)
    valid = queried_ids >= 0
    for row in range(batch):
        ids = queried_ids[row, valid[row]]
        if ids.numel() > 0:
            queried_mask[row, ids] = 1.0
    queried_forecasts = torch.zeros(
        (batch, queried_ids.shape[1], horizon, features),
        dtype=prediction_stack.dtype,
        device=device,
    )
    for slot in range(queried_ids.shape[1]):
        ids = queried_ids[:, slot].clamp_min(0)
        queried_forecasts[:, slot] = prediction_stack[torch.arange(batch, device=device), :, :, ids]
    queried_forecasts = queried_forecasts * valid[:, :, None, None].to(prediction_stack.dtype)
    current_average = current_average_from_ids(prediction_stack, queried_ids)
    return queried_mask, queried_forecasts, current_average


def greedy_oracle_order(prediction_stack: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    batch, _, _, num_experts = prediction_stack.shape
    chosen = torch.full((batch, num_experts), -1, dtype=torch.long, device=prediction_stack.device)
    used = torch.zeros((batch, num_experts), dtype=torch.bool, device=prediction_stack.device)
    current_ids = torch.full((batch, num_experts), -1, dtype=torch.long, device=prediction_stack.device)
    for step in range(num_experts):
        candidate_maes = []
        for expert_id in range(num_experts):
            ids = current_ids.clone()
            ids[:, step] = expert_id
            candidate_prediction = current_average_from_ids(prediction_stack, ids)
            candidate_maes.append(sample_mae(candidate_prediction, targets, masks))
        score = -torch.stack(candidate_maes, dim=1)
        score = score.masked_fill(used, -1e9)
        next_id = score.argmax(dim=1)
        chosen[:, step] = next_id
        current_ids[:, step] = next_id
        used[torch.arange(batch, device=prediction_stack.device), next_id] = True
    return chosen


def utility_targets(
    prediction_stack: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    queried_ids: torch.Tensor,
    query_cost: float,
) -> torch.Tensor:
    batch, _, _, num_experts = prediction_stack.shape
    current_count = (queried_ids >= 0).sum(dim=1)
    current_prediction = current_average_from_ids(prediction_stack, queried_ids)
    current_mae = sample_mae(current_prediction, targets, masks)
    targets_out = []
    for expert_id in range(num_experts):
        next_ids = queried_ids.clone()
        insert_slot = current_count.clamp(max=queried_ids.shape[1] - 1)
        next_ids[torch.arange(batch, device=prediction_stack.device), insert_slot] = expert_id
        candidate_prediction = current_average_from_ids(prediction_stack, next_ids)
        candidate_mae = sample_mae(candidate_prediction, targets, masks)
        if int(current_count.max().item()) == 0 and int(current_count.min().item()) == 0:
            utility = -candidate_mae
        else:
            utility = current_mae - candidate_mae - float(query_cost)
        targets_out.append(utility)
    target = torch.stack(targets_out, dim=1)
    for row in range(batch):
        for expert_id in queried_ids[row][queried_ids[row] >= 0].tolist():
            target[row, expert_id] = 0.0
    return target


def train_one_epoch(
    model: SequentialCOSTARTSRouterFull,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_queries: int,
    query_cost: float,
    grad_clip_norm: float,
) -> float:
    model.train()
    losses = []
    num_experts = model.num_experts
    for batch in loader:
        history = batch["history"].to(device).to(torch.float32)
        targets = batch["targets"].to(device).to(torch.float32)
        masks = batch["target_masks"].to(device).to(torch.bool)
        prediction_stack = batch["prediction_stack"].to(device).to(torch.float32)
        oracle_order = greedy_oracle_order(prediction_stack, targets, masks)
        total_loss = torch.zeros((), device=device)
        for step in range(max_queries):
            queried_ids = torch.full(
                (history.shape[0], max_queries),
                -1,
                dtype=torch.long,
                device=device,
            )
            if step > 0:
                queried_ids[:, :step] = oracle_order[:, :step]
            queried_mask, queried_forecasts, current_average = make_state(prediction_stack, queried_ids, num_experts)
            outputs = model(
                history,
                queried_mask,
                queried_ids,
                queried_forecasts,
                current_average_forecast=current_average,
            )
            target = utility_targets(prediction_stack, targets, masks, queried_ids, query_cost)
            loss_mask = 1.0 - queried_mask
            loss = F.smooth_l1_loss(outputs["utility_prediction"] * loss_mask, target * loss_mask)
            total_loss = total_loss + loss
        total_loss = total_loss / float(max_queries)
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        losses.append(float(total_loss.detach().cpu().item()))
    return float(statistics.mean(losses)) if losses else float("nan")


@torch.no_grad()
def evaluate_router(
    model: SequentialCOSTARTSRouterFull,
    cache: Mapping[str, Any],
    device: torch.device,
    max_queries: int,
    query_threshold: float,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    loader = DataLoader(CacheWindowDataset(cache), batch_size=batch_size, shuffle=False)
    maes = []
    mses = []
    query_counts = []
    first_queries = []
    queried_counts = torch.zeros(model.num_experts, dtype=torch.float64)
    stopping_counts = torch.zeros(model.num_experts, dtype=torch.float64)
    best_experts = cache["best_expert"].to(torch.long)
    offset = 0
    per_window = []
    for batch in loader:
        history = batch["history"].to(device).to(torch.float32)
        targets = batch["targets"].to(device).to(torch.float32)
        masks = batch["target_masks"].to(device).to(torch.bool)
        prediction_stack = batch["prediction_stack"].to(device).to(torch.float32)
        queried_ids = torch.full((history.shape[0], max_queries), -1, dtype=torch.long, device=device)
        queried_mask = torch.zeros((history.shape[0], model.num_experts), dtype=torch.float32, device=device)
        active = torch.ones(history.shape[0], dtype=torch.bool, device=device)
        for step in range(max_queries):
            state_mask, state_forecasts, current_average = make_state(prediction_stack, queried_ids, model.num_experts)
            utilities = model(
                history,
                state_mask,
                queried_ids,
                state_forecasts,
                current_average_forecast=current_average,
            )["utility_prediction"]
            utilities = utilities.masked_fill(state_mask.to(torch.bool), -1e9)
            values, next_ids = utilities.max(dim=1)
            should_query = active & ((step == 0) | (values > float(query_threshold)))
            if not bool(should_query.any()):
                break
            queried_ids[should_query, step] = next_ids[should_query]
            queried_mask[should_query, next_ids[should_query]] = 1.0
            active = active & should_query & (step + 1 < max_queries)
        final_prediction = current_average_from_ids(prediction_stack, queried_ids)
        batch_mae = sample_mae(final_prediction, targets, masks).cpu()
        batch_mse = sample_mse(final_prediction, targets, masks).cpu()
        counts = (queried_ids >= 0).sum(dim=1).cpu()
        maes.append(batch_mae)
        mses.append(batch_mse)
        query_counts.append(counts)
        for row in range(queried_ids.shape[0]):
            ids = queried_ids[row][queried_ids[row] >= 0].detach().cpu().tolist()
            if ids:
                first_queries.append(ids[0])
                for expert_id in set(ids):
                    queried_counts[expert_id] += 1
            stopping_counts[int(counts[row].item()) - 1] += 1
            per_window.append(
                {
                    "cache_index": offset + row,
                    "absolute_window_start": int(batch["absolute_window_start"][row].item()),
                    "query_count": int(counts[row].item()),
                    "queried_experts": " ".join(str(item) for item in ids),
                    "mae": float(batch_mae[row].item()),
                    "mse": float(batch_mse[row].item()),
                }
            )
        offset += history.shape[0]
    mae_tensor = torch.cat(maes)
    mse_tensor = torch.cat(mses)
    counts_tensor = torch.cat(query_counts).to(torch.float32)
    first_tensor = torch.tensor(first_queries, dtype=torch.long)
    top1_accuracy = float((first_tensor == best_experts[: first_tensor.numel()]).to(torch.float32).mean().item() * 100.0)
    total = float(cache["num_windows"])
    return {
        "mae": float(mae_tensor.mean().item()),
        "mse": float(mse_tensor.mean().item()),
        "average_queries": float(counts_tensor.mean().item()),
        "top1_expert_accuracy": top1_accuracy,
        "stopping_percent": {str(i + 1): float(stopping_counts[i].item() * 100.0 / total) for i in range(model.num_experts)},
        "expert_usage_percent": {EXPERT_ORDER[i]: float(queried_counts[i].item() * 100.0 / total) for i in range(model.num_experts)},
        "per_window": per_window,
    }


def load_verified_cache(path: Path, expected_role: str) -> dict[str, Any]:
    if "test" in str(path).lower():
        raise ValueError(f"Training script refuses test cache path: {path}")
    cache = torch.load(path, map_location="cpu", weights_only=False)
    validate_walkforward_cache(cache)
    if cache["cache_role"] != expected_role:
        raise ValueError(f"{path} cache_role={cache['cache_role']!r}, expected {expected_role!r}")
    return cache


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def individual_expert_metrics(cache: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "expert": expert,
            "mae": float(cache["error_matrix"][:, index].mean().item()),
            "mse": float(cache["mse_matrix"][:, index].mean().item()),
        }
        for index, expert in enumerate(EXPERT_ORDER)
    ]


def aggregate(values: Sequence[float]) -> tuple[float, float]:
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.pstdev(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", default=WalkForwardRouterTrainingConfig.train_cache)
    parser.add_argument("--val-cache", default=WalkForwardRouterTrainingConfig.val_cache)
    parser.add_argument("--checkpoint-root", default=WalkForwardRouterTrainingConfig.checkpoint_root)
    parser.add_argument("--results-root", default=WalkForwardRouterTrainingConfig.results_root)
    parser.add_argument("--seeds", default="7,11,13,17,19")
    parser.add_argument("--batch-size", type=int, default=WalkForwardRouterTrainingConfig.batch_size)
    parser.add_argument("--max-epochs", type=int, default=WalkForwardRouterTrainingConfig.max_epochs)
    parser.add_argument("--patience", type=int, default=WalkForwardRouterTrainingConfig.patience)
    parser.add_argument("--learning-rate", type=float, default=WalkForwardRouterTrainingConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=WalkForwardRouterTrainingConfig.weight_decay)
    parser.add_argument("--grad-clip-norm", type=float, default=WalkForwardRouterTrainingConfig.grad_clip_norm)
    parser.add_argument("--embedding-dim", type=int, default=WalkForwardRouterTrainingConfig.embedding_dim)
    parser.add_argument("--hidden-dim", type=int, default=WalkForwardRouterTrainingConfig.hidden_dim)
    parser.add_argument("--max-queries", type=int, default=WalkForwardRouterTrainingConfig.max_queries)
    parser.add_argument("--query-cost", type=float, default=WalkForwardRouterTrainingConfig.query_cost)
    parser.add_argument("--query-threshold", type=float, default=WalkForwardRouterTrainingConfig.query_threshold)
    parser.add_argument("--device", default=WalkForwardRouterTrainingConfig.device)
    args = parser.parse_args()

    train_cache_path = ROOT / args.train_cache
    val_cache_path = ROOT / args.val_cache
    train_cache = load_verified_cache(train_cache_path, "router_train_20_60")
    val_cache = load_verified_cache(val_cache_path, "router_val_60_80")
    if tuple(train_cache["expert_names"]) != tuple(val_cache["expert_names"]):
        raise ValueError("Expert order mismatch between train and validation caches")
    device = torch.device(args.device)
    seeds = tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
    checkpoint_root = ROOT / args.checkpoint_root
    results_root = ROOT / args.results_root
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)

    per_seed_rows = []
    for seed in seeds:
        set_seed(seed)
        model = SequentialCOSTARTSRouterFull(
            num_experts=len(EXPERT_ORDER),
            max_subset_size=args.max_queries,
            input_len=int(train_cache["input_len"]),
            forecast_horizon=int(train_cache["forecast_horizon"]),
            num_features=int(train_cache["num_features"]),
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        loader = DataLoader(CacheWindowDataset(train_cache), batch_size=args.batch_size, shuffle=True)
        seed_dir = checkpoint_root / f"seed_{seed}"
        seed_result_dir = results_root / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        seed_result_dir.mkdir(parents=True, exist_ok=True)
        best_mae = math.inf
        best_metrics = None
        best_epoch = -1
        bad_epochs = 0
        curves = []
        for epoch in range(1, args.max_epochs + 1):
            train_loss = train_one_epoch(
                model,
                loader,
                optimizer,
                device,
                args.max_queries,
                args.query_cost,
                args.grad_clip_norm,
            )
            metrics = evaluate_router(model, val_cache, device, args.max_queries, args.query_threshold, args.batch_size)
            curves.append({"epoch": epoch, "train_loss": train_loss, "val_mae": metrics["mae"], "val_mse": metrics["mse"], "average_queries": metrics["average_queries"]})
            if metrics["mae"] < best_mae:
                best_mae = metrics["mae"]
                best_metrics = metrics
                best_epoch = epoch
                bad_epochs = 0
                checkpoint = {
                    "router_type": "SequentialCOSTARTSRouterFull",
                    "router_config": {
                        "num_experts": len(EXPERT_ORDER),
                        "max_subset_size": args.max_queries,
                        "input_len": int(train_cache["input_len"]),
                        "forecast_horizon": int(train_cache["forecast_horizon"]),
                        "num_features": int(train_cache["num_features"]),
                        "embedding_dim": args.embedding_dim,
                        "hidden_dim": args.hidden_dim,
                    },
                    "router_state_dict": model.state_dict(),
                    "seed": seed,
                    "epoch": epoch,
                    "validation_metrics": {key: value for key, value in metrics.items() if key != "per_window"},
                    "query_cost": args.query_cost,
                    "query_threshold": args.query_threshold,
                    "train_cache": args.train_cache,
                    "val_cache": args.val_cache,
                    "train_cache_sha256": sha256_file(train_cache_path),
                    "val_cache_sha256": sha256_file(val_cache_path),
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "safety": "NO TEST DATA USED",
                }
                torch.save(checkpoint, seed_dir / "best_sequential_costarts_full_router.pt")
            else:
                bad_epochs += 1
                if bad_epochs >= args.patience:
                    break
        assert best_metrics is not None
        write_csv(seed_result_dir / "training_curves.csv", curves, ["epoch", "train_loss", "val_mae", "val_mse", "average_queries"])
        write_csv(seed_result_dir / "validation_per_window.csv", best_metrics["per_window"], ["cache_index", "absolute_window_start", "query_count", "queried_experts", "mae", "mse"])
        row = {
            "seed": seed,
            "best_epoch": best_epoch,
            "validation_mae": best_metrics["mae"],
            "validation_mse": best_metrics["mse"],
            "average_queries": best_metrics["average_queries"],
            "top1_expert_accuracy": best_metrics["top1_expert_accuracy"],
            "checkpoint_path": str(seed_dir / "best_sequential_costarts_full_router.pt"),
        }
        per_seed_rows.append(row)

    write_csv(results_root / "per_seed_results.csv", per_seed_rows, ["seed", "best_epoch", "validation_mae", "validation_mse", "average_queries", "top1_expert_accuracy", "checkpoint_path"])
    mae_mean, mae_std = aggregate([float(row["validation_mae"]) for row in per_seed_rows])
    mse_mean, mse_std = aggregate([float(row["validation_mse"]) for row in per_seed_rows])
    q_mean, q_std = aggregate([float(row["average_queries"]) for row in per_seed_rows])
    summary = {
        "method": "Sequential COSTAR-TS Full Walk-Forward",
        "train_cache": args.train_cache,
        "val_cache": args.val_cache,
        "train_cache_sha256": sha256_file(train_cache_path),
        "val_cache_sha256": sha256_file(val_cache_path),
        "num_block_b_plus_c_train_examples": int(train_cache["num_windows"]),
        "num_validation_examples": int(val_cache["num_windows"]),
        "seeds": list(seeds),
        "validation_mae_mean": mae_mean,
        "validation_mae_std": mae_std,
        "validation_mse_mean": mse_mean,
        "validation_mse_std": mse_std,
        "average_queries_mean": q_mean,
        "average_queries_std": q_std,
        "individual_expert_validation": individual_expert_metrics(val_cache),
        "cache_provenance": {
            "router_train": train_cache["provenance"],
            "router_val": val_cache["provenance"],
        },
        "leakage_checks": "passed",
        "safety": "NO TEST DATA USED",
    }
    (results_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
