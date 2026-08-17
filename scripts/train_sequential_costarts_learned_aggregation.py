"""Train a learned queried-forecast combiner for frozen Sequential COSTAR-TS routers."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_costarts_walkforward_cache import EXPERT_ORDER, validate_walkforward_cache
from scripts.sequential_costarts_model_full import SequentialCOSTARTSRouterFull
from scripts.train_sequential_costarts_full_walkforward import CacheWindowDataset, current_average_from_ids, make_state


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_verified_cache(path: Path, expected_role: str) -> dict[str, Any]:
    if "test" in str(path).lower():
        raise ValueError(f"Refusing test cache path: {path}")
    cache = torch.load(path, map_location="cpu", weights_only=False)
    validate_walkforward_cache(cache)
    if cache["cache_role"] != expected_role:
        raise ValueError(f"{path} cache_role={cache['cache_role']!r}, expected {expected_role!r}")
    return cache


def load_normalizer_std(path: Path) -> torch.Tensor:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "scaler_std" not in checkpoint:
        raise KeyError(f"{path} does not contain scaler_std")
    return checkpoint["scaler_std"].to(torch.float32)


def normalized_sample_mae(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    std = std.to(prediction.device, prediction.dtype).view(1, 1, -1)
    diff = (prediction - target) / std
    mask_f = mask.to(prediction.dtype)
    denom = mask_f.flatten(1).sum(dim=1).clamp_min(1.0)
    return (diff.abs() * mask_f).flatten(1).sum(dim=1) / denom


def normalized_sample_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    std = std.to(prediction.device, prediction.dtype).view(1, 1, -1)
    diff = (prediction - target) / std
    mask_f = mask.to(prediction.dtype)
    denom = mask_f.flatten(1).sum(dim=1).clamp_min(1.0)
    return (diff.square() * mask_f).flatten(1).sum(dim=1) / denom


class QueriedForecastCombiner(nn.Module):
    """Predict convex weights over already queried expert forecasts."""

    def __init__(self, num_experts: int, max_queries: int, horizon: int, num_features: int, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.num_experts = int(num_experts)
        self.max_queries = int(max_queries)
        self.horizon = int(horizon)
        self.num_features = int(num_features)
        self.state_dim = int(state_dim)
        self.forecast_encoder = nn.Sequential(
            nn.Linear(horizon * num_features, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.expert_embeddings = nn.Embedding(num_experts, hidden_dim)
        self.scorer = nn.Sequential(
            nn.Linear(state_dim + hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, queried_ids: torch.Tensor, queried_forecasts: torch.Tensor) -> dict[str, torch.Tensor]:
        valid = queried_ids >= 0
        flat = queried_forecasts.reshape(queried_forecasts.shape[0], self.max_queries, self.horizon * self.num_features)
        forecast_rep = self.forecast_encoder(flat)
        expert_rep = self.expert_embeddings(queried_ids.clamp_min(0))
        state_rep = state[:, None, :].expand(-1, self.max_queries, -1)
        logits = self.scorer(torch.cat((state_rep, forecast_rep, expert_rep), dim=-1)).squeeze(-1)
        logits = logits.masked_fill(~valid, -1e9)
        weights = torch.softmax(logits, dim=1)
        prediction = (weights[:, :, None, None] * queried_forecasts).sum(dim=1)
        return {"prediction": prediction, "weights": weights, "weight_logits": logits}


@torch.no_grad()
def rollout_router(
    router: SequentialCOSTARTSRouterFull,
    history: torch.Tensor,
    prediction_stack: torch.Tensor,
    max_queries: int,
    query_threshold: float,
) -> torch.Tensor:
    queried_ids = torch.full((history.shape[0], max_queries), -1, dtype=torch.long, device=history.device)
    active = torch.ones(history.shape[0], dtype=torch.bool, device=history.device)
    for step in range(max_queries):
        state_mask, state_forecasts, current_average = make_state(prediction_stack, queried_ids, router.num_experts)
        utilities = router(history, state_mask, queried_ids, state_forecasts, current_average_forecast=current_average)["utility_prediction"]
        utilities = utilities.masked_fill(state_mask.to(torch.bool), -1e9)
        values, next_ids = utilities.max(dim=1)
        should_query = active & ((step == 0) | (values > float(query_threshold)))
        if not bool(should_query.any()):
            break
        queried_ids[should_query, step] = next_ids[should_query]
        active = active & should_query & (step + 1 < max_queries)
    return queried_ids


@torch.no_grad()
def router_state(
    router: SequentialCOSTARTSRouterFull,
    history: torch.Tensor,
    prediction_stack: torch.Tensor,
    queried_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    state_mask, state_forecasts, current_average = make_state(prediction_stack, queried_ids, router.num_experts)
    outputs = router(history, state_mask, queried_ids, state_forecasts, current_average_forecast=current_average)
    return outputs["representation"].detach(), state_forecasts.detach(), current_average.detach()


def train_one_epoch(
    router: SequentialCOSTARTSRouterFull,
    combiner: QueriedForecastCombiner,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_queries: int,
    query_threshold: float,
    normalizer_std: torch.Tensor,
    grad_clip_norm: float,
) -> float:
    combiner.train()
    losses = []
    std = normalizer_std.to(device).to(torch.float32).view(1, 1, -1)
    for batch in loader:
        history = batch["history"].to(device).to(torch.float32)
        targets = batch["targets"].to(device).to(torch.float32)
        masks = batch["target_masks"].to(device).to(torch.bool)
        prediction_stack = batch["prediction_stack"].to(device).to(torch.float32)
        queried_ids = rollout_router(router, history, prediction_stack, max_queries, query_threshold)
        state, queried_forecasts, _ = router_state(router, history, prediction_stack, queried_ids)
        outputs = combiner(state, queried_ids, queried_forecasts)
        diff = (outputs["prediction"] - targets) / std
        loss = F.smooth_l1_loss(diff[masks], torch.zeros_like(diff[masks]))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(combiner.parameters(), grad_clip_norm)
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return float(statistics.mean(losses)) if losses else float("nan")


@torch.no_grad()
def evaluate(
    router: SequentialCOSTARTSRouterFull,
    combiner: QueriedForecastCombiner,
    cache: Mapping[str, Any],
    device: torch.device,
    max_queries: int,
    query_threshold: float,
    batch_size: int,
    normalizer_std: torch.Tensor,
) -> dict[str, Any]:
    combiner.eval()
    loader = DataLoader(CacheWindowDataset(cache), batch_size=batch_size, shuffle=False)
    equal_maes = []
    equal_mses = []
    learned_maes = []
    learned_mses = []
    oracle_queried_maes = []
    query_counts = []
    first_counts = torch.zeros(len(EXPERT_ORDER), dtype=torch.float64)
    weight_totals = torch.zeros(len(EXPERT_ORDER), dtype=torch.float64)
    weight_counts = torch.zeros(len(EXPERT_ORDER), dtype=torch.float64)
    per_window = []
    offset = 0
    for batch in loader:
        history = batch["history"].to(device).to(torch.float32)
        targets = batch["targets"].to(device).to(torch.float32)
        masks = batch["target_masks"].to(device).to(torch.bool)
        prediction_stack = batch["prediction_stack"].to(device).to(torch.float32)
        queried_ids = rollout_router(router, history, prediction_stack, max_queries, query_threshold)
        state, queried_forecasts, equal_prediction = router_state(router, history, prediction_stack, queried_ids)
        outputs = combiner(state, queried_ids, queried_forecasts)
        learned_prediction = outputs["prediction"]
        equal_mae = normalized_sample_mae(equal_prediction, targets, masks, normalizer_std)
        equal_mse = normalized_sample_mse(equal_prediction, targets, masks, normalizer_std)
        learned_mae = normalized_sample_mae(learned_prediction, targets, masks, normalizer_std)
        learned_mse = normalized_sample_mse(learned_prediction, targets, masks, normalizer_std)
        equal_maes.append(equal_mae.cpu())
        equal_mses.append(equal_mse.cpu())
        learned_maes.append(learned_mae.cpu())
        learned_mses.append(learned_mse.cpu())
        counts = (queried_ids >= 0).sum(dim=1).cpu()
        query_counts.append(counts)
        for row in range(queried_ids.shape[0]):
            global_index = offset + row
            ids = queried_ids[row][queried_ids[row] >= 0].detach().cpu().tolist()
            if ids:
                first_counts[ids[0]] += 1
            weights = outputs["weights"][row].detach().cpu()
            for slot, expert_id in enumerate(ids):
                weight_totals[expert_id] += float(weights[slot].item())
                weight_counts[expert_id] += 1.0
            if ids:
                best = torch.stack(
                    [
                        normalized_sample_mae(
                            prediction_stack[row : row + 1, :, :, expert_id],
                            targets[row : row + 1],
                            masks[row : row + 1],
                            normalizer_std,
                        ).cpu()
                        for expert_id in ids
                    ]
                ).min()
                oracle_queried_maes.append(float(best.item()))
            per_window.append(
                {
                    "cache_index": global_index,
                    "absolute_window_start": int(batch["absolute_window_start"][row].item()),
                    "query_count": int(counts[row].item()),
                    "queried_experts": " ".join(str(item) for item in ids),
                    "equal_average_mae": float(equal_mae[row].item()),
                    "equal_average_mse": float(equal_mse[row].item()),
                    "learned_aggregation_mae": float(learned_mae[row].item()),
                    "learned_aggregation_mse": float(learned_mse[row].item()),
                    "weights": " ".join(f"{float(weights[slot].item()):.6f}" for slot in range(len(ids))),
                }
            )
        offset += history.shape[0]
    total = float(cache["num_windows"])
    return {
        "equal_average_mae": float(torch.cat(equal_maes).mean().item()),
        "equal_average_mse": float(torch.cat(equal_mses).mean().item()),
        "learned_aggregation_mae": float(torch.cat(learned_maes).mean().item()),
        "learned_aggregation_mse": float(torch.cat(learned_mses).mean().item()),
        "oracle_best_queried_mae": float(statistics.mean(oracle_queried_maes)) if oracle_queried_maes else float("nan"),
        "average_queries": float(torch.cat(query_counts).to(torch.float32).mean().item()),
        "query_count_percent": {
            str(i + 1): float((torch.cat(query_counts) == i + 1).to(torch.float32).mean().item() * 100.0)
            for i in range(len(EXPERT_ORDER))
        },
        "first_query_percent": {EXPERT_ORDER[i]: float(first_counts[i].item() * 100.0 / total) for i in range(len(EXPERT_ORDER))},
        "mean_weight_by_expert_when_queried": {
            EXPERT_ORDER[i]: float(weight_totals[i].item() / max(weight_counts[i].item(), 1.0))
            for i in range(len(EXPERT_ORDER))
        },
        "per_window": per_window,
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def aggregate(values: Sequence[float]) -> tuple[float, float]:
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.pstdev(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", default="cache/costarts_walkforward/router_train_20_60_cache.pt")
    parser.add_argument("--val-cache", default="cache/costarts_walkforward/router_val_60_80_cache.pt")
    parser.add_argument("--normalizer-checkpoint", default="checkpoints/costarts_walkforward/final_60/DLinear/best_expert.pt")
    parser.add_argument("--router-checkpoint-root", default="checkpoints/costarts_walkforward/attention/embedding")
    parser.add_argument("--checkpoint-root", default="checkpoints/costarts_walkforward/learned_aggregation")
    parser.add_argument("--results-root", default="results/router_summary/costarts_walkforward/learned_aggregation")
    parser.add_argument("--seeds", default="7,11,13,17,19")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--max-queries", type=int, default=5)
    parser.add_argument("--query-threshold", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    train_cache = load_verified_cache(ROOT / args.train_cache, "router_train_20_60")
    val_cache = load_verified_cache(ROOT / args.val_cache, "router_val_60_80")
    normalizer_std = load_normalizer_std(ROOT / args.normalizer_checkpoint)
    starts = val_cache["absolute_window_starts"].to(torch.long)
    if int(starts.min().item()) != 8640 or int(starts.max().item()) + int(val_cache["forecast_horizon"]) > 11520:
        raise ValueError("Unexpected validation coverage")
    device = torch.device(args.device)
    seeds = tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
    per_seed = []
    best_metrics_by_seed = {}
    for seed in seeds:
        set_seed(seed)
        checkpoint = torch.load(ROOT / args.router_checkpoint_root / f"seed_{seed}" / "best_attention_router.pt", map_location="cpu", weights_only=False)
        router = SequentialCOSTARTSRouterFull(**checkpoint["router_config"]).to(device)
        router.load_state_dict(checkpoint["router_state_dict"], strict=True)
        router.eval()
        for param in router.parameters():
            param.requires_grad_(False)
        combiner = QueriedForecastCombiner(
            num_experts=len(EXPERT_ORDER),
            max_queries=args.max_queries,
            horizon=int(train_cache["forecast_horizon"]),
            num_features=int(train_cache["num_features"]),
            state_dim=router.embedding_dim,
            hidden_dim=args.hidden_dim,
        ).to(device)
        optimizer = torch.optim.AdamW(combiner.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        loader = DataLoader(CacheWindowDataset(train_cache), batch_size=args.batch_size, shuffle=True)
        best_mae = math.inf
        best_metrics = None
        best_epoch = -1
        bad_epochs = 0
        curves = []
        for epoch in range(1, args.max_epochs + 1):
            loss = train_one_epoch(router, combiner, loader, optimizer, device, args.max_queries, args.query_threshold, normalizer_std, args.grad_clip_norm)
            metrics = evaluate(router, combiner, val_cache, device, args.max_queries, args.query_threshold, args.batch_size, normalizer_std)
            curves.append(
                {
                    "epoch": epoch,
                    "train_loss": loss,
                    "equal_average_mae": metrics["equal_average_mae"],
                    "learned_aggregation_mae": metrics["learned_aggregation_mae"],
                    "learned_aggregation_mse": metrics["learned_aggregation_mse"],
                    "average_queries": metrics["average_queries"],
                }
            )
            if metrics["learned_aggregation_mae"] < best_mae:
                best_mae = metrics["learned_aggregation_mae"]
                best_metrics = metrics
                best_epoch = epoch
                bad_epochs = 0
                ckpt_dir = ROOT / args.checkpoint_root / f"seed_{seed}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "combiner_state_dict": combiner.state_dict(),
                        "router_checkpoint": str(ROOT / args.router_checkpoint_root / f"seed_{seed}" / "best_attention_router.pt"),
                        "seed": seed,
                        "epoch": epoch,
                        "validation_metrics": {key: value for key, value in metrics.items() if key != "per_window"},
                        "created_at_utc": datetime.now(timezone.utc).isoformat(),
                        "safety": "NO TEST DATA USED",
                    },
                    ckpt_dir / "best_learned_aggregation.pt",
                )
            else:
                bad_epochs += 1
                if bad_epochs >= args.patience:
                    break
        assert best_metrics is not None
        result_dir = ROOT / args.results_root / f"seed_{seed}"
        write_csv(result_dir / "training_curves.csv", curves)
        write_csv(result_dir / "validation_per_window.csv", best_metrics["per_window"])
        row = {
            "seed": seed,
            "best_epoch": best_epoch,
            "equal_average_mae": best_metrics["equal_average_mae"],
            "equal_average_mse": best_metrics["equal_average_mse"],
            "learned_aggregation_mae": best_metrics["learned_aggregation_mae"],
            "learned_aggregation_mse": best_metrics["learned_aggregation_mse"],
            "oracle_best_queried_mae": best_metrics["oracle_best_queried_mae"],
            "average_queries": best_metrics["average_queries"],
            "combiner_parameter_count": sum(param.numel() for param in combiner.parameters()),
        }
        per_seed.append(row)
        best_metrics_by_seed[str(seed)] = {key: value for key, value in best_metrics.items() if key != "per_window"}
    write_csv(ROOT / args.results_root / "per_seed_results.csv", per_seed)
    learned_mae_mean, learned_mae_std = aggregate([row["learned_aggregation_mae"] for row in per_seed])
    learned_mse_mean, learned_mse_std = aggregate([row["learned_aggregation_mse"] for row in per_seed])
    equal_mae_mean, equal_mae_std = aggregate([row["equal_average_mae"] for row in per_seed])
    oracle_queried_mean, oracle_queried_std = aggregate([row["oracle_best_queried_mae"] for row in per_seed])
    comparison = [
        {
            "model": "Embedding router + equal average",
            "validation_mae_mean": equal_mae_mean,
            "validation_mae_std": equal_mae_std,
            "validation_mse_mean": aggregate([row["equal_average_mse"] for row in per_seed])[0],
            "validation_mse_std": aggregate([row["equal_average_mse"] for row in per_seed])[1],
        },
        {
            "model": "Embedding router + learned aggregation",
            "validation_mae_mean": learned_mae_mean,
            "validation_mae_std": learned_mae_std,
            "validation_mse_mean": learned_mse_mean,
            "validation_mse_std": learned_mse_std,
            "improvement_vs_equal_average_mae": equal_mae_mean - learned_mae_mean,
        },
        {
            "model": "Oracle best among queried experts",
            "validation_mae_mean": oracle_queried_mean,
            "validation_mae_std": oracle_queried_std,
        },
    ]
    write_csv(ROOT / args.results_root / "comparison.csv", comparison)
    summary = {
        "method": "Frozen embedding Sequential COSTAR-TS without equal averaging",
        "train_cache": args.train_cache,
        "val_cache": args.val_cache,
        "router_checkpoint_root": args.router_checkpoint_root,
        "validation_range": [8640, 11520],
        "test_usage": "NO TEST DATA LOADED OR EVALUATED",
        "per_seed": per_seed,
        "comparison": comparison,
        "best_metrics_by_seed": best_metrics_by_seed,
        "safety": "NO TEST DATA USED",
    }
    (ROOT / args.results_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"comparison": comparison, "safety": summary["safety"]}, indent=2))


if __name__ == "__main__":
    main()
