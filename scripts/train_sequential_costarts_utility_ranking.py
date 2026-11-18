"""Train Sequential COSTAR-TS with state-dependent utility ranking objectives."""

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
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_costarts_walkforward_cache import EXPERT_ORDER, sha256_file, validate_walkforward_cache
from scripts.sequential_costarts_model_full import SequentialCOSTARTSRouterFull
from scripts.sequential_costarts_transformer_model import SequentialCOSTARTSTransformerRouter
from scripts.sequential_costarts_utility_objective import (
    available_expert_mask,
    compute_marginal_utilities,
    stop_calibration_loss,
    utility_listwise_loss,
    utility_listwise_stop_loss,
    utility_pairwise_loss,
    utility_weighted_pairwise_loss,
    utility_regret,
    utility_statistics,
)
from scripts.train_sequential_costarts_full_walkforward import (
    CacheWindowDataset,
    current_average_from_ids,
    greedy_oracle_order,
    make_state,
    utility_targets,
)


OBJECTIVES = ("existing", "utility_listwise", "utility_listwise_stop", "utility_pairwise", "utility_pairwise_weighted")
TRANSFER_MODES = ("none", "encoder", "full")
MARGIN_BINS = (
    ("lt_0.001", -math.inf, 0.001),
    ("0.001_0.005", 0.001, 0.005),
    ("0.005_0.01", 0.005, 0.01),
    ("0.01_0.02", 0.01, 0.02),
    ("gt_0.02", 0.02, math.inf),
)


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


def raw_sample_mae(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(prediction.dtype)
    denom = mask_f.flatten(1).sum(dim=1).clamp_min(1.0)
    return ((prediction - target).abs() * mask_f).flatten(1).sum(dim=1) / denom


def raw_sample_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(prediction.dtype)
    denom = mask_f.flatten(1).sum(dim=1).clamp_min(1.0)
    return ((prediction - target).square() * mask_f).flatten(1).sum(dim=1) / denom


def parameter_count(model: torch.nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters()))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
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


def build_model(args: argparse.Namespace, cache: Mapping[str, Any]) -> SequentialCOSTARTSRouterFull:
    if getattr(args, "router_arch", "costar") == "transformer":
        return SequentialCOSTARTSTransformerRouter(
            num_experts=len(EXPERT_ORDER),
            max_subset_size=args.max_queries,
            input_len=int(cache["input_len"]),
            forecast_horizon=int(cache["forecast_horizon"]),
            num_features=int(cache["num_features"]),
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.transformer_layers,
            num_heads=args.transformer_heads,
            feedforward_dim=args.transformer_ff_dim,
            dropout=args.transformer_dropout,
            state_mode=args.transformer_state_mode,
            pooling=args.transformer_pooling,
        )
    return SequentialCOSTARTSRouterFull(
        num_experts=len(EXPERT_ORDER),
        max_subset_size=args.max_queries,
        input_len=int(cache["input_len"]),
        forecast_horizon=int(cache["forecast_horizon"]),
        num_features=int(cache["num_features"]),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
    )


def load_transfer_checkpoint(model: SequentialCOSTARTSRouterFull, checkpoint_path: Path, transfer_mode: str) -> int:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = checkpoint.get("router_state_dict", checkpoint.get("state_dict"))
    if source is None:
        raise KeyError(f"{checkpoint_path} does not contain router_state_dict/state_dict")
    target = model.state_dict()
    if transfer_mode == "full":
        prefixes = None
    elif transfer_mode == "encoder":
        prefixes = (
            "history_encoder",
            "history_projection",
            "mask_encoder",
            "queried_forecast_encoder",
            "current_average_encoder",
            "scalar_encoder",
            "expert_embeddings",
            "fusion",
        )
    else:
        raise ValueError(f"Unsupported transfer_mode: {transfer_mode}")
    copied = {}
    for key, value in source.items():
        if key not in target or tuple(value.shape) != tuple(target[key].shape):
            continue
        if prefixes is not None and not key.startswith(prefixes):
            continue
        copied[key] = value
    target.update(copied)
    model.load_state_dict(target)
    return len(copied)


def objective_loss(
    objective: str,
    scores: torch.Tensor,
    prediction_stack: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    queried_ids: torch.Tensor,
    queried_mask: torch.Tensor,
    target_temperature: float,
    pairwise_epsilon: float,
    stop_calibration_weight: float,
    query_cost: float,
) -> torch.Tensor:
    available = available_expert_mask(queried_mask)
    if objective == "existing":
        target = utility_targets(prediction_stack, targets, masks, queried_ids, query_cost).detach()
        return F.smooth_l1_loss(scores.masked_select(available), target.masked_select(available))
    utilities = compute_marginal_utilities(prediction_stack, targets, masks, queried_ids)
    if query_cost:
        utilities = utilities - float(query_cost)
    if objective == "utility_listwise":
        rank_loss = utility_listwise_loss(scores, utilities, available, target_temperature)
    elif objective == "utility_listwise_stop":
        stop_available = (queried_ids >= 0).any(dim=1)
        return utility_listwise_stop_loss(scores, utilities, available, stop_available, target_temperature)
    elif objective == "utility_pairwise":
        rank_loss = utility_pairwise_loss(scores, utilities, available, pairwise_epsilon)
    elif objective == "utility_pairwise_weighted":
        rank_loss = utility_weighted_pairwise_loss(scores, utilities, available, pairwise_epsilon)
    else:
        raise ValueError(f"Unknown objective: {objective}")
    if stop_calibration_weight <= 0:
        return rank_loss
    return rank_loss + float(stop_calibration_weight) * stop_calibration_loss(scores, utilities, available)


def train_one_epoch(
    model: SequentialCOSTARTSRouterFull,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    objective: str,
    target_temperature: float,
) -> float:
    model.train()
    losses = []
    for batch in loader:
        history = batch["history"].to(device).to(torch.float32)
        targets = batch["targets"].to(device).to(torch.float32)
        masks = batch["target_masks"].to(device).to(torch.bool)
        prediction_stack = batch["prediction_stack"].to(device).to(torch.float32)
        oracle_order = greedy_oracle_order(prediction_stack, targets, masks)
        total_loss = torch.zeros((), device=device)
        for step in range(args.max_queries):
            queried_ids = torch.full((history.shape[0], args.max_queries), -1, dtype=torch.long, device=device)
            if step > 0:
                queried_ids[:, :step] = oracle_order[:, :step]
            queried_mask, queried_forecasts, current_average = make_state(prediction_stack, queried_ids, model.num_experts)
            scores = model(history, queried_mask, queried_ids, queried_forecasts, current_average_forecast=current_average)["utility_prediction"]
            total_loss = total_loss + objective_loss(
                objective,
                scores,
                prediction_stack,
                targets,
                masks,
                queried_ids,
                queried_mask,
                target_temperature,
                args.pairwise_epsilon,
                args.stop_calibration_weight,
                args.query_cost,
            )
        total_loss = total_loss / float(args.max_queries)
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        optimizer.step()
        losses.append(float(total_loss.detach().cpu().item()))
    return float(statistics.mean(losses)) if losses else float("nan")


def _percent(count: float, total: float) -> float:
    return float(100.0 * count / max(total, 1.0))


def _quantile_stats(values: Sequence[float]) -> dict[str, float]:
    tensor = torch.tensor(list(values), dtype=torch.float64)
    if tensor.numel() == 0:
        return {"mean": math.nan, "median": math.nan, "std": math.nan, "p90": math.nan, "p95": math.nan}
    q = torch.quantile(tensor, torch.tensor([0.5, 0.9, 0.95], dtype=torch.float64))
    return {
        "mean": float(tensor.mean().item()),
        "median": float(q[0].item()),
        "std": float(tensor.std(unbiased=False).item()),
        "p90": float(q[1].item()),
        "p95": float(q[2].item()),
    }


@torch.no_grad()
def evaluate_router(
    model: SequentialCOSTARTSRouterFull,
    cache: Mapping[str, Any],
    device: torch.device,
    args: argparse.Namespace,
    normalizer_std: torch.Tensor,
) -> dict[str, Any]:
    model.eval()
    loader = DataLoader(CacheWindowDataset(cache), batch_size=args.batch_size, shuffle=False)
    raw_maes = []
    raw_mses = []
    norm_maes = []
    norm_mses = []
    query_counts = []
    first_counts = torch.zeros(len(EXPERT_ORDER), dtype=torch.float64)
    second_counts = torch.zeros(len(EXPERT_ORDER), dtype=torch.float64)
    stop_counts = torch.zeros(len(EXPERT_ORDER), dtype=torch.float64)
    stop_decisions = torch.zeros(args.max_queries, dtype=torch.float64)
    stop_denoms = torch.zeros(args.max_queries, dtype=torch.float64)
    selected_top1 = []
    selected_top2 = []
    selected_positive = []
    any_positive = []
    regrets = []
    utility_values = []
    margin_rows = {name: {"count": 0, "top1": 0, "regret": [], "selected_mae": []} for name, _, _ in MARGIN_BINS}
    per_window = []
    offset = 0
    for batch in loader:
        history = batch["history"].to(device).to(torch.float32)
        targets = batch["targets"].to(device).to(torch.float32)
        masks = batch["target_masks"].to(device).to(torch.bool)
        prediction_stack = batch["prediction_stack"].to(device).to(torch.float32)
        queried_ids = torch.full((history.shape[0], args.max_queries), -1, dtype=torch.long, device=device)
        active = torch.ones(history.shape[0], dtype=torch.bool, device=device)
        for step in range(args.max_queries):
            queried_mask, queried_forecasts, current_average = make_state(prediction_stack, queried_ids, model.num_experts)
            scores = model(history, queried_mask, queried_ids, queried_forecasts, current_average_forecast=current_average)["utility_prediction"]
            available = available_expert_mask(queried_mask)
            utilities = compute_marginal_utilities(prediction_stack, targets, masks, queried_ids)
            utility_values.extend(utilities[available].detach().cpu().tolist())
            masked_scores = scores.masked_fill(~available, -1e9)
            values, next_ids = masked_scores.max(dim=1)
            should_query = active & ((step == 0) | (values > float(args.query_threshold)))
            stop_candidates = active & (step > 0)
            stop_denoms[step] += int(stop_candidates.sum().item())
            stop_decisions[step] += int((stop_candidates & ~should_query).sum().item())

            diagnostic = utility_regret(scores, utilities, available)
            masked_utilities = utilities.masked_fill(~available, -1e9)
            top2_utils = torch.topk(masked_utilities, k=2, dim=1).values
            margins = top2_utils[:, 0] - top2_utils[:, 1]
            for row in torch.where(should_query)[0].tolist():
                selected_top1.append(float(diagnostic["top1_hit"][row].item()))
                selected_top2.append(float(diagnostic["top2_hit"][row].item()))
                selected_positive.append(float(diagnostic["positive_selected"][row].item()))
                any_positive.append(float(diagnostic["any_positive"][row].item()))
                regret = float(diagnostic["regret"][row].item())
                regrets.append(regret)
                margin = float(margins[row].item())
                selected_prediction = prediction_stack[row : row + 1, :, :, int(next_ids[row].item())]
                selected_error = float(normalized_sample_mae(selected_prediction, targets[row : row + 1], masks[row : row + 1], normalizer_std).item())
                for name, low, high in MARGIN_BINS:
                    if low <= margin < high:
                        margin_rows[name]["count"] += 1
                        margin_rows[name]["top1"] += int(diagnostic["top1_hit"][row].item())
                        margin_rows[name]["regret"].append(regret)
                        margin_rows[name]["selected_mae"].append(selected_error)
                        break
            if not bool(should_query.any()):
                break
            queried_ids[should_query, step] = next_ids[should_query]
            active = active & should_query & (step + 1 < args.max_queries)

        final_prediction = current_average_from_ids(prediction_stack, queried_ids)
        raw_mae = raw_sample_mae(final_prediction, targets, masks)
        raw_mse = raw_sample_mse(final_prediction, targets, masks)
        norm_mae = normalized_sample_mae(final_prediction, targets, masks, normalizer_std)
        norm_mse = normalized_sample_mse(final_prediction, targets, masks, normalizer_std)
        raw_maes.append(raw_mae.cpu())
        raw_mses.append(raw_mse.cpu())
        norm_maes.append(norm_mae.cpu())
        norm_mses.append(norm_mse.cpu())
        counts = (queried_ids >= 0).sum(dim=1).cpu()
        query_counts.append(counts)
        for row in range(queried_ids.shape[0]):
            ids = queried_ids[row][queried_ids[row] >= 0].detach().cpu().tolist()
            if ids:
                first_counts[ids[0]] += 1
            if len(ids) > 1:
                second_counts[ids[1]] += 1
            stop_counts[int(counts[row].item()) - 1] += 1
            per_window.append(
                {
                    "cache_index": offset + row,
                    "absolute_window_start": int(batch["absolute_window_start"][row].item()),
                    "query_count": int(counts[row].item()),
                    "queried_experts": " ".join(str(item) for item in ids),
                    "raw_mae": float(raw_mae[row].item()),
                    "raw_mse": float(raw_mse[row].item()),
                    "normalized_mae": float(norm_mae[row].item()),
                    "normalized_mse": float(norm_mse[row].item()),
                }
            )
        offset += history.shape[0]
    total = float(cache["num_windows"])
    counts_tensor = torch.cat(query_counts).to(torch.float32)
    regret_stats = _quantile_stats(regrets)
    margin_summary = {}
    for name, data in margin_rows.items():
        margin_summary[name] = {
            "states": int(data["count"]),
            "top1_utility_accuracy": _percent(data["top1"], data["count"]),
            "mean_regret": float(statistics.mean(data["regret"])) if data["regret"] else math.nan,
            "mean_selected_single_expert_mae": float(statistics.mean(data["selected_mae"])) if data["selected_mae"] else math.nan,
        }
    return {
        "raw_mae": float(torch.cat(raw_maes).mean().item()),
        "raw_mse": float(torch.cat(raw_mses).mean().item()),
        "mae": float(torch.cat(norm_maes).mean().item()),
        "mse": float(torch.cat(norm_mses).mean().item()),
        "average_queries": float(counts_tensor.mean().item()),
        "top1_utility_accuracy": float(statistics.mean(selected_top1) * 100.0) if selected_top1 else math.nan,
        "top2_utility_coverage": float(statistics.mean(selected_top2) * 100.0) if selected_top2 else math.nan,
        "positive_utility_selection_rate": float(statistics.mean(selected_positive) * 100.0) if selected_positive else math.nan,
        "states_with_positive_available_utility": float(statistics.mean(any_positive) * 100.0) if any_positive else math.nan,
        "mean_regret": regret_stats["mean"],
        "median_regret": regret_stats["median"],
        "regret_std": regret_stats["std"],
        "regret_p90": regret_stats["p90"],
        "regret_p95": regret_stats["p95"],
        "query_count_percent": {str(i + 1): _percent(stop_counts[i].item(), total) for i in range(model.num_experts)},
        "first_query_percent": {EXPERT_ORDER[i]: _percent(first_counts[i].item(), total) for i in range(model.num_experts)},
        "second_query_percent_all_samples": {EXPERT_ORDER[i]: _percent(second_counts[i].item(), total) for i in range(model.num_experts)},
        "hard_stop_percent_by_step": {str(i + 1): _percent(stop_decisions[i].item(), stop_denoms[i].item()) for i in range(args.max_queries)},
        "utility_statistics": utility_statistics(torch.tensor(utility_values), args.near_zero_epsilon),
        "margin_bins": margin_summary,
        "per_window": per_window,
    }


def config_name(objective: str, temperature: float) -> str:
    if objective == "utility_listwise":
        return f"utility_listwise_tau_{str(temperature).replace('.', 'p')}"
    if objective == "utility_listwise_stop":
        return f"utility_listwise_stop_tau_{str(temperature).replace('.', 'p')}"
    return objective


def train_config(
    objective: str,
    target_temperature: float,
    args: argparse.Namespace,
    train_cache: Mapping[str, Any],
    val_cache: Mapping[str, Any],
    normalizer_std: torch.Tensor,
) -> dict[str, Any]:
    seeds = tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
    device = torch.device(args.device)
    name = config_name(objective, target_temperature)
    result_root = ROOT / args.results_root / name
    checkpoint_root = ROOT / args.checkpoint_root / name
    rows = []
    metrics_by_seed = {}
    for seed in seeds:
        set_seed(seed)
        model = build_model(args, train_cache).to(device)
        transferred_parameters = 0
        if getattr(args, "init_checkpoint", ""):
            transferred_parameters = load_transfer_checkpoint(model, ROOT / args.init_checkpoint, args.transfer_mode)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        loader = DataLoader(CacheWindowDataset(train_cache), batch_size=args.batch_size, shuffle=True)
        best_mae = math.inf
        best_metrics = None
        best_epoch = -1
        bad_epochs = 0
        curves = []
        seed_result = result_root / f"seed_{seed}"
        seed_ckpt = checkpoint_root / f"seed_{seed}"
        for epoch in range(1, args.max_epochs + 1):
            train_loss = train_one_epoch(model, loader, optimizer, device, args, objective, target_temperature)
            metrics = evaluate_router(model, val_cache, device, args, normalizer_std)
            curves.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "validation_mae": metrics["mae"],
                    "validation_mse": metrics["mse"],
                    "average_queries": metrics["average_queries"],
                    "top1_utility_accuracy": metrics["top1_utility_accuracy"],
                    "top2_utility_coverage": metrics["top2_utility_coverage"],
                    "mean_regret": metrics["mean_regret"],
                }
            )
            if metrics["mae"] < best_mae:
                best_mae = metrics["mae"]
                best_metrics = metrics
                best_epoch = epoch
                bad_epochs = 0
                seed_ckpt.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "router_type": "SequentialCOSTARTSRouterFull",
                        "objective": objective,
                        "target_temperature": target_temperature,
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
                        "init_checkpoint": getattr(args, "init_checkpoint", ""),
                        "transfer_mode": getattr(args, "transfer_mode", "none"),
                        "transferred_tensors": transferred_parameters,
                        "seed": seed,
                        "epoch": epoch,
                        "validation_metrics": {key: value for key, value in metrics.items() if key != "per_window"},
                        "created_at_utc": datetime.now(timezone.utc).isoformat(),
                        "safety": "NO TEST DATA USED",
                    },
                    seed_ckpt / "best_utility_router.pt",
                )
            else:
                bad_epochs += 1
                if bad_epochs >= args.patience:
                    break
        assert best_metrics is not None
        write_csv(seed_result / "training_curves.csv", curves)
        write_csv(seed_result / "validation_per_window.csv", best_metrics["per_window"])
        row = {
            "config": name,
            "objective": objective,
            "target_temperature": target_temperature if objective in ("utility_listwise", "utility_listwise_stop") else "",
            "seed": seed,
            "best_epoch": best_epoch,
            "validation_mae": best_metrics["mae"],
            "validation_mse": best_metrics["mse"],
            "raw_validation_mae": best_metrics["raw_mae"],
            "raw_validation_mse": best_metrics["raw_mse"],
            "average_queries": best_metrics["average_queries"],
            "top1_utility_accuracy": best_metrics["top1_utility_accuracy"],
            "top2_utility_coverage": best_metrics["top2_utility_coverage"],
            "mean_regret": best_metrics["mean_regret"],
            "median_regret": best_metrics["median_regret"],
            "regret_p90": best_metrics["regret_p90"],
            "positive_utility_selection_rate": best_metrics["positive_utility_selection_rate"],
            "states_with_positive_available_utility": best_metrics["states_with_positive_available_utility"],
            "parameter_count": parameter_count(model),
            "init_checkpoint": getattr(args, "init_checkpoint", ""),
            "transfer_mode": getattr(args, "transfer_mode", "none"),
            "transferred_tensors": transferred_parameters,
        }
        rows.append(row)
        metrics_by_seed[str(seed)] = {key: value for key, value in best_metrics.items() if key != "per_window"}
    write_csv(result_root / "per_seed_results.csv", rows)
    summary = {
        "config": name,
        "objective": objective,
        "target_temperature": target_temperature if objective in ("utility_listwise", "utility_listwise_stop") else None,
        "validation_mae_mean": aggregate([row["validation_mae"] for row in rows])[0],
        "validation_mae_std": aggregate([row["validation_mae"] for row in rows])[1],
        "validation_mse_mean": aggregate([row["validation_mse"] for row in rows])[0],
        "validation_mse_std": aggregate([row["validation_mse"] for row in rows])[1],
        "average_queries_mean": aggregate([row["average_queries"] for row in rows])[0],
        "average_queries_std": aggregate([row["average_queries"] for row in rows])[1],
        "top1_utility_accuracy_mean": aggregate([row["top1_utility_accuracy"] for row in rows])[0],
        "top2_utility_coverage_mean": aggregate([row["top2_utility_coverage"] for row in rows])[0],
        "mean_regret_mean": aggregate([row["mean_regret"] for row in rows])[0],
        "mean_regret_std": aggregate([row["mean_regret"] for row in rows])[1],
        "positive_utility_selection_rate_mean": aggregate([row["positive_utility_selection_rate"] for row in rows])[0],
        "parameter_count": int(rows[0]["parameter_count"]),
        "per_seed": rows,
        "best_metrics_by_seed": metrics_by_seed,
    }
    (result_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", default="cache/costarts_walkforward/router_train_20_60_cache.pt")
    parser.add_argument("--val-cache", default="cache/costarts_walkforward/router_val_60_80_cache.pt")
    parser.add_argument("--normalizer-checkpoint", default="checkpoints/costarts_walkforward/final_60/DLinear/best_expert.pt")
    parser.add_argument("--checkpoint-root", default="checkpoints/costarts_walkforward/utility_ranking")
    parser.add_argument("--results-root", default="results/router_summary/costarts_walkforward/utility_ranking")
    parser.add_argument("--objectives", default="existing,utility_listwise,utility_pairwise")
    parser.add_argument("--temperatures", default="0.001,0.005,0.01,0.02,0.05")
    parser.add_argument("--seeds", default="7,11,13,17,19")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--max-queries", type=int, default=5)
    parser.add_argument("--query-cost", type=float, default=0.0)
    parser.add_argument("--query-threshold", type=float, default=0.0)
    parser.add_argument("--stop-calibration-weight", type=float, default=1.0)
    parser.add_argument("--pairwise-epsilon", type=float, default=0.001)
    parser.add_argument("--near-zero-epsilon", type=float, default=0.001)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--router-arch", choices=("costar", "transformer"), default="costar")
    parser.add_argument("--transformer-state-mode", choices=("history_only", "history_ensemble", "full"), default="history_only")
    parser.add_argument("--transformer-pooling", choices=("cls", "mean", "attention"), default="cls")
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-ff-dim", type=int, default=128)
    parser.add_argument("--transformer-dropout", type=float, default=0.1)
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--transfer-mode", choices=TRANSFER_MODES, default="none")
    args = parser.parse_args()

    objectives = [item.strip() for item in args.objectives.split(",") if item.strip()]
    invalid = [item for item in objectives if item not in OBJECTIVES]
    if invalid:
        raise ValueError(f"Invalid objectives: {invalid}")
    temperatures = [float(item.strip()) for item in args.temperatures.split(",") if item.strip()]
    train_cache_path = ROOT / args.train_cache
    val_cache_path = ROOT / args.val_cache
    train_cache = load_verified_cache(train_cache_path, "router_train_20_60")
    val_cache = load_verified_cache(val_cache_path, "router_val_60_80")
    starts = val_cache["absolute_window_starts"].to(torch.long)
    val_start = int(starts.min().item())
    val_end = int(starts.max().item()) + int(val_cache["forecast_horizon"])
    if val_start != 8640 or val_end > 11520:
        raise ValueError(f"Unexpected validation coverage: starts at {val_start}, final forecast ends at {val_end}")
    normalizer_std = load_normalizer_std(ROOT / args.normalizer_checkpoint)
    summaries = []
    if "existing" in objectives:
        summaries.append(train_config("existing", temperatures[0], args, train_cache, val_cache, normalizer_std))
    if "utility_listwise" in objectives:
        for temperature in temperatures:
            summaries.append(train_config("utility_listwise", temperature, args, train_cache, val_cache, normalizer_std))
    if "utility_listwise_stop" in objectives:
        for temperature in temperatures:
            summaries.append(train_config("utility_listwise_stop", temperature, args, train_cache, val_cache, normalizer_std))
    if "utility_pairwise" in objectives:
        summaries.append(train_config("utility_pairwise", temperatures[0], args, train_cache, val_cache, normalizer_std))
    if "utility_pairwise_weighted" in objectives:
        summaries.append(train_config("utility_pairwise_weighted", temperatures[0], args, train_cache, val_cache, normalizer_std))
    comparison = []
    for summary in summaries:
        comparison.append(
            {
                "config": summary["config"],
                "objective": summary["objective"],
                "target_temperature": summary["target_temperature"] if summary["target_temperature"] is not None else "",
                "validation_mae_mean": summary["validation_mae_mean"],
                "validation_mae_std": summary["validation_mae_std"],
                "validation_mse_mean": summary["validation_mse_mean"],
                "validation_mse_std": summary["validation_mse_std"],
                "average_queries_mean": summary["average_queries_mean"],
                "top1_utility_accuracy_mean": summary["top1_utility_accuracy_mean"],
                "top2_utility_coverage_mean": summary["top2_utility_coverage_mean"],
                "mean_regret_mean": summary["mean_regret_mean"],
                "positive_utility_selection_rate_mean": summary["positive_utility_selection_rate_mean"],
                "parameter_count": summary["parameter_count"],
            }
        )
    results_root = ROOT / args.results_root
    write_csv(results_root / "comparison.csv", comparison)
    best = min(comparison, key=lambda row: float(row["validation_mae_mean"]))
    full_summary = {
        "method": "Sequential COSTAR-TS utility ranking objective comparison",
        "train_cache": args.train_cache,
        "val_cache": args.val_cache,
        "validation_range": [8640, 11520],
        "train_cache_sha256": sha256_file(train_cache_path),
        "val_cache_sha256": sha256_file(val_cache_path),
        "initial_state_target": "With no queried experts, utility_j = -MAE(expert_j, target), so first-query candidates are ranked by standalone forecast loss.",
        "listwise_loss": "-sum_j softmax(u_j/tau) * log_softmax(f_j), normalized over unqueried experts only.",
        "listwise_stop_loss": "For non-initial states, actions are unqueried experts plus STOP. STOP has utility 0 and fixed logit 0. Initial state masks STOP so the first query remains forced.",
        "pairwise_loss": "softplus(-(f_i-f_j)) for available pairs with u_i > u_j + epsilon.",
        "weighted_pairwise_loss": "pairwise loss weighted by abs utility gap for available pairs with u_i > u_j + epsilon.",
        "stop_behavior": "unchanged inference: first query is forced, later queries continue only when max predicted score > query_threshold; stop calibration anchors ranking scores to marginal utility.",
        "safety": "NO TEST DATA USED",
        "summaries": summaries,
        "comparison": comparison,
        "best_by_validation_mae": best,
    }
    (results_root / "summary.json").write_text(json.dumps(full_summary, indent=2), encoding="utf-8")
    print(json.dumps({"comparison": comparison, "best_by_validation_mae": best, "safety": full_summary["safety"]}, indent=2))


if __name__ == "__main__":
    main()
