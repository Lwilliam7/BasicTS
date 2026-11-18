"""Train Sequential COSTAR-TS with partial-subset forecast-state supervision.

This experiment keeps the frozen expert forecasts fixed and trains only the
router.  Unlike the older utility-ranking runs, each batch creates labels for
partial queried subsets so the router directly learns state-dependent marginal
utility and an explicit STOP decision.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.sequential_costarts_model_full import SequentialCOSTARTSRouterFull
from scripts.train_sequential_costarts_full_walkforward import (
    CacheWindowDataset,
    current_average_from_ids,
    greedy_oracle_order,
)
from scripts.train_sequential_costarts_utility_ranking import aggregate, sha256_file, write_csv


ABLATIONS = (
    "history_only",
    "history_mask",
    "history_queried_forecasts",
    "history_current_ensemble",
    "history_disagreement",
    "full",
)


@dataclass(frozen=True)
class StateBatch:
    queried_ids: torch.Tensor
    queried_mask: torch.Tensor
    queried_forecasts: torch.Tensor
    current_average: torch.Tensor
    utilities: torch.Tensor
    available: torch.Tensor
    stop_target: torch.Tensor
    current_count: torch.Tensor


class ForecastStateStopRouter(nn.Module):
    """Sequential COSTAR router with an explicit STOP head."""

    def __init__(
        self,
        num_experts: int,
        max_subset_size: int,
        input_len: int,
        forecast_horizon: int,
        num_features: int,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.router = SequentialCOSTARTSRouterFull(
            num_experts=num_experts,
            max_subset_size=max_subset_size,
            input_len=input_len,
            forecast_horizon=forecast_horizon,
            num_features=num_features,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
        )
        self.stop_head = nn.Linear(embedding_dim, 1)

    @property
    def num_experts(self) -> int:
        return self.router.num_experts

    @property
    def max_subset_size(self) -> int:
        return self.router.max_subset_size

    def forward(
        self,
        history: torch.Tensor,
        queried_mask: torch.Tensor,
        queried_expert_ids: torch.Tensor,
        queried_expert_forecasts: torch.Tensor,
        current_average_forecast: torch.Tensor,
        scalar_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        out = self.router(
            history,
            queried_mask,
            queried_expert_ids,
            queried_expert_forecasts,
            current_average_forecast=current_average_forecast,
            scalar_features=scalar_features,
        )
        out["stop_logit"] = self.stop_head(out["representation"]).squeeze(-1)
        return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_cache(path: Path) -> dict[str, Any]:
    if "test" in str(path).lower():
        raise ValueError(f"Refusing to load test cache path: {path}")
    cache = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "histories",
        "targets",
        "target_masks",
        "prediction_stack",
        "absolute_window_starts",
        "expert_names",
        "input_len",
        "forecast_horizon",
        "num_features",
        "num_windows",
    }
    missing = sorted(required - set(cache))
    if missing:
        raise KeyError(f"{path} is missing required cache fields: {missing}")
    role = cache.get("cache_role", cache.get("split_role", ""))
    if "test" in str(role).lower():
        raise ValueError(f"Refusing test cache role in {path}: {role}")
    return cache


def load_metric_std(path: str | None, num_features: int) -> torch.Tensor:
    if not path:
        return torch.ones(num_features, dtype=torch.float32)
    checkpoint = torch.load(ROOT / path, map_location="cpu", weights_only=False)
    if "scaler_std" not in checkpoint:
        raise KeyError(f"{path} does not contain scaler_std")
    return checkpoint["scaler_std"].to(torch.float32)


def scaled_sample_mae(
    prediction: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    metric_std: torch.Tensor,
) -> torch.Tensor:
    std = metric_std.to(prediction.device, prediction.dtype).view(1, 1, -1)
    mask = masks.to(prediction.dtype)
    denom = mask.flatten(1).sum(dim=1).clamp_min(1.0)
    return (((prediction - targets) / std).abs() * mask).flatten(1).sum(dim=1) / denom


def scaled_sample_mse(
    prediction: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    metric_std: torch.Tensor,
) -> torch.Tensor:
    std = metric_std.to(prediction.device, prediction.dtype).view(1, 1, -1)
    mask = masks.to(prediction.dtype)
    denom = mask.flatten(1).sum(dim=1).clamp_min(1.0)
    return (((prediction - targets) / std).square() * mask).flatten(1).sum(dim=1) / denom


def marginal_utilities_for_state(
    prediction_stack: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    queried_ids: torch.Tensor,
    metric_std: torch.Tensor,
) -> torch.Tensor:
    batch, _, _, num_experts = prediction_stack.shape
    current_count = (queried_ids >= 0).sum(dim=1)
    has_current = current_count > 0
    current_prediction = current_average_from_ids(prediction_stack, queried_ids)
    current_loss = scaled_sample_mae(current_prediction, targets, masks, metric_std)
    utilities = []
    row_ids = torch.arange(batch, device=prediction_stack.device)
    for expert_id in range(num_experts):
        next_ids = queried_ids.clone()
        insert_slot = current_count.clamp(max=queried_ids.shape[1] - 1)
        next_ids[row_ids, insert_slot] = expert_id
        candidate_prediction = current_average_from_ids(prediction_stack, next_ids)
        candidate_loss = scaled_sample_mae(candidate_prediction, targets, masks, metric_std)
        utilities.append(torch.where(has_current, current_loss - candidate_loss, -candidate_loss))
    return torch.stack(utilities, dim=1)


def state_supervision(
    prediction_stack: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    queried_ids: torch.Tensor,
    metric_std: torch.Tensor,
    lambda_cost: float,
    num_experts: int,
) -> StateBatch:
    queried_mask, queried_forecasts, current_average = make_state_fast(prediction_stack, queried_ids, num_experts)
    available = ~queried_mask.to(torch.bool)
    utilities = marginal_utilities_for_state(prediction_stack, targets, masks, queried_ids, metric_std)
    utilities = utilities.masked_fill(~available, 0.0)
    best_remaining = utilities.masked_fill(~available, -1e9).max(dim=1).values
    current_count = (queried_ids >= 0).sum(dim=1)
    stop_target = ((current_count > 0) & (best_remaining <= float(lambda_cost))).to(torch.float32)
    return StateBatch(
        queried_ids=queried_ids,
        queried_mask=queried_mask,
        queried_forecasts=queried_forecasts,
        current_average=current_average,
        utilities=utilities.detach(),
        available=available,
        stop_target=stop_target.detach(),
        current_count=current_count,
    )


def make_state_fast(
    prediction_stack: torch.Tensor,
    queried_ids: torch.Tensor,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, horizon, features, _ = prediction_stack.shape
    valid = queried_ids >= 0
    safe_ids = queried_ids.clamp_min(0)
    queried_mask = torch.zeros((batch, num_experts), dtype=prediction_stack.dtype, device=prediction_stack.device)
    queried_mask.scatter_add_(1, safe_ids, valid.to(prediction_stack.dtype))
    queried_mask = queried_mask.clamp_max(1.0)
    expanded = safe_ids[:, :, None, None].expand(-1, -1, horizon, features)
    source = prediction_stack.permute(0, 3, 1, 2)
    queried_forecasts = source.gather(1, expanded)
    queried_forecasts = queried_forecasts * valid[:, :, None, None].to(prediction_stack.dtype)
    denom = valid.sum(dim=1).clamp_min(1).to(prediction_stack.dtype)
    current_average = queried_forecasts.sum(dim=1) / denom[:, None, None]
    return queried_mask, queried_forecasts, current_average


def random_order(batch_size: int, num_experts: int, device: torch.device) -> torch.Tensor:
    return torch.stack([torch.randperm(num_experts, device=device) for _ in range(batch_size)], dim=0)


def trajectory_orders(
    prediction_stack: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    kinds: Sequence[str],
) -> list[torch.Tensor]:
    orders = []
    for kind in kinds:
        if kind == "oracle":
            orders.append(greedy_oracle_order(prediction_stack, targets, masks))
        elif kind == "random":
            orders.append(random_order(prediction_stack.shape[0], prediction_stack.shape[-1], prediction_stack.device))
        else:
            raise ValueError(f"Unknown trajectory kind: {kind}")
    return orders


def ids_for_step(order: torch.Tensor, step: int, max_queries: int) -> torch.Tensor:
    queried_ids = torch.full((order.shape[0], max_queries), -1, dtype=torch.long, device=order.device)
    if step > 0:
        queried_ids[:, :step] = order[:, :step]
    return queried_ids


def scalar_features(
    model: ForecastStateStopRouter,
    state: StateBatch,
) -> torch.Tensor:
    del model
    valid = state.queried_ids >= 0
    counts = valid.sum(dim=1).to(state.queried_forecasts.dtype)
    pair_values = []
    pair_masks = []
    for left in range(state.queried_forecasts.shape[1]):
        for right in range(left + 1, state.queried_forecasts.shape[1]):
            pair_values.append(torch.mean(torch.abs(state.queried_forecasts[:, left] - state.queried_forecasts[:, right]), dim=(1, 2)))
            pair_masks.append(valid[:, left] & valid[:, right])
    if not pair_values:
        pairwise_mean = torch.zeros_like(counts)
        pairwise_max = torch.zeros_like(counts)
    else:
        values = torch.stack(pair_values, dim=1)
        masks = torch.stack(pair_masks, dim=1)
        masked_values = values * masks.to(values.dtype)
        pairwise_mean = masked_values.sum(dim=1) / masks.sum(dim=1).clamp_min(1).to(values.dtype)
        pairwise_max = values.masked_fill(~masks, 0.0).max(dim=1).values
    return torch.stack((pairwise_mean, pairwise_max, counts / max(state.queried_mask.shape[1], 1)), dim=1)


def ablated_inputs(
    model: ForecastStateStopRouter,
    history: torch.Tensor,
    state: StateBatch,
    ablation: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    zeros_mask = torch.zeros_like(state.queried_mask)
    zeros_ids = torch.full_like(state.queried_ids, -1)
    zeros_forecasts = torch.zeros_like(state.queried_forecasts)
    zeros_average = torch.zeros_like(state.current_average)
    zeros_scalar = torch.zeros((history.shape[0], 3), dtype=history.dtype, device=history.device)

    if ablation == "history_only":
        return zeros_mask, zeros_ids, zeros_forecasts, zeros_average, zeros_scalar
    if ablation == "history_mask":
        return state.queried_mask, zeros_ids, zeros_forecasts, zeros_average, zeros_scalar
    if ablation == "history_queried_forecasts":
        return state.queried_mask, state.queried_ids, state.queried_forecasts, zeros_average, zeros_scalar
    if ablation == "history_current_ensemble":
        return state.queried_mask, zeros_ids, zeros_forecasts, state.current_average, zeros_scalar
    if ablation == "history_disagreement":
        return state.queried_mask, zeros_ids, zeros_forecasts, zeros_average, scalar_features(model, state)
    if ablation == "full":
        return state.queried_mask, state.queried_ids, state.queried_forecasts, state.current_average, scalar_features(model, state)
    raise ValueError(f"Unknown ablation: {ablation}")


def masked_weighted_pairwise_loss(
    scores: torch.Tensor,
    utilities: torch.Tensor,
    available: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    score_diff = scores[:, :, None] - scores[:, None, :]
    utility_diff = utilities[:, :, None] - utilities[:, None, :]
    pair_mask = available[:, :, None] & available[:, None, :]
    pair_mask = pair_mask & (utility_diff > float(epsilon))
    if not bool(pair_mask.any()):
        return scores.sum() * 0.0
    weights = utility_diff[pair_mask].detach()
    return (F.softplus(-score_diff[pair_mask]) * weights).sum() / weights.sum().clamp_min(1e-12)


def total_state_loss(
    outputs: Mapping[str, torch.Tensor],
    state: StateBatch,
    alpha_utility: float,
    alpha_rank: float,
    alpha_stop: float,
    pairwise_epsilon: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    scores = outputs["utility_prediction"]
    utility_loss = F.smooth_l1_loss(scores.masked_select(state.available), state.utilities.masked_select(state.available))
    rank_loss = masked_weighted_pairwise_loss(scores, state.utilities, state.available, pairwise_epsilon)
    stop_loss = F.binary_cross_entropy_with_logits(outputs["stop_logit"], state.stop_target)
    total = float(alpha_utility) * utility_loss + float(alpha_rank) * rank_loss + float(alpha_stop) * stop_loss
    return total, {
        "utility_loss": float(utility_loss.detach().cpu()),
        "rank_loss": float(rank_loss.detach().cpu()),
        "stop_loss": float(stop_loss.detach().cpu()),
    }


def concat_states(states: Sequence[StateBatch]) -> StateBatch:
    return StateBatch(
        queried_ids=torch.cat([state.queried_ids for state in states], dim=0),
        queried_mask=torch.cat([state.queried_mask for state in states], dim=0),
        queried_forecasts=torch.cat([state.queried_forecasts for state in states], dim=0),
        current_average=torch.cat([state.current_average for state in states], dim=0),
        utilities=torch.cat([state.utilities for state in states], dim=0),
        available=torch.cat([state.available for state in states], dim=0),
        stop_target=torch.cat([state.stop_target for state in states], dim=0),
        current_count=torch.cat([state.current_count for state in states], dim=0),
    )


def pearson_corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 2:
        return math.nan
    x = torch.tensor(xs, dtype=torch.float64)
    y = torch.tensor(ys, dtype=torch.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt((x.square().sum() * y.square().sum()).clamp_min(1e-12))
    return float((x * y).sum().item() / denom.item())


def train_one_epoch(
    model: ForecastStateStopRouter,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    metric_std: torch.Tensor,
) -> dict[str, float]:
    model.train()
    trajectory_kinds = [item.strip() for item in args.trajectory_kinds.split(",") if item.strip()]
    losses = []
    parts: dict[str, list[float]] = {"utility_loss": [], "rank_loss": [], "stop_loss": []}
    for batch in loader:
        history = batch["history"].to(device=device, dtype=torch.float32)
        targets = batch["targets"].to(device=device, dtype=torch.float32)
        masks = batch["target_masks"].to(device=device, dtype=torch.bool)
        prediction_stack = batch["prediction_stack"].to(device=device, dtype=torch.float32)
        state_chunks = []
        history_chunks = []
        for order in trajectory_orders(prediction_stack, targets, masks, trajectory_kinds):
            for step in range(min(args.max_queries, model.num_experts)):
                queried_ids = ids_for_step(order, step, args.max_queries)
                state_chunks.append(
                    state_supervision(
                        prediction_stack,
                        targets,
                        masks,
                        queried_ids,
                        metric_std,
                        args.lambda_cost,
                        model.num_experts,
                    )
                )
                history_chunks.append(history)
        state = concat_states(state_chunks)
        state_history = torch.cat(history_chunks, dim=0)
        model_mask, model_ids, model_forecasts, model_average, model_scalar = ablated_inputs(
            model,
            state_history,
            state,
            args.ablation,
        )
        outputs = model(
            state_history,
            model_mask,
            model_ids,
            model_forecasts,
            model_average,
            scalar_features=model_scalar,
        )
        batch_total, loss_parts = total_state_loss(
            outputs,
            state,
            args.alpha_utility,
            args.alpha_rank,
            args.alpha_stop,
            args.pairwise_epsilon,
        )
        for key, value in loss_parts.items():
            parts[key].append(value)
        optimizer.zero_grad()
        batch_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        optimizer.step()
        losses.append(float(batch_total.detach().cpu()))
    out = {"loss": float(statistics.mean(losses)) if losses else math.nan}
    for key, values in parts.items():
        out[key] = float(statistics.mean(values)) if values else math.nan
    return out


@torch.no_grad()
def evaluate_router(
    model: ForecastStateStopRouter,
    cache: Mapping[str, Any],
    device: torch.device,
    args: argparse.Namespace,
    metric_std: torch.Tensor,
    stop_threshold: float,
) -> dict[str, Any]:
    model.eval()
    loader = DataLoader(CacheWindowDataset(cache), batch_size=args.batch_size, shuffle=False)
    maes: list[torch.Tensor] = []
    mses: list[torch.Tensor] = []
    query_counts: list[torch.Tensor] = []
    utility_scores: list[float] = []
    utility_targets: list[float] = []
    top1_hits: list[float] = []
    stop_hits: list[float] = []
    false_stops: list[float] = []
    unnecessary_queries: list[float] = []
    stop_after = torch.zeros(model.num_experts, dtype=torch.float64)
    query_distribution = torch.zeros(model.num_experts, dtype=torch.float64)
    per_window = []
    offset = 0

    for batch in loader:
        history = batch["history"].to(device=device, dtype=torch.float32)
        targets = batch["targets"].to(device=device, dtype=torch.float32)
        masks = batch["target_masks"].to(device=device, dtype=torch.bool)
        prediction_stack = batch["prediction_stack"].to(device=device, dtype=torch.float32)
        queried_ids = torch.full((history.shape[0], args.max_queries), -1, dtype=torch.long, device=device)
        active = torch.ones(history.shape[0], dtype=torch.bool, device=device)
        for step in range(min(args.max_queries, model.num_experts)):
            state = state_supervision(
                prediction_stack,
                targets,
                masks,
                queried_ids,
                metric_std,
                args.lambda_cost,
                model.num_experts,
            )
            model_mask, model_ids, model_forecasts, model_average, model_scalar = ablated_inputs(
                model,
                history,
                state,
                args.ablation,
            )
            outputs = model(
                history,
                model_mask,
                model_ids,
                model_forecasts,
                model_average,
                scalar_features=model_scalar,
            )
            scores = outputs["utility_prediction"].masked_fill(~state.available, -1e9)
            utilities = state.utilities.masked_fill(~state.available, -1e9)
            values, next_ids = scores.max(dim=1)
            best_utility, best_ids = utilities.max(dim=1)
            stop_prob = torch.sigmoid(outputs["stop_logit"])
            oracle_stop = (state.current_count > 0) & (best_utility <= float(args.lambda_cost))
            predicted_stop = (state.current_count > 0) & (stop_prob >= float(stop_threshold))
            forced_initial = step == 0
            should_query = active & ((forced_initial | ~predicted_stop) & (step + 1 <= args.max_queries))
            should_query = should_query & state.available.any(dim=1)

            visited = active.detach().cpu()
            if bool(visited.any()):
                available_cpu = state.available.detach().cpu()
                scores_cpu = outputs["utility_prediction"].detach().cpu()
                utilities_cpu = state.utilities.detach().cpu()
                for row in torch.where(visited)[0].tolist():
                    mask = available_cpu[row]
                    utility_scores.extend(scores_cpu[row, mask].tolist())
                    utility_targets.extend(utilities_cpu[row, mask].tolist())
                top1_hits.extend((next_ids[active] == best_ids[active]).to(torch.float32).detach().cpu().tolist())
                stop_hits.extend((predicted_stop[active] == oracle_stop[active]).to(torch.float32).detach().cpu().tolist())
                false_stops.extend((predicted_stop[active] & ~oracle_stop[active]).to(torch.float32).detach().cpu().tolist())
                unnecessary_queries.extend((~predicted_stop[active] & oracle_stop[active]).to(torch.float32).detach().cpu().tolist())

            if not bool(should_query.any()):
                break
            queried_ids[should_query, step] = next_ids[should_query]
            active = active & should_query & (step + 1 < args.max_queries)

        final_prediction = current_average_from_ids(prediction_stack, queried_ids)
        batch_mae = scaled_sample_mae(final_prediction, targets, masks, metric_std).cpu()
        batch_mse = scaled_sample_mse(final_prediction, targets, masks, metric_std).cpu()
        counts = (queried_ids >= 0).sum(dim=1).cpu()
        maes.append(batch_mae)
        mses.append(batch_mse)
        query_counts.append(counts)
        for row in range(queried_ids.shape[0]):
            count = int(counts[row].item())
            query_distribution[count - 1] += 1
            if count < model.num_experts:
                stop_after[count - 1] += 1
            ids = queried_ids[row][queried_ids[row] >= 0].detach().cpu().tolist()
            per_window.append(
                {
                    "cache_index": offset + row,
                    "absolute_window_start": int(batch["absolute_window_start"][row].item()),
                    "query_count": count,
                    "queried_experts": " ".join(str(item) for item in ids),
                    "mae": float(batch_mae[row].item()),
                    "mse": float(batch_mse[row].item()),
                }
            )
        offset += history.shape[0]

    total = float(cache["num_windows"])
    mae_tensor = torch.cat(maes)
    mse_tensor = torch.cat(mses)
    counts_tensor = torch.cat(query_counts).to(torch.float32)
    return {
        "mae": float(mae_tensor.mean().item()),
        "mse": float(mse_tensor.mean().item()),
        "average_queries": float(counts_tensor.mean().item()),
        "query_count_distribution": {str(i + 1): float(query_distribution[i].item() * 100.0 / total) for i in range(model.num_experts)},
        "stop_after_1_percent": float(stop_after[0].item() * 100.0 / total),
        "stop_after_2_percent": float(stop_after[1].item() * 100.0 / total),
        "stop_after_3_percent": float(stop_after[2].item() * 100.0 / total),
        "all_experts_percent": float(query_distribution[model.num_experts - 1].item() * 100.0 / total),
        "marginal_utility_correlation": pearson_corr(utility_scores, utility_targets),
        "top1_next_query_accuracy": float(statistics.mean(top1_hits) * 100.0) if top1_hits else math.nan,
        "stop_accuracy": float(statistics.mean(stop_hits) * 100.0) if stop_hits else math.nan,
        "false_stop_rate": float(statistics.mean(false_stops) * 100.0) if false_stops else math.nan,
        "unnecessary_query_rate": float(statistics.mean(unnecessary_queries) * 100.0) if unnecessary_queries else math.nan,
        "per_window": per_window,
    }


def fixed_subset_metrics(cache: Mapping[str, Any], metric_std: torch.Tensor) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    experts = list(cache["expert_names"])
    stack = cache["prediction_stack"].to(torch.float32)
    targets = cache["targets"].to(torch.float32)
    masks = cache["target_masks"].to(torch.bool)
    rows = []
    best_by_size = {}
    for size in range(1, len(experts) + 1):
        for subset in itertools.combinations(range(len(experts)), size):
            prediction = stack[..., list(subset)].mean(dim=-1)
            row = {
                "subset": "+".join(experts[index] for index in subset),
                "num_experts": size,
                "mae": float(scaled_sample_mae(prediction, targets, masks, metric_std).mean().item()),
                "mse": float(scaled_sample_mse(prediction, targets, masks, metric_std).mean().item()),
            }
            rows.append(row)
        best_by_size[str(size)] = min((row for row in rows if row["num_experts"] == size), key=lambda item: item["mae"])
    oracle_by_budget = {}
    for budget in range(1, len(experts) + 1):
        candidate_rows = [row for row in rows if row["num_experts"] <= budget]
        per_subset = []
        for row in candidate_rows:
            subset = [experts.index(name) for name in row["subset"].split("+")]
            prediction = stack[..., subset].mean(dim=-1)
            per_subset.append(scaled_sample_mae(prediction, targets, masks, metric_std))
        oracle_by_budget[str(budget)] = float(torch.stack(per_subset, dim=1).min(dim=1).values.mean().item())
    return rows, {"best_by_size": best_by_size, "oracle_by_budget_mae": oracle_by_budget}


def run_seed(
    seed: int,
    train_cache: Mapping[str, Any],
    val_cache: Mapping[str, Any],
    metric_std: torch.Tensor,
    args: argparse.Namespace,
    result_root: Path,
    checkpoint_root: Path,
) -> dict[str, Any]:
    set_seed(seed)
    device = torch.device(args.device)
    model = ForecastStateStopRouter(
        num_experts=len(train_cache["expert_names"]),
        max_subset_size=args.max_queries,
        input_len=int(train_cache["input_len"]),
        forecast_horizon=int(train_cache["forecast_horizon"]),
        num_features=int(train_cache["num_features"]),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loader = DataLoader(CacheWindowDataset(train_cache), batch_size=args.batch_size, shuffle=True)
    seed_result = result_root / f"seed_{seed}"
    seed_ckpt = checkpoint_root / f"seed_{seed}"
    seed_result.mkdir(parents=True, exist_ok=True)
    seed_ckpt.mkdir(parents=True, exist_ok=True)
    curves = []
    best_metrics: dict[str, Any] | None = None
    best_state: dict[str, Any] | None = None
    best_epoch = -1
    best_mae = math.inf
    bad_epochs = 0
    for epoch in range(1, args.max_epochs + 1):
        train_metrics = train_one_epoch(model, loader, optimizer, device, args, metric_std)
        val_metrics = evaluate_router(model, val_cache, device, args, metric_std, args.primary_stop_threshold)
        curves.append(
            {
                "epoch": epoch,
                **train_metrics,
                "validation_mae": val_metrics["mae"],
                "validation_mse": val_metrics["mse"],
                "average_queries": val_metrics["average_queries"],
                "stop_accuracy": val_metrics["stop_accuracy"],
                "top1_next_query_accuracy": val_metrics["top1_next_query_accuracy"],
                "marginal_utility_correlation": val_metrics["marginal_utility_correlation"],
            }
        )
        if val_metrics["mae"] < best_mae:
            best_mae = val_metrics["mae"]
            best_metrics = val_metrics
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break
    assert best_metrics is not None and best_state is not None
    model.load_state_dict(best_state)
    frontier = []
    per_threshold_metrics = {}
    for threshold in [float(item.strip()) for item in args.stop_thresholds.split(",") if item.strip()]:
        metrics = evaluate_router(model, val_cache, device, args, metric_std, threshold)
        per_threshold_metrics[str(threshold)] = {key: value for key, value in metrics.items() if key != "per_window"}
        frontier.append({"stop_threshold": threshold, "average_queries": metrics["average_queries"], "mae": metrics["mae"], "mse": metrics["mse"]})
        if abs(threshold - args.primary_stop_threshold) < 1e-12:
            write_csv(seed_result / "validation_per_window.csv", metrics["per_window"])
    write_csv(seed_result / "training_curves.csv", curves)
    write_csv(seed_result / "query_budget_frontier.csv", frontier)
    torch.save(
        {
            "router_type": "ForecastStateStopRouter",
            "router_config": {
                "num_experts": len(train_cache["expert_names"]),
                "max_subset_size": args.max_queries,
                "input_len": int(train_cache["input_len"]),
                "forecast_horizon": int(train_cache["forecast_horizon"]),
                "num_features": int(train_cache["num_features"]),
                "embedding_dim": args.embedding_dim,
                "hidden_dim": args.hidden_dim,
            },
            "router_state_dict": best_state,
            "ablation": args.ablation,
            "seed": seed,
            "best_epoch": best_epoch,
            "lambda_cost": args.lambda_cost,
            "primary_stop_threshold": args.primary_stop_threshold,
            "validation_metrics": {key: value for key, value in best_metrics.items() if key != "per_window"},
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "safety": "NO TEST DATA USED",
        },
        seed_ckpt / "best_forecast_state_router.pt",
    )
    primary = per_threshold_metrics[str(float(args.primary_stop_threshold))]
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "mae": primary["mae"],
        "mse": primary["mse"],
        "average_queries": primary["average_queries"],
        "stop_after_1_percent": primary["stop_after_1_percent"],
        "stop_after_2_percent": primary["stop_after_2_percent"],
        "stop_after_3_percent": primary["stop_after_3_percent"],
        "all_experts_percent": primary["all_experts_percent"],
        "marginal_utility_correlation": primary["marginal_utility_correlation"],
        "top1_next_query_accuracy": primary["top1_next_query_accuracy"],
        "stop_accuracy": primary["stop_accuracy"],
        "false_stop_rate": primary["false_stop_rate"],
        "unnecessary_query_rate": primary["unnecessary_query_rate"],
        "frontier": frontier,
    }


def summarize_frontier(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[float, dict[str, list[float]]] = {}
    for row in rows:
        for point in row["frontier"]:
            threshold = float(point["stop_threshold"])
            grouped.setdefault(threshold, {"average_queries": [], "mae": [], "mse": []})
            grouped[threshold]["average_queries"].append(float(point["average_queries"]))
            grouped[threshold]["mae"].append(float(point["mae"]))
            grouped[threshold]["mse"].append(float(point["mse"]))
    out = []
    for threshold, values in sorted(grouped.items()):
        q_mean, q_std = aggregate(values["average_queries"])
        mae_mean, mae_std = aggregate(values["mae"])
        mse_mean, mse_std = aggregate(values["mse"])
        out.append(
            {
                "stop_threshold": threshold,
                "average_queries_mean": q_mean,
                "average_queries_std": q_std,
                "mae_mean": mae_mean,
                "mae_std": mae_std,
                "mse_mean": mse_mean,
                "mse_std": mse_std,
            }
        )
    return out


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in (
        "mae",
        "mse",
        "average_queries",
        "stop_after_1_percent",
        "stop_after_2_percent",
        "stop_after_3_percent",
        "all_experts_percent",
        "marginal_utility_correlation",
        "top1_next_query_accuracy",
        "stop_accuracy",
        "false_stop_rate",
        "unnecessary_query_rate",
    ):
        mean, std = aggregate([float(row[key]) for row in rows])
        summary[f"{key}_mean"] = mean
        summary[f"{key}_std"] = std
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", default="cache/costarts_walkforward/router_train_20_60_cache.pt")
    parser.add_argument("--val-cache", default="cache/costarts_walkforward/router_val_60_80_cache.pt")
    parser.add_argument("--normalizer-checkpoint", default="checkpoints/costarts_walkforward/final_60/DLinear/best_expert.pt")
    parser.add_argument("--checkpoint-root", default="checkpoints/costarts_walkforward/forecast_state")
    parser.add_argument("--results-root", default="results/router_summary/costarts_walkforward/forecast_state")
    parser.add_argument("--ablation", choices=ABLATIONS, default="full")
    parser.add_argument("--trajectory-kinds", default="oracle,random")
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
    parser.add_argument("--lambda-cost", type=float, default=0.0)
    parser.add_argument("--alpha-utility", type=float, default=1.0)
    parser.add_argument("--alpha-rank", type=float, default=1.0)
    parser.add_argument("--alpha-stop", type=float, default=1.0)
    parser.add_argument("--pairwise-epsilon", type=float, default=0.001)
    parser.add_argument("--primary-stop-threshold", type=float, default=0.5)
    parser.add_argument("--stop-thresholds", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    train_cache_path = ROOT / args.train_cache
    val_cache_path = ROOT / args.val_cache
    train_cache = load_cache(train_cache_path)
    val_cache = load_cache(val_cache_path)
    if tuple(train_cache["expert_names"]) != tuple(val_cache["expert_names"]):
        raise ValueError("Expert order mismatch between train and validation caches")
    metric_std = load_metric_std(args.normalizer_checkpoint, int(train_cache["num_features"]))
    seeds = tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
    result_root = ROOT / args.results_root / args.ablation
    checkpoint_root = ROOT / args.checkpoint_root / args.ablation
    result_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    baseline_rows, baseline_summary = fixed_subset_metrics(val_cache, metric_std)
    write_csv(result_root / "fixed_ensembles.csv", sorted(baseline_rows, key=lambda row: (row["num_experts"], row["mae"])))

    rows = []
    for seed in seeds:
        rows.append(run_seed(seed, train_cache, val_cache, metric_std, args, result_root, checkpoint_root))
    per_seed_rows = [{key: value for key, value in row.items() if key != "frontier"} for row in rows]
    write_csv(result_root / "per_seed_results.csv", per_seed_rows)
    frontier = summarize_frontier(rows)
    write_csv(result_root / "query_budget_frontier.csv", frontier)
    summary = {
        "method": "forecast_state_partial_subset_stop_router",
        "ablation": args.ablation,
        "seeds": list(seeds),
        "train_cache": args.train_cache,
        "val_cache": args.val_cache,
        "train_cache_sha256": sha256_file(train_cache_path),
        "val_cache_sha256": sha256_file(val_cache_path),
        "expert_names": list(train_cache["expert_names"]),
        "input_len": int(train_cache["input_len"]),
        "forecast_horizon": int(train_cache["forecast_horizon"]),
        "num_features": int(train_cache["num_features"]),
        "num_train_windows": int(train_cache["num_windows"]),
        "num_val_windows": int(val_cache["num_windows"]),
        "validation_start": int(val_cache["absolute_window_starts"].min().item()),
        "validation_end": int(val_cache["absolute_window_starts"].max().item()) + int(val_cache["forecast_horizon"]),
        "lambda_cost": args.lambda_cost,
        "primary_stop_threshold": args.primary_stop_threshold,
        "trajectory_kinds": args.trajectory_kinds,
        "loss_weights": {
            "alpha_utility": args.alpha_utility,
            "alpha_rank": args.alpha_rank,
            "alpha_stop": args.alpha_stop,
        },
        "aggregate": summarize_rows(rows),
        "query_budget_frontier": frontier,
        "fixed_ensemble_summary": baseline_summary,
        "per_seed": per_seed_rows,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "safety": "NO TEST DATA USED",
    }
    (result_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
