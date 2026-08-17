"""Train and evaluate a forecast-adaptive sequential COSTAR-TS router.

This experimental path keeps the main history-only COSTAR-TS router unchanged.
It consumes the frozen-expert COSTAR subset-state caches and rolls out a
query-observe-decide policy on router validation only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.old.build_costarts_subset_states import validate_costarts_subset_states
from scripts.old.train_costarts_subset_utility_router import _build_state_lookup, _state_batch


DEFAULT_TRAIN_CACHE = "cache/costarts_subset_states_train.pt"
DEFAULT_VAL_CACHE = "cache/costarts_subset_states_val.pt"
DEFAULT_OUTPUT_DIR = "checkpoints/costarts_adaptive"
DEFAULT_RESULTS_DIR = "results/router_summary/costarts_adaptive"


@dataclass
class AdaptiveTrainingConfig:
    train_cache: str = DEFAULT_TRAIN_CACHE
    val_cache: str = DEFAULT_VAL_CACHE
    output_dir: str = DEFAULT_OUTPUT_DIR
    results_dir: str = DEFAULT_RESULTS_DIR
    seed: int = 7
    batch_size: int = 512
    max_epochs: int = 8
    patience: int = 3
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    device: str = "cpu"
    cost: float = 0.0
    max_queries: int = 5
    variant: str = "forecast_disagreement"
    train_state_mode: str = "all"
    utility_loss_weight: float = 1.0
    action_loss_weight: float = 0.5
    stop_loss_weight: float = 0.5
    max_train_states: int | None = None


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class COSTARTSAdaptiveRouter(nn.Module):
    """Forecast-adaptive COSTAR-TS router for query-observe-decide experiments."""

    def __init__(
        self,
        num_experts: int,
        input_len: int = 96,
        forecast_horizon: int = 12,
        num_features: int = 7,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        variant: str = "forecast_disagreement",
    ) -> None:
        super().__init__()
        if variant not in {"mask_only", "forecast", "forecast_disagreement"}:
            raise ValueError("variant must be mask_only, forecast, or forecast_disagreement")
        self.num_experts = int(num_experts)
        self.input_len = int(input_len)
        self.forecast_horizon = int(forecast_horizon)
        self.num_features = int(num_features)
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.variant = str(variant)

        self.history_encoder = nn.Sequential(
            nn.Conv1d(num_features, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.GroupNorm(1, hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=4, dilation=2),
            nn.GELU(),
            nn.GroupNorm(1, hidden_dim),
            nn.AdaptiveAvgPool1d(1),
        )
        self.history_projection = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
        )
        self.mask_projection = nn.Sequential(
            nn.Linear(num_experts + 1, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
        )
        self.expert_embeddings = nn.Parameter(torch.empty(num_experts, embedding_dim))
        nn.init.normal_(self.expert_embeddings, mean=0.0, std=0.02)
        flat_forecast = forecast_horizon * num_features
        self.forecast_encoder = nn.Sequential(
            nn.Linear(flat_forecast, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
        )
        disagreement_dim = flat_forecast + forecast_horizon + num_features + 2
        self.disagreement_encoder = nn.Sequential(
            nn.Linear(disagreement_dim, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
        )
        fusion_dim = embedding_dim * 4
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.action_head = nn.Linear(hidden_dim, num_experts + 1)
        self.utility_head = nn.Linear(hidden_dim, num_experts)

    def config_dict(self) -> dict[str, Any]:
        return {
            "num_experts": self.num_experts,
            "input_len": self.input_len,
            "forecast_horizon": self.forecast_horizon,
            "num_features": self.num_features,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "variant": self.variant,
        }

    def encode_state(
        self,
        history: torch.Tensor,
        queried_mask: torch.Tensor,
        queried_expert_ids: torch.Tensor,
        queried_expert_forecasts: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = history.shape[0]
        valid_slots = queried_expert_ids >= 0
        slot_count = valid_slots.sum(dim=1).clamp_min(1).to(history.dtype)
        history_rep = self.history_projection(
            self.history_encoder(history.transpose(1, 2)).squeeze(-1)
        )
        mask_features = torch.cat(
            (
                queried_mask.to(history.dtype),
                valid_slots.sum(dim=1, keepdim=True).to(history.dtype) / max(self.num_experts, 1),
            ),
            dim=1,
        )
        mask_rep = self.mask_projection(mask_features)

        zero_rep = torch.zeros(batch_size, self.embedding_dim, device=history.device, dtype=history.dtype)
        forecast_rep = zero_rep
        disagreement_rep = zero_rep
        if self.variant != "mask_only":
            forecast_flat = queried_expert_forecasts.reshape(batch_size, self.num_experts, -1)
            encoded_slots = self.forecast_encoder(forecast_flat)
            safe_ids = queried_expert_ids.clamp_min(0)
            encoded_slots = encoded_slots + self.expert_embeddings[safe_ids]
            encoded_slots = encoded_slots * valid_slots.unsqueeze(-1).to(history.dtype)
            forecast_rep = encoded_slots.sum(dim=1) / slot_count[:, None]

        if self.variant == "forecast_disagreement":
            masked_forecasts = queried_expert_forecasts * valid_slots[:, :, None, None].to(history.dtype)
            mean_forecast = masked_forecasts.sum(dim=1) / slot_count[:, None, None]
            centered = (queried_expert_forecasts - mean_forecast[:, None]) * valid_slots[:, :, None, None].to(history.dtype)
            variance = centered.pow(2).sum(dim=1) / slot_count[:, None, None]
            horizon_disagreement = variance.mean(dim=2)
            variable_disagreement = variance.mean(dim=1)
            pairwise_mean, pairwise_max = _pairwise_forecast_difference_features(
                queried_expert_forecasts,
                valid_slots,
            )
            disagreement_features = torch.cat(
                (
                    variance.reshape(batch_size, -1),
                    horizon_disagreement,
                    variable_disagreement,
                    pairwise_mean[:, None],
                    pairwise_max[:, None],
                ),
                dim=1,
            )
            disagreement_rep = self.disagreement_encoder(disagreement_features)

        return self.fusion(torch.cat((history_rep, mask_rep, forecast_rep, disagreement_rep), dim=1))

    def forward(
        self,
        history: torch.Tensor,
        queried_mask: torch.Tensor,
        queried_expert_ids: torch.Tensor,
        queried_expert_forecasts: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        representation = self.encode_state(
            history,
            queried_mask,
            queried_expert_ids,
            queried_expert_forecasts,
        )
        return {
            "representation": representation,
            "action_logits": self.action_head(representation),
            "utility_prediction": self.utility_head(representation),
        }


def _pairwise_forecast_difference_features(
    queried_forecasts: torch.Tensor,
    valid_slots: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, max_slots = valid_slots.shape
    means = []
    maxes = []
    for row in range(batch_size):
        valid = valid_slots[row]
        if int(valid.sum()) < 2:
            means.append(torch.zeros((), device=queried_forecasts.device, dtype=queried_forecasts.dtype))
            maxes.append(torch.zeros((), device=queried_forecasts.device, dtype=queried_forecasts.dtype))
            continue
        forecasts = queried_forecasts[row, valid]
        diffs = []
        for left in range(forecasts.shape[0]):
            for right in range(left + 1, forecasts.shape[0]):
                diffs.append(torch.mean(torch.abs(forecasts[left] - forecasts[right])))
        pairwise = torch.stack(diffs)
        means.append(pairwise.mean())
        maxes.append(pairwise.max())
    assert len(means) == batch_size and max_slots == valid_slots.shape[1]
    return torch.stack(means), torch.stack(maxes)


def _load_torch(path: str | Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class AdaptiveSubsetStateDataset(Dataset):
    def __init__(self, cache: Mapping[str, Any], indices: Sequence[int] | None = None) -> None:
        self.cache = cache
        self.indices = list(range(int(cache["num_states"]))) if indices is None else list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        index = self.indices[item]
        keys = (
            "history",
            "queried_mask",
            "queried_expert_ids",
            "queried_expert_forecasts",
            "target_mask",
            "true_targets",
            "valid_action_mask",
            "marginal_gain_equal_queried_average",
            "optimal_next_action",
        )
        return {key: self.cache[key][index] for key in keys}


def _collate(rows: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([row[key] for row in rows], dim=0) for key in rows[0]}


def _move_batch(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def masked_action_logits(logits: torch.Tensor, valid_action_mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~valid_action_mask.to(torch.bool), -1e9)


def build_cost_sensitive_targets(batch: Mapping[str, torch.Tensor], cost: float) -> tuple[torch.Tensor, torch.Tensor]:
    utility = batch["marginal_gain_equal_queried_average"].to(torch.float32) - float(cost)
    valid = batch["valid_action_mask"][:, : utility.shape[1]].to(torch.bool)
    utility = utility.masked_fill(~valid, float("-inf"))
    best_utility, best_expert = utility.max(dim=1)
    stop_index = utility.shape[1]
    target_action = torch.where(best_utility > 0.0, best_expert, torch.full_like(best_expert, stop_index))
    stop_valid = batch["valid_action_mask"][:, stop_index].to(torch.bool)
    target_action = torch.where(stop_valid, target_action, best_expert)
    return utility, target_action.to(torch.long)


def adaptive_losses(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    *,
    cost: float,
    utility_weight: float,
    action_weight: float,
    stop_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    utility_target, action_target = build_cost_sensitive_targets(batch, cost)
    valid_action_mask = batch["valid_action_mask"].to(torch.bool)
    expert_valid = valid_action_mask[:, : outputs["utility_prediction"].shape[1]]
    finite = torch.isfinite(utility_target) & expert_valid
    if torch.any(finite):
        utility_loss = F.smooth_l1_loss(
            outputs["utility_prediction"][finite],
            utility_target[finite].to(outputs["utility_prediction"].dtype),
        )
    else:
        utility_loss = outputs["utility_prediction"].sum() * 0.0
    action_loss = F.cross_entropy(masked_action_logits(outputs["action_logits"], valid_action_mask), action_target)
    stop_index = outputs["action_logits"].shape[1] - 1
    stop_target = (action_target == stop_index).to(outputs["action_logits"].dtype)
    stop_loss = F.binary_cross_entropy_with_logits(outputs["action_logits"][:, stop_index], stop_target)
    total = utility_weight * utility_loss + action_weight * action_loss + stop_weight * stop_loss
    predicted = torch.argmax(masked_action_logits(outputs["action_logits"], valid_action_mask), dim=1)
    return total, {
        "loss": float(total.detach().cpu()),
        "utility_loss": float(utility_loss.detach().cpu()),
        "action_loss": float(action_loss.detach().cpu()),
        "stop_loss": float(stop_loss.detach().cpu()),
        "action_accuracy": float((predicted == action_target).to(torch.float32).mean().detach().cpu()),
        "stop_frequency": float((predicted == stop_index).to(torch.float32).mean().detach().cpu()),
    }


def assert_split_integrity(train_cache: Mapping[str, Any], val_cache: Mapping[str, Any]) -> Sequence[str]:
    validate_costarts_subset_states(train_cache)
    validate_costarts_subset_states(val_cache)
    if train_cache.get("split_role") != "router_train":
        raise ValueError("training cache is not router_train")
    if val_cache.get("split_role") != "router_val":
        raise ValueError("validation cache is not router_val")
    if train_cache.get("source_split_role") != "router_train":
        raise ValueError("training source split is not router_train")
    if val_cache.get("source_split_role") != "router_val":
        raise ValueError("validation source split is not router_val")
    if str(train_cache.get("source_cache_path")) == str(val_cache.get("source_cache_path")):
        raise AssertionError("router_train and router_val point at the same source cache")
    if not train_cache.get("source_sample_indices_contiguous", False):
        raise ValueError("training source sample indices are not marked contiguous")
    if not val_cache.get("source_sample_indices_contiguous", False):
        raise ValueError("validation source sample indices are not marked contiguous")
    expert_names = tuple(train_cache["expert_names"])
    if expert_names != tuple(val_cache["expert_names"]):
        raise ValueError("expert ordering mismatch")
    return expert_names


def _subset_indices_for_mode(cache: Mapping[str, Any], mode: str, max_states: int | None, seed: int) -> list[int]:
    all_indices = list(range(int(cache["num_states"])))
    if mode == "all":
        indices = all_indices
    elif mode == "deployable_oracle_mixture":
        optimal = cache["optimal_next_action"].to(torch.long)
        source = cache["source_row"].to(torch.long)
        subset_size = cache["subset_size"].to(torch.long)
        by_source_size: dict[tuple[int, int], list[int]] = {}
        for index in all_indices:
            by_source_size.setdefault((int(source[index]), int(subset_size[index])), []).append(index)
        selected = set()
        for row in range(int(cache["num_source_windows"])):
            state_index = by_source_size[(row, 0)][0]
            selected.add(state_index)
            for _ in range(int(cache["num_experts"])):
                action = int(optimal[state_index])
                if action >= int(cache["num_experts"]):
                    break
                current_mask = int(cache["queried_mask"][state_index].to(torch.int64).dot(2 ** torch.arange(int(cache["num_experts"]))))
                next_mask = current_mask | (1 << action)
                candidates = [
                    item for item in by_source_size.get((row, int(cache["subset_size"][state_index]) + 1), [])
                    if _mask_to_int(cache["queried_mask"][item]) == next_mask
                ]
                if not candidates:
                    break
                state_index = candidates[0]
                selected.add(state_index)
        indices = sorted(selected)
    else:
        raise ValueError("train_state_mode must be all or deployable_oracle_mixture")
    if max_states is not None and max_states < len(indices):
        rng = random.Random(seed)
        indices = sorted(rng.sample(indices, int(max_states)))
    return indices


def _mask_to_int(mask: torch.Tensor) -> int:
    value = 0
    for index, item in enumerate(mask.tolist()):
        if bool(item):
            value |= 1 << index
    return value


def _equal_average_from_state(batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    queried_ids = batch["queried_expert_ids"]
    queried_forecasts = batch["queried_expert_forecasts"].to(torch.float32)
    valid_slots = queried_ids >= 0
    weights = valid_slots.to(torch.float32)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (queried_forecasts * weights[:, :, None, None]).sum(dim=1)


def _mae_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[float, float]:
    mask_float = mask.to(prediction.dtype)
    denominator = mask_float.sum().clamp_min(1.0)
    mae = (torch.abs(prediction - target) * mask_float).sum() / denominator
    mse = ((prediction - target).pow(2) * mask_float).sum() / denominator
    return float(mae), float(mse)


@torch.no_grad()
def rollout_adaptive_router(
    router: COSTARTSAdaptiveRouter,
    cache: Mapping[str, Any],
    *,
    batch_size: int,
    device: torch.device,
    max_queries: int,
    forced_budget: int | None = None,
) -> dict[str, Any]:
    router.eval()
    num_experts = int(cache["num_experts"])
    num_windows = int(cache["num_source_windows"])
    state_lookup = _build_state_lookup(cache)
    masks = [0 for _ in range(num_windows)]
    done = [False for _ in range(num_windows)]
    final_state_indices = [-1 for _ in range(num_windows)]
    query_sequences: list[list[int]] = [[] for _ in range(num_windows)]
    stop_index = num_experts

    for _ in range(min(max_queries, num_experts)):
        active_rows = [row for row in range(num_windows) if not done[row]]
        if not active_rows:
            break
        state_indices = [state_lookup[row][masks[row]] for row in active_rows]
        actions = []
        for start in range(0, len(state_indices), batch_size):
            batch_indices = state_indices[start : start + batch_size]
            batch = _state_batch(cache, batch_indices, device)
            outputs = router(
                batch["history"],
                batch["queried_mask"],
                batch["queried_expert_ids"],
                batch["queried_expert_forecasts"],
            )
            logits = masked_action_logits(outputs["action_logits"], batch["valid_action_mask"])
            if forced_budget is not None:
                logits[:, stop_index] = -1e9
            actions.append(torch.argmax(logits, dim=1).detach().cpu())
        actions_tensor = torch.cat(actions)
        for local_index, action_tensor in enumerate(actions_tensor):
            row = active_rows[local_index]
            state_index = state_indices[local_index]
            action = int(action_tensor)
            if action == stop_index and forced_budget is None:
                done[row] = True
                final_state_indices[row] = state_index
                continue
            if action == stop_index:
                action = int(torch.argmax(cache["marginal_gain_equal_queried_average"][state_index].masked_fill(cache["queried_mask"][state_index], -1e9)))
            if masks[row] & (1 << action):
                done[row] = True
                final_state_indices[row] = state_index
                continue
            masks[row] |= 1 << action
            query_sequences[row].append(action)
            if len(query_sequences[row]) >= num_experts or (
                forced_budget is not None and len(query_sequences[row]) >= forced_budget
            ):
                done[row] = True
                final_state_indices[row] = state_lookup[row][masks[row]]

    for row in range(num_windows):
        if final_state_indices[row] < 0:
            final_state_indices[row] = state_lookup[row][masks[row]]

    predictions = []
    targets = []
    target_masks = []
    true_errors = []
    for start in range(0, len(final_state_indices), batch_size):
        batch = _state_batch(cache, final_state_indices[start : start + batch_size], torch.device("cpu"))
        predictions.append(_equal_average_from_state(batch))
        targets.append(batch["true_targets"].to(torch.float32))
        target_masks.append(batch["target_mask"].to(torch.bool))
        true_errors.append(batch["true_expert_error_vector"].to(torch.float32))
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    target_mask = torch.cat(target_masks)
    mae, mse = _mae_mse(prediction, target, target_mask)
    oracle_mae = float(torch.cat(true_errors).min(dim=1).values.mean())
    counts = torch.bincount(
        torch.tensor([len(sequence) for sequence in query_sequences], dtype=torch.long),
        minlength=num_experts + 1,
    )
    order_counts: dict[str, int] = {}
    for sequence in query_sequences:
        key = "->".join(cache["expert_names"][index] for index in sequence)
        order_counts[key] = order_counts.get(key, 0) + 1
    return {
        "mae": mae,
        "mse": mse,
        "average_experts_queried": float(sum(len(sequence) for sequence in query_sequences) / max(num_windows, 1)),
        "stop_step_distribution": {str(index): int(count) for index, count in enumerate(counts.tolist()) if count},
        "query_order_distribution": dict(sorted(order_counts.items(), key=lambda item: item[1], reverse=True)[:20]),
        "regret_to_oracle": mae - oracle_mae,
        "oracle_mae": oracle_mae,
        "all_queries_unique": all(len(sequence) == len(set(sequence)) for sequence in query_sequences),
    }


def _baseline_metrics(cache: Mapping[str, Any]) -> dict[str, Any]:
    error_matrix = cache["true_expert_error_vector"] if "true_expert_error_vector" in cache else cache["error_matrix"]
    expert_names = tuple(cache["expert_names"])
    empty_rows = cache["subset_size"] == 0
    source_rows = torch.where(empty_rows)[0]
    true_errors = error_matrix[source_rows].to(torch.float32)
    best_fixed_index = int(true_errors.mean(dim=0).argmin())
    return {
        "best_fixed_expert": expert_names[best_fixed_index],
        "best_fixed_expert_mae": float(true_errors[:, best_fixed_index].mean()),
        "oracle_best_expert_mae": float(true_errors.min(dim=1).values.mean()),
    }


def train_one_seed(config: AdaptiveTrainingConfig) -> dict[str, Any]:
    set_reproducible_seed(config.seed)
    device = torch.device(config.device)
    train_cache = _load_torch(config.train_cache)
    val_cache = _load_torch(config.val_cache)
    expert_names = assert_split_integrity(train_cache, val_cache)
    indices = _subset_indices_for_mode(train_cache, config.train_state_mode, config.max_train_states, config.seed)
    train_dataset = AdaptiveSubsetStateDataset(train_cache, indices)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=_collate,
        generator=torch.Generator().manual_seed(config.seed),
    )
    router = COSTARTSAdaptiveRouter(
        num_experts=len(expert_names),
        input_len=int(train_cache["history"].shape[1]),
        forecast_horizon=int(train_cache["forecast_horizon"]),
        num_features=int(train_cache["num_features"]),
        variant=config.variant,
    ).to(device)
    optimizer = torch.optim.AdamW(router.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_payload: dict[str, Any] | None = None
    best_mae = math.inf
    bad_epochs = 0
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config.max_epochs + 1):
        router.train()
        totals: dict[str, float] = {}
        seen = 0
        for batch in train_loader:
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = router(
                batch["history"],
                batch["queried_mask"],
                batch["queried_expert_ids"],
                batch["queried_expert_forecasts"],
            )
            loss, parts = adaptive_losses(
                outputs,
                batch,
                cost=config.cost,
                utility_weight=config.utility_loss_weight,
                action_weight=config.action_loss_weight,
                stop_weight=config.stop_loss_weight,
            )
            loss.backward()
            if config.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(router.parameters(), config.grad_clip_norm)
            optimizer.step()
            batch_size = int(batch["history"].shape[0])
            seen += batch_size
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + value * batch_size
        train_metrics = {key: value / max(seen, 1) for key, value in totals.items()}
        val_metrics = rollout_adaptive_router(
            router,
            val_cache,
            batch_size=config.batch_size,
            device=device,
            max_queries=config.max_queries,
        )
        payload = {
            "epoch": epoch,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        if val_metrics["mae"] < best_mae:
            best_mae = float(val_metrics["mae"])
            best_payload = payload
            bad_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "router_state_dict": router.state_dict(),
                    "router_config": router.config_dict(),
                    "training_config": asdict(config),
                    "expert_names": tuple(expert_names),
                    "model_selection": "router_val_mae",
                    "test_set_used": False,
                    "experts_loaded": False,
                    "experts_updated": False,
                },
                output_dir / f"best_costarts_adaptive_router_seed{config.seed}_{config.variant}.pt",
            )
        else:
            bad_epochs += 1
        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.6f} "
            f"val_mae={val_metrics['mae']:.6f} avg_q={val_metrics['average_experts_queried']:.3f}"
        )
        if bad_epochs >= config.patience:
            break

    assert best_payload is not None
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "config": asdict(config),
        "expert_names": list(expert_names),
        "best": best_payload,
        "baselines": _baseline_metrics(val_cache),
        "test_set_used": False,
    }
    result_path = results_dir / f"adaptive_router_seed{config.seed}_{config.variant}.json"
    result_path.write_text(json.dumps(_jsonable(result), indent=2), encoding="utf-8")
    print(f"Saved: {result_path}")
    return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train forecast-adaptive COSTAR-TS router.")
    parser.add_argument("--train-cache", default=DEFAULT_TRAIN_CACHE)
    parser.add_argument("--val-cache", default=DEFAULT_VAL_CACHE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cost", type=float, default=0.0)
    parser.add_argument("--max-queries", type=int, default=5)
    parser.add_argument("--variant", choices=("mask_only", "forecast", "forecast_disagreement"), default="forecast_disagreement")
    parser.add_argument("--train-state-mode", choices=("all", "deployable_oracle_mixture"), default="all")
    parser.add_argument("--utility-loss-weight", type=float, default=1.0)
    parser.add_argument("--action-loss-weight", type=float, default=0.5)
    parser.add_argument("--stop-loss-weight", type=float, default=0.5)
    parser.add_argument("--max-train-states", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_one_seed(AdaptiveTrainingConfig(**vars(args)))


if __name__ == "__main__":
    main()
