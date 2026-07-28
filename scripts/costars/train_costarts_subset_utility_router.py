"""Train SubsetUtilityCOSTARTSRouter from offline subset-state caches.

This training path is intentionally separate from the original COSTARTS
implementation in ``scripts/train_costarts_router.py``. It trains only router
parameters from cached frozen-expert predictions and selects checkpoints by
chronological router-val MAE under the deployable sequential inference rule.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from scripts.costars.build_costarts_subset_states import validate_costarts_subset_states
    from scripts.router_experiment_config import (
        load_router_experiment_config,
        print_router_experiment_config,
        validate_router_experiment_config,
    )
except ImportError:
    from scripts.costars.build_costarts_subset_states import validate_costarts_subset_states
    from router_experiment_config import (
        load_router_experiment_config,
        print_router_experiment_config,
        validate_router_experiment_config,
    )


DEFAULT_OUTPUT_DIR = "checkpoints/costarts_subset_utility"
DEFAULT_RESULTS_DIR = "results/router_summary/costarts_subset_utility"
DEFAULT_TRAIN_CACHE = "cache/costarts_subset_states_train.pt"
DEFAULT_VAL_CACHE = "cache/costarts_subset_states_val.pt"


@dataclass
class SubsetUtilityTrainingConfig:
    train_cache_path: str = DEFAULT_TRAIN_CACHE
    val_cache_path: str = DEFAULT_VAL_CACHE
    output_dir: str = DEFAULT_OUTPUT_DIR
    results_dir: str = DEFAULT_RESULTS_DIR
    batch_size: int = 512
    max_epochs: int = 50
    patience: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    seed: int = 7
    action_loss_weight: float = 1.0
    utility_loss_weight: float = 1.0
    pairwise_loss_weight: float = 0.2
    mix_loss_weight: float = 0.0
    subset_state_sampling_mode: str = "exhaustive"
    max_subset_size: Optional[int] = None
    cost_coefficient: float = 1.0
    use_expert_embeddings: bool = True
    history_encoder_type: str = "current"
    action_head_type: str = "unified"
    device: str = "cpu"
    debug: bool = False


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _load_torch(path: Union[str, Path]) -> dict:
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
    return value


class CostartsSubsetStateDataset(Dataset):
    """Thin tensor dataset over a generated COSTARTS subset-state cache."""

    def __init__(self, cache: Mapping[str, Any]) -> None:
        validate_costarts_subset_states(cache)
        self.cache = cache
        self.num_states = int(cache["num_states"])

    def __len__(self) -> int:
        return self.num_states

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "history": self.cache["history"][index],
            "queried_mask": self.cache["queried_mask"][index],
            "queried_expert_ids": self.cache["queried_expert_ids"][index],
            "queried_expert_forecasts": self.cache["queried_expert_forecasts"][index],
            "true_targets": self.cache["true_targets"][index],
            "target_mask": self.cache["target_mask"][index],
            "true_expert_error_vector": self.cache["true_expert_error_vector"][index],
            "cost_adjusted_utility": self.cache["cost_adjusted_utility"][index],
            "optimal_next_action": self.cache["optimal_next_action"][index],
            "valid_action_mask": self.cache["valid_action_mask"][index],
            "pairwise_labels_queried": self.cache["pairwise_labels_queried"][index],
            "pairwise_labels_remaining": self.cache["pairwise_labels_remaining"][index],
            "subset_size": self.cache["subset_size"][index],
            "sample_index": self.cache["sample_index"][index],
            "source_row": self.cache["source_row"][index],
        }


class SubsetUtilityCOSTARTSRouter(nn.Module):
    """Sequential subset-state router trained from offline frozen-expert states."""

    def __init__(
        self,
        num_experts: int,
        max_subset_size: int,
        input_len: int = 96,
        forecast_horizon: int = 12,
        num_features: int = 7,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        use_expert_embeddings: bool = True,
        history_encoder_type: str = "current",
        action_head_type: str = "unified",
    ) -> None:
        super().__init__()
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if max_subset_size <= 0 or max_subset_size > num_experts:
            raise ValueError("max_subset_size must be in [1, num_experts]")
        self.num_experts = int(num_experts)
        self.max_subset_size = int(max_subset_size)
        self.input_len = int(input_len)
        self.forecast_horizon = int(forecast_horizon)
        self.num_features = int(num_features)
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        if history_encoder_type not in {"current", "simple"}:
            raise ValueError("history_encoder_type must be 'current' or 'simple'")
        if action_head_type not in {"unified", "separate_stop_query"}:
            raise ValueError("action_head_type must be 'unified' or 'separate_stop_query'")
        self.use_expert_embeddings = bool(use_expert_embeddings)
        self.history_encoder_type = str(history_encoder_type)
        self.action_head_type = str(action_head_type)
        self.stop_action_index = self.num_experts

        if self.history_encoder_type == "simple":
            self.history_encoder = nn.Sequential(
                nn.Conv1d(num_features, hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1),
            )
        else:
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
        self.mask_encoder = nn.Sequential(
            nn.Linear(num_experts, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
        )
        self.expert_embeddings = nn.Embedding(num_experts, embedding_dim)
        if not self.use_expert_embeddings:
            self.expert_embeddings.weight.requires_grad_(False)
            with torch.no_grad():
                self.expert_embeddings.weight.zero_()
        self.forecast_encoder = nn.Sequential(
            nn.Linear(forecast_horizon * num_features, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
        )
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
        )
        if self.action_head_type == "separate_stop_query":
            self.query_head = nn.Linear(embedding_dim, num_experts)
            self.stop_head = nn.Linear(embedding_dim, 1)
            self.action_head = None
        else:
            self.action_head = nn.Linear(embedding_dim, num_experts + 1)
            self.query_head = None
            self.stop_head = None
        self.utility_head = nn.Linear(embedding_dim, num_experts)
        self.expert_score_head = nn.Linear(embedding_dim, num_experts)
        self.mix_head = nn.Linear(embedding_dim, num_experts)

    def encode(
        self,
        history: torch.Tensor,
        queried_mask: torch.Tensor,
        queried_expert_ids: torch.Tensor,
        queried_expert_forecasts: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = history.shape[0]
        assert tuple(history.shape[1:]) == (self.input_len, self.num_features)
        assert tuple(queried_mask.shape) == (batch_size, self.num_experts)
        assert tuple(queried_expert_ids.shape) == (batch_size, self.max_subset_size)
        assert tuple(queried_expert_forecasts.shape) == (
            batch_size,
            self.max_subset_size,
            self.forecast_horizon,
            self.num_features,
        )

        history_representation = self.history_encoder(history.transpose(1, 2)).squeeze(-1)
        history_representation = self.history_projection(history_representation)
        mask_representation = self.mask_encoder(queried_mask.to(history.dtype))

        valid_slots = queried_expert_ids >= 0
        safe_ids = queried_expert_ids.clamp_min(0)
        forecast_flat = queried_expert_forecasts.reshape(
            batch_size,
            self.max_subset_size,
            self.forecast_horizon * self.num_features,
        )
        forecast_representation = self.forecast_encoder(forecast_flat)
        if self.use_expert_embeddings:
            forecast_representation = forecast_representation + self.expert_embeddings(safe_ids)
        forecast_representation = forecast_representation * valid_slots.unsqueeze(-1).to(history.dtype)
        denominator = valid_slots.sum(dim=1, keepdim=True).clamp_min(1).to(history.dtype)
        queried_representation = forecast_representation.sum(dim=1) / denominator

        fused = torch.cat(
            (history_representation, mask_representation, queried_representation),
            dim=-1,
        )
        return self.fusion(fused)

    def forward(
        self,
        history: torch.Tensor,
        queried_mask: torch.Tensor,
        queried_expert_ids: torch.Tensor,
        queried_expert_forecasts: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        representation = self.encode(
            history,
            queried_mask,
            queried_expert_ids,
            queried_expert_forecasts,
        )
        if self.action_head_type == "separate_stop_query":
            action_logits = torch.cat(
                (self.query_head(representation), self.stop_head(representation)),
                dim=-1,
            )
        else:
            action_logits = self.action_head(representation)
        utility_prediction = self.utility_head(representation)
        expert_score = self.expert_score_head(representation)
        mix_logits = self.mix_head(representation)
        batch_size = history.shape[0]
        assert tuple(action_logits.shape) == (batch_size, self.num_experts + 1)
        assert tuple(utility_prediction.shape) == (batch_size, self.num_experts)
        assert tuple(expert_score.shape) == (batch_size, self.num_experts)
        assert tuple(mix_logits.shape) == (batch_size, self.num_experts)
        return {
            "representation": representation,
            "action_logits": action_logits,
            "utility_prediction": utility_prediction,
            "expert_score": expert_score,
            "mix_logits": mix_logits,
        }

    def config_dict(self) -> dict[str, int]:
        return {
            "num_experts": self.num_experts,
            "max_subset_size": self.max_subset_size,
            "input_len": self.input_len,
            "forecast_horizon": self.forecast_horizon,
            "num_features": self.num_features,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "use_expert_embeddings": self.use_expert_embeddings,
            "history_encoder_type": self.history_encoder_type,
            "action_head_type": self.action_head_type,
        }


def _move_batch(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _masked_action_logits(logits: torch.Tensor, valid_action_mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~valid_action_mask.to(torch.bool), -1e9)


def _utility_loss(prediction: torch.Tensor, target: torch.Tensor, valid_action_mask: torch.Tensor) -> torch.Tensor:
    expert_valid = valid_action_mask[:, : prediction.shape[1]].to(torch.bool)
    finite = torch.isfinite(target) & expert_valid
    if not torch.any(finite):
        return prediction.sum() * 0.0
    return F.smooth_l1_loss(prediction[finite], target[finite])


def _pairwise_loss(scores: torch.Tensor, queried_labels: torch.Tensor, remaining_labels: torch.Tensor) -> torch.Tensor:
    labels = torch.cat((queried_labels, remaining_labels), dim=0).to(torch.float32)
    repeated_scores = scores.repeat(2, 1)
    score_diff = repeated_scores.unsqueeze(2) - repeated_scores.unsqueeze(1)
    valid = labels != 0
    if not torch.any(valid):
        return scores.sum() * 0.0
    return F.softplus(-labels[valid] * score_diff[valid]).mean()


def _mix_loss(outputs: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    queried_ids = batch["queried_expert_ids"]
    valid_slots = queried_ids >= 0
    if not torch.any(valid_slots):
        return outputs["mix_logits"].sum() * 0.0
    gathered_logits = outputs["mix_logits"].gather(1, queried_ids.clamp_min(0))
    gathered_logits = gathered_logits.masked_fill(~valid_slots, -1e9)
    slot_weights = torch.softmax(gathered_logits, dim=1).masked_fill(~valid_slots, 0.0)
    slot_weights = slot_weights / slot_weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
    mixed = (batch["queried_expert_forecasts"] * slot_weights[:, :, None, None]).sum(dim=1)
    mask = batch["target_mask"].to(mixed.dtype)
    denominator = mask.sum().clamp_min(1.0)
    return (torch.abs(mixed - batch["true_targets"]) * mask).sum() / denominator


def subset_utility_losses(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    loss_weights: Mapping[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    masked_logits = _masked_action_logits(outputs["action_logits"], batch["valid_action_mask"])
    action_loss = F.cross_entropy(masked_logits, batch["optimal_next_action"].to(torch.long))
    utility_loss = _utility_loss(
        outputs["utility_prediction"],
        batch["cost_adjusted_utility"].to(outputs["utility_prediction"].dtype),
        batch["valid_action_mask"],
    )
    pairwise_loss = _pairwise_loss(
        outputs["expert_score"],
        batch["pairwise_labels_queried"],
        batch["pairwise_labels_remaining"],
    )
    mix_loss = _mix_loss(outputs, batch)
    total = (
        float(loss_weights.get("action", 1.0)) * action_loss
        + float(loss_weights.get("utility", 1.0)) * utility_loss
        + float(loss_weights.get("pairwise", 0.2)) * pairwise_loss
        + float(loss_weights.get("mix", 0.0)) * mix_loss
    )
    predicted_action = torch.argmax(masked_logits, dim=-1)
    stop_index = outputs["action_logits"].shape[-1] - 1
    return total, {
        "total_loss": float(total.detach().cpu()),
        "action_loss": float(action_loss.detach().cpu()),
        "utility_loss": float(utility_loss.detach().cpu()),
        "pairwise_loss": float(pairwise_loss.detach().cpu()),
        "mix_loss": float(mix_loss.detach().cpu()),
        "action_accuracy": float((predicted_action == batch["optimal_next_action"]).to(torch.float32).mean().detach().cpu()),
        "stop_frequency": float((predicted_action == stop_index).to(torch.float32).mean().detach().cpu()),
        "average_predicted_utility": float(outputs["utility_prediction"].detach().mean().cpu()),
    }


def _bitmask(mask: torch.Tensor) -> int:
    bits = 0
    for index, value in enumerate(mask.tolist()):
        if bool(value):
            bits |= 1 << index
    return bits


def _build_state_lookup(cache: Mapping[str, Any]) -> list[dict[int, int]]:
    lookup = [dict() for _ in range(int(cache["num_source_windows"]))]
    for state_index in range(int(cache["num_states"])):
        source_row = int(cache["source_row"][state_index])
        lookup[source_row][_bitmask(cache["queried_mask"][state_index])] = state_index
    return lookup


def _state_batch(cache: Mapping[str, Any], state_indices: Sequence[int], device: torch.device) -> dict[str, torch.Tensor]:
    index = torch.tensor(state_indices, dtype=torch.long)
    keys = (
        "history",
        "queried_mask",
        "queried_expert_ids",
        "queried_expert_forecasts",
        "true_targets",
        "target_mask",
        "true_expert_error_vector",
        "valid_action_mask",
        "subset_size",
    )
    return {key: cache[key][index].to(device) for key in keys}


@torch.no_grad()
def evaluate_deployable_inference(
    router: SubsetUtilityCOSTARTSRouter,
    cache: Mapping[str, Any],
    *,
    batch_size: int,
    device: Union[str, torch.device],
    debug: bool = False,
) -> dict[str, Any]:
    device = torch.device(device)
    router.eval()
    validate_costarts_subset_states(cache)
    if cache["subset_sampling_mode"] != "exhaustive":
        raise ValueError("Deployable validation requires exhaustive subset-state cache")

    lookup = _build_state_lookup(cache)
    num_windows = int(cache["num_source_windows"])
    num_experts = int(cache["num_experts"])
    max_subset_size = int(cache["max_subset_size"])
    stop_index = int(cache["stop_action_index"])
    masks = [0 for _ in range(num_windows)]
    done = [False for _ in range(num_windows)]
    query_history: list[list[int]] = [[] for _ in range(num_windows)]
    predicted_utility_by_step: list[list[float]] = []
    stop_counts = torch.zeros(max_subset_size + 1, dtype=torch.long)

    for step in range(max_subset_size):
        active = [index for index in range(num_windows) if not done[index]]
        if not active:
            break
        step_utilities = []
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
            masked_logits = _masked_action_logits(outputs["action_logits"], batch["valid_action_mask"])
            actions = torch.argmax(masked_logits, dim=-1).detach().cpu()
            utility_prediction = outputs["utility_prediction"].detach().cpu()
            for local_index, sample_index in enumerate(rows):
                action = int(actions[local_index])
                if action == stop_index:
                    done[sample_index] = True
                    stop_counts[len(query_history[sample_index])] += 1
                    step_utilities.append(0.0)
                    continue
                query_history[sample_index].append(action)
                masks[sample_index] |= 1 << action
                step_utilities.append(float(utility_prediction[local_index, action]))
                if len(query_history[sample_index]) >= max_subset_size:
                    done[sample_index] = True
                    stop_counts[len(query_history[sample_index])] += 1
        predicted_utility_by_step.append(step_utilities)

    for sample_index in range(num_windows):
        if not query_history[sample_index]:
            # Empty STOP is invalid in generated caches, but keep this guarded.
            state_index = lookup[sample_index][masks[sample_index]]
            batch = _state_batch(cache, [state_index], device)
            outputs = router(
                batch["history"],
                batch["queried_mask"],
                batch["queried_expert_ids"],
                batch["queried_expert_forecasts"],
            )
            action = int(torch.argmax(outputs["action_logits"][0, :num_experts]).detach().cpu())
            query_history[sample_index].append(action)
            masks[sample_index] |= 1 << action
        if not done[sample_index]:
            stop_counts[len(query_history[sample_index])] += 1

    final_state_indices = [lookup[row][masks[row]] for row in range(num_windows)]
    selected = torch.empty(num_windows, dtype=torch.long)
    for offset in range(0, num_windows, batch_size):
        rows = list(range(offset, min(offset + batch_size, num_windows)))
        batch = _state_batch(cache, final_state_indices[offset : offset + len(rows)], device)
        outputs = router(
            batch["history"],
            batch["queried_mask"],
            batch["queried_expert_ids"],
            batch["queried_expert_forecasts"],
        )
        expert_scores = outputs["expert_score"].detach().cpu()
        queried_mask = batch["queried_mask"].detach().cpu()
        selected_scores = expert_scores.masked_fill(~queried_mask, -1e9)
        selected[offset : offset + len(rows)] = torch.argmax(selected_scores, dim=-1)

    source_errors = torch.empty(num_windows, num_experts, dtype=torch.float32)
    for row in range(num_windows):
        source_errors[row] = cache["true_expert_error_vector"][lookup[row][0]]
    selected_mae = source_errors.gather(1, selected.view(-1, 1)).squeeze(1)
    oracle_best = torch.argmin(source_errors, dim=1)
    oracle_mae = source_errors.min(dim=1).values
    avg_queries = sum(len(history) for history in query_history) / max(num_windows, 1)
    selection_counts = torch.bincount(selected, minlength=num_experts)

    better_top_two_values = []
    for sample_index, queried in enumerate(query_history):
        if len(queried) >= 2:
            first_two = torch.tensor(queried[:2], dtype=torch.long)
            true_best_position = torch.argmin(source_errors[sample_index, first_two])
            model_scores = torch.full((num_experts,), -1e9)
            state_index = final_state_indices[sample_index]
            batch = _state_batch(cache, [state_index], device)
            outputs = router(
                batch["history"],
                batch["queried_mask"],
                batch["queried_expert_ids"],
                batch["queried_expert_forecasts"],
            )
            model_scores = outputs["expert_score"][0].detach().cpu()
            model_best_position = torch.argmax(model_scores[first_two])
            better_top_two_values.append(float(model_best_position == true_best_position))
    better_top_two_accuracy = (
        sum(better_top_two_values) / len(better_top_two_values)
        if better_top_two_values
        else math.nan
    )

    if debug:
        print("SubsetUtility validation example query history:", query_history[:3])

    return {
        "validation_mae": float(selected_mae.mean()),
        "validation_oracle_mae": float(oracle_mae.mean()),
        "validation_regret_to_oracle": float(selected_mae.mean() - oracle_mae.mean()),
        "action_accuracy": float((selected == oracle_best).to(torch.float32).mean()),
        "average_experts_selected": float(avg_queries),
        "stop_frequency": float(sum(1 for history in query_history if len(history) < max_subset_size) / max(num_windows, 1)),
        "stop_step_distribution": {
            str(index): int(value)
            for index, value in enumerate(stop_counts.tolist())
            if index > 0 and value
        },
        "selection_counts": {
            str(index): int(value)
            for index, value in enumerate(selection_counts.tolist())
        },
        "average_predicted_utility_by_step": [
            float(sum(values) / len(values)) if values else math.nan
            for values in predicted_utility_by_step
        ],
        "better_of_top_two_accuracy": float(better_top_two_accuracy),
    }


@torch.no_grad()
def evaluate_subset_state_losses(
    router: SubsetUtilityCOSTARTSRouter,
    cache: Mapping[str, Any],
    *,
    batch_size: int,
    device: Union[str, torch.device],
    loss_weights: Mapping[str, float],
) -> dict[str, float]:
    device = torch.device(device)
    router.eval()
    dataset = CostartsSubsetStateDataset(cache)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    totals = {
        "total_loss": 0.0,
        "action_loss": 0.0,
        "utility_loss": 0.0,
        "pairwise_loss": 0.0,
        "mix_loss": 0.0,
        "action_accuracy": 0.0,
        "stop_frequency": 0.0,
        "average_predicted_utility": 0.0,
    }
    count = 0
    for batch in loader:
        batch = _move_batch(batch, device)
        outputs = router(
            batch["history"],
            batch["queried_mask"],
            batch["queried_expert_ids"],
            batch["queried_expert_forecasts"],
        )
        _, parts = subset_utility_losses(outputs, batch, loss_weights)
        batch_size_actual = batch["history"].shape[0]
        for key in totals:
            totals[key] += parts[key] * batch_size_actual
        count += batch_size_actual
    return {f"validation_state_{key}": value / max(count, 1) for key, value in totals.items()}


def _write_curves(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_checkpoint(
    path: Path,
    router: SubsetUtilityCOSTARTSRouter,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    metrics: Mapping[str, Any],
    training_config: SubsetUtilityTrainingConfig,
    expert_names: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "router_state_dict": router.state_dict(),
            "router_config": router.config_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "metrics": dict(metrics),
            "training_config": asdict(training_config),
            "expert_names": list(expert_names),
            "router_type": "subset_utility_costarts",
            "experts_loaded": False,
            "experts_updated": False,
            "model_selection": "validation_mae under deployable sequential inference on router_val",
            "test_set_used": False,
        },
        path,
    )


def train_subset_utility_costarts_router(training_config: SubsetUtilityTrainingConfig) -> dict[str, Any]:
    set_reproducible_seed(training_config.seed)
    device = torch.device(training_config.device)
    experiment_config = validate_router_experiment_config(
        load_router_experiment_config(),
        require_checkpoints=False,
        require_data=False,
        require_cache_parent=True,
    )
    print_router_experiment_config(experiment_config)

    train_cache = _load_torch(training_config.train_cache_path)
    val_cache = _load_torch(training_config.val_cache_path)
    validate_costarts_subset_states(train_cache)
    validate_costarts_subset_states(val_cache)
    if train_cache["split_role"] != "router_train":
        raise ValueError("SubsetUtility training requires router_train cache")
    if val_cache["split_role"] != "router_val":
        raise ValueError("SubsetUtility validation requires router_val cache")
    if tuple(train_cache["expert_names"]) != tuple(val_cache["expert_names"]):
        raise ValueError("Train/val subset caches have different expert ordering")
    if training_config.subset_state_sampling_mode != train_cache["subset_sampling_mode"]:
        raise ValueError(
            "training_config subset_state_sampling_mode does not match train cache: "
            f"{training_config.subset_state_sampling_mode} != {train_cache['subset_sampling_mode']}"
        )
    if training_config.max_subset_size is not None and int(training_config.max_subset_size) != int(train_cache["max_subset_size"]):
        raise ValueError("training_config max_subset_size does not match train cache")

    expert_names = tuple(train_cache["expert_names"])
    router = SubsetUtilityCOSTARTSRouter(
        num_experts=int(train_cache["num_experts"]),
        max_subset_size=int(train_cache["max_subset_size"]),
        input_len=96,
        forecast_horizon=int(train_cache["forecast_horizon"]),
        num_features=int(train_cache["num_features"]),
        embedding_dim=experiment_config.embedding_dim,
        hidden_dim=experiment_config.hidden_dim,
        use_expert_embeddings=training_config.use_expert_embeddings,
        history_encoder_type=training_config.history_encoder_type,
        action_head_type=training_config.action_head_type,
    ).to(device)

    optimizer = torch.optim.AdamW(
        router.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(1, training_config.patience // 3),
    )
    train_dataset = CostartsSubsetStateDataset(train_cache)
    generator = torch.Generator()
    generator.manual_seed(training_config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        generator=generator,
    )
    loss_weights = {
        "action": training_config.action_loss_weight,
        "utility": training_config.utility_loss_weight,
        "pairwise": training_config.pairwise_loss_weight,
        "mix": training_config.mix_loss_weight,
    }

    output_dir = Path(training_config.output_dir)
    results_dir = Path(training_config.results_dir)
    best_path = output_dir / "best_subset_utility_costarts_router.pt"
    last_path = output_dir / "last_subset_utility_costarts_router.pt"
    curves_path = results_dir / "training_curves.csv"
    summary_path = results_dir / "training_summary.json"

    best_val_mae = math.inf
    best_epoch = 0
    bad_epochs = 0
    curves = []

    for epoch in range(1, training_config.max_epochs + 1):
        router.train()
        totals = {
            "total_loss": 0.0,
            "action_loss": 0.0,
            "utility_loss": 0.0,
            "pairwise_loss": 0.0,
            "mix_loss": 0.0,
            "action_accuracy": 0.0,
            "stop_frequency": 0.0,
            "average_predicted_utility": 0.0,
        }
        seen = 0
        for batch_index, batch in enumerate(train_loader):
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = router(
                batch["history"],
                batch["queried_mask"],
                batch["queried_expert_ids"],
                batch["queried_expert_forecasts"],
            )
            total_loss, parts = subset_utility_losses(outputs, batch, loss_weights)
            total_loss.backward()
            if training_config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(router.parameters(), training_config.grad_clip_norm)
            optimizer.step()
            batch_size_actual = batch["history"].shape[0]
            for key in totals:
                totals[key] += parts[key] * batch_size_actual
            seen += batch_size_actual
            if training_config.debug and epoch == 1 and batch_index == 0:
                print("SubsetUtility first training batch")
                print("  history:", tuple(batch["history"].shape))
                print("  queried_mask:", tuple(batch["queried_mask"].shape))
                print("  queried_expert_forecasts:", tuple(batch["queried_expert_forecasts"].shape))
                print("  action_logits:", tuple(outputs["action_logits"].shape))

        train_metrics = {key: value / max(seen, 1) for key, value in totals.items()}
        val_deployable = evaluate_deployable_inference(
            router,
            val_cache,
            batch_size=training_config.batch_size,
            device=device,
            debug=training_config.debug and epoch == 1,
        )
        val_state = evaluate_subset_state_losses(
            router,
            val_cache,
            batch_size=training_config.batch_size,
            device=device,
            loss_weights=loss_weights,
        )
        scheduler.step(val_deployable["validation_mae"])

        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **train_metrics,
            **val_state,
            **{
                f"validation_deployable_{key}": (
                    json.dumps(value) if isinstance(value, (dict, list)) else value
                )
                for key, value in val_deployable.items()
            },
        }
        curves.append(row)
        improved = val_deployable["validation_mae"] < best_val_mae
        if improved:
            best_val_mae = val_deployable["validation_mae"]
            best_epoch = epoch
            bad_epochs = 0
            _save_checkpoint(
                best_path,
                router,
                optimizer,
                scheduler,
                epoch,
                val_deployable,
                training_config,
                expert_names,
            )
        else:
            bad_epochs += 1

        _save_checkpoint(
            last_path,
            router,
            optimizer,
            scheduler,
            epoch,
            val_deployable,
            training_config,
            expert_names,
        )
        print(
            f"SubsetUtility epoch {epoch:03d} | "
            f"loss={train_metrics['total_loss']:.6f} "
            f"action={train_metrics['action_loss']:.6f} "
            f"utility={train_metrics['utility_loss']:.6f} "
            f"pairwise={train_metrics['pairwise_loss']:.6f} "
            f"mix={train_metrics['mix_loss']:.6f} "
            f"acc={train_metrics['action_accuracy']:.3f} "
            f"stop={train_metrics['stop_frequency']:.3f} | "
            f"val_mae={val_deployable['validation_mae']:.6f} "
            f"regret={val_deployable['validation_regret_to_oracle']:.6f} "
            f"avg_used={val_deployable['average_experts_selected']:.3f} "
            f"top2={val_deployable['better_of_top_two_accuracy']:.3f} "
            f"saved={improved}"
        )
        if bad_epochs >= training_config.patience:
            print(f"SubsetUtility early stopping after epoch {epoch}")
            break

    _write_curves(curves_path, curves)
    summary = {
        "best_epoch": best_epoch,
        "best_validation_mae": best_val_mae,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "training_curves_csv": str(curves_path),
        "training_config": asdict(training_config),
        "router_config": router.config_dict(),
        "expert_names": list(expert_names),
        "train_cache_path": training_config.train_cache_path,
        "val_cache_path": training_config.val_cache_path,
        "created_checkpoints_separate_from_original_costarts": True,
        "original_costarts_checkpoint_dir": "checkpoints/costarts",
        "model_selection": "validation_mae under deployable sequential inference on router_val",
        "test_set_used": False,
        "experts_loaded": False,
        "experts_updated": False,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    print(f"Saved: {curves_path}")
    print(f"Saved: {summary_path}")
    return summary


def parse_args() -> argparse.Namespace:
    repo_config = load_router_experiment_config()
    cache_paths = dict(repo_config.cache_paths)
    parser = argparse.ArgumentParser(description="Train SubsetUtilityCOSTARTSRouter.")
    parser.add_argument("--train-cache", default=cache_paths.get("costarts_subset_states_train", DEFAULT_TRAIN_CACHE))
    parser.add_argument("--val-cache", default=cache_paths.get("costarts_subset_states_val", DEFAULT_VAL_CACHE))
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--batch-size", type=int, default=repo_config.costarts_subset_utility_batch_size)
    parser.add_argument("--max-epochs", type=int, default=repo_config.costarts_subset_utility_max_epochs)
    parser.add_argument("--patience", type=int, default=repo_config.costarts_subset_utility_patience)
    parser.add_argument("--learning-rate", type=float, default=repo_config.costarts_subset_utility_learning_rate)
    parser.add_argument("--weight-decay", type=float, default=repo_config.costarts_subset_utility_weight_decay)
    parser.add_argument("--grad-clip-norm", type=float, default=repo_config.costarts_subset_utility_grad_clip_norm)
    parser.add_argument("--seed", type=int, default=repo_config.random_seed)
    parser.add_argument("--action-loss-weight", type=float, default=repo_config.costarts_subset_utility_action_loss_weight)
    parser.add_argument("--utility-loss-weight", type=float, default=repo_config.costarts_subset_utility_loss_weight)
    parser.add_argument("--pairwise-loss-weight", type=float, default=repo_config.costarts_subset_utility_pairwise_loss_weight)
    parser.add_argument("--mix-loss-weight", type=float, default=repo_config.costarts_subset_utility_mix_loss_weight)
    parser.add_argument("--subset-state-sampling-mode", default=repo_config.costarts_subset_sampling_mode)
    parser.add_argument("--max-subset-size", type=int, default=repo_config.costarts_subset_max_size)
    parser.add_argument("--cost-coefficient", type=float, default=repo_config.costarts_subset_utility_cost_coefficient)
    parser.add_argument("--no-expert-embeddings", action="store_true")
    parser.add_argument("--history-encoder-type", choices=("current", "simple"), default="current")
    parser.add_argument("--action-head-type", choices=("unified", "separate_stop_query"), default="unified")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_subset_utility_costarts_router(
        SubsetUtilityTrainingConfig(
            train_cache_path=args.train_cache,
            val_cache_path=args.val_cache,
            output_dir=args.output_dir,
            results_dir=args.results_dir,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            grad_clip_norm=args.grad_clip_norm,
            seed=args.seed,
            action_loss_weight=args.action_loss_weight,
            utility_loss_weight=args.utility_loss_weight,
            pairwise_loss_weight=args.pairwise_loss_weight,
            mix_loss_weight=args.mix_loss_weight,
            subset_state_sampling_mode=args.subset_state_sampling_mode,
            max_subset_size=args.max_subset_size,
            cost_coefficient=args.cost_coefficient,
            use_expert_embeddings=not args.no_expert_embeddings,
            history_encoder_type=args.history_encoder_type,
            action_head_type=args.action_head_type,
            device=args.device,
            debug=args.debug,
        )
    )


if __name__ == "__main__":
    main()
