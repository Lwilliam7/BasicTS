"""Isolated COSTAR-TS experiment runner from saved router/cache artifacts.

This script is intentionally separate from ``train_costarts_router.py`` so
small router/aggregation experiments can be run without importing the missing
chronological expert-training helpers or changing the best COSTAR-TS path.
It never loads, trains, or updates frozen forecasting experts.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.old.build_costarts_subset_states import validate_costarts_subset_states
from scripts.old.train_costarts_subset_utility_router import (
    SubsetUtilityCOSTARTSRouter,
    _build_state_lookup,
    _state_batch,
)


DEFAULT_CHECKPOINT = "checkpoints/costarts/best_costarts_router.pt"
DEFAULT_TRAIN_CACHE = "cache/costarts_router_train_cache.pt"
DEFAULT_VAL_CACHE = "cache/costarts_router_val_cache.pt"
DEFAULT_SUBSET_VAL_CACHE = "cache/costarts_subset_states_val.pt"
DEFAULT_SUBSET_CHECKPOINT_TEMPLATE = (
    "checkpoints/costarts_subset_utility/attention_compare_fair_{encoder}_seed_{seed}/"
    "best_subset_utility_costarts_router.pt"
)
DEFAULT_OUTPUT_DIR = "results/router_summary/costarts_experiments"


def _masked_logits(logits: torch.Tensor, valid_action_mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~valid_action_mask.to(torch.bool), -1e9)


class COSTARSExperimentRouter(nn.Module):
    """Self-contained copy of the saved COSTAR-TS router architecture."""

    def __init__(
        self,
        num_experts: int,
        input_len: int = 96,
        forecast_horizon: int = 12,
        num_features: int = 7,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        queried_experts_cap_k: int | None = None,
        routing_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if input_len <= 0 or forecast_horizon <= 0 or num_features <= 0:
            raise ValueError("input_len, forecast_horizon, and num_features must be positive")
        if embedding_dim <= 0 or hidden_dim <= 0:
            raise ValueError("embedding_dim and hidden_dim must be positive")
        if routing_temperature <= 0:
            raise ValueError("routing_temperature must be positive")

        self.num_experts = int(num_experts)
        self.input_len = int(input_len)
        self.forecast_horizon = int(forecast_horizon)
        self.num_features = int(num_features)
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.queried_experts_cap_k = (
            self.num_experts
            if queried_experts_cap_k is None
            else min(int(queried_experts_cap_k), self.num_experts)
        )
        if self.queried_experts_cap_k <= 0:
            raise ValueError("queried_experts_cap_k must be positive")
        self.routing_temperature = float(routing_temperature)

        self.history_encoder = nn.Sequential(
            nn.Conv1d(num_features, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.GroupNorm(1, hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=4, dilation=2),
            nn.GELU(),
            nn.GroupNorm(1, hidden_dim),
            nn.AdaptiveAvgPool1d(1),
        )
        self.query_projection = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
        )
        self.expert_embeddings = nn.Parameter(torch.empty(num_experts, embedding_dim))
        nn.init.normal_(self.expert_embeddings, mean=0.0, std=0.02)
        self.map_head = nn.Linear(embedding_dim, num_experts)
        self.ranking_head = nn.Linear(embedding_dim, num_experts)
        self.query_head = nn.Linear(embedding_dim, num_experts)
        self.mix_head = nn.Linear(embedding_dim, num_experts)
        self.stop_head = nn.Linear(embedding_dim, self.queried_experts_cap_k)

    def encode(self, history: torch.Tensor) -> torch.Tensor:
        assert history.ndim == 3
        assert history.shape[1:] == (self.input_len, self.num_features)
        encoded = self.history_encoder(history.transpose(1, 2)).squeeze(-1)
        query_embedding = self.query_projection(encoded)
        return F.normalize(query_embedding, p=2, dim=-1)

    def forward(
        self,
        history: torch.Tensor,
        teacher_forced_order: Optional[torch.Tensor] = None,
        sampled_rollout: bool = False,
        debug: bool = False,
    ) -> dict[str, torch.Tensor]:
        batch_size = history.shape[0]
        query_embedding = self.encode(history)
        expert_vectors = F.normalize(self.expert_embeddings, p=2, dim=-1)
        cosine_scores = torch.matmul(query_embedding, expert_vectors.T)
        similarity_logits = cosine_scores / self.routing_temperature
        map_prediction = F.softplus(self.map_head(query_embedding))
        ranking_logits = self.ranking_head(query_embedding) + similarity_logits
        query_logits = self.query_head(query_embedding) + similarity_logits
        mix_weights = torch.softmax(self.mix_head(query_embedding), dim=-1)
        stop_logits = self.stop_head(query_embedding)

        if teacher_forced_order is not None:
            query_order = teacher_forced_order[:, : self.queried_experts_cap_k]
        else:
            if sampled_rollout and self.training:
                probabilities = torch.softmax(query_logits, dim=-1)
                query_order = torch.multinomial(
                    probabilities,
                    num_samples=self.queried_experts_cap_k,
                    replacement=False,
                )
            else:
                query_order = torch.topk(
                    query_logits,
                    k=self.queried_experts_cap_k,
                    dim=-1,
                ).indices

        stop_step = torch.argmax(stop_logits, dim=-1) + 1
        assert query_embedding.shape == (batch_size, self.embedding_dim)
        assert map_prediction.shape == (batch_size, self.num_experts)
        assert ranking_logits.shape == (batch_size, self.num_experts)
        assert query_logits.shape == (batch_size, self.num_experts)
        assert stop_logits.shape == (batch_size, self.queried_experts_cap_k)
        assert query_order.shape == (batch_size, self.queried_experts_cap_k)

        if debug:
            print("COSTARTS history:", tuple(history.shape))
            print("COSTARTS query_embedding:", tuple(query_embedding.shape))
            print("COSTARTS map_prediction:", tuple(map_prediction.shape))
            print("COSTARTS query_order:", tuple(query_order.shape))
            print("COSTARTS stop_logits:", tuple(stop_logits.shape))
            print("COSTARTS first selected experts:", query_order[:3].detach().cpu().tolist())

        return {
            "query_embedding": query_embedding,
            "similarity_logits": similarity_logits,
            "map_prediction": map_prediction,
            "ranking_logits": ranking_logits,
            "query_logits": query_logits,
            "mix_weights": mix_weights,
            "stop_logits": stop_logits,
            "query_order": query_order,
            "stop_step": stop_step,
        }


class COSTARSAdaptiveExperimentRouter(SubsetUtilityCOSTARTSRouter):
    """COSTAR-TS experiment router that observes queried forecasts before each next action."""


def _load_torch(path: str | Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _assert_cache_pair(train_cache: Mapping[str, Any], val_cache: Mapping[str, Any]) -> Sequence[str]:
    if train_cache.get("split_role") != "router_train":
        raise ValueError("train cache must use split_role='router_train'")
    if val_cache.get("split_role") != "router_val":
        raise ValueError("validation cache must use split_role='router_val'")
    expert_names = tuple(train_cache["expert_names"])
    if expert_names != tuple(val_cache["expert_names"]):
        raise ValueError("train/validation expert ordering mismatch")
    return expert_names


@torch.no_grad()
def _router_outputs(
    router: COSTARSExperimentRouter,
    histories: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    chunks: dict[str, list[torch.Tensor]] = {
        "map_prediction": [],
        "query_logits": [],
        "query_order": [],
        "stop_step": [],
    }
    router.eval()
    for start in range(0, histories.shape[0], batch_size):
        outputs = router(histories[start : start + batch_size].to(device).to(torch.float32))
        for key in chunks:
            chunks[key].append(outputs[key].detach().cpu())
    return {key: torch.cat(value, dim=0) for key, value in chunks.items()}


def _mae_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[float, float]:
    mask_float = mask.to(prediction.dtype)
    denominator = mask_float.sum().clamp_min(1.0)
    mae = (torch.abs(prediction - target) * mask_float).sum() / denominator
    mse = ((prediction - target).pow(2) * mask_float).sum() / denominator
    return float(mae), float(mse)


def _train_error_vector(train_cache: Mapping[str, Any], expert_names: Sequence[str]) -> torch.Tensor:
    if train_cache.get("split_role") != "router_train":
        raise ValueError("train-error softmax requires a router_train cache")
    if tuple(train_cache["expert_names"]) != tuple(expert_names):
        raise ValueError("train cache expert ordering does not match experiment cache")
    return train_cache["error_matrix"].to(torch.float32).mean(dim=0)


def _utility_threshold_logits(
    utility_prediction: torch.Tensor,
    valid_action_mask: torch.Tensor,
    *,
    utility_threshold: float,
    num_experts: int,
) -> torch.Tensor:
    stop_logit = torch.full(
        (utility_prediction.shape[0], 1),
        float(utility_threshold),
        dtype=utility_prediction.dtype,
        device=utility_prediction.device,
    )
    logits = torch.cat((utility_prediction[:, :num_experts], stop_logit), dim=1)
    return logits.masked_fill(~valid_action_mask.to(torch.bool), -1e9)


def _queried_indices_for_row(query_order: torch.Tensor, stop_step: torch.Tensor, row: int) -> torch.Tensor:
    count = int(stop_step[row].clamp(1, query_order.shape[1]).item())
    return query_order[row, :count].to(torch.long)


def _selected_expert_metrics(
    outputs: Mapping[str, torch.Tensor],
    val_cache: Mapping[str, Any],
    expert_names: Sequence[str],
) -> dict[str, Any]:
    selected = []
    query_order = outputs["query_order"]
    stop_step = outputs["stop_step"]
    predicted_errors = outputs["map_prediction"]
    for row in range(query_order.shape[0]):
        queried = _queried_indices_for_row(query_order, stop_step, row)
        candidate_errors = predicted_errors[row].gather(0, queried)
        selected.append(queried[torch.argmin(candidate_errors)])
    selected_tensor = torch.stack(selected)
    error_matrix = val_cache["error_matrix"].to(torch.float32)
    mse_matrix = val_cache["mse_matrix"].to(torch.float32)
    selected_mae = error_matrix.gather(1, selected_tensor[:, None]).squeeze(1)
    selected_mse = mse_matrix.gather(1, selected_tensor[:, None]).squeeze(1)
    counts = torch.bincount(selected_tensor, minlength=len(expert_names))
    return {
        "method": "selected_expert",
        "mae": float(selected_mae.mean()),
        "mse": float(selected_mse.mean()),
        "average_experts_queried": float(stop_step.to(torch.float32).mean()),
        "selection_counts": {name: int(counts[index]) for index, name in enumerate(expert_names)},
    }


def _aggregate_prediction_metrics(
    outputs: Mapping[str, torch.Tensor],
    train_cache: Mapping[str, Any],
    val_cache: Mapping[str, Any],
    *,
    method: str,
    temperature: float,
) -> dict[str, Any]:
    query_order = outputs["query_order"]
    stop_step = outputs["stop_step"]
    prediction_stack = val_cache["prediction_stack"].to(torch.float32)
    targets = val_cache["targets"].to(torch.float32)
    masks = val_cache["target_masks"].to(torch.bool)
    train_error = train_cache["error_matrix"].to(torch.float32).mean(dim=0)
    predictions = []
    utilization = torch.zeros(len(train_cache["expert_names"]), dtype=torch.long)
    for row in range(query_order.shape[0]):
        queried = _queried_indices_for_row(query_order, stop_step, row)
        utilization[queried] += 1
        expert_predictions = prediction_stack[row, :, :, queried]
        if method == "equal_average":
            weights = torch.full((queried.numel(),), 1.0 / queried.numel())
        elif method == "train_error_softmax":
            weights = torch.softmax(-train_error[queried] / float(temperature), dim=0)
        else:
            raise ValueError(f"unknown aggregation method: {method}")
        predictions.append((expert_predictions * weights.view(1, 1, -1)).sum(dim=-1))
    prediction = torch.stack(predictions, dim=0)
    mae, mse = _mae_mse(prediction, targets, masks)
    return {
        "method": method,
        "temperature": None if method == "equal_average" else float(temperature),
        "mae": mae,
        "mse": mse,
        "average_experts_queried": float(stop_step.to(torch.float32).mean()),
        "utilization_counts": {
            name: int(utilization[index])
            for index, name in enumerate(train_cache["expert_names"])
        },
    }


def run_costars_experiments(
    *,
    checkpoint_path: str | Path,
    train_cache_path: str | Path,
    val_cache_path: str | Path,
    output_dir: str | Path,
    batch_size: int,
    device: str,
    temperatures: Sequence[float],
) -> dict[str, Any]:
    checkpoint = _load_torch(checkpoint_path)
    train_cache = _load_torch(train_cache_path)
    val_cache = _load_torch(val_cache_path)
    expert_names = _assert_cache_pair(train_cache, val_cache)
    device_obj = torch.device(device)
    router = COSTARSExperimentRouter(**checkpoint["router_config"]).to(device_obj)
    router.load_state_dict(checkpoint["router_state_dict"])

    outputs = _router_outputs(
        router,
        val_cache["histories"],
        batch_size=batch_size,
        device=device_obj,
    )
    rows = [
        _selected_expert_metrics(outputs, val_cache, expert_names),
        _aggregate_prediction_metrics(
            outputs,
            train_cache,
            val_cache,
            method="equal_average",
            temperature=1.0,
        ),
    ]
    for temperature in temperatures:
        rows.append(
            _aggregate_prediction_metrics(
                outputs,
                train_cache,
                val_cache,
                method="train_error_softmax",
                temperature=float(temperature),
            )
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / "costars_experiment_results.csv"
    json_path = output_path / "costars_experiment_results.json"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {
                key: json.dumps(value, sort_keys=True) if isinstance(value, dict) else value
                for key, value in row.items()
            }
            for row in rows
        )
    best = min(rows, key=lambda row: float(row["mae"]))
    payload = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "train_cache": str(train_cache_path),
        "val_cache": str(val_cache_path),
        "expert_names": list(expert_names),
        "rows": rows,
        "best_row": best,
        "test_set_used": False,
        "experts_loaded": False,
        "experts_updated": False,
        "results_csv": str(csv_path),
    }
    json_path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
    print(f"Best: {best['method']} mae={float(best['mae']):.6f}")
    return payload


@torch.no_grad()
def evaluate_adaptive_subset_stop(
    *,
    router: COSTARSAdaptiveExperimentRouter,
    checkpoint: Mapping[str, Any],
    subset_cache: Mapping[str, Any],
    train_error: torch.Tensor,
    seed: int,
    encoder: str,
    utility_threshold: float,
    train_error_temperature: float,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    validate_costarts_subset_states(subset_cache)
    if subset_cache.get("split_role") != "router_val":
        raise ValueError("adaptive subset stop must evaluate on router_val subset states")
    if subset_cache.get("subset_sampling_mode") != "exhaustive":
        raise ValueError("adaptive subset stop requires exhaustive subset-state cache")
    if train_error_temperature <= 0:
        raise ValueError("train_error_temperature must be positive")

    router.eval()
    num_experts = int(subset_cache["num_experts"])
    num_windows = int(subset_cache["num_source_windows"])
    state_lookup = _build_state_lookup(subset_cache)
    current_masks = [0 for _ in range(num_windows)]
    final_state_indices = [-1 for _ in range(num_windows)]
    stopped = [False for _ in range(num_windows)]
    query_sequences: list[list[int]] = [[] for _ in range(num_windows)]
    stop_index = num_experts

    for _step in range(num_experts):
        active_rows = [row for row in range(num_windows) if not stopped[row]]
        if not active_rows:
            break
        active_state_indices = [state_lookup[row][current_masks[row]] for row in active_rows]
        action_chunks = []
        for start in range(0, len(active_state_indices), batch_size):
            batch_state_indices = active_state_indices[start : start + batch_size]
            batch = _state_batch(subset_cache, batch_state_indices, device)
            outputs = router(
                batch["history"],
                batch["queried_mask"],
                batch["queried_expert_ids"],
                batch["queried_expert_forecasts"],
            )
            logits = _utility_threshold_logits(
                outputs["utility_prediction"],
                batch["valid_action_mask"],
                utility_threshold=utility_threshold,
                num_experts=num_experts,
            )
            action_chunks.append(torch.argmax(logits, dim=-1).detach().cpu())
        actions = torch.cat(action_chunks, dim=0)

        for active_index, action_tensor in enumerate(actions):
            row = active_rows[active_index]
            action = int(action_tensor.item())
            state_index = active_state_indices[active_index]
            if action == stop_index:
                stopped[row] = True
                final_state_indices[row] = state_index
                continue
            if current_masks[row] & (1 << action):
                stopped[row] = True
                final_state_indices[row] = state_index
                continue
            current_masks[row] |= 1 << action
            query_sequences[row].append(action)
            if len(query_sequences[row]) >= num_experts:
                stopped[row] = True
                final_state_indices[row] = state_lookup[row][current_masks[row]]

    for row in range(num_windows):
        if final_state_indices[row] < 0:
            final_state_indices[row] = state_lookup[row][current_masks[row]]

    predictions = []
    targets = []
    masks = []
    selected_experts = []
    true_error_rows = []
    utilization = torch.zeros(num_experts, dtype=torch.long)
    for start in range(0, num_windows, batch_size):
        batch_indices = final_state_indices[start : start + batch_size]
        batch = _state_batch(subset_cache, batch_indices, torch.device("cpu"))
        queried_ids = batch["queried_expert_ids"].to(torch.long)
        queried_forecasts = batch["queried_expert_forecasts"].to(torch.float32)
        valid_slots = queried_ids >= 0
        safe_ids = queried_ids.clamp_min(0)
        gathered_errors = train_error.gather(0, safe_ids.reshape(-1)).reshape_as(safe_ids)
        gathered_errors = gathered_errors.masked_fill(~valid_slots, 1e9)
        slot_weights = torch.softmax(-gathered_errors / float(train_error_temperature), dim=1)
        slot_weights = slot_weights.masked_fill(~valid_slots, 0.0)
        slot_weights = slot_weights / slot_weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        predictions.append((queried_forecasts * slot_weights[:, :, None, None]).sum(dim=1))
        targets.append(batch["true_targets"].to(torch.float32))
        masks.append(batch["target_mask"].to(torch.bool))
        selected_experts.append(
            queried_ids[
                torch.arange(queried_ids.shape[0]),
                torch.argmax(slot_weights, dim=1),
            ].to(torch.long)
        )
        true_error_rows.append(batch["true_expert_error_vector"].to(torch.float32))
        for sequence in query_sequences[start : start + batch_size]:
            for expert_index in sequence:
                utilization[int(expert_index)] += 1

    prediction = torch.cat(predictions, dim=0)
    target = torch.cat(targets, dim=0)
    target_mask = torch.cat(masks, dim=0)
    mae, mse = _mae_mse(prediction, target, target_mask)
    selected = torch.cat(selected_experts, dim=0)
    true_errors = torch.cat(true_error_rows, dim=0)
    oracle_best = torch.argmin(true_errors, dim=1)
    stop_counts = torch.bincount(
        torch.tensor([len(sequence) for sequence in query_sequences], dtype=torch.long),
        minlength=num_experts + 1,
    )
    expert_names = tuple(subset_cache["expert_names"])
    return {
        "experiment": "adaptive_subset_stop",
        "router_class": "COSTARSAdaptiveExperimentRouter",
        "decision_rule": "query_argmax_utility_if_above_threshold_else_stop",
        "finalizer": "train_error_softmax",
        "encoder": encoder,
        "seed": int(seed),
        "checkpoint": str(checkpoint.get("_checkpoint_path", "")),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "utility_threshold": float(utility_threshold),
        "train_error_temperature": float(train_error_temperature),
        "mae": mae,
        "mse": mse,
        "average_experts_queried": float(sum(len(sequence) for sequence in query_sequences) / max(num_windows, 1)),
        "oracle_match_rate": float((selected == oracle_best).to(torch.float32).mean()),
        "stop_step_distribution": {
            str(step): int(count)
            for step, count in enumerate(stop_counts.tolist())
            if count
        },
        "query_utilization_counts": {
            expert_names[index]: int(utilization[index])
            for index in range(num_experts)
        },
        "all_queries_unique": all(len(sequence) == len(set(sequence)) for sequence in query_sequences),
        "test_set_used": False,
    }


def _load_adaptive_router(checkpoint_path: str | Path, device: torch.device) -> tuple[COSTARSAdaptiveExperimentRouter, dict[str, Any]]:
    checkpoint = _load_torch(checkpoint_path)
    checkpoint["_checkpoint_path"] = str(checkpoint_path)
    router = COSTARSAdaptiveExperimentRouter(**checkpoint["router_config"]).to(device)
    router.load_state_dict(checkpoint["router_state_dict"], strict=False)
    return router, checkpoint


def run_adaptive_subset_stop_sweep(
    *,
    checkpoint_template: str,
    train_cache_path: str | Path,
    subset_val_cache_path: str | Path,
    output_dir: str | Path,
    batch_size: int,
    device: str,
    encoders: Sequence[str],
    seeds: Sequence[int],
    utility_thresholds: Sequence[float],
    train_error_temperatures: Sequence[float],
) -> dict[str, Any]:
    train_cache = _load_torch(train_cache_path)
    subset_cache = _load_torch(subset_val_cache_path)
    validate_costarts_subset_states(subset_cache)
    train_error = _train_error_vector(train_cache, tuple(subset_cache["expert_names"]))
    device_obj = torch.device(device)
    rows = []
    for encoder in encoders:
        for seed in seeds:
            checkpoint_path = checkpoint_template.format(encoder=encoder, seed=int(seed))
            router, checkpoint = _load_adaptive_router(checkpoint_path, device_obj)
            for utility_threshold in utility_thresholds:
                for train_error_temperature in train_error_temperatures:
                    rows.append(
                        evaluate_adaptive_subset_stop(
                            router=router,
                            checkpoint=checkpoint,
                            subset_cache=subset_cache,
                            train_error=train_error,
                            seed=int(seed),
                            encoder=str(encoder),
                            utility_threshold=float(utility_threshold),
                            train_error_temperature=float(train_error_temperature),
                            batch_size=batch_size,
                            device=device_obj,
                        )
                    )

    groups: dict[tuple[str, float, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["encoder"]),
            float(row["utility_threshold"]),
            float(row["train_error_temperature"]),
        )
        groups.setdefault(key, []).append(row)
    summary = []
    for (encoder, utility_threshold, train_error_temperature), group_rows in groups.items():
        maes = torch.tensor([float(row["mae"]) for row in group_rows], dtype=torch.float32)
        experts = torch.tensor([float(row["average_experts_queried"]) for row in group_rows], dtype=torch.float32)
        summary.append(
            {
                "encoder": encoder,
                "utility_threshold": utility_threshold,
                "train_error_temperature": train_error_temperature,
                "mae_mean": float(maes.mean()),
                "mae_std": float(maes.std(unbiased=False)),
                "avg_experts_mean": float(experts.mean()),
                "avg_experts_std": float(experts.std(unbiased=False)),
                "all_queries_unique": all(bool(row["all_queries_unique"]) for row in group_rows),
            }
        )
    summary.sort(key=lambda row: float(row["mae_mean"]))
    best = summary[0]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / "adaptive_subset_stop_sweep.csv"
    json_path = output_path / "adaptive_subset_stop_summary.json"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True) if isinstance(value, dict) else value
                    for key, value in row.items()
                }
            )
    payload = {
        "experiment": "adaptive_subset_stop",
        "router_class": "COSTARSAdaptiveExperimentRouter",
        "checkpoint_template": checkpoint_template,
        "train_cache": str(train_cache_path),
        "subset_val_cache": str(subset_val_cache_path),
        "expert_names": list(subset_cache["expert_names"]),
        "rows": rows,
        "summary": summary,
        "best_row": best,
        "baseline_current_best_validation_mae": 0.347488,
        "test_set_used": False,
        "experts_loaded": False,
        "experts_updated": False,
        "results_csv": str(csv_path),
    }
    json_path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
    print(
        "Best adaptive subset stop: "
        f"encoder={best['encoder']} threshold={best['utility_threshold']} "
        f"temp={best['train_error_temperature']} mae={best['mae_mean']:.6f} "
        f"avg_experts={best['avg_experts_mean']:.3f}"
    )
    return payload


@torch.no_grad()
def evaluate_subset_action_head_greedy(
    *,
    router: COSTARSAdaptiveExperimentRouter,
    checkpoint: Mapping[str, Any],
    subset_cache: Mapping[str, Any],
    encoder: str,
    seed: int,
    batch_size: int,
    device: torch.device,
    max_queries: int,
) -> dict[str, Any]:
    validate_costarts_subset_states(subset_cache)
    if subset_cache.get("split_role") != "router_val":
        raise ValueError("subset action-head greedy must evaluate on router_val only")
    num_experts = int(subset_cache["num_experts"])
    num_windows = int(subset_cache["num_source_windows"])
    state_lookup = _build_state_lookup(subset_cache)
    current_masks = [0 for _ in range(num_windows)]
    final_state_indices = [-1 for _ in range(num_windows)]
    stopped = [False for _ in range(num_windows)]
    query_sequences: list[list[int]] = [[] for _ in range(num_windows)]
    stop_index = num_experts
    router.eval()

    for _step in range(min(max_queries, num_experts)):
        active_rows = [row for row in range(num_windows) if not stopped[row]]
        if not active_rows:
            break
        state_indices = [state_lookup[row][current_masks[row]] for row in active_rows]
        action_chunks = []
        first_logits_chunks = []
        for start in range(0, len(state_indices), batch_size):
            batch_indices = state_indices[start : start + batch_size]
            batch = _state_batch(subset_cache, batch_indices, device)
            outputs = router(
                batch["history"],
                batch["queried_mask"],
                batch["queried_expert_ids"],
                batch["queried_expert_forecasts"],
            )
            logits = _masked_logits(outputs["action_logits"], batch["valid_action_mask"])
            action_chunks.append(torch.argmax(logits, dim=1).detach().cpu())
            if _step == 0:
                first_logits_chunks.append(logits[:, :num_experts].detach().cpu())
        actions = torch.cat(action_chunks, dim=0)

        for active_index, action_tensor in enumerate(actions):
            row = active_rows[active_index]
            state_index = state_indices[active_index]
            action = int(action_tensor.item())
            if action == stop_index:
                stopped[row] = True
                final_state_indices[row] = state_index
                continue
            if current_masks[row] & (1 << action):
                stopped[row] = True
                final_state_indices[row] = state_index
                continue
            current_masks[row] |= 1 << action
            query_sequences[row].append(action)
            if len(query_sequences[row]) >= num_experts:
                stopped[row] = True
                final_state_indices[row] = state_lookup[row][current_masks[row]]

    for row in range(num_windows):
        if final_state_indices[row] < 0:
            final_state_indices[row] = state_lookup[row][current_masks[row]]

    predictions = []
    targets = []
    masks = []
    true_error_rows = []
    selected_first = []
    for start in range(0, num_windows, batch_size):
        batch = _state_batch(subset_cache, final_state_indices[start : start + batch_size], torch.device("cpu"))
        queried_ids = batch["queried_expert_ids"].to(torch.long)
        valid_slots = queried_ids >= 0
        slot_weights = valid_slots.to(torch.float32)
        slot_weights = slot_weights / slot_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        predictions.append((batch["queried_expert_forecasts"].to(torch.float32) * slot_weights[:, :, None, None]).sum(dim=1))
        targets.append(batch["true_targets"].to(torch.float32))
        masks.append(batch["target_mask"].to(torch.bool))
        true_error_rows.append(batch["true_expert_error_vector"].to(torch.float32))
        for sequence in query_sequences[start : start + batch_size]:
            selected_first.append(sequence[0] if sequence else -1)

    prediction = torch.cat(predictions, dim=0)
    target = torch.cat(targets, dim=0)
    target_mask = torch.cat(masks, dim=0)
    mae, mse = _mae_mse(prediction, target, target_mask)
    true_errors = torch.cat(true_error_rows, dim=0)
    oracle_best = torch.argmin(true_errors, dim=1)
    first_tensor = torch.tensor(selected_first, dtype=torch.long)
    valid_first = first_tensor >= 0
    expert_names = tuple(subset_cache["expert_names"])
    first_counts = torch.bincount(first_tensor.clamp_min(0), minlength=num_experts)
    stop_counts = torch.bincount(
        torch.tensor([len(sequence) for sequence in query_sequences], dtype=torch.long),
        minlength=num_experts + 1,
    )
    order_counts: dict[str, int] = {}
    for sequence in query_sequences:
        key = "STOP" if not sequence else "->".join(expert_names[index] for index in sequence)
        order_counts[key] = order_counts.get(key, 0) + 1
    return {
        "experiment": "subset_action_head_greedy",
        "router_class": "COSTARSAdaptiveExperimentRouter",
        "encoder": encoder,
        "seed": int(seed),
        "checkpoint": str(checkpoint.get("_checkpoint_path", "")),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "mae": mae,
        "mse": mse,
        "average_experts_queried": float(sum(len(sequence) for sequence in query_sequences) / max(num_windows, 1)),
        "regret_to_oracle": mae - float(true_errors.min(dim=1).values.mean()),
        "first_query_oracle_match": float((first_tensor[valid_first] == oracle_best[valid_first]).to(torch.float32).mean()) if torch.any(valid_first) else 0.0,
        "first_query_counts": {
            expert_names[index]: int(first_counts[index])
            for index in range(num_experts)
        },
        "stop_step_distribution": {
            str(step): int(count)
            for step, count in enumerate(stop_counts.tolist())
            if count
        },
        "query_order_distribution": dict(sorted(order_counts.items(), key=lambda item: item[1], reverse=True)[:20]),
        "all_queries_unique": all(len(sequence) == len(set(sequence)) for sequence in query_sequences),
        "test_set_used": False,
    }


def run_subset_action_head_greedy_compare(
    *,
    checkpoint_template: str,
    subset_val_cache_path: str | Path,
    output_dir: str | Path,
    batch_size: int,
    device: str,
    encoders: Sequence[str],
    seeds: Sequence[int],
    max_queries: int,
) -> dict[str, Any]:
    subset_cache = _load_torch(subset_val_cache_path)
    validate_costarts_subset_states(subset_cache)
    device_obj = torch.device(device)
    rows = []
    for encoder in encoders:
        for seed in seeds:
            checkpoint_path = checkpoint_template.format(encoder=encoder, seed=int(seed))
            router, checkpoint = _load_adaptive_router(checkpoint_path, device_obj)
            rows.append(
                evaluate_subset_action_head_greedy(
                    router=router,
                    checkpoint=checkpoint,
                    subset_cache=subset_cache,
                    encoder=encoder,
                    seed=int(seed),
                    batch_size=batch_size,
                    device=device_obj,
                    max_queries=max_queries,
                )
            )

    summary = []
    for encoder in sorted({row["encoder"] for row in rows}):
        group = [row for row in rows if row["encoder"] == encoder]
        maes = torch.tensor([float(row["mae"]) for row in group], dtype=torch.float32)
        mses = torch.tensor([float(row["mse"]) for row in group], dtype=torch.float32)
        experts = torch.tensor([float(row["average_experts_queried"]) for row in group], dtype=torch.float32)
        summary.append(
            {
                "encoder": encoder,
                "mae_mean": float(maes.mean()),
                "mae_std": float(maes.std(unbiased=False)),
                "mse_mean": float(mses.mean()),
                "avg_experts_mean": float(experts.mean()),
                "avg_experts_std": float(experts.std(unbiased=False)),
            }
        )
    summary.sort(key=lambda row: row["mae_mean"])
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / "subset_action_head_greedy_rows.csv"
    json_path = output_path / "subset_action_head_greedy_summary.json"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True) if isinstance(value, dict) else value
                    for key, value in row.items()
                }
            )
    payload = {
        "experiment": "subset_action_head_greedy",
        "rows": rows,
        "summary": summary,
        "best_row": min(rows, key=lambda row: float(row["mae"])),
        "test_set_used": False,
        "results_csv": str(csv_path),
    }
    json_path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
    print(json.dumps(_jsonable(payload["summary"]), indent=2))
    return payload


def _parse_temperatures(value: str) -> list[float]:
    temperatures = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not temperatures:
        raise ValueError("at least one temperature is required")
    if any(item <= 0 for item in temperatures):
        raise ValueError("temperatures must be positive")
    return temperatures


def _parse_ints(value: str) -> list[int]:
    items = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("at least one integer is required")
    return items


def _parse_strings(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("at least one value is required")
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run isolated COSTAR-TS experiments.")
    parser.add_argument(
        "--experiment",
        choices=("preplanned", "adaptive_subset_stop", "subset_action_head_greedy"),
        default="preplanned",
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--subset-checkpoint-template", default=DEFAULT_SUBSET_CHECKPOINT_TEMPLATE)
    parser.add_argument("--train-cache", default=DEFAULT_TRAIN_CACHE)
    parser.add_argument("--val-cache", default=DEFAULT_VAL_CACHE)
    parser.add_argument("--subset-val-cache", default=DEFAULT_SUBSET_VAL_CACHE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--temperatures", default="0.05,0.1,0.25,0.5,1.0,2.0")
    parser.add_argument("--seeds", default="7,11,13,17,19")
    parser.add_argument("--encoders", default="mean,set_attention")
    parser.add_argument(
        "--utility-thresholds",
        default="-5.0,-3.0,-2.0,-1.5,-1.25,-1.0,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.0,0.1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.experiment == "adaptive_subset_stop":
        run_adaptive_subset_stop_sweep(
            checkpoint_template=args.subset_checkpoint_template,
            train_cache_path=args.train_cache,
            subset_val_cache_path=args.subset_val_cache,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            device=args.device,
            encoders=_parse_strings(args.encoders),
            seeds=_parse_ints(args.seeds),
            utility_thresholds=_parse_temperatures(args.utility_thresholds),
            train_error_temperatures=_parse_temperatures(args.temperatures),
        )
    elif args.experiment == "subset_action_head_greedy":
        run_subset_action_head_greedy_compare(
            checkpoint_template=args.subset_checkpoint_template,
            subset_val_cache_path=args.subset_val_cache,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            device=args.device,
            encoders=_parse_strings(args.encoders),
            seeds=_parse_ints(args.seeds),
            max_queries=5,
        )
    else:
        run_costars_experiments(
            checkpoint_path=args.checkpoint,
            train_cache_path=args.train_cache,
            val_cache_path=args.val_cache,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            device=args.device,
            temperatures=_parse_temperatures(args.temperatures),
        )


if __name__ == "__main__":
    main()
