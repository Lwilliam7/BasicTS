"""Final validation comparison for frozen-expert routing methods.

All computed rows use the same chronological COSTARTS validation windows and
the same cached frozen-expert prediction stack. Test evaluation is intentionally
not enabled unless a separate untouched test cache is supplied later.
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
from torch import nn
import torch.nn.functional as F

try:
    from scripts.costars.build_costarts_subset_states import validate_costarts_subset_states
    from scripts.train_costarts_router import COSTARTSRouter, _select_expert_from_outputs
    from scripts.train_costarts_subset_utility_router import (
        SubsetUtilityCOSTARTSRouter,
        _build_state_lookup,
        _masked_action_logits,
        _state_batch,
        set_reproducible_seed,
    )
except ImportError:
    from scripts.costars.build_costarts_subset_states import validate_costarts_subset_states
    from train_costarts_router import COSTARTSRouter, _select_expert_from_outputs
    from train_costarts_subset_utility_router import (
        SubsetUtilityCOSTARTSRouter,
        _build_state_lookup,
        _masked_action_logits,
        _state_batch,
        set_reproducible_seed,
    )


DEFAULT_TRAIN_CACHE = "cache/costarts_router_train_cache.pt"
DEFAULT_VAL_CACHE = "cache/costarts_router_val_cache.pt"
DEFAULT_SUBSET_VAL_CACHE = "cache/costarts_subset_states_val.pt"
DEFAULT_OLD_COSTARTS_CHECKPOINT = "checkpoints/costarts/best_costarts_router.pt"
DEFAULT_SUBSET_CHECKPOINT = "checkpoints/costarts_subset_utility/best_subset_utility_costarts_router.pt"
DEFAULT_ROUTERDC_NO_CONTRASTIVE = "checkpoints/best_routerdc_hard_no_contrastive.pt"
DEFAULT_ROUTERDC_CONTRASTIVE = "checkpoints/best_routerdc_hard_contrastive.pt"
DEFAULT_OUTPUT_DIR = "results/router_summary/costarts_subset_utility"


class RouterDCHardRouter(nn.Module):
    """History-only RouterDC-style hard selector used by router2 checkpoints."""

    def __init__(
        self,
        input_len: int = 96,
        num_features: int = 7,
        num_experts: int = 5,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        router_temperature: float = 1.0,
        router_type: str = "routerdc_hard",
    ) -> None:
        super().__init__()
        del router_type
        self.input_len = int(input_len)
        self.num_features = int(num_features)
        self.num_experts = int(num_experts)
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.router_temperature = float(router_temperature)
        self.input_projection = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.window_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.expert_embeddings = nn.Parameter(torch.randn(num_experts, embedding_dim))
        nn.init.normal_(self.expert_embeddings, mean=0.0, std=0.02)

    def encode(self, history: torch.Tensor) -> torch.Tensor:
        batch_size = history.shape[0]
        assert tuple(history.shape[1:]) == (self.input_len, self.num_features)
        projected = self.input_projection(history)
        encoded = self.temporal_encoder(projected.transpose(1, 2)).transpose(1, 2)
        window_embedding = self.window_projection(encoded.mean(dim=1))
        assert tuple(window_embedding.shape) == (batch_size, self.embedding_dim)
        return window_embedding

    def forward(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = F.normalize(self.encode(history), p=2, dim=-1)
        k = F.normalize(self.expert_embeddings, p=2, dim=-1)
        similarities = q @ k.T
        probabilities = torch.softmax(similarities / self.router_temperature, dim=-1)
        assert tuple(similarities.shape) == (history.shape[0], self.num_experts)
        return similarities, probabilities


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


def _parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _assert_base_cache(cache: Mapping[str, Any], split_role: str) -> None:
    assert cache["split_role"] == split_role
    assert tuple(cache["histories"].shape[1:]) == (96, 7)
    assert tuple(cache["targets"].shape[1:]) == (12, 7)
    assert tuple(cache["prediction_stack"].shape[1:3]) == (12, 7)
    assert tuple(cache["prediction_stack"].shape[:1]) == tuple(cache["targets"].shape[:1])
    assert tuple(cache["error_matrix"].shape) == (
        int(cache["num_windows"]),
        len(cache["expert_names"]),
    )
    assert tuple(cache["mse_matrix"].shape) == tuple(cache["error_matrix"].shape)


def _assert_cache_pair(train_cache: Mapping[str, Any], eval_cache: Mapping[str, Any]) -> None:
    _assert_base_cache(train_cache, "router_train")
    _assert_base_cache(eval_cache, "router_val")
    if tuple(train_cache["expert_names"]) != tuple(eval_cache["expert_names"]):
        raise AssertionError("Train/eval cache expert order mismatch.")
    for key in ("input_len", "forecast_horizon", "num_features"):
        if train_cache[key] != eval_cache[key]:
            raise AssertionError(f"Train/eval cache mismatch for {key}.")


def _mae_mse(prediction: torch.Tensor, target: torch.Tensor, target_mask: torch.Tensor) -> tuple[float, float]:
    mask = target_mask.to(torch.float32)
    denominator = mask.sum().clamp_min(1.0)
    mae = (torch.abs(prediction - target) * mask).sum() / denominator
    mse = ((prediction - target).pow(2) * mask).sum() / denominator
    return float(mae), float(mse)


def _prediction_from_expert_indices(cache: Mapping[str, Any], indices: torch.Tensor) -> torch.Tensor:
    stack = cache["prediction_stack"]
    rows = torch.arange(stack.shape[0])
    return stack[rows, :, :, indices]


def _metric_row(
    *,
    method: str,
    status: str,
    cache: Mapping[str, Any],
    oracle_mae: float,
    prediction: Optional[torch.Tensor] = None,
    selected_experts: Optional[torch.Tensor] = None,
    average_experts_queried: Optional[float] = None,
    latency_seconds: Optional[float] = None,
    parameter_count: Optional[int] = None,
    top2_indices: Optional[torch.Tensor] = None,
    first_query: Optional[torch.Tensor] = None,
    note: str = "",
    selection_split: str = "",
) -> dict[str, Any]:
    if status != "ok":
        return {
            "method": method,
            "status": status,
            "mae": "",
            "mse": "",
            "regret_to_oracle": "",
            "average_experts_queried": average_experts_queried if average_experts_queried is not None else "",
            "latency_seconds": latency_seconds if latency_seconds is not None else "",
            "latency_ms_per_sample": "",
            "parameter_count": parameter_count if parameter_count is not None else "",
            "top2_oracle_coverage": "",
            "first_query_oracle_match": "",
            "oracle_match_rate": "",
            "selection_split": selection_split,
            "note": note,
        }

    if prediction is None and selected_experts is not None:
        selected_mae = cache["error_matrix"].gather(1, selected_experts.view(-1, 1)).squeeze(1)
        selected_mse = cache["mse_matrix"].gather(1, selected_experts.view(-1, 1)).squeeze(1)
        mae = float(selected_mae.mean())
        mse = float(selected_mse.mean())
    elif prediction is not None:
        mae, mse = _mae_mse(prediction, cache["targets"], cache["target_masks"])
    else:
        raise ValueError(f"Method {method} needs prediction or selected_experts.")

    oracle_best = cache["best_expert"].to(torch.long)
    if selected_experts is None and prediction is not None:
        oracle_match_rate = ""
    else:
        oracle_match_rate = float((selected_experts.to(torch.long) == oracle_best).to(torch.float32).mean())

    top2_coverage = ""
    if top2_indices is not None:
        top2_coverage = float((top2_indices.to(torch.long) == oracle_best[:, None]).any(dim=1).to(torch.float32).mean())
    first_match = ""
    if first_query is not None:
        first_match = float((first_query.to(torch.long) == oracle_best).to(torch.float32).mean())

    latency_seconds = 0.0 if latency_seconds is None else float(latency_seconds)
    return {
        "method": method,
        "status": status,
        "mae": mae,
        "mse": mse,
        "regret_to_oracle": mae - oracle_mae,
        "average_experts_queried": average_experts_queried if average_experts_queried is not None else "",
        "latency_seconds": latency_seconds,
        "latency_ms_per_sample": latency_seconds * 1000.0 / max(int(cache["num_windows"]), 1),
        "parameter_count": parameter_count if parameter_count is not None else 0,
        "top2_oracle_coverage": top2_coverage,
        "first_query_oracle_match": first_match,
        "oracle_match_rate": oracle_match_rate,
        "selection_split": selection_split,
        "note": note,
    }


def _fit_linear_stacker(train_cache: Mapping[str, Any], ridge: float) -> tuple[torch.Tensor, torch.Tensor]:
    x = train_cache["prediction_stack"].reshape(-1, len(train_cache["expert_names"]))
    y = train_cache["targets"].reshape(-1, 1)
    valid = train_cache["target_masks"].reshape(-1).to(torch.bool)
    x = x[valid]
    y = y[valid]
    ones = torch.ones(x.shape[0], 1, dtype=x.dtype)
    design = torch.cat((x, ones), dim=1)
    gram = design.T @ design
    penalty = torch.eye(gram.shape[0], dtype=gram.dtype) * float(ridge)
    penalty[-1, -1] = 0.0
    rhs = design.T @ y
    try:
        solution = torch.linalg.solve(gram + penalty, rhs).squeeze(1)
    except RuntimeError:
        solution = torch.linalg.lstsq(gram + penalty, rhs).solution.squeeze(1)
    return solution[:-1], solution[-1:]


def _weighted_average_prediction(cache: Mapping[str, Any], weights: torch.Tensor) -> torch.Tensor:
    weights = weights.to(cache["prediction_stack"].dtype)
    return (cache["prediction_stack"] * weights.view(1, 1, 1, -1)).sum(dim=-1)


@torch.no_grad()
def _routerdc_row(
    *,
    checkpoint_path: Path,
    method: str,
    cache: Mapping[str, Any],
    oracle_mae: float,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    if not checkpoint_path.exists():
        return _metric_row(
            method=method,
            status="skipped",
            cache=cache,
            oracle_mae=oracle_mae,
            note=f"Missing checkpoint: {checkpoint_path}",
        )
    checkpoint = _load_torch(checkpoint_path)
    if tuple(checkpoint.get("selected_expert_names", ())) != tuple(cache["expert_names"]):
        return _metric_row(
            method=method,
            status="skipped",
            cache=cache,
            oracle_mae=oracle_mae,
            note="Checkpoint expert order does not match comparison cache.",
        )
    router = RouterDCHardRouter(**checkpoint["router_config"]).to(device)
    router.load_state_dict(checkpoint["router_state_dict"])
    router.eval()
    selections = []
    top2_rows = []
    start = time.perf_counter()
    for offset in range(0, int(cache["num_windows"]), batch_size):
        history = cache["histories"][offset : offset + batch_size].to(device)
        similarities, _ = router(history)
        selections.append(torch.argmax(similarities, dim=-1).cpu())
        top2_rows.append(torch.topk(similarities, k=min(2, similarities.shape[1]), dim=-1).indices.cpu())
    latency = time.perf_counter() - start
    selected = torch.cat(selections, dim=0)
    top2 = torch.cat(top2_rows, dim=0)
    return _metric_row(
        method=method,
        status="ok",
        cache=cache,
        oracle_mae=oracle_mae,
        selected_experts=selected,
        average_experts_queried=1.0,
        latency_seconds=latency,
        parameter_count=_parameter_count(router),
        top2_indices=top2,
        first_query=selected,
        selection_split="router_val_checkpoint_selected",
        note=f"Loaded {checkpoint_path}; no test data used.",
    )


@torch.no_grad()
def _old_costarts_predictions(
    *,
    checkpoint_path: Path,
    cache: Mapping[str, Any],
    batch_size: int,
    device: torch.device,
) -> tuple[COSTARTSRouter, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    checkpoint = _load_torch(checkpoint_path)
    router = COSTARTSRouter(**checkpoint["router_config"]).to(device)
    router.load_state_dict(checkpoint["router_state_dict"])
    router.eval()
    selected_rows = []
    stop_rows = []
    query_orders = []
    start = time.perf_counter()
    for offset in range(0, int(cache["num_windows"]), batch_size):
        history = cache["histories"][offset : offset + batch_size].to(device)
        outputs = router(history)
        selected, stop_step = _select_expert_from_outputs(outputs)
        selected_rows.append(selected.cpu())
        stop_rows.append(stop_step.cpu())
        query_orders.append(outputs["query_order"].cpu())
    latency = time.perf_counter() - start
    return (
        router,
        torch.cat(selected_rows, dim=0),
        torch.cat(stop_rows, dim=0),
        torch.cat(query_orders, dim=0),
        checkpoint.get("epoch", torch.tensor(-1)),
        latency,
    )


@torch.no_grad()
def _subset_rollout(
    *,
    router: SubsetUtilityCOSTARTSRouter,
    subset_cache: Mapping[str, Any],
    batch_size: int,
    device: torch.device,
    force_k: Optional[int] = None,
    oracle_second_query: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, list[list[int]], float]:
    lookup = _build_state_lookup(subset_cache)
    num_windows = int(subset_cache["num_source_windows"])
    num_experts = int(subset_cache["num_experts"])
    max_queries = int(subset_cache["max_subset_size"])
    stop_index = int(subset_cache["stop_action_index"])
    masks = [0 for _ in range(num_windows)]
    done = [False for _ in range(num_windows)]
    sequences: list[list[int]] = [[] for _ in range(num_windows)]
    start = time.perf_counter()

    if oracle_second_query:
        force_k = 2

    for step in range(max_queries):
        active = [index for index in range(num_windows) if not done[index]]
        if not active:
            break
        for offset in range(0, len(active), batch_size):
            rows = active[offset : offset + batch_size]
            state_indices = [lookup[row][masks[row]] for row in rows]
            batch = _state_batch(subset_cache, state_indices, device)
            outputs = router(
                batch["history"],
                batch["queried_mask"],
                batch["queried_expert_ids"],
                batch["queried_expert_forecasts"],
            )
            valid_mask = batch["valid_action_mask"].to(torch.bool)
            if force_k is None:
                action_logits = _masked_action_logits(outputs["action_logits"], valid_mask)
                actions = torch.argmax(action_logits, dim=-1).detach().cpu()
            elif oracle_second_query and step == 1:
                actions = []
                for local_index, sample_index in enumerate(rows):
                    current = set(sequences[sample_index])
                    errors = subset_cache["true_expert_error_vector"][lookup[sample_index][0]]
                    masked_errors = errors.clone()
                    if current:
                        masked_errors[list(current)] = float("inf")
                    actions.append(int(torch.argmin(masked_errors).item()))
                actions = torch.tensor(actions, dtype=torch.long)
            else:
                expert_valid = valid_mask[:, :num_experts]
                actions = torch.argmax(
                    outputs["action_logits"][:, :num_experts].masked_fill(~expert_valid, -1e9),
                    dim=-1,
                ).detach().cpu()

            for local_index, sample_index in enumerate(rows):
                action = int(actions[local_index])
                if force_k is None and action == stop_index and sequences[sample_index]:
                    done[sample_index] = True
                    continue
                if action == stop_index or action in sequences[sample_index]:
                    done[sample_index] = True
                    continue
                sequences[sample_index].append(action)
                masks[sample_index] |= 1 << action
                if len(sequences[sample_index]) >= max_queries:
                    done[sample_index] = True
                if force_k is not None and len(sequences[sample_index]) >= int(force_k):
                    done[sample_index] = True

    for sample_index in range(num_windows):
        if not sequences[sample_index]:
            errors = subset_cache["true_expert_error_vector"][lookup[sample_index][0]]
            action = int(torch.argmin(errors).item())
            sequences[sample_index].append(action)
            masks[sample_index] |= 1 << action

    final_state_indices = [lookup[row][masks[row]] for row in range(num_windows)]
    selected = torch.empty(num_windows, dtype=torch.long)
    top2 = torch.full((num_windows, 2), -1, dtype=torch.long)
    for row, sequence in enumerate(sequences):
        for col, expert in enumerate(sequence[:2]):
            top2[row, col] = expert
    predictions = []
    for offset in range(0, num_windows, batch_size):
        rows = list(range(offset, min(offset + batch_size, num_windows)))
        batch = _state_batch(subset_cache, final_state_indices[offset : offset + len(rows)], device)
        outputs = router(
            batch["history"],
            batch["queried_mask"],
            batch["queried_expert_ids"],
            batch["queried_expert_forecasts"],
        )
        queried_mask = batch["queried_mask"].detach().cpu()
        queried_ids = batch["queried_expert_ids"].detach().cpu()
        queried_forecasts = batch["queried_expert_forecasts"].detach().cpu()
        scores = outputs["expert_score"].detach().cpu().masked_fill(~queried_mask, -1e9)
        selected_batch = torch.argmax(scores, dim=-1)
        selected[offset : offset + len(rows)] = selected_batch
        positions = (queried_ids == selected_batch[:, None]).to(torch.float32).argmax(dim=1)
        predictions.append(queried_forecasts[torch.arange(len(rows)), positions])
    latency = time.perf_counter() - start
    return torch.cat(predictions, dim=0), selected, sequences, latency


def _sequence_tensor(sequences: Sequence[Sequence[int]], width: int = 2) -> torch.Tensor:
    tensor = torch.full((len(sequences), width), -1, dtype=torch.long)
    for row, sequence in enumerate(sequences):
        for col, expert in enumerate(sequence[:width]):
            tensor[row, col] = int(expert)
    return tensor


def _oracle_within_sequences(cache: Mapping[str, Any], sequences: Sequence[Sequence[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    selected = torch.empty(len(sequences), dtype=torch.long)
    for row, sequence in enumerate(sequences):
        candidates = torch.tensor(sequence, dtype=torch.long)
        errors = cache["error_matrix"][row].gather(0, candidates)
        selected[row] = candidates[torch.argmin(errors)]
    return _prediction_from_expert_indices(cache, selected), selected


def evaluate_final_comparison(
    *,
    train_cache_path: Path,
    val_cache_path: Path,
    subset_val_cache_path: Path,
    old_costarts_checkpoint: Path,
    subset_checkpoint: Path,
    routerdc_no_contrastive_checkpoint: Path,
    routerdc_contrastive_checkpoint: Path,
    output_dir: Path,
    batch_size: int,
    device: torch.device,
    seed: int,
    ridge: float,
) -> dict[str, Any]:
    set_reproducible_seed(seed)
    train_cache = _load_torch(train_cache_path)
    val_cache = _load_torch(val_cache_path)
    subset_cache = _load_torch(subset_val_cache_path)
    _assert_cache_pair(train_cache, val_cache)
    validate_costarts_subset_states(subset_cache)
    if tuple(subset_cache["expert_names"]) != tuple(val_cache["expert_names"]):
        raise AssertionError("Subset cache expert order does not match validation cache.")
    empty_rows = subset_cache["subset_size"] == 0
    if not torch.equal(subset_cache["sample_index"][empty_rows], val_cache["sample_indices"]):
        raise AssertionError("Subset empty states do not align with validation cache sample indices.")

    expert_names = tuple(val_cache["expert_names"])
    num_experts = len(expert_names)
    oracle_selected = val_cache["best_expert"].to(torch.long)
    oracle_mae = float(val_cache["error_matrix"].min(dim=1).values.mean())
    rows: list[dict[str, Any]] = []

    for expert_index, expert_name in enumerate(expert_names):
        selected = torch.full((int(val_cache["num_windows"]),), expert_index, dtype=torch.long)
        rows.append(
            _metric_row(
                method=f"individual_expert:{expert_name}",
                status="ok",
                cache=val_cache,
                oracle_mae=oracle_mae,
                selected_experts=selected,
                average_experts_queried=1.0,
                parameter_count=0,
                first_query=selected,
                selection_split="fixed",
                note="Cached frozen-expert prediction.",
            )
        )

    train_mean = train_cache["error_matrix"].mean(dim=0)
    val_mean = val_cache["error_matrix"].mean(dim=0)
    val_best = int(torch.argmin(val_mean))
    selected_best = torch.full((int(val_cache["num_windows"]),), val_best, dtype=torch.long)
    rows.append(
        _metric_row(
            method="best_fixed_expert",
            status="ok",
            cache=val_cache,
            oracle_mae=oracle_mae,
            selected_experts=selected_best,
            average_experts_queried=1.0,
            parameter_count=0,
            first_query=selected_best,
            selection_split="router_val_reference",
            note=f"Best fixed expert on these validation windows: {expert_names[val_best]}. Use validation-derived choices only for later untouched test.",
        )
    )

    equal_weights = torch.ones(num_experts) / num_experts
    start = time.perf_counter()
    equal_prediction = _weighted_average_prediction(val_cache, equal_weights)
    equal_latency = time.perf_counter() - start
    rows.append(
        _metric_row(
            method="equal_average_all_experts",
            status="ok",
            cache=val_cache,
            oracle_mae=oracle_mae,
            prediction=equal_prediction,
            average_experts_queried=float(num_experts),
            latency_seconds=equal_latency,
            parameter_count=0,
            selection_split="fixed",
            note="Equal average of all cached expert forecasts.",
        )
    )

    inv_weights = 1.0 / val_mean.clamp_min(1e-12)
    inv_weights = inv_weights / inv_weights.sum()
    start = time.perf_counter()
    weighted_prediction = _weighted_average_prediction(val_cache, inv_weights)
    weighted_latency = time.perf_counter() - start
    rows.append(
        _metric_row(
            method="validation_weighted_average",
            status="ok",
            cache=val_cache,
            oracle_mae=oracle_mae,
            prediction=weighted_prediction,
            average_experts_queried=float(num_experts),
            latency_seconds=weighted_latency,
            parameter_count=num_experts,
            selection_split="router_val_reference",
            note="Inverse-MAE weights fit on router_val; this is the validation-weighted baseline for later untouched test use.",
        )
    )

    weights, bias = _fit_linear_stacker(train_cache, ridge=ridge)
    start = time.perf_counter()
    linear_prediction = _weighted_average_prediction(val_cache, weights) + bias.view(1, 1, 1)
    linear_latency = time.perf_counter() - start
    rows.append(
        _metric_row(
            method="linear_stacker",
            status="ok",
            cache=val_cache,
            oracle_mae=oracle_mae,
            prediction=linear_prediction,
            average_experts_queried=float(num_experts),
            latency_seconds=linear_latency,
            parameter_count=num_experts + 1,
            selection_split="router_train",
            note=f"Global linear least-squares stacker with ridge={ridge}.",
        )
    )

    rows.append(
        _routerdc_row(
            checkpoint_path=routerdc_no_contrastive_checkpoint,
            method="routerdc_hard_without_contrastive",
            cache=val_cache,
            oracle_mae=oracle_mae,
            batch_size=batch_size,
            device=device,
        )
    )
    rows.append(
        _routerdc_row(
            checkpoint_path=routerdc_contrastive_checkpoint,
            method="routerdc_hard_with_contrastive",
            cache=val_cache,
            oracle_mae=oracle_mae,
            batch_size=batch_size,
            device=device,
        )
    )

    if old_costarts_checkpoint.exists():
        old_router, old_selected, old_stop, old_order, old_epoch, old_latency = _old_costarts_predictions(
            checkpoint_path=old_costarts_checkpoint,
            cache=val_cache,
            batch_size=batch_size,
            device=device,
        )
        rows.append(
            _metric_row(
                method="old_costarts",
                status="ok",
                cache=val_cache,
                oracle_mae=oracle_mae,
                selected_experts=old_selected,
                average_experts_queried=float(old_stop.to(torch.float32).mean()),
                latency_seconds=old_latency,
                parameter_count=_parameter_count(old_router),
                top2_indices=old_order[:, :2],
                first_query=old_order[:, 0],
                selection_split="router_val_checkpoint_selected",
                note=f"Old COSTARTS checkpoint epoch {old_epoch}.",
            )
        )
    else:
        rows.append(
            _metric_row(
                method="old_costarts",
                status="skipped",
                cache=val_cache,
                oracle_mae=oracle_mae,
                note=f"Missing checkpoint: {old_costarts_checkpoint}",
            )
        )
        old_order = None

    if subset_checkpoint.exists():
        checkpoint = _load_torch(subset_checkpoint)
        subset_router = SubsetUtilityCOSTARTSRouter(**checkpoint["router_config"]).to(device)
        subset_router.load_state_dict(checkpoint["router_state_dict"])
        subset_router.eval()
        subset_prediction, subset_selected, subset_sequences, subset_latency = _subset_rollout(
            router=subset_router,
            subset_cache=subset_cache,
            batch_size=batch_size,
            device=device,
        )
        top2_subset = _sequence_tensor(subset_sequences, 2)
        first_subset = top2_subset[:, 0]
        rows.append(
            _metric_row(
                method="improved_subset_utility_costarts",
                status="ok",
                cache=val_cache,
                oracle_mae=oracle_mae,
                prediction=subset_prediction,
                selected_experts=subset_selected,
                average_experts_queried=float(sum(len(seq) for seq in subset_sequences) / len(subset_sequences)),
                latency_seconds=subset_latency,
                parameter_count=_parameter_count(subset_router),
                top2_indices=top2_subset,
                first_query=first_subset,
                selection_split="router_val_checkpoint_selected",
                note=f"Subset-utility COSTARTS checkpoint epoch {checkpoint.get('epoch', -1)}.",
            )
        )

        top2_prediction, _, top2_sequences, top2_latency = _subset_rollout(
            router=subset_router,
            subset_cache=subset_cache,
            batch_size=batch_size,
            device=device,
            force_k=2,
        )
        top2_tensor = _sequence_tensor(top2_sequences, 2)
        top2_equal_prediction = torch.stack(
            [
                val_cache["prediction_stack"][row, :, :, top2_tensor[row, 0:2]].mean(dim=-1)
                for row in range(int(val_cache["num_windows"]))
            ],
            dim=0,
        )
        del top2_prediction
        rows.append(
            _metric_row(
                method="predicted_top2_equal_average",
                status="ok",
                cache=val_cache,
                oracle_mae=oracle_mae,
                prediction=top2_equal_prediction,
                average_experts_queried=2.0,
                latency_seconds=top2_latency,
                parameter_count=_parameter_count(subset_router),
                top2_indices=top2_tensor,
                first_query=top2_tensor[:, 0],
                selection_split="router_val_checkpoint_selected",
                note="Forced top-2 from improved subset router, equal average.",
            )
        )

        oracle_top2_prediction, oracle_top2_selected = _oracle_within_sequences(val_cache, top2_sequences)
        rows.append(
            _metric_row(
                method="oracle_within_predicted_top2",
                status="ok",
                cache=val_cache,
                oracle_mae=oracle_mae,
                prediction=oracle_top2_prediction,
                selected_experts=oracle_top2_selected,
                average_experts_queried=2.0,
                parameter_count=0,
                top2_indices=top2_tensor,
                first_query=top2_tensor[:, 0],
                selection_split="oracle_on_router_val",
                note="Upper bound: oracle chooses the better expert inside the improved router predicted top-2.",
            )
        )

        _, _, oracle_second_sequences, oracle_second_latency = _subset_rollout(
            router=subset_router,
            subset_cache=subset_cache,
            batch_size=batch_size,
            device=device,
            oracle_second_query=True,
        )
        oracle_second_prediction, oracle_second_selected = _oracle_within_sequences(
            val_cache,
            oracle_second_sequences,
        )
        oracle_second_top2 = _sequence_tensor(oracle_second_sequences, 2)
        rows.append(
            _metric_row(
                method="oracle_second_query_after_router_first",
                status="ok",
                cache=val_cache,
                oracle_mae=oracle_mae,
                prediction=oracle_second_prediction,
                selected_experts=oracle_second_selected,
                average_experts_queried=2.0,
                latency_seconds=oracle_second_latency,
                parameter_count=0,
                top2_indices=oracle_second_top2,
                first_query=oracle_second_top2[:, 0],
                selection_split="oracle_on_router_val",
                note="Upper bound: router first query, oracle chooses the best second query and final expert.",
            )
        )
    else:
        rows.append(
            _metric_row(
                method="improved_subset_utility_costarts",
                status="skipped",
                cache=val_cache,
                oracle_mae=oracle_mae,
                note=f"Missing checkpoint: {subset_checkpoint}",
            )
        )

    fixed_top2 = torch.argsort(train_mean)[:2]
    fixed_top2_prediction = val_cache["prediction_stack"][:, :, :, fixed_top2].mean(dim=-1)
    fixed_top2_first = torch.full((int(val_cache["num_windows"]),), int(fixed_top2[0]), dtype=torch.long)
    fixed_top2_indices = fixed_top2.view(1, 2).expand(int(val_cache["num_windows"]), -1)
    rows.append(
        _metric_row(
            method="fixed_top2_equal_average",
            status="ok",
            cache=val_cache,
            oracle_mae=oracle_mae,
            prediction=fixed_top2_prediction,
            average_experts_queried=2.0,
            parameter_count=0,
            top2_indices=fixed_top2_indices,
            first_query=fixed_top2_first,
            selection_split="router_train",
            note=f"Top-2 fixed experts selected on router_train: {expert_names[int(fixed_top2[0])]} + {expert_names[int(fixed_top2[1])]}.",
        )
    )

    rows.append(
        _metric_row(
            method="full_oracle",
            status="ok",
            cache=val_cache,
            oracle_mae=oracle_mae,
            selected_experts=oracle_selected,
            average_experts_queried=float(num_experts),
            parameter_count=0,
            first_query=oracle_selected,
            top2_indices=oracle_selected[:, None].expand(-1, 2),
            selection_split="oracle_on_router_val",
            note="Upper bound: best expert per validation window.",
        )
    )

    rows = sorted(
        rows,
        key=lambda row: (
            row["status"] != "ok",
            float(row["mae"]) if row["mae"] != "" else float("inf"),
            row["method"],
        ),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "final_comparison.csv"
    json_path = output_dir / "final_comparison.json"
    fields = [
        "method",
        "status",
        "mae",
        "mse",
        "regret_to_oracle",
        "average_experts_queried",
        "latency_seconds",
        "latency_ms_per_sample",
        "parameter_count",
        "top2_oracle_coverage",
        "first_query_oracle_match",
        "oracle_match_rate",
        "selection_split",
        "note",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "metadata": {
            "train_cache": str(train_cache_path),
            "validation_cache": str(val_cache_path),
            "subset_validation_cache": str(subset_val_cache_path),
            "expert_names": expert_names,
            "num_validation_windows": int(val_cache["num_windows"]),
            "same_chronological_windows": True,
            "test_data_used": False,
            "model_selection_note": (
                "Train-derived baselines use router_train. Checkpoint methods use saved checkpoints; "
                "oracle rows are marked as validation upper bounds."
            ),
        },
        "rows": rows,
    }
    json_path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate final COSTARTS baseline comparison.")
    parser.add_argument("--train-cache", default=DEFAULT_TRAIN_CACHE)
    parser.add_argument("--val-cache", default=DEFAULT_VAL_CACHE)
    parser.add_argument("--subset-val-cache", default=DEFAULT_SUBSET_VAL_CACHE)
    parser.add_argument("--old-costarts-checkpoint", default=DEFAULT_OLD_COSTARTS_CHECKPOINT)
    parser.add_argument("--subset-checkpoint", default=DEFAULT_SUBSET_CHECKPOINT)
    parser.add_argument("--routerdc-no-contrastive-checkpoint", default=DEFAULT_ROUTERDC_NO_CONTRASTIVE)
    parser.add_argument("--routerdc-contrastive-checkpoint", default=DEFAULT_ROUTERDC_CONTRASTIVE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = evaluate_final_comparison(
        train_cache_path=Path(args.train_cache),
        val_cache_path=Path(args.val_cache),
        subset_val_cache_path=Path(args.subset_val_cache),
        old_costarts_checkpoint=Path(args.old_costarts_checkpoint),
        subset_checkpoint=Path(args.subset_checkpoint),
        routerdc_no_contrastive_checkpoint=Path(args.routerdc_no_contrastive_checkpoint),
        routerdc_contrastive_checkpoint=Path(args.routerdc_contrastive_checkpoint),
        output_dir=Path(args.output_dir),
        batch_size=args.batch_size,
        device=torch.device(args.device),
        seed=args.seed,
        ridge=args.ridge,
    )
    ok_rows = [row for row in payload["rows"] if row["status"] == "ok"]
    print("\nTop comparison rows:")
    for row in ok_rows[:10]:
        print(
            f"{row['method']}: MAE={float(row['mae']):.6f}, "
            f"avg_q={row['average_experts_queried']}, "
            f"regret={float(row['regret_to_oracle']):.6f}"
        )


if __name__ == "__main__":
    main()
