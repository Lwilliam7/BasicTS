"""Train a truly sequential COSTAR-TS router from frozen-expert subset states.

Sequential COSTARTS repeatedly decides which unused frozen expert to query next,
or whether to stop. The router input contains only causal history and forecasts
from experts that have already been queried. The final forecast is the equal
average of all queried expert forecasts.
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
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.costars.build_costarts_subset_states import validate_costarts_subset_states
from scripts.costars.train_costarts_subset_utility_router import _bitmask, _load_torch
from scripts.costars.train_old_costarts_pair_selector import pair_error_matrices, pair_class_order
from scripts.router_experiment_config import load_router_experiment_config, validate_router_experiment_config


DEFAULT_TRAIN_CACHE = "cache/costarts_subset_states_train.pt"
DEFAULT_VAL_CACHE = "cache/costarts_subset_states_val.pt"
DEFAULT_SOURCE_TRAIN_CACHE = "cache/costarts_router_train_cache.pt"
DEFAULT_SOURCE_VAL_CACHE = "cache/costarts_router_val_cache.pt"
DEFAULT_OUTPUT_DIR = "checkpoints/costarts_sequential"
DEFAULT_RESULTS_DIR = "results/router_summary/costarts_sequential"
DEFAULT_CLASSIFIER_RESULTS = "results/router_summary/costarts/pair_selector/per_seed_results.csv"
DEFAULT_SEEDS = (7, 11, 13, 17, 19)
MARGIN_BINS = (
    ("<=0.005", None, 0.005),
    ("0.005_to_0.01", 0.005, 0.01),
    ("0.01_to_0.025", 0.01, 0.025),
    (">0.025", 0.025, None),
)


@dataclass(frozen=True)
class SequentialCOSTARTSConfig:
    train_cache_path: str = DEFAULT_TRAIN_CACHE
    val_cache_path: str = DEFAULT_VAL_CACHE
    source_train_cache_path: str = DEFAULT_SOURCE_TRAIN_CACHE
    source_val_cache_path: str = DEFAULT_SOURCE_VAL_CACHE
    classifier_results_path: str = DEFAULT_CLASSIFIER_RESULTS
    output_dir: str = DEFAULT_OUTPUT_DIR
    results_dir: str = DEFAULT_RESULTS_DIR
    batch_size: int = 1024
    max_epochs: int = 50
    patience: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    seed: int = 7
    hidden_dim: int = 64
    embedding_dim: int = 64
    max_query_count: int = 5
    cost_penalty: float = 0.0
    device: str = "cpu"
    debug: bool = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def json_default(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON encode {type(value)!r}")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def validate_sequential_caches(train_cache: Mapping[str, Any], val_cache: Mapping[str, Any]) -> None:
    validate_costarts_subset_states(train_cache)
    validate_costarts_subset_states(val_cache)
    if train_cache["split_role"] != "router_train":
        raise ValueError("Sequential COSTARTS training requires router_train subset cache")
    if val_cache["split_role"] != "router_val":
        raise ValueError("Sequential COSTARTS validation requires router_val subset cache")
    if tuple(train_cache["expert_names"]) != tuple(val_cache["expert_names"]):
        raise ValueError("Train/validation expert ordering differs")
    if int(train_cache["max_subset_size"]) != int(val_cache["max_subset_size"]):
        raise ValueError("Train/validation max subset sizes differ")
    if train_cache["subset_sampling_mode"] != "exhaustive" or val_cache["subset_sampling_mode"] != "exhaustive":
        raise ValueError("Sequential COSTARTS deployable rollout requires exhaustive subset-state caches")


def marginal_utility_targets(cache: Mapping[str, Any], cost_penalty: float = 0.0) -> torch.Tensor:
    target = cache["marginal_gain_equal_queried_average"].clone().to(torch.float32)
    if cost_penalty:
        costs = torch.tensor(
            [float(cache["cost_schedule_by_expert"].get(name, 0.0)) for name in cache["expert_names"]],
            dtype=torch.float32,
        )
        target = target - float(cost_penalty) * costs.view(1, -1)
    return target


def valid_utility_mask(cache: Mapping[str, Any]) -> torch.Tensor:
    has_queried = cache["subset_size"] > 0
    return cache["valid_action_mask"][:, : int(cache["num_experts"])].to(torch.bool) & has_queried[:, None]


def compute_state_features(
    queried_expert_ids: torch.Tensor,
    queried_expert_forecasts: torch.Tensor,
    max_subset_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid_slots = queried_expert_ids >= 0
    count = valid_slots.sum(dim=1).clamp_min(1).to(queried_expert_forecasts.dtype)
    slot_mask = valid_slots[:, :, None, None].to(queried_expert_forecasts.dtype)
    current_average = (queried_expert_forecasts * slot_mask).sum(dim=1) / count[:, None, None]
    centered_abs = (queried_expert_forecasts - current_average[:, None]).abs() * slot_mask
    mean_abs_disagreement = centered_abs.sum(dim=(1, 2, 3)) / (
        count * queried_expert_forecasts.shape[2] * queried_expert_forecasts.shape[3]
    ).clamp_min(1.0)
    max_abs_disagreement = centered_abs.amax(dim=(1, 2, 3))
    state_scalars = torch.stack(
        (
            count / float(max_subset_size),
            mean_abs_disagreement,
            max_abs_disagreement,
        ),
        dim=1,
    )
    return current_average, state_scalars, valid_slots


class SequentialStateDataset(Dataset):
    def __init__(self, cache: Mapping[str, Any], cost_penalty: float = 0.0) -> None:
        if cache["split_role"] not in {"router_train", "router_val"}:
            raise ValueError("SequentialStateDataset may only use router_train/router_val")
        mask = valid_utility_mask(cache).any(dim=1)
        self.indices = torch.nonzero(mask, as_tuple=False).flatten()
        self.cache = cache
        self.targets = marginal_utility_targets(cache, cost_penalty)

    def __len__(self) -> int:
        return int(self.indices.numel())

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        source_index = self.indices[index]
        return {
            "history": self.cache["history"][source_index],
            "queried_mask": self.cache["queried_mask"][source_index],
            "queried_expert_ids": self.cache["queried_expert_ids"][source_index],
            "queried_expert_forecasts": self.cache["queried_expert_forecasts"][source_index],
            "valid_expert_mask": self.cache["valid_action_mask"][source_index, : int(self.cache["num_experts"])],
            "utility_target": self.targets[source_index],
            "subset_size": self.cache["subset_size"][source_index],
            "sample_index": self.cache["sample_index"][source_index],
            "source_row": self.cache["source_row"][source_index],
        }


class SequentialCOSTARTSRouter(nn.Module):
    def __init__(
        self,
        num_experts: int = 5,
        max_subset_size: int = 5,
        input_len: int = 96,
        forecast_horizon: int = 12,
        num_features: int = 7,
        hidden_dim: int = 64,
        embedding_dim: int = 64,
    ) -> None:
        super().__init__()
        self.num_experts = int(num_experts)
        self.max_subset_size = int(max_subset_size)
        self.input_len = int(input_len)
        self.forecast_horizon = int(forecast_horizon)
        self.num_features = int(num_features)
        self.hidden_dim = int(hidden_dim)
        self.embedding_dim = int(embedding_dim)
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
        self.queried_forecast_encoder = nn.Sequential(
            nn.Linear(forecast_horizon * num_features, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
        )
        self.current_average_encoder = nn.Sequential(
            nn.Linear(forecast_horizon * num_features, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
        )
        self.scalar_encoder = nn.Sequential(
            nn.Linear(3, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
        )
        self.expert_embeddings = nn.Embedding(num_experts, embedding_dim)
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 5, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
        )
        self.utility_head = nn.Linear(embedding_dim, num_experts)

    def encode(
        self,
        history: torch.Tensor,
        queried_mask: torch.Tensor,
        queried_expert_ids: torch.Tensor,
        queried_expert_forecasts: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = history.shape[0]
        if tuple(history.shape[1:]) != (self.input_len, self.num_features):
            raise ValueError("history shape mismatch")
        if tuple(queried_mask.shape) != (batch_size, self.num_experts):
            raise ValueError("queried_mask shape mismatch")
        if tuple(queried_expert_ids.shape) != (batch_size, self.max_subset_size):
            raise ValueError("queried_expert_ids shape mismatch")
        if tuple(queried_expert_forecasts.shape) != (
            batch_size,
            self.max_subset_size,
            self.forecast_horizon,
            self.num_features,
        ):
            raise ValueError("queried_expert_forecasts shape mismatch")

        current_average, state_scalars, valid_slots = compute_state_features(
            queried_expert_ids,
            queried_expert_forecasts,
            self.max_subset_size,
        )
        history_rep = self.history_projection(self.history_encoder(history.transpose(1, 2)).squeeze(-1))
        mask_rep = self.mask_encoder(queried_mask.to(history.dtype))
        safe_ids = queried_expert_ids.clamp_min(0)
        flat_forecasts = queried_expert_forecasts.reshape(
            batch_size,
            self.max_subset_size,
            self.forecast_horizon * self.num_features,
        )
        queried_rep = self.queried_forecast_encoder(flat_forecasts) + self.expert_embeddings(safe_ids)
        queried_rep = queried_rep * valid_slots[:, :, None].to(history.dtype)
        queried_rep = queried_rep.sum(dim=1) / valid_slots.sum(dim=1, keepdim=True).clamp_min(1).to(history.dtype)
        avg_rep = self.current_average_encoder(current_average.reshape(batch_size, -1))
        scalar_rep = self.scalar_encoder(state_scalars.to(history.dtype))
        return self.fusion(torch.cat((history_rep, mask_rep, queried_rep, avg_rep, scalar_rep), dim=1))

    def forward(
        self,
        history: torch.Tensor,
        queried_mask: torch.Tensor,
        queried_expert_ids: torch.Tensor,
        queried_expert_forecasts: torch.Tensor,
    ) -> torch.Tensor:
        return self.utility_head(self.encode(history, queried_mask, queried_expert_ids, queried_expert_forecasts))

    def config_dict(self) -> dict[str, Any]:
        return {
            "router_type": "sequential_costarts",
            "num_experts": self.num_experts,
            "max_subset_size": self.max_subset_size,
            "input_len": self.input_len,
            "forecast_horizon": self.forecast_horizon,
            "num_features": self.num_features,
            "hidden_dim": self.hidden_dim,
            "embedding_dim": self.embedding_dim,
            "state_inputs": [
                "history",
                "queried_mask",
                "queried_expert_ids",
                "queried_expert_forecasts",
                "current_equal_average_forecast",
                "queried_forecast_disagreement_summary",
                "queried_expert_count",
            ],
        }


def masked_utility_scores(scores: torch.Tensor, queried_mask: torch.Tensor) -> torch.Tensor:
    return scores.masked_fill(queried_mask.to(torch.bool), -1e9)


def utility_regression_loss(scores: torch.Tensor, targets: torch.Tensor, valid_expert_mask: torch.Tensor) -> torch.Tensor:
    valid = valid_expert_mask.to(torch.bool) & torch.isfinite(targets)
    if not torch.any(valid):
        return scores.sum() * 0.0
    return F.smooth_l1_loss(scores[valid], targets[valid])


def move_batch(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def build_state_lookup(cache: Mapping[str, Any]) -> list[dict[int, int]]:
    lookup = [dict() for _ in range(int(cache["num_source_windows"]))]
    for state_index in range(int(cache["num_states"])):
        source_row = int(cache["source_row"][state_index])
        lookup[source_row][_bitmask(cache["queried_mask"][state_index])] = state_index
    return lookup


def state_batch(cache: Mapping[str, Any], state_indices: Sequence[int], device: torch.device) -> dict[str, torch.Tensor]:
    index = torch.tensor(state_indices, dtype=torch.long)
    keys = (
        "history",
        "queried_mask",
        "queried_expert_ids",
        "queried_expert_forecasts",
        "valid_action_mask",
        "marginal_gain_equal_queried_average",
        "true_targets",
        "target_mask",
        "true_expert_error_vector",
        "source_row",
        "sample_index",
    )
    return {key: cache[key][index].to(device) for key in keys}


def equal_average_forecast_from_state(batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    ids = batch["queried_expert_ids"]
    forecasts = batch["queried_expert_forecasts"]
    valid = ids >= 0
    count = valid.sum(dim=1).clamp_min(1).to(forecasts.dtype)
    return (forecasts * valid[:, :, None, None].to(forecasts.dtype)).sum(dim=1) / count[:, None, None]


def mae_mse_per_window(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mask_f = mask.to(prediction.dtype)
    denom = mask_f.sum(dim=(1, 2)).clamp_min(1.0)
    error = prediction - target
    mae = (error.abs() * mask_f).sum(dim=(1, 2)) / denom
    mse = (error.square() * mask_f).sum(dim=(1, 2)) / denom
    return mae, mse


def binary_auc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    if len(scores) != len(labels) or not scores:
        return float("nan")
    y = np.array(labels, dtype=bool)
    if y.sum() == 0 or y.sum() == y.size:
        return float("nan")
    s = np.array(scores, dtype=float)
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)
    pos_ranks = ranks[y].sum()
    n_pos = float(y.sum())
    n_neg = float((~y).sum())
    return float((pos_ranks - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg))


def select_fixed_first_expert(source_train: Mapping[str, Any], source_val: Mapping[str, Any]) -> tuple[int, str]:
    train_errors = source_train["error_matrix"].mean(dim=0)
    val_errors = source_val["error_matrix"].mean(dim=0)
    combined = 0.5 * (train_errors + val_errors)
    index = int(combined.argmin().item())
    return index, str(source_train["expert_names"][index])


def fixed_pair_baseline(source_val: Mapping[str, Any]) -> dict[str, Any]:
    pairs = pair_class_order()
    pair_mae, pair_mse = pair_error_matrices(source_val, pairs)
    index = int(pair_mae.mean(dim=0).argmin().item())
    return {
        "pair": pairs[index]["pair"],
        "mae": float(pair_mae[:, index].mean().item()),
        "mse": float(pair_mse[:, index].mean().item()),
        "average_experts_queried": 2.0,
    }


def source_baselines(source_val: Mapping[str, Any], classifier_rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    expert_mae = source_val["error_matrix"].mean(dim=0)
    expert_mse = source_val["mse_matrix"].mean(dim=0)
    best_expert = int(expert_mae.argmin().item())
    all_prediction = source_val["prediction_stack"].mean(dim=-1)
    all_mae, all_mse = mae_mse_per_window(all_prediction, source_val["targets"], source_val["target_masks"])
    classifier_maes = [float(row["selected_pair_mae"]) for row in classifier_rows if row.get("mode", "") in {"", "threshold"} or "selected_pair_mae" in row]
    baseline = {
        "best_fixed_single_expert": {
            "expert": source_val["expert_names"][best_expert],
            "mae": float(expert_mae[best_expert].item()),
            "mse": float(expert_mse[best_expert].item()),
            "average_experts_queried": 1.0,
        },
        "best_fixed_pair": fixed_pair_baseline(source_val),
        "all_expert_equal_average": {
            "mae": float(all_mae.mean().item()),
            "mse": float(all_mse.mean().item()),
            "average_experts_queried": 5.0,
        },
        "full_oracle_single_expert": {
            "mae": float(source_val["error_matrix"].min(dim=1).values.mean().item()),
            "average_experts_queried": 5.0,
            "diagnostic_only": True,
        },
    }
    if classifier_maes:
        values = np.array(classifier_maes, dtype=float)
        baseline["existing_one_shot_pair_selector"] = {
            "mae_mean": float(values.mean()),
            "mae_std": float(values.std(ddof=0)),
            "average_experts_queried": 2.0,
        }
    return baseline


@torch.no_grad()
def rollout_policy(
    router: SequentialCOSTARTSRouter,
    cache: Mapping[str, Any],
    *,
    fixed_first_expert: int,
    threshold: float,
    max_query_count: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    router.eval()
    validate_costarts_subset_states(cache)
    lookup = build_state_lookup(cache)
    num_windows = int(cache["num_source_windows"])
    num_experts = int(cache["num_experts"])
    masks = [1 << fixed_first_expert for _ in range(num_windows)]
    query_history = [[fixed_first_expert] for _ in range(num_windows)]
    done = [False for _ in range(num_windows)]
    decision_predicted = []
    decision_actual = []
    useful_queries = []
    queried_counts = torch.zeros(num_experts, dtype=torch.long)
    queried_counts[fixed_first_expert] = num_windows
    stop_counts: dict[int, int] = {}

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
            masked_scores = masked_utility_scores(scores, batch["queried_mask"])
            best_scores, actions = masked_scores.max(dim=1)
            actual_targets = batch["marginal_gain_equal_queried_average"]
            for local, source_row in enumerate(rows):
                action = int(actions[local].detach().cpu().item())
                predicted = float(best_scores[local].detach().cpu().item())
                if predicted <= threshold or len(query_history[source_row]) >= max_query_count:
                    done[source_row] = True
                    stop_counts[len(query_history[source_row])] = stop_counts.get(len(query_history[source_row]), 0) + 1
                    continue
                actual = float(actual_targets[local, action].detach().cpu().item())
                decision_predicted.append(predicted)
                decision_actual.append(actual)
                useful_queries.append(actual > 0)
                query_history[source_row].append(action)
                queried_counts[action] += 1
                masks[source_row] |= 1 << action
                if len(query_history[source_row]) >= max_query_count:
                    done[source_row] = True
                    stop_counts[len(query_history[source_row])] = stop_counts.get(len(query_history[source_row]), 0) + 1
    for row in range(num_windows):
        if not done[row]:
            stop_counts[len(query_history[row])] = stop_counts.get(len(query_history[row]), 0) + 1

    final_state_indices = [lookup[row][masks[row]] for row in range(num_windows)]
    maes = []
    mses = []
    query_counts = []
    oracle_maes = []
    per_window_rows = []
    for offset in range(0, num_windows, batch_size):
        rows = list(range(offset, min(offset + batch_size, num_windows)))
        batch = state_batch(cache, final_state_indices[offset : offset + len(rows)], device)
        prediction = equal_average_forecast_from_state(batch)
        mae, mse = mae_mse_per_window(prediction, batch["true_targets"], batch["target_mask"])
        maes.append(mae.cpu())
        mses.append(mse.cpu())
        oracle_maes.append(batch["true_expert_error_vector"].min(dim=1).values.cpu())
        for local, row in enumerate(rows):
            count = len(query_history[row])
            query_counts.append(count)
            per_window_rows.append({
                "source_row": row,
                "sample_index": int(batch["sample_index"][local].detach().cpu().item()),
                "query_count": count,
                "queried_experts": " ".join(str(item) for item in query_history[row]),
                "mae": float(mae[local].detach().cpu().item()),
                "mse": float(mse[local].detach().cpu().item()),
            })
    mae_all = torch.cat(maes)
    mse_all = torch.cat(mses)
    oracle_all = torch.cat(oracle_maes)
    query_count_tensor = torch.tensor(query_counts, dtype=torch.long)
    if len(decision_predicted) > 1 and np.std(decision_predicted) > 0 and np.std(decision_actual) > 0:
        utility_corr = float(np.corrcoef(np.array(decision_predicted), np.array(decision_actual))[0, 1])
    else:
        utility_corr = float("nan")
    useful_precision = float(np.mean(useful_queries) * 100.0) if useful_queries else 0.0
    harmful_rate = float((1.0 - np.mean(useful_queries)) * 100.0) if useful_queries else 0.0
    useful_auc = binary_auc(decision_predicted, useful_queries)
    high_margin = {}
    actual_array = np.array(decision_actual, dtype=float)
    useful_array = np.array(useful_queries, dtype=bool)
    for label, lower, upper in MARGIN_BINS:
        if actual_array.size == 0:
            high_margin[label] = {"count": 0, "useful_query_precision": float("nan")}
            continue
        margin = np.abs(actual_array)
        mask = np.ones_like(margin, dtype=bool)
        if lower is not None:
            mask &= margin > lower
        if upper is not None:
            mask &= margin <= upper
        high_margin[label] = {
            "count": int(mask.sum()),
            "useful_query_precision": float(useful_array[mask].mean() * 100.0) if mask.any() else float("nan"),
        }
    query_dist = {
        str(index): int((query_count_tensor == index).sum().item())
        for index in range(1, num_experts + 1)
        if int((query_count_tensor == index).sum().item())
    }
    query_count_performance = {}
    for index in range(1, num_experts + 1):
        count_mask = query_count_tensor == index
        if torch.any(count_mask):
            query_count_performance[str(index)] = {
                "count": int(count_mask.sum().item()),
                "mae": float(mae_all[count_mask].mean().item()),
                "mse": float(mse_all[count_mask].mean().item()),
            }
    return {
        "validation_mae": float(mae_all.mean().item()),
        "validation_mse": float(mse_all.mean().item()),
        "regret_to_full_oracle": float((mae_all - oracle_all).mean().item()),
        "average_experts_queried": float(query_count_tensor.to(torch.float32).mean().item()),
        "relative_expert_cost": float(query_count_tensor.to(torch.float32).mean().item() / num_experts),
        "query_count_distribution": query_dist,
        "stop_counts": {str(key): int(value) for key, value in sorted(stop_counts.items())},
        "expert_query_frequency": {
            str(cache["expert_names"][index]): float(queried_counts[index].item() * 100.0 / max(num_windows, 1))
            for index in range(num_experts)
        },
        "predicted_actual_utility_correlation": utility_corr,
        "useful_query_auc": useful_auc,
        "useful_query_precision": useful_precision,
        "harmful_query_rate": harmful_rate,
        "high_margin_windows": high_margin,
        "query_count_performance": query_count_performance,
        "per_window": per_window_rows,
    }


def threshold_grid(scores: torch.Tensor) -> list[float]:
    values = scores.detach().cpu().numpy()
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return [0.0]
    quantiles = np.quantile(finite, np.linspace(0, 1, 21)).tolist()
    fixed = [-0.05, -0.025, -0.01, -0.005, -0.001, 0.0, 0.001, 0.005, 0.01, 0.025, 0.05]
    return sorted(set(float(item) for item in [*quantiles, *fixed]))


@torch.no_grad()
def collect_candidate_scores(
    router: SequentialCOSTARTSRouter,
    cache: Mapping[str, Any],
    fixed_first_expert: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    lookup = build_state_lookup(cache)
    state_indices = [lookup[row][1 << fixed_first_expert] for row in range(int(cache["num_source_windows"]))]
    scores = []
    for offset in range(0, len(state_indices), batch_size):
        batch = state_batch(cache, state_indices[offset : offset + batch_size], device)
        output = router(
            batch["history"],
            batch["queried_mask"],
            batch["queried_expert_ids"],
            batch["queried_expert_forecasts"],
        )
        scores.append(masked_utility_scores(output, batch["queried_mask"]).max(dim=1).values.cpu())
    return torch.cat(scores)


def select_threshold(
    router: SequentialCOSTARTSRouter,
    val_cache: Mapping[str, Any],
    *,
    fixed_first_expert: int,
    max_query_count: int,
    batch_size: int,
    device: torch.device,
) -> tuple[float, list[dict[str, Any]], dict[str, Any]]:
    candidate_scores = collect_candidate_scores(router, val_cache, fixed_first_expert, batch_size, device)
    rows = []
    best_threshold = 0.0
    best_metrics = None
    for threshold in threshold_grid(candidate_scores):
        metrics = rollout_policy(
            router,
            val_cache,
            fixed_first_expert=fixed_first_expert,
            threshold=threshold,
            max_query_count=max_query_count,
            batch_size=batch_size,
            device=device,
        )
        row = {
            "threshold": threshold,
            "validation_mae": metrics["validation_mae"],
            "validation_mse": metrics["validation_mse"],
            "average_experts_queried": metrics["average_experts_queried"],
            "useful_query_precision": metrics["useful_query_precision"],
            "harmful_query_rate": metrics["harmful_query_rate"],
        }
        rows.append(row)
        if best_metrics is None or metrics["validation_mae"] < best_metrics["validation_mae"] - 1e-12:
            best_threshold = threshold
            best_metrics = metrics
    if best_metrics is None:
        raise RuntimeError("No threshold candidates evaluated")
    return best_threshold, rows, best_metrics


def evaluate_state_loss(
    router: SequentialCOSTARTSRouter,
    cache: Mapping[str, Any],
    cost_penalty: float,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    router.eval()
    dataset = SequentialStateDataset(cache, cost_penalty)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_loss = 0.0
    total_count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        scores = router(
            batch["history"],
            batch["queried_mask"],
            batch["queried_expert_ids"],
            batch["queried_expert_forecasts"],
        )
        loss = utility_regression_loss(scores, batch["utility_target"], batch["valid_expert_mask"])
        count = int(batch["history"].shape[0])
        total_loss += float(loss.detach().cpu().item()) * count
        total_count += count
    return {"validation_state_smooth_l1": total_loss / max(total_count, 1)}


def save_checkpoint(
    path: Path,
    router: SequentialCOSTARTSRouter,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Mapping[str, Any],
    config: SequentialCOSTARTSConfig,
    expert_names: Sequence[str],
    threshold: float,
    fixed_first_expert: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "router_state_dict": router.state_dict(),
            "router_config": router.config_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": dict(metrics),
            "training_config": asdict(config),
            "expert_names": list(expert_names),
            "router_type": "sequential_costarts",
            "model_name": "Sequential COSTARTS",
            "fixed_first_expert": int(fixed_first_expert),
            "fixed_first_expert_name": str(expert_names[fixed_first_expert]),
            "selected_stop_threshold": float(threshold),
            "target": "marginal_gain_equal_queried_average",
            "loss": "smooth_l1_on_unused_expert_utilities",
            "final_forecast": "equal_average_of_queried_expert_forecasts",
            "test_set_used": False,
            "experts_loaded": False,
            "experts_updated": False,
        },
        path,
    )


def train_one_seed(
    seed: int,
    base_config: SequentialCOSTARTSConfig,
    train_cache: Mapping[str, Any],
    val_cache: Mapping[str, Any],
    source_train: Mapping[str, Any],
    source_val: Mapping[str, Any],
) -> dict[str, Any]:
    config = SequentialCOSTARTSConfig(**{**asdict(base_config), "seed": seed})
    set_seed(seed)
    device = torch.device(config.device)
    fixed_first_expert, fixed_first_name = select_fixed_first_expert(source_train, source_val)
    router = SequentialCOSTARTSRouter(
        num_experts=int(train_cache["num_experts"]),
        max_subset_size=int(train_cache["max_subset_size"]),
        forecast_horizon=int(train_cache["forecast_horizon"]),
        num_features=int(train_cache["num_features"]),
        hidden_dim=config.hidden_dim,
        embedding_dim=config.embedding_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(router.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    train_dataset = SequentialStateDataset(train_cache, config.cost_penalty)
    loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    output_dir = Path(config.output_dir) / f"seed_{seed}"
    results_dir = Path(config.results_dir) / f"seed_{seed}"
    best_path = output_dir / "best_sequential_costarts_router.pt"
    last_path = output_dir / "last_sequential_costarts_router.pt"
    curves = []
    best_mae = math.inf
    best_epoch = 0
    best_threshold = 0.0
    best_metrics = None
    best_state = None
    bad_epochs = 0

    for epoch in range(1, config.max_epochs + 1):
        router.train()
        loss_sum = 0.0
        count_sum = 0
        for batch in loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            scores = router(
                batch["history"],
                batch["queried_mask"],
                batch["queried_expert_ids"],
                batch["queried_expert_forecasts"],
            )
            loss = utility_regression_loss(scores, batch["utility_target"], batch["valid_expert_mask"])
            loss.backward()
            if config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(router.parameters(), config.grad_clip_norm)
            optimizer.step()
            count = int(batch["history"].shape[0])
            loss_sum += float(loss.detach().cpu().item()) * count
            count_sum += count
        threshold, threshold_rows, val_metrics = select_threshold(
            router,
            val_cache,
            fixed_first_expert=fixed_first_expert,
            max_query_count=config.max_query_count,
            batch_size=config.batch_size,
            device=device,
        )
        state_loss = evaluate_state_loss(router, val_cache, config.cost_penalty, config.batch_size, device)
        curves.append({
            "epoch": epoch,
            "train_smooth_l1": loss_sum / max(count_sum, 1),
            "selected_threshold": threshold,
            "validation_mae": val_metrics["validation_mae"],
            "validation_mse": val_metrics["validation_mse"],
            "average_experts_queried": val_metrics["average_experts_queried"],
            "useful_query_precision": val_metrics["useful_query_precision"],
            "harmful_query_rate": val_metrics["harmful_query_rate"],
            **state_loss,
        })
        if val_metrics["validation_mae"] < best_mae - 1e-12:
            best_mae = val_metrics["validation_mae"]
            best_epoch = epoch
            best_threshold = threshold
            best_metrics = val_metrics
            best_state = {key: value.detach().cpu().clone() for key, value in router.state_dict().items()}
            bad_epochs = 0
            write_csv(results_dir / "threshold_search.csv", threshold_rows)
        else:
            bad_epochs += 1
        if config.debug:
            print(f"seed={seed} epoch={epoch} val_mae={val_metrics['validation_mae']:.6f} threshold={threshold:.6f}")
        if bad_epochs >= config.patience:
            break
    if best_state is None or best_metrics is None:
        raise RuntimeError("Sequential COSTARTS failed to produce a checkpoint")
    router.load_state_dict(best_state)
    no_threshold_metrics = rollout_policy(
        router,
        val_cache,
        fixed_first_expert=fixed_first_expert,
        threshold=0.0,
        max_query_count=config.max_query_count,
        batch_size=config.batch_size,
        device=device,
    )
    save_checkpoint(best_path, router, optimizer, best_epoch, best_metrics, config, train_cache["expert_names"], best_threshold, fixed_first_expert)
    save_checkpoint(last_path, router, optimizer, len(curves), best_metrics, config, train_cache["expert_names"], best_threshold, fixed_first_expert)
    write_csv(results_dir / "training_curves.csv", curves)
    write_csv(
        results_dir / "validation_per_window.csv",
        best_metrics["per_window"],
        ("source_row", "sample_index", "query_count", "queried_experts", "mae", "mse"),
    )
    summary = {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_mae": best_metrics["validation_mae"],
        "best_validation_mse": best_metrics["validation_mse"],
        "selected_threshold": best_threshold,
        "fixed_first_expert": fixed_first_name,
        "fixed_first_expert_index": fixed_first_expert,
        "average_experts_queried": best_metrics["average_experts_queried"],
        "relative_expert_cost": best_metrics["relative_expert_cost"],
        "query_count_distribution": best_metrics["query_count_distribution"],
        "stop_counts": best_metrics["stop_counts"],
        "expert_query_frequency": best_metrics["expert_query_frequency"],
        "predicted_actual_utility_correlation": best_metrics["predicted_actual_utility_correlation"],
        "useful_query_auc": best_metrics["useful_query_auc"],
        "useful_query_precision": best_metrics["useful_query_precision"],
        "harmful_query_rate": best_metrics["harmful_query_rate"],
        "high_margin_windows": best_metrics["high_margin_windows"],
        "query_count_performance": best_metrics["query_count_performance"],
        "regret_to_full_oracle": best_metrics["regret_to_full_oracle"],
        "no_threshold_validation_mae": no_threshold_metrics["validation_mae"],
        "no_threshold_validation_mse": no_threshold_metrics["validation_mse"],
        "no_threshold_average_experts_queried": no_threshold_metrics["average_experts_queried"],
        "no_threshold_useful_query_precision": no_threshold_metrics["useful_query_precision"],
        "no_threshold_harmful_query_rate": no_threshold_metrics["harmful_query_rate"],
        "best_checkpoint": str(best_path),
    }
    (results_dir / "seed_summary.json").write_text(json.dumps(summary, indent=2, default=json_default), encoding="utf-8")
    return summary


def greedy_oracle_rollout(cache: Mapping[str, Any], fixed_first_expert: int) -> dict[str, Any]:
    lookup = build_state_lookup(cache)
    num_windows = int(cache["num_source_windows"])
    num_experts = int(cache["num_experts"])
    masks = [1 << fixed_first_expert for _ in range(num_windows)]
    histories = [[fixed_first_expert] for _ in range(num_windows)]
    for _ in range(1, num_experts):
        changed = False
        for row in range(num_windows):
            state_index = lookup[row][masks[row]]
            gains = cache["marginal_gain_equal_queried_average"][state_index].clone()
            gains = gains.masked_fill(cache["queried_mask"][state_index], float("-inf"))
            action = int(gains.argmax().item())
            if float(gains[action].item()) > 0:
                masks[row] |= 1 << action
                histories[row].append(action)
                changed = True
        if not changed:
            break
    maes = []
    mses = []
    counts = []
    for row in range(num_windows):
        state_index = lookup[row][masks[row]]
        batch = {key: cache[key][state_index : state_index + 1] for key in ("queried_expert_ids", "queried_expert_forecasts", "true_targets", "target_mask")}
        prediction = equal_average_forecast_from_state(batch)
        mae, mse = mae_mse_per_window(prediction, batch["true_targets"], batch["target_mask"])
        maes.append(float(mae.item()))
        mses.append(float(mse.item()))
        counts.append(len(histories[row]))
    return {
        "mae": float(np.mean(maes)),
        "mse": float(np.mean(mses)),
        "average_experts_queried": float(np.mean(counts)),
        "diagnostic_only": True,
    }


def load_classifier_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def aggregate(seed_summaries: Sequence[Mapping[str, Any]], baselines: Mapping[str, Any], results_dir: Path) -> dict[str, Any]:
    per_seed_rows = []
    fixed_pair_mae = float(baselines["best_fixed_pair"]["mae"])
    for summary in seed_summaries:
        per_seed_rows.append({
            "seed": summary["seed"],
            "validation_mae": summary["best_validation_mae"],
            "validation_mse": summary["best_validation_mse"],
            "improvement_over_best_fixed_pair": fixed_pair_mae - float(summary["best_validation_mae"]),
            "regret_to_full_oracle": summary["regret_to_full_oracle"],
            "average_experts_queried": summary["average_experts_queried"],
            "relative_expert_cost": summary["relative_expert_cost"],
            "predicted_actual_utility_correlation": summary["predicted_actual_utility_correlation"],
            "useful_query_auc": summary["useful_query_auc"],
            "useful_query_precision": summary["useful_query_precision"],
            "harmful_query_rate": summary["harmful_query_rate"],
            "selected_threshold": summary["selected_threshold"],
            "fixed_first_expert": summary["fixed_first_expert"],
            "no_threshold_validation_mae": summary["no_threshold_validation_mae"],
            "no_threshold_average_experts_queried": summary["no_threshold_average_experts_queried"],
        })
    write_csv(results_dir / "per_seed_results.csv", per_seed_rows)
    diagnostics = {
        str(summary["seed"]): {
            "query_count_distribution": summary["query_count_distribution"],
            "stop_counts": summary["stop_counts"],
            "expert_query_frequency": summary["expert_query_frequency"],
            "high_margin_windows": summary["high_margin_windows"],
            "query_count_performance": summary["query_count_performance"],
        }
        for summary in seed_summaries
    }
    (results_dir / "per_seed_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, default=json_default),
        encoding="utf-8",
    )
    aggregate_rows = []
    for metric in (
        "validation_mae",
        "validation_mse",
        "improvement_over_best_fixed_pair",
        "regret_to_full_oracle",
        "average_experts_queried",
        "relative_expert_cost",
        "predicted_actual_utility_correlation",
        "useful_query_auc",
        "useful_query_precision",
        "harmful_query_rate",
        "no_threshold_validation_mae",
        "no_threshold_average_experts_queried",
    ):
        values = np.array([float(row[metric]) for row in per_seed_rows], dtype=float)
        aggregate_rows.append({
            "metric": metric,
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values)),
            "min": float(np.nanmin(values)),
            "max": float(np.nanmax(values)),
        })
    write_csv(results_dir / "aggregate_results.csv", aggregate_rows)
    mean_mae = next(row["mean"] for row in aggregate_rows if row["metric"] == "validation_mae")
    mean_queries = next(row["mean"] for row in aggregate_rows if row["metric"] == "average_experts_queried")
    mean_corr = next(row["mean"] for row in aggregate_rows if row["metric"] == "predicted_actual_utility_correlation")
    mean_auc = next(row["mean"] for row in aggregate_rows if row["metric"] == "useful_query_auc")
    summary = {
        "model_name": "Sequential COSTARTS",
        "router_type": "sequential_costarts",
        "seeds": [row["seed"] for row in per_seed_rows],
        "baselines": baselines,
        "per_seed": per_seed_rows,
        "aggregate": aggregate_rows,
        "success_checks": {
            "beats_best_fixed_pair_mean_validation_mae": mean_mae < fixed_pair_mae,
            "clear_cost_accuracy_tradeoff": mean_mae <= fixed_pair_mae * 1.01 and mean_queries < 2.0,
            "avoids_always_stop_or_query_all": 1.0 < mean_queries < 5.0,
            "utility_prediction_better_than_random_directional": mean_corr > 0.0 and mean_auc > 0.5,
            "no_data_leakage_known": True,
        },
        "leakage_assertions": {
            "train_split": "router_train",
            "validation_split": "router_val",
            "test_set_used": False,
            "forecasting_experts_retrained": False,
            "unqueried_forecasts_in_router_input": False,
            "threshold_selection": "router_val only",
        },
    }
    (results_dir / "aggregate_summary.json").write_text(json.dumps(summary, indent=2, default=json_default), encoding="utf-8")
    write_report(results_dir / "sequential_costarts_report.md", summary)
    return summary


def write_report(path: Path, summary: Mapping[str, Any]) -> None:
    agg = {row["metric"]: row for row in summary["aggregate"]}
    baselines = summary["baselines"]
    lines = [
        "# Sequential COSTARTS Report",
        "",
        "## State",
        "",
        "The router receives causal history, queried expert identities/mask, queried forecasts, current equal-average forecast, queried-forecast disagreement summaries, and queried count. Unqueried forecasts are absent from the state.",
        "",
        "## Target",
        "",
        "`utility_j = MAE(current_equal_average, target) - MAE(equal_average(S + j), target)` for each unused expert. STOP uses thresholded predicted utility.",
        "",
        "## Validation",
        "",
        f"- Sequential COSTARTS MAE `{agg['validation_mae']['mean']:.6f}` +/- `{agg['validation_mae']['std']:.6f}`.",
        f"- Zero-threshold sequential MAE `{agg['no_threshold_validation_mae']['mean']:.6f}` +/- `{agg['no_threshold_validation_mae']['std']:.6f}`.",
        f"- Average queried experts `{agg['average_experts_queried']['mean']:.3f}` +/- `{agg['average_experts_queried']['std']:.3f}`.",
        f"- Improvement over best fixed pair `{agg['improvement_over_best_fixed_pair']['mean']:.6f}` +/- `{agg['improvement_over_best_fixed_pair']['std']:.6f}`.",
        f"- Utility correlation `{agg['predicted_actual_utility_correlation']['mean']:.4f}`; useful-query AUC `{agg['useful_query_auc']['mean']:.4f}`.",
        f"- Best fixed pair `{baselines['best_fixed_pair']['pair']}` MAE `{baselines['best_fixed_pair']['mae']:.6f}`.",
        f"- All-expert equal average MAE `{baselines['all_expert_equal_average']['mae']:.6f}`.",
        f"- Existing one-shot pair selector MAE `{baselines.get('existing_one_shot_pair_selector', {}).get('mae_mean', float('nan')):.6f}`.",
        f"- Greedy oracle sequential MAE `{baselines['greedy_oracle_sequential']['mae']:.6f}`.",
        "",
        "## Success Checks",
        "",
        json.dumps(summary["success_checks"], indent=2),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_seeds(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", default=DEFAULT_TRAIN_CACHE)
    parser.add_argument("--val-cache", default=DEFAULT_VAL_CACHE)
    parser.add_argument("--source-train-cache", default=DEFAULT_SOURCE_TRAIN_CACHE)
    parser.add_argument("--source-val-cache", default=DEFAULT_SOURCE_VAL_CACHE)
    parser.add_argument("--classifier-results", default=DEFAULT_CLASSIFIER_RESULTS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--max-query-count", type=int, default=5)
    parser.add_argument("--cost-penalty", type=float, default=0.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> dict[str, Any]:
    args = parse_args(argv)
    seeds = parse_seeds(args.seeds)
    config = SequentialCOSTARTSConfig(
        train_cache_path=args.train_cache,
        val_cache_path=args.val_cache,
        source_train_cache_path=args.source_train_cache,
        source_val_cache_path=args.source_val_cache,
        classifier_results_path=args.classifier_results,
        output_dir=args.output_dir,
        results_dir=args.results_dir,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        max_query_count=args.max_query_count,
        cost_penalty=args.cost_penalty,
        device=args.device,
        debug=args.debug,
    )
    validate_router_experiment_config(load_router_experiment_config(), require_checkpoints=False, require_data=False, require_cache_parent=True)
    train_cache = _load_torch(config.train_cache_path)
    val_cache = _load_torch(config.val_cache_path)
    source_train = _load_torch(config.source_train_cache_path)
    source_val = _load_torch(config.source_val_cache_path)
    validate_sequential_caches(train_cache, val_cache)
    baselines = source_baselines(source_val, load_classifier_rows(Path(config.classifier_results_path)))
    first_index, first_name = select_fixed_first_expert(source_train, source_val)
    baselines["fixed_first_expert"] = {"expert": first_name, "index": first_index, "selection_split": "router_train_plus_router_val"}
    baselines["greedy_oracle_sequential"] = greedy_oracle_rollout(val_cache, first_index)
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    seed_summaries = [
        train_one_seed(seed, config, train_cache, val_cache, source_train, source_val)
        for seed in seeds
    ]
    summary = aggregate(seed_summaries, baselines, results_dir)
    forbidden = [Path("cache/costarts_router_test_cache.pt"), Path("cache/costarts_locked_test_cache.pt")]
    created = [str(path) for path in forbidden if path.exists()]
    if created:
        raise RuntimeError(f"Forbidden test cache exists: {created}")
    return summary


if __name__ == "__main__":
    main()
