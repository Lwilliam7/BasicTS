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
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    from scripts.build_costarts_subset_states import validate_costarts_subset_states
    from scripts.router_experiment_config import (
        load_router_experiment_config,
        print_router_experiment_config,
        validate_router_experiment_config,
    )
except ImportError:
    from build_costarts_subset_states import validate_costarts_subset_states
    from router_experiment_config import (
        load_router_experiment_config,
        print_router_experiment_config,
        validate_router_experiment_config,
    )


DEFAULT_OUTPUT_DIR = "checkpoints/costarts_subset_utility"
DEFAULT_RESULTS_DIR = "results/router_summary/costarts_subset_utility"
DEFAULT_TRAIN_CACHE = "cache/costarts_subset_states_train.pt"
DEFAULT_VAL_CACHE = "cache/costarts_subset_states_val.pt"
UTILITY_FINALIZERS = {"best_single", "equal_average"}
DEPLOYMENT_FINALIZERS = {"best_reranked", "equal_average"}
STATE_SAMPLING_MODES = {"uniform", "action_balanced"}
FIRST_QUERY_TARGETS = {"hard", "soft"}
FIRST_QUERY_HEADS = {"shared", "separate"}
FIRST_QUERY_INITIALIZATIONS = {"random", "routerdc_frozen", "routerdc_finetune"}
DEFAULT_ROUTERDC_FIRST_QUERY_CHECKPOINT = "checkpoints/best_routerdc_hard_contrastive.pt"


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
    cost_mode: str = "equal"
    cost_file: Optional[str] = None
    selection_metric: str = "cost_aware_objective"
    utility_finalizer: str = "equal_average"
    deployment_finalizer: str = "equal_average"
    allow_finalizer_mismatch: bool = False
    state_sampling: str = "uniform"
    first_query_target: str = "soft"
    first_query_temperature: float = 0.02
    first_query_loss_weight: float = 2.0
    first_query_sampling_ratio: float = 0.0
    first_query_head: str = "separate"
    first_query_regret_loss_weight: float = 1.0
    first_query_initialization: str = "random"
    routerdc_checkpoint_path: str = DEFAULT_ROUTERDC_FIRST_QUERY_CHECKPOINT
    routerdc_consistency_weight: float = 0.0
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
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return str(value)
    return value


def _read_cost_file(path: Union[str, Path]) -> dict[str, float]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"cost file does not exist: {path}")
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, Mapping):
            raise ValueError("JSON cost file must be an object mapping expert names to costs")
        return {str(key): float(value) for key, value in data.items()}
    costs: dict[str, float] = {}
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            name = row.get("expert") or row.get("model") or row.get("name")
            value = row.get("cost") or row.get("latency") or row.get("latency_ms") or row.get("ms")
            if name is None or value is None:
                raise ValueError("CSV cost file needs expert/model/name and cost/latency columns")
            costs[str(name)] = float(value)
    return costs


def load_and_normalize_expert_costs(
    expert_names: Sequence[str],
    *,
    cost_mode: str = "equal",
    cost_file: Optional[Union[str, Path]] = None,
    configured_cost_schedule: Optional[Mapping[str, float]] = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if cost_mode not in {"equal", "configured", "latency"}:
        raise ValueError("cost_mode must be one of: equal, configured, latency")
    if cost_mode == "equal":
        raw_costs = torch.ones(len(expert_names), dtype=torch.float32)
        source = "equal unit costs"
    elif cost_mode == "configured":
        schedule = dict(configured_cost_schedule or {})
        raw_costs = torch.tensor(
            [float(schedule.get(name, 1.0)) for name in expert_names],
            dtype=torch.float32,
        )
        source = "router_experiment_config.costarts_subset_cost_schedule"
    else:
        if cost_file is None:
            raise ValueError("cost_mode='latency' requires --cost-file")
        schedule = _read_cost_file(cost_file)
        provided = [float(value) for value in schedule.values() if float(value) > 0]
        fallback = sum(provided) / len(provided) if provided else 1.0
        raw_costs = torch.tensor(
            [float(schedule.get(name, fallback)) for name in expert_names],
            dtype=torch.float32,
        )
        source = str(cost_file)
    if torch.any(~torch.isfinite(raw_costs)):
        raise ValueError("expert costs must be finite")
    if torch.any(raw_costs < 0):
        raise ValueError("expert costs must be non-negative")
    if torch.all(raw_costs == 0):
        normalized_costs = torch.zeros_like(raw_costs)
        mean_cost = 0.0
    else:
        mean_cost = float(raw_costs[raw_costs > 0].mean())
        normalized_costs = raw_costs / max(mean_cost, 1e-12)
    metadata = {
        "cost_mode": cost_mode,
        "cost_source": source,
        "raw_expert_costs": {
            str(name): float(raw_costs[index]) for index, name in enumerate(expert_names)
        },
        "normalized_expert_costs": {
            str(name): float(normalized_costs[index]) for index, name in enumerate(expert_names)
        },
        "mean_positive_raw_cost": mean_cost,
    }
    return raw_costs, normalized_costs, metadata


def build_cost_aware_targets(
    batch: Mapping[str, torch.Tensor],
    *,
    cost_coefficient: float,
    normalized_expert_costs: torch.Tensor,
    utility_finalizer: str = "best_single",
) -> tuple[torch.Tensor, torch.Tensor]:
    if cost_coefficient < 0:
        raise ValueError("cost_coefficient must be non-negative")
    if utility_finalizer not in UTILITY_FINALIZERS:
        raise ValueError(f"utility_finalizer must be one of {sorted(UTILITY_FINALIZERS)}")
    queried_mask = batch["queried_mask"].to(torch.bool)
    remaining_mask = batch["remaining_mask"].to(torch.bool)
    valid_action_mask = batch["valid_action_mask"].to(torch.bool)
    expert_errors = batch["true_expert_error_vector"].to(torch.float32)
    marginal_key = (
        "marginal_gain_equal_queried_average"
        if utility_finalizer == "equal_average"
        else "marginal_gain_best_queried_oracle"
    )
    marginal_gain = batch[marginal_key].to(torch.float32)
    batch_size, num_experts = expert_errors.shape
    assert tuple(queried_mask.shape) == (batch_size, num_experts)
    assert tuple(remaining_mask.shape) == (batch_size, num_experts)
    assert tuple(valid_action_mask.shape) == (batch_size, num_experts + 1)
    assert tuple(marginal_gain.shape) == (batch_size, num_experts)
    costs = normalized_expert_costs.to(expert_errors.device, expert_errors.dtype).view(1, -1)
    assert tuple(costs.shape) == (1, num_experts)

    has_queried = queried_mask.any(dim=1)
    empty_state_utility = -expert_errors - float(cost_coefficient) * costs
    non_empty_utility = marginal_gain - float(cost_coefficient) * costs
    utility = torch.where(has_queried.view(-1, 1), non_empty_utility, empty_state_utility)

    valid_expert_mask = remaining_mask & valid_action_mask[:, :num_experts]
    utility = utility.masked_fill(~valid_expert_mask, float("-inf"))
    best_utility, best_expert_action = utility.max(dim=1)
    stop_index = num_experts
    stop_valid = valid_action_mask[:, stop_index]
    no_remaining = ~valid_expert_mask.any(dim=1)
    stop_is_best = stop_valid & (no_remaining | (best_utility <= 0.0))
    optimal_next_action = torch.where(
        stop_is_best,
        torch.full_like(best_expert_action, stop_index),
        best_expert_action,
    )
    if torch.any(~valid_action_mask.gather(1, optimal_next_action.view(-1, 1)).squeeze(1)):
        raise AssertionError("dynamic target construction produced an invalid action")
    return utility, optimal_next_action.to(torch.long)


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
            "remaining_mask": self.cache["remaining_mask"][index],
            "marginal_gain_best_queried_oracle": self.cache["marginal_gain_best_queried_oracle"][index],
            "marginal_gain_equal_queried_average": self.cache["marginal_gain_equal_queried_average"][index],
            "cost_adjusted_utility": self.cache["cost_adjusted_utility"][index],
            "optimal_next_action": self.cache["optimal_next_action"][index],
            "valid_action_mask": self.cache["valid_action_mask"][index],
            "pairwise_labels_queried": self.cache["pairwise_labels_queried"][index],
            "pairwise_labels_remaining": self.cache["pairwise_labels_remaining"][index],
            "subset_size": self.cache["subset_size"][index],
            "sample_index": self.cache["sample_index"][index],
            "source_row": self.cache["source_row"][index],
        }


class RouterDCFirstQueryModule(nn.Module):
    """RouterDC-compatible history-only selector used only for COSTARTS first query."""

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

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        query_embedding = F.normalize(self.encode(history), p=2, dim=-1)
        expert_vectors = F.normalize(self.expert_embeddings, p=2, dim=-1)
        similarities = query_embedding @ expert_vectors.T
        logits = similarities / max(self.router_temperature, 1e-12)
        assert tuple(logits.shape) == (history.shape[0], self.num_experts)
        return logits

    def config_dict(self) -> dict[str, Any]:
        return {
            "router_type": "routerdc_hard",
            "input_len": self.input_len,
            "num_features": self.num_features,
            "num_experts": self.num_experts,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "router_temperature": self.router_temperature,
        }


def inspect_routerdc_first_query_checkpoint(
    checkpoint_path: Union[str, Path],
    *,
    expert_names: Sequence[str],
    input_len: int,
    num_features: int,
    num_experts: int,
    embedding_dim: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"RouterDC first-query checkpoint does not exist: {checkpoint_path}")
    checkpoint = _load_torch(checkpoint_path)
    checkpoint_experts = tuple(checkpoint.get("selected_expert_names") or checkpoint.get("expert_names") or ())
    expected_experts = tuple(str(name) for name in expert_names)
    if checkpoint_experts != expected_experts:
        raise ValueError(
            "RouterDC checkpoint expert order mismatch. "
            f"checkpoint={checkpoint_experts}, costarts={expected_experts}"
        )
    router_config = dict(checkpoint.get("router_config") or {})
    required = {
        "input_len": int(input_len),
        "num_features": int(num_features),
        "num_experts": int(num_experts),
        "embedding_dim": int(embedding_dim),
    }
    for key, expected in required.items():
        observed = int(router_config.get(key, -1))
        if observed != expected:
            raise ValueError(
                f"RouterDC checkpoint {key} mismatch: checkpoint={observed}, costarts={expected}"
            )
    state_dict = checkpoint.get("router_state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("RouterDC checkpoint is missing router_state_dict")
    expected_embedding_shape = (int(num_experts), int(embedding_dim))
    actual_embedding_shape = tuple(state_dict.get("expert_embeddings", torch.empty(0)).shape)
    if actual_embedding_shape != expected_embedding_shape:
        raise ValueError(
            "RouterDC expert embedding shape mismatch: "
            f"checkpoint={actual_embedding_shape}, expected={expected_embedding_shape}"
        )
    module_config = {
        "input_len": int(router_config["input_len"]),
        "num_features": int(router_config["num_features"]),
        "num_experts": int(router_config["num_experts"]),
        "embedding_dim": int(router_config["embedding_dim"]),
        "hidden_dim": int(router_config.get("hidden_dim", embedding_dim)),
        "dropout": float(router_config.get("dropout", 0.1)),
        "router_temperature": float(router_config.get("router_temperature", 1.0)),
        "router_type": str(router_config.get("router_type", "routerdc_hard")),
    }
    report = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "selected_expert_names": list(checkpoint_experts),
        "router_config": module_config,
        "state_dict_shapes": {key: list(value.shape) for key, value in state_dict.items()},
        "expert_embeddings_shape": list(actual_embedding_shape),
        "normalization": "L2-normalized history query and expert embeddings with cosine similarity",
        "compatible_with_costarts_expert_order": True,
    }
    return module_config, report


def load_routerdc_first_query_weights(
    router: "SubsetUtilityCOSTARTSRouter",
    checkpoint_path: Union[str, Path],
) -> None:
    if router.routerdc_first_query is None:
        raise ValueError("router has no RouterDC first-query module to load")
    checkpoint = _load_torch(checkpoint_path)
    state_dict = checkpoint["router_state_dict"]
    router.routerdc_first_query.load_state_dict(state_dict)
    if router.routerdc_reference_first_query is not None:
        router.routerdc_reference_first_query.load_state_dict(state_dict)
        router.routerdc_reference_first_query.requires_grad_(False)
        router.routerdc_reference_first_query.eval()
    if router.first_query_initialization == "routerdc_frozen":
        router.routerdc_first_query.requires_grad_(False)
        router.routerdc_first_query.eval()


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
        first_query_head_type: str = "shared",
        first_query_initialization: str = "random",
        routerdc_config: Optional[Mapping[str, Any]] = None,
        routerdc_consistency_weight: float = 0.0,
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
        if first_query_head_type not in FIRST_QUERY_HEADS:
            raise ValueError(f"first_query_head_type must be one of {sorted(FIRST_QUERY_HEADS)}")
        if first_query_initialization not in FIRST_QUERY_INITIALIZATIONS:
            raise ValueError(f"first_query_initialization must be one of {sorted(FIRST_QUERY_INITIALIZATIONS)}")
        if routerdc_consistency_weight < 0:
            raise ValueError("routerdc_consistency_weight must be non-negative")
        self.use_expert_embeddings = bool(use_expert_embeddings)
        self.history_encoder_type = str(history_encoder_type)
        self.action_head_type = str(action_head_type)
        self.first_query_head_type = str(first_query_head_type)
        self.first_query_initialization = str(first_query_initialization)
        self.routerdc_config = dict(routerdc_config or {}) if routerdc_config is not None else None
        self.routerdc_consistency_weight = float(routerdc_consistency_weight)
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
        self.first_query_head = (
            nn.Linear(embedding_dim, num_experts)
            if self.first_query_head_type == "separate"
            else None
        )
        self.routerdc_first_query: Optional[RouterDCFirstQueryModule]
        self.routerdc_reference_first_query: Optional[RouterDCFirstQueryModule]
        if self.first_query_initialization == "random":
            self.routerdc_first_query = None
            self.routerdc_reference_first_query = None
        else:
            if self.routerdc_config is None:
                raise ValueError("RouterDC first-query initialization requires routerdc_config")
            self.routerdc_first_query = RouterDCFirstQueryModule(**self.routerdc_config)
            self.routerdc_reference_first_query = (
                deepcopy(self.routerdc_first_query)
                if self.routerdc_consistency_weight > 0
                else None
            )
            if self.routerdc_reference_first_query is not None:
                self.routerdc_reference_first_query.requires_grad_(False)
                self.routerdc_reference_first_query.eval()
            if self.first_query_initialization == "routerdc_frozen":
                self.routerdc_first_query.requires_grad_(False)
                self.routerdc_first_query.eval()

    def train(self, mode: bool = True) -> "SubsetUtilityCOSTARTSRouter":
        super().train(mode)
        if self.routerdc_first_query is not None and self.first_query_initialization == "routerdc_frozen":
            self.routerdc_first_query.eval()
        if self.routerdc_reference_first_query is not None:
            self.routerdc_reference_first_query.eval()
        return self

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
        routerdc_reference_logits = None
        if self.routerdc_first_query is not None:
            first_query_logits = self.routerdc_first_query(history)
            if self.routerdc_reference_first_query is not None:
                with torch.no_grad():
                    routerdc_reference_logits = self.routerdc_reference_first_query(history)
        else:
            first_query_logits = (
                self.first_query_head(representation)
                if self.first_query_head is not None
                else action_logits[:, : self.num_experts]
            )
        batch_size = history.shape[0]
        assert tuple(action_logits.shape) == (batch_size, self.num_experts + 1)
        assert tuple(first_query_logits.shape) == (batch_size, self.num_experts)
        assert tuple(utility_prediction.shape) == (batch_size, self.num_experts)
        assert tuple(expert_score.shape) == (batch_size, self.num_experts)
        assert tuple(mix_logits.shape) == (batch_size, self.num_experts)
        outputs = {
            "representation": representation,
            "action_logits": action_logits,
            "first_query_logits": first_query_logits,
            "utility_prediction": utility_prediction,
            "expert_score": expert_score,
            "mix_logits": mix_logits,
        }
        if routerdc_reference_logits is not None:
            outputs["routerdc_reference_logits"] = routerdc_reference_logits
        return outputs

    def config_dict(self) -> dict[str, Any]:
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
            "first_query_head_type": self.first_query_head_type,
            "first_query_initialization": self.first_query_initialization,
            "routerdc_config": self.routerdc_config,
            "routerdc_consistency_weight": self.routerdc_consistency_weight,
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


def first_query_soft_targets(expert_errors: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("first_query_temperature must be positive")
    targets = torch.softmax(-expert_errors.to(torch.float32) / float(temperature), dim=-1)
    if not torch.allclose(targets.sum(dim=-1), torch.ones(targets.shape[0], device=targets.device), atol=1e-5):
        raise AssertionError("first-query soft targets must sum to one")
    return targets


def first_query_regret_loss(logits: torch.Tensor, expert_errors: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=-1)
    expected_error = (probabilities * expert_errors.to(logits.dtype)).sum(dim=-1)
    oracle_error = expert_errors.min(dim=-1).values.to(logits.dtype)
    return (expected_error - oracle_error).mean()


def _entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    log_probabilities = torch.log_softmax(logits, dim=-1)
    probabilities = log_probabilities.exp()
    return -(probabilities * log_probabilities).sum(dim=-1)


def _pairwise_loss(scores: torch.Tensor, queried_labels: torch.Tensor, remaining_labels: torch.Tensor) -> torch.Tensor:
    labels = torch.cat((queried_labels, remaining_labels), dim=0).to(torch.float32)
    repeated_scores = scores.repeat(2, 1)
    score_diff = repeated_scores.unsqueeze(2) - repeated_scores.unsqueeze(1)
    valid = labels != 0
    if not torch.any(valid):
        return scores.sum() * 0.0
    return F.softplus(-labels[valid] * score_diff[valid]).mean()


def _active_marginal_gain(batch: Mapping[str, torch.Tensor], utility_finalizer: str) -> torch.Tensor:
    if utility_finalizer == "equal_average":
        return batch["marginal_gain_equal_queried_average"].to(torch.float32)
    if utility_finalizer == "best_single":
        return batch["marginal_gain_best_queried_oracle"].to(torch.float32)
    raise ValueError(f"utility_finalizer must be one of {sorted(UTILITY_FINALIZERS)}")


def _pairwise_labels_from_values(values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    valid = valid_mask.to(torch.bool) & torch.isfinite(values)
    diff = values.unsqueeze(2) - values.unsqueeze(1)
    labels = torch.sign(diff).to(torch.int8)
    labels = labels.masked_fill(~(valid[:, :, None] & valid[:, None, :]), 0)
    return labels


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
    *,
    cost_coefficient: Optional[float] = None,
    normalized_expert_costs: Optional[torch.Tensor] = None,
    utility_finalizer: str = "best_single",
    first_query_target: str = "hard",
    first_query_temperature: float = 0.02,
    first_query_loss_weight: float = 1.0,
    first_query_regret_loss_weight: float = 0.0,
    routerdc_consistency_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    if first_query_target not in FIRST_QUERY_TARGETS:
        raise ValueError(f"first_query_target must be one of {sorted(FIRST_QUERY_TARGETS)}")
    if routerdc_consistency_weight < 0:
        raise ValueError("routerdc_consistency_weight must be non-negative")
    if normalized_expert_costs is None or cost_coefficient is None:
        target_utility = batch["cost_adjusted_utility"].to(outputs["utility_prediction"].dtype)
        target_action = batch["optimal_next_action"].to(torch.long)
    else:
        target_utility, target_action = build_cost_aware_targets(
            batch,
            cost_coefficient=float(cost_coefficient),
            normalized_expert_costs=normalized_expert_costs,
            utility_finalizer=utility_finalizer,
        )
        target_utility = target_utility.to(outputs["utility_prediction"].dtype)
    masked_logits = _masked_action_logits(outputs["action_logits"], batch["valid_action_mask"])
    empty_mask = batch["subset_size"].to(torch.long) == 0
    non_empty_mask = ~empty_mask
    if torch.any(non_empty_mask):
        action_loss = F.cross_entropy(masked_logits[non_empty_mask], target_action[non_empty_mask])
    else:
        action_loss = masked_logits.sum() * 0.0
    first_query_logits = outputs.get("first_query_logits", outputs["action_logits"][:, : outputs["utility_prediction"].shape[1]])
    if torch.any(empty_mask):
        empty_errors = batch["true_expert_error_vector"][empty_mask].to(first_query_logits.dtype)
        empty_logits = first_query_logits[empty_mask]
        hard_first_targets = target_action[empty_mask]
        soft_targets = first_query_soft_targets(empty_errors, first_query_temperature).to(empty_logits.dtype)
        soft_ce = -(soft_targets * torch.log_softmax(empty_logits, dim=-1)).sum(dim=-1).mean()
        hard_ce = F.cross_entropy(empty_logits, hard_first_targets)
        base_first_query_loss = soft_ce if first_query_target == "soft" else hard_ce
        regret_loss = first_query_regret_loss(empty_logits, empty_errors)
        first_query_loss = (
            float(first_query_loss_weight) * base_first_query_loss
            + float(first_query_regret_loss_weight) * regret_loss
        )
        first_prediction = torch.argmax(empty_logits, dim=-1)
        first_hard_accuracy = (first_prediction == hard_first_targets).to(torch.float32).mean()
        first_entropy = _entropy_from_logits(empty_logits).mean()
    else:
        soft_ce = first_query_logits.sum() * 0.0
        hard_ce = first_query_logits.sum() * 0.0
        regret_loss = first_query_logits.sum() * 0.0
        first_query_loss = first_query_logits.sum() * 0.0
        first_hard_accuracy = first_query_logits.sum() * 0.0
        first_entropy = first_query_logits.sum() * 0.0
    if routerdc_consistency_weight > 0 and torch.any(empty_mask) and "routerdc_reference_logits" in outputs:
        reference_logits = outputs["routerdc_reference_logits"][empty_mask].detach()
        reference_probabilities = torch.softmax(reference_logits, dim=-1)
        routerdc_consistency_loss = (
            reference_probabilities
            * (torch.log_softmax(reference_logits, dim=-1) - torch.log_softmax(first_query_logits[empty_mask], dim=-1))
        ).sum(dim=-1).mean()
    else:
        routerdc_consistency_loss = first_query_logits.sum() * 0.0
    utility_loss = _utility_loss(
        outputs["utility_prediction"],
        target_utility,
        batch["valid_action_mask"],
    )
    marginal_values = _active_marginal_gain(batch, utility_finalizer).to(outputs["expert_score"].device)
    remaining_labels = _pairwise_labels_from_values(
        marginal_values,
        batch["valid_action_mask"][:, : outputs["expert_score"].shape[1]],
    )
    queried_labels = torch.zeros_like(remaining_labels)
    pairwise_loss = _pairwise_loss(outputs["expert_score"], queried_labels, remaining_labels)
    mix_loss = _mix_loss(outputs, batch)
    total = (
        float(loss_weights.get("action", 1.0)) * action_loss
        + first_query_loss
        + float(loss_weights.get("utility", 1.0)) * utility_loss
        + float(loss_weights.get("pairwise", 0.2)) * pairwise_loss
        + float(loss_weights.get("mix", 0.0)) * mix_loss
        + float(routerdc_consistency_weight) * routerdc_consistency_loss
    )
    predicted_action = torch.argmax(masked_logits, dim=-1)
    stop_index = outputs["action_logits"].shape[-1] - 1
    return total, {
        "total_loss": float(total.detach().cpu()),
        "action_loss": float(action_loss.detach().cpu()),
        "later_action_loss": float(action_loss.detach().cpu()),
        "first_query_loss": float(first_query_loss.detach().cpu()),
        "first_query_soft_target_cross_entropy": float(soft_ce.detach().cpu()),
        "first_query_hard_cross_entropy": float(hard_ce.detach().cpu()),
        "first_query_regret_loss": float(regret_loss.detach().cpu()),
        "routerdc_consistency_loss": float(routerdc_consistency_loss.detach().cpu()),
        "utility_loss": float(utility_loss.detach().cpu()),
        "pairwise_loss": float(pairwise_loss.detach().cpu()),
        "mix_loss": float(mix_loss.detach().cpu()),
        "action_accuracy": float((predicted_action == target_action).to(torch.float32).mean().detach().cpu()),
        "first_query_hard_accuracy": float(first_hard_accuracy.detach().cpu()),
        "mean_predicted_first_query_entropy": float(first_entropy.detach().cpu()),
        "stop_frequency": float((predicted_action == stop_index).to(torch.float32).mean().detach().cpu()),
        "target_stop_frequency": float((target_action == stop_index).to(torch.float32).mean().detach().cpu()),
        "average_predicted_utility": float(outputs["utility_prediction"].detach().mean().cpu()),
        "empty_states_sampled": int(empty_mask.detach().cpu().sum()),
        "non_empty_states_sampled": int(non_empty_mask.detach().cpu().sum()),
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
        "remaining_mask",
        "marginal_gain_best_queried_oracle",
        "marginal_gain_equal_queried_average",
        "valid_action_mask",
        "subset_size",
    )
    return {key: cache[key][index].to(device) for key in keys}


def _equal_average_queried_forecasts(batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    queried_ids = batch["queried_expert_ids"]
    valid_slots = queried_ids >= 0
    forecasts = batch["queried_expert_forecasts"]
    if torch.any(valid_slots.sum(dim=1) == 0):
        raise AssertionError("equal-average deployment requires at least one queried expert")
    weights = valid_slots.to(forecasts.dtype)
    return (forecasts * weights[:, :, None, None]).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)[:, None, None]


def _masked_mae_mse(prediction: torch.Tensor, target: torch.Tensor, target_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mask = target_mask.to(prediction.dtype)
    denominator = mask.sum(dim=(1, 2)).clamp_min(1.0)
    mae = (torch.abs(prediction - target) * mask).sum(dim=(1, 2)) / denominator
    mse = (((prediction - target) ** 2) * mask).sum(dim=(1, 2)) / denominator
    return mae, mse


@torch.no_grad()
def evaluate_first_query_policy(
    router: SubsetUtilityCOSTARTSRouter,
    cache: Mapping[str, Any],
    *,
    batch_size: int,
    device: Union[str, torch.device],
) -> dict[str, Any]:
    device = torch.device(device)
    router.eval()
    validate_costarts_subset_states(cache)
    empty_indices = torch.nonzero(cache["subset_size"] == 0, as_tuple=False).flatten()
    num_experts = int(cache["num_experts"])
    selected_rows = []
    top2_rows = []
    entropy_rows = []
    expected_error_rows = []
    error_rows = []
    for offset in range(0, int(empty_indices.numel()), batch_size):
        state_indices = empty_indices[offset : offset + batch_size].tolist()
        batch = _state_batch(cache, state_indices, device)
        outputs = router(
            batch["history"],
            batch["queried_mask"],
            batch["queried_expert_ids"],
            batch["queried_expert_forecasts"],
        )
        logits = outputs["first_query_logits"]
        probabilities = torch.softmax(logits, dim=-1)
        selected_rows.append(torch.argmax(probabilities, dim=-1).detach().cpu())
        top2_rows.append(torch.topk(probabilities, k=min(2, num_experts), dim=-1).indices.detach().cpu())
        entropy_rows.append(_entropy_from_logits(logits).detach().cpu())
        errors = batch["true_expert_error_vector"].to(probabilities.dtype)
        expected_error_rows.append((probabilities * errors).sum(dim=-1).detach().cpu())
        error_rows.append(errors.detach().cpu())
    selected = torch.cat(selected_rows, dim=0)
    top2 = torch.cat(top2_rows, dim=0)
    entropy = torch.cat(entropy_rows, dim=0)
    expected_error = torch.cat(expected_error_rows, dim=0)
    errors = torch.cat(error_rows, dim=0)
    oracle = torch.argmin(errors, dim=1)
    oracle_error = errors.min(dim=1).values
    selected_error = errors.gather(1, selected.view(-1, 1)).squeeze(1)
    top2_coverage = torch.tensor(
        [float(int(oracle[index]) in top2[index].tolist()) for index in range(top2.shape[0])],
        dtype=torch.float32,
    )
    best_fixed = int(torch.argmin(errors.mean(dim=0)))
    return {
        "first_query_oracle_match": float((selected == oracle).to(torch.float32).mean()),
        "first_query_top_two_ranking_coverage": float(top2_coverage.mean()),
        "first_query_selected_mae": float(selected_error.mean()),
        "first_query_regret_to_oracle": float((selected_error - oracle_error).mean()),
        "first_query_regret_to_best_fixed": float(selected_error.mean() - errors[:, best_fixed].mean()),
        "first_query_expected_error": float(expected_error.mean()),
        "first_query_expected_regret": float((expected_error - oracle_error).mean()),
        "first_query_entropy": float(entropy.mean()),
        "first_query_selection_counts": {
            str(index): int(value)
            for index, value in enumerate(torch.bincount(selected, minlength=num_experts).tolist())
        },
    }


@torch.no_grad()
def evaluate_deployable_inference(
    router: SubsetUtilityCOSTARTSRouter,
    cache: Mapping[str, Any],
    *,
    batch_size: int,
    device: Union[str, torch.device],
    cost_coefficient: float = 0.0,
    normalized_expert_costs: Optional[torch.Tensor] = None,
    raw_expert_costs: Optional[torch.Tensor] = None,
    utility_finalizer: str = "best_single",
    deployment_finalizer: str = "best_reranked",
    debug: bool = False,
) -> dict[str, Any]:
    device = torch.device(device)
    router.eval()
    validate_costarts_subset_states(cache)
    if deployment_finalizer not in DEPLOYMENT_FINALIZERS:
        raise ValueError(f"deployment_finalizer must be one of {sorted(DEPLOYMENT_FINALIZERS)}")
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
    false_stop = 0
    false_continue = 0
    predicted_stop_count = 0
    oracle_stop_count = 0
    stop_decision_count = 0
    if normalized_expert_costs is None:
        normalized_costs_cpu = torch.zeros(num_experts, dtype=torch.float32)
    else:
        normalized_costs_cpu = normalized_expert_costs.detach().cpu().to(torch.float32)
    if raw_expert_costs is None:
        raw_costs_cpu = normalized_costs_cpu.clone()
    else:
        raw_costs_cpu = raw_expert_costs.detach().cpu().to(torch.float32)

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
            subset_size = batch["subset_size"].to(torch.long)
            empty_mask = subset_size == 0
            actions_device = torch.argmax(masked_logits, dim=-1)
            if torch.any(empty_mask):
                first_logits = outputs["first_query_logits"].masked_fill(
                    ~batch["valid_action_mask"][:, :num_experts].to(torch.bool),
                    -1e9,
                )
                actions_device[empty_mask] = torch.argmax(first_logits[empty_mask], dim=-1)
            actions = actions_device.detach().cpu()
            if normalized_expert_costs is not None:
                _, target_actions = build_cost_aware_targets(
                    batch,
                    cost_coefficient=cost_coefficient,
                    normalized_expert_costs=normalized_expert_costs.to(device),
                    utility_finalizer=utility_finalizer,
                )
                target_actions = target_actions.detach().cpu()
                stop_valid = batch["valid_action_mask"][:, stop_index].detach().cpu().to(torch.bool)
                predicted_stop = actions == stop_index
                oracle_stop = target_actions == stop_index
                false_stop += int((predicted_stop & ~oracle_stop & stop_valid).sum())
                false_continue += int((~predicted_stop & oracle_stop & stop_valid).sum())
                predicted_stop_count += int((predicted_stop & stop_valid).sum())
                oracle_stop_count += int((oracle_stop & stop_valid).sum())
                stop_decision_count += int(stop_valid.sum())
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
            action = int(torch.argmax(outputs["first_query_logits"][0, :num_experts]).detach().cpu())
            query_history[sample_index].append(action)
            masks[sample_index] |= 1 << action
        if not done[sample_index]:
            stop_counts[len(query_history[sample_index])] += 1

    final_state_indices = [lookup[row][masks[row]] for row in range(num_windows)]
    selected = torch.empty(num_windows, dtype=torch.long)
    prediction_chunks = []
    mae_chunks = []
    mse_chunks = []
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
        selected_batch = torch.argmax(selected_scores, dim=-1)
        selected[offset : offset + len(rows)] = selected_batch
        if deployment_finalizer == "equal_average":
            prediction = _equal_average_queried_forecasts(batch)
        else:
            queried_ids = batch["queried_expert_ids"]
            positions = (queried_ids == selected_batch.to(device)[:, None]).to(torch.float32).argmax(dim=1)
            prediction = batch["queried_expert_forecasts"][torch.arange(len(rows), device=device), positions]
        mae_chunk, mse_chunk = _masked_mae_mse(prediction, batch["true_targets"], batch["target_mask"])
        prediction_chunks.append(prediction.detach().cpu())
        mae_chunks.append(mae_chunk.detach().cpu())
        mse_chunks.append(mse_chunk.detach().cpu())

    source_errors = torch.empty(num_windows, num_experts, dtype=torch.float32)
    for row in range(num_windows):
        source_errors[row] = cache["true_expert_error_vector"][lookup[row][0]]
    selected_mae = torch.cat(mae_chunks, dim=0)
    selected_mse = torch.cat(mse_chunks, dim=0)
    oracle_best = torch.argmin(source_errors, dim=1)
    oracle_mae = source_errors.min(dim=1).values
    avg_queries = sum(len(history) for history in query_history) / max(num_windows, 1)
    first_query = torch.tensor([history[0] for history in query_history], dtype=torch.long)
    first_query_oracle_match = (first_query == oracle_best).to(torch.float32).mean()
    top_two_oracle_coverage = torch.tensor(
        [
            float(int(oracle_best[index]) in history[:2])
            for index, history in enumerate(query_history)
        ],
        dtype=torch.float32,
    ).mean()
    normalized_query_costs = [
        float(normalized_costs_cpu[torch.tensor(history, dtype=torch.long)].sum()) if history else 0.0
        for history in query_history
    ]
    raw_query_costs = [
        float(raw_costs_cpu[torch.tensor(history, dtype=torch.long)].sum()) if history else 0.0
        for history in query_history
    ]
    avg_normalized_cost = sum(normalized_query_costs) / max(num_windows, 1)
    avg_raw_cost = sum(raw_query_costs) / max(num_windows, 1)
    validation_objective = float(selected_mae.mean()) + float(cost_coefficient) * avg_normalized_cost
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

    first_query_metrics = evaluate_first_query_policy(
        router,
        cache,
        batch_size=batch_size,
        device=device,
    )

    return {
        "validation_mae": float(selected_mae.mean()),
        "validation_mse": float(selected_mse.mean()),
        "validation_oracle_mae": float(oracle_mae.mean()),
        "validation_regret_to_oracle": float(selected_mae.mean() - oracle_mae.mean()),
        "validation_objective": float(validation_objective),
        "action_accuracy": float((selected == oracle_best).to(torch.float32).mean()),
        "first_query_oracle_match": float(first_query_oracle_match),
        "top_two_oracle_coverage": float(top_two_oracle_coverage),
        "average_experts_selected": float(avg_queries),
        "average_normalized_query_cost": float(avg_normalized_cost),
        "average_raw_query_cost": float(avg_raw_cost),
        "false_stop_rate": float(false_stop / max(stop_decision_count, 1)),
        "false_continue_rate": float(false_continue / max(stop_decision_count, 1)),
        "stop_precision": float((predicted_stop_count - false_stop) / max(predicted_stop_count, 1)),
        "stop_recall": float((oracle_stop_count - false_continue) / max(oracle_stop_count, 1)),
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
        **first_query_metrics,
    }


@torch.no_grad()
def evaluate_subset_state_losses(
    router: SubsetUtilityCOSTARTSRouter,
    cache: Mapping[str, Any],
    *,
    batch_size: int,
    device: Union[str, torch.device],
    loss_weights: Mapping[str, float],
    cost_coefficient: Optional[float] = None,
    normalized_expert_costs: Optional[torch.Tensor] = None,
    utility_finalizer: str = "best_single",
    first_query_target: str = "hard",
    first_query_temperature: float = 0.02,
    first_query_loss_weight: float = 1.0,
    first_query_regret_loss_weight: float = 0.0,
    routerdc_consistency_weight: float = 0.0,
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
        "later_action_loss": 0.0,
        "first_query_loss": 0.0,
        "first_query_soft_target_cross_entropy": 0.0,
        "first_query_hard_cross_entropy": 0.0,
        "first_query_regret_loss": 0.0,
        "routerdc_consistency_loss": 0.0,
        "first_query_hard_accuracy": 0.0,
        "mean_predicted_first_query_entropy": 0.0,
        "action_accuracy": 0.0,
        "stop_frequency": 0.0,
        "target_stop_frequency": 0.0,
        "average_predicted_utility": 0.0,
        "empty_states_sampled": 0.0,
        "non_empty_states_sampled": 0.0,
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
        _, parts = subset_utility_losses(
            outputs,
            batch,
            loss_weights,
            cost_coefficient=cost_coefficient,
            normalized_expert_costs=normalized_expert_costs,
            utility_finalizer=utility_finalizer,
            first_query_target=first_query_target,
            first_query_temperature=first_query_temperature,
            first_query_loss_weight=first_query_loss_weight,
            first_query_regret_loss_weight=first_query_regret_loss_weight,
            routerdc_consistency_weight=routerdc_consistency_weight,
        )
        batch_size_actual = batch["history"].shape[0]
        for key in totals:
            if key in {"empty_states_sampled", "non_empty_states_sampled"}:
                totals[key] += parts[key]
            else:
                totals[key] += parts[key] * batch_size_actual
        count += batch_size_actual
    return {
        f"validation_state_{key}": (
            value if key in {"empty_states_sampled", "non_empty_states_sampled"} else value / max(count, 1)
        )
        for key, value in totals.items()
    }


def _write_curves(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _cache_training_batch(cache: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    keys = (
        "queried_mask",
        "remaining_mask",
        "true_expert_error_vector",
        "marginal_gain_best_queried_oracle",
        "marginal_gain_equal_queried_average",
        "valid_action_mask",
    )
    return {key: cache[key].to(device) for key in keys}


def _build_action_balanced_sampler(
    cache: Mapping[str, Any],
    *,
    cost_coefficient: float,
    normalized_expert_costs: torch.Tensor,
    utility_finalizer: str,
) -> WeightedRandomSampler:
    batch = _cache_training_batch(cache, torch.device("cpu"))
    _, actions = build_cost_aware_targets(
        batch,
        cost_coefficient=cost_coefficient,
        normalized_expert_costs=normalized_expert_costs.cpu(),
        utility_finalizer=utility_finalizer,
    )
    stop_index = int(cache["stop_action_index"])
    categories = (actions == stop_index).to(torch.long)
    counts = torch.bincount(categories, minlength=2).to(torch.float32).clamp_min(1.0)
    weights = (1.0 / counts)[categories]
    print(
        "Action-balanced sampling target counts: "
        f"CONTINUE={int(counts[0])}, STOP={int(counts[1])}"
    )
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def _build_first_query_ratio_sampler(
    cache: Mapping[str, Any],
    *,
    first_query_sampling_ratio: float,
    cost_coefficient: float,
    normalized_expert_costs: torch.Tensor,
    utility_finalizer: str,
) -> WeightedRandomSampler:
    if not (0.0 < first_query_sampling_ratio < 1.0):
        raise ValueError("first_query_sampling_ratio must be between 0 and 1 when enabled")
    batch = _cache_training_batch(cache, torch.device("cpu"))
    _, actions = build_cost_aware_targets(
        batch,
        cost_coefficient=cost_coefficient,
        normalized_expert_costs=normalized_expert_costs.cpu(),
        utility_finalizer=utility_finalizer,
    )
    subset_size = cache["subset_size"].to(torch.long)
    stop_index = int(cache["stop_action_index"])
    empty = subset_size == 0
    non_empty_continue = (subset_size > 0) & (actions != stop_index)
    non_empty_stop = (subset_size > 0) & (actions == stop_index)
    weights = torch.zeros(int(cache["num_states"]), dtype=torch.float32)
    weights[empty] = float(first_query_sampling_ratio) / max(int(empty.sum()), 1)
    remaining_ratio = 1.0 - float(first_query_sampling_ratio)
    weights[non_empty_continue] = 0.5 * remaining_ratio / max(int(non_empty_continue.sum()), 1)
    weights[non_empty_stop] = 0.5 * remaining_ratio / max(int(non_empty_stop.sum()), 1)
    print(
        "First-query ratio sampling target counts: "
        f"EMPTY={int(empty.sum())}, "
        f"NONEMPTY_CONTINUE={int(non_empty_continue.sum())}, "
        f"NONEMPTY_STOP={int(non_empty_stop.sum())}, "
        f"target_empty_ratio={first_query_sampling_ratio:.3f}"
    )
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def _save_checkpoint(
    path: Path,
    router: SubsetUtilityCOSTARTSRouter,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    metrics: Mapping[str, Any],
    training_config: SubsetUtilityTrainingConfig,
    expert_names: Sequence[str],
    raw_expert_costs: torch.Tensor,
    normalized_expert_costs: torch.Tensor,
    cost_metadata: Mapping[str, Any],
    routerdc_report: Optional[Mapping[str, Any]] = None,
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
            "raw_expert_costs": raw_expert_costs.detach().cpu(),
            "normalized_expert_costs": normalized_expert_costs.detach().cpu(),
            "cost_metadata": dict(cost_metadata),
            "routerdc_first_query_report": dict(routerdc_report or {}),
            "router_type": "subset_utility_costarts",
            "experts_loaded": False,
            "experts_updated": False,
            "model_selection": training_config.selection_metric,
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
    if training_config.selection_metric not in {"mae", "cost_aware_objective"}:
        raise ValueError("selection_metric must be 'mae' or 'cost_aware_objective'")
    if training_config.utility_finalizer not in UTILITY_FINALIZERS:
        raise ValueError(f"utility_finalizer must be one of {sorted(UTILITY_FINALIZERS)}")
    if training_config.deployment_finalizer not in DEPLOYMENT_FINALIZERS:
        raise ValueError(f"deployment_finalizer must be one of {sorted(DEPLOYMENT_FINALIZERS)}")
    if training_config.state_sampling not in STATE_SAMPLING_MODES:
        raise ValueError(f"state_sampling must be one of {sorted(STATE_SAMPLING_MODES)}")
    if training_config.first_query_target not in FIRST_QUERY_TARGETS:
        raise ValueError(f"first_query_target must be one of {sorted(FIRST_QUERY_TARGETS)}")
    if training_config.first_query_head not in FIRST_QUERY_HEADS:
        raise ValueError(f"first_query_head must be one of {sorted(FIRST_QUERY_HEADS)}")
    if training_config.first_query_initialization not in FIRST_QUERY_INITIALIZATIONS:
        raise ValueError(f"first_query_initialization must be one of {sorted(FIRST_QUERY_INITIALIZATIONS)}")
    if training_config.first_query_temperature <= 0:
        raise ValueError("first_query_temperature must be positive")
    if training_config.routerdc_consistency_weight < 0:
        raise ValueError("routerdc_consistency_weight must be non-negative")
    if not (0.0 <= training_config.first_query_sampling_ratio < 1.0):
        raise ValueError("first_query_sampling_ratio must be in [0, 1)")
    if (
        training_config.utility_finalizer == "equal_average"
        and training_config.deployment_finalizer != "equal_average"
        and not training_config.allow_finalizer_mismatch
    ):
        raise ValueError(
            "utility_finalizer=equal_average requires deployment_finalizer=equal_average "
            "unless --allow-finalizer-mismatch is set for an ablation"
        )

    expert_names = tuple(train_cache["expert_names"])
    routerdc_config = None
    routerdc_report: Optional[dict[str, Any]] = None
    if training_config.first_query_initialization != "random":
        routerdc_config, routerdc_report = inspect_routerdc_first_query_checkpoint(
            training_config.routerdc_checkpoint_path,
            expert_names=expert_names,
            input_len=96,
            num_features=int(train_cache["num_features"]),
            num_experts=int(train_cache["num_experts"]),
            embedding_dim=experiment_config.embedding_dim,
        )
    raw_expert_costs, normalized_expert_costs, cost_metadata = load_and_normalize_expert_costs(
        expert_names,
        cost_mode=training_config.cost_mode,
        cost_file=training_config.cost_file,
        configured_cost_schedule=experiment_config.costarts_subset_cost_schedule,
    )
    if "base_expert_costs" in train_cache and "base_expert_costs" in val_cache:
        if tuple(train_cache["base_expert_costs"].shape) != tuple(val_cache["base_expert_costs"].shape):
            raise ValueError("Train/val base expert cost vectors have different shapes")
    normalized_expert_costs_device = normalized_expert_costs.to(device)
    print("COSTARTS subset utility cost configuration:")
    print(f"  cost_mode: {training_config.cost_mode}")
    print(f"  cost_coefficient: {training_config.cost_coefficient}")
    print(f"  selection_metric: {training_config.selection_metric}")
    print(f"  utility_finalizer: {training_config.utility_finalizer}")
    print(f"  deployment_finalizer: {training_config.deployment_finalizer}")
    print(f"  state_sampling: {training_config.state_sampling}")
    print(f"  first_query_target: {training_config.first_query_target}")
    print(f"  first_query_head: {training_config.first_query_head}")
    print(f"  first_query_sampling_ratio: {training_config.first_query_sampling_ratio}")
    print(f"  first_query_initialization: {training_config.first_query_initialization}")
    print(f"  routerdc_consistency_weight: {training_config.routerdc_consistency_weight}")
    if routerdc_report is not None:
        print(f"  routerdc_checkpoint: {routerdc_report['checkpoint_path']}")
        print(f"  routerdc_epoch: {routerdc_report['checkpoint_epoch']}")
    print(f"  normalized_expert_costs: {cost_metadata['normalized_expert_costs']}")
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
        first_query_head_type=training_config.first_query_head,
        first_query_initialization=training_config.first_query_initialization,
        routerdc_config=routerdc_config,
        routerdc_consistency_weight=training_config.routerdc_consistency_weight,
    ).to(device)
    if training_config.first_query_initialization != "random":
        load_routerdc_first_query_weights(router, training_config.routerdc_checkpoint_path)

    trainable_parameters = [parameter for parameter in router.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("No trainable router parameters are available")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
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
    sampler = None
    shuffle = True
    if training_config.first_query_sampling_ratio > 0:
        sampler = _build_first_query_ratio_sampler(
            train_cache,
            first_query_sampling_ratio=training_config.first_query_sampling_ratio,
            cost_coefficient=training_config.cost_coefficient,
            normalized_expert_costs=normalized_expert_costs,
            utility_finalizer=training_config.utility_finalizer,
        )
        shuffle = False
    elif training_config.state_sampling == "action_balanced":
        sampler = _build_action_balanced_sampler(
            train_cache,
            cost_coefficient=training_config.cost_coefficient,
            normalized_expert_costs=normalized_expert_costs,
            utility_finalizer=training_config.utility_finalizer,
        )
        shuffle = False
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        generator=generator if sampler is None else None,
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

    best_selection_value = math.inf
    best_metrics: dict[str, Any] = {}
    best_epoch = 0
    bad_epochs = 0
    curves = []

    for epoch in range(1, training_config.max_epochs + 1):
        router.train()
        totals = {
            "total_loss": 0.0,
            "action_loss": 0.0,
            "later_action_loss": 0.0,
            "first_query_loss": 0.0,
            "first_query_soft_target_cross_entropy": 0.0,
            "first_query_hard_cross_entropy": 0.0,
            "first_query_regret_loss": 0.0,
            "routerdc_consistency_loss": 0.0,
            "utility_loss": 0.0,
            "pairwise_loss": 0.0,
            "mix_loss": 0.0,
            "action_accuracy": 0.0,
            "first_query_hard_accuracy": 0.0,
            "mean_predicted_first_query_entropy": 0.0,
            "stop_frequency": 0.0,
            "target_stop_frequency": 0.0,
            "average_predicted_utility": 0.0,
            "empty_states_sampled": 0.0,
            "non_empty_states_sampled": 0.0,
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
            total_loss, parts = subset_utility_losses(
                outputs,
                batch,
                loss_weights,
                cost_coefficient=training_config.cost_coefficient,
                normalized_expert_costs=normalized_expert_costs_device,
                utility_finalizer=training_config.utility_finalizer,
                first_query_target=training_config.first_query_target,
                first_query_temperature=training_config.first_query_temperature,
                first_query_loss_weight=training_config.first_query_loss_weight,
                first_query_regret_loss_weight=training_config.first_query_regret_loss_weight,
                routerdc_consistency_weight=training_config.routerdc_consistency_weight,
            )
            total_loss.backward()
            trainable_parameters = [parameter for parameter in router.parameters() if parameter.requires_grad]
            if training_config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable_parameters, training_config.grad_clip_norm)
            optimizer.step()
            batch_size_actual = batch["history"].shape[0]
            for key in totals:
                if key in {"empty_states_sampled", "non_empty_states_sampled"}:
                    totals[key] += parts[key]
                else:
                    totals[key] += parts[key] * batch_size_actual
            seen += batch_size_actual
            if training_config.debug and epoch == 1 and batch_index == 0:
                print("SubsetUtility first training batch")
                print("  history:", tuple(batch["history"].shape))
                print("  queried_mask:", tuple(batch["queried_mask"].shape))
                print("  queried_expert_forecasts:", tuple(batch["queried_expert_forecasts"].shape))
                print("  action_logits:", tuple(outputs["action_logits"].shape))

        train_metrics = {
            key: (
                value if key in {"empty_states_sampled", "non_empty_states_sampled"} else value / max(seen, 1)
            )
            for key, value in totals.items()
        }
        val_deployable = evaluate_deployable_inference(
            router,
            val_cache,
            batch_size=training_config.batch_size,
            device=device,
            cost_coefficient=training_config.cost_coefficient,
            normalized_expert_costs=normalized_expert_costs_device,
            raw_expert_costs=raw_expert_costs,
            utility_finalizer=training_config.utility_finalizer,
            deployment_finalizer=training_config.deployment_finalizer,
            debug=training_config.debug and epoch == 1,
        )
        val_state = evaluate_subset_state_losses(
            router,
            val_cache,
            batch_size=training_config.batch_size,
            device=device,
            loss_weights=loss_weights,
            cost_coefficient=training_config.cost_coefficient,
            normalized_expert_costs=normalized_expert_costs_device,
            utility_finalizer=training_config.utility_finalizer,
            first_query_target=training_config.first_query_target,
            first_query_temperature=training_config.first_query_temperature,
            first_query_loss_weight=training_config.first_query_loss_weight,
            first_query_regret_loss_weight=training_config.first_query_regret_loss_weight,
            routerdc_consistency_weight=training_config.routerdc_consistency_weight,
        )
        selection_key = (
            "validation_mae"
            if training_config.selection_metric == "mae"
            else "validation_objective"
        )
        selection_value = float(val_deployable[selection_key])
        scheduler.step(selection_value)

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
        improved = selection_value < best_selection_value
        if improved:
            best_selection_value = selection_value
            best_metrics = dict(val_deployable)
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
                raw_expert_costs,
                normalized_expert_costs,
                cost_metadata,
                routerdc_report,
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
            raw_expert_costs,
            normalized_expert_costs,
            cost_metadata,
            routerdc_report,
        )
        print(
            f"SubsetUtility epoch {epoch:03d} | "
            f"loss={train_metrics['total_loss']:.6f} "
            f"action={train_metrics['action_loss']:.6f} "
            f"utility={train_metrics['utility_loss']:.6f} "
            f"pairwise={train_metrics['pairwise_loss']:.6f} "
            f"mix={train_metrics['mix_loss']:.6f} "
            f"fq={train_metrics['first_query_loss']:.6f} "
            f"fq_acc={train_metrics['first_query_hard_accuracy']:.3f} "
            f"fq_empty={int(train_metrics['empty_states_sampled'])} "
            f"acc={train_metrics['action_accuracy']:.3f} "
            f"stop={train_metrics['stop_frequency']:.3f} | "
            f"target_stop={train_metrics['target_stop_frequency']:.3f} | "
            f"val_mae={val_deployable['validation_mae']:.6f} "
            f"obj={val_deployable['validation_objective']:.6f} "
            f"regret={val_deployable['validation_regret_to_oracle']:.6f} "
            f"avg_used={val_deployable['average_experts_selected']:.3f} "
            f"avg_cost={val_deployable['average_normalized_query_cost']:.3f} "
            f"top2={val_deployable['better_of_top_two_accuracy']:.3f} "
            f"saved={improved}"
        )
        if bad_epochs >= training_config.patience:
            print(f"SubsetUtility early stopping after epoch {epoch}")
            break

    _write_curves(curves_path, curves)
    summary = {
        "best_epoch": best_epoch,
        "best_selection_metric": training_config.selection_metric,
        "best_selection_metric_value": best_selection_value,
        "best_validation_mae": best_metrics.get("validation_mae"),
        "best_validation_objective": best_metrics.get("validation_objective"),
        "best_validation_metrics": best_metrics,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "training_curves_csv": str(curves_path),
        "training_config": asdict(training_config),
        "router_config": router.config_dict(),
        "expert_names": list(expert_names),
        "cost_metadata": cost_metadata,
        "routerdc_first_query_report": routerdc_report or {},
        "raw_expert_costs": _jsonable(raw_expert_costs),
        "normalized_expert_costs": _jsonable(normalized_expert_costs),
        "train_cache_path": training_config.train_cache_path,
        "val_cache_path": training_config.val_cache_path,
        "created_checkpoints_separate_from_original_costarts": True,
        "original_costarts_checkpoint_dir": "checkpoints/costarts",
        "model_selection": training_config.selection_metric,
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
    parser.add_argument("--cost-mode", choices=("equal", "configured", "latency"), default="equal")
    parser.add_argument("--cost-file", default=None)
    parser.add_argument(
        "--selection-metric",
        choices=("mae", "cost_aware_objective"),
        default="cost_aware_objective",
    )
    parser.add_argument("--utility-finalizer", choices=tuple(sorted(UTILITY_FINALIZERS)), default="equal_average")
    parser.add_argument("--deployment-finalizer", choices=tuple(sorted(DEPLOYMENT_FINALIZERS)), default="equal_average")
    parser.add_argument("--allow-finalizer-mismatch", action="store_true")
    parser.add_argument("--state-sampling", choices=tuple(sorted(STATE_SAMPLING_MODES)), default="uniform")
    parser.add_argument("--first-query-target", choices=tuple(sorted(FIRST_QUERY_TARGETS)), default="soft")
    parser.add_argument("--first-query-temperature", type=float, default=0.02)
    parser.add_argument("--first-query-loss-weight", type=float, default=2.0)
    parser.add_argument("--first-query-sampling-ratio", type=float, default=0.0)
    parser.add_argument("--first-query-head", choices=tuple(sorted(FIRST_QUERY_HEADS)), default="separate")
    parser.add_argument("--first-query-regret-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--first-query-initialization",
        choices=tuple(sorted(FIRST_QUERY_INITIALIZATIONS)),
        default="random",
    )
    parser.add_argument("--routerdc-checkpoint-path", default=DEFAULT_ROUTERDC_FIRST_QUERY_CHECKPOINT)
    parser.add_argument("--routerdc-consistency-weight", type=float, default=0.0)
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
            cost_mode=args.cost_mode,
            cost_file=args.cost_file,
            selection_metric=args.selection_metric,
            utility_finalizer=args.utility_finalizer,
            deployment_finalizer=args.deployment_finalizer,
            allow_finalizer_mismatch=args.allow_finalizer_mismatch,
            state_sampling=args.state_sampling,
            first_query_target=args.first_query_target,
            first_query_temperature=args.first_query_temperature,
            first_query_loss_weight=args.first_query_loss_weight,
            first_query_sampling_ratio=args.first_query_sampling_ratio,
            first_query_head=args.first_query_head,
            first_query_regret_loss_weight=args.first_query_regret_loss_weight,
            first_query_initialization=args.first_query_initialization,
            routerdc_checkpoint_path=args.routerdc_checkpoint_path,
            routerdc_consistency_weight=args.routerdc_consistency_weight,
            use_expert_embeddings=not args.no_expert_embeddings,
            history_encoder_type=args.history_encoder_type,
            action_head_type=args.action_head_type,
            device=args.device,
            debug=args.debug,
        )
    )


if __name__ == "__main__":
    main()
