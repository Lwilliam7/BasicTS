"""Train optional sparse mixers over queried COSTARTS experts.

This experiment is separate from the action router. It uses the generated
subset-state caches, assigns weights only to already queried experts, and
compares queried-subset mixing against best-queried-expert selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, Union

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from scripts.build_costarts_subset_states import validate_costarts_subset_states
    from scripts.router_experiment_config import load_router_experiment_config
except ImportError:
    from build_costarts_subset_states import validate_costarts_subset_states
    from router_experiment_config import load_router_experiment_config


DEFAULT_TRAIN_CACHE = "cache/costarts_subset_states_train.pt"
DEFAULT_VAL_CACHE = "cache/costarts_subset_states_val.pt"
DEFAULT_CHECKPOINT_DIR = "checkpoints/costarts_subset_utility/sparse_mixers"
DEFAULT_RESULTS_DIR = "results/router_summary/costarts_subset_utility"
VALID_MIXER_TYPES = ("scalar", "horizon", "variable", "horizon_variable")


@dataclass
class SparseMixerTrainingConfig:
    train_cache_path: str = DEFAULT_TRAIN_CACHE
    val_cache_path: str = DEFAULT_VAL_CACHE
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR
    results_dir: str = DEFAULT_RESULTS_DIR
    mixer_types: tuple[str, ...] = ("scalar",)
    batch_size: int = 512
    max_epochs: int = 20
    patience: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    entropy_weight: float = 0.0
    seed: int = 7
    embedding_dim: int = 64
    hidden_dim: int = 64
    device: str = "cpu"
    debug: bool = False


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _load_torch(path: Union[str, Path]) -> dict[str, Any]:
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


class NonEmptySubsetStateDataset(Dataset):
    """Subset-state dataset restricted to states with at least one query."""

    def __init__(self, cache: Mapping[str, Any]) -> None:
        validate_costarts_subset_states(cache)
        subset_size = cache["subset_size"].to(torch.long)
        self.indices = torch.nonzero(subset_size > 0, as_tuple=False).flatten()
        if self.indices.numel() == 0:
            raise ValueError("Sparse mixing requires subset states with at least one queried expert.")
        self.cache = cache

    def __len__(self) -> int:
        return int(self.indices.numel())

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        source_index = int(self.indices[index])
        return {
            "history": self.cache["history"][source_index],
            "queried_mask": self.cache["queried_mask"][source_index],
            "queried_expert_ids": self.cache["queried_expert_ids"][source_index],
            "queried_expert_forecasts": self.cache["queried_expert_forecasts"][source_index],
            "true_targets": self.cache["true_targets"][source_index],
            "target_mask": self.cache["target_mask"][source_index],
            "true_expert_error_vector": self.cache["true_expert_error_vector"][source_index],
            "subset_size": self.cache["subset_size"][source_index],
            "sample_index": self.cache["sample_index"][source_index],
            "source_row": self.cache["source_row"][source_index],
        }


class SparseQueriedExpertMixer(nn.Module):
    """Mask-safe mixer that assigns weights only to queried expert slots."""

    def __init__(
        self,
        *,
        mixer_type: str,
        num_experts: int,
        max_subset_size: int,
        input_len: int = 96,
        forecast_horizon: int = 12,
        num_features: int = 7,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        if mixer_type not in VALID_MIXER_TYPES:
            raise ValueError(f"Unknown mixer_type {mixer_type!r}; expected one of {VALID_MIXER_TYPES}")
        self.mixer_type = mixer_type
        self.num_experts = int(num_experts)
        self.max_subset_size = int(max_subset_size)
        self.input_len = int(input_len)
        self.forecast_horizon = int(forecast_horizon)
        self.num_features = int(num_features)
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)

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
        self.expert_embeddings = nn.Embedding(num_experts, embedding_dim)
        self.forecast_encoder = nn.Sequential(
            nn.Linear(forecast_horizon * num_features, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
        )
        self.slot_fusion = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
        )
        if mixer_type == "scalar":
            output_dim = 1
        elif mixer_type == "horizon":
            output_dim = forecast_horizon
        elif mixer_type == "variable":
            output_dim = num_features
        else:
            output_dim = forecast_horizon * num_features
        self.logit_head = nn.Linear(embedding_dim, output_dim)

    def slot_logits(
        self,
        history: torch.Tensor,
        queried_expert_ids: torch.Tensor,
        queried_expert_forecasts: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = history.shape[0]
        assert tuple(history.shape[1:]) == (self.input_len, self.num_features)
        assert tuple(queried_expert_ids.shape) == (batch_size, self.max_subset_size)
        assert tuple(queried_expert_forecasts.shape) == (
            batch_size,
            self.max_subset_size,
            self.forecast_horizon,
            self.num_features,
        )

        history_representation = self.history_encoder(history.transpose(1, 2)).squeeze(-1)
        history_representation = self.history_projection(history_representation)
        history_slots = history_representation[:, None, :].expand(-1, self.max_subset_size, -1)

        safe_ids = queried_expert_ids.clamp_min(0)
        expert_representation = self.expert_embeddings(safe_ids)
        forecast_representation = self.forecast_encoder(
            queried_expert_forecasts.reshape(
                batch_size,
                self.max_subset_size,
                self.forecast_horizon * self.num_features,
            )
        )
        slot_features = self.slot_fusion(
            torch.cat((history_slots, expert_representation, forecast_representation), dim=-1)
        )
        logits = self.logit_head(slot_features)
        if self.mixer_type == "scalar":
            return logits.squeeze(-1)
        if self.mixer_type == "horizon":
            return logits.permute(0, 2, 1)
        if self.mixer_type == "variable":
            return logits.permute(0, 2, 1)
        return logits.view(
            batch_size,
            self.max_subset_size,
            self.forecast_horizon,
            self.num_features,
        ).permute(0, 2, 3, 1)

    def forward(
        self,
        history: torch.Tensor,
        queried_expert_ids: torch.Tensor,
        queried_expert_forecasts: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        valid_slots = queried_expert_ids >= 0
        logits = self.slot_logits(history, queried_expert_ids, queried_expert_forecasts)
        weights = mask_safe_softmax(logits, valid_slots, self.mixer_type)
        mixed_prediction = mix_queried_forecasts(queried_expert_forecasts, weights, self.mixer_type)
        assert tuple(mixed_prediction.shape[1:]) == (self.forecast_horizon, self.num_features)
        return {
            "logits": logits,
            "weights": weights,
            "mixed_prediction": mixed_prediction,
        }


def _mask_for_mixer(valid_slots: torch.Tensor, mixer_type: str) -> torch.Tensor:
    if mixer_type == "scalar":
        return valid_slots
    if mixer_type == "horizon":
        return valid_slots[:, None, :]
    if mixer_type == "variable":
        return valid_slots[:, None, :]
    return valid_slots[:, None, None, :]


def mask_safe_softmax(logits: torch.Tensor, valid_slots: torch.Tensor, mixer_type: str) -> torch.Tensor:
    mask = _mask_for_mixer(valid_slots.to(torch.bool), mixer_type)
    masked_logits = logits.masked_fill(~mask, -1e9)
    weights = torch.softmax(masked_logits, dim=-1).masked_fill(~mask, 0.0)
    denominator = weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    weights = weights / denominator
    if torch.any(valid_slots.sum(dim=-1) == 1):
        single = valid_slots.sum(dim=-1) == 1
        exact = _mask_for_mixer(valid_slots[single].to(weights.dtype), mixer_type)
        weights[single] = exact
    return weights


def mix_queried_forecasts(
    queried_expert_forecasts: torch.Tensor,
    weights: torch.Tensor,
    mixer_type: str,
) -> torch.Tensor:
    if mixer_type == "scalar":
        return (queried_expert_forecasts * weights[:, :, None, None]).sum(dim=1)
    if mixer_type == "horizon":
        forecasts = queried_expert_forecasts.permute(0, 2, 1, 3)
        return (forecasts * weights[:, :, :, None]).sum(dim=2)
    if mixer_type == "variable":
        forecasts = queried_expert_forecasts.permute(0, 3, 1, 2)
        return (forecasts * weights[:, :, :, None]).sum(dim=2).permute(0, 2, 1)
    forecasts = queried_expert_forecasts.permute(0, 2, 3, 1)
    return (forecasts * weights).sum(dim=-1)


def weight_map(weights: torch.Tensor, mixer_type: str, forecast_horizon: int, num_features: int) -> torch.Tensor:
    if mixer_type == "scalar":
        return weights[:, None, None, :].expand(-1, forecast_horizon, num_features, -1)
    if mixer_type == "horizon":
        return weights[:, :, None, :].expand(-1, -1, num_features, -1)
    if mixer_type == "variable":
        return weights[:, None, :, :].expand(-1, forecast_horizon, -1, -1)
    return weights


def masked_forecast_mae(prediction: torch.Tensor, target: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    mask = target_mask.to(prediction.dtype)
    return (torch.abs(prediction - target) * mask).sum() / mask.sum().clamp_min(1.0)


def masked_forecast_mse(prediction: torch.Tensor, target: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    mask = target_mask.to(prediction.dtype)
    return (((prediction - target) ** 2) * mask).sum() / mask.sum().clamp_min(1.0)


def mixer_entropy(weights: torch.Tensor, mixer_type: str, forecast_horizon: int, num_features: int) -> torch.Tensor:
    mapped = weight_map(weights, mixer_type, forecast_horizon, num_features)
    return -(mapped.clamp_min(1e-12) * mapped.clamp_min(1e-12).log()).sum(dim=-1).mean()


def _best_queried_oracle_forecast(batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    expert_errors = batch["true_expert_error_vector"]
    queried_ids = batch["queried_expert_ids"]
    valid_slots = queried_ids >= 0
    slot_errors = expert_errors.gather(1, queried_ids.clamp_min(0))
    slot_errors = slot_errors.masked_fill(~valid_slots, float("inf"))
    best_slot = torch.argmin(slot_errors, dim=1)
    rows = torch.arange(queried_ids.shape[0], device=queried_ids.device)
    return batch["queried_expert_forecasts"][rows, best_slot]


def _equal_average_forecast(batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    valid_slots = (batch["queried_expert_ids"] >= 0).to(batch["queried_expert_forecasts"].dtype)
    weights = valid_slots / valid_slots.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (batch["queried_expert_forecasts"] * weights[:, :, None, None]).sum(dim=1)


def _to_device(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _assert_cache_alignment(train_cache: Mapping[str, Any], val_cache: Mapping[str, Any]) -> None:
    for key in ("expert_names", "num_experts", "max_subset_size", "forecast_horizon", "num_features"):
        if train_cache[key] != val_cache[key]:
            raise AssertionError(f"Train/val subset cache mismatch for {key}: {train_cache[key]} != {val_cache[key]}")


def evaluate_baselines(
    loader: DataLoader,
    *,
    device: Union[str, torch.device],
) -> dict[str, dict[str, float]]:
    device = torch.device(device)
    totals = {
        "best_queried_expert_oracle": {"mae_sum": 0.0, "mse_sum": 0.0, "mask_sum": 0.0},
        "equal_average_queried_experts": {"mae_sum": 0.0, "mse_sum": 0.0, "mask_sum": 0.0},
    }
    for batch in loader:
        batch = _to_device(batch, device)
        target = batch["true_targets"]
        mask = batch["target_mask"].to(target.dtype)
        mask_sum = float(mask.sum().detach().cpu())
        predictions = {
            "best_queried_expert_oracle": _best_queried_oracle_forecast(batch),
            "equal_average_queried_experts": _equal_average_forecast(batch),
        }
        for name, prediction in predictions.items():
            totals[name]["mae_sum"] += float((torch.abs(prediction - target) * mask).sum().detach().cpu())
            totals[name]["mse_sum"] += float((((prediction - target) ** 2) * mask).sum().detach().cpu())
            totals[name]["mask_sum"] += mask_sum
    return {
        name: {
            "mae": values["mae_sum"] / max(values["mask_sum"], 1.0),
            "mse": values["mse_sum"] / max(values["mask_sum"], 1.0),
        }
        for name, values in totals.items()
    }


@torch.no_grad()
def evaluate_mixer(
    model: SparseQueriedExpertMixer,
    loader: DataLoader,
    *,
    device: Union[str, torch.device],
) -> tuple[dict[str, float], dict[str, Any]]:
    device = torch.device(device)
    model.eval()
    mae_sum = 0.0
    mse_sum = 0.0
    mask_sum = 0.0
    entropy_sum = 0.0
    max_weight_sum = 0.0
    invalid_weight_max = 0.0
    single_query_exact = True
    per_expert_weight = torch.zeros(model.num_experts, dtype=torch.float64)
    per_expert_count = torch.zeros(model.num_experts, dtype=torch.float64)
    subset_size_stats: dict[int, dict[str, float]] = {}

    for batch in loader:
        batch = _to_device(batch, device)
        outputs = model(
            batch["history"],
            batch["queried_expert_ids"],
            batch["queried_expert_forecasts"],
        )
        prediction = outputs["mixed_prediction"]
        target = batch["true_targets"]
        mask = batch["target_mask"].to(prediction.dtype)
        local_mask_sum = float(mask.sum().detach().cpu())
        mae_sum += float((torch.abs(prediction - target) * mask).sum().detach().cpu())
        mse_sum += float((((prediction - target) ** 2) * mask).sum().detach().cpu())
        mask_sum += local_mask_sum

        mapped_weights = weight_map(
            outputs["weights"],
            model.mixer_type,
            model.forecast_horizon,
            model.num_features,
        )
        entropy_values = -(mapped_weights.clamp_min(1e-12) * mapped_weights.clamp_min(1e-12).log()).sum(dim=-1)
        entropy_sum += float(entropy_values.sum().detach().cpu())
        max_weight_sum += float(mapped_weights.max(dim=-1).values.sum().detach().cpu())

        invalid_slots = ~(batch["queried_expert_ids"] >= 0)
        invalid_map = _mask_for_mixer(invalid_slots, model.mixer_type)
        if torch.any(invalid_map):
            invalid_weight_max = max(
                invalid_weight_max,
                float(outputs["weights"].masked_select(invalid_map).max().detach().cpu())
                if torch.any(invalid_map)
                else 0.0,
            )

        subset_sizes = batch["subset_size"].detach().cpu().to(torch.long)
        single = subset_sizes == 1
        if torch.any(single):
            single_weights = mapped_weights[single]
            single_query_exact = single_query_exact and bool(torch.allclose(
                single_weights.max(dim=-1).values,
                torch.ones_like(single_weights[..., 0]),
                atol=1e-7,
            ))

        slot_average_weights = mapped_weights.mean(dim=(1, 2)).detach().cpu()
        valid_slots = (batch["queried_expert_ids"] >= 0).detach().cpu()
        expert_ids = batch["queried_expert_ids"].detach().cpu().clamp_min(0)
        for row in range(expert_ids.shape[0]):
            size = int(subset_sizes[row])
            entry = subset_size_stats.setdefault(size, {"count": 0.0, "max_weight_sum": 0.0, "entropy_sum": 0.0})
            entry["count"] += 1.0
            entry["max_weight_sum"] += float(slot_average_weights[row].max())
            entry["entropy_sum"] += float(entropy_values[row].detach().cpu().mean())
            for slot in torch.nonzero(valid_slots[row], as_tuple=False).flatten().tolist():
                expert = int(expert_ids[row, slot])
                per_expert_weight[expert] += float(slot_average_weights[row, slot])
                per_expert_count[expert] += 1.0

    denominator_positions = mask_sum
    metric_denominator = max(mask_sum, 1.0)
    stats_denominator = max(denominator_positions, 1.0)
    metrics = {
        "mae": mae_sum / metric_denominator,
        "mse": mse_sum / metric_denominator,
        "entropy": entropy_sum / stats_denominator,
        "average_max_weight": max_weight_sum / stats_denominator,
        "max_invalid_weight": invalid_weight_max,
        "single_query_exact": single_query_exact,
    }
    stats = {
        "per_expert_average_weight_when_queried": (
            per_expert_weight / per_expert_count.clamp_min(1.0)
        ).tolist(),
        "per_expert_queried_count": per_expert_count.tolist(),
        "by_subset_size": {
            str(size): {
                "count": values["count"],
                "average_max_weight": values["max_weight_sum"] / max(values["count"], 1.0),
                "average_entropy": values["entropy_sum"] / max(values["count"], 1.0),
            }
            for size, values in sorted(subset_size_stats.items())
        },
    }
    return metrics, stats


def train_one_mixer(
    mixer_type: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_cache: Mapping[str, Any],
    config: SparseMixerTrainingConfig,
) -> tuple[dict[str, float], dict[str, Any]]:
    device = torch.device(config.device)
    model = SparseQueriedExpertMixer(
        mixer_type=mixer_type,
        num_experts=int(train_cache["num_experts"]),
        max_subset_size=int(train_cache["max_subset_size"]),
        input_len=96,
        forecast_horizon=int(train_cache["forecast_horizon"]),
        num_features=int(train_cache["num_features"]),
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / f"best_sparse_mixer_{mixer_type}.pt"
    last_path = checkpoint_dir / f"last_sparse_mixer_{mixer_type}.pt"

    best_mae = math.inf
    best_metrics: dict[str, float] = {}
    best_stats: dict[str, Any] = {}
    epochs_without_improvement = 0

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        running_total = 0.0
        running_forecast = 0.0
        running_entropy = 0.0
        batches = 0
        for batch in train_loader:
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                batch["history"],
                batch["queried_expert_ids"],
                batch["queried_expert_forecasts"],
            )
            forecast_loss = masked_forecast_mae(
                outputs["mixed_prediction"],
                batch["true_targets"],
                batch["target_mask"],
            )
            entropy_loss = mixer_entropy(
                outputs["weights"],
                mixer_type,
                int(train_cache["forecast_horizon"]),
                int(train_cache["num_features"]),
            )
            loss = forecast_loss + float(config.entropy_weight) * entropy_loss
            loss.backward()
            if config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()

            running_total += float(loss.detach().cpu())
            running_forecast += float(forecast_loss.detach().cpu())
            running_entropy += float(entropy_loss.detach().cpu())
            batches += 1

        val_metrics, val_stats = evaluate_mixer(model, val_loader, device=device)
        scheduler.step(val_metrics["mae"])
        print(
            f"{mixer_type} epoch {epoch:03d} "
            f"train_loss={running_total / max(batches, 1):.6f} "
            f"train_mae={running_forecast / max(batches, 1):.6f} "
            f"train_entropy={running_entropy / max(batches, 1):.6f} "
            f"val_mae={val_metrics['mae']:.6f} "
            f"val_mse={val_metrics['mse']:.6f}"
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "mixer_type": mixer_type,
                "config": asdict(config),
                "epoch": epoch,
                "val_metrics": val_metrics,
            },
            last_path,
        )
        if val_metrics["mae"] < best_mae:
            best_mae = val_metrics["mae"]
            best_metrics = val_metrics
            best_stats = val_stats
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "mixer_type": mixer_type,
                    "config": asdict(config),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                print(f"Early stopping {mixer_type} at epoch {epoch}.")
                break

    if not best_metrics:
        best_metrics, best_stats = evaluate_mixer(model, val_loader, device=device)
    return best_metrics, best_stats


def _parse_mixer_types(value: str) -> tuple[str, ...]:
    if value == "all":
        return VALID_MIXER_TYPES
    mixer_types = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = [item for item in mixer_types if item not in VALID_MIXER_TYPES]
    if unknown:
        raise ValueError(f"Unknown mixer types: {unknown}. Valid values: {VALID_MIXER_TYPES} or all")
    if not mixer_types:
        raise ValueError("At least one mixer type is required.")
    return mixer_types


def train_and_evaluate_sparse_mixers(config: SparseMixerTrainingConfig) -> dict[str, Any]:
    set_reproducible_seed(config.seed)
    device = torch.device(config.device)
    train_cache = _load_torch(config.train_cache_path)
    val_cache = _load_torch(config.val_cache_path)
    _assert_cache_alignment(train_cache, val_cache)
    validate_costarts_subset_states(train_cache)
    validate_costarts_subset_states(val_cache)

    train_dataset = NonEmptySubsetStateDataset(train_cache)
    val_dataset = NonEmptySubsetStateDataset(val_cache)
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    print("Sparse queried-expert mixer configuration:")
    print(f"  mixer_types: {config.mixer_types}")
    print(f"  train states: {len(train_dataset)} non-empty / {train_cache['num_states']} total")
    print(f"  val states: {len(val_dataset)} non-empty / {val_cache['num_states']} total")
    print(f"  experts: {val_cache['expert_names']}")
    print(f"  device: {device}")
    print(f"  entropy_weight: {config.entropy_weight}")

    first_batch = _to_device(next(iter(train_loader)), device)
    assert tuple(first_batch["history"].shape[1:]) == (96, 7)
    assert tuple(first_batch["queried_expert_forecasts"].shape[2:]) == (
        int(train_cache["forecast_horizon"]),
        int(train_cache["num_features"]),
    )
    assert torch.all(first_batch["subset_size"] > 0)
    if config.debug:
        print("Debug first batch:")
        print("  history:", tuple(first_batch["history"].shape))
        print("  queried_expert_ids:", tuple(first_batch["queried_expert_ids"].shape))
        print("  queried_expert_forecasts:", tuple(first_batch["queried_expert_forecasts"].shape))
        print("  true_targets:", tuple(first_batch["true_targets"].shape))

    baseline_metrics = evaluate_baselines(val_loader, device=device)
    results_rows: list[dict[str, Any]] = []
    statistics: dict[str, Any] = {
        "config": asdict(config),
        "expert_names": val_cache["expert_names"],
        "num_val_non_empty_states": len(val_dataset),
        "baselines": baseline_metrics,
        "mixers": {},
    }

    oracle_mae = baseline_metrics["best_queried_expert_oracle"]["mae"]
    for method, metrics in baseline_metrics.items():
        results_rows.append(
            {
                "mixer_type": "baseline",
                "method": method,
                "mae": metrics["mae"],
                "mse": metrics["mse"],
                "delta_mae_vs_best_queried_oracle": metrics["mae"] - oracle_mae,
                "beats_best_queried_oracle": metrics["mae"] < oracle_mae,
                "num_val_states": len(val_dataset),
            }
        )

    for mixer_type in config.mixer_types:
        print("\n" + "=" * 80)
        print(f"Training sparse mixer: {mixer_type}")
        print("=" * 80)
        metrics, stats = train_one_mixer(mixer_type, train_loader, val_loader, train_cache, config)
        if metrics["max_invalid_weight"] > 1e-7:
            raise AssertionError(f"{mixer_type} assigned nonzero weight to an unqueried slot.")
        if not metrics["single_query_exact"]:
            raise AssertionError(f"{mixer_type} did not reduce exactly to the single queried forecast.")
        statistics["mixers"][mixer_type] = {"metrics": metrics, "statistics": stats}
        results_rows.append(
            {
                "mixer_type": mixer_type,
                "method": "learned_sparse_queried_expert_mixer",
                "mae": metrics["mae"],
                "mse": metrics["mse"],
                "delta_mae_vs_best_queried_oracle": metrics["mae"] - oracle_mae,
                "beats_best_queried_oracle": metrics["mae"] < oracle_mae,
                "num_val_states": len(val_dataset),
                "entropy": metrics["entropy"],
                "average_max_weight": metrics["average_max_weight"],
                "single_query_exact": metrics["single_query_exact"],
                "max_invalid_weight": metrics["max_invalid_weight"],
            }
        )

    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "mixing_results.csv"
    stats_path = results_dir / "mix_weight_statistics.json"
    fieldnames = sorted({key for row in results_rows for key in row.keys()})
    with results_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_rows)
    stats_path.write_text(json.dumps(_jsonable(statistics), indent=2), encoding="utf-8")
    print(f"\nSaved: {results_path}")
    print(f"Saved: {stats_path}")
    return statistics


def parse_args() -> argparse.Namespace:
    repo_config = load_router_experiment_config()
    cache_paths = dict(repo_config.cache_paths)
    parser = argparse.ArgumentParser(description="Train sparse queried-expert COSTARTS mixers.")
    parser.add_argument("--train-cache", default=cache_paths.get("costarts_subset_states_train", DEFAULT_TRAIN_CACHE))
    parser.add_argument("--val-cache", default=cache_paths.get("costarts_subset_states_val", DEFAULT_VAL_CACHE))
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--mixer-types",
        default="scalar",
        help="Comma-separated mixer types or 'all'. Valid: scalar,horizon,variable,horizon_variable.",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--entropy-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=repo_config.random_seed)
    parser.add_argument("--embedding-dim", type=int, default=repo_config.embedding_dim)
    parser.add_argument("--hidden-dim", type=int, default=repo_config.hidden_dim)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_and_evaluate_sparse_mixers(
        SparseMixerTrainingConfig(
            train_cache_path=args.train_cache,
            val_cache_path=args.val_cache,
            checkpoint_dir=args.checkpoint_dir,
            results_dir=args.results_dir,
            mixer_types=_parse_mixer_types(args.mixer_types),
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            grad_clip_norm=args.grad_clip_norm,
            entropy_weight=args.entropy_weight,
            seed=args.seed,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            device=args.device,
            debug=args.debug,
        )
    )


if __name__ == "__main__":
    main()
