"""Train and evaluate queried-subset rerankers for COSTARTS-style routers."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from scripts.costars.build_costarts_subset_states import validate_costarts_subset_states
    from scripts.train_costarts_router import COSTARTSRouter
except ImportError:
    from scripts.costars.build_costarts_subset_states import validate_costarts_subset_states
    from train_costarts_router import COSTARTSRouter


DEFAULT_TRAIN_CACHE = "cache/costarts_subset_states_train.pt"
DEFAULT_VAL_CACHE = "cache/costarts_subset_states_val.pt"
DEFAULT_COSTARTS_CHECKPOINT = "checkpoints/costarts/best_costarts_router.pt"
DEFAULT_OUTPUT_DIR = "results/router_summary/costarts_subset_utility"
DEFAULT_CHECKPOINT = "checkpoints/costarts_subset_utility/queried_subset_reranker.pt"


@dataclass
class RerankerTrainingConfig:
    train_cache: str = DEFAULT_TRAIN_CACHE
    val_cache: str = DEFAULT_VAL_CACHE
    costarts_checkpoint: str = DEFAULT_COSTARTS_CHECKPOINT
    output_dir: str = DEFAULT_OUTPUT_DIR
    checkpoint_path: str = DEFAULT_CHECKPOINT
    batch_size: int = 512
    max_epochs: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    pairwise_loss_weight: float = 1.0
    winner_loss_weight: float = 1.0
    error_loss_weight: float = 0.2
    seed: int = 7
    device: str = "cpu"
    max_examples: int = 50


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _load_torch(path: str | Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return str(value)
    return value


class QueriedSubsetDataset(Dataset):
    def __init__(self, cache: Mapping[str, Any]) -> None:
        validate_costarts_subset_states(cache)
        self.cache = cache
        self.indices = torch.nonzero(cache["subset_size"] > 0, as_tuple=False).flatten()

    def __len__(self) -> int:
        return int(self.indices.numel())

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        state_index = int(self.indices[index])
        return {
            "state_index": torch.tensor(state_index, dtype=torch.long),
            "source_row": self.cache["source_row"][state_index],
            "sample_index": self.cache["sample_index"][state_index],
            "history": self.cache["history"][state_index],
            "queried_mask": self.cache["queried_mask"][state_index],
            "queried_expert_ids": self.cache["queried_expert_ids"][state_index],
            "queried_expert_forecasts": self.cache["queried_expert_forecasts"][state_index],
            "true_targets": self.cache["true_targets"][state_index],
            "target_mask": self.cache["target_mask"][state_index],
            "true_expert_error_vector": self.cache["true_expert_error_vector"][state_index],
            "pairwise_labels_queried": self.cache["pairwise_labels_queried"][state_index],
            "subset_size": self.cache["subset_size"][state_index],
        }


class QueriedSubsetReranker(nn.Module):
    def __init__(
        self,
        num_experts: int,
        max_subset_size: int,
        input_len: int = 96,
        forecast_horizon: int = 12,
        num_features: int = 7,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
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
        self.expert_embedding = nn.Embedding(num_experts, embedding_dim)
        self.mask_encoder = nn.Sequential(
            nn.Linear(num_experts, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
        )
        self.forecast_encoder = nn.Sequential(
            nn.Linear(forecast_horizon * num_features, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
        )
        self.state_fusion = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
        )
        self.pairwise_score_head = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.winner_head = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.error_head = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

    def encode_state(
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

        history_features = self.history_encoder(history.transpose(1, 2)).squeeze(-1)
        history_features = self.history_projection(history_features)
        mask_features = self.mask_encoder(queried_mask.to(history.dtype))
        valid_slots = queried_expert_ids >= 0
        safe_ids = queried_expert_ids.clamp_min(0)
        forecast_flat = queried_expert_forecasts.reshape(
            batch_size,
            self.max_subset_size,
            self.forecast_horizon * self.num_features,
        )
        forecast_features = self.forecast_encoder(forecast_flat)
        forecast_features = forecast_features + self.expert_embedding(safe_ids)
        forecast_features = forecast_features * valid_slots.unsqueeze(-1).to(history.dtype)
        denominator = valid_slots.sum(dim=1, keepdim=True).clamp_min(1).to(history.dtype)
        queried_features = forecast_features.sum(dim=1) / denominator
        return self.state_fusion(torch.cat((history_features, mask_features, queried_features), dim=-1))

    def forward(
        self,
        history: torch.Tensor,
        queried_mask: torch.Tensor,
        queried_expert_ids: torch.Tensor,
        queried_expert_forecasts: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        state = self.encode_state(history, queried_mask, queried_expert_ids, queried_expert_forecasts)
        batch_size = history.shape[0]
        expert_ids = torch.arange(self.num_experts, device=history.device)
        expert_features = self.expert_embedding(expert_ids).unsqueeze(0).expand(batch_size, -1, -1)
        state_features = state.unsqueeze(1).expand(-1, self.num_experts, -1)
        queried_indicator = queried_mask.to(history.dtype).unsqueeze(-1)
        features = torch.cat((state_features, expert_features, queried_indicator), dim=-1)
        pairwise_scores = self.pairwise_score_head(features).squeeze(-1)
        winner_logits = self.winner_head(features).squeeze(-1)
        error_prediction = self.error_head(features).squeeze(-1)
        return {
            "pairwise_scores": pairwise_scores,
            "winner_logits": winner_logits,
            "error_prediction": error_prediction,
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
        }


def _move_batch(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _queried_winner(errors: torch.Tensor, queried_mask: torch.Tensor) -> torch.Tensor:
    return torch.argmin(errors.masked_fill(~queried_mask.to(torch.bool), float("inf")), dim=-1)


def _winner_loss(logits: torch.Tensor, winner: torch.Tensor, queried_mask: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.masked_fill(~queried_mask.to(torch.bool), -1e9), winner)


def _pairwise_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    labels = labels.to(torch.float32)
    diff = scores.unsqueeze(2) - scores.unsqueeze(1)
    valid = labels != 0
    if not torch.any(valid):
        return scores.sum() * 0.0
    return F.softplus(-labels[valid] * diff[valid]).mean()


def _error_loss(error_prediction: torch.Tensor, errors: torch.Tensor, queried_mask: torch.Tensor) -> torch.Tensor:
    mask = queried_mask.to(torch.bool)
    if not torch.any(mask):
        return error_prediction.sum() * 0.0
    return F.smooth_l1_loss(error_prediction[mask], errors[mask])


def reranker_losses(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    *,
    pairwise_weight: float,
    winner_weight: float,
    error_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    errors = batch["true_expert_error_vector"].to(outputs["pairwise_scores"].dtype)
    queried_mask = batch["queried_mask"].to(torch.bool)
    winner = _queried_winner(errors, queried_mask)
    pairwise_loss = _pairwise_loss(outputs["pairwise_scores"], batch["pairwise_labels_queried"])
    winner_loss = _winner_loss(outputs["winner_logits"], winner, queried_mask)
    error_loss = _error_loss(outputs["error_prediction"], errors, queried_mask)
    total = pairwise_weight * pairwise_loss + winner_weight * winner_loss + error_weight * error_loss
    selected = torch.argmax(outputs["winner_logits"].masked_fill(~queried_mask, -1e9), dim=-1)
    return total, {
        "total_loss": float(total.detach().cpu()),
        "pairwise_loss": float(pairwise_loss.detach().cpu()),
        "winner_loss": float(winner_loss.detach().cpu()),
        "error_loss": float(error_loss.detach().cpu()),
        "winner_accuracy": float((selected == winner).to(torch.float32).mean().detach().cpu()),
    }


@torch.no_grad()
def _costarts_map_predictions(
    checkpoint_path: str | Path,
    router_val_cache_path: str | Path,
    batch_size: int,
) -> torch.Tensor:
    checkpoint = _load_torch(checkpoint_path)
    router_val = _load_torch(router_val_cache_path)
    router = COSTARTSRouter(**checkpoint["router_config"])
    router.load_state_dict(checkpoint["router_state_dict"])
    router.eval()
    outputs = []
    histories = router_val["histories"].to(torch.float32)
    with torch.no_grad():
        for start in range(0, histories.shape[0], batch_size):
            batch = histories[start : start + batch_size]
            result = router(batch, sampled_rollout=False)
            outputs.append(result["map_prediction"].detach().cpu())
    return torch.cat(outputs, dim=0)


def _masked_argmin(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.argmin(values.masked_fill(~mask.to(torch.bool), float("inf")), dim=-1)


def _masked_argmax(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.argmax(values.masked_fill(~mask.to(torch.bool), -1e9), dim=-1)


def _forecast_metrics_for_selected(batch: Mapping[str, torch.Tensor], selected: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ids = batch["queried_expert_ids"]
    positions = (ids == selected[:, None]).to(torch.float32).argmax(dim=1)
    prediction = batch["queried_expert_forecasts"][torch.arange(ids.shape[0], device=ids.device), positions]
    mask = batch["target_mask"].to(prediction.dtype)
    denominator = mask.sum(dim=(1, 2)).clamp_min(1.0)
    mae = (torch.abs(prediction - batch["true_targets"]) * mask).sum(dim=(1, 2)) / denominator
    mse = ((prediction - batch["true_targets"]).pow(2) * mask).sum(dim=(1, 2)) / denominator
    return mae, mse


def _forecast_metrics_for_average(batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    valid = batch["queried_expert_ids"] >= 0
    weights = valid.to(batch["queried_expert_forecasts"].dtype)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    prediction = (batch["queried_expert_forecasts"] * weights[:, :, None, None]).sum(dim=1)
    mask = batch["target_mask"].to(prediction.dtype)
    denominator = mask.sum(dim=(1, 2)).clamp_min(1.0)
    mae = (torch.abs(prediction - batch["true_targets"]) * mask).sum(dim=(1, 2)) / denominator
    mse = ((prediction - batch["true_targets"]).pow(2) * mask).sum(dim=(1, 2)) / denominator
    return mae, mse


def _pairwise_accuracy(scores: torch.Tensor, labels: torch.Tensor) -> tuple[float, int]:
    labels = labels.to(torch.float32)
    diff = scores.unsqueeze(2) - scores.unsqueeze(1)
    valid = labels != 0
    if not torch.any(valid):
        return math.nan, 0
    correct = (torch.sign(diff[valid]) == labels[valid]).to(torch.float32)
    return float(correct.mean()), int(valid.sum())


@torch.no_grad()
def evaluate_selectors(
    model: QueriedSubsetReranker,
    val_cache: Mapping[str, Any],
    *,
    costarts_map_prediction: torch.Tensor,
    batch_size: int,
    device: torch.device,
    max_examples: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    model.eval()
    dataset = QueriedSubsetDataset(val_cache)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    expert_names = tuple(val_cache["expert_names"])
    num_experts = int(val_cache["num_experts"])
    selector_stats: dict[str, dict[str, Any]] = {
        name: {
            "mae_sum": 0.0,
            "mse_sum": 0.0,
            "match_sum": 0.0,
            "regret_sum": 0.0,
            "top2_match_sum": 0.0,
            "top2_count": 0,
            "pairwise_correct_weighted": 0.0,
            "pairwise_count": 0,
            "selection_counts": torch.zeros(num_experts, dtype=torch.long),
            "count": 0,
        }
        for name in (
            "current_absolute_predicted_error",
            "pairwise_reranker",
            "queried_subset_winner_classifier",
            "equal_average",
        )
    }
    examples = []

    for batch in loader:
        cpu_batch = batch
        batch = _move_batch(batch, device)
        queried_mask = batch["queried_mask"].to(torch.bool)
        errors = batch["true_expert_error_vector"]
        winner = _queried_winner(errors, queried_mask)
        oracle_mae = errors.gather(1, winner[:, None]).squeeze(1)
        outputs = model(
            batch["history"],
            batch["queried_mask"],
            batch["queried_expert_ids"],
            batch["queried_expert_forecasts"],
        )
        old_scores = -costarts_map_prediction[cpu_batch["source_row"]].to(device)
        selections = {
            "current_absolute_predicted_error": _masked_argmax(old_scores, queried_mask),
            "pairwise_reranker": _masked_argmax(outputs["pairwise_scores"], queried_mask),
            "queried_subset_winner_classifier": _masked_argmax(outputs["winner_logits"], queried_mask),
        }
        selector_pairwise_scores = {
            "current_absolute_predicted_error": old_scores,
            "pairwise_reranker": outputs["pairwise_scores"],
            "queried_subset_winner_classifier": outputs["winner_logits"],
        }
        avg_mae, avg_mse = _forecast_metrics_for_average(batch)

        for selector_name, selected in selections.items():
            mae, mse = _forecast_metrics_for_selected(batch, selected)
            stats = selector_stats[selector_name]
            stats["mae_sum"] += float(mae.sum().detach().cpu())
            stats["mse_sum"] += float(mse.sum().detach().cpu())
            stats["match_sum"] += float((selected == winner).to(torch.float32).sum().detach().cpu())
            selected_error = errors.gather(1, selected[:, None]).squeeze(1)
            stats["regret_sum"] += float((selected_error - oracle_mae).sum().detach().cpu())
            stats["selection_counts"] += torch.bincount(selected.detach().cpu(), minlength=num_experts)
            pair_acc, pair_count = _pairwise_accuracy(selector_pairwise_scores[selector_name], batch["pairwise_labels_queried"])
            if pair_count:
                stats["pairwise_correct_weighted"] += pair_acc * pair_count
                stats["pairwise_count"] += pair_count
            two_mask = batch["subset_size"] == 2
            if torch.any(two_mask):
                stats["top2_match_sum"] += float((selected[two_mask] == winner[two_mask]).to(torch.float32).sum().detach().cpu())
                stats["top2_count"] += int(two_mask.sum().detach().cpu())
            stats["count"] += int(selected.numel())

        avg_stats = selector_stats["equal_average"]
        avg_stats["mae_sum"] += float(avg_mae.sum().detach().cpu())
        avg_stats["mse_sum"] += float(avg_mse.sum().detach().cpu())
        avg_stats["regret_sum"] += float((avg_mae - oracle_mae).sum().detach().cpu())
        avg_stats["count"] += int(avg_mae.numel())

        old_selected = selections["current_absolute_predicted_error"]
        pair_selected = selections["pairwise_reranker"]
        cls_selected = selections["queried_subset_winner_classifier"]
        success = (old_selected != winner) & ((pair_selected == winner) | (cls_selected == winner))
        if len(examples) < max_examples and torch.any(success):
            success_indices = torch.nonzero(success.detach().cpu(), as_tuple=False).flatten()
            for local_cpu in success_indices.tolist():
                if len(examples) >= max_examples:
                    break
                queried_ids = [
                    int(x)
                    for x in cpu_batch["queried_expert_ids"][local_cpu].tolist()
                    if int(x) >= 0
                ]
                state_index = int(cpu_batch["state_index"][local_cpu])
                row = {
                    "state_index": state_index,
                    "sample_index": int(cpu_batch["sample_index"][local_cpu]),
                    "source_row": int(cpu_batch["source_row"][local_cpu]),
                    "subset_size": int(cpu_batch["subset_size"][local_cpu]),
                    "queried_experts": " + ".join(expert_names[i] for i in queried_ids),
                    "oracle_winner": expert_names[int(winner.detach().cpu()[local_cpu])],
                    "old_absolute_error_selected": expert_names[int(old_selected.detach().cpu()[local_cpu])],
                    "pairwise_reranker_selected": expert_names[int(pair_selected.detach().cpu()[local_cpu])],
                    "winner_classifier_selected": expert_names[int(cls_selected.detach().cpu()[local_cpu])],
                    "true_errors": json.dumps({
                        expert_names[i]: float(cpu_batch["true_expert_error_vector"][local_cpu, i])
                        for i in queried_ids
                    }, sort_keys=True),
                    "old_predicted_errors": json.dumps({
                        expert_names[i]: float(costarts_map_prediction[int(cpu_batch["source_row"][local_cpu]), i])
                        for i in queried_ids
                    }, sort_keys=True),
                    "pairwise_scores": json.dumps({
                        expert_names[i]: float(outputs["pairwise_scores"].detach().cpu()[local_cpu, i])
                        for i in queried_ids
                    }, sort_keys=True),
                    "winner_logits": json.dumps({
                        expert_names[i]: float(outputs["winner_logits"].detach().cpu()[local_cpu, i])
                        for i in queried_ids
                    }, sort_keys=True),
                    "old_regret": float(
                        errors.detach().cpu()[local_cpu, old_selected.detach().cpu()[local_cpu]]
                        - oracle_mae.detach().cpu()[local_cpu]
                    ),
                    "pairwise_regret": float(
                        errors.detach().cpu()[local_cpu, pair_selected.detach().cpu()[local_cpu]]
                        - oracle_mae.detach().cpu()[local_cpu]
                    ),
                    "classifier_regret": float(
                        errors.detach().cpu()[local_cpu, cls_selected.detach().cpu()[local_cpu]]
                        - oracle_mae.detach().cpu()[local_cpu]
                    ),
                }
                examples.append(row)

    rows = []
    for selector_name, stats in selector_stats.items():
        count = max(int(stats["count"]), 1)
        selection_counts = {
            expert_names[index]: int(value)
            for index, value in enumerate(stats["selection_counts"].tolist())
        } if "selection_counts" in stats else {}
        rows.append(
            {
                "selector": selector_name,
                "num_states": int(stats["count"]),
                "mae": stats["mae_sum"] / count,
                "mse": stats["mse_sum"] / count,
                "oracle_match_within_subset": (
                    stats["match_sum"] / count if selector_name != "equal_average" else math.nan
                ),
                "better_of_top_two_accuracy": (
                    stats["top2_match_sum"] / stats["top2_count"]
                    if stats["top2_count"]
                    else math.nan
                ),
                "pairwise_queried_subset_ranking_accuracy": (
                    stats["pairwise_correct_weighted"] / stats["pairwise_count"]
                    if stats["pairwise_count"]
                    else math.nan
                ),
                "within_subset_misselection_regret": stats["regret_sum"] / count,
                "selection_counts": json.dumps(selection_counts, sort_keys=True),
            }
        )
    return rows, examples


def train_reranker(config: RerankerTrainingConfig) -> dict[str, Any]:
    set_seed(config.seed)
    device = torch.device(config.device)
    train_cache = _load_torch(config.train_cache)
    val_cache = _load_torch(config.val_cache)
    validate_costarts_subset_states(train_cache)
    validate_costarts_subset_states(val_cache)
    if train_cache["split_role"] != "router_train" or val_cache["split_role"] != "router_val":
        raise ValueError("Reranker must train on router_train and evaluate on router_val")
    if tuple(train_cache["expert_names"]) != tuple(val_cache["expert_names"]):
        raise ValueError("Train/val expert ordering mismatch")

    model = QueriedSubsetReranker(
        num_experts=int(train_cache["num_experts"]),
        max_subset_size=int(train_cache["max_subset_size"]),
        forecast_horizon=int(train_cache["forecast_horizon"]),
        num_features=int(train_cache["num_features"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    train_loader = DataLoader(
        QueriedSubsetDataset(train_cache),
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    curves = []
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        totals = {"total_loss": 0.0, "pairwise_loss": 0.0, "winner_loss": 0.0, "error_loss": 0.0, "winner_accuracy": 0.0}
        seen = 0
        for batch in train_loader:
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                batch["history"],
                batch["queried_mask"],
                batch["queried_expert_ids"],
                batch["queried_expert_forecasts"],
            )
            total_loss, parts = reranker_losses(
                outputs,
                batch,
                pairwise_weight=config.pairwise_loss_weight,
                winner_weight=config.winner_loss_weight,
                error_weight=config.error_loss_weight,
            )
            total_loss.backward()
            if config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()
            batch_size = batch["history"].shape[0]
            for key in totals:
                totals[key] += parts[key] * batch_size
            seen += batch_size
        row = {"epoch": epoch, **{key: value / max(seen, 1) for key, value in totals.items()}}
        curves.append(row)
        print(
            f"Reranker epoch {epoch:03d} | loss={row['total_loss']:.6f} "
            f"pair={row['pairwise_loss']:.6f} winner={row['winner_loss']:.6f} "
            f"error={row['error_loss']:.6f} acc={row['winner_accuracy']:.3f}"
        )

    checkpoint_path = Path(config.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "router_type": "queried_subset_reranker",
            "model_state_dict": model.state_dict(),
            "model_config": model.config_dict(),
            "training_config": config.__dict__,
            "expert_names": list(train_cache["expert_names"]),
            "test_set_used": False,
        },
        checkpoint_path,
    )

    costarts_map = _costarts_map_predictions(
        config.costarts_checkpoint,
        "cache/costarts_router_val_cache.pt",
        config.batch_size,
    )
    rows, examples = evaluate_selectors(
        model,
        val_cache,
        costarts_map_prediction=costarts_map,
        batch_size=config.batch_size,
        device=device,
        max_examples=config.max_examples,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "reranking_comparison.csv"
    examples_path = output_dir / "reranking_examples.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    example_fields = list(examples[0]) if examples else [
        "state_index",
        "sample_index",
        "source_row",
        "subset_size",
        "queried_experts",
        "oracle_winner",
        "old_absolute_error_selected",
        "pairwise_reranker_selected",
        "winner_classifier_selected",
        "true_errors",
        "old_predicted_errors",
        "pairwise_scores",
        "winner_logits",
        "old_regret",
        "pairwise_regret",
        "classifier_regret",
    ]
    with examples_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=example_fields)
        writer.writeheader()
        writer.writerows(examples)

    curves_path = output_dir / "reranking_training_curves.csv"
    with curves_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(curves[0]))
        writer.writeheader()
        writer.writerows(curves)

    print(f"Saved: {comparison_path}")
    print(f"Saved: {examples_path}")
    print(f"Saved: {curves_path}")
    print(f"Saved: {checkpoint_path}")
    return {"comparison": rows, "examples": examples}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train queried-subset COSTARTS reranker.")
    parser.add_argument("--train-cache", default=DEFAULT_TRAIN_CACHE)
    parser.add_argument("--val-cache", default=DEFAULT_VAL_CACHE)
    parser.add_argument("--costarts-checkpoint", default=DEFAULT_COSTARTS_CHECKPOINT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--pairwise-loss-weight", type=float, default=1.0)
    parser.add_argument("--winner-loss-weight", type=float, default=1.0)
    parser.add_argument("--error-loss-weight", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-examples", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_reranker(
        RerankerTrainingConfig(
            train_cache=args.train_cache,
            val_cache=args.val_cache,
            costarts_checkpoint=args.costarts_checkpoint,
            output_dir=args.output_dir,
            checkpoint_path=args.checkpoint_path,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            grad_clip_norm=args.grad_clip_norm,
            pairwise_loss_weight=args.pairwise_loss_weight,
            winner_loss_weight=args.winner_loss_weight,
            error_loss_weight=args.error_loss_weight,
            seed=args.seed,
            device=args.device,
            max_examples=args.max_examples,
        )
    )


if __name__ == "__main__":
    main()
