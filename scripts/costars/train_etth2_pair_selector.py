"""Train a history-only ETTh2 frozen-expert pair selector from clean router caches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
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

from scripts.costars.analyze_etth2_pair_potential import (
    EXPECTED_EXPERTS,
    EXPECTED_HASHES,
    load_verified_cache,
    per_window_error,
    sha256_file,
    validate_cache_pair,
)


FIXED_PAIR_NAME = "DLinear+ModernTCN"
DEFAULT_SEEDS = (7, 11, 13, 17, 19)
DEFAULT_TEMPERATURE = 0.01
MARGIN_BINS = (
    ("<=0.005", None, 0.005),
    ("0.005_to_0.01", 0.005, 0.01),
    ("0.01_to_0.025", 0.01, 0.025),
    (">0.025", 0.025, None),
)


@dataclass(frozen=True)
class PairSelectorConfig:
    input_len: int = 96
    horizon: int = 12
    num_features: int = 7
    hidden_dim: int = 64
    batch_size: int = 512
    max_epochs: int = 50
    patience: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    pair_target: str = "soft"
    pair_temperature: float = DEFAULT_TEMPERATURE
    device: str = "cpu"


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


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def pair_class_order() -> list[dict[str, Any]]:
    pairs = []
    for class_index, (a, b) in enumerate(itertools.combinations(range(len(EXPECTED_EXPERTS)), 2)):
        pairs.append({
            "class_index": class_index,
            "expert_indices": [a, b],
            "expert_names": [EXPECTED_EXPERTS[a], EXPECTED_EXPERTS[b]],
            "pair": f"{EXPECTED_EXPERTS[a]}+{EXPECTED_EXPERTS[b]}",
        })
    return pairs


def pair_name_to_index(pair_name: str, pairs: Optional[Sequence[Mapping[str, Any]]] = None) -> int:
    pairs = pair_class_order() if pairs is None else pairs
    normalized = pair_name.replace(" ", "")
    for pair in pairs:
        if str(pair["pair"]) == normalized:
            return int(pair["class_index"])
    raise ValueError(f"Unknown pair: {pair_name}")


def pair_prediction_stack(cache: Mapping[str, Any], pairs: Optional[Sequence[Mapping[str, Any]]] = None) -> torch.Tensor:
    pairs = pair_class_order() if pairs is None else pairs
    predictions = []
    stack = cache["prediction_stack"]
    for pair in pairs:
        a, b = pair["expert_indices"]
        predictions.append(0.5 * stack[..., int(a)] + 0.5 * stack[..., int(b)])
    return torch.stack(predictions, dim=-1)


def pair_error_matrices(cache: Mapping[str, Any], pairs: Optional[Sequence[Mapping[str, Any]]] = None) -> tuple[torch.Tensor, torch.Tensor]:
    pair_predictions = pair_prediction_stack(cache, pairs)
    return per_window_error(pair_predictions, cache["targets"], cache["target_masks"])


def soft_pair_targets(pair_mae: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("pair_temperature must be positive")
    return torch.softmax(-pair_mae / temperature, dim=1)


def load_inputs(args: argparse.Namespace) -> tuple[dict, dict, dict, dict]:
    report = json.loads(Path(args.validation_report).read_text(encoding="utf-8"))
    potential = json.loads(Path(args.pair_potential_summary).read_text(encoding="utf-8"))
    if potential["cache_hashes"]["router_train"] != EXPECTED_HASHES["router_train"]:
        raise ValueError("pair-potential router_train hash differs from lock")
    if potential["cache_hashes"]["router_val"] != EXPECTED_HASHES["router_val"]:
        raise ValueError("pair-potential router_val hash differs from lock")
    if potential["scaler_hash"] != EXPECTED_HASHES["scaler"]:
        raise ValueError("pair-potential scaler hash differs from lock")
    if tuple(potential["expert_order"]) != EXPECTED_EXPERTS:
        raise ValueError("pair-potential expert order differs from lock")
    if potential["training_selected_best_fixed_pair"].replace(" ", "") != FIXED_PAIR_NAME:
        raise ValueError("Expected router-training fixed pair DLinear+ModernTCN")
    train_cache = load_verified_cache(Path(args.router_train_cache), "router_train", report)
    val_cache = load_verified_cache(Path(args.router_val_cache), "router_val", report)
    validate_cache_pair(train_cache, val_cache)
    return train_cache, val_cache, report, potential


class PairSelectorDataset(Dataset):
    """History-only dataset; pair labels are precomputed from router_train cache."""

    def __init__(self, cache: Mapping[str, Any], pair_mae: torch.Tensor, pair_target: str, temperature: float) -> None:
        if cache["split_role"] != "router_train":
            raise ValueError("PairSelectorDataset may only be built from router_train")
        self.histories = cache["histories"].to(torch.float32)
        self.hard_targets = pair_mae.argmin(dim=1).to(torch.long)
        self.soft_targets = soft_pair_targets(pair_mae, temperature).to(torch.float32)
        if pair_target not in {"hard", "soft"}:
            raise ValueError("pair_target must be hard or soft")
        self.pair_target = pair_target
        self.source_indices = cache["absolute_window_starts"].to(torch.long)

    def __len__(self) -> int:
        return int(self.histories.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {
            "history": self.histories[index],
            "hard_target": self.hard_targets[index],
            "source_index": self.source_indices[index],
        }
        if self.pair_target == "soft":
            item["soft_target"] = self.soft_targets[index]
        return item


class HistoryPairSelector(nn.Module):
    def __init__(self, input_len: int = 96, num_features: int = 7, hidden_dim: int = 64, num_pairs: int = 10) -> None:
        super().__init__()
        self.input_len = int(input_len)
        self.num_features = int(num_features)
        self.hidden_dim = int(hidden_dim)
        self.num_pairs = int(num_pairs)
        self.history_encoder = nn.Sequential(
            nn.Conv1d(num_features, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.GroupNorm(1, hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=4, dilation=2),
            nn.GELU(),
            nn.GroupNorm(1, hidden_dim),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_pairs),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or tuple(history.shape[1:]) != (self.input_len, self.num_features):
            raise ValueError(f"Expected history [B,{self.input_len},{self.num_features}], got {tuple(history.shape)}")
        encoded = self.history_encoder(history.transpose(1, 2)).squeeze(-1)
        return self.head(encoded)

    def config_dict(self) -> dict[str, Any]:
        return {
            "input_len": self.input_len,
            "num_features": self.num_features,
            "hidden_dim": self.hidden_dim,
            "num_pairs": self.num_pairs,
        }


def selector_loss(logits: torch.Tensor, batch: Mapping[str, torch.Tensor], pair_target: str) -> torch.Tensor:
    if pair_target == "hard":
        return F.cross_entropy(logits, batch["hard_target"])
    log_probs = F.log_softmax(logits, dim=1)
    return F.kl_div(log_probs, batch["soft_target"], reduction="batchmean")


def evaluate_logits(
    logits: torch.Tensor,
    pair_mae: torch.Tensor,
    pair_mse: torch.Tensor,
    fixed_pair_index: int,
    hard_targets: torch.Tensor,
    soft_targets: Optional[torch.Tensor] = None,
) -> dict[str, Any]:
    probs = torch.softmax(logits, dim=1)
    selected = probs.argmax(dim=1)
    top2 = torch.topk(probs, k=2, dim=1).indices
    top3 = torch.topk(probs, k=3, dim=1).indices
    selected_mae = pair_mae.gather(1, selected.view(-1, 1)).squeeze(1)
    selected_mse = pair_mse.gather(1, selected.view(-1, 1)).squeeze(1)
    fixed_mae = pair_mae[:, fixed_pair_index]
    oracle_mae, oracle_idx = pair_mae.min(dim=1)
    improvement = fixed_mae - selected_mae
    win = improvement > 0
    lose = improvement < 0
    metrics = {
        "selected_pair_mae": float(selected_mae.mean().item()),
        "selected_pair_mse": float(selected_mse.mean().item()),
        "fixed_pair_mae": float(fixed_mae.mean().item()),
        "oracle_pair_mae": float(oracle_mae.mean().item()),
        "regret_to_oracle_pair": float((selected_mae - oracle_mae).mean().item()),
        "improvement_over_fixed_pair": float((fixed_mae.mean() - selected_mae.mean()).item()),
        "switch_win_rate_vs_fixed": float(win.to(torch.float32).mean().item() * 100.0),
        "mean_improvement_on_winning_windows": float(improvement[win].mean().item()) if bool(win.any()) else 0.0,
        "mean_harm_on_losing_windows": float((-improvement[lose]).mean().item()) if bool(lose.any()) else 0.0,
        "exact_best_pair_accuracy": float((selected == hard_targets).to(torch.float32).mean().item() * 100.0),
        "top_two_pair_coverage": float((top2 == hard_targets.view(-1, 1)).any(dim=1).to(torch.float32).mean().item() * 100.0),
        "top_three_pair_coverage": float((top3 == hard_targets.view(-1, 1)).any(dim=1).to(torch.float32).mean().item() * 100.0),
        "cross_entropy": float(F.cross_entropy(logits, hard_targets).item()),
        "selected_indices": selected.detach().cpu(),
        "probabilities": probs.detach().cpu(),
        "oracle_indices": oracle_idx.detach().cpu(),
        "selected_mae": selected_mae.detach().cpu(),
        "fixed_mae": fixed_mae.detach().cpu(),
        "actual_improvement": improvement.detach().cpu(),
    }
    if soft_targets is not None:
        metrics["soft_target_kl_divergence"] = float(F.kl_div(F.log_softmax(logits, dim=1), soft_targets, reduction="batchmean").item())
    else:
        metrics["soft_target_kl_divergence"] = ""
    return metrics


@torch.no_grad()
def run_model(model: nn.Module, histories: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    model.eval()
    logits = []
    for start in range(0, histories.shape[0], batch_size):
        batch = histories[start : start + batch_size].to(device=device, dtype=torch.float32)
        logits.append(model(batch).cpu())
    return torch.cat(logits, dim=0)


def train_one_seed(
    seed: int,
    config: PairSelectorConfig,
    train_cache: Mapping[str, Any],
    val_cache: Mapping[str, Any],
    report: Mapping[str, Any],
    output_root: Path,
    results_root: Path,
    pairs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    set_seed(seed)
    device = torch.device(config.device)
    train_pair_mae, train_pair_mse = pair_error_matrices(train_cache, pairs)
    val_pair_mae, val_pair_mse = pair_error_matrices(val_cache, pairs)
    train_dataset = PairSelectorDataset(train_cache, train_pair_mae, config.pair_target, config.pair_temperature)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, generator=generator, num_workers=0)

    model = HistoryPairSelector(
        input_len=config.input_len,
        num_features=config.num_features,
        hidden_dim=config.hidden_dim,
        num_pairs=len(pairs),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    fixed_pair_index = pair_name_to_index(FIXED_PAIR_NAME, pairs)
    val_hard = val_pair_mae.argmin(dim=1)
    val_soft = soft_pair_targets(val_pair_mae, config.pair_temperature)
    best_state = None
    best_metrics: Optional[dict[str, Any]] = None
    best_epoch = 0
    best_val_mae = math.inf
    stale_epochs = 0
    history_rows = []

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            history = batch["history"].to(device)
            batch_on_device = {
                "hard_target": batch["hard_target"].to(device),
            }
            if config.pair_target == "soft":
                batch_on_device["soft_target"] = batch["soft_target"].to(device)
            logits = model(history)
            loss = selector_loss(logits, batch_on_device, config.pair_target)
            loss.backward()
            if config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()
            batch_size = int(history.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size

        val_logits = run_model(model, val_cache["histories"], config.batch_size, device)
        val_metrics = evaluate_logits(val_logits, val_pair_mae, val_pair_mse, fixed_pair_index, val_hard, val_soft)
        train_loss = total_loss / max(total_count, 1)
        history_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "router_val_selected_pair_mae": val_metrics["selected_pair_mae"],
            "router_val_selected_pair_mse": val_metrics["selected_pair_mse"],
            "router_val_exact_best_pair_accuracy": val_metrics["exact_best_pair_accuracy"],
        })
        if val_metrics["selected_pair_mae"] < best_val_mae - 1e-12:
            best_val_mae = val_metrics["selected_pair_mae"]
            best_epoch = epoch
            stale_epochs = 0
            best_metrics = val_metrics
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1
        if stale_epochs >= config.patience:
            break

    if best_state is None or best_metrics is None:
        raise RuntimeError("Training produced no checkpoint")
    model.load_state_dict(best_state)
    train_logits = run_model(model, train_cache["histories"], config.batch_size, device)
    val_logits = run_model(model, val_cache["histories"], config.batch_size, device)
    train_metrics = evaluate_logits(
        train_logits,
        train_pair_mae,
        train_pair_mse,
        fixed_pair_index,
        train_pair_mae.argmin(dim=1),
        soft_pair_targets(train_pair_mae, config.pair_temperature),
    )
    val_metrics = evaluate_logits(val_logits, val_pair_mae, val_pair_mse, fixed_pair_index, val_hard, val_soft)

    seed_output_dir = output_root / f"seed_{seed}"
    seed_results_dir = results_root / f"seed_{seed}"
    seed_output_dir.mkdir(parents=True, exist_ok=True)
    seed_results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_state_hash = state_dict_hash(best_state)
    checkpoint = {
        "seed": seed,
        "dataset": "ETTh2",
        "input_len": config.input_len,
        "horizon": config.horizon,
        "expert_order": list(EXPECTED_EXPERTS),
        "pair_class_order": list(pairs),
        "cache_hashes": {
            "router_train": EXPECTED_HASHES["router_train"],
            "router_val": EXPECTED_HASHES["router_val"],
        },
        "scaler_hash": EXPECTED_HASHES["scaler"],
        "model_configuration": model.config_dict(),
        "training_configuration": asdict(config),
        "pair_target_mode": config.pair_target,
        "temperature": config.pair_temperature,
        "best_epoch": best_epoch,
        "router_validation_mae": val_metrics["selected_pair_mae"],
        "checkpoint_hash": checkpoint_state_hash,
        "checkpoint_hash_type": "sha256 over sorted state_dict tensors",
        "state_dict": best_state,
        "checkpoint_hashes": report["checkpoint_hashes"],
    }
    checkpoint_path = seed_output_dir / "best_pair_selector.pt"
    torch.save(checkpoint, checkpoint_path)
    checkpoint_file_hash = sha256_file(checkpoint_path)

    confidence_rows = validation_confidence_rows(
        val_cache,
        val_logits,
        val_pair_mae,
        fixed_pair_index,
        pairs,
    )
    confidence_path = results_root / f"validation_confidence_seed_{seed}.csv"
    write_csv(
        confidence_path,
        confidence_rows,
        (
            "row",
            "absolute_window_start",
            "sample_index",
            "max_predicted_probability",
            "probability_margin",
            "logit_margin",
            "entropy",
            "fixed_pair_probability",
            "predicted_minus_fixed_probability",
            "selected_pair",
            "fixed_pair",
            "oracle_pair",
            "selected_pair_error",
            "fixed_pair_error",
            "actual_improvement_from_switching",
        ),
    )
    write_csv(
        seed_results_dir / "training_history.csv",
        history_rows,
        ("epoch", "train_loss", "router_val_selected_pair_mae", "router_val_selected_pair_mse", "router_val_exact_best_pair_accuracy"),
    )
    seed_summary = {
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_ran": len(history_rows),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_state_hash": checkpoint_state_hash,
        "checkpoint_file_hash": checkpoint_file_hash,
        "confidence_path": str(confidence_path),
        "train": metrics_for_json(train_metrics, pairs),
        "validation": metrics_for_json(val_metrics, pairs),
        "margin_groups": margin_group_metrics(val_pair_mae, val_pair_mse, val_logits, fixed_pair_index, pairs),
        "dense_pair_probability_mixture": dense_mixture_metrics(val_cache, val_logits, val_pair_mae, fixed_pair_index, pairs),
    }
    (seed_results_dir / "seed_summary.json").write_text(json.dumps(seed_summary, indent=2, default=json_default), encoding="utf-8")
    return seed_summary


def state_dict_hash(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def metrics_for_json(metrics: Mapping[str, Any], pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected = metrics["selected_indices"]
    oracle = metrics["oracle_indices"]
    selection_counts = torch.bincount(selected, minlength=len(pairs)).to(torch.float32)
    oracle_counts = torch.bincount(oracle, minlength=len(pairs)).to(torch.float32)
    total = float(selected.numel())
    keep = {
        key: value
        for key, value in metrics.items()
        if key not in {"selected_indices", "probabilities", "oracle_indices", "selected_mae", "fixed_mae", "actual_improvement"}
    }
    keep["class_selection_distribution"] = {
        pairs[i]["pair"]: {
            "count": int(selection_counts[i].item()),
            "percentage": float(selection_counts[i].item() * 100.0 / total),
        }
        for i in range(len(pairs))
    }
    keep["oracle_pair_class_distribution"] = {
        pairs[i]["pair"]: {
            "count": int(oracle_counts[i].item()),
            "percentage": float(oracle_counts[i].item() * 100.0 / total),
        }
        for i in range(len(pairs))
    }
    return keep


def validation_confidence_rows(
    val_cache: Mapping[str, Any],
    logits: torch.Tensor,
    pair_mae: torch.Tensor,
    fixed_pair_index: int,
    pairs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    probs = torch.softmax(logits, dim=1)
    sorted_probs, sorted_idx = torch.sort(probs, dim=1, descending=True)
    sorted_logits, _ = torch.sort(logits, dim=1, descending=True)
    selected = sorted_idx[:, 0]
    oracle = pair_mae.argmin(dim=1)
    selected_error = pair_mae.gather(1, selected.view(-1, 1)).squeeze(1)
    fixed_error = pair_mae[:, fixed_pair_index]
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)
    rows = []
    for row in range(int(logits.shape[0])):
        rows.append({
            "row": row,
            "absolute_window_start": int(val_cache["absolute_window_starts"][row].item()),
            "sample_index": int(val_cache["sample_indices"][row].item()),
            "max_predicted_probability": float(sorted_probs[row, 0].item()),
            "probability_margin": float((sorted_probs[row, 0] - sorted_probs[row, 1]).item()),
            "logit_margin": float((sorted_logits[row, 0] - sorted_logits[row, 1]).item()),
            "entropy": float(entropy[row].item()),
            "fixed_pair_probability": float(probs[row, fixed_pair_index].item()),
            "predicted_minus_fixed_probability": float((probs[row, selected[row]] - probs[row, fixed_pair_index]).item()),
            "selected_pair": pairs[int(selected[row].item())]["pair"],
            "fixed_pair": pairs[fixed_pair_index]["pair"],
            "oracle_pair": pairs[int(oracle[row].item())]["pair"],
            "selected_pair_error": float(selected_error[row].item()),
            "fixed_pair_error": float(fixed_error[row].item()),
            "actual_improvement_from_switching": float((fixed_error[row] - selected_error[row]).item()),
        })
    return rows


def margin_group_metrics(
    val_pair_mae: torch.Tensor,
    val_pair_mse: torch.Tensor,
    logits: torch.Tensor,
    fixed_pair_index: int,
    pairs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    sorted_errors = torch.sort(val_pair_mae, dim=1).values
    margins = sorted_errors[:, 1] - sorted_errors[:, 0]
    probs = torch.softmax(logits, dim=1)
    selected = probs.argmax(dim=1)
    oracle = val_pair_mae.argmin(dim=1)
    selected_mae = val_pair_mae.gather(1, selected.view(-1, 1)).squeeze(1)
    selected_mse = val_pair_mse.gather(1, selected.view(-1, 1)).squeeze(1)
    oracle_mae = val_pair_mae.min(dim=1).values
    fixed_mae = val_pair_mae[:, fixed_pair_index]
    rows = []
    for label, lower, upper in MARGIN_BINS:
        mask = torch.ones_like(margins, dtype=torch.bool)
        if lower is not None:
            mask &= margins > lower
        if upper is not None:
            mask &= margins <= upper
        count = int(mask.sum().item())
        if count == 0:
            rows.append({"margin_group": label, "count": 0})
            continue
        rows.append({
            "margin_group": label,
            "count": count,
            "exact_pair_accuracy": float((selected[mask] == oracle[mask]).to(torch.float32).mean().item() * 100.0),
            "selected_pair_mae": float(selected_mae[mask].mean().item()),
            "selected_pair_mse": float(selected_mse[mask].mean().item()),
            "regret_to_oracle_pair": float((selected_mae[mask] - oracle_mae[mask]).mean().item()),
            "confidence": float(probs[mask].max(dim=1).values.mean().item()),
            "switch_win_rate_against_fixed": float((fixed_mae[mask] > selected_mae[mask]).to(torch.float32).mean().item() * 100.0),
        })
    return rows


def dense_mixture_metrics(
    val_cache: Mapping[str, Any],
    logits: torch.Tensor,
    val_pair_mae: torch.Tensor,
    fixed_pair_index: int,
    pairs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    probs = torch.softmax(logits, dim=1)
    pair_predictions = pair_prediction_stack(val_cache, pairs)
    dense_prediction = torch.einsum("nhtp,np->nht", pair_predictions, probs)
    mae, mse = per_window_error(dense_prediction, val_cache["targets"], val_cache["target_masks"])
    oracle = val_pair_mae.min(dim=1).values
    fixed = val_pair_mae[:, fixed_pair_index]
    return {
        "mae": float(mae.mean().item()),
        "mse": float(mse.mean().item()),
        "regret_to_oracle_pair": float((mae - oracle).mean().item()),
        "improvement_over_fixed_pair": float((fixed.mean() - mae.mean()).item()),
        "average_experts_used": 5.0,
        "diagnostic_only": True,
    }


def fixed_and_random_baselines(
    val_pair_mae: torch.Tensor,
    val_pair_mse: torch.Tensor,
    train_pair_mae: torch.Tensor,
    fixed_pair_index: int,
    pairs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    oracle = val_pair_mae.min(dim=1).values
    fixed_mae = val_pair_mae[:, fixed_pair_index]
    fixed_mse = val_pair_mse[:, fixed_pair_index]
    majority_index = int(torch.mode(train_pair_mae.argmin(dim=1)).values.item())
    majority_mae = val_pair_mae[:, majority_index]
    majority_mse = val_pair_mse[:, majority_index]
    random_maes = []
    random_mses = []
    for seed in range(100):
        generator = torch.Generator().manual_seed(1000 + seed)
        selected = torch.randint(0, len(pairs), (val_pair_mae.shape[0],), generator=generator)
        random_maes.append(val_pair_mae.gather(1, selected.view(-1, 1)).mean().item())
        random_mses.append(val_pair_mse.gather(1, selected.view(-1, 1)).mean().item())
    return {
        "fixed_pair": {
            "pair": pairs[fixed_pair_index]["pair"],
            "mae": float(fixed_mae.mean().item()),
            "mse": float(fixed_mse.mean().item()),
            "regret_to_oracle_pair": float((fixed_mae - oracle).mean().item()),
            "average_experts_used": 2.0,
        },
        "majority_oracle_pair_from_router_train": {
            "pair": pairs[majority_index]["pair"],
            "mae": float(majority_mae.mean().item()),
            "mse": float(majority_mse.mean().item()),
            "regret_to_oracle_pair": float((majority_mae - oracle).mean().item()),
            "average_experts_used": 2.0,
        },
        "random_pair_selector": {
            "mae_mean": float(np.mean(random_maes)),
            "mae_std": float(np.std(random_maes)),
            "mse_mean": float(np.mean(random_mses)),
            "mse_std": float(np.std(random_mses)),
            "average_experts_used": 2.0,
            "num_random_seeds": 100,
        },
        "per_window_oracle_pair": {
            "mae": float(oracle.mean().item()),
            "mse": float(val_pair_mse.gather(1, val_pair_mae.argmin(dim=1).view(-1, 1)).mean().item()),
            "average_experts_used": 2.0,
            "diagnostic_only": True,
        },
    }


def checkpoint_metadata_is_valid(checkpoint: Mapping[str, Any]) -> None:
    if checkpoint["dataset"] != "ETTh2":
        raise ValueError("checkpoint dataset mismatch")
    if int(checkpoint["input_len"]) != 96 or int(checkpoint["horizon"]) != 12:
        raise ValueError("checkpoint horizon mismatch")
    if tuple(checkpoint["expert_order"]) != EXPECTED_EXPERTS:
        raise ValueError("checkpoint expert ordering mismatch")
    if checkpoint["cache_hashes"]["router_train"] != EXPECTED_HASHES["router_train"]:
        raise ValueError("checkpoint router_train hash mismatch")
    if checkpoint["cache_hashes"]["router_val"] != EXPECTED_HASHES["router_val"]:
        raise ValueError("checkpoint router_val hash mismatch")
    if checkpoint["scaler_hash"] != EXPECTED_HASHES["scaler"]:
        raise ValueError("checkpoint scaler hash mismatch")


def aggregate_results(
    seed_summaries: Sequence[Mapping[str, Any]],
    val_cache: Mapping[str, Any],
    results_root: Path,
    pairs: Sequence[Mapping[str, Any]],
    baselines: Mapping[str, Any],
) -> dict[str, Any]:
    val_rows = []
    for summary in seed_summaries:
        metrics = summary["validation"]
        val_rows.append({
            "seed": summary["seed"],
            "best_epoch": summary["best_epoch"],
            "epochs_ran": summary["epochs_ran"],
            "selected_pair_mae": metrics["selected_pair_mae"],
            "selected_pair_mse": metrics["selected_pair_mse"],
            "regret_to_oracle_pair": metrics["regret_to_oracle_pair"],
            "improvement_over_fixed_pair": metrics["improvement_over_fixed_pair"],
            "switch_win_rate_vs_fixed": metrics["switch_win_rate_vs_fixed"],
            "mean_improvement_on_winning_windows": metrics["mean_improvement_on_winning_windows"],
            "mean_harm_on_losing_windows": metrics["mean_harm_on_losing_windows"],
            "exact_best_pair_accuracy": metrics["exact_best_pair_accuracy"],
            "top_two_pair_coverage": metrics["top_two_pair_coverage"],
            "top_three_pair_coverage": metrics["top_three_pair_coverage"],
            "cross_entropy": metrics["cross_entropy"],
            "soft_target_kl_divergence": metrics["soft_target_kl_divergence"],
            "checkpoint_state_hash": summary["checkpoint_state_hash"],
            "checkpoint_file_hash": summary["checkpoint_file_hash"],
        })
    write_csv(
        results_root / "per_seed_results.csv",
        val_rows,
        (
            "seed",
            "best_epoch",
            "epochs_ran",
            "selected_pair_mae",
            "selected_pair_mse",
            "regret_to_oracle_pair",
            "improvement_over_fixed_pair",
            "switch_win_rate_vs_fixed",
            "mean_improvement_on_winning_windows",
            "mean_harm_on_losing_windows",
            "exact_best_pair_accuracy",
            "top_two_pair_coverage",
            "top_three_pair_coverage",
            "cross_entropy",
            "soft_target_kl_divergence",
            "checkpoint_state_hash",
            "checkpoint_file_hash",
        ),
    )

    metric_names = [
        "selected_pair_mae",
        "selected_pair_mse",
        "regret_to_oracle_pair",
        "improvement_over_fixed_pair",
        "exact_best_pair_accuracy",
        "top_two_pair_coverage",
        "top_three_pair_coverage",
    ]
    aggregate_rows = []
    for metric in metric_names:
        values = np.array([float(row[metric]) for row in val_rows], dtype=float)
        aggregate_rows.append({
            "metric": metric,
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
        })
    write_csv(results_root / "aggregate_results.csv", aggregate_rows, ("metric", "mean", "std", "min", "max"))

    confidence_frames = []
    for summary in seed_summaries:
        rows = read_csv_dicts(Path(summary["confidence_path"]))
        confidence_frames.append(rows)
    cross_seed = cross_seed_agreement(confidence_frames, pairs)
    confidence_separation = confidence_signal_separation(confidence_frames)
    write_csv(
        results_root / "cross_seed_agreement.csv",
        cross_seed["rows"],
        (
            "row",
            "absolute_window_start",
            "agreement_fraction",
            "all_five_agree",
            "at_least_four_agree",
            "selected_pair_unique_count",
            "max_probability_variance",
            "selected_pair_error_variance",
            "modal_pair",
        ),
    )
    aggregate_summary = {
        "seeds": [summary["seed"] for summary in seed_summaries],
        "pair_class_order": list(pairs),
        "baselines": baselines,
        "per_seed": val_rows,
        "aggregate": aggregate_rows,
        "selection_distribution_mean": mean_selection_distribution(seed_summaries, pairs),
        "confidence_distribution": confidence_distribution(confidence_frames),
        "confidence_signal_separation": confidence_separation,
        "margin_group_aggregate": aggregate_margin_groups(seed_summaries),
        "cross_seed_agreement": cross_seed["summary"],
        "validation_source_alignment": {
            "num_validation_windows": int(val_cache["absolute_window_starts"].shape[0]),
            "first_absolute_window_start": int(val_cache["absolute_window_starts"][0].item()),
            "last_absolute_window_start": int(val_cache["absolute_window_starts"][-1].item()),
        },
        "leakage_assertions": {
            "training_loader_split": "router_train",
            "router_validation_targets_in_training_dataloader": False,
            "test_arrays_loaded": False,
            "test_cache_created": False,
            "confidence_gate_trained": False,
            "forecasting_experts_retrained": False,
        },
    }
    (results_root / "aggregate_summary.json").write_text(json.dumps(aggregate_summary, indent=2, default=json_default), encoding="utf-8")
    write_report(results_root / "pair_selector_report.md", aggregate_summary)
    return aggregate_summary


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def cross_seed_agreement(confidence_frames: Sequence[Sequence[Mapping[str, str]]], pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    num_rows = len(confidence_frames[0])
    rows = []
    for row_idx in range(num_rows):
        selected = [frame[row_idx]["selected_pair"] for frame in confidence_frames]
        counts = {pair["pair"]: selected.count(pair["pair"]) for pair in pairs}
        modal_pair = max(counts.items(), key=lambda item: item[1])[0]
        max_count = counts[modal_pair]
        max_probs = np.array([float(frame[row_idx]["max_predicted_probability"]) for frame in confidence_frames], dtype=float)
        selected_errors = np.array([float(frame[row_idx]["selected_pair_error"]) for frame in confidence_frames], dtype=float)
        starts = {frame[row_idx]["absolute_window_start"] for frame in confidence_frames}
        if len(starts) != 1:
            raise ValueError("validation confidence source indices are misaligned")
        rows.append({
            "row": row_idx,
            "absolute_window_start": next(iter(starts)),
            "agreement_fraction": max_count / len(confidence_frames),
            "all_five_agree": max_count == len(confidence_frames),
            "at_least_four_agree": max_count >= len(confidence_frames) - 1,
            "selected_pair_unique_count": len(set(selected)),
            "max_probability_variance": float(max_probs.var()),
            "selected_pair_error_variance": float(selected_errors.var()),
            "modal_pair": modal_pair,
        })
    agreement = np.array([float(row["agreement_fraction"]) for row in rows], dtype=float)
    return {
        "rows": rows,
        "summary": {
            "mean_agreement_rate": float(agreement.mean()),
            "all_five_agree_percentage": float(np.mean([bool(row["all_five_agree"]) for row in rows]) * 100.0),
            "at_least_four_agree_percentage": float(np.mean([bool(row["at_least_four_agree"]) for row in rows]) * 100.0),
            "mean_max_probability_variance": float(np.mean([float(row["max_probability_variance"]) for row in rows])),
            "mean_selected_pair_error_variance": float(np.mean([float(row["selected_pair_error_variance"]) for row in rows])),
        },
    }


def mean_selection_distribution(seed_summaries: Sequence[Mapping[str, Any]], pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for pair in pairs:
        values = [
            float(summary["validation"]["class_selection_distribution"][pair["pair"]]["percentage"])
            for summary in seed_summaries
        ]
        result[pair["pair"]] = {"mean_percentage": float(np.mean(values)), "std_percentage": float(np.std(values))}
    return result


def confidence_distribution(confidence_frames: Sequence[Sequence[Mapping[str, str]]]) -> dict[str, Any]:
    values = {
        "max_predicted_probability": [],
        "probability_margin": [],
        "logit_margin": [],
        "entropy": [],
        "predicted_minus_fixed_probability": [],
    }
    for frame in confidence_frames:
        for row in frame:
            for key in values:
                values[key].append(float(row[key]))
    return {
        key: {
            "mean": float(np.mean(items)),
            "std": float(np.std(items)),
            "p25": float(np.quantile(items, 0.25)),
            "median": float(np.quantile(items, 0.5)),
            "p75": float(np.quantile(items, 0.75)),
        }
        for key, items in values.items()
    }


def binary_auc(scores: np.ndarray, positives: np.ndarray) -> float:
    positive_scores = scores[positives]
    negative_scores = scores[~positives]
    if positive_scores.size == 0 or negative_scores.size == 0:
        return float("nan")
    comparisons = positive_scores[:, None] - negative_scores[None, :]
    wins = (comparisons > 0).mean()
    ties = (comparisons == 0).mean()
    return float(wins + 0.5 * ties)


def confidence_signal_separation(confidence_frames: Sequence[Sequence[Mapping[str, str]]]) -> list[dict[str, Any]]:
    signals = {
        "max_predicted_probability": [],
        "probability_margin": [],
        "logit_margin": [],
        "negative_entropy": [],
        "predicted_minus_fixed_probability": [],
        "fixed_pair_probability_negative": [],
    }
    improvements = []
    for frame in confidence_frames:
        for row in frame:
            signals["max_predicted_probability"].append(float(row["max_predicted_probability"]))
            signals["probability_margin"].append(float(row["probability_margin"]))
            signals["logit_margin"].append(float(row["logit_margin"]))
            signals["negative_entropy"].append(-float(row["entropy"]))
            signals["predicted_minus_fixed_probability"].append(float(row["predicted_minus_fixed_probability"]))
            signals["fixed_pair_probability_negative"].append(-float(row["fixed_pair_probability"]))
            improvements.append(float(row["actual_improvement_from_switching"]))
    improvement_array = np.array(improvements, dtype=float)
    switched_mask = np.abs(improvement_array) > 1e-12
    positives = improvement_array[switched_mask] > 0
    rows = []
    for name, values in signals.items():
        signal = np.array(values, dtype=float)[switched_mask]
        if signal.size == 0 or positives.sum() == 0 or (~positives).sum() == 0:
            rows.append({
                "signal": name,
                "auc_helpful_vs_harmful": "",
                "helpful_mean": "",
                "harmful_mean": "",
                "mean_gap_helpful_minus_harmful": "",
                "num_helpful_switches": int(positives.sum()),
                "num_harmful_switches": int((~positives).sum()),
            })
            continue
        helpful_mean = float(signal[positives].mean())
        harmful_mean = float(signal[~positives].mean())
        rows.append({
            "signal": name,
            "auc_helpful_vs_harmful": binary_auc(signal, positives),
            "helpful_mean": helpful_mean,
            "harmful_mean": harmful_mean,
            "mean_gap_helpful_minus_harmful": helpful_mean - harmful_mean,
            "num_helpful_switches": int(positives.sum()),
            "num_harmful_switches": int((~positives).sum()),
        })
    rows.sort(key=lambda row: float(row["auc_helpful_vs_harmful"]) if row["auc_helpful_vs_harmful"] != "" else -1.0, reverse=True)
    return rows


def aggregate_margin_groups(seed_summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_group: dict[str, list[Mapping[str, Any]]] = {}
    for summary in seed_summaries:
        for row in summary["margin_groups"]:
            by_group.setdefault(row["margin_group"], []).append(row)
    aggregate = []
    for label, rows in by_group.items():
        count = int(rows[0].get("count", 0))
        item = {"margin_group": label, "count": count}
        if count > 0:
            for metric in (
                "exact_pair_accuracy",
                "selected_pair_mae",
                "selected_pair_mse",
                "regret_to_oracle_pair",
                "confidence",
                "switch_win_rate_against_fixed",
            ):
                values = [float(row[metric]) for row in rows if metric in row]
                item[f"{metric}_mean"] = float(np.mean(values))
                item[f"{metric}_std"] = float(np.std(values))
        aggregate.append(item)
    order = {label: index for index, (label, _, _) in enumerate(MARGIN_BINS)}
    aggregate.sort(key=lambda row: order[row["margin_group"]])
    return aggregate


def write_report(path: Path, summary: Mapping[str, Any]) -> None:
    aggregate = {row["metric"]: row for row in summary["aggregate"]}
    baselines = summary["baselines"]
    mean_mae = aggregate["selected_pair_mae"]["mean"]
    fixed_mae = baselines["fixed_pair"]["mae"]
    mean_improvement = aggregate["improvement_over_fixed_pair"]["mean"]
    confidence_rows = summary["confidence_signal_separation"]
    best_signal = confidence_rows[0] if confidence_rows else {"signal": "", "auc_helpful_vs_harmful": ""}
    high_margin = next(row for row in summary["margin_group_aggregate"] if row["margin_group"] == ">0.025")
    low_margin = next(row for row in summary["margin_group_aggregate"] if row["margin_group"] == "<=0.005")
    lines = [
        "# ETTh2 Pair Selector Report",
        "",
        "## Validation Summary",
        "",
        f"- Fixed pair: `{baselines['fixed_pair']['pair']}` MAE `{fixed_mae:.6f}`.",
        f"- Predicted-pair mean MAE: `{mean_mae:.6f}` +/- `{aggregate['selected_pair_mae']['std']:.6f}`.",
        f"- Mean improvement over fixed pair: `{mean_improvement:.6f}`.",
        f"- Exact pair accuracy: `{aggregate['exact_best_pair_accuracy']['mean']:.2f}%`.",
        f"- Top-two pair coverage: `{aggregate['top_two_pair_coverage']['mean']:.2f}%`.",
        f"- Cross-seed mean agreement: `{summary['cross_seed_agreement']['mean_agreement_rate']:.3f}`.",
        f"- All five seeds agree on `{summary['cross_seed_agreement']['all_five_agree_percentage']:.2f}%` of validation windows.",
        "",
        "## Decision Answers",
        "",
        f"1. Always-use predicted pair beats fixed pair on average: `{mean_mae < fixed_mae}`.",
        f"2. Stability across five seeds: std MAE `{aggregate['selected_pair_mae']['std']:.6f}`.",
        f"3. High-margin selected-pair MAE `{high_margin.get('selected_pair_mae_mean', float('nan')):.6f}` versus low-margin `{low_margin.get('selected_pair_mae_mean', float('nan')):.6f}`.",
        f"4. Best diagnostic confidence separator: `{best_signal['signal']}` with AUC `{best_signal['auc_helpful_vs_harmful']}`.",
        f"5. Confidence stability: mean max-probability variance `{summary['cross_seed_agreement']['mean_max_probability_variance']:.6f}`.",
        f"6. Evidence for a constrained gate: `{mean_improvement > 0}` based on forecast MAE, pending confidence-separation analysis.",
        "",
        "## Leakage",
        "",
        "Only router_train/router_val caches, cache validation report, and pair-potential summary were loaded. No ETTh2 test arrays or test cache were created.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_seeds(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router-train-cache", default="cache/costarts_fresh/ETTh2_96_12/router_train_cache.pt")
    parser.add_argument("--router-val-cache", default="cache/costarts_fresh/ETTh2_96_12/router_val_cache.pt")
    parser.add_argument("--validation-report", default="cache/costarts_fresh/ETTh2_96_12/cache_validation_report.json")
    parser.add_argument("--pair-potential-summary", default="results/router_summary/costarts_fresh/ETTh2_96_12/pair_potential/pair_potential_summary.json")
    parser.add_argument("--checkpoint-root", default="checkpoints/costarts_fresh/ETTh2_96_12/pair_selector")
    parser.add_argument("--results-root", default="results/router_summary/costarts_fresh/ETTh2_96_12/pair_selector")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--pair-target", choices=("hard", "soft"), default="soft")
    parser.add_argument("--pair-temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> dict[str, Any]:
    args = parse_args(argv)
    seeds = parse_seeds(args.seeds)
    if seeds != list(DEFAULT_SEEDS):
        raise ValueError(f"This locked local run expects seeds {DEFAULT_SEEDS}, got {seeds}")
    config = PairSelectorConfig(
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        pair_target=args.pair_target,
        pair_temperature=args.pair_temperature,
        device=args.device,
    )
    if config.pair_target == "soft" and abs(config.pair_temperature - DEFAULT_TEMPERATURE) > 1e-12:
        raise ValueError("Use fixed soft-target temperature 0.01 for this initial five-seed run")
    train_cache, val_cache, report, _ = load_inputs(args)
    pairs = pair_class_order()
    results_root = Path(args.results_root)
    checkpoint_root = Path(args.checkpoint_root)
    results_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    (results_root / "pair_class_order.json").write_text(json.dumps(pairs, indent=2), encoding="utf-8")
    train_pair_mae, train_pair_mse = pair_error_matrices(train_cache, pairs)
    val_pair_mae, val_pair_mse = pair_error_matrices(val_cache, pairs)
    baselines = fixed_and_random_baselines(
        val_pair_mae,
        val_pair_mse,
        train_pair_mae,
        pair_name_to_index(FIXED_PAIR_NAME, pairs),
        pairs,
    )
    seed_summaries = []
    for seed in seeds:
        seed_summaries.append(
            train_one_seed(seed, config, train_cache, val_cache, report, checkpoint_root, results_root, pairs)
        )
    summary = aggregate_results(seed_summaries, val_cache, results_root, pairs, baselines)
    forbidden = [
        Path("cache/costarts_fresh/ETTh2_96_12/test_cache.pt"),
        Path("cache/costarts_fresh/ETTh2_96_12/locked_test_cache.pt"),
    ]
    created = [str(path) for path in forbidden if path.exists()]
    if created:
        raise RuntimeError(f"Forbidden test cache exists: {created}")
    return summary


if __name__ == "__main__":
    main()
