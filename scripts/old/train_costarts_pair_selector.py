"""Train a history-only COSTARTS selector over expert pairs.

The selector uses frozen cached expert forecasts.  For each causal router
window, it scores every unordered two-expert pair by the equal-average forecast
error and trains a small history encoder to predict a pair distribution.
Checkpoint selection is based on chronological router-validation MAE.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


DEFAULT_TRAIN_CACHE = "cache/costarts_router_train_cache.pt"
DEFAULT_VAL_CACHE = "cache/costarts_router_val_cache.pt"
DEFAULT_TEST_CACHE = ""
DEFAULT_OUTPUT_DIR = "checkpoints/costarts_pair_selector"
DEFAULT_RESULTS_DIR = "results/router_summary/costarts_pair_selector"
DEFAULT_FINAL_COMPARISON_JSON = "results/router_summary/costarts_subset_utility/final_comparison.json"


@dataclass
class PairSelectorTrainingConfig:
    train_cache_path: str = DEFAULT_TRAIN_CACHE
    val_cache_path: str = DEFAULT_VAL_CACHE
    test_cache_path: str = DEFAULT_TEST_CACHE
    output_dir: str = DEFAULT_OUTPUT_DIR
    results_dir: str = DEFAULT_RESULTS_DIR
    final_comparison_json: str = DEFAULT_FINAL_COMPARISON_JSON
    batch_size: int = 512
    max_epochs: int = 40
    patience: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    seed: int = 7
    target_temperature: float = 0.03
    hard_label_weight: float = 1.0
    embedding_dim: int = 64
    hidden_dim: int = 64
    dropout: float = 0.1
    history_encoder_type: str = "current"
    device: str = "cpu"


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
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return str(value)
    return value


def _assert_router_cache(cache: Mapping[str, Any], split_role: str) -> None:
    if cache["split_role"] != split_role:
        raise AssertionError(f"Expected {split_role} cache, got {cache['split_role']!r}.")
    required = (
        "histories",
        "targets",
        "target_masks",
        "prediction_stack",
        "error_matrix",
        "mse_matrix",
        "best_expert",
        "sample_indices",
        "expert_names",
        "num_windows",
        "input_len",
        "forecast_horizon",
        "num_features",
    )
    missing = [key for key in required if key not in cache]
    if missing:
        raise AssertionError(f"Router cache missing keys: {missing}")
    num_windows = int(cache["num_windows"])
    num_experts = len(tuple(cache["expert_names"]))
    assert tuple(cache["histories"].shape) == (num_windows, int(cache["input_len"]), int(cache["num_features"]))
    assert tuple(cache["targets"].shape) == (num_windows, int(cache["forecast_horizon"]), int(cache["num_features"]))
    assert tuple(cache["target_masks"].shape) == tuple(cache["targets"].shape)
    assert tuple(cache["prediction_stack"].shape) == (
        num_windows,
        int(cache["forecast_horizon"]),
        int(cache["num_features"]),
        num_experts,
    )
    assert tuple(cache["error_matrix"].shape) == (num_windows, num_experts)
    assert tuple(cache["mse_matrix"].shape) == (num_windows, num_experts)
    assert tuple(cache["best_expert"].shape) == (num_windows,)
    sample_indices = cache["sample_indices"].to(torch.long)
    expected = torch.arange(int(sample_indices[0]), int(sample_indices[0]) + num_windows)
    if not torch.equal(sample_indices.cpu(), expected):
        raise AssertionError("Router cache sample_indices must be chronological and contiguous.")


def _assert_cache_pair(train_cache: Mapping[str, Any], eval_cache: Mapping[str, Any], eval_role: str) -> None:
    _assert_router_cache(train_cache, "router_train")
    _assert_router_cache(eval_cache, eval_role)
    if tuple(train_cache["expert_names"]) != tuple(eval_cache["expert_names"]):
        raise AssertionError("Train/eval expert order mismatch.")
    for key in ("input_len", "forecast_horizon", "num_features"):
        if int(train_cache[key]) != int(eval_cache[key]):
            raise AssertionError(f"Train/eval cache mismatch for {key}.")


def build_pair_index(num_experts: int) -> torch.Tensor:
    """Return unordered pair indices in stable lexicographic order."""
    if num_experts < 2:
        raise ValueError("At least two experts are required.")
    return torch.tensor(
        [(left, right) for left in range(num_experts) for right in range(left + 1, num_experts)],
        dtype=torch.long,
    )


def pair_to_class_index(pair_index: torch.Tensor) -> dict[tuple[int, int], int]:
    mapping: dict[tuple[int, int], int] = {}
    for class_index, pair in enumerate(pair_index.tolist()):
        left, right = sorted((int(pair[0]), int(pair[1])))
        mapping[(left, right)] = int(class_index)
    return mapping


def pair_average_forecasts(cache: Mapping[str, Any], pair_index: torch.Tensor) -> torch.Tensor:
    prediction_stack = cache["prediction_stack"].to(torch.float32)
    pairs = pair_index.to(torch.long)
    return prediction_stack[..., pairs].mean(dim=-1)


def masked_pair_mae_mse(cache: Mapping[str, Any], pair_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pair_predictions = pair_average_forecasts(cache, pair_index)
    targets = cache["targets"].to(torch.float32).unsqueeze(-1)
    masks = cache["target_masks"].to(torch.float32).unsqueeze(-1)
    denominator = masks.sum(dim=(1, 2)).clamp_min(1.0)
    mae = (torch.abs(pair_predictions - targets) * masks).sum(dim=(1, 2)) / denominator
    mse = ((pair_predictions - targets).pow(2) * masks).sum(dim=(1, 2)) / denominator
    return mae, mse


def pair_errors_to_soft_targets(pair_mae: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if pair_mae.ndim != 2:
        raise ValueError("pair_mae must have shape [windows, pairs]")
    centered = pair_mae - pair_mae.min(dim=1, keepdim=True).values
    return torch.softmax(-centered / float(temperature), dim=1)


class CostartsPairDataset(Dataset):
    def __init__(self, cache: Mapping[str, Any], pair_index: torch.Tensor, target_temperature: float) -> None:
        self.histories = cache["histories"].to(torch.float32)
        self.pair_mae, self.pair_mse = masked_pair_mae_mse(cache, pair_index)
        self.soft_targets = pair_errors_to_soft_targets(self.pair_mae, target_temperature).to(torch.float32)
        self.best_pair = torch.argmin(self.pair_mae, dim=1).to(torch.long)

    def __len__(self) -> int:
        return int(self.histories.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "history": self.histories[index],
            "soft_targets": self.soft_targets[index],
            "best_pair": self.best_pair[index],
        }


class CostartsPairSelector(nn.Module):
    def __init__(
        self,
        *,
        input_len: int = 96,
        num_features: int = 7,
        num_pairs: int = 10,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        history_encoder_type: str = "current",
    ) -> None:
        super().__init__()
        if history_encoder_type not in {"current", "simple"}:
            raise ValueError("history_encoder_type must be 'current' or 'simple'")
        self.input_len = int(input_len)
        self.num_features = int(num_features)
        self.num_pairs = int(num_pairs)
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.history_encoder_type = str(history_encoder_type)
        if history_encoder_type == "simple":
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
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout),
        )
        self.pair_head = nn.Linear(embedding_dim, num_pairs)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if tuple(history.shape[1:]) != (self.input_len, self.num_features):
            raise AssertionError("history shape does not match router cache configuration")
        encoded = self.history_encoder(history.transpose(1, 2)).squeeze(-1)
        return self.pair_head(self.projection(encoded))

    def config_dict(self) -> dict[str, Any]:
        return {
            "input_len": self.input_len,
            "num_features": self.num_features,
            "num_pairs": self.num_pairs,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "history_encoder_type": self.history_encoder_type,
        }


def pair_selector_loss(
    logits: torch.Tensor,
    soft_targets: torch.Tensor,
    best_pair: torch.Tensor,
    hard_label_weight: float,
) -> torch.Tensor:
    soft_loss = F.kl_div(F.log_softmax(logits, dim=1), soft_targets, reduction="batchmean")
    if hard_label_weight <= 0:
        return soft_loss
    hard_loss = F.cross_entropy(logits, best_pair)
    return soft_loss + float(hard_label_weight) * hard_loss


def _selected_pair_prediction(cache: Mapping[str, Any], selected_pair: torch.Tensor) -> torch.Tensor:
    stack = cache["prediction_stack"].to(torch.float32)
    rows = torch.arange(stack.shape[0])
    first = stack[rows, :, :, selected_pair[:, 0]]
    second = stack[rows, :, :, selected_pair[:, 1]]
    return (first + second) * 0.5


def _mae_mse(prediction: torch.Tensor, cache: Mapping[str, Any]) -> tuple[float, float]:
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.float32)
    denominator = mask.sum().clamp_min(1.0)
    mae = (torch.abs(prediction - target) * mask).sum() / denominator
    mse = ((prediction - target).pow(2) * mask).sum() / denominator
    return float(mae), float(mse)


@torch.no_grad()
def evaluate_pair_selector(
    model: CostartsPairSelector,
    cache: Mapping[str, Any],
    pair_index: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    selected_classes: list[torch.Tensor] = []
    start = time.perf_counter()
    for offset in range(0, int(cache["num_windows"]), batch_size):
        history = cache["histories"][offset : offset + batch_size].to(device)
        logits = model(history)
        selected_classes.append(torch.argmax(logits, dim=1).detach().cpu())
    latency = time.perf_counter() - start
    selected_class = torch.cat(selected_classes, dim=0).to(torch.long)
    selected_pair = pair_index[selected_class].to(torch.long)
    prediction = _selected_pair_prediction(cache, selected_pair)
    mae, mse = _mae_mse(prediction, cache)
    pair_mae, pair_mse = masked_pair_mae_mse(cache, pair_index)
    best_pair = torch.argmin(pair_mae, dim=1).to(torch.long)
    best_expert = cache["best_expert"].to(torch.long)
    best_pair_accuracy = float((selected_class == best_pair).to(torch.float32).mean())
    top2_coverage = float((selected_pair == best_expert[:, None]).any(dim=1).to(torch.float32).mean())
    oracle_pair_mae = float(pair_mae.min(dim=1).values.mean())
    oracle_pair_mse = float(pair_mse.gather(1, best_pair.view(-1, 1)).mean())
    return {
        "mae": mae,
        "mse": mse,
        "average_experts_queried": 2.0,
        "best_pair_accuracy": best_pair_accuracy,
        "best_individual_expert_top2_coverage": top2_coverage,
        "oracle_pair_mae": oracle_pair_mae,
        "oracle_pair_mse": oracle_pair_mse,
        "selected_pair_class": selected_class,
        "selected_pair_indices": selected_pair,
        "latency_seconds": latency,
        "latency_ms_per_sample": latency * 1000.0 / max(int(cache["num_windows"]), 1),
    }


def evaluate_fixed_pair(
    train_cache: Mapping[str, Any],
    eval_cache: Mapping[str, Any],
    pair_index: torch.Tensor,
) -> dict[str, Any]:
    train_pair_mae, _ = masked_pair_mae_mse(train_cache, pair_index)
    selected_class = int(torch.argmin(train_pair_mae.mean(dim=0)))
    selected_pair = pair_index[selected_class].view(1, 2).expand(int(eval_cache["num_windows"]), -1)
    prediction = _selected_pair_prediction(eval_cache, selected_pair)
    mae, mse = _mae_mse(prediction, eval_cache)
    eval_pair_mae, _ = masked_pair_mae_mse(eval_cache, pair_index)
    best_pair = torch.argmin(eval_pair_mae, dim=1)
    best_expert = eval_cache["best_expert"].to(torch.long)
    return {
        "mae": mae,
        "mse": mse,
        "average_experts_queried": 2.0,
        "best_pair_accuracy": float((torch.full_like(best_pair, selected_class) == best_pair).to(torch.float32).mean()),
        "best_individual_expert_top2_coverage": float(
            (selected_pair == best_expert[:, None]).any(dim=1).to(torch.float32).mean()
        ),
        "selected_pair_class": selected_class,
        "selected_pair_indices": pair_index[selected_class].tolist(),
        "selection_split": "router_train",
    }


def evaluate_equal_average_all(cache: Mapping[str, Any]) -> dict[str, Any]:
    prediction = cache["prediction_stack"].to(torch.float32).mean(dim=-1)
    mae, mse = _mae_mse(prediction, cache)
    return {
        "mae": mae,
        "mse": mse,
        "average_experts_queried": float(len(tuple(cache["expert_names"]))),
    }


def _load_reference_rows(path: Union[str, Path]) -> dict[str, dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    return {str(row.get("method")): row for row in rows if row.get("status") == "ok"}


def _summary_stats(runs: Sequence[Mapping[str, Any]], split: str) -> dict[str, dict[str, float]]:
    fields = (
        "mae",
        "mse",
        "average_experts_queried",
        "best_pair_accuracy",
        "best_individual_expert_top2_coverage",
    )
    stats: dict[str, dict[str, float]] = {}
    values_by_field: dict[str, list[float]] = {field: [] for field in fields}
    for run in runs:
        metrics = run.get(split)
        if not isinstance(metrics, Mapping):
            continue
        for field in fields:
            if field in metrics and metrics[field] != "":
                values_by_field[field].append(float(metrics[field]))
    for field, values in values_by_field.items():
        if values:
            tensor = torch.tensor(values, dtype=torch.float32)
            stats[field] = {
                "mean": float(tensor.mean()),
                "std": float(tensor.std(unbiased=False)),
            }
    return stats


def train_pair_selector(training_config: PairSelectorTrainingConfig) -> dict[str, Any]:
    set_reproducible_seed(training_config.seed)
    device = torch.device(training_config.device)
    train_cache = _load_torch(training_config.train_cache_path)
    val_cache = _load_torch(training_config.val_cache_path)
    _assert_cache_pair(train_cache, val_cache, "router_val")
    test_cache = None
    if training_config.test_cache_path:
        test_cache = _load_torch(training_config.test_cache_path)
        _assert_cache_pair(train_cache, test_cache, "test")
        if "targets" not in test_cache or "prediction_stack" not in test_cache:
            raise AssertionError("Test cache must include targets and frozen prediction_stack for MAE/MSE.")

    pair_index = build_pair_index(len(tuple(train_cache["expert_names"])))
    train_dataset = CostartsPairDataset(train_cache, pair_index, training_config.target_temperature)
    generator = torch.Generator()
    generator.manual_seed(training_config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        generator=generator,
    )
    model = CostartsPairSelector(
        input_len=int(train_cache["input_len"]),
        num_features=int(train_cache["num_features"]),
        num_pairs=int(pair_index.shape[0]),
        embedding_dim=training_config.embedding_dim,
        hidden_dim=training_config.hidden_dim,
        dropout=training_config.dropout,
        history_encoder_type=training_config.history_encoder_type,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    output_dir = Path(training_config.output_dir)
    results_dir = Path(training_config.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / f"best_costarts_pair_selector_seed{training_config.seed}.pt"
    history: list[dict[str, Any]] = []
    best_val_mae = float("inf")
    best_epoch = -1
    best_state: Optional[dict[str, torch.Tensor]] = None
    stale_epochs = 0

    for epoch in range(1, training_config.max_epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for batch in train_loader:
            history_tensor = batch["history"].to(device)
            soft_targets = batch["soft_targets"].to(device)
            best_pair = batch["best_pair"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(history_tensor)
            loss = pair_selector_loss(
                logits,
                soft_targets,
                best_pair,
                hard_label_weight=training_config.hard_label_weight,
            )
            loss.backward()
            if training_config.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), training_config.grad_clip_norm)
            optimizer.step()
            batch_size_actual = int(history_tensor.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_size_actual
            seen += batch_size_actual

        val_metrics = evaluate_pair_selector(
            model,
            val_cache,
            pair_index,
            batch_size=training_config.batch_size,
            device=device,
        )
        epoch_row = {
            "epoch": epoch,
            "train_loss": total_loss / max(seen, 1),
            "val_mae": val_metrics["mae"],
            "val_mse": val_metrics["mse"],
            "val_best_pair_accuracy": val_metrics["best_pair_accuracy"],
            "val_best_individual_expert_top2_coverage": val_metrics[
                "best_individual_expert_top2_coverage"
            ],
        }
        history.append(epoch_row)
        print(
            f"seed={training_config.seed} epoch={epoch:03d} "
            f"loss={epoch_row['train_loss']:.6f} val_mae={epoch_row['val_mae']:.6f} "
            f"pair_acc={epoch_row['val_best_pair_accuracy']:.3f} "
            f"top2_cov={epoch_row['val_best_individual_expert_top2_coverage']:.3f}"
        )
        if val_metrics["mae"] < best_val_mae - 1e-8:
            best_val_mae = float(val_metrics["mae"])
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= training_config.patience:
                break

    if best_state is None:
        raise RuntimeError("No checkpoint was selected.")
    model.load_state_dict(best_state)
    checkpoint = {
        "router_type": "costarts_pair_selector",
        "router_config": model.config_dict(),
        "router_state_dict": best_state,
        "pair_index": pair_index,
        "expert_names": tuple(train_cache["expert_names"]),
        "training_config": asdict(training_config),
        "epoch": best_epoch,
        "selection_metric": "router_val_mae",
        "best_val_mae": best_val_mae,
    }
    torch.save(checkpoint, best_path)

    val_metrics = evaluate_pair_selector(
        model,
        val_cache,
        pair_index,
        batch_size=training_config.batch_size,
        device=device,
    )
    test_metrics = None
    if test_cache is not None:
        test_metrics = evaluate_pair_selector(
            model,
            test_cache,
            pair_index,
            batch_size=training_config.batch_size,
            device=device,
        )
    fixed_pair_val = evaluate_fixed_pair(train_cache, val_cache, pair_index)
    equal_average_val = evaluate_equal_average_all(val_cache)
    reference_rows = _load_reference_rows(training_config.final_comparison_json)
    result = {
        "seed": training_config.seed,
        "best_epoch": best_epoch,
        "checkpoint_path": str(best_path),
        "pair_index": pair_index.tolist(),
        "expert_names": tuple(train_cache["expert_names"]),
        "train_cache": training_config.train_cache_path,
        "val_cache": training_config.val_cache_path,
        "test_cache": training_config.test_cache_path,
        "test_available": test_metrics is not None,
        "val": {
            key: value
            for key, value in val_metrics.items()
            if key not in {"selected_pair_class", "selected_pair_indices"}
        },
        "test": None
        if test_metrics is None
        else {
            key: value
            for key, value in test_metrics.items()
            if key not in {"selected_pair_class", "selected_pair_indices"}
        },
        "baselines": {
            "validation": {
                "equal_average_all_experts": equal_average_val,
                "validation_selected_fixed_pair": fixed_pair_val,
                "existing_predicted_top2_equal_average": reference_rows.get("predicted_top2_equal_average", {}),
                "routerdc_hard_with_contrastive": reference_rows.get("routerdc_hard_with_contrastive", {}),
                "improved_subset_utility_costarts": reference_rows.get("improved_subset_utility_costarts", {}),
            },
        },
        "training_history": history,
        "model_selection": "best checkpoint selected by router-validation MAE only",
    }
    result_path = results_dir / f"pair_selector_seed{training_config.seed}.json"
    result_path.write_text(json.dumps(_jsonable(result), indent=2), encoding="utf-8")
    print(f"Saved checkpoint: {best_path}")
    print(f"Saved result: {result_path}")
    return result


def train_many(config: PairSelectorTrainingConfig, seeds: Sequence[int]) -> dict[str, Any]:
    runs = []
    for seed in seeds:
        run_config = PairSelectorTrainingConfig(**{**asdict(config), "seed": int(seed)})
        runs.append(train_pair_selector(run_config))
    summary = {
        "seeds": [int(seed) for seed in seeds],
        "runs": runs,
        "mean_std": {
            "validation": _summary_stats(runs, "val"),
            "test": _summary_stats(runs, "test"),
        },
        "test_available": all(bool(run.get("test_available")) for run in runs),
    }
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "pair_selector_summary.json"
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")

    csv_path = results_dir / "pair_selector_summary.csv"
    fields = [
        "seed",
        "best_epoch",
        "val_mae",
        "val_mse",
        "val_average_experts_queried",
        "val_best_pair_accuracy",
        "val_best_individual_expert_top2_coverage",
        "test_mae",
        "test_mse",
        "test_average_experts_queried",
        "test_best_pair_accuracy",
        "test_best_individual_expert_top2_coverage",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            row = {
                "seed": run["seed"],
                "best_epoch": run["best_epoch"],
                "val_mae": run["val"]["mae"],
                "val_mse": run["val"]["mse"],
                "val_average_experts_queried": run["val"]["average_experts_queried"],
                "val_best_pair_accuracy": run["val"]["best_pair_accuracy"],
                "val_best_individual_expert_top2_coverage": run["val"][
                    "best_individual_expert_top2_coverage"
                ],
            }
            if run["test"] is not None:
                row.update(
                    {
                        "test_mae": run["test"]["mae"],
                        "test_mse": run["test"]["mse"],
                        "test_average_experts_queried": run["test"]["average_experts_queried"],
                        "test_best_pair_accuracy": run["test"]["best_pair_accuracy"],
                        "test_best_individual_expert_top2_coverage": run["test"][
                            "best_individual_expert_top2_coverage"
                        ],
                    }
                )
            writer.writerow(row)
    print(f"Saved summary: {summary_path}")
    print(f"Saved summary CSV: {csv_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a direct COSTARTS predicted top-2 pair selector.")
    parser.add_argument("--train-cache", default=DEFAULT_TRAIN_CACHE)
    parser.add_argument("--val-cache", default=DEFAULT_VAL_CACHE)
    parser.add_argument("--test-cache", default=DEFAULT_TEST_CACHE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--final-comparison-json", default=DEFAULT_FINAL_COMPARISON_JSON)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 11, 13])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--target-temperature", type=float, default=0.03)
    parser.add_argument("--hard-label-weight", type=float, default=1.0)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--history-encoder-type", choices=("current", "simple"), default="current")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PairSelectorTrainingConfig(
        train_cache_path=args.train_cache,
        val_cache_path=args.val_cache,
        test_cache_path=args.test_cache,
        output_dir=args.output_dir,
        results_dir=args.results_dir,
        final_comparison_json=args.final_comparison_json,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        target_temperature=args.target_temperature,
        hard_label_weight=args.hard_label_weight,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        history_encoder_type=args.history_encoder_type,
        device=args.device,
    )
    summary = train_many(config, args.seeds)
    validation_stats = summary["mean_std"]["validation"]
    print("\nPair selector validation mean/std:")
    for field, values in validation_stats.items():
        print(f"  {field}: mean={values['mean']:.6f}, std={values['std']:.6f}")
    if not summary["test_available"]:
        print("No untouched test cache with frozen predictions was supplied; test metrics were not computed.")


if __name__ == "__main__":
    main()
