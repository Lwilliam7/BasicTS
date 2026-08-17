"""Train a one-shot COSTAR-TS router over a fixed menu of expert subsets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_costarts_walkforward_cache import EXPERT_ORDER, sha256_file, validate_walkforward_cache
from scripts.train_sequential_costarts_full_walkforward import CacheWindowDataset


CANDIDATE_SUBSET_NAMES: tuple[tuple[str, ...], ...] = (
    ("iTransformer",),
    ("PatchTST", "iTransformer"),
    ("PatchTST", "iTransformer", "TimesNet"),
    ("DLinear", "PatchTST", "iTransformer", "TimesNet"),
    ("DLinear", "PatchTST", "iTransformer", "TimesNet", "ModernTCN"),
)

FIXED3_CORRECTION_SUBSET_NAMES: tuple[tuple[str, ...], ...] = (
    ("PatchTST", "iTransformer", "TimesNet"),
    ("iTransformer", "TimesNet"),
    ("PatchTST", "TimesNet"),
    ("PatchTST", "iTransformer"),
    ("DLinear", "PatchTST", "iTransformer", "TimesNet"),
    ("PatchTST", "iTransformer", "TimesNet", "ModernTCN"),
)

FIXED3_CORRECTION_ACTIONS: tuple[str, ...] = (
    "KEEP fixed3",
    "DROP PatchTST",
    "DROP iTransformer",
    "DROP TimesNet",
    "ADD DLinear",
    "ADD ModernTCN",
)


def candidate_subset_names(menu: str) -> tuple[tuple[str, ...], ...]:
    if menu == "subset_menu":
        return CANDIDATE_SUBSET_NAMES
    if menu == "fixed3_correction":
        return FIXED3_CORRECTION_SUBSET_NAMES
    raise ValueError(f"Unknown menu: {menu}")


def candidate_action_labels(menu: str) -> list[str]:
    if menu == "fixed3_correction":
        return list(FIXED3_CORRECTION_ACTIONS)
    return ["+".join(subset) for subset in CANDIDATE_SUBSET_NAMES]


def candidate_indices(menu: str) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(EXPERT_ORDER.index(name) for name in subset) for subset in candidate_subset_names(menu))


def candidate_labels(menu: str) -> list[str]:
    return ["+".join(subset) for subset in candidate_subset_names(menu)]


class OneShotSubsetMenuRouter(nn.Module):
    """History-only router that chooses one fixed candidate expert subset."""

    def __init__(
        self,
        num_candidates: int,
        input_len: int = 96,
        num_features: int = 7,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.num_candidates = int(num_candidates)
        self.input_len = int(input_len)
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
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_candidates),
        )

    def forward(self, history: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.history_encoder(history.transpose(1, 2)).squeeze(-1)
        representation = self.history_projection(encoded)
        return {"representation": representation, "logits": self.head(representation)}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_verified_cache(path: Path, expected_role: str) -> dict[str, Any]:
    if "test" in str(path).lower():
        raise ValueError(f"Refusing test cache path: {path}")
    cache = torch.load(path, map_location="cpu", weights_only=False)
    validate_walkforward_cache(cache)
    if cache["cache_role"] != expected_role:
        raise ValueError(f"{path} cache_role={cache['cache_role']!r}, expected {expected_role!r}")
    return cache


def load_normalizer_std(path: Path) -> torch.Tensor:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "scaler_std" not in checkpoint:
        raise KeyError(f"{path} does not contain scaler_std")
    return checkpoint["scaler_std"].to(torch.float32)


def sample_mae(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, std: torch.Tensor | None = None) -> torch.Tensor:
    if std is not None:
        prediction = prediction / std.to(prediction.device, prediction.dtype).view(1, 1, -1)
        target = target / std.to(target.device, target.dtype).view(1, 1, -1)
    mask_f = mask.to(prediction.dtype)
    denom = mask_f.flatten(1).sum(dim=1).clamp_min(1.0)
    return ((prediction - target).abs() * mask_f).flatten(1).sum(dim=1) / denom


def sample_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, std: torch.Tensor | None = None) -> torch.Tensor:
    if std is not None:
        prediction = prediction / std.to(prediction.device, prediction.dtype).view(1, 1, -1)
        target = target / std.to(target.device, target.dtype).view(1, 1, -1)
    mask_f = mask.to(prediction.dtype)
    denom = mask_f.flatten(1).sum(dim=1).clamp_min(1.0)
    return ((prediction - target).square() * mask_f).flatten(1).sum(dim=1) / denom


def candidate_predictions(prediction_stack: torch.Tensor, subsets: Sequence[Sequence[int]]) -> torch.Tensor:
    predictions = [prediction_stack[..., list(subset)].mean(dim=-1) for subset in subsets]
    return torch.stack(predictions, dim=-1)


def candidate_errors(
    prediction_stack: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    subsets: Sequence[Sequence[int]],
    normalizer_std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    predictions = candidate_predictions(prediction_stack, subsets)
    maes = []
    mses = []
    raw_maes = []
    raw_mses = []
    for index in range(predictions.shape[-1]):
        pred = predictions[..., index]
        maes.append(sample_mae(pred, targets, masks, normalizer_std))
        mses.append(sample_mse(pred, targets, masks, normalizer_std))
        raw_maes.append(sample_mae(pred, targets, masks))
        raw_mses.append(sample_mse(pred, targets, masks))
    return torch.stack(maes, dim=1), torch.stack(mses, dim=1), torch.stack(raw_maes, dim=1), torch.stack(raw_mses, dim=1)


def objective_loss(logits: torch.Tensor, errors: torch.Tensor, objective: str, temperature: float) -> torch.Tensor:
    if objective == "hard":
        return F.cross_entropy(logits, errors.argmin(dim=1))
    if objective == "soft":
        utilities = -errors
        targets = torch.softmax(utilities / float(temperature), dim=1).detach()
        log_probs = F.log_softmax(logits, dim=1)
        return -(targets * log_probs).sum(dim=1).mean()
    raise ValueError(f"Unknown objective: {objective}")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def aggregate(values: Sequence[float]) -> tuple[float, float]:
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.pstdev(values))


def parameter_count(model: nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters()))


def train_one_epoch(
    model: OneShotSubsetMenuRouter,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    subsets: Sequence[Sequence[int]],
    normalizer_std: torch.Tensor,
    objective: str,
    temperature: float,
    grad_clip_norm: float,
) -> float:
    model.train()
    losses = []
    for batch in loader:
        history = batch["history"].to(device).to(torch.float32)
        targets = batch["targets"].to(device).to(torch.float32)
        masks = batch["target_masks"].to(device).to(torch.bool)
        prediction_stack = batch["prediction_stack"].to(device).to(torch.float32)
        errors, _, _, _ = candidate_errors(prediction_stack, targets, masks, subsets, normalizer_std.to(device))
        logits = model(history)["logits"]
        loss = objective_loss(logits, errors, objective, temperature)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return float(statistics.mean(losses)) if losses else float("nan")


@torch.no_grad()
def evaluate(
    model: OneShotSubsetMenuRouter,
    cache: Mapping[str, Any],
    device: torch.device,
    subsets: Sequence[Sequence[int]],
    menu: str,
    normalizer_std: torch.Tensor,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    loader = DataLoader(CacheWindowDataset(cache), batch_size=batch_size, shuffle=False)
    maes = []
    mses = []
    raw_maes = []
    raw_mses = []
    hits = []
    regrets = []
    selected_counts = torch.zeros(len(subsets), dtype=torch.float64)
    oracle_counts = torch.zeros(len(subsets), dtype=torch.float64)
    per_window = []
    offset = 0
    labels = candidate_labels(menu)
    action_labels = candidate_action_labels(menu)
    for batch in loader:
        history = batch["history"].to(device).to(torch.float32)
        targets = batch["targets"].to(device).to(torch.float32)
        masks = batch["target_masks"].to(device).to(torch.bool)
        prediction_stack = batch["prediction_stack"].to(device).to(torch.float32)
        errors, mse_errors, raw_error_matrix, raw_mse_matrix = candidate_errors(
            prediction_stack, targets, masks, subsets, normalizer_std.to(device)
        )
        logits = model(history)["logits"]
        selected = logits.argmax(dim=1)
        oracle = errors.argmin(dim=1)
        batch_indices = torch.arange(history.shape[0], device=device)
        selected_mae = errors[batch_indices, selected]
        selected_mse = mse_errors[batch_indices, selected]
        selected_raw_mae = raw_error_matrix[batch_indices, selected]
        selected_raw_mse = raw_mse_matrix[batch_indices, selected]
        oracle_mae = errors[batch_indices, oracle]
        regrets.extend((selected_mae - oracle_mae).detach().cpu().tolist())
        hits.extend((selected == oracle).detach().cpu().to(torch.float32).tolist())
        maes.append(selected_mae.detach().cpu())
        mses.append(selected_mse.detach().cpu())
        raw_maes.append(selected_raw_mae.detach().cpu())
        raw_mses.append(selected_raw_mse.detach().cpu())
        for item in selected.detach().cpu().tolist():
            selected_counts[int(item)] += 1
        for item in oracle.detach().cpu().tolist():
            oracle_counts[int(item)] += 1
        for row in range(history.shape[0]):
            per_window.append(
                {
                    "cache_index": offset + row,
                    "absolute_window_start": int(batch["absolute_window_start"][row].item()),
                    "selected_index": int(selected[row].item()),
                    "selected_action": action_labels[int(selected[row].item())],
                    "selected_subset": labels[int(selected[row].item())],
                    "oracle_index": int(oracle[row].item()),
                    "oracle_action": action_labels[int(oracle[row].item())],
                    "oracle_subset": labels[int(oracle[row].item())],
                    "normalized_mae": float(selected_mae[row].item()),
                    "normalized_mse": float(selected_mse[row].item()),
                    "oracle_menu_mae": float(oracle_mae[row].item()),
                    "regret": float((selected_mae[row] - oracle_mae[row]).item()),
                }
            )
        offset += history.shape[0]
    total = float(cache["num_windows"])
    expert_counts = [len(subset) for subset in subsets]
    selected_total = sum(float(selected_counts[index].item()) * expert_counts[index] for index in range(len(subsets)))
    return {
        "mae": float(torch.cat(maes).mean().item()),
        "mse": float(torch.cat(mses).mean().item()),
        "raw_mae": float(torch.cat(raw_maes).mean().item()),
        "raw_mse": float(torch.cat(raw_mses).mean().item()),
        "subset_selection_accuracy": float(statistics.mean(hits) * 100.0),
        "oracle_menu_regret": float(statistics.mean(regrets)),
        "oracle_menu_regret_median": float(torch.tensor(regrets).median().item()),
        "average_experts_used": float(selected_total / total),
        "selected_action_percent": {action_labels[i]: float(selected_counts[i].item() * 100.0 / total) for i in range(len(subsets))},
        "selected_subset_percent": {labels[i]: float(selected_counts[i].item() * 100.0 / total) for i in range(len(subsets))},
        "oracle_action_percent": {action_labels[i]: float(oracle_counts[i].item() * 100.0 / total) for i in range(len(subsets))},
        "oracle_subset_percent": {labels[i]: float(oracle_counts[i].item() * 100.0 / total) for i in range(len(subsets))},
        "per_window": per_window,
    }


def train_config(
    objective: str,
    temperature: float,
    args: argparse.Namespace,
    train_cache: Mapping[str, Any],
    val_cache: Mapping[str, Any],
    normalizer_std: torch.Tensor,
) -> dict[str, Any]:
    subsets = candidate_indices(args.menu)
    seeds = tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
    device = torch.device(args.device)
    config = objective if objective == "hard" else f"soft_tau_{str(temperature).replace('.', 'p')}"
    result_root = ROOT / args.results_root / config
    checkpoint_root = ROOT / args.checkpoint_root / config
    rows = []
    metrics_by_seed = {}
    for seed in seeds:
        set_seed(seed)
        model = OneShotSubsetMenuRouter(
            num_candidates=len(subsets),
            input_len=int(train_cache["input_len"]),
            num_features=int(train_cache["num_features"]),
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        loader = DataLoader(CacheWindowDataset(train_cache), batch_size=args.batch_size, shuffle=True)
        best_mae = math.inf
        best_metrics = None
        best_epoch = -1
        bad_epochs = 0
        curves = []
        for epoch in range(1, args.max_epochs + 1):
            train_loss = train_one_epoch(
                model,
                loader,
                optimizer,
                device,
                subsets,
                normalizer_std,
                objective,
                temperature,
                args.grad_clip_norm,
            )
            metrics = evaluate(model, val_cache, device, subsets, args.menu, normalizer_std, args.batch_size)
            curves.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "validation_mae": metrics["mae"],
                    "validation_mse": metrics["mse"],
                    "subset_selection_accuracy": metrics["subset_selection_accuracy"],
                    "oracle_menu_regret": metrics["oracle_menu_regret"],
                    "average_experts_used": metrics["average_experts_used"],
                }
            )
            if metrics["mae"] < best_mae:
                best_mae = metrics["mae"]
                best_metrics = metrics
                best_epoch = epoch
                bad_epochs = 0
                seed_dir = checkpoint_root / f"seed_{seed}"
                seed_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "router_type": "OneShotSubsetMenuRouter",
                        "objective": objective,
                        "temperature": temperature if objective == "soft" else None,
                        "router_config": {
                            "num_candidates": len(subsets),
                            "input_len": int(train_cache["input_len"]),
                            "num_features": int(train_cache["num_features"]),
                            "embedding_dim": args.embedding_dim,
                            "hidden_dim": args.hidden_dim,
                        },
                        "candidate_subsets": candidate_labels(args.menu),
                        "candidate_actions": candidate_action_labels(args.menu),
                        "state_dict": model.state_dict(),
                        "seed": seed,
                        "epoch": epoch,
                        "validation_metrics": {key: value for key, value in metrics.items() if key != "per_window"},
                        "created_at_utc": datetime.now(timezone.utc).isoformat(),
                        "safety": "NO TEST DATA USED",
                    },
                    seed_dir / "best_subset_menu_router.pt",
                )
            else:
                bad_epochs += 1
                if bad_epochs >= args.patience:
                    break
        assert best_metrics is not None
        seed_result = result_root / f"seed_{seed}"
        write_csv(seed_result / "training_curves.csv", curves)
        write_csv(seed_result / "validation_per_window.csv", best_metrics["per_window"])
        row = {
            "config": config,
            "objective": objective,
            "temperature": temperature if objective == "soft" else "",
            "seed": seed,
            "best_epoch": best_epoch,
            "validation_mae": best_metrics["mae"],
            "validation_mse": best_metrics["mse"],
            "raw_validation_mae": best_metrics["raw_mae"],
            "raw_validation_mse": best_metrics["raw_mse"],
            "subset_selection_accuracy": best_metrics["subset_selection_accuracy"],
            "oracle_menu_regret": best_metrics["oracle_menu_regret"],
            "average_experts_used": best_metrics["average_experts_used"],
            "parameter_count": parameter_count(model),
        }
        rows.append(row)
        metrics_by_seed[str(seed)] = {key: value for key, value in best_metrics.items() if key != "per_window"}
    write_csv(result_root / "per_seed_results.csv", rows)
    summary = {
        "config": config,
        "objective": objective,
        "temperature": temperature if objective == "soft" else None,
        "validation_mae_mean": aggregate([row["validation_mae"] for row in rows])[0],
        "validation_mae_std": aggregate([row["validation_mae"] for row in rows])[1],
        "validation_mse_mean": aggregate([row["validation_mse"] for row in rows])[0],
        "validation_mse_std": aggregate([row["validation_mse"] for row in rows])[1],
        "subset_selection_accuracy_mean": aggregate([row["subset_selection_accuracy"] for row in rows])[0],
        "oracle_menu_regret_mean": aggregate([row["oracle_menu_regret"] for row in rows])[0],
        "average_experts_used_mean": aggregate([row["average_experts_used"] for row in rows])[0],
        "parameter_count": int(rows[0]["parameter_count"]),
        "per_seed": rows,
        "best_metrics_by_seed": metrics_by_seed,
    }
    (result_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def fixed_candidate_metrics(cache: Mapping[str, Any], normalizer_std: torch.Tensor, menu: str) -> dict[str, Any]:
    subsets = candidate_indices(menu)
    labels = candidate_labels(menu)
    action_labels = candidate_action_labels(menu)
    prediction_stack = cache["prediction_stack"].to(torch.float32)
    targets = cache["targets"].to(torch.float32)
    masks = cache["target_masks"].to(torch.bool)
    errors, mses, _, _ = candidate_errors(prediction_stack, targets, masks, subsets, normalizer_std)
    rows = []
    for index, label in enumerate(labels):
        rows.append(
            {
                "subset": label,
                "action": action_labels[index],
                "num_experts": len(subsets[index]),
                "mae": float(errors[:, index].mean().item()),
                "mse": float(mses[:, index].mean().item()),
            }
        )
    oracle = errors.min(dim=1).values
    return {
        "candidate_subsets": rows,
        "fixed3": rows[2] if menu == "subset_menu" else rows[0],
        "oracle_menu_mae": float(oracle.mean().item()),
        "oracle_menu_regret_ceiling_from_fixed3": float((rows[2] if menu == "subset_menu" else rows[0])["mae"] - oracle.mean().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", default="cache/costarts_walkforward/router_train_20_60_cache.pt")
    parser.add_argument("--val-cache", default="cache/costarts_walkforward/router_val_60_80_cache.pt")
    parser.add_argument("--normalizer-checkpoint", default="checkpoints/costarts_walkforward/final_60/DLinear/best_expert.pt")
    parser.add_argument("--checkpoint-root", default="checkpoints/costarts_walkforward/subset_menu_router")
    parser.add_argument("--results-root", default="results/router_summary/costarts_walkforward/subset_menu_router")
    parser.add_argument("--objectives", default="hard,soft")
    parser.add_argument("--menu", choices=("subset_menu", "fixed3_correction"), default="subset_menu")
    parser.add_argument("--temperatures", default="0.01")
    parser.add_argument("--seeds", default="7")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    train_cache_path = ROOT / args.train_cache
    val_cache_path = ROOT / args.val_cache
    train_cache = load_verified_cache(train_cache_path, "router_train_20_60")
    val_cache = load_verified_cache(val_cache_path, "router_val_60_80")
    starts = val_cache["absolute_window_starts"].to(torch.long)
    val_start = int(starts.min().item())
    val_end = int(starts.max().item()) + int(val_cache["forecast_horizon"])
    if val_start != 8640 or val_end > 11520:
        raise ValueError(f"Unexpected validation coverage: starts at {val_start}, final forecast ends at {val_end}")
    normalizer_std = load_normalizer_std(ROOT / args.normalizer_checkpoint)
    objectives = [item.strip() for item in args.objectives.split(",") if item.strip()]
    temperatures = [float(item.strip()) for item in args.temperatures.split(",") if item.strip()]
    summaries = []
    if "hard" in objectives:
        summaries.append(train_config("hard", temperatures[0], args, train_cache, val_cache, normalizer_std))
    if "soft" in objectives:
        for temperature in temperatures:
            summaries.append(train_config("soft", temperature, args, train_cache, val_cache, normalizer_std))
    comparison = []
    for summary in summaries:
        comparison.append(
            {
                "config": summary["config"],
                "objective": summary["objective"],
                "temperature": summary["temperature"] if summary["temperature"] is not None else "",
                "validation_mae_mean": summary["validation_mae_mean"],
                "validation_mae_std": summary["validation_mae_std"],
                "validation_mse_mean": summary["validation_mse_mean"],
                "validation_mse_std": summary["validation_mse_std"],
                "subset_selection_accuracy_mean": summary["subset_selection_accuracy_mean"],
                "oracle_menu_regret_mean": summary["oracle_menu_regret_mean"],
                "average_experts_used_mean": summary["average_experts_used_mean"],
                "parameter_count": summary["parameter_count"],
                "improvement_vs_fixed3_mae": 0.36726489663124084 - float(summary["validation_mae_mean"]),
                "improvement_vs_best_sequential_costar_mae": 0.3692974388599396 - float(summary["validation_mae_mean"]),
            }
        )
    result_root = ROOT / args.results_root
    write_csv(result_root / "comparison.csv", comparison)
    fixed_metrics = fixed_candidate_metrics(val_cache, normalizer_std, args.menu)
    full_summary = {
        "method": "One-shot fixed-subset menu router diagnostic",
        "train_cache": args.train_cache,
        "val_cache": args.val_cache,
        "validation_range": [8640, 11520],
        "train_cache_sha256": sha256_file(train_cache_path),
        "val_cache_sha256": sha256_file(val_cache_path),
        "menu": args.menu,
        "candidate_actions": candidate_action_labels(args.menu),
        "candidate_subsets": candidate_labels(args.menu),
        "fixed_candidate_metrics": fixed_metrics,
        "reference": {
            "fixed3_mae": 0.36726489663124084,
            "small_menu_oracle_mae": 0.3470388650894165,
            "current_best_sequential_costar_mae": 0.3692974388599396,
        },
        "summaries": summaries,
        "comparison": comparison,
        "safety": "NO TEST DATA USED",
    }
    (result_root / "summary.json").write_text(json.dumps(full_summary, indent=2), encoding="utf-8")
    print(json.dumps({"comparison": comparison, "fixed_candidate_metrics": fixed_metrics, "safety": full_summary["safety"]}, indent=2))


if __name__ == "__main__":
    main()
