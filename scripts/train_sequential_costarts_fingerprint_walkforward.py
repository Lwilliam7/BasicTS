"""Train walk-forward frozen-expert Sequential COSTAR-TS fingerprint variants."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_costarts_walkforward_cache import EXPERT_ORDER, sha256_file, validate_walkforward_cache
from scripts.sequential_costarts_fingerprint_model import (
    FingerprintMode,
    SequentialCOSTARTSFingerprintRouter,
    compute_history_fingerprints,
)
from scripts.train_sequential_costarts_full_walkforward import (
    CacheWindowDataset,
    current_average_from_ids,
    greedy_oracle_order,
    make_state,
    utility_targets,
)


MODE_LABELS: dict[str, str] = {
    "embedding_only": "Embedding only",
    "fingerprint_only": "Fingerprint only",
    "embedding_fingerprint": "Embedding + Fingerprint",
}


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


def normalized_sample_mae(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    std = std.to(prediction.device, prediction.dtype).view(1, 1, -1)
    diff = (prediction - target) / std
    mask_f = mask.to(prediction.dtype)
    denom = mask_f.flatten(1).sum(dim=1).clamp_min(1.0)
    return (diff.abs() * mask_f).flatten(1).sum(dim=1) / denom


def normalized_sample_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    std = std.to(prediction.device, prediction.dtype).view(1, 1, -1)
    diff = (prediction - target) / std
    mask_f = mask.to(prediction.dtype)
    denom = mask_f.flatten(1).sum(dim=1).clamp_min(1.0)
    return (diff.square() * mask_f).flatten(1).sum(dim=1) / denom


def train_one_epoch(
    model: SequentialCOSTARTSFingerprintRouter,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_queries: int,
    query_cost: float,
    grad_clip_norm: float,
) -> float:
    model.train()
    losses = []
    for batch in loader:
        history = batch["history"].to(device).to(torch.float32)
        targets = batch["targets"].to(device).to(torch.float32)
        masks = batch["target_masks"].to(device).to(torch.bool)
        prediction_stack = batch["prediction_stack"].to(device).to(torch.float32)
        oracle_order = greedy_oracle_order(prediction_stack, targets, masks)
        total_loss = torch.zeros((), device=device)
        for step in range(max_queries):
            queried_ids = torch.full((history.shape[0], max_queries), -1, dtype=torch.long, device=device)
            if step > 0:
                queried_ids[:, :step] = oracle_order[:, :step]
            queried_mask, queried_forecasts, current_average = make_state(prediction_stack, queried_ids, model.num_experts)
            outputs = model(history, queried_mask, queried_ids, queried_forecasts, current_average_forecast=current_average)
            target = utility_targets(prediction_stack, targets, masks, queried_ids, query_cost)
            loss_mask = 1.0 - queried_mask
            total_loss = total_loss + F.smooth_l1_loss(outputs["utility_prediction"] * loss_mask, target * loss_mask)
        total_loss = total_loss / float(max_queries)
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        losses.append(float(total_loss.detach().cpu().item()))
    return float(statistics.mean(losses)) if losses else float("nan")


@torch.no_grad()
def evaluate_router(
    model: SequentialCOSTARTSFingerprintRouter,
    cache: Mapping[str, Any],
    device: torch.device,
    max_queries: int,
    query_threshold: float,
    batch_size: int,
    normalizer_std: torch.Tensor,
) -> dict[str, Any]:
    model.eval()
    loader = DataLoader(CacheWindowDataset(cache), batch_size=batch_size, shuffle=False)
    raw_maes = []
    raw_mses = []
    norm_maes = []
    norm_mses = []
    query_counts = []
    first_counts = torch.zeros(model.num_experts, dtype=torch.float64)
    second_counts = torch.zeros(model.num_experts, dtype=torch.float64)
    queried_counts = torch.zeros(model.num_experts, dtype=torch.float64)
    stopping_counts = torch.zeros(model.num_experts, dtype=torch.float64)
    best_experts = cache["best_expert"].to(torch.long)
    first_queries = []
    top2_hits = []
    per_window = []
    offset = 0
    for batch in loader:
        history = batch["history"].to(device).to(torch.float32)
        targets = batch["targets"].to(device).to(torch.float32)
        masks = batch["target_masks"].to(device).to(torch.bool)
        prediction_stack = batch["prediction_stack"].to(device).to(torch.float32)
        queried_ids = torch.full((history.shape[0], max_queries), -1, dtype=torch.long, device=device)
        active = torch.ones(history.shape[0], dtype=torch.bool, device=device)
        for step in range(max_queries):
            state_mask, state_forecasts, current_average = make_state(prediction_stack, queried_ids, model.num_experts)
            utilities = model(history, state_mask, queried_ids, state_forecasts, current_average_forecast=current_average)["utility_prediction"]
            utilities = utilities.masked_fill(state_mask.to(torch.bool), -1e9)
            values, next_ids = utilities.max(dim=1)
            should_query = active & ((step == 0) | (values > float(query_threshold)))
            if not bool(should_query.any()):
                break
            queried_ids[should_query, step] = next_ids[should_query]
            active = active & should_query & (step + 1 < max_queries)
        final_prediction = current_average_from_ids(prediction_stack, queried_ids)
        mask_f = masks.to(final_prediction.dtype)
        denom = mask_f.flatten(1).sum(dim=1).clamp_min(1.0)
        raw_mae = ((final_prediction - targets).abs() * mask_f).flatten(1).sum(dim=1) / denom
        raw_mse = ((final_prediction - targets).square() * mask_f).flatten(1).sum(dim=1) / denom
        norm_mae = normalized_sample_mae(final_prediction, targets, masks, normalizer_std)
        norm_mse = normalized_sample_mse(final_prediction, targets, masks, normalizer_std)
        raw_maes.append(raw_mae.cpu())
        raw_mses.append(raw_mse.cpu())
        norm_maes.append(norm_mae.cpu())
        norm_mses.append(norm_mse.cpu())
        counts = (queried_ids >= 0).sum(dim=1).cpu()
        query_counts.append(counts)
        for row in range(queried_ids.shape[0]):
            global_index = offset + row
            ids = queried_ids[row][queried_ids[row] >= 0].detach().cpu().tolist()
            if ids:
                first_counts[ids[0]] += 1
                first_queries.append(ids[0])
            if len(ids) > 1:
                second_counts[ids[1]] += 1
            for expert_id in set(ids):
                queried_counts[expert_id] += 1
            stopping_counts[int(counts[row].item()) - 1] += 1
            best = int(best_experts[global_index].item())
            top2_hits.append(best in ids[:2])
            per_window.append(
                {
                    "cache_index": global_index,
                    "absolute_window_start": int(batch["absolute_window_start"][row].item()),
                    "query_count": int(counts[row].item()),
                    "queried_experts": " ".join(str(item) for item in ids),
                    "raw_mae": float(raw_mae[row].item()),
                    "raw_mse": float(raw_mse[row].item()),
                    "normalized_mae": float(norm_mae[row].item()),
                    "normalized_mse": float(norm_mse[row].item()),
                }
            )
        offset += history.shape[0]
    total = float(cache["num_windows"])
    first_tensor = torch.tensor(first_queries, dtype=torch.long)
    counts_tensor = torch.cat(query_counts).to(torch.float32)
    return {
        "raw_mae": float(torch.cat(raw_maes).mean().item()),
        "raw_mse": float(torch.cat(raw_mses).mean().item()),
        "mae": float(torch.cat(norm_maes).mean().item()),
        "mse": float(torch.cat(norm_mses).mean().item()),
        "average_queries": float(counts_tensor.mean().item()),
        "top1_expert_accuracy": float((first_tensor == best_experts[: first_tensor.numel()]).to(torch.float32).mean().item() * 100.0),
        "top2_oracle_coverage": float(np.mean(top2_hits) * 100.0),
        "query_count_percent": {str(i + 1): float(stopping_counts[i].item() * 100.0 / total) for i in range(model.num_experts)},
        "first_query_percent": {EXPERT_ORDER[i]: float(first_counts[i].item() * 100.0 / total) for i in range(model.num_experts)},
        "second_query_percent_all_samples": {EXPERT_ORDER[i]: float(second_counts[i].item() * 100.0 / total) for i in range(model.num_experts)},
        "second_query_percent_among_second_queries": {
            EXPERT_ORDER[i]: float(second_counts[i].item() * 100.0 / max(second_counts.sum().item(), 1.0))
            for i in range(model.num_experts)
        },
        "expert_usage_percent": {EXPERT_ORDER[i]: float(queried_counts[i].item() * 100.0 / total) for i in range(model.num_experts)},
        "per_window": per_window,
    }


def individual_expert_metrics(cache: Mapping[str, Any], normalizer_std: torch.Tensor) -> list[dict[str, Any]]:
    prediction_stack = cache["prediction_stack"]
    targets = cache["targets"]
    masks = cache["target_masks"]
    rows = []
    for index, expert in enumerate(EXPERT_ORDER):
        prediction = prediction_stack[..., index]
        rows.append(
            {
                "expert": expert,
                "raw_mae": float(cache["error_matrix"][:, index].mean().item()),
                "raw_mse": float(cache["mse_matrix"][:, index].mean().item()),
                "mae": float(normalized_sample_mae(prediction, targets, masks, normalizer_std).mean().item()),
                "mse": float(normalized_sample_mse(prediction, targets, masks, normalizer_std).mean().item()),
            }
        )
    return rows


def parameter_count(model: torch.nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters()))


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


def train_mode(mode: FingerprintMode, args: argparse.Namespace, train_cache: Mapping[str, Any], val_cache: Mapping[str, Any], normalizer_std: torch.Tensor) -> dict[str, Any]:
    seeds = tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
    device = torch.device(args.device)
    result_root = ROOT / args.results_root / mode
    checkpoint_root = ROOT / args.checkpoint_root / mode
    per_seed_rows = []
    best_metrics_by_seed: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        set_seed(seed)
        model = SequentialCOSTARTSFingerprintRouter(
            num_experts=len(EXPERT_ORDER),
            max_subset_size=args.max_queries,
            input_len=int(train_cache["input_len"]),
            forecast_horizon=int(train_cache["forecast_horizon"]),
            num_features=int(train_cache["num_features"]),
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            fingerprint_mode=mode,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        loader = DataLoader(CacheWindowDataset(train_cache), batch_size=args.batch_size, shuffle=True)
        best_mae = math.inf
        best_metrics = None
        best_epoch = -1
        bad_epochs = 0
        curves = []
        seed_result_dir = result_root / f"seed_{seed}"
        seed_checkpoint_dir = checkpoint_root / f"seed_{seed}"
        for epoch in range(1, args.max_epochs + 1):
            train_loss = train_one_epoch(model, loader, optimizer, device, args.max_queries, args.query_cost, args.grad_clip_norm)
            metrics = evaluate_router(model, val_cache, device, args.max_queries, args.query_threshold, args.batch_size, normalizer_std)
            curves.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "validation_mae": metrics["mae"],
                    "validation_mse": metrics["mse"],
                    "raw_validation_mae": metrics["raw_mae"],
                    "raw_validation_mse": metrics["raw_mse"],
                    "average_queries": metrics["average_queries"],
                    "top1_expert_accuracy": metrics["top1_expert_accuracy"],
                    "top2_oracle_coverage": metrics["top2_oracle_coverage"],
                }
            )
            if metrics["mae"] < best_mae:
                best_mae = metrics["mae"]
                best_metrics = metrics
                best_epoch = epoch
                bad_epochs = 0
                seed_checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "router_type": "SequentialCOSTARTSFingerprintRouter",
                        "router_config": model.config_dict(),
                        "router_state_dict": model.state_dict(),
                        "fingerprint_mode": mode,
                        "seed": seed,
                        "epoch": epoch,
                        "validation_metrics": {key: value for key, value in metrics.items() if key != "per_window"},
                        "train_cache": args.train_cache,
                        "val_cache": args.val_cache,
                        "created_at_utc": datetime.now(timezone.utc).isoformat(),
                        "safety": "NO TEST DATA USED",
                    },
                    seed_checkpoint_dir / "best_fingerprint_router.pt",
                )
            else:
                bad_epochs += 1
                if bad_epochs >= args.patience:
                    break
        assert best_metrics is not None
        write_csv(seed_result_dir / "training_curves.csv", curves)
        write_csv(seed_result_dir / "validation_per_window.csv", best_metrics["per_window"])
        row = {
            "mode": mode,
            "model": MODE_LABELS[mode],
            "seed": seed,
            "best_epoch": best_epoch,
            "validation_mae": best_metrics["mae"],
            "validation_mse": best_metrics["mse"],
            "raw_validation_mae": best_metrics["raw_mae"],
            "raw_validation_mse": best_metrics["raw_mse"],
            "average_queries": best_metrics["average_queries"],
            "top1_expert_accuracy": best_metrics["top1_expert_accuracy"],
            "top2_oracle_coverage": best_metrics["top2_oracle_coverage"],
            "checkpoint_path": str(seed_checkpoint_dir / "best_fingerprint_router.pt"),
            "parameter_count": parameter_count(model),
        }
        per_seed_rows.append(row)
        best_metrics_by_seed[seed] = best_metrics
    write_csv(result_root / "per_seed_results.csv", per_seed_rows)
    mae_mean, mae_std = aggregate([row["validation_mae"] for row in per_seed_rows])
    mse_mean, mse_std = aggregate([row["validation_mse"] for row in per_seed_rows])
    q_mean, q_std = aggregate([row["average_queries"] for row in per_seed_rows])
    summary = {
        "mode": mode,
        "model": MODE_LABELS[mode],
        "validation_mae_mean": mae_mean,
        "validation_mae_std": mae_std,
        "validation_mse_mean": mse_mean,
        "validation_mse_std": mse_std,
        "average_queries_mean": q_mean,
        "average_queries_std": q_std,
        "parameter_count": int(per_seed_rows[0]["parameter_count"]),
        "per_seed": per_seed_rows,
        "best_metrics_by_seed": {str(seed): {key: value for key, value in metrics.items() if key != "per_window"} for seed, metrics in best_metrics_by_seed.items()},
    }
    (result_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def expert_complementarity(cache: Mapping[str, Any], normalizer_std: torch.Tensor) -> dict[str, Any]:
    prediction_stack = cache["prediction_stack"]
    targets = cache["targets"]
    masks = cache["target_masks"]
    per_expert = []
    errors = []
    for index in range(len(EXPERT_ORDER)):
        error = normalized_sample_mae(prediction_stack[..., index], targets, masks, normalizer_std)
        errors.append(error)
    error_matrix = torch.stack(errors, dim=1)
    itr_idx = EXPERT_ORDER.index("iTransformer")
    itr_error = error_matrix[:, itr_idx]
    for index, expert in enumerate(EXPERT_ORDER):
        wins = error_matrix[:, index] < itr_error
        improvement = itr_error[wins] - error_matrix[:, index][wins]
        per_expert.append(
            {
                "expert": expert,
                "pct_windows_beats_iTransformer": float(wins.to(torch.float32).mean().item() * 100.0),
                "mean_improvement_when_beats_iTransformer": float(improvement.mean().item()) if bool(wins.any()) else 0.0,
                "unique_oracle_wins_percent": float((error_matrix.argmin(dim=1) == index).to(torch.float32).mean().item() * 100.0),
            }
        )
    centered = error_matrix - error_matrix.mean(dim=0, keepdim=True)
    corr = (centered.T @ centered) / max(error_matrix.shape[0] - 1, 1)
    denom = centered.std(dim=0, unbiased=True).clamp_min(1e-8)
    corr = corr / (denom[:, None] * denom[None, :])
    return {
        "per_expert": per_expert,
        "error_correlation": {
            EXPERT_ORDER[i]: {EXPERT_ORDER[j]: float(corr[i, j].item()) for j in range(len(EXPERT_ORDER))}
            for i in range(len(EXPERT_ORDER))
        },
    }


def regime_analysis(cache: Mapping[str, Any], normalizer_std: torch.Tensor, mode_summaries: Sequence[Mapping[str, Any]], results_root: Path) -> dict[str, Any]:
    histories = cache["histories"].to(torch.float32)
    fingerprints = compute_history_fingerprints(histories)
    features = int(cache["num_features"])
    stat_names = ["mean", "std", "min", "max", "slope", "recent_slope", "diff_mean", "diff_std", "acf1", "acf2", "acf4", "acf8", "recent_mean_delta", "recent_std_delta", "trend_strength", "dominant_freq", "low_freq_energy", "high_freq_energy"]
    stat = {name: fingerprints[:, i * features : (i + 1) * features].mean(dim=1) for i, name in enumerate(stat_names)}
    regimes = {
        "high_trend": stat["trend_strength"] >= stat["trend_strength"].median(),
        "low_trend": stat["trend_strength"] < stat["trend_strength"].median(),
        "high_volatility": stat["diff_std"] >= stat["diff_std"].median(),
        "low_volatility": stat["diff_std"] < stat["diff_std"].median(),
        "strong_autocorrelation": stat["acf1"] >= stat["acf1"].median(),
        "weak_autocorrelation": stat["acf1"] < stat["acf1"].median(),
        "strong_periodicity": stat["low_freq_energy"] >= stat["low_freq_energy"].median(),
        "weak_periodicity": stat["low_freq_energy"] < stat["low_freq_energy"].median(),
        "distribution_shifted": (stat["recent_mean_delta"].abs() + stat["recent_std_delta"].abs()) >= (stat["recent_mean_delta"].abs() + stat["recent_std_delta"].abs()).median(),
    }
    expert_errors = torch.stack(
        [
            normalized_sample_mae(cache["prediction_stack"][..., index], cache["targets"], cache["target_masks"], normalizer_std)
            for index in range(len(EXPERT_ORDER))
        ],
        dim=1,
    )
    rows = []
    per_window_by_mode = {}
    for summary in mode_summaries:
        mode = str(summary["mode"])
        best = min(summary["per_seed"], key=lambda row: float(row["validation_mae"]))
        path = results_root / mode / f"seed_{best['seed']}" / "validation_per_window.csv"
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            per_window_by_mode[mode] = torch.tensor([float(row["normalized_mae"]) for row in reader], dtype=torch.float32)
    for regime, mask in regimes.items():
        if not bool(mask.any()):
            continue
        best_counts = torch.bincount(expert_errors[mask].argmin(dim=1), minlength=len(EXPERT_ORDER)).to(torch.float32)
        row = {
            "regime": regime,
            "windows": int(mask.sum().item()),
            "best_expert": EXPERT_ORDER[int(best_counts.argmax().item())],
            "best_expert_distribution": {EXPERT_ORDER[i]: float(best_counts[i].item() * 100.0 / max(mask.sum().item(), 1)) for i in range(len(EXPERT_ORDER))},
        }
        for mode, errors in per_window_by_mode.items():
            row[f"{mode}_mae"] = float(errors[mask].mean().item())
        rows.append(row)
    write_csv(results_root / "regime_analysis.csv", rows)
    return {"rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", default="cache/costarts_walkforward/router_train_20_60_cache.pt")
    parser.add_argument("--val-cache", default="cache/costarts_walkforward/router_val_60_80_cache.pt")
    parser.add_argument("--normalizer-checkpoint", default="checkpoints/costarts_walkforward/final_60/DLinear/best_expert.pt")
    parser.add_argument("--checkpoint-root", default="checkpoints/costarts_walkforward/fingerprint")
    parser.add_argument("--results-root", default="results/router_summary/costarts_walkforward/fingerprint")
    parser.add_argument("--modes", default="embedding_only,fingerprint_only,embedding_fingerprint")
    parser.add_argument("--seeds", default="7,11,13,17,19")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--max-queries", type=int, default=5)
    parser.add_argument("--query-cost", type=float, default=0.0)
    parser.add_argument("--query-threshold", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    train_cache_path = ROOT / args.train_cache
    val_cache_path = ROOT / args.val_cache
    train_cache = load_verified_cache(train_cache_path, "router_train_20_60")
    val_cache = load_verified_cache(val_cache_path, "router_val_60_80")
    normalizer_std = load_normalizer_std(ROOT / args.normalizer_checkpoint)
    if tuple(train_cache["expert_names"]) != tuple(val_cache["expert_names"]):
        raise ValueError("Expert order mismatch between train and validation caches")
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    summaries = [train_mode(mode, args, train_cache, val_cache, normalizer_std) for mode in modes]  # type: ignore[arg-type]
    comparison_rows = []
    for summary in summaries:
        comparison_rows.append(
            {
                "model": summary["model"],
                "mode": summary["mode"],
                "validation_mae_mean": summary["validation_mae_mean"],
                "validation_mae_std": summary["validation_mae_std"],
                "validation_mse_mean": summary["validation_mse_mean"],
                "validation_mse_std": summary["validation_mse_std"],
                "average_queries_mean": summary["average_queries_mean"],
                "average_queries_std": summary["average_queries_std"],
                "parameter_count": summary["parameter_count"],
                "improvement_vs_costar_baseline_mae": 0.37468 - float(summary["validation_mae_mean"]),
                "improvement_vs_best_single_iTransformer_mae": 0.37655 - float(summary["validation_mae_mean"]),
            }
        )
    results_root = ROOT / args.results_root
    write_csv(results_root / "comparison.csv", comparison_rows)
    complementarity = expert_complementarity(val_cache, normalizer_std)
    (results_root / "expert_complementarity.json").write_text(json.dumps(complementarity, indent=2), encoding="utf-8")
    regimes = regime_analysis(val_cache, normalizer_std, summaries, results_root)
    summary = {
        "method": "Walk-forward Sequential COSTAR-TS fingerprint comparison",
        "train_cache": args.train_cache,
        "val_cache": args.val_cache,
        "train_cache_sha256": sha256_file(train_cache_path),
        "val_cache_sha256": sha256_file(val_cache_path),
        "modes": summaries,
        "comparison": comparison_rows,
        "individual_expert_validation": individual_expert_metrics(val_cache, normalizer_std),
        "expert_complementarity": complementarity,
        "regime_analysis": regimes,
        "fingerprint_features": [
            "mean", "std", "min", "max", "slope", "recent_slope", "diff_mean", "diff_std",
            "acf_lag_1", "acf_lag_2", "acf_lag_4", "acf_lag_8", "recent_mean_minus_mean",
            "recent_std_minus_std", "trend_strength", "dominant_frequency", "low_frequency_energy_ratio",
            "high_frequency_energy_ratio",
        ],
        "leakage_checks": "fingerprints use only input histories from cached windows; final test cache refused",
        "safety": "NO TEST DATA USED",
    }
    (results_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"comparison": comparison_rows, "safety": summary["safety"]}, indent=2))


if __name__ == "__main__":
    main()
