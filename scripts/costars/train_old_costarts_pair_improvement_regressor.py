"""Train an ETTh1 old-COSTARTS pair-improvement regressor.

The model sees only the causal history and predicts one improvement score for
each frozen-expert pair, where improvement is measured against the
validation-selected fixed pair:

    fixed_pair_mae - candidate_pair_mae
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import asdict
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

from scripts.costars.train_etth2_pair_selector import (
    DEFAULT_SEEDS,
    HistoryPairSelector,
    MARGIN_BINS,
    PairSelectorConfig,
    binary_auc,
    json_default,
    pair_class_order,
    pair_error_matrices,
    pair_name_to_index,
    read_csv_dicts,
    run_model,
    state_dict_hash,
    write_csv,
)
from scripts.costars.train_old_costarts_pair_selector import (
    EXPECTED_EXPERTS,
    OLD_COSTARTS_HASHES,
    build_baselines,
    load_reference_baselines,
    load_verified_old_costarts_cache,
    sha256_file,
    validate_cache_pair,
)


DEFAULT_RESULTS_ROOT = "results/router_summary/costarts/pair_improvement_regressor"
DEFAULT_CHECKPOINT_ROOT = "checkpoints/costarts/pair_improvement_regressor"
DEFAULT_CLASSIFIER_RESULTS = "results/router_summary/costarts/pair_selector/per_seed_results.csv"


class HistoryPairImprovementRegressor(HistoryPairSelector):
    """Same small history encoder as the classifier, with regression outputs."""


class PairImprovementDataset(Dataset):
    def __init__(
        self,
        cache: Mapping[str, Any],
        pair_mae: torch.Tensor,
        fixed_pair_index: int,
    ) -> None:
        if cache["split_role"] != "router_train":
            raise ValueError("PairImprovementDataset may only use router_train")
        self.histories = cache["histories"].to(torch.float32)
        self.improvement_targets = pair_improvement_targets(pair_mae, fixed_pair_index).to(torch.float32)
        self.source_indices = cache["sample_indices"].to(torch.long)

    def __len__(self) -> int:
        return int(self.histories.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "history": self.histories[index],
            "improvement_target": self.improvement_targets[index],
            "source_index": self.source_indices[index],
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def pair_improvement_targets(pair_mae: torch.Tensor, fixed_pair_index: int) -> torch.Tensor:
    if fixed_pair_index < 0 or fixed_pair_index >= pair_mae.shape[1]:
        raise ValueError("fixed_pair_index is out of range")
    fixed_error = pair_mae[:, fixed_pair_index].view(-1, 1)
    return fixed_error - pair_mae


def select_validation_fixed_pair(val_pair_mae: torch.Tensor, pairs: Sequence[Mapping[str, Any]]) -> tuple[int, str]:
    mean_errors = val_pair_mae.mean(dim=0)
    index = int(mean_errors.argmin().item())
    return index, str(pairs[index]["pair"])


def threshold_grid(predicted_max: torch.Tensor) -> list[float]:
    values = predicted_max.detach().cpu().numpy()
    quantiles = np.quantile(values, np.linspace(0.0, 1.0, 41)).tolist()
    fixed = [-math.inf, -0.05, -0.025, -0.01, -0.005, -0.001, 0.0, 0.001, 0.005, 0.01, 0.025, 0.05]
    return sorted(set(float(v) for v in [*fixed, *quantiles]))


def evaluate_scores(
    scores: torch.Tensor,
    pair_mae: torch.Tensor,
    pair_mse: torch.Tensor,
    fixed_pair_index: int,
    threshold: Optional[float],
) -> dict[str, Any]:
    predicted_best_score, predicted_best_pair = scores.max(dim=1)
    if threshold is None:
        selected_pair = predicted_best_pair
        threshold_label = "none_always_best_predicted"
    else:
        selected_pair = torch.where(
            predicted_best_score > threshold,
            predicted_best_pair,
            torch.full_like(predicted_best_pair, fixed_pair_index),
        )
        threshold_label = float(threshold)
    selected_mae = pair_mae.gather(1, selected_pair.view(-1, 1)).squeeze(1)
    selected_mse = pair_mse.gather(1, selected_pair.view(-1, 1)).squeeze(1)
    fixed_mae = pair_mae[:, fixed_pair_index]
    oracle_mae = pair_mae.min(dim=1).values
    actual_improvement = fixed_mae - selected_mae
    switched = selected_pair != fixed_pair_index
    beneficial = actual_improvement > 0
    harmful = actual_improvement < 0
    if bool(switched.any()):
        selected_switch_win_rate = float(beneficial[switched].to(torch.float32).mean().item() * 100.0)
    else:
        selected_switch_win_rate = 0.0
    candidate_actual_improvement = fixed_mae - pair_mae.gather(1, predicted_best_pair.view(-1, 1)).squeeze(1)
    candidate_mask = predicted_best_pair != fixed_pair_index
    if bool(candidate_mask.any()) and bool((candidate_actual_improvement[candidate_mask] > 0).any()) and bool((candidate_actual_improvement[candidate_mask] < 0).any()):
        auc = binary_auc(
            predicted_best_score[candidate_mask].detach().cpu().numpy(),
            (candidate_actual_improvement[candidate_mask] > 0).detach().cpu().numpy().astype(bool),
        )
    else:
        auc = float("nan")
    if float(predicted_best_score.std(unbiased=False).item()) > 0 and float(candidate_actual_improvement.std(unbiased=False).item()) > 0:
        corr = float(torch.corrcoef(torch.stack([predicted_best_score, candidate_actual_improvement]))[0, 1].item())
    else:
        corr = float("nan")
    return {
        "threshold": threshold_label,
        "selected_pair_mae": float(selected_mae.mean().item()),
        "selected_pair_mse": float(selected_mse.mean().item()),
        "fixed_pair_mae": float(fixed_mae.mean().item()),
        "oracle_pair_mae": float(oracle_mae.mean().item()),
        "regret_to_oracle_pair": float((selected_mae - oracle_mae).mean().item()),
        "improvement_over_fixed_pair": float((fixed_mae.mean() - selected_mae.mean()).item()),
        "switch_rate": float(switched.to(torch.float32).mean().item() * 100.0),
        "selected_switch_win_rate": selected_switch_win_rate,
        "mean_improvement_on_winning_windows": float(actual_improvement[beneficial].mean().item()) if bool(beneficial.any()) else 0.0,
        "mean_harm_on_losing_windows": float((-actual_improvement[harmful]).mean().item()) if bool(harmful.any()) else 0.0,
        "predicted_actual_improvement_correlation": corr,
        "beneficial_switch_auc": auc,
        "selected_pair_indices": selected_pair.detach().cpu(),
        "predicted_best_pair_indices": predicted_best_pair.detach().cpu(),
        "predicted_best_scores": predicted_best_score.detach().cpu(),
        "candidate_actual_improvement": candidate_actual_improvement.detach().cpu(),
        "selected_actual_improvement": actual_improvement.detach().cpu(),
        "selected_mae": selected_mae.detach().cpu(),
        "fixed_mae_per_window": fixed_mae.detach().cpu(),
    }


def select_threshold_on_validation(
    scores: torch.Tensor,
    pair_mae: torch.Tensor,
    pair_mse: torch.Tensor,
    fixed_pair_index: int,
) -> tuple[float, list[dict[str, Any]], dict[str, Any]]:
    predicted_max = scores.max(dim=1).values
    rows = []
    best_threshold = -math.inf
    best_metrics: Optional[dict[str, Any]] = None
    for threshold in threshold_grid(predicted_max):
        metrics = evaluate_scores(scores, pair_mae, pair_mse, fixed_pair_index, threshold)
        rows.append({
            "threshold": threshold,
            "selected_pair_mae": metrics["selected_pair_mae"],
            "selected_pair_mse": metrics["selected_pair_mse"],
            "improvement_over_fixed_pair": metrics["improvement_over_fixed_pair"],
            "switch_rate": metrics["switch_rate"],
            "selected_switch_win_rate": metrics["selected_switch_win_rate"],
            "predicted_actual_improvement_correlation": metrics["predicted_actual_improvement_correlation"],
            "beneficial_switch_auc": metrics["beneficial_switch_auc"],
        })
        if best_metrics is None or metrics["selected_pair_mae"] < best_metrics["selected_pair_mae"] - 1e-12:
            best_threshold = threshold
            best_metrics = metrics
    if best_metrics is None:
        raise RuntimeError("No threshold candidates evaluated")
    return best_threshold, rows, best_metrics


def metrics_for_summary(metrics: Mapping[str, Any], pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keep = {
        key: value
        for key, value in metrics.items()
        if key not in {
            "selected_pair_indices",
            "predicted_best_pair_indices",
            "predicted_best_scores",
            "candidate_actual_improvement",
            "selected_actual_improvement",
            "selected_mae",
            "fixed_mae_per_window",
        }
    }
    selected = metrics["selected_pair_indices"]
    counts = torch.bincount(selected, minlength=len(pairs)).to(torch.float32)
    total = float(selected.numel())
    keep["selection_distribution"] = {
        str(pairs[index]["pair"]): {
            "count": int(counts[index].item()),
            "percentage": float(counts[index].item() * 100.0 / total),
        }
        for index in range(len(pairs))
    }
    return keep


def margin_group_rows(
    pair_mae: torch.Tensor,
    metrics: Mapping[str, Any],
) -> list[dict[str, Any]]:
    sorted_errors = torch.sort(pair_mae, dim=1).values
    margins = sorted_errors[:, 1] - sorted_errors[:, 0]
    selected_mae = metrics["selected_mae"]
    oracle_mae = pair_mae.min(dim=1).values.cpu()
    fixed_mae = metrics["fixed_mae_per_window"]
    actual_improvement = metrics["selected_actual_improvement"]
    rows = []
    for label, lower, upper in MARGIN_BINS:
        mask = torch.ones_like(margins, dtype=torch.bool)
        if lower is not None:
            mask &= margins > lower
        if upper is not None:
            mask &= margins <= upper
        count = int(mask.sum().item())
        row = {"margin_group": label, "count": count}
        if count:
            row.update({
                "selected_pair_mae": float(selected_mae[mask].mean().item()),
                "regret_to_oracle_pair": float((selected_mae[mask] - oracle_mae[mask]).mean().item()),
                "improvement_over_fixed_pair": float((fixed_mae[mask].mean() - selected_mae[mask].mean()).item()),
                "switch_win_rate": float((actual_improvement[mask] > 0).to(torch.float32).mean().item() * 100.0),
            })
        rows.append(row)
    return rows


def switch_score_rows(
    val_cache: Mapping[str, Any],
    no_threshold_metrics: Mapping[str, Any],
    threshold_metrics: Mapping[str, Any],
    pairs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for row in range(int(val_cache["num_windows"])):
        no_pair = int(no_threshold_metrics["selected_pair_indices"][row].item())
        gated_pair = int(threshold_metrics["selected_pair_indices"][row].item())
        rows.append({
            "row": row,
            "sample_index": int(val_cache["sample_indices"][row].item()),
            "predicted_best_pair": pairs[int(no_threshold_metrics["predicted_best_pair_indices"][row].item())]["pair"],
            "predicted_best_improvement": float(no_threshold_metrics["predicted_best_scores"][row].item()),
            "candidate_actual_improvement": float(no_threshold_metrics["candidate_actual_improvement"][row].item()),
            "no_threshold_selected_pair": pairs[no_pair]["pair"],
            "threshold_selected_pair": pairs[gated_pair]["pair"],
            "threshold_selected_actual_improvement": float(threshold_metrics["selected_actual_improvement"][row].item()),
        })
    return rows


def train_one_seed(
    seed: int,
    config: PairSelectorConfig,
    train_cache: Mapping[str, Any],
    val_cache: Mapping[str, Any],
    fixed_pair_index: int,
    checkpoint_root: Path,
    results_root: Path,
    pairs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    set_seed(seed)
    device = torch.device(config.device)
    train_pair_mae, _ = pair_error_matrices(train_cache, pairs)
    val_pair_mae, val_pair_mse = pair_error_matrices(val_cache, pairs)
    dataset = PairImprovementDataset(train_cache, train_pair_mae, fixed_pair_index)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, generator=generator, num_workers=0)
    model = HistoryPairImprovementRegressor(
        input_len=config.input_len,
        num_features=config.num_features,
        hidden_dim=config.hidden_dim,
        num_pairs=len(pairs),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_state = None
    best_epoch = 0
    best_metrics = None
    best_threshold = -math.inf
    best_val_mae = math.inf
    stale_epochs = 0
    history_rows = []

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            scores = model(batch["history"].to(device))
            target = batch["improvement_target"].to(device)
            loss = F.smooth_l1_loss(scores, target)
            loss.backward()
            if config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()
            batch_size = int(batch["history"].shape[0])
            loss_sum += float(loss.item()) * batch_size
            count += batch_size

        val_scores = run_model(model, val_cache["histories"], config.batch_size, device)
        threshold, _, threshold_metrics = select_threshold_on_validation(val_scores, val_pair_mae, val_pair_mse, fixed_pair_index)
        no_threshold_metrics = evaluate_scores(val_scores, val_pair_mae, val_pair_mse, fixed_pair_index, None)
        history_rows.append({
            "epoch": epoch,
            "train_loss": loss_sum / max(count, 1),
            "no_threshold_val_mae": no_threshold_metrics["selected_pair_mae"],
            "threshold_val_mae": threshold_metrics["selected_pair_mae"],
            "selected_threshold": threshold,
            "threshold_switch_rate": threshold_metrics["switch_rate"],
        })
        if threshold_metrics["selected_pair_mae"] < best_val_mae - 1e-12:
            best_val_mae = threshold_metrics["selected_pair_mae"]
            best_epoch = epoch
            best_threshold = threshold
            best_metrics = threshold_metrics
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= config.patience:
            break

    if best_state is None or best_metrics is None:
        raise RuntimeError("Training produced no checkpoint")
    model.load_state_dict(best_state)
    val_scores = run_model(model, val_cache["histories"], config.batch_size, device)
    no_threshold_metrics = evaluate_scores(val_scores, val_pair_mae, val_pair_mse, fixed_pair_index, None)
    selected_threshold, threshold_rows, threshold_metrics = select_threshold_on_validation(
        val_scores,
        val_pair_mae,
        val_pair_mse,
        fixed_pair_index,
    )
    if abs(selected_threshold - best_threshold) > 1e-12:
        best_threshold = selected_threshold

    seed_checkpoint_dir = checkpoint_root / f"seed_{seed}"
    seed_results_dir = results_root / f"seed_{seed}"
    seed_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    seed_results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_state_hash = state_dict_hash(best_state)
    checkpoint = {
        "seed": seed,
        "dataset": "ETTh1",
        "source": "old_costarts_router_caches",
        "input_len": config.input_len,
        "horizon": config.horizon,
        "expert_order": list(EXPECTED_EXPERTS),
        "pair_class_order": list(pairs),
        "cache_hashes": dict(OLD_COSTARTS_HASHES),
        "fixed_pair_index": fixed_pair_index,
        "fixed_pair": pairs[fixed_pair_index]["pair"],
        "model_configuration": model.config_dict(),
        "training_configuration": asdict(config),
        "target": "fixed_pair_error_minus_candidate_pair_error",
        "loss": "smooth_l1",
        "best_epoch": best_epoch,
        "selected_threshold": best_threshold,
        "router_validation_mae": threshold_metrics["selected_pair_mae"],
        "checkpoint_hash": checkpoint_state_hash,
        "state_dict": best_state,
    }
    checkpoint_path = seed_checkpoint_dir / "best_old_costarts_pair_improvement_regressor.pt"
    torch.save(checkpoint, checkpoint_path)

    write_csv(
        seed_results_dir / "training_history.csv",
        history_rows,
        ("epoch", "train_loss", "no_threshold_val_mae", "threshold_val_mae", "selected_threshold", "threshold_switch_rate"),
    )
    write_csv(
        seed_results_dir / "threshold_search.csv",
        threshold_rows,
        (
            "threshold",
            "selected_pair_mae",
            "selected_pair_mse",
            "improvement_over_fixed_pair",
            "switch_rate",
            "selected_switch_win_rate",
            "predicted_actual_improvement_correlation",
            "beneficial_switch_auc",
        ),
    )
    write_csv(
        results_root / f"validation_switch_scores_seed_{seed}.csv",
        switch_score_rows(val_cache, no_threshold_metrics, threshold_metrics, pairs),
        (
            "row",
            "sample_index",
            "predicted_best_pair",
            "predicted_best_improvement",
            "candidate_actual_improvement",
            "no_threshold_selected_pair",
            "threshold_selected_pair",
            "threshold_selected_actual_improvement",
        ),
    )
    seed_summary = {
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_ran": len(history_rows),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_state_hash": checkpoint_state_hash,
        "checkpoint_file_hash": sha256_file(checkpoint_path),
        "fixed_pair": pairs[fixed_pair_index]["pair"],
        "selected_threshold": best_threshold,
        "no_threshold": metrics_for_summary(no_threshold_metrics, pairs),
        "threshold": metrics_for_summary(threshold_metrics, pairs),
        "margin_groups_threshold": margin_group_rows(val_pair_mae, threshold_metrics),
    }
    (seed_results_dir / "seed_summary.json").write_text(json.dumps(seed_summary, indent=2, default=json_default), encoding="utf-8")
    return seed_summary


def aggregate_metric_rows(seed_summaries: Sequence[Mapping[str, Any]], mode: str) -> list[dict[str, Any]]:
    rows = []
    for summary in seed_summaries:
        metrics = summary[mode]
        rows.append({
            "seed": summary["seed"],
            "mode": mode,
            "best_epoch": summary["best_epoch"],
            "selected_threshold": summary["selected_threshold"] if mode == "threshold" else "none",
            "selected_pair_mae": metrics["selected_pair_mae"],
            "selected_pair_mse": metrics["selected_pair_mse"],
            "regret_to_oracle_pair": metrics["regret_to_oracle_pair"],
            "improvement_over_fixed_pair": metrics["improvement_over_fixed_pair"],
            "switch_rate": metrics["switch_rate"],
            "selected_switch_win_rate": metrics["selected_switch_win_rate"],
            "predicted_actual_improvement_correlation": metrics["predicted_actual_improvement_correlation"],
            "beneficial_switch_auc": metrics["beneficial_switch_auc"],
            "checkpoint_state_hash": summary["checkpoint_state_hash"],
            "checkpoint_file_hash": summary["checkpoint_file_hash"],
        })
    return rows


def aggregate_results(
    seed_summaries: Sequence[Mapping[str, Any]],
    results_root: Path,
    baselines: Mapping[str, Any],
    fixed_pair: str,
    classifier_rows: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    per_seed_rows = aggregate_metric_rows(seed_summaries, "no_threshold") + aggregate_metric_rows(seed_summaries, "threshold")
    write_csv(
        results_root / "per_seed_results.csv",
        per_seed_rows,
        (
            "seed",
            "mode",
            "best_epoch",
            "selected_threshold",
            "selected_pair_mae",
            "selected_pair_mse",
            "regret_to_oracle_pair",
            "improvement_over_fixed_pair",
            "switch_rate",
            "selected_switch_win_rate",
            "predicted_actual_improvement_correlation",
            "beneficial_switch_auc",
            "checkpoint_state_hash",
            "checkpoint_file_hash",
        ),
    )
    aggregate_rows = []
    for mode in ("no_threshold", "threshold"):
        mode_rows = [row for row in per_seed_rows if row["mode"] == mode]
        for metric in (
            "selected_pair_mae",
            "selected_pair_mse",
            "regret_to_oracle_pair",
            "improvement_over_fixed_pair",
            "switch_rate",
            "selected_switch_win_rate",
            "predicted_actual_improvement_correlation",
            "beneficial_switch_auc",
        ):
            values = np.array([float(row[metric]) for row in mode_rows], dtype=float)
            aggregate_rows.append({
                "method": f"improvement_regressor_{mode}",
                "metric": metric,
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values)),
                "min": float(np.nanmin(values)),
                "max": float(np.nanmax(values)),
            })
    if classifier_rows:
        values = np.array([float(row["selected_pair_mae"]) for row in classifier_rows], dtype=float)
        aggregate_rows.append({
            "method": "current_exact_oracle_pair_classifier",
            "metric": "selected_pair_mae",
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
        })
    write_csv(results_root / "aggregate_results.csv", aggregate_rows, ("method", "metric", "mean", "std", "min", "max"))

    margin_group_aggregate = []
    for label, _, _ in MARGIN_BINS:
        rows = []
        for summary in seed_summaries:
            rows.extend(row for row in summary["margin_groups_threshold"] if row["margin_group"] == label)
        item = {"margin_group": label, "count": int(rows[0]["count"])}
        for metric in ("selected_pair_mae", "regret_to_oracle_pair", "improvement_over_fixed_pair", "switch_win_rate"):
            values = [float(row[metric]) for row in rows if metric in row]
            item[f"{metric}_mean"] = float(np.mean(values)) if values else ""
            item[f"{metric}_std"] = float(np.std(values)) if values else ""
        margin_group_aggregate.append(item)

    summary = {
        "dataset": "ETTh1",
        "source": "old_costarts_router_caches",
        "seeds": [summary["seed"] for summary in seed_summaries],
        "cache_hashes": dict(OLD_COSTARTS_HASHES),
        "fixed_pair": fixed_pair,
        "fixed_pair_selection": "validation-selected best fixed pair on router_val",
        "baselines": baselines,
        "old_classifier_per_seed": list(classifier_rows),
        "per_seed": per_seed_rows,
        "aggregate": aggregate_rows,
        "margin_group_aggregate_threshold": margin_group_aggregate,
        "success_checks": success_checks(aggregate_rows, baselines, len(seed_summaries)),
        "leakage_assertions": {
            "training_loader_split": "router_train",
            "router_validation_used_for_threshold_and_checkpoint_selection": True,
            "test_cache_created": False,
            "forecasting_experts_retrained": False,
            "inference_inputs": ["history"],
        },
    }
    (results_root / "aggregate_summary.json").write_text(json.dumps(summary, indent=2, default=json_default), encoding="utf-8")
    write_report(results_root / "pair_improvement_regressor_report.md", summary)
    return summary


def success_checks(aggregate_rows: Sequence[Mapping[str, Any]], baselines: Mapping[str, Any], num_seeds: int) -> dict[str, Any]:
    fixed_mae = float(baselines["fixed_pair"]["mae"])
    threshold_mae = next(row for row in aggregate_rows if row["method"] == "improvement_regressor_threshold" and row["metric"] == "selected_pair_mae")
    switch_rate = next(row for row in aggregate_rows if row["method"] == "improvement_regressor_threshold" and row["metric"] == "switch_rate")
    auc = next(row for row in aggregate_rows if row["method"] == "improvement_regressor_threshold" and row["metric"] == "beneficial_switch_auc")
    per_seed_improvements = [
        row for row in aggregate_rows
        if False
    ]
    return {
        "beats_fixed_pair_mean_validation_mae": float(threshold_mae["mean"]) < fixed_mae,
        "sensible_switch_rate_mean_percent": float(switch_rate["mean"]),
        "identifies_beneficial_switches_better_than_random": float(auc["mean"]) > 0.5 if not math.isnan(float(auc["mean"])) else False,
        "num_seeds": num_seeds,
    }


def write_report(path: Path, summary: Mapping[str, Any]) -> None:
    rows = summary["aggregate"]
    def metric(method: str, name: str) -> Mapping[str, Any]:
        return next(row for row in rows if row["method"] == method and row["metric"] == name)

    fixed_mae = summary["baselines"]["fixed_pair"]["mae"]
    old_classifier = metric("current_exact_oracle_pair_classifier", "selected_pair_mae")
    no_threshold_mae = metric("improvement_regressor_no_threshold", "selected_pair_mae")
    threshold_mae = metric("improvement_regressor_threshold", "selected_pair_mae")
    threshold_switch = metric("improvement_regressor_threshold", "switch_rate")
    threshold_auc = metric("improvement_regressor_threshold", "beneficial_switch_auc")
    threshold_win = metric("improvement_regressor_threshold", "selected_switch_win_rate")
    lines = [
        "# Old ETTh1 COSTARTS Pair-Improvement Regressor",
        "",
        "## Target",
        "",
        "`target[pair] = fixed_pair_error - candidate_pair_error`; positive means the candidate pair beats the fixed pair.",
        "",
        "## Validation Results",
        "",
        f"- Fixed validation-selected pair `{summary['fixed_pair']}` MAE `{fixed_mae:.6f}`.",
        f"- Current exact oracle-pair classifier mean MAE `{old_classifier['mean']:.6f}` +/- `{old_classifier['std']:.6f}`.",
        f"- Improvement regressor no-threshold mean MAE `{no_threshold_mae['mean']:.6f}` +/- `{no_threshold_mae['std']:.6f}`.",
        f"- Improvement regressor validation-threshold mean MAE `{threshold_mae['mean']:.6f}` +/- `{threshold_mae['std']:.6f}`.",
        f"- Threshold switch rate `{threshold_switch['mean']:.2f}%`; switched-window win rate `{threshold_win['mean']:.2f}%`.",
        f"- Beneficial/harmful switch AUC `{threshold_auc['mean']:.3f}`.",
        "",
        "## Success",
        "",
        json.dumps(summary["success_checks"], indent=2),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_seeds(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", default="cache/costarts_router_train_cache.pt")
    parser.add_argument("--val-cache", default="cache/costarts_router_val_cache.pt")
    parser.add_argument("--reference-comparison", default="results/router_summary/costarts_subset_utility/final_comparison.csv")
    parser.add_argument("--classifier-results", default=DEFAULT_CLASSIFIER_RESULTS)
    parser.add_argument("--checkpoint-root", default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def load_classifier_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main(argv: Optional[Sequence[str]] = None) -> dict[str, Any]:
    args = parse_args(argv)
    seeds = parse_seeds(args.seeds)
    if seeds != list(DEFAULT_SEEDS):
        raise ValueError(f"This run expects seeds {DEFAULT_SEEDS}, got {seeds}")
    config = PairSelectorConfig(
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        pair_target="regression",
        pair_temperature=0.0,
        device=args.device,
    )
    train_cache = load_verified_old_costarts_cache(Path(args.train_cache), "router_train")
    val_cache = load_verified_old_costarts_cache(Path(args.val_cache), "router_val")
    validate_cache_pair(train_cache, val_cache)
    pairs = pair_class_order()
    train_pair_mae, _ = pair_error_matrices(train_cache, pairs)
    val_pair_mae, val_pair_mse = pair_error_matrices(val_cache, pairs)
    fixed_pair_index, fixed_pair = select_validation_fixed_pair(val_pair_mae, pairs)
    results_root = Path(args.results_root)
    checkpoint_root = Path(args.checkpoint_root)
    results_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    (results_root / "pair_class_order.json").write_text(json.dumps(pairs, indent=2), encoding="utf-8")
    baselines = build_baselines(
        train_pair_mae,
        val_pair_mae,
        val_pair_mse,
        val_cache,
        pairs,
        load_reference_baselines(Path(args.reference_comparison)),
    )
    baselines["fixed_pair"] = {
        "pair": fixed_pair,
        "mae": float(val_pair_mae[:, fixed_pair_index].mean().item()),
        "mse": float(val_pair_mse[:, fixed_pair_index].mean().item()),
        "selection_split": "router_val",
        "average_experts_used": 2.0,
    }
    seed_summaries = [
        train_one_seed(seed, config, train_cache, val_cache, fixed_pair_index, checkpoint_root, results_root, pairs)
        for seed in seeds
    ]
    summary = aggregate_results(
        seed_summaries,
        results_root,
        baselines,
        fixed_pair,
        load_classifier_rows(Path(args.classifier_results)),
    )
    forbidden = [Path("cache/costarts_router_test_cache.pt"), Path("cache/costarts_locked_test_cache.pt")]
    created = [str(path) for path in forbidden if path.exists()]
    if created:
        raise RuntimeError(f"Forbidden test cache exists: {created}")
    return summary


if __name__ == "__main__":
    main()
