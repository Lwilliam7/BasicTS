"""Train the ETTh2-style history-only pair selector on old ETTh1 COSTARTS caches."""

from __future__ import annotations

import argparse
import csv
import hashlib
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

from scripts.costars.train_etth2_pair_selector import (
    DEFAULT_SEEDS,
    DEFAULT_TEMPERATURE,
    EXPECTED_EXPERTS,
    FIXED_PAIR_NAME,
    HistoryPairSelector,
    MARGIN_BINS,
    PairSelectorConfig,
    binary_auc,
    confidence_distribution,
    confidence_signal_separation,
    cross_seed_agreement,
    dense_mixture_metrics,
    evaluate_logits,
    fixed_and_random_baselines,
    json_default,
    margin_group_metrics,
    metrics_for_json,
    pair_class_order,
    pair_error_matrices,
    pair_name_to_index,
    read_csv_dicts,
    run_model,
    selector_loss,
    soft_pair_targets,
    state_dict_hash,
    validation_confidence_rows,
    write_csv,
)


OLD_COSTARTS_HASHES = {
    "router_train": "631ca1142cbf563b257cc636e1562a1dc66141f7ad6fd51c48e9a9f1aa11e1e0",
    "router_val": "0d802634a7f1cead668382f5a85e31292157a4a15c1316049ef02268a8b56d3b",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_torch(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_verified_old_costarts_cache(path: Path, split_role: str) -> dict[str, Any]:
    observed_hash = sha256_file(path)
    expected_hash = OLD_COSTARTS_HASHES[split_role]
    if observed_hash != expected_hash:
        raise ValueError(f"{split_role} cache hash mismatch: {observed_hash} != {expected_hash}")
    cache = load_torch(path)
    validate_old_costarts_cache(cache, split_role)
    cache = dict(cache)
    if "absolute_window_starts" not in cache:
        cache["absolute_window_starts"] = cache["sample_indices"].clone()
    return cache


def validate_old_costarts_cache(cache: Mapping[str, Any], split_role: str) -> None:
    if cache["split_role"] != split_role:
        raise ValueError(f"Expected split {split_role}, found {cache['split_role']}")
    if tuple(cache["expert_names"]) != EXPECTED_EXPERTS:
        raise ValueError("Expert ordering changed")
    expected_n = 2053 if split_role == "router_train" else 613
    expected_shapes = {
        "histories": (expected_n, 96, 7),
        "targets": (expected_n, 12, 7),
        "target_masks": (expected_n, 12, 7),
        "prediction_stack": (expected_n, 12, 7, 5),
        "error_matrix": (expected_n, 5),
        "mse_matrix": (expected_n, 5),
        "sample_indices": (expected_n,),
    }
    for key, shape in expected_shapes.items():
        if tuple(cache[key].shape) != shape:
            raise ValueError(f"{split_role} {key} shape mismatch: {tuple(cache[key].shape)} != {shape}")
        tensor = cache[key]
        if tensor.dtype != torch.bool and not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"{split_role} {key} contains nonfinite values")
    if not torch.equal(cache["sample_indices"], torch.arange(expected_n, dtype=cache["sample_indices"].dtype)):
        raise ValueError(f"{split_role} sample_indices are not contiguous")


def validate_cache_pair(train_cache: Mapping[str, Any], val_cache: Mapping[str, Any]) -> None:
    train_ids = set(train_cache["sample_indices"].tolist())
    val_ids = set(val_cache["sample_indices"].tolist())
    if train_cache["split_role"] == val_cache["split_role"]:
        raise ValueError("train and validation caches have the same split role")
    if int(train_cache["num_windows"]) != 2053 or int(val_cache["num_windows"]) != 613:
        raise ValueError("Unexpected old COSTARTS cache window counts")
    # Old COSTARTS sample indices are split-local, so equality is expected; split_role is the isolation key.
    if not train_ids or not val_ids:
        raise ValueError("Missing sample ids")


class OldCostartsPairSelectorDataset(Dataset):
    def __init__(self, cache: Mapping[str, Any], pair_mae: torch.Tensor, pair_target: str, temperature: float) -> None:
        if cache["split_role"] != "router_train":
            raise ValueError("OldCostartsPairSelectorDataset may only be built from router_train")
        self.histories = cache["histories"].to(torch.float32)
        self.hard_targets = pair_mae.argmin(dim=1).to(torch.long)
        self.soft_targets = soft_pair_targets(pair_mae, temperature).to(torch.float32)
        self.source_indices = cache["sample_indices"].to(torch.long)
        self.pair_target = pair_target

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


def train_one_seed(
    seed: int,
    config: PairSelectorConfig,
    train_cache: Mapping[str, Any],
    val_cache: Mapping[str, Any],
    checkpoint_root: Path,
    results_root: Path,
    pairs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    set_seed(seed)
    device = torch.device(config.device)
    train_pair_mae, train_pair_mse = pair_error_matrices(train_cache, pairs)
    val_pair_mae, val_pair_mse = pair_error_matrices(val_cache, pairs)
    dataset = OldCostartsPairSelectorDataset(train_cache, train_pair_mae, config.pair_target, config.pair_temperature)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, generator=generator, num_workers=0)
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
    best_epoch = 0
    best_val_mae = math.inf
    stale_epochs = 0
    history_rows = []

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            batch_on_device = {
                "hard_target": batch["hard_target"].to(device),
            }
            if config.pair_target == "soft":
                batch_on_device["soft_target"] = batch["soft_target"].to(device)
            logits = model(batch["history"].to(device))
            loss = selector_loss(logits, batch_on_device, config.pair_target)
            loss.backward()
            if config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()
            batch_size = int(batch["history"].shape[0])
            loss_sum += float(loss.item()) * batch_size
            count += batch_size

        val_logits = run_model(model, val_cache["histories"], config.batch_size, device)
        val_metrics = evaluate_logits(val_logits, val_pair_mae, val_pair_mse, fixed_pair_index, val_hard, val_soft)
        history_rows.append({
            "epoch": epoch,
            "train_loss": loss_sum / max(count, 1),
            "router_val_selected_pair_mae": val_metrics["selected_pair_mae"],
            "router_val_selected_pair_mse": val_metrics["selected_pair_mse"],
            "router_val_exact_best_pair_accuracy": val_metrics["exact_best_pair_accuracy"],
        })
        if val_metrics["selected_pair_mae"] < best_val_mae - 1e-12:
            best_val_mae = val_metrics["selected_pair_mae"]
            best_epoch = epoch
            stale_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1
        if stale_epochs >= config.patience:
            break

    if best_state is None:
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
        "model_configuration": model.config_dict(),
        "training_configuration": asdict(config),
        "pair_target_mode": config.pair_target,
        "temperature": config.pair_temperature,
        "best_epoch": best_epoch,
        "router_validation_mae": val_metrics["selected_pair_mae"],
        "checkpoint_hash": checkpoint_state_hash,
        "checkpoint_hash_type": "sha256 over sorted state_dict tensors",
        "state_dict": best_state,
    }
    checkpoint_path = seed_checkpoint_dir / "best_old_costarts_pair_selector.pt"
    torch.save(checkpoint, checkpoint_path)
    confidence_rows = validation_confidence_rows(val_cache, val_logits, val_pair_mae, fixed_pair_index, pairs)
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
        "checkpoint_file_hash": sha256_file(checkpoint_path),
        "confidence_path": str(confidence_path),
        "train": metrics_for_json(train_metrics, pairs),
        "validation": metrics_for_json(val_metrics, pairs),
        "margin_groups": margin_group_metrics(val_pair_mae, val_pair_mse, val_logits, fixed_pair_index, pairs),
        "dense_pair_probability_mixture": dense_mixture_metrics(val_cache, val_logits, val_pair_mae, fixed_pair_index, pairs),
    }
    (seed_results_dir / "seed_summary.json").write_text(json.dumps(seed_summary, indent=2, default=json_default), encoding="utf-8")
    return seed_summary


def aggregate_margin_groups(seed_summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_group: dict[str, list[Mapping[str, Any]]] = {}
    for summary in seed_summaries:
        for row in summary["margin_groups"]:
            by_group.setdefault(row["margin_group"], []).append(row)
    rows = []
    for label, group_rows in by_group.items():
        count = int(group_rows[0].get("count", 0))
        row = {"margin_group": label, "count": count}
        if count:
            for metric in (
                "exact_pair_accuracy",
                "selected_pair_mae",
                "selected_pair_mse",
                "regret_to_oracle_pair",
                "confidence",
                "switch_win_rate_against_fixed",
            ):
                values = [float(item[metric]) for item in group_rows if metric in item]
                row[f"{metric}_mean"] = float(np.mean(values))
                row[f"{metric}_std"] = float(np.std(values))
        rows.append(row)
    order = {label: index for index, (label, _, _) in enumerate(MARGIN_BINS)}
    rows.sort(key=lambda row: order[row["margin_group"]])
    return rows


def mean_selection_distribution(seed_summaries: Sequence[Mapping[str, Any]], pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for pair in pairs:
        values = [
            float(summary["validation"]["class_selection_distribution"][pair["pair"]]["percentage"])
            for summary in seed_summaries
        ]
        result[pair["pair"]] = {
            "mean_percentage": float(np.mean(values)),
            "std_percentage": float(np.std(values)),
        }
    return result


def aggregate_results(
    seed_summaries: Sequence[Mapping[str, Any]],
    val_cache: Mapping[str, Any],
    results_root: Path,
    pairs: Sequence[Mapping[str, Any]],
    baselines: Mapping[str, Any],
) -> dict[str, Any]:
    per_seed_rows = []
    for summary in seed_summaries:
        metrics = summary["validation"]
        per_seed_rows.append({
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
        per_seed_rows,
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
    aggregate_rows = []
    for metric in (
        "selected_pair_mae",
        "selected_pair_mse",
        "regret_to_oracle_pair",
        "improvement_over_fixed_pair",
        "exact_best_pair_accuracy",
        "top_two_pair_coverage",
        "top_three_pair_coverage",
    ):
        values = np.array([float(row[metric]) for row in per_seed_rows], dtype=float)
        aggregate_rows.append({
            "metric": metric,
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
        })
    write_csv(results_root / "aggregate_results.csv", aggregate_rows, ("metric", "mean", "std", "min", "max"))
    confidence_frames = [read_csv_dicts(Path(summary["confidence_path"])) for summary in seed_summaries]
    cross_seed = cross_seed_agreement(confidence_frames, pairs)
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
    summary = {
        "dataset": "ETTh1",
        "source": "old_costarts_router_caches",
        "seeds": [summary["seed"] for summary in seed_summaries],
        "cache_hashes": dict(OLD_COSTARTS_HASHES),
        "expert_order": list(EXPECTED_EXPERTS),
        "pair_class_order": list(pairs),
        "baselines": baselines,
        "per_seed": per_seed_rows,
        "aggregate": aggregate_rows,
        "selection_distribution_mean": mean_selection_distribution(seed_summaries, pairs),
        "confidence_distribution": confidence_distribution(confidence_frames),
        "confidence_signal_separation": confidence_signal_separation(confidence_frames),
        "margin_group_aggregate": aggregate_margin_groups(seed_summaries),
        "cross_seed_agreement": cross_seed["summary"],
        "validation_source_alignment": {
            "num_validation_windows": int(val_cache["sample_indices"].shape[0]),
            "first_sample_index": int(val_cache["sample_indices"][0].item()),
            "last_sample_index": int(val_cache["sample_indices"][-1].item()),
        },
        "leakage_assertions": {
            "training_loader_split": "router_train",
            "router_validation_targets_in_training_dataloader": False,
            "test_cache_created": False,
            "forecasting_experts_retrained": False,
        },
    }
    (results_root / "aggregate_summary.json").write_text(json.dumps(summary, indent=2, default=json_default), encoding="utf-8")
    write_report(results_root / "old_costarts_pair_selector_report.md", summary)
    return summary


def write_report(path: Path, summary: Mapping[str, Any]) -> None:
    aggregate = {row["metric"]: row for row in summary["aggregate"]}
    baselines = summary["baselines"]
    mean_mae = aggregate["selected_pair_mae"]["mean"]
    fixed_mae = baselines["fixed_pair"]["mae"]
    equal_all = baselines["equal_average_all_experts"]["mae"]
    old_costarts = baselines["old_costarts_reference"]["mae"]
    predicted_top2 = baselines["predicted_top2_equal_average_reference"]["mae"]
    best_signal = summary["confidence_signal_separation"][0]
    lines = [
        "# Old ETTh1 COSTARTS Pair Selector Report",
        "",
        "## Validation Summary",
        "",
        f"- Fixed pair `{baselines['fixed_pair']['pair']}` MAE `{fixed_mae:.6f}`.",
        f"- New predicted pair selector mean MAE `{mean_mae:.6f}` +/- `{aggregate['selected_pair_mae']['std']:.6f}`.",
        f"- Improvement over fixed pair `{aggregate['improvement_over_fixed_pair']['mean']:.6f}`.",
        f"- Equal average all experts reference `{equal_all:.6f}`.",
        f"- Existing predicted top-2 equal-average reference `{predicted_top2:.6f}`.",
        f"- Old COSTARTS reference `{old_costarts:.6f}`.",
        f"- Exact pair accuracy `{aggregate['exact_best_pair_accuracy']['mean']:.2f}%`.",
        f"- Top-two pair coverage `{aggregate['top_two_pair_coverage']['mean']:.2f}%`.",
        f"- Cross-seed mean agreement `{summary['cross_seed_agreement']['mean_agreement_rate']:.3f}`.",
        "",
        "## Decision",
        "",
        f"The new pair selector beats old COSTARTS: `{mean_mae < old_costarts}`.",
        f"It beats existing predicted top-2 equal-average: `{mean_mae < predicted_top2}`.",
        f"It beats equal average of all five experts: `{mean_mae < equal_all}`.",
        f"Best diagnostic confidence separator: `{best_signal['signal']}` AUC `{best_signal['auc_helpful_vs_harmful']}`.",
        "",
        "No forecasting experts were retrained and no test cache was created.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def load_reference_baselines(path: Path) -> dict[str, Any]:
    rows = {}
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows[row["method"]] = row
    return rows


def build_baselines(
    train_pair_mae: torch.Tensor,
    val_pair_mae: torch.Tensor,
    val_pair_mse: torch.Tensor,
    val_cache: Mapping[str, Any],
    pairs: Sequence[Mapping[str, Any]],
    reference_rows: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    fixed_pair_index = pair_name_to_index(FIXED_PAIR_NAME, pairs)
    baselines = fixed_and_random_baselines(val_pair_mae, val_pair_mse, train_pair_mae, fixed_pair_index, pairs)
    all_prediction = val_cache["prediction_stack"].mean(dim=-1)
    mask = val_cache["target_masks"].to(torch.float32)
    denom = mask.sum().clamp_min(1.0)
    all_mae = (torch.abs(all_prediction - val_cache["targets"]) * mask).sum() / denom
    all_mse = ((all_prediction - val_cache["targets"]).pow(2) * mask).sum() / denom
    baselines["equal_average_all_experts"] = {
        "mae": float(all_mae.item()),
        "mse": float(all_mse.item()),
        "average_experts_used": 5.0,
    }
    for method, key in (
        ("old_costarts", "old_costarts_reference"),
        ("predicted_top2_equal_average", "predicted_top2_equal_average_reference"),
        ("improved_subset_utility_costarts", "improved_subset_utility_costarts_reference"),
        ("routerdc_hard_with_contrastive", "routerdc_hard_with_contrastive_reference"),
    ):
        row = reference_rows.get(method, {})
        baselines[key] = {
            "mae": float(row.get("mae", "nan")),
            "mse": float(row.get("mse", "nan")),
            "average_experts_queried": float(row.get("average_experts_queried", "nan")),
            "source": "results/router_summary/costarts_subset_utility/final_comparison.csv",
        }
    return baselines


def parse_seeds(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", default="cache/costarts_router_train_cache.pt")
    parser.add_argument("--val-cache", default="cache/costarts_router_val_cache.pt")
    parser.add_argument("--reference-comparison", default="results/router_summary/costarts_subset_utility/final_comparison.csv")
    parser.add_argument("--checkpoint-root", default="checkpoints/costarts/pair_selector")
    parser.add_argument("--results-root", default="results/router_summary/costarts/pair_selector")
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
        raise ValueError(f"This run expects seeds {DEFAULT_SEEDS}, got {seeds}")
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
    train_cache = load_verified_old_costarts_cache(Path(args.train_cache), "router_train")
    val_cache = load_verified_old_costarts_cache(Path(args.val_cache), "router_val")
    validate_cache_pair(train_cache, val_cache)
    pairs = pair_class_order()
    results_root = Path(args.results_root)
    checkpoint_root = Path(args.checkpoint_root)
    results_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    (results_root / "pair_class_order.json").write_text(json.dumps(pairs, indent=2), encoding="utf-8")
    train_pair_mae, _ = pair_error_matrices(train_cache, pairs)
    val_pair_mae, val_pair_mse = pair_error_matrices(val_cache, pairs)
    reference_rows = load_reference_baselines(Path(args.reference_comparison))
    baselines = build_baselines(train_pair_mae, val_pair_mae, val_pair_mse, val_cache, pairs, reference_rows)
    seed_summaries = [
        train_one_seed(seed, config, train_cache, val_cache, checkpoint_root, results_root, pairs)
        for seed in seeds
    ]
    summary = aggregate_results(seed_summaries, val_cache, results_root, pairs, baselines)
    forbidden = [
        Path("cache/costarts_router_test_cache.pt"),
        Path("cache/costarts_locked_test_cache.pt"),
    ]
    created = [str(path) for path in forbidden if path.exists()]
    if created:
        raise RuntimeError(f"Forbidden test cache exists: {created}")
    return summary


if __name__ == "__main__":
    main()
