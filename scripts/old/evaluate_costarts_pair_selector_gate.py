"""Confidence-gated fallback for the COSTARTS direct pair selector.

This experiment keeps the frozen expert forecasts unchanged.  A direct pair
selector first predicts one two-expert pair from causal history.  Deployment
defaults to a fixed pair selected from router train/validation data, and only
switches to the predicted pair when a validation-selected confidence threshold
is met.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from scripts.old import train_costarts_pair_selector as pair_selector
except ImportError:
    import scripts.old.train_costarts_pair_selector as pair_selector


DEFAULT_RESULTS_DIR = "results/router_summary/costarts_pair_selector_gate"
DEFAULT_PAIR_OUTPUT_DIR = "checkpoints/costarts_pair_selector_gate"


@dataclass
class GatedPairSelectorConfig:
    train_cache_path: str = pair_selector.DEFAULT_TRAIN_CACHE
    val_cache_path: str = pair_selector.DEFAULT_VAL_CACHE
    final_comparison_json: str = pair_selector.DEFAULT_FINAL_COMPARISON_JSON
    output_dir: str = DEFAULT_PAIR_OUTPUT_DIR
    results_dir: str = DEFAULT_RESULTS_DIR
    seeds: tuple[int, ...] = (7, 11, 13)
    batch_size: int = 512
    max_epochs: int = 50
    patience: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    target_temperature: float = 0.03
    hard_label_weight: float = 1.0
    embedding_dim: int = 64
    hidden_dim: int = 64
    dropout: float = 0.1
    history_encoder_type: str = "current"
    device: str = "cpu"
    fixed_pair_selection: str = "router_val"
    threshold_steps: int = 201
    min_switch_rate: float = 0.05
    max_switch_rate: float = 0.50
    min_switched_win_rate: float = 0.50
    min_switched_mean_improvement: float = 0.0


def _jsonable(value: Any) -> Any:
    return pair_selector._jsonable(value)


def select_fixed_pair_class(
    train_pair_mae: torch.Tensor,
    val_pair_mae: torch.Tensor,
    *,
    selection: str,
) -> int:
    """Select one fixed pair using only router train/validation losses."""
    if selection == "router_train":
        scores = train_pair_mae.mean(dim=0)
    elif selection == "router_val":
        scores = val_pair_mae.mean(dim=0)
    elif selection == "router_train_plus_val":
        scores = torch.cat((train_pair_mae, val_pair_mae), dim=0).mean(dim=0)
    else:
        raise ValueError("selection must be router_train, router_val, or router_train_plus_val")
    return int(torch.argmin(scores))


def confidence_targets_from_pair_losses(
    predicted_pair_class: torch.Tensor,
    fixed_pair_class: int,
    pair_mae: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Build validation labels saying whether predicted pair beats fixed pair."""
    predicted_loss = pair_mae.gather(1, predicted_pair_class.to(torch.long).view(-1, 1)).squeeze(1)
    fixed_loss = pair_mae[:, int(fixed_pair_class)]
    improvement = fixed_loss - predicted_loss
    return {
        "predicted_pair_mae": predicted_loss,
        "fixed_pair_mae": fixed_loss,
        "improvement": improvement,
        "will_beat_fixed": improvement > 0,
    }


def confidence_scores_from_logits(logits: torch.Tensor, fixed_pair_class: int) -> dict[str, torch.Tensor]:
    """Derive confidence scores from pair logits produced from causal history."""
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(logits, dim=1)
    logit_top2 = torch.topk(logits, k=2, dim=1).values
    prob_top2 = torch.topk(probabilities, k=2, dim=1).values
    predicted_logit = logits.gather(1, predicted_class.view(-1, 1)).squeeze(1)
    predicted_probability = probabilities.gather(1, predicted_class.view(-1, 1)).squeeze(1)
    return {
        "max_probability": torch.max(probabilities, dim=1).values,
        "logit_margin": logit_top2[:, 0] - logit_top2[:, 1],
        "probability_margin": prob_top2[:, 0] - prob_top2[:, 1],
        "predicted_minus_fixed_logit": predicted_logit - logits[:, int(fixed_pair_class)],
        "predicted_minus_fixed_probability": predicted_probability - probabilities[:, int(fixed_pair_class)],
    }


def threshold_candidates(score: torch.Tensor, steps: int) -> torch.Tensor:
    if steps < 2:
        raise ValueError("steps must be at least 2")
    finite_score = score[torch.isfinite(score)]
    if finite_score.numel() == 0:
        raise ValueError("score must contain finite values")
    grid = torch.linspace(float(finite_score.min()), float(finite_score.max()), int(steps))
    return torch.cat((torch.tensor([-float("inf")]), grid, torch.tensor([float("inf")])))


def gated_prediction_for_threshold(
    *,
    cache: Mapping[str, Any],
    pair_index: torch.Tensor,
    predicted_pair_class: torch.Tensor,
    fixed_pair_class: int,
    score: torch.Tensor,
    threshold: float,
) -> dict[str, Any]:
    switch_mask = score >= float(threshold)
    predicted_pair = pair_index[predicted_pair_class.to(torch.long)]
    fixed_pair = pair_index[int(fixed_pair_class)].view(1, 2).expand(int(cache["num_windows"]), -1)
    selected_pair = torch.where(switch_mask[:, None], predicted_pair, fixed_pair)
    prediction = pair_selector._selected_pair_prediction(cache, selected_pair)
    mae, mse = pair_selector._mae_mse(prediction, cache)
    return {
        "mae": mae,
        "mse": mse,
        "selected_pair_indices": selected_pair,
        "switch_mask": switch_mask,
        "switch_rate": float(switch_mask.to(torch.float32).mean()),
        "average_experts_queried": 2.0,
    }


def select_confidence_threshold(
    *,
    cache: Mapping[str, Any],
    pair_index: torch.Tensor,
    predicted_pair_class: torch.Tensor,
    fixed_pair_class: int,
    score: torch.Tensor,
    score_name: str,
    pair_mae: torch.Tensor,
    steps: int,
    min_switch_rate: float = 0.0,
    max_switch_rate: float = 1.0,
    min_switched_win_rate: float | None = None,
    min_switched_mean_improvement: float | None = None,
) -> dict[str, Any]:
    """Choose confidence threshold by router-validation MAE under stability constraints."""
    if not 0.0 <= min_switch_rate <= max_switch_rate <= 1.0:
        raise ValueError("switch-rate constraints must satisfy 0 <= min <= max <= 1")
    labels = confidence_targets_from_pair_losses(predicted_pair_class, fixed_pair_class, pair_mae)
    rows: list[dict[str, Any]] = []
    for threshold in threshold_candidates(score, steps):
        gated = gated_prediction_for_threshold(
            cache=cache,
            pair_index=pair_index,
            predicted_pair_class=predicted_pair_class,
            fixed_pair_class=fixed_pair_class,
            score=score,
            threshold=float(threshold),
        )
        switched = gated["switch_mask"]
        if bool(switched.any()):
            switched_win_rate = float(labels["will_beat_fixed"][switched].to(torch.float32).mean())
            switched_mean_improvement = float(labels["improvement"][switched].mean())
        else:
            switched_win_rate = float("nan")
            switched_mean_improvement = float("nan")
        constraint_eligible = (
            float(min_switch_rate) <= float(gated["switch_rate"]) <= float(max_switch_rate)
        )
        if min_switched_win_rate is not None:
            constraint_eligible = constraint_eligible and bool(switched.any()) and (
                switched_win_rate >= float(min_switched_win_rate)
            )
        if min_switched_mean_improvement is not None:
            constraint_eligible = constraint_eligible and bool(switched.any()) and (
                switched_mean_improvement >= float(min_switched_mean_improvement)
            )
        rows.append(
            {
                "score_name": score_name,
                "threshold": float(threshold),
                "mae": gated["mae"],
                "mse": gated["mse"],
                "switch_rate": gated["switch_rate"],
                "switched_predicted_pair_win_rate": switched_win_rate,
                "switched_mean_mae_improvement": switched_mean_improvement,
                "average_experts_queried": 2.0,
                "constraint_eligible": constraint_eligible,
                "policy": (
                    "always_predicted_pair"
                    if math.isinf(float(threshold)) and float(threshold) < 0
                    else "always_fixed_pair"
                    if math.isinf(float(threshold)) and float(threshold) > 0
                    else "confidence_gated"
                ),
            }
        )
    eligible_rows = [row for row in rows if row["constraint_eligible"]]
    candidate_rows = eligible_rows if eligible_rows else rows
    candidate_rows = sorted(
        candidate_rows,
        key=lambda row: (
            float(row["mae"]),
            abs(float(row["switch_rate"]) - ((float(min_switch_rate) + float(max_switch_rate)) * 0.5)),
            str(row["score_name"]),
            float(row["threshold"]),
        ),
    )
    return candidate_rows[0] | {
        "threshold_rows": rows,
        "constraints": {
            "min_switch_rate": float(min_switch_rate),
            "max_switch_rate": float(max_switch_rate),
            "min_switched_win_rate": (
                None if min_switched_win_rate is None else float(min_switched_win_rate)
            ),
            "min_switched_mean_improvement": (
                None
                if min_switched_mean_improvement is None
                else float(min_switched_mean_improvement)
            ),
            "eligible_threshold_count": len(eligible_rows),
            "fallback_to_unconstrained": len(eligible_rows) == 0,
        },
    }


@torch.no_grad()
def logits_for_cache(
    model: pair_selector.CostartsPairSelector,
    cache: Mapping[str, Any],
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    logits: list[torch.Tensor] = []
    for offset in range(0, int(cache["num_windows"]), batch_size):
        history = cache["histories"][offset : offset + batch_size].to(device)
        logits.append(model(history).detach().cpu())
    return torch.cat(logits, dim=0)


def _metric_without_large_tensors(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if key not in {"selected_pair_indices", "switch_mask", "threshold_rows"}
    }


def run_seed(config: GatedPairSelectorConfig, seed: int) -> dict[str, Any]:
    train_cache = pair_selector._load_torch(config.train_cache_path)
    val_cache = pair_selector._load_torch(config.val_cache_path)
    pair_selector._assert_cache_pair(train_cache, val_cache, "router_val")

    training_config = pair_selector.PairSelectorTrainingConfig(
        train_cache_path=config.train_cache_path,
        val_cache_path=config.val_cache_path,
        output_dir=config.output_dir,
        results_dir=config.results_dir,
        final_comparison_json=config.final_comparison_json,
        batch_size=config.batch_size,
        max_epochs=config.max_epochs,
        patience=config.patience,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        grad_clip_norm=config.grad_clip_norm,
        seed=int(seed),
        target_temperature=config.target_temperature,
        hard_label_weight=config.hard_label_weight,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        history_encoder_type=config.history_encoder_type,
        device=config.device,
    )
    direct_result = pair_selector.train_pair_selector(training_config)
    checkpoint = pair_selector._load_torch(direct_result["checkpoint_path"])
    pair_index = checkpoint["pair_index"].to(torch.long)
    train_pair_mae, _ = pair_selector.masked_pair_mae_mse(train_cache, pair_index)
    val_pair_mae, _ = pair_selector.masked_pair_mae_mse(val_cache, pair_index)
    fixed_pair_class = select_fixed_pair_class(
        train_pair_mae,
        val_pair_mae,
        selection=config.fixed_pair_selection,
    )
    fixed_pair = pair_index[fixed_pair_class]
    device = torch.device(config.device)
    model = pair_selector.CostartsPairSelector(**checkpoint["router_config"]).to(device)
    model.load_state_dict(checkpoint["router_state_dict"])
    logits = logits_for_cache(model, val_cache, batch_size=config.batch_size, device=device)
    predicted_pair_class = torch.argmax(logits, dim=1)
    confidence_scores = confidence_scores_from_logits(logits, fixed_pair_class)

    val_fixed_pair = fixed_pair.view(1, 2).expand(int(val_cache["num_windows"]), -1)
    val_fixed_prediction = pair_selector._selected_pair_prediction(val_cache, val_fixed_pair)
    val_fixed_mae, val_fixed_mse = pair_selector._mae_mse(val_fixed_prediction, val_cache)
    val_best_pair = torch.argmin(val_pair_mae, dim=1)
    val_best_expert = val_cache["best_expert"].to(torch.long)
    validation_selected_fixed = {
        "mae": val_fixed_mae,
        "mse": val_fixed_mse,
        "average_experts_queried": 2.0,
        "best_pair_accuracy": float(
            (torch.full_like(val_best_pair, fixed_pair_class) == val_best_pair).to(torch.float32).mean()
        ),
        "best_individual_expert_top2_coverage": float(
            (val_fixed_pair == val_best_expert[:, None]).any(dim=1).to(torch.float32).mean()
        ),
        "selected_pair_class": fixed_pair_class,
        "selected_pair_indices": fixed_pair.tolist(),
        "selection_split": config.fixed_pair_selection,
    }

    threshold_results = []
    for score_name, score in confidence_scores.items():
        threshold_results.append(
            select_confidence_threshold(
                cache=val_cache,
                pair_index=pair_index,
                predicted_pair_class=predicted_pair_class,
                fixed_pair_class=fixed_pair_class,
                score=score,
                score_name=score_name,
                pair_mae=val_pair_mae,
                steps=config.threshold_steps,
                min_switch_rate=config.min_switch_rate,
                max_switch_rate=config.max_switch_rate,
                min_switched_win_rate=config.min_switched_win_rate,
                min_switched_mean_improvement=config.min_switched_mean_improvement,
            )
        )
    best_gate = sorted(
        threshold_results,
        key=lambda row: (float(row["mae"]), str(row["score_name"]), float(row["threshold"])),
    )[0]
    labels = confidence_targets_from_pair_losses(predicted_pair_class, fixed_pair_class, val_pair_mae)
    selected_threshold_rows = best_gate.pop("threshold_rows")
    result = {
        "seed": int(seed),
        "best_epoch": direct_result["best_epoch"],
        "pair_selector_checkpoint": direct_result["checkpoint_path"],
        "fixed_pair_selection": config.fixed_pair_selection,
        "fixed_pair_class": fixed_pair_class,
        "fixed_pair_indices": fixed_pair.tolist(),
        "fixed_pair_names": [
            tuple(val_cache["expert_names"])[int(fixed_pair[0])],
            tuple(val_cache["expert_names"])[int(fixed_pair[1])],
        ],
        "direct_pair_selector": direct_result["val"],
        "validation_selected_fixed_pair": validation_selected_fixed,
        "confidence_target_positive_rate": float(labels["will_beat_fixed"].to(torch.float32).mean()),
        "stability_constraints": {
            "min_switch_rate": config.min_switch_rate,
            "max_switch_rate": config.max_switch_rate,
            "min_switched_win_rate": config.min_switched_win_rate,
            "min_switched_mean_improvement": config.min_switched_mean_improvement,
        },
        "selected_gate": _metric_without_large_tensors(best_gate),
        "all_score_best_thresholds": [_metric_without_large_tensors(row) for row in threshold_results],
        "selected_score_threshold_rows": selected_threshold_rows,
        "average_experts_queried": 2.0,
    }
    result_path = Path(config.results_dir) / f"confidence_gate_seed{seed}.json"
    result_path.write_text(json.dumps(_jsonable(result), indent=2), encoding="utf-8")
    print(
        f"seed={seed} gate_mae={best_gate['mae']:.6f} gate_mse={best_gate['mse']:.6f} "
        f"score={best_gate['score_name']} threshold={best_gate['threshold']:.6f} "
        f"switch={best_gate['switch_rate']:.3f} win={best_gate['switched_predicted_pair_win_rate']:.3f}"
    )
    return result


def _mean_std(runs: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    fields = (
        "mae",
        "mse",
        "switch_rate",
        "switched_predicted_pair_win_rate",
        "average_experts_queried",
    )
    stats: dict[str, dict[str, float]] = {}
    for field in fields:
        values = []
        for run in runs:
            selected_gate = run["selected_gate"]
            if field in selected_gate and selected_gate[field] == selected_gate[field]:
                values.append(float(selected_gate[field]))
        if values:
            tensor = torch.tensor(values, dtype=torch.float32)
            stats[field] = {
                "mean": float(tensor.mean()),
                "std": float(tensor.std(unbiased=False)),
            }
    return stats


def run_experiment(config: GatedPairSelectorConfig) -> dict[str, Any]:
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    runs = [run_seed(config, seed) for seed in config.seeds]
    reference_rows = pair_selector._load_reference_rows(config.final_comparison_json)
    summary = {
        "config": asdict(config),
        "runs": runs,
        "mean_std": _mean_std(runs),
        "baselines": {
            "equal_average_all_experts": reference_rows.get("equal_average_all_experts", {}),
            "existing_predicted_top2_equal_average": reference_rows.get("predicted_top2_equal_average", {}),
            "direct_pair_selector_mean_mae_reference": 0.3515688180923462,
            "validation_selected_fixed_pair": runs[0]["validation_selected_fixed_pair"],
        },
        "model_selection": (
            "Pair-selector checkpoints are selected by router-validation MAE. "
            "Confidence score and threshold are selected on router-validation only under "
            "predeclared switch-rate and switched-window quality constraints; no test labels are used."
        ),
    }
    summary_path = results_dir / "confidence_gate_summary.json"
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    csv_path = results_dir / "confidence_gate_summary.csv"
    fields = [
        "seed",
        "val_mae",
        "val_mse",
        "switch_rate",
        "switched_predicted_pair_win_rate",
        "average_experts_queried",
        "fixed_pair_names",
        "score_name",
        "threshold",
        "constraint_eligible",
        "fallback_to_unconstrained",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            gate = run["selected_gate"]
            writer.writerow(
                {
                    "seed": run["seed"],
                    "val_mae": gate["mae"],
                    "val_mse": gate["mse"],
                    "switch_rate": gate["switch_rate"],
                    "switched_predicted_pair_win_rate": gate["switched_predicted_pair_win_rate"],
                    "average_experts_queried": gate["average_experts_queried"],
                    "fixed_pair_names": " + ".join(run["fixed_pair_names"]),
                    "score_name": gate["score_name"],
                    "threshold": gate["threshold"],
                    "constraint_eligible": gate.get("constraint_eligible", ""),
                    "fallback_to_unconstrained": gate.get("constraints", {}).get("fallback_to_unconstrained", ""),
                }
            )
    print(f"Saved confidence gate summary: {summary_path}")
    print(f"Saved confidence gate CSV: {csv_path}")
    print("\nConfidence-gated validation mean/std:")
    for field, values in summary["mean_std"].items():
        print(f"  {field}: mean={values['mean']:.6f}, std={values['std']:.6f}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate confidence-gated COSTARTS pair selection.")
    parser.add_argument("--train-cache", default=pair_selector.DEFAULT_TRAIN_CACHE)
    parser.add_argument("--val-cache", default=pair_selector.DEFAULT_VAL_CACHE)
    parser.add_argument("--final-comparison-json", default=pair_selector.DEFAULT_FINAL_COMPARISON_JSON)
    parser.add_argument("--output-dir", default=DEFAULT_PAIR_OUTPUT_DIR)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 11, 13])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--target-temperature", type=float, default=0.03)
    parser.add_argument("--hard-label-weight", type=float, default=1.0)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--history-encoder-type", choices=("current", "simple"), default="current")
    parser.add_argument(
        "--fixed-pair-selection",
        choices=("router_train", "router_val", "router_train_plus_val"),
        default="router_val",
    )
    parser.add_argument("--threshold-steps", type=int, default=201)
    parser.add_argument("--min-switch-rate", type=float, default=0.05)
    parser.add_argument("--max-switch-rate", type=float, default=0.50)
    parser.add_argument("--min-switched-win-rate", type=float, default=0.50)
    parser.add_argument("--min-switched-mean-improvement", type=float, default=0.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = GatedPairSelectorConfig(
        train_cache_path=args.train_cache,
        val_cache_path=args.val_cache,
        final_comparison_json=args.final_comparison_json,
        output_dir=args.output_dir,
        results_dir=args.results_dir,
        seeds=tuple(args.seeds),
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
        fixed_pair_selection=args.fixed_pair_selection,
        threshold_steps=args.threshold_steps,
        min_switch_rate=args.min_switch_rate,
        max_switch_rate=args.max_switch_rate,
        min_switched_win_rate=args.min_switched_win_rate,
        min_switched_mean_improvement=args.min_switched_mean_improvement,
        device=args.device,
    )
    run_experiment(config)


if __name__ == "__main__":
    main()
