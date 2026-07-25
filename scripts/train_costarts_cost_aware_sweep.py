"""Train cost-aware SubsetUtilityCOSTARTSRouter sweeps.

Each lambda/seed run gets its own checkpoint and result directory. This script
does not load forecasting experts; it trains only from subset-state caches.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Mapping, Sequence

if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    from scripts.train_costarts_subset_utility_router import (
        DEFAULT_RESULTS_DIR,
        DEFAULT_TRAIN_CACHE,
        DEFAULT_VAL_CACHE,
        SubsetUtilityTrainingConfig,
        train_subset_utility_costarts_router,
    )
except ImportError:
    from train_costarts_subset_utility_router import (
        DEFAULT_RESULTS_DIR,
        DEFAULT_TRAIN_CACHE,
        DEFAULT_VAL_CACHE,
        SubsetUtilityTrainingConfig,
        train_subset_utility_costarts_router,
    )


DEFAULT_LAMBDAS = (0.0, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1)
DEFAULT_SEEDS = (7, 11, 13, 17, 19)


def _parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _lambda_slug(value: float) -> str:
    return f"{value:g}".replace("-", "neg").replace(".", "p")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_baselines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[float, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(float(row["cost_coefficient"]), []).append(row)
    summary = []
    for coefficient, values in sorted(grouped.items()):
        maes = [float(row["validation_mae"]) for row in values if row.get("validation_mae") is not None]
        mses = [float(row["validation_mse"]) for row in values if row.get("validation_mse") is not None]
        regrets = [
            float(row["regret_to_oracle"])
            for row in values
            if row.get("regret_to_oracle") is not None
        ]
        objectives = [
            float(row["combined_cost_aware_objective"])
            for row in values
            if row.get("combined_cost_aware_objective") is not None
        ]
        avg_queries = [
            float(row["average_experts_selected"])
            for row in values
            if row.get("average_experts_selected") is not None
        ]
        avg_costs = [
            float(row["average_normalized_query_cost"])
            for row in values
            if row.get("average_normalized_query_cost") is not None
        ]
        summary.append(
            {
                "lambda": coefficient,
                "num_seeds": len(values),
                "mae_mean": mean(maes) if maes else math.nan,
                "mae_std": pstdev(maes) if len(maes) > 1 else 0.0,
                "mse_mean": mean(mses) if mses else math.nan,
                "mse_std": pstdev(mses) if len(mses) > 1 else 0.0,
                "regret_to_oracle_mean": mean(regrets) if regrets else math.nan,
                "regret_to_oracle_std": pstdev(regrets) if len(regrets) > 1 else 0.0,
                "objective_mean": mean(objectives) if objectives else math.nan,
                "objective_std": pstdev(objectives) if len(objectives) > 1 else 0.0,
                "average_experts_selected_mean": mean(avg_queries) if avg_queries else math.nan,
                "average_experts_selected_std": pstdev(avg_queries) if len(avg_queries) > 1 else 0.0,
                "average_normalized_query_cost_mean": mean(avg_costs) if avg_costs else math.nan,
                "average_normalized_query_cost_std": pstdev(avg_costs) if len(avg_costs) > 1 else 0.0,
            }
        )
    return summary


def _save_pareto_plot(path: Path, summary_rows: Sequence[Mapping[str, Any]], baselines: Sequence[Mapping[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping Pareto plot")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    x = [float(row["average_experts_selected_mean"]) for row in summary_rows]
    y = [float(row["mae_mean"]) for row in summary_rows]
    labels = [f"lambda={row['lambda']:g}" for row in summary_rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, marker="o", label="trained cost-aware COSTARTS")
    for x_value, y_value, label in zip(x, y, labels):
        ax.annotate(label, (x_value, y_value), fontsize=8, xytext=(4, 4), textcoords="offset points")
    for baseline in baselines:
        try:
            mae = float(baseline.get("mae") or baseline.get("MAE") or baseline.get("validation_mae"))
        except (TypeError, ValueError):
            continue
        method = str(baseline.get("method") or baseline.get("router") or baseline.get("name") or "baseline")
        ax.axhline(mae, linestyle="--", linewidth=0.8, alpha=0.35)
        ax.text(max(x) if x else 1.0, mae, method, fontsize=7, va="bottom")
    ax.set_xlabel("Average experts queried")
    ax.set_ylabel("Validation MAE")
    ax.set_title("COSTARTS Cost-Aware Pareto Curve")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train cost-aware COSTARTS subset-utility sweeps.")
    parser.add_argument("--train-cache", default=DEFAULT_TRAIN_CACHE)
    parser.add_argument("--val-cache", default=DEFAULT_VAL_CACHE)
    parser.add_argument("--output-root", default="checkpoints/costarts_subset_utility/cost_aware_sweep")
    parser.add_argument("--results-root", default=f"{DEFAULT_RESULTS_DIR}/cost_aware_sweep")
    parser.add_argument("--lambdas", default=",".join(str(value) for value in DEFAULT_LAMBDAS))
    parser.add_argument("--seeds", default=",".join(str(value) for value in DEFAULT_SEEDS))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--cost-mode", choices=("equal", "configured", "latency"), default="equal")
    parser.add_argument("--cost-file", default=None)
    parser.add_argument("--selection-metric", choices=("mae", "cost_aware_objective"), default="cost_aware_objective")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--baseline-comparison-csv", default=f"{DEFAULT_RESULTS_DIR}/final_comparison.csv")
    parser.add_argument("--skip-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lambdas = _parse_csv_floats(args.lambdas)
    seeds = _parse_csv_ints(args.seeds)
    output_root = Path(args.output_root)
    results_root = Path(args.results_root)
    run_rows: list[dict[str, Any]] = []

    for coefficient in lambdas:
        for seed in seeds:
            slug = f"lambda_{_lambda_slug(coefficient)}_seed_{seed}"
            print("\n" + "=" * 80)
            print(f"Training cost-aware COSTARTS: lambda={coefficient:g}, seed={seed}")
            print("=" * 80)
            summary = train_subset_utility_costarts_router(
                SubsetUtilityTrainingConfig(
                    train_cache_path=args.train_cache,
                    val_cache_path=args.val_cache,
                    output_dir=str(output_root / slug),
                    results_dir=str(results_root / slug),
                    batch_size=args.batch_size,
                    max_epochs=args.max_epochs,
                    patience=args.patience,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    grad_clip_norm=args.grad_clip_norm,
                    seed=seed,
                    cost_coefficient=coefficient,
                    cost_mode=args.cost_mode,
                    cost_file=args.cost_file,
                    selection_metric=args.selection_metric,
                    device=args.device,
                    debug=args.debug,
                )
            )
            best_metrics = dict(summary.get("best_validation_metrics") or {})
            run_rows.append(
                {
                    "lambda": coefficient,
                    "cost_coefficient": coefficient,
                    "seed": seed,
                    "best_epoch": summary.get("best_epoch"),
                    "checkpoint_path": summary.get("best_checkpoint"),
                    "best_checkpoint": summary.get("best_checkpoint"),
                    "results_dir": str(results_root / slug),
                    "validation_mae": summary.get("best_validation_mae"),
                    "validation_mse": best_metrics.get("validation_mse"),
                    "regret_to_oracle": best_metrics.get("validation_regret_to_oracle"),
                    "combined_cost_aware_objective": summary.get("best_validation_objective"),
                    "best_validation_mae": summary.get("best_validation_mae"),
                    "best_validation_objective": summary.get("best_validation_objective"),
                    "average_experts_selected": best_metrics.get("average_experts_selected"),
                    "average_normalized_query_cost": best_metrics.get("average_normalized_query_cost"),
                    "average_raw_query_cost": best_metrics.get("average_raw_query_cost"),
                    "validation_regret_to_oracle": best_metrics.get("validation_regret_to_oracle"),
                    "first_query_oracle_match": best_metrics.get("first_query_oracle_match"),
                    "top_two_oracle_coverage": best_metrics.get("top_two_oracle_coverage"),
                    "stop_step_distribution": json.dumps(best_metrics.get("stop_step_distribution", {})),
                    "false_stop_rate": best_metrics.get("false_stop_rate"),
                    "false_continue_rate": best_metrics.get("false_continue_rate"),
                }
            )

    summary_rows = _aggregate(run_rows)
    runs_path = results_root / "cost_aware_training_runs.csv"
    summary_csv_path = results_root / "cost_aware_training_summary.csv"
    summary_json_path = results_root / "cost_aware_training_summary.json"
    pareto_path = results_root / "cost_accuracy_pareto.png"
    _write_csv(runs_path, run_rows)
    _write_csv(summary_csv_path, summary_rows)
    payload = {
        "runs": run_rows,
        "summary": summary_rows,
        "baseline_source": args.baseline_comparison_csv,
        "test_set_used": False,
        "experts_loaded": False,
        "experts_updated": False,
    }
    summary_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    baselines = _read_baselines(Path(args.baseline_comparison_csv))
    if args.skip_plot:
        print("Skipping Pareto plot because --skip-plot was set")
    else:
        _save_pareto_plot(pareto_path, summary_rows, baselines)
    print(f"Saved: {runs_path}")
    print(f"Saved: {summary_csv_path}")
    print(f"Saved: {summary_json_path}")
    if not args.skip_plot:
        print(f"Saved: {pareto_path}")


if __name__ == "__main__":
    main()
