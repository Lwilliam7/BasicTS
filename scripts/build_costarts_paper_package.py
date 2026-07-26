"""Build paper-ready summaries and diagnostics for COSTARTS experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import pandas as pd
import torch

try:
    from scripts.build_costarts_subset_states import validate_costarts_subset_states
    from scripts.evaluate_costarts_final_comparison import (
        _load_torch,
        _old_costarts_predictions,
        _parameter_count,
        _prediction_from_expert_indices,
        _subset_rollout,
    )
    from scripts.train_costarts_subset_utility_router import (
        SubsetUtilityCOSTARTSRouter,
        _state_batch,
        set_reproducible_seed,
    )
except ImportError:
    from build_costarts_subset_states import validate_costarts_subset_states
    from evaluate_costarts_final_comparison import (
        _load_torch,
        _old_costarts_predictions,
        _parameter_count,
        _prediction_from_expert_indices,
        _subset_rollout,
    )
    from train_costarts_subset_utility_router import (
        SubsetUtilityCOSTARTSRouter,
        _state_batch,
        set_reproducible_seed,
    )


DEFAULT_RESULTS_DIR = "results/router_summary/costarts_subset_utility"
DEFAULT_OUTPUT_DIR = "results/router_summary/costarts_subset_utility/paper_package"
DEFAULT_VAL_CACHE = "cache/costarts_router_val_cache.pt"
DEFAULT_SUBSET_VAL_CACHE = "cache/costarts_subset_states_val.pt"
DEFAULT_OLD_CHECKPOINT = "checkpoints/costarts/best_costarts_router.pt"
DEFAULT_IMPROVED_CHECKPOINT = "checkpoints/costarts_subset_utility/best_subset_utility_costarts_router.pt"


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


def _load_subset_router(checkpoint_path: Path, device: torch.device) -> tuple[SubsetUtilityCOSTARTSRouter, dict[str, Any]]:
    checkpoint = _load_torch(checkpoint_path)
    router = SubsetUtilityCOSTARTSRouter(**checkpoint["router_config"]).to(device)
    router.load_state_dict(checkpoint["router_state_dict"])
    router.eval()
    return router, checkpoint


def _copy_artifact(source: Path, destination: Path) -> bool:
    if not source.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _format_float_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    formatted = df.copy()
    for column in columns:
        if column in formatted.columns:
            formatted[column] = pd.to_numeric(formatted[column], errors="coerce").map(
                lambda value: "" if pd.isna(value) else f"{value:.4f}"
            )
    return formatted


def _write_latex_table(df: pd.DataFrame, path: Path, caption: str, label: str, columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        path.write_text(f"% No data available for {caption}\n", encoding="utf-8")
        return
    table = df.loc[:, [column for column in columns if column in df.columns]].copy()
    table = _format_float_columns(
        table,
        (
            "mae",
            "mse",
            "regret_to_oracle",
            "average_experts_queried",
            "top2_oracle_coverage",
            "first_query_oracle_match",
            "oracle_match_rate",
            "lambda",
            "false_stop_rate",
            "false_continue_rate",
            "stop_precision",
            "stop_recall",
        ),
    )
    latex = table.to_latex(index=False, escape=True, caption=caption, label=label)
    path.write_text(latex, encoding="utf-8")


def _masked_axis_mae(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, reduce_dims: Sequence[int]) -> torch.Tensor:
    mask_float = mask.to(torch.float32)
    numerator = (torch.abs(prediction - target) * mask_float).sum(dim=tuple(reduce_dims))
    denominator = mask_float.sum(dim=tuple(reduce_dims)).clamp_min(1.0)
    return numerator / denominator


def _method_predictions(
    *,
    val_cache: Mapping[str, Any],
    subset_cache: Mapping[str, Any],
    old_checkpoint: Path,
    improved_checkpoint: Path,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    old_router, old_selected, old_stop, old_order, old_epoch, old_latency = _old_costarts_predictions(
        checkpoint_path=old_checkpoint,
        cache=val_cache,
        batch_size=batch_size,
        device=device,
    )
    old_prediction = _prediction_from_expert_indices(val_cache, old_selected)

    improved_router, improved_checkpoint_payload = _load_subset_router(improved_checkpoint, device)
    improved_prediction, improved_selected, improved_sequences, improved_latency = _subset_rollout(
        router=improved_router,
        subset_cache=subset_cache,
        batch_size=batch_size,
        device=device,
    )
    return {
        "old": {
            "router": old_router,
            "prediction": old_prediction,
            "selected": old_selected,
            "query_order": old_order,
            "stop_steps": old_stop,
            "first_query": old_order[:, 0],
            "top2": old_order[:, :2],
            "latency": old_latency,
            "epoch": int(old_epoch),
            "parameter_count": _parameter_count(old_router),
        },
        "improved": {
            "router": improved_router,
            "prediction": improved_prediction,
            "selected": improved_selected,
            "sequences": improved_sequences,
            "first_query": torch.tensor([sequence[0] for sequence in improved_sequences], dtype=torch.long),
            "top2": _sequences_to_tensor(improved_sequences, 2),
            "stop_steps": torch.tensor([len(sequence) for sequence in improved_sequences], dtype=torch.long),
            "latency": improved_latency,
            "epoch": int(improved_checkpoint_payload.get("epoch", -1)),
            "parameter_count": _parameter_count(improved_router),
        },
    }


def _sequences_to_tensor(sequences: Sequence[Sequence[int]], width: int) -> torch.Tensor:
    tensor = torch.full((len(sequences), width), -1, dtype=torch.long)
    for row, sequence in enumerate(sequences):
        for col, expert in enumerate(sequence[:width]):
            tensor[row, col] = int(expert)
    return tensor


def _bar_from_counts(counts: Mapping[str, int], title: str, path: Path, ylabel: str = "Count") -> None:
    names = list(counts.keys())
    values = [counts[name] for name in names]
    plt.figure(figsize=(7, 4))
    plt.bar(names, values, color="#3b82f6")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_cost_pareto(cost_df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(cost_df["average_experts_queried"], cost_df["mae"], marker="o", color="#0f766e")
    for _, row in cost_df.iterrows():
        plt.annotate(f"{row['lambda']:.3g}", (row["average_experts_queried"], row["mae"]), fontsize=7)
    plt.xlabel("Average experts queried")
    plt.ylabel("MAE")
    plt.title("Cost-Accuracy Pareto Curve")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_metric_bars(df: pd.DataFrame, metric: str, path: Path, title: str, limit: int = 12) -> None:
    if df.empty or metric not in df.columns:
        return
    plot_df = df[df["status"].eq("ok")].copy() if "status" in df.columns else df.copy()
    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
    plot_df = plot_df.dropna(subset=[metric]).sort_values(metric).head(limit)
    label_column = "method" if "method" in plot_df.columns else "ablation"
    if label_column not in plot_df.columns:
        label_column = plot_df.columns[0]
    plt.figure(figsize=(8, 5))
    plt.barh(plot_df[label_column].astype(str), plot_df[metric], color="#2563eb")
    plt.xlabel(metric.replace("_", " ").title())
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_top2_coverage(final_df: pd.DataFrame, path: Path) -> None:
    plot_df = final_df.dropna(subset=["top2_oracle_coverage"]).copy()
    if plot_df.empty:
        return
    plot_df["top2_oracle_coverage"] = pd.to_numeric(plot_df["top2_oracle_coverage"], errors="coerce")
    plot_df = plot_df.sort_values("top2_oracle_coverage", ascending=False)
    plt.figure(figsize=(8, 4))
    plt.barh(plot_df["method"], plot_df["top2_oracle_coverage"], color="#7c3aed")
    plt.xlabel("Top-2 oracle coverage")
    plt.title("Top-2 Oracle Coverage")
    plt.xlim(0, 1.05)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_regret_histogram(regrets: Mapping[str, torch.Tensor], path: Path) -> None:
    plt.figure(figsize=(7, 4))
    for label, values in regrets.items():
        plt.hist(values.numpy(), bins=35, alpha=0.5, label=label)
    plt.xlabel("Per-window regret to oracle MAE")
    plt.ylabel("Windows")
    plt.title("Regret Histogram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_line_comparison(rows: list[dict[str, Any]], x_key: str, y_key: str, path: Path, title: str, xlabel: str) -> None:
    df = pd.DataFrame(rows)
    plt.figure(figsize=(7, 4))
    for method, group in df.groupby("method"):
        plt.plot(group[x_key], group[y_key], marker="o", label=method)
    plt.xlabel(xlabel)
    plt.ylabel("MAE")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


@torch.no_grad()
def _within_subset_ranking_diagnostics(
    *,
    router: SubsetUtilityCOSTARTSRouter,
    subset_cache: Mapping[str, Any],
    batch_size: int,
    device: torch.device,
    output_csv: Path,
    plot_path: Path,
    max_points: int,
) -> dict[str, Any]:
    validate_costarts_subset_states(subset_cache)
    nonempty = torch.nonzero(subset_cache["subset_size"] > 0, as_tuple=False).flatten()
    rows: list[dict[str, Any]] = []
    correct_pairs = 0
    total_pairs = 0
    for offset in range(0, nonempty.numel(), batch_size):
        indices = nonempty[offset : offset + batch_size]
        batch = _state_batch(subset_cache, indices.tolist(), device)
        outputs = router(
            batch["history"],
            batch["queried_mask"],
            batch["queried_expert_ids"],
            batch["queried_expert_forecasts"],
        )
        scores = outputs["expert_score"].detach().cpu()
        errors = batch["true_expert_error_vector"].detach().cpu()
        queried_mask = batch["queried_mask"].detach().cpu().to(torch.bool)
        for local_row in range(scores.shape[0]):
            valid = torch.nonzero(queried_mask[local_row], as_tuple=False).flatten()
            for expert in valid.tolist():
                rows.append(
                    {
                        "state_index": int(indices[local_row]),
                        "expert": tuple(subset_cache["expert_names"])[expert],
                        "expert_index": expert,
                        "predicted_score": float(scores[local_row, expert]),
                        "true_negative_mae": float(-errors[local_row, expert]),
                        "true_mae": float(errors[local_row, expert]),
                    }
                )
            for i in range(len(valid)):
                for j in range(i + 1, len(valid)):
                    a = int(valid[i])
                    b = int(valid[j])
                    predicted_better_a = scores[local_row, a] > scores[local_row, b]
                    true_better_a = errors[local_row, a] < errors[local_row, b]
                    correct_pairs += int(bool(predicted_better_a) == bool(true_better_a))
                    total_pairs += 1
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("state_index", "expert", "expert_index", "predicted_score", "true_negative_mae", "true_mae"),
        )
        writer.writeheader()
        writer.writerows(rows)
    df = pd.DataFrame(rows)
    if len(df) > max_points:
        df_plot = df.sample(max_points, random_state=7)
    else:
        df_plot = df
    plt.figure(figsize=(6, 5))
    plt.scatter(df_plot["true_negative_mae"], df_plot["predicted_score"], s=8, alpha=0.35)
    plt.xlabel("True negative MAE (higher is better)")
    plt.ylabel("Predicted expert score")
    plt.title("Within-Subset Ranking Diagnostics")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=180)
    plt.close()
    return {
        "num_points": len(rows),
        "pairwise_accuracy": correct_pairs / max(total_pairs, 1),
        "diagnostics_csv": str(output_csv),
    }


def _write_docs(output_dir: Path) -> None:
    docs = output_dir / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "architecture_pseudocode.md").write_text(
        """# Architecture Pseudocode

## Old COSTARTS

```text
input history x [B,96,7]
encode x into one window embedding
predict:
  query logits [B,M]
  stop logits [B,K]
  expert error map [B,M]
choose a fixed query order once
choose final expert from queried prefix using predicted error
no queried forecast is fed back into the state
```

## Improved Subset-Utility COSTARTS

```text
offline:
  for each window and subset S of queried experts:
    store history, target, queried mask, queried forecasts
    compute true expert errors and marginal utility for each unqueried expert
    label optimal next action as QUERY expert or STOP

training:
  encode history [B,96,7]
  encode queried mask [B,M]
  encode queried forecasts [B,|S|,12,7]
  fuse representations
  predict:
    action logits [B,M+1]
    utility map [B,M]
    queried-subset scores [B,M]
    sparse mix logits [B,M]
  optimize action, utility, pairwise ranking, and optional mix losses

inference:
  start with S = empty
  repeat:
    score QUERY actions and STOP
    query selected expert only
    update S and reveal its forecast
  finalize with equal average of queried expert forecasts
```
""",
        encoding="utf-8",
    )
    (docs / "commands.md").write_text(
        """# Reproduction Commands

## Build subset-state caches

```powershell
python scripts\\build_costarts_subset_states.py --split both --subset-sampling-mode exhaustive --force --print-examples 3
```

## Train improved subset-utility COSTARTS

```powershell
python scripts\\train_costarts_subset_utility_router.py --device cpu --max-epochs 50 --patience 10 --batch-size 1024
```

## Sequential rollout evaluation

```powershell
python scripts\\evaluate_costarts_subset_utility_rollouts.py --device cpu --mode all --finalizer all --temperature 1.0 --detailed-limit 25
```

## Cost sweep

```powershell
python scripts\\evaluate_costarts_cost_sweep.py --device cpu --batch-size 1024
```

## Final comparison

```powershell
python scripts\\evaluate_costarts_final_comparison.py --device cpu --batch-size 1024
```

## Ablations

```powershell
python scripts\\run_costarts_subset_utility_ablations.py --device cpu --max-epochs 50 --patience 10 --batch-size 1024
```

## Paper package

```powershell
python scripts\\build_costarts_paper_package.py --device cpu --batch-size 1024
```
""",
        encoding="utf-8",
    )
    (docs / "reproducibility_notes.md").write_text(
        """# Reproducibility Notes

- Dataset: ETTh1.
- Input shape: `[B,96,7]`.
- Forecast shape: `[B,12,7]`.
- All reported package diagnostics use chronological `router_val` windows unless explicitly marked as an oracle upper bound.
- The final test split is not used in this package.
- Frozen expert predictions are read from cached prediction stacks; no expert checkpoint is updated.
- Old COSTARTS and improved subset-utility COSTARTS are labeled separately in tables and plots.
- Current ablation numbers may be smoke-test numbers if the ablation runner was executed with `--max-epochs 1`.
""",
        encoding="utf-8",
    )


def build_paper_package(args: argparse.Namespace) -> dict[str, Any]:
    set_reproducible_seed(args.seed)
    device = torch.device(args.device)
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    tables_dir = output_dir / "tables"
    plots_dir = output_dir / "plots"
    diagnostics_dir = output_dir / "diagnostics"
    for directory in (tables_dir, plots_dir, diagnostics_dir):
        directory.mkdir(parents=True, exist_ok=True)

    copied = {}
    for name in (
        "final_comparison.csv",
        "final_comparison.json",
        "ablations.csv",
        "ablations.json",
        "cost_sweep.csv",
        "pareto_curve.json",
        "reranking_comparison.csv",
        "reranking_examples.csv",
        "mixing_results.csv",
        "mix_weight_statistics.json",
    ):
        copied[name] = _copy_artifact(results_dir / name, tables_dir / name)

    final_df = _read_csv(results_dir / "final_comparison.csv")
    ablation_df = _read_csv(results_dir / "ablations.csv")
    cost_df = _read_csv(results_dir / "cost_sweep.csv")
    rerank_df = _read_csv(results_dir / "reranking_comparison.csv")

    _write_latex_table(
        final_df,
        tables_dir / "final_comparison.tex",
        "Final validation comparison on identical chronological windows.",
        "tab:costarts-final-comparison",
        ("method", "mae", "mse", "regret_to_oracle", "average_experts_queried", "top2_oracle_coverage", "first_query_oracle_match"),
    )
    _write_latex_table(
        ablation_df,
        tables_dir / "ablations.tex",
        "Subset-utility COSTARTS ablations.",
        "tab:costarts-ablations",
        ("ablation", "changed_factor", "status", "mae", "mse", "average_experts_queried", "description"),
    )
    _write_latex_table(
        cost_df,
        tables_dir / "cost_sweep.tex",
        "Cost-aware stopping sweep.",
        "tab:costarts-cost-sweep",
        ("lambda", "mae", "mse", "average_experts_queried", "false_stop_rate", "false_continue_rate", "stop_precision", "stop_recall"),
    )
    _write_latex_table(
        rerank_df,
        tables_dir / "reranking_comparison.tex",
        "Queried-subset reranking diagnostics.",
        "tab:costarts-reranking",
        ("selector", "mae", "mse", "oracle_match_within_subset", "better_of_top_two_accuracy", "pairwise_queried_subset_ranking_accuracy"),
    )

    val_cache = _load_torch(Path(args.val_cache))
    subset_cache = _load_torch(Path(args.subset_val_cache))
    validate_costarts_subset_states(subset_cache)
    methods = _method_predictions(
        val_cache=val_cache,
        subset_cache=subset_cache,
        old_checkpoint=Path(args.old_checkpoint),
        improved_checkpoint=Path(args.improved_checkpoint),
        batch_size=args.batch_size,
        device=device,
    )
    expert_names = tuple(val_cache["expert_names"])
    oracle_error = val_cache["error_matrix"].min(dim=1).values

    if not cost_df.empty:
        _plot_cost_pareto(cost_df, plots_dir / "cost_accuracy_pareto.png")
        _plot_metric_bars(cost_df.rename(columns={"lambda": "method"}), "average_experts_queried", plots_dir / "average_experts_queried_cost_sweep.png", "Average Experts Queried by Cost")
        stop_counts = json.loads(cost_df.iloc[0]["stop_step_distribution"])
        _bar_from_counts(stop_counts, "Improved Method Stop-Step Distribution (lambda=0)", plots_dir / "stop_step_distribution.png")

    _plot_metric_bars(final_df, "mae", plots_dir / "final_comparison_mae.png", "Final Comparison MAE")
    _plot_metric_bars(final_df, "average_experts_queried", plots_dir / "average_experts_queried_methods.png", "Average Experts Queried by Method")
    _plot_top2_coverage(final_df, plots_dir / "top2_oracle_coverage.png")

    first_query_rows = []
    for method_name, payload in (("Old COSTARTS", methods["old"]), ("Improved subset utility", methods["improved"])):
        counts = torch.bincount(payload["first_query"], minlength=len(expert_names)).tolist()
        for index, count in enumerate(counts):
            first_query_rows.append({"method": method_name, "expert": expert_names[index], "count": int(count)})
    first_query_df = pd.DataFrame(first_query_rows)
    first_query_df.to_csv(diagnostics_dir / "first_query_distribution.csv", index=False)
    plt.figure(figsize=(8, 4))
    width = 0.38
    x = range(len(expert_names))
    for offset, method_name in enumerate(("Old COSTARTS", "Improved subset utility")):
        values = [
            int(first_query_df[(first_query_df["method"] == method_name) & (first_query_df["expert"] == expert)]["count"].iloc[0])
            for expert in expert_names
        ]
        plt.bar([item + (offset - 0.5) * width for item in x], values, width=width, label=method_name)
    plt.xticks(list(x), expert_names, rotation=25, ha="right")
    plt.ylabel("Windows")
    plt.title("First-Query Expert Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "first_query_expert_distribution.png", dpi=180)
    plt.close()

    old_selected_error = val_cache["error_matrix"].gather(1, methods["old"]["selected"].view(-1, 1)).squeeze(1)
    improved_selected_error = val_cache["error_matrix"].gather(1, methods["improved"]["selected"].view(-1, 1)).squeeze(1)
    _plot_regret_histogram(
        {
            "Old COSTARTS": old_selected_error - oracle_error,
            "Improved subset utility": improved_selected_error - oracle_error,
        },
        plots_dir / "regret_histogram.png",
    )

    horizon_rows = []
    variable_rows = []
    for method_name, prediction in (
        ("Old COSTARTS", methods["old"]["prediction"]),
        ("Improved subset utility", methods["improved"]["prediction"]),
    ):
        horizon_mae = _masked_axis_mae(prediction, val_cache["targets"], val_cache["target_masks"], reduce_dims=(0, 2))
        variable_mae = _masked_axis_mae(prediction, val_cache["targets"], val_cache["target_masks"], reduce_dims=(0, 1))
        for horizon, value in enumerate(horizon_mae.tolist(), start=1):
            horizon_rows.append({"method": method_name, "horizon": horizon, "mae": value})
        for variable, value in enumerate(variable_mae.tolist()):
            variable_rows.append({"method": method_name, "variable": variable, "mae": value})
    pd.DataFrame(horizon_rows).to_csv(diagnostics_dir / "per_horizon_mae.csv", index=False)
    pd.DataFrame(variable_rows).to_csv(diagnostics_dir / "per_variable_mae.csv", index=False)
    _plot_line_comparison(horizon_rows, "horizon", "mae", plots_dir / "per_horizon_mae.png", "Per-Horizon MAE", "Forecast horizon")
    _plot_line_comparison(variable_rows, "variable", "mae", plots_dir / "per_variable_mae.png", "Per-Variable MAE", "Variable index")

    ranking_summary = _within_subset_ranking_diagnostics(
        router=methods["improved"]["router"],
        subset_cache=subset_cache,
        batch_size=args.batch_size,
        device=device,
        output_csv=diagnostics_dir / "predicted_vs_true_within_subset_scores.csv",
        plot_path=plots_dir / "predicted_vs_true_within_subset_ranking.png",
        max_points=args.max_scatter_points,
    )

    _write_docs(output_dir)
    summary = {
        "output_dir": str(output_dir),
        "copied_artifacts": copied,
        "plots": sorted(str(path.relative_to(output_dir)) for path in plots_dir.glob("*.png")),
        "tables": sorted(str(path.relative_to(output_dir)) for path in tables_dir.glob("*")),
        "diagnostics": sorted(str(path.relative_to(output_dir)) for path in diagnostics_dir.glob("*")),
        "old_costarts": {
            "checkpoint": args.old_checkpoint,
            "epoch": methods["old"]["epoch"],
            "parameter_count": methods["old"]["parameter_count"],
        },
        "improved_subset_utility": {
            "checkpoint": args.improved_checkpoint,
            "epoch": methods["improved"]["epoch"],
            "parameter_count": methods["improved"]["parameter_count"],
        },
        "within_subset_ranking": ranking_summary,
        "test_data_used": False,
    }
    (output_dir / "paper_package_manifest.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    print(f"Saved paper package: {output_dir}")
    print(f"Saved manifest: {output_dir / 'paper_package_manifest.json'}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build COSTARTS paper package.")
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--val-cache", default=DEFAULT_VAL_CACHE)
    parser.add_argument("--subset-val-cache", default=DEFAULT_SUBSET_VAL_CACHE)
    parser.add_argument("--old-checkpoint", default=DEFAULT_OLD_CHECKPOINT)
    parser.add_argument("--improved-checkpoint", default=DEFAULT_IMPROVED_CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-scatter-points", type=int, default=8000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    build_paper_package(parse_args())


if __name__ == "__main__":
    main()
