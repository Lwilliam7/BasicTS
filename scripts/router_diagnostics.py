"""Diagnostics and plots for frozen-expert routing experiments."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_OUTPUT_DIR = "results/router_diagnostics"
DEFAULT_SUMMARY_PATHS = (
    "results/router_summary/best_5_experts/router_test_metrics.json",
    "results/router2_summary/routerdc_hard_test_metrics.json",
    "results/router_summary/costarts/costarts_training_summary.json",
)


def _safe_name(value: str) -> str:
    return (
        str(value)
        .strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .lower()
    )


def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _load_torch(path: Optional[Path]) -> Optional[dict]:
    if path is None or not path.exists():
        return None
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _save_fig(fig, output_dir: Path, filename: str, generated: list[dict]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    generated.append({"filename": filename, "path": str(path)})
    return path


def _skip(name: str, reason: str, skipped: list[dict]) -> None:
    skipped.append({"diagnostic": name, "reason": reason})


def _heatmap(
    matrix: np.ndarray,
    labels: Sequence[str],
    title: str,
    colorbar_label: str,
):
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.1), max(5, len(labels) * 0.9)))
    image = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            if np.isfinite(value):
                ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=8, color="white")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)
    return fig


def _bar(values: Mapping[str, float], title: str, ylabel: str):
    labels = list(values)
    numbers = [float(values[label]) for label in labels]
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.2), 4.8))
    bars = ax.bar(labels, numbers, color="#2f6f73")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, numbers):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    return fig


def _hist(values: np.ndarray, title: str, xlabel: str, bins: int = 40):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.hist(values[np.isfinite(values)], bins=bins, color="#6d5c9f", edgecolor="white", alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.25)
    return fig


def _expert_names_from_cache(cache: Mapping[str, Any]) -> list[str]:
    return list(cache.get("expert_names", [f"expert_{i}" for i in range(cache["error_matrix"].shape[1])]))


def _cache_arrays(cache: Mapping[str, Any]) -> dict[str, np.ndarray]:
    import torch

    arrays = {}
    for key, value in cache.items():
        if torch.is_tensor(value):
            arrays[key] = value.detach().cpu().numpy()
    return arrays


def _oracle_distribution(cache: Mapping[str, Any]) -> dict[str, float]:
    names = _expert_names_from_cache(cache)
    best_value = cache["best_expert"]
    best = (
        best_value.detach().cpu().numpy()
        if hasattr(best_value, "detach")
        else np.asarray(best_value)
    )
    counts = np.bincount(best, minlength=len(names)).astype(float)
    return {name: count / max(len(best), 1) * 100.0 for name, count in zip(names, counts)}


def _confusion_matrix(predicted: np.ndarray, oracle: np.ndarray, num_experts: int) -> np.ndarray:
    matrix = np.zeros((num_experts, num_experts), dtype=float)
    for pred, truth in zip(predicted, oracle):
        matrix[int(truth), int(pred)] += 1.0
    row_sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)


def _extract_routerdc_variants(summary: Optional[dict]) -> list[tuple[str, dict]]:
    if not summary:
        return []
    variants = []
    routerdc = summary.get("routerdc_hard", {})
    for name, payload in routerdc.items():
        if isinstance(payload, dict):
            variants.append((f"routerdc_hard_{name}", payload))
    return variants


def _extract_soft_router_payload(summary: Optional[dict]) -> Optional[dict]:
    if not summary:
        return None
    return summary.get("learned_router")


def _plot_step_variable_weights(
    payload: Mapping[str, Any],
    output_dir: Path,
    generated: list[dict],
    skipped: list[dict],
) -> None:
    by_step = payload.get("average_expert_weights_by_step")
    if by_step:
        df = pd.DataFrame(by_step)
        fig, ax = plt.subplots(figsize=(10, 4.8))
        df.plot(ax=ax)
        ax.set_title("Expert Selection Weight By Forecast Horizon")
        ax.set_xlabel("Forecast horizon index")
        ax.set_ylabel("Average router weight")
        ax.grid(alpha=0.25)
        ax.legend(title="Expert", bbox_to_anchor=(1.02, 1), loc="upper left")
        _save_fig(fig, output_dir, "expert_selection_by_horizon.png", generated)
    else:
        _skip("expert selection by horizon", "No average_expert_weights_by_step field found", skipped)

    by_variable = payload.get("average_expert_weights_by_variable")
    if by_variable:
        df = pd.DataFrame(by_variable)
        fig, ax = plt.subplots(figsize=(10, 4.8))
        df.plot(kind="bar", ax=ax)
        ax.set_title("Expert Selection Weight By Variable")
        ax.set_xlabel("Variable index")
        ax.set_ylabel("Average router weight")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(title="Expert", bbox_to_anchor=(1.02, 1), loc="upper left")
        _save_fig(fig, output_dir, "expert_selection_by_variable.png", generated)
    else:
        _skip("expert selection by variable", "No average_expert_weights_by_variable field found", skipped)

    by_step_variable = payload.get("average_expert_weights_by_step_and_variable")
    if by_step_variable:
        rows = []
        for expert_name, matrix in by_step_variable.items():
            values = np.asarray(matrix, dtype=float)
            rows.append((expert_name, values.mean(axis=1), values.mean(axis=0)))
        if rows:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
            for expert_name, step_avg, variable_avg in rows:
                axes[0].plot(step_avg, label=expert_name)
                axes[1].plot(variable_avg, marker="o", label=expert_name)
            axes[0].set_title("Loss-Map Summary By Horizon")
            axes[0].set_xlabel("Forecast horizon index")
            axes[0].set_ylabel("Average weight")
            axes[1].set_title("Loss-Map Summary By Variable")
            axes[1].set_xlabel("Variable index")
            axes[1].set_ylabel("Average weight")
            for ax in axes:
                ax.grid(alpha=0.25)
            axes[1].legend(title="Expert", bbox_to_anchor=(1.02, 1), loc="upper left")
            _save_fig(fig, output_dir, "predicted_loss_map_summary_by_horizon_and_variable.png", generated)
    else:
        _skip("predicted vs true expert loss-map summaries", "No per-step-and-variable router weights found", skipped)


def _plot_json_summary_diagnostics(
    summaries: Mapping[str, Optional[dict]],
    output_dir: Path,
    generated: list[dict],
    skipped: list[dict],
) -> None:
    soft_payload = _extract_soft_router_payload(summaries.get("router"))
    if soft_payload:
        weights = soft_payload.get("average_expert_weights")
        if weights:
            _save_fig(
                _bar(weights, "Expert Utilization From Soft Router Weights", "Average weight"),
                output_dir,
                "expert_utilization_histogram_soft_router_weights.png",
                generated,
            )
        _plot_step_variable_weights(soft_payload, output_dir, generated, skipped)
    else:
        _skip("soft-router horizon/variable diagnostics", "No learned_router payload found", skipped)

    routerdc_variants = _extract_routerdc_variants(summaries.get("routerdc"))
    for variant_name, payload in routerdc_variants:
        selection = payload.get("selection_percentage")
        if selection:
            _save_fig(
                _bar(selection, f"Expert Utilization: {variant_name}", "Selected windows (%)"),
                output_dir,
                f"expert_utilization_histogram_{_safe_name(variant_name)}.png",
                generated,
            )
        probabilities = payload.get("average_router_probabilities")
        if probabilities:
            _save_fig(
                _bar(probabilities, f"Average Router Probabilities: {variant_name}", "Probability"),
                output_dir,
                f"average_router_probabilities_{_safe_name(variant_name)}.png",
                generated,
            )
        matrix = payload.get("expert_embedding_cosine_similarity_matrix")
        names = summaries.get("routerdc", {}).get("selected_expert_names", [])
        if matrix and names:
            _save_fig(
                _heatmap(np.asarray(matrix, dtype=float), names, f"Learned Expert Similarity: {variant_name}", "Cosine similarity"),
                output_dir,
                f"learned_expert_representation_similarity_{_safe_name(variant_name)}.png",
                generated,
            )
        if "routing_entropy" in payload:
            _save_fig(
                _bar({variant_name: float(payload["routing_entropy"])}, f"Routing Entropy: {variant_name}", "Entropy"),
                output_dir,
                f"routing_entropy_scalar_{_safe_name(variant_name)}.png",
                generated,
            )


def _load_costarts_router(checkpoint: Mapping[str, Any], num_experts: int):
    from scripts.train_costarts_router import COSTARTSRouter

    router_config = dict(checkpoint.get("router_config", {}))
    router_config.setdefault("num_experts", num_experts)
    router = COSTARTSRouter(**router_config)
    state = checkpoint.get("router_state_dict")
    if state is None:
        raise KeyError("COSTARTS checkpoint has no router_state_dict")
    router.load_state_dict(state)
    router.eval()
    return router


def _costarts_predictions(cache: Mapping[str, Any], checkpoint: Mapping[str, Any], batch_size: int = 512) -> Optional[dict]:
    if not checkpoint:
        return None
    import torch

    names = _expert_names_from_cache(cache)
    router = _load_costarts_router(checkpoint, len(names))
    histories = cache["histories"]
    outputs = {
        "predicted_error": [],
        "query_logits": [],
        "stop_logits": [],
        "query_order": [],
        "selected_expert": [],
        "stop_step": [],
    }
    with torch.no_grad():
        for start in range(0, histories.shape[0], batch_size):
            history = histories[start : start + batch_size]
            result = router(history)
            selected, stop_step = _select_costarts_expert(result)
            for key in ("map_prediction", "query_logits", "stop_logits", "query_order"):
                target = "predicted_error" if key == "map_prediction" else key
                outputs[target].append(result[key].detach().cpu())
            outputs["selected_expert"].append(selected.detach().cpu())
            outputs["stop_step"].append(stop_step.detach().cpu())
    return {key: torch.cat(value, dim=0).numpy() for key, value in outputs.items()}


def _select_costarts_expert(result: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    import torch

    query_order = result["query_order"]
    predicted_errors = result["map_prediction"]
    stop_step = result["stop_step"].clamp(1, query_order.shape[1])
    selected = []
    for row_index in range(query_order.shape[0]):
        candidates = query_order[row_index, : int(stop_step[row_index].item())]
        candidate_errors = predicted_errors[row_index].gather(0, candidates)
        selected.append(candidates[torch.argmin(candidate_errors)])
    return torch.stack(selected), stop_step


def _plot_cache_diagnostics(
    cache: Optional[dict],
    checkpoint: Optional[dict],
    output_dir: Path,
    generated: list[dict],
    skipped: list[dict],
    batch_size: int,
) -> None:
    if not cache:
        for name in (
            "expert error correlation matrix",
            "oracle expert distribution",
            "confusion matrix",
            "routing entropy distribution",
            "stop-step distribution",
            "regret histogram",
            "predicted vs true scalar expert error scatterplots",
            "calibration plots",
            "regime/prototype visualizations",
        ):
            _skip(name, "No compatible offline expert cache was provided/found", skipped)
        return

    names = _expert_names_from_cache(cache)
    arrays = _cache_arrays(cache)
    error_matrix = arrays["error_matrix"]
    oracle = arrays.get("best_expert", error_matrix.argmin(axis=1)).astype(int)

    corr = np.corrcoef(error_matrix, rowvar=False)
    _save_fig(
        _heatmap(corr, names, "Expert Error Correlation Matrix", "Pearson correlation"),
        output_dir,
        "expert_error_correlation_matrix.png",
        generated,
    )

    _save_fig(
        _bar(_oracle_distribution(cache), "Oracle Best Expert Distribution", "Windows (%)"),
        output_dir,
        "oracle_expert_distribution.png",
        generated,
    )

    oracle_mae = error_matrix.min(axis=1)
    costarts = _costarts_predictions(cache, checkpoint, batch_size=batch_size) if checkpoint else None
    if costarts:
        predicted = costarts["selected_expert"].astype(int)
        cm = _confusion_matrix(predicted, oracle, len(names))
        _save_fig(
            _heatmap(cm, names, "Predicted Best Expert vs Oracle Best Expert", "Row-normalized fraction"),
            output_dir,
            "confusion_matrix_predicted_best_vs_oracle_best.png",
            generated,
        )

        selected_mae = error_matrix[np.arange(error_matrix.shape[0]), predicted]
        regret = selected_mae - oracle_mae
        _save_fig(
            _hist(regret, "Selected Expert Regret To Oracle", "MAE regret", bins=45),
            output_dir,
            "regret_histogram_selected_expert_vs_oracle.png",
            generated,
        )

        import torch
        import torch.nn.functional as F

        query_prob = F.softmax(torch.tensor(costarts["query_logits"]), dim=-1).numpy()
        entropy = -(query_prob * np.log(np.clip(query_prob, 1e-12, 1.0))).sum(axis=1)
        _save_fig(
            _hist(entropy, "Routing Entropy Distribution", "Entropy", bins=40),
            output_dir,
            "routing_entropy_distribution.png",
            generated,
        )

        stop_step = costarts["stop_step"].astype(int)
        counts = {f"step_{step}": float((stop_step == step).sum()) for step in sorted(np.unique(stop_step))}
        _save_fig(
            _bar(counts, "Stop-Step Distribution", "Windows"),
            output_dir,
            "stop_step_distribution.png",
            generated,
        )

        predicted_error = costarts["predicted_error"]
        cols = min(3, len(names))
        rows = math.ceil(len(names) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.2 * rows), squeeze=False)
        for index, name in enumerate(names):
            ax = axes[index // cols][index % cols]
            ax.scatter(error_matrix[:, index], predicted_error[:, index], s=10, alpha=0.35)
            low = min(error_matrix[:, index].min(), predicted_error[:, index].min())
            high = max(error_matrix[:, index].max(), predicted_error[:, index].max())
            ax.plot([low, high], [low, high], color="black", linewidth=1)
            ax.set_title(name)
            ax.set_xlabel("True cached MAE")
            ax.set_ylabel("Predicted scalar error")
            ax.grid(alpha=0.2)
        for index in range(len(names), rows * cols):
            axes[index // cols][index % cols].axis("off")
        _save_fig(fig, output_dir, "predicted_vs_true_scalar_expert_error_scatterplots.png", generated)

        stop_prob = F.softmax(torch.tensor(costarts["stop_logits"]), dim=-1).numpy()
        stop_conf = stop_prob.max(axis=1)
        stopped_early = (stop_step == 1).astype(float)
        _save_fig(
            _calibration_plot(stop_conf, stopped_early, "Stop Probability Calibration"),
            output_dir,
            "stop_probability_calibration.png",
            generated,
        )
    else:
        _skip("confusion matrix between predicted best expert and oracle best expert", "No compatible COSTARTS checkpoint was provided", skipped)
        _skip("routing entropy distribution", "No per-window router probabilities available", skipped)
        _skip("stop-step distribution", "No per-window stop decisions available", skipped)
        _skip("regret histogram", "No per-window predicted selections available", skipped)
        _skip("predicted vs true scalar expert error scatterplots", "No predicted scalar expert errors available", skipped)
        _skip("calibration plots for stop probabilities", "No stop probabilities available", skipped)

    if "histories" in arrays:
        histories = arrays["histories"].reshape(arrays["histories"].shape[0], -1)
        prototype_values = histories.mean(axis=1)
        _save_fig(
            _hist(prototype_values, "Optional Regime/Prototype Summary: Mean History Value", "Mean scaled history value", bins=50),
            output_dir,
            "optional_regime_prototype_history_mean_distribution.png",
            generated,
        )
    else:
        _skip("optional regime/prototype visualizations", "No histories found in cache", skipped)

    _plot_loss_map_summaries_from_cache(arrays, names, output_dir, generated, skipped)


def _plot_loss_map_summaries_from_cache(
    arrays: Mapping[str, np.ndarray],
    names: Sequence[str],
    output_dir: Path,
    generated: list[dict],
    skipped: list[dict],
) -> None:
    if "prediction_stack" not in arrays or "targets" not in arrays or "target_masks" not in arrays:
        _skip("true expert loss-map summaries", "Cache has no prediction_stack/targets/target_masks", skipped)
        return
    stack = arrays["prediction_stack"]
    targets = arrays["targets"][..., None]
    mask = arrays["target_masks"][..., None].astype(float)
    loss_map = np.abs(stack - targets) * mask
    denom_horizon = mask.sum(axis=(0, 2)).clip(min=1.0)
    denom_variable = mask.sum(axis=(0, 1)).clip(min=1.0)
    by_horizon = loss_map.sum(axis=(0, 2)) / denom_horizon
    by_variable = loss_map.sum(axis=(0, 1)) / denom_variable

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for index, name in enumerate(names):
        axes[0].plot(by_horizon[:, index], label=name)
        axes[1].plot(by_variable[:, index], marker="o", label=name)
    axes[0].set_title("True Expert Loss-Map Summary By Horizon")
    axes[0].set_xlabel("Forecast horizon index")
    axes[0].set_ylabel("MAE")
    axes[1].set_title("True Expert Loss-Map Summary By Variable")
    axes[1].set_xlabel("Variable index")
    axes[1].set_ylabel("MAE")
    for ax in axes:
        ax.grid(alpha=0.25)
    axes[1].legend(title="Expert", bbox_to_anchor=(1.02, 1), loc="upper left")
    _save_fig(fig, output_dir, "true_expert_loss_map_summary_by_horizon_and_variable.png", generated)


def _calibration_plot(confidence: np.ndarray, observed: np.ndarray, title: str):
    bins = np.linspace(0.0, 1.0, 11)
    centers = (bins[:-1] + bins[1:]) / 2
    observed_rate = []
    counts = []
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (confidence >= left) & (confidence < right if right < 1 else confidence <= right)
        counts.append(mask.sum())
        observed_rate.append(observed[mask].mean() if mask.any() else np.nan)
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    ax.plot([0, 1], [0, 1], color="black", linewidth=1, label="Perfect calibration")
    ax.plot(centers, observed_rate, marker="o", label="Observed")
    ax.set_title(title)
    ax.set_xlabel("Predicted stop confidence")
    ax.set_ylabel("Observed stop rate")
    ax.grid(alpha=0.25)
    ax.legend()
    twin = ax.twinx()
    twin.bar(centers, counts, width=0.08, color="#999999", alpha=0.25)
    twin.set_ylabel("Bin count")
    return fig


def _plot_latency_pareto(
    summaries: Mapping[str, Optional[dict]],
    output_dir: Path,
    generated: list[dict],
    skipped: list[dict],
) -> None:
    rows = []
    routerdc = summaries.get("routerdc")
    if routerdc:
        for variant_name, payload in _extract_routerdc_variants(routerdc):
            if "mae" in payload:
                selection = payload.get("selection_percentage", {})
                avg_cost = sum(float(value) for value in selection.values()) / 100.0
                rows.append({"method": variant_name, "mae": float(payload["mae"]), "relative_latency": max(avg_cost, 1.0)})
        for name, mae in routerdc.get("individual_expert_mae", {}).items():
            rows.append({"method": name, "mae": float(mae), "relative_latency": 1.0})
        if "oracle_mae" in routerdc:
            rows.append({"method": "oracle_all_experts", "mae": float(routerdc["oracle_mae"]), "relative_latency": len(routerdc.get("selected_expert_names", []))})
    router = summaries.get("router")
    if router:
        for row in router.get("comparison", []):
            method = row.get("Method")
            mae = row.get("Test MAE")
            if method and mae is not None:
                relative_latency = 1.0 if method in router.get("selected_expert_names", []) else len(router.get("selected_expert_names", []))
                rows.append({"method": method, "mae": float(mae), "relative_latency": relative_latency})
    if not rows:
        _skip("latency vs MAE Pareto plots", "No summary metrics with MAE/latency proxy found", skipped)
        return
    df = pd.DataFrame(rows).drop_duplicates(["method", "mae", "relative_latency"])
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    ax.scatter(df["relative_latency"], df["mae"], s=70, color="#2f6f73")
    for _, row in df.iterrows():
        ax.text(row["relative_latency"], row["mae"], f" {row['method']}", fontsize=8, va="center")
    ax.set_title("Latency vs MAE Pareto View")
    ax.set_xlabel("Relative expert inference cost")
    ax.set_ylabel("MAE, lower is better")
    ax.grid(alpha=0.25)
    _save_fig(fig, output_dir, "latency_vs_mae_pareto.png", generated)
    df.to_csv(output_dir / "latency_vs_mae_pareto_data.csv", index=False)
    generated.append({"filename": "latency_vs_mae_pareto_data.csv", "path": str(output_dir / "latency_vs_mae_pareto_data.csv")})


def _multi_seed_summary(
    summary_globs: Sequence[str],
    output_dir: Path,
    generated: list[dict],
    skipped: list[dict],
) -> None:
    paths = []
    for pattern in summary_globs:
        paths.extend(Path.cwd().glob(pattern))
    rows = []
    for path in sorted(set(paths)):
        payload = _load_json(path)
        if not payload:
            continue
        config = payload.get("training_config", {})
        seed = config.get("seed", payload.get("seed"))
        best = payload.get("best_validation_mae", payload.get("validation_mae"))
        if seed is not None and best is not None:
            rows.append({"path": str(path), "seed": int(seed), "best_validation_mae": float(best)})
    if not rows:
        _skip("multi-seed aggregation", "No matching training summaries with seed and validation MAE found", skipped)
        return
    df = pd.DataFrame(rows)
    csv_path = output_dir / "multi_seed_validation_summary.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    generated.append({"filename": csv_path.name, "path": str(csv_path)})
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.scatter(df["seed"], df["best_validation_mae"], s=70)
    ax.set_title("Multi-Seed Validation MAE")
    ax.set_xlabel("Seed")
    ax.set_ylabel("Best validation MAE")
    ax.grid(alpha=0.25)
    _save_fig(fig, output_dir, "multi_seed_validation_mae.png", generated)


def run_diagnostics(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir)
    generated: list[dict] = []
    skipped: list[dict] = []

    summaries = {
        "router": _load_json(Path(args.router_summary)),
        "routerdc": _load_json(Path(args.routerdc_summary)),
    }
    _plot_json_summary_diagnostics(summaries, output_dir, generated, skipped)
    _plot_latency_pareto(summaries, output_dir, generated, skipped)
    _multi_seed_summary(args.summary_glob, output_dir, generated, skipped)

    cache = _load_torch(Path(args.cache_path)) if args.cache_path else None
    checkpoint = _load_torch(Path(args.checkpoint_path)) if args.checkpoint_path else None
    _plot_cache_diagnostics(cache, checkpoint, output_dir, generated, skipped, args.batch_size)

    manifest = {
        "output_dir": str(output_dir),
        "generated": generated,
        "skipped": skipped,
        "inputs": {
            "router_summary": args.router_summary,
            "routerdc_summary": args.routerdc_summary,
            "cache_path": args.cache_path,
            "checkpoint_path": args.checkpoint_path,
            "summary_glob": args.summary_glob,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "router_diagnostics_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    generated.append({"filename": manifest_path.name, "path": str(manifest_path)})
    print(f"Saved diagnostics manifest: {manifest_path}")
    print(f"Generated {len(generated)} files; skipped {len(skipped)} unavailable diagnostics.")
    for item in generated:
        print(f"  generated: {item['path']}")
    for item in skipped:
        print(f"  skipped: {item['diagnostic']} - {item['reason']}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate frozen-expert router diagnostics and visualizations.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--router-summary", default=DEFAULT_SUMMARY_PATHS[0])
    parser.add_argument("--routerdc-summary", default=DEFAULT_SUMMARY_PATHS[1])
    parser.add_argument("--cache-path", default="", help="Optional torch cache with error_matrix/prediction_stack/histories.")
    parser.add_argument("--checkpoint-path", default="", help="Optional COSTARTS checkpoint for per-window router predictions.")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--summary-glob",
        action="append",
        default=["results/router_summary/costarts*/costarts_training_summary.json"],
        help="Glob for multi-seed training summaries. Can be passed more than once.",
    )
    return parser.parse_args()


def main() -> None:
    run_diagnostics(parse_args())


if __name__ == "__main__":
    main()
