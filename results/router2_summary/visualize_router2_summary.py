"""Visualize Router2 and RouterDC summary results.

Run from the project root:

    python results/router2_summary/visualize_router2_summary.py

The script prints compact result tables and saves PNG charts in this folder.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def find_summary_dir() -> Path:
    script_dir = Path(__file__).resolve().parent
    if (script_dir / "router2_summary.csv").exists():
        return script_dir

    cwd_candidate = Path.cwd() / "results" / "router2_summary"
    if (cwd_candidate / "router2_summary.csv").exists():
        return cwd_candidate

    raise FileNotFoundError(
        "Could not find router2_summary.csv. Run this from the repo root or "
        "from results/router2_summary/."
    )


def load_results(summary_dir: Path) -> pd.DataFrame:
    summary_path = summary_dir / "router2_summary.csv"
    results = pd.read_csv(summary_path)
    for column in ("test_mae", "test_mse", "test_rmse"):
        results[column] = pd.to_numeric(results[column], errors="coerce")
    return results


def print_tables(results: pd.DataFrame) -> None:
    print("\nAll Router2 Summary Results")
    print(
        results[
            [
                "router_variant",
                "method",
                "result_type",
                "test_mae",
                "test_mse",
                "test_rmse",
            ]
        ]
        .sort_values(["router_variant", "test_mae"])
        .round(6)
        .to_string(index=False)
    )

    best_by_variant = (
        results.sort_values(["router_variant", "test_mae"])
        .groupby("router_variant", as_index=False)
        .first()
    )
    print("\nBest Row Per Router Variant")
    print(
        best_by_variant[
            ["router_variant", "method", "result_type", "test_mae"]
        ]
        .round(6)
        .to_string(index=False)
    )

    trained = results[results["result_type"].eq("trained_router")].copy()
    print("\nTrained Routers Only")
    print(
        trained[["router_variant", "method", "test_mae", "test_mse", "test_rmse"]]
        .sort_values("test_mae")
        .round(6)
        .to_string(index=False)
    )


def save_best_by_variant_chart(results: pd.DataFrame, summary_dir: Path) -> Path:
    best_by_variant = (
        results.sort_values(["router_variant", "test_mae"])
        .groupby("router_variant", as_index=False)
        .first()
        .sort_values("test_mae")
    )

    fig, ax = plt.subplots(figsize=(10, 4.8))
    bars = ax.bar(
        best_by_variant["router_variant"],
        best_by_variant["test_mae"],
        color="#2f6f73",
    )
    ax.set_title("Best Test MAE Per Router2 Variant")
    ax.set_xlabel("Router variant")
    ax.set_ylabel("Test MAE, lower is better")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=20)
    for bar, value in zip(bars, best_by_variant["test_mae"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    output_path = summary_dir / "router2_best_mae_by_variant.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_trained_router_chart(results: pd.DataFrame, summary_dir: Path) -> Path:
    trained = (
        results[results["result_type"].eq("trained_router")]
        .copy()
        .sort_values("test_mae")
    )

    fig, ax = plt.subplots(figsize=(10, 4.8))
    bars = ax.barh(
        trained["method"],
        trained["test_mae"],
        color="#7a4f9f",
    )
    ax.set_title("Trained Router2 / RouterDC Test MAE")
    ax.set_xlabel("Test MAE, lower is better")
    ax.grid(axis="x", alpha=0.25)
    ax.invert_yaxis()
    for bar, value in zip(bars, trained["test_mae"]):
        ax.text(
            value,
            bar.get_y() + bar.get_height() / 2,
            f" {value:.4f}",
            va="center",
            fontsize=9,
        )
    fig.tight_layout()
    output_path = summary_dir / "router2_trained_routers_mae.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_soft_router_baseline_chart(results: pd.DataFrame, summary_dir: Path) -> Path:
    soft = results[results["router_family"].eq("soft_router2")].copy()
    methods = [
        "Fixed validation-based soft weights",
        "Fixed equal average",
        "Router 2 feature router",
        "Multiscale TCN expert-embedding router",
    ]
    soft = soft[soft["method"].isin(methods)]
    pivot = soft.pivot_table(
        index="router_variant",
        columns="method",
        values="test_mae",
        aggfunc="first",
    )
    pivot = pivot[[method for method in methods if method in pivot.columns]]

    fig, ax = plt.subplots(figsize=(11, 5))
    pivot.plot(kind="bar", ax=ax, width=0.8)
    ax.set_title("Soft Router2 vs Simple Baselines")
    ax.set_xlabel("Router variant")
    ax.set_ylabel("Test MAE, lower is better")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    output_path = summary_dir / "router2_soft_vs_baselines.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    summary_dir = find_summary_dir()
    results = load_results(summary_dir)
    print_tables(results)

    output_paths = [
        save_best_by_variant_chart(results, summary_dir),
        save_trained_router_chart(results, summary_dir),
        save_soft_router_baseline_chart(results, summary_dir),
    ]

    print("\nSaved charts")
    for path in output_paths:
        print(path)


if __name__ == "__main__":
    main()
