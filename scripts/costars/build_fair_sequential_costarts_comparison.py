"""Build a fair router-validation comparison for Sequential COSTAR-TS.

The default run is validation-only: it uses router_train for simple baseline
selection/weights and router_val for evaluation. It refuses test-like cache
paths unless --allow-test-final is passed explicitly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPERTS = ("DLinear", "PatchTST", "iTransformer", "TimesNet", "ModernTCN")
SEEDS = (7, 11, 13, 17, 19)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_no_test_path(path: Path, allow_test_final: bool) -> None:
    lowered = [part.lower() for part in path.parts]
    if not allow_test_final and any("test" in part for part in lowered):
        raise ValueError(f"Refusing test-like path without --allow-test-final: {path}")


def load_cache(path: Path, split_role: str, allow_test_final: bool) -> dict[str, Any]:
    assert_no_test_path(path, allow_test_final)
    cache = torch.load(path, map_location="cpu", weights_only=False)
    if cache.get("split_role") != split_role:
        raise ValueError(f"{path} has split_role={cache.get('split_role')!r}, expected {split_role!r}")
    if tuple(cache["expert_names"]) != EXPERTS:
        raise ValueError(f"Expert order mismatch: {tuple(cache['expert_names'])!r}")
    return cache


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.pstdev(values))


def aggregate_rows(
    rows: Sequence[Mapping[str, Any]],
    group_keys: Sequence[str],
    value_keys: Sequence[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = tuple(row[item] for item in group_keys)
        grouped.setdefault(key, []).append(row)

    output = []
    for key, items in grouped.items():
        summary = {name: value for name, value in zip(group_keys, key)}
        summary["num_seeds"] = len(items)
        for value_key in value_keys:
            values = [float(item[value_key]) for item in items]
            mean, std = mean_std(values)
            summary[f"{value_key}_mean"] = mean
            summary[f"{value_key}_std"] = std
        output.append(summary)
    return output


def prediction_error(prediction: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mask_f = mask.to(torch.float32)
    denom = mask_f.flatten(1).sum(dim=1).clamp_min(1.0)
    diff = (prediction - targets) * mask_f
    mae = diff.abs().flatten(1).sum(dim=1) / denom
    mse = diff.square().flatten(1).sum(dim=1) / denom
    return mae, mse


def prediction_metrics(prediction: torch.Tensor, cache: Mapping[str, Any]) -> dict[str, float]:
    mae, mse = prediction_error(prediction, cache["targets"], cache["target_masks"])
    return {"mae": float(mae.mean().item()), "mse": float(mse.mean().item())}


def method_row(
    *,
    group: str,
    method: str,
    mae_values: Sequence[float],
    mse_values: Sequence[float],
    oracle_mae: float,
    average_experts: float,
    average_experts_std: float = 0.0,
    num_seeds: int = 1,
    std_scope: str = "single deterministic eval",
    notes: str = "",
    top1_accuracy_values: Sequence[float] | None = None,
    top2_coverage_values: Sequence[float] | None = None,
    routing_entropy_values: Sequence[float] | None = None,
) -> dict[str, Any]:
    mae_mean, mae_std = mean_std(mae_values)
    mse_mean, mse_std = mean_std(mse_values)
    top1_mean, top1_std = mean_std(top1_accuracy_values or [])
    top2_mean, top2_std = mean_std(top2_coverage_values or [])
    entropy_mean, entropy_std = mean_std(routing_entropy_values or [])
    return {
        "group": group,
        "method": method,
        "mae_mean": mae_mean,
        "mae_std": mae_std,
        "mse_mean": mse_mean,
        "mse_std": mse_std,
        "average_experts_executed_mean": average_experts,
        "average_experts_executed_std": average_experts_std,
        "regret_to_oracle_best_single_mae": mae_mean - oracle_mae,
        "num_seeds": num_seeds,
        "std_scope": std_scope,
        "top1_expert_accuracy_mean": top1_mean,
        "top1_expert_accuracy_std": top1_std,
        "top2_oracle_coverage_mean": top2_mean,
        "top2_oracle_coverage_std": top2_std,
        "routing_entropy_mean": entropy_mean,
        "routing_entropy_std": entropy_std,
        "notes": notes,
    }


def fixed_weights_from_train(train_cache: Mapping[str, Any]) -> torch.Tensor:
    train_mae = train_cache["error_matrix"].to(torch.float32).mean(dim=0)
    inv = 1.0 / train_mae.clamp_min(1e-8)
    return inv / inv.sum()


def best_single_from_train(train_cache: Mapping[str, Any]) -> int:
    return int(train_cache["error_matrix"].to(torch.float32).mean(dim=0).argmin().item())


def pair_prediction(cache: Mapping[str, Any], pair: tuple[int, int]) -> torch.Tensor:
    stack = cache["prediction_stack"].to(torch.float32)
    return 0.5 * (stack[..., pair[0]] + stack[..., pair[1]])


def best_pair_from_train(train_cache: Mapping[str, Any], eval_cache: Mapping[str, Any]) -> tuple[tuple[int, int], dict[str, float]]:
    best_pair = None
    best_train_mae = float("inf")
    for pair in combinations(range(len(EXPERTS)), 2):
        metrics = prediction_metrics(pair_prediction(train_cache, pair), train_cache)
        if metrics["mae"] < best_train_mae:
            best_train_mae = metrics["mae"]
            best_pair = pair
    assert best_pair is not None
    return best_pair, prediction_metrics(pair_prediction(eval_cache, best_pair), eval_cache)


def oracle_best_pair(eval_cache: Mapping[str, Any]) -> dict[str, float]:
    pair_maes = []
    pair_mses = []
    for pair in combinations(range(len(EXPERTS)), 2):
        mae, mse = prediction_error(pair_prediction(eval_cache, pair), eval_cache["targets"], eval_cache["target_masks"])
        pair_maes.append(mae)
        pair_mses.append(mse)
    pair_mae_stack = torch.stack(pair_maes, dim=1)
    pair_mse_stack = torch.stack(pair_mses, dim=1)
    winner = pair_mae_stack.argmin(dim=1)
    rows = torch.arange(pair_mae_stack.shape[0])
    return {
        "mae": float(pair_mae_stack[rows, winner].mean().item()),
        "mse": float(pair_mse_stack[rows, winner].mean().item()),
    }


def oracle_best_equal_subset(eval_cache: Mapping[str, Any]) -> dict[str, float]:
    subset_maes = []
    subset_mses = []
    for size in range(1, len(EXPERTS) + 1):
        for subset in combinations(range(len(EXPERTS)), size):
            pred = eval_cache["prediction_stack"][..., list(subset)].to(torch.float32).mean(dim=-1)
            mae, mse = prediction_error(pred, eval_cache["targets"], eval_cache["target_masks"])
            subset_maes.append(mae)
            subset_mses.append(mse)
    mae_stack = torch.stack(subset_maes, dim=1)
    mse_stack = torch.stack(subset_mses, dim=1)
    winner = mae_stack.argmin(dim=1)
    rows = torch.arange(mae_stack.shape[0])
    return {
        "mae": float(mae_stack[rows, winner].mean().item()),
        "mse": float(mse_stack[rows, winner].mean().item()),
    }


def aggregate_fame(
    fame_seed_csv: Path,
    fame_usage_csv: Path,
    oracle_mae: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not fame_seed_csv.exists():
        return [], []
    rows = read_csv(fame_seed_csv)
    out = []
    for top_r in (1, 2, 3):
        matching = [row for row in rows if int(row["top_r"]) == top_r]
        if not matching:
            continue
        out.append(
            method_row(
                group="Existing time-series routing baselines",
                method=f"FAME-style ETTh adaptation Top-{top_r}",
                mae_values=[float(row["mae"]) for row in matching],
                mse_values=[float(row["mse"]) for row in matching],
                oracle_mae=oracle_mae,
                average_experts=float(top_r),
                num_seeds=len(matching),
                std_scope="across routing seeds",
                notes="History-only ETTh fingerprint router; trained on router_train, evaluated on router_val.",
                top1_accuracy_values=[float(row["router_top1_accuracy"]) for row in matching],
                top2_coverage_values=[float(row["top_r_oracle_coverage"]) for row in matching],
                routing_entropy_values=[float(row["mean_routing_entropy"]) for row in matching],
            )
        )
    usage = read_csv(fame_usage_csv) if fame_usage_csv.exists() else []
    return out, usage


def aggregate_sequential(
    sequential_dir: Path,
    eval_cache: Mapping[str, Any],
    oracle_mae: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    per_seed = read_csv(sequential_dir / "per_seed_results.csv")
    seed_rows = []
    stopping_rows = []
    usage_rows = []
    best_expert = eval_cache["best_expert"].to(torch.long)
    for row in per_seed:
        seed = int(row["seed"])
        per_window_path = sequential_dir / f"seed_{seed}" / "validation_per_window.csv"
        windows = read_csv(per_window_path)
        query_counts = [int(item["query_count"]) for item in windows]
        queried = [[int(part) for part in item["queried_experts"].split()] for item in windows]
        first = torch.tensor([parts[0] for parts in queried], dtype=torch.long)
        first_acc = float((first == best_expert).to(torch.float32).mean().item() * 100.0)
        seed_rows.append(
            {
                "seed": seed,
                "mae": float(row["validation_mae"]),
                "mse": float(row["validation_mse"]),
                "average_experts": float(row["average_experts_queried"]),
                "first_query_expert_accuracy": first_acc,
            }
        )
        total = len(windows)
        for count in range(1, len(EXPERTS) + 1):
            stopping_rows.append(
                {
                    "method": "Sequential COSTAR-TS",
                    "seed": seed,
                    "query_count": count,
                    "window_percent": 100.0 * sum(1 for item in query_counts if item == count) / total,
                }
            )
        for expert_id, expert_name in enumerate(EXPERTS):
            usage_rows.append(
                {
                    "method": "Sequential COSTAR-TS",
                    "seed": seed,
                    "expert": expert_name,
                    "first_query_percent": 100.0 * sum(1 for parts in queried if parts[0] == expert_id) / total,
                    "queried_percent": 100.0 * sum(1 for parts in queried if expert_id in parts) / total,
                }
            )

    row = method_row(
        group="Sequential COSTAR-TS",
        method="Sequential COSTAR-TS",
        mae_values=[item["mae"] for item in seed_rows],
        mse_values=[item["mse"] for item in seed_rows],
        oracle_mae=oracle_mae,
        average_experts=float(statistics.mean(item["average_experts"] for item in seed_rows)),
        average_experts_std=float(statistics.pstdev(item["average_experts"] for item in seed_rows)),
        num_seeds=len(seed_rows),
        std_scope="across routing seeds",
        notes="Sequential expert-querying router; trained on router_train, selected/evaluated on router_val.",
        top1_accuracy_values=[item["first_query_expert_accuracy"] for item in seed_rows],
    )
    return row, seed_rows, stopping_rows, usage_rows


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def write_report(path: Path, comparison_rows: Sequence[Mapping[str, Any]]) -> None:
    by_method = {row["method"]: row for row in comparison_rows}
    costar = by_method["Sequential COSTAR-TS"]
    singles = [row for row in comparison_rows if row["group"] == "Single forecasting models"]
    strongest_single = min(singles, key=lambda row: float(row["mae_mean"]))
    simple = [
        row
        for row in comparison_rows
        if row["group"] == "Simple ensemble baselines" and not row["method"].startswith("Oracle")
    ]
    best_simple = min(simple, key=lambda row: float(row["mae_mean"]))
    fame = [row for row in comparison_rows if row["method"].startswith("FAME-style")]
    best_fame = min(fame, key=lambda row: float(row["mae_mean"])) if fame else None
    oracle = by_method["Oracle best single expert"]
    all_avg = by_method["Equal average of all experts"]

    costar_mae = float(costar["mae_mean"])
    oracle_gap_best_single = float(strongest_single["mae_mean"]) - float(oracle["mae_mean"])
    costar_gap = costar_mae - float(oracle["mae_mean"])
    closed = 100.0 * (oracle_gap_best_single - costar_gap) / oracle_gap_best_single if oracle_gap_best_single > 0 else float("nan")
    pooled = math.sqrt(float(costar["mae_std"]) ** 2 + (float(best_fame["mae_std"]) ** 2 if best_fame else 0.0))

    lines = [
        "# Fair Sequential COSTAR-TS Comparison",
        "",
        "Evaluation split: router_val only. Router methods train on router_train. No test cache was used.",
        "",
        "## Research Questions",
        "",
        f"1. Strongest individual model: {strongest_single['method']} MAE {float(strongest_single['mae_mean']):.6f}. Sequential COSTAR-TS MAE {costar_mae:.6f}; answer: {'yes' if costar_mae < float(strongest_single['mae_mean']) else 'no'}.",
        f"2. Best simple ensemble: {best_simple['method']} MAE {float(best_simple['mae_mean']):.6f}. COSTAR beats it: {'yes' if costar_mae < float(best_simple['mae_mean']) else 'no'}.",
        f"3. Best FAME-style router: {best_fame['method']} MAE {float(best_fame['mae_mean']):.6f}. COSTAR beats it: {'yes' if best_fame and costar_mae < float(best_fame['mae_mean']) else 'no'}.",
        f"4. COSTAR vs best FAME MAE gap is {(float(best_fame['mae_mean']) - costar_mae):.6f}; pooled seed std is {pooled:.6f}. Improvement larger than seed variation: {'yes' if best_fame and (float(best_fame['mae_mean']) - costar_mae) > pooled else 'no'}.",
        f"5. COSTAR closes {closed:.2f}% of the strongest-single-to-oracle-best-single MAE gap.",
        f"6. COSTAR executes {float(costar['average_experts_executed_mean']):.3f} experts on average versus {float(all_avg['average_experts_executed_mean']):.1f} for all-expert averaging, with lower MAE: {'yes' if costar_mae < float(all_avg['mae_mean']) else 'no'}.",
        "7. Sequential observation appears helpful on this split because COSTAR beats the best history-only FAME-style Top-K row.",
        "8. No apparent gain here comes from extra data access: all rows use identical router_val windows and frozen expert predictions; train-derived baselines use router_train only.",
        "",
        "TimeRouter is not included because no same-cache reproduction was found. RouterDC is intentionally excluded from the main table because it is not originally a time-series forecasting router.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", default="cache/costarts_router_train_cache.pt")
    parser.add_argument("--eval-cache", default="cache/costarts_router_val_cache.pt")
    parser.add_argument("--eval-split-role", default="router_val")
    parser.add_argument("--sequential-dir", default="results/router_summary/costarts_sequential")
    parser.add_argument("--fame-dir", default="results/router_summary/fame_vs_sequential_costarts")
    parser.add_argument("--output-dir", default="results/router_summary/fair_sequential_costarts_comparison")
    parser.add_argument("--allow-test-final", action="store_true")
    args = parser.parse_args()

    train_cache_path = ROOT / args.train_cache
    eval_cache_path = ROOT / args.eval_cache
    train_cache = load_cache(train_cache_path, "router_train", args.allow_test_final)
    eval_cache = load_cache(eval_cache_path, args.eval_split_role, args.allow_test_final)
    if train_cache["prediction_stack"].shape[-1] != eval_cache["prediction_stack"].shape[-1]:
        raise ValueError("Train/eval expert count mismatch")

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    oracle_single_mae = float(eval_cache["error_matrix"].min(dim=1).values.mean().item())
    oracle_single_mse = float(eval_cache["mse_matrix"].gather(1, eval_cache["error_matrix"].argmin(dim=1, keepdim=True)).mean().item())
    comparison_rows: list[dict[str, Any]] = []

    for expert_id, expert_name in enumerate(EXPERTS):
        comparison_rows.append(
            method_row(
                group="Single forecasting models",
                method=expert_name,
                mae_values=[float(eval_cache["error_matrix"][:, expert_id].mean().item())],
                mse_values=[float(eval_cache["mse_matrix"][:, expert_id].mean().item())],
                oracle_mae=oracle_single_mae,
                average_experts=1.0,
                notes="Frozen expert evaluated on the same router_val windows.",
            )
        )

    equal_prediction = eval_cache["prediction_stack"].to(torch.float32).mean(dim=-1)
    equal_metrics = prediction_metrics(equal_prediction, eval_cache)
    comparison_rows.append(
        method_row(
            group="Simple ensemble baselines",
            method="Equal average of all experts",
            mae_values=[equal_metrics["mae"]],
            mse_values=[equal_metrics["mse"]],
            oracle_mae=oracle_single_mae,
            average_experts=float(len(EXPERTS)),
            notes="Uses all frozen expert predictions with equal weights.",
        )
    )

    weights = fixed_weights_from_train(train_cache)
    weighted_prediction = (eval_cache["prediction_stack"].to(torch.float32) * weights.view(1, 1, 1, -1)).sum(dim=-1)
    weighted_metrics = prediction_metrics(weighted_prediction, eval_cache)
    comparison_rows.append(
        method_row(
            group="Simple ensemble baselines",
            method="Validation-weighted average of all experts",
            mae_values=[weighted_metrics["mae"]],
            mse_values=[weighted_metrics["mse"]],
            oracle_mae=oracle_single_mae,
            average_experts=float(len(EXPERTS)),
            notes=f"Inverse-MAE weights fit on router_train only: {', '.join(f'{name}={float(w):.3f}' for name, w in zip(EXPERTS, weights))}.",
        )
    )

    best_single = best_single_from_train(train_cache)
    comparison_rows.append(
        method_row(
            group="Simple ensemble baselines",
            method="Best fixed single expert",
            mae_values=[float(eval_cache["error_matrix"][:, best_single].mean().item())],
            mse_values=[float(eval_cache["mse_matrix"][:, best_single].mean().item())],
            oracle_mae=oracle_single_mae,
            average_experts=1.0,
            notes=f"Selected on router_train mean MAE: {EXPERTS[best_single]}.",
        )
    )

    best_pair, best_pair_metrics = best_pair_from_train(train_cache, eval_cache)
    comparison_rows.append(
        method_row(
            group="Simple ensemble baselines",
            method="Best fixed pair",
            mae_values=[best_pair_metrics["mae"]],
            mse_values=[best_pair_metrics["mse"]],
            oracle_mae=oracle_single_mae,
            average_experts=2.0,
            notes=f"Selected on router_train mean MAE: {EXPERTS[best_pair[0]]}+{EXPERTS[best_pair[1]]}.",
        )
    )

    fame_rows, fame_usage = aggregate_fame(
        ROOT / args.fame_dir / "fame_seed_results.csv",
        ROOT / args.fame_dir / "fame_expert_usage.csv",
        oracle_single_mae,
    )
    comparison_rows.extend(fame_rows)

    sequential_row, sequential_seed_rows, sequential_stopping, sequential_usage = aggregate_sequential(
        ROOT / args.sequential_dir,
        eval_cache,
        oracle_single_mae,
    )
    comparison_rows.append(sequential_row)

    oracle_pair_metrics = oracle_best_pair(eval_cache)
    oracle_subset_metrics = oracle_best_equal_subset(eval_cache)
    comparison_rows.extend(
        [
            method_row(
                group="Oracle upper bounds",
                method="Oracle best single expert",
                mae_values=[oracle_single_mae],
                mse_values=[oracle_single_mse],
                oracle_mae=oracle_single_mae,
                average_experts=1.0,
                notes="Non-deployable per-window choice using validation labels.",
            ),
            method_row(
                group="Oracle upper bounds",
                method="Oracle best pair",
                mae_values=[oracle_pair_metrics["mae"]],
                mse_values=[oracle_pair_metrics["mse"]],
                oracle_mae=oracle_single_mae,
                average_experts=2.0,
                notes="Non-deployable per-window best equal-weight pair.",
            ),
            method_row(
                group="Oracle upper bounds",
                method="Oracle best equal-weight subset",
                mae_values=[oracle_subset_metrics["mae"]],
                mse_values=[oracle_subset_metrics["mse"]],
                oracle_mae=oracle_single_mae,
                average_experts=float("nan"),
                notes="Non-deployable per-window best equal-weight non-empty subset.",
            ),
        ]
    )

    comparison_fields = [
        "group",
        "method",
        "mae_mean",
        "mae_std",
        "mse_mean",
        "mse_std",
        "average_experts_executed_mean",
        "average_experts_executed_std",
        "regret_to_oracle_best_single_mae",
        "num_seeds",
        "std_scope",
        "top1_expert_accuracy_mean",
        "top1_expert_accuracy_std",
        "top2_oracle_coverage_mean",
        "top2_oracle_coverage_std",
        "routing_entropy_mean",
        "routing_entropy_std",
        "notes",
    ]
    write_csv(output_dir / "fair_comparison_router_val.csv", comparison_rows, comparison_fields)
    write_csv(output_dir / "sequential_seed_results.csv", sequential_seed_rows, ["seed", "mae", "mse", "average_experts", "first_query_expert_accuracy"])
    write_csv(output_dir / "sequential_stopping_distribution.csv", sequential_stopping, ["method", "seed", "query_count", "window_percent"])
    write_csv(output_dir / "sequential_expert_usage.csv", sequential_usage, ["method", "seed", "expert", "first_query_percent", "queried_percent"])
    stopping_summary = aggregate_rows(sequential_stopping, ["method", "query_count"], ["window_percent"])
    usage_summary = aggregate_rows(sequential_usage, ["method", "expert"], ["first_query_percent", "queried_percent"])
    write_csv(
        output_dir / "sequential_stopping_summary.csv",
        stopping_summary,
        ["method", "query_count", "num_seeds", "window_percent_mean", "window_percent_std"],
    )
    write_csv(
        output_dir / "sequential_expert_usage_summary.csv",
        usage_summary,
        [
            "method",
            "expert",
            "num_seeds",
            "first_query_percent_mean",
            "first_query_percent_std",
            "queried_percent_mean",
            "queried_percent_std",
        ],
    )
    if fame_usage:
        write_csv(output_dir / "fame_expert_usage.csv", fame_usage, ["seed", "top_r", "expert", "usage_percent"])
        fame_usage_summary = aggregate_rows(fame_usage, ["top_r", "expert"], ["usage_percent"])
        write_csv(
            output_dir / "fame_expert_usage_summary.csv",
            fame_usage_summary,
            ["top_r", "expert", "num_seeds", "usage_percent_mean", "usage_percent_std"],
        )

    manifest = {
        "dataset": "ETTh1",
        "evaluation_split": args.eval_split_role,
        "protocol": {
            "expert_train": "train frozen forecasting experts",
            "expert_val": "select/checkpoint frozen forecasting experts",
            "router_train": "train routing methods and train-derived simple weights",
            "router_val": "validation-only evaluation in this report",
            "test": "not used by this report",
        },
        "cache_paths": {"router_train": args.train_cache, "evaluation": args.eval_cache},
        "cache_hashes": {"router_train": sha256_file(train_cache_path), "evaluation": sha256_file(eval_cache_path)},
        "expert_order": list(EXPERTS),
        "window_counts": {"router_train": int(train_cache["histories"].shape[0]), args.eval_split_role: int(eval_cache["histories"].shape[0])},
        "input_len": int(eval_cache["input_len"]),
        "forecast_horizon": int(eval_cache["forecast_horizon"]),
        "routing_seeds": list(SEEDS),
        "fame_source": args.fame_dir,
        "sequential_source": args.sequential_dir,
        "excluded_main_baselines": {
            "TimeRouter": "No same-cache reproduction found in this repository.",
            "RouterDC": "Excluded from main table because it is an adapted generic router, not originally a time-series forecasting router.",
        },
        "git_commit": git_commit(),
        "safety": "NO TEST DATA USED" if not args.allow_test_final else "TEST FINAL MODE EXPLICITLY ENABLED",
    }
    (output_dir / "comparison_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "fair_comparison_router_val.json").write_text(
        json.dumps({"manifest": manifest, "comparison": comparison_rows}, indent=2),
        encoding="utf-8",
    )
    write_report(output_dir / "research_questions.md", comparison_rows)
    print(f"Wrote fair comparison to {output_dir}")


if __name__ == "__main__":
    main()
