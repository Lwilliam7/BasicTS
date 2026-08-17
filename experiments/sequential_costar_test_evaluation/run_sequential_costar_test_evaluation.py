"""Evaluate frozen Sequential COSTAR utility-ranking routers on test.

This is an after-final-test audit. It does not train, tune, or choose
checkpoints. The five seed checkpoints are the validation-selected
`utility_pairwise_weighted` Sequential COSTAR artifacts already present in the
repository.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_costarts_walkforward_cache import EXPERT_ORDER, sha256_file, validate_walkforward_cache  # noqa: E402
from scripts.sequential_costarts_model_full import SequentialCOSTARTSRouterFull  # noqa: E402
from scripts.train_sequential_costarts_utility_ranking import evaluate_router, load_normalizer_std  # noqa: E402


OUT_DIR = ROOT / "experiments" / "sequential_costar_test_evaluation"
SEEDS = (7, 11, 13, 17, 19)


DATASETS: dict[str, dict[str, Any]] = {
    "ETTh1": {
        "test_cache": ROOT / "experiments" / "final_test_evaluation" / "generated" / "caches" / "ETTh1" / "test_80_100_cache.pt",
        "expected_role": "test_80_100",
        "checkpoint_by_seed": {
            seed: ROOT
            / "checkpoints"
            / "costarts_walkforward"
            / "utility_ranking_weighted_pairwise"
            / "utility_pairwise_weighted"
            / f"seed_{seed}"
            / "best_utility_router.pt"
            for seed in SEEDS
        },
        "validation_summary": ROOT
        / "results"
        / "router_summary"
        / "costarts_walkforward"
        / "utility_ranking_weighted_pairwise"
        / "summary.json",
        "normalizer_checkpoint": ROOT / "checkpoints" / "costarts_walkforward" / "final_60" / "DLinear" / "best_expert.pt",
        "metric_scale": "normalized_by_ETTh1_DLinear_scaler",
        "validation_metric_keys": ("validation_mae", "validation_mse"),
        "baseline_refs": {
            "fixed3_mae": 0.36726489663124084,
            "final_frozen_adaptive_test_mae": 0.3263952910900116,
        },
    },
    "ETTh2": {
        "test_cache": ROOT
        / "experiments"
        / "final_test_evaluation"
        / "generated"
        / "caches"
        / "ETTh2"
        / "locked_test_cache_v2.pt",
        "expected_role": "locked_test",
        "checkpoint_by_seed": {
            7: ROOT
            / "checkpoints"
            / "costarts_fresh"
            / "ETTh2_96_12"
            / "sequential_utility_ranking"
            / "utility_pairwise_weighted"
            / "seed_7"
            / "best_utility_router.pt",
            11: ROOT
            / "checkpoints"
            / "costarts_fresh"
            / "ETTh2_96_12"
            / "sequential_utility_ranking_remaining"
            / "utility_pairwise_weighted"
            / "seed_11"
            / "best_utility_router.pt",
            13: ROOT
            / "checkpoints"
            / "costarts_fresh"
            / "ETTh2_96_12"
            / "sequential_utility_ranking_remaining"
            / "utility_pairwise_weighted"
            / "seed_13"
            / "best_utility_router.pt",
            17: ROOT
            / "checkpoints"
            / "costarts_fresh"
            / "ETTh2_96_12"
            / "sequential_utility_ranking_remaining"
            / "utility_pairwise_weighted"
            / "seed_17"
            / "best_utility_router.pt",
            19: ROOT
            / "checkpoints"
            / "costarts_fresh"
            / "ETTh2_96_12"
            / "sequential_utility_ranking_remaining"
            / "utility_pairwise_weighted"
            / "seed_19"
            / "best_utility_router.pt",
        },
        "validation_summary": ROOT
        / "results"
        / "router_summary"
        / "costarts_fresh"
        / "ETTh2_96_12"
        / "sequential_utility_ranking_combined"
        / "summary.json",
        "normalizer_checkpoint": None,
        "metric_scale": "canonical_raw_std_ones",
        "validation_metric_keys": ("mae", "mse"),
        "baseline_refs": {
            "best_fixed2_mae": 0.2752290368080139,
            "final_frozen_adaptive_test_mae": 0.29780814051628113,
        },
    },
}


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def assert_ready_without_loading_test() -> dict[str, Any]:
    checks: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "manifest_written_before_this_script_loads_test_cache",
        "test_cache_loaded_before_manifest": False,
        "datasets": {},
    }
    for dataset, spec in DATASETS.items():
        paths = [spec["test_cache"], spec["validation_summary"], *spec["checkpoint_by_seed"].values()]
        normalizer = spec.get("normalizer_checkpoint")
        if normalizer is not None:
            paths.append(normalizer)
        missing = [str(path) for path in paths if not Path(path).exists()]
        if missing:
            raise FileNotFoundError(f"{dataset} missing required frozen sequential artifacts: {missing}")
        checks["datasets"][dataset] = {
            "test_cache_planned": str(spec["test_cache"]),
            "validation_summary": str(spec["validation_summary"]),
            "checkpoint_by_seed": {str(seed): str(path) for seed, path in spec["checkpoint_by_seed"].items()},
            "all_required_paths_exist": True,
        }
    write_json(OUT_DIR / "manifest_before_test.json", checks)
    return checks


def load_test_cache(path: Path, expected_role: str) -> dict[str, Any]:
    if "test" not in str(path).lower():
        raise ValueError(f"Expected a test cache path for this authorized final audit: {path}")
    cache = torch.load(path, map_location="cpu", weights_only=False)
    role = cache.get("cache_role", cache.get("split_role"))
    if role != expected_role:
        raise ValueError(f"{path}: role={role!r}, expected {expected_role!r}")
    if tuple(cache["expert_names"]) != EXPERT_ORDER:
        raise ValueError(f"{path}: expert order mismatch: {cache['expert_names']!r}")
    if expected_role == "test_80_100":
        validate_walkforward_cache(cache, allow_test=True)
    if int(cache["forecast_horizon"]) != 12:
        raise ValueError(f"{path}: expected horizon 12, got {cache['forecast_horizon']}")
    if int(cache["num_features"]) != 7:
        raise ValueError(f"{path}: expected 7 variables, got {cache['num_features']}")
    return cache


def load_router(path: Path, device: torch.device) -> tuple[SequentialCOSTARTSRouterFull, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = SequentialCOSTARTSRouterFull(**checkpoint["router_config"])
    model.load_state_dict(checkpoint["router_state_dict"], strict=True)
    model.to(device)
    return model, checkpoint


def validation_by_seed(summary_path: Path, dataset: str) -> dict[int, dict[str, float]]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if dataset == "ETTh1":
        per_seed = payload.get("summaries", [payload])[0]["per_seed"]
        return {
            int(row["seed"]): {
                "validation_mae": float(row["validation_mae"]),
                "validation_mse": float(row["validation_mse"]),
            }
            for row in per_seed
        }
    return {
        int(row["seed"]): {
            "validation_mae": float(row["mae"]),
            "validation_mse": float(row["mse"]),
        }
        for row in payload["per_seed"]
    }


def metric_summary(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.pstdev(values))


def evaluate_dataset(dataset: str, spec: Mapping[str, Any], device: torch.device, batch_size: int) -> dict[str, Any]:
    val_lookup = validation_by_seed(Path(spec["validation_summary"]), dataset)
    cache = load_test_cache(Path(spec["test_cache"]), str(spec["expected_role"]))
    if spec.get("normalizer_checkpoint") is None:
        std = torch.ones(int(cache["num_features"]), dtype=torch.float32)
    else:
        std = load_normalizer_std(Path(spec["normalizer_checkpoint"]))
    rows = []
    per_window_dir = OUT_DIR / "per_window" / dataset
    per_window_dir.mkdir(parents=True, exist_ok=True)

    eval_args = argparse.Namespace(
        batch_size=batch_size,
        max_queries=5,
        query_threshold=0.0,
        near_zero_epsilon=0.001,
    )
    for seed in SEEDS:
        checkpoint_path = Path(spec["checkpoint_by_seed"][seed])
        model, checkpoint = load_router(checkpoint_path, device)
        metrics = evaluate_router(model, cache, device, eval_args, std)
        write_csv(
            per_window_dir / f"seed_{seed}_per_window.csv",
            metrics["per_window"],
            [
                "cache_index",
                "absolute_window_start",
                "query_count",
                "queried_experts",
                "raw_mae",
                "raw_mse",
                "normalized_mae",
                "normalized_mse",
            ],
        )
        val = val_lookup[seed]
        primary_test_mae = float(metrics["mae"])
        primary_test_mse = float(metrics["mse"])
        row = {
            "dataset": dataset,
            "method": "Sequential COSTAR utility_pairwise_weighted",
            "seed": seed,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "best_epoch": checkpoint.get("epoch", ""),
            "test_mae": primary_test_mae,
            "test_mse": primary_test_mse,
            "raw_test_mae": float(metrics["raw_mae"]),
            "raw_test_mse": float(metrics["raw_mse"]),
            "validation_mae": val["validation_mae"],
            "validation_mse": val["validation_mse"],
            "test_minus_validation_mae": primary_test_mae - val["validation_mae"],
            "test_minus_validation_mse": primary_test_mse - val["validation_mse"],
            "average_queries": float(metrics["average_queries"]),
            "top1_utility_accuracy": float(metrics["top1_utility_accuracy"]),
            "top2_utility_coverage": float(metrics["top2_utility_coverage"]),
            "mean_regret": float(metrics["mean_regret"]),
            "metric_scale": spec["metric_scale"],
            "selection_protocol": "after-final-test audit of existing frozen validation-selected sequential utility-ranking checkpoints; no retraining or tuning",
        }
        rows.append(row)

    mean_mae, std_mae = metric_summary([float(row["test_mae"]) for row in rows])
    mean_mse, std_mse = metric_summary([float(row["test_mse"]) for row in rows])
    mean_val_mae, std_val_mae = metric_summary([float(row["validation_mae"]) for row in rows])
    mean_val_mse, std_val_mse = metric_summary([float(row["validation_mse"]) for row in rows])
    mean_queries, std_queries = metric_summary([float(row["average_queries"]) for row in rows])
    aggregate = {
        "dataset": dataset,
        "method": "Sequential COSTAR utility_pairwise_weighted",
        "seeds": list(SEEDS),
        "test_mae_mean": mean_mae,
        "test_mae_std": std_mae,
        "test_mse_mean": mean_mse,
        "test_mse_std": std_mse,
        "validation_mae_mean": mean_val_mae,
        "validation_mae_std": std_val_mae,
        "validation_mse_mean": mean_val_mse,
        "validation_mse_std": std_val_mse,
        "test_minus_validation_mae": mean_mae - mean_val_mae,
        "average_queries_mean": mean_queries,
        "average_queries_std": std_queries,
        "metric_scale": spec["metric_scale"],
        "test_cache": str(spec["test_cache"]),
        "test_cache_sha256": sha256_file(Path(spec["test_cache"])),
        "num_test_windows": int(cache["num_windows"]),
        "test_start_min": int(cache["absolute_window_starts"].min().item()),
        "test_start_max": int(cache["absolute_window_starts"].max().item()),
        "baseline_refs": spec["baseline_refs"],
        "per_seed": rows,
    }
    return aggregate


def write_report(results: Sequence[Mapping[str, Any]], runtime_sec: float, device: str) -> None:
    lines = [
        "# Sequential COSTAR Test Evaluation",
        "",
        "This is an after-final-test audit requested after held-out test metrics were already seen. Existing `utility_pairwise_weighted` Sequential COSTAR checkpoints were evaluated once on the existing final-test caches. No training, tuning, or checkpoint selection was performed.",
        "",
        f"- Device: `{device}`",
        f"- Runtime seconds: `{runtime_sec:.3f}`",
        f"- Test cache loaded: `true`",
        "",
        "| Dataset | Method | Test MAE mean | Test MAE std | Test MSE mean | Val MAE mean | Avg queries | Metric scale |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        lines.append(
            "| {dataset} | {method} | `{test_mae_mean:.6f}` | `{test_mae_std:.6f}` | `{test_mse_mean:.6f}` | `{validation_mae_mean:.6f}` | `{average_queries_mean:.3f}` | `{metric_scale}` |".format(
                **result
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- ETTh1 sequential utility routing remains worse than the previously tested fixed-core and adaptive COSTAR test rows.",
            "- ETTh2 sequential utility routing remains worse than the final frozen adaptive ETTh2 test row.",
            "- These rows should be treated as additional after-final-test audit results, not preregistered final competitors.",
            "",
            "## Reproduce",
            "",
            "```powershell",
            "python experiments\\sequential_costar_test_evaluation\\run_sequential_costar_test_evaluation.py --device cuda",
            "```",
        ]
    )
    (OUT_DIR / "SEQUENTIAL_COSTAR_TEST_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    started = time.perf_counter()
    manifest = assert_ready_without_loading_test()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    results = [evaluate_dataset(dataset, spec, device, args.batch_size) for dataset, spec in DATASETS.items()]
    runtime_sec = time.perf_counter() - started
    all_seed_rows = [row for result in results for row in result["per_seed"]]
    write_csv(
        OUT_DIR / "sequential_costar_test_per_seed.csv",
        all_seed_rows,
        [
            "dataset",
            "method",
            "seed",
            "checkpoint",
            "checkpoint_sha256",
            "best_epoch",
            "test_mae",
            "test_mse",
            "raw_test_mae",
            "raw_test_mse",
            "validation_mae",
            "validation_mse",
            "test_minus_validation_mae",
            "test_minus_validation_mse",
            "average_queries",
            "top1_utility_accuracy",
            "top2_utility_coverage",
            "mean_regret",
            "metric_scale",
            "selection_protocol",
        ],
    )
    aggregate_rows = [
        {key: value for key, value in result.items() if key != "per_seed"} for result in results
    ]
    write_csv(
        OUT_DIR / "sequential_costar_test_results.csv",
        aggregate_rows,
        [
            "dataset",
            "method",
            "test_mae_mean",
            "test_mae_std",
            "test_mse_mean",
            "test_mse_std",
            "validation_mae_mean",
            "validation_mae_std",
            "validation_mse_mean",
            "validation_mse_std",
            "test_minus_validation_mae",
            "average_queries_mean",
            "average_queries_std",
            "metric_scale",
            "test_cache",
            "test_cache_sha256",
            "num_test_windows",
            "test_start_min",
            "test_start_max",
        ],
    )
    final_payload = {
        "manifest": manifest,
        "test_evaluation_complete": True,
        "test_cache_loaded_after_manifest": True,
        "runtime_sec": runtime_sec,
        "device": str(device),
        "results": results,
    }
    write_json(OUT_DIR / "SEQUENTIAL_COSTAR_TEST_RESULTS.json", final_payload)
    write_report(results, runtime_sec, str(device))
    print(json.dumps({"runtime_sec": runtime_sec, "device": str(device), "results": aggregate_rows}, indent=2))


if __name__ == "__main__":
    main()
