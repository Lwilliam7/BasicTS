"""Evaluate ETTh2 pair-potential train-fitted linear ensembles on test.

This is an after-final-test audit requested after the official final frozen
test evaluation. The two evaluated methods were already fit from ETTh2
router-train in the pair-potential validation analysis; no tuning or model
selection is performed here.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.chronological_adaptive_costar.run_chronological_adaptive_costar import paired_bootstrap  # noqa: E402
from scripts.costars.analyze_etth2_pair_potential import (  # noqa: E402
    EXPECTED_EXPERTS,
    aggregate_from_per_window,
    fit_ridge_weights,
    fit_simplex_weights,
    sha256_file,
    weighted_errors,
)


OUT_DIR = ROOT / "experiments" / "etth2_pair_potential_test_evaluation"
SUMMARY_PATH = ROOT / "results" / "router_summary" / "costarts_fresh" / "ETTh2_96_12" / "pair_potential" / "pair_potential_summary.json"
TRAIN_CACHE = ROOT / "cache" / "costarts_fresh" / "ETTh2_96_12" / "router_train_cache.pt"
VAL_CACHE = ROOT / "cache" / "costarts_fresh" / "ETTh2_96_12" / "router_val_cache.pt"
TEST_CACHE = ROOT / "experiments" / "final_test_evaluation" / "generated" / "caches" / "ETTh2" / "locked_test_cache_v2.pt"
ALL_RESULTS = ROOT / "experiments" / "all_results_summary" / "all_costar_results.csv"
TOP_RESULTS = ROOT / "experiments" / "frozen_model_test_results" / "etth2_top_costar_test_results.csv"

VALIDATION_REFS = {
    "nonnegative_simplex_linear_average": {"mae": 0.274755, "mse": 0.165479},
    "ridge_linear_stacker": {"mae": 0.276702, "mse": 0.165339},
}
FINAL_ADAPTIVE_TEST_MAE = 0.29780814051628113
SINGLE_DLINEAR_TEST_MAE = 0.30170753598213196
BEST_FIXED2_TEST_MAE = 0.29926300048828125


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "method",
        "dataset",
        "expert_set",
        "test_mae",
        "test_mse",
        "validation_mae",
        "validation_mse",
        "diff_vs_validation",
        "diff_vs_single_DLinear_test",
        "diff_vs_ETTh2_full_adaptive_test",
        "diff_vs_validation_selected_DLinear_ModernTCN_test",
        "selection_protocol",
        "status",
        "weights_json",
        "paired_ci95_diff_vs_DLinear_low",
        "paired_ci95_diff_vs_DLinear_high",
        "paired_ci_excludes_zero",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def load_cache(path: Path, expected_role: str, *, allow_test: bool) -> dict[str, Any]:
    if not allow_test and "test" in str(path).lower():
        raise AssertionError(f"Refusing to load test path before manifest: {path}")
    cache = torch.load(path, map_location="cpu", weights_only=False)
    role = cache.get("cache_role", cache.get("split_role"))
    if role != expected_role:
        raise ValueError(f"{path}: role {role!r} != expected {expected_role!r}")
    if tuple(cache["expert_names"]) != EXPECTED_EXPERTS:
        raise ValueError("ETTh2 expert order changed")
    if expected_role == "router_train":
        starts = cache["absolute_window_starts"]
        if int(starts.min()) != 8640 or int(starts.max()) != 10692 or int(cache["num_windows"]) != 2053:
            raise AssertionError("Unexpected ETTh2 router_train split")
    if expected_role == "router_val":
        starts = cache["absolute_window_starts"]
        if int(starts.min()) != 10800 or int(starts.max()) != 11412 or int(cache["num_windows"]) != 613:
            raise AssertionError("Unexpected canonical ETTh2 router_val split")
    if expected_role == "locked_test":
        starts = cache["absolute_window_starts"]
        if int(starts.min()) != 11520 or int(starts.max()) != 14292 or int(cache["num_windows"]) != 2773:
            raise AssertionError("Unexpected ETTh2 locked_test split")
    return cache


def validate_weight_vector(method: str, observed: torch.Tensor, summary: Mapping[str, Any]) -> None:
    expected_map = summary["fitted_train_only_weights"][method]
    expected = torch.tensor([expected_map[name] for name in EXPECTED_EXPERTS], dtype=torch.float32)
    if not torch.allclose(observed, expected, atol=2e-5, rtol=2e-5):
        raise AssertionError(f"{method} weights do not reproduce prior validation artifact")
    if method == "nonnegative_simplex_linear_average":
        if bool((observed < -1e-7).any()):
            raise AssertionError("Simplex weights contain negative entries")
        if abs(float(observed.sum()) - 1.0) > 2e-5:
            raise AssertionError("Simplex weights do not sum to one")


def result_row(
    method: str,
    weights: torch.Tensor,
    val_cache: Mapping[str, Any],
    test_cache: Mapping[str, Any],
    dlinear_test_mae: torch.Tensor,
) -> dict[str, Any]:
    val_mae, val_mse = weighted_errors(val_cache, weights)
    test_mae, test_mse = weighted_errors(test_cache, weights)
    ref = VALIDATION_REFS[method]
    if abs(float(val_mae.mean()) - ref["mae"]) > 5e-6:
        raise AssertionError(f"{method} validation MAE failed to reproduce")
    if abs(float(val_mse.mean()) - ref["mse"]) > 5e-6:
        raise AssertionError(f"{method} validation MSE failed to reproduce")
    test_summary = aggregate_from_per_window(test_mae, test_mse)
    boot = paired_bootstrap(test_mae, dlinear_test_mae, seed=20260814, samples=10000)
    return {
        "method": method,
        "dataset": "ETTh2",
        "expert_set": "+".join(EXPECTED_EXPERTS),
        "test_mae": test_summary["mae"],
        "test_mse": test_summary["mse"],
        "validation_mae": float(val_mae.mean()),
        "validation_mse": float(val_mse.mean()),
        "diff_vs_validation": test_summary["mae"] - float(val_mae.mean()),
        "diff_vs_single_DLinear_test": test_summary["mae"] - SINGLE_DLINEAR_TEST_MAE,
        "diff_vs_ETTh2_full_adaptive_test": test_summary["mae"] - FINAL_ADAPTIVE_TEST_MAE,
        "diff_vs_validation_selected_DLinear_ModernTCN_test": test_summary["mae"] - BEST_FIXED2_TEST_MAE,
        "selection_protocol": "router-train fitted pair-potential linear ensemble; validation reported before this after-final-test audit; no test-data feedback",
        "status": "after_final_test_audit",
        "weights_json": json.dumps({name: float(weights[i]) for i, name in enumerate(EXPECTED_EXPERTS)}),
        "paired_ci95_diff_vs_DLinear_low": boot["ci95_low"],
        "paired_ci95_diff_vs_DLinear_high": boot["ci95_high"],
        "paired_ci_excludes_zero": boot["ci_excludes_zero"],
    }


def append_unique_csv(path: Path, rows: Sequence[Mapping[str, Any]], key_fields: Sequence[str]) -> None:
    if not path.exists():
        return
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        existing = list(reader)
        fields = list(reader.fieldnames or [])
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    incoming_keys = {tuple(str(row.get(field, "")) for field in key_fields) for row in rows}
    kept = [row for row in existing if tuple(str(row.get(field, "")) for field in key_fields) not in incoming_keys]
    merged = kept + [dict(row) for row in rows]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(merged)


def write_report(rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]) -> None:
    lines = [
        "# ETTh2 Pair-Potential Linear Ensemble Test Audit",
        "",
        "This is an after-final-test audit of two ETTh2 methods that previously existed only as router-train-fitted validation rows.",
        "No hyperparameters or weights were changed after loading test.",
        "",
        f"Created UTC: `{manifest['created_at_utc']}`",
        f"Git commit: `{manifest['git_commit']}`",
        "",
        "| Method | Test MAE | Test MSE | Val MAE | Diff vs DLinear test | Diff vs full adaptive test |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda r: float(r["test_mae"])):
        lines.append(
            f"| {row['method']} | {row['test_mae']:.6f} | {row['test_mse']:.6f} | "
            f"{row['validation_mae']:.6f} | {row['diff_vs_single_DLinear_test']:+.6f} | "
            f"{row['diff_vs_ETTh2_full_adaptive_test']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Leakage Checks",
            "",
            "- Weights were fitted from `cache/costarts_fresh/ETTh2_96_12/router_train_cache.pt` only.",
            "- Validation was used only to reproduce the existing pair-potential validation numbers.",
            "- The ETTh2 locked test cache was loaded only after `manifest_before_test.json` was written.",
            "- No test result was used to change weights, hyperparameters, or method membership.",
            "",
            "## Reproduce",
            "",
            "```powershell",
            "python experiments/etth2_pair_potential_test_evaluation/run_etth2_pair_potential_test_evaluation.py",
            "```",
        ]
    )
    (OUT_DIR / "ETTH2_PAIR_POTENTIAL_TEST_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    train_cache = load_cache(TRAIN_CACHE, "router_train", allow_test=False)
    val_cache = load_cache(VAL_CACHE, "router_val", allow_test=False)
    weights = {
        "nonnegative_simplex_linear_average": fit_simplex_weights(train_cache),
        "ridge_linear_stacker": fit_ridge_weights(train_cache),
    }
    for method, weight in weights.items():
        validate_weight_vector(method, weight, summary)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "status": "after_final_test_audit",
        "label": "pair_potential_linear_ensembles_after_final_test_audit",
        "test_loaded_before_manifest": False,
        "test_metrics_seen_before_this_method_run": True,
        "selection_complete_before_test_load": True,
        "no_tuning": True,
        "train_cache": str(TRAIN_CACHE),
        "train_cache_sha256": sha256_file(TRAIN_CACHE),
        "validation_cache": str(VAL_CACHE),
        "validation_cache_sha256": sha256_file(VAL_CACHE),
        "test_cache": str(TEST_CACHE),
        "test_cache_sha256_preload": sha256_file(TEST_CACHE),
        "source_validation_summary": str(SUMMARY_PATH),
        "methods": {
            method: {
                "weights": {name: float(weight[i]) for i, name in enumerate(EXPECTED_EXPERTS)},
                "fit_source": "ETTh2 router_train only",
                "validation_reference": VALIDATION_REFS[method],
            }
            for method, weight in weights.items()
        },
        "device": "cpu",
        "gpu_note": "CPU used because these are 5-coefficient cached linear ensemble evaluations; GPU transfer would dominate.",
    }
    write_json(OUT_DIR / "manifest_before_test.json", manifest)

    test_cache = load_cache(TEST_CACHE, "locked_test", allow_test=True)
    dlinear_mae, _ = weighted_errors(
        test_cache,
        torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32),
    )
    rows = [result_row(method, weight, val_cache, test_cache, dlinear_mae) for method, weight in weights.items()]
    rows = sorted(rows, key=lambda row: float(row["test_mae"]))
    elapsed = time.perf_counter() - started
    payload = {
        **manifest,
        "test_evaluation_complete": True,
        "elapsed_seconds": elapsed,
        "results": rows,
    }
    write_csv(OUT_DIR / "test_results.csv", rows)
    write_json(OUT_DIR / "ETTH2_PAIR_POTENTIAL_TEST_RESULTS.json", payload)
    write_report(rows, manifest)

    top_rows = []
    all_rows = []
    for row in rows:
        top_rows.append(
            {
                "method": row["method"],
                "expert_set": row["expert_set"],
                "freeze_status": row["status"],
                "test_mae": row["test_mae"],
                "test_mse": row["test_mse"],
                "validation_mae": row["validation_mae"],
                "validation_mse": row["validation_mse"],
                "mae_diff_vs_validation": row["diff_vs_validation"],
                "mae_diff_vs_DLinear_test": row["diff_vs_single_DLinear_test"],
                "selection_protocol": row["selection_protocol"],
                "source": str(OUT_DIR / "test_results.csv"),
            }
        )
        all_rows.append(
            {
                "dataset": "ETTh2",
                "method": row["method"],
                "expert_set": row["expert_set"],
                "test_mae": row["test_mae"],
                "test_mse": row["test_mse"],
                "validation_mae": row["validation_mae"],
                "validation_mse": row["validation_mse"],
                "split": "test",
                "status": row["status"],
                "result_group": "etth2_pair_potential_test_evaluation",
                "selection_protocol": row["selection_protocol"],
                "source_file": str(OUT_DIR / "test_results.csv"),
                "comparison_anchor": "ETTh2 full adaptive test",
                "diff_vs_anchor": row["diff_vs_ETTh2_full_adaptive_test"],
                "diff_vs_validation": row["diff_vs_validation"],
                "source": "pair_potential_after_final_test_audit",
            }
        )
    append_unique_csv(TOP_RESULTS, top_rows, ("method",))
    append_unique_csv(ALL_RESULTS, all_rows, ("dataset", "method", "result_group", "split"))
    print(json.dumps({"test_evaluation_complete": True, "elapsed_seconds": elapsed, "results": rows}, indent=2))


if __name__ == "__main__":
    main()
