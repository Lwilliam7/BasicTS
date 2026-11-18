"""Evaluate an ETTh1 nonnegative simplex all-five linear ensemble on test.

This is an after-final-test audit. The method is the ETTh1 analogue of the
ETTh2 pair-potential nonnegative simplex linear average: fit one convex weight
vector on router-train only, reproduce validation once, then evaluate the
already-authorized ETTh1 test cache once.
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
from experiments.oracle_weight_tournament.run_tournament import load_std, sample_mae, sample_mse  # noqa: E402
from scripts.costars.analyze_etth2_pair_potential import fit_simplex_weights, sha256_file  # noqa: E402


OUT_DIR = ROOT / "experiments" / "etth1_simplex_linear_test_evaluation"
TRAIN_CACHE = ROOT / "cache" / "costarts_walkforward" / "router_train_20_60_cache.pt"
VAL_CACHE = ROOT / "cache" / "costarts_walkforward" / "router_val_60_80_cache.pt"
TEST_CACHE = ROOT / "experiments" / "final_test_evaluation" / "generated" / "caches" / "ETTh1" / "test_80_100_cache.pt"
NORMALIZER = ROOT / "checkpoints" / "costarts_walkforward" / "final_60" / "DLinear" / "best_expert.pt"
ALL_RESULTS = ROOT / "experiments" / "all_results_summary" / "all_costar_results.csv"
TOP_RESULTS = ROOT / "experiments" / "frozen_model_test_results" / "top_costar_test_results.csv"

EXPECTED_EXPERTS = ("DLinear", "PatchTST", "iTransformer", "TimesNet", "ModernTCN")
FIXED_CORE_TEST_MAE = 0.3271281123161316
FINAL_ADAPTIVE_TEST_MAE = 0.3263952910900116
BEST_SINGLE_TEST_MAE = 0.3390795886516571


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
        "diff_vs_best_single_test",
        "diff_vs_fixed_core_test",
        "diff_vs_full_adaptive_test",
        "selection_protocol",
        "status",
        "weights_json",
        "paired_ci95_diff_vs_fixed_core_low",
        "paired_ci95_diff_vs_fixed_core_high",
        "paired_ci_excludes_zero",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


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
    incoming = {tuple(str(row.get(field, "")) for field in key_fields) for row in rows}
    kept = [row for row in existing if tuple(str(row.get(field, "")) for field in key_fields) not in incoming]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(kept + [dict(row) for row in rows])


def load_cache(path: Path, role: str, *, allow_test: bool) -> dict[str, Any]:
    if not allow_test and "test" in str(path).lower():
        raise AssertionError(f"Refusing to load test path before manifest: {path}")
    cache = torch.load(path, map_location="cpu", weights_only=False)
    actual = cache.get("cache_role", cache.get("split_role"))
    if actual != role:
        raise ValueError(f"{path}: role {actual!r} != expected {role!r}")
    if tuple(cache["expert_names"]) != EXPECTED_EXPERTS:
        raise ValueError("ETTh1 expert order changed")
    starts = cache["absolute_window_starts"].to(torch.long)
    if role == "router_train_20_60":
        if int(cache["num_windows"]) != 5546 or int(starts.min()) != 2880 or int(starts.max()) != 8532:
            raise AssertionError("Unexpected ETTh1 router_train split")
    elif role == "router_val_60_80":
        if int(cache["num_windows"]) != 2773 or int(starts.min()) != 8640 or int(starts.max()) != 11412:
            raise AssertionError("Unexpected ETTh1 router_val split")
    elif role == "test_80_100":
        if int(cache["num_windows"]) != 2773 or int(starts.min()) != 11520 or int(starts.max()) != 14292:
            raise AssertionError("Unexpected ETTh1 test split")
    return cache


def weighted_prediction(cache: Mapping[str, Any], weights: torch.Tensor) -> torch.Tensor:
    return torch.tensordot(
        cache["prediction_stack"].to(torch.float32),
        weights.to(torch.float32),
        dims=([-1], [0]),
    )


def metrics(cache: Mapping[str, Any], pred: torch.Tensor, std: torch.Tensor) -> dict[str, Any]:
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.bool)
    mae = sample_mae(pred, target, mask, std)
    mse = sample_mse(pred, target, mask, std)
    return {
        "mae": float(mae.mean()),
        "mse": float(mse.mean()),
        "per_window_mae": mae.detach().cpu(),
        "per_window_mse": mse.detach().cpu(),
    }


def fixed_core_per_window_mae(cache: Mapping[str, Any], std: torch.Tensor) -> torch.Tensor:
    names = list(cache["expert_names"])
    idx = [names.index(name) for name in ("PatchTST", "iTransformer", "TimesNet")]
    pred = cache["prediction_stack"][..., idx].to(torch.float32).mean(dim=-1)
    return metrics(cache, pred, std)["per_window_mae"]


def write_report(result: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    lines = [
        "# ETTh1 Simplex Linear Ensemble Test Audit",
        "",
        "This is an after-final-test audit of the ETTh1 analogue of the ETTh2 nonnegative simplex linear average.",
        "Weights were fit once from ETTh1 router-train only; no test feedback was used to change them.",
        "",
        f"Created UTC: `{manifest['created_at_utc']}`",
        f"Git commit: `{manifest['git_commit']}`",
        "",
        "| Method | Test MAE | Test MSE | Val MAE | Diff vs fixed core test | Diff vs full adaptive test |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| {result['method']} | {result['test_mae']:.6f} | {result['test_mse']:.6f} | "
            f"{result['validation_mae']:.6f} | {result['diff_vs_fixed_core_test']:+.6f} | "
            f"{result['diff_vs_full_adaptive_test']:+.6f} |"
        ),
        "",
        "## Weights",
        "",
        f"`{result['weights_json']}`",
        "",
        "## Leakage Checks",
        "",
        "- Weights were fit from `cache/costarts_walkforward/router_train_20_60_cache.pt` only.",
        "- Router validation was used only for reporting the frozen validation metric.",
        "- The ETTh1 test cache was loaded only after `manifest_before_test.json` was written.",
        "- No model or hyperparameter was changed after seeing the test result.",
        "",
        "## Reproduce",
        "",
        "```powershell",
        "python experiments/etth1_simplex_linear_test_evaluation/run_etth1_simplex_linear_test_evaluation.py",
        "```",
    ]
    (OUT_DIR / "ETTH1_SIMPLEX_LINEAR_TEST_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    start = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_cache = load_cache(TRAIN_CACHE, "router_train_20_60", allow_test=False)
    val_cache = load_cache(VAL_CACHE, "router_val_60_80", allow_test=False)
    std = load_std(NORMALIZER, int(val_cache["num_features"]))
    weights = fit_simplex_weights(train_cache)
    if bool((weights < -1e-7).any()) or abs(float(weights.sum()) - 1.0) > 2e-5:
        raise AssertionError("Simplex constraints failed")

    val_pred = weighted_prediction(val_cache, weights)
    val_metrics = metrics(val_cache, val_pred, std)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "status": "after_final_test_audit",
        "label": "etth1_nonnegative_simplex_linear_average_after_final_test_audit",
        "test_loaded_before_manifest": False,
        "test_metrics_seen_before_this_method_run": True,
        "selection_complete_before_test_load": True,
        "no_tuning": True,
        "train_cache": str(TRAIN_CACHE),
        "train_cache_sha256": sha256_file(TRAIN_CACHE),
        "validation_cache": str(VAL_CACHE),
        "validation_cache_sha256": sha256_file(VAL_CACHE),
        "test_cache": str(TEST_CACHE),
        "normalizer": str(NORMALIZER),
        "normalizer_sha256": sha256_file(NORMALIZER),
        "fit_source": "ETTh1 router_train_20_60 only",
        "method": "nonnegative_simplex_linear_average",
        "weights": {name: float(weights[i]) for i, name in enumerate(EXPECTED_EXPERTS)},
        "validation_before_test_load": {"mae": val_metrics["mae"], "mse": val_metrics["mse"]},
        "device": "cpu",
        "gpu_note": "CPU used because this is a 5-coefficient cached linear ensemble evaluation; GPU transfer would dominate.",
    }
    write_json(OUT_DIR / "manifest_before_test.json", manifest)

    test_cache = load_cache(TEST_CACHE, "test_80_100", allow_test=True)
    if int(test_cache["num_features"]) != int(val_cache["num_features"]):
        raise AssertionError("Feature count mismatch")
    test_pred = weighted_prediction(test_cache, weights)
    test_metrics = metrics(test_cache, test_pred, std)
    fixed_core_mae = fixed_core_per_window_mae(test_cache, std)
    boot = paired_bootstrap(test_metrics["per_window_mae"], fixed_core_mae, seed=20260814, samples=10000)
    result = {
        "method": "nonnegative_simplex_linear_average",
        "dataset": "ETTh1",
        "expert_set": "+".join(EXPECTED_EXPERTS),
        "test_mae": test_metrics["mae"],
        "test_mse": test_metrics["mse"],
        "validation_mae": val_metrics["mae"],
        "validation_mse": val_metrics["mse"],
        "diff_vs_validation": test_metrics["mae"] - val_metrics["mae"],
        "diff_vs_best_single_test": test_metrics["mae"] - BEST_SINGLE_TEST_MAE,
        "diff_vs_fixed_core_test": test_metrics["mae"] - FIXED_CORE_TEST_MAE,
        "diff_vs_full_adaptive_test": test_metrics["mae"] - FINAL_ADAPTIVE_TEST_MAE,
        "selection_protocol": "router-train fitted nonnegative simplex all-five linear ensemble; no test-data feedback",
        "status": "after_final_test_audit",
        "weights_json": json.dumps({name: float(weights[i]) for i, name in enumerate(EXPECTED_EXPERTS)}),
        "paired_ci95_diff_vs_fixed_core_low": boot["ci95_low"],
        "paired_ci95_diff_vs_fixed_core_high": boot["ci95_high"],
        "paired_ci_excludes_zero": boot["ci_excludes_zero"],
    }
    payload = {
        **manifest,
        "test_evaluation_complete": True,
        "elapsed_seconds": time.perf_counter() - start,
        "results": [result],
    }
    write_csv(OUT_DIR / "test_results.csv", [result])
    write_json(OUT_DIR / "ETTH1_SIMPLEX_LINEAR_TEST_RESULTS.json", payload)
    write_report(result, manifest)

    top_row = {
        "method": result["method"],
        "freeze_status": result["status"],
        "seeds": 0,
        "test_mae": result["test_mae"],
        "test_mse": result["test_mse"],
        "validation_mae": result["validation_mae"],
        "validation_mse": result["validation_mse"],
        "mae_diff_vs_validation": result["diff_vs_validation"],
        "mae_diff_vs_test_fixed_core": result["diff_vs_fixed_core_test"],
        "paired_ci95_diff_vs_fixed_core_low": result["paired_ci95_diff_vs_fixed_core_low"],
        "paired_ci95_diff_vs_fixed_core_high": result["paired_ci95_diff_vs_fixed_core_high"],
        "paired_ci_excludes_zero": result["paired_ci_excludes_zero"],
        "selection_protocol": result["selection_protocol"],
        "source": str(OUT_DIR / "test_results.csv"),
        "weights_json": result["weights_json"],
    }
    all_row = {
        "dataset": "ETTh1",
        "method": result["method"],
        "expert_set": result["expert_set"],
        "test_mae": result["test_mae"],
        "test_mse": result["test_mse"],
        "validation_mae": result["validation_mae"],
        "validation_mse": result["validation_mse"],
        "split": "test",
        "status": result["status"],
        "result_group": "etth1_simplex_linear_test_evaluation",
        "selection_protocol": result["selection_protocol"],
        "source_file": str(OUT_DIR / "test_results.csv"),
        "comparison_anchor": "ETTh1 full adaptive test",
        "diff_vs_anchor": result["diff_vs_full_adaptive_test"],
        "diff_vs_validation": result["diff_vs_validation"],
        "source": "etth1_simplex_after_final_test_audit",
    }
    append_unique_csv(TOP_RESULTS, [top_row], ("method",))
    append_unique_csv(ALL_RESULTS, [all_row], ("dataset", "method", "result_group", "split"))
    print(json.dumps({"test_evaluation_complete": True, "elapsed_seconds": payload["elapsed_seconds"], "result": result}, indent=2))


if __name__ == "__main__":
    main()
