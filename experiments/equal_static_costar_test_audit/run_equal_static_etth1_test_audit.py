"""After-final-test audit for ETTh1 equal-static full adaptive COSTAR.

This script evaluates the active equal-static ETTh1 full adaptive path on the
already-generated ETTh1 final-test cache.  It does not tune or change any
configuration after loading test.  Because the original final test results were
already seen before this cleanup, the result is labeled as an after-final-test
audit and not as a preregistered final result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.chronological_adaptive_costar.run_chronological_adaptive_costar import SEEDS  # noqa: E402
from experiments.oracle_weight_tournament.run_tournament import load_cache, load_std  # noqa: E402
from experiments.train_selected_core_etth1.run_train_selected_core_eval import (  # noqa: E402
    evaluate_expanded,
    expert_indices,
    metrics,
    selected_forecasts,
)


OUT_DIR = ROOT / "experiments/equal_static_costar_test_audit"
TRAIN_CACHE = ROOT / "cache/costarts_walkforward/router_train_20_60_cache.pt"
TEST_CACHE = ROOT / "experiments/final_test_evaluation/generated/caches/ETTh1/test_80_100_cache.pt"
NORMALIZER = ROOT / "checkpoints/costarts_walkforward/final_60/DLinear/best_expert.pt"
EQUAL_STATIC_CONFIG = ROOT / "experiments/train_selected_core_etth1_equal_static/frozen_config_before_validation.json"
EQUAL_STATIC_VAL = ROOT / "experiments/train_selected_core_etth1_equal_static/final_report.json"
FINAL_TEST_RESULTS = ROOT / "experiments/final_test_evaluation/FINAL_TEST_RESULTS.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def load_test_cache(path: Path) -> dict[str, Any]:
    cache = torch.load(path, map_location="cpu", weights_only=False)
    role = cache.get("cache_role", cache.get("split_role"))
    if role != "test_80_100":
        raise ValueError(f"Expected ETTh1 test_80_100 cache, got {role!r}")
    starts = cache["absolute_window_starts"].to(torch.long)
    if int(cache["num_windows"]) != 2773:
        raise ValueError(f"Unexpected ETTh1 test window count: {cache['num_windows']}")
    if int(starts.min()) != 11520 or int(starts.max()) != 14292:
        raise ValueError(f"Unexpected ETTh1 test starts: {int(starts.min())}..{int(starts.max())}")
    if int(cache["forecast_horizon"]) != 12:
        raise ValueError("Unexpected forecast horizon")
    if int(cache["num_features"]) != 7:
        raise ValueError("Unexpected variable count")
    return cache


def fixed_core_prediction(cache: Mapping[str, Any], idx: Sequence[int]) -> torch.Tensor:
    return selected_forecasts(cache, idx).mean(dim=-1)


def final_result_row(method: str) -> dict[str, Any]:
    data = json.loads(FINAL_TEST_RESULTS.read_text(encoding="utf-8"))
    for row in data["results"]:
        if row["Dataset"] == "ETTh1" and row["Method"] == method:
            return row
    raise KeyError(method)


def make_report(payload: Mapping[str, Any]) -> None:
    result = payload["result"]
    fixed = payload["anchors"]["fixed_core"]
    old = payload["anchors"]["old_preregistered_full_adaptive"]
    lines = [
        "# ETTh1 Equal-Static COSTAR Test Audit",
        "",
        "This is an after-final-test audit. The original final test set had already been evaluated before the equal-static cleanup.",
        "No tuning, expert changes, or hyperparameter changes were made after loading the test cache.",
        "",
        "## Result",
        "",
        "| Method | Test MAE | Test MSE | Validation MAE | Validation MSE |",
        "|---|---:|---:|---:|---:|",
        f"| Equal-static full adaptive COSTAR | `{result['test_mae']:.6f}` | `{result['test_mse']:.6f}` | `{result['validation_mae']:.6f}` | `{result['validation_mse']:.6f}` |",
        f"| Train-selected fixed core | `{fixed['test_mae']:.6f}` | `{fixed['test_mse']:.6f}` | `{fixed['validation_mae']:.6f}` | `{fixed['validation_mse']:.6f}` |",
        f"| Old preregistered full adaptive reference | `{old['test_mae']:.6f}` | `{old['test_mse']:.6f}` | `{old['validation_mae']:.6f}` | `{old['validation_mse']:.6f}` |",
        "",
        "## Differences",
        "",
        f"- Difference vs fixed core test MAE: `{result['diff_vs_fixed_core_test_mae']:+.6f}`.",
        f"- Difference vs old preregistered full adaptive test MAE: `{result['diff_vs_old_full_adaptive_test_mae']:+.6f}`.",
        f"- Difference vs equal-static validation MAE: `{result['diff_vs_validation_mae']:+.6f}`.",
        "",
        "## Protocol",
        "",
        "- Dataset: `ETTh1`.",
        "- Split: test `80-100%`, starts `11520..14292`, `2773` windows.",
        "- Core: `PatchTST+iTransformer+TimesNet`.",
        "- Static prior: equal `1/3` for every selected triple.",
        "- Online updates: causal, using `old_start + horizon <= current_start`.",
        "- Label: `after_final_test_audit`.",
        "",
        "## Reproduce",
        "",
        "```powershell",
        payload["command"],
        "```",
    ]
    (OUT_DIR / "EQUAL_STATIC_ETTH1_TEST_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    start = time.time()
    device = torch.device(args.device)

    config = json.loads(EQUAL_STATIC_CONFIG.read_text(encoding="utf-8"))
    val_report = json.loads(EQUAL_STATIC_VAL.read_text(encoding="utf-8"))
    core = [str(x) for x in config["selected_three_experts"]]
    if core != ["PatchTST", "iTransformer", "TimesNet"]:
        raise ValueError(f"Unexpected equal-static core: {core}")

    train_cache = load_cache(TRAIN_CACHE, "router_train_20_60")
    test_cache = load_test_cache(TEST_CACHE)
    std = load_std(NORMALIZER, int(test_cache["num_features"]))
    idx = expert_indices(test_cache, core)

    fixed_pred = fixed_core_prediction(test_cache, idx)
    fixed_met = metrics(test_cache, std, fixed_pred)
    per_seed = []
    preds = []
    for seed in SEEDS:
        pred, extra = evaluate_expanded(test_cache, train_cache, std, idx, int(seed), device)
        met = metrics(test_cache, std, pred)
        per_seed.append({"seed": int(seed), "mae": met["mae"], "mse": met["mse"], **extra})
        preds.append(pred)
    mean_pred = torch.stack(preds).mean(dim=0)
    met = metrics(test_cache, std, mean_pred)

    old_full = final_result_row("Full frozen adaptive model")
    fixed_row = final_result_row("Train-selected fixed core")
    validation = val_report["train_selected_current_best_model"]
    result = {
        "dataset": "ETTh1",
        "method": "equal_static_full_adaptive_costar",
        "label": "after_final_test_audit",
        "expert_set": "+".join(core + ["DLinear", "ModernTCN"]),
        "test_mae": met["mae"],
        "test_mse": met["mse"],
        "validation_mae": float(validation["mae"]),
        "validation_mse": float(validation["mse"]),
        "diff_vs_validation_mae": met["mae"] - float(validation["mae"]),
        "diff_vs_fixed_core_test_mae": met["mae"] - fixed_met["mae"],
        "diff_vs_old_full_adaptive_test_mae": met["mae"] - float(old_full["Test MAE"]),
        "seed_mae_mean": float(torch.tensor([r["mae"] for r in per_seed]).mean()),
        "seed_mae_std": float(torch.tensor([r["mae"] for r in per_seed]).std(unbiased=False)),
        "seed_mse_mean": float(torch.tensor([r["mse"] for r in per_seed]).mean()),
        "seed_mse_std": float(torch.tensor([r["mse"] for r in per_seed]).std(unbiased=False)),
        "seeds": ",".join(str(s) for s in SEEDS),
    }
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": f"python experiments\\equal_static_costar_test_audit\\run_equal_static_etth1_test_audit.py --device {args.device}",
        "device": str(device),
        "test_loaded": True,
        "test_evaluated": True,
        "label": "after_final_test_audit",
        "no_tuning_after_test_load": True,
        "test_metrics_seen_before_this_audit": True,
        "result": result,
        "anchors": {
            "fixed_core": {
                "test_mae": fixed_met["mae"],
                "test_mse": fixed_met["mse"],
                "validation_mae": float(fixed_row["Validation MAE"]),
                "validation_mse": float(fixed_row["Validation MSE"]),
            },
            "old_preregistered_full_adaptive": {
                "test_mae": float(old_full["Test MAE"]),
                "test_mse": float(old_full["Test MSE"]),
                "validation_mae": float(old_full["Validation MAE"]),
                "validation_mse": float(old_full["Validation MSE"]),
            },
        },
        "per_seed": per_seed,
        "artifacts": {
            "train_cache": str(TRAIN_CACHE),
            "train_cache_sha256": sha256_file(TRAIN_CACHE),
            "test_cache": str(TEST_CACHE),
            "test_cache_sha256": sha256_file(TEST_CACHE),
            "normalizer": str(NORMALIZER),
            "normalizer_sha256": sha256_file(NORMALIZER),
            "equal_static_config": str(EQUAL_STATIC_CONFIG),
            "equal_static_validation_report": str(EQUAL_STATIC_VAL),
        },
        "runtime_sec": time.time() - start,
    }
    write_json(OUT_DIR / "EQUAL_STATIC_ETTH1_TEST_AUDIT.json", payload)
    write_csv(OUT_DIR / "equal_static_etth1_test_results.csv", [result])
    write_csv(OUT_DIR / "equal_static_etth1_test_per_seed.csv", per_seed)
    make_report(payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
