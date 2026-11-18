"""Frozen-model ETTh2 test results for top available COSTAR/fixed methods.

Every model evaluated here was trained, selected, configured, and frozen without
test-data feedback. These are additional frozen-model test evaluations, distinct
from the original confirmatory freeze in experiments/final_test_evaluation/.
"""

from __future__ import annotations

import csv
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

from experiments.chronological_adaptive_costar.run_chronological_adaptive_costar import paired_bootstrap  # noqa: E402
from experiments.etth2_train_selected_core.run_etth2_train_selected_core_eval import (  # noqa: E402
    full_model_prediction as fixed3_full_model_prediction,
)
from experiments.etth2_train_selected_variable_core.run_etth2_train_selected_variable_core_eval import (  # noqa: E402
    full_model_prediction as variable_full_model_prediction,
)
from experiments.oracle_weight_tournament.run_tournament import sample_mae, sample_mse  # noqa: E402


OUT_DIR = ROOT / "experiments" / "frozen_model_test_results"
TEST_CACHE = ROOT / "experiments" / "final_test_evaluation" / "generated" / "caches" / "ETTh2" / "locked_test_cache_v2.pt"
TRAIN_CACHE = ROOT / "cache" / "costarts_fresh" / "ETTh2_96_12" / "router_train_cache.pt"
FINAL_TEST_RESULTS = ROOT / "experiments" / "final_test_evaluation" / "FINAL_TEST_RESULTS.json"

VALIDATION_REFS = {
    "DLinear": {"mae": 0.28095653653144836, "mse": 0.17149297893047333},
    "DLinear+ModernTCN": {"mae": 0.2752290368080139, "mse": 0.1653451770544052},
    "DLinear+TimesNet": {"mae": 0.27765217423439026, "mse": 0.1678023487329483},
    "DLinear+TimesNet+ModernTCN": {"mae": 0.27664363384246826, "mse": 0.16693221032619476},
    "DLinear+PatchTST+ModernTCN": {"mae": 0.2808783948421478, "mse": 0.17193281650543213},
    "full_fixed3_train_selected": {"mae": 0.27683213353157043, "mse": 0.16727977991104126},
    "full_variable_core_DLinear": {"mae": 0.2804695665836334, "mse": 0.1709725558757782},
}


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_cache(path: Path, role: str) -> dict[str, Any]:
    cache = torch.load(path, map_location="cpu", weights_only=False)
    actual = cache.get("cache_role", cache.get("split_role"))
    if actual != role:
        raise ValueError(f"{path}: role={actual!r}, expected {role!r}")
    return cache


def expert_indices(cache: Mapping[str, Any], experts: Sequence[str]) -> list[int]:
    names = list(cache["expert_names"])
    return [names.index(name) for name in experts]


def average_prediction(cache: Mapping[str, Any], experts: Sequence[str]) -> torch.Tensor:
    return cache["prediction_stack"][..., expert_indices(cache, experts)].to(torch.float32).mean(dim=-1)


def metrics(cache: Mapping[str, Any], pred: torch.Tensor, std: torch.Tensor) -> dict[str, Any]:
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.bool)
    mae = sample_mae(pred, target, mask, std)
    mse = sample_mse(pred, target, mask, std)
    return {"mae": float(mae.mean()), "mse": float(mse.mean()), "per_window_mae": mae, "per_window_mse": mse}


def row(
    method: str,
    experts: Sequence[str],
    pred: torch.Tensor,
    cache: Mapping[str, Any],
    std: torch.Tensor,
    anchor_mae: torch.Tensor,
    validation_key: str,
    protocol: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    met = metrics(cache, pred, std)
    ref = VALIDATION_REFS[validation_key]
    boot = paired_bootstrap(met["per_window_mae"], anchor_mae, seed=20260813, samples=10000)
    return {
        "method": method,
        "expert_set": "+".join(experts),
        "freeze_status": "pre_test_frozen",
        "test_mae": met["mae"],
        "test_mse": met["mse"],
        "validation_mae": ref["mae"],
        "validation_mse": ref["mse"],
        "mae_diff_vs_validation": met["mae"] - ref["mae"],
        "mae_diff_vs_DLinear_test": met["mae"] - float(anchor_mae.mean()),
        "selection_protocol": protocol,
        "paired_ci95_diff_vs_DLinear_low": boot["ci95_low"],
        "paired_ci95_diff_vs_DLinear_high": boot["ci95_high"],
        "paired_ci_excludes_zero": boot["ci_excludes_zero"],
        **dict(extra or {}),
    }


def write_report(rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Frozen-Model ETTh2 Top COSTAR Test Results",
        "",
        "Every listed model was trained, selected, configured, and frozen without test-data feedback.",
        "",
        "These are frozen-model test results. The original confirmatory evaluation remains the formally frozen evaluation in `experiments/final_test_evaluation/`; the other rows are additional frozen-model evaluations performed later.",
        "",
        "| Method | Expert set | Test MAE | Test MSE | Val MAE | Diff vs DLinear test | Protocol |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            "| {method} | {expert_set} | {test_mae:.6f} | {test_mse:.6f} | {validation_mae:.6f} | {mae_diff_vs_DLinear_test:+.6f} | {selection_protocol} |".format(
                **r
            )
        )
    best = min(rows, key=lambda r: float(r["test_mae"]))
    lines.extend(
        [
            "",
            f"Best frozen-model ETTh2 test MAE: `{best['method']}` at `{best['test_mae']:.6f}`.",
            "",
            "The clean final ETTh2 result remains the preregistered train-selected full frozen adaptive model from `experiments/final_test_evaluation/`.",
        ]
    )
    (OUT_DIR / "FROZEN_MODEL_ETTH2_TOP_COSTAR_TEST_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    test_cache = load_cache(TEST_CACHE, "locked_test")
    train_cache = load_cache(TRAIN_CACHE, "router_train")
    std = torch.ones(int(test_cache["num_features"]), dtype=torch.float32)
    if int(test_cache["num_windows"]) != 2773:
        raise ValueError("Unexpected ETTh2 test window count")
    if not FINAL_TEST_RESULTS.exists():
        raise FileNotFoundError(FINAL_TEST_RESULTS)

    dlinear_pred = average_prediction(test_cache, ("DLinear",))
    dlinear_mae = metrics(test_cache, dlinear_pred, std)["per_window_mae"]
    rows: list[dict[str, Any]] = []
    rows.append(row("single_DLinear", ("DLinear",), dlinear_pred, test_cache, std, dlinear_mae, "DLinear", "canonical validation-best single anchor"))

    fixed_methods = [
        ("fixed2_DLinear_ModernTCN_validation_selected_reference", ("DLinear", "ModernTCN"), "DLinear+ModernTCN", "validation-selected reference only"),
        ("fixed2_DLinear_TimesNet_reference", ("DLinear", "TimesNet"), "DLinear+TimesNet", "validation-ranked fixed reference"),
        ("fixed3_DLinear_TimesNet_ModernTCN_validation_selected_reference", ("DLinear", "TimesNet", "ModernTCN"), "DLinear+TimesNet+ModernTCN", "validation-selected fixed-3 reference only"),
        ("fixed3_DLinear_PatchTST_ModernTCN_train_selected", ("DLinear", "PatchTST", "ModernTCN"), "DLinear+PatchTST+ModernTCN", "router-train selected fixed-3 core"),
    ]
    for method, experts, val_key, protocol in fixed_methods:
        rows.append(row(method, experts, average_prediction(test_cache, experts), test_cache, std, dlinear_mae, val_key, protocol))

    fixed3_idx = expert_indices(test_cache, ("DLinear", "PatchTST", "ModernTCN"))
    full3_pred, full3_extra = fixed3_full_model_prediction(test_cache, train_cache, fixed3_idx, std)
    rows.append(
        row(
            "full_adaptive_train_selected_fixed3_final_frozen",
            ("DLinear", "PatchTST", "ModernTCN"),
            full3_pred,
            test_cache,
            std,
            dlinear_mae,
            "full_fixed3_train_selected",
            "preregistered final ETTh2 model; train-selected fixed-3 core",
            full3_extra,
        )
    )

    dlinear_idx = expert_indices(test_cache, ("DLinear",))
    full1_pred, full1_extra = variable_full_model_prediction(test_cache, train_cache, dlinear_idx, std)
    rows.append(
        row(
            "full_adaptive_variable_size_core_DLinear",
            ("DLinear",),
            full1_pred,
            test_cache,
            std,
            dlinear_mae,
            "full_variable_core_DLinear",
            "router-train variable-size selected single core; frozen-model test audit",
            full1_extra,
        )
    )

    rows = sorted(rows, key=lambda r: float(r["test_mae"]))
    payload = {
        "freeze_status": "pre_test_frozen",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "test_cache": str(TEST_CACHE),
        "router_train_cache": str(TRAIN_CACHE),
        "final_test_results_source": str(FINAL_TEST_RESULTS),
        "metric": "ETTh2 canonical cache scale; sample_mae/sample_mse; std=ones; no inverse transform",
        "results": rows,
    }
    write_csv(OUT_DIR / "etth2_top_costar_test_results.csv", rows)
    write_json(OUT_DIR / "ETTH2_TOP_COSTAR_TEST_RESULTS.json", payload)
    write_report(rows)
    print(json.dumps({"freeze_status": payload["freeze_status"], "results": rows}, indent=2))


if __name__ == "__main__":
    main()
