"""Post-hoc frozen test audit for the published-baseline comparison suite.

ETTh1/ETTh2 test metrics were viewed before this audit.  This script therefore
does not produce a clean final-test claim; it only evaluates already-selected
published-baseline configurations and records provenance.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.frozen_costar import run_frozen_costar_validation as costar_eval  # noqa: E402
from experiments.published_baseline_comparisons import run_published_baselines as base  # noqa: E402


OUT_DIR = ROOT / "experiments" / "published_baseline_test_audit"
SOURCE_DIR = ROOT / "experiments" / "published_baseline_comparisons"
FINAL_TEST_RESULTS = ROOT / "experiments" / "final_test_evaluation" / "FINAL_TEST_RESULTS.json"
ETTH1_RESIDUAL_RESULTS = ROOT / "experiments" / "frozen_model_test_results" / "top_costar_test_results.csv"
ETTH2_RESIDUAL_RESULTS = ROOT / "experiments" / "etth2_validation_tuned_missing_methods" / "test_results.csv"
TEST_CACHES = {
    "ETTh1": ROOT / "experiments" / "final_test_evaluation" / "generated" / "caches" / "ETTh1" / "test_80_100_cache.pt",
    "ETTh2": ROOT / "experiments" / "final_test_evaluation" / "generated" / "caches" / "ETTh2" / "locked_test_cache_v2.pt",
}
EXPECTED_TEST_ROLES = {"ETTh1": "test_80_100", "ETTh2": "locked_test"}
METHODS = (
    "Equal fixed ensemble",
    "Granger-Ramanathan",
    "Bates-Granger",
    "FAME adaptation",
    "TimeRouter adaptation",
    "Frozen COSTAR",
    "Online COSTAR",
    "Frozen COSTAR + Ridge residual",
    "Frozen COSTAR + MLP residual",
    "OneNet / adaptation",
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def selected_configs() -> dict[str, Any]:
    configs = {}
    for dataset in ("ETTh1", "ETTh2"):
        path = SOURCE_DIR / dataset / "frozen_config_before_validation.json"
        payload = load_json(path)
        if payload.get("test_cache_loaded") is not False:
            raise RuntimeError(f"{path} does not certify test_cache_loaded=false")
        configs[dataset] = payload
    return configs


def validation_lookup(report: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, float]]:
    out: dict[tuple[str, str], dict[str, float]] = {}
    for row in report["comparison_table"]:
        method = row["Method"]
        for dataset in ("ETTh1", "ETTh2"):
            out[(dataset, method)] = {
                "validation_mae": float(row[f"{dataset} Val MAE"]),
                "validation_mse": float(row[f"{dataset} Val MSE"]),
            }
    return out


def best_single_anchors() -> dict[str, dict[str, Any]]:
    payload = load_json(FINAL_TEST_RESULTS)
    anchors: dict[str, dict[str, Any]] = {}
    for row in payload["results"]:
        if row["Method"] == "Best single expert":
            anchors[row["Dataset"]] = {
                "method": row["Method"],
                "expert_set": row["Expert set"],
                "test_mae": float(row["Test MAE"]),
                "test_mse": float(row["Test MSE"]),
            }
    return anchors


def load_test_cache(dataset: str) -> dict[str, Any]:
    path = TEST_CACHES[dataset]
    cache = torch.load(path, map_location="cpu", weights_only=False)
    base.validate_cache(cache, EXPECTED_TEST_ROLES[dataset], dataset)
    return cache


def strip_metric_tensors(row: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if k not in {"per_window_mae", "per_window_mse"}}


def result_row(
    dataset: str,
    method: str,
    pred: torch.Tensor,
    cache: Mapping[str, Any],
    std: torch.Tensor,
    val: Mapping[str, float],
    *,
    expert_set: str,
    selection_protocol: str,
    selected_config: Any = "",
    uses_realized_test_feedback: bool = False,
    provenance: str = "freshly_evaluated_in_audit",
    extra: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics = base.metric_tensors(cache, pred, std)
    row = {
        "dataset": dataset,
        "method": method,
        "expert_set": expert_set,
        "test_mae": metrics["mae"],
        "test_mse": metrics["mse"],
        "validation_mae": val["validation_mae"],
        "validation_mse": val["validation_mse"],
        "test_minus_validation_mae": metrics["mae"] - val["validation_mae"],
        "test_minus_validation_mse": metrics["mse"] - val["validation_mse"],
        "selection_protocol": selection_protocol,
        "selected_config": json.dumps(selected_config, sort_keys=True) if isinstance(selected_config, (dict, list)) else selected_config,
        "uses_realized_test_feedback": uses_realized_test_feedback,
        "causal_feedback_rule": "old_forecast_start + forecast_horizon <= current_forecast_start" if uses_realized_test_feedback else "",
        "audit_label": "post_hoc_comparative_audit",
        "provenance": provenance,
        "num_windows": int(cache["num_windows"]),
    }
    if extra:
        row.update(extra)
    per_window = base.per_window_rows(dataset, method, cache, pred, std)
    return row, per_window


def config_for(frozen: Mapping[str, Any], method: str) -> dict[str, Any]:
    configs = frozen["selected_configs"]
    key = "OneNet-style frozen-expert adaptation" if method == "OneNet / adaptation" else method
    return dict(configs[key]["config"])


def evaluate_dataset(
    dataset: str,
    train_cache: Mapping[str, Any],
    test_cache: Mapping[str, Any],
    std: torch.Tensor,
    frozen: Mapping[str, Any],
    val_lookup: Mapping[tuple[str, str], Mapping[str, float]],
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    per_window: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}

    def add(method: str, pred: torch.Tensor, **kwargs: Any) -> None:
        row, pwin = result_row(dataset, method, pred, test_cache, std, val_lookup[(dataset, method)], **kwargs)
        rows.append(row)
        per_window.extend(pwin)

    add(
        "Equal fixed ensemble",
        base.fixed_average_prediction(test_cache, base.EXPERTS),
        expert_set="+".join(base.EXPERTS),
        selection_protocol="equal average over fixed published-baseline expert pool; no test selection",
        uses_realized_test_feedback=False,
    )

    gr_cfg = config_for(frozen, "Granger-Ramanathan")
    gr_model = base.fit_gr(train_cache, std, base.LinearConfig(**gr_cfg))
    add(
        "Granger-Ramanathan",
        base.predict_gr(gr_model, test_cache),
        expert_set="+".join(base.EXPERTS),
        selected_config=gr_cfg,
        selection_protocol="configuration selected on router_train folds in published-baseline validation suite; fit on all router_train; test once",
    )

    bates_cfg = config_for(frozen, "Bates-Granger")
    bates_model = base.fit_bates(train_cache, std, base.BatesConfig(**bates_cfg))
    add(
        "Bates-Granger",
        base.predict_bates(bates_model, test_cache),
        expert_set="+".join(base.EXPERTS),
        selected_config=bates_cfg,
        selection_protocol="configuration selected on router_train folds in published-baseline validation suite; fit on all router_train; test once",
    )

    fame_cfg = config_for(frozen, "FAME adaptation")
    fame_model = base.train_fame_model(train_cache, std, base.FameConfig(**fame_cfg), device)
    fame_pred = base.predict_fame(fame_model, test_cache, std)
    add(
        "FAME adaptation",
        fame_pred,
        expert_set="+".join(base.EXPERTS),
        selected_config=fame_cfg,
        selection_protocol="configuration selected on router_train folds in published-baseline validation suite; trained on all router_train; test once",
        extra=base.fame_diagnostics(fame_model, test_cache),
    )

    tr_cfg = config_for(frozen, "TimeRouter adaptation")
    tr_model = base.train_timerouter_model(train_cache, std, base.TimeRouterConfig(**tr_cfg), device)
    tr_pred = base.predict_timerouter(tr_model, test_cache)
    add(
        "TimeRouter adaptation",
        tr_pred,
        expert_set="+".join(base.EXPERTS),
        selected_config=tr_cfg,
        selection_protocol="configuration selected on router_train folds in published-baseline validation suite; trained on all router_train; test once",
        extra=base.timerouter_diagnostics(tr_model, test_cache),
    )

    onenet_cfg = config_for(frozen, "OneNet / adaptation")
    onenet_model = base.fit_onenet(train_cache, std, base.OneNetConfig(**onenet_cfg))
    onenet_pred, onenet_extra = base.onenet_predict(
        test_cache,
        std,
        base.OneNetConfig(**onenet_cfg),
        onenet_model["init_error"],
        onenet_model["branches"],
    )
    add(
        "OneNet / adaptation",
        onenet_pred,
        expert_set="+".join(onenet_model["branches"]),
        selected_config=onenet_cfg,
        selection_protocol="configuration selected on router_train folds; initialized on all router_train; causal test feedback only after target observability",
        uses_realized_test_feedback=True,
        extra=onenet_extra,
    )
    diagnostics["OneNet / adaptation"] = {"num_updates": onenet_extra.get("num_updates")}

    core = base.ETTH1_COSTAR_CORE if dataset == "ETTh1" else base.ETTH2_COSTAR_CORE
    expert_idx = costar_eval.etth1.expert_indices(test_cache, core) if dataset == "ETTh1" else costar_eval.etth2.expert_indices(test_cache, core)
    seeds = list(costar_eval.SEEDS) if dataset == "ETTh1" else [7]
    frozen_preds = []
    online_preds = []
    frozen_extras = []
    online_extras = []
    for seed in seeds:
        fp, fextra = costar_eval.frozen_costar_prediction(dataset, test_cache, train_cache, std, expert_idx, int(seed), device)
        op, oextra = costar_eval.online_prediction(dataset, test_cache, train_cache, std, expert_idx, int(seed), device)
        frozen_preds.append(fp)
        online_preds.append(op)
        frozen_extras.append(fextra)
        online_extras.append(oextra)
    frozen_mean = torch.stack(frozen_preds).mean(dim=0)
    online_mean = torch.stack(online_preds).mean(dim=0)
    add(
        "Frozen COSTAR",
        frozen_mean,
        expert_set="+".join(core),
        selected_config={"core": list(core), "source": "experiments/frozen_costar/run_frozen_costar_validation.py", "seeds": [int(s) for s in seeds]},
        selection_protocol="selected COSTAR core and hyperparameters frozen before this audit; no realized test feedback",
        extra={
            "seeds": ",".join(str(int(s)) for s in seeds),
            **{f"last_extra_{k}": v for k, v in frozen_extras[-1].items() if not isinstance(v, (dict, list))},
        },
    )
    add(
        "Online COSTAR",
        online_mean,
        expert_set="+".join(core),
        selected_config={"core": list(core), "source": "experiments/frozen_costar/run_frozen_costar_validation.py", "seeds": [int(s) for s in seeds]},
        selection_protocol="selected COSTAR core and hyperparameters frozen before this audit; causal test feedback only after target observability",
        uses_realized_test_feedback=True,
        extra={
            "seeds": ",".join(str(int(s)) for s in seeds),
            **{f"last_extra_{k}": v for k, v in online_extras[-1].items() if not isinstance(v, (dict, list))},
        },
    )
    diagnostics["Online COSTAR"] = {"last_extra": online_extras[-1]}
    return rows, per_window, diagnostics


def residual_source_row(dataset: str, method: str) -> dict[str, Any]:
    if dataset == "ETTh1":
        source = ETTH1_RESIDUAL_RESULTS
        lookup = {"Frozen COSTAR + Ridge residual": "ridge_residual_corrector", "Frozen COSTAR + MLP residual": "mlp_residual_corrector"}
        rows = read_csv(source)
        match = [r for r in rows if r.get("method") == lookup[method]]
        if not match:
            raise RuntimeError(f"Missing {dataset} {method} in {source}")
        row = match[0]
        return {
            "test_mae": float(row["test_mae"]),
            "test_mse": float(row["test_mse"]),
            "source_path": str(source),
            "seeds": row.get("seeds", "5"),
            "source_method": row.get("method"),
        }
    source = ETTH2_RESIDUAL_RESULTS
    lookup = {"Frozen COSTAR + Ridge residual": "Ridge residual corrector", "Frozen COSTAR + MLP residual": "MLP residual corrector"}
    rows = read_csv(source)
    match = [r for r in rows if r.get("method") == lookup[method] and r.get("seed") == "mean"]
    if not match:
        raise RuntimeError(f"Missing {dataset} {method} in {source}")
    row = match[0]
    return {
        "test_mae": float(row["mae"]),
        "test_mse": float(row["mse"]),
        "source_path": str(source),
        "seeds": row.get("seed", "mean"),
        "source_method": row.get("method"),
    }


def append_residual_rows(
    rows: list[dict[str, Any]],
    val_lookup_map: Mapping[tuple[str, str], Mapping[str, float]],
) -> None:
    for dataset in ("ETTh1", "ETTh2"):
        for method in ("Frozen COSTAR + Ridge residual", "Frozen COSTAR + MLP residual"):
            src = residual_source_row(dataset, method)
            val = val_lookup_map[(dataset, method)]
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "expert_set": "PatchTST+iTransformer+TimesNet" if dataset == "ETTh1" else "DLinear+PatchTST+ModernTCN",
                    "test_mae": src["test_mae"],
                    "test_mse": src["test_mse"],
                    "validation_mae": val["validation_mae"],
                    "validation_mse": val["validation_mse"],
                    "test_minus_validation_mae": src["test_mae"] - val["validation_mae"],
                    "test_minus_validation_mse": src["test_mse"] - val["validation_mse"],
                    "selection_protocol": "carried from existing frozen residual test artifact selected without test-data feedback",
                    "selected_config": "",
                    "uses_realized_test_feedback": False,
                    "causal_feedback_rule": "",
                    "audit_label": "post_hoc_comparative_audit",
                    "provenance": "existing_frozen_residual_artifact",
                    "source_path": src["source_path"],
                    "source_method": src["source_method"],
                    "seeds": src["seeds"],
                    "num_windows": 2773,
                    "per_window_metrics_available_in_this_audit": False,
                }
            )


def add_deltas_and_ranks(rows: list[dict[str, Any]], anchors: Mapping[str, Mapping[str, Any]]) -> None:
    by_dataset: dict[str, list[dict[str, Any]]] = {"ETTh1": [], "ETTh2": []}
    for row in rows:
        by_dataset[row["dataset"]].append(row)
    for dataset, drows in by_dataset.items():
        equal = next(r for r in drows if r["method"] == "Equal fixed ensemble")
        online = next(r for r in drows if r["method"] == "Online COSTAR")
        single = anchors[dataset]
        ranked = sorted(drows, key=lambda r: (float(r["test_mae"]), float(r["test_mse"]), r["method"]))
        for rank, row in enumerate(ranked, 1):
            row["rank_by_test_mae"] = rank
            row["delta_mae_vs_equal_fixed_ensemble"] = float(row["test_mae"]) - float(equal["test_mae"])
            row["delta_mse_vs_equal_fixed_ensemble"] = float(row["test_mse"]) - float(equal["test_mse"])
            row["delta_mae_vs_best_single_expert"] = float(row["test_mae"]) - float(single["test_mae"])
            row["delta_mse_vs_best_single_expert"] = float(row["test_mse"]) - float(single["test_mse"])
            row["best_single_expert"] = single["expert_set"]
            row["delta_mae_vs_online_costar"] = float(row["test_mae"]) - float(online["test_mae"])
            row["delta_mse_vs_online_costar"] = float(row["test_mse"]) - float(online["test_mse"])


def comparison_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for method in METHODS:
        etth1 = next(r for r in rows if r["dataset"] == "ETTh1" and r["method"] == method)
        etth2 = next(r for r in rows if r["dataset"] == "ETTh2" and r["method"] == method)
        out.append(
            {
                "Method": method,
                "ETTh1 Test MAE": etth1["test_mae"],
                "ETTh1 Test MSE": etth1["test_mse"],
                "ETTh1 Val MAE": etth1["validation_mae"],
                "ETTh1 Val MSE": etth1["validation_mse"],
                "ETTh2 Test MAE": etth2["test_mae"],
                "ETTh2 Test MSE": etth2["test_mse"],
                "ETTh2 Val MAE": etth2["validation_mae"],
                "ETTh2 Val MSE": etth2["validation_mse"],
            }
        )
    return out


def write_report(rows: Sequence[Mapping[str, Any]], comparison: Sequence[Mapping[str, Any]], payload: Mapping[str, Any]) -> None:
    def fmt(x: Any) -> str:
        return f"{float(x):.6f}"

    lines = [
        "# Published Baseline Test Audit",
        "",
        "Label: `post_hoc_comparative_audit`.",
        "",
        "ETTh1 and ETTh2 test results had already been viewed before this run. This report is a frozen test-only comparative audit, not a clean untouched final-test claim.",
        "",
        "All listed methods use configurations already selected from router-train/validation artifacts. No method was changed, selected, or tuned using these test results.",
        "",
        "## Main Table",
        "",
        "| Method | ETTh1 Test MAE | ETTh1 Test MSE | ETTh1 Val MAE | ETTh2 Test MAE | ETTh2 Test MSE | ETTh2 Val MAE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison:
        lines.append(
            f"| {row['Method']} | {fmt(row['ETTh1 Test MAE'])} | {fmt(row['ETTh1 Test MSE'])} | {fmt(row['ETTh1 Val MAE'])} | {fmt(row['ETTh2 Test MAE'])} | {fmt(row['ETTh2 Test MSE'])} | {fmt(row['ETTh2 Val MAE'])} |"
        )
    for dataset in ("ETTh1", "ETTh2"):
        ranked = sorted([r for r in rows if r["dataset"] == dataset], key=lambda r: int(r["rank_by_test_mae"]))
        lines.extend(["", f"## {dataset} Ranking", "", "| Rank | Method | Test MAE | Delta vs Online COSTAR | Delta vs Equal fixed |", "|---:|---|---:|---:|---:|"])
        for row in ranked:
            lines.append(
                f"| {row['rank_by_test_mae']} | {row['method']} | {fmt(row['test_mae'])} | {float(row['delta_mae_vs_online_costar']):+.6f} | {float(row['delta_mae_vs_equal_fixed_ensemble']):+.6f} |"
            )
    lines.extend(
        [
            "",
            "## Causality And Provenance",
            "",
            "- Online COSTAR and OneNet use realized feedback only after `old_forecast_start + forecast_horizon <= current_forecast_start`.",
            "- Frozen COSTAR, Equal fixed, Granger-Ramanathan, Bates-Granger, FAME, TimeRouter, Ridge residual, and MLP residual do not use realized test feedback in the predictions reported here.",
            "- Ridge/MLP residual rows are carried from existing frozen residual artifacts referenced in the CSV/JSON outputs; they were not reselected during this audit.",
            f"- Git commit: `{payload['git_commit']}`.",
        ]
    )
    (OUT_DIR / "PUBLISHED_BASELINE_TEST_AUDIT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    frozen = selected_configs()
    published_report = load_json(SOURCE_DIR / "FINAL_REPORT.json")
    val_map = validation_lookup(published_report)
    anchors = best_single_anchors()
    pre_manifest = {
        "label": "post_hoc_comparative_audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(SOURCE_DIR),
        "test_results_previously_viewed": True,
        "test_cache_loaded_before_manifest": False,
        "no_parameter_selected_from_test": True,
        "frozen_config_paths": {d: str(SOURCE_DIR / d / "frozen_config_before_validation.json") for d in ("ETTh1", "ETTh2")},
        "selected_configs": frozen,
        "planned_methods": list(METHODS),
        "device_requested": args.device,
        "device_used": str(device),
        "command": "python experiments\\published_baseline_test_audit\\run_published_baseline_test_audit.py --device cuda",
    }
    write_json(OUT_DIR / "pre_test_audit_manifest.json", pre_manifest)
    print(json.dumps({"frozen_configuration_before_test_load": pre_manifest}, indent=2, sort_keys=True))

    train1, _val1, std1, hashes1 = base.load_dataset("ETTh1")
    train2, _val2, std2, hashes2 = base.load_dataset("ETTh2")
    test1 = load_test_cache("ETTh1")
    test2 = load_test_cache("ETTh2")

    all_rows: list[dict[str, Any]] = []
    all_per_window: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    for dataset, train_cache, test_cache, std in (("ETTh1", train1, test1, std1), ("ETTh2", train2, test2, std2)):
        rows, per_window, diag = evaluate_dataset(dataset, train_cache, test_cache, std, frozen[dataset], val_map, device)
        all_rows.extend(rows)
        all_per_window.extend(per_window)
        diagnostics[dataset] = diag
    append_residual_rows(all_rows, val_map)
    add_deltas_and_ranks(all_rows, anchors)
    comparison = comparison_rows(all_rows)

    cache_hashes = {
        "ETTh1": {**hashes1, "test_sha256": base.sha256_file(TEST_CACHES["ETTh1"])},
        "ETTh2": {**hashes2, "test_sha256": base.sha256_file(TEST_CACHES["ETTh2"])},
    }
    payload = {
        **pre_manifest,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": base.git_commit(),
        "runtime_sec": time.perf_counter() - started,
        "test_cache_loaded_after_manifest": True,
        "test_evaluation_complete": True,
        "test_caches": {k: str(v) for k, v in TEST_CACHES.items()},
        "cache_hashes": cache_hashes,
        "best_single_anchors": anchors,
        "causality_assertions": {
            "online_costar_rule": "old_forecast_start + forecast_horizon <= current_forecast_start",
            "onenet_rule": "old_forecast_start + forecast_horizon <= current_forecast_start",
            "test_feedback_initializes_earlier_predictions": False,
            "test_hyperparameter_selection": False,
        },
        "diagnostics": diagnostics,
        "comparison_table": comparison,
        "results": all_rows,
    }
    write_csv(OUT_DIR / "published_baseline_test_results.csv", all_rows)
    write_csv(OUT_DIR / "published_baseline_test_comparison_table.csv", comparison)
    write_csv(OUT_DIR / "published_baseline_test_per_window_metrics.csv", all_per_window)
    write_json(OUT_DIR / "PUBLISHED_BASELINE_TEST_AUDIT_RESULTS.json", payload)
    write_report(all_rows, comparison, payload)
    print(json.dumps({"results": comparison, "output_dir": str(OUT_DIR)}, indent=2))


if __name__ == "__main__":
    main()
