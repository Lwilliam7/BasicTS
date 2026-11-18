"""After-final-test audit of published-baseline methods not yet evaluated on the
canonical ETTh1/ETTh2 test caches.

ETTh1/ETTh2 test metrics have already been viewed in this project (see
`experiments/final_test_evaluation/`). This script therefore does not produce a
clean untouched final-test claim. It evaluates six already-selected
published-baseline configurations on the canonical test caches once, adds
explicit leakage/causality verification (target-replacement invariance for
non-adaptive methods, a future-target perturbation test for the one method
that uses realized-target feedback), and records cache provenance.

No hyperparameter, expert-set, or method choice is selected using these test
results. All six configurations come verbatim from
`experiments/published_baseline_comparisons/{ETTh1,ETTh2}/frozen_config_before_validation.json`,
which was written before validation was ever loaded.

Frozen COSTAR / Online COSTAR are NOT re-tuned or re-selected here; their rows
are read verbatim from existing authoritative artifacts and included only as
reference rows.
"""

from __future__ import annotations

import argparse
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

from experiments.published_baseline_comparisons import run_published_baselines as base  # noqa: E402

OUT_DIR = ROOT / "experiments" / "published_baseline_test_audit"
SOURCE_DIR = ROOT / "experiments" / "published_baseline_comparisons"
FINAL_TEST_RESULTS = ROOT / "experiments" / "final_test_evaluation" / "FINAL_TEST_RESULTS.json"
PUBLISHED_TEST_AUDIT_RESULTS = OUT_DIR / "PUBLISHED_BASELINE_TEST_AUDIT_RESULTS.json"
SPLIT_PLAN = ROOT / "results" / "router_summary" / "costarts_walkforward" / "split_plan.json"
TEST_CACHES = {
    "ETTh1": ROOT / "experiments" / "final_test_evaluation" / "generated" / "caches" / "ETTh1" / "test_80_100_cache.pt",
    "ETTh2": ROOT / "experiments" / "final_test_evaluation" / "generated" / "caches" / "ETTh2" / "locked_test_cache_v2.pt",
}
EXPECTED_TEST_ROLES = {"ETTh1": "test_80_100", "ETTh2": "locked_test"}

AUDIT_LABEL = "after_final_test_audit"

METHODS = (
    "Equal all-5 ensemble",
    "Granger-Ramanathan",
    "Bates-Granger",
    "FAME adaptation",
    "TimeRouter adaptation",
    "OneNet-style frozen-expert adaptation",
)
METHOD_LABELS_FULL = {
    "Equal all-5 ensemble": "Equal all-5 ensemble",
    "Granger-Ramanathan": "Granger-Ramanathan",
    "Bates-Granger": "Bates-Granger",
    "FAME adaptation": "FAME routing adaptation to BasicTS frozen expert pool",
    "TimeRouter adaptation": "TimeRouter routing-mechanism adaptation",
    "OneNet-style frozen-expert adaptation": "OneNet-style frozen-expert adaptation",
}
REFERENCE_METHODS = (
    "Frozen COSTAR (reference)",
    "Online COSTAR (reference)",
    "COSTAR train-selected fixed core (reference)",
    "Best single expert (reference)",
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_frozen_configs() -> dict[str, Any]:
    configs = {}
    for dataset in ("ETTh1", "ETTh2"):
        path = SOURCE_DIR / dataset / "frozen_config_before_validation.json"
        payload = load_json(path)
        if payload.get("test_cache_loaded") is not False:
            raise RuntimeError(f"{path} does not certify test_cache_loaded=false; refusing to treat as pre-test frozen config")
        configs[dataset] = payload
    return configs


def config_for(frozen: Mapping[str, Any], method: str) -> dict[str, Any]:
    key = "OneNet-style frozen-expert adaptation" if method == "OneNet-style frozen-expert adaptation" else method
    return dict(frozen["selected_configs"][key]["config"])


def load_test_cache(dataset: str) -> dict[str, Any]:
    path = TEST_CACHES[dataset]
    cache = torch.load(path, map_location="cpu", weights_only=False)
    base.validate_cache(cache, EXPECTED_TEST_ROLES[dataset], dataset)
    return cache


def perturbed_target_cache(cache: Mapping[str, Any], seed: int = 12345) -> dict[str, Any]:
    """Copy `cache` with targets replaced by independent random noise (mask unchanged)."""
    gen = torch.Generator().manual_seed(seed)
    out = dict(cache)
    out["targets"] = torch.randn(cache["targets"].shape, generator=gen) * 1000.0
    return out


def perturbed_future_target_cache(cache: Mapping[str, Any], cutoff_idx: int, bump: float = 1.0e6) -> dict[str, Any]:
    """Copy `cache` adding a huge value to targets at window index >= cutoff_idx."""
    out = dict(cache)
    targets = cache["targets"].clone()
    targets[cutoff_idx:] = targets[cutoff_idx:] + bump
    out["targets"] = targets
    return out


def invariance_check(method: str, base_pred: torch.Tensor, perturbed_pred: torch.Tensor) -> dict[str, Any]:
    diff = (base_pred - perturbed_pred).abs()
    max_abs_diff = float(diff.max())
    return {
        "method": method,
        "check": "target_replacement_invariance",
        "description": "Replacing test targets with independent random noise must leave predictions exactly unchanged because this method does not use realized test-target feedback.",
        "max_abs_diff": max_abs_diff,
        "passed": max_abs_diff == 0.0,
    }


def onenet_causality_perturbation_test(
    cache: Mapping[str, Any],
    std: torch.Tensor,
    cfg: "base.OneNetConfig",
    init_error: torch.Tensor,
    branches: Sequence[str],
) -> dict[str, Any]:
    num_windows = int(cache["num_windows"])
    cutoff_idx = num_windows // 2
    starts = cache["absolute_window_starts"].to(torch.long)
    horizon = int(cache["forecast_horizon"])
    cutoff_start = int(starts[cutoff_idx])

    baseline_pred, _ = base.onenet_predict(cache, std, cfg, init_error, branches)
    perturbed_cache = perturbed_future_target_cache(cache, cutoff_idx, bump=1.0e6)
    perturbed_pred, _ = base.onenet_predict(perturbed_cache, std, cfg, init_error, branches)

    per_row_abs_diff = (baseline_pred - perturbed_pred).abs().reshape(num_windows, -1).max(dim=1).values

    # First forecast origin at which the perturbed targets could legally have
    # become observable: smallest i such that starts[cutoff_idx] + horizon <= starts[i].
    legal_idx = None
    for i in range(num_windows):
        if cutoff_start + horizon <= int(starts[i]):
            legal_idx = i
            break
    if legal_idx is None:
        legal_idx = num_windows

    changed_idx = (per_row_abs_diff > 0).nonzero(as_tuple=True)[0]
    first_changed = int(changed_idx[0]) if changed_idx.numel() > 0 else None

    prefix_end = legal_idx  # exclusive: indices [0, legal_idx) must be exactly unchanged
    max_unaffected_prefix_abs_diff = float(per_row_abs_diff[:prefix_end].max()) if prefix_end > 0 else 0.0
    passed = max_unaffected_prefix_abs_diff == 0.0

    return {
        "check": "onenet_future_target_perturbation_causality",
        "description": "Add a huge value to test targets from the cutoff window onward and rerun OneNet prediction. All predictions before the first forecast origin at which the perturbed error could legally become observable must be exactly unchanged.",
        "num_windows": num_windows,
        "cutoff_idx": cutoff_idx,
        "perturbed_from_row": cutoff_idx,
        "cutoff_absolute_start": cutoff_start,
        "forecast_horizon": horizon,
        "first_prediction_allowed_to_change": legal_idx,
        "first_prediction_index_that_actually_changed": first_changed,
        "max_unaffected_prefix_abs_diff": max_unaffected_prefix_abs_diff,
        "max_abs_diff_full_sequence": float(per_row_abs_diff.max()),
        "passed": passed,
    }


def reference_rows() -> dict[str, dict[str, Any]]:
    """Load Frozen COSTAR / Online COSTAR / fixed-core / best-single reference rows
    verbatim from existing authoritative artifacts. Nothing here is re-run or re-tuned."""
    final_test = load_json(FINAL_TEST_RESULTS)
    audit = load_json(PUBLISHED_TEST_AUDIT_RESULTS)

    out: dict[str, dict[str, Any]] = {}
    for row in final_test["results"]:
        dataset = row["Dataset"]
        if row["Method"] == "Best single expert":
            out.setdefault(dataset, {})["Best single expert (reference)"] = {
                "test_mae": float(row["Test MAE"]),
                "test_mse": float(row["Test MSE"]),
                "expert_set": row["Expert set"],
                "source": str(FINAL_TEST_RESULTS.relative_to(ROOT)),
            }
        if row["Method"] == "Train-selected fixed core":
            out.setdefault(dataset, {})["COSTAR train-selected fixed core (reference)"] = {
                "test_mae": float(row["Test MAE"]),
                "test_mse": float(row["Test MSE"]),
                "expert_set": row["Expert set"],
                "source": str(FINAL_TEST_RESULTS.relative_to(ROOT)),
            }
    for row in audit["results"]:
        dataset = row["dataset"]
        if row["method"] == "Frozen COSTAR":
            out.setdefault(dataset, {})["Frozen COSTAR (reference)"] = {
                "test_mae": float(row["test_mae"]),
                "test_mse": float(row["test_mse"]),
                "expert_set": row["expert_set"],
                "source": str(PUBLISHED_TEST_AUDIT_RESULTS.relative_to(ROOT)),
            }
        if row["method"] == "Online COSTAR":
            out.setdefault(dataset, {})["Online COSTAR (reference)"] = {
                "test_mae": float(row["test_mae"]),
                "test_mse": float(row["test_mse"]),
                "expert_set": row["expert_set"],
                "source": str(PUBLISHED_TEST_AUDIT_RESULTS.relative_to(ROOT)),
            }
    return out


VALIDATION_REPORT_METHOD_NAMES = {
    "Equal all-5 ensemble": "Equal fixed ensemble",
    "Granger-Ramanathan": "Granger-Ramanathan",
    "Bates-Granger": "Bates-Granger",
    "FAME adaptation": "FAME adaptation",
    "TimeRouter adaptation": "TimeRouter adaptation",
    "OneNet-style frozen-expert adaptation": "OneNet / adaptation",
}


def load_existing_validation() -> dict[tuple[str, str], dict[str, float]]:
    """Existing validation MAE/MSE (reference-only, not re-derived) keyed by (dataset, method)."""
    report = load_json(SOURCE_DIR / "FINAL_REPORT.json")
    by_report_method: dict[tuple[str, str], dict[str, float]] = {}
    for row in report["comparison_table"]:
        method = row["Method"]
        for dataset in ("ETTh1", "ETTh2"):
            by_report_method[(dataset, method)] = {
                "validation_mae": float(row[f"{dataset} Val MAE"]),
                "validation_mse": float(row[f"{dataset} Val MSE"]),
            }
    out: dict[tuple[str, str], dict[str, float]] = {}
    for our_method, report_method in VALIDATION_REPORT_METHOD_NAMES.items():
        for dataset in ("ETTh1", "ETTh2"):
            out[(dataset, our_method)] = by_report_method[(dataset, report_method)]
    return out


def evaluate_dataset(
    dataset: str,
    train_cache: Mapping[str, Any],
    test_cache: Mapping[str, Any],
    std: torch.Tensor,
    frozen: Mapping[str, Any],
    val_lookup: Mapping[tuple[str, str], Mapping[str, float]],
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    per_window: list[dict[str, Any]] = []
    leakage_checks: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}

    def add(method: str, pred: torch.Tensor, **kwargs: Any) -> None:
        metrics = base.metric_tensors(test_cache, pred, std)
        val = val_lookup[(dataset, method)]
        row = {
            "dataset": dataset,
            "method": method,
            "official_method_label": METHOD_LABELS_FULL[method],
            "expert_set": kwargs.get("expert_set", ""),
            "validation_mae": val["validation_mae"],
            "validation_mse": val["validation_mse"],
            "test_mae": metrics["mae"],
            "test_mse": metrics["mse"],
            "test_minus_validation_mae": metrics["mae"] - val["validation_mae"],
            "test_minus_validation_mse": metrics["mse"] - val["validation_mse"],
            "selected_config": json.dumps(kwargs.get("selected_config", ""), sort_keys=True) if isinstance(kwargs.get("selected_config"), (dict, list)) else kwargs.get("selected_config", ""),
            "uses_realized_test_feedback": kwargs.get("uses_realized_test_feedback", False),
            "causal_feedback_rule": "old_forecast_start + forecast_horizon <= current_forecast_start" if kwargs.get("uses_realized_test_feedback", False) else "",
            "fit_protocol": kwargs.get("fit_protocol", ""),
            "audit_label": AUDIT_LABEL,
            "num_windows": int(test_cache["num_windows"]),
        }
        rows.append(row)
        starts = test_cache["absolute_window_starts"].to(torch.long).tolist()
        for i in range(len(starts)):
            per_window.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "window_index": i,
                    "absolute_window_start": int(starts[i]),
                    "mae": float(metrics["per_window_mae"][i]),
                    "mse": float(metrics["per_window_mse"][i]),
                }
            )

    # 1. Equal all-5 ensemble: no training, no target feedback of any kind.
    add(
        "Equal all-5 ensemble",
        base.fixed_average_prediction(test_cache, base.EXPERTS),
        expert_set="+".join(base.EXPERTS),
        fit_protocol="no training; mean of five frozen expert forecasts; no validation- or test-target feedback",
        uses_realized_test_feedback=False,
    )

    # 2. Granger-Ramanathan: fit (closed-form ridge) on router_train only, freeze, predict.
    gr_cfg = config_for(frozen, "Granger-Ramanathan")
    gr_model = base.fit_gr(train_cache, std, base.LinearConfig(**gr_cfg))
    gr_pred = base.predict_gr(gr_model, test_cache)
    add(
        "Granger-Ramanathan",
        gr_pred,
        expert_set="+".join(base.EXPERTS),
        selected_config=gr_cfg,
        fit_protocol="closed-form ridge-extension fit on router_train only using frozen validation-selected config; coefficients frozen before test",
        uses_realized_test_feedback=False,
    )
    gr_perturbed = base.predict_gr(gr_model, perturbed_target_cache(test_cache))
    leakage_checks.append(invariance_check(f"{dataset} Granger-Ramanathan", gr_pred, gr_perturbed))

    # 3. Bates-Granger: fit (closed-form inverse-error/covariance) on router_train only, freeze, predict.
    bates_cfg = config_for(frozen, "Bates-Granger")
    bates_model = base.fit_bates(train_cache, std, base.BatesConfig(**bates_cfg))
    bates_pred = base.predict_bates(bates_model, test_cache)
    add(
        "Bates-Granger",
        bates_pred,
        expert_set="+".join(base.EXPERTS),
        selected_config=bates_cfg,
        fit_protocol="closed-form inverse-error/covariance combination fit on router_train only using frozen validation-selected config; weights frozen before test",
        uses_realized_test_feedback=False,
    )
    bates_perturbed = base.predict_bates(bates_model, perturbed_target_cache(test_cache))
    leakage_checks.append(invariance_check(f"{dataset} Bates-Granger", bates_pred, bates_perturbed))

    # 4. FAME adaptation: train MLP router on router_train only with frozen config+seed, freeze, predict.
    fame_cfg = config_for(frozen, "FAME adaptation")
    fame_model = base.train_fame_model(train_cache, std, base.FameConfig(**fame_cfg), device)
    fame_pred = base.predict_fame(fame_model, test_cache, std)
    add(
        "FAME adaptation",
        fame_pred,
        expert_set="+".join(base.EXPERTS),
        selected_config=fame_cfg,
        fit_protocol="MLP router trained on router_train only with frozen validation-selected config/seed; weights frozen before test; top-r sparse combination at test time",
        uses_realized_test_feedback=False,
    )
    fame_perturbed = base.predict_fame(fame_model, perturbed_target_cache(test_cache), std)
    leakage_checks.append(invariance_check(f"{dataset} FAME adaptation", fame_pred, fame_perturbed))

    # 5. TimeRouter adaptation: train MLP router on router_train only with frozen config+seed, freeze, predict.
    tr_cfg = config_for(frozen, "TimeRouter adaptation")
    tr_model = base.train_timerouter_model(train_cache, std, base.TimeRouterConfig(**tr_cfg), device)
    tr_pred = base.predict_timerouter(tr_model, test_cache)
    add(
        "TimeRouter adaptation",
        tr_pred,
        expert_set="+".join(base.EXPERTS),
        selected_config=tr_cfg,
        fit_protocol="MLP router trained on router_train only with frozen validation-selected config/seed; weights and fallback frozen before test; no threshold/fallback tuning on test",
        uses_realized_test_feedback=False,
    )
    tr_perturbed = base.predict_timerouter(tr_model, perturbed_target_cache(test_cache))
    leakage_checks.append(invariance_check(f"{dataset} TimeRouter adaptation", tr_pred, tr_perturbed))

    # 6. OneNet-style frozen-expert adaptation: state initialized from router_train only,
    #    then strictly causal online updates during test (realized-target feedback, but only
    #    after the forecast horizon has elapsed).
    onenet_cfg = config_for(frozen, "OneNet-style frozen-expert adaptation")
    onenet_model = base.fit_onenet(train_cache, std, base.OneNetConfig(**onenet_cfg))
    onenet_pred, onenet_extra = base.onenet_predict(
        test_cache, std, base.OneNetConfig(**onenet_cfg), onenet_model["init_error"], onenet_model["branches"]
    )
    add(
        "OneNet-style frozen-expert adaptation",
        onenet_pred,
        expert_set="+".join(onenet_model["branches"]),
        selected_config=onenet_cfg,
        fit_protocol="combination state initialized from causal per-branch mean-abs-error over all of router_train only; state then updated online during test strictly after old_forecast_start + forecast_horizon <= current_forecast_start; underlying experts frozen throughout",
        uses_realized_test_feedback=True,
    )
    diagnostics["OneNet-style frozen-expert adaptation"] = onenet_extra
    causality_result = onenet_causality_perturbation_test(
        test_cache, std, base.OneNetConfig(**onenet_cfg), onenet_model["init_error"], onenet_model["branches"]
    )
    causality_result["dataset"] = dataset
    leakage_checks.append(causality_result)

    return rows, per_window, leakage_checks, diagnostics


def add_deltas_and_ranks(
    rows: list[dict[str, Any]],
    reference: Mapping[str, Mapping[str, Any]],
) -> None:
    by_dataset: dict[str, list[dict[str, Any]]] = {"ETTh1": [], "ETTh2": []}
    for row in rows:
        by_dataset[row["dataset"]].append(row)
    for dataset, drows in by_dataset.items():
        equal = next(r for r in drows if r["method"] == "Equal all-5 ensemble")
        ref = reference[dataset]
        fixed_core = ref["COSTAR train-selected fixed core (reference)"]
        online_costar = ref["Online COSTAR (reference)"]
        ranked = sorted(drows, key=lambda r: (float(r["test_mae"]), float(r["test_mse"]), r["method"]))
        for rank, row in enumerate(ranked, 1):
            row["rank_by_test_mae_among_six_methods"] = rank
            for label, anchor_mae in (
                ("equal_all5_ensemble", float(equal["test_mae"])),
                ("costar_fixed_core", float(fixed_core["test_mae"])),
                ("online_costar", float(online_costar["test_mae"])),
            ):
                row[f"delta_mae_vs_{label}"] = float(row["test_mae"]) - anchor_mae
                row[f"pct_mae_improvement_vs_{label}"] = 100.0 * (anchor_mae - float(row["test_mae"])) / anchor_mae


def full_ranking(rows: list[dict[str, Any]], reference: Mapping[str, Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Rank the six audited methods together with the reference rows, per dataset."""
    out: dict[str, list[dict[str, Any]]] = {}
    for dataset in ("ETTh1", "ETTh2"):
        entries = []
        for row in rows:
            if row["dataset"] != dataset:
                continue
            entries.append({"method": row["method"], "test_mae": row["test_mae"], "test_mse": row["test_mse"], "kind": "audited"})
        for name, r in reference[dataset].items():
            entries.append({"method": name, "test_mae": r["test_mae"], "test_mse": r["test_mse"], "kind": "reference"})
        entries.sort(key=lambda r: (float(r["test_mae"]), float(r["test_mse"]), r["method"]))
        for i, e in enumerate(entries, 1):
            e["rank"] = i
        out[dataset] = entries
    return out


def write_report(
    rows: list[dict[str, Any]],
    reference: Mapping[str, Mapping[str, Any]],
    rankings: Mapping[str, list[dict[str, Any]]],
    leakage: Mapping[str, Any],
    provenance: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> str:
    def fmt(x: Any) -> str:
        return f"{float(x):.6f}"

    lines = [
        "# After-Final-Test Audit: Six Previously Untested Published-Baseline Methods",
        "",
        f"Label: `{AUDIT_LABEL}`.",
        "",
        "ETTh1 and ETTh2 test results were already seen elsewhere in this project before this audit ran. This is not a clean untouched final-test claim; it is a frozen, no-further-tuning evaluation of six methods that had validation results but no prior test evaluation.",
        "",
        "All six configurations are read verbatim from `experiments/published_baseline_comparisons/{ETTh1,ETTh2}/frozen_config_before_validation.json`, written before validation was ever loaded. No hyperparameter, expert-set, or method choice was changed after loading test. Frozen COSTAR and Online COSTAR were **not** re-tuned or re-selected; their rows are reference rows read verbatim from existing authoritative artifacts.",
        "",
        "## Main Results Table",
        "",
        "| Method | ETTh1 Val MAE | ETTh1 Test MAE | ETTh1 Test MSE | ETTh2 Val MAE | ETTh2 Test MAE | ETTh2 Test MSE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    by_method: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_method.setdefault(row["method"], {})[row["dataset"]] = row
    for method in METHODS:
        e1 = by_method[method]["ETTh1"]
        e2 = by_method[method]["ETTh2"]
        lines.append(
            f"| {METHOD_LABELS_FULL[method]} | {fmt(e1['validation_mae'])} | {fmt(e1['test_mae'])} | {fmt(e1['test_mse'])} | {fmt(e2['validation_mae'])} | {fmt(e2['test_mae'])} | {fmt(e2['test_mse'])} |"
        )
    lines.append("")
    lines.append("Reference rows (existing, not re-run):")
    lines.append("")
    lines.append("| Method | ETTh1 Test MAE | ETTh1 Test MSE | ETTh2 Test MAE | ETTh2 Test MSE |")
    lines.append("|---|---:|---:|---:|---:|")
    for name in REFERENCE_METHODS:
        r1 = reference["ETTh1"][name]
        r2 = reference["ETTh2"][name]
        lines.append(f"| {name} | {fmt(r1['test_mae'])} | {fmt(r1['test_mse'])} | {fmt(r2['test_mae'])} | {fmt(r2['test_mse'])} |")

    for dataset in ("ETTh1", "ETTh2"):
        lines.extend(["", f"## {dataset} Full Ranking (audited methods + reference rows)", "", "| Rank | Method | Test MAE | Test MSE | Kind |", "|---:|---|---:|---:|---|"])
        for e in rankings[dataset]:
            lines.append(f"| {e['rank']} | {e['method']} | {fmt(e['test_mae'])} | {fmt(e['test_mse'])} | {e['kind']} |")

    lines.extend(
        [
            "",
            "## Leakage And Causality Checks",
            "",
            "| Check | Passed | Max abs diff |",
            "|---|---|---:|",
        ]
    )
    for c in leakage["invariance_checks"]:
        lines.append(f"| {c['method']} target-replacement invariance | {c['passed']} | {c['max_abs_diff']:.10f} |")
    for c in leakage["onenet_causality_perturbation_tests"]:
        lines.append(
            f"| {c['dataset']} OneNet future-target perturbation causality | {c['passed']} | {c['max_unaffected_prefix_abs_diff']:.10f} |"
        )
    lines.extend(
        [
            "",
            "## Cache Provenance",
            "",
            f"- ETTh1 test cache: `{provenance['ETTh1']['path']}`, sha256 `{provenance['ETTh1']['sha256']}`.",
            f"- ETTh2 test cache: `{provenance['ETTh2']['path']}`, sha256 `{provenance['ETTh2']['sha256']}`.",
            "- Both caches: expert order `DLinear, PatchTST, iTransformer, TimesNet, ModernTCN`; horizon `12`; input length `96`; `2773` chronological windows; target masks all-observed at forecast time.",
            f"- Git commit: `{payload['git_commit']}`.",
        ]
    )
    text = "\n".join(lines)
    (OUT_DIR / "TEST_REPORT.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    frozen = load_frozen_configs()
    val_lookup = load_existing_validation()
    reference = reference_rows()

    train1, _val1, std1, hashes1 = base.load_dataset("ETTh1")
    train2, _val2, std2, hashes2 = base.load_dataset("ETTh2")
    test1 = load_test_cache("ETTh1")
    test2 = load_test_cache("ETTh2")

    manifest = {
        "label": AUDIT_LABEL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Evaluate the six published-baseline methods that had validation results but no test evaluation on the canonical ETTh1/ETTh2 test caches, using configurations frozen before validation.",
        "frozen_config_paths": {d: str((SOURCE_DIR / d / "frozen_config_before_validation.json").relative_to(ROOT)) for d in ("ETTh1", "ETTh2")},
        "selected_configs": {d: frozen[d]["selected_configs"] for d in ("ETTh1", "ETTh2")},
        "methods_evaluated": list(METHODS),
        "reference_only_methods": ["Frozen COSTAR", "Online COSTAR"],
        "test_results_previously_viewed_elsewhere_in_project": True,
        "no_parameter_selected_from_these_test_results": True,
        "device_requested": args.device,
        "device_used": str(device),
    }

    all_rows: list[dict[str, Any]] = []
    all_per_window: list[dict[str, Any]] = []
    all_leakage: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    for dataset, train_cache, test_cache, std in (("ETTh1", train1, test1, std1), ("ETTh2", train2, test2, std2)):
        rows, per_window, leakage, diag = evaluate_dataset(dataset, train_cache, test_cache, std, frozen[dataset], val_lookup, device)
        all_rows.extend(rows)
        all_per_window.extend(per_window)
        all_leakage.extend(leakage)
        diagnostics[dataset] = diag

    add_deltas_and_ranks(all_rows, reference)
    rankings = full_ranking(all_rows, reference)

    invariance_checks = [c for c in all_leakage if c.get("check") == "target_replacement_invariance"]
    causality_tests = [c for c in all_leakage if c.get("check") == "onenet_future_target_perturbation_causality"]
    leakage_payload = {
        "label": AUDIT_LABEL,
        "general_causal_feedback_rule": "old_forecast_start + forecast_horizon <= current_forecast_start",
        "onenet_initialization": "Combination state (per-branch causal mean-abs-error) is initialized once from all of router_train before test begins; the underlying frozen experts PatchTST and iTransformer never update; only the OneNet combination weights update online during test, strictly after the forecast horizon of the contributing window has elapsed.",
        "invariance_checks": invariance_checks,
        "onenet_causality_perturbation_tests": causality_tests,
        "all_checks_passed": all(c["passed"] for c in invariance_checks) and all(c["passed"] for c in causality_tests),
    }

    cache_hashes = {
        "ETTh1": {**hashes1, "test_sha256": base.sha256_file(TEST_CACHES["ETTh1"])},
        "ETTh2": {**hashes2, "test_sha256": base.sha256_file(TEST_CACHES["ETTh2"])},
    }
    split_plan = load_json(SPLIT_PLAN) if SPLIT_PLAN.exists() else None
    provenance = {}
    for dataset, cache in (("ETTh1", test1), ("ETTh2", test2)):
        starts = cache["absolute_window_starts"].to(torch.long)
        provenance[dataset] = {
            "path": str(TEST_CACHES[dataset].relative_to(ROOT)),
            "sha256": cache_hashes[dataset]["test_sha256"],
            "cache_role": base.cache_role(cache),
            "expert_order": list(cache["expert_names"]),
            "expected_expert_order": list(base.EXPERTS),
            "expert_order_matches": tuple(cache["expert_names"]) == base.EXPERTS,
            "starts_chronological": bool(torch.all(starts[1:] > starts[:-1])),
            "num_windows": int(cache["num_windows"]),
            "forecast_horizon": int(cache["forecast_horizon"]),
            "input_length": int(cache["histories"].shape[1]),
            "prediction_stack_shape": list(cache["prediction_stack"].shape),
            "targets_shape": list(cache["targets"].shape),
            "target_masks_shape": list(cache["target_masks"].shape),
            "target_masks_all_observed": bool(torch.all(cache["target_masks"] == 1)),
            "absolute_window_start_range": [int(starts[0]), int(starts[-1])],
            "standardization": "checkpoint-derived per-variable std (DLinear best_expert.pt), non-inverse-transformed raw-scale MAE/MSE" if dataset == "ETTh1" else "std=ones (raw/original-scale MAE/MSE, no inverse transform), matching canonical ETTh2 protocol",
            "no_expert_model_updates": True,
            "cache_hashes": cache_hashes[dataset],
        }
    if split_plan is not None:
        test_block = next((r for r in split_plan.get("ranges", []) if r.get("name") == "test"), None)
        provenance["ETTh1"]["split_plan_test_block"] = test_block

    payload = {
        **manifest,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": base.git_commit(),
        "runtime_sec": time.perf_counter() - started,
        "test_caches": {k: str(v.relative_to(ROOT)) for k, v in TEST_CACHES.items()},
        "cache_hashes": cache_hashes,
        "reference_rows": reference,
        "rankings": rankings,
        "leakage_and_causality_checks": leakage_payload,
        "diagnostics": diagnostics,
        "results": all_rows,
    }

    write_csv(OUT_DIR / "TEST_RESULTS.csv", all_rows)
    write_csv(OUT_DIR / "per_window_test_metrics.csv", all_per_window)
    write_json(OUT_DIR / "TEST_RESULTS.json", payload)
    write_json(OUT_DIR / "leakage_and_causality_checks.json", leakage_payload)
    write_json(OUT_DIR / "cache_provenance.json", provenance)
    write_report(all_rows, reference, rankings, leakage_payload, provenance, payload)

    print(json.dumps({"rankings": rankings, "leakage_and_causality_checks": {"all_checks_passed": leakage_payload["all_checks_passed"]}, "output_dir": str(OUT_DIR)}, indent=2))


if __name__ == "__main__":
    main()
