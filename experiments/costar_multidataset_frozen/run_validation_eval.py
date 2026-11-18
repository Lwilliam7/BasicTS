"""Frozen multi-dataset COSTAR validation evaluation (Steps 1, 4-6, 8-9).

Freezes the candidate final router as Full horizon x variable (HxV) causal
EMA only -- no separate global branch, no global+HxV blend, no low-rank
approximation, no dual-timescale memory, no specialists, no Ridge/MLP
residual correction -- then evaluates it on router_val for ETTm1, Weather,
and Electricity alongside four comparison methods (best single expert, equal
fixed, global causal EMA, variable-only causal EMA), using the existing
canonical walk-forward split/expert-selection protocol reused unmodified from
ETTh1/ETTh2.

This script never loads a test cache and never computes a test metric. It
ends by writing `frozen_manifest.json`, the explicit gate that
`run_test_eval.py` requires before it will touch test.
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
from experiments.costar_multidataset_frozen.common import (  # noqa: E402
    CANONICAL_DECAY,
    CANONICAL_TEMPERATURE,
    CORE_SIZE,
    METHOD_ORDER,
    block_bootstrap_with_prob,
    causality_perturbation_check,
    every_kth_phase_bootstrap,
    expert_indices,
    metric_values,
    predict_method,
    refuse_test,
    select_best_single_expert,
    select_core_on_router_train,
    verify_router_train_out_of_sample,
)
from experiments.oracle_weight_tournament.run_tournament import load_cache, load_std  # noqa: E402


OUT_DIR = ROOT / "experiments/costar_multidataset_frozen"
BLOCK_LENGTHS = (12, 24, 48)
BOOTSTRAP_SAMPLES = 10000
PHASE_K = 12
KEY_COMPARISONS = [
    ("global_vs_equal", "global_causal", "equal_fixed"),
    ("variable_vs_global", "variable_only", "global_causal"),
    ("hxv_vs_global", "hxv_causal", "global_causal"),
    ("hxv_vs_variable", "hxv_causal", "variable_only"),
]

DATASETS = {
    "ETTm1": {
        "data_dir": "datasets/ETTm1",
        "cache_dir": "cache/costarts_walkforward_ETTm1",
        "checkpoint_root": "checkpoints/costarts_walkforward_ETTm1",
    },
    "Weather": {
        "data_dir": "datasets/Weather",
        "cache_dir": "cache/costarts_walkforward_Weather",
        "checkpoint_root": "checkpoints/costarts_walkforward_Weather",
    },
    "Electricity": {
        "data_dir": "datasets/Electricity",
        "cache_dir": "cache/costarts_walkforward_Electricity",
        "checkpoint_root": "checkpoints/costarts_walkforward_Electricity",
    },
}
INPUT_LEN = 96
FORECAST_HORIZON = 12


def git_commit_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def sha256_file(path: Path) -> str:
    import hashlib

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
        for row in rows:
            writer.writerow(row)


def per_window_rows(dataset: str, method: str, split: str, cache: Mapping[str, Any], std: torch.Tensor, pred: torch.Tensor) -> list[dict[str, Any]]:
    m = metric_values(cache, pred, std)
    starts = cache["absolute_window_starts"].to(torch.long)
    return [
        {
            "dataset": dataset,
            "method": method,
            "split": split,
            "window_index": i,
            "absolute_window_start": int(starts[i]),
            "mae": float(m["per_window_mae"][i]),
            "mse": float(m["per_window_mse"][i]),
        }
        for i in range(cache["num_windows"])
    ]


def evaluate_dataset(dataset: str, paths: Mapping[str, str]) -> dict[str, Any]:
    cache_dir = ROOT / paths["cache_dir"]
    checkpoint_root = ROOT / paths["checkpoint_root"]
    train_cache_path = cache_dir / "router_train_20_60_cache.pt"
    val_cache_path = cache_dir / "router_val_60_80_cache.pt"
    normalizer_path = checkpoint_root / "final_60" / "DLinear" / "best_expert.pt"
    for p in (train_cache_path, val_cache_path, normalizer_path):
        refuse_test(p)

    train_cache = load_cache(train_cache_path, "router_train_20_60")
    val_cache = load_cache(val_cache_path, "router_val_60_80")
    std = load_std(normalizer_path, int(val_cache["num_features"]))

    oos_check = verify_router_train_out_of_sample(train_cache)
    if oos_check["result"] != "PASS":
        raise AssertionError(f"{dataset}: router_train is not verifiably out-of-sample: {oos_check}")

    core_rows, selected = select_core_on_router_train(train_cache, std, core_size=CORE_SIZE)
    expert_idx = expert_indices(val_cache, selected["experts"])
    best_expert_col, best_expert_name, best_expert_train_mae = select_best_single_expert(train_cache, std, expert_idx)

    results: dict[str, dict[str, Any]] = {}
    result_rows: list[dict[str, Any]] = []
    per_window: list[dict[str, Any]] = []
    causality_rows: list[dict[str, Any]] = []

    for method, label in METHOD_ORDER:
        pred, extra = predict_method(method, val_cache, train_cache, expert_idx, std, best_expert_col)
        m = metric_values(val_cache, pred, std)
        results[method] = {"pred": pred, "mae": m["mae"], "mse": m["mse"], "per_window_mae": m["per_window_mae"], "extra": extra}
        result_rows.append(
            {
                "dataset": dataset,
                "split": "validation",
                "method": method,
                "label": label,
                "mae": m["mae"],
                "mse": m["mse"],
                "expert_set": "+".join(selected["experts"]),
                "best_single_expert_name": best_expert_name if method == "best_single_expert" else "",
                **extra,
            }
        )
        per_window.extend(per_window_rows(dataset, method, "validation", val_cache, std, pred))
        check = causality_perturbation_check(method, val_cache, train_cache, expert_idx, std, best_expert_col)
        causality_rows.append({"dataset": dataset, "split": "validation", **check})
        if check["result"] != "PASS":
            raise AssertionError(f"{dataset} {method}: causality perturbation check failed on router_val")

    delta_rows = []
    for label, cand, base in KEY_COMPARISONS:
        boot = paired_bootstrap(results[cand]["per_window_mae"], results[base]["per_window_mae"], seed=20260821, samples=5000)
        delta_rows.append(
            {
                "dataset": dataset,
                "comparison": label,
                "candidate": cand,
                "baseline": base,
                "delta_mae": results[cand]["mae"] - results[base]["mae"],
                **{f"iid_{k}": v for k, v in boot.items()},
            }
        )

    dependence_rows: list[dict[str, Any]] = []
    for label, cand, base in KEY_COMPARISONS:
        candidate, baseline = results[cand]["per_window_mae"], results[base]["per_window_mae"]
        for block in BLOCK_LENGTHS:
            b = block_bootstrap_with_prob(candidate, baseline, block=block, seed=20260821, samples=BOOTSTRAP_SAMPLES)
            dependence_rows.append({"dataset": dataset, "comparison": label, "candidate": cand, "baseline": base, "test": f"block_bootstrap_len{block}", **b})
        diff = candidate - baseline
        phase = every_kth_phase_bootstrap(diff, k=PHASE_K, seed=20260821, samples=BOOTSTRAP_SAMPLES)
        dependence_rows.append({"dataset": dataset, "comparison": label, "candidate": cand, "baseline": base, "test": f"every_{PHASE_K}th_window_phase_bootstrap", **phase})

    cache_hashes = {
        "router_train_20_60_sha256": sha256_file(train_cache_path),
        "router_val_60_80_sha256": sha256_file(val_cache_path),
        "normalizer_checkpoint_sha256": sha256_file(normalizer_path),
    }
    for expert in selected["experts"]:
        ckpt = checkpoint_root / "final_60" / expert / "best_expert.pt"
        if ckpt.exists():
            cache_hashes[f"final_60_{expert}_checkpoint_sha256"] = sha256_file(ckpt)

    return {
        "dataset": dataset,
        "num_features": int(val_cache["num_features"]),
        "num_windows_train": int(train_cache["num_windows"]),
        "num_windows_val": int(val_cache["num_windows"]),
        "selected_core": selected,
        "core_selection_leaderboard_top5": core_rows[:5],
        "best_single_expert": {"name": best_expert_name, "column_index": best_expert_col, "router_train_mae": best_expert_train_mae},
        "result_rows": result_rows,
        "per_window": per_window,
        "delta_rows": delta_rows,
        "dependence_rows": dependence_rows,
        "causality_rows": causality_rows,
        "oos_check": oos_check,
        "cache_hashes": cache_hashes,
        "cache_paths": {"router_train": str(train_cache_path.relative_to(ROOT)), "router_val": str(val_cache_path.relative_to(ROOT)), "normalizer": str(normalizer_path.relative_to(ROOT))},
    }


def make_report(out_dir: Path, report: Mapping[str, Any]) -> None:
    lines = [
        "# Frozen Multi-Dataset COSTAR Validation Report",
        "",
        "**Frozen candidate router: Full horizon x variable (HxV) causal EMA only.** "
        "No separate global branch, no global+HxV blend, no low-rank approximation, "
        "no dual-timescale memory, no specialists, no Ridge/MLP residual correction.",
        "",
        f"Git commit: `{report['git_commit_sha']}`",
        f"Canonical settings (unchanged from ETTh1/ETTh2, not tuned per dataset): decay={CANONICAL_DECAY}, temperature={CANONICAL_TEMPERATURE}, core size={CORE_SIZE}, "
        f"input_len={INPUT_LEN}, forecast_horizon={FORECAST_HORIZON}.",
        "",
        "## Step 5: validation results",
        "",
        "| Dataset | Best Single | Equal Fixed | Global | Variable-only | Full HxV |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for ds, d in report["datasets"].items():
        by_method = {r["method"]: r for r in d["result_rows"]}
        lines.append(
            "| {ds} | `{a[mae]:.6f}`/`{a[mse]:.6f}` | `{b[mae]:.6f}`/`{b[mse]:.6f}` | `{c[mae]:.6f}`/`{c[mse]:.6f}` | `{d[mae]:.6f}`/`{d[mse]:.6f}` | `{e[mae]:.6f}`/`{e[mse]:.6f}` |".format(
                ds=ds, a=by_method["best_single_expert"], b=by_method["equal_fixed"], c=by_method["global_causal"], d=by_method["variable_only"], e=by_method["hxv_causal"]
            )
        )
    lines += ["", "## Selected expert core per dataset", ""]
    for ds, d in report["datasets"].items():
        s = d["selected_core"]
        lines.append(f"- **{ds}**: `{s['subset']}` (pooled router_train OOF MAE `{s['pooled_oof_mae']:.6f}`); best single expert in core: `{d['best_single_expert']['name']}`.")
    lines += ["", "## Deltas (IID paired bootstrap, quick reference)", ""]
    lines.append("| Dataset | Comparison | Delta MAE | IID 95% CI | Excludes zero |")
    lines.append("|---|---|---:|---|---|")
    for ds, d in report["datasets"].items():
        for row in d["delta_rows"]:
            lines.append(f"| {ds} | {row['comparison']} | `{row['delta_mae']:+.6f}` | [{row['iid_ci95_low']:+.6f}, {row['iid_ci95_high']:+.6f}] | {row['iid_ci_excludes_zero']} |")
    lines += ["", "## Step 6: dependence-aware statistics (block bootstrap + every-12th phase)", ""]
    lines.append("| Dataset | Comparison | Test | Mean delta | 95% CI | P(delta<0) | Excludes zero |")
    lines.append("|---|---|---|---:|---|---:|---|")
    for ds, d in report["datasets"].items():
        for row in d["dependence_rows"]:
            lines.append(
                f"| {ds} | {row['comparison']} | {row['test']} | `{row['mean_delta']:+.6f}` | "
                f"[{row['ci95_low']:+.6f}, {row['ci95_high']:+.6f}] | {row['prob_delta_negative']:.3f} | {row['ci_excludes_zero']} |"
            )
    lines += ["", "## Step 9: causality checks", ""]
    lines.append("| Dataset | Method | Starts chronological | Earlier windows unchanged | Result |")
    lines.append("|---|---|---|---|---|")
    for ds, d in report["datasets"].items():
        for c in d["causality_rows"]:
            lines.append(f"| {ds} | {c['method']} | {c['starts_chronological']} | {c['earlier_windows_unchanged']} | {c['result']} |")
        lines.append(f"| {ds} | router_train out-of-sample | -- | -- | {d['oos_check']['result']} |")
    lines += ["", "## Hard rule compliance", "", "```text", "TEST SET ACCESSED: NO", "TEST CACHE LOADED: NO", "TEST METRICS COMPUTED: NO", "```"]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_frozen_manifest(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "manifest_type": "frozen_router_manifest",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_sha": report["git_commit_sha"],
        "router_type": "Full horizon x variable (HxV) causal EMA only",
        "excluded_from_candidate": [
            "separate global EMA branch",
            "global + HxV blend",
            "dual-timescale memory",
            "low-rank HxV",
            "specialists (DLinear/ModernTCN residual correction)",
            "Ridge residual corrector",
            "MLP residual corrector",
        ],
        "decay": CANONICAL_DECAY,
        "temperature": CANONICAL_TEMPERATURE,
        "error_to_weight_rule": "errors_to_weights(): centered per-expert error, softmax(-centered_error/temperature) over the full H x V x expert EMA (aggregate_error mode='hv')",
        "causal_observability_rule": "old_start + forecast_horizon <= current_start (enforce_observable(), raises on violation)",
        "forecast_horizon": FORECAST_HORIZON,
        "input_len": INPUT_LEN,
        "expert_pool": list(("DLinear", "PatchTST", "iTransformer", "TimesNet", "ModernTCN")),
        "expert_selection_rule": f"top-{CORE_SIZE}-of-5 combination minimizing pooled chronological out-of-fold MAE over 4 folds within router_train (unchanged from ETTh1/ETTh2 protocol)",
        "split_definitions": {
            "block_a": "0-20% (expert training range for block_b_oos)",
            "block_b": "20-40% (out-of-sample prediction range, experts trained on block_a only)",
            "block_c": "40-60% (out-of-sample prediction range, experts trained on block_a+block_b only)",
            "router_train_20_60": "block_b_oos + block_c_oos combined (20-60%), both out-of-sample for the experts that produced them",
            "router_val_60_80": "60-80%, experts trained on 0-60% (block_a+b+c)",
            "test_80_100": "80-100%, same 0-60%-trained experts, evaluated exactly once after this manifest is frozen",
        },
        "walkforward_protocol_source": "scripts/build_costarts_walkforward_cache.py + scripts/train_costarts_walkforward_experts.py (existing canonical scheme, reused; generalized to be dataset-parameterized instead of ETTh1-hardcoded -- see stage_definitions() fix)",
        "datasets": {
            ds: {
                "selected_core": d["selected_core"]["experts"],
                "best_single_expert": d["best_single_expert"]["name"],
                "num_windows_train": d["num_windows_train"],
                "num_windows_val": d["num_windows_val"],
                "num_features": d["num_features"],
                "cache_hashes": d["cache_hashes"],
                "cache_paths": d["cache_paths"],
            }
            for ds, d in report["datasets"].items()
        },
        "test_accessed": False,
        "test_cache_loaded": False,
        "test_metrics_computed": False,
        "frozen": True,
        "note": "Architecture, decay, temperature, expert pool, and expert-selection rule are frozen as of this manifest. Validation results above are for correctness-checking and reporting only and must not be used to change any of these settings.",
    }


def main() -> None:
    start = time.time()
    report: dict[str, Any] = {
        "experiment": "costar_multidataset_frozen_validation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_sha": git_commit_sha(),
        "datasets": {},
    }
    all_results: list[dict[str, Any]] = []
    all_per_window: list[dict[str, Any]] = []
    all_deltas: list[dict[str, Any]] = []
    all_dependence: list[dict[str, Any]] = []
    all_causality: list[dict[str, Any]] = []

    for dataset, paths in DATASETS.items():
        print(f"[frozen-multidataset] {dataset}: evaluating on router_val...", flush=True)
        result = evaluate_dataset(dataset, paths)
        report["datasets"][dataset] = {k: v for k, v in result.items() if k not in ("per_window",)}
        all_results.extend(result["result_rows"])
        all_per_window.extend(result["per_window"])
        all_deltas.extend(result["delta_rows"])
        all_dependence.extend(result["dependence_rows"])
        all_causality.extend(result["causality_rows"])
        print(f"[frozen-multidataset] {dataset}: done. core={result['selected_core']['subset']}", flush=True)

    report["runtime_sec"] = time.time() - start
    report["test_set_accessed"] = False
    report["test_cache_loaded"] = False
    report["test_metrics_computed"] = False

    write_json(OUT_DIR / "validation_results.json", report)
    write_csv(OUT_DIR / "validation_results.csv", all_results)
    write_csv(OUT_DIR / "per_window_metrics.csv", all_per_window)
    write_csv(OUT_DIR / "validation_deltas.csv", all_deltas)
    write_csv(OUT_DIR / "dependence_aware_bootstrap.csv", all_dependence)
    write_json(OUT_DIR / "dependence_aware_bootstrap.json", all_dependence)
    write_csv(OUT_DIR / "causality_checks.csv", all_causality)
    write_json(
        OUT_DIR / "dataset_metadata.json",
        {ds: {"num_features": d["num_features"], "num_windows_train": d["num_windows_train"], "num_windows_val": d["num_windows_val"], "cache_paths": d["cache_paths"]} for ds, d in report["datasets"].items()},
    )
    manifest = make_frozen_manifest(report)
    write_json(OUT_DIR / "frozen_manifest.json", manifest)
    make_report(OUT_DIR, report)

    print("TEST SET ACCESSED: NO")
    print("TEST CACHE LOADED: NO")
    print("TEST METRICS COMPUTED: NO")
    print(json.dumps({"runtime_sec": report["runtime_sec"], "datasets": list(report["datasets"].keys())}, indent=2))


if __name__ == "__main__":
    main()
