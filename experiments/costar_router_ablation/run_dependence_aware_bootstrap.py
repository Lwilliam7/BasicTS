"""Dependence-aware significance retest for the router ablation study.

The original `router_ablation_deltas.csv` used a plain IID paired bootstrap
(`paired_bootstrap`, resampling individual windows with replacement), which
assumes each window's error is independent. Walk-forward forecast windows
overlap in target horizon and are temporally autocorrelated, so an IID
bootstrap can understate the true uncertainty and make small deltas look
more significant than they are. This script re-tests the five key router
comparisons from the ablation study using:

  1. Paired moving/block bootstrap at block lengths 12, 24, 48
     (`paired_block_bootstrap`, resampling contiguous runs of windows so
     within-block autocorrelation is preserved).
  2. An every-12th non-overlapping-window analysis: since forecast_horizon=12
     for both datasets, sampling every 12th window (12 phase offsets) yields
     windows whose target spans are back-to-back rather than overlapping,
     which is the most direct way to approximately decorrelate the sample.
     The 12 phase-level mean differences are then bootstrapped themselves.

This script only reads the already-computed, already-validated per-window
MAE from `costar_router_ablation/router_ablation_per_window.csv`. It does
not load any cache, does not touch the test set, and does not change any
prediction -- it is a pure re-analysis of existing router_val results.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.chronological_adaptive_costar.run_chronological_adaptive_costar import paired_bootstrap  # noqa: E402
from experiments.dual_timescale_memory_costar.run_dual_timescale_memory_costar import paired_block_bootstrap  # noqa: E402


OUT_DIR = ROOT / "experiments/costar_router_ablation"
PER_WINDOW_CSV = OUT_DIR / "router_ablation_per_window.csv"
BLOCK_LENGTHS = (12, 24, 48)
PHASE_K = 12  # forecast_horizon for both ETTh1 and ETTh2 in this cache family

# The five key router comparisons (candidate, baseline), matching research
# questions A-E from the router ablation study. D uses "hxv_causal vs
# variable_only" specifically because that pair produced the one CI-includes-
# zero result on ETTh1 in the original IID bootstrap -- the case most worth
# re-checking under a dependence-aware test.
COMPARISONS = [
    ("A_causal_adaptation_helps", "global_causal", "equal_fixed"),
    ("B_horizon_specialization", "horizon_only", "global_causal"),
    ("C_variable_specialization", "variable_only", "global_causal"),
    ("D_joint_hxv_vs_variable_only", "hxv_causal", "variable_only"),
    ("E_global_adds_to_hxv", "global_plus_hxv", "hxv_causal"),
]


def refuse_test(path: str | Path) -> None:
    if "test" in str(path).lower():
        raise ValueError(f"Test access forbidden during dependence-aware retest: {path}")


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


def load_per_window_mae(path: Path) -> dict[str, dict[str, torch.Tensor]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_dataset_method: dict[str, dict[str, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        by_dataset_method[row["dataset"]][row["method"]][int(row["window_index"])] = float(row["mae"])
    out: dict[str, dict[str, torch.Tensor]] = {}
    for dataset, methods in by_dataset_method.items():
        out[dataset] = {}
        for method, idx_to_mae in methods.items():
            n = len(idx_to_mae)
            if set(idx_to_mae.keys()) != set(range(n)):
                raise ValueError(f"{dataset}/{method}: window_index is not a contiguous 0..N-1 range")
            out[dataset][method] = torch.tensor([idx_to_mae[i] for i in range(n)], dtype=torch.float32)
    return out


def every_kth_phase_bootstrap(diff: torch.Tensor, k: int = PHASE_K, seed: int = 20260820, samples: int = 5000) -> dict[str, Any]:
    """Split the window series into k non-overlapping phases (offset 0..k-1,
    stride k), average the diff within each phase, then bootstrap over the
    (approximately independent) phase-level means themselves."""
    n = diff.numel()
    phase_means = []
    phase_counts = []
    for offset in range(k):
        idx = torch.arange(offset, n, k)
        if idx.numel() == 0:
            continue
        phase_means.append(float(diff[idx].mean()))
        phase_counts.append(int(idx.numel()))
    t = torch.tensor(phase_means)
    num_phases = t.numel()
    gen = torch.Generator().manual_seed(seed)
    vals = []
    for _ in range(samples):
        idx = torch.randint(0, num_phases, (num_phases,), generator=gen)
        vals.append(float(t[idx].mean()))
    boot = torch.tensor(vals)
    return {
        "num_phases": num_phases,
        "windows_per_phase_min": min(phase_counts),
        "windows_per_phase_max": max(phase_counts),
        "phase_means": phase_means,
        "mean_diff_candidate_minus_baseline": float(t.mean()),
        "phase_mean_std": float(t.std(unbiased=True)) if num_phases > 1 else 0.0,
        "ci95_low": float(torch.quantile(boot, 0.025)),
        "ci95_high": float(torch.quantile(boot, 0.975)),
        "ci_excludes_zero": bool(torch.quantile(boot, 0.975) < 0 or torch.quantile(boot, 0.025) > 0),
    }


def run_comparison(dataset: str, label: str, candidate: torch.Tensor, baseline: torch.Tensor) -> list[dict[str, Any]]:
    rows = []
    iid = paired_bootstrap(candidate, baseline, seed=20260820, samples=5000)
    rows.append({"dataset": dataset, "comparison": label, "test": "iid_paired_bootstrap_original", **iid})
    for block in BLOCK_LENGTHS:
        b = paired_block_bootstrap(candidate, baseline, block=block, seed=20260820, samples=5000)
        rows.append({"dataset": dataset, "comparison": label, "test": f"block_bootstrap_len{block}", **b})
    diff = candidate - baseline
    phase = every_kth_phase_bootstrap(diff, k=PHASE_K, seed=20260820, samples=5000)
    phase_flat = {k: v for k, v in phase.items() if k != "phase_means"}
    rows.append({"dataset": dataset, "comparison": label, "test": f"every_{PHASE_K}th_window_phase_bootstrap", **phase_flat, "phase_means_json": json.dumps(phase["phase_means"])})
    return rows


def make_report(out_dir: Path, all_rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Dependence-Aware Bootstrap Retest (Router Ablation)",
        "",
        "Re-tests the five key router comparisons from `costar_router_ablation` "
        "using block bootstrap (lengths 12/24/48) and an every-12th "
        "non-overlapping-window phase analysis, alongside the original IID "
        "paired bootstrap for direct comparison. No cache was loaded, no test "
        "data was touched, and no prediction was changed -- this only re-analyzes "
        "the existing `router_ablation_per_window.csv`.",
        "",
        "## Comparisons retested",
        "",
        "| Label | Candidate | Baseline | Research question |",
        "|---|---|---|---|",
        "| A_causal_adaptation_helps | global_causal | equal_fixed | Does causal adaptation help at all? |",
        "| B_horizon_specialization | horizon_only | global_causal | Does horizon specialization help? |",
        "| C_variable_specialization | variable_only | global_causal | Does variable specialization help? |",
        "| D_joint_hxv_vs_variable_only | hxv_causal | variable_only | Does the horizon axis add anything once variable is present? |",
        "| E_global_adds_to_hxv | global_plus_hxv | hxv_causal | Does the global branch add value beyond HxV? |",
        "",
        "## Results",
        "",
        "| Dataset | Comparison | Test | Mean delta MAE | 95% CI | CI excludes zero |",
        "|---|---|---|---:|---|---|",
    ]
    test_order = ["iid_paired_bootstrap_original", "block_bootstrap_len12", "block_bootstrap_len24", "block_bootstrap_len48", "every_12th_window_phase_bootstrap"]
    by_key: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in all_rows:
        by_key[(row["dataset"], row["comparison"])][row["test"]] = row
    for dataset in ("ETTh1", "ETTh2"):
        for label, _, _ in COMPARISONS:
            for test in test_order:
                row = by_key[(dataset, label)].get(test)
                if row is None:
                    continue
                lines.append(
                    f"| {dataset} | {label} | {test} | `{row['mean_diff_candidate_minus_baseline']:+.6f}` | "
                    f"[{row['ci95_low']:+.6f}, {row['ci95_high']:+.6f}] | {row['ci_excludes_zero']} |"
                )
    lines.append("")
    lines.append("## Does the conclusion flip under any dependence-aware test?")
    lines.append("")
    lines.append("| Dataset | Comparison | IID excludes zero | All block/phase tests agree with IID? | Flags |")
    lines.append("|---|---|---|---|---|")
    for dataset in ("ETTh1", "ETTh2"):
        for label, _, _ in COMPARISONS:
            tests = by_key[(dataset, label)]
            iid_row = tests.get("iid_paired_bootstrap_original")
            if iid_row is None:
                continue
            iid_excl = iid_row["ci_excludes_zero"]
            other_tests = [tests[t] for t in test_order[1:] if t in tests]
            agree = all(bool(t["ci_excludes_zero"]) == bool(iid_excl) for t in other_tests)
            flags = "none" if agree else "DISAGREEMENT: at least one dependence-aware test flips the conclusion"
            lines.append(f"| {dataset} | {label} | {iid_excl} | {agree} | {flags} |")
    lines.append("")
    lines += ["## Hard rule compliance", "", "```text", "TEST SET ACCESSED: NO", "TEST CACHE LOADED: NO", "TEST METRICS COMPUTED: NO", "```"]
    (out_dir / "dependence_aware_bootstrap_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    refuse_test(PER_WINDOW_CSV)
    data = load_per_window_mae(PER_WINDOW_CSV)

    all_rows: list[dict[str, Any]] = []
    for dataset in ("ETTh1", "ETTh2"):
        for label, candidate_method, baseline_method in COMPARISONS:
            candidate = data[dataset][candidate_method]
            baseline = data[dataset][baseline_method]
            all_rows.extend(run_comparison(dataset, label, candidate, baseline))

    write_json(OUT_DIR / "dependence_aware_bootstrap_results.json", {"rows": all_rows, "block_lengths": BLOCK_LENGTHS, "phase_k": PHASE_K, "source": str(PER_WINDOW_CSV.relative_to(ROOT))})
    write_csv(OUT_DIR / "dependence_aware_bootstrap_results.csv", all_rows)
    make_report(OUT_DIR, all_rows)

    print("TEST SET ACCESSED: NO")
    print("TEST CACHE LOADED: NO")
    print("TEST METRICS COMPUTED: NO")
    print(json.dumps({"num_rows": len(all_rows), "block_lengths": BLOCK_LENGTHS, "phase_k": PHASE_K}, indent=2))


if __name__ == "__main__":
    main()
