"""Dedicated global-branch utility test for COSTAR.

Scientific question: does the separate global causal EMA branch provide
useful information beyond the full horizon x variable (HxV) causal EMA?

Reuses the already-computed, already-validated per-window MAE from
`costar_router_ablation/router_ablation_per_window.csv` (produced by
`run_router_ablation.py`). No expert is retrained, no router hyperparameter
is changed, no cache is loaded, and the test set is never touched -- this is
a pure re-analysis of three existing router_val prediction methods:

  1. global_causal    -- Global causal EMA only
  2. hxv_causal        -- Full horizon x variable causal EMA only
  3. global_plus_hxv   -- Global + HxV COSTAR (production blend)
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.costar_router_ablation.run_dependence_aware_bootstrap import (  # noqa: E402
    PER_WINDOW_CSV,
    every_kth_phase_bootstrap,
    load_per_window_mae,
    refuse_test,
)


OUT_DIR = ROOT / "experiments/costar_router_ablation"
BLOCK_LENGTHS = (12, 24, 48)
PHASE_K = 12
NUM_CHRONO_BLOCKS = 8
CANDIDATE_METHOD = "global_plus_hxv"  # Global + HxV
BASELINE_METHOD = "hxv_causal"  # Full HxV only
GLOBAL_ONLY_METHOD = "global_causal"  # Global only
DATASETS = ("ETTh1", "ETTh2")

# Thresholds for the final classification (documented, not tuned against the
# data -- these are pre-specified interpretive rules).
TINY_RELATIVE_MAE = 0.003  # 0.3% relative delta counts as "tiny"
STRONG_PROB_THRESHOLD = 0.85  # bootstrap P(delta<0) or P(delta>0) needed to call a direction "consistent"
BLOCK_WIN_MAJORITY = 6  # out of 8 chronological blocks, needed to call "consistently helps/hurts"


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


def overall_mae_mse(per_window_mae: torch.Tensor, per_window_mse: torch.Tensor) -> dict[str, float]:
    return {"mae": float(per_window_mae.mean()), "mse": float(per_window_mse.mean())}


def block_bootstrap_with_prob(candidate: torch.Tensor, baseline: torch.Tensor, block: int, seed: int = 20260821, samples: int = 5000) -> dict[str, Any]:
    """Same paired moving/block bootstrap as `paired_block_bootstrap`, plus
    P(mean delta < 0) across resamples -- the probability that Global+HxV
    improves on HxV under this resampling scheme."""
    diff = candidate - baseline
    n = diff.numel()
    block = max(1, min(block, n))
    n_blocks = max(1, -(-n // block))
    gen = torch.Generator().manual_seed(seed)
    vals = []
    for _ in range(samples):
        starts_idx = torch.randint(0, n - block + 1, (n_blocks,), generator=gen)
        idx = torch.cat([torch.arange(s, s + block) for s in starts_idx.tolist()])[:n]
        vals.append(float(diff[idx].mean()))
    t = torch.tensor(vals)
    return {
        "block_size": block,
        "mean_delta": float(diff.mean()),
        "ci95_low": float(torch.quantile(t, 0.025)),
        "ci95_high": float(torch.quantile(t, 0.975)),
        "ci_excludes_zero": bool(torch.quantile(t, 0.975) < 0 or torch.quantile(t, 0.025) > 0),
        "prob_delta_negative": float((t < 0).to(torch.float32).mean()),
    }


def chronological_block_split(candidate: torch.Tensor, baseline: torch.Tensor, num_blocks: int) -> list[dict[str, Any]]:
    n = candidate.numel()
    bounds = [i * n // num_blocks for i in range(num_blocks + 1)]
    rows = []
    for b in range(num_blocks):
        lo, hi = bounds[b], bounds[b + 1]
        cand_mae = float(candidate[lo:hi].mean())
        base_mae = float(baseline[lo:hi].mean())
        delta = cand_mae - base_mae
        rows.append(
            {
                "block": b,
                "window_lo": lo,
                "window_hi": hi - 1,
                "num_windows": hi - lo,
                "hxv_only_mae": base_mae,
                "global_plus_hxv_mae": cand_mae,
                "delta": delta,
                "winner": "global_plus_hxv" if delta < 0 else ("hxv_only" if delta > 0 else "tie"),
            }
        )
    return rows


def block_summary(block_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    helping = [r for r in block_rows if r["delta"] < 0]
    hurting = [r for r in block_rows if r["delta"] > 0]
    return {
        "num_blocks": len(block_rows),
        "blocks_global_helps": len(helping),
        "blocks_global_hurts": len(hurting),
        "blocks_tied": len(block_rows) - len(helping) - len(hurting),
        "avg_improvement_in_helping_blocks": float(sum(-r["delta"] for r in helping) / len(helping)) if helping else None,
        "avg_regression_in_hurting_blocks": float(sum(r["delta"] for r in hurting) / len(hurting)) if hurting else None,
    }


def classify(dataset_verdicts: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    """Combine both datasets' evidence into one of: consistently useful,
    conditionally useful / regime-dependent, redundant, harmful."""
    d1, d2 = dataset_verdicts["ETTh1"], dataset_verdicts["ETTh2"]
    signs = [1 if d["point_delta"] > 0 else (-1 if d["point_delta"] < 0 else 0) for d in (d1, d2)]
    both_tiny = all(abs(d["relative_delta"]) < TINY_RELATIVE_MAE for d in (d1, d2))
    any_block_bootstrap_significant = any(d["any_block_bootstrap_excludes_zero"] for d in (d1, d2))
    sign_flips = signs[0] != signs[1] and 0 not in signs
    both_help = all(s < 0 for s in signs)
    both_hurt = all(s > 0 for s in signs)
    strong_majority_help = all(d["blocks_global_helps"] >= BLOCK_WIN_MAJORITY for d in (d1, d2))
    strong_majority_hurt = all(d["blocks_global_hurts"] >= BLOCK_WIN_MAJORITY for d in (d1, d2))

    if both_help and strong_majority_help and any_block_bootstrap_significant and not both_tiny:
        label = "consistently_useful"
        rationale = "Global+HxV beats HxV on both datasets, in a clear majority of chronological blocks, with at least one block-bootstrap CI excluding zero, and the effect is not trivially small."
    elif both_hurt and strong_majority_hurt and any_block_bootstrap_significant and not both_tiny:
        label = "harmful"
        rationale = "Global+HxV is worse than HxV on both datasets, in a clear majority of chronological blocks, with at least one block-bootstrap CI excluding zero, and the effect is not trivially small."
    elif sign_flips:
        label = "conditionally_useful_regime_dependent" if not both_tiny else "redundant"
        rationale = (
            "The sign of the effect flips between ETTh1 and ETTh2 (helps on one, hurts on the other). "
            + ("Effect sizes are also tiny on both datasets, so the flip is more likely noise than a real regime effect -- classified as redundant rather than regime-dependent."
               if both_tiny else
               "Effect sizes are non-trivial, so this looks like a genuine regime-dependent effect rather than noise.")
        )
    elif not any_block_bootstrap_significant and both_tiny:
        label = "redundant"
        rationale = "No block-bootstrap CI (the more conservative, dependence-aware test) excludes zero on either dataset, and the point-estimate effect sizes are tiny relative to baseline MAE. The global branch does not earn its added complexity."
    else:
        label = "conditionally_useful_regime_dependent"
        rationale = "Evidence is mixed across datasets/tests without a clean sign flip or a clean consistent win -- effect appears to depend on dataset/regime rather than being a uniform, reliable improvement."
    return {"label": label, "rationale": rationale}


def make_report(out_dir: Path, report: Mapping[str, Any]) -> None:
    lines = [
        "# Global-Branch Utility Test",
        "",
        '**Question**: does the separate global causal EMA branch provide useful information beyond the full horizon x variable (HxV) causal EMA?',
        "",
        "Pure re-analysis of existing `router_ablation_per_window.csv` results (methods `global_causal`, `hxv_causal`, `global_plus_hxv`). No expert retrained, no router hyperparameter changed, no cache loaded, test set never touched.",
        "",
        "## A. Overall validation MAE / MSE",
        "",
        "| Dataset | Global only | Full HxV only | Global + HxV |",
        "|---|---|---|---|",
    ]
    for ds in DATASETS:
        a = report["datasets"][ds]["overall"]
        lines.append(
            f"| {ds} | MAE `{a['global_causal']['mae']:.6f}` / MSE `{a['global_causal']['mse']:.6f}` | "
            f"MAE `{a['hxv_causal']['mae']:.6f}` / MSE `{a['hxv_causal']['mse']:.6f}` | "
            f"MAE `{a['global_plus_hxv']['mae']:.6f}` / MSE `{a['global_plus_hxv']['mse']:.6f}` |"
        )
    lines += ["", "## B/C. Global+HxV vs HxV-only: paired block bootstrap", ""]
    lines.append("| Dataset | Block size | Mean delta (Global+HxV minus HxV) | 95% CI | P(delta<0) | CI excludes zero |")
    lines.append("|---|---:|---:|---|---:|---|")
    for ds in DATASETS:
        for row in report["datasets"][ds]["block_bootstrap"]:
            lines.append(
                f"| {ds} | {row['block_size']} | `{row['mean_delta']:+.6f}` | [{row['ci95_low']:+.6f}, {row['ci95_high']:+.6f}] | "
                f"{row['prob_delta_negative']:.3f} | {row['ci_excludes_zero']} |"
            )
    lines += ["", "## D. Every-12th non-overlapping-window evaluation", ""]
    lines.append("| Dataset | Mean delta | 95% CI (bootstrap over 12 phase means) | CI excludes zero |")
    lines.append("|---|---:|---|---|")
    for ds in DATASETS:
        p = report["datasets"][ds]["phase12"]
        lines.append(f"| {ds} | `{p['mean_diff_candidate_minus_baseline']:+.6f}` | [{p['ci95_low']:+.6f}, {p['ci95_high']:+.6f}] | {p['ci_excludes_zero']} |")
    lines += ["", "## E. Chronological 8-block split", ""]
    for ds in DATASETS:
        lines.append(f"### {ds}")
        lines.append("")
        lines.append("| Block | Windows | HxV-only MAE | Global+HxV MAE | Delta | Winner |")
        lines.append("|---:|---|---:|---:|---:|---|")
        for r in report["datasets"][ds]["chrono_blocks"]:
            lines.append(f"| {r['block']} | {r['window_lo']}-{r['window_hi']} | `{r['hxv_only_mae']:.6f}` | `{r['global_plus_hxv_mae']:.6f}` | `{r['delta']:+.6f}` | {r['winner']} |")
        lines.append("")
    lines += ["## F. Block win/loss summary", ""]
    lines.append("| Dataset | Blocks global helps | Blocks global hurts | Avg improvement (helping) | Avg regression (hurting) |")
    lines.append("|---|---:|---:|---:|---:|")
    for ds in DATASETS:
        s = report["datasets"][ds]["block_summary"]
        imp = f"`{s['avg_improvement_in_helping_blocks']:+.6f}`" if s["avg_improvement_in_helping_blocks"] is not None else "n/a"
        reg = f"`{s['avg_regression_in_hurting_blocks']:+.6f}`" if s["avg_regression_in_hurting_blocks"] is not None else "n/a"
        lines.append(f"| {ds} | {s['blocks_global_helps']}/{s['num_blocks']} | {s['blocks_global_hurts']}/{s['num_blocks']} | {imp} | {reg} |")
    lines += ["", "## Final classification", ""]
    lines.append(f"**{report['classification']['label']}**")
    lines.append("")
    lines.append(report["classification"]["rationale"])
    lines.append("")
    for ds in DATASETS:
        d = report["datasets"][ds]["verdict_inputs"]
        lines.append(
            f"- **{ds}**: point delta `{d['point_delta']:+.6f}` ({d['relative_delta']*100:+.3f}% relative), "
            f"{d['blocks_global_helps']}/8 chronological blocks favor Global+HxV, "
            f"any block-bootstrap CI excludes zero: {d['any_block_bootstrap_excludes_zero']}."
        )
    lines.append("")
    lines += ["## Hard rule compliance", "", "```text", "TEST SET ACCESSED: NO", "TEST CACHE LOADED: NO", "TEST METRICS COMPUTED: NO", "```"]
    (out_dir / "global_branch_utility_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    refuse_test(PER_WINDOW_CSV)
    data = load_per_window_mae(PER_WINDOW_CSV)
    # load_per_window_mae only returns MAE; also load MSE directly for section A.
    with PER_WINDOW_CSV.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    mse_by = {}
    for row in raw_rows:
        mse_by.setdefault(row["dataset"], {}).setdefault(row["method"], {})[int(row["window_index"])] = float(row["mse"])

    report: dict[str, Any] = {"datasets": {}}
    all_block_bootstrap_rows: list[dict[str, Any]] = []
    all_chrono_block_rows: list[dict[str, Any]] = []
    dataset_verdict_inputs: dict[str, dict[str, Any]] = {}

    for ds in DATASETS:
        g_only = data[ds][GLOBAL_ONLY_METHOD]
        hxv_only = data[ds][BASELINE_METHOD]
        combo = data[ds][CANDIDATE_METHOD]
        n = combo.numel()
        mse_g = torch.tensor([mse_by[ds][GLOBAL_ONLY_METHOD][i] for i in range(n)])
        mse_h = torch.tensor([mse_by[ds][BASELINE_METHOD][i] for i in range(n)])
        mse_c = torch.tensor([mse_by[ds][CANDIDATE_METHOD][i] for i in range(n)])

        overall = {
            "global_causal": overall_mae_mse(g_only, mse_g),
            "hxv_causal": overall_mae_mse(hxv_only, mse_h),
            "global_plus_hxv": overall_mae_mse(combo, mse_c),
        }

        block_rows = [block_bootstrap_with_prob(combo, hxv_only, block=L) for L in BLOCK_LENGTHS]
        for r in block_rows:
            all_block_bootstrap_rows.append({"dataset": ds, **r})

        diff = combo - hxv_only
        phase = every_kth_phase_bootstrap(diff, k=PHASE_K)
        phase_flat = {k: v for k, v in phase.items() if k != "phase_means"}

        chrono_rows = chronological_block_split(combo, hxv_only, NUM_CHRONO_BLOCKS)
        for r in chrono_rows:
            all_chrono_block_rows.append({"dataset": ds, **r})
        bsummary = block_summary(chrono_rows)

        point_delta = overall["global_plus_hxv"]["mae"] - overall["hxv_causal"]["mae"]
        verdict_inputs = {
            "point_delta": point_delta,
            "relative_delta": point_delta / overall["hxv_causal"]["mae"],
            "blocks_global_helps": bsummary["blocks_global_helps"],
            "blocks_global_hurts": bsummary["blocks_global_hurts"],
            "any_block_bootstrap_excludes_zero": any(r["ci_excludes_zero"] for r in block_rows),
        }
        dataset_verdict_inputs[ds] = verdict_inputs

        report["datasets"][ds] = {
            "overall": overall,
            "block_bootstrap": block_rows,
            "phase12": phase_flat,
            "chrono_blocks": chrono_rows,
            "block_summary": bsummary,
            "verdict_inputs": verdict_inputs,
        }

    classification = classify(dataset_verdict_inputs)
    report["classification"] = classification

    write_json(OUT_DIR / "global_branch_utility_results.json", report)
    write_csv(OUT_DIR / "global_branch_utility_block_bootstrap.csv", all_block_bootstrap_rows)
    write_csv(OUT_DIR / "global_branch_utility_chrono_blocks.csv", all_chrono_block_rows)
    make_report(OUT_DIR, report)

    print("TEST SET ACCESSED: NO")
    print("TEST CACHE LOADED: NO")
    print("TEST METRICS COMPUTED: NO")
    print(json.dumps({"classification": classification, "verdict_inputs": dataset_verdict_inputs}, indent=2))


if __name__ == "__main__":
    main()
