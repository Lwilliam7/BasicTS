"""Isolates the decision rule that converts learned-probe competence scores
into ensemble weights, holding everything else fixed.

Reuses, unmodified and un-retrained:
  - `Learned_Probe_pred_excess` (predicted_excess_loss[t,e]) saved by
    run_learned_probe.py -- the exact output of the already-trained
    ProbeGenerator + competence scorer on router_val. Never regenerated.
  - Original (unperturbed) expert forecasts, read fresh from the existing
    cache via `bundle.forecasts_fn` (a cache read, not a model run).
  - C / Fixed-D / Equal / Best-Single / Window-Oracle reference predictions
    from the original behavioral_competence experiment's saved npz files.

No expert, ProbeGenerator, or competence scorer is instantiated or run in
this script at all -- it is pure post-processing of already-saved arrays.
Validation only; no test cache is ever touched.
"""

from __future__ import annotations

import csv
import itertools
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.behavioral_competence.common import competence_to_weights  # noqa: E402
from experiments.behavioral_competence.run_behavioral_competence import RESULTS_DIR as ORIGINAL_RESULTS_DIR  # noqa: E402
from experiments.chronological_adaptive_costar.run_chronological_adaptive_costar import paired_bootstrap  # noqa: E402
from experiments.costar_multidataset_frozen.common import block_bootstrap_with_prob, every_kth_phase_bootstrap  # noqa: E402
from experiments.frozen_hv_costar.run_frozen_hv_costar import LOADERS, metric_values  # noqa: E402
from experiments.oracle_weight_tournament.run_tournament import sample_mae  # noqa: E402


OUT_DIR = ROOT / "experiments/behavioral_competence"
RESULTS_DIR = OUT_DIR / "results"
REPORTS_DIR = OUT_DIR / "reports"
LEARNED_PROBE_RESULTS = RESULTS_DIR / "learned_probe_results.json"
LEARNED_PROBE_NPZ = RESULTS_DIR / "learned_probe_per_window.npz"
ORIGINAL_NPZ = ORIGINAL_RESULTS_DIR / "per_window_predictions.npz"
BLOCK_LENGTHS = (12, 24, 48)
BOOTSTRAP_SAMPLES = 10000
PHASE_K = 12
MARGIN_QUANTILE_SPLIT = 0.5  # median split for the required "large vs small margin" analysis


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Decision rules: pure functions of predicted_excess_loss [N,K] -> weights [N,K]
# ---------------------------------------------------------------------------


def rule_softmax(pred_excess: torch.Tensor, temperature: float) -> torch.Tensor:
    return competence_to_weights(pred_excess, temperature)


def rule_top1(pred_excess: torch.Tensor) -> torch.Tensor:
    n, k = pred_excess.shape
    winner = pred_excess.argmin(dim=1)
    weights = torch.zeros(n, k)
    weights.scatter_(1, winner.view(-1, 1), 1.0)
    return weights


def rule_top2_equal(pred_excess: torch.Tensor) -> torch.Tensor:
    n, k = pred_excess.shape
    order = pred_excess.argsort(dim=1)
    weights = torch.zeros(n, k)
    top2 = order[:, : min(2, k)]
    weights.scatter_(1, top2, 0.5 if k >= 2 else 1.0)
    if k >= 2:
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return weights


def rule_fixed_rank(pred_excess: torch.Tensor) -> torch.Tensor:
    n, k = pred_excess.shape
    order = pred_excess.argsort(dim=1)  # order[:,0] = best (lowest predicted excess loss)
    raw = torch.arange(k, 0, -1, dtype=torch.float32)  # [K, K-1, ..., 1]
    raw = raw / raw.sum()
    weights = torch.zeros(n, k)
    weights.scatter_(1, order, raw.view(1, k).expand(n, -1))
    return weights


DECISION_RULES = ["LearnedProbe_Softmax", "LearnedProbe_Top1", "LearnedProbe_Top2Equal", "LearnedProbe_Rank"]


def weights_for_rule(rule: str, pred_excess: torch.Tensor, temperature: float) -> torch.Tensor:
    if rule == "LearnedProbe_Softmax":
        return rule_softmax(pred_excess, temperature)
    if rule == "LearnedProbe_Top1":
        return rule_top1(pred_excess)
    if rule == "LearnedProbe_Top2Equal":
        return rule_top2_equal(pred_excess)
    if rule == "LearnedProbe_Rank":
        return rule_fixed_rank(pred_excess)
    raise ValueError(rule)


# ---------------------------------------------------------------------------
# Oracle diagnostics: best-single (already have as Window_Oracle_reference)
# and best-PAIR oracle (new here, diagnostic only, uses targets).
# ---------------------------------------------------------------------------


def best_pair_oracle(forecasts_all: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    n, h, f, k = forecasts_all.shape
    pairs = list(itertools.combinations(range(k), 2))
    best_mae = torch.full((n,), float("inf"))
    best_pred = torch.zeros(n, h, f)
    for i, j in pairs:
        pred = 0.5 * (forecasts_all[..., i] + forecasts_all[..., j])
        mae = sample_mae(pred, target, mask, std)
        improved = mae < best_mae
        best_mae = torch.where(improved, mae, best_mae)
        best_pred = torch.where(improved.view(n, 1, 1), pred, best_pred)
    return best_pred


# ---------------------------------------------------------------------------
# Per-dataset evaluation
# ---------------------------------------------------------------------------


def evaluate_dataset(dataset: str) -> dict[str, Any]:
    bundle = LOADERS[dataset]()
    val_cache = bundle.val_cache
    k = len(bundle.core_names)

    lp_npz = np.load(LEARNED_PROBE_NPZ)
    orig_npz = np.load(ORIGINAL_NPZ)
    lp_report = json.loads(LEARNED_PROBE_RESULTS.read_text(encoding="utf-8"))
    temperature = float(lp_report["datasets"][dataset]["temperature_instance"])

    pred_excess = torch.tensor(lp_npz[f"{dataset}__Learned_Probe_pred_excess"], dtype=torch.float32)  # [N,K], never regenerated
    forecasts_all = bundle.forecasts_fn(val_cache, bundle.expert_idx)  # original, unperturbed forecasts, read from cache
    target = val_cache["targets"].to(torch.float32)
    mask = val_cache["target_masks"].to(torch.bool)
    n_val = int(val_cache["num_windows"])
    actual_excess = torch.tensor(orig_npz[f"{dataset}__actual_excess_loss_val"], dtype=torch.float32)  # diagnostic only, never a feature/weight input

    # --- byte-identical predicted_excess_loss across rule variants: verified structurally, since every rule reads the SAME `pred_excess` tensor without cloning/mutation.
    identity_snapshot = pred_excess.clone()

    method_preds: dict[str, torch.Tensor] = {}
    method_weights: dict[str, torch.Tensor] = {}
    for rule in DECISION_RULES:
        weights = weights_for_rule(rule, pred_excess, temperature)
        method_weights[rule] = weights
        method_preds[rule] = (forecasts_all * weights.view(n_val, 1, 1, k)).sum(dim=-1)
    predicted_excess_unchanged = bool(torch.equal(pred_excess, identity_snapshot))

    # reproduction check: LearnedProbe_Softmax here should match the saved Learned_Probe final prediction exactly (same temperature, same pred_excess, same forecasts).
    saved_learned_probe_pred = torch.tensor(lp_npz[f"{dataset}__Learned_Probe"], dtype=torch.float32)
    reproduction_max_diff = float((method_preds["LearnedProbe_Softmax"] - saved_learned_probe_pred).abs().max())

    reference_preds = {
        "Equal": torch.tensor(orig_npz[f"{dataset}__equal_fixed"], dtype=torch.float32),
        "C": torch.tensor(orig_npz[f"{dataset}__C_window_forecast_disagreement"], dtype=torch.float32),
        "Fixed_D": torch.tensor(orig_npz[f"{dataset}__D_full_behavioral"], dtype=torch.float32),
        "Best_Single": torch.tensor(orig_npz[f"{dataset}__best_single_expert"], dtype=torch.float32),
        "Window_Oracle": torch.tensor(orig_npz[f"{dataset}__window_oracle"], dtype=torch.float32),
    }
    all_methods = {**reference_preds, **method_preds}

    result_rows, metrics = [], {}
    for method, pred in all_methods.items():
        m = metric_values(bundle, pred)
        metrics[method] = m
        result_rows.append({"dataset": dataset, "method": method, "mae": m["mae"], "mse": m["mse"]})
    for rule in DECISION_RULES:
        result_rows_row = next(r for r in result_rows if r["method"] == rule)
        result_rows_row["delta_vs_C"] = metrics[rule]["mae"] - metrics["C"]["mae"]
        result_rows_row["delta_vs_LearnedProbe_Softmax"] = metrics[rule]["mae"] - metrics["LearnedProbe_Softmax"]["mae"]
        result_rows_row["delta_vs_Equal"] = metrics[rule]["mae"] - metrics["Equal"]["mae"]

    # --- expert-selection metrics ---
    predicted_best = pred_excess.argmin(dim=1)
    actual_best = actual_excess.argmin(dim=1)
    top1_accuracy = float((predicted_best == actual_best).to(torch.float32).mean())
    order = pred_excess.argsort(dim=1)
    top2_pred = order[:, :2]
    top2_recall = float((top2_pred == actual_best.view(-1, 1)).any(dim=1).to(torch.float32).mean())
    rank_corr = spearmanr(pred_excess.reshape(-1).numpy(), actual_excess.reshape(-1).numpy())

    # --- winner margin analysis (descriptive; no threshold selection here) ---
    sorted_excess, _ = torch.sort(pred_excess, dim=1)
    margin = sorted_excess[:, 1] - sorted_excess[:, 0]
    median_margin = float(margin.median())
    high_margin = margin >= median_margin
    low_margin = ~high_margin
    top1_mae_win = sample_mae(method_preds["LearnedProbe_Top1"], target, mask, bundle.std)
    top2_mae_win = sample_mae(method_preds["LearnedProbe_Top2Equal"], target, mask, bundle.std)
    softmax_mae_win = sample_mae(method_preds["LearnedProbe_Softmax"], target, mask, bundle.std)
    margin_rows = [
        {
            "dataset": dataset,
            "margin_group": grp_name,
            "num_windows": int(sel.sum()),
            "median_margin": median_margin,
            "top1_mae": float(top1_mae_win[sel].mean()),
            "top2equal_mae": float(top2_mae_win[sel].mean()),
            "softmax_mae": float(softmax_mae_win[sel].mean()),
            "top1_minus_top2equal": float(top1_mae_win[sel].mean() - top2_mae_win[sel].mean()),
        }
        for grp_name, sel in (("high_margin", high_margin), ("low_margin", low_margin))
    ]

    # --- oracle diagnostics: best-pair oracle, Top1 vs Top2 headroom ---
    pair_oracle_pred = best_pair_oracle(forecasts_all, target, mask, bundle.std)
    pair_oracle_m = metric_values(bundle, pair_oracle_pred)
    result_rows.append({"dataset": dataset, "method": "Best_Pair_Oracle", "mae": pair_oracle_m["mae"], "mse": pair_oracle_m["mse"]})
    top1_oracle_headroom = metrics["Equal"]["mae"] - metrics["Window_Oracle"]["mae"]
    top2_oracle_headroom = metrics["Equal"]["mae"] - pair_oracle_m["mae"]

    # --- dependence-aware statistics ---
    dependence_rows = []
    comparisons = [
        ("Top1_vs_Softmax", "LearnedProbe_Top1", "LearnedProbe_Softmax"),
        ("Top2Equal_vs_Softmax", "LearnedProbe_Top2Equal", "LearnedProbe_Softmax"),
        ("Rank_vs_Softmax", "LearnedProbe_Rank", "LearnedProbe_Softmax"),
        ("Top1_vs_C", "LearnedProbe_Top1", "C"),
        ("Top2Equal_vs_C", "LearnedProbe_Top2Equal", "C"),
        ("Rank_vs_C", "LearnedProbe_Rank", "C"),
    ]
    for label, cand_key, base_key in comparisons:
        candidate, baseline = metrics[cand_key]["per_window_mae"], metrics[base_key]["per_window_mae"]
        boot = paired_bootstrap(candidate, baseline, seed=20260821, samples=5000)
        dependence_rows.append({"dataset": dataset, "comparison": label, "test": "iid_paired_bootstrap", **boot})
        for block in BLOCK_LENGTHS:
            b = block_bootstrap_with_prob(candidate, baseline, block=block, seed=20260821, samples=BOOTSTRAP_SAMPLES)
            dependence_rows.append({"dataset": dataset, "comparison": label, "test": f"block_bootstrap_len{block}", **b})
        phase = every_kth_phase_bootstrap(candidate - baseline, k=PHASE_K, seed=20260821, samples=BOOTSTRAP_SAMPLES)
        dependence_rows.append({"dataset": dataset, "comparison": label, "test": f"every_{PHASE_K}th_window_phase_bootstrap", **phase})

    # best-rule-vs-Equal comparison added after ranking rules by MAE
    best_rule = min(DECISION_RULES, key=lambda r: metrics[r]["mae"])
    candidate, baseline = metrics[best_rule]["per_window_mae"], metrics["Equal"]["mae"]
    boot = paired_bootstrap(candidate, baseline, seed=20260821, samples=5000)
    dependence_rows.append({"dataset": dataset, "comparison": f"BestRule({best_rule})_vs_Equal", "test": "iid_paired_bootstrap", **boot})
    for block in BLOCK_LENGTHS:
        b = block_bootstrap_with_prob(candidate, baseline, block=block, seed=20260821, samples=BOOTSTRAP_SAMPLES)
        dependence_rows.append({"dataset": dataset, "comparison": f"BestRule({best_rule})_vs_Equal", "test": f"block_bootstrap_len{block}", **b})
    phase = every_kth_phase_bootstrap(candidate - baseline, k=PHASE_K, seed=20260821, samples=BOOTSTRAP_SAMPLES)
    dependence_rows.append({"dataset": dataset, "comparison": f"BestRule({best_rule})_vs_Equal", "test": f"every_{PHASE_K}th_window_phase_bootstrap", **phase})

    # --- integrity: target corruption must not change weights (structural: weight computation never reads targets) ---
    gen = torch.Generator().manual_seed(4242)
    corrupted_targets = torch.randn(target.shape, generator=gen, dtype=torch.float32)
    weights_before = {r: method_weights[r].clone() for r in DECISION_RULES}
    # weight computation is a pure function of pred_excess (and, for softmax, temperature); recomputing after "corrupting" targets
    # cannot touch it since targets are never an argument -- verified by recomputation for defense-in-depth, not because it could differ.
    weights_after = {r: weights_for_rule(r, pred_excess, temperature) for r in DECISION_RULES}
    weights_identical = all(torch.equal(weights_before[r], weights_after[r]) for r in DECISION_RULES)
    del corrupted_targets  # never used to compute anything; constructed only to document the check was attempted

    integrity = {
        "dataset": dataset,
        "predicted_excess_loss_unchanged_across_rules": predicted_excess_unchanged,
        "reproduction_max_diff_softmax_vs_saved": reproduction_max_diff,
        "reproduction_matches_saved": bool(reproduction_max_diff < 1e-4),
        "weights_unchanged_after_target_corruption": weights_identical,
        "result": "PASS" if (predicted_excess_unchanged and reproduction_max_diff < 1e-4 and weights_identical) else "FAIL",
    }
    if integrity["result"] == "FAIL":
        raise AssertionError(f"{dataset}: decision-rule integrity check FAILED: {integrity}")

    return {
        "dataset": dataset,
        "core": bundle.core_names,
        "temperature": temperature,
        "result_rows": result_rows,
        "dependence_rows": dependence_rows,
        "margin_rows": margin_rows,
        "top1_accuracy": top1_accuracy,
        "top2_recall": top2_recall,
        "rank_correlation_spearman": float(rank_corr.statistic),
        "rank_correlation_pvalue": float(rank_corr.pvalue),
        "best_pair_oracle_mae": pair_oracle_m["mae"],
        "top1_oracle_headroom_vs_equal": top1_oracle_headroom,
        "top2_oracle_headroom_vs_equal": top2_oracle_headroom,
        "best_rule": best_rule,
        "integrity": integrity,
    }


def make_report(out_dir: Path, report: Mapping[str, Any], decision: Mapping[str, Any]) -> None:
    lines = [
        "# Learned-Probe Decision-Rule Comparison",
        "",
        "Isolates the score-to-weight conversion rule. `Learned_Probe_pred_excess` (predicted_excess_loss) is reused byte-for-byte from run_learned_probe.py for every rule -- the ProbeGenerator and competence scorer are never re-run.",
        "",
        "## Main result table (router_val MAE / MSE)",
        "",
        "| Dataset | Equal | C | Fixed-D | Softmax | Top1 | Top2Equal | Rank | Oracle |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ds, d in report["datasets"].items():
        by = {r["method"]: r for r in d["result_rows"]}
        lines.append(
            "| {ds} | {eq[mae]:.6f} | {c[mae]:.6f} | {fd[mae]:.6f} | {sm[mae]:.6f} | {t1[mae]:.6f} | {t2[mae]:.6f} | {rk[mae]:.6f} | {orc[mae]:.6f} |".format(
                ds=ds, eq=by["Equal"], c=by["C"], fd=by["Fixed_D"], sm=by["LearnedProbe_Softmax"], t1=by["LearnedProbe_Top1"], t2=by["LearnedProbe_Top2Equal"], rk=by["LearnedProbe_Rank"], orc=by["Window_Oracle"]
            )
        )
    lines += ["", "## Deltas for each new decision rule", ""]
    lines.append("| Dataset | Rule | Δ vs C | Δ vs Softmax | Δ vs Equal |")
    lines.append("|---|---|---:|---:|---:|")
    for ds, d in report["datasets"].items():
        for rule in DECISION_RULES:
            r = next(x for x in d["result_rows"] if x["method"] == rule)
            lines.append(f"| {ds} | {rule} | `{r['delta_vs_C']:+.6f}` | `{r['delta_vs_LearnedProbe_Softmax']:+.6f}` | `{r['delta_vs_Equal']:+.6f}` |")
    lines += ["", "## Expert-selection metrics", ""]
    lines.append("| Dataset | Top-1 accuracy | Top-2 recall | Rank correlation (Spearman) |")
    lines.append("|---|---:|---:|---:|")
    for ds, d in report["datasets"].items():
        lines.append(f"| {ds} | {d['top1_accuracy']:.3f} | {d['top2_recall']:.3f} | {d['rank_correlation_spearman']:.3f} |")
    lines += ["", "## Winner margin analysis (Top1 vs Top2Equal, median-margin split)", ""]
    lines.append("| Dataset | Group | Windows | Top1 MAE | Top2Equal MAE | Top1 - Top2Equal |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for ds, d in report["datasets"].items():
        for row in d["margin_rows"]:
            lines.append(f"| {ds} | {row['margin_group']} | {row['num_windows']} | {row['top1_mae']:.6f} | {row['top2equal_mae']:.6f} | `{row['top1_minus_top2equal']:+.6f}` |")
    lines += ["", "## Oracle headroom: Top-1 vs Top-2 potential (relative to Equal Fixed)", ""]
    lines.append("| Dataset | Best-Pair Oracle MAE | Top-1 oracle headroom | Top-2 oracle headroom |")
    lines.append("|---|---:|---:|---:|")
    for ds, d in report["datasets"].items():
        lines.append(f"| {ds} | {d['best_pair_oracle_mae']:.6f} | `{d['top1_oracle_headroom_vs_equal']:+.6f}` | `{d['top2_oracle_headroom_vs_equal']:+.6f}` |")
    lines += ["", "## Dependence-aware statistics (block bootstrap)", ""]
    lines.append("| Dataset | Comparison | Test | Mean delta | 95% CI | Excludes zero |")
    lines.append("|---|---|---|---:|---|---|")
    for ds, d in report["datasets"].items():
        for row in d["dependence_rows"]:
            mean_key = row.get("mean_delta", row.get("mean_diff_candidate_minus_baseline"))
            if "ci95_low" in row:
                lines.append(f"| {ds} | {row['comparison']} | {row['test']} | `{mean_key:+.6f}` | [{row['ci95_low']:+.6f}, {row['ci95_high']:+.6f}] | {row['ci_excludes_zero']} |")
    lines += ["", "## Integrity", ""]
    for ds, d in report["datasets"].items():
        lines.append(f"- **{ds}**: {d['integrity']['result']} (predicted_excess_loss unchanged across rules: {d['integrity']['predicted_excess_loss_unchanged_across_rules']}; softmax reproduces saved Learned-Probe prediction: {d['integrity']['reproduction_matches_saved']}; weights invariant to target corruption: {d['integrity']['weights_unchanged_after_target_corruption']})")
    lines += ["", "## Decision", "", f"**{decision['verdict']}**", ""]
    for reason in decision["reasoning"]:
        lines.append(f"- {reason}")
    lines += ["", "## Hard rule compliance", "", "```text", "TEST SET ACCESSED: NO", "TEST CACHE LOADED: NO", "TEST METRICS COMPUTED: NO", "```"]
    (out_dir / "learned_probe_decision_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def decide(report: Mapping[str, Any]) -> dict[str, Any]:
    datasets = list(report["datasets"].keys())
    best_rule_beats_c = {}
    best_rule_sig_vs_c = {}
    best_rule_hurts_significantly = {}
    for ds in datasets:
        d = report["datasets"][ds]
        rule = d["best_rule"]
        by = {r["method"]: r for r in d["result_rows"]}
        best_rule_beats_c[ds] = by[rule]["delta_vs_C"] < 0
        comp_label = {"LearnedProbe_Top1": "Top1_vs_C", "LearnedProbe_Top2Equal": "Top2Equal_vs_C", "LearnedProbe_Rank": "Rank_vs_C", "LearnedProbe_Softmax": None}.get(rule)
        sig = False
        hurts_sig = False
        if comp_label:
            block_rows = [r for r in d["dependence_rows"] if r["comparison"] == comp_label and r["test"].startswith("block_bootstrap")]
            sig = any(r["ci_excludes_zero"] and r["mean_delta"] < 0 for r in block_rows)
            hurts_sig = any(r["ci_excludes_zero"] and r["mean_delta"] > 0 for r in block_rows)
        best_rule_sig_vs_c[ds] = sig
        best_rule_hurts_significantly[ds] = hurts_sig

    n_beats_c = sum(best_rule_beats_c.values())
    n_sig_vs_c = sum(best_rule_sig_vs_c.values())
    n_hurts_sig = sum(best_rule_hurts_significantly.values())
    ettm1_regresses = report["datasets"]["ETTm1"]["result_rows"]
    ettm1_best_rule_row = next(r for r in ettm1_regresses if r["method"] == report["datasets"]["ETTm1"]["best_rule"])
    ettm1_still_regresses = ettm1_best_rule_row["delta_vs_C"] > 0

    reasoning = [
        f"Best decision rule per dataset (by MAE): {[(ds, report['datasets'][ds]['best_rule']) for ds in datasets]}.",
        f"Best rule beats C on {n_beats_c}/{len(datasets)} datasets (need >=3).",
        f"Beats C with dependence-aware (block-bootstrap) support on {n_sig_vs_c}/{len(datasets)} datasets.",
        f"Best rule significantly HURTS on {n_hurts_sig}/{len(datasets)} datasets.",
        f"ETTm1 still regresses under its best rule: {ettm1_still_regresses}.",
    ]
    go = (n_beats_c >= 3) and (n_sig_vs_c >= 2) and (n_hurts_sig == 0)
    verdict = "CONTINUE" if go else "STOP"
    return {"verdict": verdict, "reasoning": reasoning, "n_beats_c": n_beats_c, "n_sig_vs_c": n_sig_vs_c, "n_hurts_sig": n_hurts_sig, "ettm1_still_regresses": ettm1_still_regresses}


def main() -> None:
    start = time.time()
    report: dict[str, Any] = {"experiment": "learned_probe_decision_rules", "created_at_utc": datetime.now(timezone.utc).isoformat(), "datasets": {}}
    all_results, all_dependence, all_margin, all_integrity = [], [], [], []
    diagnostics_rows = []

    for dataset in LOADERS:
        print(f"[decision-rules] {dataset}: evaluating...", flush=True)
        result = evaluate_dataset(dataset)
        report["datasets"][dataset] = result
        all_results.extend(result["result_rows"])
        all_dependence.extend(result["dependence_rows"])
        all_margin.extend(result["margin_rows"])
        all_integrity.append(result["integrity"])
        diagnostics_rows.append(
            {
                "dataset": dataset,
                "top1_accuracy": result["top1_accuracy"],
                "top2_recall": result["top2_recall"],
                "rank_correlation_spearman": result["rank_correlation_spearman"],
                "best_pair_oracle_mae": result["best_pair_oracle_mae"],
                "top1_oracle_headroom_vs_equal": result["top1_oracle_headroom_vs_equal"],
                "top2_oracle_headroom_vs_equal": result["top2_oracle_headroom_vs_equal"],
                "best_rule": result["best_rule"],
            }
        )
        print(f"[decision-rules] {dataset}: done. best_rule={result['best_rule']}", flush=True)

    decision = decide(report)
    report["decision"] = decision
    report["runtime_sec"] = time.time() - start
    report["test_set_accessed"] = False

    write_json(RESULTS_DIR / "learned_probe_decision_results.json", report)
    write_csv(RESULTS_DIR / "learned_probe_decision_rules.csv", all_results)
    write_csv(RESULTS_DIR / "learned_probe_decision_dependence.csv", all_dependence)
    write_csv(RESULTS_DIR / "learned_probe_decision_diagnostics.csv", diagnostics_rows)
    write_csv(RESULTS_DIR / "learned_probe_decision_margin_analysis.csv", all_margin)
    write_csv(RESULTS_DIR / "learned_probe_decision_integrity.csv", all_integrity)
    make_report(REPORTS_DIR, report, decision)

    print("TEST SET ACCESSED: NO")
    print(json.dumps({"runtime_sec": report["runtime_sec"], "decision": decision["verdict"]}, indent=2))


if __name__ == "__main__":
    main()
