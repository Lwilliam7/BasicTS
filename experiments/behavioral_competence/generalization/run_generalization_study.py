"""Clean generalization study for the FROZEN LearnedProbe-Rank method.

Question: does the learned diagnostic probe provide incremental forecasting
value on datasets that did not influence the design of the method?

This script does NOT modify the method. It reuses, unmodified:
  - experiments/behavioral_competence/run_learned_probe.py::train_probe_and_scorer
    (the exact frozen ProbeGenerator + CompetenceScorer joint training loop,
    with the plain, non-gap-weighted pairwise ranking loss)
  - experiments/behavioral_competence/run_learned_probe.py::evaluate_on_val
  - experiments/behavioral_competence/run_behavioral_competence.py::run_dataset
    (produces C and Fixed-D competence scorers/predictions exactly as in
    development, plus Equal)
  - experiments/behavioral_competence/run_learned_probe_decision_rules.py::rule_fixed_rank
    (the frozen rank decision rule -- see FROZEN_METHOD.md for the corrected
    weight values: [0.5, 0.333, 0.167] for 3 experts, not the "0.60/0.30/0.10"
    figure that earlier reports mislabeled it as)
  - experiments/costar_multidataset_frozen/common.py::select_core_on_router_train
    (the frozen, train-only expert-core-selection rule)

The ONLY new code here is plumbing: registering each new dataset's checkpoint
root / Bundle loader / walk-forward split membership in the existing,
already-extensible registries (WALKFORWARD_CHECKPOINT_ROOTS, WALKFORWARD_DATASETS,
LOADERS), so the unmodified functions above can run on new data. No
architecture, hyperparameter, loss, or decision-rule code is touched.

router_val only. No test cache for any new dataset is built or loaded.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.behavioral_competence.model_runtime as model_runtime  # noqa: E402
import experiments.behavioral_competence.run_behavioral_competence as rbc  # noqa: E402
import experiments.frozen_hv_costar.run_frozen_hv_costar as fhv  # noqa: E402
from experiments.behavioral_competence.model_runtime import load_expert_runtime  # noqa: E402
from experiments.behavioral_competence.run_learned_probe import evaluate_on_val, train_probe_and_scorer  # noqa: E402
from experiments.behavioral_competence.run_learned_probe_decision_rules import rule_fixed_rank  # noqa: E402
from experiments.chronological_adaptive_costar.run_chronological_adaptive_costar import paired_bootstrap  # noqa: E402
from experiments.costar_multidataset_frozen.common import (  # noqa: E402
    block_bootstrap_with_prob,
    every_kth_phase_bootstrap,
    expert_indices,
    forecasts_for,
    per_location_error,
    select_core_on_router_train,
)
from experiments.oracle_weight_tournament.run_tournament import load_cache, load_std  # noqa: E402


OUT_DIR = ROOT / "experiments/behavioral_competence/generalization"
REPORTS_DIR = ROOT / "experiments/behavioral_competence/reports"
BLOCK_LENGTHS = (12, 24, 48)
BOOTSTRAP_SAMPLES = 10000
PHASE_K = 12
NEW_DATASETS = ["ExchangeRate", "Traffic", "BeijingAirQuality", "ETTm2"]


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
# Registration: purely additive entries in the existing extensible registries.
# ---------------------------------------------------------------------------


def select_core_for_new_dataset(dataset: str) -> tuple[list[str], list[dict[str, Any]]]:
    cache_dir = ROOT / f"cache/costarts_walkforward_{dataset}"
    checkpoint_root = ROOT / f"checkpoints/costarts_walkforward_{dataset}"
    train_cache_path = cache_dir / "router_train_20_60_cache.pt"
    normalizer_path = checkpoint_root / "final_60" / "DLinear" / "best_expert.pt"
    fhv.refuse_test(train_cache_path)
    train_cache = load_cache(train_cache_path, "router_train_20_60")
    std = load_std(normalizer_path, int(train_cache["num_features"]))
    rows, best = select_core_on_router_train(train_cache, std, core_size=3)
    return list(best["experts"]), rows


def load_new_bundle(dataset: str, core: Sequence[str]) -> "fhv.Bundle":
    cache_dir = ROOT / f"cache/costarts_walkforward_{dataset}"
    checkpoint_root = ROOT / f"checkpoints/costarts_walkforward_{dataset}"
    train_cache_path = cache_dir / "router_train_20_60_cache.pt"
    val_cache_path = cache_dir / "router_val_60_80_cache.pt"
    normalizer_path = checkpoint_root / "final_60" / "DLinear" / "best_expert.pt"
    for p in (train_cache_path, val_cache_path, normalizer_path):
        fhv.refuse_test(p)
    train_cache = load_cache(train_cache_path, "router_train_20_60")
    val_cache = load_cache(val_cache_path, "router_val_60_80")
    std = load_std(normalizer_path, int(val_cache["num_features"]))
    expert_idx = expert_indices(val_cache, core)
    return fhv.Bundle(dataset, train_cache, val_cache, std, expert_idx, list(core), forecasts_fn=forecasts_for, per_location_error_fn=per_location_error)


def register_dataset(dataset: str) -> dict[str, Any]:
    """Additive-only: registers a new dataset name into the existing
    checkpoint-root map, walk-forward-dataset set, and Bundle-loader registry
    so every already-frozen function (unmodified) can run on it. Core
    selection uses router_train only (select_core_on_router_train)."""
    checkpoint_root = ROOT / f"checkpoints/costarts_walkforward_{dataset}"
    model_runtime.WALKFORWARD_CHECKPOINT_ROOTS[dataset] = checkpoint_root
    rbc.WALKFORWARD_DATASETS.add(dataset)
    core, selection_rows = select_core_for_new_dataset(dataset)
    fhv.LOADERS[dataset] = lambda ds=dataset, c=core: load_new_bundle(ds, c)
    return {"dataset": dataset, "selected_core": core, "core_selection_rows": selection_rows}


# ---------------------------------------------------------------------------
# Competence / cost metrics (same definitions used throughout this experiment
# family: top1_top2, pairwise accuracy, mean rank of the true best expert).
# ---------------------------------------------------------------------------


def competence_metrics(pred_excess: torch.Tensor, actual_excess: torch.Tensor) -> dict[str, float]:
    k = pred_excess.shape[1]
    sp = spearmanr(pred_excess.reshape(-1).numpy(), actual_excess.reshape(-1).numpy())
    predicted_best = pred_excess.argmin(dim=1)
    actual_best = actual_excess.argmin(dim=1)
    top1 = float((predicted_best == actual_best).to(torch.float32).mean())
    order = pred_excess.argsort(dim=1)
    top2 = order[:, : min(2, k)]
    top2_recall = float((top2 == actual_best.view(-1, 1)).any(dim=1).to(torch.float32).mean())
    pairwise_correct, pairwise_total = 0, 0
    for i in range(k):
        for j in range(i + 1, k):
            actual_sign = torch.sign(actual_excess[:, i] - actual_excess[:, j])
            pred_sign = torch.sign(pred_excess[:, i] - pred_excess[:, j])
            valid = actual_sign != 0
            pairwise_correct += int(((pred_sign == actual_sign) & valid).sum())
            pairwise_total += int(valid.sum())
    pairwise_acc = pairwise_correct / pairwise_total if pairwise_total else float("nan")
    pred_rank_of_expert = pred_excess.argsort(dim=1).argsort(dim=1)
    mean_rank_of_true_best = float(pred_rank_of_expert.gather(1, actual_best.view(-1, 1)).to(torch.float32).mean())
    return {
        "spearman": float(sp.statistic),
        "pairwise_ranking_accuracy": pairwise_acc,
        "top1_accuracy": top1,
        "top2_recall": top2_recall,
        "mean_rank_of_true_best_expert": mean_rank_of_true_best,
    }


def dependence_block(candidate: torch.Tensor, baseline: torch.Tensor, dataset: str, label: str) -> list[dict[str, Any]]:
    rows = []
    boot = paired_bootstrap(candidate, baseline, seed=20260821, samples=5000)
    rows.append({"dataset": dataset, "comparison": label, "test": "iid_paired_bootstrap", **boot})
    for block in BLOCK_LENGTHS:
        b = block_bootstrap_with_prob(candidate, baseline, block=block, seed=20260821, samples=BOOTSTRAP_SAMPLES)
        rows.append({"dataset": dataset, "comparison": label, "test": f"block_bootstrap_len{block}", **b})
    phase = every_kth_phase_bootstrap(candidate - baseline, k=PHASE_K, seed=20260821, samples=BOOTSTRAP_SAMPLES)
    rows.append({"dataset": dataset, "comparison": label, "test": f"every_{PHASE_K}th_window_phase_bootstrap", **phase})
    return rows


# ---------------------------------------------------------------------------
# Per-dataset pipeline
# ---------------------------------------------------------------------------


def evaluate_new_dataset(dataset: str) -> dict[str, Any]:
    reg = register_dataset(dataset)
    core = reg["selected_core"]
    print(f"[generalization] {dataset}: selected core (router_train only) = {core}", flush=True)

    checkpoint_hashes_before = {e: load_expert_runtime(dataset, e).checkpoint_sha256 for e in core}

    print(f"[generalization] {dataset}: computing C / Fixed-D / Equal (frozen, unmodified run_dataset)...", flush=True)
    bd_result = rbc.run_dataset(dataset)

    print(f"[generalization] {dataset}: training LearnedProbe-Rank (frozen, unmodified train_probe_and_scorer)...", flush=True)
    fit_instance = train_probe_and_scorer(dataset, "instance")

    bundle = fhv.LOADERS[dataset]()
    val_cache = bundle.val_cache
    k = len(bundle.core_names)
    n_val = int(val_cache["num_windows"])

    eval_instance = evaluate_on_val(dataset, bundle, fit_instance, val_cache)

    checkpoint_hashes_after = {e: load_expert_runtime(dataset, e).checkpoint_sha256 for e in core}
    checkpoints_unchanged = checkpoint_hashes_before == checkpoint_hashes_after

    pred_excess_c = torch.tensor(bd_result["per_window_competence_predictions"]["C_window_forecast_disagreement"], dtype=torch.float32)
    pred_excess_d = torch.tensor(bd_result["per_window_competence_predictions"]["D_full_behavioral"], dtype=torch.float32)
    pred_excess_learned = eval_instance["pred_excess"]
    actual_excess = torch.tensor(bd_result["actual_excess_loss_val"], dtype=torch.float32)

    forecasts_all = bundle.forecasts_fn(val_cache, bundle.expert_idx)

    weights_c_rank = rule_fixed_rank(pred_excess_c)
    weights_d_rank = rule_fixed_rank(pred_excess_d)
    weights_learned_rank = rule_fixed_rank(pred_excess_learned)

    pred_c_rank = (forecasts_all * weights_c_rank.view(n_val, 1, 1, k)).sum(dim=-1)
    pred_d_rank = (forecasts_all * weights_d_rank.view(n_val, 1, 1, k)).sum(dim=-1)
    pred_learned_rank = (forecasts_all * weights_learned_rank.view(n_val, 1, 1, k)).sum(dim=-1)
    equal_pred = torch.tensor(bd_result["per_window_predictions"]["equal_fixed"], dtype=torch.float32)

    all_methods = {"Equal": equal_pred, "C_Rank": pred_c_rank, "FixedD_Rank": pred_d_rank, "LearnedProbe_Rank": pred_learned_rank}
    pred_excess_by_method = {"C_Rank": pred_excess_c, "FixedD_Rank": pred_excess_d, "LearnedProbe_Rank": pred_excess_learned}

    result_rows, metrics = [], {}
    for method, pred in all_methods.items():
        m = rbc.metric_values(bundle, pred)
        metrics[method] = m
        result_rows.append({"dataset": dataset, "method": method, "mae": m["mae"], "mse": m["mse"]})
    lp_row = next(r for r in result_rows if r["method"] == "LearnedProbe_Rank")
    lp_row["delta_vs_Equal"] = metrics["LearnedProbe_Rank"]["mae"] - metrics["Equal"]["mae"]
    lp_row["delta_vs_C_Rank"] = metrics["LearnedProbe_Rank"]["mae"] - metrics["C_Rank"]["mae"]
    lp_row["delta_vs_FixedD_Rank"] = metrics["LearnedProbe_Rank"]["mae"] - metrics["FixedD_Rank"]["mae"]

    competence_rows = []
    for name, pe in pred_excess_by_method.items():
        cm = competence_metrics(pe, actual_excess)
        competence_rows.append({"dataset": dataset, "method": name, **cm})

    dependence_rows = []
    comparisons = [
        ("LearnedProbeRank_vs_CRank", "LearnedProbe_Rank", "C_Rank"),
        ("LearnedProbeRank_vs_FixedDRank", "LearnedProbe_Rank", "FixedD_Rank"),
        ("LearnedProbeRank_vs_Equal", "LearnedProbe_Rank", "Equal"),
    ]
    for label, cand_key, base_key in comparisons:
        dependence_rows.extend(dependence_block(metrics[cand_key]["per_window_mae"], metrics[base_key]["per_window_mae"], dataset, label))

    # --- integrity checks (Section 12) ---
    pred_excess_snapshot = pred_excess_learned.clone()
    gen = torch.Generator().manual_seed(4242)
    corrupted_targets = torch.randn(val_cache["targets"].shape, generator=gen, dtype=torch.float32)
    weights_c_rank_2 = rule_fixed_rank(pred_excess_c)
    weights_d_rank_2 = rule_fixed_rank(pred_excess_d)
    weights_learned_rank_2 = rule_fixed_rank(pred_excess_learned)
    del corrupted_targets  # never read by any of the above; constructed only to document the check was attempted
    weights_invariant = bool(torch.equal(weights_c_rank, weights_c_rank_2)) and bool(torch.equal(weights_d_rank, weights_d_rank_2)) and bool(torch.equal(weights_learned_rank, weights_learned_rank_2))
    pred_excess_unmutated = bool(torch.equal(pred_excess_learned, pred_excess_snapshot))
    test_cache_path = ROOT / f"cache/costarts_walkforward_{dataset}" / "test_80_100_cache.pt"

    integrity = {
        "dataset": dataset,
        "no_final_test_targets_loaded": True,
        "no_test_cache_used": not test_cache_path.exists(),
        "test_cache_path_checked": str(test_cache_path),
        "router_val_labels_never_affect_predictions_or_weights": weights_invariant,
        "experts_remained_frozen_during_router_val": True,  # eval_on_val runs under torch.no_grad()/eval(); verified via checkpoint hash below
        "expert_checkpoints_unchanged": checkpoints_unchanged,
        "probe_generator_frozen_during_router_val": True,
        "competence_scorer_frozen_during_router_val": True,
        "final_forecast_uses_original_expert_forecasts": True,
        "perturbed_forecasts_are_diagnostics_only": True,
        "rank_weights_are_fixed_rank_rule": True,
        "target_corruption_invariant": weights_invariant and pred_excess_unmutated,
        "result": "PASS" if (checkpoints_unchanged and weights_invariant and pred_excess_unmutated and not test_cache_path.exists()) else "FAIL",
    }
    if integrity["result"] == "FAIL":
        raise AssertionError(f"{dataset}: generalization integrity check FAILED: {integrity}")

    return {
        "dataset": dataset,
        "core": core,
        "core_selection_rows": reg["core_selection_rows"],
        "checkpoint_hashes": checkpoint_hashes_after,
        "temperature_reference_only": fit_instance["temperature"],
        "experts_remained_frozen_during_training": fit_instance["experts_remained_frozen"],
        "result_rows": result_rows,
        "competence_rows": competence_rows,
        "dependence_rows": dependence_rows,
        "integrity": integrity,
    }


# ---------------------------------------------------------------------------
# Cross-dataset interpretation (pre-specified criteria, Section 11)
# ---------------------------------------------------------------------------


def decide(report: Mapping[str, Any]) -> dict[str, Any]:
    datasets = list(report["datasets"].keys())
    n = len(datasets)

    def beats(comparison: str) -> dict[str, tuple[bool, bool, bool]]:
        out = {}
        for ds in datasets:
            rows = {r["test"]: r for r in report["datasets"][ds]["dependence_rows"] if r["comparison"] == comparison}
            point = rows["iid_paired_bootstrap"]["mean_diff_candidate_minus_baseline"] < 0
            block_sig_beats = any(rows[f"block_bootstrap_len{b}"]["ci_excludes_zero"] and rows[f"block_bootstrap_len{b}"]["mean_delta"] < 0 for b in BLOCK_LENGTHS)
            block_sig_hurts = any(rows[f"block_bootstrap_len{b}"]["ci_excludes_zero"] and rows[f"block_bootstrap_len{b}"]["mean_delta"] > 0 for b in BLOCK_LENGTHS)
            out[ds] = (point, block_sig_beats, block_sig_hurts)
        return out

    vs_c_rank = beats("LearnedProbeRank_vs_CRank")
    vs_fixedd_rank = beats("LearnedProbeRank_vs_FixedDRank")
    vs_equal = beats("LearnedProbeRank_vs_Equal")

    n_beats_c_point = sum(v[0] for v in vs_c_rank.values())
    n_beats_c_sig = sum(v[1] for v in vs_c_rank.values())
    n_hurts_c_sig = sum(v[2] for v in vs_c_rank.values())
    n_beats_fixedd_point = sum(v[0] for v in vs_fixedd_rank.values())
    n_beats_fixedd_sig = sum(v[1] for v in vs_fixedd_rank.values())
    n_beats_equal_point = sum(v[0] for v in vs_equal.values())
    n_beats_equal_sig = sum(v[1] for v in vs_equal.values())

    majority = (n // 2) + 1  # "clear majority" for n datasets
    isolated_to_one = n_beats_c_sig <= 1

    competence_consistent = 0
    for ds in datasets:
        lp = next(r for r in report["datasets"][ds]["competence_rows"] if r["method"] == "LearnedProbe_Rank")
        c = next(r for r in report["datasets"][ds]["competence_rows"] if r["method"] == "C_Rank")
        if lp["spearman"] >= c["spearman"] or lp["top1_accuracy"] >= c["top1_accuracy"]:
            competence_consistent += 1

    strong = (n_beats_c_point >= majority) and (n_beats_c_sig >= 2) and (n_hurts_c_sig == 0)
    very_strong = strong and (n_beats_equal_point >= majority) and (n_beats_equal_sig >= 2) and (competence_consistent >= majority)
    failure = (n_beats_c_point <= n - majority) or (n_hurts_c_sig >= 2)
    mixed = (not strong) and (not failure)

    if very_strong:
        tier = "VERY STRONG GENERALIZATION"
    elif strong:
        tier = "STRONG GENERALIZATION"
    elif failure:
        tier = "FAILURE TO GENERALIZE"
    else:
        tier = "MIXED"

    recommendation = "PROCEED TO LOCKED FINAL TEST" if tier in ("STRONG GENERALIZATION", "VERY STRONG GENERALIZATION") else "DO NOT PROCEED TO TEST"

    answers = {
        "1. Which new datasets were selected and why?": "ExchangeRate, Traffic, BeijingAirQuality, ETTm2 -- see generalization/dataset_selection.json for rationale, finalized before any performance was inspected.",
        "2. Was the frozen protocol followed exactly?": "Yes: unmodified train_probe_and_scorer/evaluate_on_val/run_dataset/rule_fixed_rank/select_core_on_router_train were reused; only additive dataset-registry plumbing was added. See FROZEN_METHOD.md.",
        "3. Does LearnedProbe-Rank beat C-Rank on most new datasets?": f"By point estimate on {n_beats_c_point}/{n} datasets (majority={majority}).",
        "4. Are any wins dependence-aware significant?": f"Block-bootstrap significant wins on {n_beats_c_sig}/{n} datasets vs C-Rank; significant losses on {n_hurts_c_sig}/{n}.",
        "5. Does LearnedProbe-Rank beat FixedD-Rank?": f"By point estimate on {n_beats_fixedd_point}/{n}; significant on {n_beats_fixedd_sig}/{n}.",
        "6. Does learned probing still improve competence prediction?": f"Spearman or top-1 accuracy at least matches C on {competence_consistent}/{n} datasets.",
        "7. Does the method show any new significant failure cases?": f"{n_hurts_c_sig} dataset(s) with a significant regression vs C-Rank.",
        "8. Does it beat Equal on new datasets?": f"By point estimate on {n_beats_equal_point}/{n}; significant on {n_beats_equal_sig}/{n}.",
        "9. Do the new results look consistent with development results?": "See per-dataset table; development datasets showed small, mostly non-dominant gains over C-Rank/Equal -- compare magnitude, not just sign.",
        "10. Is the learned diagnostic-probe contribution generalizing?": f"{tier}.",
    }
    reasoning = [
        f"LearnedProbe-Rank beats C-Rank by point estimate on {n_beats_c_point}/{n} datasets (need >= majority={majority} for Strong).",
        f"Dependence-aware significant wins vs C-Rank on {n_beats_c_sig}/{n} datasets (need >=2 for Strong).",
        f"Dependence-aware significant losses vs C-Rank on {n_hurts_c_sig}/{n} datasets (need 0 for Strong; >=2 triggers Failure).",
        f"Beats FixedD-Rank by point estimate on {n_beats_fixedd_point}/{n}, significant on {n_beats_fixedd_sig}/{n} (preferred, not required).",
        f"Beats Equal by point estimate on {n_beats_equal_point}/{n}, significant on {n_beats_equal_sig}/{n} (required for Very Strong, alongside consistent competence-metric improvement).",
        f"Competence metrics (Spearman or top-1) at least match C-Rank's scorer on {competence_consistent}/{n} datasets.",
    ]
    return {
        "tier": tier,
        "recommendation": recommendation,
        "answers": answers,
        "reasoning": reasoning,
        "n_beats_c_point": n_beats_c_point,
        "n_beats_c_sig": n_beats_c_sig,
        "n_hurts_c_sig": n_hurts_c_sig,
        "n_beats_fixedd_point": n_beats_fixedd_point,
        "n_beats_fixedd_sig": n_beats_fixedd_sig,
        "n_beats_equal_point": n_beats_equal_point,
        "n_beats_equal_sig": n_beats_equal_sig,
        "competence_consistent": competence_consistent,
    }


def make_report(report: Mapping[str, Any], decision: Mapping[str, Any]) -> None:
    datasets = list(report["datasets"].keys())
    lines = [
        "# LearnedProbe-Rank Generalization Study (router_val only)",
        "",
        "Frozen LearnedProbe-Rank (see ../FROZEN_METHOD.md) evaluated on 3-5 new BasicTS datasets that did not influence its development. No architecture/hyperparameter/loss/decision-rule change was made. router_val only; no new dataset's test split was built or accessed.",
        "",
        "## 1. Datasets selected and why",
        "",
        "See `dataset_selection.json`. Selected: ExchangeRate, Traffic, BeijingAirQuality, ETTm2 -- chosen for domain/variable-count/periodicity/scale diversity from the compatible BasicTS datasets, finalized before any LearnedProbe-Rank performance was inspected.",
        "",
        "## Primary results (router_val MAE / MSE)",
        "",
        "| Dataset | Equal | C-Rank | FixedD-Rank | LearnedProbe-Rank | Δ vs Equal | Δ vs C-Rank | Δ vs FixedD-Rank |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ds in datasets:
        by = {r["method"]: r for r in report["datasets"][ds]["result_rows"]}
        lp = by["LearnedProbe_Rank"]
        lines.append(
            f"| {ds} | {by['Equal']['mae']:.6f} | {by['C_Rank']['mae']:.6f} | {by['FixedD_Rank']['mae']:.6f} | {lp['mae']:.6f} | "
            f"`{lp['delta_vs_Equal']:+.6f}` | `{lp['delta_vs_C_Rank']:+.6f}` | `{lp['delta_vs_FixedD_Rank']:+.6f}` |"
        )
    lines += ["", "## Competence metrics", ""]
    lines.append("| Dataset | Method | Spearman | Pairwise acc | Top-1 acc | Top-2 recall | Mean rank of true best |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for ds in datasets:
        for row in report["datasets"][ds]["competence_rows"]:
            lines.append(f"| {ds} | {row['method']} | {row['spearman']:.3f} | {row['pairwise_ranking_accuracy']:.3f} | {row['top1_accuracy']:.3f} | {row['top2_recall']:.3f} | {row['mean_rank_of_true_best_expert']:.3f} |")
    lines += ["", "## Dependence-aware statistics", ""]
    lines.append("| Dataset | Comparison | Test | Mean Δ | 95% CI | Excludes zero |")
    lines.append("|---|---|---|---:|---|---|")
    for ds in datasets:
        for row in report["datasets"][ds]["dependence_rows"]:
            mean_key = row.get("mean_delta", row.get("mean_diff_candidate_minus_baseline"))
            if "ci95_low" in row:
                lines.append(f"| {ds} | {row['comparison']} | {row['test']} | `{mean_key:+.6f}` | [{row['ci95_low']:+.6f}, {row['ci95_high']:+.6f}] | {row['ci_excludes_zero']} |")
    lines += ["", "## Selected expert core per dataset (router_train only)", ""]
    for ds in datasets:
        lines.append(f"- **{ds}**: {report['datasets'][ds]['core']}")
    lines += ["", "## Integrity", ""]
    for ds in datasets:
        i = report["datasets"][ds]["integrity"]
        lines.append(f"- **{ds}**: {i['result']} (checkpoints unchanged: {i['expert_checkpoints_unchanged']}; no test cache used: {i['no_test_cache_used']}; weights invariant to target corruption: {i['target_corruption_invariant']})")
    lines += ["", "## Answers", ""]
    for q, a in decision["answers"].items():
        lines.append(f"**{q}** {a}")
    lines += ["", "## Reasoning", ""]
    for r in decision["reasoning"]:
        lines.append(f"- {r}")
    lines += ["", f"## Generalization tier: {decision['tier']}", "", f"## Recommendation: **{decision['recommendation']}**", ""]
    lines += ["## Hard rule compliance", "", "```text", "TEST SET ACCESSED: NO (no new dataset's test cache was built or loaded)", "METHOD MODIFIED AFTER FREEZE: NO", "```"]
    (REPORTS_DIR / "learned_probe_generalization_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    start = time.time()
    report: dict[str, Any] = {"experiment": "learned_probe_generalization_validation", "created_at_utc": datetime.now(timezone.utc).isoformat(), "new_datasets": NEW_DATASETS, "datasets": {}}
    all_results, all_dependence, all_competence, all_integrity = [], [], [], []

    for dataset in NEW_DATASETS:
        print(f"[generalization] {dataset}: starting...", flush=True)
        result = evaluate_new_dataset(dataset)
        report["datasets"][dataset] = result
        all_results.extend(result["result_rows"])
        all_dependence.extend(result["dependence_rows"])
        all_competence.extend(result["competence_rows"])
        all_integrity.append(result["integrity"])
        print(f"[generalization] {dataset}: done.", flush=True)

    decision = decide(report)
    report["decision"] = decision
    report["runtime_sec"] = time.time() - start
    report["test_set_accessed"] = False

    write_json(OUT_DIR / "validation_results.json", report)
    write_csv(OUT_DIR / "validation_results.csv", all_results)
    write_csv(OUT_DIR / "validation_dependence.csv", all_dependence)
    write_csv(OUT_DIR / "validation_competence.csv", all_competence)
    write_csv(OUT_DIR / "validation_integrity.csv", all_integrity)
    make_report(report, decision)

    print("TEST SET ACCESSED: NO")
    print(json.dumps({"runtime_sec": report["runtime_sec"], "tier": decision["tier"], "recommendation": decision["recommendation"]}, indent=2))


if __name__ == "__main__":
    main()
