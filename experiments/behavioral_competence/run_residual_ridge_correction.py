"""Residual-correction follow-up to the behavioral-competence proof of concept.

Tests whether behavioral perturbation features have value as a *correction*
on top of the already-trained C scorer (window + forecast + disagreement),
rather than being retrained jointly with everything else inside D. Motivated
by the D-vs-C result: on Electricity, individual behavioral features showed
the strongest out-of-sample correlations of any dataset, yet full D still
lost to C -- suggesting the 35-feature joint retrain may be diluting/
overfitting a real signal rather than the signal being absent.

    1. Train C normally: window + forecast + disagreement -> predicted excess loss.
    2. On router_train: residual = actual excess loss - C prediction.
    3. Train a simple Ridge model: behavioral features -> residual.
    4. New prediction: C prediction + Ridge residual correction.
    5. Evaluate on router_val.

Reuses the exact perturbation cache, feature engineering, and C-scorer
training already built in run_behavioral_competence.py -- no expert is
retrained, no cache is modified. Writes its own output files; does not
overwrite the original experiment's results.
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
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.behavioral_competence.common import competence_to_weights, train_competence_scorer  # noqa: E402
from experiments.behavioral_competence.run_behavioral_competence import (  # noqa: E402
    BLOCK_LENGTHS,
    BOOTSTRAP_SAMPLES,
    INTERNAL_VAL_FRACTION,
    PHASE_K,
    RESULTS_DIR as ORIGINAL_RESULTS_DIR,
    build_feature_bundle,
    build_perturbation_cache_router_train,
    compute_excess_loss,
    flatten_window_expert,
    load_expert_runtime,
    raw_history_cache,
    router_train_block_split,
)
from experiments.chronological_adaptive_costar.run_chronological_adaptive_costar import paired_bootstrap  # noqa: E402
from experiments.costar_multidataset_frozen.common import block_bootstrap_with_prob, every_kth_phase_bootstrap  # noqa: E402
from experiments.frozen_hv_costar.run_frozen_hv_costar import LOADERS, metric_values  # noqa: E402
from experiments.oracle_weight_tournament.run_tournament import sample_mae  # noqa: E402


OUT_DIR = ROOT / "experiments/behavioral_competence"
RESULTS_DIR = OUT_DIR / "results"
REPORTS_DIR = OUT_DIR / "reports"
RIDGE_ALPHA_GRID = (0.1, 1.0, 10.0, 100.0)


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


def select_ridge_alpha(x_train: np.ndarray, y_train: np.ndarray, x_internal_val: np.ndarray, y_internal_val: np.ndarray) -> float:
    best_alpha, best_mse = RIDGE_ALPHA_GRID[0], float("inf")
    for alpha in RIDGE_ALPHA_GRID:
        model = Ridge(alpha=alpha)
        model.fit(x_train, y_train)
        pred = model.predict(x_internal_val)
        mse = float(np.mean((pred - y_internal_val) ** 2))
        if mse < best_mse:
            best_mse, best_alpha = mse, alpha
    return best_alpha


def run_dataset(dataset: str) -> dict[str, Any]:
    bundle = LOADERS[dataset]()
    train_cache, val_cache = bundle.train_cache, bundle.val_cache
    k = len(bundle.core_names)

    split_boundary = router_train_block_split(dataset, train_cache)
    val_runtimes = {e: load_expert_runtime(dataset, e) for e in bundle.core_names}
    reference_runtime = val_runtimes[bundle.core_names[0]]
    train_cache_raw = raw_history_cache(dataset, train_cache, reference_runtime.mean, reference_runtime.std)
    val_cache_raw = raw_history_cache(dataset, val_cache, reference_runtime.mean, reference_runtime.std)

    train_forecasts_all = bundle.forecasts_fn(train_cache, bundle.expert_idx)
    val_forecasts_all = bundle.forecasts_fn(val_cache, bundle.expert_idx)

    train_payloads, val_payloads = {}, {}
    for local_i, expert_name in enumerate(bundle.core_names):
        train_payloads[expert_name] = build_perturbation_cache_router_train(dataset, expert_name, train_cache_raw, train_forecasts_all[..., local_i], split_boundary)
        from experiments.behavioral_competence.run_behavioral_competence import build_perturbation_cache

        val_payloads[expert_name] = build_perturbation_cache(dataset, "router_val", expert_name, val_cache_raw["histories"].to(torch.float32), val_cache["absolute_window_starts"], val_runtimes[expert_name], val_forecasts_all[..., local_i])

    train_features = build_feature_bundle(bundle, train_cache_raw, train_payloads)
    val_features = build_feature_bundle(bundle, val_cache_raw, val_payloads)

    excess_loss_train, _ = compute_excess_loss(train_cache, train_forecasts_all, bundle.std)
    excess_loss_val, _ = compute_excess_loss(val_cache, val_forecasts_all, bundle.std)  # diagnostic only
    excess_loss_train_flat = excess_loss_train.reshape(-1)

    n_train = int(train_cache["num_windows"])
    split_point = int(round(n_train * (1 - INTERNAL_VAL_FRACTION)))
    window_id_train = torch.arange(0, split_point)
    window_id_internal_val = torch.arange(split_point, n_train)
    row_window_id_train = (window_id_train.view(-1, 1) * k + torch.arange(k).view(1, -1)).reshape(-1)
    row_window_id_internal_val = (window_id_internal_val.view(-1, 1) * k + torch.arange(k).view(1, -1)).reshape(-1)

    # --- Step 1: train C normally ---
    feats_train_flat_c = flatten_window_expert(train_features.features_for("C_window_forecast_disagreement"))
    fit_c = train_competence_scorer(feats_train_flat_c, excess_loss_train_flat, n_train_windows=split_point * k, window_id_train=row_window_id_train, window_id_internal_val=row_window_id_internal_val)

    # --- Step 2: residual on router_train ---
    c_pred_train_flat = fit_c.predict(feats_train_flat_c).numpy()
    residual_train_flat = excess_loss_train_flat.numpy() - c_pred_train_flat

    # --- Step 3: simple Ridge, behavioral features -> residual, alpha chosen on router_train's own chronological split ---
    feats_train_flat_d_only = flatten_window_expert(train_features.group_d).numpy()
    x_train_rows = feats_train_flat_d_only[row_window_id_train.numpy()]
    y_train_rows = residual_train_flat[row_window_id_train.numpy()]
    x_internal_val_rows = feats_train_flat_d_only[row_window_id_internal_val.numpy()]
    y_internal_val_rows = residual_train_flat[row_window_id_internal_val.numpy()]
    alpha = select_ridge_alpha(x_train_rows, y_train_rows, x_internal_val_rows, y_internal_val_rows)
    ridge = Ridge(alpha=alpha)
    ridge.fit(feats_train_flat_d_only, residual_train_flat)  # refit on all of router_train with selected alpha

    # --- Step 4/5: corrected prediction, evaluate on router_val ---
    n_val = int(val_cache["num_windows"])
    feats_val_flat_c = flatten_window_expert(val_features.features_for("C_window_forecast_disagreement"))
    feats_val_flat_d_only = flatten_window_expert(val_features.group_d).numpy()
    c_pred_val_flat = fit_c.predict(feats_val_flat_c).numpy()
    ridge_correction_val_flat = ridge.predict(feats_val_flat_d_only)
    corrected_pred_val_flat = c_pred_val_flat + ridge_correction_val_flat

    corrected_pred_excess = torch.tensor(corrected_pred_val_flat, dtype=torch.float32).reshape(n_val, k)
    weights = competence_to_weights(corrected_pred_excess, fit_c.temperature)  # reuse C's own temperature, no new tuning
    final_pred = (val_forecasts_all * weights.view(n_val, 1, 1, k)).sum(dim=-1)

    c_only_excess = torch.tensor(c_pred_val_flat, dtype=torch.float32).reshape(n_val, k)
    c_weights = competence_to_weights(c_only_excess, fit_c.temperature)
    c_pred_final = (val_forecasts_all * c_weights.view(n_val, 1, 1, k)).sum(dim=-1)

    npz = np.load(ORIGINAL_RESULTS_DIR / "per_window_predictions.npz")
    d_pred = torch.tensor(npz[f"{dataset}__D_full_behavioral"])
    equal_pred = torch.tensor(npz[f"{dataset}__equal_fixed"])
    oracle_pred = torch.tensor(npz[f"{dataset}__window_oracle"])

    methods = {"C_reproduced": c_pred_final, "C_plus_ridge_residual": final_pred, "D_full_behavioral_reference": d_pred, "equal_fixed_reference": equal_pred, "window_oracle_reference": oracle_pred}
    result_rows, metrics = [], {}
    for method, pred in methods.items():
        m = metric_values(bundle, pred)
        metrics[method] = m
        result_rows.append({"dataset": dataset, "method": method, "mae": m["mae"], "mse": m["mse"]})

    actual_flat = excess_loss_val.reshape(-1).numpy()
    spearman = spearmanr(corrected_pred_val_flat, actual_flat)
    pearson = pearsonr(corrected_pred_val_flat, actual_flat)
    top1 = float((corrected_pred_excess.numpy().argmin(axis=1) == excess_loss_val.numpy().argmin(axis=1)).mean())
    useful_label = (actual_flat < 0).astype(int)
    useful_score = -corrected_pred_val_flat
    auroc = float(roc_auc_score(useful_label, useful_score)) if useful_label.min() != useful_label.max() else float("nan")
    auprc = float(average_precision_score(useful_label, useful_score)) if useful_label.min() != useful_label.max() else float("nan")
    competence_row = {"dataset": dataset, "method": "C_plus_ridge_residual", "spearman": float(spearman.statistic), "pearson": float(pearson.statistic), "top1_accuracy": top1, "auroc_useful_vs_harmful": auroc, "auprc_useful_vs_harmful": auprc, "ridge_alpha": alpha}

    dependence_rows = []
    for label, cand_key, base_key in (("C+Ridge_vs_C", "C_plus_ridge_residual", "C_reproduced"), ("C+Ridge_vs_D", "C_plus_ridge_residual", "D_full_behavioral_reference"), ("C+Ridge_vs_Equal", "C_plus_ridge_residual", "equal_fixed_reference")):
        candidate, baseline = metrics[cand_key]["per_window_mae"], metrics[base_key]["per_window_mae"]
        boot = paired_bootstrap(candidate, baseline, seed=20260821, samples=5000)
        dependence_rows.append({"dataset": dataset, "comparison": label, "test": "iid_paired_bootstrap", **boot})
        for block in BLOCK_LENGTHS:
            b = block_bootstrap_with_prob(candidate, baseline, block=block, seed=20260821, samples=BOOTSTRAP_SAMPLES)
            dependence_rows.append({"dataset": dataset, "comparison": label, "test": f"block_bootstrap_len{block}", **b})
        phase = every_kth_phase_bootstrap(candidate - baseline, k=PHASE_K, seed=20260821, samples=BOOTSTRAP_SAMPLES)
        dependence_rows.append({"dataset": dataset, "comparison": label, "test": f"every_{PHASE_K}th_window_phase_bootstrap", **phase})

    return {
        "dataset": dataset,
        "core": bundle.core_names,
        "ridge_alpha": alpha,
        "result_rows": result_rows,
        "competence_row": competence_row,
        "dependence_rows": dependence_rows,
        "c_plus_ridge_minus_c": metrics["C_plus_ridge_residual"]["mae"] - metrics["C_reproduced"]["mae"],
        "c_plus_ridge_minus_d": metrics["C_plus_ridge_residual"]["mae"] - metrics["D_full_behavioral_reference"]["mae"],
    }


def make_report(out_dir: Path, report: Mapping[str, Any]) -> None:
    lines = [
        "# Residual-Correction Follow-Up (C + Ridge(behavioral) -> residual)",
        "",
        "Tests behavioral features as a correction on top of the already-trained C scorer, rather than a joint 35-feature retrain (as in the original D). Reuses the existing perturbation cache and feature pipeline unmodified.",
        "",
        "## Results (router_val MAE / MSE)",
        "",
        "| Dataset | C (reproduced) | C + Ridge residual | D (original, reference) | Equal (ref) | Oracle (ref) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for ds, d in report["datasets"].items():
        by = {r["method"]: r for r in d["result_rows"]}
        lines.append(
            "| {ds} | {c[mae]:.6f} | {cr[mae]:.6f} | {dd[mae]:.6f} | {eq[mae]:.6f} | {orc[mae]:.6f} |".format(
                ds=ds, c=by["C_reproduced"], cr=by["C_plus_ridge_residual"], dd=by["D_full_behavioral_reference"], eq=by["equal_fixed_reference"], orc=by["window_oracle_reference"]
            )
        )
    lines += ["", "## C+Ridge vs C, and vs the original D", ""]
    lines.append("| Dataset | Ridge alpha | (C+Ridge) - C | (C+Ridge) - D |")
    lines.append("|---|---:|---:|---:|")
    for ds, d in report["datasets"].items():
        lines.append(f"| {ds} | {d['ridge_alpha']:g} | `{d['c_plus_ridge_minus_c']:+.6f}` | `{d['c_plus_ridge_minus_d']:+.6f}` |")
    lines += ["", "## Dependence-aware statistics", ""]
    lines.append("| Dataset | Comparison | Test | Mean delta | 95% CI | Excludes zero |")
    lines.append("|---|---|---|---:|---|---|")
    for ds, d in report["datasets"].items():
        for row in d["dependence_rows"]:
            mean_key = row.get("mean_delta", row.get("mean_diff_candidate_minus_baseline"))
            if "ci95_low" in row:
                lines.append(f"| {ds} | {row['comparison']} | {row['test']} | `{mean_key:+.6f}` | [{row['ci95_low']:+.6f}, {row['ci95_high']:+.6f}] | {row['ci_excludes_zero']} |")
    lines += ["", "## Competence-prediction metrics", ""]
    lines.append("| Dataset | Spearman | Pearson | Top-1 acc | AUROC |")
    lines.append("|---|---:|---:|---:|---:|")
    for ds, d in report["datasets"].items():
        c = d["competence_row"]
        lines.append(f"| {ds} | {c['spearman']:.3f} | {c['pearson']:.3f} | {c['top1_accuracy']:.3f} | {c['auroc_useful_vs_harmful']:.3f} |")
    (out_dir / "residual_ridge_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    start = time.time()
    report: dict[str, Any] = {"experiment": "behavioral_competence_residual_ridge", "created_at_utc": datetime.now(timezone.utc).isoformat(), "datasets": {}}
    all_results, all_dependence, all_competence = [], [], []
    for dataset in LOADERS:
        print(f"[residual-ridge] {dataset}: training C, fitting Ridge on residual, evaluating...", flush=True)
        result = run_dataset(dataset)
        report["datasets"][dataset] = result
        all_results.extend(result["result_rows"])
        all_dependence.extend(result["dependence_rows"])
        all_competence.append(result["competence_row"])
        print(f"[residual-ridge] {dataset}: done. (C+Ridge)-C = {result['c_plus_ridge_minus_c']:+.6f}, (C+Ridge)-D = {result['c_plus_ridge_minus_d']:+.6f}, alpha={result['ridge_alpha']:g}", flush=True)

    report["runtime_sec"] = time.time() - start
    report["test_set_accessed"] = False
    write_json(RESULTS_DIR / "residual_ridge_results.json", report)
    write_csv(RESULTS_DIR / "residual_ridge_results.csv", all_results)
    write_csv(RESULTS_DIR / "residual_ridge_dependence.csv", all_dependence)
    write_csv(RESULTS_DIR / "residual_ridge_competence.csv", all_competence)
    make_report(REPORTS_DIR, report)
    print("TEST SET ACCESSED: NO")
    print(json.dumps({"runtime_sec": report["runtime_sec"]}, indent=2))


if __name__ == "__main__":
    main()
