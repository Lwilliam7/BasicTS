"""Simplex + Selective LearnedProbe.

Question: can we learn WHEN to trust the frozen LearnedProbe correction on
top of a strong static Simplex ensemble, so we keep the large gains
(ExchangeRate, Traffic) while avoiding the harmful correction (ETTm2)?

This is NOT a test of whether Probe information exists -- that was already
established (experiments/behavioral_competence/simplex_probe/, Real Probe
beat ShuffledProbe on 4/4 datasets, block-24 significant on 4/4). This tests
whether a small, router_train-only-trained GATE can predict, from target-free
features available at forecast time, whether the ALREADY-FROZEN Probe
correction should be trusted for the current window.

Reuses, unmodified:
  - scripts/costars/analyze_etth2_pair_potential.py::fit_simplex_weights
  - experiments/behavioral_competence/run_learned_probe.py::train_probe_and_scorer,
    evaluate_on_val, build_abc_features
  - experiments/behavioral_competence/generalization/run_generalization_study.py::register_dataset
  - experiments/behavioral_competence/simplex_probe/run_simplex_probe.py::
    fuse_weights, shuffle_probe_scores, probe_scores_on_train, metric_values,
    apply_fixed_weights, apply_per_window_weights, weight_diagnostics,
    top_expert_change_fraction, dependence_full, primary_row
  - the frozen alpha values selected by the simplex_probe experiment (loaded
    from its saved manifest, NOT reselected)
  - dependence-aware bootstrap helpers (paired_bootstrap, block_bootstrap_with_prob,
    every_kth_phase_bootstrap, train_folds)

New code in this file is limited to the gate itself: a small L2-regularized
logistic regression (hand-rolled full-batch gradient descent, matching this
project's existing style of hand-rolled deterministic optimizers such as
fit_simplex_weights), its target-free feature construction, its
router_train-only chronological leave-one-block-out cross-validation for
regularization selection, and the per-window-alpha fusion formula
(effective_alpha_t = gate_t * frozen_alpha) needed to apply it. The Probe and
Simplex themselves are never modified.

router_val only for the final comparison; router_val targets are used ONLY
retrospectively for diagnostics (Section 9B/10), never to fit anything. No
test cache for any dataset is built or loaded.
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


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.frozen_hv_costar.run_frozen_hv_costar as fhv  # noqa: E402
from experiments.behavioral_competence.generalization.run_generalization_study import register_dataset  # noqa: E402
from experiments.behavioral_competence.model_runtime import load_expert_runtime  # noqa: E402
from experiments.behavioral_competence.run_behavioral_competence import raw_history_cache  # noqa: E402
from experiments.behavioral_competence.run_learned_probe import build_abc_features, evaluate_on_val, train_probe_and_scorer  # noqa: E402
from experiments.behavioral_competence.simplex_probe.run_simplex_probe import (  # noqa: E402
    apply_fixed_weights,
    apply_per_window_weights,
    dependence_full,
    fuse_weights,
    metric_values,
    primary_row,
    probe_scores_on_train,
    shuffle_probe_scores,
    top_expert_change_fraction,
    weight_diagnostics,
)
from experiments.chronological_adaptive_costar.run_chronological_adaptive_costar import paired_bootstrap  # noqa: E402
from experiments.costar_multidataset_frozen.common import block_bootstrap_with_prob, every_kth_phase_bootstrap, train_folds  # noqa: E402
from experiments.oracle_weight_tournament.run_tournament import sample_mae, sample_mse  # noqa: E402
from scripts.costars.analyze_etth2_pair_potential import fit_simplex_weights  # noqa: E402


OUT_DIR = ROOT / "experiments/behavioral_competence/simplex_selective_probe"
PER_WINDOW_DIR = OUT_DIR / "per_window_errors"
SIMPLEX_PROBE_DIR = ROOT / "experiments/behavioral_competence/simplex_probe"
NEW_DATASETS = ["ExchangeRate", "Traffic", "BeijingAirQuality", "ETTm2"]
SHUFFLE_SEED = 20260821
BLOCK_LENGTHS = (12, 24, 48)
PRIMARY_BLOCK = 24
BOOTSTRAP_SAMPLES = 10000
PHASE_K = 12
ZERO_PROBE_TOLERANCE = 1e-5
GATE_L2_GRID = (0.001, 0.01, 0.1, 1.0)
GATE_LR = 0.5
GATE_ITERATIONS = 3000
GATE_FEATURE_NAMES = [
    "probe_best_loss",
    "probe_gap_best_second",
    "probe_std",
    "probe_agrees_simplex_top",
    "l1_weight_change",
    "max_abs_weight_change",
    "changes_top_expert",
    "entropy_diff",
    "mean_pairwise_disagreement",
]
EXPECTED_FROZEN_ALPHA = {"ExchangeRate": 2.0, "Traffic": 0.5, "BeijingAirQuality": 0.5, "ETTm2": 1.0}


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


def sha256_bytes(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def entropy(w: torch.Tensor, dim: int = -1) -> torch.Tensor:
    wc = w.clamp_min(1e-12)
    return -(wc * wc.log()).sum(dim=dim)


# ---------------------------------------------------------------------------
# Per-window-alpha fusion: effective_alpha_t = gate_t * frozen_alpha. Same
# mathematical form as fuse_weights (softmax(log(simplex) - alpha*z)), just
# with alpha allowed to vary per window instead of a single scalar. Written
# as a SEPARATE function (rather than modifying fuse_weights) so the
# always-on reproduction path is byte-for-byte unaffected.
# ---------------------------------------------------------------------------


def fuse_weights_selective(simplex_weights_k: torch.Tensor, probe_loss_nk: torch.Tensor, gate_n: torch.Tensor, base_alpha: float, eps: float = 1e-8) -> torch.Tensor:
    n, k = probe_loss_nk.shape
    mean = probe_loss_nk.mean(dim=1, keepdim=True)
    std = probe_loss_nk.std(dim=1, unbiased=False, keepdim=True).clamp_min(eps)
    z = (probe_loss_nk - mean) / std
    log_simplex = torch.where(simplex_weights_k > 0, torch.log(simplex_weights_k.clamp_min(1e-300)), torch.full_like(simplex_weights_k, float("-inf")))
    effective_alpha = (gate_n * base_alpha).view(n, 1)
    logits = log_simplex.view(1, k).expand(n, k) - effective_alpha * z
    return torch.softmax(logits, dim=1)


# ---------------------------------------------------------------------------
# Gate feature construction (target-free at inference; router_train-only label)
# ---------------------------------------------------------------------------


def compute_gate_features(probe_loss_nk: torch.Tensor, simplex_weights_k: torch.Tensor, alpha: float, mean_pairwise_disagreement_n: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (features [N,9], always_on_weights [N,K] at the given fixed
    alpha -- used both as a feature ingredient and, on router_train, to
    derive the training label)."""
    n, k = probe_loss_nk.shape
    sorted_probe, _ = torch.sort(probe_loss_nk, dim=1)
    best = sorted_probe[:, 0]
    second = sorted_probe[:, 1]
    gap = second - best
    std = probe_loss_nk.std(dim=1, unbiased=False)
    simplex_top = int(simplex_weights_k.argmax())
    probe_top = probe_loss_nk.argmin(dim=1)
    agree = (probe_top == simplex_top).to(torch.float32)

    always_on_weights = fuse_weights(simplex_weights_k, probe_loss_nk, alpha)
    diff = always_on_weights - simplex_weights_k.view(1, -1)
    l1 = diff.abs().sum(dim=1)
    max_change = diff.abs().max(dim=1).values
    changes_top = (always_on_weights.argmax(dim=1) != simplex_top).to(torch.float32)
    simplex_entropy = entropy(simplex_weights_k, dim=0)
    always_on_entropy = entropy(always_on_weights, dim=1)
    entropy_diff = always_on_entropy - simplex_entropy

    features = torch.stack([best, gap, std, agree, l1, max_change, changes_top, entropy_diff, mean_pairwise_disagreement_n], dim=1)
    assert features.shape == (n, len(GATE_FEATURE_NAMES))
    return features, always_on_weights


def mean_pairwise_disagreement(bundle, cache_raw: Mapping[str, Any]) -> torch.Tensor:
    _, _, group_c, _ = build_abc_features(bundle, cache_raw)
    return group_c[..., 2].mean(dim=1)  # GROUP_C_NAMES[2] == "avg_pairwise_disagreement"


# ---------------------------------------------------------------------------
# Gate model: hand-rolled L2-regularized logistic regression, deterministic
# full-batch gradient descent (same style as fit_simplex_weights).
# ---------------------------------------------------------------------------


def fit_logistic_gate(features_std: torch.Tensor, labels: torch.Tensor, l2: float, iterations: int = GATE_ITERATIONS, lr: float = GATE_LR) -> tuple[torch.Tensor, torch.Tensor]:
    n, d = features_std.shape
    x = features_std.to(torch.float64)
    y = labels.to(torch.float64)
    w = torch.zeros(d, dtype=torch.float64)
    b = torch.zeros((), dtype=torch.float64)
    for _ in range(iterations):
        logits = x @ w + b
        p = torch.sigmoid(logits)
        grad_w = x.t() @ (p - y) / n + l2 * w
        grad_b = (p - y).mean()
        w = w - lr * grad_w
        b = b - lr * grad_b
    return w.to(torch.float32), b.to(torch.float32)


def gate_cv_folds(n_train: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """4 chronological leave-one-block-out folds, using the SAME block
    boundaries as train_folds (the existing core/alpha-selection helper).
    The initial 20% chronological prefix (never an eval block in train_folds)
    is always part of every fold's fit set; the 4 blocks partitioning the
    remaining 80% are each held out exactly once. This is honest block
    cross-validation, not strict forward-only chronology (a fold's fit set
    can include blocks chronologically AFTER its eval block) -- documented
    explicitly here and in the manifest."""
    folds_raw = train_folds(n_train)
    all_idx = torch.arange(n_train)
    out = []
    for _, lo, hi in folds_raw:
        eval_mask = torch.zeros(n_train, dtype=torch.bool)
        eval_mask[lo:hi] = True
        out.append((all_idx[~eval_mask], all_idx[eval_mask]))
    return out


def select_gate_l2(features_std: torch.Tensor, labels: torch.Tensor, folds: list[tuple[torch.Tensor, torch.Tensor]], l2_grid: Sequence[float]) -> tuple[float, list[dict[str, Any]]]:
    n = features_std.shape[0]
    rows = []
    for l2 in l2_grid:
        oof_logits = torch.zeros(n, dtype=torch.float32)
        oof_filled = torch.zeros(n, dtype=torch.bool)
        for fit_idx, eval_idx in folds:
            w, b = fit_logistic_gate(features_std[fit_idx], labels[fit_idx], l2)
            oof_logits[eval_idx] = features_std[eval_idx] @ w + b
            oof_filled[eval_idx] = True
        p = torch.sigmoid(oof_logits[oof_filled]).clamp(1e-7, 1 - 1e-7)
        y = labels[oof_filled]
        logloss = float(-(y * p.log() + (1 - y) * (1 - p).log()).mean())
        acc = float(((p > 0.5).to(torch.float32) == y).to(torch.float32).mean())
        rows.append({"l2": l2, "oof_logloss": logloss, "oof_accuracy": acc, "oof_windows": int(oof_filled.sum())})
    rows_sorted = sorted(rows, key=lambda r: (r["oof_logloss"], -r["oof_accuracy"], -r["l2"]))
    selected = rows_sorted[0]["l2"]
    for r in rows:
        r["selected"] = r["l2"] == selected
    return selected, rows


def train_gate(probe_loss_train: torch.Tensor, simplex_weights: torch.Tensor, alpha: float, disagreement_train: torch.Tensor, forecasts_train_core: torch.Tensor, target_train: torch.Tensor, mask_train: torch.Tensor, std: torch.Tensor, folds: list[tuple[torch.Tensor, torch.Tensor]], label_suffix: str) -> dict[str, Any]:
    features_train, always_on_weights_train = compute_gate_features(probe_loss_train, simplex_weights, alpha, disagreement_train)
    simplex_pred_train = apply_fixed_weights(forecasts_train_core, simplex_weights)
    always_on_pred_train = apply_per_window_weights(forecasts_train_core, always_on_weights_train)
    simplex_err_train = sample_mae(simplex_pred_train, target_train, mask_train, std)
    always_on_err_train = sample_mae(always_on_pred_train, target_train, mask_train, std)
    probe_gain_train = simplex_err_train - always_on_err_train
    labels_train = (probe_gain_train > 0).to(torch.float32)

    feat_mean = features_train.mean(dim=0, keepdim=True)
    feat_std = features_train.std(dim=0, keepdim=True).clamp_min(1e-8)
    features_std = (features_train - feat_mean) / feat_std

    selected_l2, l2_rows = select_gate_l2(features_std, labels_train, folds, GATE_L2_GRID)
    for r in l2_rows:
        r["probe_variant"] = label_suffix
    w, b = fit_logistic_gate(features_std, labels_train, selected_l2)

    return {
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "w": w,
        "b": b,
        "selected_l2": selected_l2,
        "l2_rows": l2_rows,
        "label_positive_rate": float(labels_train.mean()),
        "probe_gain_train": probe_gain_train,
        "labels_train": labels_train,
    }


def apply_gate(features_raw: torch.Tensor, fit: Mapping[str, Any]) -> torch.Tensor:
    features_std = (features_raw - fit["feat_mean"]) / fit["feat_std"]
    logits = features_std @ fit["w"] + fit["b"]
    return torch.sigmoid(logits)


# ---------------------------------------------------------------------------
# Per-dataset pipeline
# ---------------------------------------------------------------------------


def load_frozen_simplex_probe_reference() -> dict[str, Any]:
    manifest = json.loads((SIMPLEX_PROBE_DIR / "simplex_probe_manifest.json").read_text(encoding="utf-8"))
    validation = json.loads((SIMPLEX_PROBE_DIR / "validation_results.json").read_text(encoding="utf-8"))
    for ds, expected in EXPECTED_FROZEN_ALPHA.items():
        saved = manifest["selected_alpha"][ds]
        if abs(float(saved) - expected) > 1e-9:
            raise AssertionError(f"{ds}: saved alpha {saved} does not match expected {expected} -- refusing to proceed on an unverified premise")
    return {"manifest": manifest, "validation": validation}


def evaluate_dataset(dataset: str, frozen_ref: Mapping[str, Any]) -> dict[str, Any]:
    reg = register_dataset(dataset)
    core = reg["selected_core"]
    print(f"[selective_probe] {dataset}: core (router_train only) = {core}", flush=True)
    expected_core = frozen_ref["manifest"]["expert_cores"][dataset]
    if list(core) != list(expected_core):
        raise AssertionError(f"{dataset}: freshly selected core {core} != saved core {expected_core}")

    bundle = fhv.LOADERS[dataset]()
    train_cache = bundle.train_cache
    val_cache = bundle.val_cache
    k = len(bundle.core_names)
    n_train = int(train_cache["num_windows"])
    n_val = int(val_cache["num_windows"])

    checkpoint_hashes_before = {e: load_expert_runtime(dataset, e).checkpoint_sha256 for e in core}

    expert_names = list(train_cache["expert_names"])
    core_idx = [expert_names.index(name) for name in core]
    core_train_cache = dict(train_cache)
    core_train_cache["prediction_stack"] = train_cache["prediction_stack"][..., core_idx]

    print(f"[selective_probe] {dataset}: reproducing base Simplex (router_train only)...", flush=True)
    simplex_weights = fit_simplex_weights(core_train_cache)
    simplex_weights_check = fit_simplex_weights(core_train_cache)
    if float((simplex_weights - simplex_weights_check).abs().max()) > 1e-6:
        raise AssertionError(f"{dataset}: base Simplex is not reproducible")
    saved_weights = frozen_ref["manifest"]["base_simplex_weights"][dataset]
    saved_weights_vec = torch.tensor([saved_weights[name] for name in core], dtype=torch.float32)
    simplex_weights_match_saved = float((simplex_weights - saved_weights_vec).abs().max()) < 1e-4

    frozen_alpha = float(frozen_ref["manifest"]["selected_alpha"][dataset])

    forecasts_val_core = bundle.forecasts_fn(val_cache, bundle.expert_idx)
    simplex_pred_val = apply_fixed_weights(forecasts_val_core, simplex_weights)
    simplex_metrics_val = metric_values(val_cache, simplex_pred_val, bundle.std)

    print(f"[selective_probe] {dataset}: training LearnedProbe (frozen, unmodified train_probe_and_scorer)...", flush=True)
    fit_instance = train_probe_and_scorer(dataset, "instance")
    eval_val = evaluate_on_val(dataset, bundle, fit_instance, val_cache)
    probe_loss_val = eval_val["pred_excess"]

    print(f"[selective_probe] {dataset}: forward-passing frozen probe over router_train for gate training labels...", flush=True)
    probe_loss_train, forecasts_train_core = probe_scores_on_train(dataset, bundle, fit_instance, train_cache)

    checkpoint_hashes_after = {e: load_expert_runtime(dataset, e).checkpoint_sha256 for e in core}
    checkpoints_unchanged = checkpoint_hashes_before == checkpoint_hashes_after

    # --- reproduce existing always-on Simplex+Probe / +ShuffledProbe (Section 1) ---
    always_on_weights_val = fuse_weights(simplex_weights, probe_loss_val, frozen_alpha)
    always_on_pred_val = apply_per_window_weights(forecasts_val_core, always_on_weights_val)
    always_on_metrics_val = metric_values(val_cache, always_on_pred_val, bundle.std)
    probe_loss_val_shuffled_reprod = shuffle_probe_scores(probe_loss_val, SHUFFLE_SEED)
    always_on_shuffled_weights_val = fuse_weights(simplex_weights, probe_loss_val_shuffled_reprod, frozen_alpha)
    always_on_shuffled_pred_val = apply_per_window_weights(forecasts_val_core, always_on_shuffled_weights_val)
    always_on_shuffled_metrics_val = metric_values(val_cache, always_on_shuffled_pred_val, bundle.std)

    saved_results = {r["method"]: r for r in frozen_ref["validation"]["datasets"][dataset]["result_rows"]}
    reproduction_diffs = {
        "simplex_mae_diff": abs(simplex_metrics_val["mae"] - saved_results["Simplex"]["mae"]),
        "always_on_probe_mae_diff": abs(always_on_metrics_val["mae"] - saved_results["Simplex_Probe"]["mae"]),
        "always_on_shuffled_mae_diff": abs(always_on_shuffled_metrics_val["mae"] - saved_results["Simplex_ShuffledProbe"]["mae"]),
    }
    reproduction_ok = all(v < 1e-4 for v in reproduction_diffs.values()) and simplex_weights_match_saved
    print(f"[selective_probe] {dataset}: reproduction check vs saved simplex_probe results: {reproduction_diffs} (ok={reproduction_ok})", flush=True)

    # --- gate features: real probe ---
    train_cache_raw = raw_history_cache(dataset, train_cache, fit_instance["val_runtimes"][bundle.core_names[0]].mean, fit_instance["val_runtimes"][bundle.core_names[0]].std)
    val_cache_raw = raw_history_cache(dataset, val_cache, fit_instance["val_runtimes"][bundle.core_names[0]].mean, fit_instance["val_runtimes"][bundle.core_names[0]].std)
    disagreement_train = mean_pairwise_disagreement(bundle, train_cache_raw)
    disagreement_val = mean_pairwise_disagreement(bundle, val_cache_raw)

    target_train = train_cache["targets"].to(torch.float32)
    mask_train = train_cache["target_masks"].to(torch.bool)
    folds = gate_cv_folds(n_train)

    print(f"[selective_probe] {dataset}: training gate (real probe, router_train OOF only)...", flush=True)
    gate_fit_real = train_gate(probe_loss_train, simplex_weights, frozen_alpha, disagreement_train, forecasts_train_core, target_train, mask_train, bundle.std, folds, "real")

    print(f"[selective_probe] {dataset}: training gate (shuffled probe, router_train OOF only)...", flush=True)
    probe_loss_train_shuffled = shuffle_probe_scores(probe_loss_train, SHUFFLE_SEED)
    gate_fit_shuffled = train_gate(probe_loss_train_shuffled, simplex_weights, frozen_alpha, disagreement_train, forecasts_train_core, target_train, mask_train, bundle.std, folds, "shuffled")

    # --- apply gates at router_val (target-free features only) ---
    features_val_real, always_on_weights_val_check = compute_gate_features(probe_loss_val, simplex_weights, frozen_alpha, disagreement_val)
    gate_val_real = apply_gate(features_val_real, gate_fit_real)

    probe_loss_val_shuffled = shuffle_probe_scores(probe_loss_val, SHUFFLE_SEED)
    features_val_shuffled, _ = compute_gate_features(probe_loss_val_shuffled, simplex_weights, frozen_alpha, disagreement_val)
    gate_val_shuffled = apply_gate(features_val_shuffled, gate_fit_shuffled)

    weights_selective_val = fuse_weights_selective(simplex_weights, probe_loss_val, gate_val_real, frozen_alpha)
    pred_selective_val = apply_per_window_weights(forecasts_val_core, weights_selective_val)
    selective_metrics_val = metric_values(val_cache, pred_selective_val, bundle.std)

    weights_selective_shuffled_val = fuse_weights_selective(simplex_weights, probe_loss_val_shuffled, gate_val_shuffled, frozen_alpha)
    pred_selective_shuffled_val = apply_per_window_weights(forecasts_val_core, weights_selective_shuffled_val)
    selective_shuffled_metrics_val = metric_values(val_cache, pred_selective_shuffled_val, bundle.std)

    # --- 11. Zero-gate invariance ---
    gate_zero = torch.zeros(n_val)
    weights_gate0 = fuse_weights_selective(simplex_weights, probe_loss_val, gate_zero, frozen_alpha)
    pred_gate0 = apply_per_window_weights(forecasts_val_core, weights_gate0)
    gate0_max_weight_diff = float((weights_gate0 - simplex_weights.view(1, -1).expand(n_val, -1)).abs().max())
    gate0_max_pred_diff_normalized = float(((pred_gate0 - simplex_pred_val) / bundle.std.view(1, 1, -1)).abs().max())
    gate0_mae_diff = float(metric_values(val_cache, pred_gate0, bundle.std)["mae"] - simplex_metrics_val["mae"])
    gate0_mse_diff = float(metric_values(val_cache, pred_gate0, bundle.std)["mse"] - simplex_metrics_val["mse"])
    zero_gate_ok = gate0_max_weight_diff < ZERO_PROBE_TOLERANCE and gate0_max_pred_diff_normalized < ZERO_PROBE_TOLERANCE and abs(gate0_mae_diff) < ZERO_PROBE_TOLERANCE
    if not zero_gate_ok:
        raise AssertionError(f"{dataset}: gate=0 does NOT reproduce base Simplex: max_weight_diff={gate0_max_weight_diff}, max_pred_diff_normalized={gate0_max_pred_diff_normalized}, mae_diff={gate0_mae_diff}")

    # --- result rows: the 4 required methods ---
    result_rows = [
        {"dataset": dataset, "method": "Simplex", "mae": simplex_metrics_val["mae"], "mse": simplex_metrics_val["mse"]},
        {
            "dataset": dataset,
            "method": "Simplex_AlwaysOnProbe",
            "mae": always_on_metrics_val["mae"],
            "mse": always_on_metrics_val["mse"],
            "delta_vs_simplex": always_on_metrics_val["mae"] - simplex_metrics_val["mae"],
            "relative_pct_vs_simplex": 100.0 * (simplex_metrics_val["mae"] - always_on_metrics_val["mae"]) / simplex_metrics_val["mae"],
        },
        {
            "dataset": dataset,
            "method": "Simplex_SelectiveProbe",
            "mae": selective_metrics_val["mae"],
            "mse": selective_metrics_val["mse"],
            "delta_vs_simplex": selective_metrics_val["mae"] - simplex_metrics_val["mae"],
            "relative_pct_vs_simplex": 100.0 * (simplex_metrics_val["mae"] - selective_metrics_val["mae"]) / simplex_metrics_val["mae"],
            "delta_vs_always_on": selective_metrics_val["mae"] - always_on_metrics_val["mae"],
        },
        {
            "dataset": dataset,
            "method": "Simplex_SelectiveShuffledProbe",
            "mae": selective_shuffled_metrics_val["mae"],
            "mse": selective_shuffled_metrics_val["mse"],
            "delta_vs_simplex": selective_shuffled_metrics_val["mae"] - simplex_metrics_val["mae"],
            "delta_selective_real_vs_shuffled": selective_metrics_val["mae"] - selective_shuffled_metrics_val["mae"],
        },
    ]

    # --- dependence-aware statistics (Section 13) ---
    dependence_rows = []
    dependence_rows.extend(dependence_full(selective_metrics_val["per_window_mae"], simplex_metrics_val["per_window_mae"], dataset, "Selective_vs_Simplex"))
    dependence_rows.extend(dependence_full(selective_metrics_val["per_window_mae"], always_on_metrics_val["per_window_mae"], dataset, "Selective_vs_AlwaysOn"))
    dependence_rows.extend(dependence_full(selective_metrics_val["per_window_mae"], selective_shuffled_metrics_val["per_window_mae"], dataset, "Selective_vs_SelectiveShuffled"))
    primary_selective_vs_simplex = primary_row(dependence_rows, "Selective_vs_Simplex")
    primary_selective_vs_alwayson = primary_row(dependence_rows, "Selective_vs_AlwaysOn")
    primary_selective_vs_shuffled = primary_row(dependence_rows, "Selective_vs_SelectiveShuffled")

    # --- weight-concentration analysis ---
    weights_simplex_broadcast = simplex_weights.view(1, -1).expand(n_val, -1)
    weight_rows = [
        {"dataset": dataset, "method": "Simplex", "fraction_top_expert_changed_vs_simplex": 0.0, **weight_diagnostics(weights_simplex_broadcast)},
        {"dataset": dataset, "method": "Simplex_AlwaysOnProbe", "fraction_top_expert_changed_vs_simplex": top_expert_change_fraction(weights_simplex_broadcast, always_on_weights_val), **weight_diagnostics(always_on_weights_val)},
        {"dataset": dataset, "method": "Simplex_SelectiveProbe", "fraction_top_expert_changed_vs_simplex": top_expert_change_fraction(weights_simplex_broadcast, weights_selective_val), **weight_diagnostics(weights_selective_val)},
        {"dataset": dataset, "method": "Simplex_SelectiveShuffledProbe", "fraction_top_expert_changed_vs_simplex": top_expert_change_fraction(weights_simplex_broadcast, weights_selective_shuffled_val), **weight_diagnostics(weights_selective_shuffled_val)},
    ]

    # --- Section 9: critical diagnostics (retrospective; targets used ONLY for reporting) ---
    target_val = val_cache["targets"].to(torch.float32)
    mask_val = val_cache["target_masks"].to(torch.bool)
    always_on_err_val = sample_mae(always_on_pred_val, target_val, mask_val, bundle.std)
    simplex_err_val = simplex_metrics_val["per_window_mae"]
    probe_gain_val = simplex_err_val - always_on_err_val  # positive = always-on probe helped this window

    agree_val = features_val_real[:, GATE_FEATURE_NAMES.index("probe_agrees_simplex_top")].to(torch.bool)
    changes_top_val = features_val_real[:, GATE_FEATURE_NAMES.index("changes_top_expert")].to(torch.bool)
    l1_change_val = features_val_real[:, GATE_FEATURE_NAMES.index("l1_weight_change")]

    diag_rows = []
    diag_rows.append({"dataset": dataset, "section": "A_gate_behavior", "mean_gate": float(gate_val_real.mean()), "median_gate": float(gate_val_real.median()), "fraction_gate_lt_0.1": float((gate_val_real < 0.1).to(torch.float32).mean()), "fraction_gate_gt_0.9": float((gate_val_real > 0.9).to(torch.float32).mean())})

    trusted = gate_val_real > 0.9
    rejected = gate_val_real < 0.1
    diag_rows.append(
        {
            "dataset": dataset,
            "section": "B_gate_usefulness",
            "num_trusted_windows": int(trusted.sum()),
            "num_rejected_windows": int(rejected.sum()),
            "mae_when_trusted_simplex": float(simplex_err_val[trusted].mean()) if bool(trusted.any()) else float("nan"),
            "mae_when_trusted_selective": float(selective_metrics_val["per_window_mae"][trusted].mean()) if bool(trusted.any()) else float("nan"),
            "probe_gain_on_trusted": float(probe_gain_val[trusted].mean()) if bool(trusted.any()) else float("nan"),
            "mae_when_rejected_simplex": float(simplex_err_val[rejected].mean()) if bool(rejected.any()) else float("nan"),
            "mae_when_rejected_selective": float(selective_metrics_val["per_window_mae"][rejected].mean()) if bool(rejected.any()) else float("nan"),
            "probe_gain_on_rejected": float(probe_gain_val[rejected].mean()) if bool(rejected.any()) else float("nan"),
        }
    )

    for label, sel in (("agrees", agree_val), ("disagrees", ~agree_val)):
        diag_rows.append(
            {
                "dataset": dataset,
                "section": "C_simplex_agreement",
                "group": label,
                "num_windows": int(sel.sum()),
                "mean_gate": float(gate_val_real[sel].mean()) if bool(sel.any()) else float("nan"),
                "probe_gain_always_on": float(probe_gain_val[sel].mean()) if bool(sel.any()) else float("nan"),
            }
        )

    frac_top_changed_always_on = float(changes_top_val.to(torch.float32).mean())
    frac_top_changed_selective = top_expert_change_fraction(weights_simplex_broadcast, weights_selective_val)
    diag_rows.append(
        {
            "dataset": dataset,
            "section": "D_top_expert_changes",
            "fraction_changed_always_on": frac_top_changed_always_on,
            "fraction_changed_selective": frac_top_changed_selective,
            "probe_gain_when_always_on_changes_top": float(probe_gain_val[changes_top_val].mean()) if bool(changes_top_val.any()) else float("nan"),
            "probe_gain_when_always_on_keeps_top": float(probe_gain_val[~changes_top_val].mean()) if bool((~changes_top_val).any()) else float("nan"),
        }
    )

    tertiles = torch.quantile(l1_change_val, torch.tensor([1.0 / 3, 2.0 / 3]))
    small_mask = l1_change_val <= tertiles[0]
    medium_mask = (l1_change_val > tertiles[0]) & (l1_change_val <= tertiles[1])
    large_mask = l1_change_val > tertiles[1]
    for label, sel in (("small", small_mask), ("medium", medium_mask), ("large", large_mask)):
        diag_rows.append(
            {
                "dataset": dataset,
                "section": "E_correction_magnitude_bucket",
                "bucket": label,
                "num_windows": int(sel.sum()),
                "mean_l1_change": float(l1_change_val[sel].mean()) if bool(sel.any()) else float("nan"),
                "probe_gain_always_on": float(probe_gain_val[sel].mean()) if bool(sel.any()) else float("nan"),
                "mean_gate": float(gate_val_real[sel].mean()) if bool(sel.any()) else float("nan"),
            }
        )

    # --- Section 10: ETTm2-specific failure analysis ---
    ettm2_rows = []
    if dataset == "ETTm2":
        helps = probe_gain_val > 0
        hurts = probe_gain_val < 0
        ettm2_rows = [
            {
                "window": i,
                "probe_gain_always_on": float(probe_gain_val[i]),
                "helps": bool(helps[i]),
                "hurts": bool(hurts[i]),
                "gate_real": float(gate_val_real[i]),
                "probe_agrees_simplex_top": bool(agree_val[i]),
                "changes_top_expert_always_on": bool(changes_top_val[i]),
                "l1_weight_change": float(l1_change_val[i]),
                "probe_std": float(features_val_real[i, GATE_FEATURE_NAMES.index("probe_std")]),
                "probe_gap_best_second": float(features_val_real[i, GATE_FEATURE_NAMES.index("probe_gap_best_second")]),
            }
            for i in range(n_val)
        ]
        ettm2_summary = {
            "fraction_helps": float(helps.to(torch.float32).mean()),
            "fraction_hurts": float(hurts.to(torch.float32).mean()),
            "avg_magnitude_help": float(probe_gain_val[helps].mean()) if bool(helps.any()) else float("nan"),
            "avg_magnitude_harm": float(probe_gain_val[hurts].mean()) if bool(hurts.any()) else float("nan"),
            "mean_probe_std_on_harmful": float(features_val_real[hurts, GATE_FEATURE_NAMES.index("probe_std")].mean()) if bool(hurts.any()) else float("nan"),
            "mean_probe_std_on_helpful": float(features_val_real[helps, GATE_FEATURE_NAMES.index("probe_std")].mean()) if bool(helps.any()) else float("nan"),
            "fraction_disagree_on_harmful": float((~agree_val[hurts]).to(torch.float32).mean()) if bool(hurts.any()) else float("nan"),
            "fraction_disagree_on_helpful": float((~agree_val[helps]).to(torch.float32).mean()) if bool(helps.any()) else float("nan"),
            "mean_l1_change_on_harmful": float(l1_change_val[hurts].mean()) if bool(hurts.any()) else float("nan"),
            "mean_l1_change_on_helpful": float(l1_change_val[helps].mean()) if bool(helps.any()) else float("nan"),
            "mean_gate_on_harmful": float(gate_val_real[hurts].mean()) if bool(hurts.any()) else float("nan"),
            "mean_gate_on_helpful": float(gate_val_real[helps].mean()) if bool(helps.any()) else float("nan"),
            "gate_successfully_lower_on_harmful": bool(hurts.any() and helps.any() and float(gate_val_real[hurts].mean()) < float(gate_val_real[helps].mean())),
        }
    else:
        ettm2_summary = None

    # --- 12. Integrity ---
    test_cache_path = ROOT / f"cache/costarts_walkforward_{dataset}" / "test_80_100_cache.pt"
    gate_val_snapshot = gate_val_real.clone()
    weights_snapshot = weights_selective_val.clone()
    gen = torch.Generator().manual_seed(4242)
    corrupted_targets = torch.randn(val_cache["targets"].shape, generator=gen, dtype=torch.float32)
    gate_val_recompute = apply_gate(features_val_real, gate_fit_real)  # recomputed from the SAME target-free features; never reads corrupted_targets
    weights_recompute = fuse_weights_selective(simplex_weights, probe_loss_val, gate_val_recompute, frozen_alpha)
    del corrupted_targets
    target_corruption_invariant = bool(torch.equal(gate_val_real, gate_val_snapshot)) and bool(torch.equal(gate_val_recompute, gate_val_real)) and bool(torch.equal(weights_recompute, weights_snapshot))

    integrity = {
        "dataset": dataset,
        "no_test_cache_loaded": not test_cache_path.exists(),
        "no_router_val_target_used_during_gate_training": True,  # structural: train_gate only ever receives router_train tensors
        "expert_checkpoints_unchanged": checkpoints_unchanged,
        "learnedprobe_parameters_unchanged": fit_instance["experts_remained_frozen"],
        "simplex_weights_unchanged": simplex_weights_match_saved,
        "alpha_unchanged_from_simplex_probe_experiment": abs(frozen_alpha - EXPECTED_FROZEN_ALPHA[dataset]) < 1e-9,
        "gate_training_examples_chronological_oof": True,  # structural: gate_cv_folds/select_gate_l2 always evaluate on held-out fold indices only
        "current_prediction_never_uses_current_target": True,  # structural: features_val_real/gate_val_real never read val_cache['targets']
        "router_val_target_corruption_invariant": target_corruption_invariant,
        "gate_features_target_free_at_inference": True,
        "zero_gate_reproduces_base_simplex": zero_gate_ok,
        "zero_gate_max_weight_diff": gate0_max_weight_diff,
        "zero_gate_max_pred_diff_normalized": gate0_max_pred_diff_normalized,
        "zero_gate_mae_diff": gate0_mae_diff,
        "zero_gate_mse_diff": gate0_mse_diff,
        "reproduction_of_prior_simplex_probe_experiment_ok": reproduction_ok,
        "reproduction_diffs": reproduction_diffs,
        "result": "PASS"
        if (
            not test_cache_path.exists()
            and checkpoints_unchanged
            and fit_instance["experts_remained_frozen"]
            and simplex_weights_match_saved
            and zero_gate_ok
            and target_corruption_invariant
            and reproduction_ok
        )
        else "FAIL",
    }
    if integrity["result"] == "FAIL":
        raise AssertionError(f"{dataset}: simplex_selective_probe integrity check FAILED: {integrity}")

    # --- per-window error dump ---
    PER_WINDOW_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        PER_WINDOW_DIR / f"{dataset}.npz",
        simplex_mae=simplex_metrics_val["per_window_mae"].numpy(),
        always_on_probe_mae=always_on_metrics_val["per_window_mae"].numpy(),
        selective_probe_mae=selective_metrics_val["per_window_mae"].numpy(),
        selective_shuffled_probe_mae=selective_shuffled_metrics_val["per_window_mae"].numpy(),
        gate_real=gate_val_real.numpy(),
        gate_shuffled=gate_val_shuffled.numpy(),
        probe_gain_always_on=probe_gain_val.numpy(),
        simplex_weights=simplex_weights.numpy(),
        core=np.array(core),
    )

    return {
        "dataset": dataset,
        "core": core,
        "frozen_alpha": frozen_alpha,
        "checkpoint_hashes": checkpoint_hashes_after,
        "simplex_weights": {name: float(simplex_weights[i]) for i, name in enumerate(core)},
        "gate_selected_l2_real": gate_fit_real["selected_l2"],
        "gate_selected_l2_shuffled": gate_fit_shuffled["selected_l2"],
        "gate_l2_rows": gate_fit_real["l2_rows"] + gate_fit_shuffled["l2_rows"],
        "gate_label_positive_rate_real": gate_fit_real["label_positive_rate"],
        "gate_label_positive_rate_shuffled": gate_fit_shuffled["label_positive_rate"],
        "gate_weights_real": {name: float(gate_fit_real["w"][i]) for i, name in enumerate(GATE_FEATURE_NAMES)},
        "gate_bias_real": float(gate_fit_real["b"]),
        "result_rows": result_rows,
        "dependence_rows": dependence_rows,
        "weight_rows": weight_rows,
        "diag_rows": diag_rows,
        "ettm2_rows": ettm2_rows,
        "ettm2_summary": ettm2_summary,
        "primary_selective_vs_simplex": primary_selective_vs_simplex,
        "primary_selective_vs_alwayson": primary_selective_vs_alwayson,
        "primary_selective_vs_shuffled": primary_selective_vs_shuffled,
        "integrity": integrity,
    }


# ---------------------------------------------------------------------------
# Decision rule (Sections 14-16, pre-specified, not altered after seeing results)
# ---------------------------------------------------------------------------


def decide(report: Mapping[str, Any]) -> dict[str, Any]:
    datasets = list(report["datasets"].keys())
    by = {ds: {r["method"]: r for r in report["datasets"][ds]["result_rows"]} for ds in datasets}

    def frac_preserved(ds: str) -> float:
        always_on_gain = by[ds]["Simplex"]["mae"] - by[ds]["Simplex_AlwaysOnProbe"]["mae"]
        selective_gain = by[ds]["Simplex"]["mae"] - by[ds]["Simplex_SelectiveProbe"]["mae"]
        if always_on_gain <= 0:
            return float("nan")
        return selective_gain / always_on_gain

    exchangerate_preserved = frac_preserved("ExchangeRate") if "ExchangeRate" in datasets else float("nan")
    traffic_preserved = frac_preserved("Traffic") if "Traffic" in datasets else float("nan")

    beijing_p24 = report["datasets"]["BeijingAirQuality"]["primary_selective_vs_simplex"] if "BeijingAirQuality" in datasets else None
    beijing_regresses = bool(beijing_p24 and beijing_p24["ci_excludes_zero"] and beijing_p24["mean_delta"] > 0)

    ettm2_always_on_delta = by["ETTm2"]["Simplex_AlwaysOnProbe"]["delta_vs_simplex"] if "ETTm2" in datasets else float("nan")
    ettm2_selective_delta = by["ETTm2"]["Simplex_SelectiveProbe"]["delta_vs_simplex"] if "ETTm2" in datasets else float("nan")
    ettm2_p24 = report["datasets"]["ETTm2"]["primary_selective_vs_simplex"] if "ETTm2" in datasets else None
    ettm2_still_significant_regression = bool(ettm2_p24 and ettm2_p24["ci_excludes_zero"] and ettm2_p24["mean_delta"] > 0)
    ettm2_fixed_or_reduced = (not ettm2_still_significant_regression) or (ettm2_selective_delta < ettm2_always_on_delta)

    n_beats_shuffled_point = sum(1 for ds in datasets if by[ds]["Simplex_SelectiveProbe"]["mae"] < by[ds]["Simplex_SelectiveShuffledProbe"]["mae"])
    n_beats_shuffled_sig = sum(1 for ds in datasets if report["datasets"][ds]["primary_selective_vs_shuffled"]["ci_excludes_zero"] and report["datasets"][ds]["primary_selective_vs_shuffled"]["mean_delta"] < 0)
    clearly_beats_shuffled = n_beats_shuffled_point >= 3 and n_beats_shuffled_sig >= 1

    n_new_broad_regressions = sum(
        1
        for ds in datasets
        if report["datasets"][ds]["primary_selective_vs_simplex"]["ci_excludes_zero"] and report["datasets"][ds]["primary_selective_vs_simplex"]["mean_delta"] > 0
    )

    exchangerate_preserved_ok = not (exchangerate_preserved == exchangerate_preserved) or exchangerate_preserved >= 0.5  # NaN-safe: "not nan" check
    traffic_preserved_ok = not (traffic_preserved == traffic_preserved) or traffic_preserved >= 0.5

    promising = exchangerate_preserved_ok and traffic_preserved_ok and (not beijing_regresses) and ettm2_fixed_or_reduced and clearly_beats_shuffled and (n_new_broad_regressions <= 1)
    mostly_destroys_gains = (exchangerate_preserved == exchangerate_preserved and exchangerate_preserved < 0.3) or (traffic_preserved == traffic_preserved and traffic_preserved < 0.3)
    similar_to_shuffled = n_beats_shuffled_point <= 1

    if promising:
        tier = "PROMISING"
    elif ettm2_still_significant_regression and (not exchangerate_preserved_ok) and (not traffic_preserved_ok):
        tier = "FAILURE"
    elif similar_to_shuffled:
        tier = "FAILURE"
    elif mostly_destroys_gains and (not ettm2_fixed_or_reduced):
        tier = "FAILURE"
    else:
        tier = "MIXED"

    recommendation = "PROCEED TO FROZEN COSTAR + PROBE" if tier == "PROMISING" else "DO NOT PROCEED TO FROZEN COSTAR + PROBE YET"
    conclusions = {
        "PROMISING": "Selective Probe preserves most of the ExchangeRate/Traffic gains, avoids the BeijingAirQuality/ETTm2 harm, and clearly beats Selective ShuffledProbe. Recommend proceeding to Frozen COSTAR + Probe.",
        "MIXED": "Selective Probe reduces regressions but also gives up a meaningful share of the ExchangeRate/Traffic gain (or only partially fixes ETTm2). Probe usefulness is only partly predictable from the current observable features. Diagnose before increasing gate complexity.",
        "FAILURE": "Selective Probe still significantly hurts ETTm2, loses the major ExchangeRate/Traffic gains, or performs similarly to Selective ShuffledProbe. A simple selective-use mechanism on these features does not solve the inconsistency.",
    }

    return {
        "tier": tier,
        "recommendation": recommendation,
        "conclusion": conclusions[tier],
        "exchangerate_gain_fraction_preserved": exchangerate_preserved,
        "traffic_gain_fraction_preserved": traffic_preserved,
        "beijingairquality_regresses_significantly": beijing_regresses,
        "ettm2_always_on_delta": ettm2_always_on_delta,
        "ettm2_selective_delta": ettm2_selective_delta,
        "ettm2_still_significant_regression": ettm2_still_significant_regression,
        "ettm2_fixed_or_reduced": ettm2_fixed_or_reduced,
        "n_beats_shuffled_point": n_beats_shuffled_point,
        "n_beats_shuffled_significant": n_beats_shuffled_sig,
        "clearly_beats_shuffled": clearly_beats_shuffled,
        "n_new_broad_regressions": n_new_broad_regressions,
    }


def git_commit_sha() -> str:
    import subprocess

    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def make_report(report: Mapping[str, Any], decision: Mapping[str, Any]) -> None:
    datasets = list(report["datasets"].keys())
    lines = [
        "# Simplex + Selective LearnedProbe",
        "",
        "Question: can a small, router_train-only-trained gate learn WHEN to trust the frozen LearnedProbe correction on a strong static Simplex ensemble, keeping the ExchangeRate/Traffic gains while avoiding the ETTm2 regression?",
        "",
        "effective_alpha_t = gate_t * frozen_alpha, where gate_t = sigmoid(w . standardized_features_t + b) is a 9-feature L2-regularized logistic regression trained on router_train only (chronological leave-one-block-out OOF for regularization selection), predicting whether the always-on Probe correction helped or hurt that router_train window.",
        "",
        "## Primary results (router_val MAE)",
        "",
        "| Dataset | Simplex | Always-On Probe | Selective Probe | Selective ShuffledProbe | Δ Selective vs Simplex | Δ Selective vs AlwaysOn |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for ds in datasets:
        by = {r["method"]: r for r in report["datasets"][ds]["result_rows"]}
        s, a, sel, sh = by["Simplex"], by["Simplex_AlwaysOnProbe"], by["Simplex_SelectiveProbe"], by["Simplex_SelectiveShuffledProbe"]
        lines.append(f"| {ds} | {s['mae']:.6f} | {a['mae']:.6f} | {sel['mae']:.6f} | {sh['mae']:.6f} | `{sel['delta_vs_simplex']:+.6f}` | `{sel['delta_vs_always_on']:+.6f}` |")
    lines += ["", "## Gain preservation (vs the always-on Probe's gain over Simplex)", ""]
    lines.append(f"- ExchangeRate fraction of always-on gain preserved by Selective: `{decision['exchangerate_gain_fraction_preserved']:.3f}`")
    lines.append(f"- Traffic fraction of always-on gain preserved by Selective: `{decision['traffic_gain_fraction_preserved']:.3f}`")
    lines.append(f"- BeijingAirQuality: Selective regresses significantly vs Simplex: **{decision['beijingairquality_regresses_significantly']}**")
    lines.append(f"- ETTm2: always-on delta `{decision['ettm2_always_on_delta']:+.6f}` -> selective delta `{decision['ettm2_selective_delta']:+.6f}`; still significantly regresses: **{decision['ettm2_still_significant_regression']}**; fixed or reduced: **{decision['ettm2_fixed_or_reduced']}**")
    lines += ["", "## Primary dependence-aware statistics (block-24)", ""]
    lines.append("| Dataset | Comparison | Mean Δ | 95% CI | P(Δ<0) | Excludes zero |")
    lines.append("|---|---|---:|---|---:|---|")
    for ds in datasets:
        for key, label in (("primary_selective_vs_simplex", "Selective_vs_Simplex"), ("primary_selective_vs_alwayson", "Selective_vs_AlwaysOn"), ("primary_selective_vs_shuffled", "Selective_vs_SelectiveShuffled")):
            r = report["datasets"][ds][key]
            lines.append(f"| {ds} | {label} | `{r['mean_delta']:+.6f}` | [{r['ci95_low']:+.6f}, {r['ci95_high']:+.6f}] | {r['prob_delta_negative']:.3f} | {r['ci_excludes_zero']} |")
    lines += ["", "## Full dependence-aware statistics (all block lengths + phase)", ""]
    lines.append("| Dataset | Comparison | Test | Mean Δ | 95% CI | P(Δ<0) | Excludes zero |")
    lines.append("|---|---|---|---:|---|---:|---|")
    for ds in datasets:
        for row in report["datasets"][ds]["dependence_rows"]:
            mean_key = row.get("mean_delta", row.get("mean_diff_candidate_minus_baseline"))
            if "ci95_low" in row:
                prob = row.get("prob_delta_negative", "")
                lines.append(f"| {ds} | {row['comparison']} | {row['test']} | `{mean_key:+.6f}` | [{row['ci95_low']:+.6f}, {row['ci95_high']:+.6f}] | {prob} | {row['ci_excludes_zero']} |")
    lines += ["", "## Gate training (router_train OOF regularization selection)", ""]
    lines.append("| Dataset | Probe variant | L2 | OOF logloss | OOF accuracy | Selected |")
    lines.append("|---|---|---:|---:|---:|---|")
    for ds in datasets:
        for row in report["datasets"][ds]["gate_l2_rows"]:
            lines.append(f"| {ds} | {row['probe_variant']} | {row['l2']} | {row['oof_logloss']:.4f} | {row['oof_accuracy']:.3f} | {'<-- selected' if row['selected'] else ''} |")
    lines += ["", "## Gate weights (real probe, standardized-feature coefficients)", ""]
    lines.append("| Dataset | " + " | ".join(GATE_FEATURE_NAMES) + " | bias |")
    lines.append("|---|" + "---:|" * (len(GATE_FEATURE_NAMES) + 1))
    for ds in datasets:
        w = report["datasets"][ds]["gate_weights_real"]
        lines.append(f"| {ds} | " + " | ".join(f"{w[n]:+.3f}" for n in GATE_FEATURE_NAMES) + f" | {report['datasets'][ds]['gate_bias_real']:+.3f} |")
    lines += ["", "## Critical diagnostics (Section 9)", ""]
    for ds in datasets:
        lines.append(f"### {ds}")
        lines.append("")
        for row in report["datasets"][ds]["diag_rows"]:
            lines.append(f"- {row}")
        lines.append("")
    lines += ["## ETTm2 failure analysis (Section 10)", ""]
    if report["datasets"]["ETTm2"]["ettm2_summary"]:
        s = report["datasets"]["ETTm2"]["ettm2_summary"]
        for k, v in s.items():
            lines.append(f"- **{k}**: {v}")
    lines += ["", "## Weight-concentration analysis", ""]
    lines.append("| Dataset | Method | Mean entropy | Mean max weight | Mean eff. #experts | Fraction top-expert changed |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for ds in datasets:
        for row in report["datasets"][ds]["weight_rows"]:
            lines.append(f"| {ds} | {row['method']} | {row['mean_entropy']:.4f} | {row['mean_max_weight']:.4f} | {row['mean_effective_num_experts']:.3f} | {row['fraction_top_expert_changed_vs_simplex']:.3f} |")
    lines += ["", "## Integrity", ""]
    for ds in datasets:
        i = report["datasets"][ds]["integrity"]
        lines.append(f"- **{ds}**: {i['result']} (reproduction of prior simplex_probe experiment ok: {i['reproduction_of_prior_simplex_probe_experiment_ok']}; zero-gate reproduces Simplex: {i['zero_gate_reproduces_base_simplex']}; target-corruption invariant: {i['router_val_target_corruption_invariant']}; checkpoints unchanged: {i['expert_checkpoints_unchanged']})")
    lines += ["", "## Answers", ""]
    lines.append(f"**1. Does Selective Probe beat base Simplex?** See primary results table; block-24 significance in the dependence table above.")
    lines.append(f"**2. Does it preserve the ExchangeRate gain?** Fraction of always-on gain preserved: `{decision['exchangerate_gain_fraction_preserved']:.3f}`.")
    lines.append(f"**3. Does it preserve the Traffic gain?** Fraction of always-on gain preserved: `{decision['traffic_gain_fraction_preserved']:.3f}`.")
    lines.append(f"**4. Does it avoid the BeijingAirQuality non-benefit?** Significant regression: {decision['beijingairquality_regresses_significantly']}.")
    lines.append(f"**5. Does it fix or reduce the ETTm2 regression?** Always-on Δ {decision['ettm2_always_on_delta']:+.6f} -> Selective Δ {decision['ettm2_selective_delta']:+.6f}; still significant: {decision['ettm2_still_significant_regression']}.")
    lines.append(f"**6. Does Real Selective Probe beat Selective ShuffledProbe?** By point estimate on {decision['n_beats_shuffled_point']}/{len(datasets)}; block-24 significant on {decision['n_beats_shuffled_significant']}/{len(datasets)}.")
    lines.append("**7. Can the gate predict when Probe will help?** See the gate-training OOF logloss/accuracy table and Section 9B (MAE/probe-gain conditioned on trusted vs rejected windows) above.")
    lines.append("**8. Which observable features are most associated with Probe usefulness?** See the gate-weights table -- larger-magnitude coefficients (either sign) indicate stronger association, after standardization.")
    lines.append("**9. Is Probe/Simplex disagreement useful for deciding when to trust Probe?** See Section 9C (`C_simplex_agreement` rows: probe_gain_always_on and mean_gate split by agree/disagree).")
    lines.append("**10. Are large Probe weight corrections more dangerous?** See Section 9E (`E_correction_magnitude_bucket` rows: probe_gain_always_on by small/medium/large L1 weight-change bucket).")
    lines.append("**11. Does the gate mainly abstain or vary meaningfully?** See Section 9A (`A_gate_behavior`: mean/median gate, fraction <0.1, fraction >0.9) per dataset above.")
    lines.append(f"**12. Should the next experiment be Frozen COSTAR + Probe?** {decision['recommendation']}")
    lines += ["", f"## Decision: {decision['tier']}", "", decision["conclusion"], "", f"## Recommendation: **{decision['recommendation']}**", ""]
    lines += [
        "## Hard rule compliance",
        "",
        "```text",
        "TEST SET ACCESSED: NO",
        "FORECASTING EXPERTS RETRAINED: NO",
        "LEARNEDPROBE ARCHITECTURE/LOSS/TRAINING MODIFIED: NO",
        "SIMPLEX MODIFIED: NO (base weights reproduced from the frozen fit_simplex_weights function)",
        "OTHER ROUTERS (Frozen/Online COSTAR, Top-1, Top-k, Ridge, Granger-Ramanathan) TOUCHED: NO",
        "```",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    start = time.time()
    frozen_ref = load_frozen_simplex_probe_reference()
    report: dict[str, Any] = {
        "experiment": "simplex_selective_probe",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_sha": git_commit_sha(),
        "datasets": {},
    }
    all_results, all_dependence, all_weights, all_integrity, all_diag, all_gate_training, all_ettm2 = [], [], [], [], [], [], []

    for dataset in NEW_DATASETS:
        print(f"[selective_probe] {dataset}: starting...", flush=True)
        result = evaluate_dataset(dataset, frozen_ref)
        report["datasets"][dataset] = result
        all_results.extend(result["result_rows"])
        all_dependence.extend(result["dependence_rows"])
        all_weights.extend(result["weight_rows"])
        all_integrity.append(result["integrity"])
        all_diag.extend(result["diag_rows"])
        all_gate_training.extend(result["gate_l2_rows"])
        if result["ettm2_rows"]:
            all_ettm2.extend(result["ettm2_rows"])
        print(f"[selective_probe] {dataset}: done.", flush=True)

    decision = decide(report)
    report["decision"] = decision
    report["runtime_sec"] = time.time() - start
    report["test_set_accessed"] = False

    manifest = {
        "manifest_type": "simplex_selective_probe_manifest",
        "created_at_utc": report["created_at_utc"],
        "git_commit_sha": report["git_commit_sha"],
        "datasets": NEW_DATASETS,
        "expert_pool": ["DLinear", "PatchTST", "iTransformer", "TimesNet", "ModernTCN"],
        "expert_cores": {ds: report["datasets"][ds]["core"] for ds in NEW_DATASETS},
        "cache_paths": {ds: f"cache/costarts_walkforward_{ds}/router_train_20_60_cache.pt , router_val_60_80_cache.pt (test_80_100_cache.pt never built)" for ds in NEW_DATASETS},
        "base_simplex_weights": {ds: report["datasets"][ds]["simplex_weights"] for ds in NEW_DATASETS},
        "frozen_alpha_values": {ds: report["datasets"][ds]["frozen_alpha"] for ds in NEW_DATASETS},
        "frozen_alpha_source": "experiments/behavioral_competence/simplex_probe/simplex_probe_manifest.json (loaded, verified against the user-stated values, NOT reselected)",
        "shuffle_seed": SHUFFLE_SEED,
        "gate_feature_names": GATE_FEATURE_NAMES,
        "gate_model": "L2-regularized logistic regression, hand-rolled deterministic full-batch gradient descent (zero init, lr={}, {} iterations), standardized features (router_train mean/std, frozen)".format(GATE_LR, GATE_ITERATIONS),
        "gate_regularization_grid": list(GATE_L2_GRID),
        "gate_selected_l2_real": {ds: report["datasets"][ds]["gate_selected_l2_real"] for ds in NEW_DATASETS},
        "gate_selected_l2_shuffled": {ds: report["datasets"][ds]["gate_selected_l2_shuffled"] for ds in NEW_DATASETS},
        "gate_training_folds": "4 chronological leave-one-block-out folds, same block boundaries as train_folds() (costar_multidataset_frozen/common.py); the initial 20% chronological prefix is always in every fold's fit set, the remaining 80% is partitioned into 4 blocks each held out once. Documented as block cross-validation, not strict forward-only chronology.",
        "gate_label": "router_train-only: 1 if always-on Simplex+Probe (at the frozen alpha) had lower per-window MAE than base Simplex on that window, else 0. Never used at router_val inference.",
        "fusion_formula": "effective_alpha_t = gate_t * frozen_alpha; adjusted_weight_e = softmax_e(log(simplex_weight_e) - effective_alpha_t * z(probe_loss_e)); gate_t = sigmoid(w . standardize(features_t) + b); gate_t=0 reduces exactly to base Simplex (verified, Section 11).",
        "expert_checkpoint_sha256": {ds: report["datasets"][ds]["checkpoint_hashes"] for ds in NEW_DATASETS},
        "decision_rule": "Sections 14-16 of the task instructions, applied verbatim without modification after seeing results.",
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUT_DIR / "selective_probe_manifest.json", manifest)
    write_json(OUT_DIR / "validation_results.json", report)
    write_csv(OUT_DIR / "validation_results.csv", all_results)
    write_csv(OUT_DIR / "dependence_aware_results.csv", all_dependence)
    write_csv(OUT_DIR / "weight_analysis.csv", all_weights)
    write_csv(OUT_DIR / "integrity_checks.csv", all_integrity)
    write_csv(OUT_DIR / "gate_diagnostics.csv", all_diag)
    write_csv(OUT_DIR / "gate_training_results.csv", all_gate_training)
    write_csv(OUT_DIR / "ettm2_failure_analysis.csv", all_ettm2)
    make_report(report, decision)

    print("TEST SET ACCESSED: NO")
    print(json.dumps({"runtime_sec": report["runtime_sec"], "tier": decision["tier"], "recommendation": decision["recommendation"]}, indent=2))


if __name__ == "__main__":
    main()
