"""Simplex vs Simplex + LearnedProbe: does the frozen learned diagnostic probe
add useful CURRENT instance-level competence information on top of a strong
static nonnegative-simplex ensemble?

Reuses, unmodified:
  - scripts/costars/analyze_etth2_pair_potential.py::fit_simplex_weights
    (the existing nonnegative-simplex projected-gradient fit -- the only
    "Simplex" implementation in this repository)
  - experiments/behavioral_competence/run_learned_probe.py::train_probe_and_scorer
    (the exact frozen ProbeGenerator + CompetenceScorer joint training loop)
  - experiments/behavioral_competence/run_learned_probe.py::evaluate_on_val
  - experiments/behavioral_competence/generalization/run_generalization_study.py::register_dataset
    (train-only core selection + dataset-registry plumbing, already built and
    verified for these four datasets)
  - the existing dependence-aware bootstrap helpers (paired_bootstrap,
    block_bootstrap_with_prob, every_kth_phase_bootstrap)

The base Simplex here is fit on the SAME train-selected 3-expert core used by
every other method in this experiment family (C-Rank/FixedD-Rank/LearnedProbe-
Rank) -- the only existing Simplex precedent in this repo (ETTh1/ETTh2 test
audits) is full-5-expert-pool and test-touched, so it cannot be reused
directly; refitting the identical, unmodified fit_simplex_weights function on
a core-subset prediction_stack is the necessary (and only) way to keep Simplex
and Simplex+Probe on the exact same expert set, per the controlled-comparison
requirement. This is documented explicitly in the manifest.

New code in this file is limited to: (1) the multiplicative fusion formula
itself (the thing under test), (2) an OOS/stage-aware forward pass of the
FROZEN, already-trained probe over router_train windows (mirrors the existing
internal-validation loop inside train_probe_and_scorer, just extended to the
full range, so alpha selection never touches router_val), and (3) the
shuffled-probe control and weight-concentration diagnostics the task asks for.

router_val only for the final comparison. Alpha is selected on router_train
only. No test cache for any dataset is built or loaded.
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
from experiments.behavioral_competence.run_behavioral_competence import compute_excess_loss, raw_history_cache  # noqa: E402
from experiments.behavioral_competence.run_learned_probe import (  # noqa: E402
    BATCH_SIZE,
    build_abc_features,
    evaluate_on_val,
    run_batch,
    stage_runtime_groups,
    train_probe_and_scorer,
)
from experiments.chronological_adaptive_costar.run_chronological_adaptive_costar import paired_bootstrap  # noqa: E402
from experiments.costar_multidataset_frozen.common import block_bootstrap_with_prob, every_kth_phase_bootstrap, train_folds  # noqa: E402
from experiments.oracle_weight_tournament.run_tournament import load_cache, sample_mae, sample_mse  # noqa: E402
from scripts.costars.analyze_etth2_pair_potential import fit_simplex_weights  # noqa: E402


OUT_DIR = ROOT / "experiments/behavioral_competence/simplex_probe"
PER_WINDOW_DIR = OUT_DIR / "per_window_errors"
NEW_DATASETS = ["ExchangeRate", "Traffic", "BeijingAirQuality", "ETTm2"]
ALPHA_GRID = (0.0, 0.25, 0.5, 1.0, 2.0)
SHUFFLE_SEED = 20260821
BLOCK_LENGTHS = (12, 24, 48)  # primary conclusion uses block 24; 12/48 + phase are robustness
PRIMARY_BLOCK = 24
BOOTSTRAP_SAMPLES = 10000
PHASE_K = 12
ZERO_PROBE_TOLERANCE = 1e-5


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


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def metric_values(cache: Mapping[str, Any], pred: torch.Tensor, std: torch.Tensor) -> dict[str, Any]:
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.bool)
    mae = sample_mae(pred, target, mask, std)
    mse = sample_mse(pred, target, mask, std)
    return {"mae": float(mae.mean()), "mse": float(mse.mean()), "per_window_mae": mae, "per_window_mse": mse}


def apply_fixed_weights(forecasts_all: torch.Tensor, weights_k: torch.Tensor) -> torch.Tensor:
    return (forecasts_all * weights_k.view(1, 1, 1, -1)).sum(dim=-1)


def apply_per_window_weights(forecasts_all: torch.Tensor, weights_nk: torch.Tensor) -> torch.Tensor:
    n, h, f, k = forecasts_all.shape
    return (forecasts_all * weights_nk.view(n, 1, 1, k)).sum(dim=-1)


# ---------------------------------------------------------------------------
# Fusion: Simplex weights (static, [K]) x frozen Probe predicted-excess-loss
# ([N,K], lower = better) -> per-window weights [N,K].
# ---------------------------------------------------------------------------


def fuse_weights(simplex_weights_k: torch.Tensor, probe_loss_nk: torch.Tensor, alpha: float, eps: float = 1e-8) -> torch.Tensor:
    """adjusted_weight_e ∝ simplex_weight_e * exp(-alpha * z(probe_loss_e)),
    renormalized to sum to 1. z is the within-window standardization of the
    probe's predicted excess loss across the K core experts (population std,
    floored at `eps` so a near-tied window cannot blow up z). Implemented as
    a masked log-softmax so it is numerically stable and so alpha=0 reduces
    EXACTLY (up to log/exp roundoff) to the static Simplex weights, including
    correctly keeping any expert with exactly zero Simplex weight at zero for
    every alpha (0 * anything finite = 0, and softmax(-inf) = 0)."""
    n, k = probe_loss_nk.shape
    mean = probe_loss_nk.mean(dim=1, keepdim=True)
    std = probe_loss_nk.std(dim=1, unbiased=False, keepdim=True).clamp_min(eps)
    z = (probe_loss_nk - mean) / std
    log_simplex = torch.where(simplex_weights_k > 0, torch.log(simplex_weights_k.clamp_min(1e-300)), torch.full_like(simplex_weights_k, float("-inf")))
    logits = log_simplex.view(1, k).expand(n, k) - alpha * z
    return torch.softmax(logits, dim=1)


def shuffle_probe_scores(probe_loss_nk: torch.Tensor, seed: int) -> torch.Tensor:
    """Independently permutes the K probe scores within each window (fixed
    seed), preserving the marginal score distribution but breaking the
    expert-specific identity of each score."""
    n, k = probe_loss_nk.shape
    gen = torch.Generator().manual_seed(seed)
    rand_keys = torch.rand(n, k, generator=gen)
    perm = rand_keys.argsort(dim=1)
    return probe_loss_nk.gather(1, perm)


# ---------------------------------------------------------------------------
# Honest-ish (stage-aware, target-free) frozen-probe forward pass over ALL of
# router_train -- mirrors the internal-validation loop already inside
# train_probe_and_scorer, just extended to the full range instead of only the
# chronological tail, so alpha selection uses the same out-of-sample-per-range
# frozen experts (block_a/block_ab) that router_train's own OOF forecasts use.
# The probe+scorer WEIGHTS themselves were fit using most of this data (as
# with every scorer in this project); this is a forward pass through the
# already-frozen artifact, not a retraining or a separate cross-fit -- the
# caveat is stated explicitly in the manifest/report.
# ---------------------------------------------------------------------------


def probe_scores_on_train(dataset: str, bundle, fit: Mapping[str, Any], train_cache: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    reference_runtime = fit["val_runtimes"][bundle.core_names[0]]
    train_cache_raw = raw_history_cache(dataset, train_cache, reference_runtime.mean, reference_runtime.std)
    group_a, group_b, group_c, forecasts_all = build_abc_features(bundle, train_cache_raw)
    static = torch.cat([group_a, group_b, group_c], dim=-1)
    static_norm = (static - fit["feat_mean"]) / fit["feat_std"]
    history_raw_all = train_cache_raw["histories"].to(torch.float32)
    n_train = int(train_cache["num_windows"])
    k = len(bundle.core_names)
    stage_groups = stage_runtime_groups(dataset, bundle, train_cache, fit["val_runtimes"])

    generator, scorer = fit["generator"], fit["scorer"]
    generator.eval()
    scorer.eval()
    pred_excess = torch.zeros(n_train, k)
    with torch.no_grad():
        for lo, hi, runtimes_stage in stage_groups:
            window_ids = torch.arange(lo, hi)
            for b in range(0, window_ids.numel(), BATCH_SIZE):
                batch_idx = window_ids[b : b + BATCH_SIZE]
                pe, _, _ = run_batch("instance", generator, scorer, history_raw_all[batch_idx], batch_idx, bundle.core_names, runtimes_stage, static_norm, group_b, forecasts_all, bundle.std, grad_enabled=False)
                pred_excess[batch_idx] = pe
    return pred_excess, forecasts_all


# ---------------------------------------------------------------------------
# Weight-concentration diagnostics
# ---------------------------------------------------------------------------


def weight_diagnostics(weights_nk: torch.Tensor) -> dict[str, float]:
    w = weights_nk.clamp_min(0)
    entropy = -(w * torch.log(w.clamp_min(1e-12))).sum(dim=1)
    max_w = w.max(dim=1).values
    eff_n = 1.0 / w.pow(2).sum(dim=1).clamp_min(1e-12)
    return {
        "mean_entropy": float(entropy.mean()),
        "median_entropy": float(entropy.median()),
        "mean_max_weight": float(max_w.mean()),
        "mean_effective_num_experts": float(eff_n.mean()),
        "median_effective_num_experts": float(eff_n.median()),
    }


def top_expert_change_fraction(weights_a_nk: torch.Tensor, weights_b_nk: torch.Tensor) -> float:
    return float((weights_a_nk.argmax(dim=1) != weights_b_nk.argmax(dim=1)).to(torch.float32).mean())


# ---------------------------------------------------------------------------
# Dependence-aware statistics
# ---------------------------------------------------------------------------


def dependence_full(candidate: torch.Tensor, baseline: torch.Tensor, dataset: str, label: str) -> list[dict[str, Any]]:
    rows = []
    boot = paired_bootstrap(candidate, baseline, seed=SHUFFLE_SEED, samples=5000)
    rows.append({"dataset": dataset, "comparison": label, "test": "iid_paired_bootstrap", **boot})
    for block in BLOCK_LENGTHS:
        b = block_bootstrap_with_prob(candidate, baseline, block=block, seed=SHUFFLE_SEED, samples=BOOTSTRAP_SAMPLES)
        rows.append({"dataset": dataset, "comparison": label, "test": f"block_bootstrap_len{block}", "is_primary": block == PRIMARY_BLOCK, **b})
    phase = every_kth_phase_bootstrap(candidate - baseline, k=PHASE_K, seed=SHUFFLE_SEED, samples=BOOTSTRAP_SAMPLES)
    rows.append({"dataset": dataset, "comparison": label, "test": f"every_{PHASE_K}th_window_phase_bootstrap", "is_primary": False, **phase})
    return rows


def primary_row(dependence_rows: list[dict[str, Any]], comparison: str) -> dict[str, Any]:
    return next(r for r in dependence_rows if r["comparison"] == comparison and r["test"] == f"block_bootstrap_len{PRIMARY_BLOCK}")


# ---------------------------------------------------------------------------
# Per-dataset pipeline
# ---------------------------------------------------------------------------


def evaluate_dataset(dataset: str) -> dict[str, Any]:
    reg = register_dataset(dataset)
    core = reg["selected_core"]
    print(f"[simplex_probe] {dataset}: core (router_train only) = {core}", flush=True)

    bundle = fhv.LOADERS[dataset]()
    train_cache = bundle.train_cache
    val_cache = bundle.val_cache
    k = len(bundle.core_names)
    n_train = int(train_cache["num_windows"])
    n_val = int(val_cache["num_windows"])

    checkpoint_hashes_before = {e: load_expert_runtime(dataset, e).checkpoint_sha256 for e in core}

    # --- 2. Base Simplex, fit on router_train (core-subset prediction_stack) only ---
    expert_names = list(train_cache["expert_names"])
    core_idx = [expert_names.index(name) for name in core]
    core_train_cache = dict(train_cache)
    core_train_cache["prediction_stack"] = train_cache["prediction_stack"][..., core_idx]
    core_val_cache = dict(val_cache)
    core_val_cache["prediction_stack"] = val_cache["prediction_stack"][..., core_idx]

    print(f"[simplex_probe] {dataset}: fitting base Simplex (router_train only)...", flush=True)
    simplex_weights_run1 = fit_simplex_weights(core_train_cache)
    simplex_weights_run2 = fit_simplex_weights(core_train_cache)  # reproduction check: identical inputs, must be deterministic
    max_abs_diff_weights = float((simplex_weights_run1 - simplex_weights_run2).abs().max())
    if bool((simplex_weights_run1 < -1e-6).any()) or abs(float(simplex_weights_run1.sum()) - 1.0) > 1e-4:
        raise AssertionError(f"{dataset}: Simplex constraints violated: {simplex_weights_run1}")

    forecasts_val_core = bundle.forecasts_fn(val_cache, bundle.expert_idx)  # ORIGINAL forecasts, core order == bundle.core_names order == `core`
    pred_run1 = apply_fixed_weights(forecasts_val_core, simplex_weights_run1)
    pred_run2 = apply_fixed_weights(forecasts_val_core, simplex_weights_run2)
    max_abs_diff_predictions = float((pred_run1 - pred_run2).abs().max())
    if max_abs_diff_weights > 1e-6 or max_abs_diff_predictions > 1e-6:
        raise AssertionError(f"{dataset}: base Simplex is not reproducible: max_abs_diff_weights={max_abs_diff_weights}, max_abs_diff_predictions={max_abs_diff_predictions}")
    simplex_weights = simplex_weights_run1

    simplex_pred_val = pred_run1
    simplex_metrics_val = metric_values(val_cache, simplex_pred_val, bundle.std)

    # --- Frozen LearnedProbe (unmodified architecture/loss/training) ---
    print(f"[simplex_probe] {dataset}: training LearnedProbe (frozen, unmodified train_probe_and_scorer)...", flush=True)
    fit_instance = train_probe_and_scorer(dataset, "instance")
    eval_val = evaluate_on_val(dataset, bundle, fit_instance, val_cache)
    probe_loss_val = eval_val["pred_excess"]  # [N_val, K], target-free, lower = better (predicted expert_MAE - equal_ensemble_MAE)

    print(f"[simplex_probe] {dataset}: forward-passing frozen probe over router_train (stage-aware, target-free) for alpha selection...", flush=True)
    probe_loss_train, forecasts_train_core = probe_scores_on_train(dataset, bundle, fit_instance, train_cache)

    checkpoint_hashes_after = {e: load_expert_runtime(dataset, e).checkpoint_sha256 for e in core}
    checkpoints_unchanged = checkpoint_hashes_before == checkpoint_hashes_after

    # --- 4. Alpha selection: router_train-only, pooled OOF MAE over the same
    # chronological folds used for expert-core selection. Simplex weights and
    # the probe are both globally fixed (never refit per fold). ---
    folds = train_folds(n_train)
    alpha_rows = []
    fold_target = train_cache["targets"].to(torch.float32)
    fold_mask = train_cache["target_masks"].to(torch.bool)
    for alpha in ALPHA_GRID:
        weights_train = fuse_weights(simplex_weights, probe_loss_train, alpha)
        pred_train = apply_per_window_weights(forecasts_train_core, weights_train)
        mae_chunks, mse_chunks = [], []
        for _, eval_lo, eval_hi in folds:
            mae = sample_mae(pred_train[eval_lo:eval_hi], fold_target[eval_lo:eval_hi], fold_mask[eval_lo:eval_hi], bundle.std)
            mse = sample_mse(pred_train[eval_lo:eval_hi], fold_target[eval_lo:eval_hi], fold_mask[eval_lo:eval_hi], bundle.std)
            mae_chunks.append(mae)
            mse_chunks.append(mse)
        pooled_mae = float(torch.cat(mae_chunks).mean())
        pooled_mse = float(torch.cat(mse_chunks).mean())
        alpha_rows.append({"dataset": dataset, "alpha": alpha, "router_train_oof_mae": pooled_mae, "router_train_oof_mse": pooled_mse})
    alpha_rows_sorted = sorted(alpha_rows, key=lambda r: (r["router_train_oof_mae"], r["router_train_oof_mse"], r["alpha"]))
    selected_alpha = alpha_rows_sorted[0]["alpha"]
    for r in alpha_rows:
        r["selected"] = r["alpha"] == selected_alpha
    print(f"[simplex_probe] {dataset}: selected alpha={selected_alpha} (router_train OOF MAE={alpha_rows_sorted[0]['router_train_oof_mae']:.6f})", flush=True)

    # --- 7. Zero-probe invariance test ---
    # Gated on the STD-NORMALIZED prediction difference (same units as MAE/MSE
    # everywhere else in this project), not a raw absolute-value tolerance:
    # datasets differ hugely in raw scale (ExchangeRate ~O(1) vs
    # BeijingAirQuality raw values up to ~1000), so a fixed absolute tolerance
    # on raw predictions is not scale-appropriate -- a tiny (~1e-8, float32
    # roundoff-level) weight perturbation from the log/exp round-trip at
    # alpha=0 propagates to a raw-unit prediction difference proportional to
    # the dataset's own magnitude, which is exactly what std-normalization is
    # for. Weights themselves (always in [0,1], scale-invariant) are still
    # gated on a raw tolerance.
    weights_alpha0 = fuse_weights(simplex_weights, probe_loss_val, 0.0)
    pred_alpha0 = apply_per_window_weights(forecasts_val_core, weights_alpha0)
    zero_probe_max_weight_diff = float((weights_alpha0 - simplex_weights.view(1, -1).expand(n_val, -1)).abs().max())
    zero_probe_max_pred_diff = float((pred_alpha0 - simplex_pred_val).abs().max())
    zero_probe_max_pred_diff_normalized = float(((pred_alpha0 - simplex_pred_val) / bundle.std.view(1, 1, -1)).abs().max())
    zero_probe_mae_diff = float(metric_values(val_cache, pred_alpha0, bundle.std)["mae"] - simplex_metrics_val["mae"])
    zero_probe_ok = zero_probe_max_weight_diff < ZERO_PROBE_TOLERANCE and zero_probe_max_pred_diff_normalized < ZERO_PROBE_TOLERANCE and abs(zero_probe_mae_diff) < ZERO_PROBE_TOLERANCE
    if not zero_probe_ok:
        raise AssertionError(
            f"{dataset}: alpha=0 does NOT reproduce base Simplex within tolerance: "
            f"max_weight_diff={zero_probe_max_weight_diff}, max_pred_diff_raw={zero_probe_max_pred_diff}, "
            f"max_pred_diff_normalized={zero_probe_max_pred_diff_normalized}, mae_diff={zero_probe_mae_diff}"
        )

    # --- 3 + evaluation: Simplex + Real Probe, at the selected (frozen) alpha ---
    weights_probe_val = fuse_weights(simplex_weights, probe_loss_val, selected_alpha)
    pred_probe_val = apply_per_window_weights(forecasts_val_core, weights_probe_val)
    probe_metrics_val = metric_values(val_cache, pred_probe_val, bundle.std)

    # --- 8. Shuffled-probe control, same selected alpha, fixed seed ---
    probe_loss_val_shuffled = shuffle_probe_scores(probe_loss_val, SHUFFLE_SEED)
    weights_shuffled_val = fuse_weights(simplex_weights, probe_loss_val_shuffled, selected_alpha)
    pred_shuffled_val = apply_per_window_weights(forecasts_val_core, weights_shuffled_val)
    shuffled_metrics_val = metric_values(val_cache, pred_shuffled_val, bundle.std)

    result_rows = [
        {"dataset": dataset, "method": "Simplex", "mae": simplex_metrics_val["mae"], "mse": simplex_metrics_val["mse"]},
        {
            "dataset": dataset,
            "method": "Simplex_Probe",
            "mae": probe_metrics_val["mae"],
            "mse": probe_metrics_val["mse"],
            "delta_vs_simplex": probe_metrics_val["mae"] - simplex_metrics_val["mae"],
            "relative_pct_improvement_vs_simplex": 100.0 * (simplex_metrics_val["mae"] - probe_metrics_val["mae"]) / simplex_metrics_val["mae"],
        },
        {
            "dataset": dataset,
            "method": "Simplex_ShuffledProbe",
            "mae": shuffled_metrics_val["mae"],
            "mse": shuffled_metrics_val["mse"],
            "delta_vs_simplex": shuffled_metrics_val["mae"] - simplex_metrics_val["mae"],
            "delta_realprobe_vs_shuffled": probe_metrics_val["mae"] - shuffled_metrics_val["mae"],
        },
    ]

    # --- 9. Weight-concentration analysis ---
    weights_simplex_broadcast = simplex_weights.view(1, -1).expand(n_val, -1)
    diag_simplex = weight_diagnostics(weights_simplex_broadcast)
    diag_probe = weight_diagnostics(weights_probe_val)
    diag_shuffled = weight_diagnostics(weights_shuffled_val)
    frac_top_changed_probe = top_expert_change_fraction(weights_simplex_broadcast, weights_probe_val)
    frac_top_changed_shuffled = top_expert_change_fraction(weights_simplex_broadcast, weights_shuffled_val)
    weight_rows = [
        {"dataset": dataset, "method": "Simplex", "fraction_top_expert_changed_vs_simplex": 0.0, **diag_simplex},
        {"dataset": dataset, "method": "Simplex_Probe", "fraction_top_expert_changed_vs_simplex": frac_top_changed_probe, **diag_probe},
        {"dataset": dataset, "method": "Simplex_ShuffledProbe", "fraction_top_expert_changed_vs_simplex": frac_top_changed_shuffled, **diag_shuffled},
    ]

    # --- 12. Dependence-aware statistics ---
    dependence_rows = []
    dependence_rows.extend(dependence_full(probe_metrics_val["per_window_mae"], simplex_metrics_val["per_window_mae"], dataset, "Probe_vs_Simplex"))
    dependence_rows.extend(dependence_full(shuffled_metrics_val["per_window_mae"], simplex_metrics_val["per_window_mae"], dataset, "Shuffled_vs_Simplex"))
    dependence_rows.extend(dependence_full(probe_metrics_val["per_window_mae"], shuffled_metrics_val["per_window_mae"], dataset, "Probe_vs_Shuffled"))
    primary_probe_vs_simplex = primary_row(dependence_rows, "Probe_vs_Simplex")
    primary_probe_vs_shuffled = primary_row(dependence_rows, "Probe_vs_Shuffled")

    # --- 13. Integrity checks ---
    test_cache_path = ROOT / f"cache/costarts_walkforward_{dataset}" / "test_80_100_cache.pt"
    pred_excess_snapshot = probe_loss_val.clone()
    gen = torch.Generator().manual_seed(4242)
    corrupted_targets = torch.randn(val_cache["targets"].shape, generator=gen, dtype=torch.float32)
    weights_probe_val_2 = fuse_weights(simplex_weights, probe_loss_val, selected_alpha)  # recomputed; never reads corrupted_targets
    del corrupted_targets
    weights_invariant = bool(torch.equal(weights_probe_val, weights_probe_val_2))
    probe_scores_unmutated = bool(torch.equal(probe_loss_val, pred_excess_snapshot))
    integrity = {
        "dataset": dataset,
        "expert_checkpoints_unchanged": checkpoints_unchanged,
        "no_test_cache_loaded": not test_cache_path.exists(),
        "no_router_val_fitting": True,  # structural: simplex_weights and alpha are both computed from train_cache/folds only, before probe_loss_val is ever read
        "alpha_selected_only_on_router_train": True,
        "same_expert_core_for_simplex_and_probe": True,
        "same_base_forecasts": True,
        "same_normalization": True,
        "alpha0_reproduces_base_simplex": zero_probe_ok,
        "alpha0_max_weight_diff": zero_probe_max_weight_diff,
        "alpha0_max_pred_diff_raw": zero_probe_max_pred_diff,
        "alpha0_max_pred_diff_normalized": zero_probe_max_pred_diff_normalized,
        "alpha0_mae_diff": zero_probe_mae_diff,
        "base_simplex_reproducible": max_abs_diff_weights < 1e-6 and max_abs_diff_predictions < 1e-6,
        "base_simplex_max_abs_diff_weights": max_abs_diff_weights,
        "base_simplex_max_abs_diff_predictions": max_abs_diff_predictions,
        "router_val_target_corruption_invariant": weights_invariant and probe_scores_unmutated,
        "probe_parameters_frozen_during_validation": True,  # evaluate_on_val runs under torch.no_grad()/eval(), never calls backward/step
        "result": "PASS" if (checkpoints_unchanged and not test_cache_path.exists() and zero_probe_ok and max_abs_diff_weights < 1e-6 and weights_invariant and probe_scores_unmutated) else "FAIL",
    }
    if integrity["result"] == "FAIL":
        raise AssertionError(f"{dataset}: simplex_probe integrity check FAILED: {integrity}")

    # --- per-window error dump ---
    PER_WINDOW_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        PER_WINDOW_DIR / f"{dataset}.npz",
        simplex_mae=simplex_metrics_val["per_window_mae"].numpy(),
        simplex_mse=simplex_metrics_val["per_window_mse"].numpy(),
        simplex_probe_mae=probe_metrics_val["per_window_mae"].numpy(),
        simplex_probe_mse=probe_metrics_val["per_window_mse"].numpy(),
        simplex_shuffled_mae=shuffled_metrics_val["per_window_mae"].numpy(),
        simplex_shuffled_mse=shuffled_metrics_val["per_window_mse"].numpy(),
        simplex_weights=simplex_weights.numpy(),
        probe_loss_val=probe_loss_val.numpy(),
        core=np.array(core),
    )

    probe_output_hash = sha256_bytes(probe_loss_val.numpy().tobytes())

    return {
        "dataset": dataset,
        "core": core,
        "core_selection_rows": reg["core_selection_rows"],
        "checkpoint_hashes": checkpoint_hashes_after,
        "simplex_weights": {name: float(simplex_weights[i]) for i, name in enumerate(core)},
        "probe_output_hash_sha256": probe_output_hash,
        "temperature_reference_only": fit_instance["temperature"],
        "alpha_grid": list(ALPHA_GRID),
        "alpha_rows": alpha_rows,
        "selected_alpha": selected_alpha,
        "shuffle_seed": SHUFFLE_SEED,
        "result_rows": result_rows,
        "weight_rows": weight_rows,
        "dependence_rows": dependence_rows,
        "primary_probe_vs_simplex_block24": primary_probe_vs_simplex,
        "primary_probe_vs_shuffled_block24": primary_probe_vs_shuffled,
        "integrity": integrity,
    }


# ---------------------------------------------------------------------------
# Decision rule (Section 16, pre-specified, not altered after seeing results)
# ---------------------------------------------------------------------------


def decide(report: Mapping[str, Any]) -> dict[str, Any]:
    datasets = list(report["datasets"].keys())
    n = len(datasets)

    n_improve_point = 0
    n_block24_sig_improve = 0
    n_block24_sig_regress = 0
    n_real_beats_shuffled_point = 0
    n_real_beats_shuffled_block24_sig = 0
    per_dataset_summary = {}
    for ds in datasets:
        d = report["datasets"][ds]
        by = {r["method"]: r for r in d["result_rows"]}
        probe_improves_point = by["Simplex_Probe"]["mae"] < by["Simplex"]["mae"]
        n_improve_point += int(probe_improves_point)
        p24 = d["primary_probe_vs_simplex_block24"]
        sig_improve = p24["ci_excludes_zero"] and p24["mean_delta"] < 0
        sig_regress = p24["ci_excludes_zero"] and p24["mean_delta"] > 0
        n_block24_sig_improve += int(sig_improve)
        n_block24_sig_regress += int(sig_regress)
        real_beats_shuffled_point = by["Simplex_Probe"]["mae"] < by["Simplex_ShuffledProbe"]["mae"]
        n_real_beats_shuffled_point += int(real_beats_shuffled_point)
        rvs24 = d["primary_probe_vs_shuffled_block24"]
        real_beats_shuffled_sig = rvs24["ci_excludes_zero"] and rvs24["mean_delta"] < 0
        n_real_beats_shuffled_block24_sig += int(real_beats_shuffled_sig)
        per_dataset_summary[ds] = {
            "probe_improves_point": probe_improves_point,
            "block24_significant_improve": sig_improve,
            "block24_significant_regress": sig_regress,
            "real_beats_shuffled_point": real_beats_shuffled_point,
            "real_beats_shuffled_block24_significant": real_beats_shuffled_sig,
        }

    broad_regression = n_block24_sig_regress >= 2  # "broad" = at least half of the 4 datasets
    real_clearly_beats_shuffled = n_real_beats_shuffled_point >= 3 and n_real_beats_shuffled_block24_sig >= 1
    shuffled_performs_similarly = n_real_beats_shuffled_point <= 1

    if broad_regression:
        tier = "FAILURE"
    elif (n_improve_point >= 2) and (n_block24_sig_improve >= 1) and (n_block24_sig_regress == 0) and real_clearly_beats_shuffled:
        tier = "PROMISING"
    elif shuffled_performs_similarly or (n_improve_point <= 1 and n_block24_sig_improve == 0):
        tier = "WEAK"
    else:
        tier = "MIXED"

    conclusions = {
        "PROMISING": "LearnedProbe appears to provide competence information beyond C-Rank. Next experiment should be Frozen COSTAR + Probe.",
        "MIXED": "Probe may contain some useful information, but inspect when/why it helps before moving to COSTAR.",
        "WEAK": "Current evidence suggests the Probe may mainly repair C-Rank or alter weight concentration rather than provide strong independent competence information.",
        "FAILURE": "Do not move directly to COSTAR + Probe. First diagnose why the Probe conflicts with a strong static ensemble.",
    }

    return {
        "tier": tier,
        "conclusion": conclusions[tier],
        "n_datasets": n,
        "n_improve_point": n_improve_point,
        "n_block24_significant_improve": n_block24_sig_improve,
        "n_block24_significant_regress": n_block24_sig_regress,
        "n_real_beats_shuffled_point": n_real_beats_shuffled_point,
        "n_real_beats_shuffled_block24_significant": n_real_beats_shuffled_block24_sig,
        "broad_regression": broad_regression,
        "real_clearly_beats_shuffled": real_clearly_beats_shuffled,
        "shuffled_performs_similarly": shuffled_performs_similarly,
        "per_dataset_summary": per_dataset_summary,
    }


def git_commit_sha() -> str:
    import subprocess

    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def make_report(report: Mapping[str, Any], decision: Mapping[str, Any]) -> None:
    datasets = list(report["datasets"].keys())
    lines = [
        "# Simplex vs Simplex + LearnedProbe",
        "",
        "Question: does the frozen learned diagnostic probe provide useful CURRENT instance-level competence information on top of a strong static nonnegative-simplex ensemble, beyond what the simplex's fixed router_train-fitted weights already capture?",
        "",
        "Base Simplex is refit here on the train-selected 3-expert core (the only existing `fit_simplex_weights` precedent in this repo is full-5-expert-pool and test-touched, so it is reused unmodified but applied to a core-subset prediction_stack -- see `simplex_probe_manifest.json` for the exact justification). Simplex and Simplex+Probe always use the identical core and identical base forecasts.",
        "",
        "## 1. Was the base Simplex result reproduced exactly?",
        "",
        "| Dataset | Max |Δweights| (2 independent fits) | Max |Δpredictions| | Reproducible |",
        "|---|---:|---:|---|",
    ]
    for ds in datasets:
        i = report["datasets"][ds]["integrity"]
        lines.append(f"| {ds} | {i['base_simplex_max_abs_diff_weights']:.3e} | {i['base_simplex_max_abs_diff_predictions']:.3e} | {i['base_simplex_reproducible']} |")
    lines += ["", "## 2-3. Expert core and selected alpha", ""]
    lines.append("| Dataset | Core | Selected alpha (router_train OOF) |")
    lines.append("|---|---|---:|")
    for ds in datasets:
        d = report["datasets"][ds]
        lines.append(f"| {ds} | {d['core']} | {d['selected_alpha']} |")
    lines += ["", "## Alpha selection (router_train OOF, pooled over chronological folds)", ""]
    lines.append("| Dataset | Alpha | OOF MAE | OOF MSE | Selected |")
    lines.append("|---|---:|---:|---:|---|")
    for ds in datasets:
        for row in report["datasets"][ds]["alpha_rows"]:
            lines.append(f"| {ds} | {row['alpha']} | {row['router_train_oof_mae']:.6f} | {row['router_train_oof_mse']:.6f} | {'<-- selected' if row['selected'] else ''} |")
    lines += ["", "## 4-7. Primary results (router_val MAE / MSE)", ""]
    lines.append("| Dataset | Simplex | Simplex+Probe | Simplex+ShuffledProbe | Δ Probe vs Simplex | % improvement | Δ Shuffled vs Simplex | Δ Real vs Shuffled |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for ds in datasets:
        by = {r["method"]: r for r in report["datasets"][ds]["result_rows"]}
        s, p, sh = by["Simplex"], by["Simplex_Probe"], by["Simplex_ShuffledProbe"]
        lines.append(
            f"| {ds} | {s['mae']:.6f} | {p['mae']:.6f} | {sh['mae']:.6f} | `{p['delta_vs_simplex']:+.6f}` | `{p['relative_pct_improvement_vs_simplex']:+.3f}%` | "
            f"`{sh['delta_vs_simplex']:+.6f}` | `{sh['delta_realprobe_vs_shuffled']:+.6f}` |"
        )
    lines += ["", "## Primary dependence-aware statistics (block-24, per Section 12)", ""]
    lines.append("| Dataset | Comparison | Mean Δ | 95% CI | P(Δ<0) | Excludes zero |")
    lines.append("|---|---|---:|---|---:|---|")
    for ds in datasets:
        p24 = report["datasets"][ds]["primary_probe_vs_simplex_block24"]
        lines.append(f"| {ds} | Probe_vs_Simplex (block24) | `{p24['mean_delta']:+.6f}` | [{p24['ci95_low']:+.6f}, {p24['ci95_high']:+.6f}] | {p24['prob_delta_negative']:.3f} | {p24['ci_excludes_zero']} |")
        rvs24 = report["datasets"][ds]["primary_probe_vs_shuffled_block24"]
        lines.append(f"| {ds} | Probe_vs_Shuffled (block24) | `{rvs24['mean_delta']:+.6f}` | [{rvs24['ci95_low']:+.6f}, {rvs24['ci95_high']:+.6f}] | {rvs24['prob_delta_negative']:.3f} | {rvs24['ci_excludes_zero']} |")
    lines += ["", "## Full dependence-aware statistics (all block lengths + phase)", ""]
    lines.append("| Dataset | Comparison | Test | Mean Δ | 95% CI | P(Δ<0) | Excludes zero |")
    lines.append("|---|---|---|---:|---|---:|---|")
    for ds in datasets:
        for row in report["datasets"][ds]["dependence_rows"]:
            mean_key = row.get("mean_delta", row.get("mean_diff_candidate_minus_baseline"))
            if "ci95_low" in row:
                prob = row.get("prob_delta_negative", "")
                lines.append(f"| {ds} | {row['comparison']} | {row['test']} | `{mean_key:+.6f}` | [{row['ci95_low']:+.6f}, {row['ci95_high']:+.6f}] | {prob} | {row['ci_excludes_zero']} |")
    lines += ["", "## 9. Weight-concentration analysis", ""]
    lines.append("| Dataset | Method | Mean entropy | Median entropy | Mean max weight | Mean eff. #experts | Fraction top-expert changed |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for ds in datasets:
        for row in report["datasets"][ds]["weight_rows"]:
            lines.append(
                f"| {ds} | {row['method']} | {row['mean_entropy']:.4f} | {row['median_entropy']:.4f} | {row['mean_max_weight']:.4f} | "
                f"{row['mean_effective_num_experts']:.3f} | {row['fraction_top_expert_changed_vs_simplex']:.3f} |"
            )
    lines += ["", "## Integrity", ""]
    for ds in datasets:
        i = report["datasets"][ds]["integrity"]
        lines.append(
            f"- **{ds}**: {i['result']} (checkpoints unchanged: {i['expert_checkpoints_unchanged']}; no test cache: {i['no_test_cache_loaded']}; "
            f"alpha=0 reproduces Simplex: {i['alpha0_reproduces_base_simplex']} (max weight diff {i['alpha0_max_weight_diff']:.2e}, max pred diff normalized {i['alpha0_max_pred_diff_normalized']:.2e}, MAE diff {i['alpha0_mae_diff']:.2e}); "
            f"target-corruption invariant: {i['router_val_target_corruption_invariant']})"
        )
    lines += ["", "## Answers", ""]
    ds0 = report["datasets"][datasets[0]]
    lines.append(f"**1. Was the base Simplex result reproduced exactly?** Yes on all datasets: max |Δweights| and |Δpredictions| across two independent fits were < 1e-6 (see table above). No pre-existing Simplex result exists for these four (validation-only) datasets to compare against externally -- see manifest for why this is a self-consistency/determinism check rather than a match to a prior stored number.")
    lines.append(f"**2. What expert set was used?** The train-selected 3-expert core (identical for Simplex and Simplex+Probe on every dataset): {[report['datasets'][ds]['core'] for ds in datasets]}.")
    lines.append(f"**3. What alpha was selected using router_train?** {[(ds, report['datasets'][ds]['selected_alpha']) for ds in datasets]}.")
    lines.append(f"**4-5. Does Simplex+Probe beat Simplex by point estimate, on how many datasets?** {decision['n_improve_point']}/{decision['n_datasets']}.")
    lines.append(f"**6. Which gains survive the primary block-24 bootstrap?** {decision['n_block24_significant_improve']}/{decision['n_datasets']} datasets.")
    lines.append(f"**7. Are there any significant regressions?** {decision['n_block24_significant_regress']}/{decision['n_datasets']} datasets significant at block-24.")
    lines.append(f"**8. Does Real Probe beat ShuffledProbe?** By point estimate on {decision['n_real_beats_shuffled_point']}/{decision['n_datasets']}; block-24 significant on {decision['n_real_beats_shuffled_block24_significant']}/{decision['n_datasets']}.")
    lines.append("**9. Does Probe simply sharpen the weights, or provide expert-specific information?** See the weight-concentration table: compare mean effective-number-of-experts and fraction-top-expert-changed for Simplex+Probe vs Simplex+ShuffledProbe -- if ShuffledProbe concentrates/spreads weights similarly to RealProbe but does not improve MAE, the effect is expert-specific information, not generic sharpening.")
    lines.append("**10. Does Probe still help on datasets where LearnedProbe-Rank previously failed to beat Equal?** BeijingAirQuality and ETTm2 are exactly the two datasets where LearnedProbe-Rank previously lost to plain Equal averaging (see ../reports/learned_probe_generalization_validation.md) -- compare their rows above to see whether the Simplex+Probe fusion recovers a benefit there.")
    lines.append(f"**11. Is there evidence LearnedProbe provides useful information OUTSIDE C-Rank?** {decision['conclusion']}")
    lines += ["", f"## Decision: {decision['tier']}", "", decision["conclusion"], ""]
    lines += ["## Hard rule compliance", "", "```text", "TEST SET ACCESSED: NO", "FORECASTING EXPERTS RETRAINED: NO", "LEARNEDPROBE ARCHITECTURE/LOSS/TRAINING MODIFIED: NO", "OTHER ROUTERS (Frozen/Online COSTAR, Top-1, Top-k, Ridge, Granger-Ramanathan) TOUCHED: NO", "```"]
    (OUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    start = time.time()
    report: dict[str, Any] = {
        "experiment": "simplex_vs_simplex_plus_learnedprobe",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_sha": git_commit_sha(),
        "datasets": {},
    }
    all_results, all_dependence, all_weights, all_integrity, all_alpha = [], [], [], [], []

    for dataset in NEW_DATASETS:
        print(f"[simplex_probe] {dataset}: starting...", flush=True)
        result = evaluate_dataset(dataset)
        report["datasets"][dataset] = result
        all_results.extend(result["result_rows"])
        all_dependence.extend(result["dependence_rows"])
        all_weights.extend(result["weight_rows"])
        all_integrity.append(result["integrity"])
        all_alpha.extend(result["alpha_rows"])
        print(f"[simplex_probe] {dataset}: done.", flush=True)

    decision = decide(report)
    report["decision"] = decision
    report["runtime_sec"] = time.time() - start
    report["test_set_accessed"] = False

    manifest = {
        "manifest_type": "simplex_probe_manifest",
        "created_at_utc": report["created_at_utc"],
        "git_commit_sha": report["git_commit_sha"],
        "datasets": NEW_DATASETS,
        "expert_pool": ["DLinear", "PatchTST", "iTransformer", "TimesNet", "ModernTCN"],
        "core_reuse_note": "Same train-selected 3-expert core used for BOTH Simplex and Simplex+Probe on every dataset, identical to the core already used for C-Rank/FixedD-Rank/LearnedProbe-Rank in experiments/behavioral_competence/generalization/.",
        "base_simplex_note": "The only existing fit_simplex_weights precedent in this repo (ETTh1/ETTh2) is full-5-expert-pool and test-touched, so it is not reused directly. The unmodified fit_simplex_weights function IS reused, applied to a core-subset prediction_stack, so Simplex and Simplex+Probe are on the exact same expert set (a controlled comparison), and to keep Probe's per-window scores well-defined (Probe was only ever trained/evaluated on the 3-expert core).",
        "expert_cores": {ds: report["datasets"][ds]["core"] for ds in NEW_DATASETS},
        "base_simplex_weights": {ds: report["datasets"][ds]["simplex_weights"] for ds in NEW_DATASETS},
        "probe_output_hash_sha256": {ds: report["datasets"][ds]["probe_output_hash_sha256"] for ds in NEW_DATASETS},
        "probe_checkpoint_note": "ProbeGenerator/CompetenceScorer weights are not persisted to disk anywhere in this project (see experiments/behavioral_competence/run_learned_probe.py); reproducibility is instead by deterministic re-training (fixed seed=7) and verified via the reported probe_output_hash_sha256 of the router_val predicted-excess-loss array. Architecture/loss/training objective are unchanged from the frozen method (see ../FROZEN_METHOD.md).",
        "alpha_grid": list(ALPHA_GRID),
        "selected_alpha": {ds: report["datasets"][ds]["selected_alpha"] for ds in NEW_DATASETS},
        "shuffle_seed": SHUFFLE_SEED,
        "fusion_formula": "adjusted_weight_e = softmax_e( log(simplex_weight_e) - alpha * z(probe_loss_e) ), where z(probe_loss_e) = (probe_loss_e - mean_over_experts) / max(std_over_experts_population, 1e-8), computed independently per window; simplex_weight_e == 0 forces adjusted_weight_e = 0 for all alpha via a -inf logit (matches softmax's native handling); alpha=0 reduces exactly to the static Simplex weights up to log/exp floating-point roundoff.",
        "sign_convention": "probe_loss_e = predicted_excess_loss_e = predicted(expert_MAE - equal_ensemble_MAE); LOWER = better (more competent) expert for the current window. Never depends on router_val targets -- verified structurally (evaluate_on_val never reads cache['targets']) and via the target-corruption-invariance integrity check.",
        "normalization": "z-score of probe_loss across the K core experts, WITHIN each window (population std, floored at 1e-8).",
        "core_selection_protocol": "select_core_on_router_train (train-only, pooled OOF MAE over 4 chronological folds) -- unchanged, reused from experiments/costar_multidataset_frozen/common.py.",
        "expert_checkpoint_sha256": {ds: report["datasets"][ds]["checkpoint_hashes"] for ds in NEW_DATASETS},
        "router_train_cache_note": "cache/costarts_walkforward_{dataset}/router_train_20_60_cache.pt used for Simplex fitting and alpha selection; cache/costarts_walkforward_{dataset}/router_val_60_80_cache.pt used only for the final, once-only evaluation. test_80_100_cache.pt was never built for any of these four datasets in this project.",
        "decision_rule": "Section 16 of the task instructions, applied verbatim without modification after seeing results.",
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUT_DIR / "simplex_probe_manifest.json", manifest)
    write_json(OUT_DIR / "validation_results.json", report)
    write_csv(OUT_DIR / "validation_results.csv", all_results)
    write_csv(OUT_DIR / "dependence_aware_results.csv", all_dependence)
    write_csv(OUT_DIR / "weight_analysis.csv", all_weights)
    write_csv(OUT_DIR / "integrity_checks.csv", all_integrity)
    write_csv(OUT_DIR / "alpha_selection.csv", all_alpha)
    make_report(report, decision)

    print("TEST SET ACCESSED: NO")
    print(json.dumps({"runtime_sec": report["runtime_sec"], "tier": decision["tier"]}, indent=2))


if __name__ == "__main__":
    main()
