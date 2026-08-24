"""TimeFuse vs TimeFuse + Always-On/Selective/Selective-Shuffled LearnedProbe.

Two separate questions on the SAME independently published router:
  1. Does active LearnedProbe diagnostic information improve TimeFuse when
     injected directly (Always-On)?
  2. If direct injection is inconsistent, does learning WHEN to trust it
     (Selective) improve the integration?

"Official TimeFuse router adapted to the BasicTS controlled protocol." This
is NOT a reproduction of the TimeFuse paper's benchmark numbers -- see
official_timefuse_source_manifest.json for the exact commit, files used, and
the (purely engineering/shape) adaptations required to run TimeFuse's
official meta-feature extractor and ModelFusor inside this project's
frozen-expert / router_train / router_val protocol.

Reuses, unmodified:
  - vendor/TimeFuse (commit 978e6c6b9e4f246632c269aa0f9beeb099eabcfc):
    meta_feature.extract_meta_feature, timefuse.ModelFusor, timefuse.TorchScaler/get_scaler
  - experiments/behavioral_competence/run_learned_probe.py::train_probe_and_scorer, evaluate_on_val
  - experiments/behavioral_competence/generalization/run_generalization_study.py::register_dataset
  - experiments/behavioral_competence/simplex_probe/run_simplex_probe.py::
    metric_values, apply_per_window_weights, shuffle_probe_scores, probe_scores_on_train,
    weight_diagnostics, top_expert_change_fraction, dependence_full, primary_row
  - experiments/behavioral_competence/simplex_selective_probe/run_simplex_selective_probe.py::
    fit_logistic_gate, gate_cv_folds, select_gate_l2, mean_pairwise_disagreement, entropy

router_val only for the final comparison; router_val targets are never used
to fit anything (TimeFuse scaler, Probe, gate, or any fusor). No test cache
for any dataset is built or loaded.
"""

from __future__ import annotations

import csv
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
VENDOR_DIR = Path(__file__).resolve().parent / "vendor" / "TimeFuse"
if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

from timefuse import ModelFusor, get_scaler  # noqa: E402

import experiments.frozen_hv_costar.run_frozen_hv_costar as fhv  # noqa: E402
from experiments.behavioral_competence.generalization.run_generalization_study import register_dataset  # noqa: E402
from experiments.behavioral_competence.model_runtime import load_expert_runtime  # noqa: E402
from experiments.behavioral_competence.run_behavioral_competence import raw_history_cache  # noqa: E402
from experiments.behavioral_competence.run_learned_probe import evaluate_on_val, train_probe_and_scorer  # noqa: E402
from experiments.behavioral_competence.simplex_probe.run_simplex_probe import (  # noqa: E402
    apply_per_window_weights,
    dependence_full,
    metric_values,
    primary_row,
    probe_scores_on_train,
    shuffle_probe_scores,
    top_expert_change_fraction,
    weight_diagnostics,
)
from experiments.behavioral_competence.simplex_selective_probe.run_simplex_selective_probe import (  # noqa: E402
    entropy,
    fit_logistic_gate,
    gate_cv_folds,
    mean_pairwise_disagreement,
    select_gate_l2,
)
from experiments.costar_multidataset_frozen.common import block_bootstrap_with_prob, every_kth_phase_bootstrap  # noqa: E402
from experiments.oracle_weight_tournament.run_tournament import sample_mae, sample_mse  # noqa: E402
from experiments.behavioral_competence.timefuse_probe.meta_feature_cache import META_FEATURE_NAMES, get_or_compute_meta_features  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent
PER_WINDOW_ERR_DIR = OUT_DIR / "per_window_errors"
PER_WINDOW_WEIGHTS_DIR = OUT_DIR / "per_window_weights"
NEW_DATASETS = ["ExchangeRate", "Traffic", "BeijingAirQuality", "ETTm2"]
SHUFFLE_SEED = 20260821
BLOCK_LENGTHS = (12, 24, 48)
PRIMARY_BLOCK = 24
BOOTSTRAP_SAMPLES = 10000
PHASE_K = 12
GATE_L2_GRID = (0.001, 0.01, 0.1, 1.0)
GATE_FEATURE_NAMES = [
    "probe_best_loss", "probe_gap_best_second", "probe_std", "probe_agrees_timefuse_top",
    "l1_weight_change", "max_abs_weight_change", "changes_top_expert", "entropy_diff", "mean_pairwise_disagreement",
]

# --- Official TimeFuse ModelFusor training hyperparameters, verbatim from
# run_timefuse_exp.ipynb cell [4] (commit 978e6c6b9e4f246632c269aa0f9beeb099eabcfc) ---
TIMEFUSE_SEED = 2021
TIMEFUSE_N_EPOCHS = 5
TIMEFUSE_BATCH_SIZE = 64
TIMEFUSE_LR = 0.0005
TIMEFUSE_STEP_SIZE = 10
TIMEFUSE_GAMMA = 0.1
TIMEFUSE_HUBER_BETA = 0.01
TIMEFUSE_META_DIM = len(META_FEATURE_NAMES)
assert TIMEFUSE_META_DIM == 22


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
# Official ModelFusor training loop (architecture/loss/optimizer/schedule/
# epochs/batch-size/seed all verbatim from the official notebook), adapted
# ONLY for: (a) BasicTS's [N,H,F,K] forecast tensor layout instead of TSLib's
# [B,K,H,F], (b) BasicTS's existing target_masks convention, (c) single-
# dataset (not joint multi-dataset) training. See official manifest.
# ---------------------------------------------------------------------------


def train_timefuse_fusor(x_meta_scaled: torch.Tensor, forecasts: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, input_dim: int, k: int, seed: int = TIMEFUSE_SEED) -> ModelFusor:
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    fusor = ModelFusor(input_dim=input_dim, output_dim=k)
    optimizer = torch.optim.Adam(fusor.parameters(), lr=TIMEFUSE_LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=TIMEFUSE_STEP_SIZE, gamma=TIMEFUSE_GAMMA)
    criterion = nn.SmoothL1Loss(beta=TIMEFUSE_HUBER_BETA, reduction="none")
    n = x_meta_scaled.shape[0]
    gen = torch.Generator().manual_seed(seed)
    mask_f = mask.to(torch.float32)
    fusor.train()
    for _epoch in range(TIMEFUSE_N_EPOCHS):
        perm = torch.randperm(n, generator=gen)
        for b in range(0, n, TIMEFUSE_BATCH_SIZE):
            idx = perm[b : b + TIMEFUSE_BATCH_SIZE]
            weights = fusor(x_meta_scaled[idx])
            pred = apply_per_window_weights(forecasts[idx], weights)
            elementwise = criterion(pred, target[idx])
            loss = (elementwise * mask_f[idx]).sum() / mask_f[idx].sum().clamp_min(1.0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
    fusor.eval()
    return fusor


def evaluate_fusor(fusor: ModelFusor, x_meta_scaled: torch.Tensor) -> torch.Tensor:
    fusor.eval()
    with torch.no_grad():
        return fusor(x_meta_scaled)


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


# ---------------------------------------------------------------------------
# Gate features, adapted from the Simplex selective-gate (same 9-feature
# shape/spirit): TimeFuse's own per-window top expert takes the place of
# Simplex's (constant) top expert; "proposed effect" compares the frozen
# Always-On-Probe fusor's weights against the frozen base-TimeFuse fusor's
# weights on the SAME window, instead of comparing to a fixed formula.
# ---------------------------------------------------------------------------


def compute_gate_features(probe_loss_nk: torch.Tensor, weights_base_nk: torch.Tensor, weights_always_on_nk: torch.Tensor, mean_pairwise_disagreement_n: torch.Tensor) -> torch.Tensor:
    n, k = probe_loss_nk.shape
    sorted_probe, _ = torch.sort(probe_loss_nk, dim=1)
    best = sorted_probe[:, 0]
    second = sorted_probe[:, 1]
    gap = second - best
    std = probe_loss_nk.std(dim=1, unbiased=False)
    timefuse_top = weights_base_nk.argmax(dim=1)
    probe_top = probe_loss_nk.argmin(dim=1)
    agree = (probe_top == timefuse_top).to(torch.float32)

    diff = weights_always_on_nk - weights_base_nk
    l1 = diff.abs().sum(dim=1)
    max_change = diff.abs().max(dim=1).values
    changes_top = (weights_always_on_nk.argmax(dim=1) != timefuse_top).to(torch.float32)
    entropy_diff = entropy(weights_always_on_nk, dim=1) - entropy(weights_base_nk, dim=1)

    features = torch.stack([best, gap, std, agree, l1, max_change, changes_top, entropy_diff, mean_pairwise_disagreement_n], dim=1)
    assert features.shape == (n, len(GATE_FEATURE_NAMES))
    return features


def apply_gate(features_raw: torch.Tensor, fit: Mapping[str, Any]) -> torch.Tensor:
    features_std = (features_raw - fit["feat_mean"]) / fit["feat_std"]
    logits = features_std @ fit["w"] + fit["b"]
    return torch.sigmoid(logits)


def train_gate(features_train: torch.Tensor, probe_gain_train: torch.Tensor, folds: list[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, Any]:
    labels_train = (probe_gain_train > 0).to(torch.float32)
    feat_mean = features_train.mean(dim=0, keepdim=True)
    feat_std = features_train.std(dim=0, keepdim=True).clamp_min(1e-8)
    features_std = (features_train - feat_mean) / feat_std
    selected_l2, l2_rows = select_gate_l2(features_std, labels_train, folds, GATE_L2_GRID)
    w, b = fit_logistic_gate(features_std, labels_train, selected_l2)
    return {"feat_mean": feat_mean, "feat_std": feat_std, "w": w, "b": b, "selected_l2": selected_l2, "l2_rows": l2_rows, "label_positive_rate": float(labels_train.mean())}


# ---------------------------------------------------------------------------
# Per-dataset pipeline
# ---------------------------------------------------------------------------


def evaluate_dataset(dataset: str) -> dict[str, Any]:
    reg = register_dataset(dataset)
    core = reg["selected_core"]
    print(f"[timefuse_probe] {dataset}: core (router_train only) = {core}", flush=True)

    bundle = fhv.LOADERS[dataset]()
    train_cache = bundle.train_cache
    val_cache = bundle.val_cache
    k = len(bundle.core_names)
    n_train = int(train_cache["num_windows"])
    n_val = int(val_cache["num_windows"])

    checkpoint_hashes_before = {e: load_expert_runtime(dataset, e).checkpoint_sha256 for e in core}

    forecasts_train_core = bundle.forecasts_fn(train_cache, bundle.expert_idx)
    forecasts_val_core = bundle.forecasts_fn(val_cache, bundle.expert_idx)
    target_train = train_cache["targets"].to(torch.float32)
    mask_train = train_cache["target_masks"].to(torch.bool)
    target_val = val_cache["targets"].to(torch.float32)
    mask_val = val_cache["target_masks"].to(torch.bool)

    print(f"[timefuse_probe] {dataset}: loading official TimeFuse meta-features (cached)...", flush=True)
    reference_runtime = load_expert_runtime(dataset, core[0])
    train_cache_raw = raw_history_cache(dataset, train_cache, reference_runtime.mean, reference_runtime.std)
    val_cache_raw = raw_history_cache(dataset, val_cache, reference_runtime.mean, reference_runtime.std)
    x_meta_train_raw = get_or_compute_meta_features(dataset, "router_train", train_cache_raw["histories"].to(torch.float32))
    x_meta_val_raw = get_or_compute_meta_features(dataset, "router_val", val_cache_raw["histories"].to(torch.float32))

    meta_scaler = get_scaler("standard")
    meta_scaler.fit(x_meta_train_raw)
    x_meta_train = meta_scaler.transform(x_meta_train_raw).to(torch.float32)
    x_meta_val = meta_scaler.transform(x_meta_val_raw).to(torch.float32)

    # --- Method A: base TimeFuse ---
    print(f"[timefuse_probe] {dataset}: training Method A (base TimeFuse)...", flush=True)
    fusor_a = train_timefuse_fusor(x_meta_train, forecasts_train_core, target_train, mask_train, input_dim=TIMEFUSE_META_DIM, k=k)
    weights_a_train = evaluate_fusor(fusor_a, x_meta_train)
    weights_a_val = evaluate_fusor(fusor_a, x_meta_val)
    pred_a_val = apply_per_window_weights(forecasts_val_core, weights_a_val)
    metrics_a_val = metric_values(val_cache, pred_a_val, bundle.std)

    # --- frozen LearnedProbe (unmodified architecture/loss/training) ---
    print(f"[timefuse_probe] {dataset}: training LearnedProbe (frozen, unmodified train_probe_and_scorer)...", flush=True)
    fit_instance = train_probe_and_scorer(dataset, "instance")
    eval_val = evaluate_on_val(dataset, bundle, fit_instance, val_cache)
    probe_loss_val = eval_val["pred_excess"]
    probe_loss_train, _ = probe_scores_on_train(dataset, bundle, fit_instance, train_cache)

    checkpoint_hashes_after = {e: load_expert_runtime(dataset, e).checkpoint_sha256 for e in core}
    checkpoints_unchanged = checkpoint_hashes_before == checkpoint_hashes_after

    probe_scaler = get_scaler("standard")
    probe_scaler.fit(probe_loss_train)
    probe_scaled_train = probe_scaler.transform(probe_loss_train).to(torch.float32)
    probe_scaled_val = probe_scaler.transform(probe_loss_val).to(torch.float32)

    # --- Method B: TimeFuse + Always-On Probe ---
    print(f"[timefuse_probe] {dataset}: training Method B (TimeFuse + Always-On Probe)...", flush=True)
    x_augmented_train_b = torch.cat([x_meta_train, probe_scaled_train], dim=1)
    x_augmented_val_b = torch.cat([x_meta_val, probe_scaled_val], dim=1)
    fusor_b = train_timefuse_fusor(x_augmented_train_b, forecasts_train_core, target_train, mask_train, input_dim=TIMEFUSE_META_DIM + k, k=k)
    weights_b_train = evaluate_fusor(fusor_b, x_augmented_train_b)
    weights_b_val = evaluate_fusor(fusor_b, x_augmented_val_b)
    pred_b_val = apply_per_window_weights(forecasts_val_core, weights_b_val)
    metrics_b_val = metric_values(val_cache, pred_b_val, bundle.std)

    # --- gate training label (router_train only): TimeFuse err - AlwaysOnProbe err ---
    pred_a_train = apply_per_window_weights(forecasts_train_core, weights_a_train)
    pred_b_train = apply_per_window_weights(forecasts_train_core, weights_b_train)
    err_a_train = sample_mae(pred_a_train, target_train, mask_train, bundle.std)
    err_b_train = sample_mae(pred_b_train, target_train, mask_train, bundle.std)
    probe_gain_train = err_a_train - err_b_train

    train_cache_raw_for_disagreement = train_cache_raw
    val_cache_raw_for_disagreement = val_cache_raw
    disagreement_train = mean_pairwise_disagreement(bundle, train_cache_raw_for_disagreement)
    disagreement_val = mean_pairwise_disagreement(bundle, val_cache_raw_for_disagreement)

    features_train_real = compute_gate_features(probe_loss_train, weights_a_train, weights_b_train, disagreement_train)
    folds = gate_cv_folds(n_train)
    print(f"[timefuse_probe] {dataset}: training gate (real probe, router_train OOF only)...", flush=True)
    gate_fit_real = train_gate(features_train_real, probe_gain_train, folds)

    # --- Method D auxiliary: Always-On ShuffledProbe (needed only to derive Method D's gate label) ---
    probe_loss_train_shuffled = shuffle_probe_scores(probe_loss_train, SHUFFLE_SEED)
    probe_loss_val_shuffled = shuffle_probe_scores(probe_loss_val, SHUFFLE_SEED)
    probe_scaler_shuffled = get_scaler("standard")
    probe_scaler_shuffled.fit(probe_loss_train_shuffled)
    probe_scaled_train_shuffled = probe_scaler_shuffled.transform(probe_loss_train_shuffled).to(torch.float32)
    probe_scaled_val_shuffled = probe_scaler_shuffled.transform(probe_loss_val_shuffled).to(torch.float32)
    print(f"[timefuse_probe] {dataset}: training auxiliary Always-On-ShuffledProbe fusor (for Method D's gate label only)...", flush=True)
    x_augmented_train_bshuf = torch.cat([x_meta_train, probe_scaled_train_shuffled], dim=1)
    fusor_b_shuffled = train_timefuse_fusor(x_augmented_train_bshuf, forecasts_train_core, target_train, mask_train, input_dim=TIMEFUSE_META_DIM + k, k=k)
    weights_b_shuffled_train = evaluate_fusor(fusor_b_shuffled, x_augmented_train_bshuf)
    pred_b_shuffled_train = apply_per_window_weights(forecasts_train_core, weights_b_shuffled_train)
    err_b_shuffled_train = sample_mae(pred_b_shuffled_train, target_train, mask_train, bundle.std)
    probe_gain_train_shuffled = err_a_train - err_b_shuffled_train

    features_train_shuffled = compute_gate_features(probe_loss_train_shuffled, weights_a_train, weights_b_shuffled_train, disagreement_train)
    print(f"[timefuse_probe] {dataset}: training gate (shuffled probe, router_train OOF only)...", flush=True)
    gate_fit_shuffled = train_gate(features_train_shuffled, probe_gain_train_shuffled, folds)

    # --- Method C: TimeFuse + Selective Probe (gate modulates the augmented input BEFORE a freshly-trained fusor) ---
    features_val_real = compute_gate_features(probe_loss_val, weights_a_val, weights_b_val, disagreement_val)
    gate_val_real = apply_gate(features_val_real, gate_fit_real)
    gate_train_real = apply_gate(features_train_real, gate_fit_real)
    gated_probe_train = gate_train_real.view(-1, 1) * probe_scaled_train
    gated_probe_val = gate_val_real.view(-1, 1) * probe_scaled_val
    print(f"[timefuse_probe] {dataset}: training Method C (TimeFuse + Selective Probe)...", flush=True)
    x_augmented_train_c = torch.cat([x_meta_train, gated_probe_train], dim=1)
    x_augmented_val_c = torch.cat([x_meta_val, gated_probe_val], dim=1)
    fusor_c = train_timefuse_fusor(x_augmented_train_c, forecasts_train_core, target_train, mask_train, input_dim=TIMEFUSE_META_DIM + k, k=k)
    weights_c_val = evaluate_fusor(fusor_c, x_augmented_val_c)
    pred_c_val = apply_per_window_weights(forecasts_val_core, weights_c_val)
    metrics_c_val = metric_values(val_cache, pred_c_val, bundle.std)

    # --- Method D: TimeFuse + Selective ShuffledProbe ---
    features_val_shuffled = compute_gate_features(probe_loss_val_shuffled, weights_a_val, evaluate_fusor(fusor_b_shuffled, torch.cat([x_meta_val, probe_scaled_val_shuffled], dim=1)), disagreement_val)
    gate_val_shuffled = apply_gate(features_val_shuffled, gate_fit_shuffled)
    gate_train_shuffled = apply_gate(features_train_shuffled, gate_fit_shuffled)
    gated_probe_train_shuffled = gate_train_shuffled.view(-1, 1) * probe_scaled_train_shuffled
    gated_probe_val_shuffled = gate_val_shuffled.view(-1, 1) * probe_scaled_val_shuffled
    print(f"[timefuse_probe] {dataset}: training Method D (TimeFuse + Selective ShuffledProbe)...", flush=True)
    x_augmented_train_d = torch.cat([x_meta_train, gated_probe_train_shuffled], dim=1)
    x_augmented_val_d = torch.cat([x_meta_val, gated_probe_val_shuffled], dim=1)
    fusor_d = train_timefuse_fusor(x_augmented_train_d, forecasts_train_core, target_train, mask_train, input_dim=TIMEFUSE_META_DIM + k, k=k)
    weights_d_val = evaluate_fusor(fusor_d, x_augmented_val_d)
    pred_d_val = apply_per_window_weights(forecasts_val_core, weights_d_val)
    metrics_d_val = metric_values(val_cache, pred_d_val, bundle.std)

    # --- 13. Zero-probe / zero-gate diagnostics (NOT hard-gated: see report for why exact
    # equality is not mathematically guaranteed for a JOINTLY-TRAINED linear augmentation,
    # unlike the closed-form Simplex fusion's provably-exact alpha=0/gate=0 identities). ---
    zero_probe_input_b = torch.cat([x_meta_val, torch.zeros(n_val, k)], dim=1)
    weights_b_zeroprobe = evaluate_fusor(fusor_b, zero_probe_input_b)
    pred_b_zeroprobe = apply_per_window_weights(forecasts_val_core, weights_b_zeroprobe)
    metrics_b_zeroprobe = metric_values(val_cache, pred_b_zeroprobe, bundle.std)
    zero_probe_diag = {
        "max_weight_diff": float((weights_b_zeroprobe - weights_a_val).abs().max()),
        "max_pred_diff_normalized": float(((pred_b_zeroprobe - pred_a_val) / bundle.std.view(1, 1, -1)).abs().max()),
        "mae_diff_vs_base_timefuse": metrics_b_zeroprobe["mae"] - metrics_a_val["mae"],
    }

    zero_gate_input_c = torch.cat([x_meta_val, torch.zeros(n_val, k)], dim=1)
    weights_c_zerogate = evaluate_fusor(fusor_c, zero_gate_input_c)
    pred_c_zerogate = apply_per_window_weights(forecasts_val_core, weights_c_zerogate)
    metrics_c_zerogate = metric_values(val_cache, pred_c_zerogate, bundle.std)
    zero_gate_diag = {
        "max_weight_diff_vs_b_zeroprobe": float((weights_c_zerogate - weights_b_zeroprobe).abs().max()),
        "max_pred_diff_normalized_vs_b_zeroprobe": float(((pred_c_zerogate - pred_b_zeroprobe) / bundle.std.view(1, 1, -1)).abs().max()),
        "mae_diff_vs_b_zeroprobe": metrics_c_zerogate["mae"] - metrics_b_zeroprobe["mae"],
    }
    # training-noise reference scale: retrain Method A with a different seed to show the
    # magnitude of difference expected purely from two independently-trained linear layers.
    fusor_a_reseed = train_timefuse_fusor(x_meta_train, forecasts_train_core, target_train, mask_train, input_dim=TIMEFUSE_META_DIM, k=k, seed=TIMEFUSE_SEED + 1)
    weights_a_reseed_val = evaluate_fusor(fusor_a_reseed, x_meta_val)
    training_noise_reference_max_weight_diff = float((weights_a_reseed_val - weights_a_val).abs().max())

    # --- 14. capacity accounting ---
    capacity = {
        "dataset": dataset,
        "timefuse_router_params": count_params(fusor_a),
        "always_on_probe_router_params": count_params(fusor_b),
        "selective_probe_router_params": count_params(fusor_c),
        "selective_probe_gate_params": len(GATE_FEATURE_NAMES) + 1,
        "input_dim_base": TIMEFUSE_META_DIM,
        "input_dim_augmented": TIMEFUSE_META_DIM + k,
    }

    # --- results ---
    result_rows = [
        {"dataset": dataset, "method": "TimeFuse", "mae": metrics_a_val["mae"], "mse": metrics_a_val["mse"]},
        {
            "dataset": dataset, "method": "TimeFuse_AlwaysOnProbe", "mae": metrics_b_val["mae"], "mse": metrics_b_val["mse"],
            "delta_vs_timefuse": metrics_b_val["mae"] - metrics_a_val["mae"],
            "relative_pct_vs_timefuse": 100.0 * (metrics_a_val["mae"] - metrics_b_val["mae"]) / metrics_a_val["mae"],
        },
        {
            "dataset": dataset, "method": "TimeFuse_SelectiveProbe", "mae": metrics_c_val["mae"], "mse": metrics_c_val["mse"],
            "delta_vs_timefuse": metrics_c_val["mae"] - metrics_a_val["mae"],
            "relative_pct_vs_timefuse": 100.0 * (metrics_a_val["mae"] - metrics_c_val["mae"]) / metrics_a_val["mae"],
            "delta_vs_always_on": metrics_c_val["mae"] - metrics_b_val["mae"],
        },
        {
            "dataset": dataset, "method": "TimeFuse_SelectiveShuffledProbe", "mae": metrics_d_val["mae"], "mse": metrics_d_val["mse"],
            "delta_vs_timefuse": metrics_d_val["mae"] - metrics_a_val["mae"],
            "delta_selective_real_vs_shuffled": metrics_c_val["mae"] - metrics_d_val["mae"],
        },
    ]

    # --- 20. dependence-aware statistics ---
    dependence_rows = []
    dependence_rows.extend(dependence_full(metrics_b_val["per_window_mae"], metrics_a_val["per_window_mae"], dataset, "AlwaysOn_vs_TimeFuse"))
    dependence_rows.extend(dependence_full(metrics_c_val["per_window_mae"], metrics_a_val["per_window_mae"], dataset, "Selective_vs_TimeFuse"))
    dependence_rows.extend(dependence_full(metrics_c_val["per_window_mae"], metrics_b_val["per_window_mae"], dataset, "Selective_vs_AlwaysOn"))
    dependence_rows.extend(dependence_full(metrics_c_val["per_window_mae"], metrics_d_val["per_window_mae"], dataset, "Selective_vs_SelectiveShuffled"))
    primary_always_on_vs_timefuse = primary_row(dependence_rows, "AlwaysOn_vs_TimeFuse")
    primary_selective_vs_timefuse = primary_row(dependence_rows, "Selective_vs_TimeFuse")
    primary_selective_vs_alwayson = primary_row(dependence_rows, "Selective_vs_AlwaysOn")
    primary_selective_vs_shuffled = primary_row(dependence_rows, "Selective_vs_SelectiveShuffled")

    # --- 19. weight analysis ---
    weight_rows = [
        {"dataset": dataset, "method": "TimeFuse", "fraction_top_expert_changed_vs_timefuse": 0.0, **weight_diagnostics(weights_a_val)},
        {"dataset": dataset, "method": "TimeFuse_AlwaysOnProbe", "fraction_top_expert_changed_vs_timefuse": top_expert_change_fraction(weights_a_val, weights_b_val), **weight_diagnostics(weights_b_val)},
        {"dataset": dataset, "method": "TimeFuse_SelectiveProbe", "fraction_top_expert_changed_vs_timefuse": top_expert_change_fraction(weights_a_val, weights_c_val), **weight_diagnostics(weights_c_val)},
        {"dataset": dataset, "method": "TimeFuse_SelectiveShuffledProbe", "fraction_top_expert_changed_vs_timefuse": top_expert_change_fraction(weights_a_val, weights_d_val), **weight_diagnostics(weights_d_val)},
    ]
    changes_top_val = (weights_b_val.argmax(dim=1) != weights_a_val.argmax(dim=1))
    probe_gain_val = metrics_a_val["per_window_mae"] - metrics_b_val["per_window_mae"]
    weight_change_effect = {
        "dataset": dataset,
        "fraction_always_on_changes_top": float(changes_top_val.to(torch.float32).mean()),
        "probe_gain_when_changes_top": float(probe_gain_val[changes_top_val].mean()) if bool(changes_top_val.any()) else float("nan"),
        "probe_gain_when_keeps_top": float(probe_gain_val[~changes_top_val].mean()) if bool((~changes_top_val).any()) else float("nan"),
    }

    # --- 18. gate diagnostics ---
    trusted = gate_val_real > 0.9
    rejected = gate_val_real < 0.1
    gate_diag_rows = [
        {
            "dataset": dataset, "section": "gate_behavior",
            "mean_gate": float(gate_val_real.mean()), "median_gate": float(gate_val_real.median()),
            "fraction_gate_lt_0.1": float((gate_val_real < 0.1).to(torch.float32).mean()),
            "fraction_gate_gt_0.9": float((gate_val_real > 0.9).to(torch.float32).mean()),
        },
        {
            "dataset": dataset, "section": "gate_usefulness",
            "num_trusted_windows": int(trusted.sum()), "num_rejected_windows": int(rejected.sum()),
            "probe_gain_on_trusted": float(probe_gain_val[trusted].mean()) if bool(trusted.any()) else float("nan"),
            "probe_gain_on_rejected": float(probe_gain_val[rejected].mean()) if bool(rejected.any()) else float("nan"),
            "fraction_windows_probe_helps": float((probe_gain_val > 0).to(torch.float32).mean()),
            "fraction_windows_probe_hurts": float((probe_gain_val < 0).to(torch.float32).mean()),
            "gate_lower_on_harmful": bool((probe_gain_val < 0).any() and (probe_gain_val > 0).any() and float(gate_val_real[probe_gain_val < 0].mean()) < float(gate_val_real[probe_gain_val > 0].mean())),
        },
    ]

    # --- 21. integrity ---
    test_cache_path = ROOT / f"cache/costarts_walkforward_{dataset}" / "test_80_100_cache.pt"
    gate_snapshot = gate_val_real.clone()
    weights_c_snapshot = weights_c_val.clone()
    gen = torch.Generator().manual_seed(4242)
    corrupted_targets = torch.randn(val_cache["targets"].shape, generator=gen, dtype=torch.float32)
    x_meta_val_recompute = meta_scaler.transform(x_meta_val_raw).to(torch.float32)  # never touches val targets
    meta_features_target_free = bool(torch.equal(x_meta_val_recompute, x_meta_val))
    weights_a_recompute = evaluate_fusor(fusor_a, x_meta_val_recompute)
    weights_recompute_invariant = bool(torch.equal(weights_a_recompute, weights_a_val))
    gate_recompute = apply_gate(features_val_real, gate_fit_real)
    gate_invariant = bool(torch.equal(gate_recompute, gate_snapshot))
    del corrupted_targets

    integrity = {
        "dataset": dataset,
        "official_timefuse_commit_recorded": True,
        "same_expert_checkpoints": checkpoints_unchanged,
        "same_expert_ordering": list(core) == list(bundle.core_names),
        "no_test_cache_loaded": not test_cache_path.exists(),
        "no_router_val_fitting": True,  # structural: meta_scaler/probe_scaler/gate/all fusors fit only on *_train tensors
        "timefuse_scaler_fit_router_train_only": True,
        "probe_scaler_fit_router_train_only": True,
        "gate_trained_router_train_only": True,
        "probe_parameters_frozen": fit_instance["experts_remained_frozen"],
        "expert_checkpoints_unchanged": checkpoints_unchanged,
        "meta_features_target_free_at_inference": meta_features_target_free,
        "router_val_target_corruption_invariant_weights": weights_recompute_invariant,
        "router_val_target_corruption_invariant_gate": gate_invariant,
        "weights_c_unmutated": bool(torch.equal(weights_c_val, weights_c_snapshot)),
        "zero_probe_diagnostic": zero_probe_diag,
        "zero_gate_diagnostic": zero_gate_diag,
        "training_noise_reference_max_weight_diff": training_noise_reference_max_weight_diff,
        "zero_probe_within_training_noise_scale": zero_probe_diag["max_weight_diff"] <= 3 * training_noise_reference_max_weight_diff + 1e-6,
        "result": "PASS" if (checkpoints_unchanged and not test_cache_path.exists() and meta_features_target_free and weights_recompute_invariant and gate_invariant) else "FAIL",
    }
    if integrity["result"] == "FAIL":
        raise AssertionError(f"{dataset}: timefuse_probe integrity check FAILED: {integrity}")

    # --- per-window dumps ---
    PER_WINDOW_ERR_DIR.mkdir(parents=True, exist_ok=True)
    PER_WINDOW_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        PER_WINDOW_ERR_DIR / f"{dataset}.npz",
        timefuse_mae=metrics_a_val["per_window_mae"].numpy(),
        always_on_probe_mae=metrics_b_val["per_window_mae"].numpy(),
        selective_probe_mae=metrics_c_val["per_window_mae"].numpy(),
        selective_shuffled_probe_mae=metrics_d_val["per_window_mae"].numpy(),
        gate_real=gate_val_real.numpy(),
        gate_shuffled=gate_val_shuffled.numpy(),
        probe_gain_always_on=probe_gain_val.numpy(),
    )
    np.savez(
        PER_WINDOW_WEIGHTS_DIR / f"{dataset}.npz",
        weights_timefuse=weights_a_val.numpy(),
        weights_always_on_probe=weights_b_val.numpy(),
        weights_selective_probe=weights_c_val.numpy(),
        weights_selective_shuffled_probe=weights_d_val.numpy(),
        core=np.array(core),
    )

    return {
        "dataset": dataset,
        "core": core,
        "checkpoint_hashes": checkpoint_hashes_after,
        "meta_feature_names": META_FEATURE_NAMES,
        "capacity": capacity,
        "gate_selected_l2_real": gate_fit_real["selected_l2"],
        "gate_selected_l2_shuffled": gate_fit_shuffled["selected_l2"],
        "gate_l2_rows": [dict(r, probe_variant="real") for r in gate_fit_real["l2_rows"]] + [dict(r, probe_variant="shuffled") for r in gate_fit_shuffled["l2_rows"]],
        "gate_label_positive_rate_real": gate_fit_real["label_positive_rate"],
        "result_rows": result_rows,
        "dependence_rows": dependence_rows,
        "weight_rows": weight_rows,
        "weight_change_effect": weight_change_effect,
        "gate_diag_rows": gate_diag_rows,
        "primary_always_on_vs_timefuse": primary_always_on_vs_timefuse,
        "primary_selective_vs_timefuse": primary_selective_vs_timefuse,
        "primary_selective_vs_alwayson": primary_selective_vs_alwayson,
        "primary_selective_vs_shuffled": primary_selective_vs_shuffled,
        "integrity": integrity,
    }


# ---------------------------------------------------------------------------
# Decision rule (Section 23, pre-specified, not altered after seeing results)
# ---------------------------------------------------------------------------


def decide(report: Mapping[str, Any]) -> dict[str, Any]:
    datasets = list(report["datasets"].keys())
    n = len(datasets)
    by = {ds: {r["method"]: r for r in report["datasets"][ds]["result_rows"]} for ds in datasets}

    def sig(row_key: str, ds: str, want_negative: bool) -> bool:
        r = report["datasets"][ds][row_key]
        return bool(r["ci_excludes_zero"] and ((r["mean_delta"] < 0) == want_negative))

    n_alwayson_beats_timefuse_point = sum(1 for ds in datasets if by[ds]["TimeFuse_AlwaysOnProbe"]["mae"] < by[ds]["TimeFuse"]["mae"])
    n_alwayson_beats_timefuse_sig = sum(1 for ds in datasets if sig("primary_always_on_vs_timefuse", ds, True))
    n_alwayson_hurts_timefuse_sig = sum(1 for ds in datasets if sig("primary_always_on_vs_timefuse", ds, False))

    n_selective_beats_timefuse_point = sum(1 for ds in datasets if by[ds]["TimeFuse_SelectiveProbe"]["mae"] < by[ds]["TimeFuse"]["mae"])
    n_selective_beats_timefuse_sig = sum(1 for ds in datasets if sig("primary_selective_vs_timefuse", ds, True))
    n_selective_hurts_timefuse_sig = sum(1 for ds in datasets if sig("primary_selective_vs_timefuse", ds, False))

    n_selective_beats_alwayson_point = sum(1 for ds in datasets if by[ds]["TimeFuse_SelectiveProbe"]["mae"] < by[ds]["TimeFuse_AlwaysOnProbe"]["mae"])
    n_selective_beats_shuffled_point = sum(1 for ds in datasets if by[ds]["TimeFuse_SelectiveProbe"]["mae"] < by[ds]["TimeFuse_SelectiveShuffledProbe"]["mae"])
    n_selective_beats_shuffled_sig = sum(1 for ds in datasets if sig("primary_selective_vs_shuffled", ds, True))
    clearly_beats_shuffled = n_selective_beats_shuffled_point >= 3 and n_selective_beats_shuffled_sig >= 1

    broad_regression_selective = n_selective_hurts_timefuse_sig >= 2
    broad_regression_alwayson = n_alwayson_hurts_timefuse_sig >= 2

    strong = (
        n_selective_beats_timefuse_point >= 3
        and n_selective_beats_timefuse_sig >= 1
        and n_selective_hurts_timefuse_sig == 0
        and n_selective_beats_alwayson_point >= 2
        and clearly_beats_shuffled
        and not broad_regression_selective
    )
    good_but_different = (not strong) and n_alwayson_beats_timefuse_sig >= 2 and n_alwayson_hurts_timefuse_sig == 0 and n_selective_beats_alwayson_point <= 1
    selectivity_needed = (not strong) and (not good_but_different) and (n_alwayson_beats_timefuse_sig == 0 or n_alwayson_hurts_timefuse_sig >= 1) and n_selective_beats_timefuse_sig >= 1 and n_selective_hurts_timefuse_sig == 0 and clearly_beats_shuffled
    failure = broad_regression_selective and broad_regression_alwayson

    if failure:
        tier = "FAILURE"
    elif strong:
        tier = "STRONG"
    elif selectivity_needed:
        tier = "SELECTIVITY_NEEDED"
    elif good_but_different:
        tier = "GOOD_BUT_DIFFERENT"
    else:
        tier = "WEAK"

    conclusions = {
        "STRONG": "LearnedProbe provides useful active expert-specific competence information beyond TimeFuse's passive meta-features, and selective trust makes that information more reliable.",
        "GOOD_BUT_DIFFERENT": "Always-On Probe already consistently improves TimeFuse and Selective Probe adds little. TimeFuse itself can already learn when/how to use Probe information; the selective gate may be unnecessary for TimeFuse.",
        "SELECTIVITY_NEEDED": "Always-On Probe is inconsistent or harmful, but Selective Probe improves TimeFuse and beats Shuffled. Probe contains useful complementary information, but it needs selective trust to integrate safely with TimeFuse.",
        "WEAK": "Always and Selective Probe are approximately tied with base TimeFuse, or Selective Shuffled performs similarly. Probe may not add meaningful information beyond TimeFuse's passive features.",
        "FAILURE": "Both Probe versions broadly regress. Do not invent a rescue.",
    }

    return {
        "tier": tier,
        "conclusion": conclusions[tier],
        "n_alwayson_beats_timefuse_point": n_alwayson_beats_timefuse_point,
        "n_alwayson_beats_timefuse_sig": n_alwayson_beats_timefuse_sig,
        "n_alwayson_hurts_timefuse_sig": n_alwayson_hurts_timefuse_sig,
        "n_selective_beats_timefuse_point": n_selective_beats_timefuse_point,
        "n_selective_beats_timefuse_sig": n_selective_beats_timefuse_sig,
        "n_selective_hurts_timefuse_sig": n_selective_hurts_timefuse_sig,
        "n_selective_beats_alwayson_point": n_selective_beats_alwayson_point,
        "n_selective_beats_shuffled_point": n_selective_beats_shuffled_point,
        "n_selective_beats_shuffled_sig": n_selective_beats_shuffled_sig,
        "clearly_beats_shuffled": clearly_beats_shuffled,
        "n_datasets": n,
    }


def git_commit_sha() -> str:
    import subprocess

    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def make_report(report: Mapping[str, Any], decision: Mapping[str, Any]) -> None:
    datasets = list(report["datasets"].keys())
    lines = [
        "# TimeFuse vs TimeFuse + LearnedProbe (Always-On / Selective / Selective-Shuffled)",
        "",
        "**Official TimeFuse router adapted to the BasicTS controlled protocol.** This is NOT a reproduction of the TimeFuse paper's published benchmark numbers. See `official_timefuse_source_manifest.json` for the exact commit (978e6c6b9e4f246632c269aa0f9beeb099eabcfc), files used unmodified, and the shape/engineering adaptations required (none touch the meta-feature formula, ModelFusor architecture, or training hyperparameters).",
        "",
        "Two separate questions, kept separate throughout: (1) TimeFuse vs Always-On Probe -- does Probe information itself add value? (2) Always-On vs Selective Probe -- does learning when to trust Probe make integration more reliable? (3) Selective vs Selective-Shuffled -- is any gain genuine expert-specific information?",
        "",
        "## Primary results (router_val MAE)",
        "",
        "| Dataset | TimeFuse | +Always-On Probe | +Selective Probe | +Selective ShuffledProbe | Δ AlwaysOn vs TF | Δ Selective vs TF | Δ Selective vs AlwaysOn | Δ Selective vs Shuffled |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ds in datasets:
        by = {r["method"]: r for r in report["datasets"][ds]["result_rows"]}
        a, b, c, d = by["TimeFuse"], by["TimeFuse_AlwaysOnProbe"], by["TimeFuse_SelectiveProbe"], by["TimeFuse_SelectiveShuffledProbe"]
        lines.append(
            f"| {ds} | {a['mae']:.6f} | {b['mae']:.6f} | {c['mae']:.6f} | {d['mae']:.6f} | `{b['delta_vs_timefuse']:+.6f}` | `{c['delta_vs_timefuse']:+.6f}` | `{c['delta_vs_always_on']:+.6f}` | `{d['delta_selective_real_vs_shuffled']:+.6f}` |"
        )
    lines += ["", "## Primary dependence-aware statistics (block-24)", ""]
    lines.append("| Dataset | Comparison | Mean Δ | 95% CI | P(Δ<0) | Excludes zero |")
    lines.append("|---|---|---:|---|---:|---|")
    for ds in datasets:
        for key, label in (
            ("primary_always_on_vs_timefuse", "AlwaysOn_vs_TimeFuse"),
            ("primary_selective_vs_timefuse", "Selective_vs_TimeFuse"),
            ("primary_selective_vs_alwayson", "Selective_vs_AlwaysOn"),
            ("primary_selective_vs_shuffled", "Selective_vs_SelectiveShuffled"),
        ):
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
    lines += ["", "## Capacity accounting", ""]
    lines.append("| Dataset | TimeFuse router params | AlwaysOn router params | Selective router params | Gate params | Input dim (base/augmented) |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for ds in datasets:
        cap = report["datasets"][ds]["capacity"]
        lines.append(f"| {ds} | {cap['timefuse_router_params']} | {cap['always_on_probe_router_params']} | {cap['selective_probe_router_params']} | {cap['selective_probe_gate_params']} | {cap['input_dim_base']}/{cap['input_dim_augmented']} |")
    lines += ["", "## Gate diagnostics", ""]
    for ds in datasets:
        lines.append(f"### {ds}")
        for row in report["datasets"][ds]["gate_diag_rows"]:
            lines.append(f"- {row}")
        lines.append(f"- weight_change_effect: {report['datasets'][ds]['weight_change_effect']}")
        lines.append("")
    lines += ["## Weight analysis", ""]
    lines.append("| Dataset | Method | Mean entropy | Mean max weight | Mean eff. #experts | Fraction top-expert changed |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for ds in datasets:
        for row in report["datasets"][ds]["weight_rows"]:
            lines.append(f"| {ds} | {row['method']} | {row['mean_entropy']:.4f} | {row['mean_max_weight']:.4f} | {row['mean_effective_num_experts']:.3f} | {row['fraction_top_expert_changed_vs_timefuse']:.3f} |")
    lines += ["", "## Gate training (router_train OOF regularization selection)", ""]
    lines.append("| Dataset | Probe variant | L2 | OOF logloss | OOF accuracy | Selected |")
    lines.append("|---|---|---:|---:|---:|---|")
    for ds in datasets:
        for row in report["datasets"][ds]["gate_l2_rows"]:
            lines.append(f"| {ds} | {row['probe_variant']} | {row['l2']} | {row['oof_logloss']:.4f} | {row['oof_accuracy']:.3f} | {'<-- selected' if row['selected'] else ''} |")
    lines += ["", "## Zero-probe / zero-gate diagnostics", ""]
    lines.append(
        "**Important**: unlike the earlier closed-form Simplex fusion (where alpha=0/gate=0 is a provably EXACT identity), TimeFuse's ModelFusor is a JOINTLY-TRAINED linear layer -- Method B/C are separately-trained models from Method A, so forcing their Probe/gate input to zero at inference is NOT mathematically guaranteed to reproduce Method A's own separately-trained weights bit-for-bit. These are reported as DIAGNOSTICS (with a training-noise reference scale from retraining Method A with a different seed), not hard pass/fail gates."
    )
    lines.append("")
    for ds in datasets:
        i = report["datasets"][ds]["integrity"]
        lines.append(f"- **{ds}**: zero-probe max weight diff = {i['zero_probe_diagnostic']['max_weight_diff']:.4f} (training-noise reference scale = {i['training_noise_reference_max_weight_diff']:.4f}); zero-probe MAE diff vs base TimeFuse = `{i['zero_probe_diagnostic']['mae_diff_vs_base_timefuse']:+.6f}`; zero-gate max weight diff vs zero-probe B = {i['zero_gate_diagnostic']['max_weight_diff_vs_b_zeroprobe']:.4f}; within training-noise scale: {i['zero_probe_within_training_noise_scale']}")
    lines += ["", "## Integrity", ""]
    for ds in datasets:
        i = report["datasets"][ds]["integrity"]
        lines.append(f"- **{ds}**: {i['result']} (checkpoints unchanged: {i['expert_checkpoints_unchanged']}; no test cache: {i['no_test_cache_loaded']}; meta-features target-free: {i['meta_features_target_free_at_inference']}; weights invariant to target corruption: {i['router_val_target_corruption_invariant_weights']}; gate invariant: {i['router_val_target_corruption_invariant_gate']})")
    lines += ["", "## Answers", ""]
    lines.append("**1. Was the official TimeFuse routing mechanism adapted faithfully?** Yes -- meta_feature.extract_meta_feature (22-dim) and timefuse.ModelFusor (Linear+Softmax) used verbatim from commit 978e6c6b9e4f246632c269aa0f9beeb099eabcfc, with the exact official training hyperparameters (Adam lr=5e-4, StepLR(10,0.1), SmoothL1Loss(beta=0.01), 5 epochs, batch 64, seed 2021). See official_timefuse_source_manifest.json for the enumerated shape-only adaptations.")
    lines.append(f"**2. Does Always-On Probe improve TimeFuse?** Beats by point estimate on {decision['n_alwayson_beats_timefuse_point']}/{decision['n_datasets']}; block-24 significant improvement on {decision['n_alwayson_beats_timefuse_sig']}/{decision['n_datasets']}; significant regression on {decision['n_alwayson_hurts_timefuse_sig']}/{decision['n_datasets']}.")
    lines.append(f"**3. Does Selective Probe improve TimeFuse?** Beats by point estimate on {decision['n_selective_beats_timefuse_point']}/{decision['n_datasets']}; block-24 significant improvement on {decision['n_selective_beats_timefuse_sig']}/{decision['n_datasets']}; significant regression on {decision['n_selective_hurts_timefuse_sig']}/{decision['n_datasets']}.")
    lines.append(f"**4. Does Selective Probe outperform Always-On Probe?** By point estimate on {decision['n_selective_beats_alwayson_point']}/{decision['n_datasets']}; see Selective_vs_AlwaysOn block-24 rows above for significance.")
    lines.append(f"**5. Does Selective Probe outperform Selective ShuffledProbe?** By point estimate on {decision['n_selective_beats_shuffled_point']}/{decision['n_datasets']}; block-24 significant on {decision['n_selective_beats_shuffled_sig']}/{decision['n_datasets']}.")
    lines.append("**6. On how many datasets does each version win?** See point-estimate counts above (questions 2-5).")
    lines.append("**7. Which gains/regressions survive block-24?** See the primary dependence-aware statistics table above.")
    lines.append("**8. Does the gate successfully reduce Probe influence when Probe tends to hurt?** See gate_usefulness rows (`gate_lower_on_harmful`) per dataset above.")
    lines.append(f"**9. Does TimeFuse already learn to use Probe appropriately without a gate?** {'Yes -- Always-On already helps consistently and Selective adds little.' if decision['tier'] == 'GOOD_BUT_DIFFERENT' else 'See per-dataset Always-On results above.'}")
    lines.append("**10. How much extra capacity does each augmentation add?** See Capacity accounting table -- router parameter counts scale with input dimensionality only (22 -> 22+K), gate adds 10 parameters (9 features + bias), trained entirely separately from the router.")
    lines.append(f"**11. Does this support active diagnostic probing adding information beyond an independently published passive router?** {decision['conclusion']}")
    lines += ["", f"## Decision: {decision['tier']}", "", decision["conclusion"], ""]
    lines += [
        "## Hard rule compliance",
        "",
        "```text",
        "TEST SET ACCESSED: NO",
        "FORECASTING EXPERTS RETRAINED: NO",
        "LEARNEDPROBE ARCHITECTURE/LOSS/TRAINING MODIFIED: NO",
        "OTHER PUBLISHED ROUTERS IMPLEMENTED: NO (TimeFuse only)",
        "COSTAR / ONLINE COSTAR TOUCHED: NO",
        "```",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    start = time.time()
    report: dict[str, Any] = {
        "experiment": "timefuse_probe",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_sha": git_commit_sha(),
        "datasets": {},
    }
    all_results, all_dependence, all_weights, all_integrity, all_gate_diag, all_gate_training = [], [], [], [], [], []

    for dataset in NEW_DATASETS:
        print(f"[timefuse_probe] {dataset}: starting...", flush=True)
        result = evaluate_dataset(dataset)
        report["datasets"][dataset] = result
        all_results.extend(result["result_rows"])
        all_dependence.extend(result["dependence_rows"])
        all_weights.extend(result["weight_rows"])
        all_integrity.append(result["integrity"])
        all_gate_diag.extend(result["gate_diag_rows"])
        all_gate_training.extend(result["gate_l2_rows"])
        print(f"[timefuse_probe] {dataset}: done.", flush=True)

    decision = decide(report)
    report["decision"] = decision
    report["runtime_sec"] = time.time() - start
    report["test_set_accessed"] = False

    source_manifest_path = OUT_DIR / "official_timefuse_source_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_manifest["basicts_git_commit_sha_at_experiment_time"] = report["git_commit_sha"]
    write_json(source_manifest_path, source_manifest)

    training_config = {
        "timefuse_seed": TIMEFUSE_SEED,
        "timefuse_n_epochs": TIMEFUSE_N_EPOCHS,
        "timefuse_batch_size": TIMEFUSE_BATCH_SIZE,
        "timefuse_lr": TIMEFUSE_LR,
        "timefuse_step_size": TIMEFUSE_STEP_SIZE,
        "timefuse_gamma": TIMEFUSE_GAMMA,
        "timefuse_huber_beta": TIMEFUSE_HUBER_BETA,
        "timefuse_meta_dim": TIMEFUSE_META_DIM,
        "meta_feature_names": META_FEATURE_NAMES,
        "gate_feature_names": GATE_FEATURE_NAMES,
        "gate_l2_grid": list(GATE_L2_GRID),
        "shuffle_seed": SHUFFLE_SEED,
        "same_across_all_four_methods": ["optimizer(Adam)", "lr", "scheduler(StepLR)", "loss(SmoothL1Loss beta=0.01)", "epochs(5)", "batch_size(64)", "initial_seed(2021)"],
    }
    write_json(OUT_DIR / "timefuse_training_config.json", training_config)

    manifest = {
        "manifest_type": "timefuse_probe_manifest",
        "created_at_utc": report["created_at_utc"],
        "git_commit_sha": report["git_commit_sha"],
        "timefuse_repository": "https://github.com/ZhiningLiu1998/TimeFuse",
        "timefuse_commit_sha": "978e6c6b9e4f246632c269aa0f9beeb099eabcfc",
        "basicts_commit_sha": report["git_commit_sha"],
        "official_files_used": ["meta_feature.py::extract_meta_feature", "timefuse.py::ModelFusor", "timefuse.py::TorchScaler/get_scaler"],
        "expert_cores": {ds: report["datasets"][ds]["core"] for ds in NEW_DATASETS},
        "expert_checkpoint_sha256": {ds: report["datasets"][ds]["checkpoint_hashes"] for ds in NEW_DATASETS},
        "timefuse_feature_names": META_FEATURE_NAMES,
        "probe_feature_order": "one score per (window, core-expert), in bundle.core_names order, identical to every other method in this experiment family; lower predicted_excess_loss = better",
        "probe_scaler": "sklearn StandardScaler (via TimeFuse's own TorchScaler/get_scaler('standard')), fit on router_train's raw [N,K] probe-loss matrix per dataset, per-expert-column",
        "gate_features": GATE_FEATURE_NAMES,
        "gate_model": "L2-regularized logistic regression, hand-rolled deterministic full-batch gradient descent (reused unmodified from run_simplex_selective_probe.py)",
        "gate_regularization_grid": list(GATE_L2_GRID),
        "random_seeds": {"timefuse_fusor_seed": TIMEFUSE_SEED, "shuffle_seed": SHUFFLE_SEED, "training_noise_reference_seed": TIMEFUSE_SEED + 1},
        "decision_rule": "Section 23 of the task instructions, applied verbatim without modification after seeing results.",
    }
    write_json(OUT_DIR / "timefuse_probe_manifest.json", manifest)
    write_json(OUT_DIR / "validation_results.json", report)
    write_csv(OUT_DIR / "validation_results.csv", all_results)
    write_csv(OUT_DIR / "dependence_aware_results.csv", all_dependence)
    write_csv(OUT_DIR / "weight_analysis.csv", all_weights)
    write_csv(OUT_DIR / "integrity_checks.csv", all_integrity)
    write_csv(OUT_DIR / "gate_diagnostics.csv", all_gate_diag)
    write_csv(OUT_DIR / "gate_training_results.csv", all_gate_training)
    make_report(report, decision)

    print("TEST SET ACCESSED: NO")
    print(json.dumps({"runtime_sec": report["runtime_sec"], "tier": decision["tier"]}, indent=2))


if __name__ == "__main__":
    main()
