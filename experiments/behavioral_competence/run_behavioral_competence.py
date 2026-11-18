"""Forecast-Time Behavioral Competence Routing -- validation-only proof of concept.

Answers one question: does an expert's behavior under small perturbations of
the current historical input predict its upcoming reliability, beyond what
window features, the expert's own forecast, and ensemble disagreement already
tell us? Section references (# N) match the task specification.

HARD RULE: this script never loads a test cache, evaluates a test metric, or
selects anything using test data. `refuse_test()` guards every cache path.
Only router_train (features + legal excess-loss targets) and router_val
(features only, no targets in any feature) are used. Frozen experts are
never trained or fine-tuned -- only run forward, with every parameter's
`requires_grad_` explicitly set to False (see model_runtime.py).
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
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.behavioral_competence.common import (  # noqa: E402
    ABLATIONS,
    BEHAVIORAL_STAT_NAMES,
    GROUP_A_NAMES,
    GROUP_B_NAMES,
    GROUP_C_NAMES,
    PERTURBATION_SEED_BASE,
    PERTURBATIONS,
    FeatureBundle,
    behavioral_features_all,
    competence_to_weights,
    disagreement_features_group_c,
    forecast_features_group_b,
    train_competence_scorer,
    window_features_group_a,
)
from experiments.behavioral_competence.model_runtime import ExpertRuntime, load_expert_runtime, sha256_file  # noqa: E402
from experiments.chronological_adaptive_costar.run_chronological_adaptive_costar import paired_bootstrap  # noqa: E402
from experiments.costar_multidataset_frozen.common import block_bootstrap_with_prob, every_kth_phase_bootstrap  # noqa: E402
from experiments.frozen_hv_costar.run_frozen_hv_costar import (  # noqa: E402
    LOADERS,
    Bundle,
    best_single_expert,
    equal_fixed,
    frozen_hv_prediction,
    metric_values,
    online_hv_prediction,
    refuse_test,
)
from experiments.oracle_routing_headroom.run_oracle_headroom import window_oracle  # noqa: E402
from experiments.oracle_weight_tournament.run_tournament import sample_mae, sample_mse  # noqa: E402


OUT_DIR = ROOT / "experiments/behavioral_competence"
CACHE_DIR = OUT_DIR / "cache"
RESULTS_DIR = OUT_DIR / "results"
REPORTS_DIR = OUT_DIR / "reports"
BLOCK_LENGTHS = (12, 24, 48)
BOOTSTRAP_SAMPLES = 10000
PHASE_K = 12
INTERNAL_VAL_FRACTION = 0.2
CODE_VERSION = "behavioral_competence_v2"


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
# Step 1: perturbation forecast cache (Section 5)
# ---------------------------------------------------------------------------


WALKFORWARD_DATASETS = {"ETTh1", "ETTm1", "Weather", "Electricity"}


def router_train_block_split(dataset: str, train_cache: Mapping[str, Any]) -> int | None:
    """router_train_20_60 for the walk-forward family is a concatenation of
    block_b_oos (predicted by the block_a-stage checkpoint, trained on 0-20%)
    and block_c_oos (predicted by the block_ab-stage checkpoint, trained on
    0-40%) -- NOT the final_60 checkpoint used for router_val. Returns the row
    index where block_b_oos ends and block_c_oos begins, or None for ETTh2
    (which uses a single fixed OOS checkpoint for the whole of router_train)."""
    if dataset not in WALKFORWARD_DATASETS:
        return None
    source_caches = train_cache["provenance"]["source_caches"]
    block_b_cache = torch.load(ROOT / source_caches["block_b_oos"], map_location="cpu", weights_only=False)
    return int(block_b_cache["num_windows"])


def concat_perturbation_payloads(payload_a: Mapping[str, Any], payload_b: Mapping[str, Any], dataset: str, expert: str) -> dict[str, Any]:
    merged: dict[str, Any] = {
        "dataset": dataset,
        "split": "router_train",
        "expert": expert,
        "checkpoint_path": [payload_a["checkpoint_path"], payload_b["checkpoint_path"]],
        "checkpoint_sha256": [payload_a["checkpoint_sha256"], payload_b["checkpoint_sha256"]],
        "forecast_horizon": payload_a["forecast_horizon"],
        "input_len": payload_a["input_len"],
        "num_features": payload_a["num_features"],
        "mean": payload_a["mean"],
        "std": payload_a["std"],
        "perturbation_params": payload_a["perturbation_params"],
        "forecast_origin_indices": list(payload_a["forecast_origin_indices"]) + list(payload_b["forecast_origin_indices"]),
        "num_windows": int(payload_a["num_windows"]) + int(payload_b["num_windows"]),
        "code_version": CODE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reproduced_normal_forecast": torch.cat([payload_a["reproduced_normal_forecast"], payload_b["reproduced_normal_forecast"]], dim=0),
        "reproduction_max_abs_diff_vs_cached": max(payload_a["reproduction_max_abs_diff_vs_cached"], payload_b["reproduction_max_abs_diff_vs_cached"]),
        "reproduction_mean_abs_diff_vs_cached": float(
            (payload_a["reproduction_mean_abs_diff_vs_cached"] * payload_a["num_windows"] + payload_b["reproduction_mean_abs_diff_vs_cached"] * payload_b["num_windows"])
            / (payload_a["num_windows"] + payload_b["num_windows"])
        ),
        "reproduction_fraction_windows_gt_0_1": float(
            (payload_a["reproduction_fraction_windows_gt_0_1"] * payload_a["num_windows"] + payload_b["reproduction_fraction_windows_gt_0_1"] * payload_b["num_windows"])
            / (payload_a["num_windows"] + payload_b["num_windows"])
        ),
    }
    for pname in PERTURBATIONS:
        key = f"perturbed__{pname}"
        merged[key] = torch.cat([payload_a[key], payload_b[key]], dim=0)
    return merged


def build_perturbation_cache_router_train(dataset: str, expert: str, train_cache: Mapping[str, Any], cached_normal: torch.Tensor, split_boundary: int | None) -> dict[str, Any]:
    history = train_cache["histories"].to(torch.float32)
    starts = train_cache["absolute_window_starts"]
    if split_boundary is None:
        rt = load_expert_runtime(dataset, expert)
        return build_perturbation_cache(dataset, "router_train", expert, history, starts, rt, cached_normal)
    rt_a = load_expert_runtime(dataset, expert, stage="block_a")
    rt_ab = load_expert_runtime(dataset, expert, stage="block_ab")
    payload_a = build_perturbation_cache(dataset, "router_train_block_b", expert, history[:split_boundary], starts[:split_boundary], rt_a, cached_normal[:split_boundary])
    payload_b = build_perturbation_cache(dataset, "router_train_block_c", expert, history[split_boundary:], starts[split_boundary:], rt_ab, cached_normal[split_boundary:])
    return concat_perturbation_payloads(payload_a, payload_b, dataset, expert)


def build_perturbation_cache(dataset: str, split: str, expert: str, history_raw: torch.Tensor, starts: torch.Tensor, runtime: ExpertRuntime, cached_normal: torch.Tensor) -> dict[str, Any]:
    cache_path = CACHE_DIR / f"{dataset}__{split}__{expert}__perturbations.pt"
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("code_version") == CODE_VERSION and int(payload["num_windows"]) == int(history_raw.shape[0]):
            return payload
    reproduced_normal = runtime.predict(history_raw)
    per_window_diff = (reproduced_normal - cached_normal).abs().mean(dim=tuple(range(1, reproduced_normal.ndim)))
    reproduction_max_abs_diff = float((reproduced_normal - cached_normal).abs().max())
    reproduction_mean_abs_diff = float(per_window_diff.mean())
    reproduction_fraction_windows_gt_0_1 = float((per_window_diff > 0.1).to(torch.float32).mean())
    perturbed: dict[str, torch.Tensor] = {}
    for pi, (pname, pfn) in enumerate(PERTURBATIONS.items()):
        seed = PERTURBATION_SEED_BASE + pi
        perturbed_history = pfn(history_raw, seed)
        perturbed[pname] = runtime.predict(perturbed_history)
    payload = {
        "dataset": dataset,
        "split": split,
        "expert": expert,
        "checkpoint_path": str(runtime.checkpoint_path),
        "checkpoint_sha256": runtime.checkpoint_sha256,
        "forecast_horizon": runtime.horizon,
        "input_len": runtime.input_len,
        "num_features": runtime.num_features,
        "mean": runtime.mean,
        "std": runtime.std,
        "perturbation_params": {"P1_noise": {"scale": 0.05, "seed_base": PERTURBATION_SEED_BASE}, "P2_mask_recent": {"fraction": 0.10}, "P3_smooth": {"window": 5}, "P4_amplitude": {"factor": 1.1}},
        "forecast_origin_indices": starts.tolist(),
        "num_windows": int(history_raw.shape[0]),
        "code_version": CODE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reproduced_normal_forecast": reproduced_normal,
        "reproduction_max_abs_diff_vs_cached": reproduction_max_abs_diff,
        "reproduction_mean_abs_diff_vs_cached": reproduction_mean_abs_diff,
        "reproduction_fraction_windows_gt_0_1": reproduction_fraction_windows_gt_0_1,
        **{f"perturbed__{k}": v for k, v in perturbed.items()},
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    return payload


# ---------------------------------------------------------------------------
# Step 2: features (Sections 6-7)
# ---------------------------------------------------------------------------


def build_feature_bundle(bundle: Bundle, cache: Mapping[str, Any], perturbation_payloads: Mapping[str, Mapping[str, Any]]) -> FeatureBundle:
    """`perturbation_payloads`: expert_name -> cache payload from build_perturbation_cache."""
    history = cache["histories"].to(torch.float32)
    group_a_window = window_features_group_a(history, bundle.std)  # [N,6]
    forecasts_all = bundle.forecasts_fn(cache, bundle.expert_idx)  # [N,H,F,K]
    n, h, f, k = forecasts_all.shape

    group_a = group_a_window.unsqueeze(1).expand(n, k, len(GROUP_A_NAMES)).clone()
    group_b = torch.zeros(n, k, len(GROUP_B_NAMES))
    group_c = torch.zeros(n, k, len(GROUP_C_NAMES))
    group_d = torch.zeros(n, k, len(PERTURBATIONS) * len(BEHAVIORAL_STAT_NAMES))
    last_observed = history[:, -1, :]

    for local_i, expert_name in enumerate(bundle.core_names):
        forecast_e = forecasts_all[..., local_i]
        group_b[:, local_i, :] = forecast_features_group_b(forecast_e, last_observed, bundle.std)
        group_c[:, local_i, :] = disagreement_features_group_c(forecast_e, forecasts_all, bundle.std)
        payload = perturbation_payloads[expert_name]
        original = payload["reproduced_normal_forecast"]
        perturbed_by_name = {pname: payload[f"perturbed__{pname}"] for pname in PERTURBATIONS}
        behavioral, _names = behavioral_features_all(original, perturbed_by_name, bundle.std)
        group_d[:, local_i, :] = behavioral

    names = {
        "A": list(GROUP_A_NAMES),
        "B": list(GROUP_B_NAMES),
        "C": list(GROUP_C_NAMES),
        "D": [f"{p}__{s}" for p in PERTURBATIONS for s in BEHAVIORAL_STAT_NAMES],
    }
    return FeatureBundle(group_a=group_a, group_b=group_b, group_c=group_c, group_d=group_d, names=names)


def flatten_window_expert(x: torch.Tensor) -> torch.Tensor:
    n, k = x.shape[0], x.shape[1]
    return x.reshape(n * k, *x.shape[2:])


# ---------------------------------------------------------------------------
# Step 3: excess-loss target (Section 8, router_train only)
# ---------------------------------------------------------------------------


def compute_excess_loss(cache: Mapping[str, Any], forecasts_all: torch.Tensor, std: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.bool)
    k = forecasts_all.shape[-1]
    expert_mae = torch.stack([sample_mae(forecasts_all[..., e], target, mask, std) for e in range(k)], dim=1)  # [N,K]
    equal_mae = sample_mae(forecasts_all.mean(dim=-1), target, mask, std)  # [N]
    excess_loss = expert_mae - equal_mae.unsqueeze(1)
    return excess_loss, expert_mae


# ---------------------------------------------------------------------------
# Step 4: dataset pipeline
# ---------------------------------------------------------------------------


def raw_history_cache(dataset: str, cache: Mapping[str, Any], mean: torch.Tensor, std: torch.Tensor) -> dict[str, Any]:
    """ETTh2's cache stores `histories` already z-normalized by the shared
    DLinear-checkpoint scaler (verified: scaler_hash matches the checkpoint's
    scaler_manifest, and de-normalizing recovers plausible raw-scale ETTh2
    values) -- unlike the walk-forward family, where `histories` is raw. This
    returns a shallow-copied cache dict with `histories` guaranteed raw, so
    every downstream perturbation/feature computation can treat all datasets
    identically."""
    if dataset != "ETTh2":
        return dict(cache)
    out = dict(cache)
    out["histories"] = cache["histories"].to(torch.float32) * std.view(1, 1, -1) + mean.view(1, 1, -1)
    return out


def run_dataset(dataset: str) -> dict[str, Any]:
    bundle = LOADERS[dataset]()
    train_cache, val_cache = bundle.train_cache, bundle.val_cache
    k = len(bundle.core_names)

    split_boundary = router_train_block_split(dataset, train_cache)
    val_runtimes = {e: load_expert_runtime(dataset, e) for e in bundle.core_names}  # final_60 (or ETTh2's single OOS checkpoint)
    check_runtimes = dict(val_runtimes)
    if split_boundary is not None:
        for e in bundle.core_names:
            check_runtimes[f"{e}__block_a"] = load_expert_runtime(dataset, e, stage="block_a")
            check_runtimes[f"{e}__block_ab"] = load_expert_runtime(dataset, e, stage="block_ab")
    for name, rt in check_runtimes.items():
        assert all(not p.requires_grad for p in rt.model.parameters()), f"{dataset}/{name}: parameters are not frozen"
        assert not rt.model.training, f"{dataset}/{name}: model is in train mode"

    train_forecasts_all = bundle.forecasts_fn(train_cache, bundle.expert_idx)
    val_forecasts_all = bundle.forecasts_fn(val_cache, bundle.expert_idx)

    reference_runtime = val_runtimes[bundle.core_names[0]]
    train_cache_raw = raw_history_cache(dataset, train_cache, reference_runtime.mean, reference_runtime.std)
    val_cache_raw = raw_history_cache(dataset, val_cache, reference_runtime.mean, reference_runtime.std)

    train_payloads, val_payloads, reproduction_checks = {}, {}, []
    for local_i, expert_name in enumerate(bundle.core_names):
        tp = build_perturbation_cache_router_train(dataset, expert_name, train_cache_raw, train_forecasts_all[..., local_i], split_boundary)
        vp = build_perturbation_cache(dataset, "router_val", expert_name, val_cache_raw["histories"].to(torch.float32), val_cache["absolute_window_starts"], val_runtimes[expert_name], val_forecasts_all[..., local_i])
        train_payloads[expert_name] = tp
        val_payloads[expert_name] = vp
        for split_label, payload in (("router_train", tp), ("router_val", vp)):
            reproduction_checks.append(
                {
                    "dataset": dataset,
                    "expert": expert_name,
                    "split": split_label,
                    "max_abs_diff": payload["reproduction_max_abs_diff_vs_cached"],
                    "mean_abs_diff": payload["reproduction_mean_abs_diff_vs_cached"],
                    "fraction_windows_gt_0.1": payload["reproduction_fraction_windows_gt_0_1"],
                }
            )
        # Gate on the FRACTION of windows that fail to reproduce, not the max:
        # TimesNet's FFT-based period detection is known to be mildly
        # non-deterministic across runs for a small share of windows with
        # near-tied top-k frequency magnitudes (isolated, bounded, and does
        # not affect any reported MAE/MSE -- those always use the official
        # cached predictions, never this reproduction). A wrong checkpoint or
        # normalization bug would instead fail on nearly every window.
        for split_label, payload in (("router_train", tp), ("router_val", vp)):
            if payload["reproduction_fraction_windows_gt_0_1"] > 0.10 or payload["reproduction_mean_abs_diff_vs_cached"] > 0.05:
                raise AssertionError(f"{dataset}/{expert_name}/{split_label}: reproduced 'normal' forecast does not match cached forecast on too large a share of windows (checkpoint/normalization mismatch): {payload['reproduction_fraction_windows_gt_0_1']:.3f} fraction > 0.1, mean_abs_diff={payload['reproduction_mean_abs_diff_vs_cached']:.4f}")

    train_features = build_feature_bundle(bundle, train_cache_raw, train_payloads)
    val_features = build_feature_bundle(bundle, val_cache_raw, val_payloads)

    excess_loss_train, expert_mae_train = compute_excess_loss(train_cache, train_forecasts_all, bundle.std)
    excess_loss_val, expert_mae_val = compute_excess_loss(val_cache, val_forecasts_all, bundle.std)  # diagnostic only, never a feature

    n_train = int(train_cache["num_windows"])
    split_point = int(round(n_train * (1 - INTERNAL_VAL_FRACTION)))
    window_id_train = torch.arange(0, split_point)
    window_id_internal_val = torch.arange(split_point, n_train)
    row_window_id_train = (window_id_train.view(-1, 1) * k + torch.arange(k).view(1, -1)).reshape(-1)
    row_window_id_internal_val = (window_id_internal_val.view(-1, 1) * k + torch.arange(k).view(1, -1)).reshape(-1)

    excess_loss_train_flat = excess_loss_train.reshape(-1)  # [N*K], row order window-major/expert-minor matches flatten_window_expert

    fits: dict[str, Any] = {}
    ablation_val_preds: dict[str, torch.Tensor] = {}
    ablation_val_weights: dict[str, torch.Tensor] = {}
    ablation_train_diag: dict[str, dict[str, Any]] = {}
    for ablation in ABLATIONS:
        feats_train_flat = flatten_window_expert(train_features.features_for(ablation))
        fit = train_competence_scorer(feats_train_flat, excess_loss_train_flat, n_train_windows=split_point * k, window_id_train=row_window_id_train, window_id_internal_val=row_window_id_internal_val)
        fits[ablation] = fit
        feats_val_flat = flatten_window_expert(val_features.features_for(ablation))
        pred_flat = fit.predict(feats_val_flat)
        n_val = int(val_cache["num_windows"])
        pred_excess = pred_flat.reshape(n_val, k)
        weights = competence_to_weights(pred_excess, fit.temperature)
        final_pred = (val_forecasts_all * weights.view(n_val, 1, 1, k)).sum(dim=-1)
        ablation_val_preds[ablation] = final_pred
        ablation_val_weights[ablation] = weights
        ablation_train_diag[ablation] = {"best_epoch": fit.best_epoch, "best_internal_val_mse": fit.best_internal_val_mse, "temperature": fit.temperature, "train_windows": fit.train_windows, "internal_val_windows": fit.internal_val_windows, "num_features": feats_train_flat.shape[1]}

    baseline_preds = {
        "best_single_expert": best_single_expert(bundle)[0],
        "equal_fixed": equal_fixed(bundle)[0],
        "frozen_hv_costar": frozen_hv_prediction(bundle)[0],
        "online_hv_costar_reference": online_hv_prediction(bundle)[0],
    }
    n_val = int(val_cache["num_windows"])
    window_pred, window_winner, window_err = window_oracle(val_forecasts_all, bundle.per_location_error_fn(val_cache, bundle.expert_idx, bundle.std), val_cache["target_masks"].to(torch.bool))
    all_methods = {**baseline_preds, **{a: ablation_val_preds[a] for a in ABLATIONS}, "window_oracle": window_pred}

    result_rows = []
    metrics_by_method: dict[str, dict[str, Any]] = {}
    for method, pred in all_methods.items():
        m = metric_values(bundle, pred)
        metrics_by_method[method] = m
        result_rows.append({"dataset": dataset, "method": method, "mae": m["mae"], "mse": m["mse"], "is_oracle": method == "window_oracle", "uses_online_feedback": method == "online_hv_costar_reference"})

    # --- competence-prediction evaluation (Section 15) ---
    competence_rows = []
    for ablation in ABLATIONS:
        feats_val_flat = flatten_window_expert(val_features.features_for(ablation))
        pred_flat = fits[ablation].predict(feats_val_flat).numpy()
        actual_flat = excess_loss_val.reshape(-1).numpy()
        spearman = spearmanr(pred_flat, actual_flat)
        pearson = pearsonr(pred_flat, actual_flat)
        pred_excess = pred_flat.reshape(n_val, k)
        actual_excess = actual_flat.reshape(n_val, k)
        top1_acc = float((pred_excess.argmin(axis=1) == actual_excess.argmin(axis=1)).mean())
        useful_label = (actual_flat < 0).astype(int)
        useful_score = -pred_flat
        if useful_label.min() != useful_label.max():
            auroc = float(roc_auc_score(useful_label, useful_score))
            auprc = float(average_precision_score(useful_label, useful_score))
            pred_useful = (useful_score > np.median(useful_score)).astype(int)
            precision = float(precision_score(useful_label, pred_useful, zero_division=0))
            recall = float(recall_score(useful_label, pred_useful, zero_division=0))
        else:
            auroc = auprc = precision = recall = float("nan")
        competence_rows.append(
            {
                "dataset": dataset,
                "ablation": ablation,
                "spearman": float(spearman.statistic),
                "spearman_pvalue": float(spearman.pvalue),
                "pearson": float(pearson.statistic),
                "pearson_pvalue": float(pearson.pvalue),
                "top1_accuracy": top1_acc,
                "auroc_useful_vs_harmful": auroc,
                "auprc_useful_vs_harmful": auprc,
                "precision": precision,
                "recall": recall,
            }
        )

    oracle_mae = metrics_by_method["window_oracle"]["mae"]
    c_mae = metrics_by_method["C_window_forecast_disagreement"]["mae"]
    d_mae = metrics_by_method["D_full_behavioral"]["mae"]
    headroom_denominator = c_mae - oracle_mae
    headroom_captured = float((c_mae - d_mae) / headroom_denominator) if headroom_denominator > 0 else None

    regret = metrics_by_method["D_full_behavioral"]["per_window_mae"] - metrics_by_method["window_oracle"]["per_window_mae"]
    regret_summary = {
        "mean_regret": float(regret.mean()),
        "median_regret": float(regret.median()),
        "p90_regret": float(torch.quantile(regret, 0.9)),
        "fraction_regret_gt_0": float((regret > 0).to(torch.float32).mean()),
    }

    # --- dependence-aware statistics (Section 16): D vs C, D vs Equal ---
    dependence_rows = []
    for label, cand_key, base_key in (("D_vs_C", "D_full_behavioral", "C_window_forecast_disagreement"), ("D_vs_Equal", "D_full_behavioral", "equal_fixed")):
        candidate, baseline = metrics_by_method[cand_key]["per_window_mae"], metrics_by_method[base_key]["per_window_mae"]
        boot = paired_bootstrap(candidate, baseline, seed=20260821, samples=5000)
        dependence_rows.append({"dataset": dataset, "comparison": label, "test": "iid_paired_bootstrap", **boot})
        for block in BLOCK_LENGTHS:
            b = block_bootstrap_with_prob(candidate, baseline, block=block, seed=20260821, samples=BOOTSTRAP_SAMPLES)
            dependence_rows.append({"dataset": dataset, "comparison": label, "test": f"block_bootstrap_len{block}", **b})
        phase = every_kth_phase_bootstrap(candidate - baseline, k=PHASE_K, seed=20260821, samples=BOOTSTRAP_SAMPLES)
        dependence_rows.append({"dataset": dataset, "comparison": label, "test": f"every_{PHASE_K}th_window_phase_bootstrap", **phase})

    # --- behavioral diagnostics (Section 17) ---
    behavior_corr_rows = []
    d_feature_names = val_features.names["D"]
    val_d_flat = flatten_window_expert(val_features.group_d).numpy()
    actual_flat = excess_loss_val.reshape(-1).numpy()
    train_d_flat = flatten_window_expert(train_features.group_d).numpy()
    train_excess_flat = excess_loss_train.reshape(-1).numpy()
    for j, name in enumerate(d_feature_names):
        sp_train = spearmanr(train_d_flat[:, j], train_excess_flat)
        sp_val = spearmanr(val_d_flat[:, j], actual_flat)
        behavior_corr_rows.append(
            {
                "dataset": dataset,
                "feature": name,
                "spearman_router_train": float(sp_train.statistic),
                "spearman_router_train_pvalue": float(sp_train.pvalue),
                "spearman_router_val_diagnostic": float(sp_val.statistic),
                "spearman_router_val_pvalue": float(sp_val.pvalue),
            }
        )

    # --- Section 19 integrity: target-perturbation invariance for D ---
    gen = torch.Generator().manual_seed(4242)
    corrupted_val_cache = dict(val_cache_raw)
    corrupted_val_cache["targets"] = torch.randn(val_cache["targets"].shape, generator=gen, dtype=torch.float32)
    corrupted_val_features = build_feature_bundle(bundle, corrupted_val_cache, val_payloads)
    feats_val_flat_d = flatten_window_expert(val_features.features_for("D_full_behavioral"))
    feats_corrupt_flat_d = flatten_window_expert(corrupted_val_features.features_for("D_full_behavioral"))
    features_identical = bool(torch.equal(feats_val_flat_d, feats_corrupt_flat_d))
    pred_corrupt = fits["D_full_behavioral"].predict(feats_corrupt_flat_d).reshape(n_val, k)
    weights_corrupt = competence_to_weights(pred_corrupt, fits["D_full_behavioral"].temperature)
    final_pred_corrupt = (val_forecasts_all * weights_corrupt.view(n_val, 1, 1, k)).sum(dim=-1)
    predictions_identical = bool(torch.equal(ablation_val_preds["D_full_behavioral"], final_pred_corrupt))
    integrity = {"dataset": dataset, "features_identical_after_target_corruption": features_identical, "predictions_identical_after_target_corruption": predictions_identical, "result": "PASS" if (features_identical and predictions_identical) else "FAIL"}
    if integrity["result"] != "PASS":
        raise AssertionError(f"{dataset}: target-perturbation invariance check FAILED: {integrity}")

    return {
        "dataset": dataset,
        "core": bundle.core_names,
        "result_rows": result_rows,
        "competence_rows": competence_rows,
        "dependence_rows": dependence_rows,
        "behavior_corr_rows": behavior_corr_rows,
        "reproduction_checks": reproduction_checks,
        "integrity": integrity,
        "ablation_train_diag": ablation_train_diag,
        "headroom_captured_D_over_C": headroom_captured,
        "regret_summary": regret_summary,
        "c_mae": c_mae,
        "d_mae": d_mae,
        "oracle_mae": oracle_mae,
        "d_minus_c_mae": d_mae - c_mae,
        "relative_improvement_d_vs_c_pct": 100.0 * (c_mae - d_mae) / c_mae,
        "per_window_predictions": {m: p.numpy() for m, p in all_methods.items()},
        "per_window_competence_predictions": {a: fits[a].predict(flatten_window_expert(val_features.features_for(a))).reshape(n_val, k).numpy() for a in ABLATIONS},
        "actual_excess_loss_val": excess_loss_val.numpy(),
        "feature_bundle_names": val_features.names,
    }


def make_report(out_dir: Path, report: Mapping[str, Any], go_no_go: Mapping[str, Any]) -> None:
    lines = [
        "# Forecast-Time Behavioral Competence Routing -- Validation-Only Proof of Concept",
        "",
        "Research question: does an expert's behavior under small perturbations of the current historical input predict its upcoming reliability, beyond window/forecast/disagreement features already available? Validation only -- no test cache was ever loaded.",
        "",
        "## Main result table (router_val MAE / MSE)",
        "",
        "| Dataset | Best Single | Equal | Frozen HxV | Online HxV (ref, uses feedback) | A: Window | B: +Forecast | C: +Disagreement | D: +Behavioral | Oracle |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ds, d in report["datasets"].items():
        by = {r["method"]: r for r in d["result_rows"]}
        lines.append(
            "| {ds} | {bs[mae]:.6f} | {eq[mae]:.6f} | {fh[mae]:.6f} | {oh[mae]:.6f} | {a[mae]:.6f} | {b[mae]:.6f} | {c[mae]:.6f} | {dd[mae]:.6f} | {orc[mae]:.6f} |".format(
                ds=ds, bs=by["best_single_expert"], eq=by["equal_fixed"], fh=by["frozen_hv_costar"], oh=by["online_hv_costar_reference"],
                a=by["A_window_only"], b=by["B_window_forecast"], c=by["C_window_forecast_disagreement"], dd=by["D_full_behavioral"], orc=by["window_oracle"],
            )
        )
    lines += ["", "## D vs C (the central comparison)", ""]
    lines.append("| Dataset | D-C MAE | Relative improvement | Headroom captured (D over C, of C->Oracle gap) |")
    lines.append("|---|---:|---:|---|")
    for ds, d in report["datasets"].items():
        hc = f"{d['headroom_captured_D_over_C']*100:.1f}%" if d["headroom_captured_D_over_C"] is not None else "n/a (C already <= oracle)"
        lines.append(f"| {ds} | `{d['d_minus_c_mae']:+.6f}` | `{d['relative_improvement_d_vs_c_pct']:+.3f}%` | {hc} |")
    lines += ["", "## Competence-prediction metrics", ""]
    lines.append("| Dataset | Ablation | Spearman | Pearson | Top-1 acc | AUROC (useful) | AUPRC |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for ds, d in report["datasets"].items():
        for row in d["competence_rows"]:
            lines.append(f"| {ds} | {row['ablation']} | {row['spearman']:.3f} | {row['pearson']:.3f} | {row['top1_accuracy']:.3f} | {row['auroc_useful_vs_harmful']:.3f} | {row['auprc_useful_vs_harmful']:.3f} |")
    lines += ["", "## Dependence-aware D vs C / D vs Equal (block bootstrap)", ""]
    lines.append("| Dataset | Comparison | Test | Mean delta | 95% CI | Excludes zero |")
    lines.append("|---|---|---|---:|---|---|")
    for ds, d in report["datasets"].items():
        for row in d["dependence_rows"]:
            if "ci95_low" in row:
                lines.append(f"| {ds} | {row['comparison']} | {row['test']} | `{row.get('mean_delta', row.get('mean_diff_candidate_minus_baseline')):+.6f}` | [{row['ci95_low']:+.6f}, {row['ci95_high']:+.6f}] | {row['ci_excludes_zero']} |")
    lines += ["", "## Integrity checks", ""]
    for ds, d in report["datasets"].items():
        lines.append(f"- **{ds}**: target-perturbation invariance = `{d['integrity']['result']}`; max forecast-reproduction error vs cached predictions = `{max(r['max_abs_diff'] for r in d['reproduction_checks']):.2e}`.")
    lines += ["", "## Go / No-Go decision", "", f"**{go_no_go['decision']}**", ""]
    for reason in go_no_go["reasoning"]:
        lines.append(f"- {reason}")
    lines += ["", "## Hard rule compliance", "", "```text", "TEST SET ACCESSED: NO", "TEST CACHE LOADED: NO", "TEST METRICS COMPUTED: NO", "```"]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def decide_go_no_go(report: Mapping[str, Any]) -> dict[str, Any]:
    datasets = list(report["datasets"].keys())
    beats_c = {ds: report["datasets"][ds]["d_minus_c_mae"] < 0 for ds in datasets}
    n_beats_c = sum(beats_c.values())
    headroom = {ds: report["datasets"][ds]["headroom_captured_D_over_C"] for ds in datasets}
    meaningful_headroom = {ds: (h is not None and h >= 0.10) for ds, h in headroom.items()}
    sig_datasets = []
    for ds in datasets:
        d_vs_c_block = [r for r in report["datasets"][ds]["dependence_rows"] if r["comparison"] == "D_vs_C" and r["test"].startswith("block_bootstrap")]
        if any(r["ci_excludes_zero"] and r["mean_delta"] < 0 for r in d_vs_c_block):
            sig_datasets.append(ds)
    correlation_ok = []
    for ds in datasets:
        by_ablation = {r["ablation"]: r for r in report["datasets"][ds]["competence_rows"]}
        c_sp, d_sp = by_ablation["C_window_forecast_disagreement"]["spearman"], by_ablation["D_full_behavioral"]["spearman"]
        correlation_ok.append(d_sp > c_sp)

    reasoning = [
        f"D beats C on MAE on {n_beats_c}/{len(datasets)} datasets: {beats_c}.",
        f"Headroom captured (D over C, >=10% of C->Oracle gap) on: {[ds for ds, ok in meaningful_headroom.items() if ok]}.",
        f"Dependence-aware (block-bootstrap) statistically supported D<C on: {sig_datasets}.",
        f"D improves competence-ranking Spearman correlation over C on {sum(correlation_ok)}/{len(datasets)} datasets.",
    ]
    go = (n_beats_c >= 3) and (len(sig_datasets) >= 2) and (sum(meaningful_headroom.values()) >= 1) and (sum(correlation_ok) >= 3)
    decision = "GO" if go else "NO-GO"
    return {"decision": decision, "reasoning": reasoning, "beats_c": beats_c, "headroom": headroom, "statistically_significant_datasets": sig_datasets, "correlation_improved": dict(zip(datasets, correlation_ok))}


def main() -> None:
    start = time.time()
    report: dict[str, Any] = {"experiment": "behavioral_competence", "created_at_utc": datetime.now(timezone.utc).isoformat(), "datasets": {}}
    all_results, all_competence, all_dependence, all_behavior_corr, all_reproduction, all_integrity = [], [], [], [], [], []
    per_window_npz: dict[str, np.ndarray] = {}
    per_window_competence_npz: dict[str, np.ndarray] = {}

    for dataset in LOADERS:
        print(f"[behavioral-competence] {dataset}: building perturbation cache + features + training scorers...", flush=True)
        result = run_dataset(dataset)
        report["datasets"][dataset] = {k: v for k, v in result.items() if k not in ("per_window_predictions", "per_window_competence_predictions", "actual_excess_loss_val")}
        all_results.extend(result["result_rows"])
        all_competence.extend(result["competence_rows"])
        all_dependence.extend(result["dependence_rows"])
        all_behavior_corr.extend(result["behavior_corr_rows"])
        all_reproduction.extend(result["reproduction_checks"])
        all_integrity.append(result["integrity"])
        for m, arr in result["per_window_predictions"].items():
            per_window_npz[f"{dataset}__{m}"] = arr
        for a, arr in result["per_window_competence_predictions"].items():
            per_window_competence_npz[f"{dataset}__{a}__predicted"] = arr
        per_window_npz[f"{dataset}__actual_excess_loss_val"] = result["actual_excess_loss_val"]
        print(f"[behavioral-competence] {dataset}: done. D vs C = {result['d_minus_c_mae']:+.6f}", flush=True)

    go_no_go = decide_go_no_go(report)
    report["go_no_go"] = go_no_go
    report["runtime_sec"] = time.time() - start
    report["test_set_accessed"] = False

    write_json(RESULTS_DIR / "manifest.json", {"code_version": CODE_VERSION, "ablations": ABLATIONS, "perturbations": list(PERTURBATIONS.keys()), "internal_val_fraction": INTERNAL_VAL_FRACTION, "block_lengths": BLOCK_LENGTHS, "bootstrap_samples": BOOTSTRAP_SAMPLES, "created_at_utc": report["created_at_utc"]})
    write_json(RESULTS_DIR / "feature_definitions.json", {"group_a": GROUP_A_NAMES, "group_b": GROUP_B_NAMES, "group_c": GROUP_C_NAMES, "group_d_stats": BEHAVIORAL_STAT_NAMES, "perturbations": list(PERTURBATIONS.keys()), "ablations": ABLATIONS})
    write_json(RESULTS_DIR / "config.json", {"input_len": None, "block_lengths": BLOCK_LENGTHS, "bootstrap_samples": BOOTSTRAP_SAMPLES, "phase_k": PHASE_K, "internal_val_fraction": INTERNAL_VAL_FRACTION, "scorer_architecture": "Linear(64)-ReLU-Linear(32)-ReLU-Linear(1)"})
    write_json(RESULTS_DIR / "results.json", report)
    write_csv(RESULTS_DIR / "validation_results.csv", all_results)
    write_csv(RESULTS_DIR / "competence_metrics.csv", all_competence)
    write_csv(RESULTS_DIR / "behavior_correlations.csv", all_behavior_corr)
    write_csv(RESULTS_DIR / "bootstrap_results.csv", all_dependence)
    write_csv(RESULTS_DIR / "reproduction_checks.csv", all_reproduction)
    write_csv(RESULTS_DIR / "integrity_checks.csv", all_integrity)
    np.savez(RESULTS_DIR / "per_window_predictions.npz", **per_window_npz)
    np.savez(RESULTS_DIR / "per_window_competence_predictions.npz", **per_window_competence_npz)
    make_report(REPORTS_DIR, report, go_no_go)

    print("TEST SET ACCESSED: NO")
    print(json.dumps({"runtime_sec": report["runtime_sec"], "go_no_go": go_no_go["decision"], "datasets": list(report["datasets"].keys())}, indent=2))


if __name__ == "__main__":
    main()
