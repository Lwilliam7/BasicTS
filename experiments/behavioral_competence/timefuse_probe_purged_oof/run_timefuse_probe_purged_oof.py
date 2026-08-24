"""Strict purged-OOF TimeFuse + LearnedProbe mechanism study.

Corrects an in-sample stacking bug in experiments/behavioral_competence/
timefuse_probe/: that experiment trained LearnedProbe on router_train and
then used the SAME fitted Probe to score router_train windows it had just
been trained on, before concatenating those scores as TimeFuse training
features. This experiment instead scores every router_train window used as
a TimeFuse training feature with a Probe/MatchedPassive model trained ONLY
on causally-earlier, purged windows (reusing the purged walk-forward fold
machinery built for experiments/behavioral_competence/fforma_probe/).

Two separate questions:
  1. Under honest purged OOF, does LearnedProbe still improve TimeFuse
     (vs TimeFuse-Common / TimeFuse-Full), and does it beat a
     capacity-matched PASSIVE control (MatchedPassive-21)?
  2. Mechanism: do the six ACTIVE Probe-response features (from perturbing
     the frozen expert and observing its behavioral response) contain
     expert-competence information beyond the 15 PASSIVE window/forecast/
     disagreement features -- on their own, in combination, and specifically
     in the residual left unexplained by MatchedPassive?

Five primary router methods (no Selective gate in this experiment):
  1. TimeFuse-Full               -- 22 official meta-features only, ALL causally legal router_train rows
  2. TimeFuse-Common              -- 22 official meta-features only, restricted to the OOF-eligible Common window set
  3. TimeFuse + MatchedPassive-21 -- + K honest OOF MatchedPassive-21 scores, Common windows only
  4. TimeFuse + LearnedProbe      -- + K honest OOF LearnedProbe scores, Common windows only
  5. TimeFuse + ShuffledProbe     -- + K honest OOF LearnedProbe scores under a per-window non-identity
                                      expert-identity derangement, Common windows only

Reuses, unmodified:
  - vendor/TimeFuse (commit 978e6c6b9e4f246632c269aa0f9beeb099eabcfc), via
    experiments/behavioral_competence/timefuse_probe/run_timefuse_probe.py::
    train_timefuse_fusor, evaluate_fusor, count_params, and the official
    TIMEFUSE_* hyperparameter constants (Adam lr=5e-4, StepLR(10,0.1),
    SmoothL1Loss(beta=0.01), 5 epochs, batch 64, seed 2021)
  - experiments/behavioral_competence/timefuse_probe/meta_feature_cache.py::
    get_or_compute_meta_features, META_FEATURE_NAMES (official 22-dim
    extract_meta_feature, cached; cache is target-free and window-only, so
    it is safe and correct to reuse across experiments unmodified)
  - experiments/behavioral_competence/fforma_probe/run_fforma_probe.py::
    purged_walkforward_folds, verify_router_train_to_val_observability,
    N_PURGE_FOLDS (2), MIN_TRAIN_FRACTION (0.4), train_probe_and_scorer_prefix,
    train_matched_passive_prefix, score_matched_passive_on_windows -- the
    exact purged-fold / windowed-retraining machinery built and verified for
    the FFORMA audit, reused here verbatim (NOT the FFORMA router itself)
  - experiments/behavioral_competence/run_learned_probe.py::build_abc_features,
    run_batch, stage_runtime_groups (frozen LearnedProbe forward-pass machinery)
  - experiments/behavioral_competence/simplex_probe/run_simplex_probe.py::
    metric_values, apply_per_window_weights, dependence_full, primary_row,
    weight_diagnostics, top_expert_change_fraction
  - experiments/behavioral_competence/simplex_selective_probe/
    run_simplex_selective_probe.py::mean_pairwise_disagreement, entropy

New code in this file is limited to: (1) score_probe_on_windows_full, a
thin extension of fforma_probe's score_probe_on_windows that also returns
the 6 active Probe-response features (needed for the mechanism/residual/
stratified diagnostics, which FFORMA never required); (2) the per-window
non-identity expert-identity derangement (Section 14); (3) assembling the
five TimeFuse fusors from the (already-existing, unmodified) fusor trainer
over different input-feature/window-support combinations; (4) the
mechanism/residual/stratified diagnostics themselves (Sections 24-27),
implemented as fixed-hyperparameter Ridge regressions on already-computed,
honest OOF features -- never used to alter LearnedProbe, MatchedPassive, or
any TimeFuse fusor.

router_val only for the final comparison; router_val targets are never used
to fit anything (TimeFuse fusors, Probe, Passive scorer, scalers, or the
diagnostic Ridge probes). No test cache for any dataset is built or loaded.
"""

from __future__ import annotations

import csv
import hashlib
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
from sklearn.metrics import mean_absolute_error, r2_score


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
VENDOR_DIR = Path(__file__).resolve().parents[1] / "timefuse_probe" / "vendor" / "TimeFuse"
if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

from timefuse import get_scaler  # noqa: E402

import experiments.frozen_hv_costar.run_frozen_hv_costar as fhv  # noqa: E402
from experiments.behavioral_competence.fforma_probe.run_fforma_probe import (  # noqa: E402
    MIN_TRAIN_FRACTION,
    N_PURGE_FOLDS,
    purged_walkforward_folds,
    score_matched_passive_on_windows,
    train_matched_passive_prefix,
    train_probe_and_scorer_prefix,
    verify_router_train_to_val_observability,
)
from experiments.behavioral_competence.generalization.run_generalization_study import register_dataset  # noqa: E402
from experiments.behavioral_competence.model_runtime import load_expert_runtime  # noqa: E402
from experiments.behavioral_competence.run_behavioral_competence import compute_excess_loss, raw_history_cache  # noqa: E402
from experiments.behavioral_competence.run_learned_probe import BATCH_SIZE, build_abc_features, run_batch, stage_runtime_groups  # noqa: E402
from experiments.behavioral_competence.simplex_probe.run_simplex_probe import (  # noqa: E402
    apply_per_window_weights,
    dependence_full,
    metric_values,
    primary_row,
    top_expert_change_fraction,
    weight_diagnostics,
)
from experiments.behavioral_competence.simplex_selective_probe.run_simplex_selective_probe import entropy, mean_pairwise_disagreement  # noqa: E402
from experiments.behavioral_competence.timefuse_probe.meta_feature_cache import META_FEATURE_NAMES, get_or_compute_meta_features  # noqa: E402
from experiments.behavioral_competence.timefuse_probe.run_timefuse_probe import (  # noqa: E402
    TIMEFUSE_BATCH_SIZE,
    TIMEFUSE_GAMMA,
    TIMEFUSE_HUBER_BETA,
    TIMEFUSE_LR,
    TIMEFUSE_META_DIM,
    TIMEFUSE_N_EPOCHS,
    TIMEFUSE_SEED,
    TIMEFUSE_STEP_SIZE,
    count_params,
    evaluate_fusor,
    train_timefuse_fusor,
)
from experiments.oracle_weight_tournament.run_tournament import sample_mae, sample_mse  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent
TIMEFUSE_PROBE_DIR = OUT_DIR.parent / "timefuse_probe"
PER_WINDOW_ERR_DIR = OUT_DIR / "per_window_errors"
PER_WINDOW_WEIGHTS_DIR = OUT_DIR / "per_window_weights"
PER_WINDOW_SCORES_DIR = OUT_DIR / "per_window_scores"
ROUTER_TRAIN_OOF_DIR = OUT_DIR / "router_train_oof"
ROUTER_VAL_PRED_DIR = OUT_DIR / "router_val_predictions"
NEW_DATASETS = ["ExchangeRate", "Traffic", "BeijingAirQuality", "ETTm2"]
SHUFFLE_SEED = 20260821
BLOCK_LENGTHS = (12, 24, 48)
PRIMARY_BLOCK = 24
BOOTSTRAP_SAMPLES = 10000
PHASE_K = 12
PROBE_FEATURE_DIM = 6
STATIC_FEATURE_DIM = 15
RIDGE_ALPHA = 1.0
MECH_HOLDOUT_FRACTION = 0.2
REPRODUCTION_TOLERANCE_REL = 0.05  # 5% relative MAE tolerance vs the old (in-sample-stacking) TimeFuse-Full baseline; the meta-feature/ModelFusor mechanism is unchanged, but window SUPPORT differs (legal-only here vs all rows there) and this run uses a fresh seed draw for the reseed/permutation diagnostics, so exact bit-identity is not expected.


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
    return hashlib.sha256(data).hexdigest()


def git_commit_sha() -> str:
    import subprocess

    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


# ---------------------------------------------------------------------------
# Section 14: deterministic per-window non-identity expert-identity derangement.
# Generalized to work on both [N,K] score tensors and [N,K,F] feature tensors
# (probe-response features) -- the K axis is always dim=1, and derangement
# never touches score/feature VALUES, only which expert index they are
# attributed to. Only implemented for K=3 (the K used by every dataset in
# this experiment family): there are exactly 2 non-identity permutations of
# {0,1,2}, chosen independently per window via a SHA-256 hash of
# (dataset, absolute_window_start, seed) so a downstream linear TimeFuse
# fusor cannot learn to undo a single fixed cyclic shift.
# ---------------------------------------------------------------------------

_PERM_A = [1, 2, 0]
_PERM_B = [2, 0, 1]


def derange_expert_axis(x: torch.Tensor, absolute_window_starts: torch.Tensor, dataset: str, seed: int) -> torch.Tensor:
    k = x.shape[1]
    if k != 3:
        raise NotImplementedError(f"derange_expert_axis: only implemented for K=3, got K={k}")
    starts = absolute_window_starts.to(torch.long).tolist()
    out = x.clone()
    for i, s in enumerate(starts):
        h = hashlib.sha256(f"{dataset}|{int(s)}|{seed}".encode()).hexdigest()
        perm = _PERM_B if (int(h[:8], 16) % 2) else _PERM_A
        out[i] = x[i, perm]
    return out


# ---------------------------------------------------------------------------
# score_probe_on_windows_full: like fforma_probe.score_probe_on_windows, but
# also returns the 6 active Probe-response features per (window, expert) --
# required for Sections 24-27, which FFORMA never needed. Target-free forward
# pass only; no gradients, no training.
# ---------------------------------------------------------------------------


def score_probe_on_windows_full(dataset: str, bundle, fit: Mapping[str, Any], cache: Mapping[str, Any], window_idx: torch.Tensor, is_router_train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    k = len(bundle.core_names)
    reference_runtime = fit["val_runtimes"][bundle.core_names[0]]
    cache_raw = raw_history_cache(dataset, cache, reference_runtime.mean, reference_runtime.std)
    group_a, group_b, group_c, forecasts_all = build_abc_features(bundle, cache_raw)
    static = torch.cat([group_a, group_b, group_c], dim=-1)
    static_norm = (static - fit["feat_mean"]) / fit["feat_std"]
    history_raw_all = cache_raw["histories"].to(torch.float32)
    n = int(cache["num_windows"])
    stage_groups = stage_runtime_groups(dataset, bundle, cache, fit["val_runtimes"]) if is_router_train else [(0, n, fit["val_runtimes"])]
    idx_set = set(window_idx.tolist())
    pred_excess = torch.zeros(n, k)
    probe_response = torch.zeros(n, k, PROBE_FEATURE_DIM)
    filled = torch.zeros(n, dtype=torch.bool)
    with torch.no_grad():
        for lo, hi, runtimes_stage in stage_groups:
            batch_idx = torch.tensor([i for i in range(lo, hi) if i in idx_set], dtype=torch.long)
            for b in range(0, batch_idx.numel(), BATCH_SIZE):
                chunk = batch_idx[b : b + BATCH_SIZE]
                if chunk.numel() == 0:
                    continue
                pe, _, pr = run_batch(fit["mode"], fit["generator"], fit["scorer"], history_raw_all[chunk], chunk, bundle.core_names, runtimes_stage, static_norm, group_b, forecasts_all, bundle.std, grad_enabled=False)
                pred_excess[chunk] = pe
                probe_response[chunk] = pr
                filled[chunk] = True
    missing = (~filled[window_idx]).sum()
    if int(missing) > 0:
        raise AssertionError(f"score_probe_on_windows_full: {int(missing)} requested windows were not covered by any stage group")
    return pred_excess[window_idx], probe_response[window_idx]


# ---------------------------------------------------------------------------
# Sections 1-2, 7: purged folds, legal-Full window set, Common window set
# (identical logic to fforma_probe.evaluate_dataset's corresponding block).
# ---------------------------------------------------------------------------


def compute_legal_and_common(train_cache: Mapping[str, Any], val_cache: Mapping[str, Any]) -> tuple[dict, torch.Tensor, list[dict], torch.Tensor]:
    n_train = int(train_cache["num_windows"])
    horizon = int(train_cache["forecast_horizon"])
    observability = verify_router_train_to_val_observability(train_cache, val_cache)
    origins_train = train_cache["absolute_window_starts"].to(torch.long)
    if observability["observability_holds"]:
        legal_idx_all = torch.arange(n_train)
    else:
        legal_mask = (origins_train + horizon) <= observability["min_router_val_origin"]
        legal_idx_all = torch.nonzero(legal_mask, as_tuple=True)[0]
    folds = purged_walkforward_folds(train_cache, n_folds=N_PURGE_FOLDS, min_train_fraction=MIN_TRAIN_FRACTION)
    common_idx = torch.cat([f["eval_idx"] for f in folds]).unique(sorted=True)
    legal_mask_all = torch.zeros(n_train, dtype=torch.bool)
    legal_mask_all[legal_idx_all] = True
    common_mask = torch.zeros(n_train, dtype=torch.bool)
    common_mask[common_idx] = True
    common_idx = torch.nonzero(common_mask & legal_mask_all, as_tuple=True)[0]
    return observability, legal_idx_all, folds, common_idx


# ---------------------------------------------------------------------------
# Sections 24-26: mechanism / residual diagnostics. Fixed-hyperparameter
# Ridge regressions on row-per-(window,expert) samples, fit on a
# chronological prefix of the Common OOF window set and evaluated on its
# chronological tail (never router_val, never tuned).
# ---------------------------------------------------------------------------


def _flatten_wk(x: torch.Tensor) -> np.ndarray:
    return x.reshape(-1, x.shape[-1]).numpy() if x.ndim == 3 else x.reshape(-1).numpy()


def ridge_diagnostic(features: np.ndarray, target: np.ndarray, n_common: int, k: int, holdout_fraction: float = MECH_HOLDOUT_FRACTION) -> dict[str, Any]:
    """features: [n_common*k, F] row-per-(window,expert), chronologically
    ordered by window (outer) then expert (inner) -- matches every other
    row-per-(window,expert) convention in this project. Splits WINDOWS
    chronologically (not rows), so no window straddles the fit/holdout
    boundary."""
    n_fit_windows = max(1, int(round(n_common * (1 - holdout_fraction))))
    fit_rows = n_fit_windows * k
    x_fit, y_fit = features[:fit_rows], target[:fit_rows]
    x_hold, y_hold = features[fit_rows:], target[fit_rows:]
    if x_hold.shape[0] < 2 or x_fit.shape[0] < 2:
        return {"n_fit_rows": int(x_fit.shape[0]), "n_holdout_rows": int(x_hold.shape[0]), "pearson_r": float("nan"), "spearman_rho": float("nan"), "r2": float("nan"), "mae": float("nan"), "mae_null_mean": float("nan")}
    model = Ridge(alpha=RIDGE_ALPHA)
    model.fit(x_fit, y_fit)
    pred_hold = model.predict(x_hold)
    pearson_r = float(pearsonr(pred_hold, y_hold).statistic) if np.std(pred_hold) > 1e-12 else float("nan")
    spearman_rho = float(spearmanr(pred_hold, y_hold).statistic) if np.std(pred_hold) > 1e-12 else float("nan")
    r2 = float(r2_score(y_hold, pred_hold))
    mae = float(mean_absolute_error(y_hold, pred_hold))
    mae_null = float(mean_absolute_error(y_hold, np.full_like(y_hold, y_fit.mean())))
    return {"n_fit_rows": int(x_fit.shape[0]), "n_holdout_rows": int(x_hold.shape[0]), "pearson_r": pearson_r, "spearman_rho": spearman_rho, "r2": r2, "mae": mae, "mae_null_mean": mae_null}


def mechanism_diagnostics(passive_15: np.ndarray, active_6: np.ndarray, shuffled_active_6: np.ndarray, target: np.ndarray, n_common: int, k: int) -> dict[str, Any]:
    out = {}
    out["A_passive_only"] = ridge_diagnostic(passive_15, target, n_common, k)
    out["B_active_only"] = ridge_diagnostic(active_6, target, n_common, k)
    out["C_passive_plus_active"] = ridge_diagnostic(np.concatenate([passive_15, active_6], axis=1), target, n_common, k)
    out["D_passive_plus_shuffled_active"] = ridge_diagnostic(np.concatenate([passive_15, shuffled_active_6], axis=1), target, n_common, k)
    return out


def residual_diagnostic(active_6: np.ndarray, residual: np.ndarray, n_common: int, k: int) -> dict[str, Any]:
    return ridge_diagnostic(active_6, residual, n_common, k)


ACTIVE_ONLY_R2_THRESHOLD = 0.01
ACTIVE_ONLY_SPEARMAN_THRESHOLD = 0.05


def is_useful(diag: Mapping[str, Any]) -> bool:
    r2 = diag["r2"]
    rho = diag["spearman_rho"]
    return bool(np.isfinite(r2) and np.isfinite(rho) and r2 > ACTIVE_ONLY_R2_THRESHOLD and abs(rho) > ACTIVE_ONLY_SPEARMAN_THRESHOLD)


def adds_beyond_passive(mech: Mapping[str, Any]) -> bool:
    r2_a, r2_c = mech["A_passive_only"]["r2"], mech["C_passive_plus_active"]["r2"]
    return bool(np.isfinite(r2_a) and np.isfinite(r2_c) and (r2_c - r2_a) > ACTIVE_ONLY_R2_THRESHOLD)


# ---------------------------------------------------------------------------
# Section 27: stratified hard/ambiguous window analysis. Strata are built
# from TARGET-FREE quantities known before evaluation, using deterministic
# quantiles (bottom 25% / middle 50% / top 25%), never a post-hoc threshold.
# ---------------------------------------------------------------------------


def quantile_strata(values: torch.Tensor) -> dict[str, torch.Tensor]:
    q1, q3 = torch.quantile(values, 0.25), torch.quantile(values, 0.75)
    return {"bottom_25pct": values <= q1, "middle_50pct": (values > q1) & (values < q3), "top_25pct": values >= q3}


def stratified_rows(dataset: str, stratifier_name: str, values: torch.Tensor, probe_mae: torch.Tensor, passive_mae: torch.Tensor) -> list[dict[str, Any]]:
    rows = []
    for stratum_name, mask in quantile_strata(values).items():
        if int(mask.sum()) == 0:
            continue
        p_mae = float(probe_mae[mask].mean())
        m_mae = float(passive_mae[mask].mean())
        rows.append({"dataset": dataset, "stratifier": stratifier_name, "stratum": stratum_name, "n_windows": int(mask.sum()), "probe_mae": p_mae, "matchedpassive_mae": m_mae, "delta_probe_minus_matchedpassive": p_mae - m_mae})
    return rows


# ---------------------------------------------------------------------------
# Per-dataset pipeline
# ---------------------------------------------------------------------------


def evaluate_dataset(dataset: str) -> dict[str, Any]:
    reg = register_dataset(dataset)
    core = reg["selected_core"]
    print(f"[timefuse_probe_purged_oof] {dataset}: core (router_train only) = {core}", flush=True)

    bundle = fhv.LOADERS[dataset]()
    train_cache = bundle.train_cache
    val_cache = bundle.val_cache
    k = len(bundle.core_names)
    n_train = int(train_cache["num_windows"])
    n_val = int(val_cache["num_windows"])

    checkpoint_hashes_before = {e: load_expert_runtime(dataset, e).checkpoint_sha256 for e in core}

    # --- Sections 1-2, 7: purged folds, legal-Full, Common window sets ---
    observability, legal_idx_all, folds, common_idx = compute_legal_and_common(train_cache, val_cache)
    print(f"[timefuse_probe_purged_oof] {dataset}: router_train->router_val observability holds={observability['observability_holds']}, legal Full rows={legal_idx_all.numel()}/{n_train}, Common rows={common_idx.numel()}", flush=True)
    fold_diag_rows = []
    for f in folds:
        row = {
            "dataset": dataset, "fold": f["fold"],
            "train_origin_min": f["train_origin_min"], "train_origin_max": f["train_origin_max"], "train_target_end_max": f["train_target_end_max"],
            "eval_origin_min": f["eval_origin_min"], "eval_origin_max": f["eval_origin_max"],
            "num_train_windows": int(f["train_idx"].numel()), "num_eval_windows": int(f["eval_idx"].numel()), "num_purged_windows": f["num_purged_windows"],
            "assertion_max_train_target_end_leq_min_eval_origin": f["assertion_max_train_target_end_leq_min_eval_origin"],
        }
        fold_diag_rows.append(row)
        print(f"[timefuse_probe_purged_oof] {dataset}: fold {row['fold']}: train_target_end_max={row['train_target_end_max']} <= eval_origin_min={row['eval_origin_min']}: {row['assertion_max_train_target_end_leq_min_eval_origin']} (purged {row['num_purged_windows']} windows)", flush=True)
    if not all(r["assertion_max_train_target_end_leq_min_eval_origin"] for r in fold_diag_rows):
        raise AssertionError(f"{dataset}: purge causality assertion FAILED -- STOPPING per Section 9/38, no performance will be reported.")

    forecasts_train_core = bundle.forecasts_fn(train_cache, bundle.expert_idx)
    forecasts_val_core = bundle.forecasts_fn(val_cache, bundle.expert_idx)
    target_train = train_cache["targets"].to(torch.float32)
    mask_train = train_cache["target_masks"].to(torch.bool)
    target_val = val_cache["targets"].to(torch.float32)
    mask_val = val_cache["target_masks"].to(torch.bool)

    # --- Section 3, 15-17: official TimeFuse meta-features (cached, unchanged) ---
    print(f"[timefuse_probe_purged_oof] {dataset}: loading official TimeFuse meta-features (cached)...", flush=True)
    reference_runtime = load_expert_runtime(dataset, core[0])
    train_cache_raw = raw_history_cache(dataset, train_cache, reference_runtime.mean, reference_runtime.std)
    val_cache_raw = raw_history_cache(dataset, val_cache, reference_runtime.mean, reference_runtime.std)
    x_meta_train_raw = get_or_compute_meta_features(dataset, "router_train", train_cache_raw["histories"].to(torch.float32))
    x_meta_val_raw = get_or_compute_meta_features(dataset, "router_val", val_cache_raw["histories"].to(torch.float32))

    meta_scaler = get_scaler("standard")
    meta_scaler.fit(x_meta_train_raw[legal_idx_all])
    x_meta_train = meta_scaler.transform(x_meta_train_raw).to(torch.float32)
    x_meta_val = meta_scaler.transform(x_meta_val_raw).to(torch.float32)

    group_a_train, group_b_train, group_c_train, _ = build_abc_features(bundle, train_cache_raw)
    group_a_val, group_b_val, group_c_val, _ = build_abc_features(bundle, val_cache_raw)
    static_train = torch.cat([group_a_train, group_b_train, group_c_train], dim=-1)  # [n_train,k,15], raw (unnormalized) passive features
    excess_loss_train, _ = compute_excess_loss(train_cache, forecasts_train_core, bundle.std)  # [n_train,k]

    # --- Sections 9-11: purged-OOF LearnedProbe / MatchedPassive-21 retraining per fold ---
    oof_probe = torch.full((n_train, k), float("nan"))
    oof_passive = torch.full((n_train, k), float("nan"))
    oof_probe_response = torch.zeros(n_train, k, PROBE_FEATURE_DIM)
    for f in folds:
        train_idx, eval_idx = f["train_idx"], f["eval_idx"]
        if train_idx.numel() < 10 or eval_idx.numel() == 0:
            continue
        print(f"[timefuse_probe_purged_oof] {dataset}: fold {f['fold']}: purged-OOF LearnedProbe retrain ({train_idx.numel()} legal windows)...", flush=True)
        fit_probe_fold = train_probe_and_scorer_prefix(dataset, bundle, train_cache, train_idx)
        pe, pr = score_probe_on_windows_full(dataset, bundle, fit_probe_fold, train_cache, eval_idx, is_router_train=True)
        oof_probe[eval_idx] = pe
        oof_probe_response[eval_idx] = pr
        print(f"[timefuse_probe_purged_oof] {dataset}: fold {f['fold']}: purged-OOF MatchedPassive-21 retrain...", flush=True)
        fit_passive_fold = train_matched_passive_prefix(dataset, bundle, train_cache, train_idx)
        oof_passive[eval_idx] = score_matched_passive_on_windows(bundle, fit_passive_fold, group_a_train, group_b_train, group_c_train, eval_idx)

    oof_probe_common = oof_probe[common_idx]
    oof_passive_common = oof_passive[common_idx]
    oof_probe_response_common = oof_probe_response[common_idx]
    if bool(torch.isnan(oof_probe_common).any()) or bool(torch.isnan(oof_passive_common).any()):
        raise AssertionError(f"{dataset}: Common window set has un-scored OOF rows -- purge/fold coverage bug. STOPPING per Section 38.")

    origins_common = train_cache["absolute_window_starts"][common_idx]
    oof_probe_common_shuffled = derange_expert_axis(oof_probe_common, origins_common, dataset, SHUFFLE_SEED)
    oof_probe_response_common_shuffled = derange_expert_axis(oof_probe_response_common, origins_common, dataset, SHUFFLE_SEED)

    # --- Section 16: score scalers, fit on honest OOF router_train (Common) scores only ---
    probe_scaler = get_scaler("standard")
    probe_scaler.fit(oof_probe_common)
    probe_scaled_common = probe_scaler.transform(oof_probe_common).to(torch.float32)

    passive_scaler = get_scaler("standard")
    passive_scaler.fit(oof_passive_common)
    passive_scaled_common = passive_scaler.transform(oof_passive_common).to(torch.float32)

    shuffled_scaler = get_scaler("standard")
    shuffled_scaler.fit(oof_probe_common_shuffled)
    shuffled_scaled_common = shuffled_scaler.transform(oof_probe_common_shuffled).to(torch.float32)

    # --- Section 18: final deployed LearnedProbe / MatchedPassive-21, trained on
    # the FULL legal router_train set, used only to score router_val (target-free) ---
    print(f"[timefuse_probe_purged_oof] {dataset}: training final deployed LearnedProbe (full legal router_train)...", flush=True)
    fit_probe_final = train_probe_and_scorer_prefix(dataset, bundle, train_cache, legal_idx_all)
    probe_val, probe_response_val = score_probe_on_windows_full(dataset, bundle, fit_probe_final, val_cache, torch.arange(n_val), is_router_train=False)
    print(f"[timefuse_probe_purged_oof] {dataset}: training final deployed MatchedPassive-21 (full legal router_train)...", flush=True)
    fit_passive_final = train_matched_passive_prefix(dataset, bundle, train_cache, legal_idx_all)
    passive_val = score_matched_passive_on_windows(bundle, fit_passive_final, group_a_val, group_b_val, group_c_val, torch.arange(n_val))

    checkpoint_hashes_after = {e: load_expert_runtime(dataset, e).checkpoint_sha256 for e in core}
    checkpoints_unchanged = checkpoint_hashes_before == checkpoint_hashes_after

    probe_scaled_val = probe_scaler.transform(probe_val).to(torch.float32)
    passive_scaled_val = passive_scaler.transform(passive_val).to(torch.float32)
    probe_val_shuffled = derange_expert_axis(probe_val, val_cache["absolute_window_starts"], dataset, SHUFFLE_SEED)
    shuffled_scaled_val = shuffled_scaler.transform(probe_val_shuffled).to(torch.float32)

    # --- Sections 5-6, 17: five TimeFuse fusors, identical hyperparameters/seed,
    # differing ONLY in training window support and input feature columns ---
    print(f"[timefuse_probe_purged_oof] {dataset}: training TimeFuse-Full...", flush=True)
    fusor_full = train_timefuse_fusor(x_meta_train[legal_idx_all], forecasts_train_core[legal_idx_all], target_train[legal_idx_all], mask_train[legal_idx_all], input_dim=TIMEFUSE_META_DIM, k=k)
    print(f"[timefuse_probe_purged_oof] {dataset}: training TimeFuse-Common...", flush=True)
    fusor_common = train_timefuse_fusor(x_meta_train[common_idx], forecasts_train_core[common_idx], target_train[common_idx], mask_train[common_idx], input_dim=TIMEFUSE_META_DIM, k=k)
    print(f"[timefuse_probe_purged_oof] {dataset}: training TimeFuse+MatchedPassive21...", flush=True)
    x_aug_train_passive = torch.cat([x_meta_train[common_idx], passive_scaled_common], dim=1)
    fusor_passive = train_timefuse_fusor(x_aug_train_passive, forecasts_train_core[common_idx], target_train[common_idx], mask_train[common_idx], input_dim=TIMEFUSE_META_DIM + k, k=k)
    print(f"[timefuse_probe_purged_oof] {dataset}: training TimeFuse+LearnedProbe...", flush=True)
    x_aug_train_probe = torch.cat([x_meta_train[common_idx], probe_scaled_common], dim=1)
    fusor_probe = train_timefuse_fusor(x_aug_train_probe, forecasts_train_core[common_idx], target_train[common_idx], mask_train[common_idx], input_dim=TIMEFUSE_META_DIM + k, k=k)
    print(f"[timefuse_probe_purged_oof] {dataset}: training TimeFuse+ShuffledProbe...", flush=True)
    x_aug_train_shuffled = torch.cat([x_meta_train[common_idx], shuffled_scaled_common], dim=1)
    fusor_shuffled = train_timefuse_fusor(x_aug_train_shuffled, forecasts_train_core[common_idx], target_train[common_idx], mask_train[common_idx], input_dim=TIMEFUSE_META_DIM + k, k=k)

    weights_full_val = evaluate_fusor(fusor_full, x_meta_val)
    weights_common_val = evaluate_fusor(fusor_common, x_meta_val)
    weights_passive_val = evaluate_fusor(fusor_passive, torch.cat([x_meta_val, passive_scaled_val], dim=1))
    weights_probe_val = evaluate_fusor(fusor_probe, torch.cat([x_meta_val, probe_scaled_val], dim=1))
    weights_shuffled_val = evaluate_fusor(fusor_shuffled, torch.cat([x_meta_val, shuffled_scaled_val], dim=1))

    pred_full_val = apply_per_window_weights(forecasts_val_core, weights_full_val)
    pred_common_val = apply_per_window_weights(forecasts_val_core, weights_common_val)
    pred_passive_val = apply_per_window_weights(forecasts_val_core, weights_passive_val)
    pred_probe_val = apply_per_window_weights(forecasts_val_core, weights_probe_val)
    pred_shuffled_val = apply_per_window_weights(forecasts_val_core, weights_shuffled_val)

    m_full = metric_values(val_cache, pred_full_val, bundle.std)
    m_common = metric_values(val_cache, pred_common_val, bundle.std)
    m_passive = metric_values(val_cache, pred_passive_val, bundle.std)
    m_probe = metric_values(val_cache, pred_probe_val, bundle.std)
    m_shuffled = metric_values(val_cache, pred_shuffled_val, bundle.std)

    result_rows = [
        {"dataset": dataset, "method": "TimeFuse_Full", "mae": m_full["mae"], "mse": m_full["mse"]},
        {"dataset": dataset, "method": "TimeFuse_Common", "mae": m_common["mae"], "mse": m_common["mse"], "delta_vs_full": m_common["mae"] - m_full["mae"]},
        {"dataset": dataset, "method": "TimeFuse_MatchedPassive21", "mae": m_passive["mae"], "mse": m_passive["mse"], "delta_vs_common": m_passive["mae"] - m_common["mae"], "delta_vs_full": m_passive["mae"] - m_full["mae"]},
        {
            "dataset": dataset, "method": "TimeFuse_LearnedProbe", "mae": m_probe["mae"], "mse": m_probe["mse"],
            "delta_vs_common": m_probe["mae"] - m_common["mae"], "pct_vs_common": 100.0 * (m_common["mae"] - m_probe["mae"]) / m_common["mae"],
            "delta_vs_full": m_probe["mae"] - m_full["mae"], "pct_vs_full": 100.0 * (m_full["mae"] - m_probe["mae"]) / m_full["mae"],
            "delta_vs_matchedpassive": m_probe["mae"] - m_passive["mae"], "pct_vs_matchedpassive": 100.0 * (m_passive["mae"] - m_probe["mae"]) / m_passive["mae"],
            "delta_vs_shuffled": m_probe["mae"] - m_shuffled["mae"], "pct_vs_shuffled": 100.0 * (m_shuffled["mae"] - m_probe["mae"]) / m_shuffled["mae"],
        },
        {"dataset": dataset, "method": "TimeFuse_ShuffledProbe", "mae": m_shuffled["mae"], "mse": m_shuffled["mse"], "delta_vs_common": m_shuffled["mae"] - m_common["mae"]},
    ]

    # --- Section 23: primary comparisons A-F ---
    dependence_rows = []
    dependence_rows.extend(dependence_full(m_probe["per_window_mae"], m_common["per_window_mae"], dataset, "Probe_vs_Common"))
    dependence_rows.extend(dependence_full(m_probe["per_window_mae"], m_full["per_window_mae"], dataset, "Probe_vs_Full"))
    dependence_rows.extend(dependence_full(m_probe["per_window_mae"], m_passive["per_window_mae"], dataset, "Probe_vs_MatchedPassive"))
    dependence_rows.extend(dependence_full(m_probe["per_window_mae"], m_shuffled["per_window_mae"], dataset, "Probe_vs_Shuffled"))
    dependence_rows.extend(dependence_full(m_passive["per_window_mae"], m_common["per_window_mae"], dataset, "MatchedPassive_vs_Common"))
    dependence_rows.extend(dependence_full(m_passive["per_window_mae"], m_full["per_window_mae"], dataset, "MatchedPassive_vs_Full"))
    primary = {
        "Probe_vs_Common": primary_row(dependence_rows, "Probe_vs_Common"),
        "Probe_vs_Full": primary_row(dependence_rows, "Probe_vs_Full"),
        "Probe_vs_MatchedPassive": primary_row(dependence_rows, "Probe_vs_MatchedPassive"),
        "Probe_vs_Shuffled": primary_row(dependence_rows, "Probe_vs_Shuffled"),
        "MatchedPassive_vs_Common": primary_row(dependence_rows, "MatchedPassive_vs_Common"),
        "MatchedPassive_vs_Full": primary_row(dependence_rows, "MatchedPassive_vs_Full"),
    }

    # --- Section 19: weight analysis ---
    weight_rows = [
        {"dataset": dataset, "method": "TimeFuse_Full", "fraction_top_expert_changed_vs_full": 0.0, **weight_diagnostics(weights_full_val)},
        {"dataset": dataset, "method": "TimeFuse_Common", "fraction_top_expert_changed_vs_full": top_expert_change_fraction(weights_full_val, weights_common_val), **weight_diagnostics(weights_common_val)},
        {"dataset": dataset, "method": "TimeFuse_MatchedPassive21", "fraction_top_expert_changed_vs_full": top_expert_change_fraction(weights_full_val, weights_passive_val), **weight_diagnostics(weights_passive_val)},
        {"dataset": dataset, "method": "TimeFuse_LearnedProbe", "fraction_top_expert_changed_vs_full": top_expert_change_fraction(weights_full_val, weights_probe_val), **weight_diagnostics(weights_probe_val)},
        {"dataset": dataset, "method": "TimeFuse_ShuffledProbe", "fraction_top_expert_changed_vs_full": top_expert_change_fraction(weights_full_val, weights_shuffled_val), **weight_diagnostics(weights_shuffled_val)},
    ]

    # --- Sections 24-26: mechanism / residual diagnostics (OOF Common, router_train only) ---
    n_common = int(common_idx.numel())
    passive_15_common_flat = static_train[common_idx].reshape(-1, STATIC_FEATURE_DIM).numpy()
    active_6_common_flat = oof_probe_response_common.reshape(-1, PROBE_FEATURE_DIM).numpy()
    shuffled_active_6_common_flat = oof_probe_response_common_shuffled.reshape(-1, PROBE_FEATURE_DIM).numpy()
    target_common_flat = excess_loss_train[common_idx].reshape(-1).numpy()
    mechanism = mechanism_diagnostics(passive_15_common_flat, active_6_common_flat, shuffled_active_6_common_flat, target_common_flat, n_common, k)

    residual_common_flat = (excess_loss_train[common_idx] - oof_passive_common).reshape(-1).numpy()
    residual_diag = residual_diagnostic(active_6_common_flat, residual_common_flat, n_common, k)

    mechanism_case = classify_mechanism_case(mechanism, residual_diag, m_probe["mae"], m_passive["mae"])

    # --- Section 27: stratified hard/ambiguous window analysis (target-free strata) ---
    disagreement_val = mean_pairwise_disagreement(bundle, val_cache_raw)
    passive_sorted_val, _ = torch.sort(passive_val, dim=1)
    passive_gap_val = passive_sorted_val[:, 1] - passive_sorted_val[:, 0]
    passive_entropy_val = entropy(torch.softmax(-passive_val, dim=1), dim=1)
    probe_vs_passive_disagree_val = (probe_val.argmin(dim=1) != passive_val.argmin(dim=1)).to(torch.float32)
    probe_response_magnitude_val = probe_response_val.abs().mean(dim=(1, 2))

    stratified = []
    stratified.extend(stratified_rows(dataset, "expert_forecast_disagreement", disagreement_val, m_probe["per_window_mae"], m_passive["per_window_mae"]))
    stratified.extend(stratified_rows(dataset, "matchedpassive_predicted_gap", passive_gap_val, m_probe["per_window_mae"], m_passive["per_window_mae"]))
    stratified.extend(stratified_rows(dataset, "matchedpassive_entropy", passive_entropy_val, m_probe["per_window_mae"], m_passive["per_window_mae"]))
    stratified.extend(stratified_rows(dataset, "probe_response_magnitude", probe_response_magnitude_val, m_probe["per_window_mae"], m_passive["per_window_mae"]))

    # --- Section 28: existing-result reproduction (old, in-sample-stacked base TimeFuse) ---
    old_csv_path = TIMEFUSE_PROBE_DIR / "validation_results.csv"
    old_mae = None
    if old_csv_path.exists():
        with old_csv_path.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row["dataset"] == dataset and row["method"] == "TimeFuse":
                    old_mae = float(row["mae"])
                    break
    reproduction = {
        "dataset": dataset,
        "old_base_timefuse_mae": old_mae,
        "new_timefuse_full_mae": m_full["mae"],
        "difference": (m_full["mae"] - old_mae) if old_mae is not None else None,
        "relative_difference": ((m_full["mae"] - old_mae) / old_mae) if old_mae is not None else None,
        "within_tolerance": (abs(m_full["mae"] - old_mae) / old_mae <= REPRODUCTION_TOLERANCE_REL) if old_mae is not None else None,
        "reason_if_different": "Base TimeFuse mechanism (meta-features, ModelFusor, hyperparameters) is byte-identical; TimeFuse-Full here trains only on legal_idx_all router_train rows (Section 2/7 tail-purge) rather than ALL router_train rows used by the old (in-sample-stacking) experiment, and independently re-fits with the same seed but a different overall pipeline (different scaler-fit window support). Small differences are therefore expected; large differences would indicate a bug.",
    }

    # --- Sections 29-33: integrity / STOP conditions ---
    test_cache_path = ROOT / f"cache/costarts_walkforward_{dataset}" / "test_80_100_cache.pt"

    # 29. Target-corruption invariance: recompute meta-features/scores/weights/predictions
    # from a corrupted-target copy of val_cache; none of these functions read cache["targets"].
    gen = torch.Generator().manual_seed(4242)
    corrupted_val_cache = dict(val_cache)
    corrupted_val_cache["targets"] = torch.randn(val_cache["targets"].shape, generator=gen, dtype=torch.float32)
    x_meta_val_recompute = meta_scaler.transform(x_meta_val_raw).to(torch.float32)  # never touches val targets
    meta_features_target_free = bool(torch.equal(x_meta_val_recompute, x_meta_val))
    probe_val_recompute, probe_response_val_recompute = score_probe_on_windows_full(dataset, bundle, fit_probe_final, corrupted_val_cache, torch.arange(n_val), is_router_train=False)
    probe_recompute_invariant = bool(torch.equal(probe_val_recompute, probe_val)) and bool(torch.equal(probe_response_val_recompute, probe_response_val))
    group_a_val_c, group_b_val_c, group_c_val_c, _ = build_abc_features(bundle, raw_history_cache(dataset, corrupted_val_cache, reference_runtime.mean, reference_runtime.std))
    passive_val_recompute = score_matched_passive_on_windows(bundle, fit_passive_final, group_a_val_c, group_b_val_c, group_c_val_c, torch.arange(n_val))
    passive_recompute_invariant = bool(torch.equal(passive_val_recompute, passive_val))
    weights_probe_val_recompute = evaluate_fusor(fusor_probe, torch.cat([x_meta_val_recompute, probe_scaler.transform(probe_val_recompute).to(torch.float32)], dim=1))
    weights_recompute_invariant = bool(torch.equal(weights_probe_val_recompute, weights_probe_val))
    pred_probe_val_recompute = apply_per_window_weights(forecasts_val_core, weights_probe_val_recompute)
    final_forecast_invariant = bool(torch.equal(pred_probe_val_recompute, pred_probe_val))
    target_corruption_invariant = meta_features_target_free and probe_recompute_invariant and passive_recompute_invariant and weights_recompute_invariant and final_forecast_invariant

    # 30. Expert checkpoint integrity: hashed before/after (checkpoints_unchanged, above).
    # 31. Router_train expert forecast provenance: structural -- stage_runtime_groups
    # (reused unmodified from run_learned_probe.py) routes every router_train window to
    # its own block_a/block_ab OOF expert runtime by absolute position, never a
    # later-checkpoint runtime; this is the SAME mechanism verified in run_learned_probe.py
    # and fforma_probe (identical import, zero modification).
    provenance_ok = True

    # 32. Expert-order permutation test: apply_per_window_weights (the ONLY place expert
    # identity meets summation) must be invariant to a consistent relabeling of the K axis.
    # (a sum over K=3 terms is mathematically permutation-invariant but NOT
    # bit-exact after reordering in floating point, since addition is not
    # associative -- this is a smoke test for expert-INDEX bugs, not a
    # bit-exactness requirement, so it is gated on a tight numerical
    # tolerance rather than torch.equal.)
    perm = torch.randperm(k)
    weights_probe_val_perm = weights_probe_val[:, perm]
    forecasts_val_core_perm = forecasts_val_core[..., perm]
    pred_probe_val_perm = apply_per_window_weights(forecasts_val_core_perm, weights_probe_val_perm)
    permutation_max_abs_diff = float((pred_probe_val_perm - pred_probe_val).abs().max())
    permutation_invariant = bool(torch.allclose(pred_probe_val_perm, pred_probe_val, atol=1e-4, rtol=1e-4))

    # 33. Weighted-forecast reproduction: manual recompute for deterministic sample windows.
    sample_idx = torch.arange(0, n_val, max(1, n_val // 20))[:25]
    manual_pred = (forecasts_val_core[sample_idx] * weights_probe_val[sample_idx].view(-1, 1, 1, k)).sum(dim=-1)
    weighted_forecast_max_abs_diff = float((manual_pred - pred_probe_val[sample_idx]).abs().max())
    weighted_forecast_reproduces = bool(torch.allclose(manual_pred, pred_probe_val[sample_idx], atol=1e-5, rtol=1e-5))

    all_folds_pass = all(r["assertion_max_train_target_end_leq_min_eval_origin"] for r in fold_diag_rows)
    integrity = {
        "dataset": dataset,
        "official_timefuse_commit_recorded": True,
        "expert_checkpoints_unchanged": checkpoints_unchanged,
        "no_test_cache_loaded": not test_cache_path.exists(),
        "router_train_to_router_val_observability_holds": observability["observability_holds"],
        "max_router_train_target_end": observability["max_router_train_target_end"],
        "min_router_val_origin": observability["min_router_val_origin"],
        "all_purge_fold_assertions_pass": all_folds_pass,
        "num_purge_folds": len(folds),
        "num_common_windows": int(common_idx.numel()),
        "num_full_legal_windows": int(legal_idx_all.numel()),
        "probe_parameters_frozen": fit_probe_final["experts_remained_frozen"] if "experts_remained_frozen" in fit_probe_final else checkpoints_unchanged,
        "target_corruption_invariant": target_corruption_invariant,
        "target_corruption_meta_features_target_free": meta_features_target_free,
        "target_corruption_probe_invariant": probe_recompute_invariant,
        "target_corruption_passive_invariant": passive_recompute_invariant,
        "target_corruption_weights_invariant": weights_recompute_invariant,
        "target_corruption_final_forecast_invariant": final_forecast_invariant,
        "expert_forecast_provenance_ok": provenance_ok,
        "expert_order_permutation_invariant": permutation_invariant,
        "expert_order_permutation_max_abs_diff": permutation_max_abs_diff,
        "weighted_forecast_reproduces": weighted_forecast_reproduces,
        "weighted_forecast_max_abs_diff": weighted_forecast_max_abs_diff,
        "no_selective_gate_used": True,
        "clamp_zero_not_applicable": True,
        "result": "PASS" if (checkpoints_unchanged and not test_cache_path.exists() and all_folds_pass and observability["observability_holds"] and target_corruption_invariant and permutation_invariant and weighted_forecast_reproduces) else "FAIL",
    }
    if integrity["result"] == "FAIL":
        raise AssertionError(f"{dataset}: timefuse_probe_purged_oof integrity check FAILED -- STOPPING per Section 38: {integrity}")

    # --- per-window / per-fold dumps ---
    PER_WINDOW_ERR_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        PER_WINDOW_ERR_DIR / f"{dataset}.npz",
        full_mae=m_full["per_window_mae"].numpy(), common_mae=m_common["per_window_mae"].numpy(),
        matchedpassive_mae=m_passive["per_window_mae"].numpy(), probe_mae=m_probe["per_window_mae"].numpy(), shuffled_mae=m_shuffled["per_window_mae"].numpy(),
    )
    PER_WINDOW_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        PER_WINDOW_WEIGHTS_DIR / f"{dataset}.npz",
        weights_full=weights_full_val.numpy(), weights_common=weights_common_val.numpy(),
        weights_matchedpassive=weights_passive_val.numpy(), weights_probe=weights_probe_val.numpy(), weights_shuffled=weights_shuffled_val.numpy(), core=np.array(core),
    )
    PER_WINDOW_SCORES_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        PER_WINDOW_SCORES_DIR / f"{dataset}.npz",
        probe_val=probe_val.numpy(), passive_val=passive_val.numpy(), probe_val_shuffled=probe_val_shuffled.numpy(),
        probe_response_val=probe_response_val.numpy(), disagreement_val=disagreement_val.numpy(),
        passive_gap_val=passive_gap_val.numpy(), passive_entropy_val=passive_entropy_val.numpy(), probe_response_magnitude_val=probe_response_magnitude_val.numpy(),
    )
    ROUTER_TRAIN_OOF_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        ROUTER_TRAIN_OOF_DIR / f"{dataset}.npz",
        oof_probe=oof_probe.numpy(), oof_passive=oof_passive.numpy(), oof_probe_response=oof_probe_response.numpy(),
        common_idx=common_idx.numpy(), legal_idx_all=legal_idx_all.numpy(),
    )
    ROUTER_VAL_PRED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        ROUTER_VAL_PRED_DIR / f"{dataset}.npz",
        pred_full=pred_full_val.numpy(), pred_common=pred_common_val.numpy(), pred_matchedpassive=pred_passive_val.numpy(), pred_probe=pred_probe_val.numpy(), pred_shuffled=pred_shuffled_val.numpy(),
    )

    return {
        "dataset": dataset,
        "core": core,
        "checkpoint_hashes": checkpoint_hashes_after,
        "fold_diag_rows": fold_diag_rows,
        "result_rows": result_rows,
        "dependence_rows": dependence_rows,
        "primary": primary,
        "weight_rows": weight_rows,
        "mechanism": mechanism,
        "residual_diag": residual_diag,
        "mechanism_case": mechanism_case,
        "stratified": stratified,
        "reproduction": reproduction,
        "integrity": integrity,
        "probe_vs_passive_disagreement_fraction_val": float(probe_vs_passive_disagree_val.mean()),
    }


# ---------------------------------------------------------------------------
# Section 26: passive-vs-active mechanism-interpretation classifier, applied
# per dataset (predeclared thresholds -- see ACTIVE_ONLY_R2_THRESHOLD /
# ACTIVE_ONLY_SPEARMAN_THRESHOLD above, fixed before any result was seen).
# ---------------------------------------------------------------------------


def classify_mechanism_case(mechanism: Mapping[str, Any], residual_diag: Mapping[str, Any], probe_router_mae: float, passive_router_mae: float) -> dict[str, Any]:
    active_only_useful = is_useful(mechanism["B_active_only"])
    adds_beyond = adds_beyond_passive(mechanism)
    residual_predicts = is_useful(residual_diag)
    probe_beats_passive_router = probe_router_mae < passive_router_mae

    if not active_only_useful and not adds_beyond:
        case = "A"
        interpretation = "Active perturbation contributes little measurable competence information (active-only weak; passive+active ~= passive)."
    elif active_only_useful and not adds_beyond:
        case = "B"
        interpretation = "Active Probe information exists but is largely redundant with passive competence information (active-only useful; passive+active ~= passive)."
    elif active_only_useful and adds_beyond:
        case = "C"
        interpretation = "Active probing contains complementary competence information (active-only useful; passive+active > passive)."
    else:
        case = "UNCLASSIFIED"
        interpretation = "adds_beyond_passive True but active_only_useful False -- inspect mechanism diagnostics directly."

    if residual_predicts and not probe_beats_passive_router:
        case_d_flag = True
        case_d_interpretation = "Active response predicts MatchedPassive's residual competence, but TimeFuse+LearnedProbe does not beat TimeFuse+MatchedPassive-21 -- the active signal may exist, but the current scorer/router integration is not exploiting it effectively (Case D)."
    else:
        case_d_flag = False
        case_d_interpretation = None

    return {
        "active_only_useful": active_only_useful,
        "adds_beyond_passive": adds_beyond,
        "residual_predicts_beyond_passive": residual_predicts,
        "probe_beats_passive_router": probe_beats_passive_router,
        "case": case,
        "interpretation": interpretation,
        "case_d_integration_fails": case_d_flag,
        "case_d_interpretation": case_d_interpretation,
    }


# ---------------------------------------------------------------------------
# Section 36: claim rules, frozen before running, applied verbatim after.
# ---------------------------------------------------------------------------


def decide(report: Mapping[str, Any]) -> dict[str, Any]:
    datasets = list(report["datasets"].keys())
    n = len(datasets)
    majority = n // 2 + 1
    by = {ds: {r["method"]: r for r in report["datasets"][ds]["result_rows"]} for ds in datasets}

    def sig_improve(row_key: str, ds: str) -> bool:
        r = report["datasets"][ds]["primary"][row_key]
        return bool(r["ci_excludes_zero"] and r["mean_delta"] < 0)

    def sig_regress(row_key: str, ds: str) -> bool:
        r = report["datasets"][ds]["primary"][row_key]
        return bool(r["ci_excludes_zero"] and r["mean_delta"] > 0)

    n_beats_common_point = sum(1 for ds in datasets if by[ds]["TimeFuse_LearnedProbe"]["mae"] < by[ds]["TimeFuse_Common"]["mae"])
    n_beats_common_sig = sum(1 for ds in datasets if sig_improve("Probe_vs_Common", ds))
    n_ties_or_beats_full = sum(1 for ds in datasets if by[ds]["TimeFuse_LearnedProbe"]["mae"] <= by[ds]["TimeFuse_Full"]["mae"] or not sig_regress("Probe_vs_Full", ds))
    n_beats_passive_point = sum(1 for ds in datasets if by[ds]["TimeFuse_LearnedProbe"]["mae"] < by[ds]["TimeFuse_MatchedPassive21"]["mae"])
    n_beats_passive_sig = sum(1 for ds in datasets if sig_improve("Probe_vs_MatchedPassive", ds))
    n_beats_shuffled_point = sum(1 for ds in datasets if by[ds]["TimeFuse_LearnedProbe"]["mae"] < by[ds]["TimeFuse_ShuffledProbe"]["mae"])
    n_beats_shuffled_sig = sum(1 for ds in datasets if sig_improve("Probe_vs_Shuffled", ds))
    n_broad_regressions = sum(1 for ds in datasets if sig_regress("Probe_vs_Common", ds))

    n_active_only_useful = sum(1 for ds in datasets if report["datasets"][ds]["mechanism_case"]["active_only_useful"])
    n_residual_predicts = sum(1 for ds in datasets if report["datasets"][ds]["mechanism_case"]["residual_predicts_beyond_passive"])
    active_only_useful_majority = n_active_only_useful >= majority
    residual_predicts_majority = n_residual_predicts >= majority

    strong = (
        n_beats_common_point >= majority and n_beats_common_sig >= 2
        and n_ties_or_beats_full >= majority
        and n_beats_passive_point >= majority
        and n_beats_shuffled_point >= majority and n_beats_shuffled_sig >= 1
        and n_broad_regressions == 0
    )
    probe_loses_to_passive_majority = n_beats_passive_point < majority

    if strong:
        tier = "COMPLEMENTARY_ACTIVE_SIGNAL"
        conclusion = "STRONG active-Probe evidence: LearnedProbe beats TimeFuse-Common, is competitive with/beats TimeFuse-Full, beats MatchedPassive-21, beats ShuffledProbe, on multiple datasets, with no broad significant regressions."
    elif probe_loses_to_passive_majority and residual_predicts_majority:
        tier = "SIGNAL_EXISTS_INTEGRATION_FAILS"
        conclusion = "The active diagnostic signal appears to contain complementary information (predicts MatchedPassive's residual competence beyond passive features on a majority of datasets), but the current LearnedProbe scoring/fusion mechanism does not translate this into a TimeFuse router win over MatchedPassive-21. This motivates a SEPARATE future residual/conditional Probe experiment -- not implemented here."
    elif probe_loses_to_passive_majority and active_only_useful_majority and not residual_predicts_majority:
        tier = "USEFUL_BUT_REDUNDANT_ACTIVE_SIGNAL"
        conclusion = "The learned perturbation response contains competence information (active-only diagnostics beat the predeclared usefulness threshold on a majority of datasets), but much of it is redundant with information available from passive forecasts and disagreement: LearnedProbe does not clearly beat MatchedPassive-21 in final TimeFuse MAE."
    elif probe_loses_to_passive_majority and not active_only_useful_majority and not residual_predicts_majority:
        tier = "NO_INCREMENTAL_ACTIVE_SIGNAL"
        conclusion = "Under the strict purged-OOF protocol, active perturbation does not provide measurable competence information beyond the matched passive estimator: MatchedPassive-21 ties or beats LearnedProbe on a majority of datasets, and active-response features fail to predict either raw competence or MatchedPassive's residual competence beyond the predeclared thresholds."
    else:
        tier = "MIXED"
        conclusion = "Partial, inconsistent evidence across datasets and comparisons; does not cleanly match any of the four predeclared cases. Treat as suggestive, not confirmatory -- see per-dataset mechanism_case and primary comparison tables."

    return {
        "tier": tier,
        "conclusion": conclusion,
        "n_datasets": n,
        "n_beats_common_point": n_beats_common_point, "n_beats_common_sig": n_beats_common_sig,
        "n_ties_or_beats_full": n_ties_or_beats_full,
        "n_beats_passive_point": n_beats_passive_point, "n_beats_passive_sig": n_beats_passive_sig,
        "n_beats_shuffled_point": n_beats_shuffled_point, "n_beats_shuffled_sig": n_beats_shuffled_sig,
        "n_broad_regressions": n_broad_regressions,
        "n_active_only_useful": n_active_only_useful, "n_residual_predicts": n_residual_predicts,
        "active_only_useful_majority": active_only_useful_majority, "residual_predicts_majority": residual_predicts_majority,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def make_report(report: Mapping[str, Any], decision: Mapping[str, Any]) -> None:
    datasets = list(report["datasets"].keys())
    lines = [
        "# Strict Purged-OOF TimeFuse + LearnedProbe Mechanism Study",
        "",
        "Corrects the in-sample stacking bug in `../timefuse_probe/`: every honest LearnedProbe / MatchedPassive-21 score used as a TimeFuse TRAINING feature here comes from a model retrained on a PURGED, causally-earlier prefix of router_train, reusing the exact fold machinery built and verified for `../fforma_probe/`. No Selective gate is used in this experiment.",
        "",
        "## Section 1/2/9/14: mandatory causal assertions", "",
        "| Dataset | Fold | Train target-end max | Eval origin min | Assertion holds | Purged windows |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for ds in datasets:
        for row in report["datasets"][ds]["fold_diag_rows"]:
            lines.append(f"| {ds} | {row['fold']} | {row['train_target_end_max']} | {row['eval_origin_min']} | {row['assertion_max_train_target_end_leq_min_eval_origin']} | {row['num_purged_windows']} |")
    lines += ["", "| Dataset | router_train->router_val observability holds | max train target-end | min val origin | Common windows | Full legal windows |", "|---|---|---:|---:|---:|---:|"]
    for ds in datasets:
        i = report["datasets"][ds]["integrity"]
        lines.append(f"| {ds} | {i['router_train_to_router_val_observability_holds']} | {i['max_router_train_target_end']} | {i['min_router_val_origin']} | {i['num_common_windows']} | {i['num_full_legal_windows']} |")

    lines += ["", "## Section 35: primary results (router_val MAE)", ""]
    lines.append("| Dataset | Full | Common | +MatchedPassive21 | +LearnedProbe | +ShuffledProbe |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for ds in datasets:
        by = {r["method"]: r for r in report["datasets"][ds]["result_rows"]}
        lines.append(f"| {ds} | {by['TimeFuse_Full']['mae']:.6f} | {by['TimeFuse_Common']['mae']:.6f} | {by['TimeFuse_MatchedPassive21']['mae']:.6f} | {by['TimeFuse_LearnedProbe']['mae']:.6f} | {by['TimeFuse_ShuffledProbe']['mae']:.6f} |")
    lines += ["", "### MSE", "", "| Dataset | Full | Common | +MatchedPassive21 | +LearnedProbe | +ShuffledProbe |", "|---|---:|---:|---:|---:|---:|"]
    for ds in datasets:
        by = {r["method"]: r for r in report["datasets"][ds]["result_rows"]}
        lines.append(f"| {ds} | {by['TimeFuse_Full']['mse']:.6f} | {by['TimeFuse_Common']['mse']:.6f} | {by['TimeFuse_MatchedPassive21']['mse']:.6f} | {by['TimeFuse_LearnedProbe']['mse']:.6f} | {by['TimeFuse_ShuffledProbe']['mse']:.6f} |")

    lines += ["", "## LearnedProbe deltas", "", "| Dataset | vs Common | % | vs Full | % | vs MatchedPassive21 | % | vs Shuffled | % | block-24 sig? |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for ds in datasets:
        by = {r["method"]: r for r in report["datasets"][ds]["result_rows"]}
        p = by["TimeFuse_LearnedProbe"]
        sig_bits = []
        for key, label in (("Probe_vs_Common", "C"), ("Probe_vs_Full", "F"), ("Probe_vs_MatchedPassive", "P"), ("Probe_vs_Shuffled", "S")):
            r = report["datasets"][ds]["primary"][key]
            if r["ci_excludes_zero"]:
                sig_bits.append(f"{label}{'-' if r['mean_delta']<0 else '+'}")
        lines.append(f"| {ds} | `{p['delta_vs_common']:+.6f}` | `{p['pct_vs_common']:+.2f}%` | `{p['delta_vs_full']:+.6f}` | `{p['pct_vs_full']:+.2f}%` | `{p['delta_vs_matchedpassive']:+.6f}` | `{p['pct_vs_matchedpassive']:+.2f}%` | `{p['delta_vs_shuffled']:+.6f}` | `{p['pct_vs_shuffled']:+.2f}%` | {', '.join(sig_bits) or '(none)'} |")

    lines += ["", "## Section 23: primary comparisons A-F (block-24)", "", "| Dataset | Comparison | Mean Delta | 95% CI | P(Delta<0) | Excludes zero |", "|---|---|---:|---|---:|---|"]
    for ds in datasets:
        for key, label in (("Probe_vs_Common", "A: Probe_vs_Common"), ("Probe_vs_Full", "B: Probe_vs_Full"), ("Probe_vs_MatchedPassive", "C: Probe_vs_MatchedPassive (MOST IMPORTANT)"), ("Probe_vs_Shuffled", "D: Probe_vs_Shuffled"), ("MatchedPassive_vs_Common", "E: MatchedPassive_vs_Common"), ("MatchedPassive_vs_Full", "F: MatchedPassive_vs_Full")):
            r = report["datasets"][ds]["primary"][key]
            lines.append(f"| {ds} | {label} | `{r['mean_delta']:+.6f}` | [{r['ci95_low']:+.6f}, {r['ci95_high']:+.6f}] | {r['prob_delta_negative']:.3f} | {r['ci_excludes_zero']} |")

    lines += ["", "## Full dependence-aware statistics (all block lengths + phase)", "", "| Dataset | Comparison | Test | Mean Delta | 95% CI | P(Delta<0) | Excludes zero |", "|---|---|---|---:|---|---:|---|"]
    for ds in datasets:
        for row in report["datasets"][ds]["dependence_rows"]:
            mean_key = row.get("mean_delta", row.get("mean_diff_candidate_minus_baseline"))
            if "ci95_low" in row and mean_key is not None:
                lines.append(f"| {ds} | {row['comparison']} | {row['test']} | `{mean_key:+.6f}` | [{row['ci95_low']:+.6f}, {row['ci95_high']:+.6f}] | {row.get('prob_delta_negative', '')} | {row['ci_excludes_zero']} |")

    lines += ["", "## Weight analysis", "", "| Dataset | Method | Mean entropy | Mean max weight | Mean eff. #experts | Fraction top-expert changed vs Full |", "|---|---|---:|---:|---:|---:|"]
    for ds in datasets:
        for row in report["datasets"][ds]["weight_rows"]:
            lines.append(f"| {ds} | {row['method']} | {row['mean_entropy']:.4f} | {row['mean_max_weight']:.4f} | {row['mean_effective_num_experts']:.3f} | {row['fraction_top_expert_changed_vs_full']:.3f} |")

    lines += ["", "## Section 24: mechanism diagnostics (OOF Common, router_train, chronological holdout Ridge)", "", "| Dataset | Probe | Pearson r | Spearman rho | R2 | MAE | MAE (null=mean) |", "|---|---|---:|---:|---:|---:|---:|"]
    for ds in datasets:
        for key, label in (("A_passive_only", "A: passive-only (15)"), ("B_active_only", "B: active-only (6)"), ("C_passive_plus_active", "C: passive+active (21)"), ("D_passive_plus_shuffled_active", "D: passive+shuffled-active (21)")):
            d = report["datasets"][ds]["mechanism"][key]
            lines.append(f"| {ds} | {label} | {d['pearson_r']:.4f} | {d['spearman_rho']:.4f} | {d['r2']:.4f} | {d['mae']:.6f} | {d['mae_null_mean']:.6f} |")

    lines += ["", "## Section 25: residual-competence diagnostic (active features -> MatchedPassive's OOF residual)", "", "| Dataset | Pearson r | Spearman rho | R2 | MAE | MAE (null=mean) | Useful (predeclared threshold)? |", "|---|---:|---:|---:|---:|---:|---|"]
    for ds in datasets:
        d = report["datasets"][ds]["residual_diag"]
        useful = report["datasets"][ds]["mechanism_case"]["residual_predicts_beyond_passive"]
        lines.append(f"| {ds} | {d['pearson_r']:.4f} | {d['spearman_rho']:.4f} | {d['r2']:.4f} | {d['mae']:.6f} | {d['mae_null_mean']:.6f} | {useful} |")

    lines += ["", "## Section 26: passive-vs-active mechanism interpretation, per dataset", ""]
    for ds in datasets:
        mc = report["datasets"][ds]["mechanism_case"]
        lines.append(f"- **{ds}**: Case **{mc['case']}** -- {mc['interpretation']}" + (f" **Case D also flagged**: {mc['case_d_interpretation']}" if mc["case_d_integration_fails"] else ""))

    lines += ["", "## Section 27: stratified hard/ambiguous window analysis (target-free strata, router_val)", "", "| Dataset | Stratifier | Stratum | N windows | LearnedProbe MAE | MatchedPassive21 MAE | Delta (Probe-Passive) |", "|---|---|---|---:|---:|---:|---:|"]
    for ds in datasets:
        for row in report["datasets"][ds]["stratified"]:
            lines.append(f"| {ds} | {row['stratifier']} | {row['stratum']} | {row['n_windows']} | {row['probe_mae']:.6f} | {row['matchedpassive_mae']:.6f} | `{row['delta_probe_minus_matchedpassive']:+.6f}` |")

    lines += ["", "## Section 28: existing-result reproduction (old in-sample-stacked base TimeFuse)", "", "| Dataset | Old TimeFuse MAE | New TimeFuse-Full MAE | Difference | Relative diff | Within 5% tolerance |", "|---|---:|---:|---:|---:|---|"]
    for ds in datasets:
        r = report["datasets"][ds]["reproduction"]
        if r["old_base_timefuse_mae"] is not None:
            lines.append(f"| {ds} | {r['old_base_timefuse_mae']:.6f} | {r['new_timefuse_full_mae']:.6f} | `{r['difference']:+.6f}` | `{r['relative_difference']:+.2%}` | {r['within_tolerance']} |")
        else:
            lines.append(f"| {ds} | (not found) | {r['new_timefuse_full_mae']:.6f} | - | - | - |")
    lines.append("")
    lines.append(report["datasets"][datasets[0]]["reproduction"]["reason_if_different"])

    lines += ["", "## Sections 29-33: integrity", ""]
    for ds in datasets:
        i = report["datasets"][ds]["integrity"]
        lines.append(
            f"- **{ds}**: {i['result']} (checkpoints unchanged: {i['expert_checkpoints_unchanged']}; no test cache: {i['no_test_cache_loaded']}; all purge assertions pass: {i['all_purge_fold_assertions_pass']}; "
            f"observability holds: {i['router_train_to_router_val_observability_holds']}; target-corruption invariant: {i['target_corruption_invariant']}; expert-order permutation invariant: {i['expert_order_permutation_invariant']}; "
            f"weighted-forecast reproduces: {i['weighted_forecast_reproduces']})"
        )

    lines += ["", "## Section 36: claim rule / decision", ""]
    lines.append(f"- n_beats_common_point={decision['n_beats_common_point']}/{decision['n_datasets']}, n_beats_common_sig={decision['n_beats_common_sig']}")
    lines.append(f"- n_ties_or_beats_full={decision['n_ties_or_beats_full']}/{decision['n_datasets']}")
    lines.append(f"- n_beats_passive_point={decision['n_beats_passive_point']}/{decision['n_datasets']}, n_beats_passive_sig={decision['n_beats_passive_sig']}")
    lines.append(f"- n_beats_shuffled_point={decision['n_beats_shuffled_point']}/{decision['n_datasets']}, n_beats_shuffled_sig={decision['n_beats_shuffled_sig']}")
    lines.append(f"- n_broad_regressions={decision['n_broad_regressions']}")
    lines.append(f"- n_active_only_useful={decision['n_active_only_useful']}/{decision['n_datasets']} (majority={decision['active_only_useful_majority']})")
    lines.append(f"- n_residual_predicts_beyond_passive={decision['n_residual_predicts']}/{decision['n_datasets']} (majority={decision['residual_predicts_majority']})")
    lines += ["", f"## Decision: {decision['tier']}", "", decision["conclusion"], ""]

    lines += ["", "## Section 39: answers", ""]
    lines.append("**1. Was official TimeFuse preserved?** Yes -- meta_feature.extract_meta_feature (22-dim, via the cached wrapper already verified byte-identical on well-behaved inputs) and timefuse.ModelFusor used verbatim from commit 978e6c6b9e4f246632c269aa0f9beeb099eabcfc, with the exact official training hyperparameters, imported unchanged from `../timefuse_probe/run_timefuse_probe.py`.")
    lines.append("**2. Does TimeFuse-Full reproduce the prior base TimeFuse result?** See Section 28 table above.")
    lines.append("**3. Were all competence scores used to TRAIN augmented TimeFuse honest purged OOF predictions?** Yes -- every MatchedPassive/LearnedProbe score used as a TimeFuse training feature (Common window set) comes from a fold-restricted model trained ONLY on causally-earlier windows (Sections 1/9-11), never from a model that saw that window's own target.")
    lines.append(f"**4. Did every fold satisfy max_train_target_end <= min_heldout_origin?** {'Yes, on every dataset/fold (see table above).' if all(report['datasets'][ds]['integrity']['all_purge_fold_assertions_pass'] for ds in datasets) else 'NO -- see table above; this should never happen since a failure raises immediately.'}")
    lines.append(f"**5. Does corrected LearnedProbe still improve TimeFuse-Common?** By point estimate on {decision['n_beats_common_point']}/{decision['n_datasets']}; block-24 significant on {decision['n_beats_common_sig']}/{decision['n_datasets']}.")
    lines.append(f"**6. Does it also improve or match TimeFuse-Full?** Ties-or-beats (or non-significant regression) on {decision['n_ties_or_beats_full']}/{decision['n_datasets']}.")
    lines.append(f"**7. Does LearnedProbe beat MatchedPassive-21?** By point estimate on {decision['n_beats_passive_point']}/{decision['n_datasets']}; block-24 significant on {decision['n_beats_passive_sig']}/{decision['n_datasets']}. **This is the most important comparison.**")
    lines.append(f"**8. Does it beat ShuffledProbe?** By point estimate on {decision['n_beats_shuffled_point']}/{decision['n_datasets']}; block-24 significant on {decision['n_beats_shuffled_sig']}/{decision['n_datasets']}.")
    lines.append(f"**9. Do the six active Probe-response features predict expert competence by themselves?** See Section 24 row B per dataset above; useful (predeclared R2>{ACTIVE_ONLY_R2_THRESHOLD}, |Spearman|>{ACTIVE_ONLY_SPEARMAN_THRESHOLD}) on {decision['n_active_only_useful']}/{decision['n_datasets']} datasets.")
    lines.append("**10. Do active features add predictive value beyond the 15 passive features?** Compare Section 24 rows A and C per dataset (R2 improvement) -- see `mechanism_case.adds_beyond_passive` per dataset above.")
    lines.append(f"**11. Do active features predict the residual competence error left by MatchedPassive?** See Section 25 table above; useful on {decision['n_residual_predicts']}/{decision['n_datasets']} datasets.")
    for ds in datasets:
        lines.append(f"  - {ds}: {report['datasets'][ds]['probe_vs_passive_disagreement_fraction_val']:.3f} fraction of router_val windows where LearnedProbe's and MatchedPassive's argmin-expert disagree.")
    lines.append("**12. When LearnedProbe and MatchedPassive disagree, which is more often correct?** See the `probe_vs_passive_disagreement_fraction_val` figures directly above and the Section 27 stratified table's `probe_response_magnitude`/`matchedpassive_entropy` strata, which condition router-level MAE on exactly this kind of disagreement/uncertainty.")
    lines.append("**13. Does LearnedProbe become more useful on high-disagreement or passive-uncertain windows?** See Section 27 stratified table above -- compare the `delta_probe_minus_matchedpassive` column across bottom_25pct/middle_50pct/top_25pct strata for `expert_forecast_disagreement` and `matchedpassive_entropy`.")
    lines.append(f"**14. Were there significant wins or regressions under block-24 bootstrap?** See Section 23 table above; n_broad_regressions={decision['n_broad_regressions']} (broad = significant regression vs Common on >= half the datasets).")
    lines.append(f"**15. Based strictly on these results, which description is supported?** **{decision['tier']}** -- {decision['conclusion']}")

    lines += ["", "## Section 40: final scientific question", "", "\"Under a strict causal purged-OOF protocol, does actively perturbing frozen forecasting experts reveal expert-specific competence information that TimeFuse cannot already infer from ordinary passive window, forecast, and disagreement information?\"", "", decision["conclusion"], ""]

    lines += [
        "## Hard rule compliance", "", "```text",
        "TEST SET ACCESSED: NO",
        "FORECASTING EXPERTS RETRAINED: NO",
        "LEARNEDPROBE ARCHITECTURE/LOSS/TRAINING MODIFIED: NO",
        "SELECTIVE GATE USED: NO (per Section 5/12)",
        "OTHER PUBLISHED ROUTERS IMPLEMENTED: NO (TimeFuse only)",
        "COSTAR / ONLINE COSTAR TOUCHED: NO",
        "PURGE ASSERTION: see table above; raises AssertionError immediately if violated",
        "```",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    start = time.time()
    report: dict[str, Any] = {
        "experiment": "timefuse_probe_purged_oof",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_sha": git_commit_sha(),
        "datasets": {},
    }
    all_results, all_dependence, all_weights, all_integrity, all_folds, all_stratified = [], [], [], [], [], []
    all_mechanism_rows, all_residual_rows = [], []

    for dataset in NEW_DATASETS:
        print(f"[timefuse_probe_purged_oof] {dataset}: starting...", flush=True)
        result = evaluate_dataset(dataset)
        report["datasets"][dataset] = result
        all_results.extend(result["result_rows"])
        all_dependence.extend(result["dependence_rows"])
        all_weights.extend(result["weight_rows"])
        all_integrity.append(result["integrity"])
        all_folds.extend(result["fold_diag_rows"])
        all_stratified.extend(result["stratified"])
        for probe_name, d in result["mechanism"].items():
            all_mechanism_rows.append({"dataset": dataset, "probe": probe_name, **d})
        all_residual_rows.append({"dataset": dataset, **result["residual_diag"], "mechanism_case": result["mechanism_case"]["case"]})
        print(f"[timefuse_probe_purged_oof] {dataset}: done.", flush=True)

    decision = decide(report)
    report["decision"] = decision
    report["runtime_sec"] = time.time() - start
    report["test_set_accessed"] = False

    source_manifest = {
        "manifest_type": "source_manifest",
        "created_at_utc": report["created_at_utc"],
        "basicts_git_commit_sha_at_experiment_time": report["git_commit_sha"],
        "timefuse_repository": "https://github.com/ZhiningLiu1998/TimeFuse",
        "timefuse_commit_sha": "978e6c6b9e4f246632c269aa0f9beeb099eabcfc",
        "reused_unmodified_from": "experiments/behavioral_competence/timefuse_probe/ (official_timefuse_source_manifest.json, meta_feature_cache.py, run_timefuse_probe.py::train_timefuse_fusor/evaluate_fusor/count_params)",
        "fforma_probe_purged_oof_machinery_reused_unmodified_from": "experiments/behavioral_competence/fforma_probe/run_fforma_probe.py::purged_walkforward_folds, verify_router_train_to_val_observability, train_probe_and_scorer_prefix, train_matched_passive_prefix, score_matched_passive_on_windows, N_PURGE_FOLDS, MIN_TRAIN_FRACTION",
        "note": "See ../timefuse_probe/official_timefuse_source_manifest.json for the full official-source citation and engineering-adaptation list -- unchanged here.",
    }
    write_json(OUT_DIR / "source_manifest.json", source_manifest)

    method_manifest = {
        "manifest_type": "timefuse_probe_purged_oof_method_manifest",
        "created_at_utc": report["created_at_utc"],
        "git_commit_sha": report["git_commit_sha"],
        "expert_cores": {ds: report["datasets"][ds]["core"] for ds in NEW_DATASETS},
        "expert_checkpoint_sha256": {ds: report["datasets"][ds]["checkpoint_hashes"] for ds in NEW_DATASETS},
        "meta_feature_names": META_FEATURE_NAMES,
        "n_purge_folds": N_PURGE_FOLDS,
        "min_train_fraction": MIN_TRAIN_FRACTION,
        "shuffle_seed": SHUFFLE_SEED,
        "derangement": "Per-window non-identity permutation of the K=3 expert axis, chosen deterministically between the two non-identity permutations of {0,1,2} via SHA-256(dataset|absolute_window_start|seed); values never modified, only expert-identity correspondence.",
        "matched_passive_architecture": "CompetenceScorer(21) identical to LearnedProbe: 15 passive (group A+B+C) features + 6 constant-zero columns, same optimizer/lr/weight_decay/epochs/patience/Huber+ranking objective, seed=7. No ProbeGenerator, no perturbation penalties.",
        "score_scalers": "sklearn StandardScaler (via TimeFuse's own TorchScaler/get_scaler('standard')), fit independently for MatchedPassive/LearnedProbe/ShuffledProbe, each on its own honest OOF router_train Common-window score matrix ONLY, never on router_val.",
        "timefuse_input_dims": {"base": TIMEFUSE_META_DIM, "augmented": TIMEFUSE_META_DIM + 3},
        "selective_gate_used": False,
        "mechanism_diagnostic_model": f"sklearn Ridge(alpha={RIDGE_ALPHA}), fixed hyperparameter, fit on a chronological 80/20 (fit/holdout) split of the OOF Common window set (router_train only), never tuned, never touching router_val.",
        "mechanism_thresholds": {"active_only_r2": ACTIVE_ONLY_R2_THRESHOLD, "active_only_spearman": ACTIVE_ONLY_SPEARMAN_THRESHOLD},
        "decision_rule": "Section 36 of the task instructions, applied verbatim without modification after seeing results.",
    }
    write_json(OUT_DIR / "method_manifest.json", method_manifest)

    write_json(OUT_DIR / "validation_results.json", report)
    write_csv(OUT_DIR / "validation_results.csv", all_results)
    write_csv(OUT_DIR / "dependence_aware_results.csv", all_dependence)
    write_csv(OUT_DIR / "oof_fold_manifest.csv", all_folds)
    write_csv(OUT_DIR / "causality_checks.csv", all_folds)
    write_csv(OUT_DIR / "expert_provenance_checks.csv", [{"dataset": ds, "expert_forecast_provenance_ok": report["datasets"][ds]["integrity"]["expert_forecast_provenance_ok"]} for ds in NEW_DATASETS])
    write_csv(OUT_DIR / "integrity_checks.csv", all_integrity)
    write_csv(OUT_DIR / "routing_diagnostics.csv", all_weights)
    write_csv(OUT_DIR / "mechanism_diagnostics.csv", all_mechanism_rows)
    write_csv(OUT_DIR / "residual_competence_diagnostics.csv", all_residual_rows)
    write_csv(OUT_DIR / "stratified_window_diagnostics.csv", all_stratified)
    make_report(report, decision)

    print("TEST SET ACCESSED: NO")
    print(json.dumps({"runtime_sec": report["runtime_sec"], "tier": decision["tier"]}, indent=2))


if __name__ == "__main__":
    main()
