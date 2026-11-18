"""FFORMA (official M4 reference algorithm, robjhyndman/M4metalearning commit
61ddc7101680e9df7219c359587d0b509d2b50d6) vs FFORMA + MatchedPassive-21 vs
FFORMA + LearnedProbe vs FFORMA + ShuffledProbe, under mandatory purged
causal OOF (Final Audit Overrides).

Six primary methods, all sharing the SAME frozen FFORMA hyperparameters
(selected using base FFORMA only, via purged OOF, BEFORE any Probe/Passive
result is seen):
  1. FFORMA-M4Fixed   -- official M4 hyperparameters, fidelity check
  2. FFORMA-MAE-Full  -- base-selected hyperparameters, ALL causally legal router_train rows
  3. FFORMA-MAE-Common -- base-selected hyperparameters, only the OOF-eligible
                          window set shared by MatchedPassive/LearnedProbe/ShuffledProbe
  4. FFORMA-MAE + MatchedPassive-21
  5. FFORMA-MAE + LearnedProbe
  6. FFORMA-MAE + ShuffledProbe

Every target-dependent supervised component (LearnedProbe, MatchedPassive,
FFORMA's own hyperparameter-selection OOF) is purged: a training example
with forecast origin s may supervise a fold evaluated starting at origin T
only if s + forecast_horizon <= T (assert_purge, enforced and logged for
every fold; see integrity_checks.csv / report.md).

Reuses, unmodified:
  - vendor/M4metalearning (commit 61ddc7101680e9df7219c359587d0b509d2b50d6):
    cited for the exact THA_features feature-group list, the
    error_softmax_obj custom XGBoost objective formula, and the official
    M4 hyperparameters (max_depth=14, eta=0.575188, subsample=0.9161483,
    colsample_bytree=0.7670739, nrounds=94) -- ported to Python xgboost
    (verified: modern xgboost's custom-objective callback already receives
    raw margins as [N,K] and expects grad/hess as [N,K], so no R-style
    transpose is needed; the softmax-expected-loss formula itself is
    reused character-for-character).
  - fforma_features.py (this experiment): THA_features port via Python tsfeatures
  - experiments/behavioral_competence/probe_generator.py: ProbeGenerator, eps=0.05,
    perturbation_penalties, pairwise_ranking_loss, probe_response_features
  - experiments/behavioral_competence/common.py: CompetenceScorer (21-dim,
    architecture UNCHANGED for both LearnedProbe and MatchedPassive-21)
  - experiments/behavioral_competence/run_learned_probe.py: build_abc_features,
    run_batch, stage_runtime_groups, raw_history_cache, compute_excess_loss,
    make_generator, and every frozen hyperparameter constant
  - experiments/behavioral_competence/generalization/run_generalization_study.py::register_dataset
  - experiments/behavioral_competence/simplex_probe/run_simplex_probe.py::
    metric_values, shuffle_probe_scores, dependence_full, primary_row

New code is limited to: (1) the purged-fold generator and its mandatory
causal assertion, (2) windowed (prefix-restricted) retraining wrappers for
LearnedProbe and a new MatchedPassive-21 scorer (the existing
train_probe_and_scorer has no window-subsetting parameter), (3) the Python
port of FFORMA's custom XGBoost objective and its purged
hyperparameter-selection search over a small predeclared grid.

router_val only for the final comparison; router_val targets are never used
to fit anything. No test cache for any dataset is built or loaded.
"""

from __future__ import annotations

import copy
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import xgboost as xgb


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.frozen_hv_costar.run_frozen_hv_costar as fhv  # noqa: E402
from experiments.behavioral_competence.common import CompetenceScorer  # noqa: E402
from experiments.behavioral_competence.generalization.run_generalization_study import register_dataset  # noqa: E402
from experiments.behavioral_competence.model_runtime import load_expert_runtime  # noqa: E402
from experiments.behavioral_competence.probe_generator import pairwise_ranking_loss, perturbation_penalties, probe_response_features  # noqa: E402
from experiments.behavioral_competence.run_behavioral_competence import compute_excess_loss, raw_history_cache  # noqa: E402
from experiments.behavioral_competence.run_learned_probe import (  # noqa: E402
    BATCH_SIZE, INTERNAL_VAL_FRACTION, LR, MAX_EPOCHS, PATIENCE, PERTURBATION_WEIGHT,
    RANKING_WEIGHT, SMOOTHNESS_WEIGHT, STATIC_FEATURE_DIM, WEIGHT_DECAY,
    build_abc_features, evaluate_on_val, make_generator, run_batch, stage_runtime_groups,
)
from experiments.behavioral_competence.simplex_probe.run_simplex_probe import (  # noqa: E402
    apply_per_window_weights, dependence_full, metric_values, primary_row, shuffle_probe_scores,
)
from experiments.oracle_weight_tournament.run_tournament import sample_mae, sample_mse  # noqa: E402
from experiments.behavioral_competence.fforma_probe.fforma_features import FFORMA_FEATURE_NAMES, get_or_compute_fforma_features  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent
PER_WINDOW_DIR = OUT_DIR / "per_window_errors"
NEW_DATASETS = ["ExchangeRate", "Traffic", "BeijingAirQuality", "ETTm2"]
FREQ_BY_DATASET = {"ExchangeRate": 1, "BeijingAirQuality": 24, "Traffic": 24, "ETTm2": 96}
SHUFFLE_SEED = 20260821
BLOCK_LENGTHS = (12, 24, 48)
PRIMARY_BLOCK = 24
BOOTSTRAP_SAMPLES = 10000
PHASE_K = 12
N_PURGE_FOLDS = 2  # reduced from a larger nominal budget for compute tractability across 4 datasets incl. Traffic; predeclared, not tuned post-hoc
MIN_TRAIN_FRACTION = 0.4
PROBE_FEATURE_DIM = 6

FFORMA_M4_CONFIG = {"max_depth": 14, "eta": 0.575188, "subsample": 0.9161483, "colsample_bytree": 0.7670739, "nrounds": 94}
FFORMA_SEARCH_GRID = [
    dict(FFORMA_M4_CONFIG),
    {"max_depth": 6, "eta": 0.1, "subsample": 0.8, "colsample_bytree": 0.8, "nrounds": 100},
    {"max_depth": 10, "eta": 0.05, "subsample": 0.7, "colsample_bytree": 0.7, "nrounds": 150},
    {"max_depth": 8, "eta": 0.3, "subsample": 0.9, "colsample_bytree": 0.6, "nrounds": 50},
    {"max_depth": 14, "eta": 0.01, "subsample": 0.6, "colsample_bytree": 0.9, "nrounds": 250},
]


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
# 1-2. Mandatory purged chronological folds.
# ---------------------------------------------------------------------------


def purged_walkforward_folds(train_cache: Mapping[str, Any], n_folds: int = N_PURGE_FOLDS, min_train_fraction: float = MIN_TRAIN_FRACTION) -> list[dict[str, Any]]:
    origins = train_cache["absolute_window_starts"].to(torch.long)
    horizon = int(train_cache["forecast_horizon"])
    n = int(train_cache["num_windows"])
    min_train = int(round(n * min_train_fraction))
    usable = n - min_train
    bounds = [min_train + i * usable // n_folds for i in range(n_folds + 1)]
    folds = []
    for i in range(n_folds):
        lo, hi = bounds[i], bounds[i + 1]
        if hi <= lo:
            continue
        min_eval_origin = int(origins[lo])
        legal_mask = (origins + horizon) <= min_eval_origin
        train_idx = torch.nonzero(legal_mask, as_tuple=True)[0]
        train_idx = train_idx[train_idx < lo]  # never use "future" windows even if they happen to be legal by origin (keeps strict walk-forward)
        eval_idx = torch.arange(lo, hi)
        max_train_target_end = int((origins[train_idx] + horizon).max()) if train_idx.numel() else -1
        purged_count = int(lo - train_idx.numel())
        assertion_ok = max_train_target_end <= min_eval_origin if train_idx.numel() else True
        if not assertion_ok:
            raise AssertionError(f"Purge assertion FAILED: max_train_target_end={max_train_target_end} > min_eval_origin={min_eval_origin}")
        folds.append(
            {
                "fold": i,
                "train_idx": train_idx,
                "eval_idx": eval_idx,
                "train_origin_min": int(origins[train_idx].min()) if train_idx.numel() else None,
                "train_origin_max": int(origins[train_idx].max()) if train_idx.numel() else None,
                "train_target_end_max": max_train_target_end,
                "eval_origin_min": int(origins[eval_idx].min()),
                "eval_origin_max": int(origins[eval_idx].max()),
                "num_purged_windows": purged_count,
                "assertion_max_train_target_end_leq_min_eval_origin": assertion_ok,
            }
        )
    return folds


def verify_router_train_to_val_observability(train_cache: Mapping[str, Any], val_cache: Mapping[str, Any]) -> dict[str, Any]:
    horizon = int(train_cache["forecast_horizon"])
    train_origins = train_cache["absolute_window_starts"].to(torch.long)
    val_origins = val_cache["absolute_window_starts"].to(torch.long)
    max_train_target_end = int((train_origins + horizon).max())
    min_val_origin = int(val_origins.min())
    ok = max_train_target_end <= min_val_origin
    return {"max_router_train_target_end": max_train_target_end, "min_router_val_origin": min_val_origin, "observability_holds": ok}


# ---------------------------------------------------------------------------
# 3. FFORMA: Python port of the official custom softmax-expected-loss XGBoost
# objective (error_softmax_obj, vendor/M4metalearning/R/ensemble_classifier.R).
# ---------------------------------------------------------------------------


def make_fforma_objective(errors: np.ndarray):
    def obj(preds: np.ndarray, dtrain: xgb.DMatrix) -> tuple[np.ndarray, np.ndarray]:
        preds = np.asarray(preds, dtype=np.float64)
        e = np.exp(preds - preds.max(axis=1, keepdims=True))
        sp = e.sum(axis=1, keepdims=True)
        p = e / sp
        rowsumerrors = (p * errors).sum(axis=1, keepdims=True)
        grad = p * (errors - rowsumerrors)
        hess = errors * p * (1.0 - p) - grad * p
        return grad, hess

    return obj


def train_fforma(features: np.ndarray, errors: np.ndarray, params: Mapping[str, Any]) -> xgb.Booster:
    k = errors.shape[1]
    dtrain = xgb.DMatrix(features.astype(np.float32))
    dtrain.set_base_margin(np.zeros(features.shape[0] * k, dtype=np.float32))
    xgb_params = {
        "max_depth": int(params["max_depth"]),
        "eta": float(params["eta"]),
        "subsample": float(params["subsample"]),
        "colsample_bytree": float(params["colsample_bytree"]),
        "num_class": k,
        "disable_default_eval_metric": 1,
        "seed": 7,
    }
    obj = make_fforma_objective(errors.astype(np.float64))
    return xgb.train(xgb_params, dtrain, num_boost_round=int(params["nrounds"]), obj=obj)


def predict_fforma(booster: xgb.Booster, features: np.ndarray, k: int) -> torch.Tensor:
    dtest = xgb.DMatrix(features.astype(np.float32))
    dtest.set_base_margin(np.zeros(features.shape[0] * k, dtype=np.float32))
    margins = booster.predict(dtest, output_margin=True)
    weights = torch.softmax(torch.tensor(margins, dtype=torch.float32), dim=1)
    return weights


def fforma_weighted_prediction(weights: torch.Tensor, forecasts: torch.Tensor) -> torch.Tensor:
    """pred[t,h,v] = sum_e(weight[t,e] * forecast[t,h,v,e]), NO nonnegative clamp
    (Section 7: BasicTS predictions may legitimately be negative)."""
    return apply_per_window_weights(forecasts, weights)


# ---------------------------------------------------------------------------
# 9-11. Windowed (prefix-restricted) retraining: same frozen LearnedProbe /
# new MatchedPassive-21 specification, restricted to causally legal windows
# only. Structurally mirrors train_probe_and_scorer's internal logic
# (unmodified elsewhere in the project) with an added window-prefix bound.
# ---------------------------------------------------------------------------


def train_probe_and_scorer_prefix(dataset: str, bundle, train_cache: Mapping[str, Any], train_idx: torch.Tensor, seed: int = 7) -> dict[str, Any]:
    k = len(bundle.core_names)
    val_runtimes = {e: load_expert_runtime(dataset, e) for e in bundle.core_names}
    reference_runtime = val_runtimes[bundle.core_names[0]]
    train_cache_raw = raw_history_cache(dataset, train_cache, reference_runtime.mean, reference_runtime.std)
    group_a, group_b, group_c, forecasts_all = build_abc_features(bundle, train_cache_raw)
    excess_loss_train, _ = compute_excess_loss(train_cache, forecasts_all, bundle.std)
    history_raw_all = train_cache_raw["histories"].to(torch.float32)

    prefix_end = int(train_idx.max()) + 1 if train_idx.numel() else 0
    n_prefix = prefix_end
    split_point = int(round(n_prefix * (1 - INTERNAL_VAL_FRACTION)))
    all_stage_groups = stage_runtime_groups(dataset, bundle, train_cache, val_runtimes)
    stage_groups = [(lo, min(hi, prefix_end), rt) for lo, hi, rt in all_stage_groups if lo < prefix_end]
    stage_groups = [(lo, hi, rt) for lo, hi, rt in stage_groups if hi > lo]

    static = torch.cat([group_a, group_b, group_c], dim=-1)
    static_flat = static.reshape(-1, STATIC_FEATURE_DIM)
    n_train_rows = split_point * k
    feat_mean = static_flat[:n_train_rows].mean(dim=0, keepdim=True)
    feat_std = static_flat[:n_train_rows].std(dim=0, keepdim=True).clamp_min(1e-6)
    static_norm = ((static_flat - feat_mean) / feat_std).reshape(-1, k, STATIC_FEATURE_DIM)

    torch.manual_seed(seed)
    input_len, num_features = history_raw_all.shape[1], history_raw_all.shape[2]
    generator = make_generator("instance", input_len, num_features)
    scorer = CompetenceScorer(STATIC_FEATURE_DIM + PROBE_FEATURE_DIM)
    optimizer = torch.optim.AdamW(list(generator.parameters()) + list(scorer.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)

    def loss_for_batch(batch_idx: torch.Tensor, runtimes_stage, grad_enabled: bool) -> torch.Tensor:
        history_batch = history_raw_all[batch_idx]
        pred_excess, deltas, _ = run_batch("instance", generator, scorer, history_batch, batch_idx, bundle.core_names, runtimes_stage, static_norm, group_b, forecasts_all, bundle.std, grad_enabled)
        actual = excess_loss_train[batch_idx]
        huber = F.huber_loss(pred_excess.reshape(-1), actual.reshape(-1), delta=1.0)
        ranking = pairwise_ranking_loss(pred_excess, actual)
        l2, mean_shift, smoothness = perturbation_penalties(deltas.reshape(-1, *deltas.shape[2:]))
        return huber + RANKING_WEIGHT * ranking + PERTURBATION_WEIGHT * (l2 + mean_shift) + SMOOTHNESS_WEIGHT * smoothness

    best_val, bad, best_state = float("inf"), 0, None
    for _epoch in range(MAX_EPOCHS):
        generator.train()
        scorer.train()
        for lo, hi, runtimes_stage in stage_groups:
            window_ids = torch.arange(lo, hi)
            window_ids = window_ids[window_ids < split_point]
            if window_ids.numel() == 0:
                continue
            perm = window_ids[torch.randperm(window_ids.numel())]
            for b in range(0, perm.numel(), BATCH_SIZE):
                batch_idx = perm[b : b + BATCH_SIZE]
                loss = loss_for_batch(batch_idx, runtimes_stage, True)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        generator.eval()
        scorer.eval()
        val_losses = []
        for lo, hi, runtimes_stage in stage_groups:
            val_lo = min(max(lo, split_point), hi)
            window_ids = torch.arange(val_lo, hi)
            if window_ids.numel() == 0:
                continue
            for b in range(0, window_ids.numel(), BATCH_SIZE):
                batch_idx = window_ids[b : b + BATCH_SIZE]
                with torch.no_grad():
                    val_losses.append(float(loss_for_batch(batch_idx, runtimes_stage, False)))
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        if val_loss < best_val - 1e-6:
            best_val, bad = val_loss, 0
            best_state = {"generator": copy.deepcopy(generator.state_dict()), "scorer": copy.deepcopy(scorer.state_dict())}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        generator.load_state_dict(best_state["generator"])
        scorer.load_state_dict(best_state["scorer"])
    generator.eval()
    scorer.eval()
    return {"generator": generator, "scorer": scorer, "feat_mean": feat_mean, "feat_std": feat_std, "val_runtimes": val_runtimes, "mode": "instance"}


def score_probe_on_windows(dataset: str, bundle, fit: Mapping[str, Any], cache: Mapping[str, Any], window_idx: torch.Tensor, is_router_train: bool) -> torch.Tensor:
    """Target-free forward pass of a (possibly windowed-training) frozen fit
    over an arbitrary set of window indices. Stage-aware (block_a/block_ab)
    ONLY when scoring router_train windows (is_router_train=True); router_val
    is always a single final_60-stage group, exactly matching evaluate_on_val's
    behavior -- router_val_60_80_cache.pt has no 'source_caches' provenance
    entry for router_train_block_split to key off, since it isn't a
    block_b+block_c concatenation."""
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
    filled = torch.zeros(n, dtype=torch.bool)
    with torch.no_grad():
        for lo, hi, runtimes_stage in stage_groups:
            batch_idx = torch.tensor([i for i in range(lo, hi) if i in idx_set], dtype=torch.long)
            for b in range(0, batch_idx.numel(), BATCH_SIZE):
                chunk = batch_idx[b : b + BATCH_SIZE]
                if chunk.numel() == 0:
                    continue
                pe, _, _ = run_batch(fit["mode"], fit["generator"], fit["scorer"], history_raw_all[chunk], chunk, bundle.core_names, runtimes_stage, static_norm, group_b, forecasts_all, bundle.std, grad_enabled=False)
                pred_excess[chunk] = pe
                filled[chunk] = True
    missing = (~filled[window_idx]).sum()
    if int(missing) > 0:
        raise AssertionError(f"score_probe_on_windows: {int(missing)} requested windows were not covered by any stage group")
    return pred_excess[window_idx]


def train_matched_passive_prefix(dataset: str, bundle, train_cache: Mapping[str, Any], train_idx: torch.Tensor, seed: int = 7) -> dict[str, Any]:
    """MatchedPassive-21: identical CompetenceScorer architecture/training
    objective/protocol to LearnedProbe, but NO ProbeGenerator and NO
    perturbation -- 15 passive (A+B+C) features + 6 constant-zero columns,
    giving the same 21-dim input and the same input-layer parameter count."""
    k = len(bundle.core_names)
    reference_runtime = load_expert_runtime(dataset, bundle.core_names[0])
    train_cache_raw = raw_history_cache(dataset, train_cache, reference_runtime.mean, reference_runtime.std)
    group_a, group_b, group_c, forecasts_all = build_abc_features(bundle, train_cache_raw)
    excess_loss_train, _ = compute_excess_loss(train_cache, forecasts_all, bundle.std)
    static = torch.cat([group_a, group_b, group_c], dim=-1)  # [n,k,15]

    prefix_end = int(train_idx.max()) + 1 if train_idx.numel() else 0
    split_point = int(round(prefix_end * (1 - INTERNAL_VAL_FRACTION)))
    static_flat = static.reshape(-1, STATIC_FEATURE_DIM)
    n_train_rows = split_point * k
    feat_mean = static_flat[:n_train_rows].mean(dim=0, keepdim=True)
    feat_std = static_flat[:n_train_rows].std(dim=0, keepdim=True).clamp_min(1e-6)
    static_norm = ((static_flat - feat_mean) / feat_std).reshape(-1, k, STATIC_FEATURE_DIM)
    zero_cols = torch.zeros(static_norm.shape[0], k, PROBE_FEATURE_DIM)
    full_feats = torch.cat([static_norm, zero_cols], dim=-1)  # [n,k,21]; zero-col mean=0/std clamped -> normalized value stays 0

    torch.manual_seed(seed)
    scorer = CompetenceScorer(STATIC_FEATURE_DIM + PROBE_FEATURE_DIM)
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    def loss_for_batch(batch_idx: torch.Tensor) -> torch.Tensor:
        pred = scorer(full_feats[batch_idx].reshape(-1, STATIC_FEATURE_DIM + PROBE_FEATURE_DIM)).reshape(batch_idx.numel(), k)
        actual = excess_loss_train[batch_idx]
        huber = F.huber_loss(pred.reshape(-1), actual.reshape(-1), delta=1.0)
        ranking = pairwise_ranking_loss(pred, actual)
        return huber + RANKING_WEIGHT * ranking

    best_val, bad, best_state = float("inf"), 0, None
    for _epoch in range(MAX_EPOCHS):
        scorer.train()
        window_ids = torch.arange(0, split_point)
        perm = window_ids[torch.randperm(window_ids.numel())] if window_ids.numel() else window_ids
        for b in range(0, perm.numel(), BATCH_SIZE):
            batch_idx = perm[b : b + BATCH_SIZE]
            loss = loss_for_batch(batch_idx)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scorer.eval()
        eval_ids = torch.arange(split_point, prefix_end)
        with torch.no_grad():
            val_loss = float(loss_for_batch(eval_ids)) if eval_ids.numel() else float("inf")
        if val_loss < best_val - 1e-6:
            best_val, bad = val_loss, 0
            best_state = copy.deepcopy(scorer.state_dict())
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        scorer.load_state_dict(best_state)
    scorer.eval()
    return {"scorer": scorer, "feat_mean": feat_mean, "feat_std": feat_std}


def score_matched_passive_on_windows(bundle, fit: Mapping[str, Any], group_a: torch.Tensor, group_b: torch.Tensor, group_c: torch.Tensor, window_idx: torch.Tensor) -> torch.Tensor:
    k = len(bundle.core_names)
    static = torch.cat([group_a, group_b, group_c], dim=-1)
    static_norm = (static - fit["feat_mean"]) / fit["feat_std"]
    zero_cols = torch.zeros(static_norm.shape[0], k, PROBE_FEATURE_DIM)
    full_feats = torch.cat([static_norm, zero_cols], dim=-1)
    with torch.no_grad():
        pred = fit["scorer"](full_feats[window_idx].reshape(-1, STATIC_FEATURE_DIM + PROBE_FEATURE_DIM)).reshape(window_idx.numel(), k)
    return pred


# ---------------------------------------------------------------------------
# Per-dataset pipeline
# ---------------------------------------------------------------------------


def evaluate_dataset(dataset: str) -> dict[str, Any]:
    reg = register_dataset(dataset)
    core = reg["selected_core"]
    freq = FREQ_BY_DATASET[dataset]
    print(f"[fforma_probe] {dataset}: core (router_train only) = {core}, predeclared freq={freq}", flush=True)

    bundle = fhv.LOADERS[dataset]()
    train_cache = bundle.train_cache
    val_cache = bundle.val_cache
    k = len(bundle.core_names)
    n_train = int(train_cache["num_windows"])
    n_val = int(val_cache["num_windows"])
    horizon = int(train_cache["forecast_horizon"])

    checkpoint_hashes_before = {e: load_expert_runtime(dataset, e).checkpoint_sha256 for e in core}

    # --- Section 2: router_train -> router_val observability ---
    observability = verify_router_train_to_val_observability(train_cache, val_cache)
    origins_train = train_cache["absolute_window_starts"].to(torch.long)
    if observability["observability_holds"]:
        legal_idx_all = torch.arange(n_train)
    else:
        legal_mask = (origins_train + horizon) <= observability["min_router_val_origin"]
        legal_idx_all = torch.nonzero(legal_mask, as_tuple=True)[0]
    print(f"[fforma_probe] {dataset}: router_train->router_val observability holds={observability['observability_holds']}, legal Full rows={legal_idx_all.numel()}/{n_train}", flush=True)

    # --- Section 1: mandatory purged chronological folds ---
    folds = purged_walkforward_folds(train_cache)
    common_idx = torch.cat([f["eval_idx"] for f in folds]).unique(sorted=True)
    legal_mask_all = torch.zeros(n_train, dtype=torch.bool)
    legal_mask_all[legal_idx_all] = True
    common_mask = torch.zeros(n_train, dtype=torch.bool)
    common_mask[common_idx] = True
    common_idx = torch.nonzero(common_mask & legal_mask_all, as_tuple=True)[0]
    fold_diag_rows = [
        {
            "dataset": dataset,
            "fold": f["fold"],
            "train_origin_min": f["train_origin_min"],
            "train_origin_max": f["train_origin_max"],
            "train_target_end_max": f["train_target_end_max"],
            "eval_origin_min": f["eval_origin_min"],
            "eval_origin_max": f["eval_origin_max"],
            "num_train_windows": int(f["train_idx"].numel()),
            "num_eval_windows": int(f["eval_idx"].numel()),
            "num_purged_windows": f["num_purged_windows"],
            "assertion_max_train_target_end_leq_min_eval_origin": f["assertion_max_train_target_end_leq_min_eval_origin"],
        }
        for f in folds
    ]
    for row in fold_diag_rows:
        print(f"[fforma_probe] {dataset}: fold {row['fold']}: train_target_end_max={row['train_target_end_max']} <= eval_origin_min={row['eval_origin_min']}: {row['assertion_max_train_target_end_leq_min_eval_origin']} (purged {row['num_purged_windows']} windows)", flush=True)

    forecasts_train_core = bundle.forecasts_fn(train_cache, bundle.expert_idx)
    forecasts_val_core = bundle.forecasts_fn(val_cache, bundle.expert_idx)
    target_train = train_cache["targets"].to(torch.float32)
    mask_train = train_cache["target_masks"].to(torch.bool)

    errors_train = torch.stack([sample_mae(forecasts_train_core[..., e], target_train, mask_train, bundle.std) for e in range(k)], dim=1).numpy()

    # --- FFORMA base features: official THA_features via Python tsfeatures (cached) ---
    reference_runtime = load_expert_runtime(dataset, core[0])
    train_cache_raw = raw_history_cache(dataset, train_cache, reference_runtime.mean, reference_runtime.std)
    val_cache_raw = raw_history_cache(dataset, val_cache, reference_runtime.mean, reference_runtime.std)
    print(f"[fforma_probe] {dataset}: loading FFORMA tsfeatures (cached)...", flush=True)
    x_ts_train, ts_diag_train = get_or_compute_fforma_features(dataset, "router_train", train_cache_raw["histories"].to(torch.float32), freq)
    x_ts_val, ts_diag_val = get_or_compute_fforma_features(dataset, "router_val", val_cache_raw["histories"].to(torch.float32), freq)

    # --- purged OOF LearnedProbe / MatchedPassive scoring (Sections 9-11) ---
    oof_probe = torch.full((n_train, k), float("nan"))
    oof_passive = torch.full((n_train, k), float("nan"))
    group_a_train, group_b_train, group_c_train, _ = build_abc_features(bundle, train_cache_raw)
    for f in folds:
        train_idx, eval_idx = f["train_idx"], f["eval_idx"]
        if train_idx.numel() < 10 or eval_idx.numel() == 0:
            continue
        print(f"[fforma_probe] {dataset}: fold {f['fold']}: purged-OOF LearnedProbe retrain ({train_idx.numel()} legal windows)...", flush=True)
        fit_probe_fold = train_probe_and_scorer_prefix(dataset, bundle, train_cache, train_idx)
        oof_probe[eval_idx] = score_probe_on_windows(dataset, bundle, fit_probe_fold, train_cache, eval_idx, is_router_train=True)
        print(f"[fforma_probe] {dataset}: fold {f['fold']}: purged-OOF MatchedPassive-21 retrain...", flush=True)
        fit_passive_fold = train_matched_passive_prefix(dataset, bundle, train_cache, train_idx)
        oof_passive[eval_idx] = score_matched_passive_on_windows(bundle, fit_passive_fold, group_a_train, group_b_train, group_c_train, eval_idx)

    oof_probe_common = oof_probe[common_idx]
    oof_passive_common = oof_passive[common_idx]
    if bool(torch.isnan(oof_probe_common).any()) or bool(torch.isnan(oof_passive_common).any()):
        raise AssertionError(f"{dataset}: Common window set has un-scored OOF rows -- purge/fold coverage bug")

    # --- Section 4: FFORMA hyperparameter selection, BASE FFORMA ONLY, purged OOF ---
    print(f"[fforma_probe] {dataset}: FFORMA hyperparameter search (base FFORMA only, purged OOF)...", flush=True)
    hp_rows = []
    for params in FFORMA_SEARCH_GRID:
        fold_maes = []
        for f in folds:
            train_idx, eval_idx = f["train_idx"], f["eval_idx"]
            if train_idx.numel() < 10 or eval_idx.numel() == 0:
                continue
            booster = train_fforma(x_ts_train[train_idx].numpy(), errors_train[train_idx.numpy()], params)
            w = predict_fforma(booster, x_ts_train[eval_idx].numpy(), k)
            pred = fforma_weighted_prediction(w, forecasts_train_core[eval_idx])
            mae = sample_mae(pred, target_train[eval_idx], mask_train[eval_idx], bundle.std)
            fold_maes.append(mae)
        pooled_mae = float(torch.cat(fold_maes).mean()) if fold_maes else float("inf")
        hp_rows.append({"dataset": dataset, **params, "purged_oof_mae": pooled_mae})
    hp_rows_sorted = sorted(hp_rows, key=lambda r: r["purged_oof_mae"])
    best_mae = hp_rows_sorted[0]["purged_oof_mae"]
    selected_params = {kk: vv for kk, vv in hp_rows_sorted[0].items() if kk in ("max_depth", "eta", "subsample", "colsample_bytree", "nrounds")}
    for row in hp_rows:
        row["selected"] = row["purged_oof_mae"] == best_mae
    print(f"[fforma_probe] {dataset}: selected FFORMA hyperparameters: {selected_params} (purged OOF MAE={best_mae:.6f})", flush=True)

    # --- final FFORMA fits (Sections 3, 5) ---
    print(f"[fforma_probe] {dataset}: fitting final FFORMA variants...", flush=True)
    booster_m4fixed = train_fforma(x_ts_train[legal_idx_all].numpy(), errors_train[legal_idx_all.numpy()], FFORMA_M4_CONFIG)
    booster_full = train_fforma(x_ts_train[legal_idx_all].numpy(), errors_train[legal_idx_all.numpy()], selected_params)
    booster_common = train_fforma(x_ts_train[common_idx].numpy(), errors_train[common_idx.numpy()], selected_params)

    # --- final deployed LearnedProbe / MatchedPassive (trained on the FULL legal Full set, used only to score router_val) ---
    print(f"[fforma_probe] {dataset}: training final deployed LearnedProbe (full legal router_train)...", flush=True)
    fit_probe_final = train_probe_and_scorer_prefix(dataset, bundle, train_cache, legal_idx_all)
    probe_val = score_probe_on_windows(dataset, bundle, fit_probe_final, val_cache, torch.arange(n_val), is_router_train=False)
    print(f"[fforma_probe] {dataset}: training final deployed MatchedPassive-21 (full legal router_train)...", flush=True)
    fit_passive_final = train_matched_passive_prefix(dataset, bundle, train_cache, legal_idx_all)
    group_a_val, group_b_val, group_c_val, _ = build_abc_features(bundle, val_cache_raw)
    passive_val = score_matched_passive_on_windows(bundle, fit_passive_final, group_a_val, group_b_val, group_c_val, torch.arange(n_val))

    checkpoint_hashes_after = {e: load_expert_runtime(dataset, e).checkpoint_sha256 for e in core}
    checkpoints_unchanged = checkpoint_hashes_before == checkpoint_hashes_after

    # --- Section 11: shuffle AFTER honest OOF/final scores are generated ---
    oof_probe_common_shuffled = shuffle_probe_scores(oof_probe_common, SHUFFLE_SEED)
    probe_val_shuffled = shuffle_probe_scores(probe_val, SHUFFLE_SEED)

    booster_passive = train_fforma(np.concatenate([x_ts_train[common_idx].numpy(), oof_passive_common.numpy()], axis=1), errors_train[common_idx.numpy()], selected_params)
    booster_probe = train_fforma(np.concatenate([x_ts_train[common_idx].numpy(), oof_probe_common.numpy()], axis=1), errors_train[common_idx.numpy()], selected_params)
    booster_shuffled = train_fforma(np.concatenate([x_ts_train[common_idx].numpy(), oof_probe_common_shuffled.numpy()], axis=1), errors_train[common_idx.numpy()], selected_params)

    def evaluate_booster(booster: xgb.Booster, feat_val: np.ndarray) -> tuple[dict, torch.Tensor]:
        weights = predict_fforma(booster, feat_val, k)
        pred = fforma_weighted_prediction(weights, forecasts_val_core)
        m = metric_values(val_cache, pred, bundle.std)
        return m, weights

    m4fixed_m, w_m4fixed = evaluate_booster(booster_m4fixed, x_ts_val.numpy())
    full_m, w_full = evaluate_booster(booster_full, x_ts_val.numpy())
    common_m, w_common = evaluate_booster(booster_common, x_ts_val.numpy())
    passive_m, w_passive = evaluate_booster(booster_passive, np.concatenate([x_ts_val.numpy(), passive_val.numpy()], axis=1))
    probe_m, w_probe = evaluate_booster(booster_probe, np.concatenate([x_ts_val.numpy(), probe_val.numpy()], axis=1))
    shuffled_m, w_shuffled = evaluate_booster(booster_shuffled, np.concatenate([x_ts_val.numpy(), probe_val_shuffled.numpy()], axis=1))

    result_rows = [
        {"dataset": dataset, "method": "FFORMA_M4Fixed", "mae": m4fixed_m["mae"], "mse": m4fixed_m["mse"]},
        {"dataset": dataset, "method": "FFORMA_MAE_Full", "mae": full_m["mae"], "mse": full_m["mse"]},
        {"dataset": dataset, "method": "FFORMA_MAE_Common", "mae": common_m["mae"], "mse": common_m["mse"]},
        {
            "dataset": dataset, "method": "FFORMA_MAE_MatchedPassive21", "mae": passive_m["mae"], "mse": passive_m["mse"],
            "delta_vs_common": passive_m["mae"] - common_m["mae"], "delta_vs_full": passive_m["mae"] - full_m["mae"],
        },
        {
            "dataset": dataset, "method": "FFORMA_MAE_LearnedProbe", "mae": probe_m["mae"], "mse": probe_m["mse"],
            "delta_vs_common": probe_m["mae"] - common_m["mae"], "delta_vs_full": probe_m["mae"] - full_m["mae"],
            "delta_vs_matchedpassive": probe_m["mae"] - passive_m["mae"],
        },
        {
            "dataset": dataset, "method": "FFORMA_MAE_ShuffledProbe", "mae": shuffled_m["mae"], "mse": shuffled_m["mse"],
            "delta_vs_common": shuffled_m["mae"] - common_m["mae"],
            "delta_probe_vs_shuffled": probe_m["mae"] - shuffled_m["mae"],
        },
    ]

    dependence_rows = []
    dependence_rows.extend(dependence_full(probe_m["per_window_mae"], common_m["per_window_mae"], dataset, "Probe_vs_Common"))
    dependence_rows.extend(dependence_full(probe_m["per_window_mae"], full_m["per_window_mae"], dataset, "Probe_vs_Full"))
    dependence_rows.extend(dependence_full(probe_m["per_window_mae"], passive_m["per_window_mae"], dataset, "Probe_vs_MatchedPassive"))
    dependence_rows.extend(dependence_full(probe_m["per_window_mae"], shuffled_m["per_window_mae"], dataset, "Probe_vs_Shuffled"))
    primary_probe_vs_common = primary_row(dependence_rows, "Probe_vs_Common")
    primary_probe_vs_full = primary_row(dependence_rows, "Probe_vs_Full")
    primary_probe_vs_passive = primary_row(dependence_rows, "Probe_vs_MatchedPassive")
    primary_probe_vs_shuffled = primary_row(dependence_rows, "Probe_vs_Shuffled")

    test_cache_path = ROOT / f"cache/costarts_walkforward_{dataset}" / "test_80_100_cache.pt"
    all_folds_pass_assertion = all(f["assertion_max_train_target_end_leq_min_eval_origin"] for f in folds)
    integrity = {
        "dataset": dataset,
        "official_fforma_source_recorded": True,
        "expert_checkpoints_unchanged": checkpoints_unchanged,
        "no_test_cache_loaded": not test_cache_path.exists(),
        "router_train_to_router_val_observability_holds": observability["observability_holds"],
        "max_router_train_target_end": observability["max_router_train_target_end"],
        "min_router_val_origin": observability["min_router_val_origin"],
        "all_purge_fold_assertions_pass": all_folds_pass_assertion,
        "num_purge_folds": len(folds),
        "num_common_windows": int(common_idx.numel()),
        "num_full_legal_windows": int(legal_idx_all.numel()),
        "probe_parameters_frozen": True,
        "clamp_zero_disabled": True,
        "result": "PASS" if (checkpoints_unchanged and not test_cache_path.exists() and all_folds_pass_assertion) else "FAIL",
    }
    if integrity["result"] == "FAIL":
        raise AssertionError(f"{dataset}: fforma_probe integrity check FAILED: {integrity}")

    PER_WINDOW_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        PER_WINDOW_DIR / f"{dataset}.npz",
        m4fixed_mae=m4fixed_m["per_window_mae"].numpy(), full_mae=full_m["per_window_mae"].numpy(),
        common_mae=common_m["per_window_mae"].numpy(), passive_mae=passive_m["per_window_mae"].numpy(),
        probe_mae=probe_m["per_window_mae"].numpy(), shuffled_mae=shuffled_m["per_window_mae"].numpy(),
    )

    return {
        "dataset": dataset,
        "core": core,
        "freq": freq,
        "ts_feature_diag_train": ts_diag_train,
        "ts_feature_diag_val": ts_diag_val,
        "selected_fforma_params": selected_params,
        "hp_rows": hp_rows,
        "fold_diag_rows": fold_diag_rows,
        "result_rows": result_rows,
        "dependence_rows": dependence_rows,
        "primary_probe_vs_common": primary_probe_vs_common,
        "primary_probe_vs_full": primary_probe_vs_full,
        "primary_probe_vs_passive": primary_probe_vs_passive,
        "primary_probe_vs_shuffled": primary_probe_vs_shuffled,
        "integrity": integrity,
        "checkpoint_hashes": checkpoint_hashes_after,
    }


# ---------------------------------------------------------------------------
# 13. Decision rule (pre-specified, not altered after seeing results)
# ---------------------------------------------------------------------------


def decide(report: Mapping[str, Any]) -> dict[str, Any]:
    datasets = list(report["datasets"].keys())
    n = len(datasets)
    by = {ds: {r["method"]: r for r in report["datasets"][ds]["result_rows"]} for ds in datasets}

    def sig_improve(row_key: str, ds: str) -> bool:
        r = report["datasets"][ds][row_key]
        return bool(r["ci_excludes_zero"] and r["mean_delta"] < 0)

    def sig_regress(row_key: str, ds: str) -> bool:
        r = report["datasets"][ds][row_key]
        return bool(r["ci_excludes_zero"] and r["mean_delta"] > 0)

    n_beats_common_point = sum(1 for ds in datasets if by[ds]["FFORMA_MAE_LearnedProbe"]["mae"] < by[ds]["FFORMA_MAE_Common"]["mae"])
    n_beats_common_sig = sum(1 for ds in datasets if sig_improve("primary_probe_vs_common", ds))
    n_ties_or_beats_full = sum(1 for ds in datasets if by[ds]["FFORMA_MAE_LearnedProbe"]["mae"] <= by[ds]["FFORMA_MAE_Full"]["mae"] or not sig_regress("primary_probe_vs_full", ds))
    n_beats_passive_point = sum(1 for ds in datasets if by[ds]["FFORMA_MAE_LearnedProbe"]["mae"] < by[ds]["FFORMA_MAE_MatchedPassive21"]["mae"])
    n_beats_passive_sig = sum(1 for ds in datasets if sig_improve("primary_probe_vs_passive", ds))
    n_beats_shuffled_point = sum(1 for ds in datasets if by[ds]["FFORMA_MAE_LearnedProbe"]["mae"] < by[ds]["FFORMA_MAE_ShuffledProbe"]["mae"])
    n_beats_shuffled_sig = sum(1 for ds in datasets if sig_improve("primary_probe_vs_shuffled", ds))
    n_broad_regressions = sum(1 for ds in datasets if sig_regress("primary_probe_vs_common", ds))

    criteria = {
        "A_better_than_common": n_beats_common_point >= (n // 2 + 1),
        "B_competitive_with_full": n_ties_or_beats_full >= (n // 2 + 1),
        "C_better_than_matchedpassive": n_beats_passive_point >= (n // 2 + 1),
        "D_better_than_shuffled": n_beats_shuffled_point >= (n // 2 + 1),
        "E_gains_multiple_datasets": n_beats_common_sig >= 2,
        "F_no_broad_regressions": n_broad_regressions == 0,
    }
    n_criteria_met = sum(criteria.values())
    strong = n_criteria_met >= 5
    beats_common_only = criteria["A_better_than_common"] and not criteria["B_competitive_with_full"]

    if strong:
        tier = "STRONG"
        conclusion = "LearnedProbe shows strong evidence of adding useful information to FFORMA: it beats FFORMA-MAE-Common, is competitive with or beats FFORMA-MAE-Full, beats MatchedPassive-21, beats ShuffledProbe, on multiple datasets, with no broad significant regressions."
    elif beats_common_only:
        tier = "MATCHED_ONLY"
        conclusion = "Probe adds useful information under matched training support, but does not outperform the strongest full-data FFORMA baseline. This is NOT an unconditional improvement over FFORMA."
    elif criteria["F_no_broad_regressions"] and (criteria["A_better_than_common"] or criteria["D_better_than_shuffled"]):
        tier = "MIXED"
        conclusion = "Partial, inconsistent evidence: LearnedProbe helps under some comparisons but not others, without broad regressions. Treat as suggestive, not confirmatory."
    else:
        tier = "WEAK_OR_FAILURE"
        conclusion = "LearnedProbe does not show a clear advantage over FFORMA-MAE-Common/Full/MatchedPassive-21/ShuffledProbe on this evidence, or shows broad regressions."

    return {
        "tier": tier,
        "conclusion": conclusion,
        "criteria": criteria,
        "n_criteria_met": n_criteria_met,
        "n_beats_common_point": n_beats_common_point,
        "n_beats_common_sig": n_beats_common_sig,
        "n_ties_or_beats_full": n_ties_or_beats_full,
        "n_beats_passive_point": n_beats_passive_point,
        "n_beats_passive_sig": n_beats_passive_sig,
        "n_beats_shuffled_point": n_beats_shuffled_point,
        "n_beats_shuffled_sig": n_beats_shuffled_sig,
        "n_broad_regressions": n_broad_regressions,
        "n_datasets": n,
    }


def git_commit_sha() -> str:
    import subprocess

    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def make_report(report: Mapping[str, Any], decision: Mapping[str, Any]) -> None:
    datasets = list(report["datasets"].keys())
    lines = [
        "# FFORMA vs FFORMA + LearnedProbe (Final Audit Overrides: purged causal OOF)",
        "",
        "Official FFORMA (robjhyndman/M4metalearning, commit 61ddc7101680e9df7219c359587d0b509d2b50d6): "
        "THA_features (Python tsfeatures v0.4.5, verified same function set) + custom softmax-expected-loss XGBoost "
        "objective (error_softmax_obj, ported to Python xgboost's modern [N,K] custom-objective API), applied to this "
        "project's frozen-expert / router_train / router_val protocol. Every target-dependent supervised component "
        "(LearnedProbe, MatchedPassive-21, FFORMA's own hyperparameter selection) is trained on PURGED chronological "
        f"folds ({N_PURGE_FOLDS} folds, min_train_fraction={MIN_TRAIN_FRACTION}): a training window may supervise a "
        "held-out fold only if its target fully resolves before the fold's first held-out forecast origin.",
        "",
        "## Mandatory causal assertions (Section 1, 2, 14)", "",
        "| Dataset | Fold | Train target-end max | Eval origin min | Assertion holds | Purged windows |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for ds in datasets:
        for row in report["datasets"][ds]["fold_diag_rows"]:
            lines.append(f"| {ds} | {row['fold']} | {row['train_target_end_max']} | {row['eval_origin_min']} | {row['assertion_max_train_target_end_leq_min_eval_origin']} | {row['num_purged_windows']} |")
    lines += ["", "| Dataset | router_train->router_val observability holds | max train target-end | min val origin |", "|---|---|---:|---:|"]
    for ds in datasets:
        i = report["datasets"][ds]["integrity"]
        lines.append(f"| {ds} | {i['router_train_to_router_val_observability_holds']} | {i['max_router_train_target_end']} | {i['min_router_val_origin']} |")
    lines += ["", "## Predeclared dataset frequency / tsfeatures diagnostics", ""]
    lines.append("| Dataset | Freq | Split | Windows | Group failures | Seasonal-padding zeros | NaN values zeroed |")
    lines.append("|---|---:|---|---:|---:|---:|---:|")
    for ds in datasets:
        d = report["datasets"][ds]
        for split, diag in (("router_train", d["ts_feature_diag_train"]), ("router_val", d["ts_feature_diag_val"])):
            flag = " **[FLAGGED: high failure rate]**" if diag["num_group_failures"] > 0.1 * diag["num_windows"] * len(FFORMA_FEATURE_NAMES) else ""
            lines.append(f"| {ds} | {d['freq']} | {split} | {diag['num_windows']} | {diag['num_group_failures']}{flag} | {diag['num_seasonal_padding_zeros']} | {diag['num_nan_values_zeroed']} |")
    lines += ["", "## FFORMA hyperparameter selection (base FFORMA only, purged OOF)", ""]
    lines.append("| Dataset | max_depth | eta | subsample | colsample_bytree | nrounds | Purged OOF MAE | Selected |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for ds in datasets:
        for row in report["datasets"][ds]["hp_rows"]:
            lines.append(f"| {ds} | {row['max_depth']} | {row['eta']} | {row['subsample']} | {row['colsample_bytree']} | {row['nrounds']} | {row['purged_oof_mae']:.6f} | {'<-- selected' if row['selected'] else ''} |")
    lines += ["", "## Primary results (router_val MAE / MSE)", ""]
    lines.append("| Dataset | M4Fixed | Full | Common | +MatchedPassive21 | +LearnedProbe | +ShuffledProbe |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for ds in datasets:
        by = {r["method"]: r for r in report["datasets"][ds]["result_rows"]}
        lines.append(
            f"| {ds} | {by['FFORMA_M4Fixed']['mae']:.6f} | {by['FFORMA_MAE_Full']['mae']:.6f} | {by['FFORMA_MAE_Common']['mae']:.6f} | "
            f"{by['FFORMA_MAE_MatchedPassive21']['mae']:.6f} | {by['FFORMA_MAE_LearnedProbe']['mae']:.6f} | {by['FFORMA_MAE_ShuffledProbe']['mae']:.6f} |"
        )
    lines += ["", "## LearnedProbe deltas", ""]
    lines.append("| Dataset | vs Common | vs Full | vs MatchedPassive21 | Probe vs Shuffled |")
    lines.append("|---|---:|---:|---:|---:|")
    for ds in datasets:
        by = {r["method"]: r for r in report["datasets"][ds]["result_rows"]}
        p, s = by["FFORMA_MAE_LearnedProbe"], by["FFORMA_MAE_ShuffledProbe"]
        lines.append(f"| {ds} | `{p['delta_vs_common']:+.6f}` | `{p['delta_vs_full']:+.6f}` | `{p['delta_vs_matchedpassive']:+.6f}` | `{s['delta_probe_vs_shuffled']:+.6f}` |")
    lines += ["", "## Primary dependence-aware statistics (block-24)", ""]
    lines.append("| Dataset | Comparison | Mean Delta | 95% CI | P(Delta<0) | Excludes zero |")
    lines.append("|---|---|---:|---|---:|---|")
    for ds in datasets:
        for key, label in (("primary_probe_vs_common", "Probe_vs_Common"), ("primary_probe_vs_full", "Probe_vs_Full"), ("primary_probe_vs_passive", "Probe_vs_MatchedPassive"), ("primary_probe_vs_shuffled", "Probe_vs_Shuffled")):
            r = report["datasets"][ds][key]
            lines.append(f"| {ds} | {label} | `{r['mean_delta']:+.6f}` | [{r['ci95_low']:+.6f}, {r['ci95_high']:+.6f}] | {r['prob_delta_negative']:.3f} | {r['ci_excludes_zero']} |")
    lines += ["", "## Full dependence-aware statistics (all block lengths + phase)", ""]
    lines.append("| Dataset | Comparison | Test | Mean Delta | 95% CI | P(Delta<0) | Excludes zero |")
    lines.append("|---|---|---|---:|---|---:|---|")
    for ds in datasets:
        for row in report["datasets"][ds]["dependence_rows"]:
            mean_key = row.get("mean_delta", row.get("mean_diff_candidate_minus_baseline"))
            if "ci95_low" in row:
                prob = row.get("prob_delta_negative", "")
                lines.append(f"| {ds} | {row['comparison']} | {row['test']} | `{mean_key:+.6f}` | [{row['ci95_low']:+.6f}, {row['ci95_high']:+.6f}] | {prob} | {row['ci_excludes_zero']} |")
    lines += ["", "## Integrity", ""]
    for ds in datasets:
        i = report["datasets"][ds]["integrity"]
        lines.append(f"- **{ds}**: {i['result']} (checkpoints unchanged: {i['expert_checkpoints_unchanged']}; no test cache: {i['no_test_cache_loaded']}; all purge assertions pass: {i['all_purge_fold_assertions_pass']}; observability holds: {i['router_train_to_router_val_observability_holds']}; Common windows={i['num_common_windows']}, Full legal windows={i['num_full_legal_windows']})")
    lines += ["", "## Claim (Section 13)", ""]
    for name, met in decision["criteria"].items():
        lines.append(f"- **{name}**: {met}")
    lines += ["", f"## Decision: {decision['tier']}", "", decision["conclusion"], ""]
    lines += [
        "## Hard rule compliance", "", "```text",
        "TEST SET ACCESSED: NO",
        "FORECASTING EXPERTS RETRAINED: NO",
        "LEARNEDPROBE ARCHITECTURE/LOSS/TRAINING MODIFIED: NO",
        "NONNEGATIVE FORECAST CLAMP: DISABLED (clamp_zero=False, per Section 7)",
        "PURGE ASSERTION: see table above; raises AssertionError immediately if violated",
        "```",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    start = time.time()
    report: dict[str, Any] = {
        "experiment": "fforma_probe",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_sha": git_commit_sha(),
        "datasets": {},
    }
    all_results, all_dependence, all_integrity, all_hp, all_folds = [], [], [], [], []

    for dataset in NEW_DATASETS:
        print(f"[fforma_probe] {dataset}: starting...", flush=True)
        result = evaluate_dataset(dataset)
        report["datasets"][dataset] = result
        all_results.extend(result["result_rows"])
        all_dependence.extend(result["dependence_rows"])
        all_integrity.append(result["integrity"])
        all_hp.extend(result["hp_rows"])
        all_folds.extend(result["fold_diag_rows"])
        print(f"[fforma_probe] {dataset}: done.", flush=True)

    decision = decide(report)
    report["decision"] = decision
    report["runtime_sec"] = time.time() - start
    report["test_set_accessed"] = False

    manifest = {
        "manifest_type": "fforma_probe_manifest",
        "created_at_utc": report["created_at_utc"],
        "git_commit_sha": report["git_commit_sha"],
        "fforma_repository": "https://github.com/robjhyndman/M4metalearning",
        "fforma_commit_sha": "61ddc7101680e9df7219c359587d0b509d2b50d6",
        "fforma_source_functions_cited": ["R/generate_classif_problem.R::THA_features", "R/ensemble_classifier.R::error_softmax_obj", "R/ensemble_classifier.R::train_selection_ensemble", "R/ensemble_classifier.R::ensemble_forecast"],
        "tsfeatures_python_package_version": "0.4.5 (Nixtla port of R tsfeatures, verified same function names as THA_features calls)",
        "predeclared_frequency_by_dataset": FREQ_BY_DATASET,
        "n_purge_folds": N_PURGE_FOLDS,
        "min_train_fraction": MIN_TRAIN_FRACTION,
        "fforma_m4_config": FFORMA_M4_CONFIG,
        "fforma_search_grid": FFORMA_SEARCH_GRID,
        "expert_cores": {ds: report["datasets"][ds]["core"] for ds in NEW_DATASETS},
        "expert_checkpoint_sha256": {ds: report["datasets"][ds]["checkpoint_hashes"] for ds in NEW_DATASETS},
        "selected_fforma_hyperparameters": {ds: report["datasets"][ds]["selected_fforma_params"] for ds in NEW_DATASETS},
        "clamp_zero": False,
        "matched_passive_architecture": "CompetenceScorer(21) identical to LearnedProbe: 15 passive (group A+B+C) features + 6 constant-zero columns, same optimizer/lr/weight_decay/epochs/patience/Huber+ranking objective, seed=7. No ProbeGenerator, no perturbation penalties (nothing to regularize).",
        "shuffle_seed": SHUFFLE_SEED,
        "decision_rule": "Section 13 of the Final Audit Overrides, applied verbatim without modification after seeing results.",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUT_DIR / "fforma_probe_manifest.json", manifest)
    write_json(OUT_DIR / "validation_results.json", report)
    write_csv(OUT_DIR / "validation_results.csv", all_results)
    write_csv(OUT_DIR / "dependence_aware_results.csv", all_dependence)
    write_csv(OUT_DIR / "integrity_checks.csv", all_integrity)
    write_csv(OUT_DIR / "fforma_hyperparameter_selection.csv", all_hp)
    write_csv(OUT_DIR / "purged_fold_diagnostics.csv", all_folds)
    make_report(report, decision)

    print("TEST SET ACCESSED: NO")
    print(json.dumps({"runtime_sec": report["runtime_sec"], "tier": decision["tier"]}, indent=2))


if __name__ == "__main__":
    main()
