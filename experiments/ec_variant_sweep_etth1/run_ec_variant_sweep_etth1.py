"""Expert-Choice variant sweep on Window-Dependent EC, ETTh1 ONLY.

Development experiment restricted to ETTh1. Router_val is NEVER loaded for
any prediction/metric/selection in this file: the only evidence used is
strict chronological router_train OOF, exactly the same causal 4-fold
protocol as `window_dependent_expert_choice_hv` (same warmup fraction, same
full-horizon-observability legality rule, same seed, same frozen K=3 core
`PatchTST+iTransformer+TimesNet`, same checkpoints).

Starting point: the F2_local scorer (cell-local forecast features + per-
variable history features + H/V/expert identity embeddings + static_gain
scalar; NO global-history features), which `feature_ablation_affinity_
weighted_ec` found beats the full-feature model on ETTh1 OOF MAE. The
scorer architecture/training procedure is copied UNCHANGED from
`feature_ablation_affinity_weighted_ec.FlexibleResidualScorer` /
`train_scorer_flexible` with `enabled_groups={"cell","local"}`.

Only the DOWNSTREAM routing mechanism is varied, on top of the identical
trained F2_local raw-score tensor per fold (no retraining across the grid):

1. Capacity factor CF in {0.5, 1.0, 2.0}: capacity per expert
   C = max(1, round(CF * H*V / E)).
2. Assignment rule:
   - "unrestricted": current independent per-expert top-C selection
     (`unrestricted_ec_claims`, a direct capacity-parameterized copy of
     `window_dependent_expert_choice_hv.dynamic_ec_claims` -- a cell may be
     claimed by any number of experts, 0..E).
   - "max2": the same top-C competition, but no cell may be claimed by more
     than 2 experts. Implemented as a single deterministic global greedy
     assignment over all (cell, expert) pairs ordered by affinity score
     (ties broken by lower flattened pair index), accepting a pair only if
     both its expert has remaining capacity AND its cell has not yet
     reached 2 claims (`capped_ec_claims`). Experts may end up under their
     nominal capacity if the constraint binds; this is reported, not
     patched.
3. Scoring normalization:
   - "existing": exactly `window_dependent_expert_choice_hv.raw_to_affinity`
     -- one FIT-ONLY scalar (mean, std) computed over the whole raw-score
     tensor on the fold's fit windows, then softmax over experts.
   - "expert_relative": a FIT-ONLY per-expert (mean[e], std[e]) computed
     over the fold's fit windows, so each expert's raw score is standardized
     against its OWN distribution before the softmax-over-experts step
     (`raw_to_affinity_expert_relative`). This is the literal reading of
     "scores for a cell are normalized across experts" with the
     normalization statistics themselves made expert-relative rather than
     one shared scalar.

Grid: 3 CF x 2 assignment x 2 scoring = 12 configurations. The combination
CF=1.0 / unrestricted / existing is declared BEFORE any results are
inspected as "current Expert Choice baseline" (it is the direct F2_local
analogue of the existing window_dependent_expert_choice_hv dynamic_ec_cf1
method; the historical F3_full-feature number is also reported for
context, not used as the primary baseline, since only routing-mechanism
axes should vary against a fixed scorer for a clean ablation).

Every configuration is compared against: matched Dynamic Token Choice
(same scoring variant, argmax instead of top-C), the current Expert Choice
baseline above, and the Frozen Dense Ensemble (equal average of the 3
frozen experts, no routing at all). Combination rule for claimed cells is
the UNMODIFIED equal-average-of-claiming-experts rule with equal-ensemble
zero-claim fallback (`dynamic_prediction_from_claims`, copied verbatim from
window_dependent_expert_choice_hv) -- no affinity-weighted fusion, to keep
this sweep isolated to capacity/assignment/scoring only.

TEST SET ACCESSED: NO. TEST CACHE LOADED: NO. TEST METRICS COMPUTED: NO.
ROUTER_VAL ACCESSED: NO. UNTOUCHED DATA ACCESSED: NO. OTHER DATASETS: NONE.

Per explicit instruction: do not change this method after seeing OOF
results, except to fix implementation bugs.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.window_dependent_expert_choice_hv.run_window_dependent_expert_choice_hv as wdec  # noqa: E402
import experiments.feature_ablation_affinity_weighted_ec.run_feature_ablation as fab  # noqa: E402
from experiments.frozen_hv_costar.run_frozen_hv_costar import LOADERS  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
DATASET = "ETTh1"
BLOCK_LENGTH = wdec.BLOCK_LENGTH
PHASE_K = wdec.PHASE_K
BOOTSTRAP_SAMPLES = wdec.BOOTSTRAP_SAMPLES
BOOTSTRAP_SEED = wdec.BOOTSTRAP_SEED
DEVICE = wdec.DEVICE

F2_LOCAL_GROUPS = frozenset({"cell", "local"})
CAPACITY_FACTORS = (0.5, 1.0, 2.0)
ASSIGNMENT_RULES = ("unrestricted", "max2")
SCORING_VARIANTS = ("existing", "expert_relative")
MAX_CLAIMS_PER_CELL = 2
BASELINE_CONFIG = (1.0, "unrestricted", "existing")  # predeclared "current Expert Choice baseline"


def write_json(path: Path, obj: Any) -> None:
    wdec.write_json(path, obj)


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    wdec.write_csv_rows(path, rows)


def jsonable(value: Any) -> Any:
    return wdec.jsonable(value)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config_key(cf: float, assignment: str, scoring: str) -> str:
    return f"cf{cf}_{assignment}_{scoring}"


# ---------------------------------------------------------------------------
# Capacity-parameterized assignment operators.
# ---------------------------------------------------------------------------


def capacity_for(cf: float, m: int, e: int) -> int:
    return max(1, min(m, int(round(cf * m / e))))


def unrestricted_ec_claims(affinity: torch.Tensor, capacity: int) -> torch.Tensor:
    """Direct capacity-parameterized copy of wdec.dynamic_ec_claims: each
    expert independently claims its top-`capacity` cells by affinity, no
    limit on how many experts may claim the same cell. Deterministic
    tie-break: higher affinity, then lower flattened H*V cell index."""
    n, h, v, e = affinity.shape
    m = h * v
    flat = affinity.reshape(n, m, e).to(torch.float64)
    cell_index = torch.arange(m, dtype=torch.float64).view(1, m, 1)
    tie_break_key = flat * 1.0e9 - cell_index
    claim = torch.zeros((n, m, e), dtype=torch.bool)
    for expert in range(e):
        top = torch.topk(tie_break_key[:, :, expert], k=capacity, dim=1, largest=True).indices
        claim[:, :, expert].scatter_(1, top, True)
    return claim.view(n, h, v, e)


def capped_ec_claims(affinity: torch.Tensor, capacity: int, max_per_cell: int) -> torch.Tensor:
    """Deterministic global greedy assignment enforcing a per-cell claim cap.

    All (cell, expert) pairs are ranked once by affinity (ties broken by
    lower flattened pair index = cell*E + expert, so ties prefer the
    lower-index cell then the lower-index expert). Pairs are then accepted
    in rank order, per window, whenever BOTH the expert has not yet reached
    `capacity` claims AND the cell has not yet reached `max_per_cell`
    claims. The outer loop runs over pair RANKS (H*V*E, independent of CF),
    with every window processed in parallel at each rank -- not a loop over
    windows -- so this stays fast regardless of how many OOF windows there
    are. Experts may end below `capacity` if the constraint binds; this is
    a real, reported outcome, not corrected for.
    """
    n, h, v, e = affinity.shape
    m = h * v
    flat = affinity.reshape(n, m * e).to(torch.float64)
    pair_idx = torch.arange(m * e, dtype=torch.float64).view(1, -1)
    key = flat * 1.0e9 - pair_idx
    order = torch.argsort(key, dim=1, descending=True)  # [n, m*e]

    cell_of_pair = torch.arange(m * e) // e
    expert_of_pair = torch.arange(m * e) % e
    ordered_cell = cell_of_pair.view(1, -1).expand(n, -1).gather(1, order)
    ordered_expert = expert_of_pair.view(1, -1).expand(n, -1).gather(1, order)

    expert_count = torch.zeros(n, e, dtype=torch.long)
    cell_count = torch.zeros(n, m, dtype=torch.long)
    claim_flat = torch.zeros(n, m, e, dtype=torch.bool)
    idx_n = torch.arange(n)

    for rank in range(m * e):
        c = ordered_cell[:, rank]
        ex = ordered_expert[:, rank]
        cur_ex_cnt = expert_count[idx_n, ex]
        cur_cell_cnt = cell_count[idx_n, c]
        eligible = (cur_ex_cnt < capacity) & (cur_cell_cnt < max_per_cell)
        sel = idx_n[eligible]
        if sel.numel() > 0:
            claim_flat[sel, c[sel], ex[sel]] = True
            expert_count[sel, ex[sel]] += 1
            cell_count[sel, c[sel]] += 1

    return claim_flat.view(n, h, v, e)


def assign_claims(affinity: torch.Tensor, cf: float, assignment: str) -> tuple[torch.Tensor, int]:
    n, h, v, e = affinity.shape
    m = h * v
    capacity = capacity_for(cf, m, e)
    if assignment == "unrestricted":
        claim = unrestricted_ec_claims(affinity, capacity)
    elif assignment == "max2":
        claim = capped_ec_claims(affinity, capacity, MAX_CLAIMS_PER_CELL)
    else:
        raise ValueError(assignment)
    return claim, capacity


# ---------------------------------------------------------------------------
# Expert-relative scoring normalization.
# ---------------------------------------------------------------------------


def fit_only_calibration_per_expert(
    fit: wdec.TrainedScorer,
    global_feat: torch.Tensor,
    local_feat: torch.Tensor,
    forecasts_full: torch.Tensor,
    histories_full: torch.Tensor,
    std: torch.Tensor,
    legal_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Same fit-only-window convention as wdec.fit_only_calibration, but
    keeps per-expert mean/std instead of collapsing to one scalar."""
    total: torch.Tensor | None = None
    total_sq: torch.Tensor | None = None
    count = 0
    for lo in range(0, int(legal_idx.numel()), wdec.STAT_CHUNK):
        idx = legal_idx[lo : lo + wdec.STAT_CHUNK]
        raw = wdec.score_windows(fit, global_feat, local_feat, forecasts_full, histories_full, std, idx)
        raw64 = raw.to(torch.float64)
        if total is None:
            e = raw.shape[-1]
            total = torch.zeros(e, dtype=torch.float64)
            total_sq = torch.zeros(e, dtype=torch.float64)
        total += raw64.sum(dim=(0, 1, 2))
        total_sq += (raw64**2).sum(dim=(0, 1, 2))
        count += raw.shape[0] * raw.shape[1] * raw.shape[2]
    mean = total / max(count, 1)
    var = total_sq / max(count, 1) - mean**2
    std_val = var.clamp_min(1e-12).sqrt()
    return mean, std_val


def raw_to_affinity_expert_relative(raw_score: torch.Tensor, mean_e: torch.Tensor, std_e: torch.Tensor) -> torch.Tensor:
    z = (raw_score - mean_e.view(1, 1, 1, -1)) / std_e.clamp_min(1e-8).view(1, 1, 1, -1)
    return torch.softmax(z / wdec.AFFINITY_TEMPERATURE, dim=-1)


# ---------------------------------------------------------------------------
# Claim distribution / expert utilization stats.
# ---------------------------------------------------------------------------


def claim_stats(claim_mask: torch.Tensor, core_names: Sequence[str], intended_capacity: int) -> dict[str, Any]:
    n, h, v, e = claim_mask.shape
    counts = claim_mask.sum(dim=-1)  # [n,h,v]
    total_cells = float(counts.numel())
    dist = {}
    for k in range(e + 1):
        pct = float((counts == k).to(torch.float32).sum() / total_cells * 100.0)
        if k == 0:
            dist["zero_claim_cells_pct"] = pct
        elif k == 1:
            dist["one_claim_cells_pct"] = pct
        elif k == 2:
            dist["two_claim_cells_pct"] = pct
        else:
            dist[f"claim_cells_{k}_pct"] = pct
    dist["more_than_two_claim_cells_pct"] = float((counts > 2).to(torch.float32).sum() / total_cells * 100.0)

    per_window_capacity = claim_mask.sum(dim=(1, 2))  # [n,e]
    expert_util = {}
    for i, name in enumerate(core_names):
        actual_total = int(per_window_capacity[:, i].sum())
        intended_total = intended_capacity * n
        expert_util[name] = {
            "actual_total_claims": actual_total,
            "intended_total_claims": intended_total,
            "utilization_pct": float(actual_total / max(intended_total, 1) * 100.0),
            "mean_claims_per_window": float(per_window_capacity[:, i].to(torch.float32).mean()),
        }
    return {"claim_distribution_pct": dist, "expert_utilization": expert_util}


# ---------------------------------------------------------------------------
# OOF fold loop: train F2_local scorer once per fold, capture BOTH
# calibration statistics needed for the two scoring variants. No retraining
# happens anywhere else in this file -- CF/assignment/scoring are all
# downstream, cheap, post-hoc computations on this single frozen raw-score
# tensor per fold.
# ---------------------------------------------------------------------------


def run_oof(dataset: str, ckpt_dir: Path) -> dict[str, Any]:
    bundle = LOADERS[dataset]()
    train_role = wdec.validate_cache_role(bundle.train_cache, "router_train")
    val_role = wdec.validate_cache_role(bundle.val_cache, "router_val")
    before_hashes = wdec.checkpoint_hashes(dataset, bundle.core_names)
    oos_provenance = wdec.verify_router_train_out_of_sample(bundle.train_cache)

    horizon = int(bundle.train_cache["forecast_horizon"])
    variables = int(bundle.val_cache["num_features"])
    num_experts = len(bundle.expert_idx)
    n_train = int(bundle.train_cache["num_windows"])
    train_starts = bundle.train_cache["absolute_window_starts"].to(torch.long)

    train_gain = wdec.full_gain_tensor(bundle, bundle.train_cache)
    train_global, train_local = wdec.global_local_features(bundle.train_cache, bundle.std)
    train_forecasts = bundle.forecasts_fn(bundle.train_cache, bundle.expert_idx).to(torch.float32)
    train_histories = bundle.train_cache["histories"].to(torch.float32)

    oof_raw = torch.full((n_train, horizon, variables, num_experts), float("nan"), dtype=torch.float32)
    oof_mask = torch.zeros(n_train, dtype=torch.bool)
    fold_calib_scalar: list[tuple[float, float]] = []
    fold_calib_expert: list[tuple[torch.Tensor, torch.Tensor]] = []
    fold_causality: list[dict[str, Any]] = []
    fold_training: list[dict[str, Any]] = []

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for fold_id, (eval_lo, eval_hi) in enumerate(wdec.oof_bounds(n_train), start=1):
        fold_ckpt = ckpt_dir / f"F2_local_fold{fold_id}.pt"
        current_eval_origin = int(train_starts[eval_lo])
        legal = wdec.legal_fit_mask(train_starts, horizon, current_eval_origin)
        if legal.numel() == 0:
            raise AssertionError(f"{dataset} fold {fold_id}: no legal fit windows")
        latest_fit_target_end = int((train_starts[legal] + horizon).max())
        causal_ok = bool(latest_fit_target_end <= current_eval_origin)
        fold_causality.append(
            {
                "fold": fold_id, "eval_lo": eval_lo, "eval_hi": eval_hi,
                "current_eval_origin": current_eval_origin, "num_legal_fit_windows": int(legal.numel()),
                "latest_fit_target_end": latest_fit_target_end, "causal": causal_ok,
            }
        )
        if not causal_ok:
            raise AssertionError(f"{dataset} fold {fold_id}: OOF causality violation")

        if fold_ckpt.exists():
            print(f"[ec-variant-sweep] {dataset}: fold {fold_id} resuming from checkpoint...", flush=True)
            saved = torch.load(fold_ckpt, weights_only=False)
            raw = saved["raw"]
            calib_scalar = saved["calib_scalar"]
            calib_expert = saved["calib_expert"]
            fold_training.append(saved["fold_log"])
        else:
            fit = fab.train_scorer_flexible(
                F2_LOCAL_GROUPS, horizon, variables, num_experts,
                train_global, train_local, train_forecasts, train_histories, bundle.std, train_gain, legal,
            )
            calib_scalar = wdec.fit_only_calibration(fit, train_global, train_local, train_forecasts, train_histories, bundle.std, legal)
            calib_expert = fit_only_calibration_per_expert(fit, train_global, train_local, train_forecasts, train_histories, bundle.std, legal)
            eval_idx = torch.arange(eval_lo, eval_hi)
            raw = wdec.score_windows(fit, train_global, train_local, train_forecasts, train_histories, bundle.std, eval_idx)
            fold_log = {
                "fold": fold_id, "fit_windows": int(legal.numel()), "train_windows": fit.train_windows,
                "internal_val_windows": fit.internal_val_windows, "best_epoch": fit.best_epoch,
                "best_internal_val_mse": fit.best_internal_val_mse,
                "calibration_mean_scalar": calib_scalar[0], "calibration_std_scalar": calib_scalar[1],
                "calibration_mean_per_expert": calib_expert[0].tolist(), "calibration_std_per_expert": calib_expert[1].tolist(),
            }
            fold_training.append(fold_log)
            torch.save({"raw": raw, "calib_scalar": calib_scalar, "calib_expert": calib_expert, "fold_log": fold_log}, fold_ckpt)
            print(f"[ec-variant-sweep] {dataset}: fold {fold_id}/4 done (best_epoch={fit.best_epoch})", flush=True)

        eval_idx = torch.arange(eval_lo, eval_hi)
        oof_raw[eval_idx] = raw
        oof_mask[eval_idx] = True
        fold_calib_scalar.append(calib_scalar)
        fold_calib_expert.append(calib_expert)

    oof_eval_idx = torch.nonzero(oof_mask, as_tuple=False).flatten()
    oof_valid_raw = oof_raw[oof_eval_idx]

    oof_affinity_existing = torch.empty_like(oof_valid_raw)
    oof_affinity_relative = torch.empty_like(oof_valid_raw)
    cursor = 0
    for (fold_id, (eval_lo, eval_hi)), calib_s, calib_e in zip(enumerate(wdec.oof_bounds(n_train), start=1), fold_calib_scalar, fold_calib_expert):
        n_fold = eval_hi - eval_lo
        chunk = oof_valid_raw[cursor : cursor + n_fold]
        oof_affinity_existing[cursor : cursor + n_fold] = wdec.raw_to_affinity(chunk, calib_s[0], calib_s[1])
        oof_affinity_relative[cursor : cursor + n_fold] = raw_to_affinity_expert_relative(chunk, calib_e[0], calib_e[1])
        cursor += n_fold

    oof_forecasts = train_forecasts[oof_eval_idx]
    oof_target = bundle.train_cache["targets"].to(torch.float32)[oof_eval_idx]
    oof_target_mask = bundle.train_cache["target_masks"].to(torch.bool)[oof_eval_idx]

    after_hashes = wdec.checkpoint_hashes(dataset, bundle.core_names)

    return {
        "dataset": dataset, "bundle": bundle, "horizon": horizon, "variables": variables, "num_experts": num_experts,
        "oof_scored_windows": int(oof_mask.sum()), "oof_forecasts": oof_forecasts, "oof_target": oof_target,
        "oof_target_mask": oof_target_mask, "oof_affinity": {"existing": oof_affinity_existing, "expert_relative": oof_affinity_relative},
        "train_role": train_role, "val_role": val_role, "before_hashes": before_hashes, "after_hashes": after_hashes,
        "oos_provenance": oos_provenance, "fold_causality": fold_causality, "fold_training": fold_training,
    }


# ---------------------------------------------------------------------------
# Full grid evaluation.
# ---------------------------------------------------------------------------


def run_grid(oof: dict[str, Any]) -> dict[str, Any]:
    bundle = oof["bundle"]
    core_names = list(bundle.core_names)
    std = bundle.std
    target, target_mask = oof["oof_target"], oof["oof_target_mask"]
    forecasts = oof["oof_forecasts"]

    frozen_dense_pred = forecasts.mean(dim=-1)
    frozen_dense_metrics = wdec.metric_from(frozen_dense_pred, target, target_mask, std)

    token_results: dict[str, Any] = {}
    for scoring in SCORING_VARIANTS:
        affinity = oof["oof_affinity"][scoring]
        tok_claim = wdec.dynamic_token_claims(affinity)
        tok_pred, tok_fb = wdec.dynamic_prediction_from_claims(forecasts, tok_claim)
        tok_metrics = wdec.metric_from(tok_pred, target, target_mask, std)
        token_results[scoring] = {"mae": tok_metrics["mae"], "mse": tok_metrics["mse"], "fallback_rate": tok_fb, "per_window_mae": tok_metrics["per_window_mae"]}

    configs: dict[str, Any] = {}
    for cf in CAPACITY_FACTORS:
        for assignment in ASSIGNMENT_RULES:
            for scoring in SCORING_VARIANTS:
                affinity = oof["oof_affinity"][scoring]
                claim, capacity = assign_claims(affinity, cf, assignment)
                pred, fallback_rate = wdec.dynamic_prediction_from_claims(forecasts, claim)
                metrics = wdec.metric_from(pred, target, target_mask, std)
                stats = claim_stats(claim, core_names, capacity)
                key = config_key(cf, assignment, scoring)
                configs[key] = {
                    "capacity_factor": cf, "assignment": assignment, "scoring": scoring, "capacity_per_expert": capacity,
                    "mae": metrics["mae"], "mse": metrics["mse"], "fallback_rate": fallback_rate,
                    "per_window_mae": metrics["per_window_mae"], "per_window_mse": metrics["per_window_mse"],
                    **stats,
                }
                print(f"[ec-variant-sweep] {oof['dataset']}: {key} MAE={metrics['mae']:.6f} MSE={metrics['mse']:.6f} fallback={fallback_rate:.4f}", flush=True)

    baseline_key = config_key(*BASELINE_CONFIG)

    dependence_rows: list[dict[str, Any]] = []
    for key, cfg in configs.items():
        tok = token_results[cfg["scoring"]]
        for cmp_label, base_mae_series, base_name in (
            ("vs_matched_token_choice", tok["per_window_mae"], "matched_dynamic_token_choice"),
            ("vs_current_ec_baseline", configs[baseline_key]["per_window_mae"], baseline_key),
            ("vs_frozen_dense_ensemble", frozen_dense_metrics["per_window_mae"], "frozen_dense_ensemble"),
        ):
            if key == base_name:
                continue
            boot = wdec.block_bootstrap_with_prob(cfg["per_window_mae"], base_mae_series, block=BLOCK_LENGTH, seed=BOOTSTRAP_SEED, samples=BOOTSTRAP_SAMPLES)
            phase = wdec.every_kth_phase_bootstrap(cfg["per_window_mae"] - base_mae_series, k=PHASE_K, seed=BOOTSTRAP_SEED, samples=BOOTSTRAP_SAMPLES)
            dependence_rows.append({"dataset": oof["dataset"], "config": key, "comparison": cmp_label, "baseline": base_name, "test": f"block_len_{BLOCK_LENGTH}", **boot})
            dependence_rows.append({"dataset": oof["dataset"], "config": key, "comparison": cmp_label, "baseline": base_name, "test": f"every_{PHASE_K}th_phase", **phase})

    return {
        "frozen_dense_ensemble": {"mae": frozen_dense_metrics["mae"], "mse": frozen_dense_metrics["mse"], "per_window_mae": frozen_dense_metrics["per_window_mae"]},
        "token_choice": token_results,
        "configs": configs,
        "baseline_key": baseline_key,
        "dependence": dependence_rows,
    }


# ---------------------------------------------------------------------------
# Questions 1-4, answered mechanically from the grid.
# ---------------------------------------------------------------------------


def answer_questions(grid: dict[str, Any]) -> dict[str, Any]:
    configs = grid["configs"]
    baseline_key = grid["baseline_key"]

    def mae(cf: float, assignment: str, scoring: str) -> float:
        return configs[config_key(cf, assignment, scoring)]["mae"]

    # Q1: does CF change OOF performance, holding assignment/scoring fixed at baseline settings?
    cf_sweep = {cf: mae(cf, "unrestricted", "existing") for cf in CAPACITY_FACTORS}
    best_cf = min(cf_sweep, key=cf_sweep.get)
    q1 = {
        "mae_by_cf_unrestricted_existing": cf_sweep,
        "best_cf": best_cf,
        "best_beats_cf1": cf_sweep[best_cf] < cf_sweep[1.0],
        "cf1_is_best": best_cf == 1.0,
    }

    # Q2: max2 vs unrestricted, matched CF x scoring pairs (6 matched pairs)
    max2_wins = 0
    pair_deltas = {}
    for cf in CAPACITY_FACTORS:
        for scoring in SCORING_VARIANTS:
            d = mae(cf, "max2", scoring) - mae(cf, "unrestricted", scoring)
            pair_deltas[f"cf{cf}_{scoring}"] = d
            if d < 0:
                max2_wins += 1
    q2 = {
        "max2_minus_unrestricted_by_matched_pair": pair_deltas,
        "max2_wins_of_6_matched_pairs": max2_wins,
        "max2_better_majority": max2_wins >= 4,
        "forced_one_expert_reference": "Conflict-Resolved Expert Choice (2026-08-30) already tested forcing exactly ONE expert per cell on this same window-dependent-EC family and lost on 0/5 datasets by 0.0012-0.0022 MAE (experiments/conflict_resolved_expert_choice_hv/report.md) -- reused as context, not recomputed here.",
    }

    # Q3: expert_relative vs existing, matched CF x assignment pairs (6 matched pairs)
    relative_wins = 0
    scoring_deltas = {}
    for cf in CAPACITY_FACTORS:
        for assignment in ASSIGNMENT_RULES:
            d = mae(cf, assignment, "expert_relative") - mae(cf, assignment, "existing")
            scoring_deltas[f"cf{cf}_{assignment}"] = d
            if d < 0:
                relative_wins += 1
    q3 = {
        "expert_relative_minus_existing_by_matched_pair": scoring_deltas,
        "expert_relative_wins_of_6_matched_pairs": relative_wins,
        "expert_relative_better_majority": relative_wins >= 4,
    }

    ranked = sorted(configs.items(), key=lambda kv: kv[1]["mae"])
    best_key, best_cfg = ranked[0]
    # Block-24 support for the best config vs baseline (if best != baseline)
    best_vs_baseline_block24 = None
    for row in grid["dependence"]:
        if row["config"] == best_key and row["comparison"] == "vs_current_ec_baseline" and row["test"] == f"block_len_{BLOCK_LENGTH}":
            best_vs_baseline_block24 = row
    q4 = {
        "ranked_configs_by_oof_mae": [{"config": k, "mae": v["mae"], "mse": v["mse"]} for k, v in ranked],
        "best_config": best_key,
        "best_config_mae": best_cfg["mae"],
        "best_config_is_baseline": best_key == baseline_key,
        "best_vs_baseline_block24": best_vs_baseline_block24,
    }

    return {"q1_capacity_factor": q1, "q2_assignment_rule": q2, "q3_scoring_normalization": q3, "q4_best_config": q4}


# ---------------------------------------------------------------------------
# Manifest / report / main.
# ---------------------------------------------------------------------------


def build_manifest(source_hash: str) -> dict[str, Any]:
    return {
        "experiment": "ec_variant_sweep_etth1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": wdec.git_info(),
        "source_sha256": source_hash,
        "dataset": DATASET,
        "other_datasets_touched": [],
        "starting_point": "F2_local scorer (cell-local + per-variable-history features, no global-history features), identical architecture/training/seed to feature_ablation_affinity_weighted_ec.FlexibleResidualScorer with enabled_groups={'cell','local'}",
        "kept_fixed": [
            "frozen K=3 core PatchTST+iTransformer+TimesNet (checkpoints unchanged, hash-verified before/after)",
            "existing router_train/router_val split and caches, unmodified",
            "causal 4-fold OOF protocol (wdec.oof_bounds, full-horizon-observability legality rule)",
            "scorer seed=7, AdamW lr=1e-3 wd=1e-4, max_epochs=100 patience=10 batch_size=32",
            "residual-gain target (gain - static_gain, fit-only per fold)",
            "claim combination rule: equal average of claiming experts + equal-ensemble zero-claim fallback (dynamic_prediction_from_claims, unmodified)",
        ],
        "grid": {
            "capacity_factors": list(CAPACITY_FACTORS),
            "assignment_rules": list(ASSIGNMENT_RULES),
            "scoring_variants": list(SCORING_VARIANTS),
            "max_claims_per_cell_for_max2": MAX_CLAIMS_PER_CELL,
            "total_configs": len(CAPACITY_FACTORS) * len(ASSIGNMENT_RULES) * len(SCORING_VARIANTS),
            "predeclared_current_ec_baseline": config_key(*BASELINE_CONFIG),
        },
        "capacity_formula": "C = max(1, round(CF * H*V / E))",
        "assignment_definitions": {
            "unrestricted": "each expert independently claims its top-C cells by affinity; no cap on claims per cell (0..E allowed)",
            "max2": "single deterministic global greedy assignment over all (cell,expert) pairs ranked by affinity (ties -> lower cell*E+expert index), accepting a pair iff expert has remaining capacity AND cell has <2 claims so far; experts may end under capacity if the constraint binds",
        },
        "scoring_definitions": {
            "existing": "wdec.raw_to_affinity: ONE fit-only scalar (mean,std) over the whole raw-score tensor on fit windows, then softmax over experts, temperature=1.0",
            "expert_relative": "fit-only PER-EXPERT (mean[e],std[e]) over fit windows, then softmax over experts, temperature=1.0 -- same softmax-over-experts normalization, but each expert's raw score is standardized against its own distribution first",
        },
        "required_comparisons_per_config": ["matched_dynamic_token_choice (same scoring variant)", "current_ec_baseline (cf1.0_unrestricted_existing)", "frozen_dense_ensemble (equal average of the 3 frozen experts)"],
        "primary_metric": "router_train strict chronological OOF MAE",
        "also_reported": ["MSE", "zero/one/two/more-than-two claim cell fractions", "expert utilization vs intended capacity", "delta vs matched Token Choice", "delta vs current EC baseline", "block-24 CI", "every-12th-phase CI"],
        "evidence_source": "router_train causal OOF ONLY. router_val is loaded (Bundle convention) but never used for any prediction, metric, or selection in this file. Test is never loaded.",
        "no_tuning_after_results": "Per explicit instruction, this method (grid, capacity formula, assignment definitions, scoring definitions) is not to change after OOF results are seen, except to fix implementation bugs.",
        "TEST_SET_ACCESSED": False, "TEST_CACHE_LOADED": False, "TEST_METRICS_COMPUTED": False,
        "ROUTER_VAL_ACCESSED": False, "UNTOUCHED_DATA_ACCESSED": False, "OTHER_DATASETS_ACCESSED": False,
    }


def make_report(oof: dict[str, Any], grid: dict[str, Any], answers: dict[str, Any]) -> None:
    lines = [
        "Expert-Choice variant sweep on Window-Dependent EC -- ETTh1 only",
        "",
        "```text",
        "TEST SET ACCESSED: NO",
        "ROUTER_VAL ACCESSED: NO",
        "OTHER DATASETS ACCESSED: NO",
        "```",
        "",
        f"OOF scored windows: {oof['oof_scored_windows']}",
        f"Frozen dense ensemble (equal average, no routing): MAE `{grid['frozen_dense_ensemble']['mae']:.6f}`, MSE `{grid['frozen_dense_ensemble']['mse']:.6f}`",
        "",
        "## Matched Dynamic Token Choice (per scoring variant)",
        "",
        "| Scoring | Token Choice MAE | Token Choice MSE | Fallback rate |",
        "|---|---:|---:|---:|",
    ]
    for scoring in SCORING_VARIANTS:
        t = grid["token_choice"][scoring]
        lines.append(f"| {scoring} | `{t['mae']:.6f}` | `{t['mse']:.6f}` | `{t['fallback_rate']:.4f}` |")

    lines += ["", "## Full 12-configuration grid, ranked by OOF MAE", "",
              "| Rank | Config | CF | Assignment | Scoring | MAE | MSE | Fallback | 0-claim% | 1-claim% | 2-claim% | >2-claim% | vs Token | vs EC baseline | vs Frozen |",
              "|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    baseline_key = grid["baseline_key"]
    for rank, entry in enumerate(answers["q4_best_config"]["ranked_configs_by_oof_mae"], start=1):
        key = entry["config"]
        c = grid["configs"][key]
        tok = grid["token_choice"][c["scoring"]]
        dist = c["claim_distribution_pct"]
        vs_tok = c["mae"] - tok["mae"]
        vs_base = c["mae"] - grid["configs"][baseline_key]["mae"]
        vs_frozen = c["mae"] - grid["frozen_dense_ensemble"]["mae"]
        marker = " *(baseline)*" if key == baseline_key else ""
        lines.append(
            f"| {rank} | `{key}`{marker} | {c['capacity_factor']} | {c['assignment']} | {c['scoring']} | `{c['mae']:.6f}` | `{c['mse']:.6f}` | `{c['fallback_rate']:.4f}` | "
            f"`{dist['zero_claim_cells_pct']:.2f}` | `{dist['one_claim_cells_pct']:.2f}` | `{dist['two_claim_cells_pct']:.2f}` | `{dist['more_than_two_claim_cells_pct']:.2f}` | "
            f"`{vs_tok:+.6f}` | `{vs_base:+.6f}` | `{vs_frozen:+.6f}` |"
        )

    lines += ["", "## Expert utilization (actual claims / intended capacity)", "", "| Config | " + " | ".join(f"{n}" for n in grid["configs"][baseline_key]["expert_utilization"].keys()) + " |", "|---|" + "---:|" * len(grid["configs"][baseline_key]["expert_utilization"])]
    for key, cfg in grid["configs"].items():
        cells = " | ".join(f"`{u['utilization_pct']:.1f}%`" for u in cfg["expert_utilization"].values())
        lines.append(f"| `{key}` | {cells} |")

    lines += ["", "## Question 1: does capacity factor change OOF performance?", "",
              f"MAE by CF (unrestricted/existing): {json.dumps(answers['q1_capacity_factor']['mae_by_cf_unrestricted_existing'])}",
              f"Best CF: `{answers['q1_capacity_factor']['best_cf']}` (CF=1.0 is best: `{answers['q1_capacity_factor']['cf1_is_best']}`)", ""]

    lines += ["## Question 2: does max-2-per-cell beat unrestricted or forcing one expert?", "",
              f"max2 beats unrestricted on {answers['q2_assignment_rule']['max2_wins_of_6_matched_pairs']}/6 matched (CF, scoring) pairs.",
              f"Per-pair deltas (max2 - unrestricted MAE): {json.dumps(answers['q2_assignment_rule']['max2_minus_unrestricted_by_matched_pair'])}",
              f"Context: {answers['q2_assignment_rule']['forced_one_expert_reference']}", ""]

    lines += ["## Question 3: does expert-relative softmax scoring help?", "",
              f"expert_relative beats existing on {answers['q3_scoring_normalization']['expert_relative_wins_of_6_matched_pairs']}/6 matched (CF, assignment) pairs.",
              f"Per-pair deltas (expert_relative - existing MAE): {json.dumps(answers['q3_scoring_normalization']['expert_relative_minus_existing_by_matched_pair'])}", ""]

    lines += ["## Question 4: single best preregistered configuration", "",
              f"Best config by OOF MAE: `{answers['q4_best_config']['best_config']}` (MAE `{answers['q4_best_config']['best_config_mae']:.6f}`); is the baseline itself: `{answers['q4_best_config']['best_config_is_baseline']}`.", ""]
    if answers["q4_best_config"]["best_vs_baseline_block24"] is not None:
        b = answers["q4_best_config"]["best_vs_baseline_block24"]
        lines.append(f"Block-24 CI of best vs baseline: mean_delta=`{b['mean_delta']:.6f}`, CI95=[`{b['ci95_low']:.6f}`, `{b['ci95_high']:.6f}`], excludes_zero=`{b['ci_excludes_zero']}`")
    lines.append("")

    (OUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source_hash = sha256_file(Path(__file__))
    manifest = build_manifest(source_hash)
    write_json(OUT_DIR / "method_manifest.json", manifest)

    ckpt_dir = OUT_DIR / "_checkpoints" / DATASET
    print(f"[ec-variant-sweep] {DATASET}: running F2_local causal OOF folds...", flush=True)
    oof = run_oof(DATASET, ckpt_dir)

    print(f"[ec-variant-sweep] {DATASET}: evaluating 12-configuration grid...", flush=True)
    grid = run_grid(oof)
    answers = answer_questions(grid)

    config_rows = []
    for key, cfg in grid["configs"].items():
        row = {k: v for k, v in cfg.items() if k not in ("per_window_mae", "per_window_mse", "claim_distribution_pct", "expert_utilization")}
        row["config"] = key
        row.update(cfg["claim_distribution_pct"])
        for expert, u in cfg["expert_utilization"].items():
            row[f"utilization_pct_{expert}"] = u["utilization_pct"]
            row[f"actual_claims_{expert}"] = u["actual_total_claims"]
        config_rows.append(row)
    write_csv_rows(OUT_DIR / "config_grid_results.csv", config_rows)
    write_csv_rows(OUT_DIR / "dependence_tests.csv", grid["dependence"])
    write_csv_rows(OUT_DIR / "fold_training.csv", oof["fold_training"])
    write_csv_rows(OUT_DIR / "fold_causality.csv", oof["fold_causality"])

    integrity = {
        "dataset": DATASET,
        "train_role": oof["train_role"], "val_role": oof["val_role"],
        "no_test_in_roles": bool("test" not in oof["train_role"].lower() and "test" not in oof["val_role"].lower()),
        "frozen_checkpoint_hashes_unchanged": bool(oof["before_hashes"] == oof["after_hashes"]),
        "checkpoint_hash_count": len(oof["before_hashes"]),
        "oof_causality_all_folds": bool(all(row["causal"] for row in oof["fold_causality"])),
        "router_train_oos_provenance_result": oof["oos_provenance"].get("result"),
        "source_hash_before": source_hash, "source_hash_after": sha256_file(Path(__file__)),
        "TEST_SET_ACCESSED": "NO", "TEST_CACHE_LOADED": "NO", "TEST_METRICS_COMPUTED": "NO",
        "ROUTER_VAL_ACCESSED": "NO", "UNTOUCHED_DATA_ACCESSED": "NO", "OTHER_DATASETS_ACCESSED": "NO",
    }
    integrity["source_unchanged"] = integrity["source_hash_before"] == integrity["source_hash_after"]
    integrity["all_pass"] = bool(
        integrity["no_test_in_roles"] and integrity["frozen_checkpoint_hashes_unchanged"]
        and integrity["oof_causality_all_folds"] and integrity["source_unchanged"]
        and (integrity["router_train_oos_provenance_result"] != "FAIL")
    )
    write_json(OUT_DIR / "integrity_checks.json", integrity)
    if not integrity["all_pass"]:
        raise AssertionError(f"INVALID_EXPERIMENT -- integrity failure: {integrity}")

    results = {
        "dataset": DATASET,
        "oof_scored_windows": oof["oof_scored_windows"],
        "frozen_dense_ensemble": {"mae": grid["frozen_dense_ensemble"]["mae"], "mse": grid["frozen_dense_ensemble"]["mse"]},
        "token_choice": {k: {kk: vv for kk, vv in v.items() if kk != "per_window_mae"} for k, v in grid["token_choice"].items()},
        "configs": {k: {kk: vv for kk, vv in v.items() if kk not in ("per_window_mae", "per_window_mse")} for k, v in grid["configs"].items()},
        "baseline_key": grid["baseline_key"],
        "answers": answers,
    }
    write_json(OUT_DIR / "results.json", jsonable(results))

    make_report(oof, grid, answers)

    manifest["runtime_sec"] = time.time() - start
    manifest["integrity_valid"] = integrity["all_pass"]
    manifest["baseline_key"] = grid["baseline_key"]
    manifest["best_config"] = answers["q4_best_config"]["best_config"]
    write_json(OUT_DIR / "method_manifest.json", manifest)

    print(f"INTEGRITY VALID: {'YES' if integrity['all_pass'] else 'NO'}")
    print(f"BEST CONFIG BY OOF MAE: {answers['q4_best_config']['best_config']}")
    print(f"CURRENT EC BASELINE: {grid['baseline_key']}")
    print("ROUTER_VAL ACCESSED: NO")
    print("TEST ACCESSED: NO")
    print("OTHER DATASETS ACCESSED: NO")


if __name__ == "__main__":
    main()
