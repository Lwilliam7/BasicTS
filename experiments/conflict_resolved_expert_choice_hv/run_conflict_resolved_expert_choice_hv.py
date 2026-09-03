"""Conflict-Resolved Window-Dependent Expert Choice.

Development experiment (ETTh1/ETTh2/ETTm1/Weather/Electricity only -- no
test, no untouched-confirmation datasets). Reuses the frozen, already-computed
score/affinity/claim tensors from window_dependent_expert_choice_hv WITHOUT
any retraining -- exactly like affinity_weighted_expert_choice_hv did. The
only new mechanism is the ASSIGNMENT rule: instead of letting each expert
independently claim its top-C cells (producing 0/1/multi-claim cells),
resolve conflicts via expert-proposing deferred acceptance so every cell
ends up held by exactly one expert.

Algorithm (as specified): each expert ranks all H x V cells by its own
affinity, descending, deterministic lower-index tie-break; each expert
proposes down its list until it holds capacity C; on a conflict a cell keeps
the higher-affinity claimant and the loser proposes to its next cell;
repeat until every expert's capacity is filled.

Implementation note (disclosed, not a deviation from the spec): because the
cell's acceptance rule uses the SAME affinity value the proposing expert used
to rank its own preferences, this specific deferred-acceptance instance is
provably equivalent to a single global greedy pass over all (cell, expert)
pairs sorted by descending affinity (ties broken by lower cell index, then
lower expert index), assigning a pair whenever its cell is still unclaimed
and its expert is still under capacity. Proof sketch: the globally largest
affinity value is simultaneously the top preference of its expert and the
top bid for its cell, so it is accepted immediately and permanently in DA;
by induction on the remaining pairs after removing it, the same holds at
every step. This is verified empirically below (not just asserted) by
running a literal, explicit round-based proposal/rejection simulation on a
small window sample and checking it reproduces the vectorized-greedy result
exactly, before the greedy path is used at full scale.

TEST SET ACCESSED: NO. TEST CACHE LOADED: NO. TEST METRICS COMPUTED: NO.
UNTOUCHED CONFIRMATION DATASETS ACCESSED: NO.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.window_dependent_expert_choice_hv.run_window_dependent_expert_choice_hv as wdec  # noqa: E402
import experiments.affinity_weighted_expert_choice_hv.run_affinity_weighted_expert_choice_hv as aw  # noqa: E402
from experiments.frozen_hv_costar.run_frozen_hv_costar import LOADERS, frozen_hv_prediction  # noqa: E402
from experiments.expert_choice_hv.run_expert_choice_hv import metric_values as static_metric_values  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
DATASETS = wdec.DATASETS
BLOCK_LENGTH = wdec.BLOCK_LENGTH
PHASE_K = wdec.PHASE_K
BOOTSTRAP_SAMPLES = wdec.BOOTSTRAP_SAMPLES
BOOTSTRAP_SEED = wdec.BOOTSTRAP_SEED
STORED_TENSORS_PATH = ROOT / "experiments/window_dependent_expert_choice_hv/tensors.pt"

OOF_GATE_MIN_WINS = 3  # of 5 datasets, predeclared


def write_json(path: Path, obj: Any) -> None:
    wdec.write_json(path, obj)


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    wdec.write_csv_rows(path, rows)


def jsonable(value: Any) -> Any:
    return wdec.jsonable(value)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Section: the ONLY new mechanism. Conflict-Resolved assignment.
# ---------------------------------------------------------------------------


def conflict_resolved_ec_claims(affinity: torch.Tensor) -> tuple[torch.Tensor, int]:
    """affinity: [N,H,V,E]. Returns (claim_mask [N,H,V,E] bool, capacity C).
    Every cell is claimed by EXACTLY one expert; every expert holds EXACTLY
    C cells. No fallback: raises if this cannot be achieved exactly."""
    n, h, v, e = affinity.shape
    m = h * v
    capacity = int(round(m / e))
    if capacity * e != m:
        raise AssertionError(
            f"Conflict-Resolved EC requires capacity*E == M exactly (no fallback permitted by design): "
            f"M={m}, E={e}, capacity={capacity}, capacity*E={capacity * e}"
        )

    flat = affinity.reshape(n, m, e)
    # Composite priority key: higher affinity first; ties -> lower cell index
    # first (matches each expert's own tie-break rule); further ties -> lower
    # expert index first (deterministic, disclosed). Exact fp16-affinity ties
    # DO occur in practice (the stored affinity tensor was persisted as
    # float16 -- see the same disclosed precision caveat in
    # affinity_weighted_expert_choice_hv), so this must be a truly exact,
    # collision-free key, not an approximate one.
    #
    # A first version used a scaled float64 composite key
    # (affinity*1e9 - cell_idx*1e3 - expert_idx). That was WRONG for large M:
    # for Electricity (M=3852), when affinity is small, cell_idx*1e3 can
    # exceed affinity*1e9 and corrupt the intended priority order. Caught by
    # the literal-DA cross-check below (13/3852 cells differed on one sampled
    # window; 30/30 sampled windows had at least one such cell). Fixed by
    # using the float16 bit pattern as an EXACT, collision-free, monotonic
    # integer encoding of affinity magnitude (a standard IEEE754 property:
    # for non-negative floats, comparing raw bit patterns as integers gives
    # the same order as comparing the floats), then combining with cell/
    # expert index via integer arithmetic with provably non-overlapping bit
    # ranges -- no floating-point precision risk at any scale.
    aff16_bits = flat.to(torch.float16).view(torch.int16).to(torch.int64)
    if bool((aff16_bits < 0).any()):
        raise AssertionError("Conflict-Resolved EC: negative affinity encountered; the float16-bit-pattern monotonic key assumes non-negative (softmax) affinity.")
    cell_idx = torch.arange(m, dtype=torch.int64, device=affinity.device)
    expert_idx = torch.arange(e, dtype=torch.int64, device=affinity.device)
    CELL_BOUND = 1 << 20  # generous upper bound on M (Electricity's M=3852 is far below this)
    EXPERT_BOUND = 1 << 8  # generous upper bound on E (E=3 is far below this)
    BIG2 = EXPERT_BOUND
    BIG1 = BIG2 * CELL_BOUND
    key = aff16_bits * BIG1 - cell_idx.view(1, m, 1) * BIG2 - expert_idx.view(1, 1, e)
    key_flat = key.reshape(n, m * e)
    order = torch.argsort(key_flat, dim=1, descending=True)  # [N, M*E] linear (cell,expert) indices, per-window priority order

    cell_claimed = torch.zeros(n, m, dtype=torch.bool, device=affinity.device)
    expert_count = torch.zeros(n, e, dtype=torch.long, device=affinity.device)
    assign_expert = torch.full((n, m), -1, dtype=torch.long, device=affinity.device)

    for k in range(m * e):
        idx = order[:, k]
        cand_cell = idx // e
        cand_expert = idx % e
        already_claimed = cell_claimed.gather(1, cand_cell.unsqueeze(1)).squeeze(1)
        cur_count = expert_count.gather(1, cand_expert.unsqueeze(1)).squeeze(1)
        eligible = (~already_claimed) & (cur_count < capacity)
        if not bool(eligible.any()):
            continue
        current_assign = assign_expert.gather(1, cand_cell.unsqueeze(1)).squeeze(1)
        new_assign = torch.where(eligible, cand_expert, current_assign)
        assign_expert.scatter_(1, cand_cell.unsqueeze(1), new_assign.unsqueeze(1))
        newly_claimed = cell_claimed.gather(1, cand_cell.unsqueeze(1)).squeeze(1) | eligible
        cell_claimed.scatter_(1, cand_cell.unsqueeze(1), newly_claimed.unsqueeze(1))
        expert_count.scatter_add_(1, cand_expert.unsqueeze(1), eligible.long().unsqueeze(1))

    if not bool(cell_claimed.all()):
        raise AssertionError(f"Conflict-Resolved EC: {int((~cell_claimed).sum())} cells left unclaimed -- no fallback permitted, stopping.")
    if not bool((expert_count == capacity).all()):
        raise AssertionError(f"Conflict-Resolved EC: expert capacities not exactly filled -- counts={expert_count.unique(return_counts=True)}, expected {capacity}. No fallback permitted.")

    claim_mask = torch.zeros(n, m, e, dtype=torch.bool, device=affinity.device)
    claim_mask.scatter_(2, assign_expert.unsqueeze(-1), True)
    return claim_mask.view(n, h, v, e), capacity


def literal_deferred_acceptance_single_window(affinity_window: torch.Tensor, capacity: int) -> torch.Tensor:
    """Slow, explicit, human-auditable round-based proposal/rejection
    simulation for ONE window's [M,E] affinity matrix, implementing the
    5-step procedure literally (not the greedy shortcut). Used only to
    empirically validate the greedy equivalence on a small sample."""
    m, e = affinity_window.shape
    aff = affinity_window.double().tolist()
    # Each expert's preference order over cells: descending affinity, tie -> lower cell index.
    pref = [sorted(range(m), key=lambda c: (-aff[c][expert], c)) for expert in range(e)]
    next_idx = [0] * e  # next position in pref[expert] to propose from
    held_by: dict[int, int] = {}  # cell -> expert currently holding it
    held_count = [0] * e
    needs_more = list(range(e))
    guard = 0
    while needs_more and guard < m * e + 10:
        guard += 1
        progressed = False
        still_needs = []
        for expert in needs_more:
            if held_count[expert] >= capacity:
                continue
            if next_idx[expert] >= m:
                raise AssertionError("literal DA exhausted an expert's preference list without filling capacity -- should not happen when capacity*E==M")
            cell = pref[expert][next_idx[expert]]
            next_idx[expert] += 1
            progressed = True
            incumbent = held_by.get(cell)
            # Conflict tie-break (disclosed, matches the greedy composite key
            # exactly): higher affinity wins; on an EXACT tie -- which does
            # occur because the stored affinity tensor was persisted as
            # float16 (see affinity_weighted_expert_choice_hv's identical,
            # already-disclosed precision caveat) -- the lower expert index
            # wins. This was found and fixed via this exact verification
            # step: the first run of this literal simulation lacked this
            # tie-break and disagreed with the greedy implementation on
            # 2/30 sampled ETTm1 windows, always on cells with an exact
            # fp16-induced affinity tie between two experts.
            if incumbent is None:
                held_by[cell] = expert
                held_count[expert] += 1
            elif (aff[cell][expert], -expert) > (aff[cell][incumbent], -incumbent):
                held_by[cell] = expert
                held_count[expert] += 1
                held_count[incumbent] -= 1
                still_needs.append(incumbent)
            # else: rejected, expert stays in needs_more implicitly via still_needs check below
            if held_count[expert] < capacity:
                still_needs.append(expert)
        needs_more = sorted(set(still_needs))
        if not progressed:
            break
    if len(held_by) != m or any(c != capacity for c in held_count):
        raise AssertionError(f"literal DA did not reach a perfect matching: held={len(held_by)}/{m}, counts={held_count}")
    assign = torch.full((m,), -1, dtype=torch.long)
    for cell, expert in held_by.items():
        assign[cell] = expert
    return assign


def verify_greedy_equals_literal_da(affinity_sample: torch.Tensor, capacity: int, n_windows_to_check: int, seed: int) -> dict[str, Any]:
    n = affinity_sample.shape[0]
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=gen)[: min(n_windows_to_check, n)]
    h, v, e = affinity_sample.shape[1], affinity_sample.shape[2], affinity_sample.shape[3]
    m = h * v
    greedy_claims, cap = conflict_resolved_ec_claims(affinity_sample[idx])
    greedy_assign = greedy_claims.view(len(idx), m, e).to(torch.long).argmax(dim=-1)
    mismatches = 0
    for row, window_idx in enumerate(idx.tolist()):
        flat_aff = affinity_sample[window_idx].reshape(m, e)
        literal_assign = literal_deferred_acceptance_single_window(flat_aff, capacity)
        if not torch.equal(literal_assign, greedy_assign[row]):
            mismatches += 1
    return {
        "windows_checked": len(idx),
        "mismatches": mismatches,
        "all_match": mismatches == 0,
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# Loading: OOF-only first (router_val untouched until the gate passes).
# ---------------------------------------------------------------------------


@dataclass
class OOFOnlyDataset:
    dataset: str
    bundle: Any
    oof_affinity: torch.Tensor
    oof_forecasts: torch.Tensor
    oof_target: torch.Tensor
    oof_target_mask: torch.Tensor
    capacity: int


def load_oof_only(dataset: str) -> OOFOnlyDataset:
    bundle = LOADERS[dataset]()
    stored = torch.load(STORED_TENSORS_PATH, map_location="cpu", weights_only=False)[dataset]
    oof_affinity = stored["oof_affinity"].to(torch.float32)

    n_train = int(bundle.train_cache["num_windows"])
    oof_mask = torch.zeros(n_train, dtype=torch.bool)
    for eval_lo, eval_hi in wdec.oof_bounds(n_train):
        oof_mask[eval_lo:eval_hi] = True
    oof_eval_idx = torch.nonzero(oof_mask, as_tuple=False).flatten()
    if int(oof_eval_idx.numel()) != int(oof_affinity.shape[0]):
        raise AssertionError(f"{dataset}: recomputed OOF fold boundaries ({oof_eval_idx.numel()}) do not match stored oof_affinity rows ({oof_affinity.shape[0]})")

    train_forecasts = bundle.forecasts_fn(bundle.train_cache, bundle.expert_idx).to(torch.float32)
    train_target = bundle.train_cache["targets"].to(torch.float32)
    train_target_mask = bundle.train_cache["target_masks"].to(torch.bool)

    oof_forecasts = train_forecasts[oof_eval_idx]
    oof_target = train_target[oof_eval_idx]
    oof_target_mask = train_target_mask[oof_eval_idx]

    horizon = int(bundle.val_cache["forecast_horizon"])
    capacity = int(round((horizon * int(bundle.val_cache["num_features"])) / len(bundle.expert_idx)))

    return OOFOnlyDataset(
        dataset=dataset, bundle=bundle, oof_affinity=oof_affinity,
        oof_forecasts=oof_forecasts, oof_target=oof_target, oof_target_mask=oof_target_mask,
        capacity=capacity,
    )


@dataclass
class ValExtension:
    val_affinity: torch.Tensor
    val_ec_claim: torch.Tensor
    val_forecasts: torch.Tensor
    val_target: torch.Tensor
    val_target_mask: torch.Tensor


def load_val_extension(dataset: str, bundle: Any) -> ValExtension:
    """Only called AFTER the OOF gate has passed. Loads the stored
    router_val affinity/claim tensors and computes router_val forecasts --
    this is the sole point at which router_val is used for any metric."""
    stored = torch.load(STORED_TENSORS_PATH, map_location="cpu", weights_only=False)[dataset]
    val_affinity = stored["val_affinity"].to(torch.float32)
    val_ec_claim = stored["val_ec_claim"]
    val_forecasts = bundle.forecasts_fn(bundle.val_cache, bundle.expert_idx).to(torch.float32)
    val_target = bundle.val_cache["targets"].to(torch.float32)
    val_target_mask = bundle.val_cache["target_masks"].to(torch.bool)
    return ValExtension(val_affinity=val_affinity, val_ec_claim=val_ec_claim, val_forecasts=val_forecasts, val_target=val_target, val_target_mask=val_target_mask)


# ---------------------------------------------------------------------------
# Claim-rate / diagnostics helpers
# ---------------------------------------------------------------------------


def claim_rate_stats(claim_mask: torch.Tensor) -> dict[str, float]:
    counts = claim_mask.sum(dim=-1)
    total = float(counts.numel())
    e = claim_mask.shape[-1]
    return {f"fraction_{k}_claim_cells": float((counts == k).to(torch.float32).sum() / total) for k in range(e + 1)}


def metric_from(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, std: torch.Tensor) -> dict[str, Any]:
    return wdec.metric_from(pred, target, mask, std)


# ---------------------------------------------------------------------------
# Per-dataset OOF stage
# ---------------------------------------------------------------------------


def run_dataset_oof(dataset: str) -> dict[str, Any]:
    print(f"[conflict-resolved-ec] {dataset}: loading OOF-only data (router_val untouched)...", flush=True)
    d = load_oof_only(dataset)
    std = d.bundle.std

    # Existing Dynamic EC (loaded claim, not recomputed -- exact reuse).
    existing_claim, existing_cap = wdec.dynamic_ec_claims(d.oof_affinity)
    existing_pred, existing_fb = wdec.dynamic_prediction_from_claims(d.oof_forecasts, existing_claim)
    existing_metrics = metric_from(existing_pred, d.oof_target, d.oof_target_mask, std)

    # Affinity-Weighted EC (same claim mask, weighted combine).
    weighted_pred, weighted_fb, weighted_weights = aw.affinity_weighted_prediction_from_claims(d.oof_forecasts, existing_claim, d.oof_affinity)
    weighted_metrics = metric_from(weighted_pred, d.oof_target, d.oof_target_mask, std)

    print(f"[conflict-resolved-ec] {dataset}: resolving conflicts (vectorized greedy)...", flush=True)
    t0 = time.time()
    cr_claim, cr_cap = conflict_resolved_ec_claims(d.oof_affinity)
    cr_elapsed = time.time() - t0
    if cr_cap != existing_cap:
        raise AssertionError(f"{dataset}: capacity mismatch existing={existing_cap} vs conflict-resolved={cr_cap}")
    cr_pred, cr_fb = wdec.dynamic_prediction_from_claims(d.oof_forecasts, cr_claim)
    cr_metrics = metric_from(cr_pred, d.oof_target, d.oof_target_mask, std)

    print(f"[conflict-resolved-ec] {dataset}: verifying greedy == literal deferred acceptance on a sample...", flush=True)
    n_check = min(30, d.oof_affinity.shape[0])
    da_check = verify_greedy_equals_literal_da(d.oof_affinity, cr_cap, n_check, seed=20260830)
    if not da_check["all_match"]:
        raise AssertionError(f"{dataset}: greedy-vs-literal-DA verification FAILED: {da_check}")

    cr_rates = claim_rate_stats(cr_claim)
    existing_rates = claim_rate_stats(existing_claim)

    delta_cr_minus_weighted = cr_metrics["mae"] - weighted_metrics["mae"]

    return {
        "dataset": dataset,
        "capacity_per_expert": cr_cap,
        "conflict_resolution_wallclock_sec": cr_elapsed,
        "deferred_acceptance_verification": da_check,
        "oof": {
            "dynamic_ec_existing": {"mae": existing_metrics["mae"], "mse": existing_metrics["mse"], "fallback_rate": existing_fb},
            "affinity_weighted_ec": {"mae": weighted_metrics["mae"], "mse": weighted_metrics["mse"], "fallback_rate": weighted_fb},
            "conflict_resolved_ec": {"mae": cr_metrics["mae"], "mse": cr_metrics["mse"], "fallback_rate": cr_fb},
            "delta_conflict_resolved_minus_weighted": delta_cr_minus_weighted,
            "conflict_resolved_beats_weighted": delta_cr_minus_weighted < 0,
        },
        "claim_rates": {"conflict_resolved": cr_rates, "existing_dynamic_ec": existing_rates},
        "per_window_mae_for_dependence": {
            "conflict_resolved": cr_metrics["per_window_mae"],
            "affinity_weighted": weighted_metrics["per_window_mae"],
        },
        "loaded": d,
        "existing_claim": existing_claim,
        "cr_claim": cr_claim,
    }


# ---------------------------------------------------------------------------
# Per-dataset router_val stage (only runs if the OOF gate passes)
# ---------------------------------------------------------------------------


def run_dataset_val(dataset: str, oof_result: dict[str, Any]) -> dict[str, Any]:
    print(f"[conflict-resolved-ec] {dataset}: OOF gate passed -- now computing router_val (single pass)...", flush=True)
    d: OOFOnlyDataset = oof_result["loaded"]
    bundle = d.bundle
    std = bundle.std
    v = load_val_extension(dataset, bundle)

    existing_pred, existing_fb = wdec.dynamic_prediction_from_claims(v.val_forecasts, v.val_ec_claim)
    existing_metrics = metric_from(existing_pred, v.val_target, v.val_target_mask, std)

    weighted_pred, weighted_fb, _ = aw.affinity_weighted_prediction_from_claims(v.val_forecasts, v.val_ec_claim, v.val_affinity)
    weighted_metrics = metric_from(weighted_pred, v.val_target, v.val_target_mask, std)

    cr_claim, cr_cap = conflict_resolved_ec_claims(v.val_affinity)
    cr_pred, cr_fb = wdec.dynamic_prediction_from_claims(v.val_forecasts, cr_claim)
    cr_metrics = metric_from(cr_pred, v.val_target, v.val_target_mask, std)

    token_claim = wdec.dynamic_token_claims(v.val_affinity)
    token_pred, token_fb = wdec.dynamic_prediction_from_claims(v.val_forecasts, token_claim)
    token_metrics = metric_from(token_pred, v.val_target, v.val_target_mask, std)

    frozen_pred, _ = frozen_hv_prediction(bundle, forecasts_val=v.val_forecasts)
    frozen_metrics = static_metric_values(bundle, frozen_pred)

    cr_rates = claim_rate_stats(cr_claim)
    existing_rates = claim_rate_stats(v.val_ec_claim)

    # Assignment-change rate: Conflict-Resolved's per-cell expert vs Dynamic
    # Token's per-cell expert (both are exactly-one-expert-per-cell schemes,
    # so a direct comparison isolates the effect of the capacity constraint).
    cr_assign = cr_claim.to(torch.long).argmax(dim=-1)
    token_assign = token_claim.to(torch.long).argmax(dim=-1)
    assignment_change_rate_vs_token = float((cr_assign != token_assign).to(torch.float32).mean())

    # Adjacent-window churn for Conflict-Resolved claims (diagnostic, same
    # definition used throughout this family).
    n = cr_claim.shape[0]
    e = cr_claim.shape[-1]
    if n > 1:
        flat = cr_claim.reshape(n, -1, e)
        changed_adj = (flat[1:] ^ flat[:-1]).to(torch.float32).mean(dim=(1, 2))
        mean_adjacent_change = float(changed_adj.mean())
    else:
        mean_adjacent_change = 0.0

    deltas = {
        "conflict_resolved_minus_weighted": cr_metrics["mae"] - weighted_metrics["mae"],
        "conflict_resolved_minus_existing": cr_metrics["mae"] - existing_metrics["mae"],
        "conflict_resolved_minus_token": cr_metrics["mae"] - token_metrics["mae"],
        "conflict_resolved_minus_frozen_hv": cr_metrics["mae"] - frozen_metrics["mae"],
    }

    dependence = []
    for label, cand_mae, base_mae in (
        ("conflict_resolved_vs_weighted", cr_metrics["per_window_mae"], weighted_metrics["per_window_mae"]),
        ("conflict_resolved_vs_existing", cr_metrics["per_window_mae"], existing_metrics["per_window_mae"]),
        ("conflict_resolved_vs_token", cr_metrics["per_window_mae"], token_metrics["per_window_mae"]),
        ("conflict_resolved_vs_frozen_hv", cr_metrics["per_window_mae"], frozen_metrics.get("per_window_mae", None)),
    ):
        if base_mae is None:
            mae_t = torch.as_tensor(wdec.sample_mae(frozen_pred, v.val_target, v.val_target_mask, std))
            base_mae = mae_t
        boot = wdec.block_bootstrap_with_prob(cand_mae, base_mae, block=BLOCK_LENGTH, seed=BOOTSTRAP_SEED, samples=BOOTSTRAP_SAMPLES)
        phase = wdec.every_kth_phase_bootstrap(cand_mae - base_mae, k=PHASE_K, seed=BOOTSTRAP_SEED, samples=BOOTSTRAP_SAMPLES)
        dependence.append({"dataset": dataset, "split": "router_val", "comparison": label, "test": f"block_len_{BLOCK_LENGTH}", **boot})
        dependence.append({"dataset": dataset, "split": "router_val", "comparison": label, "test": f"every_{PHASE_K}th_phase", **phase})

    return {
        "dataset": dataset,
        "predictions": {
            "dynamic_token_top1": {"mae": token_metrics["mae"], "mse": token_metrics["mse"], "fallback_rate": token_fb},
            "frozen_hv": {"mae": frozen_metrics["mae"], "mse": frozen_metrics["mse"]},
            "dynamic_ec_existing": {"mae": existing_metrics["mae"], "mse": existing_metrics["mse"], "fallback_rate": existing_fb},
            "affinity_weighted_ec": {"mae": weighted_metrics["mae"], "mse": weighted_metrics["mse"]},
            "conflict_resolved_ec": {"mae": cr_metrics["mae"], "mse": cr_metrics["mse"], "fallback_rate": cr_fb, "capacity_per_expert": cr_cap},
        },
        "deltas": deltas,
        "claim_rates": {"conflict_resolved": cr_rates, "existing_dynamic_ec": existing_rates},
        "assignment_change_rate_vs_dynamic_token": assignment_change_rate_vs_token,
        "mean_adjacent_window_claim_change_fraction": mean_adjacent_change,
        "dependence": dependence,
    }


def main() -> None:
    start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    source_hash_before = sha256_file(Path(__file__))
    manifest: dict[str, Any] = {
        "experiment": "conflict_resolved_expert_choice_hv",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "datasets": list(DATASETS),
        "development_datasets_disclaimer": "ETTh1/ETTh2/ETTm1/Weather/Electricity development datasets only. No test. No untouched-confirmation datasets.",
        "no_retraining_note": "The competence scorer is not retrained or touched. Loads window_dependent_expert_choice_hv/tensors.pt (oof_affinity, val_affinity, val_ec_claim) directly, same as affinity_weighted_expert_choice_hv.",
        "new_mechanism": "Conflict-resolved assignment via expert-proposing deferred acceptance over the SAME affinity tensor and CF=1 capacity; implemented as a provably-equivalent vectorized greedy pass, empirically verified against a literal round-based simulation per dataset.",
        "predeclared_gate": f"If Conflict-Resolved EC does not beat Affinity-Weighted EC OOF MAE on >= {OOF_GATE_MIN_WINS}/5 datasets, STOP and do not evaluate router_val.",
        "capacity_formula": "C = round(H*V/E); asserted C*E == M exactly (no fallback).",
        "bootstrap": {"block_length_primary": BLOCK_LENGTH, "samples": BOOTSTRAP_SAMPLES, "phase_k": PHASE_K, "seed": BOOTSTRAP_SEED},
        "source_sha256_before": source_hash_before,
        "test_set_accessed": False, "test_cache_loaded": False, "test_metrics_computed": False,
        "untouched_confirmation_accessed": False,
    }
    write_json(OUT_DIR / "method_manifest.json", manifest)

    oof_results: dict[str, Any] = {}
    for dataset in DATASETS:
        oof_results[dataset] = run_dataset_oof(dataset)

    oof_wins = sum(1 for d in DATASETS if oof_results[d]["oof"]["conflict_resolved_beats_weighted"])
    gate_pass = oof_wins >= OOF_GATE_MIN_WINS
    print(f"OOF wins (Conflict-Resolved beats Affinity-Weighted): {oof_wins}/5", flush=True)
    print(f"OOF GATE: {'PASS' if gate_pass else 'FAIL'}", flush=True)

    oof_dependence_rows = []
    for dataset in DATASETS:
        r = oof_results[dataset]["per_window_mae_for_dependence"]
        boot = wdec.block_bootstrap_with_prob(r["conflict_resolved"], r["affinity_weighted"], block=BLOCK_LENGTH, seed=BOOTSTRAP_SEED, samples=BOOTSTRAP_SAMPLES)
        phase = wdec.every_kth_phase_bootstrap(r["conflict_resolved"] - r["affinity_weighted"], k=PHASE_K, seed=BOOTSTRAP_SEED, samples=BOOTSTRAP_SAMPLES)
        oof_dependence_rows.append({"dataset": dataset, "split": "router_train_oof", "comparison": "conflict_resolved_vs_weighted", "test": f"block_len_{BLOCK_LENGTH}", **boot})
        oof_dependence_rows.append({"dataset": dataset, "split": "router_train_oof", "comparison": "conflict_resolved_vs_weighted", "test": f"every_{PHASE_K}th_phase", **phase})

    write_json(OUT_DIR / "oof_results.json", jsonable({
        "gate": {"predeclared_min_wins": OOF_GATE_MIN_WINS, "oof_wins": oof_wins, "gate_pass": gate_pass},
        "datasets": {d: {k: v for k, v in oof_results[d].items() if k not in ("loaded", "existing_claim", "cr_claim", "per_window_mae_for_dependence")} for d in DATASETS},
    }))
    write_csv_rows(OUT_DIR / "oof_dependence_tests.csv", oof_dependence_rows)

    claim_rate_rows = []
    for dataset in DATASETS:
        for method, rates in oof_results[dataset]["claim_rates"].items():
            claim_rate_rows.append({"dataset": dataset, "split": "router_train_oof", "method": method, **rates})
    write_csv_rows(OUT_DIR / "claim_rate_stats.csv", claim_rate_rows)

    integrity: dict[str, Any] = {
        "oof_gate": {"oof_wins": oof_wins, "required": OOF_GATE_MIN_WINS, "pass": gate_pass},
        "deferred_acceptance_verification_all_datasets": {d: oof_results[d]["deferred_acceptance_verification"] for d in DATASETS},
        "conflict_resolved_capacity_exact_per_dataset": {d: oof_results[d]["capacity_per_expert"] for d in DATASETS},
        "conflict_resolved_claim_rates_zero_and_multi_are_zero": {
            d: bool(
                oof_results[d]["claim_rates"]["conflict_resolved"].get("fraction_0_claim_cells", 0.0) == 0.0
                and oof_results[d]["claim_rates"]["conflict_resolved"].get("fraction_1_claim_cells", 0.0) == 1.0
            )
            for d in DATASETS
        },
        "TEST_SET_ACCESSED": "NO", "TEST_CACHE_LOADED": "NO", "TEST_METRICS_COMPUTED": "NO",
        "UNTOUCHED_CONFIRMATION_ACCESSED": "NO",
        "ROUTER_VAL_ACCESSED_FOR_THIS_EXPERIMENT": "NO" if not gate_pass else "YES",
    }

    val_results: dict[str, Any] = {}
    val_dependence_rows: list[dict[str, Any]] = []
    val_claim_rate_rows: list[dict[str, Any]] = []
    source_hash_after_oof = sha256_file(Path(__file__))

    if gate_pass:
        for dataset in DATASETS:
            val_results[dataset] = run_dataset_val(dataset, oof_results[dataset])
            val_dependence_rows.extend(val_results[dataset]["dependence"])
            for method, rates in val_results[dataset]["claim_rates"].items():
                val_claim_rate_rows.append({"dataset": dataset, "split": "router_val", "method": method, **rates})

        write_json(OUT_DIR / "validation_results.json", jsonable({
            "gate_pass": gate_pass,
            "datasets": {d: val_results[d] for d in DATASETS},
        }))
        write_csv_rows(OUT_DIR / "val_dependence_tests.csv", val_dependence_rows)
        write_csv_rows(OUT_DIR / "claim_rate_stats.csv", claim_rate_rows + val_claim_rate_rows)

    source_hash_after = sha256_file(Path(__file__))
    integrity["source_hash_before"] = manifest["source_sha256_before"]
    integrity["source_hash_after_oof_stage"] = source_hash_after_oof
    integrity["source_hash_after"] = source_hash_after
    integrity["source_unchanged_throughout"] = bool(manifest["source_sha256_before"] == source_hash_after_oof == source_hash_after)
    integrity["all_pass"] = bool(
        integrity["source_unchanged_throughout"]
        and all(oof_results[d]["deferred_acceptance_verification"]["all_match"] for d in DATASETS)
        and all(integrity["conflict_resolved_claim_rates_zero_and_multi_are_zero"].values())
    )
    write_json(OUT_DIR / "integrity_checks.json", integrity)

    verdict = "CONTINUE" if gate_pass else "STOP"
    make_report(oof_results, oof_wins, gate_pass, val_results, verdict)

    manifest["oof_gate_result"] = {"oof_wins": oof_wins, "gate_pass": gate_pass}
    manifest["runtime_sec"] = time.time() - start
    write_json(OUT_DIR / "method_manifest.json", manifest)

    print(f"OOF GATE: {'PASS' if gate_pass else 'FAIL'}")
    print(f"ROUTER_VAL ACCESSED FOR THIS EXPERIMENT: {'YES' if gate_pass else 'NO'}")
    print("TEST ACCESSED: NO")
    print("UNTOUCHED CONFIRMATION ACCESSED: NO")
    print(f"VERDICT: {verdict}")


def fmt(d: Mapping[str, Any], key: str, field: str = "mae") -> str:
    return f"`{d[key][field]:.6f}`"


def make_report(oof_results: Mapping[str, Any], oof_wins: int, gate_pass: bool, val_results: Mapping[str, Any], verdict: str) -> None:
    lines = [
        f"Final classification: {'GATE_PASS_' + verdict if gate_pass else 'OOF_GATE_FAIL_STOP'}",
        "",
        "# Conflict-Resolved Window-Dependent Expert Choice",
        "",
        "Development experiment (ETTh1/ETTh2/ETTm1/Weather/Electricity only). Reuses the frozen "
        "window_dependent_expert_choice_hv score/affinity tensors with NO retraining. The only new "
        "mechanism is conflict-resolved (deferred-acceptance-equivalent) assignment: every cell held by "
        "exactly one expert.",
        "",
        "```text",
        "TEST SET ACCESSED: NO",
        "TEST CACHE LOADED: NO",
        "TEST METRICS COMPUTED: NO",
        "UNTOUCHED CONFIRMATION ACCESSED: NO",
        "```",
        "",
        f"## Router-train OOF (primary evidence) -- Conflict-Resolved vs Affinity-Weighted: `{oof_wins}/5` wins, gate requires >=3/5",
        "",
        "| Dataset | Dynamic EC (existing) | Affinity-Weighted EC | Conflict-Resolved EC | CR-Weighted delta | CR beats Weighted |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for dataset, r in oof_results.items():
        o = r["oof"]
        lines.append(
            f"| {dataset} | `{o['dynamic_ec_existing']['mae']:.6f}` | `{o['affinity_weighted_ec']['mae']:.6f}` | "
            f"`{o['conflict_resolved_ec']['mae']:.6f}` | `{o['delta_conflict_resolved_minus_weighted']:+.6f}` | {o['conflict_resolved_beats_weighted']} |"
        )
    lines += ["", f"`OOF GATE: {'PASS' if gate_pass else 'FAIL'}` ({oof_wins}/5 >= 3/5 required)."]

    lines += ["", "## Deferred-acceptance verification (greedy vs literal round-based simulation)", "", "| Dataset | Windows checked | Mismatches | All match |", "|---|---:|---:|---|"]
    for dataset, r in oof_results.items():
        c = r["deferred_acceptance_verification"]
        lines.append(f"| {dataset} | {c['windows_checked']} | {c['mismatches']} | {c['all_match']} |")

    lines += ["", "## Conflict-Resolved EC claim rates (router_train OOF) -- expected zero_claim=0, multi_claim=0", "", "| Dataset | 0-claim | 1-claim | 2-claim | 3-claim |", "|---|---:|---:|---:|---:|"]
    for dataset, r in oof_results.items():
        cr = r["claim_rates"]["conflict_resolved"]
        lines.append(f"| {dataset} | `{cr.get('fraction_0_claim_cells', 0):.4f}` | `{cr.get('fraction_1_claim_cells', 0):.4f}` | `{cr.get('fraction_2_claim_cells', 0):.4f}` | `{cr.get('fraction_3_claim_cells', 0):.4f}` |")

    if not gate_pass:
        lines += [
            "",
            "## STOP -- OOF gate failed",
            "",
            f"Conflict-Resolved EC beat Affinity-Weighted EC OOF MAE on only {oof_wins}/5 datasets, below the "
            "predeclared minimum of 3/5. Per protocol, router_val was NEVER loaded for prediction/metric purposes "
            "in this run. This is reported as a valid negative result. Do not tune the assignment rule, capacity, "
            "or tie-break and rerun to try to pass the gate.",
            "",
            f"`ROUTER_VAL ACCESSED FOR THIS EXPERIMENT: NO`",
            f"`TEST ACCESSED: NO`",
            f"`UNTOUCHED CONFIRMATION ACCESSED: NO`",
            f"`VERDICT: STOP`",
        ]
        (OUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    lines += [
        "",
        "## Router-val (single pass, only because the OOF gate passed)",
        "",
        "| Dataset | Dynamic Token | Frozen HxV | Dynamic EC (existing) | Affinity-Weighted EC | Conflict-Resolved EC | CR-Weighted | CR-Existing | CR-Token | CR-Frozen |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, r in val_results.items():
        p = r["predictions"]
        d = r["deltas"]
        lines.append(
            f"| {dataset} | `{p['dynamic_token_top1']['mae']:.6f}` | `{p['frozen_hv']['mae']:.6f}` | `{p['dynamic_ec_existing']['mae']:.6f}` | "
            f"`{p['affinity_weighted_ec']['mae']:.6f}` | `{p['conflict_resolved_ec']['mae']:.6f}` | `{d['conflict_resolved_minus_weighted']:+.6f}` | "
            f"`{d['conflict_resolved_minus_existing']:+.6f}` | `{d['conflict_resolved_minus_token']:+.6f}` | `{d['conflict_resolved_minus_frozen_hv']:+.6f}` |"
        )

    lines += ["", "## Assignment-change rate (Conflict-Resolved vs Dynamic Token, router_val) and adjacent-window churn", "", "| Dataset | Change rate vs Token | Mean adjacent-window claim-change fraction |", "|---|---:|---:|"]
    for dataset, r in val_results.items():
        lines.append(f"| {dataset} | `{r['assignment_change_rate_vs_dynamic_token']:.4f}` | `{r['mean_adjacent_window_claim_change_fraction']:.4f}` |")

    lines += ["", "## Dependence-aware statistics (router_val, block-24 primary)", "", "| Dataset | Comparison | Test | Mean delta | 95% CI | CI excludes zero |", "|---|---|---|---:|---|---|"]
    for dataset, r in val_results.items():
        for row in r["dependence"]:
            lines.append(f"| {dataset} | {row['comparison']} | {row['test']} | `{row['mean_delta']:+.6f}` | [`{row['ci95_low']:+.6f}`, `{row['ci95_high']:+.6f}`] | {row['ci_excludes_zero']} |")

    lines += [
        "",
        f"`ROUTER_VAL ACCESSED FOR THIS EXPERIMENT: YES`",
        f"`TEST ACCESSED: NO`",
        f"`UNTOUCHED CONFIRMATION ACCESSED: NO`",
        f"`VERDICT: {verdict}`",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
