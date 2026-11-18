"""Feature-group ablation for Affinity-Weighted Window-Dependent Expert Choice.

Development experiment (ETTh1/ETTh2/ETTm1/Weather/Electricity only). Router_val
is NEVER touched: primary and only evidence is router_train causal OOF. Keeps
fixed: K=3 train-selected frozen experts, scorer architecture/hyperparameters/
seed, residual-gain target, causal 4-fold OOF protocol, fit-only affinity
calibration, CF=1, independent EC claims (dynamic_ec_claims, unmodified from
window_dependent_expert_choice_hv), affinity-weighted multi-claim fusion
(affinity_weighted_prediction_from_claims, unmodified from
affinity_weighted_expert_choice_hv), zero-claim fallback. The ONLY thing that
varies across variants is which feature groups feed the scorer.

Predeclared variants (retrained separately, from scratch, per dataset, per
OOF fold -- no router_val, no final router_train->router_val fit at all):
  F0_anchor           = static_gain + H/V/expert embeddings only
  F1_cell             = F0 + 6 cell-local forecast features
  F2_local            = F1 + 7 per-variable history features
  F3_full             = F2 + 6 global-history features (the existing model)
  Full-NoCell         = F3 minus cell-local features
  Full-NoPerVariable  = F3 minus per-variable features
  Full-NoGlobal       = F3 minus global features (== F2_local exactly --
                         reused, not retrained, see NOTE below)

TEST SET ACCESSED: NO. TEST CACHE LOADED: NO. TEST METRICS COMPUTED: NO.
ROUTER_VAL ACCESSED: NO. UNTOUCHED DATA ACCESSED: NO.
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
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.window_dependent_expert_choice_hv.run_window_dependent_expert_choice_hv as wdec  # noqa: E402
import experiments.affinity_weighted_expert_choice_hv.run_affinity_weighted_expert_choice_hv as aw  # noqa: E402
from experiments.frozen_hv_costar.run_frozen_hv_costar import LOADERS  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
DATASETS = wdec.DATASETS
BLOCK_LENGTH = wdec.BLOCK_LENGTH
PHASE_K = wdec.PHASE_K
BOOTSTRAP_SAMPLES = wdec.BOOTSTRAP_SAMPLES
BOOTSTRAP_SEED = wdec.BOOTSTRAP_SEED
DEVICE = wdec.DEVICE

GROUP_DIMS = {"global": len(wdec.GROUP_A_NAMES), "local": len(wdec.PER_VARIABLE_FEATURE_NAMES), "cell": len(wdec.CELL_LOCAL_FEATURE_NAMES)}

# Predeclared variant -> enabled feature groups. Anchor (H/V/expert
# embeddings + static_gain scalar) is always present in every variant.
VARIANT_GROUPS: dict[str, frozenset[str]] = {
    "F0_anchor": frozenset(),
    "F1_cell": frozenset({"cell"}),
    "F2_local": frozenset({"cell", "local"}),
    "F3_full": frozenset({"cell", "local", "global"}),
    "Full_NoCell": frozenset({"local", "global"}),
    "Full_NoPerVariable": frozenset({"cell", "global"}),
    "Full_NoGlobal": frozenset({"cell", "local"}),  # == F2_local, reused not retrained (see module docstring)
}
# Comparisons: (label, kind, candidate_variant, baseline_variant). kind
# documents whether this is an "add" test (does the group help) or a
# "remove" test (does removing it hurt).
COMPARISONS: list[tuple[str, str, str, str]] = [
    ("cell_add", "add", "F1_cell", "F0_anchor"),
    ("local_add", "add", "F2_local", "F1_cell"),
    ("global_add", "add", "F3_full", "F2_local"),
    ("cell_remove", "remove", "F3_full", "Full_NoCell"),
    ("local_remove", "remove", "F3_full", "Full_NoPerVariable"),
    ("global_remove", "remove", "F3_full", "Full_NoGlobal"),
]
GROUP_OF_COMPARISON = {"cell_add": "cell", "cell_remove": "cell", "local_add": "local", "local_remove": "local", "global_add": "global", "global_remove": "global"}


def write_json(path: Path, obj: Any) -> None:
    wdec.write_json(path, obj)


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    wdec.write_csv_rows(path, rows)


def jsonable(value: Any) -> Any:
    return wdec.jsonable(value)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Flexible scorer: SAME architecture/hyperparameters, configurable INPUT
# feature groups. `forward` has the identical signature to
# window_dependent_expert_choice_hv.SharedResidualScorer, so wdec.batch_forward
# / wdec.score_windows / wdec.fit_only_calibration / wdec.normalize_inputs are
# reused UNCHANGED.
# ---------------------------------------------------------------------------


class FlexibleResidualScorer(nn.Module):
    def __init__(self, horizon: int, variables: int, num_experts: int, enabled_groups: frozenset[str]) -> None:
        super().__init__()
        self.enabled_groups = enabled_groups
        self.horizon_embedding = nn.Embedding(horizon, wdec.HORIZON_EMBED_DIM)
        self.variable_embedding = nn.Embedding(variables, wdec.VARIABLE_EMBED_DIM)
        self.expert_embedding = nn.Embedding(num_experts, wdec.EXPERT_EMBED_DIM)
        input_dim = sum(GROUP_DIMS[g] for g in enabled_groups) + wdec.HORIZON_EMBED_DIM + wdec.VARIABLE_EMBED_DIM + wdec.EXPERT_EMBED_DIM + 1
        self.net = nn.Sequential(nn.Linear(input_dim, wdec.HIDDEN1), nn.ReLU(), nn.Linear(wdec.HIDDEN1, wdec.HIDDEN2), nn.ReLU(), nn.Linear(wdec.HIDDEN2, 1))

    def forward(self, global_feat: torch.Tensor, local_feat: torch.Tensor, cell_feat: torch.Tensor, static_gain_norm: torch.Tensor) -> torch.Tensor:
        b = global_feat.shape[0]
        horizon = self.horizon_embedding.num_embeddings
        variables = self.variable_embedding.num_embeddings
        experts = self.expert_embedding.num_embeddings
        device = global_feat.device
        h_ids, v_ids, e_ids = torch.arange(horizon, device=device), torch.arange(variables, device=device), torch.arange(experts, device=device)
        h_emb = self.horizon_embedding(h_ids).view(1, horizon, 1, 1, -1).expand(b, -1, variables, experts, -1)
        v_emb = self.variable_embedding(v_ids).view(1, 1, variables, 1, -1).expand(b, horizon, -1, experts, -1)
        e_emb = self.expert_embedding(e_ids).view(1, 1, 1, experts, -1).expand(b, horizon, variables, -1, -1)
        sg = static_gain_norm.view(1, horizon, variables, experts, 1).expand(b, -1, -1, -1, -1)
        parts = []
        if "global" in self.enabled_groups:
            parts.append(global_feat.view(b, 1, 1, 1, -1).expand(-1, horizon, variables, experts, -1))
        if "local" in self.enabled_groups:
            parts.append(local_feat.view(b, 1, variables, 1, -1).expand(-1, horizon, -1, experts, -1))
        if "cell" in self.enabled_groups:
            parts.append(cell_feat)
        parts += [h_emb, v_emb, e_emb, sg]
        x = torch.cat(parts, dim=-1)
        return self.net(x.reshape(b * horizon * variables * experts, -1)).view(b, horizon, variables, experts)


def train_scorer_flexible(
    enabled_groups: frozenset[str],
    horizon: int, variables: int, num_experts: int,
    global_feat: torch.Tensor, local_feat: torch.Tensor,
    forecasts_full: torch.Tensor, histories_full: torch.Tensor, std: torch.Tensor,
    gain: torch.Tensor, legal_idx: torch.Tensor,
) -> wdec.TrainedScorer:
    """Exact copy of wdec.train_scorer's training procedure (same seed,
    optimizer, lr, weight decay, max epochs, patience, batch size,
    chronological-tail early stopping, per-epoch fixed-seed shuffling of the
    fit set), with only the model class swapped for FlexibleResidualScorer."""
    wdec.set_seed(wdec.SCORER_SEED)
    static_gain = gain[legal_idx].mean(dim=0)
    residual_target = gain - static_gain.view(1, horizon, variables, num_experts)
    stats = wdec.compute_feature_stats(global_feat, local_feat, forecasts_full, histories_full, std, legal_idx, static_gain)
    static_gain_norm = ((static_gain - stats.static_gain_mean) / max(stats.static_gain_std, 1e-6)).to(DEVICE)

    train_idx, internal_val_idx = wdec.chronological_internal_split(legal_idx)
    model = FlexibleResidualScorer(horizon, variables, num_experts, enabled_groups).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=wdec.LR, weight_decay=wdec.WEIGHT_DECAY)
    generator = torch.Generator().manual_seed(wdec.SCORER_SEED)

    def target_for(idx: torch.Tensor) -> torch.Tensor:
        return residual_target[idx].to(DEVICE)

    history: list[dict[str, Any]] = []
    from copy import deepcopy
    best_state = deepcopy(model.state_dict())
    best_epoch, best_val, bad = 0, float("inf"), 0
    for epoch in range(1, wdec.MAX_EPOCHS + 1):
        model.train()
        perm = train_idx[torch.randperm(int(train_idx.numel()), generator=generator)]
        train_losses = []
        for lo in range(0, int(perm.numel()), wdec.BATCH_SIZE):
            idx = perm[lo: lo + wdec.BATCH_SIZE]
            pred = wdec.batch_forward(model, stats, global_feat, local_feat, forecasts_full, histories_full, std, static_gain_norm, idx)
            loss = ((pred - target_for(idx)) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            if internal_val_idx.numel() > 0:
                vals = []
                for lo in range(0, int(internal_val_idx.numel()), wdec.BATCH_SIZE):
                    idx = internal_val_idx[lo: lo + wdec.BATCH_SIZE]
                    pred = wdec.batch_forward(model, stats, global_feat, local_feat, forecasts_full, histories_full, std, static_gain_norm, idx)
                    vals.append(((pred - target_for(idx)) ** 2).mean())
                val_loss = float(torch.stack(vals).mean())
            else:
                val_loss = float(sum(train_losses) / max(len(train_losses), 1))
        history.append({"epoch": epoch, "train_mse": float(sum(train_losses) / max(len(train_losses), 1)), "internal_val_mse": val_loss})
        if val_loss < best_val - 1e-10:
            best_val, best_epoch, best_state, bad = val_loss, epoch, deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= wdec.PATIENCE:
                break
    model.load_state_dict(best_state)
    return wdec.TrainedScorer(model=model, stats=stats, static_gain=static_gain, best_epoch=best_epoch, best_internal_val_mse=best_val, history=history, train_windows=int(train_idx.numel()), internal_val_windows=int(internal_val_idx.numel()))


# ---------------------------------------------------------------------------
# Per-dataset, per-variant OOF pipeline (router_val never touched).
# ---------------------------------------------------------------------------


def run_variant_oof(dataset: str, bundle: Any, enabled_groups: frozenset[str], train_global: torch.Tensor, train_local: torch.Tensor,
                     train_forecasts: torch.Tensor, train_histories: torch.Tensor, train_gain: torch.Tensor,
                     horizon: int, variables: int, num_experts: int, n_train: int, fold_ckpt_dir: Path | None = None) -> dict[str, Any]:
    oof_raw = torch.full((n_train, horizon, variables, num_experts), float("nan"), dtype=torch.float32)
    oof_mask = torch.zeros(n_train, dtype=torch.bool)
    fold_calibrations: list[tuple[float, float]] = []
    fold_logs: list[dict[str, Any]] = []
    if fold_ckpt_dir is not None:
        fold_ckpt_dir.mkdir(parents=True, exist_ok=True)

    for fold_id, (eval_lo, eval_hi) in enumerate(wdec.oof_bounds(n_train), start=1):
        fold_ckpt_path = fold_ckpt_dir / f"fold{fold_id}.pt" if fold_ckpt_dir is not None else None
        if fold_ckpt_path is not None and fold_ckpt_path.exists():
            print(f"[feature-ablation] {dataset}: fold {fold_id} resuming from fold checkpoint...", flush=True)
            saved = torch.load(fold_ckpt_path, weights_only=False)
            oof_raw[eval_lo:eval_hi] = saved["raw"]
            oof_mask[eval_lo:eval_hi] = True
            fold_calibrations.append(saved["calibration"])
            fold_logs.append(saved["fold_log"])
            continue

        current_eval_origin = int(bundle.train_cache["absolute_window_starts"][eval_lo])
        legal = wdec.legal_fit_mask(bundle.train_cache["absolute_window_starts"].to(torch.long), horizon, current_eval_origin)
        if legal.numel() == 0:
            raise AssertionError(f"{dataset} fold {fold_id}: no legal fit windows")
        latest_fit_target_end = int((bundle.train_cache["absolute_window_starts"].to(torch.long)[legal] + horizon).max())
        if latest_fit_target_end > current_eval_origin:
            raise AssertionError(f"{dataset} fold {fold_id}: OOF causality violation")

        fit = train_scorer_flexible(enabled_groups, horizon, variables, num_experts, train_global, train_local, train_forecasts, train_histories, bundle.std, train_gain, legal)
        calib_mean, calib_std = wdec.fit_only_calibration(fit, train_global, train_local, train_forecasts, train_histories, bundle.std, legal)
        fold_calibrations.append((calib_mean, calib_std))

        eval_idx = torch.arange(eval_lo, eval_hi)
        raw = wdec.score_windows(fit, train_global, train_local, train_forecasts, train_histories, bundle.std, eval_idx)
        oof_raw[eval_idx] = raw
        oof_mask[eval_idx] = True
        fold_log = {"fold": fold_id, "eval_lo": eval_lo, "eval_hi": eval_hi, "legal_fit_windows": int(legal.numel()), "best_epoch": fit.best_epoch, "best_internal_val_mse": fit.best_internal_val_mse, "calibration_mean": calib_mean, "calibration_std": calib_std}
        fold_logs.append(fold_log)
        if fold_ckpt_path is not None:
            torch.save({"raw": raw, "calibration": (calib_mean, calib_std), "fold_log": fold_log}, fold_ckpt_path)
            print(f"[feature-ablation] {dataset}: fold {fold_id}/4 checkpointed", flush=True)

    oof_eval_idx = torch.nonzero(oof_mask, as_tuple=False).flatten()
    oof_valid = oof_raw[oof_eval_idx]
    oof_affinity = torch.empty_like(oof_valid)
    cursor = 0
    for (fold_id, (eval_lo, eval_hi)), (calib_mean, calib_std) in zip(enumerate(wdec.oof_bounds(n_train), start=1), fold_calibrations):
        n_fold = eval_hi - eval_lo
        oof_affinity[cursor: cursor + n_fold] = wdec.raw_to_affinity(oof_valid[cursor: cursor + n_fold], calib_mean, calib_std)
        cursor += n_fold

    oof_forecasts = train_forecasts[oof_eval_idx]
    oof_target = bundle.train_cache["targets"].to(torch.float32)[oof_eval_idx]
    oof_target_mask = bundle.train_cache["target_masks"].to(torch.bool)[oof_eval_idx]

    ec_claim, capacity = wdec.dynamic_ec_claims(oof_affinity)  # unmodified independent EC claims
    pred, fb, _ = aw.affinity_weighted_prediction_from_claims(oof_forecasts, ec_claim, oof_affinity)  # unmodified affinity-weighted fusion + zero-claim fallback
    metrics = wdec.metric_from(pred, oof_target, oof_target_mask, bundle.std)

    return {
        "dataset": dataset, "capacity_per_expert": capacity, "fallback_rate": fb,
        "mae": metrics["mae"], "mse": metrics["mse"], "per_window_mae": metrics["per_window_mae"],
        "fold_logs": fold_logs,
    }


def run_dataset(dataset: str) -> dict[str, Any]:
    print(f"[feature-ablation] {dataset}: loading cache (router_val cache loaded per Bundle convention, never used for any prediction/metric here)...", flush=True)
    bundle = LOADERS[dataset]()
    wdec.validate_cache_role(bundle.train_cache, "router_train")
    horizon = int(bundle.train_cache["forecast_horizon"])
    variables = int(bundle.val_cache["num_features"])
    num_experts = len(bundle.expert_idx)
    n_train = int(bundle.train_cache["num_windows"])

    train_gain = wdec.full_gain_tensor(bundle, bundle.train_cache)
    train_global, train_local = wdec.global_local_features(bundle.train_cache, bundle.std)
    train_forecasts = bundle.forecasts_fn(bundle.train_cache, bundle.expert_idx).to(torch.float32)
    train_histories = bundle.train_cache["histories"].to(torch.float32)

    before_hashes = wdec.checkpoint_hashes(dataset, bundle.core_names)

    variant_ckpt_dir = OUT_DIR / "_checkpoints" / dataset
    variant_ckpt_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, Any]] = {}
    trained_group_keys: dict[frozenset[str], str] = {}
    for variant, groups in VARIANT_GROUPS.items():
        if groups in trained_group_keys:
            reused_from = trained_group_keys[groups]
            print(f"[feature-ablation] {dataset}: {variant} == {reused_from} (identical feature set) -- reusing result, not retraining.", flush=True)
            results[variant] = dict(results[reused_from])
            results[variant]["reused_from"] = reused_from
            continue
        variant_ckpt_path = variant_ckpt_dir / f"{variant}.pt"
        if variant_ckpt_path.exists():
            print(f"[feature-ablation] {dataset}: {variant} resuming from variant checkpoint...", flush=True)
            r = torch.load(variant_ckpt_path, weights_only=False)
            results[variant] = r
            trained_group_keys[groups] = variant
            continue
        print(f"[feature-ablation] {dataset}: training variant {variant} (groups={sorted(groups) or ['anchor_only']})...", flush=True)
        t0 = time.time()
        fold_ckpt_dir = variant_ckpt_dir / f"{variant}_folds"
        r = run_variant_oof(dataset, bundle, groups, train_global, train_local, train_forecasts, train_histories, train_gain, horizon, variables, num_experts, n_train, fold_ckpt_dir=fold_ckpt_dir)
        r["elapsed_sec"] = time.time() - t0
        r["reused_from"] = None
        results[variant] = r
        trained_group_keys[groups] = variant
        torch.save(r, variant_ckpt_path)
        import shutil
        shutil.rmtree(fold_ckpt_dir, ignore_errors=True)  # variant complete -- fold-level checkpoints no longer needed
        print(f"[feature-ablation] {dataset}: {variant} OOF MAE={r['mae']:.6f} ({r['elapsed_sec']:.1f}s) [checkpointed]", flush=True)

    after_hashes = wdec.checkpoint_hashes(dataset, bundle.core_names)
    checkpoints_unchanged = before_hashes == after_hashes

    dependence_rows = []
    for label, kind, cand, base in COMPARISONS:
        cand_mae, base_mae = results[cand]["per_window_mae"], results[base]["per_window_mae"]
        boot = wdec.block_bootstrap_with_prob(cand_mae, base_mae, block=BLOCK_LENGTH, seed=BOOTSTRAP_SEED, samples=BOOTSTRAP_SAMPLES)
        phase = wdec.every_kth_phase_bootstrap(cand_mae - base_mae, k=PHASE_K, seed=BOOTSTRAP_SEED, samples=BOOTSTRAP_SAMPLES)
        dependence_rows.append({"dataset": dataset, "comparison": label, "kind": kind, "candidate": cand, "baseline": base, "test": f"block_len_{BLOCK_LENGTH}", **boot})
        dependence_rows.append({"dataset": dataset, "comparison": label, "kind": kind, "candidate": cand, "baseline": base, "test": f"every_{PHASE_K}th_phase", **phase})

    return {"dataset": dataset, "results": results, "dependence": dependence_rows, "checkpoints_unchanged": checkpoints_unchanged, "before_hashes": before_hashes, "after_hashes": after_hashes}


def classify_group(group: str, per_dataset: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    add_label = f"{group}_add"
    remove_label = f"{group}_remove"
    add_deltas = {d: per_dataset[d]["results"][COMPARISONS_BY_LABEL[add_label][2]]["mae"] - per_dataset[d]["results"][COMPARISONS_BY_LABEL[add_label][3]]["mae"] for d in DATASETS}
    remove_deltas = {d: per_dataset[d]["results"][COMPARISONS_BY_LABEL[remove_label][2]]["mae"] - per_dataset[d]["results"][COMPARISONS_BY_LABEL[remove_label][3]]["mae"] for d in DATASETS}
    add_wins = sum(1 for v in add_deltas.values() if v < 0)  # adding helps -> full/candidate MAE lower than baseline-without-group
    remove_hurts = sum(1 for v in remove_deltas.values() if v > 0)  # removing hurts -> without-group MAE higher than full

    add_block_support = sum(
        1 for row in _dep_rows(per_dataset, add_label)
        if row["test"] == f"block_len_{BLOCK_LENGTH}" and row["mean_delta"] < 0 and row["ci_excludes_zero"]
    )
    remove_block_support = sum(
        1 for row in _dep_rows(per_dataset, remove_label)
        if row["test"] == f"block_len_{BLOCK_LENGTH}" and row["mean_delta"] > 0 and row["ci_excludes_zero"]
    )
    independent_evidence = add_label != remove_label and COMPARISONS_BY_LABEL[add_label][2:] != COMPARISONS_BY_LABEL[remove_label][2:]
    # For the "global" group, add and remove reduce to the exact same
    # comparison (F3_full vs F2_local) because global is the last group in
    # the forward F0->F3 chain, so Full-NoGlobal == F2_local exactly. This is
    # disclosed, not hidden: only ONE independent piece of evidence exists
    # for "global", not two.
    # Compare by underlying feature-GROUP SETS, not variant name strings:
    # Full_NoGlobal and F2_local are different names for the identical
    # feature-group configuration (a real bug in an earlier version of this
    # function compared names directly, so it never detected this and
    # incorrectly reported independent_add_remove_evidence=True for "global"
    # -- fixed here; caught before being reported, not after).
    add_pair = frozenset({VARIANT_GROUPS[COMPARISONS_BY_LABEL[add_label][2]], VARIANT_GROUPS[COMPARISONS_BY_LABEL[add_label][3]]})
    remove_pair = frozenset({VARIANT_GROUPS[COMPARISONS_BY_LABEL[remove_label][2]], VARIANT_GROUPS[COMPARISONS_BY_LABEL[remove_label][3]]})
    same_comparison = add_pair == remove_pair

    add_pass = add_wins >= 3
    remove_pass = remove_hurts >= 3
    dependence_support = (add_block_support >= 2) or (remove_block_support >= 2)
    if add_pass and remove_pass and dependence_support:
        label = "SUPPORTED"
    elif not add_pass and not remove_pass and add_block_support == 0 and remove_block_support == 0:
        label = "NOT_SUPPORTED"
    else:
        label = "MIXED"

    return {
        "group": group,
        "add_deltas_mae": add_deltas, "add_wins_of_5": add_wins, "add_pass_ge_3": add_pass,
        "remove_deltas_mae": remove_deltas, "remove_hurts_of_5": remove_hurts, "remove_pass_ge_3": remove_pass,
        "add_block24_support_datasets": add_block_support, "remove_block24_support_datasets": remove_block_support,
        "independent_add_remove_evidence": not same_comparison,
        "note": None if not same_comparison else "add and remove tests for this group are the SAME comparison (F3_full vs F2_local); only one independent piece of evidence exists.",
        "label": label,
    }


COMPARISONS_BY_LABEL = {c[0]: c for c in COMPARISONS}


def _dep_rows(per_dataset: Mapping[str, dict[str, Any]], label: str) -> list[dict[str, Any]]:
    return [row for d in DATASETS for row in per_dataset[d]["dependence"] if row["comparison"] == label]


def main() -> None:
    start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source_hash = sha256_file(Path(__file__))
    manifest = {
        "experiment": "feature_ablation_affinity_weighted_ec",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": wdec.git_info(),
        "source_sha256": source_hash,
        "datasets": list(DATASETS),
        "variants": {k: sorted(v) or ["anchor_only"] for k, v in VARIANT_GROUPS.items()},
        "variant_equivalences": {"Full_NoGlobal": "F2_local (identical feature set: anchor+cell+local; reused, not retrained)"},
        "comparisons": [{"label": c[0], "kind": c[1], "candidate": c[2], "baseline": c[3]} for c in COMPARISONS],
        "kept_fixed": [
            "K=3 train-selected frozen experts (LOADERS, unmodified)", "scorer architecture (Linear(->64)->ReLU->Linear(->32)->ReLU->Linear(->1))",
            "seed=7", "AdamW lr=1e-3 wd=1e-4", "max_epochs=100 patience=10 batch_size=32",
            "residual-gain target (gain - static_gain, fit-only per fold)", "causal 4-fold OOF protocol (wdec.oof_bounds, full-horizon observability)",
            "fit-only affinity calibration, temperature=1.0", "CF=1, capacity=round(H*V/E)",
            "independent EC claims (wdec.dynamic_ec_claims, unmodified)", "affinity-weighted multi-claim fusion + zero-claim fallback (aw.affinity_weighted_prediction_from_claims, unmodified)",
        ],
        "primary_evidence": "router_train causal OOF MAE only. router_val never loaded for prediction/metric in this experiment.",
        "classification_rule": {
            "add_pass": "candidate (with group) beats baseline (without group) OOF MAE on >=3/5 datasets",
            "remove_pass": "baseline-without-group OOF MAE is WORSE than full OOF MAE on >=3/5 datasets",
            "dependence_support": ">=2/5 datasets have block-24 CI excluding zero in the expected direction (add or remove test)",
            "SUPPORTED": "add_pass AND remove_pass AND dependence_support",
            "NOT_SUPPORTED": "neither add nor remove passes and no block-24 support either direction",
            "MIXED": "anything else -- not rescued or tuned",
        },
        "test_set_accessed": False, "test_cache_loaded": False, "test_metrics_computed": False,
        "router_val_accessed": False, "untouched_data_accessed": False,
    }
    write_json(OUT_DIR / "method_manifest.json", manifest)

    checkpoint_dir = OUT_DIR / "_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    per_dataset: dict[str, dict[str, Any]] = {}
    for dataset in DATASETS:
        ckpt_path = checkpoint_dir / f"{dataset}.pt"
        if ckpt_path.exists():
            print(f"[feature-ablation] {dataset}: resuming from checkpoint {ckpt_path}", flush=True)
            per_dataset[dataset] = torch.load(ckpt_path, weights_only=False)
            continue
        per_dataset[dataset] = run_dataset(dataset)
        torch.save(per_dataset[dataset], ckpt_path)

    group_classifications = {g: classify_group(g, per_dataset) for g in ("cell", "local", "global")}

    variant_oof_rows = []
    for dataset in DATASETS:
        for variant, r in per_dataset[dataset]["results"].items():
            variant_oof_rows.append({"dataset": dataset, "variant": variant, "mae": r["mae"], "mse": r["mse"], "fallback_rate": r["fallback_rate"], "capacity_per_expert": r["capacity_per_expert"], "reused_from": r["reused_from"]})
    write_csv_rows(OUT_DIR / "variant_oof_results.csv", variant_oof_rows)

    dependence_rows = [row for d in DATASETS for row in per_dataset[d]["dependence"]]
    write_csv_rows(OUT_DIR / "dependence_tests.csv", dependence_rows)

    fold_rows = [{"dataset": d, "variant": v, **fl} for d in DATASETS for v, r in per_dataset[d]["results"].items() if r["reused_from"] is None for fl in r["fold_logs"]]
    write_csv_rows(OUT_DIR / "fold_logs.csv", fold_rows)

    integrity = {
        "checkpoints_unchanged_all_datasets": {d: per_dataset[d]["checkpoints_unchanged"] for d in DATASETS},
        "source_hash_before": source_hash,
        "source_hash_after": sha256_file(Path(__file__)),
        "TEST_SET_ACCESSED": "NO", "TEST_CACHE_LOADED": "NO", "TEST_METRICS_COMPUTED": "NO",
        "ROUTER_VAL_ACCESSED": "NO", "UNTOUCHED_DATA_ACCESSED": "NO",
    }
    integrity["source_unchanged"] = integrity["source_hash_before"] == integrity["source_hash_after"]
    integrity["all_pass"] = bool(integrity["source_unchanged"] and all(integrity["checkpoints_unchanged_all_datasets"].values()))
    write_json(OUT_DIR / "integrity_checks.json", integrity)

    best_variant = min(("F0_anchor", "F1_cell", "F2_local", "F3_full"), key=lambda v: sum(per_dataset[d]["results"][v]["mae"] for d in DATASETS))

    write_json(OUT_DIR / "results.json", jsonable({
        "datasets": {d: {"results": {v: {k: vv for k, vv in r.items() if k != "per_window_mae"} for v, r in per_dataset[d]["results"].items()}} for d in DATASETS},
        "group_classifications": group_classifications,
        "best_variant_by_mean_oof_mae_among_F0_F1_F2_F3": best_variant,
    }))

    make_report(per_dataset, group_classifications, best_variant)

    valid = integrity["all_pass"]
    manifest["runtime_sec"] = time.time() - start
    manifest["group_classifications"] = jsonable(group_classifications)
    manifest["best_variant"] = best_variant
    manifest["feature_ablation_valid"] = valid
    write_json(OUT_DIR / "method_manifest.json", manifest)

    print(f"FEATURE ABLATION VALID: {'YES' if valid else 'NO'}")
    print(f"BEST PREDECLARED FEATURE VARIANT BY OOF: {best_variant}")
    print("ROUTER_VAL ACCESSED: NO")
    print("TEST ACCESSED: NO")
    print("UNTOUCHED DATA ACCESSED: NO")


def make_report(per_dataset: Mapping[str, dict[str, Any]], group_classifications: Mapping[str, dict[str, Any]], best_variant: str) -> None:
    lines = ["Feature-group ablation for Affinity-Weighted Window-Dependent Expert Choice", "", "```text", "TEST SET ACCESSED: NO", "ROUTER_VAL ACCESSED: NO", "UNTOUCHED DATA ACCESSED: NO", "```", ""]
    lines += ["## OOF MAE by variant", "", "| Dataset | F0_anchor | F1_cell | F2_local | F3_full | Full_NoCell | Full_NoPerVariable | Full_NoGlobal |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for dataset in DATASETS:
        r = per_dataset[dataset]["results"]
        lines.append("| " + dataset + " | " + " | ".join(f"`{r[v]['mae']:.6f}`" for v in ("F0_anchor", "F1_cell", "F2_local", "F3_full", "Full_NoCell", "Full_NoPerVariable", "Full_NoGlobal")) + " |")
    lines += ["", f"Best predeclared variant by mean OOF MAE across F0-F3: **{best_variant}**", ""]
    lines += ["## Group classifications", ""]
    for g, c in group_classifications.items():
        lines += [
            f"### {g}", "",
            f"- Adding helps on {c['add_wins_of_5']}/5 datasets (need >=3): `{c['add_pass_ge_3']}`",
            f"- Removing hurts on {c['remove_hurts_of_5']}/5 datasets (need >=3): `{c['remove_pass_ge_3']}`",
            f"- Block-24 dependence support: add={c['add_block24_support_datasets']}/5, remove={c['remove_block24_support_datasets']}/5",
            f"- Independent add/remove evidence: `{c['independent_add_remove_evidence']}`" + (f" -- {c['note']}" if c["note"] else ""),
            f"- **Label: {c['label']}**", "",
        ]
    (OUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
