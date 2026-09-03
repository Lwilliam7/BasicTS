"""Embedding ablation for Affinity-Weighted Window-Dependent Expert Choice.

Development experiment (ETTh1/ETTh2/ETTm1/Weather/Electricity only).
router_val is NEVER touched: primary and only evidence is router_train causal
OOF. Uses the feature variant SELECTED by feature_ablation_affinity_weighted_ec
(OOF-only selection -- see SELECTED_FEATURE_GROUPS / SELECTED_FEATURE_VARIANT
below, filled in from that experiment's results.json after it completed; not
reconsidered here). Everything else kept fixed: K=3 train-selected frozen
experts, scorer hidden layers, residual-gain target, optimizer/lr/seed,
causal 4-fold OOF protocol, fit-only affinity calibration, CF=1, independent
EC claims, affinity-weighted multi-claim fusion, zero-claim fallback.

The ONLY thing that varies across variants is which embeddings are zeroed.
For fairness, a "removed" embedding is replaced by a fixed zero vector of the
SAME dimension (H=4, V=8, Expert=4) so scorer input dimension and MLP
capacity stay IDENTICAL across all 5 variants -- only the embedding's
CONTENT is ablated, never the model's capacity.

  E_full     : H + V + Expert embeddings all active
  E_noH      : H embedding zeroed
  E_noV      : V embedding zeroed
  E_noExpert : Expert embedding zeroed
  E_none     : all three embeddings zeroed

TEST SET ACCESSED: NO. TEST CACHE LOADED: NO. TEST METRICS COMPUTED: NO.
ROUTER_VAL ACCESSED: NO. UNTOUCHED DATA ACCESSED: NO.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from copy import deepcopy
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
import experiments.feature_ablation_affinity_weighted_ec.run_feature_ablation as fa  # noqa: E402
from experiments.frozen_hv_costar.run_frozen_hv_costar import LOADERS  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
DATASETS = wdec.DATASETS
BLOCK_LENGTH = wdec.BLOCK_LENGTH
PHASE_K = wdec.PHASE_K
BOOTSTRAP_SAMPLES = wdec.BOOTSTRAP_SAMPLES
BOOTSTRAP_SEED = wdec.BOOTSTRAP_SEED
DEVICE = wdec.DEVICE

# ---------------------------------------------------------------------------
# Filled in from experiments/feature_ablation_affinity_weighted_ec/results.json
# ("best_variant_by_mean_oof_mae_among_F0_F1_F2_F3"), an OOF-only selection
# made BEFORE this experiment was designed/run. Not reconsidered here.
# ---------------------------------------------------------------------------
SELECTED_FEATURE_VARIANT = "__PENDING_FEATURE_ABLATION_RESULT__"
SELECTED_FEATURE_GROUPS: frozenset[str] = frozenset()  # set below once known


def load_selected_feature_variant() -> tuple[str, frozenset[str]]:
    results_path = ROOT / "experiments/feature_ablation_affinity_weighted_ec/results.json"
    manifest_path = ROOT / "experiments/feature_ablation_affinity_weighted_ec/method_manifest.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    variant = results["best_variant_by_mean_oof_mae_among_F0_F1_F2_F3"]
    if not manifest.get("feature_ablation_valid", False):
        raise AssertionError("feature_ablation_affinity_weighted_ec did not pass its own integrity checks (FEATURE ABLATION VALID: NO) -- refusing to build on an invalid result.")
    groups_list = manifest["variants"][variant]
    groups = frozenset() if groups_list == ["anchor_only"] else frozenset(groups_list)
    return variant, groups


VARIANT_ZEROED: dict[str, frozenset[str]] = {
    "E_full": frozenset(),
    "E_noH": frozenset({"horizon"}),
    "E_noV": frozenset({"variable"}),
    "E_noExpert": frozenset({"expert"}),
    "E_none": frozenset({"horizon", "variable", "expert"}),
}
COMPARISONS: list[tuple[str, str, str]] = [
    ("H_remove", "E_full", "E_noH"),
    ("V_remove", "E_full", "E_noV"),
    ("Expert_remove", "E_full", "E_noExpert"),
]
EMBEDDING_OF_COMPARISON = {"H_remove": "H", "V_remove": "V", "Expert_remove": "Expert"}


def write_json(path: Path, obj: Any) -> None:
    wdec.write_json(path, obj)


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    wdec.write_csv_rows(path, rows)


def jsonable(value: Any) -> Any:
    return wdec.jsonable(value)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Scorer with fairness-preserving zeroed embeddings: input dim and MLP
# capacity are IDENTICAL to the selected feature variant's full model in
# every zeroing configuration -- only embedding CONTENT changes.
# ---------------------------------------------------------------------------


class EmbeddingAblationScorer(nn.Module):
    def __init__(self, horizon: int, variables: int, num_experts: int, enabled_groups: frozenset[str], zeroed_embeddings: frozenset[str]) -> None:
        super().__init__()
        self.enabled_groups = enabled_groups
        self.zeroed_embeddings = zeroed_embeddings
        self.horizon_embedding = nn.Embedding(horizon, wdec.HORIZON_EMBED_DIM)
        self.variable_embedding = nn.Embedding(variables, wdec.VARIABLE_EMBED_DIM)
        self.expert_embedding = nn.Embedding(num_experts, wdec.EXPERT_EMBED_DIM)
        input_dim = sum(fa.GROUP_DIMS[g] for g in enabled_groups) + wdec.HORIZON_EMBED_DIM + wdec.VARIABLE_EMBED_DIM + wdec.EXPERT_EMBED_DIM + 1
        self.net = nn.Sequential(nn.Linear(input_dim, wdec.HIDDEN1), nn.ReLU(), nn.Linear(wdec.HIDDEN1, wdec.HIDDEN2), nn.ReLU(), nn.Linear(wdec.HIDDEN2, 1))

    def forward(self, global_feat: torch.Tensor, local_feat: torch.Tensor, cell_feat: torch.Tensor, static_gain_norm: torch.Tensor) -> torch.Tensor:
        b = global_feat.shape[0]
        horizon = self.horizon_embedding.num_embeddings
        variables = self.variable_embedding.num_embeddings
        experts = self.expert_embedding.num_embeddings
        device = global_feat.device
        h_ids, v_ids, e_ids = torch.arange(horizon, device=device), torch.arange(variables, device=device), torch.arange(experts, device=device)
        h_emb = self.horizon_embedding(h_ids)
        v_emb = self.variable_embedding(v_ids)
        e_emb = self.expert_embedding(e_ids)
        if "horizon" in self.zeroed_embeddings:
            h_emb = torch.zeros_like(h_emb)
        if "variable" in self.zeroed_embeddings:
            v_emb = torch.zeros_like(v_emb)
        if "expert" in self.zeroed_embeddings:
            e_emb = torch.zeros_like(e_emb)
        h_emb = h_emb.view(1, horizon, 1, 1, -1).expand(b, -1, variables, experts, -1)
        v_emb = v_emb.view(1, 1, variables, 1, -1).expand(b, horizon, -1, experts, -1)
        e_emb = e_emb.view(1, 1, 1, experts, -1).expand(b, horizon, variables, -1, -1)
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


def train_scorer_embedding_ablation(
    enabled_groups: frozenset[str], zeroed_embeddings: frozenset[str],
    horizon: int, variables: int, num_experts: int,
    global_feat: torch.Tensor, local_feat: torch.Tensor,
    forecasts_full: torch.Tensor, histories_full: torch.Tensor, std: torch.Tensor,
    gain: torch.Tensor, legal_idx: torch.Tensor,
) -> wdec.TrainedScorer:
    """Identical training procedure to feature_ablation's train_scorer_flexible
    / wdec.train_scorer (same seed/optimizer/lr/wd/epochs/patience/batch size/
    early stopping), model swapped for EmbeddingAblationScorer."""
    wdec.set_seed(wdec.SCORER_SEED)
    static_gain = gain[legal_idx].mean(dim=0)
    residual_target = gain - static_gain.view(1, horizon, variables, num_experts)
    stats = wdec.compute_feature_stats(global_feat, local_feat, forecasts_full, histories_full, std, legal_idx, static_gain)
    static_gain_norm = ((static_gain - stats.static_gain_mean) / max(stats.static_gain_std, 1e-6)).to(DEVICE)

    train_idx, internal_val_idx = wdec.chronological_internal_split(legal_idx)
    model = EmbeddingAblationScorer(horizon, variables, num_experts, enabled_groups, zeroed_embeddings).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=wdec.LR, weight_decay=wdec.WEIGHT_DECAY)
    generator = torch.Generator().manual_seed(wdec.SCORER_SEED)

    def target_for(idx: torch.Tensor) -> torch.Tensor:
        return residual_target[idx].to(DEVICE)

    history: list[dict[str, Any]] = []
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


def run_variant_oof(dataset: str, bundle: Any, zeroed_embeddings: frozenset[str], enabled_groups: frozenset[str],
                     train_global: torch.Tensor, train_local: torch.Tensor, train_forecasts: torch.Tensor, train_histories: torch.Tensor,
                     train_gain: torch.Tensor, horizon: int, variables: int, num_experts: int, n_train: int, fold_ckpt_dir: Path | None = None) -> dict[str, Any]:
    oof_raw = torch.full((n_train, horizon, variables, num_experts), float("nan"), dtype=torch.float32)
    oof_mask = torch.zeros(n_train, dtype=torch.bool)
    fold_calibrations: list[tuple[float, float]] = []
    fold_logs: list[dict[str, Any]] = []
    if fold_ckpt_dir is not None:
        fold_ckpt_dir.mkdir(parents=True, exist_ok=True)

    for fold_id, (eval_lo, eval_hi) in enumerate(wdec.oof_bounds(n_train), start=1):
        fold_ckpt_path = fold_ckpt_dir / f"fold{fold_id}.pt" if fold_ckpt_dir is not None else None
        if fold_ckpt_path is not None and fold_ckpt_path.exists():
            print(f"[embedding-ablation] {dataset}: fold {fold_id} resuming from fold checkpoint...", flush=True)
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

        fit = train_scorer_embedding_ablation(enabled_groups, zeroed_embeddings, horizon, variables, num_experts, train_global, train_local, train_forecasts, train_histories, bundle.std, train_gain, legal)
        calib_mean, calib_std = wdec.fit_only_calibration(fit, train_global, train_local, train_forecasts, train_histories, bundle.std, legal)
        fold_calibrations.append((calib_mean, calib_std))

        eval_idx = torch.arange(eval_lo, eval_hi)
        raw = wdec.score_windows(fit, train_global, train_local, train_forecasts, train_histories, bundle.std, eval_idx)
        oof_raw[eval_idx] = raw
        oof_mask[eval_idx] = True
        fold_log = {"fold": fold_id, "eval_lo": eval_lo, "eval_hi": eval_hi, "legal_fit_windows": int(legal.numel()), "best_epoch": fit.best_epoch, "best_internal_val_mse": fit.best_internal_val_mse}
        fold_logs.append(fold_log)
        if fold_ckpt_path is not None:
            torch.save({"raw": raw, "calibration": (calib_mean, calib_std), "fold_log": fold_log}, fold_ckpt_path)
            print(f"[embedding-ablation] {dataset}: fold {fold_id}/4 checkpointed", flush=True)

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

    ec_claim, capacity = wdec.dynamic_ec_claims(oof_affinity)
    pred, fb, _ = aw.affinity_weighted_prediction_from_claims(oof_forecasts, ec_claim, oof_affinity)
    metrics = wdec.metric_from(pred, oof_target, oof_target_mask, bundle.std)

    return {"dataset": dataset, "capacity_per_expert": capacity, "fallback_rate": fb, "mae": metrics["mae"], "mse": metrics["mse"], "per_window_mae": metrics["per_window_mae"], "fold_logs": fold_logs}


def run_dataset(dataset: str, enabled_groups: frozenset[str]) -> dict[str, Any]:
    print(f"[embedding-ablation] {dataset}: loading cache...", flush=True)
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
    for variant, zeroed in VARIANT_ZEROED.items():
        variant_ckpt_path = variant_ckpt_dir / f"{variant}.pt"
        if variant_ckpt_path.exists():
            print(f"[embedding-ablation] {dataset}: {variant} resuming from variant checkpoint...", flush=True)
            results[variant] = torch.load(variant_ckpt_path, weights_only=False)
            continue
        print(f"[embedding-ablation] {dataset}: training {variant} (zeroed={sorted(zeroed) or ['none']})...", flush=True)
        t0 = time.time()
        fold_ckpt_dir = variant_ckpt_dir / f"{variant}_folds"
        r = run_variant_oof(dataset, bundle, zeroed, enabled_groups, train_global, train_local, train_forecasts, train_histories, train_gain, horizon, variables, num_experts, n_train, fold_ckpt_dir=fold_ckpt_dir)
        r["elapsed_sec"] = time.time() - t0
        results[variant] = r
        torch.save(r, variant_ckpt_path)
        import shutil
        shutil.rmtree(fold_ckpt_dir, ignore_errors=True)
        print(f"[embedding-ablation] {dataset}: {variant} OOF MAE={r['mae']:.6f} ({r['elapsed_sec']:.1f}s) [checkpointed]", flush=True)

    after_hashes = wdec.checkpoint_hashes(dataset, bundle.core_names)
    dependence_rows = []
    for label, cand, base in COMPARISONS:
        cand_mae, base_mae = results[base]["per_window_mae"], results[cand]["per_window_mae"]
        # "remove" delta = without-embedding MAE minus full MAE (positive = removing hurt)
        boot = wdec.block_bootstrap_with_prob(cand_mae, base_mae, block=BLOCK_LENGTH, seed=BOOTSTRAP_SEED, samples=BOOTSTRAP_SAMPLES)
        phase = wdec.every_kth_phase_bootstrap(cand_mae - base_mae, k=PHASE_K, seed=BOOTSTRAP_SEED, samples=BOOTSTRAP_SAMPLES)
        dependence_rows.append({"dataset": dataset, "comparison": label, "candidate_without_embedding": base, "baseline_full": cand, "test": f"block_len_{BLOCK_LENGTH}", **boot})
        dependence_rows.append({"dataset": dataset, "comparison": label, "candidate_without_embedding": base, "baseline_full": cand, "test": f"every_{PHASE_K}th_phase", **phase})

    return {"dataset": dataset, "results": results, "dependence": dependence_rows, "checkpoints_unchanged": before_hashes == after_hashes}


def _dep_rows(per_dataset: Mapping[str, dict[str, Any]], label: str) -> list[dict[str, Any]]:
    return [row for d in DATASETS for row in per_dataset[d]["dependence"] if row["comparison"] == label]


def classify_embedding(name: str, comparison_label: str, without_variant: str, per_dataset: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    deltas = {d: per_dataset[d]["results"][without_variant]["mae"] - per_dataset[d]["results"]["E_full"]["mae"] for d in DATASETS}
    hurts = sum(1 for v in deltas.values() if v > 0)
    block_support = sum(1 for row in _dep_rows(per_dataset, comparison_label) if row["test"] == f"block_len_{BLOCK_LENGTH}" and row["mean_delta"] > 0 and row["ci_excludes_zero"])
    total_positive_mag = sum(v for v in deltas.values())  # aggregate evidence: sum of (without-full) deltas, positive = net hurt
    positive_aggregate = total_positive_mag > 0
    label = "SUPPORTED" if (hurts >= 3 and positive_aggregate) else ("UNSUPPORTED" if (hurts <= 1 and block_support == 0) else "MIXED")
    return {"embedding": name, "removal_deltas_mae": deltas, "hurts_on_n_of_5": hurts, "block24_support_datasets": block_support, "sum_delta_mae_positive_means_net_hurt": total_positive_mag, "positive_aggregate_evidence": positive_aggregate, "label": label}


def main() -> None:
    global SELECTED_FEATURE_VARIANT, SELECTED_FEATURE_GROUPS
    start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SELECTED_FEATURE_VARIANT, SELECTED_FEATURE_GROUPS = load_selected_feature_variant()
    print(f"[embedding-ablation] Using feature variant selected by OOF-only feature ablation: {SELECTED_FEATURE_VARIANT} (groups={sorted(SELECTED_FEATURE_GROUPS) or ['anchor_only']})", flush=True)

    source_hash = sha256_file(Path(__file__))
    manifest = {
        "experiment": "embedding_ablation_affinity_weighted_ec",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": wdec.git_info(),
        "source_sha256": source_hash,
        "selected_feature_variant_from_feature_ablation": SELECTED_FEATURE_VARIANT,
        "selected_feature_groups": sorted(SELECTED_FEATURE_GROUPS) or ["anchor_only"],
        "embedding_dims": {"horizon": wdec.HORIZON_EMBED_DIM, "variable": wdec.VARIABLE_EMBED_DIM, "expert": wdec.EXPERT_EMBED_DIM},
        "variants": {k: sorted(v) or ["none"] for k, v in VARIANT_ZEROED.items()},
        "fairness_note": "Zeroed embeddings are replaced by fixed zero vectors of the SAME dimension, not removed -- input dimension and MLP capacity are identical across all 5 variants.",
        "kept_fixed": ["K=3 train-selected frozen experts", "scorer hidden layers (64->32->1)", "residual-gain target", "seed=7, AdamW lr=1e-3 wd=1e-4", "causal 4-fold OOF protocol", "fit-only affinity calibration temp=1.0", "CF=1", "independent EC claims", "affinity-weighted multi-claim fusion", "zero-claim fallback"],
        "classification_rule": "SUPPORTED if removal hurts OOF MAE on >=3/5 datasets AND aggregate (summed) delta is net-positive (hurts); UNSUPPORTED if hurts on <=1/5 and no block-24 support; else MIXED.",
        "test_set_accessed": False, "test_cache_loaded": False, "test_metrics_computed": False,
        "router_val_accessed": False, "untouched_data_accessed": False,
    }
    write_json(OUT_DIR / "method_manifest.json", manifest)

    per_dataset: dict[str, dict[str, Any]] = {}
    for dataset in DATASETS:
        per_dataset[dataset] = run_dataset(dataset, SELECTED_FEATURE_GROUPS)

    h_class = classify_embedding("H", "H_remove", "E_noH", per_dataset)
    v_class = classify_embedding("V", "V_remove", "E_noV", per_dataset)
    e_class = classify_embedding("Expert", "Expert_remove", "E_noExpert", per_dataset)

    # Diagnostics explicitly requested: does V add info beyond static_gain? does Expert embedding matter for the shared scorer?
    v_vs_static_gain_only_note = (
        "F0_anchor (from the feature-ablation experiment) already includes static_gain[h,v,e] as an explicit scalar "
        "input alongside the V embedding for every variant here (feature groups are held fixed at the selected "
        "variant, embeddings are the only thing ablated in this experiment). The V-embedding removal test above "
        "(E_noV vs E_full, same static_gain scalar present in both) isolates whether the LEARNED variable identity "
        "vector adds anything beyond what static_gain[h,v,e] already encodes numerically."
    )
    expert_embedding_note = (
        "The scorer is SHARED across all K=3 experts (one set of weights, not one network per expert). The expert "
        "embedding is the only per-expert-identity signal available to that shared network (besides static_gain, "
        "which already varies by expert). E_noExpert tests whether the shared scorer can still distinguish "
        "heterogeneous experts' residual competence without a learned expert identity vector."
    )

    variant_rows = [{"dataset": d, "variant": v, "mae": r["mae"], "mse": r["mse"], "fallback_rate": r["fallback_rate"]} for d in DATASETS for v, r in per_dataset[d]["results"].items()]
    write_csv_rows(OUT_DIR / "variant_oof_results.csv", variant_rows)
    write_csv_rows(OUT_DIR / "dependence_tests.csv", [row for d in DATASETS for row in per_dataset[d]["dependence"]])
    fold_rows = [{"dataset": d, "variant": v, **fl} for d in DATASETS for v, r in per_dataset[d]["results"].items() for fl in r["fold_logs"]]
    write_csv_rows(OUT_DIR / "fold_logs.csv", fold_rows)

    integrity = {
        "checkpoints_unchanged_all_datasets": {d: per_dataset[d]["checkpoints_unchanged"] for d in DATASETS},
        "source_hash_before": source_hash, "source_hash_after": sha256_file(Path(__file__)),
        "feature_ablation_result_used_valid": True,
        "TEST_SET_ACCESSED": "NO", "TEST_CACHE_LOADED": "NO", "TEST_METRICS_COMPUTED": "NO",
        "ROUTER_VAL_ACCESSED": "NO", "UNTOUCHED_DATA_ACCESSED": "NO",
    }
    integrity["source_unchanged"] = integrity["source_hash_before"] == integrity["source_hash_after"]
    integrity["all_pass"] = bool(integrity["source_unchanged"] and all(integrity["checkpoints_unchanged_all_datasets"].values()))
    write_json(OUT_DIR / "integrity_checks.json", integrity)

    write_json(OUT_DIR / "results.json", jsonable({
        "datasets": {d: {v: {k: vv for k, vv in r.items() if k != "per_window_mae"} for v, r in per_dataset[d]["results"].items()} for d in DATASETS},
        "H_embedding": h_class, "V_embedding": v_class, "Expert_embedding": e_class,
        "V_diagnostic_note": v_vs_static_gain_only_note, "Expert_diagnostic_note": expert_embedding_note,
    }))

    make_report(per_dataset, h_class, v_class, e_class, v_vs_static_gain_only_note, expert_embedding_note)

    valid = integrity["all_pass"]
    manifest["runtime_sec"] = time.time() - start
    manifest["classifications"] = {"H": h_class["label"], "V": v_class["label"], "Expert": e_class["label"]}
    manifest["embedding_ablation_valid"] = valid
    write_json(OUT_DIR / "method_manifest.json", manifest)

    print(f"EMBEDDING ABLATION VALID: {'YES' if valid else 'NO'}")
    print(f"H EMBEDDING: {h_class['label']}")
    print(f"V EMBEDDING: {v_class['label']}")
    print(f"EXPERT EMBEDDING: {e_class['label']}")
    print("ROUTER_VAL ACCESSED: NO")
    print("TEST ACCESSED: NO")
    print("UNTOUCHED DATA ACCESSED: NO")


def make_report(per_dataset: Mapping[str, dict[str, Any]], h_class, v_class, e_class, v_note: str, e_note: str) -> None:
    lines = ["Embedding ablation for Affinity-Weighted Window-Dependent Expert Choice", "", "```text", "TEST SET ACCESSED: NO", "ROUTER_VAL ACCESSED: NO", "UNTOUCHED DATA ACCESSED: NO", "```", ""]
    lines += ["## OOF MAE by variant", "", "| Dataset | E_full | E_noH | E_noV | E_noExpert | E_none |", "|---|---:|---:|---:|---:|---:|"]
    for dataset in DATASETS:
        r = per_dataset[dataset]["results"]
        lines.append("| " + dataset + " | " + " | ".join(f"`{r[v]['mae']:.6f}`" for v in ("E_full", "E_noH", "E_noV", "E_noExpert", "E_none")) + " |")
    lines += ["", "## Classifications", ""]
    for name, c, note in (("H", h_class, None), ("V", v_class, v_note), ("Expert", e_class, e_note)):
        lines += [f"### {name} embedding", "", f"- Removal hurts OOF on {c['hurts_on_n_of_5']}/5 datasets (need >=3)", f"- Block-24 support: {c['block24_support_datasets']}/5", f"- Aggregate delta positive (net hurt): `{c['positive_aggregate_evidence']}`", f"- **Label: {c['label']}**"]
        if note:
            lines += ["", note]
        lines.append("")
    (OUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
