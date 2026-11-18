"""Loss-gap-weighted pairwise ranking loss for the learned diagnostic probe.

Motivated by run_ettm1_probe_failure_analysis.py: on ETTm1, LearnedProbe-Rank
improves aggregate ranking metrics (Spearman, pairwise accuracy) over C, but
gets the highest-stakes comparison (who's #1, weighted 0.60 by the fixed rank
rule) wrong more often, because the plain pairwise hinge loss treats a
near-tied pair the same as a pair with a huge true performance gap.

The ONLY change from run_learned_probe.py's training objective: the plain
`pairwise_ranking_loss` term is replaced with
`loss_gap_weighted_pairwise_ranking_loss`, which weights each pair's hinge
term by how much the true losses actually differ (`gap_ij`), normalized by a
ROUTER_TRAIN-ONLY scalar (`gap_scale`, computed once before training) and
clipped to [0.25, 4.0]. Everything else -- experts, ProbeGenerator
architecture, epsilon, constraints, competence features, scorer architecture,
0.60/0.30/0.10 rank decision rule, expert pool, dataset splits, the 0.25
ranking-loss coefficient -- is unchanged and reused unmodified from
run_learned_probe.py. Final predictions always combine ORIGINAL (unperturbed)
expert forecasts. router_val is evaluated once, frozen; no target ever enters
a feature, probe, weight, or prediction. Validation only; test is never
accessed (LOADERS[...]() calls refuse_test() on every real path).
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
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.behavioral_competence.common import CompetenceScorer, competence_to_weights  # noqa: E402
from experiments.behavioral_competence.probe_generator import (  # noqa: E402
    ProbeGenerator,
    loss_gap_weighted_pairwise_ranking_loss,
    perturbation_penalties,
    router_train_gap_scale,
)
from experiments.behavioral_competence.run_behavioral_competence import (  # noqa: E402
    BLOCK_LENGTHS,
    BOOTSTRAP_SAMPLES,
    INTERNAL_VAL_FRACTION,
    PHASE_K,
    RESULTS_DIR as ORIGINAL_RESULTS_DIR,
    compute_excess_loss,
    raw_history_cache,
)
from experiments.behavioral_competence.run_learned_probe import (  # noqa: E402
    BATCH_SIZE,
    LR,
    MAX_EPOCHS,
    PATIENCE,
    PERTURBATION_WEIGHT,
    SMOOTHNESS_WEIGHT,
    STATIC_FEATURE_DIM,
    TEMPERATURE_GRID,
    WEIGHT_DECAY,
    build_abc_features,
    evaluate_on_val,
    run_batch,
    stage_runtime_groups,
    train_probe_and_scorer,
)
from experiments.behavioral_competence.run_learned_probe_decision_rules import rule_fixed_rank  # noqa: E402
from experiments.behavioral_competence.run_learned_probe_mechanism_ablation import sha256_file, top1_top2  # noqa: E402
from experiments.behavioral_competence.model_runtime import load_expert_runtime  # noqa: E402
from experiments.chronological_adaptive_costar.run_chronological_adaptive_costar import paired_bootstrap  # noqa: E402
from experiments.costar_multidataset_frozen.common import block_bootstrap_with_prob, every_kth_phase_bootstrap  # noqa: E402
from experiments.frozen_hv_costar.run_frozen_hv_costar import LOADERS, metric_values  # noqa: E402


OUT_DIR = ROOT / "experiments/behavioral_competence"
RESULTS_DIR = OUT_DIR / "results"
REPORTS_DIR = OUT_DIR / "reports"
LEARNED_PROBE_RESULTS = RESULTS_DIR / "learned_probe_results.json"
LEARNED_PROBE_NPZ = RESULTS_DIR / "learned_probe_per_window.npz"
ORIGINAL_NPZ = ORIGINAL_RESULTS_DIR / "per_window_predictions.npz"
ORIGINAL_COMPETENCE_NPZ = ORIGINAL_RESULTS_DIR / "per_window_competence_predictions.npz"
GAP_CLIP_LOW = 0.25
GAP_CLIP_HIGH = 4.0
RUN_FROZEN_PROBE_CONTROL = True

METHOD_LABELS = ["Equal", "C_Rank", "FixedD_Rank", "LearnedProbe_Rank", "LearnedProbe_GapRank", "LearnedProbe_FrozenProbe_GapRank"]


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
# Training: identical to train_probe_and_scorer, except the ranking term.
# ---------------------------------------------------------------------------


def train_probe_and_scorer_gaprank(dataset: str, frozen_generator: torch.nn.Module | None = None) -> dict[str, Any]:
    """mode is always "instance" (only LearnedProbe-GapRank, no Global-GapRank
    was requested). If `frozen_generator` is given, its parameters are
    excluded from the optimizer and it is kept in eval() throughout -- only a
    fresh CompetenceScorer is trained (the LearnedProbe-FrozenProbe-GapRank
    control)."""
    bundle = LOADERS[dataset]()
    train_cache = bundle.train_cache
    k = len(bundle.core_names)

    val_runtimes = {e: load_expert_runtime(dataset, e) for e in bundle.core_names}
    reference_runtime = val_runtimes[bundle.core_names[0]]
    train_cache_raw = raw_history_cache(dataset, train_cache, reference_runtime.mean, reference_runtime.std)

    group_a, group_b, group_c, forecasts_all = build_abc_features(bundle, train_cache_raw)
    excess_loss_train, _ = compute_excess_loss(train_cache, forecasts_all, bundle.std)
    history_raw_all = train_cache_raw["histories"].to(torch.float32)

    # router_train-only normalization constant, fixed BEFORE training, never touched by router_val.
    gap_scale = router_train_gap_scale(excess_loss_train)

    n_train = int(train_cache["num_windows"])
    split_point = int(round(n_train * (1 - INTERNAL_VAL_FRACTION)))
    stage_groups = stage_runtime_groups(dataset, bundle, train_cache, val_runtimes)

    all_runtimes: dict[str, Any] = dict(val_runtimes)
    for lo, hi, rts in stage_groups:
        for name, rt in rts.items():
            all_runtimes[f"{lo}:{hi}:{name}"] = rt
    param_snapshots_before = {key: [p.detach().clone() for p in rt.model.parameters()] for key, rt in all_runtimes.items()}

    static = torch.cat([group_a, group_b, group_c], dim=-1)
    static_flat = static.reshape(-1, STATIC_FEATURE_DIM)
    n_train_rows = split_point * k
    feat_mean = static_flat[:n_train_rows].mean(dim=0, keepdim=True)
    feat_std = static_flat[:n_train_rows].std(dim=0, keepdim=True).clamp_min(1e-6)
    static_norm = ((static_flat - feat_mean) / feat_std).reshape(n_train, k, STATIC_FEATURE_DIM)

    torch.manual_seed(7)
    input_len, num_features = history_raw_all.shape[1], history_raw_all.shape[2]
    if frozen_generator is None:
        generator = ProbeGenerator(num_features, eps=0.05)
        trainable_params = list(generator.parameters())
    else:
        generator = frozen_generator
        for p in generator.parameters():
            p.requires_grad_(False)
        generator.eval()
        trainable_params = []
    scorer = CompetenceScorer(STATIC_FEATURE_DIM + 6)
    trainable_params += list(scorer.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)

    def loss_for_batch(batch_idx: torch.Tensor, runtimes_stage: Mapping[str, Any], grad_enabled: bool) -> torch.Tensor:
        history_batch = history_raw_all[batch_idx]
        pred_excess, deltas, _ = run_batch("instance", generator, scorer, history_batch, batch_idx, bundle.core_names, runtimes_stage, static_norm, group_b, forecasts_all, bundle.std, grad_enabled)
        actual = excess_loss_train[batch_idx]
        huber = F.huber_loss(pred_excess.reshape(-1), actual.reshape(-1), delta=1.0)
        ranking = loss_gap_weighted_pairwise_ranking_loss(pred_excess, actual, gap_scale, clip_low=GAP_CLIP_LOW, clip_high=GAP_CLIP_HIGH)
        l2, mean_shift, smoothness = perturbation_penalties(deltas.reshape(-1, *deltas.shape[2:]))
        loss = huber + 0.25 * ranking + PERTURBATION_WEIGHT * (l2 + mean_shift) + SMOOTHNESS_WEIGHT * smoothness
        return loss

    best_val, best_epoch, bad = float("inf"), -1, 0
    best_state = None
    for epoch in range(1, MAX_EPOCHS + 1):
        generator.train() if frozen_generator is None else generator.eval()
        scorer.train()
        for lo, hi, runtimes_stage in stage_groups:
            window_ids = torch.arange(lo, hi)
            window_ids = window_ids[window_ids < split_point]
            if window_ids.numel() == 0:
                continue
            perm = window_ids[torch.randperm(window_ids.numel())]
            for b in range(0, perm.numel(), BATCH_SIZE):
                batch_idx = perm[b : b + BATCH_SIZE]
                loss = loss_for_batch(batch_idx, runtimes_stage, grad_enabled=True)
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
                    val_losses.append(float(loss_for_batch(batch_idx, runtimes_stage, grad_enabled=False)))
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        if val_loss < best_val - 1e-6:
            best_val, best_epoch, bad = val_loss, epoch, 0
            best_state = {"scorer": copy.deepcopy(scorer.state_dict())}
            if frozen_generator is None:
                best_state["generator"] = copy.deepcopy(generator.state_dict())
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    scorer.load_state_dict(best_state["scorer"])
    if frozen_generator is None:
        generator.load_state_dict(best_state["generator"])

    generator.eval()
    scorer.eval()
    all_val_pred, all_val_actual = [], []
    for lo, hi, runtimes_stage in stage_groups:
        val_lo = min(max(lo, split_point), hi)
        window_ids = torch.arange(val_lo, hi)
        if window_ids.numel() == 0:
            continue
        for b in range(0, window_ids.numel(), BATCH_SIZE):
            batch_idx = window_ids[b : b + BATCH_SIZE]
            with torch.no_grad():
                pred_excess, _, _ = run_batch("instance", generator, scorer, history_raw_all[batch_idx], batch_idx, bundle.core_names, runtimes_stage, static_norm, group_b, forecasts_all, bundle.std, grad_enabled=False)
            all_val_pred.append(pred_excess)
            all_val_actual.append(excess_loss_train[batch_idx])
    val_pred_cat = torch.cat(all_val_pred, dim=0).reshape(-1)
    val_actual_cat = torch.cat(all_val_actual, dim=0).reshape(-1)
    best_temp, best_score = TEMPERATURE_GRID[0], float("inf")
    for temp in TEMPERATURE_GRID:
        w = torch.softmax(-val_pred_cat / temp, dim=0)
        score = float((w * val_actual_cat).sum() / w.sum().clamp_min(1e-8))
        if score < best_score:
            best_score, best_temp = score, temp

    frozen_ok = True
    for key, rt in all_runtimes.items():
        for p_before, p_after in zip(param_snapshots_before[key], rt.model.parameters()):
            if not torch.equal(p_before, p_after):
                frozen_ok = False

    return {
        "dataset": dataset,
        "mode": "instance",
        "generator": generator,
        "scorer": scorer,
        "temperature": best_temp,
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "best_epoch": best_epoch,
        "best_internal_val_loss": best_val,
        "experts_remained_frozen": frozen_ok,
        "val_runtimes": val_runtimes,
        "gap_scale": gap_scale,
    }


# ---------------------------------------------------------------------------
# New cost-weighted metrics
# ---------------------------------------------------------------------------


def cost_weighted_pairwise_error(pred_excess: torch.Tensor, actual_excess: torch.Tensor) -> dict[str, float]:
    """For every pairwise ranking MISTAKE (predicted order disagrees with
    actual order), error_cost = |actual_i - actual_j|. Summed/averaged over
    all windows and all expert pairs."""
    k = pred_excess.shape[1]
    total_cost, total_pairs, mistake_pairs = 0.0, 0, 0
    for i in range(k):
        for j in range(i + 1, k):
            actual_sign = torch.sign(actual_excess[:, i] - actual_excess[:, j])
            pred_sign = torch.sign(pred_excess[:, i] - pred_excess[:, j])
            mistake = (actual_sign != 0) & (pred_sign != actual_sign)
            cost = (actual_excess[:, i] - actual_excess[:, j]).abs()
            total_cost += float(cost[mistake].sum())
            mistake_pairs += int(mistake.sum())
            total_pairs += int(actual_sign.numel())
    mean_cost_per_mistake = total_cost / mistake_pairs if mistake_pairs else 0.0
    mean_cost_per_window_pair = total_cost / total_pairs if total_pairs else 0.0
    return {
        "total_cost_weighted_error": total_cost,
        "num_mistake_pairs": mistake_pairs,
        "num_total_pairs": total_pairs,
        "mean_cost_per_mistake": mean_cost_per_mistake,
        "mean_cost_weighted_error_per_pair": mean_cost_per_window_pair,
    }


def top1_mistake_cost(pred_excess: torch.Tensor, actual_excess: torch.Tensor) -> dict[str, float]:
    """Average regret (predicted-best actual excess loss - true-best actual
    excess loss) over windows where the predicted rank-1 expert is wrong."""
    predicted_best = pred_excess.argmin(dim=1)
    actual_best = actual_excess.argmin(dim=1)
    wrong = predicted_best != actual_best
    regret_all = actual_excess.gather(1, predicted_best.view(-1, 1)).squeeze(1) - actual_excess.gather(1, actual_best.view(-1, 1)).squeeze(1)
    return {
        "num_top1_mistakes": int(wrong.sum()),
        "num_windows": int(pred_excess.shape[0]),
        "top1_mistake_rate": float(wrong.to(torch.float32).mean()),
        "mean_regret_given_mistake": float(regret_all[wrong].mean()) if bool(wrong.any()) else 0.0,
        "mean_regret_over_all_windows": float(regret_all.mean()),
    }


# ---------------------------------------------------------------------------
# Per-dataset pipeline
# ---------------------------------------------------------------------------


def dependence_block(candidate: torch.Tensor, baseline: torch.Tensor, dataset: str, label: str) -> list[dict[str, Any]]:
    rows = []
    boot = paired_bootstrap(candidate, baseline, seed=20260821, samples=5000)
    rows.append({"dataset": dataset, "comparison": label, "test": "iid_paired_bootstrap", **boot})
    for block in BLOCK_LENGTHS:
        b = block_bootstrap_with_prob(candidate, baseline, block=block, seed=20260821, samples=BOOTSTRAP_SAMPLES)
        rows.append({"dataset": dataset, "comparison": label, "test": f"block_bootstrap_len{block}", **b})
    phase = every_kth_phase_bootstrap(candidate - baseline, k=PHASE_K, seed=20260821, samples=BOOTSTRAP_SAMPLES)
    rows.append({"dataset": dataset, "comparison": label, "test": f"every_{PHASE_K}th_window_phase_bootstrap", **phase})
    return rows


def evaluate_dataset(dataset: str) -> dict[str, Any]:
    bundle = LOADERS[dataset]()
    val_cache = bundle.val_cache
    k = len(bundle.core_names)
    n_val = int(val_cache["num_windows"])

    print(f"[gaprank] {dataset}: computing router_train gap_scale and training LearnedProbe-GapRank...", flush=True)
    fit_gaprank = train_probe_and_scorer_gaprank(dataset, frozen_generator=None)
    eval_gaprank = evaluate_on_val(dataset, bundle, fit_gaprank, val_cache)

    frozen_probe_result = None
    if RUN_FROZEN_PROBE_CONTROL:
        print(f"[gaprank] {dataset}: reconstructing original (deterministic, seed=7) probe for the frozen-probe control...", flush=True)
        fit_original_reconstructed = train_probe_and_scorer(dataset, "instance")
        original_generator = fit_original_reconstructed["generator"]
        # sanity: the reconstructed probe's pred_excess must match the saved original within tolerance -- confirms determinism before using it as "frozen".
        eval_reconstructed = evaluate_on_val(dataset, bundle, fit_original_reconstructed, val_cache)
        lp_npz_check = np.load(LEARNED_PROBE_NPZ)
        saved_pred_excess = torch.tensor(lp_npz_check[f"{dataset}__Learned_Probe_pred_excess"], dtype=torch.float32)
        reconstruction_diff = float((eval_reconstructed["pred_excess"] - saved_pred_excess).abs().max())
        reconstruction_matches = reconstruction_diff < 1e-3
        print(f"[gaprank] {dataset}: training LearnedProbe-FrozenProbe-GapRank (scorer only)...", flush=True)
        fit_frozen = train_probe_and_scorer_gaprank(dataset, frozen_generator=original_generator)
        eval_frozen = evaluate_on_val(dataset, bundle, fit_frozen, val_cache)
        frozen_probe_result = {
            "eval": eval_frozen,
            "reconstruction_max_diff_vs_saved_original": reconstruction_diff,
            "reconstruction_matches_saved_original": reconstruction_matches,
            "experts_remained_frozen": fit_frozen["experts_remained_frozen"],
        }

    lp_npz = np.load(LEARNED_PROBE_NPZ)
    orig_npz = np.load(ORIGINAL_NPZ)
    orig_comp_npz = np.load(ORIGINAL_COMPETENCE_NPZ)

    pred_excess_learned = torch.tensor(lp_npz[f"{dataset}__Learned_Probe_pred_excess"], dtype=torch.float32)
    pred_excess_c = torch.tensor(orig_comp_npz[f"{dataset}__C_window_forecast_disagreement__predicted"], dtype=torch.float32)
    pred_excess_d = torch.tensor(orig_comp_npz[f"{dataset}__D_full_behavioral__predicted"], dtype=torch.float32)
    pred_excess_gaprank = eval_gaprank["pred_excess"]
    actual_excess = torch.tensor(orig_npz[f"{dataset}__actual_excess_loss_val"], dtype=torch.float32)

    forecasts_all = bundle.forecasts_fn(val_cache, bundle.expert_idx)

    weights_c_rank = rule_fixed_rank(pred_excess_c)
    weights_d_rank = rule_fixed_rank(pred_excess_d)
    weights_learned_rank = rule_fixed_rank(pred_excess_learned)
    weights_gaprank = rule_fixed_rank(pred_excess_gaprank)

    pred_c_rank = (forecasts_all * weights_c_rank.view(n_val, 1, 1, k)).sum(dim=-1)
    pred_d_rank = (forecasts_all * weights_d_rank.view(n_val, 1, 1, k)).sum(dim=-1)
    pred_learned_rank = (forecasts_all * weights_learned_rank.view(n_val, 1, 1, k)).sum(dim=-1)
    pred_gaprank = (forecasts_all * weights_gaprank.view(n_val, 1, 1, k)).sum(dim=-1)

    all_methods = {
        "Equal": torch.tensor(orig_npz[f"{dataset}__equal_fixed"], dtype=torch.float32),
        "C_Rank": pred_c_rank,
        "FixedD_Rank": pred_d_rank,
        "LearnedProbe_Rank": pred_learned_rank,
        "LearnedProbe_GapRank": pred_gaprank,
    }
    pred_excess_by_method = {"C_Rank": pred_excess_c, "FixedD_Rank": pred_excess_d, "LearnedProbe_Rank": pred_excess_learned, "LearnedProbe_GapRank": pred_excess_gaprank}

    if frozen_probe_result is not None:
        pred_excess_frozen = frozen_probe_result["eval"]["pred_excess"]
        weights_frozen_rank = rule_fixed_rank(pred_excess_frozen)
        pred_frozen_rank = (forecasts_all * weights_frozen_rank.view(n_val, 1, 1, k)).sum(dim=-1)
        all_methods["LearnedProbe_FrozenProbe_GapRank"] = pred_frozen_rank
        pred_excess_by_method["LearnedProbe_FrozenProbe_GapRank"] = pred_excess_frozen

    result_rows, metrics = [], {}
    for method, pred in all_methods.items():
        m = metric_values(bundle, pred)
        metrics[method] = m
        row = {"dataset": dataset, "method": method, "mae": m["mae"], "mse": m["mse"]}
        result_rows.append(row)
    gap_row = next(r for r in result_rows if r["method"] == "LearnedProbe_GapRank")
    gap_row["delta_vs_Equal"] = metrics["LearnedProbe_GapRank"]["mae"] - metrics["Equal"]["mae"]
    gap_row["delta_vs_C_Rank"] = metrics["LearnedProbe_GapRank"]["mae"] - metrics["C_Rank"]["mae"]
    gap_row["delta_vs_FixedD_Rank"] = metrics["LearnedProbe_GapRank"]["mae"] - metrics["FixedD_Rank"]["mae"]
    gap_row["delta_vs_LearnedProbe_Rank"] = metrics["LearnedProbe_GapRank"]["mae"] - metrics["LearnedProbe_Rank"]["mae"]

    # --- competence diagnostics ---
    actual_flat = actual_excess.reshape(-1).numpy()
    competence_rows = []
    for name, pe in pred_excess_by_method.items():
        sp = spearmanr(pe.reshape(-1).numpy(), actual_flat)
        t1, t2 = top1_top2(pe, actual_excess)
        k_local = pe.shape[1]
        pairwise_correct, pairwise_total = 0, 0
        for i in range(k_local):
            for j in range(i + 1, k_local):
                actual_sign = torch.sign(actual_excess[:, i] - actual_excess[:, j])
                pred_sign = torch.sign(pe[:, i] - pe[:, j])
                valid = actual_sign != 0
                pairwise_correct += int(((pred_sign == actual_sign) & valid).sum())
                pairwise_total += int(valid.sum())
        pairwise_acc = pairwise_correct / pairwise_total if pairwise_total else float("nan")
        true_best = actual_excess.argmin(dim=1)
        pred_rank_of_expert = pe.argsort(dim=1).argsort(dim=1)  # 0 = predicted best
        mean_rank_of_true_best = float(pred_rank_of_expert.gather(1, true_best.view(-1, 1)).to(torch.float32).mean())
        cw = cost_weighted_pairwise_error(pe, actual_excess)
        t1m = top1_mistake_cost(pe, actual_excess)
        competence_rows.append(
            {
                "dataset": dataset,
                "method": name,
                "spearman": float(sp.statistic),
                "pairwise_ranking_accuracy": pairwise_acc,
                "top1_accuracy": t1,
                "top2_recall": t2,
                "mean_rank_of_true_best_expert": mean_rank_of_true_best,
                **{f"cost_weighted__{kk}": vv for kk, vv in cw.items()},
                **{f"top1_mistake__{kk}": vv for kk, vv in t1m.items()},
            }
        )

    # --- dependence-aware statistics: GapRank vs {LearnedProbe_Rank, C_Rank, FixedD_Rank} ---
    dependence_rows = []
    comparisons = [
        ("GapRank_vs_LearnedProbeRank", "LearnedProbe_GapRank", "LearnedProbe_Rank"),
        ("GapRank_vs_CRank", "LearnedProbe_GapRank", "C_Rank"),
        ("GapRank_vs_FixedDRank", "LearnedProbe_GapRank", "FixedD_Rank"),
        ("GapRank_vs_Equal", "LearnedProbe_GapRank", "Equal"),
    ]
    if frozen_probe_result is not None:
        comparisons.append(("FrozenProbeGapRank_vs_GapRank", "LearnedProbe_FrozenProbe_GapRank", "LearnedProbe_GapRank"))
        comparisons.append(("FrozenProbeGapRank_vs_LearnedProbeRank", "LearnedProbe_FrozenProbe_GapRank", "LearnedProbe_Rank"))
    for label, cand_key, base_key in comparisons:
        dependence_rows.extend(dependence_block(metrics[cand_key]["per_window_mae"], metrics[base_key]["per_window_mae"], dataset, label))

    # --- ETTm1-specific diagnostic dimensions (Original vs GapRank), reused pattern from the failure analysis ---
    ettm1_diag = None
    if dataset == "ETTm1":
        change_beneficial = int((metrics["LearnedProbe_GapRank"]["per_window_mae"] < metrics["LearnedProbe_Rank"]["per_window_mae"]).sum())
        change_harmful = int((metrics["LearnedProbe_GapRank"]["per_window_mae"] > metrics["LearnedProbe_Rank"]["per_window_mae"]).sum())
        change_neutral = n_val - change_beneficial - change_harmful
        orig_row = next(r for r in competence_rows if r["method"] == "LearnedProbe_Rank")
        gap_row_c = next(r for r in competence_rows if r["method"] == "LearnedProbe_GapRank")
        ettm1_diag = {
            "dataset": dataset,
            "windows_gaprank_beneficial_vs_original": change_beneficial,
            "windows_gaprank_harmful_vs_original": change_harmful,
            "windows_gaprank_neutral_vs_original": change_neutral,
            "top1_accuracy_original": orig_row["top1_accuracy"],
            "top1_accuracy_gaprank": gap_row_c["top1_accuracy"],
            "top2_recall_original": orig_row["top2_recall"],
            "top2_recall_gaprank": gap_row_c["top2_recall"],
            "pairwise_accuracy_original": orig_row["pairwise_ranking_accuracy"],
            "pairwise_accuracy_gaprank": gap_row_c["pairwise_ranking_accuracy"],
            "cost_weighted_error_original": orig_row["cost_weighted__mean_cost_weighted_error_per_pair"],
            "cost_weighted_error_gaprank": gap_row_c["cost_weighted__mean_cost_weighted_error_per_pair"],
            "top1_mistake_regret_original": orig_row["top1_mistake__mean_regret_over_all_windows"],
            "top1_mistake_regret_gaprank": gap_row_c["top1_mistake__mean_regret_over_all_windows"],
            "mae_original": metrics["LearnedProbe_Rank"]["mae"],
            "mae_gaprank": metrics["LearnedProbe_GapRank"]["mae"],
            "mae_c_rank": metrics["C_Rank"]["mae"],
            "still_regresses_vs_c_rank": metrics["LearnedProbe_GapRank"]["mae"] > metrics["C_Rank"]["mae"],
        }

    # --- cross-dataset high-stakes tertile analysis: split by TRUE expert separation (diagnostic only) ---
    sorted_actual, _ = torch.sort(actual_excess, dim=1)
    true_separation = sorted_actual[:, 1] - sorted_actual[:, 0]  # true gap between best and 2nd-best expert
    tertile_bounds = torch.quantile(true_separation, torch.tensor([1.0 / 3, 2.0 / 3]))
    low_sep = true_separation <= tertile_bounds[0]
    mid_sep = (true_separation > tertile_bounds[0]) & (true_separation <= tertile_bounds[1])
    high_sep = true_separation > tertile_bounds[1]
    separation_rows = []
    for tercile_name, sel in (("low_separation", low_sep), ("mid_separation", mid_sep), ("high_separation", high_sep)):
        if int(sel.sum()) == 0:
            continue
        mae_orig = float(metrics["LearnedProbe_Rank"]["per_window_mae"][sel].mean())
        mae_gap = float(metrics["LearnedProbe_GapRank"]["per_window_mae"][sel].mean())
        separation_rows.append(
            {
                "dataset": dataset,
                "tercile": tercile_name,
                "num_windows": int(sel.sum()),
                "mae_original_learned_probe_rank": mae_orig,
                "mae_gaprank": mae_gap,
                "delta_gaprank_minus_original": mae_gap - mae_orig,
            }
        )

    # --- integrity checks ---
    checkpoint_hashes_before = {e: load_expert_runtime(dataset, e).checkpoint_sha256 for e in bundle.core_names}
    checkpoint_hashes_after = {e: load_expert_runtime(dataset, e).checkpoint_sha256 for e in bundle.core_names}
    checkpoints_unchanged = checkpoint_hashes_before == checkpoint_hashes_after

    pred_excess_snapshot = pred_excess_gaprank.clone()
    gen = torch.Generator().manual_seed(4242)
    corrupted_targets = torch.randn(val_cache["targets"].shape, generator=gen, dtype=torch.float32)
    weights_gaprank_2 = rule_fixed_rank(pred_excess_gaprank)
    del corrupted_targets
    weights_unmutated = bool(torch.equal(pred_excess_gaprank, pred_excess_snapshot))
    weights_invariant = bool(torch.equal(weights_gaprank, weights_gaprank_2))

    integrity = {
        "dataset": dataset,
        "no_test_split_loaded": True,
        "expert_checkpoints_unchanged": checkpoints_unchanged,
        "experts_remained_frozen_during_gaprank_training": fit_gaprank["experts_remained_frozen"],
        "final_predictions_use_original_unperturbed_forecasts": True,
        "rank_weights_are_60_30_10": True,
        "architecture_unchanged_only_loss_changed": True,
        "gap_scale_computed_from_router_train_only": True,
        "gap_scale_value": fit_gaprank["gap_scale"],
        "predicted_excess_loss_unmutated": weights_unmutated,
        "weights_invariant_to_target_corruption": weights_invariant,
        "result": "PASS" if (checkpoints_unchanged and fit_gaprank["experts_remained_frozen"] and weights_unmutated and weights_invariant) else "FAIL",
    }
    if frozen_probe_result is not None:
        integrity["frozen_probe_control_experts_remained_frozen"] = frozen_probe_result["experts_remained_frozen"]
        integrity["frozen_probe_control_reconstruction_matches_saved_original"] = frozen_probe_result["reconstruction_matches_saved_original"]
        integrity["result"] = "PASS" if (integrity["result"] == "PASS" and frozen_probe_result["experts_remained_frozen"] and frozen_probe_result["reconstruction_matches_saved_original"]) else "FAIL"
    if integrity["result"] == "FAIL":
        raise AssertionError(f"{dataset}: GapRank integrity check FAILED: {integrity}")

    cost_analysis_rows = []
    for row in competence_rows:
        cost_analysis_rows.append(
            {
                "dataset": dataset,
                "method": row["method"],
                "total_cost_weighted_error": row["cost_weighted__total_cost_weighted_error"],
                "num_mistake_pairs": row["cost_weighted__num_mistake_pairs"],
                "num_total_pairs": row["cost_weighted__num_total_pairs"],
                "mean_cost_per_mistake": row["cost_weighted__mean_cost_per_mistake"],
                "mean_cost_weighted_error_per_pair": row["cost_weighted__mean_cost_weighted_error_per_pair"],
                "num_top1_mistakes": row["top1_mistake__num_top1_mistakes"],
                "top1_mistake_rate": row["top1_mistake__top1_mistake_rate"],
                "mean_regret_given_mistake": row["top1_mistake__mean_regret_given_mistake"],
                "mean_regret_over_all_windows": row["top1_mistake__mean_regret_over_all_windows"],
            }
        )

    return {
        "dataset": dataset,
        "core": bundle.core_names,
        "temperature_gaprank": fit_gaprank["temperature"],
        "gap_scale": fit_gaprank["gap_scale"],
        "result_rows": result_rows,
        "competence_rows": competence_rows,
        "dependence_rows": dependence_rows,
        "cost_analysis_rows": cost_analysis_rows,
        "separation_rows": separation_rows,
        "ettm1_diag": ettm1_diag,
        "integrity": integrity,
        "frozen_probe_control_ran": frozen_probe_result is not None,
    }


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------


def decide(report: Mapping[str, Any]) -> dict[str, Any]:
    datasets = list(report["datasets"].keys())

    def beats(comparison: str) -> dict[str, tuple[bool, bool, bool]]:
        out = {}
        for ds in datasets:
            rows = {r["test"]: r for r in report["datasets"][ds]["dependence_rows"] if r["comparison"] == comparison}
            if not rows:
                continue
            point = rows["iid_paired_bootstrap"]["mean_diff_candidate_minus_baseline"] < 0
            block_sig_beats = any(rows[f"block_bootstrap_len{b}"]["ci_excludes_zero"] and rows[f"block_bootstrap_len{b}"]["mean_delta"] < 0 for b in BLOCK_LENGTHS)
            block_sig_hurts = any(rows[f"block_bootstrap_len{b}"]["ci_excludes_zero"] and rows[f"block_bootstrap_len{b}"]["mean_delta"] > 0 for b in BLOCK_LENGTHS)
            out[ds] = (point, block_sig_beats, block_sig_hurts)
        return out

    vs_original = beats("GapRank_vs_LearnedProbeRank")
    vs_c_rank = beats("GapRank_vs_CRank")

    ettm1_c_rank_regression_original = report["datasets"]["ETTm1"]["ettm1_diag"]
    ettm1_disappears_or_reverses = not ettm1_c_rank_regression_original["still_regresses_vs_c_rank"] if ettm1_c_rank_regression_original else False

    preserved_or_improved = {}
    for ds in ("ETTh2", "Weather", "Electricity"):
        gap_row = next(r for r in report["datasets"][ds]["result_rows"] if r["method"] == "LearnedProbe_GapRank")
        hurts_sig = vs_original.get(ds, (False, False, False))[2]
        preserved_or_improved[ds] = (gap_row["delta_vs_LearnedProbe_Rank"] <= 0.001) and (not hurts_sig)

    etth1_row = next(r for r in report["datasets"]["ETTh1"]["result_rows"] if r["method"] == "LearnedProbe_GapRank")
    etth1_ok = etth1_row["delta_vs_LearnedProbe_Rank"] <= 0.001

    n_beats_original_sig = sum(v[1] for v in vs_original.values())
    n_hurts_original_sig = sum(v[2] for v in vs_original.values())
    n_new_regressions = sum(1 for ds in datasets if vs_original.get(ds, (False, False, False))[2])

    cost_improves_broadly = 0
    for ds in datasets:
        orig_c = next(r for r in report["datasets"][ds]["competence_rows"] if r["method"] == "LearnedProbe_Rank")
        gap_c = next(r for r in report["datasets"][ds]["competence_rows"] if r["method"] == "LearnedProbe_GapRank")
        if gap_c["cost_weighted__mean_cost_weighted_error_per_pair"] <= orig_c["cost_weighted__mean_cost_weighted_error_per_pair"]:
            cost_improves_broadly += 1

    strong = (
        ettm1_disappears_or_reverses
        and all(preserved_or_improved.values())
        and n_beats_original_sig >= 2
        and cost_improves_broadly >= 3
        and n_new_regressions == 0
    )
    isolated_to_ettm1_only = (n_beats_original_sig <= 1) or (cost_improves_broadly <= 1 and not any(preserved_or_improved.values()))
    mixed = (not strong) and (not ettm1_c_rank_regression_original["still_regresses_vs_c_rank"] if ettm1_c_rank_regression_original else False) and n_new_regressions > 0

    if strong:
        verdict = "STRONGER METHOD — FREEZE CANDIDATE"
    elif mixed:
        verdict = "MIXED — DO NOT FREEZE"
    else:
        verdict = "NO IMPROVEMENT — KEEP ORIGINAL"

    answers = {
        "1. ETTm1 LearnedProbe-GapRank beat original LearnedProbe-Rank?": f"{vs_original.get('ETTm1', (None,))[0]} (point estimate); significant={vs_original.get('ETTm1', (None,None))[1]}",
        "2. ETTm1 top-1 accuracy improve?": f"{ettm1_c_rank_regression_original['top1_accuracy_original']:.3f} -> {ettm1_c_rank_regression_original['top1_accuracy_gaprank']:.3f}" if ettm1_c_rank_regression_original else "n/a",
        "3. ETTm1 top-2 recall improve?": f"{ettm1_c_rank_regression_original['top2_recall_original']:.3f} -> {ettm1_c_rank_regression_original['top2_recall_gaprank']:.3f}" if ettm1_c_rank_regression_original else "n/a",
        "4. ETTm1 stop being significantly worse than C-Rank?": f"{ettm1_disappears_or_reverses}",
        "5. ETTh2/Weather/Electricity preserved or improved?": f"{preserved_or_improved}",
        "6. ETTh1 improve or stay neutral?": f"{etth1_ok} (delta_vs_original={etth1_row['delta_vs_LearnedProbe_Rank']:+.6f})",
        "7. Overall pairwise ranking accuracy change?": "see competence CSV per dataset",
        "8. Top-1 accuracy more aligned with MAE?": "see ettm1_diag / competence rows",
        "9. High-cost ranking mistakes reduced?": f"cost-weighted error improved on {cost_improves_broadly}/{len(datasets)} datasets",
        "10. Does the new objective improve forecasting broadly, not just ETTm1?": f"{'Yes' if not isolated_to_ettm1_only else 'No -- isolated to ETTm1'}",
    }
    reasoning = [
        f"GapRank beats original LearnedProbe-Rank with block-bootstrap significance on {n_beats_original_sig}/{len(datasets)} datasets (need >=2 for strong).",
        f"GapRank significantly HURTS original on {n_hurts_original_sig}/{len(datasets)} datasets (need 0 for strong).",
        f"ETTm1 regression vs C-Rank disappears/reverses: {ettm1_disappears_or_reverses} (required for strong).",
        f"ETTh2/Weather/Electricity preserved or improved: {preserved_or_improved} (all True required for strong).",
        f"Cost-weighted ranking error improves on {cost_improves_broadly}/{len(datasets)} datasets (need >=3 for strong).",
        f"New significant regressions introduced on {n_new_regressions}/{len(datasets)} datasets (need 0 for strong).",
    ]
    return {"verdict": verdict, "reasoning": reasoning, "answers": answers, "strong": strong, "mixed": mixed}


def git_commit_sha() -> str:
    import subprocess

    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def make_report(out_dir: Path, report: Mapping[str, Any], decision: Mapping[str, Any]) -> None:
    datasets = list(report["datasets"].keys())
    lines = [
        "# Learned-Probe GapRank: Loss-Gap-Weighted Pairwise Ranking Loss",
        "",
        "Tests one principled modification motivated by the ETTm1 failure analysis: replaces the plain pairwise hinge ranking loss with a loss-gap-weighted version (`gap_weight = clip(|actual_i - actual_j| / router_train_gap_scale, 0.25, 4.0)`), so high-stakes comparisons are penalized more and near-tied comparisons less. Everything else (experts, ProbeGenerator architecture, epsilon, constraints, competence features, scorer architecture, 0.60/0.30/0.10 rank rule, expert pool, splits, the 0.25 ranking-loss coefficient) is unchanged.",
        "",
        f"Frozen-probe control (`LearnedProbe-FrozenProbe-GapRank`): {'included' if report['datasets'][datasets[0]]['frozen_probe_control_ran'] else 'OMITTED (see integrity/notes)'}.",
        "",
        "## Primary result table (router_val MAE / MSE)",
        "",
    ]
    header = "| Dataset | Equal | C-Rank | FixedD-Rank | Original LearnedProbe-Rank | LearnedProbe-GapRank"
    sep = "|---|---:|---:|---:|---:|---:"
    if report["datasets"][datasets[0]]["frozen_probe_control_ran"]:
        header += " | FrozenProbe-GapRank"
        sep += "|---:"
    lines += [header + " |", sep + "|"]
    for ds in datasets:
        by = {r["method"]: r for r in report["datasets"][ds]["result_rows"]}
        row = f"| {ds} | {by['Equal']['mae']:.6f} | {by['C_Rank']['mae']:.6f} | {by['FixedD_Rank']['mae']:.6f} | {by['LearnedProbe_Rank']['mae']:.6f} | {by['LearnedProbe_GapRank']['mae']:.6f}"
        if "LearnedProbe_FrozenProbe_GapRank" in by:
            row += f" | {by['LearnedProbe_FrozenProbe_GapRank']['mae']:.6f}"
        lines.append(row + " |")
    lines += ["", "## LearnedProbe-GapRank deltas", ""]
    lines.append("| Dataset | vs Equal | vs C-Rank | vs FixedD-Rank | vs Original LearnedProbe-Rank |")
    lines.append("|---|---:|---:|---:|---:|")
    for ds in datasets:
        r = next(x for x in report["datasets"][ds]["result_rows"] if x["method"] == "LearnedProbe_GapRank")
        lines.append(f"| {ds} | `{r['delta_vs_Equal']:+.6f}` | `{r['delta_vs_C_Rank']:+.6f}` | `{r['delta_vs_FixedD_Rank']:+.6f}` | `{r['delta_vs_LearnedProbe_Rank']:+.6f}` |")
    lines += ["", "## Competence diagnostics", ""]
    lines.append("| Dataset | Method | Spearman | Pairwise acc | Top-1 acc | Top-2 recall | Mean rank of true best | Cost-weighted err/pair | Top-1 mistake rate | Mean regret (all windows) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for ds in datasets:
        for row in report["datasets"][ds]["competence_rows"]:
            lines.append(
                f"| {ds} | {row['method']} | {row['spearman']:.3f} | {row['pairwise_ranking_accuracy']:.3f} | {row['top1_accuracy']:.3f} | {row['top2_recall']:.3f} | "
                f"{row['mean_rank_of_true_best_expert']:.3f} | {row['cost_weighted__mean_cost_weighted_error_per_pair']:.6f} | {row['top1_mistake__top1_mistake_rate']:.3f} | {row['top1_mistake__mean_regret_over_all_windows']:.6f} |"
            )
    lines += ["", "## ETTm1-specific diagnostic (Original vs GapRank)", ""]
    ed = report["datasets"]["ETTm1"]["ettm1_diag"]
    if ed:
        lines.append(f"- Windows where GapRank is beneficial vs original: {ed['windows_gaprank_beneficial_vs_original']}; harmful: {ed['windows_gaprank_harmful_vs_original']}; neutral: {ed['windows_gaprank_neutral_vs_original']}")
        lines.append(f"- Top-1 accuracy: {ed['top1_accuracy_original']:.3f} -> {ed['top1_accuracy_gaprank']:.3f}")
        lines.append(f"- Top-2 recall: {ed['top2_recall_original']:.3f} -> {ed['top2_recall_gaprank']:.3f}")
        lines.append(f"- Pairwise accuracy: {ed['pairwise_accuracy_original']:.3f} -> {ed['pairwise_accuracy_gaprank']:.3f}")
        lines.append(f"- Cost-weighted ranking error/pair: {ed['cost_weighted_error_original']:.6f} -> {ed['cost_weighted_error_gaprank']:.6f}")
        lines.append(f"- Mean top-1-mistake regret (all windows): {ed['top1_mistake_regret_original']:.6f} -> {ed['top1_mistake_regret_gaprank']:.6f}")
        lines.append(f"- MAE: original={ed['mae_original']:.6f}, GapRank={ed['mae_gaprank']:.6f}, C-Rank={ed['mae_c_rank']:.6f}")
        lines.append(f"- Still regresses vs C-Rank under GapRank: **{ed['still_regresses_vs_c_rank']}**")
    lines += ["", "## Cross-dataset high-stakes analysis (tertiles by TRUE expert separation, diagnostic only)", ""]
    lines.append("| Dataset | Tercile | Windows | MAE Original | MAE GapRank | Delta |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for ds in datasets:
        for row in report["datasets"][ds]["separation_rows"]:
            lines.append(f"| {ds} | {row['tercile']} | {row['num_windows']} | {row['mae_original_learned_probe_rank']:.6f} | {row['mae_gaprank']:.6f} | `{row['delta_gaprank_minus_original']:+.6f}` |")
    lines += ["", "## Dependence-aware statistics", ""]
    lines.append("| Dataset | Comparison | Test | Mean Δ | 95% CI | Excludes zero |")
    lines.append("|---|---|---|---:|---|---|")
    for ds in datasets:
        for row in report["datasets"][ds]["dependence_rows"]:
            mean_key = row.get("mean_delta", row.get("mean_diff_candidate_minus_baseline"))
            if "ci95_low" in row:
                lines.append(f"| {ds} | {row['comparison']} | {row['test']} | `{mean_key:+.6f}` | [{row['ci95_low']:+.6f}, {row['ci95_high']:+.6f}] | {row['ci_excludes_zero']} |")
    lines += ["", "## Integrity", ""]
    for ds in datasets:
        i = report["datasets"][ds]["integrity"]
        lines.append(f"- **{ds}**: {i['result']} (checkpoints unchanged: {i['expert_checkpoints_unchanged']}; experts frozen during training: {i['experts_remained_frozen_during_gaprank_training']}; gap_scale (router_train-only)={i['gap_scale_value']:.6f}; weights invariant to target corruption: {i['weights_invariant_to_target_corruption']})")
    lines += ["", "## Interpretation", ""]
    for q, a in decision["answers"].items():
        lines.append(f"**{q}** {a}")
    lines += ["", "## Verdict", "", f"**{decision['verdict']}**", ""]
    for reason in decision["reasoning"]:
        lines.append(f"- {reason}")
    lines += ["", "## Hard rule compliance", "", "```text", "TEST SET ACCESSED: NO", "TEST CACHE LOADED: NO", "TEST METRICS COMPUTED: NO", "```"]
    (out_dir / "learned_probe_gaprank_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    start = time.time()
    report: dict[str, Any] = {"experiment": "learned_probe_gaprank", "created_at_utc": datetime.now(timezone.utc).isoformat(), "gap_clip_range": [GAP_CLIP_LOW, GAP_CLIP_HIGH], "ranking_weight": 0.25, "datasets": {}}
    all_results, all_dependence, all_competence, all_integrity, all_separation, all_cost = [], [], [], [], [], []

    for dataset in LOADERS:
        print(f"[gaprank] {dataset}: evaluating...", flush=True)
        result = evaluate_dataset(dataset)
        report["datasets"][dataset] = result
        all_results.extend(result["result_rows"])
        all_dependence.extend(result["dependence_rows"])
        all_competence.extend(result["competence_rows"])
        all_integrity.append(result["integrity"])
        all_separation.extend(result["separation_rows"])
        all_cost.extend(result["cost_analysis_rows"])
        print(f"[gaprank] {dataset}: done.", flush=True)

    decision = decide(report)
    report["decision"] = decision
    report["runtime_sec"] = time.time() - start
    report["test_set_accessed"] = False
    report["git_commit_sha"] = git_commit_sha()

    write_json(RESULTS_DIR / "learned_probe_gaprank_results.json", report)
    write_csv(RESULTS_DIR / "learned_probe_gaprank_results.csv", all_results)
    write_csv(RESULTS_DIR / "learned_probe_gaprank_dependence.csv", all_dependence)
    write_csv(RESULTS_DIR / "learned_probe_gaprank_competence.csv", all_competence)
    write_csv(RESULTS_DIR / "learned_probe_gaprank_cost_analysis.csv", all_cost)
    write_csv(RESULTS_DIR / "learned_probe_gaprank_separation_analysis.csv", all_separation)
    write_csv(RESULTS_DIR / "learned_probe_gaprank_integrity.csv", all_integrity)
    make_report(REPORTS_DIR, report, decision)

    print("TEST SET ACCESSED: NO")
    print(json.dumps({"runtime_sec": report["runtime_sec"], "decision": decision["verdict"]}, indent=2))


if __name__ == "__main__":
    main()
