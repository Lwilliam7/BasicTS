"""Learned diagnostic probes for expert competence estimation.

Hypothesis under test: an instance-conditioned learned diagnostic probe
exposes expert failure better than fixed, human-designed perturbations
(P1-P4 from run_behavioral_competence.py).

Trains a small, shared ProbeGenerator jointly with the competence scorer on
router_train, with every frozen forecasting expert's parameters excluded
from the optimizer and verified bit-identical before/after training.
Gradients flow through the frozen expert's forward computation (standard
"differentiable fixed function") to reach the perturbation `delta` -- the
expert's own weights never move. router_val is evaluated once, frozen, with
no target ever entering any feature, probe, weight, or prediction.

Compares, using the SAME final softmax-temperature weighting rule everywhere:
  C              -- window + forecast + disagreement (reloaded from the
                    original behavioral_competence experiment, unmodified)
  Fixed-D        -- C + the four manual perturbations (reloaded, unmodified)
  Learned-Probe  -- C + one instance-conditioned learned probe (new, trained here)
  Learned-Global-Probe -- C + one fixed learned perturbation pattern (new, trained here)
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
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.behavioral_competence.common import (  # noqa: E402
    CompetenceScorer,
    ScorerFit,
    competence_to_weights,
    disagreement_features_group_c,
    forecast_features_group_b,
    window_features_group_a,
)
from experiments.behavioral_competence.probe_generator import (  # noqa: E402
    GlobalProbeGenerator,
    ProbeGenerator,
    pairwise_ranking_loss,
    perturbation_penalties,
    probe_response_features,
)
from experiments.behavioral_competence.run_behavioral_competence import (  # noqa: E402
    BLOCK_LENGTHS,
    BOOTSTRAP_SAMPLES,
    INTERNAL_VAL_FRACTION,
    PHASE_K,
    RESULTS_DIR as ORIGINAL_RESULTS_DIR,
    compute_excess_loss,
    raw_history_cache,
    router_train_block_split,
)
from experiments.behavioral_competence.model_runtime import load_expert_runtime  # noqa: E402
from experiments.chronological_adaptive_costar.run_chronological_adaptive_costar import paired_bootstrap  # noqa: E402
from experiments.costar_multidataset_frozen.common import block_bootstrap_with_prob, every_kth_phase_bootstrap  # noqa: E402
from experiments.frozen_hv_costar.run_frozen_hv_costar import LOADERS, metric_values, refuse_test  # noqa: E402
from experiments.oracle_weight_tournament.run_tournament import sample_mae  # noqa: E402


OUT_DIR = ROOT / "experiments/behavioral_competence"
RESULTS_DIR = OUT_DIR / "results"
REPORTS_DIR = OUT_DIR / "reports"
EPS = 0.05
MAX_EPOCHS = 8
PATIENCE = 3
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4
RANKING_WEIGHT = 0.25
PERTURBATION_WEIGHT = 0.01
SMOOTHNESS_WEIGHT = 0.01
TEMPERATURE_GRID = (0.02, 0.05, 0.1, 0.2, 0.5)
STATIC_FEATURE_DIM = 6 + 4 + 5  # group A + B + C
PROBE_FEATURE_DIM = 6


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


def build_abc_features(bundle, cache: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    history = cache["histories"].to(torch.float32)
    group_a_window = window_features_group_a(history, bundle.std)
    forecasts_all = bundle.forecasts_fn(cache, bundle.expert_idx)
    n, h, f, k = forecasts_all.shape
    group_a = group_a_window.unsqueeze(1).expand(n, k, 6).clone()
    group_b = torch.zeros(n, k, 4)
    group_c = torch.zeros(n, k, 5)
    last_observed = history[:, -1, :]
    for local_i in range(k):
        forecast_e = forecasts_all[..., local_i]
        group_b[:, local_i, :] = forecast_features_group_b(forecast_e, last_observed, bundle.std)
        group_c[:, local_i, :] = disagreement_features_group_c(forecast_e, forecasts_all, bundle.std)
    return group_a, group_b, group_c, forecasts_all


def stage_runtime_groups(dataset: str, bundle, train_cache: Mapping[str, Any], val_runtimes: Mapping[str, Any]) -> list[tuple[int, int, Mapping[str, Any]]]:
    n_train = int(train_cache["num_windows"])
    split_boundary = router_train_block_split(dataset, train_cache)
    if split_boundary is None:
        return [(0, n_train, dict(val_runtimes))]
    rt_a = {e: load_expert_runtime(dataset, e, stage="block_a") for e in bundle.core_names}
    rt_ab = {e: load_expert_runtime(dataset, e, stage="block_ab") for e in bundle.core_names}
    return [(0, split_boundary, rt_a), (split_boundary, n_train, rt_ab)]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def make_generator(mode: str, input_len: int, num_features: int) -> torch.nn.Module:
    if mode == "instance":
        return ProbeGenerator(num_features, eps=EPS)
    return GlobalProbeGenerator(input_len, num_features, eps=EPS)


def run_batch(
    mode: str,
    generator: torch.nn.Module,
    scorer: CompetenceScorer,
    history_batch: torch.Tensor,
    batch_idx: torch.Tensor,
    core_names: Sequence[str],
    runtimes_stage: Mapping[str, Any],
    static_norm: torch.Tensor,
    group_b: torch.Tensor,
    forecasts_all: torch.Tensor,
    std: torch.Tensor,
    grad_enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (pred_excess [B,K], deltas [B,K,L,F], probe_response_stats [B,K,6])."""
    hist_std = history_batch.std(dim=1).clamp_min(1e-6)
    preds, deltas, probe_feats_all = [], [], []
    ctx = torch.enable_grad() if grad_enabled else torch.no_grad()
    with ctx:
        for local_i, expert_name in enumerate(core_names):
            rt = runtimes_stage[expert_name]
            if mode == "instance":
                fsum = group_b[batch_idx, local_i, :]
                window_norm = (history_batch - rt.mean.view(1, 1, -1)) / rt.std.view(1, 1, -1)
                x_probe, delta = generator.make_probe(history_batch, window_norm, fsum, hist_std)
            else:
                x_probe, delta = generator.make_probe(history_batch, hist_std)
            p_probe = rt.predict_differentiable(x_probe)
            original_forecast = forecasts_all[batch_idx][..., local_i].detach()
            probe_feats = probe_response_features(original_forecast, p_probe, std)
            full_feats = torch.cat([static_norm[batch_idx, local_i, :], probe_feats], dim=-1)
            preds.append(scorer(full_feats))
            deltas.append(delta)
            probe_feats_all.append(probe_feats)
    pred_excess = torch.stack(preds, dim=1)
    delta_stacked = torch.stack(deltas, dim=1)  # [B,K,L,F] -- matches pred_excess's [B,K] window-major/expert-minor layout
    probe_response = torch.stack(probe_feats_all, dim=1)
    return pred_excess, delta_stacked, probe_response


def train_probe_and_scorer(dataset: str, mode: str) -> dict[str, Any]:
    bundle = LOADERS[dataset]()
    train_cache = bundle.train_cache
    k = len(bundle.core_names)

    val_runtimes = {e: load_expert_runtime(dataset, e) for e in bundle.core_names}
    reference_runtime = val_runtimes[bundle.core_names[0]]
    train_cache_raw = raw_history_cache(dataset, train_cache, reference_runtime.mean, reference_runtime.std)

    group_a, group_b, group_c, forecasts_all = build_abc_features(bundle, train_cache_raw)
    excess_loss_train, _ = compute_excess_loss(train_cache, forecasts_all, bundle.std)
    history_raw_all = train_cache_raw["histories"].to(torch.float32)

    n_train = int(train_cache["num_windows"])
    split_point = int(round(n_train * (1 - INTERNAL_VAL_FRACTION)))
    stage_groups = stage_runtime_groups(dataset, bundle, train_cache, val_runtimes)

    # Snapshot every expert parameter used anywhere (final_60/block_a/block_ab) to verify frozen-ness after training.
    all_runtimes: dict[str, Any] = dict(val_runtimes)
    for lo, hi, rts in stage_groups:
        for name, rt in rts.items():
            all_runtimes[f"{lo}:{hi}:{name}"] = rt
    param_snapshots_before = {key: [p.detach().clone() for p in rt.model.parameters()] for key, rt in all_runtimes.items()}

    static = torch.cat([group_a, group_b, group_c], dim=-1)  # [N,K,15]
    static_flat = static.reshape(-1, STATIC_FEATURE_DIM)
    n_train_rows = split_point * k
    feat_mean = static_flat[:n_train_rows].mean(dim=0, keepdim=True)
    feat_std = static_flat[:n_train_rows].std(dim=0, keepdim=True).clamp_min(1e-6)
    static_norm = ((static_flat - feat_mean) / feat_std).reshape(n_train, k, STATIC_FEATURE_DIM)

    torch.manual_seed(7)
    input_len, num_features = history_raw_all.shape[1], history_raw_all.shape[2]
    generator = make_generator(mode, input_len, num_features)
    scorer = CompetenceScorer(STATIC_FEATURE_DIM + PROBE_FEATURE_DIM)
    optimizer = torch.optim.AdamW(list(generator.parameters()) + list(scorer.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)

    def loss_for_batch(batch_idx: torch.Tensor, runtimes_stage: Mapping[str, Any], grad_enabled: bool) -> torch.Tensor:
        history_batch = history_raw_all[batch_idx]
        pred_excess, deltas, _ = run_batch(mode, generator, scorer, history_batch, batch_idx, bundle.core_names, runtimes_stage, static_norm, group_b, forecasts_all, bundle.std, grad_enabled)
        actual = excess_loss_train[batch_idx]
        huber = F.huber_loss(pred_excess.reshape(-1), actual.reshape(-1), delta=1.0)
        ranking = pairwise_ranking_loss(pred_excess, actual)
        l2, mean_shift, smoothness = perturbation_penalties(deltas.reshape(-1, *deltas.shape[2:]))
        loss = huber + RANKING_WEIGHT * ranking + PERTURBATION_WEIGHT * (l2 + mean_shift) + SMOOTHNESS_WEIGHT * smoothness
        return loss

    best_val, best_epoch, bad = float("inf"), -1, 0
    best_state = None
    for epoch in range(1, MAX_EPOCHS + 1):
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
                loss = loss_for_batch(batch_idx, runtimes_stage, grad_enabled=True)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        generator.eval()
        scorer.eval()
        val_losses = []
        for lo, hi, runtimes_stage in stage_groups:
            val_lo = min(max(lo, split_point), hi)  # stage entirely inside the "train" portion -> empty range, not an inverted one
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
            best_state = {"generator": copy.deepcopy(generator.state_dict()), "scorer": copy.deepcopy(scorer.state_dict())}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    generator.load_state_dict(best_state["generator"])
    scorer.load_state_dict(best_state["scorer"])

    # Temperature selection on the internal validation slice only (small predefined grid, no test/val access).
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
                pred_excess, _, _ = run_batch(mode, generator, scorer, history_raw_all[batch_idx], batch_idx, bundle.core_names, runtimes_stage, static_norm, group_b, forecasts_all, bundle.std, grad_enabled=False)
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

    # Verify every touched expert parameter is bit-identical to before training.
    frozen_ok = True
    for key, rt in all_runtimes.items():
        for p_before, p_after in zip(param_snapshots_before[key], rt.model.parameters()):
            if not torch.equal(p_before, p_after):
                frozen_ok = False

    return {
        "dataset": dataset,
        "mode": mode,
        "generator": generator,
        "scorer": scorer,
        "temperature": best_temp,
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "best_epoch": best_epoch,
        "best_internal_val_loss": best_val,
        "experts_remained_frozen": frozen_ok,
        "val_runtimes": val_runtimes,
    }


# ---------------------------------------------------------------------------
# Evaluation on router_val (frozen generator + scorer, no gradients)
# ---------------------------------------------------------------------------


def evaluate_on_val(dataset: str, bundle, fit: Mapping[str, Any], val_cache: Mapping[str, Any]) -> dict[str, Any]:
    mode, generator, scorer = fit["mode"], fit["generator"], fit["scorer"]
    val_runtimes = fit["val_runtimes"]
    reference_runtime = val_runtimes[bundle.core_names[0]]
    val_cache_raw = raw_history_cache(dataset, val_cache, reference_runtime.mean, reference_runtime.std)
    group_a, group_b, group_c, forecasts_all = build_abc_features(bundle, val_cache_raw)
    static = torch.cat([group_a, group_b, group_c], dim=-1)
    static_norm = (static - fit["feat_mean"]) / fit["feat_std"]
    history_raw_all = val_cache_raw["histories"].to(torch.float32)
    n_val = int(val_cache["num_windows"])
    k = len(bundle.core_names)

    generator.eval()
    scorer.eval()
    all_pred, all_deltas, all_probe_response = [], [], []
    with torch.no_grad():
        for b in range(0, n_val, BATCH_SIZE):
            batch_idx = torch.arange(b, min(b + BATCH_SIZE, n_val))
            pred_excess, deltas, probe_response = run_batch(mode, generator, scorer, history_raw_all[batch_idx], batch_idx, bundle.core_names, val_runtimes, static_norm, group_b, forecasts_all, bundle.std, grad_enabled=False)
            all_pred.append(pred_excess)
            all_deltas.append(deltas)  # already [B,K,L,F]
            all_probe_response.append(probe_response)
    pred_excess_val = torch.cat(all_pred, dim=0)
    deltas_val = torch.cat(all_deltas, dim=0)  # [N,K,L,F]
    probe_response_val = torch.cat(all_probe_response, dim=0)  # [N,K,6]

    weights = competence_to_weights(pred_excess_val, fit["temperature"])
    final_pred = (forecasts_all * weights.view(n_val, 1, 1, k)).sum(dim=-1)
    return {"final_pred": final_pred, "pred_excess": pred_excess_val, "deltas": deltas_val, "probe_response": probe_response_val, "weights": weights}


def main() -> None:
    start = time.time()
    npz_orig = np.load(ORIGINAL_RESULTS_DIR / "per_window_predictions.npz")
    npz_orig_competence = np.load(ORIGINAL_RESULTS_DIR / "per_window_competence_predictions.npz")

    report: dict[str, Any] = {"experiment": "learned_diagnostic_probe", "created_at_utc": datetime.now(timezone.utc).isoformat(), "datasets": {}}
    all_results, all_dependence, all_competence, all_integrity = [], [], [], []
    probe_diag_rows: list[dict[str, Any]] = []
    per_window_npz: dict[str, np.ndarray] = {}

    for dataset in LOADERS:
        # LOADERS[dataset]() below calls refuse_test() internally on every real cache/config path it loads.
        print(f"[learned-probe] {dataset}: training instance-conditioned probe...", flush=True)
        fit_instance = train_probe_and_scorer(dataset, "instance")
        print(f"[learned-probe] {dataset}: training global probe...", flush=True)
        fit_global = train_probe_and_scorer(dataset, "global")

        bundle = LOADERS[dataset]()
        val_cache = bundle.val_cache
        eval_instance = evaluate_on_val(dataset, bundle, fit_instance, val_cache)
        eval_global = evaluate_on_val(dataset, bundle, fit_global, val_cache)

        c_pred = torch.tensor(npz_orig[f"{dataset}__C_window_forecast_disagreement"])
        d_pred = torch.tensor(npz_orig[f"{dataset}__D_full_behavioral"])
        equal_pred = torch.tensor(npz_orig[f"{dataset}__equal_fixed"])
        oracle_pred = torch.tensor(npz_orig[f"{dataset}__window_oracle"])
        best_single_pred = torch.tensor(npz_orig[f"{dataset}__best_single_expert"])

        methods = {
            "C": c_pred,
            "Fixed_D": d_pred,
            "Learned_Probe": eval_instance["final_pred"],
            "Learned_Global_Probe": eval_global["final_pred"],
            "Equal_reference": equal_pred,
            "Best_Single_reference": best_single_pred,
            "Window_Oracle_reference": oracle_pred,
        }
        result_rows, metrics = [], {}
        for method, pred in methods.items():
            m = metric_values(bundle, pred)
            metrics[method] = m
            result_rows.append({"dataset": dataset, "method": method, "mae": m["mae"], "mse": m["mse"]})
            per_window_npz[f"{dataset}__{method}"] = pred.numpy()
        per_window_npz[f"{dataset}__Learned_Probe_deltas"] = eval_instance["deltas"].numpy()
        per_window_npz[f"{dataset}__Learned_Probe_pred_excess"] = eval_instance["pred_excess"].numpy()

        # --- competence metrics ---
        target_train = val_cache["targets"].to(torch.float32)
        mask_train = val_cache["target_masks"].to(torch.bool)
        forecasts_all_val = bundle.forecasts_fn(val_cache, bundle.expert_idx)
        excess_loss_val, _ = compute_excess_loss(val_cache, forecasts_all_val, bundle.std)
        actual_flat = excess_loss_val.reshape(-1).numpy()
        for method_name, pred_excess in (("Learned_Probe", eval_instance["pred_excess"]), ("Learned_Global_Probe", eval_global["pred_excess"])):
            pred_flat = pred_excess.reshape(-1).numpy()
            sp = spearmanr(pred_flat, actual_flat)
            pe = pearsonr(pred_flat, actual_flat)
            top1 = float((pred_excess.numpy().argmin(axis=1) == excess_loss_val.numpy().argmin(axis=1)).mean())
            useful_label = (actual_flat < 0).astype(int)
            useful_score = -pred_flat
            auroc = float(roc_auc_score(useful_label, useful_score)) if useful_label.min() != useful_label.max() else float("nan")
            auprc = float(average_precision_score(useful_label, useful_score)) if useful_label.min() != useful_label.max() else float("nan")
            all_competence.append({"dataset": dataset, "method": method_name, "spearman": float(sp.statistic), "pearson": float(pe.statistic), "top1_accuracy": top1, "auroc_useful_vs_harmful": auroc, "auprc_useful_vs_harmful": auprc})
        # reference competence metrics for C and Fixed-D, recomputed from the original experiment's saved per-window predicted-excess arrays
        for method_name, npz_key in (("C_reference", "C_window_forecast_disagreement"), ("Fixed_D_reference", "D_full_behavioral")):
            pred_excess_ref = npz_orig_competence[f"{dataset}__{npz_key}__predicted"]
            pred_flat = pred_excess_ref.reshape(-1)
            sp = spearmanr(pred_flat, actual_flat)
            pe = pearsonr(pred_flat, actual_flat)
            top1 = float((pred_excess_ref.argmin(axis=1) == excess_loss_val.numpy().argmin(axis=1)).mean())
            useful_label = (actual_flat < 0).astype(int)
            useful_score = -pred_flat
            auroc = float(roc_auc_score(useful_label, useful_score)) if useful_label.min() != useful_label.max() else float("nan")
            auprc = float(average_precision_score(useful_label, useful_score)) if useful_label.min() != useful_label.max() else float("nan")
            all_competence.append({"dataset": dataset, "method": method_name, "spearman": float(sp.statistic), "pearson": float(pe.statistic), "top1_accuracy": top1, "auroc_useful_vs_harmful": auroc, "auprc_useful_vs_harmful": auprc})

        # --- dependence-aware stats: Learned-Probe vs C, Learned-Probe vs Fixed-D ---
        for label, cand_key, base_key in (("LearnedProbe_vs_C", "Learned_Probe", "C"), ("LearnedProbe_vs_FixedD", "Learned_Probe", "Fixed_D"), ("LearnedGlobalProbe_vs_C", "Learned_Global_Probe", "C"), ("LearnedProbe_vs_LearnedGlobalProbe", "Learned_Probe", "Learned_Global_Probe")):
            candidate, baseline = metrics[cand_key]["per_window_mae"], metrics[base_key]["per_window_mae"]
            boot = paired_bootstrap(candidate, baseline, seed=20260821, samples=5000)
            all_dependence.append({"dataset": dataset, "comparison": label, "test": "iid_paired_bootstrap", **boot})
            for block in BLOCK_LENGTHS:
                b = block_bootstrap_with_prob(candidate, baseline, block=block, seed=20260821, samples=BOOTSTRAP_SAMPLES)
                all_dependence.append({"dataset": dataset, "comparison": label, "test": f"block_bootstrap_len{block}", **b})
            phase = every_kth_phase_bootstrap(candidate - baseline, k=PHASE_K, seed=20260821, samples=BOOTSTRAP_SAMPLES)
            all_dependence.append({"dataset": dataset, "comparison": label, "test": f"every_{PHASE_K}th_window_phase_bootstrap", **phase})

        # --- oracle regret & headroom captured for Learned-Probe ---
        regret = metrics["Learned_Probe"]["per_window_mae"] - metrics["Window_Oracle_reference"]["per_window_mae"]
        oracle_mae, c_mae, lp_mae = metrics["Window_Oracle_reference"]["mae"], metrics["C"]["mae"], metrics["Learned_Probe"]["mae"]
        headroom_denom = c_mae - oracle_mae
        headroom_captured = float((c_mae - lp_mae) / headroom_denom) if headroom_denom > 0 else None
        regret_summary = {"mean_regret": float(regret.mean()), "median_regret": float(regret.median()), "p90_regret": float(torch.quantile(regret, 0.9)), "fraction_regret_gt_0": float((regret > 0).to(torch.float32).mean())}

        # --- probe diagnostics ---
        deltas = eval_instance["deltas"]  # [N,K,L,F]
        magnitude = deltas.abs()
        by_position = magnitude.mean(dim=(0, 1, 3))  # [L]
        by_expert = magnitude.mean(dim=(0, 2, 3))  # [K]
        by_window_std = magnitude.mean(dim=(1, 2, 3)).std()
        for local_i, expert_name in enumerate(bundle.core_names):
            probe_diag_rows.append(
                {
                    "dataset": dataset,
                    "expert": expert_name,
                    "mean_abs_magnitude": float(magnitude[:, local_i].mean()),
                    "max_abs_magnitude": float(magnitude[:, local_i].max()),
                    "concentration_first_quarter": float(magnitude[:, local_i, : magnitude.shape[2] // 4].mean()),
                    "concentration_last_quarter": float(magnitude[:, local_i, -magnitude.shape[2] // 4 :].mean()),
                    "across_window_std_of_magnitude": float(by_window_std),
                }
            )

        # --- integrity checks ---
        gen = torch.Generator().manual_seed(4242)
        corrupted_val_cache = dict(val_cache)
        corrupted_val_cache["targets"] = torch.randn(val_cache["targets"].shape, generator=gen, dtype=torch.float32)
        corrupted_eval = evaluate_on_val(dataset, bundle, fit_instance, corrupted_val_cache)
        predictions_identical = bool(torch.equal(eval_instance["final_pred"], corrupted_eval["final_pred"]))
        # probe magnitude constraint: |delta| <= eps * historical_std everywhere
        history_raw = raw_history_cache(dataset, val_cache, fit_instance["val_runtimes"][bundle.core_names[0]].mean, fit_instance["val_runtimes"][bundle.core_names[0]].std)["histories"].to(torch.float32)
        hist_std = history_raw.std(dim=1)  # [N,F]
        bound = EPS * hist_std.unsqueeze(1)  # [N,1,F]
        max_violation = float((deltas.abs() - bound.unsqueeze(1)).clamp_min(0).max())
        integrity = {
            "dataset": dataset,
            "target_corruption_predictions_identical": predictions_identical,
            "experts_remained_frozen_instance_mode": fit_instance["experts_remained_frozen"],
            "experts_remained_frozen_global_mode": fit_global["experts_remained_frozen"],
            "probe_magnitude_constraint_max_violation": max_violation,
            "probe_magnitude_constraint_satisfied": bool(max_violation < 1e-4),
            "result": "PASS" if (predictions_identical and fit_instance["experts_remained_frozen"] and fit_global["experts_remained_frozen"] and max_violation < 1e-4) else "FAIL",
        }
        all_integrity.append(integrity)
        if integrity["result"] != "PASS":
            raise AssertionError(f"{dataset}: learned-probe integrity check FAILED: {integrity}")

        report["datasets"][dataset] = {
            "core": bundle.core_names,
            "result_rows": result_rows,
            "headroom_captured_learned_probe": headroom_captured,
            "regret_summary": regret_summary,
            "temperature_instance": fit_instance["temperature"],
            "temperature_global": fit_global["temperature"],
            "best_epoch_instance": fit_instance["best_epoch"],
            "best_epoch_global": fit_global["best_epoch"],
        }
        all_results.extend(result_rows)
        print(f"[learned-probe] {dataset}: done. Learned-Probe MAE={lp_mae:.6f} (C={c_mae:.6f}, FixedD={metrics['Fixed_D']['mae']:.6f})", flush=True)

    report["runtime_sec"] = time.time() - start
    report["test_set_accessed"] = False
    write_json(RESULTS_DIR / "learned_probe_results.json", report)
    write_csv(RESULTS_DIR / "learned_probe_results.csv", all_results)
    write_csv(RESULTS_DIR / "learned_probe_dependence.csv", all_dependence)
    write_csv(RESULTS_DIR / "learned_probe_competence.csv", all_competence)
    write_csv(RESULTS_DIR / "learned_probe_diagnostics.csv", probe_diag_rows)
    write_csv(RESULTS_DIR / "learned_probe_integrity.csv", all_integrity)
    np.savez(RESULTS_DIR / "learned_probe_per_window.npz", **per_window_npz)
    print("TEST SET ACCESSED: NO")
    print(json.dumps({"runtime_sec": report["runtime_sec"]}, indent=2))


if __name__ == "__main__":
    main()
