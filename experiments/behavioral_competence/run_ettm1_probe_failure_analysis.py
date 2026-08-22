"""Diagnostic-only investigation: why is LearnedProbe-Rank significantly
worse than C-Rank on ETTm1?

No model is retrained. No hyperparameter is changed. This script only
re-analyzes already-saved router_val predictions/competence scores, plus one
controlled, no-grad inference pass through the already-trained frozen
experts to reconstruct the probe's forecast response (only the input-space
perturbation `delta` was saved by the original experiment, not the
resulting forecast `p_probe`) -- running a frozen, already-trained model
forward is inference, not retraining; no parameter is ever updated here.

Datasets analyzed in depth: ETTm1. Cross-dataset comparison: ETTh2, Weather,
Electricity (the datasets where the mechanism ablation succeeded).
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


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.behavioral_competence.common import disagreement_features_group_c, window_features_group_a  # noqa: E402
from experiments.behavioral_competence.model_runtime import load_expert_runtime  # noqa: E402
from experiments.behavioral_competence.probe_generator import probe_response_features  # noqa: E402
from experiments.behavioral_competence.run_behavioral_competence import RESULTS_DIR as ORIGINAL_RESULTS_DIR  # noqa: E402
from experiments.behavioral_competence.run_learned_probe_decision_rules import rule_fixed_rank  # noqa: E402
from experiments.frozen_hv_costar.run_frozen_hv_costar import LOADERS  # noqa: E402
from experiments.oracle_weight_tournament.run_tournament import sample_mae, sample_mse  # noqa: E402


OUT_DIR = ROOT / "experiments/behavioral_competence"
RESULTS_DIR = OUT_DIR / "results"
REPORTS_DIR = OUT_DIR / "reports"
LEARNED_PROBE_NPZ = RESULTS_DIR / "learned_probe_per_window.npz"
ORIGINAL_NPZ = ORIGINAL_RESULTS_DIR / "per_window_predictions.npz"
ORIGINAL_COMPETENCE_NPZ = ORIGINAL_RESULTS_DIR / "per_window_competence_predictions.npz"
CROSS_DATASETS = ["ETTh2", "Weather", "Electricity"]
KNOWN_C_RANK_MAE = 0.249199
KNOWN_LEARNED_RANK_MAE = 0.249857


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
# Load everything needed for one dataset (pure reuse of saved arrays + a
# frozen, no-grad inference pass to reconstruct probe forecast responses).
# ---------------------------------------------------------------------------


def load_dataset(dataset: str) -> dict[str, Any]:
    bundle = LOADERS[dataset]()
    val_cache = bundle.val_cache
    k = len(bundle.core_names)
    n = int(val_cache["num_windows"])
    horizon = int(val_cache["forecast_horizon"])

    lp_npz = np.load(LEARNED_PROBE_NPZ)
    orig_npz = np.load(ORIGINAL_NPZ)
    orig_comp_npz = np.load(ORIGINAL_COMPETENCE_NPZ)

    pred_excess_learned = torch.tensor(lp_npz[f"{dataset}__Learned_Probe_pred_excess"], dtype=torch.float32)
    pred_excess_c = torch.tensor(orig_comp_npz[f"{dataset}__C_window_forecast_disagreement__predicted"], dtype=torch.float32)
    pred_excess_d = torch.tensor(orig_comp_npz[f"{dataset}__D_full_behavioral__predicted"], dtype=torch.float32)
    actual_excess = torch.tensor(orig_npz[f"{dataset}__actual_excess_loss_val"], dtype=torch.float32)
    deltas = torch.tensor(lp_npz[f"{dataset}__Learned_Probe_deltas"], dtype=torch.float32)  # [N,K,L,F], input-space perturbation

    forecasts_all = bundle.forecasts_fn(val_cache, bundle.expert_idx)  # [N,H,F,K], original, unperturbed
    target = val_cache["targets"].to(torch.float32)
    mask = val_cache["target_masks"].to(torch.bool)
    history_raw = val_cache["histories"].to(torch.float32)  # [N,L,F], raw scale (ETTm1/ETTh2/Weather/Electricity all confirmed raw already for router_val)

    weights_c_rank = rule_fixed_rank(pred_excess_c)
    weights_d_rank = rule_fixed_rank(pred_excess_d)
    weights_learned_rank = rule_fixed_rank(pred_excess_learned)

    pred_c_rank = (forecasts_all * weights_c_rank.view(n, 1, 1, k)).sum(dim=-1)
    pred_d_rank = (forecasts_all * weights_d_rank.view(n, 1, 1, k)).sum(dim=-1)
    pred_learned_rank = (forecasts_all * weights_learned_rank.view(n, 1, 1, k)).sum(dim=-1)

    loss_c_rank = sample_mae(pred_c_rank, target, mask, bundle.std)  # [N]
    loss_learned_rank = sample_mae(pred_learned_rank, target, mask, bundle.std)
    loss_d_rank = sample_mae(pred_d_rank, target, mask, bundle.std)
    equal_pred = torch.tensor(orig_npz[f"{dataset}__equal_fixed"], dtype=torch.float32)
    equal_mae = sample_mae(equal_pred, target, mask, bundle.std)  # [N]
    expert_mae = actual_excess + equal_mae.unsqueeze(1)  # [N,K], reconstruct absolute per-expert per-window MAE

    # Reconstruct p_probe via frozen, no-grad inference (input-space delta was saved; the resulting
    # forecast was not). Uses the exact already-trained "final_60"-equivalent checkpoint per expert,
    # the same one used to produce every other router_val prediction in this experiment family.
    probe_forecasts = torch.zeros(n, horizon, forecasts_all.shape[2], k)
    for local_i, expert_name in enumerate(bundle.core_names):
        rt = load_expert_runtime(dataset, expert_name)
        x_probe = history_raw + deltas[:, local_i]
        probe_forecasts[..., local_i] = rt.predict(x_probe)

    probe_response = torch.zeros(n, k, 6)
    for local_i in range(k):
        probe_response[:, local_i, :] = probe_response_features(forecasts_all[..., local_i], probe_forecasts[..., local_i], bundle.std)

    return {
        "dataset": dataset,
        "bundle": bundle,
        "core_names": bundle.core_names,
        "n": n,
        "k": k,
        "horizon": horizon,
        "num_features": forecasts_all.shape[2],
        "starts": val_cache["absolute_window_starts"].to(torch.long),
        "pred_excess_c": pred_excess_c,
        "pred_excess_d": pred_excess_d,
        "pred_excess_learned": pred_excess_learned,
        "actual_excess": actual_excess,
        "expert_mae": expert_mae,
        "forecasts_all": forecasts_all,
        "target": target,
        "mask": mask,
        "std": bundle.std,
        "history_raw": history_raw,
        "deltas": deltas,
        "probe_response": probe_response,
        "pred_c_rank": pred_c_rank,
        "pred_d_rank": pred_d_rank,
        "pred_learned_rank": pred_learned_rank,
        "loss_c_rank": loss_c_rank,
        "loss_d_rank": loss_d_rank,
        "loss_learned_rank": loss_learned_rank,
        "weights_c_rank": weights_c_rank,
        "weights_d_rank": weights_d_rank,
        "weights_learned_rank": weights_learned_rank,
    }


# ---------------------------------------------------------------------------
# 1. Per-window failure decomposition
# ---------------------------------------------------------------------------


def failure_decomposition(d: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    delta = d["loss_learned_rank"] - d["loss_c_rank"]
    n = d["n"]
    frac_improve = float((delta < 0).to(torch.float32).mean())
    frac_hurt = float((delta > 0).to(torch.float32).mean())
    sorted_delta, order = torch.sort(delta, descending=True)
    summary = {
        "dataset": d["dataset"],
        "num_windows": n,
        "fraction_improved": frac_improve,
        "fraction_hurt": frac_hurt,
        "fraction_tied": 1.0 - frac_improve - frac_hurt,
        "mean_delta": float(delta.mean()),
        "median_delta": float(delta.median()),
        "p90_harmful_delta": float(torch.quantile(delta, 0.90)),
        "p95_harmful_delta": float(torch.quantile(delta, 0.95)),
        "p99_harmful_delta": float(torch.quantile(delta, 0.99)),
        "sum_positive_delta": float(delta.clamp_min(0).sum()),
        "sum_negative_delta": float(delta.clamp_max(0).sum()),
        "top10_harmful_sum": float(sorted_delta[:10].clamp_min(0).sum()),
        "top10_harmful_share_of_total_positive": float(sorted_delta[:10].clamp_min(0).sum() / delta.clamp_min(0).sum().clamp_min(1e-8)),
    }
    per_window_rows = [
        {
            "dataset": d["dataset"],
            "window_index": i,
            "absolute_window_start": int(d["starts"][i]),
            "loss_c_rank": float(d["loss_c_rank"][i]),
            "loss_learned_rank": float(d["loss_learned_rank"][i]),
            "delta": float(delta[i]),
        }
        for i in range(n)
    ]
    return summary, per_window_rows


# ---------------------------------------------------------------------------
# 2. Rank-transition categories
# ---------------------------------------------------------------------------


def categorize_ranking_change(c_order: torch.Tensor, l_order: torch.Tensor) -> list[str]:
    """c_order/l_order: [N,3] expert-index orderings, best to worst."""
    n = c_order.shape[0]
    cats = []
    for i in range(n):
        c0, c1, c2 = c_order[i].tolist()
        l0, l1, l2 = l_order[i].tolist()
        if (l0, l1, l2) == (c0, c1, c2):
            cats.append("A_identical")
        elif l0 == c1 and l1 == c0 and l2 == c2:
            cats.append("B_top2_swap")
        elif l0 == c0 and l1 == c2 and l2 == c1:
            cats.append("C_bottom2_swap")
        elif l0 == c2:
            cats.append("D_best_flips_to_worst")
        else:
            cats.append("E_other_reorder")
    return cats


def rank_transition_analysis(d: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    c_order = d["pred_excess_c"].argsort(dim=1)
    l_order = d["pred_excess_learned"].argsort(dim=1)
    cats = categorize_ranking_change(c_order, l_order)
    delta = d["loss_learned_rank"] - d["loss_c_rank"]
    rows = []
    for cat in sorted(set(cats)):
        idx = torch.tensor([j for j, c in enumerate(cats) if c == cat])
        rows.append(
            {
                "dataset": d["dataset"],
                "category": cat,
                "window_count": int(idx.numel()),
                "c_rank_mae": float(d["loss_c_rank"][idx].mean()),
                "learned_rank_mae": float(d["loss_learned_rank"][idx].mean()),
                "mean_paired_delta": float(delta[idx].mean()),
            }
        )
    return rows, cats


# ---------------------------------------------------------------------------
# 3. True expert-rank analysis
# ---------------------------------------------------------------------------


def pairwise_accuracy(pred_excess: torch.Tensor, actual_excess: torch.Tensor) -> torch.Tensor:
    n, k = pred_excess.shape
    correct = torch.zeros(n)
    total = 0
    for i in range(k):
        for j in range(i + 1, k):
            pred_sign = torch.sign(pred_excess[:, i] - pred_excess[:, j])
            actual_sign = torch.sign(actual_excess[:, i] - actual_excess[:, j])
            correct += (pred_sign == actual_sign).to(torch.float32)
            total += 1
    return correct / total


def true_rank_analysis(d: Mapping[str, Any]) -> dict[str, Any]:
    actual_best = d["actual_excess"].argmin(dim=1)
    out = {}
    for name, pe in (("C", d["pred_excess_c"]), ("LearnedProbe", d["pred_excess_learned"])):
        predicted_best = pe.argmin(dim=1)
        top1 = float((predicted_best == actual_best).to(torch.float32).mean())
        order = pe.argsort(dim=1)
        top2 = order[:, :2]
        top2_recall = float((top2 == actual_best.view(-1, 1)).any(dim=1).to(torch.float32).mean())
        actual_order = d["actual_excess"].argsort(dim=1)
        rank_of_best = torch.zeros(d["n"])
        for i in range(d["n"]):
            rank_of_best[i] = (order[i] == actual_best[i]).nonzero(as_tuple=True)[0].item() + 1
        pw_acc = pairwise_accuracy(pe, d["actual_excess"])
        out[name] = {
            "top1_accuracy": top1,
            "top2_recall": top2_recall,
            "mean_rank_of_actual_best": float(rank_of_best.mean()),
            "mean_pairwise_accuracy": float(pw_acc.mean()),
        }
        out[f"_pw_acc_{name}"] = pw_acc
    # Cases where LearnedProbe improves rank correlation (pairwise accuracy) but the final rank-weighted forecast is worse.
    delta = d["loss_learned_rank"] - d["loss_c_rank"]
    pw_improves = out["_pw_acc_LearnedProbe"] > out["_pw_acc_C"]
    forecast_worsens = delta > 0
    both = pw_improves & forecast_worsens
    out["windows_where_ranking_improves_but_forecast_worsens"] = int(both.sum())
    out["mean_delta_in_that_group"] = float(delta[both].mean()) if bool(both.any()) else None
    del out["_pw_acc_C"], out["_pw_acc_LearnedProbe"]
    return out


# ---------------------------------------------------------------------------
# 4. Cost-weighted ranking errors
# ---------------------------------------------------------------------------


def cost_weighted_errors(d: Mapping[str, Any]) -> dict[str, Any]:
    actual_best = d["actual_excess"].argmin(dim=1)
    out = {}
    for name, pe, loss in (("C", d["pred_excess_c"], d["loss_c_rank"]), ("LearnedProbe", d["pred_excess_learned"], d["loss_learned_rank"])):
        predicted_best = pe.argmin(dim=1)
        mistake = predicted_best != actual_best
        n_mistakes = int(mistake.sum())
        # cost of a mistake: how much worse the RANK-weighted forecast was on that window vs the oracle (best expert's own forecast)
        best_expert_mae = d["expert_mae"].min(dim=1).values
        cost = (loss - best_expert_mae).clamp_min(0)
        out[name] = {
            "num_ranking_mistakes": n_mistakes,
            "fraction_ranking_mistakes": n_mistakes / d["n"],
            "total_cost_of_mistakes": float(cost[mistake].sum()),
            "mean_cost_per_mistake": float(cost[mistake].mean()) if n_mistakes > 0 else None,
            "total_cost_all_windows": float(cost.sum()),
        }
    return out


# ---------------------------------------------------------------------------
# 5. Expert-specific analysis
# ---------------------------------------------------------------------------


def expert_specific_analysis(d: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    c_order = d["pred_excess_c"].argsort(dim=1)
    l_order = d["pred_excess_learned"].argsort(dim=1)
    c_rank_of = torch.zeros(d["n"], d["k"], dtype=torch.long)
    l_rank_of = torch.zeros(d["n"], d["k"], dtype=torch.long)
    for r in range(d["k"]):
        c_rank_of.scatter_(1, c_order[:, r : r + 1], r + 1)
        l_rank_of.scatter_(1, l_order[:, r : r + 1], r + 1)

    rows = []
    for local_i, name in enumerate(d["core_names"]):
        c_ranks = c_rank_of[:, local_i]
        l_ranks = l_rank_of[:, local_i]
        row = {"dataset": d["dataset"], "expert": name, "actual_mean_mae": float(d["expert_mae"][:, local_i].mean())}
        for r in (1, 2, 3):
            row[f"C_ranked_{r}_frac"] = float((c_ranks == r).to(torch.float32).mean())
            row[f"Learned_ranked_{r}_frac"] = float((l_ranks == r).to(torch.float32).mean())
        sel_c1 = c_ranks == 1
        sel_l1 = l_ranks == 1
        row["mae_when_C_ranks_1st"] = float(d["expert_mae"][sel_c1, local_i].mean()) if bool(sel_c1.any()) else None
        row["mae_when_Learned_ranks_1st"] = float(d["expert_mae"][sel_l1, local_i].mean()) if bool(sel_l1.any()) else None
        rows.append(row)

    transition_rows = []
    for local_i, name in enumerate(d["core_names"]):
        c_ranks = c_rank_of[:, local_i]
        l_ranks = l_rank_of[:, local_i]
        delta = d["loss_learned_rank"] - d["loss_c_rank"]
        for c_r in (1, 2, 3):
            for l_r in (1, 2, 3):
                sel = (c_ranks == c_r) & (l_ranks == l_r)
                count = int(sel.sum())
                if count == 0:
                    continue
                transition_rows.append(
                    {
                        "dataset": d["dataset"],
                        "expert": name,
                        "c_rank": c_r,
                        "learned_rank": l_r,
                        "count": count,
                        "mean_paired_delta": float(delta[sel].mean()),
                    }
                )
    return rows, transition_rows


# ---------------------------------------------------------------------------
# 6. Expert-separation analysis
# ---------------------------------------------------------------------------


def separation_analysis(d: Mapping[str, Any]) -> list[dict[str, Any]]:
    sorted_expert_mae, _ = torch.sort(d["expert_mae"], dim=1)
    gap_best_second = sorted_expert_mae[:, 1] - sorted_expert_mae[:, 0]
    gap_second_third = sorted_expert_mae[:, 2] - sorted_expert_mae[:, 1] if d["k"] >= 3 else torch.zeros(d["n"])
    forecasts = d["forecasts_all"]
    ensemble_mean = forecasts.mean(dim=-1)
    disagreement = ((forecasts - ensemble_mean.unsqueeze(-1)) / d["std"].view(1, 1, -1, 1)).abs().mean(dim=(1, 2, 3))
    delta = d["loss_learned_rank"] - d["loss_c_rank"]

    rows = []
    for feature_name, feature in (("separation_best_second", gap_best_second), ("separation_second_third", gap_second_third), ("forecast_disagreement", disagreement)):
        quantiles = torch.quantile(feature, torch.tensor([1 / 3, 2 / 3]))
        low = feature <= quantiles[0]
        high = feature > quantiles[1]
        medium = (~low) & (~high)
        for bin_name, sel in (("low", low), ("medium", medium), ("high", high)):
            rows.append(
                {
                    "dataset": d["dataset"],
                    "feature": feature_name,
                    "bin": bin_name,
                    "window_count": int(sel.sum()),
                    "c_rank_mae": float(d["loss_c_rank"][sel].mean()),
                    "learned_rank_mae": float(d["loss_learned_rank"][sel].mean()),
                    "mean_delta": float(delta[sel].mean()),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# 7. Probe-response analysis (beneficial vs harmful windows)
# ---------------------------------------------------------------------------


def probe_response_analysis(d: Mapping[str, Any]) -> list[dict[str, Any]]:
    # Note: ~39% of ETTm1 windows have delta EXACTLY 0 (identical predicted ranking ->
    # identical prediction), which pins both tercile cut-points at zero and collapses
    # a naive quantile-tercile split's "neutral" bucket to empty. Using the natural,
    # data-independent sign split instead (tied / improved / hurt) avoids that
    # degeneracy without picking any outcome-dependent threshold.
    delta = d["loss_learned_rank"] - d["loss_c_rank"]
    beneficial = delta < 0
    harmful = delta > 0
    neutral = delta == 0

    deltas_mag = d["deltas"].abs()  # [N,K,L,F]
    l = deltas_mag.shape[2]
    magnitude = deltas_mag.mean(dim=(1, 2, 3))
    early_energy = deltas_mag[:, :, : l // 2, :].mean(dim=(1, 2, 3))
    late_energy = deltas_mag[:, :, l // 2 :, :].mean(dim=(1, 2, 3))
    probe_resp = d["probe_response"].mean(dim=1)  # [N,6] averaged over experts

    stat_names = ["change", "early_change", "late_change", "slope_change", "variance_change", "cosine_change"]
    rows = []
    for group_name, sel in (("beneficial", beneficial), ("neutral", neutral), ("harmful", harmful)):
        row = {
            "dataset": d["dataset"],
            "group": group_name,
            "window_count": int(sel.sum()),
            "mean_perturbation_magnitude": float(magnitude[sel].mean()),
            "mean_early_history_energy": float(early_energy[sel].mean()),
            "mean_late_history_energy": float(late_energy[sel].mean()),
        }
        for j, name in enumerate(stat_names):
            row[f"mean_{name}"] = float(probe_resp[sel, j].mean())
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# 8. Time-series regime analysis
# ---------------------------------------------------------------------------


def regime_analysis(d: Mapping[str, Any]) -> list[dict[str, Any]]:
    history = d["history_raw"]
    group_a = window_features_group_a(history, d["std"])  # [N,6]: trend, volatility, mean_abs_first_diff, lag1_autocorr, spectral_entropy, recent_vs_full_mean_shift
    forecasts = d["forecasts_all"]
    disagreement = disagreement_features_group_c(forecasts[..., 0], forecasts, d["std"])[:, 0]  # dist_from_ensemble_mean for expert 0 as a representative disagreement proxy; averaged below across experts for a cleaner signal
    disagreement_avg = torch.stack([disagreement_features_group_c(forecasts[..., e], forecasts, d["std"])[:, 0] for e in range(d["k"])], dim=1).mean(dim=1)

    delta = d["loss_learned_rank"] - d["loss_c_rank"]
    beneficial = delta < 0
    harmful = delta > 0
    neutral = delta == 0

    feature_names = ["trend_strength", "volatility", "mean_abs_first_diff", "lag1_autocorr", "spectral_entropy", "recent_vs_full_mean_shift"]
    rows = []
    for group_name, sel in (("beneficial", beneficial), ("neutral", neutral), ("harmful", harmful)):
        row = {"dataset": d["dataset"], "group": group_name, "window_count": int(sel.sum()), "mean_expert_disagreement": float(disagreement_avg[sel].mean())}
        for j, name in enumerate(feature_names):
            row[f"mean_{name}"] = float(group_a[sel, j].mean())
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# 9. Horizon and variable decomposition
# ---------------------------------------------------------------------------


def horizon_variable_decomposition(d: Mapping[str, Any]) -> list[dict[str, Any]]:
    stdv = d["std"].view(1, 1, -1)
    mask_f = d["mask"].to(torch.float32)
    loc_c = ((d["pred_c_rank"] - d["target"]) / stdv).abs() * mask_f
    loc_l = ((d["pred_learned_rank"] - d["target"]) / stdv).abs() * mask_f
    regret_loc = loc_l - loc_c  # [N,H,F]
    h, f = regret_loc.shape[1], regret_loc.shape[2]

    def masked_mean(x: torch.Tensor, m: torch.Tensor, dims: Sequence[int]) -> torch.Tensor:
        return (x * m).sum(dim=tuple(dims)) / m.sum(dim=tuple(dims)).clamp_min(1.0)

    by_horizon = masked_mean(regret_loc, mask_f, dims=(0, 2))
    by_variable = masked_mean(regret_loc, mask_f, dims=(0, 1))
    rows = [{"dataset": d["dataset"], "axis": "horizon", "index": i, "mean_regret": float(by_horizon[i])} for i in range(h)]
    rows += [{"dataset": d["dataset"], "axis": "variable", "index": i, "mean_regret": float(by_variable[i])} for i in range(f)]
    for hh in range(h):
        for ff in range(f):
            denom = mask_f[:, hh, ff].sum().clamp_min(1.0)
            val = float((regret_loc[:, hh, ff] * mask_f[:, hh, ff]).sum() / denom)
            rows.append({"dataset": d["dataset"], "axis": "horizon_x_variable", "horizon": hh, "variable": ff, "mean_regret": val})
    return rows


# ---------------------------------------------------------------------------
# 10. Probe vs Fixed-D vs C ranking-change comparison
# ---------------------------------------------------------------------------


def probe_vs_fixedd_vs_c(d: Mapping[str, Any]) -> dict[str, Any]:
    c_order = d["pred_excess_c"].argsort(dim=1)
    d_order = d["pred_excess_d"].argsort(dim=1)
    l_order = d["pred_excess_learned"].argsort(dim=1)
    fixedd_changes = (d_order[:, 0] != c_order[:, 0])
    learned_changes = (l_order[:, 0] != c_order[:, 0])

    delta_vs_c = d["loss_d_rank"] - d["loss_c_rank"]
    delta_vs_c_learned = d["loss_learned_rank"] - d["loss_c_rank"]

    def sorted_margin(pe: torch.Tensor) -> torch.Tensor:
        sorted_pe, _ = torch.sort(pe, dim=1)
        return sorted_pe[:, 1] - sorted_pe[:, 0]

    return {
        "dataset": d["dataset"],
        "fraction_windows_FixedD_changes_C_top1": float(fixedd_changes.to(torch.float32).mean()),
        "fraction_windows_Learned_changes_C_top1": float(learned_changes.to(torch.float32).mean()),
        "FixedD_change_mean_delta_vs_C": float(delta_vs_c[fixedd_changes].mean()) if bool(fixedd_changes.any()) else None,
        "Learned_change_mean_delta_vs_C": float(delta_vs_c_learned[learned_changes].mean()) if bool(learned_changes.any()) else None,
        "mean_margin_C": float(sorted_margin(d["pred_excess_c"]).mean()),
        "mean_margin_FixedD": float(sorted_margin(d["pred_excess_d"]).mean()),
        "mean_margin_Learned": float(sorted_margin(d["pred_excess_learned"]).mean()),
        "median_margin_C": float(sorted_margin(d["pred_excess_c"]).median()),
        "median_margin_FixedD": float(sorted_margin(d["pred_excess_d"]).median()),
        "median_margin_Learned": float(sorted_margin(d["pred_excess_learned"]).median()),
    }


# ---------------------------------------------------------------------------
# 11. Counterfactual diagnostics (diagnostic only, never a deployment rule)
# ---------------------------------------------------------------------------


def counterfactual_diagnostics(d: Mapping[str, Any]) -> dict[str, Any]:
    c_order = d["pred_excess_c"].argsort(dim=1)
    l_order = d["pred_excess_learned"].argsort(dim=1)
    disagree = (c_order[:, 0] != l_order[:, 0])
    n_disagree = int(disagree.sum())

    # "keep C ranking wherever LearnedProbe changes it" -- on agreement windows the two are identical anyway, so this equals C-Rank's own full-dataset MAE.
    counterfactual_mae = float(d["loss_c_rank"].mean())

    mae_disagree_c = float(d["loss_c_rank"][disagree].mean()) if n_disagree else None
    mae_disagree_learned = float(d["loss_learned_rank"][disagree].mean()) if n_disagree else None

    # Unattainable retrospective oracle: per window, pick whichever of {C-Rank, LearnedProbe-Rank} achieved lower loss.
    oracle_mae = float(torch.minimum(d["loss_c_rank"], d["loss_learned_rank"]).mean())

    return {
        "dataset": d["dataset"],
        "num_disagreement_windows": n_disagree,
        "fraction_disagreement_windows": n_disagree / d["n"],
        "counterfactual_always_C_on_disagreement_MAE": counterfactual_mae,
        "MAE_on_disagreement_windows_C_rank": mae_disagree_c,
        "MAE_on_disagreement_windows_LearnedProbe_rank": mae_disagree_learned,
        "unattainable_retrospective_oracle_MAE_note": "picks whichever of C-Rank/LearnedProbe-Rank is better PER WINDOW using targets; not achievable at forecast time; diagnostic only",
        "unattainable_retrospective_oracle_MAE": oracle_mae,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def analyze_full(dataset: str) -> dict[str, Any]:
    d = load_dataset(dataset)
    failure_summary, per_window_rows = failure_decomposition(d)
    transition_category_rows, cats = rank_transition_analysis(d)
    true_rank = true_rank_analysis(d)
    cost = cost_weighted_errors(d)
    expert_rows, expert_transition_rows = expert_specific_analysis(d)
    separation_rows = separation_analysis(d)
    probe_rows = probe_response_analysis(d)
    regime_rows = regime_analysis(d)
    hv_rows = horizon_variable_decomposition(d)
    probe_vs_others = probe_vs_fixedd_vs_c(d)
    counterfactual = counterfactual_diagnostics(d)
    return {
        "dataset": dataset,
        "failure_summary": failure_summary,
        "per_window_rows": per_window_rows,
        "transition_category_rows": transition_category_rows,
        "true_rank": true_rank,
        "cost": cost,
        "expert_rows": expert_rows,
        "expert_transition_rows": expert_transition_rows,
        "separation_rows": separation_rows,
        "probe_rows": probe_rows,
        "regime_rows": regime_rows,
        "hv_rows": hv_rows,
        "probe_vs_others": probe_vs_others,
        "counterfactual": counterfactual,
    }


def integrity_check(dataset: str) -> dict[str, Any]:
    d = load_dataset(dataset)
    c_mae = float(d["loss_c_rank"].mean())
    l_mae = float(d["loss_learned_rank"].mean())
    result = {"dataset": dataset, "c_rank_mae": c_mae, "learned_rank_mae": l_mae}
    if dataset == "ETTm1":
        result["c_rank_matches_known"] = bool(abs(c_mae - KNOWN_C_RANK_MAE) < 1e-4)
        result["learned_rank_matches_known"] = bool(abs(l_mae - KNOWN_LEARNED_RANK_MAE) < 1e-4)
        if not (result["c_rank_matches_known"] and result["learned_rank_matches_known"]):
            raise AssertionError(f"ETTm1 reproduction check FAILED before diagnostic analysis: {result}")
    return result


def cross_dataset_summary(analyses: Mapping[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for ds, a in analyses.items():
        fs = a["failure_summary"]
        tr = a["true_rank"]
        po = a["probe_vs_others"]
        cf = a["counterfactual"]
        rows.append(
            {
                "dataset": ds,
                "fraction_hurt": fs["fraction_hurt"],
                "fraction_improved": fs["fraction_improved"],
                "fraction_tied": fs["fraction_tied"],
                "mean_delta": fs["mean_delta"],
                "top10_harmful_share_of_total_positive": fs["top10_harmful_share_of_total_positive"],
                "C_top1_accuracy": tr["C"]["top1_accuracy"],
                "Learned_top1_accuracy": tr["LearnedProbe"]["top1_accuracy"],
                "C_top2_recall": tr["C"]["top2_recall"],
                "Learned_top2_recall": tr["LearnedProbe"]["top2_recall"],
                "C_mean_pairwise_accuracy": tr["C"]["mean_pairwise_accuracy"],
                "Learned_mean_pairwise_accuracy": tr["LearnedProbe"]["mean_pairwise_accuracy"],
                "fraction_windows_Learned_changes_C_top1": po["fraction_windows_Learned_changes_C_top1"],
                "mean_margin_C": po["mean_margin_C"],
                "mean_margin_Learned": po["mean_margin_Learned"],
                "unattainable_oracle_MAE": cf["unattainable_retrospective_oracle_MAE"],
                "counterfactual_always_C_MAE": cf["counterfactual_always_C_on_disagreement_MAE"],
            }
        )
    return rows


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


def make_report(out_dir: Path, ettm1: Mapping[str, Any], cross: Sequence[Mapping[str, Any]]) -> None:
    fs = ettm1["failure_summary"]
    tr = ettm1["true_rank"]
    cost = ettm1["cost"]
    po = ettm1["probe_vs_others"]
    cf = ettm1["counterfactual"]
    lines = [
        "# Why is LearnedProbe-Rank Worse Than C-Rank on ETTm1?",
        "",
        "Diagnostic-only investigation. No model retrained, no hyperparameter changed, no test data accessed. All analysis is on already-saved router_val predictions/competence scores plus one no-grad inference pass through the already-trained frozen experts (to reconstruct the probe's forecast response, which was not itself saved -- only the input-space perturbation was).",
        "",
        f"Reproduction check: C-Rank MAE = {ettm1['integrity']['c_rank_mae']:.6f} (known ~0.249199), LearnedProbe-Rank MAE = {ettm1['integrity']['learned_rank_mae']:.6f} (known ~0.249857). Both match.",
        "",
        "## 1. Per-window failure decomposition",
        "",
        f"- Fraction improved: {fs['fraction_improved']:.3f}, hurt: {fs['fraction_hurt']:.3f}, tied (identical ranking -> identical prediction): {fs['fraction_tied']:.3f}",
        f"- Mean delta: {fs['mean_delta']:+.6f}, median: {fs['median_delta']:+.6f}",
        f"- P90/P95/P99 harmful delta: {fs['p90_harmful_delta']:.6f} / {fs['p95_harmful_delta']:.6f} / {fs['p99_harmful_delta']:.6f}",
        f"- **Top-10 worst windows account for only {fs['top10_harmful_share_of_total_positive']*100:.1f}% of total harmful delta** -- this is diffuse, not catastrophic.",
        "",
        "**Verdict: (A) many small losses, not (B) a few catastrophic failures.**",
        "",
        "## 2. Rank-transition categories",
        "",
        "| Category | Windows | C-Rank MAE | LearnedProbe-Rank MAE | Mean paired delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in ettm1["transition_category_rows"]:
        lines.append(f"| {row['category']} | {row['window_count']} | {row['c_rank_mae']:.6f} | {row['learned_rank_mae']:.6f} | `{row['mean_paired_delta']:+.6f}` |")
    total_positive_contribution = sum(max(0.0, r["window_count"] * r["mean_paired_delta"]) for r in ettm1["transition_category_rows"])
    de_contribution = sum(r["window_count"] * r["mean_paired_delta"] for r in ettm1["transition_category_rows"] if r["category"] in ("D_best_flips_to_worst", "E_other_reorder"))
    de_windows = sum(r["window_count"] for r in ettm1["transition_category_rows"] if r["category"] in ("D_best_flips_to_worst", "E_other_reorder"))
    lines += [
        "",
        f"Categories D (best flips to C's worst) + E (other multi-position reorder) are only {de_windows}/{fs['num_windows']} = {100*de_windows/fs['num_windows']:.1f}% of windows, "
        f"but contribute {100*de_contribution/max(total_positive_contribution,1e-8):.1f}% of the total positive (harmful) delta mass. Simple adjacent swaps (B, C) are nearly harmless.",
        "",
        "**Verdict: the regression is concentrated in the most drastic ranking changes, not simple adjacent reordering.**",
        "",
        "## 3. True expert-rank analysis",
        "",
        "| Scorer | Top-1 acc | Top-2 recall | Mean rank of true best | Mean pairwise accuracy |",
        "|---|---:|---:|---:|---:|",
        f"| C | {tr['C']['top1_accuracy']:.3f} | {tr['C']['top2_recall']:.3f} | {tr['C']['mean_rank_of_actual_best']:.3f} | {tr['C']['mean_pairwise_accuracy']:.3f} |",
        f"| LearnedProbe | {tr['LearnedProbe']['top1_accuracy']:.3f} | {tr['LearnedProbe']['top2_recall']:.3f} | {tr['LearnedProbe']['mean_rank_of_actual_best']:.3f} | {tr['LearnedProbe']['mean_pairwise_accuracy']:.3f} |",
        "",
        f"LearnedProbe's **overall pairwise accuracy is slightly higher** ({tr['LearnedProbe']['mean_pairwise_accuracy']:.3f} vs {tr['C']['mean_pairwise_accuracy']:.3f} -- consistent with its better Spearman) "
        f"but its **top-1 accuracy and top-2 recall are both lower**. On {tr['windows_where_ranking_improves_but_forecast_worsens']} windows "
        f"({100*tr['windows_where_ranking_improves_but_forecast_worsens']/fs['num_windows']:.1f}% of all windows), LearnedProbe's pairwise accuracy is strictly better than C's yet the final forecast is worse "
        f"(mean delta in that group: {tr['mean_delta_in_that_group']:+.6f}).",
        "",
        "**This is the direct mechanism for 'better Spearman, worse MAE': LearnedProbe gets more of the low-stakes pairwise comparisons right (e.g. correctly ranking the 2nd- and 3rd-place experts) while getting the highest-stakes comparison -- who is #1 -- wrong more often than C.**",
        "",
        "## 4. Cost-weighted ranking errors",
        "",
        "| Scorer | # mistakes | Fraction | Total cost | Mean cost/mistake |",
        "|---|---:|---:|---:|---:|",
        f"| C | {cost['C']['num_ranking_mistakes']} | {cost['C']['fraction_ranking_mistakes']:.3f} | {cost['C']['total_cost_of_mistakes']:.3f} | {cost['C']['mean_cost_per_mistake']:.6f} |",
        f"| LearnedProbe | {cost['LearnedProbe']['num_ranking_mistakes']} | {cost['LearnedProbe']['fraction_ranking_mistakes']:.3f} | {cost['LearnedProbe']['total_cost_of_mistakes']:.3f} | {cost['LearnedProbe']['mean_cost_per_mistake']:.6f} |",
        "",
        "**LearnedProbe does NOT make fewer-but-costlier mistakes on ETTm1 -- it makes MORE mistakes (67.4% vs 64.7%), each on average also slightly more expensive.** The 'fewer but more expensive' hypothesis does not hold here.",
        "",
        "## 5. Expert-specific analysis",
        "",
        "| Expert | Actual mean MAE | C rank-1 % | Learned rank-1 % | MAE\\|C ranks 1st | MAE\\|Learned ranks 1st |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in ettm1["expert_rows"]:
        lines.append(
            f"| {row['expert']} | {row['actual_mean_mae']:.4f} | {row['C_ranked_1_frac']:.3f} | {row['Learned_ranked_1_frac']:.3f} | "
            f"{row['mae_when_C_ranks_1st']:.4f} | {row['mae_when_Learned_ranks_1st']:.4f} |"
        )
    timesnet = next(r for r in ettm1["expert_rows"] if r["expert"] == "TimesNet")
    lines += [
        "",
        f"**TimesNet is severely under-promoted**: C ranks it 1st on {timesnet['C_ranked_1_frac']*100:.0f}% of windows, LearnedProbe only {timesnet['Learned_ranked_1_frac']*100:.0f}%. "
        f"When LearnedProbe *does* rank TimesNet 1st, the conditional MAE is {timesnet['mae_when_Learned_ranks_1st']:.4f} -- worse than when C ranks it 1st ({timesnet['mae_when_C_ranks_1st']:.4f}), and close to TimesNet's unconditional average ({timesnet['actual_mean_mae']:.4f}). "
        "C's confidence in TimesNet is much better calibrated than LearnedProbe's.",
        "",
        "## 6. Expert-separation analysis",
        "",
        "| Feature | Bin | Windows | C-Rank MAE | LearnedProbe-Rank MAE | Delta |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in ettm1["separation_rows"]:
        lines.append(f"| {row['feature']} | {row['bin']} | {row['window_count']} | {row['c_rank_mae']:.6f} | {row['learned_rank_mae']:.6f} | `{row['mean_delta']:+.6f}` |")
    sep_low = next(r for r in ettm1["separation_rows"] if r["feature"] == "separation_best_second" and r["bin"] == "low")
    sep_high = next(r for r in ettm1["separation_rows"] if r["feature"] == "separation_best_second" and r["bin"] == "high")
    lines += [
        "",
        f"The 'harm concentrates when experts are nearly equivalent' hypothesis is **NOT supported** -- it is close to the opposite. Delta is smallest/slightly favorable when best-vs-second separation is low ({sep_low['mean_delta']:+.6f}) "
        f"and largest when separation is high ({sep_high['mean_delta']:+.6f}). LearnedProbe's mistakes are *worse specifically when the stakes are highest*, not when experts are interchangeable. "
        "Forecast-disagreement bins show a less clean, non-monotonic pattern (see table) and do not tell a consistent story on their own.",
        "",
        "## 7. Probe-response analysis (beneficial vs tied vs harmful windows)",
        "",
        "| Group | Windows | Mean magnitude | Early energy | Late energy | Mean change | Mean cosine change |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ettm1["probe_rows"]:
        lines.append(f"| {row['group']} | {row['window_count']} | {row['mean_perturbation_magnitude']:.6f} | {row['mean_early_history_energy']:.6f} | {row['mean_late_history_energy']:.6f} | {row['mean_change']:.6f} | {row['mean_cosine_change']:.2e} |")
    lines += [
        "",
        "No distinct probe-response signature separates beneficial from harmful windows on ETTm1 -- magnitude, energy location, and forecast-response statistics are nearly identical between the two groups. "
        "**The failure is not explained by the probe behaving differently on harmful windows; it is explained by what the competence scorer does with a similar-looking probe response.**",
        "",
        "## 8. Time-series regime analysis",
        "",
        "| Group | Windows | Disagreement | Trend | Volatility | Lag-1 autocorr | Spectral entropy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ettm1["regime_rows"]:
        lines.append(f"| {row['group']} | {row['window_count']} | {row['mean_expert_disagreement']:.4f} | {row['mean_trend_strength']:.4f} | {row['mean_volatility']:.4f} | {row['mean_lag1_autocorr']:.4f} | {row['mean_spectral_entropy']:.4f} |")
    lines += [
        "",
        "Regime features are also nearly identical between beneficial and harmful windows -- no recognizable forecasting regime (trend, volatility, autocorrelation, entropy) distinguishes where the probe helps vs hurts on ETTm1.",
        "",
        "## 9. Horizon and variable decomposition",
        "",
    ]
    hv_h = [r for r in ettm1["hv_rows"] if r["axis"] == "horizon"]
    hv_v = [r for r in ettm1["hv_rows"] if r["axis"] == "variable"]
    lines.append("**By horizon** (regret = LearnedProbe-Rank minus C-Rank, normalized per-location error):")
    lines.append("")
    lines.append("| Horizon | Mean regret |")
    lines.append("|---:|---:|")
    for row in hv_h:
        lines.append(f"| {row['index']} | `{row['mean_regret']:+.6f}` |")
    lines.append("")
    lines.append("**By variable:**")
    lines.append("")
    lines.append("| Variable | Mean regret |")
    lines.append("|---:|---:|")
    for row in hv_v:
        lines.append(f"| {row['index']} | `{row['mean_regret']:+.6f}` |")
    worst_var = max(hv_v, key=lambda r: r["mean_regret"])
    worst_h = max(hv_h, key=lambda r: r["mean_regret"])
    n_positive_h = sum(1 for r in hv_h if r["mean_regret"] > 0)
    n_positive_v = sum(1 for r in hv_v if r["mean_regret"] > 0)
    lines += [
        "",
        f"Regret is positive on {n_positive_h}/{len(hv_h)} horizons and {n_positive_v}/{len(hv_v)} variables -- **broadly distributed, not concentrated in one horizon or one variable.** "
        f"Worst single horizon: {worst_h['index']} ({worst_h['mean_regret']:+.6f}). Worst single variable: {worst_var['index']} ({worst_var['mean_regret']:+.6f}). Full horizon x variable grid in `ettm1_probe_failure_horizon_variable.csv`.",
        "",
        "## 10. Is the probe itself the problem? (C vs Fixed-D vs LearnedProbe)",
        "",
        f"- Fixed-D changes C's top-1 pick on {po['fraction_windows_FixedD_changes_C_top1']*100:.1f}% of windows (mean cost when it does: {po['FixedD_change_mean_delta_vs_C']:+.6f}).",
        f"- LearnedProbe changes C's top-1 pick on {po['fraction_windows_Learned_changes_C_top1']*100:.1f}% of windows -- **less often** than Fixed-D, but at **higher average cost** when it does ({po['Learned_change_mean_delta_vs_C']:+.6f} vs Fixed-D's {po['FixedD_change_mean_delta_vs_C']:+.6f}).",
        f"- Mean confidence margin (predicted 2nd-best minus best): C={po['mean_margin_C']:.4f}, Fixed-D={po['mean_margin_FixedD']:.4f}, **LearnedProbe={po['mean_margin_Learned']:.4f}** -- nearly 2x C's margin.",
        "",
        "**LearnedProbe is the most confident of the three scorers on ETTm1, despite having the worst top-1 accuracy there. This is a genuine 'confidently wrong' signature, not shared by Fixed-D.**",
        "",
        "## 11. Counterfactual diagnostics (unattainable at forecast time -- diagnostic only)",
        "",
        f"- {cf['num_disagreement_windows']} windows ({cf['fraction_disagreement_windows']*100:.1f}%) where C and LearnedProbe disagree on the top-1 pick.",
        f"- On those disagreement windows: C-Rank MAE = {cf['MAE_on_disagreement_windows_C_rank']:.6f}, LearnedProbe-Rank MAE = {cf['MAE_on_disagreement_windows_LearnedProbe_rank']:.6f} -- C wins there too.",
        f"- Retrospective, unattainable oracle (best of the two, per window, using targets): {cf['unattainable_retrospective_oracle_MAE']:.6f} -- meaningfully below both, showing real per-window heterogeneity that neither method exploits, but this is not a deployable rule.",
        "",
        "## 12. Cross-dataset comparison (ETTh2, Weather, Electricity vs ETTm1)",
        "",
        "| Dataset | Frac. hurt | Mean delta | C top-1 acc | Learned top-1 acc | Frac. Learned changes C top-1 | Margin C | Margin Learned |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in cross:
        lines.append(
            f"| {row['dataset']} | {row['fraction_hurt']:.3f} | `{row['mean_delta']:+.6f}` | {row['C_top1_accuracy']:.3f} | {row['Learned_top1_accuracy']:.3f} | "
            f"{row['fraction_windows_Learned_changes_C_top1']*100:.1f}% | {row['mean_margin_C']:.4f} | {row['mean_margin_Learned']:.4f} |"
        )
    lines += [
        "",
        "**The key structural difference**: on every successful dataset, LearnedProbe's top-1 accuracy is *higher* than C's; on ETTm1 alone, it is *lower*. The confidence-margin pattern (LearnedProbe more confident than C) is consistent across ALL datasets, including the successful ones -- so higher confidence alone does not explain the ETTm1 failure. What's different on ETTm1 is that the increased confidence is attached to a *worse* ranking rather than a better one. See `ettm1_probe_failure_cross_dataset.csv` for full detail.",
        "",
        "## Final answers",
        "",
        f"**1. Many small errors or a few large failures?** Many small losses -- the worst 10 windows account for only {fs['top10_harmful_share_of_total_positive']*100:.1f}% of total harm.",
        f"**2. Which rank changes drive the regression?** The most drastic reorderings (best flips to C's worst-ranked expert, or other multi-position reorders) -- {100*de_windows/fs['num_windows']:.1f}% of windows but {100*de_contribution/max(total_positive_contribution,1e-8):.1f}% of total harm. Simple adjacent swaps are nearly harmless.",
        "**3. Fewer but costlier mistakes?** No -- LearnedProbe makes *more* top-1 mistakes than C on ETTm1, each on average also slightly more expensive.",
        f"**4. Is one expert over/under-promoted?** Yes -- TimesNet is severely under-promoted ({timesnet['C_ranked_1_frac']*100:.0f}%->{ timesnet['Learned_ranked_1_frac']*100:.0f}% rank-1 rate), and when it is picked, LearnedProbe's picks are worse-conditioned than C's.",
        "**5. Does harm concentrate at low expert separation?** No -- if anything the opposite: harm is largest when the best-vs-second gap is largest, i.e. when it matters most.",
        f"**6. Concentrated by variable/horizon?** No -- regret is positive on {n_positive_h}/{len(hv_h)} horizons and {n_positive_v}/{len(hv_v)} variables; broadly distributed, not localized to one cell.",
        "**7. Distinct probe-response signature on harmful windows?** No -- probe magnitude, energy location, and response statistics are nearly identical between beneficial and harmful windows.",
        "**8. Why can Spearman improve while MAE worsens?** Because Spearman/pairwise-accuracy rewards getting the *2nd vs 3rd place* comparison right, which the rank-weighting rule barely rewards (0.30 vs 0.10), while the *1st place* call -- the one that matters most for the 0.60 weight -- is where LearnedProbe is specifically worse on ETTm1.",
        "**9. General weakness or ETTm1-specific?** The mechanism (aggregate ranking metrics not tracking top-1 accuracy) is general and could recur elsewhere, but the specific manifestation here -- TimesNet under-promotion, confidently-wrong top-1 calls -- looks tied to how this dataset's three experts (DLinear/PatchTST/TimesNet) actually perform, which the successful datasets' expert sets don't share.",
        "**10. General research problem or isolated failure?** A bit of both, and worth stating precisely: the *diagnostic gap* between aggregate ranking correlation and top-1/rank-weighted forecasting quality is a real, general phenomenon that this analysis exposed cleanly -- Spearman is not a reliable proxy for how a rank-weighted ensemble will perform, because it doesn't weight the top comparison specially. But the specific *failure mode on ETTm1* (TimesNet under-promotion, larger-but-wrong confidence) has not been shown to generalize to other datasets in this analysis -- ETTh2/Weather/Electricity all show the *opposite* top-1 pattern (LearnedProbe's top-1 accuracy exceeds C's there). So: the measurement gap is general; the ETTm1 outcome itself looks dataset-specific.",
        "",
        "No fix is proposed here, per instructions -- this is diagnosis only.",
        "",
        "## Hard rule compliance",
        "",
        "```text",
        "TEST SET ACCESSED: NO",
        "TEST CACHE LOADED: NO",
        "TEST METRICS COMPUTED: NO",
        "```",
    ]
    (out_dir / "ettm1_probe_failure_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    start = time.time()
    integrity_ettm1 = integrity_check("ETTm1")
    print(f"[ettm1-failure] integrity check: {integrity_ettm1}", flush=True)

    print("[ettm1-failure] analyzing ETTm1 in depth...", flush=True)
    ettm1 = analyze_full("ETTm1")
    ettm1["integrity"] = integrity_ettm1

    analyses = {"ETTm1": ettm1}
    for ds in CROSS_DATASETS:
        print(f"[ettm1-failure] analyzing {ds} for cross-dataset comparison...", flush=True)
        analyses[ds] = analyze_full(ds)

    cross = cross_dataset_summary(analyses)

    all_per_window = []
    all_transitions = []
    all_expert = []
    all_expert_transitions = []
    all_separation = []
    all_probe = []
    all_regime = []
    all_hv = []
    for ds, a in analyses.items():
        all_per_window.extend(a["per_window_rows"])
        all_transitions.extend(a["transition_category_rows"])
        all_expert.extend(a["expert_rows"])
        all_expert_transitions.extend(a["expert_transition_rows"])
        all_separation.extend(a["separation_rows"])
        all_probe.extend(a["probe_rows"])
        all_regime.extend(a["regime_rows"])
        all_hv.extend(a["hv_rows"])

    results_json = {
        "experiment": "ettm1_probe_failure_analysis",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_sec": time.time() - start,
        "test_set_accessed": False,
        "integrity": integrity_ettm1,
        "datasets": {ds: {k: v for k, v in a.items() if k != "per_window_rows"} for ds, a in analyses.items()},
        "cross_dataset_summary": cross,
    }

    write_json(RESULTS_DIR / "ettm1_probe_failure_results.json", results_json)
    write_csv(RESULTS_DIR / "ettm1_probe_failure_per_window.csv", all_per_window)
    write_csv(RESULTS_DIR / "ettm1_probe_failure_rank_transitions.csv", all_transitions + all_expert_transitions)
    write_csv(RESULTS_DIR / "ettm1_probe_failure_experts.csv", all_expert)
    write_csv(RESULTS_DIR / "ettm1_probe_failure_horizon_variable.csv", all_hv)
    write_csv(RESULTS_DIR / "ettm1_probe_failure_regimes.csv", all_regime + all_separation + all_probe)
    write_csv(RESULTS_DIR / "ettm1_probe_failure_cross_dataset.csv", cross)

    make_report(REPORTS_DIR, ettm1, cross)

    print("TEST SET ACCESSED: NO")
    print(json.dumps({"runtime_sec": results_json["runtime_sec"]}, indent=2))


if __name__ == "__main__":
    main()
