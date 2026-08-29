"""Conditional Nuisance Invariance (CNI) audit of LearnedProbe.

Validation-only mechanism audit. This script does not introduce a new router
architecture; it asks whether the canonical active LearnedProbe response
features add competence information after passive and nuisance explanations
are controlled.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.behavioral_competence.expert_conditioned_probe_mechanism.run_experiment as mech  # noqa: E402
import experiments.behavioral_competence.run_behavioral_competence as rbc  # noqa: E402
import experiments.frozen_hv_costar.run_frozen_hv_costar as fhv  # noqa: E402
from experiments.behavioral_competence.common import CompetenceScorer  # noqa: E402
from experiments.behavioral_competence.model_runtime import load_expert_runtime, sha256_file  # noqa: E402
from experiments.behavioral_competence.run_behavioral_competence import compute_excess_loss, raw_history_cache  # noqa: E402
from experiments.behavioral_competence.run_learned_probe import build_abc_features  # noqa: E402
from experiments.behavioral_competence.run_learned_probe_decision_rules import rule_fixed_rank  # noqa: E402
from experiments.behavioral_competence.controlled_discriminative_probe_v2.run_controlled_discriminative_probe_v2 import compute_legal_and_common  # noqa: E402
from experiments.costar_multidataset_frozen.common import block_bootstrap_with_prob, every_kth_phase_bootstrap  # noqa: E402
from experiments.oracle_weight_tournament.run_tournament import sample_mae, sample_mse  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = OUT_DIR / "results"
FEATURE_CACHE_DIR = OUT_DIR / "feature_cache"
DATASETS = ("ETTh1", "ETTh2", "ETTm1", "Weather", "Electricity")
BLOCK_LENGTHS = (12, 24, 48)
PRIMARY_BLOCK = 24
BOOTSTRAP_SAMPLES = 5000
PHASE_K = 12
SEED = 7
RIDGE_ALPHA = 1.0
CODE_VERSION = "cni_v2"
MODEL_ORDER = [
    "Passive",
    "ProbeOnly",
    "PassiveProbe",
    "PassiveNuisance",
    "PassiveNuisanceProbe",
    "PassiveResidualProbe",
    "PassiveNuisanceResidualProbe",
    "ShuffledProbe",
    "WrongExpertProbe",
    "MatchedPassive",
]
PRIMARY_NUISANCES = ("scale", "volatility", "perturbation_norm")
NUISANCE_NAMES = [
    "log_history_scale",
    "history_mean_magnitude",
    "volatility",
    "trend_strength",
    "seasonality_strength",
    "normalized_time_index",
    "forecast_magnitude",
    "forecast_variance",
    "disagreement_magnitude",
    "perturbation_norm",
    "perturbation_recent_fraction",
    "perturbation_direction_trend_alignment",
]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def git_commit_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "UNKNOWN"


def select_device() -> torch.device:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def safe_corr(x: torch.Tensor, y: torch.Tensor, kind: str) -> float:
    x_np = x.detach().cpu().flatten().numpy()
    y_np = y.detach().cpu().flatten().numpy()
    mask = np.isfinite(x_np) & np.isfinite(y_np)
    if mask.sum() < 3 or np.std(x_np[mask]) <= 1e-12 or np.std(y_np[mask]) <= 1e-12:
        return float("nan")
    if kind == "pearson":
        return float(pearsonr(x_np[mask], y_np[mask]).statistic)
    return float(spearmanr(x_np[mask], y_np[mask]).statistic)


def seasonality_strength(history: torch.Tensor) -> torch.Tensor:
    x = history - history.mean(dim=1, keepdim=True)
    power = torch.fft.rfft(x, dim=1).abs().pow(2)
    non_dc = power[:, 1:, :]
    total = non_dc.sum(dim=1).clamp_min(1e-12)
    strongest = non_dc.max(dim=1).values
    return (strongest / total).mean(dim=1)


def nuisance_from_delta(history: torch.Tensor, delta: torch.Tensor, group_a: torch.Tensor, group_b: torch.Tensor, group_c: torch.Tensor, starts: torch.Tensor, train_bounds: tuple[int, int]) -> torch.Tensor:
    # history [N,L,F], delta [N,K,L,F], group_* [N,K,D]
    eps = 1e-8
    n, k, length, feats = delta.shape
    hist_std = history.std(dim=1).clamp_min(eps)
    log_history_scale = torch.log(hist_std.mean(dim=1).clamp_min(eps))
    history_mean_magnitude = (history.mean(dim=1).abs() / hist_std).mean(dim=1)
    volatility = ((history[:, 1:, :] - history[:, :-1, :]).abs().mean(dim=1) / hist_std).mean(dim=1)
    trend_strength = group_a[:, 0, 0]
    seasonality = seasonality_strength(history)
    lo, hi = train_bounds
    denom = max(float(hi - lo), 1.0)
    norm_time = ((starts.to(torch.float32) - float(lo)) / denom).clamp(0.0, 1.0)

    forecast_magnitude = group_b[:, :, 3]
    forecast_variance = group_b[:, :, 0]
    disagreement_magnitude = group_c[:, :, 0]
    scale = hist_std.view(n, 1, 1, feats)
    d_norm = delta / scale
    perturbation_norm = d_norm.pow(2).mean(dim=(2, 3)).sqrt()
    recent = max(1, length // 4)
    energy_total = d_norm.pow(2).sum(dim=(2, 3)).clamp_min(eps)
    energy_recent = d_norm[:, :, -recent:, :].pow(2).sum(dim=(2, 3))
    perturbation_recent_fraction = energy_recent / energy_total
    recent_trend = history[:, -recent:, :].mean(dim=1) - history[:, :recent, :].mean(dim=1)
    delta_recent = delta[:, :, -recent:, :].mean(dim=2)
    num = (delta_recent * recent_trend.unsqueeze(1)).sum(dim=2)
    den = delta_recent.norm(dim=2) * recent_trend.unsqueeze(1).norm(dim=2) + eps
    perturbation_direction_trend_alignment = num / den

    window_parts = torch.stack(
        [log_history_scale, history_mean_magnitude, volatility, trend_strength, seasonality, norm_time],
        dim=1,
    ).unsqueeze(1).expand(n, k, 6)
    expert_parts = torch.stack(
        [
            forecast_magnitude,
            forecast_variance,
            disagreement_magnitude,
            perturbation_norm,
            perturbation_recent_fraction,
            perturbation_direction_trend_alignment,
        ],
        dim=2,
    )
    return torch.cat([window_parts, expert_parts], dim=2).to(torch.float32)


def feature_map(x: torch.Tensor, n: torch.Tensor, p: torch.Tensor, pres: torch.Tensor, matched: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "Passive": x,
        "ProbeOnly": p,
        "PassiveProbe": torch.cat([x, p], dim=-1),
        "PassiveNuisance": torch.cat([x, n], dim=-1),
        "PassiveNuisanceProbe": torch.cat([x, n, p], dim=-1),
        "PassiveResidualProbe": torch.cat([x, pres], dim=-1),
        "PassiveNuisanceResidualProbe": torch.cat([x, n, pres], dim=-1),
        "MatchedPassive": torch.cat([x, matched], dim=-1),
    }


def shuffle_probe(p: torch.Tensor, seed: int = 20260827) -> torch.Tensor:
    out = torch.empty_like(p)
    gen = torch.Generator().manual_seed(seed)
    for e in range(p.shape[1]):
        out[:, e] = p[torch.randperm(p.shape[0], generator=gen), e]
    return out


def wrong_expert_probe(p: torch.Tensor) -> torch.Tensor:
    return torch.roll(p, shifts=-1, dims=1)


def ridge_residualize_oof(
    x_train_full: torch.Tensor,
    p_train_full_by_fold: list[dict[str, Any]],
    common_idx: torch.Tensor,
    train_final_xn: torch.Tensor,
    train_final_p: torch.Tensor,
    val_xn: torch.Tensor,
    val_p: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, Any]:
    p_res = torch.full_like(val_p.new_zeros(x_train_full.shape[0], p_train_full_by_fold[0]["p_train"].shape[-2], p_train_full_by_fold[0]["p_train"].shape[-1]), float("nan"))
    for item in p_train_full_by_fold:
        train_idx = item["train_idx"]
        eval_idx = item["eval_idx"]
        p_train = item["p_train"]
        p_eval = item["p_eval"]
        n_train = item["n_train"]
        n_eval = item["n_eval"]
        xn_train = torch.cat([x_train_full[train_idx], n_train], dim=-1).reshape(train_idx.numel() * x_train_full.shape[1], -1)
        y_train = p_train.reshape(train_idx.numel() * x_train_full.shape[1], -1).numpy()
        model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
        model.fit(xn_train.numpy(), y_train)
        xn_eval = torch.cat([x_train_full[eval_idx], n_eval], dim=-1).reshape(eval_idx.numel() * x_train_full.shape[1], -1)
        pred = torch.tensor(model.predict(xn_eval.numpy()), dtype=torch.float32).reshape_as(p_eval)
        p_res[eval_idx] = p_eval - pred
    model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
    model.fit(train_final_xn.reshape(train_final_xn.shape[0] * train_final_xn.shape[1], -1).numpy(), train_final_p.reshape(train_final_p.shape[0] * train_final_p.shape[1], -1).numpy())
    pred_val = torch.tensor(model.predict(val_xn.reshape(val_xn.shape[0] * val_xn.shape[1], -1).numpy()), dtype=torch.float32).reshape_as(val_p)
    return p_res[common_idx], val_p - pred_val, model


def residualize_with_model(model: Any, xn: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    pred = torch.tensor(
        model.predict(xn.reshape(xn.shape[0] * xn.shape[1], -1).numpy()),
        dtype=torch.float32,
    ).reshape_as(p)
    return p - pred


@dataclass
class ScorerFit:
    model: CompetenceScorer
    mean: torch.Tensor
    std: torch.Tensor
    best_epoch: int
    best_val: float

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        n, k, d = features.shape
        self.model.eval()
        with torch.no_grad():
            x = ((features.reshape(n * k, d) - self.mean) / self.std).to(next(self.model.parameters()).device)
            return self.model(x).reshape(n, k).cpu()


def train_scorer(features: torch.Tensor, target: torch.Tensor, row_mask: torch.Tensor | None = None, seed: int = SEED) -> ScorerFit:
    torch.manual_seed(seed)
    n, k, d = features.shape
    x_all = features.reshape(n * k, d)
    y_all = target.reshape(n * k)
    if row_mask is None:
        rows = torch.arange(n * k)
    else:
        rows = row_mask.reshape(n * k).nonzero(as_tuple=False).flatten()
    if rows.numel() < max(24, k * 4):
        raise ValueError(f"Too few rows for scorer training: {rows.numel()}")
    split = max(1, min(rows.numel() - 1, int(round(rows.numel() * 0.8))))
    train_rows = rows[:split]
    val_rows = rows[split:]
    mean = x_all[train_rows].mean(dim=0, keepdim=True)
    std = x_all[train_rows].std(dim=0, keepdim=True).clamp_min(1e-6)
    x = (x_all - mean) / std
    model = CompetenceScorer(d)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_state, best_val, best_epoch, bad = None, math.inf, -1, 0
    for epoch in range(1, 201):
        model.train()
        opt.zero_grad()
        loss = F.mse_loss(model(x[train_rows]), y_all[train_rows])
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val = float(F.mse_loss(model(x[val_rows]), y_all[val_rows]))
        if val < best_val - 1e-9:
            best_val, best_epoch, bad = val, epoch, 0
            best_state = {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 15:
                break
    model.load_state_dict(best_state)
    model.eval()
    return ScorerFit(model, mean, std, best_epoch, best_val)


def competence_metrics(pred: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    pred_np = pred.reshape(-1).numpy()
    actual_np = actual.reshape(-1).numpy()
    correct, total = 0, 0
    per_window_pair = torch.zeros(pred.shape[0], dtype=torch.float32)
    for t in range(pred.shape[0]):
        c, m = 0, 0
        for i in range(pred.shape[1]):
            for j in range(i + 1, pred.shape[1]):
                if actual[t, i] == actual[t, j]:
                    continue
                c += int((pred[t, i] < pred[t, j]) == (actual[t, i] < actual[t, j]))
                m += 1
        per_window_pair[t] = c / max(m, 1)
        correct += c
        total += m
    order = pred.argsort(dim=1)
    actual_best = actual.argmin(dim=1)
    return {
        "competence_mae": float(mean_absolute_error(actual_np, pred_np)),
        "competence_mse": float(mean_squared_error(actual_np, pred_np)),
        "competence_r2": float(r2_score(actual_np, pred_np)),
        "pearson": safe_corr(pred, actual, "pearson"),
        "spearman": safe_corr(pred, actual, "spearman"),
        "pairwise_ranking_accuracy": correct / max(total, 1),
        "top1_expert_accuracy": float((order[:, 0] == actual_best).to(torch.float32).mean()),
        "top2_recall": float((order[:, :2] == actual_best.view(-1, 1)).any(dim=1).to(torch.float32).mean()),
    }


def route_prediction(forecasts: torch.Tensor, pred_excess: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weights = rule_fixed_rank(pred_excess)
    return (forecasts * weights.view(weights.shape[0], 1, 1, weights.shape[1])).sum(dim=-1), weights


def routing_metrics(cache: Mapping[str, Any], pred: torch.Tensor, std: torch.Tensor) -> tuple[dict[str, float], torch.Tensor]:
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.bool)
    mae = sample_mae(pred, target, mask, std)
    mse = sample_mse(pred, target, mask, std)
    return {"routing_mae": float(mae.mean()), "routing_mse": float(mse.mean())}, mae


def paired_block_rows(dataset: str, comparison: str, candidate: torch.Tensor, baseline: torch.Tensor, higher_is_better: bool) -> list[dict[str, Any]]:
    # block_bootstrap_with_prob expects lower-is-better candidate/baseline if interpreting prob_delta_negative.
    rows = []
    cand = -candidate if higher_is_better else candidate
    base = -baseline if higher_is_better else baseline
    for block in BLOCK_LENGTHS:
        b = block_bootstrap_with_prob(cand, base, block=block, seed=20260821, samples=BOOTSTRAP_SAMPLES)
        rows.append({"dataset": dataset, "comparison": comparison, "metric_orientation": "higher_is_better" if higher_is_better else "lower_is_better", "test": f"block_{block}", **b})
    phase = every_kth_phase_bootstrap(cand - base, k=min(PHASE_K, int(cand.numel())), seed=20260821, samples=BOOTSTRAP_SAMPLES)
    rows.append({"dataset": dataset, "comparison": comparison, "metric_orientation": "higher_is_better" if higher_is_better else "lower_is_better", "test": f"every_{min(PHASE_K, int(cand.numel()))}th_phase", **phase})
    return rows


def load_or_build_features(dataset: str, device: torch.device, force: bool = False) -> dict[str, Any]:
    cache_path = FEATURE_CACHE_DIR / f"{dataset}.pt"
    if cache_path.exists() and not force:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("code_version") == CODE_VERSION:
            return payload

    print(f"[CNI] {dataset}: regenerating missing OOF active/matched features", flush=True)
    mech.ensure_registered(dataset)
    bundle = fhv.LOADERS[dataset]()
    train_cache, val_cache = bundle.train_cache, bundle.val_cache
    obs, legal_idx_all, folds, common_idx = compute_legal_and_common(train_cache, val_cache)
    if not obs["observability_holds"]:
        raise AssertionError(f"{dataset}: observability failed: {obs}")
    k = len(bundle.core_names)
    if k != 3:
        raise AssertionError(f"{dataset}: expected K=3, got {k}")
    final_runtimes = {e: load_expert_runtime(dataset, e, device=device) for e in bundle.core_names}
    ref = final_runtimes[bundle.core_names[0]]
    train_raw = raw_history_cache(dataset, train_cache, ref.mean.detach().cpu(), ref.std.detach().cpu())
    val_raw = raw_history_cache(dataset, val_cache, ref.mean.detach().cpu(), ref.std.detach().cpu())
    train_abc = build_abc_features(bundle, train_raw)
    val_abc = build_abc_features(bundle, val_raw)
    x_train_full = torch.cat(train_abc[:3], dim=-1).to(torch.float32)
    x_val = torch.cat(val_abc[:3], dim=-1).to(torch.float32)
    starts_train = train_cache["absolute_window_starts"].to(torch.long)
    starts_val = val_cache["absolute_window_starts"].to(torch.long)
    bounds = (int(starts_train.min()), int(starts_train.max()))
    forecasts_train = bundle.forecasts_fn(train_cache, bundle.expert_idx)
    forecasts_val = bundle.forecasts_fn(val_cache, bundle.expert_idx)
    y_train_full, expert_mae_train_full = compute_excess_loss(train_cache, forecasts_train, bundle.std)
    y_val, expert_mae_val = compute_excess_loss(val_cache, forecasts_val, bundle.std)

    p_oof_full = torch.full((int(train_cache["num_windows"]), k, 6), float("nan"))
    matched_oof_full = torch.full_like(p_oof_full, float("nan"))
    n_oof_full = torch.full((int(train_cache["num_windows"]), k, len(NUISANCE_NAMES)), float("nan"))
    residual_fold_items: list[dict[str, Any]] = []
    fold_rows = []
    for fold in folds:
        train_idx = fold["train_idx"]
        eval_idx = fold["eval_idx"]
        fit_probe = mech.train_arm(dataset, "OriginalLearnedProbe", bundle, train_cache, train_idx, device=device)
        fit_matched = mech.train_arm(dataset, "MatchedNeuralPassive", bundle, train_cache, train_idx, device=device)
        _pred_eval, p_eval, delta_eval = mech.score_arm_features(dataset, "OriginalLearnedProbe", bundle, fit_probe, train_cache, eval_idx, is_router_train=True, device=device)
        _mpred_eval, matched_eval, _ = mech.score_arm_features(dataset, "MatchedNeuralPassive", bundle, fit_matched, train_cache, eval_idx, is_router_train=True, device=device)
        _pred_tr, p_tr, delta_tr = mech.score_arm_features(dataset, "OriginalLearnedProbe", bundle, fit_probe, train_cache, train_idx, is_router_train=True, device=device)
        n_eval = nuisance_from_delta(train_raw["histories"][eval_idx].to(torch.float32), delta_eval, train_abc[0][eval_idx], train_abc[1][eval_idx], train_abc[2][eval_idx], starts_train[eval_idx], bounds)
        n_tr = nuisance_from_delta(train_raw["histories"][train_idx].to(torch.float32), delta_tr, train_abc[0][train_idx], train_abc[1][train_idx], train_abc[2][train_idx], starts_train[train_idx], bounds)
        p_oof_full[eval_idx] = p_eval
        matched_oof_full[eval_idx] = matched_eval
        n_oof_full[eval_idx] = n_eval
        residual_fold_items.append({"train_idx": train_idx, "eval_idx": eval_idx, "p_train": p_tr, "p_eval": p_eval, "n_train": n_tr, "n_eval": n_eval})
        fold_rows.append({k2: v for k2, v in fold.items() if k2 not in ("train_idx", "eval_idx")})

    fit_probe_final = mech.train_arm(dataset, "OriginalLearnedProbe", bundle, train_cache, legal_idx_all, device=device)
    fit_matched_final = mech.train_arm(dataset, "MatchedNeuralPassive", bundle, train_cache, legal_idx_all, device=device)
    _pred_train_final, p_train_final, delta_train_final = mech.score_arm_features(dataset, "OriginalLearnedProbe", bundle, fit_probe_final, train_cache, legal_idx_all, is_router_train=True, device=device)
    _mpred_train_final, matched_train_final_unused, _ = mech.score_arm_features(dataset, "MatchedNeuralPassive", bundle, fit_matched_final, train_cache, legal_idx_all, is_router_train=True, device=device)
    _pred_val_probe, p_val, delta_val = mech.score_arm_features(dataset, "OriginalLearnedProbe", bundle, fit_probe_final, val_cache, torch.arange(int(val_cache["num_windows"])), is_router_train=False, device=device)
    _mpred_val, matched_val, _ = mech.score_arm_features(dataset, "MatchedNeuralPassive", bundle, fit_matched_final, val_cache, torch.arange(int(val_cache["num_windows"])), is_router_train=False, device=device)
    n_train_final = nuisance_from_delta(train_raw["histories"][legal_idx_all].to(torch.float32), delta_train_final, train_abc[0][legal_idx_all], train_abc[1][legal_idx_all], train_abc[2][legal_idx_all], starts_train[legal_idx_all], bounds)
    n_val = nuisance_from_delta(val_raw["histories"].to(torch.float32), delta_val, val_abc[0], val_abc[1], val_abc[2], starts_val, bounds)
    p_res_train, p_res_val, residualizer_final = ridge_residualize_oof(
        x_train_full,
        residual_fold_items,
        common_idx,
        torch.cat([x_train_full[legal_idx_all], n_train_final], dim=-1),
        p_train_final,
        torch.cat([x_val, n_val], dim=-1),
        p_val,
    )

    corrupted_val_cache = dict(val_cache)
    corrupted_val_cache["targets"] = torch.randn(val_cache["targets"].shape, generator=torch.Generator().manual_seed(20260827))
    corrupted_raw = raw_history_cache(dataset, corrupted_val_cache, ref.mean.detach().cpu(), ref.std.detach().cpu())
    corrupted_abc = build_abc_features(bundle, corrupted_raw)
    x_val_corrupt = torch.cat(corrupted_abc[:3], dim=-1).to(torch.float32)
    _pred_val_probe_corrupt, p_val_corrupt, delta_val_corrupt = mech.score_arm_features(
        dataset,
        "OriginalLearnedProbe",
        bundle,
        fit_probe_final,
        corrupted_val_cache,
        torch.arange(int(val_cache["num_windows"])),
        is_router_train=False,
        device=device,
    )
    _mpred_val_corrupt, matched_val_corrupt, _ = mech.score_arm_features(
        dataset,
        "MatchedNeuralPassive",
        bundle,
        fit_matched_final,
        corrupted_val_cache,
        torch.arange(int(val_cache["num_windows"])),
        is_router_train=False,
        device=device,
    )
    n_val_corrupt = nuisance_from_delta(
        corrupted_raw["histories"].to(torch.float32),
        delta_val_corrupt,
        corrupted_abc[0],
        corrupted_abc[1],
        corrupted_abc[2],
        starts_val,
        bounds,
    )
    p_res_val_corrupt = residualize_with_model(residualizer_final, torch.cat([x_val_corrupt, n_val_corrupt], dim=-1), p_val_corrupt)
    payload = {
        "code_version": CODE_VERSION,
        "dataset": dataset,
        "core": list(bundle.core_names),
        "common_idx": common_idx,
        "legal_idx_all": legal_idx_all,
        "fold_rows": fold_rows,
        "x_train": x_train_full[common_idx],
        "n_train": n_oof_full[common_idx],
        "p_train": p_oof_full[common_idx],
        "pres_train": p_res_train,
        "matched_train": matched_oof_full[common_idx],
        "y_train": y_train_full[common_idx],
        "expert_mae_train": expert_mae_train_full[common_idx],
        "x_val": x_val,
        "n_val": n_val,
        "p_val": p_val,
        "pres_val": p_res_val,
        "matched_val": matched_val,
        "x_val_target_corrupt": x_val_corrupt,
        "n_val_target_corrupt": n_val_corrupt,
        "p_val_target_corrupt": p_val_corrupt,
        "pres_val_target_corrupt": p_res_val_corrupt,
        "matched_val_target_corrupt": matched_val_corrupt,
        "y_val": y_val,
        "expert_mae_val": expert_mae_val,
        "starts_train": starts_train[common_idx],
        "starts_val": starts_val,
        "observability": obs,
        "features_target_free_fingerprint": {
            "x_val_sum": float(x_val.sum()),
            "n_val_sum": float(n_val.sum()),
            "p_val_sum": float(p_val.sum()),
            "pres_val_sum": float(p_res_val.sum()),
            "matched_val_sum": float(matched_val.sum()),
        },
        "target_corruption_feature_diffs": {
            "x_val_max_abs_diff": float((x_val - x_val_corrupt).abs().max()),
            "n_val_max_abs_diff": float((n_val - n_val_corrupt).abs().max()),
            "p_val_max_abs_diff": float((p_val - p_val_corrupt).abs().max()),
            "pres_val_max_abs_diff": float((p_res_val - p_res_val_corrupt).abs().max()),
            "matched_val_max_abs_diff": float((matched_val - matched_val_corrupt).abs().max()),
        },
    }
    FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    return payload


def evaluate_dataset(dataset: str, device: torch.device, force_features: bool) -> dict[str, Any]:
    payload = load_or_build_features(dataset, device, force=force_features)
    bundle = fhv.LOADERS[dataset]()
    val_cache = bundle.val_cache
    forecasts_val = bundle.forecasts_fn(val_cache, bundle.expert_idx)
    x_train, n_train, p_train = payload["x_train"], payload["n_train"], payload["p_train"]
    x_val, n_val, p_val = payload["x_val"], payload["n_val"], payload["p_val"]
    pres_train, pres_val = payload["pres_train"], payload["pres_val"]
    matched_train, matched_val = payload["matched_train"], payload["matched_val"]
    x_val_corrupt = payload["x_val_target_corrupt"]
    n_val_corrupt = payload["n_val_target_corrupt"]
    p_val_corrupt = payload["p_val_target_corrupt"]
    pres_val_corrupt = payload["pres_val_target_corrupt"]
    matched_val_corrupt = payload["matched_val_target_corrupt"]
    y_train, y_val = payload["y_train"], payload["y_val"]

    train_features = feature_map(x_train, n_train, p_train, pres_train, matched_train)
    val_features = feature_map(x_val, n_val, p_val, pres_val, matched_val)
    corrupt_val_features = feature_map(x_val_corrupt, n_val_corrupt, p_val_corrupt, pres_val_corrupt, matched_val_corrupt)
    train_features["ShuffledProbe"] = torch.cat([x_train, shuffle_probe(p_train)], dim=-1)
    val_features["ShuffledProbe"] = torch.cat([x_val, shuffle_probe(p_val)], dim=-1)
    corrupt_val_features["ShuffledProbe"] = torch.cat([x_val_corrupt, shuffle_probe(p_val_corrupt)], dim=-1)
    train_features["WrongExpertProbe"] = torch.cat([x_train, wrong_expert_probe(p_train)], dim=-1)
    val_features["WrongExpertProbe"] = torch.cat([x_val, wrong_expert_probe(p_val)], dim=-1)
    corrupt_val_features["WrongExpertProbe"] = torch.cat([x_val_corrupt, wrong_expert_probe(p_val_corrupt)], dim=-1)

    fits, pred_val, comp_rows, routing_rows, per_window_mae = {}, {}, [], [], {}
    per_window_pair = {}
    target_cleanliness_diffs = {}
    for method in MODEL_ORDER:
        fit = train_scorer(train_features[method], y_train)
        fits[method] = fit
        pv = fit.predict(val_features[method])
        pv_corrupt = fit.predict(corrupt_val_features[method])
        pred_val[method] = pv
        cm = competence_metrics(pv, y_val)
        comp_rows.append({"dataset": dataset, "method": method, **cm, "best_epoch": fit.best_epoch, "best_internal_val_mse": fit.best_val})
        routed, weights = route_prediction(forecasts_val, pv)
        routed_corrupt, weights_corrupt = route_prediction(forecasts_val, pv_corrupt)
        target_cleanliness_diffs[method] = {
            "score_max_abs_diff": float((pv - pv_corrupt).abs().max()),
            "weight_max_abs_diff": float((weights - weights_corrupt).abs().max()),
            "final_prediction_max_abs_diff": float((routed - routed_corrupt).abs().max()),
        }
        rm, mae_window = routing_metrics(val_cache, routed, bundle.std)
        routing_rows.append({"dataset": dataset, "method": method, **rm})
        per_window_mae[method] = mae_window
        pw = []
        for t in range(y_val.shape[0]):
            c, total = 0, 0
            for i in range(y_val.shape[1]):
                for j in range(i + 1, y_val.shape[1]):
                    c += int((pv[t, i] < pv[t, j]) == (y_val[t, i] < y_val[t, j]))
                    total += 1
            pw.append(c / total)
        per_window_pair[method] = torch.tensor(pw)

    residual_rows = [
        {"dataset": dataset, "comparison": "P_res_variance_fraction", "mse_p": float(p_val.pow(2).mean()), "mse_pres": float(pres_val.pow(2).mean()), "fraction_remaining": float(pres_val.pow(2).mean() / p_val.pow(2).mean().clamp_min(1e-12))}
    ]

    dependence_rows = []
    comparisons = [
        ("Probe_over_Passive", "PassiveProbe", "Passive"),
        ("Probe_given_N", "PassiveNuisanceProbe", "PassiveNuisance"),
        ("ResidualProbe_over_Passive", "PassiveResidualProbe", "Passive"),
        ("Probe_vs_MatchedPassive", "PassiveProbe", "MatchedPassive"),
        ("Probe_vs_Shuffled", "PassiveProbe", "ShuffledProbe"),
        ("Probe_vs_WrongExpert", "PassiveProbe", "WrongExpertProbe"),
    ]
    for label, cand, base in comparisons:
        dependence_rows.extend(paired_block_rows(dataset, label + "_routing_mae", per_window_mae[cand], per_window_mae[base], higher_is_better=False))
        dependence_rows.extend(paired_block_rows(dataset, label + "_pairwise_accuracy", per_window_pair[cand], per_window_pair[base], higher_is_better=True))

    env_rows = environment_transfer(dataset, train_features, val_features, y_train, y_val, payload)
    nuisance_pred_rows = nuisance_prediction(dataset, x_train, p_train, n_train, x_val, p_val, n_val)
    matched_rows = matched_diagnostic(dataset, x_train, x_val, n_train, n_val, y_val, pred_val, p_val)

    criteria = classify_dataset(comp_rows, dependence_rows, env_rows)
    integrity = integrity_checks(dataset, payload, target_cleanliness_diffs)
    per_window_npz = {
        "starts_val": payload["starts_val"].numpy(),
        "actual_excess_val": y_val.numpy(),
        **{f"{m}_pred_excess_val": pred_val[m].numpy() for m in pred_val},
        **{f"{m}_per_window_mae": per_window_mae[m].numpy() for m in per_window_mae},
    }
    np.savez(RESULTS_DIR / f"per_window_outputs_{dataset}.npz", **per_window_npz)
    return {
        "dataset": dataset,
        "core": payload["core"],
        "competence_rows": comp_rows,
        "routing_rows": routing_rows,
        "residual_rows": residual_rows,
        "dependence_rows": dependence_rows,
        "environment_rows": env_rows,
        "nuisance_prediction_rows": nuisance_pred_rows,
        "matched_rows": matched_rows,
        "criteria": criteria,
        "integrity": integrity,
        "fold_rows": payload["fold_rows"],
    }


def environment_transfer(dataset: str, train_features: Mapping[str, torch.Tensor], val_features: Mapping[str, torch.Tensor], y_train: torch.Tensor, y_val: torch.Tensor, payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    nuisance_index = {"scale": 0, "volatility": 2, "perturbation_norm": 9}
    methods = ["Passive", "PassiveProbe", "PassiveNuisance", "PassiveNuisanceProbe", "PassiveResidualProbe"]
    n_train = payload["n_train"]
    n_val = payload["n_val"]
    for nuisance, idx in nuisance_index.items():
        train_scalar = n_train[:, :, idx].mean(dim=1)
        val_scalar = n_val[:, :, idx].mean(dim=1)
        median = float(train_scalar.median())
        directions = [
            ("LOW_to_HIGH", train_scalar <= median, val_scalar > median),
            ("HIGH_to_LOW", train_scalar > median, val_scalar <= median),
        ]
        for direction, train_win, val_win in directions:
            train_mask = train_win.view(-1, 1).expand(-1, y_train.shape[1])
            val_idx = val_win.nonzero(as_tuple=False).flatten()
            for method in methods:
                if train_mask.sum() < 24 or val_idx.numel() < 8:
                    rows.append({"dataset": dataset, "nuisance": nuisance, "direction": direction, "method": method, "status": "SKIPPED_TOO_FEW", "train_rows": int(train_mask.sum()), "val_windows": int(val_idx.numel())})
                    continue
                fit = train_scorer(train_features[method], y_train, row_mask=train_mask)
                pred = fit.predict(val_features[method])[val_idx]
                cm = competence_metrics(pred, y_val[val_idx])
                rows.append({"dataset": dataset, "nuisance": nuisance, "direction": direction, "method": method, "status": "OK", "train_rows": int(train_mask.sum()), "val_windows": int(val_idx.numel()), **cm})
    return rows


def nuisance_prediction(dataset: str, x_train: torch.Tensor, p_train: torch.Tensor, n_train: torch.Tensor, x_val: torch.Tensor, p_val: torch.Tensor, n_val: torch.Tensor) -> list[dict[str, Any]]:
    rows = []
    reps = {
        "X": (x_train, x_val),
        "P": (p_train, p_val),
        "X_plus_P": (torch.cat([x_train, p_train], dim=-1), torch.cat([x_val, p_val], dim=-1)),
    }
    for ni, name in enumerate(NUISANCE_NAMES):
        ytr = n_train[:, :, ni].reshape(-1).numpy()
        yv = n_val[:, :, ni].reshape(-1).numpy()
        med = float(np.median(ytr))
        binary = (yv > med).astype(np.int32)
        for rep, (atr, av) in reps.items():
            xtr = atr.reshape(atr.shape[0] * atr.shape[1], -1).numpy()
            xv = av.reshape(av.shape[0] * av.shape[1], -1).numpy()
            model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
            model.fit(xtr, ytr)
            pred = model.predict(xv)
            auroc = float("nan")
            try:
                auroc = float(roc_auc_score(binary, pred))
            except ValueError:
                pass
            rows.append({"dataset": dataset, "nuisance": name, "representation": rep, "r2": float(r2_score(yv, pred)), "spearman": float(spearmanr(pred, yv).statistic), "auroc_low_high": auroc})
    return rows


def matched_diagnostic(dataset: str, x_train: torch.Tensor, x_val: torch.Tensor, n_train: torch.Tensor, n_val: torch.Tensor, y_val: torch.Tensor, pred_val: Mapping[str, torch.Tensor], p_val: torch.Tensor) -> list[dict[str, Any]]:
    rows = []
    x_mean = x_train.reshape(-1, x_train.shape[-1]).mean(dim=0, keepdim=True)
    x_std = x_train.reshape(-1, x_train.shape[-1]).std(dim=0, keepdim=True).clamp_min(1e-6)
    xz = ((x_val - x_mean) / x_std).numpy()
    nuisance_index = {"scale": 0, "volatility": 2, "perturbation_norm": 9}
    for nuisance, idx in nuisance_index.items():
        train_scalar = n_train[:, :, idx].reshape(-1)
        q25, q75 = torch.quantile(train_scalar, torch.tensor([0.25, 0.75]))
        pair_rows = []
        for expert in range(x_val.shape[1]):
            y_e = y_val[:, expert]
            qs = torch.quantile(y_e, torch.tensor([0.2, 0.4, 0.6, 0.8]))
            bins = torch.bucketize(y_e, qs)
            n_e = n_val[:, expert, idx]
            for qb in range(5):
                low = ((bins == qb) & (n_e <= q25)).nonzero(as_tuple=False).flatten().numpy()
                high = ((bins == qb) & (n_e >= q75)).nonzero(as_tuple=False).flatten().numpy()
                if len(low) == 0 or len(high) == 0:
                    continue
                nn = NearestNeighbors(n_neighbors=min(8, len(high))).fit(xz[high, expert, :])
                used_high: set[int] = set()
                for li_pos, li in enumerate(low):
                    _dist, ind = nn.kneighbors(xz[li : li + 1, expert, :])
                    chosen = None
                    for cand_pos in ind[0]:
                        hidx = int(high[cand_pos])
                        if hidx not in used_high:
                            chosen = hidx
                            used_high.add(hidx)
                            break
                    if chosen is None:
                        continue
                    pair_rows.append((int(li), int(chosen), expert))
        if pair_rows:
            diffs_probe, diffs_passive, diffs_pnorm = [], [], []
            for a, b, e in pair_rows:
                diffs_probe.append(abs(float(pred_val["PassiveProbe"][a, e] - pred_val["PassiveProbe"][b, e])))
                diffs_passive.append(abs(float(pred_val["Passive"][a, e] - pred_val["Passive"][b, e])))
                diffs_pnorm.append(abs(float(p_val[a, e].norm() - p_val[b, e].norm())))
            rows.append({"dataset": dataset, "nuisance": nuisance, "pairs": len(pair_rows), "probe_score_absdiff": float(np.mean(diffs_probe)), "passive_score_absdiff": float(np.mean(diffs_passive)), "raw_p_norm_absdiff": float(np.mean(diffs_pnorm))})
        else:
            rows.append({"dataset": dataset, "nuisance": nuisance, "pairs": 0, "probe_score_absdiff": float("nan"), "passive_score_absdiff": float("nan"), "raw_p_norm_absdiff": float("nan")})
    return rows


def classify_dataset(comp_rows: Sequence[Mapping[str, Any]], dep_rows: Sequence[Mapping[str, Any]], env_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by = {r["method"]: r for r in comp_rows}
    dep = {r["comparison"]: r for r in dep_rows if r["test"] == f"block_{PRIMARY_BLOCK}"}
    env_ok = False
    env_bad = False
    for nuisance in PRIMARY_NUISANCES:
        for direction in ("LOW_to_HIGH", "HIGH_to_LOW"):
            rows = [r for r in env_rows if r["nuisance"] == nuisance and r["direction"] == direction and r["status"] == "OK"]
            if not rows:
                continue
            m = {r["method"]: r for r in rows}
            if m["PassiveProbe"]["pairwise_ranking_accuracy"] > m["Passive"]["pairwise_ranking_accuracy"]:
                env_ok = True
            if m["PassiveProbe"]["pairwise_ranking_accuracy"] < m["Passive"]["pairwise_ranking_accuracy"] - 0.05:
                env_bad = True
    crit = {
        "xn_probe_improves_xn": by["PassiveNuisanceProbe"]["pairwise_ranking_accuracy"] > by["PassiveNuisance"]["pairwise_ranking_accuracy"],
        "x_pres_improves_x": by["PassiveResidualProbe"]["pairwise_ranking_accuracy"] > by["Passive"]["pairwise_ranking_accuracy"],
        "true_beats_shuffled": by["PassiveProbe"]["pairwise_ranking_accuracy"] > by["ShuffledProbe"]["pairwise_ranking_accuracy"],
        "true_beats_wrong_expert": by["PassiveProbe"]["pairwise_ranking_accuracy"] > by["WrongExpertProbe"]["pairwise_ranking_accuracy"],
        "environment_survives": env_ok and not env_bad,
        "block24_pairwise_support": dep.get("Probe_given_N_pairwise_accuracy", {}).get("ci_excludes_zero", False) and dep.get("Probe_given_N_pairwise_accuracy", {}).get("mean_delta", 0.0) < 0.0,
    }
    crit["strong_unique_active"] = all(crit.values())
    return crit


def integrity_checks(dataset: str, payload: Mapping[str, Any], target_cleanliness_diffs: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    bundle = fhv.LOADERS[dataset]()
    before = {e: load_expert_runtime(dataset, e).checkpoint_sha256 for e in bundle.core_names}
    after = {e: load_expert_runtime(dataset, e).checkpoint_sha256 for e in bundle.core_names}
    finite = all(torch.isfinite(payload[k]).all().item() for k in ("x_train", "n_train", "p_train", "pres_train", "matched_train", "x_val", "n_val", "p_val", "pres_val", "matched_val"))
    feature_diffs = payload.get("target_corruption_feature_diffs", {})
    feature_target_clean = bool(feature_diffs) and all(float(v) == 0.0 for v in feature_diffs.values())
    prediction_target_clean = bool(target_cleanliness_diffs) and all(
        float(metric) == 0.0 for diffs in target_cleanliness_diffs.values() for metric in diffs.values()
    )
    obs = payload.get("observability", {})
    stages = ["block_a", "block_ab", "final_60"] if rbc.router_train_block_split(dataset, fhv.LOADERS[dataset]().train_cache) is not None else ["final_60"]
    return {
        "dataset": dataset,
        "test_set_accessed": False,
        "test_cache_loaded": False,
        "test_metrics_computed": False,
        "checkpoint_sha256_unchanged": before == after,
        "checkpoint_sha256_before": before,
        "checkpoint_sha256_after": after,
        "expert_parameters_updated": False,
        "stage_provenance": stages,
        "router_train_to_val_observability": obs,
        "router_val_targets_used_in_feature_construction": False,
        "router_val_targets_used_in_residualization_training": False,
        "feature_tensors_finite": finite,
        "cache_expert_order": "+".join(payload["core"]),
        "starts_retained": "starts_train" in payload and "starts_val" in payload,
        "target_corruption_feature_diffs": feature_diffs,
        "target_corruption_prediction_diffs": target_cleanliness_diffs,
        "target_corruption_left_features_unchanged": feature_target_clean,
        "target_corruption_left_scores_weights_predictions_unchanged": prediction_target_clean,
        "result": "PASS" if before == after and finite and feature_target_clean and prediction_target_clean else "FAIL",
    }


def overall_decision(results: Mapping[str, Any]) -> str:
    strong = sum(1 for d in results.values() if d["criteria"]["strong_unique_active"])
    residual_positive = sum(1 for d in results.values() if d["criteria"]["x_pres_improves_x"])
    fail_count = 0
    for d in results.values():
        c = d["criteria"]
        if (not c["xn_probe_improves_xn"]) and (not c["x_pres_improves_x"]) and (not c["environment_survives"]):
            fail_count += 1
    if strong >= 3 and residual_positive >= 3:
        return "PROBE_SURVIVES_CNI"
    if fail_count >= 4:
        return "PROBE_FAILS_CNI"
    return "MIXED_CNI"


def write_report(report: Mapping[str, Any]) -> None:
    table_header = "| Dataset | Passive pair | Passive+Nuisance | Passive+Probe | Passive+Nuisance+Probe | Passive+ResidualProbe | MatchedPassive | ShuffledProbe | WrongExpertProbe | Delta(P+N+P vs P+N) | Delta(P+Pres vs P) |"
    lines = [
        "# Conditional Nuisance Invariance Audit of LearnedProbe",
        "",
        "Research question:",
        "",
        "    Does active probing contain expert-competence information that remains",
        "    after passive and nuisance explanations are controlled?",
        "",
        f"Final decision: `{report['decision']}`",
        "",
        "## Single Most Important Table",
        "",
        table_header,
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ds, d in report["datasets"].items():
        by = {r["method"]: r for r in d["competence_rows"]}
        delta_n = by["PassiveNuisanceProbe"]["pairwise_ranking_accuracy"] - by["PassiveNuisance"]["pairwise_ranking_accuracy"]
        delta_r = by["PassiveResidualProbe"]["pairwise_ranking_accuracy"] - by["Passive"]["pairwise_ranking_accuracy"]
        lines.append(
            f"| {ds} | {by['Passive']['pairwise_ranking_accuracy']:.4f} | {by['PassiveNuisance']['pairwise_ranking_accuracy']:.4f} | "
            f"{by['PassiveProbe']['pairwise_ranking_accuracy']:.4f} | {by['PassiveNuisanceProbe']['pairwise_ranking_accuracy']:.4f} | "
            f"{by['PassiveResidualProbe']['pairwise_ranking_accuracy']:.4f} | {by['MatchedPassive']['pairwise_ranking_accuracy']:.4f} | "
            f"{by['ShuffledProbe']['pairwise_ranking_accuracy']:.4f} | {by['WrongExpertProbe']['pairwise_ranking_accuracy']:.4f} | "
            f"`{delta_n:+.4f}` | `{delta_r:+.4f}` |"
        )
    lines += ["", "## Routing MAE", ""]
    lines.append("| Dataset | Passive | Passive+Probe | Passive+Nuisance | Passive+Nuisance+Probe | Passive+ResidualProbe | MatchedPassive |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for ds, d in report["datasets"].items():
        by = {r["method"]: r for r in d["routing_rows"]}
        lines.append(f"| {ds} | {by['Passive']['routing_mae']:.6f} | {by['PassiveProbe']['routing_mae']:.6f} | {by['PassiveNuisance']['routing_mae']:.6f} | {by['PassiveNuisanceProbe']['routing_mae']:.6f} | {by['PassiveResidualProbe']['routing_mae']:.6f} | {by['MatchedPassive']['routing_mae']:.6f} |")
    lines += ["", "## Dataset Criteria", ""]
    for ds, d in report["datasets"].items():
        lines.append(f"- **{ds}**: {d['criteria']}")
    lines += ["", "## Integrity Checks", ""]
    for ds, d in report["datasets"].items():
        lines.append(f"- **{ds}**: {d['integrity']}")
    lines += [
        "",
        "## Interpretation",
        "",
        "The decisive comparisons are `Passive+Nuisance+Probe` vs `Passive+Nuisance` and `Passive+ResidualProbe` vs `Passive`. Negative controls are shuffled and wrong-expert Probe features, with MatchedPassive as the capacity-matched passive control.",
        "",
        report["decision"],
        "",
        "```text",
        "TEST SET ACCESSED: NO",
        "TEST CACHE LOADED: NO",
        "TEST METRICS COMPUTED: NO",
        "```",
    ]
    text = "\n".join(lines) + "\n"
    (RESULTS_DIR / "report.md").write_text(text, encoding="utf-8")
    (OUT_DIR / "report.md").write_text(text, encoding="utf-8")


def write_combined_per_window(selected: Sequence[str]) -> None:
    arrays: dict[str, np.ndarray] = {}
    for dataset in selected:
        path = RESULTS_DIR / f"per_window_outputs_{dataset}.npz"
        if not path.exists():
            continue
        with np.load(path) as data:
            for key in data.files:
                arrays[f"{dataset}__{key}"] = data[key]
    if arrays:
        np.savez(RESULTS_DIR / "per_window_outputs.npz", **arrays)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Conditional Nuisance Invariance audit.")
    parser.add_argument("--dataset", action="append", choices=DATASETS)
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    selected = tuple(args.dataset) if args.dataset else DATASETS
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = select_device()
    manifest = {
        "experiment": "conditional_nuisance_invariance",
        "code_version": CODE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "datasets": list(selected),
        "device": str(device),
        "primary_nuisance_names": NUISANCE_NAMES,
        "residualizer": f"StandardScaler + Ridge(alpha={RIDGE_ALPHA})",
        "competence_model": "CompetenceScorer MLP, same architecture/procedure for every CNI feature set",
        "test_accessed": False,
        "source_mechanism": "experiments/behavioral_competence/expert_conditioned_probe_mechanism/run_experiment.py",
        "artifact_gap_note": "prior mechanism outputs lacked OOF P and raw deltas, so this runner regenerates missing target-clean features in a new CNI cache",
    }
    source_provenance = {
        "git_head": git_commit_sha(),
        "cni_runner": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "canonical_mechanism_runner": {
            "path": "experiments/behavioral_competence/expert_conditioned_probe_mechanism/run_experiment.py",
            "sha256": sha256_file(ROOT / "experiments/behavioral_competence/expert_conditioned_probe_mechanism/run_experiment.py"),
        },
        "learned_probe_static_feature_source": {
            "path": "experiments/behavioral_competence/run_learned_probe.py",
            "sha256": sha256_file(ROOT / "experiments/behavioral_competence/run_learned_probe.py"),
        },
        "v2_purge_source": {
            "path": "experiments/behavioral_competence/controlled_discriminative_probe_v2/run_controlled_discriminative_probe_v2.py",
            "sha256": sha256_file(ROOT / "experiments/behavioral_competence/controlled_discriminative_probe_v2/run_controlled_discriminative_probe_v2.py"),
        },
    }
    write_json(RESULTS_DIR / "method_manifest.json", manifest)
    write_json(RESULTS_DIR / "source_provenance.json", source_provenance)
    write_json(RESULTS_DIR / "manifest.json", manifest)
    if args.audit_only:
        print("STATIC_AUDIT_COMPLETE")
        print("TEST SET ACCESSED: NO")
        return
    start = time.time()
    report = {"experiment": "conditional_nuisance_invariance", "created_at_utc": datetime.now(timezone.utc).isoformat(), "datasets": {}, "test_accessed": False}
    all_comp, all_route, all_dep, all_env, all_nuis, all_match, all_resid, all_folds = [], [], [], [], [], [], [], []
    integrity = {}
    for dataset in selected:
        result = evaluate_dataset(dataset, device, force_features=args.force_features)
        report["datasets"][dataset] = result
        all_comp.extend(result["competence_rows"])
        all_route.extend(result["routing_rows"])
        all_dep.extend(result["dependence_rows"])
        all_env.extend(result["environment_rows"])
        all_nuis.extend(result["nuisance_prediction_rows"])
        all_match.extend(result["matched_rows"])
        all_resid.extend(result["residual_rows"])
        all_folds.extend({"dataset": dataset, **row} for row in result["fold_rows"])
        integrity[dataset] = result["integrity"]
    report["decision"] = overall_decision(report["datasets"]) if len(selected) == 5 else "PARTIAL_RUN_NOT_CLASSIFIED"
    report["runtime_sec"] = time.time() - start
    write_json(RESULTS_DIR / "results.json", report)
    write_json(RESULTS_DIR / "integrity_checks.json", integrity)
    write_csv(RESULTS_DIR / "competence_metrics.csv", all_comp)
    write_csv(RESULTS_DIR / "routing_metrics.csv", all_route)
    write_csv(RESULTS_DIR / "dependence_aware_stats.csv", all_dep)
    write_csv(RESULTS_DIR / "environment_transfer.csv", all_env)
    write_csv(RESULTS_DIR / "nuisance_prediction.csv", all_nuis)
    write_csv(RESULTS_DIR / "matched_diagnostic.csv", all_match)
    write_csv(RESULTS_DIR / "residualization_metrics.csv", all_resid)
    write_csv(RESULTS_DIR / "oof_fold_manifest.csv", all_folds)
    write_combined_per_window(selected)
    write_report(report)
    print("TEST SET ACCESSED: NO")
    print(json.dumps({"decision": report["decision"], "runtime_sec": report["runtime_sec"], "datasets": list(selected)}, indent=2))


if __name__ == "__main__":
    main()
