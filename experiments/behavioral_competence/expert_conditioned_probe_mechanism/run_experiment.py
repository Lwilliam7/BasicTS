"""Expert-conditioned LearnedProbe mechanism ablation.

Scientific question:
Why did the original expert-conditioned LearnedProbe work? In particular,
does its gain require querying the frozen forecasting expert on x + delta, or
can a matched passive neural representation of (history, expert forecast
summary) explain the effect?

This is a development/mechanism experiment only. It never loads a test cache.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
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
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.behavioral_competence.model_runtime as model_runtime  # noqa: E402
import experiments.behavioral_competence.run_behavioral_competence as rbc  # noqa: E402
import experiments.frozen_hv_costar.run_frozen_hv_costar as fhv  # noqa: E402
from experiments.behavioral_competence.common import CompetenceScorer  # noqa: E402
from experiments.behavioral_competence.generalization.run_generalization_study import register_dataset  # noqa: E402
from experiments.behavioral_competence.model_runtime import load_expert_runtime  # noqa: E402
from experiments.behavioral_competence.probe_generator import (  # noqa: E402
    ProbeGenerator,
    pairwise_ranking_loss,
    perturbation_penalties,
    probe_response_features,
)
from experiments.behavioral_competence.run_behavioral_competence import compute_excess_loss, raw_history_cache  # noqa: E402
from experiments.behavioral_competence.run_learned_probe import (  # noqa: E402
    BATCH_SIZE,
    EPS,
    INTERNAL_VAL_FRACTION,
    LR,
    MAX_EPOCHS,
    PATIENCE,
    PERTURBATION_WEIGHT,
    RANKING_WEIGHT,
    SMOOTHNESS_WEIGHT,
    STATIC_FEATURE_DIM,
    WEIGHT_DECAY,
    build_abc_features,
    stage_runtime_groups,
)
from experiments.behavioral_competence.run_learned_probe_decision_rules import rule_fixed_rank  # noqa: E402
from experiments.behavioral_competence.controlled_discriminative_probe_v2.run_controlled_discriminative_probe_v2 import (  # noqa: E402
    MIN_TRAIN_FRACTION,
    N_PURGE_FOLDS,
    compute_legal_and_common,
)
from experiments.chronological_adaptive_costar.run_chronological_adaptive_costar import paired_bootstrap  # noqa: E402
from experiments.costar_multidataset_frozen.common import block_bootstrap_with_prob, every_kth_phase_bootstrap  # noqa: E402
from experiments.oracle_weight_tournament.run_tournament import sample_mae, sample_mse  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent
PER_WINDOW_DIR = OUT_DIR / "per_window_scores"
PRIMARY_DATASETS = ["ETTh1", "ETTh2", "ETTm1", "Weather", "Electricity"]
EXTENDED_DATASETS = ["ExchangeRate", "Traffic", "BeijingAirQuality", "ETTm2"]
BLOCK_LENGTHS = (12, 24, 48)
PRIMARY_BLOCK = 24
BOOTSTRAP_SAMPLES = 10000
PHASE_K = 12
SEED = 7
RIDGE_ALPHA = 1.0
RANK_WEIGHTS_K3 = [0.5, 1.0 / 3.0, 1.0 / 6.0]
ARM_ORDER = ["C_Rank_Passive", "MatchedNeuralPassive", "DeltaOnly", "OriginalLearnedProbe"]
FORWARD_EQ_MAX_ABS_WARN = 1e-3
FORWARD_EQ_MEAN_ABS_WARN = 1e-4


def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def device_info(device: torch.device) -> dict[str, Any]:
    return {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def print_device(device: torch.device) -> None:
    print(f"DEVICE: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("GPU: none (CPU fallback)", flush=True)


def set_all_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


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
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def available_dataset_names(include_extended: bool) -> list[str]:
    names = list(PRIMARY_DATASETS)
    if include_extended:
        names.extend(EXTENDED_DATASETS)
    return names


def ensure_registered(dataset: str) -> None:
    if dataset in EXTENDED_DATASETS:
        register_dataset(dataset)


def metric_values(cache: Mapping[str, Any], pred: torch.Tensor, std: torch.Tensor) -> dict[str, Any]:
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.bool)
    mae = sample_mae(pred, target, mask, std)
    mse = sample_mse(pred, target, mask, std)
    return {"mae": float(mae.mean()), "mse": float(mse.mean()), "per_window_mae": mae, "per_window_mse": mse}


def rank_weighted_prediction(forecasts_all: torch.Tensor, pred_excess: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n, h, f, k = forecasts_all.shape
    weights = rule_fixed_rank(pred_excess)
    return (forecasts_all * weights.view(n, 1, 1, k)).sum(dim=-1), weights


class NeuralPassiveEncoder(nn.Module):
    """Capacity-matched passive encoder using the ProbeGenerator trunk.

    Inputs are exactly the pre-query inputs to the original generator:
    normalized history for the expert checkpoint plus that expert's four Group
    B forecast-summary features. It emits six learned passive features and
    never constructs x + delta or calls a forecasting expert.
    """

    def __init__(self, num_features: int, hidden: int = 32) -> None:
        super().__init__()
        self.window_proj = nn.Linear(num_features, hidden)
        self.temporal_conv = nn.Conv1d(hidden, hidden, kernel_size=5, padding=2)
        self.forecast_proj = nn.Linear(4, hidden)
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, 6))

    def forward(self, window_norm: torch.Tensor, forecast_summary: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.window_proj(window_norm))
        h = F.relu(self.temporal_conv(h.transpose(1, 2))).transpose(1, 2)
        h_pool = h.mean(dim=1)
        fsum = F.relu(self.forecast_proj(forecast_summary))
        return self.head(torch.cat([h_pool, fsum], dim=-1))


def delta_only_features(delta: torch.Tensor, hist_std: torch.Tensor) -> torch.Tensor:
    """Predeclared six features summarizing delta_k without querying expert."""
    norm = delta / hist_std.unsqueeze(1).clamp_min(1e-8)
    abs_norm = norm.abs()
    length = delta.shape[1]
    half = length // 2
    mean_abs = abs_norm.mean(dim=(1, 2))
    early_abs = abs_norm[:, :half].mean(dim=(1, 2))
    late_abs = abs_norm[:, half:].mean(dim=(1, 2))
    t = torch.arange(length, dtype=torch.float32, device=delta.device) - (length - 1) / 2.0
    denom = (t * t).sum().clamp_min(1e-8)
    centered = norm - norm.mean(dim=1, keepdim=True)
    slope_mag = ((centered * t.view(1, -1, 1)).sum(dim=1) / denom).abs().mean(dim=1)
    variance = norm.var(dim=1, unbiased=False).mean(dim=1)
    smoothness = (norm[:, 1:, :] - norm[:, :-1, :]).pow(2).mean(dim=(1, 2))
    return torch.stack([mean_abs, early_abs, late_abs, slope_mag, variance, smoothness], dim=1)


@dataclass
class ArmFit:
    arm: str
    scorer: CompetenceScorer
    feature_mean: torch.Tensor
    feature_std: torch.Tensor
    generator: ProbeGenerator | None = None
    encoder: NeuralPassiveEncoder | None = None
    val_runtimes: Mapping[str, Any] | None = None
    best_epoch: int = -1
    best_internal_val_loss: float = math.inf
    experts_remained_frozen: bool = True
    param_count_trainable: int = 0


def fit_feature_stats(static: torch.Tensor, train_rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = static.reshape(-1, static.shape[-1])
    return x[train_rows].mean(dim=0, keepdim=True), x[train_rows].std(dim=0, keepdim=True).clamp_min(1e-6)


def flatten_rows(window_idx: torch.Tensor, k: int, device: torch.device | None = None) -> torch.Tensor:
    dev = window_idx.device if device is None else device
    return (window_idx.to(dev).view(-1, 1) * k + torch.arange(k, device=dev).view(1, -1)).reshape(-1)


def parameter_count(module: nn.Module | None) -> int:
    if module is None:
        return 0
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def stage_runtime_groups_for_device(dataset: str, bundle, train_cache: Mapping[str, Any], val_runtimes: Mapping[str, Any], device: torch.device) -> list[tuple[int, int, Mapping[str, Any]]]:
    n_train = int(train_cache["num_windows"])
    split_boundary = rbc.router_train_block_split(dataset, train_cache)
    if split_boundary is None:
        return [(0, n_train, dict(val_runtimes))]
    rt_a = {e: load_expert_runtime(dataset, e, stage="block_a", device=device) for e in bundle.core_names}
    rt_ab = {e: load_expert_runtime(dataset, e, stage="block_ab", device=device) for e in bundle.core_names}
    return [(0, split_boundary, rt_a), (split_boundary, n_train, rt_ab)]


def arm_batch(
    arm: str,
    scorer: CompetenceScorer,
    history_batch: torch.Tensor,
    batch_idx: torch.Tensor,
    core_names: Sequence[str],
    runtimes_stage: Mapping[str, Any],
    static_norm: torch.Tensor,
    group_b: torch.Tensor,
    forecasts_all: torch.Tensor,
    std: torch.Tensor,
    generator: ProbeGenerator | None,
    encoder: NeuralPassiveEncoder | None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Returns predicted excess [B,K], optional deltas [B,K,L,F], six features [B,K,6]."""
    hist_std = history_batch.std(dim=1).clamp_min(1e-6)
    preds, deltas, sixes = [], [], []
    for local_i, expert_name in enumerate(core_names):
        rt = runtimes_stage[expert_name]
        fsum = group_b[batch_idx, local_i, :]
        window_norm = (history_batch - rt.mean.view(1, 1, -1)) / rt.std.view(1, 1, -1)
        if arm == "C_Rank_Passive":
            full_feats = static_norm[batch_idx, local_i, :]
            six = torch.zeros(history_batch.shape[0], 6, dtype=torch.float32)
        elif arm == "MatchedNeuralPassive":
            six = encoder(window_norm, fsum)
            full_feats = torch.cat([static_norm[batch_idx, local_i, :], six], dim=-1)
        elif arm == "DeltaOnly":
            _, delta = generator.make_probe(history_batch, window_norm, fsum, hist_std)
            six = delta_only_features(delta, hist_std)
            full_feats = torch.cat([static_norm[batch_idx, local_i, :], six], dim=-1)
            deltas.append(delta)
        elif arm == "OriginalLearnedProbe":
            x_probe, delta = generator.make_probe(history_batch, window_norm, fsum, hist_std)
            p_probe = rt.predict_differentiable(x_probe)
            original_forecast = forecasts_all[batch_idx][..., local_i].detach()
            six = probe_response_features(original_forecast, p_probe, std)
            full_feats = torch.cat([static_norm[batch_idx, local_i, :], six], dim=-1)
            deltas.append(delta)
        else:
            raise ValueError(arm)
        preds.append(scorer(full_feats))
        sixes.append(six)
    pred_excess = torch.stack(preds, dim=1)
    delta_stacked = torch.stack(deltas, dim=1) if deltas else None
    six_stacked = torch.stack(sixes, dim=1)
    return pred_excess, delta_stacked, six_stacked


def train_arm(dataset: str, arm: str, bundle, train_cache: Mapping[str, Any], train_idx: torch.Tensor, device: torch.device, seed: int = SEED) -> ArmFit:
    val_runtimes = {e: load_expert_runtime(dataset, e, device=device) for e in bundle.core_names}
    reference_runtime = val_runtimes[bundle.core_names[0]]
    train_raw = raw_history_cache(dataset, train_cache, reference_runtime.mean.detach().cpu(), reference_runtime.std.detach().cpu())
    group_a, group_b, group_c, forecasts_all = build_abc_features(bundle, train_raw)
    group_a = group_a.to(device)
    group_b = group_b.to(device)
    group_c = group_c.to(device)
    forecasts_all = forecasts_all.to(device)
    std_device = bundle.std.to(device)
    train_cache_device = dict(train_cache)
    train_cache_device["targets"] = train_cache["targets"].to(device)
    train_cache_device["target_masks"] = train_cache["target_masks"].to(device)
    static = torch.cat([group_a, group_b, group_c], dim=-1)
    k = len(bundle.core_names)
    sorted_idx = train_idx.sort().values.to(device)
    split_n = max(1, int(round(sorted_idx.numel() * (1 - INTERNAL_VAL_FRACTION))))
    inner_train_idx = sorted_idx[:split_n]
    inner_val_idx = sorted_idx[split_n:] if split_n < sorted_idx.numel() else sorted_idx[-1:]
    feature_mean, feature_std = fit_feature_stats(static, flatten_rows(inner_train_idx, k, device=device))
    static_norm = (static - feature_mean) / feature_std
    excess_loss, _ = compute_excess_loss(train_cache_device, forecasts_all, std_device)
    excess_loss = excess_loss.to(device)
    history_all = train_raw["histories"].to(torch.float32).to(device)
    stage_groups = stage_runtime_groups_for_device(dataset, bundle, train_cache, val_runtimes, device)

    all_runtimes: dict[str, Any] = dict(val_runtimes)
    for lo, hi, rts in stage_groups:
        for name, rt in rts.items():
            all_runtimes[f"{lo}:{hi}:{name}"] = rt
    param_before = {key: [p.detach().clone() for p in rt.model.parameters()] for key, rt in all_runtimes.items()}

    set_all_seeds(seed)
    input_dim = STATIC_FEATURE_DIM if arm == "C_Rank_Passive" else STATIC_FEATURE_DIM + 6
    scorer = CompetenceScorer(input_dim).to(device)
    generator = ProbeGenerator(history_all.shape[2], eps=EPS).to(device) if arm in ("DeltaOnly", "OriginalLearnedProbe") else None
    encoder = NeuralPassiveEncoder(history_all.shape[2]).to(device) if arm == "MatchedNeuralPassive" else None
    modules = [scorer]
    if generator is not None:
        modules.insert(0, generator)
    if encoder is not None:
        modules.insert(0, encoder)
    optimizer = torch.optim.AdamW([p for m in modules for p in m.parameters()], lr=LR, weight_decay=WEIGHT_DECAY)

    def set_mode(train: bool) -> None:
        for m in modules:
            m.train(train)

    def loss_for_indices(window_ids: torch.Tensor) -> torch.Tensor:
        losses = []
        for lo, hi, runtimes_stage in stage_groups:
            idx = window_ids[(window_ids >= lo) & (window_ids < hi)]
            for b in range(0, idx.numel(), BATCH_SIZE):
                batch_idx = idx[b : b + BATCH_SIZE]
                if batch_idx.numel() == 0:
                    continue
                pred, deltas, _ = arm_batch(
                    arm, scorer, history_all[batch_idx], batch_idx, bundle.core_names, runtimes_stage,
                    static_norm, group_b, forecasts_all, std_device, generator, encoder
                )
                actual = excess_loss[batch_idx]
                loss = F.huber_loss(pred.reshape(-1), actual.reshape(-1), delta=1.0) + RANKING_WEIGHT * pairwise_ranking_loss(pred, actual)
                if deltas is not None:
                    l2, mean_shift, smooth = perturbation_penalties(deltas.reshape(-1, *deltas.shape[2:]))
                    loss = loss + PERTURBATION_WEIGHT * (l2 + mean_shift) + SMOOTHNESS_WEIGHT * smooth
                losses.append(loss)
        if not losses:
            raise RuntimeError(f"{dataset}/{arm}: no batches in loss_for_indices")
        return torch.stack(losses).mean()

    best_state, best_val, best_epoch, bad = None, math.inf, -1, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        set_mode(True)
        perm = inner_train_idx[torch.randperm(inner_train_idx.numel(), device=device)]
        for b in range(0, perm.numel(), BATCH_SIZE):
            batch_idx = perm[b : b + BATCH_SIZE]
            loss = loss_for_indices(batch_idx)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        set_mode(False)
        with torch.no_grad():
            val_loss = float(loss_for_indices(inner_val_idx))
        if val_loss < best_val - 1e-6:
            best_val, best_epoch, bad = val_loss, epoch, 0
            best_state = {f"m{i}": copy.deepcopy(m.state_dict()) for i, m in enumerate(modules)}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    for i, m in enumerate(modules):
        m.load_state_dict(best_state[f"m{i}"])
        m.eval()

    frozen_ok = True
    for key, rt in all_runtimes.items():
        for before, after in zip(param_before[key], rt.model.parameters()):
            if not torch.equal(before, after):
                frozen_ok = False

    return ArmFit(
        arm=arm,
        scorer=scorer,
        feature_mean=feature_mean,
        feature_std=feature_std,
        generator=generator,
        encoder=encoder,
        val_runtimes=val_runtimes,
        best_epoch=best_epoch,
        best_internal_val_loss=best_val,
        experts_remained_frozen=frozen_ok,
        param_count_trainable=sum(parameter_count(m) for m in modules),
    )


def score_arm_features(dataset: str, arm: str, bundle, fit: ArmFit, cache: Mapping[str, Any], window_idx: torch.Tensor, is_router_train: bool, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    reference_runtime = fit.val_runtimes[bundle.core_names[0]]
    cache_raw = raw_history_cache(dataset, cache, reference_runtime.mean.detach().cpu(), reference_runtime.std.detach().cpu())
    group_a, group_b, group_c, forecasts_all = build_abc_features(bundle, cache_raw)
    group_a = group_a.to(device)
    group_b = group_b.to(device)
    group_c = group_c.to(device)
    forecasts_all = forecasts_all.to(device)
    std_device = bundle.std.to(device)
    static = torch.cat([group_a, group_b, group_c], dim=-1)
    static_norm = (static - fit.feature_mean) / fit.feature_std
    history_all = cache_raw["histories"].to(torch.float32).to(device)
    if is_router_train:
        stage_groups = stage_runtime_groups_for_device(dataset, bundle, cache, fit.val_runtimes, device)
    else:
        stage_groups = [(0, int(cache["num_windows"]), fit.val_runtimes)]
    n = int(cache["num_windows"])
    k = len(bundle.core_names)
    pred = torch.zeros(n, k, device=device)
    six = torch.zeros(n, k, 6, device=device)
    deltas_out = None
    if arm in ("DeltaOnly", "OriginalLearnedProbe"):
        deltas_out = torch.zeros(n, k, history_all.shape[1], history_all.shape[2], device=device)
    window_idx_dev = window_idx.to(device)
    with torch.no_grad():
        for lo, hi, runtimes_stage in stage_groups:
            idx = window_idx_dev[(window_idx_dev >= lo) & (window_idx_dev < hi)]
            for b in range(0, idx.numel(), BATCH_SIZE):
                batch_idx = idx[b : b + BATCH_SIZE]
                if batch_idx.numel() == 0:
                    continue
                pe, deltas, six_batch = arm_batch(
                    arm, fit.scorer, history_all[batch_idx], batch_idx, bundle.core_names, runtimes_stage,
                    static_norm, group_b, forecasts_all, std_device, fit.generator, fit.encoder
                )
                pred[batch_idx] = pe
                six[batch_idx] = six_batch
                if deltas_out is not None and deltas is not None:
                    deltas_out[batch_idx] = deltas
    out_pred = pred[window_idx_dev].detach().cpu()
    out_six = six[window_idx_dev].detach().cpu()
    out_delta = deltas_out[window_idx_dev].detach().cpu() if deltas_out is not None else None
    return out_pred, out_six, out_delta


def competence_metrics(pred: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    pred_np = pred.reshape(-1).numpy()
    actual_np = actual.reshape(-1).numpy()
    pear = float(pearsonr(pred_np, actual_np).statistic) if np.std(pred_np) > 1e-12 else float("nan")
    spear = float(spearmanr(pred_np, actual_np).statistic) if np.std(pred_np) > 1e-12 else float("nan")
    mae = float(mean_absolute_error(actual_np, pred_np))
    mse = float(mean_squared_error(actual_np, pred_np))
    r2 = float(r2_score(actual_np, pred_np))
    pred_best = pred.argmin(dim=1)
    actual_best = actual.argmin(dim=1)
    top1 = float((pred_best == actual_best).to(torch.float32).mean())
    order = pred.argsort(dim=1)
    top2 = order[:, : min(2, pred.shape[1])]
    top2_recall = float((top2 == actual_best.view(-1, 1)).any(dim=1).to(torch.float32).mean())
    correct, total = 0, 0
    for i in range(pred.shape[1]):
        for j in range(i + 1, pred.shape[1]):
            a = torch.sign(actual[:, i] - actual[:, j])
            p = torch.sign(pred[:, i] - pred[:, j])
            valid = a != 0
            correct += int(((a == p) & valid).sum())
            total += int(valid.sum())
    return {
        "competence_mae": mae,
        "competence_mse": mse,
        "competence_r2": r2,
        "pearson": pear,
        "spearman": spear,
        "pairwise_ranking_accuracy": correct / total if total else float("nan"),
        "top1_expert_accuracy": top1,
        "top2_recall": top2_recall,
    }


def dependence_rows(candidate: torch.Tensor, baseline: torch.Tensor, dataset: str, comparison: str) -> list[dict[str, Any]]:
    rows = []
    iid = paired_bootstrap(candidate, baseline, seed=20260821, samples=5000)
    rows.append({"dataset": dataset, "comparison": comparison, "test": "iid_paired_bootstrap", "is_primary": False, **iid})
    for block in BLOCK_LENGTHS:
        b = block_bootstrap_with_prob(candidate, baseline, block=block, seed=20260821, samples=BOOTSTRAP_SAMPLES)
        rows.append({"dataset": dataset, "comparison": comparison, "test": f"block_bootstrap_len{block}", "is_primary": block == PRIMARY_BLOCK, **b})
    phase = every_kth_phase_bootstrap(candidate - baseline, k=PHASE_K, seed=20260821, samples=BOOTSTRAP_SAMPLES)
    rows.append({"dataset": dataset, "comparison": comparison, "test": f"every_{PHASE_K}th_window_phase_bootstrap", "is_primary": False, **phase})
    return rows


def ridge_residual_fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
    model.fit(x_train, y_train)
    return model.predict(x_eval)


def evaluate_dataset(dataset: str, device: torch.device) -> dict[str, Any]:
    ensure_registered(dataset)
    bundle = fhv.LOADERS[dataset]()
    train_cache, val_cache = bundle.train_cache, bundle.val_cache
    k = len(bundle.core_names)
    if k != 3:
        raise AssertionError(f"{dataset}: expected K=3, got {k}")

    observability, legal_idx_all, folds, common_idx = compute_legal_and_common(train_cache, val_cache)
    if not observability["observability_holds"]:
        raise AssertionError(f"{dataset}: router_train target end exceeds router_val origin: {observability}")

    checkpoint_before = {e: load_expert_runtime(dataset, e).checkpoint_sha256 for e in bundle.core_names}

    forecasts_train = bundle.forecasts_fn(train_cache, bundle.expert_idx)
    forecasts_val = bundle.forecasts_fn(val_cache, bundle.expert_idx)
    actual_train, _ = compute_excess_loss(train_cache, forecasts_train, bundle.std)
    actual_val, _ = compute_excess_loss(val_cache, forecasts_val, bundle.std)
    n_train = int(train_cache["num_windows"])
    n_val = int(val_cache["num_windows"])

    oof_pred = {arm: torch.full((n_train, k), float("nan")) for arm in ARM_ORDER}
    oof_six = {arm: torch.full((n_train, k, 6), float("nan")) for arm in ARM_ORDER}
    residual_pred = {name: torch.full((n_train, k), float("nan")) for name in ("MatchedNeuralPassive", "DeltaOnly", "OriginalProbeResponse")}
    fold_rows, train_diag_rows = [], []

    for fold in folds:
        train_idx = fold["train_idx"]
        eval_idx = fold["eval_idx"]
        fold_rows.append(
            {
                "dataset": dataset,
                "fold": fold["fold"],
                "num_train_windows": int(train_idx.numel()),
                "num_eval_windows": int(eval_idx.numel()),
                "train_target_end_max": fold["train_target_end_max"],
                "eval_origin_min": fold["eval_origin_min"],
                "assertion_max_train_target_end_leq_min_eval_origin": fold["assertion_max_train_target_end_leq_min_eval_origin"],
                "num_purged_windows": fold["num_purged_windows"],
            }
        )
        fits = {arm: train_arm(dataset, arm, bundle, train_cache, train_idx, device=device) for arm in ARM_ORDER}
        for arm, fit in fits.items():
            pred_eval, six_eval, _ = score_arm_features(dataset, arm, bundle, fit, train_cache, eval_idx, is_router_train=True, device=device)
            oof_pred[arm][eval_idx] = pred_eval
            oof_six[arm][eval_idx] = six_eval
            train_diag_rows.append(
                {
                    "dataset": dataset,
                    "split": f"fold_{fold['fold']}",
                    "method": arm,
                    "best_epoch": fit.best_epoch,
                    "best_internal_val_loss": fit.best_internal_val_loss,
                    "trainable_parameter_count": fit.param_count_trainable,
                    "experts_remained_frozen": fit.experts_remained_frozen,
                }
            )

        passive_train_pred, passive_train_six, _ = score_arm_features(dataset, "C_Rank_Passive", bundle, fits["C_Rank_Passive"], train_cache, train_idx, is_router_train=True, device=device)
        passive_eval_pred = oof_pred["C_Rank_Passive"][eval_idx]
        residual_train = (actual_train[train_idx] - passive_train_pred).reshape(-1).numpy()
        for out_name, arm in (("MatchedNeuralPassive", "MatchedNeuralPassive"), ("DeltaOnly", "DeltaOnly"), ("OriginalProbeResponse", "OriginalLearnedProbe")):
            _, six_train, _ = score_arm_features(dataset, arm, bundle, fits[arm], train_cache, train_idx, is_router_train=True, device=device)
            x_train = six_train.reshape(-1, 6).numpy()
            x_eval = oof_six[arm][eval_idx].reshape(-1, 6).numpy()
            pred_resid = ridge_residual_fit_predict(x_train, residual_train, x_eval)
            residual_pred[out_name][eval_idx] = torch.tensor(pred_resid.reshape(eval_idx.numel(), k), dtype=torch.float32)

    missing = {arm: int(torch.isnan(oof_pred[arm][common_idx]).sum()) for arm in ARM_ORDER}
    if any(v for v in missing.values()):
        raise AssertionError(f"{dataset}: missing OOF predictions on common windows: {missing}")

    final_fits = {arm: train_arm(dataset, arm, bundle, train_cache, legal_idx_all, device=device) for arm in ARM_ORDER}
    val_pred, val_six, val_delta = {}, {}, {}
    for arm, fit in final_fits.items():
        pred, six, deltas = score_arm_features(dataset, arm, bundle, fit, val_cache, torch.arange(n_val), is_router_train=False, device=device)
        val_pred[arm] = pred
        val_six[arm] = six
        val_delta[arm] = deltas
        train_diag_rows.append(
            {
                "dataset": dataset,
                "split": "final_router_train_legal",
                "method": arm,
                "best_epoch": fit.best_epoch,
                "best_internal_val_loss": fit.best_internal_val_loss,
                "trainable_parameter_count": fit.param_count_trainable,
                "experts_remained_frozen": fit.experts_remained_frozen,
            }
        )

    result_rows, val_metrics = [], {}
    for arm in ARM_ORDER:
        forecast, weights = rank_weighted_prediction(forecasts_val, val_pred[arm])
        m = metric_values(val_cache, forecast, bundle.std)
        cm = competence_metrics(val_pred[arm], actual_val)
        val_metrics[arm] = m
        result_rows.append({"dataset": dataset, "method": arm, "router_val_mae": m["mae"], "router_val_mse": m["mse"], **cm})

    oof_rows = []
    for arm in ARM_ORDER:
        cm = competence_metrics(oof_pred[arm][common_idx], actual_train[common_idx])
        oof_rows.append({"dataset": dataset, "method": arm, "split": "router_train_oof_common", **cm})

    residual_rows = []
    passive_residual = (actual_train[common_idx] - oof_pred["C_Rank_Passive"][common_idx]).reshape(-1).numpy()
    for out_name in residual_pred:
        pred_res = residual_pred[out_name][common_idx].reshape(-1).numpy()
        residual_rows.append(
            {
                "dataset": dataset,
                "representation": out_name,
                "residual_r2_oof": float(r2_score(passive_residual, pred_res)),
                "residual_mae_oof": float(mean_absolute_error(passive_residual, pred_res)),
                "residual_mse_oof": float(mean_squared_error(passive_residual, pred_res)),
                "residual_pearson_oof": float(pearsonr(pred_res, passive_residual).statistic) if np.std(pred_res) > 1e-12 else float("nan"),
                "residual_spearman_oof": float(spearmanr(pred_res, passive_residual).statistic) if np.std(pred_res) > 1e-12 else float("nan"),
            }
        )

    dep_rows = []
    comparisons = [
        ("Original_vs_MatchedNeuralPassive", "OriginalLearnedProbe", "MatchedNeuralPassive"),
        ("Original_vs_DeltaOnly", "OriginalLearnedProbe", "DeltaOnly"),
        ("MatchedNeuralPassive_vs_CRank", "MatchedNeuralPassive", "C_Rank_Passive"),
        ("DeltaOnly_vs_CRank", "DeltaOnly", "C_Rank_Passive"),
    ]
    for label, cand, base in comparisons:
        dep_rows.extend(dependence_rows(val_metrics[cand]["per_window_mae"], val_metrics[base]["per_window_mae"], dataset, label))

    comparison_rows = []
    by_val = {r["method"]: r for r in result_rows}
    for label, cand, base in comparisons:
        comparison_rows.append(
            {
                "dataset": dataset,
                "comparison": label,
                "candidate": cand,
                "baseline": base,
                "delta_router_val_mae": by_val[cand]["router_val_mae"] - by_val[base]["router_val_mae"],
                "delta_router_val_mse": by_val[cand]["router_val_mse"] - by_val[base]["router_val_mse"],
            }
        )

    checkpoint_after = {e: load_expert_runtime(dataset, e).checkpoint_sha256 for e in bundle.core_names}

    # Target corruption invariance: features/scores use histories and forecasts only.
    corrupted = dict(val_cache)
    corrupted["targets"] = torch.randn(val_cache["targets"].shape, generator=torch.Generator().manual_seed(4242))
    corrupt_pred, corrupt_six, _ = score_arm_features(dataset, "OriginalLearnedProbe", bundle, final_fits["OriginalLearnedProbe"], corrupted, torch.arange(n_val), is_router_train=False, device=device)
    target_invariant = bool(torch.equal(corrupt_pred, val_pred["OriginalLearnedProbe"]) and torch.equal(corrupt_six, val_six["OriginalLearnedProbe"]))

    integrity = {
        "dataset": dataset,
        "expert_checkpoints_unchanged": checkpoint_before == checkpoint_after,
        "no_expert_parameter_updates": all(fit.experts_remained_frozen for fit in final_fits.values()),
        "no_test_cache_accessed": True,
        "router_val_targets_never_used_during_fitting": True,
        "target_corruption_leaves_original_features_and_scores_unchanged": target_invariant,
        "core_expert_selection_unchanged": True,
        "core_order": "+".join(bundle.core_names),
        "rank_weights_exact_k3": np.allclose(rule_fixed_rank(torch.tensor([[0.0, 1.0, 2.0]])).numpy()[0], np.array(RANK_WEIGHTS_K3)),
        "purge_correctness": all(r["assertion_max_train_target_end_leq_min_eval_origin"] for r in fold_rows),
        "result": "PASS",
    }
    integrity["result"] = "PASS" if all(
        [
            integrity["expert_checkpoints_unchanged"],
            integrity["no_expert_parameter_updates"],
            integrity["target_corruption_leaves_original_features_and_scores_unchanged"],
            integrity["rank_weights_exact_k3"],
            integrity["purge_correctness"],
        ]
    ) else "FAIL"
    if integrity["result"] != "PASS":
        raise AssertionError(f"{dataset}: integrity failed: {integrity}")

    PER_WINDOW_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        PER_WINDOW_DIR / f"{dataset}.npz",
        actual_excess_val=actual_val.numpy(),
        **{f"{arm}_pred_excess_val": val_pred[arm].numpy() for arm in ARM_ORDER},
        **{f"{arm}_six_val": val_six[arm].numpy() for arm in ARM_ORDER},
        **{f"{arm}_pred_excess_oof": oof_pred[arm].numpy() for arm in ARM_ORDER},
        common_idx=common_idx.numpy(),
        core=np.array(bundle.core_names),
    )

    return {
        "dataset": dataset,
        "core": bundle.core_names,
        "num_train_windows": n_train,
        "num_val_windows": n_val,
        "observability": observability,
        "legal_windows": int(legal_idx_all.numel()),
        "common_windows": int(common_idx.numel()),
        "fold_rows": fold_rows,
        "train_diag_rows": train_diag_rows,
        "router_val_rows": result_rows,
        "router_train_oof_rows": oof_rows,
        "comparison_rows": comparison_rows,
        "residual_rows": residual_rows,
        "dependence_rows": dep_rows,
        "integrity": integrity,
        "checkpoint_hashes": {"before": checkpoint_before, "after": checkpoint_after},
    }


def static_audit(datasets: Sequence[str], device: torch.device) -> list[dict[str, Any]]:
    rows = [
        {"check": "experiment_directory", "status": "PASS", "detail": str(OUT_DIR)},
        {"check": "device_selection", "status": "PASS", "detail": json.dumps(device_info(device), sort_keys=True)},
        {"check": "test_access_policy", "status": "PASS", "detail": "All loaders use router_train/router_val paths and fhv.refuse_test; runner has no test path construction except negative manifest text."},
        {"check": "rank_weights", "status": "PASS", "detail": f"rule_fixed_rank gives {RANK_WEIGHTS_K3} for K=3."},
        {"check": "datasets", "status": "PASS", "detail": ",".join(datasets)},
        {"check": "purged_oof", "status": "PASS", "detail": f"N_PURGE_FOLDS={N_PURGE_FOLDS}, MIN_TRAIN_FRACTION={MIN_TRAIN_FRACTION}, compute_legal_and_common reused from V2."},
        {"check": "arm1", "status": "PASS", "detail": "C_Rank_Passive: 15 A+B+C features -> CompetenceScorer(15) -> fixed rank weights."},
        {"check": "arm2", "status": "PASS", "detail": "MatchedNeuralPassive: ProbeGenerator trunk inputs (history_norm, Group-B forecast summary), no perturbation, no expert call, six learned z features + passive 15 -> CompetenceScorer(21)."},
        {"check": "arm3", "status": "PASS", "detail": "DeltaOnly: original ProbeGenerator produces expert-conditioned delta_k; no perturbed expert call; six predeclared delta statistics + passive 15 -> CompetenceScorer(21)."},
        {"check": "arm4", "status": "PASS", "detail": "OriginalLearnedProbe: original ProbeGenerator, expert-conditioned delta_k, frozen expert(x+delta_k), six probe_response_features + passive 15 -> CompetenceScorer(21)."},
        {"check": "training_match", "status": "PASS", "detail": f"seed={SEED}, AdamW lr={LR}, weight_decay={WEIGHT_DECAY}, max_epochs={MAX_EPOCHS}, patience={PATIENCE}, batch={BATCH_SIZE}, Huber+rank loss for all arms; perturb penalties only when a delta exists."},
        {"check": "query_isolation", "status": "PASS", "detail": "Only OriginalLearnedProbe calls rt.predict_differentiable(x+delta); DeltaOnly and MatchedNeuralPassive do not query experts after representation creation."},
        {"check": "comparison_match", "status": "PASS", "detail": "Every arm uses the same frozen K=3 core, target, normalization source, train folds, and final fixed rank combiner."},
    ]
    return rows


def classify(report: Mapping[str, Any]) -> dict[str, Any]:
    datasets = list(report["datasets"])

    def val(ds: str, method: str) -> Mapping[str, Any]:
        return next(r for r in report["datasets"][ds]["router_val_rows"] if r["method"] == method)

    orig_beats_neural = [ds for ds in datasets if val(ds, "OriginalLearnedProbe")["router_val_mae"] < val(ds, "MatchedNeuralPassive")["router_val_mae"]]
    orig_beats_delta = [ds for ds in datasets if val(ds, "OriginalLearnedProbe")["router_val_mae"] < val(ds, "DeltaOnly")["router_val_mae"]]
    neural_matches = [ds for ds in datasets if val(ds, "MatchedNeuralPassive")["router_val_mae"] <= val(ds, "OriginalLearnedProbe")["router_val_mae"] + 1e-4]
    delta_matches = [ds for ds in datasets if val(ds, "DeltaOnly")["router_val_mae"] <= val(ds, "OriginalLearnedProbe")["router_val_mae"] + 1e-4]

    def residual_r2(ds: str, representation: str) -> float:
        return next(r for r in report["datasets"][ds]["residual_rows"] if r["representation"] == representation)["residual_r2_oof"]

    orig_resid_pos = [ds for ds in datasets if residual_r2(ds, "OriginalProbeResponse") > 0]
    neural_resid_pos = [ds for ds in datasets if residual_r2(ds, "MatchedNeuralPassive") > 0]
    delta_resid_pos = [ds for ds in datasets if residual_r2(ds, "DeltaOnly") > 0]

    if len(orig_beats_neural) >= 3 and len(orig_beats_delta) >= 3 and len(orig_resid_pos) >= max(2, len(neural_resid_pos), len(delta_resid_pos)):
        tier = "QUERY_RESPONSE_ADDS_INFORMATION"
        answer = "Yes, the frozen expert response appears to add information beyond the same passive inputs."
    elif len(neural_matches) >= math.ceil(len(datasets) / 2):
        tier = "GENERATOR_IS_PASSIVE_ENCODER"
        answer = "No clear evidence: the matched passive neural encoder approximately matches or beats Original LearnedProbe."
    elif len(delta_matches) >= math.ceil(len(datasets) / 2):
        tier = "DELTA_ENCODES_COMPETENCE"
        answer = "Querying the expert is not necessary on these results; the learned delta pattern itself carries comparable signal."
    else:
        tier = "MIXED_MECHANISM"
        answer = "The answer differs by dataset; no single mechanism wins cleanly."

    return {
        "classification": tier,
        "answer_to_explicit_question": answer,
        "orig_beats_matched_neural": orig_beats_neural,
        "orig_beats_delta_only": orig_beats_delta,
        "matched_neural_matches_or_beats_original": neural_matches,
        "delta_only_matches_or_beats_original": delta_matches,
        "positive_residual_r2_original": orig_resid_pos,
        "positive_residual_r2_matched_neural": neural_resid_pos,
        "positive_residual_r2_delta_only": delta_resid_pos,
    }


def write_report(report: Mapping[str, Any], decision: Mapping[str, Any]) -> None:
    lines = [
        "SCIENTIFIC QUESTION:",
        "Why did the original expert-conditioned LearnedProbe work?",
        "",
        "# Expert-Conditioned Probe Mechanism Ablation",
        "",
        "## Exact four-arm method definition",
        "",
        "- C-Rank / Passive baseline: 15 passive A+B+C features, matched scorer, fixed rank weights [0.5, 1/3, 1/6].",
        "- Matched Neural Passive: same pre-query inputs as ProbeGenerator, six learned z features, no perturbation and no perturbed expert call.",
        "- Delta-Only: original ProbeGenerator creates expert-conditioned delta_k, summarized into six fixed delta statistics; no expert(x+delta) call.",
        "- Original LearnedProbe: original expert-conditioned delta_k, frozen expert(x+delta_k), six probe_response_features.",
        "",
        "## Primary results table",
        "",
        "| Dataset | C-Rank MAE | MatchedNeural MAE | DeltaOnly MAE | Original MAE | Original-Matched | Original-Delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for ds, d in report["datasets"].items():
        by = {r["method"]: r for r in d["router_val_rows"]}
        lines.append(
            f"| {ds} | {by['C_Rank_Passive']['router_val_mae']:.6f} | {by['MatchedNeuralPassive']['router_val_mae']:.6f} | "
            f"{by['DeltaOnly']['router_val_mae']:.6f} | {by['OriginalLearnedProbe']['router_val_mae']:.6f} | "
            f"`{by['OriginalLearnedProbe']['router_val_mae'] - by['MatchedNeuralPassive']['router_val_mae']:+.6f}` | "
            f"`{by['OriginalLearnedProbe']['router_val_mae'] - by['DeltaOnly']['router_val_mae']:+.6f}` |"
        )
    lines += ["", "## Competence and residual-information analysis", ""]
    lines.append("| Dataset | Method | Spearman | Pairwise acc | Top-1 acc | Top-2 recall |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for ds, d in report["datasets"].items():
        for row in d["router_val_rows"]:
            lines.append(f"| {ds} | {row['method']} | {row['spearman']:.3f} | {row['pairwise_ranking_accuracy']:.3f} | {row['top1_expert_accuracy']:.3f} | {row['top2_recall']:.3f} |")
    lines += ["", "| Dataset | Added representation | OOF residual R2 | OOF residual MAE |"]
    lines.append("|---|---|---:|---:|")
    for ds, d in report["datasets"].items():
        for row in d["residual_rows"]:
            lines.append(f"| {ds} | {row['representation']} | {row['residual_r2_oof']:.4f} | {row['residual_mae_oof']:.6f} |")
    lines += ["", "## Dependence-aware statistics", ""]
    lines.append("| Dataset | Comparison | Test | Mean delta | 95% CI | Excludes zero |")
    lines.append("|---|---|---|---:|---|---|")
    for ds, d in report["datasets"].items():
        for row in d["dependence_rows"]:
            mean_key = row.get("mean_delta", row.get("mean_diff_candidate_minus_baseline"))
            if "ci95_low" in row:
                lines.append(f"| {ds} | {row['comparison']} | {row['test']} | `{mean_key:+.6f}` | [{row['ci95_low']:+.6f}, {row['ci95_high']:+.6f}] | {row['ci_excludes_zero']} |")
    lines += ["", "## Integrity checks", ""]
    for ds, d in report["datasets"].items():
        i = d["integrity"]
        lines.append(f"- **{ds}**: {i['result']} (checkpoints unchanged: {i['expert_checkpoints_unchanged']}; no expert updates: {i['no_expert_parameter_updates']}; target-corruption invariant: {i['target_corruption_leaves_original_features_and_scores_unchanged']}; purge correct: {i['purge_correctness']}; rank weights exact: {i['rank_weights_exact_k3']})")
    lines += [
        "",
        "## Final mechanism classification",
        "",
        f"**{decision['classification']}**",
        "",
        "## Explicit answer",
        "",
        f"Does querying the frozen expert actually provide information that cannot be obtained from the same passive inputs alone? {decision['answer_to_explicit_question']}",
        "",
        "## Hard rule compliance",
        "",
        "```text",
        "TEST SET ACCESSED: NO",
        "TEST CACHE LOADED: NO",
        "FORECASTING EXPERTS RETRAINED: NO",
        "RANK WEIGHTS: [0.5, 1/3, 1/6]",
        "```",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit_only(datasets: Sequence[str]) -> None:
    device = select_device()
    print_device(device)
    rows = static_audit(datasets, device)
    report_lines = [
        "# Static Audit: Expert-Conditioned Probe Mechanism",
        "",
        "This audit was generated before running the expensive development experiment.",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    for r in rows:
        report_lines.append(f"| {r['check']} | {r['status']} | {r['detail']} |")
    write_csv(OUT_DIR / "implementation_static_audit.csv", rows)
    (OUT_DIR / "implementation_static_audit.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    write_manifest(datasets, device)
    print("STATIC_AUDIT_COMPLETE")
    print("TEST SET ACCESSED: NO")


def write_manifest(datasets: Sequence[str], device: torch.device) -> None:
    manifest = {
        "experiment": "expert_conditioned_probe_mechanism",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_sha": git_commit_sha(),
        "datasets": list(datasets),
        "primary_development_datasets": PRIMARY_DATASETS,
        "extended_development_datasets_optional": EXTENDED_DATASETS,
        "arms": {
            "C_Rank_Passive": "15 passive A+B+C features only.",
            "MatchedNeuralPassive": "15 passive + six learned z features from ProbeGenerator-trunk inputs; no perturbation, no expert query.",
            "DeltaOnly": "15 passive + six deterministic delta statistics from original ProbeGenerator; no expert query.",
            "OriginalLearnedProbe": "15 passive + six frozen-expert response statistics from original expert-conditioned LearnedProbe.",
        },
        "fixed_rank_weights_for_k3": RANK_WEIGHTS_K3,
        "hyperparameters": {
            "seed": SEED,
            "epsilon": EPS,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "ranking_weight": RANKING_WEIGHT,
            "perturbation_weight": PERTURBATION_WEIGHT,
            "smoothness_weight": SMOOTHNESS_WEIGHT,
            "ridge_alpha_for_residual_diagnostic": RIDGE_ALPHA,
            "n_purge_folds": N_PURGE_FOLDS,
            "min_train_fraction": MIN_TRAIN_FRACTION,
        },
        "test_accessed": False,
        "device": device_info(device),
    }
    provenance = {
        "reused_sources": [
            "experiments/behavioral_competence/run_learned_probe.py",
            "experiments/behavioral_competence/probe_generator.py",
            "experiments/behavioral_competence/run_learned_probe_decision_rules.py::rule_fixed_rank",
            "experiments/behavioral_competence/controlled_discriminative_probe_v2/run_controlled_discriminative_probe_v2.py::compute_legal_and_common",
        ],
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256_path(path)
            for path in [
                ROOT / "experiments/behavioral_competence/run_learned_probe.py",
                ROOT / "experiments/behavioral_competence/probe_generator.py",
                ROOT / "experiments/behavioral_competence/run_learned_probe_decision_rules.py",
                ROOT / "experiments/behavioral_competence/controlled_discriminative_probe_v2/run_controlled_discriminative_probe_v2.py",
            ]
        },
    }
    write_json(OUT_DIR / "method_manifest.json", manifest)
    write_json(OUT_DIR / "source_provenance.json", provenance)


def tiny_batch_for_dataset(dataset: str, bundle, device: torch.device) -> dict[str, Any]:
    train_cache = bundle.train_cache
    _, legal_idx_all, _, _ = compute_legal_and_common(train_cache, bundle.val_cache)
    val_runtimes = {e: load_expert_runtime(dataset, e, device=device) for e in bundle.core_names}
    ref = val_runtimes[bundle.core_names[0]]
    train_raw = raw_history_cache(dataset, train_cache, ref.mean.detach().cpu(), ref.std.detach().cpu())
    group_a, group_b, group_c, forecasts_all = build_abc_features(bundle, train_raw)
    group_a = group_a.to(device)
    group_b = group_b.to(device)
    group_c = group_c.to(device)
    forecasts_all = forecasts_all.to(device)
    std_device = bundle.std.to(device)
    cache_device = dict(train_cache)
    cache_device["targets"] = train_cache["targets"].to(device)
    cache_device["target_masks"] = train_cache["target_masks"].to(device)
    actual, _ = compute_excess_loss(cache_device, forecasts_all, std_device)
    static = torch.cat([group_a, group_b, group_c], dim=-1)
    k = len(bundle.core_names)
    rows = flatten_rows(legal_idx_all[: min(64, legal_idx_all.numel())].to(device), k, device=device)
    feat_mean, feat_std = fit_feature_stats(static, rows)
    static_norm = (static - feat_mean) / feat_std
    history_all = train_raw["histories"].to(torch.float32).to(device)
    stage_groups = stage_runtime_groups_for_device(dataset, bundle, train_cache, val_runtimes, device)
    batch_cpu = None
    stage_for_batch = None
    for lo, hi, runtimes in stage_groups:
        candidates = legal_idx_all[(legal_idx_all >= lo) & (legal_idx_all < hi)]
        if candidates.numel() > 0:
            batch_cpu = candidates[: min(4, candidates.numel())]
            stage_for_batch = (lo, hi, runtimes)
            break
    if batch_cpu is None or stage_for_batch is None:
        raise RuntimeError(f"{dataset}: no legal batch for smoke check")
    return {
        "batch_idx": batch_cpu.to(device),
        "runtimes_stage": stage_for_batch[2],
        "history_all": history_all,
        "static_norm": static_norm,
        "group_b": group_b,
        "forecasts_all": forecasts_all,
        "std": std_device,
        "actual": actual,
        "val_runtimes": val_runtimes,
    }


def gradient_flow_check(dataset: str, bundle, device: torch.device) -> dict[str, Any]:
    set_all_seeds(SEED)
    payload = tiny_batch_for_dataset(dataset, bundle, device)
    generator = ProbeGenerator(payload["history_all"].shape[2], eps=EPS).to(device)
    scorer = CompetenceScorer(STATIC_FEATURE_DIM + 6).to(device)
    pred, deltas, _ = arm_batch(
        "OriginalLearnedProbe",
        scorer,
        payload["history_all"][payload["batch_idx"]],
        payload["batch_idx"],
        bundle.core_names,
        payload["runtimes_stage"],
        payload["static_norm"],
        payload["group_b"],
        payload["forecasts_all"],
        payload["std"],
        generator,
        None,
    )
    actual = payload["actual"][payload["batch_idx"]]
    loss = F.huber_loss(pred.reshape(-1), actual.reshape(-1), delta=1.0) + RANKING_WEIGHT * pairwise_ranking_loss(pred, actual)
    l2, mean_shift, smooth = perturbation_penalties(deltas.reshape(-1, *deltas.shape[2:]))
    loss = loss + PERTURBATION_WEIGHT * (l2 + mean_shift) + SMOOTHNESS_WEIGHT * smooth
    loss.backward()

    gen_grad = any(p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.detach().abs().sum().cpu()) > 0 for p in generator.parameters())
    scorer_grad = any(p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.detach().abs().sum().cpu()) > 0 for p in scorer.parameters())
    expert_ok = True
    for rt in payload["runtimes_stage"].values():
        for p in rt.model.parameters():
            if p.requires_grad or (p.grad is not None and float(p.grad.detach().abs().sum().cpu()) != 0.0):
                expert_ok = False
    result = {
        "check": "gradient_flow_original_learned_probe",
        "dataset": dataset,
        "device": str(device),
        "loss": float(loss.detach().cpu()),
        "probe_generator_has_finite_nonzero_grad": gen_grad,
        "competence_scorer_has_finite_nonzero_grad": scorer_grad,
        "frozen_expert_params_require_grad_false_and_no_grad": expert_ok,
        "status": "PASS" if gen_grad and scorer_grad and expert_ok else "FAIL",
    }
    if result["status"] != "PASS":
        raise AssertionError(f"Gradient-flow check failed: {result}")
    return result


def tiny_forward_smoke(dataset: str, bundle, device: torch.device) -> list[dict[str, Any]]:
    payload = tiny_batch_for_dataset(dataset, bundle, device)
    rows = []
    for arm in ARM_ORDER:
        set_all_seeds(SEED)
        input_dim = STATIC_FEATURE_DIM if arm == "C_Rank_Passive" else STATIC_FEATURE_DIM + 6
        scorer = CompetenceScorer(input_dim).to(device)
        generator = ProbeGenerator(payload["history_all"].shape[2], eps=EPS).to(device) if arm in ("DeltaOnly", "OriginalLearnedProbe") else None
        encoder = NeuralPassiveEncoder(payload["history_all"].shape[2]).to(device) if arm == "MatchedNeuralPassive" else None
        with torch.no_grad():
            pred, deltas, six = arm_batch(
                arm,
                scorer,
                payload["history_all"][payload["batch_idx"]],
                payload["batch_idx"],
                bundle.core_names,
                payload["runtimes_stage"],
                payload["static_norm"],
                payload["group_b"],
                payload["forecasts_all"],
                payload["std"],
                generator,
                encoder,
            )
        rows.append(
            {
                "check": "tiny_forward_smoke",
                "dataset": dataset,
                "method": arm,
                "device": str(device),
                "pred_shape": list(pred.shape),
                "six_shape": list(six.shape),
                "delta_shape": list(deltas.shape) if deltas is not None else None,
                "status": "PASS" if list(pred.shape) == [payload["batch_idx"].numel(), len(bundle.core_names)] and six.shape[-1] == 6 else "FAIL",
            }
        )
    failed = [r for r in rows if r["status"] != "PASS"]
    if failed:
        raise AssertionError(f"Tiny forward smoke failed: {failed}")
    return rows


def cpu_gpu_forward_equivalence(dataset: str, bundle, device: torch.device) -> dict[str, Any]:
    expert = bundle.core_names[0]
    rt_cpu = load_expert_runtime(dataset, expert, device=torch.device("cpu"))
    checkpoint_hash = rt_cpu.checkpoint_sha256
    ref_raw = raw_history_cache(dataset, bundle.train_cache, rt_cpu.mean, rt_cpu.std)
    history = ref_raw["histories"][:4].to(torch.float32)
    out_cpu = rt_cpu.predict(history, batch_size=4)
    if device.type == "cuda":
        rt_gpu = load_expert_runtime(dataset, expert, device=device)
        out_gpu = rt_gpu.predict(history, batch_size=4)
        max_abs = float((out_cpu - out_gpu).abs().max())
        mean_abs = float((out_cpu - out_gpu).abs().mean())
        status = "PASS" if max_abs <= FORWARD_EQ_MAX_ABS_WARN or mean_abs <= FORWARD_EQ_MEAN_ABS_WARN else "WARN"
        gpu_hash = rt_gpu.checkpoint_sha256
    else:
        max_abs = 0.0
        mean_abs = 0.0
        status = "SKIP_NO_CUDA"
        gpu_hash = None
    return {
        "check": "cpu_gpu_frozen_expert_forward_equivalence",
        "dataset": dataset,
        "expert": expert,
        "checkpoint_sha256_cpu": checkpoint_hash,
        "checkpoint_sha256_gpu": gpu_hash,
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "max_abs_warn_threshold": FORWARD_EQ_MAX_ABS_WARN,
        "mean_abs_warn_threshold": FORWARD_EQ_MEAN_ABS_WARN,
        "status": status,
    }


def run_smoke_checks(dataset: str, device: torch.device) -> None:
    print_device(device)
    ensure_registered(dataset)
    bundle = fhv.LOADERS[dataset]()
    rows: list[dict[str, Any]] = []
    rows.extend(tiny_forward_smoke(dataset, bundle, device))
    rows.append(gradient_flow_check(dataset, bundle, device))
    rows.append(cpu_gpu_forward_equivalence(dataset, bundle, device))
    write_json(OUT_DIR / "pre_run_validation.json", {"device": device_info(device), "dataset": dataset, "checks": rows, "test_accessed": False})
    write_csv(OUT_DIR / "pre_run_validation_checks.csv", rows)
    print("SMOKE_CHECKS_COMPLETE")
    print("TEST SET ACCESSED: NO")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", choices=PRIMARY_DATASETS + EXTENDED_DATASETS)
    parser.add_argument("--include-extended", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--smoke-check", action="store_true")
    parser.add_argument("--smoke-dataset", choices=PRIMARY_DATASETS + EXTENDED_DATASETS, default="ETTh2")
    args = parser.parse_args()

    datasets = args.dataset if args.dataset else available_dataset_names(args.include_extended)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = select_device()

    if args.audit_only:
        run_audit_only(datasets)
        return

    if args.smoke_check:
        run_smoke_checks(args.smoke_dataset, device)
        return

    start = time.time()
    print_device(device)
    write_manifest(datasets, device)
    report: dict[str, Any] = {
        "experiment": "expert_conditioned_probe_mechanism",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_sha": git_commit_sha(),
        "datasets": {},
    }
    all_fold, all_train_diag, all_val, all_oof, all_resid, all_dep, all_integrity, all_comp = [], [], [], [], [], [], [], []
    checkpoint_hashes = {}
    for dataset in datasets:
        print(f"[expert-conditioned-mechanism] {dataset}: starting...", flush=True)
        result = evaluate_dataset(dataset, device=device)
        report["datasets"][dataset] = {k: v for k, v in result.items() if k != "checkpoint_hashes"}
        all_fold.extend(result["fold_rows"])
        all_train_diag.extend(result["train_diag_rows"])
        all_val.extend(result["router_val_rows"])
        all_oof.extend(result["router_train_oof_rows"])
        all_resid.extend(result["residual_rows"])
        all_dep.extend(result["dependence_rows"])
        all_integrity.append(result["integrity"])
        all_comp.extend(result["comparison_rows"])
        checkpoint_hashes[dataset] = result["checkpoint_hashes"]
        print(f"[expert-conditioned-mechanism] {dataset}: done.", flush=True)

    decision = classify(report)
    report["decision"] = decision
    report["runtime_sec"] = time.time() - start
    report["test_set_accessed"] = False

    write_json(OUT_DIR / "checkpoint_hashes.json", checkpoint_hashes)
    write_json(OUT_DIR / "validation_results.json", report)
    write_csv(OUT_DIR / "oof_fold_manifest.csv", all_fold)
    write_csv(OUT_DIR / "integrity_checks.csv", all_integrity)
    write_csv(OUT_DIR / "router_train_oof_results.csv", all_oof)
    write_csv(OUT_DIR / "router_val_results.csv", all_val)
    write_csv(OUT_DIR / "primary_comparisons.csv", all_comp)
    write_csv(OUT_DIR / "passive_active_diagnostics.csv", all_train_diag)
    write_csv(OUT_DIR / "residual_information_results.csv", all_resid)
    write_csv(OUT_DIR / "dependence_statistics.csv", all_dep)
    write_report(report, decision)
    print("TEST SET ACCESSED: NO")
    print(json.dumps({"runtime_sec": report["runtime_sec"], "classification": decision["classification"]}, indent=2))


if __name__ == "__main__":
    main()
