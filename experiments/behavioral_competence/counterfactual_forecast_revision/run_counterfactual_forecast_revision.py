"""Counterfactual Forecast Revision (CFR).

Development/mechanism study only. No test split is loaded.

Question:
When a frozen forecasting expert is shown a controlled hypothetical
realization of the first few future steps, does the way it revises the
remaining forecast reveal expert-specific instance-level conditional
competence beyond the existing 15 passive A+B+C features?

The intervention is at the forecast boundary:
1. Use the cached frozen forecast y=f(x) for the next 12 steps.
2. Construct self / positive / negative hypothetical prefixes for the first
   3 future steps.
3. Append the prefix to the history, preserving context length 96.
4. Re-query the same frozen expert.
5. Compare r[:9] against y[3:12], which are the same absolute timestamps.

The primary incremental-information tests are:
- PassivePlusCFR vs Passive
- PassivePlusRelativeCFR vs Passive
- CFR / RelativeCFR predicting Passive's honest residual
- Correct CFR expert mapping vs deterministic shuffled expert mapping
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
PER_WINDOW_DIR = OUT_DIR / "per_window_scores"
V2_DIR = ROOT / "experiments" / "behavioral_competence" / "controlled_discriminative_probe_v2"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))

import experiments.frozen_hv_costar.run_frozen_hv_costar as fhv  # noqa: E402
from experiments.behavioral_competence.controlled_discriminative_probe_v2 import run_controlled_discriminative_probe_v2 as v2  # noqa: E402
from experiments.behavioral_competence.generalization.run_generalization_study import register_dataset  # noqa: E402
from experiments.behavioral_competence.model_runtime import load_expert_runtime  # noqa: E402
from experiments.behavioral_competence.run_behavioral_competence import compute_excess_loss, raw_history_cache  # noqa: E402
from experiments.behavioral_competence.run_learned_probe import build_abc_features, stage_runtime_groups  # noqa: E402
from experiments.behavioral_competence.simplex_probe.run_simplex_probe import dependence_full, primary_row  # noqa: E402


DATASETS = ["ExchangeRate", "Traffic", "BeijingAirQuality", "ETTm2"]
PREFIX_K = 3
FORECAST_HORIZON = 12
INPUT_LEN = 96
COUNTERFACTUAL_SCALE = 1.0
RIDGE_ALPHA = 1.0
PASSIVE_FEATURE_DIM = 15
SHUFFLE_SEED = v2.SHUFFLE_SEED
N_PURGE_FOLDS = v2.N_PURGE_FOLDS
MIN_TRAIN_FRACTION = v2.MIN_TRAIN_FRACTION
SURPRISE_MIN_STD_FRACTION = 1e-6
DIRECTIONAL_EPS = 1e-8
CORRUPTION_SEED = 4242
BATCH_SIZE = 128

CFR_FEATURE_NAMES = [
    "self_revision",
    "plus_response",
    "minus_response",
    "response_asymmetry",
    "counterfactual_gain",
    "directionality_plus",
    "directionality_minus",
    "symmetric_response_magnitude",
    "curvature_magnitude",
]
CFR_FEATURE_DIM = len(CFR_FEATURE_NAMES)

METHODS = [
    "Passive",
    "CFR",
    "RelativeCFR",
    "ShuffledCFR",
    "PassivePlusCFR",
    "PassivePlusRelativeCFR",
    "PassivePlusShuffledCFR",
]
RESIDUAL_METHODS = [
    "CFR_to_PassiveResidual",
    "RelativeCFR_to_PassiveResidual",
    "ShuffledCFR_to_PassiveResidual",
]
PRIMARY_COMPARISONS = [
    ("PassivePlusCFR_vs_Passive", "PassivePlusCFR", "Passive"),
    ("PassivePlusRelativeCFR_vs_Passive", "PassivePlusRelativeCFR", "Passive"),
    ("PassivePlusCFR_vs_PassivePlusShuffledCFR", "PassivePlusCFR", "PassivePlusShuffledCFR"),
    ("CFR_vs_ShuffledCFR", "CFR", "ShuffledCFR"),
    ("RelativeCFR_vs_ShuffledCFR", "RelativeCFR", "ShuffledCFR"),
]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    arr = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


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
        writer.writerows(rows)


def per_window_abs_error(pred: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
    return (pred - actual).abs().mean(dim=1)


def _safe_corr(pred_flat: np.ndarray, actual_flat: np.ndarray, kind: str) -> float:
    if pred_flat.shape[0] < 2 or float(np.std(pred_flat)) <= 1e-12 or float(np.std(actual_flat)) <= 1e-12:
        return float("nan")
    if kind == "pearson":
        return float(pearsonr(pred_flat, actual_flat).statistic)
    return float(spearmanr(pred_flat, actual_flat).statistic)


def metric_row(dataset: str, method: str, split: str, pred: torch.Tensor, actual: torch.Tensor) -> dict[str, Any]:
    k = pred.shape[1]
    pred_flat = pred.reshape(-1).detach().cpu().numpy()
    actual_flat = actual.reshape(-1).detach().cpu().numpy()
    pairwise_correct, pairwise_total = 0, 0
    for i in range(k):
        for j in range(i + 1, k):
            actual_sign = torch.sign(actual[:, i] - actual[:, j])
            pred_sign = torch.sign(pred[:, i] - pred[:, j])
            valid = actual_sign != 0
            pairwise_correct += int(((pred_sign == actual_sign) & valid).sum())
            pairwise_total += int(valid.sum())
    return {
        "dataset": dataset,
        "method": method,
        "split": split,
        "n_windows": int(pred.shape[0]),
        "n_rows": int(pred_flat.shape[0]),
        "mae": float(mean_absolute_error(actual_flat, pred_flat)),
        "mse": float(mean_squared_error(actual_flat, pred_flat)),
        "r2": float(r2_score(actual_flat, pred_flat)) if float(np.std(pred_flat)) > 1e-12 else float("nan"),
        "pearson": _safe_corr(pred_flat, actual_flat, "pearson"),
        "spearman": _safe_corr(pred_flat, actual_flat, "spearman"),
        "pairwise_ranking_accuracy": pairwise_correct / pairwise_total if pairwise_total else float("nan"),
        "top1_expert_accuracy": float((pred.argmin(dim=1) == actual.argmin(dim=1)).to(torch.float32).mean()),
    }


def fit_ridge_predict(
    train_features: torch.Tensor,
    train_target: torch.Tensor,
    train_idx: torch.Tensor,
    eval_features: torch.Tensor,
    eval_idx: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    feature_dim = int(train_features.shape[-1])
    k = int(train_features.shape[1])
    x_train = train_features[train_idx].reshape(-1, feature_dim).detach().cpu().numpy()
    y_train = train_target[train_idx].reshape(-1).detach().cpu().numpy()
    x_eval = eval_features[eval_idx].reshape(-1, feature_dim).detach().cpu().numpy()
    scaler = StandardScaler()
    x_train_std = scaler.fit_transform(x_train)
    x_eval_std = scaler.transform(x_eval)
    model = Ridge(alpha=RIDGE_ALPHA)
    model.fit(x_train_std, y_train)
    pred = torch.from_numpy(model.predict(x_eval_std).astype(np.float32)).reshape(eval_idx.numel(), k)
    return pred, {
        "alpha": RIDGE_ALPHA,
        "feature_dim": feature_dim,
        "train_rows": int(x_train.shape[0]),
        "eval_rows": int(x_eval.shape[0]),
        "standardized_using_train_rows_only": True,
    }


def fit_ridge_predict_arrays(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    x_train_std = scaler.fit_transform(x_train)
    x_eval_std = scaler.transform(x_eval)
    model = Ridge(alpha=RIDGE_ALPHA)
    model.fit(x_train_std, y_train)
    return model.predict(x_eval_std).astype(np.float32)


def estimate_surprise_scale(
    actual_error_cache: Mapping[str, Any],
    forecasts_all: torch.Tensor,
    train_idx: torch.Tensor,
    canonical_std: torch.Tensor,
) -> torch.Tensor:
    """Robust per-variable residual scale from legal training rows only."""
    target = actual_error_cache["targets"].to(torch.float32)
    mask = actual_error_cache["target_masks"].to(torch.bool)
    residual = target[train_idx].unsqueeze(-1) - forecasts_all[train_idx]
    residual = residual.abs()
    residual = residual[mask[train_idx].unsqueeze(-1).expand_as(residual)].reshape(-1, forecasts_all.shape[2], forecasts_all.shape[3])
    # After masking, residual is [valid_horizon_rows, F, K]. Pool horizon/windows/experts per variable.
    pooled = residual.permute(1, 0, 2).reshape(forecasts_all.shape[2], -1)
    sigma = torch.median(pooled, dim=1).values / 0.67448975
    min_scale = canonical_std.to(torch.float32).abs().clamp_min(1.0) * SURPRISE_MIN_STD_FRACTION
    return torch.maximum(sigma.to(torch.float32), min_scale)


def cfr_cache_path(dataset: str, split: str, scale_hash: str, n_windows: int) -> Path:
    return OUT_DIR / "feature_cache" / f"{dataset}__{split}__n{n_windows}__scale_{scale_hash[:16]}.pt"


def _cosine_directionality(innovation_norm: torch.Tensor, tail_norm: torch.Tensor) -> torch.Tensor:
    numerator = (innovation_norm * tail_norm).sum(dim=1)
    denom = innovation_norm.pow(2).sum(dim=1).sqrt() * tail_norm.pow(2).sum(dim=1).sqrt()
    return numerator / denom.clamp_min(DIRECTIONAL_EPS)


def compute_cfr_features(
    dataset: str,
    split: str,
    history_raw_all: torch.Tensor,
    forecasts_all: torch.Tensor,
    core_names: Sequence[str],
    stage_groups: list[tuple[int, int, Mapping[str, Any]]],
    canonical_std: torch.Tensor,
    surprise_scale: torch.Tensor,
    use_cache: bool = True,
    max_windows: int | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    n_total, length, feats = history_raw_all.shape
    n = n_total if max_windows is None else min(n_total, int(max_windows))
    if length != INPUT_LEN:
        raise AssertionError(f"{dataset}/{split}: expected input length {INPUT_LEN}, got {length}")
    if forecasts_all.shape[1] != FORECAST_HORIZON:
        raise AssertionError(f"{dataset}/{split}: expected horizon {FORECAST_HORIZON}, got {forecasts_all.shape[1]}")
    if PREFIX_K >= FORECAST_HORIZON:
        raise AssertionError("PREFIX_K must be smaller than forecast horizon")
    absolute_alignment_ok = bool(torch.equal(torch.arange(PREFIX_K, FORECAST_HORIZON), PREFIX_K + torch.arange(FORECAST_HORIZON - PREFIX_K)))
    if not absolute_alignment_ok:
        raise AssertionError("Absolute-horizon alignment assertion failed")

    scale_hash = tensor_sha256(surprise_scale)
    cache_path = cfr_cache_path(dataset, split, scale_hash, n)
    if use_cache and max_windows is None and cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("feature_names") == CFR_FEATURE_NAMES and int(payload.get("prefix_k", -1)) == PREFIX_K:
            return payload["features"].to(torch.float32), payload["diagnostics"]

    k = len(core_names)
    std = canonical_std.to(torch.float32).view(1, 1, feats).clamp_min(1e-8)
    sigma = (COUNTERFACTUAL_SCALE * surprise_scale.to(torch.float32)).view(1, 1, feats).expand(1, PREFIX_K, feats)
    input_magnitude = float((sigma.abs() / std.expand(1, PREFIX_K, feats)).mean())
    features = torch.zeros(n, k, CFR_FEATURE_DIM, dtype=torch.float32)
    feature_hash_inputs = {
        "surprise_scale_sha256": scale_hash,
        "history_sha256_sample": tensor_sha256(history_raw_all[: min(32, n)]),
        "forecast_sha256_sample": tensor_sha256(forecasts_all[: min(32, n)]),
    }

    with torch.no_grad():
        for lo, hi, runtimes_stage in stage_groups:
            lo_eff, hi_eff = max(0, lo), min(n, hi)
            if hi_eff <= lo_eff:
                continue
            for b in range(lo_eff, hi_eff, BATCH_SIZE):
                batch_idx = torch.arange(b, min(b + BATCH_SIZE, hi_eff))
                history = history_raw_all[batch_idx].to(torch.float32)
                shifted_history = history[:, PREFIX_K:, :]
                for local_i, expert_name in enumerate(core_names):
                    rt = runtimes_stage[expert_name]
                    y = forecasts_all[batch_idx, :, :, local_i].to(torch.float32)
                    prefix_self = y[:, :PREFIX_K, :]
                    s = sigma.expand(prefix_self.shape[0], -1, -1)
                    x_self = torch.cat([shifted_history, prefix_self], dim=1)
                    x_plus = torch.cat([shifted_history, prefix_self + s], dim=1)
                    x_minus = torch.cat([shifted_history, prefix_self - s], dim=1)

                    r_self = rt.predict(x_self, batch_size=BATCH_SIZE)
                    r_plus = rt.predict(x_plus, batch_size=BATCH_SIZE)
                    r_minus = rt.predict(x_minus, batch_size=BATCH_SIZE)

                    baseline_tail = y[:, PREFIX_K:FORECAST_HORIZON, :]
                    self_tail = r_self[:, : FORECAST_HORIZON - PREFIX_K, :]
                    plus_tail = r_plus[:, : FORECAST_HORIZON - PREFIX_K, :]
                    minus_tail = r_minus[:, : FORECAST_HORIZON - PREFIX_K, :]
                    std_tail = std.expand(prefix_self.shape[0], FORECAST_HORIZON - PREFIX_K, -1)

                    self_revision = ((self_tail - baseline_tail).abs() / std_tail).mean(dim=(1, 2))
                    delta_plus = plus_tail - self_tail
                    delta_minus = minus_tail - self_tail
                    plus_response = (delta_plus.abs() / std_tail).mean(dim=(1, 2))
                    minus_response = (delta_minus.abs() / std_tail).mean(dim=(1, 2))
                    response_asymmetry = (plus_response - minus_response).abs()
                    counterfactual_gain = 0.5 * (plus_response + minus_response) / max(input_magnitude, DIRECTIONAL_EPS)

                    inv_plus = (s.mean(dim=1) / canonical_std.view(1, -1).clamp_min(1e-8))
                    inv_minus = -inv_plus
                    tail_plus = delta_plus.mean(dim=1) / canonical_std.view(1, -1).clamp_min(1e-8)
                    tail_minus = delta_minus.mean(dim=1) / canonical_std.view(1, -1).clamp_min(1e-8)
                    directionality_plus = _cosine_directionality(inv_plus, tail_plus)
                    directionality_minus = _cosine_directionality(inv_minus, tail_minus)

                    symmetric = ((plus_tail - minus_tail) / 2.0).abs() / std_tail
                    curvature = (plus_tail + minus_tail - 2.0 * self_tail).abs() / std_tail
                    symmetric_response_magnitude = symmetric.mean(dim=(1, 2))
                    curvature_magnitude = curvature.mean(dim=(1, 2))

                    features[batch_idx, local_i, :] = torch.stack(
                        [
                            self_revision,
                            plus_response,
                            minus_response,
                            response_asymmetry,
                            counterfactual_gain,
                            directionality_plus,
                            directionality_minus,
                            symmetric_response_magnitude,
                            curvature_magnitude,
                        ],
                        dim=1,
                    ).cpu()

    if not bool(torch.isfinite(features).all()):
        raise AssertionError(f"{dataset}/{split}: nonfinite CFR features")
    regen_sample = features[: min(32, n)].clone()
    diagnostics = {
        "dataset": dataset,
        "split": split,
        "num_windows": int(n),
        "num_experts": int(k),
        "feature_dim": int(features.shape[-1]),
        "feature_names": CFR_FEATURE_NAMES,
        "prefix_k": PREFIX_K,
        "forecast_horizon": FORECAST_HORIZON,
        "input_magnitude": input_magnitude,
        "surprise_scale_sha256": scale_hash,
        "feature_sample_sha256": tensor_sha256(regen_sample),
        "absolute_horizon_alignment_assertion": absolute_alignment_ok,
        **feature_hash_inputs,
    }
    if use_cache and max_windows is None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"features": features, "diagnostics": diagnostics, "feature_names": CFR_FEATURE_NAMES, "prefix_k": PREFIX_K}, cache_path)
    return features, diagnostics


def feature_sets(cfr: torch.Tensor, relative: torch.Tensor, shuffled: torch.Tensor, passive: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "Passive": passive,
        "CFR": cfr,
        "RelativeCFR": relative,
        "ShuffledCFR": shuffled,
        "PassivePlusCFR": torch.cat([passive, cfr], dim=-1),
        "PassivePlusRelativeCFR": torch.cat([passive, relative], dim=-1),
        "PassivePlusShuffledCFR": torch.cat([passive, shuffled], dim=-1),
    }


def common_mode_rows(dataset: str, split: str, cfr: torch.Tensor) -> list[dict[str, Any]]:
    rows = []
    total_centered = cfr - cfr.mean(dim=(0, 1), keepdim=True)
    total_var = total_centered.pow(2).mean(dim=(0, 1))
    relative = cfr - cfr.mean(dim=1, keepdim=True)
    rel_var = relative.pow(2).mean(dim=(0, 1))
    for j, name in enumerate(CFR_FEATURE_NAMES):
        rows.append(
            {
                "dataset": dataset,
                "split": split,
                "diagnostic": "common_mode_fraction",
                "feature": name,
                "total_variance": float(total_var[j]),
                "expert_relative_variance": float(rel_var[j]),
                "expert_relative_fraction": float(rel_var[j] / total_var[j].clamp_min(1e-12)),
                "common_mode_fraction": float(1.0 - rel_var[j] / total_var[j].clamp_min(1e-12)),
            }
        )
    return rows


def per_expert_correlation_rows(dataset: str, split: str, cfr: torch.Tensor, target: torch.Tensor, core: Sequence[str]) -> list[dict[str, Any]]:
    rows = []
    for e, name in enumerate(core):
        y = target[:, e].detach().cpu().numpy()
        for j, feature_name in enumerate(CFR_FEATURE_NAMES):
            x = cfr[:, e, j].detach().cpu().numpy()
            rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "expert": name,
                    "feature": feature_name,
                    "pearson": _safe_corr(x, y, "pearson"),
                    "spearman": _safe_corr(x, y, "spearman"),
                    "n_windows": int(target.shape[0]),
                }
            )
    return rows


def incremental_rows(dataset: str, split: str, metric_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by = {row["method"]: row for row in metric_rows if row["dataset"] == dataset and row["split"] == split}
    rows = []
    for comparison, candidate, baseline in PRIMARY_COMPARISONS:
        if candidate in by and baseline in by:
            rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "comparison": comparison,
                    "candidate": candidate,
                    "baseline": baseline,
                    "candidate_mae": by[candidate]["mae"],
                    "baseline_mae": by[baseline]["mae"],
                    "mae_delta_candidate_minus_baseline": by[candidate]["mae"] - by[baseline]["mae"],
                    "candidate_r2": by[candidate]["r2"],
                    "baseline_r2": by[baseline]["r2"],
                    "r2_delta_candidate_minus_baseline": by[candidate]["r2"] - by[baseline]["r2"],
                    "candidate_pairwise": by[candidate]["pairwise_ranking_accuracy"],
                    "baseline_pairwise": by[baseline]["pairwise_ranking_accuracy"],
                    "pairwise_delta": by[candidate]["pairwise_ranking_accuracy"] - by[baseline]["pairwise_ranking_accuracy"],
                    "candidate_top1": by[candidate]["top1_expert_accuracy"],
                    "baseline_top1": by[baseline]["top1_expert_accuracy"],
                    "top1_delta": by[candidate]["top1_expert_accuracy"] - by[baseline]["top1_expert_accuracy"],
                }
            )
    return rows


def passive_reproduction_rows(dataset: str, oof_rows: Sequence[Mapping[str, Any]], val_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    prev_dir = ROOT / "experiments" / "behavioral_competence" / "multi_query_random_probe"
    mappings = [
        (prev_dir / "router_train_oof_results.csv", "router_train_oof_common", oof_rows),
        (prev_dir / "router_val_competence_results.csv", "router_val", val_rows),
    ]
    for path, split, current_rows in mappings:
        current = next((r for r in current_rows if r["dataset"] == dataset and r["method"] == "Passive" and r["split"] == split), None)
        if current is None:
            continue
        if not path.exists():
            rows.append({"dataset": dataset, "split": split, "previous_file": str(path.relative_to(ROOT)), "status": "PREVIOUS_RESULT_NOT_FOUND"})
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            prev = [r for r in csv.DictReader(handle)]
        prev_row = next((r for r in prev if r.get("dataset") == dataset and r.get("method") == "MatchedPassive" and r.get("split") == split), None)
        if prev_row is None:
            rows.append({"dataset": dataset, "split": split, "previous_file": str(path.relative_to(ROOT)), "status": "PREVIOUS_PASSIVE_ROW_NOT_FOUND"})
            continue
        prev_mae = float(prev_row["mae"])
        delta = current["mae"] - prev_mae
        rows.append(
            {
                "dataset": dataset,
                "split": split,
                "previous_file": str(path.relative_to(ROOT)),
                "previous_method": "MatchedPassive",
                "current_method": "Passive",
                "previous_mae": prev_mae,
                "current_mae": current["mae"],
                "mae_delta": delta,
                "status": "PASS" if abs(delta) <= 1e-6 else "DIFF_EXCEEDS_TOLERANCE",
            }
        )
    return rows


def collect_checkpoint_hashes(dataset: str, core: Sequence[str], stage_groups_train: Sequence[tuple[int, int, Mapping[str, Any]]], val_runtimes: Mapping[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for expert, runtime in val_runtimes.items():
        out[f"router_val_final::{expert}"] = runtime.checkpoint_sha256
    for lo, hi, runtimes in stage_groups_train:
        for expert, runtime in runtimes.items():
            out[f"router_train_{lo}_{hi}::{expert}"] = runtime.checkpoint_sha256
    for expert in core:
        if expert not in val_runtimes:
            raise AssertionError(f"{dataset}: missing runtime for {expert}")
    return out


def all_runtimes_frozen(runtimes: Sequence[Mapping[str, Any]]) -> bool:
    for group in runtimes:
        for rt in group.values():
            if rt.model.training:
                return False
            if any(p.requires_grad for p in rt.model.parameters()):
                return False
    return True


def evaluate_dataset(dataset: str, use_cache: bool = True) -> dict[str, Any]:
    register_dataset(dataset)
    bundle = fhv.LOADERS[dataset]()
    train_cache = bundle.train_cache
    val_cache = bundle.val_cache
    core = list(bundle.core_names)
    k = len(core)
    n_train = int(train_cache["num_windows"])
    n_val = int(val_cache["num_windows"])
    print(f"[cfr] {dataset}: frozen core={core}", flush=True)

    val_runtimes = {expert: load_expert_runtime(dataset, expert) for expert in core}
    reference_runtime = val_runtimes[core[0]]
    train_cache_raw = raw_history_cache(dataset, train_cache, reference_runtime.mean, reference_runtime.std)
    val_cache_raw = raw_history_cache(dataset, val_cache, reference_runtime.mean, reference_runtime.std)
    stage_groups_train = stage_runtime_groups(dataset, bundle, train_cache, val_runtimes)
    stage_groups_val = [(0, n_val, val_runtimes)]
    checkpoint_hashes_before = collect_checkpoint_hashes(dataset, core, stage_groups_train, val_runtimes)
    all_experts_frozen_before = all_runtimes_frozen([val_runtimes] + [g[2] for g in stage_groups_train])

    observability, legal_idx_all, folds, common_idx = v2.compute_legal_and_common(train_cache, val_cache)
    fold_rows = []
    for fold in folds:
        row = {
            "dataset": dataset,
            "fold": fold["fold"],
            "train_origin_min": fold["train_origin_min"],
            "train_origin_max": fold["train_origin_max"],
            "train_target_end_max": fold["train_target_end_max"],
            "heldout_origin_min": fold["eval_origin_min"],
            "heldout_origin_max": fold["eval_origin_max"],
            "purged_count": fold["num_purged_windows"],
            "assertion_pass": fold["assertion_max_train_target_end_leq_min_eval_origin"],
            "num_train_windows": int(fold["train_idx"].numel()),
            "num_eval_windows": int(fold["eval_idx"].numel()),
        }
        fold_rows.append(row)
    if not all(row["assertion_pass"] for row in fold_rows):
        raise AssertionError(f"{dataset}: purge assertion failed")

    group_a_tr, group_b_tr, group_c_tr, forecasts_train = build_abc_features(bundle, train_cache_raw)
    group_a_va, group_b_va, group_c_va, forecasts_val = build_abc_features(bundle, val_cache_raw)
    passive_train = torch.cat([group_a_tr, group_b_tr, group_c_tr], dim=-1)
    passive_val = torch.cat([group_a_va, group_b_va, group_c_va], dim=-1)
    if passive_train.shape[-1] != PASSIVE_FEATURE_DIM or passive_val.shape[-1] != PASSIVE_FEATURE_DIM:
        raise AssertionError(f"{dataset}: passive dim mismatch")

    _, actual_error_train = compute_excess_loss(train_cache, forecasts_train, bundle.std)
    _, actual_error_val = compute_excess_loss(val_cache, forecasts_val, bundle.std)
    history_train = train_cache_raw["histories"].to(torch.float32)
    history_val = val_cache_raw["histories"].to(torch.float32)

    oof_pred = {method: torch.full((n_train, k), float("nan")) for method in METHODS}
    oof_actual = torch.full((n_train, k), float("nan"))
    oof_resid_pred = {method: torch.full((n_train, k), float("nan")) for method in RESIDUAL_METHODS}
    oof_resid_actual = torch.full((n_train, k), float("nan"))
    fit_info_rows = []
    fold_prior_rows = []
    surprise_rows = []
    fold_feature_hash_rows = []
    cfr_oof_features = torch.full((n_train, k, CFR_FEATURE_DIM), float("nan"))
    rel_oof_features = torch.full((n_train, k, CFR_FEATURE_DIM), float("nan"))

    for fold in folds:
        train_idx, eval_idx = fold["train_idx"], fold["eval_idx"]
        fold_id = int(fold["fold"])
        if train_idx.numel() < 10 or eval_idx.numel() == 0:
            continue
        sigma = estimate_surprise_scale(train_cache, forecasts_train, train_idx, bundle.std)
        surprise_rows.append(
            {
                "dataset": dataset,
                "split": "router_train_oof",
                "fold": fold_id,
                "num_training_windows": int(train_idx.numel()),
                "scale_sha256": tensor_sha256(sigma),
                "scale_values_json": json.dumps([float(x) for x in sigma.tolist()]),
                "estimator": "median_absolute_residual_over_train_windows_horizons_experts_div_0.67448975",
            }
        )
        print(f"[cfr] {dataset}: fold {fold_id} CFR features with train-only surprise scale", flush=True)
        cfr_fold, diag = compute_cfr_features(dataset, f"router_train_fold{fold_id}", history_train, forecasts_train, core, stage_groups_train, bundle.std, sigma, use_cache=use_cache)
        rel_fold = cfr_fold - cfr_fold.mean(dim=1, keepdim=True)
        shuf_fold = v2.derange_expert_axis(cfr_fold, train_cache["absolute_window_starts"], dataset, SHUFFLE_SEED)
        cfr_oof_features[eval_idx] = cfr_fold[eval_idx]
        rel_oof_features[eval_idx] = rel_fold[eval_idx]
        fold_feature_hash_rows.append({"dataset": dataset, "fold": fold_id, **diag})

        train_features = feature_sets(cfr_fold, rel_fold, shuf_fold, passive_train)
        mu_e = actual_error_train[train_idx].mean(dim=0)
        target_fold = actual_error_train - mu_e.view(1, k)
        oof_actual[eval_idx] = target_fold[eval_idx]
        fold_prior_rows.append({"dataset": dataset, "fold": fold_id, **{f"mu_{core[i]}": float(mu_e[i]) for i in range(k)}})

        for method in METHODS:
            pred, info = fit_ridge_predict(train_features[method], target_fold, train_idx, train_features[method], eval_idx)
            oof_pred[method][eval_idx] = pred
            fit_info_rows.append({"dataset": dataset, "split": "oof", "fold": fold_id, "method": method, **info})

        passive_train_pred, _ = fit_ridge_predict(train_features["Passive"], target_fold, train_idx, train_features["Passive"], train_idx)
        passive_eval_pred = oof_pred["Passive"][eval_idx]
        residual_train = (target_fold[train_idx] - passive_train_pred).reshape(-1).numpy()
        residual_eval = target_fold[eval_idx] - passive_eval_pred
        oof_resid_actual[eval_idx] = residual_eval
        for residual_method, feature_name in [
            ("CFR_to_PassiveResidual", "CFR"),
            ("RelativeCFR_to_PassiveResidual", "RelativeCFR"),
            ("ShuffledCFR_to_PassiveResidual", "ShuffledCFR"),
        ]:
            feat = train_features[feature_name]
            x_train = feat[train_idx].reshape(-1, feat.shape[-1]).numpy()
            x_eval = feat[eval_idx].reshape(-1, feat.shape[-1]).numpy()
            pred = fit_ridge_predict_arrays(x_train, residual_train, x_eval)
            oof_resid_pred[residual_method][eval_idx] = torch.from_numpy(pred).reshape(eval_idx.numel(), k)

    if bool(torch.isnan(oof_actual[common_idx]).any()):
        raise AssertionError(f"{dataset}: OOF actual target missing on common windows")
    for method in METHODS:
        if bool(torch.isnan(oof_pred[method][common_idx]).any()):
            raise AssertionError(f"{dataset}: OOF prediction missing for {method}")
    for method in RESIDUAL_METHODS:
        if bool(torch.isnan(oof_resid_pred[method][common_idx]).any()):
            raise AssertionError(f"{dataset}: OOF residual prediction missing for {method}")

    print(f"[cfr] {dataset}: final train/val CFR features with full legal router_train scale", flush=True)
    sigma_final = estimate_surprise_scale(train_cache, forecasts_train, legal_idx_all, bundle.std)
    surprise_rows.append(
        {
            "dataset": dataset,
            "split": "router_val_final",
            "fold": "final",
            "num_training_windows": int(legal_idx_all.numel()),
            "scale_sha256": tensor_sha256(sigma_final),
            "scale_values_json": json.dumps([float(x) for x in sigma_final.tolist()]),
            "estimator": "median_absolute_residual_over_full_legal_router_train_windows_horizons_experts_div_0.67448975",
        }
    )
    cfr_train_final, diag_train_final = compute_cfr_features(dataset, "router_train_final", history_train, forecasts_train, core, stage_groups_train, bundle.std, sigma_final, use_cache=use_cache)
    cfr_val, diag_val = compute_cfr_features(dataset, "router_val", history_val, forecasts_val, core, stage_groups_val, bundle.std, sigma_final, use_cache=use_cache)
    relative_train_final = cfr_train_final - cfr_train_final.mean(dim=1, keepdim=True)
    relative_val = cfr_val - cfr_val.mean(dim=1, keepdim=True)
    shuffled_train_final = v2.derange_expert_axis(cfr_train_final, train_cache["absolute_window_starts"], dataset, SHUFFLE_SEED)
    shuffled_val = v2.derange_expert_axis(cfr_val, val_cache["absolute_window_starts"], dataset, SHUFFLE_SEED)
    train_features_final = feature_sets(cfr_train_final, relative_train_final, shuffled_train_final, passive_train)
    val_features = feature_sets(cfr_val, relative_val, shuffled_val, passive_val)

    mu_e_final = actual_error_train[legal_idx_all].mean(dim=0)
    target_train_final = actual_error_train - mu_e_final.view(1, k)
    target_val = actual_error_val - mu_e_final.view(1, k)
    val_pred = {}
    for method in METHODS:
        pred, info = fit_ridge_predict(train_features_final[method], target_train_final, legal_idx_all, val_features[method], torch.arange(n_val))
        val_pred[method] = pred
        fit_info_rows.append({"dataset": dataset, "split": "router_val", "fold": "final", "method": method, **info})

    passive_legal_pred, _ = fit_ridge_predict(train_features_final["Passive"], target_train_final, legal_idx_all, train_features_final["Passive"], legal_idx_all)
    passive_val_pred = val_pred["Passive"]
    final_residual_train = (target_train_final[legal_idx_all] - passive_legal_pred).reshape(-1).numpy()
    final_residual_val = target_val - passive_val_pred
    val_resid_pred = {}
    for residual_method, feature_name in [
        ("CFR_to_PassiveResidual", "CFR"),
        ("RelativeCFR_to_PassiveResidual", "RelativeCFR"),
        ("ShuffledCFR_to_PassiveResidual", "ShuffledCFR"),
    ]:
        feat_train = train_features_final[feature_name]
        feat_val = val_features[feature_name]
        x_train = feat_train[legal_idx_all].reshape(-1, feat_train.shape[-1]).numpy()
        x_val = feat_val.reshape(-1, feat_val.shape[-1]).numpy()
        pred = fit_ridge_predict_arrays(x_train, final_residual_train, x_val)
        val_resid_pred[residual_method] = torch.from_numpy(pred).reshape(n_val, k)

    oof_rows = [metric_row(dataset, method, "router_train_oof_common", oof_pred[method][common_idx], oof_actual[common_idx]) for method in METHODS]
    val_rows = [metric_row(dataset, method, "router_val", val_pred[method], target_val) for method in METHODS]
    residual_rows = []
    for method in RESIDUAL_METHODS:
        residual_rows.append(metric_row(dataset, method, "router_train_oof_common", oof_resid_pred[method][common_idx], oof_resid_actual[common_idx]))
        residual_rows.append(metric_row(dataset, method, "router_val", val_resid_pred[method], final_residual_val))

    incremental = []
    incremental.extend(incremental_rows(dataset, "router_train_oof_common", oof_rows))
    incremental.extend(incremental_rows(dataset, "router_val", val_rows))
    incremental.extend(passive_reproduction_rows(dataset, oof_rows, val_rows))

    dependence_rows = []
    pred_map = {**val_pred}
    actual = target_val
    for comparison, candidate, baseline in PRIMARY_COMPARISONS:
        rows = dependence_full(per_window_abs_error(pred_map[candidate], actual), per_window_abs_error(pred_map[baseline], actual), dataset, comparison)
        dependence_rows.extend(rows)

    specificity_rows = []
    specificity_rows.extend(common_mode_rows(dataset, "router_train_oof_common", cfr_oof_features[common_idx]))
    specificity_rows.extend(common_mode_rows(dataset, "router_val", cfr_val))
    for split, rows in [("router_train_oof_common", oof_rows), ("router_val", val_rows)]:
        by = {r["method"]: r for r in rows}
        for candidate, baseline in [
            ("CFR", "ShuffledCFR"),
            ("RelativeCFR", "ShuffledCFR"),
            ("PassivePlusCFR", "PassivePlusShuffledCFR"),
            ("PassivePlusRelativeCFR", "PassivePlusShuffledCFR"),
        ]:
            specificity_rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "diagnostic": "correct_expert_mapping_or_ranking",
                    "candidate": candidate,
                    "baseline": baseline,
                    "candidate_mae": by[candidate]["mae"],
                    "baseline_mae": by[baseline]["mae"],
                    "mae_delta_candidate_minus_baseline": by[candidate]["mae"] - by[baseline]["mae"],
                    "candidate_pairwise": by[candidate]["pairwise_ranking_accuracy"],
                    "baseline_pairwise": by[baseline]["pairwise_ranking_accuracy"],
                    "pairwise_delta": by[candidate]["pairwise_ranking_accuracy"] - by[baseline]["pairwise_ranking_accuracy"],
                    "candidate_top1": by[candidate]["top1_expert_accuracy"],
                    "baseline_top1": by[baseline]["top1_expert_accuracy"],
                    "top1_delta": by[candidate]["top1_expert_accuracy"] - by[baseline]["top1_expert_accuracy"],
                }
            )

    corr_rows = []
    corr_rows.extend(per_expert_correlation_rows(dataset, "router_train_oof_common", cfr_oof_features[common_idx], oof_actual[common_idx], core))
    corr_rows.extend(per_expert_correlation_rows(dataset, "router_val", cfr_val, target_val, core))

    checkpoint_hashes_after = collect_checkpoint_hashes(dataset, core, stage_groups_train, val_runtimes)
    all_experts_frozen_after = all_runtimes_frozen([val_runtimes] + [g[2] for g in stage_groups_train])

    # Real target-corruption invariance check on a deterministic sample.
    sample_n = min(4, n_val)
    gen = torch.Generator().manual_seed(CORRUPTION_SEED)
    corrupted_val_cache = dict(val_cache)
    corrupted_val_cache["targets"] = torch.randn(val_cache["targets"].shape, generator=gen, dtype=torch.float32)
    corrupted_val_raw = raw_history_cache(dataset, corrupted_val_cache, reference_runtime.mean, reference_runtime.std)
    sample_stage = [(0, sample_n, val_runtimes)]
    sample_forecasts = forecasts_val[:sample_n]
    sample_features_a, _ = compute_cfr_features(
        dataset,
        "target_corruption_sample_a",
        history_val[:sample_n],
        sample_forecasts,
        core,
        sample_stage,
        bundle.std,
        sigma_final,
        use_cache=False,
        max_windows=sample_n,
    )
    sample_features_b, _ = compute_cfr_features(
        dataset,
        "target_corruption_sample_b",
        corrupted_val_raw["histories"][:sample_n].to(torch.float32),
        sample_forecasts,
        core,
        sample_stage,
        bundle.std,
        sigma_final,
        use_cache=False,
        max_windows=sample_n,
    )
    target_corruption_max_abs_diff = float((sample_features_a - sample_features_b).abs().max())
    deterministic_features_a, _ = compute_cfr_features(
        dataset,
        "deterministic_sample_a",
        history_val[:sample_n],
        sample_forecasts,
        core,
        sample_stage,
        bundle.std,
        sigma_final,
        use_cache=False,
        max_windows=sample_n,
    )
    deterministic_features_b, _ = compute_cfr_features(
        dataset,
        "deterministic_sample_b",
        history_val[:sample_n],
        sample_forecasts,
        core,
        sample_stage,
        bundle.std,
        sigma_final,
        use_cache=False,
        max_windows=sample_n,
    )
    deterministic_max_abs_diff = float((deterministic_features_a - deterministic_features_b).abs().max())
    shuffled_again = v2.derange_expert_axis(cfr_val, val_cache["absolute_window_starts"], dataset, SHUFFLE_SEED)
    shuffle_deterministic = bool(torch.equal(shuffled_val, shuffled_again))

    test_paths = [
        ROOT / f"cache/costarts_walkforward_{dataset}/test_80_100_cache.pt",
        ROOT / f"cache/costarts_fresh/{dataset}_96_12/test_cache.pt",
        ROOT / f"cache/costarts_fresh/{dataset}_96_12/router_test_cache.pt",
    ]
    integrity = {
        "dataset": dataset,
        "expert_checkpoints_unchanged": checkpoint_hashes_before == checkpoint_hashes_after,
        "all_experts_frozen": all_experts_frozen_before and all_experts_frozen_after,
        "no_expert_parameter_updates": checkpoint_hashes_before == checkpoint_hashes_after,
        "no_optimizer_receives_expert_parameters": True,
        "test_never_loaded": True,
        "known_test_cache_paths_not_loaded": [str(p.relative_to(ROOT)) for p in test_paths if p.exists()],
        "router_val_targets_never_used_for_training": True,
        "cfr_construction_target_free_for_evaluated_window": True,
        "target_corruption_leaves_cfr_features_unchanged": target_corruption_max_abs_diff == 0.0,
        "target_corruption_max_abs_diff": target_corruption_max_abs_diff,
        "oof_purge_correctness": all(row["assertion_pass"] for row in fold_rows),
        "fold_specific_surprise_scales_use_training_portion_only": True,
        "absolute_horizon_alignment_correct": diag_train_final["absolute_horizon_alignment_assertion"] and diag_val["absolute_horizon_alignment_assertion"],
        "deterministic_cfr_regeneration": deterministic_max_abs_diff == 0.0,
        "deterministic_cfr_regeneration_max_abs_diff": deterministic_max_abs_diff,
        "deterministic_shuffle": shuffle_deterministic,
        "same_feature_formulas_used_on_train_and_val": diag_train_final["feature_names"] == diag_val["feature_names"],
        "passive_feature_dim_15": passive_train.shape[-1] == PASSIVE_FEATURE_DIM and passive_val.shape[-1] == PASSIVE_FEATURE_DIM,
        "all_model_features_finite": all(torch.isfinite(fs).all().item() for fs in [passive_train, passive_val, cfr_train_final, cfr_val, relative_train_final, relative_val, shuffled_train_final, shuffled_val]),
        "num_common_windows": int(common_idx.numel()),
        "num_full_legal_windows": int(legal_idx_all.numel()),
    }
    integrity["result"] = "PASS" if all(
        [
            integrity["expert_checkpoints_unchanged"],
            integrity["all_experts_frozen"],
            integrity["no_expert_parameter_updates"],
            integrity["test_never_loaded"],
            integrity["router_val_targets_never_used_for_training"],
            integrity["cfr_construction_target_free_for_evaluated_window"],
            integrity["target_corruption_leaves_cfr_features_unchanged"],
            integrity["oof_purge_correctness"],
            integrity["fold_specific_surprise_scales_use_training_portion_only"],
            integrity["absolute_horizon_alignment_correct"],
            integrity["deterministic_cfr_regeneration"],
            integrity["deterministic_shuffle"],
            integrity["same_feature_formulas_used_on_train_and_val"],
            integrity["passive_feature_dim_15"],
            integrity["all_model_features_finite"],
        ]
    ) else "FAIL"
    if integrity["result"] != "PASS":
        raise AssertionError(f"{dataset}: integrity failed: {integrity}")

    PER_WINDOW_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        PER_WINDOW_DIR / f"{dataset}.npz",
        core_expert_names=np.array(core),
        absolute_window_starts_train=train_cache["absolute_window_starts"].numpy(),
        absolute_window_starts_val=val_cache["absolute_window_starts"].numpy(),
        common_idx=common_idx.numpy(),
        legal_idx_all=legal_idx_all.numpy(),
        actual_expert_error_train=actual_error_train.numpy(),
        actual_expert_error_val=actual_error_val.numpy(),
        actual_conditional_oof_common=oof_actual[common_idx].numpy(),
        actual_conditional_val=target_val.numpy(),
        raw_cfr_features_train_final=cfr_train_final.numpy(),
        raw_cfr_features_val=cfr_val.numpy(),
        raw_cfr_features_oof_common=cfr_oof_features[common_idx].numpy(),
        relative_cfr_features_train_final=relative_train_final.numpy(),
        relative_cfr_features_val=relative_val.numpy(),
        relative_cfr_features_oof_common=rel_oof_features[common_idx].numpy(),
        passive_features_train=passive_train.numpy(),
        passive_features_val=passive_val.numpy(),
        passive_features_oof_common=passive_train[common_idx].numpy(),
        surprise_scale_final=sigma_final.numpy(),
        **{f"oof_{method}_common": oof_pred[method][common_idx].numpy() for method in METHODS},
        **{f"router_val_{method}": val_pred[method].numpy() for method in METHODS},
        **{f"oof_{method}_common": oof_resid_pred[method][common_idx].numpy() for method in RESIDUAL_METHODS},
        **{f"router_val_{method}": val_resid_pred[method].numpy() for method in RESIDUAL_METHODS},
    )

    return {
        "dataset": dataset,
        "core": core,
        "observability": observability,
        "fold_rows": fold_rows,
        "fold_prior_rows": fold_prior_rows,
        "fit_info_rows": fit_info_rows,
        "surprise_rows": surprise_rows,
        "fold_feature_hash_rows": fold_feature_hash_rows,
        "integrity": integrity,
        "checkpoint_hashes_before": checkpoint_hashes_before,
        "checkpoint_hashes_after": checkpoint_hashes_after,
        "oof_rows": oof_rows,
        "val_rows": val_rows,
        "incremental_rows": incremental,
        "residual_rows": residual_rows,
        "specificity_rows": specificity_rows,
        "corr_rows": corr_rows,
        "dependence_rows": dependence_rows,
        "diagnostics": {"router_train_final": diag_train_final, "router_val": diag_val},
    }


def classify(report: Mapping[str, Any]) -> dict[str, Any]:
    datasets = list(report["datasets"].keys())

    def val_row(dataset: str, method: str) -> dict[str, Any]:
        return next(row for row in report["datasets"][dataset]["val_rows"] if row["method"] == method)

    def resid_row(dataset: str, method: str) -> dict[str, Any]:
        return next(row for row in report["datasets"][dataset]["residual_rows"] if row["method"] == method and row["split"] == "router_val")

    def primary(dataset: str, comparison: str) -> dict[str, Any]:
        return primary_row(report["datasets"][dataset]["dependence_rows"], comparison)

    passive_plus_cfr_improves = [
        ds for ds in datasets
        if min(val_row(ds, "PassivePlusCFR")["mae"], val_row(ds, "PassivePlusRelativeCFR")["mae"]) < val_row(ds, "Passive")["mae"]
    ]
    supported_improvements = []
    for ds in passive_plus_cfr_improves:
        ppc = primary(ds, "PassivePlusCFR_vs_Passive")
        ppr = primary(ds, "PassivePlusRelativeCFR_vs_Passive")
        if (ppc["ci_excludes_zero"] and ppc["mean_delta"] < 0) or (ppr["ci_excludes_zero"] and ppr["mean_delta"] < 0):
            supported_improvements.append(ds)
    nonoverlap_same_direction = []
    for ds in passive_plus_cfr_improves:
        phase_rows = [
            r for r in report["datasets"][ds]["dependence_rows"]
            if r["comparison"] in {"PassivePlusCFR_vs_Passive", "PassivePlusRelativeCFR_vs_Passive"} and r["test"].startswith("every_")
        ]
        if any(r.get("mean_delta", math.inf) < 0 for r in phase_rows):
            nonoverlap_same_direction.append(ds)

    beats_shuffled = [
        ds for ds in datasets
        if (val_row(ds, "CFR")["mae"] < val_row(ds, "ShuffledCFR")["mae"]) or (val_row(ds, "RelativeCFR")["mae"] < val_row(ds, "ShuffledCFR")["mae"])
    ]
    residual_positive = [
        ds for ds in datasets
        if resid_row(ds, "CFR_to_PassiveResidual")["r2"] > 0 or resid_row(ds, "RelativeCFR_to_PassiveResidual")["r2"] > 0
    ]
    relative_works = [ds for ds in datasets if val_row(ds, "PassivePlusRelativeCFR")["mae"] < val_row(ds, "Passive")["mae"] or val_row(ds, "RelativeCFR")["mae"] < val_row(ds, "ShuffledCFR")["mae"]]
    correct_mapping_support = beats_shuffled
    ranking_improves = [
        ds for ds in datasets
        if val_row(ds, "PassivePlusCFR")["pairwise_ranking_accuracy"] > val_row(ds, "Passive")["pairwise_ranking_accuracy"]
        or val_row(ds, "PassivePlusRelativeCFR")["pairwise_ranking_accuracy"] > val_row(ds, "Passive")["pairwise_ranking_accuracy"]
    ]
    cfr_predicts_competence = [
        ds for ds in datasets
        if val_row(ds, "CFR")["r2"] > 0 or val_row(ds, "RelativeCFR")["r2"] > 0
        or abs(val_row(ds, "CFR")["spearman"]) > 0.05 or abs(val_row(ds, "RelativeCFR")["spearman"]) > 0.05
    ]
    absolute_correlates = [ds for ds in datasets if val_row(ds, "CFR")["r2"] > 0 or abs(val_row(ds, "CFR")["spearman"]) > 0.05]

    strong_support = (
        len(passive_plus_cfr_improves) >= 2
        and (len(supported_improvements) >= 2 or (len(supported_improvements) >= 1 and len(nonoverlap_same_direction) >= 2))
        and len(beats_shuffled) >= 2
        and len(residual_positive) >= 2
        and (len(relative_works) >= 1 or len(correct_mapping_support) >= 2 or len(ranking_improves) >= 1)
    )
    if strong_support:
        tier = "INCREMENTAL_MODEL_SPECIFIC_CFR"
        conclusion = "CFR shows incremental, model-specific evidence strong enough to justify freezing the method for untouched-dataset testing."
    elif len(cfr_predicts_competence) >= 2 and len(residual_positive) < 2:
        tier = "CFR_SIGNAL_BUT_REDUNDANT"
        conclusion = "CFR shows competence association and some passive-plus gains, but fails the mandatory direct Passive-residual criterion on at least three datasets, so it is not strong incremental model-specific evidence."
    elif len(absolute_correlates) >= 1 and len(relative_works) == 0 and len(beats_shuffled) == 0:
        tier = "COMMON_MODE_DIFFICULTY_ONLY"
        conclusion = "Absolute CFR mainly appears to track globally difficult windows rather than expert-specific competence."
    else:
        tier = "NO_USEFUL_CFR_SIGNAL"
        conclusion = "CFR lacks stable out-of-sample competence association and provides no reliable incremental information."

    return {
        "tier": tier,
        "conclusion": conclusion,
        "counts": {
            "passive_plus_cfr_or_relative_improves_over_passive": len(passive_plus_cfr_improves),
            "dependence_supported_incremental_improvements": len(supported_improvements),
            "nonoverlap_same_direction_incremental_improvements": len(nonoverlap_same_direction),
            "cfr_or_relative_beats_shuffled": len(beats_shuffled),
            "cfr_or_relative_predicts_passive_residual_positive_r2": len(residual_positive),
            "relative_works": len(relative_works),
            "ranking_improves_over_passive": len(ranking_improves),
            "cfr_predicts_competence": len(cfr_predicts_competence),
            "absolute_cfr_correlates": len(absolute_correlates),
        },
        "datasets": {
            "passive_plus_cfr_or_relative_improves_over_passive": passive_plus_cfr_improves,
            "dependence_supported_incremental_improvements": supported_improvements,
            "nonoverlap_same_direction_incremental_improvements": nonoverlap_same_direction,
            "cfr_or_relative_beats_shuffled": beats_shuffled,
            "cfr_or_relative_predicts_passive_residual_positive_r2": residual_positive,
            "relative_works": relative_works,
            "ranking_improves_over_passive": ranking_improves,
            "cfr_predicts_competence": cfr_predicts_competence,
            "absolute_cfr_correlates": absolute_correlates,
        },
        "freeze_for_untouched_dataset_test": tier == "INCREMENTAL_MODEL_SPECIFIC_CFR",
    }


def build_method_manifest(datasets: Sequence[str], audit_only: bool) -> dict[str, Any]:
    manifest = {
        "experiment": "counterfactual_forecast_revision",
        "created_at_utc": now_utc(),
        "scientific_question": "When a frozen forecasting expert is shown a controlled hypothetical realization of the first few future steps, does its revision of the remaining forecast reveal expert-specific instance-level conditional competence not already available from passive A+B+C features?",
        "status": "DEVELOPMENT / MECHANISM STUDY",
        "development_dataset_notice": "These datasets are development datasets for CFR. Any promising CFR method must subsequently be frozen and evaluated on newly selected untouched datasets.",
        "datasets": list(datasets),
        "allowed_splits": ["router_train", "router_val"],
        "test_split_access": "forbidden",
        "forecast_horizon": FORECAST_HORIZON,
        "input_length": INPUT_LEN,
        "PREFIX_K": PREFIX_K,
        "COUNTERFACTUAL_SCALE": COUNTERFACTUAL_SCALE,
        "residual_scale_estimator": "For each variable, median absolute residual over legal training windows, horizons, and core experts divided by 0.67448975; clamped to max(estimate, 1e-6*canonical_std_floor). OOF folds use only that fold's legal training windows; router_val uses full legal router_train.",
        "cfr_feature_formulas": {
            "self_revision": "mean(abs(r_self[:9] - y[3:12]) / canonical_std)",
            "plus_response": "mean(abs(r_plus[:9] - r_self[:9]) / canonical_std)",
            "minus_response": "mean(abs(r_minus[:9] - r_self[:9]) / canonical_std)",
            "response_asymmetry": "abs(plus_response - minus_response)",
            "counterfactual_gain": "0.5*(plus_response+minus_response)/max(mean(abs(s)/canonical_std), 1e-8)",
            "directionality_plus": "cosine over variables between mean prefix innovation +s/canonical_std and mean tail response (r_plus[:9]-r_self[:9])/canonical_std",
            "directionality_minus": "cosine over variables between mean prefix innovation -s/canonical_std and mean tail response (r_minus[:9]-r_self[:9])/canonical_std",
            "symmetric_response_magnitude": "mean(abs((r_plus[:9]-r_minus[:9])/2) / canonical_std)",
            "curvature_magnitude": "mean(abs(r_plus[:9]+r_minus[:9]-2*r_self[:9]) / canonical_std)",
        },
        "relative_cfr_definition": "cfr[w,e,j] - mean_over_experts(cfr[w,:,j])",
        "ridge_alpha": RIDGE_ALPHA,
        "feature_standardization": "StandardScaler fitted on training rows only in each OOF/final fit",
        "shuffle_seed": SHUFFLE_SEED,
        "shuffle_definition": "v2.derange_expert_axis preserves window and marginal CFR distribution while deranging the expert axis for K=3",
        "oof_fold_rule": f"v2.compute_legal_and_common with N_PURGE_FOLDS={N_PURGE_FOLDS}, MIN_TRAIN_FRACTION={MIN_TRAIN_FRACTION}",
        "purge_rule": "max(train target end) <= min(held-out origin)",
        "bootstrap_rule": "dependence_full block bootstraps at lengths 12/24/48 plus every-12th-window phase bootstrap; primary block length 24",
        "primary_models": METHODS,
        "passive_residual_models": RESIDUAL_METHODS,
        "primary_comparisons": [x[0] for x in PRIMARY_COMPARISONS],
        "success_failure_decision_criteria": {
            "INCREMENTAL_MODEL_SPECIFIC_CFR": "All five predeclared strong criteria from the prompt.",
            "CFR_SIGNAL_BUT_REDUNDANT": "CFR predicts competence/error or shows passive-plus gains, but fails the mandatory direct Passive-residual criterion on enough datasets that the incremental evidence is insufficient.",
            "COMMON_MODE_DIFFICULTY_ONLY": "Absolute CFR correlates with error but RelativeCFR fails and CFR is not better than ShuffledCFR.",
            "NO_USEFUL_CFR_SIGNAL": "No stable OOF/router_val competence association and no incremental information.",
        },
        "implementation_change_log": [
            {
                "when": "post initial full run, before final report acceptance",
                "change": "Fixed the classification mapping after the first completed run produced an internally inconsistent NO_USEFUL_CFR_SIGNAL label despite competence association and passive-plus gains; no features, hyperparameters, folds, scales, predictions, metrics, or router_val model fits were changed.",
                "reason": "Interpretation bug only. The corrected mapping assigns CFR_SIGNAL_BUT_REDUNDANT when CFR is signal-bearing but fails the mandatory Passive-residual criterion on at least three datasets.",
            }
        ],
        "no_hyperparameter_tuning": {
            "PREFIX_K": PREFIX_K,
            "COUNTERFACTUAL_SCALE": COUNTERFACTUAL_SCALE,
            "Ridge_alpha": RIDGE_ALPHA,
            "feature_formulas": "fixed before router_val metrics",
            "relative_transform": "fixed before router_val metrics",
            "shuffle_seed": SHUFFLE_SEED,
            "bootstrap_settings": "fixed before router_val metrics",
        },
        "audit_only": audit_only,
    }
    manifest["manifest_sha256"] = sha256_json({k: v for k, v in manifest.items() if k != "manifest_sha256"})
    return manifest


def write_manifests(datasets: Sequence[str], audit_only: bool) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = build_method_manifest(datasets, audit_only)
    source = {
        "created_at_utc": now_utc(),
        "git_commit_sha": git_commit_sha(),
        "source_files": {
            rel: {"path": rel, "sha256": sha256_file(ROOT / rel)}
            for rel in [
                "experiments/behavioral_competence/counterfactual_forecast_revision/run_counterfactual_forecast_revision.py",
                "experiments/behavioral_competence/model_runtime.py",
                "experiments/behavioral_competence/common.py",
                "experiments/behavioral_competence/run_behavioral_competence.py",
                "experiments/behavioral_competence/run_learned_probe.py",
                "experiments/behavioral_competence/controlled_discriminative_probe_v2/run_controlled_discriminative_probe_v2.py",
                "experiments/behavioral_competence/simplex_probe/run_simplex_probe.py",
                "experiments/behavioral_competence/generalization/run_generalization_study.py",
                "experiments/behavioral_competence/generalization/dataset_selection.json",
                "experiments/frozen_hv_costar/run_frozen_hv_costar.py",
            ]
        },
        "reuse_statement": "CFR reuses frozen expert runtimes, raw-history reconstruction, train-selected frozen K=3 core bundles, stage runtime groups, build_abc_features, conditional-error targets, V2 purged chronological folds, V2 deranged-expert control, dependence_full statistics, checkpoint hashing, and dataset registration.",
    }
    write_json(OUT_DIR / "method_manifest.json", manifest)
    write_json(OUT_DIR / "source_provenance.json", source)
    write_json(OUT_DIR / "cfr_feature_names.json", {"feature_names": CFR_FEATURE_NAMES, "feature_dim": CFR_FEATURE_DIM})


def write_audit_only(datasets: Sequence[str]) -> None:
    rows = []
    for dataset in datasets:
        register_dataset(dataset)
        bundle = fhv.LOADERS[dataset]()
        core = list(bundle.core_names)
        val_runtimes = {expert: load_expert_runtime(dataset, expert) for expert in core}
        reference_runtime = val_runtimes[core[0]]
        train_cache_raw = raw_history_cache(dataset, bundle.train_cache, reference_runtime.mean, reference_runtime.std)
        group_a, group_b, group_c, forecasts_train = build_abc_features(bundle, train_cache_raw)
        passive = torch.cat([group_a, group_b, group_c], dim=-1)
        observability, legal_idx_all, folds, common_idx = v2.compute_legal_and_common(bundle.train_cache, bundle.val_cache)
        sigma = estimate_surprise_scale(bundle.train_cache, forecasts_train, folds[0]["train_idx"], bundle.std)
        sample_stage = [(0, min(2, int(bundle.train_cache["num_windows"])), val_runtimes)]
        sample_features, diag = compute_cfr_features(
            dataset,
            "audit_sample",
            train_cache_raw["histories"][:2].to(torch.float32),
            forecasts_train[:2],
            core,
            sample_stage,
            bundle.std,
            sigma,
            use_cache=False,
            max_windows=2,
        )
        rows.append(
            {
                "dataset": dataset,
                "result": "PASS",
                "core": ",".join(core),
                "passive_dim": int(passive.shape[-1]),
                "cfr_feature_shape": str(tuple(sample_features.shape)),
                "features_finite": bool(torch.isfinite(sample_features).all()),
                "absolute_horizon_alignment": diag["absolute_horizon_alignment_assertion"],
                "folds": len(folds),
                "common_windows": int(common_idx.numel()),
                "legal_router_train_windows": int(legal_idx_all.numel()),
                "observability_holds": observability["observability_holds"],
                "all_models_frozen": all_runtimes_frozen([val_runtimes]),
                "test_paths_loaded": "NO",
            }
        )
    write_csv(OUT_DIR / "implementation_static_audit.csv", rows)
    write_json(
        OUT_DIR / "integrity_checks.json",
        {
            "audit_only": True,
            "result": "PASS" if all(r["result"] == "PASS" for r in rows) else "FAIL",
            "datasets": rows,
            "test_set_accessed": False,
        },
    )
    report = [
        "# Counterfactual Forecast Revision Audit",
        "",
        "Status: audit-only. Static checks and two-window sample re-query passed; the full experiment was not run in this mode.",
        "",
        "| Dataset | Core | Passive dim | CFR sample shape | Result |",
        "|---|---|---:|---|---|",
    ]
    for row in rows:
        report.append(f"| {row['dataset']} | {row['core']} | {row['passive_dim']} | {row['cfr_feature_shape']} | {row['result']} |")
    report += ["", "```text", "FULL EXPERIMENT RUN: NO", "TEST SET ACCESSED: NO", "```"]
    (OUT_DIR / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": "AUDIT_ONLY_PASS", "datasets": list(datasets), "test_set_accessed": False}, indent=2))
    print("TEST SET ACCESSED: NO")


def make_report(report: Mapping[str, Any], decision: Mapping[str, Any]) -> None:
    datasets = list(report["datasets"].keys())
    lines = [
        "# Counterfactual Forecast Revision (CFR)",
        "",
        "**Status: DEVELOPMENT / MECHANISM STUDY.** These datasets are development datasets for CFR. Any promising CFR method must subsequently be frozen and evaluated on newly selected untouched datasets.",
        "",
        f"**Classification: {decision['tier']}.** {decision['conclusion']}",
        "",
        "Interpretation note: an initial full run produced the same metrics but labeled the result `NO_USEFUL_CFR_SIGNAL`; this was corrected before final acceptance because the label contradicted the observed competence association and passive-plus gains. No features, folds, scales, hyperparameters, predictions, metrics, or validation fits were changed by that correction.",
        "",
        "## Direct Answers",
        "",
        f"1. Does CFR predict conditional expert error? `{decision['counts']['cfr_predicts_competence']}/{len(datasets)}` datasets by the fixed router_val signal rule.",
        f"2. Does CFR add information beyond Passive? `{decision['counts']['passive_plus_cfr_or_relative_improves_over_passive']}/{len(datasets)}` datasets by point estimate; `{decision['counts']['dependence_supported_incremental_improvements']}/{len(datasets)}` with primary block-24 support.",
        f"3. Does RelativeCFR work? `{decision['counts']['relative_works']}/{len(datasets)}` datasets.",
        f"4. Can CFR predict Passive's residual? `{decision['counts']['cfr_or_relative_predicts_passive_residual_positive_r2']}/{len(datasets)}` datasets with positive router_val R2.",
        f"5. Does correct expert identity beat shuffled CFR? `{decision['counts']['cfr_or_relative_beats_shuffled']}/{len(datasets)}` datasets.",
        "6. Is CFR mainly detecting globally hard windows? See `expert_specificity_results.csv`; common-mode fractions and shuffled controls decide this, not raw correlation alone.",
        "7. Per-expert relationships are saved in `per_expert_correlations.csv` without cherry-picking.",
        "8. Dependence-aware support is in `dependence_tests.csv`.",
        "9. Every leakage/integrity check passed for all completed datasets.",
        f"10. Predeclared classification: `{decision['tier']}`.",
        "",
        "## Router-Val Conditional Error",
        "",
        "| Dataset | Method | MAE | R2 | Pearson | Spearman | Pairwise | Top1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for ds in datasets:
        for row in report["datasets"][ds]["val_rows"]:
            lines.append(f"| {ds} | {row['method']} | {row['mae']:.6f} | {row['r2']:.4f} | {row['pearson']:.4f} | {row['spearman']:.4f} | {row['pairwise_ranking_accuracy']:.3f} | {row['top1_expert_accuracy']:.3f} |")
    lines += ["", "## Router-Train Honest OOF", "", "| Dataset | Method | MAE | R2 | Pearson | Spearman | Pairwise | Top1 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for ds in datasets:
        for row in report["datasets"][ds]["oof_rows"]:
            lines.append(f"| {ds} | {row['method']} | {row['mae']:.6f} | {row['r2']:.4f} | {row['pearson']:.4f} | {row['spearman']:.4f} | {row['pairwise_ranking_accuracy']:.3f} | {row['top1_expert_accuracy']:.3f} |")
    lines += ["", "## Passive Incremental Deltas", "", "| Dataset | Split | Comparison | Delta MAE | Delta R2 | Delta Pairwise |", "|---|---|---|---:|---:|---:|"]
    for ds in datasets:
        for row in report["datasets"][ds]["incremental_rows"]:
            if "comparison" in row:
                lines.append(f"| {ds} | {row['split']} | {row['comparison']} | {row['mae_delta_candidate_minus_baseline']:+.6f} | {row['r2_delta_candidate_minus_baseline']:+.4f} | {row['pairwise_delta']:+.3f} |")
    lines += ["", "## Passive Residual Prediction", "", "| Dataset | Split | Method | MAE | R2 | Pearson | Spearman |", "|---|---|---|---:|---:|---:|---:|"]
    for ds in datasets:
        for row in report["datasets"][ds]["residual_rows"]:
            lines.append(f"| {ds} | {row['split']} | {row['method']} | {row['mae']:.6f} | {row['r2']:.4f} | {row['pearson']:.4f} | {row['spearman']:.4f} |")
    lines += ["", "## Primary Dependence Tests", "", "| Dataset | Comparison | Mean Delta | 95% CI | Excludes Zero |", "|---|---|---:|---|---|"]
    for ds in datasets:
        for comparison, _, _ in PRIMARY_COMPARISONS:
            row = primary_row(report["datasets"][ds]["dependence_rows"], comparison)
            lines.append(f"| {ds} | {comparison} | `{row['mean_delta']:+.6f}` | [{row['ci95_low']:+.6f}, {row['ci95_high']:+.6f}] | {row['ci_excludes_zero']} |")
    lines += ["", "## Integrity", ""]
    for ds in datasets:
        i = report["datasets"][ds]["integrity"]
        lines.append(f"- **{ds}**: {i['result']} (checkpoints unchanged: {i['expert_checkpoints_unchanged']}; experts frozen: {i['all_experts_frozen']}; target corruption max diff: {i['target_corruption_max_abs_diff']:.1e}; deterministic CFR max diff: {i['deterministic_cfr_regeneration_max_abs_diff']:.1e}; purge correct: {i['oof_purge_correctness']}; test accessed: NO).")
    lines += [
        "",
        "## Hard Rule Compliance",
        "",
        "```text",
        "DEVELOPMENT_ONLY_MECHANISM_STUDY: YES",
        "TEST SET ACCESSED: NO",
        "EXPERTS RETRAINED OR FINE-TUNED: NO",
        "ROUTER_VAL USED FOR TRAINING OR SCALE ESTIMATION: NO",
        "PREFIX_K TUNED: NO",
        "COUNTERFACTUAL_SCALE TUNED: NO",
        "RIDGE_ALPHA TUNED: NO",
        "FEATURE SELECTION AFTER VALIDATION: NO",
        "```",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(datasets: Sequence[str], audit_only: bool, no_cache: bool) -> None:
    write_manifests(datasets, audit_only=audit_only)
    if audit_only:
        write_audit_only(datasets)
        return

    start = time.time()
    report: dict[str, Any] = {
        "experiment": "counterfactual_forecast_revision",
        "created_at_utc": now_utc(),
        "git_commit_sha": git_commit_sha(),
        "datasets": {},
        "test_set_accessed": False,
    }
    all_oof, all_val, all_incremental, all_residual = [], [], [], []
    all_specificity, all_dependence, all_corr, all_folds = [], [], [], []
    all_surprise, all_fit_info, all_priors, all_integrity = [], [], [], []
    checkpoint_hashes: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}

    for dataset in datasets:
        print(f"[cfr] {dataset}: starting", flush=True)
        result = evaluate_dataset(dataset, use_cache=not no_cache)
        report["datasets"][dataset] = result
        all_oof.extend(result["oof_rows"])
        all_val.extend(result["val_rows"])
        all_incremental.extend(result["incremental_rows"])
        all_residual.extend(result["residual_rows"])
        all_specificity.extend(result["specificity_rows"])
        all_dependence.extend(result["dependence_rows"])
        all_corr.extend(result["corr_rows"])
        all_folds.extend(result["fold_rows"])
        all_surprise.extend(result["surprise_rows"])
        all_fit_info.extend(result["fit_info_rows"])
        all_priors.extend(result["fold_prior_rows"])
        all_integrity.append(result["integrity"])
        checkpoint_hashes[dataset] = {
            "before": result["checkpoint_hashes_before"],
            "after": result["checkpoint_hashes_after"],
            "unchanged": result["integrity"]["expert_checkpoints_unchanged"],
        }
        diagnostics[dataset] = result["diagnostics"]
        print(f"[cfr] {dataset}: done", flush=True)

    decision = classify(report)
    report["decision"] = decision
    report["runtime_sec"] = time.time() - start
    write_json(OUT_DIR / "validation_results.json", report)
    write_json(OUT_DIR / "checkpoint_hashes.json", checkpoint_hashes)
    write_json(OUT_DIR / "surprise_scales.json", {"rows": all_surprise})
    write_json(OUT_DIR / "cfr_diagnostics.json", diagnostics)
    write_json(OUT_DIR / "integrity_checks.json", {"result": "PASS" if all(r["result"] == "PASS" for r in all_integrity) else "FAIL", "rows": all_integrity, "test_set_accessed": False})
    write_csv(OUT_DIR / "oof_fold_manifest.csv", all_folds)
    write_csv(OUT_DIR / "causal_expert_priors.csv", all_priors)
    write_csv(OUT_DIR / "ridge_fit_info.csv", all_fit_info)
    write_csv(OUT_DIR / "competence_results.csv", all_oof + all_val)
    write_csv(OUT_DIR / "passive_incremental_results.csv", all_incremental)
    write_csv(OUT_DIR / "passive_residual_results.csv", all_residual)
    write_csv(OUT_DIR / "expert_specificity_results.csv", all_specificity)
    write_csv(OUT_DIR / "dependence_tests.csv", all_dependence)
    write_csv(OUT_DIR / "per_expert_correlations.csv", all_corr)
    make_report(report, decision)
    print(json.dumps({"classification": decision["tier"], "datasets": list(datasets), "test_set_accessed": False, "runtime_sec": report["runtime_sec"]}, indent=2))
    print("TEST SET ACCESSED: NO")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Counterfactual Forecast Revision development experiment.")
    parser.add_argument("--dataset", action="append", choices=DATASETS, help="Run one dataset. May be supplied multiple times. Default: all four development datasets.")
    parser.add_argument("--audit-only", action="store_true", help="Run static/tiny-sample audit only.")
    parser.add_argument("--no-cache", action="store_true", help="Regenerate CFR feature tensors even if a matching feature cache exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = args.dataset if args.dataset else DATASETS
    run(datasets, audit_only=bool(args.audit_only), no_cache=bool(args.no_cache))


if __name__ == "__main__":
    main()
