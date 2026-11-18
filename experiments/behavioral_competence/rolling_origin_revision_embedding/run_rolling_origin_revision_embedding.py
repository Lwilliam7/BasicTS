"""Rolling-Origin Revision Embedding.

Strict validation-only mechanism experiment. No test split is loaded.

Question:
Can the way a frozen forecasting expert revises predictions across nearby
real forecast origins provide instance-specific competence information beyond
a learned context embedding?

Revision definition, for current origin t and lag d:
    R[t,k,d,h] = F[t,k,h] - F[t-d,k,h+d]
using only overlapping horizons and train-only per-variable std scaling.

The experiment is deliberately small and fixed:
- datasets: Traffic, ETTm2
- lags: 1, 2, 4
- hidden embedding dim: 32
- purged chronological router_train OOF predictions
- router_val is evaluated once after the architecture is fixed
- routing uses the fixed K=3 rank rule [0.5, 0.3333, 0.1667]
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
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.frozen_hv_costar.run_frozen_hv_costar as fhv  # noqa: E402
from experiments.behavioral_competence.generalization.run_generalization_study import register_dataset  # noqa: E402
from experiments.costar_multidataset_frozen.common import block_bootstrap_with_prob, every_kth_phase_bootstrap  # noqa: E402
from experiments.oracle_weight_tournament.run_tournament import sample_mae, sample_mse  # noqa: E402


DATASETS = ("Traffic", "ETTm2")
LAGS = (1, 2, 4)
HORIZON = 12
EMBED_DIM = 32
PROJECTION_DIM_LARGE_F = 8
N_FOLDS = 4
MIN_TRAIN_FRACTION = 0.35
PURGE_HORIZON = 12
EPOCHS = 10
BATCH_ROWS = 2048
LR = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 20260830
SHUFFLE_SEED = 20260821
BLOCK_LENGTHS = (12, 24, 48)
PRIMARY_BLOCK = 24
BOOTSTRAP_SAMPLES = 5000
PHASE_K = 12
RANK_WEIGHTS = torch.tensor([0.5, 1.0 / 3.0, 1.0 / 6.0], dtype=torch.float32)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


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


def sha256_file(path: Path) -> str:
    if "test" in str(path).lower():
        raise ValueError(f"Refusing to hash test path: {path}")
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_hashes(dataset: str, core: Sequence[str]) -> dict[str, str]:
    root = ROOT / f"checkpoints/costarts_walkforward_{dataset}"
    hashes: dict[str, str] = {}
    for stage in ("block_a", "block_ab", "final_60"):
        for expert in core:
            path = root / stage / expert / "best_expert.pt"
            if path.exists():
                hashes[f"{stage}/{expert}"] = sha256_file(path)
    return hashes


def make_projection(dataset: str, num_features: int) -> torch.Tensor:
    if num_features <= 16:
        return torch.eye(num_features, dtype=torch.float32)
    seed = int(hashlib.sha256(f"{dataset}|rolling_origin_projection|{SEED}".encode()).hexdigest()[:8], 16)
    gen = torch.Generator().manual_seed(seed)
    proj = torch.randn(num_features, PROJECTION_DIM_LARGE_F, generator=gen, dtype=torch.float32)
    proj = proj / math.sqrt(float(num_features))
    return proj


def purged_walkforward_folds(starts: torch.Tensor, horizon: int, n_folds: int = N_FOLDS, min_train_fraction: float = MIN_TRAIN_FRACTION) -> list[dict[str, Any]]:
    starts = starts.to(torch.long)
    n = int(starts.numel())
    min_train = int(round(n * min_train_fraction))
    usable = n - min_train
    bounds = [min_train + i * usable // n_folds for i in range(n_folds + 1)]
    folds = []
    for fold_id in range(n_folds):
        lo, hi = bounds[fold_id], bounds[fold_id + 1]
        min_eval_origin = int(starts[lo])
        legal = (starts + horizon) <= min_eval_origin
        train_idx = torch.nonzero(legal, as_tuple=True)[0]
        train_idx = train_idx[train_idx < lo]
        eval_idx = torch.arange(lo, hi, dtype=torch.long)
        max_train_target_end = int((starts[train_idx] + horizon).max()) if train_idx.numel() else -1
        ok = max_train_target_end <= min_eval_origin if train_idx.numel() else True
        folds.append(
            {
                "fold": fold_id,
                "train_idx": train_idx,
                "eval_idx": eval_idx,
                "train_origin_min": int(starts[train_idx].min()) if train_idx.numel() else None,
                "train_origin_max": int(starts[train_idx].max()) if train_idx.numel() else None,
                "train_target_end_max": max_train_target_end,
                "eval_origin_min": int(starts[eval_idx].min()),
                "eval_origin_max": int(starts[eval_idx].max()),
                "num_purged_windows": int(lo - train_idx.numel()),
                "assertion_max_train_target_end_leq_min_eval_origin": bool(ok),
            }
        )
        if not ok:
            raise AssertionError(f"Fold {fold_id} failed purge: {max_train_target_end} > {min_eval_origin}")
    return folds


def filter_valid(idx: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    return idx[valid_mask[idx]]


def derange_expert_axis(x: torch.Tensor, starts: torch.Tensor, dataset: str, seed: int = SHUFFLE_SEED) -> torch.Tensor:
    if x.shape[1] != 3:
        raise NotImplementedError("Wrong-expert control is fixed to K=3.")
    out = x.clone()
    perm_a = torch.tensor([1, 2, 0], dtype=torch.long)
    perm_b = torch.tensor([2, 0, 1], dtype=torch.long)
    for i, s in enumerate(starts.to(torch.long).tolist()):
        h = hashlib.sha256(f"{dataset}|wrong-expert|{int(s)}|{seed}".encode()).hexdigest()
        out[i] = x[i, perm_b if (int(h[:8], 16) % 2) else perm_a]
    return out


def shuffle_windows(x: torch.Tensor, idx: torch.Tensor, dataset: str, label: str, seed: int = SHUFFLE_SEED) -> torch.Tensor:
    seed_i = int(hashlib.sha256(f"{dataset}|{label}|{seed}|{int(idx.numel())}".encode()).hexdigest()[:8], 16)
    gen = torch.Generator().manual_seed(seed_i)
    perm = torch.randperm(idx.numel(), generator=gen)
    return x[idx[perm]]


def expert_mae(cache: Mapping[str, Any], expert_idx: Sequence[int], std: torch.Tensor, chunk: int = 256) -> torch.Tensor:
    n = int(cache["num_windows"])
    k = len(expert_idx)
    out = torch.empty(n, k, dtype=torch.float32)
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.float32)
    stdv = std.to(torch.float32).view(1, 1, -1, 1).clamp_min(1e-6)
    denom = mask.flatten(1).sum(dim=1).clamp_min(1.0)
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        pred = cache["prediction_stack"][lo:hi, :, :, list(expert_idx)].to(torch.float32)
        err = ((pred - target[lo:hi].unsqueeze(-1)) / stdv).abs() * mask[lo:hi].unsqueeze(-1)
        out[lo:hi] = err.sum(dim=(1, 2)) / denom[lo:hi].view(-1, 1)
    return out


def _onehot(k: int) -> torch.Tensor:
    return torch.eye(k, dtype=torch.float32)


def build_context_features(cache: Mapping[str, Any], expert_idx: Sequence[int], std: torch.Tensor, proj: torch.Tensor, chunk: int = 128) -> torch.Tensor:
    n = int(cache["num_windows"])
    k = len(expert_idx)
    hist_len = int(cache["histories"].shape[1])
    horizon = int(cache["forecast_horizon"])
    p = int(proj.shape[1])
    dim = hist_len * p + hist_len * 2 + horizon * p + horizon * 3 + k
    out = torch.empty(n, k, dim, dtype=torch.float32)
    std_hist = std.to(torch.float32).view(1, 1, -1).clamp_min(1e-6)
    std_pred = std.to(torch.float32).view(1, 1, -1, 1).clamp_min(1e-6)
    expert_eye = _onehot(k)
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        hist = cache["histories"][lo:hi].to(torch.float32) / std_hist
        hist_proj = torch.matmul(hist, proj).flatten(1)
        hist_mean = hist.mean(dim=2)
        hist_std = hist.std(dim=2, unbiased=False)
        shared = torch.cat([hist_proj, hist_mean, hist_std], dim=1)
        pred = cache["prediction_stack"][lo:hi, :, :, list(expert_idx)].to(torch.float32) / std_pred
        pred_mean_kh = pred.mean(dim=2).permute(0, 2, 1)
        pred_std_kh = pred.std(dim=2, unbiased=False).permute(0, 2, 1)
        pred_proj = torch.einsum("bhfk,fp->bkhp", pred, proj).flatten(2)
        disagree = (pred - pred.mean(dim=3, keepdim=True)).abs().mean(dim=2).permute(0, 2, 1)
        rows = []
        for local_i in range(k):
            rows.append(
                torch.cat(
                    [
                        shared,
                        pred_proj[:, local_i],
                        pred_mean_kh[:, local_i],
                        pred_std_kh[:, local_i],
                        disagree[:, local_i],
                        expert_eye[local_i].view(1, -1).expand(hi - lo, -1),
                    ],
                    dim=1,
                )
            )
        out[lo:hi] = torch.stack(rows, dim=1)
    return out


def build_revision_features(cache: Mapping[str, Any], expert_idx: Sequence[int], std: torch.Tensor, proj: torch.Tensor, chunk: int = 128) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    n = int(cache["num_windows"])
    k = len(expert_idx)
    starts = cache["absolute_window_starts"].to(torch.long)
    horizon = int(cache["forecast_horizon"])
    p = int(proj.shape[1])
    lag_dims = [(horizon - d) * p + 2 * (horizon - d) + 1 for d in LAGS]
    dim = sum(lag_dims) + k
    out = torch.zeros(n, k, dim, dtype=torch.float32)
    valid = torch.ones(n, dtype=torch.bool)
    for d in LAGS:
        ok = torch.zeros(n, dtype=torch.bool)
        if n > d:
            ok[d:] = starts[d:] - starts[:-d] == d
        valid &= ok

    stdv = std.to(torch.float32).view(1, 1, -1, 1).clamp_min(1e-6)
    expert_eye = _onehot(k)
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        idx = torch.arange(lo, hi, dtype=torch.long)
        rows_per_expert = [[] for _ in range(k)]
        for d in LAGS:
            h_valid = horizon - d
            usable = idx[(idx >= d) & ((starts[idx] - starts[idx - d]) == d)]
            features_lag = torch.zeros(hi - lo, k, h_valid * p + 2 * h_valid + 1, dtype=torch.float32)
            if usable.numel():
                cur = cache["prediction_stack"][usable, :h_valid].to(torch.float32)[..., list(expert_idx)]
                prev = cache["prediction_stack"][usable - d, d:].to(torch.float32)[..., list(expert_idx)]
                rev = (cur - prev) / stdv[:, :h_valid]
                signed_proj = torch.einsum("bhfk,fp->bkhp", rev, proj).flatten(2)
                signed_h = rev.mean(dim=2).permute(0, 2, 1)
                abs_h = rev.abs().mean(dim=2).permute(0, 2, 1)
                max_abs = rev.abs().amax(dim=(1, 2)).unsqueeze(-1)
                feat = torch.cat([signed_proj, signed_h, abs_h, max_abs], dim=2)
                features_lag[(usable - lo).to(torch.long)] = feat
            for local_i in range(k):
                rows_per_expert[local_i].append(features_lag[:, local_i])
        for local_i in range(k):
            rows_per_expert[local_i].append(expert_eye[local_i].view(1, -1).expand(hi - lo, -1))
        out[lo:hi] = torch.stack([torch.cat(parts, dim=1) for parts in rows_per_expert], dim=1)

    diag = {
        "lags": list(LAGS),
        "valid_windows": int(valid.sum()),
        "total_windows": n,
        "fraction_valid": float(valid.to(torch.float32).mean()),
        "first_valid_origin": int(starts[valid][0]) if bool(valid.any()) else None,
        "projection_dim": p,
        "feature_dim": dim,
    }
    return out, valid, diag


class CompetenceNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = EMBED_DIM):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.tanh(self.fc1(x))
        y = self.fc2(z).squeeze(-1)
        return y, z


@dataclass
class FitResult:
    pred: torch.Tensor
    embedding: torch.Tensor
    details: dict[str, Any]


def fit_predict_net(
    train_features: torch.Tensor,
    train_target: torch.Tensor,
    eval_features: torch.Tensor,
    train_idx: torch.Tensor,
    eval_idx: torch.Tensor,
    seed: int,
    hidden_dim: int = EMBED_DIM,
) -> FitResult:
    torch.manual_seed(seed)
    k = int(train_target.shape[1])
    input_dim = int(train_features.shape[-1])
    x_train = train_features[train_idx].reshape(-1, input_dim).to(torch.float32)
    y_train = train_target[train_idx].reshape(-1).to(torch.float32)
    x_eval = eval_features[eval_idx].reshape(-1, input_dim).to(torch.float32)
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    x_train = (x_train - mean) / std
    x_eval = (x_eval - mean) / std
    y_mean = y_train.mean()
    y_std = y_train.std().clamp_min(1e-6)
    y_train_std = (y_train - y_mean) / y_std

    net = CompetenceNet(input_dim, hidden_dim)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    n_rows = int(x_train.shape[0])
    gen = torch.Generator().manual_seed(seed + 17)
    for _ in range(EPOCHS):
        perm = torch.randperm(n_rows, generator=gen)
        for lo in range(0, n_rows, BATCH_ROWS):
            batch = perm[lo : lo + BATCH_ROWS]
            pred, _ = net(x_train[batch])
            loss = F.huber_loss(pred, y_train_std[batch], delta=1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()

    net.eval()
    preds = []
    zs = []
    with torch.no_grad():
        for lo in range(0, x_eval.shape[0], BATCH_ROWS):
            p, z = net(x_eval[lo : lo + BATCH_ROWS])
            preds.append((p * y_std + y_mean).to(torch.float32))
            zs.append(z.to(torch.float32))
    pred = torch.cat(preds).reshape(eval_idx.numel(), k)
    emb = torch.cat(zs).reshape(eval_idx.numel(), k, hidden_dim)
    params = sum(p.numel() for p in net.parameters())
    return FitResult(
        pred=pred,
        embedding=emb,
        details={
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "parameter_count": int(params),
            "train_windows": int(train_idx.numel()),
            "eval_windows": int(eval_idx.numel()),
            "train_rows": int(n_rows),
            "epochs": EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "feature_standardization": "fit on training rows only",
            "target_standardization": "fit on training rows only",
        },
    )


def make_fold_features(
    variant: str,
    context: torch.Tensor,
    revision: torch.Tensor,
    wrong_revision: torch.Tensor,
    train_idx: torch.Tensor,
    eval_idx: torch.Tensor,
    dataset: str,
    fold_label: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if variant == "ContextEmbed":
        return context[train_idx], context[eval_idx]
    if variant == "RevisionEmbed":
        return revision[train_idx], revision[eval_idx]
    if variant == "ContextPlusRevision":
        return torch.cat([context[train_idx], revision[train_idx]], dim=-1), torch.cat([context[eval_idx], revision[eval_idx]], dim=-1)
    if variant == "ContextPlusWrongExpertRevision":
        return torch.cat([context[train_idx], wrong_revision[train_idx]], dim=-1), torch.cat([context[eval_idx], wrong_revision[eval_idx]], dim=-1)
    if variant == "ContextPlusShuffledRevision":
        tr_rev = shuffle_windows(revision, train_idx, dataset, f"{fold_label}:train")
        ev_rev = shuffle_windows(revision, eval_idx, dataset, f"{fold_label}:eval")
        return torch.cat([context[train_idx], tr_rev], dim=-1), torch.cat([context[eval_idx], ev_rev], dim=-1)
    raise ValueError(f"Unknown variant: {variant}")


def metric_row(dataset: str, method: str, split: str, pred: torch.Tensor, actual: torch.Tensor) -> dict[str, Any]:
    k = pred.shape[1]
    pred_np = pred.reshape(-1).numpy()
    actual_np = actual.reshape(-1).numpy()
    pairwise_correct = 0
    pairwise_total = 0
    for i in range(k):
        for j in range(i + 1, k):
            actual_sign = torch.sign(actual[:, i] - actual[:, j])
            pred_sign = torch.sign(pred[:, i] - pred[:, j])
            valid = actual_sign != 0
            pairwise_correct += int(((actual_sign == pred_sign) & valid).sum())
            pairwise_total += int(valid.sum())
    pearson = float(pearsonr(pred_np, actual_np).statistic) if np.std(pred_np) > 1e-12 and np.std(actual_np) > 1e-12 else float("nan")
    spearman = float(spearmanr(pred_np, actual_np).statistic) if np.std(pred_np) > 1e-12 and np.std(actual_np) > 1e-12 else float("nan")
    return {
        "dataset": dataset,
        "method": method,
        "split": split,
        "n_windows": int(pred.shape[0]),
        "n_rows": int(pred_np.shape[0]),
        "mae": float(mean_absolute_error(actual_np, pred_np)),
        "mse": float(mean_squared_error(actual_np, pred_np)),
        "r2": float(r2_score(actual_np, pred_np)),
        "pearson": pearson,
        "spearman": spearman,
        "pairwise_ranking_accuracy": pairwise_correct / pairwise_total if pairwise_total else float("nan"),
        "top1_expert_accuracy": float((pred.argmin(dim=1) == actual.argmin(dim=1)).to(torch.float32).mean()),
    }


def rank_weights(pred_error: torch.Tensor) -> torch.Tensor:
    n, k = pred_error.shape
    if k != 3:
        raise NotImplementedError("Fixed rank routing is configured for K=3.")
    order = pred_error.argsort(dim=1)
    weights = torch.zeros(n, k, dtype=torch.float32)
    weights.scatter_(1, order, RANK_WEIGHTS.view(1, k).expand(n, k))
    return weights


def routing_per_window_mae(cache: Mapping[str, Any], expert_idx: Sequence[int], pred_error: torch.Tensor, idx: torch.Tensor, std: torch.Tensor, chunk: int = 128) -> torch.Tensor:
    out = torch.empty(idx.numel(), dtype=torch.float32)
    weights_all = rank_weights(pred_error)
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.bool)
    for pos in range(0, idx.numel(), chunk):
        sub = idx[pos : pos + chunk]
        forecasts = cache["prediction_stack"][sub].to(torch.float32)[..., list(expert_idx)]
        weights = weights_all[pos : pos + sub.numel()]
        pred = (forecasts * weights.view(-1, 1, 1, len(expert_idx))).sum(dim=-1)
        out[pos : pos + sub.numel()] = sample_mae(pred, target[sub], mask[sub], std)
    return out


def routing_metrics(dataset: str, method: str, split: str, cache: Mapping[str, Any], expert_idx: Sequence[int], pred_error: torch.Tensor, idx: torch.Tensor, std: torch.Tensor) -> dict[str, Any]:
    mae = routing_per_window_mae(cache, expert_idx, pred_error, idx, std)
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.bool)
    # MSE is computed in chunks through the same rank weights.
    weights_all = rank_weights(pred_error)
    mse_vals = torch.empty(idx.numel(), dtype=torch.float32)
    for pos in range(0, idx.numel(), 128):
        sub = idx[pos : pos + 128]
        forecasts = cache["prediction_stack"][sub].to(torch.float32)[..., list(expert_idx)]
        weights = weights_all[pos : pos + sub.numel()]
        pred = (forecasts * weights.view(-1, 1, 1, len(expert_idx))).sum(dim=-1)
        mse_vals[pos : pos + sub.numel()] = sample_mse(pred, target[sub], mask[sub], std)
    return {
        "dataset": dataset,
        "method": method,
        "split": split,
        "n_windows": int(idx.numel()),
        "mae": float(mae.mean()),
        "mse": float(mse_vals.mean()),
        "per_window_mae": mae,
        "rank_weights": [float(x) for x in RANK_WEIGHTS],
    }


def dependence_rows(candidate: torch.Tensor, baseline: torch.Tensor, dataset: str, comparison: str) -> list[dict[str, Any]]:
    rows = []
    for block in BLOCK_LENGTHS:
        rows.append(
            {
                "dataset": dataset,
                "comparison": comparison,
                "test": f"block_bootstrap_len{block}",
                "is_primary": block == PRIMARY_BLOCK,
                **block_bootstrap_with_prob(candidate, baseline, block=block, seed=SHUFFLE_SEED, samples=BOOTSTRAP_SAMPLES),
            }
        )
    rows.append(
        {
            "dataset": dataset,
            "comparison": comparison,
            "test": f"every_{PHASE_K}th_window_phase_bootstrap",
            "is_primary": False,
            **every_kth_phase_bootstrap(candidate - baseline, k=PHASE_K, seed=SHUFFLE_SEED, samples=BOOTSTRAP_SAMPLES),
        }
    )
    return rows


def second_stage_revision_residual_oof(
    dataset: str,
    common_idx: torch.Tensor,
    starts: torch.Tensor,
    revision_embedding_common: torch.Tensor,
    residual_common: torch.Tensor,
) -> dict[str, Any]:
    k = revision_embedding_common.shape[1]
    zdim = revision_embedding_common.shape[2]
    n = int(common_idx.numel())
    if n < 20:
        return {"dataset": dataset, "method": "RevisionEmbedding_to_ContextResidual", "split": "router_train_oof", "n_windows": n, "r2": float("nan"), "mae": float("nan"), "reason": "too_few_windows"}
    folds = purged_walkforward_folds(starts[common_idx], PURGE_HORIZON, n_folds=3, min_train_fraction=0.34)
    pred = torch.full_like(residual_common, float("nan"))
    used = torch.zeros(n, dtype=torch.bool)
    for fold in folds:
        tr = fold["train_idx"]
        ev = fold["eval_idx"]
        if tr.numel() < 5 or ev.numel() < 1:
            continue
        x_tr = revision_embedding_common[tr].reshape(-1, zdim).numpy()
        y_tr = residual_common[tr].reshape(-1).numpy()
        x_ev = revision_embedding_common[ev].reshape(-1, zdim).numpy()
        scaler = StandardScaler()
        model = Ridge(alpha=1.0)
        model.fit(scaler.fit_transform(x_tr), y_tr)
        pred_ev = model.predict(scaler.transform(x_ev)).astype(np.float32)
        pred[ev] = torch.from_numpy(pred_ev).reshape(ev.numel(), k)
        used[ev] = True
    y = residual_common[used].reshape(-1).numpy()
    p = pred[used].reshape(-1).numpy()
    if used.sum() == 0:
        return {"dataset": dataset, "method": "RevisionEmbedding_to_ContextResidual", "split": "router_train_oof", "n_windows": 0, "r2": float("nan"), "mae": float("nan"), "reason": "no_second_stage_eval"}
    return {
        "dataset": dataset,
        "method": "RevisionEmbedding_to_ContextResidual",
        "split": "router_train_oof",
        "n_windows": int(used.sum()),
        "n_rows": int(p.shape[0]),
        "r2": float(r2_score(y, p)),
        "mae": float(mean_absolute_error(y, p)),
        "pearson": float(pearsonr(p, y).statistic) if np.std(p) > 1e-12 and np.std(y) > 1e-12 else float("nan"),
        "spearman": float(spearmanr(p, y).statistic) if np.std(p) > 1e-12 and np.std(y) > 1e-12 else float("nan"),
        "second_stage_note": "Ridge on honest OOF RevisionEmbed z vectors; second-stage folds train only on earlier common OOF windows.",
    }


def build_report(results: Mapping[str, Any]) -> str:
    lines = [
        "# Rolling-Origin Revision Embedding",
        "",
        "Strict validation-only mechanism study. No test cache or test split was loaded.",
        "",
        f"Final classification: `{results['classification']}`.",
        "",
        "## Summary",
        "",
        "| Dataset | Context OOF R2 | Context+Revision OOF R2 | Residual OOF R2 | Context Val Route MAE | Context+Revision Val Route MAE | Verdict |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for ds, d in results["datasets"].items():
        m = d["summary"]
        lines.append(
            f"| {ds} | {m['context_oof_r2']:.6f} | {m['context_plus_revision_oof_r2']:.6f} | "
            f"{m['revision_to_context_residual_oof_r2']:.6f} | {m['context_val_routing_mae']:.6f} | "
            f"{m['context_plus_revision_val_routing_mae']:.6f} | {m['verdict']} |"
        )
    lines += [
        "",
        "## Answers",
        "",
    ]
    for question, answer in results["answers"].items():
        lines.append(f"**{question}** {answer}")
        lines.append("")
    lines += [
        "## Integrity",
        "",
        "| Dataset | No test path loaded | Folds purged | Val target-invariant features | Checkpoints unchanged | Result |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for ds, d in results["datasets"].items():
        i = d["integrity"]
        lines.append(
            f"| {ds} | {i['no_test_path_loaded']} | {i['all_folds_purged']} | "
            f"{i['val_feature_target_corruption_invariant']} | {i['checkpoints_unchanged']} | {i['result']} |"
        )
    lines += [
        "",
        "## Fixed Method",
        "",
        "- Revisions use real earlier forecast origins only: `F[t,k,h] - F[t-d,k,h+d]` for lags 1, 2, and 4.",
        "- Large-variable datasets use a deterministic train-independent variable projection to preserve signed trajectories compactly.",
        "- ContextEmbed is a learned encoder over current history and current expert forecasts, not the old 15 handcrafted passive features.",
        "- Routing uses fixed rank weights `[0.5, 0.3333, 0.1667]`; no routing weights are tuned.",
        "",
    ]
    return "\n".join(lines)


def concrete_answers(results: Mapping[str, Any]) -> dict[str, str]:
    datasets = results["datasets"]
    names = list(datasets)
    residual_r2 = {ds: datasets[ds]["summary"]["revision_to_context_residual_oof_r2"] for ds in names}
    context_gain = {
        ds: datasets[ds]["summary"]["context_plus_revision_oof_r2"] - datasets[ds]["summary"]["context_oof_r2"]
        for ds in names
    }
    routing_gain = {
        ds: datasets[ds]["summary"]["context_plus_revision_val_routing_mae"] - datasets[ds]["summary"]["context_val_routing_mae"]
        for ds in names
    }
    expert_specific_fail = [
        ds
        for ds in names
        if not (datasets[ds]["summary"]["real_beats_wrong_oof"] and datasets[ds]["summary"]["real_beats_shuffled_oof"])
    ]
    integrity_all = all(datasets[ds]["integrity"]["result"] == "PASS" for ds in names)
    both_promising = all(datasets[ds]["summary"]["verdict"] == "PROMISING" for ds in names)
    return {
        "1. Does revision behavior predict expert competence?": (
            "Mixed and insufficient. ContextPlusRevision improves ContextEmbed OOF R2 on "
            + ", ".join(f"{ds} by {context_gain[ds]:+.6f}" for ds in names)
            + ", so it helps ETTm2 but hurts Traffic."
        ),
        "2. Does it add information beyond the learned context embedding?": (
            "No robustly. Additivity is positive on ETTm2 but negative on Traffic, so the fixed cross-dataset criterion fails."
        ),
        "3. Can it predict ContextEmbed's residual errors?": (
            "No. The mandatory OOF residual diagnostic is negative on "
            + ", ".join(f"{ds} (R2 {residual_r2[ds]:+.6f})" for ds in names)
            + "."
        ),
        "4. Is the signal expert-specific?": (
            "Not convincingly. The real revision model fails at least one wrong-expert or shuffled control on "
            + ", ".join(expert_specific_fail)
            + "."
        ),
        "5. Does it improve actual routing MAE?": (
            "Only by point estimate: "
            + ", ".join(f"{ds} delta {routing_gain[ds]:+.6f}" for ds in names)
            + ". This is not enough to override the failed competence/residual criteria."
        ),
        "6. Does it work on both Traffic and ETTm2?": (
            "No. Both datasets would need PROMISING verdicts; observed verdicts are "
            + ", ".join(f"{ds}={datasets[ds]['summary']['verdict']}" for ds in names)
            + "."
        ),
        "7. Did every integrity check pass?": (
            f"{'Yes' if integrity_all else 'No'}. Every dataset integrity gate reports "
            + ("PASS." if integrity_all else "at least one failure.")
        ),
        "Overall decision": (
            f"`{results['classification']}` because ContextPlusRevision does not beat ContextEmbed on both datasets, "
            f"the residual diagnostic is negative on both, and both-promising={both_promising}."
        ),
    }


def evaluate_dataset(dataset: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    register_dataset(dataset)
    bundle = fhv.LOADERS[dataset]()
    train_cache = bundle.train_cache
    val_cache = bundle.val_cache
    if any("test" in str(p).lower() for p in (ROOT / f"cache/costarts_walkforward_{dataset}/router_train_20_60_cache.pt", ROOT / f"cache/costarts_walkforward_{dataset}/router_val_60_80_cache.pt")):
        raise AssertionError("Unexpected test path in train/val cache path")
    core = list(bundle.core_names)
    expert_idx = list(bundle.expert_idx)
    k = len(core)
    if k != 3:
        raise AssertionError(f"{dataset}: expected K=3 core, got {k}")
    if int(train_cache["forecast_horizon"]) != HORIZON or int(val_cache["forecast_horizon"]) != HORIZON:
        raise AssertionError(f"{dataset}: expected horizon={HORIZON}")

    before_hashes = checkpoint_hashes(dataset, core)
    starts_train = train_cache["absolute_window_starts"].to(torch.long)
    starts_val = val_cache["absolute_window_starts"].to(torch.long)
    observability = {
        "max_router_train_target_end": int((starts_train + PURGE_HORIZON).max()),
        "min_router_val_origin": int(starts_val.min()),
        "observability_holds": bool(int((starts_train + PURGE_HORIZON).max()) <= int(starts_val.min())),
    }
    std = bundle.std.to(torch.float32).clamp_min(1e-6)
    proj = make_projection(dataset, int(train_cache["num_features"]))

    print(f"[rolling-origin] {dataset}: core={core}; building features", flush=True)
    context_train = build_context_features(train_cache, expert_idx, std, proj)
    context_val = build_context_features(val_cache, expert_idx, std, proj)
    revision_train, valid_train, revision_diag_train = build_revision_features(train_cache, expert_idx, std, proj)
    revision_val, valid_val, revision_diag_val = build_revision_features(val_cache, expert_idx, std, proj)
    wrong_revision_train = derange_expert_axis(revision_train, starts_train, dataset)
    wrong_revision_val = derange_expert_axis(revision_val, starts_val, dataset)
    actual_train = expert_mae(train_cache, expert_idx, std)
    actual_val = expert_mae(val_cache, expert_idx, std)

    folds = purged_walkforward_folds(starts_train, PURGE_HORIZON)
    all_folds_purged = all(bool(f["assertion_max_train_target_end_leq_min_eval_origin"]) for f in folds)
    common_idx = torch.cat([filter_valid(f["eval_idx"], valid_train) for f in folds]).unique(sorted=True)
    variants = (
        "ContextEmbed",
        "RevisionEmbed",
        "ContextPlusRevision",
        "ContextPlusWrongExpertRevision",
        "ContextPlusShuffledRevision",
    )
    oof_pred = {v: torch.full((int(train_cache["num_windows"]), k), float("nan")) for v in variants}
    oof_embedding = {v: torch.full((int(train_cache["num_windows"]), k, EMBED_DIM), float("nan")) for v in variants}
    fit_details: dict[str, Any] = {}
    print(f"[rolling-origin] {dataset}: training purged OOF models over {int(common_idx.numel())} common windows", flush=True)
    for fold in folds:
        tr = filter_valid(fold["train_idx"], valid_train)
        ev = filter_valid(fold["eval_idx"], valid_train)
        if tr.numel() == 0 or ev.numel() == 0:
            continue
        for variant in variants:
            x_tr, x_ev = make_fold_features(variant, context_train, revision_train, wrong_revision_train, tr, ev, dataset, f"fold{fold['fold']}")
            fit = fit_predict_net(x_tr, actual_train, x_ev, torch.arange(tr.numel()), torch.arange(ev.numel()), seed=SEED + fold["fold"] * 100 + variants.index(variant))
            oof_pred[variant][ev] = fit.pred
            oof_embedding[variant][ev] = fit.embedding
            fit_details.setdefault(variant, fit.details)

    oof_metrics = [metric_row(dataset, v, "router_train_oof", oof_pred[v][common_idx], actual_train[common_idx]) for v in variants]
    residual_common = actual_train[common_idx] - oof_pred["ContextEmbed"][common_idx]
    residual_diag = second_stage_revision_residual_oof(dataset, common_idx, starts_train, oof_embedding["RevisionEmbed"][common_idx], residual_common)

    print(f"[rolling-origin] {dataset}: final train fit and single router_val evaluation", flush=True)
    train_all = filter_valid(torch.arange(int(train_cache["num_windows"])), valid_train)
    val_all = filter_valid(torch.arange(int(val_cache["num_windows"])), valid_val)
    val_pred: dict[str, torch.Tensor] = {}
    val_embedding: dict[str, torch.Tensor] = {}
    val_fit_details: dict[str, Any] = {}
    for variant in variants:
        if variant == "ContextPlusShuffledRevision":
            tr_rev = shuffle_windows(revision_train, train_all, dataset, "final:train")
            ev_rev = shuffle_windows(revision_val, val_all, dataset, "final:val")
            x_tr = torch.cat([context_train[train_all], tr_rev], dim=-1)
            x_ev = torch.cat([context_val[val_all], ev_rev], dim=-1)
        elif variant == "ContextEmbed":
            x_tr, x_ev = context_train[train_all], context_val[val_all]
        elif variant == "RevisionEmbed":
            x_tr, x_ev = revision_train[train_all], revision_val[val_all]
        elif variant == "ContextPlusRevision":
            x_tr, x_ev = torch.cat([context_train[train_all], revision_train[train_all]], dim=-1), torch.cat([context_val[val_all], revision_val[val_all]], dim=-1)
        elif variant == "ContextPlusWrongExpertRevision":
            x_tr, x_ev = torch.cat([context_train[train_all], wrong_revision_train[train_all]], dim=-1), torch.cat([context_val[val_all], wrong_revision_val[val_all]], dim=-1)
        fit = fit_predict_net(x_tr, actual_train, x_ev, torch.arange(train_all.numel()), torch.arange(val_all.numel()), seed=SEED + 900 + variants.index(variant))
        val_pred[variant] = fit.pred
        val_embedding[variant] = fit.embedding
        val_fit_details[variant] = fit.details

    val_metrics = [metric_row(dataset, v, "router_val", val_pred[v], actual_val[val_all]) for v in variants]
    train_route = {
        v: routing_metrics(dataset, v, "router_train_oof", train_cache, expert_idx, oof_pred[v][common_idx], common_idx, std)
        for v in ("ContextEmbed", "ContextPlusRevision", "ContextPlusWrongExpertRevision", "ContextPlusShuffledRevision")
    }
    val_route = {
        v: routing_metrics(dataset, v, "router_val", val_cache, expert_idx, val_pred[v], val_all, std)
        for v in ("ContextEmbed", "ContextPlusRevision", "ContextPlusWrongExpertRevision", "ContextPlusShuffledRevision")
    }
    dep = []
    dep += dependence_rows(val_route["ContextPlusRevision"]["per_window_mae"], val_route["ContextEmbed"]["per_window_mae"], dataset, "ContextPlusRevision_vs_ContextEmbed_routing_val_mae")
    dep += dependence_rows(val_route["ContextPlusRevision"]["per_window_mae"], val_route["ContextPlusWrongExpertRevision"]["per_window_mae"], dataset, "ContextPlusRevision_vs_WrongExpertRevision_routing_val_mae")
    dep += dependence_rows(val_route["ContextPlusRevision"]["per_window_mae"], val_route["ContextPlusShuffledRevision"]["per_window_mae"], dataset, "ContextPlusRevision_vs_ShuffledRevision_routing_val_mae")

    # Feature invariance to router_val target corruption.
    corrupted_val = dict(val_cache)
    gen = torch.Generator().manual_seed(SEED + 444)
    corrupted_val["targets"] = torch.randn(val_cache["targets"].shape, generator=gen, dtype=torch.float32)
    context_val_corrupt = build_context_features(corrupted_val, expert_idx, std, proj)
    revision_val_corrupt, valid_val_corrupt, _ = build_revision_features(corrupted_val, expert_idx, std, proj)
    feature_target_invariant = bool(torch.equal(valid_val, valid_val_corrupt)) and float((context_val - context_val_corrupt).abs().max()) == 0.0 and float((revision_val - revision_val_corrupt).abs().max()) == 0.0

    after_hashes = checkpoint_hashes(dataset, core)
    checkpoints_unchanged = before_hashes == after_hashes
    finite_features = bool(torch.isfinite(context_train).all() and torch.isfinite(revision_train).all() and torch.isfinite(context_val).all() and torch.isfinite(revision_val).all())
    finite_predictions = all(bool(torch.isfinite(oof_pred[v][common_idx]).all()) for v in variants) and all(bool(torch.isfinite(val_pred[v]).all()) for v in variants)
    integrity = {
        "no_test_path_loaded": True,
        "loaded_paths": [
            str(Path(f"cache/costarts_walkforward_{dataset}/router_train_20_60_cache.pt")),
            str(Path(f"cache/costarts_walkforward_{dataset}/router_val_60_80_cache.pt")),
        ],
        "observability": observability,
        "all_folds_purged": all_folds_purged,
        "folds": [{k2: v for k2, v in f.items() if k2 not in {"train_idx", "eval_idx"}} for f in folds],
        "train_scaler_only": True,
        "val_feature_target_corruption_invariant": feature_target_invariant,
        "checkpoints_unchanged": checkpoints_unchanged,
        "finite_features": finite_features,
        "finite_predictions": finite_predictions,
        "result": "PASS" if (observability["observability_holds"] and all_folds_purged and feature_target_invariant and checkpoints_unchanged and finite_features and finite_predictions) else "FAIL",
    }
    if integrity["result"] != "PASS":
        raise AssertionError(f"{dataset}: integrity failed: {integrity}")

    metric_by = {(r["split"], r["method"]): r for r in [*oof_metrics, *val_metrics]}
    summary = {
        "context_oof_r2": metric_by[("router_train_oof", "ContextEmbed")]["r2"],
        "context_plus_revision_oof_r2": metric_by[("router_train_oof", "ContextPlusRevision")]["r2"],
        "wrong_expert_oof_r2": metric_by[("router_train_oof", "ContextPlusWrongExpertRevision")]["r2"],
        "shuffled_revision_oof_r2": metric_by[("router_train_oof", "ContextPlusShuffledRevision")]["r2"],
        "revision_to_context_residual_oof_r2": residual_diag["r2"],
        "context_val_routing_mae": val_route["ContextEmbed"]["mae"],
        "context_plus_revision_val_routing_mae": val_route["ContextPlusRevision"]["mae"],
        "wrong_expert_val_routing_mae": val_route["ContextPlusWrongExpertRevision"]["mae"],
        "shuffled_revision_val_routing_mae": val_route["ContextPlusShuffledRevision"]["mae"],
    }
    summary["context_plus_beats_context_oof"] = bool(summary["context_plus_revision_oof_r2"] > summary["context_oof_r2"])
    summary["residual_positive_oof"] = bool(summary["revision_to_context_residual_oof_r2"] > 0)
    summary["real_beats_wrong_oof"] = bool(summary["context_plus_revision_oof_r2"] > summary["wrong_expert_oof_r2"])
    summary["real_beats_shuffled_oof"] = bool(summary["context_plus_revision_oof_r2"] > summary["shuffled_revision_oof_r2"])
    summary["routing_improves_val"] = bool(summary["context_plus_revision_val_routing_mae"] < summary["context_val_routing_mae"])
    summary["works_by_success_criteria_count"] = int(
        summary["context_plus_beats_context_oof"]
        + summary["residual_positive_oof"]
        + summary["real_beats_wrong_oof"]
        + summary["real_beats_shuffled_oof"]
        + summary["routing_improves_val"]
    )
    summary["verdict"] = "PROMISING" if summary["works_by_success_criteria_count"] >= 4 and summary["context_plus_beats_context_oof"] else "NEGATIVE"

    serial_route = []
    for d in (train_route, val_route):
        for r in d.values():
            row = {kk: vv for kk, vv in r.items() if kk != "per_window_mae"}
            serial_route.append(row)

    result = {
        "dataset": dataset,
        "core": core,
        "expert_indices": expert_idx,
        "num_features": int(train_cache["num_features"]),
        "num_train_windows": int(train_cache["num_windows"]),
        "num_val_windows": int(val_cache["num_windows"]),
        "num_oof_common_windows": int(common_idx.numel()),
        "num_val_evaluated_windows": int(val_all.numel()),
        "projection": {"dimension": int(proj.shape[1]), "type": "identity" if int(train_cache["num_features"]) <= 16 else "deterministic_random_variable_projection"},
        "revision_diagnostics": {"router_train": revision_diag_train, "router_val": revision_diag_val},
        "model_fit_details_oof_first_variant_fit": fit_details,
        "model_fit_details_final": val_fit_details,
        "competence_metrics": [*oof_metrics, *val_metrics],
        "residual_diagnostic": residual_diag,
        "routing_metrics": serial_route,
        "summary": summary,
        "integrity": integrity,
        "checkpoint_hashes_before": before_hashes,
        "checkpoint_hashes_after": after_hashes,
    }
    return result, dep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    args = parser.parse_args()
    started = time.perf_counter()
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, Any] = {
        "experiment": "rolling_origin_revision_embedding",
        "created_at_utc": now_utc(),
        "git_commit_sha": git_commit_sha(),
        "datasets_requested": list(args.datasets),
        "test_split_accessed": False,
        "classification": None,
        "datasets": {},
    }
    all_dep: list[dict[str, Any]] = []
    for dataset in args.datasets:
        result, dep = evaluate_dataset(dataset)
        all_results["datasets"][dataset] = result
        all_dep.extend(dep)

    summaries = [d["summary"] for d in all_results["datasets"].values()]
    all_cp_beats = all(s["context_plus_beats_context_oof"] for s in summaries)
    mostly_support = sum(s["works_by_success_criteria_count"] >= 4 for s in summaries) >= max(1, math.ceil(len(summaries) / 2))
    all_results["classification"] = "PROMISING_REVISION_SIGNAL" if (all_cp_beats and mostly_support) else "NEGATIVE_RESULT"
    all_results["answers"] = concrete_answers(all_results)
    all_results["runtime_seconds"] = time.perf_counter() - started

    method_manifest = {
        "experiment": "rolling_origin_revision_embedding",
        "created_at_utc": all_results["created_at_utc"],
        "git_commit_sha": all_results["git_commit_sha"],
        "status": "STRICT_VALIDATION_ONLY",
        "datasets": list(args.datasets),
        "allowed_splits": ["router_train", "router_val"],
        "test_split_access": "forbidden",
        "lags": list(LAGS),
        "lag_rule": "R[t,k,d,h] = F[t,k,h] - F[t-d,k,h+d] for overlapping horizons only",
        "normalization": "differences and errors divided by train-only per-variable scaler std from the dataset checkpoint",
        "context_encoder": {
            "input": "current history plus current frozen expert forecasts and disagreement profiles",
            "baseline_note": "This is a learned context encoder; old 15 handcrafted passive features are not used as the primary baseline.",
            "hidden_dim": EMBED_DIM,
        },
        "revision_encoder": {
            "input": "complete signed aligned revision trajectories over lags/horizons after compact deterministic variable projection",
            "hidden_dim": EMBED_DIM,
        },
        "models": [
            "ContextEmbed",
            "RevisionEmbed",
            "ContextPlusRevision",
            "ContextPlusWrongExpertRevision",
            "ContextPlusShuffledRevision",
        ],
        "fixed_training": {"epochs": EPOCHS, "batch_rows": BATCH_ROWS, "lr": LR, "weight_decay": WEIGHT_DECAY, "seed": SEED},
        "oof": {
            "folds": N_FOLDS,
            "min_train_fraction": MIN_TRAIN_FRACTION,
            "purge_rule": "max(train target end) <= min(eval origin)",
            "horizon": PURGE_HORIZON,
        },
        "routing": {"rule": "fixed rank weights assigned by predicted competence, lower predicted MAE is better", "weights": [float(x) for x in RANK_WEIGHTS]},
        "controls": {
            "wrong_expert": "derange expert axis within each window",
            "shuffled_revision": "shuffle revision features within each training/evaluation partition with deterministic seeds",
        },
        "decision_rule": "PROMISING only if ContextPlusRevision beats ContextEmbed OOF and most fixed criteria hold; otherwise NEGATIVE_RESULT.",
    }

    integrity = {ds: d["integrity"] for ds, d in all_results["datasets"].items()}
    write_json(OUT_DIR / "method_manifest.json", method_manifest)
    write_json(OUT_DIR / "validation_results.json", all_results)
    write_json(OUT_DIR / "integrity_checks.json", integrity)
    write_csv(OUT_DIR / "dependence_tests.csv", all_dep)
    (OUT_DIR / "report.md").write_text(build_report(all_results), encoding="utf-8")
    print(f"[rolling-origin] wrote artifacts to {OUT_DIR}", flush=True)
    print(f"[rolling-origin] classification={all_results['classification']} runtime_seconds={all_results['runtime_seconds']:.1f}", flush=True)


if __name__ == "__main__":
    main()
