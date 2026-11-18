"""Autonomous COSTAR-TS oracle-weight tournament.

This runner uses frozen ETTh1 walk-forward caches only.  It never loads a test
cache and never constructs teacher/retrieval labels from validation targets.
Results are written after every trial so the run can be resumed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


EXPERTS = ("DLinear", "PatchTST", "iTransformer", "TimesNet", "ModernTCN")
FIXED3 = ("PatchTST", "iTransformer", "TimesNet")
DYNAMIC_FIXED3_REFERENCE_MAE = 0.3663418054580688
FIXED3_REFERENCE_MAE = 0.36726489663124084
DYNAMIC_BASELINE_ROOT = ROOT / "results/router_summary/costarts_walkforward/fixed3_dynamic_weighting_5seed"


@dataclass(frozen=True)
class TrialConfig:
    family: str
    name: str
    seed: int = 7
    epochs: int = 8
    lr: float = 1e-3
    batch_size: int = 512
    teacher_lambda: float = 0.0
    teacher_weight: float = 1.0
    forecast_weight: float = 1.0
    residual_weight: float = 0.001
    residual_scale: float = 0.20
    k: int = 16
    temperature: float = 0.05
    num_prototypes: int = 8
    rank: int = 1
    forgetting: float = 0.99
    eta: float = 0.2
    feature_mix: str = "full"


class Fixed3WindowDataset(Dataset):
    def __init__(self, cache: Mapping[str, Any], fixed3_indices: Sequence[int]) -> None:
        self.histories = cache["histories"].to(torch.float32)
        self.forecasts = cache["prediction_stack"][..., list(fixed3_indices)].to(torch.float32)
        self.targets = cache["targets"].to(torch.float32)
        self.masks = cache["target_masks"].to(torch.bool)
        self.starts = cache["absolute_window_starts"].to(torch.long)

    def __len__(self) -> int:
        return int(self.histories.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "history": self.histories[index],
            "forecasts": self.forecasts[index],
            "target": self.targets[index],
            "mask": self.masks[index],
            "start": self.starts[index],
            "index": torch.tensor(index, dtype=torch.long),
        }


class WeightStudent(nn.Module):
    def __init__(
        self,
        global_weights: Sequence[float],
        input_len: int,
        horizon: int,
        num_features: int,
        hidden_dim: int = 64,
        mode: str = "direct",
        num_prototypes: int = 8,
        rank: int = 1,
        residual_scale: float = 0.2,
        feature_mix: str = "full",
    ) -> None:
        super().__init__()
        self.mode = mode
        self.horizon = int(horizon)
        self.rank = int(rank)
        self.residual_scale = float(residual_scale)
        self.feature_mix = feature_mix
        base = torch.tensor(list(global_weights), dtype=torch.float32)
        self.register_buffer("global_weights", base / base.sum().clamp_min(1e-8))
        self.history_projection = nn.Linear(num_features, hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.history_encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.pos = nn.Parameter(torch.zeros(1, input_len, hidden_dim))
        flat_dim = horizon * num_features
        self.forecast_encoder = nn.Sequential(
            nn.Linear(flat_dim * 4 + 16, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.delta_head = nn.Linear(hidden_dim, 3)
        self.prototype_head = nn.Linear(hidden_dim, num_prototypes)
        self.horizon_a = nn.Linear(hidden_dim, horizon * rank)
        self.horizon_b = nn.Linear(hidden_dim, rank * 3)
        nn.init.normal_(self.pos, std=0.02)

    def encode(self, history: torch.Tensor, forecasts: torch.Tensor) -> torch.Tensor:
        hist = self.history_projection(history) + self.pos[:, : history.shape[1]]
        hist = self.history_encoder(hist).mean(dim=1)
        route_forecasts = forecasts
        if self.feature_mix == "forecast":
            hist = torch.zeros_like(hist)
        elif self.feature_mix == "history":
            route_forecasts = torch.zeros_like(forecasts)
        equal = route_forecasts.mean(dim=-1)
        scalars = forecast_scalars(route_forecasts)
        flat = torch.cat([route_forecasts[..., i].flatten(1) for i in range(3)] + [equal.flatten(1), scalars], dim=1)
        frep = self.forecast_encoder(flat)
        return self.head(torch.cat((hist, frep), dim=1))

    def forward(self, history: torch.Tensor, forecasts: torch.Tensor, prototypes: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        rep = self.encode(history, forecasts)
        base_logits = self.global_weights.clamp_min(1e-8).log().to(rep.device, rep.dtype)
        if self.mode == "prototype":
            logits = self.prototype_head(rep)
            proto_weights = torch.softmax(logits, dim=1) @ prototypes.to(rep.device, rep.dtype)
            weights = proto_weights / proto_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
            return {"weights": weights, "logits": logits}
        if self.mode == "prototype_residual":
            logits = self.prototype_head(rep)
            proto_weights = torch.softmax(logits, dim=1) @ prototypes.to(rep.device, rep.dtype)
            delta = self.residual_scale * torch.tanh(self.delta_head(rep))
            weights = (proto_weights + delta).clamp_min(1e-5)
            weights = weights / weights.sum(dim=1, keepdim=True)
            return {"weights": weights, "logits": logits, "delta": delta}
        if self.mode == "horizon":
            a = self.horizon_a(rep).view(rep.shape[0], self.horizon, self.rank)
            b = self.horizon_b(rep).view(rep.shape[0], self.rank, 3)
            delta = self.residual_scale * torch.tanh(a @ b)
            base = self.global_weights.to(rep.device, rep.dtype).view(1, 1, 3)
            weights = (base + delta).clamp_min(1e-5)
            weights = weights / weights.sum(dim=2, keepdim=True)
            return {"horizon_weights": weights, "delta": delta}
        delta = self.residual_scale * torch.tanh(self.delta_head(rep))
        weights = torch.softmax(base_logits[None, :] + delta, dim=1)
        return {"weights": weights, "delta": delta}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_cache(path: Path, expected_role: str) -> dict[str, Any]:
    if "test" in str(path).lower():
        raise ValueError(f"Refusing test cache path: {path}")
    cache = torch.load(path, map_location="cpu", weights_only=False)
    role = cache.get("cache_role", cache.get("split_role"))
    if role != expected_role:
        raise ValueError(f"{path} role={role!r}, expected {expected_role!r}")
    if "test" in str(role).lower():
        raise ValueError(f"Refusing test cache role: {role}")
    return cache


def load_std(path: Path, num_features: int) -> torch.Tensor:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "scaler_std" not in ckpt:
        return torch.ones(num_features, dtype=torch.float32)
    return ckpt["scaler_std"].to(torch.float32)


def fixed3_indices(cache: Mapping[str, Any]) -> list[int]:
    names = list(cache["expert_names"])
    return [names.index(name) for name in FIXED3]


def load_dynamic_baseline_per_window(seed: int) -> torch.Tensor | None:
    path = DYNAMIC_BASELINE_ROOT / f"seed_{seed}" / "validation_per_window.csv"
    if not path.exists():
        return None
    values = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            values.append(float(row["mae"]))
    return torch.tensor(values, dtype=torch.float32)


def weighted_forecast(forecasts: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if weights.ndim == 2:
        return (forecasts * weights[:, None, None, :]).sum(dim=-1)
    return (forecasts * weights[:, :, None, :]).sum(dim=-1)


def sample_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    std = std.to(pred.device, pred.dtype).view(1, 1, -1)
    mask_f = mask.to(pred.dtype)
    denom = mask_f.flatten(1).sum(dim=1).clamp_min(1.0)
    return (((pred - target) / std).abs() * mask_f).flatten(1).sum(dim=1) / denom


def sample_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    std = std.to(pred.device, pred.dtype).view(1, 1, -1)
    mask_f = mask.to(pred.dtype)
    denom = mask_f.flatten(1).sum(dim=1).clamp_min(1.0)
    return (((pred - target) / std).square() * mask_f).flatten(1).sum(dim=1) / denom


def forecast_scalars(forecasts: torch.Tensor) -> torch.Tensor:
    a, b, c = forecasts[..., 0], forecasts[..., 1], forecasts[..., 2]
    tensors = [a - b, a - c, b - c, (a - b).abs(), (a - c).abs(), (b - c).abs(), forecasts.var(dim=-1, unbiased=False), forecasts.max(dim=-1).values - forecasts.min(dim=-1).values]
    vals = []
    for x in tensors:
        vals.append(x.mean(dim=(1, 2)))
        vals.append(x.abs().amax(dim=(1, 2)))
    return torch.stack(vals, dim=1)


def simplex_grid(step: float) -> torch.Tensor:
    units = int(round(1.0 / step))
    rows = []
    for a in range(units + 1):
        for b in range(units + 1 - a):
            c = units - a - b
            rows.append((a / units, b / units, c / units))
    return torch.tensor(rows, dtype=torch.float32)


def oracle_weights_grid(
    forecasts: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    std: torch.Tensor,
    global_weights: torch.Tensor,
    reg_lambda: float,
    step: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    grid = simplex_grid(step)
    best_loss = torch.full((forecasts.shape[0],), float("inf"))
    best_mae = torch.full_like(best_loss, float("inf"))
    best_idx = torch.zeros((forecasts.shape[0],), dtype=torch.long)
    for start in range(0, grid.shape[0], 512):
        w = grid[start : start + 512]
        pred = (forecasts.unsqueeze(-1) * w.T.view(1, 1, 1, 3, -1)).sum(dim=3)
        stdv = std.view(1, 1, -1, 1)
        mask = masks.to(pred.dtype).unsqueeze(-1)
        denom = mask.flatten(1, 2).sum(dim=1).clamp_min(1.0)
        mae = ((((pred - targets.unsqueeze(-1)) / stdv).abs() * mask).flatten(1, 2).sum(dim=1) / denom)
        penalty = float(reg_lambda) * (w - global_weights.view(1, 3)).square().sum(dim=1)
        loss = mae + penalty.view(1, -1)
        vals, idx = loss.min(dim=1)
        update = vals < best_loss
        best_loss[update] = vals[update]
        best_mae[update] = mae[torch.where(update)[0], idx[update]]
        best_idx[update] = idx[update] + start
    return grid[best_idx], best_mae


def horizon_oracle_weights(
    forecasts: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    std: torch.Tensor,
    global_weights: torch.Tensor,
    reg_lambda: float,
    step: float,
) -> torch.Tensor:
    per_horizon = []
    for h in range(forecasts.shape[1]):
        w, _ = oracle_weights_grid(
            forecasts[:, h : h + 1],
            targets[:, h : h + 1],
            masks[:, h : h + 1],
            std,
            global_weights,
            reg_lambda,
            step,
        )
        per_horizon.append(w)
    return torch.stack(per_horizon, dim=1)


def kmeans(x: torch.Tensor, k: int, seed: int, iterations: int = 30) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    centers = x[torch.randperm(x.shape[0], generator=gen)[:k]].clone()
    labels = torch.zeros(x.shape[0], dtype=torch.long)
    for _ in range(iterations):
        dist = torch.cdist(x, centers)
        labels = dist.argmin(dim=1)
        for i in range(k):
            if bool((labels == i).any()):
                centers[i] = x[labels == i].mean(dim=0)
    centers = centers.clamp_min(1e-5)
    centers = centers / centers.sum(dim=1, keepdim=True)
    return centers, labels


def ensure_teachers(args: argparse.Namespace, train_cache: Mapping[str, Any], std: torch.Tensor, out_dir: Path) -> dict[str, Any]:
    teacher_dir = out_dir / "teacher_cache"
    teacher_dir.mkdir(parents=True, exist_ok=True)
    path = teacher_dir / f"teachers_step_{str(args.teacher_grid_step).replace('.', 'p')}.pt"
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False)
    forecasts = train_cache["prediction_stack"][..., fixed3_indices(train_cache)].to(torch.float32)
    targets = train_cache["targets"].to(torch.float32)
    masks = train_cache["target_masks"].to(torch.bool)
    global_weights = torch.tensor(args.global_weights, dtype=torch.float32)
    cache: dict[str, Any] = {"global_weights": global_weights, "step": args.teacher_grid_step}
    for lam in args.teacher_lambdas:
        weights, maes = oracle_weights_grid(forecasts, targets, masks, std, global_weights, lam, args.teacher_grid_step)
        cache[f"weights_lambda_{lam}"] = weights
        cache[f"mae_lambda_{lam}"] = maes
    for lam in args.horizon_lambdas:
        cache[f"horizon_weights_lambda_{lam}"] = horizon_oracle_weights(forecasts, targets, masks, std, global_weights, lam, args.horizon_grid_step)
    torch.save(cache, path)
    return cache


def evaluate_weights(cache: Mapping[str, Any], weights: torch.Tensor, std: torch.Tensor) -> dict[str, Any]:
    forecasts = cache["prediction_stack"][..., fixed3_indices(cache)].to(torch.float32)
    pred = weighted_forecast(forecasts, weights.to(torch.float32))
    mae = sample_mae(pred, cache["targets"].to(torch.float32), cache["target_masks"].to(torch.bool), std)
    mse = sample_mse(pred, cache["targets"].to(torch.float32), cache["target_masks"].to(torch.bool), std)
    return {"mae": float(mae.mean()), "mse": float(mse.mean()), "per_window_mae": mae}


def retrieval_features_raw(cache: Mapping[str, Any], mix: str) -> torch.Tensor:
    forecasts = cache["prediction_stack"][..., fixed3_indices(cache)].to(torch.float32)
    pieces = []
    if mix in {"history", "full"}:
        h = cache["histories"].to(torch.float32)
        pieces.append(torch.cat((h.mean(dim=1), h.std(dim=1, unbiased=False), h[:, -1], h[:, -1] - h[:, 0]), dim=1))
    if mix in {"forecast", "full"}:
        pieces.append(torch.cat([forecasts[..., i].flatten(1) for i in range(3)], dim=1))
        pieces.append(forecast_scalars(forecasts))
    return torch.cat(pieces, dim=1)


def fit_retrieval_scaler(x_train: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    return mean, std


def apply_retrieval_scaler(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std


def train_student(trial: TrialConfig, train_cache: Mapping[str, Any], val_cache: Mapping[str, Any], teachers: Mapping[str, Any], std: torch.Tensor, out_dir: Path, device: torch.device) -> dict[str, Any]:
    set_seed(trial.seed)
    global_weights = torch.tensor(args_global_weights(), dtype=torch.float32)
    train_ds = Fixed3WindowDataset(train_cache, fixed3_indices(train_cache))
    val_ds = Fixed3WindowDataset(val_cache, fixed3_indices(val_cache))
    model_mode = "direct"
    prototypes = None
    proto_labels = None
    teacher = teachers[f"weights_lambda_{trial.teacher_lambda}"]
    horizon_teacher = None
    if trial.family.startswith("prototype"):
        model_mode = "prototype_residual" if "residual" in trial.family else "prototype"
        prototypes, proto_labels = kmeans(teacher, trial.num_prototypes, trial.seed)
    if trial.family == "horizon":
        model_mode = "horizon"
        horizon_teacher = teachers[f"horizon_weights_lambda_{trial.teacher_lambda}"]
    model = WeightStudent(
        global_weights,
        int(train_cache["input_len"]),
        int(train_cache["forecast_horizon"]),
        int(train_cache["num_features"]),
        mode=model_mode,
        num_prototypes=trial.num_prototypes,
        rank=trial.rank,
        residual_scale=trial.residual_scale,
        feature_mix=trial.feature_mix,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=trial.lr)
    loader = DataLoader(train_ds, batch_size=trial.batch_size, shuffle=True)
    best = None
    best_state = None
    start_time = time.time()
    for epoch in range(1, trial.epochs + 1):
        model.train()
        for batch in loader:
            hist = batch["history"].to(device)
            forecasts = batch["forecasts"].to(device)
            target = batch["target"].to(device)
            mask = batch["mask"].to(device)
            idx = batch["index"]
            out = model(hist, forecasts, prototypes=prototypes)
            if model_mode == "horizon":
                weights = out["horizon_weights"]
                teach = horizon_teacher[idx].to(device)
                smooth = (weights[:, 1:] - weights[:, :-1]).square().mean()
                teacher_loss = F.smooth_l1_loss(weights, teach)
            else:
                weights = out["weights"]
                teach = teacher[idx].to(device)
                teacher_loss = F.smooth_l1_loss(weights, teach)
                smooth = torch.zeros((), device=device)
                if proto_labels is not None:
                    logits = out["logits"]
                    labels = proto_labels[idx].to(device)
                    teacher_loss = teacher_loss + F.cross_entropy(logits, labels)
            pred = weighted_forecast(forecasts, weights)
            forecast_loss = sample_mae(pred, target, mask, std.to(device)).mean()
            residual_loss = (weights - global_weights.to(device).view(*((1,) * (weights.ndim - 1)), 3)).square().mean()
            loss = trial.forecast_weight * forecast_loss + trial.teacher_weight * teacher_loss + trial.residual_weight * residual_loss + 0.01 * smooth
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        metrics = eval_model(model, val_ds, std, device, prototypes)
        if best is None or metrics["mae"] < best["mae"]:
            best = metrics | {"epoch": epoch}
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    assert best is not None and best_state is not None
    ckpt_dir = out_dir / "checkpoints" / trial.name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"trial": asdict(trial), "state_dict": best_state, "prototypes": prototypes, "metrics": best}, ckpt_dir / "best.pt")
    return best | {"training_time_sec": time.time() - start_time}


@torch.no_grad()
def eval_model(model: WeightStudent, dataset: Fixed3WindowDataset, std: torch.Tensor, device: torch.device, prototypes: torch.Tensor | None = None) -> dict[str, Any]:
    model.eval()
    loader = DataLoader(dataset, batch_size=1024, shuffle=False)
    maes, mses = [], []
    for batch in loader:
        hist = batch["history"].to(device)
        forecasts = batch["forecasts"].to(device)
        out = model(hist, forecasts, prototypes=prototypes)
        weights = out.get("horizon_weights", out.get("weights"))
        pred = weighted_forecast(forecasts, weights)
        target = batch["target"].to(device)
        mask = batch["mask"].to(device)
        maes.append(sample_mae(pred, target, mask, std.to(device)).cpu())
        mses.append(sample_mse(pred, target, mask, std.to(device)).cpu())
    return {"mae": float(torch.cat(maes).mean()), "mse": float(torch.cat(mses).mean()), "per_window_mae": torch.cat(maes)}


def run_retrieval(trial: TrialConfig, train_cache: Mapping[str, Any], val_cache: Mapping[str, Any], teachers: Mapping[str, Any], std: torch.Tensor) -> dict[str, Any]:
    start_time = time.time()
    train_raw = retrieval_features_raw(train_cache, trial.feature_mix)
    val_raw = retrieval_features_raw(val_cache, trial.feature_mix)
    mean, feat_std = fit_retrieval_scaler(train_raw)
    train_raw = apply_retrieval_scaler(train_raw, mean, feat_std)
    val_raw = apply_retrieval_scaler(val_raw, mean, feat_std)
    teacher = teachers[f"weights_lambda_{trial.teacher_lambda}"]
    weights_out = []
    for start in range(0, val_raw.shape[0], 256):
        dist = torch.cdist(val_raw[start : start + 256], train_raw)
        vals, idx = torch.topk(dist, k=min(trial.k, dist.shape[1]), dim=1, largest=False)
        sims = torch.softmax(-vals / max(trial.temperature, 1e-6), dim=1)
        weights = (teacher[idx] * sims[:, :, None]).sum(dim=1)
        weights = weights.clamp_min(1e-5)
        weights_out.append(weights / weights.sum(dim=1, keepdim=True))
    weights_tensor = torch.cat(weights_out, dim=0)
    metrics = evaluate_weights(val_cache, weights_tensor, std)
    return metrics | {"training_time_sec": time.time() - start_time}


def run_online(trial: TrialConfig, val_cache: Mapping[str, Any], std: torch.Tensor) -> dict[str, Any]:
    start_time = time.time()
    forecasts = val_cache["prediction_stack"][..., fixed3_indices(val_cache)].to(torch.float32)
    targets = val_cache["targets"].to(torch.float32)
    masks = val_cache["target_masks"].to(torch.bool)
    starts = val_cache["absolute_window_starts"].to(torch.long)
    base = torch.tensor(args_global_weights(), dtype=torch.float32)
    weights = base.clone()
    pending: list[tuple[int, torch.Tensor]] = []
    final_weights = []
    expert_losses = []
    for i in range(forecasts.shape[0]):
        now = int(starts[i])
        still_pending = []
        for due, loss in pending:
            if due <= now:
                expert_losses.append(loss)
                avg_loss = torch.stack(expert_losses[-64:]).mean(dim=0)
                weights = base * float(trial.forgetting) + (1.0 - float(trial.forgetting)) * torch.softmax(-trial.eta * avg_loss, dim=0)
                weights = weights / weights.sum()
            else:
                still_pending.append((due, loss))
        pending = still_pending
        final_weights.append(weights.clone())
        per_expert = []
        for j in range(3):
            per_expert.append(sample_mae(forecasts[i : i + 1, :, :, j], targets[i : i + 1], masks[i : i + 1], std)[0])
        pending.append((now + int(val_cache["forecast_horizon"]), torch.stack(per_expert)))
    weights_tensor = torch.stack(final_weights)
    metrics = evaluate_weights(val_cache, weights_tensor, std)
    return metrics | {"training_time_sec": time.time() - start_time}


def args_global_weights() -> tuple[float, float, float]:
    return (0.36, 0.42, 0.22)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row if k != "per_window_mae"})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def bootstrap_diff(candidate: torch.Tensor, baseline: torch.Tensor, seed: int, samples: int = 2000) -> dict[str, Any]:
    gen = torch.Generator().manual_seed(seed)
    diff = candidate - baseline
    means = []
    n = diff.numel()
    for _ in range(samples):
        idx = torch.randint(0, n, (n,), generator=gen)
        means.append(float(diff[idx].mean()))
    means_t = torch.tensor(means)
    return {
        "mean_diff_candidate_minus_baseline": float(diff.mean()),
        "ci95_low": float(torch.quantile(means_t, 0.025)),
        "ci95_high": float(torch.quantile(means_t, 0.975)),
        "ci_excludes_zero": bool(torch.quantile(means_t, 0.975) < 0 or torch.quantile(means_t, 0.025) > 0),
    }


def trial_grid() -> list[TrialConfig]:
    trials: list[TrialConfig] = []
    for lam in (0.0, 0.001, 0.01):
        for tw in (0.2, 1.0):
            trials.append(TrialConfig("distill", f"distill_lam{lam}_tw{tw}", teacher_lambda=lam, teacher_weight=tw, epochs=6))
    for k in (4, 8, 16, 32):
        trials.append(TrialConfig("retrieval", f"retrieval_k{k}", k=k, teacher_lambda=0.001, temperature=0.1))
    for kp in (4, 8, 16):
        trials.append(TrialConfig("prototype", f"prototype_k{kp}", num_prototypes=kp, teacher_lambda=0.001, epochs=6))
        trials.append(TrialConfig("prototype_residual", f"prototype_residual_k{kp}", num_prototypes=kp, teacher_lambda=0.001, epochs=6))
    for rank in (1, 2):
        trials.append(TrialConfig("horizon", f"horizon_rank{rank}", rank=rank, teacher_lambda=0.001, epochs=6, residual_scale=0.1))
    for eta in (0.05, 0.2, 1.0):
        for forgetting in (0.9, 0.98, 0.995):
            trials.append(TrialConfig("online", f"online_eta{eta}_forget{forgetting}", eta=eta, forgetting=forgetting))
    return trials


def phase2_grid() -> list[TrialConfig]:
    trials: list[TrialConfig] = []
    for lam in (0.0, 0.001, 0.01):
        for kp in (4, 8, 16, 32):
            for scale in (0.05, 0.10, 0.20, 0.30):
                for rw in (0.0001, 0.001, 0.01):
                    trials.append(
                        TrialConfig(
                            "prototype_residual",
                            f"phase2_protores_lam{lam}_k{kp}_scale{scale}_rw{rw}",
                            num_prototypes=kp,
                            teacher_lambda=lam,
                            residual_scale=scale,
                            residual_weight=rw,
                            epochs=10,
                        )
                    )
    for lam in (0.0, 0.001, 0.01):
        for kp in (4, 8, 16, 32):
            trials.append(TrialConfig("prototype", f"phase2_proto_lam{lam}_k{kp}", num_prototypes=kp, teacher_lambda=lam, epochs=10))
    for k in (4, 8, 16, 32, 64):
        for temp in (0.02, 0.05, 0.10, 0.20):
            trials.append(TrialConfig("retrieval", f"phase2_retrieval_k{k}_temp{temp}", k=k, temperature=temp, teacher_lambda=0.001))
    for eta in (0.5, 1.0, 2.0, 4.0):
        for forgetting in (0.85, 0.9, 0.95, 0.98, 0.995):
            trials.append(TrialConfig("online", f"phase2_online_eta{eta}_forget{forgetting}", eta=eta, forgetting=forgetting))
    return trials


def finalist_grid() -> list[TrialConfig]:
    seeds = (7, 11, 13, 17, 19)
    configs = [
        TrialConfig("prototype_residual", "final_phase2_protores_lam0.01_k32_scale0.3_rw0.01", num_prototypes=32, teacher_lambda=0.01, residual_scale=0.30, residual_weight=0.01, epochs=10),
        TrialConfig("prototype_residual", "final_phase2_protores_lam0.01_k16_scale0.3_rw0.001", num_prototypes=16, teacher_lambda=0.01, residual_scale=0.30, residual_weight=0.001, epochs=10),
        TrialConfig("prototype_residual", "final_protores_k16", num_prototypes=16, teacher_lambda=0.001, residual_scale=0.20, residual_weight=0.001, epochs=10),
        TrialConfig("prototype_residual", "final_protores_k4", num_prototypes=4, teacher_lambda=0.001, residual_scale=0.20, residual_weight=0.001, epochs=10),
        TrialConfig("prototype_residual", "final_protores_k4_lowrw", num_prototypes=4, teacher_lambda=0.001, residual_scale=0.20, residual_weight=0.0001, epochs=10),
    ]
    trials: list[TrialConfig] = []
    for config in configs:
        for seed in seeds:
            trials.append(TrialConfig(**(asdict(config) | {"name": f"{config.name}_seed{seed}", "seed": seed})))
    return trials


def ablation_grid() -> list[TrialConfig]:
    return [
        TrialConfig("prototype_residual", "ablate_full_teacher_forecast_constrained", seed=7, num_prototypes=16, teacher_lambda=0.01, residual_scale=0.30, residual_weight=0.001, epochs=10, feature_mix="full"),
        TrialConfig("prototype_residual", "ablate_history_only", seed=7, num_prototypes=16, teacher_lambda=0.01, residual_scale=0.30, residual_weight=0.001, epochs=10, feature_mix="history"),
        TrialConfig("prototype_residual", "ablate_forecasts_only", seed=7, num_prototypes=16, teacher_lambda=0.01, residual_scale=0.30, residual_weight=0.001, epochs=10, feature_mix="forecast"),
        TrialConfig("prototype_residual", "ablate_teacher_only", seed=7, num_prototypes=16, teacher_lambda=0.01, residual_scale=0.30, residual_weight=0.001, epochs=10, forecast_weight=0.0, feature_mix="full"),
        TrialConfig("prototype_residual", "ablate_forecast_loss_only", seed=7, num_prototypes=16, teacher_lambda=0.01, residual_scale=0.30, residual_weight=0.001, epochs=10, teacher_weight=0.0, feature_mix="full"),
        TrialConfig("prototype_residual", "ablate_no_residual_constraint", seed=7, num_prototypes=16, teacher_lambda=0.01, residual_scale=0.30, residual_weight=0.0, epochs=10, feature_mix="full"),
        TrialConfig("prototype", "ablate_prototype_only", seed=7, num_prototypes=16, teacher_lambda=0.01, epochs=10, feature_mix="full"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", default="cache/costarts_walkforward/router_train_20_60_cache.pt")
    parser.add_argument("--val-cache", default="cache/costarts_walkforward/router_val_60_80_cache.pt")
    parser.add_argument("--normalizer-checkpoint", default="checkpoints/costarts_walkforward/final_60/DLinear/best_expert.pt")
    parser.add_argument("--out-dir", default="experiments/oracle_weight_tournament")
    parser.add_argument("--time-budget-hours", type=float, default=6.0)
    parser.add_argument("--teacher-grid-step", type=float, default=0.02)
    parser.add_argument("--horizon-grid-step", type=float, default=0.05)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--phase", choices=("screen", "phase2", "finalists", "ablation", "all"), default="screen")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.teacher_lambdas = (0.0, 0.001, 0.01)
    args.horizon_lambdas = (0.001,)
    args.global_weights = args_global_weights()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)
    train_path = ROOT / args.train_cache
    val_path = ROOT / args.val_cache
    train_cache = load_cache(train_path, "router_train_20_60")
    val_cache = load_cache(val_path, "router_val_60_80")
    if int(val_cache["absolute_window_starts"].min()) < 8640:
        raise ValueError("Validation cache starts before expected validation block")
    std = load_std(ROOT / args.normalizer_checkpoint, int(train_cache["num_features"]))
    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_cache": args.train_cache,
        "val_cache": args.val_cache,
        "train_sha256": sha256_file(train_path),
        "val_sha256": sha256_file(val_path),
        "global_weights": args.global_weights,
        "safety": "NO TEST DATA USED; TEACHERS/RETRIEVAL USE ROUTER-TRAIN ONLY",
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    teachers = ensure_teachers(args, train_cache, std, out_dir)
    all_trials_path = out_dir / "all_trials.csv"
    done = {row["name"] for row in read_csv_rows(all_trials_path)}
    rows: list[dict[str, Any]] = []
    if all_trials_path.exists():
        rows.extend(read_csv_rows(all_trials_path))
    base_metrics = evaluate_weights(val_cache, torch.tensor(args.global_weights).view(1, 3).expand(int(val_cache["num_windows"]), -1), std)
    base_row = {"family": "baseline", "name": "global_0p36_0p42_0p22", "mae": base_metrics["mae"], "mse": base_metrics["mse"], "diff_vs_dynamic_ref": base_metrics["mae"] - DYNAMIC_FIXED3_REFERENCE_MAE}
    if "global_0p36_0p42_0p22" not in done:
        rows.append(base_row)
        write_csv(all_trials_path, rows)
        done.add("global_0p36_0p42_0p22")
    if args.phase == "screen":
        trials = trial_grid()
    elif args.phase == "phase2":
        trials = phase2_grid()
    elif args.phase == "finalists":
        trials = finalist_grid()
    elif args.phase == "ablation":
        trials = ablation_grid()
    else:
        trials = trial_grid() + phase2_grid() + finalist_grid() + ablation_grid()
    if args.smoke:
        trials = trials[:3]
    deadline = time.time() + args.time_budget_hours * 3600.0
    device = torch.device(args.device)
    global_baseline_per_window = base_metrics["per_window_mae"]
    per_window_dir = out_dir / "per_window"
    per_window_dir.mkdir(exist_ok=True)
    for trial in trials:
        if trial.name in done:
            continue
        if time.time() > deadline:
            break
        print(f"[trial] {trial.name}", flush=True)
        t0 = time.time()
        if trial.family in {"distill", "prototype", "prototype_residual", "horizon"}:
            metrics = train_student(trial, train_cache, val_cache, teachers, std, out_dir, device)
        elif trial.family == "retrieval":
            metrics = run_retrieval(trial, train_cache, val_cache, teachers, std)
        elif trial.family == "online":
            metrics = run_online(trial, val_cache, std)
        else:
            raise ValueError(trial.family)
        per_window = metrics.pop("per_window_mae")
        torch.save(per_window, per_window_dir / f"{trial.name}.pt")
        dynamic_baseline = load_dynamic_baseline_per_window(trial.seed)
        if dynamic_baseline is not None:
            if dynamic_baseline.shape != per_window.shape:
                raise ValueError(f"Dynamic baseline shape {dynamic_baseline.shape} does not match candidate {per_window.shape}")
            dynamic_boot = bootstrap_diff(per_window, dynamic_baseline, trial.seed, samples=500 if args.smoke else 2000)
            diff_vs_dynamic_seed = metrics["mae"] - float(dynamic_baseline.mean())
        else:
            dynamic_boot = {}
            diff_vs_dynamic_seed = metrics["mae"] - DYNAMIC_FIXED3_REFERENCE_MAE
        global_boot = bootstrap_diff(per_window, global_baseline_per_window, trial.seed, samples=500 if args.smoke else 2000)
        row = asdict(trial) | metrics | {
            "diff_vs_dynamic_ref": metrics["mae"] - DYNAMIC_FIXED3_REFERENCE_MAE,
            "diff_vs_dynamic_seed": diff_vs_dynamic_seed,
            "diff_vs_global_baseline": metrics["mae"] - base_metrics["mae"],
            "wall_time_sec": time.time() - t0,
            **{f"dynamic_{k}": v for k, v in dynamic_boot.items()},
            **{f"global_{k}": v for k, v in global_boot.items()},
        }
        rows.append(row)
        write_csv(all_trials_path, rows)
        done.add(trial.name)
        leaderboard = sorted(rows, key=lambda r: float(r["mae"]))
        write_csv(out_dir / "leaderboard.csv", leaderboard)
        (out_dir / "finalists.json").write_text(json.dumps(leaderboard[:10], indent=2), encoding="utf-8")
    leaderboard = sorted(rows, key=lambda r: float(r["mae"]))
    write_csv(out_dir / "leaderboard.csv", leaderboard)
    bootstrap = {
        row["name"]: {k: row[k] for k in row if k.startswith("dynamic_") or k.startswith("global_")}
        for row in leaderboard
        if any(k.startswith("dynamic_") or k.startswith("global_") for k in row)
    }
    (out_dir / "bootstrap_results.json").write_text(json.dumps(bootstrap, indent=2), encoding="utf-8")
    report = {
        "best": leaderboard[0] if leaderboard else None,
        "dynamic_fixed3_reference_mae": DYNAMIC_FIXED3_REFERENCE_MAE,
        "fixed3_reference_mae": FIXED3_REFERENCE_MAE,
        "num_trials": len(leaderboard),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "safety": meta["safety"],
    }
    (out_dir / "final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
