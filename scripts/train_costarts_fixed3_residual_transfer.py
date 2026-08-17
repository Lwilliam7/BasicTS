"""Residual dynamic fixed-3 weighting with optional ETTh2 pretraining transfer.

This script always forecasts with PatchTST, iTransformer, and TimesNet.  It
uses a dataset-level global weight vector, then learns per-window residual
logit adjustments from history and forecast-state features.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_sequential_costarts_full_walkforward import CacheWindowDataset


FIXED3_NAMES = ("PatchTST", "iTransformer", "TimesNet")
TRANSFER_MODES = ("none", "encoder", "full")


class ResidualFixed3WeightRouter(nn.Module):
    def __init__(
        self,
        global_weights: Sequence[float],
        input_len: int = 96,
        horizon: int = 12,
        num_features: int = 7,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.input_len = int(input_len)
        self.horizon = int(horizon)
        self.num_features = int(num_features)
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        global_tensor = torch.tensor(list(global_weights), dtype=torch.float32)
        if global_tensor.numel() != 3:
            raise ValueError("global_weights must have length 3")
        global_tensor = global_tensor / global_tensor.sum().clamp_min(1e-8)
        self.register_buffer("global_weights", global_tensor)
        self.history_encoder = nn.Sequential(
            nn.Conv1d(num_features, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.GroupNorm(1, hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=4, dilation=2),
            nn.GELU(),
            nn.GroupNorm(1, hidden_dim),
            nn.AdaptiveAvgPool1d(1),
        )
        self.history_projection = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
        )
        flat_dim = horizon * num_features
        self.forecast_encoder = nn.Sequential(
            nn.Linear(flat_dim * 4, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
        )
        self.scalar_encoder = nn.Sequential(
            nn.Linear(12, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
        )
        self.delta_head = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, history: torch.Tensor, forecasts: torch.Tensor) -> dict[str, torch.Tensor]:
        history_rep = self.history_encoder(history.transpose(1, 2)).squeeze(-1)
        history_rep = self.history_projection(history_rep)
        equal_forecast = forecasts.mean(dim=-1)
        flat = torch.cat([forecasts[..., i].flatten(1) for i in range(3)] + [equal_forecast.flatten(1)], dim=1)
        forecast_rep = self.forecast_encoder(flat)
        scalar_rep = self.scalar_encoder(forecast_state_scalars(forecasts))
        delta_logits = self.delta_head(torch.cat((history_rep, forecast_rep, scalar_rep), dim=1))
        base_logits = self.global_weights.clamp_min(1e-8).log().to(delta_logits.device, delta_logits.dtype)
        weights = torch.softmax(base_logits[None, :] + delta_logits, dim=1)
        return {"weights": weights, "delta_logits": delta_logits}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_cache(path: Path, expected_role: str | None = None) -> dict[str, Any]:
    if "test" in str(path).lower():
        raise ValueError(f"Refusing test cache path: {path}")
    cache = torch.load(path, map_location="cpu", weights_only=False)
    role = cache.get("cache_role", cache.get("split_role", ""))
    if "test" in str(role).lower():
        raise ValueError(f"Refusing test cache role: {role}")
    if expected_role is not None and role != expected_role:
        raise ValueError(f"{path} role={role!r}, expected {expected_role!r}")
    return cache


def load_metric_std(path: str | None, num_features: int) -> torch.Tensor:
    if not path:
        return torch.ones(num_features, dtype=torch.float32)
    checkpoint = torch.load(ROOT / path, map_location="cpu", weights_only=False)
    if "scaler_std" not in checkpoint:
        raise KeyError(f"{path} does not contain scaler_std")
    return checkpoint["scaler_std"].to(torch.float32)


def fixed3_indices(cache: Mapping[str, Any]) -> list[int]:
    names = list(cache["expert_names"])
    return [names.index(name) for name in FIXED3_NAMES]


def fixed3_stack(batch_or_cache: Mapping[str, Any], indices: Sequence[int], device: torch.device | None = None) -> torch.Tensor:
    stack = batch_or_cache["prediction_stack"].to(torch.float32)
    if device is not None:
        stack = stack.to(device)
    return stack[..., list(indices)]


def forecast_state_scalars(forecasts: torch.Tensor) -> torch.Tensor:
    a, b, c = forecasts[..., 0], forecasts[..., 1], forecasts[..., 2]
    tensors = [a - b, a - c, b - c, (a - b).abs(), (a - c).abs(), (b - c).abs(), forecasts.var(dim=-1, unbiased=False), forecasts.max(dim=-1).values - forecasts.min(dim=-1).values]
    features = []
    for tensor in tensors:
        features.append(tensor.mean(dim=(1, 2)))
        features.append(tensor.abs().amax(dim=(1, 2)))
    return torch.stack(features[:12], dim=1)


def weighted_forecast(forecasts: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (forecasts * weights[:, None, None, :]).sum(dim=-1)


def scaled_sample_mae(prediction: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    std = std.to(prediction.device, prediction.dtype).view(1, 1, -1)
    mask = masks.to(prediction.dtype)
    denom = mask.flatten(1).sum(dim=1).clamp_min(1.0)
    return (((prediction - targets) / std).abs() * mask).flatten(1).sum(dim=1) / denom


def scaled_sample_mse(prediction: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    std = std.to(prediction.device, prediction.dtype).view(1, 1, -1)
    mask = masks.to(prediction.dtype)
    denom = mask.flatten(1).sum(dim=1).clamp_min(1.0)
    return (((prediction - targets) / std).square() * mask).flatten(1).sum(dim=1) / denom


@torch.no_grad()
def best_global_weights(cache: Mapping[str, Any], metric_std: torch.Tensor, step: float = 0.01) -> list[float]:
    forecasts = fixed3_stack(cache, fixed3_indices(cache))
    targets = cache["targets"].to(torch.float32)
    masks = cache["target_masks"].to(torch.bool)
    weights = []
    units = int(round(1.0 / step))
    for a in range(units + 1):
        for b in range(units + 1 - a):
            c = units - a - b
            weights.append((a / units, b / units, c / units))
    W = torch.tensor(weights, dtype=torch.float32)
    best_mae = math.inf
    best_weight = W[0]
    for start in range(0, len(W), 512):
        w = W[start : start + 512]
        pred = (forecasts.unsqueeze(-1) * w.T.view(1, 1, 1, 3, -1)).sum(dim=3)
        std = metric_std.view(1, 1, -1, 1)
        mask = masks.to(pred.dtype).unsqueeze(-1)
        denom = mask.flatten(1, 2).sum(dim=1).clamp_min(1.0)
        mae = ((((pred - targets.unsqueeze(-1)) / std).abs() * mask).flatten(1, 2).sum(dim=1) / denom).mean(dim=0)
        value, index = mae.min(dim=0)
        if float(value.item()) < best_mae:
            best_mae = float(value.item())
            best_weight = w[int(index.item())]
    return [float(item) for item in best_weight.tolist()]


def train_one_epoch(
    model: ResidualFixed3WeightRouter,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    metric_std: torch.Tensor,
    indices: Sequence[int],
    lambda_delta: float,
    grad_clip_norm: float,
) -> float:
    model.train()
    losses = []
    for batch in loader:
        history = batch["history"].to(device=device, dtype=torch.float32)
        targets = batch["targets"].to(device=device, dtype=torch.float32)
        masks = batch["target_masks"].to(device=device, dtype=torch.bool)
        forecasts = fixed3_stack(batch, indices, device)
        out = model(history, forecasts)
        prediction = weighted_forecast(forecasts, out["weights"])
        forecast_loss = scaled_sample_mae(prediction, targets, masks, metric_std).mean()
        delta_loss = out["delta_logits"].square().mean()
        loss = forecast_loss + float(lambda_delta) * delta_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(statistics.mean(losses)) if losses else math.nan


@torch.no_grad()
def evaluate(
    model: ResidualFixed3WeightRouter,
    cache: Mapping[str, Any],
    device: torch.device,
    metric_std: torch.Tensor,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    indices = fixed3_indices(cache)
    loader = DataLoader(CacheWindowDataset(cache), batch_size=batch_size, shuffle=False)
    maes, mses, weights, delta_norms = [], [], [], []
    for batch in loader:
        history = batch["history"].to(device=device, dtype=torch.float32)
        targets = batch["targets"].to(device=device, dtype=torch.float32)
        masks = batch["target_masks"].to(device=device, dtype=torch.bool)
        forecasts = fixed3_stack(batch, indices, device)
        out = model(history, forecasts)
        prediction = weighted_forecast(forecasts, out["weights"])
        maes.append(scaled_sample_mae(prediction, targets, masks, metric_std).cpu())
        mses.append(scaled_sample_mse(prediction, targets, masks, metric_std).cpu())
        weights.append(out["weights"].cpu())
        delta_norms.append(out["delta_logits"].norm(dim=1).cpu())
    weight_tensor = torch.cat(weights, dim=0)
    return {
        "mae": float(torch.cat(maes).mean().item()),
        "mse": float(torch.cat(mses).mean().item()),
        "mean_delta_norm": float(torch.cat(delta_norms).mean().item()),
        "mean_weights": {FIXED3_NAMES[i]: float(weight_tensor[:, i].mean().item()) for i in range(3)},
        "weight_std": {FIXED3_NAMES[i]: float(weight_tensor[:, i].std(unbiased=False).item()) for i in range(3)},
    }


def load_transfer(model: ResidualFixed3WeightRouter, checkpoint_path: Path, mode: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = checkpoint["state_dict"]
    target = model.state_dict()
    if mode == "full":
        copied = {key: value for key, value in source.items() if key in target and key != "global_weights" and tuple(value.shape) == tuple(target[key].shape)}
    elif mode == "encoder":
        prefixes = ("history_encoder", "history_projection", "forecast_encoder", "scalar_encoder")
        copied = {key: value for key, value in source.items() if key.startswith(prefixes) and key in target and tuple(value.shape) == tuple(target[key].shape)}
    else:
        raise ValueError(f"Unsupported transfer mode: {mode}")
    target.update(copied)
    model.load_state_dict(target)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def run(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    train_cache = load_cache(ROOT / args.train_cache, args.train_role if args.train_role else None)
    val_cache = load_cache(ROOT / args.val_cache, args.val_role if args.val_role else None)
    metric_std = load_metric_std(args.normalizer_checkpoint, int(train_cache["num_features"]))
    global_weights = best_global_weights(train_cache, metric_std, args.global_weight_step) if args.global_weights == "train_grid" else [1 / 3, 1 / 3, 1 / 3]
    device = torch.device(args.device)
    model = ResidualFixed3WeightRouter(
        global_weights=global_weights,
        input_len=int(train_cache["input_len"]),
        horizon=int(train_cache["forecast_horizon"]),
        num_features=int(train_cache["num_features"]),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    if args.init_checkpoint:
        load_transfer(model, ROOT / args.init_checkpoint, args.transfer_mode)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loader = DataLoader(CacheWindowDataset(train_cache), batch_size=args.batch_size, shuffle=True)
    indices = fixed3_indices(train_cache)
    best_metrics = None
    best_state = None
    best_epoch = -1
    best_mae = math.inf
    bad_epochs = 0
    curves = []
    for epoch in range(1, args.max_epochs + 1):
        train_loss = train_one_epoch(model, loader, optimizer, device, metric_std, indices, args.lambda_delta, args.grad_clip_norm)
        metrics = evaluate(model, val_cache, device, metric_std, args.batch_size)
        curves.append({"epoch": epoch, "train_loss": train_loss, "validation_mae": metrics["mae"], "validation_mse": metrics["mse"], "mean_delta_norm": metrics["mean_delta_norm"]})
        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            best_metrics = metrics
            best_epoch = epoch
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break
    assert best_metrics is not None and best_state is not None
    result_dir = ROOT / args.results_root / args.run_name
    ckpt_dir = ROOT / args.checkpoint_root / args.run_name
    result_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    write_csv(result_dir / "training_curves.csv", curves)
    torch.save(
        {
            "model": "ResidualFixed3WeightRouter",
            "state_dict": best_state,
            "global_weights": global_weights,
            "seed": args.seed,
            "best_epoch": best_epoch,
            "metrics": best_metrics,
            "args": vars(args),
            "safety": "NO TEST DATA USED",
        },
        ckpt_dir / "best_residual_fixed3_router.pt",
    )
    summary = {
        "run_name": args.run_name,
        "seed": args.seed,
        "pretraining_data": args.pretraining_label,
        "transfer_mode": args.transfer_mode if args.init_checkpoint else "none",
        "init_checkpoint": args.init_checkpoint,
        "best_epoch": best_epoch,
        "global_weights": {FIXED3_NAMES[i]: global_weights[i] for i in range(3)},
        "mae": best_metrics["mae"],
        "mse": best_metrics["mse"],
        "mean_delta_norm": best_metrics["mean_delta_norm"],
        "mean_weights": best_metrics["mean_weights"],
        "weight_std": best_metrics["weight_std"],
        "train_cache": args.train_cache,
        "val_cache": args.val_cache,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "safety": "NO TEST DATA USED",
    }
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--val-cache", required=True)
    parser.add_argument("--train-role", default="")
    parser.add_argument("--val-role", default="")
    parser.add_argument("--normalizer-checkpoint", default="")
    parser.add_argument("--pretraining-label", default="")
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--transfer-mode", choices=TRANSFER_MODES, default="none")
    parser.add_argument("--checkpoint-root", default="checkpoints/costarts_transfer/fixed3_residual")
    parser.add_argument("--results-root", default="results/router_summary/costarts_transfer/fixed3_residual")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--lambda-delta", type=float, default=0.001)
    parser.add_argument("--global-weights", choices=("train_grid", "equal"), default="train_grid")
    parser.add_argument("--global-weight-step", type=float, default=0.01)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
