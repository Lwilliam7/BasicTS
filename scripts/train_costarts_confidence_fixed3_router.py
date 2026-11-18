"""Confidence-aware fixed-3 weighting router for ETTh1 COSTAR validation.

The fallback forecast is always the strong global fixed-3 weighted ensemble:
PatchTST/iTransformer/TimesNet with weights (0.36, 0.42, 0.22).  The router
learns a small candidate correction plus an uncertainty-aware utility estimate;
validation screens confidence gates without retraining.
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

from scripts.build_costarts_walkforward_cache import EXPERT_ORDER, sha256_file, validate_walkforward_cache
from scripts.train_sequential_costarts_full_walkforward import CacheWindowDataset


FIXED3_NAMES = ("PatchTST", "iTransformer", "TimesNet")
BASE_WEIGHTS = (0.36, 0.42, 0.22)


class ConfidenceFixed3Router(nn.Module):
    def __init__(
        self,
        input_len: int = 96,
        horizon: int = 12,
        num_features: int = 7,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        delta_scale: float = 0.15,
    ) -> None:
        super().__init__()
        self.input_len = int(input_len)
        self.horizon = int(horizon)
        self.num_features = int(num_features)
        self.delta_scale = float(delta_scale)
        self.register_buffer("base_weights", torch.tensor(BASE_WEIGHTS, dtype=torch.float32))
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
        self.shared = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.delta_head = nn.Linear(hidden_dim, 3)
        self.mu_head = nn.Linear(hidden_dim, 1)
        self.log_var_head = nn.Linear(hidden_dim, 1)

    def forward(self, history: torch.Tensor, forecasts: torch.Tensor) -> dict[str, torch.Tensor]:
        history_rep = self.history_encoder(history.transpose(1, 2)).squeeze(-1)
        history_rep = self.history_projection(history_rep)
        base_forecast = weighted_forecast(forecasts, self.base_weights.to(forecasts.device, forecasts.dtype)[None, :].expand(history.shape[0], -1))
        flat = torch.cat([forecasts[..., i].flatten(1) for i in range(3)] + [base_forecast.flatten(1)], dim=1)
        forecast_rep = self.forecast_encoder(flat)
        scalar_rep = self.scalar_encoder(forecast_state_scalars(forecasts, base_forecast))
        rep = self.shared(torch.cat((history_rep, forecast_rep, scalar_rep), dim=1))
        delta = self.delta_scale * torch.tanh(self.delta_head(rep))
        candidate_raw = (self.base_weights.to(delta.device, delta.dtype)[None, :] + delta).clamp_min(1e-5)
        weights = candidate_raw / candidate_raw.sum(dim=1, keepdim=True)
        log_var = self.log_var_head(rep).squeeze(1).clamp(-10.0, 4.0)
        return {
            "delta": delta,
            "weights": weights,
            "mu": self.mu_head(rep).squeeze(1),
            "log_var": log_var,
            "sigma": torch.exp(0.5 * log_var),
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_verified_cache(path: Path, expected_role: str) -> dict[str, Any]:
    if "test" in str(path).lower():
        raise ValueError(f"Refusing test cache path: {path}")
    cache = torch.load(path, map_location="cpu", weights_only=False)
    validate_walkforward_cache(cache)
    if cache["cache_role"] != expected_role:
        raise ValueError(f"{path} cache_role={cache['cache_role']!r}, expected {expected_role!r}")
    return cache


def load_normalizer_std(path: Path) -> torch.Tensor:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "scaler_std" not in checkpoint:
        raise KeyError(f"{path} does not contain scaler_std")
    return checkpoint["scaler_std"].to(torch.float32)


def fixed3_indices(cache: Mapping[str, Any]) -> list[int]:
    names = list(cache["expert_names"])
    return [names.index(name) for name in FIXED3_NAMES]


def fixed3_stack(batch: Mapping[str, torch.Tensor], indices: Sequence[int], device: torch.device) -> torch.Tensor:
    return batch["prediction_stack"].to(device=device, dtype=torch.float32)[..., list(indices)]


def weighted_forecast(forecasts: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (forecasts * weights[:, None, None, :]).sum(dim=-1)


def forecast_state_scalars(forecasts: torch.Tensor, base_forecast: torch.Tensor) -> torch.Tensor:
    a, b, c = forecasts[..., 0], forecasts[..., 1], forecasts[..., 2]
    tensors = [
        a - b,
        a - c,
        b - c,
        (a - b).abs(),
        (a - c).abs(),
        (b - c).abs(),
        forecasts.var(dim=-1, unbiased=False),
        forecasts.max(dim=-1).values - forecasts.min(dim=-1).values,
        forecasts.mean(dim=-1) - base_forecast,
    ]
    features = []
    for tensor in tensors:
        features.append(tensor.mean(dim=(1, 2)))
        features.append(tensor.abs().amax(dim=(1, 2)))
    return torch.stack(features[:12], dim=1)


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


def heteroscedastic_loss(mu: torch.Tensor, log_var: torch.Tensor, utility: torch.Tensor) -> torch.Tensor:
    log_var = log_var.clamp(-10.0, 4.0)
    return (0.5 * torch.exp(-log_var) * (utility - mu).square() + 0.5 * log_var).mean()


def train_one_epoch(
    model: ConfidenceFixed3Router,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    indices: Sequence[int],
    normalizer_std: torch.Tensor,
    args: argparse.Namespace,
) -> float:
    model.train()
    losses = []
    base = torch.tensor(BASE_WEIGHTS, dtype=torch.float32, device=device)
    for batch in loader:
        history = batch["history"].to(device=device, dtype=torch.float32)
        targets = batch["targets"].to(device=device, dtype=torch.float32)
        masks = batch["target_masks"].to(device=device, dtype=torch.bool)
        forecasts = fixed3_stack(batch, indices, device)
        out = model(history, forecasts)
        base_prediction = weighted_forecast(forecasts, base[None, :].expand(history.shape[0], -1))
        candidate_prediction = weighted_forecast(forecasts, out["weights"])
        base_mae = scaled_sample_mae(base_prediction, targets, masks, normalizer_std)
        candidate_mae = scaled_sample_mae(candidate_prediction, targets, masks, normalizer_std)
        utility = (base_mae - candidate_mae).detach()
        forecast_loss = candidate_mae.mean()
        utility_loss = heteroscedastic_loss(out["mu"], out["log_var"], utility)
        delta_loss = out["delta"].square().mean()
        loss = forecast_loss + args.alpha_utility * utility_loss + args.lambda_delta * delta_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(statistics.mean(losses)) if losses else math.nan


@torch.no_grad()
def collect_predictions(
    model: ConfidenceFixed3Router,
    cache: Mapping[str, Any],
    device: torch.device,
    normalizer_std: torch.Tensor,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    model.eval()
    indices = fixed3_indices(cache)
    loader = DataLoader(CacheWindowDataset(cache), batch_size=batch_size, shuffle=False)
    base = torch.tensor(BASE_WEIGHTS, dtype=torch.float32, device=device)
    chunks: dict[str, list[torch.Tensor]] = {key: [] for key in ("base_mae", "base_mse", "cand_mae", "cand_mse", "mu", "sigma", "utility", "weights")}
    for batch in loader:
        history = batch["history"].to(device=device, dtype=torch.float32)
        targets = batch["targets"].to(device=device, dtype=torch.float32)
        masks = batch["target_masks"].to(device=device, dtype=torch.bool)
        forecasts = fixed3_stack(batch, indices, device)
        out = model(history, forecasts)
        base_prediction = weighted_forecast(forecasts, base[None, :].expand(history.shape[0], -1))
        candidate_prediction = weighted_forecast(forecasts, out["weights"])
        base_mae = scaled_sample_mae(base_prediction, targets, masks, normalizer_std)
        cand_mae = scaled_sample_mae(candidate_prediction, targets, masks, normalizer_std)
        chunks["base_mae"].append(base_mae.cpu())
        chunks["base_mse"].append(scaled_sample_mse(base_prediction, targets, masks, normalizer_std).cpu())
        chunks["cand_mae"].append(cand_mae.cpu())
        chunks["cand_mse"].append(scaled_sample_mse(candidate_prediction, targets, masks, normalizer_std).cpu())
        chunks["utility"].append((base_mae - cand_mae).cpu())
        chunks["mu"].append(out["mu"].cpu())
        chunks["sigma"].append(out["sigma"].cpu())
        chunks["weights"].append(out["weights"].cpu())
    return {key: torch.cat(value, dim=0) for key, value in chunks.items()}


def gate_metrics(data: Mapping[str, torch.Tensor], k: float, threshold: float) -> dict[str, Any]:
    lcb = data["mu"] - float(k) * data["sigma"]
    adapt = lcb > float(threshold)
    final_mae = torch.where(adapt, data["cand_mae"], data["base_mae"])
    final_mse = torch.where(adapt, data["cand_mse"], data["base_mse"])
    utility = data["utility"]
    bad_adapt = adapt & (utility < 0)
    return {
        "k": float(k),
        "threshold": float(threshold),
        "mae": float(final_mae.mean().item()),
        "mse": float(final_mse.mean().item()),
        "adapted_percent": float(adapt.to(torch.float32).mean().item() * 100.0),
        "fallback_percent": float((~adapt).to(torch.float32).mean().item() * 100.0),
        "mean_mu": float(data["mu"].mean().item()),
        "mean_sigma": float(data["sigma"].mean().item()),
        "mean_utility": float(utility.mean().item()),
        "actual_avg_improvement_adapted": float(utility[adapt].mean().item()) if bool(adapt.any()) else math.nan,
        "actual_avg_degradation_wrong_adapt": float((-utility[bad_adapt]).mean().item()) if bool(bad_adapt.any()) else 0.0,
        "wrong_adapt_percent": float(bad_adapt.to(torch.float32).mean().item() * 100.0),
    }


def confidence_buckets(data: Mapping[str, torch.Tensor], k: float, threshold: float) -> list[dict[str, Any]]:
    lcb = data["mu"] - float(k) * data["sigma"]
    order = torch.argsort(lcb)
    rows = []
    n = int(lcb.numel())
    for bucket in range(5):
        start = bucket * n // 5
        end = (bucket + 1) * n // 5
        idx = order[start:end]
        utility = data["utility"][idx]
        predicted = data["mu"][idx]
        rows.append(
            {
                "bucket": bucket + 1,
                "lcb_min": float(lcb[idx].min().item()),
                "lcb_max": float(lcb[idx].max().item()),
                "predicted_utility_mean": float(predicted.mean().item()),
                "actual_utility_mean": float(utility.mean().item()),
                "calibration_error": float((predicted.mean() - utility.mean()).item()),
                "adaptation_helps_percent": float((utility > 0).to(torch.float32).mean().item() * 100.0),
                "average_true_improvement": float(utility.mean().item()),
                "would_adapt_percent": float((lcb[idx] > float(threshold)).to(torch.float32).mean().item() * 100.0),
            }
        )
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def run_seed(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    set_seed(seed)
    train_cache_path = ROOT / args.train_cache
    val_cache_path = ROOT / args.val_cache
    train_cache = load_verified_cache(train_cache_path, "router_train_20_60")
    val_cache = load_verified_cache(val_cache_path, "router_val_60_80")
    normalizer_std = load_normalizer_std(ROOT / args.normalizer_checkpoint)
    device = torch.device(args.device)
    model = ConfidenceFixed3Router(
        input_len=int(train_cache["input_len"]),
        horizon=int(train_cache["forecast_horizon"]),
        num_features=int(train_cache["num_features"]),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        delta_scale=args.delta_scale,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loader = DataLoader(CacheWindowDataset(train_cache), batch_size=args.batch_size, shuffle=True)
    indices = fixed3_indices(train_cache)
    best_state = None
    best_epoch = -1
    best_mae = math.inf
    bad_epochs = 0
    curves = []
    for epoch in range(1, args.max_epochs + 1):
        train_loss = train_one_epoch(model, loader, optimizer, device, indices, normalizer_std, args)
        data = collect_predictions(model, val_cache, device, normalizer_std, args.batch_size)
        always_candidate_mae = float(data["cand_mae"].mean().item())
        curves.append({"epoch": epoch, "train_loss": train_loss, "always_candidate_mae": always_candidate_mae, "base_mae": float(data["base_mae"].mean().item())})
        if always_candidate_mae < best_mae:
            best_mae = always_candidate_mae
            best_epoch = epoch
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    data = collect_predictions(model, val_cache, device, normalizer_std, args.batch_size)
    k_values = [float(item.strip()) for item in args.k_values.split(",") if item.strip()]
    thresholds = [float(item.strip()) for item in args.thresholds.split(",") if item.strip()]
    gate_rows = [gate_metrics(data, k, threshold) for k in k_values for threshold in thresholds]
    gate_rows.append({**gate_metrics(data, 0.0, -math.inf), "k": "always_candidate", "threshold": ""})
    gate_rows.append({"k": "always_base", "threshold": "", "mae": float(data["base_mae"].mean().item()), "mse": float(data["base_mse"].mean().item()), "adapted_percent": 0.0, "fallback_percent": 100.0, "mean_mu": float(data["mu"].mean().item()), "mean_sigma": float(data["sigma"].mean().item()), "mean_utility": float(data["utility"].mean().item())})
    best_gate = min((row for row in gate_rows if isinstance(row["mae"], float)), key=lambda row: float(row["mae"]))
    out_root = ROOT / args.results_root / f"seed_{seed}"
    ckpt_root = ROOT / args.checkpoint_root / f"seed_{seed}"
    out_root.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    write_csv(out_root / "training_curves.csv", curves)
    write_csv(out_root / "gate_sweep.csv", gate_rows)
    buckets = confidence_buckets(data, float(best_gate["k"]) if isinstance(best_gate["k"], float) else 0.0, float(best_gate["threshold"]) if isinstance(best_gate["threshold"], float) else 0.0)
    write_csv(out_root / "confidence_buckets.csv", buckets)
    torch.save(
        {
            "model": "ConfidenceFixed3Router",
            "state_dict": best_state,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_gate": best_gate,
            "args": vars(args),
            "safety": "NO TEST DATA USED",
        },
        ckpt_root / "best_confidence_fixed3_router.pt",
    )
    summary = {
        "seed": seed,
        "best_epoch": best_epoch,
        "base_weights": dict(zip(FIXED3_NAMES, BASE_WEIGHTS)),
        "base_mae": float(data["base_mae"].mean().item()),
        "base_mse": float(data["base_mse"].mean().item()),
        "always_candidate_mae": float(data["cand_mae"].mean().item()),
        "always_candidate_mse": float(data["cand_mse"].mean().item()),
        "mean_candidate_utility": float(data["utility"].mean().item()),
        "mean_mu": float(data["mu"].mean().item()),
        "mean_sigma": float(data["sigma"].mean().item()),
        "best_gate": best_gate,
        "best_gate_diff_vs_0p36610": 0.36610 - float(best_gate["mae"]),
        "confidence_buckets": buckets,
        "train_cache_sha256": sha256_file(train_cache_path),
        "val_cache_sha256": sha256_file(val_cache_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "safety": "NO TEST DATA USED",
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def aggregate(values: Sequence[float]) -> tuple[float, float]:
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.pstdev(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", default="cache/costarts_walkforward/router_train_20_60_cache.pt")
    parser.add_argument("--val-cache", default="cache/costarts_walkforward/router_val_60_80_cache.pt")
    parser.add_argument("--normalizer-checkpoint", default="checkpoints/costarts_walkforward/final_60/DLinear/best_expert.pt")
    parser.add_argument("--checkpoint-root", default="checkpoints/costarts_walkforward/confidence_fixed3")
    parser.add_argument("--results-root", default="results/router_summary/costarts_walkforward/confidence_fixed3")
    parser.add_argument("--seeds", default="7")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--alpha-utility", type=float, default=0.1)
    parser.add_argument("--lambda-delta", type=float, default=0.01)
    parser.add_argument("--delta-scale", type=float, default=0.15)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--k-values", default="0,0.5,1.0,1.5,2.0")
    parser.add_argument("--thresholds", default="0,0.001,0.002,0.005")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    summaries = [run_seed(args, seed) for seed in seeds]
    rows = []
    for item in summaries:
        row = {"seed": item["seed"], **item["best_gate"]}
        rows.append(row)
    write_csv(ROOT / args.results_root / "per_seed_best_gate.csv", rows)
    maes = [float(row["mae"]) for row in rows]
    mses = [float(row["mse"]) for row in rows]
    summary = {
        "method": "confidence_aware_fixed3_router",
        "seeds": seeds,
        "mae_mean": aggregate(maes)[0],
        "mae_std": aggregate(maes)[1],
        "mse_mean": aggregate(mses)[0],
        "mse_std": aggregate(mses)[1],
        "per_seed_best_gate": rows,
        "safety": "NO TEST DATA USED",
    }
    (ROOT / args.results_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
