"""Train an adaptive weighting baseline for the fixed ETTh1 COSTAR best-3 ensemble.

The forecasting experts stay frozen.  The model always uses PatchTST,
iTransformer, and TimesNet, and learns nonnegative simplex weights from history
plus forecast-state features.
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
from scripts.train_costarts_subset_menu_router import sample_mae, sample_mse
from scripts.train_sequential_costarts_full_walkforward import CacheWindowDataset


FIXED3_NAMES = ("PatchTST", "iTransformer", "TimesNet")
FIXED3_INDICES = tuple(EXPERT_ORDER.index(name) for name in FIXED3_NAMES)
ABLATIONS = ("history_only", "history_forecasts", "history_disagreement", "full")


class Fixed3DynamicWeightRouter(nn.Module):
    def __init__(
        self,
        input_len: int = 96,
        horizon: int = 12,
        num_features: int = 7,
        num_experts: int = 3,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.input_len = int(input_len)
        self.horizon = int(horizon)
        self.num_features = int(num_features)
        self.num_experts = int(num_experts)
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
        scalar_dim = 12
        self.scalar_encoder = nn.Sequential(
            nn.Linear(scalar_dim, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
        )
        self.head = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, history: torch.Tensor, fixed3_forecasts: torch.Tensor, ablation: str = "full") -> dict[str, torch.Tensor]:
        history_rep = self.history_encoder(history.transpose(1, 2)).squeeze(-1)
        history_rep = self.history_projection(history_rep)
        mean_forecast = fixed3_forecasts.mean(dim=-1)
        flattened = torch.cat(
            [fixed3_forecasts[..., index].flatten(1) for index in range(self.num_experts)]
            + [mean_forecast.flatten(1)],
            dim=1,
        )
        scalar = forecast_state_scalars(fixed3_forecasts)
        if ablation == "history_only":
            flattened = torch.zeros_like(flattened)
            scalar = torch.zeros_like(scalar)
        elif ablation == "history_forecasts":
            scalar = torch.zeros_like(scalar)
        elif ablation == "history_disagreement":
            flattened = torch.zeros_like(flattened)
        elif ablation != "full":
            raise ValueError(f"Unknown ablation: {ablation}")
        forecast_rep = self.forecast_encoder(flattened)
        scalar_rep = self.scalar_encoder(scalar)
        logits = self.head(torch.cat((history_rep, forecast_rep, scalar_rep), dim=1))
        weights = torch.softmax(logits, dim=1)
        return {"logits": logits, "weights": weights}


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


def forecast_state_scalars(fixed3_forecasts: torch.Tensor) -> torch.Tensor:
    a = fixed3_forecasts[..., 0]
    b = fixed3_forecasts[..., 1]
    c = fixed3_forecasts[..., 2]
    diffs = (a - b, a - c, b - c)
    abs_diffs = [item.abs() for item in diffs]
    variance = fixed3_forecasts.var(dim=-1, unbiased=False)
    spread = fixed3_forecasts.max(dim=-1).values - fixed3_forecasts.min(dim=-1).values
    features = []
    for tensor in list(diffs) + abs_diffs + [variance, spread]:
        features.append(tensor.mean(dim=(1, 2)))
        features.append(tensor.amax(dim=(1, 2)) if tensor.min() >= 0 else tensor.abs().amax(dim=(1, 2)))
    return torch.stack(features[:12], dim=1)


def weighted_forecast(fixed3_forecasts: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (fixed3_forecasts * weights[:, None, None, :]).sum(dim=-1)


def fixed3_stack(batch: Mapping[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    stack = batch["prediction_stack"].to(device=device, dtype=torch.float32)
    return stack[..., list(FIXED3_INDICES)]


def train_one_epoch(
    model: Fixed3DynamicWeightRouter,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    normalizer_std: torch.Tensor,
    grad_clip_norm: float,
    entropy_weight: float,
    ablation: str,
) -> float:
    model.train()
    losses = []
    for batch in loader:
        history = batch["history"].to(device=device, dtype=torch.float32)
        targets = batch["targets"].to(device=device, dtype=torch.float32)
        masks = batch["target_masks"].to(device=device, dtype=torch.bool)
        forecasts = fixed3_stack(batch, device)
        outputs = model(history, forecasts, ablation)
        prediction = weighted_forecast(forecasts, outputs["weights"])
        mae = sample_mae(prediction, targets, masks, normalizer_std.to(device)).mean()
        entropy = -(outputs["weights"] * outputs["weights"].clamp_min(1e-8).log()).sum(dim=1).mean()
        loss = mae - float(entropy_weight) * entropy
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(statistics.mean(losses)) if losses else math.nan


@torch.no_grad()
def evaluate(
    model: Fixed3DynamicWeightRouter,
    cache: Mapping[str, Any],
    device: torch.device,
    normalizer_std: torch.Tensor,
    batch_size: int,
    ablation: str,
) -> dict[str, Any]:
    model.eval()
    loader = DataLoader(CacheWindowDataset(cache), batch_size=batch_size, shuffle=False)
    maes = []
    mses = []
    equal_maes = []
    equal_mses = []
    weights = []
    per_window = []
    offset = 0
    for batch in loader:
        history = batch["history"].to(device=device, dtype=torch.float32)
        targets = batch["targets"].to(device=device, dtype=torch.float32)
        masks = batch["target_masks"].to(device=device, dtype=torch.bool)
        forecasts = fixed3_stack(batch, device)
        outputs = model(history, forecasts, ablation)
        prediction = weighted_forecast(forecasts, outputs["weights"])
        equal_prediction = forecasts.mean(dim=-1)
        batch_mae = sample_mae(prediction, targets, masks, normalizer_std.to(device)).detach().cpu()
        batch_mse = sample_mse(prediction, targets, masks, normalizer_std.to(device)).detach().cpu()
        batch_equal_mae = sample_mae(equal_prediction, targets, masks, normalizer_std.to(device)).detach().cpu()
        batch_equal_mse = sample_mse(equal_prediction, targets, masks, normalizer_std.to(device)).detach().cpu()
        maes.append(batch_mae)
        mses.append(batch_mse)
        equal_maes.append(batch_equal_mae)
        equal_mses.append(batch_equal_mse)
        weights.append(outputs["weights"].detach().cpu())
        for row in range(history.shape[0]):
            row_weights = outputs["weights"][row].detach().cpu().tolist()
            per_window.append(
                {
                    "cache_index": offset + row,
                    "absolute_window_start": int(batch["absolute_window_start"][row].item()),
                    "mae": float(batch_mae[row].item()),
                    "mse": float(batch_mse[row].item()),
                    "equal_fixed3_mae": float(batch_equal_mae[row].item()),
                    "w_patchtst": row_weights[0],
                    "w_itransformer": row_weights[1],
                    "w_timesnet": row_weights[2],
                }
            )
        offset += history.shape[0]
    weight_tensor = torch.cat(weights)
    return {
        "mae": float(torch.cat(maes).mean().item()),
        "mse": float(torch.cat(mses).mean().item()),
        "equal_fixed3_mae": float(torch.cat(equal_maes).mean().item()),
        "equal_fixed3_mse": float(torch.cat(equal_mses).mean().item()),
        "mean_weights": {FIXED3_NAMES[i]: float(weight_tensor[:, i].mean().item()) for i in range(3)},
        "weight_std": {FIXED3_NAMES[i]: float(weight_tensor[:, i].std(unbiased=False).item()) for i in range(3)},
        "weight_p10": {FIXED3_NAMES[i]: float(torch.quantile(weight_tensor[:, i], 0.1).item()) for i in range(3)},
        "weight_p50": {FIXED3_NAMES[i]: float(torch.quantile(weight_tensor[:, i], 0.5).item()) for i in range(3)},
        "weight_p90": {FIXED3_NAMES[i]: float(torch.quantile(weight_tensor[:, i], 0.9).item()) for i in range(3)},
        "per_window": per_window,
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def aggregate(values: Sequence[float]) -> tuple[float, float]:
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.pstdev(values))


def run_seed(seed: int, train_cache: Mapping[str, Any], val_cache: Mapping[str, Any], args: argparse.Namespace, normalizer_std: torch.Tensor) -> dict[str, Any]:
    set_seed(seed)
    device = torch.device(args.device)
    model = Fixed3DynamicWeightRouter(
        input_len=int(train_cache["input_len"]),
        horizon=int(train_cache["forecast_horizon"]),
        num_features=int(train_cache["num_features"]),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loader = DataLoader(CacheWindowDataset(train_cache), batch_size=args.batch_size, shuffle=True)
    best_metrics = None
    best_state = None
    best_mae = math.inf
    best_epoch = -1
    bad_epochs = 0
    curves = []
    for epoch in range(1, args.max_epochs + 1):
        train_loss = train_one_epoch(model, loader, optimizer, device, normalizer_std, args.grad_clip_norm, args.entropy_weight, args.ablation)
        metrics = evaluate(model, val_cache, device, normalizer_std, args.batch_size, args.ablation)
        curves.append({"epoch": epoch, "train_loss": train_loss, "validation_mae": metrics["mae"], "validation_mse": metrics["mse"]})
        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            best_metrics = metrics
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break
    assert best_metrics is not None and best_state is not None
    result_dir = ROOT / args.results_root / args.ablation / f"seed_{seed}"
    ckpt_dir = ROOT / args.checkpoint_root / args.ablation / f"seed_{seed}"
    result_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    write_csv(result_dir / "training_curves.csv", curves)
    write_csv(result_dir / "validation_per_window.csv", best_metrics["per_window"])
    torch.save(
        {
            "router_type": "Fixed3DynamicWeightRouter",
            "router_config": {
                "input_len": int(train_cache["input_len"]),
                "horizon": int(train_cache["forecast_horizon"]),
                "num_features": int(train_cache["num_features"]),
                "embedding_dim": args.embedding_dim,
                "hidden_dim": args.hidden_dim,
            },
            "ablation": args.ablation,
            "state_dict": best_state,
            "seed": seed,
            "best_epoch": best_epoch,
            "validation_metrics": {key: value for key, value in best_metrics.items() if key != "per_window"},
            "safety": "NO TEST DATA USED",
        },
        ckpt_dir / "best_fixed3_dynamic_weight_router.pt",
    )
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "mae": best_metrics["mae"],
        "mse": best_metrics["mse"],
        "equal_fixed3_mae": best_metrics["equal_fixed3_mae"],
        "equal_fixed3_mse": best_metrics["equal_fixed3_mse"],
        "improvement_vs_fixed3": best_metrics["equal_fixed3_mae"] - best_metrics["mae"],
        "mean_weight_patchtst": best_metrics["mean_weights"]["PatchTST"],
        "mean_weight_itransformer": best_metrics["mean_weights"]["iTransformer"],
        "mean_weight_timesnet": best_metrics["mean_weights"]["TimesNet"],
        "weight_std_patchtst": best_metrics["weight_std"]["PatchTST"],
        "weight_std_itransformer": best_metrics["weight_std"]["iTransformer"],
        "weight_std_timesnet": best_metrics["weight_std"]["TimesNet"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", default="cache/costarts_walkforward/router_train_20_60_cache.pt")
    parser.add_argument("--val-cache", default="cache/costarts_walkforward/router_val_60_80_cache.pt")
    parser.add_argument("--normalizer-checkpoint", default="checkpoints/costarts_walkforward/final_60/DLinear/best_expert.pt")
    parser.add_argument("--checkpoint-root", default="checkpoints/costarts_walkforward/fixed3_dynamic_weighting")
    parser.add_argument("--results-root", default="results/router_summary/costarts_walkforward/fixed3_dynamic_weighting")
    parser.add_argument("--ablation", choices=ABLATIONS, default="full")
    parser.add_argument("--seeds", default="7")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--entropy-weight", type=float, default=0.0)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    train_cache_path = ROOT / args.train_cache
    val_cache_path = ROOT / args.val_cache
    train_cache = load_verified_cache(train_cache_path, "router_train_20_60")
    val_cache = load_verified_cache(val_cache_path, "router_val_60_80")
    normalizer_std = load_normalizer_std(ROOT / args.normalizer_checkpoint)
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    rows = [run_seed(seed, train_cache, val_cache, args, normalizer_std) for seed in seeds]
    result_root = ROOT / args.results_root / args.ablation
    result_root.mkdir(parents=True, exist_ok=True)
    write_csv(result_root / "per_seed_results.csv", rows)
    mae_mean, mae_std = aggregate([row["mae"] for row in rows])
    mse_mean, mse_std = aggregate([row["mse"] for row in rows])
    summary = {
        "method": "fixed3_dynamic_weighting",
        "fixed3_experts": list(FIXED3_NAMES),
        "seeds": seeds,
        "mae_mean": mae_mean,
        "mae_std": mae_std,
        "mse_mean": mse_mean,
        "mse_std": mse_std,
        "equal_fixed3_mae": rows[0]["equal_fixed3_mae"],
        "equal_fixed3_mse": rows[0]["equal_fixed3_mse"],
        "improvement_vs_fixed3_mean": rows[0]["equal_fixed3_mae"] - mae_mean,
        "per_seed": rows,
        "train_cache": args.train_cache,
        "val_cache": args.val_cache,
        "train_cache_sha256": sha256_file(train_cache_path),
        "val_cache_sha256": sha256_file(val_cache_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "safety": "NO TEST DATA USED",
    }
    (result_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
