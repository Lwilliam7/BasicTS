"""Train Full End-to-End Sequential COSTAR-TS on ETTh1 validation only."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.sequential_costarts_end_to_end import (
    EXPERT_ORDER,
    EndToEndCOSTARTSConfig,
    FullEndToEndCOSTARTS,
    end_to_end_costarts_loss,
    gradient_report,
    masked_sample_mae,
    masked_sample_mse,
)


@dataclass
class TrainingConfig:
    dataset: str = "ETTh1"
    data_dir: str = "datasets/ETTh1"
    checkpoint_root: str = "checkpoints/costarts_end_to_end"
    results_root: str = "results/router_summary/costarts_end_to_end"
    seeds: tuple[int, ...] = (7,)
    input_len: int = 96
    forecast_horizon: int = 12
    num_features: int = 7
    train_end: int = 8640
    val_end: int = 11520
    batch_size: int = 128
    max_epochs: int = 5
    patience: int = 3
    router_lr: float = 1e-3
    expert_lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    expert_hidden_size: int = 64
    router_embedding_dim: int = 64
    router_hidden_dim: int = 64
    max_queries: int = 5
    temp_start: float = 2.0
    temp_end: float = 0.5
    alpha_expert: float = 0.5
    lambda_query: float = 0.01
    lambda_balance: float = 0.05
    lambda_stop: float = 0.1
    query_cost: float = 0.0
    stop_threshold: float = 0.5
    device: str = "cpu"


class ETThWindowDataset(Dataset):
    def __init__(
        self,
        full_data: np.ndarray,
        start: int,
        end: int,
        input_len: int,
        horizon: int,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> None:
        usable_end = int(end) - int(input_len) - int(horizon) + 1
        self.starts = torch.arange(int(start), usable_end, dtype=torch.long)
        self.full = torch.tensor(full_data, dtype=torch.float32)
        self.input_len = int(input_len)
        self.horizon = int(horizon)
        self.mean = mean.to(torch.float32)
        self.std = std.to(torch.float32)

    def __len__(self) -> int:
        return int(self.starts.numel())

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start = int(self.starts[index].item())
        history = self.full[start : start + self.input_len]
        target = self.full[start + self.input_len : start + self.input_len + self.horizon]
        history = (history - self.mean) / self.std
        target = (target - self.mean) / self.std
        mask = torch.isfinite(target)
        return {
            "history": torch.nan_to_num(history),
            "target": torch.nan_to_num(target),
            "mask": mask,
            "absolute_window_start": torch.tensor(start, dtype=torch.long),
        }


def load_train_val_data(data_dir: Path) -> np.ndarray:
    arrays = []
    for name in ("train_data.npy", "val_data.npy"):
        path = data_dir / name
        if not path.exists():
            raise FileNotFoundError(path)
        arrays.append(np.load(path))
    full = np.concatenate(arrays, axis=0)
    if full.shape != (11520, 7):
        raise ValueError(f"Expected ETTh1 train+val shape (11520, 7), got {full.shape}")
    return full.astype(np.float32)


def load_full_data(data_dir: Path) -> np.ndarray:
    arrays = []
    for name in ("train_data.npy", "val_data.npy", "test_data.npy"):
        path = data_dir / name
        if not path.exists():
            raise FileNotFoundError(path)
        arrays.append(np.load(path))
    full = np.concatenate(arrays, axis=0)
    if full.shape != (14400, 7):
        raise ValueError(f"Expected ETTh1 full shape (14400, 7), got {full.shape}")
    return full.astype(np.float32)


def fit_scaler(full_data: np.ndarray, train_end: int) -> tuple[torch.Tensor, torch.Tensor]:
    train = torch.tensor(full_data[:train_end], dtype=torch.float32)
    return train.mean(dim=0), train.std(dim=0).clamp_min(1e-6)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def temperature_for_epoch(epoch: int, max_epochs: int, start: float, end: float) -> float:
    if max_epochs <= 1:
        return float(end)
    ratio = float(epoch - 1) / float(max_epochs - 1)
    return float(start + ratio * (end - start))


def make_model_config(args: argparse.Namespace) -> EndToEndCOSTARTSConfig:
    return EndToEndCOSTARTSConfig(
        input_len=args.input_len,
        forecast_horizon=args.forecast_horizon,
        num_features=args.num_features,
        expert_hidden_size=args.expert_hidden_size,
        router_embedding_dim=args.router_embedding_dim,
        router_hidden_dim=args.router_hidden_dim,
        max_queries=args.max_queries,
        route_temperature=args.temp_start,
        gumbel_routing=args.gumbel_routing,
        straight_through=args.straight_through,
    )


def make_optimizer(model: FullEndToEndCOSTARTS, args: argparse.Namespace) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        [
            {"params": model.router_parameters(), "lr": args.router_lr, "name": "router"},
            {"params": model.expert_parameters(), "lr": args.expert_lr, "name": "experts"},
        ],
        weight_decay=args.weight_decay,
    )


def train_one_epoch(
    model: FullEndToEndCOSTARTS,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    temperature: float,
) -> dict[str, float]:
    model.train()
    sums: dict[str, float] = {}
    count = 0
    for batch in loader:
        history = batch["history"].to(device)
        target = batch["target"].to(device)
        mask = batch["mask"].to(device)
        outputs = model.forward_soft(history, temperature=temperature)
        losses = end_to_end_costarts_loss(
            outputs,
            target,
            mask,
            alpha_expert=args.alpha_expert,
            lambda_query=args.lambda_query,
            lambda_balance=args.lambda_balance,
            lambda_stop=args.lambda_stop,
            query_cost=args.query_cost,
        )
        optimizer.zero_grad()
        losses["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        optimizer.step()
        count += 1
        for key, value in losses.items():
            sums[key] = sums.get(key, 0.0) + float(value.detach().cpu().item())
    return {key: value / max(count, 1) for key, value in sums.items()}


@torch.no_grad()
def evaluate(
    model: FullEndToEndCOSTARTS,
    loader: DataLoader,
    device: torch.device,
    stop_threshold: float,
) -> dict[str, Any]:
    model.eval()
    maes = []
    mses = []
    query_counts = []
    usage = torch.zeros(len(EXPERT_ORDER), dtype=torch.float64)
    stop_counts = torch.zeros(len(EXPERT_ORDER), dtype=torch.float64)
    expert_mae_sum = torch.zeros(len(EXPERT_ORDER), dtype=torch.float64)
    expert_mse_sum = torch.zeros(len(EXPERT_ORDER), dtype=torch.float64)
    entropy_sum = 0.0
    n_samples = 0
    for batch in loader:
        history = batch["history"].to(device)
        target = batch["target"].to(device)
        mask = batch["mask"].to(device)
        hard = model.forward_hard(history, stop_threshold=stop_threshold)
        batch_mae = masked_sample_mae(hard["forecast"], target, mask).cpu()
        batch_mse = masked_sample_mse(hard["forecast"], target, mask).cpu()
        maes.append(batch_mae)
        mses.append(batch_mse)
        query_counts.append(hard["query_counts"].cpu())
        usage += hard["expert_usage"].cpu().sum(dim=0).to(torch.float64)
        for count in hard["query_counts"].cpu().tolist():
            if count > 0:
                stop_counts[int(count) - 1] += 1

        soft = model.forward_soft(history, temperature=model.config.route_temperature)
        route_probs = soft["route_probs"]
        entropy_sum += float((-(route_probs * (route_probs + 1e-8).log()).sum(dim=-1)).mean().cpu().item()) * history.shape[0]
        expert_forecasts = soft["expert_forecasts"]
        for idx in range(len(EXPERT_ORDER)):
            expert_mae_sum[idx] += float(masked_sample_mae(expert_forecasts[..., idx], target, mask).sum().cpu().item())
            expert_mse_sum[idx] += float(masked_sample_mse(expert_forecasts[..., idx], target, mask).sum().cpu().item())
        n_samples += history.shape[0]

    mae = torch.cat(maes)
    mse = torch.cat(mses)
    counts = torch.cat(query_counts).to(torch.float32)
    total = float(max(n_samples, 1))
    return {
        "mae": float(mae.mean().item()),
        "mse": float(mse.mean().item()),
        "average_hard_queries": float(counts.mean().item()),
        "routing_entropy": entropy_sum / total,
        "expert_usage_percent": {EXPERT_ORDER[i]: float(usage[i].item() * 100.0 / total) for i in range(len(EXPERT_ORDER))},
        "stop_distribution_percent": {str(i + 1): float(stop_counts[i].item() * 100.0 / total) for i in range(len(EXPERT_ORDER))},
        "individual_experts": [
            {
                "expert": EXPERT_ORDER[i],
                "mae": float(expert_mae_sum[i].item() / total),
                "mse": float(expert_mse_sum[i].item() / total),
            }
            for i in range(len(EXPERT_ORDER))
        ],
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def save_checkpoint(path: Path, model: FullEndToEndCOSTARTS, epoch: int, args: argparse.Namespace, metrics: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model.config_dict(),
            "epoch": epoch,
            "args": vars(args),
            "validation_metrics": dict(metrics),
            "expert_order": EXPERT_ORDER,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "safety": "NO TEST DATA USED",
        },
        path,
    )


def run_seed(seed: int, args: argparse.Namespace, full_data: np.ndarray, mean: torch.Tensor, std: torch.Tensor) -> dict[str, Any]:
    set_seed(seed)
    device = torch.device(args.device)
    train_ds = ETThWindowDataset(full_data, 0, args.train_end, args.input_len, args.forecast_horizon, mean, std)
    val_ds = ETThWindowDataset(full_data, args.train_end, args.val_end, args.input_len, args.forecast_horizon, mean, std)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    model = FullEndToEndCOSTARTS(make_model_config(args)).to(device)
    optimizer = make_optimizer(model, args)
    seed_root = ROOT / args.results_root / f"seed_{seed}"
    ckpt_path = ROOT / args.checkpoint_root / f"seed_{seed}" / "best_full_end_to_end_costarts.pt"
    rows = []
    best_mae = float("inf")
    best_metrics: dict[str, Any] | None = None
    best_epoch = 0
    bad_epochs = 0
    first_grad_report: dict[str, float] | None = None

    for epoch in range(1, args.max_epochs + 1):
        temperature = temperature_for_epoch(epoch, args.max_epochs, args.temp_start, args.temp_end)
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, args, temperature)
        if first_grad_report is None:
            first_grad_report = gradient_report(model)
        val_metrics = evaluate(model, val_loader, device, args.stop_threshold)
        row = {
            "epoch": epoch,
            "temperature": temperature,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            "validation_mae": val_metrics["mae"],
            "validation_mse": val_metrics["mse"],
            "average_hard_queries": val_metrics["average_hard_queries"],
            "routing_entropy": val_metrics["routing_entropy"],
        }
        rows.append(row)
        if val_metrics["mae"] < best_mae:
            best_mae = float(val_metrics["mae"])
            best_metrics = val_metrics
            best_epoch = epoch
            save_checkpoint(ckpt_path, model, epoch, args, val_metrics)
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= args.patience:
            break

    fieldnames = list(rows[0].keys()) if rows else []
    write_csv(seed_root / "training_curve.csv", rows, fieldnames)
    if first_grad_report is not None:
        (seed_root / "gradient_report.json").write_text(json.dumps(first_grad_report, indent=2), encoding="utf-8")
    assert best_metrics is not None
    summary = {
        "seed": seed,
        "best_epoch": best_epoch,
        "checkpoint_path": str(ckpt_path),
        "validation": best_metrics,
        "first_batch_gradient_report": first_grad_report,
    }
    (seed_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = TrainingConfig()
    for field, value in asdict(defaults).items():
        if field == "seeds":
            parser.add_argument("--seeds", default=",".join(str(seed) for seed in defaults.seeds))
        elif isinstance(value, bool):
            parser.add_argument(f"--{field.replace('_', '-')}", action="store_true", default=value)
        else:
            parser.add_argument(f"--{field.replace('_', '-')}", type=type(value), default=value)
    parser.add_argument("--gumbel-routing", action="store_true")
    parser.add_argument("--straight-through", action="store_true")
    args = parser.parse_args()
    args.seeds = tuple(int(item) for item in str(args.seeds).split(",") if item.strip())
    if args.val_end > 11520:
        raise ValueError("Validation-first training refuses ranges beyond [8640,11520); test is not allowed.")
    full_data = load_train_val_data(ROOT / args.data_dir)
    mean, std = fit_scaler(full_data, args.train_end)
    summaries = [run_seed(seed, args, full_data, mean, std) for seed in args.seeds]
    mae_values = [float(item["validation"]["mae"]) for item in summaries]
    mse_values = [float(item["validation"]["mse"]) for item in summaries]
    aggregate = {
        "method": "Full End-to-End Sequential COSTAR-TS",
        "dataset": args.dataset,
        "expert_order": EXPERT_ORDER,
        "train_split": {"start": 0, "end": args.train_end},
        "validation_split": {"start": args.train_end, "end": args.val_end},
        "test_accessed": False,
        "seeds": args.seeds,
        "validation_mae_mean": float(statistics.mean(mae_values)),
        "validation_mae_std": float(statistics.pstdev(mae_values)) if len(mae_values) > 1 else 0.0,
        "validation_mse_mean": float(statistics.mean(mse_values)),
        "validation_mse_std": float(statistics.pstdev(mse_values)) if len(mse_values) > 1 else 0.0,
        "runs": summaries,
        "safety": "NO TEST DATA USED",
    }
    output = ROOT / args.results_root / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
