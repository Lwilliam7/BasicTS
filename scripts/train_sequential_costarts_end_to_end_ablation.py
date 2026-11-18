"""Run optimization ablations for true end-to-end Sequential COSTAR-TS.

This script intentionally reads only ETTh1 train/validation files. It does not
load ``test_data.npy``.
"""

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


@dataclass(frozen=True)
class AblationSpec:
    name: str
    max_epochs: int
    patience: int
    min_query_warmup_epochs: int
    min_queries_during_warmup: int
    alpha_schedule: tuple[tuple[int, float], ...]
    lambda_counterfactual_query: float
    lambda_counterfactual_stop: float


ABLATIONS: dict[str, AblationSpec] = {
    "A_baseline_reproduction": AblationSpec("A_baseline_reproduction", 3, 2, 0, 1, ((1, 0.5),), 0.0, 0.0),
    "B_longer_training": AblationSpec("B_longer_training", 30, 7, 0, 1, ((1, 0.5),), 0.0, 0.0),
    "C_warmup_alpha_schedule": AblationSpec(
        "C_warmup_alpha_schedule",
        30,
        7,
        5,
        2,
        ((1, 1.0), (6, 0.5), (11, 0.1)),
        0.0,
        0.0,
    ),
    "D_counterfactual_query": AblationSpec(
        "D_counterfactual_query",
        30,
        7,
        5,
        2,
        ((1, 1.0), (6, 0.5), (11, 0.1)),
        0.1,
        0.0,
    ),
    "E_counterfactual_query_stop": AblationSpec(
        "E_counterfactual_query_stop",
        30,
        7,
        5,
        2,
        ((1, 1.0), (6, 0.5), (11, 0.1)),
        0.1,
        0.1,
    ),
}


def apply_runtime_overrides(spec: AblationSpec, args: argparse.Namespace) -> AblationSpec:
    max_epochs = spec.max_epochs if args.override_max_epochs is None else int(args.override_max_epochs)
    patience = spec.patience if args.override_patience is None else int(args.override_patience)
    return AblationSpec(
        name=spec.name,
        max_epochs=max_epochs,
        patience=patience,
        min_query_warmup_epochs=min(spec.min_query_warmup_epochs, max_epochs),
        min_queries_during_warmup=spec.min_queries_during_warmup,
        alpha_schedule=spec.alpha_schedule,
        lambda_counterfactual_query=spec.lambda_counterfactual_query,
        lambda_counterfactual_stop=spec.lambda_counterfactual_stop,
    )


class ETThTrainValDataset(Dataset):
    def __init__(
        self,
        data: np.ndarray,
        start: int,
        end: int,
        input_len: int,
        horizon: int,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> None:
        self.data = torch.tensor(data, dtype=torch.float32)
        self.starts = torch.arange(int(start), int(end) - int(input_len) - int(horizon) + 1, dtype=torch.long)
        self.input_len = int(input_len)
        self.horizon = int(horizon)
        self.mean = mean.to(torch.float32)
        self.std = std.to(torch.float32)

    def __len__(self) -> int:
        return int(self.starts.numel())

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start = int(self.starts[index].item())
        history = self.data[start : start + self.input_len]
        target = self.data[start + self.input_len : start + self.input_len + self.horizon]
        history = (history - self.mean) / self.std
        target = (target - self.mean) / self.std
        mask = torch.isfinite(target)
        return {
            "history": torch.nan_to_num(history),
            "target": torch.nan_to_num(target),
            "mask": mask,
            "absolute_window_start": torch.tensor(start, dtype=torch.long),
        }


def load_train_val_only(data_dir: Path) -> np.ndarray:
    train = np.load(data_dir / "train_data.npy")
    val = np.load(data_dir / "val_data.npy")
    full = np.concatenate((train, val), axis=0).astype(np.float32)
    if full.shape != (11520, 7):
        raise ValueError(f"Expected train+val shape (11520, 7), got {full.shape}")
    return full


def fit_scaler(data: np.ndarray, train_end: int) -> tuple[torch.Tensor, torch.Tensor]:
    train = torch.tensor(data[:train_end], dtype=torch.float32)
    return train.mean(dim=0), train.std(dim=0).clamp_min(1e-6)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def scheduled_alpha(spec: AblationSpec, epoch: int) -> float:
    value = spec.alpha_schedule[0][1]
    for start_epoch, alpha in spec.alpha_schedule:
        if epoch >= start_epoch:
            value = alpha
    return float(value)


def scheduled_min_queries(spec: AblationSpec, epoch: int) -> int:
    if spec.min_query_warmup_epochs > 0 and epoch <= spec.min_query_warmup_epochs:
        return int(spec.min_queries_during_warmup)
    return 1


def temperature_for_epoch(epoch: int, max_epochs: int, start: float, end: float) -> float:
    if max_epochs <= 1:
        return float(end)
    return float(start + (end - start) * (epoch - 1) / (max_epochs - 1))


def make_model(args: argparse.Namespace) -> FullEndToEndCOSTARTS:
    return FullEndToEndCOSTARTS(
        EndToEndCOSTARTSConfig(
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
    )


def make_optimizer(model: FullEndToEndCOSTARTS, args: argparse.Namespace) -> torch.optim.Optimizer:
    stop_ids = {id(param) for param in model.stop_head.parameters()}
    router_params = [param for param in model.router_parameters() if id(param) not in stop_ids]
    return torch.optim.AdamW(
        [
            {"params": router_params, "lr": args.router_lr, "name": "router"},
            {"params": model.stop_head.parameters(), "lr": args.stop_lr, "name": "stop_head"},
            {"params": model.expert_parameters(), "lr": args.expert_lr, "name": "experts"},
        ],
        weight_decay=args.weight_decay,
    )


def optimizer_lr_row(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    return {f"lr_{group.get('name', index)}": float(group["lr"]) for index, group in enumerate(optimizer.param_groups)}


def train_one_epoch(
    model: FullEndToEndCOSTARTS,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    spec: AblationSpec,
    epoch: int,
) -> tuple[dict[str, float], dict[str, float]]:
    model.train()
    sums: dict[str, float] = {}
    grad_sums: dict[str, float] = {name: 0.0 for name in (*EXPERT_ORDER, "COSTAR_router")}
    batches = 0
    alpha = scheduled_alpha(spec, epoch)
    min_queries = scheduled_min_queries(spec, epoch)
    temperature = temperature_for_epoch(epoch, spec.max_epochs, args.temp_start, args.temp_end)
    for batch in loader:
        history = batch["history"].to(device)
        target = batch["target"].to(device)
        mask = batch["mask"].to(device)
        outputs = model.forward_soft(history, temperature=temperature, minimum_queries=min_queries)
        losses = end_to_end_costarts_loss(
            outputs,
            target,
            mask,
            alpha_expert=alpha,
            lambda_query=args.lambda_query,
            lambda_balance=args.lambda_balance,
            lambda_stop=args.lambda_stop,
            lambda_counterfactual_query=spec.lambda_counterfactual_query,
            lambda_counterfactual_stop=spec.lambda_counterfactual_stop,
            counterfactual_tau=args.counterfactual_tau,
            minimum_queries=min_queries,
            query_cost=args.query_cost,
        )
        optimizer.zero_grad()
        losses["total_loss"].backward()
        report = gradient_report(model)
        for key, value in report.items():
            grad_sums[key] += value
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        optimizer.step()
        batches += 1
        for key, value in losses.items():
            if value.ndim == 0:
                sums[key] = sums.get(key, 0.0) + float(value.detach().cpu().item())
    metrics = {key: value / max(batches, 1) for key, value in sums.items()}
    grads = {f"grad_{key}": value / max(batches, 1) for key, value in grad_sums.items()}
    metrics["alpha_expert"] = alpha
    metrics["minimum_queries"] = float(min_queries)
    metrics["temperature"] = temperature
    return metrics, grads


@torch.no_grad()
def evaluate(model: FullEndToEndCOSTARTS, loader: DataLoader, device: torch.device, stop_threshold: float, minimum_queries: int) -> dict[str, Any]:
    model.eval()
    maes: list[torch.Tensor] = []
    mses: list[torch.Tensor] = []
    counts_all: list[torch.Tensor] = []
    query_counts = torch.zeros(len(EXPERT_ORDER), dtype=torch.float64)
    first_counts = torch.zeros(len(EXPERT_ORDER), dtype=torch.float64)
    second_counts = torch.zeros(len(EXPERT_ORDER), dtype=torch.float64)
    hist_counts = torch.zeros(len(EXPERT_ORDER), dtype=torch.float64)
    stop_sum = torch.zeros(len(EXPERT_ORDER), dtype=torch.float64)
    stop_n = torch.zeros(len(EXPERT_ORDER), dtype=torch.float64)
    expert_mae_sum = torch.zeros(len(EXPERT_ORDER), dtype=torch.float64)
    expert_mse_sum = torch.zeros(len(EXPERT_ORDER), dtype=torch.float64)
    n = 0
    for batch in loader:
        history = batch["history"].to(device)
        target = batch["target"].to(device)
        mask = batch["mask"].to(device)
        hard = model.forward_hard(history, stop_threshold=stop_threshold, minimum_queries=minimum_queries)
        maes.append(masked_sample_mae(hard["forecast"], target, mask).cpu())
        mses.append(masked_sample_mse(hard["forecast"], target, mask).cpu())
        counts = hard["query_counts"].cpu()
        counts_all.append(counts)
        for count in counts.tolist():
            if count > 0:
                hist_counts[int(count) - 1] += 1
        for row in hard["queried_ids"].cpu():
            valid = row[row >= 0]
            if valid.numel() > 0:
                first_counts[int(valid[0])] += 1
            if valid.numel() > 1:
                second_counts[int(valid[1])] += 1
            for expert_id in valid.unique().tolist():
                query_counts[int(expert_id)] += 1

        soft = model.forward_soft(history, temperature=model.config.route_temperature)
        expert_forecasts = soft["expert_forecasts"]
        for index in range(len(EXPERT_ORDER)):
            expert_mae_sum[index] += float(masked_sample_mae(expert_forecasts[..., index], target, mask).sum().cpu().item())
            expert_mse_sum[index] += float(masked_sample_mse(expert_forecasts[..., index], target, mask).sum().cpu().item())

        # Stop probabilities under the same hard-observed sequence.
        expert_all = model.expert_forecasts_all(history)
        batch_size = history.shape[0]
        current = torch.zeros(batch_size, model.config.forecast_horizon, model.config.num_features, device=device)
        observed = torch.zeros_like(current)
        mask_q = torch.zeros(batch_size, len(EXPERT_ORDER), device=device)
        forecast_sum = torch.zeros_like(current)
        active = torch.ones(batch_size, dtype=torch.bool, device=device)
        cvec = torch.zeros(batch_size, dtype=torch.long, device=device)
        for step in range(model.max_queries):
            scalar = torch.stack(
                (
                    cvec.to(torch.float32) / max(float(model.max_queries), 1.0),
                    mask_q.mean(dim=1),
                    active.to(torch.float32),
                    torch.full((batch_size,), float(step) / max(float(model.max_queries - 1), 1.0), device=device),
                ),
                dim=1,
            )
            rep = model._encode_state(history, mask_q, current, observed, scalar)
            stop_prob = torch.sigmoid(model.stop_head(rep).squeeze(-1))
            stop_sum[step] += float(stop_prob.sum().cpu().item())
            stop_n[step] += batch_size
            should = active if step < minimum_queries else active & (stop_prob <= float(stop_threshold))
            if not bool(should.any()):
                break
            logits = model.route_head(rep).masked_fill(mask_q.bool(), -1e9)
            next_ids = logits.argmax(dim=1)
            for expert_id in range(len(EXPERT_ORDER)):
                rows = should & (next_ids == expert_id)
                if bool(rows.any()):
                    forecast_sum[rows] += expert_all[rows, :, :, expert_id]
                    mask_q[rows, expert_id] = 1.0
                    cvec[rows] += 1
            current = forecast_sum / cvec.clamp_min(1).to(torch.float32)[:, None, None]
            observed = current
            active = should & (cvec < model.max_queries)
        n += history.shape[0]

    mae = torch.cat(maes)
    mse = torch.cat(mses)
    counts_tensor = torch.cat(counts_all).to(torch.float32)
    total = float(max(n, 1))
    second_total = float(max(second_counts.sum().item(), 1.0))
    return {
        "mae": float(mae.mean().item()),
        "mse": float(mse.mean().item()),
        "average_queries": float(counts_tensor.mean().item()),
        "query_count_percent": {str(i + 1): float(hist_counts[i].item() * 100.0 / total) for i in range(len(EXPERT_ORDER))},
        "first_query_percent": {EXPERT_ORDER[i]: float(first_counts[i].item() * 100.0 / total) for i in range(len(EXPERT_ORDER))},
        "second_query_percent_all_samples": {EXPERT_ORDER[i]: float(second_counts[i].item() * 100.0 / total) for i in range(len(EXPERT_ORDER))},
        "second_query_percent_among_second_queries": {EXPERT_ORDER[i]: float(second_counts[i].item() * 100.0 / second_total) for i in range(len(EXPERT_ORDER))},
        "expert_usage_percent": {EXPERT_ORDER[i]: float(query_counts[i].item() * 100.0 / total) for i in range(len(EXPERT_ORDER))},
        "mean_stop_prob_by_step": {str(i + 1): (float(stop_sum[i].item() / stop_n[i].item()) if stop_n[i] > 0 else None) for i in range(len(EXPERT_ORDER))},
        "individual_experts": [
            {"expert": EXPERT_ORDER[i], "mae": float(expert_mae_sum[i].item() / total), "mse": float(expert_mse_sum[i].item() / total)}
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


def flat_expert_maes(metrics: Mapping[str, Any], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{row['expert']}_mae": float(row["mae"]) for row in metrics["individual_experts"]}


def save_checkpoint(path: Path, model: FullEndToEndCOSTARTS, args: argparse.Namespace, spec: AblationSpec, epoch: int, metrics: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model.config_dict(),
            "args": vars(args),
            "ablation": asdict(spec),
            "epoch": epoch,
            "validation_metrics": dict(metrics),
            "expert_order": EXPERT_ORDER,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "safety": "NO TEST DATA USED; test_data.npy not loaded",
        },
        path,
    )


def run_ablation(spec: AblationSpec, args: argparse.Namespace, data: np.ndarray, mean: torch.Tensor, std: torch.Tensor) -> dict[str, Any]:
    set_seed(args.seed)
    device = torch.device(args.device)
    train_ds = ETThTrainValDataset(data, 0, args.train_end, args.input_len, args.forecast_horizon, mean, std)
    val_ds = ETThTrainValDataset(data, args.train_end, args.val_end, args.input_len, args.forecast_horizon, mean, std)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    model = make_model(args).to(device)
    optimizer = make_optimizer(model, args)
    result_dir = ROOT / args.results_root / spec.name / f"seed_{args.seed}"
    ckpt_path = ROOT / args.checkpoint_root / spec.name / f"seed_{args.seed}" / "best_full_end_to_end_costarts.pt"
    rows: list[dict[str, Any]] = []
    best_mae = float("inf")
    best_epoch = 0
    best_normal: dict[str, Any] | None = None
    best_forced2: dict[str, Any] | None = None
    bad_epochs = 0

    for epoch in range(1, spec.max_epochs + 1):
        train_metrics, grad_metrics = train_one_epoch(model, train_loader, optimizer, device, args, spec, epoch)
        normal = evaluate(model, val_loader, device, args.stop_threshold, minimum_queries=1)
        row = {
            "epoch": epoch,
            **optimizer_lr_row(optimizer),
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **grad_metrics,
            "validation_mae": normal["mae"],
            "validation_mse": normal["mse"],
            "validation_average_queries": normal["average_queries"],
            **flat_expert_maes(normal, "validation"),
        }
        rows.append(row)
        if normal["mae"] < best_mae:
            forced2 = evaluate(model, val_loader, device, args.stop_threshold, minimum_queries=2)
            row["forced2_validation_mae"] = forced2["mae"]
            row["forced2_validation_mse"] = forced2["mse"]
            row["forced2_average_queries"] = forced2["average_queries"]
            best_mae = float(normal["mae"])
            best_epoch = epoch
            best_normal = normal
            best_forced2 = forced2
            save_checkpoint(ckpt_path, model, args, spec, epoch, normal)
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= spec.patience:
            break

    fieldnames = sorted({key for row in rows for key in row})
    write_csv(result_dir / "training_curve.csv", rows, fieldnames)
    assert best_normal is not None and best_forced2 is not None
    summary = {
        "experiment": spec.name,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "checkpoint_path": str(ckpt_path),
        "normal_hard_validation": best_normal,
        "forced_min2_hard_validation": best_forced2,
        "spec": asdict(spec),
        "optimizer": {"expert_lr": args.expert_lr, "router_lr": args.router_lr, "stop_lr": args.stop_lr},
        "train_split": {"start": 0, "end": args.train_end},
        "validation_split": {"start": args.train_end, "end": args.val_end},
        "test_accessed": False,
        "safety": "NO TEST DATA USED; test_data.npy not loaded",
    }
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", default="A_baseline_reproduction,B_longer_training,C_warmup_alpha_schedule,D_counterfactual_query,E_counterfactual_query_stop")
    parser.add_argument("--override-max-epochs", type=int)
    parser.add_argument("--override-patience", type=int)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--data-dir", default="datasets/ETTh1")
    parser.add_argument("--results-root", default="results/router_summary/costarts_end_to_end_ablation")
    parser.add_argument("--checkpoint-root", default="checkpoints/costarts_end_to_end_ablation")
    parser.add_argument("--input-len", type=int, default=96)
    parser.add_argument("--forecast-horizon", type=int, default=12)
    parser.add_argument("--num-features", type=int, default=7)
    parser.add_argument("--train-end", type=int, default=8640)
    parser.add_argument("--val-end", type=int, default=11520)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--expert-hidden-size", type=int, default=64)
    parser.add_argument("--router-embedding-dim", type=int, default=64)
    parser.add_argument("--router-hidden-dim", type=int, default=64)
    parser.add_argument("--max-queries", type=int, default=5)
    parser.add_argument("--temp-start", type=float, default=2.0)
    parser.add_argument("--temp-end", type=float, default=0.5)
    parser.add_argument("--expert-lr", type=float, default=1e-4)
    parser.add_argument("--router-lr", type=float, default=3e-4)
    parser.add_argument("--stop-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--lambda-query", type=float, default=0.01)
    parser.add_argument("--lambda-balance", type=float, default=0.05)
    parser.add_argument("--lambda-stop", type=float, default=0.1)
    parser.add_argument("--query-cost", type=float, default=0.0)
    parser.add_argument("--counterfactual-tau", type=float, default=0.1)
    parser.add_argument("--stop-threshold", type=float, default=0.5)
    parser.add_argument("--gumbel-routing", action="store_true")
    parser.add_argument("--straight-through", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.val_end > 11520:
        raise ValueError("This validation-only ablation runner refuses to touch the final test range.")
    data = load_train_val_only(ROOT / args.data_dir)
    mean, std = fit_scaler(data, args.train_end)
    selected = [item.strip() for item in args.experiments.split(",") if item.strip()]
    summaries = [run_ablation(apply_runtime_overrides(ABLATIONS[name], args), args, data, mean, std) for name in selected]
    rows = [
        {
            "experiment": item["experiment"],
            "best_epoch": item["best_epoch"],
            "validation_mae": item["normal_hard_validation"]["mae"],
            "validation_mse": item["normal_hard_validation"]["mse"],
            "average_queries": item["normal_hard_validation"]["average_queries"],
            "forced2_mae": item["forced_min2_hard_validation"]["mae"],
            "forced2_mse": item["forced_min2_hard_validation"]["mse"],
            "forced2_average_queries": item["forced_min2_hard_validation"]["average_queries"],
        }
        for item in summaries
    ]
    output_root = ROOT / args.results_root
    write_csv(output_root / "ablation_summary.csv", rows, list(rows[0].keys()))
    aggregate = {
        "seed": args.seed,
        "experiments": summaries,
        "summary_rows": rows,
        "train_split": {"start": 0, "end": args.train_end},
        "validation_split": {"start": args.train_end, "end": args.val_end},
        "test_accessed": False,
        "safety": "NO TEST DATA USED; test_data.npy not loaded",
    }
    (output_root / "summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
