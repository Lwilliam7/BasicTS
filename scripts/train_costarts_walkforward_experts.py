"""Train fresh walk-forward COSTAR-TS forecasting experts and build caches.

This helper uses direct PyTorch training on chronological ETTh windows so each
stage can control exactly which raw rows are visible to expert weights.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicts.models.DLinear import DLinear, DLinearConfig
from basicts.models.PatchTST import PatchTSTConfig, PatchTSTForForecasting
from basicts.models.TimesNet import TimesNetConfig, TimesNetForForecasting
from basicts.models.iTransformer import iTransformerConfig, iTransformerForForecasting

from scripts.build_costarts_walkforward_cache import (
    EXPERT_ORDER,
    build_stage_cache,
    chronological_ranges,
    combine_router_train,
    load_full_array,
    sha256_file,
    stage_specs,
    valid_window_starts,
)


@dataclass(frozen=True)
class ExpertStage:
    name: str
    train_start: int
    train_end: int
    checkpoint_dir: str
    predict_role: str | None


class WindowDataset(Dataset):
    def __init__(
        self,
        full_data: np.ndarray,
        starts: torch.Tensor,
        input_len: int,
        horizon: int,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> None:
        self.full_data = torch.tensor(full_data, dtype=torch.float32)
        self.starts = starts.to(torch.long)
        self.input_len = int(input_len)
        self.horizon = int(horizon)
        self.mean = mean.to(torch.float32)
        self.std = std.to(torch.float32)

    def __len__(self) -> int:
        return int(self.starts.numel())

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start = int(self.starts[index].item())
        history = self.full_data[start : start + self.input_len]
        target = self.full_data[start + self.input_len : start + self.input_len + self.horizon]
        return {
            "history": (history - self.mean) / self.std,
            "target": (target - self.mean) / self.std,
            "start": torch.tensor(start, dtype=torch.long),
        }


class ModernTCNForForecasting(nn.Module):
    """Compact ModernTCN-style depthwise temporal convolution forecaster."""

    def __init__(
        self,
        input_len: int,
        output_len: int,
        num_features: int,
        hidden_size: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(num_features, hidden_size)
        blocks = []
        for layer in range(num_layers):
            dilation = 2**layer
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(
                        hidden_size,
                        hidden_size,
                        kernel_size=5,
                        padding=2 * dilation,
                        dilation=dilation,
                        groups=hidden_size,
                    ),
                    nn.GELU(),
                    nn.Conv1d(hidden_size, hidden_size, kernel_size=1),
                    nn.Dropout(dropout),
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_size) for _ in range(num_layers))
        self.head = nn.Linear(input_len * hidden_size, output_len * num_features)
        self.output_len = int(output_len)
        self.num_features = int(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(inputs)
        for block, norm in zip(self.blocks, self.norms):
            residual = hidden
            conv = block(hidden.transpose(1, 2)).transpose(1, 2)
            hidden = norm(residual + conv[:, : hidden.shape[1], :])
        out = self.head(hidden.flatten(1))
        return out.reshape(inputs.shape[0], self.output_len, self.num_features)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_scaler(full_data: np.ndarray, train_start: int, train_end: int) -> tuple[torch.Tensor, torch.Tensor]:
    train = torch.tensor(full_data[train_start:train_end], dtype=torch.float32)
    mean = train.mean(dim=0)
    std = train.std(dim=0).clamp_min(1e-6)
    return mean, std


def train_val_starts(train_start: int, train_end: int, input_len: int, horizon: int, val_fraction: float) -> tuple[torch.Tensor, torch.Tensor]:
    usable_end = train_end - input_len - horizon + 1
    starts = torch.arange(train_start, usable_end, dtype=torch.long)
    if starts.numel() < 4:
        raise ValueError("Training range is too short")
    val_count = max(1, int(starts.numel() * val_fraction))
    if val_count >= starts.numel():
        val_count = 1
    return starts[:-val_count], starts[-val_count:]


def build_model(expert: str, input_len: int, horizon: int, features: int, hidden_size: int) -> nn.Module:
    if expert == "DLinear":
        return DLinear(DLinearConfig(input_len=input_len, output_len=horizon, num_features=features, individual=False))
    if expert == "PatchTST":
        return PatchTSTForForecasting(
            PatchTSTConfig(
                input_len=input_len,
                output_len=horizon,
                num_features=features,
                hidden_size=hidden_size,
                intermediate_size=hidden_size * 2,
                n_heads=1,
                num_layers=1,
                patch_len=16,
                patch_stride=8,
            )
        )
    if expert == "iTransformer":
        return iTransformerForForecasting(
            iTransformerConfig(
                input_len=input_len,
                output_len=horizon,
                num_features=features,
                hidden_size=hidden_size,
                intermediate_size=hidden_size * 2,
                n_heads=1,
                num_layers=1,
            )
        )
    if expert == "TimesNet":
        return TimesNetForForecasting(
            TimesNetConfig(
                input_len=input_len,
                output_len=horizon,
                num_features=features,
                hidden_size=hidden_size,
                intermediate_size=hidden_size * 2,
                num_layers=1,
                num_kernels=3,
                top_k=3,
                use_timestamps=False,
            )
        )
    if expert == "ModernTCN":
        return ModernTCNForForecasting(input_len, horizon, features, hidden_size=hidden_size)
    raise ValueError(f"Unknown expert: {expert}")


def call_model(model: nn.Module, history: torch.Tensor) -> torch.Tensor:
    try:
        out = model(history)
    except TypeError:
        out = model(history, None)
    if isinstance(out, Mapping):
        out = out["prediction"]
    return out


def train_expert(
    *,
    expert: str,
    full_data: np.ndarray,
    train_start: int,
    train_end: int,
    input_len: int,
    horizon: int,
    hidden_size: int,
    max_epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    val_fraction: float,
    seed: int,
    device: torch.device,
    checkpoint_path: Path,
) -> dict[str, Any]:
    set_seed(seed)
    mean, std = fit_scaler(full_data, train_start, train_end)
    train_starts, val_starts = train_val_starts(train_start, train_end, input_len, horizon, val_fraction)
    train_ds = WindowDataset(full_data, train_starts, input_len, horizon, mean, std)
    val_ds = WindowDataset(full_data, val_starts, input_len, horizon, mean, std)
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model = build_model(expert, input_len, horizon, full_data.shape[1], hidden_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_val = math.inf
    best_state = None
    best_epoch = -1
    bad_epochs = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch in train_loader:
            history = batch["history"].to(device)
            target = batch["target"].to(device)
            pred = call_model(model, history)
            loss = F.mse_loss(pred, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        val_losses = []
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                history = batch["history"].to(device)
                target = batch["target"].to(device)
                pred = call_model(model, history)
                val_losses.append(float(F.l1_loss(pred, target).cpu().item()))
        val_mae = float(np.mean(val_losses))
        if val_mae < best_val:
            best_val = val_mae
            best_epoch = epoch
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    if best_state is None:
        raise RuntimeError(f"No checkpoint produced for {expert}")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "expert": expert,
        "model_state_dict": best_state,
        "model_config": {
            "input_len": input_len,
            "output_len": horizon,
            "num_features": int(full_data.shape[1]),
            "hidden_size": hidden_size,
        },
        "train_range": {"start": train_start, "end": train_end},
        "internal_validation": {
            "policy": "chronological tail carved from expert training range only",
            "val_fraction": val_fraction,
            "num_train_windows": int(train_starts.numel()),
            "num_val_windows": int(val_starts.numel()),
            "first_val_window_start": int(val_starts[0].item()),
            "last_val_window_start": int(val_starts[-1].item()),
        },
        "scaler_mean": mean,
        "scaler_std": std,
        "best_epoch": best_epoch,
        "best_internal_val_mae": best_val,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
    }
    torch.save(payload, checkpoint_path)
    payload["checkpoint_path"] = str(checkpoint_path)
    payload["checkpoint_sha256"] = sha256_file(checkpoint_path)
    return {key: value for key, value in payload.items() if key not in {"model_state_dict", "scaler_mean", "scaler_std"}}


def load_expert(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, torch.Tensor, torch.Tensor, dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["model_config"]
    model = build_model(ckpt["expert"], int(cfg["input_len"]), int(cfg["output_len"]), int(cfg["num_features"]), int(cfg["hidden_size"]))
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, ckpt["scaler_mean"].to(device), ckpt["scaler_std"].to(device), ckpt


def predict_expert(
    *,
    checkpoint_path: Path,
    full_data: np.ndarray,
    starts: torch.Tensor,
    input_len: int,
    horizon: int,
    batch_size: int,
    device: torch.device,
    output_path: Path,
) -> dict[str, Any]:
    model, mean, std, ckpt = load_expert(checkpoint_path, device)
    dataset = WindowDataset(full_data, starts, input_len, horizon, mean.cpu(), std.cpu())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    preds = []
    with torch.no_grad():
        for batch in loader:
            history = batch["history"].to(device)
            pred_scaled = call_model(model, history)
            pred_raw = pred_scaled * std.view(1, 1, -1) + mean.view(1, 1, -1)
            preds.append(pred_raw.detach().cpu())
    prediction = torch.cat(preds, dim=0).to(torch.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, prediction.numpy())
    return {
        "expert": ckpt["expert"],
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "prediction_path": str(output_path),
        "prediction_sha256": sha256_file(output_path),
        "num_windows": int(prediction.shape[0]),
    }


def stage_definitions(total_timestamps: int, checkpoint_root: str = "checkpoints/costarts_walkforward") -> dict[str, ExpertStage]:
    """Stage train-end boundaries are always 20/40/60% of the dataset's own
    length, using the exact same `int(total * fraction)` truncation as
    `chronological_ranges()` in build_costarts_walkforward_cache.py so the two
    boundary definitions can never disagree -- not hardcoded to ETTh1's 14400
    timestamps. `checkpoint_root` defaults to the original ETTh1 path so
    existing behavior/artifacts are unchanged; pass a dataset-specific root
    for any other dataset so checkpoints never collide."""
    return {
        "block_a": ExpertStage("block_a", 0, int(total_timestamps * 0.2), f"{checkpoint_root}/block_a", "block_b_oos"),
        "block_ab": ExpertStage("block_ab", 0, int(total_timestamps * 0.4), f"{checkpoint_root}/block_ab", "block_c_oos"),
        "final_60": ExpertStage("final_60", 0, int(total_timestamps * 0.6), f"{checkpoint_root}/final_60", "router_val_60_80"),
    }


def checkpoint_path_for(stage: ExpertStage, expert: str) -> Path:
    return ROOT / stage.checkpoint_dir / expert / "best_expert.pt"


def train_stage(args: argparse.Namespace, stage: ExpertStage, full_data: np.ndarray, device: torch.device) -> dict[str, Any]:
    rows = []
    for expert in EXPERT_ORDER:
        path = checkpoint_path_for(stage, expert)
        if path.exists() and not args.force:
            rows.append({"expert": expert, "checkpoint_path": str(path), "checkpoint_sha256": sha256_file(path), "reused": True})
            continue
        rows.append(
            train_expert(
                expert=expert,
                full_data=full_data,
                train_start=stage.train_start,
                train_end=stage.train_end,
                input_len=args.input_len,
                horizon=args.forecast_horizon,
                hidden_size=args.hidden_size,
                max_epochs=args.max_epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                val_fraction=args.internal_val_fraction,
                seed=args.expert_seed,
                device=device,
                checkpoint_path=path,
            )
        )
    manifest = {
        "stage": stage.name,
        "train_range": {"start": stage.train_start, "end": stage.train_end},
        "expert_order": list(EXPERT_ORDER),
        "checkpoints": {row["expert"]: row["checkpoint_path"] for row in rows},
        "checkpoint_hashes": {row["expert"]: row["checkpoint_sha256"] for row in rows},
        "rows": rows,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
    }
    manifest_path = ROOT / stage.checkpoint_dir / "expert_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def predict_stage(args: argparse.Namespace, stage: ExpertStage, full_data: np.ndarray, device: torch.device) -> dict[str, Any]:
    if stage.predict_role is None:
        raise ValueError(f"Stage {stage.name} has no prediction role")
    ranges = chronological_ranges(full_data.shape[0])
    stages = stage_specs(ROOT / args.cache_dir, ranges)
    cache_stage = stages[stage.predict_role]
    starts = valid_window_starts(cache_stage.prediction_range, args.input_len, args.forecast_horizon)
    rows = []
    for expert in EXPERT_ORDER:
        rows.append(
            predict_expert(
                checkpoint_path=checkpoint_path_for(stage, expert),
                full_data=full_data,
                starts=starts,
                input_len=args.input_len,
                horizon=args.forecast_horizon,
                batch_size=args.batch_size,
                device=device,
                output_path=ROOT / args.prediction_dir / stage.predict_role / f"{expert}.npy",
            )
        )
    prediction_manifest = {
        "stage": stage.name,
        "predict_role": stage.predict_role,
        "expert_training_range": {"start": stage.train_start, "end": stage.train_end},
        "prediction_range": asdict(cache_stage.prediction_range),
        "expert_order": list(EXPERT_ORDER),
        "rows": rows,
        "expert_checkpoint_paths": {row["expert"]: row["checkpoint_path"] for row in rows},
        "expert_checkpoint_hashes": {row["expert"]: row["checkpoint_sha256"] for row in rows},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
    }
    manifest_path = ROOT / args.prediction_dir / stage.predict_role / "prediction_manifest.json"
    manifest_path.write_text(json.dumps(prediction_manifest, indent=2), encoding="utf-8")
    build_stage_cache(
        dataset=args.dataset,
        data_dir=ROOT / args.data_dir,
        prediction_dir=ROOT / args.prediction_dir,
        stage=cache_stage,
        input_len=args.input_len,
        horizon=args.forecast_horizon,
        error_temperature=args.error_temperature,
        checkpoint_manifest=manifest_path,
        allow_test=False,
    )
    return prediction_manifest


def write_pipeline_report(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="ETTh1")
    parser.add_argument("--data-dir", default="datasets/ETTh1")
    parser.add_argument("--cache-dir", default="cache/costarts_walkforward")
    parser.add_argument("--prediction-dir", default="results/router_summary/costarts_walkforward/expert_predictions")
    parser.add_argument("--results-dir", default="results/router_summary/costarts_walkforward/experts")
    parser.add_argument("--checkpoint-root", default="checkpoints/costarts_walkforward")
    parser.add_argument("--input-len", type=int, default=96)
    parser.add_argument("--forecast-horizon", type=int, default=12)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--internal-val-fraction", type=float, default=0.1)
    parser.add_argument("--error-temperature", type=float, default=0.1)
    parser.add_argument("--expert-seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stage", choices=("block_a", "block_ab", "final_60"))
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--combine", action="store_true")
    args = parser.parse_args()

    if args.predict_only and args.train_only:
        raise ValueError("--train-only and --predict-only are mutually exclusive")
    full_data = load_full_array(ROOT / args.data_dir)
    device = torch.device(args.device)
    definitions = stage_definitions(full_data.shape[0], args.checkpoint_root)
    selected = [definitions[args.stage]] if args.stage else [definitions["block_a"], definitions["block_ab"], definitions["final_60"]]
    report: dict[str, Any] = {
        "dataset": args.dataset,
        "data_dir": args.data_dir,
        "expert_order": list(EXPERT_ORDER),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "stages": {},
        "safety": "NO TEST DATA USED",
    }
    for stage in selected:
        stage_report: dict[str, Any] = {}
        if not args.predict_only:
            stage_report["training"] = train_stage(args, stage, full_data, device)
        if not args.train_only and stage.predict_role is not None:
            stage_report["prediction"] = predict_stage(args, stage, full_data, device)
        report["stages"][stage.name] = stage_report
        if stage.name == "block_ab" and args.combine:
            cache = combine_router_train(
                ROOT / args.cache_dir / "block_b_oos_cache.pt",
                ROOT / args.cache_dir / "block_c_oos_cache.pt",
                ROOT / args.cache_dir / "router_train_20_60_cache.pt",
            )
            report["router_train_20_60"] = {"num_windows": int(cache["num_windows"])}
    if args.combine and "router_train_20_60" not in report:
        cache = combine_router_train(
            ROOT / args.cache_dir / "block_b_oos_cache.pt",
            ROOT / args.cache_dir / "block_c_oos_cache.pt",
            ROOT / args.cache_dir / "router_train_20_60_cache.pt",
        )
        report["router_train_20_60"] = {"num_windows": int(cache["num_windows"])}
    report_path = ROOT / args.results_dir / "walkforward_expert_pipeline_report.json"
    write_pipeline_report(report_path, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
