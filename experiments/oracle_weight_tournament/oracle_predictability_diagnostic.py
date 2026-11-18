"""Predictability diagnostic for COSTAR-TS oracle fixed-3 weights.

This script trains a diagnostic router on router-train windows only:

input: history + all five frozen expert forecasts
target: train-only fixed-3 oracle weights and train-only prototype labels

Validation oracle labels are computed only after training for measurement.  They
are not used for training, early stopping, or model selection.  No test cache is
loaded.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.oracle_weight_tournament.run_tournament import (  # noqa: E402
    FIXED3,
    args_global_weights,
    fixed3_indices,
    kmeans,
    load_cache,
    load_std,
    oracle_weights_grid,
)


class OraclePredictabilityDataset(Dataset):
    def __init__(self, cache: Mapping[str, Any], weights: torch.Tensor, labels: torch.Tensor) -> None:
        self.histories = cache["histories"].to(torch.float32)
        self.all_forecasts = cache["prediction_stack"].to(torch.float32)
        self.weights = weights.to(torch.float32)
        self.labels = labels.to(torch.long)

    def __len__(self) -> int:
        return int(self.histories.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "history": self.histories[index],
            "all_forecasts": self.all_forecasts[index],
            "weights": self.weights[index],
            "label": self.labels[index],
        }


class OraclePredictor(nn.Module):
    def __init__(self, input_len: int, horizon: int, num_features: int, num_experts: int, num_prototypes: int, hidden: int = 64) -> None:
        super().__init__()
        self.history_proj = nn.Linear(num_features, hidden)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.history_encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.pos = nn.Parameter(torch.zeros(1, input_len, hidden))
        nn.init.normal_(self.pos, std=0.02)

        forecast_dim = horizon * num_features * num_experts
        scalar_dim = num_experts * 4 + (num_experts * (num_experts - 1) // 2) * 2
        self.forecast_encoder = nn.Sequential(
            nn.Linear(forecast_dim + scalar_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.weight_head = nn.Linear(hidden, 3)
        self.prototype_head = nn.Linear(hidden, num_prototypes)

    def forward(self, history: torch.Tensor, all_forecasts: torch.Tensor) -> dict[str, torch.Tensor]:
        hist = self.history_proj(history) + self.pos[:, : history.shape[1]]
        hist = self.history_encoder(hist).mean(dim=1)
        frep = self.forecast_encoder(torch.cat((all_forecasts.flatten(1), all_forecast_scalars(all_forecasts)), dim=1))
        rep = self.head(torch.cat((hist, frep), dim=1))
        return {
            "weights": torch.softmax(self.weight_head(rep), dim=1),
            "prototype_logits": self.prototype_head(rep),
        }


def all_forecast_scalars(forecasts: torch.Tensor) -> torch.Tensor:
    vals = []
    for i in range(forecasts.shape[-1]):
        x = forecasts[..., i]
        vals.extend([x.mean(dim=(1, 2)), x.std(dim=(1, 2), unbiased=False), x.abs().amax(dim=(1, 2)), x[:, -1].mean(dim=1)])
    for i in range(forecasts.shape[-1]):
        for j in range(i + 1, forecasts.shape[-1]):
            diff = forecasts[..., i] - forecasts[..., j]
            vals.extend([diff.abs().mean(dim=(1, 2)), diff.abs().amax(dim=(1, 2))])
    return torch.stack(vals, dim=1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def nearest_prototype_labels(weights: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    return torch.cdist(weights.to(torch.float32), prototypes.to(torch.float32)).argmin(dim=1)


@torch.no_grad()
def evaluate(model: OraclePredictor, dataset: OraclePredictabilityDataset, device: torch.device) -> dict[str, Any]:
    model.eval()
    loader = DataLoader(dataset, batch_size=1024, shuffle=False)
    preds, targets, logits, labels = [], [], [], []
    for batch in loader:
        out = model(batch["history"].to(device), batch["all_forecasts"].to(device))
        preds.append(out["weights"].cpu())
        targets.append(batch["weights"].cpu())
        logits.append(out["prototype_logits"].cpu())
        labels.append(batch["label"].cpu())
    pred = torch.cat(preds)
    target = torch.cat(targets)
    logit = torch.cat(logits)
    label = torch.cat(labels)
    ss_res = (pred - target).square().sum(dim=0)
    ss_tot = (target - target.mean(dim=0, keepdim=True)).square().sum(dim=0).clamp_min(1e-12)
    r2_per_weight = 1.0 - ss_res / ss_tot
    overall_ss_res = (pred - target).square().sum()
    overall_ss_tot = (target - target.mean(dim=0, keepdim=True)).square().sum().clamp_min(1e-12)
    return {
        "r2_overall": float(1.0 - overall_ss_res / overall_ss_tot),
        "r2_patchtst": float(r2_per_weight[0]),
        "r2_itransformer": float(r2_per_weight[1]),
        "r2_timesnet": float(r2_per_weight[2]),
        "prototype_accuracy": float((logit.argmax(dim=1) == label).to(torch.float32).mean()),
        "top1_oracle_expert_accuracy": float((pred.argmax(dim=1) == target.argmax(dim=1)).to(torch.float32).mean()),
        "oracle_weight_cosine_similarity": float(F.cosine_similarity(pred, target, dim=1).mean()),
        "weight_l1": float((pred - target).abs().mean()),
        "pred_mean_weights": {FIXED3[i]: float(pred[:, i].mean()) for i in range(3)},
        "target_mean_weights": {FIXED3[i]: float(target[:, i].mean()) for i in range(3)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", default="cache/costarts_walkforward/router_train_20_60_cache.pt")
    parser.add_argument("--val-cache", default="cache/costarts_walkforward/router_val_60_80_cache.pt")
    parser.add_argument("--normalizer-checkpoint", default="checkpoints/costarts_walkforward/final_60/DLinear/best_expert.pt")
    parser.add_argument("--teacher-cache", default="experiments/oracle_weight_tournament/teacher_cache/teachers_step_0p02.pt")
    parser.add_argument("--out-dir", default="experiments/oracle_weight_tournament/predictability_diagnostic")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--teacher-lambda", type=float, default=0.01)
    parser.add_argument("--num-prototypes", type=int, default=16)
    parser.add_argument("--teacher-grid-step", type=float, default=0.02)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    train_cache = load_cache(ROOT / args.train_cache, "router_train_20_60")
    val_cache = load_cache(ROOT / args.val_cache, "router_val_60_80")
    if "test" in str(args.train_cache).lower() or "test" in str(args.val_cache).lower():
        raise ValueError("Refusing to load test data")

    std = load_std(ROOT / args.normalizer_checkpoint, int(train_cache["num_features"]))
    teachers = torch.load(ROOT / args.teacher_cache, map_location="cpu", weights_only=False)
    key = f"weights_lambda_{args.teacher_lambda}"
    train_weights = teachers[key].to(torch.float32)
    prototypes, train_labels = kmeans(train_weights, args.num_prototypes, args.seed)

    val_oracle_path = out_dir / f"val_oracle_lambda_{str(args.teacher_lambda).replace('.', 'p')}_step_{str(args.teacher_grid_step).replace('.', 'p')}.pt"
    if val_oracle_path.exists():
        val_weights = torch.load(val_oracle_path, map_location="cpu", weights_only=False)
    else:
        val_weights, _ = oracle_weights_grid(
            val_cache["prediction_stack"][..., fixed3_indices(val_cache)].to(torch.float32),
            val_cache["targets"].to(torch.float32),
            val_cache["target_masks"].to(torch.bool),
            std,
            torch.tensor(args_global_weights(), dtype=torch.float32),
            args.teacher_lambda,
            args.teacher_grid_step,
        )
        torch.save(val_weights, val_oracle_path)
    val_labels = nearest_prototype_labels(val_weights, prototypes)

    train_ds = OraclePredictabilityDataset(train_cache, train_weights, train_labels)
    val_ds = OraclePredictabilityDataset(val_cache, val_weights, val_labels)
    model = OraclePredictor(
        int(train_cache["input_len"]),
        int(train_cache["forecast_horizon"]),
        int(train_cache["num_features"]),
        len(train_cache["expert_names"]),
        args.num_prototypes,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    history = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in loader:
            out = model(batch["history"].to(device), batch["all_forecasts"].to(device))
            target = batch["weights"].to(device)
            label = batch["label"].to(device)
            loss = F.smooth_l1_loss(out["weights"], target) + F.cross_entropy(out["prototype_logits"], label)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch == args.epochs or epoch % 10 == 0:
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "train": evaluate(model, train_ds, device), "validation": evaluate(model, val_ds, device)})

    result = {
        "config": vars(args),
        "fixed3_target_experts": list(FIXED3),
        "input_experts": list(train_cache["expert_names"]),
        "train_windows": int(train_cache["num_windows"]),
        "validation_windows": int(val_cache["num_windows"]),
        "metrics": {"train": evaluate(model, train_ds, device), "validation": evaluate(model, val_ds, device)},
        "history": history,
        "runtime_sec": time.time() - start,
        "safety": "NO TEST DATA USED. Validation oracle labels are diagnostic-only and never used for training/model selection.",
    }
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    torch.save({"state_dict": model.state_dict(), "prototypes": prototypes, "config": vars(args)}, out_dir / "model.pt")
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
