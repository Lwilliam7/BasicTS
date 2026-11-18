"""Cheap grokking diagnostic for the strongest trainable COSTAR-TS router.

The selected model is the oracle-weight prototype-residual fixed-three router:

    final_phase2_protores_lam0.01_k16_scale0.3_rw0.001

It uses PatchTST, iTransformer, and TimesNet forecasts only.  The diagnostic
trains on an early chronological slice of router-train and evaluates on a later
router-train fold.  It does not load the validation or test cache.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.oracle_weight_tournament.run_tournament import (  # noqa: E402
    FIXED3,
    TrialConfig,
    WeightStudent,
    args_global_weights,
    fixed3_indices,
    kmeans,
    load_cache,
    load_std,
    oracle_weights_grid,
    sample_mae,
    sample_mse,
    weighted_forecast,
)


SELECTED = TrialConfig(
    "prototype_residual",
    "final_phase2_protores_lam0.01_k16_scale0.3_rw0.001",
    seed=7,
    num_prototypes=16,
    teacher_lambda=0.01,
    residual_scale=0.30,
    residual_weight=0.001,
    epochs=10,
    lr=1e-3,
    batch_size=512,
    teacher_weight=1.0,
    forecast_weight=1.0,
    feature_mix="full",
)
ORIGINAL_EPOCHS = 10
LONG_EPOCHS = 100
CURRENT_WEIGHT_DECAY = 0.01
WEIGHT_DECAYS = (0.001, 0.01, 0.1)
EARLY_REGION_END = ORIGINAL_EPOCHS
DELAYED_REGION_START = 50
MEANINGFUL_MARGIN = 0.0005
SUSTAINED_CHECKPOINTS = 5


class TensorWindowDataset(Dataset):
    def __init__(self, cache: Mapping[str, Any], indices: torch.Tensor) -> None:
        self.histories = cache["histories"][indices].to(torch.float32)
        self.forecasts = cache["prediction_stack"][indices][..., fixed3_indices(cache)].to(torch.float32)
        self.targets = cache["targets"][indices].to(torch.float32)
        self.masks = cache["target_masks"][indices].to(torch.bool)
        self.starts = cache["absolute_window_starts"][indices].to(torch.long)
        self.indices = indices.to(torch.long)

    def __len__(self) -> int:
        return int(self.indices.numel())

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "history": self.histories[index],
            "forecasts": self.forecasts[index],
            "target": self.targets[index],
            "mask": self.masks[index],
            "source_index": self.indices[index],
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def refuse_test(path: str | Path) -> None:
    if "test" in str(path).lower():
        raise ValueError(f"Refusing test path: {path}")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def make_split(n: int, eval_fraction: float) -> tuple[torch.Tensor, torch.Tensor]:
    split = int(round(n * (1.0 - eval_fraction)))
    split = max(1, min(split, n - 1))
    return torch.arange(split, dtype=torch.long), torch.arange(split, n, dtype=torch.long)


def fixed_weight_metrics(ds: TensorWindowDataset, std: torch.Tensor) -> dict[str, float]:
    weights = torch.tensor(args_global_weights(), dtype=torch.float32).view(1, 3).expand(len(ds), -1)
    pred = weighted_forecast(ds.forecasts, weights)
    mae = sample_mae(pred, ds.targets, ds.masks, std)
    mse = sample_mse(pred, ds.targets, ds.masks, std)
    return {"mae": float(mae.mean()), "mse": float(mse.mean())}


def build_teacher(cache: Mapping[str, Any], train_idx: torch.Tensor, std: torch.Tensor, grid_step: float, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    forecasts = cache["prediction_stack"][train_idx][..., fixed3_indices(cache)].to(torch.float32)
    targets = cache["targets"][train_idx].to(torch.float32)
    masks = cache["target_masks"][train_idx].to(torch.bool)
    global_weights = torch.tensor(args_global_weights(), dtype=torch.float32)
    teacher, teacher_mae = oracle_weights_grid(
        forecasts,
        targets,
        masks,
        std,
        global_weights,
        SELECTED.teacher_lambda,
        grid_step,
    )
    prototypes, labels = kmeans(teacher, SELECTED.num_prototypes, seed)
    return teacher, prototypes, labels


def loss_for_batch(
    model: WeightStudent,
    batch: Mapping[str, torch.Tensor],
    std: torch.Tensor,
    teacher: torch.Tensor,
    proto_labels: torch.Tensor,
    prototypes: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hist = batch["history"].to(device)
    forecasts = batch["forecasts"].to(device)
    target = batch["target"].to(device)
    mask = batch["mask"].to(device)
    source_idx = batch["source_index"].to(torch.long)
    out = model(hist, forecasts, prototypes=prototypes.to(device))
    weights = out["weights"]
    teach = teacher[source_idx].to(device)
    labels = proto_labels[source_idx].to(device)
    teacher_loss = F.smooth_l1_loss(weights, teach) + F.cross_entropy(out["logits"], labels)
    pred = weighted_forecast(forecasts, weights)
    forecast_loss = sample_mae(pred, target, mask, std.to(device)).mean()
    global_weights = torch.tensor(args_global_weights(), dtype=torch.float32, device=device)
    residual_loss = (weights - global_weights.view(1, 3)).square().mean()
    loss = SELECTED.forecast_weight * forecast_loss + SELECTED.teacher_weight * teacher_loss + SELECTED.residual_weight * residual_loss
    return loss, forecast_loss, teacher_loss, residual_loss


def train_epoch(
    model: WeightStudent,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    std: torch.Tensor,
    teacher: torch.Tensor,
    proto_labels: torch.Tensor,
    prototypes: torch.Tensor,
    device: torch.device,
) -> tuple[float, int]:
    model.train()
    losses: list[float] = []
    steps = 0
    for batch in loader:
        loss, _, _, _ = loss_for_batch(model, batch, std, teacher, proto_labels, prototypes, device)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        steps += 1
    return float(statistics.mean(losses)), steps


@torch.no_grad()
def evaluate_model(
    model: WeightStudent,
    ds: TensorWindowDataset,
    std: torch.Tensor,
    device: torch.device,
    prototypes: torch.Tensor,
    batch_size: int,
    teacher: torch.Tensor | None = None,
    proto_labels: torch.Tensor | None = None,
) -> dict[str, float]:
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    maes: list[torch.Tensor] = []
    mses: list[torch.Tensor] = []
    weights_all: list[torch.Tensor] = []
    losses: list[float] = []
    for batch in loader:
        hist = batch["history"].to(device)
        forecasts = batch["forecasts"].to(device)
        target = batch["target"].to(device)
        mask = batch["mask"].to(device)
        out = model(hist, forecasts, prototypes=prototypes.to(device))
        weights = out["weights"]
        pred = weighted_forecast(forecasts, weights)
        maes.append(sample_mae(pred, target, mask, std.to(device)).cpu())
        mses.append(sample_mse(pred, target, mask, std.to(device)).cpu())
        weights_all.append(weights.cpu())
        if teacher is not None and proto_labels is not None:
            loss, _, _, _ = loss_for_batch(model, batch, std, teacher, proto_labels, prototypes, device)
            losses.append(float(loss.cpu()))
    weights_cat = torch.cat(weights_all)
    global_weights = torch.tensor(args_global_weights(), dtype=torch.float32).view(1, 3)
    return {
        "mae": float(torch.cat(maes).mean()),
        "mse": float(torch.cat(mses).mean()),
        "loss": float(statistics.mean(losses)) if losses else math.nan,
        "mean_weight_adjustment_l1": float((weights_cat - global_weights).abs().mean()),
        "max_weight_adjustment_l1": float((weights_cat - global_weights).abs().max()),
        **{f"mean_weight_{FIXED3[i]}": float(weights_cat[:, i].mean()) for i in range(3)},
    }


def weight_norm(model: torch.nn.Module) -> float:
    return float(torch.sqrt(sum((p.detach().float().square().sum() for p in model.parameters()), torch.tensor(0.0))).cpu())


def save_checkpoint(path: Path, model: WeightStudent, epoch: int, optimizer_steps: int, config: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "optimizer_steps": optimizer_steps,
            "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "config": dict(config),
        },
        path,
    )


def run_one(
    cache: Mapping[str, Any],
    train_idx: torch.Tensor,
    eval_idx: torch.Tensor,
    teacher: torch.Tensor,
    prototypes: torch.Tensor,
    proto_labels: torch.Tensor,
    std: torch.Tensor,
    seed: int,
    weight_decay: float,
    epochs: int,
    out_dir: Path,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    set_seed(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train_ds = TensorWindowDataset(cache, train_idx)
    eval_ds = TensorWindowDataset(cache, eval_idx)
    train_loader = DataLoader(train_ds, batch_size=SELECTED.batch_size, shuffle=True)
    model = WeightStudent(
        args_global_weights(),
        int(cache["input_len"]),
        int(cache["forecast_horizon"]),
        int(cache["num_features"]),
        mode="prototype_residual",
        num_prototypes=SELECTED.num_prototypes,
        residual_scale=SELECTED.residual_scale,
        feature_mix=SELECTED.feature_mix,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=SELECTED.lr, weight_decay=float(weight_decay))
    train_fixed = fixed_weight_metrics(train_ds, std)
    eval_fixed = fixed_weight_metrics(eval_ds, std)
    rows: list[dict[str, Any]] = []
    optimizer_steps = 0
    t0 = time.time()

    def record(epoch: int, train_loss: float) -> None:
        train_metrics = evaluate_model(model, train_ds, std, device, prototypes, SELECTED.batch_size, teacher=teacher, proto_labels=proto_labels)
        eval_metrics = evaluate_model(model, eval_ds, std, device, prototypes, SELECTED.batch_size)
        row = {
            "seed": seed,
            "weight_decay": weight_decay,
            "epoch": epoch,
            "optimizer_step": optimizer_steps,
            "training_loss": train_loss if not math.isnan(train_loss) else train_metrics["loss"],
            "training_eval_loss": train_metrics["loss"],
            "training_mae": train_metrics["mae"],
            "training_mse": train_metrics["mse"],
            "chronological_fold_mae": eval_metrics["mae"],
            "chronological_fold_mse": eval_metrics["mse"],
            "train_improvement_vs_fixed_mae": train_fixed["mae"] - train_metrics["mae"],
            "fold_improvement_vs_fixed_mae": eval_fixed["mae"] - eval_metrics["mae"],
            "fixed_train_mae": train_fixed["mae"],
            "fixed_fold_mae": eval_fixed["mae"],
            "weight_norm": weight_norm(model),
            "mean_weight_adjustment_l1": eval_metrics["mean_weight_adjustment_l1"],
            "max_weight_adjustment_l1": eval_metrics["max_weight_adjustment_l1"],
            "mean_weight_PatchTST": eval_metrics["mean_weight_PatchTST"],
            "mean_weight_iTransformer": eval_metrics["mean_weight_iTransformer"],
            "mean_weight_TimesNet": eval_metrics["mean_weight_TimesNet"],
            "elapsed_sec": time.time() - t0,
            "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0,
        }
        rows.append(row)

    record(epoch=0, train_loss=math.nan)
    for epoch in range(1, epochs + 1):
        train_loss, steps = train_epoch(model, train_loader, optimizer, std, teacher, proto_labels, prototypes, device)
        optimizer_steps += steps
        record(epoch, train_loss)
        if epoch in {ORIGINAL_EPOCHS, epochs} or epoch % 10 == 0:
            save_checkpoint(
                out_dir / "checkpoints" / f"seed{seed}_wd{weight_decay:g}_epoch{epoch}.pt",
                model,
                epoch,
                optimizer_steps,
                {"seed": seed, "weight_decay": weight_decay, "epochs": epochs, "selected_model": SELECTED.name},
            )
    summary = {
        "seed": seed,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "runtime_sec": time.time() - t0,
        "peak_gpu_memory_mb": max((float(r["peak_gpu_memory_mb"]) for r in rows), default=0.0),
    }
    return rows, summary


def assess_grokking(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_wd: dict[float, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_wd.setdefault(float(row["weight_decay"]), []).append(row)
    assessments = []
    for wd, group in sorted(by_wd.items()):
        group = sorted(group, key=lambda r: int(r["epoch"]))
        early = [r for r in group if int(r["epoch"]) <= EARLY_REGION_END]
        delayed = [r for r in group if int(r["epoch"]) >= DELAYED_REGION_START]
        best_early = min(early, key=lambda r: float(r["chronological_fold_mae"]))
        best_late = min(delayed, key=lambda r: float(r["chronological_fold_mae"])) if delayed else None
        train_early = [r for r in group if 5 <= int(r["epoch"]) <= EARLY_REGION_END]
        plateau = False
        if len(train_early) >= 2:
            train_maes = [float(r["training_mae"]) for r in train_early]
            plateau = max(train_maes) - min(train_maes) < 0.0005
        sustained = False
        if best_late is not None:
            threshold = float(best_early["chronological_fold_mae"]) - MEANINGFUL_MARGIN
            streak = 0
            for row in delayed:
                if float(row["chronological_fold_mae"]) <= threshold:
                    streak += 1
                    sustained = sustained or streak >= SUSTAINED_CHECKPOINTS
                else:
                    streak = 0
        late_margin = float(best_early["chronological_fold_mae"]) - float(best_late["chronological_fold_mae"]) if best_late is not None else 0.0
        possible = bool(plateau and sustained and late_margin >= MEANINGFUL_MARGIN)
        assessments.append(
            {
                "weight_decay": wd,
                "best_early_epoch": int(best_early["epoch"]),
                "best_early_fold_mae": float(best_early["chronological_fold_mae"]),
                "best_late_epoch": int(best_late["epoch"]) if best_late is not None else None,
                "best_late_fold_mae": float(best_late["chronological_fold_mae"]) if best_late is not None else None,
                "late_minus_early_mae": -late_margin,
                "training_plateau_by_epoch10": plateau,
                "sustained_late_gain": sustained,
                "meaningful_margin": MEANINGFUL_MARGIN,
                "possible_grokking": possible,
            }
        )
    best = min(assessments, key=lambda r: float(r["best_late_fold_mae"] if r["best_late_fold_mae"] is not None else r["best_early_fold_mae"]))
    return {"by_weight_decay": assessments, "best_weight_decay_assessment": best, "any_possible_grokking": any(a["possible_grokking"] for a in assessments)}


def write_report(out_dir: Path, report: Mapping[str, Any]) -> None:
    best = report["assessment"]["best_weight_decay_assessment"]
    late_mae = "n/a" if best["best_late_fold_mae"] is None else f"{best['best_late_fold_mae']:.6f}"
    late_epoch = "n/a" if best["best_late_epoch"] is None else str(best["best_late_epoch"])
    lines = [
        "# COSTAR-TS Grokking Diagnostic",
        "",
        "## Selected Model",
        "",
        f"- Model: `{SELECTED.name}`",
        "- Family: prototype-residual oracle-weight student over PatchTST, iTransformer, TimesNet.",
        f"- Original training duration: `{ORIGINAL_EPOCHS}` epochs.",
        f"- Diagnostic duration: `{LONG_EPOCHS}` epochs.",
        f"- Weight decay settings: `{', '.join(str(x) for x in WEIGHT_DECAYS)}`.",
        "",
        "## Predefined Grokking Criteria",
        "",
        f"- Early checkpoint region: epochs `0..{EARLY_REGION_END}`.",
        f"- Delayed region starts at epoch `{DELAYED_REGION_START}`.",
        f"- Meaningful margin: `{MEANINGFUL_MARGIN}` MAE below best early checkpoint.",
        f"- Sustained: at least `{SUSTAINED_CHECKPOINTS}` consecutive delayed checkpoints.",
        "",
        "## Result",
        "",
        f"- Best delayed setting: weight decay `{best['weight_decay']}`.",
        f"- Best early fold MAE: `{best['best_early_fold_mae']:.6f}` at epoch `{best['best_early_epoch']}`.",
        f"- Best late fold MAE: `{late_mae}` at epoch `{late_epoch}`.",
        f"- Possible grokking: `{report['assessment']['any_possible_grokking']}`.",
        f"- Repeat seeds run: `{report['repeat_seeds_run']}`.",
        f"- Peak GPU memory MB: `{report['compute']['peak_gpu_memory_mb']:.1f}`.",
        f"- Runtime seconds: `{report['compute']['runtime_sec']:.1f}`.",
        "",
        "## Decision",
        "",
        report["decision"],
        "",
        "## Reproduce",
        "",
        "```powershell",
        report["reproduce_command"],
        "```",
    ]
    (out_dir / "grokking_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_curve_svg(out_dir: Path, rows: Sequence[Mapping[str, Any]], metric: str, filename: str) -> None:
    width, height = 840, 520
    pad_l, pad_r, pad_t, pad_b = 70, 24, 24, 60
    seed_rows = [r for r in rows if int(r["seed"]) == 7]
    if not seed_rows:
        return
    xs = [int(r["epoch"]) for r in seed_rows]
    ys = [float(r[metric]) for r in seed_rows]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if abs(ymax - ymin) < 1e-12:
        ymax = ymin + 1e-6
    colors = {0.001: "#2563eb", 0.01: "#16a34a", 0.1: "#dc2626"}

    def sx(x: float) -> float:
        return pad_l + (x - xmin) / max(xmax - xmin, 1) * (width - pad_l - pad_r)

    def sy(y: float) -> float:
        return height - pad_b - (y - ymin) / (ymax - ymin) * (height - pad_t - pad_b)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{pad_l}" y1="{height-pad_b}" x2="{width-pad_r}" y2="{height-pad_b}" stroke="#111" stroke-width="1"/>',
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{height-pad_b}" stroke="#111" stroke-width="1"/>',
        f'<rect x="{sx(0)}" y="{pad_t}" width="{sx(EARLY_REGION_END)-sx(0)}" height="{height-pad_t-pad_b}" fill="#999" opacity="0.12"/>',
        f'<line x1="{sx(DELAYED_REGION_START)}" y1="{pad_t}" x2="{sx(DELAYED_REGION_START)}" y2="{height-pad_b}" stroke="#111" stroke-dasharray="5 5"/>',
        f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-family="Arial" font-size="14">epoch</text>',
        f'<text x="18" y="{height/2}" text-anchor="middle" font-family="Arial" font-size="14" transform="rotate(-90 18 {height/2})">{metric}</text>',
        f'<text x="{pad_l}" y="18" font-family="Arial" font-size="14">{metric}: seed 7</text>',
        f'<text x="{pad_l}" y="{height-pad_b+22}" font-family="Arial" font-size="11">{xmin}</text>',
        f'<text x="{width-pad_r}" y="{height-pad_b+22}" text-anchor="end" font-family="Arial" font-size="11">{xmax}</text>',
        f'<text x="{pad_l-8}" y="{sy(ymin)+4}" text-anchor="end" font-family="Arial" font-size="11">{ymin:.6f}</text>',
        f'<text x="{pad_l-8}" y="{sy(ymax)+4}" text-anchor="end" font-family="Arial" font-size="11">{ymax:.6f}</text>',
    ]
    legend_y = pad_t + 18
    for idx, wd in enumerate(WEIGHT_DECAYS):
        group = sorted([r for r in seed_rows if abs(float(r["weight_decay"]) - wd) < 1e-12], key=lambda r: int(r["epoch"]))
        if not group:
            continue
        pts = " ".join(f"{sx(int(r['epoch'])):.2f},{sy(float(r[metric])):.2f}" for r in group)
        color = colors.get(wd, "#111")
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{pts}"/>')
        parts.append(f'<line x1="{width-170}" y1="{legend_y+idx*20}" x2="{width-145}" y2="{legend_y+idx*20}" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<text x="{width-138}" y="{legend_y+idx*20+4}" font-family="Arial" font-size="12">wd={wd:g}</text>')
    parts.append("</svg>")
    (out_dir / filename).write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", default="cache/costarts_walkforward/router_train_20_60_cache.pt")
    parser.add_argument("--normalizer-checkpoint", default="checkpoints/costarts_walkforward/final_60/DLinear/best_expert.pt")
    parser.add_argument("--out-dir", default="experiments/grokking_diagnostic_costar")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--teacher-grid-step", type=float, default=0.02)
    parser.add_argument("--long-epochs", type=int, default=LONG_EPOCHS)
    args = parser.parse_args()

    refuse_test(args.train_cache)
    refuse_test(args.normalizer_checkpoint)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    t0 = time.time()
    cache = load_cache(ROOT / args.train_cache, "router_train_20_60")
    starts = cache["absolute_window_starts"].to(torch.long)
    if not bool(torch.all(starts[1:] > starts[:-1])):
        raise ValueError("Router-train cache must be chronological")
    if int(starts[-1]) >= 8640:
        raise ValueError("Router-train cache touches validation or later")
    std = load_std(ROOT / args.normalizer_checkpoint, int(cache["num_features"]))
    train_idx, eval_idx = make_split(int(cache["num_windows"]), args.eval_fraction)
    teacher, prototypes, labels_local = build_teacher(cache, train_idx, std, args.teacher_grid_step, seed=7)
    # Expand train-local labels/teachers to cache-global indices for simple batch lookup.
    teacher_global = torch.zeros((int(cache["num_windows"]), 3), dtype=torch.float32)
    labels_global = torch.zeros((int(cache["num_windows"]),), dtype=torch.long)
    teacher_global[train_idx] = teacher
    labels_global[train_idx] = labels_local

    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_model": asdict(SELECTED),
        "model_choice_reason": "Best five-seed neural trainable fixed-three router; static component used by later chronological/HV hybrids. Ridge residual current best is not a long-trained neural router.",
        "fixed_three_experts": list(FIXED3),
        "router_train_cache": args.train_cache,
        "train_start_range": [int(starts[train_idx[0]]), int(starts[train_idx[-1]])],
        "chronological_eval_start_range": [int(starts[eval_idx[0]]), int(starts[eval_idx[-1]])],
        "validation_used": False,
        "test_used": False,
        "early_region_end": EARLY_REGION_END,
        "delayed_region_start": DELAYED_REGION_START,
        "meaningful_margin": MEANINGFUL_MARGIN,
        "sustained_checkpoints": SUSTAINED_CHECKPOINTS,
        "weight_decays": list(WEIGHT_DECAYS),
        "device": args.device,
    }
    write_json(out_dir / "run_metadata.json", meta)

    all_rows: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []
    for wd in WEIGHT_DECAYS:
        rows, summary = run_one(
            cache,
            train_idx,
            eval_idx,
            teacher_global,
            prototypes,
            labels_global,
            std,
            seed=7,
            weight_decay=wd,
            epochs=args.long_epochs,
            out_dir=out_dir,
            device=device,
        )
        all_rows.extend(rows)
        run_summaries.append(summary)
        write_csv(out_dir / "checkpoint_metrics.csv", all_rows)
        write_json(out_dir / "run_summaries.json", run_summaries)

    assessment = assess_grokking(all_rows)
    repeat_rows: list[dict[str, Any]] = []
    if assessment["any_possible_grokking"]:
        winning_wd = float(assessment["best_weight_decay_assessment"]["weight_decay"])
        for seed in (11, 13):
            repeat_teacher, repeat_prototypes, repeat_labels_local = build_teacher(cache, train_idx, std, args.teacher_grid_step, seed=seed)
            repeat_teacher_global = torch.zeros((int(cache["num_windows"]), 3), dtype=torch.float32)
            repeat_labels_global = torch.zeros((int(cache["num_windows"]),), dtype=torch.long)
            repeat_teacher_global[train_idx] = repeat_teacher
            repeat_labels_global[train_idx] = repeat_labels_local
            rows, summary = run_one(
                cache,
                train_idx,
                eval_idx,
                repeat_teacher_global,
                repeat_prototypes,
                repeat_labels_global,
                std,
                seed=seed,
                weight_decay=winning_wd,
                epochs=args.long_epochs,
                out_dir=out_dir,
                device=device,
            )
            repeat_rows.extend(rows)
            run_summaries.append(summary)
        all_rows.extend(repeat_rows)
        write_csv(out_dir / "checkpoint_metrics.csv", all_rows)
        write_csv(out_dir / "repeat_seed_checkpoint_metrics.csv", repeat_rows)
        write_json(out_dir / "run_summaries.json", run_summaries)
        assessment = assess_grokking(all_rows)

    peak = max((float(r["peak_gpu_memory_mb"]) for r in all_rows), default=0.0)
    decision = "Abandon grokking-style long training for this router unless a new signal/objective is introduced."
    if assessment["any_possible_grokking"]:
        decision = "Continue only after confirming the delayed pattern on validation-free repeat seeds, then request approval before any validation evaluation."
    report = {
        "metadata": meta,
        "assessment": assessment,
        "repeat_seeds_run": bool(repeat_rows),
        "compute": {"runtime_sec": time.time() - t0, "peak_gpu_memory_mb": peak},
        "decision": decision,
        "validation_used": False,
        "test_used": False,
        "reproduce_command": f"python experiments\\grokking_diagnostic_costar\\run_grokking_diagnostic.py --device {args.device}",
    }
    write_json(out_dir / "final_report.json", report)
    write_report(out_dir, report)

    write_curve_svg(out_dir, all_rows, "chronological_fold_mae", "learning_curve_fold_mae.svg")
    write_curve_svg(out_dir, all_rows, "training_mae", "learning_curve_training_mae.svg")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
