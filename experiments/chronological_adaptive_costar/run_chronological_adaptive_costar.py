"""Chronologically adaptive COSTAR experiments.

The runner evaluates frozen ETTh1 walk-forward forecasts only.  Online state is
updated strictly after an old prediction's full horizon is observable:

    old_start + horizon <= current_start

No test cache is loaded.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.oracle_weight_tournament.run_tournament import (  # noqa: E402
    FIXED3,
    DYNAMIC_FIXED3_REFERENCE_MAE,
    FIXED3_REFERENCE_MAE,
    Fixed3WindowDataset,
    WeightStudent,
    args_global_weights,
    fixed3_indices,
    load_cache,
    load_dynamic_baseline_per_window,
    load_std,
    sample_mae,
    sample_mse,
    weighted_forecast,
)


CURRENT_WINNER = "final_phase2_protores_lam0.01_k16_scale0.3_rw0.001"
CURRENT_WINNER_MAE = 0.3660282492637634
CURRENT_WINNER_MSE = 0.308755099773407
SEEDS = (7, 11, 13, 17, 19)


@dataclass(frozen=True)
class Trial:
    family: str
    name: str
    decay: float = 0.98
    temperature: float = 0.05
    eta: float = 1.0
    discount: float = 1.0
    window: int = 48
    alpha: float = 0.25
    threshold: float = 0.0
    seed: int = 7
    update_interval: int = 24
    buffer_size: int = 192
    lr: float = 0.02
    reg: float = 10.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fixed3_forecasts(cache: Mapping[str, Any]) -> torch.Tensor:
    return cache["prediction_stack"][..., fixed3_indices(cache)].to(torch.float32)


def per_expert_errors(cache: Mapping[str, Any], std: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    forecasts = fixed3_forecasts(cache)
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.bool)
    maes, mses, biases = [], [], []
    stdv = std.view(1, 1, -1)
    mask_f = mask.to(torch.float32)
    denom = mask_f.flatten(1).sum(dim=1).clamp_min(1.0)
    for i in range(forecasts.shape[-1]):
        pred = forecasts[..., i]
        maes.append(sample_mae(pred, target, mask, std))
        mses.append(sample_mse(pred, target, mask, std))
        bias = (((pred - target) / stdv) * mask_f).flatten(1).sum(dim=1) / denom
        biases.append(bias)
    return torch.stack(maes, dim=1), torch.stack(mses, dim=1), torch.stack(biases, dim=1)


def metrics_from_weights(cache: Mapping[str, Any], std: torch.Tensor, weights: torch.Tensor) -> dict[str, Any]:
    pred = weighted_forecast(fixed3_forecasts(cache), weights)
    mae = sample_mae(pred, cache["targets"].to(torch.float32), cache["target_masks"].to(torch.bool), std)
    mse = sample_mse(pred, cache["targets"].to(torch.float32), cache["target_masks"].to(torch.bool), std)
    return {"mae": float(mae.mean()), "mse": float(mse.mean()), "per_window_mae": mae, "per_window_mse": mse}


def softmax_neg(errors: torch.Tensor, temperature: float) -> torch.Tensor:
    return torch.softmax(-errors / max(float(temperature), 1e-8), dim=-1)


def enforce_observable(due_start: int, current_start: int, horizon: int) -> None:
    if due_start + horizon > current_start:
        raise RuntimeError(f"Leakage: attempted to use start={due_start} at current={current_start} before horizon={horizon} elapsed")


def chronological_online_weights(
    starts: torch.Tensor,
    expert_mae: torch.Tensor,
    horizon: int,
    trial: Trial,
    train_mean_mae: torch.Tensor,
    mode: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    n = expert_mae.shape[0]
    weights = []
    updates = 0
    pending: list[int] = []
    obs_losses: list[torch.Tensor] = []
    ema = train_mean_mae.clone()
    logw = torch.log(torch.tensor(args_global_weights(), dtype=torch.float32))
    for i in range(n):
        now = int(starts[i])
        still_pending: list[int] = []
        for j in pending:
            due = int(starts[j]) + horizon
            if due <= now:
                enforce_observable(int(starts[j]), now, horizon)
                loss = expert_mae[j]
                obs_losses.append(loss)
                updates += 1
                if mode == "ema":
                    ema = float(trial.decay) * ema + (1.0 - float(trial.decay)) * loss
                elif mode == "hedge":
                    logw = float(trial.discount) * logw - float(trial.eta) * loss
                    logw = logw - torch.logsumexp(logw, dim=0)
            else:
                still_pending.append(j)
        pending = still_pending
        if mode == "ema":
            w = softmax_neg(ema, trial.temperature)
        elif mode == "hedge":
            w = torch.softmax(logw, dim=0)
        elif mode == "rolling":
            if obs_losses:
                recent = torch.stack(obs_losses[-int(trial.window) :]).mean(dim=0)
            else:
                recent = train_mean_mae
            w = softmax_neg(recent, trial.temperature)
        else:
            raise ValueError(mode)
        weights.append(w)
        pending.append(i)
    return torch.stack(weights), {"num_updates": updates, "mean_weight": torch.stack(weights).mean(dim=0).tolist()}


def load_static_winner_per_window(seed: int, val_cache: Mapping[str, Any], std: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ckpt_path = ROOT / "experiments/oracle_weight_tournament/checkpoints" / f"{CURRENT_WINNER}_seed{seed}" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    trial = ckpt["trial"]
    model = WeightStudent(
        args_global_weights(),
        int(val_cache["input_len"]),
        int(val_cache["forecast_horizon"]),
        int(val_cache["num_features"]),
        mode="prototype_residual",
        num_prototypes=int(trial["num_prototypes"]),
        residual_scale=float(trial["residual_scale"]),
        feature_mix=trial.get("feature_mix", "full"),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    ds = Fixed3WindowDataset(val_cache, fixed3_indices(val_cache))
    weights, maes, mses = [], [], []
    with torch.no_grad():
        for batch in DataLoader(ds, batch_size=1024, shuffle=False):
            out = model(batch["history"].to(device), batch["forecasts"].to(device), prototypes=ckpt["prototypes"])
            w = out["weights"].detach().cpu()
            pred = weighted_forecast(batch["forecasts"], w)
            weights.append(w)
            maes.append(sample_mae(pred, batch["target"], batch["mask"], std))
            mses.append(sample_mse(pred, batch["target"], batch["mask"], std))
    return torch.cat(weights), torch.cat(maes), torch.cat(mses)


def evaluate_hybrid(
    val_cache: Mapping[str, Any],
    std: torch.Tensor,
    online_weights: torch.Tensor,
    static_weights: torch.Tensor,
    trial: Trial,
    detector: bool,
    expert_mae: torch.Tensor,
    starts: torch.Tensor,
    horizon: int,
    train_mean_mae: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if not detector:
        weights = (1.0 - trial.alpha) * static_weights + trial.alpha * online_weights
        weights = weights.clamp_min(1e-6)
        return weights / weights.sum(dim=1, keepdim=True), {"detected": int(weights.shape[0]), "num_updates": None}
    final = []
    pending: list[int] = []
    obs_losses: list[torch.Tensor] = []
    detected = 0
    static_global_best = int(torch.tensor(args_global_weights()).argmax())
    for i in range(online_weights.shape[0]):
        now = int(starts[i])
        still = []
        for j in pending:
            if int(starts[j]) + horizon <= now:
                enforce_observable(int(starts[j]), now, horizon)
                obs_losses.append(expert_mae[j])
            else:
                still.append(j)
        pending = still
        recent = torch.stack(obs_losses[-int(trial.window) :]).mean(dim=0) if obs_losses else train_mean_mae
        sorted_vals, sorted_idx = torch.sort(recent)
        shifted_rank = int(sorted_idx[0]) != static_global_best
        shifted_gap = float(sorted_vals[1] - sorted_vals[0]) > float(trial.threshold)
        if shifted_rank and shifted_gap:
            w = (1.0 - trial.alpha) * static_weights[i] + trial.alpha * online_weights[i]
            detected += 1
        else:
            w = static_weights[i]
        final.append(w / w.sum().clamp_min(1e-8))
        pending.append(i)
    return torch.stack(final), {"detected": detected, "num_updates": len(obs_losses)}


def evaluate_online_ft_delta(
    val_cache: Mapping[str, Any],
    std: torch.Tensor,
    static_weights: torch.Tensor,
    trial: Trial,
    starts: torch.Tensor,
    horizon: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    forecasts = fixed3_forecasts(val_cache)
    target = val_cache["targets"].to(torch.float32)
    mask = val_cache["target_masks"].to(torch.bool)
    delta = torch.zeros(3, requires_grad=True)
    optimizer = torch.optim.SGD([delta], lr=float(trial.lr))
    pending: list[int] = []
    buffer: deque[int] = deque(maxlen=int(trial.buffer_size))
    weights = []
    updates = 0
    for i in range(forecasts.shape[0]):
        now = int(starts[i])
        still = []
        for j in pending:
            if int(starts[j]) + horizon <= now:
                enforce_observable(int(starts[j]), now, horizon)
                buffer.append(j)
            else:
                still.append(j)
        pending = still
        if buffer and i % int(trial.update_interval) == 0:
            idx = torch.tensor(list(buffer), dtype=torch.long)
            for _ in range(2):
                base = static_weights[idx].clamp_min(1e-8)
                w = torch.softmax(base.log() + delta.view(1, 3), dim=1)
                pred = weighted_forecast(forecasts[idx], w)
                loss = sample_mae(pred, target[idx], mask[idx], std).mean() + float(trial.reg) * delta.square().mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            updates += 1
        with torch.no_grad():
            w_now = torch.softmax(static_weights[i].clamp_min(1e-8).log() + delta, dim=0)
        weights.append(w_now)
        pending.append(i)
    return torch.stack(weights), {"num_updates": updates, "final_delta": delta.detach().tolist()}


def paired_bootstrap(candidate: torch.Tensor, baseline: torch.Tensor, seed: int = 777, samples: int = 5000) -> dict[str, Any]:
    diff = candidate - baseline
    gen = torch.Generator().manual_seed(seed)
    vals = []
    n = diff.numel()
    for _ in range(samples):
        idx = torch.randint(0, n, (n,), generator=gen)
        vals.append(float(diff[idx].mean()))
    t = torch.tensor(vals)
    return {
        "mean_diff_candidate_minus_baseline": float(diff.mean()),
        "ci95_low": float(torch.quantile(t, 0.025)),
        "ci95_high": float(torch.quantile(t, 0.975)),
        "ci_excludes_zero": bool(torch.quantile(t, 0.975) < 0 or torch.quantile(t, 0.025) > 0),
    }


def block_diagnostics(
    val_cache: Mapping[str, Any],
    std: torch.Tensor,
    weights: torch.Tensor,
    expert_mae: torch.Tensor,
    block_count: int = 6,
) -> list[dict[str, Any]]:
    n = expert_mae.shape[0]
    equal = torch.full((n, 3), 1.0 / 3.0)
    equal_mae = metrics_from_weights(val_cache, std, equal)["per_window_mae"]
    adaptive_mae = metrics_from_weights(val_cache, std, weights)["per_window_mae"]
    rows = []
    prev_best = None
    for b in range(block_count):
        lo = b * n // block_count
        hi = (b + 1) * n // block_count
        indiv = expert_mae[lo:hi].mean(dim=0)
        best_idx = int(indiv.argmin())
        changed = prev_best is not None and best_idx != prev_best
        prev_best = best_idx
        rows.append(
            {
                "block": b,
                "start_index": lo,
                "end_index": hi - 1,
                "start_time": int(val_cache["absolute_window_starts"][lo]),
                "end_time": int(val_cache["absolute_window_starts"][hi - 1]),
                "best_expert": FIXED3[best_idx],
                "best_changed_from_previous": changed,
                **{f"{FIXED3[i]}_mae": float(indiv[i]) for i in range(3)},
                "equal_fixed3_mae": float(equal_mae[lo:hi].mean()),
                "adaptive_mae": float(adaptive_mae[lo:hi].mean()),
                "adaptive_minus_equal": float((adaptive_mae[lo:hi] - equal_mae[lo:hi]).mean()),
                **{f"mean_weight_{FIXED3[i]}": float(weights[lo:hi, i].mean()) for i in range(3)},
            }
        )
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def trial_grid() -> list[Trial]:
    trials = []
    for decay in (0.90, 0.95, 0.97, 0.98, 0.99, 0.995):
        for temp in (0.01, 0.02, 0.05, 0.10, 0.20, 0.50):
            trials.append(Trial("ema", f"ema_decay{decay}_temp{temp}", decay=decay, temperature=temp))
    for eta in (0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0):
        for discount in (1.0, 0.995, 0.99, 0.98, 0.95):
            trials.append(Trial("hedge", f"hedge_eta{eta}_discount{discount}", eta=eta, discount=discount))
    for window in (12, 24, 48, 96, 192):
        for temp in (0.01, 0.02, 0.05, 0.10, 0.20):
            trials.append(Trial("rolling", f"rolling_w{window}_temp{temp}", window=window, temperature=temp))
    return trials


def finalist_grid(best_online: Sequence[Trial]) -> list[Trial]:
    trials = []
    for base in best_online:
        for alpha in (0.05, 0.10, 0.20, 0.35, 0.50):
            for seed in SEEDS:
                trials.append(Trial("hybrid", f"hybrid_{base.name}_alpha{alpha}_seed{seed}", alpha=alpha, seed=seed, **{k: v for k, v in asdict(base).items() if k not in {"family", "name", "alpha", "seed"}}))
        for alpha in (0.10, 0.20, 0.35):
            for threshold in (0.0, 0.005, 0.01, 0.02):
                for seed in SEEDS:
                    trials.append(Trial("detector_hybrid", f"detector_{base.name}_alpha{alpha}_thr{threshold}_seed{seed}", alpha=alpha, threshold=threshold, seed=seed, **{k: v for k, v in asdict(base).items() if k not in {"family", "name", "alpha", "threshold", "seed"}}))
    for interval in (12, 24, 48, 96):
        for lr in (0.005, 0.01, 0.02):
            for seed in SEEDS:
                trials.append(Trial("online_ft_delta", f"online_ft_delta_interval{interval}_lr{lr}_seed{seed}", update_interval=interval, lr=lr, seed=seed))
    return trials


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", default="cache/costarts_walkforward/router_train_20_60_cache.pt")
    parser.add_argument("--val-cache", default="cache/costarts_walkforward/router_val_60_80_cache.pt")
    parser.add_argument("--normalizer-checkpoint", default="checkpoints/costarts_walkforward/final_60/DLinear/best_expert.pt")
    parser.add_argument("--out-dir", default="experiments/chronological_adaptive_costar")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--phase", choices=("all", "screen", "finalists"), default="all")
    parser.add_argument("--top-online", type=int, default=4)
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_window").mkdir(exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)
    train_cache = load_cache(ROOT / args.train_cache, "router_train_20_60")
    val_cache = load_cache(ROOT / args.val_cache, "router_val_60_80")
    if "test" in str(args.train_cache).lower() or "test" in str(args.val_cache).lower():
        raise ValueError("Refusing test cache path")
    starts = val_cache["absolute_window_starts"].to(torch.long)
    if not bool(torch.all(starts[1:] > starts[:-1])):
        raise ValueError("Validation starts are not strictly chronological")
    horizon = int(val_cache["forecast_horizon"])
    std = load_std(ROOT / args.normalizer_checkpoint, int(val_cache["num_features"]))
    device = torch.device(args.device)

    train_expert_mae, _, _ = per_expert_errors(train_cache, std)
    expert_mae, _, _ = per_expert_errors(val_cache, std)
    train_mean_mae = train_expert_mae.mean(dim=0)
    equal_metrics = metrics_from_weights(val_cache, std, torch.full((int(val_cache["num_windows"]), 3), 1.0 / 3.0))
    global_metrics = metrics_from_weights(val_cache, std, torch.tensor(args_global_weights()).view(1, 3).expand(int(val_cache["num_windows"]), -1))

    rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    static_cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def current_baseline_for_seed(seed: int) -> torch.Tensor:
        if seed not in static_cache:
            static_cache[seed] = load_static_winner_per_window(seed, val_cache, std, device)
        return static_cache[seed][1]

    def static_weights_for_seed(seed: int) -> torch.Tensor:
        if seed not in static_cache:
            static_cache[seed] = load_static_winner_per_window(seed, val_cache, std, device)
        return static_cache[seed][0]

    def record(trial: Trial, weights: torch.Tensor, extra: Mapping[str, Any]) -> dict[str, Any]:
        metrics = metrics_from_weights(val_cache, std, weights)
        torch.save(metrics["per_window_mae"], out_dir / "per_window" / f"{trial.name}.pt")
        baseline = current_baseline_for_seed(trial.seed)
        boot = paired_bootstrap(metrics["per_window_mae"], baseline, seed=trial.seed, samples=3000)
        dyn = load_dynamic_baseline_per_window(trial.seed)
        row = asdict(trial) | {
            "mae": metrics["mae"],
            "mse": metrics["mse"],
            "diff_vs_current_winner_seed": metrics["mae"] - float(baseline.mean()),
            "diff_vs_current_winner_mean": metrics["mae"] - CURRENT_WINNER_MAE,
            "diff_vs_dynamic_fixed3_mean": metrics["mae"] - DYNAMIC_FIXED3_REFERENCE_MAE,
            "diff_vs_equal_fixed3": metrics["mae"] - FIXED3_REFERENCE_MAE,
            "mean_weight_patchtst": float(weights[:, 0].mean()),
            "mean_weight_itransformer": float(weights[:, 1].mean()),
            "mean_weight_timesnet": float(weights[:, 2].mean()),
            **{f"current_{k}": v for k, v in boot.items()},
            **extra,
        }
        if dyn is not None:
            row["diff_vs_dynamic_seed"] = metrics["mae"] - float(dyn.mean())
        rows.append(row)
        block_rows.extend([{"trial": trial.name, **r} for r in block_diagnostics(val_cache, std, weights, expert_mae)])
        write_csv(out_dir / "all_trials.csv", rows)
        write_csv(out_dir / "block_diagnostics.csv", block_rows)
        write_csv(out_dir / "leaderboard.csv", sorted(rows, key=lambda r: float(r["mae"])))
        return row

    screen_trials = trial_grid()
    if args.phase in {"all", "screen"}:
        for trial in screen_trials:
            if trial.family in {"ema", "hedge", "rolling"}:
                weights, extra = chronological_online_weights(starts, expert_mae, horizon, trial, train_mean_mae, trial.family)
                record(trial, weights, extra)

    leaderboard = sorted(rows, key=lambda r: float(r["mae"]))
    if args.phase == "finalists":
        if not (out_dir / "leaderboard.csv").exists():
            raise FileNotFoundError("Run --phase screen first or use --phase all")
        prior = list(csv.DictReader((out_dir / "leaderboard.csv").open(newline="", encoding="utf-8")))
        name_to_trial = {t.name: t for t in screen_trials}
        best_online = [name_to_trial[r["name"]] for r in prior if r["name"] in name_to_trial][: args.top_online]
    else:
        name_to_trial = {t.name: t for t in screen_trials}
        best_online = [name_to_trial[r["name"]] for r in leaderboard if r["name"] in name_to_trial][: args.top_online]

    if args.phase in {"all", "finalists"}:
        for trial in finalist_grid(best_online):
            set_seed(trial.seed)
            if trial.family in {"hybrid", "detector_hybrid"}:
                source_name = trial.name.split("_alpha")[0].replace("hybrid_", "").replace("detector_", "")
                source = name_to_trial[source_name]
                online_weights, online_extra = chronological_online_weights(starts, expert_mae, horizon, source, train_mean_mae, source.family)
                weights, extra = evaluate_hybrid(
                    val_cache,
                    std,
                    online_weights,
                    static_weights_for_seed(trial.seed),
                    trial,
                    detector=trial.family == "detector_hybrid",
                    expert_mae=expert_mae,
                    starts=starts,
                    horizon=horizon,
                    train_mean_mae=train_mean_mae,
                )
                record(trial, weights, {**online_extra, **extra, "source_online": source.name})
            elif trial.family == "online_ft_delta":
                weights, extra = evaluate_online_ft_delta(val_cache, std, static_weights_for_seed(trial.seed), trial, starts, horizon)
                record(trial, weights, extra)

    leaderboard = sorted(rows, key=lambda r: float(r["mae"]))
    # Aggregate five-seed finalist groups.
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["family"] in {"hybrid", "detector_hybrid", "online_ft_delta"}:
            base = row["name"].rsplit("_seed", 1)[0]
            groups.setdefault(base, []).append(row)
    finalist_summary = []
    for name, group in groups.items():
        if len(group) < len(SEEDS):
            continue
        maes = torch.tensor([float(r["mae"]) for r in group])
        mses = torch.tensor([float(r["mse"]) for r in group])
        diffs = torch.tensor([float(r["diff_vs_current_winner_seed"]) for r in group])
        cand = torch.cat([torch.load(out_dir / "per_window" / f"{r['name']}.pt", map_location="cpu", weights_only=False) for r in group])
        base = torch.cat([current_baseline_for_seed(int(r["seed"])) for r in group])
        finalist_summary.append(
            {
                "name": name,
                "family": group[0]["family"],
                "mae_mean": float(maes.mean()),
                "mae_std": float(maes.std(unbiased=False)),
                "mse_mean": float(mses.mean()),
                "mse_std": float(mses.std(unbiased=False)),
                "diff_vs_current_winner_mean": float(diffs.mean()),
                "wins_vs_current_winner": int(sum(float(r["diff_vs_current_winner_seed"]) < 0 for r in group)),
                **paired_bootstrap(cand, base, seed=777, samples=5000),
            }
        )
    finalist_summary = sorted(finalist_summary, key=lambda r: float(r["mae_mean"]))
    write_csv(out_dir / "finalist_summary.csv", finalist_summary)

    expert_block_rows = block_diagnostics(val_cache, std, torch.full((int(val_cache["num_windows"]), 3), 1.0 / 3.0), expert_mae)
    write_csv(out_dir / "expert_ranking_blocks.csv", expert_block_rows)
    ranking_changes = sum(1 for row in expert_block_rows if row["best_changed_from_previous"])

    report = {
        "best_single_trial": leaderboard[0] if leaderboard else None,
        "best_five_seed_finalist": finalist_summary[0] if finalist_summary else None,
        "current_winner_reference": {"mae": CURRENT_WINNER_MAE, "mse": CURRENT_WINNER_MSE},
        "dynamic_fixed3_reference": {"mae": DYNAMIC_FIXED3_REFERENCE_MAE},
        "equal_fixed3": {"mae": equal_metrics["mae"], "mse": equal_metrics["mse"]},
        "global_weighted_fixed3": {"mae": global_metrics["mae"], "mse": global_metrics["mse"], "weights": args_global_weights()},
        "expert_ranking_block_changes": ranking_changes,
        "expert_ranking_blocks": expert_block_rows,
        "safety": "NO TEST DATA USED; validation labels only enter online state after start+horizon <= current_start",
    }
    (out_dir / "final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    if leaderboard:
        best = leaderboard[0]
        best_mae = torch.load(out_dir / "per_window" / f"{best['name']}.pt", map_location="cpu", weights_only=False)
        static = current_baseline_for_seed(int(best["seed"]))
        win = 192
        kernel = torch.ones(win) / win
        y1 = F.conv1d(best_mae.view(1, 1, -1), kernel.view(1, 1, -1), padding=win // 2).view(-1)[: best_mae.numel()]
        y2 = F.conv1d(static.view(1, 1, -1), kernel.view(1, 1, -1), padding=win // 2).view(-1)[: static.numel()]
        write_csv(
            out_dir / "plots" / "rolling_mae_best.csv",
            [
                {"index": i, "candidate_rolling_mae": float(y1[i]), "current_winner_rolling_mae": float(y2[i])}
                for i in range(best_mae.numel())
            ],
        )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
