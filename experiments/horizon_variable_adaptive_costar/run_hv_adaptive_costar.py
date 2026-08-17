"""Horizon/variable chronological COSTAR adaptation.

This runner tests two cheap high-signal extensions of the current best
chronological COSTAR validation method:

1. Causal horizon x variable expert weighting, with low-rank residual logits.
2. Causal recursive least-squares stacking with intercepts.

Validation labels are only consumed after the full forecast horizon is
observable.  No test cache is loaded.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.chronological_adaptive_costar.run_chronological_adaptive_costar import (  # noqa: E402
    CURRENT_WINNER,
    CURRENT_WINNER_MAE,
    CURRENT_WINNER_MSE,
    SEEDS,
    enforce_observable,
    load_static_winner_per_window,
    paired_bootstrap,
)
from experiments.oracle_weight_tournament.run_tournament import (  # noqa: E402
    FIXED3,
    DYNAMIC_FIXED3_REFERENCE_MAE,
    FIXED3_REFERENCE_MAE,
    args_global_weights,
    fixed3_indices,
    load_cache,
    load_std,
    sample_mae,
    sample_mse,
    weighted_forecast,
)


@dataclass(frozen=True)
class Trial:
    family: str
    name: str
    mode: str = "hv"
    rank: int = 2
    decay: float = 0.97
    temperature: float = 0.1
    alpha: float = 0.5
    seed: int = 7
    forgetting: float = 0.995
    ridge: float = 10.0
    shrink: float = 0.0
    constrained: bool = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def fixed3_forecasts(cache: Mapping[str, Any]) -> torch.Tensor:
    return cache["prediction_stack"][..., fixed3_indices(cache)].to(torch.float32)


def predict_from_hv_weights(forecasts: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (forecasts * weights).sum(dim=-1)


def metrics_from_prediction(cache: Mapping[str, Any], std: torch.Tensor, pred: torch.Tensor) -> dict[str, Any]:
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.bool)
    mae = sample_mae(pred, target, mask, std)
    mse = sample_mse(pred, target, mask, std)
    return {"mae": float(mae.mean()), "mse": float(mse.mean()), "per_window_mae": mae, "per_window_mse": mse}


def per_location_abs_error(cache: Mapping[str, Any], std: torch.Tensor) -> torch.Tensor:
    forecasts = fixed3_forecasts(cache)
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.float32)
    stdv = std.view(1, 1, -1, 1)
    return ((forecasts - target.unsqueeze(-1)) / stdv).abs() * mask.unsqueeze(-1)


def low_rank_matrix(x: torch.Tensor, rank: int) -> torch.Tensor:
    # x shape: [H, V, E]. Approximate each expert's HxV error surface.
    if rank <= 0:
        return x.mean(dim=(0, 1), keepdim=True).expand_as(x)
    pieces = []
    for e in range(x.shape[-1]):
        mat = x[..., e]
        u, s, vh = torch.linalg.svd(mat, full_matrices=False)
        r = min(int(rank), s.numel())
        pieces.append((u[:, :r] * s[:r]) @ vh[:r])
    return torch.stack(pieces, dim=-1)


def aggregate_error(err_hve: torch.Tensor, mode: str, rank: int) -> torch.Tensor:
    if mode == "global":
        return err_hve.mean(dim=(0, 1), keepdim=True).expand_as(err_hve)
    if mode == "horizon":
        return err_hve.mean(dim=1, keepdim=True).expand_as(err_hve)
    if mode == "variable":
        return err_hve.mean(dim=0, keepdim=True).expand_as(err_hve)
    if mode == "hv":
        return err_hve
    if mode == "hv_lowrank":
        return low_rank_matrix(err_hve, rank)
    raise ValueError(mode)


def errors_to_weights(err_hve: torch.Tensor, trial: Trial) -> torch.Tensor:
    err = aggregate_error(err_hve, trial.mode, trial.rank)
    centered = err - err.mean(dim=-1, keepdim=True)
    base = torch.tensor(args_global_weights(), dtype=torch.float32).log().view(1, 1, 3)
    logits = base - centered / max(float(trial.temperature), 1e-8)
    return torch.softmax(logits, dim=-1)


def chronological_hv_weights(
    starts: torch.Tensor,
    train_err_mean: torch.Tensor,
    val_err: torch.Tensor,
    horizon: int,
    trial: Trial,
) -> tuple[torch.Tensor, dict[str, Any]]:
    pending: list[int] = []
    ema = train_err_mean.clone()
    weights = []
    updates = 0
    for i in range(val_err.shape[0]):
        now = int(starts[i])
        still = []
        for j in pending:
            if int(starts[j]) + horizon <= now:
                enforce_observable(int(starts[j]), now, horizon)
                ema = float(trial.decay) * ema + (1.0 - float(trial.decay)) * val_err[j]
                updates += 1
            else:
                still.append(j)
        pending = still
        weights.append(errors_to_weights(ema, trial))
        pending.append(i)
    w = torch.stack(weights)
    return w, {"num_updates": updates, **{f"mean_weight_{FIXED3[e]}": float(w[..., e].mean()) for e in range(3)}}


def rls_predict_update(
    starts: torch.Tensor,
    forecasts: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    horizon: int,
    trial: Trial,
) -> tuple[torch.Tensor, dict[str, Any]]:
    n, h, v, e = forecasts.shape
    base_beta = torch.tensor([0.0, *args_global_weights()], dtype=torch.float32)
    if trial.mode == "global":
        beta = base_beta.view(1, 1, 4).repeat(1, 1, 1)
        pmat = torch.eye(4).view(1, 1, 4, 4).repeat(1, 1, 1, 1) / float(trial.ridge)
    elif trial.mode == "horizon":
        beta = base_beta.view(1, 1, 4).repeat(h, 1, 1)
        pmat = torch.eye(4).view(1, 1, 4, 4).repeat(h, 1, 1, 1) / float(trial.ridge)
    elif trial.mode == "variable":
        beta = base_beta.view(1, 1, 4).repeat(1, v, 1)
        pmat = torch.eye(4).view(1, 1, 4, 4).repeat(1, v, 1, 1) / float(trial.ridge)
    elif trial.mode == "hv":
        beta = base_beta.view(1, 1, 4).repeat(h, v, 1)
        pmat = torch.eye(4).view(1, 1, 4, 4).repeat(h, v, 1, 1) / float(trial.ridge)
    else:
        raise ValueError(trial.mode)

    def idx(hh: int, vv: int) -> tuple[int, int]:
        if trial.mode == "global":
            return 0, 0
        if trial.mode == "horizon":
            return hh, 0
        if trial.mode == "variable":
            return 0, vv
        return hh, vv

    preds = []
    pending: list[int] = []
    updates = 0
    lam = float(trial.forgetting)
    for i in range(n):
        now = int(starts[i])
        still = []
        for j in pending:
            if int(starts[j]) + horizon <= now:
                enforce_observable(int(starts[j]), now, horizon)
                for hh in range(h):
                    for vv in range(v):
                        if not bool(mask[j, hh, vv]):
                            continue
                        ih, iv = idx(hh, vv)
                        x = torch.tensor([1.0, *forecasts[j, hh, vv].tolist()], dtype=torch.float32)
                        y = target[j, hh, vv]
                        p = pmat[ih, iv]
                        px = p @ x
                        gain = px / (lam + x @ px)
                        err = y - x @ beta[ih, iv]
                        beta[ih, iv] = beta[ih, iv] + gain * err
                        if trial.constrained:
                            beta[ih, iv, 1:] = beta[ih, iv, 1:].clamp_min(0)
                            s = beta[ih, iv, 1:].sum().clamp_min(1e-8)
                            beta[ih, iv, 1:] = beta[ih, iv, 1:] / s
                        if trial.shrink > 0:
                            beta[ih, iv] = (1.0 - trial.shrink) * beta[ih, iv] + trial.shrink * base_beta
                        pmat[ih, iv] = (p - torch.outer(gain, x) @ p) / lam
                updates += 1
            else:
                still.append(j)
        pending = still
        pred_i = torch.empty((h, v), dtype=torch.float32)
        for hh in range(h):
            for vv in range(v):
                ih, iv = idx(hh, vv)
                x = torch.tensor([1.0, *forecasts[i, hh, vv].tolist()], dtype=torch.float32)
                pred_i[hh, vv] = x @ beta[ih, iv]
        preds.append(pred_i)
        pending.append(i)
    return torch.stack(preds), {"num_updates": updates, "coef_abs_mean": float(beta.abs().mean())}


def load_current_prediction(seed: int, val_cache: Mapping[str, Any], std: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    weights, mae, mse = load_static_winner_per_window(seed, val_cache, std, device)
    pred = weighted_forecast(fixed3_forecasts(val_cache), weights)
    return pred, weights, mae, mse


def block_specialization(cache: Mapping[str, Any], std: torch.Tensor, pred: torch.Tensor, weights_hve: torch.Tensor | None, rows_prefix: Mapping[str, Any]) -> list[dict[str, Any]]:
    forecasts = fixed3_forecasts(cache)
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.bool)
    stdv = std.view(1, 1, -1)
    abs_err = ((pred - target) / stdv).abs() * mask.to(torch.float32)
    expert_err = per_location_abs_error(cache, std)
    n, h, v, e = forecasts.shape
    rows = []
    for hh in range(h):
        rows.append({**rows_prefix, "axis": "horizon", "index": hh, "mae": float(abs_err[:, hh].sum() / mask[:, hh].sum().clamp_min(1)), **{f"{FIXED3[k]}_mae": float(expert_err[:, hh, :, k].sum() / mask[:, hh].sum().clamp_min(1)) for k in range(e)}})
    for vv in range(v):
        rows.append({**rows_prefix, "axis": "variable", "index": vv, "mae": float(abs_err[:, :, vv].sum() / mask[:, :, vv].sum().clamp_min(1)), **{f"{FIXED3[k]}_mae": float(expert_err[:, :, vv, k].sum() / mask[:, :, vv].sum().clamp_min(1)) for k in range(e)}})
    if weights_hve is not None:
        w_mean = weights_hve.mean(dim=0)
        for hh in range(h):
            rows.append({**rows_prefix, "axis": "horizon_weight", "index": hh, **{f"w_{FIXED3[k]}": float(w_mean[hh, :, k].mean()) for k in range(e)}})
        for vv in range(v):
            rows.append({**rows_prefix, "axis": "variable_weight", "index": vv, **{f"w_{FIXED3[k]}": float(w_mean[:, vv, k].mean()) for k in range(e)}})
    return rows


def screen_grid() -> list[Trial]:
    trials: list[Trial] = []
    for mode in ("global", "horizon", "variable", "hv"):
        for decay in (0.90, 0.95, 0.97, 0.98, 0.99, 0.995):
            for temp in (0.02, 0.05, 0.10, 0.20):
                trials.append(Trial("hv_ema", f"hvema_{mode}_decay{decay}_temp{temp}", mode=mode, decay=decay, temperature=temp))
    for rank in (1, 2, 4, 8):
        for decay in (0.95, 0.97, 0.98, 0.99):
            for temp in (0.05, 0.10, 0.20):
                trials.append(Trial("hv_ema", f"hvema_lowrank{rank}_decay{decay}_temp{temp}", mode="hv_lowrank", rank=rank, decay=decay, temperature=temp))
    for mode in ("global", "horizon", "variable", "hv"):
        for forgetting in (0.90, 0.95, 0.98, 0.99, 0.995, 0.999):
            for ridge in (0.1, 1.0, 10.0, 100.0):
                trials.append(Trial("rls", f"rls_{mode}_forget{forgetting}_ridge{ridge}", mode=mode, forgetting=forgetting, ridge=ridge))
                trials.append(Trial("rls", f"rls_{mode}_forget{forgetting}_ridge{ridge}_constrained", mode=mode, forgetting=forgetting, ridge=ridge, constrained=True))
    return trials


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", default="cache/costarts_walkforward/router_train_20_60_cache.pt")
    parser.add_argument("--val-cache", default="cache/costarts_walkforward/router_val_60_80_cache.pt")
    parser.add_argument("--normalizer-checkpoint", default="checkpoints/costarts_walkforward/final_60/DLinear/best_expert.pt")
    parser.add_argument("--out-dir", default="experiments/horizon_variable_adaptive_costar")
    parser.add_argument("--phase", choices=("screen", "finalists", "all"), default="all")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_window").mkdir(exist_ok=True)
    train_cache = load_cache(ROOT / args.train_cache, "router_train_20_60")
    val_cache = load_cache(ROOT / args.val_cache, "router_val_60_80")
    if "test" in str(args.train_cache).lower() or "test" in str(args.val_cache).lower():
        raise ValueError("Refusing test cache")
    starts = val_cache["absolute_window_starts"].to(torch.long)
    if not bool(torch.all(starts[1:] > starts[:-1])):
        raise ValueError("Validation windows are not chronological")
    horizon = int(val_cache["forecast_horizon"])
    std = load_std(ROOT / args.normalizer_checkpoint, int(val_cache["num_features"]))
    device = torch.device(args.device)
    forecasts = fixed3_forecasts(val_cache)
    train_err = per_location_abs_error(train_cache, std)
    val_err = per_location_abs_error(val_cache, std)
    train_err_mean = train_err.mean(dim=0)

    rows: list[dict[str, Any]] = []
    spec_rows: list[dict[str, Any]] = []
    static_cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def current(seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if seed not in static_cache:
            static_cache[seed] = load_current_prediction(seed, val_cache, std, device)
        return static_cache[seed]

    def record(trial: Trial, pred: torch.Tensor, weights_hve: torch.Tensor | None, extra: Mapping[str, Any]) -> dict[str, Any]:
        metrics = metrics_from_prediction(val_cache, std, pred)
        torch.save(metrics["per_window_mae"], out_dir / "per_window" / f"{trial.name}.pt")
        base_pred, _, base_mae, _ = current(trial.seed)
        boot = paired_bootstrap(metrics["per_window_mae"], base_mae, seed=trial.seed, samples=3000)
        row = asdict(trial) | {
            "mae": metrics["mae"],
            "mse": metrics["mse"],
            "diff_vs_current_winner_seed": metrics["mae"] - float(base_mae.mean()),
            "diff_vs_current_winner_mean": metrics["mae"] - CURRENT_WINNER_MAE,
            "diff_vs_dynamic_fixed3_mean": metrics["mae"] - DYNAMIC_FIXED3_REFERENCE_MAE,
            "diff_vs_equal_fixed3": metrics["mae"] - FIXED3_REFERENCE_MAE,
            **{f"current_{k}": v for k, v in boot.items()},
            **extra,
        }
        rows.append(row)
        spec_rows.extend(block_specialization(val_cache, std, pred, weights_hve, {"trial": trial.name}))
        write_csv(out_dir / "all_trials.csv", rows)
        write_csv(out_dir / "leaderboard.csv", sorted(rows, key=lambda r: float(r["mae"])))
        write_csv(out_dir / "specialization.csv", spec_rows)
        return row

    trials = screen_grid()
    if args.phase in {"screen", "all"}:
        for trial in trials:
            set_seed(trial.seed)
            if trial.family == "hv_ema":
                weights, extra = chronological_hv_weights(starts, train_err_mean, val_err, horizon, trial)
                pred = predict_from_hv_weights(forecasts, weights)
                record(trial, pred, weights, extra)
            elif trial.family == "rls":
                pred, extra = rls_predict_update(starts, forecasts, val_cache["targets"].to(torch.float32), val_cache["target_masks"].to(torch.bool), horizon, trial)
                record(trial, pred, None, extra)

    leaderboard = sorted(rows, key=lambda r: float(r["mae"]))
    if args.phase == "finalists":
        prior = list(csv.DictReader((out_dir / "leaderboard.csv").open(newline="", encoding="utf-8")))
        name_to_trial = {t.name: t for t in trials}
        finalists = [name_to_trial[r["name"]] for r in prior if r["name"] in name_to_trial][: args.top_k]
    else:
        name_to_trial = {t.name: t for t in trials}
        finalists = [name_to_trial[r["name"]] for r in leaderboard if r["name"] in name_to_trial][: args.top_k]

    if args.phase in {"finalists", "all"}:
        for base in finalists:
            for alpha in (0.25, 0.5, 0.75, 1.0):
                for seed in SEEDS:
                    trial = Trial(**(asdict(base) | {"family": f"{base.family}_hybrid", "name": f"hybrid_{base.name}_alpha{alpha}_seed{seed}", "alpha": alpha, "seed": seed}))
                    base_pred, _, _, _ = current(seed)
                    if base.family == "hv_ema":
                        weights, extra = chronological_hv_weights(starts, train_err_mean, val_err, horizon, base)
                        adaptive_pred = predict_from_hv_weights(forecasts, weights)
                        pred = (1.0 - alpha) * base_pred + alpha * adaptive_pred
                        record(trial, pred, weights, extra | {"source_trial": base.name})
                    elif base.family == "rls":
                        adaptive_pred, extra = rls_predict_update(starts, forecasts, val_cache["targets"].to(torch.float32), val_cache["target_masks"].to(torch.bool), horizon, base)
                        pred = (1.0 - alpha) * base_pred + alpha * adaptive_pred
                        record(trial, pred, None, extra | {"source_trial": base.name})

    leaderboard = sorted(rows, key=lambda r: float(r["mae"]))
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["name"].startswith("hybrid_") and "_seed" in row["name"]:
            groups.setdefault(row["name"].rsplit("_seed", 1)[0], []).append(row)
    summaries = []
    for name, group in groups.items():
        if len(group) < len(SEEDS):
            continue
        maes = torch.tensor([float(r["mae"]) for r in group])
        mses = torch.tensor([float(r["mse"]) for r in group])
        diffs = torch.tensor([float(r["diff_vs_current_winner_seed"]) for r in group])
        cand = torch.cat([torch.load(out_dir / "per_window" / f"{r['name']}.pt", map_location="cpu", weights_only=False) for r in group])
        base = torch.cat([current(int(r["seed"]))[2] for r in group])
        summaries.append(
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
    summaries = sorted(summaries, key=lambda r: float(r["mae_mean"]))
    write_csv(out_dir / "finalist_summary.csv", summaries)

    report = {
        "best_single_trial": leaderboard[0] if leaderboard else None,
        "best_five_seed_finalist": summaries[0] if summaries else None,
        "current_winner_reference": {"mae": CURRENT_WINNER_MAE, "mse": CURRENT_WINNER_MSE, "name": CURRENT_WINNER},
        "dynamic_fixed3_reference": {"mae": DYNAMIC_FIXED3_REFERENCE_MAE},
        "target_mae": 0.3619,
        "num_trials": len(rows),
        "safety": "NO TEST DATA USED; validation labels enter adaptation only after start+horizon <= current_start",
    }
    (out_dir / "final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
