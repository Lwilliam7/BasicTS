"""Regime-change adaptive forgetting diagnostic for COSTAR-TS.

The runner changes only the causal horizon-variable EMA update speed used by
the fixed-three COSTAR baseline.  It selects detector settings on chronological
router-train folds, freezes the selected setting, and evaluates validation once.
Test caches are refused.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.chronological_adaptive_costar.run_chronological_adaptive_costar import (  # noqa: E402
    SEEDS,
    Trial as ChronoTrial,
    chronological_online_weights,
    enforce_observable,
    load_static_winner_per_window,
    paired_bootstrap,
)
from experiments.horizon_variable_adaptive_costar.run_hv_adaptive_costar import (  # noqa: E402
    Trial as HvTrial,
    errors_to_weights,
    fixed3_forecasts,
    per_location_abs_error,
    predict_from_hv_weights,
)
from experiments.oracle_weight_tournament.run_tournament import (  # noqa: E402
    FIXED3,
    load_cache,
    load_std,
    sample_mae,
    sample_mse,
    weighted_forecast,
)
from experiments.residual_correction_costar.run_residual_correction_experiments import (  # noqa: E402
    BASELINE_MAE,
    BASELINE_MSE,
    fixed_current_best_prediction,
)


BASELINE_NAME = "hybrid_chrono_hvema_lowrank1_decay0.95_temp0.1_alpha0.75"
STRONG_TARGET = 0.3619


@dataclass(frozen=True)
class RegimeConfig:
    detector: str
    slow_decay: float
    fast_decay: float
    threshold: float
    delta: float
    reset_strength: float
    cooldown: int
    boost_duration: int

    @property
    def name(self) -> str:
        return (
            f"{self.detector}_slow{self.slow_decay:g}_fast{self.fast_decay:g}"
            f"_thr{self.threshold:g}_delta{self.delta:g}_reset{self.reset_strength:g}"
            f"_cool{self.cooldown}_boost{self.boost_duration}"
        )


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


def metrics(cache: Mapping[str, Any], std: torch.Tensor, pred: torch.Tensor) -> dict[str, Any]:
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.bool)
    mae = sample_mae(pred, target, mask, std)
    mse = sample_mse(pred, target, mask, std)
    return {"mae": float(mae.mean()), "mse": float(mse.mean()), "per_window_mae": mae, "per_window_mse": mse}


def normalized_abs_error(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return ((pred - target) / std.view(1, 1, -1)).abs() * mask.to(torch.float32)


def per_axis_rows(cache: Mapping[str, Any], std: torch.Tensor, pred: torch.Tensor, base: torch.Tensor, method: str) -> list[dict[str, Any]]:
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.float32)
    cand = normalized_abs_error(pred, target, mask.to(torch.bool), std)
    ref = normalized_abs_error(base, target, mask.to(torch.bool), std)
    rows: list[dict[str, Any]] = []
    for h in range(cand.shape[1]):
        denom = mask[:, h].sum().clamp_min(1)
        mae = float(cand[:, h].sum() / denom)
        bmae = float(ref[:, h].sum() / denom)
        rows.append({"method": method, "axis": "horizon", "index": h, "mae": mae, "baseline_mae": bmae, "delta_vs_baseline": mae - bmae})
    for v in range(cand.shape[2]):
        denom = mask[:, :, v].sum().clamp_min(1)
        mae = float(cand[:, :, v].sum() / denom)
        bmae = float(ref[:, :, v].sum() / denom)
        rows.append({"method": method, "axis": "variable", "index": v, "mae": mae, "baseline_mae": bmae, "delta_vs_baseline": mae - bmae})
    return rows


def per_hv_rows(cache: Mapping[str, Any], std: torch.Tensor, pred: torch.Tensor, base: torch.Tensor, method: str) -> list[dict[str, Any]]:
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.float32)
    cand = normalized_abs_error(pred, target, mask.to(torch.bool), std)
    ref = normalized_abs_error(base, target, mask.to(torch.bool), std)
    rows: list[dict[str, Any]] = []
    for h in range(cand.shape[1]):
        for v in range(cand.shape[2]):
            denom = mask[:, h, v].sum().clamp_min(1)
            mae = float(cand[:, h, v].sum() / denom)
            bmae = float(ref[:, h, v].sum() / denom)
            rows.append({"method": method, "horizon": h, "variable": v, "mae": mae, "baseline_mae": bmae, "delta_vs_baseline": mae - bmae})
    return rows


def train_folds(n: int) -> list[tuple[int, int, int]]:
    min_train = int(round(n * 0.2))
    usable = n - min_train
    bounds = [min_train + i * usable // 4 for i in range(5)]
    return [(0, bounds[i], bounds[i + 1]) for i in range(4)]


def detector_grid() -> list[RegimeConfig]:
    configs = [RegimeConfig("fixed", 0.95, 0.95, 0.0, 0.0, 0.0, 0, 0)]
    for detector in ("zscore", "page_hinkley"):
        for slow in (0.97, 0.99):
            for fast in (0.90, 0.95):
                for reset in (0.0, 0.25):
                    cooldown = 24
                    boost = 24
                    if detector == "zscore":
                        for threshold in (2.0, 2.5):
                            configs.append(RegimeConfig(detector, slow, fast, threshold, 0.0, reset, cooldown, boost))
                    else:
                        for threshold in (0.05, 0.10):
                            for delta in (0.0, 0.005):
                                configs.append(RegimeConfig(detector, slow, fast, threshold, delta, reset, cooldown, boost))
    return configs


def simplicity_key(config: RegimeConfig) -> tuple[Any, ...]:
    detector_rank = {"fixed": 0, "zscore": 1, "page_hinkley": 2}[config.detector]
    return (detector_rank, config.reset_strength, config.boost_duration, config.cooldown, -config.threshold, config.fast_decay, -config.slow_decay)


def update_detector(
    score: float,
    config: RegimeConfig,
    state: dict[str, float],
) -> bool:
    if config.detector == "fixed":
        return False
    if config.detector == "zscore":
        mean = state["mean"]
        var = max(state["var"], 1e-8)
        trigger = abs(score - mean) > float(config.threshold) * math.sqrt(var)
        diff = score - mean
        state["mean"] = 0.99 * mean + 0.01 * score
        state["var"] = 0.99 * var + 0.01 * diff * diff
        return trigger
    if config.detector == "page_hinkley":
        mean = 0.99 * state["mean"] + 0.01 * score
        state["mean"] = mean
        state["cum"] = max(0.0, state["cum"] + score - mean - float(config.delta))
        if state["cum"] > float(config.threshold):
            state["cum"] = 0.0
            return True
        return False
    raise ValueError(config.detector)


def adaptive_hv_weights(
    starts: torch.Tensor,
    init_err: torch.Tensor,
    eval_err: torch.Tensor,
    horizon: int,
    config: RegimeConfig,
    oracle_change_points: set[int] | None = None,
) -> tuple[torch.Tensor, dict[str, Any], list[dict[str, Any]]]:
    ema = init_err.mean(dim=0)
    train_mean = ema.clone()
    init_scores = init_err.mean(dim=(1, 2, 3))
    detector_state = {
        "mean": float(init_scores.mean()),
        "var": float(init_scores.var(unbiased=False).clamp_min(1e-8)),
        "cum": 0.0,
    }
    trial = HvTrial("hv_ema", "regime_dynamic", mode="hv_lowrank", rank=1, decay=0.95, temperature=0.1)
    pending: list[int] = []
    weights = []
    traces = []
    updates = 0
    triggers = 0
    cooldown = 0
    boost = 0
    fast_updates = 0
    for i in range(eval_err.shape[0]):
        now = int(starts[i])
        still: list[int] = []
        for j in pending:
            if int(starts[j]) + horizon <= now:
                enforce_observable(int(starts[j]), now, horizon)
                score = float(eval_err[j].mean())
                trigger = False
                if oracle_change_points is not None:
                    trigger = int(starts[j]) in oracle_change_points
                elif cooldown <= 0:
                    trigger = update_detector(score, config, detector_state)
                if trigger:
                    triggers += 1
                    boost = int(config.boost_duration)
                    cooldown = int(config.cooldown)
                    if float(config.reset_strength) > 0:
                        ema = (1.0 - float(config.reset_strength)) * ema + float(config.reset_strength) * train_mean
                decay = float(config.fast_decay if boost > 0 else config.slow_decay)
                if boost > 0:
                    boost -= 1
                    fast_updates += 1
                if cooldown > 0:
                    cooldown -= 1
                ema = decay * ema + (1.0 - decay) * eval_err[j]
                updates += 1
            else:
                still.append(j)
        pending = still
        w = errors_to_weights(ema, trial)
        weights.append(w)
        traces.append(
            {
                "row": i,
                "start": now,
                "completed_updates": updates,
                "boost_remaining": boost,
                "cooldown_remaining": cooldown,
                "detector_mean": detector_state["mean"],
                "detector_var": detector_state["var"],
                **{f"mean_weight_{FIXED3[e]}": float(w[..., e].mean()) for e in range(3)},
            }
        )
        pending.append(i)
    w_t = torch.stack(weights)
    return w_t, {
        "num_updates": updates,
        "num_triggers": triggers,
        "fast_update_rate": float(fast_updates / max(updates, 1)),
        **{f"mean_weight_{FIXED3[e]}": float(w_t[..., e].mean()) for e in range(3)},
    }, traces


def prediction_for_arrays(
    starts: torch.Tensor,
    forecasts: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    std: torch.Tensor,
    static_weights: torch.Tensor,
    init_forecasts: torch.Tensor,
    init_target: torch.Tensor,
    init_mask: torch.Tensor,
    config: RegimeConfig,
    oracle_change_points: set[int] | None = None,
) -> tuple[torch.Tensor, dict[str, Any], list[dict[str, Any]]]:
    horizon = forecasts.shape[1]
    eval_hv_err = ((forecasts - target.unsqueeze(-1)) / std.view(1, 1, -1, 1)).abs() * mask.to(torch.float32).unsqueeze(-1)
    init_hv_err = ((init_forecasts - init_target.unsqueeze(-1)) / std.view(1, 1, -1, 1)).abs() * init_mask.to(torch.float32).unsqueeze(-1)
    eval_expert_err = eval_hv_err.mean(dim=(1, 2))
    init_expert_err = init_hv_err.mean(dim=(1, 2))
    online_weights, online_extra = chronological_online_weights(
        starts=starts,
        expert_mae=eval_expert_err,
        horizon=horizon,
        trial=ChronoTrial("ema", "ema_decay0.97_temp0.1", decay=0.97, temperature=0.1),
        train_mean_mae=init_expert_err.mean(dim=0),
        mode="ema",
    )
    chrono_weights = 0.5 * static_weights + 0.5 * online_weights
    chrono_weights = chrono_weights / chrono_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    chrono_pred = weighted_forecast(forecasts, chrono_weights)
    hv_weights, hv_extra, traces = adaptive_hv_weights(starts, init_hv_err, eval_hv_err, horizon, config, oracle_change_points=oracle_change_points)
    hv_pred = predict_from_hv_weights(forecasts, hv_weights)
    return 0.25 * chrono_pred + 0.75 * hv_pred, {**online_extra, **hv_extra}, traces


def oracle_change_points(starts: torch.Tensor, err: torch.Tensor, blocks: int = 6) -> set[int]:
    n = err.shape[0]
    out: set[int] = set()
    prev_best = None
    for b in range(blocks):
        lo = b * n // blocks
        hi = (b + 1) * n // blocks
        best = int(err[lo:hi].mean(dim=(0, 1, 2)).argmin())
        if prev_best is not None and best != prev_best:
            out.add(int(starts[lo]))
        prev_best = best
    return out


def evaluate_folds(
    cache: Mapping[str, Any],
    std: torch.Tensor,
    configs: Sequence[RegimeConfig],
    folds: Sequence[tuple[int, int, int]],
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    starts = cache["absolute_window_starts"].to(torch.long)
    forecasts = fixed3_forecasts(cache)
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.bool)
    static_weights, _, _ = load_static_winner_per_window(7, cache, std, device)
    fixed_config = RegimeConfig("fixed", 0.95, 0.95, 0.0, 0.0, 0.0, 0, 0)
    base_by_fold: list[tuple[torch.Tensor, torch.Tensor]] = []
    for train_lo, eval_lo, eval_hi in folds:
        base_pred, _, _ = prediction_for_arrays(
            starts[eval_lo:eval_hi],
            forecasts[eval_lo:eval_hi],
            target[eval_lo:eval_hi],
            mask[eval_lo:eval_hi],
            std,
            static_weights[eval_lo:eval_hi],
            forecasts[train_lo:eval_lo],
            target[train_lo:eval_lo],
            mask[train_lo:eval_lo],
            fixed_config,
        )
        base_mae = sample_mae(base_pred, target[eval_lo:eval_hi], mask[eval_lo:eval_hi], std)
        base_by_fold.append((base_pred, base_mae))
    rows = []
    details = []
    for idx, config in enumerate(configs, start=1):
        if idx == 1 or idx % 50 == 0 or idx == len(configs):
            print(f"[regime-select] {idx}/{len(configs)} {config.name}", flush=True)
        cand_all = []
        base_all = []
        fold_rows = []
        for fold_id, (train_lo, eval_lo, eval_hi) in enumerate(folds):
            pred, extra, _ = prediction_for_arrays(
                starts[eval_lo:eval_hi],
                forecasts[eval_lo:eval_hi],
                target[eval_lo:eval_hi],
                mask[eval_lo:eval_hi],
                std,
                static_weights[eval_lo:eval_hi],
                forecasts[train_lo:eval_lo],
                target[train_lo:eval_lo],
                mask[train_lo:eval_lo],
                config,
            )
            cm = sample_mae(pred, target[eval_lo:eval_hi], mask[eval_lo:eval_hi], std)
            bm = base_by_fold[fold_id][1]
            cand_all.append(cm)
            base_all.append(bm)
            frow = {"name": config.name, "fold": fold_id, "mae": float(cm.mean()), "baseline_mae": float(bm.mean()), "delta": float(cm.mean() - bm.mean()), **extra}
            fold_rows.append(frow)
            details.append({**asdict(config), **frow})
        cand = torch.cat(cand_all)
        base = torch.cat(base_all)
        deltas = torch.tensor([float(r["delta"]) for r in fold_rows])
        rows.append(
            {
                "name": config.name,
                **asdict(config),
                "fold_mae_mean": float(cand.mean()),
                "fold_baseline_mae_mean": float(base.mean()),
                "fold_delta_mean": float(cand.mean() - base.mean()),
                "fold_delta_std": float(deltas.std(unbiased=False)),
                "fold_mae_se": float(deltas.std(unbiased=True) / math.sqrt(len(fold_rows))),
                "fold_wins": int(sum(float(r["delta"]) < 0 for r in fold_rows)),
                "mean_triggers": float(np.mean([float(r["num_triggers"]) for r in fold_rows])),
                "fast_update_rate": float(np.mean([float(r["fast_update_rate"]) for r in fold_rows])),
            }
        )
    return rows, details


def select_one_se(rows: Sequence[Mapping[str, Any]], configs: Mapping[str, RegimeConfig]) -> Mapping[str, Any]:
    best = min(rows, key=lambda r: float(r["fold_mae_mean"]))
    threshold = float(best["fold_mae_mean"]) + float(best["fold_mae_se"])
    eligible = [r for r in rows if float(r["fold_mae_mean"]) <= threshold]
    selected = sorted(eligible, key=lambda r: simplicity_key(configs[str(r["name"])]))[0]
    return {"best": best, "selected": selected, "one_se_threshold": threshold}


def aggregate_seed_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    out = []
    for method, group in grouped.items():
        maes = torch.tensor([float(r["mae"]) for r in group])
        mses = torch.tensor([float(r["mse"]) for r in group])
        out.append(
            {
                "method": method,
                "seeds": len(group),
                "mae_mean": float(maes.mean()),
                "mae_std": float(maes.std(unbiased=False)) if len(group) > 1 else 0.0,
                "mse_mean": float(mses.mean()),
                "mse_std": float(mses.std(unbiased=False)) if len(group) > 1 else 0.0,
                "improvement_vs_0.363642": BASELINE_MAE - float(maes.mean()),
                "percent_improvement_vs_0.363642": 100.0 * (BASELINE_MAE - float(maes.mean())) / BASELINE_MAE,
            }
        )
    return sorted(out, key=lambda r: float(r["mae_mean"]))


def make_report(out_dir: Path, report: Mapping[str, Any]) -> None:
    sel = report["selection"]["selected"]
    best = report["best_validation_method"]
    ci = report["aggregate_bootstrap_ci"].get("regime_selected", {})
    lines = [
        "# Regime Adaptive Forgetting COSTAR-TS",
        "",
        "## Protocol",
        "",
        f"- Frozen baseline shape: `{BASELINE_NAME}`.",
        "- Only the horizon-variable EMA forgetting speed changes.",
        "- Detector settings were selected on router-train chronological folds.",
        "- Oracle change points are diagnostic only and ineligible.",
        "- Test cache was not loaded.",
        "",
        "## Selection",
        "",
        f"- Selected config: `{sel['name']}`.",
        f"- Best fold config: `{report['selection']['best']['name']}`.",
        f"- One-SE threshold: `{report['selection']['one_se_threshold']:.6f}`.",
        "",
        "## Validation",
        "",
        f"- Best eligible validation method: `{best['method']}`.",
        f"- MAE / MSE: `{best['mae_mean']:.6f}` / `{best['mse_mean']:.6f}`.",
        f"- Selected-method CI vs fixed decay: `[{ci.get('ci95_low', 0.0):.6f}, {ci.get('ci95_high', 0.0):.6f}]`.",
        f"- Strong target `<= 0.3619`: `{bool(best['mae_mean'] <= STRONG_TARGET)}`.",
        "",
        "## Reproduce",
        "",
        "```powershell",
        report["reproduce_command"],
        "```",
    ]
    (out_dir / "regime_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", default="cache/costarts_walkforward/router_train_20_60_cache.pt")
    parser.add_argument("--val-cache", default="cache/costarts_walkforward/router_val_60_80_cache.pt")
    parser.add_argument("--normalizer-checkpoint", default="checkpoints/costarts_walkforward/final_60/DLinear/best_expert.pt")
    parser.add_argument("--out-dir", default="experiments/regime_adaptive_forgetting_costar")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    args = parser.parse_args()
    t0 = time.time()
    for path in (args.train_cache, args.val_cache, args.normalizer_checkpoint):
        refuse_test(path)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_window").mkdir(exist_ok=True)
    train_cache = load_cache(ROOT / args.train_cache, "router_train_20_60")
    val_cache = load_cache(ROOT / args.val_cache, "router_val_60_80")
    std = load_std(ROOT / args.normalizer_checkpoint, int(val_cache["num_features"]))
    device = torch.device(args.device)
    val_starts = val_cache["absolute_window_starts"].to(torch.long)
    if int(val_starts[-1]) + int(val_cache["forecast_horizon"]) > 11520:
        raise ValueError("Validation target crosses held-out test boundary")

    configs = detector_grid()
    cfg_by_name = {c.name: c for c in configs}
    folds = train_folds(int(train_cache["num_windows"]))
    fold_rows, fold_details = evaluate_folds(train_cache, std, configs, folds, device)
    write_csv(out_dir / "router_train_fold_leaderboard.csv", sorted(fold_rows, key=lambda r: float(r["fold_mae_mean"])))
    write_csv(out_dir / "router_train_fold_details.csv", fold_details)
    selection = select_one_se(fold_rows, cfg_by_name)
    write_json(out_dir / "selected_config.json", selection)
    selected_cfg = cfg_by_name[str(selection["selected"]["name"])]

    validation_rows = []
    trace_rows = []
    axis_rows = []
    hv_rows = []
    per_method: dict[str, list[torch.Tensor]] = defaultdict(list)
    per_base: dict[int, torch.Tensor] = {}
    val_forecasts = fixed3_forecasts(val_cache)
    val_target = val_cache["targets"].to(torch.float32)
    val_mask = val_cache["target_masks"].to(torch.bool)
    train_forecasts = fixed3_forecasts(train_cache)
    train_target = train_cache["targets"].to(torch.float32)
    train_mask = train_cache["target_masks"].to(torch.bool)
    val_err = per_location_abs_error(val_cache, std)
    oracle_points = oracle_change_points(val_starts, val_err)
    oracle_cfg = RegimeConfig("zscore", 0.99, 0.90, 0.0, 0.0, 0.25, 48, 48)

    for seed in SEEDS:
        base_pred, base_extra = fixed_current_best_prediction(val_cache, train_cache, std, seed, device)
        bm = metrics(val_cache, std, base_pred)
        per_base[seed] = bm["per_window_mae"]
        per_method["baseline_fixed_decay"].append(bm["per_window_mae"])
        validation_rows.append({"method": "baseline_fixed_decay", "seed": seed, "mae": bm["mae"], "mse": bm["mse"], "diff_vs_0.363642": bm["mae"] - BASELINE_MAE, **base_extra})
        static_weights, _, _ = load_static_winner_per_window(seed, val_cache, std, device)
        pred, extra, traces = prediction_for_arrays(
            val_starts,
            val_forecasts,
            val_target,
            val_mask,
            std,
            static_weights,
            train_forecasts,
            train_target,
            train_mask,
            selected_cfg,
        )
        mm = metrics(val_cache, std, pred)
        boot = paired_bootstrap(mm["per_window_mae"], bm["per_window_mae"], seed=seed, samples=args.bootstrap_samples)
        validation_rows.append({"method": "regime_selected", "seed": seed, "mae": mm["mae"], "mse": mm["mse"], "diff_vs_0.363642": mm["mae"] - BASELINE_MAE, **boot, **extra})
        per_method["regime_selected"].append(mm["per_window_mae"])
        trace_rows.extend([{**r, "method": "regime_selected", "seed": seed, "config": selected_cfg.name} for r in traces])
        axis_rows.extend(per_axis_rows(val_cache, std, pred, base_pred, f"regime_selected_seed{seed}"))
        hv_rows.extend(per_hv_rows(val_cache, std, pred, base_pred, f"regime_selected_seed{seed}"))
        torch.save(mm["per_window_mae"], out_dir / "per_window" / f"regime_selected_seed{seed}.pt")

        opred, oextra, otraces = prediction_for_arrays(
            val_starts,
            val_forecasts,
            val_target,
            val_mask,
            std,
            static_weights,
            train_forecasts,
            train_target,
            train_mask,
            oracle_cfg,
            oracle_change_points=oracle_points,
        )
        om = metrics(val_cache, std, opred)
        validation_rows.append({"method": "oracle_change_diagnostic_ineligible", "seed": seed, "mae": om["mae"], "mse": om["mse"], "diff_vs_0.363642": om["mae"] - BASELINE_MAE, **oextra})
        per_method["oracle_change_diagnostic_ineligible"].append(om["per_window_mae"])
        trace_rows.extend([{**r, "method": "oracle_change_diagnostic_ineligible", "seed": seed, "config": "oracle_block_changes"} for r in otraces])

    summary = aggregate_seed_summary([r for r in validation_rows if int(r["seed"]) >= 0])
    base_all = torch.cat([per_base[s] for s in SEEDS])
    agg_boot: dict[str, dict[str, Any]] = {}
    for method, chunks in per_method.items():
        agg_boot[method] = paired_bootstrap(torch.cat(chunks), base_all, seed=20260812, samples=args.bootstrap_samples)
    hv_group: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    for row in hv_rows:
        method = row["method"].rsplit("_seed", 1)[0]
        hv_group[(method, int(row["horizon"]), int(row["variable"]))].append(float(row["delta_vs_baseline"]))
    worst = []
    for method in sorted({key[0] for key in hv_group}):
        candidates = [{"method": m, "horizon": h, "variable": v, "delta_vs_baseline_mean": float(np.mean(vals))} for (m, h, v), vals in hv_group.items() if m == method]
        worst.append(max(candidates, key=lambda r: float(r["delta_vs_baseline_mean"])))

    write_csv(out_dir / "validation_per_seed_results.csv", validation_rows)
    write_csv(out_dir / "validation_seed_summary.csv", summary)
    write_csv(out_dir / "validation_regime_traces.csv", trace_rows)
    write_csv(out_dir / "per_axis_mae.csv", axis_rows)
    write_csv(out_dir / "per_horizon_variable_mae.csv", hv_rows)
    write_json(out_dir / "aggregate_bootstrap_ci.json", agg_boot)
    eligible = [r for r in summary if r["method"] != "oracle_change_diagnostic_ineligible"]
    best = min(eligible, key=lambda r: float(r["mae_mean"]))
    selected_boot = agg_boot.get("regime_selected", {})
    promote = bool(
        best["method"] == "regime_selected"
        and float(best["mae_mean"]) < BASELINE_MAE
        and selected_boot.get("ci_excludes_zero", False)
        and float(selected_boot.get("ci95_high", 1.0)) < 0.0
        and int(selection["selected"]["fold_wins"]) >= 3
        and all(float(r["delta_vs_baseline_mean"]) <= 0.001 for r in worst)
    )
    report = {
        "baseline_reference": {"name": BASELINE_NAME, "mae": BASELINE_MAE, "mse": BASELINE_MSE},
        "selection": selection,
        "validation_seed_summary": summary,
        "aggregate_bootstrap_ci": agg_boot,
        "oracle_change_points": sorted(oracle_points),
        "worst_horizon_variable_regression": worst,
        "best_validation_method": best,
        "promotion_decision": "Promote regime-adaptive forgetting." if promote else "Do not promote regime-adaptive forgetting.",
        "promotion_checks": {
            "beats_0.363642": float(best["mae_mean"]) < BASELINE_MAE,
            "selected_fold_wins": int(selection["selected"]["fold_wins"]),
            "ci_excludes_zero": selected_boot.get("ci_excludes_zero", False),
            "no_major_regression": all(float(r["delta_vs_baseline_mean"]) <= 0.001 for r in worst),
        },
        "safety": {
            "test_cache_loaded": False,
            "validation_used_for_selection": False,
            "oracle_diagnostic_ineligible": True,
            "causality_rule": "old_start + prediction_length <= current_start",
        },
        "runtime_sec": time.time() - t0,
        "device": str(device),
        "reproduce_command": f"python experiments\\regime_adaptive_forgetting_costar\\run_regime_adaptive_forgetting.py --device {args.device}",
    }
    write_json(out_dir / "final_report.json", report)
    make_report(out_dir, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
