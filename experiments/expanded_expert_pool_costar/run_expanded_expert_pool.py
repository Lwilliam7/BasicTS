"""Focused expanded expert-pool experiment for COSTAR-TS.

DLinear and ModernTCN are tested as small causal specialists on top of the
frozen fixed-three horizon-variable baseline.  The test cache is refused.
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
    enforce_observable,
    paired_bootstrap,
)
from experiments.oracle_weight_tournament.run_tournament import (  # noqa: E402
    load_cache,
    load_std,
    sample_mae,
    sample_mse,
)
from experiments.residual_correction_costar.run_residual_correction_experiments import (  # noqa: E402
    BASELINE_MAE,
    BASELINE_MSE,
    fixed_current_best_prediction,
)


BASELINE_NAME = "hybrid_chrono_hvema_lowrank1_decay0.95_temp0.1_alpha0.75"
OPTIONAL_EXPERTS = ("DLinear", "ModernTCN")
ADVANTAGE_SCALE = 0.05
EPS = 1e-8


@dataclass(frozen=True)
class Config:
    scenario: str
    structure: str
    decay: float
    extra_weight_cap: float
    activation_margin: float
    warmup: int

    @property
    def name(self) -> str:
        pct = int(round(self.activation_margin * 10000))
        return f"{self.scenario}_{self.structure}_decay{self.decay:g}_cap{self.extra_weight_cap:g}_marginbp{pct}_warm{self.warmup}"


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


def expert_index(cache: Mapping[str, Any], name: str) -> int:
    return list(cache["expert_names"]).index(name)


def optional_predictions(cache: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    stack = cache["prediction_stack"].to(torch.float32)
    return stack[..., expert_index(cache, "DLinear")], stack[..., expert_index(cache, "ModernTCN")]


def metrics(cache: Mapping[str, Any], std: torch.Tensor, pred: torch.Tensor) -> dict[str, Any]:
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.bool)
    mae = sample_mae(pred, target, mask, std)
    mse = sample_mse(pred, target, mask, std)
    return {"mae": float(mae.mean()), "mse": float(mse.mean()), "per_window_mae": mae, "per_window_mse": mse}


def normalized_abs_error(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return ((pred - target) / std.view(1, 1, -1)).abs() * mask.to(torch.float32)


def aggregate_error(err: torch.Tensor, structure: str) -> torch.Tensor:
    if structure == "global":
        return err.mean().view(1, 1)
    if structure == "variable":
        return err.mean(dim=0, keepdim=True)
    if structure == "hv":
        return err
    raise ValueError(structure)


def expand_group(x: torch.Tensor, structure: str, h: int, v: int) -> torch.Tensor:
    if structure == "global":
        return x.expand(h, v)
    if structure == "variable":
        return x.expand(h, v)
    if structure == "hv":
        return x
    raise ValueError(structure)


def init_state_from_errors(err: torch.Tensor, structure: str) -> torch.Tensor:
    grouped = torch.stack([aggregate_error(e, structure) for e in err])
    return grouped.mean(dim=0)


def active_mask_for_scenario(scenario: str) -> tuple[bool, bool]:
    if scenario == "dlinear_only":
        return True, False
    if scenario == "moderntcn_only":
        return False, True
    if scenario == "both":
        return True, True
    raise ValueError(scenario)


def weights_from_advantage(adv_d: torch.Tensor, adv_m: torch.Tensor, config: Config) -> tuple[torch.Tensor, torch.Tensor]:
    use_d, use_m = active_mask_for_scenario(config.scenario)
    raw_d = ((adv_d - float(config.activation_margin)).clamp_min(0.0) / ADVANTAGE_SCALE).clamp_max(1.0) if use_d else torch.zeros_like(adv_d)
    raw_m = ((adv_m - float(config.activation_margin)).clamp_min(0.0) / ADVANTAGE_SCALE).clamp_max(1.0) if use_m else torch.zeros_like(adv_m)
    # Each specialist is capped at half the combined budget; the combined cap is enforced again below.
    w_d = raw_d * (float(config.extra_weight_cap) / 2.0)
    w_m = raw_m * (float(config.extra_weight_cap) / 2.0)
    total = w_d + w_m
    scale = torch.where(total > float(config.extra_weight_cap), float(config.extra_weight_cap) / total.clamp_min(EPS), torch.ones_like(total))
    w_d = w_d * scale
    w_m = w_m * scale
    if torch.any(w_d < -1e-8) or torch.any(w_m < -1e-8) or torch.any(w_d + w_m > float(config.extra_weight_cap) + 1e-6):
        raise AssertionError("Expanded-pool weights violate convex cap")
    return w_d, w_m


def run_causal_specialists(
    starts: torch.Tensor,
    base_pred: torch.Tensor,
    d_pred: torch.Tensor,
    m_pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    std: torch.Tensor,
    config: Config,
    init_base_err: torch.Tensor,
    init_d_err: torch.Tensor,
    init_m_err: torch.Tensor,
    trace_prefix: Mapping[str, Any] | None = None,
) -> tuple[torch.Tensor, dict[str, Any], list[dict[str, Any]]]:
    h, v = base_pred.shape[1], base_pred.shape[2]
    base_state = init_state_from_errors(init_base_err, config.structure)
    d_state = init_state_from_errors(init_d_err, config.structure)
    m_state = init_state_from_errors(init_m_err, config.structure)
    count = int(init_base_err.shape[0])
    pending: list[int] = []
    preds: list[torch.Tensor] = []
    traces: list[dict[str, Any]] = []
    updates = 0
    for i in range(base_pred.shape[0]):
        now = int(starts[i])
        still: list[int] = []
        for j in pending:
            if int(starts[j]) + h <= now:
                enforce_observable(int(starts[j]), now, h)
                base_e = aggregate_error(normalized_abs_error(base_pred[j : j + 1], target[j : j + 1], mask[j : j + 1], std)[0], config.structure)
                d_e = aggregate_error(normalized_abs_error(d_pred[j : j + 1], target[j : j + 1], mask[j : j + 1], std)[0], config.structure)
                m_e = aggregate_error(normalized_abs_error(m_pred[j : j + 1], target[j : j + 1], mask[j : j + 1], std)[0], config.structure)
                base_state = float(config.decay) * base_state + (1.0 - float(config.decay)) * base_e
                d_state = float(config.decay) * d_state + (1.0 - float(config.decay)) * d_e
                m_state = float(config.decay) * m_state + (1.0 - float(config.decay)) * m_e
                count += 1
                updates += 1
            else:
                still.append(j)
        pending = still
        base_s = expand_group(base_state, config.structure, h, v)
        d_s = expand_group(d_state, config.structure, h, v)
        m_s = expand_group(m_state, config.structure, h, v)
        adv_d = (base_s - d_s) / base_s.clamp_min(EPS)
        adv_m = (base_s - m_s) / base_s.clamp_min(EPS)
        if count < int(config.warmup):
            w_d = torch.zeros_like(adv_d)
            w_m = torch.zeros_like(adv_m)
        else:
            w_d, w_m = weights_from_advantage(adv_d, adv_m, config)
        pred = (1.0 - w_d - w_m) * base_pred[i] + w_d * d_pred[i] + w_m * m_pred[i]
        preds.append(pred)
        trace_base = dict(trace_prefix or {})
        traces.append(
            {
                **trace_base,
                "row": i,
                "start": now,
                "completed_count": count,
                "mean_weight_DLinear": float(w_d.mean()),
                "max_weight_DLinear": float(w_d.max()),
                "mean_weight_ModernTCN": float(w_m.mean()),
                "max_weight_ModernTCN": float(w_m.max()),
                "activated_DLinear": bool((w_d > 0).any()),
                "activated_ModernTCN": bool((w_m > 0).any()),
                "mean_adv_DLinear": float(adv_d.mean()),
                "mean_adv_ModernTCN": float(adv_m.mean()),
                "mean_adv_DLinear_when_active": float(adv_d[w_d > 0].mean()) if bool((w_d > 0).any()) else 0.0,
                "mean_adv_ModernTCN_when_active": float(adv_m[w_m > 0].mean()) if bool((w_m > 0).any()) else 0.0,
            }
        )
        pending.append(i)
    pred_t = torch.stack(preds)
    mean_wd = torch.tensor([r["mean_weight_DLinear"] for r in traces], dtype=torch.float32)
    mean_wm = torch.tensor([r["mean_weight_ModernTCN"] for r in traces], dtype=torch.float32)
    extra = {
        "num_updates": updates,
        "avg_weight_DLinear": float(mean_wd.mean()),
        "max_window_weight_DLinear": float(mean_wd.max()),
        "avg_weight_ModernTCN": float(mean_wm.mean()),
        "max_window_weight_ModernTCN": float(mean_wm.max()),
        "activation_rate_DLinear": float(torch.tensor([r["activated_DLinear"] for r in traces], dtype=torch.float32).mean()),
        "activation_rate_ModernTCN": float(torch.tensor([r["activated_ModernTCN"] for r in traces], dtype=torch.float32).mean()),
        "mean_adv_DLinear_when_active": float(np.mean([r["mean_adv_DLinear_when_active"] for r in traces if r["activated_DLinear"]])) if any(r["activated_DLinear"] for r in traces) else 0.0,
        "mean_adv_ModernTCN_when_active": float(np.mean([r["mean_adv_ModernTCN_when_active"] for r in traces if r["activated_ModernTCN"]])) if any(r["activated_ModernTCN"] for r in traces) else 0.0,
    }
    return pred_t, extra, traces


def activation_turnover(traces: Sequence[Mapping[str, Any]], expert: str) -> dict[str, Any]:
    key = f"activated_{expert}"
    vals = [bool(r[key]) for r in traces]
    if not vals:
        return {f"{expert}_activation_segments": 0, f"{expert}_mean_activation_duration": 0.0, f"{expert}_turnover_rate": 0.0}
    segments = []
    cur = 0
    transitions = 0
    prev = vals[0]
    for val in vals:
        if val:
            cur += 1
        elif cur:
            segments.append(cur)
            cur = 0
        if val != prev:
            transitions += 1
        prev = val
    if cur:
        segments.append(cur)
    return {
        f"{expert}_activation_segments": len(segments),
        f"{expert}_mean_activation_duration": float(np.mean(segments)) if segments else 0.0,
        f"{expert}_max_activation_duration": int(max(segments)) if segments else 0,
        f"{expert}_turnover_rate": float(transitions / max(len(vals) - 1, 1)),
    }


def train_folds(n: int) -> list[tuple[int, int, int]]:
    min_train = int(round(n * 0.2))
    usable = n - min_train
    bounds = [min_train + i * usable // 4 for i in range(5)]
    return [(0, bounds[i], bounds[i + 1]) for i in range(4)]


def grid() -> list[Config]:
    return [
        Config(scenario, structure, decay, cap, margin, warm)
        for scenario in ("dlinear_only", "moderntcn_only", "both")
        for structure in ("global", "variable", "hv")
        for decay in (0.95, 0.97, 0.98, 0.99)
        for cap in (0.025, 0.05, 0.10)
        for margin in (0.0, 0.005, 0.01, 0.02)
        for warm in (24, 48, 96)
    ]


def simplicity_key(config: Config) -> tuple[Any, ...]:
    structure_rank = {"global": 0, "variable": 1, "hv": 2}[config.structure]
    return (structure_rank, config.extra_weight_cap, -config.activation_margin, -config.warmup, -config.decay)


def select_with_one_se(rows: list[dict[str, Any]], configs_by_name: Mapping[str, Config]) -> dict[str, Any]:
    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_scenario[row["scenario"]].append(row)
    selected = {}
    for scenario, group in by_scenario.items():
        best = min(group, key=lambda r: float(r["fold_mae_mean"]))
        threshold = float(best["fold_mae_mean"]) + float(best["fold_mae_se"])
        candidates = [r for r in group if float(r["fold_mae_mean"]) <= threshold]
        chosen = sorted(candidates, key=lambda r: simplicity_key(configs_by_name[r["name"]]))[0]
        selected[scenario] = {"best": best, "selected": chosen, "one_se_threshold": threshold}
    return selected


def fold_eval(
    cache: Mapping[str, Any],
    std: torch.Tensor,
    base_all: torch.Tensor,
    config: Config,
    folds: Sequence[tuple[int, int, int]],
) -> dict[str, Any]:
    starts = cache["absolute_window_starts"].to(torch.long)
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.bool)
    d_all, m_all = optional_predictions(cache)
    fold_rows = []
    cand_all, base_mae_all = [], []
    for fold_id, (train_lo, eval_lo, eval_hi) in enumerate(folds):
        init_slice = slice(train_lo, eval_lo)
        eval_slice = slice(eval_lo, eval_hi)
        init_base_err = normalized_abs_error(base_all[init_slice], target[init_slice], mask[init_slice], std)
        init_d_err = normalized_abs_error(d_all[init_slice], target[init_slice], mask[init_slice], std)
        init_m_err = normalized_abs_error(m_all[init_slice], target[init_slice], mask[init_slice], std)
        pred, extra, traces = run_causal_specialists(
            starts[eval_slice],
            base_all[eval_slice],
            d_all[eval_slice],
            m_all[eval_slice],
            target[eval_slice],
            mask[eval_slice],
            std,
            config,
            init_base_err,
            init_d_err,
            init_m_err,
            trace_prefix={"fold": fold_id, "config": config.name},
        )
        cm = sample_mae(pred, target[eval_slice], mask[eval_slice], std)
        bm = sample_mae(base_all[eval_slice], target[eval_slice], mask[eval_slice], std)
        cand_all.append(cm)
        base_mae_all.append(bm)
        fold_rows.append(
            {
                "name": config.name,
                "scenario": config.scenario,
                "fold": fold_id,
                "mae": float(cm.mean()),
                "baseline_mae": float(bm.mean()),
                "delta": float(cm.mean() - bm.mean()),
                **extra,
                **activation_turnover(traces, "DLinear"),
                **activation_turnover(traces, "ModernTCN"),
            }
        )
    cand = torch.cat(cand_all)
    base = torch.cat(base_mae_all)
    deltas = torch.tensor([r["delta"] for r in fold_rows], dtype=torch.float32)
    return {
        "name": config.name,
        "scenario": config.scenario,
        **asdict(config),
        "fold_mae_mean": float(cand.mean()),
        "fold_baseline_mae_mean": float(base.mean()),
        "fold_delta_mean": float(cand.mean() - base.mean()),
        "fold_delta_std": float(deltas.std(unbiased=False)),
        "fold_mae_se": float(deltas.std(unbiased=True) / math.sqrt(len(fold_rows))),
        "fold_wins": int(sum(r["delta"] < 0 for r in fold_rows)),
        "fold_rows": fold_rows,
    }


def causal_advantage_sequence(
    starts: torch.Tensor,
    base_err: torch.Tensor,
    d_err: torch.Tensor,
    m_err: torch.Tensor,
    structure: str,
    decay: float,
    init_base_err: torch.Tensor,
    init_d_err: torch.Tensor,
    init_m_err: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    h, v = base_err.shape[1], base_err.shape[2]
    base_state = init_state_from_errors(init_base_err, structure)
    d_state = init_state_from_errors(init_d_err, structure)
    m_state = init_state_from_errors(init_m_err, structure)
    count = int(init_base_err.shape[0])
    pending: list[int] = []
    adv_d_seq: list[torch.Tensor] = []
    adv_m_seq: list[torch.Tensor] = []
    count_seq: list[int] = []
    for i in range(base_err.shape[0]):
        now = int(starts[i])
        still: list[int] = []
        for j in pending:
            if int(starts[j]) + h <= now:
                enforce_observable(int(starts[j]), now, h)
                base_state = float(decay) * base_state + (1.0 - float(decay)) * aggregate_error(base_err[j], structure)
                d_state = float(decay) * d_state + (1.0 - float(decay)) * aggregate_error(d_err[j], structure)
                m_state = float(decay) * m_state + (1.0 - float(decay)) * aggregate_error(m_err[j], structure)
                count += 1
            else:
                still.append(j)
        pending = still
        base_s = expand_group(base_state, structure, h, v)
        d_s = expand_group(d_state, structure, h, v)
        m_s = expand_group(m_state, structure, h, v)
        adv_d_seq.append((base_s - d_s) / base_s.clamp_min(EPS))
        adv_m_seq.append((base_s - m_s) / base_s.clamp_min(EPS))
        count_seq.append(count)
        pending.append(i)
    return torch.stack(adv_d_seq), torch.stack(adv_m_seq), torch.tensor(count_seq, dtype=torch.long)


def grid_eval_cached(
    cache: Mapping[str, Any],
    std: torch.Tensor,
    base_all: torch.Tensor,
    configs: Sequence[Config],
    folds: Sequence[tuple[int, int, int]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    starts = cache["absolute_window_starts"].to(torch.long)
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.bool)
    d_all, m_all = optional_predictions(cache)
    base_err_all = normalized_abs_error(base_all, target, mask, std)
    d_err_all = normalized_abs_error(d_all, target, mask, std)
    m_err_all = normalized_abs_error(m_all, target, mask, std)
    cache_seq: dict[tuple[str, float, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for fold_id, (train_lo, eval_lo, eval_hi) in enumerate(folds):
        for structure in ("global", "variable", "hv"):
            for decay in (0.95, 0.97, 0.98, 0.99):
                cache_seq[(structure, decay, fold_id)] = causal_advantage_sequence(
                    starts[eval_lo:eval_hi],
                    base_err_all[eval_lo:eval_hi],
                    d_err_all[eval_lo:eval_hi],
                    m_err_all[eval_lo:eval_hi],
                    structure,
                    decay,
                    base_err_all[train_lo:eval_lo],
                    d_err_all[train_lo:eval_lo],
                    m_err_all[train_lo:eval_lo],
                )

    leaderboard: list[dict[str, Any]] = []
    fold_detail_rows: list[dict[str, Any]] = []
    for config in configs:
        fold_rows = []
        cand_all = []
        base_all_mae = []
        for fold_id, (_, eval_lo, eval_hi) in enumerate(folds):
            adv_d, adv_m, count_seq = cache_seq[(config.structure, config.decay, fold_id)]
            if count_seq.numel() and int(count_seq.min()) < int(config.warmup):
                warm = (count_seq >= int(config.warmup)).to(torch.float32).view(-1, 1, 1)
            else:
                warm = 1.0
            w_d, w_m = weights_from_advantage(adv_d, adv_m, config)
            w_d = w_d * warm
            w_m = w_m * warm
            pred = (1.0 - w_d - w_m) * base_all[eval_lo:eval_hi] + w_d * d_all[eval_lo:eval_hi] + w_m * m_all[eval_lo:eval_hi]
            cm = sample_mae(pred, target[eval_lo:eval_hi], mask[eval_lo:eval_hi], std)
            bm = sample_mae(base_all[eval_lo:eval_hi], target[eval_lo:eval_hi], mask[eval_lo:eval_hi], std)
            cand_all.append(cm)
            base_all_mae.append(bm)
            active_d = (w_d.flatten(1).sum(dim=1) > 0)
            active_m = (w_m.flatten(1).sum(dim=1) > 0)
            fold_rows.append(
                {
                    "name": config.name,
                    "scenario": config.scenario,
                    "fold": fold_id,
                    "mae": float(cm.mean()),
                    "baseline_mae": float(bm.mean()),
                    "delta": float(cm.mean() - bm.mean()),
                    "avg_weight_DLinear": float(w_d.mean()),
                    "avg_weight_ModernTCN": float(w_m.mean()),
                    "max_window_weight_DLinear": float(w_d.flatten(1).mean(dim=1).max()),
                    "max_window_weight_ModernTCN": float(w_m.flatten(1).mean(dim=1).max()),
                    "activation_rate_DLinear": float(active_d.to(torch.float32).mean()),
                    "activation_rate_ModernTCN": float(active_m.to(torch.float32).mean()),
                }
            )
        cand = torch.cat(cand_all)
        base = torch.cat(base_all_mae)
        deltas = torch.tensor([r["delta"] for r in fold_rows], dtype=torch.float32)
        leaderboard.append(
            {
                "name": config.name,
                "scenario": config.scenario,
                **asdict(config),
                "fold_mae_mean": float(cand.mean()),
                "fold_baseline_mae_mean": float(base.mean()),
                "fold_delta_mean": float(cand.mean() - base.mean()),
                "fold_delta_std": float(deltas.std(unbiased=False)),
                "fold_mae_se": float(deltas.std(unbiased=True) / math.sqrt(len(fold_rows))),
                "fold_wins": int(sum(r["delta"] < 0 for r in fold_rows)),
                "avg_weight_DLinear": float(np.mean([r["avg_weight_DLinear"] for r in fold_rows])),
                "avg_weight_ModernTCN": float(np.mean([r["avg_weight_ModernTCN"] for r in fold_rows])),
                "activation_rate_DLinear": float(np.mean([r["activation_rate_DLinear"] for r in fold_rows])),
                "activation_rate_ModernTCN": float(np.mean([r["activation_rate_ModernTCN"] for r in fold_rows])),
            }
        )
        fold_detail_rows.extend(fold_rows)
    return leaderboard, fold_detail_rows


def per_axis_rows(cache: Mapping[str, Any], std: torch.Tensor, pred: torch.Tensor, base: torch.Tensor, method: str) -> list[dict[str, Any]]:
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.float32)
    ce = normalized_abs_error(pred, target, mask.to(torch.bool), std)
    be = normalized_abs_error(base, target, mask.to(torch.bool), std)
    rows = []
    for h in range(ce.shape[1]):
        denom = mask[:, h].sum().clamp_min(1)
        rows.append({"method": method, "axis": "horizon", "index": h, "mae": float(ce[:, h].sum() / denom), "baseline_mae": float(be[:, h].sum() / denom)})
    for v in range(ce.shape[2]):
        denom = mask[:, :, v].sum().clamp_min(1)
        rows.append({"method": method, "axis": "variable", "index": v, "mae": float(ce[:, :, v].sum() / denom), "baseline_mae": float(be[:, :, v].sum() / denom)})
    return rows


def per_hv_rows(cache: Mapping[str, Any], std: torch.Tensor, pred: torch.Tensor, base: torch.Tensor, method: str) -> list[dict[str, Any]]:
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.float32)
    ce = normalized_abs_error(pred, target, mask.to(torch.bool), std)
    be = normalized_abs_error(base, target, mask.to(torch.bool), std)
    rows = []
    for h in range(ce.shape[1]):
        for v in range(ce.shape[2]):
            denom = mask[:, h, v].sum().clamp_min(1)
            mae = float(ce[:, h, v].sum() / denom)
            bmae = float(be[:, h, v].sum() / denom)
            rows.append({"method": method, "horizon": h, "variable": v, "mae": mae, "baseline_mae": bmae, "delta_vs_baseline": mae - bmae})
    return rows


def help_hurt_after_activation(cache: Mapping[str, Any], std: torch.Tensor, pred: torch.Tensor, base: torch.Tensor, traces: Sequence[Mapping[str, Any]], expert: str) -> dict[str, Any]:
    cm = sample_mae(pred, cache["targets"].to(torch.float32), cache["target_masks"].to(torch.bool), std)
    bm = sample_mae(base, cache["targets"].to(torch.float32), cache["target_masks"].to(torch.bool), std)
    active = torch.tensor([bool(r[f"activated_{expert}"]) for r in traces], dtype=torch.bool)
    if not bool(active.any()):
        return {f"{expert}_active_windows": 0, f"{expert}_active_help_rate": 0.0, f"{expert}_active_mean_delta": 0.0}
    diff = cm[active] - bm[active]
    return {f"{expert}_active_windows": int(active.sum()), f"{expert}_active_help_rate": float((diff < 0).to(torch.float32).mean()), f"{expert}_active_mean_delta": float(diff.mean())}


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
    lines = [
        "# Expanded Expert Pool COSTAR-TS",
        "",
        "## Protocol",
        "",
        f"- Frozen fixed-three baseline: `{BASELINE_NAME}`.",
        "- Optional experts: DLinear and ModernTCN.",
        "- Optional weights are nonnegative and capped; no unconstrained stacking.",
        "- Hyperparameters selected on router-train chronological folds only.",
        "- Test cache was not loaded.",
        "",
        "## Selected Configs",
        "",
    ]
    for scenario, item in report["selected_configs"].items():
        lines.append(f"- `{scenario}`: `{item['selected']['name']}`")
    lines.extend(
        [
            "",
            "## Validation",
            "",
            "| Method | MAE | MSE | Improvement vs 0.363642 | Aggregate CI |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in report["validation_seed_summary"]:
        ci = report["aggregate_bootstrap_ci"].get(row["method"], {})
        ci_s = "n/a" if not ci else f"[{ci['ci95_low']:.6f}, {ci['ci95_high']:.6f}]"
        lines.append(f"| `{row['method']}` | `{row['mae_mean']:.6f}` | `{row['mse_mean']:.6f}` | `{row['improvement_vs_0.363642']:.6f}` | `{ci_s}` |")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            report["promotion_decision"],
            "",
            "## Reproduce",
            "",
            "```powershell",
            report["reproduce_command"],
            "```",
        ]
    )
    (out_dir / "expanded_pool_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", default="cache/costarts_walkforward/router_train_20_60_cache.pt")
    parser.add_argument("--val-cache", default="cache/costarts_walkforward/router_val_60_80_cache.pt")
    parser.add_argument("--normalizer-checkpoint", default="checkpoints/costarts_walkforward/final_60/DLinear/best_expert.pt")
    parser.add_argument("--out-dir", default="experiments/expanded_expert_pool_costar")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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
    train_starts = train_cache["absolute_window_starts"].to(torch.long)
    val_starts = val_cache["absolute_window_starts"].to(torch.long)
    if int(val_starts[-1]) + int(val_cache["forecast_horizon"]) > 11520:
        raise ValueError("Validation cache crosses split end")
    if not bool(torch.all(train_starts[1:] > train_starts[:-1])) or not bool(torch.all(val_starts[1:] > val_starts[:-1])):
        raise ValueError("Caches must be chronological")

    train_base, _ = fixed_current_best_prediction(train_cache, train_cache, std, 7, device)
    folds = train_folds(int(train_cache["num_windows"]))
    configs = grid()
    configs_by_name = {c.name: c for c in configs}
    leaderboard_rows, fold_detail_rows = grid_eval_cached(train_cache, std, train_base, configs, folds)
    write_csv(out_dir / "router_train_fold_leaderboard.csv", sorted(leaderboard_rows, key=lambda r: (r["scenario"], float(r["fold_mae_mean"]))))
    write_csv(out_dir / "router_train_fold_details.csv", fold_detail_rows)
    selected = select_with_one_se(leaderboard_rows, configs_by_name)
    write_json(out_dir / "selected_configs.json", selected)

    # Router-train diagnostic: how often each optional expert beats the frozen baseline.
    d_train, m_train = optional_predictions(train_cache)
    target_train = train_cache["targets"].to(torch.float32)
    mask_train = train_cache["target_masks"].to(torch.bool)
    b_mae = sample_mae(train_base, target_train, mask_train, std)
    d_mae = sample_mae(d_train, target_train, mask_train, std)
    m_mae = sample_mae(m_train, target_train, mask_train, std)
    diagnostic = {
        "DLinear_window_win_rate_vs_baseline": float((d_mae < b_mae).to(torch.float32).mean()),
        "ModernTCN_window_win_rate_vs_baseline": float((m_mae < b_mae).to(torch.float32).mean()),
        "DLinear_mean_mae_minus_baseline": float(d_mae.mean() - b_mae.mean()),
        "ModernTCN_mean_mae_minus_baseline": float(m_mae.mean() - b_mae.mean()),
    }
    write_json(out_dir / "router_train_optional_expert_diagnostic.json", diagnostic)

    validation_rows = []
    trace_rows = []
    axis_rows = []
    hv_rows = []
    per_method_per_window: dict[str, list[torch.Tensor]] = defaultdict(list)
    baseline_per_window_by_seed: dict[int, torch.Tensor] = {}
    d_val, m_val = optional_predictions(val_cache)
    val_target = val_cache["targets"].to(torch.float32)
    val_mask = val_cache["target_masks"].to(torch.bool)
    init_base_err = normalized_abs_error(train_base, target_train, mask_train, std)
    init_d_err = normalized_abs_error(d_train, target_train, mask_train, std)
    init_m_err = normalized_abs_error(m_train, target_train, mask_train, std)

    equal5 = val_cache["prediction_stack"].to(torch.float32).mean(dim=-1)
    equal5_metrics = metrics(val_cache, std, equal5)
    validation_rows.append({"method": "equal5_reference", "seed": -1, "mae": equal5_metrics["mae"], "mse": equal5_metrics["mse"], "diff_vs_0.363642": equal5_metrics["mae"] - BASELINE_MAE})

    for seed in SEEDS:
        base_val, _ = fixed_current_best_prediction(val_cache, train_cache, std, seed, device)
        bm = metrics(val_cache, std, base_val)
        baseline_per_window_by_seed[seed] = bm["per_window_mae"]
        per_method_per_window["baseline_fixed3_hv"].append(bm["per_window_mae"])
        validation_rows.append({"method": "baseline_fixed3_hv", "seed": seed, "mae": bm["mae"], "mse": bm["mse"], "diff_vs_0.363642": bm["mae"] - BASELINE_MAE})
        torch.save(bm["per_window_mae"], out_dir / "per_window" / f"baseline_seed{seed}.pt")
        for scenario in ("dlinear_only", "moderntcn_only", "both"):
            config = configs_by_name[selected[scenario]["selected"]["name"]]
            pred, extra, traces = run_causal_specialists(
                val_starts,
                base_val,
                d_val,
                m_val,
                val_target,
                val_mask,
                std,
                config,
                init_base_err,
                init_d_err,
                init_m_err,
                trace_prefix={"method": scenario, "seed": seed, "config": config.name},
            )
            mm = metrics(val_cache, std, pred)
            boot = paired_bootstrap(mm["per_window_mae"], bm["per_window_mae"], seed=seed, samples=3000)
            method = f"expanded_{scenario}"
            extra2 = {
                **extra,
                **activation_turnover(traces, "DLinear"),
                **activation_turnover(traces, "ModernTCN"),
                **help_hurt_after_activation(val_cache, std, pred, base_val, traces, "DLinear"),
                **help_hurt_after_activation(val_cache, std, pred, base_val, traces, "ModernTCN"),
            }
            validation_rows.append({"method": method, "seed": seed, "mae": mm["mae"], "mse": mm["mse"], "diff_vs_0.363642": mm["mae"] - BASELINE_MAE, **boot, **extra2})
            trace_rows.extend([{**r, "method": method} for r in traces])
            axis_rows.extend(per_axis_rows(val_cache, std, pred, base_val, f"{method}_seed{seed}"))
            hv_rows.extend(per_hv_rows(val_cache, std, pred, base_val, f"{method}_seed{seed}"))
            per_method_per_window[method].append(mm["per_window_mae"])
            torch.save(mm["per_window_mae"], out_dir / "per_window" / f"{method}_seed{seed}.pt")

    summary = aggregate_seed_summary([r for r in validation_rows if int(r["seed"]) >= 0])
    agg_boot: dict[str, dict[str, Any]] = {}
    for method, chunks in per_method_per_window.items():
        cand = torch.cat(chunks)
        base = torch.cat([baseline_per_window_by_seed[seed] for seed in SEEDS])
        agg_boot[method] = paired_bootstrap(cand, base, seed=20260812, samples=5000)
    write_csv(out_dir / "validation_per_seed_results.csv", validation_rows)
    write_csv(out_dir / "validation_seed_summary.csv", summary)
    write_csv(out_dir / "validation_activation_traces.csv", trace_rows)
    write_csv(out_dir / "per_axis_mae.csv", axis_rows)
    write_csv(out_dir / "per_horizon_variable_mae.csv", hv_rows)

    # Aggregate worst HV regression by method across seeds.
    hv_group: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    for row in hv_rows:
        method = row["method"].rsplit("_seed", 1)[0]
        hv_group[(method, int(row["horizon"]), int(row["variable"]))].append(float(row["delta_vs_baseline"]))
    worst = []
    for method in sorted({k[0] for k in hv_group}):
        candidates = []
        for (m, h, v), vals in hv_group.items():
            if m == method:
                candidates.append({"method": m, "horizon": h, "variable": v, "delta_vs_baseline_mean": float(np.mean(vals))})
        worst.append(max(candidates, key=lambda r: r["delta_vs_baseline_mean"]))

    best_expanded = min([r for r in summary if r["method"].startswith("expanded_")], key=lambda r: r["mae_mean"])
    best_boot = agg_boot[best_expanded["method"]]
    selected_folds = selected[best_expanded["method"].replace("expanded_", "")]["selected"]
    fold_consistent = int(selected_folds["fold_wins"]) >= 3
    reliable = bool(best_boot["ci_excludes_zero"] and best_boot["ci95_high"] < 0)
    beats = float(best_expanded["mae_mean"]) < BASELINE_MAE
    major_regression = any(r["method"] == best_expanded["method"] and float(r["delta_vs_baseline_mean"]) > 0.001 for r in worst)
    promote = bool(beats and reliable and fold_consistent and not major_regression)
    report = {
        "baseline_reference": {"name": BASELINE_NAME, "mae": BASELINE_MAE, "mse": BASELINE_MSE},
        "selected_configs": selected,
        "router_train_optional_expert_diagnostic": diagnostic,
        "validation_seed_summary": summary,
        "aggregate_bootstrap_ci": agg_boot,
        "worst_horizon_variable_regression": worst,
        "best_expanded_method": best_expanded,
        "promotion_decision": "Promote expanded pool." if promote else "Do not promote expanded pool; retain the fixed-three horizon-variable baseline.",
        "promotion_checks": {"beats_baseline": beats, "ci_excludes_zero": reliable, "fold_consistent": fold_consistent, "no_major_regression": not major_regression},
        "safety": {"validation_used_for_selection": False, "test_cache_loaded": False, "causality_rule": "old_start + prediction_length <= current_start"},
        "runtime_sec": time.time() - t0,
        "reproduce_command": f"python experiments\\expanded_expert_pool_costar\\run_expanded_expert_pool.py --device {args.device}",
    }
    write_json(out_dir / "final_report.json", report)
    make_report(out_dir, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
