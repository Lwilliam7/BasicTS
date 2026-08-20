"""Diagnostic oracle / routing-headroom experiment.

NOT a COSTAR variant. This experiment never feeds oracle results into router
training, hyperparameter selection, expert selection, or normal COSTAR
predictions -- it only measures how much MAE/MSE headroom would exist if
expert selection were perfect (using future targets, which is only legal
because this is a diagnostic upper bound, never a deployable method).

Reuses, unmodified: the dataset bundles, expert core, std normalizer, and
"Current COSTAR" (Online HxV COSTAR) prediction path from
`experiments/frozen_hv_costar/run_frozen_hv_costar.py`, which in turn reuses
the existing cached expert predictions/targets/splits/selected cores. No
expert is retrained. No existing cache or historical result file is modified.

Datasets: ETTh1, ETTh2, ETTm1, Weather, Electricity -- router_val only. No
test cache is accessed.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.frozen_hv_costar.run_frozen_hv_costar import (  # noqa: E402
    LOADERS,
    Bundle,
    best_single_expert,
    equal_fixed,
    metric_values,
    online_hv_prediction,
    refuse_test,
)
from experiments.oracle_weight_tournament.run_tournament import sample_mae, sample_mse  # noqa: E402


OUT_DIR = ROOT / "experiments/oracle_routing_headroom"
CONVEX_GRID_STEP = 0.1  # illustrative, coarse; optional oracle only
CHRONO_REGIONS = 8


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Alignment verification -- fail loudly on any mismatch.
# ---------------------------------------------------------------------------


def verify_alignment(bundle: Bundle, forecasts: torch.Tensor, per_location_error: torch.Tensor) -> dict[str, Any]:
    cache = bundle.val_cache
    n = int(cache["num_windows"])
    starts = cache["absolute_window_starts"].to(torch.long)
    target = cache["targets"]
    mask = cache["target_masks"]
    if not bool(torch.all(starts[1:] > starts[:-1])):
        raise AssertionError(f"{bundle.dataset}: absolute_window_starts are not strictly chronological")
    if forecasts.shape[0] != n or target.shape[0] != n or mask.shape[0] != n:
        raise AssertionError(f"{bundle.dataset}: window-count mismatch: forecasts={forecasts.shape[0]} target={target.shape[0]} mask={mask.shape[0]} declared_n={n}")
    if forecasts.shape[:3] != target.shape[:3] or forecasts.shape[:3] != mask.shape[:3]:
        raise AssertionError(f"{bundle.dataset}: (window,horizon,variable) shape mismatch: forecasts={tuple(forecasts.shape)} target={tuple(target.shape)} mask={tuple(mask.shape)}")
    if forecasts.shape[-1] != len(bundle.expert_idx):
        raise AssertionError(f"{bundle.dataset}: expert-count mismatch: forecasts last dim={forecasts.shape[-1]} expert_idx={len(bundle.expert_idx)}")
    if tuple(per_location_error.shape) != tuple(forecasts.shape):
        raise AssertionError(f"{bundle.dataset}: per_location_error shape {tuple(per_location_error.shape)} != forecasts shape {tuple(forecasts.shape)}")
    if not torch.isfinite(forecasts).all():
        raise AssertionError(f"{bundle.dataset}: non-finite values in expert forecasts")
    return {"dataset": bundle.dataset, "num_windows": n, "horizon": int(cache["forecast_horizon"]), "num_variables": int(forecasts.shape[2]), "num_experts": int(forecasts.shape[3]), "result": "PASS"}


# ---------------------------------------------------------------------------
# Oracles (diagnostic only -- use targets)
# ---------------------------------------------------------------------------


def masked_agg_error(per_location_error: torch.Tensor, mask: torch.Tensor, dims: Sequence[int]) -> torch.Tensor:
    """Sum per_location_error and mask over `dims`, divide -- a masked mean
    that leaves the remaining axes (including the trailing expert axis, if
    present in per_location_error but not in mask) intact."""
    denom = mask.to(torch.float32).sum(dim=tuple(dims)).clamp_min(1.0)
    num = per_location_error.sum(dim=tuple(dims))
    while denom.ndim < num.ndim:
        denom = denom.unsqueeze(-1)
    return num / denom


def window_oracle(forecasts: torch.Tensor, per_location_error: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    window_err = masked_agg_error(per_location_error, mask, dims=(1, 2))  # [N,E]
    winner = window_err.argmin(dim=-1)  # [N]
    n, h, v, _ = forecasts.shape
    idx = winner.view(n, 1, 1, 1).expand(n, h, v, 1)
    pred = forecasts.gather(-1, idx).squeeze(-1)
    return pred, winner, window_err


def variable_oracle(forecasts: torch.Tensor, per_location_error: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    variable_err = masked_agg_error(per_location_error, mask, dims=(1,))  # [N,V,E]
    winner = variable_err.argmin(dim=-1)  # [N,V]
    n, h, v, _ = forecasts.shape
    idx = winner.view(n, 1, v, 1).expand(n, h, v, 1)
    pred = forecasts.gather(-1, idx).squeeze(-1)
    return pred, winner


def hxv_oracle(forecasts: torch.Tensor, per_location_error: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    winner = per_location_error.argmin(dim=-1)  # [N,H,V]
    pred = forecasts.gather(-1, winner.unsqueeze(-1)).squeeze(-1)
    return pred, winner


def simplex_grid(num_experts: int, step: float) -> list[tuple[float, ...]]:
    steps = round(1.0 / step)
    points = []
    for combo in itertools.product(range(steps + 1), repeat=num_experts - 1):
        if sum(combo) <= steps:
            last = steps - sum(combo)
            weights = tuple(c * step for c in combo) + (last * step,)
            points.append(weights)
    return points


def convex_oracle(bundle: Bundle, forecasts: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, std: torch.Tensor) -> dict[str, Any] | None:
    num_experts = forecasts.shape[-1]
    if num_experts > 4:
        return None  # grid search infeasible at this resolution beyond ~4 experts; skip rather than silently degrade
    grid = simplex_grid(num_experts, CONVEX_GRID_STEP)
    n = forecasts.shape[0]
    best_mae = torch.full((n,), float("inf"))
    best_weights_idx = torch.zeros(n, dtype=torch.long)
    for gi, weights in enumerate(grid):
        w = torch.tensor(weights, dtype=torch.float32).view(1, 1, 1, -1)
        pred = (forecasts * w).sum(dim=-1)
        per_window_mae = sample_mae(pred, target, mask, std)
        improved = per_window_mae < best_mae
        best_mae = torch.where(improved, per_window_mae, best_mae)
        best_weights_idx = torch.where(improved, torch.full_like(best_weights_idx, gi), best_weights_idx)
    chosen_weights = torch.tensor([grid[i] for i in best_weights_idx.tolist()], dtype=torch.float32)
    pred = (forecasts * chosen_weights.view(n, 1, 1, -1)).sum(dim=-1)
    m_mae = sample_mae(pred, target, mask, std)
    m_mse = sample_mse(pred, target, mask, std)
    return {
        "mae": float(m_mae.mean()),
        "mse": float(m_mse.mean()),
        "grid_step": CONVEX_GRID_STEP,
        "grid_size": len(grid),
        "note": f"approximate convex oracle via {CONVEX_GRID_STEP}-resolution simplex grid search per window; diagnostic only",
    }


# ---------------------------------------------------------------------------
# Winner dynamics + regret + concentration
# ---------------------------------------------------------------------------


def run_lengths(sequence: Sequence[int]) -> list[int]:
    return [len(list(g)) for _, g in itertools.groupby(sequence)]


def pairwise_error_correlation(window_err: torch.Tensor, core_names: Sequence[str]) -> list[dict[str, Any]]:
    corr = torch.corrcoef(window_err.T)
    rows = []
    for i, a in enumerate(core_names):
        for j, b in enumerate(core_names):
            if j <= i:
                continue
            rows.append({"expert_a": a, "expert_b": b, "pearson_correlation": float(corr[i, j])})
    return rows


def winner_dynamics(bundle: Bundle, winner: torch.Tensor, window_err: torch.Tensor) -> dict[str, Any]:
    n, k = window_err.shape
    seq = winner.tolist()
    counts = torch.bincount(winner, minlength=k)
    fractions = {bundle.core_names[i]: float(counts[i]) / n for i in range(k)}
    changes = int((winner[1:] != winner[:-1]).sum())
    change_rate = changes / max(n - 1, 1)

    trans_counts = torch.zeros(k, k, dtype=torch.long)
    for a, b in zip(seq[:-1], seq[1:]):
        trans_counts[a, b] += 1
    row_sums = trans_counts.sum(dim=1, keepdim=True).clamp_min(1)
    trans_probs = trans_counts.float() / row_sums

    runs = run_lengths(seq)
    sorted_err, _ = torch.sort(window_err, dim=-1)
    gap_best_second = sorted_err[:, 1] - sorted_err[:, 0]

    return {
        "dataset": bundle.dataset,
        "num_windows": n,
        "num_experts": k,
        "expert_names": list(bundle.core_names),
        "win_fraction_by_expert": fractions,
        "dominant_expert": bundle.core_names[int(counts.argmax())],
        "dominant_expert_fraction": float(counts.max()) / n,
        "num_winner_changes": changes,
        "winner_change_rate": change_rate,
        "transition_matrix_counts": trans_counts.tolist(),
        "transition_matrix_probs": trans_probs.tolist(),
        "run_lengths": runs,
        "run_length_mean": float(torch.tensor(runs, dtype=torch.float32).mean()),
        "run_length_median": float(torch.tensor(runs, dtype=torch.float32).median()),
        "run_length_max": int(max(runs)),
        "mean_gap_best_vs_second": float(gap_best_second.mean()),
        "median_gap_best_vs_second": float(gap_best_second.median()),
        "pairwise_error_correlation": pairwise_error_correlation(window_err, bundle.core_names),
    }


def oracle_regret(costar_pred: torch.Tensor, oracle_pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, std: torch.Tensor) -> dict[str, Any]:
    costar_window_mae = sample_mae(costar_pred, target, mask, std)
    oracle_window_mae = sample_mae(oracle_pred, target, mask, std)
    regret = costar_window_mae - oracle_window_mae
    relative = regret / costar_window_mae.clamp_min(1e-8)
    return {
        "mean_regret": float(regret.mean()),
        "median_regret": float(regret.median()),
        "p90_regret": float(torch.quantile(regret, 0.9)),
        "fraction_regret_gt_0": float((regret > 0).to(torch.float32).mean()),
        "fraction_relative_regret_gt_1pct": float((relative > 0.01).to(torch.float32).mean()),
        "regret_per_window": regret,
        "costar_window_mae": costar_window_mae,
        "oracle_window_mae": oracle_window_mae,
    }


def regret_concentration(
    costar_pred: torch.Tensor, oracle_pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, std: torch.Tensor, num_regions: int = CHRONO_REGIONS
) -> dict[str, list[dict[str, Any]]]:
    stdv = std.view(1, 1, -1)
    mask_f = mask.to(torch.float32)
    costar_loc = ((costar_pred - target) / stdv).abs() * mask_f
    oracle_loc = ((oracle_pred - target) / stdv).abs() * mask_f
    regret_loc = costar_loc - oracle_loc  # [N,H,V]
    n, h, v = regret_loc.shape

    def masked_mean(x: torch.Tensor, m: torch.Tensor, dims: Sequence[int]) -> torch.Tensor:
        return (x * m).sum(dim=tuple(dims)) / m.sum(dim=tuple(dims)).clamp_min(1.0)

    by_variable = masked_mean(regret_loc, mask_f, dims=(0, 1))  # [V]
    by_horizon = masked_mean(regret_loc, mask_f, dims=(0, 2))  # [H]
    bounds = [i * n // num_regions for i in range(num_regions + 1)]
    by_region = []
    for r in range(num_regions):
        lo, hi = bounds[r], bounds[r + 1]
        val = masked_mean(regret_loc[lo:hi], mask_f[lo:hi], dims=(0, 1, 2))
        by_region.append({"region": r, "window_lo": lo, "window_hi": hi - 1, "mean_regret": float(val)})
    return {
        "by_variable": [{"variable": i, "mean_regret": float(by_variable[i])} for i in range(v)],
        "by_horizon": [{"horizon_index": i, "mean_regret": float(by_horizon[i])} for i in range(h)],
        "by_chronological_region": by_region,
    }


# ---------------------------------------------------------------------------
# Per-dataset evaluation
# ---------------------------------------------------------------------------


def evaluate_dataset(dataset: str) -> dict[str, Any]:
    bundle = LOADERS[dataset]()
    forecasts = bundle.forecasts_fn(bundle.val_cache, bundle.expert_idx)
    per_loc_err = bundle.per_location_error_fn(bundle.val_cache, bundle.expert_idx, bundle.std)
    target = bundle.val_cache["targets"].to(torch.float32)
    mask = bundle.val_cache["target_masks"].to(torch.bool)

    alignment = verify_alignment(bundle, forecasts, per_loc_err)

    best_single_pred, best_single_extra = best_single_expert(bundle)
    equal_pred, _ = equal_fixed(bundle)
    costar_pred, costar_extra = online_hv_prediction(bundle)

    window_pred, window_winner, window_err = window_oracle(forecasts, per_loc_err, mask)
    variable_pred, variable_winner = variable_oracle(forecasts, per_loc_err, mask)
    hxv_pred, hxv_winner = hxv_oracle(forecasts, per_loc_err)
    convex = convex_oracle(bundle, forecasts, target, mask, bundle.std)

    methods = {
        "best_single_expert": best_single_pred,
        "equal_fixed": equal_pred,
        "current_costar": costar_pred,
        "window_oracle": window_pred,
        "variable_oracle": variable_pred,
        "hxv_oracle": hxv_pred,
    }
    result_rows = []
    metrics: dict[str, dict[str, Any]] = {}
    for method, pred in methods.items():
        m = metric_values(bundle, pred)
        metrics[method] = m
        result_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "mae": m["mae"],
                "mse": m["mse"],
                "noncausal_oracle": method in ("window_oracle", "variable_oracle", "hxv_oracle"),
                "expert_set": "+".join(bundle.core_names),
            }
        )
    if convex is not None:
        result_rows.append({"dataset": dataset, "method": "convex_oracle_approx", "mae": convex["mae"], "mse": convex["mse"], "noncausal_oracle": True, "expert_set": "+".join(bundle.core_names), **{k: v for k, v in convex.items() if k not in ("mae", "mse")}})

    headroom_rows = []
    for oracle_name in ("window_oracle", "variable_oracle", "hxv_oracle"):
        for baseline_name in ("current_costar", "equal_fixed"):
            base_mae = metrics[baseline_name]["mae"]
            oracle_mae = metrics[oracle_name]["mae"]
            abs_headroom = base_mae - oracle_mae
            headroom_rows.append(
                {
                    "dataset": dataset,
                    "baseline": baseline_name,
                    "oracle": oracle_name,
                    "baseline_mae": base_mae,
                    "oracle_mae": oracle_mae,
                    "absolute_headroom_mae": abs_headroom,
                    "relative_headroom_pct": 100.0 * abs_headroom / base_mae,
                }
            )

    dynamics = winner_dynamics(bundle, window_winner, window_err)
    regret = oracle_regret(costar_pred, window_pred, target, mask, bundle.std)
    concentration = regret_concentration(costar_pred, window_pred, target, mask, bundle.std)

    per_window_rows = [
        {
            "dataset": dataset,
            "window_index": i,
            "absolute_window_start": int(bundle.val_cache["absolute_window_starts"][i]),
            "window_oracle_winner": bundle.core_names[int(window_winner[i])],
            **{f"error_{bundle.core_names[e]}": float(window_err[i, e]) for e in range(len(bundle.core_names))},
            "costar_window_mae": float(regret["costar_window_mae"][i]),
            "oracle_window_mae": float(regret["oracle_window_mae"][i]),
            "regret": float(regret["regret_per_window"][i]),
        }
        for i in range(int(bundle.val_cache["num_windows"]))
    ]

    cache_hashes = {"router_train": None, "router_val": None}  # populated by caller with real paths/hashes

    return {
        "dataset": dataset,
        "core": bundle.core_names,
        "alignment": alignment,
        "result_rows": result_rows,
        "headroom_rows": headroom_rows,
        "dynamics": dynamics,
        "regret_summary": {k: v for k, v in regret.items() if not isinstance(v, torch.Tensor)},
        "concentration": concentration,
        "per_window_rows": per_window_rows,
        "best_single_expert_name": best_single_extra["selected_expert"],
        "costar_num_causal_updates": costar_extra["num_causal_updates"],
    }


DATASET_CACHE_PATHS = {
    "ETTh1": ("cache/costarts_walkforward/router_train_20_60_cache.pt", "cache/costarts_walkforward/router_val_60_80_cache.pt"),
    "ETTh2": ("cache/costarts_fresh/ETTh2_96_12/router_train_cache.pt", "cache/costarts_fresh/ETTh2_96_12/router_val_cache.pt"),
    "ETTm1": ("cache/costarts_walkforward_ETTm1/router_train_20_60_cache.pt", "cache/costarts_walkforward_ETTm1/router_val_60_80_cache.pt"),
    "Weather": ("cache/costarts_walkforward_Weather/router_train_20_60_cache.pt", "cache/costarts_walkforward_Weather/router_val_60_80_cache.pt"),
    "Electricity": ("cache/costarts_walkforward_Electricity/router_train_20_60_cache.pt", "cache/costarts_walkforward_Electricity/router_val_60_80_cache.pt"),
}


def make_report(out_dir: Path, report: Mapping[str, Any]) -> None:
    lines = [
        "# Oracle Routing Headroom Diagnostic",
        "",
        "**Diagnostic only.** Oracle methods below use validation targets to select the "
        "best expert after the fact -- they are not deployable and are never used to "
        "train, tune, or select anything in COSTAR. Labeled `noncausal_oracle=true` "
        "everywhere they appear in the machine-readable outputs.",
        "",
        "## Main table: MAE / MSE",
        "",
        "| Dataset | Best Single | Equal Fixed | COSTAR | Window Oracle | Variable Oracle | HxV Oracle |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for ds, d in report["datasets"].items():
        by = {r["method"]: r for r in d["result_rows"]}
        lines.append(
            "| {ds} | `{a[mae]:.6f}`/`{a[mse]:.6f}` | `{b[mae]:.6f}`/`{b[mse]:.6f}` | `{c[mae]:.6f}`/`{c[mse]:.6f}` | `{d[mae]:.6f}`/`{d[mse]:.6f}` | `{e[mae]:.6f}`/`{e[mse]:.6f}` | `{f[mae]:.6f}`/`{f[mse]:.6f}` |".format(
                ds=ds, a=by["best_single_expert"], b=by["equal_fixed"], c=by["current_costar"], d=by["window_oracle"], e=by["variable_oracle"], f=by["hxv_oracle"]
            )
        )
    lines += ["", "## Headroom vs COSTAR (absolute MAE / relative %)", ""]
    lines.append("| Dataset | -> Window Oracle | -> Variable Oracle | -> HxV Oracle |")
    lines.append("|---|---:|---:|---:|")
    for ds, d in report["datasets"].items():
        by = {(r["baseline"], r["oracle"]): r for r in d["headroom_rows"]}
        lines.append(
            "| {ds} | `{w[absolute_headroom_mae]:+.6f}` ({w[relative_headroom_pct]:.2f}%) | `{v[absolute_headroom_mae]:+.6f}` ({v[relative_headroom_pct]:.2f}%) | `{h[absolute_headroom_mae]:+.6f}` ({h[relative_headroom_pct]:.2f}%) |".format(
                ds=ds,
                w=by[("current_costar", "window_oracle")],
                v=by[("current_costar", "variable_oracle")],
                h=by[("current_costar", "hxv_oracle")],
            )
        )
    lines += ["", "## Winner dynamics (window oracle)", ""]
    lines.append("| Dataset | % winner changes | Dominant expert (%) | Mean run length | Mean COSTAR regret | Median regret | P90 regret |")
    lines.append("|---|---:|---|---:|---:|---:|---:|")
    for ds, d in report["datasets"].items():
        dyn = d["dynamics"]
        reg = d["regret_summary"]
        lines.append(
            f"| {ds} | {dyn['winner_change_rate']*100:.2f}% | {dyn['dominant_expert']} ({dyn['dominant_expert_fraction']*100:.1f}%) | "
            f"{dyn['run_length_mean']:.2f} | `{reg['mean_regret']:+.6f}` | `{reg['median_regret']:+.6f}` | `{reg['p90_regret']:+.6f}` |"
        )
    lines += ["", "## Win fraction by expert", ""]
    for ds, d in report["datasets"].items():
        frac = ", ".join(f"{k}={v*100:.1f}%" for k, v in d["dynamics"]["win_fraction_by_expert"].items())
        lines.append(f"- **{ds}**: {frac}")
    lines += ["", "## Hard rule compliance", "", "```text", "TEST SET ACCESSED: NO", "TEST CACHE LOADED: NO", "TEST METRICS COMPUTED: NO", "```"]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    start = time.time()
    report: dict[str, Any] = {"experiment": "oracle_routing_headroom", "created_at_utc": datetime.now(timezone.utc).isoformat(), "datasets": {}}
    all_results, all_headroom, all_per_window, all_dynamics = [], [], [], []
    all_regret_by_variable, all_regret_by_horizon, all_regret_by_region, all_transitions = [], [], [], []
    provenance: dict[str, Any] = {}

    for dataset in LOADERS:
        print(f"[oracle-headroom] {dataset}: evaluating...", flush=True)
        train_rel, val_rel = DATASET_CACHE_PATHS[dataset]
        train_path, val_path = ROOT / train_rel, ROOT / val_rel
        refuse_test(train_path)
        refuse_test(val_path)
        result = evaluate_dataset(dataset)
        report["datasets"][dataset] = {k: v for k, v in result.items() if k != "per_window_rows"}
        all_results.extend(result["result_rows"])
        all_headroom.extend(result["headroom_rows"])
        all_per_window.extend(result["per_window_rows"])
        all_dynamics.append({"dataset": dataset, **{k: v for k, v in result["dynamics"].items() if k not in ("transition_matrix_counts", "transition_matrix_probs", "run_lengths", "pairwise_error_correlation", "win_fraction_by_expert")}})
        for i, name_a in enumerate(result["dynamics"]["expert_names"]):
            for j, name_b in enumerate(result["dynamics"]["expert_names"]):
                all_transitions.append({"dataset": dataset, "from_expert": name_a, "to_expert": name_b, "count": result["dynamics"]["transition_matrix_counts"][i][j], "probability": result["dynamics"]["transition_matrix_probs"][i][j]})
        for row in result["concentration"]["by_variable"]:
            all_regret_by_variable.append({"dataset": dataset, **row})
        for row in result["concentration"]["by_horizon"]:
            all_regret_by_horizon.append({"dataset": dataset, **row})
        for row in result["concentration"]["by_chronological_region"]:
            all_regret_by_region.append({"dataset": dataset, **row})
        provenance[dataset] = {
            "core": result["core"],
            "router_train_cache": train_rel,
            "router_val_cache": val_rel,
            "router_train_sha256": sha256_file(train_path),
            "router_val_sha256": sha256_file(val_path),
            "best_single_expert": result["best_single_expert_name"],
        }
        print(f"[oracle-headroom] {dataset}: done. core={'+'.join(result['core'])}", flush=True)

    report["runtime_sec"] = time.time() - start
    report["test_set_accessed"] = False
    report["provenance"] = provenance

    write_json(OUT_DIR / "results.json", report)
    write_csv(OUT_DIR / "results.csv", all_results)
    write_csv(OUT_DIR / "headroom.csv", all_headroom)
    write_csv(OUT_DIR / "per_window_winner_and_error.csv", all_per_window)
    write_csv(OUT_DIR / "winner_dynamics_summary.csv", all_dynamics)
    write_csv(OUT_DIR / "transition_matrix.csv", all_transitions)
    write_csv(OUT_DIR / "regret_by_variable.csv", all_regret_by_variable)
    write_csv(OUT_DIR / "regret_by_horizon.csv", all_regret_by_horizon)
    write_csv(OUT_DIR / "regret_by_chronological_region.csv", all_regret_by_region)
    write_json(OUT_DIR / "provenance.json", provenance)
    make_report(OUT_DIR, report)

    print("TEST SET ACCESSED: NO")
    print(json.dumps({"runtime_sec": report["runtime_sec"], "datasets": list(report["datasets"].keys())}, indent=2))


if __name__ == "__main__":
    main()
