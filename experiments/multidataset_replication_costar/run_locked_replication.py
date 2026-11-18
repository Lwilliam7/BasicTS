"""Locked multi-dataset COSTAR-TS replication where caches permit.

Only ETTh2 has the requested non-test expert caches in this workspace.  The
ETTh1 static neural winner used inside the full current baseline is not
available for ETTh2, so this script reports a limited locked specialist
transfer over the available fixed-three cache baseline rather than claiming a
full reproduction of the ETTh1 primary model.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.chronological_adaptive_costar.run_chronological_adaptive_costar import paired_bootstrap  # noqa: E402
from experiments.expanded_expert_pool_costar.run_expanded_expert_pool import (  # noqa: E402
    BASELINE_NAME,
    Config,
    grid,
    grid_eval_cached,
    optional_predictions,
    per_axis_rows,
    per_hv_rows,
    run_causal_specialists,
    select_with_one_se,
    train_folds,
)
from experiments.horizon_variable_adaptive_costar.run_hv_adaptive_costar import fixed3_forecasts  # noqa: E402
from experiments.oracle_weight_tournament.run_tournament import load_cache, sample_mae, sample_mse, weighted_forecast  # noqa: E402


REQUESTED_DATASETS = ("ETTh2", "ETTm1", "ETTm2", "Weather", "Electricity")
LOCKED_CONFIG = Config("both", "variable", 0.95, 0.10, 0.02, 96)


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


def load_std_flexible(path: Path, num_features: int) -> torch.Tensor:
    refuse_test(path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "scaler_std" in ckpt:
        return ckpt["scaler_std"].to(torch.float32).view(-1)
    if "scaler_stats" in ckpt and "std" in ckpt["scaler_stats"]:
        return ckpt["scaler_stats"]["std"].to(torch.float32).view(-1)
    return torch.ones(num_features, dtype=torch.float32)


def metrics(cache: Mapping[str, Any], std: torch.Tensor, pred: torch.Tensor) -> dict[str, Any]:
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.bool)
    mae = sample_mae(pred, target, mask, std)
    mse = sample_mse(pred, target, mask, std)
    return {"mae": float(mae.mean()), "mse": float(mse.mean()), "per_window_mae": mae, "per_window_mse": mse}


def normalized_abs_error(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return ((pred - target) / std.view(1, 1, -1)).abs() * mask.to(torch.float32)


def equal_fixed3_prediction(cache: Mapping[str, Any]) -> torch.Tensor:
    forecasts = fixed3_forecasts(cache)
    weights = torch.full((forecasts.shape[0], 3), 1.0 / 3.0)
    return weighted_forecast(forecasts, weights)


def single_expert_predictions(cache: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    names = list(cache["expert_names"])
    stack = cache["prediction_stack"].to(torch.float32)
    return {name: stack[..., idx] for idx, name in enumerate(names)}


def run_dataset(
    dataset: str,
    train_cache_path: Path,
    val_cache_path: Path,
    normalizer_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    train_cache = load_cache(train_cache_path, "router_train")
    val_cache = load_cache(val_cache_path, "router_val")
    std = load_std_flexible(normalizer_path, int(val_cache["num_features"]))
    train_base = equal_fixed3_prediction(train_cache)
    val_base = equal_fixed3_prediction(val_cache)
    folds = train_folds(int(train_cache["num_windows"]))
    configs = [c for c in grid() if c.scenario == "both"]
    cfg_by_name = {c.name: c for c in configs}
    leaderboard, fold_details = grid_eval_cached(train_cache, std, train_base, configs, folds)
    selected = select_with_one_se(leaderboard, cfg_by_name)["both"]
    selected_cfg = cfg_by_name[selected["selected"]["name"]]
    locked_cfg = LOCKED_CONFIG

    d_train, m_train = optional_predictions(train_cache)
    d_val, m_val = optional_predictions(val_cache)
    target_train = train_cache["targets"].to(torch.float32)
    mask_train = train_cache["target_masks"].to(torch.bool)
    target_val = val_cache["targets"].to(torch.float32)
    mask_val = val_cache["target_masks"].to(torch.bool)
    init_base_err = normalized_abs_error(train_base, target_train, mask_train, std)
    init_d_err = normalized_abs_error(d_train, target_train, mask_train, std)
    init_m_err = normalized_abs_error(m_train, target_train, mask_train, std)
    starts = val_cache["absolute_window_starts"].to(torch.long)

    rows = []
    traces = []
    axes = []
    hvs = []
    per_window: dict[str, torch.Tensor] = {}
    base_metrics = metrics(val_cache, std, val_base)
    rows.append({"dataset": dataset, "method": "equal_fixed3_available_baseline", "mae": base_metrics["mae"], "mse": base_metrics["mse"]})
    for expert_name, expert_pred in single_expert_predictions(val_cache).items():
        em = metrics(val_cache, std, expert_pred)
        boot = paired_bootstrap(em["per_window_mae"], base_metrics["per_window_mae"], seed=20260812, samples=5000)
        rows.append(
            {
                "dataset": dataset,
                "method": f"single_{expert_name}",
                "mae": em["mae"],
                "mse": em["mse"],
                "baseline_mae": base_metrics["mae"],
                "baseline_mse": base_metrics["mse"],
                "absolute_improvement_mae": base_metrics["mae"] - em["mae"],
                "percent_improvement_mae": 100.0 * (base_metrics["mae"] - em["mae"]) / base_metrics["mae"],
                **boot,
            }
        )
    for label, cfg in (("locked_etth1_expanded_both_limited", locked_cfg), ("selected_predefined_limited", selected_cfg)):
        pred, extra, tr = run_causal_specialists(
            starts,
            val_base,
            d_val,
            m_val,
            target_val,
            mask_val,
            std,
            cfg,
            init_base_err,
            init_d_err,
            init_m_err,
            trace_prefix={"dataset": dataset, "method": label, "config": cfg.name},
        )
        mm = metrics(val_cache, std, pred)
        bm = metrics(val_cache, std, val_base)
        boot = paired_bootstrap(mm["per_window_mae"], bm["per_window_mae"], seed=20260812, samples=5000)
        rows.append(
            {
                "dataset": dataset,
                "method": label,
                "mae": mm["mae"],
                "mse": mm["mse"],
                "baseline_mae": bm["mae"],
                "baseline_mse": bm["mse"],
                "absolute_improvement_mae": bm["mae"] - mm["mae"],
                "percent_improvement_mae": 100.0 * (bm["mae"] - mm["mae"]) / bm["mae"],
                **boot,
                **extra,
                **asdict(cfg),
                "config_name": cfg.name,
            }
        )
        traces.extend(tr)
        axes.extend(per_axis_rows(val_cache, std, pred, val_base, label))
        hvs.extend(per_hv_rows(val_cache, std, pred, val_base, label))
        per_window[label] = mm["per_window_mae"]

    write_csv(out_dir / f"{dataset}_fold_leaderboard.csv", sorted(leaderboard, key=lambda r: float(r["fold_mae_mean"])))
    write_csv(out_dir / f"{dataset}_fold_details.csv", fold_details)
    write_csv(out_dir / f"{dataset}_validation_results.csv", rows)
    write_csv(out_dir / f"{dataset}_activation_traces.csv", traces)
    write_csv(out_dir / f"{dataset}_per_axis_mae.csv", axes)
    write_csv(out_dir / f"{dataset}_per_horizon_variable_mae.csv", hvs)
    torch.save(per_window, out_dir / f"{dataset}_per_window.pt")

    hv_group: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    for row in hvs:
        hv_group[(row["method"], int(row["horizon"]), int(row["variable"]))].append(float(row["delta_vs_baseline"]))
    worst = []
    for method in sorted({key[0] for key in hv_group}):
        candidates = [{"method": m, "horizon": h, "variable": v, "delta_vs_baseline_mean": float(np.mean(vals))} for (m, h, v), vals in hv_group.items() if m == method]
        worst.append(max(candidates, key=lambda r: float(r["delta_vs_baseline_mean"])))
    return {
        "dataset": dataset,
        "status": "completed_limited_available_cache_replication",
        "limitation": "Full ETTh1 current-best HV baseline needs ETTh1-specific static neural winner artifacts; ETTh2 run uses equal fixed-three available-cache baseline.",
        "locked_config": {"name": locked_cfg.name, **asdict(locked_cfg)},
        "selected_config": selected,
        "validation_results": rows,
        "worst_horizon_variable_regression": worst,
        "test_cache_loaded": False,
    }


def make_report(out_dir: Path, report: Mapping[str, Any]) -> None:
    lines = [
        "# Locked Multi-Dataset Replication",
        "",
        "## Scope",
        "",
        "- Requested datasets: ETTh2, ETTm1, ETTm2, Weather, Electricity.",
        "- Available non-test frozen expert caches found: ETTh2 only.",
        "- No test cache was loaded.",
        "- ETTh2 replication is limited because the full ETTh1 static neural winner artifact is not available for ETTh2.",
        "",
        "## Results",
        "",
    ]
    for ds in report["datasets"]:
        if ds["status"].startswith("missing"):
            lines.append(f"- `{ds['dataset']}`: `{ds['status']}`.")
            continue
        lines.append(f"- `{ds['dataset']}`: completed limited available-cache replication.")
        for row in ds["validation_results"]:
            if row["method"] == "equal_fixed3_available_baseline":
                lines.append(f"  - `{row['method']}` MAE/MSE: `{row['mae']:.6f}` / `{row['mse']:.6f}`.")
            else:
                lines.append(
                    f"  - `{row['method']}` MAE/MSE: `{row['mae']:.6f}` / `{row['mse']:.6f}`, "
                    f"improvement `{row['absolute_improvement_mae']:.6f}`, CI `[{row['ci95_low']:.6f}, {row['ci95_high']:.6f}]`."
                )
    lines.extend(["", "## Reproduce", "", "```powershell", report["reproduce_command"], "```"])
    (out_dir / "replication_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="experiments/multidataset_replication_costar")
    args = parser.parse_args()
    t0 = time.time()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = []
    for dataset in REQUESTED_DATASETS:
        if dataset == "ETTh2":
            train_cache = ROOT / "cache/costarts_fresh/ETTh2_96_12/router_train_cache.pt"
            val_cache = ROOT / "cache/costarts_fresh/ETTh2_96_12/router_val_cache.pt"
            normalizer = ROOT / "checkpoints/costarts_fresh/ETTh2_96_12/clean_candidates/best_dlinear.pt"
            if train_cache.exists() and val_cache.exists() and normalizer.exists():
                datasets.append(run_dataset(dataset, train_cache, val_cache, normalizer, out_dir))
            else:
                datasets.append({"dataset": dataset, "status": "missing_required_cache_or_normalizer", "test_cache_loaded": False})
        else:
            datasets.append({"dataset": dataset, "status": "missing_existing_expert_caches", "test_cache_loaded": False})
    report = {
        "baseline_context": {
            "etth1_primary": "expanded_both over fixed-three horizon-variable baseline",
            "etth1_locked_specialist_config": LOCKED_CONFIG.name,
            "etth1_core_baseline": BASELINE_NAME,
        },
        "datasets": datasets,
        "runtime_sec": time.time() - t0,
        "safety": {"test_cache_loaded": False},
        "reproduce_command": "python experiments\\multidataset_replication_costar\\run_locked_replication.py",
    }
    write_json(out_dir / "final_report.json", report)
    make_report(out_dir, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
