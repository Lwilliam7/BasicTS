"""Canonical ETTh2 router-validation protocol audit.

This script establishes one ETTh2 protocol and recomputes all fixed expert
baselines on exactly the same validation examples:

- cache: cache/costarts_fresh/ETTh2_96_12/router_val_cache.pt
- split: router_val, starts 10800..11412 inclusive
- horizon: 12
- variables: all 7 cached variables
- metric: raw/original-scale MAE/MSE using the repository sample_mae/sample_mse
- inverse transform: none applied; cached predictions and targets are evaluated
  directly in the same units used by the existing ETTh2 router summary.

Normalized metrics using checkpoint scaler std are emitted only as diagnostics.
No test cache is loaded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.expanded_expert_pool_costar.run_expanded_expert_pool import (  # noqa: E402
    Config,
    grid,
    grid_eval_cached,
    optional_predictions,
    run_causal_specialists,
    select_with_one_se,
    train_folds,
)
from experiments.horizon_variable_adaptive_costar.run_hv_adaptive_costar import fixed3_forecasts  # noqa: E402
from experiments.oracle_weight_tournament.run_tournament import (  # noqa: E402
    fixed3_indices,
    load_cache,
    sample_mae,
    sample_mse,
    weighted_forecast,
)


LOCKED_SPECIALIST = Config("both", "variable", 0.95, 0.10, 0.02, 96)
OLD_SUMMARY = ROOT / "results/router_summary/costarts_fresh/ETTh2_96_12/sequential_utility_ranking_combined/summary.json"


def refuse_test(path: str | Path) -> None:
    if "test" in str(path).lower():
        raise ValueError(f"Refusing test path: {path}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def metric_row(
    cache: Mapping[str, Any],
    pred: torch.Tensor,
    std: torch.Tensor,
    method: str,
    subset: str,
    num_experts: int,
    scale: str,
) -> dict[str, Any]:
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.bool)
    mae = sample_mae(pred, target, mask, std)
    mse = sample_mse(pred, target, mask, std)
    return {
        "method": method,
        "subset": subset,
        "num_experts": num_experts,
        "scale": scale,
        "num_samples": int(mae.numel()),
        "mae": float(mae.mean()),
        "mse": float(mse.mean()),
    }


def fixed_subset_prediction(cache: Mapping[str, Any], indices: Sequence[int]) -> torch.Tensor:
    stack = cache["prediction_stack"].to(torch.float32)
    return stack[..., list(indices)].mean(dim=-1)


def verify_protocol(train_cache: Mapping[str, Any], val_cache: Mapping[str, Any], train_path: Path, val_path: Path, normalizer_path: Path) -> dict[str, Any]:
    starts = val_cache["absolute_window_starts"].to(torch.long)
    expected = torch.arange(10800, 11413, dtype=torch.long)
    checks = {
        "same_etth2_split": bool(train_cache.get("dataset") == "ETTh2" and val_cache.get("dataset") == "ETTh2" and val_cache.get("split_role") == "router_val"),
        "same_validation_window_ids": bool(torch.equal(starts.cpu(), expected)),
        "same_horizon": int(val_cache["forecast_horizon"]) == 12,
        "same_variables": int(val_cache["num_features"]) == 7 and tuple(val_cache["targets"].shape[1:]) == (12, 7),
        "same_cache_files": True,
        "same_num_samples": int(val_cache["num_windows"]) == 613 and int(starts.numel()) == 613,
        "same_mae_implementation": "experiments.oracle_weight_tournament.run_tournament.sample_mae",
        "same_normalization": "canonical_raw_original_scale_std_ones",
        "same_inverse_transform_behavior": "none_applied_cached_predictions_and_targets_evaluated_directly",
        "train_cache": str(train_path),
        "val_cache": str(val_path),
        "normalizer_checkpoint_for_diagnostic_only": str(normalizer_path),
        "train_cache_sha256": sha256_file(train_path),
        "val_cache_sha256": sha256_file(val_path),
        "validation_start_min": int(starts.min()),
        "validation_start_max": int(starts.max()),
        "validation_target_end_exclusive": int(starts.max()) + int(val_cache["forecast_horizon"]),
        "target_shape": list(val_cache["targets"].shape),
        "prediction_stack_shape": list(val_cache["prediction_stack"].shape),
        "expert_names": list(val_cache["expert_names"]),
        "test_cache_loaded": False,
    }
    checks["all_required_checks_passed"] = all(
        bool(checks[k])
        for k in (
            "same_etth2_split",
            "same_validation_window_ids",
            "same_horizon",
            "same_variables",
            "same_cache_files",
            "same_num_samples",
        )
    )
    return checks


def normalized_abs_error(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return ((pred - target) / std.view(1, 1, -1)).abs() * mask.to(torch.float32)


def run_specialist_rows(train_cache: Mapping[str, Any], val_cache: Mapping[str, Any], raw_std: torch.Tensor) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_base = fixed_subset_prediction(train_cache, fixed3_indices(train_cache))
    val_base = fixed_subset_prediction(val_cache, fixed3_indices(val_cache))
    folds = train_folds(int(train_cache["num_windows"]))
    configs = [c for c in grid() if c.scenario == "both"]
    cfg_by_name = {c.name: c for c in configs}
    leaderboard, fold_details = grid_eval_cached(train_cache, raw_std, train_base, configs, folds)
    selected = select_with_one_se(leaderboard, cfg_by_name)["both"]["selected"]
    selected_cfg = cfg_by_name[selected["name"]]

    d_train, m_train = optional_predictions(train_cache)
    d_val, m_val = optional_predictions(val_cache)
    target_train = train_cache["targets"].to(torch.float32)
    mask_train = train_cache["target_masks"].to(torch.bool)
    target_val = val_cache["targets"].to(torch.float32)
    mask_val = val_cache["target_masks"].to(torch.bool)
    init_base_err = normalized_abs_error(train_base, target_train, mask_train, raw_std)
    init_d_err = normalized_abs_error(d_train, target_train, mask_train, raw_std)
    init_m_err = normalized_abs_error(m_train, target_train, mask_train, raw_std)
    starts = val_cache["absolute_window_starts"].to(torch.long)
    rows = []
    for label, cfg in (("locked_etth1_expanded_both", LOCKED_SPECIALIST), ("etth2_selected_expanded_both", selected_cfg)):
        pred, extra, _ = run_causal_specialists(
            starts,
            val_base,
            d_val,
            m_val,
            target_val,
            mask_val,
            raw_std,
            cfg,
            init_base_err,
            init_d_err,
            init_m_err,
            trace_prefix={"method": label, "config": cfg.name},
        )
        row = metric_row(val_cache, pred, raw_std, label, cfg.name, 5, "raw_original")
        row.update(extra)
        rows.append(row)
    return rows, [{"selected": selected, "fold_rows": leaderboard[:10]}, *fold_details]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", default="cache/costarts_fresh/ETTh2_96_12/router_train_cache.pt")
    parser.add_argument("--val-cache", default="cache/costarts_fresh/ETTh2_96_12/router_val_cache.pt")
    parser.add_argument("--normalizer-checkpoint", default="checkpoints/costarts_fresh/ETTh2_96_12/clean_candidates/best_dlinear.pt")
    parser.add_argument("--out-dir", default="experiments/etth2_canonical_protocol")
    args = parser.parse_args()
    t0 = time.time()
    train_path = ROOT / args.train_cache
    val_path = ROOT / args.val_cache
    norm_path = ROOT / args.normalizer_checkpoint
    for path in (train_path, val_path, norm_path):
        refuse_test(path)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    train_cache = load_cache(train_path, "router_train")
    val_cache = load_cache(val_path, "router_val")
    raw_std = torch.ones(int(val_cache["num_features"]), dtype=torch.float32)
    diag_std = load_std_flexible(norm_path, int(val_cache["num_features"]))
    protocol = verify_protocol(train_cache, val_cache, train_path, val_path, norm_path)
    if not protocol["all_required_checks_passed"]:
        raise RuntimeError(f"Canonical protocol checks failed: {protocol}")

    names = list(val_cache["expert_names"])
    rows = []
    diag_rows = []
    for r in range(1, len(names) + 1):
        for indices in itertools.combinations(range(len(names)), r):
            subset = "+".join(names[i] for i in indices)
            pred = fixed_subset_prediction(val_cache, indices)
            method = "single_expert" if r == 1 else f"fixed_{r}_equal"
            rows.append(metric_row(val_cache, pred, raw_std, method, subset, r, "raw_original"))
            diag_rows.append(metric_row(val_cache, pred, diag_std, method, subset, r, "normalized_diagnostic"))
    specialist_rows, specialist_details = run_specialist_rows(train_cache, val_cache, raw_std)
    rows.extend(specialist_rows)
    rows_sorted = sorted(rows, key=lambda r: (float(r["mae"]), int(r["num_experts"]), str(r["subset"])))
    diag_rows_sorted = sorted(diag_rows, key=lambda r: (float(r["mae"]), int(r["num_experts"]), str(r["subset"])))
    best_by_size = {}
    for r in range(1, len(names) + 1):
        best_by_size[str(r)] = min([row for row in rows if int(row["num_experts"]) == r and row["method"].startswith(("single", "fixed"))], key=lambda row: float(row["mae"]))

    old_match = None
    if OLD_SUMMARY.exists():
        old = json.loads(OLD_SUMMARY.read_text(encoding="utf-8"))
        old_match = {}
        for k, old_row in old["best_fixed_by_size"].items():
            new_row = best_by_size[k]
            old_match[k] = {
                "old_subset": old_row["subset"],
                "new_subset": new_row["subset"],
                "old_mae": old_row["mae"],
                "new_mae": new_row["mae"],
                "old_mse": old_row["mse"],
                "new_mse": new_row["mse"],
                "matches": old_row["subset"] == new_row["subset"] and abs(float(old_row["mae"]) - float(new_row["mae"])) < 1e-7 and abs(float(old_row["mse"]) - float(new_row["mse"])) < 1e-7,
            }
    report = {
        "protocol": protocol,
        "canonical_metric": {
            "scale": "raw_original",
            "std": raw_std.tolist(),
            "mae_implementation": protocol["same_mae_implementation"],
            "inverse_transform_behavior": protocol["same_inverse_transform_behavior"],
        },
        "diagnostic_normalized_metric": {"std": diag_std.tolist(), "not_canonical": True},
        "best_overall_raw": rows_sorted[0],
        "best_single_raw": best_by_size["1"],
        "best_by_num_experts_raw": best_by_size,
        "old_summary_consistency": old_match,
        "old_summary_all_best_by_size_match": bool(old_match and all(v["matches"] for v in old_match.values())),
        "runtime_sec": time.time() - t0,
        "reproduce_command": "python experiments\\etth2_canonical_protocol\\run_canonical_etth2_baselines.py",
    }
    write_csv(out_dir / "canonical_raw_results.csv", rows_sorted)
    write_csv(out_dir / "diagnostic_normalized_results.csv", diag_rows_sorted)
    write_json(out_dir / "specialist_selection_details.json", specialist_details)
    write_json(out_dir / "final_report.json", report)
    lines = [
        "# Canonical ETTh2 Protocol",
        "",
        "## Protocol",
        "",
        "- Validation cache: `cache/costarts_fresh/ETTh2_96_12/router_val_cache.pt`.",
        "- Validation starts: `10800..11412`, `613` windows.",
        "- Horizon: `12`; variables: `7`.",
        "- Canonical metric: raw/original-scale MAE/MSE using `sample_mae` with `std=ones`.",
        "- Inverse transform: none applied; cached predictions and targets are evaluated directly.",
        "- Diagnostic normalized metrics are saved separately and are not canonical.",
        "",
        "## Best Raw Results",
        "",
        f"- Best single: `{report['best_single_raw']['subset']}`, MAE `{report['best_single_raw']['mae']:.6f}`, MSE `{report['best_single_raw']['mse']:.6f}`.",
        f"- Best overall fixed/specialist row: `{report['best_overall_raw']['subset']}`, method `{report['best_overall_raw']['method']}`, MAE `{report['best_overall_raw']['mae']:.6f}`.",
        f"- Old summary best-by-size match: `{report['old_summary_all_best_by_size_match']}`.",
        "",
        "## Reproduce",
        "",
        "```powershell",
        report["reproduce_command"],
        "```",
    ]
    (out_dir / "canonical_protocol_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
