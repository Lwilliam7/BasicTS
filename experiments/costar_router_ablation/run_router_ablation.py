"""Clean router ablation study for COSTAR (isolates the router only).

Reuses the existing causal router machinery unmodified:

- `chronological_online_weights()` (global chronological EMA branch)
- `chronological_hv_weights()` / `errors_to_weights()` / `aggregate_error()`
  (horizon-variable EMA branch; already supports "global"/"horizon"/
  "variable"/"hv"/"hv_lowrank" granularities via `trial.mode`)
- `parameterized_current_base_prediction()` (ETTh1) / `current_base_prediction()`
  (ETTh2) for the current production "Global + HxV" blend.

Nothing about the expert pool, data splits, core-selection protocol, causal
rules, or frozen configuration is changed. No new adaptive mechanism is
implemented: every ablation is produced by calling the above functions with
the canonical decay/temperature/rank already used throughout the repo,
varying only the aggregation granularity. Specialists (DLinear/ModernTCN
residual correction) are intentionally excluded from every row -- this
experiment isolates the router itself.

HARD RULE: never touches the test set. `refuse_test()` guards every
cache/config/checkpoint path. Only `router_train` (existing frozen core
selection, reused as-is) and a single frozen `router_val` evaluation are
used, exactly like `frozen_costar` and `dual_timescale_memory_costar`.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.etth2_train_selected_core.run_etth2_train_selected_core_eval as etth2  # noqa: E402
import experiments.train_selected_core_etth1.run_train_selected_core_eval as etth1  # noqa: E402
from experiments.chronological_adaptive_costar.run_chronological_adaptive_costar import (  # noqa: E402
    paired_bootstrap,
)
from experiments.frozen_costar.run_frozen_costar_validation import (  # noqa: E402
    ETTH1_FROZEN,
    ETTH2_FROZEN,
    load_frozen_core,
    sha256_file,
)
from experiments.horizon_variable_adaptive_costar.run_hv_adaptive_costar import (  # noqa: E402
    Trial as HvTrial,
    chronological_hv_weights,
    predict_from_hv_weights,
)
from experiments.oracle_weight_tournament.run_tournament import (  # noqa: E402
    load_cache,
    load_std,
    sample_mae,
    sample_mse,
)


OUT_DIR = ROOT / "experiments/costar_router_ablation"

# Canonical settings already used throughout the repo (frozen_costar,
# train_selected_core_etth1, etth2_train_selected_core). Not tuned here.
CANONICAL_HV_DECAY = 0.95
CANONICAL_HV_TEMPERATURE = 0.1
CANONICAL_HV_RANK = 1
CANONICAL_CHRONO_DECAY = 0.97
CANONICAL_CHRONO_TEMPERATURE = 0.1
CANONICAL_CHRONO_STATIC_BLEND = 0.5  # 0.5 * equal_static + 0.5 * online, as in the production chrono branch
CANONICAL_BASE_BLEND = {"chrono": 0.25, "hv": 0.75}  # production Global + HxV blend

GRANULARITY_LABELS = {
    "global": "Global causal EMA",
    "horizon": "Horizon-only causal EMA",
    "variable": "Variable-only causal EMA",
    "hv": "Full horizon x variable causal EMA",
    "hv_lowrank": "Low-rank horizon x variable causal EMA",
}


def refuse_test(path: str | Path) -> None:
    if "test" in str(path).lower():
        raise ValueError(f"Test access forbidden during router ablation: {path}")


def git_commit_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


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
# Dataset-agnostic helpers (thin wrappers around existing per-dataset code)
# ---------------------------------------------------------------------------


def dataset_per_location_error(dataset: str, cache: Mapping[str, Any], expert_idx: Sequence[int], std: torch.Tensor) -> torch.Tensor:
    if dataset == "ETTh1":
        return etth1.per_location_abs_error_for_indices(cache, std, expert_idx)
    if dataset == "ETTh2":
        return etth2.per_location_error(cache, expert_idx, std)
    raise ValueError(dataset)


def dataset_forecasts(cache: Mapping[str, Any], expert_idx: Sequence[int]) -> torch.Tensor:
    return cache["prediction_stack"][..., list(expert_idx)].to(torch.float32)


def metric_values(cache: Mapping[str, Any], pred: torch.Tensor, std: torch.Tensor) -> dict[str, Any]:
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.bool)
    mae = sample_mae(pred, target, mask, std)
    mse = sample_mse(pred, target, mask, std)
    return {"mae": float(mae.mean()), "mse": float(mse.mean()), "per_window_mae": mae, "per_window_mse": mse}


# ---------------------------------------------------------------------------
# Ablation predictions -- each one calls existing functions unmodified
# ---------------------------------------------------------------------------


def equal_fixed_prediction(cache: Mapping[str, Any], expert_idx: Sequence[int]) -> tuple[torch.Tensor, dict[str, Any]]:
    pred = dataset_forecasts(cache, expert_idx).mean(dim=-1)
    return pred, {"num_causal_updates": 0, "decay": None, "temperature": None, "blend_coefficients": None}


def granularity_trial(mode: str) -> HvTrial:
    return HvTrial("hv_ema", f"router_ablation_{mode}", mode=mode, rank=CANONICAL_HV_RANK, decay=CANONICAL_HV_DECAY, temperature=CANONICAL_HV_TEMPERATURE)


def granularity_ema_prediction(
    dataset: str, cache: Mapping[str, Any], train_cache: Mapping[str, Any], expert_idx: Sequence[int], std: torch.Tensor, mode: str
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Ablations 2-5 and 7: a single unified formula (`errors_to_weights` via
    `chronological_hv_weights`), varying only the aggregation granularity.
    `mode="global"` -> one EMA per expert (ablation 2). `mode="horizon"` ->
    one EMA per horizon x expert, variables averaged before update (ablation 3,
    exact by EMA/mean-averaging linearity -- see report). `mode="variable"` ->
    one EMA per variable x expert (ablation 4). `mode="hv"` -> full H x V x E
    EMA (ablation 5). `mode="hv_lowrank"` -> rank-1 low-rank H x V approximation
    (ablation 7 comparison partner for ablation 5).
    """
    starts = cache["absolute_window_starts"].to(torch.long)
    horizon = int(cache["forecast_horizon"])
    forecasts = dataset_forecasts(cache, expert_idx)
    train_err = dataset_per_location_error(dataset, train_cache, expert_idx, std)
    val_err = dataset_per_location_error(dataset, cache, expert_idx, std)
    trial = granularity_trial(mode)
    weights, extra = chronological_hv_weights(starts, train_err.mean(dim=0), val_err, horizon, trial)
    pred = predict_from_hv_weights(forecasts, weights)
    return pred, {
        "num_causal_updates": extra["num_updates"],
        "decay": CANONICAL_HV_DECAY,
        "temperature": CANONICAL_HV_TEMPERATURE,
        "rank": CANONICAL_HV_RANK if mode == "hv_lowrank" else None,
        "blend_coefficients": None,
        **{k: v for k, v in extra.items() if k != "num_updates"},
    }


def global_plus_hv_prediction(
    dataset: str, cache: Mapping[str, Any], train_cache: Mapping[str, Any], expert_idx: Sequence[int], std: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Ablation 6: the current production base COSTAR, called verbatim."""
    if dataset == "ETTh1":
        pred, extra = etth1.parameterized_current_base_prediction(cache, train_cache, std, expert_idx, 7, device)
    elif dataset == "ETTh2":
        pred, extra = etth2.current_base_prediction(cache, train_cache, expert_idx, std)
    else:
        raise ValueError(dataset)
    return pred, {
        "num_causal_updates": (extra.get("chrono_num_updates") or 0) + (extra.get("hv_num_updates") or 0),
        "chrono_num_updates": extra.get("chrono_num_updates"),
        "hv_num_updates": extra.get("hv_num_updates"),
        "chrono_decay": CANONICAL_CHRONO_DECAY,
        "chrono_temperature": CANONICAL_CHRONO_TEMPERATURE,
        "chrono_static_blend": CANONICAL_CHRONO_STATIC_BLEND,
        "hv_decay": CANONICAL_HV_DECAY,
        "hv_temperature": CANONICAL_HV_TEMPERATURE,
        "hv_rank": CANONICAL_HV_RANK,
        "blend_coefficients": CANONICAL_BASE_BLEND,
    }


METHODS = [
    ("equal_fixed", "Equal fixed ensemble"),
    ("global_causal", "Global causal EMA"),
    ("horizon_only", "Horizon-only causal EMA"),
    ("variable_only", "Variable-only causal EMA"),
    ("hxv_causal", "Full horizon x variable causal EMA"),
    ("global_plus_hxv", "Global + HxV COSTAR"),
    ("hxv_lowrank", "Low-rank horizon x variable causal EMA"),
]
MODE_BY_METHOD = {"global_causal": "global", "horizon_only": "horizon", "variable_only": "variable", "hxv_causal": "hv", "hxv_lowrank": "hv_lowrank"}


def predict_method(
    method: str, dataset: str, cache: Mapping[str, Any], train_cache: Mapping[str, Any], expert_idx: Sequence[int], std: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, dict[str, Any]]:
    if method == "equal_fixed":
        return equal_fixed_prediction(cache, expert_idx)
    if method == "global_plus_hxv":
        return global_plus_hv_prediction(dataset, cache, train_cache, expert_idx, std, device)
    return granularity_ema_prediction(dataset, cache, train_cache, expert_idx, std, MODE_BY_METHOD[method])


# ---------------------------------------------------------------------------
# Causality perturbation check: mutate only the tail of router_val and verify
# earlier-window predictions are bit-identical.
# ---------------------------------------------------------------------------


def perturb_tail_targets(cache: Mapping[str, Any], suffix_start: int, seed: int = 20260821) -> dict[str, Any]:
    cloned = dict(cache)
    gen = torch.Generator().manual_seed(seed)
    targets = cache["targets"].clone()
    noise = torch.randn(targets[suffix_start:].shape, generator=gen, dtype=torch.float32)
    targets[suffix_start:] = noise
    cloned["targets"] = targets
    return cloned


def causality_perturbation_check(
    method: str, dataset: str, val_cache: Mapping[str, Any], train_cache: Mapping[str, Any], expert_idx: Sequence[int], std: torch.Tensor, device: torch.device
) -> dict[str, Any]:
    n = int(val_cache["num_windows"])
    suffix_start = int(round(n * 0.75))
    horizon = int(val_cache["forecast_horizon"])
    base_pred, _ = predict_method(method, dataset, val_cache, train_cache, expert_idx, std, device)
    mutated = perturb_tail_targets(val_cache, suffix_start)
    mut_pred, _ = predict_method(method, dataset, mutated, train_cache, expert_idx, std, device)
    prefix_equal = bool(torch.equal(base_pred[:suffix_start], mut_pred[:suffix_start]))
    tail_differs = not bool(torch.equal(base_pred[suffix_start:], mut_pred[suffix_start:]))
    return {
        "method": method,
        "dataset": dataset,
        "suffix_start_window": suffix_start,
        "num_windows": n,
        "forecast_horizon": horizon,
        "earlier_windows_unchanged": prefix_equal,
        "tail_predictions_reacted": tail_differs,
        "result": "PASS" if prefix_equal else "FAIL",
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def per_window_rows(dataset: str, method: str, cache: Mapping[str, Any], std: torch.Tensor, pred: torch.Tensor) -> list[dict[str, Any]]:
    m = metric_values(cache, pred, std)
    starts = cache["absolute_window_starts"].to(torch.long)
    return [
        {
            "dataset": dataset,
            "method": method,
            "window_index": i,
            "absolute_window_start": int(starts[i]),
            "mae": float(m["per_window_mae"][i]),
            "mse": float(m["per_window_mse"][i]),
        }
        for i in range(cache["num_windows"])
    ]


def evaluate_dataset(dataset: str, train_cache: Mapping[str, Any], val_cache: Mapping[str, Any], std: torch.Tensor, core: Sequence[str], device: torch.device) -> dict[str, Any]:
    expert_idx = etth1.expert_indices(val_cache, core) if dataset == "ETTh1" else etth2.expert_indices(val_cache, core)

    results: dict[str, dict[str, Any]] = {}
    result_rows: list[dict[str, Any]] = []
    per_window: list[dict[str, Any]] = []
    causality_rows: list[dict[str, Any]] = []

    for method, label in METHODS:
        pred, extra = predict_method(method, dataset, val_cache, train_cache, expert_idx, std, device)
        m = metric_values(val_cache, pred, std)
        results[method] = {"pred": pred, "mae": m["mae"], "mse": m["mse"], "per_window_mae": m["per_window_mae"], "extra": extra}
        result_rows.append({"dataset": dataset, "method": method, "label": label, "mae": m["mae"], "mse": m["mse"], **extra})
        per_window.extend(per_window_rows(dataset, method, val_cache, std, pred))
        check = causality_perturbation_check(method, dataset, val_cache, train_cache, expert_idx, std, device)
        causality_rows.append(check)
        if check["result"] != "PASS":
            raise AssertionError(f"{dataset} {method}: earlier router_val predictions changed after mutating only tail targets")

    def delta_rows(baseline: str) -> list[dict[str, Any]]:
        base_mae = results[baseline]["per_window_mae"]
        out = []
        for method, label in METHODS:
            if method == baseline:
                continue
            boot = paired_bootstrap(results[method]["per_window_mae"], base_mae, seed=20260821, samples=5000)
            out.append(
                {
                    "dataset": dataset,
                    "baseline": baseline,
                    "method": method,
                    "label": label,
                    "delta_mae": results[method]["mae"] - results[baseline]["mae"],
                    "delta_mse": results[method]["mse"] - results[baseline]["mse"],
                    **{f"boot_{k}": v for k, v in boot.items()},
                }
            )
        return out

    deltas = delta_rows("equal_fixed") + delta_rows("global_causal") + delta_rows("hxv_causal")

    return {
        "dataset": dataset,
        "expert_indices": list(expert_idx),
        "core": list(core),
        "result_rows": result_rows,
        "per_window": per_window,
        "deltas": deltas,
        "causality": causality_rows,
    }


def make_report(out_dir: Path, report: Mapping[str, Any]) -> None:
    lines = [
        "# COSTAR Router Ablation Study",
        "",
        "Isolates the router: every row below is a base-mixture prediction only "
        "(no DLinear/ModernTCN specialist correction, no Ridge/MLP residual "
        "correction). All ablations reuse the existing `chronological_hv_weights` "
        "/ `errors_to_weights` machinery unmodified, varying only the aggregation "
        "granularity (`trial.mode`); ablation 6 reuses the production "
        "`parameterized_current_base_prediction` / `current_base_prediction` "
        "verbatim. Canonical settings only -- nothing tuned per ablation:",
        "",
        f"- HxV family: decay={CANONICAL_HV_DECAY}, temperature={CANONICAL_HV_TEMPERATURE}, low-rank rank={CANONICAL_HV_RANK}",
        f"- Chrono branch (ablation 6 only): decay={CANONICAL_CHRONO_DECAY}, temperature={CANONICAL_CHRONO_TEMPERATURE}, static blend={CANONICAL_CHRONO_STATIC_BLEND}",
        f"- Global+HxV blend (ablation 6 only): {CANONICAL_BASE_BLEND}",
        "",
        f"Git commit: `{report['git_commit_sha']}`",
        "",
        "## Result table (router_val only)",
        "",
        "| Method | ETTh1 MAE | ETTh1 MSE | ETTh2 MAE | ETTh2 MSE |",
        "|---|---:|---:|---:|---:|",
    ]
    by_ds = {ds: {r["method"]: r for r in report["datasets"][ds]["result_rows"]} for ds in ("ETTh1", "ETTh2")}
    main_methods = ["equal_fixed", "global_causal", "horizon_only", "variable_only", "hxv_causal", "global_plus_hxv"]
    for method in main_methods:
        e1, e2 = by_ds["ETTh1"][method], by_ds["ETTh2"][method]
        lines.append(f"| {e1['label']} | `{e1['mae']:.6f}` | `{e1['mse']:.6f}` | `{e2['mae']:.6f}` | `{e2['mse']:.6f}` |")
    lines.append("")
    lines.append("Ranking by MAE (lower is better):")
    for ds in ("ETTh1", "ETTh2"):
        ranked = sorted((by_ds[ds][m] for m in main_methods), key=lambda r: r["mae"])
        lines.append(f"- **{ds}**: " + " < ".join(f"{r['label']} (`{r['mae']:.6f}`)" for r in ranked))
    lines.append("")
    lines.append("## Ablation 7: full HxV vs low-rank HxV")
    lines.append("")
    lines.append("| Dataset | Full HxV MAE | Low-rank HxV MAE | Delta (low-rank minus full) |")
    lines.append("|---|---:|---:|---:|")
    for ds in ("ETTh1", "ETTh2"):
        full, low = by_ds[ds]["hxv_causal"], by_ds[ds]["hxv_lowrank"]
        lines.append(f"| {ds} | `{full['mae']:.6f}` | `{low['mae']:.6f}` | `{low['mae']-full['mae']:+.6f}` |")
    lines.append("")
    lines.append("## Deltas vs baselines, with paired-bootstrap 95% CI on the difference")
    lines.append("")
    lines.append("| Dataset | Baseline | Method | Delta MAE | 95% CI | CI excludes zero |")
    lines.append("|---|---|---|---:|---|---|")
    for ds in ("ETTh1", "ETTh2"):
        for row in report["datasets"][ds]["deltas"]:
            lines.append(
                f"| {ds} | {row['baseline']} | {row['label']} | `{row['delta_mae']:+.6f}` | "
                f"[{row['boot_ci95_low']:+.6f}, {row['boot_ci95_high']:+.6f}] | {row['boot_ci_excludes_zero']} |"
            )
    lines.append("")
    lines.append("## Causality perturbation check")
    lines.append("")
    lines.append("Only the last 25% of router_val window targets were randomized; earlier-window "
                  "predictions must be bit-identical to the unperturbed run.")
    lines.append("")
    lines.append("| Dataset | Method | Earlier windows unchanged | Tail reacted | Result |")
    lines.append("|---|---|---|---|---|")
    for ds in ("ETTh1", "ETTh2"):
        for c in report["datasets"][ds]["causality"]:
            lines.append(f"| {ds} | {c['method']} | {c['earlier_windows_unchanged']} | {c['tail_predictions_reacted']} | {c['result']} |")
    lines.append("")
    lines += [
        "## Hard rule compliance",
        "",
        "```text",
        "TEST SET ACCESSED: NO",
        "TEST CACHE LOADED: NO",
        "TEST METRICS COMPUTED: NO",
        "```",
        "",
        "## Reproduce",
        "",
        "```powershell",
        report["command"],
        "```",
    ]
    (out_dir / "router_ablation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    global OUT_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    OUT_DIR = Path(args.out_dir)
    device = torch.device(args.device)
    start = time.time()

    paths = {
        "ETTh1": {
            "train_cache": ROOT / "cache/costarts_walkforward/router_train_20_60_cache.pt",
            "val_cache": ROOT / "cache/costarts_walkforward/router_val_60_80_cache.pt",
            "normalizer": ROOT / "checkpoints/costarts_walkforward/final_60/DLinear/best_expert.pt",
            "frozen_config": ETTH1_FROZEN,
        },
        "ETTh2": {
            "train_cache": ROOT / "cache/costarts_fresh/ETTh2_96_12/router_train_cache.pt",
            "val_cache": ROOT / "cache/costarts_fresh/ETTh2_96_12/router_val_cache.pt",
            "normalizer": None,
            "frozen_config": ETTH2_FROZEN,
        },
    }
    for dataset_paths in paths.values():
        for key in ["train_cache", "val_cache", "frozen_config"]:
            refuse_test(dataset_paths[key])
        if dataset_paths["normalizer"] is not None:
            refuse_test(dataset_paths["normalizer"])
    refuse_test(args.out_dir)

    report: dict[str, Any] = {
        "experiment": "costar_router_ablation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": f"python experiments\\costar_router_ablation\\run_router_ablation.py --device {args.device}",
        "git_commit_sha": git_commit_sha(),
        "datasets": {},
    }
    all_result_rows: list[dict[str, Any]] = []
    all_per_window: list[dict[str, Any]] = []
    all_deltas: list[dict[str, Any]] = []
    all_causality: list[dict[str, Any]] = []

    for dataset in ["ETTh1", "ETTh2"]:
        p = paths[dataset]
        train_cache = load_cache(p["train_cache"], "router_train_20_60" if dataset == "ETTh1" else "router_train")
        val_cache = load_cache(p["val_cache"], "router_val_60_80" if dataset == "ETTh1" else "router_val")
        if dataset == "ETTh1":
            std = load_std(p["normalizer"], int(val_cache["num_features"]))
        else:
            std = torch.ones(int(val_cache["num_features"]), dtype=torch.float32)
        core = load_frozen_core(p["frozen_config"])
        print(f"[router-ablation] {dataset}: evaluating {len(METHODS)} methods on router_val...", flush=True)
        result = evaluate_dataset(dataset, train_cache, val_cache, std, core, device)
        result["cache_hashes"] = {
            "train_cache_sha256": sha256_file(p["train_cache"]),
            "val_cache_sha256": sha256_file(p["val_cache"]),
        }
        if p["normalizer"] is not None:
            result["cache_hashes"]["normalizer_sha256"] = sha256_file(p["normalizer"])
        report["datasets"][dataset] = {k: v for k, v in result.items() if k not in ("per_window",)}
        all_result_rows.extend(result["result_rows"])
        all_per_window.extend(result["per_window"])
        all_deltas.extend(result["deltas"])
        all_causality.extend(result["causality"])
        print(f"[router-ablation] {dataset}: done.", flush=True)

    report["runtime_sec"] = time.time() - start
    report["test_set_accessed"] = False
    report["test_cache_loaded"] = False
    report["test_metrics_computed"] = False

    write_json(OUT_DIR / "router_ablation_results.json", report)
    write_csv(OUT_DIR / "router_ablation_results.csv", all_result_rows)
    write_csv(OUT_DIR / "router_ablation_per_window.csv", all_per_window)
    write_csv(OUT_DIR / "router_ablation_deltas.csv", all_deltas)
    write_csv(OUT_DIR / "router_ablation_causality.csv", all_causality)
    make_report(OUT_DIR, report)

    print("TEST SET ACCESSED: NO")
    print("TEST CACHE LOADED: NO")
    print("TEST METRICS COMPUTED: NO")
    print(json.dumps({k: v for k, v in report.items() if k != "datasets"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
