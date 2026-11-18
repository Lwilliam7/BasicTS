"""Expert-Choice HxV Pilot.

Validation-only Electricity experiment. No test split is loaded.

Research question:
Does reversing hard HxV routing from "each horizon-variable cell chooses an
expert" to "each expert claims a capacity-limited set of horizon-variable
cells" improve forecasting, while using the exact same train-derived
competence score tensor?
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.costar_multidataset_frozen.common import (  # noqa: E402
    block_bootstrap_with_prob,
    every_kth_phase_bootstrap,
    forecasts_for,
    granularity_ema_prediction,
    metric_values,
    per_location_error,
)
from experiments.oracle_weight_tournament.run_tournament import load_cache, load_std, sample_mae, sample_mse  # noqa: E402


DATASET = "Electricity"
CORE_EXPERTS = ("PatchTST", "iTransformer", "TimesNet")
EXPECTED_CACHE_EXPERT_ORDER = ("DLinear", "PatchTST", "iTransformer", "TimesNet", "ModernTCN")
TRAIN_CACHE_PATH = ROOT / "cache/costarts_walkforward_Electricity/router_train_20_60_cache.pt"
VAL_CACHE_PATH = ROOT / "cache/costarts_walkforward_Electricity/router_val_60_80_cache.pt"
NORMALIZER_PATH = ROOT / "checkpoints/costarts_walkforward_Electricity/final_60/DLinear/best_expert.pt"
CAPACITY_FACTORS = (1.0, 1.25, 1.5)
BLOCK_LENGTH = 24
BOOTSTRAP_SAMPLES = 10000
PHASE_K = 12
SEED = 20260830
REFERENCE_ELECTRICITY_HXV_MAE = 0.211775


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def refuse_test(path: str | Path) -> None:
    if "test" in str(path).lower():
        raise ValueError(f"Test access forbidden: {path}")


def sha256_file(path: Path) -> str:
    refuse_test(path)
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def expert_indices(cache: Mapping[str, Any]) -> list[int]:
    names = tuple(cache["expert_names"])
    if names != EXPECTED_CACHE_EXPERT_ORDER:
        raise AssertionError(f"Unexpected cache expert order: {names}")
    return [names.index(name) for name in CORE_EXPERTS]


def checkpoint_hashes() -> dict[str, str]:
    root = ROOT / "checkpoints/costarts_walkforward_Electricity/final_60"
    return {expert: sha256_file(root / expert / "best_expert.pt") for expert in CORE_EXPERTS}


def build_score_tensor(train_cache: Mapping[str, Any], expert_idx: Sequence[int], std: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns score [K,H,V] and mean normalized abs error [H,V,K].

    This intentionally reuses the same per-location error convention as the
    existing HxV COSTAR path: normalized absolute error is masked, then averaged
    over router_train windows.
    """
    train_err = per_location_error(train_cache, expert_idx, std).to(torch.float32)
    err_hvk = train_err.mean(dim=0)
    score_khv = -err_hvk.permute(2, 0, 1).contiguous()
    return score_khv, err_hvk


def capacity_limits(total_cells: int, num_experts: int, factor: float) -> tuple[list[int], bool]:
    if factor == 1.0:
        base = total_cells // num_experts
        caps = [base] * num_experts
        for i in range(total_cells - sum(caps)):
            caps[i] += 1
        return caps, True
    cap = int(math.ceil((total_cells / num_experts) * factor))
    return [cap] * num_experts, False


def solve_capacity_assignment(score_khv: torch.Tensor, factor: float) -> dict[str, Any]:
    k, h, v = score_khv.shape
    total = h * v
    caps, exact = capacity_limits(total, k, factor)
    slots = []
    for expert, cap in enumerate(caps):
        slots.extend([expert] * cap)
    if len(slots) < total:
        raise AssertionError(f"Capacity infeasible: total slots {len(slots)} < cells {total}")
    score_cells = score_khv.reshape(k, total).t().numpy()
    slot_experts = np.asarray(slots, dtype=np.int64)
    cost = -score_cells[:, slot_experts]
    row_ind, col_ind = linear_sum_assignment(cost)
    if row_ind.shape[0] != total:
        raise AssertionError(f"Assignment did not cover every cell: {row_ind.shape[0]} of {total}")
    assigned_flat = np.empty(total, dtype=np.int64)
    assigned_flat[row_ind] = slot_experts[col_ind]
    counts = np.bincount(assigned_flat, minlength=k).astype(int).tolist()
    if exact and counts != caps:
        raise AssertionError(f"Exact capacity counts mismatch: counts={counts}, caps={caps}")
    if any(counts[i] > caps[i] for i in range(k)):
        raise AssertionError(f"Upper capacity exceeded: counts={counts}, caps={caps}")
    assignment = torch.from_numpy(assigned_flat.reshape(h, v)).to(torch.long)
    objective = float(score_khv.gather(0, assignment.unsqueeze(0)).sum())
    return {
        "assignment": assignment,
        "capacity_factor": factor,
        "capacity_limits": caps,
        "exact_capacity": exact,
        "counts": counts,
        "allocation_pct": [c / total for c in counts],
        "objective_score_sum": objective,
        "solver": "scipy.optimize.linear_sum_assignment with cloned expert-capacity slots",
    }


def hard_normal_assignment(score_khv: torch.Tensor) -> dict[str, Any]:
    assignment = score_khv.argmax(dim=0).to(torch.long)
    counts = torch.bincount(assignment.flatten(), minlength=score_khv.shape[0]).tolist()
    total = int(assignment.numel())
    return {
        "assignment": assignment,
        "counts": [int(c) for c in counts],
        "allocation_pct": [float(c) / total for c in counts],
        "objective_score_sum": float(score_khv.gather(0, assignment.unsqueeze(0)).sum()),
    }


def permuted_score_locations(score_khv: torch.Tensor, seed: int = SEED + 11) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    k, h, v = score_khv.shape
    out = torch.empty_like(score_khv)
    for expert in range(k):
        perm = torch.randperm(h * v, generator=gen)
        out[expert] = score_khv[expert].flatten()[perm].reshape(h, v)
    return out


def random_scores_like(score_khv: torch.Tensor, seed: int = SEED + 17) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(score_khv.shape, generator=gen, dtype=torch.float32)


def predict_static_assignment(forecasts_nhvk: torch.Tensor, assignment_hv: torch.Tensor) -> torch.Tensor:
    n, h, v, _ = forecasts_nhvk.shape
    idx = assignment_hv.view(1, h, v, 1).expand(n, h, v, 1)
    return forecasts_nhvk.gather(dim=3, index=idx).squeeze(-1)


def predict_dynamic_assignment(forecasts_nhvk: torch.Tensor, assignment_nhv: torch.Tensor) -> torch.Tensor:
    idx = assignment_nhv.unsqueeze(-1)
    return forecasts_nhvk.gather(dim=3, index=idx).squeeze(-1)


def summarize_static_assignment(name: str, assignment_hv: torch.Tensor | None, k: int, weights: Sequence[float] | None = None) -> dict[str, Any]:
    h, v = 12, 321
    total = h * v
    if assignment_hv is None:
        if weights is None:
            raise ValueError("weights required for soft/no-assignment summary")
        counts = [float(w) * total for w in weights]
        return {
            "method": name,
            "assigned_cells": None,
            "allocation_pct": [float(w) for w in weights],
            "soft_or_equal_weighted_cell_equivalent_counts": counts,
        }
    counts_t = torch.bincount(assignment_hv.flatten(), minlength=k).to(torch.long)
    counts = [int(x) for x in counts_t.tolist()]
    return {
        "method": name,
        "assigned_cells": dict(zip(CORE_EXPERTS, counts)),
        "allocation_pct": dict(zip(CORE_EXPERTS, [c / total for c in counts])),
    }


def metrics_for_prediction(name: str, pred: torch.Tensor, val_cache: Mapping[str, Any], std: torch.Tensor, allocation: Mapping[str, Any]) -> dict[str, Any]:
    target = val_cache["targets"].to(torch.float32)
    mask = val_cache["target_masks"].to(torch.bool)
    mae = sample_mae(pred, target, mask, std)
    mse = sample_mse(pred, target, mask, std)
    return {
        "method": name,
        "mae": float(mae.mean()),
        "mse": float(mse.mean()),
        "per_window_mae": mae,
        **allocation,
    }


def serial_metric(row: Mapping[str, Any], equal_mae: float, hard_mae: float) -> dict[str, Any]:
    out = {k: v for k, v in row.items() if k != "per_window_mae"}
    out["delta_mae_vs_equal"] = float(row["mae"] - equal_mae)
    out["delta_mae_vs_hard_hxv"] = float(row["mae"] - hard_mae)
    return out


def phase_analysis(candidate: torch.Tensor, baseline: torch.Tensor, k: int = PHASE_K) -> dict[str, Any]:
    diff = candidate - baseline
    phase_rows = []
    for phase in range(k):
        vals = diff[phase::k]
        if vals.numel() == 0:
            continue
        phase_rows.append(
            {
                "phase": phase,
                "count": int(vals.numel()),
                "mean_delta": float(vals.mean()),
                "expert_choice_better": bool(vals.mean() < 0),
            }
        )
    agree = sum(1 for r in phase_rows if r["mean_delta"] < 0)
    return {
        "k": k,
        "num_phases": len(phase_rows),
        "num_phases_negative": agree,
        "all_phases_agree_negative": agree == len(phase_rows),
        "majority_phases_agree_negative": agree > len(phase_rows) / 2,
        "phase_rows": phase_rows,
    }


def comparison_stats(candidate: torch.Tensor, baseline: torch.Tensor, label: str) -> dict[str, Any]:
    block = block_bootstrap_with_prob(candidate, baseline, block=BLOCK_LENGTH, seed=SEED, samples=BOOTSTRAP_SAMPLES)
    phase_boot = every_kth_phase_bootstrap(candidate - baseline, k=PHASE_K, seed=SEED, samples=BOOTSTRAP_SAMPLES)
    phases = phase_analysis(candidate, baseline, k=PHASE_K)
    return {
        "comparison": label,
        "block24": block,
        "every_12th_phase_bootstrap": phase_boot,
        "phase_analysis": phases,
        "decision_inputs": {
            "mean_delta": block["mean_delta"],
            "ci_excludes_zero": block["ci_excludes_zero"],
            "prob_delta_negative": block["prob_delta_negative"],
            "every_12th_agrees": phases["majority_phases_agree_negative"] and phase_boot["mean_delta"] < 0,
        },
    }


def oracle_predictions(val_cache: Mapping[str, Any], forecasts_val: torch.Tensor, std: torch.Tensor) -> dict[str, Any]:
    target = val_cache["targets"].to(torch.float32)
    mask = val_cache["target_masks"].to(torch.float32)
    err = ((forecasts_val - target.unsqueeze(-1)) / std.view(1, 1, -1, 1)).abs() * mask.unsqueeze(-1)
    dynamic_assignment = err.argmin(dim=3).to(torch.long)
    dynamic_pred = predict_dynamic_assignment(forecasts_val, dynamic_assignment)
    static_assignment = err.mean(dim=0).argmin(dim=2).to(torch.long)
    static_pred = predict_static_assignment(forecasts_val, static_assignment)
    return {
        "dynamic_oracle": (dynamic_pred, dynamic_assignment),
        "static_val_oracle": (static_pred, static_assignment),
    }


def build_results_md(results: Mapping[str, Any]) -> str:
    rows = results["deployable_results"]
    lines = [
        "# Expert-Choice HxV Pilot",
        "",
        "| Method | MAE | Delta vs Equal | Delta vs Hard HxV |",
        "|---|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(f"| {r['method']} | {r['mae']:.6f} | {r['delta_mae_vs_equal']:+.6f} | {r['delta_mae_vs_hard_hxv']:+.6f} |")
    lines += [
        "",
        f"Verdict: {results['verdict']}",
        "",
        results["interpretation"],
        "",
        "## Allocation",
        "",
        "| Method | PatchTST cells | iTransformer cells | TimesNet cells |",
        "|---|---:|---:|---:|",
    ]
    for r in rows:
        assigned = r.get("assigned_cells")
        if isinstance(assigned, dict):
            cells = [assigned.get(e, "") for e in CORE_EXPERTS]
        else:
            equiv = r.get("soft_or_equal_weighted_cell_equivalent_counts")
            cells = [f"{x:.1f}" for x in equiv] if equiv else ["n/a", "n/a", "n/a"]
        lines.append(f"| {r['method']} | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines += [
        "",
        "## Main Statistics",
        "",
        "| Comparison | Mean Delta | 95% CI | P(delta < 0) | Phase Agreement |",
        "|---|---:|---:|---:|---:|",
    ]
    for s in results["expert_choice_vs_hard_stats"]:
        b = s["block24"]
        p = s["phase_analysis"]
        lines.append(
            f"| {s['comparison']} | {b['mean_delta']:+.6f} | "
            f"[{b['ci95_low']:+.6f}, {b['ci95_high']:+.6f}] | "
            f"{b['prob_delta_negative']:.3f} | {p['num_phases_negative']}/{p['num_phases']} negative |"
        )
    lines += [
        "",
        "## Controls",
        "",
        f"- No-capacity Expert Choice identical to Hard Normal HxV: `{results['controls']['no_capacity_identical_to_hard_hxv']}`.",
        "- Random-score and permuted-location controls are reported in `results.json`; neither uses validation targets to form assignments.",
        "",
        "## Oracle Diagnostics",
        "",
        "| Oracle / Non-deployable | MAE | MSE |",
        "|---|---:|---:|",
    ]
    for r in results["oracle_diagnostics"]:
        lines.append(f"| {r['method']} | {r['mae']:.6f} | {r['mse']:.6f} |")
    lines += [
        "",
        "## Integrity",
        "",
        f"- No test cache/file loaded: `{results['integrity']['no_test_cache_loaded']}`.",
        f"- Expert ordering verified: `{results['integrity']['expert_ordering_verified']}`.",
        f"- Same score tensor used for Hard HxV and Expert Choice: `{results['integrity']['same_score_tensor_for_hard_and_expert_choice']}`.",
        f"- Assignments train-only: `{results['integrity']['assignments_train_only']}`.",
        f"- Checkpoints unchanged: `{results['integrity']['checkpoints_unchanged']}`.",
        "",
        f"Existing Electricity HxV reference note: previous frozen multidataset report lists HxV around MAE `{REFERENCE_ELECTRICITY_HXV_MAE:.6f}`. This run's `Existing soft HxV` row reuses the canonical causal HxV utility, while the hard allocation rows use a static train-only score tensor, so exact equality is not required.",
    ]
    return "\n".join(lines)


def main() -> None:
    started = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in (TRAIN_CACHE_PATH, VAL_CACHE_PATH, NORMALIZER_PATH):
        refuse_test(path)

    print("[expert-choice-hv] loading Electricity router_train/router_val caches", flush=True)
    train_cache = load_cache(TRAIN_CACHE_PATH, "router_train_20_60")
    val_cache = load_cache(VAL_CACHE_PATH, "router_val_60_80")
    std = load_std(NORMALIZER_PATH, int(val_cache["num_features"])).to(torch.float32)
    expert_idx = expert_indices(train_cache)
    if expert_idx != expert_indices(val_cache):
        raise AssertionError("Train and validation expert indices differ")
    before_hashes = checkpoint_hashes()

    h = int(val_cache["forecast_horizon"])
    v = int(val_cache["num_features"])
    total_cells = h * v
    score_khv, train_error_hvk = build_score_tensor(train_cache, expert_idx, std)
    hard = hard_normal_assignment(score_khv)

    print("[expert-choice-hv] solving constrained global assignments", flush=True)
    expert_choice = {factor: solve_capacity_assignment(score_khv, factor) for factor in CAPACITY_FACTORS}
    random_score = random_scores_like(score_khv)
    permuted_score = permuted_score_locations(score_khv)
    random_controls = {factor: solve_capacity_assignment(random_score, factor) for factor in CAPACITY_FACTORS}
    permuted_controls = {factor: solve_capacity_assignment(permuted_score, factor) for factor in CAPACITY_FACTORS}
    no_capacity_assignment = score_khv.argmax(dim=0).to(torch.long)
    no_capacity_identical = bool(torch.equal(no_capacity_assignment, hard["assignment"]))

    forecasts_val = forecasts_for(val_cache, expert_idx)
    equal_pred = forecasts_val.mean(dim=-1)
    equal_row = metrics_for_prediction("Equal ensemble", equal_pred, val_cache, std, summarize_static_assignment("Equal ensemble", None, len(CORE_EXPERTS), weights=[1 / 3] * 3))
    hard_pred = predict_static_assignment(forecasts_val, hard["assignment"])
    hard_row = metrics_for_prediction("Hard Normal HxV", hard_pred, val_cache, std, summarize_static_assignment("Hard Normal HxV", hard["assignment"], len(CORE_EXPERTS)))

    print("[expert-choice-hv] evaluating deployable methods on router_val", flush=True)
    soft_pred, soft_extra = granularity_ema_prediction(val_cache, train_cache, expert_idx, std, mode="hv")
    soft_row = metrics_for_prediction(
        "Existing soft HxV",
        soft_pred,
        val_cache,
        std,
        {
            "method": "Existing soft HxV",
            "assigned_cells": None,
            "allocation_pct": None,
            "soft_hxv_extra": soft_extra,
        },
    )

    deployable_metric_rows = [equal_row, soft_row, hard_row]
    for factor, assignment in expert_choice.items():
        pred = predict_static_assignment(forecasts_val, assignment["assignment"])
        deployable_metric_rows.append(
            metrics_for_prediction(
                f"Expert Choice cap {factor:.2f}",
                pred,
                val_cache,
                std,
                summarize_static_assignment(f"Expert Choice cap {factor:.2f}", assignment["assignment"], len(CORE_EXPERTS)),
            )
        )

    control_rows = []
    for label, assignments in (("Random scores", random_controls), ("Permuted HxV locations", permuted_controls)):
        for factor, assignment in assignments.items():
            pred = predict_static_assignment(forecasts_val, assignment["assignment"])
            control_rows.append(
                metrics_for_prediction(
                    f"{label} cap {factor:.2f}",
                    pred,
                    val_cache,
                    std,
                    summarize_static_assignment(f"{label} cap {factor:.2f}", assignment["assignment"], len(CORE_EXPERTS)),
                )
            )

    no_cap_pred = predict_static_assignment(forecasts_val, no_capacity_assignment)
    control_rows.append(
        metrics_for_prediction(
            "Expert Choice no capacity",
            no_cap_pred,
            val_cache,
            std,
            summarize_static_assignment("Expert Choice no capacity", no_capacity_assignment, len(CORE_EXPERTS)),
        )
    )

    oracle = oracle_predictions(val_cache, forecasts_val, std)
    oracle_rows = []
    for name, (pred, assignment) in oracle.items():
        label = "Dynamic oracle per-window HxV (ORACLE / NON-DEPLOYABLE)" if name == "dynamic_oracle" else "Static val-average HxV oracle (ORACLE / NON-DEPLOYABLE)"
        if assignment.ndim == 3:
            allocation = {"method": label, "assigned_cells": "per-window dynamic oracle", "allocation_pct": "per-window dynamic oracle"}
        else:
            allocation = summarize_static_assignment(label, assignment, len(CORE_EXPERTS))
        oracle_rows.append(metrics_for_prediction(label, pred, val_cache, std, allocation))

    equal_mae = equal_row["mae"]
    hard_mae = hard_row["mae"]
    deployable_results = [serial_metric(r, equal_mae, hard_mae) for r in deployable_metric_rows]
    controls_serial = [serial_metric(r, equal_mae, hard_mae) for r in control_rows]
    oracle_serial = [{k: v for k, v in r.items() if k != "per_window_mae"} for r in oracle_rows]

    stats = []
    for factor, row in zip(CAPACITY_FACTORS, deployable_metric_rows[3:]):
        stats.append(comparison_stats(row["per_window_mae"], hard_row["per_window_mae"], f"Expert Choice cap {factor:.2f} vs Hard Normal HxV"))

    ec_rows = [r for r in deployable_results if r["method"].startswith("Expert Choice cap")]
    stat_by_method = {s["comparison"].split(" vs ")[0]: s for s in stats}
    strong_go_variants = []
    weak_variants = []
    for row in ec_rows:
        stat = stat_by_method[row["method"]]
        improvement = -row["delta_mae_vs_hard_hxv"]
        if improvement >= 0.0005 and stat["block24"]["ci_excludes_zero"] and stat["block24"]["mean_delta"] < 0 and stat["decision_inputs"]["every_12th_agrees"]:
            strong_go_variants.append(row["method"])
        elif improvement > 0:
            weak_variants.append(row["method"])
    best_ec = min(ec_rows, key=lambda r: r["mae"])
    strong = bool(strong_go_variants)
    weak = bool(weak_variants)
    verdict = "STRONG GO" if strong else ("WEAK" if weak else "NO-GO")
    interpretation = (
        f"Among the predeclared capacity settings, `{best_ec['method']}` had the lowest reported router_val MAE with delta {best_ec['delta_mae_vs_hard_hxv']:+.6f} versus Hard Normal HxV. "
        f"Strong-Go variants by the fixed rule: {strong_go_variants or 'none'}; Weak variants: {weak_variants or 'none'}. "
        "Because Hard Normal HxV and all Expert Choice variants use the same train-derived score tensor, any difference comes from the capacity-constrained assignment mechanism rather than a changed competence model. "
        f"No-capacity Expert Choice was exactly identical to Hard Normal HxV: {no_capacity_identical}. "
        "This pilot does not select a deployment capacity; it reports all predeclared capacities."
    )

    after_hashes = checkpoint_hashes()
    integrity = {
        "dataset": DATASET,
        "loaded_paths": [str(TRAIN_CACHE_PATH.relative_to(ROOT)), str(VAL_CACHE_PATH.relative_to(ROOT)), str(NORMALIZER_PATH.relative_to(ROOT))],
        "no_test_cache_loaded": True,
        "cache_roles": {"router_train": train_cache.get("cache_role"), "router_val": val_cache.get("cache_role")},
        "expert_ordering_verified": tuple(train_cache["expert_names"]) == EXPECTED_CACHE_EXPERT_ORDER and tuple(val_cache["expert_names"]) == EXPECTED_CACHE_EXPERT_ORDER,
        "core_experts": list(CORE_EXPERTS),
        "core_expert_indices": expert_idx,
        "assignments_train_only": True,
        "same_score_tensor_for_hard_and_expert_choice": True,
        "normalization": "per-location absolute errors divided by train checkpoint scaler std via existing per_location_error utility",
        "horizon": h,
        "num_variables": v,
        "total_hxv_cells": total_cells,
        "checkpoints_unchanged": before_hashes == after_hashes,
        "checkpoint_hashes_before": before_hashes,
        "checkpoint_hashes_after": after_hashes,
        "no_capacity_identical_to_hard_hxv": no_capacity_identical,
        "result": "PASS" if before_hashes == after_hashes and no_capacity_identical else "FAIL",
    }
    if integrity["result"] != "PASS":
        raise AssertionError(f"Integrity failure: {integrity}")

    results = {
        "experiment": "expert_choice_hv_pilot",
        "created_at_utc": now_utc(),
        "git_commit_sha": git_commit_sha(),
        "dataset": DATASET,
        "test_split_accessed": False,
        "verdict": verdict,
        "interpretation": interpretation,
        "strong_go_variants": strong_go_variants,
        "weak_variants": weak_variants,
        "reference_electricity_hxv_mae_from_existing_report": REFERENCE_ELECTRICITY_HXV_MAE,
        "score_definition": "score[k,h,v] = -mean_router_train_normalized_abs_error[k,h,v]",
        "train_error_mean_hvk_shape": list(train_error_hvk.shape),
        "hard_normal_assignment": {k: v for k, v in hard.items() if k != "assignment"},
        "expert_choice_assignments": {f"{factor:.2f}": {k: v for k, v in a.items() if k != "assignment"} for factor, a in expert_choice.items()},
        "deployable_results": deployable_results,
        "controls": {
            "no_capacity_identical_to_hard_hxv": no_capacity_identical,
            "control_results": controls_serial,
            "random_assignment_metadata": {f"{factor:.2f}": {k: v for k, v in a.items() if k != "assignment"} for factor, a in random_controls.items()},
            "permuted_assignment_metadata": {f"{factor:.2f}": {k: v for k, v in a.items() if k != "assignment"} for factor, a in permuted_controls.items()},
        },
        "expert_choice_vs_hard_stats": stats,
        "oracle_diagnostics": oracle_serial,
        "integrity": integrity,
        "runtime_seconds": time.perf_counter() - started,
    }

    write_json(OUT_DIR / "results.json", results)
    write_json(OUT_DIR / "integrity_report.json", integrity)
    write_csv(OUT_DIR / "method_metrics.csv", deployable_results + controls_serial + oracle_serial)
    (OUT_DIR / "RESULTS.md").write_text(build_results_md(results), encoding="utf-8")
    print(f"[expert-choice-hv] wrote artifacts to {OUT_DIR}", flush=True)
    print(f"[expert-choice-hv] verdict={verdict} best={best_ec['method']} delta_vs_hard={best_ec['delta_mae_vs_hard_hxv']:+.6f}", flush=True)


if __name__ == "__main__":
    main()
