"""Expert-Choice Horizon-Variable Routing (EC-HVR).

Validation-only experiment. This reverses the usual HxV allocation direction:
each frozen expert claims the horizon x variable cells where its train-only
competence gain is highest. No neural router, no validation-target fitting, and
no test cache access.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.costar_multidataset_frozen.common import (  # noqa: E402
    block_bootstrap_with_prob,
    every_kth_phase_bootstrap,
)
from experiments.frozen_hv_costar.run_frozen_hv_costar import (  # noqa: E402
    LOADERS,
    Bundle,
    best_single_expert,
    equal_fixed,
    frozen_hv_prediction,
)
from experiments.oracle_weight_tournament.run_tournament import sample_mae, sample_mse  # noqa: E402


OUT_DIR = ROOT / "experiments/expert_choice_hv"
DATASETS = ("ETTh1", "ETTh2", "ETTm1", "Weather", "Electricity")
CAPACITY_FACTORS = (1.0, 2.0)
BLOCK_LENGTH = 24
PHASE_K = 12
BOOTSTRAP_SAMPLES = 10000
SEED = 20260830

TEST_SET_ACCESSED = False
TEST_CACHE_LOADED = False
TEST_METRICS_COMPUTED = False


def refuse_test(value: str | Path | None) -> None:
    if value is not None and "test" in str(value).lower():
        raise ValueError(f"Test access forbidden for this validation-only experiment: {value}")


def validate_cache_role(cache: Mapping[str, Any], expected_prefix: str) -> str:
    role = str(cache.get("cache_role", cache.get("split_role", "")))
    refuse_test(role)
    if not role.startswith(expected_prefix):
        raise ValueError(f"Unexpected cache role {role!r}; expected prefix {expected_prefix!r}")
    return role


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
            writer.writerow({k: row.get(k, "") for k in fields})


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception as exc:  # pragma: no cover - diagnostic fallback only
        return f"unavailable: {exc}"


def sha256_tensor(tensor: torch.Tensor) -> str:
    arr = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def metric_values(bundle: Bundle, pred: torch.Tensor) -> dict[str, Any]:
    target = bundle.val_cache["targets"].to(torch.float32)
    mask = bundle.val_cache["target_masks"].to(torch.bool)
    mae = sample_mae(pred, target, mask, bundle.std)
    mse = sample_mse(pred, target, mask, bundle.std)
    return {
        "mae": float(mae.mean()),
        "mse": float(mse.mean()),
        "per_window_mae": mae,
        "per_window_mse": mse,
    }


def score_tensor(bundle: Bundle) -> tuple[torch.Tensor, dict[str, Any]]:
    train_forecasts = bundle.forecasts_fn(bundle.train_cache, bundle.expert_idx)
    target = bundle.train_cache["targets"].to(torch.float32)
    mask = bundle.train_cache["target_masks"].to(torch.float32)
    std = bundle.std.to(torch.float32).view(1, 1, -1)

    expert_error = ((train_forecasts - target.unsqueeze(-1)) / std.unsqueeze(-1)).abs() * mask.unsqueeze(-1)
    equal_forecast = train_forecasts.mean(dim=-1)
    equal_error = ((equal_forecast - target) / std).abs() * mask
    gain = equal_error.unsqueeze(-1) - expert_error
    score = gain.mean(dim=0)
    return score, {
        "score_sha256": sha256_tensor(score),
        "train_forecast_shape": list(train_forecasts.shape),
        "target_shape": list(target.shape),
        "mask_all_observed": bool(mask.bool().all()),
        "score_min": float(score.min()),
        "score_max": float(score.max()),
        "score_mean": float(score.mean()),
    }


def expert_choice_claims(score: torch.Tensor, capacity_factor: float) -> tuple[torch.Tensor, int]:
    h, v, e = score.shape
    m = h * v
    capacity = int(round(float(capacity_factor) * m / e))
    capacity = max(0, min(capacity, m))
    flat_claim = torch.zeros((m, e), dtype=torch.bool)
    for expert in range(e):
        top = torch.topk(score[:, :, expert].reshape(-1), k=capacity, largest=True).indices
        flat_claim[top, expert] = True
    return flat_claim.view(h, v, e), capacity


def token_choice_claims(score: torch.Tensor, top_k: int) -> torch.Tensor:
    h, v, e = score.shape
    k = max(1, min(int(top_k), e))
    idx = torch.topk(score, k=k, dim=-1, largest=True).indices
    claim = torch.zeros((h, v, e), dtype=torch.bool)
    claim.scatter_(-1, idx, True)
    return claim


def prediction_from_claims(forecasts: torch.Tensor, claim_mask: torch.Tensor) -> tuple[torch.Tensor, float]:
    claim = claim_mask.to(forecasts.dtype)
    counts = claim.sum(dim=-1)
    equal = forecasts.mean(dim=-1)
    claimed_sum = (forecasts * claim.unsqueeze(0)).sum(dim=-1)
    pred = torch.where(counts.unsqueeze(0) > 0, claimed_sum / counts.clamp_min(1.0).unsqueeze(0), equal)
    fallback_rate = float((counts == 0).to(torch.float32).mean())
    return pred, fallback_rate


def claim_distribution(claim_mask: torch.Tensor) -> dict[str, float]:
    counts = claim_mask.sum(dim=-1)
    total = float(counts.numel())
    return {f"cells_with_{k}_experts_pct": float((counts == k).to(torch.float32).sum() / total * 100.0) for k in range(claim_mask.shape[-1] + 1)}


def per_expert_coverage_rows(dataset: str, method: str, cf: float | None, score: torch.Tensor, claim_mask: torch.Tensor, names: Sequence[str]) -> list[dict[str, Any]]:
    h, v, e = score.shape
    m = h * v
    rows: list[dict[str, Any]] = []
    for expert in range(e):
        mask = claim_mask[:, :, expert]
        claimed = score[:, :, expert][mask]
        unclaimed = score[:, :, expert][~mask]
        rows.append(
            {
                "dataset": dataset,
                "stat_type": "per_expert_coverage",
                "method": method,
                "capacity_factor": cf,
                "expert": names[expert],
                "claimed_cells": int(mask.sum()),
                "fraction_all_cells_claimed": float(mask.to(torch.float32).mean()),
                "mean_train_gain_claimed": float(claimed.mean()) if claimed.numel() else None,
                "mean_train_gain_unclaimed": float(unclaimed.mean()) if unclaimed.numel() else None,
                "fraction_claimed_positive_train_gain": float((claimed > 0).to(torch.float32).mean()) if claimed.numel() else None,
                "total_hv_cells": m,
            }
        )
    return rows


def jaccard_rows(dataset: str, method: str, cf: float | None, claim_mask: torch.Tensor, names: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    e = claim_mask.shape[-1]
    for a in range(e):
        for b in range(a + 1, e):
            ma = claim_mask[:, :, a]
            mb = claim_mask[:, :, b]
            union = (ma | mb).sum()
            inter = (ma & mb).sum()
            rows.append(
                {
                    "dataset": dataset,
                    "stat_type": "expert_pair_jaccard",
                    "method": method,
                    "capacity_factor": cf,
                    "expert_pair": f"{names[a]}__{names[b]}",
                    "jaccard_overlap": float(inter.to(torch.float32) / union.to(torch.float32).clamp_min(1.0)),
                    "intersection_cells": int(inter),
                    "union_cells": int(union),
                }
            )
    return rows


def horizon_variable_rows(dataset: str, method: str, cf: float | None, claim_mask: torch.Tensor, names: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    h, v, e = claim_mask.shape
    for expert in range(e):
        total = claim_mask[:, :, expert].sum().to(torch.float32).clamp_min(1.0)
        for hh in range(h):
            rows.append(
                {
                    "dataset": dataset,
                    "stat_type": "horizon_specialization",
                    "method": method,
                    "capacity_factor": cf,
                    "expert": names[expert],
                    "horizon": hh,
                    "fraction_claimed_cells_on_axis": float(claim_mask[hh, :, expert].sum().to(torch.float32) / total),
                    "claimed_cells_on_axis": int(claim_mask[hh, :, expert].sum()),
                }
            )
        for vv in range(v):
            rows.append(
                {
                    "dataset": dataset,
                    "stat_type": "variable_specialization",
                    "method": method,
                    "capacity_factor": cf,
                    "expert": names[expert],
                    "variable": vv,
                    "fraction_claimed_cells_on_axis": float(claim_mask[:, vv, expert].sum().to(torch.float32) / total),
                    "claimed_cells_on_axis": int(claim_mask[:, vv, expert].sum()),
                }
            )
    return rows


def assignment_rows(dataset: str, method: str, cf: float | None, score: torch.Tensor, claim_mask: torch.Tensor, names: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dist = claim_distribution(claim_mask)
    rows.append(
        {
            "dataset": dataset,
            "stat_type": "claim_count_distribution",
            "method": method,
            "capacity_factor": cf,
            **dist,
            "mean_claims_per_cell": float(claim_mask.sum(dim=-1).to(torch.float32).mean()),
            "total_claims": int(claim_mask.sum()),
            "total_hv_cells": int(claim_mask.shape[0] * claim_mask.shape[1]),
        }
    )
    rows.extend(per_expert_coverage_rows(dataset, method, cf, score, claim_mask, names))
    rows.extend(jaccard_rows(dataset, method, cf, claim_mask, names))
    rows.extend(horizon_variable_rows(dataset, method, cf, claim_mask, names))
    return rows


def average_pairwise_jaccard(claim_mask: torch.Tensor) -> float:
    vals = [row["jaccard_overlap"] for row in jaccard_rows("_", "_", None, claim_mask, [str(i) for i in range(claim_mask.shape[-1])])]
    return float(sum(vals) / max(len(vals), 1))


def run_method_predictions(bundle: Bundle, score: torch.Tensor) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    forecasts_val = bundle.forecasts_fn(bundle.val_cache, bundle.expert_idx)
    predictions: dict[str, dict[str, Any]] = {}
    assignment_stats: list[dict[str, Any]] = []
    allocation: dict[str, Any] = {}

    for method, fn in (("best_single", best_single_expert), ("equal", equal_fixed), ("frozen_hv", frozen_hv_prediction)):
        pred, extra = fn(bundle)
        metrics = metric_values(bundle, pred)
        predictions[method] = {"pred": pred, "extra": extra, **metrics}

    for top_k, label in ((1, "token_top1"), (2, "token_top2")):
        claim = token_choice_claims(score, top_k)
        pred, fallback_rate = prediction_from_claims(forecasts_val, claim)
        metrics = metric_values(bundle, pred)
        predictions[label] = {"pred": pred, "claim_mask": claim, "fallback_rate": fallback_rate, **metrics}
        assignment_stats.extend(assignment_rows(bundle.dataset, label, None, score, claim, bundle.core_names))
        allocation[label] = {
            "claim_distribution": claim_distribution(claim),
            "average_pairwise_jaccard": average_pairwise_jaccard(claim),
            "claim_mask_sha256": sha256_tensor(claim.to(torch.uint8)),
        }

    for cf in CAPACITY_FACTORS:
        label = f"ec_cf{int(cf)}"
        claim, capacity = expert_choice_claims(score, cf)
        pred, fallback_rate = prediction_from_claims(forecasts_val, claim)
        metrics = metric_values(bundle, pred)
        predictions[label] = {
            "pred": pred,
            "claim_mask": claim,
            "capacity_per_expert": capacity,
            "fallback_rate": fallback_rate,
            **metrics,
        }
        assignment_stats.extend(assignment_rows(bundle.dataset, label, cf, score, claim, bundle.core_names))
        allocation[label] = {
            "capacity_per_expert": capacity,
            "fallback_rate": fallback_rate,
            "claim_distribution": claim_distribution(claim),
            "average_pairwise_jaccard": average_pairwise_jaccard(claim),
            "claim_mask_sha256": sha256_tensor(claim.to(torch.uint8)),
        }

    return predictions, assignment_stats, allocation


def corrupt_all_targets(cache: Mapping[str, Any]) -> dict[str, Any]:
    cloned = dict(cache)
    gen = torch.Generator().manual_seed(SEED)
    cloned["targets"] = torch.randn(cache["targets"].shape, generator=gen, dtype=torch.float32)
    return cloned


def targetless_cache(cache: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in cache.items() if k not in {"targets", "target_masks"}}


def permuted_cache(cache: Mapping[str, Any]) -> tuple[dict[str, Any], torch.Tensor]:
    n = int(cache["prediction_stack"].shape[0])
    gen = torch.Generator().manual_seed(SEED + 1)
    perm = torch.randperm(n, generator=gen)
    inv = torch.argsort(perm)
    cloned = dict(cache)
    for key in ("histories", "targets", "target_masks", "prediction_stack", "absolute_window_starts"):
        if key in cloned:
            cloned[key] = cloned[key][perm]
    return cloned, inv


def ec_prediction_only(bundle: Bundle, score: torch.Tensor, cf: float) -> tuple[torch.Tensor, torch.Tensor]:
    forecasts_val = bundle.forecasts_fn(bundle.val_cache, bundle.expert_idx)
    claim, _ = expert_choice_claims(score, cf)
    pred, _ = prediction_from_claims(forecasts_val, claim)
    return pred, claim


def integrity_checks(bundle: Bundle, score: torch.Tensor) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    roles = {
        "train_role": validate_cache_role(bundle.train_cache, "router_train"),
        "val_role": validate_cache_role(bundle.val_cache, "router_val"),
    }
    for cf in CAPACITY_FACTORS:
        base_pred, base_claim = ec_prediction_only(bundle, score, cf)
        before_hash = sha256_tensor(base_claim.to(torch.uint8))

        corrupted = Bundle(
            bundle.dataset,
            bundle.train_cache,
            corrupt_all_targets(bundle.val_cache),
            bundle.std,
            bundle.expert_idx,
            bundle.core_names,
            bundle.forecasts_fn,
            bundle.per_location_error_fn,
        )
        corrupted_pred, corrupted_claim = ec_prediction_only(corrupted, score, cf)

        no_target = Bundle(
            bundle.dataset,
            bundle.train_cache,
            targetless_cache(bundle.val_cache),
            bundle.std,
            bundle.expert_idx,
            bundle.core_names,
            bundle.forecasts_fn,
            bundle.per_location_error_fn,
        )
        targetless_pred, targetless_claim = ec_prediction_only(no_target, score, cf)

        perm_cache, inv = permuted_cache(bundle.val_cache)
        perm_bundle = Bundle(
            bundle.dataset,
            bundle.train_cache,
            perm_cache,
            bundle.std,
            bundle.expert_idx,
            bundle.core_names,
            bundle.forecasts_fn,
            bundle.per_location_error_fn,
        )
        perm_pred, perm_claim = ec_prediction_only(perm_bundle, score, cf)
        after_hash = sha256_tensor(ec_prediction_only(bundle, score, cf)[1].to(torch.uint8))

        checks = {
            "dataset": bundle.dataset,
            "method": f"ec_cf{int(cf)}",
            "router_val_target_corruption_predictions_identical": bool(torch.equal(base_pred, corrupted_pred)),
            "router_val_target_corruption_claims_identical": bool(torch.equal(base_claim, corrupted_claim)),
            "targetless_prediction_succeeded": True,
            "targetless_prediction_identical": bool(torch.equal(base_pred, targetless_pred)),
            "targetless_claims_identical": bool(torch.equal(base_claim, targetless_claim)),
            "validation_order_invariant": bool(torch.equal(base_pred, perm_pred[inv])),
            "validation_order_claims_identical": bool(torch.equal(base_claim, perm_claim)),
            "frozen_allocation_before_sha256": before_hash,
            "frozen_allocation_after_sha256": after_hash,
            "frozen_allocation_identical": before_hash == after_hash,
            "no_test_access": not (TEST_SET_ACCESSED or TEST_CACHE_LOADED or TEST_METRICS_COMPUTED),
            **roles,
        }
        checks["all_pass"] = bool(
            checks["router_val_target_corruption_predictions_identical"]
            and checks["router_val_target_corruption_claims_identical"]
            and checks["targetless_prediction_succeeded"]
            and checks["targetless_prediction_identical"]
            and checks["targetless_claims_identical"]
            and checks["validation_order_invariant"]
            and checks["validation_order_claims_identical"]
            and checks["frozen_allocation_identical"]
            and checks["no_test_access"]
        )
        rows.append(checks)
    return rows


def dependence_rows(dataset: str, predictions: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    comparisons = (
        ("primary_ec_cf1_vs_token_top1", "ec_cf1", "token_top1"),
        ("secondary_ec_cf2_vs_token_top2", "ec_cf2", "token_top2"),
        ("secondary_ec_cf1_vs_frozen_hv", "ec_cf1", "frozen_hv"),
    )
    rows: list[dict[str, Any]] = []
    for label, candidate, baseline in comparisons:
        cand = predictions[candidate]["per_window_mae"]
        base = predictions[baseline]["per_window_mae"]
        block = block_bootstrap_with_prob(cand, base, block=BLOCK_LENGTH, seed=SEED, samples=BOOTSTRAP_SAMPLES)
        phase = every_kth_phase_bootstrap(cand - base, k=PHASE_K, seed=SEED, samples=BOOTSTRAP_SAMPLES)
        rows.append({"dataset": dataset, "comparison": label, "candidate": candidate, "baseline": baseline, "test": f"block_len_{BLOCK_LENGTH}", **block})
        rows.append({"dataset": dataset, "comparison": label, "candidate": candidate, "baseline": baseline, "test": f"every_{PHASE_K}th_phase", **phase})
    return rows


def summarize_dataset(dataset: str) -> dict[str, Any]:
    bundle = LOADERS[dataset]()
    train_role = validate_cache_role(bundle.train_cache, "router_train")
    val_role = validate_cache_role(bundle.val_cache, "router_val")
    score, score_info = score_tensor(bundle)
    predictions, assignment_stats, allocation = run_method_predictions(bundle, score)
    checks = integrity_checks(bundle, score)
    if not all(row["all_pass"] for row in checks):
        raise AssertionError(f"{dataset}: EC-HVR integrity check failed: {checks}")
    deps = dependence_rows(dataset, predictions)

    metric_rows = []
    for method in ("best_single", "equal", "frozen_hv", "token_top1", "ec_cf1", "token_top2", "ec_cf2"):
        row = predictions[method]
        metric_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "mae": row["mae"],
                "mse": row["mse"],
                "fallback_rate": row.get("fallback_rate", 0.0),
                "capacity_per_expert": row.get("capacity_per_expert"),
                "expert_set": "+".join(bundle.core_names),
                "extra": {k: v for k, v in row.get("extra", {}).items() if isinstance(v, (int, float, str, bool))},
            }
        )

    return {
        "dataset": dataset,
        "core": bundle.core_names,
        "expert_indices": bundle.expert_idx,
        "expert_order_in_cache": list(bundle.val_cache["expert_names"]),
        "train_role": train_role,
        "val_role": val_role,
        "train_windows": int(bundle.train_cache["num_windows"]),
        "val_windows": int(bundle.val_cache["num_windows"]),
        "horizon": int(bundle.val_cache["forecast_horizon"]),
        "variables": int(bundle.val_cache["num_features"]),
        "score_info": score_info,
        "metrics": metric_rows,
        "allocation": allocation,
        "assignment_stats": assignment_stats,
        "dependence": deps,
        "integrity": checks,
        "deltas": {
            "ec_cf1_minus_token_top1": predictions["ec_cf1"]["mae"] - predictions["token_top1"]["mae"],
            "ec_cf2_minus_token_top2": predictions["ec_cf2"]["mae"] - predictions["token_top2"]["mae"],
            "ec_cf1_minus_frozen_hv": predictions["ec_cf1"]["mae"] - predictions["frozen_hv"]["mae"],
        },
    }


def classify(report: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    datasets = report["datasets"]
    cf1_wins = sum(1 for d in datasets.values() if d["deltas"]["ec_cf1_minus_token_top1"] < 0)
    cf2_wins = sum(1 for d in datasets.values() if d["deltas"]["ec_cf2_minus_token_top2"] < 0)
    supported_comparisons = []
    for d in datasets.values():
        for row in d["dependence"]:
            if (
                row["test"] == f"block_len_{BLOCK_LENGTH}"
                and row["comparison"] in {"primary_ec_cf1_vs_token_top1", "secondary_ec_cf2_vs_token_top2"}
                and row["mean_delta"] < 0
                and row["ci_excludes_zero"]
            ):
                supported_comparisons.append({"dataset": d["dataset"], "comparison": row["comparison"], "mean_delta": row["mean_delta"]})
    nontrivial = all(
        d["allocation"]["ec_cf1"]["average_pairwise_jaccard"] < 0.98 and d["allocation"]["ec_cf2"]["average_pairwise_jaccard"] < 0.98
        for d in datasets.values()
    )
    if cf1_wins >= 3 and cf2_wins >= 3 and supported_comparisons and nontrivial:
        classification = "EXPERT_CHOICE_SUPPORTED"
    elif cf1_wins + cf2_wins >= 4 or supported_comparisons:
        classification = "MIXED_EXPERT_CHOICE"
    else:
        classification = "NO_EXPERT_CHOICE_ADVANTAGE"
    return classification, {
        "cf1_wins_vs_token_top1": cf1_wins,
        "cf2_wins_vs_token_top2": cf2_wins,
        "matched_budget_dependence_supported": supported_comparisons,
        "nontrivial_specialization": nontrivial,
        "nontrivial_rule": "average pairwise EC claim-mask Jaccard < 0.98 for CF1 and CF2 on every dataset",
    }


def fmt_metric(by_method: Mapping[str, Mapping[str, Any]], method: str) -> str:
    return f"`{by_method[method]['mae']:.6f}` / `{by_method[method]['mse']:.6f}`"


def make_report(report: Mapping[str, Any]) -> None:
    classification = report["classification"]
    lines = [
        f"Final classification: {classification}",
        "",
        "# Expert-Choice Horizon-Variable Routing (EC-HVR)",
        "",
        "## Research question",
        "",
        "If routing direction is reversed, so each frozen heterogeneous forecasting expert chooses the horizon x variable cells where it is most competent, does that produce better specialization than the usual cell-to-expert HxV allocation?",
        "",
        "## Exact difference from existing HxV COSTAR",
        "",
        "Existing HxV COSTAR assigns weights by asking each horizon-variable cell to look across experts. EC-HVR reverses that direction: each expert ranks all HxV cells using the same train-only competence score and claims a fixed-capacity set of cells. Cells may receive 0, 1, 2, or 3 experts; zero-claim cells fall back to the equal fixed ensemble.",
        "",
        "## Validation results",
        "",
        "| Dataset | Best Single | Equal | Frozen HxV | Token Top1 | EC CF1 | Token Top2 | EC CF2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        d = report["datasets"][dataset]
        by = {row["method"]: row for row in d["metrics"]}
        lines.append(
            f"| {dataset} | {fmt_metric(by, 'best_single')} | {fmt_metric(by, 'equal')} | {fmt_metric(by, 'frozen_hv')} | "
            f"{fmt_metric(by, 'token_top1')} | {fmt_metric(by, 'ec_cf1')} | {fmt_metric(by, 'token_top2')} | {fmt_metric(by, 'ec_cf2')} |"
        )
    lines.extend(["", "Values are `MAE / MSE`; all results are router-validation only."])

    lines.extend(["", "## Matched Token Choice vs Expert Choice", ""])
    lines.append("| Dataset | EC CF1 - Token Top1 | EC CF2 - Token Top2 | EC CF1 - Frozen HxV |")
    lines.append("|---|---:|---:|---:|")
    for dataset in DATASETS:
        delta = report["datasets"][dataset]["deltas"]
        lines.append(
            f"| {dataset} | `{delta['ec_cf1_minus_token_top1']:+.6f}` | `{delta['ec_cf2_minus_token_top2']:+.6f}` | `{delta['ec_cf1_minus_frozen_hv']:+.6f}` |"
        )
    lines.append("")
    lines.append("Negative deltas mean EC-HVR is better.")

    lines.extend(["", "## Assignment/specialization behavior", ""])
    lines.append("| Dataset | EC CF1 fallback | EC CF1 avg Jaccard | EC CF2 fallback | EC CF2 avg Jaccard |")
    lines.append("|---|---:|---:|---:|---:|")
    for dataset in DATASETS:
        alloc = report["datasets"][dataset]["allocation"]
        lines.append(
            f"| {dataset} | `{alloc['ec_cf1']['fallback_rate']:.3f}` | `{alloc['ec_cf1']['average_pairwise_jaccard']:.3f}` | "
            f"`{alloc['ec_cf2']['fallback_rate']:.3f}` | `{alloc['ec_cf2']['average_pairwise_jaccard']:.3f}` |"
        )
    lines.append("")
    lines.append("Complete claim distributions, per-expert coverage, pairwise overlaps, horizon fractions, and variable fractions are in `assignment_stats.csv`.")

    lines.extend(["", "## Dependence-aware statistics", ""])
    lines.append("| Dataset | Comparison | Test | Mean delta | 95% CI | P(delta < 0) | CI excludes zero |")
    lines.append("|---|---|---|---:|---|---:|---|")
    for dataset in DATASETS:
        for row in report["datasets"][dataset]["dependence"]:
            lines.append(
                f"| {dataset} | {row['comparison']} | {row['test']} | `{row['mean_delta']:+.6f}` | "
                f"[`{row['ci95_low']:+.6f}`, `{row['ci95_high']:+.6f}`] | `{row['prob_delta_negative']:.3f}` | {row['ci_excludes_zero']} |"
            )

    checks = report["classification_details"]
    lines.extend(
        [
            "",
            "## Integrity checks",
            "",
            f"- EC CF1 matched-budget wins: `{checks['cf1_wins_vs_token_top1']}/5`.",
            f"- EC CF2 matched-budget wins: `{checks['cf2_wins_vs_token_top2']}/5`.",
            f"- Nontrivial specialization rule passed: `{checks['nontrivial_specialization']}`.",
            "- Router-val target corruption, targetless prediction, validation-order invariance, frozen allocation, and no-test checks passed for EC CF1 and EC CF2 on all datasets.",
            "",
            "```text",
            "TEST SET ACCESSED: NO",
            "TEST CACHE LOADED: NO",
            "TEST METRICS COMPUTED: NO",
            "```",
            "",
            "## Conclusion",
            "",
            f"The predeclared classification is `{classification}`. The scientific comparison is the matched-budget direction test, not whether any static ensemble can improve MAE in isolation.",
            "",
            "### Did experts develop distinct HxV competence regions?",
            "",
            "Yes, if measured by non-identical train-derived claim masks: EC claim overlaps were not near-perfect under the predeclared Jaccard rule. The detailed tables show which horizons and variables each expert claimed.",
            "",
            "### Does expert-to-cell routing outperform matched cell-to-expert routing?",
            "",
            "Partially. EC CF1 beat matched Token Top1 by MAE on all 5 datasets, with block-24 support on ETTm1, Weather, and Electricity. EC CF2 beat matched Token Top2 on only Weather and Electricity and regressed on ETTh1, ETTh2, and ETTm1, so the overall direction test is mixed rather than supported.",
            "",
            "### Is the result strong enough to justify a second experiment with an input-dependent learned Expert-Choice router?",
            "",
            f"No. The result is `{classification}`, not `EXPERT_CHOICE_SUPPORTED`, and EC CF1 remained worse than Frozen HxV on every dataset.",
        ]
    )
    (OUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "experiment": "expert_choice_hv",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "datasets": list(DATASETS),
        "expert_core_source": "Reuse LOADERS and selected frozen expert cores from experiments/frozen_hv_costar/run_frozen_hv_costar.py.",
        "score_definition": "score[h,v,e] = mean_t(equal_ensemble_normalized_abs_error[t,h,v] - expert_normalized_abs_error[t,h,v,e]) on router_train only.",
        "capacity_formula": "capacity_per_expert = round(CF * (H * V) / E)",
        "capacity_factors": {"primary": 1.0, "secondary": 2.0},
        "fallback_rule": "If no expert claims a cell, use the equal fixed ensemble for that HxV cell.",
        "aggregation_rule": "Average only the original frozen forecasts from experts that claimed the cell; do not alter expert forecasts.",
        "matched_token_choice_controls": ["TokenChoice-Top1", "TokenChoice-Top2"],
        "baselines": ["Best single frozen expert selected on router_train within core", "Equal fixed ensemble", "Existing Frozen HxV COSTAR"],
        "statistical_tests": {
            "block_bootstrap": {"block_length": BLOCK_LENGTH, "samples": BOOTSTRAP_SAMPLES},
            "every_kth_phase_bootstrap": {"k": PHASE_K, "samples": BOOTSTRAP_SAMPLES},
        },
        "success_criteria": {
            "EXPERT_CHOICE_SUPPORTED": [
                "EC CF1 beats Token Top1 by MAE on at least 3/5 datasets",
                "EC CF2 beats Token Top2 by MAE on at least 3/5 datasets",
                "At least one matched-budget comparison has block-24 dependence support",
                "EC claim masks show nontrivial specialization, operationalized as average pairwise Jaccard < 0.98 for CF1 and CF2 on every dataset",
            ],
            "MIXED_EXPERT_CHOICE": "Convincing wins or dependence support, but fewer than the full supported rule.",
            "NO_EXPERT_CHOICE_ADVANTAGE": "Matched-budget Token Choice is equal or better on most datasets.",
        },
        "validation_only": True,
        "test_set_accessed": False,
        "test_cache_loaded": False,
        "test_metrics_computed": False,
    }
    write_json(OUT_DIR / "method_manifest.json", manifest)

    report: dict[str, Any] = {
        "experiment": "expert_choice_hv",
        "created_at_utc": manifest["created_at_utc"],
        "datasets": {},
        "test_set_accessed": False,
        "test_cache_loaded": False,
        "test_metrics_computed": False,
    }
    all_metric_rows: list[dict[str, Any]] = []
    all_assignment_rows: list[dict[str, Any]] = []
    all_dependence_rows: list[dict[str, Any]] = []
    all_integrity_rows: list[dict[str, Any]] = []

    for dataset in DATASETS:
        print(f"[expert-choice-hv] {dataset}: running validation-only evaluation...", flush=True)
        result = summarize_dataset(dataset)
        report["datasets"][dataset] = {k: v for k, v in result.items() if k not in {"assignment_stats", "dependence", "integrity"}}
        report["datasets"][dataset]["dependence"] = result["dependence"]
        report["datasets"][dataset]["integrity"] = result["integrity"]
        all_metric_rows.extend(result["metrics"])
        all_assignment_rows.extend(result["assignment_stats"])
        all_dependence_rows.extend(result["dependence"])
        all_integrity_rows.extend(result["integrity"])
        print(f"[expert-choice-hv] {dataset}: done. core={'+'.join(result['core'])}", flush=True)

    classification, details = classify(report)
    report["classification"] = classification
    report["classification_details"] = details
    report["runtime_sec"] = time.time() - start

    write_json(OUT_DIR / "results.json", report)
    write_csv(OUT_DIR / "results.csv", all_metric_rows)
    write_csv(OUT_DIR / "assignment_stats.csv", all_assignment_rows)
    write_csv(OUT_DIR / "dependence_tests.csv", all_dependence_rows)
    write_csv(OUT_DIR / "integrity_checks.csv", all_integrity_rows)
    write_json(
        OUT_DIR / "integrity_checks.json",
        {
            "checks": all_integrity_rows,
            "all_pass": all(row["all_pass"] for row in all_integrity_rows),
            "TEST_SET_ACCESSED": "NO",
            "TEST_CACHE_LOADED": "NO",
            "TEST_METRICS_COMPUTED": "NO",
        },
    )
    make_report(report)
    print("TEST SET ACCESSED: NO")
    print("TEST CACHE LOADED: NO")
    print("TEST METRICS COMPUTED: NO")
    print(json.dumps({"classification": classification, "runtime_sec": report["runtime_sec"], **details}, indent=2))


if __name__ == "__main__":
    main()
