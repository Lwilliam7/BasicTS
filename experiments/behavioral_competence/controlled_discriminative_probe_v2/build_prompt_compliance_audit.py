"""Build post-run prompt-compliance audit artifacts.

This script does not train or score any model. It reads the frozen outputs
from controlled_discriminative_probe_v2 and writes supplemental diagnostics
that the original prompt requested explicitly: RMS perturbation magnitude,
expert-order permutation metric equivariance, and a literal compliance
checklist.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import experiments.frozen_hv_costar.run_frozen_hv_costar as fhv  # noqa: E402
from experiments.behavioral_competence.generalization.run_generalization_study import register_dataset  # noqa: E402
from run_controlled_discriminative_probe_v2 import competence_table_row  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent
DATASETS = ["ExchangeRate", "Traffic", "BeijingAirQuality", "ETTm2"]
PERMUTATION_EQUIVARIANCE_TOLERANCE = 1e-3
METHOD_KEYS = {
    "SharedRandomProbe": ("random_delta_val", "random_val_pred"),
    "SharedConditionalLearnedProbe": ("conditional_delta_val", "conditional_val_pred"),
}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def normalized_delta_stats(delta: np.ndarray, hist_std: np.ndarray) -> dict[str, float]:
    norm = np.abs(delta) / np.maximum(hist_std[:, None, :], 1e-8)
    return {
        "mean_normalized_abs_delta": float(norm.mean()),
        "rms_normalized_delta": float(math.sqrt(float((norm * norm).mean()))),
        "max_normalized_abs_delta": float(norm.max()),
    }


def build_rms_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        register_dataset(dataset)
        bundle = fhv.LOADERS[dataset]()
        histories = bundle.val_cache["histories"].to(torch.float32)
        hist_std = histories.std(dim=1).numpy()
        raw = np.load(OUT_DIR / "raw_response_cache" / f"{dataset}.npz")
        for method, (delta_key, _) in METHOD_KEYS.items():
            delta = np.asarray(raw[delta_key], dtype=np.float32)
            rows.append({"dataset": dataset, "method": method, "split": "router_val", **normalized_delta_stats(delta, hist_std)})
            del delta
        del histories, hist_std
    return rows


def metric_diff(a: dict[str, Any], b: dict[str, Any], key: str) -> float:
    av, bv = float(a[key]), float(b[key])
    if math.isnan(av) and math.isnan(bv):
        return 0.0
    return abs(av - bv)


def build_permutation_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    perm = np.array([1, 2, 0], dtype=np.int64)
    metric_keys = [
        "conditional_mae",
        "conditional_r2",
        "pearson",
        "spearman",
        "pairwise_ranking_accuracy",
        "top1_conditional_best_accuracy",
    ]
    pred_keys = {
        "SharedRandomProbe": "random_val_pred",
        "SharedConditionalLearnedProbe": "conditional_val_pred",
        "MatchedPassive": "passive_val_pred",
        "ShuffledConditionalProbe": "shuffled_val_pred",
    }
    for dataset in DATASETS:
        pw = np.load(OUT_DIR / "per_window_scores" / f"{dataset}.npz")
        actual = torch.from_numpy(np.asarray(pw["actual_conditional_val"], dtype=np.float32))
        for method, pred_key in pred_keys.items():
            pred = torch.from_numpy(np.asarray(pw[pred_key], dtype=np.float32))
            original = competence_table_row(dataset, method, "router_val", pred, actual)
            permuted = competence_table_row(dataset, method, "router_val_permuted", pred[:, perm], actual[:, perm])
            max_metric_abs_diff = max(metric_diff(original, permuted, key) for key in metric_keys)
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "permutation": "1,2,0",
                    "tolerance": PERMUTATION_EQUIVARIANCE_TOLERANCE,
                    "max_metric_abs_diff": max_metric_abs_diff,
                    "permutation_equivariant": bool(max_metric_abs_diff < PERMUTATION_EQUIVARIANCE_TOLERANCE),
                }
            )
    return rows


def main() -> None:
    rms_rows = build_rms_rows()
    permutation_rows = build_permutation_rows()
    write_csv(OUT_DIR / "perturbation_rms_diagnostics.csv", rms_rows)
    write_csv(OUT_DIR / "expert_order_permutation_checks.csv", permutation_rows)

    audit = {
        "status": "post_run_prompt_compliance_audit",
        "no_training_or_scoring_performed": True,
        "supplemental_files": [
            "perturbation_rms_diagnostics.csv",
            "expert_order_permutation_checks.csv",
            "prompt_compliance_audit.json",
            "prompt_compliance_audit.md",
        ],
        "fully_satisfied": [
            "New method directory; LearnedProbe v1 untouched.",
            "Four specified development datasets.",
            "No test cache access.",
            "Shared window-only perturbation architecture.",
            "Active-only six-feature primary scorer.",
            "MatchedPassive separate control.",
            "Conditional competence target with purged OOF folds.",
            "Required controls and router_val report tables.",
            "Dependence-aware block/bootstrap statistics.",
            "Checkpoint/frozen-expert/target-corruption integrity checks.",
        ],
        "satisfied_structurally_or_by_supplement": [
            "Same-question invariant is structural: one x_probe tensor per batch is reused for every expert. The run records max_abs_diff=0 and this audit documents that per-expert delta copies were not separately persisted.",
            "Expert-order permutation metric equivariance is now checked in expert_order_permutation_checks.csv from saved predictions/targets.",
            "RMS normalized perturbation magnitude is now reported in perturbation_rms_diagnostics.csv.",
        ],
        "partial_or_not_available_without_rerun": [
            "Section 39 requested a wide per-window/per-expert mechanism table including absolute origins and SharedTotalLearnedProbe raw response. Existing caches contain core names, actual errors, conditional targets, all primary predictions, random response, conditional response, and common indices, but not SharedTotalLearnedProbe raw response or explicit absolute origins in the npz. Reconstructing total raw response would require rerunning a learned model that was not serialized.",
            "Section 32 requested saving delta per expert path. The implementation applies the same tensor object by construction and verifies it is not mutated by expert forward calls, but the cache stores one shared delta per window rather than duplicated per-expert delta copies.",
        ],
        "result_unchanged": {
            "tier": "ACTIVE_SIGNAL_BUT_REDUNDANT",
            "proceed_to_router_integration": False,
        },
    }
    (OUT_DIR / "prompt_compliance_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Prompt Compliance Audit",
        "",
        "This is a post-run audit over frozen outputs. It does not train, score, tune, or rerun any model.",
        "",
        "## Supplemental Diagnostics",
        "",
        "- `perturbation_rms_diagnostics.csv`: adds RMS normalized delta requested by Section 38.",
        "- `expert_order_permutation_checks.csv`: verifies metric equivariance under a fixed expert-axis permutation.",
        "",
        "## Result",
        "",
        "- Tier remains `ACTIVE_SIGNAL_BUT_REDUNDANT`.",
        "- `proceed_to_router_integration` remains `false`.",
        "",
        "## Literal Gaps",
        "",
        "- The saved raw-response cache does not include `SharedTotalLearnedProbe` raw response tensors.",
        "- The per-window cache does not explicitly store absolute forecast origins; it stores common indices and predictions/targets.",
        "- The shared delta is stored once per window, not duplicated per expert path. This matches the implemented mechanism but is not the literal Section 32 storage format.",
    ]
    (OUT_DIR / "prompt_compliance_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
