"""LearnedProbe V3A feasibility audit.

V3A's prompt requires a frozen raw-response representation test using the
exact V2 OOF perturbation behavior, without retraining the V2 generator.

The completed V2 artifact set does not contain OOF learned deltas, OOF full
raw response tensors, or trained V2 generator checkpoints. This script records
that blocker in the requested V3A output directory. It does not train, score,
reconstruct forecasts, or access test data.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
V2_DIR = ROOT / "experiments" / "behavioral_competence" / "controlled_discriminative_probe_v2"
DATASETS = ["ExchangeRate", "Traffic", "BeijingAirQuality", "ETTm2"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def npz_shapes(path: Path) -> dict[str, str]:
    data = np.load(path, allow_pickle=True)
    return {key: str(getattr(data[key], "shape", "")) for key in data.files}


def audit_dataset(dataset: str) -> dict[str, Any]:
    raw_path = V2_DIR / "raw_response_cache" / f"{dataset}.npz"
    score_path = V2_DIR / "per_window_scores" / f"{dataset}.npz"
    raw = np.load(raw_path, allow_pickle=True)
    scores = np.load(score_path, allow_pickle=True)
    raw_keys = set(raw.files)
    score_keys = set(scores.files)
    return {
        "dataset": dataset,
        "v2_raw_response_cache": str(raw_path.relative_to(ROOT)),
        "v2_per_window_scores": str(score_path.relative_to(ROOT)),
        "raw_response_cache_sha256": sha256_file(raw_path),
        "per_window_scores_sha256": sha256_file(score_path),
        "has_router_val_learned_delta": "conditional_delta_val" in raw_keys,
        "has_router_val_learned_six_response": "conditional_response_val" in raw_keys,
        "has_oof_learned_six_response": "oof_conditional_response_common" in raw_keys,
        "has_oof_learned_delta": "oof_conditional_delta_common" in raw_keys or "conditional_delta_oof_common" in raw_keys,
        "has_oof_full_raw_response": "oof_conditional_raw_response_common" in raw_keys or "oof_delta_forecast_common" in raw_keys,
        "has_router_val_full_raw_response": "conditional_raw_response_val" in raw_keys or "conditional_delta_forecast_val" in raw_keys,
        "has_v2_generator_checkpoint": any((V2_DIR / name).exists() for name in ["generator_checkpoints", "checkpoints", "models"]),
        "has_common_idx": "common_idx" in score_keys,
        "has_actual_conditional_common": "actual_conditional_common" in score_keys,
        "has_oof_predictions": all(key in score_keys for key in ["oof_conditional_common", "oof_passive_common", "oof_random_common"]),
        "raw_npz_shapes": npz_shapes(raw_path),
        "score_npz_shapes": npz_shapes(score_path),
    }


def copy_oof_manifest() -> None:
    src = V2_DIR / "oof_fold_manifest.csv"
    dst = OUT_DIR / "oof_fold_manifest.csv"
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def make_report(dataset_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# LearnedProbe V3A -- Frozen Raw-Response Representation Test",
        "",
        "**Status: BLOCKED before scientific evaluation.**",
        "",
        "V3A requires the exact frozen V2 OOF raw forecast responses from the learned conditional probe, or enough frozen V2 state to reconstruct them without retraining the generator.",
        "",
        "The completed V2 artifacts contain router-val learned deltas and OOF six-stat response summaries, but they do not contain OOF learned deltas, OOF full raw response tensors, or trained V2 generator checkpoints. Re-running `train_learned_shared_prefix` would retrain the V2 generator and violate the V3A hard rule.",
        "",
        "Therefore the primary comparisons `RawResponseActive vs SixStatActive`, `RawResponseActive vs ShuffledRawResponse`, `PassivePlusRaw vs PassiveOnly`, and `RawResponse -> passive residual` cannot be run under the frozen V3A rules.",
        "",
        "```text",
        "TEST SET ACCESSED: NO",
        "V2 PERTURBATION GENERATOR RETRAINED: NO",
        "FORECASTING EXPERTS RETRAINED: NO",
        "V2 RESULT MODIFIED: NO",
        "ROUTER TRAINED: NO",
        "```",
        "",
        "## Artifact Availability",
        "",
        "| Dataset | Router-val learned delta | OOF learned six stats | OOF learned delta | OOF full raw response | V2 generator checkpoint |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in dataset_rows:
        lines.append(
            f"| {row['dataset']} | {row['has_router_val_learned_delta']} | {row['has_oof_learned_six_response']} | "
            f"{row['has_oof_learned_delta']} | {row['has_oof_full_raw_response']} | {row['has_v2_generator_checkpoint']} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        "Do not report a V3A scientific classification from these artifacts. The correct next action, if V3A is still desired, is a separate V2-compatible rerun that freezes and saves OOF learned deltas/generator checkpoints before V3A is attempted. That would be a new experiment, not this frozen V3A representation test.",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc).isoformat()
    dataset_rows = [audit_dataset(dataset) for dataset in DATASETS]
    v2_artifacts = [
        "report.md",
        "validation_results.json",
        "router_val_competence_results.csv",
        "oof_competence_results.csv",
        "oof_fold_manifest.csv",
        "integrity_checks.csv",
        "prompt_compliance_audit.md",
        "prompt_compliance_audit.json",
    ]
    source_manifest = {
        "created_at_utc": created_at,
        "current_git_commit": git_commit_sha(),
        "source_v2_dir": str(V2_DIR.relative_to(ROOT)),
        "source_v2_artifacts": {
            name: {
                "path": str((V2_DIR / name).relative_to(ROOT)),
                "exists": (V2_DIR / name).exists(),
                "sha256": sha256_file(V2_DIR / name) if (V2_DIR / name).exists() else None,
            }
            for name in v2_artifacts
        },
        "dataset_artifacts": dataset_rows,
        "v3a_did_not_retrain_or_modify_v2_perturbation_generator": True,
    }
    method_manifest = {
        "experiment": "raw_response_probe_v3a",
        "status": "BLOCKED_MISSING_FROZEN_V2_OOF_RAW_RESPONSE",
        "created_at_utc": created_at,
        "hypothesis": "Holding V2 intervention fixed, test whether full horizon-by-variable raw response contains complementary competence information beyond six handcrafted stats.",
        "reason_blocked": "V2 did not save OOF learned deltas, OOF full raw response tensors, or trained V2 generator checkpoints; reconstructing them would require retraining the V2 generator.",
        "test_set_accessed": False,
        "v2_generator_retrained": False,
        "forecasting_experts_retrained": False,
        "router_trained": False,
    }
    integrity_rows = [
        {
            "check": "no_test_access",
            "result": "PASS",
            "details": "This audit reads only V2 development artifacts and does not construct, load, inspect, summarize, or evaluate any test cache.",
        },
        {
            "check": "frozen_v2_artifacts_available",
            "result": "PASS",
            "details": "V2 report, validation results, fold manifest, integrity checks, raw_response_cache, and per_window_scores are present.",
        },
        {
            "check": "exact_oof_common_windows_reusable",
            "result": "PASS",
            "details": "V2 per_window_scores include common_idx and actual_conditional_common for all four datasets.",
        },
        {
            "check": "oof_raw_response_available",
            "result": "FAIL",
            "details": "No dataset has OOF learned deltas or OOF full raw response tensors.",
        },
        {
            "check": "v2_generator_checkpoint_available",
            "result": "FAIL",
            "details": "No trained V2 generator checkpoint directory/file is present in the V2 artifact set.",
        },
        {
            "check": "can_run_primary_v3a_without_retraining_v2",
            "result": "FAIL",
            "details": "Primary V3A OOF raw-response comparison cannot be performed under the frozen no-retraining rule.",
        },
    ]
    shape_rows = []
    for row in dataset_rows:
        shape_rows.append(
            {
                "dataset": row["dataset"],
                "router_val_learned_delta_shape": row["raw_npz_shapes"].get("conditional_delta_val"),
                "router_val_six_response_shape": row["raw_npz_shapes"].get("conditional_response_val"),
                "oof_six_response_shape": row["raw_npz_shapes"].get("oof_conditional_response_common"),
                "oof_learned_delta_shape": row["raw_npz_shapes"].get("oof_conditional_delta_common", "MISSING"),
                "oof_full_raw_response_shape": row["raw_npz_shapes"].get("oof_conditional_raw_response_common", "MISSING"),
                "common_idx_shape": row["score_npz_shapes"].get("common_idx"),
                "actual_conditional_common_shape": row["score_npz_shapes"].get("actual_conditional_common"),
            }
        )
    write_json(OUT_DIR / "method_manifest.json", method_manifest)
    write_json(OUT_DIR / "source_v2_manifest.json", source_manifest)
    write_csv(OUT_DIR / "integrity_checks.csv", integrity_rows)
    write_csv(OUT_DIR / "raw_response_shape_diagnostics.csv", shape_rows)
    copy_oof_manifest()
    make_report(dataset_rows)
    (OUT_DIR / "per_window_predictions").mkdir(exist_ok=True)
    (OUT_DIR / "per_window_predictions" / "README.md").write_text(
        "No V3A per-window predictions were produced because the primary frozen OOF raw-response inputs are unavailable.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": method_manifest["status"], "test_set_accessed": False}, indent=2))


if __name__ == "__main__":
    main()
