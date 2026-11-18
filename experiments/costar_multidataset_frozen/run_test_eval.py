"""First clean test evaluation for the frozen multi-dataset COSTAR router.

STEP 7 of the frozen protocol. This script REFUSES to run unless
`frozen_manifest.json` already exists and is marked `frozen: true` -- i.e.
router architecture, decay, temperature, expert pool, and expert-selection
rule must already be committed from `run_validation_eval.py` before this
script touches anything test-related.

It builds the held-out test cache (80-100%) once, using the SAME final_60
expert checkpoints already trained for router_val (no retraining), evaluates
the five frozen methods exactly once per dataset, and writes results. It
contains no logic that could feed test results back into architecture,
hyperparameter, expert-pool, or expert-selection decisions -- there is
nothing here to tune.
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
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_costarts_walkforward_cache import (  # noqa: E402
    build_stage_cache,
    chronological_ranges,
    load_full_array,
    stage_specs,
    valid_window_starts,
)
from scripts.train_costarts_walkforward_experts import predict_expert  # noqa: E402

from experiments.costar_multidataset_frozen.common import (  # noqa: E402
    EXPERT_ORDER,
    causality_perturbation_check,
    expert_indices,
    metric_values,
    predict_method,
)
from experiments.oracle_weight_tournament.run_tournament import load_cache, load_std  # noqa: E402
from experiments.costar_multidataset_frozen.run_validation_eval import (  # noqa: E402
    DATASETS,
    FORECAST_HORIZON,
    INPUT_LEN,
    METHOD_ORDER,
    OUT_DIR,
    per_window_rows,
)


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
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def require_frozen_manifest() -> dict[str, Any]:
    manifest_path = OUT_DIR / "frozen_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("frozen_manifest.json does not exist. Run run_validation_eval.py first -- test cannot be touched before the architecture is frozen.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("frozen"):
        raise RuntimeError("frozen_manifest.json is not marked frozen=true. Refusing to evaluate test.")
    current_sha = git_commit_sha()
    if manifest.get("git_commit_sha") != current_sha:
        raise RuntimeError(
            f"Git commit has changed since the manifest was frozen ({manifest.get('git_commit_sha')} -> {current_sha}). "
            "Refusing to evaluate test against a manifest that may no longer match the code."
        )
    return manifest


def build_test_cache(dataset: str, paths: Mapping[str, str], device: torch.device) -> Path:
    data_dir = ROOT / paths["data_dir"]
    cache_dir = ROOT / paths["cache_dir"]
    checkpoint_root = ROOT / paths["checkpoint_root"]
    prediction_dir = ROOT / f"results/router_summary/costarts_walkforward_{dataset}/expert_predictions"

    full_data = load_full_array(data_dir)
    ranges = chronological_ranges(full_data.shape[0])
    stages = stage_specs(cache_dir, ranges)
    test_stage = stages["test_80_100"]
    starts = valid_window_starts(test_stage.prediction_range, INPUT_LEN, FORECAST_HORIZON)

    rows = []
    for expert in EXPERT_ORDER:
        checkpoint_path = checkpoint_root / "final_60" / expert / "best_expert.pt"
        rows.append(
            predict_expert(
                checkpoint_path=checkpoint_path,
                full_data=full_data,
                starts=starts,
                input_len=INPUT_LEN,
                horizon=FORECAST_HORIZON,
                batch_size=256,
                device=device,
                output_path=prediction_dir / "test_80_100" / f"{expert}.npy",
            )
        )
    manifest_path = prediction_dir / "test_80_100" / "prediction_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "stage": "final_60",
                "predict_role": "test_80_100",
                "prediction_range": {"start": test_stage.prediction_range.start, "end": test_stage.prediction_range.end},
                "expert_order": list(EXPERT_ORDER),
                "rows": rows,
                "expert_checkpoint_paths": {r["expert"]: r["checkpoint_path"] for r in rows},
                "expert_checkpoint_hashes": {r["expert"]: r["checkpoint_sha256"] for r in rows},
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "git_commit": git_commit_sha(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    build_stage_cache(
        dataset=dataset,
        data_dir=data_dir,
        prediction_dir=prediction_dir,
        stage=test_stage,
        input_len=INPUT_LEN,
        horizon=FORECAST_HORIZON,
        error_temperature=0.1,
        checkpoint_manifest=manifest_path,
        allow_test=True,
    )
    return test_stage.output_path


def evaluate_test(dataset: str, paths: Mapping[str, str], manifest: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    ds_manifest = manifest["datasets"][dataset]
    checkpoint_root = ROOT / paths["checkpoint_root"]
    cache_dir = ROOT / paths["cache_dir"]

    test_cache_path = build_test_cache(dataset, paths, device)
    test_cache = torch.load(test_cache_path, map_location="cpu", weights_only=False)
    train_cache = torch.load(cache_dir / "router_train_20_60_cache.pt", map_location="cpu", weights_only=False)
    std = load_std(checkpoint_root / "final_60" / "DLinear" / "best_expert.pt", int(test_cache["num_features"]))

    frozen_core = ds_manifest["selected_core"]
    expert_idx = expert_indices(test_cache, frozen_core)
    frozen_best_name = ds_manifest["best_single_expert"]
    best_expert_col = expert_indices(test_cache, [frozen_best_name])[0]

    result_rows, per_window, causality_rows = [], [], []
    for method, label in METHOD_ORDER:
        pred, extra = predict_method(method, test_cache, train_cache, expert_idx, std, best_expert_col)
        m = metric_values(test_cache, pred, std)
        result_rows.append(
            {
                "dataset": dataset,
                "split": "test",
                "method": method,
                "label": label,
                "mae": m["mae"],
                "mse": m["mse"],
                "expert_set": "+".join(frozen_core),
                "first_clean_test_evaluation": True,
                **extra,
            }
        )
        per_window.extend(per_window_rows(dataset, method, "test", test_cache, std, pred))
        check = causality_perturbation_check(method, test_cache, train_cache, expert_idx, std, best_expert_col)
        causality_rows.append({"dataset": dataset, "split": "test", **check})

    return {
        "dataset": dataset,
        "result_rows": result_rows,
        "per_window": per_window,
        "causality_rows": causality_rows,
        "test_cache_sha256": sha256_file(test_cache_path),
        "test_cache_path": str(test_cache_path.relative_to(ROOT)),
        "num_windows_test": int(test_cache["num_windows"]),
    }


def main() -> None:
    manifest = require_frozen_manifest()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start = time.time()

    print("=" * 70)
    print("FIRST CLEAN TEST EVALUATION -- architecture frozen, running test ONCE per dataset")
    print("=" * 70)

    report: dict[str, Any] = {
        "experiment": "costar_multidataset_frozen_test",
        "label": "first_clean_test_evaluation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_sha": manifest["git_commit_sha"],
        "frozen_manifest_used": str((OUT_DIR / "frozen_manifest.json").relative_to(ROOT)),
        "datasets": {},
    }
    all_results: list[dict[str, Any]] = []
    all_per_window: list[dict[str, Any]] = []
    all_causality: list[dict[str, Any]] = []

    for dataset, paths in DATASETS.items():
        print(f"[test-eval] {dataset}: building test cache and evaluating frozen methods (first and only time)...", flush=True)
        result = evaluate_test(dataset, paths, manifest, device)
        report["datasets"][dataset] = {k: v for k, v in result.items() if k != "per_window"}
        all_results.extend(result["result_rows"])
        all_per_window.extend(result["per_window"])
        all_causality.extend(result["causality_rows"])
        print(f"[test-eval] {dataset}: done.", flush=True)

    report["runtime_sec"] = time.time() - start
    write_json(OUT_DIR / "test_results.json", report)
    write_csv(OUT_DIR / "test_results.csv", all_results)
    write_csv(OUT_DIR / "test_per_window_metrics.csv", all_per_window)
    write_csv(OUT_DIR / "test_causality_checks.csv", all_causality)

    print(json.dumps({"runtime_sec": report["runtime_sec"], "datasets": list(report["datasets"].keys())}, indent=2))


if __name__ == "__main__":
    main()
