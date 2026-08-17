from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_costarts_walkforward_cache import (
    EXPERT_ORDER,
    RangeSpec,
    assert_stage_no_leakage,
    build_stage_cache,
    chronological_ranges,
    combine_router_train,
    stage_specs,
    validate_walkforward_cache,
)


def _write_dataset(root: Path, n: int = 100, f: int = 2) -> Path:
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    full = np.arange(n * f, dtype=np.float32).reshape(n, f) / 100.0
    np.save(data_dir / "train_data.npy", full[:60])
    np.save(data_dir / "val_data.npy", full[60:80])
    np.save(data_dir / "test_data.npy", full[80:])
    return data_dir


def _write_predictions(root: Path, role: str, n: int, horizon: int, features: int) -> Path:
    pred_dir = root / "preds"
    role_dir = pred_dir / role
    role_dir.mkdir(parents=True, exist_ok=True)
    for offset, expert in enumerate(EXPERT_ORDER):
        values = np.full((n, horizon, features), 0.01 * offset, dtype=np.float32)
        np.save(role_dir / f"{expert}.npy", values)
    return pred_dir


def test_walkforward_ranges_are_chronological() -> None:
    ranges = chronological_ranges(100)
    assert ranges["block_a"].start == 0
    assert ranges["block_b"].start == 20
    assert ranges["block_c"].start == 40
    assert ranges["validation"].start == 60
    assert ranges["test"].start == 80


def test_leakage_assertion_rejects_prediction_before_training_end() -> None:
    starts = torch.arange(35, 40)
    try:
        assert_stage_no_leakage(
            role="bad",
            expert_training_range=RangeSpec("train", 0.0, 0.4, 0, 40),
            prediction_range=RangeSpec("pred", 0.3, 0.5, 35, 50),
            starts=starts,
            input_len=2,
            horizon=1,
            num_timestamps=100,
            allow_test=False,
        )
    except AssertionError:
        return
    raise AssertionError("Expected leakage assertion to fail")


def test_build_and_combine_synthetic_oos_caches() -> None:
    root = ROOT / "results" / "tmp_tests" / "walkforward_protocol"
    root.mkdir(parents=True, exist_ok=True)
    data_dir = _write_dataset(root)
    ranges = chronological_ranges(100)
    stages = stage_specs(root / "cache", ranges)
    for role in ("block_b_oos", "block_c_oos"):
        starts = torch.arange(
            stages[role].prediction_range.start,
            stages[role].prediction_range.end - 2 - 1 + 1,
            dtype=torch.long,
        )
        pred_dir = _write_predictions(root, role, int(starts.numel()), 1, 2)
        cache = build_stage_cache(
            dataset="SYN",
            data_dir=data_dir,
            prediction_dir=pred_dir,
            stage=stages[role],
            input_len=2,
            horizon=1,
            error_temperature=0.1,
            checkpoint_manifest=None,
            allow_test=False,
        )
        validate_walkforward_cache(cache)
    combined = combine_router_train(
        stages["block_b_oos"].output_path,
        stages["block_c_oos"].output_path,
        root / "cache" / "router_train_20_60_cache.pt",
    )
    validate_walkforward_cache(combined)
    assert combined["cache_role"] == "router_train_20_60"
    assert int(combined["num_windows"]) == 36


if __name__ == "__main__":
    test_walkforward_ranges_are_chronological()
    test_leakage_assertion_rejects_prediction_before_training_end()
    test_build_and_combine_synthetic_oos_caches()
    print("walkforward protocol tests passed")
