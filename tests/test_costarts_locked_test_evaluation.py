import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import costarts_locked_test_evaluation as locked


def _manifest(tmp_path: Path) -> dict:
    ckpt = tmp_path / "pair.pt"
    ckpt.write_bytes(b"locked checkpoint")
    expert = tmp_path / "expert.pt"
    expert.write_bytes(b"expert checkpoint")
    cache = tmp_path / "router_cache.pt"
    cache.write_bytes(b"router cache")
    return {
        "splits": {
            "expert_train": {"start": 0, "end": 5},
            "expert_val": {"start": 5, "end": 7},
            "router_train": {"start": 7, "end": 10},
            "router_val": {"start": 10, "end": 12},
            "test": {"start": 12, "end": 18},
        },
        "preprocessing": {
            "scaler": {
                "fit_split": "expert_train",
                "mean": [2.0],
                "std": [1.4142135],
            }
        },
        "metric_definition": {
            "input_len": 2,
            "forecast_horizon": 1,
            "num_features": 1,
            "batch_size": 2,
        },
        "locked_routing": {
            "expert_names": ["a", "b", "c"],
            "fixed_pair_selection_split": "router_val",
            "seeds": [
                {
                    "seed": 7,
                    "pair_selector_checkpoint": str(ckpt.relative_to(tmp_path)),
                    "pair_selector_checkpoint_sha256": locked.sha256_file(ckpt),
                    "checkpoint_selection_split": "router_val",
                    "threshold_selection_split": "router_val",
                    "confidence_score_selection_split": "router_val",
                    "fixed_pair_class": 0,
                    "fixed_pair_names": ["a", "b"],
                    "confidence_score": "probability_margin",
                    "threshold": 0.5,
                }
            ],
        },
        "artifacts": {
            "expert_checkpoints": [
                {
                    "expert_name": "a",
                    "path": str(expert.relative_to(tmp_path)),
                    "sha256": locked.sha256_file(expert),
                }
            ],
            "pair_selector_checkpoints": [
                {
                    "seed": 7,
                    "path": str(ckpt.relative_to(tmp_path)),
                    "sha256": locked.sha256_file(ckpt),
                }
            ],
            "router_caches": [
                {
                    "split_role": "router_train",
                    "path": str(cache.relative_to(tmp_path)),
                    "sha256": locked.sha256_file(cache),
                }
            ],
        },
    }


def test_split_isolation_rejects_overlap(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["splits"]["router_val"]["start"] = 9

    with pytest.raises(AssertionError, match="starts before|overlaps"):
        locked.assert_chronological_split_isolation(manifest)


def test_scaler_leakage_check_uses_pre_router_fit_range(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    full = np.arange(18, dtype=np.float32).reshape(18, 1)
    np.save(data_dir / "train_data.npy", full[:10])
    np.save(data_dir / "val_data.npy", full[10:12])
    np.save(data_dir / "test_data.npy", full[12:])
    manifest = _manifest(tmp_path)

    locked.assert_scaler_fit_is_pre_router(manifest, data_dir)
    manifest["preprocessing"]["scaler"]["fit_end"] = 13
    manifest["splits"]["expert_train"]["end"] = 13
    with pytest.raises(AssertionError, match="router-training"):
        locked.assert_scaler_fit_is_pre_router(manifest, data_dir)


def test_cache_alignment_requires_test_absolute_indices(tmp_path):
    manifest = _manifest(tmp_path)
    cache = {
        "split_role": "test",
        "num_windows": 3,
        "expert_names": ("a", "b", "c"),
        "sample_indices": torch.arange(3),
        "absolute_start_indices": torch.tensor([12, 13, 14]),
    }

    locked.assert_test_cache_alignment(cache, manifest)
    cache["absolute_start_indices"] = torch.tensor([11, 12, 13])
    with pytest.raises(AssertionError, match="align"):
        locked.assert_test_cache_alignment(cache, manifest)


def test_checkpoint_hash_validation_rejects_tampering(tmp_path):
    manifest = _manifest(tmp_path)
    locked.assert_manifest_hashes(manifest, tmp_path)

    (tmp_path / "pair.pt").write_bytes(b"changed")
    with pytest.raises(AssertionError, match="Hash mismatch"):
        locked.assert_manifest_hashes(manifest, tmp_path)


def test_locked_cache_builder_aligns_predictions_and_metadata(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    full = np.arange(18, dtype=np.float32).reshape(18, 1)
    np.save(data_dir / "train_data.npy", full[:10])
    np.save(data_dir / "val_data.npy", full[10:12])
    np.save(data_dir / "test_data.npy", full[12:])
    manifest = _manifest(tmp_path)
    manifest_path = tmp_path / "lock.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    num_windows = 4
    for expert_index, expert_name in enumerate(("a", "b", "c")):
        prediction = np.full((num_windows, 1, 1), float(expert_index), dtype=np.float32)
        np.save(tmp_path / f"{expert_name}.npy", prediction)

    cache = locked.build_locked_test_cache(
        manifest_path=manifest_path,
        data_dir=data_dir,
        expert_prediction_specs=[
            f"a={tmp_path / 'a.npy'}",
            f"b={tmp_path / 'b.npy'}",
            f"c={tmp_path / 'c.npy'}",
        ],
        output_path=tmp_path / "cache.pt",
        repo_root=tmp_path,
    )

    assert cache["split_role"] == "test"
    assert tuple(cache["prediction_stack"].shape) == (4, 1, 1, 3)
    assert cache["absolute_start_indices"].tolist() == [12, 13, 14, 15]


def test_deterministic_routing_score_from_locked_logits():
    logits = torch.tensor([[2.0, 1.0, 0.0], [0.0, 3.0, 1.0]])
    score_a = locked._score_for_locked_name(logits, fixed_pair_class=0, score_name="logit_margin")
    score_b = locked._score_for_locked_name(logits, fixed_pair_class=0, score_name="logit_margin")

    assert torch.equal(score_a, score_b)
    assert torch.allclose(score_a, torch.tensor([1.0, 2.0]))


def test_test_data_cannot_select_threshold_or_checkpoint(tmp_path):
    manifest = _manifest(tmp_path)
    locked.assert_no_test_selection(manifest)

    manifest["locked_routing"]["seeds"][0]["threshold_selection_split"] = "test"
    with pytest.raises(AssertionError, match="Threshold"):
        locked.assert_no_test_selection(manifest)
