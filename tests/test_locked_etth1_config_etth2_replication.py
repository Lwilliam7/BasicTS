import json
import unittest
from pathlib import Path

import torch

from experiments.locked_etth1_config_etth2_replication.run_locked_etth1_config_etth2_replication import (
    CORE,
    OUT_DIR,
    build_feature_tensor,
    causal_residual_stats,
    refuse_test_path,
    sha256_file,
)


class LockedEtth1ConfigEtth2ReplicationTests(unittest.TestCase):
    def test_refuse_test_path_blocks_test_cache_names(self) -> None:
        with self.assertRaises(ValueError):
            refuse_test_path("experiments/final_test_evaluation/generated/caches/ETTh2/locked_test_cache_v2.pt")

    def test_causal_residual_stats_waits_for_complete_window(self) -> None:
        starts = torch.tensor([100, 105, 111, 112], dtype=torch.long)
        residual = torch.ones(4, 12, 7)
        stats, extra = causal_residual_stats(starts, residual, horizon=12, init_residuals_norm=None)
        self.assertEqual(extra["num_residual_stat_updates"], 1)
        self.assertTrue(torch.allclose(stats[0, :, :, 0], torch.zeros(12, 7)))
        self.assertTrue(torch.allclose(stats[1, :, :, 0], torch.zeros(12, 7)))
        self.assertTrue(torch.allclose(stats[2, :, :, 0], torch.zeros(12, 7)))
        self.assertTrue(torch.allclose(stats[3, :, :, 0], torch.ones(12, 7)))

    def test_feature_tensor_shape_and_core_names(self) -> None:
        cache = {
            "num_windows": 2,
            "forecast_horizon": 12,
            "num_features": 7,
            "absolute_window_starts": torch.tensor([8640, 8652], dtype=torch.long),
            "expert_names": ["DLinear", "PatchTST", "iTransformer", "TimesNet", "ModernTCN"],
            "prediction_stack": torch.zeros(2, 12, 7, 5),
            "targets": torch.zeros(2, 12, 7),
            "target_masks": torch.ones(2, 12, 7, dtype=torch.bool),
        }
        series = torch.zeros(11520, 7)
        x, names, extra = build_feature_tensor(
            cache,
            cache["absolute_window_starts"],
            baseline=torch.zeros(2, 12, 7),
            std=torch.ones(7),
            series=series,
            init_residuals_norm=None,
            allow_test_history=False,
        )
        self.assertEqual(x.shape[0], 2 * 12 * 7)
        self.assertTrue(all(f"expert_{name}" in names for name in CORE))
        self.assertEqual(extra["num_residual_stat_updates"], 1)

    def test_manifest_artifact_hashes_match_generated_files(self) -> None:
        manifest_path = OUT_DIR / "manifest_before_test.json"
        if not manifest_path.exists():
            raise unittest.SkipTest("ETTh2 replication manifest has not been generated")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertFalse(manifest["test_loaded_before_manifest"])
        self.assertTrue(manifest["checks"]["router_train_only_fitting"])
        ridge = manifest["methods"]["ridge_residual_corrector"]
        self.assertEqual(sha256_file(Path(ridge["artifact_path"])), ridge["artifact_sha256"])
        for method in ("mlp_residual_corrector", "oracle_prototype_residual"):
            hashes = manifest["methods"][method]["artifact_sha256"]
            paths = manifest["methods"][method]["artifact_paths"]
            for path, digest in zip(paths, hashes.values()):
                self.assertEqual(sha256_file(Path(path)), digest)


if __name__ == "__main__":
    unittest.main()
