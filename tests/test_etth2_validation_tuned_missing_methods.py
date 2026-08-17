import json
import unittest
from pathlib import Path

from experiments.etth2_validation_tuned_missing_methods.run_etth2_validation_tuned_missing_methods import (
    LABEL,
    OUT_DIR,
    declared_search_spaces,
    sha256_file,
)


class Etth2ValidationTunedMissingMethodsTests(unittest.TestCase):
    def test_declared_search_spaces_are_small_and_include_locked_configs(self) -> None:
        spaces = declared_search_spaces()
        self.assertEqual(set(spaces), {"ridge_residual_corrector", "mlp_residual_corrector", "oracle_prototype_residual", "dynamic_fixed_three"})
        self.assertTrue(all(len(configs) <= 5 for configs in spaces.values()))
        self.assertIn({"ridge": 1.0, "alpha": 0.1, "clip_multiple": 0.25, "feature_set": "full"}, spaces["ridge_residual_corrector"])
        self.assertIn({"teacher_lambda": 0.01, "num_prototypes": 16, "residual_scale": 0.3, "residual_weight": 0.001, "epochs": 10}, spaces["oracle_prototype_residual"])

    def test_tuned_manifest_freezes_before_test_and_hashes_artifacts(self) -> None:
        manifest_path = OUT_DIR / "tuned_manifest_before_test.json"
        if not manifest_path.exists():
            raise unittest.SkipTest("Tuned manifest has not been generated")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["label"], LABEL)
        self.assertFalse(manifest["test_loaded_before_tuned_freeze"])
        self.assertFalse(manifest["test_metrics_used_for_decision"])
        self.assertEqual(len(manifest["selected_winners"]), 4)
        for payload in manifest["selected_winners"].values():
            for path in payload["frozen_artifact_paths"]:
                digest = payload["frozen_artifact_sha256"][Path(path).name]
                self.assertEqual(sha256_file(Path(path)), digest)

    def test_final_results_are_labeled_validation_tuned(self) -> None:
        report_path = OUT_DIR / "final_report.json"
        if not report_path.exists():
            raise unittest.SkipTest("Tuned final report has not been generated")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertTrue(report["test_evaluation_complete"])
        self.assertTrue(all(row["status"] == LABEL for row in report["test_results"]))


if __name__ == "__main__":
    unittest.main()
