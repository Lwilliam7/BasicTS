"""Local orchestration and actual-port component checks; no server access or research data."""
import ast
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
MAIN = Path(r"C:\Data\Documents\Code\BasicTS")
sys.path.insert(0, str(ROOT / "scripts/research_suite_20261020"))
sys.path.insert(0, str(MAIN))
import run_suite as suite
import run_router_checks as router


class SuiteChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / "research_review")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def instance(self):
        with patch.object(suite, "identity", return_value={"source": "fixed"}), patch.object(suite, "versions", return_value={}):
            return suite.Suite(self.root, self.root / "output", 5)

    def test_kernel_lock_rejects_duplicate(self):
        with suite.lock(self.root / "run.lock"):
            with self.assertRaises(OSError):
                with suite.lock(self.root / "run.lock"):
                    self.fail("Duplicate acquired lock")

    def test_failure_recorded_and_independent_lane_finishes(self):
        obj = self.instance()
        def cpu():
            raise RuntimeError("Planted lane failure")
        def gpu():
            obj.mark("traffic/recovery", "COMPLETE")
        with patch.object(obj, "cpu_lane", side_effect=cpu), patch.object(obj, "gpu_lane", side_effect=gpu), patch.object(suite, "pack"):
            self.assertEqual(obj.execute(), 2)
        self.assertEqual(obj.state["stages"]["lane/cpu"]["status"], "FAILED")
        self.assertEqual(obj.state["stages"]["traffic/recovery"]["status"], "COMPLETE")
        self.assertEqual(obj.state["status"], "PARTIAL")

    def test_failed_worker_cannot_be_complete(self):
        obj = self.instance()
        with patch.object(obj, "verify_sources"), patch.object(suite.subprocess, "run", return_value=type("R", (), {"returncode": 3})()):
            self.assertFalse(obj.run("broken", ["fake.py"]))
        self.assertEqual(obj.state["stages"]["broken"]["status"], "FAILED")

    def test_idle_gpu_guard_and_lock(self):
        obj = self.instance()
        with patch.object(suite, "protocol_processes", return_value=[]), patch.object(suite, "gpu_rows", return_value=[(0, 40000, 99), (1, 4, 0)]):
            with obj.idle_gpu("test/gpu") as chosen:
                self.assertEqual(chosen, 1)

    def test_resume_rejects_changed_source(self):
        self.instance()
        with patch.object(suite, "identity", return_value={"source": "changed"}), patch.object(suite, "versions", return_value={}):
            with self.assertRaisesRegex(RuntimeError, "changed"):
                suite.Suite(self.root, self.root / "output")

    def test_analysis_exit_zero_does_not_hide_failed_seed(self):
        obj = self.instance()
        folder = self.root / "experiments/long_horizon_protocol"
        suite.write(folder / "queue_state.json", {"finished": True, "analysis_rc": 0,
                    "jobs": {str(i): {"status": "failed" if i == 0 else "done"} for i in range(148)}})
        with self.assertRaisesRegex(RuntimeError, "jobs incomplete"):
            obj.recovery_done()
        state = suite.read(folder / "queue_state.json")
        state["jobs"]["0"]["status"] = "done"
        suite.write(folder / "queue_state.json", state)
        suite.write(folder / "integrity_checks.json", {str(i): {"ok": i != 0} for i in range(21)})
        with self.assertRaisesRegex(RuntimeError, "integrity"):
            obj.recovery_done()
        suite.write(folder / "integrity_checks.json", {str(i): {"ok": True} for i in range(21)})
        obj.recovery_done()

    def test_granularity_science_functions_unchanged(self):
        source = ast.parse((MAIN / "experiments/gate_ablation/run_gate_ablation.py").read_text())
        copied = ast.parse((ROOT / suite.GRAN / "run_granularity.py").read_text())
        original = {n.name: ast.dump(n, include_attributes=False) for n in source.body if isinstance(n, ast.FunctionDef)}
        derived = {n.name: ast.dump(n, include_attributes=False) for n in copied.body if isinstance(n, ast.FunctionDef)}
        changed = {"source_hashes", "manifest", "analyze", "main", "queue"}
        for name in set(original) - changed:
            self.assertEqual(original[name], derived[name], name)
        text = (ROOT / suite.GRAN / "run_granularity.py").read_text()
        self.assertIn("EXPLORATORY POST-RESULT", text)
        self.assertNotIn("Only HELPS licenses", text)

    def test_synthetic_pool_and_real_timerouter_alignment(self):
        import numpy as np
        import torch
        import experiments.timerouter_official_replication.run_timerouter_official_replication as tr
        b, winners = router.synthetic_pool()
        shuffled, _ = router.synthetic_pool(True)
        self.assertTrue(torch.equal(b.train_cache["prediction_stack"], shuffled.train_cache["prediction_stack"]))
        self.assertTrue(torch.equal(b.val_cache["prediction_stack"], shuffled.val_cache["prediction_stack"]))
        self.assertFalse(torch.equal(b.val_cache["targets"], shuffled.val_cache["targets"]))
        for c in (b.train_cache, b.val_cache):
            check = tr.check_alignment(c, b.std, tr.start_lookup_of(c), 12, np.random.default_rng(7))
            self.assertTrue(check["pass"], check)
            self.assertTrue(torch.isfinite(c["prediction_stack"]).all())
        self.assertGreaterEqual(np.mean(winners == 0), .25)
        self.assertLessEqual(np.mean(winners == 0), .75)
        self.assertLessEqual(int(b.train_cache["absolute_window_starts"][-1]) + 12,
                             int(b.val_cache["absolute_window_starts"][0]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
