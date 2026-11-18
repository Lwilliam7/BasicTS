"""Independently check saved evidence from the October 26 baseline audit."""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREVIOUS = HERE.parent / "baseline_health_20261026_recheck"
EARLIER_CPU = HERE.parent / "baseline_health_20261025_cpu"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture_ast(path):
    return ast.dump(next(node for node in ast.parse(path.read_text(encoding="utf-8")).body
                         if isinstance(node, ast.FunctionDef) and node.name == "synthetic_pool"))


def main():
    result = read(HERE / "results.json")
    manifest = read(HERE / "manifest.json")
    previous_manifest = read(PREVIOUS / "manifest.json")
    assert result["status"] == "SOFTWARE_HEALTH_PASS"
    assert not result["table_eligible"]
    assert manifest["environment"] == previous_manifest["environment"]
    for path, expected in manifest["source_hashes"].items():
        assert digest(Path(path)) == expected, path
    changed_sources = [path for path, expected in manifest["source_hashes"].items()
                       if previous_manifest["source_hashes"].get(path) != expected]
    assert {Path(path).name for path in changed_sources} == {"reference_fixture.py", "PROTOCOL.md"}
    assert fixture_ast(PREVIOUS / "source_snapshot/reference_fixture.py") == fixture_ast(
        ROOT / "scripts/baseline_health_20261025/reference_fixture.py")

    checked_cases = {}
    hash_count = 0
    for method, method_result in result["methods"].items():
        assert method_result["software_health"] == "PASS"
        assert not method_result["table_eligible"]
        for case, case_result in method_result["cases"].items():
            dest = HERE / method / case
            assert case_result["software_case_passed"]
            for relative, expected in case_result["artifact_hashes"].items():
                assert digest(dest / relative) == expected, str(dest / relative)
                hash_count += 1
            losses = np.load(dest / "per_window_losses.npz")["losses"]
            assert losses.shape == (128, 2) and np.isfinite(losses).all()
            assert float(losses[:, 0].mean()) == case_result["mae"]
            for earlier in (PREVIOUS, EARLIER_CPU):
                old = np.load(earlier / method / case / "per_window_losses.npz")["losses"]
                assert np.array_equal(losses, old), (method, case, str(earlier))

            name = "seed7_ETTh1.pt" if method in ("fame", "onenet") else "ETTh1.pt"
            exported = torch.load(dest / "_work" / name, map_location="cpu", weights_only=False)
            if method != "onenet":
                key = {"fame": "top2", "timerouter": "timerouter", "superlinear": "A_seed7"}[method]
                exported = exported[key]
            expected = np.stack([exported[f"per_window_{metric}"].numpy() for metric in ("mae", "mse")], axis=1)
            assert np.array_equal(losses, expected), (method, case, "wrong metric export")
            if method == "onenet":
                trace = read(dest / "causality_trace.json")["trace"]
                assert len(trace) == 127
                assert all(row["update_origin"] + 12 <= row["prediction_origin"] for row in trace)
            checked_cases[f"{method}/{case}"] = dict(
                artifact_hashes_valid=True, exported_metrics_match=True,
                identical_losses_to_both_prior_cpu_runs=True)
    assert len(checked_cases) == 14

    unpoisoned = np.load(HERE / "onenet/branch1/one_net_trace.npz")["standardized_predictions"]
    poisoned = np.load(HERE / "onenet/poisoned/one_net_trace.npz")["standardized_predictions"]
    assert np.array_equal(unpoisoned[:65], poisoned[:65])
    suffix_difference = float(np.max(np.abs(unpoisoned[65:] - poisoned[65:])))
    assert suffix_difference > 0, "Poison must affect later predictions once labels resolve"

    old_fame = read(PREVIOUS / "fame/shuffled/result.json")
    new_fame = read(HERE / "fame/shuffled/result.json")
    verified = dict(status="PASS", source_hashes_verified=len(manifest["source_hashes"]),
                    artifact_hashes_verified=hash_count, cases=checked_cases,
                    fixture_generator_ast_unchanged=True, environment_unchanged=True,
                    changed_hashed_sources=changed_sources,
                    poison_prefix_identical_through_row_64=True,
                    poison_observable_suffix_max_difference=suffix_difference,
                    fame_shuffled_accuracy_correction=dict(
                        old_secondary_top1=old_fame["diagnostics"]["winner_accuracy"],
                        corrected_primary_top2=new_fame["diagnostics"]["winner_accuracy"]),
                    paper_fidelity="NOT_ESTABLISHED", table_eligible=False)
    (HERE / "artifact_verification.json").write_text(json.dumps(verified, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in verified.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
