import csv
import importlib.util
import json
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/regenerate_costarts_tracked_results.py"
spec = importlib.util.spec_from_file_location("regenerate_costarts_tracked_results", MODULE_PATH)
regen = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(regen)


def test_sha256_and_combined_hash_are_deterministic(tmp_path):
    first = tmp_path / "a.bin"
    second = tmp_path / "b.bin"
    first.write_bytes(b"alpha")
    second.write_bytes(b"beta")

    combined_a, hashes_a = regen.combined_file_hash({"b": second, "a": first})
    combined_b, hashes_b = regen.combined_file_hash({"a": first, "b": second})

    assert combined_a == combined_b
    assert hashes_a == hashes_b
    assert len(regen.sha256_file(first)) == 64


def test_final_comparison_annotation_adds_required_provenance(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    checkpoint = tmp_path / "subset.pt"
    checkpoint.write_bytes(b"checkpoint")
    rows = [{"method": "improved_subset_utility_costarts", "status": "ok"}]

    with (results / "final_comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["method", "status"])
        writer.writeheader()
        writer.writerows(rows)
    (results / "final_comparison.json").write_text(
        json.dumps({"metadata": {}, "rows": rows}), encoding="utf-8"
    )

    regen.annotate_final_comparison(
        results,
        seed=7,
        cache_hash="c" * 64,
        cache_hashes={"router_val": "d" * 64},
        checkpoints={"subset": checkpoint},
    )

    regen.validate_csv_provenance(results / "final_comparison.csv")
    _, annotated = regen._read_csv(results / "final_comparison.csv")
    assert annotated[0]["finalizer"] == "equal_average"
    assert annotated[0]["inference_rule"] == "action_logits_argmax_with_stop"
    assert annotated[0]["checkpoint_hash"] == regen.sha256_file(checkpoint)


def test_validation_rejects_missing_provenance(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("method,status\nmodel,ok\n", encoding="utf-8")

    try:
        regen.validate_csv_provenance(path)
    except RuntimeError as exc:
        assert "missing provenance columns" in str(exc)
    else:
        raise AssertionError("missing provenance should fail")


def test_paper_package_removes_legacy_finalizer_artifacts(tmp_path):
    results = tmp_path / "results"
    paper = tmp_path / "paper"
    results.mkdir()
    fields = ["name", *regen.REQUIRED_PROVENANCE_FIELDS]
    row = {
        "name": "x",
        "checkpoint_hash": "not_applicable",
        "finalizer": "equal_average",
        "seed": "7",
        "inference_rule": "fixed_expert_set",
        "cache_hash": "a" * 64,
    }
    for filename in ("final_comparison.csv", "cost_sweep.csv", "ablations.csv"):
        with (results / filename).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerow(row)

    legacy = paper / "tables" / "mixing_results.csv"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("stale", encoding="utf-8")

    regen._write_provenance_tables(results, paper)
    regen.validate_paper_package(results, paper)
    assert not legacy.exists()
