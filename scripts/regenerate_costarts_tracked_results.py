"""Regenerate tracked COSTARTS evidence with row-level provenance.

This command deletes stale tracked result artifacts, reruns the current evaluators,
adds provenance to every reported row, rebuilds the paper package, and validates
that stale reranker/sparse-mixing artifacts are not packaged as current evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT / "results/router_summary/costarts_subset_utility"
DEFAULT_PAPER_DIR = DEFAULT_RESULTS_DIR / "paper_package"

REQUIRED_PROVENANCE_FIELDS = (
    "checkpoint_hash",
    "finalizer",
    "seed",
    "inference_rule",
    "cache_hash",
)

CORE_RESULT_FILES = (
    "final_comparison.csv",
    "final_comparison.json",
    "cost_sweep.csv",
    "pareto_curve.json",
    "ablations.csv",
    "ablations.json",
)

LEGACY_FINALIZER_ARTIFACTS = (
    "reranking_comparison.csv",
    "reranking_examples.csv",
    "mixing_results.csv",
    "mix_weight_statistics.json",
)


def sha256_file(path: Path, *, missing_value: str = "missing") -> str:
    """Return a full SHA-256 hex digest without loading the whole file in memory."""
    if not path.exists() or not path.is_file():
        return missing_value
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def combined_file_hash(paths: Mapping[str, Path]) -> tuple[str, dict[str, str]]:
    """Hash an ordered manifest of named input-file hashes."""
    hashes = {name: sha256_file(path) for name, path in sorted(paths.items())}
    payload = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), hashes


def _run(script: str, arguments: Sequence[str]) -> None:
    command = [sys.executable, str(ROOT / script), *map(str, arguments)]
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def _delete_stale(results_dir: Path, paper_dir: Path) -> None:
    for name in (*CORE_RESULT_FILES, *LEGACY_FINALIZER_ARTIFACTS):
        path = results_dir / name
        if path.exists():
            path.unlink()
            print(f"Deleted stale artifact: {path}")
    if paper_dir.exists():
        shutil.rmtree(paper_dir)
        print(f"Deleted stale paper package: {paper_dir}")


def _checkpoint_hash(path: str | Path | None) -> str:
    if path in (None, "", "not_applicable"):
        return "not_applicable"
    return sha256_file(Path(path))


def _final_comparison_provenance(
    method: str,
    *,
    seed: int,
    cache_hash: str,
    checkpoints: Mapping[str, Path],
) -> dict[str, Any]:
    checkpoint_key = ""
    finalizer = "single_expert"
    inference_rule = "fixed_or_cached_expert"

    if method == "routerdc_hard_without_contrastive":
        checkpoint_key = "routerdc_no_contrastive"
        inference_rule = "history_embedding_cosine_argmax"
    elif method == "routerdc_hard_with_contrastive":
        checkpoint_key = "routerdc_contrastive"
        inference_rule = "history_embedding_cosine_argmax"
    elif method == "old_costarts":
        checkpoint_key = "old_costarts"
        finalizer = "predicted_error_selected_expert"
        inference_rule = "one_shot_query_order_and_stop"
    elif method == "improved_subset_utility_costarts":
        checkpoint_key = "subset"
        finalizer = "equal_average"
        inference_rule = "action_logits_argmax_with_stop"
    elif method == "predicted_top2_equal_average":
        checkpoint_key = "subset"
        finalizer = "equal_average"
        inference_rule = "force_two_action_logits_queries"
    elif method == "oracle_within_predicted_top2":
        checkpoint_key = "subset"
        finalizer = "oracle_best_queried"
        inference_rule = "force_two_action_logits_queries_then_oracle_select"
    elif method == "oracle_second_query_after_router_first":
        checkpoint_key = "subset"
        finalizer = "oracle_best_queried"
        inference_rule = "router_first_query_then_oracle_second_query"
    elif method in {"equal_average_all_experts", "fixed_top2_equal_average"}:
        finalizer = "equal_average"
        inference_rule = "fixed_expert_set"
    elif method == "train_weighted_average":
        finalizer = "train_weighted_average"
        inference_rule = "router_train_inverse_mae_weights"
    elif method == "linear_stacker":
        finalizer = "linear_stacker"
        inference_rule = "router_train_linear_least_squares"
    elif method == "full_oracle":
        finalizer = "oracle_single_expert"
        inference_rule = "validation_oracle"
    elif method == "best_fixed_expert":
        inference_rule = "router_train_best_fixed_expert"
    elif method.startswith("individual_expert:"):
        inference_rule = "fixed_individual_expert"

    checkpoint_path = checkpoints.get(checkpoint_key) if checkpoint_key else None
    return {
        "checkpoint_hash": _checkpoint_hash(checkpoint_path),
        "finalizer": finalizer,
        "seed": int(seed),
        "inference_rule": inference_rule,
        "cache_hash": cache_hash,
    }


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader.fieldnames or ()), list(reader)


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _append_fields(fieldnames: Sequence[str]) -> list[str]:
    output = list(fieldnames)
    for field in REQUIRED_PROVENANCE_FIELDS:
        if field not in output:
            output.append(field)
    return output


def annotate_final_comparison(
    results_dir: Path,
    *,
    seed: int,
    cache_hash: str,
    cache_hashes: Mapping[str, str],
    checkpoints: Mapping[str, Path],
) -> None:
    csv_path = results_dir / "final_comparison.csv"
    json_path = results_dir / "final_comparison.json"
    fields, rows = _read_csv(csv_path)
    by_method: dict[str, dict[str, Any]] = {}
    for row in rows:
        provenance = _final_comparison_provenance(
            row["method"], seed=seed, cache_hash=cache_hash, checkpoints=checkpoints
        )
        row.update(provenance)
        by_method[row["method"]] = provenance
    _write_csv(csv_path, _append_fields(fields), rows)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    for row in payload.get("rows", []):
        row.update(by_method[row["method"]])
    payload.setdefault("metadata", {}).update(
        {
            "seed": int(seed),
            "cache_hash": cache_hash,
            "cache_hashes": dict(cache_hashes),
            "provenance_schema": list(REQUIRED_PROVENANCE_FIELDS),
        }
    )
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def annotate_cost_sweep(
    results_dir: Path,
    *,
    seed: int,
    checkpoint: Path,
    cache_hash: str,
    cache_hashes: Mapping[str, str],
    finalizer: str,
) -> None:
    csv_path = results_dir / "cost_sweep.csv"
    pareto_path = results_dir / "pareto_curve.json"
    provenance = {
        "checkpoint_hash": _checkpoint_hash(checkpoint),
        "finalizer": finalizer,
        "seed": int(seed),
        "inference_rule": "predicted_utility_argmax_stop_when_cost_adjusted_utility_nonpositive",
        "cache_hash": cache_hash,
    }
    fields, rows = _read_csv(csv_path)
    for row in rows:
        row.update(provenance)
    _write_csv(csv_path, _append_fields(fields), rows)

    payload = json.loads(pareto_path.read_text(encoding="utf-8"))
    for point in payload.get("points", []):
        point.update(provenance)
    payload.setdefault("metadata", {}).update(
        {
            **provenance,
            "cache_hashes": dict(cache_hashes),
            "provenance_schema": list(REQUIRED_PROVENANCE_FIELDS),
        }
    )
    pareto_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _ablation_inference_rule(row: Mapping[str, Any]) -> str:
    if row.get("ablation") == "no_subset_state_supervision":
        return "one_shot_query_order_and_stop"
    try:
        cost_lambda = float(row.get("cost_lambda") or 0.0)
    except (TypeError, ValueError):
        cost_lambda = 0.0
    if cost_lambda > 0:
        return "predicted_utility_argmax_stop_when_cost_adjusted_utility_nonpositive"
    if str(row.get("status", "")).startswith("skipped"):
        return "not_run"
    return "action_logits_argmax_with_stop"


def annotate_ablations(
    results_dir: Path,
    *,
    seed: int,
    cache_hash: str,
    cache_hashes: Mapping[str, str],
) -> None:
    csv_path = results_dir / "ablations.csv"
    json_path = results_dir / "ablations.json"
    fields, rows = _read_csv(csv_path)
    by_ablation: dict[str, dict[str, Any]] = {}
    for row in rows:
        provenance = {
            "checkpoint_hash": _checkpoint_hash(row.get("checkpoint")),
            "finalizer": row.get("finalizer") or "not_applicable",
            "seed": int(seed),
            "inference_rule": _ablation_inference_rule(row),
            "cache_hash": cache_hash,
        }
        row.update(provenance)
        by_ablation[row["ablation"]] = provenance
    _write_csv(csv_path, _append_fields(fields), rows)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    for row in payload.get("rows", []):
        row.update(by_ablation[row["ablation"]])
    payload.setdefault("metadata", {}).update(
        {
            "cache_hash": cache_hash,
            "cache_hashes": dict(cache_hashes),
            "provenance_schema": list(REQUIRED_PROVENANCE_FIELDS),
        }
    )
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def validate_csv_provenance(path: Path) -> None:
    fields, rows = _read_csv(path)
    missing_columns = [field for field in REQUIRED_PROVENANCE_FIELDS if field not in fields]
    if missing_columns:
        raise RuntimeError(f"{path} is missing provenance columns: {missing_columns}")
    for index, row in enumerate(rows, start=2):
        missing_values = [field for field in REQUIRED_PROVENANCE_FIELDS if not str(row.get(field, "")).strip()]
        if missing_values:
            raise RuntimeError(f"{path}:{index} is missing provenance values: {missing_values}")


def _write_provenance_tables(results_dir: Path, paper_dir: Path) -> None:
    tables_dir = paper_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    for name in ("final_comparison.csv", "cost_sweep.csv", "ablations.csv"):
        source = results_dir / name
        destination = tables_dir / name
        shutil.copy2(source, destination)

    manifest = {
        name: sha256_file(tables_dir / name)
        for name in ("final_comparison.csv", "cost_sweep.csv", "ablations.csv")
    }
    (tables_dir / "provenance_manifest.json").write_text(
        json.dumps(
            {
                "required_fields": list(REQUIRED_PROVENANCE_FIELDS),
                "table_hashes": manifest,
                "legacy_reranker_or_sparse_mixing_tables_included": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    for name in LEGACY_FINALIZER_ARTIFACTS:
        stale = tables_dir / name
        if stale.exists():
            stale.unlink()
    stale_tex = tables_dir / "reranking_comparison.tex"
    if stale_tex.exists():
        stale_tex.unlink()


def validate_paper_package(results_dir: Path, paper_dir: Path) -> None:
    tables_dir = paper_dir / "tables"
    for name in ("final_comparison.csv", "cost_sweep.csv", "ablations.csv"):
        source = results_dir / name
        packaged = tables_dir / name
        validate_csv_provenance(packaged)
        if sha256_file(source) != sha256_file(packaged):
            raise RuntimeError(f"Paper-package table does not match current source: {name}")
    for name in LEGACY_FINALIZER_ARTIFACTS:
        if (tables_dir / name).exists():
            raise RuntimeError(f"Stale finalizer artifact was packaged: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate tracked COSTARTS results with row-level provenance.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--paper-dir", type=Path, default=DEFAULT_PAPER_DIR)
    parser.add_argument("--train-cache", type=Path, default=ROOT / "cache/costarts_router_train_cache.pt")
    parser.add_argument("--val-cache", type=Path, default=ROOT / "cache/costarts_router_val_cache.pt")
    parser.add_argument("--subset-train-cache", type=Path, default=ROOT / "cache/costarts_subset_states_train.pt")
    parser.add_argument("--subset-val-cache", type=Path, default=ROOT / "cache/costarts_subset_states_val.pt")
    parser.add_argument("--old-checkpoint", type=Path, default=ROOT / "checkpoints/costarts/best_costarts_router.pt")
    parser.add_argument("--subset-checkpoint", type=Path, default=ROOT / "checkpoints/costarts_subset_utility/best_subset_utility_costarts_router.pt")
    parser.add_argument("--routerdc-no-contrastive", type=Path, default=ROOT / "checkpoints/best_routerdc_hard_no_contrastive.pt")
    parser.add_argument("--routerdc-contrastive", type=Path, default=ROOT / "checkpoints/best_routerdc_hard_contrastive.pt")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--cost-lambda", type=float, default=0.2)
    parser.add_argument("--reuse-existing-ablations", action="store_true")
    parser.add_argument("--skip-execution", action="store_true", help="Only annotate and validate already-regenerated outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    paper_dir = args.paper_dir.resolve()

    if not args.skip_execution:
        _delete_stale(results_dir, paper_dir)
        _run(
            "scripts/evaluate_costarts_final_comparison.py",
            [
                "--train-cache", args.train_cache,
                "--val-cache", args.val_cache,
                "--subset-val-cache", args.subset_val_cache,
                "--old-costarts-checkpoint", args.old_checkpoint,
                "--subset-checkpoint", args.subset_checkpoint,
                "--routerdc-no-contrastive-checkpoint", args.routerdc_no_contrastive,
                "--routerdc-contrastive-checkpoint", args.routerdc_contrastive,
                "--output-dir", results_dir,
                "--batch-size", args.batch_size,
                "--device", args.device,
                "--seed", args.seed,
            ],
        )
        _run(
            "scripts/evaluate_costarts_cost_sweep.py",
            [
                "--cache", args.subset_val_cache,
                "--checkpoint", args.subset_checkpoint,
                "--output-dir", results_dir,
                "--finalizer", "equal_average",
                "--batch-size", args.batch_size,
                "--device", args.device,
                "--seed", args.seed,
            ],
        )
        ablation_args: list[Any] = [
            "--base-train-cache", args.train_cache,
            "--base-val-cache", args.val_cache,
            "--subset-train-cache", args.subset_train_cache,
            "--subset-val-cache", args.subset_val_cache,
            "--old-costarts-checkpoint", args.old_checkpoint,
            "--existing-full-checkpoint", args.subset_checkpoint,
            "--output-dir", results_dir,
            "--batch-size", args.batch_size,
            "--max-epochs", args.max_epochs,
            "--patience", args.patience,
            "--cost-lambda", args.cost_lambda,
            "--device", args.device,
            "--seed", args.seed,
        ]
        if args.reuse_existing_ablations:
            ablation_args.append("--reuse-existing")
        _run("scripts/run_costarts_subset_utility_ablations.py", ablation_args)

    final_cache_hash, final_cache_hashes = combined_file_hash(
        {
            "router_train": args.train_cache,
            "router_val": args.val_cache,
            "subset_router_val": args.subset_val_cache,
        }
    )
    subset_cache_hash, subset_cache_hashes = combined_file_hash({"subset_router_val": args.subset_val_cache})
    ablation_cache_hash, ablation_cache_hashes = combined_file_hash(
        {
            "router_val": args.val_cache,
            "subset_router_train": args.subset_train_cache,
            "subset_router_val": args.subset_val_cache,
        }
    )

    annotate_final_comparison(
        results_dir,
        seed=args.seed,
        cache_hash=final_cache_hash,
        cache_hashes=final_cache_hashes,
        checkpoints={
            "old_costarts": args.old_checkpoint,
            "subset": args.subset_checkpoint,
            "routerdc_no_contrastive": args.routerdc_no_contrastive,
            "routerdc_contrastive": args.routerdc_contrastive,
        },
    )
    annotate_cost_sweep(
        results_dir,
        seed=args.seed,
        checkpoint=args.subset_checkpoint,
        cache_hash=subset_cache_hash,
        cache_hashes=subset_cache_hashes,
        finalizer="equal_average",
    )
    annotate_ablations(
        results_dir,
        seed=args.seed,
        cache_hash=ablation_cache_hash,
        cache_hashes=ablation_cache_hashes,
    )

    for name in ("final_comparison.csv", "cost_sweep.csv", "ablations.csv"):
        validate_csv_provenance(results_dir / name)

    if not args.skip_execution:
        _run(
            "scripts/build_costarts_paper_package.py",
            [
                "--results-dir", results_dir,
                "--output-dir", paper_dir,
                "--val-cache", args.val_cache,
                "--subset-val-cache", args.subset_val_cache,
                "--old-checkpoint", args.old_checkpoint,
                "--improved-checkpoint", args.subset_checkpoint,
                "--batch-size", args.batch_size,
                "--device", args.device,
                "--seed", args.seed,
            ],
        )
    _write_provenance_tables(results_dir, paper_dir)
    validate_paper_package(results_dir, paper_dir)

    print("Regenerated and validated COSTARTS tracked results.")
    print(f"Results: {results_dir}")
    print(f"Paper package: {paper_dir}")


if __name__ == "__main__":
    main()
