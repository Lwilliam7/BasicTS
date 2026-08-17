"""Evaluate full Sequential COSTAR-TS checkpoints on walk-forward caches."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_costarts_walkforward_cache import EXPERT_ORDER, sha256_file, validate_walkforward_cache
from scripts.sequential_costarts_model_full import SequentialCOSTARTSRouterFull
from scripts.train_sequential_costarts_full_walkforward import evaluate_router, individual_expert_metrics


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def load_cache(path: Path, allow_test: bool) -> dict[str, Any]:
    if "test" in str(path).lower() and not allow_test:
        raise ValueError(f"Refusing test cache without --allow-test-final: {path}")
    cache = torch.load(path, map_location="cpu", weights_only=False)
    validate_walkforward_cache(cache, allow_test=allow_test)
    if "test" in str(cache["cache_role"]).lower() and not allow_test:
        raise ValueError("Refusing test cache role without --allow-test-final")
    return cache


def load_checkpoint(path: Path) -> tuple[SequentialCOSTARTSRouterFull, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = SequentialCOSTARTSRouterFull(**checkpoint["router_config"])
    model.load_state_dict(checkpoint["router_state_dict"], strict=True)
    return model, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache", default="cache/costarts_walkforward/router_val_60_80_cache.pt")
    parser.add_argument("--output-dir", default="results/router_summary/costarts_walkforward/sequential_full/evaluation")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--query-threshold", type=float)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-test-final", action="store_true")
    args = parser.parse_args()

    checkpoint_path = ROOT / args.checkpoint
    cache_path = ROOT / args.cache
    model, checkpoint = load_checkpoint(checkpoint_path)
    cache = load_cache(cache_path, allow_test=args.allow_test_final)
    if tuple(cache["expert_names"]) != EXPERT_ORDER:
        raise ValueError("Expert order mismatch")
    threshold = float(args.query_threshold) if args.query_threshold is not None else float(checkpoint.get("query_threshold", 0.0))
    device = torch.device(args.device)
    model.to(device)
    metrics = evaluate_router(
        model,
        cache,
        device,
        max_queries=int(checkpoint["router_config"]["max_subset_size"]),
        query_threshold=threshold,
        batch_size=args.batch_size,
    )
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "per_window.csv", metrics["per_window"], ["cache_index", "absolute_window_start", "query_count", "queried_experts", "mae", "mse"])
    write_csv(output_dir / "individual_expert_metrics.csv", individual_expert_metrics(cache), ["expert", "mae", "mse"])
    summary = {
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "cache": args.cache,
        "cache_sha256": sha256_file(cache_path),
        "cache_role": cache["cache_role"],
        "mae": metrics["mae"],
        "mse": metrics["mse"],
        "average_queries": metrics["average_queries"],
        "top1_expert_accuracy": metrics["top1_expert_accuracy"],
        "stopping_percent": metrics["stopping_percent"],
        "expert_usage_percent": metrics["expert_usage_percent"],
        "individual_expert_metrics": individual_expert_metrics(cache),
        "leakage_checks": "passed",
        "safety": "TEST FINAL MODE" if args.allow_test_final else "NO TEST DATA USED",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
