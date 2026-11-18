"""Evaluate a saved forecast-adaptive COSTAR-TS router on router validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.old.train_costarts_adaptive_router import (
    COSTARTSAdaptiveRouter,
    DEFAULT_RESULTS_DIR,
    DEFAULT_VAL_CACHE,
    _jsonable,
    _load_torch,
    rollout_adaptive_router,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a forecast-adaptive COSTAR-TS router.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val-cache", default=DEFAULT_VAL_CACHE)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-queries", type=int, default=5)
    parser.add_argument("--forced-budget", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = _load_torch(args.checkpoint)
    val_cache = _load_torch(args.val_cache)
    device = torch.device(args.device)
    router = COSTARTSAdaptiveRouter(**checkpoint["router_config"]).to(device)
    router.load_state_dict(checkpoint["router_state_dict"])
    metrics = rollout_adaptive_router(
        router,
        val_cache,
        batch_size=args.batch_size,
        device=device,
        max_queries=args.max_queries,
        forced_budget=args.forced_budget,
    )
    output_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"forced{args.forced_budget}" if args.forced_budget is not None else "adaptive"
    output_path = output_dir / f"evaluation_{Path(args.checkpoint).stem}_{suffix}.json"
    payload = {
        "checkpoint": args.checkpoint,
        "val_cache": args.val_cache,
        "metrics": metrics,
        "test_set_used": False,
    }
    output_path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    print(json.dumps(_jsonable(payload), indent=2))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
