"""Evaluate a Full End-to-End Sequential COSTAR-TS checkpoint on validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.sequential_costarts_end_to_end import EndToEndCOSTARTSConfig, FullEndToEndCOSTARTS
from scripts.train_sequential_costarts_end_to_end import ETThWindowDataset, evaluate, fit_scaler, load_full_data, load_train_val_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="datasets/ETTh1")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--stop-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-test-final", action="store_true")
    parser.add_argument("--output", default="results/router_summary/costarts_end_to_end/evaluation_summary.json")
    args = parser.parse_args()

    if args.split == "test" and not args.allow_test_final:
        raise ValueError("Refusing test evaluation without --allow-test-final")
    full = load_full_data(ROOT / args.data_dir) if args.split == "test" else load_train_val_data(ROOT / args.data_dir)
    train_end = 8640
    val_end = 11520
    start, end = (train_end, val_end) if args.split == "validation" else (val_end, 14400)
    mean, std = fit_scaler(full, train_end)
    checkpoint = torch.load(ROOT / args.checkpoint, map_location="cpu", weights_only=False)
    config = EndToEndCOSTARTSConfig(**checkpoint["model_config"])
    model = FullEndToEndCOSTARTS(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(torch.device(args.device))
    dataset = ETThWindowDataset(full, start, end, config.input_len, config.forecast_horizon, mean, std)
    metrics = evaluate(model, DataLoader(dataset, batch_size=args.batch_size, shuffle=False), torch.device(args.device), args.stop_threshold)
    summary = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "range": {"start": start, "end": end},
        "metrics": metrics,
        "safety": "TEST FINAL MODE" if args.allow_test_final else "NO TEST DATA USED",
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
