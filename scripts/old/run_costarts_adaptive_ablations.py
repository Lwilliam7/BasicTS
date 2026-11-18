"""Run lightweight forecast-adaptive COSTAR-TS ablations."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.old.train_costarts_adaptive_router import (
    AdaptiveTrainingConfig,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RESULTS_DIR,
    _jsonable,
    train_one_seed,
)


def _parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run adaptive COSTAR-TS ablations.")
    parser.add_argument("--seeds", default="7,11,13,17,19")
    parser.add_argument("--variants", default="mask_only,forecast,forecast_disagreement")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--max-train-states", type=int, default=None)
    parser.add_argument("--train-state-mode", choices=("all", "deployable_oracle_mixture"), default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for variant in _parse_strings(args.variants):
        for seed in _parse_ints(args.seeds):
            result = train_one_seed(
                AdaptiveTrainingConfig(
                    seed=seed,
                    variant=variant,
                    output_dir=str(Path(args.output_dir) / variant),
                    results_dir=str(Path(args.results_dir) / variant),
                    device=args.device,
                    batch_size=args.batch_size,
                    max_epochs=args.max_epochs,
                    patience=args.patience,
                    max_train_states=args.max_train_states,
                    train_state_mode=args.train_state_mode,
                )
            )
            best = result["best"]["val_metrics"]
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "mae": best["mae"],
                    "mse": best["mse"],
                    "average_experts_queried": best["average_experts_queried"],
                    "regret_to_oracle": best["regret_to_oracle"],
                }
            )

    summary = []
    for variant in sorted({row["variant"] for row in rows}):
        variant_rows = [row for row in rows if row["variant"] == variant]
        maes = torch.tensor([float(row["mae"]) for row in variant_rows])
        experts = torch.tensor([float(row["average_experts_queried"]) for row in variant_rows])
        summary.append(
            {
                "variant": variant,
                "mae_mean": float(maes.mean()),
                "mae_std": float(maes.std(unbiased=False)),
                "avg_experts_mean": float(experts.mean()),
                "avg_experts_std": float(experts.std(unbiased=False)),
            }
        )
    output_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "adaptive_ablation_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("variant", "seed", "mae", "mse", "average_experts_queried", "regret_to_oracle"))
        writer.writeheader()
        writer.writerows(rows)
    json_path = output_dir / "adaptive_ablation_summary.json"
    payload = {"rows": rows, "summary": summary, "test_set_used": False}
    json_path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
