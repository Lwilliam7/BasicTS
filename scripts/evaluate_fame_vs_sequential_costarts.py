"""Validation-only FAME-style ETTh adaptation vs Sequential COSTAR-TS."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.fame_etth_router import (
    FameTrainingConfig,
    FingerprintScaler,
    evaluate_best_fixed_expert,
    evaluate_oracle_best_single,
    evaluate_sparse_router,
    evaluate_weighted_average,
    extract_fame_etth_fingerprint,
    load_router_cache,
    parameter_count,
    soft_expert_targets,
    train_fame_router,
    validate_cache_pair,
)
from scripts.sequential_costarts_model_full import load_sequential_costarts_router_full


METHOD_LABEL = "FAME-style ETTh adaptation"
NO_TEST_DATA_USED = "NO TEST DATA USED"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_sequential_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize(values: list[float]) -> tuple[float, float]:
    return mean(values), pstdev(values) if len(values) > 1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", default="cache/costarts_router_train_cache.pt")
    parser.add_argument("--val-cache", default="cache/costarts_router_val_cache.pt")
    parser.add_argument("--sequential-results", default="results/router_summary/costarts_sequential/per_seed_results.csv")
    parser.add_argument("--sequential-checkpoint", default="checkpoints/costarts_sequential/seed_11/best_sequential_costarts_router.pt")
    parser.add_argument("--output-dir", default="results/router_summary/fame_vs_sequential_costarts")
    parser.add_argument("--seeds", default="7,11,13,17,19")
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--delta", type=float, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_cache = load_router_cache(args.train_cache, "router_train")
    val_cache = load_router_cache(args.val_cache, "router_val")
    validate_cache_pair(train_cache, val_cache)

    train_fp = extract_fame_etth_fingerprint(train_cache["histories"])
    val_fp = extract_fame_etth_fingerprint(val_cache["histories"])
    if not torch.isfinite(train_fp).all() or not torch.isfinite(val_fp).all():
        raise ValueError("Fingerprint extraction produced nonfinite values")
    scaler = FingerprintScaler.fit(train_fp)
    train_fp = scaler.transform(train_fp)
    val_fp = scaler.transform(val_fp)
    if not torch.allclose(soft_expert_targets(train_cache["error_matrix"], args.tau).sum(dim=1), torch.ones(train_cache["error_matrix"].shape[0])):
        raise ValueError("Soft targets do not sum to 1")

    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    fame_rows: list[dict[str, Any]] = []
    usage_rows: list[dict[str, Any]] = []
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        config = FameTrainingConfig(
            seed=seed,
            tau=args.tau,
            hidden_size=args.hidden_size,
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            epochs=args.epochs,
        )
        model, train_metrics = train_fame_router(train_cache, train_fp, config)
        model.eval()
        with torch.no_grad():
            probabilities = model.probabilities(val_fp)
        checkpoint_path = checkpoint_dir / f"fame_etth_router_seed_{seed}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "training_config": asdict(config),
                "fingerprint_mean": scaler.mean,
                "fingerprint_std": scaler.std,
                "method": METHOD_LABEL,
                "no_test_data_used": True,
                "train_metrics": train_metrics,
                "expert_names": tuple(train_cache["expert_names"]),
            },
            checkpoint_path,
        )
        for top_r in (1, 2, 3):
            metrics = evaluate_sparse_router(probabilities, val_cache, top_r=top_r, delta=args.delta)
            row = {
                "method": METHOD_LABEL,
                "seed": seed,
                "top_r": top_r,
                "delta": args.delta,
                "mae": metrics["mae"],
                "mse": metrics["mse"],
                "average_experts_used": metrics["average_experts_used"],
                "router_param_count": parameter_count(model),
                "router_top1_accuracy": metrics["top1_accuracy"],
                "top_r_oracle_coverage": metrics["top_r_oracle_coverage"],
                "mean_routing_entropy": metrics["routing_entropy"],
                "train_kl": train_metrics["train_kl"],
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
            }
            fame_rows.append(row)
            for expert_name, usage in metrics["expert_usage"].items():
                usage_rows.append({"seed": seed, "top_r": top_r, "expert": expert_name, "usage_percent": usage})

    sequential_rows = read_sequential_rows(Path(args.sequential_results))
    sequential_router = load_sequential_costarts_router_full(args.sequential_checkpoint)
    sequential_param_count = parameter_count(sequential_router)

    equal_average = evaluate_weighted_average(val_cache)
    best_fixed = evaluate_best_fixed_expert(val_cache)
    oracle_single = evaluate_oracle_best_single(val_cache)
    oracle_mae = oracle_single["mae"]

    comparison_rows: list[dict[str, Any]] = []
    for top_r in (1, 2, 3):
        top_rows = [row for row in fame_rows if row["top_r"] == top_r]
        mae_mean, mae_std = summarize([float(row["mae"]) for row in top_rows])
        mse_mean, mse_std = summarize([float(row["mse"]) for row in top_rows])
        avg_experts_mean, avg_experts_std = summarize([float(row["average_experts_used"]) for row in top_rows])
        comparison_rows.append(
            {
                "method": f"{METHOD_LABEL} Top-{top_r}",
                "mae_mean": mae_mean,
                "mae_std": mae_std,
                "mse_mean": mse_mean,
                "mse_std": mse_std,
                "regret_to_oracle": mae_mean - oracle_mae,
                "average_experts_used_mean": avg_experts_mean,
                "average_experts_used_std": avg_experts_std,
                "router_param_count": top_rows[0]["router_param_count"],
                "num_seeds": len(top_rows),
                "primary": top_r == 2,
            }
        )

    seq_maes = [float(row["validation_mae"]) for row in sequential_rows]
    seq_mses = [float(row["validation_mse"]) for row in sequential_rows]
    seq_queries = [float(row["average_experts_queried"]) for row in sequential_rows]
    seq_mae_mean, seq_mae_std = summarize(seq_maes)
    seq_mse_mean, seq_mse_std = summarize(seq_mses)
    seq_queries_mean, seq_queries_std = summarize(seq_queries)
    comparison_rows.append(
        {
            "method": "Sequential COSTAR-TS",
            "mae_mean": seq_mae_mean,
            "mae_std": seq_mae_std,
            "mse_mean": seq_mse_mean,
            "mse_std": seq_mse_std,
            "regret_to_oracle": seq_mae_mean - oracle_mae,
            "average_experts_used_mean": seq_queries_mean,
            "average_experts_used_std": seq_queries_std,
            "router_param_count": sequential_param_count,
            "num_seeds": len(sequential_rows),
            "primary": False,
        }
    )

    for method, metrics, param_count in (
        ("Equal average all 5", equal_average, 0),
        (f"Best fixed single expert ({best_fixed['expert']})", best_fixed, 0),
        ("Oracle best single expert", oracle_single, 0),
    ):
        comparison_rows.append(
            {
                "method": method,
                "mae_mean": metrics["mae"],
                "mae_std": 0.0,
                "mse_mean": metrics["mse"],
                "mse_std": 0.0,
                "regret_to_oracle": metrics["mae"] - oracle_mae,
                "average_experts_used_mean": metrics["average_experts_used"],
                "average_experts_used_std": 0.0,
                "router_param_count": param_count,
                "num_seeds": 1,
                "primary": False,
            }
        )

    fieldnames = [
        "method",
        "mae_mean",
        "mae_std",
        "mse_mean",
        "mse_std",
        "regret_to_oracle",
        "average_experts_used_mean",
        "average_experts_used_std",
        "router_param_count",
        "num_seeds",
        "primary",
    ]
    write_csv(output_dir / "validation_comparison.csv", comparison_rows, fieldnames)
    write_csv(
        output_dir / "fame_seed_results.csv",
        fame_rows,
        [
            "method",
            "seed",
            "top_r",
            "delta",
            "mae",
            "mse",
            "average_experts_used",
            "router_param_count",
            "router_top1_accuracy",
            "top_r_oracle_coverage",
            "mean_routing_entropy",
            "train_kl",
            "checkpoint_path",
            "checkpoint_sha256",
        ],
    )
    write_csv(output_dir / "fame_expert_usage.csv", usage_rows, ["seed", "top_r", "expert", "usage_percent"])

    manifest = {
        "method": METHOD_LABEL,
        "official_fame_reference": "https://github.com/hit636/FAME",
        "adaptation_note": "ETTh history-only fingerprint adaptation; no retail lifecycle/context features used.",
        "dataset": "ETTh1",
        "split_roles": {"train": train_cache["split_role"], "validation": val_cache["split_role"]},
        "cache_paths": {"train": args.train_cache, "validation": args.val_cache},
        "cache_hashes": {"train": sha256_file(args.train_cache), "validation": sha256_file(args.val_cache)},
        "expert_ordering": list(train_cache["expert_names"]),
        "seeds": seeds,
        "tau": args.tau,
        "top_r_values": [1, 2, 3],
        "delta": args.delta,
        "hidden_size": args.hidden_size,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "loss_weights": {
            "kl": 1.0,
            "prediction_loss": 0.0,
            "load_balance": 0.0,
            "expert_cost": 0.0,
        },
        "sequential_costar_checkpoint": args.sequential_checkpoint,
        "sequential_costar_checkpoint_sha256": sha256_file(args.sequential_checkpoint),
        "sequential_costar_results": args.sequential_results,
        "git_commit": git_commit(),
        "safety": NO_TEST_DATA_USED,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "validation_comparison.json").write_text(
        json.dumps({"manifest": manifest, "comparison": comparison_rows, "fame_seed_results": fame_rows}, indent=2),
        encoding="utf-8",
    )

    print(json.dumps({"comparison": comparison_rows, "output_dir": str(output_dir), "safety": NO_TEST_DATA_USED}, indent=2))


if __name__ == "__main__":
    main()
