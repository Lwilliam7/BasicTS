"""Run reproducible ablations for SubsetUtilityCOSTARTSRouter.

The script reuses the existing subset-state cache builder outputs, trainer,
rollout evaluator, and cost-sweep evaluator. Each ablation changes one declared
factor from the baseline configuration and writes a compact paper-ready table.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Optional

import torch

try:
    from scripts.costars.build_costarts_subset_states import validate_costarts_subset_states
    from scripts.evaluate_costarts_cost_sweep import evaluate_cost_lambda
    from scripts.evaluate_costarts_final_comparison import (
        _load_torch,
        _metric_row,
        _old_costarts_predictions,
        _parameter_count,
        _sequence_tensor,
        _subset_rollout,
    )
    from scripts.evaluate_costarts_subset_utility_rollouts import evaluate_rollouts
    from scripts.train_costarts_subset_utility_router import (
        SubsetUtilityCOSTARTSRouter,
        SubsetUtilityTrainingConfig,
        _build_state_lookup,
        set_reproducible_seed,
        train_subset_utility_costarts_router,
    )
except ImportError:
    from scripts.costars.build_costarts_subset_states import validate_costarts_subset_states
    from evaluate_costarts_cost_sweep import evaluate_cost_lambda
    from evaluate_costarts_final_comparison import (
        _load_torch,
        _metric_row,
        _old_costarts_predictions,
        _parameter_count,
        _sequence_tensor,
        _subset_rollout,
    )
    from evaluate_costarts_subset_utility_rollouts import evaluate_rollouts
    from train_costarts_subset_utility_router import (
        SubsetUtilityCOSTARTSRouter,
        SubsetUtilityTrainingConfig,
        _build_state_lookup,
        set_reproducible_seed,
        train_subset_utility_costarts_router,
    )


DEFAULT_BASE_TRAIN_CACHE = "cache/costarts_router_train_cache.pt"
DEFAULT_BASE_VAL_CACHE = "cache/costarts_router_val_cache.pt"
DEFAULT_SUBSET_TRAIN_CACHE = "cache/costarts_subset_states_train.pt"
DEFAULT_SUBSET_VAL_CACHE = "cache/costarts_subset_states_val.pt"
DEFAULT_OLD_COSTARTS_CHECKPOINT = "checkpoints/costarts/best_costarts_router.pt"
DEFAULT_OUTPUT_DIR = "results/router_summary/costarts_subset_utility"
DEFAULT_ABLATION_CHECKPOINT_DIR = "checkpoints/costarts_subset_utility/ablations"
DEFAULT_ABLATION_CACHE_DIR = "cache/costarts_ablations"


@dataclass(frozen=True)
class AblationSpec:
    name: str
    changed_factor: str
    description: str
    train: bool = True
    train_cache_path: Optional[str] = None
    action_loss_weight: float = 1.0
    utility_loss_weight: float = 1.0
    pairwise_loss_weight: float = 0.2
    mix_loss_weight: float = 1.0
    use_expert_embeddings: bool = True
    history_encoder_type: str = "current"
    action_head_type: str = "unified"
    finalizer: str = "sparse_mixture"
    cost_lambda: float = 0.0
    dagger_fine_tune: bool = False
    status_override: Optional[str] = None


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return str(value)
    return value


def _copy_state_cache_rows(cache: Mapping[str, Any], row_indices: torch.Tensor, sampling_mode: str) -> dict[str, Any]:
    num_states = int(cache["num_states"])
    selected = row_indices.to(torch.long)
    new_cache: dict[str, Any] = {}
    for key, value in cache.items():
        if torch.is_tensor(value) and value.shape[:1] == (num_states,):
            new_cache[key] = value[selected].clone()
        else:
            new_cache[key] = value
    new_cache["num_states"] = int(selected.numel())
    new_cache["subset_sampling_mode"] = sampling_mode
    counts = torch.bincount(new_cache["subset_size"].to(torch.long), minlength=int(cache["max_subset_size"]) + 1)
    new_cache["state_counts_by_subset_size"] = {
        str(index): int(value)
        for index, value in enumerate(counts.tolist())
        if value
    }
    return new_cache


def _oracle_path_indices(cache: Mapping[str, Any]) -> torch.Tensor:
    validate_costarts_subset_states(cache)
    lookup = _build_state_lookup(cache)
    stop_index = int(cache["stop_action_index"])
    num_source_windows = int(cache["num_source_windows"])
    selected: list[int] = []
    for row in range(num_source_windows):
        mask = 0
        seen_masks = set()
        while mask not in seen_masks:
            seen_masks.add(mask)
            state_index = lookup[row][mask]
            selected.append(state_index)
            action = int(cache["optimal_next_action"][state_index])
            if action == stop_index:
                break
            mask |= 1 << action
    return torch.tensor(sorted(set(selected)), dtype=torch.long)


def _ensure_oracle_path_cache(source_path: Path, output_path: Path, force: bool = False) -> Path:
    if output_path.exists() and not force:
        return output_path
    cache = _load_torch(source_path)
    indices = _oracle_path_indices(cache)
    oracle_cache = _copy_state_cache_rows(cache, indices, "oracle_path_only")
    validate_costarts_subset_states(oracle_cache)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(oracle_cache, output_path)
    print(f"Saved oracle-path-only cache: {output_path} ({len(indices)} states)")
    return output_path


def _load_subset_router(checkpoint_path: Path, device: torch.device) -> tuple[SubsetUtilityCOSTARTSRouter, dict[str, Any]]:
    checkpoint = _load_torch(checkpoint_path)
    router = SubsetUtilityCOSTARTSRouter(**checkpoint["router_config"]).to(device)
    router.load_state_dict(checkpoint["router_state_dict"])
    router.eval()
    return router, checkpoint


def _evaluate_subset_checkpoint(
    *,
    spec: AblationSpec,
    checkpoint_path: Path,
    val_cache: Mapping[str, Any],
    subset_val_cache: Mapping[str, Any],
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    router, checkpoint = _load_subset_router(checkpoint_path, device)
    oracle_mae = float(val_cache["error_matrix"].min(dim=1).values.mean())
    if spec.cost_lambda > 0:
        expert_costs = torch.ones(int(subset_val_cache["num_experts"]), dtype=torch.float32)
        metrics = evaluate_cost_lambda(
            router=router,
            cache=subset_val_cache,
            query_lambda=spec.cost_lambda,
            expert_costs=expert_costs,
            batch_size=batch_size,
            device=device,
            finalizer="reranker",
            max_queries=None,
        )
        return {
            "ablation": spec.name,
            "changed_factor": spec.changed_factor,
            "status": "ok",
            "mae": metrics["mae"],
            "mse": metrics["mse"],
            "regret_to_oracle": metrics["regret_to_oracle"],
            "average_experts_queried": metrics["average_experts_queried"],
            "top2_oracle_coverage": "",
            "first_query_oracle_match": "",
            "oracle_match_rate": metrics["oracle_match_rate"],
            "parameter_count": _parameter_count(router),
            "checkpoint": str(checkpoint_path),
            "best_epoch": checkpoint.get("epoch", ""),
            "finalizer": "cost_aware_reranker",
            "cost_lambda": spec.cost_lambda,
            "dagger_fine_tune": spec.dagger_fine_tune,
            "description": spec.description,
        }

    if spec.finalizer == "sparse_mixture":
        payload = evaluate_rollouts(
            router=router,
            cache=subset_val_cache,
            mode="greedy",
            finalizer="sparse_mixture",
            force_k=None,
            temperature=1.0,
            max_queries=None,
            batch_size=batch_size,
            device=device,
            seed=7,
            detailed_limit=0,
        )
        metrics = payload["metrics"]
        return {
            "ablation": spec.name,
            "changed_factor": spec.changed_factor,
            "status": "ok",
            "mae": metrics["mae"],
            "mse": metrics["mse"],
            "regret_to_oracle": metrics["regret_to_oracle"],
            "average_experts_queried": metrics["average_experts_queried"],
            "top2_oracle_coverage": "",
            "first_query_oracle_match": "",
            "oracle_match_rate": metrics["oracle_match_rate"],
            "latency_seconds": metrics["latency_seconds"],
            "parameter_count": _parameter_count(router),
            "checkpoint": str(checkpoint_path),
            "best_epoch": checkpoint.get("epoch", ""),
            "finalizer": spec.finalizer,
            "cost_lambda": spec.cost_lambda,
            "dagger_fine_tune": spec.dagger_fine_tune,
            "description": spec.description,
        }

    prediction, selected, sequences, latency = _subset_rollout(
        router=router,
        subset_cache=subset_val_cache,
        batch_size=batch_size,
        device=device,
    )
    top2 = _sequence_tensor(sequences, 2)
    first_query = top2[:, 0]
    metric = _metric_row(
        method=spec.name,
        status="ok",
        cache=val_cache,
        oracle_mae=oracle_mae,
        prediction=prediction,
        selected_experts=selected,
        average_experts_queried=float(sum(len(sequence) for sequence in sequences) / max(len(sequences), 1)),
        latency_seconds=latency,
        parameter_count=_parameter_count(router),
        top2_indices=top2,
        first_query=first_query,
        selection_split="router_val_checkpoint_selected",
        note=spec.description,
    )
    return {
        "ablation": spec.name,
        "changed_factor": spec.changed_factor,
        "status": "ok",
        "mae": metric["mae"],
        "mse": metric["mse"],
        "regret_to_oracle": metric["regret_to_oracle"],
        "average_experts_queried": metric["average_experts_queried"],
        "top2_oracle_coverage": metric["top2_oracle_coverage"],
        "first_query_oracle_match": metric["first_query_oracle_match"],
        "oracle_match_rate": metric["oracle_match_rate"],
        "latency_seconds": metric["latency_seconds"],
        "parameter_count": metric["parameter_count"],
        "checkpoint": str(checkpoint_path),
        "best_epoch": checkpoint.get("epoch", ""),
        "finalizer": spec.finalizer,
        "cost_lambda": spec.cost_lambda,
        "dagger_fine_tune": spec.dagger_fine_tune,
        "description": spec.description,
    }


def _old_costarts_row(
    *,
    checkpoint_path: Path,
    val_cache: Mapping[str, Any],
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    oracle_mae = float(val_cache["error_matrix"].min(dim=1).values.mean())
    if not checkpoint_path.exists():
        return {
            "ablation": "no_subset_state_supervision",
            "changed_factor": "subset_state_supervision",
            "status": "skipped",
            "description": f"Missing old COSTARTS checkpoint: {checkpoint_path}",
        }
    router, selected, stop_step, query_order, epoch, latency = _old_costarts_predictions(
        checkpoint_path=checkpoint_path,
        cache=val_cache,
        batch_size=batch_size,
        device=device,
    )
    metric = _metric_row(
        method="no_subset_state_supervision",
        status="ok",
        cache=val_cache,
        oracle_mae=oracle_mae,
        selected_experts=selected,
        average_experts_queried=float(stop_step.to(torch.float32).mean()),
        latency_seconds=latency,
        parameter_count=_parameter_count(router),
        top2_indices=query_order[:, :2],
        first_query=query_order[:, 0],
        selection_split="router_val_checkpoint_selected",
        note="Old one-shot COSTARTS, no subset-state supervision.",
    )
    return {
        "ablation": "no_subset_state_supervision",
        "changed_factor": "subset_state_supervision",
        "status": "ok",
        "mae": metric["mae"],
        "mse": metric["mse"],
        "regret_to_oracle": metric["regret_to_oracle"],
        "average_experts_queried": metric["average_experts_queried"],
        "top2_oracle_coverage": metric["top2_oracle_coverage"],
        "first_query_oracle_match": metric["first_query_oracle_match"],
        "oracle_match_rate": metric["oracle_match_rate"],
        "latency_seconds": metric["latency_seconds"],
        "parameter_count": metric["parameter_count"],
        "checkpoint": str(checkpoint_path),
        "best_epoch": epoch,
        "finalizer": "old_costarts_predicted_error",
        "cost_lambda": 0.0,
        "dagger_fine_tune": False,
        "description": "Old COSTARTS without exhaustive subset-state recovery supervision.",
    }


def _ablation_specs(oracle_train_cache: str, subset_train_cache: str, cost_lambda: float) -> list[AblationSpec]:
    return [
        AblationSpec(
            name="full_improved_zero_cost",
            changed_factor="none",
            description="Baseline improved subset-utility router with exhaustive subset states, utility, pairwise, mix loss, expert embeddings, current encoder, unified M+1 action.",
            train_cache_path=subset_train_cache,
        ),
        AblationSpec(
            name="oracle_path_only_supervision",
            changed_factor="subset_state_sampling",
            description="Train only on states reachable under the oracle path.",
            train_cache_path=oracle_train_cache,
        ),
        AblationSpec(
            name="no_utility_regression",
            changed_factor="utility_loss_weight",
            description="Utility regression loss weight set to zero.",
            train_cache_path=subset_train_cache,
            utility_loss_weight=0.0,
        ),
        AblationSpec(
            name="no_pairwise_reranking",
            changed_factor="pairwise_loss_weight",
            description="Pairwise reranking loss weight set to zero.",
            train_cache_path=subset_train_cache,
            pairwise_loss_weight=0.0,
        ),
        AblationSpec(
            name="no_sparse_mixing",
            changed_factor="mix_loss_weight_and_finalizer",
            description="Mix loss disabled; final selection uses the reranker path.",
            train_cache_path=subset_train_cache,
            mix_loss_weight=0.0,
            finalizer="reranker",
        ),
        AblationSpec(
            name="separate_stop_query_heads",
            changed_factor="action_head_type",
            description="Replace unified M+1 action head with separate query and stop heads.",
            train_cache_path=subset_train_cache,
            action_head_type="separate_stop_query",
        ),
        AblationSpec(
            name="cost_aware_stopping",
            changed_factor="stopping_cost",
            description=f"Evaluate the full improved router with cost-aware stopping at lambda={cost_lambda}.",
            train=False,
            cost_lambda=cost_lambda,
            finalizer="cost_aware_reranker",
        ),
        AblationSpec(
            name="no_expert_embeddings",
            changed_factor="expert_embeddings",
            description="Disable trainable expert ID embeddings inside queried forecast encoding.",
            train_cache_path=subset_train_cache,
            use_expert_embeddings=False,
        ),
        AblationSpec(
            name="simple_history_encoder",
            changed_factor="history_encoder",
            description="Use a single-convolution history encoder instead of the current two-layer dilated encoder.",
            train_cache_path=subset_train_cache,
            history_encoder_type="simple",
        ),
        AblationSpec(
            name="no_dagger_fine_tune",
            changed_factor="dagger",
            description="Baseline path without DAgger fine-tuning.",
            train=False,
            dagger_fine_tune=False,
        ),
        AblationSpec(
            name="dagger_fine_tune",
            changed_factor="dagger",
            description="Optional DAgger fine-tune is not implemented yet; row is retained for paper-ready bookkeeping.",
            train=False,
            dagger_fine_tune=True,
            status_override="skipped_optional",
        ),
    ]


def run_ablations(args: argparse.Namespace) -> dict[str, Any]:
    set_reproducible_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    ablation_checkpoint_root = Path(args.ablation_checkpoint_dir)
    ablation_results_root = output_dir / "ablation_runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    ablation_checkpoint_root.mkdir(parents=True, exist_ok=True)
    ablation_results_root.mkdir(parents=True, exist_ok=True)

    val_cache = _load_torch(Path(args.base_val_cache))
    subset_val_cache = _load_torch(Path(args.subset_val_cache))
    validate_costarts_subset_states(subset_val_cache)
    if tuple(val_cache["expert_names"]) != tuple(subset_val_cache["expert_names"]):
        raise AssertionError("Base val and subset val caches have different expert ordering.")

    oracle_train_cache = _ensure_oracle_path_cache(
        Path(args.subset_train_cache),
        Path(args.ablation_cache_dir) / "costarts_subset_states_train_oracle_path_only.pt",
        force=args.force_rebuild_oracle_path_cache,
    )
    specs = _ablation_specs(str(oracle_train_cache), args.subset_train_cache, args.cost_lambda)
    selected_names = set(args.only.split(",")) if args.only else None
    rows: list[dict[str, Any]] = []
    run_details: dict[str, Any] = {}
    full_checkpoint_path: Optional[Path] = None

    rows.append(
        _old_costarts_row(
            checkpoint_path=Path(args.old_costarts_checkpoint),
            val_cache=val_cache,
            batch_size=args.batch_size,
            device=device,
        )
    )

    for spec in specs:
        if selected_names is not None and spec.name not in selected_names:
            continue
        if spec.status_override:
            rows.append(
                {
                    "ablation": spec.name,
                    "changed_factor": spec.changed_factor,
                    "status": spec.status_override,
                    "description": spec.description,
                    "dagger_fine_tune": spec.dagger_fine_tune,
                }
            )
            continue

        if not spec.train:
            checkpoint_path = full_checkpoint_path or Path(args.existing_full_checkpoint)
            if not checkpoint_path.exists():
                rows.append(
                    {
                        "ablation": spec.name,
                        "changed_factor": spec.changed_factor,
                        "status": "skipped",
                        "description": f"Needs full improved checkpoint, missing: {checkpoint_path}",
                    }
                )
                continue
            row = _evaluate_subset_checkpoint(
                spec=spec,
                checkpoint_path=checkpoint_path,
                val_cache=val_cache,
                subset_val_cache=subset_val_cache,
                batch_size=args.batch_size,
                device=device,
            )
            rows.append(row)
            continue

        checkpoint_dir = ablation_checkpoint_root / spec.name
        results_dir = ablation_results_root / spec.name
        checkpoint_path = checkpoint_dir / "best_subset_utility_costarts_router.pt"
        if not (args.reuse_existing and checkpoint_path.exists()):
            training_config = SubsetUtilityTrainingConfig(
                train_cache_path=str(spec.train_cache_path),
                val_cache_path=args.subset_val_cache,
                output_dir=str(checkpoint_dir),
                results_dir=str(results_dir),
                batch_size=args.batch_size,
                max_epochs=args.max_epochs,
                patience=args.patience,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                grad_clip_norm=args.grad_clip_norm,
                seed=args.seed,
                action_loss_weight=spec.action_loss_weight,
                utility_loss_weight=spec.utility_loss_weight,
                pairwise_loss_weight=spec.pairwise_loss_weight,
                mix_loss_weight=spec.mix_loss_weight,
                subset_state_sampling_mode=(
                    "oracle_path_only"
                    if "oracle_path_only" in str(spec.train_cache_path)
                    else "exhaustive"
                ),
                max_subset_size=None,
                cost_coefficient=1.0,
                use_expert_embeddings=spec.use_expert_embeddings,
                history_encoder_type=spec.history_encoder_type,
                action_head_type=spec.action_head_type,
                device=args.device,
                debug=args.debug,
            )
            print("\n" + "=" * 80)
            print(f"Training ablation: {spec.name}")
            print("=" * 80)
            summary = train_subset_utility_costarts_router(training_config)
            run_details[spec.name] = summary
        else:
            run_details[spec.name] = {"reused_checkpoint": str(checkpoint_path)}

        if spec.name == "full_improved_zero_cost":
            full_checkpoint_path = checkpoint_path
        row = _evaluate_subset_checkpoint(
            spec=spec,
            checkpoint_path=checkpoint_path,
            val_cache=val_cache,
            subset_val_cache=subset_val_cache,
            batch_size=args.batch_size,
            device=device,
        )
        rows.append(row)

    fields = [
        "ablation",
        "changed_factor",
        "status",
        "mae",
        "mse",
        "regret_to_oracle",
        "average_experts_queried",
        "top2_oracle_coverage",
        "first_query_oracle_match",
        "oracle_match_rate",
        "latency_seconds",
        "parameter_count",
        "checkpoint",
        "best_epoch",
        "finalizer",
        "cost_lambda",
        "dagger_fine_tune",
        "description",
    ]
    csv_path = output_dir / "ablations.csv"
    json_path = output_dir / "ablations.json"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "metadata": {
            "base_train_cache": args.base_train_cache,
            "base_val_cache": args.base_val_cache,
            "subset_train_cache": args.subset_train_cache,
            "subset_val_cache": args.subset_val_cache,
            "num_validation_windows": int(val_cache["num_windows"]),
            "expert_names": tuple(val_cache["expert_names"]),
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "seed": args.seed,
            "test_data_used": False,
            "one_factor_at_a_time": True,
            "note": "Rows share the same chronological router_val cache. DAgger is optional and skipped until implemented.",
        },
        "rows": rows,
        "run_details": run_details,
    }
    json_path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run subset-utility COSTARTS ablations.")
    parser.add_argument("--base-train-cache", default=DEFAULT_BASE_TRAIN_CACHE)
    parser.add_argument("--base-val-cache", default=DEFAULT_BASE_VAL_CACHE)
    parser.add_argument("--subset-train-cache", default=DEFAULT_SUBSET_TRAIN_CACHE)
    parser.add_argument("--subset-val-cache", default=DEFAULT_SUBSET_VAL_CACHE)
    parser.add_argument("--old-costarts-checkpoint", default=DEFAULT_OLD_COSTARTS_CHECKPOINT)
    parser.add_argument("--existing-full-checkpoint", default="checkpoints/costarts_subset_utility/best_subset_utility_costarts_router.pt")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ablation-checkpoint-dir", default=DEFAULT_ABLATION_CHECKPOINT_DIR)
    parser.add_argument("--ablation-cache-dir", default=DEFAULT_ABLATION_CACHE_DIR)
    parser.add_argument("--force-rebuild-oracle-path-cache", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--only", default=None, help="Comma-separated ablation names to run after the old-COSTARTS reference row.")
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--cost-lambda", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    payload = run_ablations(parse_args())
    ok_rows = [row for row in payload["rows"] if row.get("status") == "ok" and row.get("mae") != ""]
    print("\nAblation rows:")
    for row in ok_rows:
        print(
            f"{row['ablation']}: MAE={float(row['mae']):.6f}, "
            f"avg_q={row.get('average_experts_queried', '')}, "
            f"factor={row['changed_factor']}"
        )


if __name__ == "__main__":
    main()
