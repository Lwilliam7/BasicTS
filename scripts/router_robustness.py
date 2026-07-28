"""Robustness experiments for frozen-expert routing methods."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.chronological_expert_training import (
    DEFAULT_INPUT_LEN,
    DEFAULT_NUM_FEATURES,
    DEFAULT_OUTPUT_LEN,
    _accumulate_errors,
    _call_expert_model,
    _prepare_forecasting_batch,
    _assert_full_data_contract,
    assert_experts_frozen,
    build_selected_candidate_experts,
    load_full_chronological_data,
    prepare_chronological_dataloaders,
    selected_router_model_groups,
)
from scripts.router_experiment_config import (
    load_router_experiment_config,
    print_router_experiment_config,
    validate_router_experiment_config,
)


DEFAULT_SEEDS = (7, 11, 13, 17, 19)


@dataclass(frozen=True)
class RobustnessCondition:
    condition_name: str
    train_fraction: float = 1.0
    noise_std: float = 0.0
    missing_variable_fraction: float = 0.0
    masked_timestamp_fraction: float = 0.0
    temporal_shift_steps: int = 0
    max_queried_experts_k: Optional[int] = None
    routing_temperature: float = 1.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_csv_floats(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def parse_csv_strings(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def make_conditions(
    train_fractions: Sequence[float],
    noise_stds: Sequence[float],
    missing_variable_fractions: Sequence[float],
    masked_timestamp_fractions: Sequence[float],
    temporal_shift_steps: Sequence[int],
    max_k_values: Sequence[int],
    routing_temperatures: Sequence[float],
    full_factorial: bool,
) -> tuple[RobustnessCondition, ...]:
    if full_factorial:
        conditions = []
        for values in itertools.product(
            train_fractions,
            noise_stds,
            missing_variable_fractions,
            masked_timestamp_fractions,
            temporal_shift_steps,
            max_k_values,
            routing_temperatures,
        ):
            train_fraction, noise_std, missing_var, masked_ts, shift, max_k, temp = values
            conditions.append(
                RobustnessCondition(
                    condition_name=(
                        f"full_tf{train_fraction:g}_noise{noise_std:g}_"
                        f"missv{missing_var:g}_masit{masked_ts:g}_"
                        f"shift{shift}_k{max_k}_temp{temp:g}"
                    ),
                    train_fraction=train_fraction,
                    noise_std=noise_std,
                    missing_variable_fraction=missing_var,
                    masked_timestamp_fraction=masked_ts,
                    temporal_shift_steps=shift,
                    max_queried_experts_k=max_k,
                    routing_temperature=temp,
                )
            )
        return tuple(conditions)

    baseline_k = max(max_k_values) if max_k_values else None
    conditions = [
        RobustnessCondition(
            condition_name="baseline_clean",
            max_queried_experts_k=baseline_k,
        )
    ]
    for value in train_fractions:
        if value != 1.0:
            conditions.append(
                RobustnessCondition(
                    condition_name=f"reduced_train_fraction_{value:g}",
                    train_fraction=value,
                    max_queried_experts_k=baseline_k,
                )
            )
    for value in noise_stds:
        if value != 0.0:
            conditions.append(
                RobustnessCondition(
                    condition_name=f"history_noise_std_{value:g}",
                    noise_std=value,
                    max_queried_experts_k=baseline_k,
                )
            )
    for value in missing_variable_fractions:
        if value != 0.0:
            conditions.append(
                RobustnessCondition(
                    condition_name=f"missing_variable_fraction_{value:g}",
                    missing_variable_fraction=value,
                    max_queried_experts_k=baseline_k,
                )
            )
    for value in masked_timestamp_fractions:
        if value != 0.0:
            conditions.append(
                RobustnessCondition(
                    condition_name=f"masked_timestamp_fraction_{value:g}",
                    masked_timestamp_fraction=value,
                    max_queried_experts_k=baseline_k,
                )
            )
    for value in temporal_shift_steps:
        if value != 0:
            conditions.append(
                RobustnessCondition(
                    condition_name=f"mild_temporal_shift_{value}",
                    temporal_shift_steps=value,
                    max_queried_experts_k=baseline_k,
                )
            )
    for value in max_k_values:
        conditions.append(
            RobustnessCondition(
                condition_name=f"max_queried_experts_k_{value}",
                max_queried_experts_k=value,
            )
        )
    for value in routing_temperatures:
        if value != 1.0:
            conditions.append(
                RobustnessCondition(
                    condition_name=f"routing_temperature_{value:g}",
                    routing_temperature=value,
                    max_queried_experts_k=baseline_k,
                )
            )
    unique = {}
    for condition in conditions:
        unique[condition.condition_name] = condition
    return tuple(unique.values())


def _clone_targets_for_alignment(targets: torch.Tensor) -> torch.Tensor:
    return targets.detach().clone()


def perturb_histories(
    histories: torch.Tensor,
    targets: torch.Tensor,
    condition: RobustnessCondition,
    generator: torch.Generator,
) -> torch.Tensor:
    before_targets = _clone_targets_for_alignment(targets)
    perturbed = histories.clone()
    if condition.noise_std > 0:
        noise = torch.randn(
            perturbed.shape,
            dtype=perturbed.dtype,
            device=perturbed.device,
            generator=generator if perturbed.device.type == "cpu" else None,
        )
        perturbed = perturbed + noise * float(condition.noise_std)
    if condition.missing_variable_fraction > 0:
        variable_count = max(
            1,
            min(
                perturbed.shape[-1],
                int(round(perturbed.shape[-1] * condition.missing_variable_fraction)),
            ),
        )
        indices = torch.randperm(
            perturbed.shape[-1],
            generator=generator if perturbed.device.type == "cpu" else None,
            device=perturbed.device,
        )[:variable_count]
        perturbed[:, :, indices] = 0.0
    if condition.masked_timestamp_fraction > 0:
        timestamp_count = max(
            1,
            min(
                perturbed.shape[1],
                int(round(perturbed.shape[1] * condition.masked_timestamp_fraction)),
            ),
        )
        indices = torch.randperm(
            perturbed.shape[1],
            generator=generator if perturbed.device.type == "cpu" else None,
            device=perturbed.device,
        )[:timestamp_count]
        perturbed[:, indices, :] = 0.0
    if condition.temporal_shift_steps:
        perturbed = torch.roll(perturbed, shifts=int(condition.temporal_shift_steps), dims=1)
    if not torch.equal(targets, before_targets):
        raise AssertionError("Perturbation pipeline altered targets and broke alignment")
    if perturbed.shape != histories.shape:
        raise AssertionError("Perturbation pipeline changed history tensor shape")
    return perturbed


def limited_batches(loader: DataLoader, max_windows: Optional[int]) -> Iterable[dict]:
    seen = 0
    for batch in loader:
        if max_windows is None:
            yield batch
            continue
        batch_size = len(batch["inputs"])
        if seen >= max_windows:
            break
        if seen + batch_size <= max_windows:
            yield batch
        else:
            keep = max_windows - seen
            yield {
                key: value[:keep] if hasattr(value, "__getitem__") else value
                for key, value in batch.items()
            }
        seen += batch_size


@torch.no_grad()
def expert_error_matrix(
    experts: Sequence[torch.nn.Module],
    expert_names: Sequence[str],
    loader: DataLoader,
    scaler,
    device: torch.device,
    condition: RobustnessCondition,
    seed: int,
    max_windows: Optional[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    all_abs = []
    all_sq = []
    for expert in experts:
        expert.eval()
    assert_experts_frozen(*experts)
    for batch in limited_batches(loader, max_windows):
        histories, targets, target_mask = _prepare_forecasting_batch(batch, device, scaler)
        histories = perturb_histories(histories, targets, condition, generator)
        if histories.shape[1:] != (DEFAULT_INPUT_LEN, DEFAULT_NUM_FEATURES):
            raise AssertionError(f"History shape changed: {tuple(histories.shape)}")
        if targets.shape[1:] != (DEFAULT_OUTPUT_LEN, DEFAULT_NUM_FEATURES):
            raise AssertionError(f"Target shape changed: {tuple(targets.shape)}")

        predictions = []
        for expert_name, expert in zip(expert_names, experts):
            prediction = _call_expert_model(expert, histories)
            if prediction.shape != targets.shape:
                raise ValueError(
                    f"{expert_name} prediction shape {tuple(prediction.shape)} "
                    f"does not match target shape {tuple(targets.shape)}"
                )
            predictions.append(prediction)
        stack = torch.stack(predictions, dim=-1)
        mask = target_mask.to(stack.dtype).unsqueeze(-1)
        denominator = mask.sum(dim=(1, 2)).clamp_min(1.0)
        all_abs.append((torch.abs(stack - targets.unsqueeze(-1)) * mask).sum(dim=(1, 2)) / denominator)
        all_sq.append(((stack - targets.unsqueeze(-1)).pow(2) * mask).sum(dim=(1, 2)) / denominator)
    return torch.cat(all_abs, dim=0).cpu(), torch.cat(all_sq, dim=0).cpu()


def train_subset_mean_errors(
    error_matrix: torch.Tensor,
    train_fraction: float,
) -> torch.Tensor:
    if not (0 < train_fraction <= 1):
        raise ValueError("train_fraction must be in (0, 1]")
    count = max(1, int(math.ceil(error_matrix.shape[0] * train_fraction)))
    return error_matrix[:count].mean(dim=0)


def top_k_indices(mean_errors: torch.Tensor, max_k: Optional[int]) -> torch.Tensor:
    expert_count = mean_errors.numel()
    k = expert_count if max_k is None else min(max(1, int(max_k)), expert_count)
    return torch.argsort(mean_errors)[:k]


def evaluate_static_methods(
    train_errors: torch.Tensor,
    val_errors: torch.Tensor,
    val_mse: torch.Tensor,
    expert_names: Sequence[str],
    condition: RobustnessCondition,
    seed: int,
    expert_subset_name: str,
) -> list[dict]:
    mean_errors = train_subset_mean_errors(train_errors, condition.train_fraction)
    allowed = top_k_indices(mean_errors, condition.max_queried_experts_k)
    allowed_names = [expert_names[int(index)] for index in allowed]
    allowed_val = val_errors[:, allowed]
    allowed_mse = val_mse[:, allowed]
    oracle_selected = torch.argmin(val_errors, dim=1)
    capped_oracle_local = torch.argmin(allowed_val, dim=1)
    capped_oracle_errors = allowed_val.gather(1, capped_oracle_local[:, None]).squeeze(1)
    capped_oracle_mse = allowed_mse.gather(1, capped_oracle_local[:, None]).squeeze(1)

    best_index = torch.argmin(mean_errors)
    best_errors = val_errors[:, best_index]
    best_mse = val_mse[:, best_index]

    weights = torch.softmax(-mean_errors[allowed] / float(condition.routing_temperature), dim=0)
    soft_errors = (allowed_val * weights.view(1, -1)).sum(dim=1)
    soft_mse = (allowed_mse * weights.view(1, -1)).sum(dim=1)

    generator = torch.Generator()
    generator.manual_seed(seed)
    random_local = torch.randint(0, len(allowed), (val_errors.shape[0],), generator=generator)
    random_errors = allowed_val.gather(1, random_local[:, None]).squeeze(1)
    random_mse = allowed_mse.gather(1, random_local[:, None]).squeeze(1)

    rows = []
    method_payloads = [
        ("validation_selected_best_expert", best_errors, best_mse, 1.0, expert_names[int(best_index)]),
        ("fixed_validation_soft_weights", soft_errors, soft_mse, float(len(allowed)), "+".join(allowed_names)),
        ("oracle_within_allowed_k", capped_oracle_errors, capped_oracle_mse, float(len(allowed)), "+".join(allowed_names)),
        ("random_routing_within_allowed_k", random_errors, random_mse, 1.0, "+".join(allowed_names)),
    ]
    full_oracle = torch.min(val_errors, dim=1).values
    for method, errors, mse_values, relative_cost, selected in method_payloads:
        rows.append(
            {
                "seed": seed,
                "condition_name": condition.condition_name,
                "expert_subset": expert_subset_name,
                "selected_experts": "+".join(expert_names),
                "method": method,
                "train_fraction": condition.train_fraction,
                "noise_std": condition.noise_std,
                "missing_variable_fraction": condition.missing_variable_fraction,
                "masked_timestamp_fraction": condition.masked_timestamp_fraction,
                "temporal_shift_steps": condition.temporal_shift_steps,
                "max_queried_experts_k": len(allowed),
                "routing_temperature": condition.routing_temperature,
                "selected_by_method": selected,
                "val_mae": float(errors.mean()),
                "val_mse": float(mse_values.mean()),
                "val_regret_to_full_oracle": float((errors - full_oracle).mean()),
                "oracle_expert_match_rate": float((oracle_selected == allowed[capped_oracle_local]).float().mean()),
                "relative_expert_cost": relative_cost,
                "num_val_windows": int(val_errors.shape[0]),
            }
        )
    return rows


def aggregate_results(rows: Sequence[dict]) -> list[dict]:
    if not rows:
        return []
    groups = {}
    keys = (
        "condition_name",
        "expert_subset",
        "method",
        "train_fraction",
        "noise_std",
        "missing_variable_fraction",
        "masked_timestamp_fraction",
        "temporal_shift_steps",
        "max_queried_experts_k",
        "routing_temperature",
    )
    for row in rows:
        key = tuple(row[item] for item in keys)
        groups.setdefault(key, []).append(row)
    aggregated = []
    for key, group_rows in groups.items():
        values = {name: key[index] for index, name in enumerate(keys)}
        for metric in ("val_mae", "val_mse", "val_regret_to_full_oracle", "relative_expert_cost"):
            metric_values = np.asarray([float(row[metric]) for row in group_rows], dtype=float)
            values[f"{metric}_mean"] = float(metric_values.mean())
            values[f"{metric}_std"] = float(metric_values.std(ddof=1)) if len(metric_values) > 1 else 0.0
        values["num_seeds"] = len({row["seed"] for row in group_rows})
        values["seeds"] = ",".join(str(row["seed"]) for row in group_rows)
        aggregated.append(values)
    return aggregated


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_environment(args: argparse.Namespace, device: torch.device):
    from basicts.scaler import ZScoreScaler

    config = validate_router_experiment_config(
        load_router_experiment_config(),
        require_checkpoints=True,
        require_data=True,
        require_cache_parent=True,
    )
    print_router_experiment_config(config)
    full_data = load_full_chronological_data(args.data_dir)
    _assert_full_data_contract(full_data, DEFAULT_NUM_FEATURES)
    loaders, scaler = prepare_chronological_dataloaders(
        full_data=full_data,
        scaler=ZScoreScaler(norm_each_channel=True, rescale=False),
        batch_size=args.batch_size,
        input_len=DEFAULT_INPUT_LEN,
        output_len=DEFAULT_OUTPUT_LEN,
    )
    if getattr(loaders["router_train"].dataset, "split_role", None) != "router_train":
        raise AssertionError("Robustness training subset must use router_train")
    if getattr(loaders["router_val"].dataset, "split_role", None) != "router_val":
        raise AssertionError("Robustness evaluation must use router_val")
    return loaders, scaler


def selected_expert_groups(args: argparse.Namespace):
    groups = selected_router_model_groups()
    if args.max_expert_groups is not None:
        groups = groups[: args.max_expert_groups]
    return groups


def run_robustness(args: argparse.Namespace) -> dict:
    if len(args.seeds) < 5:
        raise ValueError("Robustness testing requires at least 5 random seeds")
    device = torch.device(args.device)
    loaders, scaler = load_environment(args, device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    latency_rows = []
    groups = selected_expert_groups(args)

    for group_index, specs in enumerate(groups, start=1):
        subset_name = f"group_{group_index}_{len(specs)}_experts"
        experts, expert_names, _, _ = build_selected_candidate_experts(
            checkpoint_dir=args.checkpoint_dir,
            device=device,
            scaler=scaler,
            specs=specs,
        )
        assert_experts_frozen(*experts)
        max_k_values = tuple(
            value for value in args.max_k_values if value <= len(expert_names)
        ) or (len(expert_names),)
        conditions = make_conditions(
            args.train_fractions,
            args.noise_stds,
            args.missing_variable_fractions,
            args.masked_timestamp_fractions,
            args.temporal_shift_steps,
            max_k_values,
            args.routing_temperatures,
            args.full_factorial,
        )

        for seed in args.seeds:
            set_seed(seed)
            clean_condition = RobustnessCondition("train_cache_clean")
            train_errors, _ = expert_error_matrix(
                experts,
                expert_names,
                loaders["router_train"],
                scaler,
                device,
                clean_condition,
                seed,
                args.max_train_windows,
            )
            train_errors = train_errors.cpu()
            train_count = train_errors.shape[0]
            for condition in conditions:
                start_time = time.perf_counter()
                val_errors, val_mse = expert_error_matrix(
                    experts,
                    expert_names,
                    loaders["router_val"],
                    scaler,
                    device,
                    condition,
                    seed,
                    args.max_val_windows,
                )
                elapsed = time.perf_counter() - start_time
                method_rows = evaluate_static_methods(
                    train_errors,
                    val_errors,
                    val_mse,
                    expert_names,
                    condition,
                    seed,
                    subset_name,
                )
                for row in method_rows:
                    row["train_windows_used"] = int(
                        max(1, math.ceil(train_count * condition.train_fraction))
                    )
                    row["latency_device"] = str(device)
                    row["eval_elapsed_seconds"] = elapsed
                    rows.append(row)
                latency_rows.append(
                    {
                        "seed": seed,
                        "expert_subset": subset_name,
                        "condition_name": condition.condition_name,
                        "device": str(device),
                        "num_experts": len(expert_names),
                        "num_val_windows": int(val_errors.shape[0]),
                        "elapsed_seconds": elapsed,
                        "windows_per_second": float(val_errors.shape[0] / max(elapsed, 1e-12)),
                        "latency_note": "matrix evaluation timing",
                    }
                )
                print(
                    f"{subset_name} seed={seed} condition={condition.condition_name} "
                    f"experts={len(expert_names)} val_windows={val_errors.shape[0]} "
                    f"elapsed={elapsed:.2f}s"
                )

    latency_rows.extend(run_latency_device_benchmarks(args, loaders, scaler, groups))

    detail_path = output_dir / "router_robustness_results.csv"
    aggregate_path = output_dir / "router_robustness_aggregate.csv"
    latency_path = output_dir / "router_robustness_latency.csv"
    manifest_path = output_dir / "router_robustness_manifest.json"
    aggregate_rows = aggregate_results(rows)
    write_csv(detail_path, rows)
    write_csv(aggregate_path, aggregate_rows)
    write_csv(latency_path, latency_rows)
    manifest = {
        "detail_csv": str(detail_path),
        "aggregate_csv": str(aggregate_path),
        "latency_csv": str(latency_path),
        "uses_final_test_split": False,
        "selection_split": "router_train",
        "evaluation_split": "router_val",
        "chronological_order_preserved": True,
        "args": vars(args),
    }
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    print(f"Saved: {detail_path}")
    print(f"Saved: {aggregate_path}")
    print(f"Saved: {latency_path}")
    print(f"Saved: {manifest_path}")
    return manifest


def run_latency_device_benchmarks(
    args: argparse.Namespace,
    loaders: Mapping[str, DataLoader],
    scaler,
    groups,
) -> list[dict]:
    rows = []
    if not args.latency_devices:
        return rows
    seed = args.seeds[0]
    condition = RobustnessCondition("latency_clean")
    for requested_device in args.latency_devices:
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            rows.append(
                {
                    "seed": seed,
                    "expert_subset": "latency_probe",
                    "condition_name": condition.condition_name,
                    "device": requested_device,
                    "num_experts": 0,
                    "num_val_windows": 0,
                    "elapsed_seconds": float("nan"),
                    "windows_per_second": float("nan"),
                    "latency_note": "CUDA unavailable; skipped",
                }
            )
            continue
        device = torch.device(requested_device)
        for group_index, specs in enumerate(groups, start=1):
            subset_name = f"group_{group_index}_{len(specs)}_experts"
            experts, expert_names, _, _ = build_selected_candidate_experts(
                checkpoint_dir=args.checkpoint_dir,
                device=device,
                scaler=scaler,
                specs=specs,
            )
            assert_experts_frozen(*experts)
            start_time = time.perf_counter()
            val_errors, _ = expert_error_matrix(
                experts,
                expert_names,
                loaders["router_val"],
                scaler,
                device,
                condition,
                seed,
                args.max_val_windows,
            )
            elapsed = time.perf_counter() - start_time
            rows.append(
                {
                    "seed": seed,
                    "expert_subset": subset_name,
                    "condition_name": condition.condition_name,
                    "device": requested_device,
                    "num_experts": len(expert_names),
                    "num_val_windows": int(val_errors.shape[0]),
                    "elapsed_seconds": elapsed,
                    "windows_per_second": float(val_errors.shape[0] / max(elapsed, 1e-12)),
                    "latency_note": "latency-only clean router_val benchmark",
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen-expert router robustness experiments on router_val.")
    parser.add_argument("--data-dir", default="datasets/ETTh1")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--output-dir", default="results/router_robustness")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", type=parse_csv_ints, default=DEFAULT_SEEDS)
    parser.add_argument("--train-fractions", type=parse_csv_floats, default=(1.0, 0.5, 0.25))
    parser.add_argument("--noise-stds", type=parse_csv_floats, default=(0.0, 0.02, 0.05))
    parser.add_argument("--missing-variable-fractions", type=parse_csv_floats, default=(0.0, 0.25))
    parser.add_argument("--masked-timestamp-fractions", type=parse_csv_floats, default=(0.0, 0.10))
    parser.add_argument("--temporal-shift-steps", type=parse_csv_ints, default=(0, 4))
    parser.add_argument("--max-k-values", type=parse_csv_ints, default=(1, 2, 3, 4, 5))
    parser.add_argument("--routing-temperatures", type=parse_csv_floats, default=(0.5, 1.0, 2.0))
    parser.add_argument("--latency-devices", type=parse_csv_strings, default=("cpu", "cuda"))
    parser.add_argument("--full-factorial", action="store_true")
    parser.add_argument("--max-expert-groups", type=int, default=None)
    parser.add_argument("--max-train-windows", type=int, default=None)
    parser.add_argument("--max-val-windows", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        if len(args.seeds) < 5:
            raise ValueError("Robustness testing requires at least 5 random seeds")
        conditions = make_conditions(
            args.train_fractions,
            args.noise_stds,
            args.missing_variable_fractions,
            args.masked_timestamp_fractions,
            args.temporal_shift_steps,
            args.max_k_values,
            args.routing_temperatures,
            args.full_factorial,
        )
        print(f"Dry run: {len(args.seeds)} seeds, {len(conditions)} conditions")
        print("Seeds:", args.seeds)
        print("Conditions:", [condition.condition_name for condition in conditions])
        print("Final test split will not be used.")
        return
    run_robustness(args)


if __name__ == "__main__":
    main()
