"""Paired significance tests for saved prediction-aware router results."""

import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from basicts.scaler import ZScoreScaler
from scripts.chronological_expert_training import (
    DEFAULT_INPUT_LEN,
    DEFAULT_NUM_FEATURES,
    DEFAULT_OUTPUT_LEN,
    PredictionAwareRouter,
    _assert_full_data_contract,
    _call_expert_model,
    _dataset_config_summary,
    _load_torch_checkpoint,
    _prepare_forecasting_batch,
    _prediction_aware_router_forward,
    assert_experts_frozen,
    build_selected_candidate_experts,
    load_full_chronological_data,
    load_prediction_aware_router_checkpoint,
    prepare_chronological_dataloaders,
    selected_router_model_groups,
)


def _paired_t_p_value(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("nan")
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    if std == 0.0:
        return 0.0 if mean != 0.0 else 1.0
    statistic = mean / (std / math.sqrt(values.size))
    try:
        from scipy import stats

        return float(2.0 * stats.t.sf(abs(statistic), df=values.size - 1))
    except Exception:
        return float(math.erfc(abs(statistic) / math.sqrt(2.0)))


def _bootstrap_ci(
    values: np.ndarray,
    seed: int,
    samples: int = 10000,
) -> Tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(samples, values.size))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _holm_adjust(p_values: Sequence[float]) -> Tuple[float, ...]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [float("nan")] * len(p_values)
    running_max = 0.0
    total = len(p_values)
    for rank, (index, p_value) in enumerate(indexed):
        if not math.isfinite(p_value):
            adjusted[index] = float("nan")
            continue
        candidate = min(1.0, (total - rank) * p_value)
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return tuple(adjusted)


def _window_mae(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    errors = torch.abs(prediction - target)
    mask = target_mask.to(errors.dtype)
    return (errors * mask).sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp_min(1.0)


def _group_name(count: int) -> str:
    return f"best_{count}_{'expert' if count == 1 else 'experts'}"


def _checkpoint_path(checkpoint_dir: Path, group_name: str) -> Path:
    return checkpoint_dir / f"best_router_{group_name}.pt"


def _collect_validation_weights(
    router: PredictionAwareRouter,
    experts: Sequence[torch.nn.Module],
    expert_names: Sequence[str],
    loader: Iterable[dict],
    device: torch.device,
    scaler,
) -> Tuple[Dict[str, float], str]:
    totals = {expert_name: 0.0 for expert_name in expert_names}
    counts = {expert_name: 0.0 for expert_name in expert_names}
    with torch.no_grad():
        for batch in loader:
            inputs, targets, targets_mask = _prepare_forecasting_batch(
                batch,
                device,
                scaler,
            )
            expert_predictions = torch.stack(
                [_call_expert_model(expert, inputs).detach() for expert in experts],
                dim=2,
            )
            for index, expert_name in enumerate(expert_names):
                mae = _window_mae(
                    expert_predictions[:, :, index, :],
                    targets,
                    targets_mask,
                )
                totals[expert_name] += float(mae.sum().item())
                counts[expert_name] += float(mae.numel())
    expert_mae = {
        expert_name: totals[expert_name] / counts[expert_name]
        for expert_name in expert_names
    }
    epsilon = 1e-6
    inverse = {
        expert_name: 1.0 / (expert_mae[expert_name] + epsilon)
        for expert_name in expert_names
    }
    inverse_total = sum(inverse.values())
    fixed_weights = {
        expert_name: inverse[expert_name] / inverse_total
        for expert_name in expert_names
    }
    best_expert = min(expert_names, key=lambda expert_name: expert_mae[expert_name])
    return fixed_weights, best_expert


def run_significance_tests(
    data_dir: Path = Path("datasets/ETTh1"),
    checkpoint_dir: Path = Path("checkpoints"),
    output_dir: Path = Path("results/router_summary"),
    batch_size: int = 512,
    device: str = "cpu",
    seed: int = 7,
) -> Tuple[dict, ...]:
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device)
    data_dir = Path(data_dir)
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_data = load_full_chronological_data(data_dir)
    _assert_full_data_contract(full_data, DEFAULT_NUM_FEATURES)
    loaders, scaler = prepare_chronological_dataloaders(
        full_data=full_data,
        scaler=ZScoreScaler(norm_each_channel=True, rescale=False),
        batch_size=batch_size,
        input_len=DEFAULT_INPUT_LEN,
        output_len=DEFAULT_OUTPUT_LEN,
    )

    rows = []
    per_window_rows = []
    for specs in selected_router_model_groups():
        if len(specs) < 2:
            continue
        group_name = _group_name(len(specs))
        checkpoint_path = _checkpoint_path(checkpoint_dir, group_name)
        if not checkpoint_path.exists():
            print(f"Skipping {group_name}; missing {checkpoint_path}")
            continue

        experts, expert_names, _, _ = build_selected_candidate_experts(
            checkpoint_dir=checkpoint_dir,
            device=device,
            scaler=scaler,
            specs=specs,
        )
        router_checkpoint = _load_torch_checkpoint(checkpoint_path, device)
        router_config = dict(router_checkpoint["router_config"])
        router_config.setdefault("num_experts", len(experts))
        router = PredictionAwareRouter(**router_config).to(device)
        load_prediction_aware_router_checkpoint(
            router,
            checkpoint_path,
            device=device,
        )
        router.eval()
        assert_experts_frozen(*experts)

        fixed_weights, validation_best_expert = _collect_validation_weights(
            router,
            experts,
            expert_names,
            loaders["router_val"],
            device,
            scaler,
        )
        validation_best_index = expert_names.index(validation_best_expert)
        method_errors = {
            expert_name: []
            for expert_name in expert_names
        }
        method_errors.update(
            {
                "Fixed equal average": [],
                "Fixed validation-based soft weights": [],
                "Validation-selected best expert": [],
                "Learned prediction-aware router": [],
            }
        )

        with torch.no_grad():
            for batch in loaders["test"]:
                inputs, targets, targets_mask = _prepare_forecasting_batch(
                    batch,
                    device,
                    scaler,
                )
                expert_predictions, _, _, _, router_prediction = (
                    _prediction_aware_router_forward(router, experts, inputs)
                )
                equal_prediction = expert_predictions.mean(dim=2)
                fixed_prediction = torch.zeros_like(equal_prediction)
                for index, expert_name in enumerate(expert_names):
                    prediction = expert_predictions[:, :, index, :]
                    method_errors[expert_name].extend(
                        _window_mae(prediction, targets, targets_mask).cpu().tolist()
                    )
                    fixed_prediction += fixed_weights[expert_name] * prediction
                method_errors["Fixed equal average"].extend(
                    _window_mae(equal_prediction, targets, targets_mask).cpu().tolist()
                )
                method_errors["Fixed validation-based soft weights"].extend(
                    _window_mae(fixed_prediction, targets, targets_mask).cpu().tolist()
                )
                method_errors["Validation-selected best expert"].extend(
                    _window_mae(
                        expert_predictions[:, :, validation_best_index, :],
                        targets,
                        targets_mask,
                    ).cpu().tolist()
                )
                method_errors["Learned prediction-aware router"].extend(
                    _window_mae(router_prediction, targets, targets_mask).cpu().tolist()
                )

        router_errors = np.asarray(
            method_errors["Learned prediction-aware router"],
            dtype=np.float64,
        )
        strongest_baseline = min(
            (
                method_name
                for method_name in method_errors
                if method_name != "Learned prediction-aware router"
            ),
            key=lambda method_name: float(np.mean(method_errors[method_name])),
        )

        group_rows = []
        for baseline_name, baseline_values in method_errors.items():
            if baseline_name == "Learned prediction-aware router":
                continue
            baseline_errors = np.asarray(baseline_values, dtype=np.float64)
            difference = baseline_errors - router_errors
            ci_low, ci_high = _bootstrap_ci(difference, seed=seed)
            group_rows.append(
                {
                    "model_group": group_name,
                    "model_count": len(specs),
                    "selected_models": " + ".join(expert_names),
                    "baseline_method": baseline_name,
                    "router_mae": float(router_errors.mean()),
                    "baseline_mae": float(baseline_errors.mean()),
                    "mae_improvement_baseline_minus_router": float(
                        difference.mean()
                    ),
                    "bootstrap_95_ci_low": ci_low,
                    "bootstrap_95_ci_high": ci_high,
                    "paired_t_p_value": _paired_t_p_value(difference),
                    "n_windows": int(router_errors.size),
                    "strongest_baseline": baseline_name == strongest_baseline,
                }
            )

        adjusted = _holm_adjust(
            [row["paired_t_p_value"] for row in group_rows]
        )
        for row, adjusted_p in zip(group_rows, adjusted):
            row["holm_adjusted_p_value"] = adjusted_p
            row["significant_at_0_05"] = bool(
                math.isfinite(adjusted_p)
                and adjusted_p < 0.05
                and row["bootstrap_95_ci_low"] > 0.0
            )
            rows.append(row)

        for index in range(router_errors.size):
            row = {"model_group": group_name, "window_index": index}
            for method_name, values in method_errors.items():
                row[f"{method_name}_mae"] = values[index]
            per_window_rows.append(row)

    csv_path = output_dir / "router_significance_tests.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    per_window_path = output_dir / "router_per_window_mae.csv"
    with per_window_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = sorted({key for row in per_window_rows for key in row})
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_window_rows)

    json_path = output_dir / "router_significance_tests.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2)

    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
    print(f"Saved: {per_window_path}")
    return tuple(rows)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="datasets/ETTh1")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--output-dir", default="results/router_summary")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    run_significance_tests(
        data_dir=Path(args.data_dir),
        checkpoint_dir=Path(args.checkpoint_dir),
        output_dir=Path(args.output_dir),
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
    )
