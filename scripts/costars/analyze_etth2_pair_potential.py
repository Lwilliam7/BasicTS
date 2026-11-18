"""Analyze clean ETTh2 router caches for fixed-pair routing potential."""

import argparse
import csv
import hashlib
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPECTED_EXPERTS = ("DLinear", "PatchTST", "iTransformer", "TimesNet", "ModernTCN")
EXPECTED_HASHES = {
    "router_train": "34e1b02882bbdbabb25636acf6dbd656699bf60e1ffdd91ecb032e7a3827d17b",
    "router_val": "ba814cfe6f6d8bc23ec8693757f0c40426b4365f828fcb0f5f15858d09795088",
    "scaler": "4a24d2d9e46d97c80f80da889563a0d2cb7bb7aabe174d9b94fdbb82ce7982de",
}
NEAR_TIE_THRESHOLDS = (0.001, 0.005, 0.01, 0.025, 0.05)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: object):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"Cannot JSON encode {type(value)!r}")


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def load_verified_cache(path: Path, split_role: str, report: Mapping[str, object]) -> dict:
    observed_hash = sha256_file(path)
    expected_hash = EXPECTED_HASHES[split_role]
    if observed_hash != expected_hash:
        raise ValueError(f"{split_role} cache hash mismatch: {observed_hash} != {expected_hash}")
    if report["cache_hashes"][split_role] != expected_hash:
        raise ValueError(f"{split_role} report hash mismatch")
    cache = torch.load(path, map_location="cpu", weights_only=False)
    validate_cache_schema(cache, split_role, report)
    return cache


def validate_cache_schema(cache: Mapping[str, object], split_role: str, report: Mapping[str, object]) -> None:
    if cache["split_role"] != split_role:
        raise ValueError(f"Expected split {split_role}, found {cache['split_role']}")
    if tuple(cache["expert_names"]) != EXPECTED_EXPERTS:
        raise ValueError("Expert ordering changed")
    if cache["scaler_hash"] != EXPECTED_HASHES["scaler"]:
        raise ValueError("Scaler hash mismatch")
    if cache["checkpoint_hashes"] != report["checkpoint_hashes"]:
        raise ValueError("Checkpoint hashes mismatch")
    expected_n = 2053 if split_role == "router_train" else 613
    expected_shapes = {
        "histories": (expected_n, 96, 7),
        "targets": (expected_n, 12, 7),
        "target_masks": (expected_n, 12, 7),
        "prediction_stack": (expected_n, 12, 7, 5),
        "error_matrix": (expected_n, 5),
        "mse_matrix": (expected_n, 5),
    }
    for key, shape in expected_shapes.items():
        if tuple(cache[key].shape) != shape:
            raise ValueError(f"{split_role} {key} shape mismatch: {tuple(cache[key].shape)} != {shape}")
        tensor = cache[key]
        if tensor.dtype != torch.bool and not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"{split_role} {key} contains nonfinite values")
    starts = cache["absolute_window_starts"]
    if int(starts.max()) >= 11520:
        raise ValueError("Locked-test index appears in cache")


def validate_cache_pair(train_cache: Mapping[str, object], val_cache: Mapping[str, object]) -> None:
    train_starts = set(train_cache["absolute_window_starts"].tolist())
    val_starts = set(val_cache["absolute_window_starts"].tolist())
    if train_starts.intersection(val_starts):
        raise ValueError("router_train and router_val source ranges overlap")
    for cache in (train_cache, val_cache):
        pred = cache["prediction_stack"]
        targets = cache["targets"]
        mask = cache["target_masks"]
        mae, mse = per_window_error(pred, targets, mask)
        if not torch.allclose(mae, cache["error_matrix"], atol=1e-6, rtol=1e-6):
            raise ValueError(f"{cache['split_role']} cached MAE does not reproduce")
        if not torch.allclose(mse, cache["mse_matrix"], atol=1e-6, rtol=1e-6):
            raise ValueError(f"{cache['split_role']} cached MSE does not reproduce")


def per_window_error(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if predictions.ndim == 3:
        errors = predictions - targets
        mask_f = mask.to(errors.dtype)
        denom = mask_f.sum(dim=(1, 2)).clamp_min(1.0)
        mae = (errors.abs() * mask_f).sum(dim=(1, 2)) / denom
        mse = (errors.square() * mask_f).sum(dim=(1, 2)) / denom
        return mae, mse
    errors = predictions - targets.unsqueeze(-1)
    mask_f = mask.unsqueeze(-1).to(errors.dtype)
    denom = mask_f.sum(dim=(1, 2)).clamp_min(1.0)
    mae = (errors.abs() * mask_f).sum(dim=(1, 2)) / denom
    mse = (errors.square() * mask_f).sum(dim=(1, 2)) / denom
    return mae, mse


def aggregate_from_per_window(mae: torch.Tensor, mse: torch.Tensor) -> dict:
    return {
        "mae": float(mae.mean().item()),
        "mse": float(mse.mean().item()),
        "mean_per_window_mae": float(mae.mean().item()),
        "std_per_window_mae": float(mae.std(unbiased=False).item()),
        "median_per_window_mae": float(mae.median().item()),
    }


def individual_rows(cache: Mapping[str, object]) -> Tuple[List[dict], dict]:
    errors = cache["error_matrix"]
    mses = cache["mse_matrix"]
    oracle = errors.min(dim=1).values
    best_idx = errors.argmin(dim=1)
    rows = []
    metrics_by_name = {}
    for index, name in enumerate(EXPECTED_EXPERTS):
        row = {
            "split": cache["split_role"],
            "expert": name,
            **aggregate_from_per_window(errors[:, index], mses[:, index]),
            "best_window_percentage": float((best_idx == index).to(torch.float32).mean().item() * 100.0),
            "average_regret_to_oracle_expert": float((errors[:, index] - oracle).mean().item()),
            "average_experts_used": 1.0,
        }
        rows.append(row)
        metrics_by_name[name] = row
    rows.sort(key=lambda row: row["mae"])
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows, metrics_by_name


def subset_prediction(cache: Mapping[str, object], indices: Sequence[int]) -> torch.Tensor:
    return cache["prediction_stack"][..., list(indices)].mean(dim=-1)


def subset_errors(cache: Mapping[str, object], indices: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    prediction = subset_prediction(cache, indices)
    return per_window_error(prediction, cache["targets"], cache["target_masks"])


def all_subset_metrics(cache: Mapping[str, object], size: int) -> Tuple[List[dict], torch.Tensor, torch.Tensor, List[Tuple[int, ...]]]:
    combos = list(itertools.combinations(range(len(EXPECTED_EXPERTS)), size))
    maes = []
    mses = []
    for combo in combos:
        mae, mse = subset_errors(cache, combo)
        maes.append(mae)
        mses.append(mse)
    mae_matrix = torch.stack(maes, dim=1)
    mse_matrix = torch.stack(mses, dim=1)
    oracle = mae_matrix.min(dim=1).values
    best_idx = mae_matrix.argmin(dim=1)
    rows = []
    for combo_index, combo in enumerate(combos):
        names = tuple(EXPECTED_EXPERTS[i] for i in combo)
        row = {
            "split": cache["split_role"],
            "subset": "+".join(names),
            "subset_size": size,
            "expert_a": names[0],
            "expert_b": names[1] if len(names) > 1 else "",
            **aggregate_from_per_window(mae_matrix[:, combo_index], mse_matrix[:, combo_index]),
            "average_regret_to_oracle_subset": float((mae_matrix[:, combo_index] - oracle).mean().item()),
            "best_window_percentage": float((best_idx == combo_index).to(torch.float32).mean().item() * 100.0),
            "average_experts_used": float(size),
        }
        rows.append(row)
    rows.sort(key=lambda row: row["mae"])
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows, mae_matrix, mse_matrix, combos


def complementarity_rows(cache: Mapping[str, object], expert_metrics: Mapping[str, dict]) -> List[dict]:
    rows = []
    for combo in itertools.combinations(range(len(EXPECTED_EXPERTS)), 2):
        pair_mae, _ = subset_errors(cache, combo)
        a, b = combo
        name_a, name_b = EXPECTED_EXPERTS[a], EXPECTED_EXPERTS[b]
        expert_a = cache["error_matrix"][:, a]
        expert_b = cache["error_matrix"][:, b]
        beats_both = pair_mae < torch.minimum(expert_a, expert_b)
        harmed = pair_mae > torch.maximum(expert_a, expert_b)
        improvement = torch.minimum(expert_a, expert_b) - pair_mae
        row = {
            "split": cache["split_role"],
            "pair": f"{name_a}+{name_b}",
            "expert_a": name_a,
            "expert_b": name_b,
            "pair_mae": float(pair_mae.mean().item()),
            "better_constituent_mae": min(expert_metrics[name_a]["mae"], expert_metrics[name_b]["mae"]),
            "worse_constituent_mae": max(expert_metrics[name_a]["mae"], expert_metrics[name_b]["mae"]),
            "gain_over_better": min(expert_metrics[name_a]["mae"], expert_metrics[name_b]["mae"]) - float(pair_mae.mean().item()),
            "gain_over_worse": max(expert_metrics[name_a]["mae"], expert_metrics[name_b]["mae"]) - float(pair_mae.mean().item()),
            "per_window_complementarity_rate": float(beats_both.to(torch.float32).mean().item() * 100.0),
            "per_window_harm_rate": float(harmed.to(torch.float32).mean().item() * 100.0),
            "average_conditional_improvement": float(improvement[beats_both].mean().item()) if bool(beats_both.any()) else 0.0,
        }
        rows.append(row)
    rows.sort(key=lambda row: row["pair_mae"])
    return rows


def simplex_projection(vector: torch.Tensor) -> torch.Tensor:
    sorted_values, _ = torch.sort(vector, descending=True)
    cssv = torch.cumsum(sorted_values, dim=0) - 1
    ind = torch.arange(1, vector.numel() + 1, dtype=vector.dtype, device=vector.device)
    cond = sorted_values - cssv / ind > 0
    rho = int(torch.nonzero(cond, as_tuple=False)[-1].item())
    theta = cssv[rho] / (rho + 1)
    return torch.clamp(vector - theta, min=0)


def flatten_xy(cache: Mapping[str, object]) -> Tuple[torch.Tensor, torch.Tensor]:
    stack = cache["prediction_stack"]
    target = cache["targets"]
    mask = cache["target_masks"]
    flat_stack = stack.reshape(-1, stack.shape[-1]).to(torch.float64)
    flat_target = target.reshape(-1).to(torch.float64)
    flat_mask = mask.reshape(-1).bool()
    return flat_stack[flat_mask], flat_target[flat_mask]


def fit_simplex_weights(train_cache: Mapping[str, object], iterations: int = 800) -> torch.Tensor:
    x, y = flatten_xy(train_cache)
    weights = torch.full((x.shape[1],), 1.0 / x.shape[1], dtype=torch.float64)
    lipschitz = float((x.square().sum(dim=1).mean() * 2.0).item())
    step = 1.0 / max(lipschitz, 1e-6)
    for _ in range(iterations):
        residual = x.matmul(weights) - y
        grad = 2.0 * x.t().matmul(residual) / x.shape[0]
        weights = simplex_projection(weights - step * grad)
    return weights.to(torch.float32)


def fit_ridge_weights(train_cache: Mapping[str, object], ridge_lambda: float = 1e-3) -> torch.Tensor:
    x, y = flatten_xy(train_cache)
    xtx = x.t().matmul(x)
    penalty = ridge_lambda * torch.eye(xtx.shape[0], dtype=x.dtype)
    xty = x.t().matmul(y)
    return torch.linalg.solve(xtx + penalty, xty).to(torch.float32)


def weighted_errors(cache: Mapping[str, object], weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    prediction = torch.tensordot(cache["prediction_stack"], weights.to(cache["prediction_stack"].dtype), dims=([-1], [0]))
    return per_window_error(prediction, cache["targets"], cache["target_masks"])


def matrix_csv_rows(matrix: np.ndarray, labels: Sequence[str], split: str, value_name: str) -> List[dict]:
    rows = []
    for i, left in enumerate(labels):
        row = {"split": split, "row": left}
        for j, right in enumerate(labels):
            row[right] = float(matrix[i, j])
        rows.append(row)
    return rows


def diversity_diagnostics(cache: Mapping[str, object], pair_error_matrix: torch.Tensor) -> dict:
    errors = cache["error_matrix"].numpy()
    error_corr = np.corrcoef(errors, rowvar=False)
    stack = cache["prediction_stack"]
    targets = cache["targets"]
    mask = cache["target_masks"].bool()
    disagreement = np.zeros((5, 5), dtype=float)
    residual_corr_inputs = []
    for expert_idx in range(5):
        residual = (stack[..., expert_idx] - targets)[mask].numpy()
        residual_corr_inputs.append(residual)
    residual_corr = np.corrcoef(np.stack(residual_corr_inputs, axis=1), rowvar=False)
    for i, j in itertools.product(range(5), repeat=2):
        disagreement[i, j] = float((stack[..., i] - stack[..., j]).abs().mean().item())
    winner = np.zeros((5, 5), dtype=float)
    for i, j in itertools.product(range(5), repeat=2):
        winner[i, j] = float((cache["error_matrix"][:, i] < cache["error_matrix"][:, j]).to(torch.float32).mean().item())
    expert_margin = torch.sort(cache["error_matrix"], dim=1).values[:, 1] - torch.sort(cache["error_matrix"], dim=1).values[:, 0]
    pair_margin = torch.sort(pair_error_matrix, dim=1).values[:, 1] - torch.sort(pair_error_matrix, dim=1).values[:, 0]
    return {
        "error_correlation": error_corr,
        "prediction_disagreement": disagreement,
        "residual_correlation": residual_corr,
        "winner_matrix": winner,
        "expert_margin": margin_summary(expert_margin, "expert", cache["split_role"]),
        "pair_margin": margin_summary(pair_margin, "pair", cache["split_role"]),
    }


def margin_summary(margins: torch.Tensor, label_type: str, split: str) -> dict:
    row = {
        "split": split,
        "label_type": label_type,
        "mean_margin": float(margins.mean().item()),
        "median_margin": float(margins.median().item()),
        "p25_margin": float(torch.quantile(margins, 0.25).item()),
        "p75_margin": float(torch.quantile(margins, 0.75).item()),
    }
    for threshold in NEAR_TIE_THRESHOLDS:
        row[f"near_tie_within_{threshold}"] = float((margins <= threshold).to(torch.float32).mean().item() * 100.0)
    return row


def oracle_distribution_rows(
    split: str,
    expert_errors: torch.Tensor,
    pair_errors: torch.Tensor,
    pair_combos: Sequence[Tuple[int, ...]],
) -> List[dict]:
    rows = []
    expert_best = expert_errors.argmin(dim=1)
    pair_best = pair_errors.argmin(dim=1)
    for i, name in enumerate(EXPECTED_EXPERTS):
        rows.append({
            "split": split,
            "oracle_type": "expert",
            "selection": name,
            "count": int((expert_best == i).sum().item()),
            "percentage": float((expert_best == i).to(torch.float32).mean().item() * 100.0),
        })
    for i, combo in enumerate(pair_combos):
        pair = "+".join(EXPECTED_EXPERTS[j] for j in combo)
        rows.append({
            "split": split,
            "oracle_type": "pair",
            "selection": pair,
            "count": int((pair_best == i).sum().item()),
            "percentage": float((pair_best == i).to(torch.float32).mean().item() * 100.0),
        })
    return rows


def ensemble_baselines(
    train_cache: Mapping[str, object],
    val_cache: Mapping[str, object],
    train_individual: Mapping[str, dict],
    best_train_pair: str,
    best_train_expert: str,
) -> Tuple[List[dict], dict]:
    train_maes = torch.tensor([train_individual[name]["mae"] for name in EXPECTED_EXPERTS])
    inverse_weights = (1.0 / train_maes).to(torch.float32)
    inverse_weights = inverse_weights / inverse_weights.sum()
    simplex_weights = fit_simplex_weights(train_cache)
    ridge_weights = fit_ridge_weights(train_cache)
    weights = {
        "inverse_training_mae_average": inverse_weights,
        "nonnegative_simplex_linear_average": simplex_weights,
        "ridge_linear_stacker": ridge_weights,
    }
    rows = []
    for method, weight in weights.items():
        for split, cache in (("router_train", train_cache), ("router_val", val_cache)):
            mae, mse = weighted_errors(cache, weight)
            rows.append({
                "split": split,
                "method": method,
                **aggregate_from_per_window(mae, mse),
                "average_experts_used": 5.0,
                "selection_source": "router-train fitted",
                "weights": json.dumps({name: float(weight[i]) for i, name in enumerate(EXPECTED_EXPERTS)}),
            })
    metadata = {
        "best_fixed_pair_train_selected": best_train_pair,
        "best_fixed_expert_train_selected": best_train_expert,
        "weights": {
            method: {name: float(weight[i]) for i, name in enumerate(EXPECTED_EXPERTS)}
            for method, weight in weights.items()
        },
    }
    return rows, metadata


def same_cache_validation_table(
    val_cache: Mapping[str, object],
    val_individual_rows: Sequence[dict],
    val_pair_rows: Sequence[dict],
    best_train_expert: str,
    best_train_pair: str,
    ensemble_rows: Sequence[dict],
    oracle_expert_mae: float,
    oracle_expert_mse: float,
    oracle_pair_mae: float,
    oracle_pair_mse: float,
) -> List[dict]:
    rows = []
    oracle_pair_for_regret = oracle_pair_mae
    for row in val_individual_rows:
        rows.append({
            "method": row["expert"],
            "mae": row["mae"],
            "mse": row["mse"],
            "average_experts_used": 1.0,
            "regret_to_oracle_pair": row["mae"] - oracle_pair_for_regret,
            "selection_source": "router-train selected" if row["expert"] == best_train_expert else "fixed",
        })
    for row in val_pair_rows:
        rows.append({
            "method": row["pair"],
            "mae": row["mae"],
            "mse": row["mse"],
            "average_experts_used": 2.0,
            "regret_to_oracle_pair": row["mae"] - oracle_pair_for_regret,
            "selection_source": "router-train selected" if row["pair"] == best_train_pair else "fixed",
        })
    for row in ensemble_rows:
        if row["split"] == "router_val":
            rows.append({
                "method": row["method"],
                "mae": row["mae"],
                "mse": row["mse"],
                "average_experts_used": row["average_experts_used"],
                "regret_to_oracle_pair": row["mae"] - oracle_pair_for_regret,
                "selection_source": row["selection_source"],
            })
    rows.append({
        "method": "per_window_oracle_expert",
        "mae": oracle_expert_mae,
        "mse": oracle_expert_mse,
        "average_experts_used": 1.0,
        "regret_to_oracle_pair": oracle_expert_mae - oracle_pair_for_regret,
        "selection_source": "oracle diagnostic",
    })
    rows.append({
        "method": "per_window_oracle_pair",
        "mae": oracle_pair_mae,
        "mse": oracle_pair_mse,
        "average_experts_used": 2.0,
        "regret_to_oracle_pair": 0.0,
        "selection_source": "oracle diagnostic",
    })
    rows.sort(key=lambda row: float(row["mae"]))
    return rows


def analyze(args: argparse.Namespace) -> dict:
    report = json.loads(Path(args.validation_report).read_text(encoding="utf-8"))
    summary = json.loads(Path(args.cache_build_summary).read_text(encoding="utf-8"))
    if report["cache_hashes"] != summary["cache_hashes"]:
        raise ValueError("Cache validation report and build summary disagree")
    train_cache = load_verified_cache(Path(args.router_train_cache), "router_train", report)
    val_cache = load_verified_cache(Path(args.router_val_cache), "router_val", report)
    validate_cache_pair(train_cache, val_cache)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ind_rows, train_ind_metrics = individual_rows(train_cache)
    val_ind_rows, val_ind_metrics = individual_rows(val_cache)
    best_train_expert = train_ind_rows[0]["expert"]
    val_best_expert = val_ind_rows[0]["expert"]
    for row in train_ind_rows:
        row["training_selected_best_fixed_expert"] = row["expert"] == best_train_expert
        row["validation_best_expert_diagnostic"] = row["expert"] == val_best_expert
    for row in val_ind_rows:
        row["training_selected_best_fixed_expert"] = row["expert"] == best_train_expert
        row["validation_best_expert_diagnostic"] = row["expert"] == val_best_expert

    train_pair_rows, train_pair_errors, train_pair_mses, pair_combos = all_subset_metrics(train_cache, 2)
    val_pair_rows, val_pair_errors, val_pair_mses, _ = all_subset_metrics(val_cache, 2)
    best_train_pair = train_pair_rows[0]["subset"]
    val_best_pair = val_pair_rows[0]["subset"]
    for row in train_pair_rows:
        row["pair"] = row["subset"]
        row["training_selected_best_fixed_pair"] = row["pair"] == best_train_pair
    for row in val_pair_rows:
        row["pair"] = row["subset"]
        row["training_selected_best_fixed_pair"] = row["pair"] == best_train_pair
        row["validation_best_pair_diagnostic"] = row["pair"] == val_best_pair

    train_triplet_rows, train_triplet_errors, train_triplet_mses, triplet_combos = all_subset_metrics(train_cache, 3)
    val_triplet_rows, val_triplet_errors, val_triplet_mses, _ = all_subset_metrics(val_cache, 3)
    train_four_rows, train_four_errors, train_four_mses, four_combos = all_subset_metrics(train_cache, 4)
    val_four_rows, val_four_errors, val_four_mses, _ = all_subset_metrics(val_cache, 4)
    train_five_rows, train_five_errors, train_five_mses, five_combos = all_subset_metrics(train_cache, 5)
    val_five_rows, val_five_errors, val_five_mses, _ = all_subset_metrics(val_cache, 5)

    comp_rows = complementarity_rows(train_cache, train_ind_metrics) + complementarity_rows(val_cache, val_ind_metrics)
    ensemble_rows, ensemble_metadata = ensemble_baselines(
        train_cache,
        val_cache,
        train_ind_metrics,
        best_train_pair,
        best_train_expert,
    )
    for row in train_triplet_rows + val_triplet_rows + train_four_rows + val_four_rows + train_five_rows + val_five_rows:
        ensemble_rows.append({
            "split": row["split"],
            "method": row["subset"],
            "mae": row["mae"],
            "mse": row["mse"],
            "average_experts_used": row["average_experts_used"],
            "selection_source": "fixed",
            "weights": "",
        })

    train_div = diversity_diagnostics(train_cache, train_pair_errors)
    val_div = diversity_diagnostics(val_cache, val_pair_errors)

    oracle_pair_val = val_pair_errors.min(dim=1).values
    oracle_pair_train = train_pair_errors.min(dim=1).values
    oracle_expert_val = val_cache["error_matrix"].min(dim=1).values
    oracle_expert_train = train_cache["error_matrix"].min(dim=1).values
    oracle_pair_val_mse = val_pair_mses.gather(1, val_pair_errors.argmin(dim=1).view(-1, 1)).squeeze(1)
    oracle_expert_val_mse = val_cache["mse_matrix"].gather(1, val_cache["error_matrix"].argmin(dim=1).view(-1, 1)).squeeze(1)
    oracle_triplet_val = val_triplet_errors.min(dim=1).values
    oracle_triplet_train = train_triplet_errors.min(dim=1).values
    fixed_pair_index = [i for i, combo in enumerate(pair_combos) if "+".join(EXPECTED_EXPERTS[j] for j in combo) == best_train_pair][0]
    fixed_pair_val_errors = val_pair_errors[:, fixed_pair_index]
    switch_improvement = fixed_pair_val_errors - oracle_pair_val
    switch_rows = []
    for threshold in NEAR_TIE_THRESHOLDS:
        useful = switch_improvement >= threshold
        switch_rows.append({
            "split": "router_val",
            "fixed_pair": best_train_pair,
            "improvement_margin": threshold,
            "useful_switch_percentage": float(useful.to(torch.float32).mean().item() * 100.0),
            "average_improvement_on_useful_windows": float(switch_improvement[useful].mean().item()) if bool(useful.any()) else 0.0,
            "total_improvement_on_useful_windows": float(switch_improvement[useful].sum().item()) if bool(useful.any()) else 0.0,
        })
    switch_rows.append({
        "split": "router_val",
        "fixed_pair": best_train_pair,
        "improvement_margin": 0.0,
        "useful_switch_percentage": float((switch_improvement > 0).to(torch.float32).mean().item() * 100.0),
        "average_improvement_on_useful_windows": float(switch_improvement[switch_improvement > 0].mean().item()),
        "total_improvement_on_useful_windows": float(switch_improvement[switch_improvement > 0].sum().item()),
    })

    oracle_rows = (
        oracle_distribution_rows("router_train", train_cache["error_matrix"], train_pair_errors, pair_combos)
        + oracle_distribution_rows("router_val", val_cache["error_matrix"], val_pair_errors, pair_combos)
    )
    same_cache_rows = same_cache_validation_table(
        val_cache,
        val_ind_rows,
        val_pair_rows,
        best_train_expert,
        best_train_pair,
        ensemble_rows,
        float(oracle_expert_val.mean().item()),
        float(oracle_expert_val_mse.mean().item()),
        float(oracle_pair_val.mean().item()),
        float(oracle_pair_val_mse.mean().item()),
    )

    write_outputs(
        output_dir,
        train_ind_rows,
        val_ind_rows,
        train_pair_rows,
        val_pair_rows,
        train_triplet_rows,
        val_triplet_rows,
        train_four_rows,
        val_four_rows,
        train_five_rows,
        val_five_rows,
        ensemble_rows,
        comp_rows,
        train_div,
        val_div,
        oracle_rows,
        switch_rows,
        same_cache_rows,
    )

    fixed_pair_val_row = next(row for row in val_pair_rows if row["pair"] == best_train_pair)
    fixed_pair_train_row = train_pair_rows[0]
    fixed_expert_val_row = next(row for row in val_ind_rows if row["expert"] == best_train_expert)
    fixed_pair_constituents = best_train_pair.split("+")
    val_constituents = [next(row for row in val_ind_rows if row["expert"] == name) for name in fixed_pair_constituents]
    summary_data = {
        "input_artifacts": {
            "router_train_cache": args.router_train_cache,
            "router_val_cache": args.router_val_cache,
            "cache_validation_report": args.validation_report,
            "cache_build_summary": args.cache_build_summary,
        },
        "cache_hashes": report["cache_hashes"],
        "scaler_hash": report["scaler_hash"],
        "expert_order": list(EXPECTED_EXPERTS),
        "training_selected_best_fixed_expert": best_train_expert,
        "training_selected_best_fixed_pair": best_train_pair,
        "validation_diagnostic_best_expert": val_best_expert,
        "validation_diagnostic_best_pair": val_best_pair,
        "fixed_pair_train_mae": fixed_pair_train_row["mae"],
        "fixed_pair_val_mae": fixed_pair_val_row["mae"],
        "fixed_pair_val_mse": fixed_pair_val_row["mse"],
        "fixed_pair_beats_both_constituents_on_validation": fixed_pair_val_row["mae"] < min(row["mae"] for row in val_constituents),
        "validation_constituent_maes": {row["expert"]: row["mae"] for row in val_constituents},
        "oracle_diagnostics": {
            "router_train_oracle_expert_mae": float(oracle_expert_train.mean().item()),
            "router_train_oracle_pair_mae": float(oracle_pair_train.mean().item()),
            "router_train_oracle_triplet_mae": float(oracle_triplet_train.mean().item()),
            "router_val_oracle_expert_mae": float(oracle_expert_val.mean().item()),
            "router_val_oracle_pair_mae": float(oracle_pair_val.mean().item()),
            "router_val_oracle_triplet_mae": float(oracle_triplet_val.mean().item()),
            "validation_fixed_pair_to_oracle_pair_improvement": float((fixed_pair_val_errors - oracle_pair_val).mean().item()),
            "validation_fixed_expert_to_oracle_expert_improvement": float((torch.tensor(fixed_expert_val_row["mae"]) - oracle_expert_val.mean()).item()),
            "maximum_recoverable_total_window_mae_improvement": float((fixed_pair_val_errors - oracle_pair_val).sum().item()),
        },
        "routing_margin_diagnostics": {
            "router_train_expert": train_div["expert_margin"],
            "router_train_pair": train_div["pair_margin"],
            "router_val_expert": val_div["expert_margin"],
            "router_val_pair": val_div["pair_margin"],
        },
        "switch_opportunity": switch_rows,
        "fitted_train_only_weights": ensemble_metadata["weights"],
        "leakage_assertions": {
            "used_only_allowed_input_artifacts": True,
            "test_arrays_loaded": False,
            "test_cache_created": False,
            "validation_not_used_for_fixed_pair_selection": True,
            "oracle_diagnostics_not_deployable_inputs": True,
            "no_selector_router_gate_or_expert_training": True,
        },
        "working_tree_status": subprocess.check_output(
            ["git", "status", "--short", "--branch", "--untracked-files=all"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines(),
    }
    (output_dir / "pair_potential_summary.json").write_text(
        json.dumps(summary_data, indent=2, default=_json_default),
        encoding="utf-8",
    )
    write_markdown_report(output_dir / "pair_potential_report.md", summary_data, same_cache_rows)
    return summary_data


def write_outputs(
    output_dir: Path,
    train_ind_rows: Sequence[dict],
    val_ind_rows: Sequence[dict],
    train_pair_rows: Sequence[dict],
    val_pair_rows: Sequence[dict],
    train_triplet_rows: Sequence[dict],
    val_triplet_rows: Sequence[dict],
    train_four_rows: Sequence[dict],
    val_four_rows: Sequence[dict],
    train_five_rows: Sequence[dict],
    val_five_rows: Sequence[dict],
    ensemble_rows: Sequence[dict],
    comp_rows: Sequence[dict],
    train_div: Mapping[str, object],
    val_div: Mapping[str, object],
    oracle_rows: Sequence[dict],
    switch_rows: Sequence[dict],
    same_cache_rows: Sequence[dict],
) -> None:
    individual_fields = ("split", "rank", "expert", "mae", "mse", "mean_per_window_mae", "std_per_window_mae", "median_per_window_mae", "best_window_percentage", "average_regret_to_oracle_expert", "average_experts_used", "training_selected_best_fixed_expert", "validation_best_expert_diagnostic")
    write_csv(output_dir / "individual_experts.csv", [*train_ind_rows, *val_ind_rows], individual_fields)
    pair_fields = ("split", "rank", "pair", "expert_a", "expert_b", "mae", "mse", "mean_per_window_mae", "std_per_window_mae", "median_per_window_mae", "average_regret_to_oracle_subset", "best_window_percentage", "average_experts_used", "training_selected_best_fixed_pair", "validation_best_pair_diagnostic")
    write_csv(output_dir / "fixed_pairs.csv", [*train_pair_rows, *val_pair_rows], pair_fields)
    subset_fields = ("split", "rank", "subset", "subset_size", "mae", "mse", "mean_per_window_mae", "std_per_window_mae", "median_per_window_mae", "average_regret_to_oracle_subset", "best_window_percentage", "average_experts_used")
    write_csv(output_dir / "fixed_triplets.csv", [*train_triplet_rows, *val_triplet_rows], subset_fields)
    ensemble_fields = ("split", "method", "mae", "mse", "average_experts_used", "selection_source", "weights")
    write_csv(output_dir / "ensemble_baselines.csv", ensemble_rows, ensemble_fields)
    comp_fields = ("split", "pair", "expert_a", "expert_b", "pair_mae", "better_constituent_mae", "worse_constituent_mae", "gain_over_better", "gain_over_worse", "per_window_complementarity_rate", "per_window_harm_rate", "average_conditional_improvement")
    write_csv(output_dir / "complementarity_by_pair.csv", comp_rows, comp_fields)
    labels = EXPECTED_EXPERTS
    matrix_fields = ("split", "row", *labels)
    write_csv(output_dir / "expert_error_correlation.csv", matrix_csv_rows(train_div["error_correlation"], labels, "router_train", "corr") + matrix_csv_rows(val_div["error_correlation"], labels, "router_val", "corr"), matrix_fields)
    write_csv(output_dir / "prediction_disagreement.csv", matrix_csv_rows(train_div["prediction_disagreement"], labels, "router_train", "disagreement") + matrix_csv_rows(val_div["prediction_disagreement"], labels, "router_val", "disagreement"), matrix_fields)
    write_csv(output_dir / "residual_correlation.csv", matrix_csv_rows(train_div["residual_correlation"], labels, "router_train", "corr") + matrix_csv_rows(val_div["residual_correlation"], labels, "router_val", "corr"), matrix_fields)
    write_csv(output_dir / "pairwise_winner_matrix.csv", matrix_csv_rows(train_div["winner_matrix"], labels, "router_train", "winner") + matrix_csv_rows(val_div["winner_matrix"], labels, "router_val", "winner"), matrix_fields)
    oracle_fields = ("split", "oracle_type", "selection", "count", "percentage")
    write_csv(output_dir / "oracle_selection_distribution.csv", oracle_rows, oracle_fields)
    margin_fields = ("split", "label_type", "mean_margin", "median_margin", "p25_margin", "p75_margin", *(f"near_tie_within_{threshold}" for threshold in NEAR_TIE_THRESHOLDS))
    write_csv(output_dir / "routing_margin_diagnostics.csv", [train_div["expert_margin"], train_div["pair_margin"], val_div["expert_margin"], val_div["pair_margin"]], margin_fields)
    switch_fields = ("split", "fixed_pair", "improvement_margin", "useful_switch_percentage", "average_improvement_on_useful_windows", "total_improvement_on_useful_windows")
    write_csv(output_dir / "switch_opportunity.csv", switch_rows, switch_fields)
    same_fields = ("method", "mae", "mse", "average_experts_used", "regret_to_oracle_pair", "selection_source")
    write_csv(output_dir / "same_cache_validation_comparison.csv", same_cache_rows, same_fields)


def write_markdown_report(path: Path, summary: Mapping[str, object], same_cache_rows: Sequence[dict]) -> None:
    top_rows = same_cache_rows[:10]
    lines = [
        "# ETTh2 Pair Potential Report",
        "",
        "## Decision Answers",
        f"1. Router-training-selected best fixed expert: `{summary['training_selected_best_fixed_expert']}`.",
        f"2. Router-training-selected best fixed pair: `{summary['training_selected_best_fixed_pair']}`.",
        f"3. Validation MAE for that pair: `{summary['fixed_pair_val_mae']:.6f}`.",
        f"4. Pair beats both validation constituents: `{summary['fixed_pair_beats_both_constituents_on_validation']}`.",
        f"5. Fixed-pair to oracle-pair validation improvement: `{summary['oracle_diagnostics']['validation_fixed_pair_to_oracle_pair_improvement']:.6f}` MAE.",
        f"6. Useful switch rate at 0.01 margin: `{next(row['useful_switch_percentage'] for row in summary['switch_opportunity'] if row['improvement_margin'] == 0.01):.2f}%`.",
        f"7. Validation pair mean margin: `{summary['routing_margin_diagnostics']['router_val_pair']['mean_margin']:.6f}`, median `{summary['routing_margin_diagnostics']['router_val_pair']['median_margin']:.6f}`.",
        "",
        "## Top Validation Methods",
        "",
        "| method | MAE | MSE | avg experts | source |",
        "|---|---:|---:|---:|---|",
    ]
    for row in top_rows:
        mse = row["mse"] if row["mse"] != "" else ""
        mse_text = f"{float(mse):.6f}" if mse != "" else ""
        lines.append(
            f"| {row['method']} | {float(row['mae']):.6f} | {mse_text} | "
            f"{float(row['average_experts_used']):.2f} | {row['selection_source']} |"
        )
    lines.extend([
        "",
        "## Leakage",
        "",
        "Only the two clean router caches and cache reports were loaded. No ETTh2 test arrays were read and no test cache was created.",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router-train-cache", default="cache/costarts_fresh/ETTh2_96_12/router_train_cache.pt")
    parser.add_argument("--router-val-cache", default="cache/costarts_fresh/ETTh2_96_12/router_val_cache.pt")
    parser.add_argument("--validation-report", default="cache/costarts_fresh/ETTh2_96_12/cache_validation_report.json")
    parser.add_argument("--cache-build-summary", default="results/router_summary/costarts_fresh/ETTh2_96_12/cache_build_summary.json")
    parser.add_argument("--output-dir", default="results/router_summary/costarts_fresh/ETTh2_96_12/pair_potential")
    return parser.parse_args(argv)


def main() -> None:
    analyze(parse_args())


if __name__ == "__main__":
    main()
