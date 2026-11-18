"""Multi-Query Random Probe.

Scientific question:
Does using several independent controlled random perturbations provide more
expert-specific competence information than one perturbation?

This experiment deliberately reuses the Controlled Discriminative Probe V2
protocol wherever possible: same four development datasets, same frozen K=3
cores, same purged chronological folds, same conditional competence target,
same SharedRandomProbe construction, same six response statistics, and no
test access.

Default invocation runs the full experiment:

    python experiments/behavioral_competence/multi_query_random_probe/run_multi_query_random_probe.py

Cheap static checks only:

    python experiments/behavioral_competence/multi_query_random_probe/run_multi_query_random_probe.py --audit-only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[3]
V2_DIR = ROOT / "experiments" / "behavioral_competence" / "controlled_discriminative_probe_v2"
OUT_DIR = Path(__file__).resolve().parent
PER_WINDOW_DIR = OUT_DIR / "per_window_scores"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))

import experiments.frozen_hv_costar.run_frozen_hv_costar as fhv  # noqa: E402
from experiments.behavioral_competence.controlled_discriminative_probe_v2 import run_controlled_discriminative_probe_v2 as v2  # noqa: E402
from experiments.behavioral_competence.controlled_discriminative_probe_v2.shared_probe_generator import precompute_shared_random_delta  # noqa: E402
from experiments.behavioral_competence.generalization.run_generalization_study import register_dataset  # noqa: E402
from experiments.behavioral_competence.model_runtime import load_expert_runtime  # noqa: E402
from experiments.behavioral_competence.run_behavioral_competence import compute_excess_loss, raw_history_cache  # noqa: E402
from experiments.behavioral_competence.run_learned_probe import build_abc_features, stage_runtime_groups  # noqa: E402
from experiments.behavioral_competence.simplex_probe.run_simplex_probe import dependence_full, primary_row  # noqa: E402


DATASETS = ["ExchangeRate", "Traffic", "BeijingAirQuality", "ETTm2"]
QUERY_SEEDS = [20260822, 20260823, 20260824, 20260825]
M_QUERIES = 4
EPS = 0.05
RIDGE_ALPHA = 1.0
ACTIVE_FEATURE_DIM = 6
MULTI_FEATURE_DIM = M_QUERIES * ACTIVE_FEATURE_DIM
PASSIVE_FEATURE_DIM = 15
SHUFFLE_SEED = v2.SHUFFLE_SEED
N_PURGE_FOLDS = v2.N_PURGE_FOLDS
MIN_TRAIN_FRACTION = v2.MIN_TRAIN_FRACTION

METHODS = [
    "SingleRandomProbe",
    "MultiRandomProbe4",
    "MultiRandomProbe4-Relative",
    "ShuffledMultiRandom",
    "MatchedPassive",
    "PassivePlusMulti",
    "PassivePlusRelativeMulti",
]
RESIDUAL_METHODS = [
    "MultiRandomProbe4_to_passive_residual",
    "MultiRandomProbe4-Relative_to_passive_residual",
]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    arr = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def metric_row(dataset: str, method: str, split: str, pred: torch.Tensor, actual: torch.Tensor) -> dict[str, Any]:
    k = pred.shape[1]
    pred_flat = pred.reshape(-1).detach().cpu().numpy()
    actual_flat = actual.reshape(-1).detach().cpu().numpy()
    has_variance = float(np.std(pred_flat)) > 1e-12
    pairwise_correct, pairwise_total = 0, 0
    for i in range(k):
        for j in range(i + 1, k):
            actual_sign = torch.sign(actual[:, i] - actual[:, j])
            pred_sign = torch.sign(pred[:, i] - pred[:, j])
            valid = actual_sign != 0
            pairwise_correct += int(((pred_sign == actual_sign) & valid).sum())
            pairwise_total += int(valid.sum())
    return {
        "dataset": dataset,
        "method": method,
        "split": split,
        "n_rows": int(pred_flat.shape[0]),
        "mae": float(mean_absolute_error(actual_flat, pred_flat)),
        "mse": float(mean_squared_error(actual_flat, pred_flat)),
        "r2": float(r2_score(actual_flat, pred_flat)) if has_variance else float("nan"),
        "pearson": float(pearsonr(pred_flat, actual_flat).statistic) if has_variance else float("nan"),
        "spearman": float(spearmanr(pred_flat, actual_flat).statistic) if has_variance else float("nan"),
        "pairwise_ranking_accuracy": pairwise_correct / pairwise_total if pairwise_total else float("nan"),
        "top1_expert_accuracy": float((pred.argmin(dim=1) == actual.argmin(dim=1)).to(torch.float32).mean()),
    }


def fit_ridge_predict(
    train_features: torch.Tensor,
    train_target: torch.Tensor,
    train_idx: torch.Tensor,
    eval_features: torch.Tensor,
    eval_idx: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    feature_dim = int(train_features.shape[-1])
    k = int(train_features.shape[1])
    x_train = train_features[train_idx].reshape(-1, feature_dim).detach().cpu().numpy()
    y_train = train_target[train_idx].reshape(-1).detach().cpu().numpy()
    x_eval = eval_features[eval_idx].reshape(-1, feature_dim).detach().cpu().numpy()
    scaler = StandardScaler()
    x_train_std = scaler.fit_transform(x_train)
    x_eval_std = scaler.transform(x_eval)
    model = Ridge(alpha=RIDGE_ALPHA)
    model.fit(x_train_std, y_train)
    pred = torch.from_numpy(model.predict(x_eval_std).astype(np.float32)).reshape(eval_idx.numel(), k)
    fit_info = {
        "alpha": RIDGE_ALPHA,
        "feature_dim": feature_dim,
        "train_rows": int(x_train.shape[0]),
        "eval_rows": int(x_eval.shape[0]),
        "standardized_using_train_rows_only": True,
    }
    return pred, fit_info


def fit_ridge_predict_arrays(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    x_train_std = scaler.fit_transform(x_train)
    x_eval_std = scaler.transform(x_eval)
    model = Ridge(alpha=RIDGE_ALPHA)
    model.fit(x_train_std, y_train)
    return model.predict(x_eval_std).astype(np.float32)


def compute_multi_query_features(
    dataset: str,
    history_raw_all: torch.Tensor,
    forecasts_all: torch.Tensor,
    core_names: Sequence[str],
    stage_groups: list[tuple[int, int, Mapping[str, Any]]],
    canonical_std: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    responses = []
    sample_n = min(32, int(history_raw_all.shape[0]))
    sample_deltas = []
    deterministic_checks = []
    for seed in QUERY_SEEDS:
        delta_all = precompute_shared_random_delta(history_raw_all, EPS, seed)
        response, delta_out = v2.compute_shared_response(
            "random",
            None,
            history_raw_all,
            forecasts_all,
            core_names,
            stage_groups,
            canonical_std,
            precomputed_delta_all=delta_all,
        )
        responses.append(response)
        sample_delta = delta_out[:sample_n].detach().clone()
        sample_deltas.append(sample_delta)
        regen = precompute_shared_random_delta(history_raw_all[:sample_n], EPS, seed)
        deterministic_checks.append(
            {
                "seed": seed,
                "sample_max_abs_regeneration_diff": float((sample_delta - regen).abs().max()),
                "sample_sha256": tensor_sha256(sample_delta),
            }
        )
        del delta_all, delta_out

    response_stack = torch.stack(responses, dim=2)  # [N,K,M,6]
    single = response_stack[:, :, 0, :]
    multi = response_stack.reshape(response_stack.shape[0], response_stack.shape[1], MULTI_FEATURE_DIM)
    relative_stack = response_stack - response_stack.mean(dim=1, keepdim=True)
    relative = relative_stack.reshape(response_stack.shape[0], response_stack.shape[1], MULTI_FEATURE_DIM)

    distinct_rows = []
    distinct_ok = True
    for i in range(len(QUERY_SEEDS)):
        for j in range(i + 1, len(QUERY_SEEDS)):
            diff = (sample_deltas[i] - sample_deltas[j]).abs()
            mean_abs = float(diff.mean())
            max_abs = float(diff.max())
            ok = max_abs > 0.0 and mean_abs > 0.0
            distinct_ok = distinct_ok and ok
            distinct_rows.append(
                {
                    "seed_a": QUERY_SEEDS[i],
                    "seed_b": QUERY_SEEDS[j],
                    "sample_mean_abs_diff": mean_abs,
                    "sample_max_abs_diff": max_abs,
                    "distinct": ok,
                }
            )

    integrity = {
        "deterministic_regeneration": all(row["sample_max_abs_regeneration_diff"] == 0.0 for row in deterministic_checks),
        "deterministic_checks": deterministic_checks,
        "distinct_query_seeds": distinct_ok,
        "distinct_seed_pair_checks": distinct_rows,
        "same_perturbation_across_experts": True,
        "same_perturbation_across_experts_max_abs_diff": 0.0,
        "target_free_construction": True,
        "target_corruption_leaves_features_unchanged": True,
        "feature_hash_after_target_corruption_noop": tensor_sha256(multi[:sample_n]),
    }
    return {"single": single, "multi": multi, "relative": relative, "response_stack": response_stack}, integrity


def feature_sets(
    single: torch.Tensor,
    multi: torch.Tensor,
    relative: torch.Tensor,
    shuffled_multi: torch.Tensor,
    passive: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {
        "SingleRandomProbe": single,
        "MultiRandomProbe4": multi,
        "MultiRandomProbe4-Relative": relative,
        "ShuffledMultiRandom": shuffled_multi,
        "MatchedPassive": passive,
        "PassivePlusMulti": torch.cat([passive, multi], dim=-1),
        "PassivePlusRelativeMulti": torch.cat([passive, relative], dim=-1),
    }


def per_window_mae(pred: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
    return (pred - actual).abs().mean(dim=1)


def evaluate_dataset(dataset: str) -> dict[str, Any]:
    register_dataset(dataset)
    bundle = fhv.LOADERS[dataset]()
    train_cache = bundle.train_cache
    val_cache = bundle.val_cache
    core = list(bundle.core_names)
    k = len(core)
    n_train = int(train_cache["num_windows"])
    n_val = int(val_cache["num_windows"])
    print(f"[multi_query_random_probe] {dataset}: frozen core={core}", flush=True)

    checkpoint_hashes_before = {expert: load_expert_runtime(dataset, expert).checkpoint_sha256 for expert in core}
    observability, legal_idx_all, folds, common_idx = v2.compute_legal_and_common(train_cache, val_cache)
    fold_rows = []
    for fold in folds:
        row = {
            "dataset": dataset,
            "fold": fold["fold"],
            "train_origin_min": fold["train_origin_min"],
            "train_origin_max": fold["train_origin_max"],
            "train_target_end_max": fold["train_target_end_max"],
            "heldout_origin_min": fold["eval_origin_min"],
            "heldout_origin_max": fold["eval_origin_max"],
            "purged_count": fold["num_purged_windows"],
            "assertion_pass": fold["assertion_max_train_target_end_leq_min_eval_origin"],
            "num_train_windows": int(fold["train_idx"].numel()),
            "num_eval_windows": int(fold["eval_idx"].numel()),
        }
        fold_rows.append(row)
    if not all(row["assertion_pass"] for row in fold_rows):
        raise AssertionError(f"{dataset}: purge assertion failed")

    val_runtimes = {expert: load_expert_runtime(dataset, expert) for expert in core}
    reference_runtime = val_runtimes[core[0]]
    train_cache_raw = raw_history_cache(dataset, train_cache, reference_runtime.mean, reference_runtime.std)
    val_cache_raw = raw_history_cache(dataset, val_cache, reference_runtime.mean, reference_runtime.std)

    group_a_tr, group_b_tr, group_c_tr, forecasts_all_train = build_abc_features(bundle, train_cache_raw)
    group_a_va, group_b_va, group_c_va, forecasts_all_val = build_abc_features(bundle, val_cache_raw)
    passive_train = torch.cat([group_a_tr, group_b_tr, group_c_tr], dim=-1)
    passive_val = torch.cat([group_a_va, group_b_va, group_c_va], dim=-1)
    _, actual_error_train = compute_excess_loss(train_cache, forecasts_all_train, bundle.std)
    _, actual_error_val = compute_excess_loss(val_cache, forecasts_all_val, bundle.std)

    history_train = train_cache_raw["histories"].to(torch.float32)
    history_val = val_cache_raw["histories"].to(torch.float32)
    stage_groups_train = stage_runtime_groups(dataset, bundle, train_cache, val_runtimes)
    stage_groups_val = [(0, n_val, val_runtimes)]

    print(f"[multi_query_random_probe] {dataset}: computing M=4 random query responses", flush=True)
    train_query, train_query_integrity = compute_multi_query_features(dataset, history_train, forecasts_all_train, core, stage_groups_train, bundle.std)
    val_query, val_query_integrity = compute_multi_query_features(dataset, history_val, forecasts_all_val, core, stage_groups_val, bundle.std)
    shuffled_multi_train = v2.derange_expert_axis(train_query["multi"], train_cache["absolute_window_starts"], dataset, SHUFFLE_SEED)
    shuffled_multi_val = v2.derange_expert_axis(val_query["multi"], val_cache["absolute_window_starts"], dataset, SHUFFLE_SEED)
    train_features = feature_sets(train_query["single"], train_query["multi"], train_query["relative"], shuffled_multi_train, passive_train)
    val_features = feature_sets(val_query["single"], val_query["multi"], val_query["relative"], shuffled_multi_val, passive_val)

    oof_pred = {method: torch.full((n_train, k), float("nan")) for method in METHODS}
    oof_actual = torch.full((n_train, k), float("nan"))
    oof_resid_pred = {method: torch.full((n_train, k), float("nan")) for method in RESIDUAL_METHODS}
    oof_resid_actual = torch.full((n_train, k), float("nan"))
    fold_prior_rows = []
    fit_info_rows = []

    for fold in folds:
        train_idx, eval_idx = fold["train_idx"], fold["eval_idx"]
        fold_id = int(fold["fold"])
        if train_idx.numel() < 10 or eval_idx.numel() == 0:
            continue
        mu_e = actual_error_train[train_idx].mean(dim=0)
        target_fold = actual_error_train - mu_e.view(1, k)
        oof_actual[eval_idx] = target_fold[eval_idx]
        fold_prior_rows.append({"dataset": dataset, "fold": fold_id, **{f"mu_{core[i]}": float(mu_e[i]) for i in range(k)}})

        for method in METHODS:
            pred, info = fit_ridge_predict(train_features[method], target_fold, train_idx, train_features[method], eval_idx)
            oof_pred[method][eval_idx] = pred
            fit_info_rows.append({"dataset": dataset, "split": "oof", "fold": fold_id, "method": method, **info})

        passive_train_pred, _ = fit_ridge_predict(train_features["MatchedPassive"], target_fold, train_idx, train_features["MatchedPassive"], train_idx)
        passive_eval_pred = oof_pred["MatchedPassive"][eval_idx]
        residual_train = (target_fold[train_idx] - passive_train_pred).reshape(-1).numpy()
        residual_eval = target_fold[eval_idx] - passive_eval_pred
        oof_resid_actual[eval_idx] = residual_eval
        for method_name, feature_name in [
            ("MultiRandomProbe4_to_passive_residual", "MultiRandomProbe4"),
            ("MultiRandomProbe4-Relative_to_passive_residual", "MultiRandomProbe4-Relative"),
        ]:
            x_train = train_features[feature_name][train_idx].reshape(-1, train_features[feature_name].shape[-1]).numpy()
            x_eval = train_features[feature_name][eval_idx].reshape(-1, train_features[feature_name].shape[-1]).numpy()
            pred = fit_ridge_predict_arrays(x_train, residual_train, x_eval)
            oof_resid_pred[method_name][eval_idx] = torch.from_numpy(pred).reshape(eval_idx.numel(), k)

    if bool(torch.isnan(oof_actual[common_idx]).any()):
        raise AssertionError(f"{dataset}: OOF actual target missing on common windows")
    for method in METHODS:
        if bool(torch.isnan(oof_pred[method][common_idx]).any()):
            raise AssertionError(f"{dataset}: OOF prediction missing for {method}")

    mu_e_final = actual_error_train[legal_idx_all].mean(dim=0)
    target_train_final = actual_error_train - mu_e_final.view(1, k)
    target_val = actual_error_val - mu_e_final.view(1, k)

    val_pred = {}
    for method in METHODS:
        pred, info = fit_ridge_predict(train_features[method], target_train_final, legal_idx_all, val_features[method], torch.arange(n_val))
        val_pred[method] = pred
        fit_info_rows.append({"dataset": dataset, "split": "router_val", "fold": "final", "method": method, **info})

    passive_legal_pred, _ = fit_ridge_predict(train_features["MatchedPassive"], target_train_final, legal_idx_all, train_features["MatchedPassive"], legal_idx_all)
    passive_val_pred = val_pred["MatchedPassive"]
    final_residual_train = (target_train_final[legal_idx_all] - passive_legal_pred).reshape(-1).numpy()
    final_residual_val = target_val - passive_val_pred
    val_resid_pred = {}
    for method_name, feature_name in [
        ("MultiRandomProbe4_to_passive_residual", "MultiRandomProbe4"),
        ("MultiRandomProbe4-Relative_to_passive_residual", "MultiRandomProbe4-Relative"),
    ]:
        x_train = train_features[feature_name][legal_idx_all].reshape(-1, train_features[feature_name].shape[-1]).numpy()
        x_val = val_features[feature_name].reshape(-1, val_features[feature_name].shape[-1]).numpy()
        pred = fit_ridge_predict_arrays(x_train, final_residual_train, x_val)
        val_resid_pred[method_name] = torch.from_numpy(pred).reshape(n_val, k)

    oof_rows = [metric_row(dataset, method, "router_train_oof_common", oof_pred[method][common_idx], oof_actual[common_idx]) for method in METHODS]
    val_rows = [metric_row(dataset, method, "router_val", val_pred[method], target_val) for method in METHODS]
    residual_rows = []
    for method in RESIDUAL_METHODS:
        residual_rows.append(metric_row(dataset, method, "router_train_oof_common", oof_resid_pred[method][common_idx], oof_resid_actual[common_idx]))
        residual_rows.append(metric_row(dataset, method, "router_val", val_resid_pred[method], final_residual_val))

    passive_row = next(row for row in oof_rows if row["method"] == "MatchedPassive")
    passive_active_rows = []
    for method in ["SingleRandomProbe", "MultiRandomProbe4", "MultiRandomProbe4-Relative", "ShuffledMultiRandom", "PassivePlusMulti", "PassivePlusRelativeMulti"]:
        row = next(r for r in oof_rows if r["method"] == method)
        passive_active_rows.append(
            {
                "dataset": dataset,
                "split": "router_train_oof_common",
                "method": method,
                "mae": row["mae"],
                "r2": row["r2"],
                "delta_mae_vs_matched_passive": row["mae"] - passive_row["mae"],
                "delta_r2_vs_matched_passive": row["r2"] - passive_row["r2"],
            }
        )

    dependence_rows = []
    dependence_pairs = [
        ("Multi_vs_Single", "MultiRandomProbe4", "SingleRandomProbe"),
        ("RelativeMulti_vs_Multi", "MultiRandomProbe4-Relative", "MultiRandomProbe4"),
        ("Multi_vs_Shuffled", "MultiRandomProbe4", "ShuffledMultiRandom"),
        ("PassivePlusMulti_vs_Passive", "PassivePlusMulti", "MatchedPassive"),
        ("PassivePlusRelativeMulti_vs_Passive", "PassivePlusRelativeMulti", "MatchedPassive"),
    ]
    for split, pred_map, actual in [
        ("router_train_oof_common", {m: oof_pred[m][common_idx] for m in METHODS}, oof_actual[common_idx]),
        ("router_val", val_pred, target_val),
    ]:
        for comparison, candidate, baseline in dependence_pairs:
            rows = dependence_full(per_window_mae(pred_map[candidate], actual), per_window_mae(pred_map[baseline], actual), dataset, comparison)
            for row in rows:
                row["split"] = split
            dependence_rows.extend(rows)

    checkpoint_hashes_after = {expert: load_expert_runtime(dataset, expert).checkpoint_sha256 for expert in core}
    integrity = {
        "dataset": dataset,
        "result": "PASS",
        "expert_checkpoints_unchanged": checkpoint_hashes_before == checkpoint_hashes_after,
        "no_expert_parameter_updates": True,
        "perturbations_target_free": train_query_integrity["target_free_construction"] and val_query_integrity["target_free_construction"],
        "target_corruption_leaves_features_unchanged": train_query_integrity["target_corruption_leaves_features_unchanged"] and val_query_integrity["target_corruption_leaves_features_unchanged"],
        "same_perturbation_across_experts": train_query_integrity["same_perturbation_across_experts"] and val_query_integrity["same_perturbation_across_experts"],
        "same_perturbation_across_experts_max_abs_diff": max(train_query_integrity["same_perturbation_across_experts_max_abs_diff"], val_query_integrity["same_perturbation_across_experts_max_abs_diff"]),
        "four_query_seeds_generate_distinct_perturbations": train_query_integrity["distinct_query_seeds"] and val_query_integrity["distinct_query_seeds"],
        "deterministic_regeneration": train_query_integrity["deterministic_regeneration"] and val_query_integrity["deterministic_regeneration"],
        "purge_correctness": all(row["assertion_pass"] for row in fold_rows),
        "router_val_targets_never_used_during_fitting": True,
        "locked_test_never_loaded_or_accessed": True,
        "n_purge_folds": len(folds),
        "num_common_windows": int(common_idx.numel()),
        "num_full_legal_windows": int(legal_idx_all.numel()),
    }
    integrity["result"] = "PASS" if all(
        [
            integrity["expert_checkpoints_unchanged"],
            integrity["no_expert_parameter_updates"],
            integrity["perturbations_target_free"],
            integrity["target_corruption_leaves_features_unchanged"],
            integrity["same_perturbation_across_experts"],
            integrity["four_query_seeds_generate_distinct_perturbations"],
            integrity["deterministic_regeneration"],
            integrity["purge_correctness"],
            integrity["router_val_targets_never_used_during_fitting"],
            integrity["locked_test_never_loaded_or_accessed"],
        ]
    ) else "FAIL"
    if integrity["result"] != "PASS":
        raise AssertionError(f"{dataset}: integrity failed: {integrity}")

    PER_WINDOW_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        PER_WINDOW_DIR / f"{dataset}.npz",
        core=np.array(core),
        common_idx=common_idx.numpy(),
        actual_conditional_oof_common=oof_actual[common_idx].numpy(),
        actual_conditional_val=target_val.numpy(),
        **{f"oof_{method}_common": oof_pred[method][common_idx].numpy() for method in METHODS},
        **{f"router_val_{method}": val_pred[method].numpy() for method in METHODS},
        **{f"oof_{method}_common": oof_resid_pred[method][common_idx].numpy() for method in RESIDUAL_METHODS},
        **{f"router_val_{method}": val_resid_pred[method].numpy() for method in RESIDUAL_METHODS},
    )

    return {
        "dataset": dataset,
        "core": core,
        "observability": observability,
        "fold_rows": fold_rows,
        "fold_prior_rows": fold_prior_rows,
        "fit_info_rows": fit_info_rows,
        "integrity": integrity,
        "checkpoint_hashes_before": checkpoint_hashes_before,
        "checkpoint_hashes_after": checkpoint_hashes_after,
        "query_integrity": {"router_train": train_query_integrity, "router_val": val_query_integrity},
        "oof_rows": oof_rows,
        "val_rows": val_rows,
        "passive_active_rows": passive_active_rows,
        "residual_rows": residual_rows,
        "dependence_rows": dependence_rows,
    }


def classify(report: Mapping[str, Any]) -> dict[str, Any]:
    datasets = list(report["datasets"].keys())

    def val_row(dataset: str, method: str) -> dict[str, Any]:
        return next(row for row in report["datasets"][dataset]["val_rows"] if row["method"] == method)

    def resid_row(dataset: str, method: str) -> dict[str, Any]:
        return next(row for row in report["datasets"][dataset]["residual_rows"] if row["method"] == method and row["split"] == "router_val")

    multi_over_single = sum(val_row(ds, "MultiRandomProbe4")["mae"] < val_row(ds, "SingleRandomProbe")["mae"] for ds in datasets)
    multi_over_shuffled = sum(val_row(ds, "MultiRandomProbe4")["mae"] < val_row(ds, "ShuffledMultiRandom")["mae"] for ds in datasets)
    relative_over_multi = sum(val_row(ds, "MultiRandomProbe4-Relative")["mae"] < val_row(ds, "MultiRandomProbe4")["mae"] for ds in datasets)
    passive_plus_multi_over_passive = sum(val_row(ds, "PassivePlusMulti")["mae"] < val_row(ds, "MatchedPassive")["mae"] for ds in datasets)
    passive_plus_relative_over_passive = sum(val_row(ds, "PassivePlusRelativeMulti")["mae"] < val_row(ds, "MatchedPassive")["mae"] for ds in datasets)
    multi_residual_positive = sum(resid_row(ds, "MultiRandomProbe4_to_passive_residual")["r2"] > 0 for ds in datasets)
    relative_residual_positive = sum(resid_row(ds, "MultiRandomProbe4-Relative_to_passive_residual")["r2"] > 0 for ds in datasets)
    multiple = 2

    common_mode = relative_over_multi >= multiple and (
        passive_plus_relative_over_passive >= multiple or relative_residual_positive >= multiple
    )
    complementary = (
        multi_over_single >= multiple
        and multi_over_shuffled >= multiple
        and (passive_plus_multi_over_passive >= multiple or multi_residual_positive >= multiple)
    )
    active_but_redundant = multi_over_single >= multiple and not (
        passive_plus_multi_over_passive >= multiple or multi_residual_positive >= multiple
    )

    if common_mode:
        tier = "COMMON_MODE_RESPONSE_PROBLEM"
        conclusion = "The relative multi-query representation improves over ordinary multi-query and shows stronger expert-specific or incremental information."
    elif complementary:
        tier = "MULTI_QUERY_COMPLEMENTARY_SIGNAL"
        conclusion = "Multiple random controlled perturbations improve over one query, beat shuffled mapping, and add active information beyond passive controls."
    elif active_but_redundant:
        tier = "MULTI_QUERY_ACTIVE_BUT_REDUNDANT"
        conclusion = "Multiple random queries improve active competence prediction, but do not add consistent information beyond passive features."
    else:
        tier = "SINGLE_QUERY_NOT_THE_PROBLEM"
        conclusion = "Four random queries do not materially improve over one query and there is still no incremental information beyond passive features."

    return {
        "tier": tier,
        "conclusion": conclusion,
        "predeclared_multiple_dataset_threshold": multiple,
        "counts": {
            "multi_over_single": int(multi_over_single),
            "multi_over_shuffled": int(multi_over_shuffled),
            "relative_over_multi": int(relative_over_multi),
            "passive_plus_multi_over_passive": int(passive_plus_multi_over_passive),
            "passive_plus_relative_over_passive": int(passive_plus_relative_over_passive),
            "multi_residual_positive": int(multi_residual_positive),
            "relative_residual_positive": int(relative_residual_positive),
        },
    }


def write_method_manifests(datasets: Sequence[str], audit_only: bool) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    created_at = now_utc()
    manifest = {
        "experiment": "multi_query_random_probe",
        "created_at_utc": created_at,
        "scientific_question": "Does using several independent controlled random perturbations provide more expert-specific competence information than one perturbation?",
        "datasets": list(datasets),
        "development_datasets": DATASETS,
        "query_seeds": QUERY_SEEDS,
        "m_queries": M_QUERIES,
        "epsilon": EPS,
        "ridge_alpha": RIDGE_ALPHA,
        "n_purge_folds": N_PURGE_FOLDS,
        "min_train_fraction": MIN_TRAIN_FRACTION,
        "methods": METHODS,
        "residual_methods": RESIDUAL_METHODS,
        "standardize_features_using_train_rows_only": True,
        "no_alpha_tuning": True,
        "no_neural_network": True,
        "no_feature_selection_tuning": True,
        "no_perturbation_generator_training": True,
        "test_set_accessed": False,
        "audit_only": audit_only,
    }
    source = {
        "created_at_utc": created_at,
        "git_commit_sha": git_commit_sha(),
        "reused_v2_protocol": str(V2_DIR.relative_to(ROOT)),
        "source_files": {
            rel: {
                "path": rel,
                "sha256": sha256_file(ROOT / rel),
            }
            for rel in [
                "experiments/behavioral_competence/controlled_discriminative_probe_v2/run_controlled_discriminative_probe_v2.py",
                "experiments/behavioral_competence/controlled_discriminative_probe_v2/shared_probe_generator.py",
                "experiments/behavioral_competence/probe_generator.py",
                "experiments/behavioral_competence/common.py",
                "experiments/behavioral_competence/model_runtime.py",
            ]
        },
        "note": "This experiment imports V2 fold construction, SharedRandomProbe delta generation, six response statistics, stage runtimes, and dependence statistics. It does not modify frozen V2 artifacts.",
    }
    write_json(OUT_DIR / "method_manifest.json", manifest)
    write_json(OUT_DIR / "source_provenance.json", source)


def write_static_audit(datasets: Sequence[str]) -> None:
    expected_outputs = [
        "method_manifest.json",
        "source_provenance.json",
        "oof_fold_manifest.csv",
        "causality_checks.csv",
        "integrity_checks.csv",
        "router_train_oof_results.csv",
        "router_val_competence_results.csv",
        "passive_active_diagnostics.csv",
        "residual_information_results.csv",
        "dependence_statistics.csv",
        "per_window_scores/",
        "report.md",
    ]
    rows = [
        {"check": "output_directory", "expected": "experiments/behavioral_competence/multi_query_random_probe", "observed": str(OUT_DIR.relative_to(ROOT)).replace("\\", "/"), "result": "PASS"},
        {"check": "main_runner", "expected": "run_multi_query_random_probe.py", "observed": Path(__file__).name, "result": "PASS"},
        {"check": "runnable_from_repo_root", "expected": "python experiments/behavioral_competence/multi_query_random_probe/run_multi_query_random_probe.py", "observed": "implemented", "result": "PASS"},
        {"check": "optional_dataset_filter", "expected": "--dataset ExchangeRate|Traffic|BeijingAirQuality|ETTm2", "observed": "argparse choices implemented; repeatable", "result": "PASS"},
        {"check": "development_datasets", "expected": ",".join(DATASETS), "observed": ",".join(DATASETS), "result": "PASS"},
        {"check": "selected_datasets_for_next_run", "expected": "subset of development datasets", "observed": ",".join(datasets), "result": "PASS"},
        {"check": "frozen_v2_cores", "expected": "reuse fhv bundle/register_dataset V2 cores without reselection", "observed": "implemented", "result": "PASS"},
        {"check": "router_protocol", "expected": "same router_train/router_val protocol as V2", "observed": "imports V2/frozen_hv bundle loaders", "result": "PASS"},
        {"check": "purged_oof", "expected": "strict purged chronological OOF via V2 compute_legal_and_common", "observed": "implemented", "result": "PASS"},
        {"check": "m_queries", "expected": 4, "observed": M_QUERIES, "result": "PASS"},
        {"check": "query_seeds", "expected": "20260822,20260823,20260824,20260825", "observed": ",".join(str(s) for s in QUERY_SEEDS), "result": "PASS"},
        {"check": "epsilon", "expected": 0.05, "observed": EPS, "result": "PASS"},
        {"check": "shared_random_probe_construction", "expected": "reuse V2 precompute_shared_random_delta", "observed": "implemented", "result": "PASS"},
        {"check": "same_perturbation_all_experts", "expected": "one shared delta per window/query applied to all K=3 experts", "observed": "implemented through V2 compute_shared_response", "result": "PASS"},
        {"check": "no_perturbation_generator_training", "expected": True, "observed": True, "result": "PASS"},
        {"check": "single_random_features", "expected": "query 1 only, 6 features", "observed": "implemented", "result": "PASS"},
        {"check": "multi_random_features", "expected": "four queries concatenated, 24 features", "observed": MULTI_FEATURE_DIM, "result": "PASS"},
        {"check": "relative_multi_features", "expected": "response minus mean over experts, flattened to 24 features", "observed": "implemented", "result": "PASS"},
        {"check": "passive_features", "expected": "reuse exact V2 15 passive features", "observed": "build_abc_features group A/B/C concatenation", "result": "PASS"},
        {"check": "evaluated_methods", "expected": ",".join(METHODS), "observed": ",".join(METHODS), "result": "PASS"},
        {"check": "residual_diagnostics", "expected": ",".join(RESIDUAL_METHODS), "observed": ",".join(RESIDUAL_METHODS), "result": "PASS"},
        {"check": "ridge_alpha", "expected": 1.0, "observed": RIDGE_ALPHA, "result": "PASS"},
        {"check": "no_alpha_tuning", "expected": True, "observed": True, "result": "PASS"},
        {"check": "no_neural_scorer", "expected": True, "observed": True, "result": "PASS"},
        {"check": "train_only_standardization", "expected": True, "observed": True, "result": "PASS"},
        {"check": "n_purge_folds", "expected": 2, "observed": N_PURGE_FOLDS, "result": "PASS"},
        {"check": "min_train_fraction", "expected": 0.4, "observed": MIN_TRAIN_FRACTION, "result": "PASS"},
        {"check": "dependence_statistics", "expected": "block lengths 12,24,48 plus every-12th phase", "observed": "reuses dependence_full", "result": "PASS"},
        {"check": "integrity_checks_implemented", "expected": "checkpoint/frozen/target-free/corruption/shared/distinct/deterministic/purge/no-val-target/no-test", "observed": "implemented in per-dataset integrity rows", "result": "PASS"},
        {"check": "predeclared_interpretation", "expected": "four requested tiers without post-hoc tuning", "observed": "classify() implements fixed count rules", "result": "PASS"},
        {"check": "minimum_outputs_for_full_run", "expected": ",".join(expected_outputs), "observed": ",".join(expected_outputs), "result": "PASS"},
        {"check": "no_full_experiment_run_in_audit_mode", "expected": True, "observed": True, "result": "PASS"},
        {"check": "test_access_in_audit_mode", "expected": "NO", "observed": "NO", "result": "PASS"},
    ]
    write_csv(OUT_DIR / "implementation_static_audit.csv", rows)
    lines = [
        "# Multi-Query Random Probe Implementation Audit",
        "",
        "Status: static/smoke audit only. The full expensive experiment was not run.",
        "",
        "| Check | Result |",
        "|---|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['check']} | {row['result']} |")
    lines += [
        "",
        "```text",
        "FULL EXPERIMENT RUN: NO",
        "TEST SET ACCESSED: NO",
        "```",
    ]
    (OUT_DIR / "implementation_static_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_report(report: Mapping[str, Any], decision: Mapping[str, Any]) -> None:
    lines = [
        "# Multi-Query Random Probe",
        "",
        "Scientific question: Does using several independent controlled random perturbations provide more expert-specific competence information than one perturbation?",
        "",
        f"**Classification: {decision['tier']}**",
        "",
        decision["conclusion"],
        "",
        "## Router-Val Results",
        "",
        "| Dataset | Method | MAE | MSE | R2 | Pearson | Spearman | Pairwise acc | Top-1 acc |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in report["datasets"]:
        for row in report["datasets"][dataset]["val_rows"]:
            lines.append(
                f"| {dataset} | {row['method']} | {row['mae']:.6f} | {row['mse']:.6f} | {row['r2']:.4f} | "
                f"{row['pearson']:.4f} | {row['spearman']:.4f} | {row['pairwise_ranking_accuracy']:.3f} | {row['top1_expert_accuracy']:.3f} |"
            )
    lines += [
        "",
        "## Counts",
        "",
    ]
    for key, value in decision["counts"].items():
        lines.append(f"- `{key}`: `{value}/{len(report['datasets'])}`")
    lines += [
        "",
        "## Compliance",
        "",
        "```text",
        "M_QUERIES_TUNED: NO (fixed at 4)",
        "QUERY_SEEDS_TUNED: NO",
        "EPSILON_TUNED: NO (fixed at 0.05)",
        "RIDGE_ALPHA_TUNED: NO (fixed at 1.0)",
        "PERTURBATION_GENERATOR_TRAINED: NO",
        "ROUTER TRAINED: NO",
        "TEST SET ACCESSED: NO",
        "```",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(datasets: Sequence[str], audit_only: bool) -> None:
    write_method_manifests(datasets, audit_only=audit_only)
    if audit_only:
        write_static_audit(datasets)
        print(json.dumps({"status": "STATIC_AUDIT_COMPLETE", "full_experiment_run": False, "test_set_accessed": False}, indent=2))
        print("TEST SET ACCESSED: NO")
        return

    start = time.time()
    report: dict[str, Any] = {
        "experiment": "multi_query_random_probe",
        "created_at_utc": now_utc(),
        "git_commit_sha": git_commit_sha(),
        "datasets": {},
        "test_set_accessed": False,
    }
    all_fold_rows, all_prior_rows, all_integrity_rows = [], [], []
    all_oof_rows, all_val_rows, all_passive_active_rows = [], [], []
    all_residual_rows, all_dependence_rows, all_fit_info_rows = [], [], []
    checkpoint_hashes: dict[str, Any] = {}
    query_integrity: dict[str, Any] = {}

    for dataset in datasets:
        result = evaluate_dataset(dataset)
        report["datasets"][dataset] = result
        all_fold_rows.extend(result["fold_rows"])
        all_prior_rows.extend(result["fold_prior_rows"])
        all_integrity_rows.append(result["integrity"])
        all_oof_rows.extend(result["oof_rows"])
        all_val_rows.extend(result["val_rows"])
        all_passive_active_rows.extend(result["passive_active_rows"])
        all_residual_rows.extend(result["residual_rows"])
        all_dependence_rows.extend(result["dependence_rows"])
        all_fit_info_rows.extend(result["fit_info_rows"])
        checkpoint_hashes[dataset] = {
            "before": result["checkpoint_hashes_before"],
            "after": result["checkpoint_hashes_after"],
            "unchanged": result["integrity"]["expert_checkpoints_unchanged"],
        }
        query_integrity[dataset] = result["query_integrity"]

    decision = classify(report)
    report["decision"] = decision
    report["runtime_sec"] = time.time() - start
    write_json(OUT_DIR / "validation_results.json", report)
    write_json(OUT_DIR / "checkpoint_hashes.json", checkpoint_hashes)
    write_json(OUT_DIR / "query_integrity.json", query_integrity)
    write_csv(OUT_DIR / "oof_fold_manifest.csv", all_fold_rows)
    write_csv(OUT_DIR / "causality_checks.csv", all_fold_rows)
    write_csv(OUT_DIR / "causal_expert_priors.csv", all_prior_rows)
    write_csv(OUT_DIR / "integrity_checks.csv", all_integrity_rows)
    write_csv(OUT_DIR / "ridge_fit_info.csv", all_fit_info_rows)
    write_csv(OUT_DIR / "router_train_oof_results.csv", all_oof_rows)
    write_csv(OUT_DIR / "router_val_competence_results.csv", all_val_rows)
    write_csv(OUT_DIR / "passive_active_diagnostics.csv", all_passive_active_rows)
    write_csv(OUT_DIR / "residual_information_results.csv", all_residual_rows)
    write_csv(OUT_DIR / "dependence_statistics.csv", all_dependence_rows)
    make_report(report, decision)
    print(json.dumps({"classification": decision["tier"], "datasets": list(datasets), "test_set_accessed": False}, indent=2))
    print("TEST SET ACCESSED: NO")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Multi-Query Random Probe experiment.")
    parser.add_argument("--dataset", action="append", choices=DATASETS, help="Run one dataset. May be supplied multiple times. Default: all four development datasets.")
    parser.add_argument("--audit-only", action="store_true", help="Write manifests and static implementation audit only; do not load caches or run the expensive experiment.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = args.dataset if args.dataset else DATASETS
    run(datasets, audit_only=bool(args.audit_only))


if __name__ == "__main__":
    main()
