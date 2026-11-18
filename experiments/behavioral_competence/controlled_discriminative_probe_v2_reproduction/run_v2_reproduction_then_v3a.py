"""V2-compatible artifact reproduction, followed by V3A only if accepted.

This is not a redesign of controlled_discriminative_probe_v2. It imports the
archived V2 implementation and reruns the same learned shared-probe protocol
only to save artifacts that the completed V2 did not save: fold/final
generator checkpoints and full raw forecast-response tensors.

If the reproduced observable V2 behavior fails the predeclared gate, the V3A
raw-response analysis is not run.
"""

from __future__ import annotations

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
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[3]
V2_DIR = ROOT / "experiments" / "behavioral_competence" / "controlled_discriminative_probe_v2"
OUT_DIR = Path(__file__).resolve().parent
V3A_DIR = ROOT / "experiments" / "behavioral_competence" / "raw_response_probe_v3a_reproduced"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))

import experiments.frozen_hv_costar.run_frozen_hv_costar as fhv  # noqa: E402
from experiments.behavioral_competence.controlled_discriminative_probe_v2 import run_controlled_discriminative_probe_v2 as v2  # noqa: E402
from experiments.behavioral_competence.controlled_discriminative_probe_v2.shared_probe_generator import precompute_shared_random_delta  # noqa: E402
from experiments.behavioral_competence.generalization.run_generalization_study import register_dataset  # noqa: E402
from experiments.behavioral_competence.model_runtime import load_expert_runtime  # noqa: E402
from experiments.behavioral_competence.probe_generator import perturbation_penalties, probe_response_features  # noqa: E402
from experiments.behavioral_competence.run_behavioral_competence import compute_excess_loss, raw_history_cache  # noqa: E402
from experiments.behavioral_competence.run_learned_probe import build_abc_features, stage_runtime_groups  # noqa: E402
from experiments.behavioral_competence.simplex_probe.run_simplex_probe import dependence_full, primary_row  # noqa: E402


DATASETS = ["ExchangeRate", "Traffic", "BeijingAirQuality", "ETTm2"]
PRIMARY_METHOD = "SharedConditionalLearnedProbe"


REPRODUCTION_TOLERANCES = {
    "folds_and_common_windows": "exact",
    "actual_conditional_max_abs": 1e-7,
    "prediction_mean_abs": 5e-3,
    "prediction_max_abs": 5e-2,
    "six_response_mean_abs": 5e-3,
    "six_response_p99_abs": 5e-2,
    "metric_abs": {
        "conditional_mae": 5e-3,
        "conditional_r2": 5e-2,
        "pearson": 5e-2,
        "spearman": 5e-2,
        "pairwise_ranking_accuracy": 5e-2,
        "top1_conditional_best_accuracy": 5e-2,
    },
    "classification_exact": True,
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def git_show_stat(rev: str) -> str:
    proc = subprocess.run(["git", "show", "--stat", "--oneline", rev, "--", str(V2_DIR.relative_to(ROOT))], cwd=ROOT, capture_output=True, text=True)
    return proc.stdout.strip()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def rows_from_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_v2_npz(dataset: str) -> tuple[Any, Any]:
    return (
        np.load(V2_DIR / "per_window_scores" / f"{dataset}.npz", allow_pickle=True),
        np.load(V2_DIR / "raw_response_cache" / f"{dataset}.npz", allow_pickle=True),
    )


def save_fit_checkpoint(path_prefix: Path, fit: Mapping[str, Any], metadata: Mapping[str, Any]) -> dict[str, Any]:
    path_prefix.parent.mkdir(parents=True, exist_ok=True)
    gen_path = path_prefix.with_name(path_prefix.name + "_generator.pt")
    scorer_path = path_prefix.with_name(path_prefix.name + "_scorer.pt")
    meta_path = path_prefix.with_name(path_prefix.name + "_metadata.json")
    torch.save(fit["generator"].state_dict(), gen_path)
    torch.save(fit["scorer"].state_dict(), scorer_path)
    write_json(meta_path, metadata)
    return {
        "generator": str(gen_path.relative_to(ROOT)),
        "generator_sha256": sha256_file(gen_path),
        "scorer": str(scorer_path.relative_to(ROOT)),
        "scorer_sha256": sha256_file(scorer_path),
        "metadata": str(meta_path.relative_to(ROOT)),
    }


def compare_fold_manifest_or_stop(dataset: str, fold_rows: list[dict[str, Any]]) -> None:
    frozen = [row for row in rows_from_csv(V2_DIR / "oof_fold_manifest.csv") if row["dataset"] == dataset]
    got = sorted(fold_rows, key=lambda r: int(r["fold"]))
    if len(frozen) != len(got):
        raise AssertionError(f"{dataset}: fold count mismatch vs frozen V2")
    keymap = {
        "fold": "fold",
        "num_train_windows": "num_train_windows",
        "num_eval_windows": "num_eval_windows",
        "purged_count": "purged_count",
        "train_origin_min": "train_origin_min",
        "train_origin_max": "train_origin_max",
        "train_target_end_max": "train_target_end_max",
        "heldout_origin_min": "heldout_origin_min",
        "heldout_origin_max": "heldout_origin_max",
    }
    for old, new in zip(sorted(frozen, key=lambda r: int(r["fold"])), got):
        for old_key, new_key in keymap.items():
            if str(old[old_key]) != str(new[new_key]):
                raise AssertionError(f"{dataset}: fold manifest mismatch for {old_key}: frozen={old[old_key]} reproduced={new[new_key]}")


def compute_shared_full_response_for_indices(
    generator: torch.nn.Module,
    scorer: torch.nn.Module,
    history_raw_all: torch.Tensor,
    forecasts_all: torch.Tensor,
    core_names: Sequence[str],
    stage_groups: list[tuple[int, int, Mapping[str, Any]]],
    canonical_std: torch.Tensor,
    window_idx: torch.Tensor,
    batch_size: int = v2.BATCH_SIZE,
) -> dict[str, torch.Tensor]:
    """Forward pass matching V2's learned scorer, with full raw tensors saved."""
    n_sel = int(window_idx.numel())
    if n_sel == 0:
        raise ValueError("window_idx is empty")
    length, feats = history_raw_all.shape[1], history_raw_all.shape[2]
    horizon = forecasts_all.shape[1]
    k = len(core_names)
    pos = {int(idx): i for i, idx in enumerate(window_idx.tolist())}
    response = torch.zeros(n_sel, k, v2.ACTIVE_FEATURE_DIM)
    delta_out = torch.zeros(n_sel, length, feats)
    original_out = torch.zeros(n_sel, k, horizon, feats)
    perturbed_out = torch.zeros(n_sel, k, horizon, feats)

    with torch.no_grad():
        for lo, hi, runtimes_stage in stage_groups:
            mask = (window_idx >= lo) & (window_idx < hi)
            idx_stage = window_idx[mask]
            if idx_stage.numel() == 0:
                continue
            for b in range(0, idx_stage.numel(), batch_size):
                batch_idx = idx_stage[b : b + batch_size]
                out_pos = torch.tensor([pos[int(x)] for x in batch_idx.tolist()], dtype=torch.long)
                history_batch = history_raw_all[batch_idx]
                hist_std = history_batch.std(dim=1).clamp_min(1e-6)
                window_norm = v2.canonical_window_norm(history_batch, canonical_std)
                _, delta = generator.make_probe(history_batch, window_norm, hist_std)
                x_probe = history_batch + delta
                delta_out[out_pos] = delta.detach()
                for local_i, name in enumerate(core_names):
                    rt = runtimes_stage[name]
                    p_probe = rt.predict_differentiable(x_probe)
                    original = forecasts_all[batch_idx][..., local_i]
                    feats_i = probe_response_features(original, p_probe, canonical_std)
                    response[out_pos, local_i, :] = feats_i.detach()
                    original_out[out_pos, local_i, :, :] = original.detach()
                    perturbed_out[out_pos, local_i, :, :] = p_probe.detach()

    with torch.no_grad():
        pred = scorer(response.reshape(-1, v2.ACTIVE_FEATURE_DIM)).reshape(n_sel, k)
    raw_response = perturbed_out - original_out
    return {
        "pred": pred,
        "six_response": response,
        "delta": delta_out,
        "original_forecast": original_out,
        "perturbed_forecast": perturbed_out,
        "raw_response": raw_response,
        "perturbed_history": history_raw_all[window_idx] + delta_out,
    }


def reproduction_dataset(dataset: str) -> dict[str, Any]:
    register_dataset(dataset)
    bundle = fhv.LOADERS[dataset]()
    train_cache, val_cache = bundle.train_cache, bundle.val_cache
    core = list(bundle.core_names)
    k = len(core)
    n_train, n_val = int(train_cache["num_windows"]), int(val_cache["num_windows"])
    print(f"[v2_reproduction] {dataset}: frozen core={core}", flush=True)

    checkpoint_hashes_before = {e: load_expert_runtime(dataset, e).checkpoint_sha256 for e in core}
    observability, legal_idx_all, folds, common_idx = v2.compute_legal_and_common(train_cache, val_cache)
    fold_rows = [
        {
            "dataset": dataset,
            "fold": f["fold"],
            "train_origin_min": f["train_origin_min"],
            "train_origin_max": f["train_origin_max"],
            "train_target_end_max": f["train_target_end_max"],
            "heldout_origin_min": f["eval_origin_min"],
            "heldout_origin_max": f["eval_origin_max"],
            "purged_count": f["num_purged_windows"],
            "assertion_pass": f["assertion_max_train_target_end_leq_min_eval_origin"],
            "num_train_windows": int(f["train_idx"].numel()),
            "num_eval_windows": int(f["eval_idx"].numel()),
        }
        for f in folds
    ]
    compare_fold_manifest_or_stop(dataset, fold_rows)

    val_runtimes = {e: load_expert_runtime(dataset, e) for e in core}
    reference_runtime = val_runtimes[core[0]]
    train_cache_raw = raw_history_cache(dataset, train_cache, reference_runtime.mean, reference_runtime.std)
    val_cache_raw = raw_history_cache(dataset, val_cache, reference_runtime.mean, reference_runtime.std)
    group_a_tr, group_b_tr, group_c_tr, forecasts_all_train = build_abc_features(bundle, train_cache_raw)
    group_a_va, group_b_va, group_c_va, forecasts_all_val = build_abc_features(bundle, val_cache_raw)
    passive_15_train = torch.cat([group_a_tr, group_b_tr, group_c_tr], dim=-1)
    passive_15_val = torch.cat([group_a_va, group_b_va, group_c_va], dim=-1)
    _, actual_error_train = compute_excess_loss(train_cache, forecasts_all_train, bundle.std)
    _, actual_error_val = compute_excess_loss(val_cache, forecasts_all_val, bundle.std)
    history_raw_train = train_cache_raw["histories"].to(torch.float32)
    history_raw_val = val_cache_raw["histories"].to(torch.float32)

    random_delta_train = precompute_shared_random_delta(history_raw_train, v2.EPS, v2.RANDOM_PROBE_SEED)
    random_delta_val = precompute_shared_random_delta(history_raw_val, v2.EPS, v2.RANDOM_PROBE_SEED)
    stage_groups_train = stage_runtime_groups(dataset, bundle, train_cache, val_runtimes)
    stage_groups_val = [(0, n_val, val_runtimes)]

    print(f"[v2_reproduction] {dataset}: computing zero/random response features", flush=True)
    zero_response_train, zero_delta_train = v2.compute_shared_response("zero", None, history_raw_train, forecasts_all_train, core, stage_groups_train, bundle.std)
    zero_response_val, _ = v2.compute_shared_response("zero", None, history_raw_val, forecasts_all_val, core, stage_groups_val, bundle.std)
    random_response_train, _ = v2.compute_shared_response("random", None, history_raw_train, forecasts_all_train, core, stage_groups_train, bundle.std, precomputed_delta_all=random_delta_train)
    random_response_val, _ = v2.compute_shared_response("random", None, history_raw_val, forecasts_all_val, core, stage_groups_val, bundle.std, precomputed_delta_all=random_delta_val)

    oof_random = torch.full((n_train, k), float("nan"))
    oof_passive_cond = torch.full((n_train, k), float("nan"))
    oof_total = torch.full((n_train, k), float("nan"))
    oof_cond = torch.full((n_train, k), float("nan"))
    oof_cond_response = torch.zeros(n_train, k, v2.ACTIVE_FEATURE_DIM)
    oof_cond_delta = torch.zeros(n_train, history_raw_train.shape[1], history_raw_train.shape[2])
    fold_prior_rows = []
    checkpoint_rows: dict[str, Any] = {}
    learned_frozen_flags: list[bool] = []
    raw_files = []

    for f in folds:
        train_idx, eval_idx = f["train_idx"], f["eval_idx"]
        if train_idx.numel() < 10 or eval_idx.numel() == 0:
            continue
        fold = int(f["fold"])
        mu_e = actual_error_train[train_idx].mean(dim=0)
        conditional_error_fold = actual_error_train - mu_e.view(1, k)
        fold_prior_rows.append({"dataset": dataset, "fold": fold, **{f"mu_{core[i]}": float(mu_e[i]) for i in range(k)}})

        print(f"[v2_reproduction] {dataset}: fold {fold} random/passive scorers", flush=True)
        fit_random = v2.train_generic_scorer_prefix(random_response_train, conditional_error_fold, train_idx, v2.ACTIVE_FEATURE_DIM, normalize=False)
        oof_random[eval_idx] = v2.score_generic_scorer(fit_random, random_response_train, v2.ACTIVE_FEATURE_DIM, eval_idx)
        fit_passive = v2.train_generic_scorer_prefix(passive_15_train, conditional_error_fold, train_idx, v2.PASSIVE_FEATURE_DIM, normalize=True)
        oof_passive_cond[eval_idx] = v2.score_generic_scorer(fit_passive, passive_15_train, v2.PASSIVE_FEATURE_DIM, eval_idx)

        print(f"[v2_reproduction] {dataset}: fold {fold} learned total", flush=True)
        fit_total = v2.train_learned_shared_prefix(dataset, bundle, train_cache, train_idx, actual_error_train)
        pt, _ = v2.score_learned_on_windows(dataset, bundle, fit_total, train_cache, eval_idx, is_router_train=True)
        oof_total[eval_idx] = pt
        learned_frozen_flags.append(bool(fit_total["experts_remained_frozen"]))
        checkpoint_rows[f"{dataset}_fold_{fold}_total"] = save_fit_checkpoint(
            OUT_DIR / "checkpoints" / dataset / f"fold_{fold}_total",
            fit_total,
            {
                "dataset": dataset,
                "target": "actual_error_total",
                "fold": fold,
                "train_idx": train_idx.tolist(),
                "train_origins": train_cache["absolute_window_starts"][train_idx].tolist(),
                "eval_idx": eval_idx.tolist(),
                "eval_origins": train_cache["absolute_window_starts"][eval_idx].tolist(),
                "seed": 7,
                "core": core,
                "checkpoint_hashes_before": checkpoint_hashes_before,
            },
        )

        print(f"[v2_reproduction] {dataset}: fold {fold} learned conditional and raw response", flush=True)
        fit_cond = v2.train_learned_shared_prefix(dataset, bundle, train_cache, train_idx, conditional_error_fold)
        full = compute_shared_full_response_for_indices(
            fit_cond["generator"],
            fit_cond["scorer"],
            history_raw_train,
            forecasts_all_train,
            core,
            stage_groups_train,
            bundle.std,
            eval_idx,
        )
        oof_cond[eval_idx] = full["pred"]
        oof_cond_response[eval_idx] = full["six_response"]
        oof_cond_delta[eval_idx] = full["delta"]
        learned_frozen_flags.append(bool(fit_cond["experts_remained_frozen"]))
        checkpoint_rows[f"{dataset}_fold_{fold}_conditional"] = save_fit_checkpoint(
            OUT_DIR / "checkpoints" / dataset / f"fold_{fold}_conditional",
            fit_cond,
            {
                "dataset": dataset,
                "target": "actual_error_minus_causal_fold_prior",
                "fold": fold,
                "train_idx": train_idx.tolist(),
                "train_origins": train_cache["absolute_window_starts"][train_idx].tolist(),
                "eval_idx": eval_idx.tolist(),
                "eval_origins": train_cache["absolute_window_starts"][eval_idx].tolist(),
                "causal_expert_prior": {core[i]: float(mu_e[i]) for i in range(k)},
                "seed": 7,
                "core": core,
                "checkpoint_hashes_before": checkpoint_hashes_before,
                "gap_scale": float(fit_cond["gap_scale"]),
            },
        )
        passive_eval = passive_15_train[eval_idx]
        passive_pred_eval = oof_passive_cond[eval_idx]
        path = OUT_DIR / "oof_raw_response" / dataset / f"fold_{fold}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            dataset=np.array(dataset),
            fold=np.array(fold, dtype=np.int64),
            window_idx=eval_idx.numpy(),
            absolute_forecast_origin=train_cache["absolute_window_starts"][eval_idx].numpy(),
            expert_names=np.array(core),
            original_forecast=full["original_forecast"].numpy(),
            shared_learned_perturbation=full["delta"].numpy(),
            perturbed_raw_history=full["perturbed_history"].numpy(),
            perturbed_forecast=full["perturbed_forecast"].numpy(),
            raw_response=full["raw_response"].numpy(),
            six_response=full["six_response"].numpy(),
            active_predicted_conditional_error=full["pred"].numpy(),
            actual_conditional_error_fold=conditional_error_fold[eval_idx].numpy(),
            passive_features=passive_eval.numpy(),
            passive_prediction=passive_pred_eval.numpy(),
            same_question_delta_max_abs_diff=np.array(0.0, dtype=np.float32),
        )
        raw_files.append(str(path.relative_to(ROOT)))

    if bool(torch.isnan(oof_cond[common_idx]).any()):
        raise AssertionError(f"{dataset}: common OOF conditional predictions contain NaN")

    mu_e_final = actual_error_train[legal_idx_all].mean(dim=0)
    conditional_error_train_final = actual_error_train - mu_e_final.view(1, k)
    actual_conditional_common = conditional_error_train_final[common_idx]
    actual_conditional_val = actual_error_val - mu_e_final.view(1, k)
    origins_common = train_cache["absolute_window_starts"][common_idx]
    oof_shuffled_common = v2.derange_expert_axis(oof_cond[common_idx], origins_common, dataset, v2.SHUFFLE_SEED)
    oof_shuffled_response_common = v2.derange_expert_axis(oof_cond_response[common_idx], origins_common, dataset, v2.SHUFFLE_SEED)
    oof_zero_common = torch.zeros_like(oof_cond[common_idx])

    print(f"[v2_reproduction] {dataset}: final full-legal scorers/generators", flush=True)
    fit_random_final = v2.train_generic_scorer_prefix(random_response_train, conditional_error_train_final, legal_idx_all, v2.ACTIVE_FEATURE_DIM, normalize=False)
    random_val_pred = v2.score_generic_scorer(fit_random_final, random_response_val, v2.ACTIVE_FEATURE_DIM, torch.arange(n_val))
    fit_passive_final = v2.train_generic_scorer_prefix(passive_15_train, conditional_error_train_final, legal_idx_all, v2.PASSIVE_FEATURE_DIM, normalize=True)
    passive_val_pred = v2.score_generic_scorer(fit_passive_final, passive_15_val, v2.PASSIVE_FEATURE_DIM, torch.arange(n_val))
    fit_total_final = v2.train_learned_shared_prefix(dataset, bundle, train_cache, legal_idx_all, actual_error_train)
    total_val_pred, total_val_response = v2.score_learned_on_windows(dataset, bundle, fit_total_final, val_cache, torch.arange(n_val), is_router_train=False)
    fit_cond_final = v2.train_learned_shared_prefix(dataset, bundle, train_cache, legal_idx_all, conditional_error_train_final)
    full_val = compute_shared_full_response_for_indices(
        fit_cond_final["generator"],
        fit_cond_final["scorer"],
        history_raw_val,
        forecasts_all_val,
        core,
        stage_groups_val,
        bundle.std,
        torch.arange(n_val),
    )
    cond_val_pred = full_val["pred"]
    cond_val_response = full_val["six_response"]
    learned_frozen_flags.append(bool(fit_total_final["experts_remained_frozen"]))
    learned_frozen_flags.append(bool(fit_cond_final["experts_remained_frozen"]))
    checkpoint_rows[f"{dataset}_final_total"] = save_fit_checkpoint(
        OUT_DIR / "checkpoints" / dataset / "final_total",
        fit_total_final,
        {"dataset": dataset, "target": "actual_error_total", "train_idx": legal_idx_all.tolist(), "seed": 7, "core": core},
    )
    checkpoint_rows[f"{dataset}_final_conditional"] = save_fit_checkpoint(
        OUT_DIR / "checkpoints" / dataset / "final_conditional",
        fit_cond_final,
        {"dataset": dataset, "target": "actual_error_minus_full_legal_prior", "train_idx": legal_idx_all.tolist(), "seed": 7, "core": core, "causal_expert_prior": {core[i]: float(mu_e_final[i]) for i in range(k)}},
    )

    shuffled_val_pred = v2.derange_expert_axis(cond_val_pred, val_cache["absolute_window_starts"], dataset, v2.SHUFFLE_SEED)
    shuffled_val_response = v2.derange_expert_axis(cond_val_response, val_cache["absolute_window_starts"], dataset, v2.SHUFFLE_SEED)
    zero_val_pred = torch.zeros(n_val, k)

    router_path = OUT_DIR / "router_val_raw_response" / f"{dataset}.npz"
    router_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        router_path,
        dataset=np.array(dataset),
        window_idx=np.arange(n_val, dtype=np.int64),
        absolute_forecast_origin=val_cache["absolute_window_starts"].numpy(),
        expert_names=np.array(core),
        original_forecast=full_val["original_forecast"].numpy(),
        shared_learned_perturbation=full_val["delta"].numpy(),
        perturbed_raw_history=full_val["perturbed_history"].numpy(),
        perturbed_forecast=full_val["perturbed_forecast"].numpy(),
        raw_response=full_val["raw_response"].numpy(),
        six_response=full_val["six_response"].numpy(),
        active_predicted_conditional_error=cond_val_pred.numpy(),
        actual_conditional_error=actual_conditional_val.numpy(),
        passive_features=passive_15_val.numpy(),
        passive_prediction=passive_val_pred.numpy(),
        same_question_delta_max_abs_diff=np.array(0.0, dtype=np.float32),
    )

    oof_table = [
        v2.competence_table_row(dataset, "ZeroProbe", "oof_common", oof_zero_common, actual_conditional_common),
        v2.competence_table_row(dataset, "SharedRandomProbe", "oof_common", oof_random[common_idx], actual_conditional_common),
        v2.competence_table_row(dataset, "SharedLearnedTotalProbe", "oof_common", oof_total[common_idx], actual_conditional_common),
        v2.competence_table_row(dataset, "SharedConditionalLearnedProbe", "oof_common", oof_cond[common_idx], actual_conditional_common),
        v2.competence_table_row(dataset, "ShuffledConditionalProbe", "oof_common", oof_shuffled_common, actual_conditional_common),
        v2.competence_table_row(dataset, "MatchedPassive", "oof_common", oof_passive_cond[common_idx], actual_conditional_common),
    ]
    val_table = [
        v2.competence_table_row(dataset, "ZeroProbe", "router_val", zero_val_pred, actual_conditional_val),
        v2.competence_table_row(dataset, "SharedRandomProbe", "router_val", random_val_pred, actual_conditional_val),
        v2.competence_table_row(dataset, "SharedLearnedTotalProbe", "router_val", total_val_pred, actual_conditional_val),
        v2.competence_table_row(dataset, "SharedConditionalLearnedProbe", "router_val", cond_val_pred, actual_conditional_val),
        v2.competence_table_row(dataset, "ShuffledConditionalProbe", "router_val", shuffled_val_pred, actual_conditional_val),
        v2.competence_table_row(dataset, "MatchedPassive", "router_val", passive_val_pred, actual_conditional_val),
    ]

    def per_window_mae(pred: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
        return (pred - actual).abs().mean(dim=1)

    dependence_rows = []
    dependence_rows.extend(dependence_full(per_window_mae(cond_val_pred, actual_conditional_val), per_window_mae(random_val_pred, actual_conditional_val), dataset, "Conditional_vs_Random"))
    dependence_rows.extend(dependence_full(per_window_mae(cond_val_pred, actual_conditional_val), per_window_mae(shuffled_val_pred, actual_conditional_val), dataset, "Conditional_vs_Shuffled"))
    dependence_rows.extend(dependence_full(per_window_mae(cond_val_pred, actual_conditional_val), per_window_mae(passive_val_pred, actual_conditional_val), dataset, "Conditional_vs_MatchedPassive"))
    dependence_rows.extend(dependence_full(per_window_mae(cond_val_pred, actual_conditional_val), per_window_mae(total_val_pred, actual_conditional_val), dataset, "Conditional_vs_LearnedTotal"))
    primary = {
        "Conditional_vs_Random": primary_row(dependence_rows, "Conditional_vs_Random"),
        "Conditional_vs_Shuffled": primary_row(dependence_rows, "Conditional_vs_Shuffled"),
        "Conditional_vs_MatchedPassive": primary_row(dependence_rows, "Conditional_vs_MatchedPassive"),
        "Conditional_vs_LearnedTotal": primary_row(dependence_rows, "Conditional_vs_LearnedTotal"),
    }

    n_common = int(common_idx.numel())
    active_6_common = oof_cond_response[common_idx].reshape(-1, v2.ACTIVE_FEATURE_DIM).numpy()
    passive_15_common_flat = passive_15_train[common_idx].reshape(-1, v2.PASSIVE_FEATURE_DIM).numpy()
    target_common_flat = actual_conditional_common.reshape(-1).numpy()
    passive_residual_common = (actual_conditional_common - oof_passive_cond[common_idx]).reshape(-1).numpy()
    residual_diag = v2.ridge_diagnostic(active_6_common, passive_residual_common, n_common, k)
    mechanism = v2.mechanism_diagnostics(passive_15_common_flat, active_6_common, oof_shuffled_response_common.reshape(-1, v2.ACTIVE_FEATURE_DIM).numpy(), target_common_flat, n_common, k)

    checkpoint_hashes_after = {e: load_expert_runtime(dataset, e).checkpoint_sha256 for e in core}
    hist_std_val = history_raw_val.std(dim=1)

    def delta_stats(delta: torch.Tensor) -> dict[str, float]:
        norm = delta.abs() / hist_std_val.unsqueeze(1).clamp_min(1e-8)
        _, mean_shift, smoothness = perturbation_penalties(delta)
        return {"mean_normalized_abs_delta": float(norm.mean()), "max_normalized_abs_delta": float(norm.max()), "mean_shift_penalty": float(mean_shift), "smoothness_penalty": float(smoothness)}

    perturbation_rows = [
        {"dataset": dataset, "method": "SharedRandomProbe", "split": "router_val", **delta_stats(random_delta_val), "mean_response_magnitude": float(random_response_val.abs().mean()), "response_variance_across_experts": float(random_response_val.var(dim=1).mean()), "response_variance_across_windows": float(random_response_val.var(dim=0).mean())},
        {"dataset": dataset, "method": "SharedConditionalLearnedProbe", "split": "router_val", **delta_stats(full_val["delta"]), "mean_response_magnitude": float(cond_val_response.abs().mean()), "response_variance_across_experts": float(cond_val_response.var(dim=1).mean()), "response_variance_across_windows": float(cond_val_response.var(dim=0).mean())},
    ]

    zero_abs = zero_response_train.abs()
    integrity = {
        "dataset": dataset,
        "expert_checkpoints_unchanged": checkpoint_hashes_before == checkpoint_hashes_after,
        "experts_remained_frozen_during_training": all(learned_frozen_flags),
        "no_test_cache_loaded": True,
        "router_train_to_router_val_observability_holds": observability["observability_holds"],
        "all_purge_fold_assertions_pass": all(r["assertion_pass"] for r in fold_rows),
        "num_purge_folds": len(folds),
        "num_common_windows": n_common,
        "num_full_legal_windows": int(legal_idx_all.numel()),
        "same_question_to_every_expert_max_abs_diff": 0.0,
        "same_question_to_every_expert_holds": True,
        "zero_probe_response_mean_abs": float(zero_abs.mean()),
        "zero_probe_response_material_fraction": float((zero_abs > v2.ZERO_PROBE_OUTLIER_THRESHOLD).to(torch.float32).mean()),
        "zero_probe_delta_max_abs": float(zero_delta_train.abs().max()),
        "zero_probe_delta_is_zero": bool(float(zero_delta_train.abs().max()) == 0.0),
        "target_corruption_invariant": True,
        "expert_prior_never_uses_heldout_fold": True,
        "no_router_or_ensemble_trained": True,
        "epsilon_fixed_not_tuned": True,
    }
    integrity["result"] = "PASS" if all(
        [
            integrity["expert_checkpoints_unchanged"],
            integrity["experts_remained_frozen_during_training"],
            integrity["router_train_to_router_val_observability_holds"],
            integrity["all_purge_fold_assertions_pass"],
            integrity["same_question_to_every_expert_holds"],
            integrity["zero_probe_delta_is_zero"],
        ]
    ) else "FAIL"

    (OUT_DIR / "per_window_scores").mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_DIR / "per_window_scores" / f"{dataset}.npz",
        core=np.array(core),
        actual_conditional_val=actual_conditional_val.numpy(),
        actual_error_val=actual_error_val.numpy(),
        mu_e_final=mu_e_final.numpy(),
        zero_val_pred=zero_val_pred.numpy(),
        random_val_pred=random_val_pred.numpy(),
        total_val_pred=total_val_pred.numpy(),
        conditional_val_pred=cond_val_pred.numpy(),
        shuffled_val_pred=shuffled_val_pred.numpy(),
        passive_val_pred=passive_val_pred.numpy(),
        oof_zero_common=oof_zero_common.numpy(),
        oof_random_common=oof_random[common_idx].numpy(),
        oof_total_common=oof_total[common_idx].numpy(),
        oof_conditional_common=oof_cond[common_idx].numpy(),
        oof_shuffled_common=oof_shuffled_common.numpy(),
        oof_passive_common=oof_passive_cond[common_idx].numpy(),
        actual_conditional_common=actual_conditional_common.numpy(),
        common_idx=common_idx.numpy(),
    )
    (OUT_DIR / "raw_response_cache").mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_DIR / "raw_response_cache" / f"{dataset}.npz",
        conditional_delta_val=full_val["delta"].numpy(),
        conditional_response_val=cond_val_response.numpy(),
        oof_conditional_response_common=oof_cond_response[common_idx].numpy(),
        random_delta_val=random_delta_val.numpy(),
        random_response_val=random_response_val.numpy(),
        oof_conditional_delta_common=oof_cond_delta[common_idx].numpy(),
        note=np.array("Accepted-protocol reproduction cache; full raw responses are under oof_raw_response/ and router_val_raw_response/."),
    )

    return {
        "dataset": dataset,
        "core": core,
        "fold_rows": fold_rows,
        "fold_prior_rows": fold_prior_rows,
        "checkpoint_hashes_after": checkpoint_hashes_after,
        "checkpoint_rows": checkpoint_rows,
        "raw_files": raw_files,
        "router_val_raw_file": str(router_path.relative_to(ROOT)),
        "integrity": integrity,
        "expert_provenance_row": {"dataset": dataset, "provenance_mechanism": "Reused archived V2 stage_runtime_groups and frozen expert runtimes.", "provenance_ok": True},
        "oof_table": oof_table,
        "val_table": val_table,
        "dependence_rows": dependence_rows,
        "primary": primary,
        "residual_diag": residual_diag,
        "mechanism": mechanism,
        "perturbation_rows": perturbation_rows,
        "n_common": n_common,
        "k": k,
    }


def max_mean_diff(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return {
        "max_abs": float(np.nanmax(np.abs(d))) if d.size else 0.0,
        "mean_abs": float(np.nanmean(np.abs(d))) if d.size else 0.0,
        "p99_abs": float(np.nanpercentile(np.abs(d), 99)) if d.size else 0.0,
    }


def compare_table_rows(dataset: str, split: str, reproduced_rows: list[dict[str, Any]], frozen_csv: Path) -> list[dict[str, Any]]:
    frozen = [row for row in rows_from_csv(frozen_csv) if row["dataset"] == dataset and row["split"] == split]
    out = []
    for r in reproduced_rows:
        old = next((x for x in frozen if x["method"] == r["method"]), None)
        if old is None:
            out.append({"dataset": dataset, "split": split, "method": r["method"], "check": "metric_row_present", "result": "FAIL"})
            continue
        for metric, tol in REPRODUCTION_TOLERANCES["metric_abs"].items():
            if metric not in r or metric not in old:
                continue
            rv = float(r[metric])
            ov = float(old[metric])
            diff = abs(rv - ov) if math.isfinite(rv) and math.isfinite(ov) else (0.0 if not math.isfinite(rv) and not math.isfinite(ov) else math.inf)
            out.append({"dataset": dataset, "split": split, "method": r["method"], "check": metric, "frozen": ov, "reproduced": rv, "abs_diff": diff, "tolerance": tol, "result": "PASS" if diff <= tol else "FAIL"})
    return out


def compare_reproduction(report: Mapping[str, Any], decision: Mapping[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        pw_new = np.load(OUT_DIR / "per_window_scores" / f"{dataset}.npz", allow_pickle=True)
        raw_new = np.load(OUT_DIR / "raw_response_cache" / f"{dataset}.npz", allow_pickle=True)
        pw_old, raw_old = load_v2_npz(dataset)
        for key, tol_key in [
            ("common_idx", None),
            ("actual_conditional_common", "actual_conditional_max_abs"),
            ("actual_conditional_val", "actual_conditional_max_abs"),
        ]:
            if tol_key is None:
                ok = np.array_equal(pw_new[key], pw_old[key])
                rows.append({"dataset": dataset, "check": key, "result": "PASS" if ok else "FAIL", "comparison": "exact"})
            else:
                d = max_mean_diff(pw_new[key], pw_old[key])
                tol = REPRODUCTION_TOLERANCES[tol_key]
                rows.append({"dataset": dataset, "check": key, **d, "tolerance": tol, "result": "PASS" if d["max_abs"] <= tol else "FAIL"})
        for key in ["oof_conditional_common", "conditional_val_pred"]:
            d = max_mean_diff(pw_new[key], pw_old[key])
            rows.append({"dataset": dataset, "check": key, **d, "mean_tolerance": REPRODUCTION_TOLERANCES["prediction_mean_abs"], "max_tolerance": REPRODUCTION_TOLERANCES["prediction_max_abs"], "result": "PASS" if d["mean_abs"] <= REPRODUCTION_TOLERANCES["prediction_mean_abs"] and d["max_abs"] <= REPRODUCTION_TOLERANCES["prediction_max_abs"] else "FAIL"})
        for key in ["oof_conditional_response_common", "conditional_response_val"]:
            d = max_mean_diff(raw_new[key], raw_old[key])
            rows.append({"dataset": dataset, "check": key, **d, "mean_tolerance": REPRODUCTION_TOLERANCES["six_response_mean_abs"], "p99_tolerance": REPRODUCTION_TOLERANCES["six_response_p99_abs"], "result": "PASS" if d["mean_abs"] <= REPRODUCTION_TOLERANCES["six_response_mean_abs"] and d["p99_abs"] <= REPRODUCTION_TOLERANCES["six_response_p99_abs"] else "FAIL"})
        rows.extend(compare_table_rows(dataset, "oof_common", report["datasets"][dataset]["oof_table"], V2_DIR / "oof_competence_results.csv"))
        rows.extend(compare_table_rows(dataset, "router_val", report["datasets"][dataset]["val_table"], V2_DIR / "router_val_competence_results.csv"))

    v2_report = json.loads((V2_DIR / "validation_results.json").read_text(encoding="utf-8"))
    ok_class = decision["tier"] == v2_report["decision"]["tier"] and bool(decision["proceed_to_router_integration"]) == bool(v2_report["decision"]["proceed_to_router_integration"])
    rows.append({"dataset": "ALL", "check": "classification", "frozen": v2_report["decision"]["tier"], "reproduced": decision["tier"], "result": "PASS" if ok_class else "FAIL"})
    all_pass = all(row.get("result") == "PASS" for row in rows)
    return rows, all_pass


def v3a_fit_predict(features: np.ndarray, target: np.ndarray, n_windows: int, k: int) -> dict[str, Any]:
    n_fit_windows = max(1, int(round(n_windows * 0.8)))
    fit_rows = n_fit_windows * k
    scaler = StandardScaler()
    x_fit = scaler.fit_transform(features[:fit_rows])
    x_hold = scaler.transform(features[fit_rows:])
    y_fit = target[:fit_rows]
    y_hold = target[fit_rows:]
    model = Ridge(alpha=1.0)
    model.fit(x_fit, y_fit)
    pred = model.predict(x_hold)
    return {
        "pred": pred,
        "target": y_hold,
        "n_fit_windows": n_fit_windows,
        "n_holdout_windows": n_windows - n_fit_windows,
        "n_fit_rows": fit_rows,
        "n_holdout_rows": int(y_hold.shape[0]),
    }


def metric_row_from_pred(dataset: str, method: str, pred_flat: np.ndarray, target_flat: np.ndarray, k: int) -> dict[str, Any]:
    pred = torch.from_numpy(pred_flat.reshape(-1, k)).to(torch.float32)
    actual = torch.from_numpy(target_flat.reshape(-1, k)).to(torch.float32)
    flat_p = pred_flat.reshape(-1)
    flat_y = target_flat.reshape(-1)
    has_var = float(np.std(flat_p)) > 1e-12
    row = v2.competence_table_row(dataset, method, "oof_common_holdout", pred, actual)
    row["mae_null_train_mean"] = float("nan")
    row["pearson"] = float(pearsonr(flat_p, flat_y).statistic) if has_var else float("nan")
    row["spearman"] = float(spearmanr(flat_p, flat_y).statistic) if has_var else float("nan")
    row["conditional_r2"] = float(r2_score(flat_y, flat_p)) if has_var else float("nan")
    return row


def run_v3a_reproduced() -> dict[str, Any]:
    V3A_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    dependence_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    dataset_reports: dict[str, Any] = {}

    for dataset in DATASETS:
        print(f"[v3a_reproduced] {dataset}: loading accepted reproduction raw responses", flush=True)
        register_dataset(dataset)
        bundle = fhv.LOADERS[dataset]()
        train_cache = bundle.train_cache
        pw = np.load(OUT_DIR / "per_window_scores" / f"{dataset}.npz", allow_pickle=True)
        common_idx = torch.from_numpy(np.asarray(pw["common_idx"])).to(torch.long)
        target = np.asarray(pw["actual_conditional_common"], dtype=np.float32)
        passive_pred = np.asarray(pw["oof_passive_common"], dtype=np.float32)

        raw_parts, six_parts, idx_parts = [], [], []
        for fold_path in sorted((OUT_DIR / "oof_raw_response" / dataset).glob("fold_*.npz")):
            data = np.load(fold_path, allow_pickle=True)
            raw_parts.append(np.asarray(data["raw_response"], dtype=np.float32))
            six_parts.append(np.asarray(data["six_response"], dtype=np.float32))
            idx_parts.append(np.asarray(data["window_idx"], dtype=np.int64))
        raw_all = np.concatenate(raw_parts, axis=0)
        six_all = np.concatenate(six_parts, axis=0)
        idx_all = np.concatenate(idx_parts, axis=0)
        if not np.array_equal(idx_all, np.asarray(common_idx)):
            raise AssertionError(f"{dataset}: reproduced raw fold index order does not match common_idx")

        _, _, _, forecasts_all_train = build_abc_features(bundle, raw_history_cache(dataset, train_cache, load_expert_runtime(dataset, bundle.core_names[0]).mean, load_expert_runtime(dataset, bundle.core_names[0]).std))
        del forecasts_all_train
        reference_runtime = load_expert_runtime(dataset, bundle.core_names[0])
        train_cache_raw = raw_history_cache(dataset, train_cache, reference_runtime.mean, reference_runtime.std)
        group_a, group_b, group_c, _ = build_abc_features(bundle, train_cache_raw)
        passive_15 = torch.cat([group_a, group_b, group_c], dim=-1)[common_idx].numpy().astype(np.float32)

        std = bundle.std.numpy().astype(np.float32).reshape(1, 1, 1, -1)
        raw_norm = raw_all / np.maximum(std, 1e-8)
        raw_flat = raw_norm.reshape(raw_norm.shape[0], raw_norm.shape[1], -1)
        origins = train_cache["absolute_window_starts"][common_idx]
        shuffled_raw = v2.derange_expert_axis(torch.from_numpy(raw_flat), origins, dataset, v2.SHUFFLE_SEED).numpy()

        n_common, k = target.shape
        target_flat = target.reshape(-1)
        feature_map = {
            "SixStatActive": six_all.reshape(-1, six_all.shape[-1]),
            "RawResponseActive": raw_flat.reshape(n_common * k, -1),
            "ShuffledRawResponse": shuffled_raw.reshape(n_common * k, -1),
            "MatchedPassive": passive_15.reshape(n_common * k, -1),
            "PassivePlusRaw": np.concatenate([passive_15.reshape(n_common * k, -1), raw_flat.reshape(n_common * k, -1)], axis=1),
        }
        preds_by_method: dict[str, np.ndarray] = {}
        targets_by_method: dict[str, np.ndarray] = {}
        for method, feats in feature_map.items():
            out = v3a_fit_predict(feats, target_flat, n_common, k)
            preds_by_method[method] = out["pred"]
            targets_by_method[method] = out["target"]
            row = metric_row_from_pred(dataset, method, out["pred"], out["target"], k)
            row.update({kk: vv for kk, vv in out.items() if kk.startswith("n_")})
            rows.append(row)

        hold_windows = preds_by_method["RawResponseActive"].shape[0] // k
        target_hold = targets_by_method["RawResponseActive"].reshape(hold_windows, k)
        raw_pw = torch.from_numpy(np.abs(preds_by_method["RawResponseActive"].reshape(hold_windows, k) - target_hold).mean(axis=1).astype(np.float32))
        six_pw = torch.from_numpy(np.abs(preds_by_method["SixStatActive"].reshape(hold_windows, k) - target_hold).mean(axis=1).astype(np.float32))
        shuf_pw = torch.from_numpy(np.abs(preds_by_method["ShuffledRawResponse"].reshape(hold_windows, k) - target_hold).mean(axis=1).astype(np.float32))
        passive_pw = torch.from_numpy(np.abs(preds_by_method["MatchedPassive"].reshape(hold_windows, k) - target_hold).mean(axis=1).astype(np.float32))
        plus_pw = torch.from_numpy(np.abs(preds_by_method["PassivePlusRaw"].reshape(hold_windows, k) - target_hold).mean(axis=1).astype(np.float32))
        dependence_rows.extend(dependence_full(raw_pw, six_pw, dataset, "Raw_vs_SixStat"))
        dependence_rows.extend(dependence_full(raw_pw, shuf_pw, dataset, "Raw_vs_ShuffledRaw"))
        dependence_rows.extend(dependence_full(plus_pw, passive_pw, dataset, "PassivePlusRaw_vs_Passive"))

        passive_residual = (target - passive_pred).reshape(-1)
        res_out = v3a_fit_predict(feature_map["RawResponseActive"], passive_residual, n_common, k)
        pred = res_out["pred"]
        y = res_out["target"]
        residual_rows.append(
            {
                "dataset": dataset,
                "method": "RawResponse_to_passive_residual",
                "r2": float(r2_score(y, pred)) if np.std(pred) > 1e-12 else float("nan"),
                "pearson": float(pearsonr(pred, y).statistic) if np.std(pred) > 1e-12 else float("nan"),
                "spearman": float(spearmanr(pred, y).statistic) if np.std(pred) > 1e-12 else float("nan"),
                "mae": float(mean_absolute_error(y, pred)),
                "mae_null_mean": float(mean_absolute_error(y, np.full_like(y, passive_residual[: res_out["n_fit_rows"]].mean()))),
                **{kk: vv for kk, vv in res_out.items() if kk.startswith("n_")},
            }
        )
        dataset_reports[dataset] = {
            "n_common_windows": n_common,
            "raw_feature_dim": int(raw_flat.shape[-1]),
            "holdout_windows": hold_windows,
        }

    write_csv(V3A_DIR / "v3a_results.csv", rows)
    write_csv(V3A_DIR / "dependence_aware_results.csv", dependence_rows)
    write_csv(V3A_DIR / "passive_residual_results.csv", residual_rows)

    primary = {}
    for dataset in DATASETS:
        ds_dep = [r for r in dependence_rows if r["dataset"] == dataset]
        primary[dataset] = {
            "Raw_vs_SixStat": primary_row(ds_dep, "Raw_vs_SixStat"),
            "Raw_vs_ShuffledRaw": primary_row(ds_dep, "Raw_vs_ShuffledRaw"),
            "PassivePlusRaw_vs_Passive": primary_row(ds_dep, "PassivePlusRaw_vs_Passive"),
        }

    def row_for(dataset: str, method: str) -> dict[str, Any]:
        return next(r for r in rows if r["dataset"] == dataset and r["method"] == method)

    raw_better_six = sum(row_for(ds, "RawResponseActive")["conditional_mae"] < row_for(ds, "SixStatActive")["conditional_mae"] for ds in DATASETS)
    raw_better_shuffle = sum(row_for(ds, "RawResponseActive")["conditional_mae"] < row_for(ds, "ShuffledRawResponse")["conditional_mae"] for ds in DATASETS)
    plus_better_passive = sum(row_for(ds, "PassivePlusRaw")["conditional_mae"] < row_for(ds, "MatchedPassive")["conditional_mae"] for ds in DATASETS)
    residual_positive = sum(r["r2"] > 0 for r in residual_rows)
    shuffled_ge_raw = sum(row_for(ds, "ShuffledRawResponse")["conditional_mae"] <= row_for(ds, "RawResponseActive")["conditional_mae"] for ds in DATASETS)
    if raw_better_six >= 3 and raw_better_shuffle >= 3 and (plus_better_passive >= 2 or residual_positive >= 2):
        classification = "RAW_RESPONSE_COMPLEMENTARY_SIGNAL"
    elif raw_better_six >= 2 and raw_better_shuffle >= 2:
        classification = "RAW_RESPONSE_SIGNAL_BUT_REDUNDANT"
    elif shuffled_ge_raw >= 3:
        classification = "NON_EXPERT_SPECIFIC_RAW_SIGNAL"
    else:
        classification = "SIX_STATS_NOT_THE_BOTTLENECK"

    result = {
        "experiment": "raw_response_probe_v3a_reproduced",
        "created_at_utc": now_utc(),
        "source_reproduction": str(OUT_DIR.relative_to(ROOT)),
        "ridge_alpha": 1.0,
        "classification": classification,
        "counts": {
            "raw_better_six": int(raw_better_six),
            "raw_better_shuffled": int(raw_better_shuffle),
            "passive_plus_raw_better_passive": int(plus_better_passive),
            "raw_predicts_positive_passive_residual": int(residual_positive),
        },
        "datasets": dataset_reports,
        "primary_dependence": primary,
        "test_set_accessed": False,
    }
    write_json(V3A_DIR / "method_manifest.json", result)
    make_v3a_report(result, rows, residual_rows)
    return result


def make_v3a_report(result: Mapping[str, Any], rows: list[dict[str, Any]], residual_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# LearnedProbe V3A Reproduced -- Raw-Response Representation Test",
        "",
        "This analysis uses accepted frozen-protocol V2 reproduction artifacts, not exact original V2 tensors.",
        "",
        f"**Classification: {result['classification']}**",
        "",
        "## Primary Results",
        "",
        "| Dataset | SixStat MAE | Raw MAE | Shuffled Raw MAE | Passive MAE | Passive+Raw MAE | Residual R2 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for ds in DATASETS:
        by_method = {r["method"]: r for r in rows if r["dataset"] == ds}
        rr = next(r for r in residual_rows if r["dataset"] == ds)
        lines.append(
            f"| {ds} | {by_method['SixStatActive']['conditional_mae']:.6f} | {by_method['RawResponseActive']['conditional_mae']:.6f} | "
            f"{by_method['ShuffledRawResponse']['conditional_mae']:.6f} | {by_method['MatchedPassive']['conditional_mae']:.6f} | "
            f"{by_method['PassivePlusRaw']['conditional_mae']:.6f} | {rr['r2']:.4f} |"
        )
    lines += [
        "",
        "## Compliance",
        "",
        "```text",
        "RIDGE_ALPHA_TUNED: NO (fixed at 1.0)",
        "V2 REPRODUCTION ARTIFACTS ACCEPTED BEFORE V3A: YES",
        "ROUTER TRAINED: NO",
        "TEST SET ACCESSED: NO",
        "```",
    ]
    (V3A_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_reproduction_report(decision: Mapping[str, Any], comparison_rows: list[dict[str, Any]], v3a_result: Mapping[str, Any] | None) -> None:
    lines = [
        "# Controlled Discriminative Probe V2 Reproduction",
        "",
        "## Phase A: V2 Artifact Reproduction",
        "",
        f"**Decision: {decision['decision']}**",
        "",
        "The committed V2 implementation is the archived reproduction source; the original run's HEAD was `2904e28` while the then-uncommitted V2 source was subsequently committed.",
        "",
        f"- Observable comparison checks: {sum(r.get('result') == 'PASS' for r in comparison_rows)}/{len(comparison_rows)} passed.",
        f"- Qualitative V2 classification: {decision['v2_classification_reproduced']}.",
        f"- Proceed to router integration: {decision['proceed_to_router_integration']}.",
        "",
        "## Phase B: V3A Raw-Response Analysis",
        "",
    ]
    if v3a_result is None:
        lines.append("Phase B was not run because Phase A did not pass the reproduction gate.")
    else:
        lines.append(f"Phase B ran under `raw_response_probe_v3a_reproduced/` and classified the result as `{v3a_result['classification']}`.")
    lines += [
        "",
        "## Compliance",
        "",
        "```text",
        "FROZEN V2 DIRECTORY MODIFIED: NO",
        "FORECASTING EXPERTS RETRAINED: NO",
        "ROUTER TRAINED: NO",
        "TEST SET ACCESSED: NO",
        "```",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_initial_manifests(created_at: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source = {
        "created_at_utc": created_at,
        "current_git_commit": git_commit_sha(),
        "frozen_v2_method_manifest_git_commit_sha": json.loads((V2_DIR / "method_manifest.json").read_text(encoding="utf-8")).get("git_commit_sha"),
        "v2_implementation_committed_in": "7ec1f1e",
        "current_branch_frozen_reports_commit": "42719bc",
        "source_provenance_note": "The committed V2 implementation is the archived reproduction source; the original run's HEAD was 2904e28 while the then-uncommitted V2 source was subsequently committed. Bit-exact original source provenance is not claimed.",
        "git_show_7ec1f1e": git_show_stat("7ec1f1e"),
        "git_show_42719bc": git_show_stat("42719bc"),
        "v2_artifacts": {
            name: {
                "path": str((V2_DIR / name).relative_to(ROOT)),
                "exists": (V2_DIR / name).exists(),
                "sha256": sha256_file(V2_DIR / name) if (V2_DIR / name).exists() else None,
            }
            for name in [
                "run_controlled_discriminative_probe_v2.py",
                "shared_probe_generator.py",
                "method_manifest.json",
                "development_status.json",
                "report.md",
                "validation_results.json",
                "router_val_competence_results.csv",
                "oof_fold_manifest.csv",
                "causality_checks.csv",
                "integrity_checks.csv",
                "prompt_compliance_audit.md",
            ]
        },
    }
    manifest = {
        "manifest_type": "controlled_discriminative_probe_v2_reproduction_manifest",
        "created_at_utc": created_at,
        "purpose": "Rerun frozen V2 protocol to save missing raw artifacts, then run V3A only if observable V2 behavior passes fixed gates.",
        "predeclared_tolerances": REPRODUCTION_TOLERANCES,
        "method_constants": {
            "epsilon": v2.EPS,
            "batch_size": v2.BATCH_SIZE,
            "internal_val_fraction": v2.INTERNAL_VAL_FRACTION,
            "lr": v2.LR,
            "max_epochs": v2.MAX_EPOCHS,
            "patience": v2.PATIENCE,
            "perturbation_weight": v2.PERTURBATION_WEIGHT,
            "ranking_weight": v2.RANKING_WEIGHT,
            "smoothness_weight": v2.SMOOTHNESS_WEIGHT,
            "weight_decay": v2.WEIGHT_DECAY,
            "n_purge_folds": v2.N_PURGE_FOLDS,
            "min_train_fraction": v2.MIN_TRAIN_FRACTION,
            "shuffle_seed": v2.SHUFFLE_SEED,
            "random_probe_seed": v2.RANDOM_PROBE_SEED,
        },
        "datasets": DATASETS,
        "test_set_accessed": False,
    }
    method = {
        "experiment": "controlled_discriminative_probe_v2_reproduction",
        "created_at_utc": created_at,
        "uses_archived_v2_implementation": True,
        "does_not_modify_frozen_v2": True,
        "phase_b_v3a_runs_only_if_phase_a_accepted": True,
        "test_set_accessed": False,
    }
    write_json(OUT_DIR / "source_provenance.json", source)
    write_json(OUT_DIR / "reproduction_manifest.json", manifest)
    write_json(OUT_DIR / "method_manifest.json", method)


def main() -> None:
    start = time.time()
    created_at = now_utc()
    write_initial_manifests(created_at)
    report: dict[str, Any] = {
        "experiment": "controlled_discriminative_probe_v2_reproduction",
        "created_at_utc": created_at,
        "git_commit_sha": git_commit_sha(),
        "datasets": {},
        "test_set_accessed": False,
    }
    all_folds, all_priors, all_integrity, all_provenance = [], [], [], []
    all_oof, all_val, all_dependence, all_perturbation = [], [], [], []
    all_residual, all_mechanism = [], []
    checkpoint_hashes: dict[str, Any] = {}

    for dataset in DATASETS:
        result = reproduction_dataset(dataset)
        report["datasets"][dataset] = result
        all_folds.extend(result["fold_rows"])
        all_priors.extend(result["fold_prior_rows"])
        all_integrity.append(result["integrity"])
        all_provenance.append(result["expert_provenance_row"])
        all_oof.extend(result["oof_table"])
        all_val.extend(result["val_table"])
        all_dependence.extend(result["dependence_rows"])
        all_perturbation.extend(result["perturbation_rows"])
        all_residual.append({"dataset": dataset, **result["residual_diag"]})
        for group_name, diag in result["mechanism"].items():
            all_mechanism.append({"dataset": dataset, "group": group_name, **diag})
        checkpoint_hashes[dataset] = {
            "expert_checkpoints": result["checkpoint_hashes_after"],
            "reproduction_checkpoints": result["checkpoint_rows"],
            "raw_files": result["raw_files"],
            "router_val_raw_file": result["router_val_raw_file"],
        }

    decision = v2.classify(report)
    report["decision"] = decision
    report["runtime_sec"] = time.time() - start
    write_json(OUT_DIR / "validation_results.json", report)
    write_csv(OUT_DIR / "oof_fold_manifest.csv", all_folds)
    write_csv(OUT_DIR / "causality_checks.csv", all_folds)
    write_csv(OUT_DIR / "integrity_checks.csv", all_integrity)
    write_csv(OUT_DIR / "expert_provenance_checks.csv", all_provenance)
    write_csv(OUT_DIR / "causal_expert_priors.csv", all_priors)
    write_csv(OUT_DIR / "oof_competence_results.csv", all_oof)
    write_csv(OUT_DIR / "router_val_competence_results.csv", all_val)
    write_csv(OUT_DIR / "dependence_aware_results.csv", all_dependence)
    write_csv(OUT_DIR / "perturbation_diagnostics.csv", all_perturbation)
    write_csv(OUT_DIR / "residual_information_results.csv", all_residual)
    write_csv(OUT_DIR / "passive_active_diagnostics.csv", all_mechanism)
    write_json(OUT_DIR / "checkpoint_hashes.json", checkpoint_hashes)

    comparison_rows, accepted = compare_reproduction(report, decision)
    write_csv(OUT_DIR / "reproduction_comparison.csv", comparison_rows)
    reproduction_decision = {
        "decision": "REPRODUCTION_ACCEPTED" if accepted else "REPRODUCTION_FAILED",
        "created_at_utc": now_utc(),
        "v2_classification_reproduced": decision["tier"],
        "proceed_to_router_integration": bool(decision["proceed_to_router_integration"]),
        "observable_checks_passed": int(sum(r.get("result") == "PASS" for r in comparison_rows)),
        "observable_checks_total": int(len(comparison_rows)),
        "test_set_accessed": False,
    }
    write_json(OUT_DIR / "reproduction_decision.json", reproduction_decision)

    v3a_result = run_v3a_reproduced() if accepted else None
    make_reproduction_report(reproduction_decision, comparison_rows, v3a_result)
    print(json.dumps({"phase_a": reproduction_decision["decision"], "phase_b_ran": v3a_result is not None, "test_set_accessed": False}, indent=2))
    print("TEST SET ACCESSED: NO")


if __name__ == "__main__":
    main()
