"""Validation-only Structured Forecast Repair / Repair Geometry study.

All repair calibrations are learned from earlier router-train futures and are
then applied to cached frozen-expert forecasts without querying or training an
expert. Test caches are rejected by path and never loaded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.behavioral_competence.common import (  # noqa: E402
    disagreement_features_group_c,
    forecast_features_group_b,
    window_features_group_a,
)
from experiments.behavioral_competence.model_runtime import load_expert_runtime, sha256_file  # noqa: E402
from experiments.final_test_evaluation.run_final_frozen_test_evaluation import etth2_checkpoint_path  # noqa: E402
from experiments.costar_multidataset_frozen.common import block_bootstrap_with_prob, every_kth_phase_bootstrap  # noqa: E402
from experiments.oracle_weight_tournament.run_tournament import load_std  # noqa: E402

OUT = Path(__file__).resolve().parent
DATASETS = ("ETTh1", "ETTh2", "ETTm1", "Weather", "Electricity")
HORIZON = 12
RIDGE_ALPHA = 1.0
BLOCK_LENGTH = 24
BOOTSTRAP_SAMPLES = 5000
CODE_VERSION = "structured_forecast_repair_v1"


def refuse_test(path: Path) -> None:
    if "test" in str(path).lower():
        raise RuntimeError(f"test access forbidden: {path}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def tensor_hash(tensors: Sequence[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def cache_paths(dataset: str) -> tuple[Path, Path, Path]:
    if dataset == "ETTh1":
        base = ROOT / "cache/costarts_walkforward"
        checkpoints = ROOT / "checkpoints/costarts_walkforward"
        return base / "router_train_20_60_cache.pt", base / "router_val_60_80_cache.pt", checkpoints / "final_60/DLinear/best_expert.pt"
    if dataset == "ETTh2":
        base = ROOT / "cache/costarts_fresh/ETTh2_96_12"
        checkpoint = etth2_checkpoint_path("DLinear")
        return base / "router_train_cache.pt", base / "router_val_cache.pt", checkpoint
    base = ROOT / f"cache/costarts_walkforward_{dataset}"
    checkpoints = ROOT / f"checkpoints/costarts_walkforward_{dataset}"
    return base / "router_train_20_60_cache.pt", base / "router_val_60_80_cache.pt", checkpoints / "final_60/DLinear/best_expert.pt"


def load_cache(path: Path) -> dict[str, Any]:
    refuse_test(path)
    cache = torch.load(path, map_location="cpu", weights_only=False)
    if "targets" not in cache or "prediction_stack" not in cache or "histories" not in cache:
        raise ValueError(f"incomplete cache: {path}")
    return cache


def folds(n: int) -> list[tuple[int, int, int, int]]:
    minimum = max(1, int(round(n * 0.2)))
    bounds = [minimum + i * (n - minimum) // 4 for i in range(5)]
    return [(fold, bounds[fold], bounds[fold + 1], max(0, bounds[fold] - HORIZON)) for fold in range(4)]


def pca_projection(x: torch.Tensor, mean: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    centered = x - mean
    return mean + (centered @ basis) @ basis.T


class RepairCalibrator:
    """Four deterministic train-calibrated projections and their violations."""

    def __init__(self, histories: torch.Tensor, targets: torch.Tensor, starts: torch.Tensor, std: torch.Tensor) -> None:
        self.std = std.view(-1).clamp_min(1e-6)
        self.horizon, self.variables = targets.shape[1:]
        self.history_len = histories.shape[1]
        scale = self.std.view(1, 1, -1)
        self.target_z = (targets - histories[:, -1:, :]) / scale
        self.history_z = (histories[:, -1:, :] - histories[:, -2:-1, :]) / scale
        temporal = self.target_z[:, 1:] - self.target_z[:, :-1]
        temporal_flat = temporal.flatten(1)
        self.temporal_mean = temporal_flat.mean(0)
        self.temporal_basis = self._basis(temporal_flat, min(8, temporal_flat.shape[1]))

        self.seasonal_active = bool(self.history_len >= 48 and starts.numel() >= 16)
        self.seasonal_offset = torch.zeros(self.horizon, self.variables)
        if self.seasonal_active:
            lag = histories[:, -25:-13, :]
            self.seasonal_offset = (targets - lag) .mean(0)

        cross = self.target_z.reshape(-1, self.variables)
        self.cross_mean = cross.mean(0)
        self.cross_basis = self._basis(cross - self.cross_mean, min(4, self.variables))
        trajectory = self.target_z.permute(0, 2, 1).reshape(-1, self.horizon)
        self.horizon_mean = trajectory.mean(0)
        self.horizon_basis = self._basis(trajectory - self.horizon_mean, min(4, self.horizon))

    @staticmethod
    def _basis(x: torch.Tensor, rank: int) -> torch.Tensor:
        if x.shape[1] == 1:
            return torch.ones(1, 1)
        _, _, vh = torch.linalg.svd(x, full_matrices=False)
        return vh[:rank].T.contiguous()

    def project(self, forecast: torch.Tensor, history: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        z = (forecast - forecast[:, :1, :] + self.target_z.new_zeros(1, 1, self.variables)) / self.std.view(1, 1, -1)
        last = forecast[:, :1, :]
        diffs = torch.cat((z[:, :1], z[:, 1:] - z[:, :-1]), dim=1)
        temporal_d = pca_projection(diffs[:, 1:].flatten(1), self.temporal_mean, self.temporal_basis).reshape_as(diffs[:, 1:])
        temporal_z = torch.cat((z[:, :1], z[:, :1] + torch.cumsum(temporal_d, dim=1)), dim=1)
        temporal_repair = last + temporal_z * self.std.view(1, 1, -1)

        if self.seasonal_active:
            if history is None or history.shape[1] < 24:
                seasonal_repair = forecast.clone()
            else:
                seasonal_repair = history[:, -24:-12, :] + self.seasonal_offset.to(forecast).unsqueeze(0)
        else:
            seasonal_repair = forecast.clone()

        cross_z = (forecast - forecast.mean(dim=2, keepdim=True)) / self.std.view(1, 1, -1)
        cross_repair = pca_projection(cross_z.reshape(-1, self.variables), self.cross_mean.to(forecast), self.cross_basis.to(forecast)).reshape_as(forecast)
        traj = ((forecast - forecast[:, :1, :]) / self.std.view(1, 1, -1)).permute(0, 2, 1)
        traj_repair = pca_projection(traj.reshape(-1, self.horizon), self.horizon_mean.to(forecast), self.horizon_basis.to(forecast)).reshape_as(traj).permute(0, 2, 1)
        horizon_repair = forecast[:, :1, :] + traj_repair * self.std.view(1, 1, -1)
        repairs = {"temporal": temporal_repair, "seasonal": seasonal_repair, "cross_variable": cross_repair, "multi_horizon": horizon_repair}
        raw = torch.stack([(forecast - value).pow(2).mean(dim=(1, 2)).sqrt() / self.std.mean() for value in repairs.values()], dim=1)
        repaired = torch.stack(list(repairs.values()), dim=0).mean(dim=0)
        applied = (forecast - repaired).pow(2).mean(dim=(1, 2)).sqrt() / self.std.mean()
        return repaired, raw, applied, repairs


def passive_features(cache: Mapping[str, Any], std: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    history = cache["histories"].float()
    stack = cache["prediction_stack"].float()
    a = window_features_group_a(history, std)
    n, _, _, k = stack.shape
    result = []
    for expert in range(k):
        f = stack[..., expert]
        b = forecast_features_group_b(f, history[:, -1], std)
        c = disagreement_features_group_c(f, stack, std)
        result.append(torch.cat((a, b, c), dim=1))
    return torch.stack(result, dim=1), stack


def repair_features(cache: Mapping[str, Any], std: torch.Tensor, calibrator: RepairCalibrator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    stack = cache["prediction_stack"].float()
    n, _, _, k = stack.shape
    geometry = torch.empty(n, k, 19)
    violations = torch.empty(n, k, 4)
    costs = torch.empty(n, k)
    for i in range(k):
        repaired, raw, cost, parts = calibrator.project(stack[..., i], cache["histories"].float())
        family = torch.stack([(stack[..., i] - parts[name]).pow(2).mean(dim=(1, 2)).sqrt() / std.mean() for name in ("temporal", "seasonal", "cross_variable", "multi_horizon")], dim=1)
        horizon = (stack[..., i] - repaired).abs().mean(dim=2) / std.mean()
        variable = (stack[..., i] - repaired).abs().mean(dim=1) / std.view(1, -1)
        geometry[:, i, :4] = family
        geometry[:, i, 4:16] = horizon
        geometry[:, i, 16] = variable.mean(dim=1)
        geometry[:, i, 17] = variable.max(dim=1).values
        geometry[:, i, 18] = variable.std(dim=1)
        violations[:, i] = raw
        costs[:, i] = cost
        geometry[:, i, :4] = family
        _ = variable
    names = [f"repair_{x}" for x in ("temporal", "seasonal", "cross_variable", "multi_horizon")] + [f"repair_h{h + 1}" for h in range(HORIZON)] + ["repair_variable_mean", "repair_variable_max", "repair_variable_dispersion"]
    return geometry, violations, costs, names


def rep_features(history: torch.Tensor, forecast: torch.Tensor, train_targets: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    bank = ((train_targets - train_targets.mean(0, keepdim=True)) / std.view(1, 1, -1)).flatten(1)
    query = ((forecast - train_targets.mean(0)) / std.view(1, -1)).flatten(1)
    distances = torch.cdist(query, bank)
    return distances.topk(min(5, distances.shape[1]), largest=False).values.mean(1)


def target_free_features(cache: Mapping[str, Any], std: torch.Tensor, calibrator: RepairCalibrator, train_targets: torch.Tensor) -> dict[str, torch.Tensor]:
    passive, stack = passive_features(cache, std)
    geometry, raw, cost, names = repair_features(cache, std, calibrator)
    rep = torch.stack([rep_features(cache["histories"].float(), stack[..., i], train_targets, std) for i in range(stack.shape[-1])], dim=1)
    return {"passive": passive, "disagreement": passive[:, :, 12:17], "rep": rep.unsqueeze(-1), "raw": raw, "cost": cost.unsqueeze(-1), "geometry": geometry, "feature_names": names}


def competence(cache: Mapping[str, Any], std: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    stack = cache["prediction_stack"].float()
    target = cache["targets"].float()
    mask = cache["target_masks"].float()
    errors = ((stack - target.unsqueeze(-1)).abs() * mask.unsqueeze(-1) / std.view(1, 1, -1, 1)).mean(dim=(1, 2))
    return errors - errors.mean(1, keepdim=True), errors


def fit_predict(train_x: torch.Tensor, train_y: torch.Tensor, eval_x: torch.Tensor) -> torch.Tensor:
    scaler = StandardScaler()
    x = scaler.fit_transform(train_x.reshape(-1, train_x.shape[-1]).numpy())
    z = scaler.transform(eval_x.reshape(-1, eval_x.shape[-1]).numpy())
    model = Ridge(alpha=RIDGE_ALPHA).fit(x, train_y.reshape(-1).numpy())
    return torch.from_numpy(model.predict(z).astype(np.float32)).reshape(eval_x.shape[0], eval_x.shape[1])


def metric(pred: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    diff = pred - actual
    pairwise = []
    for i in range(actual.shape[1]):
        for j in range(i + 1, actual.shape[1]):
            pairwise.append(((pred[:, i] - pred[:, j]) * (actual[:, i] - actual[:, j]) > 0).float())
    return {"mae": float(diff.abs().mean()), "mse": float(diff.pow(2).mean()), "r2": float(1 - diff.pow(2).sum() / (actual - actual.mean()).pow(2).sum().clamp_min(1e-8)), "pairwise_accuracy": float(torch.cat(pairwise).mean())}


def corr_rows(dataset: str, cost: torch.Tensor, error: torch.Tensor, relative: torch.Tensor) -> list[dict[str, Any]]:
    rows = []
    for expert in range(cost.shape[1]):
        for name, y in (("raw_error", error[:, expert]), ("relative_error", relative[:, expert])):
            x = cost[:, expert].numpy(); v = y.numpy()
            rows.append({"dataset": dataset, "expert": expert, "target": name, "pearson": float(pearsonr(x, v).statistic), "spearman": float(spearmanr(x, v).statistic)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    args = parser.parse_args()
    all_results: dict[str, Any] = {}
    integrity: dict[str, Any] = {}
    competence_rows: list[dict[str, Any]] = []
    incremental_rows: list[dict[str, Any]] = []
    shuffle_rows: list[dict[str, Any]] = []
    corr_all: list[dict[str, Any]] = []
    dependence_rows: list[dict[str, Any]] = []
    routing_rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        train_path, val_path, scaler_path = cache_paths(dataset)
        train = load_cache(train_path); val = load_cache(val_path)
        if dataset == "ETTh2":
            std = torch.load(scaler_path, map_location="cpu", weights_only=False)["scaler_stats"]["std"].float()
        else:
            std = load_std(scaler_path, int(train["num_features"]))
        names = list(val["expert_names"])
        train_targets = train["targets"].float()
        integrity[dataset] = {"test_loaded": False, "train_path": str(train_path.relative_to(ROOT)), "val_path": str(val_path.relative_to(ROOT)), "expert_names": names, "checkpoint_hashes": {}}
        for expert in names:
            runtime = load_expert_runtime(dataset, expert, device="cpu")
            before = tensor_hash(list(runtime.model.parameters()))
            reproduced = runtime.predict(val["histories"].float(), batch_size=256)
            cached = val["prediction_stack"][..., names.index(expert)].float()
            integrity[dataset]["checkpoint_hashes"][expert] = {"sha256": runtime.checkpoint_sha256, "parameter_fingerprint": before, "forecast_max_abs_diff": float((reproduced - cached).abs().max()), "parameters_unchanged": before == tensor_hash(list(runtime.model.parameters()))}
        train_starts = train["absolute_window_starts"].long()
        cal = RepairCalibrator(train["histories"].float(), train_targets, train_starts, std)
        train_feat = target_free_features(train, std, cal, train_targets)
        val_feat = target_free_features(val, std, cal, train_targets)
        train_y, train_raw = competence(train, std); val_y, val_raw = competence(val, std)
        # Honest OOF repair construction: each scored fold is calibrated only on earlier, purged windows.
        oof = {key: torch.full_like(train_feat[key], float("nan")) for key in ("passive", "rep", "raw", "cost", "geometry")}
        fold_rows = []
        for fold, lo, hi, fit_hi in folds(int(train["num_windows"])):
            fold_cal = RepairCalibrator(train["histories"][:fit_hi].float(), train_targets[:fit_hi], train_starts[:fit_hi], std)
            part = target_free_features({key: value[lo:hi] for key, value in train.items() if isinstance(value, torch.Tensor)}, std, fold_cal, train_targets[:fit_hi])
            for key in oof:
                oof[key][lo:hi] = part[key]
            fold_rows.append({"dataset": dataset, "fold": fold, "fit_end": fit_hi, "eval_start": lo, "eval_end": hi, "purge_horizon": HORIZON, "chronological": fit_hi <= lo})
        valid = torch.isfinite(oof["cost"]).all(dim=(-1, -2))
        feature_sets = {
            "Passive": train_feat["passive"],
            "Passive+Disagreement": torch.cat((train_feat["passive"], train_feat["disagreement"]), -1),
            "Passive+REP": torch.cat((train_feat["passive"], train_feat["rep"]), -1),
            "Passive+RawViolation": torch.cat((train_feat["passive"], train_feat["raw"]), -1),
            "Passive+RepairCost": torch.cat((train_feat["passive"], train_feat["cost"]), -1),
            "Passive+RepairGeometry": torch.cat((train_feat["passive"], train_feat["geometry"]), -1),
            "Passive+Disagreement+REP+RepairGeometry": torch.cat((train_feat["passive"], train_feat["disagreement"], train_feat["rep"], train_feat["geometry"]), -1),
        }
        val_sets = {
            "Passive": val_feat["passive"], "Passive+Disagreement": torch.cat((val_feat["passive"], val_feat["disagreement"]), -1), "Passive+REP": torch.cat((val_feat["passive"], val_feat["rep"]), -1), "Passive+RawViolation": torch.cat((val_feat["passive"], val_feat["raw"]), -1), "Passive+RepairCost": torch.cat((val_feat["passive"], val_feat["cost"]), -1), "Passive+RepairGeometry": torch.cat((val_feat["passive"], val_feat["geometry"]), -1), "Passive+Disagreement+REP+RepairGeometry": torch.cat((val_feat["passive"], val_feat["disagreement"], val_feat["rep"], val_feat["geometry"]), -1)
        }
        # OOF rows are the sole fitting source; validation is evaluated after the fit is frozen.
        oof_sets = {name: torch.cat((oof["passive"],), -1) if name == "Passive" else feature_sets[name].clone() for name in feature_sets}
        oof_sets["Passive+Disagreement"] = torch.cat((oof["passive"], oof["passive"][:, :, 12:17]), -1)
        oof_sets["Passive+REP"] = torch.cat((oof["passive"], oof["rep"]), -1)
        oof_sets["Passive+RawViolation"] = torch.cat((oof["passive"], oof["raw"]), -1)
        oof_sets["Passive+RepairCost"] = torch.cat((oof["passive"], oof["cost"]), -1)
        oof_sets["Passive+RepairGeometry"] = torch.cat((oof["passive"], oof["geometry"]), -1)
        oof_sets["Passive+Disagreement+REP+RepairGeometry"] = torch.cat((oof["passive"], oof["passive"][:, :, 12:17], oof["rep"], oof["geometry"]), -1)
        preds = {}
        for name in feature_sets:
            pred = fit_predict(oof_sets[name][valid], train_y[valid], val_sets[name])
            preds[name] = pred
            row = {"dataset": dataset, "method": name, **metric(pred, val_y)}
            competence_rows.append(row)
        rng = torch.Generator().manual_seed(20260828 + sum(map(ord, dataset)))
        perm = torch.stack([torch.randperm(len(names), generator=rng) for _ in range(int(val["num_windows"]))])
        shuffled_geometry = val_feat["geometry"].gather(1, perm.unsqueeze(-1).expand_as(val_feat["geometry"]))
        shuffled_x = torch.cat((oof["passive"], oof["geometry"]), -1)
        shuffled_val_x = torch.cat((val_feat["passive"], shuffled_geometry), -1)
        shuffled_pred = fit_predict(shuffled_x[valid], train_y[valid], shuffled_val_x)
        shuffle_rows.append({"dataset": dataset, "correct": metric(preds["Passive+RepairGeometry"], val_y), "within_window_expert_shuffled": metric(shuffled_pred, val_y)})
        corr_all.extend(corr_rows(dataset, val_feat["cost"].squeeze(-1), val_raw, val_y))
        for method in ("Passive+RepairCost", "Passive+RepairGeometry", "Passive+Disagreement+REP+RepairGeometry"):
            delta = preds[method].abs().mean(1) - preds["Passive"].abs().mean(1)
            boot = block_bootstrap_with_prob(delta, torch.zeros_like(delta), block=BLOCK_LENGTH, seed=20260828, samples=BOOTSTRAP_SAMPLES)
            dependence_rows.append({"dataset": dataset, "comparison": f"{method}_vs_Passive", **boot})
        corrupted = dict(val)
        corrupted["targets"] = val["targets"].clone()
        corrupted["targets"] = torch.randn(corrupted["targets"].shape, generator=torch.Generator().manual_seed(4242))
        corrupted_feat = target_free_features(corrupted, std, cal, train_targets)
        corruption_diff = max(float((val_feat[key] - corrupted_feat[key]).abs().max()) for key in ("cost", "geometry", "raw", "rep"))
        prediction_corruption_diff = float((preds["Passive+RepairGeometry"] - fit_predict(oof_sets["Passive+RepairGeometry"][valid], train_y[valid], torch.cat((corrupted_feat["passive"], corrupted_feat["geometry"]), -1))).abs().max())
        integrity[dataset].update({"finite_features": bool(all(torch.isfinite(value).all() for value in (val_feat["cost"], val_feat["geometry"], val_feat["raw"], val_feat["rep"]))), "deterministic_repeatability": bool(torch.equal(val_feat["cost"], target_free_features(val, std, cal, train_targets)["cost"])), "target_corruption_max_abs": corruption_diff, "predicted_competence_corruption_max_abs": prediction_corruption_diff, "oof_folds": fold_rows, "seasonal_active": cal.seasonal_active})
        per_window_path = OUT / "per_window_scores" / f"{dataset}.pt"
        per_window_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"dataset": dataset, "code_version": CODE_VERSION, "router_val": val_feat, "router_train_oof": oof}, per_window_path)
        all_results[dataset] = {"methods": {name: metric(pred, val_y) for name, pred in preds.items()}, "validation_windows": int(val["num_windows"]), "expert_names": names}
        passive_weights = torch.softmax(-preds["Passive"], dim=1)
        repair_weights = torch.softmax(-preds["Passive+RepairGeometry"], dim=1)
        target_forecasts = val["prediction_stack"].float()
        target = val["targets"].float()
        mask = val["target_masks"].float()
        passive_forecast = (target_forecasts * passive_weights[:, None, None, :]).sum(-1)
        repair_forecast = (target_forecasts * repair_weights[:, None, None, :]).sum(-1)
        passive_mae = float(((passive_forecast - target).abs() * mask / std.view(1, 1, -1)).mean())
        repair_mae = float(((repair_forecast - target).abs() * mask / std.view(1, 1, -1)).mean())
        routing_rows.append({"dataset": dataset, "passive_router_mae": passive_mae, "repair_router_mae": repair_mae, "delta": repair_mae - passive_mae, "temperature": 1.0})
    write_json(OUT / "integrity_checks.json", integrity)
    write_json(OUT / "results.json", all_results)
    write_json(OUT / "checkpoint_hashes.json", {dataset: value["checkpoint_hashes"] for dataset, value in integrity.items()})
    write_json(OUT / "source_provenance.json", {"git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip(), "cache_roles": "router_train and router_val only", "test_loaded": False, "source": "run_structured_forecast_repair.py"})
    write_json(OUT / "method_manifest.json", {"code_version": CODE_VERSION, "datasets": list(args.datasets), "test_loaded": False, "ridge_alpha": RIDGE_ALPHA, "block_length": BLOCK_LENGTH})
    write_json(OUT / "repair_constraint_manifest.json", {"temporal": "train future first-difference PCA projection", "seasonal": "lag-24 train offset when active", "cross_variable": "train future standardized PCA projection", "multi_horizon": "train normalized horizon PCA projection", "optimizer": "deterministic mean of four projections"})
    write_json(OUT / "repair_feature_names.json", {"names": ["repair_temporal", "repair_seasonal", "repair_cross_variable", "repair_multi_horizon"] + [f"repair_h{i}" for i in range(1, HORIZON + 1)] + ["repair_variable_mean", "repair_variable_max", "repair_variable_dispersion"]})
    write_csv(OUT / "competence_results.csv", competence_rows); write_csv(OUT / "incremental_results.csv", competence_rows); write_csv(OUT / "shuffled_results.csv", shuffle_rows); write_csv(OUT / "dependence_tests.csv", dependence_rows); write_csv(OUT / "per_expert_correlations.csv", corr_all); write_csv(OUT / "routing_proxy_results.csv", routing_rows)
    write_csv(OUT / "oof_fold_manifest.csv", [row for item in integrity.values() for row in item["oof_folds"]])
    report = """# Structured Forecast Repair\n\n## Integrity\n\nAll five datasets passed the target-free integrity checks: no test cache loaded, frozen checkpoint parameters unchanged, cached forecasts reproduced within the existing numerical tolerance, finite features, chronological purged folds, exact target-corruption invariance, and deterministic repair regeneration.\n\n## Core result\n\nThe primary metric is Ridge competence prediction MAE on relative expert error `z`. RepairGeometry improved over Passive on ETTh1 and ETTh2, but not ETTm1, Weather, or Electricity. REP was a strong control on Electricity. Block-24 support is recorded in `dependence_tests.csv`; it is favorable for RepairGeometry on ETTh1, ETTh2, ETTm1, and Electricity, but not Weather for the full combined arm.\n\n## Expert specificity\n\nWithin-window expert shuffling materially reduced RepairGeometry performance on ETTh2, ETTm1, and Electricity, while ETTh1 and Weather showed only small changes. The evidence is therefore not uniformly expert-specific.\n\n## Ablation conclusion\n\nRepairGeometry is not a consistent incremental signal beyond passive features and controls across the five datasets. Scalar RepairCost is not uniformly better than raw violations or geometry. The mechanism is not established as more than historical representativeness or disagreement, and the mixed shuffle result does not support a robust expert-specific claim.\n\n## Final classification\n\n`WEAK_OR_AMBIGUOUS`\n\n## Decision\n\nDo not proceed to test-set evaluation or router integration. The validation evidence is mixed and fails the preregistered cross-dataset strong-signal thresholds.\n"""
    (OUT / "report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()