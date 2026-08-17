"""Pooled-router-train selection for Ridge and MLP residual correctors.

This uses the same meaning of "pooled" as pooled_router_train_core:

1. choose the configuration by MAE/MSE over all router_train windows together;
2. fit one final corrector on all router_train windows;
3. only then evaluate router_val and test.

The test split has already been viewed in prior authorized experiments, so all
test rows from this script are `after_final_test_audit` sensitivity results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.locked_etth1_config_etth2_replication import run_locked_etth1_config_etth2_replication as etth2_locked  # noqa: E402
from experiments.residual_correction_costar import run_residual_correction_experiments as etth1_resid  # noqa: E402


OUT_DIR = ROOT / "experiments" / "pooled_router_train_residual_correctors"
ALL_RESULTS_CSV = ROOT / "experiments" / "all_results_summary" / "all_costar_results.csv"
LABEL = "after_final_test_audit"
SEEDS = (7, 11, 13, 17, 19)

ETTH1_CORE = ("PatchTST", "iTransformer", "TimesNet")
ETTH2_CORE = ("DLinear", "PatchTST", "ModernTCN")

ETTH1_TRAIN_CACHE = ROOT / "cache" / "costarts_walkforward" / "router_train_20_60_cache.pt"
ETTH1_VAL_CACHE = ROOT / "cache" / "costarts_walkforward" / "router_val_60_80_cache.pt"
ETTH1_TEST_CACHE = ROOT / "experiments" / "final_test_evaluation" / "generated" / "caches" / "ETTh1" / "test_80_100_cache.pt"
ETTH1_NORMALIZER = ROOT / "checkpoints" / "costarts_walkforward" / "final_60" / "DLinear" / "best_expert.pt"
ETTH1_DATASET_DIR = ROOT / "datasets" / "ETTh1"

ANCHORS = {
    "ETTh1": {
        "fixed_core": {"validation_mae": 0.36726489663124084, "validation_mse": 0.3105303645133972, "test_mae": 0.3271281123161316, "test_mse": 0.26658302545547485},
        "full_adaptive": {"validation_mae": 0.3631121516227722, "validation_mse": 0.30605703592300415, "test_mae": 0.3263927400112152, "test_mse": 0.2675056755542755},
        "existing_ridge": {"validation_mae": 0.36330097913742065, "validation_mse": 0.3062863349914551, "test_mae": 0.32644808292388916, "test_mse": 0.2674521803855896},
        "existing_mlp": {"validation_mae": 0.3633176386356354, "validation_mse": 0.306606650352478, "test_mae": 0.32604682445526123, "test_mse": 0.2673218250274658},
    },
    "ETTh2": {
        "fixed_core": {"validation_mae": 0.2808783948421478, "validation_mse": 0.17193281650543213, "test_mae": 0.30464205145835876, "test_mse": 0.22518493235111237},
        "full_adaptive": {"validation_mae": 0.27683213353157043, "validation_mse": 0.16727977991104126, "test_mae": 0.29780814051628113, "test_mse": 0.21861204504966736},
        "existing_ridge": {"validation_mae": 0.2750360667705536, "validation_mse": 0.1656191200017929, "test_mae": 0.29678699374198914, "test_mse": 0.21771253645420074},
        "existing_mlp": {"validation_mae": 0.2756431996822357, "validation_mse": 0.16614720225334167, "test_mae": 0.29704129695892334, "test_mse": 0.21814896166324615},
    },
}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_tensor(tensor: torch.Tensor) -> str:
    arr = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def refuse_test_path(path: Path) -> None:
    if "test" in str(path).lower():
        raise ValueError(f"Refusing test path before manifest: {path}")


def load_etth1_cache(path: Path, role: str, allow_test: bool = False) -> dict[str, Any]:
    if not allow_test:
        refuse_test_path(path)
        return etth1_resid.load_cache(path, role)
    cache = torch.load(path, map_location="cpu", weights_only=False)
    actual = cache.get("cache_role", cache.get("split_role"))
    if actual != role:
        raise ValueError(f"{path}: role={actual!r}, expected {role!r}")
    return cache


def load_etth1_series_prefix() -> torch.Tensor:
    train = torch.from_numpy(np.load(ETTH1_DATASET_DIR / "train_data.npy")).to(torch.float32)
    val = torch.from_numpy(np.load(ETTH1_DATASET_DIR / "val_data.npy")).to(torch.float32)
    return torch.cat((train, val), dim=0)


def load_etth1_series_with_test() -> torch.Tensor:
    return torch.cat(
        (
            torch.from_numpy(np.load(ETTH1_DATASET_DIR / "train_data.npy")).to(torch.float32),
            torch.from_numpy(np.load(ETTH1_DATASET_DIR / "val_data.npy")).to(torch.float32),
            torch.from_numpy(np.load(ETTH1_DATASET_DIR / "test_data.npy")).to(torch.float32),
        ),
        dim=0,
    )


def load_etth2_series_with_test() -> torch.Tensor:
    return torch.cat(
        (
            etth2_locked.load_series_prefix(),
            torch.from_numpy(np.load(etth2_locked.DATASET_DIR / "test_data.npy")).to(torch.float32),
        ),
        dim=0,
    )


def metric_tensors(dataset: str, cache: Mapping[str, Any], pred: torch.Tensor, std: torch.Tensor) -> dict[str, Any]:
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.bool)
    if dataset == "ETTh1":
        mae = etth1_resid.sample_mae(pred, target, mask, std)
        mse = etth1_resid.sample_mse(pred, target, mask, std)
    else:
        mae = etth2_locked.sample_mae(pred, target, mask, std)
        mse = etth2_locked.sample_mse(pred, target, mask, std)
    return {"mae": float(mae.mean()), "mse": float(mse.mean()), "per_window_mae": mae, "per_window_mse": mse}


def paired_bootstrap_ci(candidate: torch.Tensor, anchor: torch.Tensor, n: int = 5000, seed: int = 20260814) -> dict[str, Any]:
    diff = (candidate.detach().cpu().numpy() - anchor.detach().cpu().numpy()).astype(np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.shape[0], size=(n, diff.shape[0]))
    means = diff[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {"mean_diff": float(diff.mean()), "ci95_low": float(lo), "ci95_high": float(hi), "ci95_excludes_zero": bool(hi < 0 or lo > 0)}


def fixed_core_prediction(cache: Mapping[str, Any], core: Sequence[str]) -> torch.Tensor:
    names = list(cache["expert_names"])
    idx = [names.index(name) for name in core]
    return cache["prediction_stack"][..., idx].to(torch.float32).mean(dim=-1)


def etth1_ridge_grid() -> list[etth1_resid.RidgeConfig]:
    return etth1_resid.build_ridge_grid()


def etth2_ridge_grid() -> list[etth2_locked.RidgeConfig]:
    return [
        etth2_locked.RidgeConfig(ridge=1.0, alpha=0.10, clip_multiple=0.25),
        etth2_locked.RidgeConfig(ridge=1.0, alpha=0.05, clip_multiple=0.25),
        etth2_locked.RidgeConfig(ridge=10.0, alpha=0.10, clip_multiple=0.25),
        etth2_locked.RidgeConfig(ridge=1.0, alpha=0.10, clip_multiple=0.50),
        etth2_locked.RidgeConfig(ridge=1.0, alpha=0.25, clip_multiple=0.25),
    ]


def mlp_grid(dataset: str) -> list[Any]:
    cls = etth1_resid.MlpConfig if dataset == "ETTh1" else etth2_locked.MlpConfig
    return [
        cls(seed=7, alpha=0.10, clip_multiple=0.25, hidden=64, lr=3e-4, weight_decay=1e-2, epochs=40, patience=6),
        cls(seed=7, alpha=0.05, clip_multiple=0.25, hidden=64, lr=3e-4, weight_decay=1e-2, epochs=40, patience=6),
        cls(seed=7, alpha=0.10, clip_multiple=0.50, hidden=64, lr=3e-4, weight_decay=1e-2, epochs=40, patience=6),
        cls(seed=7, alpha=0.10, clip_multiple=0.25, hidden=64, lr=3e-4, weight_decay=3e-2, epochs=40, patience=6),
        cls(seed=7, alpha=0.10, clip_multiple=0.25, hidden=32, lr=3e-4, weight_decay=1e-2, epochs=40, patience=6),
    ]


def build_features(
    dataset: str,
    cache: Mapping[str, Any],
    baseline: torch.Tensor,
    std: torch.Tensor,
    series: torch.Tensor,
    init_residuals_norm: torch.Tensor | None,
    allow_test_history: bool,
) -> tuple[torch.Tensor, list[str], dict[str, Any]]:
    starts = cache["absolute_window_starts"].to(torch.long)
    if dataset == "ETTh1":
        return etth1_resid.build_feature_tensor(cache, starts, baseline, std, series, init_residuals_norm=init_residuals_norm)
    return etth2_locked.build_feature_tensor(cache, starts, baseline, std, series, init_residuals_norm=init_residuals_norm, allow_test_history=allow_test_history)


def flattened_targets(dataset: str, cache: Mapping[str, Any], baseline: torch.Tensor, std: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if dataset == "ETTh1":
        return etth1_resid.flattened_targets(cache, baseline, std)
    return etth2_locked.flattened_targets(cache, baseline, std)


def fit_scaler(dataset: str, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return etth1_resid.fit_scaler(x) if dataset == "ETTh1" else etth2_locked.fit_scaler(x)


def apply_scaler(dataset: str, x: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return etth1_resid.apply_scaler(x, mean, scale) if dataset == "ETTh1" else etth2_locked.apply_scaler(x, mean, scale)


def fit_ridge(dataset: str, x: torch.Tensor, y: torch.Tensor, ridge: float) -> torch.Tensor:
    return etth1_resid.fit_ridge(x, y, ridge) if dataset == "ETTh1" else etth2_locked.fit_ridge(x, y, ridge)


def predict_linear(dataset: str, x: torch.Tensor, coef: torch.Tensor) -> torch.Tensor:
    return etth1_resid.predict_linear(x, coef) if dataset == "ETTh1" else etth2_locked.predict_linear(x, coef)


def apply_residual_delta(dataset: str, baseline: torch.Tensor, delta: torch.Tensor, std: torch.Tensor, alpha: float, clip: float | None, resid_std: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    if dataset == "ETTh1":
        return etth1_resid.apply_residual_delta(baseline, delta, std, alpha, clip, resid_std)
    return etth2_locked.apply_residual_delta(baseline, delta, std, alpha, clip, resid_std)


def tiny_mlp(dataset: str, dim: int, hidden: int) -> torch.nn.Module:
    return etth1_resid.TinyResidualMlp(dim, hidden) if dataset == "ETTh1" else etth2_locked.TinyResidualMlp(dim, hidden)


def set_seed(dataset: str, seed: int) -> None:
    if dataset == "ETTh1":
        etth1_resid.set_seed(seed)
    else:
        etth2_locked.set_seed(seed)


def prepare_train_matrix(dataset: str, train_cache: Mapping[str, Any], train_baseline: torch.Tensor, std: torch.Tensor, series: torch.Tensor) -> dict[str, Any]:
    x_all, names, stat_extra = build_features(dataset, train_cache, train_baseline, std, series, None, allow_test_history=False)
    y_all, mask = flattened_targets(dataset, train_cache, train_baseline, std)
    residual_norm = (train_cache["targets"].to(torch.float32) - train_baseline) / std.view(1, 1, -1)
    return {
        "x_all": x_all,
        "x_fit": x_all[mask],
        "y_fit": y_all[mask],
        "mask": mask,
        "feature_names": names,
        "stat_extra": stat_extra,
        "residual_norm": residual_norm,
        "residual_std": residual_norm.std(dim=0, unbiased=False).clamp_min(1e-6),
    }


def evaluate_train_delta(dataset: str, train_cache: Mapping[str, Any], train_baseline: torch.Tensor, std: torch.Tensor, delta_flat: torch.Tensor, config: Any, residual_std: torch.Tensor) -> tuple[dict[str, Any], dict[str, Any]]:
    pred, extra = apply_residual_delta(dataset, train_baseline, delta_flat, std, float(config.alpha), config.clip_multiple, residual_std)
    return metric_tensors(dataset, train_cache, pred, std), extra


def fit_pooled_ridge(dataset: str, config: Any, prep: Mapping[str, Any]) -> dict[str, Any]:
    mean, scale = fit_scaler(dataset, prep["x_fit"])
    coef = fit_ridge(dataset, apply_scaler(dataset, prep["x_fit"], mean, scale), prep["y_fit"], float(config.ridge))
    return {
        "config": asdict(config),
        "feature_names": prep["feature_names"],
        "feature_mean": mean,
        "feature_scale": scale,
        "coef": coef,
        "residual_train_std_norm": prep["residual_std"],
        "train_residual_norm": prep["residual_norm"],
        "pooled_selection_rule": "lowest router_train pooled MAE; pooled MSE tie-breaker",
        "pooled_final_fit": True,
    }


def predict_ridge_artifact(dataset: str, cache: Mapping[str, Any], baseline: torch.Tensor, artifact: Mapping[str, Any], std: torch.Tensor, series: torch.Tensor, allow_test_history: bool) -> tuple[torch.Tensor, dict[str, Any]]:
    x, names, stat_extra = build_features(dataset, cache, baseline, std, series, artifact["train_residual_norm"], allow_test_history)
    if names != artifact["feature_names"]:
        raise RuntimeError(f"{dataset} ridge feature names changed")
    delta = predict_linear(dataset, apply_scaler(dataset, x, artifact["feature_mean"], artifact["feature_scale"]), artifact["coef"])
    pred, extra = apply_residual_delta(dataset, baseline, delta, std, float(artifact["config"]["alpha"]), artifact["config"]["clip_multiple"], artifact["residual_train_std_norm"])
    return pred, {**extra, **stat_extra}


def score_ridge_grid(dataset: str, train_cache: Mapping[str, Any], train_baseline: torch.Tensor, std: torch.Tensor, prep: Mapping[str, Any], configs: Sequence[Any]) -> tuple[list[dict[str, Any]], Any, dict[str, Any]]:
    rows = []
    best_key = None
    best_config = None
    best_artifact = None
    for config in configs:
        artifact = fit_pooled_ridge(dataset, config, prep)
        delta = predict_linear(dataset, apply_scaler(dataset, prep["x_all"], artifact["feature_mean"], artifact["feature_scale"]), artifact["coef"])
        train_metrics, extra = evaluate_train_delta(dataset, train_cache, train_baseline, std, delta, config, prep["residual_std"])
        row = {
            "dataset": dataset,
            "method": "Ridge residual corrector",
            "config_name": config.name,
            **asdict(config),
            "router_train_mae": train_metrics["mae"],
            "router_train_mse": train_metrics["mse"],
            **extra,
        }
        rows.append(row)
        key = (train_metrics["mae"], train_metrics["mse"])
        if best_key is None or key < best_key:
            best_key = key
            best_config = config
            best_artifact = artifact
    assert best_config is not None and best_artifact is not None
    return rows, best_config, best_artifact


def fit_pooled_mlp(dataset: str, config: Any, prep: Mapping[str, Any], device: torch.device, artifact_path: Path) -> dict[str, Any]:
    if artifact_path.exists():
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
        if artifact.get("config") == asdict(config) and artifact.get("pooled_final_fit") is True:
            return artifact

    set_seed(dataset, int(config.seed))
    mean, scale = fit_scaler(dataset, prep["x_fit"])
    x_fit = apply_scaler(dataset, prep["x_fit"], mean, scale)
    y_fit = prep["y_fit"]
    model = tiny_mlp(dataset, x_fit.shape[1], int(config.hidden)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(config.lr), weight_decay=float(config.weight_decay))
    generator = torch.Generator()
    generator.manual_seed(int(config.seed))
    loader = DataLoader(TensorDataset(x_fit, y_fit), batch_size=4096, shuffle=True, generator=generator)
    curve = []
    for epoch in range(1, int(config.epochs) + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = F.smooth_l1_loss(pred, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        curve.append({"epoch": epoch, "train_loss": float(statistics.mean(losses))})
    artifact = {
        "config": asdict(config),
        "feature_names": prep["feature_names"],
        "feature_mean": mean,
        "feature_scale": scale,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "residual_train_std_norm": prep["residual_std"],
        "train_residual_norm": prep["residual_norm"],
        "pooled_selection_rule": "lowest router_train pooled MAE; pooled MSE tie-breaker",
        "pooled_final_fit": True,
        "internal_router_train_holdout": 0,
        "epochs_trained": int(config.epochs),
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, artifact_path)
    write_csv(artifact_path.parent / "training_curve.csv", curve)
    return artifact


def predict_mlp_artifact(dataset: str, cache: Mapping[str, Any], baseline: torch.Tensor, artifact: Mapping[str, Any], std: torch.Tensor, series: torch.Tensor, device: torch.device, allow_test_history: bool) -> tuple[torch.Tensor, dict[str, Any]]:
    x, names, stat_extra = build_features(dataset, cache, baseline, std, series, artifact["train_residual_norm"], allow_test_history)
    if names != artifact["feature_names"]:
        raise RuntimeError(f"{dataset} MLP feature names changed")
    model = tiny_mlp(dataset, len(names), int(artifact["config"]["hidden"])).to(device)
    model.load_state_dict(artifact["state_dict"], strict=True)
    model.eval()
    x = apply_scaler(dataset, x, artifact["feature_mean"], artifact["feature_scale"])
    outs = []
    with torch.no_grad():
        for i in range(0, x.shape[0], 16384):
            outs.append(model(x[i : i + 16384].to(device)).cpu())
    delta = torch.cat(outs)
    pred, extra = apply_residual_delta(dataset, baseline, delta, std, float(artifact["config"]["alpha"]), artifact["config"]["clip_multiple"], artifact["residual_train_std_norm"])
    return pred, {**extra, **stat_extra}


def score_mlp_grid(dataset: str, train_cache: Mapping[str, Any], train_baseline: torch.Tensor, std: torch.Tensor, prep: Mapping[str, Any], configs: Sequence[Any], device: torch.device) -> tuple[list[dict[str, Any]], Any, dict[str, Any]]:
    rows = []
    best_key = None
    best_config = None
    best_artifact = None
    for config in configs:
        name = config_name("mlp", asdict(config))
        artifact_path = OUT_DIR / "artifacts" / dataset / "mlp_search" / name / f"seed_{config.seed}" / "mlp_artifact.pt"
        artifact = fit_pooled_mlp(dataset, config, prep, device, artifact_path)
        pred, extra = predict_mlp_artifact(dataset, train_cache, train_baseline, artifact, std, torch.empty(0), device, allow_test_history=False) if False else (None, None)
        x_all = apply_scaler(dataset, prep["x_all"], artifact["feature_mean"], artifact["feature_scale"])
        model = tiny_mlp(dataset, len(artifact["feature_names"]), int(config.hidden)).to(device)
        model.load_state_dict(artifact["state_dict"], strict=True)
        model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, x_all.shape[0], 16384):
                outs.append(model(x_all[i : i + 16384].to(device)).cpu())
        delta = torch.cat(outs)
        train_metrics, extra = evaluate_train_delta(dataset, train_cache, train_baseline, std, delta, config, prep["residual_std"])
        row = {
            "dataset": dataset,
            "method": "MLP residual corrector",
            "config_name": name,
            **asdict(config),
            "router_train_mae": train_metrics["mae"],
            "router_train_mse": train_metrics["mse"],
            **extra,
            "artifact_path": str(artifact_path),
        }
        rows.append(row)
        key = (train_metrics["mae"], train_metrics["mse"])
        if best_key is None or key < best_key:
            best_key = key
            best_config = config
            best_artifact = artifact
    assert best_config is not None and best_artifact is not None
    return rows, best_config, best_artifact


def config_name(prefix: str, cfg: Mapping[str, Any]) -> str:
    parts = []
    for key in sorted(cfg):
        if key == "seed":
            continue
        val = cfg[key]
        parts.append(f"{key}{val:g}" if isinstance(val, float) else f"{key}{val}")
    return (prefix + "_" + "_".join(parts)).replace(".", "p").replace("None", "none")


def save_artifact(path: Path, artifact: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(artifact), path)


def evaluate_dataset(
    dataset: str,
    train_cache: Mapping[str, Any],
    val_cache: Mapping[str, Any],
    test_cache: Mapping[str, Any],
    train_baseline: torch.Tensor,
    val_baseline_by_seed: Mapping[int, torch.Tensor],
    test_baseline_by_seed: Mapping[int, torch.Tensor],
    std: torch.Tensor,
    train_series: torch.Tensor,
    test_series: torch.Tensor,
    device: torch.device,
    bootstrap_samples: int,
) -> dict[str, Any]:
    prep = prepare_train_matrix(dataset, train_cache, train_baseline, std, train_series)
    ridge_configs = etth1_ridge_grid() if dataset == "ETTh1" else etth2_ridge_grid()
    ridge_rows, ridge_config, ridge_artifact = score_ridge_grid(dataset, train_cache, train_baseline, std, prep, ridge_configs)
    ridge_artifact_path = OUT_DIR / "artifacts" / dataset / "ridge_selected" / "ridge_artifact.pt"
    save_artifact(ridge_artifact_path, ridge_artifact)

    mlp_rows, mlp_config_seed7, _ = score_mlp_grid(dataset, train_cache, train_baseline, std, prep, mlp_grid(dataset), device)
    selected_mlp_cfgs = [type(mlp_config_seed7)(**{**asdict(mlp_config_seed7), "seed": seed}) for seed in SEEDS]
    mlp_artifacts = []
    for cfg in selected_mlp_cfgs:
        name = config_name("mlp_selected", asdict(cfg))
        artifact_path = OUT_DIR / "artifacts" / dataset / "mlp_selected" / name / f"seed_{cfg.seed}" / "mlp_artifact.pt"
        mlp_artifacts.append({"seed": int(cfg.seed), "path": artifact_path, "artifact": fit_pooled_mlp(dataset, cfg, prep, device, artifact_path)})

    val_base_mean = torch.stack([val_baseline_by_seed[s] for s in SEEDS]).mean(dim=0)
    test_base_mean = torch.stack([test_baseline_by_seed[s] for s in SEEDS]).mean(dim=0)

    ridge_val_pred, ridge_val_extra = predict_ridge_artifact(dataset, val_cache, val_base_mean, ridge_artifact, std, train_series, allow_test_history=False)
    ridge_test_pred, ridge_test_extra = predict_ridge_artifact(dataset, test_cache, test_base_mean, ridge_artifact, std, test_series, allow_test_history=True)
    ridge_val_metrics = metric_tensors(dataset, val_cache, ridge_val_pred, std)
    ridge_test_metrics = metric_tensors(dataset, test_cache, ridge_test_pred, std)

    mlp_val_preds = []
    mlp_test_preds = []
    seed_rows = []
    for item in mlp_artifacts:
        seed = item["seed"]
        val_pred, val_extra = predict_mlp_artifact(dataset, val_cache, val_baseline_by_seed[seed], item["artifact"], std, train_series, device, allow_test_history=False)
        test_pred, test_extra = predict_mlp_artifact(dataset, test_cache, test_baseline_by_seed[seed], item["artifact"], std, test_series, device, allow_test_history=True)
        val_met = metric_tensors(dataset, val_cache, val_pred, std)
        test_met = metric_tensors(dataset, test_cache, test_pred, std)
        mlp_val_preds.append(val_pred)
        mlp_test_preds.append(test_pred)
        seed_rows.extend(
            [
                {"dataset": dataset, "method": "MLP residual corrector", "split": "router_val", "seed": seed, "mae": val_met["mae"], "mse": val_met["mse"], **val_extra},
                {"dataset": dataset, "method": "MLP residual corrector", "split": "test", "seed": seed, "mae": test_met["mae"], "mse": test_met["mse"], **test_extra},
            ]
        )
    mlp_val_pred = torch.stack(mlp_val_preds).mean(dim=0)
    mlp_test_pred = torch.stack(mlp_test_preds).mean(dim=0)
    mlp_val_metrics = metric_tensors(dataset, val_cache, mlp_val_pred, std)
    mlp_test_metrics = metric_tensors(dataset, test_cache, mlp_test_pred, std)

    fixed_test = metric_tensors(dataset, test_cache, fixed_core_prediction(test_cache, ETTH1_CORE if dataset == "ETTh1" else ETTH2_CORE), std)
    fixed_val = metric_tensors(dataset, val_cache, fixed_core_prediction(val_cache, ETTH1_CORE if dataset == "ETTh1" else ETTH2_CORE), std)
    # Full-adaptive anchor per-window for ETTh2 is the residual base; for ETTh1
    # only scalar anchor diffs are available here without rerunning specialists.
    full_anchor_test_per_window = metric_tensors(dataset, test_cache, test_base_mean, std)["per_window_mae"]

    ci_rows = []
    for method, metrics, pred in [
        ("Ridge residual corrector", ridge_test_metrics, ridge_test_pred),
        ("MLP residual corrector", mlp_test_metrics, mlp_test_pred),
    ]:
        ci_rows.append({"dataset": dataset, "method": method, "anchor": "fixed_core", **paired_bootstrap_ci(metrics["per_window_mae"], fixed_test["per_window_mae"], bootstrap_samples)})
        ci_rows.append({"dataset": dataset, "method": method, "anchor": "residual_base_full_adaptive_for_ETTh2_or_HV_for_ETTh1", **paired_bootstrap_ci(metrics["per_window_mae"], full_anchor_test_per_window, bootstrap_samples)})

    result_rows = []
    for method, config, val_metrics, test_metrics, extra in [
        ("Ridge residual corrector", ridge_config, ridge_val_metrics, ridge_test_metrics, ridge_test_extra),
        ("MLP residual corrector", mlp_config_seed7, mlp_val_metrics, mlp_test_metrics, {"seed_mae_mean": statistics.mean([r["mae"] for r in seed_rows if r["split"] == "test"]), "seed_mae_std": statistics.pstdev([r["mae"] for r in seed_rows if r["split"] == "test"])}),
    ]:
        key = "existing_ridge" if method.startswith("Ridge") else "existing_mlp"
        result_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "variant": "pooled_router_train_selection",
                "expert_set": "+".join(ETTH1_CORE if dataset == "ETTh1" else ETTH2_CORE),
                "selected_config": config.name if hasattr(config, "name") else config_name("mlp", asdict(config)),
                "router_train_selection_rule": "pooled router_train MAE, pooled MSE tie-breaker; no folds",
                "router_train_mae": min([r["router_train_mae"] for r in (ridge_rows if method.startswith("Ridge") else mlp_rows)]),
                "router_train_mse": min([r["router_train_mse"] for r in (ridge_rows if method.startswith("Ridge") else mlp_rows) if r["router_train_mae"] == min(rr["router_train_mae"] for rr in (ridge_rows if method.startswith("Ridge") else mlp_rows))]),
                "validation_mae": val_metrics["mae"],
                "validation_mse": val_metrics["mse"],
                "test_mae": test_metrics["mae"],
                "test_mse": test_metrics["mse"],
                "diff_vs_existing_corrector_mae": test_metrics["mae"] - ANCHORS[dataset][key]["test_mae"],
                "diff_vs_fixed_core_mae": test_metrics["mae"] - ANCHORS[dataset]["fixed_core"]["test_mae"],
                "diff_vs_full_adaptive_mae": test_metrics["mae"] - ANCHORS[dataset]["full_adaptive"]["test_mae"],
                "label": LABEL,
                **extra,
            }
        )

    return {
        "dataset": dataset,
        "ridge_grid_rows": ridge_rows,
        "mlp_grid_rows": mlp_rows,
        "selected": {
            "ridge": {"config": asdict(ridge_config), "artifact_path": str(ridge_artifact_path), "artifact_sha256": sha256_file(ridge_artifact_path)},
            "mlp": {"config": asdict(mlp_config_seed7), "artifact_paths": [str(x["path"]) for x in mlp_artifacts], "artifact_sha256": {str(x["seed"]): sha256_file(x["path"]) for x in mlp_artifacts}},
        },
        "result_rows": result_rows,
        "seed_rows": seed_rows,
        "ci_rows": ci_rows,
        "anchor_validation": {"fixed_core": fixed_val["mae"]},
    }


def append_all_results(rows: Sequence[Mapping[str, Any]]) -> None:
    existing = read_csv(ALL_RESULTS_CSV)
    fields = sorted(set().union(*(r.keys() for r in existing), *(r.keys() for r in rows)))
    keys = {(r.get("dataset"), r.get("method"), r.get("result_group"), r.get("status"), r.get("source_file")) for r in existing}
    new_rows = []
    for row in rows:
        out = {
            "dataset": row["dataset"],
            "method": f"Pooled-selection {row['method']}",
            "expert_set": row["expert_set"],
            "result_group": "pooled_router_train_residual_correctors",
            "status": LABEL,
            "selection_protocol": row["router_train_selection_rule"],
            "validation_mae": row["validation_mae"],
            "validation_mse": row["validation_mse"],
            "test_mae": row["test_mae"],
            "test_mse": row["test_mse"],
            "source": str(OUT_DIR / "pooled_router_train_residual_results.csv"),
            "source_file": str(OUT_DIR / "pooled_router_train_residual_results.csv"),
            "comparison_anchor": "existing corresponding residual corrector",
            "diff_vs_anchor": row["diff_vs_existing_corrector_mae"],
        }
        key = (out.get("dataset"), out.get("method"), out.get("result_group"), out.get("status"), out.get("source_file"))
        if key not in keys:
            new_rows.append({f: out.get(f, "") for f in fields})
            keys.add(key)
    if not new_rows:
        return
    with ALL_RESULTS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for row in existing:
            writer.writerow({f: row.get(f, "") for f in fields})
        writer.writerows(new_rows)


def select_dataset(
    dataset: str,
    train_cache: Mapping[str, Any],
    train_baseline: torch.Tensor,
    std: torch.Tensor,
    train_series: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    prep = prepare_train_matrix(dataset, train_cache, train_baseline, std, train_series)
    ridge_configs = etth1_ridge_grid() if dataset == "ETTh1" else etth2_ridge_grid()
    ridge_rows, ridge_config, ridge_artifact = score_ridge_grid(dataset, train_cache, train_baseline, std, prep, ridge_configs)
    ridge_artifact_path = OUT_DIR / "artifacts" / dataset / "ridge_selected" / "ridge_artifact.pt"
    save_artifact(ridge_artifact_path, ridge_artifact)

    mlp_rows, mlp_config_seed7, _ = score_mlp_grid(dataset, train_cache, train_baseline, std, prep, mlp_grid(dataset), device)
    selected_mlp_cfgs = [type(mlp_config_seed7)(**{**asdict(mlp_config_seed7), "seed": seed}) for seed in SEEDS]
    mlp_artifacts = []
    for cfg in selected_mlp_cfgs:
        name = config_name("mlp_selected", asdict(cfg))
        artifact_path = OUT_DIR / "artifacts" / dataset / "mlp_selected" / name / f"seed_{cfg.seed}" / "mlp_artifact.pt"
        artifact = fit_pooled_mlp(dataset, cfg, prep, device, artifact_path)
        mlp_artifacts.append({"seed": int(cfg.seed), "path": artifact_path, "artifact": artifact})

    ridge_selected_row = min(ridge_rows, key=lambda r: (r["router_train_mae"], r["router_train_mse"]))
    mlp_selected_row = min(mlp_rows, key=lambda r: (r["router_train_mae"], r["router_train_mse"]))
    return {
        "prep": prep,
        "ridge_grid_rows": ridge_rows,
        "mlp_grid_rows": mlp_rows,
        "selected": {
            "ridge": {
                "config": asdict(ridge_config),
                "config_name": ridge_config.name,
                "router_train_mae": ridge_selected_row["router_train_mae"],
                "router_train_mse": ridge_selected_row["router_train_mse"],
                "artifact_path": str(ridge_artifact_path),
                "artifact_sha256": sha256_file(ridge_artifact_path),
            },
            "mlp": {
                "config": asdict(mlp_config_seed7),
                "config_name": config_name("mlp", asdict(mlp_config_seed7)),
                "router_train_mae": mlp_selected_row["router_train_mae"],
                "router_train_mse": mlp_selected_row["router_train_mse"],
                "artifact_paths": [str(x["path"]) for x in mlp_artifacts],
                "artifact_sha256": {str(x["seed"]): sha256_file(x["path"]) for x in mlp_artifacts},
            },
        },
        "ridge_artifact": ridge_artifact,
        "mlp_artifacts": mlp_artifacts,
    }


def evaluate_frozen_dataset(
    dataset: str,
    val_cache: Mapping[str, Any],
    test_cache: Mapping[str, Any],
    val_baseline_by_seed: Mapping[int, torch.Tensor],
    test_baseline_by_seed: Mapping[int, torch.Tensor],
    std: torch.Tensor,
    train_series: torch.Tensor,
    test_series: torch.Tensor,
    selected_bundle: Mapping[str, Any],
    device: torch.device,
    bootstrap_samples: int,
) -> dict[str, Any]:
    ridge_artifact = selected_bundle["ridge_artifact"]
    mlp_artifacts = selected_bundle["mlp_artifacts"]
    selected = selected_bundle["selected"]
    core = ETTH1_CORE if dataset == "ETTh1" else ETTH2_CORE

    val_base_mean = torch.stack([val_baseline_by_seed[s] for s in SEEDS]).mean(dim=0)
    test_base_mean = torch.stack([test_baseline_by_seed[s] for s in SEEDS]).mean(dim=0)

    ridge_val_pred, ridge_val_extra = predict_ridge_artifact(dataset, val_cache, val_base_mean, ridge_artifact, std, train_series, allow_test_history=False)
    ridge_test_pred, ridge_test_extra = predict_ridge_artifact(dataset, test_cache, test_base_mean, ridge_artifact, std, test_series, allow_test_history=True)
    ridge_val_metrics = metric_tensors(dataset, val_cache, ridge_val_pred, std)
    ridge_test_metrics = metric_tensors(dataset, test_cache, ridge_test_pred, std)

    seed_rows = []
    mlp_val_preds = []
    mlp_test_preds = []
    for item in mlp_artifacts:
        seed = int(item["seed"])
        artifact = item["artifact"]
        val_pred, val_extra = predict_mlp_artifact(dataset, val_cache, val_baseline_by_seed[seed], artifact, std, train_series, device, allow_test_history=False)
        test_pred, test_extra = predict_mlp_artifact(dataset, test_cache, test_baseline_by_seed[seed], artifact, std, test_series, device, allow_test_history=True)
        val_met = metric_tensors(dataset, val_cache, val_pred, std)
        test_met = metric_tensors(dataset, test_cache, test_pred, std)
        mlp_val_preds.append(val_pred)
        mlp_test_preds.append(test_pred)
        seed_rows.extend(
            [
                {"dataset": dataset, "method": "MLP residual corrector", "split": "router_val", "seed": seed, "mae": val_met["mae"], "mse": val_met["mse"], **val_extra},
                {"dataset": dataset, "method": "MLP residual corrector", "split": "test", "seed": seed, "mae": test_met["mae"], "mse": test_met["mse"], **test_extra},
            ]
        )
    mlp_val_pred = torch.stack(mlp_val_preds).mean(dim=0)
    mlp_test_pred = torch.stack(mlp_test_preds).mean(dim=0)
    mlp_val_metrics = metric_tensors(dataset, val_cache, mlp_val_pred, std)
    mlp_test_metrics = metric_tensors(dataset, test_cache, mlp_test_pred, std)

    fixed_test = metric_tensors(dataset, test_cache, fixed_core_prediction(test_cache, core), std)
    full_anchor_test_per_window = metric_tensors(dataset, test_cache, test_base_mean, std)["per_window_mae"]

    ci_rows = []
    for method, metrics in [
        ("Ridge residual corrector", ridge_test_metrics),
        ("MLP residual corrector", mlp_test_metrics),
    ]:
        ci_rows.append({"dataset": dataset, "method": method, "anchor": "fixed_core", **paired_bootstrap_ci(metrics["per_window_mae"], fixed_test["per_window_mae"], bootstrap_samples)})
        ci_rows.append({"dataset": dataset, "method": method, "anchor": "residual_base_full_adaptive_for_ETTh2_or_HV_for_ETTh1", **paired_bootstrap_ci(metrics["per_window_mae"], full_anchor_test_per_window, bootstrap_samples)})

    result_rows = []
    for method, val_metrics, test_metrics, extra, train_key in [
        ("Ridge residual corrector", ridge_val_metrics, ridge_test_metrics, ridge_test_extra, "ridge"),
        (
            "MLP residual corrector",
            mlp_val_metrics,
            mlp_test_metrics,
            {
                "seed_mae_mean": statistics.mean([r["mae"] for r in seed_rows if r["split"] == "test"]),
                "seed_mae_std": statistics.pstdev([r["mae"] for r in seed_rows if r["split"] == "test"]),
            },
            "mlp",
        ),
    ]:
        existing_key = "existing_ridge" if method.startswith("Ridge") else "existing_mlp"
        result_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "variant": "pooled_router_train_selection",
                "expert_set": "+".join(core),
                "selected_config": selected[train_key]["config_name"],
                "router_train_selection_rule": "pooled router_train MAE, pooled MSE tie-breaker; no folds",
                "router_train_mae": selected[train_key]["router_train_mae"],
                "router_train_mse": selected[train_key]["router_train_mse"],
                "validation_mae": val_metrics["mae"],
                "validation_mse": val_metrics["mse"],
                "test_mae": test_metrics["mae"],
                "test_mse": test_metrics["mse"],
                "diff_vs_existing_corrector_mae": test_metrics["mae"] - ANCHORS[dataset][existing_key]["test_mae"],
                "diff_vs_fixed_core_mae": test_metrics["mae"] - ANCHORS[dataset]["fixed_core"]["test_mae"],
                "diff_vs_full_adaptive_mae": test_metrics["mae"] - ANCHORS[dataset]["full_adaptive"]["test_mae"],
                "label": LABEL,
                **extra,
            }
        )

    return {"result_rows": result_rows, "seed_rows": seed_rows, "ci_rows": ci_rows}


def render_report(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Pooled Router-Train Residual Correctors",
        "",
        f"Label: `{LABEL}`",
        "",
        "Pooled means configuration selection used all router-train windows as one block, with no chronological folds.",
        "",
        "## Results",
        "",
        "| Dataset | Method | Selected config | Train MAE | Val MAE | Test MAE | Test MSE | Diff vs existing | Diff vs full adaptive |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["results"]:
        lines.append(
            f"| {row['dataset']} | {row['method']} | `{row['selected_config']}` | "
            f"{row['router_train_mae']:.6f} | {row['validation_mae']:.6f} | {row['test_mae']:.6f} | {row['test_mse']:.6f} | "
            f"{row['diff_vs_existing_corrector_mae']:+.6f} | {row['diff_vs_full_adaptive_mae']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is an after-final-test sensitivity audit and does not supersede preregistered final-test rows.",
            "- Ridge selection is deterministic.",
            "- MLP config selection used seed 7 on pooled router-train; the selected config was then refit for seeds 7, 11, 13, 17, and 19.",
            "- Validation and test were loaded only after selected configs and artifacts were recorded in `manifest_before_test.json`.",
            "",
            "## Reproduce",
            "",
            "```powershell",
            "python experiments\\pooled_router_train_residual_correctors\\run_pooled_router_train_residual_correctors.py --device cuda",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    args = parser.parse_args()

    start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    # Train/selection phase: no validation or test cache is needed for the
    # pooled config choice.
    etth1_train = load_etth1_cache(ETTH1_TRAIN_CACHE, "router_train_20_60")
    etth1_std = etth1_resid.load_std(ETTH1_NORMALIZER, int(etth1_train["num_features"]))
    etth1_series = load_etth1_series_prefix()
    etth1_train_base, _ = etth1_resid.fixed_current_best_prediction(etth1_train, etth1_train, etth1_std, 7, device)

    etth2_train = etth2_locked.load_cache(etth2_locked.TRAIN_CACHE, "router_train")
    etth2_locked.validate_cache_shape(etth2_train, "router_train")
    etth2_std = torch.ones(int(etth2_train["num_features"]), dtype=torch.float32)
    etth2_series = etth2_locked.load_series_prefix()
    etth2_train_base, _ = etth2_locked.build_train_baseline(etth2_train, etth2_std)

    etth1_selected = select_dataset("ETTh1", etth1_train, etth1_train_base, etth1_std, etth1_series, device)
    etth2_selected = select_dataset("ETTh2", etth2_train, etth2_train_base, etth2_std, etth2_series, device)

    train_phase_manifest = {
        "label": LABEL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_loaded_before_manifest": False,
        "validation_loaded_before_manifest": False,
        "pooled_definition": "configuration selected by pooled router_train MAE/MSE, no folds",
        "device": str(device),
        "datasets": {
            "ETTh1": {
                "core": list(ETTH1_CORE),
                "base_forecast": "horizon-variable hybrid fixed_current_best_prediction",
                "ridge_search_configs": [asdict(c) for c in etth1_ridge_grid()],
                "mlp_search_configs": [asdict(c) for c in mlp_grid("ETTh1")],
                "cache_hashes": {"router_train": sha256_file(ETTH1_TRAIN_CACHE), "router_val": sha256_file(ETTH1_VAL_CACHE)},
                "test_planned_after_manifest": str(ETTH1_TEST_CACHE),
            },
            "ETTh2": {
                "core": list(ETTH2_CORE),
                "base_forecast": "full adaptive train-selected fixed-three residual base",
                "ridge_search_configs": [asdict(c) for c in etth2_ridge_grid()],
                "mlp_search_configs": [asdict(c) for c in mlp_grid("ETTh2")],
                "cache_hashes": {"router_train": sha256_file(etth2_locked.TRAIN_CACHE), "router_val": sha256_file(etth2_locked.VAL_CACHE)},
                "test_planned_after_manifest": str(etth2_locked.TEST_CACHE),
            },
        },
        "selected_before_validation_or_test": {"ETTh1": etth1_selected["selected"], "ETTh2": etth2_selected["selected"]},
    }
    write_json(OUT_DIR / "manifest_before_test.json", train_phase_manifest)

    # Frozen evaluation phase after manifest.
    etth1_val = load_etth1_cache(ETTH1_VAL_CACHE, "router_val_60_80")
    etth1_val_base_by_seed = {seed: etth1_resid.fixed_current_best_prediction(etth1_val, etth1_train, etth1_std, seed, device)[0] for seed in SEEDS}
    etth1_test = load_etth1_cache(ETTH1_TEST_CACHE, "test_80_100", allow_test=True)
    etth1_series_test = load_etth1_series_with_test()
    etth1_test_base_by_seed = {seed: etth1_resid.fixed_current_best_prediction(etth1_test, etth1_train, etth1_std, seed, device)[0] for seed in SEEDS}

    etth2_val = etth2_locked.load_cache(etth2_locked.VAL_CACHE, "router_val")
    etth2_locked.validate_cache_shape(etth2_val, "router_val")
    etth2_val_base, _ = etth2_locked.build_eval_baseline(etth2_val, etth2_train, etth2_std)
    etth2_val_base_by_seed = {seed: etth2_val_base for seed in SEEDS}
    etth2_test = etth2_locked.load_cache(etth2_locked.TEST_CACHE, "locked_test", allow_test=True)
    etth2_locked.validate_cache_shape(etth2_test, "locked_test")
    etth2_series_test = load_etth2_series_with_test()
    etth2_test_base, _ = etth2_locked.build_eval_baseline(etth2_test, etth2_train, etth2_std)
    etth2_test_base_by_seed = {seed: etth2_test_base for seed in SEEDS}

    etth1_eval = evaluate_frozen_dataset(
        "ETTh1",
        etth1_val,
        etth1_test,
        etth1_val_base_by_seed,
        etth1_test_base_by_seed,
        etth1_std,
        etth1_series,
        etth1_series_test,
        etth1_selected,
        device,
        args.bootstrap_samples,
    )
    etth2_eval = evaluate_frozen_dataset(
        "ETTh2",
        etth2_val,
        etth2_test,
        etth2_val_base_by_seed,
        etth2_test_base_by_seed,
        etth2_std,
        etth2_series,
        etth2_series_test,
        etth2_selected,
        device,
        args.bootstrap_samples,
    )

    results = [*etth1_eval["result_rows"], *etth2_eval["result_rows"]]
    write_csv(OUT_DIR / "ridge_pooled_train_grid.csv", [*etth1_selected["ridge_grid_rows"], *etth2_selected["ridge_grid_rows"]])
    write_csv(OUT_DIR / "mlp_pooled_train_grid.csv", [*etth1_selected["mlp_grid_rows"], *etth2_selected["mlp_grid_rows"]])
    write_csv(OUT_DIR / "pooled_router_train_residual_results.csv", results)
    write_csv(OUT_DIR / "seed_results.csv", [*etth1_eval["seed_rows"], *etth2_eval["seed_rows"]])
    write_csv(OUT_DIR / "paired_ci.csv", [*etth1_eval["ci_rows"], *etth2_eval["ci_rows"]])

    payload = {
        "label": LABEL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_sec": time.time() - start,
        "device": str(device),
        "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated(device) / (1024 * 1024)) if device.type == "cuda" else 0.0,
        "test_loaded_after_manifest": True,
        "pooled_definition": "configuration selected by pooled router_train MAE/MSE, no folds",
        "selected": {"ETTh1": etth1_selected["selected"], "ETTh2": etth2_selected["selected"]},
        "results": results,
        "paired_ci": [*etth1_eval["ci_rows"], *etth2_eval["ci_rows"]],
        "cache_hashes_after_test": {"ETTh1_test": sha256_file(ETTH1_TEST_CACHE), "ETTh2_test": sha256_file(etth2_locked.TEST_CACHE)},
    }
    write_json(OUT_DIR / "POOLED_ROUTER_TRAIN_RESIDUAL_RESULTS.json", payload)
    (OUT_DIR / "POOLED_ROUTER_TRAIN_RESIDUAL_REPORT.md").write_text(render_report(payload), encoding="utf-8")
    append_all_results(results)


if __name__ == "__main__":
    main()
