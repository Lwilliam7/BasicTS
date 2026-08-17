"""ETTh2 validation-tuned versions of ETTh1-to-ETTh2 missing methods.

This sweep is intentionally small and validation-tuned:

- Train on ETTh2 router_train.
- Use ETTh2 router_val to select hyperparameters and checkpoints.
- Do not load ETTh2 test until tuned winners are frozen in
  `tuned_manifest_before_test.json`.

These rows are labeled `etth2_validation_tuned`. They are not preregistered and
are not untouched-test confirmation because ETTh2 test metrics are already known
from the earlier final evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import statistics
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.locked_etth1_config_etth2_replication.run_locked_etth1_config_etth2_replication import (  # noqa: E402
    CORE,
    DATASET_DIR,
    FULL_ADAPTIVE_TEST_MAE,
    LABEL as LOCKED_LABEL,
    OUT_DIR as LOCKED_OUT_DIR,
    SINGLE_DLINEAR_TEST_MAE,
    TEST_CACHE,
    TRAIN_CACHE,
    VAL_CACHE,
    DynamicConfig,
    DynamicDataset,
    Fixed3DynamicWeightRouter,
    MlpConfig,
    RidgeConfig,
    TinyResidualMlp,
    aggregate_seed_predictions,
    aggregate_seed_rows,
    apply_residual_delta,
    apply_scaler,
    build_chrono_oof_baseline,
    build_eval_baseline,
    build_feature_tensor,
    eval_dynamic,
    eval_oracle_model,
    fit_ridge,
    fit_scaler,
    flattened_targets,
    load_artifact,
    load_cache,
    load_series_prefix,
    metric_row,
    metrics,
    predict_dynamic,
    predict_linear,
    predict_oracle,
    predict_ridge,
    refuse_test_path,
    set_seed,
    sha256_file,
    sha256_tensor,
    validate_cache_shape,
    write_csv,
    write_json,
)
from experiments.oracle_weight_tournament.run_tournament import (  # noqa: E402
    Fixed3WindowDataset as OracleWindowDataset,
    TrialConfig as OracleTrialConfig,
    WeightStudent,
    args_global_weights,
    kmeans,
    oracle_weights_grid,
    sample_mae,
    weighted_forecast,
)
from experiments.etth2_train_selected_core.run_etth2_train_selected_core_eval import expert_indices, forecasts_for  # noqa: E402


OUT_DIR = ROOT / "experiments" / "etth2_validation_tuned_missing_methods"
FROZEN_RESULTS_DIR = ROOT / "experiments" / "frozen_model_test_results"
ALL_RESULTS_DIR = ROOT / "experiments" / "all_results_summary"
LABEL = "etth2_validation_tuned"
SEEDS = (7, 11, 13, 17, 19)


def declared_search_spaces() -> dict[str, list[dict[str, Any]]]:
    return {
        "ridge_residual_corrector": [
            asdict(RidgeConfig(ridge=1.0, alpha=0.10, clip_multiple=0.25)),
            asdict(RidgeConfig(ridge=1.0, alpha=0.05, clip_multiple=0.25)),
            asdict(RidgeConfig(ridge=10.0, alpha=0.10, clip_multiple=0.25)),
            asdict(RidgeConfig(ridge=1.0, alpha=0.10, clip_multiple=0.50)),
            asdict(RidgeConfig(ridge=1.0, alpha=0.25, clip_multiple=0.25)),
        ],
        "mlp_residual_corrector": [
            asdict(MlpConfig(seed=7, alpha=0.10, clip_multiple=0.25, hidden=64, lr=3e-4, weight_decay=1e-2, epochs=40, patience=6)),
            asdict(MlpConfig(seed=7, alpha=0.05, clip_multiple=0.25, hidden=64, lr=3e-4, weight_decay=1e-2, epochs=40, patience=6)),
            asdict(MlpConfig(seed=7, alpha=0.10, clip_multiple=0.50, hidden=64, lr=3e-4, weight_decay=1e-2, epochs=40, patience=6)),
            asdict(MlpConfig(seed=7, alpha=0.10, clip_multiple=0.25, hidden=64, lr=3e-4, weight_decay=3e-2, epochs=40, patience=6)),
            asdict(MlpConfig(seed=7, alpha=0.10, clip_multiple=0.25, hidden=32, lr=3e-4, weight_decay=1e-2, epochs=40, patience=6)),
        ],
        "oracle_prototype_residual": [
            {"teacher_lambda": 0.01, "num_prototypes": 16, "residual_scale": 0.30, "residual_weight": 0.001, "epochs": 10},
            {"teacher_lambda": 0.01, "num_prototypes": 8, "residual_scale": 0.30, "residual_weight": 0.001, "epochs": 10},
            {"teacher_lambda": 0.01, "num_prototypes": 32, "residual_scale": 0.30, "residual_weight": 0.001, "epochs": 10},
            {"teacher_lambda": 0.01, "num_prototypes": 16, "residual_scale": 0.30, "residual_weight": 0.010, "epochs": 10},
            {"teacher_lambda": 0.001, "num_prototypes": 16, "residual_scale": 0.30, "residual_weight": 0.001, "epochs": 10},
        ],
        "dynamic_fixed_three": [
            asdict(DynamicConfig(seed=7, epochs=2, learning_rate=1e-3, weight_decay=0.0, entropy_weight=0.0, hidden_dim=64, embedding_dim=64)),
            asdict(DynamicConfig(seed=7, epochs=5, learning_rate=1e-3, weight_decay=0.0, entropy_weight=0.0, hidden_dim=64, embedding_dim=64)),
            asdict(DynamicConfig(seed=7, epochs=5, learning_rate=5e-4, weight_decay=0.0, entropy_weight=0.0, hidden_dim=64, embedding_dim=64)),
            asdict(DynamicConfig(seed=7, epochs=5, learning_rate=1e-3, weight_decay=1e-4, entropy_weight=0.0, hidden_dim=64, embedding_dim=64)),
            asdict(DynamicConfig(seed=7, epochs=5, learning_rate=1e-3, weight_decay=0.0, entropy_weight=0.001, hidden_dim=64, embedding_dim=64)),
        ],
    }


def config_name(prefix: str, cfg: Mapping[str, Any]) -> str:
    parts = []
    for key in sorted(cfg):
        if key == "seed":
            continue
        val = cfg[key]
        if isinstance(val, float):
            parts.append(f"{key}{val:g}")
        else:
            parts.append(f"{key}{val}")
    raw = prefix + "_" + "_".join(parts)
    clean = raw.replace(".", "p").replace("None", "none")
    if prefix == "dynamic" and len(clean) > 96:
        digest = hashlib.sha256(json.dumps(dict(cfg), sort_keys=True).encode("utf-8")).hexdigest()[:12]
        return f"{prefix}_{digest}"
    return clean


def ridge_artifact_for_config(config: RidgeConfig, train_cache: Mapping[str, Any], train_baseline: torch.Tensor, std: torch.Tensor, series: torch.Tensor) -> dict[str, Any]:
    name = config_name("ridge", asdict(config))
    path = OUT_DIR / "artifacts" / "ridge" / name / "ridge_artifact.pt"
    if path.exists():
        art = load_artifact(path)
        if art.get("config") == asdict(config):
            if "early_stop_loss" not in art:
                art["early_stop_loss"] = float(art.get("validation_metrics", {}).get("mae", math.nan))
            return {"name": name, "path": path, "config": config, "artifact": art}
    starts = train_cache["absolute_window_starts"].to(torch.long)
    x_all, feature_names, stat_extra = build_feature_tensor(train_cache, starts, train_baseline, std, series, init_residuals_norm=None, allow_test_history=False)
    y_all, m_all = flattened_targets(train_cache, train_baseline, std)
    x_fit = x_all[m_all]
    y_fit = y_all[m_all]
    mean, scale = fit_scaler(x_fit)
    coef = fit_ridge(apply_scaler(x_fit, mean, scale), y_fit, config.ridge)
    residual_norm = (train_cache["targets"].to(torch.float32) - train_baseline) / std.view(1, 1, -1)
    art = {
        "config": asdict(config),
        "feature_names": feature_names,
        "feature_mean": mean,
        "feature_scale": scale,
        "coef": coef,
        "residual_train_std_norm": residual_norm.std(dim=0, unbiased=False).clamp_min(1e-6),
        "train_residual_norm": residual_norm,
        "stat_extra": stat_extra,
        "x_shape": list(x_all.shape),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(art, path)
    return {"name": name, "path": path, "config": config, "artifact": art}


def train_tuned_mlp_artifact(config: MlpConfig, train_cache: Mapping[str, Any], val_cache: Mapping[str, Any], train_baseline: torch.Tensor, val_baseline: torch.Tensor, std: torch.Tensor, series: torch.Tensor, device: torch.device) -> dict[str, Any]:
    name = config_name("mlp", asdict(config))
    path = OUT_DIR / "artifacts" / "mlp" / name / f"seed_{config.seed}" / "mlp_tuned_artifact.pt"
    if path.exists():
        art = load_artifact(path)
        if art.get("config") == asdict(config):
            return {"name": name, "path": path, "config": config, "artifact": art}
    set_seed(config.seed)
    train_starts = train_cache["absolute_window_starts"].to(torch.long)
    val_starts = val_cache["absolute_window_starts"].to(torch.long)
    train_resid = (train_cache["targets"].to(torch.float32) - train_baseline) / std.view(1, 1, -1)
    x_all, feature_names, _ = build_feature_tensor(train_cache, train_starts, train_baseline, std, series, init_residuals_norm=None, allow_test_history=False)
    y_all, m_all = flattened_targets(train_cache, train_baseline, std)
    x_all = x_all[m_all]
    y_all = y_all[m_all]
    mean, scale = fit_scaler(x_all)
    x_fit = apply_scaler(x_all, mean, scale)
    x_val, val_feature_names, _ = build_feature_tensor(val_cache, val_starts, val_baseline, std, series, init_residuals_norm=train_resid, allow_test_history=False)
    if val_feature_names != feature_names:
        raise RuntimeError("MLP val feature names changed")
    x_val = apply_scaler(x_val, mean, scale)
    model = TinyResidualMlp(x_fit.shape[1], config.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loader = DataLoader(TensorDataset(x_fit, y_all), batch_size=4096, shuffle=True)
    best_state = None
    best_epoch = -1
    best_mae = math.inf
    best_mse = math.inf
    best_extra: dict[str, Any] = {}
    curve = []
    for epoch in range(1, config.epochs + 1):
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
        model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, x_val.shape[0], 16384):
                outs.append(model(x_val[i : i + 16384].to(device)).cpu())
        delta = torch.cat(outs)
        val_pred, extra = apply_residual_delta(val_baseline, delta, std, config.alpha, config.clip_multiple, train_resid.std(dim=0, unbiased=False).clamp_min(1e-6))
        met = metrics(val_cache, val_pred, std)
        curve.append({"epoch": epoch, "train_loss": float(statistics.mean(losses)), "validation_mae": met["mae"], "validation_mse": met["mse"], **extra})
        if (met["mae"], met["mse"]) < (best_mae, best_mse):
            best_mae = met["mae"]
            best_mse = met["mse"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_extra = extra
    assert best_state is not None
    art = {
        "config": asdict(config),
        "feature_names": feature_names,
        "feature_mean": mean,
        "feature_scale": scale,
        "state_dict": best_state,
        "residual_train_std_norm": train_resid.std(dim=0, unbiased=False).clamp_min(1e-6),
        "train_residual_norm": train_resid,
        "best_epoch": best_epoch,
        "early_stop_loss": best_mae,
        "validation_metrics": {"mae": best_mae, "mse": best_mse, **best_extra},
        "x_shape": list(x_fit.shape),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(art, path)
    write_csv(path.parent / "training_curve.csv", curve)
    return {"name": name, "path": path, "config": config, "artifact": art}


def predict_tuned_mlp(cache: Mapping[str, Any], baseline: torch.Tensor, train_artifact: Mapping[str, Any], std: torch.Tensor, series: torch.Tensor, device: torch.device, allow_test_history: bool) -> tuple[torch.Tensor, dict[str, Any]]:
    art = train_artifact["artifact"]
    x, names, stat_extra = build_feature_tensor(cache, cache["absolute_window_starts"].to(torch.long), baseline, std, series, init_residuals_norm=art["train_residual_norm"], allow_test_history=allow_test_history)
    if names != art["feature_names"]:
        raise RuntimeError("MLP feature names changed")
    model = TinyResidualMlp(len(names), int(art["config"]["hidden"])).to(device)
    model.load_state_dict(art["state_dict"])
    model.eval()
    x = apply_scaler(x, art["feature_mean"], art["feature_scale"])
    outs = []
    with torch.no_grad():
        for i in range(0, x.shape[0], 16384):
            outs.append(model(x[i : i + 16384].to(device)).cpu())
    delta = torch.cat(outs)
    pred, extra = apply_residual_delta(baseline, delta, std, float(art["config"]["alpha"]), art["config"]["clip_multiple"], art["residual_train_std_norm"])
    return pred, {**extra, **stat_extra, "best_epoch": art.get("best_epoch", -1), "early_stop_loss": art.get("early_stop_loss", art.get("validation_metrics", {}).get("mae", math.nan))}


def oracle_config_from_dict(cfg: Mapping[str, Any], seed: int) -> OracleTrialConfig:
    name = config_name("oracle", cfg) + f"_seed{seed}"
    return OracleTrialConfig(
        family="prototype_residual",
        name=name,
        seed=seed,
        num_prototypes=int(cfg["num_prototypes"]),
        teacher_lambda=float(cfg["teacher_lambda"]),
        residual_scale=float(cfg["residual_scale"]),
        residual_weight=float(cfg["residual_weight"]),
        epochs=int(cfg["epochs"]),
    )


def train_tuned_oracle_artifact(config: OracleTrialConfig, train_cache: Mapping[str, Any], val_cache: Mapping[str, Any], std: torch.Tensor, device: torch.device) -> dict[str, Any]:
    name = config.name.rsplit("_seed", 1)[0]
    path = OUT_DIR / "artifacts" / "oracle" / name / f"seed_{config.seed}" / "oracle_tuned_artifact.pt"
    if path.exists():
        art = load_artifact(path)
        if art.get("config") == asdict(config):
            return {"name": name, "path": path, "config": config, "artifact": art}
    set_seed(config.seed)
    core_idx = expert_indices(train_cache, CORE)
    forecasts = forecasts_for(train_cache, core_idx)
    targets = train_cache["targets"].to(torch.float32)
    masks = train_cache["target_masks"].to(torch.bool)
    global_weights = torch.tensor(args_global_weights(), dtype=torch.float32)
    teacher, teacher_mae = oracle_weights_grid(forecasts, targets, masks, std, global_weights, config.teacher_lambda, step=0.02)
    prototypes, proto_labels = kmeans(teacher, config.num_prototypes, config.seed)
    train_ds = OracleWindowDataset(train_cache, core_idx)
    model = WeightStudent(
        global_weights,
        int(train_cache["input_len"]),
        int(train_cache["forecast_horizon"]),
        int(train_cache["num_features"]),
        mode="prototype_residual",
        num_prototypes=config.num_prototypes,
        rank=config.rank,
        residual_scale=config.residual_scale,
        feature_mix=config.feature_mix,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr)
    loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    best_state = None
    best_epoch = -1
    best_mae = math.inf
    best_mse = math.inf
    curve = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        for batch in loader:
            hist = batch["history"].to(device)
            fcast = batch["forecasts"].to(device)
            target = batch["target"].to(device)
            mask = batch["mask"].to(device)
            idx = batch["index"]
            out = model(hist, fcast, prototypes=prototypes)
            weights = out["weights"]
            teacher_loss = F.smooth_l1_loss(weights, teacher[idx].to(device)) + F.cross_entropy(out["logits"], proto_labels[idx].to(device))
            pred = weighted_forecast(fcast, weights)
            forecast_loss = sample_mae(pred, target, mask, std.to(device)).mean()
            residual_loss = (weights - global_weights.to(device).view(1, 3)).square().mean()
            loss = config.forecast_weight * forecast_loss + config.teacher_weight * teacher_loss + config.residual_weight * residual_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val_eval = eval_oracle_model(model, val_cache, core_idx, prototypes, std, device)
        curve.append({"epoch": epoch, "train_loss": float(statistics.mean(losses)), "validation_mae": val_eval["mae"], "validation_mse": val_eval["mse"]})
        if (val_eval["mae"], val_eval["mse"]) < (best_mae, best_mse):
            best_mae = val_eval["mae"]
            best_mse = val_eval["mse"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    assert best_state is not None
    art = {
        "config": asdict(config),
        "state_dict": best_state,
        "prototypes": prototypes,
        "teacher_sha256": sha256_tensor(teacher),
        "teacher_train_mae_mean": float(teacher_mae.mean()),
        "best_epoch": best_epoch,
        "validation_metrics": {"mae": best_mae, "mse": best_mse},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(art, path)
    write_csv(path.parent / "training_curve.csv", curve)
    return {"name": name, "path": path, "config": config, "artifact": art}


def train_tuned_dynamic_artifact(config: DynamicConfig, train_cache: Mapping[str, Any], val_cache: Mapping[str, Any], std: torch.Tensor, device: torch.device) -> dict[str, Any]:
    name = config_name("dynamic", asdict(config))
    path = OUT_DIR / "artifacts" / "dynamic" / name / "dynamic_tuned_artifact.pt"
    if path.exists():
        art = load_artifact(path)
        if art.get("config") == asdict(config):
            return {"name": name, "path": path, "config": config, "artifact": art}
    set_seed(config.seed)
    model = Fixed3DynamicWeightRouter(
        input_len=int(train_cache["input_len"]),
        horizon=int(train_cache["forecast_horizon"]),
        num_features=int(train_cache["num_features"]),
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loader = DataLoader(DynamicDataset(train_cache), batch_size=config.batch_size, shuffle=True)
    best_state = None
    best_epoch = -1
    best_mae = math.inf
    best_mse = math.inf
    best_extra: dict[str, Any] = {}
    curves = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        for batch in loader:
            hist = batch["history"].to(device)
            forecasts = batch["forecasts"].to(device)
            target = batch["targets"].to(device)
            mask = batch["target_masks"].to(device)
            out = model(hist, forecasts, config.ablation)
            pred = weighted_forecast(forecasts, out["weights"])
            mae = sample_mae(pred, target, mask, std.to(device)).mean()
            entropy = -(out["weights"] * out["weights"].clamp_min(1e-8).log()).sum(dim=1).mean()
            loss = mae - float(config.entropy_weight) * entropy
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_eval = eval_dynamic(model, val_cache, std, device, config)
        curves.append({"epoch": epoch, "train_loss": float(statistics.mean(losses)), "validation_mae": val_eval["mae"], "validation_mse": val_eval["mse"], **{f"mean_weight_{k}": v for k, v in val_eval["mean_weights"].items()}})
        if (val_eval["mae"], val_eval["mse"]) < (best_mae, best_mse):
            best_mae = val_eval["mae"]
            best_mse = val_eval["mse"]
            best_epoch = epoch
            best_extra = {"mean_weights": val_eval["mean_weights"], "weight_std": val_eval["weight_std"]}
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    assert best_state is not None
    art = {"config": asdict(config), "state_dict": best_state, "best_epoch": best_epoch, "validation_metrics": {"mae": best_mae, "mse": best_mse, **best_extra}}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(art, path)
    write_csv(path.parent / "training_curve.csv", curves)
    return {"name": name, "path": path, "config": config, "artifact": art}


def evaluate_locked_side_by_side() -> list[dict[str, Any]]:
    locked_path = LOCKED_OUT_DIR / "final_report.json"
    if not locked_path.exists():
        raise FileNotFoundError("Locked ETTh1-config ETTh2 replication final_report.json is required")
    locked = json.loads(locked_path.read_text(encoding="utf-8"))
    rows = []
    for row in locked["test_results"]:
        if row["method"] in {"MLP residual corrector", "Ridge residual corrector", "Oracle prototype residual", "Dynamic fixed-three, seed 7"}:
            rows.append({
                "method": row["method"],
                "version": LOCKED_LABEL,
                "validation_mae": row["validation_mae"],
                "validation_mse": row["validation_mse"],
                "test_mae": row["mae"],
                "test_mse": row["mse"],
                "diff_vs_single_DLinear_test": float(row["mae"]) - SINGLE_DLINEAR_TEST_MAE,
                "diff_vs_full_adaptive_test": float(row["mae"]) - FULL_ADAPTIVE_TEST_MAE,
                "diff_vs_locked_test": 0.0,
            })
    return rows


def train_validate(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    spaces = declared_search_spaces()
    write_json(OUT_DIR / "declared_search_space.json", spaces)
    train_cache = load_cache(TRAIN_CACHE, "router_train", allow_test=False)
    val_cache = load_cache(VAL_CACHE, "router_val", allow_test=False)
    validate_cache_shape(train_cache, "router_train")
    validate_cache_shape(val_cache, "router_val")
    std = torch.ones(int(train_cache["num_features"]), dtype=torch.float32)
    device = torch.device(args.device)
    series = load_series_prefix()
    train_baseline, _ = build_chrono_oof_baseline(train_cache, std)
    val_baseline, _ = build_eval_baseline(val_cache, train_cache, std)

    sweep_rows: list[dict[str, Any]] = []
    winner_artifacts: dict[str, dict[str, Any]] = {}

    for cfg_dict in spaces["ridge_residual_corrector"]:
        cfg = RidgeConfig(**cfg_dict)
        art = ridge_artifact_for_config(cfg, train_cache, train_baseline, std, series)
        pred, extra = predict_ridge(val_cache, val_baseline, art, std, series, allow_test_history=False)
        row = metric_row("Ridge residual corrector", "router_val", val_cache, pred, std, CORE)
        row.update({"version": LABEL, "config_name": art["name"], "config": cfg_dict, **extra, "artifact_path": str(art["path"])})
        sweep_rows.append(row)
    ridge_rows = [r for r in sweep_rows if r["method"] == "Ridge residual corrector"]
    ridge_winner = min(ridge_rows, key=lambda r: (float(r["mae"]), float(r["mse"])))
    winner_artifacts["Ridge residual corrector"] = {"row": ridge_winner, "path": Path(ridge_winner["artifact_path"])}

    for base_cfg in spaces["mlp_residual_corrector"]:
        cfg_rows = []
        preds = []
        artifacts = []
        for seed in SEEDS:
            cfg = MlpConfig(**{**base_cfg, "seed": seed})
            art = train_tuned_mlp_artifact(cfg, train_cache, val_cache, train_baseline, val_baseline, std, series, device)
            pred, extra = predict_tuned_mlp(val_cache, val_baseline, art, std, series, device, allow_test_history=False)
            preds.append(pred)
            artifacts.append(art)
            row = metric_row("MLP residual corrector", "router_val", val_cache, pred, std, CORE, seed=seed)
            row.update({"version": LABEL, "config_name": art["name"], "config": {**base_cfg, "seed": seed}, **extra, "artifact_path": str(art["path"])})
            cfg_rows.append(row)
            sweep_rows.append(row)
        mean_pred = aggregate_seed_predictions(preds)
        mean_row = metric_row("MLP residual corrector", "router_val", val_cache, mean_pred, std, CORE)
        mean_row.update({"version": LABEL, "config_name": artifacts[0]["name"], "config": base_cfg, "seed": "mean", "artifact_paths": [str(a["path"]) for a in artifacts]})
        sweep_rows.append(mean_row)
    mlp_mean_rows = [r for r in sweep_rows if r["method"] == "MLP residual corrector" and r.get("seed") == "mean"]
    mlp_winner = min(mlp_mean_rows, key=lambda r: (float(r["mae"]), float(r["mse"])))
    winner_artifacts["MLP residual corrector"] = {"row": mlp_winner, "paths": [Path(p) for p in mlp_winner["artifact_paths"]]}

    for base_cfg in spaces["oracle_prototype_residual"]:
        preds = []
        artifacts = []
        for seed in SEEDS:
            cfg = oracle_config_from_dict(base_cfg, seed)
            art = train_tuned_oracle_artifact(cfg, train_cache, val_cache, std, device)
            pred, extra = predict_oracle(val_cache, train_cache, art, std, device)
            preds.append(pred)
            artifacts.append(art)
            row = metric_row("Oracle prototype residual", "router_val", val_cache, pred, std, CORE, seed=seed)
            row.update({"version": LABEL, "config_name": art["name"], "config": asdict(cfg), **extra, "artifact_path": str(art["path"])})
            sweep_rows.append(row)
        mean_pred = aggregate_seed_predictions(preds)
        mean_row = metric_row("Oracle prototype residual", "router_val", val_cache, mean_pred, std, CORE)
        mean_row.update({"version": LABEL, "config_name": artifacts[0]["name"], "config": base_cfg, "seed": "mean", "artifact_paths": [str(a["path"]) for a in artifacts]})
        sweep_rows.append(mean_row)
    oracle_mean_rows = [r for r in sweep_rows if r["method"] == "Oracle prototype residual" and r.get("seed") == "mean"]
    oracle_winner = min(oracle_mean_rows, key=lambda r: (float(r["mae"]), float(r["mse"])))
    winner_artifacts["Oracle prototype residual"] = {"row": oracle_winner, "paths": [Path(p) for p in oracle_winner["artifact_paths"]]}

    for cfg_dict in spaces["dynamic_fixed_three"]:
        cfg = DynamicConfig(**cfg_dict)
        art = train_tuned_dynamic_artifact(cfg, train_cache, val_cache, std, device)
        pred, extra = predict_dynamic(val_cache, art, std, device)
        row = metric_row("Dynamic fixed-three", "router_val", val_cache, pred, std, CORE, seed=7)
        row.update({"version": LABEL, "config_name": art["name"], "config": cfg_dict, **extra, "artifact_path": str(art["path"]), "best_epoch": art["artifact"]["best_epoch"]})
        sweep_rows.append(row)
    dyn_rows = [r for r in sweep_rows if r["method"] == "Dynamic fixed-three"]
    dyn_winner = min(dyn_rows, key=lambda r: (float(r["mae"]), float(r["mse"])))
    winner_artifacts["Dynamic fixed-three"] = {"row": dyn_winner, "path": Path(dyn_winner["artifact_path"])}

    write_csv(OUT_DIR / "validation_sweep_results.csv", sweep_rows)
    winners = [winner_artifacts[k]["row"] for k in ("Ridge residual corrector", "MLP residual corrector", "Oracle prototype residual", "Dynamic fixed-three")]
    write_csv(OUT_DIR / "validation_tuned_winners.csv", winners)
    frozen_dir = OUT_DIR / "frozen_winners"
    frozen_dir.mkdir(parents=True, exist_ok=True)
    frozen: dict[str, Any] = {}
    for method, payload in winner_artifacts.items():
        method_dir = frozen_dir / method.lower().replace(" ", "_").replace(",", "")
        method_dir.mkdir(parents=True, exist_ok=True)
        if "path" in payload:
            dst = method_dir / Path(payload["path"]).name
            shutil.copy2(payload["path"], dst)
            paths = [dst]
        else:
            paths = []
            for src in payload["paths"]:
                dst = method_dir / f"{src.parent.name}_{src.name}"
                shutil.copy2(src, dst)
                paths.append(dst)
        frozen[method] = {
            "selected_by": "router_val_mae_then_mse",
            "row": payload["row"],
            "frozen_artifact_paths": [str(p) for p in paths],
            "frozen_artifact_sha256": {p.name: sha256_file(p) for p in paths},
        }
    manifest = {
        "label": LABEL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_loaded_during_sweep": False,
        "test_loaded_before_tuned_freeze": False,
        "test_metrics_used_for_decision": False,
        "dataset": "ETTh2",
        "core": list(CORE),
        "declared_search_space": spaces,
        "selection_metric": "validation MAE primary, validation MSE secondary",
        "splits": {"router_train": [8640, 10800], "router_val": [10800, 11520], "test": [11520, 14400]},
        "cache_paths": {"router_train": str(TRAIN_CACHE), "router_val": str(VAL_CACHE), "test_planned_after_freeze": str(TEST_CACHE)},
        "cache_hashes": {"router_train": sha256_file(TRAIN_CACHE), "router_val": sha256_file(VAL_CACHE)},
        "validation_sweep_results_file": str(OUT_DIR / "validation_sweep_results.csv"),
        "selected_winners": frozen,
        "leakage_checks": {
            "router_train_used_for_training": True,
            "router_val_used_for_hyperparameter_and_checkpoint_selection": True,
            "test_not_loaded_before_winner_freeze": True,
            "test_not_used_for_any_decision": True,
            "causal_feature_rule": "old_start + horizon <= current_start",
            "raw_scale_std_ones_no_inverse_transform": True,
        },
        "runtime_sec_train_validate": time.perf_counter() - started,
    }
    write_json(OUT_DIR / "tuned_manifest_before_test.json", manifest)
    print(json.dumps({"phase": "train_validate", "winners": winners, "manifest": str(OUT_DIR / "tuned_manifest_before_test.json")}, indent=2))


def artifact_payload(path: str | Path) -> dict[str, Any]:
    return {"artifact": load_artifact(Path(path))}


def evaluate_test(args: argparse.Namespace) -> None:
    manifest_path = OUT_DIR / "tuned_manifest_before_test.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Run train_validate before test")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("test_loaded_before_tuned_freeze") is not False:
        raise RuntimeError("Tuned manifest does not certify no test load before freeze")
    started = time.perf_counter()
    train_cache = load_cache(TRAIN_CACHE, "router_train", allow_test=False)
    test_cache = load_cache(TEST_CACHE, "locked_test", allow_test=True)
    validate_cache_shape(train_cache, "router_train")
    validate_cache_shape(test_cache, "locked_test")
    std = torch.ones(int(test_cache["num_features"]), dtype=torch.float32)
    device = torch.device(args.device)
    train_val_series = load_series_prefix()
    test_series = torch.cat((train_val_series, torch.from_numpy(np.load(DATASET_DIR / "test_data.npy")).to(torch.float32)), dim=0)
    test_baseline, _ = build_eval_baseline(test_cache, train_cache, std)
    rows: list[dict[str, Any]] = []
    per_seed_rows: list[dict[str, Any]] = []

    ridge = manifest["selected_winners"]["Ridge residual corrector"]
    ridge_art = artifact_payload(ridge["frozen_artifact_paths"][0])
    ridge_pred, ridge_extra = predict_ridge(test_cache, test_baseline, ridge_art, std, test_series, allow_test_history=True)
    rows.append(metric_row("Ridge residual corrector", "locked_test", test_cache, ridge_pred, std, CORE))
    rows[-1].update({"version": LABEL, **ridge_extra})

    mlp = manifest["selected_winners"]["MLP residual corrector"]
    mlp_preds = []
    for path in mlp["frozen_artifact_paths"]:
        art = artifact_payload(path)
        pred, extra = predict_tuned_mlp(test_cache, test_baseline, art, std, test_series, device, allow_test_history=True)
        mlp_preds.append(pred)
        row = metric_row("MLP residual corrector", "locked_test", test_cache, pred, std, CORE, seed=art["artifact"]["config"]["seed"])
        row.update({"version": LABEL, **extra})
        per_seed_rows.append(row)
    rows.append(metric_row("MLP residual corrector", "locked_test", test_cache, aggregate_seed_predictions(mlp_preds), std, CORE))
    rows[-1].update({"version": LABEL})

    oracle = manifest["selected_winners"]["Oracle prototype residual"]
    oracle_preds = []
    for path in oracle["frozen_artifact_paths"]:
        art = artifact_payload(path)
        pred, extra = predict_oracle(test_cache, train_cache, art, std, device)
        oracle_preds.append(pred)
        row = metric_row("Oracle prototype residual", "locked_test", test_cache, pred, std, CORE, seed=art["artifact"]["config"]["seed"])
        row.update({"version": LABEL, **extra})
        per_seed_rows.append(row)
    rows.append(metric_row("Oracle prototype residual", "locked_test", test_cache, aggregate_seed_predictions(oracle_preds), std, CORE))
    rows[-1].update({"version": LABEL})

    dyn = manifest["selected_winners"]["Dynamic fixed-three"]
    dyn_art = artifact_payload(dyn["frozen_artifact_paths"][0])
    dyn_pred, dyn_extra = predict_dynamic(test_cache, dyn_art, std, device)
    rows.append(metric_row("Dynamic fixed-three", "locked_test", test_cache, dyn_pred, std, CORE, seed=7))
    rows[-1].update({"version": LABEL, **dyn_extra})

    locked_lookup = {r["method"]: r for r in evaluate_locked_side_by_side()}
    winner_lookup = {
        "Ridge residual corrector": manifest["selected_winners"]["Ridge residual corrector"]["row"],
        "MLP residual corrector": manifest["selected_winners"]["MLP residual corrector"]["row"],
        "Oracle prototype residual": manifest["selected_winners"]["Oracle prototype residual"]["row"],
        "Dynamic fixed-three": manifest["selected_winners"]["Dynamic fixed-three"]["row"],
    }
    for row in rows:
        method = row["method"]
        locked_key = method if method != "Dynamic fixed-three" else "Dynamic fixed-three, seed 7"
        locked = locked_lookup[locked_key]
        winner = winner_lookup[method]
        row["validation_mae"] = winner["mae"]
        row["validation_mse"] = winner["mse"]
        row["test_minus_validation_mae"] = float(row["mae"]) - float(row["validation_mae"])
        row["diff_vs_single_DLinear_test"] = float(row["mae"]) - SINGLE_DLINEAR_TEST_MAE
        row["diff_vs_full_adaptive_test"] = float(row["mae"]) - FULL_ADAPTIVE_TEST_MAE
        row["diff_vs_corresponding_locked_test"] = float(row["mae"]) - float(locked["test_mae"])
        row["selection_protocol"] = "ETTh2 validation-tuned; router_train training, router_val config/checkpoint selection, test once after tuned freeze"
        row["status"] = LABEL
    write_csv(OUT_DIR / "test_results.csv", rows)
    write_csv(OUT_DIR / "test_per_seed_results.csv", per_seed_rows)
    write_csv(OUT_DIR / "test_seed_summary.csv", aggregate_seed_rows(per_seed_rows))

    locked_rows = evaluate_locked_side_by_side()
    tuned_rows = [
        {
            "method": r["method"],
            "version": LABEL,
            "validation_mae": r["validation_mae"],
            "validation_mse": r["validation_mse"],
            "test_mae": r["mae"],
            "test_mse": r["mse"],
            "diff_vs_single_DLinear_test": r["diff_vs_single_DLinear_test"],
            "diff_vs_full_adaptive_test": r["diff_vs_full_adaptive_test"],
            "diff_vs_locked_test": r["diff_vs_corresponding_locked_test"],
        }
        for r in rows
    ]
    side_by_side = locked_rows + tuned_rows
    write_csv(OUT_DIR / "locked_vs_tuned_results.csv", side_by_side)
    final = {**manifest, "test_evaluation_complete": True, "test_cache_loaded_after_tuned_freeze": True, "test_cache_hash": sha256_file(TEST_CACHE), "test_results": rows, "locked_vs_tuned": side_by_side, "runtime_sec_test": time.perf_counter() - started}
    write_json(OUT_DIR / "final_report.json", final)
    write_report(final)
    update_matched_results(final)
    update_all_results_summary(final)
    update_project_memory(final)
    print(json.dumps({"phase": "test", "test_results": rows, "report": str(OUT_DIR / "ETTH2_VALIDATION_TUNED_MISSING_METHODS_REPORT.md")}, indent=2))


def fmt(x: Any) -> str:
    return f"{float(x):.6f}"


def write_report(payload: Mapping[str, Any]) -> None:
    lines = [
        "# ETTh2 Validation-Tuned Missing Methods",
        "",
        "Label: `etth2_validation_tuned`.",
        "",
        "These runs use ETTh2 router-validation for hyperparameter and checkpoint selection. They are not preregistered and are not untouched-test confirmation.",
        "",
        "## Declared Search Spaces",
        "",
    ]
    for method, configs in payload["declared_search_space"].items():
        lines.append(f"### {method}")
        for i, cfg in enumerate(configs, 1):
            lines.append(f"{i}. `{cfg}`")
        lines.append("")
    lines.extend(["## Complete Validation Sweep", "", "| Method | Config | Seed | Val MAE | Val MSE |", "|---|---|---:|---:|---:|"])
    with (OUT_DIR / "validation_sweep_results.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            lines.append(f"| {row['method']} | `{row['config_name']}` | {row.get('seed', '')} | {fmt(row['mae'])} | {fmt(row['mse'])} |")
    lines.extend(["", "## Selected Tuned Configurations", "", "| Method | Config | Val MAE | Val MSE | Frozen artifacts |", "|---|---|---:|---:|---|"])
    for method, frozen in payload["selected_winners"].items():
        row = frozen["row"]
        lines.append(f"| {method} | `{row['config_name']}` | {fmt(row['mae'])} | {fmt(row['mse'])} | `{len(frozen['frozen_artifact_paths'])}` |")
    lines.extend(["", "## Locked vs Tuned Test Results", "", "| Method | Version | Val MAE | Test MAE | Test MSE | Diff vs DLinear | Diff vs full adaptive | Diff vs locked |", "|---|---|---:|---:|---:|---:|---:|---:|"])
    for row in payload["locked_vs_tuned"]:
        lines.append(f"| {row['method']} | `{row['version']}` | {fmt(row['validation_mae'])} | {fmt(row['test_mae'])} | {fmt(row['test_mse'])} | {float(row['diff_vs_single_DLinear_test']):+.6f} | {float(row['diff_vs_full_adaptive_test']):+.6f} | {float(row['diff_vs_locked_test']):+.6f} |")
    lines.extend(
        [
            "",
            "## Leakage Checks",
            "",
            "- Search space declared in `declared_search_space.json` before sweep results were written.",
            "- Router-train was used for fitting.",
            "- Router-validation was used for hyperparameter and checkpoint selection.",
            "- Test cache was not loaded before `tuned_manifest_before_test.json` was written.",
            "- Test was evaluated once for each tuned winner after freeze.",
            "- Causal residual features enforce `old_start + horizon <= current_start`.",
            "",
            "## Reproduce",
            "",
            "```powershell",
            "python experiments\\etth2_validation_tuned_missing_methods\\run_etth2_validation_tuned_missing_methods.py --phase all --device cuda",
            "```",
        ]
    )
    (OUT_DIR / "ETTH2_VALIDATION_TUNED_MISSING_METHODS_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_matched_results(payload: Mapping[str, Any]) -> None:
    path = FROZEN_RESULTS_DIR / "matched_etth1_etth2_results.csv"
    rows: list[dict[str, Any]] = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    etth1_lookup = {r["method"]: r for r in rows}
    tuned_existing = {r["method"] for r in rows if r.get("etth2_status") == LABEL}
    for row in payload["test_results"]:
        base_method = row["method"] if row["method"] != "Dynamic fixed-three" else "Dynamic fixed-three, seed 7"
        if f"{base_method} (ETTh2 validation tuned)" in tuned_existing:
            continue
        base = etth1_lookup.get(base_method, {})
        rows.append({
            "method": f"{base_method} (ETTh2 validation tuned)",
            "etth1_test_mae": base.get("etth1_test_mae", ""),
            "etth1_test_mse": base.get("etth1_test_mse", ""),
            "etth1_validation_mae": base.get("etth1_validation_mae", ""),
            "etth2_test_mae": row["mae"],
            "etth2_test_mse": row["mse"],
            "etth2_validation_mae": row["validation_mae"],
            "etth2_expert_set": "+".join(CORE),
            "etth2_status": LABEL,
            "etth2_note": "ETTh2 validation-tuned version; router_val selected config/checkpoint; test evaluated once after tuned freeze.",
        })
    write_csv(path, rows)


def update_all_results_summary(payload: Mapping[str, Any]) -> None:
    path = ALL_RESULTS_DIR / "all_costar_results.csv"
    if not path.exists():
        return
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    rows = [r for r in rows if not (r.get("dataset") == "ETTh2" and r.get("status") == LABEL)]
    for row in payload["test_results"]:
        out = {
            "dataset": "ETTh2",
            "split": "test",
            "method": row["method"],
            "expert_set": row["expert_set"],
            "result_group": LABEL,
            "status": LABEL,
            "test_mae": row["mae"],
            "test_mse": row["mse"],
            "validation_mae": row["validation_mae"],
            "validation_mse": row["validation_mse"],
            "diff_vs_validation": row["test_minus_validation_mae"],
            "comparison_anchor": "ETTh2 full adaptive test",
            "diff_vs_anchor": row["diff_vs_full_adaptive_test"],
            "seeds": "5" if row["method"] in {"MLP residual corrector", "Oracle prototype residual"} else "1",
            "selection_protocol": row["selection_protocol"],
            "source_file": "experiments/etth2_validation_tuned_missing_methods/test_results.csv",
        }
        for field in fields:
            out.setdefault(field, "")
        rows.append(out)
    write_csv(path, rows)


def update_project_memory(payload: Mapping[str, Any]) -> None:
    log = ROOT / "project_memory" / "experiments" / "2026-08-13_etth2_validation_tuned_missing_methods.md"
    lines = [
        "# ETTh2 Validation-Tuned Missing Methods",
        "",
        "Ran small ETTh2 router-validation-tuned sweeps for MLP residual, ridge residual, oracle prototype residual, and dynamic fixed-three. These are labeled `etth2_validation_tuned` and are not pre-test preregistered results.",
        "",
        "| Method | Val MAE | Test MAE | Test MSE | Diff vs locked |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in payload["test_results"]:
        lines.append(f"| {row['method']} | `{row['validation_mae']:.6f}` | `{row['mae']:.6f}` | `{row['mse']:.6f}` | `{row['diff_vs_corresponding_locked_test']:+.6f}` |")
    lines.extend(["", "Artifacts:", "", "- `experiments/etth2_validation_tuned_missing_methods/final_report.json`", "- `experiments/etth2_validation_tuned_missing_methods/ETTH2_VALIDATION_TUNED_MISSING_METHODS_REPORT.md`"])
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    exp = ROOT / "project_memory" / "EXPERIMENTS.md"
    text = exp.read_text(encoding="utf-8")
    marker = "| 2026-08-13 | ETTh2 | Validation-tuned missing methods |"
    if marker not in text:
        best = min(payload["test_results"], key=lambda r: float(r["mae"]))
        row = f"| 2026-08-13 | ETTh2 | Validation-tuned missing methods | Small router-val-tuned sweeps for MLP/ridge/oracle/dynamic missing-method counterparts | MLP/prototype 5; ridge deterministic; dynamic 1 | best `{best['mae']:.6f}` ({best['method']}) | best `{best['mse']:.6f}` | n/a | corresponding locked ETTh1-config rows | see report | Validation-tuned after final test; not preregistered | `experiments/etth2_validation_tuned_missing_methods/final_report.json`; `project_memory/experiments/2026-08-13_etth2_validation_tuned_missing_methods.md` |"
        text = text.replace("\nUnverified or partial:", "\n" + row + "\n\nUnverified or partial:")
        exp.write_text(text, encoding="utf-8")

    current = ROOT / "project_memory" / "CURRENT_STATE.md"
    cur = current.read_text(encoding="utf-8")
    if "## ETTh2 Validation-Tuned Missing Methods" not in cur:
        block = """
## ETTh2 Validation-Tuned Missing Methods

ADDITIONAL VALIDATION-TUNED RESULT:

Small ETTh2 validation-tuned sweeps now exist for MLP residual, ridge residual, oracle prototype residual, and dynamic fixed-three. These use ETTh2 router-validation for selection and are labeled `etth2_validation_tuned`; they are not pre-test preregistered or untouched-test confirmation.

Artifacts:

- `experiments/etth2_validation_tuned_missing_methods/final_report.json`
- `experiments/etth2_validation_tuned_missing_methods/ETTH2_VALIDATION_TUNED_MISSING_METHODS_REPORT.md`

"""
        cur = cur.replace("## Final Pre-Test Freeze", block + "## Final Pre-Test Freeze")
        current.write_text(cur, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("train_validate", "test", "all"), default="all")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.phase in {"train_validate", "all"}:
        train_validate(args)
    if args.phase in {"test", "all"}:
        evaluate_test(args)


if __name__ == "__main__":
    main()
