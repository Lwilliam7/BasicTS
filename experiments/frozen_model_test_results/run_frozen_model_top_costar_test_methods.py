"""Frozen-model test results for top ETTh1 COSTAR-TS methods.

Every model evaluated here was trained, selected, configured, and frozen without
test-data feedback. These are additional frozen-model test evaluations, distinct
from the original confirmatory freeze in experiments/final_test_evaluation/.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.chronological_adaptive_costar.run_chronological_adaptive_costar import (  # noqa: E402
    SEEDS,
    Trial as ChronoTrial,
    chronological_online_weights,
    load_static_winner_per_window,
    paired_bootstrap,
)
from experiments.expanded_expert_pool_costar.run_expanded_expert_pool import (  # noqa: E402
    Config as SpecialistConfig,
    normalized_abs_error,
    optional_predictions,
    run_causal_specialists,
)
from experiments.horizon_variable_adaptive_costar.run_hv_adaptive_costar import (  # noqa: E402
    fixed3_forecasts,
    per_location_abs_error,
)
from experiments.oracle_weight_tournament.run_tournament import (  # noqa: E402
    load_std,
    sample_mae,
    sample_mse,
    weighted_forecast,
)
from experiments.residual_correction_costar.run_residual_correction_experiments import (  # noqa: E402
    MlpConfig,
    RidgeConfig,
    apply_residual_delta,
    apply_scaler,
    build_feature_tensor,
    fit_ridge,
    fit_scaler,
    fixed_current_best_prediction,
    flattened_targets,
    predict_linear,
    train_mlp_final,
)
from scripts.train_costarts_fixed3_dynamic_weighting import (  # noqa: E402
    Fixed3DynamicWeightRouter,
    evaluate as evaluate_dynamic_fixed3,
)


OUT_DIR = ROOT / "experiments" / "frozen_model_test_results"
FINAL_TEST_DIR = ROOT / "experiments" / "final_test_evaluation"
TRAIN_CACHE = ROOT / "cache" / "costarts_walkforward" / "router_train_20_60_cache.pt"
TEST_CACHE = FINAL_TEST_DIR / "generated" / "caches" / "ETTh1" / "test_80_100_cache.pt"
NORMALIZER = ROOT / "checkpoints" / "costarts_walkforward" / "final_60" / "DLinear" / "best_expert.pt"
DATASET_DIR = ROOT / "datasets" / "ETTh1"
FIXED_CORE_VAL = {"mae": 0.36726489663124084, "mse": 0.3105303645133972}
VALIDATION_REFS = {
    "fixed_core_equal": {"mae": 0.36726489663124084, "mse": 0.3105303645133972},
    "dynamic_fixed3_seed7": {"mae": 0.36598527431488037, "mse": 0.3088900148868561},
    "oracle_prototype_residual": {"mae": 0.3660282492637634, "mse": 0.308755099773407},
    "chronological_ema_hybrid": {"mae": 0.36553388833999634, "mse": 0.3083403706550598},
    "horizon_variable_hybrid": {"mae": 0.36364156007766724, "mse": 0.3067120909690857},
    "ridge_residual_corrector": {"mae": 0.36330097913742065, "mse": 0.3062863349914551},
    "mlp_residual_corrector": {"mae": 0.3633176386356354, "mse": 0.306606650352478},
    "expanded_dlinear_only": {"mae": 0.3635100722312927, "mse": 0.30655738711357117},
    "expanded_moderntcn_only": {"mae": 0.36343517899513245, "mse": 0.3064517080783844},
    "expanded_both_final_frozen": {"mae": 0.3631121516227722, "mse": 0.30605703592300415},
}


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_test_cache(path: Path) -> dict[str, Any]:
    cache = torch.load(path, map_location="cpu", weights_only=False)
    if cache.get("cache_role") != "test_80_100":
        raise ValueError(f"Expected ETTh1 test_80_100 cache, got {cache.get('cache_role')}")
    if int(cache["num_windows"]) != 2773:
        raise ValueError("Unexpected ETTh1 test window count")
    return cache


def metrics(cache: Mapping[str, Any], std: torch.Tensor, pred: torch.Tensor) -> dict[str, Any]:
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.bool)
    mae = sample_mae(pred, target, mask, std)
    mse = sample_mse(pred, target, mask, std)
    return {"mae": float(mae.mean()), "mse": float(mse.mean()), "per_window_mae": mae, "per_window_mse": mse}


def fixed_core_prediction(cache: Mapping[str, Any]) -> torch.Tensor:
    names = list(cache["expert_names"])
    idx = [names.index(name) for name in ("PatchTST", "iTransformer", "TimesNet")]
    return cache["prediction_stack"][..., idx].to(torch.float32).mean(dim=-1)


def load_series_with_test() -> torch.Tensor:
    parts = [
        torch.from_numpy(np.load(DATASET_DIR / "train_data.npy")).to(torch.float32),
        torch.from_numpy(np.load(DATASET_DIR / "val_data.npy")).to(torch.float32),
        torch.from_numpy(np.load(DATASET_DIR / "test_data.npy")).to(torch.float32),
    ]
    return torch.cat(parts, dim=0)


def chronological_prediction(
    train_cache: Mapping[str, Any],
    test_cache: Mapping[str, Any],
    std: torch.Tensor,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    starts = test_cache["absolute_window_starts"].to(torch.long)
    horizon = int(test_cache["forecast_horizon"])
    train_err = per_location_abs_error(train_cache, std).mean(dim=(1, 2))
    test_err = per_location_abs_error(test_cache, std).mean(dim=(1, 2))
    online, extra = chronological_online_weights(
        starts=starts,
        expert_mae=test_err,
        horizon=horizon,
        trial=ChronoTrial("ema", "ema_decay0.97_temp0.1", decay=0.97, temperature=0.1),
        train_mean_mae=train_err.mean(dim=0),
        mode="ema",
    )
    static, _, _ = load_static_winner_per_window(seed, test_cache, std, device)
    weights = 0.5 * static + 0.5 * online
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    pred = weighted_forecast(fixed3_forecasts(test_cache), weights)
    return pred, {"num_updates": extra.get("num_updates"), "mean_weights": weights.mean(dim=0).tolist()}


def dynamic_fixed3_seed7(test_cache: Mapping[str, Any], std: torch.Tensor, device: torch.device) -> tuple[torch.Tensor | None, dict[str, Any]]:
    ckpt_path = ROOT / "checkpoints" / "costarts_walkforward" / "fixed3_dynamic_weighting" / "seed_7" / "best_fixed3_dynamic_weight_router.pt"
    if not ckpt_path.exists():
        return None, {"missing_checkpoint": str(ckpt_path)}
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = Fixed3DynamicWeightRouter(**ckpt["router_config"]).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    result = evaluate_dynamic_fixed3(model, test_cache, device, std, batch_size=1024, ablation="full")
    names = list(test_cache["expert_names"])
    idx = [names.index(name) for name in ("PatchTST", "iTransformer", "TimesNet")]
    forecasts = test_cache["prediction_stack"][..., idx].to(torch.float32)
    weights = []
    with torch.no_grad():
        for i in range(0, forecasts.shape[0], 1024):
            out = model(test_cache["histories"][i : i + 1024].to(device), forecasts[i : i + 1024].to(device), "full")
            weights.append(out["weights"].detach().cpu())
    pred = weighted_forecast(forecasts, torch.cat(weights))
    return pred, {"checkpoint": str(ckpt_path), "best_epoch": ckpt.get("best_epoch"), "mean_weights": result["mean_weights"]}


def ridge_prediction(
    train_cache: Mapping[str, Any],
    test_cache: Mapping[str, Any],
    train_baseline: torch.Tensor,
    test_baseline: torch.Tensor,
    std: torch.Tensor,
    series: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    cfg = RidgeConfig(ridge=1.0, alpha=0.1, clip_multiple=0.25)
    x_train, names, stat_extra = build_feature_tensor(
        train_cache,
        train_cache["absolute_window_starts"].to(torch.long),
        train_baseline,
        std,
        series,
        init_residuals_norm=None,
    )
    y_train, m_train = flattened_targets(train_cache, train_baseline, std)
    x_train = x_train[m_train]
    y_train = y_train[m_train]
    mean, scale = fit_scaler(x_train)
    coef = fit_ridge(apply_scaler(x_train, mean, scale), y_train, cfg.ridge)
    train_resid_norm = (train_cache["targets"].to(torch.float32) - train_baseline) / std.view(1, 1, -1)
    x_test, _, test_stat_extra = build_feature_tensor(
        test_cache,
        test_cache["absolute_window_starts"].to(torch.long),
        test_baseline,
        std,
        series,
        init_residuals_norm=train_resid_norm,
    )
    delta = predict_linear(apply_scaler(x_test, mean, scale), coef)
    pred, extra = apply_residual_delta(
        test_baseline,
        delta,
        std,
        cfg.alpha,
        cfg.clip_multiple,
        train_resid_norm.std(dim=0, unbiased=False).clamp_min(1e-6),
    )
    return pred, {"config": cfg.name, "num_features": len(names), **stat_extra, **test_stat_extra, **extra}


def expanded_prediction(
    train_cache: Mapping[str, Any],
    test_cache: Mapping[str, Any],
    train_base: torch.Tensor,
    test_base: torch.Tensor,
    std: torch.Tensor,
    config: SpecialistConfig,
) -> tuple[torch.Tensor, dict[str, Any]]:
    train_target = train_cache["targets"].to(torch.float32)
    train_mask = train_cache["target_masks"].to(torch.bool)
    test_target = test_cache["targets"].to(torch.float32)
    test_mask = test_cache["target_masks"].to(torch.bool)
    d_train, m_train = optional_predictions(train_cache)
    d_test, m_test = optional_predictions(test_cache)
    pred, extra, _ = run_causal_specialists(
        test_cache["absolute_window_starts"].to(torch.long),
        test_base,
        d_test,
        m_test,
        test_target,
        test_mask,
        std,
        config,
        normalized_abs_error(train_base, train_target, train_mask, std),
        normalized_abs_error(d_train, train_target, train_mask, std),
        normalized_abs_error(m_train, train_target, train_mask, std),
    )
    return pred, {"config": config.name, **extra}


def summarize(
    method: str,
    pred: torch.Tensor,
    test_cache: Mapping[str, Any],
    std: torch.Tensor,
    fixed_core_mae: torch.Tensor,
    validation_key: str,
    seeds: int,
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    met = metrics(test_cache, std, pred)
    boot = paired_bootstrap(met["per_window_mae"], fixed_core_mae, seed=20260813, samples=10000)
    ref = VALIDATION_REFS[validation_key]
    return {
        "method": method,
        "freeze_status": "pre_test_frozen",
        "seeds": seeds,
        "test_mae": met["mae"],
        "test_mse": met["mse"],
        "validation_mae": ref["mae"],
        "validation_mse": ref["mse"],
        "mae_diff_vs_validation": met["mae"] - ref["mae"],
        "mae_diff_vs_test_fixed_core": met["mae"] - float(fixed_core_mae.mean()),
        "mse_diff_vs_test_fixed_core": met["mse"] - FIXED_CORE_TEST_MSE,
        "paired_ci95_diff_vs_fixed_core_low": boot["ci95_low"],
        "paired_ci95_diff_vs_fixed_core_high": boot["ci95_high"],
        "paired_ci_excludes_zero": boot["ci_excludes_zero"],
        **extra,
    }


FIXED_CORE_TEST_MSE = float("nan")


def mean_prediction(preds: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.stack(list(preds)).mean(dim=0)


def mean_metric(values: Sequence[float]) -> tuple[float, float]:
    arr = np.array(list(values), dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def write_report(rows: Sequence[Mapping[str, Any]], payload: Mapping[str, Any]) -> None:
    lines = [
        "# Frozen-Model Top COSTAR Test Results",
        "",
        "Every listed model was trained, selected, configured, and frozen without test-data feedback.",
        "",
        "These are frozen-model test results. The original confirmatory evaluation remains the formally frozen evaluation in `experiments/final_test_evaluation/`; the other rows are additional frozen-model evaluations performed later.",
        "",
        "| Method | Test MAE | Test MSE | Val MAE | Diff vs test fixed core | Seeds | Note |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {test_mae:.6f} | {test_mse:.6f} | {validation_mae:.6f} | {mae_diff_vs_test_fixed_core:+.6f} | {seeds} | {freeze_status} |".format(
                **row
            )
        )
    best = min(rows, key=lambda r: float(r["test_mae"]))
    lines.extend(
        [
            "",
            f"Best frozen-model test MAE: `{best['method']}` at `{best['test_mae']:.6f}`.",
            "",
            "The clean final result remains the preregistered frozen adaptive model from `experiments/final_test_evaluation/`.",
        ]
    )
    (OUT_DIR / "FROZEN_MODEL_TOP_COSTAR_TEST_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    start = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_cache = torch.load(TRAIN_CACHE, map_location="cpu", weights_only=False)
    test_cache = load_test_cache(TEST_CACHE)
    std = load_std(NORMALIZER, int(test_cache["num_features"]))
    series = load_series_with_test()

    fixed_pred = fixed_core_prediction(test_cache)
    fixed_met = metrics(test_cache, std, fixed_pred)
    global FIXED_CORE_TEST_MSE
    FIXED_CORE_TEST_MSE = fixed_met["mse"]
    rows: list[dict[str, Any]] = []
    per_seed: list[dict[str, Any]] = []
    fixed_core_mae = fixed_met["per_window_mae"]
    rows.append(
        summarize(
            "fixed_core_equal",
            fixed_pred,
            test_cache,
            std,
            fixed_core_mae,
            "fixed_core_equal",
            0,
            {"selection_protocol": "anchor: train-selected fixed core"},
        )
    )

    dyn_pred, dyn_extra = dynamic_fixed3_seed7(test_cache, std, device)
    if dyn_pred is not None:
        rows.append(summarize("dynamic_fixed3_seed7", dyn_pred, test_cache, std, fixed_core_mae, "dynamic_fixed3_seed7", 1, dyn_extra))

    train_base_seed7, _ = fixed_current_best_prediction(train_cache, train_cache, std, 7, device)
    hv_preds = []
    chrono_preds = []
    oracle_preds = []
    ridge_preds = []
    mlp_preds = []
    expanded_configs = {
        "expanded_dlinear_only": SpecialistConfig("dlinear_only", "variable", 0.95, 0.10, 0.02, 96),
        "expanded_moderntcn_only": SpecialistConfig("moderntcn_only", "variable", 0.95, 0.05, 0.02, 96),
        "expanded_both_final_frozen": SpecialistConfig("both", "variable", 0.95, 0.10, 0.02, 96),
    }
    expanded_preds: dict[str, list[torch.Tensor]] = {key: [] for key in expanded_configs}
    for seed in SEEDS:
        seed = int(seed)
        static_w, _, _ = load_static_winner_per_window(seed, test_cache, std, device)
        oracle_pred = weighted_forecast(fixed3_forecasts(test_cache), static_w)
        oracle_preds.append(oracle_pred)
        per_seed.append({"method": "oracle_prototype_residual", "seed": seed, **metrics(test_cache, std, oracle_pred)})

        chrono_pred, _ = chronological_prediction(train_cache, test_cache, std, seed, device)
        chrono_preds.append(chrono_pred)
        per_seed.append({"method": "chronological_ema_hybrid", "seed": seed, **metrics(test_cache, std, chrono_pred)})

        hv_pred, _ = fixed_current_best_prediction(test_cache, train_cache, std, seed, device)
        hv_preds.append(hv_pred)
        per_seed.append({"method": "horizon_variable_hybrid", "seed": seed, **metrics(test_cache, std, hv_pred)})

        ridge_pred, _ = ridge_prediction(train_cache, test_cache, train_base_seed7, hv_pred, std, series)
        ridge_preds.append(ridge_pred)
        per_seed.append({"method": "ridge_residual_corrector", "seed": seed, **metrics(test_cache, std, ridge_pred)})

        mlp_cfg = MlpConfig(seed=seed, alpha=0.1, clip_multiple=0.25)
        mlp_pred, _ = train_mlp_final(train_cache, test_cache, train_base_seed7, hv_pred, std, series, mlp_cfg, device)
        mlp_preds.append(mlp_pred)
        per_seed.append({"method": "mlp_residual_corrector", "seed": seed, **metrics(test_cache, std, mlp_pred)})

        for key, cfg in expanded_configs.items():
            expanded_pred, _ = expanded_prediction(train_cache, test_cache, train_base_seed7, hv_pred, std, cfg)
            expanded_preds[key].append(expanded_pred)
            per_seed.append({"method": key, "seed": seed, **metrics(test_cache, std, expanded_pred)})

    for key, preds in (
        ("oracle_prototype_residual", oracle_preds),
        ("chronological_ema_hybrid", chrono_preds),
        ("horizon_variable_hybrid", hv_preds),
        ("ridge_residual_corrector", ridge_preds),
        ("mlp_residual_corrector", mlp_preds),
    ):
        maes = [r["mae"] for r in per_seed if r["method"] == key]
        mses = [r["mse"] for r in per_seed if r["method"] == key]
        rows.append(
            summarize(
                key,
                mean_prediction(preds),
                test_cache,
                std,
                fixed_core_mae,
                key,
                len(preds),
                {"mae_seed_mean": mean_metric(maes)[0], "mae_seed_std": mean_metric(maes)[1], "mse_seed_mean": mean_metric(mses)[0], "mse_seed_std": mean_metric(mses)[1]},
            )
        )
    for key, preds in expanded_preds.items():
        maes = [r["mae"] for r in per_seed if r["method"] == key]
        mses = [r["mse"] for r in per_seed if r["method"] == key]
        rows.append(
            summarize(
                key,
                mean_prediction(preds),
                test_cache,
                std,
                fixed_core_mae,
                key,
                len(preds),
                {"mae_seed_mean": mean_metric(maes)[0], "mae_seed_std": mean_metric(maes)[1], "mse_seed_mean": mean_metric(mses)[0], "mse_seed_std": mean_metric(mses)[1]},
            )
        )

    rows = sorted(rows, key=lambda r: float(r["test_mae"]))
    payload = {
        "freeze_status": "pre_test_frozen",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "elapsed_seconds": time.perf_counter() - start,
        "final_test_results_source": str(FINAL_TEST_DIR / "FINAL_TEST_RESULTS.json"),
        "test_cache": str(TEST_CACHE),
        "router_train_cache": str(TRAIN_CACHE),
        "normalizer": str(NORMALIZER),
        "results": rows,
        "per_seed": [
            {k: v for k, v in row.items() if k not in {"per_window_mae", "per_window_mse"}}
            for row in per_seed
        ],
    }
    write_csv(OUT_DIR / "top_costar_test_results.csv", rows)
    write_csv(OUT_DIR / "top_costar_test_per_seed.csv", payload["per_seed"])
    write_json(OUT_DIR / "TOP_COSTAR_TEST_RESULTS.json", payload)
    write_report(rows, payload)
    print(json.dumps({"freeze_status": payload["freeze_status"], "device": str(device), "results": rows}, indent=2))


if __name__ == "__main__":
    main()
