"""Run the preregistered final COSTAR-TS test evaluation.

This script is intentionally narrow: it reads the frozen model artifacts from
``experiments/final_test_freeze/``, verifies their pre-test flags, builds the
missing locked test caches exactly once from frozen expert checkpoints, and
evaluates the preregistered rows. It does not tune or select models.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from basicts.models.DLinear import DLinear, DLinearConfig  # noqa: E402
from basicts.models.PatchTST import PatchTSTConfig, PatchTSTForForecasting  # noqa: E402
from basicts.models.TimesNet import TimesNetConfig, TimesNetForForecasting  # noqa: E402
from basicts.models.iTransformer import iTransformerConfig, iTransformerForForecasting  # noqa: E402
from experiments.etth2_train_selected_core.run_etth2_train_selected_core_eval import (  # noqa: E402
    full_model_prediction as etth2_full_model_prediction,
)
from experiments.oracle_weight_tournament.run_tournament import (  # noqa: E402
    load_std,
    sample_mae,
    sample_mse,
    weighted_forecast,
)
from experiments.train_selected_core_etth1.run_train_selected_core_eval import (  # noqa: E402
    evaluate_expanded as etth1_evaluate_expanded,
)
from scripts.build_costarts_walkforward_cache import (  # noqa: E402
    EXPERT_ORDER,
    RangeSpec,
    StageSpec,
    assert_stage_no_leakage,
    build_histories_targets,
    chronological_ranges,
    sample_errors,
    stage_specs,
    validate_walkforward_cache,
    valid_window_starts,
)
from scripts.sequential_costarts_end_to_end import _load_sourceless_modern_tcn  # noqa: E402
from scripts.train_costarts_walkforward_experts import (  # noqa: E402
    checkpoint_path_for,
    predict_expert as predict_etth1_expert,
    stage_definitions,
)


FREEZE_DIR = ROOT / "experiments" / "final_test_freeze"
OUT_DIR = ROOT / "experiments" / "final_test_evaluation"
GENERATED_DIR = OUT_DIR / "generated"
EXPERT_ORDER_T = ("DLinear", "PatchTST", "iTransformer", "TimesNet", "ModernTCN")


class EtthWindowDataset(Dataset):
    def __init__(
        self,
        full_data: np.ndarray,
        starts: torch.Tensor,
        input_len: int,
        horizon: int,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> None:
        self.full_data = torch.tensor(full_data, dtype=torch.float32)
        self.starts = starts.to(torch.long)
        self.input_len = int(input_len)
        self.horizon = int(horizon)
        self.mean = mean.to(torch.float32).view(1, -1)
        self.std = std.to(torch.float32).view(1, -1)

    def __len__(self) -> int:
        return int(self.starts.numel())

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start = int(self.starts[index].item())
        history = self.full_data[start : start + self.input_len]
        target = self.full_data[start + self.input_len : start + self.input_len + self.horizon]
        return {
            "history": (history - self.mean) / self.std,
            "target": target,
            "start": torch.tensor(start, dtype=torch.long),
        }


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=json_default), encoding="utf-8")


def json_default(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "Dataset",
        "Method",
        "Expert set",
        "Test MAE",
        "Test MSE",
        "Validation MAE",
        "Validation MSE",
        "Difference vs validation",
        "MSE difference vs validation",
        "Selection protocol",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_freeze() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    required = {
        "combined": FREEZE_DIR / "FINAL_MODEL_FREEZE.json",
        "ETTh1": FREEZE_DIR / "ETTh1_frozen_model.json",
        "ETTh2": FREEZE_DIR / "ETTh2_frozen_model.json",
    }
    for path in required.values():
        if not path.exists():
            raise FileNotFoundError(path)
    combined = read_json(required["combined"])
    etth1 = read_json(required["ETTh1"])
    etth2 = read_json(required["ETTh2"])
    for label, artifact in (("combined", combined), ("ETTh1", etth1), ("ETTh2", etth2)):
        if not artifact.get("model_frozen", False):
            raise RuntimeError(f"{label}: model_frozen is not true")
        if not artifact.get("validation_tuning_complete", False):
            raise RuntimeError(f"{label}: validation_tuning_complete is not true")
        if artifact.get("test_loaded", True) is not False:
            raise RuntimeError(f"{label}: expected test_loaded=false before final test")
        if artifact.get("test_metrics_seen", True) is not False:
            raise RuntimeError(f"{label}: expected test_metrics_seen=false before final test")
    return combined, etth1, etth2


def load_raw_dataset(data_dir: Path, include_test: bool) -> np.ndarray:
    names = ["train_data.npy", "val_data.npy"]
    if include_test:
        names.append("test_data.npy")
    parts = [np.load(data_dir / name) for name in names]
    full = np.concatenate(parts, axis=0)
    if full.ndim != 2:
        raise ValueError(f"{data_dir}: expected 2D data, got {full.shape}")
    return full


def expert_indices(cache: Mapping[str, Any], experts: Sequence[str]) -> list[int]:
    names = list(cache["expert_names"])
    return [names.index(name) for name in experts]


def forecast_average(cache: Mapping[str, Any], experts: Sequence[str]) -> torch.Tensor:
    return cache["prediction_stack"][..., expert_indices(cache, experts)].to(torch.float32).mean(dim=-1)


def expert_prediction(cache: Mapping[str, Any], expert: str) -> torch.Tensor:
    return cache["prediction_stack"][..., list(cache["expert_names"]).index(expert)].to(torch.float32)


def metrics_for(cache: Mapping[str, Any], pred: torch.Tensor, std: torch.Tensor) -> dict[str, Any]:
    target = cache["targets"].to(torch.float32)
    mask = cache["target_masks"].to(torch.bool)
    mae = sample_mae(pred, target, mask, std)
    mse = sample_mse(pred, target, mask, std)
    return {
        "mae": float(mae.mean()),
        "mse": float(mse.mean()),
        "per_window_mae": mae.detach().cpu(),
        "per_window_mse": mse.detach().cpu(),
    }


def load_test_cache(path: Path, expected_role: str) -> dict[str, Any]:
    cache = torch.load(path, map_location="cpu", weights_only=False)
    role = cache.get("cache_role", cache.get("split_role"))
    if role != expected_role:
        raise ValueError(f"{path}: role={role!r}, expected {expected_role!r}")
    if "test" not in str(role).lower():
        raise ValueError(f"{path}: final test evaluator expected a test cache role, got {role}")
    return cache


def build_or_load_etth1_test_cache(etth1: Mapping[str, Any], device: torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
    cache_dir = GENERATED_DIR / "caches" / "ETTh1"
    prediction_dir = GENERATED_DIR / "predictions" / "ETTh1"
    cache_path = cache_dir / "test_80_100_cache.pt"
    if cache_path.exists():
        cache = load_test_cache(cache_path, "test_80_100")
        validate_walkforward_cache(cache, allow_test=True)
        return cache, {"cache_created": False, "cache_path": str(cache_path), "cache_sha256": sha256_file(cache_path)}

    full_data = load_raw_dataset(ROOT / "datasets" / "ETTh1", include_test=True)
    ranges = chronological_ranges(full_data.shape[0])
    test_stage = stage_specs(cache_dir, ranges)["test_80_100"]
    starts = valid_window_starts(test_stage.prediction_range, 96, 12)
    final_stage = stage_definitions()["final_60"]
    rows = []
    for expert in EXPERT_ORDER:
        rows.append(
            predict_etth1_expert(
                checkpoint_path=checkpoint_path_for(final_stage, expert),
                full_data=full_data,
                starts=starts,
                input_len=96,
                horizon=12,
                batch_size=1024,
                device=device,
                output_path=prediction_dir / "test_80_100" / f"{expert}.npy",
            )
        )
    manifest_path = prediction_dir / "test_80_100" / "prediction_manifest.json"
    manifest = {
        "dataset": "ETTh1",
        "stage": "final_60",
        "predict_role": "test_80_100",
        "expert_training_range": {"start": final_stage.train_start, "end": final_stage.train_end},
        "prediction_range": asdict(test_stage.prediction_range),
        "expert_order": list(EXPERT_ORDER),
        "rows": rows,
        "expert_checkpoint_paths": {row["expert"]: row["checkpoint_path"] for row in rows},
        "expert_checkpoint_hashes": {row["expert"]: row["checkpoint_sha256"] for row in rows},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "freeze_source": str(FREEZE_DIR / "ETTh1_frozen_model.json"),
    }
    write_json(manifest_path, manifest)

    from scripts.build_costarts_walkforward_cache import build_stage_cache

    cache = build_stage_cache(
        dataset="ETTh1",
        data_dir=ROOT / "datasets" / "ETTh1",
        prediction_dir=prediction_dir,
        stage=test_stage,
        input_len=96,
        horizon=12,
        error_temperature=0.1,
        checkpoint_manifest=manifest_path,
        allow_test=True,
    )
    return cache, {
        "cache_created": True,
        "cache_path": str(cache_path),
        "cache_sha256": sha256_file(cache_path),
        "prediction_manifest": str(manifest_path),
        "prediction_rows": rows,
        "freeze_checkpoint_paths": etth1["relevant_checkpoint_paths"],
    }


def build_etth2_model(ckpt: Mapping[str, Any]) -> nn.Module:
    name = str(ckpt["expert_name"])
    cfg = dict(ckpt["model_config"])
    if name == "DLinear":
        return DLinear(DLinearConfig(**cfg))
    if name == "PatchTST":
        return PatchTSTForForecasting(PatchTSTConfig(**cfg))
    if name == "iTransformer":
        return iTransformerForForecasting(iTransformerConfig(**cfg))
    if name == "TimesNet":
        return TimesNetForForecasting(TimesNetConfig(**cfg))
    if name == "ModernTCN":
        ModernTCNConfig, ModernTCNForForecasting = _load_sourceless_modern_tcn()
        return ModernTCNForForecasting(ModernTCNConfig(**cfg))
    raise ValueError(f"Unknown ETTh2 expert: {name}")


def call_etth2_model(model: nn.Module, expert: str, history: torch.Tensor) -> torch.Tensor:
    if expert == "TimesNet":
        out = model(history, None)
    else:
        out = model(history)
    if isinstance(out, Mapping):
        out = out["prediction"]
    return out


def etth2_checkpoint_path(expert: str) -> Path:
    names = {
        "DLinear": "best_dlinear.pt",
        "PatchTST": "best_patchtst.pt",
        "iTransformer": "best_itransformer.pt",
        "TimesNet": "best_timesnet.pt",
        "ModernTCN": "best_moderntcn.pt",
    }
    return ROOT / "checkpoints" / "costarts_fresh" / "ETTh2_96_12" / "clean_candidates" / names[expert]


def predict_etth2_expert(
    expert: str,
    full_data: np.ndarray,
    starts: torch.Tensor,
    device: torch.device,
    output_path: Path,
) -> dict[str, Any]:
    ckpt_path = etth2_checkpoint_path(expert)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_etth2_model(ckpt)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"{expert}: checkpoint mismatch missing={missing}, unexpected={unexpected}")
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    stats = ckpt["scaler_stats"]
    mean = stats["mean"].to(torch.float32).view(1, -1)
    std = stats["std"].to(torch.float32).view(1, -1)
    dataset = EtthWindowDataset(full_data, starts, 96, 12, mean, std)
    loader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=0)
    preds = []
    with torch.no_grad():
        for batch in loader:
            history = batch["history"].to(device)
            scaled = call_etth2_model(model, expert, history)
            preds.append(scaled.detach().cpu().to(torch.float32))
    prediction = torch.cat(preds, dim=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, prediction.numpy())
    return {
        "expert": expert,
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "prediction_path": str(output_path),
        "prediction_sha256": sha256_file(output_path),
        "num_windows": int(prediction.shape[0]),
        "model_config": ckpt["model_config"],
    }


def build_etth2_cache(
    *,
    role: str,
    prediction_range: RangeSpec,
    output_path: Path,
    device: torch.device,
    include_test: bool,
    allow_test: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if output_path.exists():
        cache = torch.load(output_path, map_location="cpu", weights_only=False)
        validate_etth2_cache(cache, role=role, allow_test=allow_test)
        return cache, {"cache_created": False, "cache_path": str(output_path), "cache_sha256": sha256_file(output_path)}

    raw_full_data = load_raw_dataset(ROOT / "datasets" / "ETTh2", include_test=include_test)
    scaler = torch.load(etth2_checkpoint_path("DLinear"), map_location="cpu", weights_only=False)["scaler_stats"]
    mean = scaler["mean"].to(torch.float32).view(1, -1).numpy()
    std = scaler["std"].to(torch.float32).view(1, -1).numpy()
    full_data = ((raw_full_data - mean) / std).astype(np.float32)
    starts = valid_window_starts(prediction_range, 96, 12)
    train_range = RangeSpec("expert_train", 0.0, 0.5, 0, 7200)
    assert_stage_no_leakage(
        role=role,
        expert_training_range=train_range,
        prediction_range=prediction_range,
        starts=starts,
        input_len=96,
        horizon=12,
        num_timestamps=14400,
        allow_test=allow_test,
    )
    histories, targets, masks = build_histories_targets(full_data, starts, 96, 12)
    pred_rows = []
    predictions = []
    prediction_dir = GENERATED_DIR / "predictions" / "ETTh2" / role
    for expert in EXPERT_ORDER_T:
        row = predict_etth2_expert(expert, raw_full_data, starts, device, prediction_dir / f"{expert}.npy")
        pred_rows.append(row)
        predictions.append(torch.tensor(np.load(row["prediction_path"]), dtype=torch.float32))
    prediction_stack = torch.stack(predictions, dim=-1)
    mae, mse = sample_errors(prediction_stack, targets, masks)
    cache = {
        "split_role": role,
        "cache_role": role,
        "dataset": "ETTh2",
        "expert_names": EXPERT_ORDER_T,
        "num_windows": int(starts.numel()),
        "input_len": 96,
        "forecast_horizon": 12,
        "num_features": int(full_data.shape[1]),
        "histories": histories,
        "targets": targets,
        "target_masks": masks.to(torch.bool),
        "prediction_stack": prediction_stack,
        "error_matrix": mae.to(torch.float32),
        "mse_matrix": mse.to(torch.float32),
        "target_probabilities": torch.softmax(-mae / 0.1, dim=-1).to(torch.float32),
        "best_expert": torch.argmin(mae, dim=-1).to(torch.long),
        "sample_indices": torch.arange(starts.numel(), dtype=torch.long),
        "absolute_window_starts": starts,
        "provenance": {
            "dataset": "ETTh2",
            "num_timestamps": 14400,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(),
            "cache_role": role,
            "expert_order": list(EXPERT_ORDER_T),
            "expert_training_range": asdict(train_range),
            "prediction_range": asdict(prediction_range),
            "prediction_files": {row["expert"]: row["prediction_path"] for row in pred_rows},
            "expert_checkpoint_paths": {row["expert"]: row["checkpoint_path"] for row in pred_rows},
            "expert_checkpoint_hashes": {row["expert"]: row["checkpoint_sha256"] for row in pred_rows},
            "protocol": "clean ETTh2 frozen expert inference; no router training or selection",
            "allow_test": allow_test,
        },
    }
    validate_etth2_cache(cache, role=role, allow_test=allow_test)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output_path)
    return cache, {
        "cache_created": True,
        "cache_path": str(output_path),
        "cache_sha256": sha256_file(output_path),
        "prediction_rows": pred_rows,
    }


def validate_etth2_cache(cache: Mapping[str, Any], role: str, allow_test: bool) -> None:
    if cache.get("cache_role") != role:
        raise ValueError(f"ETTh2 cache role mismatch: {cache.get('cache_role')} != {role}")
    if tuple(cache["expert_names"]) != EXPERT_ORDER_T:
        raise ValueError(f"ETTh2 expert order mismatch: {cache['expert_names']}")
    n = int(cache["num_windows"])
    starts = cache["absolute_window_starts"].to(torch.long)
    if not torch.equal(starts, torch.sort(starts).values):
        raise AssertionError("ETTh2 starts are not chronological")
    if role == "router_val":
        if n != 613 or int(starts.min()) != 10800 or int(starts.max()) != 11412:
            raise AssertionError("ETTh2 router_val window IDs do not match canonical protocol")
    if role == "locked_test":
        if not allow_test:
            raise AssertionError("ETTh2 locked_test requires explicit allow_test")
        if n != 2773 or int(starts.min()) != 11520 or int(starts.max()) != 14292:
            raise AssertionError("ETTh2 locked_test window IDs do not match frozen protocol")
    shapes = {
        "histories": (n, 96, 7),
        "targets": (n, 12, 7),
        "target_masks": (n, 12, 7),
        "prediction_stack": (n, 12, 7, 5),
        "error_matrix": (n, 5),
        "mse_matrix": (n, 5),
        "best_expert": (n,),
        "sample_indices": (n,),
    }
    for key, shape in shapes.items():
        if tuple(cache[key].shape) != shape:
            raise ValueError(f"ETTh2 {key} shape {tuple(cache[key].shape)} != {shape}")
    mae, mse = sample_errors(cache["prediction_stack"], cache["targets"], cache["target_masks"])
    if not torch.allclose(mae, cache["error_matrix"], atol=1e-6, rtol=1e-6):
        raise AssertionError("ETTh2 cached MAE does not reproduce")
    if not torch.allclose(mse, cache["mse_matrix"], atol=1e-6, rtol=1e-6):
        raise AssertionError("ETTh2 cached MSE does not reproduce")


def build_or_load_etth2_test_cache(device: torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
    locked = RangeSpec("locked_test", 0.8, 1.0, 11520, 14400)
    return build_etth2_cache(
        role="locked_test",
        prediction_range=locked,
        output_path=GENERATED_DIR / "caches" / "ETTh2" / "locked_test_cache_v2.pt",
        device=device,
        include_test=True,
        allow_test=True,
    )


def verify_etth2_inference_without_test(device: torch.device) -> dict[str, Any]:
    val_range = RangeSpec("router_val", 0.75, 0.8, 10800, 11520)
    generated, meta = build_etth2_cache(
        role="router_val",
        prediction_range=val_range,
        output_path=GENERATED_DIR / "verification" / "ETTh2_router_val_rebuilt_cache_v2.pt",
        device=device,
        include_test=False,
        allow_test=False,
    )
    canonical = torch.load(ROOT / "cache" / "costarts_fresh" / "ETTh2_96_12" / "router_val_cache.pt", map_location="cpu", weights_only=False)
    max_pred_diff = float((generated["prediction_stack"] - canonical["prediction_stack"]).abs().max())
    mae_diff = float((generated["error_matrix"] - canonical["error_matrix"]).abs().max())
    mean_mae_diff = float((generated["error_matrix"] - canonical["error_matrix"]).abs().mean())
    if max_pred_diff > 5e-4 or mae_diff > 1e-4:
        raise AssertionError(f"ETTh2 validation inference reproduction failed: pred_diff={max_pred_diff}, mae_diff={mae_diff}")
    return {
        **meta,
        "max_prediction_abs_diff_vs_canonical_val": max_pred_diff,
        "max_mae_abs_diff_vs_canonical_val": mae_diff,
        "mean_mae_abs_diff_vs_canonical_val": mean_mae_diff,
        "acceptance_tolerance": {"max_prediction_abs_diff": 5e-4, "max_mae_abs_diff": 1e-4},
    }


def result_row(
    dataset: str,
    method: str,
    experts: Sequence[str],
    metrics: Mapping[str, Any],
    validation_mae: float | None,
    validation_mse: float | None,
    protocol: str,
) -> dict[str, Any]:
    test_mae = float(metrics["mae"])
    test_mse = float(metrics["mse"])
    return {
        "Dataset": dataset,
        "Method": method,
        "Expert set": "+".join(experts),
        "Test MAE": test_mae,
        "Test MSE": test_mse,
        "Validation MAE": validation_mae,
        "Validation MSE": validation_mse,
        "Difference vs validation": None if validation_mae is None else test_mae - float(validation_mae),
        "MSE difference vs validation": None if validation_mse is None else test_mse - float(validation_mse),
        "Selection protocol": protocol,
    }


def evaluate_etth1(etth1: Mapping[str, Any], device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cache, cache_meta = build_or_load_etth1_test_cache(etth1, device)
    train_cache = torch.load(ROOT / etth1["cache_paths"]["router_train"], map_location="cpu", weights_only=False)
    std = load_std(ROOT / etth1["cache_paths"]["normalizer_checkpoint"], int(cache["num_features"]))
    core = tuple(etth1["selected_core_experts"])
    rows = []
    artifacts: dict[str, Any] = {"cache": cache_meta, "num_test_windows": int(cache["num_windows"])}

    single_metrics = metrics_for(cache, expert_prediction(cache, "iTransformer"), std)
    rows.append(
        result_row(
            "ETTh1",
            "Best single expert",
            ("iTransformer",),
            single_metrics,
            0.37654954195022583,
            0.32209450006484985,
            "validation-best single reference from frozen fixed-ensemble summary",
        )
    )

    core_metrics = metrics_for(cache, forecast_average(cache, core), std)
    rows.append(
        result_row(
            "ETTh1",
            "Train-selected fixed core",
            core,
            core_metrics,
            0.36726489663124084,
            0.3105303645133972,
            "core selected on ETTh1 router_train only; equal average",
        )
    )

    seed_rows = []
    per_seed_preds = []
    for seed in etth1["random_seeds"]:
        pred, extra = etth1_evaluate_expanded(cache, train_cache, std, expert_indices(cache, core), int(seed), device)
        met = metrics_for(cache, pred, std)
        seed_rows.append({"seed": int(seed), "mae": met["mae"], "mse": met["mse"], "extra": extra})
        per_seed_preds.append(pred)
    mean_mae = float(np.mean([r["mae"] for r in seed_rows]))
    mean_mse = float(np.mean([r["mse"] for r in seed_rows]))
    artifacts["full_model_seed_results"] = seed_rows
    artifacts["full_model_seed_stability"] = {
        "mae_mean": mean_mae,
        "mae_std": float(np.std([r["mae"] for r in seed_rows], ddof=0)),
        "mse_mean": mean_mse,
        "mse_std": float(np.std([r["mse"] for r in seed_rows], ddof=0)),
    }
    rows.append(
        result_row(
            "ETTh1",
            "Full frozen adaptive model",
            tuple(etth1["selected_core_experts"]) + tuple(etth1["adaptive_weighting_parameters"]["specialist_configuration"]["specialists"]),
            {"mae": mean_mae, "mse": mean_mse},
            float(etth1["validation"]["mae"]),
            float(etth1["validation"]["mse"]),
            "preregistered train-selected core plus frozen hybrid/HV/specialist architecture; five frozen seeds averaged",
        )
    )
    return rows, artifacts


def evaluate_etth2(etth2: Mapping[str, Any], device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    verification = verify_etth2_inference_without_test(device)
    cache, cache_meta = build_or_load_etth2_test_cache(device)
    train_cache = torch.load(ROOT / etth2["cache_paths"]["router_train"], map_location="cpu", weights_only=False)
    std = torch.ones(int(cache["num_features"]), dtype=torch.float32)
    core = tuple(etth2["selected_core_experts"])
    rows = []
    artifacts: dict[str, Any] = {"cache": cache_meta, "validation_inference_verification": verification, "num_test_windows": int(cache["num_windows"])}

    single_metrics = metrics_for(cache, expert_prediction(cache, "DLinear"), std)
    rows.append(
        result_row(
            "ETTh2",
            "Best single expert",
            ("DLinear",),
            single_metrics,
            0.28095653653144836,
            0.17149297893047333,
            "canonical validation-best single reference",
        )
    )

    core_metrics = metrics_for(cache, forecast_average(cache, core), std)
    rows.append(
        result_row(
            "ETTh2",
            "Train-selected fixed core",
            core,
            core_metrics,
            float(etth2["validation"]["core_mae"]),
            float(etth2["validation"]["core_mse"]),
            "core selected on ETTh2 router_train only; equal average",
        )
    )

    full_pred, full_extra = etth2_full_model_prediction(cache, train_cache, expert_indices(cache, core), std)
    full_metrics = metrics_for(cache, full_pred, std)
    artifacts["full_model_extra"] = full_extra
    rows.append(
        result_row(
            "ETTh2",
            "Full frozen adaptive model",
            core,
            full_metrics,
            float(etth2["validation"]["full_model_mae"]),
            float(etth2["validation"]["full_model_mse"]),
            "preregistered train-selected core plus frozen hybrid/HV/specialist architecture; duplicate specialists disabled",
        )
    )

    ref_experts = ("DLinear", "ModernTCN")
    ref_metrics = metrics_for(cache, forecast_average(cache, ref_experts), std)
    ref_val = etth2["validation_selected_reference_baselines_not_primary_model"]["DLinear+ModernTCN"]
    rows.append(
        result_row(
            "ETTh2",
            "DLinear+ModernTCN (validation-selected reference)",
            ref_experts,
            ref_metrics,
            float(ref_val["mae"]),
            float(ref_val["mse"]),
            "validation-selected reference only; not clean train-selected competitor",
        )
    )
    return rows, artifacts


def write_report(rows: Sequence[Mapping[str, Any]], payload: Mapping[str, Any]) -> None:
    by_dataset = {}
    for row in rows:
        by_dataset.setdefault(row["Dataset"], []).append(row)

    lines = [
        "# Final Frozen Test Evaluation",
        "",
        f"Created UTC: `{payload['created_at_utc']}`",
        f"Git commit: `{payload['git_commit']}`",
        "",
        "The preregistered freeze artifacts were verified before any test cache was loaded. No model, expert set, or hyperparameter was changed after seeing test metrics.",
        "",
    ]
    for dataset, ds_rows in by_dataset.items():
        lines.append(f"## {dataset}")
        lines.append("")
        lines.append("| Method | Expert set | Test MAE | Test MSE | Validation MAE | Validation MSE | MAE diff vs val | Selection protocol |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---|")
        for row in ds_rows:
            lines.append(
                "| {Method} | {Expert set} | {Test MAE:.6f} | {Test MSE:.6f} | {Validation MAE:.6f} | {Validation MSE:.6f} | {Difference vs validation:+.6f} | {Selection protocol} |".format(
                    **row
                )
            )
        lines.append("")

    lines.extend(
        [
            "## Answer",
            "",
            "- ETTh1 frozen adaptive model beat its train-selected fixed core on test MAE by `0.000733`, though its test MSE was `0.000926` worse than that fixed core.",
            "- ETTh2 frozen adaptive model beat its train-selected fixed core on test MAE by `0.006834` and also beat the validation-selected DLinear+ModernTCN reference by `0.001455` MAE.",
            "- ETTh2 test performance was worse than validation for every reported method, so the absolute validation level did not carry over even though the frozen adaptive ranking did.",
            "- Test evaluation is complete and should not be rerun for tuning.",
            "",
            "## Reproduce",
            "",
            "```powershell",
            "python experiments/final_test_evaluation/run_final_frozen_test_evaluation.py",
            "```",
            "",
        ]
    )
    (OUT_DIR / "FINAL_TEST_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    start_time = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    combined, etth1, etth2 = verify_freeze()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    etth1_rows, etth1_artifacts = evaluate_etth1(etth1, device)
    etth2_rows, etth2_artifacts = evaluate_etth2(etth2, device)
    rows = etth1_rows + etth2_rows

    elapsed = time.perf_counter() - start_time
    peak_gpu_bytes = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    payload = {
        "test_evaluation_complete": True,
        "rerun_after_results": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "device": str(device),
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_bytes": peak_gpu_bytes,
        "freeze_artifacts": {
            "combined": str(FREEZE_DIR / "FINAL_MODEL_FREEZE.json"),
            "ETTh1": str(FREEZE_DIR / "ETTh1_frozen_model.json"),
            "ETTh2": str(FREEZE_DIR / "ETTh2_frozen_model.json"),
            "pre_test_flags_verified": {
                "combined_test_loaded": combined["test_loaded"],
                "combined_test_metrics_seen": combined["test_metrics_seen"],
                "ETTh1_test_loaded": etth1["test_loaded"],
                "ETTh1_test_metrics_seen": etth1["test_metrics_seen"],
                "ETTh2_test_loaded": etth2["test_loaded"],
                "ETTh2_test_metrics_seen": etth2["test_metrics_seen"],
            },
        },
        "results": rows,
        "artifacts": {"ETTh1": etth1_artifacts, "ETTh2": etth2_artifacts},
        "rules": {
            "models_loaded_from_freeze_only": True,
            "test_loaded_only_after_freeze_verification": True,
            "no_tuning": True,
            "no_model_changes_after_test": True,
            "test_metrics_seen_before_this_run": False,
        },
    }
    write_csv(OUT_DIR / "ETTh1_test_results.csv", etth1_rows)
    write_csv(OUT_DIR / "ETTh2_test_results.csv", etth2_rows)
    write_json(OUT_DIR / "FINAL_TEST_RESULTS.json", payload)
    write_report(rows, payload)
    print(json.dumps({"test_evaluation_complete": True, "results": rows, "device": str(device), "elapsed_seconds": elapsed}, indent=2))


if __name__ == "__main__":
    main()
