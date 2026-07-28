"""Train clean ETTh2 96->12 frozen experts without loading test arrays."""

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from basicts.scaler import ZScoreScaler
from scripts.chronological_expert_training import (
    DEFAULT_INPUT_LEN,
    DEFAULT_NUM_FEATURES,
    DEFAULT_OUTPUT_LEN,
    _accumulate_errors,
    _configured_forecasting_loss,
    _prepare_forecasting_batch,
    _prediction_tensor,
)
from scripts.costars.train_candidate_experts import EXPERT_SPECS, ExpertSpec


SPLIT_FRACTIONS = {
    "expert_train": (0.00, 0.50),
    "expert_val": (0.50, 0.60),
    "router_train": (0.60, 0.75),
    "router_val": (0.75, 0.80),
    "locked_test": (0.80, 1.00),
}
SPLIT_ORDER = tuple(SPLIT_FRACTIONS)
EXPERT_ORDER = ("DLinear", "PatchTST", "iTransformer", "TimesNet", "ModernTCN")


class AbsoluteWindowDataset(Dataset):
    """Windows inside one absolute chronological split.

    The backing array may be only a prefix of the full dataset. Construction
    fails if a requested split needs indices beyond that loaded prefix.
    """

    def __init__(
        self,
        loaded_prefix: np.ndarray,
        split_manifest: Dict[str, dict],
        split_role: str,
        input_len: int = DEFAULT_INPUT_LEN,
        output_len: int = DEFAULT_OUTPUT_LEN,
    ) -> None:
        if split_role not in split_manifest:
            raise ValueError(f"Unknown split_role {split_role!r}")
        boundary = split_manifest[split_role]
        if boundary["end"] > len(loaded_prefix):
            raise ValueError(
                f"{split_role} ends at {boundary['end']}, but only "
                f"{len(loaded_prefix)} pre-test timestamps were loaded"
            )
        self.loaded_prefix = loaded_prefix
        self.split_manifest = split_manifest
        self.split_role = split_role
        self.input_len = input_len
        self.output_len = output_len
        self.boundary = boundary
        self._data = loaded_prefix[boundary["start"] : boundary["end"]]

    def __len__(self) -> int:
        return int(self.boundary["num_windows"])

    def __getitem__(self, index: int) -> dict:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        absolute_start = self.boundary["first_valid_window_start"] + index
        relative_start = absolute_start - self.boundary["start"]
        target_start = relative_start + self.input_len
        return {
            "inputs": self._data[relative_start:target_start],
            "targets": self._data[target_start : target_start + self.output_len],
            "absolute_window_start": absolute_start,
        }

    @property
    def data(self) -> np.ndarray:
        return self._data


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _json_default(value: object):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON encode {type(value)!r}")


def config_to_dict(config: object) -> dict:
    if isinstance(config, dict):
        return dict(config)
    return asdict(config)


def split_manifest_for_total_length(
    total_length: int,
    input_len: int = DEFAULT_INPUT_LEN,
    output_len: int = DEFAULT_OUTPUT_LEN,
) -> Dict[str, dict]:
    cut_points = (
        0,
        int(total_length * 0.50),
        int(total_length * 0.60),
        int(total_length * 0.75),
        int(total_length * 0.80),
        total_length,
    )
    manifest = {}
    for index, role in enumerate(SPLIT_ORDER):
        start = cut_points[index]
        end = cut_points[index + 1]
        windows = end - start - input_len - output_len + 1
        if windows <= 0:
            raise ValueError(f"{role} has no valid forecasting windows")
        manifest[role] = {
            "fraction": list(SPLIT_FRACTIONS[role]),
            "start": start,
            "end": end,
            "num_timestamps": end - start,
            "first_valid_window_start": start,
            "last_valid_window_start": end - input_len - output_len,
            "num_windows": windows,
        }
    return manifest


def load_etth2_expert_prefix(data_dir: Path) -> Tuple[np.ndarray, dict]:
    """Load only the ETTh2 prefix needed for expert_train and expert_val."""

    data_dir = Path(data_dir)
    metadata = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    total_length = int(metadata["num_time_steps"])
    num_vars = int(metadata["num_vars"])
    if num_vars != DEFAULT_NUM_FEATURES:
        raise ValueError(f"Expected {DEFAULT_NUM_FEATURES} variables, found {num_vars}")

    split_manifest = split_manifest_for_total_length(total_length)
    train_data = np.load(data_dir / "train_data.npy", allow_pickle=False)
    if train_data.ndim != 2 or train_data.shape[1] != DEFAULT_NUM_FEATURES:
        raise ValueError(f"Unexpected train_data shape {train_data.shape}")
    required_end = split_manifest["expert_val"]["end"]
    if len(train_data) < required_end:
        raise ValueError(
            "ETTh2 train_data.npy does not contain expert_train + expert_val: "
            f"len={len(train_data)}, required_end={required_end}"
        )
    return train_data.astype(np.float32, copy=False), split_manifest


def build_clean_expert_dataloaders(
    data_dir: Path,
    batch_size: int,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, np.ndarray, Dict[str, dict]]:
    expert_prefix, split_manifest = load_etth2_expert_prefix(data_dir)
    train_loader = DataLoader(
        AbsoluteWindowDataset(expert_prefix, split_manifest, "expert_train"),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        AbsoluteWindowDataset(expert_prefix, split_manifest, "expert_val"),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, expert_prefix, split_manifest


def fit_clean_scaler(train_loader: DataLoader) -> Tuple[ZScoreScaler, dict]:
    dataset = train_loader.dataset
    if getattr(dataset, "split_role", None) != "expert_train":
        raise ValueError("Scaler must be fit only on expert_train")
    scaler = ZScoreScaler(norm_each_channel=True, rescale=False)
    scaler.fit(dataset.data)
    metadata = {
        "scaler_class": "ZScoreScaler",
        "configuration": {"norm_each_channel": True, "rescale": False},
        "fit_split": "expert_train",
        "source_index_range": [
            int(dataset.boundary["start"]),
            int(dataset.boundary["end"]),
        ],
        "mean": scaler.stats["mean"].detach().cpu().tolist(),
        "std": scaler.stats["std"].detach().cpu().tolist(),
    }
    metadata["sha256"] = canonical_json_sha256(metadata)
    return scaler, metadata


def call_forecasting_model(
    model: nn.Module,
    inputs: torch.Tensor,
    requires_timestamps: bool = False,
) -> torch.Tensor:
    if requires_timestamps:
        return _prediction_tensor(model(inputs, None))
    try:
        return _prediction_tensor(model(inputs))
    except TypeError:
        return _prediction_tensor(model(inputs, None))


def evaluate_expert_model(
    model: nn.Module,
    loader: Iterable[dict],
    spec: ExpertSpec,
    device: torch.device,
    scaler: ZScoreScaler,
) -> Tuple[float, float]:
    model.eval()
    absolute_error_sum = 0.0
    squared_error_sum = 0.0
    element_count = 0
    with torch.no_grad():
        for batch in loader:
            inputs, targets, target_mask = _prepare_forecasting_batch(batch, device, scaler)
            prediction = call_forecasting_model(model, inputs, spec.requires_timestamps)
            if tuple(prediction.shape) != tuple(targets.shape):
                raise ValueError(
                    f"{spec.display_name} prediction shape {tuple(prediction.shape)} "
                    f"does not match target shape {tuple(targets.shape)}"
                )
            abs_sum, squared_sum, count = _accumulate_errors(
                prediction,
                targets,
                target_mask,
            )
            absolute_error_sum += abs_sum
            squared_error_sum += squared_sum
            element_count += count
    if element_count == 0:
        raise ValueError("Validation loader produced no prediction elements")
    return absolute_error_sum / element_count, squared_error_sum / element_count


def train_one_expert(
    spec: ExpertSpec,
    train_loader: DataLoader,
    val_loader: DataLoader,
    scaler: ZScoreScaler,
    split_manifest: Dict[str, dict],
    checkpoint_path: Path,
    args: argparse.Namespace,
) -> Tuple[dict, ...]:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".training")
    if temp_path.exists():
        backup = temp_path.with_suffix(temp_path.suffix + ".stale")
        shutil.move(str(temp_path), str(_next_backup_path(backup)))

    config = spec.config_factory()
    model = spec.model_class(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    device = torch.device(args.device)
    model.to(device)
    history = []
    best_val_mae = float("inf")
    best_val_mse = float("inf")
    best_epoch = 0
    misses = 0

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        train_abs = 0.0
        train_sq = 0.0
        train_count = 0
        for batch_index, batch in enumerate(train_loader):
            inputs, targets, target_mask = _prepare_forecasting_batch(batch, device, scaler)
            optimizer.zero_grad(set_to_none=True)
            prediction = call_forecasting_model(model, inputs, spec.requires_timestamps)
            if tuple(prediction.shape) != tuple(targets.shape):
                raise ValueError(
                    f"{spec.display_name} prediction shape {tuple(prediction.shape)} "
                    f"does not match target shape {tuple(targets.shape)}"
                )
            if epoch == 1 and batch_index == 0:
                print(f"\n{spec.display_name} first expert-training batch")
                print(f"input shape:      {list(inputs.shape)}")
                print(f"target shape:     {list(targets.shape)}")
                print(f"prediction shape: {list(prediction.shape)}")
            loss = _configured_forecasting_loss(None, prediction, targets, target_mask)
            loss.backward()
            optimizer.step()
            abs_sum, squared_sum, count = _accumulate_errors(prediction, targets, target_mask)
            train_abs += abs_sum
            train_sq += squared_sum
            train_count += count

        train_mae = train_abs / train_count
        train_mse = train_sq / train_count
        val_mae, val_mse = evaluate_expert_model(model, val_loader, spec, device, scaler)
        saved = val_mae < best_val_mae
        if saved:
            best_val_mae = val_mae
            best_val_mse = val_mse
            best_epoch = epoch
            optimizer_state = optimizer.state_dict()
            torch.save(
                {
                    "completion_status": "training",
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer_state,
                    "optim_state_dict": optimizer_state,
                    "epoch": epoch,
                    "best_epoch": epoch,
                    "validation_mae": val_mae,
                    "validation_mse": val_mse,
                    "val_mae": val_mae,
                    "val_mse": val_mse,
                    "best_validation_mae": val_mae,
                    "best_metrics": {"val/MAE": val_mae, "val/MSE": val_mse},
                    "model_config": asdict(config),
                    "dataset": args.dataset,
                    "dataset_config": split_manifest,
                    "scaler_stats": scaler.stats,
                    "input_len": DEFAULT_INPUT_LEN,
                    "forecast_len": DEFAULT_OUTPUT_LEN,
                    "output_len": DEFAULT_OUTPUT_LEN,
                    "num_features": DEFAULT_NUM_FEATURES,
                    "expert_name": spec.display_name,
                    "model_key": spec.key,
                    "module_name": spec.module_name,
                    "model_class_name": spec.model_class_name,
                    "config_class_name": spec.config_class_name,
                    "requires_timestamps": spec.requires_timestamps,
                    "expert_order": list(EXPERT_ORDER),
                    "seed": args.seed,
                    "training": training_metadata(args),
                },
                temp_path,
            )
            misses = 0
        else:
            misses += 1

        row = {
            "expert": spec.display_name,
            "epoch": epoch,
            "train_mae": train_mae,
            "train_mse": train_mse,
            "val_mae": val_mae,
            "val_mse": val_mse,
            "checkpoint_saved": saved,
            "early_stopping_counter": misses,
        }
        history.append(row)
        print(
            f"{spec.display_name} epoch {epoch:>3d}/{args.max_epochs}: "
            f"training MAE={train_mae:.6f}, training MSE={train_mse:.6f}, "
            f"validation MAE={val_mae:.6f}, validation MSE={val_mse:.6f}, "
            f"early-stop counter={misses}/{args.patience}"
        )
        if misses >= args.patience:
            print(f"{spec.display_name}: early stopping after epoch {epoch}.")
            break

    if best_epoch == 0:
        raise RuntimeError(f"{spec.display_name} did not save a checkpoint")

    verification = verify_checkpoint(
        temp_path,
        spec,
        val_loader,
        scaler,
        torch.device(args.device),
        expected_config=asdict(spec.config_factory()),
        split_manifest=split_manifest,
    )
    checkpoint = torch.load(temp_path, map_location="cpu", weights_only=False)
    checkpoint["completion_status"] = "complete"
    checkpoint["training_history"] = tuple(history)
    checkpoint["verification"] = verification
    checkpoint["scaler_manifest"] = scaler_manifest_from_scaler(scaler, train_loader)
    if checkpoint_path.exists():
        backup = _next_backup_path(checkpoint_path.with_suffix(checkpoint_path.suffix + ".bak"))
        shutil.move(str(checkpoint_path), str(backup))
    torch.save(checkpoint, checkpoint_path)
    temp_path.unlink()
    return tuple(history)


def _next_backup_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(1, 1000):
        candidate = path.with_name(f"{path.name}.{index}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find free backup path for {path}")


def instantiate_from_checkpoint(spec: ExpertSpec, checkpoint: dict) -> nn.Module:
    config = spec.config_factory()
    expected = config_to_dict(config)
    saved = checkpoint.get("model_config")
    if saved != expected:
        raise ValueError(
            f"{spec.display_name} config mismatch: saved={saved}, expected={expected}"
        )
    try:
        return spec.model_class(config)
    except TypeError:
        return spec.model_class()


def verify_checkpoint(
    checkpoint_path: Path,
    spec: ExpertSpec,
    val_loader: DataLoader,
    scaler: ZScoreScaler,
    device: torch.device,
    expected_config: Optional[dict] = None,
    split_manifest: Optional[Dict[str, dict]] = None,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    required = {
        "model_state_dict",
        "model_config",
        "dataset",
        "validation_mae",
        "validation_mse",
        "dataset_config",
        "scaler_stats",
        "model_key",
        "input_len",
        "output_len",
        "num_features",
    }
    missing_fields = sorted(required.difference(checkpoint))
    if missing_fields:
        raise ValueError(f"{checkpoint_path} is missing fields: {missing_fields}")
    if checkpoint.get("model_key") != spec.key:
        raise ValueError(f"{checkpoint_path} is for {checkpoint.get('model_key')}, not {spec.key}")
    if checkpoint.get("dataset") != "ETTh2":
        raise ValueError(f"{checkpoint_path} is for dataset {checkpoint.get('dataset')}, not ETTh2")
    if checkpoint.get("input_len") != DEFAULT_INPUT_LEN or checkpoint.get("output_len") != DEFAULT_OUTPUT_LEN:
        raise ValueError(f"{checkpoint_path} has the wrong horizon")
    if expected_config is not None and checkpoint.get("model_config") != expected_config:
        raise ValueError(f"{checkpoint_path} has an incompatible model config")
    if split_manifest is not None and checkpoint.get("dataset_config") != split_manifest:
        raise ValueError(f"{checkpoint_path} has incompatible split metadata")

    model = instantiate_from_checkpoint(spec, checkpoint)
    result = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None

    batch = next(iter(val_loader))
    inputs, _, _ = _prepare_forecasting_batch(batch, device, scaler)
    with torch.no_grad():
        pred1 = call_forecasting_model(model, inputs, spec.requires_timestamps)
        pred2 = call_forecasting_model(model, inputs, spec.requires_timestamps)
    expected_shape = (inputs.shape[0], DEFAULT_OUTPUT_LEN, DEFAULT_NUM_FEATURES)
    if tuple(pred1.shape) != expected_shape:
        raise ValueError(f"{spec.display_name} output shape {tuple(pred1.shape)} != {expected_shape}")
    if not torch.isfinite(pred1).all():
        raise ValueError(f"{spec.display_name} produced NaN or infinity")
    deterministic = bool(torch.allclose(pred1, pred2, atol=1e-6, rtol=1e-6))
    if not deterministic:
        raise ValueError(f"{spec.display_name} predictions are not deterministic")

    grad_inputs = inputs.detach().clone().requires_grad_(True)
    prediction = call_forecasting_model(model, grad_inputs, spec.requires_timestamps)
    prediction.mean().backward()
    frozen_gradient_ok = all(parameter.grad is None for parameter in model.parameters())
    if not frozen_gradient_ok:
        raise ValueError(f"{spec.display_name} received gradients while frozen")

    return {
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
        "eval_mode": not model.training,
        "all_parameters_frozen": all(not parameter.requires_grad for parameter in model.parameters()),
        "output_shape": list(pred1.shape),
        "finite_output": True,
        "deterministic_inference": deterministic,
        "frozen_gradient_verification": frozen_gradient_ok,
    }


def training_metadata(args: argparse.Namespace) -> dict:
    return {
        "optimizer": "Adam",
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "loss": "masked_mae",
        "checkpoint_selection_metric": "expert_val_mae",
        "seed": args.seed,
        "device": args.device,
    }


def scaler_manifest_from_scaler(scaler: ZScoreScaler, train_loader: DataLoader) -> dict:
    dataset = train_loader.dataset
    metadata = {
        "scaler_class": "ZScoreScaler",
        "configuration": {"norm_each_channel": True, "rescale": False},
        "fit_split": "expert_train",
        "source_index_range": [
            int(dataset.boundary["start"]),
            int(dataset.boundary["end"]),
        ],
        "mean": scaler.stats["mean"].detach().cpu().tolist(),
        "std": scaler.stats["std"].detach().cpu().tolist(),
    }
    metadata["sha256"] = canonical_json_sha256(metadata)
    return metadata


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def git_status_summary() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--short"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        return {"error": repr(exc)}
    return {
        "head_commit": commit,
        "has_uncommitted_changes": bool(status.strip()),
        "status_short": status.splitlines(),
    }


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_reference_configs(reference_dir: Path) -> Dict[str, dict]:
    configs = {}
    for key, spec in EXPERT_SPECS.items():
        path = reference_dir / spec.checkpoint_name
        if not path.exists():
            continue
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        configs[spec.display_name] = checkpoint.get("model_config")
    return configs


def config_comparison_rows(reference_dir: Path) -> Tuple[Sequence[dict], Dict[str, bool]]:
    references = load_reference_configs(reference_dir)
    rows = []
    matches = {}
    for key, spec in EXPERT_SPECS.items():
        proposed = asdict(spec.config_factory())
        reference = references.get(spec.display_name)
        all_fields = sorted(set(proposed) | set(reference or {}))
        expert_match = True
        for field in all_fields:
            left = (reference or {}).get(field)
            right = proposed.get(field)
            equal = left == right
            expert_match = expert_match and equal
            rows.append(
                {
                    "expert": spec.display_name,
                    "field": field,
                    "etth1_reference": left,
                    "proposed_etth2": right,
                    "matches": equal,
                }
            )
        matches[spec.display_name] = expert_match
    return rows, matches


def verify_existing_checkpoint(
    checkpoint_path: Path,
    spec: ExpertSpec,
    val_loader: DataLoader,
    scaler: ZScoreScaler,
    split_manifest: Dict[str, dict],
    device: torch.device,
) -> Optional[dict]:
    if not checkpoint_path.exists():
        return None
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("completion_status") != "complete":
        return None
    verify_checkpoint(
        checkpoint_path,
        spec,
        val_loader,
        scaler,
        device,
        expected_config=asdict(spec.config_factory()),
        split_manifest=split_manifest,
    )
    return checkpoint


def run(args: argparse.Namespace) -> dict:
    if args.dataset != "ETTh2":
        raise ValueError("This clean trainer is intentionally ETTh2-only")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(args.deterministic_algorithms)

    data_dir = Path(args.data_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    results_dir = Path(args.results_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, expert_prefix, split_manifest = build_clean_expert_dataloaders(
        data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    scaler, scaler_manifest = fit_clean_scaler(train_loader)
    split_manifest = dict(split_manifest)
    split_manifest["sha256"] = canonical_json_sha256(split_manifest)

    comparison_rows, config_matches = config_comparison_rows(Path(args.reference_checkpoint_dir))
    print("Expert configuration comparison: ETTh1 reference vs proposed ETTh2")
    for row in comparison_rows:
        if not row["matches"]:
            print(
                f"  MISMATCH {row['expert']} {row['field']}: "
                f"ETTh1={row['etth1_reference']!r} ETTh2={row['proposed_etth2']!r}"
            )
    if not all(config_matches.values()):
        raise RuntimeError(f"Proposed ETTh2 configs do not match ETTh1 reference: {config_matches}")

    write_csv(
        results_dir / "expert_configuration_comparison.csv",
        comparison_rows,
        ("expert", "field", "etth1_reference", "proposed_etth2", "matches"),
    )
    (results_dir / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2),
        encoding="utf-8",
    )
    (results_dir / "scaler_manifest.json").write_text(
        json.dumps(scaler_manifest, indent=2),
        encoding="utf-8",
    )

    selected_keys = tuple(EXPERT_SPECS)
    history_rows = []
    summary_rows = []
    integrity = {
        "dataset": args.dataset,
        "input_len": DEFAULT_INPUT_LEN,
        "output_len": DEFAULT_OUTPUT_LEN,
        "num_features": DEFAULT_NUM_FEATURES,
        "test_values_accessed": False,
        "router_or_gate_training_performed": False,
        "audit_followed": True,
        "split_manifest_sha256": split_manifest["sha256"],
        "scaler_manifest_sha256": scaler_manifest["sha256"],
        "git": git_status_summary(),
        "hashes": {
            "metric_mae_file": sha256_file(ROOT / "src/basicts/metrics/mae.py"),
            "metric_mse_file": sha256_file(ROOT / "src/basicts/metrics/mse.py"),
            "expert_inference_helper_file": sha256_file(Path(__file__).resolve()),
        },
        "experts": {},
    }

    for key in selected_keys:
        spec = EXPERT_SPECS[key]
        checkpoint_path = checkpoint_dir / spec.checkpoint_name
        existing = verify_existing_checkpoint(
            checkpoint_path,
            spec,
            val_loader,
            scaler,
            split_manifest,
            torch.device(args.device),
        )
        if existing is None:
            print(f"\nTraining clean ETTh2 expert: {spec.display_name}")
            history = train_one_expert(
                spec,
                train_loader,
                val_loader,
                scaler,
                split_manifest,
                checkpoint_path,
                args,
            )
            history_rows.extend(history)
        else:
            print(f"\nReusing verified clean checkpoint: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        verification = verify_checkpoint(
            checkpoint_path,
            spec,
            val_loader,
            scaler,
            torch.device(args.device),
            expected_config=asdict(spec.config_factory()),
            split_manifest=split_manifest,
        )
        ckpt_hash = sha256_file(checkpoint_path)
        config_hash = canonical_json_sha256(checkpoint["model_config"])
        model_for_count = spec.model_class(spec.config_factory())
        row = {
            "expert_name": spec.display_name,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_hash": ckpt_hash,
            "parameter_count": parameter_count(model_for_count),
            "configuration": checkpoint["model_config"],
            "training_epochs": len(checkpoint.get("training_history", ())),
            "best_epoch": checkpoint["best_epoch"],
            "expert_validation_mae": checkpoint["validation_mae"],
            "expert_validation_mse": checkpoint["validation_mse"],
            "scaler_hash": scaler_manifest["sha256"],
            "split_ranges": split_manifest,
            "seed": args.seed,
            "reload_verification_status": verification,
            "frozen_gradient_verification_status": verification["frozen_gradient_verification"],
            "model_config_sha256": config_hash,
        }
        summary_rows.append(row)
        integrity["experts"][spec.display_name] = row

    write_csv(
        results_dir / "expert_training_history.csv",
        history_rows,
        (
            "expert",
            "epoch",
            "train_mae",
            "train_mse",
            "val_mae",
            "val_mse",
            "checkpoint_saved",
            "early_stopping_counter",
        ),
    )
    summary = {
        "dataset": args.dataset,
        "input_len": DEFAULT_INPUT_LEN,
        "output_len": DEFAULT_OUTPUT_LEN,
        "num_features": DEFAULT_NUM_FEATURES,
        "training_metadata": training_metadata(args),
        "split_manifest": split_manifest,
        "scaler_manifest": scaler_manifest,
        "summary": summary_rows,
        "completion_criteria": {
            "all_five_valid_checkpoints_exist": len(summary_rows) == 5,
            "intended_configurations": all(config_matches.values()),
            "trained_only_on_expert_train": True,
            "selected_only_on_expert_val": True,
            "scaler_used_only_expert_train": True,
            "every_checkpoint_reloads": all(
                row["reload_verification_status"]["missing_keys"] == []
                and row["reload_verification_status"]["unexpected_keys"] == []
                for row in summary_rows
            ),
            "every_expert_frozen": all(
                row["frozen_gradient_verification_status"] for row in summary_rows
            ),
            "no_test_values_accessed": True,
            "no_router_or_gate_training": True,
        },
    }
    (results_dir / "expert_training_summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default),
        encoding="utf-8",
    )
    (results_dir / "training_integrity_report.json").write_text(
        json.dumps(integrity, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="ETTh2")
    parser.add_argument("--data-dir", default="datasets/ETTh2")
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints/costarts_fresh/ETTh2_96_12/clean_candidates",
    )
    parser.add_argument(
        "--results-dir",
        default="results/router_summary/costarts_fresh/ETTh2_96_12/clean_experts",
    )
    parser.add_argument("--reference-checkpoint-dir", default="checkpoints/candidates")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--deterministic-algorithms", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
