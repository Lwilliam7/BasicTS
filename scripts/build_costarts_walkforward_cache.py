"""Build and validate walk-forward COSTAR-TS router caches.

This script owns the chronology/provenance layer for the full Sequential
COSTAR-TS protocol:

    A: 0-20%, B: 20-40%, C: 40-60%, validation: 60-80%, test: 80-100%.

It intentionally does not train forecasting experts. It consumes frozen expert
prediction arrays/checkpoints after they have been produced by the expert
training jobs and turns them into the explicit COSTAR router caches.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
EXPERT_ORDER = ("DLinear", "PatchTST", "iTransformer", "TimesNet", "ModernTCN")
DEFAULT_CACHE_DIR = Path("cache/costarts_walkforward")


@dataclass(frozen=True)
class RangeSpec:
    name: str
    start_fraction: float
    end_fraction: float
    start: int
    end: int

    @property
    def num_timestamps(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class StageSpec:
    role: str
    expert_training_range: RangeSpec
    prediction_range: RangeSpec
    output_path: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def load_full_array(data_dir: Path) -> np.ndarray:
    parts = []
    for name in ("train_data.npy", "val_data.npy", "test_data.npy"):
        path = data_dir / name
        if not path.exists():
            raise FileNotFoundError(path)
        parts.append(np.load(path))
    full = np.concatenate(parts, axis=0)
    if full.ndim != 2:
        raise ValueError(f"Expected [time, features] data, got {full.shape}")
    return full


def chronological_ranges(num_timestamps: int) -> dict[str, RangeSpec]:
    cuts = [int(num_timestamps * fraction) for fraction in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)]
    names = ("block_a", "block_b", "block_c", "validation", "test")
    ranges = {}
    for index, name in enumerate(names):
        ranges[name] = RangeSpec(
            name=name,
            start_fraction=index * 0.2,
            end_fraction=(index + 1) * 0.2,
            start=cuts[index],
            end=cuts[index + 1],
        )
    return ranges


def stage_specs(cache_dir: Path, ranges: Mapping[str, RangeSpec]) -> dict[str, StageSpec]:
    train_ab = RangeSpec("block_a_plus_b", 0.0, 0.4, ranges["block_a"].start, ranges["block_b"].end)
    train_abc = RangeSpec("block_a_plus_b_plus_c", 0.0, 0.6, ranges["block_a"].start, ranges["block_c"].end)
    return {
        "block_b_oos": StageSpec(
            role="block_b_oos",
            expert_training_range=ranges["block_a"],
            prediction_range=ranges["block_b"],
            output_path=cache_dir / "block_b_oos_cache.pt",
        ),
        "block_c_oos": StageSpec(
            role="block_c_oos",
            expert_training_range=train_ab,
            prediction_range=ranges["block_c"],
            output_path=cache_dir / "block_c_oos_cache.pt",
        ),
        "router_val_60_80": StageSpec(
            role="router_val_60_80",
            expert_training_range=train_abc,
            prediction_range=ranges["validation"],
            output_path=cache_dir / "router_val_60_80_cache.pt",
        ),
        "test_80_100": StageSpec(
            role="test_80_100",
            expert_training_range=train_abc,
            prediction_range=ranges["test"],
            output_path=cache_dir / "test_80_100_cache.pt",
        ),
    }


def valid_window_starts(range_spec: RangeSpec, input_len: int, horizon: int) -> torch.Tensor:
    last_exclusive = range_spec.end - input_len - horizon + 1
    if last_exclusive <= range_spec.start:
        raise ValueError(
            f"{range_spec.name} is too short for input_len={input_len}, horizon={horizon}: "
            f"[{range_spec.start}, {range_spec.end})"
        )
    return torch.arange(range_spec.start, last_exclusive, dtype=torch.long)


def build_histories_targets(
    full_data: np.ndarray,
    starts: torch.Tensor,
    input_len: int,
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    histories = []
    targets = []
    for start in starts.tolist():
        histories.append(full_data[start : start + input_len])
        targets.append(full_data[start + input_len : start + input_len + horizon])
    histories_t = torch.tensor(np.stack(histories), dtype=torch.float32)
    targets_t = torch.tensor(np.stack(targets), dtype=torch.float32)
    mask = torch.isfinite(targets_t)
    if not torch.isfinite(histories_t).all():
        raise ValueError("Histories contain non-finite values")
    targets_t = torch.nan_to_num(targets_t)
    return histories_t, targets_t, mask


def read_prediction_array(path: Path, expected_shape: Sequence[int]) -> torch.Tensor:
    if path.suffix == ".npy":
        tensor = torch.tensor(np.load(path), dtype=torch.float32)
    elif path.suffix in {".pt", ".pth"}:
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        tensor = loaded if isinstance(loaded, torch.Tensor) else loaded.get("predictions")
        if tensor is None:
            raise ValueError(f"{path} did not contain a tensor or 'predictions'")
        tensor = tensor.to(torch.float32)
    else:
        raise ValueError(f"Unsupported prediction file extension: {path}")
    if tuple(tensor.shape) != tuple(expected_shape):
        raise ValueError(f"{path} shape {tuple(tensor.shape)} != expected {tuple(expected_shape)}")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{path} contains non-finite predictions")
    return tensor


def prediction_file_for(prediction_dir: Path, role: str, expert: str) -> Path:
    candidates = [
        prediction_dir / role / f"{expert}.npy",
        prediction_dir / role / f"{expert}.pt",
        prediction_dir / f"{role}_{expert}.npy",
        prediction_dir / f"{role}_{expert}.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing prediction file for role={role}, expert={expert} under {prediction_dir}")


def sample_errors(
    prediction_stack: torch.Tensor,
    targets: torch.Tensor,
    target_masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = target_masks.to(torch.float32).unsqueeze(-1)
    denom = mask.sum(dim=(1, 2)).clamp_min(1.0)
    diff = (prediction_stack - targets.unsqueeze(-1)) * mask
    mae = diff.abs().sum(dim=(1, 2)) / denom
    mse = diff.square().sum(dim=(1, 2)) / denom
    return mae, mse


def load_checkpoint_manifest(path: Path | None) -> tuple[dict[str, str], dict[str, str]]:
    if path is None:
        return {}, {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    paths = payload.get("expert_checkpoint_paths", payload.get("checkpoint_paths", payload))
    checkpoint_paths = {name: str(value) for name, value in paths.items() if name in EXPERT_ORDER}
    checkpoint_hashes = {}
    for name, item in checkpoint_paths.items():
        p = Path(item)
        if not p.is_absolute():
            p = ROOT / p
        checkpoint_hashes[name] = sha256_file(p) if p.exists() else "missing"
    return checkpoint_paths, checkpoint_hashes


def assert_stage_no_leakage(
    *,
    role: str,
    expert_training_range: RangeSpec,
    prediction_range: RangeSpec,
    starts: torch.Tensor,
    input_len: int,
    horizon: int,
    num_timestamps: int,
    allow_test: bool,
) -> None:
    if prediction_range.start < expert_training_range.end:
        raise AssertionError(
            f"{role}: prediction_start {prediction_range.start} < expert_training_end {expert_training_range.end}"
        )
    target_starts = starts + input_len
    target_ends = starts + input_len + horizon
    if int(starts.min().item()) < prediction_range.start:
        raise AssertionError(f"{role}: history start before prediction range")
    if int(target_ends.max().item()) > prediction_range.end:
        raise AssertionError(f"{role}: target horizon crosses prediction range end")
    if int(target_starts.min().item()) < expert_training_range.end:
        raise AssertionError(f"{role}: target horizon overlaps expert training data")
    validation_start = int(num_timestamps * 0.6)
    test_start = int(num_timestamps * 0.8)
    if role in {"block_b_oos", "block_c_oos", "router_train_20_60"} and int(target_ends.max().item()) > validation_start:
        raise AssertionError(f"{role}: training cache touches validation/test region")
    if "test" in role and not allow_test:
        raise AssertionError(f"{role}: test cache creation requires --allow-test-cache")
    if "test" not in role and int(target_ends.max().item()) > test_start:
        raise AssertionError(f"{role}: non-test cache touches test region")


def validate_walkforward_cache(cache: Mapping[str, Any], allow_test: bool = False) -> None:
    required = (
        "split_role",
        "cache_role",
        "expert_names",
        "num_windows",
        "input_len",
        "forecast_horizon",
        "num_features",
        "histories",
        "targets",
        "target_masks",
        "prediction_stack",
        "error_matrix",
        "mse_matrix",
        "target_probabilities",
        "best_expert",
        "sample_indices",
        "absolute_window_starts",
        "provenance",
    )
    missing = [key for key in required if key not in cache]
    if missing:
        raise ValueError(f"Missing cache keys: {missing}")
    expert_names = tuple(cache["expert_names"])
    if expert_names != EXPERT_ORDER:
        raise ValueError(f"Expert order mismatch: {expert_names}")
    n = int(cache["num_windows"])
    input_len = int(cache["input_len"])
    horizon = int(cache["forecast_horizon"])
    features = int(cache["num_features"])
    experts = len(expert_names)
    shapes = {
        "histories": (n, input_len, features),
        "targets": (n, horizon, features),
        "target_masks": (n, horizon, features),
        "prediction_stack": (n, horizon, features, experts),
        "error_matrix": (n, experts),
        "mse_matrix": (n, experts),
        "target_probabilities": (n, experts),
        "best_expert": (n,),
        "sample_indices": (n,),
        "absolute_window_starts": (n,),
    }
    for key, shape in shapes.items():
        if tuple(cache[key].shape) != shape:
            raise ValueError(f"{key} shape {tuple(cache[key].shape)} != {shape}")
    for key in ("histories", "targets", "prediction_stack", "error_matrix", "mse_matrix", "target_probabilities"):
        if not torch.isfinite(cache[key]).all():
            raise ValueError(f"{key} contains non-finite values")
    if not torch.equal(cache["sample_indices"].cpu(), torch.arange(n, dtype=cache["sample_indices"].dtype)):
        raise AssertionError("sample_indices are not contiguous")
    starts = cache["absolute_window_starts"].to(torch.long)
    if not torch.equal(starts, torch.sort(starts).values):
        raise AssertionError("absolute_window_starts are not chronological")
    provenance = cache["provenance"]
    total = int(provenance["num_timestamps"])
    if str(cache["cache_role"]) == "router_train_20_60":
        validation_start = int(total * 0.6)
        if int((starts + input_len + horizon).max().item()) > validation_start:
            raise AssertionError("router_train_20_60 touches validation/test region")
        source_caches = provenance.get("source_caches", {})
        if not {"block_b_oos", "block_c_oos"}.issubset(source_caches):
            raise AssertionError("router_train_20_60 must record block_b_oos and block_c_oos source caches")
    else:
        train_range = RangeSpec(**provenance["expert_training_range"])
        pred_range = RangeSpec(**provenance["prediction_range"])
        assert_stage_no_leakage(
            role=str(cache["cache_role"]),
            expert_training_range=train_range,
            prediction_range=pred_range,
            starts=starts,
            input_len=input_len,
            horizon=horizon,
            num_timestamps=total,
            allow_test=allow_test,
        )
    mae, mse = sample_errors(cache["prediction_stack"], cache["targets"], cache["target_masks"])
    if not torch.allclose(mae, cache["error_matrix"], atol=1e-6, rtol=1e-6):
        raise AssertionError("Cached MAE does not reproduce")
    if not torch.allclose(mse, cache["mse_matrix"], atol=1e-6, rtol=1e-6):
        raise AssertionError("Cached MSE does not reproduce")


def build_stage_cache(
    *,
    dataset: str,
    data_dir: Path,
    prediction_dir: Path,
    stage: StageSpec,
    input_len: int,
    horizon: int,
    error_temperature: float,
    checkpoint_manifest: Path | None,
    allow_test: bool,
) -> dict[str, Any]:
    full_data = load_full_array(data_dir)
    starts = valid_window_starts(stage.prediction_range, input_len, horizon)
    assert_stage_no_leakage(
        role=stage.role,
        expert_training_range=stage.expert_training_range,
        prediction_range=stage.prediction_range,
        starts=starts,
        input_len=input_len,
        horizon=horizon,
        num_timestamps=full_data.shape[0],
        allow_test=allow_test,
    )
    histories, targets, masks = build_histories_targets(full_data, starts, input_len, horizon)
    expected_prediction_shape = (starts.numel(), horizon, full_data.shape[1])
    predictions = []
    prediction_files = {}
    for expert in EXPERT_ORDER:
        path = prediction_file_for(prediction_dir, stage.role, expert)
        prediction_files[expert] = str(path)
        predictions.append(read_prediction_array(path, expected_prediction_shape))
    prediction_stack = torch.stack(predictions, dim=-1).to(torch.float32)
    mae, mse = sample_errors(prediction_stack, targets, masks)
    target_probabilities = torch.softmax(-mae / float(error_temperature), dim=-1)
    checkpoint_paths, checkpoint_hashes = load_checkpoint_manifest(checkpoint_manifest)
    cache = {
        "split_role": stage.role,
        "cache_role": stage.role,
        "dataset": dataset,
        "expert_names": EXPERT_ORDER,
        "num_windows": int(starts.numel()),
        "input_len": int(input_len),
        "forecast_horizon": int(horizon),
        "num_features": int(full_data.shape[1]),
        "error_temperature": float(error_temperature),
        "histories": histories,
        "targets": targets,
        "target_masks": masks.to(torch.bool),
        "prediction_stack": prediction_stack,
        "error_matrix": mae.to(torch.float32),
        "mse_matrix": mse.to(torch.float32),
        "target_probabilities": target_probabilities.to(torch.float32),
        "best_expert": torch.argmin(mae, dim=-1).to(torch.long),
        "sample_indices": torch.arange(starts.numel(), dtype=torch.long),
        "absolute_window_starts": starts,
        "prediction_origin_index": starts.clone(),
        "provenance": {
            "dataset": dataset,
            "num_timestamps": int(full_data.shape[0]),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(),
            "cache_role": stage.role,
            "expert_order": list(EXPERT_ORDER),
            "expert_training_range": asdict(stage.expert_training_range),
            "prediction_range": asdict(stage.prediction_range),
            "prediction_files": prediction_files,
            "expert_checkpoint_paths": checkpoint_paths,
            "expert_checkpoint_hashes": checkpoint_hashes,
            "protocol": "walk-forward OOS: predict block before adding it to expert training",
        },
    }
    validate_walkforward_cache(cache, allow_test=allow_test)
    stage.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, stage.output_path)
    return cache


def combine_router_train(block_b_path: Path, block_c_path: Path, output_path: Path) -> dict[str, Any]:
    b_cache = torch.load(block_b_path, map_location="cpu", weights_only=False)
    c_cache = torch.load(block_c_path, map_location="cpu", weights_only=False)
    validate_walkforward_cache(b_cache)
    validate_walkforward_cache(c_cache)
    if b_cache["cache_role"] != "block_b_oos" or c_cache["cache_role"] != "block_c_oos":
        raise ValueError("Expected block_b_oos and block_c_oos caches")
    if tuple(b_cache["expert_names"]) != tuple(c_cache["expert_names"]):
        raise ValueError("Expert order mismatch between block caches")
    if int(b_cache["absolute_window_starts"].max().item()) >= int(c_cache["absolute_window_starts"].min().item()):
        raise AssertionError("Block B and C starts are not chronological")

    concat_keys = (
        "histories",
        "targets",
        "target_masks",
        "prediction_stack",
        "error_matrix",
        "mse_matrix",
        "target_probabilities",
        "best_expert",
        "absolute_window_starts",
        "prediction_origin_index",
    )
    cache = {
        key: torch.cat((b_cache[key], c_cache[key]), dim=0)
        for key in concat_keys
    }
    n = int(cache["histories"].shape[0])
    cache.update(
        {
            "split_role": "router_train_20_60",
            "cache_role": "router_train_20_60",
            "dataset": b_cache.get("dataset", "unknown"),
            "expert_names": tuple(b_cache["expert_names"]),
            "num_windows": n,
            "input_len": int(b_cache["input_len"]),
            "forecast_horizon": int(b_cache["forecast_horizon"]),
            "num_features": int(b_cache["num_features"]),
            "error_temperature": float(b_cache["error_temperature"]),
            "sample_indices": torch.arange(n, dtype=torch.long),
            "provenance": {
                "dataset": b_cache.get("dataset", "unknown"),
                "num_timestamps": int(b_cache["provenance"]["num_timestamps"]),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "git_commit": git_commit(),
                "cache_role": "router_train_20_60",
                "expert_order": list(EXPERT_ORDER),
                "expert_training_range": {
                    "note": "Mixed OOS sources: Block B experts trained on 0-20%; Block C experts trained on 0-40%.",
                    "block_b_expert_training_range": b_cache["provenance"]["expert_training_range"],
                    "block_c_expert_training_range": c_cache["provenance"]["expert_training_range"],
                },
                "prediction_range": asdict(
                    RangeSpec(
                        "block_b_plus_c",
                        0.2,
                        0.6,
                        int(b_cache["provenance"]["prediction_range"]["start"]),
                        int(c_cache["provenance"]["prediction_range"]["end"]),
                    )
                ),
                "source_caches": {
                    "block_b_oos": str(block_b_path),
                    "block_c_oos": str(block_c_path),
                    "block_b_sha256": sha256_file(block_b_path),
                    "block_c_sha256": sha256_file(block_c_path),
                },
                "protocol": "combined honest OOS predictions from Block B and Block C",
            },
        }
    )
    validate_walkforward_cache(cache)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output_path)
    return cache


def write_plan(dataset: str, data_dir: Path, cache_dir: Path, input_len: int, horizon: int, output: Path) -> None:
    full = load_full_array(data_dir)
    ranges = chronological_ranges(full.shape[0])
    stages = stage_specs(cache_dir, ranges)
    rows = []
    for range_spec in ranges.values():
        starts = valid_window_starts(range_spec, input_len, horizon)
        rows.append(
            {
                "name": range_spec.name,
                "fraction": f"{range_spec.start_fraction:.1f}-{range_spec.end_fraction:.1f}",
                "start": range_spec.start,
                "end": range_spec.end,
                "num_timestamps": range_spec.num_timestamps,
                "first_valid_window_start": int(starts[0].item()),
                "last_valid_window_start": int(starts[-1].item()),
                "num_windows": int(starts.numel()),
            }
        )
    payload = {
        "dataset": dataset,
        "data_dir": str(data_dir),
        "num_timestamps": int(full.shape[0]),
        "num_features": int(full.shape[1]),
        "input_len": input_len,
        "forecast_horizon": horizon,
        "ranges": rows,
        "stage_outputs": {name: str(spec.output_path) for name, spec in stages.items()},
        "router_train_output": str(cache_dir / "router_train_20_60_cache.pt"),
        "git_commit": git_commit(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="ETTh1")
    parser.add_argument("--data-dir", default="datasets/ETTh1")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--input-len", type=int, default=96)
    parser.add_argument("--forecast-horizon", type=int, default=12)
    parser.add_argument("--error-temperature", type=float, default=0.1)
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan")
    plan.add_argument("--output", default="results/router_summary/costarts_walkforward/split_plan.json")

    build = sub.add_parser("build-stage")
    build.add_argument("--role", choices=("block_b_oos", "block_c_oos", "router_val_60_80", "test_80_100"), required=True)
    build.add_argument("--prediction-dir", required=True)
    build.add_argument("--checkpoint-manifest")
    build.add_argument("--allow-test-cache", action="store_true")

    combine = sub.add_parser("combine-router-train")
    combine.add_argument("--block-b-cache", default=str(DEFAULT_CACHE_DIR / "block_b_oos_cache.pt"))
    combine.add_argument("--block-c-cache", default=str(DEFAULT_CACHE_DIR / "block_c_oos_cache.pt"))
    combine.add_argument("--output", default=str(DEFAULT_CACHE_DIR / "router_train_20_60_cache.pt"))

    validate = sub.add_parser("validate")
    validate.add_argument("cache_path")
    validate.add_argument("--allow-test-cache", action="store_true")

    args = parser.parse_args()
    data_dir = ROOT / args.data_dir
    cache_dir = ROOT / args.cache_dir
    ranges = chronological_ranges(load_full_array(data_dir).shape[0])
    stages = stage_specs(cache_dir, ranges)

    if args.command == "plan":
        write_plan(args.dataset, data_dir, cache_dir, args.input_len, args.forecast_horizon, ROOT / args.output)
        print(f"Wrote walk-forward split plan to {args.output}")
    elif args.command == "build-stage":
        cache = build_stage_cache(
            dataset=args.dataset,
            data_dir=data_dir,
            prediction_dir=ROOT / args.prediction_dir,
            stage=stages[args.role],
            input_len=args.input_len,
            horizon=args.forecast_horizon,
            error_temperature=args.error_temperature,
            checkpoint_manifest=ROOT / args.checkpoint_manifest if args.checkpoint_manifest else None,
            allow_test=args.allow_test_cache,
        )
        print(f"Wrote {args.role}: {stages[args.role].output_path} ({cache['num_windows']} windows)")
    elif args.command == "combine-router-train":
        cache = combine_router_train(ROOT / args.block_b_cache, ROOT / args.block_c_cache, ROOT / args.output)
        print(f"Wrote router_train_20_60: {args.output} ({cache['num_windows']} windows)")
    elif args.command == "validate":
        cache = torch.load(ROOT / args.cache_path, map_location="cpu", weights_only=False)
        validate_walkforward_cache(cache, allow_test=args.allow_test_cache)
        print(f"Validated {args.cache_path}: {cache['num_windows']} windows")


if __name__ == "__main__":
    main()
