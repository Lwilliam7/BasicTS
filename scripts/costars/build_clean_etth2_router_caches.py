"""Build clean ETTh2 router caches from frozen experts without test access."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

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
    _prepare_forecasting_batch,
)
from scripts.costars.train_candidate_experts import EXPERT_SPECS
from scripts.costars.train_clean_etth2_experts import (
    AbsoluteWindowDataset,
    call_forecasting_model,
    canonical_json_sha256,
    config_to_dict,
    sha256_file,
    split_manifest_for_total_length,
    verify_checkpoint,
)


EXPERT_KEYS = ("dlinear", "patchtst", "itransformer", "timesnet", "moderntcn")
EXPERT_NAMES = tuple(EXPERT_SPECS[key].display_name for key in EXPERT_KEYS)
PERMITTED_CACHE_SPLITS = ("router_train", "router_val")


def _json_default(value: object):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON encode {type(value)!r}")


def load_etth2_pretest_prefix(data_dir: Path) -> Tuple[np.ndarray, Dict[str, dict]]:
    """Load only ETTh2 train/val arrays, never test arrays."""

    data_dir = Path(data_dir)
    metadata = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    total_length = int(metadata["num_time_steps"])
    num_features = int(metadata["num_vars"])
    if num_features != DEFAULT_NUM_FEATURES:
        raise ValueError(f"Expected {DEFAULT_NUM_FEATURES} variables, found {num_features}")
    split_manifest = split_manifest_for_total_length(total_length)
    train_data = np.load(data_dir / "train_data.npy", allow_pickle=False)
    val_data = np.load(data_dir / "val_data.npy", allow_pickle=False)
    pretest_prefix = np.concatenate((train_data, val_data), axis=0).astype(
        np.float32,
        copy=False,
    )
    required_end = split_manifest["router_val"]["end"]
    if len(pretest_prefix) < required_end:
        raise ValueError(
            "ETTh2 train+val prefix does not contain router_train and router_val: "
            f"len={len(pretest_prefix)}, required_end={required_end}"
        )
    return pretest_prefix, split_manifest


def build_router_loader(
    pretest_prefix: np.ndarray,
    split_manifest: Dict[str, dict],
    split_role: str,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    if split_role not in PERMITTED_CACHE_SPLITS:
        raise ValueError(f"Only {PERMITTED_CACHE_SPLITS} cache splits are permitted")
    return DataLoader(
        AbsoluteWindowDataset(pretest_prefix, split_manifest, split_role),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )


def scaler_from_manifest(path: Path) -> Tuple[ZScoreScaler, dict]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("scaler_class") != "ZScoreScaler":
        raise ValueError("Clean ETTh2 caches require the ZScoreScaler manifest")
    if manifest.get("fit_split") != "expert_train":
        raise ValueError("Scaler was not fit on expert_train")
    stats = {
        "mean": torch.tensor(manifest["mean"], dtype=torch.float32),
        "std": torch.tensor(manifest["std"], dtype=torch.float32),
    }
    calculated = dict(manifest)
    saved_hash = calculated.pop("sha256", None)
    if canonical_json_sha256(calculated) != saved_hash:
        raise ValueError("Scaler manifest hash mismatch")
    return ZScoreScaler(norm_each_channel=True, rescale=False, stats=stats), manifest


def load_clean_manifest(path: Path) -> dict:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    criteria = manifest.get("completion_criteria", {})
    required_true = (
        "all_five_valid_checkpoints_exist",
        "intended_configurations",
        "trained_only_on_expert_train",
        "selected_only_on_expert_val",
        "scaler_used_only_expert_train",
        "every_checkpoint_reloads",
        "every_expert_frozen",
        "no_test_values_accessed",
        "no_router_or_gate_training",
    )
    failures = [key for key in required_true if criteria.get(key) is not True]
    if failures:
        raise ValueError(f"Clean expert manifest failed criteria: {failures}")
    names = tuple(row["expert_name"] for row in manifest["summary"])
    if names != EXPERT_NAMES:
        raise ValueError(f"Expert ordering mismatch: {names} != {EXPERT_NAMES}")
    return manifest


def _row_by_name(manifest: dict) -> Dict[str, dict]:
    return {row["expert_name"]: row for row in manifest["summary"]}


def instantiate_verified_experts(
    clean_manifest: dict,
    scaler: ZScoreScaler,
    split_manifest: Dict[str, dict],
    expert_val_loader: DataLoader,
    device: torch.device,
) -> Tuple[Tuple[nn.Module, ...], Dict[str, str], Dict[str, str]]:
    rows = _row_by_name(clean_manifest)
    experts = []
    checkpoint_hashes = {}
    config_hashes = {}
    for key in EXPERT_KEYS:
        spec = EXPERT_SPECS[key]
        row = rows[spec.display_name]
        checkpoint_path = ROOT / row["checkpoint_path"]
        observed_hash = sha256_file(checkpoint_path)
        if observed_hash != row["checkpoint_hash"]:
            raise ValueError(f"{spec.display_name} checkpoint hash mismatch")
        expected_config = config_to_dict(spec.config_factory())
        if canonical_json_sha256(expected_config) != row["model_config_sha256"]:
            raise ValueError(f"{spec.display_name} model config hash mismatch")
        verification = verify_checkpoint(
            checkpoint_path,
            spec,
            expert_val_loader,
            scaler,
            device,
            expected_config=expected_config,
            split_manifest=split_manifest,
        )
        if verification["missing_keys"] or verification["unexpected_keys"]:
            raise ValueError(f"{spec.display_name} did not load strictly")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = spec.model_class(spec.config_factory())
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.to(device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        if model.training or any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError(f"{spec.display_name} is not frozen in eval mode")
        experts.append(model)
        checkpoint_hashes[spec.display_name] = observed_hash
        config_hashes[spec.display_name] = row["model_config_sha256"]
    return tuple(experts), checkpoint_hashes, config_hashes


def _sample_errors(
    prediction_stack: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    errors = prediction_stack - targets.unsqueeze(-1)
    mask = target_mask.unsqueeze(-1).to(errors.dtype)
    denom = mask.sum(dim=(1, 2)).clamp_min(1.0)
    mae = (errors.abs() * mask).sum(dim=(1, 2)) / denom
    mse = (errors.square() * mask).sum(dim=(1, 2)) / denom
    return mae, mse


def build_cache(
    split_role: str,
    loader: DataLoader,
    experts: Sequence[nn.Module],
    scaler: ZScoreScaler,
    checkpoint_hashes: Dict[str, str],
    scaler_hash: str,
    device: torch.device,
    cache_path: Path,
    error_temperature: float = 0.1,
) -> dict:
    if split_role not in PERMITTED_CACHE_SPLITS:
        raise ValueError(f"Refusing to build unsupported split {split_role!r}")
    dataset = loader.dataset
    if dataset.split_role != split_role:
        raise ValueError(f"Loader split {dataset.split_role!r} != {split_role!r}")
    for model in experts:
        model.to(device)
        model.eval()
        for parameter in model.parameters():
            if parameter.requires_grad:
                raise ValueError("Expert parameter unexpectedly requires gradients")
            parameter.grad = None

    histories = []
    targets_list = []
    masks = []
    stacks = []
    maes = []
    mses = []
    sample_indices = []
    absolute_starts = []
    cursor = 0
    with torch.inference_mode():
        for batch in loader:
            inputs, targets, target_mask = _prepare_forecasting_batch(batch, device, scaler)
            predictions = []
            for spec, expert in zip((EXPERT_SPECS[key] for key in EXPERT_KEYS), experts):
                prediction = call_forecasting_model(expert, inputs, spec.requires_timestamps)
                if tuple(prediction.shape) != tuple(targets.shape):
                    raise ValueError(
                        f"{spec.display_name} prediction shape {tuple(prediction.shape)} "
                        f"does not match {tuple(targets.shape)}"
                    )
                predictions.append(prediction.detach())
            prediction_stack = torch.stack(predictions, dim=-1)
            mae, mse = _sample_errors(prediction_stack, targets, target_mask)
            batch_size = inputs.shape[0]
            histories.append(inputs.cpu())
            targets_list.append(targets.cpu())
            masks.append(target_mask.cpu())
            stacks.append(prediction_stack.cpu())
            maes.append(mae.cpu())
            mses.append(mse.cpu())
            sample_indices.append(torch.arange(cursor, cursor + batch_size, dtype=torch.long))
            absolute_starts.append(batch["absolute_window_start"].to(dtype=torch.long).cpu())
            cursor += batch_size

    error_matrix = torch.cat(maes, dim=0).to(torch.float32)
    cache = {
        "dataset": "ETTh2",
        "split_role": split_role,
        "expert_names": EXPERT_NAMES,
        "expert_order": EXPERT_NAMES,
        "checkpoint_hashes": dict(checkpoint_hashes),
        "scaler_hash": scaler_hash,
        "input_len": DEFAULT_INPUT_LEN,
        "forecast_horizon": DEFAULT_OUTPUT_LEN,
        "output_len": DEFAULT_OUTPUT_LEN,
        "num_features": DEFAULT_NUM_FEATURES,
        "num_windows": int(error_matrix.shape[0]),
        "source_index_range": [
            int(dataset.boundary["first_valid_window_start"]),
            int(dataset.boundary["last_valid_window_start"]),
        ],
        "split_boundary": dict(dataset.boundary),
        "histories": torch.cat(histories, dim=0).to(torch.float32),
        "targets": torch.cat(targets_list, dim=0).to(torch.float32),
        "target_masks": torch.cat(masks, dim=0).to(torch.bool),
        "prediction_stack": torch.cat(stacks, dim=0).to(torch.float32),
        "error_matrix": error_matrix,
        "mse_matrix": torch.cat(mses, dim=0).to(torch.float32),
        "best_expert": torch.argmin(error_matrix, dim=-1).to(torch.long),
        "sample_indices": torch.cat(sample_indices, dim=0).to(torch.long),
        "absolute_window_starts": torch.cat(absolute_starts, dim=0).to(torch.long),
        "error_temperature": float(error_temperature),
        "target_probabilities": torch.softmax(-error_matrix / error_temperature, dim=-1),
        "deployable_router_inputs": ("histories",),
        "offline_supervision_fields": (
            "targets",
            "target_masks",
            "error_matrix",
            "mse_matrix",
            "best_expert",
            "target_probabilities",
        ),
    }
    validate_cache(cache, split_role, checkpoint_hashes, scaler_hash)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    return cache


def _tensor_finite(tensor: torch.Tensor) -> bool:
    if tensor.dtype == torch.bool:
        return True
    return bool(torch.isfinite(tensor).all().item())


def validate_cache(
    cache: dict,
    split_role: str,
    checkpoint_hashes: Dict[str, str],
    scaler_hash: str,
) -> None:
    if cache.get("dataset") != "ETTh2":
        raise ValueError("Cache dataset is not ETTh2")
    if cache.get("split_role") != split_role or split_role not in PERMITTED_CACHE_SPLITS:
        raise ValueError("Cache split is not permitted")
    if tuple(cache.get("expert_names", ())) != EXPERT_NAMES:
        raise ValueError("Cache expert order mismatch")
    if cache.get("checkpoint_hashes") != dict(checkpoint_hashes):
        raise ValueError("Cache checkpoint hashes mismatch")
    if cache.get("scaler_hash") != scaler_hash:
        raise ValueError("Cache scaler hash mismatch")
    if cache.get("input_len") != DEFAULT_INPUT_LEN or cache.get("forecast_horizon") != DEFAULT_OUTPUT_LEN:
        raise ValueError("Cache horizon mismatch")
    if cache.get("num_features") != DEFAULT_NUM_FEATURES:
        raise ValueError("Cache feature count mismatch")
    n = int(cache["num_windows"])
    shapes = {
        "histories": (n, DEFAULT_INPUT_LEN, DEFAULT_NUM_FEATURES),
        "targets": (n, DEFAULT_OUTPUT_LEN, DEFAULT_NUM_FEATURES),
        "target_masks": (n, DEFAULT_OUTPUT_LEN, DEFAULT_NUM_FEATURES),
        "prediction_stack": (n, DEFAULT_OUTPUT_LEN, DEFAULT_NUM_FEATURES, len(EXPERT_NAMES)),
        "error_matrix": (n, len(EXPERT_NAMES)),
        "mse_matrix": (n, len(EXPERT_NAMES)),
        "best_expert": (n,),
        "sample_indices": (n,),
        "absolute_window_starts": (n,),
    }
    for key, shape in shapes.items():
        if tuple(cache[key].shape) != shape:
            raise ValueError(f"{key} shape {tuple(cache[key].shape)} != {shape}")
        if not _tensor_finite(cache[key]):
            raise ValueError(f"{key} contains NaN or infinity")
    expected_samples = torch.arange(n, dtype=cache["sample_indices"].dtype)
    if not torch.equal(cache["sample_indices"].cpu(), expected_samples):
        raise ValueError("sample_indices are not contiguous")
    boundary = cache["split_boundary"]
    starts = cache["absolute_window_starts"]
    if int(starts.min()) != boundary["first_valid_window_start"]:
        raise ValueError("First absolute start does not match split")
    if int(starts.max()) != boundary["last_valid_window_start"]:
        raise ValueError("Last absolute start does not match split")
    if int(starts.max()) + DEFAULT_INPUT_LEN + DEFAULT_OUTPUT_LEN > boundary["end"]:
        raise ValueError("A cached window crosses its split boundary")
    if int(starts.max()) >= 11520:
        raise ValueError("Locked-test source index appeared in cache")


def validate_disjoint(train_cache: dict, val_cache: dict) -> None:
    train_starts = set(train_cache["absolute_window_starts"].tolist())
    val_starts = set(val_cache["absolute_window_starts"].tolist())
    if train_starts.intersection(val_starts):
        raise ValueError("router_train and router_val source indices overlap")


def verify_direct_samples(
    cache: dict,
    loader: DataLoader,
    experts: Sequence[nn.Module],
    scaler: ZScoreScaler,
    device: torch.device,
) -> dict:
    indices = sorted(set([0, cache["num_windows"] // 2, cache["num_windows"] - 1]))
    results = []
    for index in indices:
        batch = loader.dataset[index]
        collated = {
            "inputs": torch.tensor(batch["inputs"]).unsqueeze(0),
            "targets": torch.tensor(batch["targets"]).unsqueeze(0),
        }
        inputs, targets, target_mask = _prepare_forecasting_batch(collated, device, scaler)
        with torch.inference_mode():
            predictions = [
                call_forecasting_model(expert, inputs, EXPERT_SPECS[key].requires_timestamps)
                for key, expert in zip(EXPERT_KEYS, experts)
            ]
            stack = torch.stack(predictions, dim=-1).cpu()
            mae, mse = _sample_errors(stack, targets.cpu(), target_mask.cpu())
        cached_stack = cache["prediction_stack"][index : index + 1]
        cached_mae = cache["error_matrix"][index : index + 1]
        cached_mse = cache["mse_matrix"][index : index + 1]
        if not torch.allclose(stack, cached_stack, atol=1e-6, rtol=1e-6):
            raise ValueError(f"Cached predictions mismatch for {cache['split_role']} index {index}")
        if not torch.allclose(mae, cached_mae, atol=1e-6, rtol=1e-6):
            raise ValueError(f"Cached MAE mismatch for {cache['split_role']} index {index}")
        if not torch.allclose(mse, cached_mse, atol=1e-6, rtol=1e-6):
            raise ValueError(f"Cached MSE mismatch for {cache['split_role']} index {index}")
        results.append(
            {
                "cache_index": index,
                "absolute_window_start": int(cache["absolute_window_starts"][index]),
                "prediction_match": True,
                "mae_match": True,
                "mse_match": True,
            }
        )
    return {"sampled_windows": results}


def git_snapshot(checkpoint_hashes: Dict[str, str], output_path: Path) -> dict:
    def _git(args: Sequence[str]) -> str:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )

    status = _git(["status", "--short", "--branch", "--untracked-files=all"])
    branch = _git(["branch", "--show-current"]).strip()
    commit = _git(["rev-parse", "HEAD"]).strip()
    modified = []
    untracked_relevant = []
    for line in status.splitlines():
        if not line or line.startswith("##"):
            continue
        path = line[3:] if len(line) > 3 else line
        if line[:2].strip() and not line.startswith("??"):
            modified.append(path)
        if line.startswith("??") and (
            "ETTh2_96_12" in path
            or path.startswith("scripts/costars/")
            or path.startswith("tests/test_clean_etth2")
        ):
            untracked_relevant.append(path)
    snapshot = {
        "current_commit_sha": commit,
        "branch": branch,
        "branch_status_line": status.splitlines()[0] if status.splitlines() else "",
        "local_master_may_be_behind_origin_master": "behind" in (status.splitlines()[0] if status.splitlines() else ""),
        "modified_files": modified,
        "untracked_files_relevant_to_etth2": untracked_relevant,
        "clean_expert_checkpoint_hashes": checkpoint_hashes,
        "pre_cache_git_status_file": "results/router_summary/costarts_fresh/ETTh2_96_12/pre_cache_git_status.txt",
        "pre_cache_git_diff_file": "results/router_summary/costarts_fresh/ETTh2_96_12/pre_cache_git_diff.patch",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    return snapshot


def cache_shapes(cache: dict) -> dict:
    return {
        key: list(cache[key].shape)
        for key in (
            "histories",
            "targets",
            "target_masks",
            "prediction_stack",
            "error_matrix",
            "mse_matrix",
            "best_expert",
            "sample_indices",
            "absolute_window_starts",
        )
    }


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    clean_manifest_path = Path(args.clean_expert_manifest)
    scaler_manifest_path = Path(args.scaler_manifest)
    clean_manifest = load_clean_manifest(clean_manifest_path)
    scaler, scaler_manifest = scaler_from_manifest(scaler_manifest_path)
    if clean_manifest["scaler_manifest"]["sha256"] != scaler_manifest["sha256"]:
        raise ValueError("Clean expert summary scaler hash differs from scaler manifest")

    pretest_prefix, split_manifest = load_etth2_pretest_prefix(Path(args.data_dir))
    split_manifest = dict(split_manifest)
    split_manifest["sha256"] = canonical_json_sha256(split_manifest)
    if split_manifest != clean_manifest["split_manifest"]:
        raise ValueError("Split manifest differs from clean expert manifest")

    expert_val_loader = DataLoader(
        AbsoluteWindowDataset(pretest_prefix, split_manifest, "expert_val"),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    experts, checkpoint_hashes, config_hashes = instantiate_verified_experts(
        clean_manifest,
        scaler,
        split_manifest,
        expert_val_loader,
        device,
    )
    git_snapshot(
        checkpoint_hashes,
        Path("results/router_summary/costarts_fresh/ETTh2_96_12/pre_cache_workspace_snapshot.json"),
    )

    loaders = {
        split: build_router_loader(
            pretest_prefix,
            split_manifest,
            split,
            args.batch_size,
            args.num_workers,
        )
        for split in PERMITTED_CACHE_SPLITS
    }
    output_dir = Path(args.cache_dir)
    caches = {
        "router_train": build_cache(
            "router_train",
            loaders["router_train"],
            experts,
            scaler,
            checkpoint_hashes,
            scaler_manifest["sha256"],
            device,
            output_dir / "router_train_cache.pt",
        ),
        "router_val": build_cache(
            "router_val",
            loaders["router_val"],
            experts,
            scaler,
            checkpoint_hashes,
            scaler_manifest["sha256"],
            device,
            output_dir / "router_val_cache.pt",
        ),
    }
    validate_disjoint(caches["router_train"], caches["router_val"])
    direct_checks = {
        split: verify_direct_samples(caches[split], loaders[split], experts, scaler, device)
        for split in PERMITTED_CACHE_SPLITS
    }
    grad_ok = all(
        parameter.grad is None
        for expert in experts
        for parameter in expert.parameters()
    )
    if not grad_ok:
        raise ValueError("Frozen expert parameters received gradients")

    cache_hashes = {
        "router_train": sha256_file(output_dir / "router_train_cache.pt"),
        "router_val": sha256_file(output_dir / "router_val_cache.pt"),
    }
    report = {
        "dataset": "ETTh2",
        "input_len": DEFAULT_INPUT_LEN,
        "forecast_horizon": DEFAULT_OUTPUT_LEN,
        "num_features": DEFAULT_NUM_FEATURES,
        "splits_built": list(PERMITTED_CACHE_SPLITS),
        "chronological_boundaries": split_manifest,
        "cache_shapes": {split: cache_shapes(cache) for split, cache in caches.items()},
        "num_windows": {split: int(cache["num_windows"]) for split, cache in caches.items()},
        "source_index_ranges": {
            split: cache["source_index_range"] for split, cache in caches.items()
        },
        "expert_order": EXPERT_NAMES,
        "checkpoint_hashes": checkpoint_hashes,
        "model_config_hashes": config_hashes,
        "scaler_hash": scaler_manifest["sha256"],
        "cache_hashes": cache_hashes,
        "hashes": {
            "clean_expert_manifest": sha256_file(clean_manifest_path),
            "scaler_manifest": sha256_file(scaler_manifest_path),
            "metric_mae_file": sha256_file(ROOT / "src/basicts/metrics/mae.py"),
            "metric_mse_file": sha256_file(ROOT / "src/basicts/metrics/mse.py"),
            "cache_builder": sha256_file(Path(__file__).resolve()),
            "expert_inference_helper": sha256_file(ROOT / "scripts/costars/train_clean_etth2_experts.py"),
        },
        "direct_inference_checks": direct_checks,
        "leakage_assertions": {
            "test_arrays_loaded": False,
            "test_cache_created": False,
            "locked_test_indices_absent": True,
            "router_train_router_val_disjoint": True,
            "no_gradients_enabled_for_cache_generation": True,
            "frozen_expert_parameters_received_no_gradients": grad_ok,
            "no_expert_router_selector_or_gate_training": True,
        },
        "working_tree_status": subprocess.check_output(
            ["git", "status", "--short", "--branch", "--untracked-files=all"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cache_validation_report.json").write_text(
        json.dumps(report, indent=2, default=_json_default),
        encoding="utf-8",
    )
    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="datasets/ETTh2")
    parser.add_argument(
        "--clean-expert-manifest",
        default="results/router_summary/costarts_fresh/ETTh2_96_12/clean_experts/expert_training_summary.json",
    )
    parser.add_argument(
        "--scaler-manifest",
        default="results/router_summary/costarts_fresh/ETTh2_96_12/clean_experts/scaler_manifest.json",
    )
    parser.add_argument("--cache-dir", default="cache/costarts_fresh/ETTh2_96_12")
    parser.add_argument(
        "--summary-path",
        default="results/router_summary/costarts_fresh/ETTh2_96_12/cache_build_summary.json",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
