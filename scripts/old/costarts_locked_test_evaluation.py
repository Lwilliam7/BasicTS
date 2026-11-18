"""Locked COSTAR-TS confidence-gate test cache and final evaluator.

This module is intentionally split into two explicit commands:

* ``build-cache`` creates a deterministic test cache from frozen expert
  prediction arrays and test targets.
* ``evaluate`` consumes that cache using only choices already frozen in the
  lock manifest.

The evaluator refuses to select checkpoints, fixed pairs, confidence score
types, or thresholds from test data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from scripts.old import train_costarts_pair_selector as pair_selector
    from scripts.old.evaluate_costarts_pair_selector_gate import confidence_scores_from_logits
except ImportError:
    import scripts.old.train_costarts_pair_selector as pair_selector
    from scripts.old.evaluate_costarts_pair_selector_gate import confidence_scores_from_logits


DEFAULT_LOCK_MANIFEST = "docs/costarts_pair_selector_gate_test_lock.json"
DEFAULT_TEST_CACHE = "cache/costarts_pair_selector_locked_test_cache.pt"
DEFAULT_OUTPUT = "results/router_summary/costarts_pair_selector_gate_locked/test_evaluation.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonable(value: Any) -> Any:
    return pair_selector._jsonable(value)


def assert_hash(path: Path, expected_sha256: str) -> None:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise AssertionError(f"Hash mismatch for {path}: expected {expected_sha256}, got {actual}")


def assert_manifest_hashes(manifest: Mapping[str, Any], repo_root: Path) -> None:
    for item in manifest["artifacts"]["expert_checkpoints"]:
        assert_hash(repo_root / item["path"], item["sha256"])
    for item in manifest["artifacts"]["pair_selector_checkpoints"]:
        assert_hash(repo_root / item["path"], item["sha256"])
    for item in manifest["artifacts"].get("router_caches", []):
        path = repo_root / item["path"]
        if path.exists():
            assert_hash(path, item["sha256"])


def split_ranges(manifest: Mapping[str, Any]) -> dict[str, range]:
    ranges = {}
    for name, meta in manifest["splits"].items():
        ranges[name] = range(int(meta["start"]), int(meta["end"]))
    return ranges


def assert_chronological_split_isolation(manifest: Mapping[str, Any]) -> None:
    ranges = split_ranges(manifest)
    ordered_names = ["expert_train", "expert_val", "router_train", "router_val", "test"]
    previous_end = None
    seen: set[int] = set()
    for name in ordered_names:
        current = ranges[name]
        if previous_end is not None and current.start < previous_end:
            raise AssertionError(f"Split {name} starts before previous split ended.")
        overlap = seen.intersection(current)
        if overlap:
            raise AssertionError(f"Split {name} overlaps earlier splits.")
        seen.update(current)
        previous_end = current.stop
    if ranges["router_train"].stop > ranges["router_val"].start:
        raise AssertionError("router_train overlaps router_val")
    if ranges["router_val"].stop > ranges["test"].start:
        raise AssertionError("router_val overlaps test")


def assert_scaler_fit_is_pre_router(manifest: Mapping[str, Any], data_dir: Path) -> None:
    train_data = np.load(data_dir / "train_data.npy")
    val_data = np.load(data_dir / "val_data.npy")
    test_data = np.load(data_dir / "test_data.npy")
    full_data = np.concatenate((train_data, val_data, test_data), axis=0)
    scaler = manifest["preprocessing"]["scaler"]
    fit_split = scaler["fit_split"]
    fit_range = manifest["splits"][fit_split]
    if int(fit_range["end"]) > int(manifest["splits"]["router_train"]["start"]):
        raise AssertionError("Scaler fit range reaches router-training or later data.")
    fit_data = full_data[int(fit_range["start"]) : int(fit_range["end"])]
    mean = fit_data.mean(axis=0)
    std = fit_data.std(axis=0)
    expected_mean = np.array(scaler["mean"], dtype=np.float32)
    expected_std = np.array(scaler["std"], dtype=np.float32)
    if not np.allclose(mean, expected_mean, atol=1e-5):
        raise AssertionError("Locked scaler mean does not match expert_train data.")
    if not np.allclose(std, expected_std, atol=1e-5):
        raise AssertionError("Locked scaler std does not match expert_train data.")


def assert_no_test_selection(manifest: Mapping[str, Any]) -> None:
    locked = manifest["locked_routing"]
    if locked.get("fixed_pair_selection_split") == "test":
        raise AssertionError("Fixed pair must not be selected from test data.")
    for seed_config in locked["seeds"]:
        if seed_config.get("checkpoint_selection_split") == "test":
            raise AssertionError("Checkpoint must not be selected from test data.")
        if seed_config.get("threshold_selection_split") == "test":
            raise AssertionError("Threshold must not be selected from test data.")
        if seed_config.get("confidence_score_selection_split") == "test":
            raise AssertionError("Confidence score type must not be selected from test data.")


def parse_expert_prediction_specs(specs: Sequence[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError("--expert-prediction entries must be ExpertName=path")
        name, path = spec.split("=", 1)
        parsed[name] = Path(path)
    return parsed


def _window_targets(test_data: np.ndarray, input_len: int, output_len: int) -> tuple[np.ndarray, np.ndarray]:
    histories = []
    targets = []
    for index in range(test_data.shape[0] - input_len - output_len + 1):
        histories.append(test_data[index : index + input_len])
        targets.append(test_data[index + input_len : index + input_len + output_len])
    return np.stack(histories).astype(np.float32), np.stack(targets).astype(np.float32)


def build_locked_test_cache(
    *,
    manifest_path: Path,
    data_dir: Path,
    expert_prediction_specs: Sequence[str],
    output_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    assert_chronological_split_isolation(manifest)
    assert_scaler_fit_is_pre_router(manifest, data_dir)
    assert_no_test_selection(manifest)
    assert_manifest_hashes(manifest, repo_root)

    input_len = int(manifest["metric_definition"]["input_len"])
    output_len = int(manifest["metric_definition"]["forecast_horizon"])
    num_features = int(manifest["metric_definition"]["num_features"])
    test_data = np.load(data_dir / "test_data.npy")
    histories_np, targets_np = _window_targets(test_data, input_len, output_len)
    if targets_np.shape[-1] != num_features:
        raise AssertionError("Test target feature count does not match manifest.")

    expert_names = tuple(manifest["locked_routing"]["expert_names"])
    prediction_paths = parse_expert_prediction_specs(expert_prediction_specs)
    predictions = []
    prediction_hashes = {}
    for expert_name in expert_names:
        if expert_name not in prediction_paths:
            raise AssertionError(f"Missing prediction array for expert {expert_name}")
        path = prediction_paths[expert_name]
        prediction_hashes[expert_name] = sha256_file(path)
        array = np.load(path)
        if tuple(array.shape) != tuple(targets_np.shape):
            raise AssertionError(
                f"Prediction shape for {expert_name} {array.shape} does not match targets {targets_np.shape}"
            )
        predictions.append(torch.tensor(array, dtype=torch.float32))

    prediction_stack = torch.stack(predictions, dim=-1)
    targets = torch.tensor(targets_np, dtype=torch.float32)
    histories = torch.tensor(histories_np, dtype=torch.float32)
    target_masks = torch.ones_like(targets, dtype=torch.bool)
    denominator = target_masks.to(torch.float32).sum(dim=(1, 2)).clamp_min(1.0).unsqueeze(-1)
    error_matrix = (
        torch.abs(prediction_stack - targets.unsqueeze(-1)) * target_masks.unsqueeze(-1)
    ).sum(dim=(1, 2)) / denominator
    mse_matrix = (
        (prediction_stack - targets.unsqueeze(-1)).pow(2) * target_masks.unsqueeze(-1)
    ).sum(dim=(1, 2)) / denominator
    num_windows = int(targets.shape[0])
    test_start = int(manifest["splits"]["test"]["start"])
    cache = {
        "cache_type": "costarts_locked_test_cache",
        "split_role": "test",
        "num_windows": num_windows,
        "input_len": input_len,
        "forecast_horizon": output_len,
        "num_features": num_features,
        "expert_names": expert_names,
        "histories": histories,
        "targets": targets,
        "target_masks": target_masks,
        "prediction_stack": prediction_stack,
        "error_matrix": error_matrix.to(torch.float32),
        "mse_matrix": mse_matrix.to(torch.float32),
        "best_expert": torch.argmin(error_matrix, dim=1).to(torch.long),
        "sample_indices": torch.arange(num_windows, dtype=torch.long),
        "absolute_start_indices": torch.arange(test_start, test_start + num_windows, dtype=torch.long),
        "lock_manifest_path": str(manifest_path),
        "lock_manifest_sha256": sha256_file(manifest_path),
        "expert_prediction_sha256": prediction_hashes,
        "expert_checkpoint_sha256": {
            item["expert_name"]: item["sha256"] for item in manifest["artifacts"]["expert_checkpoints"]
        },
        "pair_selector_checkpoint_sha256": {
            int(item["seed"]): item["sha256"] for item in manifest["artifacts"]["pair_selector_checkpoints"]
        },
        "scaler_metadata": manifest["preprocessing"]["scaler"],
        "split_metadata": manifest["splits"],
        "metric_definition": manifest["metric_definition"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output_path)
    return cache


def assert_test_cache_alignment(cache: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    if cache.get("split_role") != "test":
        raise AssertionError("Locked final cache must have split_role='test'.")
    if tuple(cache["expert_names"]) != tuple(manifest["locked_routing"]["expert_names"]):
        raise AssertionError("Cache expert order does not match lock manifest.")
    num_windows = int(cache["num_windows"])
    expected = torch.arange(num_windows, dtype=torch.long)
    if not torch.equal(cache["sample_indices"].cpu(), expected):
        raise AssertionError("Test cache sample_indices must be contiguous from zero.")
    test_start = int(manifest["splits"]["test"]["start"])
    expected_absolute = torch.arange(test_start, test_start + num_windows, dtype=torch.long)
    if not torch.equal(cache["absolute_start_indices"].cpu(), expected_absolute):
        raise AssertionError("Test cache absolute_start_indices do not align with locked test split.")
    if int(cache["absolute_start_indices"][0]) < int(manifest["splits"]["test"]["start"]):
        raise AssertionError("Test cache begins before test split.")


def _score_for_locked_name(logits: torch.Tensor, fixed_pair_class: int, score_name: str) -> torch.Tensor:
    scores = confidence_scores_from_logits(logits, fixed_pair_class)
    if score_name not in scores:
        raise AssertionError(f"Unknown locked score name: {score_name}")
    return scores[score_name]


@torch.no_grad()
def _locked_seed_prediction(
    *,
    manifest: Mapping[str, Any],
    cache: Mapping[str, Any],
    seed_config: Mapping[str, Any],
    repo_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint_path = repo_root / seed_config["pair_selector_checkpoint"]
    assert_hash(checkpoint_path, seed_config["pair_selector_checkpoint_sha256"])
    checkpoint = pair_selector._load_torch(checkpoint_path)
    model = pair_selector.CostartsPairSelector(**checkpoint["router_config"]).to(device)
    model.load_state_dict(checkpoint["router_state_dict"])
    model.eval()
    pair_index = checkpoint["pair_index"].to(torch.long)
    logits = []
    batch_size = int(manifest["metric_definition"].get("batch_size", 512))
    for offset in range(0, int(cache["num_windows"]), batch_size):
        history = cache["histories"][offset : offset + batch_size].to(device)
        logits.append(model(history).detach().cpu())
    logits_tensor = torch.cat(logits, dim=0)
    predicted_class = torch.argmax(logits_tensor, dim=1)
    fixed_pair_class = int(seed_config["fixed_pair_class"])
    score = _score_for_locked_name(logits_tensor, fixed_pair_class, seed_config["confidence_score"])
    switch = score >= float(seed_config["threshold"])
    predicted_pair = pair_index[predicted_class]
    fixed_pair = pair_index[fixed_pair_class].view(1, 2).expand(int(cache["num_windows"]), -1)
    selected_pair = torch.where(switch[:, None], predicted_pair, fixed_pair)
    prediction = pair_selector._selected_pair_prediction(cache, selected_pair)
    mae, mse = pair_selector._mae_mse(prediction, cache)
    return {
        "seed": int(seed_config["seed"]),
        "mae": mae,
        "mse": mse,
        "switch_rate": float(switch.to(torch.float32).mean()),
        "average_experts_queried": 2.0,
        "fixed_pair_class": fixed_pair_class,
        "fixed_pair_names": seed_config["fixed_pair_names"],
        "confidence_score": seed_config["confidence_score"],
        "threshold": float(seed_config["threshold"]),
    }


def evaluate_locked_test_cache(
    *,
    manifest_path: Path,
    test_cache_path: Path,
    output_path: Path,
    repo_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    assert_chronological_split_isolation(manifest)
    assert_no_test_selection(manifest)
    assert_manifest_hashes(manifest, repo_root)
    cache = pair_selector._load_torch(test_cache_path)
    assert_test_cache_alignment(cache, manifest)
    if cache.get("lock_manifest_sha256") != sha256_file(manifest_path):
        raise AssertionError("Test cache was not built from the supplied lock manifest.")
    runs = [
        _locked_seed_prediction(
            manifest=manifest,
            cache=cache,
            seed_config=seed_config,
            repo_root=repo_root,
            device=device,
        )
        for seed_config in manifest["locked_routing"]["seeds"]
    ]
    mae_values = torch.tensor([run["mae"] for run in runs], dtype=torch.float32)
    mse_values = torch.tensor([run["mse"] for run in runs], dtype=torch.float32)
    payload = {
        "lock_manifest": str(manifest_path),
        "test_cache": str(test_cache_path),
        "test_cache_sha256": sha256_file(test_cache_path),
        "runs": runs,
        "mean_std": {
            "mae": {"mean": float(mae_values.mean()), "std": float(mae_values.std(unbiased=False))},
            "mse": {"mean": float(mse_values.mean()), "std": float(mse_values.std(unbiased=False))},
            "average_experts_queried": {"mean": 2.0, "std": 0.0},
        },
        "locked_before_test": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Locked COSTAR-TS pair-gate final evaluation tools.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit-lock")
    audit.add_argument("--manifest", default=DEFAULT_LOCK_MANIFEST)
    audit.add_argument("--repo-root", default=".")
    audit.add_argument("--data-dir", default="datasets/ETTh1")

    build = subparsers.add_parser("build-cache")
    build.add_argument("--manifest", default=DEFAULT_LOCK_MANIFEST)
    build.add_argument("--repo-root", default=".")
    build.add_argument("--data-dir", default="datasets/ETTh1")
    build.add_argument("--output", default=DEFAULT_TEST_CACHE)
    build.add_argument("--expert-prediction", action="append", default=[])

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--manifest", default=DEFAULT_LOCK_MANIFEST)
    evaluate.add_argument("--repo-root", default=".")
    evaluate.add_argument("--test-cache", default=DEFAULT_TEST_CACHE)
    evaluate.add_argument("--output", default=DEFAULT_OUTPUT)
    evaluate.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root)
    manifest_path = Path(args.manifest)
    if args.command == "audit-lock":
        manifest = load_json(manifest_path)
        assert_chronological_split_isolation(manifest)
        assert_scaler_fit_is_pre_router(manifest, Path(args.data_dir))
        assert_no_test_selection(manifest)
        assert_manifest_hashes(manifest, repo_root)
        print("costarts_locked_test_evaluation audit-lock passed")
    elif args.command == "build-cache":
        cache = build_locked_test_cache(
            manifest_path=manifest_path,
            data_dir=Path(args.data_dir),
            expert_prediction_specs=args.expert_prediction,
            output_path=Path(args.output),
            repo_root=repo_root,
        )
        print(f"Saved locked test cache with {cache['num_windows']} windows: {args.output}")
    elif args.command == "evaluate":
        payload = evaluate_locked_test_cache(
            manifest_path=manifest_path,
            test_cache_path=Path(args.test_cache),
            output_path=Path(args.output),
            repo_root=repo_root,
            device=torch.device(args.device),
        )
        print(json.dumps(_jsonable(payload["mean_std"]), indent=2))


if __name__ == "__main__":
    main()
