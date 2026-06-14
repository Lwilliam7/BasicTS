"""Create a hybrid prediction from two saved BasicTS test result files.

BasicTS writes test results as raw NumPy memmaps with a ``.npy`` suffix. This
script reads those files with the expected shape, combines two model outputs,
and writes a normal ``.npy`` file that can be loaded with ``np.load``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_FIRST_MODEL_DIR = Path("checkpoints/SimpleMLPForecaster/ETTh1_96_12")
DEFAULT_SECOND_MODEL_DIR = Path("checkpoints/iTransformerForForecasting/ETTh1_96_12")
DEFAULT_OUTPUT = Path("checkpoints/hybrid_mlp_first6_transformer_next6_ETTh1_96_12_prediction.npy")


def latest_run_dir(model_dir: Path) -> Path:
    """Return the newest run directory that contains test prediction results."""
    candidates = [
        path.parent.parent
        for path in model_dir.glob("*/test_results/prediction.npy")
        if path.is_file()
    ]
    if not candidates:
        raise FileNotFoundError(f"No test_results/prediction.npy found under {model_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def read_cfg(run_dir: Path) -> dict:
    cfg_path = run_dir / "cfg.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing config file: {cfg_path}")
    return json.loads(cfg_path.read_text())


def prediction_shape(cfg: dict) -> tuple[int, int, int]:
    model_config = cfg["model_config"]
    dataset_params = cfg.get("dataset_params", {})
    dataset_name = cfg["dataset_name"]
    input_len = int(model_config.get("input_len", dataset_params.get("input_len")))
    output_len = int(model_config.get("output_len", dataset_params.get("output_len")))
    num_features = int(model_config["num_features"])

    test_data_path = Path("datasets") / dataset_name / "test_data.npy"
    if not test_data_path.is_file():
        raise FileNotFoundError(f"Missing test data file: {test_data_path}")

    test_data = np.load(test_data_path, mmap_mode="r")
    num_samples = len(test_data) - input_len - output_len + 1
    if num_samples <= 0:
        raise ValueError(
            f"Invalid sample count {num_samples} for {dataset_name}: "
            f"len(test_data)={len(test_data)}, input_len={input_len}, output_len={output_len}"
        )
    return num_samples, output_len, num_features


def load_prediction(run_dir: Path) -> tuple[Path, np.ndarray]:
    cfg = read_cfg(run_dir)
    shape = prediction_shape(cfg)
    prediction_path = run_dir / "test_results" / "prediction.npy"
    if not prediction_path.is_file():
        raise FileNotFoundError(f"Missing prediction file: {prediction_path}")

    expected_bytes = int(np.prod(shape)) * np.dtype("float32").itemsize
    if prediction_path.stat().st_size != expected_bytes:
        raise ValueError(
            f"Unexpected file size for {prediction_path}. "
            f"Expected {expected_bytes} bytes for shape {shape} and dtype float32, "
            f"got {prediction_path.stat().st_size} bytes."
        )

    prediction = np.memmap(prediction_path, dtype=np.float32, mode="r", shape=shape)
    return prediction_path, np.asarray(prediction)


def combine_predictions(
    first_prediction: np.ndarray,
    second_prediction: np.ndarray,
    strategy: str,
    split_step: int,
    first_weight: float,
) -> np.ndarray:
    if first_prediction.shape != second_prediction.shape:
        raise ValueError(
            f"Prediction shapes do not match: {first_prediction.shape} vs {second_prediction.shape}"
        )

    if strategy == "split":
        output_len = first_prediction.shape[1]
        if not 0 <= split_step <= output_len:
            raise ValueError(f"--split-step must be between 0 and {output_len}, got {split_step}")
        return np.concatenate(
            [first_prediction[:, :split_step, :], second_prediction[:, split_step:, :]],
            axis=1,
        )

    if strategy == "average":
        return (first_prediction + second_prediction) / 2.0

    if strategy == "weighted":
        if not 0.0 <= first_weight <= 1.0:
            raise ValueError(f"--first-weight must be between 0 and 1, got {first_weight}")
        return first_weight * first_prediction + (1.0 - first_weight) * second_prediction

    raise ValueError(f"Unknown strategy: {strategy}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-model-dir", type=Path, default=DEFAULT_FIRST_MODEL_DIR)
    parser.add_argument("--second-model-dir", type=Path, default=DEFAULT_SECOND_MODEL_DIR)
    parser.add_argument("--strategy", choices=["split", "average", "weighted"], default="split")
    parser.add_argument("--split-step", type=int, default=6)
    parser.add_argument("--first-weight", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    first_run_dir = latest_run_dir(args.first_model_dir)
    second_run_dir = latest_run_dir(args.second_model_dir)
    first_path, first_prediction = load_prediction(first_run_dir)
    second_path, second_prediction = load_prediction(second_run_dir)

    hybrid_prediction = combine_predictions(
        first_prediction=first_prediction,
        second_prediction=second_prediction,
        strategy=args.strategy,
        split_step=args.split_step,
        first_weight=args.first_weight,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, hybrid_prediction)

    print(f"First prediction:  {first_path}")
    print(f"Second prediction: {second_path}")
    print(f"Strategy:          {args.strategy}")
    if args.strategy == "split":
        print(f"Split step:        {args.split_step}")
    if args.strategy == "weighted":
        print(f"First weight:      {args.first_weight}")
    print(f"Hybrid shape:      {hybrid_prediction.shape}")
    print(f"Saved to:          {args.output}")


if __name__ == "__main__":
    main()
