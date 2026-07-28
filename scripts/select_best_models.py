import argparse
import importlib
import json
import os
import pkgutil
import sys
import traceback
from math import sqrt
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

os.chdir(ROOT)

from basicts.configs import BasicTSForecastingConfig
from basicts.launcher import BasicTSLauncher
from basicts.runners.taskflow import BasicTSForecastingTaskFlow
from basicts.scaler import ZScoreScaler
from basicts import models as _models_pkg

DATASET_NAME = "ETTh1"
INPUT_LEN = 96
OUTPUT_LEN = 12
NUM_FEATURES = 7
BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 1e-3

SHARED_CONFIG = {
    "dataset_name": DATASET_NAME,
    "input_len": INPUT_LEN,
    "dataset_params": {
        "input_len": INPUT_LEN,
        "output_len": OUTPUT_LEN,
        "use_timestamps": False,
        "memmap": False,
    },
    "use_timestamps": False,
    "batch_size": BATCH_SIZE,
    "num_epochs": NUM_EPOCHS,
    "scaler": ZScoreScaler,
    "norm_each_channel": True,
    "rescale": False,
    "metrics": ["MAE", "MSE", "RMSE", "MAPE", "WAPE"],
    "optimizer_params": {"lr": LEARNING_RATE, "weight_decay": 5e-4},
    "gpus": None,
    "train_data_num_workers": 0,
    "val_data_num_workers": 0,
    "test_data_num_workers": 0,
    "save_results": True,
}

DEFAULT_MODEL_LIST = [
    "Autoformer", "Crossformer", "DLinear", "DUET", "FiLM", "FITS", "FreTS", "HI", "Informer",
    "iTransformer", "Koopa", "Leddam", "LightTS", "MTSMixer", "NLinear", "NonstationaryTransformer",
    "PatchTST", "SegRNN", "SOFTS", "SparseTSF", "StemGNN", "STID", "TiDE", "TimeKAN", "TimeMixer",
    "Timer", "TimesNet", "TimeXer"
]


def discover_models():
    candidates = []
    for _, name, _ in pkgutil.iter_modules(_models_pkg.__path__):
        try:
            mod = importlib.import_module(f"basicts.models.{name}")
        except Exception:
            continue
        has_config = any(attr.endswith("Config") for attr in dir(mod))
        has_forecast = any("Forecast" in attr for attr in dir(mod))
        has_modelname = hasattr(mod, name)
        if has_config and (has_forecast or has_modelname):
            candidates.append(name)
    return sorted(candidates)


def select_model_registry(explicit_models=None):
    discovered = discover_models()
    if explicit_models:
        return [name for name in explicit_models if name in discovered]
    return [name for name in DEFAULT_MODEL_LIST if name in discovered]


def benchmark_models(model_names, results_csv, results_json, failed_csv):
    results = []
    failed_models = []

    for model_name in model_names:
        try:
            mod = importlib.import_module(f"basicts.models.{model_name}")
        except Exception as e:
            failed_models.append({"model": model_name, "error": f"import error: {e}"})
            continue

        model_cls = None
        for attr in dir(mod):
            if "Forecast" in attr:
                model_cls = getattr(mod, attr)
                break
        if model_cls is None and hasattr(mod, model_name):
            model_cls = getattr(mod, model_name)

        config_cls = None
        if hasattr(mod, f"{model_name}Config"):
            config_cls = getattr(mod, f"{model_name}Config")
        else:
            for attr in dir(mod):
                if attr.endswith("Config"):
                    config_cls = getattr(mod, attr)
                    break

        if model_cls is None or config_cls is None:
            failed_models.append({"model": model_name, "error": "missing model class or config"})
            continue

        print(f"Benchmarking {model_name}...")

        try:
            model_config = config_cls(input_len=INPUT_LEN, output_len=OUTPUT_LEN, num_features=NUM_FEATURES)
        except TypeError:
            model_config = config_cls(input_len=INPUT_LEN, output_len=OUTPUT_LEN)

        ckpt_dir = Path(f"checkpoints/benchmark/{model_name}/{DATASET_NAME}_{INPUT_LEN}_{OUTPUT_LEN}")
        cfg = BasicTSForecastingConfig(
            model=model_cls,
            model_config=model_config,
            taskflow=BasicTSForecastingTaskFlow(),
            ckpt_save_dir=str(ckpt_dir),
            **SHARED_CONFIG,
        )

        try:
            BasicTSLauncher.launch_training(cfg)
        except Exception as e:
            failed_models.append({"model": model_name, "error": f"train error: {e}\n{traceback.format_exc()}"})
            continue

        try:
            BasicTSLauncher.launch_evaluation(cfg, None)
        except Exception as e:
            failed_models.append({"model": model_name, "error": f"eval error: {e}\n{traceback.format_exc()}"})

        metrics_files = list(Path(ckpt_dir).rglob("test_metrics.json"))
        if not metrics_files:
            failed_models.append({"model": model_name, "error": "no test_metrics.json found after eval"})
            continue

        metrics_file = max(metrics_files, key=lambda p: p.stat().st_mtime)
        try:
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
        except Exception as e:
            failed_models.append({"model": model_name, "error": f"failed reading metrics: {e}"})
            continue

        overall = metrics.get("overall", {})
        mae = overall.get("MAE")
        mse = overall.get("MSE")
        rmse = overall.get("RMSE") if overall.get("RMSE") is not None else (sqrt(mse) if mse is not None else None)
        mape = overall.get("MAPE")
        wape = overall.get("WAPE")

        results.append({
            "model": model_name,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "MAPE": mape,
            "WAPE": wape,
            "metrics_file": str(metrics_file),
            "ckpt_dir": str(ckpt_dir),
        })

    if results:
        df = pd.DataFrame(results)
        df_sorted = df.sort_values(by=["MAE", "MSE"], na_position="last")
        df_sorted.to_csv(results_csv, index=False)
        df_sorted.to_json(results_json, orient="records", indent=2)
    if failed_models:
        pd.DataFrame(failed_models).to_csv(failed_csv, index=False)

    return results, failed_models


def load_saved_results(results_csv):
    if not results_csv.exists():
        return None
    return pd.read_csv(results_csv)


def print_rankings(df, top_n=10):
    if df is None or df.empty:
        print("No successful model results found.")
        return

    ranked = df.sort_values(by=["MAE", "MSE"], na_position="last")
    print(f"Top {min(top_n, len(ranked))} models by MAE/MSE:")
    print(ranked.head(top_n).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Rank BasicTS forecasting models by benchmark results")
    parser.add_argument("--fresh", action="store_true", help="Rerun the full benchmark instead of reading saved results")
    parser.add_argument("--top", type=int, default=10, help="Number of models to display")
    parser.add_argument("--models", nargs="+", help="Optional list of model names to benchmark")
    args = parser.parse_args()

    root = ROOT
    results_dir = root / "results"
    results_dir.mkdir(exist_ok=True)
    results_csv = results_dir / "model_benchmark_results.csv"
    results_json = results_dir / "model_benchmark_results.json"
    failed_csv = results_dir / "model_benchmark_failed.csv"

    model_names = select_model_registry(args.models)
    print(f"Using {len(model_names)} models: {model_names}")

    if args.fresh or not results_csv.exists():
        print("Running benchmark sweep...")
        results, failed_models = benchmark_models(model_names, results_csv, results_json, failed_csv)
        df = pd.DataFrame(results) if results else None
    else:
        print("Using saved benchmark results from results/model_benchmark_results.csv")
        df = load_saved_results(results_csv)

    print_rankings(df, top_n=args.top)

    if failed_csv.exists():
        failed_df = pd.read_csv(failed_csv)
        if not failed_df.empty:
            print("\nFailed models:")
            print(failed_df.to_string(index=False))


if __name__ == "__main__":
    main()
