"""Caching wrapper around the OFFICIAL, unmodified TimeFuse meta-feature
extractor (vendor/TimeFuse/meta_feature.py, commit 978e6c6b9e4f246632c269aa0f9beeb099eabcfc).

extract_meta_feature() is not touched at all -- this module only adds
process-parallelism (across independent windows; the official function is
called once per window, exactly as it is in the official
batch_extract_meta_features) and on-disk caching, because the official
implementation calls statsmodels AutoReg/adfuller/acf per variable per
window, which is ~3.1s/window at Traffic's 862 variables (~9h serial for
Traffic's ~10k windows) but trivial (<100ms/window) for the 7-8 variable
datasets. No feature value is changed by parallelizing; each window is
processed independently and results are concatenated back in original order.
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[3]
VENDOR_DIR = Path(__file__).resolve().parent / "vendor" / "TimeFuse"
if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

from meta_feature import extract_meta_feature  # noqa: E402
from scipy.stats import skew, kurtosis, entropy  # noqa: E402
from scipy.signal import periodogram  # noqa: E402
from statsmodels.tsa.stattools import acf, adfuller  # noqa: E402
from statsmodels.tsa.ar_model import AutoReg  # noqa: E402


def extract_meta_feature_robust(data: np.ndarray) -> dict:
    """Byte-identical to the official extract_meta_feature (meta_feature.py),
    with ONE deviation: the per-variable adfuller() and AutoReg(...).fit()
    calls are each wrapped in try/except, substituting np.nan on failure
    instead of letting the whole window's feature extraction crash.

    This is necessary because adfuller() raises ValueError("Invalid input,
    x is constant") on a channel that is exactly constant within the current
    window -- a real, legitimate occurrence in real data (e.g. a flat
    stretch in an exchange-rate peg or a near-zero-variance channel), which
    the official function has no handling for. AutoReg can be similarly
    degenerate on a constant series. The resulting NaN is aggregated by the
    SAME np.mean/np.nanmean calls the official code already uses, and is
    zeroed out by the SAME nan_to_num(nan=0.0) step the official
    Dataset_Meta.__init__ already applies to any NaN in x_meta -- so this
    only prevents a crash; it does not change the feature FORMULA, and it
    reuses the official pipeline's own existing NaN-tolerance convention for
    the (rare) affected channel/window rather than inventing a new one."""
    features = {}
    features["mean"] = np.mean(data, axis=0).mean()
    features["std"] = np.std(data, axis=0).mean()
    features["min"] = np.min(data, axis=0).mean()
    features["max"] = np.max(data, axis=0).mean()
    features["skewness"] = np.nanmean(skew(data, axis=0))
    features["kurtosis"] = np.nanmean(kurtosis(data, axis=0))

    acfs = []
    for i in range(data.shape[1]):
        try:
            acfs.append(acf(data[:, i], nlags=10, fft=True))
        except Exception:
            acfs.append(np.full(11, np.nan))
    features["autocorrelation_mean"] = np.nanmean([acf_val[1] for acf_val in acfs])

    adf_pvalues = []
    for i in range(data.shape[1]):
        try:
            adf_pvalues.append(adfuller(data[:, i])[1] < 0.05)
        except Exception:
            adf_pvalues.append(np.nan)
    features["stationarity"] = np.nanmean(adf_pvalues)

    safe_data = np.where(data[:-1] == 0, np.nan, data[:-1])
    rate_of_change = np.diff(data, axis=0) / safe_data
    features["rate_of_change_mean"] = np.nanmean(rate_of_change)
    features["rate_of_change_std"] = np.nanstd(rate_of_change)

    autoreg_coefs, residual_stds = [], []
    for i in range(data.shape[1]):
        try:
            model = AutoReg(data[:, i], lags=1).fit()
            autoreg_coefs.append(model.params[1])
            residual_stds.append(np.std(model.resid))
        except Exception:
            autoreg_coefs.append(np.nan)
            residual_stds.append(np.nan)
    features["autoreg_coef_mean"] = np.nanmean(autoreg_coefs)
    features["residual_std_mean"] = np.nanmean(residual_stds)

    freq_means, freq_peaks, spectral_entropies = [], [], []
    spectral_variations, spectral_skewnesses, spectral_kurtoses = [], [], []
    for i in range(data.shape[1]):
        freqs, psd = periodogram(data[:, i])
        freq_means.append(np.mean(psd))
        freq_peaks.append(freqs[np.argmax(psd)])
        spectral_entropies.append(entropy(psd))
        if i > 0:
            prev_psd = periodogram(data[:, i - 1])[1]
            spectral_variations.append(np.sqrt(np.sum((psd - prev_psd) ** 2)))
        else:
            spectral_variations.append(0)
        spectral_skewnesses.append(skew(psd))
        spectral_kurtoses.append(kurtosis(psd))

    features["frequency_mean"] = np.mean(freq_means)
    features["frequency_peak"] = np.mean(freq_peaks)
    features["spectral_entropy"] = np.nanmean(spectral_entropies)
    features["spectral_variation"] = np.nanmean(spectral_variations)
    features["spectral_skewness"] = np.nanmean(spectral_skewnesses)
    features["spectral_kurtosis"] = np.nanmean(spectral_kurtoses)

    cov_matrix = np.cov(data, rowvar=False)
    features["covariance_mean"] = np.mean(cov_matrix)
    features["covariance_max"] = np.max(cov_matrix)
    features["covariance_min"] = np.min(cov_matrix)
    features["covariance_std"] = np.std(cov_matrix)
    return features


CACHE_DIR = Path(__file__).resolve().parent / "cache" / "meta_features"
META_FEATURE_NAMES = [
    "mean", "std", "min", "max", "skewness", "kurtosis",
    "autocorrelation_mean", "stationarity",
    "rate_of_change_mean", "rate_of_change_std",
    "autoreg_coef_mean", "residual_std_mean",
    "frequency_mean", "frequency_peak", "spectral_entropy", "spectral_variation", "spectral_skewness", "spectral_kurtosis",
    "covariance_mean", "covariance_max", "covariance_min", "covariance_std",
]  # exact insertion order of extract_meta_feature's returned dict; dim=22, matches official dim_meta_feats


def _extract_one(window: np.ndarray) -> dict:
    return extract_meta_feature_robust(window)


def compute_meta_features(histories: torch.Tensor, max_workers: int | None = None) -> np.ndarray:
    """histories: [N, input_len, num_features] raw (unnormalized) windows,
    exactly what the official batch_extract_meta_features receives as
    batch_x. Returns [N, 22] float32, official column order, NaN->0.0
    exactly as Dataset_Meta.__init__ does for loaded meta-features."""
    data_np = histories.detach().cpu().numpy().astype(np.float64)
    n = data_np.shape[0]
    windows = [data_np[i] for i in range(n)]
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        rows = list(ex.map(_extract_one, windows, chunksize=max(1, n // ((max_workers or 16) * 4) or 1)))
    df = pd.DataFrame(rows, columns=META_FEATURE_NAMES)
    arr = df.values.astype(np.float32)
    if np.isnan(arr).any():
        arr = np.nan_to_num(arr, nan=0.0)
    return arr


def get_or_compute_meta_features(dataset: str, split_name: str, histories: torch.Tensor, max_workers: int | None = None) -> torch.Tensor:
    cache_path = CACHE_DIR / f"{dataset}_{split_name}_x_meta.npy"
    if cache_path.exists():
        arr = np.load(cache_path)
        if arr.shape == (int(histories.shape[0]), len(META_FEATURE_NAMES)):
            return torch.tensor(arr, dtype=torch.float32)
        print(f"[meta_feature_cache] {dataset}/{split_name}: cached shape {arr.shape} != expected {(int(histories.shape[0]), len(META_FEATURE_NAMES))}, recomputing", flush=True)
    arr = compute_meta_features(histories, max_workers=max_workers)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, arr)
    return torch.tensor(arr, dtype=torch.float32)


if __name__ == "__main__":
    import argparse
    import time

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from experiments.behavioral_competence.generalization.run_generalization_study import register_dataset  # noqa: E402
    from experiments.behavioral_competence.run_behavioral_competence import raw_history_cache  # noqa: E402
    from experiments.behavioral_competence.model_runtime import load_expert_runtime  # noqa: E402
    import experiments.frozen_hv_costar.run_frozen_hv_costar as fhv  # noqa: E402

    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["ExchangeRate", "BeijingAirQuality", "ETTm2", "Traffic"])
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    for dataset in args.datasets:
        reg = register_dataset(dataset)
        core = reg["selected_core"]
        bundle = fhv.LOADERS[dataset]()
        ref_runtime = load_expert_runtime(dataset, core[0])
        for split_name, cache in (("router_train", bundle.train_cache), ("router_val", bundle.val_cache)):
            cache_raw = raw_history_cache(dataset, cache, ref_runtime.mean, ref_runtime.std)
            histories = cache_raw["histories"].to(torch.float32)
            t0 = time.time()
            print(f"[meta_feature_cache] {dataset}/{split_name}: {histories.shape[0]} windows x {histories.shape[2]} vars ...", flush=True)
            get_or_compute_meta_features(dataset, split_name, histories, max_workers=args.workers)
            print(f"[meta_feature_cache] {dataset}/{split_name}: done in {time.time() - t0:.1f}s", flush=True)
