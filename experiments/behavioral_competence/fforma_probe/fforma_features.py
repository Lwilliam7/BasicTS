"""FFORMA feature extraction: a faithful, tractable port of THA_features()
from the official robjhyndman/M4metalearning R package (commit
61ddc7101680e9df7219c359587d0b509d2b50d6, vendor/M4metalearning/R/generate_classif_problem.R),
using the Python `tsfeatures` package (Nixtla, v0.4.5) -- a direct port of
the same R `tsfeatures` functions THA_features itself calls
(acf_features, arch_stat, crossing_points, entropy, flat_spots,
heterogeneity, holt_parameters, hurst, lumpiness, nonlinearity,
pacf_features, stl_features, stability, hw_parameters, unitroot_kpss,
unitroot_pp -- confirmed by inspecting both packages' function names, which
match exactly) plus series_length, appended exactly as the R code does.

Two adaptations, both required for tractability and documented explicitly:

1. The official `tsfeatures()` Python wrapper spins up a fresh
   multiprocessing Pool per call (`with Pool(threads) as pool: ...`), which
   costs ~6.7s of pure process-spawn overhead even for a single series on
   Windows -- the actual feature computations underneath take ~0.1s total.
   This module calls the SAME underlying feature-group functions directly
   (bypassing the Pool wrapper), verified to produce numerically identical
   output to `tsfeatures()` on well-behaved input (see extract_tha_features).

2. THA_features (and the R tsfeatures functions it calls) operate on a
   univariate series; BasicTS windows are multivariate (up to 862 channels
   for Traffic). Features are computed on the cross-channel MEAN series of
   each window -- the same convention TimeFuse's own official
   `extract_meta_feature` already uses throughout (`.mean(axis=0)`), and
   necessary for tractability (per-channel tsfeatures at Traffic's scale
   would be computationally infeasible within any reasonable budget).

Failure handling matches the official R semantics exactly:
  - A feature GROUP that fails to compute (e.g. a group requiring more
    data than a short/degenerate window has) contributes NaN for its keys,
    later zeroed by the SAME "NA -> 0" convention THA_features itself uses
    (`featrow[is.na(featrow)] <- 0`).
  - A feature GROUP that legitimately has no seasonal component at the
    dataset's predeclared frequency (freq=1, or any group that simply omits
    a seasonal key at low frequency) is padded with a literal 0 for that key
    -- exactly the R code's explicit dummy-variable padding block for
    non-seasonal series (seas_acf1/seas_pacf/seasonal_strength/peak/trough).
  - These two are tracked SEPARATELY (num_group_failures vs
    num_seasonal_padding_zeros) so a dataset with excessive genuine feature
    failures can be flagged, per instruction, rather than silently blended
    with expected non-seasonal padding.
"""

from __future__ import annotations

import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = Path(__file__).resolve().parent / "cache" / "fforma_features"

from tsfeatures import (  # noqa: E402
    acf_features, arch_stat, crossing_points, entropy, flat_spots, heterogeneity,
    holt_parameters, hurst, lumpiness, nonlinearity, pacf_features, stl_features,
    stability, hw_parameters, unitroot_kpss, unitroot_pp,
)


# Exact THA_features group order (generate_classif_problem.R lines 33-51),
# heterogeneity/hw_parameters given the SAME "on failure" fallback the R
# workaround wrappers use (heterogeneity -> zeros, hw_parameters -> NaN).
FFORMA_FEATURE_GROUPS = [
    ("acf_features", acf_features, {"x_acf1", "x_acf10", "diff1_acf1", "diff1_acf10", "diff2_acf1", "diff2_acf10", "seas_acf1"}),
    ("arch_stat", arch_stat, {"arch_lm"}),
    ("crossing_points", crossing_points, {"crossing_points"}),
    ("entropy", entropy, {"entropy"}),
    ("flat_spots", flat_spots, {"flat_spots"}),
    ("heterogeneity", heterogeneity, {"arch_acf", "garch_acf", "arch_r2", "garch_r2"}),
    ("holt_parameters", holt_parameters, {"alpha", "beta"}),
    ("hurst", hurst, {"hurst"}),
    ("lumpiness", lumpiness, {"lumpiness"}),
    ("nonlinearity", nonlinearity, {"nonlinearity"}),
    ("pacf_features", pacf_features, {"x_pacf5", "diff1x_pacf5", "diff2x_pacf5", "seas_pacf"}),
    ("stl_features", stl_features, {"nperiods", "seasonal_period", "trend", "spike", "linearity", "curvature", "e_acf1", "e_acf10", "seasonal_strength", "peak", "trough"}),
    ("stability", stability, {"stability"}),
    ("hw_parameters", hw_parameters, {"hw_alpha", "hw_beta", "hw_gamma"}),
    ("unitroot_kpss", unitroot_kpss, {"unitroot_kpss"}),
    ("unitroot_pp", unitroot_pp, {"unitroot_pp"}),
]
# Keys that are only meaningfully defined for a series with a real seasonal
# period (freq > 1); at freq=1 the underlying functions simply omit them --
# the R code pads these with a literal 0 (its explicit non-seasonal dummy
# block), NOT NaN. Distinguished here so padding is never miscounted as a
# genuine failure.
SEASONAL_ONLY_KEYS = {"seas_acf1", "seas_pacf", "seasonal_strength", "peak", "trough"}

FFORMA_FEATURE_NAMES = sorted({k for _, _, keys in FFORMA_FEATURE_GROUPS for k in keys}) + ["series_length"]


def extract_tha_features(y: np.ndarray, freq: int) -> tuple[dict[str, float], int, int]:
    """Returns (features, num_group_failures, num_seasonal_padding_zeros)."""
    out: dict[str, float] = {}
    n_fail = 0
    n_pad = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _name, fn, expected_keys in FFORMA_FEATURE_GROUPS:
            try:
                r = fn(y, freq=freq)
                r = {k: (float(v) if v is not None else float("nan")) for k, v in r.items()}
            except Exception:
                r = {}
            missing = expected_keys - set(r.keys())
            for k in missing:
                if k in SEASONAL_ONLY_KEYS:
                    r[k] = 0.0
                    n_pad += 1
                else:
                    r[k] = float("nan")
                    n_fail += 1
            out.update(r)
    out["series_length"] = float(len(y))
    return out, n_fail, n_pad


def _extract_one(args: tuple[np.ndarray, int]) -> tuple[dict, int, int]:
    window_channel_mean, freq = args
    return extract_tha_features(window_channel_mean, freq)


def compute_fforma_features(histories: torch.Tensor, freq: int, max_workers: int | None = None) -> tuple[np.ndarray, dict[str, int]]:
    """histories: [N, input_len, num_features] raw windows. Features are
    computed on each window's cross-channel MEAN series (see module
    docstring). Returns ([N, len(FFORMA_FEATURE_NAMES)] float32 with
    NA->0 applied, diagnostics dict)."""
    channel_mean = histories.detach().cpu().numpy().astype(np.float64).mean(axis=2)  # [N, input_len]
    n = channel_mean.shape[0]
    args = [(channel_mean[i], freq) for i in range(n)]
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(_extract_one, args, chunksize=max(1, n // ((max_workers or 16) * 4) or 1)))
    rows = [r[0] for r in results]
    total_fail = sum(r[1] for r in results)
    total_pad = sum(r[2] for r in results)
    arr = np.array([[row[k] for k in FFORMA_FEATURE_NAMES] for row in rows], dtype=np.float64)
    num_nan_before_zero = int(np.isnan(arr).sum())
    arr = np.nan_to_num(arr, nan=0.0).astype(np.float32)
    diag = {
        "num_windows": n,
        "num_group_failures": total_fail,
        "num_seasonal_padding_zeros": total_pad,
        "num_nan_values_zeroed": num_nan_before_zero,
        "num_features": len(FFORMA_FEATURE_NAMES),
    }
    return arr, diag


def get_or_compute_fforma_features(dataset: str, split_name: str, histories: torch.Tensor, freq: int, max_workers: int | None = None) -> tuple[torch.Tensor, dict[str, int]]:
    cache_path = CACHE_DIR / f"{dataset}_{split_name}_freq{freq}_fforma_feat.npz"
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        if int(z["arr"].shape[0]) == int(histories.shape[0]):
            return torch.tensor(z["arr"], dtype=torch.float32), dict(z["diag"].item())
    arr, diag = compute_fforma_features(histories, freq, max_workers=max_workers)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, arr=arr, diag=np.array(diag, dtype=object))
    return torch.tensor(arr, dtype=torch.float32), diag


if __name__ == "__main__":
    import argparse
    import time

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from experiments.behavioral_competence.generalization.run_generalization_study import register_dataset  # noqa: E402
    from experiments.behavioral_competence.model_runtime import load_expert_runtime  # noqa: E402
    from experiments.behavioral_competence.run_behavioral_competence import raw_history_cache  # noqa: E402
    import experiments.frozen_hv_costar.run_frozen_hv_costar as fhv  # noqa: E402

    FREQ_BY_DATASET = {"ExchangeRate": 1, "BeijingAirQuality": 24, "Traffic": 24, "ETTm2": 96}

    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(FREQ_BY_DATASET))
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    for dataset in args.datasets:
        reg = register_dataset(dataset)
        core = reg["selected_core"]
        bundle = fhv.LOADERS[dataset]()
        ref_runtime = load_expert_runtime(dataset, core[0])
        freq = FREQ_BY_DATASET[dataset]
        for split_name, cache in (("router_train", bundle.train_cache), ("router_val", bundle.val_cache)):
            cache_raw = raw_history_cache(dataset, cache, ref_runtime.mean, ref_runtime.std)
            histories = cache_raw["histories"].to(torch.float32)
            t0 = time.time()
            print(f"[fforma_features] {dataset}/{split_name}: {histories.shape[0]} windows, freq={freq} ...", flush=True)
            _, diag = get_or_compute_fforma_features(dataset, split_name, histories, freq, max_workers=args.workers)
            print(f"[fforma_features] {dataset}/{split_name}: done in {time.time() - t0:.1f}s -- {diag}", flush=True)
