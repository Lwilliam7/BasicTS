# Frozen COSTAR Validation Diagnostic

Date: 2026-08-16

## Goal

Isolate validation-time sequential adaptation in the current full adaptive COSTAR model by creating a non-sequential `frozen_costar` path. The frozen path keeps the same core experts, frozen forecasts, static neural artifact or equal fallback, hyperparameters, and `0.25` chronological / `0.75` horizon-variable mixture, but repeats router-train initialized weights across all validation windows.

No test cache was loaded or evaluated.

## Target-Feedback Trace

Validation/test target feedback in the online path occurs in:

- `parameterized_current_base_prediction()` and ETTh2 `current_base_prediction()`: construct validation expert errors using `per_location_abs_error_for_indices(cache, ...)` / `per_location_error(cache, ...)`.
- `chronological_online_weights()`: updates EMA expert weights from realized validation expert MAE when `old_start + horizon <= current_start`.
- `chronological_hv_weights()`: updates horizon-variable EMA weights from realized validation expert errors under the same causal delay.
- `run_causal_specialists()` and ETTh2 `run_specialists_no_duplicate()`: update base/DLinear/ModernTCN specialist states from realized validation absolute errors.

The new frozen static ETTh1 weight loader uses the existing neural checkpoint but computes only weights from current history and forecasts; it does not compute the target-based MAE/MSE returned by the older helper.

## Results

| Dataset | Equal fixed-three | Frozen COSTAR | Online COSTAR |
|---|---:|---:|---:|
| ETTh1 | `0.367265` / `0.310530` | `0.365868` / `0.308465` | `0.363111` / `0.306056` |
| ETTh2 | `0.280878` / `0.171933` | `0.277481` / `0.167632` | `0.276832` / `0.167280` |

ETTh1 frozen 5-seed MAE mean/std: `0.365869 +/- 0.000012`.

ETTh2 frozen is deterministic under the equal static fallback.

## Leakage Checks

Passed:

- Replacing validation targets leaves frozen predictions exactly unchanged.
- Replacing validation masks leaves frozen predictions exactly unchanged.
- Validation cache prediction/history/start tensors are unchanged after frozen prediction.
- Online COSTAR target replacement changes predictions, confirming the check can detect target feedback.
- Frozen and online predictions begin from equivalent train-derived first-window initialization.

## Interpretation

The frozen path improves over equal fixed-three on validation, but online COSTAR is materially better on both ETTh1 and ETTh2. This supports the current interpretation that causal validation-time updates are contributing useful signal rather than being a removable implementation detail.

## Artifacts

- `experiments/frozen_costar/run_frozen_costar_validation.py`
- `experiments/frozen_costar/frozen_costar_validation_results.json`
- `experiments/frozen_costar/frozen_costar_validation_results.csv`
- `experiments/frozen_costar/frozen_costar_seed_results.csv`
- `experiments/frozen_costar/frozen_costar_config.json`
- `experiments/frozen_costar/frozen_costar_report.md`
- `tests/test_frozen_costar.py`

Reproduce:

```powershell
python experiments\frozen_costar\run_frozen_costar_validation.py --device cuda
```
