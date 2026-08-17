# Frozen COSTAR Validation Comparison

This validation-only experiment isolates sequential validation-target feedback.
The frozen path repeats router-train initialized general and horizon-variable weights across all validation windows.

## Target-Feedback Trace

- `parameterized_current_base_prediction()` / ETTh2 `current_base_prediction()` read validation errors through `per_location_abs_error_for_indices(cache, ...)` or `per_location_error(cache, ...)`.
- `chronological_online_weights()` updates EMA state from validation expert MAE after `old_start + horizon <= current_start`.
- `chronological_hv_weights()` updates horizon-variable EMA state from validation per-location expert errors after the same causal delay.
- `run_causal_specialists()` and ETTh2 `run_specialists_no_duplicate()` update specialist states from validation base/DLinear/ModernTCN absolute errors.
- Static ETTh1 neural weights are produced from current history and forecasts only in the frozen path; target-based MAE/MSE reporting from the legacy loader is not used.

## Results

| Dataset | Equal fixed-three | Frozen COSTAR | Online COSTAR |
|---|---:|---:|---:|
| ETTh1 | `0.367265` / `0.310530` | `0.365868` / `0.308465` | `0.363111` / `0.306056` |
| ETTh2 | `0.280878` / `0.171933` | `0.277481` / `0.167632` | `0.276832` / `0.167280` |

## Configuration

- ETTh1 core: `PatchTST+iTransformer+TimesNet`.
- ETTh2 core: `DLinear+PatchTST+ModernTCN`.
- Base mixture: `0.25` chronological branch, `0.75` horizon-variable branch.
- Chronological branch: `0.5` static prior, `0.5` router-train EMA initialization.
- Horizon-variable branch: low-rank rank `1`, decay `0.95`, temperature `0.1`, frozen at router-train initialization.
- Specialist config: `both_variable_decay0.95_cap0.1_marginbp200_warm96`, frozen at router-train initialization.

## Leakage Checks

- ETTh1: all frozen target/mask mutation and cache immutability checks passed.
  Online target replacement changed predictions: `True`.
- ETTh2: all frozen target/mask mutation and cache immutability checks passed.
  Online target replacement changed predictions: `True`.

## Reproduce

```powershell
python experiments\frozen_costar\run_frozen_costar_validation.py --device cuda
```
