# Residual-Covariance Weighting COSTAR-TS

## Protocol

- Frozen baseline: `hybrid_chrono_hvema_lowrank1_decay0.95_temp0.1_alpha0.75`.
- Core experts: PatchTST, iTransformer, TimesNet.
- Hyperparameters selected on chronological router-train folds only.
- Validation was evaluated once after selection.
- Test cache was not loaded.

## Selection

- Selected config: `diagonal_variance_hv_decay0.99_ridge0.0001_sd1_sg0_bias0_alpha0.5_warm96`.
- Best fold config: `full_covariance_hv_decay0.99_ridge0.0001_sd0.25_sg0_bias0_alpha0.75_warm96`.
- One-SE threshold: `0.342067`.

## Validation

- Best validation method: `baseline_fixed3_hv`.
- MAE / MSE: `0.363642` / `0.306712`.
- Improvement vs `0.363642`: `0.000000` (0.000%).
- Aggregate paired CI: `[0.000000, 0.000000]`.
- Strong target `<= 0.3619`: `False`.

## Diagnostics

- Mean fallback group rate: `0.000000`.
- Mean condition: `1.412`.
- Mean absolute delta vs HV baseline: `0.058473`.

## Reproduce

```powershell
python experiments\residual_covariance_weighting_costar\run_residual_covariance_weighting.py --device cuda
```
