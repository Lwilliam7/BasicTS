# Residual-Covariance Weighting COSTAR-TS

## Hypothesis

Accounting for cross-expert residual covariance might improve causal weights over the fixed-three horizon-variable baseline.

## Protocol

- Dataset: ETTh1 router-train `20-60%`, validation `60-80%`.
- Experts: PatchTST, iTransformer, TimesNet.
- Frozen comparison baseline: `hybrid_chrono_hvema_lowrank1_decay0.95_temp0.1_alpha0.75`, MAE `0.363642`.
- Tested inverse-error, diagonal-variance, and full shrunk covariance weighting over global, horizon, variable, and horizon x variable structures.
- Decays: `0.95`, `0.97`, `0.98`, `0.99`.
- Hyperparameters selected on chronological router-train folds only with a one-standard-error simplicity rule.
- Test cache was not loaded.

## Selection

Best fold config:

- `full_covariance_hv_decay0.99_ridge0.0001_sd0.25_sg0_bias0_alpha0.75_warm96`
- Fold MAE `0.341481`, baseline fold MAE `0.342418`, `3/4` fold wins.

One-SE selected config:

- `diagonal_variance_hv_decay0.99_ridge0.0001_sd1_sg0_bias0_alpha0.5_warm96`
- Fold MAE `0.342055`, baseline fold MAE `0.342418`, `3/4` fold wins.
- Mean condition `1.412`; fallback group rate `0.0`.

## Validation Result

| Method | Seeds | MAE | MSE | Improvement vs `0.363642` | Aggregate CI |
|---|---:|---:|---:|---:|---|
| fixed-three HV baseline | 5 | `0.363642 +/- 0.000014` | `0.306712 +/- 0.000016` | `0.000000` | n/a |
| selected residual covariance | 5 | `0.363649 +/- 0.000012` | `0.306458 +/- 0.000016` | `-0.000008` | `[-0.000105, 0.000121]` |

Worst horizon-variable regression:

- horizon `11`, variable `4`, delta `+0.007174` MAE.

## Decision

Do not promote residual-covariance weighting. Router-train fold signal did not transfer to validation, the paired CI crossed zero, and the selected method caused a major local regression.

## Reproduce

```powershell
python experiments\residual_covariance_weighting_costar\run_residual_covariance_weighting.py --device cuda
```

