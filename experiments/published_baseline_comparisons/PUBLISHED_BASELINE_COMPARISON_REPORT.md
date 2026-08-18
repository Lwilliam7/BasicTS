# Published Baseline Comparisons for COSTAR

Validation-only comparison. No ETTh1 or ETTh2 test cache is loaded.

## Comparison Table

| Method | ETTh1 Val MAE | ETTh1 Val MSE | ETTh2 Val MAE | ETTh2 Val MSE |
|---|---:|---:|---:|---:|
| Equal fixed ensemble | `0.371099` | `0.311582` | `0.300772` | `0.200833` |
| Granger-Ramanathan | `0.382960` | `0.336499` | `0.276704` | `0.165286` |
| Bates-Granger | `0.368891` | `0.309925` | `0.274915` | `0.165315` |
| FAME adaptation | `0.379212` | `0.326919` | `0.277008` | `0.167165` |
| TimeRouter adaptation | `0.368234` | `0.309054` | `0.283288` | `0.175959` |
| Frozen COSTAR | `0.365825` | `0.308399` | `0.277481` | `0.167632` |
| Online COSTAR | `0.363100` | `0.306026` | `0.276832` | `0.167280` |
| Frozen COSTAR + Ridge residual | `0.363301` | `0.306286` | `0.275036` | `0.165619` |
| Frozen COSTAR + MLP residual | `0.363318` | `0.306607` | `0.275643` | `0.166147` |
| OneNet / adaptation | `0.370137` | `0.314488` | `0.402666` | `0.394105` |

## Selected Hyperparameters

### ETTh1

- Granger-Ramanathan: `{"alpha": 1.0, "method": "Granger-Ramanathan", "structure": "horizon_variable", "variant": "ridge_extension"}`
- FAME adaptation: `{"dropout": 0.1, "epochs": 80, "hidden": 64, "lr": 0.0003, "seed": 7, "tau": 0.1, "top_r": 3, "weight_decay": 0.001}`
- TimeRouter adaptation: `{"epochs": 80, "hidden": 64, "lr": 0.0003, "seed": 7, "tau_d": 0.5, "tau_m": 0.15, "weight_decay": 0.001}`
- Bates-Granger: `{"estimator": "diagonal_inverse_error", "shrinkage": 1.0, "structure": "horizon_variable"}`
- OneNet-style frozen-expert adaptation: `{"decay": 0.97, "eta": 0.5}`

### ETTh2

- Granger-Ramanathan: `{"alpha": 0.01, "method": "Granger-Ramanathan", "structure": "global", "variant": "ridge_extension"}`
- FAME adaptation: `{"dropout": 0.1, "epochs": 80, "hidden": 64, "lr": 0.0003, "seed": 7, "tau": 0.1, "top_r": 3, "weight_decay": 0.001}`
- TimeRouter adaptation: `{"epochs": 80, "hidden": 64, "lr": 0.0003, "seed": 7, "tau_d": 0.5, "tau_m": 0.15, "weight_decay": 0.001}`
- Bates-Granger: `{"estimator": "covariance", "shrinkage": 0.0, "structure": "global"}`
- OneNet-style frozen-expert adaptation: `{"decay": 0.97, "eta": 0.5}`

## Leakage And Provenance Checks

- Test cache loaded: `False`.
- Every dataset loader rejects paths containing `test`.
- Cache schemas were checked for expert order, chronological starts, `[N,12,7,5]` prediction stacks, `[N,12,7]` targets/masks, input length `96`, and horizon `12`.
- Hyperparameter selection used chronological prefixes inside `router_train`; selected configs were saved under per-dataset `frozen_config_before_validation.json` before final `router_val` rows were recorded.
- OneNet-style online updates call `enforce_observable(old_start, current_start, horizon)` before any realized error update.
- Frozen experts are never trained or updated; only routers/combination weights are fit on cached predictions.

## Artifacts

- `FINAL_REPORT.json`: machine-readable full report.
- `validation_results.csv`: validation MAE/MSE rows.
- `ablation_results.csv`: global/horizon-variable, covariance/diagonal, sparse/top-k, hard/fallback, and online ablations.
- `per_window_metrics.csv`: per-window validation MAE/MSE.
- `ETTh1/frozen_config_before_validation.json` and `ETTh2/frozen_config_before_validation.json`: frozen selected configs and cache hashes.

## Implemented Algorithms

- Granger-Ramanathan: direct linear target regression from expert forecasts, with global and horizon-variable OLS/ridge candidates.
- Bates-Granger: covariance-weighted and diagonal inverse-error forecast combination using router-train forecast errors only.
- FAME adaptation: forecastability fingerprint, soft expert-suitability targets from router-train losses, and sparse Top-r routing over the BasicTS frozen expert pool.
- TimeRouter adaptation: lightweight discriminative routing head with margin/diversity selective gate and inverse-error fallback ensemble.
- OneNet-style adaptation: delayed-feedback online ensembling over frozen PatchTST/iTransformer forecasts.

## Deviations From Official Papers

- Granger-Ramanathan and Bates-Granger are direct classical frozen-forecast combinations on the COSTAR expert cache.
- FAME is labeled `FAME routing adaptation to BasicTS frozen expert pool`: it keeps FAME's forecastability fingerprints, oracle suitability targets, and sparse Top-r routing, but replaces the official retail/industrial expert pool, metadata, context, and cost model with the five frozen BasicTS experts.
- TimeRouter is labeled `TimeRouter routing-mechanism adaptation`: it keeps discriminative routing, selective margin/diversity gating, and inverse-error fallback behavior, but replaces the official XGBoost TSFM router/checkpoints with a small Torch routing head over BasicTS cache features.
- OneNet is labeled `OneNet-style frozen-expert adaptation`: it adapts delayed online ensembling to frozen PatchTST/iTransformer forecasts and does not update forecasting experts.

## Sources Inspected

- FAME official repository: https://github.com/hit636/FAME
- TimeRouter official repository: https://github.com/UConn-DSIS/TimeRouter
- OneNet official repository: https://github.com/yfzhang114/OneNet

## Reproduce

```powershell
python experiments\published_baseline_comparisons\run_published_baselines.py --phase all --device cuda
```
