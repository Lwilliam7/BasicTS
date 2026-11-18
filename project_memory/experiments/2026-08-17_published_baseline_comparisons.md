# Published Baseline Comparisons

Date: 2026-08-17

## Goal

Implement validation-only published comparison baselines for COSTAR without changing existing COSTAR algorithms:

1. Granger-Ramanathan direct forecast stacking
2. FAME routing adaptation
3. TimeRouter routing-mechanism adaptation
4. Bates-Granger forecast combination
5. OneNet-style frozen-expert adaptation

No ETTh1 or ETTh2 test cache was loaded.

## Protocol

- Frozen expert forecasts reused from the existing COSTAR caches.
- Expert pool: `DLinear`, `PatchTST`, `iTransformer`, `TimesNet`, `ModernTCN`.
- ETTh1 router-train: `cache/costarts_walkforward/router_train_20_60_cache.pt`.
- ETTh1 router-val: `cache/costarts_walkforward/router_val_60_80_cache.pt`.
- ETTh2 router-train: `cache/costarts_fresh/ETTh2_96_12/router_train_cache.pt`.
- ETTh2 router-val: `cache/costarts_fresh/ETTh2_96_12/router_val_cache.pt`.
- Hyperparameters selected only by chronological `train_folds(...)` inside router-train.
- Selected configurations were written before router-val scoring.
- Metrics use repository `sample_mae` / `sample_mse`.

## Results

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

## Selected Configurations

ETTh1:

- Granger-Ramanathan: horizon-variable ridge extension, alpha `1.0`.
- FAME adaptation: tau `0.1`, top-r `3`, hidden `64`, dropout `0.1`, lr `0.0003`, weight decay `0.001`, epochs `80`, seed `7`.
- TimeRouter adaptation: tau_m `0.15`, tau_d `0.5`, hidden `64`, lr `0.0003`, weight decay `0.001`, epochs `80`, seed `7`.
- Bates-Granger: horizon-variable diagonal inverse-error.
- OneNet-style adaptation: eta `0.5`, decay `0.97`.

ETTh2:

- Granger-Ramanathan: global ridge extension, alpha `0.01`.
- FAME adaptation: tau `0.1`, top-r `3`, hidden `64`, dropout `0.1`, lr `0.0003`, weight decay `0.001`, epochs `80`, seed `7`.
- TimeRouter adaptation: tau_m `0.15`, tau_d `0.5`, hidden `64`, lr `0.0003`, weight decay `0.001`, epochs `80`, seed `7`.
- Bates-Granger: global covariance, shrinkage `0.0`.
- OneNet-style adaptation: eta `0.5`, decay `0.97`.

## Interpretation

Bates-Granger is the strongest new published baseline on ETTh2 validation, beating online COSTAR MAE but not the ETTh2 validation-tuned Ridge residual row. None of the newly implemented published baselines beat the main ETTh1 online COSTAR validation result.

FAME, TimeRouter, and OneNet are adaptations, not exact reproductions, because the official settings use different expert pools, metadata/context, TSFM checkpoints, or online expert updating.

## Artifacts

- `experiments/published_baseline_comparisons/run_published_baselines.py`
- `experiments/published_baseline_comparisons/FINAL_REPORT.json`
- `experiments/published_baseline_comparisons/PUBLISHED_BASELINE_COMPARISON_REPORT.md`
- `experiments/published_baseline_comparisons/validation_results.csv`
- `experiments/published_baseline_comparisons/ablation_results.csv`
- `experiments/published_baseline_comparisons/per_window_metrics.csv`
- `experiments/published_baseline_comparisons/ETTh1/frozen_config_before_validation.json`
- `experiments/published_baseline_comparisons/ETTh2/frozen_config_before_validation.json`
- `tests/test_published_baseline_comparisons.py`

## Reproduce

```powershell
python experiments\published_baseline_comparisons\run_published_baselines.py --phase all --device cuda
```

Focused checks were run directly because `pytest` is not installed in the current Python environment:

```powershell
python -m py_compile experiments\published_baseline_comparisons\run_published_baselines.py tests\test_published_baseline_comparisons.py
```
