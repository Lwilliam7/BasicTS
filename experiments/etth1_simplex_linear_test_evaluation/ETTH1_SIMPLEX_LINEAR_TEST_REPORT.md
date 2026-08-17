# ETTh1 Simplex Linear Ensemble Test Audit

This is an after-final-test audit of the ETTh1 analogue of the ETTh2 nonnegative simplex linear average.
Weights were fit once from ETTh1 router-train only; no test feedback was used to change them.

Created UTC: `2026-08-14T06:22:28.178085+00:00`
Git commit: `c336955f9421f8c04983e856fb317c1db5bc2b5c`

| Method | Test MAE | Test MSE | Val MAE | Diff vs fixed core test | Diff vs full adaptive test |
|---|---:|---:|---:|---:|---:|
| nonnegative_simplex_linear_average | 0.326926 | 0.267713 | 0.366483 | -0.000203 | +0.000530 |

## Weights

`{"DLinear": 0.11675126105546951, "PatchTST": 0.36408838629722595, "iTransformer": 0.33965399861335754, "TimesNet": 0.1488531231880188, "ModernTCN": 0.030653247609734535}`

## Leakage Checks

- Weights were fit from `cache/costarts_walkforward/router_train_20_60_cache.pt` only.
- Router validation was used only for reporting the frozen validation metric.
- The ETTh1 test cache was loaded only after `manifest_before_test.json` was written.
- No model or hyperparameter was changed after seeing the test result.

## Reproduce

```powershell
python experiments/etth1_simplex_linear_test_evaluation/run_etth1_simplex_linear_test_evaluation.py
```