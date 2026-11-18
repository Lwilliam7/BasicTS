# ETTh2 Pair-Potential Linear Ensemble Test Audit

This is an after-final-test audit of two ETTh2 methods that previously existed only as router-train-fitted validation rows.
No hyperparameters or weights were changed after loading test.

Created UTC: `2026-08-14T06:14:17.558949+00:00`
Git commit: `c336955f9421f8c04983e856fb317c1db5bc2b5c`

| Method | Test MAE | Test MSE | Val MAE | Diff vs DLinear test | Diff vs full adaptive test |
|---|---:|---:|---:|---:|---:|
| nonnegative_simplex_linear_average | 0.297120 | 0.218587 | 0.274755 | -0.004588 | -0.000688 |
| ridge_linear_stacker | 0.298382 | 0.218201 | 0.276702 | -0.003325 | +0.000574 |

## Leakage Checks

- Weights were fitted from `cache/costarts_fresh/ETTh2_96_12/router_train_cache.pt` only.
- Validation was used only to reproduce the existing pair-potential validation numbers.
- The ETTh2 locked test cache was loaded only after `manifest_before_test.json` was written.
- No test result was used to change weights, hyperparameters, or method membership.

## Reproduce

```powershell
python experiments/etth2_pair_potential_test_evaluation/run_etth2_pair_potential_test_evaluation.py
```