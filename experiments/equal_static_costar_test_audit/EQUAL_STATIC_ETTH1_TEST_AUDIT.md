# ETTh1 Main Active Equal-Static COSTAR Test Audit

This is an after-final-test audit. The original final test set had already been evaluated before the equal-static cleanup.
No tuning, expert changes, or hyperparameter changes were made after loading the test cache.

This equal-static path is now the main active ETTh1 full adaptive COSTAR implementation going forward. The older preregistered final-test result remains the historical confirmatory record.

## Result

| Method | Test MAE | Test MSE | Validation MAE | Validation MSE |
|---|---:|---:|---:|---:|
| Equal-static full adaptive COSTAR | `0.326408` | `0.267378` | `0.363100` | `0.306026` |
| Train-selected fixed core | `0.327128` | `0.266583` | `0.367265` | `0.310530` |
| Old preregistered full adaptive reference | `0.326395` | `0.267509` | `0.363112` | `0.306057` |

## Differences

- Difference vs fixed core test MAE: `-0.000720`.
- Difference vs old preregistered full adaptive test MAE: `+0.000013`.
- Difference vs equal-static validation MAE: `-0.036692`.

## Protocol

- Dataset: `ETTh1`.
- Split: test `80-100%`, starts `11520..14292`, `2773` windows.
- Core: `PatchTST+iTransformer+TimesNet`.
- Static prior: equal `1/3` for every selected triple.
- Online updates: causal, using `old_start + horizon <= current_start`.
- Label: `after_final_test_audit`.

## Reproduce

```powershell
python experiments\equal_static_costar_test_audit\run_equal_static_etth1_test_audit.py --device cuda
```
