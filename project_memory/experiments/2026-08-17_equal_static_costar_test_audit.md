# Equal-Static COSTAR Test Audit

Date: 2026-08-17

## Goal

Evaluate the active ETTh1 equal-static full adaptive COSTAR path on the already-generated final test cache after explicit user request.

This is an after-final-test audit. The original final test results had already been viewed before the equal-static cleanup, so this row does not replace the preregistered final-test result.

The equal-static path is now the main full adaptive COSTAR implementation for ETTh1 going forward. The older preregistered result remains the historical confirmatory final-test record.

## Protocol

- Dataset: `ETTh1`
- Split: test `80-100%`
- Test starts: `11520..14292`
- Test windows: `2773`
- Horizon: `12`
- Variables: `7`
- Core: `PatchTST+iTransformer+TimesNet`
- Specialists: `DLinear+ModernTCN`
- Static prior: equal `1/3` for every selected triple
- Online updates: causal, using `old_start + horizon <= current_start`
- Seeds: `7,11,13,17,19`, deterministic after equal-static cleanup
- Label: `after_final_test_audit`

No tuning, expert-set changes, or hyperparameter changes were made after loading the test cache.

## Result

| Method | Test MAE | Test MSE | Validation MAE | Validation MSE |
|---|---:|---:|---:|---:|
| Equal-static full adaptive COSTAR | `0.326408` | `0.267378` | `0.363100` | `0.306026` |
| Train-selected fixed core | `0.327128` | `0.266583` | `0.367265` | `0.310530` |
| Old preregistered full adaptive reference | `0.326395` | `0.267509` | `0.363112` | `0.306057` |

Differences:

- Equal-static vs fixed core test MAE: `-0.000720`
- Equal-static vs old preregistered full adaptive test MAE: `+0.000013`
- Equal-static test vs equal-static validation MAE: `-0.036692`

## Interpretation

The equal-static cleanup preserves nearly the same ETTh1 test performance as the old neural-static-prior path. It is slightly worse on test MAE by `0.000013`, while improving test MSE by `0.000131`.

This supports the structural cleanup as the main full adaptive model for ETTh1, but it should not supersede the original preregistered final-test result because it was evaluated after final test results were already known.

## Artifacts

- `experiments/equal_static_costar_test_audit/run_equal_static_etth1_test_audit.py`
- `experiments/equal_static_costar_test_audit/MAIN_ETTH1_FULL_ADAPTIVE_MODEL.json`
- `experiments/equal_static_costar_test_audit/EQUAL_STATIC_ETTH1_TEST_AUDIT.json`
- `experiments/equal_static_costar_test_audit/EQUAL_STATIC_ETTH1_TEST_AUDIT.md`
- `experiments/equal_static_costar_test_audit/equal_static_etth1_test_results.csv`
- `experiments/equal_static_costar_test_audit/equal_static_etth1_test_per_seed.csv`

Reproduce:

```powershell
python experiments\equal_static_costar_test_audit\run_equal_static_etth1_test_audit.py --device cuda
```
