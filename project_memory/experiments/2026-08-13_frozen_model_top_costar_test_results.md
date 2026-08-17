# frozen-model Top COSTAR Test Methods

Date: 2026-08-13

Artifacts:

- `experiments/frozen_model_test_results/run_frozen_model_top_costar_test_methods.py`
- `experiments/frozen_model_test_results/top_costar_test_results.csv`
- `experiments/frozen_model_test_results/top_costar_test_per_seed.csv`
- `experiments/frozen_model_test_results/TOP_COSTAR_TEST_RESULTS.json`
- `experiments/frozen_model_test_results/FROZEN_MODEL_TOP_COSTAR_TEST_REPORT.md`

## Status

additional frozen-model result test audit. The final preregistered test evaluation had already been run, so these results must not be used for tuning or for replacing the clean final frozen model.

Device: `cuda`

Elapsed time: `717.33` seconds

## Methods Tested

ETTh1 top validated COSTAR-style methods:

- Dynamic fixed-three router, seed 7 only because only that checkpoint was present.
- Oracle prototype-residual router.
- Chronological EMA hybrid.
- Horizon-variable hybrid.
- Ridge residual corrector.
- MLP residual corrector.
- Expanded DLinear-only specialist.
- Expanded ModernTCN-only specialist.
- Expanded both-specialists final frozen model.

The train-selected fixed core was included as the test anchor.

For multi-seed rows, the primary table uses the mean prediction across seeds, and the JSON/CSV also records per-seed mean and standard deviation. The clean final report used the preregistered frozen model's seed-mean metric.

## Results

| Method | Test MAE | Test MSE | Validation MAE | Diff vs test fixed core | Seeds |
|---|---:|---:|---:|---:|---:|
| MLP residual corrector | `0.326047` | `0.267322` | `0.363318` | `-0.001081` | 5 |
| Expanded both final frozen | `0.326393` | `0.267506` | `0.363112` | `-0.000735` | 5 |
| Expanded DLinear-only | `0.326437` | `0.267593` | `0.363510` | `-0.000691` | 5 |
| Ridge residual corrector | `0.326448` | `0.267452` | `0.363301` | `-0.000680` | 5 |
| Expanded ModernTCN-only | `0.326468` | `0.267591` | `0.363435` | `-0.000660` | 5 |
| Horizon-variable hybrid | `0.326493` | `0.267638` | `0.363642` | `-0.000635` | 5 |
| Chronological EMA hybrid | `0.326548` | `0.266643` | `0.365534` | `-0.000580` | 5 |
| Oracle prototype-residual | `0.326829` | `0.267364` | `0.366028` | `-0.000299` | 5 |
| Fixed core equal | `0.327128` | `0.266583` | `0.367265` | `+0.000000` | n/a |
| Dynamic fixed-three seed 7 | `0.329249` | `0.272063` | `0.365985` | `+0.002121` | 1 |

Per-seed mean highlights:

- MLP residual corrector: MAE `0.326062 +/- 0.000053`, MSE `0.267336 +/- 0.000144`.
- Expanded both final frozen: MAE `0.326395 +/- 0.000021`, MSE `0.267509 +/- 0.000055`.
- Ridge residual corrector: MAE `0.326451 +/- 0.000018`, MSE `0.267456 +/- 0.000051`.

## Interpretation

The MLP residual corrector is the best frozen-model ETTh1 test MAE in this audit, but it was not the preregistered final frozen model and it was rerun after test metrics were already known. Treat it as hypothesis-generating only.

The clean conclusion remains:

- The preregistered expanded-both final model beat the fixed core on ETTh1 test MAE.
- Test is now seen; do not use these rankings for another tuning loop.
