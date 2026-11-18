# ETTh2 Pair-Potential Linear Ensemble Test Audit

Date: 2026-08-14

Status: Completed after-final-test audit.

## Question

Evaluate the two ETTh2 pair-potential methods that previously had only router-validation metrics:

- `nonnegative_simplex_linear_average`
- `ridge_linear_stacker`

## Protocol

- Dataset: ETTh2.
- Router train: `cache/costarts_fresh/ETTh2_96_12/router_train_cache.pt`.
- Router validation: `cache/costarts_fresh/ETTh2_96_12/router_val_cache.pt`.
- Test: `experiments/final_test_evaluation/generated/caches/ETTh2/locked_test_cache_v2.pt`.
- Metric: canonical raw/cache-scale `sample_mae`/`sample_mse` equivalent with `std=ones`, no inverse transform.
- Expert order: `DLinear`, `PatchTST`, `iTransformer`, `TimesNet`, `ModernTCN`.
- No hyperparameter tuning or model selection after test load.
- The test cache was loaded only after writing `experiments/etth2_pair_potential_test_evaluation/manifest_before_test.json`.

## Frozen Weights

`nonnegative_simplex_linear_average`:

- DLinear `0.4706095457`
- PatchTST `0.1392433792`
- iTransformer `0.0`
- TimesNet `0.1643986851`
- ModernTCN `0.2257483751`

`ridge_linear_stacker`:

- DLinear `0.5178712606`
- PatchTST `0.2733554542`
- iTransformer `-0.0529077388`
- TimesNet `0.1123543382`
- ModernTCN `0.1667473316`

Both weight vectors reproduced the prior pair-potential validation metrics before test scoring.

## Results

| Method | Test MAE | Test MSE | Validation MAE | Diff vs DLinear test | Diff vs full adaptive test |
|---|---:|---:|---:|---:|---:|
| `nonnegative_simplex_linear_average` | `0.297120` | `0.218587` | `0.274755` | `-0.004588` | `-0.000688` |
| `ridge_linear_stacker` | `0.298382` | `0.218201` | `0.276702` | `-0.003325` | `+0.000574` |

The simplex linear average has the best MAE among these two and is below the earlier full adaptive ETTh2 test MAE `0.297808`. The ridge linear stacker improves over single `DLinear` and the validation-selected `DLinear+ModernTCN` reference on MAE, but it does not beat the full adaptive ETTh2 test MAE.

Paired bootstrap versus single `DLinear` test MAE:

- Simplex CI: `[-0.005597, -0.003595]`.
- Ridge CI: `[-0.004243, -0.002402]`.

## Artifacts

- `experiments/etth2_pair_potential_test_evaluation/run_etth2_pair_potential_test_evaluation.py`
- `experiments/etth2_pair_potential_test_evaluation/manifest_before_test.json`
- `experiments/etth2_pair_potential_test_evaluation/test_results.csv`
- `experiments/etth2_pair_potential_test_evaluation/ETTH2_PAIR_POTENTIAL_TEST_RESULTS.json`
- `experiments/etth2_pair_potential_test_evaluation/ETTH2_PAIR_POTENTIAL_TEST_REPORT.md`

## Verification

- `python -m py_compile experiments\etth2_pair_potential_test_evaluation\run_etth2_pair_potential_test_evaluation.py` passed.
- `python -m pytest ...` could not run because `pytest` is not installed in the active Python environment.

## Decision

Record these as after-final-test audit rows, not preregistered final competitors. The result is useful evidence that a simple train-fitted convex all-five ETTh2 ensemble generalized well on test, but it should not supersede the official preregistered frozen ETTh2 result unless the research program explicitly reopens model selection with a fresh holdout.
