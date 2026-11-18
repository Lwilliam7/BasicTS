# ETTh1 Simplex Linear Ensemble Test Audit

Date: 2026-08-14

Status: Completed after-final-test audit.

## Question

Evaluate the ETTh1 analogue of the ETTh2 `nonnegative_simplex_linear_average`: one all-five convex linear ensemble fit on router-train only.

## Protocol

- Dataset: ETTh1.
- Router train: `cache/costarts_walkforward/router_train_20_60_cache.pt`.
- Router validation: `cache/costarts_walkforward/router_val_60_80_cache.pt`.
- Test: `experiments/final_test_evaluation/generated/caches/ETTh1/test_80_100_cache.pt`.
- Normalizer: `checkpoints/costarts_walkforward/final_60/DLinear/best_expert.pt`.
- Metric: repository `sample_mae` / `sample_mse` with the ETTh1 normalizer.
- Expert order: `DLinear`, `PatchTST`, `iTransformer`, `TimesNet`, `ModernTCN`.
- No hyperparameter tuning or model selection after test load.
- The test cache was loaded only after writing `experiments/etth1_simplex_linear_test_evaluation/manifest_before_test.json`.

## Frozen Weights

- DLinear `0.1167512611`
- PatchTST `0.3640883863`
- iTransformer `0.3396539986`
- TimesNet `0.1488531232`
- ModernTCN `0.0306532476`

## Results

| Method | Test MAE | Test MSE | Validation MAE | Diff vs fixed core test | Diff vs full adaptive test |
|---|---:|---:|---:|---:|---:|
| `nonnegative_simplex_linear_average` | `0.326926` | `0.267713` | `0.366483` | `-0.000203` | `+0.000530` |

Paired bootstrap versus fixed-three core test MAE:

- CI: `[-0.000767, 0.000368]`.
- Excludes zero: `False`.

## Artifacts

- `experiments/etth1_simplex_linear_test_evaluation/run_etth1_simplex_linear_test_evaluation.py`
- `experiments/etth1_simplex_linear_test_evaluation/manifest_before_test.json`
- `experiments/etth1_simplex_linear_test_evaluation/test_results.csv`
- `experiments/etth1_simplex_linear_test_evaluation/ETTH1_SIMPLEX_LINEAR_TEST_RESULTS.json`
- `experiments/etth1_simplex_linear_test_evaluation/ETTH1_SIMPLEX_LINEAR_TEST_REPORT.md`

## Verification

- `python -m py_compile experiments\etth1_simplex_linear_test_evaluation\run_etth1_simplex_linear_test_evaluation.py tests\test_etth1_simplex_linear_test_evaluation.py` passed.
- `pytest` is not installed in the active Python environment, so the focused tests were compiled but not executed through pytest.

## Decision

Record this as an after-final-test audit row. ETTh1 simplex is a reasonable simple baseline and slightly beats the fixed-three core on MAE, but it does not beat the full adaptive model or the stronger residual/specialist audit rows, and its CI versus fixed-three crosses zero.
