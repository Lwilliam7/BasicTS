# Sequential COSTAR Test Evaluation

## Request

Evaluate the existing original/comparable Sequential COSTAR test performance after explicit user authorization.

## Protocol

- No retraining.
- No hyperparameter changes.
- Existing frozen `utility_pairwise_weighted` Sequential COSTAR checkpoints only.
- Seeds: `7, 11, 13, 17, 19`.
- ETTh1 test cache: `experiments/final_test_evaluation/generated/caches/ETTh1/test_80_100_cache.pt`.
- ETTh2 test cache: `experiments/final_test_evaluation/generated/caches/ETTh2/locked_test_cache_v2.pt`.
- Device: `cuda`.
- This is an after-final-test audit, not a preregistered final competitor.

## Results

| Dataset | Test MAE mean | Test MAE std | Test MSE mean | Validation MAE mean | Avg queries |
|---|---:|---:|---:|---:|---:|
| ETTh1 | `0.330832` | `0.005462` | `0.271398` | `0.368074` | `3.776` |
| ETTh2 | `0.300576` | `0.000032` | `0.222171` | `0.277681` | `3.998` |

## Interpretation

Sequential COSTAR did not beat the already tested frozen adaptive COSTAR models on held-out test:

- ETTh1 sequential was worse than final frozen adaptive by about `0.004437` MAE.
- ETTh2 sequential was worse than final frozen adaptive by about `0.002768` MAE.

The result reinforces the existing decision that ranking-objective sequential routing is not the current best direction.

## Artifacts

- `experiments/sequential_costar_test_evaluation/run_sequential_costar_test_evaluation.py`
- `experiments/sequential_costar_test_evaluation/manifest_before_test.json`
- `experiments/sequential_costar_test_evaluation/sequential_costar_test_results.csv`
- `experiments/sequential_costar_test_evaluation/sequential_costar_test_per_seed.csv`
- `experiments/sequential_costar_test_evaluation/SEQUENTIAL_COSTAR_TEST_RESULTS.json`
- `experiments/sequential_costar_test_evaluation/SEQUENTIAL_COSTAR_TEST_REPORT.md`

## Reproduce

```powershell
python experiments\sequential_costar_test_evaluation\run_sequential_costar_test_evaluation.py --device cuda
```
