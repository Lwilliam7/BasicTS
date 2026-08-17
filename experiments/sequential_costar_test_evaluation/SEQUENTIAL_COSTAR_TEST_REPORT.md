# Sequential COSTAR Test Evaluation

This is an after-final-test audit requested after held-out test metrics were already seen. Existing `utility_pairwise_weighted` Sequential COSTAR checkpoints were evaluated once on the existing final-test caches. No training, tuning, or checkpoint selection was performed.

- Device: `cuda`
- Runtime seconds: `116.379`
- Test cache loaded: `true`

| Dataset | Method | Test MAE mean | Test MAE std | Test MSE mean | Val MAE mean | Avg queries | Metric scale |
|---|---|---:|---:|---:|---:|---:|---|
| ETTh1 | Sequential COSTAR utility_pairwise_weighted | `0.330832` | `0.005462` | `0.271398` | `0.368074` | `3.776` | `normalized_by_ETTh1_DLinear_scaler` |
| ETTh2 | Sequential COSTAR utility_pairwise_weighted | `0.300576` | `0.000032` | `0.222171` | `0.277681` | `3.998` | `canonical_raw_std_ones` |

## Interpretation

- ETTh1 sequential utility routing remains worse than the previously tested fixed-core and adaptive COSTAR test rows.
- ETTh2 sequential utility routing remains worse than the final frozen adaptive ETTh2 test row.
- These rows should be treated as additional after-final-test audit results, not preregistered final competitors.

## Reproduce

```powershell
python experiments\sequential_costar_test_evaluation\run_sequential_costar_test_evaluation.py --device cuda
```
