# Conditional combination comparison

Status: BASELINES_ONLY_PRIMARY_MISSING

Development evaluation on reused validation data. No test access.

| Method | MAE | MSE |
| --- | ---: | ---: |
| equal_all | 0.371099 | 0.311582 |
| train_best_single | 0.379558 | 0.326528 |
| fforma_mae | 0.369073 | 0.310212 |
| conditional_error_softmax | 0.368080 | 0.310985 |
| conditional_mlp | 0.386498 | 0.337496 |

Paired differences, horizon-scaled moving-block intervals, selection audit,
and feature failures are in results.json, selection.json and protocol.json.
Intervals are descriptive pairwise intervals without multiplicity adjustment.
Seeds average losses, not forecasts; they do not increase the temporal sample size.

The frozen primary was not supplied. This is not a completed comparison against CellCatch.
