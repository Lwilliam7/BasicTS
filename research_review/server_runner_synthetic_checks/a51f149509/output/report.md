# Conditional combination comparison

Status: COMPLETE

Development evaluation on reused validation data. No test access.

| Method | MAE | MSE |
| --- | ---: | ---: |
| archived_equal5 | 0.270444 | 0.128504 |
| archived_equal3 | 0.347463 | 0.211576 |
| archived_bates_all5 | 0.270444 | 0.128504 |
| equal_all | 0.270444 | 0.128504 |
| train_best_single | 0.602164 | 0.633318 |
| fforma_mae | 0.270444 | 0.128504 |
| conditional_error_softmax | 0.270444 | 0.128504 |
| conditional_mlp | 0.270444 | 0.128504 |
| wide_top3 | 0.347463 | 0.211576 |

Paired differences, horizon-scaled moving-block intervals, selection audit,
and feature failures are in results.json, selection.json and protocol.json.
Intervals are descriptive pairwise intervals without multiplicity adjustment.
Seeds average losses, not forecasts; they do not increase the temporal sample size.

Archived primary association passed numerical and metadata checks. Its original
cache hashes were not recorded, so historical cache byte identity is not proven.
