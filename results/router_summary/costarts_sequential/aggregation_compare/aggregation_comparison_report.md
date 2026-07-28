# Sequential COSTARTS Aggregation Comparison

The router decisions, thresholds, and selected expert subsets are fixed from the existing sequential COSTARTS checkpoints. Only the final aggregation over queried forecasts changes.

## Mean Validation MAE

- `equal_average`: `0.347949` +/- `0.000991`
- `learned_global_convex_train`: `0.350585` +/- `0.001087`
- `validation_selected_train_error_softmax`: `0.347488` +/- `0.001104`

## Winner

`validation_selected_train_error_softmax` has the lowest mean validation MAE.

## Leakage Note

No final test cache is loaded. Learned convex weights are fit on `router_train`; train-error softmax temperatures are selected on `router_val`, so that row is a validation diagnostic and should not be treated as locked-test evidence.