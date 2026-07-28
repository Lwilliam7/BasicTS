# Sequential COSTARTS Report

## State

The router receives causal history, queried expert identities/mask, queried forecasts, current equal-average forecast, queried-forecast disagreement summaries, and queried count. Unqueried forecasts are absent from the state.

## Target

`utility_j = MAE(current_equal_average, target) - MAE(equal_average(S + j), target)` for each unused expert. STOP uses thresholded predicted utility.

## Validation

- Sequential COSTARTS MAE `0.347949` +/- `0.000991`.
- Zero-threshold sequential MAE `0.350830` +/- `0.002853`.
- Average queried experts `4.046` +/- `0.604`.
- Improvement over best fixed pair `0.003818` +/- `0.000991`.
- Utility correlation `0.0315`; useful-query AUC `0.5151`.
- Best fixed pair `PatchTST+ModernTCN` MAE `0.351767`.
- All-expert equal average MAE `0.350065`.
- Existing one-shot pair selector MAE `0.351172`.
- Greedy oracle sequential MAE `0.321288`.

## Success Checks

{
  "beats_best_fixed_pair_mean_validation_mae": true,
  "clear_cost_accuracy_tradeoff": false,
  "avoids_always_stop_or_query_all": true,
  "utility_prediction_better_than_random_directional": true,
  "no_data_leakage_known": true
}