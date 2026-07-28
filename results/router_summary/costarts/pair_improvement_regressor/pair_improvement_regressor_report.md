# Old ETTh1 COSTARTS Pair-Improvement Regressor

## Target

`target[pair] = fixed_pair_error - candidate_pair_error`; positive means the candidate pair beats the fixed pair.

## Validation Results

- Fixed validation-selected pair `PatchTST+ModernTCN` MAE `0.351767`.
- Current exact oracle-pair classifier mean MAE `0.351172` +/- `0.001435`.
- Improvement regressor no-threshold mean MAE `0.359916` +/- `0.003775`.
- Improvement regressor validation-threshold mean MAE `0.351341` +/- `0.000322`.
- Threshold switch rate `6.56%`; switched-window win rate `61.34%`.
- Beneficial/harmful switch AUC `0.472`.

## Success

{
  "beats_fixed_pair_mean_validation_mae": true,
  "sensible_switch_rate_mean_percent": 6.557911932468414,
  "identifies_beneficial_switches_better_than_random": false,
  "num_seeds": 5
}