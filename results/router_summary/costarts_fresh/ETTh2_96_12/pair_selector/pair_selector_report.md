# ETTh2 Pair Selector Report

## Validation Summary

- Fixed pair: `DLinear+ModernTCN` MAE `0.275229`.
- Predicted-pair mean MAE: `0.276947` +/- `0.003435`.
- Mean improvement over fixed pair: `-0.001718`.
- Exact pair accuracy: `28.35%`.
- Top-two pair coverage: `49.46%`.
- Cross-seed mean agreement: `0.819`.
- All five seeds agree on `9.62%` of validation windows.

## Decision Answers

1. Always-use predicted pair beats fixed pair on average: `False`.
2. Stability across five seeds: std MAE `0.003435`.
3. High-margin selected-pair MAE `0.385374` versus low-margin `0.266340`.
4. Best diagnostic confidence separator: `fixed_pair_probability_negative` with AUC `0.5118937151105416`.
5. Confidence stability: mean max-probability variance `0.000255`.
6. Evidence for a constrained gate: `False` based on forecast MAE, pending confidence-separation analysis.

## Leakage

Only router_train/router_val caches, cache validation report, and pair-potential summary were loaded. No ETTh2 test arrays or test cache were created.