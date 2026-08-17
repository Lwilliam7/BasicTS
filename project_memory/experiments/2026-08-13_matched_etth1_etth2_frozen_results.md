# Matched ETTh1 / ETTh2 Frozen-Model Results

Date: 2026-08-13

Status: completed

## Purpose

Create a matched table for the ETTh1 top frozen-model test rows and the corresponding ETTh2 frozen-model test results where a valid ETTh2 analogue exists.

## Artifacts

- `experiments/frozen_model_test_results/run_etth2_matched_table.py`
- `experiments/frozen_model_test_results/matched_etth1_etth2_results.csv`
- `experiments/frozen_model_test_results/MATCHED_ETTH1_ETTH2_RESULTS.json`
- `experiments/frozen_model_test_results/MATCHED_ETTH1_ETTH2_RESULTS.md`
- `experiments/all_results_summary/all_costar_results.csv`
- `experiments/all_results_summary/ALL_COSTAR_RESULTS.md`

## ETTh2 Matched Results

| Method | ETTh2 Test MAE | ETTh2 Test MSE | ETTh2 Val MAE | Status |
|---|---:|---:|---:|---|
| Full adaptive model | `0.297808` | `0.218612` | `0.276832` | `pre_test_frozen` |
| Expanded DLinear only | `0.297808` | `0.218612` | `0.276832` | duplicate specialist disabled |
| Expanded ModernTCN only | `0.297808` | `0.218612` | `0.276832` | duplicate specialist disabled |
| Horizon-variable hybrid | `0.297808` | `0.218612` | `0.276832` | same prediction as full adaptive on ETTh2 core |
| Chronological EMA hybrid | `0.301689` | `0.222371` | `0.278806` | `pre_test_frozen` |
| Fixed-three core | `0.304642` | `0.225185` | `0.280878` | `pre_test_frozen` |
| Best single | `0.301708` | `0.222694` | `0.280957` | `pre_test_frozen` |

Rows without a frozen ETTh2 artifact:

- MLP residual corrector
- Ridge residual corrector
- Oracle prototype residual
- Dynamic fixed-three, seed 7

## Interpretation

The pasted ETTh1 table did not have ETTh2 test values for every row because several rows are ETTh1-only trained residual/router artifacts. The repository does not contain matching frozen ETTh2 artifacts for those methods.

For valid ETTh2 analogues, the matched table has been generated and appended to `all_costar_results.csv`. On ETTh2, the final selected core includes both `DLinear` and `ModernTCN`; therefore DLinear-only and ModernTCN-only specialist rows are not distinct from the full/horizon-variable prediction because duplicate specialist branches are disabled.
