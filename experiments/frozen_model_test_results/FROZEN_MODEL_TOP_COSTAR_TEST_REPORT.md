# Frozen-Model Top COSTAR Test Results

Every listed model was trained, selected, configured, and frozen without test-data feedback.

These are frozen-model test results. The original confirmatory evaluation remains the formally frozen evaluation in `experiments/final_test_evaluation/`; the other rows are additional frozen-model evaluations performed later.

| Method | Test MAE | Test MSE | Val MAE | Diff vs test fixed core | Seeds | Note |
|---|---:|---:|---:|---:|---:|---|
| mlp_residual_corrector | 0.326047 | 0.267322 | 0.363318 | -0.001081 | 5 | pre_test_frozen |
| expanded_both_final_frozen | 0.326393 | 0.267506 | 0.363112 | -0.000735 | 5 | pre_test_frozen |
| expanded_dlinear_only | 0.326437 | 0.267593 | 0.363510 | -0.000691 | 5 | pre_test_frozen |
| ridge_residual_corrector | 0.326448 | 0.267452 | 0.363301 | -0.000680 | 5 | pre_test_frozen |
| expanded_moderntcn_only | 0.326468 | 0.267591 | 0.363435 | -0.000660 | 5 | pre_test_frozen |
| horizon_variable_hybrid | 0.326493 | 0.267638 | 0.363642 | -0.000635 | 5 | pre_test_frozen |
| chronological_ema_hybrid | 0.326548 | 0.266643 | 0.365534 | -0.000580 | 5 | pre_test_frozen |
| oracle_prototype_residual | 0.326829 | 0.267364 | 0.366028 | -0.000299 | 5 | pre_test_frozen |
| nonnegative_simplex_linear_average | 0.326926 | 0.267713 | 0.366483 | -0.000203 | 0 | after_final_test_audit |
| fixed_core_equal | 0.327128 | 0.266583 | 0.367265 | +0.000000 | 0 | pre_test_frozen |
| dynamic_fixed3_seed7 | 0.329249 | 0.272063 | 0.365985 | +0.002121 | 1 | pre_test_frozen |

Best frozen-model test MAE: `mlp_residual_corrector` at `0.326047`.

The original confirmatory result remains the preregistered frozen adaptive model from `experiments/final_test_evaluation/`.
