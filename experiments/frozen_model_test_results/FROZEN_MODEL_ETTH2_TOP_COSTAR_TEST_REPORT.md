# Frozen-Model ETTh2 Top COSTAR Test Results

Every listed model was trained, selected, configured, and frozen without test-data feedback.

These are frozen-model test results. The original confirmatory evaluation remains the formally frozen evaluation in `experiments/final_test_evaluation/`; the other rows are additional frozen-model evaluations performed later.

| Method | Expert set | Test MAE | Test MSE | Val MAE | Diff vs DLinear test | Protocol |
|---|---|---:|---:|---:|---:|---|
| nonnegative_simplex_linear_average | DLinear+PatchTST+iTransformer+TimesNet+ModernTCN | 0.297120 | 0.218587 | 0.274755 | -0.004588 | after-final-test audit; router-train fitted pair-potential linear ensemble |
| full_adaptive_train_selected_fixed3_final_frozen | DLinear+PatchTST+ModernTCN | 0.297808 | 0.218612 | 0.276832 | -0.003899 | preregistered final ETTh2 model; train-selected fixed-3 core |
| ridge_linear_stacker | DLinear+PatchTST+iTransformer+TimesNet+ModernTCN | 0.298382 | 0.218201 | 0.276702 | -0.003325 | after-final-test audit; router-train fitted pair-potential linear stacker |
| fixed2_DLinear_TimesNet_reference | DLinear+TimesNet | 0.298398 | 0.221926 | 0.277652 | -0.003310 | validation-ranked fixed reference |
| fixed3_DLinear_TimesNet_ModernTCN_validation_selected_reference | DLinear+TimesNet+ModernTCN | 0.299169 | 0.223927 | 0.276644 | -0.002539 | validation-selected fixed-3 reference only |
| fixed2_DLinear_ModernTCN_validation_selected_reference | DLinear+ModernTCN | 0.299263 | 0.221853 | 0.275229 | -0.002445 | validation-selected reference only |
| full_adaptive_variable_size_core_DLinear | DLinear | 0.301093 | 0.222139 | 0.280470 | -0.000615 | router-train variable-size selected single core; frozen-model test result |
| single_DLinear | DLinear | 0.301708 | 0.222694 | 0.280957 | +0.000000 | canonical validation-best single anchor |
| fixed3_DLinear_PatchTST_ModernTCN_train_selected | DLinear+PatchTST+ModernTCN | 0.304642 | 0.225185 | 0.280878 | +0.002935 | router-train selected fixed-3 core |

Best frozen-model ETTh2 test MAE in this expanded audit: `nonnegative_simplex_linear_average` at `0.297120`.

The original confirmatory ETTh2 result remains the preregistered train-selected full frozen adaptive model from `experiments/final_test_evaluation/`.
