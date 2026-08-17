# Final Frozen Test Evaluation

Date: 2026-08-13

Artifacts:

- `experiments/final_test_evaluation/run_final_frozen_test_evaluation.py`
- `experiments/final_test_evaluation/ETTh1_test_results.csv`
- `experiments/final_test_evaluation/ETTh2_test_results.csv`
- `experiments/final_test_evaluation/FINAL_TEST_RESULTS.json`
- `experiments/final_test_evaluation/FINAL_TEST_REPORT.md`

## Protocol

The preregistered freeze artifacts under `experiments/final_test_freeze/` were treated as authoritative.

Before loading test data, the runner verified:

- `model_frozen=true`
- `validation_tuning_complete=true`
- `test_loaded=false`
- `test_metrics_seen=false`

No model identities, subset sizes, hyperparameters, blend ratios, specialist parameters, or validation-selected choices were changed. The test split was loaded only after the freeze checks passed.

Device: `cuda`

Elapsed time: `46.20` seconds

Peak GPU memory: `499131392` bytes

## Results

| Dataset | Method | Expert set | Test MAE | Test MSE | Validation MAE | Validation MSE | MAE diff vs val |
|---|---|---|---:|---:|---:|---:|---:|
| ETTh1 | Best single expert | iTransformer | `0.339080` | `0.278551` | `0.376550` | `0.322095` | `-0.037470` |
| ETTh1 | Train-selected fixed core | PatchTST+iTransformer+TimesNet | `0.327128` | `0.266583` | `0.367265` | `0.310530` | `-0.040137` |
| ETTh1 | Full frozen adaptive model | PatchTST+iTransformer+TimesNet+DLinear+ModernTCN | `0.326395` | `0.267509` | `0.363112` | `0.306057` | `-0.036717` |
| ETTh2 | Best single expert | DLinear | `0.301708` | `0.222694` | `0.280957` | `0.171493` | `+0.020751` |
| ETTh2 | Train-selected fixed core | DLinear+PatchTST+ModernTCN | `0.304642` | `0.225185` | `0.280878` | `0.171933` | `+0.023764` |
| ETTh2 | Full frozen adaptive model | DLinear+PatchTST+ModernTCN | `0.297808` | `0.218612` | `0.276832` | `0.167280` | `+0.020976` |
| ETTh2 | DLinear+ModernTCN validation-selected reference | DLinear+ModernTCN | `0.299263` | `0.221853` | `0.275229` | `0.165345` | `+0.024034` |

## Interpretation

ETTh1:

- The full frozen adaptive model beat its train-selected fixed core on test MAE by `0.000733`.
- It did not beat the fixed core on test MSE; MSE was worse by `0.000926`.
- ETTh1 test was easier than validation for all reported rows.

ETTh2:

- The full frozen adaptive model beat its train-selected fixed core on test MAE by `0.006834`.
- It also beat the validation-selected `DLinear+ModernTCN` reference by `0.001455` MAE.
- ETTh2 test was worse than validation for every reported row, so the absolute validation level did not carry over.

Answer to the final question:

The frozen adaptive model's relative MAE improvement survived on genuinely unseen test data for both ETTh1 and ETTh2. The ETTh2 validation-selected fixed pair is recorded only as a reference, not a clean train-selected competitor.

## Leakage Notes

- The ETTh1 test cache was absent before authorization and was generated from frozen `final_60` expert checkpoints during the final-evaluation workflow.
- The ETTh2 locked-test cache was absent before authorization and was generated from frozen clean-candidate checkpoints during the final-evaluation workflow.
- ETTh2 direct-inference validation replay was checked before creating the ETTh2 locked-test cache. Maximum prediction difference versus the canonical validation cache was `0.000321`, maximum MAE-matrix difference was `0.000046`, and mean MAE-matrix difference was `0.00000239`.
- Chronological online updates reported `old_start + horizon <= current_start` through the existing `enforce_observable` checks.
- `test_evaluation_complete=true` is recorded in `experiments/final_test_evaluation/FINAL_TEST_RESULTS.json`.

## Reproduction

```powershell
python experiments/final_test_evaluation/run_final_frozen_test_evaluation.py
```

Do not rerun this command for tuning; the test results are now seen.
