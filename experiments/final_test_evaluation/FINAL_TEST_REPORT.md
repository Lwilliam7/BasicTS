# Final Frozen Test Evaluation

Created UTC: `2026-08-13T03:39:11.276370+00:00`
Git commit: `c336955f9421f8c04983e856fb317c1db5bc2b5c`

The preregistered freeze artifacts were verified before any test cache was loaded. No model, expert set, or hyperparameter was changed after seeing test metrics.

## ETTh1

| Method | Expert set | Test MAE | Test MSE | Validation MAE | Validation MSE | MAE diff vs val | Selection protocol |
|---|---|---:|---:|---:|---:|---:|---|
| Best single expert | iTransformer | 0.339080 | 0.278551 | 0.376550 | 0.322095 | -0.037470 | validation-best single reference from frozen fixed-ensemble summary |
| Train-selected fixed core | PatchTST+iTransformer+TimesNet | 0.327128 | 0.266583 | 0.367265 | 0.310530 | -0.040137 | core selected on ETTh1 router_train only; equal average |
| Full frozen adaptive model | PatchTST+iTransformer+TimesNet+DLinear+ModernTCN | 0.326395 | 0.267509 | 0.363112 | 0.306057 | -0.036717 | preregistered train-selected core plus frozen hybrid/HV/specialist architecture; five frozen seeds averaged |

## ETTh2

| Method | Expert set | Test MAE | Test MSE | Validation MAE | Validation MSE | MAE diff vs val | Selection protocol |
|---|---|---:|---:|---:|---:|---:|---|
| Best single expert | DLinear | 0.301708 | 0.222694 | 0.280957 | 0.171493 | +0.020751 | canonical validation-best single reference |
| Train-selected fixed core | DLinear+PatchTST+ModernTCN | 0.304642 | 0.225185 | 0.280878 | 0.171933 | +0.023764 | core selected on ETTh2 router_train only; equal average |
| Full frozen adaptive model | DLinear+PatchTST+ModernTCN | 0.297808 | 0.218612 | 0.276832 | 0.167280 | +0.020976 | preregistered train-selected core plus frozen hybrid/HV/specialist architecture; duplicate specialists disabled |
| DLinear+ModernTCN (validation-selected reference) | DLinear+ModernTCN | 0.299263 | 0.221853 | 0.275229 | 0.165345 | +0.024034 | validation-selected reference only; not clean train-selected competitor |

## Answer

- ETTh1 frozen adaptive model beat its train-selected fixed core on test MAE by `0.000733`, though its test MSE was `0.000926` worse than that fixed core.
- ETTh2 frozen adaptive model beat its train-selected fixed core on test MAE by `0.006834` and also beat the validation-selected DLinear+ModernTCN reference by `0.001455` MAE.
- ETTh2 test performance was worse than validation for every reported method, so the absolute validation level did not carry over even though the frozen adaptive ranking did.
- Test evaluation is complete and should not be rerun for tuning.

## Reproduce

```powershell
python experiments/final_test_evaluation/run_final_frozen_test_evaluation.py
```
