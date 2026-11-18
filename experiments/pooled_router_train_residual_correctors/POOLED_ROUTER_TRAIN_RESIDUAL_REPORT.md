# Pooled Router-Train Residual Correctors

Label: `after_final_test_audit`

Pooled means configuration selection used all router-train windows as one block, with no chronological folds.

## Results

| Dataset | Method | Selected config | Train MAE | Val MAE | Test MAE | Test MSE | Diff vs existing | Diff vs full adaptive |
|---|---|---|---:|---:|---:|---:|---:|---:|
| ETTh1 | Ridge residual corrector | `ridge1_alpha0.5_clip0.5_full` | 0.337319 | 0.363088 | 0.328022 | 0.267930 | +0.001574 | +0.001629 |
| ETTh1 | MLP residual corrector | `mlp_alpha0p1_clip_multiple0p5_epochs40_hidden64_lr0p0003_patience6_weight_decay0p01` | 0.334217 | 0.364111 | 0.325964 | 0.266587 | -0.000083 | -0.000429 |
| ETTh2 | Ridge residual corrector | `ridge1_alpha0.25_clip0.25_full` | 0.277668 | 0.275036 | 0.296787 | 0.217713 | +0.000000 | -0.001021 |
| ETTh2 | MLP residual corrector | `mlp_alpha0p1_clip_multiple0p5_epochs40_hidden64_lr0p0003_patience6_weight_decay0p01` | 0.275051 | 0.275975 | 0.297427 | 0.218405 | +0.000386 | -0.000381 |

## Interpretation

- This is an after-final-test sensitivity audit and does not supersede preregistered final-test rows.
- Ridge selection is deterministic.
- MLP config selection used seed 7 on pooled router-train; the selected config was then refit for seeds 7, 11, 13, 17, and 19.
- Validation and test were loaded only after selected configs and artifacts were recorded in `manifest_before_test.json`.

## Reproduce

```powershell
python experiments\pooled_router_train_residual_correctors\run_pooled_router_train_residual_correctors.py --device cuda
```
