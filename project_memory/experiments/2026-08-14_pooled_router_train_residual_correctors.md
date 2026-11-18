# Pooled Router-Train Residual Correctors

Date: 2026-08-14

Status: completed after-final-test audit

## Definition

This uses the same pooled definition as `pooled_router_train_core`: select the configuration by MAE/MSE over all router-train windows together, with no chronological folds. Then fit the final corrector on all router-train rows and evaluate validation/test.

## Protocol

- Label: `after_final_test_audit`
- Test metrics had already been viewed before this experiment.
- Validation and test caches were not loaded before `manifest_before_test.json`.
- Ridge selection is deterministic.
- MLP config selection used seed `7` on pooled router-train, then the selected config was refit for seeds `7, 11, 13, 17, 19`.
- MLP fitting used CUDA.

## Results

| Dataset | Method | Selected config | Train MAE | Val MAE | Val MSE | Test MAE | Test MSE | Diff vs existing corrector | Diff vs full adaptive |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| ETTh1 | Ridge residual corrector | `ridge1_alpha0.5_clip0.5_full` | `0.337319` | `0.363088` | `0.305485` | `0.328022` | `0.267930` | `+0.001574` | `+0.001629` |
| ETTh1 | MLP residual corrector | `mlp_alpha0p1_clip_multiple0p5_epochs40_hidden64_lr0p0003_patience6_weight_decay0p01` | `0.334217` | `0.364111` | `0.307359` | `0.325964` | `0.266587` | `-0.000083` | `-0.000429` |
| ETTh2 | Ridge residual corrector | `ridge1_alpha0.25_clip0.25_full` | `0.277668` | `0.275036` | `0.165619` | `0.296787` | `0.217713` | `+0.000000` | `-0.001021` |
| ETTh2 | MLP residual corrector | `mlp_alpha0p1_clip_multiple0p5_epochs40_hidden64_lr0p0003_patience6_weight_decay0p01` | `0.275051` | `0.275975` | `0.166649` | `0.297427` | `0.218405` | `+0.000386` | `-0.000381` |

## Interpretation

Pooled router-train selection changes the story compared with fold/validation selection:

- ETTh1 pooled-selected Ridge overfits router-train/validation and fails on test, scoring worse than the fixed core and full adaptive model.
- ETTh1 pooled-selected MLP has the best audited ETTh1 test MAE so far, `0.325964`, but it is an after-final-test sensitivity row, not a preregistered final result.
- ETTh2 pooled-selected Ridge chooses the same config as the ETTh2 validation-tuned Ridge and reproduces the same test result.
- ETTh2 pooled-selected MLP is worse than the ETTh2 validation-tuned MLP.

## Artifacts

- `experiments/pooled_router_train_residual_correctors/run_pooled_router_train_residual_correctors.py`
- `experiments/pooled_router_train_residual_correctors/manifest_before_test.json`
- `experiments/pooled_router_train_residual_correctors/ridge_pooled_train_grid.csv`
- `experiments/pooled_router_train_residual_correctors/mlp_pooled_train_grid.csv`
- `experiments/pooled_router_train_residual_correctors/pooled_router_train_residual_results.csv`
- `experiments/pooled_router_train_residual_correctors/seed_results.csv`
- `experiments/pooled_router_train_residual_correctors/paired_ci.csv`
- `experiments/pooled_router_train_residual_correctors/POOLED_ROUTER_TRAIN_RESIDUAL_RESULTS.json`
- `experiments/pooled_router_train_residual_correctors/POOLED_ROUTER_TRAIN_RESIDUAL_REPORT.md`
