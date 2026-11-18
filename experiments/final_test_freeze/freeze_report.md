# Final COSTAR-TS Pre-Test Model Freeze

This is a preregistered snapshot created before any test cache was loaded.

## ETTh1

- Core: `PatchTST+iTransformer+TimesNet`
- Core selection: router_train only
- Model: `hybrid_chrono_hvema_lowrank1_decay0.95_temp0.1_alpha0.75`
- Specialists: `DLinear+ModernTCN`
- Specialist config: `both_variable_decay0.95_cap0.1_marginbp200_warm96`
- Frozen validation MAE/MSE: `0.363112` / `0.306057`

## ETTh2

- Core: `DLinear+PatchTST+ModernTCN`
- Core selection: router_train only
- Model: `hybrid_chrono_hvema_lowrank1_decay0.95_temp0.1_alpha0.75`
- Core validation MAE/MSE: `0.280878` / `0.171933`
- Full frozen adaptive validation MAE/MSE: `0.276832` / `0.167280`
- Validation-selected `DLinear+ModernTCN` is retained only as a reference baseline.

## Frozen Hyperparameters

- Chronological EMA decay `0.97`, temperature `0.1`, online blend alpha `0.5`.
- Horizon-variable low-rank rank `1`, decay `0.95`, temperature `0.1`, alpha `0.75`.
- Chrono/HV blend: chrono `0.25`, HV `0.75`.
- Specialist config: variable decay `0.95`, cap `0.1`, margin `0.02`, warmup `96`.

## Freeze Status

- Validation tuning complete: `True`
- Model frozen: `True`
- Test loaded: `False`
- Test metrics seen: `False`
- Git commit: `c336955f9421f8c04983e856fb317c1db5bc2b5c`
- Timestamp UTC: `2026-08-12T23:57:07.884378+00:00`

## Artifacts

- `experiments/final_test_freeze/ETTh1_frozen_model.json`
- `experiments/final_test_freeze/ETTh2_frozen_model.json`
- `experiments/final_test_freeze/FINAL_MODEL_FREEZE.json`
- `experiments/final_test_freeze/freeze_report.md`
