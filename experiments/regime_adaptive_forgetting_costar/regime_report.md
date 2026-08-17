# Regime Adaptive Forgetting COSTAR-TS

## Protocol

- Frozen baseline shape: `hybrid_chrono_hvema_lowrank1_decay0.95_temp0.1_alpha0.75`.
- Only the horizon-variable EMA forgetting speed changes.
- Detector settings were selected on router-train chronological folds.
- Oracle change points are diagnostic only and ineligible.
- Test cache was not loaded.

## Selection

- Selected config: `zscore_slow0.99_fast0.95_thr2.5_delta0_reset0_cool24_boost24`.
- Best fold config: `zscore_slow0.99_fast0.95_thr2.5_delta0_reset0_cool24_boost24`.
- One-SE threshold: `0.342249`.

## Validation

- Best eligible validation method: `baseline_fixed_decay`.
- MAE / MSE: `0.363642` / `0.306712`.
- Selected-method CI vs fixed decay: `[0.000618, 0.000790]`.
- Strong target `<= 0.3619`: `False`.

## Reproduce

```powershell
python experiments\regime_adaptive_forgetting_costar\run_regime_adaptive_forgetting.py --device cuda
```
