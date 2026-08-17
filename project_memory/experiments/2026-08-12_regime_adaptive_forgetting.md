# Regime Adaptive Forgetting COSTAR-TS

## Hypothesis

Changing the causal horizon-variable EMA adaptation speed after detected residual shifts might improve over the fixed decay used by the current fixed-three HV baseline.

## Protocol

- Dataset: ETTh1 router-train `20-60%`, validation `60-80%`.
- Only the horizon-variable EMA forgetting speed was changed.
- Compared fixed decay, residual mean/variance z-score detector, Page-Hinkley detector, and oracle change points as an ineligible diagnostic.
- Detector thresholds, slow/fast decay, reset, cooldown, and boost duration were selected on chronological router-train folds only.
- Test cache was not loaded.

## Selection

Selected config:

- `zscore_slow0.99_fast0.95_thr2.5_delta0_reset0_cool24_boost24`
- Fold MAE `0.341991`, fixed-decay fold MAE `0.342333`.
- Fold delta `-0.000342`, `3/4` fold wins.
- Mean triggers `8.0`; fast update rate `0.175`.

## Validation Result

| Method | Seeds | MAE | MSE | Delta vs fixed decay | Aggregate CI |
|---|---:|---:|---:|---:|---|
| fixed decay baseline | 5 | `0.363642 +/- 0.000014` | `0.306712 +/- 0.000016` | `0.000000` | n/a |
| selected adaptive forgetting | 5 | `0.364346 +/- 0.000015` | `0.307367 +/- 0.000018` | `+0.000704` worse | `[0.000618, 0.000790]` |
| oracle change diagnostic, ineligible | 5 | `0.364015 +/- 0.000012` | `0.306907 +/- 0.000017` | `+0.000374` worse | `[0.000258, 0.000494]` |

Worst selected-method horizon-variable regression:

- horizon `10`, variable `5`, delta `+0.003153` MAE.

## Decision

Do not promote adaptive forgetting. The router-train gain reversed on validation, and even the oracle change-point diagnostic worsened performance.

## Reproduce

```powershell
python experiments\regime_adaptive_forgetting_costar\run_regime_adaptive_forgetting.py --device cuda
```

