# Expanded Expert Pool COSTAR-TS

## Protocol

- Frozen fixed-three baseline: `hybrid_chrono_hvema_lowrank1_decay0.95_temp0.1_alpha0.75`.
- Optional experts: DLinear and ModernTCN.
- Optional weights are nonnegative and capped; no unconstrained stacking.
- Hyperparameters selected on router-train chronological folds only.
- Test cache was not loaded.

## Selected Configs

- `dlinear_only`: `dlinear_only_variable_decay0.95_cap0.1_marginbp200_warm96`
- `moderntcn_only`: `moderntcn_only_variable_decay0.95_cap0.05_marginbp200_warm96`
- `both`: `both_variable_decay0.95_cap0.1_marginbp200_warm96`

## Validation

| Method | MAE | MSE | Improvement vs 0.363642 | Aggregate CI |
|---|---:|---:|---:|---|
| `expanded_both` | `0.363112` | `0.306057` | `0.000529` | `[-0.000557, -0.000502]` |
| `expanded_moderntcn_only` | `0.363435` | `0.306452` | `0.000206` | `[-0.000219, -0.000194]` |
| `expanded_dlinear_only` | `0.363510` | `0.306557` | `0.000131` | `[-0.000144, -0.000119]` |
| `baseline_fixed3_hv` | `0.363642` | `0.306712` | `0.000000` | `[0.000000, 0.000000]` |

## Decision

Promote expanded pool.

## Reproduce

```powershell
python experiments\expanded_expert_pool_costar\run_expanded_expert_pool.py --device cuda
```
