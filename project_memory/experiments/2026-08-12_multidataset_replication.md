# Locked Multi-Dataset Replication

## Scope

Requested datasets: ETTh2, ETTm1, ETTm2, Weather, Electricity.

Available non-test expert caches in this workspace:

- ETTh2 only: `cache/costarts_fresh/ETTh2_96_12/router_train_cache.pt` and `router_val_cache.pt`.

Missing existing expert caches:

- ETTm1
- ETTm2
- Weather
- Electricity

## Important Limitation

The full ETTh1 current-best model uses an ETTh1-specific static neural winner inside `hybrid_chrono_hvema_lowrank1_decay0.95_temp0.1_alpha0.75`. Equivalent static-winner artifacts were not available for ETTh2. Therefore the ETTh2 run is a limited available-cache replication of the locked expanded-specialist rule over the equal fixed-three cache baseline, not a full primary-model replication.

## ETTh2 Results

Baseline:

- equal PatchTST+iTransformer+TimesNet available-cache baseline: MAE `0.098339`, MSE `0.038581`.

Locked ETTh1 specialist config:

- Config: `both_variable_decay0.95_cap0.1_marginbp200_warm96`
- MAE `0.093369`, MSE `0.034170`
- Improvement `0.004970` MAE (`5.054%`)
- Paired CI `[-0.005121, -0.004820]`
- Worst horizon-variable regression: horizon `10`, variable `1`, delta `+0.000012`

Small predefined ETTh2 router-train-only selection:

- Selected config: `both_global_decay0.99_cap0.1_marginbp200_warm96`
- MAE `0.093239`, MSE `0.034157`
- Improvement `0.005100` MAE (`5.186%`)
- Paired CI `[-0.005253, -0.004949]`
- Worst horizon-variable regression: horizon `7`, variable `1`, delta `+0.000729`

## Decision

The capped DLinear/ModernTCN specialist idea transfers positively to ETTh2 under the limited available-cache setup. This supports the specialist mechanism, but it is not proof that the exact ETTh1 primary model generalizes because the full ETTh2 artifact set is incomplete.

## Reproduce

```powershell
python experiments\multidataset_replication_costar\run_locked_replication.py
```

