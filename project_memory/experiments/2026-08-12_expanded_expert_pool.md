# Expanded Expert Pool COSTAR-TS

## Hypothesis

DLinear and ModernTCN are poor average members of the ensemble, but they may still help as small causal specialists when recent completed forecasts show they are better than the fixed-three horizon-variable baseline.

## Configuration

Frozen core baseline:

- `hybrid_chrono_hvema_lowrank1_decay0.95_temp0.1_alpha0.75`
- Core experts: `PatchTST`, `iTransformer`, `TimesNet`
- Baseline validation reproduction: MAE `0.363642 +/- 0.000014`, MSE `0.306712 +/- 0.000016`

Optional experts:

- `DLinear`
- `ModernTCN`

Expanded prediction:

```text
final_prediction =
    (1 - weight_D - weight_M) * base_prediction
    + weight_D * DLinear_prediction
    + weight_M * ModernTCN_prediction
```

Constraints:

- `weight_D >= 0`
- `weight_M >= 0`
- `weight_D + weight_M <= extra_weight_cap`
- no negative or unconstrained stacking weights

Grid selected on router-train chronological folds only:

- Evidence structure: global, variable, horizon x variable
- EMA decay: `0.95`, `0.97`, `0.98`, `0.99`
- Combined cap: `0.025`, `0.05`, `0.10`
- Required relative advantage: `0.0%`, `0.5%`, `1.0%`, `2.0%`
- Warm-up: `24`, `48`, `96`

One-standard-error simplicity rule was used within each scenario.

Selected validation configs:

- DLinear only: `dlinear_only_variable_decay0.95_cap0.1_marginbp200_warm96`
- ModernTCN only: `moderntcn_only_variable_decay0.95_cap0.05_marginbp200_warm96`
- Both: `both_variable_decay0.95_cap0.1_marginbp200_warm96`

## Dataset / Split

Router-train `20-60%` was used for selection. Router validation `60-80%` was evaluated once after selection.

No test cache was loaded or evaluated.

## Command

```powershell
python experiments\expanded_expert_pool_costar\run_expanded_expert_pool.py --device cuda
```

## Router-Train Diagnostics

Window-level optional expert win rates against the fixed-three HV baseline on router-train:

- DLinear beat baseline on `28.47%` of windows, but mean MAE was worse by `+0.02946`.
- ModernTCN beat baseline on `13.04%` of windows, but mean MAE was worse by `+0.20881`.

This supports the backup-specialist framing: neither optional expert should be an equal ensemble member.

Router-train fold selection:

- DLinear-only selected config improved by `0.000129` MAE with `3/4` fold wins.
- ModernTCN-only selected config improved by `0.000100` MAE with `3/4` fold wins.
- Both selected config improved by `0.000315` MAE with `3/4` fold wins.

## Validation Results

| Method | Seeds | Validation MAE | Validation MSE | Improvement vs `0.363642` | Aggregate paired CI |
|---|---:|---:|---:|---:|---|
| Fixed-three HV baseline | 5 | `0.363642 +/- 0.000014` | `0.306712 +/- 0.000016` | `0.000000` | n/a |
| DLinear only | 5 | `0.363510 +/- 0.000014` | `0.306557 +/- 0.000016` | `0.000131` | `[-0.000144, -0.000119]` |
| ModernTCN only | 5 | `0.363435 +/- 0.000014` | `0.306452 +/- 0.000016` | `0.000206` | `[-0.000219, -0.000194]` |
| Both optional experts | 5 | `0.363112 +/- 0.000013` | `0.306057 +/- 0.000016` | `0.000529` | `[-0.000557, -0.000502]` |
| Equal five-expert reference | n/a | `0.371099` | `0.311582` | worse | n/a |

`expanded_both` also beats the previous current ridge residual result (`0.363301`) by `0.000189` MAE with paired CI `[-0.000233, -0.000143]`.

## Activation Diagnostics

For `expanded_both`:

- DLinear activation rate: `74.16%`
- DLinear average weight: `0.00738`
- DLinear max window-mean weight: `0.03649`
- DLinear active help rate: `63.54%`
- DLinear active mean delta: `-0.000606`
- DLinear mean activation duration: `41.48` windows
- DLinear turnover rate: `3.54%`
- ModernTCN activation rate: `50.58%`
- ModernTCN average weight: `0.00554`
- ModernTCN max window-mean weight: `0.03571`
- ModernTCN active help rate: `65.91%`
- ModernTCN active mean delta: `-0.000959`
- ModernTCN mean activation duration: `25.05` windows
- ModernTCN turnover rate: `4.00%`

Variable-specific activation for `expanded_both`:

| Variable | DLinear act. | DLinear avg w | ModernTCN act. | ModernTCN avg w |
|---:|---:|---:|---:|---:|
| `0` | `13.13%` | `0.00467` | `19.44%` | `0.00740` |
| `1` | `17.26%` | `0.00469` | `10.94%` | `0.00468` |
| `2` | `14.29%` | `0.00522` | `21.05%` | `0.00841` |
| `3` | `16.57%` | `0.00476` | `9.66%` | `0.00405` |
| `4` | `26.61%` | `0.00801` | `5.62%` | `0.00251` |
| `5` | `28.31%` | `0.00857` | `14.81%` | `0.00650` |
| `6` | `40.32%` | `0.01575` | `12.07%` | `0.00526` |

## Per-Axis Diagnostics

`expanded_both` improved every horizon and every variable on average.

Worst average horizon-variable regression:

- horizon `9`, variable `1`, delta `+0.000138` MAE.

This is below the major-regression threshold used for promotion.

## Decision

Promote the expanded-pool specialist layer as the new best validation result. Do not replace the core fixed-three baseline internally; keep DLinear and ModernTCN as capped optional specialists only.

## Relevant Files

- `experiments/expanded_expert_pool_costar/run_expanded_expert_pool.py`
- `experiments/expanded_expert_pool_costar/final_report.json`
- `experiments/expanded_expert_pool_costar/expanded_pool_report.md`
- `experiments/expanded_expert_pool_costar/router_train_fold_leaderboard.csv`
- `experiments/expanded_expert_pool_costar/validation_per_seed_results.csv`
- `experiments/expanded_expert_pool_costar/validation_activation_traces.csv`
- `experiments/expanded_expert_pool_costar/activation_summary.csv`
- `tests/test_expanded_expert_pool_costar.py`
