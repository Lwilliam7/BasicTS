# Causal Residual Correction COSTAR-TS

## Hypothesis

The remaining validation gap might come from systematic residual error in the current best horizon-variable adaptive predictor, not only from expert weighting.

## Configuration

Frozen baseline:

- `hybrid_chrono_hvema_lowrank1_decay0.95_temp0.1_alpha0.75`
- Baseline validation reproduction: MAE `0.363642 +/- 0.000014`, MSE `0.306712 +/- 0.000016`

Experiment 1:

- Causal signed residual bias correction.
- Structures swept: global, horizon, variable, horizon x variable.
- EMA decay: `0.90, 0.95, 0.97, 0.98, 0.99`.
- Alpha: `0.10, 0.25, 0.50, 0.75, 1.00`.
- Clipping: `0.5, 1.0, 2.0, 3.0` train residual std multiples plus unclipped.
- Warm-up: `0, 12, 24, 48, 96`.

Selected on router-train chronological folds:

- `variable_decay0.99_alpha0.1_clip0.5_warm0`

Experiment 2:

- Conservative residual corrector using baseline prediction, three expert forecasts, pairwise disagreement, causal residual stats, horizon/variable identity, and causal multi-scale history summaries at `96/192/336/720`.
- Ridge first; MLP only because ridge showed useful router-train fold signal.
- MLP seeds: `7, 11, 13, 17, 19`.

Selected ridge config:

- `ridge1_alpha0.1_clip0.25_full`

## Dataset / Split

ETTh1 router-train `20-60%`, validation `60-80%`.

No test cache was loaded or evaluated.

## Commands

```powershell
python experiments\residual_correction_costar\run_residual_correction_experiments.py --device cuda
```

Focused causality tests were added in `tests/test_residual_correction_costar.py`, but this environment does not have `pytest` installed, so they could not be run with `python -m pytest`. A direct Python smoke check of the residual-release rule passed.

## Results

| Method | Seeds | Validation MAE | Validation MSE | Improvement vs `0.363642` | Aggregate paired bootstrap CI |
|---|---:|---:|---:|---:|---|
| Baseline current best | 5 | `0.363642 +/- 0.000014` | `0.306712 +/- 0.000016` | `0.000000` | n/a |
| Experiment 1 bias | 5 | `0.363591 +/- 0.000012` | `0.306650 +/- 0.000014` | `0.000051` | `[-0.000098, -0.000004]` |
| Experiment 2 ridge | 5 | `0.363301 +/- 0.000015` | `0.306286 +/- 0.000017` | `0.000341` | `[-0.000378, -0.000305]` |
| Experiment 2 MLP | 5 | `0.363318 +/- 0.000109` | `0.306607 +/- 0.000141` | `0.000324` | `[-0.000359, -0.000290]` |

Ridge is the best result, improves the current baseline with CI excluding zero, and is more stable than the MLP.

Target status:

- Strong target `<= 0.3619`: not reached.
- Exceptional target `<= 0.3600`: not reached.

## Diagnostics

Ridge correction:

- Mean applied normalized delta: `0.004691`.
- Clip frequency: `14.04%`.
- Worst average horizon-variable regression: horizon `0`, variable `5`, delta `+0.000130` MAE.
- Per-horizon MAE improved at every horizon on average.
- Per-variable average changed by: variable `0` `-0.000736`, variable `1` `-0.000153`, variable `2` `-0.001135`, variable `3` `-0.000166`, variable `4` `+0.000023`, variable `5` `+0.000044`, variable `6` `-0.000261`.

Experiment 1 caveat:

- The selected bias correction did not improve any router-train fold (`fold_wins=0`) even though it improved validation slightly. Treat it as a weak/fragile result, not a durable direction by itself.

MLP caveat:

- MLP mean MAE was close to ridge but less stable across seeds and had larger local regressions on variable `4`.

## Leakage Checks

- Online residual state releases only after `old_start + horizon <= current_start`.
- Long-history feature summaries end before the forecast start.
- Validation target end remains within `[8640, 11520)`.
- Test cache was not loaded.

## Relevant Files

- `experiments/residual_correction_costar/run_residual_correction_experiments.py`
- `experiments/residual_correction_costar/final_report.json`
- `experiments/residual_correction_costar/experiment_report.md`
- `experiments/residual_correction_costar/validation_seed_summary.csv`
- `experiments/residual_correction_costar/aggregate_bootstrap_ci.csv`
- `experiments/residual_correction_costar/per_axis_mae_aggregate.csv`
- `experiments/residual_correction_costar/per_horizon_variable_mae_aggregate.csv`
