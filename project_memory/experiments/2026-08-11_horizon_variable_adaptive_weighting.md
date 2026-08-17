# Horizon x Variable Adaptive Weighting

## Hypothesis

Global expert weights leave performance on the table because expert strengths vary by forecast horizon and variable.

## Configuration

Tested:

- global weights
- horizon-only weights
- variable-only weights
- horizon x variable weights
- low-rank horizon x variable weights
- RLS / online ridge stacking with intercepts
- five-seed hybrid comparison

Winning setting against chronological baseline:

- `hybrid_chrono_hvema_lowrank1_decay0.95_temp0.1_alpha0.75`
- low-rank rank `1`
- decay `0.95`
- temperature `0.1`
- blend alpha `0.75`

## Dataset / Split

ETTh1 validation `60-80%`, strict chronological updates. Test not used.

## Baselines

- Chronological adaptive COSTAR: MAE `0.365534`, MSE `0.308340`.
- Target from prompt: MAE `<= 0.3619`.

## Implementation

Relevant script:

- `experiments/horizon_variable_adaptive_costar/run_hv_adaptive_costar.py`

## Commands

The broad screen timed out after partial completion but saved results. Finalist reproduction:

```powershell
python experiments\horizon_variable_adaptive_costar\run_hv_adaptive_costar.py --phase finalists --top-k 8 --device cuda
```

## Results

Best against chronological current:

- MAE `0.363642 +/- 0.000014`
- MSE `0.306712 +/- 0.000016`
- improvement vs chronological best: `0.001892` MAE
- wins `5/5`
- paired bootstrap CI: `[-0.002084, -0.001704]`
- target `0.3619`: not reached

Specialization:

- biggest variable gain: variable `4`, improvement about `0.01097` MAE
- small losses: variables `1` and `6`
- biggest horizon gains: horizons `11`, `8`, `0`, `7`, `10`

## Interpretation

Horizon x variable structure is a major source of recoverable validation performance. Rank-1 structure was enough to capture most of the gain.

## Decision

This is the current best validation direction.

## Relevant Files

- `experiments/horizon_variable_adaptive_costar/final_report.json`
- `experiments/horizon_variable_adaptive_costar/chrono_baseline_comparison.csv`
- `experiments/horizon_variable_adaptive_costar/chrono_hv_winner_specialization_agg.csv`
