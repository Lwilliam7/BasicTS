# Chronological Adaptive COSTAR

## Hypothesis

Expert usefulness changes over validation time; causal online adaptation to recently observed expert errors should beat static routers.

## Configuration

Tested:

- EMA recent-error weighting
- Hedge / discounted Hedge
- rolling-window performance weighting
- hybrid blend with prototype-residual router
- online fine-tuning delta
- conservative detector hybrids

Winning five-seed setting:

- `hybrid_ema_decay0.97_temp0.1_alpha0.5`

## Dataset / Split

ETTh1 validation `60-80%`, strict chronological inference. Test not used.

Online leakage rule:

`old_start + horizon <= current_start`

## Baselines

- Oracle prototype-residual: MAE `0.366028`.
- Dynamic fixed-3: MAE `0.366342`.
- Equal fixed-3: MAE `0.367265`.

## Implementation

Relevant script:

- `experiments/chronological_adaptive_costar/run_chronological_adaptive_costar.py`

## Commands

```powershell
python experiments\chronological_adaptive_costar\run_chronological_adaptive_costar.py --phase all --top-online 4 --device cuda
```

## Results

Best five-seed finalist:

- MAE `0.365534 +/- 0.000112`
- MSE `0.308340 +/- 0.000146`
- improvement vs prototype-residual: `0.000494` MAE
- wins `5/5`
- paired bootstrap CI: `[-0.000568, -0.000420]`

Expert ranking changed 4 times across 6 validation blocks.

## Interpretation

Chronological adaptation directly addressed the observed train-to-validation shift.

## Decision

Chronological adaptation superseded static oracle prototype-residual as the best direction until horizon-variable adaptation improved further.

## Relevant Files

- `experiments/chronological_adaptive_costar/final_report.json`
- `experiments/chronological_adaptive_costar/finalist_summary.csv`
- `experiments/chronological_adaptive_costar/winner_block_vs_current_agg.csv`
