# Oracle Weight Tournament

## Hypothesis

Train-only oracle convex weights over the strong fixed-3 experts can provide better supervision than noisy marginal utility ranking.

## Configuration

Families tested:

- direct oracle-weight distillation
- prototype oracle weights
- prototype + residual
- horizon-specific weighting
- retrieval
- online variants

Five-seed finalist used prototype-residual oracle distillation.

## Dataset / Split

ETTh1 router train `20-60%`, router validation `60-80%`. Validation targets were not used to build train teachers. Test not used.

## Baselines

- Dynamic fixed-3: MAE `0.366342`.
- Equal fixed-3: MAE `0.367265`.

## Implementation

Relevant script:

- `experiments/oracle_weight_tournament/run_tournament.py`

## Commands

Reproduction commands from previous run:

```powershell
python experiments\oracle_weight_tournament\run_tournament.py --phase phase2 --time-budget-hours 6 --device cuda
python experiments\oracle_weight_tournament\run_tournament.py --phase finalists --time-budget-hours 6 --device cuda
python experiments\oracle_weight_tournament\run_tournament.py --phase ablation --time-budget-hours 1 --device cuda
```

## Results

Best five-seed finalist:

- `final_phase2_protores_lam0.01_k16_scale0.3_rw0.001`
- MAE `0.366028 +/- 0.000242`
- MSE `0.308755 +/- 0.000343`
- improvement vs dynamic fixed-3: `0.000314` MAE
- wins vs dynamic fixed-3: `4/5`
- paired bootstrap CI vs dynamic fixed-3: `[-0.000432, -0.000199]`

Ablation insight:

- teacher-only seed7 MAE `0.365734`
- forecast-loss-only seed7 MAE `0.366631`

## Interpretation

Oracle teacher distillation helped, but the gain remained small. Teacher signal mattered more than direct forecast loss.

## Decision

Keep oracle teacher/prototype results as a useful building block, but current best direction is chronological and horizon-variable adaptation.

## Relevant Files

- `experiments/oracle_weight_tournament/final_report.json`
- `experiments/oracle_weight_tournament/finalist_five_seed_summary.csv`
- `experiments/oracle_weight_tournament/ablation_summary.csv`
