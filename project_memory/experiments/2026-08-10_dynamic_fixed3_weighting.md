# Dynamic Fixed-3 Weighting

## Hypothesis

A learned router can improve on equal fixed-3 by predicting dynamic convex weights over `PatchTST`, `iTransformer`, and `TimesNet`.

## Configuration

Five seeds: `7, 11, 13, 17, 19`.

## Dataset / Split

ETTh1 router train `20-60%`, router validation `60-80%`. Test not used.

## Baselines

- Equal fixed-3: MAE `0.367265`, MSE `0.310530`.

## Implementation

Relevant script:

- `scripts/train_costarts_fixed3_dynamic_weighting.py`

## Commands

Exact original command not recovered.

## Results

- MAE `0.366342 +/- 0.000223`
- MSE `0.309214 +/- 0.000237`
- improvement vs equal fixed-3: `0.000923` MAE

## Interpretation

Dynamic weighting can beat the fixed-3 baseline, but the margin is modest.

## Decision

Use dynamic fixed-3 as an important baseline for later adaptive methods.

## Relevant Files

- `results/router_summary/costarts_walkforward/fixed3_dynamic_weighting_5seed/summary.json`
