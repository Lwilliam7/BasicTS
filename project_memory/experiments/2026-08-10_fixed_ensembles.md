# Fixed Ensemble Brute Force

## Hypothesis

Simple fixed equal-average expert subsets may be strong enough that adaptive routers must beat them at comparable compute.

## Configuration

Evaluated all fixed expert subsets of sizes 1 through 5 using equal-average aggregation.

## Dataset / Split

ETTh1 router validation `60-80%`, `2773` windows. Test not used.

## Baselines

Single experts and all fixed subsets from the five frozen experts.

## Implementation

Summary file: `results/router_summary/costarts_walkforward/fixed_ensembles/summary.json`.

## Commands

Exact original command not recovered.

## Results

Best by size:

- Best single: `iTransformer`, MAE `0.376550`, MSE `0.322095`.
- Best fixed 2: `PatchTST+iTransformer`, MAE `0.370154`, MSE `0.314509`.
- Best fixed 3: `PatchTST+iTransformer+TimesNet`, MAE `0.367265`, MSE `0.310530`.
- Best fixed 4: `DLinear+PatchTST+iTransformer+TimesNet`, MAE `0.368216`, MSE `0.310938`.
- All 5: MAE `0.371099`, MSE `0.311582`.

## Interpretation

The fixed-3 subset is a very strong baseline. Adding weaker experts can degrade equal-average performance.

## Decision

Always compare COSTAR routing/adaptation against fixed-3, not just individual experts.

## Relevant Files

- `results/router_summary/costarts_walkforward/fixed_ensembles/summary.json`
- `cache/costarts_walkforward/router_val_60_80_cache.pt`
