# Sequential Utility Ranking

## Hypothesis

Sequential COSTAR can improve forecasting by querying experts one at a time according to marginal utility.

## Configuration

Main verified objective:

- `utility_pairwise_weighted`
- five seeds: `7, 11, 13, 17, 19`
- frozen experts: `DLinear`, `PatchTST`, `iTransformer`, `TimesNet`, `ModernTCN`

Also tested STOP-aware listwise variants.

## Dataset / Split

ETTh1 router train `20-60%`, router validation `60-80%`. Test not used.

## Baselines

- Fixed-3 equal: MAE `0.367265`.
- Dynamic fixed-3: MAE `0.366342`.

## Implementation

Relevant scripts:

- `scripts/train_sequential_costarts_utility_ranking.py`
- `scripts/sequential_costarts_model.py`
- `scripts/sequential_costarts_transformer_model.py`

## Commands

Exact original commands not recovered.

## Results

Weighted pairwise:

- MAE `0.368074 +/- 0.000078`
- MSE `0.310607 +/- 0.000483`
- average queries `3.944`
- top-1 utility accuracy `32.28%`
- top-2 utility coverage `63.76%`
- mean regret `0.11046`

STOP-aware listwise:

- MAE `0.375253 +/- 0.000874`
- average queries `1.0`
- collapsed to stopping after one forced query.

Full-sequence Transformer seed7 variants:

- current seed7 baseline: MAE `0.368062`
- history-only Transformer: MAE `0.368225`
- history + ensemble: MAE `0.368169`
- full forecast-state Transformer: MAE `0.368411`

## Interpretation

Ranking objectives and larger sequence encoders did not beat fixed-3. STOP-aware listwise over-stopped.

## Decision

Do not prioritize ranking-objective-only sequential routing unless the signal source changes.

## Relevant Files

- `results/router_summary/costarts_walkforward/utility_ranking_weighted_pairwise/summary.json`
- `results/router_summary/costarts_walkforward/utility_ranking/summary.json`
- `results/router_summary/costarts_walkforward/transformer_router_seed7_comparison.json`
