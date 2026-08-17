# ETTh2 Router Runs And Transfer Screen

## Hypothesis

ETTh2 may provide useful pretraining or an alternate setting where sequential COSTAR performs differently.

## Configuration

ETTh2 clean caches under `cache/costarts_fresh/ETTh2_96_12/`.

Verified sequential run:

- Sequential COSTAR-TS Full `utility_pairwise_weighted`
- seeds `7, 11, 13, 17, 19`

Transfer screen:

- dynamic residual fixed-3 scratch vs ETTh2 encoder/full transfer
- sequential encoder/full transfer

## Dataset / Split

ETTh2 validation split as stored in `results/router_summary/costarts_fresh/ETTh2_96_12/sequential_utility_ranking_combined/summary.json`.

## Baselines

ETTh2 best fixed by size:

- best fixed 1: `DLinear`, MAE `0.280957`
- best fixed 2: `DLinear+ModernTCN`, MAE `0.275229`
- best fixed 3: `DLinear+TimesNet+ModernTCN`, MAE `0.276644`
- best fixed 4: `DLinear+PatchTST+TimesNet+ModernTCN`, MAE `0.277681`

## Implementation

Relevant results:

- `results/router_summary/costarts_fresh/ETTh2_96_12/sequential_utility_ranking_combined/summary.json`
- `results/router_summary/costarts_transfer/etth2_transfer_screen_summary.json`

## Commands

Exact original commands not recovered.

## Results

ETTh2 sequential:

- MAE `0.277681`
- MSE `0.168231`
- avg queries `4.0`
- effectively matched fixed-4 and did not beat fixed-2.

ETTh2 transfer to ETTh1:

- scratch dynamic residual seed7 MAE `0.366098`
- ETTh2 encoder transfer MAE `0.366526`
- ETTh2 full transfer MAE `0.366820`
- sequential transfer variants around `0.3694-0.3695`

## Interpretation

ETTh2 sequential routing tended to query four experts and did not beat the best fixed-2. Simple ETTh2 transfer degraded ETTh1 performance.

## Decision

Do not pursue ETTh2 transfer as implemented without a new hypothesis.

## Relevant Files

- `results/router_summary/costarts_fresh/ETTh2_96_12/sequential_utility_ranking_combined/summary.json`
- `results/router_summary/costarts_transfer/etth2_transfer_screen_summary.json`
