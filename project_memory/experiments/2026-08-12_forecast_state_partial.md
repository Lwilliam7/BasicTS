# Forecast-State Partial Sequential Routing

## Hypothesis

Forecast-state sequential routing might still contain useful diagnostics, but prior sequential evidence suggested it was unlikely to beat fixed ensembles or chronological adaptation.

## Configuration

- Method: `forecast_state_partial_subset_stop_router`
- Ablation: `full`
- Seeds: `7, 11, 13, 17, 19`
- Training shape: one epoch per seed, matching the existing partial seed artifacts.
- Primary stop threshold: `0.5`
- Trajectory kinds: `oracle,random`
- Frozen experts: `DLinear`, `PatchTST`, `iTransformer`, `TimesNet`, `ModernTCN`

## Dataset / Split

ETTh1 router train `20-60%`, router validation `60-80%`.

No test data was used.

## Completion Note

The folder `results/router_summary/costarts_walkforward/forecast_state_partial/full/` originally had completed artifacts for seeds `7, 11, 13, 17`, while `seed_19` was empty. Completed `seed_19` with:

```powershell
python scripts/train_sequential_costarts_forecast_state.py --results-root results/router_summary/costarts_walkforward/forecast_state_partial --checkpoint-root checkpoints/costarts_walkforward/forecast_state_partial --ablation full --seeds 19 --max-epochs 1 --stop-thresholds 0.5 --device cpu
```

Then re-evaluated all five saved checkpoints at stop threshold `0.5` and regenerated the aggregate summary.

Metric consistency check:

- Summary metrics use checkpoint re-evaluation for all five seeds.
- Seed `11` checkpoint re-evaluation differs slightly from its original one-row training curve: MAE `0.371307522` vs `0.371294647`, delta `+0.000012875`; MSE `0.315089107` vs `0.315078348`, delta `+0.000010759`.
- Other seeds matched their training-curve validation metrics exactly at shown precision.
- This discrepancy is too small to affect the conclusion; both versions remain worse than fixed-3.

## Baselines

- Fixed-3 equal: MAE `0.367265`, MSE `0.310530`.
- Current best horizon-variable adaptive: MAE `0.363642`, MSE `0.306712`.

## Results

- MAE: `0.371078 +/- 0.000676`
- MSE: `0.315205 +/- 0.000776`
- Average queries: `3.114 +/- 0.271`
- Stop accuracy: `75.57% +/- 1.89`
- Top-1 next-query accuracy: `36.36% +/- 2.18`
- Marginal utility correlation: `0.4737 +/- 0.0565`

Compared with baselines:

- Worse than fixed-3 by `+0.003813` MAE.
- Worse than current best horizon-variable adaptive by `+0.007436` MAE.

## Interpretation

The diagnostic query/ranking signals are nonzero, but they do not translate into competitive validation forecasts. The one-epoch partial forecast-state router underperforms even the fixed-3 equal ensemble, so this folder should be considered closed out as a noncompetitive partial sequential-routing result.

## Decision

Do not continue this exact forecast-state partial sequential setup. Any future sequential routing work should require a materially new signal or objective, and should compare against fixed-3 plus the current horizon-variable adaptive best.

## Relevant Files

- `results/router_summary/costarts_walkforward/forecast_state_partial/full/summary.json`
- `results/router_summary/costarts_walkforward/forecast_state_partial/full/per_seed_results.csv`
- `checkpoints/costarts_walkforward/forecast_state_partial/full/seed_19/best_forecast_state_router.pt`
