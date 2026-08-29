# Structured Forecast Repair

## Integrity

ETTh1, ETTm1, Weather, and Electricity passed the target-free integrity checks: no test cache loaded, frozen checkpoint parameters unchanged, cached forecasts reproduced within small numerical tolerance, finite features, chronological purged folds, exact target-corruption invariance, and deterministic repair regeneration. ETTh2 passed the no-test, frozen-parameter, finite-feature, chronological/purge, corruption, and repeatability checks, but forecast reproduction failed: the existing ETTh2 runtime differs materially from the cached forecasts. ETTh2 results must therefore be treated as integrity-unresolved until that cache/runtime discrepancy is reconciled.

## Core result

The primary metric is Ridge competence prediction MAE on relative expert error `z`. RepairGeometry improved over Passive on ETTh1 and ETTh2, but not ETTm1, Weather, or Electricity. REP was a strong control on Electricity. Block-24 support is recorded in `dependence_tests.csv`; it is favorable for RepairGeometry on ETTh1, ETTh2, ETTm1, and Electricity, but not Weather for the full combined arm.

## Expert specificity

Within-window expert shuffling materially reduced RepairGeometry performance on ETTh2, ETTm1, and Electricity, while ETTh1 and Weather showed only small changes. The evidence is therefore not uniformly expert-specific.

## Ablation conclusion

RepairGeometry is not a consistent incremental signal beyond passive features and controls across the five datasets. Scalar RepairCost is not uniformly better than raw violations or geometry. The mechanism is not established as more than historical representativeness or disagreement, and the mixed shuffle result does not support a robust expert-specific claim.

## Final classification

`WEAK_OR_AMBIGUOUS`

## Decision

Do not proceed to test-set evaluation or router integration. The validation evidence is mixed and fails the preregistered cross-dataset strong-signal thresholds.
