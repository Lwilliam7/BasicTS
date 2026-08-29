# Structured Forecast Repair / Repair Geometry

Date: 2026-08-28

## Protocol

Validation-only study on ETTh1, ETTh2, ETTm1, Weather, and Electricity. It used only router-train and router-val frozen forecast caches, horizon 12, chronological purged OOF calibration, and frozen router-val calibration. Repair families were temporal continuity, observable lag-24 seasonality when active, cross-variable PCA, and multi-horizon PCA. Controls included Passive, disagreement, REP, raw violations, scalar RepairCost, within-window expert shuffling, and block-24 bootstrap.

## Result

Final classification: `WEAK_OR_AMBIGUOUS`. RepairGeometry improved relative-competence prediction over Passive on ETTh1 and ETTh2, but regressed on ETTm1, Weather, and Electricity. REP was substantially useful on Electricity. Within-window shuffling materially reduced performance on ETTh2, ETTm1, and Electricity but not consistently across datasets. RepairCost was not uniformly better than raw violations or RepairGeometry.

## Integrity and decision

Integrity passed for no-test access, frozen expert parameters, finite features, chronological/purged folds, target-corruption invariance, and deterministic regeneration. ETTh1, ETTm1, Weather, and Electricity reproduced cached forecasts within small numerical tolerance. ETTh2 had a material cached-forecast/runtime reproduction discrepancy and is integrity-unresolved. Do not run test evaluation or router integration for this mechanism.

Artifacts: `experiments/behavioral_competence/structured_forecast_repair/`.