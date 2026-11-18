# Counterfactual Forecast Revision (CFR)

Date: 2026-08-27

Status: Completed, development/mechanism study only

## Question

When a frozen forecasting expert is shown a controlled hypothetical realization
of the first three future steps, does its revision of the remaining forecast
reveal expert-specific instance-level conditional competence beyond passive
history, forecast, and expert-disagreement features?

## Protocol

- Datasets: `ExchangeRate`, `Traffic`, `BeijingAirQuality`, `ETTm2`.
- Label: `DEVELOPMENT / MECHANISM STUDY`.
- Splits used: `router_train`, `router_val` only.
- Test split: not loaded or scored.
- Frozen core: exact `fhv.LOADERS[dataset]().core_names`.
- CFR intervention: horizon `12`, prefix `k=3`, scale `1.0`.
- Prefixes: self, positive one train-derived robust residual scale, negative one train-derived robust residual scale.
- Re-query history: `torch.cat([x[3:], prefix], dim=0)`.
- Tail alignment: compare `r[:9]` to original `y[3:12]`.
- Features: 9 mechanistic CFR features per expert (`self_revision`, plus/minus response, asymmetry, gain, two directional diagnostics, symmetric response magnitude, curvature magnitude).
- Models: `StandardScaler` fitted on training rows only plus `Ridge(alpha=1.0)`.
- OOF: reused V2 purged chronological folds; fold-specific surprise scales estimated from fold training windows only.
- Router-val: full legal router_train only for priors, surprise scales, and Ridge fitting.

## Result

Final predeclared classification:

`CFR_SIGNAL_BUT_REDUNDANT`

Interpretation:

CFR contains some competence-associated signal and several Passive+CFR point
improvements, but it fails the mandatory direct Passive-residual criterion on
3/4 datasets. This is not strong incremental model-specific evidence and is
not enough to freeze CFR for untouched-dataset testing.

## Key Router-Val Results

Passive vs best passive-plus CFR variant by MAE:

| Dataset | Passive MAE | Best Passive+CFR MAE | Delta |
|---|---:|---:|---:|
| ExchangeRate | `0.045411` | `0.043796` (`PassivePlusCFR`) | `-0.001615` |
| Traffic | `0.050686` | `0.047863` (`PassivePlusCFR`) | `-0.002823` |
| BeijingAirQuality | `0.104370` | `0.103896` (`PassivePlusRelativeCFR`) | `-0.000474` |
| ETTm2 | `0.047096` | `0.047084` (`PassivePlusRelativeCFR`) | `-0.000012` |

Primary block-24 support:

- `PassivePlusCFR_vs_Passive`: supported on `ExchangeRate` and `Traffic`, but significantly regressed on `BeijingAirQuality` and `ETTm2`.
- `PassivePlusRelativeCFR_vs_Passive`: supported on `ExchangeRate` and `BeijingAirQuality`; flat/unsupported on `Traffic` and `ETTm2`.
- Correct mapping vs shuffled was mixed: `CFR` beat `ShuffledCFR` on `Traffic`, `BeijingAirQuality`, and `ETTm2`, but not `ExchangeRate`; `RelativeCFR` was often worse than shuffled except on `ETTm2`.

Passive-residual prediction, router_val positive R2:

- `ExchangeRate`: no (`CFR` R2 `-0.0619`, `RelativeCFR` R2 `-0.0251`).
- `Traffic`: no (`CFR` R2 `-0.1430`, `RelativeCFR` R2 `-0.1896`).
- `BeijingAirQuality`: no (`CFR` R2 `-0.0425`, `RelativeCFR` R2 `-0.0226`).
- `ETTm2`: yes but small (`CFR` R2 `0.0197`, `RelativeCFR` R2 `0.0124`).

## Integrity

All integrity checks passed:

- Checkpoint SHA256 unchanged before/after.
- All experts remained frozen.
- No optimizer received expert parameters.
- Test split was not loaded.
- Router-val targets were never used for training or scale estimation.
- CFR feature construction was target-free for evaluated windows.
- Target corruption left CFR features unchanged exactly (`max_abs_diff = 0.0`).
- OOF purge assertions passed.
- Fold-specific surprise scales used only each fold's legal training portion.
- Absolute-horizon alignment assertion passed.
- CFR regeneration and shuffled control were deterministic.
- Same feature formulas were used on train and validation.

## Artifacts

- `experiments/behavioral_competence/counterfactual_forecast_revision/run_counterfactual_forecast_revision.py`
- `experiments/behavioral_competence/counterfactual_forecast_revision/report.md`
- `experiments/behavioral_competence/counterfactual_forecast_revision/validation_results.json`
- `experiments/behavioral_competence/counterfactual_forecast_revision/integrity_checks.json`
- `experiments/behavioral_competence/counterfactual_forecast_revision/dependence_tests.csv`
- `experiments/behavioral_competence/counterfactual_forecast_revision/passive_incremental_results.csv`
- `experiments/behavioral_competence/counterfactual_forecast_revision/passive_residual_results.csv`
- `experiments/behavioral_competence/counterfactual_forecast_revision/expert_specificity_results.csv`
- `experiments/behavioral_competence/counterfactual_forecast_revision/per_expert_correlations.csv`
- `experiments/behavioral_competence/counterfactual_forecast_revision/per_window_scores/`
- `experiments/behavioral_competence/counterfactual_forecast_revision/feature_cache/`

## Decision

Do not freeze CFR as a strong untouched-dataset candidate yet. If revisited,
the next experiment should be explicitly post-hoc/hypothesis-generating unless
a new mechanism is frozen before any new validation datasets are inspected.
