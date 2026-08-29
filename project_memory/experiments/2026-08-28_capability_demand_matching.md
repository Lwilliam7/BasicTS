# Natural Capability-Demand Matching

Date: 2026-08-28

Status: completed validation-only

Artifacts:

- `experiments/behavioral_competence/capability_demand_matching/run_capability_demand_matching.py`
- `experiments/behavioral_competence/capability_demand_matching/report.md`
- `experiments/behavioral_competence/capability_demand_matching/results.json`
- `experiments/behavioral_competence/capability_demand_matching/integrity_checks.json`
- `experiments/behavioral_competence/capability_demand_matching/etth2_integrity_audit.json`

## Question

Can frozen forecast experts be routed by matching a window's natural demand fingerprint to expert capability profiles learned from earlier router_train variation, without perturbations or generic learned embeddings?

## Protocol

- Datasets: `ETTh1`, `ETTh2`, `ETTm1`, `Weather`, `Electricity`.
- Split: router_train/router_val only; no test cache or test target access.
- Horizon: `12`; input length: `96`.
- Expert pool: cached order `DLinear`, `PatchTST`, `iTransformer`, `TimesNet`, `ModernTCN`.
- Evaluated core: the frozen train-selected three-expert core from `experiments.frozen_hv_costar.run_frozen_hv_costar.LOADERS`.
- Target: relative competence `z[t,k] = expert_error[t,k] - mean_j expert_error[t,j]`; lower is better.
- Demand axes from input history only: trend, seasonality, frequency, volatility, shift, crossvar.
- OOF: four chronological router_train folds after 20% warmup; fitting windows must satisfy `old_start + horizon <= current_origin`.
- Capability profiles:
  - LOW/MED/HIGH regime tables using train-prefix q33/q67 bins, shrunk toward expert global means.
  - Per-axis quadratic Ridge capability curves.
  - Primary score is fixed `0.5 * regime + 0.5 * continuous`.
- Baselines/controls: GlobalPrior, Passive ABC Ridge, FAME-style one-sided Ridge from demand axes, Demand+ExpertID Ridge, expert-profile shuffle, semantic-axis shuffle, window shuffle diagnostic.
- Dependence-aware tests: block-24 and every-12th phase bootstraps.

## ETTh2 Integrity Resolution

The prior Structured Forecast Repair run found large ETTh2 cached-forecast/runtime reproduction differences. This audit resolved the discrepancy:

- ETTh2 cache histories/targets/predictions are stored in DLinear scaler-normalized units, with metrics using `std=ones`.
- Calling the model directly on normalized cache histories reproduced cached forecasts within `<= 9.54e-07`.
- De-normalizing histories first and then using the runtime wrapper also reproduced cached forecasts within `<= 9.54e-07`.
- Passing already-normalized cache histories into the runtime wrapper caused large differences, matching the prior failure mode.

Recorded status: `ETTH2_INTEGRITY_RESOLVED`.

## Result

Final classification: `CAPABILITY_SIGNAL_BUT_NO_MATCHING_GAIN`.

Competence MAE on router_val:

| Dataset | Global | Passive | FAME-style | Capability | Expert shuffle | Axis shuffle |
|---|---:|---:|---:|---:|---:|---:|
| ETTh1 | 0.036173 | 0.033263 | 0.037434 | 0.035850 | 0.045754 | 0.036378 |
| ETTh2 | 0.023789 | 0.023971 | 0.026848 | 0.024121 | 0.036248 | 0.029687 |
| ETTm1 | 0.032954 | 0.030820 | 0.035093 | 0.033159 | 0.035076 | 0.035339 |
| Weather | 0.019614 | 0.017493 | 0.019140 | 0.019518 | 0.023500 | 0.032378 |
| Electricity | 0.017378 | 0.016233 | 0.017445 | 0.017006 | 0.034447 | 0.148623 |

CapabilityMatch showed real competence association and consistently beat the expert-profile shuffle, with strong axis-shuffle degradation on several datasets. However, it did not consistently beat Passive ABC, and it beat the FAME-style direct demand baseline only on ETTh1, ETTh2, ETTm1, and Electricity while losing on Weather. Routing proxy gains were also mixed and not better than the strongest simple alternatives.

## Integrity

Passed:

- No test cache loaded.
- Cache roles, shapes, expert order, horizon, and chronological starts checked.
- Checkpoint hashes recorded and unchanged.
- Demand fingerprints finite and deterministic.
- Capability profiles and LOW/MED/HIGH bins constructed from legal train prefixes only.
- Router_val target corruption left demand features, passive features, capability predictions, and passive predictions unchanged exactly.
- ETTh2 status resolved as above.

## Decision

Do not promote this capability-demand matcher to router integration or test evaluation. The mechanism is useful as evidence that semantic capability profiles carry signal, but the explicit matching formulation is not yet a robust incremental improvement over passive or direct demand baselines.
