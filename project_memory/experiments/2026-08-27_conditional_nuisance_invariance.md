# Conditional Nuisance Invariance LearnedProbe Audit

Date: 2026-08-27

Status: Completed validation-only

Location:

- `experiments/behavioral_competence/conditional_nuisance_invariance/`
- Main runner: `experiments/behavioral_competence/conditional_nuisance_invariance/run_cni.py`
- Results: `experiments/behavioral_competence/conditional_nuisance_invariance/results/`

## Question

Does the canonical expert-conditioned LearnedProbe active response `P` contain expert competence information after passive features `X` and explicit nuisance features `N` are controlled?

## Protocol

- Datasets: `ETTh1`, `ETTh2`, `ETTm1`, `Weather`, `Electricity`.
- Development/router-val only. No test cache was loaded or scored.
- Reused canonical mechanism implementation from `experiments/behavioral_competence/expert_conditioned_probe_mechanism/run_experiment.py`.
- Regenerated missing OOF active/matched features in a new CNI cache because prior mechanism-ablation artifacts did not save OOF `P` and raw deltas.
- Used strict purged chronological OOF via the V2-compatible `compute_legal_and_common` protocol.
- `X`: existing passive A+B+C 15 features.
- `P`: canonical LearnedProbe six response statistics.
- `N`: 12 nuisance features covering history scale/mean/volatility/trend/seasonality/time, forecast magnitude/variance/disagreement, and perturbation norm/recency/trend alignment.
- Models included Passive, ProbeOnly, Passive+Probe, Passive+Nuisance, Passive+Nuisance+Probe, residualized Probe variants, MatchedPassive, shuffled Probe, and wrong-expert Probe.
- Residualized `P` used chronological cross-fitted `StandardScaler + Ridge(alpha=1.0)` without using competence targets.

## Result

Final predeclared classification: `MIXED_CNI`.

Primary pairwise competence deltas:

| Dataset | Delta(Passive+Nuisance+Probe - Passive+Nuisance) | Delta(Passive+ResidualProbe - Passive) | True Probe - Shuffled | True Probe - Wrong Expert |
|---|---:|---:|---:|---:|
| ETTh1 | `+0.020075` | `+0.006251` | `-0.002284` | `-0.003246` |
| ETTh2 | `+0.006525` | `-0.020663` | `-0.094617` | `-0.153888` |
| ETTm1 | `+0.000117` | `-0.007915` | `-0.009054` | `-0.006396` |
| Weather | `-0.011567` | `-0.020098` | `-0.065088` | `-0.005911` |
| Electricity | `+0.088410` | `+0.010542` | `+0.022636` | `+0.000647` |

Block-24 `Passive+Nuisance+Probe` vs `Passive+Nuisance` pairwise support:

- `Electricity` was significant and positive for active information.
- `ETTh1`, `ETTh2`, `ETTm1`, and `Weather` were not significant.

Routing MAE was mixed:

- `Passive+Nuisance+Probe` improved routing MAE over `Passive+Nuisance` on `ETTh1`, `ETTh2`, and `Electricity`.
- It was worse on `ETTm1` and `Weather`.

## Integrity

All dataset integrity checks passed:

- No test cache loaded.
- No test metrics computed.
- Checkpoint hashes unchanged.
- Expert parameters remained frozen.
- Router-val targets not used in feature construction or residualization.
- Target corruption left recomputed `X`, `N`, `P`, residualized `P`, matched features, final scores, routing weights, and final predictions unchanged with max absolute diff `0.0`.
- Router-train to router-val observability held on every dataset.

## Interpretation

The active LearnedProbe signal does not fail uniformly after nuisance controls, because `Electricity` shows strong positive evidence and some smaller improvements appear elsewhere. It also does not survive CNI robustly: most datasets fail negative controls, environment transfer, or block-24 support. Treat the result as mechanism evidence that is dataset-dependent and not yet a strong router-integration candidate.

