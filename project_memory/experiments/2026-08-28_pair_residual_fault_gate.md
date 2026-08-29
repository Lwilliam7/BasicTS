# Signed Pair Residual Fault Gate

Date: 2026-08-28

Status: completed validation-only

Artifacts:

- `experiments/behavioral_competence/pair_residual_fault_gate/run_pair_residual_fault_gate.py`
- `experiments/behavioral_competence/pair_residual_fault_gate/report.md`
- `experiments/behavioral_competence/pair_residual_fault_gate/results.json`
- `experiments/behavioral_competence/pair_residual_fault_gate/routing_results.csv`
- `experiments/behavioral_competence/pair_residual_fault_gate/fault_detector_results.csv`
- `experiments/behavioral_competence/pair_residual_fault_gate/dependence_tests.csv`
- `experiments/behavioral_competence/pair_residual_fault_gate/integrity_checks.json`

## Question

Can signed expert-to-expert forecast residual/parity patterns identify when a specific frozen forecasting expert is about to be a relative bust, and can a fault-isolation gate improve an existing train-only baseline by suppressing only high-risk experts?

## Protocol

- Datasets: `ETTh1`, `ETTh2`, `ETTm1`, `Weather`, `Electricity`.
- Split: router_train/router_val only; no test cache or test target access.
- Core: each dataset's already-selected K=3 frozen expert core from `experiments.frozen_hv_costar.run_frozen_hv_costar.LOADERS`.
- Baseline: strict train-only frozen HxV COSTAR weights over the selected K=3 core. Router_train OOF folds build weights only from earlier legal targets; router_val uses the all-router_train frozen weight tensor.
- Fault target: `regret_k = MAE_k - median(MAE_other_experts)`. Fault thresholds are q80/q90 of positive router_train regret, selected only by chronological OOF routing MAE.
- Features:
  - Passive: existing A+B+C window/forecast/disagreement features.
  - Parity: signed pair residual features `(forecast_i - forecast_j) / dataset_std` for every expert pair plus target-expert-oriented pairs, preserving signed mean, early/late means, absolute/max magnitude, horizon profile, and variable profile when manageable.
  - Passive+Parity.
  - Shuffled parity control: same architecture as Passive+Parity with parity features shuffled across windows inside train/eval splits.
  - Raw forecast control: Passive plus compressed raw expert forecast summaries instead of explicit pairwise parity.
- Detector: fixed logistic regression with train-only standardization and balanced class weights.
- Gate: `w_new[k] = w_base[k] * (1 - p_fault[k]) ** gamma`, renormalized over experts.
- Selection: q80/q90, gamma `[0.5, 1.0, 2.0]`, and intervention threshold `[none, 0.4, 0.6, 0.8]` chosen using router_train OOF only.

## Result

Final classification: `WEAK_OR_INCONSISTENT_PARITY_FAULT_SIGNAL`.

Router-val MAE:

| Dataset | Baseline | Passive | Parity | Passive+Parity | Shuffled | Raw Control |
|---|---:|---:|---:|---:|---:|---:|
| ETTh1 | 0.366022 | 0.366662 | 0.366943 | 0.367019 | 0.366670 | 0.366483 |
| ETTh2 | 0.276898 | 0.276275 | 0.276465 | 0.275640 | 0.275973 | 0.276318 |
| ETTm1 | 0.250690 | 0.251171 | 0.251457 | 0.251200 | 0.251191 | 0.251296 |
| Weather | 0.159818 | 0.160169 | 0.160172 | 0.160052 | 0.160127 | 0.159600 |
| Electricity | 0.215355 | 0.216783 | 0.215893 | 0.215883 | 0.216525 | 0.215735 |

ETTh2 is the supportive case: Passive+Parity improved over Baseline by `-0.001258` MAE and beat Passive, Shuffled Parity, and Raw Forecast Control by point estimate. However, ETTh1, ETTm1, and Electricity regressed versus Baseline, while Weather's best improvement came from Raw Forecast Control rather than parity. The cross-dataset pattern does not support a robust fault-isolation mechanism.

Fault detectors often identified relative bust labels with high AUC, especially Passive on ETTh1/ETTm1/Weather and Passive+Parity on ETTh2, but classifier discrimination did not translate into stable routing gains. This reinforces that the gating intervention, not only bust detection, is the bottleneck.

## Integrity

Passed:

- No test cache loaded.
- Every cache/checkpoint path is refused if it contains `test`.
- Cache roles, shapes, expert order, horizon, and chronological starts checked.
- Checkpoint hashes recorded before feature work and unchanged after.
- Router_train detector predictions are chronological OOF with horizon-12 purge.
- Fault thresholds, gamma, and intervention thresholds selected from router_train OOF only.
- Router_val target corruption leaves passive, parity, and raw features unchanged exactly.
- All generated features finite.

## Decision

Do not promote the current pair-residual fault gate to router integration or test evaluation. Keep ETTh2 as a hypothesis-generating positive case, but require a future mechanism to beat Passive and Raw Forecast Control consistently before treating parity residuals as a robust fault-isolation signal.
