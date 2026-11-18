# Raw Response Probe V3A Feasibility Audit

Date: 2026-08-25

Status: blocked before scientific evaluation.

## Intended Question

Holding the learned V2 intervention completely fixed, does retaining the full horizon-by-variable forecast response reveal complementary expert-competence information that was lost when V2 compressed the response into six handcrafted statistics?

## Required Frozen Inputs

V3A requires exact frozen V2 OOF raw forecast responses from the learned conditional probe:

```text
DeltaForecast[t,e,h,v] = F_e(X_t + delta_t)[h,v] - F_e(X_t)[h,v]
```

It also requires the same V2 perturbation, expert forecasts, windows, OOF folds, and conditional targets, while changing only the response representation.

## Finding

The completed V2 artifact set is insufficient to run the primary V3A OOF raw-response test without violating V3A's hard no-retraining rule.

Available in V2:

- Router-val learned deltas: `conditional_delta_val`.
- Router-val six-stat learned responses: `conditional_response_val`.
- OOF six-stat learned responses on common windows: `oof_conditional_response_common`.
- OOF common indices and conditional targets.

Missing from V2:

- OOF learned deltas.
- OOF full raw response tensors.
- Trained V2 generator checkpoints.

Because the trained OOF fold generators were not serialized, reconstructing the exact OOF raw responses would require rerunning `train_learned_shared_prefix`, which is generator retraining and violates the V3A prompt.

## Artifact Availability

| Dataset | Router-val learned delta | OOF learned six stats | OOF learned delta | OOF full raw response | V2 generator checkpoint |
|---|---:|---:|---:|---:|---:|
| ExchangeRate | true | true | false | false | false |
| Traffic | true | true | false | false | false |
| BeijingAirQuality | true | true | false | false | false |
| ETTm2 | true | true | false | false | false |

## Integrity

- Test set accessed: no.
- V2 perturbation generator retrained: no.
- Forecasting experts retrained: no.
- V2 result modified: no.
- Router trained: no.

## Artifacts

- `experiments/behavioral_competence/raw_response_probe_v3a/report.md`
- `experiments/behavioral_competence/raw_response_probe_v3a/method_manifest.json`
- `experiments/behavioral_competence/raw_response_probe_v3a/source_v2_manifest.json`
- `experiments/behavioral_competence/raw_response_probe_v3a/integrity_checks.csv`
- `experiments/behavioral_competence/raw_response_probe_v3a/raw_response_shape_diagnostics.csv`

## Decision

Do not report a V3A scientific classification from the current artifacts. If this question is still desired, run a separate V2-compatible artifact-generation experiment that freezes and saves OOF learned deltas/full raw responses or trained fold generator checkpoints before attempting V3A.
