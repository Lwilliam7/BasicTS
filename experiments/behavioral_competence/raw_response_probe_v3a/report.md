# LearnedProbe V3A -- Frozen Raw-Response Representation Test

**Status: BLOCKED before scientific evaluation.**

V3A requires the exact frozen V2 OOF raw forecast responses from the learned conditional probe, or enough frozen V2 state to reconstruct them without retraining the generator.

The completed V2 artifacts contain router-val learned deltas and OOF six-stat response summaries, but they do not contain OOF learned deltas, OOF full raw response tensors, or trained V2 generator checkpoints. Re-running `train_learned_shared_prefix` would retrain the V2 generator and violate the V3A hard rule.

Therefore the primary comparisons `RawResponseActive vs SixStatActive`, `RawResponseActive vs ShuffledRawResponse`, `PassivePlusRaw vs PassiveOnly`, and `RawResponse -> passive residual` cannot be run under the frozen V3A rules.

```text
TEST SET ACCESSED: NO
V2 PERTURBATION GENERATOR RETRAINED: NO
FORECASTING EXPERTS RETRAINED: NO
V2 RESULT MODIFIED: NO
ROUTER TRAINED: NO
```

## Artifact Availability

| Dataset | Router-val learned delta | OOF learned six stats | OOF learned delta | OOF full raw response | V2 generator checkpoint |
|---|---:|---:|---:|---:|---:|
| ExchangeRate | True | True | False | False | False |
| Traffic | True | True | False | False | False |
| BeijingAirQuality | True | True | False | False | False |
| ETTm2 | True | True | False | False | False |

## Decision

Do not report a V3A scientific classification from these artifacts. The correct next action, if V3A is still desired, is a separate V2-compatible rerun that freezes and saves OOF learned deltas/generator checkpoints before V3A is attempted. That would be a new experiment, not this frozen V3A representation test.
