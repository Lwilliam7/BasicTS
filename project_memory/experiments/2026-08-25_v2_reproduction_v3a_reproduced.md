# V2 Reproduction And V3A Reproduced Raw-Response Test

Date: 2026-08-25

Status: Completed

## Question

Can the blocked V3A raw-response representation test be unblocked by a separate V2-compatible artifact reproduction that saves the missing fold checkpoints and full raw response tensors, without modifying frozen V2 or touching test data?

## Phase A: V2 Artifact Reproduction

Created:

- `experiments/behavioral_competence/controlled_discriminative_probe_v2_reproduction/`

The reproduction reused the archived V2 implementation and exact V2 constants:

- `SharedControlledProbeGenerator`
- one shared `delta_t = G(X_t)` per window
- `epsilon=0.05`
- active scorer input limited to six response statistics
- conditional target: actual error minus causal train-only expert prior
- Huber + `0.25` gap-weighted pairwise ranking + perturbation penalties
- frozen datasets and cores from V2

Source provenance:

The committed V2 implementation is the archived reproduction source. The original run's HEAD was `2904e28` while the then-uncommitted V2 source was subsequently committed in `7ec1f1e`. Bit-exact original source provenance is not claimed.

Gate result:

- `REPRODUCTION_ACCEPTED`
- Observable checks passed: `317/317`
- Folds/common windows matched exactly.
- Reproduced V2 qualitative classification: `ACTIVE_SIGNAL_BUT_REDUNDANT`
- Reproduced `proceed_to_router_integration=false`
- Test accessed: no.

Key artifacts:

- `experiments/behavioral_competence/controlled_discriminative_probe_v2_reproduction/reproduction_decision.json`
- `experiments/behavioral_competence/controlled_discriminative_probe_v2_reproduction/reproduction_comparison.csv`
- `experiments/behavioral_competence/controlled_discriminative_probe_v2_reproduction/checkpoints/`
- `experiments/behavioral_competence/controlled_discriminative_probe_v2_reproduction/oof_raw_response/`
- `experiments/behavioral_competence/controlled_discriminative_probe_v2_reproduction/router_val_raw_response/`
- `experiments/behavioral_competence/controlled_discriminative_probe_v2_reproduction/report.md`

## Phase B: V3A Reproduced

Created:

- `experiments/behavioral_competence/raw_response_probe_v3a_reproduced/`

This analysis used the accepted frozen-protocol V2 reproduction artifacts, not exact original V2 tensors.

Method:

- fixed `Ridge(alpha=1.0)`
- train-only standardization
- chronological 80/20 split over OOF common windows
- compared `SixStatActive`, `RawResponseActive`, `ShuffledRawResponse`, `MatchedPassive`, and `PassivePlusRaw`
- tested `RawResponse -> passive_residual`
- used dependence-aware block bootstrap with primary block length `24`

Classification:

- `SIX_STATS_NOT_THE_BOTTLENECK`

Summary:

| Dataset | SixStat MAE | Raw MAE | Shuffled Raw MAE | Passive MAE | Passive+Raw MAE | Residual R2 |
|---|---:|---:|---:|---:|---:|---:|
| ExchangeRate | `0.032797` | `0.034368` | `0.033657` | `0.028014` | `0.029058` | `-0.1766` |
| Traffic | `0.062098` | `0.143470` | `0.186858` | `0.038388` | `0.143986` | `-13.3361` |
| BeijingAirQuality | `0.245695` | `0.243078` | `0.243072` | `0.212210` | `0.215003` | `-0.0440` |
| ETTm2 | `0.057150` | `0.061785` | `0.063162` | `0.051905` | `0.051536` | `-0.3305` |

Counts:

- Raw better than six-stat: `1/4`
- Raw better than shuffled raw: `2/4`
- Passive+Raw better than Passive: `1/4`
- Positive passive-residual R2: `0/4`

Conclusion:

The full raw forecast response did not consistently outperform the six-stat summary, did not reliably beat shuffled raw responses, did not add consistent incremental value beyond MatchedPassive, and did not predict passive residuals. The six-stat compression is therefore not supported as the bottleneck for the V2 active-probe result.

## Compliance

```text
FROZEN V2 DIRECTORY MODIFIED: NO
FORECASTING EXPERTS RETRAINED: NO
ROUTER TRAINED: NO
TEST SET ACCESSED: NO
```
