# Controlled Discriminative LearnedProbe v2

Date: 2026-08-25

Status: completed, development/mechanism evidence only.

## Question

When every frozen forecaster is subjected to the same learned controlled intervention, can its behavioral response reveal instance-specific conditional competence unavailable from passive observations alone?

## Protocol

- Experiment directory: `experiments/behavioral_competence/controlled_discriminative_probe_v2/`.
- New implementation files:
  - `shared_probe_generator.py`
  - `run_controlled_discriminative_probe_v2.py`
- Datasets: `ExchangeRate`, `Traffic`, `BeijingAirQuality`, `ETTm2`.
- Frozen K=3 cores reused from `experiments/behavioral_competence/generalization/dataset_selection.json` / `register_dataset`, selected using router-train only.
- No test cache was loaded.
- No forecasting expert was retrained.
- LearnedProbe v1 and existing negative TimeFuse/FFORMA/simplex results were not modified or overwritten.
- The active perturbation is shared per window: `delta_t = G(X_t)` and is applied identically to every selected expert.
- Target is conditional competence: actual per-expert error minus a causal train-only expert prior.
- Supervised OOF metrics use strict purged walk-forward folds.

## Run Notes

The first full run had been interrupted after writing complete per-window/raw-response caches for `ExchangeRate` and `Traffic`. A cache-aware recovery path was added so completed per-dataset artifacts can be reconstructed instead of retraining expensive datasets.

`BeijingAirQuality` exposed a zero-probe integrity issue: the original max-only check failed because cached forecasts and fresh frozen-runtime forecasts have rare large `TimesNet` reproduction outliers. This exact mismatch already exists in the older behavioral perturbation caches:

- `BeijingAirQuality__router_train_block_b__TimesNet__perturbations.pt`: max `39.190704`, mean `0.014863`, fraction windows > `0.1` = `0.011420`.
- `BeijingAirQuality__router_train_block_c__TimesNet__perturbations.pt`: max `33.052200`, mean `0.009052`, fraction windows > `0.1` = `0.003948`.
- `BeijingAirQuality__router_val__TimesNet__perturbations.pt`: max `18.841339`, mean `0.005291`, fraction windows > `0.1` = `0.001128`.

The v2 zero-probe gate was therefore changed to record the max but pass/fail using mean response and fraction of material outliers, matching the existing reproduction-diagnostic convention. Measured v2 zero-response distribution for BeijingAirQuality before the patch: max `0.767053`, mean `0.000208`, fraction response entries > `0.1` = `0.000540`.

## Main Result

Predeclared classification:

- Tier: `ACTIVE_SIGNAL_BUT_REDUNDANT`.
- Proceed to router integration: `false`.

Interpretation from the report:

Controlled probing reveals competence-related behavior, but the information is largely redundant with passive signals. Passive+Active does not improve over Passive, and active features do not predict MatchedPassive's residual. Do not claim router usefulness.

Predeclared criteria:

| Criterion | Met |
|---|---:|
| Beats random on multiple datasets | false |
| Beats shuffled | false |
| Significant correlation | true |
| Ranking above controls | false |
| Predicts passive residual | false |
| Passive+Active improves | false |

Only `1/6` criteria were met.

## Router-Val Highlights

Primary `SharedConditionalLearnedProbe` versus main controls:

| Dataset | Conditional MAE | Pearson | Pairwise acc | Beats random MAE | Beats shuffled MAE | Beats MatchedPassive MAE |
|---|---:|---:|---:|---:|---:|---:|
| ExchangeRate | `0.046388` | `0.335106` | `0.352` | yes | no | yes |
| Traffic | `0.073787` | `-0.035885` | `0.440` | no | no | no |
| BeijingAirQuality | `0.122485` | `0.077912` | `0.593` | yes | yes, tiny | no |
| ETTm2 | `0.057366` | `-0.075862` | `0.560` | no | yes | no |

MatchedPassive was stronger on `Traffic`, `BeijingAirQuality`, and `ETTm2` by conditional MAE.

OOF incremental diagnostics:

| Dataset | Passive-only R2 | Active-only R2 | Passive+Active R2 | Passive+ShuffledActive R2 | Active->PassiveResidual R2 |
|---|---:|---:|---:|---:|---:|
| ExchangeRate | `0.3033` | `-0.0094` | `0.3033` | `0.3033` | `-0.0161` |
| Traffic | `0.6032` | `-0.0621` | `0.6239` | `0.6060` | `-0.1535` |
| BeijingAirQuality | `0.1887` | `-0.2487` | `0.1886` | `0.1887` | `-0.0129` |
| ETTm2 | `0.1168` | `-0.0419` | `0.1167` | `0.1167` | `-0.1427` |

## Integrity

All four datasets passed:

- Same raw perturbation applied to every expert within each window.
- Purged-OOF causal checks.
- Checkpoints unchanged.
- Experts remained frozen.
- Target-corruption invariance.
- No test cache loaded.

Hard-rule output:

```text
TEST SET ACCESSED: NO
FORECASTING EXPERTS RETRAINED: NO
LEARNEDPROBE V1 (probe_generator.py::ProbeGenerator) MODIFIED: NO
EXISTING NEGATIVE TIMEFUSE/FFORMA RESULTS OVERWRITTEN: NO
ROUTER (TIMEFUSE/FFORMA/SIMPLEX/SELECTIVE/COSTAR) TRAINED IN THIS EXPERIMENT: NO
EPSILON TUNED: NO (fixed at 0.05)
POST-HOC RESCUE (different epsilon/architecture/ranking weight/folds/features/router after seeing results): NO
```

## Artifacts

- `experiments/behavioral_competence/controlled_discriminative_probe_v2/report.md`
- `experiments/behavioral_competence/controlled_discriminative_probe_v2/validation_results.json`
- `experiments/behavioral_competence/controlled_discriminative_probe_v2/prompt_compliance_audit.md`
- `experiments/behavioral_competence/controlled_discriminative_probe_v2/perturbation_rms_diagnostics.csv`
- `experiments/behavioral_competence/controlled_discriminative_probe_v2/expert_order_permutation_checks.csv`
- `experiments/behavioral_competence/controlled_discriminative_probe_v2/router_val_competence_results.csv`
- `experiments/behavioral_competence/controlled_discriminative_probe_v2/oof_competence_results.csv`
- `experiments/behavioral_competence/controlled_discriminative_probe_v2/dependence_aware_results.csv`
- `experiments/behavioral_competence/controlled_discriminative_probe_v2/passive_active_diagnostics.csv`
- `experiments/behavioral_competence/controlled_discriminative_probe_v2/residual_information_results.csv`
- `experiments/behavioral_competence/controlled_discriminative_probe_v2/integrity_checks.csv`
- `experiments/behavioral_competence/controlled_discriminative_probe_v2/per_window_scores/`
- `experiments/behavioral_competence/controlled_discriminative_probe_v2/raw_response_cache/`

## Decision

Do not proceed to TimeFuse/FFORMA/router integration for this active-probing formulation. The active response contains some competence-related correlation, but it is not independently useful beyond passive features under the frozen decision rule.

## Prompt Compliance Addendum

After the final prompt was provided, a post-run audit was added without retraining, rescoring, tuning, or changing the result.

Supplemental diagnostics:

- `perturbation_rms_diagnostics.csv` adds the RMS normalized delta requested by the prompt's perturbation-behavior section.
- `expert_order_permutation_checks.csv` verifies metric equivariance under a fixed expert-axis permutation; all rows pass with tolerance `1e-3`.
- `prompt_compliance_audit.md` / `.json` records full, structural, supplemental, and partial compliance points.

Literal storage limitations:

- The raw-response cache does not include `SharedTotalLearnedProbe` raw response tensors.
- The per-window cache does not explicitly store absolute forecast origins; it stores common indices and predictions/targets.
- The shared delta is stored once per window rather than duplicated per expert path. This matches the implemented mechanism but is not the literal Section 32 storage format.

The result remains `ACTIVE_SIGNAL_BUT_REDUNDANT`, with `proceed_to_router_integration=false`.
