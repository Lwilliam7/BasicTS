# Controlled Discriminative LearnedProbe v2 -- strict purged-OOF mechanism experiment

**Status: DEVELOPMENT / MECHANISM EVIDENCE, not a final generalization claim (Section 45).** These four datasets (ExchangeRate, Traffic, BeijingAirQuality, ETTm2) already influenced the frozen K=3 expert-core selection reused here (`../generalization/dataset_selection.json`); if this method shows a strong active signal, it must be frozen and re-evaluated on new, untouched datasets before any generalization claim is made.

## Final scientific question (Section 46)

*When every frozen forecaster is subjected to the SAME learned controlled intervention, can its behavioral response reveal instance-specific conditional competence that is unavailable from passive observations alone?*

**Answer: ACTIVE_SIGNAL_BUT_REDUNDANT.** Controlled probing reveals competence-related behavior, but the information is largely redundant with passive signals (Passive+Active does not improve over Passive, and active features do not predict MatchedPassive's residual). Do not claim router usefulness.

## Section 44 answers

1. **Same raw perturbation applied to every expert within each window?** True (structural: delta computed once per window batch, before the per-expert loop; max_abs diff reported per dataset in `integrity_checks.csv`).
2. **Did all purged-OOF causal checks pass?** True (see `causality_checks.csv`, `oof_fold_manifest.csv`).
3. **Does a random shared perturbation contain competence signal?** SharedRandomProbe beats ZeroProbe's null predictor on 1/4 datasets by point estimate (router_val).
4. **Does LEARNING the shared perturbation improve over random?** SharedConditionalLearnedProbe beats SharedRandomProbe on 2/4 datasets by point estimate, significant (block-24) on 2/4.
5. **Does predicting CONDITIONAL competence improve usefulness vs total-error probing?** See per-dataset `SharedLearnedTotalProbe` vs `SharedConditionalLearnedProbe` rows in `router_val_competence_results.csv`.
6. **Does the real expert mapping beat shuffled identity?** SharedConditionalLearnedProbe beats ShuffledConditionalProbe on 2/4 datasets by point estimate, significant on 2/4.
7. **Can active responses predict instance-specific good/bad?** Significant |Pearson| correlation with actual conditional error on 3/4 datasets.
8. **Can active responses predict what MatchedPassive gets wrong?** Positive R² predicting MatchedPassive's OOF residual on 0/4 datasets (`residual_information_results.csv`).
9. **Does Passive+Active outperform Passive alone?** On 1/4 datasets (`passive_active_diagnostics.csv`).
10. **Is any signal consistent across multiple datasets?** 1/6 predeclared criteria met (see below).
11. **Classification:** ACTIVE_SIGNAL_BUT_REDUNDANT.
12. **Proceed to TimeFuse/FFORMA integration?** NO.

## Predeclared criteria (Section 28)

- **1_beats_random_multiple_datasets**: False
- **2_beats_shuffled**: False
- **3_significant_correlation**: True
- **4_ranking_above_controls**: False
- **5_predicts_passive_residual**: False
- **6_passive_plus_active_improves**: False

## Table 1 -- primary active signal (router_val)

| Dataset | Method | Conditional MAE | R² | Pearson | Spearman | Pairwise acc | Top-1 acc |
|---|---|---:|---:|---:|---:|---:|---:|
| ExchangeRate | ZeroProbe | 0.046511 | nan | nan | nan | 0.000 | 0.163 |
| ExchangeRate | SharedRandomProbe | 0.049718 | -0.3047 | 0.3210 | 0.3036 | 0.608 | 0.486 |
| ExchangeRate | SharedLearnedTotalProbe | 0.106481 | -1.6635 | -0.0275 | -0.0771 | 0.347 | 0.150 |
| ExchangeRate | SharedConditionalLearnedProbe | 0.046388 | -0.1455 | 0.3351 | 0.2145 | 0.352 | 0.162 |
| ExchangeRate | ShuffledConditionalProbe | 0.046378 | -0.1455 | 0.3338 | 0.2646 | 0.567 | 0.411 |
| ExchangeRate | MatchedPassive | 0.059542 | -0.4534 | 0.3856 | 0.3377 | 0.646 | 0.621 |
| Traffic | ZeroProbe | 0.071337 | nan | nan | nan | 0.000 | 0.307 |
| Traffic | SharedRandomProbe | 0.071783 | -0.0076 | -0.0876 | -0.0679 | 0.409 | 0.240 |
| Traffic | SharedLearnedTotalProbe | 0.322976 | -13.3682 | 0.1955 | 0.2093 | 0.653 | 0.416 |
| Traffic | SharedConditionalLearnedProbe | 0.073787 | -0.0920 | -0.0359 | 0.0046 | 0.440 | 0.260 |
| Traffic | ShuffledConditionalProbe | 0.072660 | -0.0415 | 0.0593 | 0.0909 | 0.535 | 0.375 |
| Traffic | MatchedPassive | 0.050451 | 0.4590 | 0.7198 | 0.7537 | 0.682 | 0.475 |
| BeijingAirQuality | ZeroProbe | 0.152412 | nan | nan | nan | 0.000 | 0.181 |
| BeijingAirQuality | SharedRandomProbe | 0.139379 | -0.0848 | 0.3078 | 0.2848 | 0.542 | 0.400 |
| BeijingAirQuality | SharedLearnedTotalProbe | 0.385150 | -3.8000 | 0.2777 | 0.2624 | 0.415 | 0.211 |
| BeijingAirQuality | SharedConditionalLearnedProbe | 0.122485 | -0.0088 | 0.0779 | 0.2604 | 0.593 | 0.489 |
| BeijingAirQuality | ShuffledConditionalProbe | 0.122485 | -0.0088 | 0.0753 | 0.2125 | 0.459 | 0.251 |
| BeijingAirQuality | MatchedPassive | 0.102524 | 0.2767 | 0.5324 | 0.4930 | 0.549 | 0.369 |
| ETTm2 | ZeroProbe | 0.051054 | nan | nan | nan | 0.000 | 0.186 |
| ETTm2 | SharedRandomProbe | 0.051525 | -0.0429 | 0.1492 | 0.2307 | 0.577 | 0.487 |
| ETTm2 | SharedLearnedTotalProbe | 0.217454 | -9.4154 | -0.0825 | -0.1640 | 0.510 | 0.382 |
| ETTm2 | SharedConditionalLearnedProbe | 0.057366 | -0.1990 | -0.0759 | -0.0823 | 0.560 | 0.459 |
| ETTm2 | ShuffledConditionalProbe | 0.057799 | -0.2142 | -0.1106 | -0.1267 | 0.470 | 0.266 |
| ETTm2 | MatchedPassive | 0.053003 | -0.0833 | 0.2130 | 0.2056 | 0.586 | 0.533 |

## Table 1 (honest OOF, router_train Common windows)

| Dataset | Method | Conditional MAE | R² | Pearson | Spearman | Pairwise acc | Top-1 acc |
|---|---|---:|---:|---:|---:|---:|---:|
| ExchangeRate | ZeroProbe | 0.032704 | nan | nan | nan | 0.000 | 0.204 |
| ExchangeRate | SharedRandomProbe | 0.032754 | -0.0338 | -0.1195 | 0.0107 | 0.601 | 0.451 |
| ExchangeRate | SharedLearnedTotalProbe | 0.112673 | -6.9207 | 0.1319 | 0.2274 | 0.350 | 0.201 |
| ExchangeRate | SharedConditionalLearnedProbe | 0.032439 | -0.0305 | -0.1198 | 0.0193 | 0.510 | 0.375 |
| ExchangeRate | ShuffledConditionalProbe | 0.032439 | -0.0305 | -0.1198 | 0.0192 | 0.490 | 0.308 |
| ExchangeRate | MatchedPassive | 0.054302 | -1.9191 | -0.1055 | -0.0822 | 0.629 | 0.580 |
| Traffic | ZeroProbe | 0.078732 | nan | nan | nan | 0.000 | 0.369 |
| Traffic | SharedRandomProbe | 0.079007 | -0.0132 | -0.1048 | -0.0703 | 0.517 | 0.437 |
| Traffic | SharedLearnedTotalProbe | 0.378188 | -15.3526 | 0.0785 | 0.0713 | 0.489 | 0.293 |
| Traffic | SharedConditionalLearnedProbe | 0.070460 | 0.1086 | 0.3506 | 0.3618 | 0.573 | 0.412 |
| Traffic | ShuffledConditionalProbe | 0.076373 | -0.0177 | 0.2061 | 0.2280 | 0.469 | 0.291 |
| Traffic | MatchedPassive | 0.062014 | 0.2504 | 0.5587 | 0.5950 | 0.594 | 0.422 |
| BeijingAirQuality | ZeroProbe | 0.164208 | nan | nan | nan | 0.000 | 0.249 |
| BeijingAirQuality | SharedRandomProbe | 0.164302 | 0.0019 | 0.0849 | 0.2285 | 0.531 | 0.413 |
| BeijingAirQuality | SharedLearnedTotalProbe | 0.395402 | -2.1623 | -0.0674 | 0.0235 | 0.470 | 0.297 |
| BeijingAirQuality | SharedConditionalLearnedProbe | 0.163750 | 0.0027 | 0.1085 | 0.0171 | 0.529 | 0.428 |
| BeijingAirQuality | ShuffledConditionalProbe | 0.163751 | 0.0027 | 0.1083 | 0.0099 | 0.486 | 0.287 |
| BeijingAirQuality | MatchedPassive | 0.131026 | 0.3034 | 0.5554 | 0.5699 | 0.541 | 0.360 |
| ETTm2 | ZeroProbe | 0.074481 | nan | nan | nan | 0.000 | 0.223 |
| ETTm2 | SharedRandomProbe | 0.073497 | 0.0020 | 0.1583 | 0.3091 | 0.504 | 0.361 |
| ETTm2 | SharedLearnedTotalProbe | 0.189390 | -1.3357 | -0.1422 | -0.2886 | 0.506 | 0.353 |
| ETTm2 | SharedConditionalLearnedProbe | 0.074791 | -0.0134 | -0.1294 | -0.2817 | 0.557 | 0.458 |
| ETTm2 | ShuffledConditionalProbe | 0.074792 | -0.0135 | -0.1311 | -0.3050 | 0.472 | 0.273 |
| ETTm2 | MatchedPassive | 0.069936 | 0.2693 | 0.5209 | 0.2841 | 0.556 | 0.386 |

## Table 2 -- incremental information (OOF common)

| Dataset | Passive-only R² | Active-only R² | Passive+Active R² | Passive+ShuffledActive R² | Active->PassiveResidual R² |
|---|---:|---:|---:|---:|---:|
| ExchangeRate | 0.3033 | -0.0094 | 0.3033 | 0.3033 | -0.0161 |
| Traffic | 0.6032 | -0.0621 | 0.6239 | 0.6060 | -0.1535 |
| BeijingAirQuality | 0.1887 | -0.2487 | 0.1886 | 0.1887 | -0.0129 |
| ETTm2 | 0.1168 | -0.0419 | 0.1167 | 0.1167 | -0.1427 |

## Table 3 -- perturbation behavior (router_val)

| Dataset | Method | Mean norm |delta| | Max norm |delta| | Mean-shift penalty | Smoothness penalty | Mean response magnitude |
|---|---|---:|---:|---:|---:|---:|
| ExchangeRate | SharedRandomProbe | 0.016169 | 0.049783 | 0.000000 | 0.000000 | 0.001961 |
| ExchangeRate | SharedConditionalLearnedProbe | 0.020301 | 0.048909 | 0.000000 | 0.000000 | 0.002199 |
| Traffic | SharedRandomProbe | 0.016154 | 0.049955 | 0.000000 | 0.000000 | 0.006263 |
| Traffic | SharedConditionalLearnedProbe | 0.046098 | 0.050000 | 0.000003 | 0.000001 | 0.022008 |
| BeijingAirQuality | SharedRandomProbe | 0.017384 | 4.761807 | 0.027081 | 0.213195 | 0.003969 |
| BeijingAirQuality | SharedConditionalLearnedProbe | 0.000711 | 0.379959 | 0.000158 | 0.000116 | 0.000172 |
| ETTm2 | SharedRandomProbe | 0.038509 | 4.903950 | 0.000137 | 0.001079 | 0.002110 |
| ETTm2 | SharedConditionalLearnedProbe | 0.084044 | 4.999546 | 0.001746 | 0.001146 | 0.004144 |

## Dependence-aware statistics, primary block=24 (router_val, per-window competence MAE)

| Dataset | Comparison | Mean Delta | 95% CI | Excludes zero |
|---|---|---:|---|---|
| ExchangeRate | Conditional_vs_Random | `-0.003330` | [-0.004939, -0.001707] | True |
| ExchangeRate | Conditional_vs_Shuffled | `+0.000009` | [+0.000006, +0.000014] | True |
| ExchangeRate | Conditional_vs_MatchedPassive | `-0.013154` | [-0.019416, -0.006508] | True |
| ExchangeRate | Conditional_vs_LearnedTotal | `-0.060093` | [-0.074376, -0.043765] | True |
| Traffic | Conditional_vs_Random | `+0.002003` | [+0.000686, +0.003359] | True |
| Traffic | Conditional_vs_Shuffled | `+0.001126` | [-0.000143, +0.002477] | False |
| Traffic | Conditional_vs_MatchedPassive | `+0.023336` | [+0.020223, +0.026447] | True |
| Traffic | Conditional_vs_LearnedTotal | `-0.249189` | [-0.257640, -0.239951] | True |
| BeijingAirQuality | Conditional_vs_Random | `-0.016894` | [-0.019125, -0.014772] | True |
| BeijingAirQuality | Conditional_vs_Shuffled | `-0.000000` | [-0.000000, -0.000000] | True |
| BeijingAirQuality | Conditional_vs_MatchedPassive | `+0.019961` | [+0.015001, +0.025032] | True |
| BeijingAirQuality | Conditional_vs_LearnedTotal | `-0.262665` | [-0.275689, -0.250042] | True |
| ETTm2 | Conditional_vs_Random | `+0.005842` | [+0.005066, +0.006691] | True |
| ETTm2 | Conditional_vs_Shuffled | `-0.000432` | [-0.000594, -0.000278] | True |
| ETTm2 | Conditional_vs_MatchedPassive | `+0.004363` | [+0.002516, +0.006204] | True |
| ETTm2 | Conditional_vs_LearnedTotal | `-0.160088` | [-0.163960, -0.156544] | True |

## Causal fold assertions (Section 15)

| Dataset | Fold | Train target-end max | Heldout origin min | Assertion | Purged |
|---|---:|---:|---:|---|---:|
| ExchangeRate | 0 | 2645 | 2645 | True | 11 |
| ExchangeRate | 1 | 3598 | 3598 | True | 11 |
| Traffic | 0 | 6230 | 6230 | True | 11 |
| Traffic | 1 | 8378 | 8378 | True | 11 |
| BeijingAirQuality | 0 | 12874 | 12874 | True | 11 |
| BeijingAirQuality | 1 | 17237 | 17237 | True | 11 |
| ETTm2 | 0 | 20650 | 20650 | True | 11 |
| ETTm2 | 1 | 27605 | 27605 | True | 11 |

## Integrity

- **ExchangeRate**: PASS (checkpoints unchanged: True; experts frozen: True; same-question invariant: True; zero-probe near-zero: True; target-corruption invariant: True; all purge assertions pass: True; Common windows=1693, Full legal windows=2821)
- **Traffic**: PASS (checkpoints unchanged: True; experts frozen: True; same-question invariant: True; zero-probe near-zero: True; target-corruption invariant: True; all purge assertions pass: True; Common windows=4082, Full legal windows=6804)
- **BeijingAirQuality**: PASS (checkpoints unchanged: True; experts frozen: True; same-question invariant: True; zero-probe near-zero: True; target-corruption invariant: True; all purge assertions pass: True; Common windows=8512, Full legal windows=14186)
- **ETTm2**: PASS (checkpoints unchanged: True; experts frozen: True; same-question invariant: True; zero-probe near-zero: True; target-corruption invariant: True; all purge assertions pass: True; Common windows=13696, Full legal windows=22826)

## Hard rule compliance

```text
TEST SET ACCESSED: NO
FORECASTING EXPERTS RETRAINED: NO
LEARNEDPROBE V1 (probe_generator.py::ProbeGenerator) MODIFIED: NO
EXISTING NEGATIVE TIMEFUSE/FFORMA RESULTS OVERWRITTEN: NO
ROUTER (TIMEFUSE/FFORMA/SIMPLEX/SELECTIVE/COSTAR) TRAINED IN THIS EXPERIMENT: NO
EPSILON TUNED: NO (fixed at 0.05)
POST-HOC RESCUE (different epsilon/architecture/ranking weight/folds/features/router after seeing results): NO
```

## Section 30/31: what is deferred, not answered here

- **Response-encoder experiment**: NOT run. `raw_response_cache/{dataset}.npz` stores the shared delta and 6-feature response summary for the primary conditional probe (plus checkpoint SHA256 pins in `checkpoint_hashes.json`) so a future experiment can test whether the six-statistic summary, not the intervention itself, is the bottleneck.
- **Multi-amplitude experiment**: NOT run. This experiment uses exactly one fixed epsilon=0.05.

## Prompt compliance addendum

After comparing the completed run against the final self-contained prompt, a post-run audit was added without retraining, rescoring, tuning, or changing the result:

- `prompt_compliance_audit.md` / `prompt_compliance_audit.json`
- `perturbation_rms_diagnostics.csv`
- `expert_order_permutation_checks.csv`

The addendum records that RMS normalized perturbation magnitude and expert-order permutation metric equivariance are now explicitly reported. It also records literal storage limitations: the cache stores one shared delta per window rather than duplicated per-expert delta copies, does not include explicit absolute origins in the per-window npz, and does not include `SharedTotalLearnedProbe` raw response tensors. The final tier remains `ACTIVE_SIGNAL_BUT_REDUNDANT`, and `proceed_to_router_integration` remains `false`.
