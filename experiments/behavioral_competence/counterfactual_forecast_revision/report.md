# Counterfactual Forecast Revision (CFR)

**Status: DEVELOPMENT / MECHANISM STUDY.** These datasets are development datasets for CFR. Any promising CFR method must subsequently be frozen and evaluated on newly selected untouched datasets.

**Classification: CFR_SIGNAL_BUT_REDUNDANT.** CFR shows competence association and some passive-plus gains, but fails the mandatory direct Passive-residual criterion on at least three datasets, so it is not strong incremental model-specific evidence.

Interpretation note: an initial full run produced the same metrics but labeled the result `NO_USEFUL_CFR_SIGNAL`; this was corrected before final acceptance because the label contradicted the observed competence association and passive-plus gains. No features, folds, scales, hyperparameters, predictions, metrics, or validation fits were changed by that correction.

## Direct Answers

1. Does CFR predict conditional expert error? `4/4` datasets by the fixed router_val signal rule.
2. Does CFR add information beyond Passive? `4/4` datasets by point estimate; `3/4` with primary block-24 support.
3. Does RelativeCFR work? `3/4` datasets.
4. Can CFR predict Passive's residual? `1/4` datasets with positive router_val R2.
5. Does correct expert identity beat shuffled CFR? `3/4` datasets.
6. Is CFR mainly detecting globally hard windows? See `expert_specificity_results.csv`; common-mode fractions and shuffled controls decide this, not raw correlation alone.
7. Per-expert relationships are saved in `per_expert_correlations.csv` without cherry-picking.
8. Dependence-aware support is in `dependence_tests.csv`.
9. Every leakage/integrity check passed for all completed datasets.
10. Predeclared classification: `CFR_SIGNAL_BUT_REDUNDANT`.

## Router-Val Conditional Error

| Dataset | Method | MAE | R2 | Pearson | Spearman | Pairwise | Top1 |
|---|---|---:|---:|---:|---:|---:|---:|
| ExchangeRate | Passive | 0.045411 | 0.2426 | 0.5226 | 0.4017 | 0.448 | 0.270 |
| ExchangeRate | CFR | 0.045492 | -0.0645 | 0.1971 | 0.1131 | 0.437 | 0.247 |
| ExchangeRate | RelativeCFR | 0.046704 | -0.1524 | 0.0300 | -0.0369 | 0.484 | 0.322 |
| ExchangeRate | ShuffledCFR | 0.045451 | -0.0911 | 0.1642 | 0.1992 | 0.521 | 0.371 |
| ExchangeRate | PassivePlusCFR | 0.043796 | 0.2616 | 0.5298 | 0.4164 | 0.529 | 0.371 |
| ExchangeRate | PassivePlusRelativeCFR | 0.044104 | 0.2639 | 0.5327 | 0.4381 | 0.567 | 0.425 |
| ExchangeRate | PassivePlusShuffledCFR | 0.044462 | 0.2520 | 0.5190 | 0.4067 | 0.507 | 0.345 |
| Traffic | Passive | 0.050686 | 0.4880 | 0.7452 | 0.7658 | 0.618 | 0.438 |
| Traffic | CFR | 0.066191 | 0.1011 | 0.3186 | 0.3275 | 0.543 | 0.364 |
| Traffic | RelativeCFR | 0.071006 | 0.0004 | 0.0667 | 0.0761 | 0.566 | 0.393 |
| Traffic | ShuffledCFR | 0.068444 | 0.0570 | 0.2391 | 0.2296 | 0.492 | 0.331 |
| Traffic | PassivePlusCFR | 0.047863 | 0.5286 | 0.7495 | 0.7729 | 0.627 | 0.436 |
| Traffic | PassivePlusRelativeCFR | 0.050716 | 0.4837 | 0.7413 | 0.7658 | 0.606 | 0.418 |
| Traffic | PassivePlusShuffledCFR | 0.050887 | 0.4816 | 0.7388 | 0.7586 | 0.584 | 0.405 |
| BeijingAirQuality | Passive | 0.104370 | 0.2853 | 0.5513 | 0.5290 | 0.488 | 0.306 |
| BeijingAirQuality | CFR | 0.132228 | 0.0187 | 0.3261 | 0.3263 | 0.513 | 0.360 |
| BeijingAirQuality | RelativeCFR | 0.152421 | -0.1733 | 0.0232 | 0.0151 | 0.535 | 0.386 |
| BeijingAirQuality | ShuffledCFR | 0.133988 | -0.0008 | 0.3043 | 0.3222 | 0.501 | 0.325 |
| BeijingAirQuality | PassivePlusCFR | 0.105501 | 0.2717 | 0.5383 | 0.5194 | 0.540 | 0.383 |
| BeijingAirQuality | PassivePlusRelativeCFR | 0.103896 | 0.2913 | 0.5562 | 0.5379 | 0.557 | 0.389 |
| BeijingAirQuality | PassivePlusShuffledCFR | 0.104085 | 0.2819 | 0.5452 | 0.5240 | 0.496 | 0.305 |
| ETTm2 | Passive | 0.047096 | 0.0241 | 0.2519 | 0.2465 | 0.391 | 0.225 |
| ETTm2 | CFR | 0.051943 | -0.0568 | 0.1838 | 0.1731 | 0.496 | 0.332 |
| ETTm2 | RelativeCFR | 0.051266 | -0.0449 | 0.0123 | 0.0116 | 0.493 | 0.339 |
| ETTm2 | ShuffledCFR | 0.052392 | -0.0526 | 0.1750 | 0.1742 | 0.501 | 0.330 |
| ETTm2 | PassivePlusCFR | 0.047922 | 0.0344 | 0.2686 | 0.2557 | 0.430 | 0.259 |
| ETTm2 | PassivePlusRelativeCFR | 0.047084 | 0.0217 | 0.2555 | 0.2510 | 0.427 | 0.254 |
| ETTm2 | PassivePlusShuffledCFR | 0.047278 | 0.0199 | 0.2473 | 0.2386 | 0.397 | 0.230 |

## Router-Train Honest OOF

| Dataset | Method | MAE | R2 | Pearson | Spearman | Pairwise | Top1 |
|---|---|---:|---:|---:|---:|---:|---:|
| ExchangeRate | Passive | 0.032966 | -0.0230 | 0.1879 | 0.0860 | 0.402 | 0.253 |
| ExchangeRate | CFR | 0.034082 | -0.0416 | 0.1114 | 0.1567 | 0.462 | 0.180 |
| ExchangeRate | RelativeCFR | 0.034830 | -0.0563 | -0.0321 | -0.0393 | 0.488 | 0.264 |
| ExchangeRate | ShuffledCFR | 0.034025 | -0.0327 | 0.1153 | 0.1781 | 0.498 | 0.405 |
| ExchangeRate | PassivePlusCFR | 0.032514 | 0.0720 | 0.3191 | 0.2754 | 0.720 | 0.684 |
| ExchangeRate | PassivePlusRelativeCFR | 0.031420 | 0.0605 | 0.3156 | 0.2704 | 0.716 | 0.689 |
| ExchangeRate | PassivePlusShuffledCFR | 0.032978 | -0.0326 | 0.1826 | 0.0886 | 0.466 | 0.305 |
| Traffic | Passive | 0.053916 | 0.4418 | 0.6688 | 0.6899 | 0.508 | 0.283 |
| Traffic | CFR | 0.073405 | 0.0871 | 0.3075 | 0.3556 | 0.559 | 0.451 |
| Traffic | RelativeCFR | 0.079698 | -0.0200 | 0.1301 | 0.0997 | 0.599 | 0.519 |
| Traffic | ShuffledCFR | 0.078305 | -0.0033 | 0.1806 | 0.2404 | 0.495 | 0.306 |
| Traffic | PassivePlusCFR | 0.052319 | 0.4716 | 0.6869 | 0.7085 | 0.562 | 0.347 |
| Traffic | PassivePlusRelativeCFR | 0.052911 | 0.4619 | 0.6835 | 0.7067 | 0.609 | 0.445 |
| Traffic | PassivePlusShuffledCFR | 0.054695 | 0.4374 | 0.6729 | 0.6942 | 0.543 | 0.316 |
| BeijingAirQuality | Passive | 0.131301 | 0.3169 | 0.5673 | 0.5807 | 0.467 | 0.284 |
| BeijingAirQuality | CFR | 0.150530 | 0.1297 | 0.3621 | 0.3900 | 0.498 | 0.337 |
| BeijingAirQuality | RelativeCFR | 0.164511 | -0.0005 | 0.0103 | -0.0050 | 0.509 | 0.350 |
| BeijingAirQuality | ShuffledCFR | 0.150642 | 0.1235 | 0.3540 | 0.3877 | 0.495 | 0.327 |
| BeijingAirQuality | PassivePlusCFR | 0.131492 | 0.3187 | 0.5682 | 0.5833 | 0.509 | 0.339 |
| BeijingAirQuality | PassivePlusRelativeCFR | 0.130791 | 0.3198 | 0.5697 | 0.5852 | 0.528 | 0.363 |
| BeijingAirQuality | PassivePlusShuffledCFR | 0.130967 | 0.3176 | 0.5678 | 0.5845 | 0.484 | 0.304 |
| ETTm2 | Passive | 0.058017 | 0.4180 | 0.6577 | 0.4740 | 0.530 | 0.344 |
| ETTm2 | CFR | 0.070003 | 0.1704 | 0.4162 | 0.3342 | 0.532 | 0.349 |
| ETTm2 | RelativeCFR | 0.072633 | -0.0001 | 0.1424 | 0.0078 | 0.533 | 0.344 |
| ETTm2 | ShuffledCFR | 0.071826 | 0.0745 | 0.2886 | 0.3102 | 0.488 | 0.326 |
| ETTm2 | PassivePlusCFR | 0.059526 | 0.4212 | 0.6512 | 0.4706 | 0.540 | 0.356 |
| ETTm2 | PassivePlusRelativeCFR | 0.058137 | 0.4213 | 0.6592 | 0.4681 | 0.542 | 0.368 |
| ETTm2 | PassivePlusShuffledCFR | 0.058684 | 0.4209 | 0.6535 | 0.4830 | 0.523 | 0.335 |

## Passive Incremental Deltas

| Dataset | Split | Comparison | Delta MAE | Delta R2 | Delta Pairwise |
|---|---|---|---:|---:|---:|
| ExchangeRate | router_train_oof_common | PassivePlusCFR_vs_Passive | -0.000452 | +0.0950 | +0.318 |
| ExchangeRate | router_train_oof_common | PassivePlusRelativeCFR_vs_Passive | -0.001547 | +0.0835 | +0.314 |
| ExchangeRate | router_train_oof_common | PassivePlusCFR_vs_PassivePlusShuffledCFR | -0.000464 | +0.1047 | +0.254 |
| ExchangeRate | router_train_oof_common | CFR_vs_ShuffledCFR | +0.000056 | -0.0089 | -0.036 |
| ExchangeRate | router_train_oof_common | RelativeCFR_vs_ShuffledCFR | +0.000805 | -0.0236 | -0.009 |
| ExchangeRate | router_val | PassivePlusCFR_vs_Passive | -0.001615 | +0.0190 | +0.081 |
| ExchangeRate | router_val | PassivePlusRelativeCFR_vs_Passive | -0.001307 | +0.0214 | +0.120 |
| ExchangeRate | router_val | PassivePlusCFR_vs_PassivePlusShuffledCFR | -0.000666 | +0.0096 | +0.022 |
| ExchangeRate | router_val | CFR_vs_ShuffledCFR | +0.000041 | +0.0266 | -0.084 |
| ExchangeRate | router_val | RelativeCFR_vs_ShuffledCFR | +0.001253 | -0.0613 | -0.037 |
| Traffic | router_train_oof_common | PassivePlusCFR_vs_Passive | -0.001598 | +0.0297 | +0.054 |
| Traffic | router_train_oof_common | PassivePlusRelativeCFR_vs_Passive | -0.001005 | +0.0201 | +0.101 |
| Traffic | router_train_oof_common | PassivePlusCFR_vs_PassivePlusShuffledCFR | -0.002376 | +0.0342 | +0.019 |
| Traffic | router_train_oof_common | CFR_vs_ShuffledCFR | -0.004901 | +0.0904 | +0.064 |
| Traffic | router_train_oof_common | RelativeCFR_vs_ShuffledCFR | +0.001393 | -0.0166 | +0.104 |
| Traffic | router_val | PassivePlusCFR_vs_Passive | -0.002823 | +0.0406 | +0.009 |
| Traffic | router_val | PassivePlusRelativeCFR_vs_Passive | +0.000029 | -0.0042 | -0.012 |
| Traffic | router_val | PassivePlusCFR_vs_PassivePlusShuffledCFR | -0.003024 | +0.0469 | +0.043 |
| Traffic | router_val | CFR_vs_ShuffledCFR | -0.002253 | +0.0441 | +0.051 |
| Traffic | router_val | RelativeCFR_vs_ShuffledCFR | +0.002562 | -0.0567 | +0.074 |
| BeijingAirQuality | router_train_oof_common | PassivePlusCFR_vs_Passive | +0.000191 | +0.0019 | +0.041 |
| BeijingAirQuality | router_train_oof_common | PassivePlusRelativeCFR_vs_Passive | -0.000510 | +0.0029 | +0.061 |
| BeijingAirQuality | router_train_oof_common | PassivePlusCFR_vs_PassivePlusShuffledCFR | +0.000525 | +0.0011 | +0.024 |
| BeijingAirQuality | router_train_oof_common | CFR_vs_ShuffledCFR | -0.000112 | +0.0061 | +0.003 |
| BeijingAirQuality | router_train_oof_common | RelativeCFR_vs_ShuffledCFR | +0.013869 | -0.1241 | +0.014 |
| BeijingAirQuality | router_val | PassivePlusCFR_vs_Passive | +0.001131 | -0.0136 | +0.052 |
| BeijingAirQuality | router_val | PassivePlusRelativeCFR_vs_Passive | -0.000474 | +0.0060 | +0.069 |
| BeijingAirQuality | router_val | PassivePlusCFR_vs_PassivePlusShuffledCFR | +0.001417 | -0.0102 | +0.044 |
| BeijingAirQuality | router_val | CFR_vs_ShuffledCFR | -0.001760 | +0.0196 | +0.012 |
| BeijingAirQuality | router_val | RelativeCFR_vs_ShuffledCFR | +0.018433 | -0.1725 | +0.034 |
| ETTm2 | router_train_oof_common | PassivePlusCFR_vs_Passive | +0.001509 | +0.0032 | +0.010 |
| ETTm2 | router_train_oof_common | PassivePlusRelativeCFR_vs_Passive | +0.000120 | +0.0033 | +0.013 |
| ETTm2 | router_train_oof_common | PassivePlusCFR_vs_PassivePlusShuffledCFR | +0.000842 | +0.0003 | +0.017 |
| ETTm2 | router_train_oof_common | CFR_vs_ShuffledCFR | -0.001823 | +0.0958 | +0.044 |
| ETTm2 | router_train_oof_common | RelativeCFR_vs_ShuffledCFR | +0.000807 | -0.0747 | +0.046 |
| ETTm2 | router_val | PassivePlusCFR_vs_Passive | +0.000827 | +0.0103 | +0.039 |
| ETTm2 | router_val | PassivePlusRelativeCFR_vs_Passive | -0.000012 | -0.0024 | +0.036 |
| ETTm2 | router_val | PassivePlusCFR_vs_PassivePlusShuffledCFR | +0.000644 | +0.0145 | +0.033 |
| ETTm2 | router_val | CFR_vs_ShuffledCFR | -0.000449 | -0.0042 | -0.005 |
| ETTm2 | router_val | RelativeCFR_vs_ShuffledCFR | -0.001126 | +0.0076 | -0.007 |

## Passive Residual Prediction

| Dataset | Split | Method | MAE | R2 | Pearson | Spearman |
|---|---|---|---:|---:|---:|---:|
| ExchangeRate | router_train_oof_common | CFR_to_PassiveResidual | 0.032462 | 0.0685 | 0.2949 | 0.2910 |
| ExchangeRate | router_val | CFR_to_PassiveResidual | 0.046625 | -0.0619 | 0.0093 | 0.0679 |
| ExchangeRate | router_train_oof_common | RelativeCFR_to_PassiveResidual | 0.032058 | 0.0484 | 0.2618 | 0.2954 |
| ExchangeRate | router_val | RelativeCFR_to_PassiveResidual | 0.045042 | -0.0251 | 0.1175 | 0.1819 |
| ExchangeRate | router_train_oof_common | ShuffledCFR_to_PassiveResidual | 0.033099 | -0.0198 | -0.0284 | -0.0567 |
| ExchangeRate | router_val | ShuffledCFR_to_PassiveResidual | 0.044161 | -0.0147 | 0.0759 | 0.1535 |
| Traffic | router_train_oof_common | CFR_to_PassiveResidual | 0.051859 | 0.0548 | 0.2463 | 0.2448 |
| Traffic | router_val | CFR_to_PassiveResidual | 0.049897 | -0.1430 | 0.0509 | 0.0415 |
| Traffic | router_train_oof_common | RelativeCFR_to_PassiveResidual | 0.052515 | 0.0384 | 0.2226 | 0.2373 |
| Traffic | router_val | RelativeCFR_to_PassiveResidual | 0.051587 | -0.1896 | 0.0224 | 0.0314 |
| Traffic | router_train_oof_common | ShuffledCFR_to_PassiveResidual | 0.054582 | -0.0170 | 0.1193 | 0.1500 |
| Traffic | router_val | ShuffledCFR_to_PassiveResidual | 0.051199 | -0.1846 | -0.0502 | -0.0791 |
| BeijingAirQuality | router_train_oof_common | CFR_to_PassiveResidual | 0.131230 | 0.0012 | 0.0576 | 0.0438 |
| BeijingAirQuality | router_val | CFR_to_PassiveResidual | 0.105127 | -0.0425 | -0.0047 | 0.1040 |
| BeijingAirQuality | router_train_oof_common | RelativeCFR_to_PassiveResidual | 0.130910 | 0.0006 | 0.0620 | 0.0953 |
| BeijingAirQuality | router_val | RelativeCFR_to_PassiveResidual | 0.104078 | -0.0226 | 0.0629 | 0.1342 |
| BeijingAirQuality | router_train_oof_common | ShuffledCFR_to_PassiveResidual | 0.130985 | -0.0029 | 0.0316 | 0.0458 |
| BeijingAirQuality | router_val | ShuffledCFR_to_PassiveResidual | 0.104182 | -0.0324 | -0.0295 | -0.0179 |
| ETTm2 | router_train_oof_common | CFR_to_PassiveResidual | 0.059195 | -0.0164 | 0.0597 | 0.1089 |
| ETTm2 | router_val | CFR_to_PassiveResidual | 0.047544 | 0.0197 | 0.1529 | 0.1413 |
| ETTm2 | router_train_oof_common | RelativeCFR_to_PassiveResidual | 0.058355 | -0.0256 | 0.0218 | 0.0785 |
| ETTm2 | router_val | RelativeCFR_to_PassiveResidual | 0.046664 | 0.0124 | 0.1263 | 0.1545 |
| ETTm2 | router_train_oof_common | ShuffledCFR_to_PassiveResidual | 0.058444 | -0.0093 | 0.0608 | 0.1104 |
| ETTm2 | router_val | ShuffledCFR_to_PassiveResidual | 0.046931 | -0.0007 | 0.0566 | 0.0652 |

## Primary Dependence Tests

| Dataset | Comparison | Mean Delta | 95% CI | Excludes Zero |
|---|---|---:|---|---|
| ExchangeRate | PassivePlusCFR_vs_Passive | `-0.001615` | [-0.002534, -0.000703] | True |
| ExchangeRate | PassivePlusRelativeCFR_vs_Passive | `-0.001307` | [-0.001792, -0.000806] | True |
| ExchangeRate | PassivePlusCFR_vs_PassivePlusShuffledCFR | `-0.000666` | [-0.001504, +0.000146] | False |
| ExchangeRate | CFR_vs_ShuffledCFR | `+0.000041` | [-0.000766, +0.000867] | False |
| ExchangeRate | RelativeCFR_vs_ShuffledCFR | `+0.001253` | [-0.000039, +0.002749] | False |
| Traffic | PassivePlusCFR_vs_Passive | `-0.002823` | [-0.003500, -0.002127] | True |
| Traffic | PassivePlusRelativeCFR_vs_Passive | `+0.000029` | [-0.000504, +0.000604] | False |
| Traffic | PassivePlusCFR_vs_PassivePlusShuffledCFR | `-0.003024` | [-0.003672, -0.002355] | True |
| Traffic | CFR_vs_ShuffledCFR | `-0.002253` | [-0.003033, -0.001403] | True |
| Traffic | RelativeCFR_vs_ShuffledCFR | `+0.002562` | [+0.001862, +0.003255] | True |
| BeijingAirQuality | PassivePlusCFR_vs_Passive | `+0.001131` | [+0.000419, +0.001814] | True |
| BeijingAirQuality | PassivePlusRelativeCFR_vs_Passive | `-0.000474` | [-0.000839, -0.000084] | True |
| BeijingAirQuality | PassivePlusCFR_vs_PassivePlusShuffledCFR | `+0.001417` | [+0.000749, +0.002108] | True |
| BeijingAirQuality | CFR_vs_ShuffledCFR | `-0.001760` | [-0.002682, -0.000908] | True |
| BeijingAirQuality | RelativeCFR_vs_ShuffledCFR | `+0.018433` | [+0.015375, +0.021590] | True |
| ETTm2 | PassivePlusCFR_vs_Passive | `+0.000827` | [+0.000554, +0.001122] | True |
| ETTm2 | PassivePlusRelativeCFR_vs_Passive | `-0.000012` | [-0.000109, +0.000083] | False |
| ETTm2 | PassivePlusCFR_vs_PassivePlusShuffledCFR | `+0.000644` | [+0.000365, +0.000943] | True |
| ETTm2 | CFR_vs_ShuffledCFR | `-0.000449` | [-0.000786, -0.000119] | True |
| ETTm2 | RelativeCFR_vs_ShuffledCFR | `-0.001126` | [-0.001838, -0.000458] | True |

## Integrity

- **ExchangeRate**: PASS (checkpoints unchanged: True; experts frozen: True; target corruption max diff: 0.0e+00; deterministic CFR max diff: 0.0e+00; purge correct: True; test accessed: NO).
- **Traffic**: PASS (checkpoints unchanged: True; experts frozen: True; target corruption max diff: 0.0e+00; deterministic CFR max diff: 0.0e+00; purge correct: True; test accessed: NO).
- **BeijingAirQuality**: PASS (checkpoints unchanged: True; experts frozen: True; target corruption max diff: 0.0e+00; deterministic CFR max diff: 0.0e+00; purge correct: True; test accessed: NO).
- **ETTm2**: PASS (checkpoints unchanged: True; experts frozen: True; target corruption max diff: 0.0e+00; deterministic CFR max diff: 0.0e+00; purge correct: True; test accessed: NO).

## Hard Rule Compliance

```text
DEVELOPMENT_ONLY_MECHANISM_STUDY: YES
TEST SET ACCESSED: NO
EXPERTS RETRAINED OR FINE-TUNED: NO
ROUTER_VAL USED FOR TRAINING OR SCALE ESTIMATION: NO
PREFIX_K TUNED: NO
COUNTERFACTUAL_SCALE TUNED: NO
RIDGE_ALPHA TUNED: NO
FEATURE SELECTION AFTER VALIDATION: NO
```
