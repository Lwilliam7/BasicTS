# Expert-Native Latent Competence

Date: 2026-08-29

Artifact directory: `experiments/behavioral_competence/expert_native_competence/`

## Question

Do the internal hidden representations of a frozen forecasting expert contain information about when that same expert will outperform the existing frozen HxV ensemble baseline?

## Protocol

- Strict validation-only: no test cache/file was loaded or evaluated.
- Datasets: `ETTh1`, `ETTh2`, `ETTm1`, `Weather`, `Electricity`.
- Horizon: `12`.
- Expert cores: current K=3 frozen cores from `experiments.frozen_hv_costar.run_frozen_hv_costar.LOADERS`.
- Baseline: frozen HxV COSTAR weights fit from router_train errors only.
- Target: `gain_k = MAE(frozen HxV baseline ensemble) - MAE(expert_k)`, positive when expert `k` beats the baseline.
- OOF: chronological forward folds on router_train, with `fit_start + horizon <= eval_origin`.
- Readouts: expert-specific `Ridge(alpha=1.0)` for continuous gain and linear `LogisticRegression` for `gain > 0`.
- Hidden dimensionality: selected per dataset/expert from the train-only PCA grid using router_train OOF Hidden Only MSE.
- Controls: Passive A+B+C, Hidden Only, Passive+Hidden, Shuffled Hidden, Raw Forecast Control, Matched-Dimension Passive Control, and prototype good-minus-bad axes.

## Representations

- DLinear: `linear_seasonal` and `linear_trend` branch outputs before seasonal+trend summation.
- PatchTST: `backbone` output before flatten and forecasting head.
- iTransformer: `backbone` output before forecasting head.
- TimesNet: `backbone` output before projection.
- ModernTCN: `backbone` output before temporal/feature heads.

Each captured tensor was pooled deterministically by treating internal states as tokens and concatenating mean, std, max, min, first-token, and last-token summaries. DLinear used its seasonal/trend branch outputs as tokens rather than inventing a hidden layer.

## Primary Router-Val Metrics

| Dataset | Passive R2 | Hidden R2 | Passive+Hidden R2 | Delta R2 | Passive AUROC | Hidden AUROC | Passive+Hidden AUROC | Delta AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ETTh1 | `0.133959` | `0.037692` | `0.150784` | `+0.016825` | `0.487321` | `0.489691` | `0.482347` | `-0.004974` |
| ETTh2 | `0.420947` | `0.116447` | `0.242852` | `-0.178095` | `0.633669` | `0.528356` | `0.502345` | `-0.131324` |
| ETTm1 | `0.002757` | `-0.185444` | `-0.025166` | `-0.027923` | `0.564605` | `0.530102` | `0.555871` | `-0.008734` |
| Weather | `0.212101` | `-0.044743` | `0.209657` | `-0.002443` | `0.569251` | `0.522571` | `0.605277` | `+0.036025` |
| Electricity | `-0.043196` | `0.251719` | `0.299067` | `+0.342262` | `0.785554` | `0.773196` | `0.818520` | `+0.032966` |

## Ranking Metrics

| Dataset | Passive pairwise acc | Hidden pairwise acc | Passive+Hidden pairwise acc | Shuffled Hidden |
|---|---:|---:|---:|---:|
| ETTh1 | `0.554153` | `0.566414` | `0.544897` | `0.547422` |
| ETTh2 | `0.705818` | `0.694943` | `0.682436` | `0.694943` |
| ETTm1 | `0.560764` | `0.535062` | `0.554309` | `0.560238` |
| Weather | `0.618577` | `0.592759` | `0.617555` | `0.618418` |
| Electricity | `0.723516` | `0.761609` | `0.793752` | `0.727526` |

## Dependence-Aware Result

Passive+Hidden versus Passive gain-prediction MSE:

- ETTh1: block-24 CI crossed zero; every-12th phase supported improvement.
- ETTh2: block-24 and every-12th significantly regressed.
- ETTm1: block-24 and every-12th significantly regressed.
- Weather: both dependence checks crossed zero.
- Electricity: block-24 and every-12th significantly improved.

## Integrity

- `test_loaded`: `false`.
- Checkpoint hashes unchanged before/after hidden extraction.
- Hooked-vs-unhooked prediction max difference: `0.0` for all datasets.
- Router_train OOF purge passed for all folds.
- Router_val target corruption left features and pre-evaluation competence predictions unchanged.
- All feature tensors finite.
- Cached forecast reproduction diagnostic: most experts were near-exact; TimesNet had rare device/batch-size-sensitive outliers on Weather/Electricity, while hook invariance stayed exactly zero. This was recorded in `representation_manifest.json` and `integrity_report.json`.

## Classification

`MIXED_SUPPORT`

The frozen forecasters do appear to internally encode some expert-relative competence information, but the evidence is not robust. Electricity is strongly positive and ETTh1 has a small continuous R2 increment, while ETTh2, ETTm1, and Weather fail or regress on the primary Passive+Hidden minus Passive comparison.

## Decision

Do not build, tune, or test an Expert-Native router from this mechanism yet. A follow-up would need to explain why Electricity carries strong unique hidden signal and prove that the effect is not a dataset-specific artifact before any routing integration.
