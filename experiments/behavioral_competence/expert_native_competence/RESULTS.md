# Expert-Native Latent Competence

Final classification: `MIXED_SUPPORT`

Strict validation-only. No test cache, target, or metric was loaded.

## Primary Validation Metrics

| Dataset | Passive R2 | Hidden R2 | Passive+Hidden R2 | dR2 | Passive AUROC | Hidden AUROC | Passive+Hidden AUROC | dAUROC |
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

## Integrity

- Test loaded: `False`.
- Checkpoint hashes were recorded before and after hidden-state extraction.
- Hooked and unhooked predictions are compared for every extracted batch.
- Hook extraction did not change model outputs: max hooked-vs-unhooked difference was `0.0` for all datasets.
- Cached-forecast reproduction was near-exact for most experts but TimesNet showed rare device/batch-size-sensitive outliers on Weather/Electricity; these diagnostics are recorded in `representation_manifest.json` and `integrity_report.json`. The experiment uses cached original forecasts for targets/controls and hidden hooks only as frozen-model observations.
- Router_train predictions are chronological OOF with horizon-12 purge.
- PCA dimensions are selected from router_train OOF only.
- Router_val target corruption leaves features and pre-evaluation competence predictions unchanged.

## Answer

Do frozen forecasters internally encode information about when they themselves are relatively competent? `Yes, but not robustly`: Electricity gives strong evidence and ETTh1 gives a small R2-only positive, while ETTh2, ETTm1, and Weather fail the incremental R2 test.

Is it unique enough to justify Expert-Native routing? `Not yet`. Classification is `MIXED_SUPPORT`, not `STRONG_SUPPORT`; do not build or evaluate a router from these hidden representations without a sharper follow-up mechanism.
