# Expert-Choice HxV Pilot

| Method | MAE | Delta vs Equal | Delta vs Hard HxV |
|---|---:|---:|---:|
| Equal ensemble | 0.214457 | +0.000000 | -0.008303 |
| Existing soft HxV | 0.211775 | -0.002682 | -0.010985 |
| Hard Normal HxV | 0.222761 | +0.008303 | +0.000000 |
| Expert Choice cap 1.00 | 0.222948 | +0.008491 | +0.000187 |
| Expert Choice cap 1.25 | 0.220627 | +0.006170 | -0.002134 |
| Expert Choice cap 1.50 | 0.222734 | +0.008277 | -0.000027 |

Verdict: STRONG GO

Among the predeclared capacity settings, `Expert Choice cap 1.25` had the lowest reported router_val MAE with delta -0.002134 versus Hard Normal HxV. Strong-Go variants by the fixed rule: ['Expert Choice cap 1.25']; Weak variants: ['Expert Choice cap 1.50']. Because Hard Normal HxV and all Expert Choice variants use the same train-derived score tensor, any difference comes from the capacity-constrained assignment mechanism rather than a changed competence model. No-capacity Expert Choice was exactly identical to Hard Normal HxV: True. This pilot does not select a deployment capacity; it reports all predeclared capacities.

## Allocation

| Method | PatchTST cells | iTransformer cells | TimesNet cells |
|---|---:|---:|---:|
| Equal ensemble | 1284.0 | 1284.0 | 1284.0 |
| Existing soft HxV | n/a | n/a | n/a |
| Hard Normal HxV | 1945 | 1768 | 139 |
| Expert Choice cap 1.00 | 1284 | 1284 | 1284 |
| Expert Choice cap 1.25 | 1605 | 1605 | 642 |
| Expert Choice cap 1.50 | 1926 | 1787 | 139 |

## Main Statistics

| Comparison | Mean Delta | 95% CI | P(delta < 0) | Phase Agreement |
|---|---:|---:|---:|---:|
| Expert Choice cap 1.00 vs Hard Normal HxV | +0.000187 | [-0.001234, +0.001591] | 0.394 | 6/12 negative |
| Expert Choice cap 1.25 vs Hard Normal HxV | -0.002134 | [-0.003012, -0.001256] | 1.000 | 12/12 negative |
| Expert Choice cap 1.50 vs Hard Normal HxV | -0.000026 | [-0.000036, -0.000017] | 1.000 | 12/12 negative |

## Controls

- No-capacity Expert Choice identical to Hard Normal HxV: `True`.
- Random-score and permuted-location controls are reported in `results.json`; neither uses validation targets to form assignments.

## Oracle Diagnostics

| Oracle / Non-deployable | MAE | MSE |
|---|---:|---:|
| Dynamic oracle per-window HxV (ORACLE / NON-DEPLOYABLE) | 0.138875 | 0.064471 |
| Static val-average HxV oracle (ORACLE / NON-DEPLOYABLE) | 0.215789 | 0.118859 |

## Integrity

- No test cache/file loaded: `True`.
- Expert ordering verified: `True`.
- Same score tensor used for Hard HxV and Expert Choice: `True`.
- Assignments train-only: `True`.
- Checkpoints unchanged: `True`.

Existing Electricity HxV reference note: previous frozen multidataset report lists HxV around MAE `0.211775`. This run's `Existing soft HxV` row reuses the canonical causal HxV utility, while the hard allocation rows use a static train-only score tensor, so exact equality is not required.