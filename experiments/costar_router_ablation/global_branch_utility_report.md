# Global-Branch Utility Test

**Question**: does the separate global causal EMA branch provide useful information beyond the full horizon x variable (HxV) causal EMA?

Pure re-analysis of existing `router_ablation_per_window.csv` results (methods `global_causal`, `hxv_causal`, `global_plus_hxv`). No expert retrained, no router hyperparameter changed, no cache loaded, test set never touched.

## A. Overall validation MAE / MSE

| Dataset | Global only | Full HxV only | Global + HxV |
|---|---|---|---|
| ETTh1 | MAE `0.365755` / MSE `0.308944` | MAE `0.363949` / MSE `0.307478` | MAE `0.363634` / MSE `0.306684` |
| ETTh2 | MAE `0.280153` / MSE `0.171443` | MAE `0.276354` / MSE `0.167381` | MAE `0.276832` / MSE `0.167280` |

## B/C. Global+HxV vs HxV-only: paired block bootstrap

| Dataset | Block size | Mean delta (Global+HxV minus HxV) | 95% CI | P(delta<0) | CI excludes zero |
|---|---:|---:|---|---:|---|
| ETTh1 | 12 | `-0.000315` | [-0.000806, +0.000199] | 0.887 | False |
| ETTh1 | 24 | `-0.000315` | [-0.000900, +0.000240] | 0.857 | False |
| ETTh1 | 48 | `-0.000315` | [-0.000925, +0.000259] | 0.848 | False |
| ETTh2 | 12 | `+0.000478` | [-0.000351, +0.001279] | 0.129 | False |
| ETTh2 | 24 | `+0.000478` | [-0.000468, +0.001408] | 0.162 | False |
| ETTh2 | 48 | `+0.000478` | [-0.000492, +0.001417] | 0.163 | False |

## D. Every-12th non-overlapping-window evaluation

| Dataset | Mean delta | 95% CI (bootstrap over 12 phase means) | CI excludes zero |
|---|---:|---|---|
| ETTh1 | `-0.000315` | [-0.000515, -0.000115] | True |
| ETTh2 | `+0.000478` | [+0.000255, +0.000710] | True |

## E. Chronological 8-block split

### ETTh1

| Block | Windows | HxV-only MAE | Global+HxV MAE | Delta | Winner |
|---:|---|---:|---:|---:|---|
| 0 | 0-345 | `0.359692` | `0.359949` | `+0.000258` | hxv_only |
| 1 | 346-692 | `0.333822` | `0.334519` | `+0.000697` | hxv_only |
| 2 | 693-1038 | `0.448414` | `0.447119` | `-0.001295` | global_plus_hxv |
| 3 | 1039-1385 | `0.445400` | `0.444501` | `-0.000900` | global_plus_hxv |
| 4 | 1386-1732 | `0.336030` | `0.335966` | `-0.000064` | global_plus_hxv |
| 5 | 1733-2078 | `0.320509` | `0.320064` | `-0.000446` | global_plus_hxv |
| 6 | 2079-2425 | `0.374712` | `0.374236` | `-0.000476` | global_plus_hxv |
| 7 | 2426-2772 | `0.293117` | `0.292820` | `-0.000297` | global_plus_hxv |

### ETTh2

| Block | Windows | HxV-only MAE | Global+HxV MAE | Delta | Winner |
|---:|---|---:|---:|---:|---|
| 0 | 0-75 | `0.231811` | `0.232493` | `+0.000682` | hxv_only |
| 1 | 76-152 | `0.209198` | `0.210863` | `+0.001665` | hxv_only |
| 2 | 153-228 | `0.246298` | `0.247438` | `+0.001140` | hxv_only |
| 3 | 229-305 | `0.284877` | `0.285872` | `+0.000994` | hxv_only |
| 4 | 306-382 | `0.296019` | `0.298258` | `+0.002239` | hxv_only |
| 5 | 383-458 | `0.319078` | `0.318442` | `-0.000635` | global_plus_hxv |
| 6 | 459-535 | `0.321825` | `0.320526` | `-0.001299` | global_plus_hxv |
| 7 | 536-612 | `0.301310` | `0.300348` | `-0.000962` | global_plus_hxv |

## F. Block win/loss summary

| Dataset | Blocks global helps | Blocks global hurts | Avg improvement (helping) | Avg regression (hurting) |
|---|---:|---:|---:|---:|
| ETTh1 | 6/8 | 2/8 | `+0.000580` | `+0.000477` |
| ETTh2 | 3/8 | 5/8 | `+0.000966` | `+0.001344` |

## Final classification

**redundant**

The sign of the effect flips between ETTh1 and ETTh2 (helps on one, hurts on the other). Effect sizes are also tiny on both datasets, so the flip is more likely noise than a real regime effect -- classified as redundant rather than regime-dependent.

- **ETTh1**: point delta `-0.000315` (-0.087% relative), 6/8 chronological blocks favor Global+HxV, any block-bootstrap CI excludes zero: False.
- **ETTh2**: point delta `+0.000478` (+0.173% relative), 3/8 chronological blocks favor Global+HxV, any block-bootstrap CI excludes zero: False.

## Hard rule compliance

```text
TEST SET ACCESSED: NO
TEST CACHE LOADED: NO
TEST METRICS COMPUTED: NO
```
