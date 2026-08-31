Final classification: AFFINITY_WEIGHTED_EC_SUPPORTED

# Affinity-Weighted Expert Choice H x V

Post-hoc development experiment: not untouched confirmation. Reuses the frozen, already-trained window_dependent_expert_choice_hv score/affinity/claim tensors with NO retraining; changes only how multiple claiming experts' forecasts are combined.

```text
TEST SET ACCESSED: NO
TEST CACHE LOADED: NO
TEST METRICS COMPUTED: NO
```

## Router-val metrics (MAE)

| Dataset | Dynamic Token | Existing Dynamic EC | Weighted Dynamic EC | Frozen HxV | Weighted-Existing delta | Weighted-Frozen delta | Block-24 (Weighted vs Existing) |
|---|---:|---:|---:|---:|---:|---:|---|
| ETTh1 | `0.377209` | `0.375640` | `0.375671` | `0.366022` | `+0.000031` | `+0.009649` | no |
| ETTh2 | `0.280452` | `0.280951` | `0.280903` | `0.276898` | `-0.000049` | `+0.004005` | no |
| ETTm1 | `0.254659` | `0.253556` | `0.253540` | `0.250690` | `-0.000016` | `+0.002850` | no |
| Weather | `0.155880` | `0.155621` | `0.155585` | `0.159818` | `-0.000037` | `-0.004233` | YES |
| Electricity | `0.207034` | `0.206356` | `0.206322` | `0.215355` | `-0.000034` | `-0.009033` | YES |

## Router-train OOF (checked first)

| Dataset | Existing EC OOF MAE | Weighted EC OOF MAE | Delta |
|---|---:|---:|---:|
| ETTh1 | `0.351369` | `0.351321` | `-0.000047` |
| ETTh2 | `0.286207` | `0.286132` | `-0.000075` |
| ETTm1 | `0.254823` | `0.254765` | `-0.000059` |
| Weather | `0.166868` | `0.166781` | `-0.000087` |
| Electricity | `0.229083` | `0.229061` | `-0.000022` |

`OOF_SUPPORT = True` (5/5 datasets improved by the weighted rule).

## Multi-claim cell prevalence (router_val)

| Dataset | Zero-claim | One-claim | Multi-claim | Mean abs pred diff on multi-claim |
|---|---:|---:|---:|---:|
| ETTh1 | `0.1620` | `0.6759` | `0.1620` | `0.021856` |
| ETTh2 | `0.1660` | `0.6679` | `0.1660` | `0.007023` |
| ETTm1 | `0.1643` | `0.6714` | `0.1643` | `0.014205` |
| Weather | `0.2044` | `0.5912` | `0.2044` | `0.244370` |
| Electricity | `0.2283` | `0.5434` | `0.2283` | `1.826167` |

## Oracle headroom on multi-claim OOF cells (analysis only, never used to fit anything)

| Dataset | Multi-claim cells | Equal MAE | Weighted MAE | Oracle MAE | Headroom vs Equal | Headroom vs Weighted |
|---|---:|---:|---:|---:|---:|---:|
| ETTh1 | 51797 | `0.379627` | `0.379287` | `0.318984` | `0.060643` | `0.060303` |
| ETTh2 | 22031 | `0.323629` | `0.323162` | `0.273691` | `0.049939` | `0.049471` |
| ETTm1 | 233425 | `0.262892` | `0.262506` | `0.219252` | `0.043640` | `0.043255` |
| Weather | 818698 | `0.189104` | `0.188656` | `0.160325` | `0.028779` | `0.028331` |
| Electricity | 6554120 | `0.236101` | `0.235994` | `0.202763` | `0.033338` | `0.033231` |

## Classification counts

```json
{
  "OOF_SUPPORT": true,
  "block24_ci_below_zero_datasets": 2,
  "criteria": {
    "block24_support_ge_2": true,
    "integrity_pass": true,
    "oof_wins_ge_3": true,
    "token_wins_ge_3": true,
    "val_wins_ge_3": true
  },
  "integrity_pass": true,
  "negligible_multiclaim_everywhere": false,
  "oof_wins_vs_existing": 5,
  "total_multiclaim_cells_across_datasets": 7680071,
  "val_wins_vs_dynamic_token": 4,
  "val_wins_vs_existing": 4
}
```

## Seven questions

1. Did preserving affinity weights improve EC? Router-val: `4/5`. Router-train OOF: `5/5` (`OOF_SUPPORT=True`).
2. Was improvement concentrated on multi-claim cells as expected? Yes by construction (single/zero-claim parity is bit-identical, verified in `integrity_checks.json`); see the multi-claim prevalence table above for how much of the routing surface this actually touches.
3. Was there meaningful oracle headroom? See the oracle table; `oracle_headroom_vs_equal`/`oracle_headroom_vs_weighted` quantify the ceiling on multi-claim cells using true OOF targets (analysis-only).
4. Did EC retain its advantage over Dynamic Token? `4/5` router-val datasets.
5. Did it close any of the gap to Frozen HxV? Expert-side sparse assignment remains a useful matched-routing mechanism, but the current Expert Choice forecasting method does not yet outperform the dense H x V mixture (Weighted EC beat Frozen HxV on only 2/5 datasets).
6. Did all integrity checks pass? `True`.
7. Should this weighted EC formulation be frozen for confirmation, or should this development direction stop? FREEZE for confirmation-style evaluation on untouched datasets.
