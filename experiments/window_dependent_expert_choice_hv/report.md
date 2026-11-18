Final classification: WINDOW_DEPENDENT_EC_SUPPORTED

# Window-Dependent Expert-Choice H x V Routing

Development experiment (not untouched confirmation). All five datasets already informed the prior static Expert-Choice CF=1 result and the decision to test window dependence.

```text
TEST SET ACCESSED: NO
TEST CACHE LOADED: NO
TEST METRICS COMPUTED: NO
```

## Router-val metrics (MAE)

| Dataset | Static Token Top1 | Static EC CF1 | Dynamic Token Top1 | Dynamic EC CF1 | Frozen HxV | Shuffled Dynamic EC | Dyn EC - Dyn Token | Dyn EC - Static EC | Block-24 support |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ETTh1 | `0.376680` | `0.375352` | `0.377209` | `0.375640` | `0.366022` | `0.382096` | `-0.001568` | `+0.000288` | YES |
| ETTh2 | `0.279132` | `0.277764` | `0.280452` | `0.280951` | `0.276898` | `0.281466` | `+0.000499` | `+0.003188` | no |
| ETTm1 | `0.262524` | `0.253570` | `0.254659` | `0.253556` | `0.250690` | `0.259875` | `-0.001103` | `-0.000015` | YES |
| Weather | `0.164399` | `0.160330` | `0.155880` | `0.155621` | `0.159818` | `0.166644` | `-0.000259` | `-0.004709` | YES |
| Electricity | `0.222761` | `0.219015` | `0.207034` | `0.206356` | `0.215355` | `0.226429` | `-0.000677` | `-0.012658` | YES |

## Router-train OOF (mechanism check before router_val)

| Dataset | Dynamic Token OOF MAE | Dynamic EC OOF MAE | Delta |
|---|---:|---:|---:|
| ETTh1 | `0.352005` | `0.351372` | `-0.000633` |
| ETTh2 | `0.285400` | `0.286206` | `+0.000806` |
| ETTm1 | `0.255623` | `0.254822` | `-0.000801` |
| Weather | `0.167161` | `0.166870` | `-0.000291` |
| Electricity | `0.228871` | `0.229083` | `+0.000211` |

## Classification counts

```json
{
  "block24_ci_below_zero_datasets": 4,
  "criteria": {
    "block24_support_ge_2": true,
    "genuinely_dynamic_ge_3": true,
    "integrity_pass": true,
    "oof_wins_ge_3": true,
    "shuffle_wins_ge_3": true,
    "static_wins_ge_3": true,
    "val_wins_ge_3": true
  },
  "genuinely_dynamic_datasets": 5,
  "integrity_pass": true,
  "oof_wins_vs_dynamic_token": 3,
  "shuffle_weakened_datasets": 5,
  "val_wins_vs_dynamic_token": 4,
  "val_wins_vs_static_ec": 3
}
```

## Nine questions

1. Did Dynamic EC beat matched Dynamic Token Choice? Router-val: `4/5`. Router-train OOF: `3/5`.
2. Did Dynamic EC improve on Static EC? `3/5` router-val datasets.
3. Did the learned scores genuinely vary by current window? See per-expert `predicted_residual_score_std_t` in `routing_diagnostics.csv`; `5/5` datasets met the predeclared adjacent-change/Jaccard threshold (question 4 detail).
4. Did expert claim masks genuinely change by current window? `5/5` datasets had mean adjacent claim-change fraction > 5% and mean adjacent Jaccard < 0.95.
5. Did shuffled current-window context weaken performance? Shuffled-window MAE was worse than correctly matched Dynamic EC on `5/5` datasets.
6. Did Dynamic EC close the gap to Frozen HxV? See `Dynamic EC CF1` vs `Frozen HxV` columns above; report is descriptive, this is not a required success criterion.
7. Were router_train OOF results consistent with router_val? OOF wins `3/5`, router_val wins `4/5`.
8. Did every integrity check pass? `True`.
9. Should the window-dependent EC direction CONTINUE or STOP? CONTINUE -- proceed to a frozen-method evaluation on untouched datasets.

## Interpretation discipline

The strongest claim supportable by this development experiment, if successful, is: under a matched competence tensor and matched assignment budget, expert-side H x V allocation appears to be a better routing operator than cell-side Top1 selection, and its specialization can depend meaningfully on the current forecasting window. This is NOT a claim of state of the art, compute savings, untouched external generalization, test improvement, or universal superiority of Expert Choice. All five datasets are development datasets.
