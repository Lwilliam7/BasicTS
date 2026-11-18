Expert-Choice variant sweep on Window-Dependent EC -- ETTh1 only

```text
TEST SET ACCESSED: NO
ROUTER_VAL ACCESSED: NO
OTHER DATASETS ACCESSED: NO
```

OOF scored windows: 4436
Frozen dense ensemble (equal average, no routing): MAE `0.345591`, MSE `0.260821`

## Matched Dynamic Token Choice (per scoring variant)

| Scoring | Token Choice MAE | Token Choice MSE | Fallback rate |
|---|---:|---:|---:|
| existing | `0.351491` | `0.273037` | `0.0000` |
| expert_relative | `0.360539` | `0.285014` | `0.0000` |

## Full 12-configuration grid, ranked by OOF MAE

| Rank | Config | CF | Assignment | Scoring | MAE | MSE | Fallback | 0-claim% | 1-claim% | 2-claim% | >2-claim% | vs Token | vs EC baseline | vs Frozen |
|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `cf2.0_unrestricted_existing` | 2.0 | unrestricted | existing | `0.349150` | `0.269491` | `0.0000` | `0.00` | `19.60` | `60.79` | `19.60` | `-0.002341` | `-0.002774` | `+0.003559` |
| 2 | `cf0.5_unrestricted_existing` | 0.5 | unrestricted | existing | `0.349727` | `0.270366` | `0.5152` | `51.52` | `46.95` | `1.52` | `0.00` | `-0.001765` | `-0.002198` | `+0.004136` |
| 3 | `cf0.5_max2_existing` | 0.5 | max2 | existing | `0.349727` | `0.270366` | `0.5152` | `51.52` | `46.95` | `1.52` | `0.00` | `-0.001765` | `-0.002198` | `+0.004136` |
| 4 | `cf2.0_max2_expert_relative` | 2.0 | max2 | expert_relative | `0.349862` | `0.267659` | `0.0000` | `0.00` | `3.18` | `96.82` | `0.00` | `-0.010677` | `-0.002062` | `+0.004271` |
| 5 | `cf2.0_unrestricted_expert_relative` | 2.0 | unrestricted | expert_relative | `0.351479` | `0.273660` | `0.0000` | `0.00` | `20.77` | `58.47` | `20.77` | `-0.009060` | `-0.000445` | `+0.005888` |
| 6 | `cf2.0_max2_existing` | 2.0 | max2 | existing | `0.351819` | `0.271380` | `0.0000` | `0.00` | `3.80` | `96.20` | `0.00` | `+0.000327` | `-0.000106` | `+0.006228` |
| 7 | `cf1.0_unrestricted_existing` *(baseline)* | 1.0 | unrestricted | existing | `0.351924` | `0.273279` | `0.1392` | `13.92` | `72.16` | `13.92` | `0.00` | `+0.000433` | `+0.000000` | `+0.006333` |
| 8 | `cf1.0_max2_existing` | 1.0 | max2 | existing | `0.351924` | `0.273279` | `0.1392` | `13.92` | `72.16` | `13.92` | `0.00` | `+0.000433` | `+0.000000` | `+0.006333` |
| 9 | `cf0.5_unrestricted_expert_relative` | 0.5 | unrestricted | expert_relative | `0.352907` | `0.275634` | `0.5097` | `50.97` | `48.06` | `0.97` | `0.00` | `-0.007633` | `+0.000982` | `+0.007316` |
| 10 | `cf0.5_max2_expert_relative` | 0.5 | max2 | expert_relative | `0.352907` | `0.275634` | `0.5097` | `50.97` | `48.06` | `0.97` | `0.00` | `-0.007633` | `+0.000982` | `+0.007316` |
| 11 | `cf1.0_unrestricted_expert_relative` | 1.0 | unrestricted | expert_relative | `0.356171` | `0.280013` | `0.1243` | `12.43` | `75.14` | `12.43` | `0.00` | `-0.004368` | `+0.004247` | `+0.010580` |
| 12 | `cf1.0_max2_expert_relative` | 1.0 | max2 | expert_relative | `0.356171` | `0.280013` | `0.1243` | `12.43` | `75.14` | `12.43` | `0.00` | `-0.004368` | `+0.004247` | `+0.010580` |

## Expert utilization (actual claims / intended capacity)

| Config | PatchTST | iTransformer | TimesNet |
|---|---:|---:|---:|
| `cf0.5_unrestricted_existing` | `100.0%` | `100.0%` | `100.0%` |
| `cf0.5_unrestricted_expert_relative` | `100.0%` | `100.0%` | `100.0%` |
| `cf0.5_max2_existing` | `100.0%` | `100.0%` | `100.0%` |
| `cf0.5_max2_expert_relative` | `100.0%` | `100.0%` | `100.0%` |
| `cf1.0_unrestricted_existing` | `100.0%` | `100.0%` | `100.0%` |
| `cf1.0_unrestricted_expert_relative` | `100.0%` | `100.0%` | `100.0%` |
| `cf1.0_max2_existing` | `100.0%` | `100.0%` | `100.0%` |
| `cf1.0_max2_expert_relative` | `100.0%` | `100.0%` | `100.0%` |
| `cf2.0_unrestricted_existing` | `100.0%` | `100.0%` | `100.0%` |
| `cf2.0_unrestricted_expert_relative` | `100.0%` | `100.0%` | `100.0%` |
| `cf2.0_max2_existing` | `99.4%` | `99.7%` | `95.2%` |
| `cf2.0_max2_expert_relative` | `96.8%` | `98.9%` | `99.6%` |

## Question 1: does capacity factor change OOF performance?

MAE by CF (unrestricted/existing): {"0.5": 0.34972673654556274, "1.0": 0.35192441940307617, "2.0": 0.3491500914096832}
Best CF: `2.0` (CF=1.0 is best: `False`)

## Question 2: does max-2-per-cell beat unrestricted or forcing one expert?

max2 beats unrestricted on 1/6 matched (CF, scoring) pairs.
Per-pair deltas (max2 - unrestricted MAE): {"cf0.5_existing": 0.0, "cf0.5_expert_relative": 0.0, "cf1.0_existing": 0.0, "cf1.0_expert_relative": 0.0, "cf2.0_existing": 0.0026687979698181152, "cf2.0_expert_relative": -0.00161704421043396}
Context: Conflict-Resolved Expert Choice (2026-08-30) already tested forcing exactly ONE expert per cell on this same window-dependent-EC family and lost on 0/5 datasets by 0.0012-0.0022 MAE (experiments/conflict_resolved_expert_choice_hv/report.md) -- reused as context, not recomputed here.

## Question 3: does expert-relative softmax scoring help?

expert_relative beats existing on 1/6 matched (CF, assignment) pairs.
Per-pair deltas (expert_relative - existing MAE): {"cf0.5_unrestricted": 0.003179997205734253, "cf0.5_max2": 0.003179997205734253, "cf1.0_unrestricted": 0.0042465925216674805, "cf1.0_max2": 0.0042465925216674805, "cf2.0_unrestricted": 0.00232890248298645, "cf2.0_max2": -0.001956939697265625}

## Question 4: single best preregistered configuration

Best config by OOF MAE: `cf2.0_unrestricted_existing` (MAE `0.349150`); is the baseline itself: `False`.

Block-24 CI of best vs baseline: mean_delta=`-0.002774`, CI95=[`-0.003227`, `-0.002338`], excludes_zero=`True`

