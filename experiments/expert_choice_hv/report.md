Final classification: MIXED_EXPERT_CHOICE

# Expert-Choice Horizon-Variable Routing (EC-HVR)

## Research question

If routing direction is reversed, so each frozen heterogeneous forecasting expert chooses the horizon x variable cells where it is most competent, does that produce better specialization than the usual cell-to-expert HxV allocation?

## Exact difference from existing HxV COSTAR

Existing HxV COSTAR assigns weights by asking each horizon-variable cell to look across experts. EC-HVR reverses that direction: each expert ranks all HxV cells using the same train-only competence score and claims a fixed-capacity set of cells. Cells may receive 0, 1, 2, or 3 experts; zero-claim cells fall back to the equal fixed ensemble.

## Validation results

| Dataset | Best Single | Equal | Frozen HxV | Token Top1 | EC CF1 | Token Top2 | EC CF2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ETTh1 | `0.379558` / `0.326528` | `0.367265` / `0.310530` | `0.366022` / `0.308672` | `0.376680` / `0.322693` | `0.375352` / `0.320141` | `0.369798` / `0.314286` | `0.375002` / `0.322928` |
| ETTh2 | `0.280957` / `0.171493` | `0.280878` / `0.171933` | `0.276898` / `0.167005` | `0.279132` / `0.169669` | `0.277764` / `0.168163` | `0.275230` / `0.165441` | `0.276706` / `0.167195` |
| ETTm1 | `0.261771` / `0.161838` | `0.248161` / `0.146694` | `0.250690` / `0.149956` | `0.262524` / `0.163088` | `0.253570` / `0.153139` | `0.256256` / `0.156230` | `0.259210` / `0.161534` |
| Weather | `0.164673` / `0.287468` | `0.160341` / `0.278815` | `0.159818` / `0.279092` | `0.164399` / `0.287168` | `0.160330` / `0.283485` | `0.162339` / `0.286809` | `0.159962` / `0.278532` |
| Electricity | `0.225385` / `0.135767` | `0.214457` / `0.117846` | `0.215355` / `0.122737` | `0.222761` / `0.131272` | `0.219015` / `0.120918` | `0.219130` / `0.127440` | `0.216481` / `0.119002` |

Values are `MAE / MSE`; all results are router-validation only.

## Matched Token Choice vs Expert Choice

| Dataset | EC CF1 - Token Top1 | EC CF2 - Token Top2 | EC CF1 - Frozen HxV |
|---|---:|---:|---:|
| ETTh1 | `-0.001327` | `+0.005205` | `+0.009330` |
| ETTh2 | `-0.001369` | `+0.001476` | `+0.000866` |
| ETTm1 | `-0.008954` | `+0.002954` | `+0.002881` |
| Weather | `-0.004069` | `-0.002377` | `+0.000513` |
| Electricity | `-0.003746` | `-0.002649` | `+0.003660` |

Negative deltas mean EC-HVR is better.

## Assignment/specialization behavior

| Dataset | EC CF1 fallback | EC CF1 avg Jaccard | EC CF2 fallback | EC CF2 avg Jaccard |
|---|---:|---:|---:|---:|
| ETTh1 | `0.214` | `0.138` | `0.012` | `0.525` |
| ETTh2 | `0.286` | `0.177` | `0.083` | `0.565` |
| ETTm1 | `0.369` | `0.328` | `0.000` | `0.510` |
| Weather | `0.349` | `0.253` | `0.119` | `0.622` |
| Electricity | `0.241` | `0.170` | `0.031` | `0.534` |

Complete claim distributions, per-expert coverage, pairwise overlaps, horizon fractions, and variable fractions are in `assignment_stats.csv`.

## Dependence-aware statistics

| Dataset | Comparison | Test | Mean delta | 95% CI | P(delta < 0) | CI excludes zero |
|---|---|---|---:|---|---:|---|
| ETTh1 | primary_ec_cf1_vs_token_top1 | block_len_24 | `-0.001327` | [`-0.006127`, `+0.003725`] | `0.704` | False |
| ETTh1 | primary_ec_cf1_vs_token_top1 | every_12th_phase | `-0.001327` | [`-0.001945`, `-0.000760`] | `1.000` | True |
| ETTh1 | secondary_ec_cf2_vs_token_top2 | block_len_24 | `+0.005205` | [`+0.001519`, `+0.009376`] | `0.003` | True |
| ETTh1 | secondary_ec_cf2_vs_token_top2 | every_12th_phase | `+0.005205` | [`+0.004086`, `+0.006290`] | `0.000` | True |
| ETTh1 | secondary_ec_cf1_vs_frozen_hv | block_len_24 | `+0.009330` | [`+0.006031`, `+0.013061`] | `0.000` | True |
| ETTh1 | secondary_ec_cf1_vs_frozen_hv | every_12th_phase | `+0.009330` | [`+0.008590`, `+0.009975`] | `0.000` | True |
| ETTh2 | primary_ec_cf1_vs_token_top1 | block_len_24 | `-0.001369` | [`-0.004823`, `+0.002852`] | `0.701` | False |
| ETTh2 | primary_ec_cf1_vs_token_top1 | every_12th_phase | `-0.001371` | [`-0.002412`, `-0.000449`] | `1.000` | True |
| ETTh2 | secondary_ec_cf2_vs_token_top2 | block_len_24 | `+0.001476` | [`-0.000488`, `+0.003588`] | `0.065` | False |
| ETTh2 | secondary_ec_cf2_vs_token_top2 | every_12th_phase | `+0.001477` | [`+0.000660`, `+0.002363`] | `0.000` | True |
| ETTh2 | secondary_ec_cf1_vs_frozen_hv | block_len_24 | `+0.000866` | [`-0.000598`, `+0.002419`] | `0.126` | False |
| ETTh2 | secondary_ec_cf1_vs_frozen_hv | every_12th_phase | `+0.000865` | [`+0.000355`, `+0.001435`] | `0.000` | True |
| ETTm1 | primary_ec_cf1_vs_token_top1 | block_len_24 | `-0.008954` | [`-0.010476`, `-0.007360`] | `1.000` | True |
| ETTm1 | primary_ec_cf1_vs_token_top1 | every_12th_phase | `-0.008954` | [`-0.009154`, `-0.008766`] | `1.000` | True |
| ETTm1 | secondary_ec_cf2_vs_token_top2 | block_len_24 | `+0.002954` | [`+0.001617`, `+0.004267`] | `0.000` | True |
| ETTm1 | secondary_ec_cf2_vs_token_top2 | every_12th_phase | `+0.002954` | [`+0.002172`, `+0.003649`] | `0.000` | True |
| ETTm1 | secondary_ec_cf1_vs_frozen_hv | block_len_24 | `+0.002881` | [`+0.001843`, `+0.003950`] | `0.000` | True |
| ETTm1 | secondary_ec_cf1_vs_frozen_hv | every_12th_phase | `+0.002881` | [`+0.002300`, `+0.003447`] | `0.000` | True |
| Weather | primary_ec_cf1_vs_token_top1 | block_len_24 | `-0.004069` | [`-0.005095`, `-0.002977`] | `1.000` | True |
| Weather | primary_ec_cf1_vs_token_top1 | every_12th_phase | `-0.004069` | [`-0.004467`, `-0.003692`] | `1.000` | True |
| Weather | secondary_ec_cf2_vs_token_top2 | block_len_24 | `-0.002377` | [`-0.003350`, `-0.001312`] | `1.000` | True |
| Weather | secondary_ec_cf2_vs_token_top2 | every_12th_phase | `-0.002377` | [`-0.002563`, `-0.002183`] | `1.000` | True |
| Weather | secondary_ec_cf1_vs_frozen_hv | block_len_24 | `+0.000513` | [`-0.000471`, `+0.001485`] | `0.141` | False |
| Weather | secondary_ec_cf1_vs_frozen_hv | every_12th_phase | `+0.000513` | [`+0.000346`, `+0.000684`] | `0.000` | True |
| Electricity | primary_ec_cf1_vs_token_top1 | block_len_24 | `-0.003746` | [`-0.005365`, `-0.002165`] | `1.000` | True |
| Electricity | primary_ec_cf1_vs_token_top1 | every_12th_phase | `-0.003744` | [`-0.005022`, `-0.002456`] | `1.000` | True |
| Electricity | secondary_ec_cf2_vs_token_top2 | block_len_24 | `-0.002649` | [`-0.004289`, `-0.000959`] | `0.999` | True |
| Electricity | secondary_ec_cf2_vs_token_top2 | every_12th_phase | `-0.002647` | [`-0.003785`, `-0.001537`] | `1.000` | True |
| Electricity | secondary_ec_cf1_vs_frozen_hv | block_len_24 | `+0.003660` | [`+0.002342`, `+0.005020`] | `0.000` | True |
| Electricity | secondary_ec_cf1_vs_frozen_hv | every_12th_phase | `+0.003661` | [`+0.002818`, `+0.004513`] | `0.000` | True |

## Integrity checks

- EC CF1 matched-budget wins: `5/5`.
- EC CF2 matched-budget wins: `2/5`.
- Nontrivial specialization rule passed: `True`.
- Router-val target corruption, targetless prediction, validation-order invariance, frozen allocation, and no-test checks passed for EC CF1 and EC CF2 on all datasets.

```text
TEST SET ACCESSED: NO
TEST CACHE LOADED: NO
TEST METRICS COMPUTED: NO
```

## Conclusion

The predeclared classification is `MIXED_EXPERT_CHOICE`. The scientific comparison is the matched-budget direction test, not whether any static ensemble can improve MAE in isolation.

### Did experts develop distinct HxV competence regions?

Yes, if measured by non-identical train-derived claim masks: EC claim overlaps were not near-perfect under the predeclared Jaccard rule. The detailed tables show which horizons and variables each expert claimed.

### Does expert-to-cell routing outperform matched cell-to-expert routing?

Partially. EC CF1 beat matched Token Top1 by MAE on all 5 datasets, with block-24 support on ETTm1, Weather, and Electricity. EC CF2 beat matched Token Top2 on only Weather and Electricity and regressed on ETTh1, ETTh2, and ETTm1, so the overall direction test is mixed rather than supported.

### Is the result strong enough to justify a second experiment with an input-dependent learned Expert-Choice router?

No. The result is `MIXED_EXPERT_CHOICE`, not `EXPERT_CHOICE_SUPPORTED`, and EC CF1 remained worse than Frozen HxV on every dataset.
