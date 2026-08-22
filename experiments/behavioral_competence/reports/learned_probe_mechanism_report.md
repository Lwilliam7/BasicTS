# Learned-Probe Mechanism Ablation: Probe vs. Decision Rule

Isolates whether the learned diagnostic probe's improvement comes from the probe itself or from the 0.60/0.30/0.10 rank decision rule. C, Fixed-D, and the learned probe are all evaluated under the IDENTICAL rank rule, using their already-saved, un-retrained competence predictions.

## Primary result table (router_val MAE / MSE)

| Dataset | Equal | C-Rank | FixedD-Rank | LearnedProbe-Softmax | LearnedProbe-Rank | C (ref) | Fixed-D (ref) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ETTh1 | 0.367265 | 0.367102 | 0.367910 | 0.368085 | 0.366729 | 0.368334 | 0.370864 |
| ETTh2 | 0.280878 | 0.280969 | 0.277822 | 0.277853 | 0.277202 | 0.286494 | 0.279710 |
| ETTm1 | 0.248161 | 0.249199 | 0.249618 | 0.252576 | 0.249857 | 0.250401 | 0.251473 |
| Weather | 0.160341 | 0.159772 | 0.160089 | 0.159234 | 0.159185 | 0.159672 | 0.160337 |
| Electricity | 0.214457 | 0.215616 | 0.216754 | 0.217415 | 0.213625 | 0.219244 | 0.222766 |

## LearnedProbe-Rank deltas

| Dataset | vs Equal | vs C | vs C-Rank | vs Fixed-D | vs FixedD-Rank | vs LearnedProbe-Softmax |
|---|---:|---:|---:|---:|---:|---:|
| ETTh1 | `-0.000536` | `-0.001605` | `-0.000373` | `-0.004135` | `-0.001181` | `-0.001356` |
| ETTh2 | `-0.003676` | `-0.009292` | `-0.003767` | `-0.002508` | `-0.000620` | `-0.000651` |
| ETTm1 | `+0.001696` | `-0.000544` | `+0.000658` | `-0.001615` | `+0.000239` | `-0.002718` |
| Weather | `-0.001156` | `-0.000486` | `-0.000587` | `-0.001152` | `-0.000904` | `-0.000048` |
| Electricity | `-0.000832` | `-0.005619` | `-0.001991` | `-0.009141` | `-0.003129` | `-0.003790` |

## The two most important comparisons

### A. LearnedProbe-Rank vs C-Rank (does the probe add info beyond window+forecast+disagreement, same decision rule?)

| Dataset | Δ MAE | 95% CI (IID) | block12 excl.0 | block24 excl.0 | block48 excl.0 | phase excl.0 |
|---|---:|---|---|---|---|---|
| ETTh1 | `-0.000373` | [-0.000901, +0.000163] | False | False | False | False |
| ETTh2 | `-0.003767` | [-0.004676, -0.002883] | True | True | True | True |
| ETTm1 | `+0.000658` | [+0.000432, +0.000881] | True | True | True | True |
| Weather | `-0.000587` | [-0.000732, -0.000444] | True | True | True | True |
| Electricity | `-0.001991` | [-0.002258, -0.001733] | True | True | True | True |

### B. LearnedProbe-Rank vs FixedD-Rank (does learning the probe beat hand-designed perturbations, same decision rule?)

| Dataset | Δ MAE | 95% CI (IID) | block12 excl.0 | block24 excl.0 | block48 excl.0 | phase excl.0 |
|---|---:|---|---|---|---|---|
| ETTh1 | `-0.001181` | [-0.001746, -0.000612] | True | True | True | True |
| ETTh2 | `-0.000620` | [-0.001272, -0.000009] | False | False | False | True |
| ETTm1 | `+0.000239` | [-0.000008, +0.000490] | False | False | False | False |
| Weather | `-0.000904` | [-0.001040, -0.000768] | True | True | True | True |
| Electricity | `-0.003129` | [-0.003399, -0.002870] | True | True | True | True |

## Full dependence-aware statistics

| Dataset | Comparison | Test | Mean Δ | 95% CI | Excludes zero |
|---|---|---|---:|---|---|
| ETTh1 | LearnedProbeRank_vs_CRank | iid_paired_bootstrap | `-0.000373` | [-0.000901, +0.000163] | False |
| ETTh1 | LearnedProbeRank_vs_CRank | block_bootstrap_len12 | `-0.000373` | [-0.001223, +0.000404] | False |
| ETTh1 | LearnedProbeRank_vs_CRank | block_bootstrap_len24 | `-0.000373` | [-0.001257, +0.000434] | False |
| ETTh1 | LearnedProbeRank_vs_CRank | block_bootstrap_len48 | `-0.000373` | [-0.001249, +0.000488] | False |
| ETTh1 | LearnedProbeRank_vs_CRank | every_12th_window_phase_bootstrap | `-0.000373` | [-0.000977, +0.000183] | False |
| ETTh1 | LearnedProbeRank_vs_FixedDRank | iid_paired_bootstrap | `-0.001181` | [-0.001746, -0.000612] | True |
| ETTh1 | LearnedProbeRank_vs_FixedDRank | block_bootstrap_len12 | `-0.001181` | [-0.002114, -0.000286] | True |
| ETTh1 | LearnedProbeRank_vs_FixedDRank | block_bootstrap_len24 | `-0.001181` | [-0.002249, -0.000226] | True |
| ETTh1 | LearnedProbeRank_vs_FixedDRank | block_bootstrap_len48 | `-0.001181` | [-0.002372, -0.000111] | True |
| ETTh1 | LearnedProbeRank_vs_FixedDRank | every_12th_window_phase_bootstrap | `-0.001181` | [-0.001788, -0.000608] | True |
| ETTh1 | LearnedProbeRank_vs_LearnedProbeSoftmax | iid_paired_bootstrap | `-0.001356` | [-0.001797, -0.000929] | True |
| ETTh1 | LearnedProbeRank_vs_LearnedProbeSoftmax | block_bootstrap_len12 | `-0.001356` | [-0.002227, -0.000503] | True |
| ETTh1 | LearnedProbeRank_vs_LearnedProbeSoftmax | block_bootstrap_len24 | `-0.001356` | [-0.002271, -0.000414] | True |
| ETTh1 | LearnedProbeRank_vs_LearnedProbeSoftmax | block_bootstrap_len48 | `-0.001356` | [-0.002319, -0.000408] | True |
| ETTh1 | LearnedProbeRank_vs_LearnedProbeSoftmax | every_12th_window_phase_bootstrap | `-0.001356` | [-0.001778, -0.000963] | True |
| ETTh1 | LearnedProbeRank_vs_Equal | iid_paired_bootstrap | `-0.000536` | [-0.001086, +0.000046] | False |
| ETTh1 | LearnedProbeRank_vs_Equal | block_bootstrap_len12 | `-0.000536` | [-0.001849, +0.000696] | False |
| ETTh1 | LearnedProbeRank_vs_Equal | block_bootstrap_len24 | `-0.000536` | [-0.002044, +0.000808] | False |
| ETTh1 | LearnedProbeRank_vs_Equal | block_bootstrap_len48 | `-0.000536` | [-0.002298, +0.000972] | False |
| ETTh1 | LearnedProbeRank_vs_Equal | every_12th_window_phase_bootstrap | `-0.000536` | [-0.000928, -0.000024] | True |
| ETTh2 | LearnedProbeRank_vs_CRank | iid_paired_bootstrap | `-0.003767` | [-0.004676, -0.002883] | True |
| ETTh2 | LearnedProbeRank_vs_CRank | block_bootstrap_len12 | `-0.003767` | [-0.005514, -0.002232] | True |
| ETTh2 | LearnedProbeRank_vs_CRank | block_bootstrap_len24 | `-0.003767` | [-0.005556, -0.002252] | True |
| ETTh2 | LearnedProbeRank_vs_CRank | block_bootstrap_len48 | `-0.003767` | [-0.005673, -0.002132] | True |
| ETTh2 | LearnedProbeRank_vs_CRank | every_12th_window_phase_bootstrap | `-0.003769` | [-0.004387, -0.003233] | True |
| ETTh2 | LearnedProbeRank_vs_FixedDRank | iid_paired_bootstrap | `-0.000620` | [-0.001272, -0.000009] | True |
| ETTh2 | LearnedProbeRank_vs_FixedDRank | block_bootstrap_len12 | `-0.000620` | [-0.001617, +0.000227] | False |
| ETTh2 | LearnedProbeRank_vs_FixedDRank | block_bootstrap_len24 | `-0.000620` | [-0.001620, +0.000180] | False |
| ETTh2 | LearnedProbeRank_vs_FixedDRank | block_bootstrap_len48 | `-0.000620` | [-0.001607, +0.000181] | False |
| ETTh2 | LearnedProbeRank_vs_FixedDRank | every_12th_window_phase_bootstrap | `-0.000621` | [-0.000995, -0.000235] | True |
| ETTh2 | LearnedProbeRank_vs_LearnedProbeSoftmax | iid_paired_bootstrap | `-0.000651` | [-0.001544, +0.000222] | False |
| ETTh2 | LearnedProbeRank_vs_LearnedProbeSoftmax | block_bootstrap_len12 | `-0.000651` | [-0.002112, +0.001037] | False |
| ETTh2 | LearnedProbeRank_vs_LearnedProbeSoftmax | block_bootstrap_len24 | `-0.000651` | [-0.002163, +0.001264] | False |
| ETTh2 | LearnedProbeRank_vs_LearnedProbeSoftmax | block_bootstrap_len48 | `-0.000651` | [-0.001871, +0.001250] | False |
| ETTh2 | LearnedProbeRank_vs_LearnedProbeSoftmax | every_12th_window_phase_bootstrap | `-0.000650` | [-0.001245, +0.000009] | False |
| ETTh2 | LearnedProbeRank_vs_Equal | iid_paired_bootstrap | `-0.003676` | [-0.004451, -0.002933] | True |
| ETTh2 | LearnedProbeRank_vs_Equal | block_bootstrap_len12 | `-0.003676` | [-0.005392, -0.002119] | True |
| ETTh2 | LearnedProbeRank_vs_Equal | block_bootstrap_len24 | `-0.003676` | [-0.005596, -0.002173] | True |
| ETTh2 | LearnedProbeRank_vs_Equal | block_bootstrap_len48 | `-0.003676` | [-0.005902, -0.002037] | True |
| ETTh2 | LearnedProbeRank_vs_Equal | every_12th_window_phase_bootstrap | `-0.003678` | [-0.004049, -0.003327] | True |
| ETTm1 | LearnedProbeRank_vs_CRank | iid_paired_bootstrap | `+0.000658` | [+0.000432, +0.000881] | True |
| ETTm1 | LearnedProbeRank_vs_CRank | block_bootstrap_len12 | `+0.000658` | [+0.000293, +0.001033] | True |
| ETTm1 | LearnedProbeRank_vs_CRank | block_bootstrap_len24 | `+0.000658` | [+0.000280, +0.001040] | True |
| ETTm1 | LearnedProbeRank_vs_CRank | block_bootstrap_len48 | `+0.000658` | [+0.000294, +0.001023] | True |
| ETTm1 | LearnedProbeRank_vs_CRank | every_12th_window_phase_bootstrap | `+0.000658` | [+0.000488, +0.000825] | True |
| ETTm1 | LearnedProbeRank_vs_FixedDRank | iid_paired_bootstrap | `+0.000239` | [-0.000008, +0.000490] | False |
| ETTm1 | LearnedProbeRank_vs_FixedDRank | block_bootstrap_len12 | `+0.000239` | [-0.000145, +0.000629] | False |
| ETTm1 | LearnedProbeRank_vs_FixedDRank | block_bootstrap_len24 | `+0.000239` | [-0.000172, +0.000635] | False |
| ETTm1 | LearnedProbeRank_vs_FixedDRank | block_bootstrap_len48 | `+0.000239` | [-0.000182, +0.000653] | False |
| ETTm1 | LearnedProbeRank_vs_FixedDRank | every_12th_window_phase_bootstrap | `+0.000239` | [-0.000018, +0.000455] | False |
| ETTm1 | LearnedProbeRank_vs_LearnedProbeSoftmax | iid_paired_bootstrap | `-0.002718` | [-0.002927, -0.002511] | True |
| ETTm1 | LearnedProbeRank_vs_LearnedProbeSoftmax | block_bootstrap_len12 | `-0.002718` | [-0.003122, -0.002317] | True |
| ETTm1 | LearnedProbeRank_vs_LearnedProbeSoftmax | block_bootstrap_len24 | `-0.002718` | [-0.003146, -0.002296] | True |
| ETTm1 | LearnedProbeRank_vs_LearnedProbeSoftmax | block_bootstrap_len48 | `-0.002718` | [-0.003162, -0.002292] | True |
| ETTm1 | LearnedProbeRank_vs_LearnedProbeSoftmax | every_12th_window_phase_bootstrap | `-0.002718` | [-0.002907, -0.002547] | True |
| ETTm1 | LearnedProbeRank_vs_Equal | iid_paired_bootstrap | `+0.001696` | [+0.001458, +0.001931] | True |
| ETTm1 | LearnedProbeRank_vs_Equal | block_bootstrap_len12 | `+0.001696` | [+0.001197, +0.002174] | True |
| ETTm1 | LearnedProbeRank_vs_Equal | block_bootstrap_len24 | `+0.001696` | [+0.001158, +0.002213] | True |
| ETTm1 | LearnedProbeRank_vs_Equal | block_bootstrap_len48 | `+0.001696` | [+0.001143, +0.002237] | True |
| ETTm1 | LearnedProbeRank_vs_Equal | every_12th_window_phase_bootstrap | `+0.001696` | [+0.001531, +0.001860] | True |
| Weather | LearnedProbeRank_vs_CRank | iid_paired_bootstrap | `-0.000587` | [-0.000732, -0.000444] | True |
| Weather | LearnedProbeRank_vs_CRank | block_bootstrap_len12 | `-0.000587` | [-0.000838, -0.000331] | True |
| Weather | LearnedProbeRank_vs_CRank | block_bootstrap_len24 | `-0.000587` | [-0.000863, -0.000305] | True |
| Weather | LearnedProbeRank_vs_CRank | block_bootstrap_len48 | `-0.000587` | [-0.000898, -0.000297] | True |
| Weather | LearnedProbeRank_vs_CRank | every_12th_window_phase_bootstrap | `-0.000587` | [-0.000750, -0.000425] | True |
| Weather | LearnedProbeRank_vs_FixedDRank | iid_paired_bootstrap | `-0.000904` | [-0.001040, -0.000768] | True |
| Weather | LearnedProbeRank_vs_FixedDRank | block_bootstrap_len12 | `-0.000904` | [-0.001155, -0.000665] | True |
| Weather | LearnedProbeRank_vs_FixedDRank | block_bootstrap_len24 | `-0.000904` | [-0.001186, -0.000652] | True |
| Weather | LearnedProbeRank_vs_FixedDRank | block_bootstrap_len48 | `-0.000904` | [-0.001219, -0.000642] | True |
| Weather | LearnedProbeRank_vs_FixedDRank | every_12th_window_phase_bootstrap | `-0.000904` | [-0.001022, -0.000784] | True |
| Weather | LearnedProbeRank_vs_LearnedProbeSoftmax | iid_paired_bootstrap | `-0.000048` | [-0.000277, +0.000193] | False |
| Weather | LearnedProbeRank_vs_LearnedProbeSoftmax | block_bootstrap_len12 | `-0.000048` | [-0.000636, +0.000627] | False |
| Weather | LearnedProbeRank_vs_LearnedProbeSoftmax | block_bootstrap_len24 | `-0.000048` | [-0.000721, +0.000818] | False |
| Weather | LearnedProbeRank_vs_LearnedProbeSoftmax | block_bootstrap_len48 | `-0.000048` | [-0.000821, +0.001053] | False |
| Weather | LearnedProbeRank_vs_LearnedProbeSoftmax | every_12th_window_phase_bootstrap | `-0.000048` | [-0.000231, +0.000102] | False |
| Weather | LearnedProbeRank_vs_Equal | iid_paired_bootstrap | `-0.001156` | [-0.001408, -0.000907] | True |
| Weather | LearnedProbeRank_vs_Equal | block_bootstrap_len12 | `-0.001156` | [-0.001938, -0.000485] | True |
| Weather | LearnedProbeRank_vs_Equal | block_bootstrap_len24 | `-0.001156` | [-0.002183, -0.000337] | True |
| Weather | LearnedProbeRank_vs_Equal | block_bootstrap_len48 | `-0.001156` | [-0.002463, -0.000206] | True |
| Weather | LearnedProbeRank_vs_Equal | every_12th_window_phase_bootstrap | `-0.001156` | [-0.001364, -0.000949] | True |
| Electricity | LearnedProbeRank_vs_CRank | iid_paired_bootstrap | `-0.001991` | [-0.002258, -0.001733] | True |
| Electricity | LearnedProbeRank_vs_CRank | block_bootstrap_len12 | `-0.001991` | [-0.002522, -0.001498] | True |
| Electricity | LearnedProbeRank_vs_CRank | block_bootstrap_len24 | `-0.001991` | [-0.002546, -0.001492] | True |
| Electricity | LearnedProbeRank_vs_CRank | block_bootstrap_len48 | `-0.001991` | [-0.002534, -0.001500] | True |
| Electricity | LearnedProbeRank_vs_CRank | every_12th_window_phase_bootstrap | `-0.001990` | [-0.002634, -0.001310] | True |
| Electricity | LearnedProbeRank_vs_FixedDRank | iid_paired_bootstrap | `-0.003129` | [-0.003399, -0.002870] | True |
| Electricity | LearnedProbeRank_vs_FixedDRank | block_bootstrap_len12 | `-0.003129` | [-0.003758, -0.002557] | True |
| Electricity | LearnedProbeRank_vs_FixedDRank | block_bootstrap_len24 | `-0.003129` | [-0.003779, -0.002536] | True |
| Electricity | LearnedProbeRank_vs_FixedDRank | block_bootstrap_len48 | `-0.003129` | [-0.003819, -0.002490] | True |
| Electricity | LearnedProbeRank_vs_FixedDRank | every_12th_window_phase_bootstrap | `-0.003128` | [-0.003931, -0.002232] | True |
| Electricity | LearnedProbeRank_vs_LearnedProbeSoftmax | iid_paired_bootstrap | `-0.003790` | [-0.004074, -0.003514] | True |
| Electricity | LearnedProbeRank_vs_LearnedProbeSoftmax | block_bootstrap_len12 | `-0.003790` | [-0.004277, -0.003322] | True |
| Electricity | LearnedProbeRank_vs_LearnedProbeSoftmax | block_bootstrap_len24 | `-0.003790` | [-0.004257, -0.003345] | True |
| Electricity | LearnedProbeRank_vs_LearnedProbeSoftmax | block_bootstrap_len48 | `-0.003790` | [-0.004292, -0.003329] | True |
| Electricity | LearnedProbeRank_vs_LearnedProbeSoftmax | every_12th_window_phase_bootstrap | `-0.003791` | [-0.004112, -0.003495] | True |
| Electricity | LearnedProbeRank_vs_Equal | iid_paired_bootstrap | `-0.000832` | [-0.001011, -0.000663] | True |
| Electricity | LearnedProbeRank_vs_Equal | block_bootstrap_len12 | `-0.000832` | [-0.001211, -0.000450] | True |
| Electricity | LearnedProbeRank_vs_Equal | block_bootstrap_len24 | `-0.000832` | [-0.001273, -0.000385] | True |
| Electricity | LearnedProbeRank_vs_Equal | block_bootstrap_len48 | `-0.000832` | [-0.001350, -0.000290] | True |
| Electricity | LearnedProbeRank_vs_Equal | every_12th_window_phase_bootstrap | `-0.000832` | [-0.001317, -0.000294] | True |

## Competence diagnostics (should reproduce prior experiments exactly)

| Dataset | Scorer | Spearman | Top-1 accuracy | Top-2 recall |
|---|---|---:|---:|---:|
| ETTh1 | C | 0.181 | 0.358 | 0.703 |
| ETTh1 | Fixed_D | 0.152 | 0.344 | 0.697 |
| ETTh1 | Learned_Probe | 0.241 | 0.390 | 0.719 |
| ETTh2 | C | 0.128 | 0.352 | 0.706 |
| ETTh2 | Fixed_D | 0.443 | 0.478 | 0.845 |
| ETTh2 | Learned_Probe | 0.397 | 0.460 | 0.847 |
| ETTm1 | C | 0.134 | 0.353 | 0.669 |
| ETTm1 | Fixed_D | 0.119 | 0.330 | 0.655 |
| ETTm1 | Learned_Probe | 0.161 | 0.326 | 0.640 |
| Weather | C | 0.190 | 0.399 | 0.721 |
| Weather | Fixed_D | 0.236 | 0.369 | 0.708 |
| Weather | Learned_Probe | 0.301 | 0.444 | 0.738 |
| Electricity | C | 0.517 | 0.545 | 0.863 |
| Electricity | Fixed_D | 0.320 | 0.366 | 0.807 |
| Electricity | Learned_Probe | 0.693 | 0.629 | 0.924 |

## Integrity

- **ETTh1**: PASS (predicted-excess-loss unmutated: True; reproduces saved C/Fixed-D/Learned-Probe-Softmax predictions: True; weights invariant to target corruption: True)
- **ETTh2**: PASS (predicted-excess-loss unmutated: True; reproduces saved C/Fixed-D/Learned-Probe-Softmax predictions: True; weights invariant to target corruption: True)
- **ETTm1**: PASS (predicted-excess-loss unmutated: True; reproduces saved C/Fixed-D/Learned-Probe-Softmax predictions: True; weights invariant to target corruption: True)
- **Weather**: PASS (predicted-excess-loss unmutated: True; reproduces saved C/Fixed-D/Learned-Probe-Softmax predictions: True; weights invariant to target corruption: True)
- **Electricity**: PASS (predicted-excess-loss unmutated: True; reproduces saved C/Fixed-D/Learned-Probe-Softmax predictions: True; weights invariant to target corruption: True)

## Interpretation

**1. Does LearnedProbe-Rank beat C-Rank?** By point estimate on 4/5 datasets; dependence-aware (block) significant on 3/5; significantly worse on 1/5.
**2. Does LearnedProbe-Rank beat FixedD-Rank?** By point estimate on 4/5 datasets; dependence-aware significant on 3/5.
**3. Does LearnedProbe-Rank still beat C (original softmax reference)?** See delta_vs_C column in the results table for each dataset.
**4. Was Rank weighting alone responsible for the improvement?** Partially -- see whether C-Rank and FixedD-Rank themselves already close most of the gap to LearnedProbe-Rank in the primary table.
**5. Does the learned probe provide incremental value after controlling for the decision rule?** Limited/no, based on the LearnedProbe-Rank vs C-Rank comparison.
**6. Are gains dependence-aware statistically supported?** 3/5 datasets vs C-Rank; 3/5 vs FixedD-Rank.
**7. Does the learned probe improve beyond ETTh2 and Electricity?** See per-dataset table above for ETTh1/ETTm1/Weather.
**8. Does ETTm1 remain non-harmful?** True.
**9. Does LearnedProbe-Rank beat Equal on multiple datasets?** By point estimate on 4/5 datasets.
**10. Is there enough evidence to freeze the method?** No.

## Decision

**LEARNED PROBE NOT JUSTIFIED**

- LearnedProbe-Rank beats C-Rank by point estimate on 4/5 (need >=3), with block-bootstrap significance on 3/5 (need >=2), and significantly HURTS 0-required on 1/5 (need 0).
- LearnedProbe-Rank beats FixedD-Rank with block-bootstrap significance on 3/5 datasets (extra evidence if >=2: True).
- ETTm1 non-harmful under LearnedProbe-Rank: True.
- Beats Equal by point estimate on 4/5 datasets.

## Hard rule compliance

```text
TEST SET ACCESSED: NO
TEST CACHE LOADED: NO
TEST METRICS COMPUTED: NO
```
