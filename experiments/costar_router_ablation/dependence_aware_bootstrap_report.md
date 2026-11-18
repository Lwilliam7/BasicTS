# Dependence-Aware Bootstrap Retest (Router Ablation)

Re-tests the five key router comparisons from `costar_router_ablation` using block bootstrap (lengths 12/24/48) and an every-12th non-overlapping-window phase analysis, alongside the original IID paired bootstrap for direct comparison. No cache was loaded, no test data was touched, and no prediction was changed -- this only re-analyzes the existing `router_ablation_per_window.csv`.

## Comparisons retested

| Label | Candidate | Baseline | Research question |
|---|---|---|---|
| A_causal_adaptation_helps | global_causal | equal_fixed | Does causal adaptation help at all? |
| B_horizon_specialization | horizon_only | global_causal | Does horizon specialization help? |
| C_variable_specialization | variable_only | global_causal | Does variable specialization help? |
| D_joint_hxv_vs_variable_only | hxv_causal | variable_only | Does the horizon axis add anything once variable is present? |
| E_global_adds_to_hxv | global_plus_hxv | hxv_causal | Does the global branch add value beyond HxV? |

## Results

| Dataset | Comparison | Test | Mean delta MAE | 95% CI | CI excludes zero |
|---|---|---|---:|---|---|
| ETTh1 | A_causal_adaptation_helps | iid_paired_bootstrap_original | `-0.001510` | [-0.002167, -0.000833] | True |
| ETTh1 | A_causal_adaptation_helps | block_bootstrap_len12 | `-0.001510` | [-0.003394, +0.000238] | False |
| ETTh1 | A_causal_adaptation_helps | block_bootstrap_len24 | `-0.001510` | [-0.003653, +0.000504] | False |
| ETTh1 | A_causal_adaptation_helps | block_bootstrap_len48 | `-0.001510` | [-0.004093, +0.000806] | False |
| ETTh1 | A_causal_adaptation_helps | every_12th_window_phase_bootstrap | `-0.001509` | [-0.001734, -0.001287] | True |
| ETTh1 | B_horizon_specialization | iid_paired_bootstrap_original | `-0.000117` | [-0.000218, -0.000017] | True |
| ETTh1 | B_horizon_specialization | block_bootstrap_len12 | `-0.000117` | [-0.000295, +0.000060] | False |
| ETTh1 | B_horizon_specialization | block_bootstrap_len24 | `-0.000117` | [-0.000310, +0.000066] | False |
| ETTh1 | B_horizon_specialization | block_bootstrap_len48 | `-0.000117` | [-0.000312, +0.000085] | False |
| ETTh1 | B_horizon_specialization | every_12th_window_phase_bootstrap | `-0.000117` | [-0.000270, +0.000034] | False |
| ETTh1 | C_variable_specialization | iid_paired_bootstrap_original | `-0.001670` | [-0.002074, -0.001266] | True |
| ETTh1 | C_variable_specialization | block_bootstrap_len12 | `-0.001670` | [-0.002743, -0.000639] | True |
| ETTh1 | C_variable_specialization | block_bootstrap_len24 | `-0.001670` | [-0.002817, -0.000567] | True |
| ETTh1 | C_variable_specialization | block_bootstrap_len48 | `-0.001670` | [-0.002779, -0.000631] | True |
| ETTh1 | C_variable_specialization | every_12th_window_phase_bootstrap | `-0.001671` | [-0.001802, -0.001539] | True |
| ETTh1 | D_joint_hxv_vs_variable_only | iid_paired_bootstrap_original | `-0.000136` | [-0.000317, +0.000046] | False |
| ETTh1 | D_joint_hxv_vs_variable_only | block_bootstrap_len12 | `-0.000136` | [-0.000452, +0.000181] | False |
| ETTh1 | D_joint_hxv_vs_variable_only | block_bootstrap_len24 | `-0.000136` | [-0.000440, +0.000159] | False |
| ETTh1 | D_joint_hxv_vs_variable_only | block_bootstrap_len48 | `-0.000136` | [-0.000436, +0.000146] | False |
| ETTh1 | D_joint_hxv_vs_variable_only | every_12th_window_phase_bootstrap | `-0.000136` | [-0.000462, +0.000187] | False |
| ETTh1 | E_global_adds_to_hxv | iid_paired_bootstrap_original | `-0.000315` | [-0.000532, -0.000107] | True |
| ETTh1 | E_global_adds_to_hxv | block_bootstrap_len12 | `-0.000315` | [-0.000807, +0.000181] | False |
| ETTh1 | E_global_adds_to_hxv | block_bootstrap_len24 | `-0.000315` | [-0.000876, +0.000238] | False |
| ETTh1 | E_global_adds_to_hxv | block_bootstrap_len48 | `-0.000315` | [-0.000912, +0.000260] | False |
| ETTh1 | E_global_adds_to_hxv | every_12th_window_phase_bootstrap | `-0.000315` | [-0.000515, -0.000115] | True |
| ETTh2 | A_causal_adaptation_helps | iid_paired_bootstrap_original | `-0.000725` | [-0.001204, -0.000252] | True |
| ETTh2 | A_causal_adaptation_helps | block_bootstrap_len12 | `-0.000725` | [-0.001948, +0.000351] | False |
| ETTh2 | A_causal_adaptation_helps | block_bootstrap_len24 | `-0.000725` | [-0.002118, +0.000420] | False |
| ETTh2 | A_causal_adaptation_helps | block_bootstrap_len48 | `-0.000725` | [-0.002464, +0.000577] | False |
| ETTh2 | A_causal_adaptation_helps | every_12th_window_phase_bootstrap | `-0.000725` | [-0.001046, -0.000371] | True |
| ETTh2 | B_horizon_specialization | iid_paired_bootstrap_original | `-0.000475` | [-0.000604, -0.000347] | True |
| ETTh2 | B_horizon_specialization | block_bootstrap_len12 | `-0.000475` | [-0.000694, -0.000274] | True |
| ETTh2 | B_horizon_specialization | block_bootstrap_len24 | `-0.000475` | [-0.000685, -0.000291] | True |
| ETTh2 | B_horizon_specialization | block_bootstrap_len48 | `-0.000475` | [-0.000706, -0.000280] | True |
| ETTh2 | B_horizon_specialization | every_12th_window_phase_bootstrap | `-0.000476` | [-0.000582, -0.000368] | True |
| ETTh2 | C_variable_specialization | iid_paired_bootstrap_original | `-0.002682` | [-0.003688, -0.001653] | True |
| ETTh2 | C_variable_specialization | block_bootstrap_len12 | `-0.002682` | [-0.004891, -0.000377] | True |
| ETTh2 | C_variable_specialization | block_bootstrap_len24 | `-0.002682` | [-0.004976, -0.000308] | True |
| ETTh2 | C_variable_specialization | block_bootstrap_len48 | `-0.002682` | [-0.004733, -0.000482] | True |
| ETTh2 | C_variable_specialization | every_12th_window_phase_bootstrap | `-0.002682` | [-0.003128, -0.002287] | True |
| ETTh2 | D_joint_hxv_vs_variable_only | iid_paired_bootstrap_original | `-0.001117` | [-0.001307, -0.000931] | True |
| ETTh2 | D_joint_hxv_vs_variable_only | block_bootstrap_len12 | `-0.001117` | [-0.001527, -0.000744] | True |
| ETTh2 | D_joint_hxv_vs_variable_only | block_bootstrap_len24 | `-0.001117` | [-0.001579, -0.000722] | True |
| ETTh2 | D_joint_hxv_vs_variable_only | block_bootstrap_len48 | `-0.001117` | [-0.001681, -0.000652] | True |
| ETTh2 | D_joint_hxv_vs_variable_only | every_12th_window_phase_bootstrap | `-0.001118` | [-0.001347, -0.000916] | True |
| ETTh2 | E_global_adds_to_hxv | iid_paired_bootstrap_original | `+0.000478` | [+0.000111, +0.000849] | True |
| ETTh2 | E_global_adds_to_hxv | block_bootstrap_len12 | `+0.000478` | [-0.000375, +0.001303] | False |
| ETTh2 | E_global_adds_to_hxv | block_bootstrap_len24 | `+0.000478` | [-0.000454, +0.001420] | False |
| ETTh2 | E_global_adds_to_hxv | block_bootstrap_len48 | `+0.000478` | [-0.000503, +0.001408] | False |
| ETTh2 | E_global_adds_to_hxv | every_12th_window_phase_bootstrap | `+0.000478` | [+0.000255, +0.000710] | True |

## Does the conclusion flip under any dependence-aware test?

| Dataset | Comparison | IID excludes zero | All block/phase tests agree with IID? | Flags |
|---|---|---|---|---|
| ETTh1 | A_causal_adaptation_helps | True | False | DISAGREEMENT: at least one dependence-aware test flips the conclusion |
| ETTh1 | B_horizon_specialization | True | False | DISAGREEMENT: at least one dependence-aware test flips the conclusion |
| ETTh1 | C_variable_specialization | True | True | none |
| ETTh1 | D_joint_hxv_vs_variable_only | False | True | none |
| ETTh1 | E_global_adds_to_hxv | True | False | DISAGREEMENT: at least one dependence-aware test flips the conclusion |
| ETTh2 | A_causal_adaptation_helps | True | False | DISAGREEMENT: at least one dependence-aware test flips the conclusion |
| ETTh2 | B_horizon_specialization | True | True | none |
| ETTh2 | C_variable_specialization | True | True | none |
| ETTh2 | D_joint_hxv_vs_variable_only | True | True | none |
| ETTh2 | E_global_adds_to_hxv | True | False | DISAGREEMENT: at least one dependence-aware test flips the conclusion |

## Hard rule compliance

```text
TEST SET ACCESSED: NO
TEST CACHE LOADED: NO
TEST METRICS COMPUTED: NO
```
