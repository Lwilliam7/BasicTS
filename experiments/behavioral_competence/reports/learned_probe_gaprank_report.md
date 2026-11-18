# Learned-Probe GapRank: Loss-Gap-Weighted Pairwise Ranking Loss

Tests one principled modification motivated by the ETTm1 failure analysis: replaces the plain pairwise hinge ranking loss with a loss-gap-weighted version (`gap_weight = clip(|actual_i - actual_j| / router_train_gap_scale, 0.25, 4.0)`), so high-stakes comparisons are penalized more and near-tied comparisons less. Everything else (experts, ProbeGenerator architecture, epsilon, constraints, competence features, scorer architecture, 0.60/0.30/0.10 rank rule, expert pool, splits, the 0.25 ranking-loss coefficient) is unchanged.

Frozen-probe control (`LearnedProbe-FrozenProbe-GapRank`): included.

## Primary result table (router_val MAE / MSE)

| Dataset | Equal | C-Rank | FixedD-Rank | Original LearnedProbe-Rank | LearnedProbe-GapRank | FrozenProbe-GapRank |
|---|---:|---:|---:|---:|---:|---:|
| ETTh1 | 0.367265 | 0.367102 | 0.367910 | 0.366729 | 0.367012 | 0.368118 |
| ETTh2 | 0.280878 | 0.280969 | 0.277822 | 0.277202 | 0.277742 | 0.279693 |
| ETTm1 | 0.248161 | 0.249199 | 0.249618 | 0.249857 | 0.249238 | 0.249553 |
| Weather | 0.160341 | 0.159772 | 0.160089 | 0.159185 | 0.159489 | 0.159495 |
| Electricity | 0.214457 | 0.215616 | 0.216754 | 0.213625 | 0.214063 | 0.214068 |

## LearnedProbe-GapRank deltas

| Dataset | vs Equal | vs C-Rank | vs FixedD-Rank | vs Original LearnedProbe-Rank |
|---|---:|---:|---:|---:|
| ETTh1 | `-0.000253` | `-0.000090` | `-0.000898` | `+0.000283` |
| ETTh2 | `-0.003137` | `-0.003227` | `-0.000080` | `+0.000540` |
| ETTm1 | `+0.001076` | `+0.000039` | `-0.000381` | `-0.000620` |
| Weather | `-0.000852` | `-0.000283` | `-0.000600` | `+0.000304` |
| Electricity | `-0.000394` | `-0.001553` | `-0.002691` | `+0.000438` |

## Competence diagnostics

| Dataset | Method | Spearman | Pairwise acc | Top-1 acc | Top-2 recall | Mean rank of true best | Cost-weighted err/pair | Top-1 mistake rate | Mean regret (all windows) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ETTh1 | C_Rank | 0.181 | 0.573 | 0.358 | 0.703 | 0.938 | 0.023201 | 0.642 | 0.035472 |
| ETTh1 | FixedD_Rank | 0.152 | 0.559 | 0.344 | 0.697 | 0.959 | 0.024968 | 0.656 | 0.037703 |
| ETTh1 | LearnedProbe_Rank | 0.241 | 0.606 | 0.390 | 0.719 | 0.891 | 0.021761 | 0.610 | 0.031973 |
| ETTh1 | LearnedProbe_GapRank | 0.227 | 0.600 | 0.384 | 0.709 | 0.907 | 0.022209 | 0.616 | 0.032590 |
| ETTh1 | LearnedProbe_FrozenProbe_GapRank | 0.189 | 0.573 | 0.342 | 0.681 | 0.977 | 0.024477 | 0.658 | 0.035481 |
| ETTh2 | C_Rank | 0.128 | 0.543 | 0.352 | 0.706 | 0.941 | 0.019816 | 0.648 | 0.029570 |
| ETTh2 | FixedD_Rank | 0.443 | 0.691 | 0.478 | 0.845 | 0.677 | 0.011495 | 0.522 | 0.018593 |
| ETTh2 | LearnedProbe_Rank | 0.397 | 0.691 | 0.460 | 0.847 | 0.693 | 0.010588 | 0.540 | 0.016682 |
| ETTh2 | LearnedProbe_GapRank | 0.364 | 0.672 | 0.454 | 0.834 | 0.713 | 0.012005 | 0.546 | 0.018144 |
| ETTh2 | LearnedProbe_FrozenProbe_GapRank | 0.130 | 0.598 | 0.411 | 0.772 | 0.817 | 0.018201 | 0.589 | 0.025513 |
| ETTm1 | C_Rank | 0.134 | 0.552 | 0.353 | 0.669 | 0.978 | 0.021410 | 0.647 | 0.031372 |
| ETTm1 | FixedD_Rank | 0.119 | 0.538 | 0.330 | 0.655 | 1.015 | 0.022748 | 0.670 | 0.033258 |
| ETTm1 | LearnedProbe_Rank | 0.161 | 0.558 | 0.326 | 0.640 | 1.033 | 0.021749 | 0.674 | 0.031193 |
| ETTm1 | LearnedProbe_GapRank | 0.141 | 0.561 | 0.338 | 0.669 | 0.994 | 0.021430 | 0.662 | 0.031828 |
| ETTm1 | LearnedProbe_FrozenProbe_GapRank | 0.170 | 0.560 | 0.330 | 0.644 | 1.026 | 0.021111 | 0.670 | 0.030801 |
| Weather | C_Rank | 0.190 | 0.600 | 0.399 | 0.721 | 0.881 | 0.010205 | 0.601 | 0.015130 |
| Weather | FixedD_Rank | 0.236 | 0.583 | 0.369 | 0.708 | 0.923 | 0.010885 | 0.631 | 0.016400 |
| Weather | LearnedProbe_Rank | 0.301 | 0.640 | 0.444 | 0.738 | 0.817 | 0.008853 | 0.556 | 0.012677 |
| Weather | LearnedProbe_GapRank | 0.271 | 0.637 | 0.441 | 0.730 | 0.829 | 0.009082 | 0.559 | 0.013072 |
| Weather | LearnedProbe_FrozenProbe_GapRank | 0.307 | 0.634 | 0.434 | 0.722 | 0.844 | 0.009097 | 0.566 | 0.012980 |
| Electricity | C_Rank | 0.517 | 0.757 | 0.545 | 0.863 | 0.593 | 0.005590 | 0.455 | 0.008065 |
| Electricity | FixedD_Rank | 0.320 | 0.643 | 0.366 | 0.807 | 0.827 | 0.008625 | 0.634 | 0.013770 |
| Electricity | LearnedProbe_Rank | 0.693 | 0.809 | 0.629 | 0.924 | 0.446 | 0.003355 | 0.371 | 0.005124 |
| Electricity | LearnedProbe_GapRank | 0.677 | 0.812 | 0.655 | 0.908 | 0.437 | 0.003679 | 0.345 | 0.005349 |
| Electricity | LearnedProbe_FrozenProbe_GapRank | 0.674 | 0.813 | 0.673 | 0.906 | 0.421 | 0.003711 | 0.327 | 0.005094 |

## ETTm1-specific diagnostic (Original vs GapRank)

- Windows where GapRank is beneficial vs original: 2106; harmful: 1745; neutral: 7562
- Top-1 accuracy: 0.326 -> 0.338
- Top-2 recall: 0.640 -> 0.669
- Pairwise accuracy: 0.558 -> 0.561
- Cost-weighted ranking error/pair: 0.021749 -> 0.021430
- Mean top-1-mistake regret (all windows): 0.031193 -> 0.031828
- MAE: original=0.249857, GapRank=0.249238, C-Rank=0.249199
- Still regresses vs C-Rank under GapRank: **True**

## Cross-dataset high-stakes analysis (tertiles by TRUE expert separation, diagnostic only)

| Dataset | Tercile | Windows | MAE Original | MAE GapRank | Delta |
|---|---|---:|---:|---:|---:|
| ETTh1 | low_separation | 925 | 0.329791 | 0.330078 | `+0.000287` |
| ETTh1 | mid_separation | 924 | 0.339170 | 0.338973 | `-0.000197` |
| ETTh1 | high_separation | 924 | 0.431266 | 0.432025 | `+0.000759` |
| ETTh2 | low_separation | 205 | 0.266604 | 0.266580 | `-0.000024` |
| ETTh2 | mid_separation | 204 | 0.268083 | 0.268796 | `+0.000713` |
| ETTh2 | high_separation | 204 | 0.296971 | 0.297904 | `+0.000932` |
| ETTm1 | low_separation | 3805 | 0.212584 | 0.212751 | `+0.000168` |
| ETTm1 | mid_separation | 3804 | 0.231353 | 0.231773 | `+0.000420` |
| ETTm1 | high_separation | 3804 | 0.305645 | 0.303198 | `-0.002447` |
| Weather | low_separation | 3478 | 0.136689 | 0.136712 | `+0.000023` |
| Weather | mid_separation | 3477 | 0.147065 | 0.147028 | `-0.000036` |
| Weather | high_separation | 3477 | 0.193808 | 0.194734 | `+0.000926` |
| Electricity | low_separation | 1718 | 0.194522 | 0.194709 | `+0.000187` |
| Electricity | mid_separation | 1718 | 0.202303 | 0.202463 | `+0.000160` |
| Electricity | high_separation | 1718 | 0.244049 | 0.245017 | `+0.000968` |

## Dependence-aware statistics

| Dataset | Comparison | Test | Mean Δ | 95% CI | Excludes zero |
|---|---|---|---:|---|---|
| ETTh1 | GapRank_vs_LearnedProbeRank | iid_paired_bootstrap | `+0.000283` | [-0.000046, +0.000627] | False |
| ETTh1 | GapRank_vs_LearnedProbeRank | block_bootstrap_len12 | `+0.000283` | [-0.000185, +0.000768] | False |
| ETTh1 | GapRank_vs_LearnedProbeRank | block_bootstrap_len24 | `+0.000283` | [-0.000208, +0.000824] | False |
| ETTh1 | GapRank_vs_LearnedProbeRank | block_bootstrap_len48 | `+0.000283` | [-0.000248, +0.000839] | False |
| ETTh1 | GapRank_vs_LearnedProbeRank | every_12th_window_phase_bootstrap | `+0.000283` | [-0.000037, +0.000556] | False |
| ETTh1 | GapRank_vs_CRank | iid_paired_bootstrap | `-0.000090` | [-0.000586, +0.000433] | False |
| ETTh1 | GapRank_vs_CRank | block_bootstrap_len12 | `-0.000090` | [-0.000837, +0.000603] | False |
| ETTh1 | GapRank_vs_CRank | block_bootstrap_len24 | `-0.000090` | [-0.000894, +0.000654] | False |
| ETTh1 | GapRank_vs_CRank | block_bootstrap_len48 | `-0.000090` | [-0.000894, +0.000705] | False |
| ETTh1 | GapRank_vs_CRank | every_12th_window_phase_bootstrap | `-0.000090` | [-0.000693, +0.000493] | False |
| ETTh1 | GapRank_vs_FixedDRank | iid_paired_bootstrap | `-0.000898` | [-0.001443, -0.000351] | True |
| ETTh1 | GapRank_vs_FixedDRank | block_bootstrap_len12 | `-0.000898` | [-0.001707, -0.000120] | True |
| ETTh1 | GapRank_vs_FixedDRank | block_bootstrap_len24 | `-0.000898` | [-0.001798, -0.000091] | True |
| ETTh1 | GapRank_vs_FixedDRank | block_bootstrap_len48 | `-0.000898` | [-0.001927, -0.000023] | True |
| ETTh1 | GapRank_vs_FixedDRank | every_12th_window_phase_bootstrap | `-0.000898` | [-0.001470, -0.000373] | True |
| ETTh1 | GapRank_vs_Equal | iid_paired_bootstrap | `-0.000253` | [-0.000795, +0.000325] | False |
| ETTh1 | GapRank_vs_Equal | block_bootstrap_len12 | `-0.000253` | [-0.001595, +0.001017] | False |
| ETTh1 | GapRank_vs_Equal | block_bootstrap_len24 | `-0.000253` | [-0.001802, +0.001171] | False |
| ETTh1 | GapRank_vs_Equal | block_bootstrap_len48 | `-0.000253` | [-0.002047, +0.001328] | False |
| ETTh1 | GapRank_vs_Equal | every_12th_window_phase_bootstrap | `-0.000253` | [-0.000589, +0.000132] | False |
| ETTh1 | FrozenProbeGapRank_vs_GapRank | iid_paired_bootstrap | `+0.001106` | [+0.000619, +0.001567] | True |
| ETTh1 | FrozenProbeGapRank_vs_GapRank | block_bootstrap_len12 | `+0.001106` | [+0.000390, +0.001900] | True |
| ETTh1 | FrozenProbeGapRank_vs_GapRank | block_bootstrap_len24 | `+0.001106` | [+0.000328, +0.001974] | True |
| ETTh1 | FrozenProbeGapRank_vs_GapRank | block_bootstrap_len48 | `+0.001106` | [+0.000316, +0.002069] | True |
| ETTh1 | FrozenProbeGapRank_vs_GapRank | every_12th_window_phase_bootstrap | `+0.001106` | [+0.000774, +0.001448] | True |
| ETTh1 | FrozenProbeGapRank_vs_LearnedProbeRank | iid_paired_bootstrap | `+0.001389` | [+0.000879, +0.001898] | True |
| ETTh1 | FrozenProbeGapRank_vs_LearnedProbeRank | block_bootstrap_len12 | `+0.001389` | [+0.000555, +0.002291] | True |
| ETTh1 | FrozenProbeGapRank_vs_LearnedProbeRank | block_bootstrap_len24 | `+0.001389` | [+0.000499, +0.002396] | True |
| ETTh1 | FrozenProbeGapRank_vs_LearnedProbeRank | block_bootstrap_len48 | `+0.001389` | [+0.000432, +0.002464] | True |
| ETTh1 | FrozenProbeGapRank_vs_LearnedProbeRank | every_12th_window_phase_bootstrap | `+0.001389` | [+0.000990, +0.001767] | True |
| ETTh2 | GapRank_vs_LearnedProbeRank | iid_paired_bootstrap | `+0.000540` | [+0.000176, +0.000920] | True |
| ETTh2 | GapRank_vs_LearnedProbeRank | block_bootstrap_len12 | `+0.000540` | [+0.000118, +0.000964] | True |
| ETTh2 | GapRank_vs_LearnedProbeRank | block_bootstrap_len24 | `+0.000540` | [+0.000133, +0.000987] | True |
| ETTh2 | GapRank_vs_LearnedProbeRank | block_bootstrap_len48 | `+0.000540` | [+0.000138, +0.000967] | True |
| ETTh2 | GapRank_vs_LearnedProbeRank | every_12th_window_phase_bootstrap | `+0.000538` | [+0.000136, +0.000918] | True |
| ETTh2 | GapRank_vs_CRank | iid_paired_bootstrap | `-0.003227` | [-0.004218, -0.002234] | True |
| ETTh2 | GapRank_vs_CRank | block_bootstrap_len12 | `-0.003227` | [-0.005105, -0.001563] | True |
| ETTh2 | GapRank_vs_CRank | block_bootstrap_len24 | `-0.003227` | [-0.005137, -0.001520] | True |
| ETTh2 | GapRank_vs_CRank | block_bootstrap_len48 | `-0.003227` | [-0.005266, -0.001457] | True |
| ETTh2 | GapRank_vs_CRank | every_12th_window_phase_bootstrap | `-0.003231` | [-0.003968, -0.002540] | True |
| ETTh2 | GapRank_vs_FixedDRank | iid_paired_bootstrap | `-0.000080` | [-0.000746, +0.000565] | False |
| ETTh2 | GapRank_vs_FixedDRank | block_bootstrap_len12 | `-0.000080` | [-0.001112, +0.000829] | False |
| ETTh2 | GapRank_vs_FixedDRank | block_bootstrap_len24 | `-0.000080` | [-0.001109, +0.000830] | False |
| ETTh2 | GapRank_vs_FixedDRank | block_bootstrap_len48 | `-0.000080` | [-0.001142, +0.000843] | False |
| ETTh2 | GapRank_vs_FixedDRank | every_12th_window_phase_bootstrap | `-0.000083` | [-0.000600, +0.000453] | False |
| ETTh2 | GapRank_vs_Equal | iid_paired_bootstrap | `-0.003137` | [-0.003937, -0.002339] | True |
| ETTh2 | GapRank_vs_Equal | block_bootstrap_len12 | `-0.003137` | [-0.004906, -0.001527] | True |
| ETTh2 | GapRank_vs_Equal | block_bootstrap_len24 | `-0.003137` | [-0.005163, -0.001514] | True |
| ETTh2 | GapRank_vs_Equal | block_bootstrap_len48 | `-0.003137` | [-0.005533, -0.001348] | True |
| ETTh2 | GapRank_vs_Equal | every_12th_window_phase_bootstrap | `-0.003140` | [-0.003843, -0.002461] | True |
| ETTh2 | FrozenProbeGapRank_vs_GapRank | iid_paired_bootstrap | `+0.001951` | [+0.000992, +0.002890] | True |
| ETTh2 | FrozenProbeGapRank_vs_GapRank | block_bootstrap_len12 | `+0.001951` | [+0.000624, +0.003465] | True |
| ETTh2 | FrozenProbeGapRank_vs_GapRank | block_bootstrap_len24 | `+0.001951` | [+0.000813, +0.003488] | True |
| ETTh2 | FrozenProbeGapRank_vs_GapRank | block_bootstrap_len48 | `+0.001951` | [+0.000727, +0.003374] | True |
| ETTh2 | FrozenProbeGapRank_vs_GapRank | every_12th_window_phase_bootstrap | `+0.001954` | [+0.001346, +0.002577] | True |
| ETTh2 | FrozenProbeGapRank_vs_LearnedProbeRank | iid_paired_bootstrap | `+0.002491` | [+0.001596, +0.003403] | True |
| ETTh2 | FrozenProbeGapRank_vs_LearnedProbeRank | block_bootstrap_len12 | `+0.002491` | [+0.001095, +0.004082] | True |
| ETTh2 | FrozenProbeGapRank_vs_LearnedProbeRank | block_bootstrap_len24 | `+0.002491` | [+0.001256, +0.004133] | True |
| ETTh2 | FrozenProbeGapRank_vs_LearnedProbeRank | block_bootstrap_len48 | `+0.002491` | [+0.001177, +0.003991] | True |
| ETTh2 | FrozenProbeGapRank_vs_LearnedProbeRank | every_12th_window_phase_bootstrap | `+0.002492` | [+0.001953, +0.003056] | True |
| ETTm1 | GapRank_vs_LearnedProbeRank | iid_paired_bootstrap | `-0.000620` | [-0.000809, -0.000431] | True |
| ETTm1 | GapRank_vs_LearnedProbeRank | block_bootstrap_len12 | `-0.000620` | [-0.000958, -0.000287] | True |
| ETTm1 | GapRank_vs_LearnedProbeRank | block_bootstrap_len24 | `-0.000620` | [-0.000973, -0.000264] | True |
| ETTm1 | GapRank_vs_LearnedProbeRank | block_bootstrap_len48 | `-0.000620` | [-0.000982, -0.000252] | True |
| ETTm1 | GapRank_vs_LearnedProbeRank | every_12th_window_phase_bootstrap | `-0.000620` | [-0.000737, -0.000503] | True |
| ETTm1 | GapRank_vs_CRank | iid_paired_bootstrap | `+0.000039` | [-0.000210, +0.000286] | False |
| ETTm1 | GapRank_vs_CRank | block_bootstrap_len12 | `+0.000039` | [-0.000411, +0.000503] | False |
| ETTm1 | GapRank_vs_CRank | block_bootstrap_len24 | `+0.000039` | [-0.000436, +0.000520] | False |
| ETTm1 | GapRank_vs_CRank | block_bootstrap_len48 | `+0.000039` | [-0.000446, +0.000533] | False |
| ETTm1 | GapRank_vs_CRank | every_12th_window_phase_bootstrap | `+0.000039` | [-0.000162, +0.000256] | False |
| ETTm1 | GapRank_vs_FixedDRank | iid_paired_bootstrap | `-0.000381` | [-0.000655, -0.000100] | True |
| ETTm1 | GapRank_vs_FixedDRank | block_bootstrap_len12 | `-0.000381` | [-0.000846, +0.000093] | False |
| ETTm1 | GapRank_vs_FixedDRank | block_bootstrap_len24 | `-0.000381` | [-0.000877, +0.000113] | False |
| ETTm1 | GapRank_vs_FixedDRank | block_bootstrap_len48 | `-0.000381` | [-0.000905, +0.000128] | False |
| ETTm1 | GapRank_vs_FixedDRank | every_12th_window_phase_bootstrap | `-0.000381` | [-0.000658, -0.000117] | True |
| ETTm1 | GapRank_vs_Equal | iid_paired_bootstrap | `+0.001076` | [+0.000825, +0.001323] | True |
| ETTm1 | GapRank_vs_Equal | block_bootstrap_len12 | `+0.001076` | [+0.000574, +0.001562] | True |
| ETTm1 | GapRank_vs_Equal | block_bootstrap_len24 | `+0.001076` | [+0.000518, +0.001604] | True |
| ETTm1 | GapRank_vs_Equal | block_bootstrap_len48 | `+0.001076` | [+0.000493, +0.001649] | True |
| ETTm1 | GapRank_vs_Equal | every_12th_window_phase_bootstrap | `+0.001076` | [+0.000827, +0.001339] | True |
| ETTm1 | FrozenProbeGapRank_vs_GapRank | iid_paired_bootstrap | `+0.000316` | [+0.000121, +0.000515] | True |
| ETTm1 | FrozenProbeGapRank_vs_GapRank | block_bootstrap_len12 | `+0.000316` | [-0.000048, +0.000668] | False |
| ETTm1 | FrozenProbeGapRank_vs_GapRank | block_bootstrap_len24 | `+0.000316` | [-0.000077, +0.000672] | False |
| ETTm1 | FrozenProbeGapRank_vs_GapRank | block_bootstrap_len48 | `+0.000316` | [-0.000071, +0.000680] | False |
| ETTm1 | FrozenProbeGapRank_vs_GapRank | every_12th_window_phase_bootstrap | `+0.000316` | [+0.000180, +0.000438] | True |
| ETTm1 | FrozenProbeGapRank_vs_LearnedProbeRank | iid_paired_bootstrap | `-0.000304` | [-0.000487, -0.000125] | True |
| ETTm1 | FrozenProbeGapRank_vs_LearnedProbeRank | block_bootstrap_len12 | `-0.000304` | [-0.000619, -0.000006] | True |
| ETTm1 | FrozenProbeGapRank_vs_LearnedProbeRank | block_bootstrap_len24 | `-0.000304` | [-0.000655, +0.000016] | False |
| ETTm1 | FrozenProbeGapRank_vs_LearnedProbeRank | block_bootstrap_len48 | `-0.000304` | [-0.000652, +0.000024] | False |
| ETTm1 | FrozenProbeGapRank_vs_LearnedProbeRank | every_12th_window_phase_bootstrap | `-0.000304` | [-0.000414, -0.000218] | True |
| Weather | GapRank_vs_LearnedProbeRank | iid_paired_bootstrap | `+0.000304` | [+0.000200, +0.000415] | True |
| Weather | GapRank_vs_LearnedProbeRank | block_bootstrap_len12 | `+0.000304` | [+0.000129, +0.000483] | True |
| Weather | GapRank_vs_LearnedProbeRank | block_bootstrap_len24 | `+0.000304` | [+0.000112, +0.000490] | True |
| Weather | GapRank_vs_LearnedProbeRank | block_bootstrap_len48 | `+0.000304` | [+0.000127, +0.000502] | True |
| Weather | GapRank_vs_LearnedProbeRank | every_12th_window_phase_bootstrap | `+0.000304` | [+0.000206, +0.000414] | True |
| Weather | GapRank_vs_CRank | iid_paired_bootstrap | `-0.000283` | [-0.000429, -0.000139] | True |
| Weather | GapRank_vs_CRank | block_bootstrap_len12 | `-0.000283` | [-0.000528, -0.000039] | True |
| Weather | GapRank_vs_CRank | block_bootstrap_len24 | `-0.000283` | [-0.000549, -0.000040] | True |
| Weather | GapRank_vs_CRank | block_bootstrap_len48 | `-0.000283` | [-0.000540, -0.000027] | True |
| Weather | GapRank_vs_CRank | every_12th_window_phase_bootstrap | `-0.000283` | [-0.000440, -0.000140] | True |
| Weather | GapRank_vs_FixedDRank | iid_paired_bootstrap | `-0.000600` | [-0.000735, -0.000460] | True |
| Weather | GapRank_vs_FixedDRank | block_bootstrap_len12 | `-0.000600` | [-0.000842, -0.000376] | True |
| Weather | GapRank_vs_FixedDRank | block_bootstrap_len24 | `-0.000600` | [-0.000876, -0.000364] | True |
| Weather | GapRank_vs_FixedDRank | block_bootstrap_len48 | `-0.000600` | [-0.000884, -0.000356] | True |
| Weather | GapRank_vs_FixedDRank | every_12th_window_phase_bootstrap | `-0.000600` | [-0.000732, -0.000466] | True |
| Weather | GapRank_vs_Equal | iid_paired_bootstrap | `-0.000852` | [-0.001097, -0.000612] | True |
| Weather | GapRank_vs_Equal | block_bootstrap_len12 | `-0.000852` | [-0.001601, -0.000208] | True |
| Weather | GapRank_vs_Equal | block_bootstrap_len24 | `-0.000852` | [-0.001869, -0.000048] | True |
| Weather | GapRank_vs_Equal | block_bootstrap_len48 | `-0.000852` | [-0.002139, +0.000071] | False |
| Weather | GapRank_vs_Equal | every_12th_window_phase_bootstrap | `-0.000852` | [-0.001022, -0.000677] | True |
| Weather | FrozenProbeGapRank_vs_GapRank | iid_paired_bootstrap | `+0.000006` | [-0.000090, +0.000097] | False |
| Weather | FrozenProbeGapRank_vs_GapRank | block_bootstrap_len12 | `+0.000006` | [-0.000108, +0.000122] | False |
| Weather | FrozenProbeGapRank_vs_GapRank | block_bootstrap_len24 | `+0.000006` | [-0.000121, +0.000129] | False |
| Weather | FrozenProbeGapRank_vs_GapRank | block_bootstrap_len48 | `+0.000006` | [-0.000130, +0.000129] | False |
| Weather | FrozenProbeGapRank_vs_GapRank | every_12th_window_phase_bootstrap | `+0.000006` | [-0.000077, +0.000090] | False |
| Weather | FrozenProbeGapRank_vs_LearnedProbeRank | iid_paired_bootstrap | `+0.000310` | [+0.000209, +0.000414] | True |
| Weather | FrozenProbeGapRank_vs_LearnedProbeRank | block_bootstrap_len12 | `+0.000310` | [+0.000135, +0.000479] | True |
| Weather | FrozenProbeGapRank_vs_LearnedProbeRank | block_bootstrap_len24 | `+0.000310` | [+0.000113, +0.000481] | True |
| Weather | FrozenProbeGapRank_vs_LearnedProbeRank | block_bootstrap_len48 | `+0.000310` | [+0.000121, +0.000492] | True |
| Weather | FrozenProbeGapRank_vs_LearnedProbeRank | every_12th_window_phase_bootstrap | `+0.000310` | [+0.000230, +0.000376] | True |
| Electricity | GapRank_vs_LearnedProbeRank | iid_paired_bootstrap | `+0.000438` | [+0.000334, +0.000556] | True |
| Electricity | GapRank_vs_LearnedProbeRank | block_bootstrap_len12 | `+0.000438` | [+0.000253, +0.000638] | True |
| Electricity | GapRank_vs_LearnedProbeRank | block_bootstrap_len24 | `+0.000438` | [+0.000257, +0.000640] | True |
| Electricity | GapRank_vs_LearnedProbeRank | block_bootstrap_len48 | `+0.000438` | [+0.000266, +0.000644] | True |
| Electricity | GapRank_vs_LearnedProbeRank | every_12th_window_phase_bootstrap | `+0.000438` | [+0.000263, +0.000624] | True |
| Electricity | GapRank_vs_CRank | iid_paired_bootstrap | `-0.001553` | [-0.001803, -0.001303] | True |
| Electricity | GapRank_vs_CRank | block_bootstrap_len12 | `-0.001553` | [-0.002079, -0.001064] | True |
| Electricity | GapRank_vs_CRank | block_bootstrap_len24 | `-0.001553` | [-0.002092, -0.001052] | True |
| Electricity | GapRank_vs_CRank | block_bootstrap_len48 | `-0.001553` | [-0.002051, -0.001086] | True |
| Electricity | GapRank_vs_CRank | every_12th_window_phase_bootstrap | `-0.001552` | [-0.002197, -0.000926] | True |
| Electricity | GapRank_vs_FixedDRank | iid_paired_bootstrap | `-0.002691` | [-0.002952, -0.002441] | True |
| Electricity | GapRank_vs_FixedDRank | block_bootstrap_len12 | `-0.002691` | [-0.003299, -0.002140] | True |
| Electricity | GapRank_vs_FixedDRank | block_bootstrap_len24 | `-0.002691` | [-0.003305, -0.002137] | True |
| Electricity | GapRank_vs_FixedDRank | block_bootstrap_len48 | `-0.002691` | [-0.003294, -0.002137] | True |
| Electricity | GapRank_vs_FixedDRank | every_12th_window_phase_bootstrap | `-0.002690` | [-0.003476, -0.001869] | True |
| Electricity | GapRank_vs_Equal | iid_paired_bootstrap | `-0.000394` | [-0.000584, -0.000212] | True |
| Electricity | GapRank_vs_Equal | block_bootstrap_len12 | `-0.000394` | [-0.000830, +0.000030] | False |
| Electricity | GapRank_vs_Equal | block_bootstrap_len24 | `-0.000394` | [-0.000882, +0.000111] | False |
| Electricity | GapRank_vs_Equal | block_bootstrap_len48 | `-0.000394` | [-0.000965, +0.000203] | False |
| Electricity | GapRank_vs_Equal | every_12th_window_phase_bootstrap | `-0.000394` | [-0.000786, +0.000036] | False |
| Electricity | FrozenProbeGapRank_vs_GapRank | iid_paired_bootstrap | `+0.000005` | [-0.000087, +0.000099] | False |
| Electricity | FrozenProbeGapRank_vs_GapRank | block_bootstrap_len12 | `+0.000005` | [-0.000121, +0.000134] | False |
| Electricity | FrozenProbeGapRank_vs_GapRank | block_bootstrap_len24 | `+0.000005` | [-0.000125, +0.000133] | False |
| Electricity | FrozenProbeGapRank_vs_GapRank | block_bootstrap_len48 | `+0.000005` | [-0.000113, +0.000126] | False |
| Electricity | FrozenProbeGapRank_vs_GapRank | every_12th_window_phase_bootstrap | `+0.000005` | [-0.000180, +0.000188] | False |
| Electricity | FrozenProbeGapRank_vs_LearnedProbeRank | iid_paired_bootstrap | `+0.000444` | [+0.000329, +0.000568] | True |
| Electricity | FrozenProbeGapRank_vs_LearnedProbeRank | block_bootstrap_len12 | `+0.000444` | [+0.000262, +0.000639] | True |
| Electricity | FrozenProbeGapRank_vs_LearnedProbeRank | block_bootstrap_len24 | `+0.000444` | [+0.000263, +0.000635] | True |
| Electricity | FrozenProbeGapRank_vs_LearnedProbeRank | block_bootstrap_len48 | `+0.000444` | [+0.000271, +0.000642] | True |
| Electricity | FrozenProbeGapRank_vs_LearnedProbeRank | every_12th_window_phase_bootstrap | `+0.000443` | [+0.000183, +0.000691] | True |

## Integrity

- **ETTh1**: PASS (checkpoints unchanged: True; experts frozen during training: True; gap_scale (router_train-only)=0.064703; weights invariant to target corruption: True)
- **ETTh2**: PASS (checkpoints unchanged: True; experts frozen during training: True; gap_scale (router_train-only)=0.046205; weights invariant to target corruption: True)
- **ETTm1**: PASS (checkpoints unchanged: True; experts frozen during training: True; gap_scale (router_train-only)=0.061772; weights invariant to target corruption: True)
- **Weather**: PASS (checkpoints unchanged: True; experts frozen during training: True; gap_scale (router_train-only)=0.038498; weights invariant to target corruption: True)
- **Electricity**: PASS (checkpoints unchanged: True; experts frozen during training: True; gap_scale (router_train-only)=0.044430; weights invariant to target corruption: True)

## Interpretation

**1. ETTm1 LearnedProbe-GapRank beat original LearnedProbe-Rank?** True (point estimate); significant=True
**2. ETTm1 top-1 accuracy improve?** 0.326 -> 0.338
**3. ETTm1 top-2 recall improve?** 0.640 -> 0.669
**4. ETTm1 stop being significantly worse than C-Rank?** False
**5. ETTh2/Weather/Electricity preserved or improved?** {'ETTh2': False, 'Weather': False, 'Electricity': False}
**6. ETTh1 improve or stay neutral?** True (delta_vs_original=+0.000283)
**7. Overall pairwise ranking accuracy change?** see competence CSV per dataset
**8. Top-1 accuracy more aligned with MAE?** see ettm1_diag / competence rows
**9. High-cost ranking mistakes reduced?** cost-weighted error improved on 1/5 datasets
**10. Does the new objective improve forecasting broadly, not just ETTm1?** No -- isolated to ETTm1

## Verdict

**NO IMPROVEMENT — KEEP ORIGINAL**

- GapRank beats original LearnedProbe-Rank with block-bootstrap significance on 1/5 datasets (need >=2 for strong).
- GapRank significantly HURTS original on 3/5 datasets (need 0 for strong).
- ETTm1 regression vs C-Rank disappears/reverses: False (required for strong).
- ETTh2/Weather/Electricity preserved or improved: {'ETTh2': False, 'Weather': False, 'Electricity': False} (all True required for strong).
- Cost-weighted ranking error improves on 1/5 datasets (need >=3 for strong).
- New significant regressions introduced on 3/5 datasets (need 0 for strong).

## Hard rule compliance

```text
TEST SET ACCESSED: NO
TEST CACHE LOADED: NO
TEST METRICS COMPUTED: NO
```
