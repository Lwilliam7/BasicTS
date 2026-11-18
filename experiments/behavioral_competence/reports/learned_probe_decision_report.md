# Learned-Probe Decision-Rule Comparison

Isolates the score-to-weight conversion rule. `Learned_Probe_pred_excess` (predicted_excess_loss) is reused byte-for-byte from run_learned_probe.py for every rule -- the ProbeGenerator and competence scorer are never re-run.

## Main result table (router_val MAE / MSE)

| Dataset | Equal | C | Fixed-D | Softmax | Top1 | Top2Equal | Rank | Oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ETTh1 | 0.367265 | 0.368334 | 0.370864 | 0.368085 | 0.375957 | 0.370918 | 0.366729 | 0.343984 |
| ETTh2 | 0.280878 | 0.286494 | 0.279710 | 0.277853 | 0.283165 | 0.279118 | 0.277202 | 0.266483 |
| ETTm1 | 0.248161 | 0.250401 | 0.251473 | 0.252576 | 0.258499 | 0.255023 | 0.249857 | 0.227306 |
| Weather | 0.160341 | 0.159672 | 0.160337 | 0.159234 | 0.162920 | 0.161609 | 0.159185 | 0.150243 |
| Electricity | 0.214457 | 0.219244 | 0.222766 | 0.217415 | 0.222564 | 0.218425 | 0.213625 | 0.217440 |

## Deltas for each new decision rule

| Dataset | Rule | Δ vs C | Δ vs Softmax | Δ vs Equal |
|---|---|---:|---:|---:|
| ETTh1 | LearnedProbe_Softmax | `-0.000249` | `+0.000000` | `+0.000820` |
| ETTh1 | LearnedProbe_Top1 | `+0.007623` | `+0.007872` | `+0.008692` |
| ETTh1 | LearnedProbe_Top2Equal | `+0.002584` | `+0.002833` | `+0.003653` |
| ETTh1 | LearnedProbe_Rank | `-0.001605` | `-0.001356` | `-0.000536` |
| ETTh2 | LearnedProbe_Softmax | `-0.008641` | `+0.000000` | `-0.003026` |
| ETTh2 | LearnedProbe_Top1 | `-0.003329` | `+0.005312` | `+0.002287` |
| ETTh2 | LearnedProbe_Top2Equal | `-0.007377` | `+0.001265` | `-0.001761` |
| ETTh2 | LearnedProbe_Rank | `-0.009292` | `-0.000651` | `-0.003676` |
| ETTm1 | LearnedProbe_Softmax | `+0.002174` | `+0.000000` | `+0.004414` |
| ETTm1 | LearnedProbe_Top1 | `+0.008098` | `+0.005924` | `+0.010338` |
| ETTm1 | LearnedProbe_Top2Equal | `+0.004622` | `+0.002448` | `+0.006862` |
| ETTm1 | LearnedProbe_Rank | `-0.000544` | `-0.002718` | `+0.001696` |
| Weather | LearnedProbe_Softmax | `-0.000438` | `+0.000000` | `-0.001108` |
| Weather | LearnedProbe_Top1 | `+0.003248` | `+0.003686` | `+0.002578` |
| Weather | LearnedProbe_Top2Equal | `+0.001937` | `+0.002375` | `+0.001268` |
| Weather | LearnedProbe_Rank | `-0.000486` | `-0.000048` | `-0.001156` |
| Electricity | LearnedProbe_Softmax | `-0.001829` | `+0.000000` | `+0.002958` |
| Electricity | LearnedProbe_Top1 | `+0.003320` | `+0.005150` | `+0.008107` |
| Electricity | LearnedProbe_Top2Equal | `-0.000819` | `+0.001010` | `+0.003968` |
| Electricity | LearnedProbe_Rank | `-0.005619` | `-0.003790` | `-0.000832` |

## Expert-selection metrics

| Dataset | Top-1 accuracy | Top-2 recall | Rank correlation (Spearman) |
|---|---:|---:|---:|
| ETTh1 | 0.390 | 0.719 | 0.241 |
| ETTh2 | 0.460 | 0.847 | 0.397 |
| ETTm1 | 0.326 | 0.640 | 0.161 |
| Weather | 0.444 | 0.738 | 0.301 |
| Electricity | 0.629 | 0.924 | 0.693 |

## Winner margin analysis (Top1 vs Top2Equal, median-margin split)

| Dataset | Group | Windows | Top1 MAE | Top2Equal MAE | Top1 - Top2Equal |
|---|---|---:|---:|---:|---:|
| ETTh1 | high_margin | 1387 | 0.375078 | 0.372289 | `+0.002789` |
| ETTh1 | low_margin | 1386 | 0.376836 | 0.369546 | `+0.007290` |
| ETTh2 | high_margin | 307 | 0.289961 | 0.289405 | `+0.000556` |
| ETTh2 | low_margin | 306 | 0.276347 | 0.268796 | `+0.007551` |
| ETTm1 | high_margin | 5707 | 0.265558 | 0.263234 | `+0.002324` |
| ETTm1 | low_margin | 5706 | 0.251440 | 0.246811 | `+0.004628` |
| Weather | high_margin | 5217 | 0.159997 | 0.160290 | `-0.000293` |
| Weather | low_margin | 5215 | 0.165843 | 0.162928 | `+0.002915` |
| Electricity | high_margin | 2578 | 0.233040 | 0.229332 | `+0.003709` |
| Electricity | low_margin | 2576 | 0.212081 | 0.207510 | `+0.004570` |

## Oracle headroom: Top-1 vs Top-2 potential (relative to Equal Fixed)

| Dataset | Best-Pair Oracle MAE | Top-1 oracle headroom | Top-2 oracle headroom |
|---|---:|---:|---:|
| ETTh1 | 0.349074 | `+0.023281` | `+0.018191` |
| ETTh2 | 0.268403 | `+0.014395` | `+0.012475` |
| ETTm1 | 0.232556 | `+0.020855` | `+0.015605` |
| Weather | 0.151323 | `+0.010098` | `+0.009018` |
| Electricity | 0.211688 | `-0.002983` | `+0.002769` |

## Dependence-aware statistics (block bootstrap)

| Dataset | Comparison | Test | Mean delta | 95% CI | Excludes zero |
|---|---|---|---:|---|---|
| ETTh1 | Top1_vs_Softmax | iid_paired_bootstrap | `+0.007872` | [+0.006981, +0.008781] | True |
| ETTh1 | Top1_vs_Softmax | block_bootstrap_len12 | `+0.007872` | [+0.006525, +0.009246] | True |
| ETTh1 | Top1_vs_Softmax | block_bootstrap_len24 | `+0.007872` | [+0.006396, +0.009442] | True |
| ETTh1 | Top1_vs_Softmax | block_bootstrap_len48 | `+0.007872` | [+0.006324, +0.009572] | True |
| ETTh1 | Top1_vs_Softmax | every_12th_window_phase_bootstrap | `+0.007871` | [+0.006974, +0.008826] | True |
| ETTh1 | Top2Equal_vs_Softmax | iid_paired_bootstrap | `+0.002833` | [+0.002153, +0.003528] | True |
| ETTh1 | Top2Equal_vs_Softmax | block_bootstrap_len12 | `+0.002833` | [+0.001817, +0.003856] | True |
| ETTh1 | Top2Equal_vs_Softmax | block_bootstrap_len24 | `+0.002833` | [+0.001719, +0.003946] | True |
| ETTh1 | Top2Equal_vs_Softmax | block_bootstrap_len48 | `+0.002833` | [+0.001629, +0.004073] | True |
| ETTh1 | Top2Equal_vs_Softmax | every_12th_window_phase_bootstrap | `+0.002833` | [+0.002098, +0.003538] | True |
| ETTh1 | Rank_vs_Softmax | iid_paired_bootstrap | `-0.001356` | [-0.001797, -0.000929] | True |
| ETTh1 | Rank_vs_Softmax | block_bootstrap_len12 | `-0.001356` | [-0.002227, -0.000503] | True |
| ETTh1 | Rank_vs_Softmax | block_bootstrap_len24 | `-0.001356` | [-0.002271, -0.000414] | True |
| ETTh1 | Rank_vs_Softmax | block_bootstrap_len48 | `-0.001356` | [-0.002319, -0.000408] | True |
| ETTh1 | Rank_vs_Softmax | every_12th_window_phase_bootstrap | `-0.001356` | [-0.001778, -0.000963] | True |
| ETTh1 | Top1_vs_C | iid_paired_bootstrap | `+0.007623` | [+0.006377, +0.008859] | True |
| ETTh1 | Top1_vs_C | block_bootstrap_len12 | `+0.007623` | [+0.005741, +0.009449] | True |
| ETTh1 | Top1_vs_C | block_bootstrap_len24 | `+0.007623` | [+0.005657, +0.009510] | True |
| ETTh1 | Top1_vs_C | block_bootstrap_len48 | `+0.007623` | [+0.005584, +0.009685] | True |
| ETTh1 | Top1_vs_C | every_12th_window_phase_bootstrap | `+0.007622` | [+0.006311, +0.009102] | True |
| ETTh1 | Top2Equal_vs_C | iid_paired_bootstrap | `+0.002584` | [+0.001660, +0.003547] | True |
| ETTh1 | Top2Equal_vs_C | block_bootstrap_len12 | `+0.002584` | [+0.000922, +0.004207] | True |
| ETTh1 | Top2Equal_vs_C | block_bootstrap_len24 | `+0.002584` | [+0.000728, +0.004327] | True |
| ETTh1 | Top2Equal_vs_C | block_bootstrap_len48 | `+0.002584` | [+0.000722, +0.004549] | True |
| ETTh1 | Top2Equal_vs_C | every_12th_window_phase_bootstrap | `+0.002584` | [+0.001717, +0.003504] | True |
| ETTh1 | Rank_vs_C | iid_paired_bootstrap | `-0.001605` | [-0.002302, -0.000883] | True |
| ETTh1 | Rank_vs_C | block_bootstrap_len12 | `-0.001605` | [-0.003027, -0.000213] | True |
| ETTh1 | Rank_vs_C | block_bootstrap_len24 | `-0.001605` | [-0.003185, -0.000084] | True |
| ETTh1 | Rank_vs_C | block_bootstrap_len48 | `-0.001605` | [-0.003352, +0.000126] | False |
| ETTh1 | Rank_vs_C | every_12th_window_phase_bootstrap | `-0.001605` | [-0.002153, -0.001079] | True |
| ETTh1 | BestRule(LearnedProbe_Rank)_vs_Equal | iid_paired_bootstrap | `-0.000536` | [-0.006764, +0.005911] | False |
| ETTh1 | BestRule(LearnedProbe_Rank)_vs_Equal | block_bootstrap_len12 | `-0.000536` | [-0.019254, +0.018530] | False |
| ETTh1 | BestRule(LearnedProbe_Rank)_vs_Equal | block_bootstrap_len24 | `-0.000536` | [-0.022810, +0.022183] | False |
| ETTh1 | BestRule(LearnedProbe_Rank)_vs_Equal | block_bootstrap_len48 | `-0.000536` | [-0.026458, +0.027212] | False |
| ETTh1 | BestRule(LearnedProbe_Rank)_vs_Equal | every_12th_window_phase_bootstrap | `-0.000530` | [-0.008074, +0.007245] | False |
| ETTh2 | Top1_vs_Softmax | iid_paired_bootstrap | `+0.005312` | [+0.004300, +0.006366] | True |
| ETTh2 | Top1_vs_Softmax | block_bootstrap_len12 | `+0.005312` | [+0.003819, +0.006811] | True |
| ETTh2 | Top1_vs_Softmax | block_bootstrap_len24 | `+0.005312` | [+0.003753, +0.006771] | True |
| ETTh2 | Top1_vs_Softmax | block_bootstrap_len48 | `+0.005312` | [+0.003575, +0.006853] | True |
| ETTh2 | Top1_vs_Softmax | every_12th_window_phase_bootstrap | `+0.005310` | [+0.004370, +0.006206] | True |
| ETTh2 | Top2Equal_vs_Softmax | iid_paired_bootstrap | `+0.001265` | [+0.000023, +0.002544] | True |
| ETTh2 | Top2Equal_vs_Softmax | block_bootstrap_len12 | `+0.001265` | [-0.000811, +0.003721] | False |
| ETTh2 | Top2Equal_vs_Softmax | block_bootstrap_len24 | `+0.001265` | [-0.000884, +0.004013] | False |
| ETTh2 | Top2Equal_vs_Softmax | block_bootstrap_len48 | `+0.001265` | [-0.000717, +0.004447] | False |
| ETTh2 | Top2Equal_vs_Softmax | every_12th_window_phase_bootstrap | `+0.001265` | [+0.000423, +0.002094] | True |
| ETTh2 | Rank_vs_Softmax | iid_paired_bootstrap | `-0.000651` | [-0.001544, +0.000222] | False |
| ETTh2 | Rank_vs_Softmax | block_bootstrap_len12 | `-0.000651` | [-0.002112, +0.001037] | False |
| ETTh2 | Rank_vs_Softmax | block_bootstrap_len24 | `-0.000651` | [-0.002163, +0.001264] | False |
| ETTh2 | Rank_vs_Softmax | block_bootstrap_len48 | `-0.000651` | [-0.001871, +0.001250] | False |
| ETTh2 | Rank_vs_Softmax | every_12th_window_phase_bootstrap | `-0.000650` | [-0.001245, +0.000009] | False |
| ETTh2 | Top1_vs_C | iid_paired_bootstrap | `-0.003329` | [-0.005585, -0.000962] | True |
| ETTh2 | Top1_vs_C | block_bootstrap_len12 | `-0.003329` | [-0.007679, +0.000576] | False |
| ETTh2 | Top1_vs_C | block_bootstrap_len24 | `-0.003329` | [-0.007549, +0.000269] | False |
| ETTh2 | Top1_vs_C | block_bootstrap_len48 | `-0.003329` | [-0.007459, +0.000218] | False |
| ETTh2 | Top1_vs_C | every_12th_window_phase_bootstrap | `-0.003336` | [-0.005193, -0.001549] | True |
| ETTh2 | Top2Equal_vs_C | iid_paired_bootstrap | `-0.007377` | [-0.009323, -0.005437] | True |
| ETTh2 | Top2Equal_vs_C | block_bootstrap_len12 | `-0.007377` | [-0.010878, -0.004030] | True |
| ETTh2 | Top2Equal_vs_C | block_bootstrap_len24 | `-0.007377` | [-0.010983, -0.004148] | True |
| ETTh2 | Top2Equal_vs_C | block_bootstrap_len48 | `-0.007377` | [-0.011447, -0.003436] | True |
| ETTh2 | Top2Equal_vs_C | every_12th_window_phase_bootstrap | `-0.007381` | [-0.008684, -0.006155] | True |
| ETTh2 | Rank_vs_C | iid_paired_bootstrap | `-0.009292` | [-0.011039, -0.007665] | True |
| ETTh2 | Rank_vs_C | block_bootstrap_len12 | `-0.009292` | [-0.012451, -0.006308] | True |
| ETTh2 | Rank_vs_C | block_bootstrap_len24 | `-0.009292` | [-0.012308, -0.006548] | True |
| ETTh2 | Rank_vs_C | block_bootstrap_len48 | `-0.009292` | [-0.012727, -0.006235] | True |
| ETTh2 | Rank_vs_C | every_12th_window_phase_bootstrap | `-0.009296` | [-0.010576, -0.008048] | True |
| ETTh2 | BestRule(LearnedProbe_Rank)_vs_Equal | iid_paired_bootstrap | `-0.003676` | [-0.011940, +0.005078] | False |
| ETTh2 | BestRule(LearnedProbe_Rank)_vs_Equal | block_bootstrap_len12 | `-0.003676` | [-0.027959, +0.022295] | False |
| ETTh2 | BestRule(LearnedProbe_Rank)_vs_Equal | block_bootstrap_len24 | `-0.003676` | [-0.033017, +0.028844] | False |
| ETTh2 | BestRule(LearnedProbe_Rank)_vs_Equal | block_bootstrap_len48 | `-0.003676` | [-0.034047, +0.032039] | False |
| ETTh2 | BestRule(LearnedProbe_Rank)_vs_Equal | every_12th_window_phase_bootstrap | `-0.003671` | [-0.006052, -0.001084] | True |
| ETTm1 | Top1_vs_Softmax | iid_paired_bootstrap | `+0.005924` | [+0.005589, +0.006261] | True |
| ETTm1 | Top1_vs_Softmax | block_bootstrap_len12 | `+0.005924` | [+0.005420, +0.006409] | True |
| ETTm1 | Top1_vs_Softmax | block_bootstrap_len24 | `+0.005924` | [+0.005384, +0.006439] | True |
| ETTm1 | Top1_vs_Softmax | block_bootstrap_len48 | `+0.005924` | [+0.005386, +0.006429] | True |
| ETTm1 | Top1_vs_Softmax | every_12th_window_phase_bootstrap | `+0.005924` | [+0.005640, +0.006228] | True |
| ETTm1 | Top2Equal_vs_Softmax | iid_paired_bootstrap | `+0.002448` | [+0.002102, +0.002784] | True |
| ETTm1 | Top2Equal_vs_Softmax | block_bootstrap_len12 | `+0.002448` | [+0.001843, +0.003029] | True |
| ETTm1 | Top2Equal_vs_Softmax | block_bootstrap_len24 | `+0.002448` | [+0.001801, +0.003055] | True |
| ETTm1 | Top2Equal_vs_Softmax | block_bootstrap_len48 | `+0.002448` | [+0.001827, +0.003025] | True |
| ETTm1 | Top2Equal_vs_Softmax | every_12th_window_phase_bootstrap | `+0.002448` | [+0.002128, +0.002756] | True |
| ETTm1 | Rank_vs_Softmax | iid_paired_bootstrap | `-0.002718` | [-0.002927, -0.002511] | True |
| ETTm1 | Rank_vs_Softmax | block_bootstrap_len12 | `-0.002718` | [-0.003122, -0.002317] | True |
| ETTm1 | Rank_vs_Softmax | block_bootstrap_len24 | `-0.002718` | [-0.003146, -0.002296] | True |
| ETTm1 | Rank_vs_Softmax | block_bootstrap_len48 | `-0.002718` | [-0.003162, -0.002292] | True |
| ETTm1 | Rank_vs_Softmax | every_12th_window_phase_bootstrap | `-0.002718` | [-0.002907, -0.002547] | True |
| ETTm1 | Top1_vs_C | iid_paired_bootstrap | `+0.008098` | [+0.007613, +0.008588] | True |
| ETTm1 | Top1_vs_C | block_bootstrap_len12 | `+0.008098` | [+0.007310, +0.008883] | True |
| ETTm1 | Top1_vs_C | block_bootstrap_len24 | `+0.008098` | [+0.007250, +0.008922] | True |
| ETTm1 | Top1_vs_C | block_bootstrap_len48 | `+0.008098` | [+0.007282, +0.008903] | True |
| ETTm1 | Top1_vs_C | every_12th_window_phase_bootstrap | `+0.008098` | [+0.007633, +0.008533] | True |
| ETTm1 | Top2Equal_vs_C | iid_paired_bootstrap | `+0.004622` | [+0.004199, +0.005042] | True |
| ETTm1 | Top2Equal_vs_C | block_bootstrap_len12 | `+0.004622` | [+0.003860, +0.005388] | True |
| ETTm1 | Top2Equal_vs_C | block_bootstrap_len24 | `+0.004622` | [+0.003837, +0.005396] | True |
| ETTm1 | Top2Equal_vs_C | block_bootstrap_len48 | `+0.004622` | [+0.003830, +0.005348] | True |
| ETTm1 | Top2Equal_vs_C | every_12th_window_phase_bootstrap | `+0.004622` | [+0.004235, +0.005053] | True |
| ETTm1 | Rank_vs_C | iid_paired_bootstrap | `-0.000544` | [-0.000842, -0.000257] | True |
| ETTm1 | Rank_vs_C | block_bootstrap_len12 | `-0.000544` | [-0.001119, +0.000041] | False |
| ETTm1 | Rank_vs_C | block_bootstrap_len24 | `-0.000544` | [-0.001172, +0.000063] | False |
| ETTm1 | Rank_vs_C | block_bootstrap_len48 | `-0.000544` | [-0.001188, +0.000092] | False |
| ETTm1 | Rank_vs_C | every_12th_window_phase_bootstrap | `-0.000544` | [-0.000918, -0.000189] | True |
| ETTm1 | BestRule(LearnedProbe_Rank)_vs_Equal | iid_paired_bootstrap | `+0.001696` | [-0.000634, +0.004008] | False |
| ETTm1 | BestRule(LearnedProbe_Rank)_vs_Equal | block_bootstrap_len12 | `+0.001696` | [-0.005269, +0.008846] | False |
| ETTm1 | BestRule(LearnedProbe_Rank)_vs_Equal | block_bootstrap_len24 | `+0.001696` | [-0.006798, +0.010746] | False |
| ETTm1 | BestRule(LearnedProbe_Rank)_vs_Equal | block_bootstrap_len48 | `+0.001696` | [-0.008066, +0.012377] | False |
| ETTm1 | BestRule(LearnedProbe_Rank)_vs_Equal | every_12th_window_phase_bootstrap | `+0.001696` | [+0.000260, +0.003098] | True |
| Weather | Top1_vs_Softmax | iid_paired_bootstrap | `+0.003686` | [+0.003495, +0.003876] | True |
| Weather | Top1_vs_Softmax | block_bootstrap_len12 | `+0.003686` | [+0.003376, +0.004007] | True |
| Weather | Top1_vs_Softmax | block_bootstrap_len24 | `+0.003686` | [+0.003347, +0.004032] | True |
| Weather | Top1_vs_Softmax | block_bootstrap_len48 | `+0.003686` | [+0.003309, +0.004069] | True |
| Weather | Top1_vs_Softmax | every_12th_window_phase_bootstrap | `+0.003686` | [+0.003503, +0.003869] | True |
| Weather | Top2Equal_vs_Softmax | iid_paired_bootstrap | `+0.002375` | [+0.002189, +0.002554] | True |
| Weather | Top2Equal_vs_Softmax | block_bootstrap_len12 | `+0.002375` | [+0.002050, +0.002697] | True |
| Weather | Top2Equal_vs_Softmax | block_bootstrap_len24 | `+0.002375` | [+0.002023, +0.002716] | True |
| Weather | Top2Equal_vs_Softmax | block_bootstrap_len48 | `+0.002375` | [+0.002013, +0.002732] | True |
| Weather | Top2Equal_vs_Softmax | every_12th_window_phase_bootstrap | `+0.002376` | [+0.002208, +0.002555] | True |
| Weather | Rank_vs_Softmax | iid_paired_bootstrap | `-0.000048` | [-0.000277, +0.000193] | False |
| Weather | Rank_vs_Softmax | block_bootstrap_len12 | `-0.000048` | [-0.000636, +0.000627] | False |
| Weather | Rank_vs_Softmax | block_bootstrap_len24 | `-0.000048` | [-0.000721, +0.000818] | False |
| Weather | Rank_vs_Softmax | block_bootstrap_len48 | `-0.000048` | [-0.000821, +0.001053] | False |
| Weather | Rank_vs_Softmax | every_12th_window_phase_bootstrap | `-0.000048` | [-0.000231, +0.000102] | False |
| Weather | Top1_vs_C | iid_paired_bootstrap | `+0.003248` | [+0.002920, +0.003576] | True |
| Weather | Top1_vs_C | block_bootstrap_len12 | `+0.003248` | [+0.002652, +0.003859] | True |
| Weather | Top1_vs_C | block_bootstrap_len24 | `+0.003248` | [+0.002588, +0.003925] | True |
| Weather | Top1_vs_C | block_bootstrap_len48 | `+0.003248` | [+0.002530, +0.003966] | True |
| Weather | Top1_vs_C | every_12th_window_phase_bootstrap | `+0.003248` | [+0.002859, +0.003612] | True |
| Weather | Top2Equal_vs_C | iid_paired_bootstrap | `+0.001937` | [+0.001675, +0.002185] | True |
| Weather | Top2Equal_vs_C | block_bootstrap_len12 | `+0.001937` | [+0.001434, +0.002453] | True |
| Weather | Top2Equal_vs_C | block_bootstrap_len24 | `+0.001937` | [+0.001365, +0.002505] | True |
| Weather | Top2Equal_vs_C | block_bootstrap_len48 | `+0.001937` | [+0.001317, +0.002536] | True |
| Weather | Top2Equal_vs_C | every_12th_window_phase_bootstrap | `+0.001938` | [+0.001520, +0.002349] | True |
| Weather | Rank_vs_C | iid_paired_bootstrap | `-0.000486` | [-0.000721, -0.000252] | True |
| Weather | Rank_vs_C | block_bootstrap_len12 | `-0.000486` | [-0.001022, +0.000101] | False |
| Weather | Rank_vs_C | block_bootstrap_len24 | `-0.000486` | [-0.001132, +0.000266] | False |
| Weather | Rank_vs_C | block_bootstrap_len48 | `-0.000486` | [-0.001218, +0.000437] | False |
| Weather | Rank_vs_C | every_12th_window_phase_bootstrap | `-0.000486` | [-0.000725, -0.000259] | True |
| Weather | BestRule(LearnedProbe_Rank)_vs_Equal | iid_paired_bootstrap | `-0.001156` | [-0.003316, +0.001033] | False |
| Weather | BestRule(LearnedProbe_Rank)_vs_Equal | block_bootstrap_len12 | `-0.001156` | [-0.007643, +0.006103] | False |
| Weather | BestRule(LearnedProbe_Rank)_vs_Equal | block_bootstrap_len24 | `-0.001156` | [-0.009598, +0.008072] | False |
| Weather | BestRule(LearnedProbe_Rank)_vs_Equal | block_bootstrap_len48 | `-0.001156` | [-0.011105, +0.009486] | False |
| Weather | BestRule(LearnedProbe_Rank)_vs_Equal | every_12th_window_phase_bootstrap | `-0.001156` | [-0.001639, -0.000667] | True |
| Electricity | Top1_vs_Softmax | iid_paired_bootstrap | `+0.005150` | [+0.004932, +0.005371] | True |
| Electricity | Top1_vs_Softmax | block_bootstrap_len12 | `+0.005150` | [+0.004811, +0.005511] | True |
| Electricity | Top1_vs_Softmax | block_bootstrap_len24 | `+0.005150` | [+0.004792, +0.005538] | True |
| Electricity | Top1_vs_Softmax | block_bootstrap_len48 | `+0.005150` | [+0.004739, +0.005618] | True |
| Electricity | Top1_vs_Softmax | every_12th_window_phase_bootstrap | `+0.005150` | [+0.004762, +0.005566] | True |
| Electricity | Top2Equal_vs_Softmax | iid_paired_bootstrap | `+0.001010` | [+0.000686, +0.001332] | True |
| Electricity | Top2Equal_vs_Softmax | block_bootstrap_len12 | `+0.001010` | [+0.000507, +0.001524] | True |
| Electricity | Top2Equal_vs_Softmax | block_bootstrap_len24 | `+0.001010` | [+0.000569, +0.001445] | True |
| Electricity | Top2Equal_vs_Softmax | block_bootstrap_len48 | `+0.001010` | [+0.000585, +0.001428] | True |
| Electricity | Top2Equal_vs_Softmax | every_12th_window_phase_bootstrap | `+0.001010` | [+0.000758, +0.001287] | True |
| Electricity | Rank_vs_Softmax | iid_paired_bootstrap | `-0.003790` | [-0.004074, -0.003514] | True |
| Electricity | Rank_vs_Softmax | block_bootstrap_len12 | `-0.003790` | [-0.004277, -0.003322] | True |
| Electricity | Rank_vs_Softmax | block_bootstrap_len24 | `-0.003790` | [-0.004257, -0.003345] | True |
| Electricity | Rank_vs_Softmax | block_bootstrap_len48 | `-0.003790` | [-0.004292, -0.003329] | True |
| Electricity | Rank_vs_Softmax | every_12th_window_phase_bootstrap | `-0.003791` | [-0.004112, -0.003495] | True |
| Electricity | Top1_vs_C | iid_paired_bootstrap | `+0.003320` | [+0.002721, +0.003934] | True |
| Electricity | Top1_vs_C | block_bootstrap_len12 | `+0.003320` | [+0.002225, +0.004373] | True |
| Electricity | Top1_vs_C | block_bootstrap_len24 | `+0.003320` | [+0.002224, +0.004334] | True |
| Electricity | Top1_vs_C | block_bootstrap_len48 | `+0.003320` | [+0.002297, +0.004334] | True |
| Electricity | Top1_vs_C | every_12th_window_phase_bootstrap | `+0.003323` | [+0.001968, +0.004658] | True |
| Electricity | Top2Equal_vs_C | iid_paired_bootstrap | `-0.000819` | [-0.001233, -0.000420] | True |
| Electricity | Top2Equal_vs_C | block_bootstrap_len12 | `-0.000819` | [-0.001645, -0.000058] | True |
| Electricity | Top2Equal_vs_C | block_bootstrap_len24 | `-0.000819` | [-0.001663, -0.000033] | True |
| Electricity | Top2Equal_vs_C | block_bootstrap_len48 | `-0.000819` | [-0.001619, -0.000102] | True |
| Electricity | Top2Equal_vs_C | every_12th_window_phase_bootstrap | `-0.000817` | [-0.001863, +0.000258] | False |
| Electricity | Rank_vs_C | iid_paired_bootstrap | `-0.005619` | [-0.005998, -0.005241] | True |
| Electricity | Rank_vs_C | block_bootstrap_len12 | `-0.005619` | [-0.006465, -0.004831] | True |
| Electricity | Rank_vs_C | block_bootstrap_len24 | `-0.005619` | [-0.006572, -0.004776] | True |
| Electricity | Rank_vs_C | block_bootstrap_len48 | `-0.005619` | [-0.006594, -0.004755] | True |
| Electricity | Rank_vs_C | every_12th_window_phase_bootstrap | `-0.005618` | [-0.006556, -0.004732] | True |
| Electricity | BestRule(LearnedProbe_Rank)_vs_Equal | iid_paired_bootstrap | `-0.000832` | [-0.002359, +0.000758] | False |
| Electricity | BestRule(LearnedProbe_Rank)_vs_Equal | block_bootstrap_len12 | `-0.000832` | [-0.005421, +0.004188] | False |
| Electricity | BestRule(LearnedProbe_Rank)_vs_Equal | block_bootstrap_len24 | `-0.000832` | [-0.006464, +0.005627] | False |
| Electricity | BestRule(LearnedProbe_Rank)_vs_Equal | block_bootstrap_len48 | `-0.000832` | [-0.007709, +0.007882] | False |
| Electricity | BestRule(LearnedProbe_Rank)_vs_Equal | every_12th_window_phase_bootstrap | `-0.000832` | [-0.002529, +0.000919] | False |

## Integrity

- **ETTh1**: PASS (predicted_excess_loss unchanged across rules: True; softmax reproduces saved Learned-Probe prediction: True; weights invariant to target corruption: True)
- **ETTh2**: PASS (predicted_excess_loss unchanged across rules: True; softmax reproduces saved Learned-Probe prediction: True; weights invariant to target corruption: True)
- **ETTm1**: PASS (predicted_excess_loss unchanged across rules: True; softmax reproduces saved Learned-Probe prediction: True; weights invariant to target corruption: True)
- **Weather**: PASS (predicted_excess_loss unchanged across rules: True; softmax reproduces saved Learned-Probe prediction: True; weights invariant to target corruption: True)
- **Electricity**: PASS (predicted_excess_loss unchanged across rules: True; softmax reproduces saved Learned-Probe prediction: True; weights invariant to target corruption: True)

## Decision

**CONTINUE**

- Best decision rule per dataset (by MAE): [('ETTh1', 'LearnedProbe_Rank'), ('ETTh2', 'LearnedProbe_Rank'), ('ETTm1', 'LearnedProbe_Rank'), ('Weather', 'LearnedProbe_Rank'), ('Electricity', 'LearnedProbe_Rank')].
- Best rule beats C on 5/5 datasets (need >=3).
- Beats C with dependence-aware (block-bootstrap) support on 3/5 datasets.
- Best rule significantly HURTS on 0/5 datasets.
- ETTm1 still regresses under its best rule: False.

## Hard rule compliance

```text
TEST SET ACCESSED: NO
TEST CACHE LOADED: NO
TEST METRICS COMPUTED: NO
```
