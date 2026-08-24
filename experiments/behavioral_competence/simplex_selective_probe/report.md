# Simplex + Selective LearnedProbe

Question: can a small, router_train-only-trained gate learn WHEN to trust the frozen LearnedProbe correction on a strong static Simplex ensemble, keeping the ExchangeRate/Traffic gains while avoiding the ETTm2 regression?

effective_alpha_t = gate_t * frozen_alpha, where gate_t = sigmoid(w . standardized_features_t + b) is a 9-feature L2-regularized logistic regression trained on router_train only (chronological leave-one-block-out OOF for regularization selection), predicting whether the always-on Probe correction helped or hurt that router_train window.

## Primary results (router_val MAE)

| Dataset | Simplex | Always-On Probe | Selective Probe | Selective ShuffledProbe | Δ Selective vs Simplex | Δ Selective vs AlwaysOn |
|---|---:|---:|---:|---:|---:|---:|
| ExchangeRate | 0.127167 | 0.119971 | 0.120328 | 0.127168 | `-0.006839` | `+0.000357` |
| Traffic | 0.280925 | 0.270460 | 0.270592 | 0.277661 | `-0.010332` | `+0.000133` |
| BeijingAirQuality | 0.257947 | 0.258301 | 0.257649 | 0.258298 | `-0.000298` | `-0.000651` |
| ETTm2 | 0.161504 | 0.162743 | 0.161622 | 0.161800 | `+0.000118` | `-0.001121` |

## Gain preservation (vs the always-on Probe's gain over Simplex)

- ExchangeRate fraction of always-on gain preserved by Selective: `0.950`
- Traffic fraction of always-on gain preserved by Selective: `0.987`
- BeijingAirQuality: Selective regresses significantly vs Simplex: **False**
- ETTm2: always-on delta `+0.001239` -> selective delta `+0.000118`; still significantly regresses: **False**; fixed or reduced: **True**

## Primary dependence-aware statistics (block-24)

| Dataset | Comparison | Mean Δ | 95% CI | P(Δ<0) | Excludes zero |
|---|---|---:|---|---:|---|
| ExchangeRate | Selective_vs_Simplex | `-0.006839` | [-0.009060, -0.004766] | 1.000 | True |
| ExchangeRate | Selective_vs_AlwaysOn | `+0.000357` | [+0.000124, +0.000626] | 0.000 | True |
| ExchangeRate | Selective_vs_SelectiveShuffled | `-0.006840` | [-0.009250, -0.004679] | 1.000 | True |
| Traffic | Selective_vs_Simplex | `-0.010332` | [-0.011698, -0.009041] | 1.000 | True |
| Traffic | Selective_vs_AlwaysOn | `+0.000133` | [-0.000289, +0.000544] | 0.271 | False |
| Traffic | Selective_vs_SelectiveShuffled | `-0.007069` | [-0.008232, -0.005920] | 1.000 | True |
| BeijingAirQuality | Selective_vs_Simplex | `-0.000298` | [-0.000919, +0.000284] | 0.848 | False |
| BeijingAirQuality | Selective_vs_AlwaysOn | `-0.000651` | [-0.001018, -0.000288] | 1.000 | True |
| BeijingAirQuality | Selective_vs_SelectiveShuffled | `-0.000649` | [-0.001262, -0.000021] | 0.978 | True |
| ETTm2 | Selective_vs_Simplex | `+0.000118` | [-0.000136, +0.000369] | 0.184 | False |
| ETTm2 | Selective_vs_AlwaysOn | `-0.001121` | [-0.001272, -0.000972] | 1.000 | True |
| ETTm2 | Selective_vs_SelectiveShuffled | `-0.000178` | [-0.000422, +0.000075] | 0.922 | False |

## Full dependence-aware statistics (all block lengths + phase)

| Dataset | Comparison | Test | Mean Δ | 95% CI | P(Δ<0) | Excludes zero |
|---|---|---|---:|---|---:|---|
| ExchangeRate | Selective_vs_Simplex | iid_paired_bootstrap | `-0.006839` | [-0.007724, -0.005964] |  | True |
| ExchangeRate | Selective_vs_Simplex | block_bootstrap_len12 | `-0.006839` | [-0.008916, -0.004899] | 1.0 | True |
| ExchangeRate | Selective_vs_Simplex | block_bootstrap_len24 | `-0.006839` | [-0.009060, -0.004766] | 1.0 | True |
| ExchangeRate | Selective_vs_Simplex | block_bootstrap_len48 | `-0.006839` | [-0.009462, -0.004578] | 1.0 | True |
| ExchangeRate | Selective_vs_Simplex | every_12th_window_phase_bootstrap | `-0.006837` | [-0.007506, -0.006204] | 1.0 | True |
| ExchangeRate | Selective_vs_AlwaysOn | iid_paired_bootstrap | `+0.000357` | [+0.000236, +0.000483] |  | True |
| ExchangeRate | Selective_vs_AlwaysOn | block_bootstrap_len12 | `+0.000357` | [+0.000136, +0.000610] | 0.00039999998989515007 | True |
| ExchangeRate | Selective_vs_AlwaysOn | block_bootstrap_len24 | `+0.000357` | [+0.000124, +0.000626] | 0.0003000000142492354 | True |
| ExchangeRate | Selective_vs_AlwaysOn | block_bootstrap_len48 | `+0.000357` | [+0.000115, +0.000693] | 0.0012000000569969416 | True |
| ExchangeRate | Selective_vs_AlwaysOn | every_12th_window_phase_bootstrap | `+0.000357` | [+0.000248, +0.000468] | 0.0 | True |
| ExchangeRate | Selective_vs_SelectiveShuffled | iid_paired_bootstrap | `-0.006840` | [-0.007808, -0.005868] |  | True |
| ExchangeRate | Selective_vs_SelectiveShuffled | block_bootstrap_len12 | `-0.006840` | [-0.008950, -0.004939] | 1.0 | True |
| ExchangeRate | Selective_vs_SelectiveShuffled | block_bootstrap_len24 | `-0.006840` | [-0.009250, -0.004679] | 1.0 | True |
| ExchangeRate | Selective_vs_SelectiveShuffled | block_bootstrap_len48 | `-0.006840` | [-0.009805, -0.004370] | 1.0 | True |
| ExchangeRate | Selective_vs_SelectiveShuffled | every_12th_window_phase_bootstrap | `-0.006839` | [-0.007493, -0.006183] | 1.0 | True |
| Traffic | Selective_vs_Simplex | iid_paired_bootstrap | `-0.010332` | [-0.010859, -0.009815] |  | True |
| Traffic | Selective_vs_Simplex | block_bootstrap_len12 | `-0.010332` | [-0.011542, -0.009137] | 1.0 | True |
| Traffic | Selective_vs_Simplex | block_bootstrap_len24 | `-0.010332` | [-0.011698, -0.009041] | 1.0 | True |
| Traffic | Selective_vs_Simplex | block_bootstrap_len48 | `-0.010332` | [-0.011970, -0.008842] | 1.0 | True |
| Traffic | Selective_vs_Simplex | every_12th_window_phase_bootstrap | `-0.010331` | [-0.011631, -0.009004] | 1.0 | True |
| Traffic | Selective_vs_AlwaysOn | iid_paired_bootstrap | `+0.000133` | [-0.000075, +0.000342] |  | False |
| Traffic | Selective_vs_AlwaysOn | block_bootstrap_len12 | `+0.000133` | [-0.000261, +0.000497] | 0.24449999630451202 | False |
| Traffic | Selective_vs_AlwaysOn | block_bootstrap_len24 | `+0.000133` | [-0.000289, +0.000544] | 0.2712000012397766 | False |
| Traffic | Selective_vs_AlwaysOn | block_bootstrap_len48 | `+0.000133` | [-0.000318, +0.000567] | 0.2872999906539917 | False |
| Traffic | Selective_vs_AlwaysOn | every_12th_window_phase_bootstrap | `+0.000133` | [-0.000276, +0.000536] | 0.2590999901294708 | False |
| Traffic | Selective_vs_SelectiveShuffled | iid_paired_bootstrap | `-0.007069` | [-0.007653, -0.006464] |  | True |
| Traffic | Selective_vs_SelectiveShuffled | block_bootstrap_len12 | `-0.007069` | [-0.008170, -0.005951] | 1.0 | True |
| Traffic | Selective_vs_SelectiveShuffled | block_bootstrap_len24 | `-0.007069` | [-0.008232, -0.005920] | 1.0 | True |
| Traffic | Selective_vs_SelectiveShuffled | block_bootstrap_len48 | `-0.007069` | [-0.008391, -0.005748] | 1.0 | True |
| Traffic | Selective_vs_SelectiveShuffled | every_12th_window_phase_bootstrap | `-0.007068` | [-0.008458, -0.005662] | 1.0 | True |
| BeijingAirQuality | Selective_vs_Simplex | iid_paired_bootstrap | `-0.000298` | [-0.000567, -0.000034] |  | True |
| BeijingAirQuality | Selective_vs_Simplex | block_bootstrap_len12 | `-0.000298` | [-0.000887, +0.000280] | 0.8526999950408936 | False |
| BeijingAirQuality | Selective_vs_Simplex | block_bootstrap_len24 | `-0.000298` | [-0.000919, +0.000284] | 0.8475000262260437 | False |
| BeijingAirQuality | Selective_vs_Simplex | block_bootstrap_len48 | `-0.000298` | [-0.000947, +0.000262] | 0.8672999739646912 | False |
| BeijingAirQuality | Selective_vs_Simplex | every_12th_window_phase_bootstrap | `-0.000298` | [-0.000601, +0.000012] | 0.9688000082969666 | False |
| BeijingAirQuality | Selective_vs_AlwaysOn | iid_paired_bootstrap | `-0.000651` | [-0.000833, -0.000473] |  | True |
| BeijingAirQuality | Selective_vs_AlwaysOn | block_bootstrap_len12 | `-0.000651` | [-0.000988, -0.000314] | 0.9997000098228455 | True |
| BeijingAirQuality | Selective_vs_AlwaysOn | block_bootstrap_len24 | `-0.000651` | [-0.001018, -0.000288] | 0.9998999834060669 | True |
| BeijingAirQuality | Selective_vs_AlwaysOn | block_bootstrap_len48 | `-0.000651` | [-0.001001, -0.000255] | 0.9995999932289124 | True |
| BeijingAirQuality | Selective_vs_AlwaysOn | every_12th_window_phase_bootstrap | `-0.000652` | [-0.000808, -0.000496] | 1.0 | True |
| BeijingAirQuality | Selective_vs_SelectiveShuffled | iid_paired_bootstrap | `-0.000649` | [-0.000969, -0.000322] |  | True |
| BeijingAirQuality | Selective_vs_SelectiveShuffled | block_bootstrap_len12 | `-0.000649` | [-0.001234, -0.000044] | 0.982200026512146 | True |
| BeijingAirQuality | Selective_vs_SelectiveShuffled | block_bootstrap_len24 | `-0.000649` | [-0.001262, -0.000021] | 0.9782999753952026 | True |
| BeijingAirQuality | Selective_vs_SelectiveShuffled | block_bootstrap_len48 | `-0.000649` | [-0.001289, -0.000048] | 0.9815000295639038 | True |
| BeijingAirQuality | Selective_vs_SelectiveShuffled | every_12th_window_phase_bootstrap | `-0.000649` | [-0.001005, -0.000273] | 0.9997000098228455 | True |
| ETTm2 | Selective_vs_Simplex | iid_paired_bootstrap | `+0.000118` | [-0.000009, +0.000245] |  | False |
| ETTm2 | Selective_vs_Simplex | block_bootstrap_len12 | `+0.000118` | [-0.000114, +0.000342] | 0.1639000028371811 | False |
| ETTm2 | Selective_vs_Simplex | block_bootstrap_len24 | `+0.000118` | [-0.000136, +0.000369] | 0.1842000037431717 | False |
| ETTm2 | Selective_vs_Simplex | block_bootstrap_len48 | `+0.000118` | [-0.000139, +0.000362] | 0.1867000013589859 | False |
| ETTm2 | Selective_vs_Simplex | every_12th_window_phase_bootstrap | `+0.000118` | [+0.000038, +0.000192] | 0.0017999999690800905 | True |
| ETTm2 | Selective_vs_AlwaysOn | iid_paired_bootstrap | `-0.001121` | [-0.001203, -0.001037] |  | True |
| ETTm2 | Selective_vs_AlwaysOn | block_bootstrap_len12 | `-0.001121` | [-0.001260, -0.000983] | 1.0 | True |
| ETTm2 | Selective_vs_AlwaysOn | block_bootstrap_len24 | `-0.001121` | [-0.001272, -0.000972] | 1.0 | True |
| ETTm2 | Selective_vs_AlwaysOn | block_bootstrap_len48 | `-0.001121` | [-0.001269, -0.000967] | 1.0 | True |
| ETTm2 | Selective_vs_AlwaysOn | every_12th_window_phase_bootstrap | `-0.001121` | [-0.001167, -0.001070] | 1.0 | True |
| ETTm2 | Selective_vs_SelectiveShuffled | iid_paired_bootstrap | `-0.000178` | [-0.000341, -0.000018] |  | True |
| ETTm2 | Selective_vs_SelectiveShuffled | block_bootstrap_len12 | `-0.000178` | [-0.000409, +0.000054] | 0.9326000213623047 | False |
| ETTm2 | Selective_vs_SelectiveShuffled | block_bootstrap_len24 | `-0.000178` | [-0.000422, +0.000075] | 0.9222999811172485 | False |
| ETTm2 | Selective_vs_SelectiveShuffled | block_bootstrap_len48 | `-0.000178` | [-0.000427, +0.000071] | 0.9214000105857849 | False |
| ETTm2 | Selective_vs_SelectiveShuffled | every_12th_window_phase_bootstrap | `-0.000178` | [-0.000279, -0.000084] | 0.9998999834060669 | True |

## Gate training (router_train OOF regularization selection)

| Dataset | Probe variant | L2 | OOF logloss | OOF accuracy | Selected |
|---|---|---:|---:|---:|---|
| ExchangeRate | real | 0.001 | 0.5590 | 0.755 |  |
| ExchangeRate | real | 0.01 | 0.5588 | 0.756 | <-- selected |
| ExchangeRate | real | 0.1 | 0.5598 | 0.755 |  |
| ExchangeRate | real | 1.0 | 0.5596 | 0.755 |  |
| ExchangeRate | shuffled | 0.001 | 0.5471 | 0.767 |  |
| ExchangeRate | shuffled | 0.01 | 0.5457 | 0.767 | <-- selected |
| ExchangeRate | shuffled | 0.1 | 0.5467 | 0.767 |  |
| ExchangeRate | shuffled | 1.0 | 0.5979 | 0.744 |  |
| Traffic | real | 0.001 | 0.4551 | 0.796 |  |
| Traffic | real | 0.01 | 0.4549 | 0.797 | <-- selected |
| Traffic | real | 0.1 | 0.4616 | 0.800 |  |
| Traffic | real | 1.0 | 0.4884 | 0.798 |  |
| Traffic | shuffled | 0.001 | 0.5022 | 0.780 | <-- selected |
| Traffic | shuffled | 0.01 | 0.5038 | 0.781 |  |
| Traffic | shuffled | 0.1 | 0.5349 | 0.716 |  |
| Traffic | shuffled | 1.0 | 0.6120 | 0.631 |  |
| BeijingAirQuality | real | 0.001 | 0.6808 | 0.559 | <-- selected |
| BeijingAirQuality | real | 0.01 | 0.6810 | 0.559 |  |
| BeijingAirQuality | real | 0.1 | 0.6831 | 0.553 |  |
| BeijingAirQuality | real | 1.0 | 0.6871 | 0.550 |  |
| BeijingAirQuality | shuffled | 0.001 | 0.6846 | 0.542 |  |
| BeijingAirQuality | shuffled | 0.01 | 0.6844 | 0.541 |  |
| BeijingAirQuality | shuffled | 0.1 | 0.6842 | 0.542 | <-- selected |
| BeijingAirQuality | shuffled | 1.0 | 0.6859 | 0.550 |  |
| ETTm2 | real | 0.001 | 0.6685 | 0.595 |  |
| ETTm2 | real | 0.01 | 0.6675 | 0.596 | <-- selected |
| ETTm2 | real | 0.1 | 0.6710 | 0.590 |  |
| ETTm2 | real | 1.0 | 0.6849 | 0.565 |  |
| ETTm2 | shuffled | 0.001 | 0.6775 | 0.572 |  |
| ETTm2 | shuffled | 0.01 | 0.6762 | 0.576 |  |
| ETTm2 | shuffled | 0.1 | 0.6752 | 0.584 | <-- selected |
| ETTm2 | shuffled | 1.0 | 0.6772 | 0.584 |  |

## Gate weights (real probe, standardized-feature coefficients)

| Dataset | probe_best_loss | probe_gap_best_second | probe_std | probe_agrees_simplex_top | l1_weight_change | max_abs_weight_change | changes_top_expert | entropy_diff | mean_pairwise_disagreement | bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ExchangeRate | +0.138 | +0.110 | +0.435 | +0.048 | +0.116 | +0.116 | -0.025 | +0.087 | -0.260 | +1.346 |
| Traffic | +0.093 | -0.126 | +0.706 | -0.055 | -0.034 | -0.034 | +0.168 | +0.199 | -0.347 | +1.554 |
| BeijingAirQuality | +0.063 | -0.114 | +0.398 | -0.034 | +0.010 | +0.010 | +0.093 | -0.096 | -0.095 | +0.217 |
| ETTm2 | -0.071 | +0.091 | +0.314 | -0.073 | +0.076 | +0.076 | -0.110 | +0.361 | +0.157 | +0.053 |

## Critical diagnostics (Section 9)

### ExchangeRate

- {'dataset': 'ExchangeRate', 'section': 'A_gate_behavior', 'mean_gate': 0.7342897653579712, 'median_gate': 0.7414722442626953, 'fraction_gate_lt_0.1': 0.0, 'fraction_gate_gt_0.9': 0.02905740588903427}
- {'dataset': 'ExchangeRate', 'section': 'B_gate_usefulness', 'num_trusted_windows': 41, 'num_rejected_windows': 0, 'mae_when_trusted_simplex': 0.18165448307991028, 'mae_when_trusted_selective': 0.1753264218568802, 'probe_gain_on_trusted': 0.006210021674633026, 'mae_when_rejected_simplex': nan, 'mae_when_rejected_selective': nan, 'probe_gain_on_rejected': nan}
- {'dataset': 'ExchangeRate', 'section': 'C_simplex_agreement', 'group': 'agrees', 'num_windows': 1242, 'mean_gate': 0.748491108417511, 'probe_gain_always_on': 0.006969213951379061}
- {'dataset': 'ExchangeRate', 'section': 'C_simplex_agreement', 'group': 'disagrees', 'num_windows': 169, 'mean_gate': 0.6299231052398682, 'probe_gain_always_on': 0.00886105839163065}
- {'dataset': 'ExchangeRate', 'section': 'D_top_expert_changes', 'fraction_changed_always_on': 0.11481218785047531, 'fraction_changed_selective': 0.1098511666059494, 'probe_gain_when_always_on_changes_top': 0.008778895251452923, 'probe_gain_when_always_on_keeps_top': 0.006990473251789808}
- {'dataset': 'ExchangeRate', 'section': 'E_correction_magnitude_bucket', 'bucket': 'small', 'num_windows': 471, 'mean_l1_change': 0.6301692128181458, 'probe_gain_always_on': 0.006554469931870699, 'mean_gate': 0.6964979767799377}
- {'dataset': 'ExchangeRate', 'section': 'E_correction_magnitude_bucket', 'bucket': 'medium', 'num_windows': 470, 'mean_l1_change': 0.8320021629333496, 'probe_gain_always_on': 0.007536755409091711, 'mean_gate': 0.7372405529022217}
- {'dataset': 'ExchangeRate', 'section': 'E_correction_magnitude_bucket', 'bucket': 'large', 'num_windows': 470, 'mean_l1_change': 1.1407597064971924, 'probe_gain_always_on': 0.007497556507587433, 'mean_gate': 0.7692112922668457}

### Traffic

- {'dataset': 'Traffic', 'section': 'A_gate_behavior', 'mean_gate': 0.767300546169281, 'median_gate': 0.7942216992378235, 'fraction_gate_lt_0.1': 0.0, 'fraction_gate_gt_0.9': 0.21164020895957947}
- {'dataset': 'Traffic', 'section': 'B_gate_usefulness', 'num_trusted_windows': 720, 'num_rejected_windows': 0, 'mae_when_trusted_simplex': 0.23751966655254364, 'mae_when_trusted_selective': 0.21862420439720154, 'probe_gain_on_trusted': 0.019286850467324257, 'mae_when_rejected_simplex': nan, 'mae_when_rejected_selective': nan, 'probe_gain_on_rejected': nan}
- {'dataset': 'Traffic', 'section': 'C_simplex_agreement', 'group': 'agrees', 'num_windows': 2229, 'mean_gate': 0.7458580136299133, 'probe_gain_always_on': 0.013201271183788776}
- {'dataset': 'Traffic', 'section': 'C_simplex_agreement', 'group': 'disagrees', 'num_windows': 1173, 'mean_gate': 0.8080469369888306, 'probe_gain_always_on': 0.005264583975076675}
- {'dataset': 'Traffic', 'section': 'D_top_expert_changes', 'fraction_changed_always_on': 0.07818929851055145, 'fraction_changed_selective': 0.07231040298938751, 'probe_gain_when_always_on_changes_top': -0.03077702410519123, 'probe_gain_when_always_on_keeps_top': 0.01396290771663189}
- {'dataset': 'Traffic', 'section': 'E_correction_magnitude_bucket', 'bucket': 'small', 'num_windows': 1134, 'mean_l1_change': 0.29407668113708496, 'probe_gain_always_on': 0.00771334720775485, 'mean_gate': 0.7600416541099548}
- {'dataset': 'Traffic', 'section': 'E_correction_magnitude_bucket', 'bucket': 'medium', 'num_windows': 1134, 'mean_l1_change': 0.4314751923084259, 'probe_gain_always_on': 0.015784241259098053, 'mean_gate': 0.8316918611526489}
- {'dataset': 'Traffic', 'section': 'E_correction_magnitude_bucket', 'bucket': 'large', 'num_windows': 1134, 'mean_l1_change': 0.4745844006538391, 'probe_gain_always_on': 0.007896583527326584, 'mean_gate': 0.710168182849884}

### BeijingAirQuality

- {'dataset': 'BeijingAirQuality', 'section': 'A_gate_behavior', 'mean_gate': 0.5388784408569336, 'median_gate': 0.5336835980415344, 'fraction_gate_lt_0.1': 0.0, 'fraction_gate_gt_0.9': 0.0}
- {'dataset': 'BeijingAirQuality', 'section': 'B_gate_usefulness', 'num_trusted_windows': 0, 'num_rejected_windows': 0, 'mae_when_trusted_simplex': nan, 'mae_when_trusted_selective': nan, 'probe_gain_on_trusted': nan, 'mae_when_rejected_simplex': nan, 'mae_when_rejected_selective': nan, 'probe_gain_on_rejected': nan}
- {'dataset': 'BeijingAirQuality', 'section': 'C_simplex_agreement', 'group': 'agrees', 'num_windows': 3578, 'mean_gate': 0.5399441123008728, 'probe_gain_always_on': 0.000411138404160738}
- {'dataset': 'BeijingAirQuality', 'section': 'C_simplex_agreement', 'group': 'disagrees', 'num_windows': 3515, 'mean_gate': 0.5377936363220215, 'probe_gain_always_on': -0.0011321945348754525}
- {'dataset': 'BeijingAirQuality', 'section': 'D_top_expert_changes', 'fraction_changed_always_on': 0.16410546004772186, 'fraction_changed_selective': 0.05935429409146309, 'probe_gain_when_always_on_changes_top': -0.00040118698962032795, 'probe_gain_when_always_on_keeps_top': -0.00034434624831192195}
- {'dataset': 'BeijingAirQuality', 'section': 'E_correction_magnitude_bucket', 'bucket': 'small', 'num_windows': 2365, 'mean_l1_change': 0.35385996103286743, 'probe_gain_always_on': -0.0018080971203744411, 'mean_gate': 0.5292842984199524}
- {'dataset': 'BeijingAirQuality', 'section': 'E_correction_magnitude_bucket', 'bucket': 'medium', 'num_windows': 2364, 'mean_l1_change': 0.4026792645454407, 'probe_gain_always_on': 1.7560187188792042e-05, 'mean_gate': 0.5565631985664368}
- {'dataset': 'BeijingAirQuality', 'section': 'E_correction_magnitude_bucket', 'bucket': 'large', 'num_windows': 2364, 'mean_l1_change': 0.4753352999687195, 'probe_gain_always_on': 0.0007301299483515322, 'mean_gate': 0.5307918190956116}

### ETTm2

- {'dataset': 'ETTm2', 'section': 'A_gate_behavior', 'mean_gate': 0.4813297986984253, 'median_gate': 0.45372650027275085, 'fraction_gate_lt_0.1': 0.0, 'fraction_gate_gt_0.9': 0.0}
- {'dataset': 'ETTm2', 'section': 'B_gate_usefulness', 'num_trusted_windows': 0, 'num_rejected_windows': 0, 'mae_when_trusted_simplex': nan, 'mae_when_trusted_selective': nan, 'probe_gain_on_trusted': nan, 'mae_when_rejected_simplex': nan, 'mae_when_rejected_selective': nan, 'probe_gain_on_rejected': nan}
- {'dataset': 'ETTm2', 'section': 'C_simplex_agreement', 'group': 'agrees', 'num_windows': 3689, 'mean_gate': 0.43253543972969055, 'probe_gain_always_on': -0.0017806004034355283}
- {'dataset': 'ETTm2', 'section': 'C_simplex_agreement', 'group': 'disagrees', 'num_windows': 7724, 'mean_gate': 0.5046341419219971, 'probe_gain_always_on': -0.0009799966355785728}
- {'dataset': 'ETTm2', 'section': 'D_top_expert_changes', 'fraction_changed_always_on': 0.6436519622802734, 'fraction_changed_selective': 0.6100937724113464, 'probe_gain_when_always_on_changes_top': -0.001072242041118443, 'probe_gain_when_always_on_keeps_top': -0.0015395720256492496}
- {'dataset': 'ETTm2', 'section': 'E_correction_magnitude_bucket', 'bucket': 'small', 'num_windows': 3805, 'mean_l1_change': 0.5432343482971191, 'probe_gain_always_on': -0.0009396735113114119, 'mean_gate': 0.49741047620773315}
- {'dataset': 'ETTm2', 'section': 'E_correction_magnitude_bucket', 'bucket': 'medium', 'num_windows': 3804, 'mean_l1_change': 0.7761684656143188, 'probe_gain_always_on': -0.0011919756652787328, 'mean_gate': 0.4716666638851166}
- {'dataset': 'ETTm2', 'section': 'E_correction_magnitude_bucket', 'bucket': 'large', 'num_windows': 3804, 'mean_l1_change': 0.901454508304596, 'probe_gain_always_on': -0.0015847517643123865, 'mean_gate': 0.47490811347961426}

## ETTm2 failure analysis (Section 10)

- **fraction_helps**: 0.42320162057876587
- **fraction_hurts**: 0.5767983794212341
- **avg_magnitude_help**: 0.00734687177464366
- **avg_magnitude_harm**: -0.007538131438195705
- **mean_probe_std_on_harmful**: 0.012209293432533741
- **mean_probe_std_on_helpful**: 0.01285043265670538
- **fraction_disagree_on_harmful**: 0.6621600985527039
- **fraction_disagree_on_helpful**: 0.6966874003410339
- **mean_l1_change_on_harmful**: 0.7366343140602112
- **mean_l1_change_on_helpful**: 0.745221734046936
- **mean_gate_on_harmful**: 0.470596581697464
- **mean_gate_on_helpful**: 0.495958536863327
- **gate_successfully_lower_on_harmful**: True

## Weight-concentration analysis

| Dataset | Method | Mean entropy | Mean max weight | Mean eff. #experts | Fraction top-expert changed |
|---|---|---:|---:|---:|---:|
| ExchangeRate | Simplex | 1.0982 | 0.3435 | 2.998 | 0.000 |
| ExchangeRate | Simplex_AlwaysOnProbe | 0.5291 | 0.7528 | 1.574 | 0.115 |
| ExchangeRate | Simplex_SelectiveProbe | 0.6432 | 0.6985 | 1.727 | 0.110 |
| ExchangeRate | Simplex_SelectiveShuffledProbe | 0.8943 | 0.5459 | 2.314 | 0.641 |
| Traffic | Simplex | 0.9248 | 0.5099 | 2.331 | 0.000 |
| Traffic | Simplex_AlwaysOnProbe | 0.7983 | 0.6760 | 1.905 | 0.078 |
| Traffic | Simplex_SelectiveProbe | 0.8428 | 0.6392 | 2.021 | 0.072 |
| Traffic | Simplex_SelectiveShuffledProbe | 0.9009 | 0.5489 | 2.237 | 0.000 |
| BeijingAirQuality | Simplex | 1.0512 | 0.4389 | 2.755 | 0.000 |
| BeijingAirQuality | Simplex_AlwaysOnProbe | 0.9493 | 0.5646 | 2.341 | 0.164 |
| BeijingAirQuality | Simplex_SelectiveProbe | 1.0181 | 0.5085 | 2.589 | 0.059 |
| BeijingAirQuality | Simplex_SelectiveShuffledProbe | 1.0274 | 0.4732 | 2.651 | 0.360 |
| ETTm2 | Simplex | 1.0571 | 0.4127 | 2.786 | 0.000 |
| ETTm2 | Simplex_AlwaysOnProbe | 0.7093 | 0.7108 | 1.760 | 0.644 |
| ETTm2 | Simplex_SelectiveProbe | 0.9564 | 0.5487 | 2.367 | 0.610 |
| ETTm2 | Simplex_SelectiveShuffledProbe | 0.9900 | 0.5173 | 2.496 | 0.560 |

## Integrity

- **ExchangeRate**: PASS (reproduction of prior simplex_probe experiment ok: True; zero-gate reproduces Simplex: True; target-corruption invariant: True; checkpoints unchanged: True)
- **Traffic**: PASS (reproduction of prior simplex_probe experiment ok: True; zero-gate reproduces Simplex: True; target-corruption invariant: True; checkpoints unchanged: True)
- **BeijingAirQuality**: PASS (reproduction of prior simplex_probe experiment ok: True; zero-gate reproduces Simplex: True; target-corruption invariant: True; checkpoints unchanged: True)
- **ETTm2**: PASS (reproduction of prior simplex_probe experiment ok: True; zero-gate reproduces Simplex: True; target-corruption invariant: True; checkpoints unchanged: True)

## Answers

**1. Does Selective Probe beat base Simplex?** See primary results table; block-24 significance in the dependence table above.
**2. Does it preserve the ExchangeRate gain?** Fraction of always-on gain preserved: `0.950`.
**3. Does it preserve the Traffic gain?** Fraction of always-on gain preserved: `0.987`.
**4. Does it avoid the BeijingAirQuality non-benefit?** Significant regression: False.
**5. Does it fix or reduce the ETTm2 regression?** Always-on Δ +0.001239 -> Selective Δ +0.000118; still significant: False.
**6. Does Real Selective Probe beat Selective ShuffledProbe?** By point estimate on 4/4; block-24 significant on 3/4.
**7. Can the gate predict when Probe will help?** See the gate-training OOF logloss/accuracy table and Section 9B (MAE/probe-gain conditioned on trusted vs rejected windows) above.
**8. Which observable features are most associated with Probe usefulness?** See the gate-weights table -- larger-magnitude coefficients (either sign) indicate stronger association, after standardization.
**9. Is Probe/Simplex disagreement useful for deciding when to trust Probe?** See Section 9C (`C_simplex_agreement` rows: probe_gain_always_on and mean_gate split by agree/disagree).
**10. Are large Probe weight corrections more dangerous?** See Section 9E (`E_correction_magnitude_bucket` rows: probe_gain_always_on by small/medium/large L1 weight-change bucket).
**11. Does the gate mainly abstain or vary meaningfully?** See Section 9A (`A_gate_behavior`: mean/median gate, fraction <0.1, fraction >0.9) per dataset above.
**12. Should the next experiment be Frozen COSTAR + Probe?** PROCEED TO FROZEN COSTAR + PROBE

## Decision: PROMISING

Selective Probe preserves most of the ExchangeRate/Traffic gains, avoids the BeijingAirQuality/ETTm2 harm, and clearly beats Selective ShuffledProbe. Recommend proceeding to Frozen COSTAR + Probe.

## Recommendation: **PROCEED TO FROZEN COSTAR + PROBE**

## Hard rule compliance

```text
TEST SET ACCESSED: NO
FORECASTING EXPERTS RETRAINED: NO
LEARNEDPROBE ARCHITECTURE/LOSS/TRAINING MODIFIED: NO
SIMPLEX MODIFIED: NO (base weights reproduced from the frozen fit_simplex_weights function)
OTHER ROUTERS (Frozen/Online COSTAR, Top-1, Top-k, Ridge, Granger-Ramanathan) TOUCHED: NO
```
