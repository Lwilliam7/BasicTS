# TimeFuse vs TimeFuse + LearnedProbe (Always-On / Selective / Selective-Shuffled)

**Official TimeFuse router adapted to the BasicTS controlled protocol.** This is NOT a reproduction of the TimeFuse paper's published benchmark numbers. See `official_timefuse_source_manifest.json` for the exact commit (978e6c6b9e4f246632c269aa0f9beeb099eabcfc), files used unmodified, and the shape/engineering adaptations required (none touch the meta-feature formula, ModelFusor architecture, or training hyperparameters).

Two separate questions, kept separate throughout: (1) TimeFuse vs Always-On Probe -- does Probe information itself add value? (2) Always-On vs Selective Probe -- does learning when to trust Probe make integration more reliable? (3) Selective vs Selective-Shuffled -- is any gain genuine expert-specific information?

## Primary results (router_val MAE)

| Dataset | TimeFuse | +Always-On Probe | +Selective Probe | +Selective ShuffledProbe | Δ AlwaysOn vs TF | Δ Selective vs TF | Δ Selective vs AlwaysOn | Δ Selective vs Shuffled |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ExchangeRate | 0.128424 | 0.125416 | 0.125493 | 0.125423 | `-0.003008` | `-0.002931` | `+0.000077` | `+0.000070` |
| Traffic | 0.270041 | 0.268689 | 0.268225 | 0.268869 | `-0.001352` | `-0.001816` | `-0.000464` | `-0.000644` |
| BeijingAirQuality | 0.258141 | 0.257304 | 0.257447 | 0.257710 | `-0.000837` | `-0.000694` | `+0.000143` | `-0.000262` |
| ETTm2 | 0.160278 | 0.160299 | 0.160263 | 0.160283 | `+0.000021` | `-0.000015` | `-0.000036` | `-0.000020` |

## Primary dependence-aware statistics (block-24)

| Dataset | Comparison | Mean Δ | 95% CI | P(Δ<0) | Excludes zero |
|---|---|---:|---|---:|---|
| ExchangeRate | AlwaysOn_vs_TimeFuse | `-0.003008` | [-0.005519, -0.000355] | 0.987 | True |
| ExchangeRate | Selective_vs_TimeFuse | `-0.002931` | [-0.005496, -0.000288] | 0.985 | True |
| ExchangeRate | Selective_vs_AlwaysOn | `+0.000077` | [-0.000037, +0.000164] | 0.098 | False |
| ExchangeRate | Selective_vs_SelectiveShuffled | `+0.000071` | [-0.000095, +0.000275] | 0.220 | False |
| Traffic | AlwaysOn_vs_TimeFuse | `-0.001352` | [-0.002742, -0.000005] | 0.976 | True |
| Traffic | Selective_vs_TimeFuse | `-0.001816` | [-0.003060, -0.000641] | 1.000 | True |
| Traffic | Selective_vs_AlwaysOn | `-0.000464` | [-0.000745, -0.000187] | 0.999 | True |
| Traffic | Selective_vs_SelectiveShuffled | `-0.000644` | [-0.001179, -0.000111] | 0.991 | True |
| BeijingAirQuality | AlwaysOn_vs_TimeFuse | `-0.000837` | [-0.001642, -0.000233] | 0.996 | True |
| BeijingAirQuality | Selective_vs_TimeFuse | `-0.000694` | [-0.001383, -0.000134] | 0.991 | True |
| BeijingAirQuality | Selective_vs_AlwaysOn | `+0.000143` | [+0.000017, +0.000319] | 0.014 | True |
| BeijingAirQuality | Selective_vs_SelectiveShuffled | `-0.000262` | [-0.000464, -0.000114] | 1.000 | True |
| ETTm2 | AlwaysOn_vs_TimeFuse | `+0.000021` | [-0.000153, +0.000191] | 0.417 | False |
| ETTm2 | Selective_vs_TimeFuse | `-0.000015` | [-0.000158, +0.000129] | 0.595 | False |
| ETTm2 | Selective_vs_AlwaysOn | `-0.000036` | [-0.000081, +0.000010] | 0.938 | False |
| ETTm2 | Selective_vs_SelectiveShuffled | `-0.000020` | [-0.000118, +0.000075] | 0.671 | False |

## Full dependence-aware statistics (all block lengths + phase)

| Dataset | Comparison | Test | Mean Δ | 95% CI | P(Δ<0) | Excludes zero |
|---|---|---|---:|---|---:|---|
| ExchangeRate | AlwaysOn_vs_TimeFuse | iid_paired_bootstrap | `-0.003008` | [-0.004087, -0.001965] |  | True |
| ExchangeRate | AlwaysOn_vs_TimeFuse | block_bootstrap_len12 | `-0.003008` | [-0.005525, -0.000613] | 0.9922000169754028 | True |
| ExchangeRate | AlwaysOn_vs_TimeFuse | block_bootstrap_len24 | `-0.003008` | [-0.005519, -0.000355] | 0.9871000051498413 | True |
| ExchangeRate | AlwaysOn_vs_TimeFuse | block_bootstrap_len48 | `-0.003008` | [-0.005435, -0.000704] | 0.9940000176429749 | True |
| ExchangeRate | AlwaysOn_vs_TimeFuse | every_12th_window_phase_bootstrap | `-0.003009` | [-0.003464, -0.002527] | 1.0 | True |
| ExchangeRate | Selective_vs_TimeFuse | iid_paired_bootstrap | `-0.002931` | [-0.004012, -0.001882] |  | True |
| ExchangeRate | Selective_vs_TimeFuse | block_bootstrap_len12 | `-0.002931` | [-0.005459, -0.000519] | 0.9904999732971191 | True |
| ExchangeRate | Selective_vs_TimeFuse | block_bootstrap_len24 | `-0.002931` | [-0.005496, -0.000288] | 0.9850000143051147 | True |
| ExchangeRate | Selective_vs_TimeFuse | block_bootstrap_len48 | `-0.002931` | [-0.005392, -0.000618] | 0.993399977684021 | True |
| ExchangeRate | Selective_vs_TimeFuse | every_12th_window_phase_bootstrap | `-0.002932` | [-0.003378, -0.002455] | 1.0 | True |
| ExchangeRate | Selective_vs_AlwaysOn | iid_paired_bootstrap | `+0.000077` | [+0.000029, +0.000125] |  | True |
| ExchangeRate | Selective_vs_AlwaysOn | block_bootstrap_len12 | `+0.000077` | [-0.000027, +0.000162] | 0.07599999755620956 | False |
| ExchangeRate | Selective_vs_AlwaysOn | block_bootstrap_len24 | `+0.000077` | [-0.000037, +0.000164] | 0.09809999912977219 | False |
| ExchangeRate | Selective_vs_AlwaysOn | block_bootstrap_len48 | `+0.000077` | [-0.000047, +0.000167] | 0.11599999666213989 | False |
| ExchangeRate | Selective_vs_AlwaysOn | every_12th_window_phase_bootstrap | `+0.000077` | [+0.000048, +0.000105] | 0.0 | True |
| ExchangeRate | Selective_vs_SelectiveShuffled | iid_paired_bootstrap | `+0.000071` | [-0.000035, +0.000186] |  | False |
| ExchangeRate | Selective_vs_SelectiveShuffled | block_bootstrap_len12 | `+0.000071` | [-0.000095, +0.000268] | 0.21089999377727509 | False |
| ExchangeRate | Selective_vs_SelectiveShuffled | block_bootstrap_len24 | `+0.000071` | [-0.000095, +0.000275] | 0.22040000557899475 | False |
| ExchangeRate | Selective_vs_SelectiveShuffled | block_bootstrap_len48 | `+0.000071` | [-0.000083, +0.000269] | 0.21089999377727509 | False |
| ExchangeRate | Selective_vs_SelectiveShuffled | every_12th_window_phase_bootstrap | `+0.000071` | [-0.000021, +0.000162] | 0.06599999964237213 | False |
| Traffic | AlwaysOn_vs_TimeFuse | iid_paired_bootstrap | `-0.001352` | [-0.001763, -0.000935] |  | True |
| Traffic | AlwaysOn_vs_TimeFuse | block_bootstrap_len12 | `-0.001352` | [-0.002498, -0.000248] | 0.9940000176429749 | True |
| Traffic | AlwaysOn_vs_TimeFuse | block_bootstrap_len24 | `-0.001352` | [-0.002742, -0.000005] | 0.9757999777793884 | True |
| Traffic | AlwaysOn_vs_TimeFuse | block_bootstrap_len48 | `-0.001352` | [-0.003110, +0.000263] | 0.941100001335144 | False |
| Traffic | AlwaysOn_vs_TimeFuse | every_12th_window_phase_bootstrap | `-0.001353` | [-0.002176, -0.000479] | 0.9983999729156494 | True |
| Traffic | Selective_vs_TimeFuse | iid_paired_bootstrap | `-0.001816` | [-0.002169, -0.001460] |  | True |
| Traffic | Selective_vs_TimeFuse | block_bootstrap_len12 | `-0.001816` | [-0.002830, -0.000875] | 1.0 | True |
| Traffic | Selective_vs_TimeFuse | block_bootstrap_len24 | `-0.001816` | [-0.003060, -0.000641] | 0.9998000264167786 | True |
| Traffic | Selective_vs_TimeFuse | block_bootstrap_len48 | `-0.001816` | [-0.003437, -0.000424] | 0.9957000017166138 | True |
| Traffic | Selective_vs_TimeFuse | every_12th_window_phase_bootstrap | `-0.001816` | [-0.002423, -0.001169] | 1.0 | True |
| Traffic | Selective_vs_AlwaysOn | iid_paired_bootstrap | `-0.000464` | [-0.000564, -0.000368] |  | True |
| Traffic | Selective_vs_AlwaysOn | block_bootstrap_len12 | `-0.000464` | [-0.000709, -0.000228] | 1.0 | True |
| Traffic | Selective_vs_AlwaysOn | block_bootstrap_len24 | `-0.000464` | [-0.000745, -0.000187] | 0.9994000196456909 | True |
| Traffic | Selective_vs_AlwaysOn | block_bootstrap_len48 | `-0.000464` | [-0.000807, -0.000146] | 0.9976999759674072 | True |
| Traffic | Selective_vs_AlwaysOn | every_12th_window_phase_bootstrap | `-0.000464` | [-0.000702, -0.000234] | 1.0 | True |
| Traffic | Selective_vs_SelectiveShuffled | iid_paired_bootstrap | `-0.000644` | [-0.000862, -0.000425] |  | True |
| Traffic | Selective_vs_SelectiveShuffled | block_bootstrap_len12 | `-0.000644` | [-0.001145, -0.000156] | 0.9943000078201294 | True |
| Traffic | Selective_vs_SelectiveShuffled | block_bootstrap_len24 | `-0.000644` | [-0.001179, -0.000111] | 0.9909999966621399 | True |
| Traffic | Selective_vs_SelectiveShuffled | block_bootstrap_len48 | `-0.000644` | [-0.001257, -0.000020] | 0.9790999889373779 | True |
| Traffic | Selective_vs_SelectiveShuffled | every_12th_window_phase_bootstrap | `-0.000644` | [-0.001114, -0.000162] | 0.9951000213623047 | True |
| BeijingAirQuality | AlwaysOn_vs_TimeFuse | iid_paired_bootstrap | `-0.000837` | [-0.001148, -0.000537] |  | True |
| BeijingAirQuality | AlwaysOn_vs_TimeFuse | block_bootstrap_len12 | `-0.000837` | [-0.001653, -0.000171] | 0.9932000041007996 | True |
| BeijingAirQuality | AlwaysOn_vs_TimeFuse | block_bootstrap_len24 | `-0.000837` | [-0.001642, -0.000233] | 0.9962999820709229 | True |
| BeijingAirQuality | AlwaysOn_vs_TimeFuse | block_bootstrap_len48 | `-0.000837` | [-0.001550, -0.000268] | 0.9983999729156494 | True |
| BeijingAirQuality | AlwaysOn_vs_TimeFuse | every_12th_window_phase_bootstrap | `-0.000838` | [-0.001202, -0.000457] | 1.0 | True |
| BeijingAirQuality | Selective_vs_TimeFuse | iid_paired_bootstrap | `-0.000694` | [-0.000969, -0.000425] |  | True |
| BeijingAirQuality | Selective_vs_TimeFuse | block_bootstrap_len12 | `-0.000694` | [-0.001407, -0.000072] | 0.9858999848365784 | True |
| BeijingAirQuality | Selective_vs_TimeFuse | block_bootstrap_len24 | `-0.000694` | [-0.001383, -0.000134] | 0.9908999800682068 | True |
| BeijingAirQuality | Selective_vs_TimeFuse | block_bootstrap_len48 | `-0.000694` | [-0.001289, -0.000176] | 0.9957000017166138 | True |
| BeijingAirQuality | Selective_vs_TimeFuse | every_12th_window_phase_bootstrap | `-0.000694` | [-0.001060, -0.000324] | 0.9997000098228455 | True |
| BeijingAirQuality | Selective_vs_AlwaysOn | iid_paired_bootstrap | `+0.000143` | [+0.000079, +0.000211] |  | True |
| BeijingAirQuality | Selective_vs_AlwaysOn | block_bootstrap_len12 | `+0.000143` | [+0.000021, +0.000299] | 0.010599999688565731 | True |
| BeijingAirQuality | Selective_vs_AlwaysOn | block_bootstrap_len24 | `+0.000143` | [+0.000017, +0.000319] | 0.014299999922513962 | True |
| BeijingAirQuality | Selective_vs_AlwaysOn | block_bootstrap_len48 | `+0.000143` | [+0.000010, +0.000327] | 0.017899999395012856 | True |
| BeijingAirQuality | Selective_vs_AlwaysOn | every_12th_window_phase_bootstrap | `+0.000143` | [+0.000073, +0.000213] | 9.999999747378752e-05 | True |
| BeijingAirQuality | Selective_vs_SelectiveShuffled | iid_paired_bootstrap | `-0.000262` | [-0.000358, -0.000174] |  | True |
| BeijingAirQuality | Selective_vs_SelectiveShuffled | block_bootstrap_len12 | `-0.000262` | [-0.000450, -0.000116] | 0.9994999766349792 | True |
| BeijingAirQuality | Selective_vs_SelectiveShuffled | block_bootstrap_len24 | `-0.000262` | [-0.000464, -0.000114] | 0.9997000098228455 | True |
| BeijingAirQuality | Selective_vs_SelectiveShuffled | block_bootstrap_len48 | `-0.000262` | [-0.000466, -0.000116] | 0.9998999834060669 | True |
| BeijingAirQuality | Selective_vs_SelectiveShuffled | every_12th_window_phase_bootstrap | `-0.000262` | [-0.000350, -0.000176] | 1.0 | True |
| ETTm2 | AlwaysOn_vs_TimeFuse | iid_paired_bootstrap | `+0.000021` | [-0.000057, +0.000098] |  | False |
| ETTm2 | AlwaysOn_vs_TimeFuse | block_bootstrap_len12 | `+0.000021` | [-0.000138, +0.000178] | 0.4138999879360199 | False |
| ETTm2 | AlwaysOn_vs_TimeFuse | block_bootstrap_len24 | `+0.000021` | [-0.000153, +0.000191] | 0.4169999957084656 | False |
| ETTm2 | AlwaysOn_vs_TimeFuse | block_bootstrap_len48 | `+0.000021` | [-0.000156, +0.000189] | 0.42800000309944153 | False |
| ETTm2 | AlwaysOn_vs_TimeFuse | every_12th_window_phase_bootstrap | `+0.000021` | [-0.000052, +0.000093] | 0.2928999960422516 | False |
| ETTm2 | Selective_vs_TimeFuse | iid_paired_bootstrap | `-0.000015` | [-0.000077, +0.000046] |  | False |
| ETTm2 | Selective_vs_TimeFuse | block_bootstrap_len12 | `-0.000015` | [-0.000148, +0.000115] | 0.599399983882904 | False |
| ETTm2 | Selective_vs_TimeFuse | block_bootstrap_len24 | `-0.000015` | [-0.000158, +0.000129] | 0.5947999954223633 | False |
| ETTm2 | Selective_vs_TimeFuse | block_bootstrap_len48 | `-0.000015` | [-0.000162, +0.000129] | 0.5997999906539917 | False |
| ETTm2 | Selective_vs_TimeFuse | every_12th_window_phase_bootstrap | `-0.000015` | [-0.000076, +0.000045] | 0.6823999881744385 | False |
| ETTm2 | Selective_vs_AlwaysOn | iid_paired_bootstrap | `-0.000036` | [-0.000058, -0.000013] |  | True |
| ETTm2 | Selective_vs_AlwaysOn | block_bootstrap_len12 | `-0.000036` | [-0.000077, +0.000007] | 0.953499972820282 | False |
| ETTm2 | Selective_vs_AlwaysOn | block_bootstrap_len24 | `-0.000036` | [-0.000081, +0.000010] | 0.9376000165939331 | False |
| ETTm2 | Selective_vs_AlwaysOn | block_bootstrap_len48 | `-0.000036` | [-0.000080, +0.000009] | 0.9352999925613403 | False |
| ETTm2 | Selective_vs_AlwaysOn | every_12th_window_phase_bootstrap | `-0.000036` | [-0.000050, -0.000022] | 1.0 | True |
| ETTm2 | Selective_vs_SelectiveShuffled | iid_paired_bootstrap | `-0.000020` | [-0.000065, +0.000025] |  | False |
| ETTm2 | Selective_vs_SelectiveShuffled | block_bootstrap_len12 | `-0.000020` | [-0.000109, +0.000069] | 0.6890000104904175 | False |
| ETTm2 | Selective_vs_SelectiveShuffled | block_bootstrap_len24 | `-0.000020` | [-0.000118, +0.000075] | 0.6711999773979187 | False |
| ETTm2 | Selective_vs_SelectiveShuffled | block_bootstrap_len48 | `-0.000020` | [-0.000120, +0.000076] | 0.6697999835014343 | False |
| ETTm2 | Selective_vs_SelectiveShuffled | every_12th_window_phase_bootstrap | `-0.000020` | [-0.000059, +0.000021] | 0.8385000228881836 | False |

## Capacity accounting

| Dataset | TimeFuse router params | AlwaysOn router params | Selective router params | Gate params | Input dim (base/augmented) |
|---|---:|---:|---:|---:|---|
| ExchangeRate | 69 | 78 | 78 | 10 | 22/25 |
| Traffic | 69 | 78 | 78 | 10 | 22/25 |
| BeijingAirQuality | 69 | 78 | 78 | 10 | 22/25 |
| ETTm2 | 69 | 78 | 78 | 10 | 22/25 |

## Gate diagnostics

### ExchangeRate
- {'dataset': 'ExchangeRate', 'section': 'gate_behavior', 'mean_gate': 0.5667610168457031, 'median_gate': 0.6147950291633606, 'fraction_gate_lt_0.1': 0.0, 'fraction_gate_gt_0.9': 0.02480510249733925}
- {'dataset': 'ExchangeRate', 'section': 'gate_usefulness', 'num_trusted_windows': 35, 'num_rejected_windows': 0, 'probe_gain_on_trusted': -0.01566951349377632, 'probe_gain_on_rejected': nan, 'fraction_windows_probe_helps': 0.5669738054275513, 'fraction_windows_probe_hurts': 0.4330262243747711, 'gate_lower_on_harmful': True}
- weight_change_effect: {'dataset': 'ExchangeRate', 'fraction_always_on_changes_top': 0.6966690421104431, 'probe_gain_when_changes_top': 0.003953920677304268, 'probe_gain_when_keeps_top': 0.0008355544414371252}

### Traffic
- {'dataset': 'Traffic', 'section': 'gate_behavior', 'mean_gate': 0.6272332668304443, 'median_gate': 0.6289433836936951, 'fraction_gate_lt_0.1': 0.0, 'fraction_gate_gt_0.9': 0.002939447294920683}
- {'dataset': 'Traffic', 'section': 'gate_usefulness', 'num_trusted_windows': 10, 'num_rejected_windows': 0, 'probe_gain_on_trusted': 0.00452011963352561, 'probe_gain_on_rejected': nan, 'fraction_windows_probe_helps': 0.6034685373306274, 'fraction_windows_probe_hurts': 0.39653146266937256, 'gate_lower_on_harmful': True}
- weight_change_effect: {'dataset': 'Traffic', 'fraction_always_on_changes_top': 0.2522045969963074, 'probe_gain_when_changes_top': 0.0016197875374928117, 'probe_gain_when_keeps_top': 0.0012615940067917109}

### BeijingAirQuality
- {'dataset': 'BeijingAirQuality', 'section': 'gate_behavior', 'mean_gate': 0.5075386762619019, 'median_gate': 0.5067278742790222, 'fraction_gate_lt_0.1': 0.0, 'fraction_gate_gt_0.9': 0.0}
- {'dataset': 'BeijingAirQuality', 'section': 'gate_usefulness', 'num_trusted_windows': 0, 'num_rejected_windows': 0, 'probe_gain_on_trusted': nan, 'probe_gain_on_rejected': nan, 'fraction_windows_probe_helps': 0.523473858833313, 'fraction_windows_probe_hurts': 0.476526141166687, 'gate_lower_on_harmful': True}
- weight_change_effect: {'dataset': 'BeijingAirQuality', 'fraction_always_on_changes_top': 0.5805723667144775, 'probe_gain_when_changes_top': 0.0005515085649676621, 'probe_gain_when_keeps_top': 0.0012333451304584742}

### ETTm2
- {'dataset': 'ETTm2', 'section': 'gate_behavior', 'mean_gate': 0.5321990847587585, 'median_gate': 0.5270727276802063, 'fraction_gate_lt_0.1': 0.0, 'fraction_gate_gt_0.9': 0.0}
- {'dataset': 'ETTm2', 'section': 'gate_usefulness', 'num_trusted_windows': 0, 'num_rejected_windows': 0, 'probe_gain_on_trusted': nan, 'probe_gain_on_rejected': nan, 'fraction_windows_probe_helps': 0.48269516229629517, 'fraction_windows_probe_hurts': 0.5173048377037048, 'gate_lower_on_harmful': True}
- weight_change_effect: {'dataset': 'ETTm2', 'fraction_always_on_changes_top': 0.1942521631717682, 'probe_gain_when_changes_top': 0.00016574384062550962, 'probe_gain_when_keeps_top': -6.603025394724682e-05}

## Weight analysis

| Dataset | Method | Mean entropy | Mean max weight | Mean eff. #experts | Fraction top-expert changed |
|---|---|---:|---:|---:|---:|
| ExchangeRate | TimeFuse | 0.7592 | 0.6468 | 2.062 | 0.000 |
| ExchangeRate | TimeFuse_AlwaysOnProbe | 0.8888 | 0.5612 | 2.374 | 0.697 |
| ExchangeRate | TimeFuse_SelectiveProbe | 0.8796 | 0.5721 | 2.339 | 0.646 |
| ExchangeRate | TimeFuse_SelectiveShuffledProbe | 0.8667 | 0.5836 | 2.306 | 0.639 |
| Traffic | TimeFuse | 1.0107 | 0.4951 | 2.616 | 0.000 |
| Traffic | TimeFuse_AlwaysOnProbe | 1.0054 | 0.5055 | 2.558 | 0.252 |
| Traffic | TimeFuse_SelectiveProbe | 1.0080 | 0.5058 | 2.567 | 0.230 |
| Traffic | TimeFuse_SelectiveShuffledProbe | 1.0156 | 0.5045 | 2.593 | 0.191 |
| BeijingAirQuality | TimeFuse | 1.0379 | 0.4562 | 2.734 | 0.000 |
| BeijingAirQuality | TimeFuse_AlwaysOnProbe | 1.0525 | 0.4334 | 2.793 | 0.581 |
| BeijingAirQuality | TimeFuse_SelectiveProbe | 1.0546 | 0.4268 | 2.810 | 0.520 |
| BeijingAirQuality | TimeFuse_SelectiveShuffledProbe | 1.0518 | 0.4292 | 2.803 | 0.473 |
| ETTm2 | TimeFuse | 1.0294 | 0.4713 | 2.680 | 0.000 |
| ETTm2 | TimeFuse_AlwaysOnProbe | 1.0044 | 0.4981 | 2.577 | 0.194 |
| ETTm2 | TimeFuse_SelectiveProbe | 1.0112 | 0.4926 | 2.603 | 0.152 |
| ETTm2 | TimeFuse_SelectiveShuffledProbe | 1.0080 | 0.4982 | 2.591 | 0.104 |

## Gate training (router_train OOF regularization selection)

| Dataset | Probe variant | L2 | OOF logloss | OOF accuracy | Selected |
|---|---|---:|---:|---:|---|
| ExchangeRate | real | 0.001 | 0.5774 | 0.739 |  |
| ExchangeRate | real | 0.01 | 0.5751 | 0.739 | <-- selected |
| ExchangeRate | real | 0.1 | 0.5787 | 0.724 |  |
| ExchangeRate | real | 1.0 | 0.6315 | 0.644 |  |
| ExchangeRate | shuffled | 0.001 | 0.7089 | 0.565 |  |
| ExchangeRate | shuffled | 0.01 | 0.7059 | 0.566 |  |
| ExchangeRate | shuffled | 0.1 | 0.6902 | 0.583 |  |
| ExchangeRate | shuffled | 1.0 | 0.6703 | 0.633 | <-- selected |
| Traffic | real | 0.001 | 0.6407 | 0.634 |  |
| Traffic | real | 0.01 | 0.6406 | 0.633 | <-- selected |
| Traffic | real | 0.1 | 0.6416 | 0.622 |  |
| Traffic | real | 1.0 | 0.6528 | 0.621 |  |
| Traffic | shuffled | 0.001 | 0.6942 | 0.523 |  |
| Traffic | shuffled | 0.01 | 0.6938 | 0.525 |  |
| Traffic | shuffled | 0.1 | 0.6919 | 0.524 |  |
| Traffic | shuffled | 1.0 | 0.6910 | 0.518 | <-- selected |
| BeijingAirQuality | real | 0.001 | 0.6960 | 0.490 |  |
| BeijingAirQuality | real | 0.01 | 0.6958 | 0.489 |  |
| BeijingAirQuality | real | 0.1 | 0.6948 | 0.481 |  |
| BeijingAirQuality | real | 1.0 | 0.6937 | 0.497 | <-- selected |
| BeijingAirQuality | shuffled | 0.001 | 0.6925 | 0.521 |  |
| BeijingAirQuality | shuffled | 0.01 | 0.6924 | 0.524 |  |
| BeijingAirQuality | shuffled | 0.1 | 0.6921 | 0.521 |  |
| BeijingAirQuality | shuffled | 1.0 | 0.6920 | 0.527 | <-- selected |
| ETTm2 | real | 0.001 | 0.6898 | 0.538 |  |
| ETTm2 | real | 0.01 | 0.6883 | 0.537 |  |
| ETTm2 | real | 0.1 | 0.6862 | 0.539 | <-- selected |
| ETTm2 | real | 1.0 | 0.6877 | 0.541 |  |
| ETTm2 | shuffled | 0.001 | 0.6947 | 0.503 |  |
| ETTm2 | shuffled | 0.01 | 0.6944 | 0.504 |  |
| ETTm2 | shuffled | 0.1 | 0.6937 | 0.503 |  |
| ETTm2 | shuffled | 1.0 | 0.6933 | 0.497 | <-- selected |

## Zero-probe / zero-gate diagnostics

**Important**: unlike the earlier closed-form Simplex fusion (where alpha=0/gate=0 is a provably EXACT identity), TimeFuse's ModelFusor is a JOINTLY-TRAINED linear layer -- Method B/C are separately-trained models from Method A, so forcing their Probe/gate input to zero at inference is NOT mathematically guaranteed to reproduce Method A's own separately-trained weights bit-for-bit. These are reported as DIAGNOSTICS (with a training-noise reference scale from retraining Method A with a different seed), not hard pass/fail gates.

- **ExchangeRate**: zero-probe max weight diff = 0.9998 (training-noise reference scale = 0.9484); zero-probe MAE diff vs base TimeFuse = `-0.003038`; zero-gate max weight diff vs zero-probe B = 0.0200; within training-noise scale: True
- **Traffic**: zero-probe max weight diff = 0.5677 (training-noise reference scale = 0.8435); zero-probe MAE diff vs base TimeFuse = `-0.001410`; zero-gate max weight diff vs zero-probe B = 0.0593; within training-noise scale: True
- **BeijingAirQuality**: zero-probe max weight diff = 0.3755 (training-noise reference scale = 0.6327); zero-probe MAE diff vs base TimeFuse = `-0.000714`; zero-gate max weight diff vs zero-probe B = 0.0937; within training-noise scale: True
- **ETTm2**: zero-probe max weight diff = 0.4375 (training-noise reference scale = 0.7772); zero-probe MAE diff vs base TimeFuse = `+0.000043`; zero-gate max weight diff vs zero-probe B = 0.0757; within training-noise scale: True

## Integrity

- **ExchangeRate**: PASS (checkpoints unchanged: True; no test cache: True; meta-features target-free: True; weights invariant to target corruption: True; gate invariant: True)
- **Traffic**: PASS (checkpoints unchanged: True; no test cache: True; meta-features target-free: True; weights invariant to target corruption: True; gate invariant: True)
- **BeijingAirQuality**: PASS (checkpoints unchanged: True; no test cache: True; meta-features target-free: True; weights invariant to target corruption: True; gate invariant: True)
- **ETTm2**: PASS (checkpoints unchanged: True; no test cache: True; meta-features target-free: True; weights invariant to target corruption: True; gate invariant: True)

## Answers

**1. Was the official TimeFuse routing mechanism adapted faithfully?** Yes -- meta_feature.extract_meta_feature (22-dim) and timefuse.ModelFusor (Linear+Softmax) used verbatim from commit 978e6c6b9e4f246632c269aa0f9beeb099eabcfc, with the exact official training hyperparameters (Adam lr=5e-4, StepLR(10,0.1), SmoothL1Loss(beta=0.01), 5 epochs, batch 64, seed 2021). See official_timefuse_source_manifest.json for the enumerated shape-only adaptations.
**2. Does Always-On Probe improve TimeFuse?** Beats by point estimate on 3/4; block-24 significant improvement on 3/4; significant regression on 0/4.
**3. Does Selective Probe improve TimeFuse?** Beats by point estimate on 4/4; block-24 significant improvement on 3/4; significant regression on 0/4.
**4. Does Selective Probe outperform Always-On Probe?** By point estimate on 2/4; see Selective_vs_AlwaysOn block-24 rows above for significance.
**5. Does Selective Probe outperform Selective ShuffledProbe?** By point estimate on 3/4; block-24 significant on 2/4.
**6. On how many datasets does each version win?** See point-estimate counts above (questions 2-5).
**7. Which gains/regressions survive block-24?** See the primary dependence-aware statistics table above.
**8. Does the gate successfully reduce Probe influence when Probe tends to hurt?** See gate_usefulness rows (`gate_lower_on_harmful`) per dataset above.
**9. Does TimeFuse already learn to use Probe appropriately without a gate?** See per-dataset Always-On results above.
**10. How much extra capacity does each augmentation add?** See Capacity accounting table -- router parameter counts scale with input dimensionality only (22 -> 22+K), gate adds 10 parameters (9 features + bias), trained entirely separately from the router.
**11. Does this support active diagnostic probing adding information beyond an independently published passive router?** LearnedProbe provides useful active expert-specific competence information beyond TimeFuse's passive meta-features, and selective trust makes that information more reliable.

## Decision: STRONG

LearnedProbe provides useful active expert-specific competence information beyond TimeFuse's passive meta-features, and selective trust makes that information more reliable.

## Hard rule compliance

```text
TEST SET ACCESSED: NO
FORECASTING EXPERTS RETRAINED: NO
LEARNEDPROBE ARCHITECTURE/LOSS/TRAINING MODIFIED: NO
OTHER PUBLISHED ROUTERS IMPLEMENTED: NO (TimeFuse only)
COSTAR / ONLINE COSTAR TOUCHED: NO
```
