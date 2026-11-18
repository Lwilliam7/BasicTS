# Strict Purged-OOF TimeFuse + LearnedProbe Mechanism Study

Corrects the in-sample stacking bug in `../timefuse_probe/`: every honest LearnedProbe / MatchedPassive-21 score used as a TimeFuse TRAINING feature here comes from a model retrained on a PURGED, causally-earlier prefix of router_train, reusing the exact fold machinery built and verified for `../fforma_probe/`. No Selective gate is used in this experiment.

## Section 1/2/9/14: mandatory causal assertions

| Dataset | Fold | Train target-end max | Eval origin min | Assertion holds | Purged windows |
|---|---:|---:|---:|---|---:|
| BeijingAirQuality | 0 | 12874 | 12874 | True | 11 |
| BeijingAirQuality | 1 | 17237 | 17237 | True | 11 |
| ETTm2 | 0 | 20650 | 20650 | True | 11 |
| ETTm2 | 1 | 27605 | 27605 | True | 11 |
| ExchangeRate | 0 | 2645 | 2645 | True | 11 |
| ExchangeRate | 1 | 3598 | 3598 | True | 11 |
| Traffic | 0 | 6230 | 6230 | True | 11 |
| Traffic | 1 | 8378 | 8378 | True | 11 |

| Dataset | router_train->router_val observability holds | max train target-end | min val origin | Common windows | Full legal windows |
|---|---|---:|---:|---:|---:|
| BeijingAirQuality | True | 21504 | 21600 | 8512 | 14186 |
| ETTm2 | True | 34464 | 34560 | 13696 | 22826 |
| ExchangeRate | True | 4456 | 4552 | 1693 | 2821 |
| Traffic | True | 10430 | 10526 | 4082 | 6804 |

## Section 35: primary results (router_val MAE)

| Dataset | Full | Common | +MatchedPassive21 | +LearnedProbe | +ShuffledProbe |
|---|---:|---:|---:|---:|---:|
| BeijingAirQuality | 0.258141 | 0.258126 | 0.257932 | 0.258011 | 0.257935 |
| ETTm2 | 0.160278 | 0.160669 | 0.160420 | 0.160500 | 0.160465 |
| ExchangeRate | 0.128424 | 0.126711 | 0.124332 | 0.124270 | 0.124015 |
| Traffic | 0.270041 | 0.270846 | 0.272680 | 0.273463 | 0.272275 |

### MSE

| Dataset | Full | Common | +MatchedPassive21 | +LearnedProbe | +ShuffledProbe |
|---|---:|---:|---:|---:|---:|
| BeijingAirQuality | 0.193536 | 0.193116 | 0.192973 | 0.192973 | 0.192909 |
| ETTm2 | 0.060408 | 0.060801 | 0.060504 | 0.060566 | 0.060532 |
| ExchangeRate | 0.036707 | 0.036315 | 0.035189 | 0.035209 | 0.035064 |
| Traffic | 0.337650 | 0.338991 | 0.337769 | 0.338800 | 0.337278 |

## LearnedProbe deltas

| Dataset | vs Common | % | vs Full | % | vs MatchedPassive21 | % | vs Shuffled | % | block-24 sig? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BeijingAirQuality | `-0.000116` | `+0.04%` | `-0.000130` | `+0.05%` | `+0.000079` | `-0.03%` | `+0.000076` | `-0.03%` | (none) |
| ETTm2 | `-0.000170` | `+0.11%` | `+0.000222` | `-0.14%` | `+0.000080` | `-0.05%` | `+0.000035` | `-0.02%` | C-, F+, P+ |
| ExchangeRate | `-0.002441` | `+1.93%` | `-0.004154` | `+3.23%` | `-0.000063` | `+0.05%` | `+0.000255` | `-0.21%` | C-, F- |
| Traffic | `+0.002617` | `-0.97%` | `+0.003422` | `-1.27%` | `+0.000783` | `-0.29%` | `+0.001188` | `-0.44%` | C+, F+, P+, S+ |

## Section 23: primary comparisons A-F (block-24)

| Dataset | Comparison | Mean Delta | 95% CI | P(Delta<0) | Excludes zero |
|---|---|---:|---|---:|---|
| BeijingAirQuality | A: Probe_vs_Common | `-0.000116` | [-0.000651, +0.000396] | 0.683 | False |
| BeijingAirQuality | B: Probe_vs_Full | `-0.000131` | [-0.000766, +0.000507] | 0.666 | False |
| BeijingAirQuality | C: Probe_vs_MatchedPassive (MOST IMPORTANT) | `+0.000079` | [-0.000069, +0.000213] | 0.159 | False |
| BeijingAirQuality | D: Probe_vs_Shuffled | `+0.000076` | [-0.000080, +0.000234] | 0.182 | False |
| BeijingAirQuality | E: MatchedPassive_vs_Common | `-0.000195` | [-0.000736, +0.000338] | 0.763 | False |
| BeijingAirQuality | F: MatchedPassive_vs_Full | `-0.000210` | [-0.000803, +0.000393] | 0.754 | False |
| ETTm2 | A: Probe_vs_Common | `-0.000170` | [-0.000343, -0.000001] | 0.975 | True |
| ETTm2 | B: Probe_vs_Full | `+0.000222` | [+0.000078, +0.000368] | 0.001 | True |
| ETTm2 | C: Probe_vs_MatchedPassive (MOST IMPORTANT) | `+0.000080` | [+0.000024, +0.000137] | 0.003 | True |
| ETTm2 | D: Probe_vs_Shuffled | `+0.000035` | [-0.000022, +0.000095] | 0.112 | False |
| ETTm2 | E: MatchedPassive_vs_Common | `-0.000249` | [-0.000446, -0.000057] | 0.994 | True |
| ETTm2 | F: MatchedPassive_vs_Full | `+0.000143` | [-0.000003, +0.000289] | 0.028 | False |
| ExchangeRate | A: Probe_vs_Common | `-0.002441` | [-0.004872, -0.000123] | 0.981 | True |
| ExchangeRate | B: Probe_vs_Full | `-0.004154` | [-0.006691, -0.001441] | 0.998 | True |
| ExchangeRate | C: Probe_vs_MatchedPassive (MOST IMPORTANT) | `-0.000063` | [-0.000241, +0.000164] | 0.665 | False |
| ExchangeRate | D: Probe_vs_Shuffled | `+0.000255` | [-0.000012, +0.000598] | 0.032 | False |
| ExchangeRate | E: MatchedPassive_vs_Common | `-0.002378` | [-0.004857, -0.000067] | 0.978 | True |
| ExchangeRate | F: MatchedPassive_vs_Full | `-0.004092` | [-0.006723, -0.001385] | 0.998 | True |
| Traffic | A: Probe_vs_Common | `+0.002617` | [+0.001260, +0.004131] | 0.000 | True |
| Traffic | B: Probe_vs_Full | `+0.003422` | [+0.002134, +0.004787] | 0.000 | True |
| Traffic | C: Probe_vs_MatchedPassive (MOST IMPORTANT) | `+0.000783` | [+0.000635, +0.000943] | 0.000 | True |
| Traffic | D: Probe_vs_Shuffled | `+0.001188` | [+0.000903, +0.001496] | 0.000 | True |
| Traffic | E: MatchedPassive_vs_Common | `+0.001834` | [+0.000521, +0.003297] | 0.004 | True |
| Traffic | F: MatchedPassive_vs_Full | `+0.002639` | [+0.001369, +0.003976] | 0.000 | True |

## Full dependence-aware statistics (all block lengths + phase)

| Dataset | Comparison | Test | Mean Delta | 95% CI | P(Delta<0) | Excludes zero |
|---|---|---|---:|---|---:|---|
| BeijingAirQuality | Probe_vs_Common | iid_paired_bootstrap | `-0.000116` | [-0.000338, +0.000102] |  | False |
| BeijingAirQuality | Probe_vs_Common | block_bootstrap_len12 | `-0.000116` | [-0.000604, +0.000371] | 0.6823999881744385 | False |
| BeijingAirQuality | Probe_vs_Common | block_bootstrap_len24 | `-0.000116` | [-0.000651, +0.000396] | 0.6830000281333923 | False |
| BeijingAirQuality | Probe_vs_Common | block_bootstrap_len48 | `-0.000116` | [-0.000679, +0.000388] | 0.7001000046730042 | False |
| BeijingAirQuality | Probe_vs_Common | every_12th_window_phase_bootstrap | `-0.000116` | [-0.000570, +0.000345] | 0.6862999796867371 | False |
| BeijingAirQuality | Probe_vs_Full | iid_paired_bootstrap | `-0.000131` | [-0.000388, +0.000128] |  | False |
| BeijingAirQuality | Probe_vs_Full | block_bootstrap_len12 | `-0.000131` | [-0.000731, +0.000444] | 0.6687999963760376 | False |
| BeijingAirQuality | Probe_vs_Full | block_bootstrap_len24 | `-0.000131` | [-0.000766, +0.000507] | 0.6656000018119812 | False |
| BeijingAirQuality | Probe_vs_Full | block_bootstrap_len48 | `-0.000131` | [-0.000790, +0.000547] | 0.6442999839782715 | False |
| BeijingAirQuality | Probe_vs_Full | every_12th_window_phase_bootstrap | `-0.000131` | [-0.000787, +0.000515] | 0.6499999761581421 | False |
| BeijingAirQuality | Probe_vs_MatchedPassive | iid_paired_bootstrap | `+0.000079` | [+0.000019, +0.000136] |  | True |
| BeijingAirQuality | Probe_vs_MatchedPassive | block_bootstrap_len12 | `+0.000079` | [-0.000061, +0.000204] | 0.14020000398159027 | False |
| BeijingAirQuality | Probe_vs_MatchedPassive | block_bootstrap_len24 | `+0.000079` | [-0.000069, +0.000213] | 0.15919999778270721 | False |
| BeijingAirQuality | Probe_vs_MatchedPassive | block_bootstrap_len48 | `+0.000079` | [-0.000071, +0.000213] | 0.16449999809265137 | False |
| BeijingAirQuality | Probe_vs_MatchedPassive | every_12th_window_phase_bootstrap | `+0.000079` | [+0.000030, +0.000133] | 0.0003000000142492354 | True |
| BeijingAirQuality | Probe_vs_Shuffled | iid_paired_bootstrap | `+0.000076` | [-0.000034, +0.000188] |  | False |
| BeijingAirQuality | Probe_vs_Shuffled | block_bootstrap_len12 | `+0.000076` | [-0.000083, +0.000235] | 0.18490000069141388 | False |
| BeijingAirQuality | Probe_vs_Shuffled | block_bootstrap_len24 | `+0.000076` | [-0.000080, +0.000234] | 0.1817999929189682 | False |
| BeijingAirQuality | Probe_vs_Shuffled | block_bootstrap_len48 | `+0.000076` | [-0.000084, +0.000233] | 0.17499999701976776 | False |
| BeijingAirQuality | Probe_vs_Shuffled | every_12th_window_phase_bootstrap | `+0.000076` | [-0.000005, +0.000159] | 0.035100001841783524 | False |
| BeijingAirQuality | MatchedPassive_vs_Common | iid_paired_bootstrap | `-0.000195` | [-0.000423, +0.000033] |  | False |
| BeijingAirQuality | MatchedPassive_vs_Common | block_bootstrap_len12 | `-0.000195` | [-0.000690, +0.000308] | 0.7688000202178955 | False |
| BeijingAirQuality | MatchedPassive_vs_Common | block_bootstrap_len24 | `-0.000195` | [-0.000736, +0.000338] | 0.7631999850273132 | False |
| BeijingAirQuality | MatchedPassive_vs_Common | block_bootstrap_len48 | `-0.000195` | [-0.000764, +0.000323] | 0.7810999751091003 | False |
| BeijingAirQuality | MatchedPassive_vs_Common | every_12th_window_phase_bootstrap | `-0.000195` | [-0.000618, +0.000228] | 0.8137999773025513 | False |
| BeijingAirQuality | MatchedPassive_vs_Full | iid_paired_bootstrap | `-0.000210` | [-0.000456, +0.000040] |  | False |
| BeijingAirQuality | MatchedPassive_vs_Full | block_bootstrap_len12 | `-0.000210` | [-0.000768, +0.000350] | 0.7659000158309937 | False |
| BeijingAirQuality | MatchedPassive_vs_Full | block_bootstrap_len24 | `-0.000210` | [-0.000803, +0.000393] | 0.7540000081062317 | False |
| BeijingAirQuality | MatchedPassive_vs_Full | block_bootstrap_len48 | `-0.000210` | [-0.000810, +0.000416] | 0.7339000105857849 | False |
| BeijingAirQuality | MatchedPassive_vs_Full | every_12th_window_phase_bootstrap | `-0.000210` | [-0.000829, +0.000394] | 0.7439000010490417 | False |
| ETTm2 | Probe_vs_Common | iid_paired_bootstrap | `-0.000170` | [-0.000243, -0.000096] |  | True |
| ETTm2 | Probe_vs_Common | block_bootstrap_len12 | `-0.000170` | [-0.000325, -0.000011] | 0.982699990272522 | True |
| ETTm2 | Probe_vs_Common | block_bootstrap_len24 | `-0.000170` | [-0.000343, -0.000001] | 0.9751999974250793 | True |
| ETTm2 | Probe_vs_Common | block_bootstrap_len48 | `-0.000170` | [-0.000346, +0.000006] | 0.9713000059127808 | False |
| ETTm2 | Probe_vs_Common | every_12th_window_phase_bootstrap | `-0.000170` | [-0.000225, -0.000120] | 1.0 | True |
| ETTm2 | Probe_vs_Full | iid_paired_bootstrap | `+0.000222` | [+0.000153, +0.000289] |  | True |
| ETTm2 | Probe_vs_Full | block_bootstrap_len12 | `+0.000222` | [+0.000087, +0.000361] | 0.000699999975040555 | True |
| ETTm2 | Probe_vs_Full | block_bootstrap_len24 | `+0.000222` | [+0.000078, +0.000368] | 0.0008999999845400453 | True |
| ETTm2 | Probe_vs_Full | block_bootstrap_len48 | `+0.000222` | [+0.000069, +0.000380] | 0.002400000113993883 | True |
| ETTm2 | Probe_vs_Full | every_12th_window_phase_bootstrap | `+0.000222` | [+0.000183, +0.000258] | 0.0 | True |
| ETTm2 | Probe_vs_MatchedPassive | iid_paired_bootstrap | `+0.000080` | [+0.000049, +0.000111] |  | True |
| ETTm2 | Probe_vs_MatchedPassive | block_bootstrap_len12 | `+0.000080` | [+0.000028, +0.000134] | 0.0019000000320374966 | True |
| ETTm2 | Probe_vs_MatchedPassive | block_bootstrap_len24 | `+0.000080` | [+0.000024, +0.000137] | 0.0034000000450760126 | True |
| ETTm2 | Probe_vs_MatchedPassive | block_bootstrap_len48 | `+0.000080` | [+0.000023, +0.000140] | 0.0035000001080334187 | True |
| ETTm2 | Probe_vs_MatchedPassive | every_12th_window_phase_bootstrap | `+0.000080` | [+0.000052, +0.000111] | 0.0 | True |
| ETTm2 | Probe_vs_Shuffled | iid_paired_bootstrap | `+0.000035` | [+0.000004, +0.000067] |  | True |
| ETTm2 | Probe_vs_Shuffled | block_bootstrap_len12 | `+0.000035` | [-0.000018, +0.000092] | 0.0966000035405159 | False |
| ETTm2 | Probe_vs_Shuffled | block_bootstrap_len24 | `+0.000035` | [-0.000022, +0.000095] | 0.11180000007152557 | False |
| ETTm2 | Probe_vs_Shuffled | block_bootstrap_len48 | `+0.000035` | [-0.000023, +0.000099] | 0.10859999805688858 | False |
| ETTm2 | Probe_vs_Shuffled | every_12th_window_phase_bootstrap | `+0.000035` | [+0.000012, +0.000059] | 0.0010999999940395355 | True |
| ETTm2 | MatchedPassive_vs_Common | iid_paired_bootstrap | `-0.000249` | [-0.000330, -0.000166] |  | True |
| ETTm2 | MatchedPassive_vs_Common | block_bootstrap_len12 | `-0.000249` | [-0.000426, -0.000072] | 0.9965999722480774 | True |
| ETTm2 | MatchedPassive_vs_Common | block_bootstrap_len24 | `-0.000249` | [-0.000446, -0.000057] | 0.9943000078201294 | True |
| ETTm2 | MatchedPassive_vs_Common | block_bootstrap_len48 | `-0.000249` | [-0.000449, -0.000057] | 0.9922999739646912 | True |
| ETTm2 | MatchedPassive_vs_Common | every_12th_window_phase_bootstrap | `-0.000249` | [-0.000308, -0.000190] | 1.0 | True |
| ETTm2 | MatchedPassive_vs_Full | iid_paired_bootstrap | `+0.000143` | [+0.000074, +0.000208] |  | True |
| ETTm2 | MatchedPassive_vs_Full | block_bootstrap_len12 | `+0.000143` | [+0.000003, +0.000283] | 0.023099999874830246 | True |
| ETTm2 | MatchedPassive_vs_Full | block_bootstrap_len24 | `+0.000143` | [-0.000003, +0.000289] | 0.028200000524520874 | False |
| ETTm2 | MatchedPassive_vs_Full | block_bootstrap_len48 | `+0.000143` | [-0.000008, +0.000296] | 0.03189999982714653 | False |
| ETTm2 | MatchedPassive_vs_Full | every_12th_window_phase_bootstrap | `+0.000143` | [+0.000109, +0.000175] | 0.0 | True |
| ExchangeRate | Probe_vs_Common | iid_paired_bootstrap | `-0.002441` | [-0.003513, -0.001415] |  | True |
| ExchangeRate | Probe_vs_Common | block_bootstrap_len12 | `-0.002441` | [-0.004922, -0.000110] | 0.9807999730110168 | True |
| ExchangeRate | Probe_vs_Common | block_bootstrap_len24 | `-0.002441` | [-0.004872, -0.000123] | 0.9807999730110168 | True |
| ExchangeRate | Probe_vs_Common | block_bootstrap_len48 | `-0.002441` | [-0.004685, -0.000461] | 0.9902999997138977 | True |
| ExchangeRate | Probe_vs_Common | every_12th_window_phase_bootstrap | `-0.002441` | [-0.002879, -0.001967] | 1.0 | True |
| ExchangeRate | Probe_vs_Full | iid_paired_bootstrap | `-0.004154` | [-0.005268, -0.003071] |  | True |
| ExchangeRate | Probe_vs_Full | block_bootstrap_len12 | `-0.004154` | [-0.006694, -0.001669] | 0.9990000128746033 | True |
| ExchangeRate | Probe_vs_Full | block_bootstrap_len24 | `-0.004154` | [-0.006691, -0.001441] | 0.9983000159263611 | True |
| ExchangeRate | Probe_vs_Full | block_bootstrap_len48 | `-0.004154` | [-0.006507, -0.001765] | 0.9994999766349792 | True |
| ExchangeRate | Probe_vs_Full | every_12th_window_phase_bootstrap | `-0.004155` | [-0.004638, -0.003631] | 1.0 | True |
| ExchangeRate | Probe_vs_MatchedPassive | iid_paired_bootstrap | `-0.000063` | [-0.000147, +0.000027] |  | False |
| ExchangeRate | Probe_vs_MatchedPassive | block_bootstrap_len12 | `-0.000063` | [-0.000239, +0.000141] | 0.7134000062942505 | False |
| ExchangeRate | Probe_vs_MatchedPassive | block_bootstrap_len24 | `-0.000063` | [-0.000241, +0.000164] | 0.6654999852180481 | False |
| ExchangeRate | Probe_vs_MatchedPassive | block_bootstrap_len48 | `-0.000063` | [-0.000244, +0.000192] | 0.6092000007629395 | False |
| ExchangeRate | Probe_vs_MatchedPassive | every_12th_window_phase_bootstrap | `-0.000062` | [-0.000133, +0.000011] | 0.9524999856948853 | False |
| ExchangeRate | Probe_vs_Shuffled | iid_paired_bootstrap | `+0.000255` | [+0.000118, +0.000406] |  | True |
| ExchangeRate | Probe_vs_Shuffled | block_bootstrap_len12 | `+0.000255` | [-0.000008, +0.000595] | 0.029999999329447746 | False |
| ExchangeRate | Probe_vs_Shuffled | block_bootstrap_len24 | `+0.000255` | [-0.000012, +0.000598] | 0.03150000050663948 | False |
| ExchangeRate | Probe_vs_Shuffled | block_bootstrap_len48 | `+0.000255` | [-0.000003, +0.000610] | 0.026000000536441803 | False |
| ExchangeRate | Probe_vs_Shuffled | every_12th_window_phase_bootstrap | `+0.000256` | [+0.000149, +0.000361] | 0.0 | True |
| ExchangeRate | MatchedPassive_vs_Common | iid_paired_bootstrap | `-0.002378` | [-0.003457, -0.001365] |  | True |
| ExchangeRate | MatchedPassive_vs_Common | block_bootstrap_len12 | `-0.002378` | [-0.004877, -0.000078] | 0.9783999919891357 | True |
| ExchangeRate | MatchedPassive_vs_Common | block_bootstrap_len24 | `-0.002378` | [-0.004857, -0.000067] | 0.9779000282287598 | True |
| ExchangeRate | MatchedPassive_vs_Common | block_bootstrap_len48 | `-0.002378` | [-0.004678, -0.000386] | 0.9890999794006348 | True |
| ExchangeRate | MatchedPassive_vs_Common | every_12th_window_phase_bootstrap | `-0.002379` | [-0.002819, -0.001899] | 1.0 | True |
| ExchangeRate | MatchedPassive_vs_Full | iid_paired_bootstrap | `-0.004092` | [-0.005213, -0.003015] |  | True |
| ExchangeRate | MatchedPassive_vs_Full | block_bootstrap_len12 | `-0.004092` | [-0.006665, -0.001603] | 0.9988999962806702 | True |
| ExchangeRate | MatchedPassive_vs_Full | block_bootstrap_len24 | `-0.004092` | [-0.006723, -0.001385] | 0.9983999729156494 | True |
| ExchangeRate | MatchedPassive_vs_Full | block_bootstrap_len48 | `-0.004092` | [-0.006539, -0.001700] | 0.9994000196456909 | True |
| ExchangeRate | MatchedPassive_vs_Full | every_12th_window_phase_bootstrap | `-0.004093` | [-0.004592, -0.003541] | 1.0 | True |
| Traffic | Probe_vs_Common | iid_paired_bootstrap | `+0.002617` | [+0.002209, +0.003028] |  | True |
| Traffic | Probe_vs_Common | block_bootstrap_len12 | `+0.002617` | [+0.001471, +0.003825] | 0.0 | True |
| Traffic | Probe_vs_Common | block_bootstrap_len24 | `+0.002617` | [+0.001260, +0.004131] | 0.0 | True |
| Traffic | Probe_vs_Common | block_bootstrap_len48 | `+0.002617` | [+0.001008, +0.004496] | 0.0006000000284984708 | True |
| Traffic | Probe_vs_Common | every_12th_window_phase_bootstrap | `+0.002615` | [+0.002052, +0.003204] | 0.0 | True |
| Traffic | Probe_vs_Full | iid_paired_bootstrap | `+0.003422` | [+0.003041, +0.003805] |  | True |
| Traffic | Probe_vs_Full | block_bootstrap_len12 | `+0.003422` | [+0.002312, +0.004523] | 0.0 | True |
| Traffic | Probe_vs_Full | block_bootstrap_len24 | `+0.003422` | [+0.002134, +0.004787] | 0.0 | True |
| Traffic | Probe_vs_Full | block_bootstrap_len48 | `+0.003422` | [+0.001881, +0.005148] | 0.0 | True |
| Traffic | Probe_vs_Full | every_12th_window_phase_bootstrap | `+0.003421` | [+0.003080, +0.003787] | 0.0 | True |
| Traffic | Probe_vs_MatchedPassive | iid_paired_bootstrap | `+0.000783` | [+0.000718, +0.000848] |  | True |
| Traffic | Probe_vs_MatchedPassive | block_bootstrap_len12 | `+0.000783` | [+0.000644, +0.000927] | 0.0 | True |
| Traffic | Probe_vs_MatchedPassive | block_bootstrap_len24 | `+0.000783` | [+0.000635, +0.000943] | 0.0 | True |
| Traffic | Probe_vs_MatchedPassive | block_bootstrap_len48 | `+0.000783` | [+0.000602, +0.000975] | 0.0 | True |
| Traffic | Probe_vs_MatchedPassive | every_12th_window_phase_bootstrap | `+0.000783` | [+0.000636, +0.000928] | 0.0 | True |
| Traffic | Probe_vs_Shuffled | iid_paired_bootstrap | `+0.001188` | [+0.001048, +0.001328] |  | True |
| Traffic | Probe_vs_Shuffled | block_bootstrap_len12 | `+0.001188` | [+0.000921, +0.001459] | 0.0 | True |
| Traffic | Probe_vs_Shuffled | block_bootstrap_len24 | `+0.001188` | [+0.000903, +0.001496] | 0.0 | True |
| Traffic | Probe_vs_Shuffled | block_bootstrap_len48 | `+0.001188` | [+0.000877, +0.001534] | 0.0 | True |
| Traffic | Probe_vs_Shuffled | every_12th_window_phase_bootstrap | `+0.001187` | [+0.001014, +0.001369] | 0.0 | True |
| Traffic | MatchedPassive_vs_Common | iid_paired_bootstrap | `+0.001834` | [+0.001441, +0.002239] |  | True |
| Traffic | MatchedPassive_vs_Common | block_bootstrap_len12 | `+0.001834` | [+0.000709, +0.003021] | 0.0006000000284984708 | True |
| Traffic | MatchedPassive_vs_Common | block_bootstrap_len24 | `+0.001834` | [+0.000521, +0.003297] | 0.003700000001117587 | True |
| Traffic | MatchedPassive_vs_Common | block_bootstrap_len48 | `+0.001834` | [+0.000281, +0.003660] | 0.009600000455975533 | True |
| Traffic | MatchedPassive_vs_Common | every_12th_window_phase_bootstrap | `+0.001832` | [+0.001321, +0.002321] | 0.0 | True |
| Traffic | MatchedPassive_vs_Full | iid_paired_bootstrap | `+0.002639` | [+0.002265, +0.003017] |  | True |
| Traffic | MatchedPassive_vs_Full | block_bootstrap_len12 | `+0.002639` | [+0.001545, +0.003720] | 0.0 | True |
| Traffic | MatchedPassive_vs_Full | block_bootstrap_len24 | `+0.002639` | [+0.001369, +0.003976] | 0.0 | True |
| Traffic | MatchedPassive_vs_Full | block_bootstrap_len48 | `+0.002639` | [+0.001114, +0.004325] | 0.0003000000142492354 | True |
| Traffic | MatchedPassive_vs_Full | every_12th_window_phase_bootstrap | `+0.002638` | [+0.002324, +0.002945] | 0.0 | True |

## Weight analysis

| Dataset | Method | Mean entropy | Mean max weight | Mean eff. #experts | Fraction top-expert changed vs Full |
|---|---|---:|---:|---:|---:|
| BeijingAirQuality | TimeFuse_Full | 1.0379 | 0.4562 | 2.734 | 0.000 |
| BeijingAirQuality | TimeFuse_Common | 1.0525 | 0.4294 | 2.812 | 0.359 |
| BeijingAirQuality | TimeFuse_MatchedPassive21 | 1.0346 | 0.4430 | 2.749 | 0.757 |
| BeijingAirQuality | TimeFuse_LearnedProbe | 1.0320 | 0.4428 | 2.741 | 0.753 |
| BeijingAirQuality | TimeFuse_ShuffledProbe | 1.0364 | 0.4397 | 2.755 | 0.735 |
| ETTm2 | TimeFuse_Full | 1.0294 | 0.4713 | 2.680 | 0.000 |
| ETTm2 | TimeFuse_Common | 1.0437 | 0.4575 | 2.730 | 0.254 |
| ETTm2 | TimeFuse_MatchedPassive21 | 1.0240 | 0.4833 | 2.639 | 0.193 |
| ETTm2 | TimeFuse_LearnedProbe | 1.0259 | 0.4820 | 2.646 | 0.202 |
| ETTm2 | TimeFuse_ShuffledProbe | 1.0246 | 0.4835 | 2.640 | 0.201 |
| ExchangeRate | TimeFuse_Full | 0.7592 | 0.6468 | 2.062 | 0.000 |
| ExchangeRate | TimeFuse_Common | 0.8963 | 0.5663 | 2.338 | 0.322 |
| ExchangeRate | TimeFuse_MatchedPassive21 | 0.8773 | 0.5735 | 2.349 | 0.729 |
| ExchangeRate | TimeFuse_LearnedProbe | 0.8576 | 0.5906 | 2.280 | 0.734 |
| ExchangeRate | TimeFuse_ShuffledProbe | 0.8473 | 0.5971 | 2.272 | 0.764 |
| Traffic | TimeFuse_Full | 1.0107 | 0.4951 | 2.616 | 0.000 |
| Traffic | TimeFuse_Common | 1.0195 | 0.4857 | 2.640 | 0.070 |
| Traffic | TimeFuse_MatchedPassive21 | 1.0152 | 0.4865 | 2.616 | 0.255 |
| Traffic | TimeFuse_LearnedProbe | 1.0275 | 0.4732 | 2.664 | 0.258 |
| Traffic | TimeFuse_ShuffledProbe | 1.0300 | 0.4699 | 2.677 | 0.257 |

## Section 24: mechanism diagnostics (OOF Common, router_train, chronological holdout Ridge)

| Dataset | Probe | Pearson r | Spearman rho | R2 | MAE | MAE (null=mean) |
|---|---|---:|---:|---:|---:|---:|
| BeijingAirQuality | A: passive-only (15) | 0.2996 | 0.2091 | 0.0893 | 0.063280 | 0.066274 |
| BeijingAirQuality | B: active-only (6) | 0.0073 | 0.0589 | -0.0052 | 0.066275 | 0.066274 |
| BeijingAirQuality | C: passive+active (21) | 0.2996 | 0.2091 | 0.0893 | 0.063281 | 0.066274 |
| BeijingAirQuality | D: passive+shuffled-active (21) | 0.2996 | 0.2092 | 0.0893 | 0.063279 | 0.066274 |
| ETTm2 | A: passive-only (15) | 0.4613 | 0.0719 | 0.0788 | 0.020496 | 0.018099 |
| ETTm2 | B: active-only (6) | 0.0832 | 0.0463 | 0.0012 | 0.017970 | 0.018099 |
| ETTm2 | C: passive+active (21) | 0.4613 | 0.0718 | 0.0786 | 0.020502 | 0.018099 |
| ETTm2 | D: passive+shuffled-active (21) | 0.4613 | 0.0718 | 0.0785 | 0.020505 | 0.018099 |
| ExchangeRate | A: passive-only (15) | 0.6878 | 0.5825 | 0.4330 | 0.010877 | 0.015085 |
| ExchangeRate | B: active-only (6) | 0.3582 | 0.2729 | -0.0082 | 0.015039 | 0.015085 |
| ExchangeRate | C: passive+active (21) | 0.6879 | 0.5827 | 0.4331 | 0.010876 | 0.015085 |
| ExchangeRate | D: passive+shuffled-active (21) | 0.6876 | 0.5820 | 0.4329 | 0.010879 | 0.015085 |
| Traffic | A: passive-only (15) | 0.6922 | 0.6608 | 0.4790 | 0.024832 | 0.040661 |
| Traffic | B: active-only (6) | 0.5409 | 0.4210 | 0.1303 | 0.033822 | 0.040661 |
| Traffic | C: passive+active (21) | 0.6791 | 0.6525 | 0.4548 | 0.024845 | 0.040661 |
| Traffic | D: passive+shuffled-active (21) | 0.6822 | 0.6514 | 0.4636 | 0.025032 | 0.040661 |

## Section 25: residual-competence diagnostic (active features -> MatchedPassive's OOF residual)

| Dataset | Pearson r | Spearman rho | R2 | MAE | MAE (null=mean) | Useful (predeclared threshold)? |
|---|---:|---:|---:|---:|---:|---|
| BeijingAirQuality | 0.0154 | -0.0373 | -0.0001 | 0.065501 | 0.065501 | False |
| ETTm2 | 0.0427 | -0.0019 | -0.0143 | 0.027521 | 0.027480 | False |
| ExchangeRate | 0.0891 | -0.0046 | -0.0264 | 0.044212 | 0.044541 | False |
| Traffic | -0.1265 | -0.0919 | -0.0478 | 0.033114 | 0.032445 | False |

## Section 26: passive-vs-active mechanism interpretation, per dataset

- **BeijingAirQuality**: Case **A** -- Active perturbation contributes little measurable competence information (active-only weak; passive+active ~= passive).
- **ETTm2**: Case **A** -- Active perturbation contributes little measurable competence information (active-only weak; passive+active ~= passive).
- **ExchangeRate**: Case **A** -- Active perturbation contributes little measurable competence information (active-only weak; passive+active ~= passive).
- **Traffic**: Case **B** -- Active Probe information exists but is largely redundant with passive competence information (active-only useful; passive+active ~= passive).

## Section 27: stratified hard/ambiguous window analysis (target-free strata, router_val)

| Dataset | Stratifier | Stratum | N windows | LearnedProbe MAE | MatchedPassive21 MAE | Delta (Probe-Passive) |
|---|---|---|---:|---:|---:|---:|
| BeijingAirQuality | expert_forecast_disagreement | bottom_25pct | 1774 | 0.197108 | 0.197016 | `+0.000093` |
| BeijingAirQuality | expert_forecast_disagreement | middle_50pct | 3545 | 0.251526 | 0.251522 | `+0.000005` |
| BeijingAirQuality | expert_forecast_disagreement | top_25pct | 1774 | 0.331871 | 0.331657 | `+0.000214` |
| BeijingAirQuality | matchedpassive_predicted_gap | bottom_25pct | 1774 | 0.253027 | 0.253021 | `+0.000006` |
| BeijingAirQuality | matchedpassive_predicted_gap | middle_50pct | 3545 | 0.252137 | 0.252047 | `+0.000090` |
| BeijingAirQuality | matchedpassive_predicted_gap | top_25pct | 1774 | 0.274731 | 0.274601 | `+0.000130` |
| BeijingAirQuality | matchedpassive_entropy | bottom_25pct | 1774 | 0.277202 | 0.277076 | `+0.000127` |
| BeijingAirQuality | matchedpassive_entropy | middle_50pct | 3539 | 0.253337 | 0.253264 | `+0.000073` |
| BeijingAirQuality | matchedpassive_entropy | top_25pct | 1780 | 0.248176 | 0.248133 | `+0.000043` |
| BeijingAirQuality | probe_response_magnitude | bottom_25pct | 1774 | 0.209605 | 0.209769 | `-0.000164` |
| BeijingAirQuality | probe_response_magnitude | middle_50pct | 3545 | 0.249621 | 0.249473 | `+0.000147` |
| BeijingAirQuality | probe_response_magnitude | top_25pct | 1774 | 0.323182 | 0.322997 | `+0.000185` |
| ETTm2 | expert_forecast_disagreement | bottom_25pct | 2854 | 0.141756 | 0.141751 | `+0.000004` |
| ETTm2 | expert_forecast_disagreement | middle_50pct | 5705 | 0.159336 | 0.159257 | `+0.000079` |
| ETTm2 | expert_forecast_disagreement | top_25pct | 2854 | 0.181570 | 0.181414 | `+0.000156` |
| ETTm2 | matchedpassive_predicted_gap | bottom_25pct | 2854 | 0.159903 | 0.159831 | `+0.000072` |
| ETTm2 | matchedpassive_predicted_gap | middle_50pct | 5705 | 0.160207 | 0.160164 | `+0.000043` |
| ETTm2 | matchedpassive_predicted_gap | top_25pct | 2854 | 0.161681 | 0.161521 | `+0.000160` |
| ETTm2 | matchedpassive_entropy | bottom_25pct | 2854 | 0.161299 | 0.161156 | `+0.000143` |
| ETTm2 | matchedpassive_entropy | middle_50pct | 5704 | 0.158660 | 0.158587 | `+0.000073` |
| ETTm2 | matchedpassive_entropy | top_25pct | 2855 | 0.163375 | 0.163346 | `+0.000029` |
| ETTm2 | probe_response_magnitude | bottom_25pct | 2854 | 0.140545 | 0.140548 | `-0.000004` |
| ETTm2 | probe_response_magnitude | middle_50pct | 5705 | 0.161173 | 0.161096 | `+0.000077` |
| ETTm2 | probe_response_magnitude | top_25pct | 2854 | 0.179108 | 0.178940 | `+0.000169` |
| ExchangeRate | expert_forecast_disagreement | bottom_25pct | 353 | 0.100251 | 0.100408 | `-0.000157` |
| ExchangeRate | expert_forecast_disagreement | middle_50pct | 705 | 0.114567 | 0.114745 | `-0.000178` |
| ExchangeRate | expert_forecast_disagreement | top_25pct | 353 | 0.167667 | 0.167405 | `+0.000262` |
| ExchangeRate | matchedpassive_predicted_gap | bottom_25pct | 353 | 0.127657 | 0.127511 | `+0.000146` |
| ExchangeRate | matchedpassive_predicted_gap | middle_50pct | 705 | 0.117740 | 0.117827 | `-0.000087` |
| ExchangeRate | matchedpassive_predicted_gap | top_25pct | 353 | 0.133924 | 0.134146 | `-0.000223` |
| ExchangeRate | matchedpassive_entropy | bottom_25pct | 353 | 0.155200 | 0.155261 | `-0.000061` |
| ExchangeRate | matchedpassive_entropy | middle_50pct | 705 | 0.114969 | 0.115190 | `-0.000221` |
| ExchangeRate | matchedpassive_entropy | top_25pct | 353 | 0.111915 | 0.111663 | `+0.000252` |
| ExchangeRate | probe_response_magnitude | bottom_25pct | 353 | 0.107933 | 0.108214 | `-0.000280` |
| ExchangeRate | probe_response_magnitude | middle_50pct | 705 | 0.118017 | 0.117933 | `+0.000084` |
| ExchangeRate | probe_response_magnitude | top_25pct | 353 | 0.153094 | 0.153231 | `-0.000137` |
| Traffic | expert_forecast_disagreement | bottom_25pct | 851 | 0.217643 | 0.216607 | `+0.001036` |
| Traffic | expert_forecast_disagreement | middle_50pct | 1700 | 0.280236 | 0.279348 | `+0.000888` |
| Traffic | expert_forecast_disagreement | top_25pct | 851 | 0.315752 | 0.315431 | `+0.000321` |
| Traffic | matchedpassive_predicted_gap | bottom_25pct | 851 | 0.294304 | 0.293581 | `+0.000723` |
| Traffic | matchedpassive_predicted_gap | middle_50pct | 1700 | 0.272913 | 0.272089 | `+0.000824` |
| Traffic | matchedpassive_predicted_gap | top_25pct | 851 | 0.253721 | 0.252959 | `+0.000762` |
| Traffic | matchedpassive_entropy | bottom_25pct | 851 | 0.243608 | 0.241959 | `+0.001649` |
| Traffic | matchedpassive_entropy | middle_50pct | 1700 | 0.273031 | 0.272385 | `+0.000647` |
| Traffic | matchedpassive_entropy | top_25pct | 851 | 0.304179 | 0.303989 | `+0.000190` |
| Traffic | probe_response_magnitude | bottom_25pct | 851 | 0.283676 | 0.282884 | `+0.000792` |
| Traffic | probe_response_magnitude | middle_50pct | 1700 | 0.271065 | 0.270381 | `+0.000684` |
| Traffic | probe_response_magnitude | top_25pct | 851 | 0.268039 | 0.267068 | `+0.000972` |

## Section 28: existing-result reproduction (old in-sample-stacked base TimeFuse)

| Dataset | Old TimeFuse MAE | New TimeFuse-Full MAE | Difference | Relative diff | Within 5% tolerance |
|---|---:|---:|---:|---:|---|
| BeijingAirQuality | 0.258141 | 0.258141 | `+0.000000` | `+0.00%` | True |
| ETTm2 | 0.160278 | 0.160278 | `+0.000000` | `+0.00%` | True |
| ExchangeRate | 0.128424 | 0.128424 | `+0.000000` | `+0.00%` | True |
| Traffic | 0.270041 | 0.270041 | `+0.000000` | `+0.00%` | True |

Base TimeFuse mechanism (meta-features, ModelFusor, hyperparameters) is byte-identical; TimeFuse-Full here trains only on legal_idx_all router_train rows (Section 2/7 tail-purge) rather than ALL router_train rows used by the old (in-sample-stacking) experiment, and independently re-fits with the same seed but a different overall pipeline (different scaler-fit window support). Small differences are therefore expected; large differences would indicate a bug.

## Sections 29-33: integrity

- **BeijingAirQuality**: PASS (checkpoints unchanged: True; no test cache: True; all purge assertions pass: True; observability holds: True; target-corruption invariant: True; expert-order permutation invariant: True; weighted-forecast reproduces: True)
- **ETTm2**: PASS (checkpoints unchanged: True; no test cache: True; all purge assertions pass: True; observability holds: True; target-corruption invariant: True; expert-order permutation invariant: True; weighted-forecast reproduces: True)
- **ExchangeRate**: PASS (checkpoints unchanged: True; no test cache: True; all purge assertions pass: True; observability holds: True; target-corruption invariant: True; expert-order permutation invariant: True; weighted-forecast reproduces: True)
- **Traffic**: PASS (checkpoints unchanged: True; no test cache: True; all purge assertions pass: True; observability holds: True; target-corruption invariant: True; expert-order permutation invariant: True; weighted-forecast reproduces: True)

## Section 36: claim rule / decision

- n_beats_common_point=3/4, n_beats_common_sig=2
- n_ties_or_beats_full=2/4
- n_beats_passive_point=1/4, n_beats_passive_sig=0
- n_beats_shuffled_point=0/4, n_beats_shuffled_sig=0
- n_broad_regressions=1
- n_active_only_useful=1/4 (majority=False)
- n_residual_predicts_beyond_passive=0/4 (majority=False)

## Decision: NO_INCREMENTAL_ACTIVE_SIGNAL

Under the strict purged-OOF protocol, active perturbation does not provide measurable competence information beyond the matched passive estimator: MatchedPassive-21 ties or beats LearnedProbe on a majority of datasets, and active-response features fail to predict either raw competence or MatchedPassive's residual competence beyond the predeclared thresholds.


## Section 39: answers

**1. Was official TimeFuse preserved?** Yes -- meta_feature.extract_meta_feature (22-dim, via the cached wrapper already verified byte-identical on well-behaved inputs) and timefuse.ModelFusor used verbatim from commit 978e6c6b9e4f246632c269aa0f9beeb099eabcfc, with the exact official training hyperparameters, imported unchanged from `../timefuse_probe/run_timefuse_probe.py`.
**2. Does TimeFuse-Full reproduce the prior base TimeFuse result?** See Section 28 table above.
**3. Were all competence scores used to TRAIN augmented TimeFuse honest purged OOF predictions?** Yes -- every MatchedPassive/LearnedProbe score used as a TimeFuse training feature (Common window set) comes from a fold-restricted model trained ONLY on causally-earlier windows (Sections 1/9-11), never from a model that saw that window's own target.
**4. Did every fold satisfy max_train_target_end <= min_heldout_origin?** Yes, on every dataset/fold (see table above).
**5. Does corrected LearnedProbe still improve TimeFuse-Common?** By point estimate on 3/4; block-24 significant on 2/4.
**6. Does it also improve or match TimeFuse-Full?** Ties-or-beats (or non-significant regression) on 2/4.
**7. Does LearnedProbe beat MatchedPassive-21?** By point estimate on 1/4; block-24 significant on 0/4. **This is the most important comparison.**
**8. Does it beat ShuffledProbe?** By point estimate on 0/4; block-24 significant on 0/4.
**9. Do the six active Probe-response features predict expert competence by themselves?** See Section 24 row B per dataset above; useful (predeclared R2>0.01, |Spearman|>0.05) on 1/4 datasets.
**10. Do active features add predictive value beyond the 15 passive features?** Compare Section 24 rows A and C per dataset (R2 improvement) -- see `mechanism_case.adds_beyond_passive` per dataset above.
**11. Do active features predict the residual competence error left by MatchedPassive?** See Section 25 table above; useful on 0/4 datasets.
  - BeijingAirQuality: 0.174 fraction of router_val windows where LearnedProbe's and MatchedPassive's argmin-expert disagree.
  - ETTm2: 0.263 fraction of router_val windows where LearnedProbe's and MatchedPassive's argmin-expert disagree.
  - ExchangeRate: 0.128 fraction of router_val windows where LearnedProbe's and MatchedPassive's argmin-expert disagree.
  - Traffic: 0.162 fraction of router_val windows where LearnedProbe's and MatchedPassive's argmin-expert disagree.
**12. When LearnedProbe and MatchedPassive disagree, which is more often correct?** See the `probe_vs_passive_disagreement_fraction_val` figures directly above and the Section 27 stratified table's `probe_response_magnitude`/`matchedpassive_entropy` strata, which condition router-level MAE on exactly this kind of disagreement/uncertainty.
**13. Does LearnedProbe become more useful on high-disagreement or passive-uncertain windows?** See Section 27 stratified table above -- compare the `delta_probe_minus_matchedpassive` column across bottom_25pct/middle_50pct/top_25pct strata for `expert_forecast_disagreement` and `matchedpassive_entropy`.
**14. Were there significant wins or regressions under block-24 bootstrap?** See Section 23 table above; n_broad_regressions=1 (broad = significant regression vs Common on >= half the datasets).
**15. Based strictly on these results, which description is supported?** **NO_INCREMENTAL_ACTIVE_SIGNAL** -- Under the strict purged-OOF protocol, active perturbation does not provide measurable competence information beyond the matched passive estimator: MatchedPassive-21 ties or beats LearnedProbe on a majority of datasets, and active-response features fail to predict either raw competence or MatchedPassive's residual competence beyond the predeclared thresholds.

## Section 40: final scientific question

"Under a strict causal purged-OOF protocol, does actively perturbing frozen forecasting experts reveal expert-specific competence information that TimeFuse cannot already infer from ordinary passive window, forecast, and disagreement information?"

Under the strict purged-OOF protocol, active perturbation does not provide measurable competence information beyond the matched passive estimator: MatchedPassive-21 ties or beats LearnedProbe on a majority of datasets, and active-response features fail to predict either raw competence or MatchedPassive's residual competence beyond the predeclared thresholds.

## Hard rule compliance

```text
TEST SET ACCESSED: NO
FORECASTING EXPERTS RETRAINED: NO
LEARNEDPROBE ARCHITECTURE/LOSS/TRAINING MODIFIED: NO
SELECTIVE GATE USED: NO (per Section 5/12)
OTHER PUBLISHED ROUTERS IMPLEMENTED: NO (TimeFuse only)
COSTAR / ONLINE COSTAR TOUCHED: NO
PURGE ASSERTION: see table above; raises AssertionError immediately if violated
```
