# FFORMA vs FFORMA + LearnedProbe (Final Audit Overrides: purged causal OOF)

Official FFORMA (robjhyndman/M4metalearning, commit 61ddc7101680e9df7219c359587d0b509d2b50d6): THA_features (Python tsfeatures v0.4.5, verified same function set) + custom softmax-expected-loss XGBoost objective (error_softmax_obj, ported to Python xgboost's modern [N,K] custom-objective API), applied to this project's frozen-expert / router_train / router_val protocol. Every target-dependent supervised component (LearnedProbe, MatchedPassive-21, FFORMA's own hyperparameter selection) is trained on PURGED chronological folds (2 folds, min_train_fraction=0.4): a training window may supervise a held-out fold only if its target fully resolves before the fold's first held-out forecast origin.

## Mandatory causal assertions (Section 1, 2, 14)

| Dataset | Fold | Train target-end max | Eval origin min | Assertion holds | Purged windows |
|---|---:|---:|---:|---|---:|
| ExchangeRate | 0 | 2645 | 2645 | True | 11 |
| ExchangeRate | 1 | 3598 | 3598 | True | 11 |
| Traffic | 0 | 6230 | 6230 | True | 11 |
| Traffic | 1 | 8378 | 8378 | True | 11 |
| BeijingAirQuality | 0 | 12874 | 12874 | True | 11 |
| BeijingAirQuality | 1 | 17237 | 17237 | True | 11 |
| ETTm2 | 0 | 20650 | 20650 | True | 11 |
| ETTm2 | 1 | 27605 | 27605 | True | 11 |

| Dataset | router_train->router_val observability holds | max train target-end | min val origin |
|---|---|---:|---:|
| ExchangeRate | True | 4456 | 4552 |
| Traffic | True | 10430 | 10526 |
| BeijingAirQuality | True | 21504 | 21600 |
| ETTm2 | True | 34464 | 34560 |

## Predeclared dataset frequency / tsfeatures diagnostics

| Dataset | Freq | Split | Windows | Group failures | Seasonal-padding zeros | NaN values zeroed |
|---|---:|---|---:|---:|---:|---:|
| ExchangeRate | 1 | router_train | 2821 | 0 | 14105 | 36682 |
| ExchangeRate | 1 | router_val | 1411 | 0 | 7055 | 18394 |
| Traffic | 24 | router_train | 6804 | 0 | 0 | 27216 |
| Traffic | 24 | router_val | 3402 | 0 | 0 | 13608 |
| BeijingAirQuality | 24 | router_train | 14186 | 0 | 0 | 56769 |
| BeijingAirQuality | 24 | router_val | 7093 | 0 | 0 | 28383 |
| ETTm2 | 96 | router_train | 22826 | 0 | 0 | 229169 |
| ETTm2 | 96 | router_val | 11413 | 0 | 0 | 114367 |

## FFORMA hyperparameter selection (base FFORMA only, purged OOF)

| Dataset | max_depth | eta | subsample | colsample_bytree | nrounds | Purged OOF MAE | Selected |
|---|---:|---:|---:|---:|---:|---:|---|
| ExchangeRate | 14 | 0.575188 | 0.9161483 | 0.7670739 | 94 | 0.088728 | <-- selected |
| ExchangeRate | 6 | 0.1 | 0.8 | 0.8 | 100 | 0.088796 |  |
| ExchangeRate | 10 | 0.05 | 0.7 | 0.7 | 150 | 0.088967 |  |
| ExchangeRate | 8 | 0.3 | 0.9 | 0.6 | 50 | 0.088739 |  |
| ExchangeRate | 14 | 0.01 | 0.6 | 0.9 | 250 | 0.091364 |  |
| Traffic | 14 | 0.575188 | 0.9161483 | 0.7670739 | 94 | 0.289650 |  |
| Traffic | 6 | 0.1 | 0.8 | 0.8 | 100 | 0.282312 |  |
| Traffic | 10 | 0.05 | 0.7 | 0.7 | 150 | 0.280329 |  |
| Traffic | 8 | 0.3 | 0.9 | 0.6 | 50 | 0.284703 |  |
| Traffic | 14 | 0.01 | 0.6 | 0.9 | 250 | 0.277127 | <-- selected |
| BeijingAirQuality | 14 | 0.575188 | 0.9161483 | 0.7670739 | 94 | 0.333654 |  |
| BeijingAirQuality | 6 | 0.1 | 0.8 | 0.8 | 100 | 0.331549 |  |
| BeijingAirQuality | 10 | 0.05 | 0.7 | 0.7 | 150 | 0.331060 |  |
| BeijingAirQuality | 8 | 0.3 | 0.9 | 0.6 | 50 | 0.331836 |  |
| BeijingAirQuality | 14 | 0.01 | 0.6 | 0.9 | 250 | 0.330965 | <-- selected |
| ETTm2 | 14 | 0.575188 | 0.9161483 | 0.7670739 | 94 | 0.179969 |  |
| ETTm2 | 6 | 0.1 | 0.8 | 0.8 | 100 | 0.177727 | <-- selected |
| ETTm2 | 10 | 0.05 | 0.7 | 0.7 | 150 | 0.177927 |  |
| ETTm2 | 8 | 0.3 | 0.9 | 0.6 | 50 | 0.178088 |  |
| ETTm2 | 14 | 0.01 | 0.6 | 0.9 | 250 | 0.178255 |  |

## Primary results (router_val MAE / MSE)

| Dataset | M4Fixed | Full | Common | +MatchedPassive21 | +LearnedProbe | +ShuffledProbe |
|---|---:|---:|---:|---:|---:|---:|
| ExchangeRate | 0.120615 | 0.120615 | 0.120601 | 0.119945 | 0.120381 | 0.120450 |
| Traffic | 0.281366 | 0.265846 | 0.265591 | 0.265333 | 0.265448 | 0.265555 |
| BeijingAirQuality | 0.260575 | 0.257333 | 0.257469 | 0.257334 | 0.257392 | 0.257472 |
| ETTm2 | 0.161337 | 0.160792 | 0.161144 | 0.160855 | 0.160908 | 0.161036 |

## LearnedProbe deltas

| Dataset | vs Common | vs Full | vs MatchedPassive21 | Probe vs Shuffled |
|---|---:|---:|---:|---:|
| ExchangeRate | `-0.000220` | `-0.000234` | `+0.000436` | `-0.000069` |
| Traffic | `-0.000143` | `-0.000398` | `+0.000116` | `-0.000106` |
| BeijingAirQuality | `-0.000077` | `+0.000059` | `+0.000058` | `-0.000080` |
| ETTm2 | `-0.000236` | `+0.000116` | `+0.000053` | `-0.000128` |

## Primary dependence-aware statistics (block-24)

| Dataset | Comparison | Mean Delta | 95% CI | P(Delta<0) | Excludes zero |
|---|---|---:|---|---:|---|
| ExchangeRate | Probe_vs_Common | `-0.000221` | [-0.000600, +0.000040] | 0.942 | False |
| ExchangeRate | Probe_vs_Full | `-0.000234` | [-0.000730, +0.000158] | 0.854 | False |
| ExchangeRate | Probe_vs_MatchedPassive | `+0.000436` | [+0.000191, +0.000754] | 0.000 | True |
| ExchangeRate | Probe_vs_Shuffled | `-0.000069` | [-0.000289, +0.000124] | 0.741 | False |
| Traffic | Probe_vs_Common | `-0.000143` | [-0.000321, +0.000041] | 0.937 | False |
| Traffic | Probe_vs_Full | `-0.000398` | [-0.000862, +0.000049] | 0.959 | False |
| Traffic | Probe_vs_MatchedPassive | `+0.000115` | [-0.000091, +0.000318] | 0.152 | False |
| Traffic | Probe_vs_Shuffled | `-0.000106` | [-0.000291, +0.000084] | 0.854 | False |
| BeijingAirQuality | Probe_vs_Common | `-0.000077` | [-0.000242, +0.000045] | 0.896 | False |
| BeijingAirQuality | Probe_vs_Full | `+0.000059` | [-0.000147, +0.000274] | 0.299 | False |
| BeijingAirQuality | Probe_vs_MatchedPassive | `+0.000058` | [-0.000227, +0.000341] | 0.368 | False |
| BeijingAirQuality | Probe_vs_Shuffled | `-0.000080` | [-0.000201, +0.000018] | 0.949 | False |
| ETTm2 | Probe_vs_Common | `-0.000236` | [-0.000320, -0.000156] | 1.000 | True |
| ETTm2 | Probe_vs_Full | `+0.000116` | [-0.000032, +0.000264] | 0.063 | False |
| ETTm2 | Probe_vs_MatchedPassive | `+0.000053` | [-0.000021, +0.000127] | 0.075 | False |
| ETTm2 | Probe_vs_Shuffled | `-0.000128` | [-0.000179, -0.000078] | 1.000 | True |

## Full dependence-aware statistics (all block lengths + phase)

| Dataset | Comparison | Test | Mean Delta | 95% CI | P(Delta<0) | Excludes zero |
|---|---|---|---:|---|---:|---|
| ExchangeRate | Probe_vs_Common | iid_paired_bootstrap | `-0.000221` | [-0.000345, -0.000098] |  | True |
| ExchangeRate | Probe_vs_Common | block_bootstrap_len12 | `-0.000221` | [-0.000532, +0.000039] | 0.9495000243186951 | False |
| ExchangeRate | Probe_vs_Common | block_bootstrap_len24 | `-0.000221` | [-0.000600, +0.000040] | 0.9419999718666077 | False |
| ExchangeRate | Probe_vs_Common | block_bootstrap_len48 | `-0.000221` | [-0.000575, +0.000018] | 0.9574000239372253 | False |
| ExchangeRate | Probe_vs_Common | every_12th_window_phase_bootstrap | `-0.000220` | [-0.000318, -0.000129] | 1.0 | True |
| ExchangeRate | Probe_vs_Full | iid_paired_bootstrap | `-0.000234` | [-0.000411, -0.000072] |  | True |
| ExchangeRate | Probe_vs_Full | block_bootstrap_len12 | `-0.000234` | [-0.000644, +0.000137] | 0.8830999732017517 | False |
| ExchangeRate | Probe_vs_Full | block_bootstrap_len24 | `-0.000234` | [-0.000730, +0.000158] | 0.8539000153541565 | False |
| ExchangeRate | Probe_vs_Full | block_bootstrap_len48 | `-0.000234` | [-0.000725, +0.000174] | 0.8652999997138977 | False |
| ExchangeRate | Probe_vs_Full | every_12th_window_phase_bootstrap | `-0.000234` | [-0.000337, -0.000138] | 1.0 | True |
| ExchangeRate | Probe_vs_MatchedPassive | iid_paired_bootstrap | `+0.000436` | [+0.000292, +0.000590] |  | True |
| ExchangeRate | Probe_vs_MatchedPassive | block_bootstrap_len12 | `+0.000436` | [+0.000213, +0.000722] | 0.0 | True |
| ExchangeRate | Probe_vs_MatchedPassive | block_bootstrap_len24 | `+0.000436` | [+0.000191, +0.000754] | 0.0 | True |
| ExchangeRate | Probe_vs_MatchedPassive | block_bootstrap_len48 | `+0.000436` | [+0.000170, +0.000808] | 0.0 | True |
| ExchangeRate | Probe_vs_MatchedPassive | every_12th_window_phase_bootstrap | `+0.000435` | [+0.000329, +0.000539] | 0.0 | True |
| ExchangeRate | Probe_vs_Shuffled | iid_paired_bootstrap | `-0.000069` | [-0.000197, +0.000057] |  | False |
| ExchangeRate | Probe_vs_Shuffled | block_bootstrap_len12 | `-0.000069` | [-0.000283, +0.000133] | 0.7475000023841858 | False |
| ExchangeRate | Probe_vs_Shuffled | block_bootstrap_len24 | `-0.000069` | [-0.000289, +0.000124] | 0.7409999966621399 | False |
| ExchangeRate | Probe_vs_Shuffled | block_bootstrap_len48 | `-0.000069` | [-0.000304, +0.000124] | 0.7294999957084656 | False |
| ExchangeRate | Probe_vs_Shuffled | every_12th_window_phase_bootstrap | `-0.000068` | [-0.000246, +0.000089] | 0.7854999899864197 | False |
| Traffic | Probe_vs_Common | iid_paired_bootstrap | `-0.000143` | [-0.000226, -0.000055] |  | True |
| Traffic | Probe_vs_Common | block_bootstrap_len12 | `-0.000143` | [-0.000317, +0.000022] | 0.953499972820282 | False |
| Traffic | Probe_vs_Common | block_bootstrap_len24 | `-0.000143` | [-0.000321, +0.000041] | 0.9366999864578247 | False |
| Traffic | Probe_vs_Common | block_bootstrap_len48 | `-0.000143` | [-0.000316, +0.000039] | 0.9350000023841858 | False |
| Traffic | Probe_vs_Common | every_12th_window_phase_bootstrap | `-0.000143` | [-0.000269, -0.000002] | 0.9758999943733215 | True |
| Traffic | Probe_vs_Full | iid_paired_bootstrap | `-0.000398` | [-0.000576, -0.000226] |  | True |
| Traffic | Probe_vs_Full | block_bootstrap_len12 | `-0.000398` | [-0.000823, +0.000011] | 0.972100019454956 | False |
| Traffic | Probe_vs_Full | block_bootstrap_len24 | `-0.000398` | [-0.000862, +0.000049] | 0.9592000246047974 | False |
| Traffic | Probe_vs_Full | block_bootstrap_len48 | `-0.000398` | [-0.000844, +0.000024] | 0.9664000272750854 | False |
| Traffic | Probe_vs_Full | every_12th_window_phase_bootstrap | `-0.000398` | [-0.000660, -0.000141] | 0.9987000226974487 | True |
| Traffic | Probe_vs_MatchedPassive | iid_paired_bootstrap | `+0.000115` | [+0.000023, +0.000208] |  | True |
| Traffic | Probe_vs_MatchedPassive | block_bootstrap_len12 | `+0.000115` | [-0.000081, +0.000312] | 0.12929999828338623 | False |
| Traffic | Probe_vs_MatchedPassive | block_bootstrap_len24 | `+0.000115` | [-0.000091, +0.000318] | 0.15189999341964722 | False |
| Traffic | Probe_vs_MatchedPassive | block_bootstrap_len48 | `+0.000115` | [-0.000084, +0.000304] | 0.12280000001192093 | False |
| Traffic | Probe_vs_MatchedPassive | every_12th_window_phase_bootstrap | `+0.000115` | [+0.000020, +0.000212] | 0.007199999876320362 | True |
| Traffic | Probe_vs_Shuffled | iid_paired_bootstrap | `-0.000106` | [-0.000191, -0.000017] |  | True |
| Traffic | Probe_vs_Shuffled | block_bootstrap_len12 | `-0.000106` | [-0.000283, +0.000064] | 0.8862000107765198 | False |
| Traffic | Probe_vs_Shuffled | block_bootstrap_len24 | `-0.000106` | [-0.000291, +0.000084] | 0.8543000221252441 | False |
| Traffic | Probe_vs_Shuffled | block_bootstrap_len48 | `-0.000106` | [-0.000283, +0.000085] | 0.8478000164031982 | False |
| Traffic | Probe_vs_Shuffled | every_12th_window_phase_bootstrap | `-0.000106` | [-0.000236, +0.000040] | 0.9266999959945679 | False |
| BeijingAirQuality | Probe_vs_Common | iid_paired_bootstrap | `-0.000077` | [-0.000151, -0.000004] |  | True |
| BeijingAirQuality | Probe_vs_Common | block_bootstrap_len12 | `-0.000077` | [-0.000241, +0.000049] | 0.8791000247001648 | False |
| BeijingAirQuality | Probe_vs_Common | block_bootstrap_len24 | `-0.000077` | [-0.000242, +0.000045] | 0.8964999914169312 | False |
| BeijingAirQuality | Probe_vs_Common | block_bootstrap_len48 | `-0.000077` | [-0.000243, +0.000044] | 0.8952000141143799 | False |
| BeijingAirQuality | Probe_vs_Common | every_12th_window_phase_bootstrap | `-0.000077` | [-0.000123, -0.000025] | 0.9973999857902527 | True |
| BeijingAirQuality | Probe_vs_Full | iid_paired_bootstrap | `+0.000059` | [-0.000047, +0.000165] |  | False |
| BeijingAirQuality | Probe_vs_Full | block_bootstrap_len12 | `+0.000059` | [-0.000154, +0.000273] | 0.31709998846054077 | False |
| BeijingAirQuality | Probe_vs_Full | block_bootstrap_len24 | `+0.000059` | [-0.000147, +0.000274] | 0.2992999851703644 | False |
| BeijingAirQuality | Probe_vs_Full | block_bootstrap_len48 | `+0.000059` | [-0.000138, +0.000283] | 0.27720001339912415 | False |
| BeijingAirQuality | Probe_vs_Full | every_12th_window_phase_bootstrap | `+0.000059` | [+0.000002, +0.000119] | 0.019899999722838402 | True |
| BeijingAirQuality | Probe_vs_MatchedPassive | iid_paired_bootstrap | `+0.000058` | [-0.000094, +0.000207] |  | False |
| BeijingAirQuality | Probe_vs_MatchedPassive | block_bootstrap_len12 | `+0.000058` | [-0.000283, +0.000366] | 0.38100001215934753 | False |
| BeijingAirQuality | Probe_vs_MatchedPassive | block_bootstrap_len24 | `+0.000058` | [-0.000227, +0.000341] | 0.367900013923645 | False |
| BeijingAirQuality | Probe_vs_MatchedPassive | block_bootstrap_len48 | `+0.000058` | [-0.000178, +0.000286] | 0.364300012588501 | False |
| BeijingAirQuality | Probe_vs_MatchedPassive | every_12th_window_phase_bootstrap | `+0.000058` | [-0.000041, +0.000155] | 0.1234000027179718 | False |
| BeijingAirQuality | Probe_vs_Shuffled | iid_paired_bootstrap | `-0.000080` | [-0.000147, -0.000015] |  | True |
| BeijingAirQuality | Probe_vs_Shuffled | block_bootstrap_len12 | `-0.000080` | [-0.000194, +0.000017] | 0.9491000175476074 | False |
| BeijingAirQuality | Probe_vs_Shuffled | block_bootstrap_len24 | `-0.000080` | [-0.000201, +0.000018] | 0.9487000107765198 | False |
| BeijingAirQuality | Probe_vs_Shuffled | block_bootstrap_len48 | `-0.000080` | [-0.000201, +0.000015] | 0.9538000226020813 | False |
| BeijingAirQuality | Probe_vs_Shuffled | every_12th_window_phase_bootstrap | `-0.000080` | [-0.000154, -0.000006] | 0.9829999804496765 | True |
| ETTm2 | Probe_vs_Common | iid_paired_bootstrap | `-0.000236` | [-0.000278, -0.000193] |  | True |
| ETTm2 | Probe_vs_Common | block_bootstrap_len12 | `-0.000236` | [-0.000312, -0.000161] | 1.0 | True |
| ETTm2 | Probe_vs_Common | block_bootstrap_len24 | `-0.000236` | [-0.000320, -0.000156] | 1.0 | True |
| ETTm2 | Probe_vs_Common | block_bootstrap_len48 | `-0.000236` | [-0.000322, -0.000154] | 1.0 | True |
| ETTm2 | Probe_vs_Common | every_12th_window_phase_bootstrap | `-0.000236` | [-0.000270, -0.000198] | 1.0 | True |
| ETTm2 | Probe_vs_Full | iid_paired_bootstrap | `+0.000116` | [+0.000047, +0.000188] |  | True |
| ETTm2 | Probe_vs_Full | block_bootstrap_len12 | `+0.000116` | [-0.000016, +0.000252] | 0.04149999842047691 | False |
| ETTm2 | Probe_vs_Full | block_bootstrap_len24 | `+0.000116` | [-0.000032, +0.000264] | 0.06319999694824219 | False |
| ETTm2 | Probe_vs_Full | block_bootstrap_len48 | `+0.000116` | [-0.000037, +0.000266] | 0.06530000269412994 | False |
| ETTm2 | Probe_vs_Full | every_12th_window_phase_bootstrap | `+0.000116` | [+0.000062, +0.000171] | 0.0 | True |
| ETTm2 | Probe_vs_MatchedPassive | iid_paired_bootstrap | `+0.000053` | [+0.000015, +0.000091] |  | True |
| ETTm2 | Probe_vs_MatchedPassive | block_bootstrap_len12 | `+0.000053` | [-0.000014, +0.000125] | 0.060499999672174454 | False |
| ETTm2 | Probe_vs_MatchedPassive | block_bootstrap_len24 | `+0.000053` | [-0.000021, +0.000127] | 0.07490000128746033 | False |
| ETTm2 | Probe_vs_MatchedPassive | block_bootstrap_len48 | `+0.000053` | [-0.000020, +0.000130] | 0.07660000026226044 | False |
| ETTm2 | Probe_vs_MatchedPassive | every_12th_window_phase_bootstrap | `+0.000053` | [+0.000030, +0.000077] | 0.0 | True |
| ETTm2 | Probe_vs_Shuffled | iid_paired_bootstrap | `-0.000128` | [-0.000155, -0.000101] |  | True |
| ETTm2 | Probe_vs_Shuffled | block_bootstrap_len12 | `-0.000128` | [-0.000175, -0.000082] | 1.0 | True |
| ETTm2 | Probe_vs_Shuffled | block_bootstrap_len24 | `-0.000128` | [-0.000179, -0.000078] | 1.0 | True |
| ETTm2 | Probe_vs_Shuffled | block_bootstrap_len48 | `-0.000128` | [-0.000179, -0.000079] | 1.0 | True |
| ETTm2 | Probe_vs_Shuffled | every_12th_window_phase_bootstrap | `-0.000128` | [-0.000144, -0.000113] | 1.0 | True |

## Integrity

- **ExchangeRate**: PASS (checkpoints unchanged: True; no test cache: True; all purge assertions pass: True; observability holds: True; Common windows=1693, Full legal windows=2821)
- **Traffic**: PASS (checkpoints unchanged: True; no test cache: True; all purge assertions pass: True; observability holds: True; Common windows=4082, Full legal windows=6804)
- **BeijingAirQuality**: PASS (checkpoints unchanged: True; no test cache: True; all purge assertions pass: True; observability holds: True; Common windows=8512, Full legal windows=14186)
- **ETTm2**: PASS (checkpoints unchanged: True; no test cache: True; all purge assertions pass: True; observability holds: True; Common windows=13696, Full legal windows=22826)

## Claim (Section 13)

- **A_better_than_common**: True
- **B_competitive_with_full**: True
- **C_better_than_matchedpassive**: False
- **D_better_than_shuffled**: True
- **E_gains_multiple_datasets**: False
- **F_no_broad_regressions**: True

## Decision: MIXED

Partial, inconsistent evidence: LearnedProbe helps under some comparisons but not others, without broad regressions. Treat as suggestive, not confirmatory.

## Hard rule compliance

```text
TEST SET ACCESSED: NO
FORECASTING EXPERTS RETRAINED: NO
LEARNEDPROBE ARCHITECTURE/LOSS/TRAINING MODIFIED: NO
NONNEGATIVE FORECAST CLAMP: DISABLED (clamp_zero=False, per Section 7)
PURGE ASSERTION: see table above; raises AssertionError immediately if violated
```
