# Simplex vs Simplex + LearnedProbe

Question: does the frozen learned diagnostic probe provide useful CURRENT instance-level competence information on top of a strong static nonnegative-simplex ensemble, beyond what the simplex's fixed router_train-fitted weights already capture?

Base Simplex is refit here on the train-selected 3-expert core (the only existing `fit_simplex_weights` precedent in this repo is full-5-expert-pool and test-touched, so it is reused unmodified but applied to a core-subset prediction_stack -- see `simplex_probe_manifest.json` for the exact justification). Simplex and Simplex+Probe always use the identical core and identical base forecasts.

## 1. Was the base Simplex result reproduced exactly?

| Dataset | Max |Δweights| (2 independent fits) | Max |Δpredictions| | Reproducible |
|---|---:|---:|---|
| ExchangeRate | 0.000e+00 | 0.000e+00 | True |
| Traffic | 0.000e+00 | 0.000e+00 | True |
| BeijingAirQuality | 0.000e+00 | 0.000e+00 | True |
| ETTm2 | 0.000e+00 | 0.000e+00 | True |

## 2-3. Expert core and selected alpha

| Dataset | Core | Selected alpha (router_train OOF) |
|---|---|---:|
| ExchangeRate | ['PatchTST', 'iTransformer', 'TimesNet'] | 2.0 |
| Traffic | ['PatchTST', 'iTransformer', 'ModernTCN'] | 0.5 |
| BeijingAirQuality | ['PatchTST', 'iTransformer', 'TimesNet'] | 0.5 |
| ETTm2 | ['PatchTST', 'iTransformer', 'TimesNet'] | 1.0 |

## Alpha selection (router_train OOF, pooled over chronological folds)

| Dataset | Alpha | OOF MAE | OOF MSE | Selected |
|---|---:|---:|---:|---|
| ExchangeRate | 0.0 | 0.095196 | 0.018539 |  |
| ExchangeRate | 0.25 | 0.092052 | 0.017565 |  |
| ExchangeRate | 0.5 | 0.089917 | 0.016948 |  |
| ExchangeRate | 1.0 | 0.087808 | 0.016385 |  |
| ExchangeRate | 2.0 | 0.086824 | 0.016161 | <-- selected |
| Traffic | 0.0 | 0.295090 | 0.283148 |  |
| Traffic | 0.25 | 0.287594 | 0.280012 |  |
| Traffic | 0.5 | 0.285461 | 0.282223 | <-- selected |
| Traffic | 1.0 | 0.288506 | 0.291191 |  |
| Traffic | 2.0 | 0.294270 | 0.301227 |  |
| BeijingAirQuality | 0.0 | 0.344003 | 0.321867 |  |
| BeijingAirQuality | 0.25 | 0.341954 | 0.321201 |  |
| BeijingAirQuality | 0.5 | 0.341339 | 0.322012 | <-- selected |
| BeijingAirQuality | 1.0 | 0.342147 | 0.324994 |  |
| BeijingAirQuality | 2.0 | 0.344066 | 0.328606 |  |
| ETTm2 | 0.0 | 0.173103 | 0.093067 |  |
| ETTm2 | 0.25 | 0.171488 | 0.091105 |  |
| ETTm2 | 0.5 | 0.170602 | 0.090041 |  |
| ETTm2 | 1.0 | 0.170202 | 0.089365 | <-- selected |
| ETTm2 | 2.0 | 0.170932 | 0.089652 |  |

## 4-7. Primary results (router_val MAE / MSE)

| Dataset | Simplex | Simplex+Probe | Simplex+ShuffledProbe | Δ Probe vs Simplex | % improvement | Δ Shuffled vs Simplex | Δ Real vs Shuffled |
|---|---:|---:|---:|---:|---:|---:|---:|
| ExchangeRate | 0.127167 | 0.119971 | 0.130543 | `-0.007196` | `+5.659%` | `+0.003376` | `-0.010572` |
| Traffic | 0.280925 | 0.270460 | 0.288867 | `-0.010465` | `+3.725%` | `+0.007942` | `-0.018407` |
| BeijingAirQuality | 0.257947 | 0.258301 | 0.259735 | `+0.000354` | `-0.137%` | `+0.001788` | `-0.001434` |
| ETTm2 | 0.161504 | 0.162743 | 0.163426 | `+0.001239` | `-0.767%` | `+0.001922` | `-0.000683` |

## Primary dependence-aware statistics (block-24, per Section 12)

| Dataset | Comparison | Mean Δ | 95% CI | P(Δ<0) | Excludes zero |
|---|---|---:|---|---:|---|
| ExchangeRate | Probe_vs_Simplex (block24) | `-0.007196` | [-0.009612, -0.004933] | 1.000 | True |
| ExchangeRate | Probe_vs_Shuffled (block24) | `-0.010572` | [-0.013505, -0.007881] | 1.000 | True |
| Traffic | Probe_vs_Simplex (block24) | `-0.010465` | [-0.012138, -0.008832] | 1.000 | True |
| Traffic | Probe_vs_Shuffled (block24) | `-0.018407` | [-0.020154, -0.016663] | 1.000 | True |
| BeijingAirQuality | Probe_vs_Simplex (block24) | `+0.000354` | [-0.000619, +0.001281] | 0.255 | False |
| BeijingAirQuality | Probe_vs_Shuffled (block24) | `-0.001434` | [-0.002487, -0.000357] | 0.996 | True |
| ETTm2 | Probe_vs_Simplex (block24) | `+0.001239` | [+0.000846, +0.001631] | 0.000 | True |
| ETTm2 | Probe_vs_Shuffled (block24) | `-0.000683` | [-0.001063, -0.000296] | 1.000 | True |

## Full dependence-aware statistics (all block lengths + phase)

| Dataset | Comparison | Test | Mean Δ | 95% CI | P(Δ<0) | Excludes zero |
|---|---|---|---:|---|---:|---|
| ExchangeRate | Probe_vs_Simplex | iid_paired_bootstrap | `-0.007196` | [-0.008167, -0.006248] |  | True |
| ExchangeRate | Probe_vs_Simplex | block_bootstrap_len12 | `-0.007196` | [-0.009444, -0.005094] | 1.0 | True |
| ExchangeRate | Probe_vs_Simplex | block_bootstrap_len24 | `-0.007196` | [-0.009612, -0.004933] | 1.0 | True |
| ExchangeRate | Probe_vs_Simplex | block_bootstrap_len48 | `-0.007196` | [-0.010088, -0.004745] | 1.0 | True |
| ExchangeRate | Probe_vs_Simplex | every_12th_window_phase_bootstrap | `-0.007194` | [-0.007956, -0.006474] | 1.0 | True |
| ExchangeRate | Shuffled_vs_Simplex | iid_paired_bootstrap | `+0.003376` | [+0.002411, +0.004320] |  | True |
| ExchangeRate | Shuffled_vs_Simplex | block_bootstrap_len12 | `+0.003376` | [+0.002339, +0.004390] | 0.0 | True |
| ExchangeRate | Shuffled_vs_Simplex | block_bootstrap_len24 | `+0.003376` | [+0.002212, +0.004485] | 0.0 | True |
| ExchangeRate | Shuffled_vs_Simplex | block_bootstrap_len48 | `+0.003376` | [+0.002132, +0.004663] | 0.0 | True |
| ExchangeRate | Shuffled_vs_Simplex | every_12th_window_phase_bootstrap | `+0.003378` | [+0.002819, +0.003897] | 0.0 | True |
| ExchangeRate | Probe_vs_Shuffled | iid_paired_bootstrap | `-0.010572` | [-0.011900, -0.009215] |  | True |
| ExchangeRate | Probe_vs_Shuffled | block_bootstrap_len12 | `-0.010572` | [-0.013165, -0.008213] | 1.0 | True |
| ExchangeRate | Probe_vs_Shuffled | block_bootstrap_len24 | `-0.010572` | [-0.013505, -0.007881] | 1.0 | True |
| ExchangeRate | Probe_vs_Shuffled | block_bootstrap_len48 | `-0.010572` | [-0.014214, -0.007563] | 1.0 | True |
| ExchangeRate | Probe_vs_Shuffled | every_12th_window_phase_bootstrap | `-0.010572` | [-0.011398, -0.009765] | 1.0 | True |
| Traffic | Probe_vs_Simplex | iid_paired_bootstrap | `-0.010465` | [-0.011158, -0.009790] |  | True |
| Traffic | Probe_vs_Simplex | block_bootstrap_len12 | `-0.010465` | [-0.011966, -0.008942] | 1.0 | True |
| Traffic | Probe_vs_Simplex | block_bootstrap_len24 | `-0.010465` | [-0.012138, -0.008832] | 1.0 | True |
| Traffic | Probe_vs_Simplex | block_bootstrap_len48 | `-0.010465` | [-0.012462, -0.008604] | 1.0 | True |
| Traffic | Probe_vs_Simplex | every_12th_window_phase_bootstrap | `-0.010464` | [-0.012050, -0.008793] | 1.0 | True |
| Traffic | Shuffled_vs_Simplex | iid_paired_bootstrap | `+0.007942` | [+0.007066, +0.008835] |  | True |
| Traffic | Shuffled_vs_Simplex | block_bootstrap_len12 | `+0.007942` | [+0.007069, +0.008772] | 0.0 | True |
| Traffic | Shuffled_vs_Simplex | block_bootstrap_len24 | `+0.007942` | [+0.007068, +0.008779] | 0.0 | True |
| Traffic | Shuffled_vs_Simplex | block_bootstrap_len48 | `+0.007942` | [+0.007081, +0.008782] | 0.0 | True |
| Traffic | Shuffled_vs_Simplex | every_12th_window_phase_bootstrap | `+0.007942` | [+0.007262, +0.008702] | 0.0 | True |
| Traffic | Probe_vs_Shuffled | iid_paired_bootstrap | `-0.018407` | [-0.019501, -0.017318] |  | True |
| Traffic | Probe_vs_Shuffled | block_bootstrap_len12 | `-0.018407` | [-0.020003, -0.016783] | 1.0 | True |
| Traffic | Probe_vs_Shuffled | block_bootstrap_len24 | `-0.018407` | [-0.020154, -0.016663] | 1.0 | True |
| Traffic | Probe_vs_Shuffled | block_bootstrap_len48 | `-0.018407` | [-0.020406, -0.016537] | 1.0 | True |
| Traffic | Probe_vs_Shuffled | every_12th_window_phase_bootstrap | `-0.018406` | [-0.020220, -0.016463] | 1.0 | True |
| BeijingAirQuality | Probe_vs_Simplex | iid_paired_bootstrap | `+0.000354` | [-0.000078, +0.000776] |  | False |
| BeijingAirQuality | Probe_vs_Simplex | block_bootstrap_len12 | `+0.000354` | [-0.000553, +0.001251] | 0.2345000058412552 | False |
| BeijingAirQuality | Probe_vs_Simplex | block_bootstrap_len24 | `+0.000354` | [-0.000619, +0.001281] | 0.2551000118255615 | False |
| BeijingAirQuality | Probe_vs_Simplex | block_bootstrap_len48 | `+0.000354` | [-0.000675, +0.001256] | 0.27480000257492065 | False |
| BeijingAirQuality | Probe_vs_Simplex | every_12th_window_phase_bootstrap | `+0.000354` | [-0.000100, +0.000812] | 0.06620000302791595 | False |
| BeijingAirQuality | Shuffled_vs_Simplex | iid_paired_bootstrap | `+0.001788` | [+0.001337, +0.002234] |  | True |
| BeijingAirQuality | Shuffled_vs_Simplex | block_bootstrap_len12 | `+0.001788` | [+0.001272, +0.002249] | 0.0 | True |
| BeijingAirQuality | Shuffled_vs_Simplex | block_bootstrap_len24 | `+0.001788` | [+0.001239, +0.002262] | 0.0 | True |
| BeijingAirQuality | Shuffled_vs_Simplex | block_bootstrap_len48 | `+0.001788` | [+0.001205, +0.002229] | 0.0 | True |
| BeijingAirQuality | Shuffled_vs_Simplex | every_12th_window_phase_bootstrap | `+0.001788` | [+0.001330, +0.002253] | 0.0 | True |
| BeijingAirQuality | Probe_vs_Shuffled | iid_paired_bootstrap | `-0.001434` | [-0.002024, -0.000846] |  | True |
| BeijingAirQuality | Probe_vs_Shuffled | block_bootstrap_len12 | `-0.001434` | [-0.002426, -0.000421] | 0.9973999857902527 | True |
| BeijingAirQuality | Probe_vs_Shuffled | block_bootstrap_len24 | `-0.001434` | [-0.002487, -0.000357] | 0.9955999851226807 | True |
| BeijingAirQuality | Probe_vs_Shuffled | block_bootstrap_len48 | `-0.001434` | [-0.002527, -0.000385] | 0.9951000213623047 | True |
| BeijingAirQuality | Probe_vs_Shuffled | every_12th_window_phase_bootstrap | `-0.001434` | [-0.002067, -0.000776] | 1.0 | True |
| ETTm2 | Probe_vs_Simplex | iid_paired_bootstrap | `+0.001239` | [+0.001033, +0.001441] |  | True |
| ETTm2 | Probe_vs_Simplex | block_bootstrap_len12 | `+0.001239` | [+0.000878, +0.001595] | 0.0 | True |
| ETTm2 | Probe_vs_Simplex | block_bootstrap_len24 | `+0.001239` | [+0.000846, +0.001631] | 0.0 | True |
| ETTm2 | Probe_vs_Simplex | block_bootstrap_len48 | `+0.001239` | [+0.000842, +0.001621] | 0.0 | True |
| ETTm2 | Probe_vs_Simplex | every_12th_window_phase_bootstrap | `+0.001239` | [+0.001115, +0.001354] | 0.0 | True |
| ETTm2 | Shuffled_vs_Simplex | iid_paired_bootstrap | `+0.001922` | [+0.001706, +0.002144] |  | True |
| ETTm2 | Shuffled_vs_Simplex | block_bootstrap_len12 | `+0.001922` | [+0.001673, +0.002158] | 0.0 | True |
| ETTm2 | Shuffled_vs_Simplex | block_bootstrap_len24 | `+0.001922` | [+0.001667, +0.002166] | 0.0 | True |
| ETTm2 | Shuffled_vs_Simplex | block_bootstrap_len48 | `+0.001922` | [+0.001655, +0.002181] | 0.0 | True |
| ETTm2 | Shuffled_vs_Simplex | every_12th_window_phase_bootstrap | `+0.001922` | [+0.001790, +0.002063] | 0.0 | True |
| ETTm2 | Probe_vs_Shuffled | iid_paired_bootstrap | `-0.000683` | [-0.000967, -0.000397] |  | True |
| ETTm2 | Probe_vs_Shuffled | block_bootstrap_len12 | `-0.000683` | [-0.001054, -0.000308] | 0.9998000264167786 | True |
| ETTm2 | Probe_vs_Shuffled | block_bootstrap_len24 | `-0.000683` | [-0.001063, -0.000296] | 0.9998000264167786 | True |
| ETTm2 | Probe_vs_Shuffled | block_bootstrap_len48 | `-0.000683` | [-0.001071, -0.000301] | 0.9997000098228455 | True |
| ETTm2 | Probe_vs_Shuffled | every_12th_window_phase_bootstrap | `-0.000683` | [-0.000871, -0.000518] | 1.0 | True |

## 9. Weight-concentration analysis

| Dataset | Method | Mean entropy | Median entropy | Mean max weight | Mean eff. #experts | Fraction top-expert changed |
|---|---|---:|---:|---:|---:|---:|
| ExchangeRate | Simplex | 1.0982 | 1.0982 | 0.3435 | 2.998 | 0.000 |
| ExchangeRate | Simplex_Probe | 0.5291 | 0.5813 | 0.7528 | 1.574 | 0.115 |
| ExchangeRate | Simplex_ShuffledProbe | 0.5340 | 0.5850 | 0.7498 | 1.581 | 0.663 |
| Traffic | Simplex | 0.9248 | 0.9248 | 0.5099 | 2.331 | 0.000 |
| Traffic | Simplex_Probe | 0.7983 | 0.7894 | 0.6760 | 1.905 | 0.078 |
| Traffic | Simplex_ShuffledProbe | 0.8487 | 0.8166 | 0.6154 | 2.082 | 0.423 |
| BeijingAirQuality | Simplex | 1.0512 | 1.0512 | 0.4389 | 2.755 | 0.000 |
| BeijingAirQuality | Simplex_Probe | 0.9493 | 0.9522 | 0.5646 | 2.341 | 0.164 |
| BeijingAirQuality | Simplex_ShuffledProbe | 0.9538 | 0.9256 | 0.5458 | 2.377 | 0.496 |
| ETTm2 | Simplex | 1.0571 | 1.0571 | 0.4127 | 2.786 | 0.000 |
| ETTm2 | Simplex_Probe | 0.7093 | 0.7056 | 0.7108 | 1.760 | 0.644 |
| ETTm2 | Simplex_ShuffledProbe | 0.7261 | 0.7408 | 0.7027 | 1.792 | 0.620 |

## Integrity

- **ExchangeRate**: PASS (checkpoints unchanged: True; no test cache: True; alpha=0 reproduces Simplex: True (max weight diff 0.00e+00, max pred diff normalized 0.00e+00, MAE diff 0.00e+00); target-corruption invariant: True)
- **Traffic**: PASS (checkpoints unchanged: True; no test cache: True; alpha=0 reproduces Simplex: True (max weight diff 0.00e+00, max pred diff normalized 0.00e+00, MAE diff 0.00e+00); target-corruption invariant: True)
- **BeijingAirQuality**: PASS (checkpoints unchanged: True; no test cache: True; alpha=0 reproduces Simplex: True (max weight diff 2.98e-08, max pred diff normalized 1.28e-06, MAE diff 0.00e+00); target-corruption invariant: True)
- **ETTm2**: PASS (checkpoints unchanged: True; no test cache: True; alpha=0 reproduces Simplex: True (max weight diff 2.98e-08, max pred diff normalized 7.29e-07, MAE diff 1.49e-08); target-corruption invariant: True)

## Answers

**1. Was the base Simplex result reproduced exactly?** Yes on all datasets: max |Δweights| and |Δpredictions| across two independent fits were < 1e-6 (see table above). No pre-existing Simplex result exists for these four (validation-only) datasets to compare against externally -- see manifest for why this is a self-consistency/determinism check rather than a match to a prior stored number.
**2. What expert set was used?** The train-selected 3-expert core (identical for Simplex and Simplex+Probe on every dataset): [['PatchTST', 'iTransformer', 'TimesNet'], ['PatchTST', 'iTransformer', 'ModernTCN'], ['PatchTST', 'iTransformer', 'TimesNet'], ['PatchTST', 'iTransformer', 'TimesNet']].
**3. What alpha was selected using router_train?** [('ExchangeRate', 2.0), ('Traffic', 0.5), ('BeijingAirQuality', 0.5), ('ETTm2', 1.0)].
**4-5. Does Simplex+Probe beat Simplex by point estimate, on how many datasets?** 2/4.
**6. Which gains survive the primary block-24 bootstrap?** 2/4 datasets.
**7. Are there any significant regressions?** 1/4 datasets significant at block-24.
**8. Does Real Probe beat ShuffledProbe?** By point estimate on 4/4; block-24 significant on 4/4.
**9. Does Probe simply sharpen the weights, or provide expert-specific information?** See the weight-concentration table: compare mean effective-number-of-experts and fraction-top-expert-changed for Simplex+Probe vs Simplex+ShuffledProbe -- if ShuffledProbe concentrates/spreads weights similarly to RealProbe but does not improve MAE, the effect is expert-specific information, not generic sharpening.
**10. Does Probe still help on datasets where LearnedProbe-Rank previously failed to beat Equal?** BeijingAirQuality and ETTm2 are exactly the two datasets where LearnedProbe-Rank previously lost to plain Equal averaging (see ../reports/learned_probe_generalization_validation.md) -- compare their rows above to see whether the Simplex+Probe fusion recovers a benefit there.
**11. Is there evidence LearnedProbe provides useful information OUTSIDE C-Rank?** Probe may contain some useful information, but inspect when/why it helps before moving to COSTAR.

## Decision: MIXED

Probe may contain some useful information, but inspect when/why it helps before moving to COSTAR.

## Hard rule compliance

```text
TEST SET ACCESSED: NO
FORECASTING EXPERTS RETRAINED: NO
LEARNEDPROBE ARCHITECTURE/LOSS/TRAINING MODIFIED: NO
OTHER ROUTERS (Frozen/Online COSTAR, Top-1, Top-k, Ridge, Granger-Ramanathan) TOUCHED: NO
```
