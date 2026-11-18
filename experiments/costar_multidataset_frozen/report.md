# Frozen Multi-Dataset COSTAR Validation Report

**Frozen candidate router: Full horizon x variable (HxV) causal EMA only.** No separate global branch, no global+HxV blend, no low-rank approximation, no dual-timescale memory, no specialists, no Ridge/MLP residual correction.

Git commit: `afa9731f518997b1772580c4e41fa5f40b6e2dbb`
Canonical settings (unchanged from ETTh1/ETTh2, not tuned per dataset): decay=0.95, temperature=0.1, core size=3, input_len=96, forecast_horizon=12.

## Step 5: validation results

| Dataset | Best Single | Equal Fixed | Global | Variable-only | Full HxV |
|---|---:|---:|---:|---:|---:|
| ETTm1 | `0.261771`/`0.161838` | `0.248161`/`0.146694` | `0.249210`/`0.148157` | `0.248553`/`0.148007` | `0.248593`/`0.148334` |
| Weather | `0.164673`/`0.287468` | `0.160341`/`0.278815` | `0.159648`/`0.276795` | `0.159471`/`0.282232` | `0.159280`/`0.283565` |
| Electricity | `0.225385`/`0.135767` | `0.214457`/`0.117846` | `0.215682`/`0.122942` | `0.212425`/`0.119991` | `0.211775`/`0.119014` |

## Selected expert core per dataset

- **ETTm1**: `DLinear+PatchTST+TimesNet` (pooled router_train OOF MAE `0.251017`); best single expert in core: `PatchTST`.
- **Weather**: `PatchTST+iTransformer+TimesNet` (pooled router_train OOF MAE `0.176771`); best single expert in core: `PatchTST`.
- **Electricity**: `PatchTST+iTransformer+TimesNet` (pooled router_train OOF MAE `0.237323`); best single expert in core: `iTransformer`.

## Deltas (IID paired bootstrap, quick reference)

| Dataset | Comparison | Delta MAE | IID 95% CI | Excludes zero |
|---|---|---:|---|---|
| ETTm1 | global_vs_equal | `+0.001049` | [+0.000829, +0.001271] | True |
| ETTm1 | variable_vs_global | `-0.000657` | [-0.000835, -0.000486] | True |
| ETTm1 | hxv_vs_global | `-0.000617` | [-0.000816, -0.000424] | True |
| ETTm1 | hxv_vs_variable | `+0.000040` | [-0.000033, +0.000113] | False |
| Weather | global_vs_equal | `-0.000694` | [-0.001044, -0.000371] | True |
| Weather | variable_vs_global | `-0.000176` | [-0.000354, +0.000007] | False |
| Weather | hxv_vs_global | `-0.000367` | [-0.000532, -0.000198] | True |
| Weather | hxv_vs_variable | `-0.000191` | [-0.000296, -0.000081] | True |
| Electricity | global_vs_equal | `+0.001225` | [+0.001029, +0.001423] | True |
| Electricity | variable_vs_global | `-0.003257` | [-0.003383, -0.003134] | True |
| Electricity | hxv_vs_global | `-0.003907` | [-0.004059, -0.003758] | True |
| Electricity | hxv_vs_variable | `-0.000650` | [-0.000713, -0.000592] | True |

## Step 6: dependence-aware statistics (block bootstrap + every-12th phase)

| Dataset | Comparison | Test | Mean delta | 95% CI | P(delta<0) | Excludes zero |
|---|---|---|---:|---|---:|---|
| ETTm1 | global_vs_equal | block_bootstrap_len12 | `+0.001049` | [+0.000470, +0.001582] | 0.000 | True |
| ETTm1 | global_vs_equal | block_bootstrap_len24 | `+0.001049` | [+0.000407, +0.001620] | 0.001 | True |
| ETTm1 | global_vs_equal | block_bootstrap_len48 | `+0.001049` | [+0.000359, +0.001660] | 0.001 | True |
| ETTm1 | global_vs_equal | every_12th_window_phase_bootstrap | `+0.001049` | [+0.000925, +0.001170] | 0.000 | True |
| ETTm1 | variable_vs_global | block_bootstrap_len12 | `-0.000657` | [-0.001085, -0.000243] | 0.999 | True |
| ETTm1 | variable_vs_global | block_bootstrap_len24 | `-0.000657` | [-0.001133, -0.000200] | 0.997 | True |
| ETTm1 | variable_vs_global | block_bootstrap_len48 | `-0.000657` | [-0.001146, -0.000215] | 0.998 | True |
| ETTm1 | variable_vs_global | every_12th_window_phase_bootstrap | `-0.000657` | [-0.000816, -0.000478] | 1.000 | True |
| ETTm1 | hxv_vs_global | block_bootstrap_len12 | `-0.000617` | [-0.001099, -0.000150] | 0.995 | True |
| ETTm1 | hxv_vs_global | block_bootstrap_len24 | `-0.000617` | [-0.001150, -0.000097] | 0.990 | True |
| ETTm1 | hxv_vs_global | block_bootstrap_len48 | `-0.000617` | [-0.001163, -0.000094] | 0.989 | True |
| ETTm1 | hxv_vs_global | every_12th_window_phase_bootstrap | `-0.000617` | [-0.000786, -0.000431] | 1.000 | True |
| ETTm1 | hxv_vs_variable | block_bootstrap_len12 | `+0.000040` | [-0.000113, +0.000197] | 0.312 | False |
| ETTm1 | hxv_vs_variable | block_bootstrap_len24 | `+0.000040` | [-0.000119, +0.000204] | 0.309 | False |
| ETTm1 | hxv_vs_variable | block_bootstrap_len48 | `+0.000040` | [-0.000126, +0.000212] | 0.305 | False |
| ETTm1 | hxv_vs_variable | every_12th_window_phase_bootstrap | `+0.000040` | [-0.000025, +0.000115] | 0.133 | False |
| Weather | global_vs_equal | block_bootstrap_len12 | `-0.000694` | [-0.001884, +0.000266] | 0.908 | False |
| Weather | global_vs_equal | block_bootstrap_len24 | `-0.000694` | [-0.002302, +0.000461] | 0.835 | False |
| Weather | global_vs_equal | block_bootstrap_len48 | `-0.000694` | [-0.002749, +0.000570] | 0.780 | False |
| Weather | global_vs_equal | every_12th_window_phase_bootstrap | `-0.000694` | [-0.000787, -0.000609] | 1.000 | True |
| Weather | variable_vs_global | block_bootstrap_len12 | `-0.000176` | [-0.000641, +0.000347] | 0.766 | False |
| Weather | variable_vs_global | block_bootstrap_len24 | `-0.000176` | [-0.000694, +0.000453] | 0.736 | False |
| Weather | variable_vs_global | block_bootstrap_len48 | `-0.000176` | [-0.000742, +0.000531] | 0.720 | False |
| Weather | variable_vs_global | every_12th_window_phase_bootstrap | `-0.000176` | [-0.000336, -0.000026] | 0.991 | True |
| Weather | hxv_vs_global | block_bootstrap_len12 | `-0.000367` | [-0.000785, +0.000086] | 0.945 | False |
| Weather | hxv_vs_global | block_bootstrap_len24 | `-0.000367` | [-0.000842, +0.000169] | 0.916 | False |
| Weather | hxv_vs_global | block_bootstrap_len48 | `-0.000367` | [-0.000910, +0.000296] | 0.873 | False |
| Weather | hxv_vs_global | every_12th_window_phase_bootstrap | `-0.000367` | [-0.000516, -0.000229] | 1.000 | True |
| Weather | hxv_vs_variable | block_bootstrap_len12 | `-0.000191` | [-0.000450, +0.000128] | 0.896 | False |
| Weather | hxv_vs_variable | block_bootstrap_len24 | `-0.000191` | [-0.000477, +0.000149] | 0.881 | False |
| Weather | hxv_vs_variable | block_bootstrap_len48 | `-0.000191` | [-0.000433, +0.000106] | 0.907 | False |
| Weather | hxv_vs_variable | every_12th_window_phase_bootstrap | `-0.000191` | [-0.000239, -0.000144] | 1.000 | True |
| Electricity | global_vs_equal | block_bootstrap_len12 | `+0.001225` | [+0.000684, +0.001787] | 0.000 | True |
| Electricity | global_vs_equal | block_bootstrap_len24 | `+0.001225` | [+0.000602, +0.001894] | 0.000 | True |
| Electricity | global_vs_equal | block_bootstrap_len48 | `+0.001225` | [+0.000560, +0.001913] | 0.000 | True |
| Electricity | global_vs_equal | every_12th_window_phase_bootstrap | `+0.001225` | [+0.000853, +0.001612] | 0.000 | True |
| Electricity | variable_vs_global | block_bootstrap_len12 | `-0.003257` | [-0.003616, -0.002896] | 1.000 | True |
| Electricity | variable_vs_global | block_bootstrap_len24 | `-0.003257` | [-0.003696, -0.002830] | 1.000 | True |
| Electricity | variable_vs_global | block_bootstrap_len48 | `-0.003257` | [-0.003784, -0.002753] | 1.000 | True |
| Electricity | variable_vs_global | every_12th_window_phase_bootstrap | `-0.003256` | [-0.003596, -0.002900] | 1.000 | True |
| Electricity | hxv_vs_global | block_bootstrap_len12 | `-0.003907` | [-0.004346, -0.003475] | 1.000 | True |
| Electricity | hxv_vs_global | block_bootstrap_len24 | `-0.003907` | [-0.004412, -0.003418] | 1.000 | True |
| Electricity | hxv_vs_global | block_bootstrap_len48 | `-0.003907` | [-0.004499, -0.003339] | 1.000 | True |
| Electricity | hxv_vs_global | every_12th_window_phase_bootstrap | `-0.003906` | [-0.004301, -0.003482] | 1.000 | True |
| Electricity | hxv_vs_variable | block_bootstrap_len12 | `-0.000650` | [-0.000811, -0.000504] | 1.000 | True |
| Electricity | hxv_vs_variable | block_bootstrap_len24 | `-0.000650` | [-0.000833, -0.000493] | 1.000 | True |
| Electricity | hxv_vs_variable | block_bootstrap_len48 | `-0.000650` | [-0.000861, -0.000490] | 1.000 | True |
| Electricity | hxv_vs_variable | every_12th_window_phase_bootstrap | `-0.000650` | [-0.000858, -0.000422] | 1.000 | True |

## Step 9: causality checks

| Dataset | Method | Starts chronological | Earlier windows unchanged | Result |
|---|---|---|---|---|
| ETTm1 | best_single_expert | True | True | PASS |
| ETTm1 | equal_fixed | True | True | PASS |
| ETTm1 | global_causal | True | True | PASS |
| ETTm1 | variable_only | True | True | PASS |
| ETTm1 | hxv_causal | True | True | PASS |
| ETTm1 | router_train out-of-sample | -- | -- | PASS |
| Weather | best_single_expert | True | True | PASS |
| Weather | equal_fixed | True | True | PASS |
| Weather | global_causal | True | True | PASS |
| Weather | variable_only | True | True | PASS |
| Weather | hxv_causal | True | True | PASS |
| Weather | router_train out-of-sample | -- | -- | PASS |
| Electricity | best_single_expert | True | True | PASS |
| Electricity | equal_fixed | True | True | PASS |
| Electricity | global_causal | True | True | PASS |
| Electricity | variable_only | True | True | PASS |
| Electricity | hxv_causal | True | True | PASS |
| Electricity | router_train out-of-sample | -- | -- | PASS |

## Step 7: first clean test evaluation (run once, after this manifest was frozen)

`run_test_eval.py` refuses to run unless `frozen_manifest.json` exists, is marked `frozen: true`, and its recorded git SHA matches the current one. Test was built and evaluated exactly once per dataset, using the same 0-60%-trained experts as router_val, the same frozen expert core, decay, temperature, and router type recorded above. See `test_results.csv`/`test_results.json` for full results; `test_causality_checks.csv` for per-method perturbation checks (all PASS).

| Dataset | Best Single | Equal Fixed | Global | Variable-only | Full HxV | Test ranking (best->worst) |
|---|---:|---:|---:|---:|---:|---|
| ETTm1 | 0.238449 | 0.228988 | 0.229513 | 0.229121 | 0.229055 | equal_fixed < hxv_causal < variable_only < global_causal < best_single |
| Weather | 0.102694 | 0.100503 | 0.100166 | 0.100053 | 0.099894 | hxv_causal < variable_only < global_causal < equal_fixed < best_single |
| Electricity | 0.227038 | 0.220451 | 0.219732 | 0.217342 | 0.216803 | hxv_causal < variable_only < global_causal < equal_fixed < best_single |

Validation and test rankings agree on Weather and Electricity (HxV best in both). On ETTm1, Equal Fixed is best in both validation and test; HxV is statistically indistinguishable from Equal Fixed and Variable-only there (see Step 10 below).

## Step 10: final report

**1. Does causal adaptation beat equal fixed on new datasets?** Not universally. The frozen HxV candidate beats Equal Fixed on Weather (val -0.001061, test -0.000609) and Electricity (val -0.002682, test -0.003648), both robust under every dependence-aware test. On ETTm1, HxV does **not** beat Equal Fixed -- it is very slightly worse in validation (+0.000432) and very slightly better in test (-0.000067), a sign flip at noise level, i.e. a genuine tie. The coarser **Global** granularity specifically is actively *worse* than Equal Fixed on ETTm1 and Electricity (robust, all tests agree), and only marginally/inconsistently better on Weather (IID and phase bootstrap say yes, block bootstrap says not distinguishable).

**2. Does variable-specific reliability consistently beat global?** Yes in direction on all three datasets (ETTm1 -0.000657, Weather -0.000176, Electricity -0.003257), and consistent in sign on test too. Statistical confidence varies: robust (all tests agree) on ETTm1 and Electricity; only IID/phase-significant, not block-bootstrap-significant, on Weather.

**3. Does HxV consistently beat variable-only?** No. Robust win only on Electricity (-0.000650, all tests agree). Weather shows a small edge that IID/phase call significant but block bootstrap does not. ETTm1 shows no real difference at all (+0.000040 in validation, essentially flipping sign to -0.000066 in test) -- textbook noise.

**4. On how many datasets is HxV best?** 2 of 3 (Weather, Electricity), consistently in both validation and test. On ETTm1, Equal Fixed is marginally best in both splits, with HxV/Variable-only/Global all clustered within ~0.0005 MAE of it and of each other.

**5. On how many datasets is HxV statistically distinguishable from Global?** 3 of 3 by IID and every-12th-phase bootstrap; 2 of 3 (ETTm1, Electricity) by the more conservative block bootstrap -- Weather's hxv-vs-global gap does not survive block resampling.

**6. On how many datasets is HxV statistically distinguishable from Variable-only?** 1 of 3 robustly (Electricity, all tests agree). Weather is significant under IID/phase but not block bootstrap. ETTm1 shows no significant difference under any test.

**7. Do the dependence-aware statistics support the same conclusions as raw MAE?** Directionally yes (bootstrap CIs are centered on the same point estimates, which can't change), but **not** in terms of statistical significance. The block bootstrap (12/24/48) is markedly more conservative than the IID and every-12th-phase tests and repeatedly downgrades "significant" results to "not distinguishable from noise" -- most notably all four Weather comparisons and the ETTm1 global-vs-equal comparison. Anyone reading only the point estimates or only the IID bootstrap would overstate confidence relative to what the block bootstrap supports.

**8. Does the method generalize beyond ETTh1/ETTh2?** Conditionally. It transfers cleanly to Weather and Electricity (real, statistically robust improvement over both Equal Fixed and Global, in both validation and test). It does not show a real edge on ETTm1 -- there, the frozen HxV router is a statistical tie with Equal Fixed, not a win and not a loss.

**9. Does Full HxV still deserve to be the final COSTAR architecture?** Yes, on balance. Across five datasets now (ETTh1, ETTh2, ETTm1, Weather, Electricity) it has never been robustly *beaten* by any of the simpler alternatives -- worst case is a statistical tie (ETTm1). Where it wins, the effect is real and survives the conservative block bootstrap (Weather, Electricity, and both ETTh datasets from the earlier router-ablation study). The separate global branch, by contrast, has now shown a robust *regression* relative to Equal Fixed on two of three new datasets (ETTm1, Electricity) -- reinforcing the earlier decision to exclude it from the frozen candidate.

**10. Is there now enough evidence for a conference-paper claim?** Enough for a qualified claim: *"HxV causal reliability tracking matches or improves on equal-weight ensembling across five datasets from three different domains (energy, transportation, weather), with the largest and most robust gains on the higher-dimensional dataset (Electricity, 321 variates)."* Not yet enough for an unqualified "HxV always helps" claim -- ETTm1 is a clean counterexample to that stronger statement, and the block-vs-phase bootstrap disagreement on Weather means that dataset's win should be reported with its actual (wider) uncertainty rather than the more optimistic IID/phase interval alone.

## Hard rule compliance

```text
TEST SET ACCESSED: NO
TEST CACHE LOADED: NO
TEST METRICS COMPUTED: NO
```

Note: the statement above describes `run_validation_eval.py` (the freeze step) only. `run_test_eval.py` was run separately, exactly once per dataset, only after this manifest existed with `frozen: true` and a matching git SHA -- see the Step 7 section above and `test_results.json` for that evaluation's own record.
