# Current State

Last updated: 2026-08-17

Read this first. It is a compact project memory for the COSTAR-TS research branch in this BasicTS repository.

## Current Research Goal

Improve ETTh1 multivariate forecasting by combining frozen expert forecasts with adaptive COSTAR-style weighting/routing, while keeping chronological train/validation/test separation clean. The current target from recent prompts is validation MAE `<= 0.3619`; this has not been reached.

## Final Frozen Test Evaluation

CONFIRMED RESULT:

The preregistered frozen models were evaluated on test after explicit user authorization. Test metrics are now seen and must not be reused for tuning.

Artifacts:

- `experiments/final_test_evaluation/FINAL_TEST_RESULTS.json`
- `experiments/final_test_evaluation/FINAL_TEST_REPORT.md`
- `project_memory/experiments/2026-08-13_final_frozen_test_evaluation.md`

Final test results:

| Dataset | Method | Test MAE | Test MSE | Validation MAE | Selection protocol |
|---|---:|---:|---:|---|
| ETTh1 | Best single `iTransformer` | `0.339080` | `0.278551` | `0.376550` | validation-best single reference |
| ETTh1 | Train-selected fixed core `PatchTST+iTransformer+TimesNet` | `0.327128` | `0.266583` | `0.367265` | router-train selected core |
| ETTh1 | Full frozen adaptive | `0.326395` | `0.267509` | `0.363112` | preregistered final model |
| ETTh2 | Best single `DLinear` | `0.301708` | `0.222694` | `0.280957` | canonical single reference |
| ETTh2 | Train-selected fixed core `DLinear+PatchTST+ModernTCN` | `0.304642` | `0.225185` | `0.280878` | router-train selected core |
| ETTh2 | Full frozen adaptive | `0.297808` | `0.218612` | `0.276832` | preregistered final model |
| ETTh2 | `DLinear+ModernTCN` | `0.299263` | `0.221853` | `0.275229` | validation-selected reference only |

Conclusion:

The frozen adaptive model's relative MAE gain survived test on both datasets versus its own train-selected fixed core. ETTh2 also beat the validation-selected `DLinear+ModernTCN` reference on test, although ETTh2 absolute test metrics were worse than validation for every reported method.

## frozen-model Top COSTAR Test Audit

FROZEN-MODEL TEST RESULTS:

After the original confirmatory evaluation, frozen-model audits evaluated top validated COSTAR-style methods. These results are not clean preregistered competitors and must not be used for tuning.

Artifacts:

- `experiments/frozen_model_test_results/TOP_COSTAR_TEST_RESULTS.json`
- `experiments/frozen_model_test_results/FROZEN_MODEL_TOP_COSTAR_TEST_REPORT.md`
- `project_memory/experiments/2026-08-13_frozen_model_top_costar_test_results.md`
- `experiments/frozen_model_test_results/ETTH2_TOP_COSTAR_TEST_RESULTS.json`
- `experiments/frozen_model_test_results/FROZEN_MODEL_ETTH2_TOP_COSTAR_TEST_REPORT.md`
- `project_memory/experiments/2026-08-13_frozen_model_etth2_top_costar_test_results.md`

Top frozen-model ETTh1 test MAE rows:

| Method | Test MAE | Test MSE | Diff vs fixed core | Notes |
|---|---:|---:|---:|---|
| MLP residual corrector | `0.326047` | `0.267322` | `-0.001081` | additional frozen-model result |
| Expanded both final frozen | `0.326393` | `0.267506` | `-0.000735` | clean final model |
| Ridge residual corrector | `0.326448` | `0.267452` | `-0.000680` | former validation best |
| Horizon-variable hybrid | `0.326493` | `0.267638` | `-0.000635` | former validation best |
| Fixed core equal | `0.327128` | `0.266583` | `0.000000` | anchor |

Per-seed mean for MLP was MAE `0.326062 +/- 0.000053`; the table's primary multi-seed rows use the mean prediction across seeds. The MLP result is additional frozen-model evidence not listed in the original confirmatory freeze artifact.

Top frozen-model ETTh2 test MAE rows:

| Method | Test MAE | Test MSE | Diff vs DLinear | Notes |
|---|---:|---:|---:|---|
| Full adaptive train-selected fixed-3 final frozen | `0.297808` | `0.218612` | `-0.003899` | clean final ETTh2 model |
| Fixed-2 `DLinear+TimesNet` reference | `0.298398` | `0.221926` | `-0.003310` | validation-ranked fixed reference |
| Fixed-3 `DLinear+TimesNet+ModernTCN` reference | `0.299169` | `0.223927` | `-0.002539` | validation-selected reference only |
| Fixed-2 `DLinear+ModernTCN` reference | `0.299263` | `0.221853` | `-0.002445` | validation-selected reference only |
| Full adaptive variable-size `DLinear` core | `0.301093` | `0.222139` | `-0.000615` | frozen-model audit |
| Single `DLinear` | `0.301708` | `0.222694` | `0.000000` | anchor |

The original ETTh2 frozen-model audit's best row was the preregistered final frozen adaptive model. A later locked ETTh1-config ETTh2 replication found MLP/ridge/dynamic analogues slightly below the frozen adaptive test MAE, but those rows were generated after the earlier final ETTh2 test evaluation and do not supersede the preregistered result.

## Sequential COSTAR Test Audit

ADDITIONAL AFTER-FINAL-TEST AUDIT:

After explicit user authorization on 2026-08-14, existing frozen `utility_pairwise_weighted` Sequential COSTAR checkpoints were evaluated on the already generated final-test caches. No retraining or tuning was performed.

Results:

- ETTh1 Sequential COSTAR test MAE/MSE: `0.330832 +/- 0.005462` / `0.271398 +/- 0.005434`, average queries `3.776`.
- ETTh2 Sequential COSTAR test MAE/MSE: `0.300576 +/- 0.000032` / `0.222171 +/- 0.000069`, average queries `3.998`.

Both are worse than the final frozen adaptive test rows, so this does not change the current best result.

Artifacts:

- `experiments/sequential_costar_test_evaluation/SEQUENTIAL_COSTAR_TEST_RESULTS.json`
- `experiments/sequential_costar_test_evaluation/SEQUENTIAL_COSTAR_TEST_REPORT.md`
- `project_memory/experiments/2026-08-14_sequential_costar_test_evaluation.md`

## Frozen COSTAR Validation Diagnostic

VALIDATION-ONLY DIAGNOSTIC:

On 2026-08-16, a non-sequential `frozen_costar` path was implemented to isolate validation-time target feedback in the current adaptive COSTAR family. It keeps the same selected core, frozen expert forecasts, static/equal prior, hyperparameters, and `0.25` chronological / `0.75` horizon-variable mixture, but repeats router-train initialized weights across validation and does not update from validation targets or masks.

Validation results:

| Dataset | Equal fixed-three | Frozen COSTAR | Online COSTAR |
|---|---:|---:|---:|
| ETTh1 | `0.367265` / `0.310530` | `0.365868` / `0.308465` | `0.363111` / `0.306056` |
| ETTh2 | `0.280878` / `0.171933` | `0.277481` / `0.167632` | `0.276832` / `0.167280` |

Leakage checks passed: replacing validation targets or masks leaves frozen predictions exactly unchanged, online predictions change under target replacement, and frozen/online begin from equivalent train-derived first-window initialization.

Conclusion:

The frozen path improves over equal fixed-three, but the existing online causal updates remain materially better on both datasets. This supports the interpretation that sequential causal adaptation is contributing useful validation signal.

Artifacts:

- `experiments/frozen_costar/frozen_costar_validation_results.json`
- `experiments/frozen_costar/frozen_costar_report.md`
- `project_memory/experiments/2026-08-16_frozen_costar_validation.md`

## Equal-Static COSTAR Cleanup

VALIDATION-ONLY STRUCTURAL CLEANUP:

On 2026-08-17, the ETTh1 full adaptive path was changed so every selected triple receives the same equal static prior. The old `OLD_FIXED3` exception that loaded a trained static neural-router checkpoint was removed from the active path.

New ETTh1 equal-static validation result:

- Core: `PatchTST+iTransformer+TimesNet`
- Full adaptive validation MAE/MSE: `0.363100` / `0.306026`
- Previous neural-prior path reference: `0.363112` / `0.306057`

Updated frozen-vs-online diagnostic:

| Dataset | Equal fixed-three | Frozen COSTAR | Online COSTAR |
|---|---:|---:|---:|
| ETTh1 | `0.367265` / `0.310530` | `0.365825` / `0.308399` | `0.363100` / `0.306026` |
| ETTh2 | `0.280878` / `0.171933` | `0.277481` / `0.167632` | `0.276832` / `0.167280` |

No test cache was loaded. Because this cleanup occurred after prior final-test results were already seen, do not treat it as a replacement preregistered final-test result without a new explicit freeze/evaluation protocol.

Artifacts:

- `experiments/train_selected_core_etth1_equal_static/final_report.json`
- `experiments/frozen_costar/frozen_costar_validation_results.json`
- `project_memory/experiments/2026-08-17_equal_static_costar_cleanup.md`

## ETTh2 Pair-Potential Test Audit

ADDITIONAL AFTER-FINAL-TEST AUDIT:

After explicit user authorization on 2026-08-14, the two ETTh2 pair-potential linear ensembles that were previously validation-only were evaluated once on the locked ETTh2 test cache. No weights or hyperparameters were changed after test load.

Results:

- `nonnegative_simplex_linear_average`: test MAE/MSE `0.297120` / `0.218587`, validation MAE/MSE `0.274755` / `0.165479`.
- `ridge_linear_stacker`: test MAE/MSE `0.298382` / `0.218201`, validation MAE/MSE `0.276702` / `0.165339`.

The simplex ensemble beat the final frozen adaptive ETTh2 test MAE by `0.000688`, but it is an after-final-test audit row and does not supersede the preregistered frozen ETTh2 model.

Artifacts:

- `experiments/etth2_pair_potential_test_evaluation/ETTH2_PAIR_POTENTIAL_TEST_RESULTS.json`
- `experiments/etth2_pair_potential_test_evaluation/ETTH2_PAIR_POTENTIAL_TEST_REPORT.md`
- `project_memory/experiments/2026-08-14_etth2_pair_potential_test_evaluation.md`

## ETTh1 Simplex Linear Test Audit

ADDITIONAL AFTER-FINAL-TEST AUDIT:

After explicit user authorization on 2026-08-14, the ETTh1 analogue of the ETTh2 nonnegative simplex all-five linear ensemble was fit on ETTh1 router-train only and evaluated once on the already generated ETTh1 test cache.

Result:

- `nonnegative_simplex_linear_average`: validation MAE/MSE `0.366483` / `0.308484`; test MAE/MSE `0.326926` / `0.267713`.
- Weights: DLinear `0.116751`, PatchTST `0.364088`, iTransformer `0.339654`, TimesNet `0.148853`, ModernTCN `0.030653`.
- Diff vs fixed-three core test MAE: `-0.000203`; paired CI `[-0.000767, 0.000368]`, crossing zero.
- Diff vs full adaptive ETTh1 test MAE: `+0.000530`.

This does not change the current ETTh1 interpretation: simplex is a useful simple baseline but does not beat the full adaptive or stronger residual/specialist audit rows.

Artifacts:

- `experiments/etth1_simplex_linear_test_evaluation/ETTH1_SIMPLEX_LINEAR_TEST_RESULTS.json`
- `experiments/etth1_simplex_linear_test_evaluation/ETTH1_SIMPLEX_LINEAR_TEST_REPORT.md`
- `project_memory/experiments/2026-08-14_etth1_simplex_linear_test_evaluation.md`

Matched ETTh1/ETTh2 frozen-model table:

- `experiments/frozen_model_test_results/matched_etth1_etth2_results.csv`
- `experiments/frozen_model_test_results/MATCHED_ETTH1_ETTH2_RESULTS.md`
- `project_memory/experiments/2026-08-13_matched_etth1_etth2_frozen_results.md`

Valid ETTh2 matched rows:

- MLP residual corrector: test MAE/MSE `0.297254` / `0.218303`.
- Full adaptive / horizon-variable / duplicate-disabled DLinear-only / duplicate-disabled ModernTCN-only: test MAE/MSE `0.297808` / `0.218612`.
- Ridge residual corrector: test MAE/MSE `0.297313` / `0.218187`.
- Chronological EMA hybrid: test MAE/MSE `0.301689` / `0.222371`.
- Oracle prototype residual: test MAE/MSE `0.301185` / `0.222719`.
- Fixed-three core: test MAE/MSE `0.304642` / `0.225185`.
- Dynamic fixed-three seed7: test MAE/MSE `0.297398` / `0.218294`.
- Best single `DLinear`: test MAE/MSE `0.301708` / `0.222694`.


## Locked ETTh1-Config ETTh2 Replication

ADDITIONAL AFTER-FINAL-TEST REPLICATION:

ETTh2 artifacts now exist for the previously ETTh1-only MLP residual, ridge residual, oracle prototype residual, and dynamic fixed-three seed7 rows. These were fit on ETTh2 router-train only and evaluated after writing a manifest, but they were run after the earlier final ETTh2 test evaluation. Do not treat them as pre-test preregistered final competitors.

Artifacts:

- `experiments/locked_etth1_config_etth2_replication/final_report.json`
- `experiments/locked_etth1_config_etth2_replication/LOCKED_ETTH1_CONFIG_ETTH2_REPLICATION_REPORT.md`
- `experiments/frozen_model_test_results/matched_etth1_etth2_results.csv`


## ETTh2 Validation-Tuned Missing Methods

ADDITIONAL VALIDATION-TUNED RESULT:

Small ETTh2 validation-tuned sweeps now exist for MLP residual, ridge residual, oracle prototype residual, and dynamic fixed-three. These use ETTh2 router-validation for selection and are labeled `etth2_validation_tuned`; they are not pre-test preregistered or untouched-test confirmation.

Artifacts:

- `experiments/etth2_validation_tuned_missing_methods/final_report.json`
- `experiments/etth2_validation_tuned_missing_methods/ETTH2_VALIDATION_TUNED_MISSING_METHODS_REPORT.md`

## Final Pre-Test Freeze

CONFIRMED RESULT:

`experiments/final_test_freeze/FINAL_MODEL_FREEZE.json` is the preregistered freeze snapshot that was used for final test evaluation. At freeze time, no test cache was loaded and no test metrics were seen.

Frozen ETTh1 model:

- core: `PatchTST+iTransformer+TimesNet`
- core selection: router-train only
- model: `hybrid_chrono_hvema_lowrank1_decay0.95_temp0.1_alpha0.75`
- specialists: `DLinear+ModernTCN`
- specialist config: `both_variable_decay0.95_cap0.1_marginbp200_warm96`
- final frozen development result: MAE `0.363112`, MSE `0.306057`

Frozen ETTh2 model:

- core: `DLinear+PatchTST+ModernTCN`
- core selection: router-train only
- full frozen adaptive validation result: MAE `0.276832`, MSE `0.167280`
- validation-selected `DLinear+ModernTCN` remains only a reference baseline, not the primary frozen model

No further validation or test tuning is allowed after this freeze and final evaluation unless the freeze is explicitly superseded and documented as a new research program.

## Dataset And Splits

CONFIRMED RESULT:

- Dataset: ETTh1, `14400` timestamps, `7` variables.
- Input length: `96`.
- Forecast horizon: `12`.
- Split plan from `results/router_summary/costarts_walkforward/split_plan.json`:
  - `block_a`: `0-20%`, indices `0..2880`, valid starts `0..2772`, `2773` windows.
  - `block_b`: `20-40%`, indices `2880..5760`, valid starts `2880..5652`, `2773` windows.
  - `block_c`: `40-60%`, indices `5760..8640`, valid starts `5760..8532`, `2773` windows.
  - `validation`: `60-80%`, indices `8640..11520`, valid starts `8640..11412`, `2773` windows.
  - `test`: `80-100%`, indices `11520..14400`, valid starts `11520..14292`, `2773` windows.
- Router train cache: `cache/costarts_walkforward/router_train_20_60_cache.pt`, `5546` windows from block B and C OOS expert predictions.
- Router validation cache: `cache/costarts_walkforward/router_val_60_80_cache.pt`, `2773` windows.
- Test was evaluated once after explicit authorization on 2026-08-13. It must not be reused for tuning.

## Expert Models

CONFIRMED RESULT:

Frozen expert pool in the walk-forward cache:

- `DLinear`
- `PatchTST`
- `iTransformer`
- `TimesNet`
- `ModernTCN`

Primary strong fixed-3 subset used in most successful experiments:

- `PatchTST`
- `iTransformer`
- `TimesNet`

## Validation Baselines

CONFIRMED RESULT:

From `results/router_summary/costarts_walkforward/fixed_ensembles/summary.json`:

| Method | Validation MAE | Validation MSE |
|---|---:|---:|
| Best single: `iTransformer` | `0.376550` | `0.322095` |
| Best fixed 2: `PatchTST+iTransformer` | `0.370154` | `0.314509` |
| Best fixed 3: `PatchTST+iTransformer+TimesNet` | `0.367265` | `0.310530` |
| Best fixed 4: `DLinear+PatchTST+iTransformer+TimesNet` | `0.368216` | `0.310938` |
| Fixed all 5 | `0.371099` | `0.311582` |

From `results/router_summary/costarts_walkforward/fixed3_dynamic_weighting_5seed/summary.json`:

| Method | Validation MAE | Validation MSE |
|---|---:|---:|
| Dynamic fixed-3 weighting, 5 seeds | `0.366342 +/- 0.000223` | `0.309214 +/- 0.000237` |

## Current Best Adaptive Result

CONFIRMED RESULT:

Best current validation-only result for the active structurally even implementation is the equal-static capped expanded-pool specialist layer:

| Method | Validation MAE | Validation MSE | Evidence |
|---|---:|---:|---|
| equal-static `expanded_both` over `hybrid_chrono_hvema_lowrank1_decay0.95_temp0.1_alpha0.75` | `0.363100` | `0.306026` | `experiments/train_selected_core_etth1_equal_static/final_report.json` |

This result removes the old `OLD_FIXED3` static neural-prior exception. It is validation-only and was produced after final test results had already been seen, so it does not supersede the preregistered final-test result.

Historical pre-test development result:

| Method | Validation MAE | Validation MSE | Evidence |
|---|---:|---:|---|
| `expanded_both` over `hybrid_chrono_hvema_lowrank1_decay0.95_temp0.1_alpha0.75` | `0.363112 +/- 0.000013` | `0.306057 +/- 0.000016` | `experiments/expanded_expert_pool_costar/final_report.json` |

It beats the previous horizon-variable best `0.363642` by `0.000529` MAE over 5 seeds with aggregate paired bootstrap CI `[-0.000557, -0.000502]`.

It also beats the prior ridge residual best `0.363301` by `0.000189` MAE with paired CI `[-0.000233, -0.000143]`.

It does not reach the current target `0.3619`.

Previous best:

| Method | Validation MAE | Validation MSE | Evidence |
|---|---:|---:|---|
| `experiment2_ridge` over `hybrid_chrono_hvema_lowrank1_decay0.95_temp0.1_alpha0.75` | `0.363301 +/- 0.000015` | `0.306286 +/- 0.000017` | `experiments/residual_correction_costar/final_report.json` |
| `hybrid_chrono_hvema_lowrank1_decay0.95_temp0.1_alpha0.75` | `0.363642 +/- 0.000014` | `0.306712 +/- 0.000016` | `experiments/horizon_variable_adaptive_costar/final_report.json` |

## Important Oracle Results

CONFIRMED RESULT:

- Small-menu oracle over selected fixed subsets: MAE `0.347039`; fixed-3 gap `0.020226`. Evidence: `results/router_summary/costarts_walkforward/subset_menu_router_seed7/summary.json`.
- Oracle-weight/prototype direction showed learnable headroom but direct train-to-validation utility prediction generalized poorly. Evidence: `experiments/oracle_weight_tournament/predictability_diagnostic/summary.json`.
- Oracle predictability diagnostic:
  - Train oracle-weight R2: `+0.2810`.
  - Validation oracle-weight R2: `-0.2898`.
  - Validation prototype accuracy: `8.84%`.
  - Validation top-1 oracle expert accuracy: `32.13%`.

## Current Bottleneck

CONFIRMED RESULT:

Static/history-only prediction of per-window expert utility does not generalize across the router-train to router-validation shift. Chronological adaptation works better than trying to infer every window's oracle utility from history alone.

WORKING HYPOTHESIS:

Expert usefulness changes with time, horizon, and variable. Causal recent-performance statistics and low-rank horizon x variable specialization are currently more valuable than larger routers or new ranking losses.

## Latest Experiment

CONFIRMED RESULT:

The latest requested phase sequence closed residual covariance, adaptive forgetting, and available-cache multi-dataset replication.

Result:

- Residual-covariance weighting selected `diagonal_variance_hv_decay0.99_ridge0.0001_sd1_sg0_bias0_alpha0.5_warm96` on router-train folds, but validation was `0.363649 +/- 0.000012`, slightly worse than `0.363642`, with CI crossing zero and a large horizon `11` / variable `4` regression. Do not promote.
- Regime-adaptive forgetting selected `zscore_slow0.99_fast0.95_thr2.5_delta0_reset0_cool24_boost24` on router-train folds, but validation worsened to `0.364346 +/- 0.000015`; even the oracle change-point diagnostic worsened to `0.364015`. Do not promote.
- Phase 4 found existing non-test expert caches only for ETTh2. A follow-up canonical protocol audit established that ETTh2 router-summary comparisons must use raw/original-scale MAE with `std=ones`, no inverse transform, and the same 613 validation starts `10800..11412`. Under that canonical protocol, best single is `DLinear` MAE `0.280957`, best fixed baseline is `DLinear+ModernTCN` MAE `0.275229`, and the earlier `0.093369` specialist result is only a normalized diagnostic metric, not a canonical router-summary number.

Current best remains:

- ETTh1 `expanded_both` over `hybrid_chrono_hvema_lowrank1_decay0.95_temp0.1_alpha0.75`
- MAE `0.363112 +/- 0.000013`, MSE `0.306057 +/- 0.000016`

Train-selected core audit:

- `experiments/train_selected_core_etth1/run_train_selected_core_eval.py` selected the three core experts using router-train only before loading validation.
- Router-train OOF ranking selected `PatchTST+iTransformer+TimesNet` with OOF MAE `0.345568`, OOF MSE `0.260786`.
- Because train-only selection chose the same core, the current best validation result remains exactly `0.363112 +/- 0.000013`.
- The current-best architecture improves over train-selected fixed-3 `0.367265` by `0.004153` MAE with paired CI `[-0.004460, -0.003856]`.

Previous grokking diagnostic:

`experiments/grokking_diagnostic_costar/run_grokking_diagnostic.py` tested whether the strongest trainable neural fixed-three router (`final_phase2_protores_lam0.01_k16_scale0.3_rw0.001`) shows grokking under 10x longer training.

Result: no delayed sustained improvement; no credible grokking.

Previous residual-correction experiment:

`experiments/residual_correction_costar/run_residual_correction_experiments.py` tested:

- Causal residual-bias correction over global, horizon, variable, and horizon x variable structures.
- Conservative ridge residual correction using frozen baseline/expert forecasts, disagreement, causal residual stats, horizon/variable identity, and causal multi-scale history summaries.
- Tiny two-layer MLP because ridge showed useful router-train fold signal.

Winner: ridge residual correction, config `ridge1_alpha0.1_clip0.25_full`.

Important caveats:

- Bias correction improved validation slightly but had no router-train fold wins, so it is weak evidence.
- MLP improved validation but was less seed-stable than ridge and had larger local variable-4 regressions.

## Next Recommended Experiment

WORKING HYPOTHESIS:

Residual covariance and regime-change forgetting did not transfer from router-train folds to ETTh1 validation. The most reliable remaining direction is to test whether the conservative ridge residual signal is still additive on top of the current `expanded_both` specialist layer.

Recommended next:

1. Freeze `expanded_both` as the validation-best specialist layer.
2. Test whether the ridge residual correction still adds signal on top of `expanded_both`.
3. Keep optional DLinear/ModernTCN weights capped and nonnegative.
4. Compare to `0.363112` with 5 seeds and paired bootstrap.

UNTESTED IDEA:

Build complete equivalent caches/artifacts for ETTh2/ETTm1/ETTm2/Weather/Electricity if full primary-model replication is required; the current Phase 4 was limited by missing non-test expert caches and missing static-winner artifacts.

ETTh2 CANONICAL BASELINE:

Use `experiments/etth2_canonical_protocol/final_report.json` and `canonical_raw_results.csv` for ETTh2 comparisons. The canonical raw/original-scale protocol exactly reproduces the old ETTh2 best-by-size summary:

- validation cache: `cache/costarts_fresh/ETTh2_96_12/router_val_cache.pt`
- validation starts: `10800..11412`, `613` windows
- horizon `12`, variables `7`
- MAE implementation: `sample_mae`
- normalization: `std=ones`
- inverse transform: none
- best single: `DLinear`, MAE `0.280957`, MSE `0.171493`
- best fixed: `DLinear+ModernTCN`, MAE `0.275229`, MSE `0.165345`

ETTh2 TRAIN-SELECTED CORE AUDIT:

`experiments/etth2_train_selected_core/run_etth2_train_selected_core_eval.py` selected a core using ETTh2 router-train only:

- selected core: `DLinear+PatchTST+ModernTCN`
- router-train OOF MAE/MSE: `0.284658` / `0.181718`
- train-selected fixed-3 validation MAE/MSE: `0.280878` / `0.171933`
- frozen full current-best architecture validation MAE/MSE: `0.276832` / `0.167280`
- canonical best fixed-2 remains better: `DLinear+ModernTCN`, MAE `0.275229`

Conclusion: the architecture improves over its own train-selected fixed-3 on ETTh2, but it does not beat the simpler fixed-2 baseline and should not be promoted for ETTh2.

ETTh2 VARIABLE-SIZE CORE AUDIT:

`experiments/etth2_train_selected_variable_core/run_etth2_train_selected_variable_core_eval.py` selected subset size and membership using ETTh2 router-train only over all 31 non-empty expert subsets:

- selected core: `DLinear`
- router-train OOF MAE/MSE: `0.283464` / `0.182616`
- selected core validation MAE/MSE: `0.280957` / `0.171493`
- frozen full current-best architecture on selected core validation MAE/MSE: `0.280470` / `0.170973`
- previous full fixed-3 model remains better: MAE `0.276832`
- canonical best fixed-2 remains better: `DLinear+ModernTCN`, MAE `0.275229`
- fold-best subset sizes were not stable: `3, 1, 1, 3`

Conclusion: router-train did independently choose a smaller ETTh2 core, but the selected single-expert core did not validate better than fixed-3 or fixed-2 references. Do not promote.

POST-TEST POOLED CORE SENSITIVITY:

`experiments/pooled_router_train_core/run_pooled_router_train_core.py` tested the simpler pooled router-train core selection rule after final test results had already been viewed, so it is not a clean preregistered replacement.

- ETTh1 pooled router-train selection chose the same core as fold selection: `PatchTST+iTransformer+TimesNet`.
- ETTh1 pooled full adaptive test MAE/MSE: `0.326393` / `0.267506`, effectively unchanged from the existing full adaptive result `0.326395` / `0.267509`.
- ETTh2 pooled router-train selection chose `DLinear+TimesNet+ModernTCN`, while the earlier fold-selected core was `DLinear+PatchTST+ModernTCN`.
- ETTh2 pooled full adaptive validation MAE/MSE: `0.275602` / `0.166460`.
- ETTh2 pooled full adaptive test MAE/MSE: `0.295829` / `0.219681`, better than the existing fold-selected full adaptive test MAE `0.297808`.

Conclusion: pooled selection supports the user's point that chronological fold ranking can be redundant for frozen equal-average core selection. ETTh2 shows a real sensitivity to the selection rule, but because test was already seen before this experiment, this should be treated as post-test sensitivity evidence rather than a new final model claim.

POST-TEST POOLED RESIDUAL-CORRECTOR SENSITIVITY:

`experiments/pooled_router_train_residual_correctors/run_pooled_router_train_residual_correctors.py` applied the same pooled-selection rule to Ridge and MLP residual correctors: select by MAE/MSE over all router-train windows together, with no folds, then freeze before validation/test.

- ETTh1 pooled-selected Ridge validation/test MAE: `0.363088` / `0.328022`.
- ETTh1 pooled-selected MLP validation/test MAE: `0.364111` / `0.325964`.
- ETTh2 pooled-selected Ridge validation/test MAE: `0.275036` / `0.296787`.
- ETTh2 pooled-selected MLP validation/test MAE: `0.275975` / `0.297427`.

Conclusion: ETTh1 pooled-selected MLP is the lowest audited ETTh1 test MAE, but it was run after final test results were already viewed and does not supersede the preregistered frozen result. ETTh1 pooled-selected Ridge overfits and fails on test. ETTh2 pooled-selected Ridge matches the validation-tuned Ridge, while ETTh2 pooled-selected MLP is worse than the validation-tuned MLP.
