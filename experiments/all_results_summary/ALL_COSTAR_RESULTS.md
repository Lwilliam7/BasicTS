# COSTAR Results Summary

Created: 2026-08-13

Updated: 2026-08-14

This file consolidates frozen-model test results from the original confirmatory final frozen test evaluation and additional frozen-model evaluations performed later. Rows marked `clean_preregistered` are the official frozen results from the pre-test freeze artifacts. Rows marked `pre_test_frozen` were trained, selected, configured, and frozen without test-data feedback; they are additional frozen-model evaluations unless also marked `clean_preregistered`.

Machine-readable version: `experiments/all_results_summary/all_costar_results.csv`

Coverage note: this file now includes official final frozen test rows, additional final-test audit rows, and ETTh2 validation-only COSTAR/fixed-subset records from the canonical protocol, train-selected core audits, sequential COSTAR, pair selector, pair-potential diagnostics, and limited normalized replication. Validation-only rows have blank test columns in the CSV.

Matched ETTh1/ETTh2 table:

- `experiments/frozen_model_test_results/matched_etth1_etth2_results.csv`
- `experiments/frozen_model_test_results/MATCHED_ETTH1_ETTH2_RESULTS.md`

ETTh2 is filled in that matched table where a valid analogue exists. Four formerly ETTh1-only methods now have ETTh2 locked-config replication rows labeled `locked_etth1_config_etth2_replication`; those rows were generated after the earlier final ETTh2 test evaluation and should not be treated as pre-test preregistered final competitors.

## Official Final Frozen Test Results

| Dataset | Method | Expert set | Test MAE | Test MSE | Val MAE | Val MSE | Status |
|---|---|---|---:|---:|---:|---:|---|
| ETTh1 | Best single expert | `iTransformer` | `0.339080` | `0.278551` | `0.376550` | `0.322095` | clean preregistered reference |
| ETTh1 | Train-selected fixed core | `PatchTST+iTransformer+TimesNet` | `0.327128` | `0.266583` | `0.367265` | `0.310530` | clean preregistered reference |
| ETTh1 | Full frozen adaptive model | `PatchTST+iTransformer+TimesNet+DLinear+ModernTCN` | `0.326395` | `0.267509` | `0.363112` | `0.306057` | clean final model |
| ETTh2 | Best single expert | `DLinear` | `0.301708` | `0.222694` | `0.280957` | `0.171493` | clean preregistered reference |
| ETTh2 | Train-selected fixed core | `DLinear+PatchTST+ModernTCN` | `0.304642` | `0.225185` | `0.280878` | `0.171933` | clean preregistered reference |
| ETTh2 | Full frozen adaptive model | `DLinear+PatchTST+ModernTCN` | `0.297808` | `0.218612` | `0.276832` | `0.167280` | clean final model |
| ETTh2 | `DLinear+ModernTCN` | `DLinear+ModernTCN` | `0.299263` | `0.221853` | `0.275229` | `0.165345` | validation-selected reference only |

## ETTh1 Additional Final-Test Audit

| Rank | Method | Test MAE | Test MSE | Val MAE | Diff vs fixed core | Status |
|---:|---|---:|---:|---:|---:|---|
| 1 | MLP residual corrector | `0.326047` | `0.267322` | `0.363318` | `-0.001081` | additional frozen-model result |
| 2 | Expanded both final frozen | `0.326393` | `0.267506` | `0.363112` | `-0.000735` | official final model replayed |
| 3 | Expanded DLinear only | `0.326437` | `0.267593` | `0.363510` | `-0.000691` | additional frozen-model result |
| 4 | Ridge residual corrector | `0.326448` | `0.267452` | `0.363301` | `-0.000680` | additional frozen-model result |
| 5 | Expanded ModernTCN only | `0.326468` | `0.267591` | `0.363435` | `-0.000660` | additional frozen-model result |
| 6 | Horizon-variable hybrid | `0.326493` | `0.267638` | `0.363642` | `-0.000635` | additional frozen-model result |
| 7 | Chronological EMA hybrid | `0.326548` | `0.266643` | `0.365534` | `-0.000580` | additional frozen-model result |
| 8 | Oracle prototype residual | `0.326829` | `0.267364` | `0.366028` | `-0.000299` | additional frozen-model result |
| 9 | Nonnegative simplex linear average | `0.326926` | `0.267713` | `0.366483` | `-0.000203` | after-final-test audit |
| 10 | Fixed core equal | `0.327128` | `0.266583` | `0.367265` | `0.000000` | anchor |
| 11 | Dynamic fixed3 seed7 | `0.329249` | `0.272063` | `0.365985` | `+0.002121` | additional frozen-model result |

## ETTh2 Frozen-Model Top COSTAR Results

| Rank | Method | Expert set | Test MAE | Test MSE | Val MAE | Diff vs DLinear | Status |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | Nonnegative simplex linear average | `DLinear+PatchTST+iTransformer+TimesNet+ModernTCN` | `0.297120` | `0.218587` | `0.274755` | `-0.004588` | after-final-test audit |
| 2 | Full adaptive train-selected fixed-3 final frozen | `DLinear+PatchTST+ModernTCN` | `0.297808` | `0.218612` | `0.276832` | `-0.003899` | official final model |
| 3 | Ridge linear stacker | `DLinear+PatchTST+iTransformer+TimesNet+ModernTCN` | `0.298382` | `0.218201` | `0.276702` | `-0.003325` | after-final-test audit |
| 4 | Fixed-2 `DLinear+TimesNet` reference | `DLinear+TimesNet` | `0.298398` | `0.221926` | `0.277652` | `-0.003310` | additional frozen-model reference |
| 5 | Fixed-3 `DLinear+TimesNet+ModernTCN` reference | `DLinear+TimesNet+ModernTCN` | `0.299169` | `0.223927` | `0.276644` | `-0.002539` | validation-selected reference only |
| 6 | Fixed-2 `DLinear+ModernTCN` reference | `DLinear+ModernTCN` | `0.299263` | `0.221853` | `0.275229` | `-0.002445` | validation-selected reference only |
| 7 | Full adaptive variable-size `DLinear` core | `DLinear` | `0.301093` | `0.222139` | `0.280470` | `-0.000615` | additional frozen-model result |
| 8 | Single `DLinear` | `DLinear` | `0.301708` | `0.222694` | `0.280957` | `0.000000` | anchor |
| 9 | Train-selected fixed-3 core | `DLinear+PatchTST+ModernTCN` | `0.304642` | `0.225185` | `0.280878` | `+0.002935` | clean core reference |

## ETTh2 Locked ETTh1-Config Replication

These rows port the ETTh1 method configs to ETTh2 with core `DLinear+PatchTST+ModernTCN`, fit on ETTh2 router-train only, write a manifest, then evaluate ETTh2 test once. They were run after the earlier final ETTh2 test evaluation.

| Method | Test MAE | Test MSE | Val MAE | Diff vs full adaptive test | Status |
|---|---:|---:|---:|---:|---|
| MLP residual corrector | `0.297254` | `0.218303` | `0.276129` | `-0.000554` | locked ETTh1-config ETTh2 replication |
| Ridge residual corrector | `0.297313` | `0.218187` | `0.276038` | `-0.000495` | locked ETTh1-config ETTh2 replication |
| Dynamic fixed-three, seed 7 | `0.297398` | `0.218294` | `0.275379` | `-0.000410` | locked ETTh1-config ETTh2 replication |
| Oracle prototype residual | `0.301185` | `0.222719` | `0.276404` | `+0.003377` | locked ETTh1-config ETTh2 replication |

## ETTh2 Validation-Tuned Missing Methods

These rows use ETTh2 router-validation for hyperparameter and checkpoint selection. They are not preregistered and are not untouched-test confirmation.

| Method | Test MAE | Test MSE | Val MAE | Diff vs full adaptive test | Diff vs locked counterpart |
|---|---:|---:|---:|---:|---:|
| Ridge residual corrector | `0.296787` | `0.217713` | `0.275036` | `-0.001021` | `-0.000526` |
| MLP residual corrector | `0.297041` | `0.218149` | `0.275643` | `-0.000767` | `-0.000213` |
| Oracle prototype residual | `0.298475` | `0.219894` | `0.274829` | `+0.000667` | `-0.002710` |
| Dynamic fixed-three, seed 7 | `0.298079` | `0.219521` | `0.274746` | `+0.000271` | `+0.000681` |

## Pooled Router-Train Residual Correctors

These rows use the same pooled definition as `pooled_router_train_core`: select the configuration by MAE/MSE over all router-train windows together, with no chronological folds. They are labeled `after_final_test_audit`.

| Dataset | Method | Selected config | Train MAE | Val MAE | Test MAE | Test MSE | Diff vs existing corrector | Diff vs full adaptive |
|---|---|---|---:|---:|---:|---:|---:|---:|
| ETTh1 | Ridge residual corrector | `ridge1_alpha0.5_clip0.5_full` | `0.337319` | `0.363088` | `0.328022` | `0.267930` | `+0.001574` | `+0.001629` |
| ETTh1 | MLP residual corrector | `mlp_alpha0p1_clip_multiple0p5_epochs40_hidden64_lr0p0003_patience6_weight_decay0p01` | `0.334217` | `0.364111` | `0.325964` | `0.266587` | `-0.000083` | `-0.000429` |
| ETTh2 | Ridge residual corrector | `ridge1_alpha0.25_clip0.25_full` | `0.277668` | `0.275036` | `0.296787` | `0.217713` | `+0.000000` | `-0.001021` |
| ETTh2 | MLP residual corrector | `mlp_alpha0p1_clip_multiple0p5_epochs40_hidden64_lr0p0003_patience6_weight_decay0p01` | `0.275051` | `0.275975` | `0.297427` | `0.218405` | `+0.000386` | `-0.000381` |

## Sequential COSTAR Test Audit

These rows evaluate the existing frozen `utility_pairwise_weighted` Sequential COSTAR checkpoints after explicit user authorization. They are after-final-test audit rows, not preregistered final competitors.

| Dataset | Test MAE | Test MSE | Val MAE | Avg queries | Diff vs final frozen adaptive |
|---|---:|---:|---:|---:|---:|
| ETTh1 | `0.330832 +/- 0.005462` | `0.271398 +/- 0.005434` | `0.368074` | `3.776` | `+0.004437` |
| ETTh2 | `0.300576 +/- 0.000032` | `0.222171 +/- 0.000069` | `0.277681` | `3.998` | `+0.002768` |

## Matched ETTh1 / ETTh2 Frozen-Model Table

| Method | ETTh1 Test MAE | ETTh1 Test MSE | ETTh1 Val MAE | ETTh2 Test MAE | ETTh2 Test MSE | ETTh2 Val MAE | ETTh2 Status |
|---|---:|---:|---:|---:|---:|---:|---|
| MLP residual corrector | `0.326047` | `0.267322` | `0.363318` | `0.297254` | `0.218303` | `0.276129` | `locked_etth1_config_etth2_replication` |
| Full adaptive model | `0.326395` | `0.267509` | `0.363112` | `0.297808` | `0.218612` | `0.276832` | `pre_test_frozen` |
| Expanded DLinear only | `0.326437` | `0.267593` | `0.363510` | `0.297808` | `0.218612` | `0.276832` | `pre_test_frozen` |
| Ridge residual corrector | `0.326448` | `0.267452` | `0.363301` | `0.297313` | `0.218187` | `0.276038` | `locked_etth1_config_etth2_replication` |
| Expanded ModernTCN only | `0.326468` | `0.267591` | `0.363435` | `0.297808` | `0.218612` | `0.276832` | `pre_test_frozen` |
| Horizon-variable hybrid | `0.326493` | `0.267638` | `0.363642` | `0.297808` | `0.218612` | `0.276832` | `pre_test_frozen` |
| Chronological EMA hybrid | `0.326548` | `0.266643` | `0.365534` | `0.301689` | `0.222371` | `0.278806` | `pre_test_frozen` |
| Oracle prototype residual | `0.326829` | `0.267364` | `0.366028` | `0.301185` | `0.222719` | `0.276404` | `locked_etth1_config_etth2_replication` |
| Fixed-three core | `0.327128` | `0.266583` | `0.367265` | `0.304642` | `0.225185` | `0.280878` | `pre_test_frozen` |
| Dynamic fixed-three, seed 7 | `0.329249` | `0.272063` | `0.365985` | `0.297398` | `0.218294` | `0.275379` | `locked_etth1_config_etth2_replication` |
| Best single | `0.339080` | `0.278551` | `0.376550` | `0.301708` | `0.222694` | `0.280957` | `pre_test_frozen` |

For ETTh2, `Expanded DLinear only` and `Expanded ModernTCN only` are not distinct predictions under the train-selected core because `DLinear` and `ModernTCN` are already core experts; duplicate specialist branches are disabled.

## ETTh2 Validation-Only Coverage

The CSV also includes the broader ETTh2 validation record:

- all 31 canonical raw fixed/single expert subsets from `experiments/etth2_canonical_protocol/canonical_raw_results.csv`;
- train-selected fixed-3 and variable-size core audits;
- sequential COSTAR utility-pairwise weighted, 5 seeds;
- pair selector aggregate and per-seed rows;
- pair-potential diagnostics including oracle pair/expert, nonnegative simplex average, and ridge stacker;
- limited multidataset replication rows on the older normalized diagnostic scale.

The pair-potential `nonnegative_simplex_linear_average` and `ridge_linear_stacker` now also have ETTh2 test rows in `experiments/etth2_pair_potential_test_evaluation/test_results.csv`; their older validation-only rows are retained as provenance for the original validation report.

Important scale note: the limited replication rows are marked `normalized_diagnostic`; do not compare those `0.09` MAE values directly to canonical raw ETTh2 values around `0.27-0.30`.

## Bottom Line

Official clean final test winners:

- ETTh1: full frozen adaptive model, test MAE/MSE `0.326395` / `0.267509`.
- ETTh2: full frozen adaptive model, test MAE/MSE `0.297808` / `0.218612`.

Frozen-model test result notes:

- ETTh1 frozen-model MLP residual corrector had the lowest audited test MAE, `0.326047`, but it is an additional frozen-model result not listed in the original confirmatory freeze artifact.
- ETTh1 pooled-router-train selected MLP residual corrector scored the lowest audited ETTh1 test MAE, `0.325964`, but it is an after-final-test sensitivity result and does not supersede preregistered rows.
- ETTh1 nonnegative simplex linear average scored `0.326926`, slightly better than the fixed-three core but worse than the full adaptive and stronger residual/specialist audit rows.
- ETTh2 nonnegative simplex linear average had the lowest audited ETTh2 test MAE, `0.297120`, but it was evaluated after the original final ETTh2 test evaluation and does not supersede the preregistered frozen ETTh2 result.
- ETTh2 locked ETTh1-config replications found MLP, ridge, and dynamic rows slightly below the earlier full adaptive test MAE, but these were run after the earlier final ETTh2 test evaluation and do not supersede the preregistered frozen ETTh2 result.
