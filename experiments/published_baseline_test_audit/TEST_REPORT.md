# After-Final-Test Audit: Six Previously Untested Published-Baseline Methods

Label: `after_final_test_audit`.

ETTh1 and ETTh2 test results were already seen elsewhere in this project before this audit ran. This is not a clean untouched final-test claim; it is a frozen, no-further-tuning evaluation of six methods that had validation results but no prior test evaluation.

All six configurations are read verbatim from `experiments/published_baseline_comparisons/{ETTh1,ETTh2}/frozen_config_before_validation.json`, written before validation was ever loaded. No hyperparameter, expert-set, or method choice was changed after loading test. Frozen COSTAR and Online COSTAR were **not** re-tuned or re-selected; their rows are reference rows read verbatim from existing authoritative artifacts.

## Main Results Table

| Method | ETTh1 Val MAE | ETTh1 Test MAE | ETTh1 Test MSE | ETTh2 Val MAE | ETTh2 Test MAE | ETTh2 Test MSE |
|---|---:|---:|---:|---:|---:|---:|
| Equal all-5 ensemble | 0.371099 | 0.332001 | 0.270050 | 0.300772 | 0.322330 | 0.249527 |
| Granger-Ramanathan | 0.382960 | 0.340765 | 0.289594 | 0.276704 | 0.298419 | 0.218160 |
| Bates-Granger | 0.368891 | 0.327848 | 0.267809 | 0.274915 | 0.296294 | 0.217423 |
| FAME routing adaptation to BasicTS frozen expert pool | 0.379212 | 0.331314 | 0.271990 | 0.277008 | 0.298372 | 0.220674 |
| TimeRouter routing-mechanism adaptation | 0.368234 | 0.328178 | 0.267896 | 0.283288 | 0.306324 | 0.228592 |
| OneNet-style frozen-expert adaptation | 0.370137 | 0.330721 | 0.272812 | 0.402666 | 0.407526 | 0.413704 |

Reference rows (existing, not re-run):

| Method | ETTh1 Test MAE | ETTh1 Test MSE | ETTh2 Test MAE | ETTh2 Test MSE |
|---|---:|---:|---:|---:|
| Frozen COSTAR (reference) | 0.327175 | 0.267094 | 0.300574 | 0.220499 |
| Online COSTAR (reference) | 0.326408 | 0.267378 | 0.297808 | 0.218612 |
| COSTAR train-selected fixed core (reference) | 0.327128 | 0.266583 | 0.304642 | 0.225185 |
| Best single expert (reference) | 0.339080 | 0.278551 | 0.301708 | 0.222694 |

## ETTh1 Full Ranking (audited methods + reference rows)

| Rank | Method | Test MAE | Test MSE | Kind |
|---:|---|---:|---:|---|
| 1 | Online COSTAR (reference) | 0.326408 | 0.267378 | reference |
| 2 | COSTAR train-selected fixed core (reference) | 0.327128 | 0.266583 | reference |
| 3 | Frozen COSTAR (reference) | 0.327175 | 0.267094 | reference |
| 4 | Bates-Granger | 0.327848 | 0.267809 | audited |
| 5 | TimeRouter adaptation | 0.328178 | 0.267896 | audited |
| 6 | OneNet-style frozen-expert adaptation | 0.330721 | 0.272812 | audited |
| 7 | FAME adaptation | 0.331314 | 0.271990 | audited |
| 8 | Equal all-5 ensemble | 0.332001 | 0.270050 | audited |
| 9 | Best single expert (reference) | 0.339080 | 0.278551 | reference |
| 10 | Granger-Ramanathan | 0.340765 | 0.289594 | audited |

## ETTh2 Full Ranking (audited methods + reference rows)

| Rank | Method | Test MAE | Test MSE | Kind |
|---:|---|---:|---:|---|
| 1 | Bates-Granger | 0.296294 | 0.217423 | audited |
| 2 | Online COSTAR (reference) | 0.297808 | 0.218612 | reference |
| 3 | FAME adaptation | 0.298372 | 0.220674 | audited |
| 4 | Granger-Ramanathan | 0.298419 | 0.218160 | audited |
| 5 | Frozen COSTAR (reference) | 0.300574 | 0.220499 | reference |
| 6 | Best single expert (reference) | 0.301708 | 0.222694 | reference |
| 7 | COSTAR train-selected fixed core (reference) | 0.304642 | 0.225185 | reference |
| 8 | TimeRouter adaptation | 0.306324 | 0.228592 | audited |
| 9 | Equal all-5 ensemble | 0.322330 | 0.249527 | audited |
| 10 | OneNet-style frozen-expert adaptation | 0.407526 | 0.413704 | audited |

## Leakage And Causality Checks

| Check | Passed | Max abs diff |
|---|---|---:|
| ETTh1 Granger-Ramanathan target-replacement invariance | True | 0.0000000000 |
| ETTh1 Bates-Granger target-replacement invariance | True | 0.0000000000 |
| ETTh1 FAME adaptation target-replacement invariance | True | 0.0000000000 |
| ETTh1 TimeRouter adaptation target-replacement invariance | True | 0.0000000000 |
| ETTh2 Granger-Ramanathan target-replacement invariance | True | 0.0000000000 |
| ETTh2 Bates-Granger target-replacement invariance | True | 0.0000000000 |
| ETTh2 FAME adaptation target-replacement invariance | True | 0.0000000000 |
| ETTh2 TimeRouter adaptation target-replacement invariance | True | 0.0000000000 |
| ETTh1 OneNet future-target perturbation causality | True | 0.0000000000 |
| ETTh2 OneNet future-target perturbation causality | True | 0.0000000000 |

## Cache Provenance

- ETTh1 test cache: `experiments\final_test_evaluation\generated\caches\ETTh1\test_80_100_cache.pt`, sha256 `b7e41bea3a321183f09015d39c82107f7d31f279f929a77b2499ee0895b6b714`.
- ETTh2 test cache: `experiments\final_test_evaluation\generated\caches\ETTh2\locked_test_cache_v2.pt`, sha256 `c9aa614e40058d45b6bfde62fd9a2a4ff27064ae0dec7189ecaf3e2cae6a7ade`.
- Both caches: expert order `DLinear, PatchTST, iTransformer, TimesNet, ModernTCN`; horizon `12`; input length `96`; `2773` chronological windows; target masks all-observed at forecast time.
- Git commit: `8e86f0c5d9140ba44afb3c46fe69cf270b6b4317`.