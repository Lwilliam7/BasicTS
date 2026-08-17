# Research Decisions

Last updated: 2026-08-17

These are durable conclusions supported by repository outputs. Do not repeatedly rediscover them unless a new hypothesis materially changes the experiment.

## Test Split

Status: Evaluated once after explicit authorization

Evidence:

- Walk-forward split reserves ETTh1 `80-100%` for test: `results/router_summary/costarts_walkforward/split_plan.json`.
- Recent experiment final reports explicitly say no test data was used.
- Final freeze artifacts record `test_loaded=false` and `test_metrics_seen=false`.
- Final test evaluation was run on 2026-08-13 from the preregistered freeze artifacts.

Decision:

The ETTh1 and ETTh2 test splits have now been evaluated once. Do not use these test results for tuning, model selection, or additional iterative development.

Reason:

The validation set was used for iterative research. The final test evaluation is now the held-out endpoint, not another development signal.

## Final Frozen Test Evaluation Is Complete

Status: Permanent final-evaluation record

Evidence:

- `experiments/final_test_evaluation/FINAL_TEST_RESULTS.json`
- `experiments/final_test_evaluation/FINAL_TEST_REPORT.md`
- `project_memory/experiments/2026-08-13_final_frozen_test_evaluation.md`

Decision:

Treat the following as the final held-out test results for the preregistered COSTAR-TS freeze:

- ETTh1 full frozen adaptive: MAE `0.326395`, MSE `0.267509`.
- ETTh1 train-selected fixed core: MAE `0.327128`, MSE `0.266583`.
- ETTh2 full frozen adaptive: MAE `0.297808`, MSE `0.218612`.
- ETTh2 train-selected fixed core: MAE `0.304642`, MSE `0.225185`.
- ETTh2 validation-selected `DLinear+ModernTCN` reference: MAE `0.299263`, MSE `0.221853`.

Reason:

The frozen adaptive model's relative MAE improvement survived test for both datasets, but the test metrics are now seen and must not influence further tuning of this frozen family.

## frozen-model Test Audits Cannot Supersede The Frozen Final Model

Status: Permanent interpretation rule

Evidence:

- `experiments/frozen_model_test_results/TOP_COSTAR_TEST_RESULTS.json`
- `project_memory/experiments/2026-08-13_frozen_model_top_costar_test_results.md`
- `experiments/frozen_model_test_results/ETTH2_TOP_COSTAR_TEST_RESULTS.json`
- `project_memory/experiments/2026-08-13_frozen_model_etth2_top_costar_test_results.md`

Decision:

The frozen-model ETTh1 MLP residual corrector test MAE `0.326047` is hypothesis-generating only. It does not replace the preregistered final frozen adaptive model, whose final ETTh1 test MAE was `0.326395`.

The frozen-model ETTh2 audit did not find a stronger COSTAR method than the preregistered final frozen adaptive model. The ETTh2 final model remains MAE `0.297808`, ahead of the validation-selected `DLinear+ModernTCN` reference test MAE `0.299263`.

Reason:

The additional frozen-model evaluations were performed after the original confirmatory evaluation, but the listed models were trained, selected, configured, and frozen without test-data feedback. Do not describe every additional row as formally preregistered unless a preregistration artifact explicitly supports that claim.

## ETTh2 Pair-Potential Linear Ensembles Are Audit Rows

Status: Additional after-final-test audit

Evidence:

- `experiments/etth2_pair_potential_test_evaluation/ETTH2_PAIR_POTENTIAL_TEST_RESULTS.json`
- `project_memory/experiments/2026-08-14_etth2_pair_potential_test_evaluation.md`

Decision:

The ETTh2 `nonnegative_simplex_linear_average` all-five ensemble scored test MAE/MSE `0.297120` / `0.218587`, beating the official full adaptive ETTh2 test MAE by `0.000688`. The `ridge_linear_stacker` scored test MAE/MSE `0.298382` / `0.218201`, improving over single DLinear but not over the full adaptive model.

Treat both as after-final-test audit rows, not preregistered final competitors.

Reason:

These methods were train-fitted before test feedback, but the decision to test them came after the original final ETTh2 test evaluation. They are useful evidence that simple all-five ETTh2 linear ensembles can generalize, but they should not replace the official frozen model without a new held-out protocol.

## ETTh1 Simplex Linear Ensemble Is Only A Simple Baseline

Status: Additional after-final-test audit

Evidence:

- `experiments/etth1_simplex_linear_test_evaluation/ETTH1_SIMPLEX_LINEAR_TEST_RESULTS.json`
- `project_memory/experiments/2026-08-14_etth1_simplex_linear_test_evaluation.md`

Decision:

The ETTh1 `nonnegative_simplex_linear_average` all-five ensemble scored validation MAE/MSE `0.366483` / `0.308484` and test MAE/MSE `0.326926` / `0.267713`. It slightly beat the fixed-three core on test MAE by `0.000203`, but the paired bootstrap CI versus fixed core crossed zero. It was worse than the full adaptive ETTh1 test MAE by `0.000530`.

Treat it as a useful simple after-final-test baseline, not a promoted COSTAR model.

Reason:

The all-five simplex fit is much simpler than the adaptive routers and has decent validation/test behavior, but it does not beat the best ETTh1 audited methods and its edge over fixed-three is not statistically robust.

## Locked ETTh1-Config ETTh2 Replications Are Audit Rows

Status: Additional after-final-test replication

Evidence:

- `experiments/locked_etth1_config_etth2_replication/final_report.json`
- `experiments/frozen_model_test_results/matched_etth1_etth2_results.csv`

Decision:

Use the new ETTh2 MLP residual, ridge residual, oracle prototype residual, and dynamic fixed-three seed7 rows only as locked ETTh1-config ETTh2 replication audits. They were fit without ETTh2 validation tuning but were run after the earlier final ETTh2 test evaluation, so they do not supersede the final frozen ETTh2 result.

Reason:

This preserves the distinction between the original confirmatory frozen test evaluation and later matched-table completeness work.

## Final Models Are Frozen Before Test

Status: Permanent preregistration until explicitly superseded

Evidence:

- `experiments/final_test_freeze/ETTh1_frozen_model.json`
- `experiments/final_test_freeze/ETTh2_frozen_model.json`
- `experiments/final_test_freeze/FINAL_MODEL_FREEZE.json`
- `experiments/final_test_freeze/freeze_report.md`

Decision:

Freeze ETTh1 as `PatchTST+iTransformer+TimesNet` with `hybrid_chrono_hvema_lowrank1_decay0.95_temp0.1_alpha0.75` and `both_variable_decay0.95_cap0.1_marginbp200_warm96`, final development MAE/MSE `0.363112` / `0.306057`.

Freeze ETTh2 as `DLinear+PatchTST+ModernTCN` with the same frozen adaptive architecture from the clean ETTh2 train-selected-core experiment, validation MAE/MSE `0.276832` / `0.167280`.

Reason:

These are the final preregistered models selected before any test cache was loaded. No further validation tuning is allowed after this freeze.

## Fixed Ensembles Are Strong Baselines

Status: Confirmed

Evidence:

- Best fixed 3 equal-average subset is `PatchTST+iTransformer+TimesNet`, MAE `0.367265`.
- Many sequential router variants failed to beat this.

Decision:

Every new router or adaptive method must compare against fixed-3 and the current best adaptive method, not only single experts.

Reason:

Fixed ensembles close much of the gap and expose when routers add complexity without value.

## Static COSTAR Prior Is Equal Across Triples

Status: Implemented as structural cleanup

Evidence:

- `experiments/train_selected_core_etth1/run_train_selected_core_eval.py` now sets `static_weights = torch.full((num_windows, 3), 1.0 / 3.0)` for every selected triple.
- `experiments/frozen_costar/run_frozen_costar_validation.py` no longer imports or loads the ETTh1 static neural-router checkpoint.
- ETTh1 equal-static full adaptive validation MAE/MSE: `0.363100` / `0.306026`, saved in `experiments/train_selected_core_etth1_equal_static/final_report.json`.

Decision:

Use equal static weights for every selected triple in the active full adaptive COSTAR path. Do not give `PatchTST+iTransformer+TimesNet` a special trained static neural prior unless a future experiment trains compatible static priors for all compared triples.

Reason:

The old `OLD_FIXED3` exception made cross-core comparisons structurally uneven. Equal static weights isolate the causal online/horizon-variable/specialist mechanisms and keep the implementation symmetric across triples.

## Sequential Query Routing Is Not The Current Best Direction

Status: Tested / not current direction

Evidence:

- Weighted pairwise sequential COSTAR: MAE `0.368074 +/- 0.000078`, worse than fixed-3 `0.367265`.
- STOP-aware listwise collapsed to one query and scored MAE `0.375253`.
- Full-sequence Transformer router did not beat the current sequential seed7 baseline or fixed-3.
- Forecast-state partial sequential full ablation scored MAE `0.371078 +/- 0.000676`, worse than fixed-3 and current best horizon-variable adaptive weighting.
- After-final-test audit of frozen `utility_pairwise_weighted` Sequential COSTAR checkpoints scored ETTh1 test MAE `0.330832 +/- 0.005462` and ETTh2 test MAE `0.300576 +/- 0.000032`, worse than the final frozen adaptive test rows.

Decision:

Do not spend more effort on ranking-objective-only, bigger-sequence-router, or this exact forecast-state partial sequential setup unless paired with a materially new source of signal.

Reason:

The main failure appears to be weak out-of-split predictability of per-window utility, not insufficient ranking loss capacity.

## History/Fingerprint/Attention Representations Did Not Solve Routing

Status: Tested / not promising

Evidence:

- Fingerprint embedding comparison: representative mean MAE `0.374123`, worse than fixed-3.
- Q/K attention summary exists and did not change the overall direction.
- Oracle predictability diagnostic: train R2 positive but validation R2 negative.

Decision:

Do not repeat simple fingerprint-history or Q/K/V routing experiments unless the input signal or target changes substantially.

Reason:

Representation changes alone did not make expert utility predictable on validation.

## Oracle Utility Is Real But Hard To Predict Directly

Status: Confirmed

Evidence:

- Small-menu oracle MAE `0.347039` shows large headroom.
- Oracle-weight predictability diagnostic validation R2 `-0.2898`; validation top-1 oracle expert accuracy `32.13%`.

Decision:

Use oracle analyses to identify structure and upper bounds, but avoid relying on direct per-window oracle label prediction as the main mechanism.

Reason:

Train labels do not generalize cleanly to validation.

## Chronological Adaptation Works

Status: Confirmed

Evidence:

- Chronological EMA hybrid: MAE `0.365534 +/- 0.000112`, 5/5 seed wins vs oracle prototype-residual.
- Expert best identity changed 4 times across 6 validation blocks.
- Frozen COSTAR validation diagnostic: removing validation-time target feedback while preserving train-derived weights worsened ETTh1 from online `0.363111` to frozen `0.365868`, and ETTh2 from online `0.276832` to frozen `0.277481`.

Decision:

Prefer causal recent-performance adaptation over static router-only selection.

Reason:

Expert usefulness shifts over time; past realized performance is a stronger signal than history-only routing.

## Horizon x Variable Adaptation Is Current Best

Status: Confirmed former best

Evidence:

- Low-rank rank-1 horizon x variable EMA blend: MAE `0.363642 +/- 0.000014`, MSE `0.306712 +/- 0.000016`.
- Beats chronological best by `0.001892` MAE, 5/5 seeds, CI excludes zero.
- Variable 4 accounted for the largest measured variable-level gain.

Decision:

Treat horizon x variable chronological adaptive weighting as the strongest frozen baseline for residual-correction work, but no longer as the absolute best validation result.

Reason:

It captures specialization hidden by global expert weights.

## Conservative Ridge Residual Correction Is Former Best

Status: Confirmed former best

Evidence:

- Ridge residual corrector on top of `hybrid_chrono_hvema_lowrank1_decay0.95_temp0.1_alpha0.75`: MAE `0.363301 +/- 0.000015`, MSE `0.306286 +/- 0.000017`.
- Improvement over previous best `0.363642`: `0.000341` MAE.
- Aggregate paired bootstrap CI versus previous best: `[-0.000378, -0.000305]`.
- Per-horizon MAE improved at every horizon on average.
- Worst average horizon-variable regression was small: horizon `0`, variable `5`, delta `+0.000130` MAE.

Decision:

Treat the conservative ridge residual layer as a strong former-best residual baseline. It may be tested as an add-on to newer methods, but is no longer the absolute best validation result.

Reason:

There is a modest but statistically supported residual signal beyond horizon-variable adaptive weighting.

## Expanded Optional Expert Specialists Are Current Best

Status: Confirmed; active implementation now equal-static

Evidence:

- Expanded-pool method `expanded_both` on top of `hybrid_chrono_hvema_lowrank1_decay0.95_temp0.1_alpha0.75`: MAE `0.363112 +/- 0.000013`, MSE `0.306057 +/- 0.000016`.
- Improvement over fixed-three HV baseline `0.363642`: `0.000529` MAE, aggregate paired CI `[-0.000557, -0.000502]`.
- Improvement over prior ridge residual best `0.363301`: `0.000189` MAE, paired CI `[-0.000233, -0.000143]`.
- Router-train selection had `3/4` fold wins for the promoted both-specialist config.
- `expanded_both` improved every horizon and every variable on average; worst horizon-variable regression was small (`+0.000138` MAE).
- Equal-static cleanup removed the ETTh1 neural static-prior exception and scored validation MAE/MSE `0.363100` / `0.306026`.

Decision:

Promote DLinear and ModernTCN as capped optional specialists on top of the fixed-three horizon-variable baseline. Do not make them equal ensemble members.

Reason:

Although DLinear and ModernTCN are poor average experts, causal recent-advantage activation found repeatable windows where small positive weights help.

## MLP Residual Correction Is Not Preferred

Status: Tested / not current direction

Evidence:

- MLP residual corrector improved over `0.363642` with MAE `0.363318 +/- 0.000109`.
- It was worse and less seed-stable than ridge.
- It had larger local regressions on variable `4`.

Decision:

Do not continue generic tiny-MLP residual correction unless the new experiment specifically addresses seed stability and local-regression control.

Reason:

The extra nonlinearity did not beat ridge and increased variance/regression risk.

## Long-Training Grokking Is Unsupported For Prototype-Residual Router

Status: Tested / do not continue as-is

Evidence:

- Diagnostic model: `final_phase2_protores_lam0.01_k16_scale0.3_rw0.001`.
- Used only router-train chronological fold: train starts `2880..7423`, eval starts `7424..8532`.
- Original duration checkpoint at epoch `10` had fold MAE around `0.34423`.
- Best fold checkpoint occurred at epoch `26`, not in the delayed region: MAE `0.342053`, MSE `0.257475`, weight decay `0.1`.
- Delayed checkpoints were worse: best delayed epoch `50` MAE `0.344086`; epoch `100` MAE `0.347171`.
- Training MAE continued improving while chronological fold MAE degraded.

Decision:

Do not spend more compute on 10x/longer training of this prototype-residual router for grokking. If revisiting this family, focus on router-train checkpoint selection or new features/objectives.

Reason:

The curve shows ordinary mid-training improvement followed by overfitting, not delayed sustained generalization.

## ETTh2 Transfer Did Not Help ETTh1

Status: Tested / not promising as implemented

Evidence:

- ETTh2 transfer screen showed encoder/full transfer worse than scratch for ETTh1 dynamic residual and sequential routers.

Decision:

Do not launch large ETTh2 transfer sweeps without a new transfer hypothesis.

Reason:

The simple transfer setup degraded validation performance.

## Residual-Covariance Weighting Is Not Promoted

Status: Tested / not current direction

Evidence:

- Selected config: `diagonal_variance_hv_decay0.99_ridge0.0001_sd1_sg0_bias0_alpha0.5_warm96`.
- Router-train folds improved by `0.000364` MAE with `3/4` wins.
- Validation MAE was `0.363649 +/- 0.000012`, slightly worse than fixed-three HV `0.363642`.
- Aggregate paired CI versus fixed-three HV crossed zero: `[-0.000105, 0.000121]`.
- Worst horizon-variable regression was large: horizon `11`, variable `4`, delta `+0.007174`.

Decision:

Do not promote covariance or diagonal-variance residual weighting as implemented.

Reason:

The fold signal did not transfer to validation and produced unacceptable local regression.

## Regime Adaptive Forgetting Is Not Promoted

Status: Tested / not current direction

Evidence:

- Selected config: `zscore_slow0.99_fast0.95_thr2.5_delta0_reset0_cool24_boost24`.
- Router-train folds improved by `0.000342` MAE with `3/4` wins.
- Validation worsened to `0.364346 +/- 0.000015`; paired CI versus fixed decay was `[0.000618, 0.000790]`.
- Oracle change points were an ineligible diagnostic and also worsened validation to `0.364015`.

Decision:

Do not continue detector-controlled EMA forgetting in this form.

Reason:

Detector-selected adaptation speed overfit router-train shifts and hurt validation.

## Expanded-Specialist Mechanism Transfers To ETTh2 In Limited Form

Status: Superseded / normalization caveat

Evidence:

- Existing non-test replication caches were available only for ETTh2.
- The initial limited replication used scaler-normalized diagnostics and reported MAE `0.093369` versus equal fixed-three `0.098339`.
- A canonical follow-up showed the ETTh2 router-summary protocol is raw/original-scale MAE with `std=ones`; it exactly reproduces the older ETTh2 fixed-baseline summary.
- Under canonical raw metrics, best single is `DLinear` MAE `0.280957`, and best fixed is `DLinear+ModernTCN` MAE `0.275229`.

Decision:

Do not cite normalized ETTh2 values as router-summary results. Use the canonical raw/original-scale protocol in `experiments/etth2_canonical_protocol/`.

Reason:

The normalized diagnostic scale and canonical router-summary scale answer different questions. ETTh2 comparisons need one protocol on the same examples.

## ETTh2 Canonical Router Baseline

Status: Confirmed

Evidence:

- `experiments/etth2_canonical_protocol/run_canonical_etth2_baselines.py` verifies same ETTh2 split, validation window IDs, horizon, variables, cache files, sample count, MAE implementation, normalization, and inverse-transform behavior.
- Validation starts are exactly `10800..11412`, `613` windows.
- Canonical metric uses `sample_mae`/`sample_mse` with `std=ones`; no inverse transform is applied.
- The recomputed best-by-size rows exactly match `results/router_summary/costarts_fresh/ETTh2_96_12/sequential_utility_ranking_combined/summary.json`.

Decision:

For ETTh2, compare against best single `DLinear` MAE `0.280957`, MSE `0.171493`, and best fixed baseline `DLinear+ModernTCN` MAE `0.275229`, MSE `0.165345`.

Reason:

This is the only currently verified ETTh2 protocol where every baseline is evaluated on the same examples with matching metric semantics.

## ETTh1 Current Best Survives Train-Only Core Selection

Status: Confirmed; static-prior asymmetry removed in later cleanup

Evidence:

- `experiments/train_selected_core_etth1/frozen_config_before_validation.json` was written before validation was loaded.
- Router-train-only chronological OOF selection over all 10 triples chose `PatchTST+iTransformer+TimesNet`.
- Selected triple OOF MAE/MSE: `0.345568` / `0.260786`.
- Validation fixed-3 MAE/MSE: `0.367265` / `0.310530`.
- Current-best architecture with train-selected core: `0.363112 +/- 0.000013`, MSE `0.306057 +/- 0.000016`.
- Paired CI versus selected fixed-3: `[-0.004460, -0.003856]`.
- Later equal-static cleanup with the same train-selected core scored validation MAE/MSE `0.363100` / `0.306026`.

Decision:

Keep `expanded_both` over `hybrid_chrono_hvema_lowrank1_decay0.95_temp0.1_alpha0.75` as the current ETTh1 best. The suspected validation-based fixed-three selection issue is resolved for this core because router-train independently selects the same experts.

Reason:

The clean train-only selection and the prior development selection agree. The later equal-static cleanup also removes the core-specific static neural-prior asymmetry, so the active validation path no longer depends on a special `OLD_FIXED3` checkpoint.

## ETTh2 Train-Selected Core Does Not Beat Fixed-2

Status: Tested / not promoted

Evidence:

- ETTh2 router-train-only OOF selection chose `DLinear+PatchTST+ModernTCN`.
- Selected OOF MAE/MSE: `0.284658` / `0.181718`.
- Train-selected fixed-3 validation MAE/MSE: `0.280878` / `0.171933`.
- Frozen current-best architecture validation MAE/MSE: `0.276832` / `0.167280`.
- Canonical best fixed-2 `DLinear+ModernTCN` remains better: MAE `0.275229`, MSE `0.165345`.
- Duplicate DLinear/ModernTCN specialist branches were disabled because both optional specialists were already in the selected core.

Decision:

Do not promote the ETTh1 frozen current-best architecture as an ETTh2 model under this train-selected-core protocol.

Reason:

The architecture improves over its own selected fixed-3, but not enough to beat the simpler fixed-2 baseline. ETTh2 also lacks a compatible static neural fixed-three artifact, so the chronological static prior was necessarily equal-weight and explicitly frozen.

## ETTh2 Variable-Size Core Selection Is Not Promoted

Status: Tested / not promoted

Evidence:

- ETTh2 router-train-only OOF selection over all 31 non-empty expert subsets chose the single expert `DLinear`.
- Selected OOF MAE/MSE: `0.283464` / `0.182616`.
- Selected core validation MAE/MSE: `0.280957` / `0.171493`.
- Frozen current-best architecture on the selected single-expert core scored validation MAE/MSE `0.280470` / `0.170973`.
- Previous full fixed-3 model remains better: MAE `0.276832`.
- Canonical best fixed-2 `DLinear+ModernTCN` remains better: MAE `0.275229`.
- Fold-best selected sizes were unstable: `3, 1, 1, 3`.

Decision:

Do not promote variable-size router-train core selection for ETTh2 as implemented.

Reason:

Allowing router-train to choose the subset size selected a simpler core, but that core did not validate better than the forced fixed-3 or fixed-2 baselines. The result suggests pooled router-train OOF can over-prefer `DLinear` for ETTh2 and is not enough by itself to select a validation-robust adaptive core.

## Pooled Router-Train Core Selection Is A Valid Sensitivity Check, Not A New Final Claim

Status: Tested after final test was seen

Evidence:

- `experiments/pooled_router_train_core/run_pooled_router_train_core.py` enumerated all 10 three-expert cores and selected by pooled router-train MAE, with MSE as tie-breaker.
- ETTh1 selected `PatchTST+iTransformer+TimesNet`, matching the existing fold-selected core.
- ETTh2 selected `DLinear+TimesNet+ModernTCN`, differing from the existing fold-selected `DLinear+PatchTST+ModernTCN`.
- ETTh2 pooled full adaptive test MAE/MSE was `0.295829` / `0.219681`, better than the existing fold-selected full adaptive test MAE/MSE `0.297808` / `0.218612` on MAE but worse on MSE.

Decision:

Record pooled router-train selection as a useful post-test sensitivity result. Do not treat it as replacing the preregistered frozen final ETTh2 model, because the experiment was requested and run after prior test metrics were already viewed.

Reason:

For a frozen equal-average core, pooled router-train MAE is a simpler and mostly equivalent selection statistic to equal-sized chronological fold averaging. The ETTh2 difference shows the selection rule matters enough to pre-register in any future clean benchmark.

## Pooled Router-Train Residual Selection Is Sensitivity Evidence

Status: Tested after final test was seen

Evidence:

- `experiments/pooled_router_train_residual_correctors/POOLED_ROUTER_TRAIN_RESIDUAL_RESULTS.json`
- `project_memory/experiments/2026-08-14_pooled_router_train_residual_correctors.md`

Decision:

Treat pooled router-train residual-corrector selection as an after-final-test sensitivity audit. The ETTh1 pooled-selected MLP residual corrector scored test MAE/MSE `0.325964` / `0.266587`, which is the lowest audited ETTh1 test MAE, but it was evaluated after final test results had already been viewed and does not supersede the preregistered frozen ETTh1 model.

Do not promote pooled-selected ETTh1 Ridge despite validation MAE `0.363088`; it worsened on test to MAE `0.328022`, worse than the fixed core and full adaptive model.

Reason:

Pooled in-sample router-train selection can select more aggressive configs. The ETTh1 MLP result is interesting, but the ETTh1 Ridge failure shows this selection rule can overfit and should be preregistered before being used as a final model-selection protocol.
