# Research Decisions

## Window-Dependent Expert-Choice Family: Keep the Core Result, Reject Two Follow-Up Mechanisms, Keep Embeddings

Status: Tested 2026-08-30 through 2026-08-31 (OOF-only for the two ablations below).

Evidence: `experiments/window_dependent_expert_choice_hv/`, `experiments/affinity_weighted_expert_choice_hv/`, `experiments/conflict_resolved_expert_choice_hv/`, `experiments/feature_ablation_affinity_weighted_ec/`, `experiments/embedding_ablation_affinity_weighted_ec/`, and the five matching `project_memory/experiments/2026-08-3*_*.md` records.

Decisions:

1. `WINDOW_DEPENDENT_EC_SUPPORTED` is real but fragile (two of six predeclared criteria clear the bar at the exact minimum, 3/5). Keep it as the current best Expert-Choice mechanism, but do not oversell it — Dynamic EC still loses to Frozen HxV on 0/5 datasets, and ETTh2 loses on both OOF and val.
2. Do NOT replace the equal-average multi-claim rule with affinity-weighted combination: the effect is real by the predeclared rule but ~100x smaller than the window-dependent effect itself (practically negligible). An oracle diagnostic shows real headroom in combining multi-claim forecasts better exists, but simple affinity-renormalization does not capture it — if this is revisited, it needs a genuinely smarter combination rule, not a variant of affinity weighting.
3. Do NOT pursue conflict-resolved (exactly-one-expert-per-cell) assignment: it lost to affinity-weighted EC OOF MAE on 0/5 datasets, uniformly. Multi-claim cells are providing real ensembling value; removing that redundancy costs more than it gains.
4. Prefer `F2_local` (cell-local + per-variable features, no global-history features) over the current full model `F3_full` for this scorer family — it wins OOF MAE on 4/5 datasets. None of the three feature groups (cell/local/global) reached a clean `SUPPORTED` bar in the leave-one-group-out ablation; global in particular has only one independent piece of evidence (its add-test and remove-test are mathematically the same comparison).
5. Keep all three identity embeddings (H, V, Expert) — all three are `SUPPORTED` in the embedding ablation (V and Expert unanimous 5/5), including confirming V adds real information beyond the `static_gain[h,v,e]` scalar and that the shared (not per-expert) scorer needs the Expert embedding to distinguish heterogeneous experts.

None of these five experiments touched `router_val` except the original `window_dependent_expert_choice_hv` run (once) and its direct reuse in `affinity_weighted_expert_choice_hv`/the passed-gate half of `conflict_resolved_expert_choice_hv`; the two ablations are router_train-OOF-only by design and were never extended to router_val.

## Static Expert-Choice HxV Routing Is Mixed, Not Ready For Learned Router Integration

Status: Tested validation-only on 2026-08-30

Evidence: `experiments/expert_choice_hv/report.md`, `results.json`, `assignment_stats.csv`, `dependence_tests.csv`, `integrity_checks.json`, and `project_memory/experiments/2026-08-30_expert_choice_hv.md`.

Decision: Do not proceed directly to an input-dependent learned Expert-Choice HxV router from the current static EC-HVR mechanism. Treat the result as `MIXED_EXPERT_CHOICE`: the primary CF1 expert-to-cell allocation beats matched TokenChoice Top1 on all five datasets and creates distinct HxV claim regions, but the secondary CF2 budget fails the predeclared consistency requirement and EC CF1 remains worse than Frozen HxV on every dataset.

Reason: Reversing routing direction is not empty; CF1 improved over Token Top1 on ETTh1, ETTh2, ETTm1, Weather, and Electricity, with block-24 support on ETTm1, Weather, and Electricity. However, CF2 beat Token Top2 only on Weather and Electricity, while ETTh1 and ETTm1 significantly regressed. Since existing Frozen HxV still beats EC CF1 across the board, the durable lesson is that expert-choice allocation may be a useful constraint or diagnostic, not that static expert-to-cell HxV routing is a replacement router.

## Expert-Choice HxV Allocation Has A Positive Pilot Signal

Status: Tested validation-only on 2026-08-30

Evidence: `experiments/behavioral_competence/expert_choice_hv_pilot/RESULTS.md`, `results.json`, `integrity_report.json`, and `project_memory/experiments/2026-08-30_expert_choice_hv_pilot.md`.

Decision: Treat the capacity-constrained Expert-Choice HxV allocation direction as worth a targeted follow-up, but do not replace the existing Electricity soft/causal HxV router. The requested decision rule gives `STRONG GO` for `Expert Choice cap 1.25` versus static Hard Normal HxV because it improves MAE by `-0.002134`, block-24 CI excludes zero, and every-12th phase analysis agrees.

Reason: The pilot isolated allocation from competence modeling: Hard Normal HxV and Expert Choice consumed the same train-derived score tensor. No-capacity Expert Choice was exactly identical to Hard Normal HxV, so the cap-constrained improvement is genuinely from the allocation constraint. However, Hard Normal HxV itself is weak on Electricity (`0.222761` MAE), and the best Expert Choice variant (`0.220627`) remains worse than Equal (`0.214457`) and existing soft/causal HxV (`0.211775`), so the durable lesson is "capacity constraints can repair hard HxV over-allocation," not "deploy this static hard router."

## Rolling-Origin Revision Embeddings Are Negative

Status: Tested validation-only on 2026-08-30

Evidence: `experiments/behavioral_competence/rolling_origin_revision_embedding/report.md`, `validation_results.json`, `dependence_tests.csv`, `integrity_checks.json`, and `project_memory/experiments/2026-08-30_rolling_origin_revision_embedding.md`.

Decision: Do not proceed to test-set evaluation or router integration from the current rolling-origin revision embedding mechanism. Treat the outcome as `NEGATIVE_RESULT`: real historical revision trajectories produced small router_val point-estimate routing gains, but did not establish robust competence information beyond the learned context encoder.

Reason: The mandatory incremental evidence failed. ContextPlusRevision beat ContextEmbed OOF R2 on ETTm2 but regressed on Traffic; `RevisionEmbedding -> ContextEmbed residual` OOF R2 was negative on both datasets (ETTm2 `-0.157214`, Traffic `-0.021563`); and the expert-specific wrong/shuffled controls were not consistently beaten. Fixed-rank routing improved by only small point estimates, which is not enough to rescue the failed mechanism criteria.

## Expert-Native Latent Competence Is Mixed

Status: Tested validation-only on 2026-08-29

Evidence: `experiments/behavioral_competence/expert_native_competence/RESULTS.md`, `results.json`, `validation_results.csv`, `per_expert_results.csv`, `dependence_tests.csv`, `representation_manifest.json`, and `integrity_report.json`.

Decision: Do not proceed to test-set evaluation or router integration from the current expert-native hidden representation mechanism. Treat the outcome as `MIXED_SUPPORT`: Electricity showed strong unique hidden value and ETTh1 showed a small R2-only positive, but ETTh2, ETTm1, and Weather did not support robust incremental value beyond Passive features.

Reason: The mandatory incremental comparisons were inconsistent. Passive+Hidden minus Passive R2 was ETTh1 `+0.016825`, ETTh2 `-0.178095`, ETTm1 `-0.027923`, Weather `-0.002443`, Electricity `+0.342262`; block-24 dependence support was clear only for Electricity, while ETTh2/ETTm1 significantly regressed. Shuffled and raw/control comparisons do not support a strong cross-dataset Expert-Native routing claim.

## Pair-Residual Fault Gate Is Weak/Inconsistent

Status: Tested validation-only on 2026-08-28

Evidence: `experiments/behavioral_competence/pair_residual_fault_gate/report.md`, `routing_results.csv`, `fault_detector_results.csv`, `dependence_tests.csv`, and `integrity_checks.json`.

Decision: Do not proceed to test-set evaluation or router integration. Signed expert-pair residual/parity features produced a supportive ETTh2 case, but the effect did not generalize: ETTh1, ETTm1, and Electricity regressed versus the train-only Baseline, and Weather's improvement came from the Passive+Raw forecast control rather than parity. Treat the outcome as `WEAK_OR_INCONSISTENT_PARITY_FAULT_SIGNAL`.

Reason: The experiment separates relative bust detection from routing intervention. Fault detectors can rank bust risk, but the multiplicative suppression gate does not reliably improve held-out router_val MAE and does not consistently beat Passive or Raw Forecast Control.

## Capability-Demand Matching Has Signal But No Matching Gain

Status: Tested validation-only on 2026-08-28

Evidence: `experiments/behavioral_competence/capability_demand_matching/report.md`, `competence_results.csv`, `dependence_tests.csv`, `integrity_checks.json`, and `etth2_integrity_audit.json`.

Decision: Do not proceed to test-set evaluation or router integration. The semantic demand/capability profiles are signal-bearing, and correct profiles beat expert-profile shuffles on all five datasets, but the primary CapabilityMatch does not consistently improve over Passive ABC or the capacity-matched FAME-style direct demand baseline. Treat the outcome as `CAPABILITY_SIGNAL_BUT_NO_MATCHING_GAIN`.

ETTh2 note: The earlier ETTh2 runtime/cache reproduction discrepancy is resolved. ETTh2 cache histories are already DLinear-scaler-normalized; passing them through a wrapper that expects raw histories double-normalized them. Direct normalized-cache inference and de-normalized raw-history wrapper inference reproduce the cached forecasts within tolerance.

## Structured Forecast Repair Is Weak/Ambiguous

Status: Tested validation-only on 2026-08-28

Evidence: `experiments/behavioral_competence/structured_forecast_repair/report.md`, `competence_results.csv`, `dependence_tests.csv`, and `integrity_checks.json`.

Decision: Do not proceed to test-set evaluation or router integration. RepairGeometry improved Passive relative-error prediction on only ETTh1 and ETTh2 of five datasets, while REP dominated on Electricity and the expert-shuffle result was mixed. The mechanism does not meet the strong cross-dataset criteria. The later capability-demand audit resolved the ETTh2 cache/runtime convention mismatch, but that does not change this weak/ambiguous scientific conclusion.

Last updated: 2026-08-29

These are durable conclusions supported by repository outputs. Do not repeatedly rediscover them unless a new hypothesis materially changes the experiment.

## Counterfactual Forecast Revision Is Signal-Bearing But Not Strong Incremental Evidence

Status: Tested validation-only

Evidence:

- `experiments/behavioral_competence/counterfactual_forecast_revision/report.md`
- `experiments/behavioral_competence/counterfactual_forecast_revision/validation_results.json`
- `experiments/behavioral_competence/counterfactual_forecast_revision/integrity_checks.json`
- `project_memory/experiments/2026-08-27_counterfactual_forecast_revision.md`

Decision:

Do not freeze the current CFR mechanism for untouched-dataset testing or router integration. Treat the preregistered outcome as `CFR_SIGNAL_BUT_REDUNDANT`, not `INCREMENTAL_MODEL_SPECIFIC_CFR`.

Reason:

CFR and RelativeCFR showed competence association and several Passive+CFR point improvements, with some dependence-aware support. However, the mandatory direct Passive-residual test was positive on only `1/4` datasets (`ETTm2`) and negative on ExchangeRate, Traffic, and BeijingAirQuality. Correct expert mapping beat shuffled mapping on some datasets, but not consistently enough to overcome the residual-test failure. The mechanism is interesting but not yet a robust incremental expert-specific signal beyond passive features.

## Conditional Nuisance Invariance Does Not Establish A Robust LearnedProbe Mechanism

Status: Tested validation-only

Evidence:

- `experiments/behavioral_competence/conditional_nuisance_invariance/results/report.md`
- `experiments/behavioral_competence/conditional_nuisance_invariance/results/results.json`
- `experiments/behavioral_competence/conditional_nuisance_invariance/results/integrity_checks.json`
- `project_memory/experiments/2026-08-27_conditional_nuisance_invariance.md`

Decision:

Do not treat the canonical expert-conditioned LearnedProbe as a robust cross-dataset active competence mechanism after passive and nuisance controls. The preregistered outcome is `MIXED_CNI`, not `PROBE_SURVIVES_CNI`.

Reason:

`Electricity` produced strong supportive evidence, including `+0.088410` pairwise improvement for `Passive+Nuisance+Probe` over `Passive+Nuisance` with block-24 support. The broader pattern did not hold: `ETTh1` and `ETTh2` had smaller non-significant post-nuisance gains, `ETTm1` was nearly flat, `Weather` regressed, and shuffled/wrong-expert/environment-transfer controls mostly failed. This suggests dataset-dependent active information, not a stable mechanism ready for router integration.

## Data-Model Dynamics Alignment Is Weak/Ambiguous, Not A Strong Integration Candidate

Status: Tested validation-only

Evidence:

- `experiments/data_model_dynamics_alignment/report.md`
- `experiments/data_model_dynamics_alignment/results.json`
- `experiments/data_model_dynamics_alignment/integrity_checks.json`
- `project_memory/experiments/2026-08-27_data_model_dynamics_alignment.md`

Decision:

Do not treat the current local PCA + ridge VAR(1) vs frozen-expert JVP mismatch mechanism as a strong-go router integration candidate. It may be useful as a hypothesis-generating feature, but the preregistered outcome is `WEAK_OR_AMBIGUOUS`, not `STRONG_GO`.

Reason:

Only `Weather` passed all criteria A-E. Passive+Align had real routing improvements on `ETTh2`, `Weather`, and `Electricity`, but `ETTm1` showed a block-24 significant regression and `ETTh1` did not improve. Controls also weakened the mechanism story: `Electricity` did not beat J-magnitude or VAR-closeness controls, and `ETTm1` significantly lost to shuffled dynamics. Passive-residual R2 was small or negative across datasets.

## Raw Response Probe V3A Cannot Run From Current Frozen V2 Artifacts

Status: Blocked before scientific evaluation

Evidence:

- `experiments/behavioral_competence/raw_response_probe_v3a/report.md`
- `experiments/behavioral_competence/raw_response_probe_v3a/raw_response_shape_diagnostics.csv`
- `project_memory/experiments/2026-08-25_raw_response_probe_v3a_blocked.md`

Decision:

Do not report a V3A scientific classification from the current V2 artifact set. Do not reconstruct OOF raw responses by retraining `SharedControlledProbeGenerator`; that violates the V3A frozen-intervention rule.

If V3A is still desired, first run a separate V2-compatible artifact-generation experiment that saves OOF learned deltas/full raw response tensors or trained fold generator checkpoints. That would be a new experiment, not a continuation of the completed frozen V2 result.

Reason:

The current V2 artifacts have router-val learned deltas and OOF six-stat responses, but lack OOF learned deltas, OOF full raw response tensors, and trained V2 generator checkpoints. Therefore the primary OOF comparisons `RawResponseActive vs SixStatActive`, `RawResponseActive vs ShuffledRawResponse`, `PassivePlusRaw vs PassiveOnly`, and `RawResponse -> passive residual` cannot be run under the hard no-retraining rule.

## V3A Reproduced: Six-Stat Compression Is Not The Bottleneck

Status: Tested after accepted V2-compatible artifact reproduction

Evidence:

- `experiments/behavioral_competence/controlled_discriminative_probe_v2_reproduction/reproduction_decision.json`
- `experiments/behavioral_competence/controlled_discriminative_probe_v2_reproduction/reproduction_comparison.csv`
- `experiments/behavioral_competence/raw_response_probe_v3a_reproduced/report.md`
- `project_memory/experiments/2026-08-25_v2_reproduction_v3a_reproduced.md`

Decision:

Treat `experiments/behavioral_competence/controlled_discriminative_probe_v2_reproduction/` as accepted frozen-protocol V2 reproduction artifacts, not as exact original V2 tensors. They may be used for the reproduced V3A raw-response representation analysis because the reproduction gate passed `317/317` observable checks and reproduced V2's `ACTIVE_SIGNAL_BUT_REDUNDANT` classification with `proceed_to_router_integration=false`.

Do not treat full raw forecast responses as rescuing the V2 active-probe result. V3A reproduced classified as `SIX_STATS_NOT_THE_BOTTLENECK`: raw response beat six-stat on only `1/4` datasets, real raw beat shuffled raw on `2/4`, Passive+Raw beat Passive on only `1/4`, and raw-response passive-residual R2 was positive on `0/4`.

Reason:

The fixed `Ridge(alpha=1.0)` comparison used train-only standardization and accepted reproduced OOF raw responses. The full raw response did not consistently improve active prediction, did not add reliable incremental value beyond passive features, and did not predict what MatchedPassive missed. The completed V2 interpretation remains active signal but redundant with passive information.

## Controlled Discriminative LearnedProbe v2 Is Redundant With Passive Signals

Status: Tested / do not integrate into routers

Evidence:

- `experiments/behavioral_competence/controlled_discriminative_probe_v2/report.md`
- `experiments/behavioral_competence/controlled_discriminative_probe_v2/validation_results.json`
- `project_memory/experiments/2026-08-25_controlled_discriminative_probe_v2.md`

Decision:

Do not proceed to TimeFuse, FFORMA, simplex/selective, or COSTAR router integration for the current shared active-probing formulation.

Treat the result as `ACTIVE_SIGNAL_BUT_REDUNDANT`: the shared learned intervention produces some competence-related correlation, but it does not provide robust incremental information beyond passive features. Only `1/6` predeclared criteria were met. Active features failed to predict MatchedPassive residuals on all four datasets, and Passive+Active improved over Passive on only one dataset.

Reason:

The experiment directly tested the intended mechanism under strict purged-OOF causality, frozen experts, same-question perturbation invariance, target-corruption invariance, and no test-cache access. Since the active signal is mostly redundant with passive observations, router integration would add complexity without a supported independent signal.

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

## Published Baseline Test Audit Is Post-Hoc Comparative Evidence

Status: Additional after-final-test audit

Evidence:

- `experiments/published_baseline_test_audit/PUBLISHED_BASELINE_TEST_AUDIT_RESULTS.json`
- `experiments/published_baseline_test_audit/PUBLISHED_BASELINE_TEST_AUDIT_REPORT.md`
- `project_memory/experiments/2026-08-18_published_baseline_test_audit.md`

Decision:

Treat the published-baseline test audit as frozen post-hoc comparative evidence only. The strongest audited rows were:

- ETTh1 Frozen COSTAR + MLP residual: MAE/MSE `0.326047` / `0.267322`.
- ETTh2 Bates-Granger: MAE/MSE `0.296294` / `0.217423`.

Do not use these rows to tune COSTAR, choose a new official final model, or claim a clean untouched final-test comparison.

Reason:

The configurations were selected from router-train/validation artifacts and the audit did not select parameters from test performance, but ETTh1/ETTh2 test metrics had already been viewed before this comparison was requested.

## Six Previously Untested Published Baselines Are Frozen After-Final-Test Audit Rows

Status: Additional after-final-test audit

Evidence:

- `experiments/published_baseline_test_audit/TEST_RESULTS.json`
- `experiments/published_baseline_test_audit/leakage_and_causality_checks.json`
- `project_memory/experiments/2026-08-18_after_final_test_audit_six_methods.md`

Decision:

Treat the ETTh1/ETTh2 test evaluation of Equal all-5 ensemble, Granger-Ramanathan, Bates-Granger, FAME adaptation, TimeRouter adaptation, and OneNet-style frozen-expert adaptation as frozen after-final-test audit rows, labeled `after_final_test_audit`. On ETTh1, Online COSTAR beats all six. On ETTh2, Bates-Granger (test MAE `0.296294`), FAME (`0.298372`), and Granger-Ramanathan (`0.298419`) all beat Online COSTAR's test MAE `0.297808`; TimeRouter and Equal all-5 do not; OneNet is far worse than Online COSTAR on ETTh2 test (`0.407526`). Do not use these rows to tune COSTAR, choose a new official final model, or claim a clean untouched final-test comparison.

Reason:

All six configurations were read verbatim from `experiments/published_baseline_comparisons/{ETTh1,ETTh2}/frozen_config_before_validation.json`, written before validation was ever loaded, and no parameter, threshold, or expert-set choice was changed from these test results. Explicit target-replacement invariance checks (GR/Bates-Granger/FAME/TimeRouter) and a future-target perturbation causality test (OneNet) all passed with zero leakage. However, ETTh1/ETTh2 test metrics had already been viewed elsewhere in this project before this audit was requested, so this remains comparative evidence, not a new preregistered final-test claim.

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

Status: Main full adaptive model for ETTh1

Evidence:

- `experiments/train_selected_core_etth1/run_train_selected_core_eval.py` now sets `static_weights = torch.full((num_windows, 3), 1.0 / 3.0)` for every selected triple.
- `experiments/frozen_costar/run_frozen_costar_validation.py` no longer imports or loads the ETTh1 static neural-router checkpoint.
- ETTh1 equal-static full adaptive validation MAE/MSE: `0.363100` / `0.306026`, saved in `experiments/train_selected_core_etth1_equal_static/final_report.json`.
- ETTh1 equal-static full adaptive after-final-test audit MAE/MSE: `0.326408` / `0.267378`, nearly matching the old preregistered neural-static-prior test MAE `0.326395`.

Decision:

Use equal static weights for every selected triple in the active full adaptive COSTAR path. The equal-static ETTh1 full adaptive path is now the main full adaptive model for ETTh1 going forward. Do not give `PatchTST+iTransformer+TimesNet` a special trained static neural prior unless a future experiment trains compatible static priors for all compared triples.

This does not rewrite the historical preregistered final-test artifact: the old final frozen adaptive result remains MAE/MSE `0.326395` / `0.267509`, while the main active equal-static audit result is MAE/MSE `0.326408` / `0.267378`.

Reason:

The old `OLD_FIXED3` exception made cross-core comparisons structurally uneven. Equal static weights isolate the causal online/horizon-variable/specialist mechanisms and keep the implementation symmetric across triples.

## Published Baselines Are Validation Comparators

Status: Implemented validation-only

Evidence:

- `experiments/published_baseline_comparisons/FINAL_REPORT.json`
- `project_memory/experiments/2026-08-17_published_baseline_comparisons.md`

Decision:

Use Granger-Ramanathan, Bates-Granger, FAME adaptation, TimeRouter adaptation, and OneNet-style adaptation as validation comparators against COSTAR. Treat FAME, TimeRouter, and OneNet as adaptations rather than exact official reproductions because their original expert pools, metadata/context features, TSFM checkpoints, or online expert updating are not the same as the frozen BasicTS expert cache.

Reason:

The new runner provides fair validation-only comparisons over the same frozen forecasts, with chronological router-train selection and no test loading. Bates-Granger is promising on ETTh2 validation; none of the new baselines beat ETTh1 online COSTAR validation.

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
