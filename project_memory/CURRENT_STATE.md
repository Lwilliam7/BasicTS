# Current State

Last updated: 2026-09-03

## Expert-Choice Variant Sweep on Window-Dependent EC -- ETTh1 only (2026-09-03)

Completed `experiments/ec_variant_sweep_etth1/` per explicit user instruction to work ONLY on
ETTh1, using ONLY router_train OOF (no router_val, no test, no other datasets). Starting from the
F2_local scorer (identified as the best feature variant by `feature_ablation_affinity_weighted_ec`),
swept a 12-configuration grid over capacity factor (`CF in {0.5,1.0,2.0}`), assignment rule
(`unrestricted` vs `max2` claims per cell), and scoring normalization (`existing` scalar-calibrated
softmax vs `expert_relative` per-expert-calibrated softmax) -- all downstream of one frozen
per-fold raw-score tensor, no retraining across the grid. Best config: `cf2.0_unrestricted_existing`,
OOF MAE `0.349150` vs the F2_local `cf1.0_unrestricted_existing` baseline `0.351924` (block-24
supported win, CI `[-0.003227,-0.002338]`). CF=2.0 beat CF=1.0; `max2` beat `unrestricted` on only
1/6 matched pairs (identical results at CF<=1.0 since the 2-claim cap never actually bound);
`expert_relative` scoring beat `existing` on only 1/6 matched pairs. The best config still lost to
the Frozen Dense Ensemble reference by `+0.003559` MAE (block-24 supported), continuing the
established pattern that no Expert-Choice variant in this line of work has beaten Frozen HxV/dense
ensembling. `INTEGRITY VALID: YES`; no router_val/test/other-dataset access.

Evidence: `experiments/ec_variant_sweep_etth1/report.md`, `results.json`;
`project_memory/experiments/2026-09-03_ec_variant_sweep_etth1.md`.

## Embedding Ablation for Affinity-Weighted Window-Dependent EC (2026-08-31)

Completed OOF-only `experiments/embedding_ablation_affinity_weighted_ec/` on ETTh1, ETTh2, ETTm1, Weather, and Electricity. Used the feature variant selected by the feature-ablation experiment below (`F2_local` = cell-local + per-variable features, no global) without reconsidering it. Tested whether the H(4)/V(8)/Expert(4) identity embeddings are useful to the shared scorer by zeroing each one (fixed zero vector of the same dimension, so MLP capacity is unchanged) and retraining from scratch under the identical causal-OOF recipe. Result: `H EMBEDDING: SUPPORTED` (hurts OOF on 4/5 datasets), `V EMBEDDING: SUPPORTED` (hurts on 5/5, unanimous — V adds real information beyond the `static_gain[h,v,e]` scalar already present in every variant), `EXPERT EMBEDDING: SUPPORTED` (hurts on 5/5, unanimous — the shared scorer needs learned per-expert identity to distinguish heterogeneous experts). `EMBEDDING ABLATION VALID: YES`. `router_val` never touched.

Evidence: `experiments/embedding_ablation_affinity_weighted_ec/report.md`, `results.json`; `project_memory/experiments/2026-08-31_embedding_ablation_affinity_weighted_ec.md`.

## Feature-Group Ablation for Affinity-Weighted Window-Dependent EC (2026-08-31)

Completed OOF-only `experiments/feature_ablation_affinity_weighted_ec/` on ETTh1, ETTh2, ETTm1, Weather, and Electricity (`router_val` never touched; primary and only evidence is router_train causal OOF). Retrained 6 distinct scorer variants (predeclared F0_anchor/F1_cell/F2_local/F3_full plus leave-one-group-out Full-NoCell/Full-NoPerVariable; Full-NoGlobal is provably identical to F2_local and was reused, not retrained) under the identical causal-OOF recipe used by `window_dependent_expert_choice_hv`. Surprising result: **`F2_local` (cell-local + per-variable features, NO global-history features) beats `F3_full` (the current full model) on OOF MAE on 4/5 datasets** — `BEST PREDECLARED FEATURE VARIANT BY OOF: F2_local`. All three feature groups classified `MIXED` (none reached `SUPPORTED`): cell and local features help when added but removing them from the full model doesn't clearly hurt; global features neither help when added nor have independent removal evidence (add and remove reduce to the same F3_full-vs-F2_local comparison by construction). `FEATURE ABLATION VALID: YES`. One post-hoc labeling bug was found and fixed (a disclosure field compared variant name strings instead of underlying feature-group sets) and the report was regenerated from the already-computed OOF numbers without any retraining.

Evidence: `experiments/feature_ablation_affinity_weighted_ec/report.md`, `results.json`; `project_memory/experiments/2026-08-31_feature_ablation_affinity_weighted_ec.md`.

## Conflict-Resolved Window-Dependent Expert Choice (2026-08-30)

Completed OOF-gated `experiments/conflict_resolved_expert_choice_hv/` on ETTh1, ETTh2, ETTm1, Weather, and Electricity. Tested whether resolving duplicate H×V claims via expert-proposing deferred acceptance (every cell held by exactly one expert) beats the affinity-weighted multi-claim combination, using the same frozen affinity tensors (no retraining). Predeclared gate required beating Affinity-Weighted EC OOF MAE on ≥3/5 datasets; result was **0/5** — Conflict-Resolved EC was worse on every dataset by 0.0012-0.0022 MAE, uniformly. `OOF GATE: FAIL`, `router_val` never accessed. Two real implementation bugs (an fp16-affinity exact-tie handling gap, and a numeric key-scaling bug that broke down for Electricity's large cell count) were found and fixed via an explicit literal-deferred-acceptance cross-check before any OOF number was trusted. Interpretation: multi-claim cells provide real ensembling value that strict one-expert-per-cell assignment removes at a net cost.

Evidence: `experiments/conflict_resolved_expert_choice_hv/report.md`; `project_memory/experiments/2026-08-30_conflict_resolved_expert_choice_hv.md`.

## Affinity-Weighted Expert Choice H×V (2026-08-30)

Completed `experiments/affinity_weighted_expert_choice_hv/` on ETTh1, ETTh2, ETTm1, Weather, and Electricity, reusing the frozen `window_dependent_expert_choice_hv` score/affinity/claim tensors with no retraining. Tested whether restoring affinity-weighted multi-claim combination (vs. the existing simple equal-average rule, verified from source) improves OOF/val MAE. Classification `AFFINITY_WEIGHTED_EC_SUPPORTED` by the letter of the predeclared rule, but the effect size (~1e-5 MAE) is about two orders of magnitude smaller than the underlying window-dependent effect and is judged practically negligible. An oracle diagnostic (analysis-only, never used to fit anything) showed real headroom exists in combining multi-claim forecasts better, but affinity-renormalization captures only ~0.5-1% of it. Weighted EC still loses to Frozen HxV on 3/5 datasets. Motivated the conflict-resolved follow-up above.

Evidence: `experiments/affinity_weighted_expert_choice_hv/report.md`; `project_memory/experiments/2026-08-30_affinity_weighted_expert_choice_hv.md`.

## Window-Dependent Expert-Choice H×V Routing (2026-08-30)

`experiments/window_dependent_expert_choice_hv/` tests whether making the static Expert-Choice competence score `S[h,v,e]` window-dependent (`S[t,h,v,e] = static_gain + predicted_residual_gain[t,h,v,e]`, one shared scorer, strict causal 4-fold OOF, fit-only affinity calibration) makes expert-side H×V allocation beat matched cell-side Top1 allocation using the identical score/affinity tensor. **Provenance note:** this implementation was found already on disk, uncommitted, with no prior `project_memory` entry; it was verified and reproduced directly from its raw stored artifacts (not trusted from any prior summary) before being backfilled into memory. Recomputed win counts: router-val EC beats Token on 4/5 datasets (ETTh2 the sole loser), router-train OOF EC beats Token on 3/5 (ETTh2 and Electricity lose OOF despite Electricity winning on router-val). Classification `WINDOW_DEPENDENT_EC_SUPPORTED`, but two of six predeclared criteria clear the bar only at the exact minimum (3/5) — treat as real but fragile support, not decisive. Dynamic EC beat Frozen HxV on 0/5 datasets. Static parity vs `experiments/expert_choice_hv/` reproduces exactly (0.0 diff); all integrity checks passed; no test access.

Evidence: `experiments/window_dependent_expert_choice_hv/report.md`; `project_memory/experiments/2026-08-30_window_dependent_expert_choice_hv.md`.

## Expert-Choice Horizon-Variable Routing (2026-08-30)

Completed validation-only `experiments/expert_choice_hv/` on ETTh1, ETTh2, ETTm1, Weather, and Electricity using only router_train/router_val caches and existing frozen selected cores. Final classification: `MIXED_EXPERT_CHOICE`. EC-HVR used the train-only score tensor `score[h,v,e] = mean_t(equal_error[t,h,v] - expert_error[t,h,v,e])`, then reversed allocation so each expert claimed its top HxV cells at predeclared capacities `CF=1.0` and `CF=2.0`. Matched TokenChoice Top1/Top2 controls used the exact same score tensor with cell-to-expert direction.

Results: EC CF1 beat Token Top1 by point estimate on `5/5` datasets, with block-24 support on ETTm1 (`-0.008954`, CI `[-0.010476,-0.007360]`), Weather (`-0.004069`, CI `[-0.005095,-0.002977]`), and Electricity (`-0.003746`, CI `[-0.005365,-0.002165]`). EC CF2 beat Token Top2 on only `2/5` datasets, with supported gains on Weather and Electricity but significant regressions on ETTh1 and ETTm1. EC CF1 was worse than existing Frozen HxV on every dataset. Claim masks were non-identical and passed the predeclared specialization rule (average pairwise EC Jaccard `<0.98` for both capacities on all datasets).

Interpretation: expert-to-cell allocation has a real primary-budget direction signal and creates distinct HxV claim regions, but static EC-HVR is not strong enough to replace Frozen HxV or justify a learned input-dependent Expert-Choice router yet. No test cache/file was loaded or scored; target-corruption, targetless prediction, validation-order invariance, frozen-allocation, and no-test checks passed.

## Expert-Choice HxV Pilot (2026-08-30)

Completed validation-only `experiments/behavioral_competence/expert_choice_hv_pilot/` on Electricity using only `router_train_20_60` and `router_val_60_80` caches with the fixed core `PatchTST+iTransformer+TimesNet`. Final verdict by the requested decision rule: `STRONG GO` for the predeclared `Expert Choice cap 1.25` variant versus `Hard Normal HxV`. Both methods used the exact same router_train score tensor `score[k,h,v] = -mean_train_normalized_abs_error[k,h,v]`; only the allocation mechanism changed.

Results: Equal ensemble MAE `0.214457`; existing soft/causal HxV MAE `0.211775`; static Hard Normal HxV MAE `0.222761`; Expert Choice cap `1.00` MAE `0.222948`; cap `1.25` MAE `0.220627`; cap `1.50` MAE `0.222734`. Cap `1.25` improved Hard Normal HxV by `-0.002134` MAE with block-24 CI `[-0.003012, -0.001256]`, probability delta<0 `1.000`, and `12/12` phase agreement. No-capacity Expert Choice was exactly identical to Hard Normal HxV, confirming any difference comes from capacity constraints. Important caveat: even the best Expert Choice variant is worse than Equal and the existing soft/causal HxV reference, so this supports pursuing allocation constraints only as a component inside a stronger HxV router, not replacing the current Electricity HxV baseline.

Integrity: no test cache/file was loaded; expert ordering was verified as `DLinear, PatchTST, iTransformer, TimesNet, ModernTCN`; assignments were train-only; checkpoints were unchanged.

## Rolling-Origin Revision Embedding (2026-08-30)

Completed strict validation-only `experiments/behavioral_competence/rolling_origin_revision_embedding/` on Traffic and ETTm2 using only router_train/router_val frozen forecast caches. Final classification: `NEGATIVE_RESULT`. The experiment measured real historical forecast-origin revisions with lags `[1, 2, 4]`, preserved signed lag/horizon trajectories through a deterministic compact variable projection, trained small learned ContextEmbed/RevisionEmbed/ContextPlusRevision competence encoders with purged chronological router_train OOF predictions, and evaluated router_val once with fixed rank routing weights `[0.5, 0.3333, 0.1667]`.

Results: ContextPlusRevision improved ContextEmbed OOF R2 on ETTm2 (`+0.071681`) but regressed on Traffic (`-0.070834`). The mandatory `RevisionEmbedding -> ContextEmbed residual` OOF diagnostic was negative on both datasets: ETTm2 R2 `-0.157214`, Traffic R2 `-0.021563`. Router_val rank-routing MAE improved only by point estimate on both datasets (ETTm2 `-0.000370`, Traffic `-0.000332`), which is insufficient because the competence/residual and expert-specific controls failed. Do not promote rolling-origin revision embeddings to router integration or test evaluation.

Integrity: no test cache/file was loaded; router_train OOF folds passed the horizon-12 purge; router_train-to-router_val observability held; router_val target-corruption feature invariance passed; all features/predictions were finite; checkpoint hashes were unchanged.

## Expert-Native Latent Competence (2026-08-29)

Completed strict validation-only `experiments/behavioral_competence/expert_native_competence/` on ETTh1, ETTh2, ETTm1, Weather, and Electricity using only router_train/router_val caches and existing frozen K=3 cores. Final classification: `MIXED_SUPPORT`. The study extracted frozen expert-native internal representations with temporary hooks, used purged chronological OOF Ridge/LogisticRegression readouts, compared Passive, Hidden Only, Passive+Hidden, Shuffled Hidden, Raw Forecast Control, Matched-Dimension Passive Control, and prototype-axis geometry, and evaluated router_val once after train-only PCA choices. Passive+Hidden improved Passive R2 on ETTh1 (`+0.016825`) and Electricity (`+0.342262`), but regressed on ETTh2 (`-0.178095`), ETTm1 (`-0.027923`), and Weather (`-0.002443`). AUROC improved on Weather (`+0.036025`) and Electricity (`+0.032966`) but regressed elsewhere. Do not build an Expert-Native router yet.

Integrity: no test cache/file was loaded; checkpoint hashes were unchanged; hooked-vs-unhooked prediction difference was `0.0`; router_train OOF folds passed the horizon-12 purge; target-corruption feature and prediction invariance passed. Cached-forecast reproduction is recorded as a diagnostic: most experts were near-exact, while TimesNet had rare device/batch-size-sensitive outliers on Weather/Electricity despite exact hook invariance. Treat this as a numeric reproduction caveat, not a permission to touch test data.

## Signed Pair Residual Fault Gate (2026-08-28)

Completed validation-only `experiments/behavioral_competence/pair_residual_fault_gate/` on ETTh1, ETTh2, ETTm1, Weather, and Electricity using router-train/router-val frozen forecast caches only. Final classification: `WEAK_OR_INCONSISTENT_PARITY_FAULT_SIGNAL`. The study tested signed expert-pair residual/parity features for relative bust detection, with q80/q90 router_train-only fault thresholds, purged chronological OOF detectors, train-only gamma/intervention-threshold selection, Passive, Passive+Parity, shuffled parity, and Passive+Raw forecast controls. ETTh2 was supportive (`Passive+Parity` improved Baseline MAE by `-0.001258` and beat Passive/Raw/Shuffled by point estimate), but ETTh1, ETTm1, and Electricity regressed versus Baseline, and Weather's best improvement came from Raw Forecast Control. Do not promote to router integration or test evaluation.

## Natural Capability-Demand Matching (2026-08-28)

Completed validation-only `experiments/behavioral_competence/capability_demand_matching/` on ETTh1, ETTh2, ETTm1, Weather, and Electricity using router-train/router-val frozen forecast caches only. Final classification: `CAPABILITY_SIGNAL_BUT_NO_MATCHING_GAIN`. The mechanism used history-only demand axes (`trend`, `seasonality`, `frequency`, `volatility`, `shift`, `crossvar`) and natural router_train-derived expert capability profiles with purged chronological OOF construction. CapabilityMatch showed meaningful competence association and consistently beat expert-profile shuffles, but it did not consistently beat Passive ABC or the FAME-style direct demand baseline. Do not promote to router integration or test evaluation.

ETTh2 integrity note: the previous cached-forecast/runtime discrepancy is resolved. ETTh2 caches store histories/targets/predictions in DLinear scaler-normalized units; calling the runtime wrapper on those already-normalized histories caused the prior large reproduction mismatch. Direct normalized-cache inference and de-normalized raw-history wrapper inference both reproduce cached ETTh2 forecasts within `<= 9.54e-07`. See `experiments/behavioral_competence/capability_demand_matching/etth2_integrity_audit.json`.

## Structured Forecast Repair Study (2026-08-28)

Completed validation-only `experiments/behavioral_competence/structured_forecast_repair/` on ETTh1, ETTh2, ETTm1, Weather, and Electricity using router-train/router-val frozen forecast caches only. Final classification: `WEAK_OR_AMBIGUOUS`. RepairGeometry improved relative-competence Ridge MAE over Passive on 2/5 datasets; REP was a strong control on Electricity; shuffle and block-24 evidence was mixed. No test cache was loaded. Do not promote to router integration or test evaluation. The ETTh2 reproduction issue noted in that report was later resolved by the capability-demand audit as a normalized-history/runtime-wrapper convention mismatch, not cache corruption.

Read this first. It is a compact project memory for the COSTAR-TS research branch in this BasicTS repository.

## Current Research Goal

Improve ETTh1 multivariate forecasting by combining frozen expert forecasts with adaptive COSTAR-style weighting/routing, while keeping chronological train/validation/test separation clean. The current target from recent prompts is validation MAE `<= 0.3619`; this has not been reached.

## Latest Counterfactual Forecast Revision Experiment

CONFIRMED RESULT:

`experiments/behavioral_competence/counterfactual_forecast_revision/` completed on 2026-08-27.

Question:

When a frozen forecasting expert is shown a controlled hypothetical realization of the first few future steps, does the way it revises the remainder of its forecast reveal expert-specific instance-level competence beyond passive A+B+C features?

Result:

- Final predeclared classification: `CFR_SIGNAL_BUT_REDUNDANT`.
- CFR/RelativeCFR had competence association by the fixed router-val signal rule on `4/4` datasets.
- `PassivePlusCFR` or `PassivePlusRelativeCFR` improved router-val conditional-error MAE by point estimate on `4/4` datasets: ExchangeRate `-0.001615`, Traffic `-0.002823`, BeijingAirQuality `-0.000474` via RelativeCFR, ETTm2 `-0.000012` via RelativeCFR.
- Dependence-aware support existed for some passive-plus deltas (`ExchangeRate`, `Traffic`, `BeijingAirQuality`), but the sign was not uniformly favorable: `PassivePlusCFR` significantly regressed on `BeijingAirQuality` and `ETTm2`.
- Correct CFR expert mapping beat shuffled mapping on `3/4` datasets by point estimate.
- The mandatory direct Passive-residual test was positive on only `1/4` datasets (`ETTm2`; CFR R2 `0.0197`, RelativeCFR R2 `0.0124`) and negative on ExchangeRate, Traffic, and BeijingAirQuality.

Integrity:

- No test data accessed.
- Checkpoint hashes unchanged and all experts remained frozen.
- Router-val targets were never used for fitting or surprise-scale estimation.
- Fold-specific surprise scales used fold training windows only.
- Target corruption left CFR features unchanged exactly (`max_abs_diff = 0.0`).
- OOF purge, absolute-horizon alignment, deterministic CFR regeneration, deterministic shuffle, and finite-feature checks passed.

Artifacts:

- `experiments/behavioral_competence/counterfactual_forecast_revision/report.md`
- `experiments/behavioral_competence/counterfactual_forecast_revision/validation_results.json`
- `experiments/behavioral_competence/counterfactual_forecast_revision/integrity_checks.json`
- `project_memory/experiments/2026-08-27_counterfactual_forecast_revision.md`

Decision:

Do not freeze CFR for untouched-dataset testing yet. Treat it as signal-bearing but redundant/insufficient because it fails the direct passive-residual criterion on `3/4` datasets.

## Latest Conditional Nuisance Invariance Experiment

CONFIRMED RESULT:

`experiments/behavioral_competence/conditional_nuisance_invariance/` completed on 2026-08-27.

Question:

Does the canonical expert-conditioned LearnedProbe active response contain expert-competence information after passive features and explicit nuisance features are controlled?

Result:

- Final predeclared classification: `MIXED_CNI`.
- `Electricity` showed the strongest CNI-surviving evidence: `Passive+Nuisance+Probe` improved pairwise competence over `Passive+Nuisance` by `+0.088410` and had block-24 support.
- `ETTh1` and `ETTh2` had smaller positive `Passive+Nuisance+Probe` deltas (`+0.020075`, `+0.006525`) but no block-24 support and failed negative controls.
- `ETTm1` was essentially flat after nuisance (`+0.000117`) and `Weather` regressed (`-0.011567`).
- Shuffled/wrong-expert controls and environment transfer did not support a robust cross-dataset active mechanism.

Integrity:

- No test data accessed.
- Checkpoint hashes unchanged and frozen experts remained frozen.
- Router-val targets not used during feature construction or residualization.
- Target corruption left recomputed features, scores, weights, and final predictions unchanged with max absolute diff `0.0`.
- Router-train to router-val observability held on every dataset.

Artifacts:

- `experiments/behavioral_competence/conditional_nuisance_invariance/results/report.md`
- `experiments/behavioral_competence/conditional_nuisance_invariance/results/results.json`
- `experiments/behavioral_competence/conditional_nuisance_invariance/results/integrity_checks.json`
- `project_memory/experiments/2026-08-27_conditional_nuisance_invariance.md`

## Latest Dynamics-Alignment Mechanism Experiment

CONFIRMED RESULT:

`experiments/data_model_dynamics_alignment/` completed on 2026-08-27.

Question:

Does agreement between observed local dynamics and a frozen forecaster's implied local dynamics predict upcoming expert error?

Result:

- Final predeclared classification: `WEAK_OR_AMBIGUOUS`.
- Only `Weather` passed all criteria A-E.
- Passive+Align routing MAE improved over Passive on `ETTh2`, `Weather`, and `Electricity`, but regressed on `ETTh1` and `ETTm1`; the `ETTm1` regression was block-24 significant.
- Direct `D_align` Spearman with excess/error was positive on `ETTh1`, `ETTm1`, `Weather`, and `Electricity`, but negative on `ETTh2`.
- Passive-residual R2 was small/positive only on `ETTh1` (`0.0033`) and `Electricity` (`0.0090`), negative on the other three.

Integrity:

- No test data accessed.
- Checkpoint hashes and frozen expert parameter fingerprints unchanged.
- Router-val targets not used during fitting.
- Target corruption left features unchanged.
- All model features finite. ETTm1 had `438` nonfinite condition-number diagnostics, recorded but unused by scorer features.

Artifacts:

- `experiments/data_model_dynamics_alignment/report.md`
- `experiments/data_model_dynamics_alignment/results.json`
- `experiments/data_model_dynamics_alignment/integrity_checks.json`
- `project_memory/experiments/2026-08-27_data_model_dynamics_alignment.md`

## Latest Behavioral-Competence Experiment

CONFIRMED RESULT:

V2-compatible artifact reproduction plus V3A reproduced raw-response representation test completed on 2026-08-25.

Phase A:

- Created `experiments/behavioral_competence/controlled_discriminative_probe_v2_reproduction/`.
- Reran the archived V2 protocol in a separate directory, without modifying frozen V2.
- Saved fold/final generator and scorer checkpoints plus OOF/router-val raw response artifacts.
- Reproduction gate: `REPRODUCTION_ACCEPTED`.
- Observable checks: `317/317` passed.
- Reproduced qualitative V2 result: `ACTIVE_SIGNAL_BUT_REDUNDANT`, `proceed_to_router_integration=false`.
- Source-provenance caveat: the committed V2 implementation is the archived reproduction source; original run HEAD was `2904e28`, while then-uncommitted V2 source was later committed in `7ec1f1e`. Bit-exact original source provenance is not claimed.

Phase B:

- Created `experiments/behavioral_competence/raw_response_probe_v3a_reproduced/`.
- Used accepted frozen-protocol V2 reproduction artifacts, not exact original V2 tensors.
- Fixed `Ridge(alpha=1.0)`, train-only standardization, no tuning.
- Classification: `SIX_STATS_NOT_THE_BOTTLENECK`.
- Counts: raw better than six-stat `1/4`; raw better than shuffled `2/4`; Passive+Raw better than Passive `1/4`; positive passive-residual R2 `0/4`.
- Interpretation: full raw forecast responses do not explain V2's redundancy by exposing a missed representation bottleneck.
- Test accessed: no.

Artifacts:

- `experiments/behavioral_competence/controlled_discriminative_probe_v2_reproduction/report.md`
- `experiments/behavioral_competence/controlled_discriminative_probe_v2_reproduction/reproduction_decision.json`
- `experiments/behavioral_competence/raw_response_probe_v3a_reproduced/report.md`
- `experiments/behavioral_competence/raw_response_probe_v3a_reproduced/method_manifest.json`
- `project_memory/experiments/2026-08-25_v2_reproduction_v3a_reproduced.md`

CONFIRMED RESULT:

Follow-up V3A feasibility audit:

- `experiments/behavioral_competence/raw_response_probe_v3a/` was created on 2026-08-25.
- Status: `BLOCKED_MISSING_FROZEN_V2_OOF_RAW_RESPONSE`.
- Reason: V2 saved router-val learned deltas and OOF six-stat responses, but did not save OOF learned deltas, OOF full raw response tensors, or trained V2 generator checkpoints.
- Running V3A's primary OOF raw-response comparison would require retraining `SharedControlledProbeGenerator`, which violates the V3A hard rule.
- Test accessed: no. V2 generator retrained: no. Experts retrained: no.
- Evidence: `experiments/behavioral_competence/raw_response_probe_v3a/report.md`; `project_memory/experiments/2026-08-25_raw_response_probe_v3a_blocked.md`.

On 2026-08-25, `controlled_discriminative_probe_v2` completed across `ExchangeRate`, `Traffic`, `BeijingAirQuality`, and `ETTm2`.

Question:

Can a shared learned intervention `delta_t = G(X_t)`, applied identically to every frozen expert on a window, reveal instance-specific conditional competence that passive observations do not already provide?

Result:

- Predeclared tier: `ACTIVE_SIGNAL_BUT_REDUNDANT`.
- Predeclared criteria met: `1/6`.
- Proceed to router integration: `false`.
- All four integrity suites passed: same-question invariant, purged-OOF causality, checkpoint unchanged/frozen experts, target-corruption invariance, and no test-cache access.

Interpretation:

The active shared probe has some competence-related correlation, but the signal is redundant with passive features. MatchedPassive is stronger by router-val conditional MAE on `Traffic`, `BeijingAirQuality`, and `ETTm2`, and active features do not predict MatchedPassive residuals on any dataset. Do not continue to TimeFuse/FFORMA/router integration for this active-probing formulation.

Artifacts:

- `experiments/behavioral_competence/controlled_discriminative_probe_v2/report.md`
- `experiments/behavioral_competence/controlled_discriminative_probe_v2/validation_results.json`
- `experiments/behavioral_competence/controlled_discriminative_probe_v2/prompt_compliance_audit.md`
- `project_memory/experiments/2026-08-25_controlled_discriminative_probe_v2.md`

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
| ETTh1 | Historical preregistered full adaptive reference | `0.326395` | `0.267509` | `0.363112` | historical final model |
| ETTh2 | Best single `DLinear` | `0.301708` | `0.222694` | `0.280957` | canonical single reference |
| ETTh2 | Train-selected fixed core `DLinear+PatchTST+ModernTCN` | `0.304642` | `0.225185` | `0.280878` | router-train selected core |
| ETTh2 | Full frozen adaptive | `0.297808` | `0.218612` | `0.276832` | preregistered final model |
| ETTh2 | `DLinear+ModernTCN` | `0.299263` | `0.221853` | `0.275229` | validation-selected reference only |

Conclusion:

The frozen adaptive model's relative MAE gain survived test on both datasets versus its own train-selected fixed core. ETTh2 also beat the validation-selected `DLinear+ModernTCN` reference on test, although ETTh2 absolute test metrics were worse than validation for every reported method.

Current main full adaptive model:

- ETTh1: equal-static full adaptive COSTAR, validation MAE/MSE `0.363100` / `0.306026`, after-final-test audit MAE/MSE `0.326408` / `0.267378`.
- ETTh2: preregistered full frozen adaptive model, validation MAE/MSE `0.276832` / `0.167280`, test MAE/MSE `0.297808` / `0.218612`.

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

## Published Baseline Test Audit

POST-HOC COMPARATIVE AUDIT:

On 2026-08-18, the published-baseline comparison suite was evaluated on the already-seen ETTh1/ETTh2 test caches under frozen configurations from `experiments/published_baseline_comparisons/`. No hyperparameters or method choices were changed from test feedback.

Artifacts:

- `experiments/published_baseline_test_audit/PUBLISHED_BASELINE_TEST_AUDIT_RESULTS.json`
- `experiments/published_baseline_test_audit/PUBLISHED_BASELINE_TEST_AUDIT_REPORT.md`
- `experiments/published_baseline_test_audit/published_baseline_test_results.csv`
- `project_memory/experiments/2026-08-18_published_baseline_test_audit.md`

Best audited rows:

| Dataset | Best audited method | Test MAE | Test MSE | Online COSTAR test MAE |
|---|---|---:|---:|---:|
| ETTh1 | Frozen COSTAR + MLP residual | `0.326047` | `0.267322` | `0.326408` |
| ETTh2 | Bates-Granger | `0.296294` | `0.217423` | `0.297808` |

Interpretation:

This audit shows that the published-baseline Bates-Granger adaptation is strongest on ETTh2 among the audited rows, and the MLP residual row remains strongest on ETTh1. Because test results were already known before this audit, these results are comparative evidence only and do not supersede the preregistered final-test record.

## After-Final-Test Audit: Six Previously Untested Published-Baseline Methods

ADDITIONAL AFTER-FINAL-TEST AUDIT:

On 2026-08-18, the six published-baseline methods that had validation results but no prior test evaluation (Equal all-5 ensemble, Granger-Ramanathan, Bates-Granger, FAME adaptation, TimeRouter adaptation, OneNet-style frozen-expert adaptation) were evaluated once on the canonical ETTh1/ETTh2 test caches using configs frozen before validation. Frozen/Online COSTAR were included only as reference rows, not re-tuned.

Results:

| Method | ETTh1 Test MAE | ETTh2 Test MAE |
|---|---:|---:|
| Equal all-5 ensemble | `0.332001` | `0.322330` |
| Granger-Ramanathan | `0.340765` | `0.298419` |
| Bates-Granger | `0.327848` | `0.296294` |
| FAME adaptation | `0.331314` | `0.298372` |
| TimeRouter adaptation | `0.328178` | `0.306324` |
| OneNet-style adaptation | `0.330721` | `0.407526` |

Reference: Online COSTAR test MAE `0.326408` (ETTh1) / `0.297808` (ETTh2); Frozen COSTAR `0.327175` / `0.300574`.

Online COSTAR beats all six on ETTh1 test. On ETTh2 test, Bates-Granger, FAME, and Granger-Ramanathan all beat Online COSTAR; OneNet collapses badly on ETTh2 test (MAE `0.407526`, worst row in the table).

All 10 leakage/causality checks passed: exact target-replacement invariance for GR/Bates-Granger/FAME/TimeRouter, and an exact-boundary future-target perturbation causality test for OneNet (first legally-influenced prediction matched the first prediction that actually changed, with zero leakage into the unaffected prefix).

Artifacts:

- `experiments/published_baseline_test_audit/run_after_final_test_audit.py`
- `experiments/published_baseline_test_audit/TEST_RESULTS.json`
- `experiments/published_baseline_test_audit/TEST_REPORT.md`
- `experiments/published_baseline_test_audit/leakage_and_causality_checks.json`
- `experiments/published_baseline_test_audit/cache_provenance.json`
- `project_memory/experiments/2026-08-18_after_final_test_audit_six_methods.md`

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

## Equal-Static COSTAR Test Audit

ADDITIONAL AFTER-FINAL-TEST AUDIT:

After explicit user authorization on 2026-08-17, the active ETTh1 equal-static full adaptive COSTAR path was evaluated on the already-generated ETTh1 final test cache. No tuning, expert-set changes, or hyperparameter changes were made after loading test.

This equal-static path is now the main full adaptive COSTAR implementation for ETTh1 going forward. The older preregistered result remains the historical confirmatory final-test record.

Result:

- Equal-static full adaptive COSTAR test MAE/MSE: `0.326408` / `0.267378`.
- Validation MAE/MSE: `0.363100` / `0.306026`.
- Difference vs train-selected fixed core test MAE: `-0.000720`.
- Difference vs old preregistered full adaptive test MAE: `+0.000013`.

Interpretation:

The equal-static cleanup preserves nearly the same ETTh1 test MAE as the old neural-static-prior path and slightly improves test MSE, but it is an after-final-test audit row and does not replace the preregistered final-test result.

Artifacts:

- `experiments/equal_static_costar_test_audit/MAIN_ETTH1_FULL_ADAPTIVE_MODEL.json`
- `experiments/equal_static_costar_test_audit/EQUAL_STATIC_ETTH1_TEST_AUDIT.json`
- `experiments/equal_static_costar_test_audit/EQUAL_STATIC_ETTH1_TEST_AUDIT.md`
- `project_memory/experiments/2026-08-17_equal_static_costar_test_audit.md`

## Published Baseline Comparisons

VALIDATION-ONLY COMPARISON:

On 2026-08-17, published comparison baselines were implemented over the same frozen COSTAR expert forecasts. Hyperparameters were selected with chronological router-train folds, selected configs were written before validation, and no test cache was loaded.

Main validation results:

- Granger-Ramanathan: ETTh1 `0.382960` / `0.336499`, ETTh2 `0.276704` / `0.165286`.
- Bates-Granger: ETTh1 `0.368891` / `0.309925`, ETTh2 `0.274915` / `0.165315`.
- FAME adaptation: ETTh1 `0.379212` / `0.326919`, ETTh2 `0.277008` / `0.167165`.
- TimeRouter adaptation: ETTh1 `0.368234` / `0.309054`, ETTh2 `0.283288` / `0.175959`.
- OneNet-style frozen-expert adaptation: ETTh1 `0.370137` / `0.314488`, ETTh2 `0.402666` / `0.394105`.

Interpretation:

None of the new published baselines beat the main ETTh1 online COSTAR validation result. ETTh2 Bates-Granger is a strong new validation baseline and beats ETTh2 online COSTAR MAE, but this is validation-only and should not be tuned further using seen test metrics.

Artifacts:

- `experiments/published_baseline_comparisons/FINAL_REPORT.json`
- `experiments/published_baseline_comparisons/PUBLISHED_BASELINE_COMPARISON_REPORT.md`
- `project_memory/experiments/2026-08-17_published_baseline_comparisons.md`

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
