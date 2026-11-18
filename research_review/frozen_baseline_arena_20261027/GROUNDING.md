# Shared evidence for the frozen-baseline arena

User request: "can you help me generate an arena method to see which frozen baselines I should use?"

Working interpretation: choose comparison methods that consume the same frozen forecasting pool. If the user meant selecting the forecasters themselves, treat that as a different task. Keep explanations plain.

Produce a reusable arena protocol and a recommended baseline roster. No numerical research experiments are authorized as part of creating this plan. Do not call this document a prospective predeclaration for results that already exist.

## Research context

Read-only main research repository: C:/Data/Documents/Code/BasicTS.
Writable output directory for this task: C:/Users/luwil/OneDrive/Documents/Code/BasicTS/research_review/frozen_baseline_arena_20261027.

The method is named CellShrink in user decisions. Code calls the primary wide_top3. Narrow and wide_soft are secondary variants, not external baselines. Pool: DLinear, PatchTST, iTransformer, TimesNet, ModernTCN. Input history 96. Main horizon 192; horizons 12 and 96 alongside it. Experts remain frozen while combiners train. Five scorer seeds 7, 11, 13, 17, 19. Seed losses are averaged, not predictions.

The standard training protocol uses walk-forward expert stages 0-20%, 0-40%, 0-60%. Router training forecasts cover 20-60% and evaluation 60-80%. Locked test access is not part of this task. Existing evaluation data are development data; baseline choice now does not create independent confirmation.

Seven datasets currently have relevant standard pools: ETTh1, Electricity, Traffic, AppliancesEnergy, OikolabWeather, SolarAL, TetouanPower. A separate proposed extension adds ETTh2, ETTm1, Weather at 12, 96, 192, but its runner does not yet exist in the inspected directory. Do not make the baseline arena depend on rebuilding this extension.

## Local evidence to inspect

- project_memory/DECISIONS.md, dated 2026-10-27 and 2026-10-25. Primary method wide_top3, main horizon192, MSE not improved, paper-table exclusions for ports.
- project_memory/experiments/2026-10-18_gate_ablation.md. Learned gate versus static_hv: HELPS at12 under the declared interval rule, but across-dataset improvement not established; NO_GAIN at192. Fixed blending and input-dependent routing have more support. Constant and static_hv must remain visible.
- project_memory/experiments/2026-10-27_emtsf_style_baseline.md. CellShrink beats the EMTSF-style adaptation under its stated rules, but Electricity's dense baseline gate stayed at initialization at both horizons. Constant blending performs similarly to the full method. This is not full EMTSF reproduction. Inspect before claiming readiness.
- project_memory/experiments/2026-10-25_baseline_port_health.md. FAME, TimeRouter, OneNet OCP, Super-Linear gate pass synthetic CPU controls. All four still have paper_fidelity NOT_ESTABLISHED and table_eligible false. FAME CPU pass does not validate historical CUDA runs. OneNet is only its delayed-feedback OCP block on two frozen branches. Super-Linear covers its spectral gate, not the whole model.
- scripts/compare_conditional_combinations.md and .py. Existing isolated runner has FFORMA-MAE adaptation, conditional-error softmax, conditional MLP. Same frozen pool and THA history features; window-level weights. Five fixed configurations, four chronological folds, five final seeds. Generic controls are not replicas of named published methods. Default queued design seven datasets at96 and192, CPU processes; no server launch observed in the inspected record.
- project_memory/prompts/baseline_roster_completion.md. Older roster instructions include no redundant fixed-weight baselines, no online OneNet in the offline ranking, spatial methods related-work-only on non-spatial channels. Some claims are older than the 2026-10-25 exclusions; newer explicit user decisions take precedence.
- experiments/long_horizon_extension/PREDECLARATION.md. Separate development-only extension. It expressly excludes running gate ablation, modern baselines, conditional comparisons, and EMTSF-style comparisons as part of that extension.
- experiments/router_roster_paired_ci/. Existing supporting pairings may still read narrow. Do not compare them as if they were wide_top3 without checking origins and source.

Existing conventional comparator candidates include equal5, equal3, best_single, best_single_all5, bates_all5, bates_core3, bates_all5_purged, frozen_hv, Granger-Ramanathan. Verify exact availability and semantics before choosing paper rows. A core chosen on router_train is not an oracle chosen on evaluation labels.

## Primary literature leads

- FFORMA author page: https://fpp.robjhyndman.com/publications/fforma/ . Features condition model-combination weights. Local FFORMA-MAE changes the loss and adapts series features to windows.
- FAME paper: https://arxiv.org/abs/2606.08896 . Author implementation https://github.com/hit636/FAME . Sparse forecastability-aware routing by series. A rolling-window port is an adaptation.
- TimeRouter paper: https://arxiv.org/abs/2606.11625 . Author model card https://huggingface.co/nkh/timerouter-v1 . Frozen foundation-model pool, XGBoost selection, gated fallback. Replacing its pool requires fitting an adapted router; do not reuse incompatible checkpoint weights.
- EMTSF paper: https://arxiv.org/abs/2510.23396 . Author implementation https://github.com/muslehal/EMTSF . Gate adaptation is not automatically equivalent to the complete method. Whether all upstream experts stay frozen needs direct verification.

No inference-time speed or sparse-compute savings may be inferred from cached all-expert forecasts. No measured budget or winner is invented. Predictions about usefulness are recommendations.

## Later evidence checked during this arena

The full 2026-10-27 EMTSF-style experiment record has an Amendment 2 section at lines 132-162. Its diagnosis is FAIL. GPU gradients became nonfinite in the first batch with raw-scale RevIN inputs near 1e6. Standardizing before RevIN changes every dataset, so the departure from upstream affects all units. The user decided on a full 14-unit, 70-job rerun under Amendment 3, using finite-through-best-epoch checks. Treat corrected EMTSF-style as pending. Preserve the original results and their failure disclosures. Older CURRENT_STATE and TODO summaries are explicitly stale on this point.

Direct code check: wide_top3_untouched_confirmation/run_untouched_confirmation.py COMPARATORS has seven entries: bates_all5, bates_all5_purged, bates_core3, equal5, equal3, best_single, frozen_hv. long_horizon_protocol/run_long_horizon.py adds best_single_all5, chosen on router_train. best_single and best_single_all5 are not interchangeable. Granger-Ramanathan exists historically but is not in that standard eight-comparator long-horizon list.
