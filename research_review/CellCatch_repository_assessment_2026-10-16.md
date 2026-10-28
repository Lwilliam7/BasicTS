# CellCatch research assessment — October 16, 2026

## Recommendation

Write the paper now, with a scoped claim about frozen-pool combination and dataset-dependent MAE improvements. The next research work should isolate what the conditional gate and cell-level decisions add beyond simpler combinations. More architecture searches or another broad horizon grid are not the priority.

The evidence is sufficient for a substantive Methods and Results draft. It does not yet establish general superiority, an MSE advantage, or that the selected top-three design is uniquely justified. ICML is the target recorded in the repository; readiness for that submission depends on the remaining comparisons, novelty, and the final evaluation design. Acceptance cannot be inferred from these results alone.

The strongest candidate question is: **When does conditioning a frozen forecast combination on each horizon-variable cell improve MAE, and does learning how far to depart from a fixed average provide value beyond static shrinkage?**

## Scope and correction to the preliminary review

The research repository is C:/Data/Documents/Code/BasicTS. The earlier notebook-only review concerned a different, sparse folder and is not an assessment of this project's evidence.

This review covers the current project records and ledger, the paper and introduction, detailed confirmation/pilot/long-horizon reports and JSON results, method freeze and relevant code, supporting historical experiment records and errata, and targeted primary-literature checks. The Markdown inventory contains 93 files under project_memory and 185 under experiments after the new bundle arrived. Inventorying the archive is distinct from independently revalidating every historical experiment. The main numerical conclusions below were checked against the saved result summaries with a research review agent.

The long-horizon bundle became available locally during the review. Server-only checkpoints, cached forecasts, and full training tensors were not independently recomputed. Saved integrity checks are evidence about the run, not a substitute for reproducing it. No research experiments, new model training, or locked-test evaluation were performed. Further server checks were skipped at the user's request.

## What the evidence now establishes

### The first untouched-dataset confirmation remains mixed

The frozen primary is wide_top3: scores over all five experts, the equal average of the three highest-scoring experts at each cell, and a learned interpolation toward a router-train-selected fixed core-three average. The ambiguity in the originally pasted description is resolved by the freeze and current code.

Against the strongest reported classical comparator per dataset, at horizon 12:

| Dataset | Comparator | Primary MAE change | Reported paired result |
| --- | --- | ---: | --- |
| AppliancesEnergy | Bates all-five | -2.22% | Improvement |
| SolarAL | Bates all-five | -3.59% | Improvement |
| TetouanPower | Bates core-three | -0.57% | No detected difference |
| OikolabWeather | Bates all-five | +0.80% | Deterioration |

The predeclared primary label is **TIED_WITH_BATES**, because its rule permitted no significant loss. This label is not statistical proof of equivalence. Report the heterogeneous outcomes alongside it. A secondary arm cannot replace the primary's verdict after results are seen.

The introduction's two gains/no-detected-difference/one loss is defensible. Tetouan's -4.63% versus Bates all-five is not its gain against the stronger core-three comparator.

Retraining experts with a cap of 200 epochs and patience eight has already been completed. The follow-up retains gains on three of the four new datasets against their strongest reported comparators, including about -3.31% on Tetouan versus Frozen HxV, and retains an Oikolab loss. It is a robustness follow-up on used data, not fresh confirmation. Do not restart this training-duration check unchanged.

Sources: [confirmation report](C:/Data/Documents/Code/BasicTS/experiments/wide_top3_untouched_confirmation/report.md), [freeze](C:/Data/Documents/Code/BasicTS/experiments/wide_top3_method_freeze/FREEZE.md), [early-stopping record](C:/Data/Documents/Code/BasicTS/project_memory/experiments/2026-10-06_experts_trained_to_early_stopping.md).

### The long-horizon run has finished and its results are local

The returned queue state records completion on **October 12, 2026, at 11:07 a.m. EDT**, analysis exit code zero, 147 completed jobs and one failed job. Twenty units completed. This is a historical fact established by the returned record, not a live server-status assertion.

| Evidence | Primary result versus Bates all-five | Interpretation |
| --- | --- | --- |
| H=96 pilot, six datasets | MAE HOLDS; MSE NOT_SHOWN | Completed development evidence; Traffic subsequently added separately |
| H=192, all seven datasets | MAE HOLDS; across-dataset CONSISTENT_ADVANTAGE | Strongest completed horizon extension |
| H=336 | INCOMPLETE | Traffic scorer seed 17 failed with GPU out-of-memory; six other datasets completed |
| H=720 | UNDERPOWERED; across-dataset NOT_ESTABLISHED | Five of six completed datasets lack enough blocks for counted interval conclusions |

At H=192, the primary improves MAE on six of seven datasets by point estimate versus Bates all-five. The arithmetic mean of dataset percentage changes is -2.98%, and the median is -1.45%. The predeclared exact Wilcoxon p-value is 0.046875; the sign-test p-value is 0.125. Report the named tests and limited scope rather than treating this as decisive, universal evidence. The datasets are concentrated in related application domains and are reused development data.

The stronger-comparator reading is more informative for practical claims:

| H=192 dataset | Strongest reported comparator | Primary relative MAE change | Counted paired result |
| --- | --- | ---: | --- |
| Electricity | Bates core-three | -0.72% | Improvement |
| Traffic | Frozen HxV | -2.25% | Improvement |
| SolarAL | Bates all-five | -5.86% | Improvement |
| AppliancesEnergy | Bates all-five | -0.56% | No detected difference |
| OikolabWeather | Bates core-three | +0.18% | No detected difference |
| TetouanPower | Bates core-three | +0.28% | No detected difference |
| ETTh1 | Equal-five | +0.73% | Too few blocks for counted significance |

For example, Tetouan's -10.13% versus Bates all-five becomes +0.28%, with no detected difference, versus Bates core-three. Do not headline the larger value as the method's gain over the strongest combination.

At H=336, the primary has supported strongest-comparator MAE gains on OikolabWeather and SolarAL. Preserve INCOMPLETE until the planned missing Traffic work is resolved. At H=720, there is no adequately supported primary win against the strongest comparator: Oikolab has enough blocks but is about +0.66%, with no detected difference, versus the best single expert; the other completed units are underpowered. Lack of power is not evidence of equivalence.

MSE remains a limitation. At H=192 the primary is NOT_SHOWN on MSE, including a significant +5.17% deterioration on Traffic versus Bates all-five. Top-three reduces the Traffic MSE cost relative to wide-soft, but this does not establish that top-three is generally the best arm. At H=336 Tetouan MSE is +5.92% against Bates all-five despite lower MAE.

Traffic H=720 was excluded by the predeclared hardware rule. The run also declared window strides and smaller scorer/head batches on larger units before observing results. This is a memory-constrained extension of the frozen procedure, not a horizon-only treatment. Compact experts, custom chronological splits, and strided windows prevent direct numerical comparison with published long-horizon tables.

Saved checks report an exact reference anchor, passing integrity summaries for completed units, causal folds, unchanged checkpoints, and expert training ended by early stopping. The missing Traffic H=336 unit remains missing regardless of the passing summaries for finished units.

Sources: [long-horizon report](C:/Data/Documents/Code/BasicTS/experiments/long_horizon_protocol/report.md), [raw summaries](C:/Data/Documents/Code/BasicTS/experiments/long_horizon_protocol/results.json), [queue state](C:/Data/Documents/Code/BasicTS/experiments/long_horizon_protocol/queue_state.json), [predeclaration](C:/Data/Documents/Code/BasicTS/experiments/long_horizon_protocol/PREDECLARATION.md), [H=96 report](C:/Data/Documents/Code/BasicTS/experiments/horizon_pilot_h96/report.md).

## Highest-value next actions

1. **Reconcile the records and start the manuscript's Methods and Results.** The current-state/TODO text and introduction still describe parts of the completed work as pending. Add a dated long-horizon record, retain the failure and the original confirmation labels, and link the bundle's report and checks. Make one claims table distinguish untouched confirmation, reused development extensions, training-duration follow-ups, and historical test audits.

2. **Complete a focused matched-control experiment.** Keep the current primary, expert predictions, splits, loss/selection rules, and tuning budget fixed. Compare the core-only endpoint, the routed top-three endpoint without interpolation, a train-selected constant interpolation coefficient, a static horizon-variable coefficient without current-window conditioning, and the full learned gate. The old gamma sweep and null-expert study do not replace this comparison. Use temporal out-of-fold predictions and training-only selection; treat results on reused blocks as development findings.

3. **Close the matched conditional-baseline and granularity comparisons.** FFORMA is already implemented under behavioral_competence/fforma_probe, on ExchangeRate, Traffic, BeijingAirQuality and ETTm2 in an older core-three protocol. Reuse that work after checking fidelity; what is missing is a fair comparison on the current pools/primary and horizons, not a first implementation. The predeclared Traffic Bates granularity ladder has no result report; it concerns the older null-expert mechanism. If using it for the new paper claim, distinguish static horizon/variable structure from input-dependent routing and compare the current method directly. Audit the existing modern baseline ports with a predictable planted-signal case before making strong claims from their poor real-data performance.

4. **Recover only the missing Traffic H=336 computation if feasible.** The saved failure is one scorer-seed job, not a reason to repeat the whole grid. A same-settings retry with sufficient available GPU memory may resolve it. Preserve the failure record, completed artifacts, and source hashes. Any batch/method change needs its own disclosed amendment and cannot silently replace the planned unit. This review recommends a recovery; it did not launch one.

5. **Plan independent evaluation after settling the paper claim.** The current H=96/192/336/720 results reuse development periods. More scorer seeds do not replace independent expert pools or temporal origins. Choose a final primary and protocol before seeking the explicitly required authorization for a locked-test pass. This review is not that authorization and uses no new test outcomes.

A useful decision rule for the research: if the adaptive gate outperforms constant/static controls and a credible conditional combiner, pursue a method contribution about learnable conditional advantage. If simpler controls match it, simplify the method or develop a systematic empirical paper explaining the boundary. Do not search variants until a reused split produces another winner.

## Historical findings to preserve

- Cell-winner predictability is dataset-dependent. The existing scorer achieves 33.56% top-one accuracy on ETTh1 versus 35.64% for position-majority, and 39.00% on Electricity versus 37.01%. Winner shares near one third do not establish that routing is useful or learnable. These diagnostics are more relevant than the chance argument in the pasted draft, though they concern an earlier scorer/protocol.
- The null-average candidate helps its own earlier baseline broadly, including all six datasets and five scorer seeds in the seed audit. This supports the value of averaging within that mechanism; it does not identify the cause of the current gate's gains.
- The September 18 feature-ablation erratum supersedes older labels: cell and local groups are SUPPORTED; global is NOT_SUPPORTED. Do not cite the old all-MIXED conclusion.
- Low-rank restrictions, the tested uncertainty gate, several residual/probe mechanisms, and joint-training under nested OOF have already produced negative or limited findings. Longer training as a grokking search also failed its gate. These are reasons to avoid repeating unchanged searches.
- The archive contains historical ETTh1/ETTh2 test evaluations of earlier methods. The current experiment's no-test-access flag does not mean those datasets' test histories are universally untouched.

Sources: [cell-winner precheck](C:/Data/Documents/Code/BasicTS/project_memory/experiments/2026-09-03_cell_winner_classification_precheck.md), [feature erratum](C:/Data/Documents/Code/BasicTS/project_memory/experiments/2026-09-18_feature_ablation_classify_group_erratum.md), [seed audit](C:/Data/Documents/Code/BasicTS/project_memory/experiments/2026-09-26_ec_seed_variance_audit.md), [nested OOF](C:/Data/Documents/Code/BasicTS/project_memory/experiments/2026-09-04_nested_oof_joint_experts_ec.md), [older FFORMA](C:/Data/Documents/Code/BasicTS/experiments/behavioral_competence/fforma_probe/report.md).

## Introduction and paper corrections

The repository's fifth introduction draft is materially stronger than the pasted version: it identifies the frozen variant, qualifies the original confirmation, and acknowledges MSE. Preserve those corrections. The most important remaining changes are scientific, followed by completion of the manuscript.

1. **Update the horizon/results narrative.** H=96 is no longer pending and the long-horizon run has returned. State the positive H=192 MAE evidence, incomplete H=336 result, and insufficient H=720 evidence with their development status. Retain the original H=12 confirmation independently.
2. **Correct the prior-work contrast.** [Synapse](https://arxiv.org/html/2511.05460) already changes weights at individual forecast timestamps. [TimeRouter](https://arxiv.org/html/2606.11625v1) already combines frozen models, selective gating, and an ensemble fallback. [FFORMA](https://robjhyndman.com/papers/fforma.pdf) learns feature-conditioned forecast combinations. The defensible comparison must specify CellCatch's exact granularity, selection target, and learned core interpolation; a general claim that previous frozen routers only operate per window overlooks relevant work.
3. **Turn the untested explanation into a hypothesis.** The text says the gate withholds trust exactly where rankings are uncertain and that retaining the average is what makes fine routing work. Entropy and margin are inputs, not calibrated uncertainty guarantees. The null-expert ablation is not a removal/constant-gate control for the final method. Use "allows" or "we test whether" until matched controls and gate-behavior analyses support the stronger account. Earlier coarsening and low-rank negatives also caution against claiming that fine granularity is always excessive capacity.
4. **Match the strength of every result sentence.** The abstract placeholder in main.tex says three of four improvements, whereas the strongest-comparator original confirmation has two significant gains, one no detected difference, and one loss. The longer-training follow-up is a separate three-gain result. Name which comparison each sentence describes. At long horizons report the strongest-comparator table beside Bates all-five.
5. **Describe the method and estimand precisely.** The core is selected using router_train. Each new dataset refits the scorer, identity embeddings, and lambda head: this evaluates transfer of a frozen learning procedure, not zero-shot deployment of one trained router. The gate optimizes normalized forecast MSE; do not explain its MSE regressions by claiming that no component penalizes squared error. Five scorer-seed losses are averaged; this is not a five-model prediction ensemble and does not measure expert-pool variability.
6. **Keep protocol claims accurate.** Outer out-of-fold forecasts and target observability are checked. The inner early-stopping split is a chronological cut without a horizon purge; disclose its scope. This limitation does not by itself demonstrate leakage into the external confirmation block. Treat the auditable protocol as supporting reproducibility rather than automatically claiming a novel leakage-control method.
7. **Complete the manuscript essentials.** Methods, Experiments, Related Work, and the abstract are still placeholders. No references.bib exists. Prioritize a reproducible evidence table, method description, results/failure analysis, and verified bibliography before polishing the generic opening. Venue/style preparation comes after the content is coherent.

Suggested results framing:

> CellCatch shows dataset-dependent MAE improvements over strong combinations fitted to the same frozen expert pool. Its original untouched-dataset evaluation does not satisfy the predeclared no-loss criterion, while development extensions support MAE improvements through 192 steps. Squared-error results remain mixed, and evidence at longer horizons is limited by incomplete computation and temporal sample size.

Suggested contribution structure, conditional on the missing controls:

- A precisely specified horizon-variable combiner with a fixed equal-weight core and learned interpolation toward a predicted subset.
- A controlled comparison identifying whether cell granularity and input-dependent interpolation add value beyond simpler, properly fitted combinations.
- Evidence about the conditions and metric trade-offs under which adaptation helps or fails, with declared selection history and dependence-aware evaluation.

The practical next deliverable is a coherent full draft plus a small experiment plan that can falsify the proposed mechanism. The project already has substantial results; the remaining bottleneck is connecting them to a distinctive and adequately tested scientific claim.
