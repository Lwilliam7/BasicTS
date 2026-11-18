# An arena for choosing CellShrink's frozen baselines

Use the arena to choose credible comparisons, not opponents CellShrink can beat. Keep the five forecasting models fixed. Compare the rules that combine their predictions. This proposal selects a roster and a verification order. It reports no new forecasting result.

## Recommended roster

| Priority | Baseline | Reason to include it | Admission condition |
|---|---|---|---|
| Required | Equal average of all five, `equal5` | Tests whether learning helps at all. | Recompute losses from the exact shared forecasts. |
| Required | Equal average of the selected core three, `equal3` | Measures the value of core selection. | Core chosen on router training data only. |
| Required | Best single among all five, `best_single_all5` | Tests selection versus combination. | Choose the expert on training data, never evaluation labels. Do not substitute core-only `best_single`. |
| Required | Bates all-five, `bates_all5` | Preserves the project's declared primary comparison. | Match pool, normalization, origins, and existing configuration rule. |
| Required, complete supporting table | `bates_core3`, `bates_all5_purged`, `frozen_hv` | Strong fixed combinations and the training-fold sensitivity check. | Preserve each method's actual scope and fit rule. |
| Required mechanism controls | `constant`, `static_hv`, `router_only` | Test whether adaptive blending earns its complexity. | Reuse the gate experiment's matched outputs; keep these separate from published baselines. |
| First conditional addition | FFORMA-MAE adaptation | Tests feature-conditioned weights without CellShrink's cell-specific design. | Use the existing isolated runner and disclose the changed loss and window features. |
| Same conditional experiment | Conditional error softmax and conditional MLP | Check whether a general tree or neural combiner explains the gain. | Include all three declared families. Do not keep only whichever family loses. |
| Cell-level challenger, pending | Corrected EMTSF-style adaptation | Closest implemented challenge to per-cell combination. | Complete the already decided all-unit correction and audit trained fits. Preserve the original run and its disclosure. |
| Later published-method additions | TimeRouter adaptation, then FAME adaptation | Cover selective fallback and sparse forecastability-based selection. | Clear the existing fidelity exclusions before admitting paper rows. |

The short main table can show `equal5`, `best_single_all5`, `bates_all5`, FFORMA-MAE, the corrected cell-level challenger, and CellShrink's three variants. Keep the full static and conditional roster in accompanying tables. Preserve the rule that quoted gains identify the strongest measured comparator in each pool. Never remove a valid comparator because it wins.

Primary sources motivate coverage but do not certify local adaptations: [FFORMA](https://fpp.robjhyndman.com/publications/fforma/), [TimeRouter](https://arxiv.org/abs/2606.11625), [FAME](https://arxiv.org/abs/2606.08896), [EMTSF](https://github.com/muslehal/EMTSF).

## The reusable arena process

1. **Freeze the comparison question.** Name `wide_top3` as primary, horizon 192 as main, and 12 and 96 as supporting settings. List the existing seven pools. Keep the three-dataset extension separate until its matched pools exist. Record all previously seen results. This is a development-plan document, not a retrospective claim of predeclaration.

2. **Make one evidence card per candidate.** Record the original method and pinned source, frozen or trainable components, input history, label availability, prediction granularity, objective, configuration budget, implementation path, adaptation differences, known failures, and missing artifacts. Assign one state: ready for comparison, requires verification, or outside this comparison. A synthetic pass alone cannot promote a published-method row.

3. **Ask independent reviewers for rosters.** Give every reviewer the same evidence and request a minimal set that covers fixed weighting, conditional weighting, and cell-level weighting. Require one reason for each inclusion, one reason for every exclusion, and exact evidence that would change a decision. Reviewers must not use CellShrink's wins to choose opponents. They may use known software failures to require repairs.

4. **Judge the completed proposals.** A separate reviewer checks relevance, source fidelity, matching inputs and labels, feasibility, and selection honesty. Reject any proposal that confuses a partial port with the complete published method, treats an online algorithm as an offline peer, or promises a winner without an experiment. Pick the smallest roster with complete coverage. Retain the strongest justified additions from the other proposals.

5. **Verify readiness before performance.** Check original versus local computations, expert-column permutation, constant histories, missing-target handling, finite gradients and stopping losses, and whether fitted parameters actually change. Verify the real pool and per-window alignment separately. Stop a method at a failed check. Do not silently delete its failed datasets or rewrite the old result.

6. **Specify the future empirical comparison.** Use identical cached forecasts, 96-step histories, target masks, forecast origins, and checkpoint normalization. Tune only on chronological router-training folds with fully observable targets. The conditional runner already fixes five configurations, four folds, and five final seeds. Preserve that budget rather than reopening it after results. Report both MAE and MSE and average seed losses. Keep insufficient-block results descriptive. No locked-test access belongs to this task.

7. **Publish a complete decision table.** Include admitted and withheld methods, reasons, failures, configuration budgets, and evidence paths. Report every planned eligible dataset. State that all these periods are development data. A change to a baseline gets a dated version and its own result, not an overwrite.

## What changes the immediate work order

The latest EMTSF record says Amendment 2 failed. Standardizing before RevIN affects every dataset, not just Electricity. The user has already decided an all-14-unit rerun under Amendment 3. That correction is the necessary cell-level challenger; a fresh, unrelated gate would not resolve it.

Next complete the existing conditional comparison rather than invent another combiner. TimeRouter and FAME stay pending under the October 25 exclusions. Their reported synthetic health is useful, but full-method fidelity remains unestablished.

## Rejected alternatives

I rejected an accuracy-only tournament because choosing opponents on reused evaluation results biases the comparison. I rejected making every recent MoE mandatory because jointly trained experts do not share this budget. OneNet OCP needs a separate delayed-feedback setting; Super-Linear's spectral gate is a component adaptation. Spatial routers need spatial data. New fixed-weight variants duplicate existing coverage. None is excluded for outperforming CellShrink.

Evidence under `C:/Data/Documents/Code/BasicTS`: `project_memory/DECISIONS.md`, `project_memory/experiments/2026-10-25_baseline_port_health.md`, `project_memory/experiments/2026-10-27_emtsf_style_baseline.md` including Amendment 2, `scripts/compare_conditional_combinations.md`, and confirmation and long-horizon comparator declarations.
