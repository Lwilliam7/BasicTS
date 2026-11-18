# Choose baselines by the question they answer

This proposal chooses ways to combine the existing five frozen forecasts. Keep CellShrink (`wide_top3`) primary, with 192 steps main and 12 and 96 beside it.

## Recommended roster

Choose comparisons that challenge each part of the method. Readiness requires evidence for the exact pool, horizon, implementation version, and comparison.

| Priority | Comparison | What it tests | Inclusion condition |
|---|---|---|---|
| Required | `bates_all5` | Whether CellShrink improves on a strong fixed combination | Retain as the existing primary comparator. Keep its purged-fold sensitivity beside it. |
| Required | `equal5` | Whether learning weights earns its complexity | Same five experts and windows. |
| Required | `best_single_all5` | Whether combination beats choosing one model | Choose the model on training data, never evaluation errors. Do not substitute the core-only `best_single`. |
| Required | `equal3` and `bates_core3` | Whether selecting the dataset's core explains the gain | Select the core using that pool's `router_train` only. |
| Required | `frozen_hv` | Whether fixed patterns across forecast steps and variables explain the gain | Verify exact implementation and both metrics in the matching pool. |
| Required controls | `constant` and `static_hv` | Whether changing the blend with the current window adds value | Label as CellShrink mechanism controls, not external methods. Preserve the existing gate experiment. |
| Next external comparison | FFORMA-MAE adaptation | Whether conventional feature-conditioned weighting is enough | Use the existing conditional runner and name its changed loss and window-level feature adaptation. |
| Next generic control | Conditional MLP | Whether a direct learned combination competes without CellShrink's construction | Retain its declared training budget. Name it as a generic window-level control. |
| Cell-level challenge | EMTSF-style adaptation | Whether dense weights at each forecast cell compete with subset-plus-average | Require the versioned full rerun and trained-fit checks described below. Never label it full EMTSF. |

Retain conditional-error softmax, Granger-Ramanathan where matching outputs exist, and all existing conventional comparators in complete supplementary tables. Do not discard them because they beat CellShrink. Narrow and wide_soft remain method variants, not extra external baselines.

FFORMA is a justified first external addition because its original purpose is feature-conditioned model averaging ([author description](https://fpp.robjhyndman.com/publications/fforma/)). Its local adaptation and the conditional MLP already share a runner and declared five-configuration, four-fold selection procedure ([local guide](C:/Data/Documents/Code/BasicTS/scripts/compare_conditional_combinations.md)). This saves implementation work without inventing a new baseline.

## Methods that need separate treatment

FAME and TimeRouter are relevant literature challengers: [FAME](https://arxiv.org/abs/2606.08896) uses forecastability-aware sparse routing, and [TimeRouter](https://arxiv.org/abs/2606.11625) combines routing, a selective gate, and fallback. They remain candidates for a later audit, not cleared paper rows. The explicit October 25 decision withholds both until fidelity and protocol compatibility are supported. A synthetic CPU pass does not reverse that decision. Reusing the author's foundation-pool checkpoint on five different experts would also be invalid.

Keep OneNet OCP in a separate delayed-feedback comparison if pursued. Its predictions use feedback unavailable to an offline router. The current Super-Linear port covers a spectral gate rather than the complete expert-bank model. Neither belongs in the main offline frozen-pool ranking under its published method name ([port record](C:/Data/Documents/Code/BasicTS/project_memory/experiments/2026-10-25_baseline_port_health.md)).

The EMTSF record has newer evidence than its headline. Electricity's original gate stayed untrained. Amendment 2 found non-finite GPU gradients, and its proposed normalization change affects every dataset. That diagnostic failed its stated criterion. The user chose a new amendment and a full 14-unit rerun. Require that version's report and fit evidence before presenting a comparison against a functioning gate. Preserve the original result and disclosures ([record](C:/Data/Documents/Code/BasicTS/project_memory/experiments/2026-10-27_emtsf_style_baseline.md)).

Reject spatial routers without meaningful channel neighbourhoods and jointly trained expert systems as direct peers. Published leaderboard values under another split are context, not numbers to subtract from this experiment. Cached all-expert forecasts cannot establish sparse inference savings.

## Reusable arena procedure

1. **Freeze the comparison contract.** Record expert identities, checkpoint hashes, 96-step history, split roles, horizons, origins, masks, normalization, seeds, selection budgets, and output paths. Read existing evidence only. Do not access the locked test block.
2. **Request three independent proposals.** Give each reviewer the same evidence and prompt below. Each proposes a minimal roster, identifies missing evidence, and rejects alternatives for methodological reasons.
3. **Judge before seeing new forecast scores.** Assess claim coverage, implementation fidelity, information fairness, feasibility, and clarity. Choose a base roster and incorporate justified missing controls. An inclusion decision must not depend on CellShrink winning.
4. **Audit eligibility.** Map each candidate's paper equations to the actual implementation. Inspect selected checkpoints, finite losses and gradients, target timing, expert-column permutation, planted predictable winners, and frozen-parameter hashes. Record pass, blocked, adaptation-only, or incompatible. Uniform weights can be legitimate; initialization retained because training failed is different.
5. **Declare the empirical comparison separately.** For eligible methods, select configurations only inside chronological training folds. Evaluate every declared dataset and seed with paired per-window MAE and MSE. Average seed losses. Keep the existing primary comparator, statistical rules, and development-only label. Report incomplete units, too-few-block intervals, failures, and losses.
6. **Stop.** Stop planning when the roster and evidence gaps are explicit. Stop a candidate at a failed validity check rather than tuning on evaluation scores. A repair receives a versioned amendment. Do not wait for the separate three-dataset rebuild or launch research work as part of this document.

Reusable prompt:

> Recommend the smallest defensible frozen-pool baseline roster for CellShrink. Use the supplied repository evidence and primary papers. For each method state the scientific question, exact adaptation, information available at prediction time, readiness evidence, and remaining work. Keep strong fixed combinations and a functioning cell-level challenger. Separate mechanism controls, published adaptations, online feedback, and joint-training systems. Do not select for favorable CellShrink results, invent measurements, access test data, or launch experiments. Finish with the roster, exclusion reasons, and the next verifiable action.

Next, inventory matching outputs and training diagnostics, then complete the existing conditional comparison and versioned EMTSF rerun through their own workflows.
