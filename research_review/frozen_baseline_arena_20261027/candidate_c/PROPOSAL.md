# Choose baselines by the claim they challenge

Use the arena to select a defensible comparison set, not to find methods CellShrink can beat. Keep the five forecasting models fixed. Train only the methods that select or combine their forecasts. This proposal is planning; it contains no new performance result.

## Recommended set

| Priority | Baselines | Purpose and condition |
|---|---|---|
| Keep | Equal-five, equal-core-three | Establish whether learning a combination beats averaging. Select the core using router_train only. |
| Keep | Best single of all five | Establish whether combining forecasts adds value. Select the expert on training data, never evaluation data. Retain the historical core-only best-single row in supporting tables. |
| Keep | Bates all-five, Bates core-three | Strong learned fixed combinations. Bates all-five remains the declared primary comparator. Show the purged-fold version as a sensitivity check without replacing the historical primary. |
| Keep | Frozen HxV | Tests whether fixed differences across forecast steps and variables explain the result. |
| Keep as internal controls | Constant blend, static_hv blend, router_only | Directly challenge the need for the learned changing gate. These are CellShrink controls, not external published baselines. Existing gate results make their omission particularly hard to justify. |
| Next to complete | FFORMA-MAE adaptation | A named feature-conditioned combination method. Reuse the isolated conditional-comparison runner. State that the local loss and rolling-window interpretation differ from the original method. |
| Supporting controls | Conditional-error softmax, conditional MLP | Existing generic approaches with the same features and selection budget as FFORMA-MAE. Run the existing trio together when authorized; do not retain only whichever loses most. |
| Conditional priority | EMTSF-style gate | The closest existing cell-level learned-weight comparison. Retain it as an adaptation, subject to the training and normalization checks below. |
| Conditional priority | TimeRouter-style selection with fallback | A useful modern selective-routing challenger, but the current port remains withheld pending source-fidelity and protocol review. FAME is a second candidate if it adds a distinct question the final roster still lacks. |

CellShrink-core and CellShrink-soft remain method variants, never external baselines.

## Evidence that changes readiness

The [gate experiment](C:/Data/Documents/Code/BasicTS/project_memory/experiments/2026-10-18_gate_ablation.md) found no established across-dataset benefit for input-dependent gating at either horizon. Constant and static_hv controls therefore stay even if they outperform CellShrink.

The [EMTSF-style record](C:/Data/Documents/Code/BasicTS/project_memory/experiments/2026-10-27_emtsf_style_baseline.md) now includes a later diagnosis. Raw-scale normalization caused non-finite GPU gradients, and changing to standardized inputs affects every dataset. Amendment 2 failed its declared check. The record says the user decided on a new all-14-unit rerun under Amendment 3. Inspect its actual amendment, implementation, logs, and results before treating any replacement as complete. Preserve the original verdict and disclose its failed Electricity training; do not present it as a clean trained-baseline win.

The [October 25 audit](C:/Data/Documents/Code/BasicTS/project_memory/experiments/2026-10-25_baseline_port_health.md) withholds FAME, TimeRouter, OneNet OCP, and Super-Linear paper rows. Synthetic success does not establish reproduction of a paper. Renaming a port as an adaptation does not by itself override that decision.

The original [FFORMA](https://fpp.robjhyndman.com/publications/fforma/) learns combination weights from series features. [TimeRouter](https://arxiv.org/abs/2606.11625) combines discriminative routing, selective gating, and fallback. [FAME](https://arxiv.org/abs/2606.08896) uses forecastability fingerprints for sparse series-level expert routing. Those distinct roles justify investigating them; publication alone does not establish their local implementations.

## A reusable arena procedure

1. **Inventory.** Build a card per method with its scientific role, paper and code links, local implementation, expert pool, inputs, target use, selection rule, and known failures. Label it ready, adaptation awaiting audit, incompatible, or missing. Stop a method here if its source or training-data provenance cannot be established.

2. **Obtain three independent proposals.** Give each reviewer the same evidence and prompt below. Keep outputs separate. Each reviewer nominates a minimum useful roster, an expanded roster, and rejected alternatives. They must not see each other's drafts. Independence of sessions is not model-family diversity.

3. **Judge the proposals.** A separate reviewer checks scientific coverage, implementation evidence, fair inputs, cost, and simplicity. Require a concrete explanation for every retained row. Merge the most useful omissions into the best proposal. Reject any proposal that drops a valid strong comparator because CellShrink loses.

4. **Audit implementations before scheduling.** Check forecast-column identity, origins, masks, normalization, frozen checkpoints, and final primary wide_top3 alignment. Require finite inputs, loss, gradients, retained parameters, and predictions. Use planted-winner, shuffled-signal, and named-column permutation controls. Record collapsed routing and fallback frequency without treating averaging as a software failure. A failed check holds that method; it does not erase its historical run.

5. **Freeze a comparison plan.** Fix methods, configurations, budgets, windows, seeds, and outputs before further evaluation. Use the existing five-configuration, four-fold budget for the conditional trio. Declare other methods' budgets separately; do not claim equal budget against CellShrink's historical design search. This is a new development comparison, not retroactive confirmation. Stop before empirical execution until the separate experiment scope is established.

6. **Compare and retain all valid results.** For an authorized run, use identical frozen forecasts and 96-step histories, fully observed training labels, the 60–80% evaluation block, and all five final seeds. Average seed losses, not predictions. Report paired MAE and MSE, uncertainty, missing units, and training failures. Preserve the original readings. No test-block access, end-to-end retraining, or post-result roster changes belongs to this arena.

## Prompt to reuse

> Recommend the smallest credible baseline set for CellShrink on the same frozen forecast pool. Read the evidence cards independently. For each proposed row, identify the claim it tests, source evidence, required adaptation, local readiness, missing checks, and estimated work labeled as an estimate. Keep strong fixed combinations and the constant/static gate controls. Separate offline methods from methods receiving evaluation feedback. Do not invent results, rewrite historical conclusions, run training, or touch test data. Return a core roster, supporting roster, held methods, rejected alternatives, and exact next actions.

I rejected a larger model zoo, another fixed-weight formula, spatial routers on channels without spatial structure, and online OneNet in the offline ranking. These add redundancy or change the question. Cached forecasts also cannot establish inference speed savings.
