# CellCatch: preliminary research review

This introduction-only assessment is superseded by [the actual-repository assessment](CellCatch_repository_assessment_2026-10-16.md), which reviews the results in C:/Data/Documents/Code/BasicTS, including the returned long-horizon bundle. Several experiments described below as needed were already completed in that repository. Keep this file as the record of what was known before the correct repository was located.

Date: October 16, 2026

## Assessment and evidence boundary

There is a coherent paper hypothesis here: input-dependent selection at each forecast horizon-variable cell may exploit complementary frozen forecasters, while interpolation toward a fixed equal-weight ensemble may limit the damage from unreliable selections. Start writing the precise method and assembling the evidence table now. The introduction alone does not establish that the method works or that it is novel.

Reviewed material: the introduction pasted into the chat, a research review agent's independent critique, the single notebook currently in BasicTS, and the primary literature linked below. The CellCatch experiment files, code, paper bibliography, and Markdown research notes have not been located. The introduction's 2.2%, 3.6%, tie, and 0.8% results are author-reported and remain unverified. The DLinear notebook does not validate those CellCatch results.

## What to do next, in order

1. Locate the CellCatch results and notes; map every numerical claim to an exact output file and configuration. Recover saved expert predictions if available so initial analyses need not retrain experts.
2. Document the chronological evaluation pipeline and write the actual method. Identify what each selection or gate sees during training and at forecast time.
3. Assemble a compact ablation table that separates the benefit of expert subset selection, routing granularity, and gating. Use development data for design choices and preserve an untouched evaluation where available.
4. Revise the introduction around what that evidence establishes. Expand benchmarks when needed to test a specific claim, rather than adding experiments without a decision they will resolve.

## Introduction: substantive corrections

### Winner frequency is not evidence of predictable advantage

The 36% and 33% figures describe the distribution of winning experts. They do not show that winners can be predicted, or that choosing them reduces error enough to matter. Errors could differ by negligible amounts, and winner identities could be unpredictable noise. Define the counting unit: individual forecast origin/horizon/variable, or horizon-variable positions after averaging over origins. State tie handling and rounding.

Remove the inference from being near one third by chance. A uniform random selector has expected agreement of one third with a unique winner among three experts; a measured winner share is a different statistic. Neither statistic measures the error saved by a deployable combiner.

Instead quantify the error gap between the fixed ensemble and an oracle using the same allowable subset family, then measure how much of that gap a learned selector captures on unseen periods. Oracle analyses use future outcomes only as diagnostics or training supervision; they cannot be deployed.

### State the selection rule precisely

The fixed core has three experts, and the cell branch also selects three. If the latter selects only from the core, the two means are identical. If it selects from all five experts, say so explicitly.

A possible description, to check against the implementation, is:

\[
\hat y_{h,v}(x)=(1-g_{h,v}(x))\frac{1}{3}\sum_{e\in C}\hat y^{(e)}_{h,v}(x)
+g_{h,v}(x)\frac{1}{3}\sum_{e\in S_{h,v}(x)}\hat y^{(e)}_{h,v}(x),
\qquad g_{h,v}(x)\in[0,1].
\]

Here C is a fixed three-expert core, and S is a predicted three-expert subset of the full five-expert pool. This equation is an interpretation of the prose, not a verified description of the code.

Specify core selection data, ranking targets and loss, selector inputs, gate inputs and loss, whether routing actually changes with each input x, and whether gates share parameters across cells. If the cell subsets are fixed after training, describe them as static cell-specific combinations rather than input-dependent routing. Replacing a frozen expert may still require retraining the router.

The three best individual experts need not produce the best averaged forecast: signed errors can cancel. There are only ten three-expert subsets of five. On development data, compare individual-error ranking with selection based on the averaged subset's error. A per-cell oracle over those ten subset averages is a best-case diagnostic for this restricted branch family; a single-expert oracle is not a general bound for forecast combinations.

### Narrow and verify the novelty claim

- [TimeRouter](https://arxiv.org/html/2606.11625v1) already combines frozen forecasters, selective gating, and an ensemble fallback. Its stated policy chooses an expert or ensemble for a univariate forecast window. Gating toward an ensemble alone is therefore insufficient to distinguish CellCatch.
- [Synapse](https://arxiv.org/html/2511.05460) already adapts forecast-model weights at each forecast timestamp. It combines predictive distributions through sampling and updates weights through forward simulation. A claim that prior frozen ensembles only use window-level weights would overlook this work.
- [FAME](https://arxiv.org/html/2606.08896v1) trains routing over frozen pretrained experts using series-level suitability information. It is relevant to distinguishing selection granularity and supervision.
- [AFG-EL](https://www.mdpi.com/2227-7390/14/10/1686) describes a heterogeneous frozen forecasting pool, sample-level routing, top-k selection, and an uncertainty fallback. Inspect its method and compare its setting with CellCatch before finalizing related work.

These are targeted overlap checks, not an exhaustive novelty review. A potentially useful distinction is the exact combination of horizon-variable subset prediction and learned interpolation toward a fixed core. Establish its difference from prior methods and its empirical value; do not claim it is first without a fuller search.

### Qualify the empirical claim

Two improvements, one tie, and one deterioration can support a useful research paper if the comparison is sound and explains when the mechanism helps. They do not establish consistent superiority. Report the dataset names, absolute MAE values, comparator names, forecast horizons, and aggregation rule. Define whether a tie means equal rounded scores, a predeclared practical tolerance, or an uncertainty-based conclusion; failure to find significance does not establish equivalence.

Use relative improvement = 100 * (comparator MAE - CellCatch MAE) / comparator MAE. A best comparator identified retrospectively on each test dataset is a descriptive comparison; it is not a deployable selection policy. Label it accordingly.

Distinguish held-out test periods from entire datasets excluded during router training and design. The phrase "four held-out datasets" should match the actual protocol. If all expert forecasts must be computed to select or combine cells, selecting three does not by itself establish inference savings.

### Improve the argument and wording

Compress the generic application opening and introduce the frozen-pool problem earlier. Explain the mechanism and the evidence gap before listing contributions. Replace the awkward research question with: "Can input-dependent horizon-variable routing improve on strong forecast combinations, and does interpolation toward a fixed average make those decisions more reliable?"

Do not say that finer cells necessarily contain fewer training examples. They may instead share less information across horizons and variables, while overlapping windows and correlated targets reduce effective independent evidence. Tie the estimation-noise explanation to the actual training procedure.

The equal-average motivation is reasonable. [Smith and Wallis (2009)](https://onlinelibrary.wiley.com/doi/10.1111/j.1468-0084.2008.00541.x) and [Claeskens et al. (2016)](https://www.sciencedirect.com/science/article/pii/S0169207016000327) analyze how estimating combination weights can harm performance. These results motivate a fallback; they do not guarantee the proposed gate improves MAE.

Fix "convocational" to "convolutional," "change" to "chance" if retained, agreement/plural errors around "routers," repeated phrases, the corrupted MAE sentence, and the duplicated contribution. Method, abstract, experiments, and conclusion are placeholders, so this is an early paper skeleton rather than a complete manuscript.

## Minimum useful evidence table

Use the same frozen predictions, preprocessing, splits, and validation selection rules wherever the comparison permits. Give learned baselines a reasonable tuning budget.

| Comparison | Question it answers |
| --- | --- |
| Best single expert selected on validation; mean of all five; fixed-core mean | Does combining and core selection help? |
| Validation-selected best fixed three-expert subset; regularized linear/convex combination | Is improvement explained by simple subset choice or learned static weights? |
| Window weights, variable-specific weights, horizon-specific weights, full cell routing | Does joint horizon-variable granularity add value? |
| Predicted cell subset without interpolation; full method | Does the gate add value? |
| Constant interpolation strength selected on validation; learned gate | Is input-dependent gating necessary? |
| Development-only oracle over the ten subset averages | Is useful subset-selection headroom present? |

Match architecture and capacity where feasible, and report differences when exact matching is impossible. Compare a relevant coarse selective-gating baseline and an appropriate fine-grained adaptive combination; use a faithful implementation or label an adaptation clearly.

For the gate, inspect realized error change relative to the core by predicted gate strength. This tests whether larger moves are associated with better selections. Report this as empirical behavior rather than a safety guarantee.

## Evaluation details needed before a readiness verdict

- Expert-training targets, router-fitting targets, calibration/model-selection targets, and final evaluation targets, in chronological order. Router fitting should use expert forecasts produced without training those experts on the same target outcomes, through later periods or suitable temporal out-of-fold construction.
- Preprocessing fitted only on the permitted training portion, exact target channels and scale, loss definitions, forecast-origin alignment, and handling of overlapping windows. Historical input contexts can legitimately overlap; ensure evaluation outcomes never inform training or selection.
- Core selection, hyperparameters, gate calibration, and checkpoint selection without final-test feedback. If test results already guided research choices, document that and plan a fresh temporal evaluation or other untouched test where feasible.
- Per-dataset/horizon scores, learned-model seed variability where relevant, and paired uncertainty estimates that respect serial dependence and overlapping forecast targets. Horizon-variable cells are not independent experimental replicates. More seeds alone do not measure time-series sampling uncertainty.
- Saved predictions/configurations, split definitions, software versions, seeds, and a reproducible route from artifacts to every reported number.

## Writing direction

Suggested framing to adapt after checking the implementation:

> Combining frozen forecasters raises a tradeoff between flexibility and estimation reliability. A single set of weights can overlook differences across forecast horizons and variables, while fine-grained decisions may be unstable when expert errors are similar. We study whether input-dependent subset selection at each horizon-variable cell improves on strong combinations, and whether interpolation toward a fixed equal-weight core controls the resulting instability.

Possible contributions, conditional on results:

1. A precise combiner that interpolates between a fixed equal-weight core and a predicted horizon-variable subset of frozen forecasters.
2. A controlled study of routing granularity and interpolation under a common expert pool.
3. An empirical account of when cell routing helps or fails, supported by headroom, ablation, and uncertainty analyses.

Write the method and results before polishing the opening again. If cell routing and the learned gate each provide reproducible gains, develop the method paper. If static combinations explain the gains, simplify the method or frame the contribution as an empirical study. If headroom exists but learned routing cannot capture it, investigate the supervision and estimation problem. These are decisions for the missing results to resolve, not conclusions already established.
