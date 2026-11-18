# Frozen Method: LearnedProbe-Rank

**The LearnedProbe-Rank architecture and hyperparameters were frozen before evaluating the new generalization datasets. No changes are permitted based on new-dataset validation results.**

Full machine-readable details: [`frozen_method_manifest.json`](frozen_method_manifest.json).

## What is frozen

- **ETTh1, ETTh2, ETTm1, Weather, Electricity** are now development datasets. They are not used for any further tuning.
- The standing method is the **ORIGINAL LearnedProbe-Rank** (plain pairwise ranking loss), not GapRank. GapRank was tested as a candidate improvement and rejected: it improved ETTm1 but produced statistically significant regressions on ETTh2, Weather, and Electricity (see [`reports/learned_probe_gaprank_report.md`](reports/learned_probe_gaprank_report.md), verdict `NO IMPROVEMENT — KEEP ORIGINAL`).
- Frozen pipeline:

  ```
  current window + original expert forecasts + expert disagreement
    -> learned diagnostic probe (ProbeGenerator, eps=0.05)
    -> probe each frozen expert (differentiable forward pass; expert weights never updated)
    -> probe-response features (6 stats)
    -> competence scorer (CompetenceScorer, 21-dim input)
    -> rank experts by predicted competence
    -> fixed rank weights: raw=[K,K-1,...,1]/sum -> 0.5000 / 0.3333 / 0.1667 for K=3
    -> combine ORIGINAL expert forecasts
  ```

- Frozen exactly: ProbeGenerator architecture, epsilon (0.05), perturbation constraints (magnitude/mean-shift/smoothness/window-only), competence features (groups A/B/C/D, 21-dim), competence scorer architecture, feature normalization procedure, training objective (`Huber + 0.25*pairwise_ranking_loss + 0.01*perturbation_penalties + 0.01*smoothness`), optimizer (AdamW, lr=1e-3, weight_decay=1e-4), training epochs/early-stopping protocol (max 8, patience 3, seed 7), rank weights ([0.5, 0.333, 0.167] for K=3, see correction note below), expert-selection protocol (pooled router_train OOF MAE over C(5,3) subsets), input length (96) and horizon (12).

> **Correction on rank weights**: every prior report in this experiment family (including this file's first draft) described the rank weights as "0.60/0.30/0.10." That number was never actually computed by any code path — it was an incorrect annotation written next to the correct formula in `run_learned_probe_mechanism_ablation.py`'s never-triggered `write_frozen_manifest()`, then echoed forward through every subsequent report. The function actually used everywhere, `rule_fixed_rank`, computes `[K,K-1,...,1]/sum`, i.e. **[0.5000, 0.3333, 0.1667]** for 3 experts. Every previously reported "Rank" MAE/MSE number (development datasets and GapRank alike) used this correct rule consistently — only the prose label was wrong. Confirmed with the user: this is the rule being frozen, to preserve exact numerical continuity with everything already reported.

## Generalization extension

New datasets reuse this exact protocol unmodified. The only additions are registry/plumbing entries — a new dataset name mapped to its own checkpoint root and data-loading bundle — so the existing, unmodified functions (`train_probe_and_scorer`, `evaluate_on_val`, `select_core_on_router_train`, `run_dataset` for C/Fixed-D) can run on new data. No architectural, hyperparameter, loss, or decision-rule change was made for any new dataset. Any unavoidable shape-only adaptation (e.g. a dataset having a different number of variables) is documented explicitly wherever it occurs.

## Process discipline

- New datasets are evaluated on **router_val only** in this first pass. The final test split for every new dataset stays locked and is not built or accessed.
- Expert-core selection for new datasets uses **router_train only** (pooled out-of-fold MAE); router_val is never used to select experts.
- Dataset selection (`generalization/dataset_selection.json`) was finalized before any LearnedProbe-Rank performance was inspected on any new dataset.
- No new tuning: epsilon, rank weights, ProbeGenerator size, scorer architecture, loss coefficients, dataset-specific features, confidence thresholds, or expert-specific corrections are not touched, regardless of what the new-dataset results show.
