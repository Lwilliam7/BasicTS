# Expert-Choice Variant Sweep on Window-Dependent EC -- ETTh1 only

Date: 2026-09-03. Directory: `experiments/ec_variant_sweep_etth1/`. Development follow-up to
`window_dependent_expert_choice_hv` and `feature_ablation_affinity_weighted_ec`, restricted to
ETTh1 only per explicit user instruction (no ETTh2/ETTm1/Weather/Electricity, no router_val, no
test). Primary and only evidence is router_train strict chronological OOF (same 4-fold protocol,
warmup fraction, full-horizon-observability legality rule, seed=7, frozen K=3 core
`PatchTST+iTransformer+TimesNet`, checkpoints unchanged and hash-verified).

## Starting point

The F2_local scorer (cell-local forecast features + per-variable history features + H/V/expert
identity embeddings + static_gain scalar, **no** global-history features) that
`feature_ablation_affinity_weighted_ec` found beats the full-feature model on ETTh1 OOF MAE. Reused
verbatim (`feature_ablation_affinity_weighted_ec.FlexibleResidualScorer` /
`train_scorer_flexible` with `enabled_groups={"cell","local"}`). The scorer is trained exactly once
per OOF fold; every downstream configuration below reuses that same frozen raw-score tensor -- no
retraining anywhere in the grid.

## Grid tested (12 configurations = 3 CF x 2 assignment x 2 scoring)

1. **Capacity factor** CF in `{0.5, 1.0, 2.0}`, `C = max(1, round(CF*H*V/E))`.
2. **Assignment rule**: `unrestricted` (existing independent per-expert top-C selection, any number
   of experts 0..E may claim a cell) vs `max2` (deterministic global greedy assignment capping every
   cell at 2 claims; experts may end under nominal capacity if the constraint binds).
3. **Scoring normalization**: `existing` (one fit-only scalar mean/std over the whole raw-score
   tensor, then softmax over experts) vs `expert_relative` (fit-only PER-EXPERT mean/std, then
   softmax over experts).

Predeclared "current Expert Choice baseline" = `cf1.0_unrestricted_existing` (the F2_local analogue
of the existing method, held fixed as the reference row). Combination rule for claimed cells is the
unmodified equal-average-of-claiming-experts rule with equal-ensemble zero-claim fallback (no
affinity-weighted fusion, to isolate this sweep to capacity/assignment/scoring only). Compared
against matched Dynamic Token Choice (same scoring variant), the current EC baseline, and the
Frozen Dense Ensemble (equal average of the 3 experts, no routing).

## Results (router_train OOF, 4436 scored windows)

| Config | MAE | MSE | Fallback | vs Token | vs EC baseline | vs Frozen |
|---|---:|---:|---:|---:|---:|---:|
| `cf2.0_unrestricted_existing` (best) | `0.349150` | `0.269491` | `0.0000` | `-0.002341` | `-0.002774` | `+0.003559` |
| `cf0.5_unrestricted_existing` = `cf0.5_max2_existing` | `0.349727` | `0.270366` | `0.5152` | `-0.001765` | `-0.002198` | `+0.004136` |
| `cf2.0_max2_expert_relative` | `0.349862` | `0.267659` | `0.0000` | `-0.010677` | `-0.002062` | `+0.004271` |
| `cf1.0_unrestricted_existing` (baseline) | `0.351924` | `0.273279` | `0.1392` | `+0.000433` | `0.000000` | `+0.006333` |
| Frozen Dense Ensemble (no routing) | `0.345591` | `0.260821` | n/a | n/a | n/a | n/a |

Best config `cf2.0_unrestricted_existing` beats matched Token Choice and the current EC baseline
with block-24 CI entirely below zero (vs baseline: mean delta `-0.002774`, CI95
`[-0.003227, -0.002338]`), but is still significantly **worse** than the Frozen Dense Ensemble
(delta `+0.003559`, CI95 `[0.001341, 0.005697]`) -- consistent with every prior EC variant in this
line of work never beating Frozen HxV/dense-style ensembling.

At CF=0.5 and CF=1.0, `unrestricted` and `max2` produced numerically **identical** claim masks and
metrics on ETTh1: no cell ever received all 3 experts' claims at those capacities, so the >2-claim
constraint never bound. It only bites at CF=2.0 (where it binds and actually *hurts* under
`existing` scoring but *helps* under `expert_relative` scoring by a small amount).

## Answers to the four preregistered questions

1. **Does capacity factor improve OOF performance?** Yes, mildly and non-monotonically: CF=2.0
   (`0.349150`) < CF=0.5 (`0.349727`) < CF=1.0 (`0.351924`) holding assignment=unrestricted,
   scoring=existing. CF=1.0 is not the best CF.
2. **Does max-2-per-cell beat unrestricted or forcing one expert?** No: max2 beats unrestricted on
   only 1/6 matched (CF, scoring) pairs, and is identical to unrestricted on 4/6 pairs (constraint
   never binds at CF<=1.0). This is consistent with the earlier Conflict-Resolved EC finding that
   forcing exactly ONE expert per cell lost on 0/5 datasets (reused as context, not recomputed
   here) -- the pattern across both experiments is that restricting multi-claim overlap does not
   help this router family on ETTh1-style data.
3. **Does expert-relative softmax scoring help?** No: expert_relative beats existing on only 1/6
   matched (CF, assignment) pairs and is worse everywhere else, sometimes by a wide margin (up to
   `+0.0042` MAE at CF=1.0).
4. **Single best preregistered configuration to take forward:** `cf2.0_unrestricted_existing`
   (CF=2.0, unrestricted multi-claim, existing scalar-calibrated softmax scoring) -- OOF MAE
   `0.349150`, block-24-supported win over both matched Token Choice and the current EC baseline,
   but still meaningfully worse than the Frozen Dense Ensemble reference.

## Integrity

`INTEGRITY VALID: YES`. No test cache loaded, no router_val used for any prediction/metric/
selection, no other dataset touched, causal OOF holds on all 4 folds, frozen checkpoint hashes
unchanged before/after, source hash logged. One implementation bug was found and fixed before
trusting output (a report-generation loop unpacked ranked-config dicts as 2-tuples instead of
reading their `config` key) -- caught by a crash on the very first run, fixed, and the OOF numbers
themselves were unaffected (fold checkpoints made the rerun a pure re-evaluation of the same
cached raw-score tensors, not a retrain).

Evidence: `experiments/ec_variant_sweep_etth1/report.md`, `results.json`,
`config_grid_results.csv`, `dependence_tests.csv`, `method_manifest.json`,
`integrity_checks.json`.
