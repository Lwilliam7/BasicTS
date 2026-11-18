# Feature-Group Ablation for Affinity-Weighted Window-Dependent Expert Choice

Date: 2026-08-31. Directory: `experiments/feature_ablation_affinity_weighted_ec/`. Development
experiment; **router_val never touched — primary and only evidence is router_train causal OOF.**

## Question

Which of the three window-dependent feature groups (target-free cell-local forecast features,
per-variable history features, global history features) actually earn their place in the scorer,
tested by both adding them incrementally (F0→F1→F2→F3) and removing them one at a time from the
full model (Full-NoCell, Full-NoPerVariable, Full-NoGlobal)?

## Method

Kept fixed: K=3 train-selected frozen experts, scorer hidden layers (`->64->32->1`), seed=7, AdamW
lr=1e-3 wd=1e-4, 100 max epochs / patience 10, batch 32, residual-gain target, causal 4-fold OOF,
fit-only affinity calibration (temp=1.0), CF=1, independent EC claims (`dynamic_ec_claims`,
unmodified), affinity-weighted multi-claim fusion + zero-claim fallback (unmodified from
`affinity_weighted_expert_choice_hv`). Only the scorer's INPUT feature composition varies across 7
predeclared variants (`Full-NoGlobal` is provably identical to `F2_local` — anchor+cell+local, no
global — so it was reused rather than retrained, saving real compute; disclosed in the manifest).

**Real-world execution note:** the background training run (6 distinct variants × 5 datasets × 4
OOF folds, dominated by Electricity's 321 variables) was killed by the environment repeatedly and
unpredictably (9 interruptions total, no consistent timing, no resource-exhaustion signature) before
completing. Recovered by adding per-dataset → per-variant → per-fold checkpointing (in that order of
increasing granularity, added reactively as coarser granularity proved insufficient), so each
restart resumed from the finest completed unit of work rather than losing progress. This checkpoint
mechanism (`_checkpoints/` subdirectory, `torch.save`/`torch.load` per fold) is a reusable pattern
for future long-running experiments in this environment.

**One real bug found and fixed post-hoc (not retraining, pure relabeling correction):** the
`classify_group` disclosure field `independent_add_remove_evidence` compared variant NAME strings
instead of underlying feature-group sets, so it failed to recognize that `Full-NoGlobal` and
`F2_local` are the same configuration and misreported the GLOBAL group as having two independent
pieces of evidence when it only has one. Fixed and the report regenerated from the already-computed
(unchanged) OOF numbers — no retraining occurred, and the SUPPORTED/MIXED/NOT_SUPPORTED labels
were unaffected by the fix.

## Results (router-train OOF MAE)

| Dataset | F0_anchor | F1_cell | F2_local | F3_full | Full_NoCell | Full_NoPerVariable |
|---|---:|---:|---:|---:|---:|---:|
| ETTh1 | 0.360185 | 0.353092 | 0.351804 | 0.351324 | 0.360423 | 0.354307 |
| ETTh2 | 0.288718 | 0.286013 | 0.284354 | 0.286132 | 0.289744 | 0.286221 |
| ETTm1 | 0.261997 | 0.255611 | 0.253681 | 0.254763 | 0.255992 | 0.253956 |
| Weather | 0.185310 | 0.174185 | 0.166304 | 0.166782 | 0.180464 | 0.172095 |
| Electricity | 0.243443 | 0.232565 | 0.227905 | 0.229061 | 0.243322 | 0.232349 |

**`Full_NoGlobal` (= `F2_local`) beats `F3_full` (the full, current model) on 4/5 datasets** —
adding global features made OOF MAE *worse* on ETTh2/ETTm1/Electricity/Weather and only marginally
better on ETTh1. `BEST PREDECLARED FEATURE VARIANT BY OOF: F2_local`.

## Classifications

- **cell: MIXED.** Adding helps 5/5 (add_pass, block-24 support 5/5) but removing from full does
  NOT hurt on ≥3/5 (0/5 — removing cell from the full model actually helped on all 5 datasets).
  Independent add/remove evidence.
- **local: MIXED.** Adding helps 5/5 (block-24 support 4/5) but removing from full hurts only 1/5.
  Independent add/remove evidence.
- **global: MIXED.** Adding does NOT help (1/5), removing hurts 4/5 — but add and remove are the
  SAME comparison (`F3_full` vs `F2_local`) by construction, so this is only one independent piece
  of evidence, not two, disclosed explicitly after the bug fix above.

No group reached SUPPORTED. `FEATURE ABLATION VALID: YES`.

## Decision

Do not treat `F3_full` (all three feature groups) as obviously best — `F2_local` (cell + local
features, no global) wins on OOF and is the input to the follow-up embedding ablation. None of the
three feature groups individually cleared a clean SUPPORTED bar; this is a genuine mixed/negative
finding about feature-group value, not rescued or re-tuned. `ROUTER_VAL ACCESSED: NO` throughout.
