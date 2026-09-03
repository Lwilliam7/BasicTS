# Affinity-Weighted Expert Choice H×V

Date: 2026-08-30. Directory: `experiments/affinity_weighted_expert_choice_hv/`. Development
follow-up to `window_dependent_expert_choice_hv` (post-hoc; not untouched confirmation).

## Question

The original Expert-Choice mechanism (Zhou et al., NeurIPS 2022) retains gating weights for
selected outputs; the local Dynamic EC implementation instead equal-averaged claiming experts on
multi-claim cells (verified directly from source:
`CURRENT_MULTI_CLAIM_RULE = simple unweighted average of claiming experts' forecasts, fallback to
equal ensemble on zero claims; affinity previously used only for top-C selection`). Does restoring
affinity-weighted combination (`w_e = a_e / sum_{claimants} a_e`) on multi-claim cells improve OOF
MAE over the existing equal-average rule?

## Method

**No retraining.** `window_dependent_expert_choice_hv` never persisted scorer checkpoints, only its
final score/affinity/claim tensors (`tensors.pt`); those are loaded directly (float16 on disk,
upcast to float32) and reused as-is. Single-claim cells reduce to the existing rule exactly
(`x/x==1.0` in IEEE754); zero-claim cells use the identical equal-ensemble fallback — both verified
bit-identical, not assumed.

## Results (router-val MAE)

| Dataset | Existing Dynamic EC | Weighted EC | Delta | OOF Delta |
|---|---:|---:|---:|---:|
| ETTh1 | 0.375640 | 0.375671 | +0.000031 | −0.000047 |
| ETTh2 | 0.280951 | 0.280903 | −0.000049 | −0.000075 |
| ETTm1 | 0.253556 | 0.253540 | −0.000016 | −0.000059 |
| Weather | 0.155621 | 0.155585 | −0.000037 | −0.000087 |
| Electricity | 0.206356 | 0.206322 | −0.000034 | −0.000022 |

Classification: **`AFFINITY_WEIGHTED_EC_SUPPORTED`** by the letter of the predeclared rule (OOF
wins 5/5, val wins 4/5, block-24 support 2/5 exact minimum, still beats Dynamic Token 4/5) — but
**effect size is ~2 orders of magnitude smaller** than the window-dependent effect itself (1e-5 vs
1e-3 MAE), and Weighted EC still loses to Frozen HxV on 3/5 datasets. Oracle diagnostic (best
achievable convex combination given true OOF targets, analysis-only, never used to fit anything):
real headroom exists in how overlapping claims should be combined (0.03–0.06 MAE gap vs equal-
average on multi-claim cells), but affinity-renormalization captures only ~0.5–1% of that gap.

## Decision

**Practically negligible.** Do not promote as the canonical multi-claim rule; the equal-average
rule remains canonical. The oracle-headroom finding is the useful takeaway: a smarter (not yet
built) combination rule might matter, simple affinity renormalization does not. Directly motivated
`conflict_resolved_expert_choice_hv`.
