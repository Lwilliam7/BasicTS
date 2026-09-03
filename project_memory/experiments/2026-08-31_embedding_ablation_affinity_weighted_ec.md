# Embedding Ablation for Affinity-Weighted Window-Dependent Expert Choice

Date: 2026-08-31. Directory: `experiments/embedding_ablation_affinity_weighted_ec/`. Development
experiment; **router_val never touched — primary and only evidence is router_train causal OOF.**
Uses the feature variant selected by `feature_ablation_affinity_weighted_ec` on OOF alone
(`F2_local` = cell-local + per-variable features, no global features) — not reconsidered here.

## Question

Do the H (dim 4), V (dim 8), and Expert (dim 4) identity embeddings actually help the shared
scorer? Specifically: does V add information beyond the `static_gain[h,v,e]` scalar already given
as input, and does the Expert embedding matter for a scorer SHARED across all K=3 heterogeneous
experts (one set of weights, not one network per expert)?

## Method

Kept fixed: K=3 frozen experts, scorer hidden layers, residual-gain target, seed/optimizer/lr,
causal 4-fold OOF, calibration, CF=1, independent EC claims, affinity-weighted fusion. For fairness,
a "removed" embedding is replaced by a fixed ZERO vector of the same dimension (not deleted), so
input dimension and MLP capacity are identical across all 5 variants (`E_full`, `E_noH`, `E_noV`,
`E_noExpert`, `E_none`) — only embedding content is ablated. `E_full`'s OOF MAE reproduces
`F2_local`'s exactly (0.227905 on Electricity, etc.), a useful cross-check that the two experiments'
pipelines are consistent.

## Results (router-train OOF MAE)

| Dataset | E_full | E_noH | E_noV | E_noExpert | E_none |
|---|---:|---:|---:|---:|---:|
| ETTh1 | 0.351804 | 0.350493 | 0.352906 | 0.352012 | 0.353683 |
| ETTh2 | 0.284354 | 0.284739 | 0.284941 | 0.286267 | 0.285672 |
| ETTm1 | 0.253681 | 0.255513 | 0.255265 | 0.254532 | 0.253749 |
| Weather | 0.166304 | 0.167757 | 0.167497 | 0.166520 | 0.168787 |
| Electricity | 0.227905 | 0.228791 | 0.229250 | 0.228480 | 0.229829 |

## Classifications

- **H embedding: SUPPORTED.** Removing hurts OOF on 4/5 datasets (ETTh1 is the sole exception,
  where removing H actually helps slightly). Block-24 support on 3/5. Aggregate delta net-positive.
- **V embedding: SUPPORTED.** Removing hurts OOF on 5/5 datasets (unanimous). Block-24 support 3/5.
  Directly answers the "does V add beyond static_gain?" diagnostic: yes — `static_gain[h,v,e]` is
  present as an explicit input in every variant here, so this isolates the learned V-identity
  vector's marginal value beyond that scalar.
- **Expert embedding: SUPPORTED.** Removing hurts OOF on 5/5 datasets (unanimous). Block-24 support
  3/5. Directly answers the "does the shared scorer need expert identity?" diagnostic: yes — a
  single shared network benefits from a learned per-expert identity vector to distinguish
  heterogeneous experts' residual competence, beyond what `static_gain` alone provides.

`EMBEDDING ABLATION VALID: YES`.

## Decision

All three embeddings are genuinely useful to the shared scorer; none should be removed. V and
Expert have the strongest (unanimous 5/5) evidence. `ROUTER_VAL ACCESSED: NO` throughout.
