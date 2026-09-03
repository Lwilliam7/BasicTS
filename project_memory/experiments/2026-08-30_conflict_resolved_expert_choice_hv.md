# Conflict-Resolved Window-Dependent Expert Choice

Date: 2026-08-30. Directory: `experiments/conflict_resolved_expert_choice_hv/`. Development
follow-up to `window_dependent_expert_choice_hv` (post-hoc; not untouched confirmation).

## Question

Keep experts selecting H×V cells, but resolve duplicate claims via expert-proposing deferred
acceptance (each expert ranks cells by its own affinity descending; proposes down its list until
holding capacity C; on conflict the cell keeps the higher-affinity claimant) instead of allowing
0/1/multi-claim cells. Same affinity tensor, same CF=1, no retraining (reused
`window_dependent_expert_choice_hv/tensors.pt` exactly as in the affinity-weighted experiment).
**Predeclared gate: if Conflict-Resolved does not beat Affinity-Weighted EC on OOF MAE on ≥3/5
datasets, STOP before touching router_val.**

## Implementation note (two real bugs found and fixed before any OOF number was trusted)

Implemented as a vectorized greedy pass, proven equivalent to the literal deferred-acceptance
procedure (the global-max affinity pair is always immediately/permanently accepted; induction
closes the argument) — but the equivalence was verified empirically against an actual literal
round-based simulation on sampled windows before being trusted at scale, and this caught two bugs:
1. **ETTm1**: exact fp16-precision affinity ties (the stored tensor is float16) need a deterministic
   tie-break for conflicts; the literal-DA reference initially lacked one.
2. **Electricity**: the greedy's composite sort key (`affinity*1e9 - cell_idx*1e3 - expert_idx`)
   numerically broke down for large M (3852 cells) combined with small affinity values — cell_idx
   could dominate affinity and corrupt priority order. Fixed with an exact, collision-free integer
   key built from the float16 bit pattern (IEEE754 guarantees bit-pattern order = float order for
   non-negative values).

After both fixes, all 5 datasets verify exactly (0 mismatches) against the literal simulation.

## Result

**OOF wins: 0/5.** Conflict-Resolved EC was worse than Affinity-Weighted EC on every single
dataset, uniformly, by 0.0012–0.0022 MAE:

| Dataset | Existing EC | Weighted EC | Conflict-Resolved EC | CR−Weighted |
|---|---:|---:|---:|---:|
| ETTh1 | 0.351369 | 0.351321 | 0.353485 | +0.002164 |
| ETTh2 | 0.286207 | 0.286132 | 0.288293 | +0.002161 |
| ETTm1 | 0.254823 | 0.254765 | 0.256769 | +0.002005 |
| Weather | 0.166868 | 0.166781 | 0.167967 | +0.001186 |
| Electricity | 0.229083 | 0.229061 | 0.230268 | +0.001207 |

`OOF GATE: FAIL` (0/5, required ≥3/5). Per protocol, **router_val was never loaded for any
prediction or metric.** Zero-claim=0, multi-claim=0, exact capacity per expert verified on every
dataset (integrity `all_pass=True`).

## Decision

**Negative result, not rescued.** Interpretation: the multi-claim cells the original design allows
(14–23% of cells, per the affinity-weighted experiment's diagnostics) provide real ensembling/
variance-reduction value; forcing strict one-expert-per-cell assignment removes that redundancy and
costs more than the "cleaner" assignment gains. Do not pursue conflict-resolved/stable-matching
assignment further for this method; the original independent-claims + equal-average combination
rule remains canonical.
