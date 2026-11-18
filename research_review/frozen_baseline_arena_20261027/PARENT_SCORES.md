# Parent assessment

All three proposals were read in full after completion. Scores use RUBRIC.md. They assess the plans, not forecasting accuracy.

| Criterion | A | B | C |
|---|---:|---:|---:|
| Scientific relevance | 4 | 4 | 4 |
| Fidelity and evidence | 4 | 4 | 4 |
| Comparison fairness | 3 | 3 | 4 |
| Honest selection | 4 | 4 | 4 |
| Feasibility | 4 | 4 | 4 |
| Reusability and clarity | 3 | 4 | 4 |
| Total | 22 | 23 | 24 |

No candidate triggers a hard rejection. Candidate C is the provisional base because it explicitly separates each opponent's declared tuning budget from CellShrink's historical design search. Candidate A has no stand-alone reusable prompt. Candidate B has a good prompt and explicitly distinguishes legitimately uniform weights from failed training.

Useful additions to C:

- From A, preserve the complete roster and distinguish the best expert among all five from the existing core-only best expert. Add its role for router_only.
- From B, preserve the evidence that uniform weights can be a valid fitted result. Require training records before attributing uniform weights to a bug.

Corrections needed in any base:

- Make evidence eligibility specific to dataset, horizon, pool, and version. Existing gate controls cover 12 and 192, not an assumed complete 96-step grid.
- constant and static_hv fix the blend coefficient, but their routed forecast still depends on the scorer and current history. They are not fully static forecasts or seed-independent methods.
- State Amendment 3's exact rule. A best epoch at least 1 and finite stop loss through that epoch are required. Later nonfinite stop losses are allowed and counted by that amendment. Do not silently impose an every-epoch rule on the existing run.
- The local four-port exclusion remains in force. Calling a method an adaptation does not automatically clear its paper row.
- Set an explicit future roster and plan before any new comparison runs. Do not turn this planning exercise into a performance claim.
