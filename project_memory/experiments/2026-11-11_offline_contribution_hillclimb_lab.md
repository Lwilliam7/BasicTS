# Offline replacement for the learned gate: lab hillclimb (2026-11-11)

Status: LAB ONLY. Public data, re-implemented experts and a re-implemented gate, not the user's pools. No test split
was touched. The candidate it produced failed all three pre-committed confirmations. It enters the server run only as
exploratory arms (`experiments/offline_bates_router/PREDECLARATION.md`, amendment 2).

## Question

Is there an offline contribution, other than the learned gate (LambdaHead), that lowers MAE? Offline means no value
realised after the forecast origin is used at forecast time, so recent expert errors are out. The method must stay
offline and keep the frozen experts and the scorer.

## Setup

- The lab mirrors CellCatch wide_top3 on the first 14,400 rows, with blocks at 20/40/60/80%.
  - Five walk-forward experts: DLinear, PatchTST, iTransformer, TimesNet, ModernTCN.
  - Router_train is 20-60%: block B is forecast by experts trained on 0-20%, block C by experts trained on 0-40%.
    Router_val is 60-80%.
  - The scorer uses 4 causal OOF folds. The lab learned gate is the reference.
- Metric: router_val MAE on z-scored data. Per-window losses are averaged over scorer seeds, and each arm is
  reported as a % change against the lab learned gate, with a 95% moving-block bootstrap (block = max(24, H)).
- Protocol:
  - Development slices: ETTh1 H12 and ETTm1 H12 (seeds 7/11/13), ETTh1 H96 and ETTm1 H96 (seed 7).
  - Every arm fits only on router_train. The harness gives arms no router_val target.
  - Confirmation used slices that had not been scored before, with the candidate's code hashed and the pass rule
    written down first. Pass rule: mean change < 0 and no slice significantly worse than the gate.

## Headroom (in-sample oracle on router_val, so not reachable)

| Oracle | ETTh1 H12 | ETTm1 H12 | ETTh1 H96 | ETTm1 H96 |
| --- | ---: | ---: | ---: | ---: |
| Per-cell best equal-weight subset of the 5 experts | −1.86% | −1.81% | −2.02% | −1.19% |
| Per window and cell, best of core-3 or routed | −9.82% | −12.40% | −5.26% | −11.06% |
| Per window and cell, best single expert | −37.2% | −45.0% | −25.9% | −42.3% |

Static re-weighting can gain at most about 2%, even with hindsight. Window-level choices have large headroom, but the
earlier gate diagnosis found no offline input that predicts them.

## Hypotheses (mean change vs the learned gate over the 4 development slices unless noted)

| # | Idea | Result | Verdict |
| --- | --- | --- | --- |
| R0 | Gate removed: one closed-form constant λ blend of core-3 and routed (`constant`) | −0.08% (2 better, 0 worse) | reference |
| 1 | Median instead of mean (all 5, or the routed 3) | +0.86%, +0.54% | worse |
| 2 | Static per-cell best subset of 31 equal averages, chosen on router_train | +0.24% to +0.38% | worse |
| 3 | Static per-cell top-3 (static twin of the router) | +0.47% | worse |
| 4 | Router bagged over seeds | +0.39% (routed alone +0.43%) | no gain |
| 5 | Averaging forecasts from earlier origins for the same target | +0.66% (1 origin) to +8.9% (all) | worse |
| 6 | Static fits on block C only (closest expert vintage) | −0.11% best | no gain |
| 7 | One blend weight per horizon | −0.09% | no gain |
| 8 | Softmax routing with a temperature chosen on OOF | +0.82% alone, +0.26% blended | worse |
| 9 | Scores smoothed along the horizon before top-3 | −0.08% | no gain |
| 10 | Simplex stack of core-3, routed, Bates, equal-5 | −0.09% | no gain |
| 11 | Router trained end to end on the MAE of its softmax forecast | dynamic +3.10% (H12); static −0.25% (H12), −0.51% (ETTh1 H96, interval crosses 0) | worse / no gain |
| 12 | Blend plus a per-horizon step toward the last observed value (pooled fit) | −0.20% (−0.12% vs the blend) | small gain |
| 13 | Ridge on the last k values instead of one | no gain over 12 | reverted |
| 14 | 12 plus the value one day before each target | −0.69% | failed round 1 |
| 15 | 14 with anchor weights fitted per expert vintage, smaller kept | −0.63% | failed round 2 |
| 16 | The same vintage rule for λ | +0.07% vs the blend | worse |
| 17 | Last value only, vintage rule | post hoc only | failed round 3 |

## Confirmations

| Round | Candidate | Slices and result vs gate | Verdict |
| --- | --- | --- | --- |
| 1 | 14 | ETTh2 H12 +5.15% w, ETTm2 H12 −0.23% b, ETTh2 H96 +1.57% w, ETTh1 H192 −0.73% b | failed |
| 2 | 15 | ETTm1 rows 28800-43200 +2.84% w, ETTm2 same rows −0.20% b, exchange rate −3.11% b | failed |
| 3 | 17 | ETTm1 rows 43200-57600 +0.08%, electricity (7 clients) +0.24% w, traffic (7 sensors) −0.47% b | failed |

Root causes, from diagnosis rather than tuning:

- Round 1. ETTh2 block C is a shifted regime: target sd is 1.36 against 0.74 in block B. There the experts fail
  (MAE 0.30-0.38, against 0.32 for the last value), and the pooled least-squares fit, dominated by block C, puts
  weight 0.25-0.39 on the last value at every step. In router_val the experts are good again (0.22 against 0.30 for
  the last value). Fitting on block B alone gives 0.04-0.10.
- Round 2. The day-before anchor lost 3.2% against the blend on the new ETTm1 period. The last value alone gained
  on all three slices.
- Round 3. The electricity loss is the constant blend's, not the anchor's (the anchor weight is zero there).

## Conclusions

1. Offline re-weighting of the five frozen experts cannot replace the gate with a real MAE gain. The oracle bound is
   about 2%, and 12 re-weighting ideas landed within noise or worse.
2. Giving a router freedom to use window features overfits (+3.1%). This is consistent with the gate diagnosis: no
   offline window-level signal.
3. Anchoring to the last observed value is real but small and dataset-dependent. The gain is about −3% on exchange
   rates, −0.05% to −0.2% on most ETT slices and zero on electricity and traffic. It sits entirely in steps 1-3, so
   none is expected at 192 steps.
4. Lesson, post hoc: router_train mixes expert vintages, and a fit pooled over them can be dominated by a block where
   the experts fail. Fitting per vintage and keeping the smaller weight cut the ETTh2 loss from +5.5% to +0.3%.
5. Removing the gate (the closed-form constant blend) is neutral on most of the 14 lab slices: 3 significantly
   better, 2 significantly worse (ETTh2 H12 +0.21%, electricity +0.24%), 9 not significant.

## Limits

The lab experts are weaker than the real pools (at ETTh1 H192 DLinear alone beats every combination). The lab uses
7 variables and mostly H12. The gate is a re-implementation from bytecode. None of these numbers transfers to the
real pools without the server run.

## Files

- Server arms: `experiments/offline_bates_router/run_offline_bates_router.py` (amendment 2 arms; `--selftest` covers
  them). Before commit, a run through stand-in modules on the lab data reproduced the lab arms' MAE exactly
  (0.36422 and 0.35956 at ETTh1 H12, seed 7).
- The lab harness, the arms, the decision log (`decision.tsv`) and the slice caches stayed in the session
  scratchpad, not in the repo.
