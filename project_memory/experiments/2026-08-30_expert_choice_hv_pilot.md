# Expert-Choice HxV Pilot

Date: 2026-08-30

Experiment directory: `experiments/behavioral_competence/expert_choice_hv_pilot/`

## Question

Does reversing HxV routing from "each horizon-variable cell chooses an expert" to "each expert claims the HxV cells where it expects to perform best" improve forecasting?

The experiment isolates the allocation mechanism: Hard Normal HxV and Expert Choice use the exact same train-derived competence score tensor.

## Protocol

- Dataset: `Electricity` only.
- Splits: `router_train_20_60` for score construction; `router_val_60_80` for evaluation.
- Test access: none.
- Experts: `PatchTST`, `iTransformer`, `TimesNet`.
- Cache expert order verified: `DLinear`, `PatchTST`, `iTransformer`, `TimesNet`, `ModernTCN`.
- Score: `score[k,h,v] = -mean_router_train_normalized_abs_error[k,h,v]`.
- Normalization: existing per-location error utility, using the train checkpoint scaler std.
- Cells: `H=12`, `V=321`, total `3852`.
- Hard Normal HxV: each cell assigned to argmax expert by score.
- Expert Choice: global constrained assignment via `scipy.optimize.linear_sum_assignment` with cloned expert-capacity slots.
- Capacity settings: equal strict `1.00`, capacity factor `1.25`, capacity factor `1.50`.
- Controls: random expert scores, permuted HxV score locations, no-capacity Expert Choice.
- Oracle diagnostics: per-window dynamic oracle and static router_val-average HxV oracle, explicitly non-deployable.

## Results

Verdict by requested rule: `STRONG GO` for `Expert Choice cap 1.25` versus static `Hard Normal HxV`.

| Method | MAE | Delta vs Equal | Delta vs Hard HxV |
|---|---:|---:|---:|
| Equal ensemble | `0.214457` | `+0.000000` | `-0.008303` |
| Existing soft HxV | `0.211775` | `-0.002682` | `-0.010985` |
| Hard Normal HxV | `0.222761` | `+0.008303` | `+0.000000` |
| Expert Choice cap 1.00 | `0.222948` | `+0.008491` | `+0.000187` |
| Expert Choice cap 1.25 | `0.220627` | `+0.006170` | `-0.002134` |
| Expert Choice cap 1.50 | `0.222734` | `+0.008277` | `-0.000027` |

Main comparison stats:

| Comparison | Mean delta | 95% CI | P(delta < 0) | Phase agreement |
|---|---:|---:|---:|---:|
| EC cap 1.00 vs Hard | `+0.000187` | `[-0.001234, +0.001591]` | `0.394` | `6/12` negative |
| EC cap 1.25 vs Hard | `-0.002134` | `[-0.003012, -0.001256]` | `1.000` | `12/12` negative |
| EC cap 1.50 vs Hard | `-0.000026` | `[-0.000036, -0.000017]` | `1.000` | `12/12` negative |

Allocation counts:

| Method | PatchTST | iTransformer | TimesNet |
|---|---:|---:|---:|
| Hard Normal HxV | `1945` | `1768` | `139` |
| EC cap 1.00 | `1284` | `1284` | `1284` |
| EC cap 1.25 | `1605` | `1605` | `642` |
| EC cap 1.50 | `1926` | `1787` | `139` |

Oracle diagnostics:

- Dynamic per-window HxV oracle MAE/MSE: `0.138875` / `0.064471`.
- Static router_val-average HxV oracle MAE/MSE: `0.215789` / `0.118859`.

## Controls

- No-capacity Expert Choice was exactly identical to Hard Normal HxV, confirming that the no-capacity formulation collapses to normal cell-to-expert argmax.
- Random-score controls were much worse than Hard Normal HxV.
- Permuted HxV score-location controls were much worse than Hard Normal HxV, supporting that the real HxV score locations matter.

## Integrity

All integrity gates passed:

- no test cache/file loaded;
- cache roles verified as `router_train_20_60` and `router_val_60_80`;
- expert ordering verified;
- assignments derived from router_train scores only;
- same score tensor used for Hard Normal HxV and Expert Choice;
- checkpoint hashes unchanged.

## Decision

Capacity-constrained expert-choice allocation is worth follow-up as a mechanism for repairing hard HxV over-allocation. Do not replace the existing Electricity soft/causal HxV router: the best Expert Choice static hard variant (`0.220627`) is still worse than Equal (`0.214457`) and existing soft HxV (`0.211775`).

## Artifacts

- `experiments/behavioral_competence/expert_choice_hv_pilot/run_expert_choice_hv_pilot.py`
- `experiments/behavioral_competence/expert_choice_hv_pilot/results.json`
- `experiments/behavioral_competence/expert_choice_hv_pilot/RESULTS.md`
- `experiments/behavioral_competence/expert_choice_hv_pilot/integrity_report.json`
- `experiments/behavioral_competence/expert_choice_hv_pilot/method_metrics.csv`
