Final classification: OOF_GATE_FAIL_STOP

# Conflict-Resolved Window-Dependent Expert Choice

Development experiment (ETTh1/ETTh2/ETTm1/Weather/Electricity only). Reuses the frozen window_dependent_expert_choice_hv score/affinity tensors with NO retraining. The only new mechanism is conflict-resolved (deferred-acceptance-equivalent) assignment: every cell held by exactly one expert.

```text
TEST SET ACCESSED: NO
TEST CACHE LOADED: NO
TEST METRICS COMPUTED: NO
UNTOUCHED CONFIRMATION ACCESSED: NO
```

## Router-train OOF (primary evidence) -- Conflict-Resolved vs Affinity-Weighted: `0/5` wins, gate requires >=3/5

| Dataset | Dynamic EC (existing) | Affinity-Weighted EC | Conflict-Resolved EC | CR-Weighted delta | CR beats Weighted |
|---|---:|---:|---:|---:|---|
| ETTh1 | `0.351369` | `0.351321` | `0.353485` | `+0.002164` | False |
| ETTh2 | `0.286207` | `0.286132` | `0.288293` | `+0.002161` | False |
| ETTm1 | `0.254823` | `0.254765` | `0.256769` | `+0.002005` | False |
| Weather | `0.166868` | `0.166781` | `0.167967` | `+0.001186` | False |
| Electricity | `0.229083` | `0.229061` | `0.230268` | `+0.001207` | False |

`OOF GATE: FAIL` (0/5 >= 3/5 required).

## Deferred-acceptance verification (greedy vs literal round-based simulation)

| Dataset | Windows checked | Mismatches | All match |
|---|---:|---:|---|
| ETTh1 | 30 | 0 | True |
| ETTh2 | 30 | 0 | True |
| ETTm1 | 30 | 0 | True |
| Weather | 30 | 0 | True |
| Electricity | 30 | 0 | True |

## Conflict-Resolved EC claim rates (router_train OOF) -- expected zero_claim=0, multi_claim=0

| Dataset | 0-claim | 1-claim | 2-claim | 3-claim |
|---|---:|---:|---:|---:|
| ETTh1 | `0.0000` | `1.0000` | `0.0000` | `0.0000` |
| ETTh2 | `0.0000` | `1.0000` | `0.0000` | `0.0000` |
| ETTm1 | `0.0000` | `1.0000` | `0.0000` | `0.0000` |
| Weather | `0.0000` | `1.0000` | `0.0000` | `0.0000` |
| Electricity | `0.0000` | `1.0000` | `0.0000` | `0.0000` |

## STOP -- OOF gate failed

Conflict-Resolved EC beat Affinity-Weighted EC OOF MAE on only 0/5 datasets, below the predeclared minimum of 3/5. Per protocol, router_val was NEVER loaded for prediction/metric purposes in this run. This is reported as a valid negative result. Do not tune the assignment rule, capacity, or tie-break and rerun to try to pass the gate.

`ROUTER_VAL ACCESSED FOR THIS EXPERIMENT: NO`
`TEST ACCESSED: NO`
`UNTOUCHED CONFIRMATION ACCESSED: NO`
`VERDICT: STOP`
