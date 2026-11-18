# Signed Pair Residual Fault Gate

Final classification: `WEAK_OR_INCONSISTENT_PARITY_FAULT_SIGNAL`

Strict validation-only. No test cache, target, or metric was loaded.

## Routing MAE

| Dataset | Baseline | Passive | Parity | Passive+Parity | Shuffled | Raw Control |
|---|---:|---:|---:|---:|---:|---:|
| ETTh1 | `0.366022` | `0.366662` | `0.366943` | `0.367019` | `0.366670` | `0.366483` |
| ETTh2 | `0.276898` | `0.276275` | `0.276465` | `0.275640` | `0.275973` | `0.276318` |
| ETTm1 | `0.250690` | `0.251171` | `0.251457` | `0.251200` | `0.251191` | `0.251296` |
| Weather | `0.159818` | `0.160169` | `0.160172` | `0.160052` | `0.160127` | `0.159600` |
| Electricity | `0.215355` | `0.216783` | `0.215893` | `0.215883` | `0.216525` | `0.215735` |

## Interpretation

Faults are relative busts, not absolute high-error events: `regret_k = L_k - median(L_other_experts)`. The gate suppresses only experts predicted to be faulty by multiplying baseline HxV weights by `(1 - p_fault)^gamma` and renormalizing.

The Raw Forecast Control is the critical comparator: parity supports the fault-isolation hypothesis only if Passive+Parity improves beyond Passive and beyond Passive+Raw while also beating shuffled parity.

## Integrity

- Test loaded: `False`.
- Every cache/checkpoint path is refused if it contains `test`.
- Router_train detector predictions are chronological OOF with a horizon-12 purge.
- Fault thresholds, gamma, and intervention thresholds are selected on router_train OOF only.
- Router_val target corruption leaves passive, parity, and raw features unchanged exactly.
- Forecasting checkpoint hashes are recorded and unchanged.
