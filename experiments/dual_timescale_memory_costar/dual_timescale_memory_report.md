# Dual-Timescale (Short/Long) Memory COSTAR

Adds a second, slower EMA memory to both the global chronological branch and
the horizon-variable branch. Both memories are built causally from `router_train`
only, frozen before `router_val` for Frozen COSTAR, and updated causally during
`router_val` (only after `old_start + horizon <= current_start`) for Online COSTAR.

## Selection (router_train only)

| Dataset | short_decay | long_decay | chrono mix (train-selected fixed) | hv mix (train-selected fixed) | hv schedule (start,end) |
|---|---:|---:|---:|---:|---|
| ETTh1 | 0.95 | 0.99 | 0.3 | 0.1 | (0.5, 0.5) |
| ETTh2 | 0.8 | 0.995 | 0.7 | 0.5 | (0.5, 0.5) |

## Validation results

### ETTh1

| Method | Mode | MAE | MSE | Delta MAE vs original |
|---|---|---:|---:|---:|
| `original_single_memory` | frozen | `0.365825` | `0.308399` | `+0.000000` |
| `original_single_memory` | online | `0.363100` | `0.306026` | `+0.000000` |
| `dual_memory_short_only` | frozen | `0.366780` | `0.309844` | `+0.000954` |
| `dual_memory_short_only` | online | `0.363064` | `0.305923` | `-0.000036` |
| `dual_memory_long_only` | frozen | `0.365961` | `0.308675` | `+0.000135` |
| `dual_memory_long_only` | online | `0.363622` | `0.306463` | `+0.000522` |
| `dual_memory_fifty_fifty` | frozen | `0.366356` | `0.309242` | `+0.000530` |
| `dual_memory_fifty_fifty` | online | `0.363147` | `0.305943` | `+0.000047` |
| `dual_memory_train_selected_fixed` | frozen | `0.366048` | `0.308796` | `+0.000222` |
| `dual_memory_train_selected_fixed` | online | `0.363472` | `0.306302` | `+0.000372` |
| `dual_memory` | frozen | `0.366341` | `0.309224` | `+0.000516` |
| `dual_memory` | online | `0.363152` | `0.305938` | `+0.000052` |

- Causality audit: [{'dataset': 'ETTh1', 'mode': 'frozen', 'rows_checked': 0, 'violations': 0, 'result': 'PASS'}, {'dataset': 'ETTh1', 'mode': 'online', 'rows_checked': 5522, 'violations': 0, 'result': 'PASS'}]
- Invariant checks: {'frozen_unchanged_after_target_randomization': True, 'frozen_unchanged_after_mask_randomization': True, 'frozen_repeat_call_identical': True, 'online_changed_after_target_randomization': True}

### ETTh2

| Method | Mode | MAE | MSE | Delta MAE vs original |
|---|---|---:|---:|---:|
| `original_single_memory` | frozen | `0.277481` | `0.167632` | `+0.000000` |
| `original_single_memory` | online | `0.276832` | `0.167280` | `+0.000000` |
| `dual_memory_short_only` | frozen | `0.276128` | `0.165719` | `-0.001353` |
| `dual_memory_short_only` | online | `0.278580` | `0.168716` | `+0.001748` |
| `dual_memory_long_only` | frozen | `0.280493` | `0.171200` | `+0.003012` |
| `dual_memory_long_only` | online | `0.277181` | `0.167409` | `+0.000349` |
| `dual_memory_fifty_fifty` | frozen | `0.277079` | `0.167010` | `-0.000402` |
| `dual_memory_fifty_fifty` | online | `0.277404` | `0.167589` | `+0.000571` |
| `dual_memory_train_selected_fixed` | frozen | `0.277030` | `0.166966` | `-0.000450` |
| `dual_memory_train_selected_fixed` | online | `0.277411` | `0.167609` | `+0.000579` |
| `dual_memory` | frozen | `0.277030` | `0.166966` | `-0.000450` |
| `dual_memory` | online | `0.277411` | `0.167609` | `+0.000579` |

- Causality audit: [{'dataset': 'ETTh2', 'mode': 'frozen', 'rows_checked': 0, 'violations': 0, 'result': 'PASS'}, {'dataset': 'ETTh2', 'mode': 'online', 'rows_checked': 1202, 'violations': 0, 'result': 'PASS'}]
- Invariant checks: {'frozen_unchanged_after_target_randomization': True, 'frozen_unchanged_after_mask_randomization': True, 'frozen_repeat_call_identical': True, 'online_changed_after_target_randomization': True}

## Short-vs-long weight disagreement (router_val, per window)

Every router_val window is scored under the *same* dual-memory state read two ways: purely through the short EMA and purely through the long EMA. Disagreement between the two readings is measured by L1 distance, top-expert mismatch rate, correlation, and MAE on the highest-disagreement windows (top/bottom quartile by L1 distance).

| Dataset | Branch | Mean L1 | Top-expert mismatch rate | Per-expert corr. (across windows) | Overall corr. | Mean per-window corr. (n=3, noisy) |
|---|---|---:|---:|---|---:|---:|
| ETTh1 | chrono | `0.1154` | `0.3444` | PatchTST=+0.786, iTransformer=+0.721, TimesNet=+0.758 | `+0.8031` | `+0.6242` |
| ETTh1 | horizon-variable (mean-pooled) | `0.0908` | `0.1432` | PatchTST=+0.767, iTransformer=+0.745, TimesNet=+0.703 | `+0.9361` | `+0.9315` |
| ETTh2 | chrono | `0.1475` | `0.4437` | DLinear=+0.374, PatchTST=+0.473, ModernTCN=+0.217 | `+0.7672` | `+0.7531` |
| ETTh2 | horizon-variable (mean-pooled) | `0.1182` | `0.2561` | DLinear=+0.351, PatchTST=+0.348, ModernTCN=+0.165 | `+0.7929` | `+0.8391` |

MAE on high- vs low-disagreement windows (top/bottom quartile by chronological-branch L1 distance):

| Dataset | Method | High-disagreement MAE | Low-disagreement MAE | All-windows MAE | High minus low |
|---|---|---:|---:|---:|---:|
| ETTh1 | dual_memory (online) | `0.359950` | `0.349535` | `0.363152` | `+0.010416` |
| ETTh1 | original_single_memory (online) | `0.360051` | `0.349323` | `0.363100` | `+0.010728` |
| ETTh2 | dual_memory (online) | `0.319270` | `0.260325` | `0.277411` | `+0.058945` |
| ETTh2 | original_single_memory (online) | `0.319270` | `0.258783` | `0.276832` | `+0.060487` |

## Short vs long: which memory actually produces the better forecast?

Diagnostic only, computed once on router_val with nothing tuned against it. For every window, the full online dual-memory pipeline (base branches + specialists) is built twice from the *same* `causal_dual_state_walk` per branch: once reading only the short EMA, once reading only the long EMA. `delta = short_only_mae - long_only_mae`; `delta < 0` means short memory won that window, `delta > 0` means long memory won.

| Dataset | Condition | Windows | Short win rate | Long win rate | Tie rate | Mean (short-long) MAE |
|---|---|---:|---:|---:|---:|---:|
| ETTh1 | All router_val windows | 2773 | 0.554 | 0.446 | 0.000 | -0.000557 |
| ETTh1 | Chrono top-expert differs | 955 | 0.548 | 0.452 | 0.000 | -0.000731 |
| ETTh1 | Chrono top-25% L1 disagreement | 693 | 0.545 | 0.455 | 0.000 | -0.000611 |
| ETTh1 | HV (mean-pooled) top-expert differs | 397 | 0.549 | 0.451 | 0.000 | -0.001050 |
| ETTh1 | HV (mean-pooled) top-25% L1 disagreement | 693 | 0.532 | 0.468 | 0.000 | -0.000854 |
| ETTh2 | All router_val windows | 613 | 0.483 | 0.517 | 0.000 | +0.001399 |
| ETTh2 | Chrono top-expert differs | 272 | 0.445 | 0.555 | 0.000 | +0.002305 |
| ETTh2 | Chrono top-25% L1 disagreement | 153 | 0.542 | 0.458 | 0.000 | +0.000045 |
| ETTh2 | HV (mean-pooled) top-expert differs | 157 | 0.484 | 0.516 | 0.000 | +0.001847 |
| ETTh2 | HV (mean-pooled) top-25% L1 disagreement | 153 | 0.471 | 0.529 | 0.000 | +0.002534 |

| Dataset | Mean MAE short-only | Mean MAE long-only | Avg margin when short wins | Avg margin when long wins |
|---|---:|---:|---:|---:|
| ETTh1 | `0.363064` | `0.363622` | +0.005073 | +0.005058 |
| ETTh2 | `0.278580` | `0.277181` | +0.004200 | +0.006627 |

### Does short-vs-long disagreement tell us which memory should be trusted?

- **ETTh1**: On the highest-disagreement windows, short wins 0.545 of the time (chrono split) and 0.532 of the time (HV split) -- close to 50/50. Disagreement alone does not identify which memory to trust here, so **dynamic mixing is not justified by this signal alone** for this dataset.
- **ETTh2**: On the highest-disagreement windows, short wins 0.542 of the time (chrono split) and 0.471 of the time (HV split) -- close to 50/50. Disagreement alone does not identify which memory to trust here, so **dynamic mixing is not justified by this signal alone** for this dataset.

## Hard rule compliance

```text
TEST SET ACCESSED: NO
TEST CACHE LOADED: NO
TEST METRICS COMPUTED: NO
```

## Reproduce

```powershell
python experiments\dual_timescale_memory_costar\run_dual_timescale_memory_costar.py --device cpu
```
