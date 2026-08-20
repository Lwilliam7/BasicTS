# Step 1 Audit: Deployment-Time Target Feedback in COSTAR

Written before any new code was created. Nothing existing was modified.

## Target-feedback paths found

| Location | Mechanism | Removed for `frozen_hv`? |
|---|---|---|
| `experiments/chronological_adaptive_costar/run_chronological_adaptive_costar.py` :: `chronological_online_weights()` | Walks forward through evaluation windows; for each elapsed window `j` (once `starts[j] + horizon <= now`), folds `expert_mae[j]` -- computed from **evaluation-split targets** -- into a single per-expert EMA (`ema = decay*ema + (1-decay)*loss`), then reads a fresh weight from that EMA for every subsequent window. | Not used at all by `frozen_hv` (the global chronological branch is out of scope for this HxV-only baseline, matching Step 4). |
| `experiments/horizon_variable_adaptive_costar/run_hv_adaptive_costar.py` :: `chronological_hv_weights()` | Identical pattern at H x V x expert granularity: `ema = decay*ema + (1-decay)*val_err[j]` inside the same walk-forward loop, where `val_err` comes from `per_location_error(val_cache, ...)`. **This is the primary path removed.** | Removed. `frozen_hv_costar` never calls this function on `router_val`/`test`; it is only used to reproduce the existing "Online HxV COSTAR" baseline for comparison. |
| `experiments/expanded_expert_pool_costar/run_expanded_expert_pool.py` :: `run_causal_specialists()`, and `etth2...run_specialists_no_duplicate()` | A second, independent target-feedback path: DLinear/ModernTCN "specialist" advantage EMA updated from realized evaluation-split prediction errors. | Out of scope for this first (Step 4) baseline -- no specialists are used at all, same exclusion as the earlier router-ablation study. |
| `enforce_observable(old_start, current_start, horizon)` | The causal *gate*. Makes the online loop non-leaking (no future information), but does not make it offline -- the loop still exists and still adapts from realized evaluation-split labels. | Not weakened or removed -- the whole loop it gates is simply never invoked on evaluation data. Loosening the gate while keeping the loop would be leakage; that is explicitly NOT what was done. |
| `experiments/frozen_costar/run_frozen_costar_validation.py` :: `frozen_costar_base_prediction()` / `frozen_hv_weights()` | Existing precedent: already freezes the *full* blended pipeline (chrono 0.25 + HxV 0.75 + specialists) at router_train-derived state. | Not reused directly -- this experiment builds an isolated, minimal HxV-only frozen baseline as Step 4 explicitly requests, distinct from that fuller pipeline. |
| Everything downstream (`dual_timescale_memory_costar`, `costar_router_ablation`, `costar_multidataset_frozen`) | All call `chronological_hv_weights()` / `chronological_online_weights()` whenever `online=True`. | Unaffected; none of that code was touched. |

## What was NOT done

The causal gate `enforce_observable()` was not removed, loosened, or bypassed anywhere. The frozen router never reads a future (or any) validation/test label to produce a prediction -- not "sees them late," not "sees them at all." This is the distinction the task asked to preserve: **non-causal (forbidden) vs. non-online (requested)** are different things, and only the latter was implemented.

## What was built instead

`experiments/frozen_hv_costar/run_frozen_hv_costar.py`:

```
router_train per-location (H x V x expert) errors
    -> mean over router_train windows            [aggregation step, unchanged from the training-side procedure]
    -> errors_to_weights()                        [same softmax/temperature rule as the online path -- unchanged]
    -> FROZEN H x V x expert weight tensor
    -> repeated unchanged for every router_val window
```

No loop, no `enforce_observable` call, no state that survives past the single weight computation. Verified explicitly (Step 6, `verification_tests.csv`): byte-identical predictions when validation targets are perturbed (early window, all windows), predictions succeed without validation targets ever being loaded, predictions are invariant to window order, and the frozen weight tensor is bit-identical before and after a full validation pass. The same tests are run against `online_hv` as a negative control -- it correctly *fails* the target-perturbation tests, confirming the test methodology actually discriminates online from frozen behavior rather than passing trivially.
