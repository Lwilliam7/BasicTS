# COSTAR-TS Architecture Notes

Last updated: 2026-08-17

This file describes the active and historical implementations in this repository. "Current best" means best ETTh1 validation result, not final test performance.

## Shared Data Interface

Relevant files:

- `scripts/build_costarts_walkforward_cache.py`
- `scripts/train_costarts_walkforward_experts.py`
- `cache/costarts_walkforward/router_train_20_60_cache.pt`
- `cache/costarts_walkforward/router_val_60_80_cache.pt`

Cache tensors:

- `histories`: `[num_windows, 96, 7]`
- `targets`: `[num_windows, 12, 7]`
- `target_masks`: `[num_windows, 12, 7]`
- `prediction_stack`: `[num_windows, 12, 7, num_experts]`
- `expert_names`: `DLinear`, `PatchTST`, `iTransformer`, `TimesNet`, `ModernTCN`
- `absolute_window_starts`: chronological window starts

Normalization:

- Metrics use the stored expert checkpoint scaler, commonly `checkpoints/costarts_walkforward/final_60/DLinear/best_expert.pt`.

Leakage restrictions:

- Router train uses OOS forecasts over the `20-60%` region.
- Router validation uses `60-80%`.
- Test is `80-100%` and must not be used without explicit authorization.
- Online validation updates can only use an old label when `old_start + horizon <= current_start`.

## Frozen Expert Components

The forecasting experts are trained separately and then frozen for router/adaptation experiments:

- `DLinear`
- `PatchTST`
- `iTransformer`
- `TimesNet`
- `ModernTCN`

Most successful methods use only the strong fixed-3 subset:

- `PatchTST`
- `iTransformer`
- `TimesNet`

## Historical Sequential COSTAR Router

Relevant files:

- `scripts/sequential_costarts_model.py`
- `scripts/sequential_costarts_model_full.py`
- `scripts/sequential_costarts_transformer_model.py`
- `scripts/train_sequential_costarts_utility_ranking.py`
- `scripts/train_sequential_costarts_forecast_state.py`

Core idea:

1. Start with no queried experts.
2. Encode history and sequential state.
3. Score unqueried experts.
4. Query an expert forecast.
5. Update ensemble and queried mask.
6. Stop when score threshold says no more useful query, or max queries reached.

State features typically include:

- input history encoding
- queried expert mask
- number queried
- current ensemble forecast or forecast summary
- disagreement/spread features
- optional queried forecast sequences
- optional expert embeddings

Training supervision:

- marginal utility of querying each available expert
- listwise utility ranking
- STOP-aware listwise variant
- pairwise or weighted pairwise utility ranking

Status:

- Historical and diagnostic, not current best. Weighted pairwise is the best verified sequential result but remains worse than fixed-3.

## Oracle-Weight Prototype Residual Router

Relevant files:

- `experiments/oracle_weight_tournament/run_tournament.py`
- `experiments/oracle_weight_tournament/final_report.json`

Core idea:

1. On router-train only, compute oracle convex weights over `PatchTST`, `iTransformer`, `TimesNet`.
2. Cluster train oracle weights into prototypes.
3. Train a small history/forecast encoder to predict prototype weights plus a small residual.
4. Evaluate on validation without using validation targets for training labels.

Status:

- Former best: MAE `0.366028 +/- 0.000242`.
- Important bridge from fixed-3 to chronological adaptation.

## Chronological Adaptive COSTAR

Relevant files:

- `experiments/chronological_adaptive_costar/run_chronological_adaptive_costar.py`
- `experiments/chronological_adaptive_costar/final_report.json`

Core idea:

1. At validation time `t`, predict using only information available at `t`.
2. Maintain causal online expert performance state from old fully observed windows.
3. Convert recent expert errors into weights using EMA, Hedge, or rolling windows.
4. Blend online weights with the prototype-residual router.

Winning setting:

- EMA decay `0.97`
- temperature `0.1`
- alpha `0.5`

Status:

- Former best: MAE `0.365534 +/- 0.000112`.

## Horizon x Variable Adaptive Weighting

Relevant files:

- `experiments/horizon_variable_adaptive_costar/run_hv_adaptive_costar.py`
- `experiments/horizon_variable_adaptive_costar/final_report.json`

Core idea:

1. Track causal recent expert absolute error per horizon, variable, and expert.
2. Use low-rank structure to avoid free `12 x 7 x 3` weights.
3. Convert recent error surfaces into horizon x variable expert weights.
4. Blend with the chronological adaptive baseline.

Winning setting:

- family: horizon x variable EMA hybrid
- mode: low-rank horizon x variable
- rank: `1`
- decay: `0.95`
- temperature: `0.1`
- alpha: `0.75`

Status:

- Former best validation result: MAE `0.363642 +/- 0.000014`, MSE `0.306712 +/- 0.000016`.
- Target `0.3619` not yet reached.

## Conservative Ridge Residual Correction

Relevant files:

- `experiments/residual_correction_costar/run_residual_correction_experiments.py`
- `experiments/residual_correction_costar/final_report.json`

Core idea:

1. Freeze the horizon-variable adaptive predictor `hybrid_chrono_hvema_lowrank1_decay0.95_temp0.1_alpha0.75`.
2. Select residual-correction hyperparameters only on chronological router-train folds.
3. Predict a small normalized residual delta from forecast-time features:
   - baseline and expert forecasts
   - expert disagreement/dispersion
   - causal residual mean/variance
   - horizon and variable identity
   - causal multi-scale history summaries at `96/192/336/720`
4. Apply `final_prediction = baseline_prediction + alpha * clipped_delta`.

Winning setting:

- Ridge regression
- ridge penalty `1.0`
- correction strength `alpha=0.1`
- clipping `0.25` train residual std

Status:

- Former best validation result: MAE `0.363301 +/- 0.000015`, MSE `0.306286 +/- 0.000017`.
- Aggregate paired bootstrap CI vs `0.363642`: `[-0.000378, -0.000305]`.
- Target `0.3619` not reached.
- Tiny MLP residual correction was tested but was less stable and slightly worse.

## Current Best: Expanded Optional Expert Specialists

Relevant files:

- `experiments/expanded_expert_pool_costar/run_expanded_expert_pool.py`
- `experiments/expanded_expert_pool_costar/final_report.json`
- `experiments/train_selected_core_etth1/run_train_selected_core_eval.py`
- `experiments/train_selected_core_etth1_equal_static/final_report.json`

Core idea:

1. Preserve the fixed-three horizon-variable baseline prediction.
2. Track causal recent absolute error for that baseline, DLinear, and ModernTCN.
3. Activate DLinear and/or ModernTCN only when recent causal relative advantage exceeds the selected margin.
4. Combine convexly:
   - `final = (1 - weight_D - weight_M) * base + weight_D * DLinear + weight_M * ModernTCN`
   - `weight_D >= 0`, `weight_M >= 0`, `weight_D + weight_M <= cap`

Winning setting:

- Scenario: both optional experts
- structure: variable-specific
- EMA decay: `0.95`
- combined cap: `0.10`
- required relative advantage: `2.0%`
- warm-up: `96`

Status:

- Main full adaptive model for ETTh1: equal static weights for every selected triple.
- Active equal-static validation result: MAE `0.363100`, MSE `0.306026`.
- Active equal-static after-final-test audit result: MAE `0.326408`, MSE `0.267378`.
- Historical pre-test development result: MAE `0.363112 +/- 0.000013`, MSE `0.306057 +/- 0.000016`.
- Aggregate paired bootstrap CI vs fixed-three HV `0.363642`: `[-0.000557, -0.000502]`.
- Paired CI vs prior ridge residual best `0.363301`: `[-0.000233, -0.000143]`.
- DLinear and ModernTCN are backup specialists only, not equal ensemble members.
- The active implementation now uses equal static weights for every selected triple; the old ETTh1 fixed-three neural static-prior checkpoint is not loaded.

## Grokking Diagnostic Runner

Relevant files:

- `experiments/grokking_diagnostic_costar/run_grokking_diagnostic.py`
- `experiments/grokking_diagnostic_costar/final_report.json`

Core idea:

1. Select the strongest neural trainable fixed-three router, `final_phase2_protores_lam0.01_k16_scale0.3_rw0.001`.
2. Train only on an early chronological fold inside router-train.
3. Evaluate every epoch on a later chronological fold inside router-train.
4. Compare original-duration epoch `10` to 10x duration epoch `100` across weight decays `0.001`, `0.01`, and `0.1`.

Status:

- No validation or test cache is loaded.
- No grokking observed: best fold MAE occurred at epoch `26`; delayed/epoch-100 checkpoints degraded.

## RLS / Online Ridge Stacking

Relevant file:

- `experiments/horizon_variable_adaptive_costar/run_hv_adaptive_costar.py`

Core idea:

- Maintain causal linear stacking coefficients with intercepts for global, horizon, variable, or horizon x variable groups.
- Update when old targets become observable.

Status:

- Implemented in the broad horizon-variable runner.
- Not the current winner in verified summaries.

## Inference Loop For Active Online Methods

For each validation window in chronological order:

1. Read `current_start`.
2. Release only pending windows where `old_start + horizon <= current_start`.
3. Update recent-performance state from released errors.
4. Compute weights for current forecasts.
5. Produce prediction.
6. Add current index to pending queue.

This pattern is enforced by helper checks such as `enforce_observable` in the chronological runners.
