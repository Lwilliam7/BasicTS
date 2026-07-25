# COSTARTS Failure Mechanism Inspection

This is an inspection-only pass. No model architecture was changed and no training was run.

## Files Inspected
- `scripts/train_costarts_router.py`
- `scripts/router_experiment_config.py`
- `scripts/router_model_config.py`
- `scripts/router_diagnostics.py`
- `scripts/chronological_expert_training.py`

## Artifacts Loaded
- `cache/costarts_router_train_cache.pt`
- `cache/costarts_router_val_cache.pt`
- `checkpoints/costarts/best_costarts_router.pt`
- `checkpoints/costarts/last_costarts_router.pt`
- `results/router_summary/costarts/costarts_training_summary.json`
- `results/router_summary/costarts/costarts_training_curves.csv`

## Architecture And Inference Verification
- `COSTARTSRouter` is in `scripts/train_costarts_router.py` near line 74.
- `encode()` asserts histories are `[B, 96, 7]`, applies two Conv1d blocks, `AdaptiveAvgPool1d(1)`, projection to `[B,64]`, then L2-normalization.
- `forward()` emits `map_prediction [B,M]`, `ranking_logits [B,M]`, `query_logits [B,M]`, `mix_weights [B,M]`, `stop_logits [B,K]`, `query_order [B,K]`, and `stop_step [B]`.
- Shape assertions passed for `histories [B,96,7]`, `prediction_stack [B,12,7,M]`, `query_logits [B,M]`, `stop_logits [B,K]`, and `query_order [B,K]`.
- Queried-mask terms in COSTARTS train/diagnostic source: `0` total; queried mask in inference path: `False`.
- State-updater/feedback terms in COSTARTS train/diagnostic source: `0` total; state updater in inference path: `False`.
- Conclusion: current COSTARTS is a one-shot ranker/selector with a separate stop head, not a true sequential feedback router.

## Teacher Forcing And Targets
- Training computes `oracle_order = argsort(error_matrix)` and passes it as `teacher_forced_order` when `teacher_forcing=True`.
- Query CE target is `best_expert = argmin(error_matrix)`, so it learns the global oracle best expert, not the next best expert after an arbitrary queried subset.
- Stop target uses `_target_stop_index(error_matrix, oracle_order, expert_costs, K, stop_threshold)`.
- Saved run has `mix_forecast_loss_weight = 0.0`; mix loss disabled: `True`.
- Saved/loaded stop settings: `stop_threshold=0.0`, `cost_weights={}`.
- Stop target degenerate because threshold is zero and costs are empty: `True`.

## Reproduced Router-Val Metrics
- Stop target distribution, zero-based: `{'0': 613}`.
- Best checkpoint stop prediction, one-based: `{'1': 613}`.
- Last checkpoint stop prediction, one-based: `{'1': 613}`.
- Best stop average logits: `{'step_1': 0.4408074617385864, 'step_2': -0.22240851819515228, 'step_3': -0.26808828115463257, 'step_4': -0.1543753743171692, 'step_5': -0.07871966063976288}`.
- Best stop average probabilities: `{'step_1': 0.3171154856681824, 'step_2': 0.16335394978523254, 'step_3': 0.15605281293392181, 'step_4': 0.17485231161117554, 'step_5': 0.18862544000148773}`.
- Last stop average logits: `{'step_1': 0.7262682914733887, 'step_2': -0.49938252568244934, 'step_3': -0.5502247214317322, 'step_4': -0.45467182993888855, 'step_5': -0.32604876160621643}`.
- Last stop average probabilities: `{'step_1': 0.4486546516418457, 'step_2': 0.1316574215888977, 'step_3': 0.1251513510942459, 'step_4': 0.13769693672657013, 'step_5': 0.15683962404727936}`.
- Query top-1 oracle match: `0.234910280`.
- Query top-2 oracle coverage: `0.494290382`.
- Map argmin oracle match: `0.190864608`.
- Average Spearman predicted-vs-true expert error: `0.018270804`.
- Pairwise ranking accuracy: `0.508809149`.
- Selected MAE: `0.365392596`.
- Selected MSE: `0.335232645`.
- Regret to oracle: `0.047249347`.
- Selection counts: `{'DLinear': 297, 'PatchTST': 0, 'iTransformer': 0, 'TimesNet': 0, 'ModernTCN': 316}`.
- Saved summary best validation MAE: `0.365392613`; reproduced minus summary: `-0.000000029802`. The tiny mismatch is float aggregation/order only.

## Loss Magnitudes From Saved Curves
- `first_epoch` epoch 1: total=5.616727, map=0.917145, ranking=1.556839, query=1.573014, stop=1.569730, mix=0.000000, val_mae=0.365516, avg_stop=1.000.
- `best_val_epoch` epoch 4: total=4.746369, map=0.571813, ranking=1.499072, query=1.501960, stop=1.173524, mix=0.000000, val_mae=0.365393, avg_stop=1.000.
- `last_epoch` epoch 14: total=3.959770, map=0.279406, ranking=1.444879, query=1.443618, stop=0.791866, mix=0.000000, val_mae=0.372097, avg_stop=1.000.

## Confirmed Bugs
- No confirmed architecture/runtime bug was found in this inspection. Shape contracts pass and the validation loop is not forcibly setting K=1.

## Confirmed Objective Issues
- STOP supervision is degenerate: all router-val stop targets are class 0, so the learned stop head predicts step 1 for every window.
- The router has no queried mask and no state updater consuming queried expert forecasts, so it cannot learn recovery behavior after a bad first query.
- The query head is supervised toward the global best expert instead of the next best action for a reachable subset state.
- The final selection rule depends on `map_prediction`, but the predicted-error ranking metrics are near random.
- `mix_forecast_loss_weight=0.0`, so the optional forecast mixture head is not optimized for the deployed forecast metric.

## Likely Issues
- Multitask interference between map, ranking, query, and stop heads is likely, but not proven by these saved artifacts alone.
- The globally pooled one-vector history encoder may be too weak for per-window expert ranking, but that requires controlled experiments.
- Close expert margins make the per-window ranking target noisy; many windows have small oracle best-vs-second gaps.

## Files That Should Later Be Modified
- `scripts/train_costarts_router.py`: to add true subset-state/feedback training, queried masks, state updates, and non-degenerate stop supervision.
- `scripts/router_experiment_config.py` and `scripts/router_model_config.py`: to add subset-cache paths and sequential-router hyperparameters.
- `scripts/router_diagnostics.py`: to add subset-state diagnostics for future runs.
- Potential new training script/module for subset-state COSTARTS if preserving the current router unchanged is preferred.

## Exact Commands Used
- `rg -n "class COSTARTSRouter|def encode\(|def forward\(|teacher_forced_order|sampled_rollout|torch.topk|stop_logits|query_logits|map_prediction|mix_weights|def _target_stop_index|def costarts_losses|def _select_expert_from_outputs|def evaluate_costarts_router|def train_costarts_router|queried_mask|query_mask|state_updater|queried_forecasts|mix_forecast_loss_weight|cost_weights|stop_threshold" scripts/train_costarts_router.py`
- `rg -n "RouterExperimentConfig|stop_threshold|cost_weights|queried_experts_cap_k|routing_temperature|cache_paths|SUPPORTED_ROUTER_TYPES|SELECTED_MODELS|ROUTER_EXPERIMENT_CONFIG|mix_forecast_loss_weight" scripts/router_experiment_config.py scripts/router_model_config.py scripts/router_diagnostics.py scripts/chronological_expert_training.py`
- `python inline inspection script loading saved COSTARTS artifacts and writing subset_utility_inspection outputs`
