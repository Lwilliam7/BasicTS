# COSTARTS Repository Inspection Report

Generated from repository inspection only. I did not edit project code and did not retrain COSTARTS. I loaded the saved COSTARTS caches/checkpoints and wrote inspection artifacts under `results/router_summary/costarts/inspection/`.

## Executive Findings

1. The current COSTARTS implementation is not a full Context -> Action -> Feedback sequential router. It encodes the history once into one `[B, 64]` window embedding, predicts a full query order from that embedding, and predicts a stop step from the same embedding. There is no state updater, no queried-expert mask, no remaining-budget feature, and no mechanism that consumes the forecast returned by a queried expert.
2. The immediate stopping behavior is explained by the current objective. With `STOP_THRESHOLD = 0.0` and empty `cost_weights`, the stop target is always class `0`, meaning stop after the first query. On validation, target distribution is `{'0': 613}` and prediction distribution is `{'1': 613}`.
3. The predicted-error head is effectively not ranking experts well. Validation map-vs-true Spearman is `0.0183`, pairwise ranking accuracy is `0.5088`, and map argmin matches the oracle only `19.09%`.
4. The COSTARTS shortlist is useful, but the final selection rule wastes it. Oracle-best within COSTARTS top 2 is `0.339212` MAE, while choosing within that top 2 by predicted error is `0.370614` MAE.
5. The saved same-validation comparison file contains a labeling/selection inconsistency: it labels `best fixed expert (DLinear)` at MAE `0.370099`, but the current COSTARTS validation cache shows the best fixed expert is `ModernTCN` with MAE `0.358645`. I did not alter that existing file.

## Files And Roles

- `scripts/train_costarts_router.py`: main COSTARTS implementation. `COSTARTSTrainingConfig` starts at line 48; `COSTARTSRouter` at line 74; `encode` at line 136; `forward` at line 143; cache validation/building at lines 286 and 327; stop target/loss at lines 462 and 479; inference selection at line 537; validation at line 552; training at line 642.
- `scripts/router_experiment_config.py`: central config. `RouterExperimentConfig` starts at line 45; loading at line 127; validation at line 256; runtime printing at line 323. It supports `router_type='costarts'`, cache paths, K, temperature, dimensions, losses, stop threshold, and costs.
- `scripts/router_model_config.py`: user-editable config. Current defaults include `selected_expert_models=('DLinear','PatchTST')`, `auto_select_best_by_size=True`, `best_model_counts='all'`, `queried_experts_cap_k=None`, `stop_threshold=0.0`, and COSTARTS cache paths.
- `scripts/chronological_expert_training.py`: shared data/expert utilities. Constants `[96,12,7]` are at lines 25-27; chronological split fractions at lines 17-22; `load_full_chronological_data` at line 688; `prepare_chronological_dataloaders` at line 939; `_prepare_forecasting_batch` at line 1055; `load_and_freeze_expert` at line 2056; `assert_experts_frozen` at line 2287; `_call_expert_model` at line 2870; `build_selected_candidate_experts` at line 2914.
- `scripts/router_diagnostics.py`: diagnostics for COSTARTS and other routers. It loads COSTARTS checkpoints at line 286, collects predictions at line 300, selects experts at line 329, and plots predicted-vs-true errors around line 428.
- `scripts/router_robustness.py`: robustness suite for routing baselines/static/oracle methods. It is related infrastructure but not a trained COSTARTS evaluator.
- `notebooks/router2(RouterDC inspired).ipynb`: notebook-only RouterDC hard selector. `RouterDCHardRouter` appears around notebook source line 948, hard training around line 1546, and final test around line 1631.

Search results: `costarts`, `stop_logits`, `query_logits`, `predicted_error`, `ranking_logits`, `teacher_forcing`, `sampled_rollout`, `stop_loss`, `regret`, `expert_cost`, `oracle`, `rollout`, and `topk` exist in the files above. No `queried_mask`, `query_mask`, `state_updater`, `marginal_gain`, `trajectory`, `current_best`, or `remaining_budget` occurrence was found in `scripts/train_costarts_router.py`.

## Exact COSTARTS Architecture

Source: `scripts/train_costarts_router.py:COSTARTSRouter` lines 74-210.

Input history is `[B, 96, 7]`. The history encoder is:

```python
Conv1d(7, 64, kernel_size=5, padding=2)
GELU()
GroupNorm(1, 64)
Conv1d(64, 64, kernel_size=5, padding=4, dilation=2)
GELU()
GroupNorm(1, 64)
AdaptiveAvgPool1d(1)
Linear(64, 64)
GELU()
LayerNorm(64)
L2 normalize
```

There is no dropout. The `AdaptiveAvgPool1d(1)` collapses all 96 timestamps to one vector. The same vector feeds every head.

|Tensor|Code location|Shape|Meaning|
|---|---|---|---|
|history input|cache / router input|`[B, 96, 7]`|normalized z-score|
|encoded CNN output after pool|`COSTARTSRouter.encode`|`[B, 64]`|single window embedding|
|query_embedding|`COSTARTSRouter.forward`|`[B, 64]`|L2-normalized|
|expert_embeddings|`COSTARTSRouter`|`[M, 64]`|trainable, L2-normalized at use|
|similarity_logits|`forward`|`[B, M]`|cosine / temperature|
|map_prediction|`map_head + softplus`|`[B, M]`|predicted absolute expert MAE-like value|
|ranking_logits|`ranking_head + similarity`|`[B, M]`|CE target is global oracle best expert|
|query_logits|`query_head + similarity`|`[B, M]`|top-k or sampled query order|
|mix_weights|`mix_head softmax`|`[B, M]`|exists but not trained when mix weight is 0|
|stop_logits|`stop_head`|`[B, K]`|class 0 means stop after first queried expert|
|query_order|`teacher/topk/multinomial`|`[B, K]`|ordered expert indices|
|selected expert|`_select_expert_from_outputs`|`[B]`|lowest predicted error among queried prefix|

Expert embeddings are trainable `[M, 64]` vectors initialized with normal std `0.02`; they are normalized and compared to the normalized query embedding by cosine similarity. Similarities are added to the ranking and query logits.

Forward behavior:

- Before any expert is queried: encode history -> compute `[B,M]` scores/heads.
- After first expert is queried: no new code path exists. The state does not change.
- After second expert is queried: no new code path exists. The state does not change.
- STOP: `stop_step = argmax(stop_logits)+1`; selection uses the prefix `query_order[:stop_step]`, then picks the queried expert with lowest `map_prediction`.

Raw expert predictions, disagreement features, expert forecasts, query order history, current best forecast, remaining budget, uncertainty, and queried masks are not inputs to COSTARTSRouter. The offline cache contains predictions, but the router forward pass does not consume them.

## Expert Cache Inspection

Caches inspected:

- `cache/costarts_router_train_cache.pt`
- `cache/costarts_router_val_cache.pt`

Top-level cache keys are: `best_expert`, `error_matrix`, `error_temperature`, `expert_names`, `forecast_horizon`, `histories`, `input_len`, `mse_matrix`, `num_features`, `num_windows`, `prediction_stack`, `sample_indices`, `split_role`, `target_masks`, `target_probabilities`, `targets`.

Train cache samples: `2053`. Validation cache samples: `613`. Expert order is `['DLinear', 'PatchTST', 'iTransformer', 'TimesNet', 'ModernTCN']` in both. `sample_indices_contiguous=True` for both, so the saved cache order is aligned with the non-shuffled chronological loader order.

Tensor shapes in validation cache:

- `histories`: `[613, 96, 7]`
- `targets`: `[613, 12, 7]`
- `target_masks`: `[613, 12, 7]`
- `prediction_stack`: `[613, 12, 7, 5]`
- `error_matrix`: `[613, 5]`
- `mse_matrix`: `[613, 5]`
- `target_probabilities`: `[613, 5]`
- `best_expert`: `[613]`

Dtypes include float32 for histories, targets, predictions, error matrices, and bool for target masks. NaN counts are zero for every floating tensor and inf counts are zero for every floating tensor. No duplicated expert prediction pairs were detected by exact all-close comparison.

Histories/targets/predictions/errors are in normalized units. Evidence: `prepare_chronological_dataloaders` fits one `ZScoreScaler` on `expert_train`; `_prepare_forecasting_batch` transforms both inputs and targets before cache generation; errors are computed after that transform. Validation history mean/std/min/max: `[0.0644, 0.7192, -2.7412, 5.1539]`. Validation target mean/std/min/max: `[0.0133, 0.7224, -2.7412, 5.1539]`.

Chronological/test separation: `ChronologicalForecastingDataset` keeps windows wholly inside its split; split fractions are expert train 0-50%, expert val 50-60%, router train 60-75%, router val 75-80%, test 80-100%. `build_costarts_expert_cache` rejects any split except `router_train` or `router_val`, so the COSTARTS caches do not include test samples.

Per-expert validation statistics:

|Expert|Mean MAE|MAE Std|Oracle Winner %|Oracle Top-2 Member %|Oracle Top-3 Member %|
|---|---|---|---|---|---|
|DLinear|0.370099|0.189694|19.09%|41.60%|62.48%|
|PatchTST|0.370266|0.180773|14.19%|37.52%|57.59%|
|iTransformer|0.390501|0.205030|14.52%|28.87%|48.29%|
|TimesNet|0.374062|0.174229|22.68%|39.31%|59.54%|
|ModernTCN|0.358645|0.168983|29.53%|52.69%|72.10%|

Validation oracle MAE: `0.318143`. Best fixed expert on this cache: `ModernTCN` at `0.358645` MAE.

Best-vs-second-best margin: mean `0.028640`, median `0.020806`. The best and second-best experts are within 0.02 MAE on `48.29%` of windows and within 0.05 MAE on `83.52%` of windows. This makes the ranking problem noisy and close-margin heavy.

Expert error correlation matrix on validation:

|Expert|DLinear|PatchTST|iTransformer|TimesNet|ModernTCN|
|---|---|---|---|---|---|
|DLinear|1.000|0.894|0.834|0.896|0.896|
|PatchTST|0.894|1.000|0.875|0.895|0.882|
|iTransformer|0.834|0.875|1.000|0.860|0.840|
|TimesNet|0.896|0.895|0.860|1.000|0.918|
|ModernTCN|0.896|0.882|0.840|0.918|1.000|

## Target Generation And Losses

Source: `scripts/train_costarts_router.py` lines 413, 449-525.

- Expert error target: cache computes per-window MAE and MSE from `prediction_stack [B,12,7,M]`, `targets [B,12,7]`, and `target_mask [B,12,7]`. Shape `[B,M]`. True future targets are used offline only.
- Soft suitability labels: `target_probabilities = softmax(-error_matrix / error_temperature)`, shape `[B,M]`, created in cache but not used by `costarts_losses`.
- Oracle best expert: `best_expert = argmin(error_matrix)`, shape `[B]`, hard target.
- Oracle query order: `argsort(error_matrix)`, shape `[B,M]`, hard true-error order.
- Stop target: compute ordered errors + cumulative expert costs, then choose first index whose utility is within `stop_threshold` of the best utility. With zero costs and threshold 0, this is always index 0.
- Map regression loss: `MSE(map_prediction / row_mean_error, error_matrix / row_mean_error)`.
- Ranking loss: `cross_entropy(ranking_logits, best_expert)`.
- Query loss: `cross_entropy(query_logits, best_expert)`.
- Stop loss: `cross_entropy(stop_logits, stop_target)`.
- Optional mix forecast loss: MAE of `sum(prediction_stack * mix_weights)` against target. It is disabled in the saved run because `mix_forecast_loss_weight=0.0`.

Saved loss weights: map=1, ranking=1, query=1, stop=1, mix=0. No entropy, contrastive, marginal-utility, expected-regret, or cost-normalization loss is implemented for COSTARTS.

Specific target answers:

- First teacher query is always the true oracle-best expert because teacher order is `argsort(error_matrix)`.
- The next teacher query is selected using true future errors, but the code only passes a full teacher order; it does not build stepwise states.
- The teacher never intentionally begins with a non-oracle expert.
- Trajectories are not explicitly generated; training uses one oracle-sorted order per sample.
- States for arbitrary queried subsets are not generated.
- Training does not include model-generated states when `sampled_rollout=False` in the saved run.
- STOP is labeled immediately after the oracle-best expert is queried under current zero-cost settings.
- STOP supervision is maximally imbalanced: all `613` validation windows target stop index 0.
- The query head learns the global best expert, not the next best remaining expert.
- Ties and small margins are not handled specially.
- Expert costs are added directly to normalized MAE. No scale normalization is applied.

## Teacher Forcing, Rollout, And Inference

Training loop lines 796-804 compute `oracle_order` and pass it as `teacher_forced_order` when `teacher_forcing=True`. Therefore the returned `query_order` during training is oracle order, not model order. Losses still supervise `query_logits` and `ranking_logits` against the global best expert.

Inference/validation calls `router(history, sampled_rollout=False)` and uses `topk(query_logits, k=K)`. There is no already-queried mask beyond top-k returning unique indices. STOP is not a competing `M+1` action; it is a separate `K`-class head predicting how many entries of the precomputed query order to consider.

Maximum query count is `K=min(queried_experts_cap_k, M)`, here `5`. There is no minimum-query hyperparameter except the implementation clamps stop to at least 1 query. There is no scheduled sampling. `sampled_rollout=True` would sample a full order from initial query logits during training, but still has no feedback state.

Exposure-bias finding: training uses oracle order in the `query_order` output, while validation uses model top-k order. Since no per-step state exists, this is not classic recurrent exposure bias; it is an objective mismatch between supervised oracle-first behavior and deployable model-chosen behavior.

## Stop Mechanism Inspection

Stop head input is the same normalized `[B,64]` query embedding. Architecture is `Linear(64,K)`. Class index 0 means stop after one queried expert; class index 4 means stop after five experts when M=K=5. Inference uses `argmax`, not a threshold.

Best checkpoint stop diagnostics:

- stop logit means by step: `[0.4408, -0.2224, -0.2681, -0.1544, -0.0787]`
- stop probability means by step: `[0.3171, 0.1634, 0.1561, 0.1749, 0.1886]`
- stop-step counts: `{'1': 613}`

Last checkpoint stop diagnostics:

- stop probability means by step: `[0.4487, 0.1317, 0.1252, 0.1377, 0.1568]`
- stop-step counts: `{'1': 613}`

Bug checklist:

- No evidence that all experts are accidentally masked after one query; no queried mask exists.
- No mask-broadcast bug found in STOP; no mask exists in the stop mechanism.
- STOP threshold is not reversed in code; the design makes step 1 optimal under zero costs.
- STOP label convention is consistent: class 0 -> stop after first, then `+1` in inference.
- No sigmoid/BCE double-application bug; stop uses raw logits with cross entropy.
- Stop target is always zero under current config, confirmed on validation.
- Stop loss weight is 1.0, not numerically dominant at epoch 1, but the target itself is degenerate.
- No expert-cost coefficient makes every marginal utility negative; cost weights are empty.
- No validation loop forces stop after one query; the checkpoint's learned logits choose step 1 for all windows.
- Maximum query count is not accidentally set to one; K is 5.

Stop precision/recall/accuracy for the only current positive class `stop after first`: accuracy 100%, recall 100%, precision 100%, but this is not a useful success metric because all labels are the same.

## Predicted-Error And Ranking Heads

Validation diagnostics from the best checkpoint:

- selected MAE `0.365393` vs oracle `0.318143`, regret `0.047249`
- selected expert equals oracle winner `23.49%`
- query top-1 oracle match `23.49%`
- query top-2 oracle coverage `49.43%`
- query top-3 oracle coverage `71.62%`
- ranking-logit top-1 oracle match `24.96%`
- map argmin oracle match `19.09%`
- map-vs-true Spearman `0.0183`
- map pairwise ranking accuracy `0.5088`

Predicted-error scale is badly calibrated relative to true errors. True validation MAEs range around 0.36-0.39 per expert, but predicted means are `DLinear: 0.526082, PatchTST: 0.620706, iTransformer: 0.571495, TimesNet: 0.581431, ModernTCN: 0.571981` with tiny stds `DLinear: 0.004155, PatchTST: 0.012468, iTransformer: 0.006273, TimesNet: 0.008131, ModernTCN: 0.007174`. This is evidence of collapse toward near-constant expert-specific values, not useful per-window error estimation.

Selection distribution from COSTARTS first query: DLinear `48.45%`, ModernTCN `51.55%`, and 0% for PatchTST/iTransformer/TimesNet. This is much narrower than the oracle winner distribution.

Concrete high-regret validation examples are saved at `results/router_summary/costarts/inspection/worst_costarts_regret_examples.csv`. First five:

- sample 400: oracle=ModernTCN (0.9070), selected=DLinear (2.3813), regret=1.4743, query_order=DLinear > ModernTCN > TimesNet > iTransformer > PatchTST, stop_step=1.
- sample 280: oracle=ModernTCN (0.7193), selected=DLinear (1.8645), regret=1.1452, query_order=DLinear > ModernTCN > TimesNet > PatchTST > iTransformer, stop_step=1.
- sample 26: oracle=ModernTCN (0.9499), selected=DLinear (1.8809), regret=0.9310, query_order=DLinear > ModernTCN > TimesNet > iTransformer > PatchTST, stop_step=1.
- sample 510: oracle=ModernTCN (0.9737), selected=DLinear (1.8851), regret=0.9114, query_order=DLinear > ModernTCN > TimesNet > PatchTST > iTransformer, stop_step=1.
- sample 339: oracle=ModernTCN (0.6523), selected=DLinear (1.5022), regret=0.8499, query_order=DLinear > ModernTCN > TimesNet > PatchTST > iTransformer, stop_step=1.

The full CSV also contains the true and predicted error for every expert in each example.

## Reproduced Diagnostics And Counterfactuals

All rows below were computed from `checkpoints/costarts/best_costarts_router.pt` and `cache/costarts_router_val_cache.pt` without retraining.

|Case|MAE|MSE|Regret|Avg Experts Used|
|---|---|---|---|---|
|Best fixed expert on COSTARTS val cache|0.358645|0.327452||1|
|COSTARTS forced 1 query|0.365393|0.335233|0.047249|1|
|COSTARTS top 2, choose by predicted error|0.370614|0.343729|0.052471|2|
|Oracle best within COSTARTS top 2|0.339212|0.296081|0.021069|2|
|Oracle best within COSTARTS top 3|0.328340|0.278859|0.010197|3|
|Oracle best within COSTARTS top 5|0.318143|0.253928|0.000000|5|
|Equal average of COSTARTS top 2 forecasts|0.353071|||2|
|Current untrained mix head over all experts|0.349396||||
|Full oracle best expert per window|0.318143|0.247970||5|
|Oracle per horizon-variable expert upper bound|0.216561||||

Interpretation: stopping is a real problem, but not the only problem. If an oracle chooses within COSTARTS top 2, MAE improves to `0.339212`; if the current predicted-error head chooses within top 2, MAE worsens to `0.370614`. Therefore the final queried-expert selection/error-ranking head is a major bottleneck. Equal-averaging top 2 (`0.353071`) beats the current predicted-error choice and is close to RouterDC hard contrastive (`0.354483`).

Existing same-val comparison file rows:

|Method|MAE|Regret|Avg Queries|
|---|---|---|---|
|best fixed expert (DLinear)|0.370099|0.051955|nan|
|RouterDC hard no contrastive|0.359718|0.041575|1.0|
|RouterDC hard contrastive|0.354483|0.036340|1.0|
|COSTARTS first-query expert|0.365393|0.047249|1.0|
|COSTARTS top-2 predicted experts (choose lower predicted error)|0.370614|0.052471|2.0|
|Oracle best within COSTARTS top-2 predicted experts|0.339212|0.021069|2.0|
|Oracle second query after COSTARTS first query|0.318143|0.000000|2.0|
|Full oracle best expert per window|0.318143|0.000000|5.0|

Note again: the comparison CSV's `best fixed expert (DLinear)` row is not the best fixed expert according to the loaded COSTARTS validation cache; ModernTCN is.

## RouterDC Comparison

RouterDC hard selector lives in `notebooks/router2(RouterDC inspired).ipynb`. It receives only `[B,96,7]`, encodes a window, compares it to trainable expert embeddings, and selects one expert by cosine argmax. It does not mix forecasts and does not produce `[B,12,7,M]` weights.

Compared with COSTARTS:

- Both use history-only input for hard selection and trainable expert embeddings.
- RouterDC uses soft target probabilities from expert errors and optional window-window contrastive learning; COSTARTS creates soft target probabilities in cache but does not use them.
- RouterDC has one action: choose one expert. COSTARTS has a query order and stop count, but no feedback state, so its sequential structure is mostly nominal.
- RouterDC contrastive validation MAE from the saved comparison is `0.354483`; COSTARTS first-query validation MAE is `0.365393`.
- COSTARTS has more conflicting heads: map regression, ranking, query, stop, and unused mix head. RouterDC's objective is more aligned with one-step expert selection.

Likely useful RouterDC components missing from COSTARTS: soft suitability labels in the main loss, contrastive/window representation supervision, and a simpler action objective. COSTARTS also lacks a real query-feedback state updater, which would be necessary if keeping the sequential idea.

## Training Quality

Training summary: best epoch `4`, best validation MAE `0.365393`, last epoch `14`, last validation MAE `0.372097`. Early stopping recorded 14 epochs with patience 10, consistent with best epoch 4 and no later improvement.

First epoch losses: total `5.616727`, map `0.917145`, ranking `1.556839`, query `1.573014`, stop `1.569730`.

Best epoch losses: total `4.746369`, map `0.571813`, ranking `1.499072`, query `1.501960`, stop `1.173524`.

Last epoch losses: total `3.959770`, map `0.279406`, ranking `1.444879`, query `1.443618`, stop `0.791866`.

Validation overfits after epoch 4: last validation MAE is worse than best by `0.006704`. Map loss keeps decreasing after validation stops improving, suggesting the model is fitting the error-regression task without improving the deployed selection rule. I did not measure per-head gradient norms because that would require an additional backward diagnostic pass not present in the saved artifacts.

Frozen-expert proof from code: experts are loaded through `load_and_freeze_expert`, moved to device, set to eval, `requires_grad_(False)`, and gradients cleared. Cache generation runs all experts in `torch.no_grad()`. The COSTARTS optimizer is created from `router.parameters()` only, and `assert_optimizer_excludes_experts` plus `assert_no_expert_gradients` are called during training.

## Confirmed Problems

### Confirmed implementation bugs

- The saved comparison output labels DLinear as the best fixed expert on the same validation windows, but direct cache inspection shows ModernTCN is best fixed on `cache/costarts_router_val_cache.pt`. This is an artifact/reporting inconsistency, not a router-training bug.

### Confirmed objective-design problems

- STOP target degenerates to immediate stop under `stop_threshold=0.0` and zero expert costs.
- The model has no feedback state after querying an expert, so it cannot implement Context -> Action -> Feedback.
- Teacher forcing begins with the oracle-best expert, making immediate STOP optimal.
- Query loss learns the global best expert, not the next best remaining expert after partial observations.
- The deployable final selection rule depends on `map_prediction`, but validation shows the map head has near-random ranking quality.
- `mix_forecast_loss_weight=0.0`, so the mix head is not trained for the forecast metric.
- Soft target probabilities are cached but unused by COSTARTS losses.

### Likely problems requiring experiments

- Multitask interference among map/ranking/query/stop heads.
- Weak representation capacity from global pooling into one vector.
- Small router train split and close expert margins.
- Need for staged training, scheduled sampling, DAgger, subset-state supervision, or contextual-bandit/RL-style objective.
- Need to normalize expert costs relative to normalized MAE improvements.

### Working correctly

- Chronological split definitions and cache split guards are present.
- Expert checkpoints are loaded and frozen through explicit utility code.
- Cache tensor shapes match `[N,96,7]`, `[N,12,7]`, and `[N,12,7,M]`.
- No NaN/inf values were found in floating cache tensors.
- COSTARTS first query beats the incorrectly labeled DLinear fixed baseline, but not the true best fixed expert ModernTCN.
- COSTARTS top-2 shortlist has real value: oracle within top 2 reaches `0.339212` MAE.

## What Could Not Be Fully Verified

- RouterDC source is notebook JSON, not an importable Python module, so line numbers are notebook source line indices from ripgrep rather than Python file line numbers.
- Average stop logits/probabilities by every epoch are not saved. I verified best and last checkpoints only.
- Gradient norms and per-head gradient contributions were not measured because saved curves do not contain them.
- Exact original-unit MAE was not computed; caches and reported numbers here are normalized-unit MAE/MSE.

## DEEP RESEARCH INPUT PACKET

Project objective: build a sequential frozen-expert router for ETTh1 forecasting that can decide which independently trained forecasting expert(s) to query and when to stop, without updating expert checkpoints.

Dataset and shapes: ETTh1, history `[B,96,7]`, forecast `[B,12,7]`. Offline expert prediction stack `[B,12,7,M]`; current M=5.

Frozen experts: DLinear, PatchTST, iTransformer, TimesNet, ModernTCN. They are loaded from `checkpoints/candidates/best_*.pt`, moved to device, set eval, and frozen with `requires_grad=False`.

Current COSTARTS architecture: CNN history encoder with global `AdaptiveAvgPool1d(1)` -> one normalized `[B,64]` query embedding. Trainable expert embeddings `[M,64]`. Heads: map regression `[B,M]`, ranking logits `[B,M]`, query logits `[B,M]`, mix weights `[B,M]`, stop logits `[B,K]`. K=5. No state updater, no queried mask, no queried forecast feedback, no remaining budget encoding.

Current training: build offline caches from router_train/router_val only. For each window, run all frozen experts under `torch.no_grad()`, store normalized histories, targets, masks, prediction stack, MAE/MSE error matrix, soft target probabilities, argmin best expert, and sample index. Train on router_train, select checkpoint by router_val MAE only.

Teacher forcing behavior: during training, query order is oracle `argsort(error_matrix)` when teacher forcing is enabled. With current config, first teacher query is always the true oracle-best expert. Validation uses model top-k query logits, not oracle order.

Inference behavior: compute one query order from initial history. Compute stop step from separate stop head. Select the queried prefix up to stop step. Return the expert with lowest predicted error among that prefix. No forecasts are mixed in COSTARTS first-query evaluation.

Losses and weights: map normalized MSE weight 1; ranking CE to global best weight 1; query CE to global best weight 1; stop CE to target stop index weight 1; optional mix forecast MAE weight 0. Error temperature 0.1 creates cache soft labels but COSTARTS does not use them.

Cache structure: histories `[N,96,7]`, targets `[N,12,7]`, target_masks `[N,12,7]`, prediction_stack `[N,12,7,5]`, error_matrix `[N,5]`, mse_matrix `[N,5]`, target_probabilities `[N,5]`, best_expert `[N]`, sample_indices `[N]`.

Current validation results from loaded artifacts: full oracle `0.318143` MAE; best fixed actual ModernTCN `0.358645` MAE; COSTARTS first query `0.365393` MAE; RouterDC hard contrastive `0.354483` MAE from saved comparison; RouterDC no contrastive `0.359718` MAE; oracle best within COSTARTS top 2 `0.339212` MAE; COSTARTS top 2 selected by predicted error `0.370614` MAE.

Key diagnostics: stop target is 100% immediate stop; predicted stop is 100% immediate stop; query top-1 oracle match `23.49%`; top-2 oracle coverage `49.43%`; map argmin oracle match `19.09%`; map Spearman `0.0183`; map pairwise ranking accuracy `0.5088`.

Strongest evidence for failure: the top-2 shortlist contains useful experts, but the learned predicted-error head chooses poorly. Forcing oracle selection inside the top 2 improves MAE from `0.370614` to `0.339212`. Equal averaging top 2 reaches `0.353071`, which beats COSTARTS first-query and is near RouterDC.

Implementation constraints: preserve chronological splits; freeze experts; support any M; avoid test tuning; no forecast architecture changes; use router_val for model selection; use test only for final reporting.

Experiments completed: COSTARTS training saved best epoch 4; same-val comparison with RouterDC; cache validation; counterfactual forced-query analysis; predicted-vs-true error CSV; worst-regret examples CSV.

Relevant code excerpts: `COSTARTSRouter.forward` computes `query_order = teacher_forced_order[:, :K]` during teacher forcing else `torch.topk(query_logits,k=K)`. `_target_stop_index` chooses the first acceptable utility index. `costarts_losses` uses CE(query_logits, best_expert) and CE(stop_logits, stop_target). `_select_expert_from_outputs` picks the lowest map-predicted error among the stopped prefix.

## Questions For Deep Research

1. How should a sequential frozen-expert router be supervised when the true best first expert is unavailable at inference?
2. How can the router learn recovery actions after an incorrect first query?
3. Should STOP be a separate binary head or a unified action among `M+1` actions?
4. Should stopping be based on predicted marginal regret reduction?
5. How should expert query cost be normalized relative to MAE improvement?
6. Should the model predict absolute expert errors, pairwise rankings, marginal utility, or all three?
7. How can top-two shortlist quality be converted into good final expert selection?
8. Should the router use DAgger, scheduled sampling, reinforcement learning, contextual bandits, or supervised subset-state training?
9. How should queried expert predictions update the router state?
10. When should sparse mixing replace best-queried-expert selection?
11. What is the simplest defensible novel method based on the current evidence?
12. Which losses and architecture components are necessary, and which are unnecessary complexity?
13. How can the method avoid overfitting on the small router-training split?
14. What evaluations are required for a publishable paper?
15. Which existing papers are the closest methodological comparisons?
