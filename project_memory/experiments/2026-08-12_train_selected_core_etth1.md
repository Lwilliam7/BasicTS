# Train-Selected Core ETTh1 Re-Evaluation

## Question

The current best ETTh1 development model used the fixed-three core `PatchTST+iTransformer+TimesNet`, but that core had originally been identified after validation inspection. This audit asks how much of the `0.363112` result survives when the core three experts are selected using router-train only.

## Protocol

Two explicit phases:

1. Phase A loaded only `cache/costarts_walkforward/router_train_20_60_cache.pt`.
2. It enumerated all 10 three-expert subsets and selected by pooled chronological OOF MAE over four router-train folds.
3. Tie breakers were OOF MSE, worst-fold MAE, then deterministic expert-name ordering.
4. It wrote `experiments/train_selected_core_etth1/frozen_config_before_validation.json`.
5. Phase B then loaded validation once and evaluated the frozen configuration.

Test cache was not loaded.

## Router-Train Triple Ranking

| Rank | Triple | OOF MAE | OOF MSE | Worst-fold MAE |
|---:|---|---:|---:|---:|
| 1 | `PatchTST+iTransformer+TimesNet` | `0.345568` | `0.260786` | `0.402570` |
| 2 | `DLinear+PatchTST+iTransformer` | `0.346441` | `0.261499` | `0.411108` |
| 3 | `DLinear+PatchTST+TimesNet` | `0.347316` | `0.261530` | `0.403718` |
| 4 | `DLinear+iTransformer+TimesNet` | `0.348282` | `0.262334` | `0.402645` |
| 5 | `PatchTST+iTransformer+ModernTCN` | `0.372980` | `0.286032` | `0.411819` |
| 6 | `PatchTST+TimesNet+ModernTCN` | `0.379215` | `0.294662` | `0.409342` |
| 7 | `DLinear+PatchTST+ModernTCN` | `0.380096` | `0.292044` | `0.412031` |
| 8 | `iTransformer+TimesNet+ModernTCN` | `0.381534` | `0.300646` | `0.411172` |
| 9 | `DLinear+iTransformer+ModernTCN` | `0.382245` | `0.297408` | `0.413139` |
| 10 | `DLinear+TimesNet+ModernTCN` | `0.388690` | `0.306455` | `0.410888` |

Router-train selected the same core as the old development model:

- `PatchTST`
- `iTransformer`
- `TimesNet`

## Validation

| Method | Validation MAE | Validation MSE |
|---|---:|---:|
| Best single: `iTransformer` | `0.376550` | `0.322095` |
| Train-selected fixed 3 | `0.367265` | `0.310530` |
| Old validation-selected fixed 3, reference | `0.367265` | `0.310530` |
| Train-selected current-best architecture | `0.363112` | `0.306057` |
| Previous current-best model, reference | `0.363112` | `0.306057` |

The train-selected current-best architecture improves over train-selected fixed-3 by `0.004153` MAE. Paired bootstrap CI versus selected fixed-3: `[-0.004460, -0.003856]`.

Differences:

- versus selected fixed-3: `-0.004153`
- versus old HV baseline `0.363642`: `-0.000529`
- versus previous `0.363112`: `0.000000`

## Decision

The suspected fixed-three validation-selection issue does not change the current best result, because router-train-only selection independently chooses `PatchTST+iTransformer+TimesNet`. The `0.363112` validation result survives this audit.

## Reproduce

```powershell
python experiments\train_selected_core_etth1\run_train_selected_core_eval.py --phase all --device cuda
```

