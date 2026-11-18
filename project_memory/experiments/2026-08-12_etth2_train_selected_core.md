# ETTh2 Train-Selected Core Audit

## Question

Test whether the ETTh1 current-best architecture and train-only core-selection procedure transfer to ETTh2 without using ETTh2 router-validation for tuning.

## Protocol

Two explicit phases:

1. Phase A loaded only `cache/costarts_fresh/ETTh2_96_12/router_train_cache.pt`.
2. It enumerated all 10 three-expert subsets and ranked them by pooled chronological OOF MAE on ETTh2 router-train.
3. It wrote `experiments/etth2_train_selected_core/frozen_config_before_validation.json`.
4. Phase B loaded canonical ETTh2 `router_val` once and evaluated the frozen model.

Canonical ETTh2 metric:

- validation starts `10800..11412`
- `613` windows
- horizon `12`
- variables `7`
- raw/original scale
- `std=ones`
- no inverse transform
- repository `sample_mae` / `sample_mse`

No test cache was loaded.

## Router-Train Triple Ranking

| Rank | Triple | OOF MAE | OOF MSE | Worst-fold MAE |
|---:|---|---:|---:|---:|
| 1 | `DLinear+PatchTST+ModernTCN` | `0.284658` | `0.181718` | `0.306314` |
| 2 | `DLinear+PatchTST+TimesNet` | `0.285580` | `0.182717` | `0.307959` |
| 3 | `DLinear+TimesNet+ModernTCN` | `0.286029` | `0.186941` | `0.306465` |
| 4 | `PatchTST+TimesNet+ModernTCN` | `0.290482` | `0.188716` | `0.312758` |
| 5 | `DLinear+iTransformer+ModernTCN` | `0.305731` | `0.212325` | `0.330307` |
| 6 | `DLinear+iTransformer+TimesNet` | `0.306145` | `0.212784` | `0.331630` |
| 7 | `iTransformer+TimesNet+ModernTCN` | `0.310785` | `0.218149` | `0.335959` |
| 8 | `DLinear+PatchTST+iTransformer` | `0.318007` | `0.232131` | `0.348979` |
| 9 | `PatchTST+iTransformer+ModernTCN` | `0.318291` | `0.231010` | `0.349036` |
| 10 | `PatchTST+iTransformer+TimesNet` | `0.319288` | `0.231975` | `0.351417` |

Selected core:

- `DLinear`
- `PatchTST`
- `ModernTCN`

It did not select canonical validation-best fixed-3 `DLinear+TimesNet+ModernTCN`.

## Validation Result

| Method | Validation MAE | Validation MSE |
|---|---:|---:|
| Best single: `DLinear` | `0.280957` | `0.171493` |
| Best fixed-2, reference: `DLinear+ModernTCN` | `0.275229` | `0.165345` |
| Train-selected fixed-3: `DLinear+PatchTST+ModernTCN` | `0.280878` | `0.171933` |
| Canonical best fixed-3, reference: `DLinear+TimesNet+ModernTCN` | `0.276644` | `0.166932` |
| Train-selected full current-best model | `0.276832` | `0.167280` |

The full model improved over its own train-selected fixed-3 by `0.004046` MAE.

It did not beat canonical best fixed-2:

- difference vs `0.275229`: `+0.001603` MAE worse.

## Specialist Handling

The selected ETTh2 core included both `DLinear` and `ModernTCN`. To avoid duplicate forecast use, both specialist branches were disabled as duplicates:

- DLinear specialist disabled: `true`
- ModernTCN specialist disabled: `true`

The evaluated full model was therefore the frozen hybrid chronological/HV adaptive core with duplicate specialists safely set to zero.

ETTh2 does not have a compatible static neural fixed-three artifact for arbitrary selected triples, so the chronological component used a frozen equal-weight static prior. No ETTh2 validation tuning was performed.

## Decision

The train-only selection procedure transfers partially: the frozen adaptive architecture improves over its own selected fixed-3. It does not beat the simpler canonical best fixed-2 baseline, so it should not be promoted for ETTh2.

Compared with ETTh1:

- ETTh1 train-only selection chose the prior core and preserved the current-best `0.363112`.
- ETTh2 train-only selection chose a different core and the frozen architecture did not beat the best fixed baseline.

## Reproduce

```powershell
python experiments\etth2_train_selected_core\run_etth2_train_selected_core_eval.py --phase all
```

