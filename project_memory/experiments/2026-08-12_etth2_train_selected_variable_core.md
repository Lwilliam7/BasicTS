# ETTh2 Train-Selected Variable-Size Core Audit

## Question

Test whether ETTh2 router-train can choose both the core subset size and membership for the current frozen adaptive architecture without using ETTh2 router-validation for selection.

## Protocol

Two clean phases:

1. Phase A loaded only `cache/costarts_fresh/ETTh2_96_12/router_train_cache.pt`.
2. It enumerated all 31 non-empty subsets of `DLinear`, `PatchTST`, `iTransformer`, `TimesNet`, and `ModernTCN`.
3. It ranked each equal-weight subset by pooled chronological OOF MAE on ETTh2 router-train, with tie breakers: OOF MSE, worst-fold MAE, smaller subset, deterministic alphabetical ordering.
4. It wrote `experiments/etth2_train_selected_variable_core/frozen_config_before_validation.json`.
5. Phase B loaded canonical ETTh2 `router_val` once and evaluated the frozen result.

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

## Selection Result

Router-train selected the single-expert core:

- `DLinear`

Router-train OOF:

- MAE `0.283464`
- MSE `0.182616`
- worst-fold MAE `0.302688`

The top router-train subsets were:

| Rank | Subset | Size | OOF MAE | OOF MSE | Worst-fold MAE |
|---:|---|---:|---:|---:|---:|
| 1 | `DLinear` | 1 | `0.283464` | `0.182616` | `0.302688` |
| 2 | `DLinear+ModernTCN` | 2 | `0.283635` | `0.183577` | `0.303769` |
| 3 | `DLinear+PatchTST+ModernTCN` | 3 | `0.284658` | `0.181718` | `0.306314` |
| 4 | `DLinear+PatchTST+TimesNet+ModernTCN` | 4 | `0.284742` | `0.182757` | `0.306124` |
| 5 | `DLinear+TimesNet` | 2 | `0.285530` | `0.185327` | `0.306192` |
| 6 | `DLinear+PatchTST+TimesNet` | 3 | `0.285580` | `0.182717` | `0.307959` |
| 7 | `DLinear+TimesNet+ModernTCN` | 3 | `0.286029` | `0.186941` | `0.306465` |
| 8 | `DLinear+PatchTST` | 2 | `0.289715` | `0.186317` | `0.312956` |
| 9 | `PatchTST+TimesNet+ModernTCN` | 3 | `0.290482` | `0.188716` | `0.312758` |
| 10 | `PatchTST+ModernTCN` | 2 | `0.291861` | `0.188537` | `0.315582` |

Fold-best subset sizes were not stable:

- fold 0: size 3, `DLinear+TimesNet+ModernTCN`
- fold 1: size 1, `DLinear`
- fold 2: size 1, `DLinear`
- fold 3: size 3, `DLinear+PatchTST+TimesNet`

## Validation Result

| Method | Validation MAE | Validation MSE |
|---|---:|---:|
| Best single: `DLinear` | `0.280957` | `0.171493` |
| Best fixed-2, reference: `DLinear+ModernTCN` | `0.275229` | `0.165345` |
| Train-selected variable-size core: `DLinear` | `0.280957` | `0.171493` |
| Train-selected fixed-3 from prior experiment: `DLinear+PatchTST+ModernTCN` | `0.280878` | `0.171933` |
| Canonical best fixed-3, reference: `DLinear+TimesNet+ModernTCN` | `0.276644` | `0.166932` |
| Full model on train-selected variable core | `0.280470` | `0.170973` |
| Previous full fixed-3 model | `0.276832` | `0.167280` |

The full adaptive model on the selected single-expert core improved over raw `DLinear` by `0.000487` MAE, but it was worse than:

- previous full fixed-3 model by `0.003637` MAE
- canonical best fixed-2 by `0.005241` MAE

## Specialist Handling

The selected core already included `DLinear`, so the DLinear specialist branch was disabled to avoid duplicate forecast use.

`ModernTCN` was not in the selected core, so the existing frozen ModernTCN specialist mechanism remained eligible. It used average validation specialist weight `0.011663`.

No ETTh2 validation tuning was performed.

## Decision

Do not promote variable-size router-train core selection for ETTh2 as implemented.

Router-train independently chose a smaller core, but the selected core did not beat the forced fixed-3 validation result and the full adaptive model was worse than the previous full fixed-3 and canonical fixed-2 baselines.

## Evidence

- `experiments/etth2_train_selected_variable_core/final_report.json`
- `experiments/etth2_train_selected_variable_core/router_train_all_subsets.csv`
- `experiments/etth2_train_selected_variable_core/frozen_config_before_validation.json`

## Reproduce

```powershell
python experiments\etth2_train_selected_variable_core\run_etth2_train_selected_variable_core_eval.py --phase all
```
