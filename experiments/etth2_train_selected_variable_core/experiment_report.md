# ETTh2 Train-Selected Variable-Size Core Audit

## Result

- Router-train selected subset: `DLinear`.
- Selected subset size: `1`.
- Router-train OOF MAE/MSE: `0.283464` / `0.182616`.
- Full model validation MAE/MSE: `0.280470` / `0.170973`.
- Beat previous full fixed-3: `False`.
- Beat canonical best fixed-2: `False`.
- Selected subset size stable across folds: `False`.

## Top 10 Router-Train Subsets

| Rank | Subset | Size | OOF MAE | OOF MSE | Worst Fold MAE |
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

## Required Table

| Method | Val MAE | Val MSE | Detail |
|---|---:|---:|---|
| Best single | `0.280957` | `0.171493` | `DLinear` |
| Best fixed-2 [reference] | `0.275229` | `0.165345` | `DLinear+ModernTCN` |
| Train-selected variable-size core | `0.280957` | `0.171493` | `DLinear` |
| Train-selected fixed-3 from prior experiment | `0.280878` | `0.171933` | `DLinear+PatchTST+ModernTCN` |
| Canonical best fixed-3 [reference] | `0.276644` | `0.166932` | `DLinear+TimesNet+ModernTCN` |
| Full model on train-selected variable core | `0.280470` | `0.170973` | `both_variable_decay0.95_cap0.1_marginbp200_warm96` |
| Previous full fixed-3 model | `0.276832` | `0.167280` | `DLinear+PatchTST+ModernTCN` |

## Reproduce

```powershell
python experiments\etth2_train_selected_variable_core\run_etth2_train_selected_variable_core_eval.py --phase all
```
