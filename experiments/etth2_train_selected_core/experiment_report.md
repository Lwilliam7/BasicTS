# ETTh2 Train-Selected Core Audit

## Result

- Router-train selected experts: `DLinear+PatchTST+ModernTCN`.
- Router-train OOF MAE/MSE: `0.284658` / `0.181718`.
- Full model validation MAE/MSE: `0.276832` / `0.167280`.
- Beat own selected fixed-3: `True`.
- Beat canonical best fixed-2: `False`.

## Required Table

| Method | Val MAE | Val MSE | Detail |
|---|---:|---:|---|
| Best single | `0.280957` | `0.171493` | `DLinear` |
| Best fixed-2 [reference] | `0.275229` | `0.165345` | `DLinear+ModernTCN` |
| Train-selected fixed-3 | `0.280878` | `0.171933` | `DLinear+PatchTST+ModernTCN` |
| Canonical best fixed-3 [reference] | `0.276644` | `0.166932` | `DLinear+TimesNet+ModernTCN` |
| Train-selected full current-best model | `0.276832` | `0.167280` | `both_variable_decay0.95_cap0.1_marginbp200_warm96` |

## Reproduce

```powershell
python experiments\etth2_train_selected_core\run_etth2_train_selected_core_eval.py --phase all
```
