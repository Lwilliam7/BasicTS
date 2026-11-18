# Train-Selected Core ETTh1 Re-Evaluation

## Result

- Router-train selected experts: `PatchTST+iTransformer+TimesNet`.
- Router-train OOF MAE/MSE: `0.345568` / `0.260786`.
- Train-selected fixed-3 validation MAE: `0.367265`.
- Train-selected current-best architecture validation MAE: `0.363100`.
- Difference vs previous `0.363112`: `-0.000012`.

## Required Table

| Method | Val MAE | Val MSE | Detail |
|---|---:|---:|---|
| Best single | `0.376550` | `0.322095` | `iTransformer` |
| Train-selected fixed 3 | `0.367265` | `0.310530` | `PatchTST+iTransformer+TimesNet` |
| Old validation-selected fixed 3 [reference] | `0.367265` | `0.310530` | `PatchTST+iTransformer+TimesNet` |
| Train-selected current-best model | `0.363100` | `0.306026` | `both_variable_decay0.95_cap0.1_marginbp200_warm96` |
| Previous current-best model [reference] | `0.363112` | `0.306057` | `validation-optimized development result` |

## Reproduce

```powershell
python experiments\train_selected_core_etth1\run_train_selected_core_eval.py --device cuda --out-dir experiments\train_selected_core_etth1_equal_static --phase all
```
