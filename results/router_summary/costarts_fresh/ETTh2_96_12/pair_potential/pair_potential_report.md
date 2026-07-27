# ETTh2 Pair Potential Report

## Decision Answers
1. Router-training-selected best fixed expert: `DLinear`.
2. Router-training-selected best fixed pair: `DLinear+ModernTCN`.
3. Validation MAE for that pair: `0.275229`.
4. Pair beats both validation constituents: `True`.
5. Fixed-pair to oracle-pair validation improvement: `0.013259` MAE.
6. Useful switch rate at 0.01 margin: `42.74%`.
7. Validation pair mean margin: `0.009227`, median `0.006647`.

## Top Validation Methods

| method | MAE | MSE | avg experts | source |
|---|---:|---:|---:|---|
| per_window_oracle_pair | 0.261970 | 0.152997 | 2.00 | oracle diagnostic |
| per_window_oracle_expert | 0.263661 | 0.154336 | 1.00 | oracle diagnostic |
| nonnegative_simplex_linear_average | 0.274755 | 0.165479 | 5.00 | router-train fitted |
| DLinear+ModernTCN | 0.275229 | 0.165345 | 2.00 | router-train selected |
| DLinear+TimesNet+ModernTCN | 0.276644 | 0.166932 | 3.00 | fixed |
| ridge_linear_stacker | 0.276702 | 0.165339 | 5.00 | router-train fitted |
| DLinear+TimesNet | 0.277652 | 0.167802 | 2.00 | fixed |
| DLinear+PatchTST+TimesNet+ModernTCN | 0.277681 | 0.168231 | 4.00 | fixed |
| DLinear+PatchTST+ModernTCN | 0.280878 | 0.171933 | 3.00 | fixed |
| DLinear | 0.280957 | 0.171493 | 1.00 | router-train selected |

## Leakage

Only the two clean router caches and cache reports were loaded. No ETTh2 test arrays were read and no test cache was created.