# ETTh2 Pair Potential Report

## Decision Answers
1. Router-training-selected best fixed expert: `DLinear`.
2. Router-training-selected best fixed pair: `DLinear+PatchTST`.
3. Validation MAE for that pair: `1.000000`.
4. Pair beats both validation constituents: `False`.
5. Fixed-pair to oracle-pair validation improvement: `0.100000` MAE.
6. Useful switch rate at 0.01 margin: `5.00%`.
7. Validation pair mean margin: `0.100000`, median `0.100000`.

## Top Validation Methods

| method | MAE | MSE | avg experts | source |
|---|---:|---:|---:|---|

## Leakage

Only the two clean router caches and cache reports were loaded. No ETTh2 test arrays were read and no test cache was created.