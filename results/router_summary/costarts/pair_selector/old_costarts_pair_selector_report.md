# Old ETTh1 COSTARTS Pair Selector Report

## Validation Summary

- Fixed pair `DLinear+ModernTCN` MAE `0.352726`.
- New predicted pair selector mean MAE `0.351172` +/- `0.001435`.
- Improvement over fixed pair `0.001554`.
- Equal average all experts reference `0.350065`.
- Existing predicted top-2 equal-average reference `0.350748`.
- Old COSTARTS reference `0.365393`.
- Exact pair accuracy `16.48%`.
- Top-two pair coverage `23.36%`.
- Cross-seed mean agreement `0.806`.

## Decision

The new pair selector beats old COSTARTS: `True`.
It beats existing predicted top-2 equal-average: `False`.
It beats equal average of all five experts: `False`.
Best diagnostic confidence separator: `logit_margin` AUC `0.5456074476576149`.

No forecasting experts were retrained and no test cache was created.