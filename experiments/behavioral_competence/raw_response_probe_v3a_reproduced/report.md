# LearnedProbe V3A Reproduced -- Raw-Response Representation Test

This analysis uses accepted frozen-protocol V2 reproduction artifacts, not exact original V2 tensors.

**Classification: SIX_STATS_NOT_THE_BOTTLENECK**

## Primary Results

| Dataset | SixStat MAE | Raw MAE | Shuffled Raw MAE | Passive MAE | Passive+Raw MAE | Residual R2 |
|---|---:|---:|---:|---:|---:|---:|
| ExchangeRate | 0.032797 | 0.034368 | 0.033657 | 0.028014 | 0.029058 | -0.1766 |
| Traffic | 0.062098 | 0.143470 | 0.186858 | 0.038388 | 0.143986 | -13.3361 |
| BeijingAirQuality | 0.245695 | 0.243078 | 0.243072 | 0.212210 | 0.215003 | -0.0440 |
| ETTm2 | 0.057150 | 0.061785 | 0.063162 | 0.051905 | 0.051536 | -0.3305 |

## Compliance

```text
RIDGE_ALPHA_TUNED: NO (fixed at 1.0)
V2 REPRODUCTION ARTIFACTS ACCEPTED BEFORE V3A: YES
ROUTER TRAINED: NO
TEST SET ACCESSED: NO
```
