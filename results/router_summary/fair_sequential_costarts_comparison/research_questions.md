# Fair Sequential COSTAR-TS Comparison

Evaluation split: router_val only. Router methods train on router_train. No test cache was used.

## Research Questions

1. Strongest individual model: ModernTCN MAE 0.358645. Sequential COSTAR-TS MAE 0.347949; answer: yes.
2. Best simple ensemble: Validation-weighted average of all experts MAE 0.349751. COSTAR beats it: yes.
3. Best FAME-style router: FAME-style ETTh adaptation Top-3 MAE 0.350860. COSTAR beats it: yes.
4. COSTAR vs best FAME MAE gap is 0.002911; pooled seed std is 0.001976. Improvement larger than seed variation: yes.
5. COSTAR closes 26.41% of the strongest-single-to-oracle-best-single MAE gap.
6. COSTAR executes 4.046 experts on average versus 5.0 for all-expert averaging, with lower MAE: yes.
7. Sequential observation appears helpful on this split because COSTAR beats the best history-only FAME-style Top-K row.
8. No apparent gain here comes from extra data access: all rows use identical router_val windows and frozen expert predictions; train-derived baselines use router_train only.

TimeRouter is not included because no same-cache reproduction was found. RouterDC is intentionally excluded from the main table because it is not originally a time-series forecasting router.
