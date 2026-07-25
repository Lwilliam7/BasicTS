# Reproducibility Notes

- Dataset: ETTh1.
- Input shape: `[B,96,7]`.
- Forecast shape: `[B,12,7]`.
- All reported package diagnostics use chronological `router_val` windows unless explicitly marked as an oracle upper bound.
- The final test split is not used in this package.
- Frozen expert predictions are read from cached prediction stacks; no expert checkpoint is updated.
- Old COSTARTS and improved subset-utility COSTARTS are labeled separately in tables and plots.
- Current ablation numbers may be smoke-test numbers if the ablation runner was executed with `--max-epochs 1`.
