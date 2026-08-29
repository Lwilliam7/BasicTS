# Pair Residual Fault Gate Plan

Strict validation-only experiment. No test cache/file may be opened.

## Baseline

Use the existing train-only frozen HxV COSTAR weighting primitive over each dataset's already-selected K=3 core from `experiments.frozen_hv_costar.run_frozen_hv_costar.LOADERS`. Router_train OOF folds build baseline weights from earlier legal targets only; router_val uses one final baseline weight tensor fit from all router_train targets.

## Fault Target

`regret_k = MAE_k - median(MAE_other_experts)`. A fault is `regret_k >= q` where `q` is the 80th or 90th percentile of positive router_train regret, chosen by chronological OOF router_train routing MAE only.

## Features

- Passive: existing A+B+C window/forecast/disagreement features.
- Parity: signed pair residual summaries for every expert pair plus target-expert-oriented pairs: signed mean, early/late signed mean, absolute magnitude, max magnitude, horizon profile, and variable profile when manageable.
- Shuffled parity: Passive+Parity architecture with parity windows independently shuffled inside train/eval splits.
- Raw forecast control: Passive plus compressed raw expert forecast summaries instead of explicit pair residuals.

## Gate

`w_new[k] = w_base[k] * (1 - p_fault[k]) ** gamma`, renormalized over experts. `gamma` in `[0.5, 1.0, 2.0]` and intervention threshold in `[none, 0.4, 0.6, 0.8]` are selected from router_train OOF only.
