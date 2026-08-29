# Expert-Native Latent Competence Plan

Strict validation-only experiment. No test cache/file path is opened.

## Reused Components

- Dataset loaders and selected K=3 cores from `experiments.frozen_hv_costar.run_frozen_hv_costar.LOADERS`.
- Cached frozen forecasts from router_train and router_val only.
- Passive A+B+C features from `experiments.behavioral_competence.common`.
- Frozen HxV train-only baseline weights from existing `errors_to_weights` and `predict_from_hv_weights` utilities.
- Chronological OOF folds with horizon-12 purge.

## Target

`gain_k = MAE(frozen HxV baseline ensemble) - MAE(expert_k)`. Positive gain means expert `k` beats the baseline ensemble on that window.

## Models

Expert-specific Ridge readouts for continuous gain and LogisticRegression readouts for `gain > 0`: Passive, Hidden Only, Passive+Hidden, Shuffled Hidden, Raw Forecast Control, and Matched-Dimension Passive Control. No MLP router is trained.
