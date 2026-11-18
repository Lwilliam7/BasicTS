# Natural Capability-Demand Matching Implementation Plan

Validation-only study. No test cache, test target, or test metric is loaded.

## Dataset Scope

- `ETTh1`: `cache/costarts_walkforward/router_train_20_60_cache.pt`, `cache/costarts_walkforward/router_val_60_80_cache.pt`
- `ETTh2`: `cache/costarts_fresh/ETTh2_96_12/router_train_cache.pt`, `cache/costarts_fresh/ETTh2_96_12/router_val_cache.pt`
- `ETTm1`: `cache/costarts_walkforward_ETTm1/router_train_20_60_cache.pt`, `cache/costarts_walkforward_ETTm1/router_val_60_80_cache.pt`
- `Weather`: `cache/costarts_walkforward_Weather/router_train_20_60_cache.pt`, `cache/costarts_walkforward_Weather/router_val_60_80_cache.pt`
- `Electricity`: `cache/costarts_walkforward_Electricity/router_train_20_60_cache.pt`, `cache/costarts_walkforward_Electricity/router_val_60_80_cache.pt`

All runs use input length `96`, forecast horizon `12`, chronological starts, and router_train/router_val only. The full cached expert order is `DLinear`, `PatchTST`, `iTransformer`, `TimesNet`, `ModernTCN`. The primary competence/routing evaluation uses the frozen train-selected three-expert core exposed by `experiments.frozen_hv_costar.run_frozen_hv_costar.LOADERS`.

## Exact Splits And OOF Protocol

Router_train is scored out-of-fold with the existing chronological pattern: keep the first 20% as warmup, then evaluate four chronological folds over the remaining 80%. For an evaluation fold starting at window index `lo`, profile/model fitting uses only windows with `absolute_window_start + horizon <= absolute_window_start[lo]`; equivalently `old_target_end <= current_origin`. The same fold-specific training prefix supplies LOW/MED/HIGH demand-bin quantiles and capability profiles. Router_val is scored once after fitting on all legal router_train windows.

## Feature Sources

- Passive baseline: `window_features_group_a`, `forecast_features_group_b`, and `disagreement_features_group_c` from `experiments.behavioral_competence.common`.
- FAME-style source: the prior BasicTS FAME adaptation in `experiments/published_baseline_comparisons/run_published_baselines.py` and `scripts/fame_etth_router.py`. This study uses a capacity-matched one-sided Ridge baseline from the same six demand axes to relative competence, rather than the neural sparse router, so the direct baseline does not get more information than the matching method.

## Demand Axes

All demand fingerprints are computed from the input history only:

- `trend`: normalized least-squares slope magnitude and low-frequency historical variation proxy.
- `seasonality`: strongest meaningful observed-lag autocorrelation over lags inside the 96-step history, excluding lag 1.
- `frequency`: normalized spectral entropy plus high/low power ratio summarized as a scalar irregular-frequency demand.
- `volatility`: normalized first-difference variance and robust local variability.
- `shift`: first-half versus second-half mean/variance and slope-change nonstationarity.
- `crossvar`: common-factor dependence, using correlation to the within-window cross-variable factor; single-variable windows get zero.

Raw continuous values are saved. LOW/MED/HIGH bins use fold-specific or final train-prefix quantiles only, never router_val.

## Capability Representations

- Regime table: per expert, per axis, LOW/MED/HIGH average relative competence, shrunk toward the expert global mean from the same train prefix.
- Continuous curve: per expert, per axis fixed quadratic Ridge curve on standardized axis values.

Primary capability-demand score is the fixed equal average of regime-table and continuous-curve predictions. Lower predicted relative competence means better routing suitability.

## Baselines And Controls

- Global expert prior.
- Passive ABC Ridge.
- FAME-style one-sided Ridge from demand axes to expert relative competence.
- Demand + expert ID Ridge with capacity matched to the simple demand representation.
- Capability-Demand Match.
- Expert fingerprint shuffle.
- Semantic-axis shuffle.

## Integrity Checks

Checks include test-path refusal, schema/expert-order/horizon/start validation, checkpoint hashes, finite outputs, deterministic repeatability, train-only bins/profiles, purged chronological OOF folds, router_val target-corruption invariance for fingerprints/profiles/scores/predictions/weights, and unchanged frozen checkpoints. ETTh2 receives a separate cache/runtime audit before being counted.

## ETTh2 Cache Status

Initial status is unresolved based on Structured Forecast Repair: ETTh2 passed no-test, chronology, finite, and target-corruption checks, but cached forecasts did not reproduce from the current runtime (`~4.4-4.9` max raw forecast differences). This run records `ETTH2_INTEGRITY_RESOLVED` only if provenance and runtime reproduction checks clear the discrepancy; otherwise ETTh2 remains diagnostic and is excluded from success counts.
