# Structured Forecast Repair implementation plan

## Scope and data

- Validation-only study on `ETTh1`, `ETTh2`, `ETTm1`, `Weather`, and `Electricity`.
- Load only router-train and router-val cached histories, targets, target masks,
  absolute starts, and frozen-expert prediction stacks.
- Use the existing dataset-specific frozen expert set and checkpoint scaler;
  never load a test cache or retrain/modify an expert.

## Causal protocol

- Router-train features use chronological four-fold OOF construction. Each fold
  fits all repair calibrations and the REP reference bank on earlier windows,
  purges the preceding horizon overlap, then scores the later fold.
- Router-val repair calibration and REP bank are fit once on all router-train
  windows and frozen before validation targets are read for scoring.
- Competence is `z[t,k] = expert_MAE[t,k] - mean_k(expert_MAE[t,k])`.

## Repair mechanism

- Temporal: train-window standardized first-difference trajectory subspace and
  robust residual limits, with a second-difference diagnostic when available.
- Seasonal: use lag 24 when history supports it and train-derived lag-to-future
  structure has nontrivial variance; otherwise mark inactive.
- Cross-variable: train-derived PCA subspace on standardized future trajectories.
- Multi-horizon: train-derived PCA subspace on normalized horizon trajectories.
- For every family, compute raw violation and deterministic projected minimum
  repair; use the same fixed-step projection algorithm for every expert.
- Store raw and normalized repair costs, per-family costs, horizon and variable
  geometry, plus a compact scalar-only `RepairCost` arm.

## Controls and evaluation

- Ridge with train-only standardization and the same protocol for every arm:
  Passive, Passive+Disagreement, Passive+REP, Passive+RawViolation,
  Passive+RepairCost, Passive+RepairGeometry,
  Passive+Disagreement+REP+RepairGeometry.
- Compare relative-competence prediction, raw-error correlations, correct versus
  within-window expert-shuffled geometry, and a fixed softmax routing proxy.
- Report IID and existing dependence-aware block bootstrap, including block 24.

## Integrity outputs

- Verify checkpoint hashes and parameter fingerprints, cache forecast
  reproduction, chronological/purge manifests, no test paths, target-corruption
  invariance for all target-free features and predictions, finite values, and
  deterministic repeated repair features.
- Save all requested CSV/JSON/markdown artifacts and cached per-window features.