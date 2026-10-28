# Standalone conditional-combination comparison

`compare_conditional_combinations.py` imports no BasicTS modules, retrains no
forecasting experts, reads only explicitly supplied inputs, and writes to a new
output directory. It does not modify existing experiments. Python dependencies:
`numpy`, `torch`, `xgboost>=2.1`, `tsfeatures==0.4.5`, `threadpoolctl`.

## Server: upload one Python file and run it

Upload only `scripts/compare_conditional_combinations.py` to the existing
server repository's `scripts/` folder. No local result files or new data archives
are needed. This follows `server.md`: repository `/mnt/nvme0/SERVER_USER/BasicTS`,
Python `/mnt/nvme0/SERVER_USER/envs/basicts-ec/bin/python`, thread caps and `nohup`.
In the commands below `$USER` means the server account that owns those paths.

On the server:

```bash
cd /mnt/nvme0/$USER/BasicTS
P=/mnt/nvme0/$USER/envs/basicts-ec/bin/python
R=scripts/compare_conditional_combinations.py
mkdir -p logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

# Read-only inventory: packages, paths and missing primary results.
"$P" "$R" --server plan --threads 8 --workers 4 --jobs 2

# Software positive controls; no research data needed.
"$P" "$R" --self-test --threads 2

# Start once. Safe to rerun after a stopped run; finished fits are reused.
nohup "$P" "$R" --server run --threads 8 --workers 4 --jobs 2 \
  > logs/conditional_combinations.log 2>&1 < /dev/null &
echo "queue PID: $!"
```

If the environment is missing XGBoost/tsfeatures/threadpoolctl, install those
dependencies before the plan command:

```bash
"$P" -m pip install 'xgboost>=2.1' 'tsfeatures==0.4.5' threadpoolctl
```

The default queue runs **all seven datasets at H=96 and H=192**, with at most two
CPU unit processes. GPUs are not required. It discovers the existing pools:

| Units | Pool directory | Primary/comparator directory |
| --- | --- | --- |
| H=96, six datasets excluding Traffic | `cache/costarts_h96/<DS>` | `experiments/horizon_pilot_h96/_work` |
| Traffic H=96; all H=192/336/720 | `cache/costarts_lh/h<H>/<DS>` | `experiments/long_horizon_protocol/units/<DS>_h<H>/_work` |

Each pool's normalizer is read from the corresponding
`checkpoints/<pool>/final_60/DLinear/best_expert.pt`. The script imports all five
wide-top3 seeds and the classical comparator errors from the server's existing
`.pt` files. **No manual primary NPZ export is necessary in server mode.**

Before fitting a unit, it checks dataset/horizon/expert metadata, integrity flags,
the primary's fold origins and purge boundaries, training core-selection scores,
primary tensor-versus-JSON means, and **per-window** equal-five/equal-core MAE and
MSE against freshly recomputed values. Differences fail the unit. Historical
primary files did not record input hashes; these numerical and metadata checks
establish association, not historical byte identity. That limitation is recorded,
and current input fingerprints are saved for reproducibility and resume checks.

Daily periods are predeclared as 24 for hourly series and 144 for SolarAL,
AppliancesEnergy and TetouanPower (10-minute observations). The 96-point histories
cannot support every seasonal feature at period 144; the feature diagnostics
record undefined features. No extra history is supplied to a comparator.

To include the full long-horizon grid, add this **to both plan and run**, and use
a separate output directory if a default queue has already started:

```bash
--horizons 96 192 336 720 --output experiments/conditional_comparison_all_horizons
```

For a small first unit, use `--datasets ETTh1 --horizons 96 --jobs 1 --output
experiments/conditional_comparison_smoke`. Keep the same selection of datasets,
horizons, settings, packages and source when resuming that output directory.

Traffic H=720 retains the original hardware exclusion. Missing seed files (such
as the recorded Traffic H=336 seed-17 failure) remain **BLOCKED**, rather than
being dropped or averaged over fewer seeds. Other ready units still run. Restore
the original missing job's results and rerun this queue to complete it; this
script does not retrain experts or repair the old scorer job.

Monitor and collect results:

```bash
"$P" "$R" --server status
tail -n 30 logs/conditional_combinations.log
tail -n 30 experiments/conditional_comparison_server/logs/ETTh1_h96.log

# After the queue finishes (including a partial run):
"$P" "$R" --server pack
```

With a custom output directory, pass the same `--output` to status and pack.
Results are in `experiments/conditional_comparison_server/report.md`,
`summary.csv`, `summary.json`, and `units/<DS>_h<H>/`. Pack creates
`experiments/conditional_comparison_server.tar.gz`, including per-window results,
logs, protocols and source snapshots, but excluding the reusable feature caches.
Copy this archive back with `scp` using the connection in `server.local.md`.

A queue lock rejects duplicate launches. An identical restart revalidates input
hashes and reuses complete configurations and seed fits. Do not replace the
Python file or change the environment while the queue is running. Failures remain
in the unit logs and `failure.json`; the queue reports INCOMPLETE and exits 2 if
any requested non-excluded unit is unavailable or failed. No server run has been
launched as part of preparing this file.

## Methods and scope

- **FFORMA-MAE:** the official softmax expected-loss objective, with normalized
  MAE instead of M4 OWA, and the existing project's adaptation of THA features
  to the mean across variables in each history window. Negative forecasts are
  allowed. This is an adaptation, not a reproduction of the M4 competition.
- **Conditional error softmax:** XGBoost predicts each expert's conditional
  normalized MAE; softmax converts those estimates into convex weights.
- **Conditional MLP:** a neural network learns convex combination weights by
  minimizing normalized forecast MAE directly.
- **Static controls:** equal weighting and the single expert selected on
  router_train. All methods have the identical expert pool.

All three conditional families receive identical THA inputs and use window-level
weights. Their comparison with cell-level CellCatch evaluates the complete
procedures; it does not isolate granularity. The generic controls are not
replications of FAME, TimeRouter, OneNet, or other named methods.

Each family searches five fixed configurations using four expanding chronological
folds, starting after 20% of router_train. Training labels must fully resolve
before the held-out origin. The selected configuration minimizes pooled OOF
normalized MAE using seed 7. Final fits use seeds 7, 11, 13, 17, 19. Selection
budgets match across these three families; runtime does not. The frozen primary's
historical architecture and dataset selection are not retroactively budget-matched.

Metrics follow the project's per-window masked, checkpoint-standardized MAE and
MSE. Seed losses, rather than predictions, are averaged. Paired non-wrapping
moving-block intervals use `ceil(2 * horizon / stride)` windows and are withheld
below eight blocks. Intervals are descriptive, without multiplicity correction.
Evaluation windows must have a constant stride. Validation targets never fit or
select a model. This is development evidence on reused validation periods.

## Commands

From the repository root (PowerShell):

```powershell
python scripts/compare_conditional_combinations.py --self-test

python scripts/compare_conditional_combinations.py --plan `
  --train-cache cache/costarts_walkforward/router_train_20_60_cache.pt `
  --val-cache cache/costarts_walkforward/router_val_60_80_cache.pt `
  --normalizer checkpoints/costarts_walkforward/final_60/DLinear/best_expert.pt `
  --frequency 24

python scripts/compare_conditional_combinations.py `
  --train-cache cache/costarts_walkforward/router_train_20_60_cache.pt `
  --val-cache cache/costarts_walkforward/router_val_60_80_cache.pt `
  --normalizer checkpoints/costarts_walkforward/final_60/DLinear/best_expert.pt `
  --frequency 24 --output experiments/conditional_comparison_etth1_h12
```

For another pool or horizon supply its own two frozen caches and matching expert
normalizer, and declare its seasonal frequency before scoring. Do not substitute
a different pool just because the desired cache is unavailable. Cache roles must
be `router_train_20_60` and `router_val_60_80`; training provenance must identify
both `block_b_oos` and `block_c_oos`. No test roles are accepted.

Only load trusted local PyTorch caches/checkpoints: they use Python pickle.

## Add the frozen primary

Supply `--primary primary_wide_top3.npz` to complete the paired comparison.
Export this NPZ from the original primary run where its forecast caches and
per-window losses are available, with these fields:

| Field | Meaning |
| --- | --- |
| `losses` | Float array `[5, N, 2]`; normalized MAE then MSE per window |
| `seeds` | Integer array `[7, 11, 13, 17, 19]`, in loss-array order |
| `origins` | Exact validation `absolute_window_starts`, in loss-array order |
| `expert_names` | Exact cache expert names, in order |
| `train_sha256` | Scalar string: SHA256 of the source router_train cache |
| `val_sha256` | Scalar string: SHA256 of the source router_val cache |
| `normalizer_sha256` | Scalar string: SHA256 of the matching scaler checkpoint |

Do not attach new cache fingerprints to unidentified legacy loss vectors. Export
the identity fields from the source run; a vector's length alone does not prove
alignment. The script refuses mismatching fingerprints, expert order, origins,
or seed lists. Without the primary it explicitly reports
`BASELINES_ONLY_PRIMARY_MISSING` and makes no CellCatch comparison claim.

## Outputs and checks

`protocol.json` records input fingerprints, grids, seeds, folds, package versions,
and source fingerprint before fitting. `selection.json` contains every OOF
candidate score. `per_window_losses.npz`, `results.json`, and `report.md` contain
the evaluation. Feature diagnostics record failed groups and nonfinite values
replaced with zero; inspect them before interpreting a weak baseline.

An existing output directory is refused unless `--resume` is used. Resume requires
identical input hashes, settings, source and package versions. Server mode resumes
automatically, with a queue lock and per-unit locks. The script rechecks input
hashes before writing final results.

The planted-regime self-test checks that all three families learn a predictable
winner, convex weights, causal purging, the official objective calculation,
masked normalization, and the insufficient-block guard. It is a software positive
control, not a reproduction of a published forecasting result.

FFORMA objective reference:
https://github.com/robjhyndman/M4metalearning/blob/61ddc7101680e9df7219c359587d0b509d2b50d6/R/ensemble_classifier.R
