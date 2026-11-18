# Plan: gate ablation (prompt.md, Priority 1 / review action 2)

Preparation status: step 1 only. No server anchor, canary, experiment, commit or
push has been performed as part of implementation. This file preserves the supplied
plan below. The code and this predeclaration must be committed with the user's
go-ahead before a server job runs.

Implementation details: the result archive is written to
`experiments/gate_ablation/results_bundle_gate_ablation.tar` to respect output
isolation; queue logs use the declared `logs/gate_ablation/` exception. Static
zero-denominator gates (including the scalar and secondary static-all gate) are
zero. Population variances are used for the exact variance decomposition;
correlation is recorded as null when either vector has zero variance. Failed
integrity records are preserved as INVALID; runtime failures are retried once.
Source, reference and input identities are checked before reusing completed jobs.
The self-test additionally checks masks, clipping, chunked reduction, verdicts,
and scoring plumbing with a mock head (no learned-model training).

## Context

The frozen method is `w = (1-λ)·f_core3 + λ·topk_equal(affinity_5, 3)`, where λ comes from `LambdaHead`, an input-dependent MLP. The research review asks what that gate adds beyond simpler combinations: "the decisive test of whether input-dependent gating earns its complexity". Until this is answered, the paper must describe uncertainty-aware or input-dependent gating as a hypothesis.

This plan compares the full gate with four simpler gates on the same scorer, the same routing and the same units, and keeps all of them in one job:
- core only (λ = 0)
- routing only, i.e. no gate (λ = 1)
- a train-selected constant
- a static horizon-by-variable (h×v) gate

A fully static arm is added as a secondary, non-gating arm.

Settings, as chosen by the user:
- Horizons: 12 and 192 steps.
- Datasets (7): ETTh1, Electricity, Traffic, AppliancesEnergy, OikolabWeather, SolarAL, TetouanPower. All pools are early-stopped.
- Seeds: 7, 11, 13, 17, 19.
- Total: 14 units × 5 seeds = 70 jobs.

Evidence status: this is development evidence. Every unit was already scored for the frozen method on the same 60–80% block. Nothing about the frozen method changes, whatever the outcome. No test split is used.

## Arms

All arms in a job share one 5-candidate scorer: `wide.train_scorer_bank_protocol`, with 4 causal OOF folds and a final fit. They also share the same OOF and router_val affinities from `wide.head_inputs`.

Notation:
- `c` is the equal core-3 forecast.
- `r` is the forecast from the equal top-3 of the affinities.
- Every gated arm predicts `c + λ·(r − c)`.

Every gated arm is built with `topk.weights_fn(fallback, 3)(val_aff, λ)`, the full arm's own weight code. Only the λ tensor changes between arms.

| Arm | Router part | λ | What it answers |
|---|---|---|---|
| `core_only` | none | 0 | No routing at all (equals the `equal3` comparator) |
| `router_only` | input-dependent top-3 | 1 | Routing without interpolation ("no gate") |
| `constant` | input-dependent top-3 | one scalar, closed form on OOF | Train-selected constant gate |
| `static_hv` | input-dependent top-3 | λ[h,v], closed form on OOF | Static horizon-variable gate |
| `full` | input-dependent top-3 | frozen `LambdaHead`, via `topk.head_arm` | The frozen method, recomputed |
| `static_all` (secondary) | static per-cell top-3 | λ[h,v], closed form | Nothing input-dependent at all |

### Static estimators

These are the exact minimisers of the head's own training loss within each static family. The loss is `Σ m·((ŷ−y)/s_v)² / Σ m` over the pooled OOF rows. Define `e = c − y` and `d = r − c` on the OOF rows.
- `constant`: λ_c = clip(−Σ m·e·d/s_v² ÷ Σ m·d²/s_v², 0, 1).
- `static_hv`: λ[h,v] = clip(−Σ_n m·e·d ÷ Σ_n m·d², 0, 1).
  - A cell with Σ m·d² = 0 gets λ = 0. The number of such cells is recorded.
- Sums are float64, accumulated in window chunks of `wdec.STAT_CHUNK`.
- The static arms are fitted on all pooled OOF rows. The head fits only on its chronological train part.
  - So the static arms get their best case, and a win for the full gate is a conservative finding.
- `static_all`, per cell:
  1. Take the 3 candidates with the lowest Σ_n m·(f_e − y)² on the same OOF rows. Use a stable sort, so ties go to candidate order.
  2. Weight them equally.
  3. Refit λ[h,v] with the same closed form, with `r` replaced by this static forecast.
  - It uses no scorer output, so it must come out identical across seeds. That is checked.

## Units

| Horizon | Datasets | Pool | How it is opened | Reference for per-job reproduction |
|---|---|---|---|---|
| 12 | ETTh1, Electricity, Traffic | R4 (`cache/costarts_poolv2/R4/poolv2_<DS>`) | `run_pool_evaluation.pool_bundle("R4", ds)`, then `verify_router_train_out_of_sample` and `select_core_on_router_train`; the core must equal the core in `_eval/R4_<DS>.json` | None. R4 was scored with the older narrow method, so this is the first wide top-3 run on these pools; it relies on the global anchor |
| 12 | AppliancesEnergy, OikolabWeather, SolarAL, TetouanPower | `costarts_untouched_long/<DS>` | `run_long_training.confirmation()`, then `conf.load_pool` | `wide_top3_confirmation_long_training/_work/seed<s>_<DS>.pt`, falling back to the scalar MAE in the `.json` |
| 192 | all 7 | `costarts_lh` | `lh.configure(ds, 192, scoring=True)`, then the checked `conf.load_pool` | `long_horizon_protocol/units/<DS>_h192/_work/seed<s>_<DS>.pt` |

Settings per horizon:
- 12 steps: block 24 and phase 12, batch 32, chunk 256. These are the frozen settings, asserted equal to `lh.BLOCK_12`, `PHASE_12`, `BATCH_12` and `STAT_CHUNK_12`.
- 192 steps: stride, batch, chunk, block, phase and eligibility come from `pool_plan.json`.
  - ETTh1_h192 is not interval-eligible (6.75 blocks).
  - At 12 steps every unit is eligible: windows/24 ≥ `lh.MIN_BLOCKS`.

Isolation:
- `lh.configure` and `lt.confirmation` patch module globals, so each process scores exactly one unit.
- The ablation never calls `conf.run_one` or `conf.analyze`, and writes only under `experiments/gate_ablation/`.
- The reference files are hashed before and after each job.

## Readings (predeclared)

For each horizon (12, 192), each metric (MAE, MSE) and each contrast A − B, the procedure is:
1. Seed-average the per-window errors.
2. Run `cf.paired` with the unit's block and phase.
3. Build each unit's contrast with `lh.contrast(block, level = B's mean, eligible, per-seed deltas)`.
4. Over the 7 datasets, compute `lh.interval_reading` (a win needs 4 of 5 seeds) and `lh.across_reading` (exact Wilcoxon).

**Primary contrast: `full − static_hv`.** The verdict per horizon is:
- `HELPS`: the MAE interval reading is HOLDS, the MSE interval reading is not REVERSED, and the MSE across reading is not CONSISTENT_DISADVANTAGE.
- `HELPS_MAE_MSE_COST`: the MAE interval reading is HOLDS but the MSE condition fails.
- `MIXED`: the MAE interval reading is PARTIAL.
- `NO_GAIN`: the MAE interval reading is NOT_SHOWN.
- `STATIC_BETTER`: the MAE interval reading is REVERSED.
- `UNDERPOWERED`, `INCOMPLETE` and `INVALID` pass through unchanged.

What the paper may say:
- Only `HELPS` licenses "input-dependent gating improves on the best static gate at h steps".
- `NO_GAIN` or `STATIC_BETTER` mean the paper attributes the gain to static per-cell gating, and input-dependent gating stays a hypothesis.
- The across reading is always reported beside the verdict.

**Ladder (descriptive):** the same readings, with no verdict:
- `constant − core_only`: does interpolation help at all?
- `static_hv − constant`: does variation across cells help?
- `full − static_hv`: does input dependence help? (the primary contrast)
- `router_only − full`: what does the gate add beyond routing?
- `static_hv − static_all`: does input-dependent routing help when the gate is static?

**Other descriptive outputs:**
- Each arm minus Bates all-5, MAE and MSE, read from the existing `comparators_<DS>.pt` files. The R4 units get MAE only, if their per-window file exists on the server; otherwise this is omitted and noted.
- Diagnostics per job:
  - The variance of the full λ on router_val, split into its static part (variance across cells of the window-mean λ̄[h,v]) and its input-dependent part (the mean variance within a cell).
  - corr(λ̄_full, λ_hv).
  - λ_c.
  - The share of λ_hv values at 0 or 1.
  - The routing set change rate: windows whose top-3 set differs from the most common set for that cell.
  - Mean λ and the weight on non-core experts, per arm.

## Files to create (in `experiments/gate_ablation/`)

- **`PREDECLARATION.md`:** everything above, plus the integrity rules. Committed before any server job runs.
- **`run_gate_ablation.py`:** structured like `run_topk_wide_router_ec.py` and `run_untouched_confirmation.py`.
  - Modes:
    - `--selftest`: synthetic tensors on CPU, no data.
    - `--anchor`.
    - `--unit <DS> --horizon <H> --seeds ...`: resumable.
    - `--queue`.
    - `--status`.
    - `--analyze`: refuses until every job is done or recorded as failed. A failed unit makes its horizon INCOMPLETE.
  - Outputs: `_work/<DS>_h<H>/seed<s>.{json,pt}` holding the per-window MAE and MSE per arm, λ_hv[h,v], λ_c, λ̄_full[h,v] and the diagnostics. Also `anchor_check.json`, `results.json`, `integrity_checks.json`, `method_manifest.json` (source hashes and the git commit), `contrasts.csv` and `report.md`.
  - No error metric is printed before `--analyze`.
  - Queue, modelled on `run_queue.py`:
    - Longest jobs first, by the earlier runtimes.
    - 2 jobs per GPU on the 6 GPUs.
    - `OMP_NUM_THREADS=8` and `CUDA_VISIBLE_DEVICES` set per job.
    - Logs go to `logs/gate_ablation/`.
    - A job is retried once, then recorded as failed.
    - When all jobs are final, the queue runs `--analyze` and packs `results_bundle_gate_ablation.tar`.
- **`launch.sh`:** copies the pattern of `long_horizon_protocol/launch.sh`. It refuses a second start, runs the queue under nohup, and has `--status`.

Existing code reused, without changing any frozen file:
- `wide.widened`, `fallback_vector`, `train_scorer_bank_protocol`, `head_inputs`
- `topk.head_arm`, `weights_fn`, `topk_equal`
- `conf.freeze_ok`, `load_pool`, `checkpoint_hashes`, `EXPERT_CHECKPOINTS`
- `lh.configure`, `planned`, `contrast`, `interval_reading`, `across_reading`, `MIN_BLOCKS`
- `lt.confirmation`
- `pe.pool_bundle`, `pool_root`
- `costar_multidataset_frozen.common.verify_router_train_out_of_sample`, `select_core_on_router_train`
- `cf.metrics_of`, `paired`
- `wdec.write_json`, `jsonable`, `sha256_file`, `git_info`, `write_csv_rows`

## Integrity and anchors

**Global anchor** (`--anchor`, once on the server before anything else):
- Runs on the ETTh1 8-epoch development pool, seed 7, with the saved scores `topk_wide_router_ec/_scores/ETTh1_seed7.pt` and the frozen 12-step settings.
- `full` must equal the top-k run's `top3` per-window MAE **EXACTLY**.
- `core_only` must equal the λ = 0 blend exactly.
- `router_only` must equal the λ = 1 blend exactly.

**Per job:** every check below must hold, or the unit is INVALID. It is documented, never silently replaced.
- The freeze matches.
- The global anchor passed.
- router_train is out-of-sample (PASS).
- The core equals the reference run's core.
- All scorer folds are causal.
- The tensors are finite.
- The OOF rows cover the folds.
- The 15 checkpoints are unchanged.
- The reference files are unchanged.
- The stamped settings match the plan.
- `full` reproduces the reference within 1e-3. Whether it is EXACT is recorded.
- `core_only` matches `equal3` within 1e-5.
- The static fits read only OOF tensors.
- `static_all` is identical across seeds (checked at analysis).
- `test_set_accessed: false`.

## Execution order

1. **Laptop** (no training, no GPU):
   - Write the three files.
   - Run `python -m py_compile`, then `python run_gate_ablation.py --selftest`. The self-test checks that:
     - the closed form matches a brute-force grid within 1e-4;
     - λ = 0 and λ = 1 give the fallback and router blends;
     - the variance split adds up;
     - ties in the static top-3 go to candidate order.
   - With the user's go-ahead, commit the predeclaration and the code, then push. No server identifiers go into git.
2. **Server:** `git pull`, then `--anchor`. It must be EXACT.
3. **Server canary:** `--unit AppliancesEnergy --horizon 12 --seeds 7`, about 2 minutes. Check only its reproduction and integrity flags; no metrics are printed.
4. **Server:** `bash experiments/gate_ablation/launch.sh`, then check `--status` until all 70 jobs are final.
5. **Laptop:** copy back `results_bundle_gate_ablation.tar` with PowerShell `scp`, then read `report.md`.

Server instructions will be given one step per reply.

Estimated cost:
- About 0.5–0.6 of the earlier per-seed times, because there is no 3-candidate scorer and only one head instead of three. The static arms are closed form.
- About 18 GPU-hours in total, roughly 3–5 hours of wall time on 6 GPUs.
- The longest single job is Traffic_h192, at about 45–50 minutes.
- No scores are saved, so disk use is negligible.

## Records after the results

- New: `project_memory/experiments/<date>_gate_ablation.md`, the full record including any failures.
- Update `EXPERIMENTS.md` (an entry) and `CURRENT_STATE.md` (the status of the mechanism evidence).
- Update `DECISIONS.md` only for a durable verdict per horizon.
- Update `TODO.md`: Priority 1 done, then Priority 2 (matched conditional baselines).
- The paper's ablation table and wording come later and follow the predeclared mapping above.

## Verification

- The self-test passes on the laptop.
- The server anchor is EXACT.
- The canary record shows reproduction EXACT or within tolerance, with every integrity flag true.
- `--status` shows 70 of 70 jobs final.
- `integrity_checks.json` has `ok` set to true for every unit.
- `report.md` shows the primary verdict per horizon, both readings for MAE and MSE, the ladder and the diagnostics.
- `static_all` is identical across seeds.
