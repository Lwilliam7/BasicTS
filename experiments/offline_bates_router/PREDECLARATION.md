# Predeclaration: offline Bates-router (replacing the learned gate)

Written 2026-11-10, before any real-pool outcome of the arms below was computed. The design was chosen from a
throwaway rebuild on public ETTh1 with re-implemented experts (not your pools); those lab results are
disclosed at the end. The method is strictly offline: no realised value from the evaluation period is used at
forecast time.

## Question

Does an offline, closed-form replacement for LambdaHead do better than the learned gate, and does it beat the
archived Bates all-5 comparator?

## Units, seeds, data

- Units: the 14 gate-ablation units. Datasets: ETTh1, Electricity, Traffic, AppliancesEnergy, OikolabWeather,
  SolarAL, TetouanPower. Horizons: 12 and 192.
- Seeds: 7, 11, 13, 17, 19 (scorer seeds, as in the gate ablation). Seed losses are averaged, not predictions.
- A seed-7-only pass over all 14 units is a pilot. It is descriptive and licenses no claim.
- Fit: pooled OOF rows of router_train (20-60%). Score: router_val (60-80%). These are previously scored
  development blocks. No test cache is opened.
- Loaders, the scorer protocol (`wide.train_scorer_bank_protocol`), `head_inputs` and `score_arms` are used
  unchanged from `experiments/gate_ablation/run_gate_ablation.py`.

## Arms

r is the routed forecast: the equal average of the scorer's top 3 of 5 experts in each cell, unchanged.
b is a Bates all-5 fallback: per-cell inverse-MSE weights over all five experts.

- On OOF rows, b is refitted inside each OOF fold, on windows whose targets end before that fold starts.
- On router_val, b uses all router_train windows.

| Arm | Definition |
| --- | --- |
| `bates_only` | b alone. It checks this fallback against the archived comparator. |
| `bates_router_constant` | b + λ(r − b), with one λ fitted in closed form on OOF rows (static_fits' estimator). |
| `bates_router_static` | The same, with one λ per horizon-variable cell. |
| `bates_router_chosen` | Constant or per-cell λ, whichever has lower MAE on the last 30% of OOF windows when fitted on the first 70%. It is then refitted on all OOF rows. **This is the proposed method.** |
| `bates_router_switch` | `bates_router_chosen`, with routing kept only in variables where the 95% moving-block-bootstrap lower bound of the late-OOF gain over b is above zero. Elsewhere it is b alone. |

These reference arms are recomputed by `score_arms`, so their windows match exactly: `full` (the learned
gate), `core_only`, `router_only`, `constant` and `static_hv`. Bates all-5 comes from the archived comparator
files, as in the gate ablation.

## Contrasts

A negative delta favours the first arm. Readings use the gate-ablation machinery: `cf.paired`, `lh.contrast`,
`lh.interval_reading` (a win needs 4 of 5 seeds) and `lh.across_reading` (exact Wilcoxon).

| Contrast | Role |
| --- | --- |
| `bates_router_chosen − full` | Primary 1. Does the offline replacement beat the learned gate? |
| `bates_router_chosen − bates_all5` | Primary 2. Does the method beat Bates all-5? |
| `bates_router_switch − bates_all5` | Primary 3. Does the switched variant beat Bates all-5? |
| `bates_router_switch − bates_router_chosen` | Secondary. Does the switch help or hurt? |
| `bates_router_constant − constant` | Secondary. What does the Bates fallback add at a constant λ? |
| `bates_only − bates_all5` | Descriptive. Agreement between this fallback and the archived comparator. |

## Decision rules (per horizon, MAE)

1. If Primary 1 HOLDS, the learned gate is removed in favour of `bates_router_chosen`. If it is NOT_SHOWN,
   the gate is still removed, because it adds a learned component with no shown benefit, and the paper says
   so. If it is REVERSED, stop and investigate before any change.
2. The paper's offline method is `bates_router_chosen`. It is replaced by `bates_router_switch` only if both
   of these hold:
   - the switch has fewer datasets with a significant MAE loss against Bates all-5;
   - the switch is not significantly worse than `bates_router_chosen` on any dataset.
3. Claims against Bates follow the interval reading of the selected arm against `bates_all5`:
   - HOLDS: "improves MAE over Bates all-5 at h steps".
   - NOT_SHOWN: no improvement is claimed. Lack of significance is not equivalence.
   - REVERSED: report the loss.
4. MSE is reported for every contrast. An MAE claim with a significant MSE loss is labelled as such.

## Integrity per job

- Test cache names are refused, router_train causality must PASS, and checkpoints and input caches must be
  unchanged (sha256).
- All predictions must be finite.
- Where a gate-ablation result exists for the same unit and seed, the recomputed `full` arm is compared window
  for window and the result is recorded. Both the exact flag and the max abs difference are kept.
- The self-test (`--selftest`) covers four things:
  - routing helps on planted regimes;
  - the switch turns off a planted harmful variable;
  - router_val targets cannot change any prediction;
  - each fold's Bates fit sees only windows whose targets end before that fold.

## Cost

There is no new expert training. Each job retrains the scorer exactly as the gate ablation did, so it costs
about one gate-ablation job: the 70 jobs take about the gate ablation's 18 GPU-hours. The seed-7 pilot is
14 jobs.

## Lab disclosure (public ETTh1, re-implemented experts, not your pools)

| Arm | 12 steps (3 seeds) | 96 steps (seed 7) |
| --- | ---: | ---: |
| Learned gate | 0.3655 | 0.5471 |
| `bates_router_constant` | 0.3630 | 0.5475 |
| `bates_router_chosen` | 0.3635 | 0.5464 |
| `bates_router_switch` | 0.3633 | 0.5450 |
| `bates_only` | 0.3638 | 0.5453 |

At 12 steps, every Bates-router arm beat the learned gate, and the 95% block-bootstrap intervals excluded 0.
No arm beat Bates alone significantly, at either horizon.

## Amendment 1 (2026-11-10, before any real-pool result)

A code review of the server path found no interface mismatch with the gate-ablation modules but found
failure-handling, memory and reporting problems. Changes:

- Each job now runs in its own process, with a per-job lock, as the gate-ablation queue does. A job is skipped
  on resume only if its record is valid; an invalid record is renamed `*.invalid-<UTC time>` and the job reruns.
  `analyze` uses valid records only and lists invalid seeds per unit.
- Window chunks follow the unit's `wdec.STAT_CHUNK` (16 at 192 steps), and the scorer's raw score tensors are
  freed after `head_inputs`, to keep the widest 192-step units within memory.
- Each job sets its thread count from `OMP_NUM_THREADS` (default 8), as the gate-ablation main does.
- `analyze` checks that compared arms cover the same windows and prints the block-bootstrap interval
  (`ci95_low`, `ci95_high`).
- **Underpowered switch.** When the late-OOF slice holds fewer than `lh.MIN_BLOCKS` moving blocks
  (`late_independent_blocks` in each record), the switch arm is labelled underpowered. At 192 steps this is
  expected, because the slice holds about 4 blocks. An underpowered switch cannot replace
  `bates_router_chosen` under decision rule 2; it is reported descriptively.

Correction to the lab disclosure: the 96-step lab intervals above used 24-window blocks, which is too short when
consecutive windows share 96 targets. With 96-window blocks, no 96-step lab contrast excludes zero; for example,
the switch against the learned gate is −0.0021 [−0.0052, +0.0005].

## Lab results after amendment 1 (disclosure; no design change)

Public ETT data, re-implemented experts, first 14,400 rows, lab mode at commit d383d12. Each cell is the MAE change
against the learned gate, in percent. "Better" or "worse" marks a 95% moving-block interval (block = max(24, H))
that excludes zero.

| Slice | Seeds | Core-3 constant | Bates alone | Bates + router, constant | Bates + router, chosen | Bates + router, switch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ETTh1 H12 | 3 | −0.25 better | −0.46 better | −0.66 better | −0.54 better | −0.59 better |
| ETTh2 H12 | 3 | +0.21 worse | −0.29 | +0.10 | −0.08 | −0.35 |
| ETTm1 H12 | 3 | −0.32 better | +0.11 | +0.06 | −0.13 | +0.02 |
| ETTm2 H12 | 3 | +0.04 | −0.01 | −0.02 | −0.15 | −0.15 |
| ETTh1 H96 | 1 | +0.13 | −0.32 | +0.07 | −0.13 | −0.39 |
| ETTh2 H96 | 1 | +0.01 | +0.90 | +0.90 | +0.96 | +0.90 |
| ETTm1 H96 | 1 | +0.10 | +1.52 worse | +0.23 | +0.23 | +0.49 |
| ETTh1 H192 | 1 | +0.00 | +0.62 | +0.08 | +0.41 | +0.41 |

No offline arm was reliably better than the learned gate. The Bates fallback helped where Bates beat the equal
core-3 and hurt where it did not. The late-OOF choice picked the per-cell form in 7 of 8 slices, although the
constant form did better on router_val in some of them. The switch's routed-variable count varied across seeds.
