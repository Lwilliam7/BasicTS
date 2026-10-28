# Amendment 1 to the gate-ablation predeclaration — 2026-10-19

> **Note added 2026-10-20, timeline corrected the same day: the precondition failed. This
> amendment is not a pre-result change and has no predeclared status.**
>
> - All 70 unit jobs of the original predeclaration had already run when this file was written.
>   The server timeline, all on 2026-10-18 (UTC):
>   - anchor at 21:18;
>   - canary at 21:38;
>   - last job at 23:24;
>   - `--analyze` at 23:26.
> - The result bundle was on the laptop by 04:06 UTC on 2026-10-19. At 04:07 UTC another Claude
>   chat read `report.md` from it, and at 04:09 UTC it reported both verdicts in that chat.
> - This file was drafted in a separate chat from 06:23 UTC. That chat worked from notes that
>   still said the queue had not started, and it never saw the results. The user approved the
>   draft at 06:45 UTC, and it was committed at 06:50 UTC (`ddaaf89`).
> - The precondition check was given to the user only after the commit, and no result was
>   reported. The precondition below (no `experiments/gate_ablation/_work/` and no
>   `queue_state.json` on the server) was never checked, and it was false. By its own rule, this
>   amendment cannot be applied as a pre-result change.
> - The statement under "Provenance and preconditions" that it was written before any outcome was
>   known is false. The outcome had been read and reported 2 h 36 min before the approval.
> - The full timeline, with chat references, is in the gate-ablation record named below.
> - Consequence: `window_gate`, `window_route` and secondary verdicts G and R have no predeclared
>   status. If they are ever run, report them as exploratory, post-result analyses. They cannot
>   license the sentences under "What the paper may say".
> - The original run's results are recorded in
>   `project_memory/experiments/2026-10-18_gate_ablation.md`. They come from the runner at
>   `e815f5b` (hash-identical to the server copy), not from the amended runner in `ddaaf89`.
> - The text below is unchanged.

**Status: approved by the user as drafted on 2026-10-19.** Not yet committed. It becomes binding
when it is committed together with the matching code change, before the canary or any other unit
job runs. `PREDECLARATION.md` stays byte-identical to commit `e815f5b`. This file adds to it and
changes nothing in it.

## What this amendment does

It adds two arms that test **cell granularity**, the one part of CellCatch's claim the original
arms leave open. It also adds one descriptive ladder rung. Neither arm trains anything. The
primary contrast (`full − static_hv`), its verdict mapping, the six original arms, the units, the
seeds, the anchor rules and the per-job integrity rules are all unchanged.

Why: the original arms test whether the gate's **input dependence** helps. They do not test
whether making decisions **per cell** (rather than once per window) helps, for routing or for the
gate. EMTSF (arXiv 2510.23396) already learns input-dependent per-variable, per-step weights over
frozen experts. So the paper's "per-cell" wording needs its own evidence, separate from the
input-dependence evidence.

## Provenance and preconditions

- 2026-10-18: the original files were committed and pushed (`e815f5b`) and uploaded to the server.
  Server `--selftest` passed all 9 groups. The global anchor ran once and wrote `EXACT; integrity
  PASS` under the original manifest.
- The anchor records booleans and hashes only (`anchor_check.json`). No error value of any arm was
  printed or saved in a readable form. This amendment is therefore written before any gate-ablation
  outcome is known.
- Motivation comes from other experiments' records and the 2026-10-19 planning review, not from
  this experiment.
- **Precondition, checked on the server before the amended files are uploaded:**
  `experiments/gate_ablation/_work/` and `experiments/gate_ablation/queue_state.json` do not exist.
  If either exists, a unit job has started. In that case this amendment cannot be applied as a
  pre-result change: stop, and record what ran.

## New arms

Notation as in `PREDECLARATION.md`:
- `c` is the equal core-3 forecast. `r` is the per-cell equal top-3 forecast of the frozen method.
- `a[n,h,v,·]` are the 5 scorer affinities of cell (h, v) in window n.
- λ[n,h,v] is the frozen `LambdaHead` output captured on router_val (the `full` arm).
- All means over cells are plain means over all H×V cells. They use no targets and no masks.

| Arm | Router part | Gate | What it answers |
|---|---|---|---|
| `window_gate` | per-cell top-3 (as `full`) | λ̄[n] = mean over (h, v) of λ[n,h,v], applied to every cell of window n | Does per-cell variation of the learned gate matter, compared with its own window average? |
| `window_route` | one top-3 set per window, from ā[n,·] = mean over (h, v) of a[n,h,v,·] | static λ_wr[h,v], same closed form as `static_hv`, refitted for this router on the same pooled OOF rows | Does per-cell expert selection matter, compared with per-window selection, under the same static gate family? |

### `window_gate`

- Prediction: `c + λ̄[n] · (r − c)`, built with `topk.weights_fn(fallback, 3)(val_aff, λ̄ expanded to every cell)`.
  That is the full arm's own weight code with only the λ tensor changed, as for the original arms.
- Inputs: the captured validation λ only. No fit, no OOF rows, no targets.
- **Bias, declared in advance.** λ̄ is a projection of a head trained for per-cell output. It is not
  a window-level gate trained for its own loss, so the contrast favours `full`.
  - A `full` win shows that removing the gate's cell variation hurts. It does not show that no
    window-level gate could match the full gate.
  - `NO_GAIN` or `WINDOW_BETTER` is the stronger reading. Even an unoptimised window gate matches
    the per-cell gate.

### `window_route`

- Window set: S[n] is the 3 candidates with the highest ā[n,·]. Use a stable descending sort, so a
  tie goes to candidate order, as in `static_all`. The number of OOF and validation windows with a
  tie at ranks 3/4 is recorded.
- Routed forecast: `r_win[n,h,v]` is the equal average of the S[n] experts at every cell of window n.
- Gate: λ_wr[h,v] = clip(−Σ_n m·e·d ÷ Σ_n m·d², 0, 1), with e = c − y and d = r_win − c.
  - This is the `static_hv` closed form, on the same pooled OOF rows, in float64, in
    `wdec.STAT_CHUNK` window chunks.
  - A zero-denominator cell gets λ = 0, and the count is recorded.
  - The OOF window sets come from the OOF affinities. Validation sets come from validation
    affinities only.
- Prediction: `c + λ_wr[h,v] · (r_win − c)`.
- Why it is compared with `static_hv`, not `full`: both arms then use the identical static
  estimator on identical rows. Only the routing granularity differs.
- Considered and not adopted:
  - Applying the frozen head's λ to `r_win`. The head was trained for per-cell routing, so that
    would penalise window routing by a calibration mismatch.
  - Training a new head for window routing. That needs training, which this amendment excludes.

## Readings (amended)

`LADDER` gains three rungs. Every rung uses the existing machinery: seed-averaged per-window
errors, `cf.paired` with the unit's block and phase, `lh.contrast`, `lh.interval_reading` (a win
needs 4 of 5 seeds) and `lh.across_reading` (exact Wilcoxon). Each new arm is also compared with
Bates all-5 automatically, MAE and MSE where available, as for the original arms.

| Rung | Question | Status |
|---|---|---|
| `full − window_gate` | Does per-cell variation of the learned gate help? | **Secondary verdict G** |
| `static_hv − window_route` | Does per-cell selection help, under the same static gate family? | **Secondary verdict R** |
| `window_gate − constant` | Does a window-level, input-dependent gate beat a constant gate? | Descriptive, no verdict |

A negative delta favours the first arm, as for the primary contrast.

**Secondary verdict mapping** (per horizon, for G and R separately; same structure as the primary):
- `CELL_HELPS`: MAE interval HOLDS, MSE interval not REVERSED, and MSE across reading not
  CONSISTENT_DISADVANTAGE.
- `CELL_HELPS_MSE_COST`: MAE interval HOLDS but the MSE condition fails.
- `MIXED`: MAE interval PARTIAL.
- `NO_GAIN`: MAE interval NOT_SHOWN.
- `WINDOW_BETTER`: MAE interval REVERSED.
- `UNDERPOWERED`, `INCOMPLETE` and `INVALID` pass through unchanged.

**What the paper may say:**
- **Primary verdict.** Unchanged. It alone decides the input-dependence claim.
- **G = `CELL_HELPS`.** "Replacing each window's per-cell gate with its window average increases
  MAE at h steps." Not "a window-level gate cannot match it" (see the declared bias).
- **G = `NO_GAIN` or `WINDOW_BETTER`.** The paper does not claim that the gate's cell resolution
  matters. It describes the gate as effectively window-level at that horizon.
- **R = `CELL_HELPS`.** "Per-cell top-3 selection improves on per-window top-3 selection under the
  same static gate family at h steps."
- **R = `NO_GAIN` or `WINDOW_BETTER`.** Per-cell selection is not shown to matter. The paper does
  not credit routing granularity.
- **G and R both `NO_GAIN` or `WINDOW_BETTER` at a horizon.** The "per-cell" framing is unsupported
  for this method at that horizon, and the paper says so.
- Neither verdict compares CellCatch with EMTSF. That needs its own experiment.
- **Multiplicity.** Each horizon now has three verdicts: the primary, G and R. There is no
  multiplicity adjustment. Each verdict licenses only its own sentence, and all three are reported
  together.

## Added diagnostics (per job; descriptive)

- **λ split.** `variance_window_means` is the population variance across windows of λ̄[n].
  `variance_within_window` is the mean over windows of the population variance of λ[n,·,·] across
  cells. Together they sum exactly to `variance_total`. The original static/input split is kept.
- **λ fits.** `lambda_window_route_hv[h,v]` and its zero-denominator count.
- **Agreement.** `window_route_cell_agreement` is the share of validation cells whose per-cell
  top-3 set equals their window's set. Agreement near 1 means `window_route` cannot differ much from
  `static_hv`, and the report states it next to R.
- **Ties.** The number of OOF and validation windows with a tie at ranks 3/4 of ā[n,·].
- **Per arm.** Mean λ and non-core weight share, extended to the new arms by the existing code.

## Integrity (additions)

- The `static_fits_oof_only` flag covers the `window_route` fit.
- `window_gate` and `window_route` are finite on every job.
- No new reference exists for either arm. The global anchor keeps its three EXACT checks (`full`,
  `core_only`, `router_only`) and adds finiteness flags for the two new arms, without values.
- Self-test additions (synthetic CPU tensors only, no data loads):
  1. With λ constant across the cells of each window, `window_gate` equals `full` exactly.
  2. With affinities identical across the cells of each window, `r_win` equals `r`, and λ_wr equals
     λ_hv exactly.
  3. λ_wr matches a brute-force grid within 1e-4 per cell.
  4. Ties in ā go to candidate order.
  5. Changing validation targets leaves λ_wr unchanged (validation-label isolation).
  6. The λ variance split (window means plus within window) adds up.
  7. The eight-arm plumbing runs with the mock head.

## Files to change after approval

- `run_gate_ablation.py`:
  - `ARMS` gains `window_gate` and `window_route`. `LADDER` gains the three rungs.
  - A window-route fit runs alongside `static_fits` on the same OOF tensors.
  - `score_arms` builds the two arms.
  - `lambda_diagnostics` gains the window split and agreement/tie counts.
  - `analyze` writes secondary verdicts G and R to `results.json` and `report.md` beside the
    primary.
  - The self-test gains the cases above. `source_hashes()` includes this file.
- `PREDECLARATION.md` and `launch.sh`: unchanged.
- Commit this amendment and the code together, before any upload. The commit message names the
  amendment.

## Server consequences

The source manifest changes, so the existing anchor no longer matches and the runner will refuse
it. Nothing is deleted.

1. Check the precondition above.
2. Move `experiments/gate_ablation/anchor_check.json` and `method_manifest.json` into
   `experiments/gate_ablation/_superseded_e815f5b/`. Rename `logs/gate_ablation/anchor.log` to
   `anchor_e815f5b.log`. The result bundle will carry the preserved files.
3. Upload the amended files, rerun `--selftest`, then `--anchor` (it must be EXACT again), then the
   canary and the queue as in `PREDECLARATION.md`.

Server commands are given one step per reply after the code is approved.

## Cost

There is no new training. Per job, the amendment adds one more closed-form pass over the OOF rows
and two extra validation scorings. The predeclared estimate (about 18 GPU-hours, 3–5 hours of wall
time on 6 GPUs, 70 jobs) is unchanged up to that overhead. The per-window outputs for two more arms
add little disk.

## Evidence status

Unchanged: development evidence on previously scored 60–80% blocks. No test split is used. The
frozen method does not change, whatever the outcome.
