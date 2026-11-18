# Server follow-up suite — 2026-10-20

Prepared after the original gate-ablation results were available. All new
granularity results are exploratory, post-result development analyses. This
document does not retroactively predeclare them. No test split is accessed.

The user requested all four follow-ups. The suite uses the existing server
repository and environment; it does not install packages, stop other workloads,
retrain forecasting experts, or delete files. Original experiment outputs remain
in place. New work is isolated below `experiments/research_suite_20261020` and
`experiments/gate_granularity_exploratory_20261020`.

## Work

1. Retry the original long-horizon queue, with original settings, on one idle GPU.
   Reuse completed jobs; retain a backup of its report bundle, queue state and
   failed-job log before restarting. Require all 148 jobs done, successful
   analysis, and passing integrity checks before calling this recovery complete.
2. Run the unchanged standalone conditional-combination comparison on all seven
   datasets at H=96 and H=192, five seeds, original frozen caches and primary
   results. Use two CPU processes, eight threads and four feature workers each.
   Its original identity and causal checks remain in force.
3. Exercise the actual FAME, TimeRouter and Super-Linear ports on a synthetic
   continuous series, using their pinned source checks and existing fitting and
   selection functions. Repeat with shuffled expert winners as a negative
   control. Histories have 96 observations, forecasts 12, one channel; 384 train
   and 128 evaluation windows; deterministic seed 20261020. Training and
   evaluation are chronological and target-disjoint. Five forecasting experts
   are synthetic, and only the first two alternate as the best candidate. Fixed
   candidate forecasts are independent of the winner; targets choose one of two
   offsets. Shuffled winners cannot be inferred from the current forecasts.
   The positive control's MAE must be less than 75% of both equal weighting and
   the train-selected single expert. Shuffled-control MAE must be at least 75%
   of that single expert. TimeRouter must additionally have winner accuracy at
   least 75% on the positive and below 65% on the shuffled case. Winner accuracy
   is descriptive for convex combiners, which can reconstruct a winning forecast
   with other candidates. Negative-control improvement over weak equal-five
   alone is not evidence of routing.
   FAME uses its original tau/delta grids and one seed (7). TimeRouter uses its
   original OOF/threshold grids and internal five-seed OVA ensemble. Super-Linear
   uses matched pool A, original k grid, one seed (7), and original training
   settings. This is software health evidence, not a published-result reproduction.
   Only ports passing both controls are extended to the seven H=96/192 pools.
   Real extensions retain five seeds and original fitting/selection settings;
   Super-Linear uses pool A only to preserve the identical five experts.
   TimeRouter H=192 is explicitly UNSUPPORTED: its existing CV alignment requires
   a target of length H within the 96-point history. No extra history or silent
   alternative CV rule is introduced. Undefined/failed results stay visible.
4. Run the two window-granularity arms of the amended gate runner in a separate
   directory, retaining the six original arms as matched controls. Require a new
   EXACT anchor, a passing AppliancesEnergy H=12 seed-7 canary, then 70 jobs and
   analysis. Each job retains the original dataset/horizon/seed/batch/chunk
   settings and reference reproduction checks. Schedule one job at a time on an
   idle GPU. This changes scheduling and output location only. The original
   six-arm results are not overwritten. New report text labels every reading
   exploratory and does not license a confirmatory claim.

## Execution and failures

CPU work can proceed while the GPU lane waits. The GPU lane handles the Traffic
recovery first, then exploratory granularity. A GPU must have less than 1 GiB
allocated and less than 5% utilization before a new job starts. This checks
availability, not an exclusive reservation against other server users.
One suite process holds a kernel lock. Stop the earlier interactive GPU-wait
loop before starting this suite. Existing protocol jobs are allowed to finish;
the suite does not kill them. Runtime failures are recorded and are not silently
reclassified as completed. Independent stages continue after a failed stage.

Source/package fingerprints bind resumes. Workers recheck input fingerprints
before reusing or completing real-data units. Status distinguishes WAITING,
RUNNING, COMPLETE, FAILED, BLOCKED and UNSUPPORTED. Overall COMPLETE requires
every requested item to complete; unsupported extension units make the overall
report PARTIAL. The results archive excludes reusable FAME fingerprint caches.
Nothing in this preparation asserts that any server run has started or finished.
