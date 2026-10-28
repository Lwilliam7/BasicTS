# Four-port synthetic health audit — 2026-10-25

Purpose: execute FAME, TimeRouter, OneNet and Super-Linear on controlled synthetic
cases with known oracle forecasts, and preserve failures. This is a software
health audit. It is not a reproduction of a result from any of their papers.
No paper-comparison row is cleared by this audit: `table_eligible` remains false.
The earlier suite omitted OneNet; its results and deployed source are unchanged.

Execution correction after the first local audit: an empty CUDA visibility mask
did not disable CUDA on this Windows host. FAME's upstream auto-device selector
used CUDA and independent fits failed the exact permutation/reproducibility
check despite identical canonical inputs. The initial results and executed
sources are preserved in `research_review/baseline_health_20261025/`. A separate
`baseline_health_20261025_cpu` run explicitly sets FAME's configuration device to
CPU and records/asserts the actual device. Fitting, seeds, grids and thresholds
are unchanged. This is an execution correction, not a relaxed pass criterion.

Diagnostic correction on 2026-10-26: capture FAME winner accuracy from its primary
top-2 evaluation, matching the exported losses. The earlier helper captured the
last probability call, from the secondary top-1 model, which can select a
different tau. This changes only the supplementary diagnostic; fitting,
selection, forecasts and health thresholds are unchanged. The preceding recheck
and its source snapshot remain in `research_review/baseline_health_20261026_recheck`.

## Source and scope

Use the existing port implementation and its original fitting/selection settings.
Run the port's pinned upstream file-hash/source-equality gate before any fit.
Record both upstream and port hashes and the local environment. Public upstream
snapshots can be used locally; no clone is silently substituted or downloaded by
the health runner. Each worker is a separate CPU process. No real forecast cache,
checkpoint, validation block or locked test block is loaded.

- FAME: authors' oracle/router and parity-checked fingerprint/sparse-selection
  port, over synthetic experts; original tau/delta selection; seed 7.
- TimeRouter: authors' features, probability averaging, gate and fallback;
  existing retrained OVA head and OOF threshold selection, including its five
  internal seeds. This head is an adaptation rather than released training code.
- Super-Linear: existing spectral-gate port, matched five-expert pool A,
  original k grid and training settings, seed 7. Its frequency expert training
  is outside the frozen-expert port.
- OneNet: the existing two-branch OCP block on frozen PatchTST/iTransformer
  forecasts, seed 7; original four warmup epochs, reset, optimizer and dropout
  rules. Branch co-adaptation is absent. Feedback is delayed until a target
  resolves. This is explicitly not full OneNet.

## Shared offline controls

Reuse the October 20 continuous synthetic fixture: input 96, H=12, V=1,
384 training and 128 evaluation windows, stride H. Histories and targets agree
at CV origins. A spectral regime determines which of two synthetic forecasts
is best. Candidate forecasts are independent of the winner. The analytic oracle
MAE is 0.02. A shuffled-winner control removes predictability while leaving the
candidate forecasts unchanged.

FAME, TimeRouter and Super-Linear must:

1. On the predictable case, achieve MAE below 75% of both equal-five and the
   training-selected single expert.
2. On shuffled winners, have MAE at least 75% of that single expert. Beating weak
   equal-five is insufficient to establish input-dependent learning.
3. Produce finite errors, convex finite weights where captured, and aligned
   diagnostics. Report accuracy, expert usage and fallback frequency when available.
4. Produce the same per-window losses, within 1e-6, after jointly permuting the
   cache's expert names and forecast columns. The canonical method input remains
   identical; this catches index/name wiring errors.

For TimeRouter, also require winner accuracy >=75% on the predictable case and
<65% on shuffled winners. Winner accuracy is descriptive for convex combiners.

## OneNet controls

OneNet does not consume the history fingerprint to choose a current-window
winner, so do not demand a cold-start spectral router from this OCP port.
Exercise its actual online mechanism instead, on the same continuous-series
layout with the two correct branch names:

1. Branch 1 consistently wins; then a separate case where branch 2 consistently
   wins. The winning forecast has known MAE 0.02 and the other branch has MAE
   about 2. The two-branch equal mix has MAE about 1. Require at least half the
   achievable MAE reduction from equal mix toward the oracle, over the complete
   evaluation period. Report tail error, effective weights and online updates.
2. IID shuffled winners with the same candidate forecasts. Require MAE at least
   75% of equal-two; current labels cannot predict their own window.
3. Audit every online update against its origin: `update_origin + H <= prediction_origin`.
   Verify the tensors passed to update/predict correspond to the recorded row.
4. Poison targets from evaluation row 64 onward, hold all inputs and training
   data fixed, repeat with the same seed, and require predictions through row 64
   inclusive to be exactly unchanged. No poisoned target is observable by then.
5. Jointly permute expert names/columns and require equivalent per-window losses.

All branch learning settings remain unchanged. Instrumentation observes the
actual OCP methods; it does not replace their optimization or predictions.

## Reporting

Archive a source manifest, environment, exact controls/thresholds, per-window
losses, OneNet causality trace and negative cases. Runtime failures and failed
criteria remain explicit. A software pass leaves paper fidelity NOT_ESTABLISHED
and paper-table eligibility false. Full-method claims require an additional
paper-fidelity/reproduction audit and a protocol-compatible comparison. The
online OneNet block cannot silently join a ranking of offline frozen routers.
