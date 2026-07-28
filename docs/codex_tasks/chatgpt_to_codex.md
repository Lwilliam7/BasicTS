# ChatGPT to Codex Prompt Inbox

Status: active
Queue ID: COSTAR-TS-loss-aware-pair-selector-2026-07-28

## Active task: redesign the history-only pair selector using all pair errors

Run one bounded COSTAR-TS research iteration. Replace the current hard oracle-best-pair classification objective with a loss-aware objective that uses the individual validation-safe training error of every candidate pair.

### Safety and scope

1. Inspect the current branch, latest `origin/master`, `git status`, current pair-selector implementation, focused tests, cached frozen-expert predictions, and the newest pair-selector reports.
2. Preserve unrelated user changes. Do not overwrite or delete existing results.
3. Use only the existing chronological expert-train, expert-validation, router-train, and router-validation splits.
4. Do not inspect, regenerate, tune on, or report final test-set values.
5. Keep every forecasting expert frozen. Train only the history encoder and selector heads.
6. Do not launch a broad architecture or hyperparameter search. Make one focused redesign and compare it fairly with the existing exact-pair classifier and fixed-pair baseline.

### Required redesign

The deployment objective is not to identify the exact oracle pair. It is to decide whether any candidate pair is expected to improve on the strong fixed pair.

For each router-training window and each candidate pair `j`, use the already available frozen forecasts to compute:

`pair_error[j] = MAE(pair_forecast[j], true_future)`

`improvement_target[j] = fixed_pair_error - pair_error[j]`

Use the individual target for every pair, not only the identity of the minimum-error oracle pair.

Implement a history-only selector that:

1. Receives exactly the same causal history input as the current selector.
2. Outputs one predicted improvement value for every candidate pair.
3. Trains with a stable regression loss across all pair targets, preferably SmoothL1 unless the existing implementation gives a strong reason for another simple loss.
4. Includes the fixed pair as the default action.
5. At inference, chooses the pair with the largest predicted improvement only when that prediction exceeds a threshold selected exclusively on router-validation data; otherwise it keeps the fixed pair.
6. Never uses true future values, realized pair errors, or oracle labels at inference.

Keep the current exact-pair classifier intact as a comparison baseline rather than silently replacing its historical results.

### Validation protocol

Run the same supported seed set used by the current report, preferably `7, 11, 13, 17, 19`, unless repository evidence shows a different locked set. Reuse the same pair pool, fixed pair, caches, preprocessing, and split boundaries so the comparison is apples-to-apples.

Select the switching threshold using router-validation only. Do not choose a threshold by looking at test results. Include a no-switch option and avoid arbitrary switch-rate constraints unless reported only as an explicit secondary diagnostic.

Report at minimum:

1. Fixed-pair validation MAE.
2. Existing exact-pair classifier validation MAE.
3. New loss-aware selector validation MAE, mean and standard deviation across seeds.
4. Per-seed MAE differences relative to the fixed pair.
5. Switch rate and its variation across seeds.
6. Mean regret to the oracle pair.
7. Realized MAE on switched windows versus the fixed pair on those same windows.
8. Fraction of switches that actually improve over the fixed pair.
9. AUC or another clearly defined separator metric for predicted improvement versus whether switching truly helps.
10. Regression diagnostics for predicted versus realized improvement, such as MAE and Spearman correlation.
11. Results on the previously defined high-margin subset, including oracle pair margin above `0.025`.

Exact-pair accuracy may be reported as a secondary diagnostic, but it is not the optimization target and must not be used as the main success criterion.

### Success criteria

Call the redesign successful only if all of the following hold:

1. It improves mean router-validation MAE over the fixed pair.
2. The improvement is not driven by only one seed.
3. Switching behavior is not degenerate across seeds.
4. Predicted improvement meaningfully separates helpful from harmful switches.
5. The result is obtained without test-set tuning or leakage.

If it does not meet these criteria, state clearly that history-only loss-aware routing failed. Do not add a confidence gate, larger encoder, reinforcement learning, or another complicated repair in this iteration. Recommend the next controlled experiment as a forecast-aware router using frozen expert forecasts or disagreement summaries.

### Tests and evidence

Add or update focused tests that verify:

1. Every pair contributes an individual improvement target.
2. Improvement signs are correct: positive means the candidate beats the fixed pair.
3. Targets are constructed only from the permitted training split.
4. The inference path never consumes future targets or realized errors.
5. The default fixed pair is selected below threshold.
6. The highest predicted-improvement pair is selected above threshold.
7. Metrics and threshold selection use router-validation only.

Run the smallest meaningful experiment that completes the multi-seed comparison. Save a concise report beside the existing pair-selector results without overwriting prior reports.

### Finish

Review the exact diff and `git status`. Stage only files belonging to this experiment. If the implementation is correct and the evidence is reproducible, commit with a specific message and push to `origin master`; never force-push. Report the files changed, tests run, experiment command, per-seed and aggregate results, selected threshold behavior, commit hash, and any blocker. If the method fails scientifically, preserve only useful tests and a clearly labeled negative-result report; do not claim success or retain unjustified complexity.