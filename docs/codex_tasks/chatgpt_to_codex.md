# ChatGPT to Codex Prompt Inbox

Status: active
Queue ID: COSTAR-TS-one-hour-2026-07-26-run-2

## Active task: verified one-hour COSTAR-TS improvement loop

Run a completion-driven COSTAR-TS research loop for one hour, controlled by `watch-chatgpt-codex.ps1 -LoopHours 1`.

Before changing research code, verify that the loop can work safely:

1. Confirm the repository is on the intended branch and inspect `git status`.
2. Confirm the watcher can launch one Codex iteration, wait for it to finish, and launch the next iteration only after successful completion.
3. Confirm the one-hour deadline prevents new iterations after the deadline.
4. Preserve unrelated user changes and never run two research iterations concurrently.
5. If the watcher logic is broken, fix and test the watcher first. Do not pretend research work ran.

For every research iteration:

1. Inspect the latest COSTAR-TS results, commits, tests, saved summaries, and unresolved blockers.
2. Identify the single largest verified weakness.
3. Make one bounded change that can be evaluated independently.
4. Run focused correctness tests and the smallest meaningful experiment.
5. Select changes using training and validation data only. Do not tune on the final test split.
6. Keep the change only if it improves correctness, reproducibility, evidence quality, validation performance, or the cost-accuracy tradeoff.
7. If evidence does not support the change, revert only that iteration's changes and record the negative result.
8. Commit and push only a valid, tested improvement, then repeat until the one-hour deadline.

Research priorities, in order:

1. Data leakage, chronological splits, subset utilities, equal-average finalization, stopping targets, rollout behavior, and cost accounting.
2. Strong simple baselines: best fixed expert, fixed top-2 equal average, all-expert equal average, RouterDC hard, and oracle bounds.
3. Reproducible multi-seed validation without test-set tuning.
4. MAE versus inference cost, regret, average queried experts, query-count distribution, and Pareto results.
5. Ablations for cost penalty and sequential forecast evidence.
6. Multiple datasets and statistical uncertainty only after the core pipeline is correct.

Do not add a large encoder, reinforcement learning, DAgger, an LLM agent, or a broad sweep unless existing evidence directly requires it. More code is not an improvement. Do not invent numerical results or claim paper readiness from one seed or dataset.

The current focused hypothesis is: COSTAR-TS sequentially queries frozen forecasting experts, returns the equal average of all queried forecasts, and stops when expected ensemble improvement no longer justifies incremental expert cost.

At the end of each completed iteration:

1. Inspect the exact diff and run `git status`.
2. Stage only files belonging to that iteration.
3. Commit with a task-specific message and push to `origin master`.
4. Report changed files, commands and tests run, real results, commit hash, branch, and blockers.
5. If authentication, conflicts, branch protection, failing tests, missing data, or compute blocks completion, stop and report the exact blocker.

At the end of the hour, produce a final summary separating verified improvements, negative results, unfinished work, and the next highest-priority experiment.
