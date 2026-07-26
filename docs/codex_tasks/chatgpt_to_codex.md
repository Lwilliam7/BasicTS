# ChatGPT to Codex Prompt Inbox

Status: active
Queue ID: COSTAR-TS-six-hour-2026-07-26-run-1

## Active task: verified six-hour COSTAR-TS improvement loop

Run a completion-driven COSTAR-TS research loop under the six-hour deadline enforced by `watch-chatgpt-codex.ps1 -LoopHours 6`.

First verify the run is safe:

1. Inspect the current branch, latest `origin/master`, and `git status`.
2. Preserve unrelated user changes. Never run overlapping Codex iterations.
3. Confirm the watcher deadline prevents starting another iteration after six hours.
4. If watcher behavior is broken, fix and test it first. If it cannot be fixed safely, stop and create the repair prompt with the exact error and log path.

For each completed research iteration:

1. Inspect the newest COSTAR-TS code, tests, result files, and previous negative findings.
2. Find the single largest evidence-backed weakness.
3. Make one bounded, independently testable change.
4. Run focused correctness tests and the smallest meaningful experiment.
5. Use training and validation data for development. Never tune against the final test split.
6. Keep the change only if it improves correctness, reproducibility, evidence quality, validation performance, or the measured cost–accuracy tradeoff.
7. If the idea fails, revert only that iteration's edits, record the negative result, and do not commit useless complexity.
8. After a valid pushed commit, begin the next review iteration if the six-hour deadline has not been reached.

Priorities:

1. Leakage, chronological splits, target construction, rollout behavior, stopping logic, equal-average finalization, and cost accounting.
2. Strong simple baselines: best fixed expert, fixed top-2 equal average, all-expert equal average, RouterDC hard, and oracle bounds.
3. Multi-seed validation and uncertainty without test-set tuning.
4. Validation MAE, inference cost, regret, average queried experts, query-count distribution, and Pareto results.
5. Ablations for the cost penalty and sequential forecast evidence.
6. Multiple datasets only after the core pipeline is demonstrably correct.

Do not add a large encoder, reinforcement learning, DAgger, an LLM agent, or a broad hyperparameter sweep unless existing results directly justify it. Do not invent results or declare the work paper-ready from one dataset or seed.

Core hypothesis: COSTAR-TS sequentially queries frozen forecasting experts, returns the equal average of all queried forecasts, and stops when expected ensemble improvement no longer justifies the incremental expert cost.

At the end of every valid iteration:

1. Review the exact diff and `git status`.
2. Stage only files belonging to the iteration.
3. Commit with a specific message and push to `origin master`. Never force-push.
4. Verify that `origin/master` contains the new commit.
5. Report files changed, tests and experiments run, actual results, commit hash, and blockers.
6. On authentication failure, conflict, branch protection, failed tests, missing data, or compute failure, stop and write the ready-to-run repair prompt with the exact failure and log location.

When the watcher reaches its six-hour deadline, do not start another iteration. Allow an already-running iteration to finish safely, then provide a final summary separating verified improvements, negative results, unfinished work, and the next highest-priority experiment.
