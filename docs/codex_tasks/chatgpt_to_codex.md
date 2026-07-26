# ChatGPT to Codex Prompt Inbox

Status: active

## Active task: begin bounded COSTAR-TS paper-evidence loop

Work on COSTAR-TS for the one-hour completion-driven loop controlled by `watch-chatgpt-codex.ps1`.

For this first iteration, inspect the current repository, recent COSTAR-TS commits, existing tests, saved result summaries, and git status. Then identify and implement the single highest-priority weakness that can be completed safely in one iteration.

Start with the current focused hypothesis: COSTAR-TS should sequentially query frozen forecasting experts, return the equal average of all queried forecasts, and stop when expected ensemble improvement no longer justifies incremental expert cost.

Priority order:

1. Correctness and leakage: verify subset utilities, equal-average finalization, stopping targets, rollout behavior, chronological split use, and cost accounting.
2. Reproducible evidence: make the cost-aware sweep and final comparison runnable across multiple seeds without using the final test split for tuning.
3. Essential baselines and ablations: best fixed expert, fixed top-2 equal average, all-expert equal average, RouterDC hard, COSTAR-TS without cost, COSTAR-TS without sequential forecast evidence, and oracle upper bounds.
4. Paper-facing metrics: MAE, regret, average queried experts, relative expert cost, query-count distribution, stop rate by step, and Pareto cost-accuracy results.
5. Robustness only after the above is correct.

Do not add a large new encoder, reinforcement learning, DAgger, an LLM agent, or a broad hyperparameter search. Do not tune on the final test split. Do not claim paper readiness from one dataset or one seed.

Complete one bounded improvement, run the relevant tests, and record real evidence. If the required caches or checkpoints are unavailable, improve and test the evaluation or training infrastructure without inventing numerical results.

After completing and testing the task:

1. Run `git status` and inspect the exact diff.
2. Preserve unrelated existing changes.
3. Stage only files changed for this task.
4. Commit with a clear task-specific message.
5. Push the completed commit to `origin master`.
6. Never force-push.
7. Report changed files, tests and commands run, real results, commit hash, pushed branch, and blockers.
8. If authentication, conflicts, branch protection, failing tests, missing data, or major compute blocks completion, stop and report the exact blocker.
