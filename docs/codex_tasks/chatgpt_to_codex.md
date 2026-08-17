# ChatGPT to Codex Prompt Inbox

Status: active
Queue ID: COSTAR-TS-artifact-provenance-cleanup-2026-07-28

## Active task: repair and regenerate inconsistent sequential COSTAR-TS artifacts

Run one bounded repository-cleanup and reproducibility iteration for the sequential subset-utility COSTAR-TS system. The implementation has already moved to deployable equal-average finalization, but tracked evaluation artifacts still contain results produced with reranker or sparse-mixture finalizers. Fix the generators and provenance schema first, then regenerate the affected validation-only artifacts.

## Verified repository problems to confirm

1. `scripts/evaluate_costarts_cost_sweep.py` currently defaults to `equal_average`.
2. `scripts/run_costarts_subset_utility_ablations.py` currently defaults to `mix_loss_weight=0.0` and `finalizer="equal_average"`.
3. The tracked `results/router_summary/costarts_subset_utility/pareto_curve.json` still records `finalizer: reranker`.
4. The tracked `results/router_summary/costarts_subset_utility/ablations.csv` still contains active rows labeled `sparse_mixture`, `reranker`, and `cost_aware_reranker`.
5. The copied paper-package ablation table contains the same stale rows.
6. `final_comparison.csv`, `cost_sweep.csv`, and `ablations.csv` do not record enough row-level provenance to identify the exact checkpoint, cache, seed, finalizer, and inference rule that produced each number.
7. The active Codex task before this one concerned the loss-aware pair selector. Do not modify that experiment unless required to preserve unrelated work. This task is limited to sequential COSTAR-TS artifact correctness.

Treat these as starting evidence, not assumptions. Inspect current `origin/master`, the local branch, `git status`, relevant scripts, focused tests, checkpoints, caches, and tracked result files before editing.

## Safety and scope

1. Preserve unrelated user changes. Stage only files required by this cleanup.
2. Do not inspect, regenerate, tune on, or report any test-set or locked-test values.
3. Use only existing chronological router-train and router-validation data, caches, and frozen expert predictions.
4. Do not update forecasting expert parameters.
5. Do not change the sequential architecture, routing objective, or scientific method in this iteration.
6. Do not claim stale numbers are valid. Delete or replace only the explicitly targeted generated artifacts after the generators and tests are ready.
7. Never force-push.

## Required provenance schema

Create one shared, tested provenance implementation instead of duplicating ad hoc hashing logic across scripts. Use streaming SHA-256 file hashing.

Every row written to each relevant CSV must contain nonempty values for these exact fields:

- `checkpoint_hash`
- `finalizer`
- `seed`
- `inference_rule`
- `cache_hash`

Also record these where practical:

- `checkpoint_path`
- `cache_path`
- `code_commit`
- `artifact_schema_version`
- `selection_split`
- `test_data_used`

Rules:

1. If a method has no checkpoint, write an explicit value such as `not_applicable`; do not leave the field blank.
2. If a method uses multiple checkpoints or caches, store a deterministic JSON mapping with sorted keys in both the path and hash fields.
3. `cache_hash` must refer to the actual cache or caches used for the reported row.
4. `checkpoint_hash` must refer to the actual checkpoint or checkpoints used for the reported row.
5. `inference_rule` must describe the real deployable rule, for example `sequential_action_logits_then_equal_average`, `forced_top2_then_equal_average`, `fixed_equal_average_all`, or another precise rule supported by the code.
6. `finalizer` must describe the prediction actually evaluated. Do not label an equal-average prediction as a reranker result.
7. Repeat the provenance fields on every cost-sweep row rather than storing them only once in JSON metadata.
8. JSON outputs must contain the same provenance and must agree with the CSV rows.

## Generator repairs

Audit and update at least these paths as needed:

- `scripts/evaluate_costarts_final_comparison.py`
- `scripts/evaluate_costarts_cost_sweep.py`
- `scripts/run_costarts_subset_utility_ablations.py`
- `scripts/build_costarts_paper_package.py`
- focused COSTAR-TS tests

Required behavior:

1. The deployable sequential subset-utility result uses the equal average of queried frozen-expert forecasts.
2. The cost sweep defaults to and explicitly records `equal_average`.
3. The active ablation baseline uses `mix_loss_weight=0.0` and `equal_average` finalization.
4. `no_sparse_mixing` is not presented as a live improvement when sparse mixing is not part of the deployable method. It should remain skipped or clearly non-applicable.
5. `cost_aware_stopping` must use cost-aware stopping with the equal-average finalizer, not a reranker.
6. Reranker, sparse-mixture, and oracle-best-queried outputs may remain only as clearly labeled diagnostic results when intentionally requested. They must not be substituted for the deployable method or copied into the main paper tables as its performance.
7. Paper-package tables must be rebuilt from the newly generated source artifacts, not copied from stale files.
8. The paper package must not silently mix artifacts from different checkpoints, caches, seeds, finalizers, or inference rules.

## Artifacts to remove and regenerate

Regenerate the current validation-only versions of:

- `results/router_summary/costarts_subset_utility/final_comparison.csv`
- the matching `final_comparison.json`
- `results/router_summary/costarts_subset_utility/cost_sweep.csv`
- `results/router_summary/costarts_subset_utility/pareto_curve.json`
- `results/router_summary/costarts_subset_utility/ablations.csv`
- the matching `ablations.json`
- generated files under `results/router_summary/costarts_subset_utility/paper_package/tables/`

Regenerate any paper-package LaTeX tables that depend on those files. Rebuild plots only when the normal paper-package command does so and they directly depend on the regenerated tables.

Do not delete unrelated diagnostics, reports, checkpoints, caches, or historical experiment outputs.

## Ablation integrity

Inspect whether existing ablation checkpoints are full supported runs or one-epoch smoke tests.

1. Do not publish smoke-test rows as paper-ready evidence.
2. Prefer valid existing checkpoints when they match the current architecture and configuration exactly.
3. If a required active ablation lacks a valid checkpoint, run the supported full configuration rather than silently using `--max-epochs 1`.
4. Record checkpoint epoch, configuration, checkpoint hash, and whether the row came from a full run, reused valid checkpoint, skipped ablation, or smoke diagnostic.
5. If a full regeneration cannot be completed because a required valid checkpoint or cache is absent, still fix the code and tests, regenerate everything that is valid, mark the blocked rows clearly, and report the exact blocker. Do not invent numbers.

## Required tests

Add or update focused tests that verify all of the following:

1. Every generated CSV row has nonempty `checkpoint_hash`, `finalizer`, `seed`, `inference_rule`, and `cache_hash` fields.
2. Stored SHA-256 values match the actual files.
3. Deterministic multi-file provenance mappings have stable ordering.
4. The cost-sweep default finalizer is `equal_average`.
5. The active ablation specifications use `equal_average` and disable unused mix loss.
6. `no_sparse_mixing` is skipped or non-applicable.
7. `cost_aware_stopping` evaluates the equal-average finalizer.
8. Active deployable rows and main paper tables do not contain `sparse_mixture`, `reranker`, or `cost_aware_reranker` finalizers.
9. Intentionally retained reranker or oracle diagnostics are clearly labeled diagnostic and excluded from deployable comparisons.
10. CSV and JSON provenance agree.
11. Paper-package copies match the newly generated source artifacts.
12. No test or locked-test cache is loaded by this workflow.

## Execution and validation

Use the repository-supported commands after inspecting their current CLI. At minimum, run the focused tests and then regenerate the final comparison, equal-average cost sweep, valid ablations, and paper package.

Do not use old command examples blindly. Print and record the exact commands actually run.

After generation, perform an explicit audit:

1. Search the regenerated deployable CSV, JSON, and paper tables for stale finalizer names.
2. Recompute or independently verify the SHA-256 values recorded in several representative rows, including the main sequential method, a fixed baseline, a cost-sweep point, and an ablation.
3. Confirm all reported rows use router-validation or permitted router-training selection data only.
4. Confirm the main sequential MAE is reproduced with the inference rule and finalizer stated in its row.
5. Confirm source artifacts and paper-package copies are synchronized.
6. Review the exact diff and `git status` before committing.

## Finish

If the cleanup is correct and reproducible:

1. Stage only the generator, test, regenerated artifact, and paper-table files belonging to this task.
2. Commit with a specific message such as `Regenerate COSTARTS artifacts with provenance`.
3. Push to `origin master` without force.
4. Report:
   - the problems confirmed;
   - files changed;
   - tests and commands run;
   - regenerated row counts;
   - finalizer and inference rule used by the main sequential result;
   - checkpoint and cache hashes;
   - whether any ablation remained blocked or was only a smoke diagnostic;
   - whether any deployable table still contains stale finalizer names;
   - commit hash and push result.

Do not work on the next sequential architecture improvement in this iteration. Artifact consistency must be resolved first.