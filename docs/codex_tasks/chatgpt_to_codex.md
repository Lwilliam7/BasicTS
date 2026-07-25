# ChatGPT to Codex Prompt Inbox

Status: active

This file is the shared task handoff between ChatGPT and the local Codex CLI.

Codex instruction:

1. Read this file completely whenever it changes.
2. Treat the newest task section as the active implementation request.
3. Inspect the repository before editing.
4. Preserve unrelated user changes and existing working routers.
5. Run relevant tests and report actual results.
6. Do not claim completion for work or tests that were not performed.
7. Update `docs/codex_tasks/chatgpt_to_codex_progress.md` after each major phase.
8. Do not edit or delete this prompt file.

## Active task

Implement a new, genuinely sequential version of COSTARTS for frozen time-series forecasting experts. Preserve the current COSTARTS implementation for reproducibility. Add an optional LLM meta-controller only after the numerical sequential router is working.

### Project context

- Repository: `Lwilliam7/BasicTS`
- Dataset: ETTh1
- History shape: `[B, 96, 7]`
- Forecast shape: `[B, 12, 7]`
- Frozen experts:
  - DLinear
  - PatchTST
  - iTransformer
  - TimesNet
  - ModernTCN
- Chronological splits:
  - expert training: 50%
  - expert validation: 10%
  - router training: 15%
  - router validation: 5%
  - final test: 20%
- The final test split must remain untouched until the method and hyperparameters are finalized.

### Known problems in the current implementation

The current `scripts/train_costarts_router.py` is not truly sequential. It:

- encodes the history once into one `[B, 64]` vector;
- predicts the complete expert order before any expert is run;
- predicts the stop depth from the same initial vector;
- does not consume the forecast returned by a queried expert;
- has no state updater;
- has no queried-expert mask;
- has no remaining-budget feature;
- teacher-forces the true oracle-best expert first;
- therefore learns degenerate immediate-stop targets when costs are zero;
- uses a predicted-error head whose current validation ranking quality is near random;
- has a mix head that is not trained when its loss weight is zero.

The new implementation must fix the decision process itself. Do not simply add an LLM in front of the current static router.

## Phase 1: inspect before editing

Inspect at least:

- `scripts/train_costarts_router.py`
- `scripts/router_experiment_config.py`
- `scripts/router_model_config.py`
- `scripts/chronological_expert_training.py`
- `scripts/router_diagnostics.py`
- COSTARTS cache generation and evaluation code
- RouterDC-related code and tests

Before changing code, write an inspection summary to:

`docs/codex_tasks/chatgpt_to_codex_progress.md`

The summary must state:

1. reusable existing utilities;
2. exact files that will be added or modified;
3. current cache tensor shapes and expert ordering;
4. leakage, scaling, masking, and sample-alignment risks;
5. how backward compatibility will be preserved.

Then proceed with implementation without waiting for another user message unless a real blocker exists.

## Phase 2: implement a true sequential router

Create a new logically named module, for example:

- `scripts/train_agentic_costarts_router.py`
- `scripts/agentic_costarts_router.py`

Do not overwrite the old COSTARTS class, training script, checkpoints, or result files.

### Required sequential loop

For each sample:

1. Encode the history.
2. Choose one unqueried expert.
3. Obtain that expert's cached prediction during router training or live prediction during deployment.
4. Encode the returned forecast.
5. Update the router state using the new forecast and trajectory information.
6. Choose another unqueried expert or `STOP`.
7. Repeat until `STOP` or a configurable maximum query count is reached.
8. Return either one queried expert or a sparse mixture over only queried experts.

The router decision after query 1 must be different from the initial decision path because it must consume the returned forecast.

### State contents

The state after step `t` must include at least:

- history representation;
- queried-expert mask `[B, M]`;
- encoded queried forecasts;
- current aggregate forecast `[B, 12, 7]`;
- disagreement among queried forecasts;
- query count;
- remaining query budget;
- optional normalized expert cost features.

Do not expose future targets, true expert errors, or oracle labels to the model forward pass.

### State updater

Use a compact recurrent or gated updater, such as:

- GRUCell;
- gated residual state update;
- another small justified recurrent mechanism.

Add a test or assertion proving the state changes after a forecast is queried.

### Unified action space

Use one action distribution over:

`M expert-query actions + 1 STOP action`

Requirements:

- mask experts that were already queried;
- do not allow `STOP` before the minimum required number of queries;
- avoid all-masked softmax failures;
- support any number and ordering of selected experts;
- force `STOP` when the maximum query count is reached.

Do not use one static query-order head plus a separate stop-depth head as the primary design.

### Forecast encoder

Encode each queried forecast `[B, 12, 7]` into a compact vector. Include expert identity so the same numerical forecast from different experts is distinguishable.

### Finalization modes

Support both:

1. `best_queried`
   - select one expert from the queried set;
2. `sparse_mix`
   - assign normalized weights only to queried experts;
   - combine their forecasts into `[B, 12, 7]`.

The sparse mixer must assign exactly zero weight to unqueried experts.

Use equal averaging over queried experts as a required baseline and safe fallback.

## Phase 3: build supervised subset-state targets

Use existing router-train and router-validation caches. Never use final-test targets for training or tuning.

Generate training states from diverse queried subsets, including:

- empty initial states;
- random one-expert states;
- random multi-expert states;
- states where the first query is not oracle-best;
- states where the best expert remains unqueried;
- states generated by the current learned policy after initial supervised training.

Do not train only on oracle-sorted trajectories.

### Counterfactual action utility

For each state and every legal unqueried expert action, compute an offline target utility using cached forecasts and router-training targets.

Primary finalizer for target construction:

- equal average of the currently queried forecasts;
- after adding candidate expert `m`, equal average of the enlarged queried set.

Define the candidate query utility as:

`marginal reduction in forecast MAE - normalized query cost`

Define `STOP` utility as the utility of the current queried subset with no additional cost.

Use the same finalizer for target construction and deployment by default. Add a configuration guard that rejects a mismatch unless an explicit override is enabled.

Normalize query costs relative to the router-training MAE improvement scale. Do not add raw latency numbers directly to normalized MAE.

### Training target

Convert action utilities to a soft target distribution with a configurable temperature.

Use a primary loss aligned with the actual action decision:

- cross-entropy or KL divergence between predicted action probabilities and soft utility targets.

Optional additions are allowed only when justified:

- pairwise action-ranking loss;
- final sparse forecast MAE loss when `sparse_mix` is enabled.

Do not keep several redundant heads unless an ablation shows they help.

## Phase 4: rollout-state training

First implement supervised random/subset-state training.

Then add one optional policy-state aggregation stage:

- scheduled policy-generated states; or
- a simple DAgger-style pass.

Do not implement reinforcement learning until the supervised sequential system is complete and tested.

## Phase 5: configuration

Extend the central configuration while preserving existing routers.

Add settings for at least:

- router type;
- selected experts;
- expert checkpoint paths;
- state dimension;
- history encoder dimension;
- forecast encoder dimension;
- maximum queries;
- minimum queries;
- finalizer mode;
- action target temperature;
- action loss weight;
- optional pairwise ranking loss weight;
- sparse forecast loss weight;
- cost coefficient;
- per-expert costs;
- subset-state sampling strategy;
- number of sampled subset states per sample;
- policy-generated state ratio;
- random seed;
- debug tensor-shape printing.

All new code must support any selected number of experts.

## Phase 6: optional LLM meta-controller

Implement this only after the numerical sequential router passes its tests.

The LLM must not run once per forecasting window.

Create a provider-agnostic, disabled-by-default meta-controller that runs periodically on aggregated causal diagnostics. It may adjust only bounded high-level policy settings.

Allowed inputs may include:

- recent expert residual summaries when labels have become causally available;
- expert utilization;
- routing entropy;
- average query count;
- latency summaries;
- missing-data rate;
- volatility and regime statistics;
- distribution-shift indicators;
- calibration summaries.

Allowed outputs may include only bounded settings such as:

- active expert subset;
- maximum query budget;
- cost coefficient;
- confidence threshold;
- fallback finalizer;
- sparse mixing enabled or disabled.

Requirements:

- disabled by default;
- deterministic fallback when unavailable;
- strict schema-validated JSON output;
- hard bounds on every output;
- no future-target access;
- no API keys committed to the repository;
- mock provider for tests;
- log every meta-controller decision;
- no arbitrary shell commands or code execution from LLM output.

## Phase 7: training pipeline

Implement:

- reproducible seeding;
- router training and validation;
- early stopping based only on router validation;
- checkpointing;
- gradient clipping;
- learning-rate scheduling if already used elsewhere;
- loss logging;
- action and stopping diagnostics;
- optional debug shape output.

Frozen-expert requirements:

- experts must remain in `eval()` mode;
- every expert parameter must have `requires_grad=False`;
- expert parameters must be excluded from every optimizer;
- assert no expert gradients exist after backward;
- cache generation must run experts under `torch.no_grad()`.

## Phase 8: evaluation

Evaluate all methods on identical chronological samples and cached predictions where applicable.

Required baselines:

- every individual expert;
- true best validation-selected fixed expert;
- equal average of all experts;
- validation-based weighted average;
- linear stacking;
- standard MLP selector;
- RouterDC hard selector;
- current COSTARTS;
- fixed top-2 shortlist with equal averaging;
- new sequential COSTARTS with `best_queried`;
- new sequential COSTARTS with `sparse_mix`;
- full per-window oracle;
- oracle within the selected top-k set.

Required metrics:

- MAE;
- MSE;
- regret to full oracle;
- action accuracy against utility-optimal action;
- oracle winner accuracy;
- top-2 oracle coverage;
- expert utilization;
- stop-step distribution;
- routing entropy;
- average experts executed;
- CPU latency;
- GPU latency when available;
- router parameter count;
- memory usage when practical.

For final comparisons, support at least five seeds and save confidence intervals.

Save results as CSV and JSON under a new logically named directory without overwriting prior COSTARTS results.

## Phase 9: ablations

Implement configuration switches without duplicating large code sections for:

- no forecast feedback;
- no queried-expert mask;
- no disagreement feature;
- no remaining-budget feature;
- separate stop head versus unified action;
- hard final selection versus sparse mixing;
- equal-average finalizer versus learned sparse finalizer;
- fixed top-2 versus adaptive stopping;
- no policy-generated states;
- different maximum query counts;
- different cost coefficients;
- LLM meta-controller off versus on.

## Phase 10: tests

Add focused tests for:

- tensor shapes;
- queried-expert masking;
- STOP legality;
- forced STOP at maximum query count;
- state changes after receiving a forecast;
- sparse weights sum to one over queried experts only;
- unqueried experts receive zero final weight;
- deterministic behavior under a fixed seed;
- no expert gradients;
- cache sample alignment;
- expert ordering alignment;
- no final-test split used in training;
- finalizer mismatch guard;
- mock LLM schema validation and bounded outputs.

Run every relevant test that is practical in the environment. Report actual commands and outcomes. Do not claim unrun tests passed.

## Required completion report

When finished, update:

`docs/codex_tasks/chatgpt_to_codex_progress.md`

Include:

1. inspection findings;
2. every changed and added file;
3. architecture and tensor shapes;
4. exact commands to train and evaluate;
5. tests run and actual outcomes;
6. unresolved blockers or assumptions;
7. direct comparison between old static COSTARTS and new sequential COSTARTS;
8. no novelty or performance claim without experimental evidence.

Do not stop after writing a plan. Continue through implementation and testing unless a concrete blocker prevents further work.