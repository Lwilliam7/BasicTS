# Repository Working Instructions

Before starting COSTAR-TS research work:

1. Read `project_memory/CURRENT_STATE.md`.
2. Read `project_memory/DECISIONS.md`.
3. Read `project_memory/TODO.md`.
4. Read `project_memory/ARCHITECTURE.md` when modifying models, routers, caches, or evaluation.
5. Consult `project_memory/EXPERIMENTS.md` and detailed logs under `project_memory/experiments/` only when relevant.

After completing a significant experiment:

1. Save a detailed experiment record under `project_memory/experiments/`.
2. Update `project_memory/EXPERIMENTS.md`.
3. Update `project_memory/CURRENT_STATE.md` if the best result, hypothesis, split, or project direction changed.
4. Update `project_memory/DECISIONS.md` if the experiment supports a durable conclusion.
5. Update `project_memory/TODO.md`.

Rules:

- Never silently replace previous results.
- Never delete negative experiments simply because they failed.
- Clearly mark hypotheses and untested ideas.
- Do not use the untouched ETTh1 test split unless the user explicitly authorizes final test evaluation.
- Prefer existing frozen forecast caches and validation protocols for comparable COSTAR-TS work.
- If result files conflict, investigate and document the discrepancy instead of choosing arbitrarily.
