# Selection criteria

Score each criterion from 0 to 4. Maximum 24. These scores assess proposal quality, not forecasting performance.

1. Scientific relevance. Covers strong fixed combinations, feature-conditioned combinations, and a cell-level alternative that can challenge CellShrink's actual claims.
2. Fidelity and evidence. Uses primary sources and repository facts. Separates full methods from adapted components. Preserves existing exclusions and identifies missing evidence.
3. Comparison fairness. Uses the same forecasts, history, origins, normalization, training labels, and declared selection budgets. Separates offline, delayed-feedback, and joint-training settings.
4. Honest selection. Keeps strong baselines even when CellShrink loses. Does not select methods or configurations using evaluation or test outcomes. Preserves development-only status and original readings.
5. Feasibility. Reuses existing runners, prioritizes remaining work, defines readiness criteria, and avoids an unnecessary new model zoo.
6. Reusability and clarity. Provides an executable-in-principle arena process, a concrete roster with roles and eligibility, and a short explanation the user can understand.

Hard rejection: recommends evaluating a locked test block now, promotes a withheld adaptation as a verified full method, presents invented performance or cost figures as measured, or drops a valid baseline because it beats CellShrink.
