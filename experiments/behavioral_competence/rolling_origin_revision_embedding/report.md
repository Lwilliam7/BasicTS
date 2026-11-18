# Rolling-Origin Revision Embedding

Strict validation-only mechanism study. No test cache or test split was loaded.

Final classification: `NEGATIVE_RESULT`.

## Summary

| Dataset | Context OOF R2 | Context+Revision OOF R2 | Residual OOF R2 | Context Val Route MAE | Context+Revision Val Route MAE | Verdict |
|---|---:|---:|---:|---:|---:|---|
| ETTm2 | 0.001468 | 0.073149 | -0.157214 | 0.161021 | 0.160651 | NEGATIVE |
| Traffic | 0.032520 | -0.038314 | -0.021563 | 0.269113 | 0.268781 | NEGATIVE |

## Answers

**1. Does revision behavior predict expert competence?** Mixed and insufficient. ContextPlusRevision improves ContextEmbed OOF R2 on ETTm2 by +0.071681, Traffic by -0.070834, so it helps ETTm2 but hurts Traffic.

**2. Does it add information beyond the learned context embedding?** No robustly. Additivity is positive on ETTm2 but negative on Traffic, so the fixed cross-dataset criterion fails.

**3. Can it predict ContextEmbed's residual errors?** No. The mandatory OOF residual diagnostic is negative on ETTm2 (R2 -0.157214), Traffic (R2 -0.021563).

**4. Is the signal expert-specific?** Not convincingly. The real revision model fails at least one wrong-expert or shuffled control on ETTm2, Traffic.

**5. Does it improve actual routing MAE?** Only by point estimate: ETTm2 delta -0.000370, Traffic delta -0.000332. This is not enough to override the failed competence/residual criteria.

**6. Does it work on both Traffic and ETTm2?** No. Both datasets would need PROMISING verdicts; observed verdicts are ETTm2=NEGATIVE, Traffic=NEGATIVE.

**7. Did every integrity check pass?** Yes. Every dataset integrity gate reports PASS.

**Overall decision** `NEGATIVE_RESULT` because ContextPlusRevision does not beat ContextEmbed on both datasets, the residual diagnostic is negative on both, and both-promising=False.

## Integrity

| Dataset | No test path loaded | Folds purged | Val target-invariant features | Checkpoints unchanged | Result |
|---|---:|---:|---:|---:|---|
| ETTm2 | True | True | True | True | PASS |
| Traffic | True | True | True | True | PASS |

## Fixed Method

- Revisions use real earlier forecast origins only: `F[t,k,h] - F[t-d,k,h+d]` for lags 1, 2, and 4.
- Large-variable datasets use a deterministic train-independent variable projection to preserve signed trajectories compactly.
- ContextEmbed is a learned encoder over current history and current expert forecasts, not the old 15 handcrafted passive features.
- Routing uses fixed rank weights `[0.5, 0.3333, 0.1667]`; no routing weights are tuned.
