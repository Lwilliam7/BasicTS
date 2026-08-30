# Rolling-Origin Revision Embedding

Date: 2026-08-30

Experiment directory: `experiments/behavioral_competence/rolling_origin_revision_embedding/`

## Question

Can the way a frozen forecasting expert revises predictions across nearby real forecast origins provide new instance-specific competence information beyond a strong learned context embedding?

This is distinct from `counterfactual_forecast_revision`: no hypothetical future observations are revealed and no expert is re-queried. The revision signal uses only original frozen forecasts produced at earlier real forecast origins.

## Protocol

- Datasets: `Traffic`, `ETTm2`.
- Splits: `router_train` for OOF training/analysis; `router_val` for a single final held-out evaluation.
- Test access: none.
- K=3 cores selected/reused through the existing router_train-only generalization registration path.
- Revision lags fixed before evaluation: `[1, 2, 4]`.
- Revision formula: `R[t,k,d,h] = F[t,k,h] - F[t-d,k,h+d]`, normalized by train-only per-variable std.
- Large-variable data used a deterministic, target-independent compact variable projection to preserve full signed lag/horizon trajectories without flattening all Traffic variables.
- Models:
  - `ContextEmbed`
  - `RevisionEmbed`
  - `ContextPlusRevision`
  - `ContextPlusWrongExpertRevision`
  - `ContextPlusShuffledRevision`
- Encoder: tiny fixed MLP, hidden dimension `32`, deterministic seed `20260830`.
- OOF: chronological walk-forward folds with horizon-12 purge; `max(train target end) <= min(eval origin)`.
- Routing: fixed rank weights `[0.5, 0.3333, 0.1667]`; no routing-weight tuning.
- Dependence checks: block bootstrap lengths `12/24/48` plus every-12th phase bootstrap.

## Results

Final classification: `NEGATIVE_RESULT`.

| Dataset | Context OOF R2 | Context+Revision OOF R2 | Residual OOF R2 | Context Val Route MAE | Context+Revision Val Route MAE | Verdict |
|---|---:|---:|---:|---:|---:|---|
| Traffic | `0.032520` | `-0.038314` | `-0.021563` | `0.269113` | `0.268781` | `NEGATIVE` |
| ETTm2 | `0.001468` | `0.073149` | `-0.157214` | `0.161021` | `0.160651` | `NEGATIVE` |

Traffic fails the primary additivity criterion: `ContextPlusRevision` underperforms `ContextEmbed` in OOF competence R2 by `-0.070834`.

ETTm2 shows a positive ContextPlusRevision OOF R2 gain (`+0.071681`) and a small router_val routing point gain (`-0.000370` MAE), but the mandatory residual diagnostic is strongly negative and the wrong-expert control has higher OOF R2 than the real expert-matched revision model.

Both datasets have small router_val routing point gains, but those do not rescue the hypothesis because the experiment was designed to require residual/additivity and expert-specific control evidence.

## Integrity

All integrity gates passed:

- no test cache/file loaded;
- frozen checkpoint hashes unchanged before/after;
- router_train OOF folds horizon-12 purged;
- router_train-to-router_val observability held;
- train-only standardization/scaling;
- router_val target-corruption feature invariance;
- finite features and predictions.

## Decision

Do not promote rolling-origin revision embeddings to router integration or test evaluation. The mechanism is interesting on ETTm2 by point estimate, but it does not provide robust, expert-specific competence information beyond the learned context embedding under the preregistered checks.

## Artifacts

- `experiments/behavioral_competence/rolling_origin_revision_embedding/run_rolling_origin_revision_embedding.py`
- `experiments/behavioral_competence/rolling_origin_revision_embedding/method_manifest.json`
- `experiments/behavioral_competence/rolling_origin_revision_embedding/validation_results.json`
- `experiments/behavioral_competence/rolling_origin_revision_embedding/integrity_checks.json`
- `experiments/behavioral_competence/rolling_origin_revision_embedding/dependence_tests.csv`
- `experiments/behavioral_competence/rolling_origin_revision_embedding/report.md`
