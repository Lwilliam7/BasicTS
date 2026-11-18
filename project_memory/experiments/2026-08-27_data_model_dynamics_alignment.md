# Data-Model Dynamics Alignment Mechanism Test

Date: 2026-08-27

Status: completed validation-only mechanism test.

Artifacts:

- `experiments/data_model_dynamics_alignment/report.md`
- `experiments/data_model_dynamics_alignment/results.json`
- `experiments/data_model_dynamics_alignment/manifest.json`
- `experiments/data_model_dynamics_alignment/alignment_features.pt`
- `experiments/data_model_dynamics_alignment/competence_metrics.csv`
- `experiments/data_model_dynamics_alignment/routing_metrics.csv`
- `experiments/data_model_dynamics_alignment/dependence_aware_stats.csv`
- `experiments/data_model_dynamics_alignment/integrity_checks.json`
- `experiments/data_model_dynamics_alignment/per_window_scores/`

Question:

Does agreement between observed local dynamics in a raw history window and a frozen forecaster's implied local dynamics predict upcoming expert error?

Frozen protocol:

- Validation only; no test cache loaded or scored.
- Datasets: `ETTh1`, `ETTh2`, `ETTm1`, `Weather`, `Electricity`.
- Expert cores and loaders reused from `experiments.frozen_hv_costar.run_frozen_hv_costar.LOADERS`.
- Raw histories reused via `raw_history_cache()`, including the ETTh2 convention.
- Router-train checkpoint provenance reused: walk-forward datasets use `block_a` for `block_b_oos`, `block_ab` for `block_c_oos`, and `final_60` for router-val; ETTh2 uses the existing single OOS checkpoint path.
- Local PCA dimension `d=min(4,F)`, ridge VAR(1) lambda `1e-2`, horizons `1..12`, normalized Frobenius mismatch.
- Competence target: `compute_excess_loss()` = expert MAE minus equal-ensemble MAE.
- Matched scorer: existing `CompetenceScorer`, same seed and chronological train/internal-val split across Passive and controls; router-val targets withheld until final scoring.
- Decision rule: existing fixed-rank rule.

Main classification:

- `WEAK_OR_AMBIGUOUS`.
- Only `Weather` passed all predeclared criteria A-E.
- `ETTh2` had a strong Passive+Align competence/routing improvement but failed the direct-alignment-positive criterion because direct `D_align` Spearman with excess was negative.
- `Electricity` improved Passive but did not beat J-magnitude or VAR-closeness controls consistently.
- `ETTm1` had a block-24 significant routing regression for Passive+Align vs Passive.

Competence/routing highlights:

| Dataset | Direct `D_align` Spearman | Passive pair | Passive+Align pair | Passive MAE | Passive+Align MAE | Residual R2 | Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| ETTh1 | `0.1692` | `0.5727` | `0.5495` | `0.367102` | `0.367938` | `0.0033` | no |
| ETTh2 | `-0.1107` | `0.5432` | `0.7064` | `0.280969` | `0.276910` | `-0.3828` | no |
| ETTm1 | `0.0291` | `0.5526` | `0.5495` | `0.249197` | `0.249602` | `-0.0016` | no |
| Weather | `0.2107` | `0.5996` | `0.6210` | `0.159772` | `0.159470` | `-0.0038` | yes |
| Electricity | `0.3045` | `0.7570` | `0.7705` | `0.215616` | `0.215202` | `0.0090` | no |

Dependence-aware block-24 notes:

- Passive+Align vs Passive improved significantly on `ETTh2`, `Weather`, and `Electricity`.
- Passive+Align regressed significantly on `ETTm1`.
- `ETTh1` was not significant.
- Against controls, only Weather consistently beat shuffled and VAR-closeness under block-24; ETTm1 significantly lost to shuffled.

Integrity:

- Checkpoint hashes unchanged on all five datasets.
- Frozen expert parameter fingerprints unchanged on all five datasets.
- Reproduction checks passed on all five datasets.
- JVP repeatability max absolute difference was `0.0` on all five datasets.
- Target corruption left passive/alignment features unchanged.
- Router-val targets were not used during fitting.
- Test data accessed: no.
- All model features were finite. ETTm1 recorded `438` nonfinite condition-number diagnostics, but these were numerical diagnostics only and were not used by any scorer/control feature.

Interpretation:

The data-model dynamics alignment idea has some signal, especially on `ETTh2`, `Weather`, and `Electricity` routing MAE, but it is not a robust strong-go mechanism under the preregistered criteria. The result is ambiguous rather than a clear integration candidate.

