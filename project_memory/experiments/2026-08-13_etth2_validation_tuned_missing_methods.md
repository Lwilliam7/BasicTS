# ETTh2 Validation-Tuned Missing Methods

Ran small ETTh2 router-validation-tuned sweeps for MLP residual, ridge residual, oracle prototype residual, and dynamic fixed-three. These are labeled `etth2_validation_tuned` and are not pre-test preregistered results.

| Method | Val MAE | Test MAE | Test MSE | Diff vs locked |
|---|---:|---:|---:|---:|
| Ridge residual corrector | `0.275036` | `0.296787` | `0.217713` | `-0.000526` |
| MLP residual corrector | `0.275643` | `0.297041` | `0.218149` | `-0.000213` |
| Oracle prototype residual | `0.274829` | `0.298475` | `0.219894` | `-0.002710` |
| Dynamic fixed-three | `0.274746` | `0.298079` | `0.219521` | `+0.000681` |

Artifacts:

- `experiments/etth2_validation_tuned_missing_methods/final_report.json`
- `experiments/etth2_validation_tuned_missing_methods/ETTH2_VALIDATION_TUNED_MISSING_METHODS_REPORT.md`
