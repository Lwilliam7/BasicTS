# Published Baseline Test Audit

Date: 2026-08-18

Label: `post_hoc_comparative_audit`

## Purpose

Run a frozen test-only audit of the published-baseline comparison suite after ETTh1/ETTh2 test metrics had already been viewed. This is not a clean untouched final-test claim.

## Protocol

- Loaded frozen validation-selected configs from `experiments/published_baseline_comparisons/`.
- Wrote `experiments/published_baseline_test_audit/pre_test_audit_manifest.json` before loading test caches.
- Evaluated Equal fixed ensemble, Granger-Ramanathan, Bates-Granger, FAME adaptation, TimeRouter adaptation, Frozen COSTAR, Online COSTAR, Ridge residual, MLP residual, and OneNet/adaptation.
- Used CUDA for trainable FAME/TimeRouter fits.
- Preserved causal feedback rule for Online COSTAR and OneNet: `old_forecast_start + forecast_horizon <= current_forecast_start`.
- Did not tune or select any parameter from test results.

## Results

| Method | ETTh1 Test MAE | ETTh1 Test MSE | ETTh2 Test MAE | ETTh2 Test MSE |
|---|---:|---:|---:|---:|
| Equal fixed ensemble | `0.332001` | `0.270050` | `0.322330` | `0.249527` |
| Granger-Ramanathan | `0.340765` | `0.289594` | `0.298419` | `0.218160` |
| Bates-Granger | `0.327848` | `0.267809` | `0.296294` | `0.217423` |
| FAME adaptation | `0.331314` | `0.271990` | `0.298372` | `0.220674` |
| TimeRouter adaptation | `0.328178` | `0.267896` | `0.306324` | `0.228592` |
| Frozen COSTAR | `0.327175` | `0.267094` | `0.300574` | `0.220499` |
| Online COSTAR | `0.326408` | `0.267378` | `0.297808` | `0.218612` |
| Frozen COSTAR + Ridge residual | `0.326448` | `0.267452` | `0.296787` | `0.217713` |
| Frozen COSTAR + MLP residual | `0.326047` | `0.267322` | `0.297041` | `0.218149` |
| OneNet / adaptation | `0.330721` | `0.272812` | `0.407526` | `0.413704` |

## Interpretation

- ETTh1 best audited row: Frozen COSTAR + MLP residual, MAE `0.326047`.
- ETTh2 best audited row: Bates-Granger, MAE `0.296294`.
- Online COSTAR beat Equal fixed, Granger-Ramanathan, FAME, TimeRouter, Frozen COSTAR, and OneNet on ETTh1, but did not beat the Ridge/MLP residual audit rows or Bates-Granger on ETTh2.
- Because test results were already known before this audit, these results are comparative/provenance evidence only and do not supersede the final frozen-test record.

## Artifacts

- `experiments/published_baseline_test_audit/run_published_baseline_test_audit.py`
- `experiments/published_baseline_test_audit/PUBLISHED_BASELINE_TEST_AUDIT_RESULTS.json`
- `experiments/published_baseline_test_audit/PUBLISHED_BASELINE_TEST_AUDIT_REPORT.md`
- `experiments/published_baseline_test_audit/published_baseline_test_results.csv`
- `experiments/published_baseline_test_audit/published_baseline_test_comparison_table.csv`
- `experiments/published_baseline_test_audit/published_baseline_test_per_window_metrics.csv`

## Verification

- `python -m py_compile experiments\published_baseline_test_audit\run_published_baseline_test_audit.py`
- `python -m pytest tests\test_published_baseline_comparisons.py` could not run because `pytest` is not installed.
- Direct invocation of the existing published-baseline assertion functions passed.
- Generated audit artifact validation passed.
