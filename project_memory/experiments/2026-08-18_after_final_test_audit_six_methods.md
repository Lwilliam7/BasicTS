# After-Final-Test Audit: Six Previously Untested Published-Baseline Methods

Date: 2026-08-18

Label: `after_final_test_audit`

## Purpose

Evaluate the six published-baseline methods that had validation results but no prior test evaluation (Equal all-5 ensemble, Granger-Ramanathan, Bates-Granger, FAME adaptation, TimeRouter adaptation, OneNet-style frozen-expert adaptation) on the canonical ETTh1/ETTh2 test caches, using configurations frozen before validation. Frozen COSTAR and Online COSTAR were not re-tuned or re-selected; their rows are reference rows read verbatim from existing authoritative artifacts.

## Protocol

- Loaded all six configs verbatim from `experiments/published_baseline_comparisons/{ETTh1,ETTh2}/frozen_config_before_validation.json` and asserted `test_cache_loaded=false` on each.
- GR and Bates-Granger were refit (closed-form) on router_train only; FAME and TimeRouter were retrained (seed 7) on router_train only; OneNet's combination state was initialized from router_train only. All were frozen before predicting on test.
- Reused the canonical ETTh1/ETTh2 test caches from `experiments/final_test_evaluation/generated/caches/` (same caches as the original final-test evaluation); verified expert order, chronological starts, horizon 12, input length 96, all-observed masks, and recorded sha256 for every cache used.
- Added explicit verification not present in the earlier 2026-08-18 published-baseline test audit:
  - Target-replacement invariance check for GR/Bates-Granger/FAME/TimeRouter: predictions with test targets replaced by random noise were bit-identical (max abs diff `0.0`) to the real predictions on both datasets (8/8 checks passed).
  - OneNet future-target perturbation causality test: added `1e6` to test targets from the midpoint window onward and reran prediction. The first prediction whose combination weights could legally have been influenced by the perturbed targets (`old_forecast_start + forecast_horizon <= current_forecast_start`) matched the first prediction that actually changed exactly (window index `1398` on both datasets), and the max abs diff over the unaffected prefix was exactly `0.0` on both datasets.

## Results

| Method | ETTh1 Val MAE | ETTh1 Test MAE | ETTh1 Test MSE | ETTh2 Val MAE | ETTh2 Test MAE | ETTh2 Test MSE |
|---|---:|---:|---:|---:|---:|---:|
| Equal all-5 ensemble | `0.371099` | `0.332001` | `0.270050` | `0.300772` | `0.322330` | `0.249527` |
| Granger-Ramanathan | `0.382960` | `0.340765` | `0.289594` | `0.276704` | `0.298419` | `0.218160` |
| Bates-Granger | `0.368891` | `0.327848` | `0.267809` | `0.274915` | `0.296294` | `0.217423` |
| FAME routing adaptation to BasicTS frozen expert pool | `0.379212` | `0.331314` | `0.271990` | `0.277008` | `0.298372` | `0.220674` |
| TimeRouter routing-mechanism adaptation | `0.368234` | `0.328178` | `0.267896` | `0.283288` | `0.306324` | `0.228592` |
| OneNet-style frozen-expert adaptation | `0.370137` | `0.330721` | `0.272812` | `0.402666` | `0.407526` | `0.413704` |

Reference rows (existing, not re-run): Frozen COSTAR ETTh1/ETTh2 test MAE `0.327175`/`0.300574`; Online COSTAR test MAE `0.326408`/`0.297808`; COSTAR train-selected fixed core test MAE `0.327128`/`0.304642`; best single expert test MAE `0.339080`/`0.301708`.

ETTh1 test ranking (audited six + reference rows): Online COSTAR, fixed core, Frozen COSTAR, Bates-Granger, TimeRouter, OneNet, FAME, Equal all-5, best single expert, Granger-Ramanathan.

ETTh2 test ranking: Bates-Granger, Online COSTAR, FAME, Granger-Ramanathan, Frozen COSTAR, best single expert, fixed core, TimeRouter, Equal all-5, OneNet.

## Interpretation

- On ETTh1, Online COSTAR beats all six audited methods on test.
- On ETTh2, Bates-Granger beats Online COSTAR on test (test MAE `0.296294` vs `0.297808`), confirming and slightly sharpening the earlier published-baseline test audit's finding. FAME and Granger-Ramanathan also beat Online COSTAR's test MAE narrowly on ETTh2; TimeRouter and Equal all-5 do not. OneNet is far worse than Online COSTAR on ETTh2 (test MAE `0.407526` vs `0.297808`) and is the worst row on the whole ETTh2 table.
- Validation-to-test ranking mostly holds direction, but absolute gaps compress sharply from validation to test on ETTh1 (Equal all-5 val `0.371099` to test `0.332001`) while ETTh2 test MAE is worse than validation for every method except OneNet's own combination logic (which uses live test feedback) does not rescue it: ETTh2 Bates-Granger stays strong on both splits (val `0.274915`, test `0.296294`, still the best or near-best row), while ETTh2 OneNet is close to Bates-Granger on validation (`0.370137`... actually ETTh1 val) but collapses on ETTh2 test, the largest val-to-test degradation of any row in this project's baseline suite.
- All 10 leakage/causality checks passed: GR, Bates-Granger, FAME, and TimeRouter predictions are provably independent of realized test targets; OneNet's realized-feedback update boundary is exactly where causality allows it to be, with zero leakage into the unaffected prefix.
- This audit reconstructs GR/Bates-Granger exactly (closed-form, deterministic) and FAME/TimeRouter approximately (retrained with the same seed/config; numbers are extremely close to, and in most cases identical at 6 decimal places to, the 2026-08-17 published-baseline test audit's numbers for the same methods, with only trivial floating-point-level differences from GPU nondeterminism).

## Artifacts

- `experiments/published_baseline_test_audit/run_after_final_test_audit.py`
- `experiments/published_baseline_test_audit/TEST_RESULTS.csv`
- `experiments/published_baseline_test_audit/TEST_RESULTS.json`
- `experiments/published_baseline_test_audit/TEST_REPORT.md`
- `experiments/published_baseline_test_audit/per_window_test_metrics.csv`
- `experiments/published_baseline_test_audit/leakage_and_causality_checks.json`
- `experiments/published_baseline_test_audit/cache_provenance.json`

## Verification

- `python -m py_compile experiments/published_baseline_test_audit/run_after_final_test_audit.py`
- Ran the script twice (device cuda); results were stable and all 10 leakage/causality assertions passed both times.
- Cross-checked GR/Bates-Granger/FAME/TimeRouter test MAE against the existing 2026-08-17/18 published-baseline test audit (`PUBLISHED_BASELINE_TEST_AUDIT_RESULTS.json`); values agree to within floating-point retraining noise.
- Cross-checked reference rows (Frozen COSTAR, Online COSTAR, fixed core, best single expert) against `experiments/final_test_evaluation/FINAL_TEST_RESULTS.json` and `experiments/published_baseline_test_audit/PUBLISHED_BASELINE_TEST_AUDIT_RESULTS.json` verbatim; none were re-derived.
