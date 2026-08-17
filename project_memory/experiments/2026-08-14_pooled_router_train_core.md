# Pooled Router-Train Core Sensitivity

Date: 2026-08-14

Label: `post_test_pooled_core_sensitivity`

This experiment was run after final ETTh1/ETTh2 test metrics had already been viewed. It is a post-test sensitivity check, not a preregistered replacement for the frozen final models.

## Goal

Test the user's proposed simpler core-selection rule:

1. Load existing frozen expert forecast caches.
2. Enumerate every three-expert combination.
3. Select the lowest pooled MAE over all valid `router_train` windows, using MSE only as a tie-breaker.
4. Freeze that core, then evaluate fixed-core and full adaptive variants on `router_val` and test.

No expert was retrained. No adaptive hyperparameters or seed lists were changed.

## Protocol

- ETTh1 train cache: `cache/costarts_walkforward/router_train_20_60_cache.pt`
- ETTh1 validation cache: `cache/costarts_walkforward/router_val_60_80_cache.pt`
- ETTh1 test cache: `experiments/final_test_evaluation/generated/caches/ETTh1/test_80_100_cache.pt`
- ETTh2 train cache: `cache/costarts_fresh/ETTh2_96_12/router_train_cache.pt`
- ETTh2 validation cache: `cache/costarts_fresh/ETTh2_96_12/router_val_cache.pt`
- ETTh2 test cache: `experiments/final_test_evaluation/generated/caches/ETTh2/locked_test_cache_v2.pt`

ETTh1 used the existing ETTh1 scaler behavior from the walk-forward DLinear normalizer. ETTh2 used the canonical raw/cache-scale protocol with `std=ones` and no inverse transform.

## Selected Cores

| Dataset | Pooled-selected core | Router-train MAE | Router-train MSE | Existing fold-selected core |
|---|---|---:|---:|---|
| ETTh1 | `PatchTST+iTransformer+TimesNet` | `0.342026` | `0.253654` | `PatchTST+iTransformer+TimesNet` |
| ETTh2 | `DLinear+TimesNet+ModernTCN` | `0.280987` | `0.176914` | `DLinear+PatchTST+ModernTCN` |

## Results

| Dataset | Method | Split | Expert set | MAE | MSE | Seed MAE mean/std |
|---|---|---|---|---:|---:|---:|
| ETTh1 | pooled fixed core | router_train | `PatchTST+iTransformer+TimesNet` | `0.342026` | `0.253654` | n/a |
| ETTh1 | pooled fixed core | router_val | `PatchTST+iTransformer+TimesNet` | `0.367265` | `0.310530` | n/a |
| ETTh1 | pooled fixed core | test | `PatchTST+iTransformer+TimesNet` | `0.327128` | `0.266583` | n/a |
| ETTh1 | pooled full adaptive | router_val | `PatchTST+iTransformer+TimesNet` | `0.363111` | `0.306056` | `0.363112 / 0.000013` |
| ETTh1 | pooled full adaptive | test | `PatchTST+iTransformer+TimesNet` | `0.326393` | `0.267506` | `0.326395 / 0.000021` |
| ETTh2 | pooled fixed core | router_train | `DLinear+TimesNet+ModernTCN` | `0.280987` | `0.176914` | n/a |
| ETTh2 | pooled fixed core | router_val | `DLinear+TimesNet+ModernTCN` | `0.276644` | `0.166932` | n/a |
| ETTh2 | pooled fixed core | test | `DLinear+TimesNet+ModernTCN` | `0.299169` | `0.223927` | n/a |
| ETTh2 | pooled full adaptive | router_val | `DLinear+TimesNet+ModernTCN` | `0.275602` | `0.166460` | `0.275602 / 0.000000` |
| ETTh2 | pooled full adaptive | test | `DLinear+TimesNet+ModernTCN` | `0.295829` | `0.219681` | `0.295829 / 0.000000` |

## Comparison

- ETTh1: pooled selection chose the same fixed-three core as chronological fold selection. Test metrics are effectively unchanged from the existing full adaptive model: `0.326393` vs existing `0.326395`.
- ETTh2: pooled selection chose `DLinear+TimesNet+ModernTCN`, not the fold-selected `DLinear+PatchTST+ModernTCN`.
- ETTh2 pooled full adaptive test MAE `0.295829` beats the existing fold-selected full adaptive test MAE `0.297808` by `0.001979`.
- ETTh2 pooled fixed core test MAE `0.299169` beats the existing fold-selected fixed core test MAE `0.304642` by `0.005473`.

Because this was run after prior final-test evaluation, the ETTh2 improvement should be treated as sensitivity evidence, not a new clean confirmatory result.

## Artifacts

- Script: `experiments/pooled_router_train_core/run_pooled_router_train_core.py`
- Freeze manifest: `experiments/pooled_router_train_core/frozen_config_before_validation_and_test.json`
- Checkpoint manifest: `experiments/pooled_router_train_core/checkpoint_manifest.json`
- All triple ranking: `experiments/pooled_router_train_core/pooled_router_train_all_triples.csv`
- Results CSV: `experiments/pooled_router_train_core/pooled_core_results.csv`
- Per-seed CSV: `experiments/pooled_router_train_core/pooled_core_adaptive_per_seed.csv`
- Comparison CSV: `experiments/pooled_router_train_core/pooled_core_comparisons.csv`
- JSON: `experiments/pooled_router_train_core/POOLED_ROUTER_TRAIN_CORE_RESULTS.json`
- Report: `experiments/pooled_router_train_core/POOLED_ROUTER_TRAIN_CORE_REPORT.md`
- Tests: `tests/test_pooled_router_train_core.py`

## Verification

- `python -m py_compile experiments\pooled_router_train_core\run_pooled_router_train_core.py tests\test_pooled_router_train_core.py` passed.
- `python -m pytest tests\test_pooled_router_train_core.py` could not run because `pytest` is not installed in the active environment.
- Direct in-process checks for pooled ranking and the pre-freeze test-path guard passed.
- The experiment command selected `cuda` and completed successfully.

## Reproduce

```powershell
python experiments\pooled_router_train_core\run_pooled_router_train_core.py
```
