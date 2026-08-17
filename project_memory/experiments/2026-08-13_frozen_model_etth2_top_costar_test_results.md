# frozen-model ETTh2 Top COSTAR Test Audit

Date: 2026-08-13

Status: pre_test_frozen

## Purpose

After the preregistered final test evaluation was complete, replay top ETTh2 COSTAR-style candidates and references on the already-seen test split. These results are audit/hypothesis-generating only and must not be used to retune or replace the frozen model family.

## Artifacts

- `experiments/frozen_model_test_results/run_frozen_model_top_costar_etth2.py`
- `experiments/frozen_model_test_results/ETTH2_TOP_COSTAR_TEST_RESULTS.json`
- `experiments/frozen_model_test_results/etth2_top_costar_test_results.csv`
- `experiments/frozen_model_test_results/FROZEN_MODEL_ETTH2_TOP_COSTAR_TEST_REPORT.md`

## Results

| Method | Expert set | Test MAE | Test MSE | Validation MAE | Difference vs DLinear test | Protocol |
|---|---|---:|---:|---:|---:|---|
| Full adaptive train-selected fixed-3 final frozen | `DLinear+PatchTST+ModernTCN` | `0.297808` | `0.218612` | `0.276832` | `-0.003899` | preregistered final ETTh2 model |
| Fixed-2 reference | `DLinear+TimesNet` | `0.298398` | `0.221926` | `0.277652` | `-0.003310` | validation-ranked fixed reference |
| Fixed-3 validation-selected reference | `DLinear+TimesNet+ModernTCN` | `0.299169` | `0.223927` | `0.276644` | `-0.002539` | validation-selected reference only |
| Fixed-2 validation-selected reference | `DLinear+ModernTCN` | `0.299263` | `0.221853` | `0.275229` | `-0.002445` | validation-selected reference only |
| Full adaptive variable-size core | `DLinear` | `0.301093` | `0.222139` | `0.280470` | `-0.000615` | router-train variable-size selected core |
| Single expert anchor | `DLinear` | `0.301708` | `0.222694` | `0.280957` | `0.000000` | canonical validation-best single |
| Train-selected fixed-3 core | `DLinear+PatchTST+ModernTCN` | `0.304642` | `0.225185` | `0.280878` | `+0.002935` | router-train selected fixed-3 |

## Interpretation

The best ETTh2 test MAE in this frozen-model audit is the preregistered full frozen adaptive model: `0.297808`.

Unlike the ETTh1 frozen-model audit, the ETTh2 audit did not find a frozen-model method that beats the preregistered final model. The final ETTh2 model also beat the validation-selected `DLinear+ModernTCN` reference on test, despite that reference having better validation MAE.

Because these checks were run after final test metrics were already known, they confirm robustness of the frozen ETTh2 choice but do not create a new clean model-selection result.

## Decision

Keep the preregistered ETTh2 full adaptive model as the final clean ETTh2 COSTAR result:

- core: `DLinear+PatchTST+ModernTCN`
- method: frozen adaptive current-best architecture
- test MAE/MSE: `0.297808` / `0.218612`

Do not use the frozen-model ranking for new tuning.
