# Locked Multi-Dataset Replication

## Scope

- Requested datasets: ETTh2, ETTm1, ETTm2, Weather, Electricity.
- Available non-test frozen expert caches found: ETTh2 only.
- No test cache was loaded.
- ETTh2 replication is limited because the full ETTh1 static neural winner artifact is not available for ETTh2.

## Results

- `ETTh2`: completed limited available-cache replication.
  - `equal_fixed3_available_baseline` MAE/MSE: `0.098339` / `0.038581`.
  - `single_DLinear` MAE/MSE: `0.069131` / `0.015874`, improvement `0.029208`, CI `[-0.030731, -0.027656]`.
  - `single_PatchTST` MAE/MSE: `0.090272` / `0.029436`, improvement `0.008067`, CI `[-0.008898, -0.007269]`.
  - `single_iTransformer` MAE/MSE: `0.171211` / `0.139716`, improvement `-0.072872`, CI `[0.071357, 0.074471]`.
  - `single_TimesNet` MAE/MSE: `0.071868` / `0.016389`, improvement `0.026471`, CI `[-0.027988, -0.024903]`.
  - `single_ModernTCN` MAE/MSE: `0.069838` / `0.015480`, improvement `0.028501`, CI `[-0.029986, -0.026950]`.
  - `locked_etth1_expanded_both_limited` MAE/MSE: `0.093369` / `0.034170`, improvement `0.004970`, CI `[-0.005121, -0.004820]`.
  - `selected_predefined_limited` MAE/MSE: `0.093239` / `0.034157`, improvement `0.005100`, CI `[-0.005253, -0.004949]`.
- `ETTm1`: `missing_existing_expert_caches`.
- `ETTm2`: `missing_existing_expert_caches`.
- `Weather`: `missing_existing_expert_caches`.
- `Electricity`: `missing_existing_expert_caches`.

## Reproduce

```powershell
python experiments\multidataset_replication_costar\run_locked_replication.py
```
