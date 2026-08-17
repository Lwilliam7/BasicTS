# Locked ETTh1-Config ETTh2 Replication

Label: `locked_etth1_config_etth2_replication`.

These ETTh2 artifacts were created after the earlier final ETTh2 test evaluation. They port the locked ETTh1 method configurations, fit on ETTh2 `router_train` only, save a manifest, and then evaluate ETTh2 test once. They are not described as frozen before the already-completed ETTh2 final test evaluation.

## ETTh2 Results

| Method | Val MAE | Val MSE | Test MAE | Test MSE | Gain vs DLinear test | Gain vs ETTh2 full adaptive test |
|---|---:|---:|---:|---:|---:|---:|
| Ridge residual corrector | 0.276038 | 0.166560 | 0.297313 | 0.218187 | +0.004394 | +0.000495 |
| MLP residual corrector | 0.276129 | 0.166645 | 0.297254 | 0.218303 | +0.004453 | +0.000554 |
| Oracle prototype residual | 0.276404 | 0.167296 | 0.301185 | 0.222719 | +0.000522 | -0.003377 |
| Dynamic fixed-three, seed 7 | 0.275379 | 0.166028 | 0.297398 | 0.218294 | +0.004310 | +0.000410 |

## Matched ETTh1 vs ETTh2

| Method | ETTh1 Test MAE | ETTh1 Val MAE | ETTh2 Test MAE | ETTh2 Val MAE |
|---|---:|---:|---:|---:|
| Ridge residual corrector | 0.326448 | 0.363301 | 0.297313 | 0.276038 |
| MLP residual corrector | 0.326047 | 0.363318 | 0.297254 | 0.276129 |
| Oracle prototype residual | 0.326829 | 0.366028 | 0.301185 | 0.276404 |
| Dynamic fixed-three, seed 7 | 0.329249 | 0.365985 | 0.297398 | 0.275379 |

## Checks

- Router-train fitting only: passed.
- ETTh2 `router_val` was not used for training, feature selection, or checkpoint selection.
- Test cache loaded only after `manifest_before_test.json` was written.
- Causal residual updates enforce `old_start + horizon <= current_start`.
- Long-history summaries use data before each forecast start; pre-test manifest used train+val prefix only.
- Checkpoint/config hashes are recorded in the manifest.

## Reproduce

```powershell
python experiments\locked_etth1_config_etth2_replication\run_locked_etth1_config_etth2_replication.py --phase all --device cuda
```
