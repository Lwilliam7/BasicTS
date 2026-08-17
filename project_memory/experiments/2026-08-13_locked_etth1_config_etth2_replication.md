# Locked ETTh1-Config ETTh2 Replication

Created ETTh2 counterparts for ETTh1-only residual/prototype/dynamic methods using canonical ETTh2 splits and the train-selected core `DLinear+PatchTST+ModernTCN`.

Important interpretation: these runs happened after the earlier final ETTh2 test evaluation and are labeled `locked_etth1_config_etth2_replication`, not pre-test preregistered final results.

| Method | Val MAE | Test MAE | Test MSE | Gain vs ETTh2 full adaptive test |
|---|---:|---:|---:|---:|
| Ridge residual corrector | `0.276038` | `0.297313` | `0.218187` | `+0.000495` |
| MLP residual corrector | `0.276129` | `0.297254` | `0.218303` | `+0.000554` |
| Oracle prototype residual | `0.276404` | `0.301185` | `0.222719` | `-0.003377` |
| Dynamic fixed-three, seed 7 | `0.275379` | `0.297398` | `0.218294` | `+0.000410` |

Artifacts:

- `experiments\locked_etth1_config_etth2_replication\manifest_before_test.json`
- `experiments\locked_etth1_config_etth2_replication\test_results.csv`
- `experiments\locked_etth1_config_etth2_replication\LOCKED_ETTH1_CONFIG_ETTH2_REPLICATION_REPORT.md`
