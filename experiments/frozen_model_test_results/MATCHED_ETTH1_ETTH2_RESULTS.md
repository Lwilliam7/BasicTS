# Matched ETTh1/ETTh2 Results

This table includes ETTh2 locked ETTh1-config replications where valid artifacts now exist. Rows labeled `locked_etth1_config_etth2_replication` were generated after the earlier final ETTh2 test evaluation and are not pre-test preregistrations.

| Method | ETTh1 Test MAE | ETTh2 Test MAE | ETTh2 Status | ETTh2 Note |
|---|---:|---:|---|---|
| MLP residual corrector | 0.326047 | 0.297254 | locked_etth1_config_etth2_replication | Locked ETTh1 configuration replicated on ETTh2; fitted on router_train only; evaluated after manifest. |
| Full adaptive model | 0.326395 | 0.297808 | pre_test_frozen | ETTh2 frozen full adaptive model; DLinear/ModernTCN duplicate specialists disabled. |
| Expanded DLinear only | 0.326437 | 0.297808 | pre_test_frozen | Not a distinct ETTh2 model: DLinear is already in the selected core, so the duplicate specialist is disabled. |
| Ridge residual corrector | 0.326448 | 0.297313 | locked_etth1_config_etth2_replication | Locked ETTh1 configuration replicated on ETTh2; fitted on router_train only; evaluated after manifest. |
| Expanded ModernTCN only | 0.326468 | 0.297808 | pre_test_frozen | Not a distinct ETTh2 model: ModernTCN is already in the selected core, so the duplicate specialist is disabled. |
| Horizon-variable hybrid | 0.326493 | 0.297808 | pre_test_frozen | ETTh2 horizon-variable hybrid analogue; same prediction as full model because duplicate specialists are disabled. |
| Chronological EMA hybrid | 0.326548 | 0.301689 | pre_test_frozen | ETTh2 chronological EMA analogue over train-selected core. |
| Oracle prototype residual | 0.326829 | 0.301185 | locked_etth1_config_etth2_replication | Locked ETTh1 configuration replicated on ETTh2; fitted on router_train only; evaluated after manifest. |
| Fixed-three core | 0.327128 | 0.304642 | pre_test_frozen | ETTh2 router-train selected fixed-three equal core. |
| Dynamic fixed-three, seed 7 | 0.329249 | 0.297398 | locked_etth1_config_etth2_replication | Locked ETTh1 configuration replicated on ETTh2; fitted on router_train only; evaluated after manifest. |
| Best single | 0.339080 | 0.301708 | pre_test_frozen | ETTh2 best single expert. |
