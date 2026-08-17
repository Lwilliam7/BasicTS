# ETTh2 Validation-Tuned Missing Methods

Label: `etth2_validation_tuned`.

These runs use ETTh2 router-validation for hyperparameter and checkpoint selection. They are not preregistered and are not untouched-test confirmation.

## Declared Search Spaces

### ridge_residual_corrector
1. `{'ridge': 1.0, 'alpha': 0.1, 'clip_multiple': 0.25, 'feature_set': 'full'}`
2. `{'ridge': 1.0, 'alpha': 0.05, 'clip_multiple': 0.25, 'feature_set': 'full'}`
3. `{'ridge': 10.0, 'alpha': 0.1, 'clip_multiple': 0.25, 'feature_set': 'full'}`
4. `{'ridge': 1.0, 'alpha': 0.1, 'clip_multiple': 0.5, 'feature_set': 'full'}`
5. `{'ridge': 1.0, 'alpha': 0.25, 'clip_multiple': 0.25, 'feature_set': 'full'}`

### mlp_residual_corrector
1. `{'seed': 7, 'hidden': 64, 'lr': 0.0003, 'weight_decay': 0.01, 'alpha': 0.1, 'clip_multiple': 0.25, 'epochs': 40, 'patience': 6}`
2. `{'seed': 7, 'hidden': 64, 'lr': 0.0003, 'weight_decay': 0.01, 'alpha': 0.05, 'clip_multiple': 0.25, 'epochs': 40, 'patience': 6}`
3. `{'seed': 7, 'hidden': 64, 'lr': 0.0003, 'weight_decay': 0.01, 'alpha': 0.1, 'clip_multiple': 0.5, 'epochs': 40, 'patience': 6}`
4. `{'seed': 7, 'hidden': 64, 'lr': 0.0003, 'weight_decay': 0.03, 'alpha': 0.1, 'clip_multiple': 0.25, 'epochs': 40, 'patience': 6}`
5. `{'seed': 7, 'hidden': 32, 'lr': 0.0003, 'weight_decay': 0.01, 'alpha': 0.1, 'clip_multiple': 0.25, 'epochs': 40, 'patience': 6}`

### oracle_prototype_residual
1. `{'teacher_lambda': 0.01, 'num_prototypes': 16, 'residual_scale': 0.3, 'residual_weight': 0.001, 'epochs': 10}`
2. `{'teacher_lambda': 0.01, 'num_prototypes': 8, 'residual_scale': 0.3, 'residual_weight': 0.001, 'epochs': 10}`
3. `{'teacher_lambda': 0.01, 'num_prototypes': 32, 'residual_scale': 0.3, 'residual_weight': 0.001, 'epochs': 10}`
4. `{'teacher_lambda': 0.01, 'num_prototypes': 16, 'residual_scale': 0.3, 'residual_weight': 0.01, 'epochs': 10}`
5. `{'teacher_lambda': 0.001, 'num_prototypes': 16, 'residual_scale': 0.3, 'residual_weight': 0.001, 'epochs': 10}`

### dynamic_fixed_three
1. `{'seed': 7, 'batch_size': 512, 'epochs': 2, 'learning_rate': 0.001, 'weight_decay': 0.0, 'grad_clip_norm': 1.0, 'entropy_weight': 0.0, 'embedding_dim': 64, 'hidden_dim': 64, 'ablation': 'full'}`
2. `{'seed': 7, 'batch_size': 512, 'epochs': 5, 'learning_rate': 0.001, 'weight_decay': 0.0, 'grad_clip_norm': 1.0, 'entropy_weight': 0.0, 'embedding_dim': 64, 'hidden_dim': 64, 'ablation': 'full'}`
3. `{'seed': 7, 'batch_size': 512, 'epochs': 5, 'learning_rate': 0.0005, 'weight_decay': 0.0, 'grad_clip_norm': 1.0, 'entropy_weight': 0.0, 'embedding_dim': 64, 'hidden_dim': 64, 'ablation': 'full'}`
4. `{'seed': 7, 'batch_size': 512, 'epochs': 5, 'learning_rate': 0.001, 'weight_decay': 0.0001, 'grad_clip_norm': 1.0, 'entropy_weight': 0.0, 'embedding_dim': 64, 'hidden_dim': 64, 'ablation': 'full'}`
5. `{'seed': 7, 'batch_size': 512, 'epochs': 5, 'learning_rate': 0.001, 'weight_decay': 0.0, 'grad_clip_norm': 1.0, 'entropy_weight': 0.001, 'embedding_dim': 64, 'hidden_dim': 64, 'ablation': 'full'}`

## Complete Validation Sweep

| Method | Config | Seed | Val MAE | Val MSE |
|---|---|---:|---:|---:|
| Ridge residual corrector | `ridge_alpha0p1_clip_multiple0p25_feature_setfull_ridge1` | mean | 0.276038 | 0.166560 |
| Ridge residual corrector | `ridge_alpha0p05_clip_multiple0p25_feature_setfull_ridge1` | mean | 0.276423 | 0.166911 |
| Ridge residual corrector | `ridge_alpha0p1_clip_multiple0p25_feature_setfull_ridge10` | mean | 0.276035 | 0.166560 |
| Ridge residual corrector | `ridge_alpha0p1_clip_multiple0p5_feature_setfull_ridge1` | mean | 0.275883 | 0.166489 |
| Ridge residual corrector | `ridge_alpha0p25_clip_multiple0p25_feature_setfull_ridge1` | mean | 0.275036 | 0.165619 |
| MLP residual corrector | `mlp_alpha0p1_clip_multiple0p25_epochs40_hidden64_lr0p0003_patience6_weight_decay0p01` | 7 | 0.275892 | 0.166390 |
| MLP residual corrector | `mlp_alpha0p1_clip_multiple0p25_epochs40_hidden64_lr0p0003_patience6_weight_decay0p01` | 11 | 0.275947 | 0.166489 |
| MLP residual corrector | `mlp_alpha0p1_clip_multiple0p25_epochs40_hidden64_lr0p0003_patience6_weight_decay0p01` | 13 | 0.275938 | 0.166348 |
| MLP residual corrector | `mlp_alpha0p1_clip_multiple0p25_epochs40_hidden64_lr0p0003_patience6_weight_decay0p01` | 17 | 0.275917 | 0.166390 |
| MLP residual corrector | `mlp_alpha0p1_clip_multiple0p25_epochs40_hidden64_lr0p0003_patience6_weight_decay0p01` | 19 | 0.275855 | 0.166340 |
| MLP residual corrector | `mlp_alpha0p1_clip_multiple0p25_epochs40_hidden64_lr0p0003_patience6_weight_decay0p01` | mean | 0.275896 | 0.166384 |
| MLP residual corrector | `mlp_alpha0p05_clip_multiple0p25_epochs40_hidden64_lr0p0003_patience6_weight_decay0p01` | 7 | 0.276341 | 0.166819 |
| MLP residual corrector | `mlp_alpha0p05_clip_multiple0p25_epochs40_hidden64_lr0p0003_patience6_weight_decay0p01` | 11 | 0.276374 | 0.166871 |
| MLP residual corrector | `mlp_alpha0p05_clip_multiple0p25_epochs40_hidden64_lr0p0003_patience6_weight_decay0p01` | 13 | 0.276369 | 0.166800 |
| MLP residual corrector | `mlp_alpha0p05_clip_multiple0p25_epochs40_hidden64_lr0p0003_patience6_weight_decay0p01` | 17 | 0.276356 | 0.166820 |
| MLP residual corrector | `mlp_alpha0p05_clip_multiple0p25_epochs40_hidden64_lr0p0003_patience6_weight_decay0p01` | 19 | 0.276326 | 0.166796 |
| MLP residual corrector | `mlp_alpha0p05_clip_multiple0p25_epochs40_hidden64_lr0p0003_patience6_weight_decay0p01` | mean | 0.276349 | 0.166819 |
| MLP residual corrector | `mlp_alpha0p1_clip_multiple0p5_epochs40_hidden64_lr0p0003_patience6_weight_decay0p01` | 7 | 0.275605 | 0.166159 |
| MLP residual corrector | `mlp_alpha0p1_clip_multiple0p5_epochs40_hidden64_lr0p0003_patience6_weight_decay0p01` | 11 | 0.275694 | 0.166291 |
| MLP residual corrector | `mlp_alpha0p1_clip_multiple0p5_epochs40_hidden64_lr0p0003_patience6_weight_decay0p01` | 13 | 0.275669 | 0.166092 |
| MLP residual corrector | `mlp_alpha0p1_clip_multiple0p5_epochs40_hidden64_lr0p0003_patience6_weight_decay0p01` | 17 | 0.275737 | 0.166169 |
| MLP residual corrector | `mlp_alpha0p1_clip_multiple0p5_epochs40_hidden64_lr0p0003_patience6_weight_decay0p01` | 19 | 0.275603 | 0.166078 |
| MLP residual corrector | `mlp_alpha0p1_clip_multiple0p5_epochs40_hidden64_lr0p0003_patience6_weight_decay0p01` | mean | 0.275643 | 0.166147 |
| MLP residual corrector | `mlp_alpha0p1_clip_multiple0p25_epochs40_hidden64_lr0p0003_patience6_weight_decay0p03` | 7 | 0.275892 | 0.166390 |
| MLP residual corrector | `mlp_alpha0p1_clip_multiple0p25_epochs40_hidden64_lr0p0003_patience6_weight_decay0p03` | 11 | 0.275947 | 0.166489 |
| MLP residual corrector | `mlp_alpha0p1_clip_multiple0p25_epochs40_hidden64_lr0p0003_patience6_weight_decay0p03` | 13 | 0.275938 | 0.166348 |
| MLP residual corrector | `mlp_alpha0p1_clip_multiple0p25_epochs40_hidden64_lr0p0003_patience6_weight_decay0p03` | 17 | 0.275917 | 0.166390 |
| MLP residual corrector | `mlp_alpha0p1_clip_multiple0p25_epochs40_hidden64_lr0p0003_patience6_weight_decay0p03` | 19 | 0.275855 | 0.166340 |
| MLP residual corrector | `mlp_alpha0p1_clip_multiple0p25_epochs40_hidden64_lr0p0003_patience6_weight_decay0p03` | mean | 0.275895 | 0.166384 |
| MLP residual corrector | `mlp_alpha0p1_clip_multiple0p25_epochs40_hidden32_lr0p0003_patience6_weight_decay0p01` | 7 | 0.275968 | 0.166493 |
| MLP residual corrector | `mlp_alpha0p1_clip_multiple0p25_epochs40_hidden32_lr0p0003_patience6_weight_decay0p01` | 11 | 0.275733 | 0.166243 |
| MLP residual corrector | `mlp_alpha0p1_clip_multiple0p25_epochs40_hidden32_lr0p0003_patience6_weight_decay0p01` | 13 | 0.275929 | 0.166347 |
| MLP residual corrector | `mlp_alpha0p1_clip_multiple0p25_epochs40_hidden32_lr0p0003_patience6_weight_decay0p01` | 17 | 0.275841 | 0.166357 |
| MLP residual corrector | `mlp_alpha0p1_clip_multiple0p25_epochs40_hidden32_lr0p0003_patience6_weight_decay0p01` | 19 | 0.275840 | 0.166335 |
| MLP residual corrector | `mlp_alpha0p1_clip_multiple0p25_epochs40_hidden32_lr0p0003_patience6_weight_decay0p01` | mean | 0.275847 | 0.166346 |
| Oracle prototype residual | `oracle_epochs10_num_prototypes16_residual_scale0p3_residual_weight0p001_teacher_lambda0p01` | 7 | 0.274949 | 0.165736 |
| Oracle prototype residual | `oracle_epochs10_num_prototypes16_residual_scale0p3_residual_weight0p001_teacher_lambda0p01` | 11 | 0.275434 | 0.166269 |
| Oracle prototype residual | `oracle_epochs10_num_prototypes16_residual_scale0p3_residual_weight0p001_teacher_lambda0p01` | 13 | 0.275755 | 0.166657 |
| Oracle prototype residual | `oracle_epochs10_num_prototypes16_residual_scale0p3_residual_weight0p001_teacher_lambda0p01` | 17 | 0.274675 | 0.165108 |
| Oracle prototype residual | `oracle_epochs10_num_prototypes16_residual_scale0p3_residual_weight0p001_teacher_lambda0p01` | 19 | 0.275219 | 0.165687 |
| Oracle prototype residual | `oracle_epochs10_num_prototypes16_residual_scale0p3_residual_weight0p001_teacher_lambda0p01` | mean | 0.274829 | 0.165538 |
| Oracle prototype residual | `oracle_epochs10_num_prototypes8_residual_scale0p3_residual_weight0p001_teacher_lambda0p01` | 7 | 0.275155 | 0.165962 |
| Oracle prototype residual | `oracle_epochs10_num_prototypes8_residual_scale0p3_residual_weight0p001_teacher_lambda0p01` | 11 | 0.275713 | 0.166456 |
| Oracle prototype residual | `oracle_epochs10_num_prototypes8_residual_scale0p3_residual_weight0p001_teacher_lambda0p01` | 13 | 0.276023 | 0.166833 |
| Oracle prototype residual | `oracle_epochs10_num_prototypes8_residual_scale0p3_residual_weight0p001_teacher_lambda0p01` | 17 | 0.276359 | 0.167323 |
| Oracle prototype residual | `oracle_epochs10_num_prototypes8_residual_scale0p3_residual_weight0p001_teacher_lambda0p01` | 19 | 0.275319 | 0.165656 |
| Oracle prototype residual | `oracle_epochs10_num_prototypes8_residual_scale0p3_residual_weight0p001_teacher_lambda0p01` | mean | 0.275258 | 0.166032 |
| Oracle prototype residual | `oracle_epochs10_num_prototypes32_residual_scale0p3_residual_weight0p001_teacher_lambda0p01` | 7 | 0.274800 | 0.165436 |
| Oracle prototype residual | `oracle_epochs10_num_prototypes32_residual_scale0p3_residual_weight0p001_teacher_lambda0p01` | 11 | 0.275397 | 0.166171 |
| Oracle prototype residual | `oracle_epochs10_num_prototypes32_residual_scale0p3_residual_weight0p001_teacher_lambda0p01` | 13 | 0.275110 | 0.165925 |
| Oracle prototype residual | `oracle_epochs10_num_prototypes32_residual_scale0p3_residual_weight0p001_teacher_lambda0p01` | 17 | 0.274707 | 0.165338 |
| Oracle prototype residual | `oracle_epochs10_num_prototypes32_residual_scale0p3_residual_weight0p001_teacher_lambda0p01` | 19 | 0.275135 | 0.165739 |
| Oracle prototype residual | `oracle_epochs10_num_prototypes32_residual_scale0p3_residual_weight0p001_teacher_lambda0p01` | mean | 0.274929 | 0.165626 |
| Oracle prototype residual | `oracle_epochs10_num_prototypes16_residual_scale0p3_residual_weight0p01_teacher_lambda0p01` | 7 | 0.274974 | 0.165775 |
| Oracle prototype residual | `oracle_epochs10_num_prototypes16_residual_scale0p3_residual_weight0p01_teacher_lambda0p01` | 11 | 0.275506 | 0.166356 |
| Oracle prototype residual | `oracle_epochs10_num_prototypes16_residual_scale0p3_residual_weight0p01_teacher_lambda0p01` | 13 | 0.275822 | 0.166737 |
| Oracle prototype residual | `oracle_epochs10_num_prototypes16_residual_scale0p3_residual_weight0p01_teacher_lambda0p01` | 17 | 0.274664 | 0.165113 |
| Oracle prototype residual | `oracle_epochs10_num_prototypes16_residual_scale0p3_residual_weight0p01_teacher_lambda0p01` | 19 | 0.275215 | 0.165699 |
| Oracle prototype residual | `oracle_epochs10_num_prototypes16_residual_scale0p3_residual_weight0p01_teacher_lambda0p01` | mean | 0.274855 | 0.165581 |
| Oracle prototype residual | `oracle_epochs10_num_prototypes16_residual_scale0p3_residual_weight0p001_teacher_lambda0p001` | 7 | 0.274748 | 0.165214 |
| Oracle prototype residual | `oracle_epochs10_num_prototypes16_residual_scale0p3_residual_weight0p001_teacher_lambda0p001` | 11 | 0.274779 | 0.165445 |
| Oracle prototype residual | `oracle_epochs10_num_prototypes16_residual_scale0p3_residual_weight0p001_teacher_lambda0p001` | 13 | 0.275933 | 0.166618 |
| Oracle prototype residual | `oracle_epochs10_num_prototypes16_residual_scale0p3_residual_weight0p001_teacher_lambda0p001` | 17 | 0.274707 | 0.165345 |
| Oracle prototype residual | `oracle_epochs10_num_prototypes16_residual_scale0p3_residual_weight0p001_teacher_lambda0p001` | 19 | 0.275264 | 0.165866 |
| Oracle prototype residual | `oracle_epochs10_num_prototypes16_residual_scale0p3_residual_weight0p001_teacher_lambda0p001` | mean | 0.274887 | 0.165514 |
| Dynamic fixed-three | `dynamic_cc0cf12ca47e` | 7 | 0.275379 | 0.166028 |
| Dynamic fixed-three | `dynamic_ef60d1c03d09` | 7 | 0.274817 | 0.165294 |
| Dynamic fixed-three | `dynamic_77df34e6b069` | 7 | 0.274746 | 0.165350 |
| Dynamic fixed-three | `dynamic_7714d9065803` | 7 | 0.274817 | 0.165294 |
| Dynamic fixed-three | `dynamic_82b4d724af20` | 7 | 0.274782 | 0.165276 |

## Selected Tuned Configurations

| Method | Config | Val MAE | Val MSE | Frozen artifacts |
|---|---|---:|---:|---|
| Ridge residual corrector | `ridge_alpha0p25_clip_multiple0p25_feature_setfull_ridge1` | 0.275036 | 0.165619 | `1` |
| MLP residual corrector | `mlp_alpha0p1_clip_multiple0p5_epochs40_hidden64_lr0p0003_patience6_weight_decay0p01` | 0.275643 | 0.166147 | `5` |
| Oracle prototype residual | `oracle_epochs10_num_prototypes16_residual_scale0p3_residual_weight0p001_teacher_lambda0p01` | 0.274829 | 0.165538 | `5` |
| Dynamic fixed-three | `dynamic_77df34e6b069` | 0.274746 | 0.165350 | `1` |

## Locked vs Tuned Test Results

| Method | Version | Val MAE | Test MAE | Test MSE | Diff vs DLinear | Diff vs full adaptive | Diff vs locked |
|---|---|---:|---:|---:|---:|---:|---:|
| Ridge residual corrector | `locked_etth1_config_etth2_replication` | 0.276038 | 0.297313 | 0.218187 | -0.004394 | -0.000495 | +0.000000 |
| MLP residual corrector | `locked_etth1_config_etth2_replication` | 0.276129 | 0.297254 | 0.218303 | -0.004453 | -0.000554 | +0.000000 |
| Oracle prototype residual | `locked_etth1_config_etth2_replication` | 0.276404 | 0.301185 | 0.222719 | -0.000522 | +0.003377 | +0.000000 |
| Dynamic fixed-three, seed 7 | `locked_etth1_config_etth2_replication` | 0.275379 | 0.297398 | 0.218294 | -0.004310 | -0.000410 | +0.000000 |
| Ridge residual corrector | `etth2_validation_tuned` | 0.275036 | 0.296787 | 0.217713 | -0.004921 | -0.001021 | -0.000526 |
| MLP residual corrector | `etth2_validation_tuned` | 0.275643 | 0.297041 | 0.218149 | -0.004666 | -0.000767 | -0.000213 |
| Oracle prototype residual | `etth2_validation_tuned` | 0.274829 | 0.298475 | 0.219894 | -0.003232 | +0.000667 | -0.002710 |
| Dynamic fixed-three | `etth2_validation_tuned` | 0.274746 | 0.298079 | 0.219521 | -0.003628 | +0.000271 | +0.000681 |

## Leakage Checks

- Search space declared in `declared_search_space.json` before sweep results were written.
- Router-train was used for fitting.
- Router-validation was used for hyperparameter and checkpoint selection.
- Test cache was not loaded before `tuned_manifest_before_test.json` was written.
- Test was evaluated once for each tuned winner after freeze.
- Causal residual features enforce `old_start + horizon <= current_start`.

## Reproduce

```powershell
python experiments\etth2_validation_tuned_missing_methods\run_etth2_validation_tuned_missing_methods.py --phase all --device cuda
```
