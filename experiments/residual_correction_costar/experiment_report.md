# Causal Residual Correction COSTAR-TS

## Protocol

- Dataset: ETTh1 router-train `20-60%`, validation `60-80%`.
- Test cache was not loaded or evaluated.
- Frozen baseline: `hybrid_chrono_hvema_lowrank1_decay0.95_temp0.1_alpha0.75`.
- Correction hyperparameters were selected on chronological router-train folds only.
- Online residual updates use only windows satisfying `old_start + horizon <= current_start`.

## Results

- Baseline reproduction mean MAE: `0.363642`.
- Best validation method: `experiment2_ridge`.
- Best MAE / MSE: `0.363301` / `0.306286`.
- Improvement vs `0.363642`: `0.000341` MAE (0.094%).
- Strong target `<= 0.3619`: `False`.
- Exceptional target `<= 0.3600`: `False`.
- Experiment 1 selected config: `variable_decay0.99_alpha0.1_clip0.5_warm0`.
- Experiment 2 ridge selected config: `ridge1_alpha0.1_clip0.25_full`.
- MLP run: `True`.
- Aggregate paired bootstrap CI for winner: `[-0.000378, -0.000305]`.

## Leakage Checks

- Causal update assertions passed: `True`.
- Test cache loaded: `False`.
- Long-history summaries end before forecast start: `True`.

## Reproduce

```powershell
python experiments\residual_correction_costar\run_residual_correction_experiments.py --device cuda
```
