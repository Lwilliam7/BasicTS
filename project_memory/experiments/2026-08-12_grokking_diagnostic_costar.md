# Grokking Diagnostic: Prototype-Residual Fixed-Three Router

## Objective

Determine whether much longer training of the strongest existing neural trainable COSTAR-TS fixed-three router produces delayed chronological generalization consistent with grokking.

## Selected Model

Tested:

- `final_phase2_protores_lam0.01_k16_scale0.3_rw0.001`
- Family: oracle-weight prototype-residual `WeightStudent`
- Experts: `PatchTST`, `iTransformer`, `TimesNet`
- Original duration: `10` epochs
- Optimizer: AdamW
- Current/default weight decay: `0.01`
- Weight decay settings: `0.001`, `0.01`, `0.1`

Reason for selection:

- Best five-seed neural trainable fixed-three router in the oracle-weight tournament: validation MAE `0.366028 +/- 0.000242`.
- It contributed the static trainable component used by later chronological and horizon-variable hybrids.
- The current ridge residual result is stronger overall but does not have a neural long-training process suitable for a grokking diagnostic.

## Split And Safety

Used only router-train `20-60%`.

- Train fold: starts `2880..7423`
- Chronological evaluation fold: starts `7424..8532`
- Validation cache was not loaded.
- Test cache was not loaded.

## Predefined Grokking Criteria

Early region:

- epochs `0..10`

Delayed region:

- epochs `>= 50`

Required meaningful margin:

- delayed fold MAE at least `0.0005` below the best early checkpoint

Required sustained improvement:

- at least `5` consecutive delayed checkpoints satisfying the margin

## Command

```powershell
python experiments\grokking_diagnostic_costar\run_grokking_diagnostic.py --device cuda
```

## Results

| Weight decay | Epoch 10 fold MAE | Best epoch | Best fold MAE | Epoch 100 fold MAE |
|---:|---:|---:|---:|---:|
| `0.001` | `0.344265` | `26` | `0.342431` | `0.347406` |
| `0.01` | `0.344252` | `26` | `0.342420` | `0.347434` |
| `0.1` | `0.344233` | `26` | `0.342053` | `0.347171` |

Best overall internal-fold checkpoint:

- weight decay `0.1`
- epoch `26`
- fold MAE `0.342053`
- fold MSE `0.257475`
- improvement vs epoch 10: `0.002179` MAE
- improvement vs fixed-weight fold baseline: `0.001444` MAE

Late behavior:

- Best delayed checkpoint for weight decay `0.1` was epoch `50`, fold MAE `0.344086`.
- This was worse than the best early checkpoint by `0.001314` MAE.
- Epoch `100` degraded further to `0.347171`.

Compute:

- Runtime: `151.6` seconds.
- Peak GPU memory: `236.8 MB`.

## Interpretation

Longer training did improve chronological fold MAE compared with the original 10-epoch duration, but the improvement happened around epoch `26`, not after a long period of poor chronological performance. After that, chronological performance degraded while training MAE continued improving.

This is ordinary mid-training generalization followed by overfitting, not grokking.

Seed repeats were not run because seed `7` did not satisfy the predefined grokking criteria.

## Decision

Do not allocate more compute to grokking-style long training for this router. If this model family is revisited, focus on checkpoint selection within router-train folds or new signals/objectives, not 10x longer training.

## Relevant Files

- `experiments/grokking_diagnostic_costar/run_grokking_diagnostic.py`
- `experiments/grokking_diagnostic_costar/final_report.json`
- `experiments/grokking_diagnostic_costar/checkpoint_metrics.csv`
- `experiments/grokking_diagnostic_costar/weight_decay_summary.csv`
- `experiments/grokking_diagnostic_costar/learning_curve_fold_mae.svg`
- `experiments/grokking_diagnostic_costar/learning_curve_training_mae.svg`
- `experiments/grokking_diagnostic_costar/grokking_report.md`
