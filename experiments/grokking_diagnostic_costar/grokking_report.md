# COSTAR-TS Grokking Diagnostic

## Selected Model

- Model: `final_phase2_protores_lam0.01_k16_scale0.3_rw0.001`
- Family: prototype-residual oracle-weight student over PatchTST, iTransformer, TimesNet.
- Original training duration: `10` epochs.
- Diagnostic duration: `100` epochs.
- Weight decay settings: `0.001, 0.01, 0.1`.

## Predefined Grokking Criteria

- Early checkpoint region: epochs `0..10`.
- Delayed region starts at epoch `50`.
- Meaningful margin: `0.0005` MAE below best early checkpoint.
- Sustained: at least `5` consecutive delayed checkpoints.

## Result

- Best delayed setting: weight decay `0.1`.
- Best early fold MAE: `0.342773` at epoch `8`.
- Best late fold MAE: `0.344086` at epoch `50`.
- Possible grokking: `False`.
- Repeat seeds run: `False`.
- Peak GPU memory MB: `236.8`.
- Runtime seconds: `151.6`.


## Curve Summary

| Weight decay | Epoch 10 fold MAE | Best epoch | Best fold MAE | Epoch 100 fold MAE |
|---:|---:|---:|---:|---:|
| `0.001` | `0.344265` | `26` | `0.342431` | `0.347406` |
| `0.01` | `0.344252` | `26` | `0.342420` | `0.347434` |
| `0.1` | `0.344233` | `26` | `0.342053` | `0.347171` |
## Decision

Abandon grokking-style long training for this router unless a new signal/objective is introduced.

## Reproduce

```powershell
python experiments\grokking_diagnostic_costar\run_grokking_diagnostic.py --device cuda
```
