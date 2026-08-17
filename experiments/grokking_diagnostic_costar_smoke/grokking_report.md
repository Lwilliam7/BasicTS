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
- Best early fold MAE: `0.343337` at epoch `1`.
- Best late fold MAE: `n/a` at epoch `n/a`.
- Possible grokking: `False`.
- Repeat seeds run: `False`.
- Peak GPU memory MB: `236.8`.
- Runtime seconds: `7.8`.

## Decision

Abandon grokking-style long training for this router unless a new signal/objective is introduced.

## Reproduce

```powershell
python experiments\grokking_diagnostic_costar\run_grokking_diagnostic.py --device cuda
```
