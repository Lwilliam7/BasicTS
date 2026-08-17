# Canonical ETTh2 Protocol

## Protocol

- Validation cache: `cache/costarts_fresh/ETTh2_96_12/router_val_cache.pt`.
- Validation starts: `10800..11412`, `613` windows.
- Horizon: `12`; variables: `7`.
- Canonical metric: raw/original-scale MAE/MSE using `sample_mae` with `std=ones`.
- Inverse transform: none applied; cached predictions and targets are evaluated directly.
- Diagnostic normalized metrics are saved separately and are not canonical.

## Best Raw Results

- Best single: `DLinear`, MAE `0.280957`, MSE `0.171493`.
- Best overall fixed/specialist row: `DLinear+ModernTCN`, method `fixed_2_equal`, MAE `0.275229`.
- Old summary best-by-size match: `True`.

## Reproduce

```powershell
python experiments\etth2_canonical_protocol\run_canonical_etth2_baselines.py
```
