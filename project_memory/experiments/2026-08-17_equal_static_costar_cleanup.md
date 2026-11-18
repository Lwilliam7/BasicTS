# Equal-Static COSTAR Cleanup

Date: 2026-08-17

## Goal

Remove the structural asymmetry where ETTh1 `OLD_FIXED3` received a trained static neural-router prior while any other selected triple received equal static weights.

## Change

Updated the active ETTh1 full adaptive path:

- Removed `load_static_winner_per_window` from `experiments/train_selected_core_etth1/run_train_selected_core_eval.py`.
- Replaced the `OLD_FIXED3` exception with:

```python
static_weights = torch.full(
    (int(cache["num_windows"]), 3),
    1.0 / 3.0,
)
```

- `static_weight_source` is now `equal_static_all_triples`.
- Updated `experiments/frozen_costar/run_frozen_costar_validation.py` so the frozen diagnostic also uses equal static weights and no longer imports or loads the neural-router checkpoint.

The online causal target updates remain unchanged; this only changes the static prior branch.

## Validation Results

No test cache was loaded.

ETTh1 equal-static full adaptive validation:

- Core: `PatchTST+iTransformer+TimesNet`
- Validation MAE/MSE: `0.363100` / `0.306026`
- Previous neural-prior path reference: `0.363112` / `0.306057`
- Difference: `-0.000012` MAE

Frozen-vs-online diagnostic after equal-static change:

| Dataset | Equal fixed-three | Frozen COSTAR | Online COSTAR |
|---|---:|---:|---:|
| ETTh1 | `0.367265` / `0.310530` | `0.365825` / `0.308399` | `0.363100` / `0.306026` |
| ETTh2 | `0.280878` / `0.171933` | `0.277481` / `0.167632` | `0.276832` / `0.167280` |

## Interpretation

Option A is implemented: the static prior is now structurally identical across triples. The old ETTh1 neural-router prior is not used in the active full adaptive COSTAR path.

The equal-static ETTh1 validation result is slightly better than the previous neural-prior path, but this is a validation-only cleanup performed after prior test results had been viewed. It should not be treated as a replacement preregistered final-test model unless a new freeze/evaluation protocol is explicitly created.

## Artifacts

- `experiments/train_selected_core_etth1/run_train_selected_core_eval.py`
- `experiments/train_selected_core_etth1_equal_static/final_report.json`
- `experiments/train_selected_core_etth1_equal_static/experiment_report.md`
- `experiments/frozen_costar/run_frozen_costar_validation.py`
- `experiments/frozen_costar/frozen_costar_validation_results.json`
- `experiments/frozen_costar/frozen_costar_report.md`
- `tests/test_frozen_costar.py`

Reproduce:

```powershell
python experiments\train_selected_core_etth1\run_train_selected_core_eval.py --device cuda --out-dir experiments\train_selected_core_etth1_equal_static --phase all
python experiments\frozen_costar\run_frozen_costar_validation.py --device cuda
```
