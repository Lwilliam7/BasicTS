# ETTh2 96 -> 12 Pre-Lock Run Audit

Audit created: 2026-07-26T22:03:44.561268-04:00

## Bottom Line

- ETTh2 training is not still running.
- Five ETTh2 checkpoint files exist and load, but the run is not eligible as strict fresh confirmation.
- The strict failure is pre-lock test-value access: the training helper loaded `test_data.npy` while reconstructing the full chronological series.
- There is no evidence that ETTh2 test forecasts or test metrics were generated.
- Some ETTh2 expert configs are inconsistent with the ETTh1 five-expert reference, especially PatchTST and ModernTCN.

## Active Python Processes

- PID 15636: start=2026-07-26T21:59:30.9694713-04:00, CPU=1.140625, classification=unrelated_python, state=likely active or idle Jupyter kernel; do not terminate
  Command line: `c:\Users\luwil\miniconda3\envs\BasicTS\python.exe -m ipykernel_launcher --f=c:\Users\luwil\AppData\Roaming\jupyter\runtime\kernel-v3eb779fd44dc95711a3d5fc3276bad7636dcf7429.json`

No process should be stopped based on this audit.

## Generated Artifacts

### current_checkout
- C:\Users\luwil\OneDrive\Documents\Code\BasicTS\checkpoints\costarts_fresh\ETTh2_96_12: missing
- C:\Users\luwil\OneDrive\Documents\Code\BasicTS\results\router_summary\costarts_fresh\ETTh2_96_12: exists
- C:\Users\luwil\OneDrive\Documents\Code\BasicTS\cache\costarts_fresh\ETTh2_96_12: missing

### isolated_runner
- C:\Users\luwil\AppData\Local\BasicTSCodexWatcher\runner\checkpoints\costarts_fresh\ETTh2_96_12: exists
  - checkpoints/costarts_fresh/ETTh2_96_12/candidates (directory, size=None, modified=2026-07-26T18:04:29.234042-04:00)
  - checkpoints/costarts_fresh/ETTh2_96_12/candidates/best_dlinear.pt (file, size=34510, modified=2026-07-26T17:41:47.606647-04:00)
  - checkpoints/costarts_fresh/ETTh2_96_12/candidates/best_itransformer.pt (file, size=515700, modified=2026-07-26T17:43:02.838308-04:00)
  - checkpoints/costarts_fresh/ETTh2_96_12/candidates/best_moderntcn.pt (file, size=162533, modified=2026-07-26T18:04:47.864689-04:00)
  - checkpoints/costarts_fresh/ETTh2_96_12/candidates/best_patchtst.pt (file, size=1835705, modified=2026-07-26T17:42:17.769489-04:00)
  - checkpoints/costarts_fresh/ETTh2_96_12/candidates/best_timesnet.pt (file, size=1885656, modified=2026-07-26T17:57:56.375731-04:00)
- C:\Users\luwil\AppData\Local\BasicTSCodexWatcher\runner\results\router_summary\costarts_fresh\ETTh2_96_12: exists
  - results/router_summary/costarts_fresh/ETTh2_96_12/expert_training_resume_summary.json (file, size=9987, modified=2026-07-26T18:05:06.295474-04:00)
- C:\Users\luwil\AppData\Local\BasicTSCodexWatcher\runner\cache\costarts_fresh\ETTh2_96_12: missing

## Expert Checkpoints

- DLinear: `checkpoints/costarts_fresh/ETTh2_96_12/candidates/best_dlinear.pt`, size=34510, modified=2026-07-26T17:41:47.606647-04:00, epoch=28, val MAE=0.37146313992002467, val MSE=0.3426404260802924, reload-valid=True
- PatchTST: `checkpoints/costarts_fresh/ETTh2_96_12/candidates/best_patchtst.pt`, size=1835705, modified=2026-07-26T17:42:17.769489-04:00, epoch=10, val MAE=0.37401145417492765, val MSE=0.3485877224049316, reload-valid=True
- iTransformer: `checkpoints/costarts_fresh/ETTh2_96_12/candidates/best_itransformer.pt`, size=515700, modified=2026-07-26T17:43:02.838308-04:00, epoch=15, val MAE=0.8559427803379416, val MSE=1.7919062285102525, reload-valid=True
- TimesNet: `checkpoints/costarts_fresh/ETTh2_96_12/candidates/best_timesnet.pt`, size=1885656, modified=2026-07-26T17:57:56.375731-04:00, epoch=11, val MAE=0.41811901252377826, val MSE=0.5082132523923392, reload-valid=True
- ModernTCN: `checkpoints/costarts_fresh/ETTh2_96_12/candidates/best_moderntcn.pt`, size=162533, modified=2026-07-26T18:04:47.864689-04:00, epoch=6, val MAE=1.1558120068521596, val MSE=1.929773493875701, reload-valid=True

## Config Consistency

- DLinear: matches ETTh1 reference config fields present in checkpoints.
- PatchTST: differs from ETTh1 reference:
  - affine: ETTh2=False, ETTh1 reference=True
  - norm_type: ETTh2='batch_norm', ETTh1 reference='layer_norm'
- iTransformer: matches ETTh1 reference config fields present in checkpoints.
- TimesNet: differs from ETTh1 reference:
  - hidden_size: ETTh2=32, ETTh1 reference=64
  - intermediate_size: ETTh2=64, ETTh1 reference=128
- ModernTCN: differs from ETTh1 reference:
  - hidden_size: ETTh2=32, ETTh1 reference=64
  - num_layers: ETTh2=2, ETTh1 reference=3
  - use_revin: ETTh2=False, ETTh1 reference=True

ETTh2 observed training hyperparameters: Adam, lr=0.001, max_epochs=40, patience=6, batch_size=512, seed=7, CPU, masked_mse.

## Test-Data Access

- Test file existence checked: yes.
- Test file size checked: yes.
- Raw test files copied to the runner before lock: yes.
- Test values loaded before lock: yes, via `load_full_chronological_data`, which loads `train_data.npy`, `val_data.npy`, and `test_data.npy`.
- Test forecasts generated: no evidence.
- Test metrics generated: no evidence.

Because test values were loaded before lock, these artifacts are debug-only under the strict fresh-confirmation rule.

## Split Safety

- expert_train: [0, 7200), timestamps=7200, windows=7093
- expert_val: [7200, 8640), timestamps=1440, windows=1333
- router_train: [8640, 10800), timestamps=2160, windows=2053
- router_val: [10800, 11520), timestamps=720, windows=613
- locked_test: [11520, 14400), timestamps=2880, windows=2773

The recorded checkpoint split metadata follows these boundaries; scaler metadata is consistent with expert-train-only fitting; expert checkpoint selection used expert-val metrics. However, locked-test exclusion failed at the data-loading protocol level.

## Required Conclusions

1. ETTh2 training is not still running.
2. DLinear, PatchTST, iTransformer, TimesNet, and ModernTCN checkpoint files are complete enough to load.
3. The checkpoints are reload-valid PyTorch checkpoints, but not valid for strict fresh confirmation.
4. A strict rerun is needed; PatchTST and ModernTCN also need architecture correction if matching the ETTh1 setup is required.
5. Configurations are not fully consistent with ETTh1 reference.
6. ETTh2 test values were accessed before lock; test metrics were not.
7. Existing artifacts are not eligible for strict fresh confirmation.
8. All runner ETTh2 fresh-run artifacts listed in `debug_only_artifacts` must be classified as debug-only.
9. No process should be stopped.
10. Smallest safe next action: label/archive these as prelock debug-only, then start a new ETTh2 prep that avoids loading test arrays until after lock.
