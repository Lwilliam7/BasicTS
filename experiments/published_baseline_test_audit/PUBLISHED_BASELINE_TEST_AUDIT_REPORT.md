# Published Baseline Test Audit

Label: `post_hoc_comparative_audit`.

ETTh1 and ETTh2 test results had already been viewed before this run. This report is a frozen test-only comparative audit, not a clean untouched final-test claim.

All listed methods use configurations already selected from router-train/validation artifacts. No method was changed, selected, or tuned using these test results.

## Main Table

| Method | ETTh1 Test MAE | ETTh1 Test MSE | ETTh1 Val MAE | ETTh2 Test MAE | ETTh2 Test MSE | ETTh2 Val MAE |
|---|---:|---:|---:|---:|---:|---:|
| Equal fixed ensemble | 0.332001 | 0.270050 | 0.371099 | 0.322330 | 0.249527 | 0.300772 |
| Granger-Ramanathan | 0.340765 | 0.289594 | 0.382960 | 0.298419 | 0.218160 | 0.276704 |
| Bates-Granger | 0.327848 | 0.267809 | 0.368891 | 0.296294 | 0.217423 | 0.274915 |
| FAME adaptation | 0.331314 | 0.271990 | 0.379212 | 0.298372 | 0.220674 | 0.277008 |
| TimeRouter adaptation | 0.328178 | 0.267896 | 0.368234 | 0.306324 | 0.228592 | 0.283288 |
| Frozen COSTAR | 0.327175 | 0.267094 | 0.365825 | 0.300574 | 0.220499 | 0.277481 |
| Online COSTAR | 0.326408 | 0.267378 | 0.363100 | 0.297808 | 0.218612 | 0.276832 |
| Frozen COSTAR + Ridge residual | 0.326448 | 0.267452 | 0.363301 | 0.296787 | 0.217713 | 0.275036 |
| Frozen COSTAR + MLP residual | 0.326047 | 0.267322 | 0.363318 | 0.297041 | 0.218149 | 0.275643 |
| OneNet / adaptation | 0.330721 | 0.272812 | 0.370137 | 0.407526 | 0.413704 | 0.402666 |

## ETTh1 Ranking

| Rank | Method | Test MAE | Delta vs Online COSTAR | Delta vs Equal fixed |
|---:|---|---:|---:|---:|
| 1 | Frozen COSTAR + MLP residual | 0.326047 | -0.000361 | -0.005954 |
| 2 | Online COSTAR | 0.326408 | +0.000000 | -0.005593 |
| 3 | Frozen COSTAR + Ridge residual | 0.326448 | +0.000040 | -0.005553 |
| 4 | Frozen COSTAR | 0.327175 | +0.000767 | -0.004827 |
| 5 | Bates-Granger | 0.327848 | +0.001440 | -0.004153 |
| 6 | TimeRouter adaptation | 0.328178 | +0.001770 | -0.003823 |
| 7 | OneNet / adaptation | 0.330721 | +0.004313 | -0.001280 |
| 8 | FAME adaptation | 0.331314 | +0.004906 | -0.000687 |
| 9 | Equal fixed ensemble | 0.332001 | +0.005593 | +0.000000 |
| 10 | Granger-Ramanathan | 0.340765 | +0.014357 | +0.008764 |

## ETTh2 Ranking

| Rank | Method | Test MAE | Delta vs Online COSTAR | Delta vs Equal fixed |
|---:|---|---:|---:|---:|
| 1 | Bates-Granger | 0.296294 | -0.001514 | -0.026037 |
| 2 | Frozen COSTAR + Ridge residual | 0.296787 | -0.001021 | -0.025543 |
| 3 | Frozen COSTAR + MLP residual | 0.297041 | -0.000767 | -0.025289 |
| 4 | Online COSTAR | 0.297808 | +0.000000 | -0.024522 |
| 5 | FAME adaptation | 0.298372 | +0.000564 | -0.023959 |
| 6 | Granger-Ramanathan | 0.298419 | +0.000611 | -0.023912 |
| 7 | Frozen COSTAR | 0.300574 | +0.002765 | -0.021757 |
| 8 | TimeRouter adaptation | 0.306324 | +0.008516 | -0.016007 |
| 9 | Equal fixed ensemble | 0.322330 | +0.024522 | +0.000000 |
| 10 | OneNet / adaptation | 0.407526 | +0.109718 | +0.085196 |

## Causality And Provenance

- Online COSTAR and OneNet use realized feedback only after `old_forecast_start + forecast_horizon <= current_forecast_start`.
- Frozen COSTAR, Equal fixed, Granger-Ramanathan, Bates-Granger, FAME, TimeRouter, Ridge residual, and MLP residual do not use realized test feedback in the predictions reported here.
- Ridge/MLP residual rows are carried from existing frozen residual artifacts referenced in the CSV/JSON outputs; they were not reselected during this audit.
- Git commit: `8e86f0c5d9140ba44afb3c46fe69cf270b6b4317`.