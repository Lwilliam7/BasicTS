# Forecast-Time Behavioral Competence Routing -- Validation-Only Proof of Concept

Research question: does an expert's behavior under small perturbations of the current historical input predict its upcoming reliability, beyond window/forecast/disagreement features already available? Validation only -- no test cache was ever loaded.

## Main result table (router_val MAE / MSE)

| Dataset | Best Single | Equal | Frozen HxV | Online HxV (ref, uses feedback) | A: Window | B: +Forecast | C: +Disagreement | D: +Behavioral | Oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ETTh1 | 0.379558 | 0.367265 | 0.366022 | 0.363949 | 0.367265 | 0.368723 | 0.368334 | 0.370864 | 0.343984 |
| ETTh2 | 0.280957 | 0.280878 | 0.276898 | 0.276354 | 0.280878 | 0.281922 | 0.286494 | 0.279710 | 0.266483 |
| ETTm1 | 0.261771 | 0.248161 | 0.250690 | 0.248593 | 0.248161 | 0.250199 | 0.250401 | 0.251473 | 0.227306 |
| Weather | 0.164673 | 0.160341 | 0.159818 | 0.159280 | 0.160341 | 0.161034 | 0.159672 | 0.160337 | 0.150243 |
| Electricity | 0.225385 | 0.214457 | 0.215355 | 0.211775 | 0.214457 | 0.217118 | 0.219244 | 0.222766 | 0.217440 |

## D vs C (the central comparison)

| Dataset | D-C MAE | Relative improvement | Headroom captured (D over C, of C->Oracle gap) |
|---|---:|---:|---|
| ETTh1 | `+0.002530` | `-0.687%` | -10.4% |
| ETTh2 | `-0.006784` | `+2.368%` | 33.9% |
| ETTm1 | `+0.001071` | `-0.428%` | -4.6% |
| Weather | `+0.000666` | `-0.417%` | -7.1% |
| Electricity | `+0.003522` | `-1.606%` | -195.2% |

## Competence-prediction metrics

| Dataset | Ablation | Spearman | Pearson | Top-1 acc | AUROC (useful) | AUPRC |
|---|---|---:|---:|---:|---:|---:|
| ETTh1 | A_window_only | 0.058 | 0.100 | 0.355 | 0.492 | 0.342 |
| ETTh1 | B_window_forecast | 0.103 | 0.075 | 0.354 | 0.537 | 0.364 |
| ETTh1 | C_window_forecast_disagreement | 0.181 | 0.314 | 0.358 | 0.539 | 0.360 |
| ETTh1 | D_full_behavioral | 0.152 | 0.254 | 0.344 | 0.531 | 0.364 |
| ETTh2 | A_window_only | -0.067 | -0.059 | 0.529 | 0.461 | 0.302 |
| ETTh2 | B_window_forecast | -0.073 | -0.112 | 0.235 | 0.503 | 0.315 |
| ETTh2 | C_window_forecast_disagreement | 0.128 | 0.244 | 0.352 | 0.511 | 0.315 |
| ETTh2 | D_full_behavioral | 0.443 | 0.480 | 0.478 | 0.669 | 0.418 |
| ETTm1 | A_window_only | 0.066 | 0.076 | 0.309 | 0.506 | 0.327 |
| ETTm1 | B_window_forecast | 0.109 | 0.191 | 0.344 | 0.512 | 0.336 |
| ETTm1 | C_window_forecast_disagreement | 0.134 | 0.266 | 0.353 | 0.514 | 0.332 |
| ETTm1 | D_full_behavioral | 0.119 | 0.189 | 0.330 | 0.520 | 0.345 |
| Weather | A_window_only | 0.033 | 0.059 | 0.428 | 0.505 | 0.299 |
| Weather | B_window_forecast | 0.076 | 0.053 | 0.346 | 0.536 | 0.318 |
| Weather | C_window_forecast_disagreement | 0.190 | 0.400 | 0.399 | 0.548 | 0.344 |
| Weather | D_full_behavioral | 0.236 | 0.365 | 0.369 | 0.555 | 0.337 |
| Electricity | A_window_only | 0.043 | 0.059 | 0.231 | 0.524 | 0.137 |
| Electricity | B_window_forecast | 0.327 | 0.275 | 0.368 | 0.574 | 0.152 |
| Electricity | C_window_forecast_disagreement | 0.517 | 0.424 | 0.545 | 0.623 | 0.170 |
| Electricity | D_full_behavioral | 0.320 | 0.254 | 0.366 | 0.540 | 0.133 |

## Dependence-aware D vs C / D vs Equal (block bootstrap)

| Dataset | Comparison | Test | Mean delta | 95% CI | Excludes zero |
|---|---|---|---:|---|---|
| ETTh1 | D_vs_C | iid_paired_bootstrap | `+0.002530` | [+0.001763, +0.003335] | True |
| ETTh1 | D_vs_C | block_bootstrap_len12 | `+0.002530` | [+0.001302, +0.003833] | True |
| ETTh1 | D_vs_C | block_bootstrap_len24 | `+0.002530` | [+0.001195, +0.003955] | True |
| ETTh1 | D_vs_C | block_bootstrap_len48 | `+0.002530` | [+0.001123, +0.004216] | True |
| ETTh1 | D_vs_C | every_12th_window_phase_bootstrap | `+0.002530` | [+0.001502, +0.003521] | True |
| ETTh1 | D_vs_Equal | iid_paired_bootstrap | `+0.003599` | [+0.002640, +0.004582] | True |
| ETTh1 | D_vs_Equal | block_bootstrap_len12 | `+0.003599` | [+0.001514, +0.005685] | True |
| ETTh1 | D_vs_Equal | block_bootstrap_len24 | `+0.003599` | [+0.001347, +0.005816] | True |
| ETTh1 | D_vs_Equal | block_bootstrap_len48 | `+0.003599` | [+0.001176, +0.006060] | True |
| ETTh1 | D_vs_Equal | every_12th_window_phase_bootstrap | `+0.003600` | [+0.002814, +0.004414] | True |
| ETTh2 | D_vs_C | iid_paired_bootstrap | `-0.006784` | [-0.008438, -0.005077] | True |
| ETTh2 | D_vs_C | block_bootstrap_len12 | `-0.006784` | [-0.009747, -0.003856] | True |
| ETTh2 | D_vs_C | block_bootstrap_len24 | `-0.006784` | [-0.009852, -0.003895] | True |
| ETTh2 | D_vs_C | block_bootstrap_len48 | `-0.006784` | [-0.010321, -0.003777] | True |
| ETTh2 | D_vs_C | every_12th_window_phase_bootstrap | `-0.006783` | [-0.008055, -0.005507] | True |
| ETTh2 | D_vs_Equal | iid_paired_bootstrap | `-0.001168` | [-0.002582, +0.000269] | False |
| ETTh2 | D_vs_Equal | block_bootstrap_len12 | `-0.001168` | [-0.004257, +0.001743] | False |
| ETTh2 | D_vs_Equal | block_bootstrap_len24 | `-0.001168` | [-0.004405, +0.001630] | False |
| ETTh2 | D_vs_Equal | block_bootstrap_len48 | `-0.001168` | [-0.004206, +0.001248] | False |
| ETTh2 | D_vs_Equal | every_12th_window_phase_bootstrap | `-0.001165` | [-0.002036, -0.000356] | True |
| ETTm1 | D_vs_C | iid_paired_bootstrap | `+0.001071` | [+0.000695, +0.001455] | True |
| ETTm1 | D_vs_C | block_bootstrap_len12 | `+0.001071` | [+0.000372, +0.001779] | True |
| ETTm1 | D_vs_C | block_bootstrap_len24 | `+0.001071` | [+0.000314, +0.001820] | True |
| ETTm1 | D_vs_C | block_bootstrap_len48 | `+0.001071` | [+0.000270, +0.001858] | True |
| ETTm1 | D_vs_C | every_12th_window_phase_bootstrap | `+0.001071` | [+0.000809, +0.001354] | True |
| ETTm1 | D_vs_Equal | iid_paired_bootstrap | `+0.003311` | [+0.002894, +0.003740] | True |
| ETTm1 | D_vs_Equal | block_bootstrap_len12 | `+0.003311` | [+0.002498, +0.004088] | True |
| ETTm1 | D_vs_Equal | block_bootstrap_len24 | `+0.003311` | [+0.002464, +0.004133] | True |
| ETTm1 | D_vs_Equal | block_bootstrap_len48 | `+0.003311` | [+0.002417, +0.004157] | True |
| ETTm1 | D_vs_Equal | every_12th_window_phase_bootstrap | `+0.003311` | [+0.002791, +0.003895] | True |
| Weather | D_vs_C | iid_paired_bootstrap | `+0.000666` | [+0.000463, +0.000866] | True |
| Weather | D_vs_C | block_bootstrap_len12 | `+0.000666` | [+0.000336, +0.001040] | True |
| Weather | D_vs_C | block_bootstrap_len24 | `+0.000666` | [+0.000313, +0.001071] | True |
| Weather | D_vs_C | block_bootstrap_len48 | `+0.000666` | [+0.000305, +0.001091] | True |
| Weather | D_vs_C | every_12th_window_phase_bootstrap | `+0.000666` | [+0.000495, +0.000852] | True |
| Weather | D_vs_Equal | iid_paired_bootstrap | `-0.000004` | [-0.000377, +0.000360] | False |
| Weather | D_vs_Equal | block_bootstrap_len12 | `-0.000004` | [-0.001179, +0.001009] | False |
| Weather | D_vs_Equal | block_bootstrap_len24 | `-0.000004` | [-0.001558, +0.001259] | False |
| Weather | D_vs_Equal | block_bootstrap_len48 | `-0.000004` | [-0.001992, +0.001422] | False |
| Weather | D_vs_Equal | every_12th_window_phase_bootstrap | `-0.000004` | [-0.000195, +0.000207] | False |
| Electricity | D_vs_C | iid_paired_bootstrap | `+0.003522` | [+0.003094, +0.003974] | True |
| Electricity | D_vs_C | block_bootstrap_len12 | `+0.003522` | [+0.002586, +0.004653] | True |
| Electricity | D_vs_C | block_bootstrap_len24 | `+0.003522` | [+0.002433, +0.004913] | True |
| Electricity | D_vs_C | block_bootstrap_len48 | `+0.003522` | [+0.002261, +0.005141] | True |
| Electricity | D_vs_C | every_12th_window_phase_bootstrap | `+0.003522` | [+0.002525, +0.004416] | True |
| Electricity | D_vs_Equal | iid_paired_bootstrap | `+0.008309` | [+0.007835, +0.008793] | True |
| Electricity | D_vs_Equal | block_bootstrap_len12 | `+0.008309` | [+0.007004, +0.009761] | True |
| Electricity | D_vs_Equal | block_bootstrap_len24 | `+0.008309` | [+0.006732, +0.010120] | True |
| Electricity | D_vs_Equal | block_bootstrap_len48 | `+0.008309` | [+0.006466, +0.010480] | True |
| Electricity | D_vs_Equal | every_12th_window_phase_bootstrap | `+0.008308` | [+0.007332, +0.009266] | True |

## Integrity checks

- **ETTh1**: target-perturbation invariance = `PASS`; max forecast-reproduction error vs cached predictions = `8.45e+00`.
- **ETTh2**: target-perturbation invariance = `PASS`; max forecast-reproduction error vs cached predictions = `1.91e-06`.
- **ETTm1**: target-perturbation invariance = `PASS`; max forecast-reproduction error vs cached predictions = `1.96e+00`.
- **Weather**: target-perturbation invariance = `PASS`; max forecast-reproduction error vs cached predictions = `1.15e+02`.
- **Electricity**: target-perturbation invariance = `PASS`; max forecast-reproduction error vs cached predictions = `5.51e+01`.

## Go / No-Go decision

**NO-GO**

- D beats C on MAE on 1/5 datasets: {'ETTh1': False, 'ETTh2': True, 'ETTm1': False, 'Weather': False, 'Electricity': False}.
- Headroom captured (D over C, >=10% of C->Oracle gap) on: ['ETTh2'].
- Dependence-aware (block-bootstrap) statistically supported D<C on: ['ETTh2'].
- D improves competence-ranking Spearman correlation over C on 2/5 datasets.

## Hard rule compliance

```text
TEST SET ACCESSED: NO
TEST CACHE LOADED: NO
TEST METRICS COMPUTED: NO
```
