# Residual-Correction Follow-Up (C + Ridge(behavioral) -> residual)

Tests behavioral features as a correction on top of the already-trained C scorer, rather than a joint 35-feature retrain (as in the original D). Reuses the existing perturbation cache and feature pipeline unmodified.

## Results (router_val MAE / MSE)

| Dataset | C (reproduced) | C + Ridge residual | D (original, reference) | Equal (ref) | Oracle (ref) |
|---|---:|---:|---:|---:|---:|
| ETTh1 | 0.368334 | 0.368293 | 0.370864 | 0.367265 | 0.343984 |
| ETTh2 | 0.286494 | 0.286410 | 0.279710 | 0.280878 | 0.266483 |
| ETTm1 | 0.250401 | 0.250408 | 0.251473 | 0.248161 | 0.227306 |
| Weather | 0.159672 | 0.159689 | 0.160337 | 0.160341 | 0.150243 |
| Electricity | 0.219244 | 0.218574 | 0.222766 | 0.214457 | 0.217440 |

## C+Ridge vs C, and vs the original D

| Dataset | Ridge alpha | (C+Ridge) - C | (C+Ridge) - D |
|---|---:|---:|---:|
| ETTh1 | 100 | `-0.000041` | `-0.002571` |
| ETTh2 | 100 | `-0.000084` | `+0.006700` |
| ETTm1 | 100 | `+0.000006` | `-0.001065` |
| Weather | 100 | `+0.000018` | `-0.000648` |
| Electricity | 0.1 | `-0.000670` | `-0.004192` |

## Dependence-aware statistics

| Dataset | Comparison | Test | Mean delta | 95% CI | Excludes zero |
|---|---|---|---:|---|---|
| ETTh1 | C+Ridge_vs_C | iid_paired_bootstrap | `-0.000041` | [-0.000077, -0.000005] | True |
| ETTh1 | C+Ridge_vs_C | block_bootstrap_len12 | `-0.000041` | [-0.000100, +0.000017] | False |
| ETTh1 | C+Ridge_vs_C | block_bootstrap_len24 | `-0.000041` | [-0.000104, +0.000019] | False |
| ETTh1 | C+Ridge_vs_C | block_bootstrap_len48 | `-0.000041` | [-0.000109, +0.000023] | False |
| ETTh1 | C+Ridge_vs_C | every_12th_window_phase_bootstrap | `-0.000041` | [-0.000093, +0.000011] | False |
| ETTh1 | C+Ridge_vs_D | iid_paired_bootstrap | `-0.002571` | [-0.003378, -0.001808] | True |
| ETTh1 | C+Ridge_vs_D | block_bootstrap_len12 | `-0.002571` | [-0.003878, -0.001345] | True |
| ETTh1 | C+Ridge_vs_D | block_bootstrap_len24 | `-0.002571` | [-0.003998, -0.001226] | True |
| ETTh1 | C+Ridge_vs_D | block_bootstrap_len48 | `-0.002571` | [-0.004276, -0.001142] | True |
| ETTh1 | C+Ridge_vs_D | every_12th_window_phase_bootstrap | `-0.002571` | [-0.003532, -0.001570] | True |
| ETTh1 | C+Ridge_vs_Equal | iid_paired_bootstrap | `+0.001028` | [+0.000156, +0.001921] | True |
| ETTh1 | C+Ridge_vs_Equal | block_bootstrap_len12 | `+0.001028` | [-0.001092, +0.003136] | False |
| ETTh1 | C+Ridge_vs_Equal | block_bootstrap_len24 | `+0.001028` | [-0.001385, +0.003385] | False |
| ETTh1 | C+Ridge_vs_Equal | block_bootstrap_len48 | `+0.001028` | [-0.001775, +0.003718] | False |
| ETTh1 | C+Ridge_vs_Equal | every_12th_window_phase_bootstrap | `+0.001028` | [+0.000573, +0.001490] | True |
| ETTh2 | C+Ridge_vs_C | iid_paired_bootstrap | `-0.000084` | [-0.000112, -0.000054] | True |
| ETTh2 | C+Ridge_vs_C | block_bootstrap_len12 | `-0.000084` | [-0.000122, -0.000049] | True |
| ETTh2 | C+Ridge_vs_C | block_bootstrap_len24 | `-0.000084` | [-0.000124, -0.000050] | True |
| ETTh2 | C+Ridge_vs_C | block_bootstrap_len48 | `-0.000084` | [-0.000129, -0.000045] | True |
| ETTh2 | C+Ridge_vs_C | every_12th_window_phase_bootstrap | `-0.000084` | [-0.000108, -0.000061] | True |
| ETTh2 | C+Ridge_vs_D | iid_paired_bootstrap | `+0.006700` | [+0.005002, +0.008343] | True |
| ETTh2 | C+Ridge_vs_D | block_bootstrap_len12 | `+0.006700` | [+0.003777, +0.009648] | True |
| ETTh2 | C+Ridge_vs_D | block_bootstrap_len24 | `+0.006700` | [+0.003819, +0.009746] | True |
| ETTh2 | C+Ridge_vs_D | block_bootstrap_len48 | `+0.006700` | [+0.003697, +0.010231] | True |
| ETTh2 | C+Ridge_vs_D | every_12th_window_phase_bootstrap | `+0.006700` | [+0.005410, +0.007976] | True |
| ETTh2 | C+Ridge_vs_Equal | iid_paired_bootstrap | `+0.005532` | [+0.003768, +0.007314] | True |
| ETTh2 | C+Ridge_vs_Equal | block_bootstrap_len12 | `+0.005532` | [+0.001859, +0.009088] | True |
| ETTh2 | C+Ridge_vs_Equal | block_bootstrap_len24 | `+0.005532` | [+0.001916, +0.008950] | True |
| ETTh2 | C+Ridge_vs_Equal | block_bootstrap_len48 | `+0.005532` | [+0.001294, +0.009573] | True |
| ETTh2 | C+Ridge_vs_Equal | every_12th_window_phase_bootstrap | `+0.005535` | [+0.004234, +0.006811] | True |
| ETTm1 | C+Ridge_vs_C | iid_paired_bootstrap | `+0.000006` | [-0.000001, +0.000014] | False |
| ETTm1 | C+Ridge_vs_C | block_bootstrap_len12 | `+0.000006` | [-0.000007, +0.000020] | False |
| ETTm1 | C+Ridge_vs_C | block_bootstrap_len24 | `+0.000006` | [-0.000008, +0.000020] | False |
| ETTm1 | C+Ridge_vs_C | block_bootstrap_len48 | `+0.000006` | [-0.000008, +0.000021] | False |
| ETTm1 | C+Ridge_vs_C | every_12th_window_phase_bootstrap | `+0.000006` | [-0.000007, +0.000019] | False |
| ETTm1 | C+Ridge_vs_D | iid_paired_bootstrap | `-0.001065` | [-0.001448, -0.000685] | True |
| ETTm1 | C+Ridge_vs_D | block_bootstrap_len12 | `-0.001065` | [-0.001773, -0.000363] | True |
| ETTm1 | C+Ridge_vs_D | block_bootstrap_len24 | `-0.001065` | [-0.001818, -0.000309] | True |
| ETTm1 | C+Ridge_vs_D | block_bootstrap_len48 | `-0.001065` | [-0.001851, -0.000263] | True |
| ETTm1 | C+Ridge_vs_D | every_12th_window_phase_bootstrap | `-0.001065` | [-0.001351, -0.000798] | True |
| ETTm1 | C+Ridge_vs_Equal | iid_paired_bootstrap | `+0.002246` | [+0.001875, +0.002626] | True |
| ETTm1 | C+Ridge_vs_Equal | block_bootstrap_len12 | `+0.002246` | [+0.001396, +0.003038] | True |
| ETTm1 | C+Ridge_vs_Equal | block_bootstrap_len24 | `+0.002246` | [+0.001343, +0.003141] | True |
| ETTm1 | C+Ridge_vs_Equal | block_bootstrap_len48 | `+0.002246` | [+0.001284, +0.003184] | True |
| ETTm1 | C+Ridge_vs_Equal | every_12th_window_phase_bootstrap | `+0.002246` | [+0.001795, +0.002716] | True |
| Weather | C+Ridge_vs_C | iid_paired_bootstrap | `+0.000018` | [+0.000014, +0.000021] | True |
| Weather | C+Ridge_vs_C | block_bootstrap_len12 | `+0.000018` | [+0.000010, +0.000025] | True |
| Weather | C+Ridge_vs_C | block_bootstrap_len24 | `+0.000018` | [+0.000010, +0.000025] | True |
| Weather | C+Ridge_vs_C | block_bootstrap_len48 | `+0.000018` | [+0.000010, +0.000026] | True |
| Weather | C+Ridge_vs_C | every_12th_window_phase_bootstrap | `+0.000018` | [+0.000015, +0.000020] | True |
| Weather | C+Ridge_vs_D | iid_paired_bootstrap | `-0.000648` | [-0.000848, -0.000446] | True |
| Weather | C+Ridge_vs_D | block_bootstrap_len12 | `-0.000648` | [-0.001020, -0.000318] | True |
| Weather | C+Ridge_vs_D | block_bootstrap_len24 | `-0.000648` | [-0.001052, -0.000296] | True |
| Weather | C+Ridge_vs_D | block_bootstrap_len48 | `-0.000648` | [-0.001071, -0.000287] | True |
| Weather | C+Ridge_vs_D | every_12th_window_phase_bootstrap | `-0.000648` | [-0.000833, -0.000478] | True |
| Weather | C+Ridge_vs_Equal | iid_paired_bootstrap | `-0.000652` | [-0.001057, -0.000263] | True |
| Weather | C+Ridge_vs_Equal | block_bootstrap_len12 | `-0.000652` | [-0.001903, +0.000399] | False |
| Weather | C+Ridge_vs_Equal | block_bootstrap_len24 | `-0.000652` | [-0.002330, +0.000614] | False |
| Weather | C+Ridge_vs_Equal | block_bootstrap_len48 | `-0.000652` | [-0.002783, +0.000764] | False |
| Weather | C+Ridge_vs_Equal | every_12th_window_phase_bootstrap | `-0.000652` | [-0.000821, -0.000463] | True |
| Electricity | C+Ridge_vs_C | iid_paired_bootstrap | `-0.000670` | [-0.000737, -0.000610] | True |
| Electricity | C+Ridge_vs_C | block_bootstrap_len12 | `-0.000670` | [-0.000824, -0.000536] | True |
| Electricity | C+Ridge_vs_C | block_bootstrap_len24 | `-0.000670` | [-0.000829, -0.000537] | True |
| Electricity | C+Ridge_vs_C | block_bootstrap_len48 | `-0.000670` | [-0.000832, -0.000538] | True |
| Electricity | C+Ridge_vs_C | every_12th_window_phase_bootstrap | `-0.000670` | [-0.000875, -0.000470] | True |
| Electricity | C+Ridge_vs_D | iid_paired_bootstrap | `-0.004192` | [-0.004668, -0.003739] | True |
| Electricity | C+Ridge_vs_D | block_bootstrap_len12 | `-0.004192` | [-0.005430, -0.003179] | True |
| Electricity | C+Ridge_vs_D | block_bootstrap_len24 | `-0.004192` | [-0.005676, -0.003026] | True |
| Electricity | C+Ridge_vs_D | block_bootstrap_len48 | `-0.004192` | [-0.005926, -0.002842] | True |
| Electricity | C+Ridge_vs_D | every_12th_window_phase_bootstrap | `-0.004192` | [-0.005161, -0.003143] | True |
| Electricity | C+Ridge_vs_Equal | iid_paired_bootstrap | `+0.004117` | [+0.003773, +0.004449] | True |
| Electricity | C+Ridge_vs_Equal | block_bootstrap_len12 | `+0.004117` | [+0.003286, +0.004991] | True |
| Electricity | C+Ridge_vs_Equal | block_bootstrap_len24 | `+0.004117` | [+0.003145, +0.005170] | True |
| Electricity | C+Ridge_vs_Equal | block_bootstrap_len48 | `+0.004117` | [+0.003004, +0.005363] | True |
| Electricity | C+Ridge_vs_Equal | every_12th_window_phase_bootstrap | `+0.004116` | [+0.003629, +0.004704] | True |

## Competence-prediction metrics

| Dataset | Spearman | Pearson | Top-1 acc | AUROC |
|---|---:|---:|---:|---:|
| ETTh1 | 0.182 | 0.312 | 0.360 | 0.541 |
| ETTh2 | 0.130 | 0.246 | 0.354 | 0.512 |
| ETTm1 | 0.133 | 0.264 | 0.353 | 0.515 |
| Weather | 0.187 | 0.400 | 0.398 | 0.547 |
| Electricity | 0.519 | 0.425 | 0.537 | 0.621 |
