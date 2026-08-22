# LearnedProbe-Rank Generalization Study (router_val only)

Frozen LearnedProbe-Rank (see ../FROZEN_METHOD.md) evaluated on 3-5 new BasicTS datasets that did not influence its development. No architecture/hyperparameter/loss/decision-rule change was made. router_val only; no new dataset's test split was built or accessed.

## 1. Datasets selected and why

See `dataset_selection.json`. Selected: ExchangeRate, Traffic, BeijingAirQuality, ETTm2 -- chosen for domain/variable-count/periodicity/scale diversity from the compatible BasicTS datasets, finalized before any LearnedProbe-Rank performance was inspected.

## Primary results (router_val MAE / MSE)

| Dataset | Equal | C-Rank | FixedD-Rank | LearnedProbe-Rank | Δ vs Equal | Δ vs C-Rank | Δ vs FixedD-Rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| ExchangeRate | 0.127527 | 0.126555 | 0.126470 | 0.123129 | `-0.004399` | `-0.003427` | `-0.003341` |
| Traffic | 0.275882 | 0.269214 | 0.267274 | 0.268150 | `-0.007732` | `-0.001064` | `+0.000876` |
| BeijingAirQuality | 0.257548 | 0.258336 | 0.258143 | 0.258058 | `+0.000509` | `-0.000279` | `-0.000085` |
| ETTm2 | 0.160422 | 0.161221 | 0.161529 | 0.160877 | `+0.000455` | `-0.000344` | `-0.000652` |

## Competence metrics

| Dataset | Method | Spearman | Pairwise acc | Top-1 acc | Top-2 recall | Mean rank of true best |
|---|---|---:|---:|---:|---:|---:|
| ExchangeRate | C_Rank | 0.095 | 0.548 | 0.338 | 0.733 | 0.929 |
| ExchangeRate | FixedD_Rank | 0.176 | 0.568 | 0.373 | 0.734 | 0.894 |
| ExchangeRate | LearnedProbe_Rank | 0.519 | 0.739 | 0.578 | 0.863 | 0.559 |
| Traffic | C_Rank | 0.739 | 0.857 | 0.706 | 0.964 | 0.330 |
| Traffic | FixedD_Rank | 0.776 | 0.856 | 0.673 | 0.984 | 0.344 |
| Traffic | LearnedProbe_Rank | 0.555 | 0.820 | 0.673 | 0.952 | 0.374 |
| BeijingAirQuality | C_Rank | 0.140 | 0.545 | 0.326 | 0.658 | 1.016 |
| BeijingAirQuality | FixedD_Rank | 0.149 | 0.548 | 0.340 | 0.677 | 0.983 |
| BeijingAirQuality | LearnedProbe_Rank | 0.210 | 0.575 | 0.335 | 0.670 | 0.995 |
| ETTm2 | C_Rank | 0.022 | 0.518 | 0.328 | 0.647 | 1.025 |
| ETTm2 | FixedD_Rank | 0.014 | 0.500 | 0.319 | 0.637 | 1.043 |
| ETTm2 | LearnedProbe_Rank | 0.090 | 0.543 | 0.326 | 0.663 | 1.011 |

## Dependence-aware statistics

| Dataset | Comparison | Test | Mean Δ | 95% CI | Excludes zero |
|---|---|---|---:|---|---|
| ExchangeRate | LearnedProbeRank_vs_CRank | iid_paired_bootstrap | `-0.003427` | [-0.003859, -0.002973] | True |
| ExchangeRate | LearnedProbeRank_vs_CRank | block_bootstrap_len12 | `-0.003427` | [-0.004280, -0.002501] | True |
| ExchangeRate | LearnedProbeRank_vs_CRank | block_bootstrap_len24 | `-0.003427` | [-0.004303, -0.002433] | True |
| ExchangeRate | LearnedProbeRank_vs_CRank | block_bootstrap_len48 | `-0.003427` | [-0.004200, -0.002306] | True |
| ExchangeRate | LearnedProbeRank_vs_CRank | every_12th_window_phase_bootstrap | `-0.003426` | [-0.003748, -0.003108] | True |
| ExchangeRate | LearnedProbeRank_vs_FixedDRank | iid_paired_bootstrap | `-0.003341` | [-0.003832, -0.002893] | True |
| ExchangeRate | LearnedProbeRank_vs_FixedDRank | block_bootstrap_len12 | `-0.003341` | [-0.004384, -0.002460] | True |
| ExchangeRate | LearnedProbeRank_vs_FixedDRank | block_bootstrap_len24 | `-0.003341` | [-0.004627, -0.002327] | True |
| ExchangeRate | LearnedProbeRank_vs_FixedDRank | block_bootstrap_len48 | `-0.003341` | [-0.005067, -0.002220] | True |
| ExchangeRate | LearnedProbeRank_vs_FixedDRank | every_12th_window_phase_bootstrap | `-0.003340` | [-0.003749, -0.002908] | True |
| ExchangeRate | LearnedProbeRank_vs_Equal | iid_paired_bootstrap | `-0.004399` | [-0.004873, -0.003933] | True |
| ExchangeRate | LearnedProbeRank_vs_Equal | block_bootstrap_len12 | `-0.004399` | [-0.005619, -0.003286] | True |
| ExchangeRate | LearnedProbeRank_vs_Equal | block_bootstrap_len24 | `-0.004399` | [-0.005797, -0.003152] | True |
| ExchangeRate | LearnedProbeRank_vs_Equal | block_bootstrap_len48 | `-0.004399` | [-0.006164, -0.002978] | True |
| ExchangeRate | LearnedProbeRank_vs_Equal | every_12th_window_phase_bootstrap | `-0.004397` | [-0.004725, -0.004092] | True |
| Traffic | LearnedProbeRank_vs_CRank | iid_paired_bootstrap | `-0.001064` | [-0.001567, -0.000555] | True |
| Traffic | LearnedProbeRank_vs_CRank | block_bootstrap_len12 | `-0.001064` | [-0.002175, +0.000124] | False |
| Traffic | LearnedProbeRank_vs_CRank | block_bootstrap_len24 | `-0.001064` | [-0.002278, +0.000302] | False |
| Traffic | LearnedProbeRank_vs_CRank | block_bootstrap_len48 | `-0.001064` | [-0.002319, +0.000499] | False |
| Traffic | LearnedProbeRank_vs_CRank | every_12th_window_phase_bootstrap | `-0.001067` | [-0.002167, +0.000080] | False |
| Traffic | LearnedProbeRank_vs_FixedDRank | iid_paired_bootstrap | `+0.000876` | [+0.000330, +0.001419] | True |
| Traffic | LearnedProbeRank_vs_FixedDRank | block_bootstrap_len12 | `+0.000876` | [-0.000276, +0.002094] | False |
| Traffic | LearnedProbeRank_vs_FixedDRank | block_bootstrap_len24 | `+0.000876` | [-0.000405, +0.002257] | False |
| Traffic | LearnedProbeRank_vs_FixedDRank | block_bootstrap_len48 | `+0.000876` | [-0.000555, +0.002498] | False |
| Traffic | LearnedProbeRank_vs_FixedDRank | every_12th_window_phase_bootstrap | `+0.000875` | [+0.000065, +0.001698] | True |
| Traffic | LearnedProbeRank_vs_Equal | iid_paired_bootstrap | `-0.007732` | [-0.008184, -0.007274] | True |
| Traffic | LearnedProbeRank_vs_Equal | block_bootstrap_len12 | `-0.007732` | [-0.008703, -0.006740] | True |
| Traffic | LearnedProbeRank_vs_Equal | block_bootstrap_len24 | `-0.007732` | [-0.008822, -0.006656] | True |
| Traffic | LearnedProbeRank_vs_Equal | block_bootstrap_len48 | `-0.007732` | [-0.008959, -0.006487] | True |
| Traffic | LearnedProbeRank_vs_Equal | every_12th_window_phase_bootstrap | `-0.007732` | [-0.008588, -0.006812] | True |
| BeijingAirQuality | LearnedProbeRank_vs_CRank | iid_paired_bootstrap | `-0.000279` | [-0.000642, +0.000106] | False |
| BeijingAirQuality | LearnedProbeRank_vs_CRank | block_bootstrap_len12 | `-0.000279` | [-0.000887, +0.000425] | False |
| BeijingAirQuality | LearnedProbeRank_vs_CRank | block_bootstrap_len24 | `-0.000279` | [-0.000872, +0.000349] | False |
| BeijingAirQuality | LearnedProbeRank_vs_CRank | block_bootstrap_len48 | `-0.000279` | [-0.000853, +0.000339] | False |
| BeijingAirQuality | LearnedProbeRank_vs_CRank | every_12th_window_phase_bootstrap | `-0.000279` | [-0.000458, -0.000111] | True |
| BeijingAirQuality | LearnedProbeRank_vs_FixedDRank | iid_paired_bootstrap | `-0.000085` | [-0.000496, +0.000326] | False |
| BeijingAirQuality | LearnedProbeRank_vs_FixedDRank | block_bootstrap_len12 | `-0.000085` | [-0.000778, +0.000670] | False |
| BeijingAirQuality | LearnedProbeRank_vs_FixedDRank | block_bootstrap_len24 | `-0.000085` | [-0.000792, +0.000683] | False |
| BeijingAirQuality | LearnedProbeRank_vs_FixedDRank | block_bootstrap_len48 | `-0.000085` | [-0.000802, +0.000640] | False |
| BeijingAirQuality | LearnedProbeRank_vs_FixedDRank | every_12th_window_phase_bootstrap | `-0.000085` | [-0.000544, +0.000394] | False |
| BeijingAirQuality | LearnedProbeRank_vs_Equal | iid_paired_bootstrap | `+0.000509` | [+0.000122, +0.000900] | True |
| BeijingAirQuality | LearnedProbeRank_vs_Equal | block_bootstrap_len12 | `+0.000509` | [-0.000316, +0.001330] | False |
| BeijingAirQuality | LearnedProbeRank_vs_Equal | block_bootstrap_len24 | `+0.000509` | [-0.000362, +0.001343] | False |
| BeijingAirQuality | LearnedProbeRank_vs_Equal | block_bootstrap_len48 | `+0.000509` | [-0.000401, +0.001310] | False |
| BeijingAirQuality | LearnedProbeRank_vs_Equal | every_12th_window_phase_bootstrap | `+0.000510` | [+0.000025, +0.001009] | True |
| ETTm2 | LearnedProbeRank_vs_CRank | iid_paired_bootstrap | `-0.000344` | [-0.000502, -0.000182] | True |
| ETTm2 | LearnedProbeRank_vs_CRank | block_bootstrap_len12 | `-0.000344` | [-0.000620, -0.000079] | True |
| ETTm2 | LearnedProbeRank_vs_CRank | block_bootstrap_len24 | `-0.000344` | [-0.000631, -0.000060] | True |
| ETTm2 | LearnedProbeRank_vs_CRank | block_bootstrap_len48 | `-0.000344` | [-0.000628, -0.000070] | True |
| ETTm2 | LearnedProbeRank_vs_CRank | every_12th_window_phase_bootstrap | `-0.000344` | [-0.000523, -0.000173] | True |
| ETTm2 | LearnedProbeRank_vs_FixedDRank | iid_paired_bootstrap | `-0.000652` | [-0.000814, -0.000493] | True |
| ETTm2 | LearnedProbeRank_vs_FixedDRank | block_bootstrap_len12 | `-0.000652` | [-0.000932, -0.000389] | True |
| ETTm2 | LearnedProbeRank_vs_FixedDRank | block_bootstrap_len24 | `-0.000652` | [-0.000950, -0.000373] | True |
| ETTm2 | LearnedProbeRank_vs_FixedDRank | block_bootstrap_len48 | `-0.000652` | [-0.000954, -0.000382] | True |
| ETTm2 | LearnedProbeRank_vs_FixedDRank | every_12th_window_phase_bootstrap | `-0.000652` | [-0.000794, -0.000516] | True |
| ETTm2 | LearnedProbeRank_vs_Equal | iid_paired_bootstrap | `+0.000455` | [+0.000343, +0.000569] | True |
| ETTm2 | LearnedProbeRank_vs_Equal | block_bootstrap_len12 | `+0.000455` | [+0.000257, +0.000644] | True |
| ETTm2 | LearnedProbeRank_vs_Equal | block_bootstrap_len24 | `+0.000455` | [+0.000241, +0.000665] | True |
| ETTm2 | LearnedProbeRank_vs_Equal | block_bootstrap_len48 | `+0.000455` | [+0.000234, +0.000662] | True |
| ETTm2 | LearnedProbeRank_vs_Equal | every_12th_window_phase_bootstrap | `+0.000455` | [+0.000371, +0.000537] | True |

## Selected expert core per dataset (router_train only)

- **ExchangeRate**: ['PatchTST', 'iTransformer', 'TimesNet']
- **Traffic**: ['PatchTST', 'iTransformer', 'ModernTCN']
- **BeijingAirQuality**: ['PatchTST', 'iTransformer', 'TimesNet']
- **ETTm2**: ['PatchTST', 'iTransformer', 'TimesNet']

## Integrity

- **ExchangeRate**: PASS (checkpoints unchanged: True; no test cache used: True; weights invariant to target corruption: True)
- **Traffic**: PASS (checkpoints unchanged: True; no test cache used: True; weights invariant to target corruption: True)
- **BeijingAirQuality**: PASS (checkpoints unchanged: True; no test cache used: True; weights invariant to target corruption: True)
- **ETTm2**: PASS (checkpoints unchanged: True; no test cache used: True; weights invariant to target corruption: True)

## Answers

**1. Which new datasets were selected and why?** ExchangeRate, Traffic, BeijingAirQuality, ETTm2 -- see generalization/dataset_selection.json for rationale, finalized before any performance was inspected.
**2. Was the frozen protocol followed exactly?** Yes: unmodified train_probe_and_scorer/evaluate_on_val/run_dataset/rule_fixed_rank/select_core_on_router_train were reused; only additive dataset-registry plumbing was added. See FROZEN_METHOD.md.
**3. Does LearnedProbe-Rank beat C-Rank on most new datasets?** By point estimate on 4/4 datasets (majority=3).
**4. Are any wins dependence-aware significant?** Block-bootstrap significant wins on 2/4 datasets vs C-Rank; significant losses on 0/4.
**5. Does LearnedProbe-Rank beat FixedD-Rank?** By point estimate on 3/4; significant on 2/4.
**6. Does learned probing still improve competence prediction?** Spearman or top-1 accuracy at least matches C on 3/4 datasets.
**7. Does the method show any new significant failure cases?** 0 dataset(s) with a significant regression vs C-Rank.
**8. Does it beat Equal on new datasets?** By point estimate on 2/4; significant on 2/4.
**9. Do the new results look consistent with development results?** See per-dataset table; development datasets showed small, mostly non-dominant gains over C-Rank/Equal -- compare magnitude, not just sign.
**10. Is the learned diagnostic-probe contribution generalizing?** STRONG GENERALIZATION.

## Reasoning

- LearnedProbe-Rank beats C-Rank by point estimate on 4/4 datasets (need >= majority=3 for Strong).
- Dependence-aware significant wins vs C-Rank on 2/4 datasets (need >=2 for Strong).
- Dependence-aware significant losses vs C-Rank on 0/4 datasets (need 0 for Strong; >=2 triggers Failure).
- Beats FixedD-Rank by point estimate on 3/4, significant on 2/4 (preferred, not required).
- Beats Equal by point estimate on 2/4, significant on 2/4 (required for Very Strong, alongside consistent competence-metric improvement).
- Competence metrics (Spearman or top-1) at least match C-Rank's scorer on 3/4 datasets.

## Generalization tier: STRONG GENERALIZATION

## Recommendation: **PROCEED TO LOCKED FINAL TEST**

## Hard rule compliance

```text
TEST SET ACCESSED: NO (no new dataset's test cache was built or loaded)
METHOD MODIFIED AFTER FREEZE: NO
```
