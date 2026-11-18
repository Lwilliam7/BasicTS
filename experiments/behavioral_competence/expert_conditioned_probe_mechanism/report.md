SCIENTIFIC QUESTION:
Why did the original expert-conditioned LearnedProbe work?

# Expert-Conditioned Probe Mechanism Ablation

## Exact four-arm method definition

- C-Rank / Passive baseline: 15 passive A+B+C features, matched scorer, fixed rank weights [0.5, 1/3, 1/6].
- Matched Neural Passive: same pre-query inputs as ProbeGenerator, six learned z features, no perturbation and no perturbed expert call.
- Delta-Only: original ProbeGenerator creates expert-conditioned delta_k, summarized into six fixed delta statistics; no expert(x+delta) call.
- Original LearnedProbe: original expert-conditioned delta_k, frozen expert(x+delta_k), six probe_response_features.

## Primary results table

| Dataset | C-Rank MAE | MatchedNeural MAE | DeltaOnly MAE | Original MAE | Original-Matched | Original-Delta |
|---|---:|---:|---:|---:|---:|---:|
| ETTh1 | 0.366585 | 0.366622 | 0.368802 | 0.368571 | `+0.001950` | `-0.000231` |
| ETTh2 | 0.277266 | 0.277718 | 0.279790 | 0.279602 | `+0.001884` | `-0.000189` |
| ETTm1 | 0.248902 | 0.249784 | 0.249070 | 0.249389 | `-0.000396` | `+0.000319` |
| Weather | 0.159444 | 0.159366 | 0.159293 | 0.159268 | `-0.000098` | `-0.000026` |
| Electricity | 0.214649 | 0.213496 | 0.214167 | 0.214210 | `+0.000714` | `+0.000043` |

## Competence and residual-information analysis

| Dataset | Method | Spearman | Pairwise acc | Top-1 acc | Top-2 recall |
|---|---|---:|---:|---:|---:|
| ETTh1 | C_Rank_Passive | 0.231 | 0.605 | 0.386 | 0.720 |
| ETTh1 | MatchedNeuralPassive | 0.188 | 0.603 | 0.371 | 0.713 |
| ETTh1 | DeltaOnly | 0.183 | 0.567 | 0.339 | 0.683 |
| ETTh1 | OriginalLearnedProbe | 0.187 | 0.572 | 0.339 | 0.690 |
| ETTh2 | C_Rank_Passive | 0.356 | 0.704 | 0.486 | 0.856 |
| ETTh2 | MatchedNeuralPassive | 0.213 | 0.671 | 0.463 | 0.819 |
| ETTh2 | DeltaOnly | 0.130 | 0.595 | 0.409 | 0.778 |
| ETTh2 | OriginalLearnedProbe | 0.134 | 0.597 | 0.411 | 0.778 |
| ETTm1 | C_Rank_Passive | 0.163 | 0.567 | 0.344 | 0.672 |
| ETTm1 | MatchedNeuralPassive | 0.138 | 0.549 | 0.328 | 0.644 |
| ETTm1 | DeltaOnly | 0.172 | 0.564 | 0.339 | 0.669 |
| ETTm1 | OriginalLearnedProbe | 0.163 | 0.561 | 0.333 | 0.658 |
| Weather | C_Rank_Passive | 0.322 | 0.635 | 0.436 | 0.733 |
| Weather | MatchedNeuralPassive | 0.309 | 0.635 | 0.435 | 0.734 |
| Weather | DeltaOnly | 0.306 | 0.635 | 0.436 | 0.734 |
| Weather | OriginalLearnedProbe | 0.307 | 0.636 | 0.441 | 0.734 |
| Electricity | C_Rank_Passive | 0.639 | 0.791 | 0.613 | 0.892 |
| Electricity | MatchedNeuralPassive | 0.681 | 0.810 | 0.630 | 0.929 |
| Electricity | DeltaOnly | 0.585 | 0.757 | 0.511 | 0.906 |
| Electricity | OriginalLearnedProbe | 0.586 | 0.758 | 0.516 | 0.904 |

| Dataset | Added representation | OOF residual R2 | OOF residual MAE |
|---|---|---:|---:|
| ETTh1 | MatchedNeuralPassive | -0.0976 | 0.049185 |
| ETTh1 | DeltaOnly | -0.0501 | 0.050840 |
| ETTh1 | OriginalProbeResponse | -3.1055 | 0.052430 |
| ETTh2 | MatchedNeuralPassive | 0.5402 | 0.055154 |
| ETTh2 | DeltaOnly | -0.5865 | 0.098304 |
| ETTh2 | OriginalProbeResponse | 0.1861 | 0.067833 |
| ETTm1 | MatchedNeuralPassive | -0.0825 | 0.047433 |
| ETTm1 | DeltaOnly | -0.0475 | 0.046331 |
| ETTm1 | OriginalProbeResponse | -0.0270 | 0.046400 |
| Weather | MatchedNeuralPassive | 0.0282 | 0.034272 |
| Weather | DeltaOnly | -0.1881 | 0.036350 |
| Weather | OriginalProbeResponse | -7.3352 | 0.038145 |
| Electricity | MatchedNeuralPassive | -0.0245 | 0.026444 |
| Electricity | DeltaOnly | 0.0172 | 0.026070 |
| Electricity | OriginalProbeResponse | 0.0391 | 0.025752 |

## Dependence-aware statistics

| Dataset | Comparison | Test | Mean delta | 95% CI | Excludes zero |
|---|---|---|---:|---|---|
| ETTh1 | Original_vs_MatchedNeuralPassive | iid_paired_bootstrap | `+0.001950` | [+0.001443, +0.002456] | True |
| ETTh1 | Original_vs_MatchedNeuralPassive | block_bootstrap_len12 | `+0.001950` | [+0.000980, +0.003059] | True |
| ETTh1 | Original_vs_MatchedNeuralPassive | block_bootstrap_len24 | `+0.001950` | [+0.000851, +0.003280] | True |
| ETTh1 | Original_vs_MatchedNeuralPassive | block_bootstrap_len48 | `+0.001950` | [+0.000717, +0.003601] | True |
| ETTh1 | Original_vs_MatchedNeuralPassive | every_12th_window_phase_bootstrap | `+0.001949` | [+0.001516, +0.002380] | True |
| ETTh1 | Original_vs_DeltaOnly | iid_paired_bootstrap | `-0.000231` | [-0.000495, +0.000027] | False |
| ETTh1 | Original_vs_DeltaOnly | block_bootstrap_len12 | `-0.000231` | [-0.000628, +0.000147] | False |
| ETTh1 | Original_vs_DeltaOnly | block_bootstrap_len24 | `-0.000231` | [-0.000670, +0.000170] | False |
| ETTh1 | Original_vs_DeltaOnly | block_bootstrap_len48 | `-0.000231` | [-0.000711, +0.000224] | False |
| ETTh1 | Original_vs_DeltaOnly | every_12th_window_phase_bootstrap | `-0.000231` | [-0.000468, -0.000005] | True |
| ETTh1 | MatchedNeuralPassive_vs_CRank | iid_paired_bootstrap | `+0.000037` | [-0.000333, +0.000417] | False |
| ETTh1 | MatchedNeuralPassive_vs_CRank | block_bootstrap_len12 | `+0.000037` | [-0.000550, +0.000567] | False |
| ETTh1 | MatchedNeuralPassive_vs_CRank | block_bootstrap_len24 | `+0.000037` | [-0.000576, +0.000592] | False |
| ETTh1 | MatchedNeuralPassive_vs_CRank | block_bootstrap_len48 | `+0.000037` | [-0.000618, +0.000639] | False |
| ETTh1 | MatchedNeuralPassive_vs_CRank | every_12th_window_phase_bootstrap | `+0.000037` | [-0.000253, +0.000345] | False |
| ETTh1 | DeltaOnly_vs_CRank | iid_paired_bootstrap | `+0.002218` | [+0.001645, +0.002821] | True |
| ETTh1 | DeltaOnly_vs_CRank | block_bootstrap_len12 | `+0.002218` | [+0.001008, +0.003504] | True |
| ETTh1 | DeltaOnly_vs_CRank | block_bootstrap_len24 | `+0.002218` | [+0.000806, +0.003792] | True |
| ETTh1 | DeltaOnly_vs_CRank | block_bootstrap_len48 | `+0.002218` | [+0.000586, +0.004228] | True |
| ETTh1 | DeltaOnly_vs_CRank | every_12th_window_phase_bootstrap | `+0.002217` | [+0.001655, +0.002703] | True |
| ETTh2 | Original_vs_MatchedNeuralPassive | iid_paired_bootstrap | `+0.001884` | [+0.001201, +0.002588] | True |
| ETTh2 | Original_vs_MatchedNeuralPassive | block_bootstrap_len12 | `+0.001884` | [+0.000936, +0.002983] | True |
| ETTh2 | Original_vs_MatchedNeuralPassive | block_bootstrap_len24 | `+0.001884` | [+0.000942, +0.003067] | True |
| ETTh2 | Original_vs_MatchedNeuralPassive | block_bootstrap_len48 | `+0.001884` | [+0.000803, +0.003203] | True |
| ETTh2 | Original_vs_MatchedNeuralPassive | every_12th_window_phase_bootstrap | `+0.001884` | [+0.001392, +0.002395] | True |
| ETTh2 | Original_vs_DeltaOnly | iid_paired_bootstrap | `-0.000189` | [-0.000383, -0.000020] | True |
| ETTh2 | Original_vs_DeltaOnly | block_bootstrap_len12 | `-0.000189` | [-0.000387, -0.000019] | True |
| ETTh2 | Original_vs_DeltaOnly | block_bootstrap_len24 | `-0.000189` | [-0.000381, -0.000028] | True |
| ETTh2 | Original_vs_DeltaOnly | block_bootstrap_len48 | `-0.000189` | [-0.000376, -0.000020] | True |
| ETTh2 | Original_vs_DeltaOnly | every_12th_window_phase_bootstrap | `-0.000188` | [-0.000358, -0.000014] | True |
| ETTh2 | MatchedNeuralPassive_vs_CRank | iid_paired_bootstrap | `+0.000451` | [-0.000348, +0.001274] | False |
| ETTh2 | MatchedNeuralPassive_vs_CRank | block_bootstrap_len12 | `+0.000451` | [-0.000763, +0.001732] | False |
| ETTh2 | MatchedNeuralPassive_vs_CRank | block_bootstrap_len24 | `+0.000451` | [-0.000750, +0.001764] | False |
| ETTh2 | MatchedNeuralPassive_vs_CRank | block_bootstrap_len48 | `+0.000451` | [-0.000729, +0.001627] | False |
| ETTh2 | MatchedNeuralPassive_vs_CRank | every_12th_window_phase_bootstrap | `+0.000451` | [-0.000256, +0.001161] | False |
| ETTh2 | DeltaOnly_vs_CRank | iid_paired_bootstrap | `+0.002524` | [+0.001563, +0.003508] | True |
| ETTh2 | DeltaOnly_vs_CRank | block_bootstrap_len12 | `+0.002524` | [+0.001025, +0.004188] | True |
| ETTh2 | DeltaOnly_vs_CRank | block_bootstrap_len24 | `+0.002524` | [+0.001114, +0.004237] | True |
| ETTh2 | DeltaOnly_vs_CRank | block_bootstrap_len48 | `+0.002524` | [+0.001125, +0.004049] | True |
| ETTh2 | DeltaOnly_vs_CRank | every_12th_window_phase_bootstrap | `+0.002522` | [+0.001355, +0.003647] | True |
| ETTm1 | Original_vs_MatchedNeuralPassive | iid_paired_bootstrap | `-0.000396` | [-0.000580, -0.000203] | True |
| ETTm1 | Original_vs_MatchedNeuralPassive | block_bootstrap_len12 | `-0.000396` | [-0.000706, -0.000091] | True |
| ETTm1 | Original_vs_MatchedNeuralPassive | block_bootstrap_len24 | `-0.000396` | [-0.000724, -0.000080] | True |
| ETTm1 | Original_vs_MatchedNeuralPassive | block_bootstrap_len48 | `-0.000396` | [-0.000735, -0.000063] | True |
| ETTm1 | Original_vs_MatchedNeuralPassive | every_12th_window_phase_bootstrap | `-0.000396` | [-0.000625, -0.000161] | True |
| ETTm1 | Original_vs_DeltaOnly | iid_paired_bootstrap | `+0.000319` | [+0.000176, +0.000464] | True |
| ETTm1 | Original_vs_DeltaOnly | block_bootstrap_len12 | `+0.000319` | [+0.000101, +0.000533] | True |
| ETTm1 | Original_vs_DeltaOnly | block_bootstrap_len24 | `+0.000319` | [+0.000085, +0.000553] | True |
| ETTm1 | Original_vs_DeltaOnly | block_bootstrap_len48 | `+0.000319` | [+0.000076, +0.000561] | True |
| ETTm1 | Original_vs_DeltaOnly | every_12th_window_phase_bootstrap | `+0.000319` | [+0.000204, +0.000452] | True |
| ETTm1 | MatchedNeuralPassive_vs_CRank | iid_paired_bootstrap | `+0.000883` | [+0.000658, +0.001102] | True |
| ETTm1 | MatchedNeuralPassive_vs_CRank | block_bootstrap_len12 | `+0.000883` | [+0.000508, +0.001291] | True |
| ETTm1 | MatchedNeuralPassive_vs_CRank | block_bootstrap_len24 | `+0.000883` | [+0.000483, +0.001327] | True |
| ETTm1 | MatchedNeuralPassive_vs_CRank | block_bootstrap_len48 | `+0.000883` | [+0.000467, +0.001369] | True |
| ETTm1 | MatchedNeuralPassive_vs_CRank | every_12th_window_phase_bootstrap | `+0.000883` | [+0.000595, +0.001160] | True |
| ETTm1 | DeltaOnly_vs_CRank | iid_paired_bootstrap | `+0.000168` | [-0.000030, +0.000364] | False |
| ETTm1 | DeltaOnly_vs_CRank | block_bootstrap_len12 | `+0.000168` | [-0.000156, +0.000519] | False |
| ETTm1 | DeltaOnly_vs_CRank | block_bootstrap_len24 | `+0.000168` | [-0.000171, +0.000556] | False |
| ETTm1 | DeltaOnly_vs_CRank | block_bootstrap_len48 | `+0.000168` | [-0.000182, +0.000586] | False |
| ETTm1 | DeltaOnly_vs_CRank | every_12th_window_phase_bootstrap | `+0.000168` | [-0.000028, +0.000366] | False |
| Weather | Original_vs_MatchedNeuralPassive | iid_paired_bootstrap | `-0.000098` | [-0.000187, -0.000009] | True |
| Weather | Original_vs_MatchedNeuralPassive | block_bootstrap_len12 | `-0.000098` | [-0.000222, +0.000040] | False |
| Weather | Original_vs_MatchedNeuralPassive | block_bootstrap_len24 | `-0.000098` | [-0.000236, +0.000047] | False |
| Weather | Original_vs_MatchedNeuralPassive | block_bootstrap_len48 | `-0.000098` | [-0.000253, +0.000052] | False |
| Weather | Original_vs_MatchedNeuralPassive | every_12th_window_phase_bootstrap | `-0.000098` | [-0.000168, -0.000025] | True |
| Weather | Original_vs_DeltaOnly | iid_paired_bootstrap | `-0.000026` | [-0.000089, +0.000039] | False |
| Weather | Original_vs_DeltaOnly | block_bootstrap_len12 | `-0.000026` | [-0.000102, +0.000062] | False |
| Weather | Original_vs_DeltaOnly | block_bootstrap_len24 | `-0.000026` | [-0.000101, +0.000062] | False |
| Weather | Original_vs_DeltaOnly | block_bootstrap_len48 | `-0.000026` | [-0.000099, +0.000061] | False |
| Weather | Original_vs_DeltaOnly | every_12th_window_phase_bootstrap | `-0.000026` | [-0.000075, +0.000033] | False |
| Weather | MatchedNeuralPassive_vs_CRank | iid_paired_bootstrap | `-0.000078` | [-0.000186, +0.000032] | False |
| Weather | MatchedNeuralPassive_vs_CRank | block_bootstrap_len12 | `-0.000078` | [-0.000274, +0.000111] | False |
| Weather | MatchedNeuralPassive_vs_CRank | block_bootstrap_len24 | `-0.000078` | [-0.000284, +0.000136] | False |
| Weather | MatchedNeuralPassive_vs_CRank | block_bootstrap_len48 | `-0.000078` | [-0.000286, +0.000148] | False |
| Weather | MatchedNeuralPassive_vs_CRank | every_12th_window_phase_bootstrap | `-0.000078` | [-0.000166, +0.000000] | False |
| Weather | DeltaOnly_vs_CRank | iid_paired_bootstrap | `-0.000151` | [-0.000261, -0.000047] | True |
| Weather | DeltaOnly_vs_CRank | block_bootstrap_len12 | `-0.000151` | [-0.000341, +0.000032] | False |
| Weather | DeltaOnly_vs_CRank | block_bootstrap_len24 | `-0.000151` | [-0.000357, +0.000043] | False |
| Weather | DeltaOnly_vs_CRank | block_bootstrap_len48 | `-0.000151` | [-0.000358, +0.000047] | False |
| Weather | DeltaOnly_vs_CRank | every_12th_window_phase_bootstrap | `-0.000151` | [-0.000251, -0.000049] | True |
| Electricity | Original_vs_MatchedNeuralPassive | iid_paired_bootstrap | `+0.000714` | [+0.000555, +0.000869] | True |
| Electricity | Original_vs_MatchedNeuralPassive | block_bootstrap_len12 | `+0.000714` | [+0.000469, +0.000966] | True |
| Electricity | Original_vs_MatchedNeuralPassive | block_bootstrap_len24 | `+0.000714` | [+0.000461, +0.000988] | True |
| Electricity | Original_vs_MatchedNeuralPassive | block_bootstrap_len48 | `+0.000714` | [+0.000425, +0.001033] | True |
| Electricity | Original_vs_MatchedNeuralPassive | every_12th_window_phase_bootstrap | `+0.000714` | [+0.000510, +0.000922] | True |
| Electricity | Original_vs_DeltaOnly | iid_paired_bootstrap | `+0.000043` | [-0.000031, +0.000120] | False |
| Electricity | Original_vs_DeltaOnly | block_bootstrap_len12 | `+0.000043` | [-0.000023, +0.000111] | False |
| Electricity | Original_vs_DeltaOnly | block_bootstrap_len24 | `+0.000043` | [-0.000022, +0.000111] | False |
| Electricity | Original_vs_DeltaOnly | block_bootstrap_len48 | `+0.000043` | [-0.000021, +0.000110] | False |
| Electricity | Original_vs_DeltaOnly | every_12th_window_phase_bootstrap | `+0.000043` | [-0.000046, +0.000159] | False |
| Electricity | MatchedNeuralPassive_vs_CRank | iid_paired_bootstrap | `-0.001153` | [-0.001364, -0.000947] | True |
| Electricity | MatchedNeuralPassive_vs_CRank | block_bootstrap_len12 | `-0.001153` | [-0.001565, -0.000779] | True |
| Electricity | MatchedNeuralPassive_vs_CRank | block_bootstrap_len24 | `-0.001153` | [-0.001585, -0.000760] | True |
| Electricity | MatchedNeuralPassive_vs_CRank | block_bootstrap_len48 | `-0.001153` | [-0.001579, -0.000767] | True |
| Electricity | MatchedNeuralPassive_vs_CRank | every_12th_window_phase_bootstrap | `-0.001153` | [-0.001444, -0.000851] | True |
| Electricity | DeltaOnly_vs_CRank | iid_paired_bootstrap | `-0.000482` | [-0.000652, -0.000323] | True |
| Electricity | DeltaOnly_vs_CRank | block_bootstrap_len12 | `-0.000482` | [-0.000830, -0.000198] | True |
| Electricity | DeltaOnly_vs_CRank | block_bootstrap_len24 | `-0.000482` | [-0.000852, -0.000181] | True |
| Electricity | DeltaOnly_vs_CRank | block_bootstrap_len48 | `-0.000482` | [-0.000827, -0.000203] | True |
| Electricity | DeltaOnly_vs_CRank | every_12th_window_phase_bootstrap | `-0.000482` | [-0.000628, -0.000337] | True |

## Integrity checks

- **ETTh1**: PASS (checkpoints unchanged: True; no expert updates: True; target-corruption invariant: True; purge correct: True; rank weights exact: True)
- **ETTh2**: PASS (checkpoints unchanged: True; no expert updates: True; target-corruption invariant: True; purge correct: True; rank weights exact: True)
- **ETTm1**: PASS (checkpoints unchanged: True; no expert updates: True; target-corruption invariant: True; purge correct: True; rank weights exact: True)
- **Weather**: PASS (checkpoints unchanged: True; no expert updates: True; target-corruption invariant: True; purge correct: True; rank weights exact: True)
- **Electricity**: PASS (checkpoints unchanged: True; no expert updates: True; target-corruption invariant: True; purge correct: True; rank weights exact: True)

## Final mechanism classification

**GENERATOR_IS_PASSIVE_ENCODER**

## Explicit answer

Does querying the frozen expert actually provide information that cannot be obtained from the same passive inputs alone? No clear evidence: the matched passive neural encoder approximately matches or beats Original LearnedProbe.

## Hard rule compliance

```text
TEST SET ACCESSED: NO
TEST CACHE LOADED: NO
FORECASTING EXPERTS RETRAINED: NO
RANK WEIGHTS: [0.5, 1/3, 1/6]
```
