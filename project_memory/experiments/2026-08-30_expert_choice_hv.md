# Expert-Choice Horizon-Variable Routing

Date: 2026-08-30

Status: Completed validation-only

Artifacts:

- `experiments/expert_choice_hv/run_expert_choice_hv.py`
- `experiments/expert_choice_hv/method_manifest.json`
- `experiments/expert_choice_hv/results.json`
- `experiments/expert_choice_hv/results.csv`
- `experiments/expert_choice_hv/assignment_stats.csv`
- `experiments/expert_choice_hv/dependence_tests.csv`
- `experiments/expert_choice_hv/integrity_checks.json`
- `experiments/expert_choice_hv/report.md`

## Question

If routing direction is reversed so each frozen heterogeneous forecasting expert chooses the horizon x variable cells where it is most competent, does that produce better specialization than the usual approach where each horizon x variable cell chooses experts?

## Protocol

This was a static, validation-only structure test. No neural router or learned competence model was added.

Datasets: `ETTh1`, `ETTh2`, `ETTm1`, `Weather`, `Electricity`.

Selected frozen cores reused from existing frozen-HxV loaders:

- ETTh1: `PatchTST+iTransformer+TimesNet`
- ETTh2: `DLinear+PatchTST+ModernTCN`
- ETTm1: `DLinear+PatchTST+TimesNet`
- Weather: `PatchTST+iTransformer+TimesNet`
- Electricity: `PatchTST+iTransformer+TimesNet`

Train-only score:

```text
score[h,v,e] = mean_t(equal_error[t,h,v] - expert_error[t,h,v,e])
```

where `expert_error` and `equal_error` are normalized absolute errors computed only on router_train. Positive score means the expert is better than the equal ensemble at that train HxV location.

Expert Choice capacities were predeclared:

- `CF=1.0` primary, capacity `round(M/E)` per expert
- `CF=2.0` secondary, capacity `round(2M/E)` per expert

For each expert, EC-HVR ranked all HxV cells by `score[:,:,e]` and claimed its top-capacity cells. Multiple experts could claim the same cell; zero-claim cells fell back to the equal fixed ensemble. Matched TokenChoice controls used the exact same score tensor with cell-to-expert Top1 and Top2 choices.

## Main Results

Final classification: `MIXED_EXPERT_CHOICE`.

| Dataset | Best Single | Equal | Frozen HxV | Token Top1 | EC CF1 | Token Top2 | EC CF2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ETTh1 | `0.379558` | `0.367265` | `0.366022` | `0.376680` | `0.375352` | `0.369798` | `0.375002` |
| ETTh2 | `0.280957` | `0.280878` | `0.276898` | `0.279132` | `0.277764` | `0.275230` | `0.276706` |
| ETTm1 | `0.261771` | `0.248161` | `0.250690` | `0.262524` | `0.253570` | `0.256256` | `0.259210` |
| Weather | `0.164673` | `0.160341` | `0.159818` | `0.164399` | `0.160330` | `0.162339` | `0.159962` |
| Electricity | `0.225385` | `0.214457` | `0.215355` | `0.222761` | `0.219015` | `0.219130` | `0.216481` |

Matched-budget deltas, negative means EC-HVR is better:

- EC CF1 beat Token Top1 on `5/5` datasets by point estimate.
- EC CF2 beat Token Top2 on `2/5` datasets by point estimate.
- EC CF1 was worse than Frozen HxV on `5/5` datasets.

Block-24 dependence support:

- EC CF1 vs Token Top1: supported on ETTm1 (`-0.008954`, CI `[-0.010476,-0.007360]`), Weather (`-0.004069`, CI `[-0.005095,-0.002977]`), and Electricity (`-0.003746`, CI `[-0.005365,-0.002165]`); not supported on ETTh1/ETTh2.
- EC CF2 vs Token Top2: supported positively only on Weather (`-0.002377`, CI `[-0.003350,-0.001312]`) and Electricity (`-0.002649`, CI `[-0.004289,-0.000959]`); ETTh1 and ETTm1 significantly regressed, ETTh2 was not supported.
- EC CF1 vs Frozen HxV: EC CF1 did not beat Frozen HxV on any dataset and was significantly worse on ETTh1, ETTm1, and Electricity.

## Specialization

The EC claim masks were non-identical and passed the predeclared nontrivial-specialization rule: average pairwise EC claim-mask Jaccard `< 0.98` for CF1 and CF2 on every dataset.

Average pairwise Jaccard:

- ETTh1: CF1 `0.138`, CF2 `0.525`
- ETTh2: CF1 `0.177`, CF2 `0.565`
- ETTm1: CF1 `0.328`, CF2 `0.510`
- Weather: CF1 `0.253`, CF2 `0.622`
- Electricity: CF1 `0.170`, CF2 `0.534`

Fallback rates:

- CF1: ETTh1 `21.4%`, ETTh2 `28.6%`, ETTm1 `36.9%`, Weather `34.9%`, Electricity `24.1%`
- CF2: ETTh1 `1.2%`, ETTh2 `8.3%`, ETTm1 `0.0%`, Weather `11.9%`, Electricity `3.1%`

## Integrity

All EC CF1/CF2 checks passed on every dataset:

- Router-val target corruption left predictions and claim masks identical.
- Targetless router-val prediction succeeded and matched.
- Validation-order permutation/unpermutation preserved predictions.
- Train-derived allocation was identical before and after evaluation.
- Cache roles and paths rejected any `"test"` access.

```text
TEST SET ACCESSED: NO
TEST CACHE LOADED: NO
TEST METRICS COMPUTED: NO
```

## Conclusion

Experts did develop distinct train-derived HxV claim regions, and the primary CF1 expert-to-cell direction beat matched Token Top1 on all datasets. However, CF2 did not generalize, and EC-HVR was worse than the existing Frozen HxV baseline everywhere. This supports expert-choice allocation as a useful diagnostic or constraint idea, but the static reversed routing structure alone is not strong enough to justify an input-dependent learned Expert-Choice router yet.
