# Multi-Query Random Probe

Scientific question: Does using several independent controlled random perturbations provide more expert-specific competence information than one perturbation?

**Classification: SINGLE_QUERY_NOT_THE_PROBLEM**

Four random queries do not materially improve over one query and there is still no incremental information beyond passive features.

## Router-Val Results

| Dataset | Method | MAE | MSE | R2 | Pearson | Spearman | Pairwise acc | Top-1 acc |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ETTm2 | SingleRandomProbe | 0.051203 | 0.005025 | -0.0319 | 0.1507 | 0.2062 | 0.558 | 0.457 |
| ETTm2 | MultiRandomProbe4 | 0.051086 | 0.005012 | -0.0293 | 0.1713 | 0.2503 | 0.579 | 0.499 |
| ETTm2 | MultiRandomProbe4-Relative | 0.051645 | 0.005165 | -0.0608 | -0.0309 | -0.0934 | 0.415 | 0.246 |
| ETTm2 | ShuffledMultiRandom | 0.053011 | 0.005403 | -0.1095 | 0.1267 | 0.1785 | 0.462 | 0.249 |
| ETTm2 | MatchedPassive | 0.047096 | 0.004752 | 0.0241 | 0.2519 | 0.2465 | 0.391 | 0.225 |
| ETTm2 | PassivePlusMulti | 0.047172 | 0.004744 | 0.0258 | 0.2546 | 0.2508 | 0.398 | 0.233 |
| ETTm2 | PassivePlusRelativeMulti | 0.047051 | 0.004747 | 0.0251 | 0.2530 | 0.2481 | 0.395 | 0.230 |

## Counts

- `multi_over_single`: `1/1`
- `multi_over_shuffled`: `1/1`
- `relative_over_multi`: `0/1`
- `passive_plus_multi_over_passive`: `0/1`
- `passive_plus_relative_over_passive`: `1/1`
- `multi_residual_positive`: `1/1`
- `relative_residual_positive`: `1/1`

## Compliance

```text
M_QUERIES_TUNED: NO (fixed at 4)
QUERY_SEEDS_TUNED: NO
EPSILON_TUNED: NO (fixed at 0.05)
RIDGE_ALPHA_TUNED: NO (fixed at 1.0)
PERTURBATION_GENERATOR_TRAINED: NO
ROUTER TRAINED: NO
TEST SET ACCESSED: NO
```
