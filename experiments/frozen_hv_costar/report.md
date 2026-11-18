# Frozen HxV COSTAR (removing deployment-time target feedback)

New, separate experiment. The existing online/causal COSTAR implementation is unmodified and evaluated here as `online_hv`.

## Step 4/5: validation results

| Dataset | Best Single | Equal Fixed | Frozen HxV (NEW) | Online HxV (existing) |
|---|---:|---:|---:|---:|
| ETTh1 | `0.379558`/`0.326528` | `0.367265`/`0.310530` | `0.366022`/`0.308672` | `0.363949`/`0.307478` |
| ETTh2 | `0.280957`/`0.171493` | `0.280878`/`0.171933` | `0.276898`/`0.167005` | `0.276354`/`0.167381` |
| ETTm1 | `0.261771`/`0.161838` | `0.248161`/`0.146694` | `0.250690`/`0.149956` | `0.248593`/`0.148334` |
| Weather | `0.164673`/`0.287468` | `0.160341`/`0.278815` | `0.159818`/`0.279092` | `0.159280`/`0.283565` |
| Electricity | `0.225385`/`0.135767` | `0.214457`/`0.117846` | `0.215355`/`0.122737` | `0.211775`/`0.119014` |

## Deltas (IID paired bootstrap, quick reference)

| Dataset | Comparison | Delta MAE | 95% CI | Excludes zero |
|---|---|---:|---|---|
| ETTh1 | frozen_hv_vs_equal | `-0.001242` | [-0.001888, -0.000565] | True |
| ETTh1 | online_hv_vs_equal | `-0.003316` | [-0.004085, -0.002553] | True |
| ETTh1 | frozen_hv_vs_online_hv | `+0.002074` | [+0.001542, +0.002622] | True |
| ETTh2 | frozen_hv_vs_equal | `-0.003980` | [-0.004797, -0.003184] | True |
| ETTh2 | online_hv_vs_equal | `-0.004525` | [-0.005816, -0.003284] | True |
| ETTh2 | frozen_hv_vs_online_hv | `+0.000544` | [-0.000076, +0.001190] | False |
| ETTm1 | frozen_hv_vs_equal | `+0.002528` | [+0.002245, +0.002819] | True |
| ETTm1 | online_hv_vs_equal | `+0.000431` | [+0.000112, +0.000747] | True |
| ETTm1 | frozen_hv_vs_online_hv | `+0.002097` | [+0.001816, +0.002370] | True |
| Weather | frozen_hv_vs_equal | `-0.000524` | [-0.000789, -0.000264] | True |
| Weather | online_hv_vs_equal | `-0.001061` | [-0.001466, -0.000673] | True |
| Weather | frozen_hv_vs_online_hv | `+0.000537` | [+0.000302, +0.000772] | True |
| Electricity | frozen_hv_vs_equal | `+0.000898` | [+0.000687, +0.001104] | True |
| Electricity | online_hv_vs_equal | `-0.002682` | [-0.002887, -0.002470] | True |
| Electricity | frozen_hv_vs_online_hv | `+0.003580` | [+0.003388, +0.003779] | True |

## Step 6: frozen-behavior verification (A-E)

| Dataset | Method | A: early target | B: all targets | C: no targets loaded | D: order-invariant | E: state unchanged | All pass |
|---|---|---|---|---|---|---|---|
| ETTh1 | best_single_expert | True | True | True & matches=True | True | True | True |
| ETTh1 | equal_fixed | True | True | True & matches=True | True | True | True |
| ETTh1 | frozen_hv | True | True | True & matches=True | True | True | True |
| ETTh1 | online_hv | False | False | False & matches=False | None | False | None |
| ETTh2 | best_single_expert | True | True | True & matches=True | True | True | True |
| ETTh2 | equal_fixed | True | True | True & matches=True | True | True | True |
| ETTh2 | frozen_hv | True | True | True & matches=True | True | True | True |
| ETTh2 | online_hv | False | False | False & matches=False | None | False | None |
| ETTm1 | best_single_expert | True | True | True & matches=True | True | True | True |
| ETTm1 | equal_fixed | True | True | True & matches=True | True | True | True |
| ETTm1 | frozen_hv | True | True | True & matches=True | True | True | True |
| ETTm1 | online_hv | False | False | False & matches=False | None | False | None |
| Weather | best_single_expert | True | True | True & matches=True | True | True | True |
| Weather | equal_fixed | True | True | True & matches=True | True | True | True |
| Weather | frozen_hv | True | True | True & matches=True | True | True | True |
| Weather | online_hv | False | False | False & matches=False | None | False | None |
| Electricity | best_single_expert | True | True | True & matches=True | True | True | True |
| Electricity | equal_fixed | True | True | True & matches=True | True | True | True |
| Electricity | frozen_hv | True | True | True & matches=True | True | True | True |
| Electricity | online_hv | False | False | False & matches=False | None | False | None |

## Hard rule compliance

```text
TEST SET ACCESSED: NO
TEST CACHE LOADED: NO
TEST METRICS COMPUTED: NO
```
