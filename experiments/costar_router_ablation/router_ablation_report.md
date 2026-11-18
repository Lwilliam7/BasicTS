# COSTAR Router Ablation Study

Isolates the router: every row below is a base-mixture prediction only (no DLinear/ModernTCN specialist correction, no Ridge/MLP residual correction). All ablations reuse the existing `chronological_hv_weights` / `errors_to_weights` machinery unmodified, varying only the aggregation granularity (`trial.mode`); ablation 6 reuses the production `parameterized_current_base_prediction` / `current_base_prediction` verbatim. Canonical settings only -- nothing tuned per ablation:

- HxV family: decay=0.95, temperature=0.1, low-rank rank=1
- Chrono branch (ablation 6 only): decay=0.97, temperature=0.1, static blend=0.5
- Global+HxV blend (ablation 6 only): {'chrono': 0.25, 'hv': 0.75}

Git commit: `afa9731f518997b1772580c4e41fa5f40b6e2dbb`

## Result table (router_val only)

| Method | ETTh1 MAE | ETTh1 MSE | ETTh2 MAE | ETTh2 MSE |
|---|---:|---:|---:|---:|
| Equal fixed ensemble | `0.367265` | `0.310530` | `0.280878` | `0.171933` |
| Global causal EMA | `0.365755` | `0.308944` | `0.280153` | `0.171443` |
| Horizon-only causal EMA | `0.365638` | `0.309045` | `0.279678` | `0.171424` |
| Variable-only causal EMA | `0.364085` | `0.307298` | `0.277471` | `0.167658` |
| Full horizon x variable causal EMA | `0.363949` | `0.307478` | `0.276354` | `0.167381` |
| Global + HxV COSTAR | `0.363634` | `0.306684` | `0.276832` | `0.167280` |

Ranking by MAE (lower is better):
- **ETTh1**: Global + HxV COSTAR (`0.363634`) < Full horizon x variable causal EMA (`0.363949`) < Variable-only causal EMA (`0.364085`) < Horizon-only causal EMA (`0.365638`) < Global causal EMA (`0.365755`) < Equal fixed ensemble (`0.367265`)
- **ETTh2**: Full horizon x variable causal EMA (`0.276354`) < Global + HxV COSTAR (`0.276832`) < Variable-only causal EMA (`0.277471`) < Horizon-only causal EMA (`0.279678`) < Global causal EMA (`0.280153`) < Equal fixed ensemble (`0.280878`)

## Ablation 7: full HxV vs low-rank HxV

| Dataset | Full HxV MAE | Low-rank HxV MAE | Delta (low-rank minus full) |
|---|---:|---:|---:|
| ETTh1 | `0.363949` | `0.363876` | `-0.000072` |
| ETTh2 | `0.276354` | `0.277148` | `+0.000794` |

## Deltas vs baselines, with paired-bootstrap 95% CI on the difference

| Dataset | Baseline | Method | Delta MAE | 95% CI | CI excludes zero |
|---|---|---|---:|---|---|
| ETTh1 | equal_fixed | Global causal EMA | `-0.001510` | [-0.002161, -0.000851] | True |
| ETTh1 | equal_fixed | Horizon-only causal EMA | `-0.001627` | [-0.002271, -0.000958] | True |
| ETTh1 | equal_fixed | Variable-only causal EMA | `-0.003180` | [-0.003956, -0.002405] | True |
| ETTh1 | equal_fixed | Full horizon x variable causal EMA | `-0.003316` | [-0.004085, -0.002553] | True |
| ETTh1 | equal_fixed | Global + HxV COSTAR | `-0.003631` | [-0.004270, -0.002999] | True |
| ETTh1 | equal_fixed | Low-rank horizon x variable causal EMA | `-0.003388` | [-0.004156, -0.002613] | True |
| ETTh1 | global_causal | Equal fixed ensemble | `+0.001510` | [+0.000851, +0.002161] | True |
| ETTh1 | global_causal | Horizon-only causal EMA | `-0.000117` | [-0.000218, -0.000016] | True |
| ETTh1 | global_causal | Variable-only causal EMA | `-0.001670` | [-0.002082, -0.001261] | True |
| ETTh1 | global_causal | Full horizon x variable causal EMA | `-0.001806` | [-0.002225, -0.001387] | True |
| ETTh1 | global_causal | Global + HxV COSTAR | `-0.002122` | [-0.002482, -0.001775] | True |
| ETTh1 | global_causal | Low-rank horizon x variable causal EMA | `-0.001879` | [-0.002295, -0.001464] | True |
| ETTh1 | hxv_causal | Equal fixed ensemble | `+0.003316` | [+0.002553, +0.004085] | True |
| ETTh1 | hxv_causal | Global causal EMA | `+0.001806` | [+0.001387, +0.002225] | True |
| ETTh1 | hxv_causal | Horizon-only causal EMA | `+0.001689` | [+0.001274, +0.002098] | True |
| ETTh1 | hxv_causal | Variable-only causal EMA | `+0.000136` | [-0.000048, +0.000320] | False |
| ETTh1 | hxv_causal | Global + HxV COSTAR | `-0.000315` | [-0.000521, -0.000112] | True |
| ETTh1 | hxv_causal | Low-rank horizon x variable causal EMA | `-0.000072` | [-0.000201, +0.000055] | False |
| ETTh2 | equal_fixed | Global causal EMA | `-0.000725` | [-0.001211, -0.000252] | True |
| ETTh2 | equal_fixed | Horizon-only causal EMA | `-0.001200` | [-0.001687, -0.000705] | True |
| ETTh2 | equal_fixed | Variable-only causal EMA | `-0.003407` | [-0.004649, -0.002192] | True |
| ETTh2 | equal_fixed | Full horizon x variable causal EMA | `-0.004525` | [-0.005816, -0.003284] | True |
| ETTh2 | equal_fixed | Global + HxV COSTAR | `-0.004046` | [-0.005023, -0.003070] | True |
| ETTh2 | equal_fixed | Low-rank horizon x variable causal EMA | `-0.003731` | [-0.004991, -0.002487] | True |
| ETTh2 | global_causal | Equal fixed ensemble | `+0.000725` | [+0.000252, +0.001211] | True |
| ETTh2 | global_causal | Horizon-only causal EMA | `-0.000475` | [-0.000603, -0.000349] | True |
| ETTh2 | global_causal | Variable-only causal EMA | `-0.002682` | [-0.003686, -0.001709] | True |
| ETTh2 | global_causal | Full horizon x variable causal EMA | `-0.003800` | [-0.004851, -0.002787] | True |
| ETTh2 | global_causal | Global + HxV COSTAR | `-0.003321` | [-0.004087, -0.002588] | True |
| ETTh2 | global_causal | Low-rank horizon x variable causal EMA | `-0.003006` | [-0.004042, -0.002005] | True |
| ETTh2 | hxv_causal | Equal fixed ensemble | `+0.004525` | [+0.003284, +0.005816] | True |
| ETTh2 | hxv_causal | Global causal EMA | `+0.003800` | [+0.002787, +0.004851] | True |
| ETTh2 | hxv_causal | Horizon-only causal EMA | `+0.003325` | [+0.002321, +0.004359] | True |
| ETTh2 | hxv_causal | Variable-only causal EMA | `+0.001117` | [+0.000934, +0.001317] | True |
| ETTh2 | hxv_causal | Global + HxV COSTAR | `+0.000478` | [+0.000126, +0.000854] | True |
| ETTh2 | hxv_causal | Low-rank horizon x variable causal EMA | `+0.000794` | [+0.000616, +0.000976] | True |

## Causality perturbation check

Only the last 25% of router_val window targets were randomized; earlier-window predictions must be bit-identical to the unperturbed run.

| Dataset | Method | Earlier windows unchanged | Tail reacted | Result |
|---|---|---|---|---|
| ETTh1 | equal_fixed | True | False | PASS |
| ETTh1 | global_causal | True | True | PASS |
| ETTh1 | horizon_only | True | True | PASS |
| ETTh1 | variable_only | True | True | PASS |
| ETTh1 | hxv_causal | True | True | PASS |
| ETTh1 | global_plus_hxv | True | True | PASS |
| ETTh1 | hxv_lowrank | True | True | PASS |
| ETTh2 | equal_fixed | True | False | PASS |
| ETTh2 | global_causal | True | True | PASS |
| ETTh2 | horizon_only | True | True | PASS |
| ETTh2 | variable_only | True | True | PASS |
| ETTh2 | hxv_causal | True | True | PASS |
| ETTh2 | global_plus_hxv | True | True | PASS |
| ETTh2 | hxv_lowrank | True | True | PASS |

## Hard rule compliance

```text
TEST SET ACCESSED: NO
TEST CACHE LOADED: NO
TEST METRICS COMPUTED: NO
```

## Reproduce

```powershell
python experiments\costar_router_ablation\run_router_ablation.py --device cpu
```
