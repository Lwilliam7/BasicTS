# Oracle Routing Headroom Diagnostic

**Diagnostic only.** Oracle methods below use validation targets to select the best expert after the fact -- they are not deployable and are never used to train, tune, or select anything in COSTAR. Labeled `noncausal_oracle=true` everywhere they appear in the machine-readable outputs.

## Main table: MAE / MSE

| Dataset | Best Single | Equal Fixed | COSTAR | Window Oracle | Variable Oracle | HxV Oracle |
|---|---:|---:|---:|---:|---:|---:|
| ETTh1 | `0.379558`/`0.326528` | `0.367265`/`0.310530` | `0.363949`/`0.307478` | `0.343984`/`0.274078` | `0.315050`/`0.242071` | `0.263760`/`0.204707` |
| ETTh2 | `0.280957`/`0.171493` | `0.280878`/`0.171933` | `0.276354`/`0.167381` | `0.266483`/`0.156911` | `0.244202`/`0.139469` | `0.196911`/`0.109455` |
| ETTm1 | `0.261771`/`0.161838` | `0.248161`/`0.146694` | `0.248593`/`0.148334` | `0.227306`/`0.124339` | `0.203083`/`0.107679` | `0.172828`/`0.092603` |
| Weather | `0.164673`/`0.287468` | `0.160341`/`0.278815` | `0.159280`/`0.283565` | `0.150243`/`0.246443` | `0.130085`/`0.230937` | `0.114696`/`0.205872` |
| Electricity | `0.225385`/`0.135767` | `0.214457`/`0.117846` | `0.211775`/`0.119014` | `0.217440`/`0.120503` | `0.183580`/`0.089079` | `0.138875`/`0.064471` |

## Headroom vs COSTAR (absolute MAE / relative %)

| Dataset | -> Window Oracle | -> Variable Oracle | -> HxV Oracle |
|---|---:|---:|---:|
| ETTh1 | `+0.019965` (5.49%) | `+0.048899` (13.44%) | `+0.100189` (27.53%) |
| ETTh2 | `+0.009871` (3.57%) | `+0.032152` (11.63%) | `+0.079443` (28.75%) |
| ETTm1 | `+0.021287` (8.56%) | `+0.045510` (18.31%) | `+0.075765` (30.48%) |
| Weather | `+0.009037` (5.67%) | `+0.029195` (18.33%) | `+0.044584` (27.99%) |
| Electricity | `-0.005665` (-2.68%) | `+0.028196` (13.31%) | `+0.072900` (34.42%) |

## Winner dynamics (window oracle)

| Dataset | % winner changes | Dominant expert (%) | Mean run length | Mean COSTAR regret | Median regret | P90 regret |
|---|---:|---|---:|---:|---:|---:|
| ETTh1 | 27.56% | iTransformer (37.5%) | 3.62 | `+0.019965` | `+0.012221` | `+0.059268` |
| ETTh2 | 28.92% | DLinear (53.2%) | 3.44 | `+0.009871` | `+0.005674` | `+0.036565` |
| ETTm1 | 28.18% | TimesNet (37.4%) | 3.55 | `+0.021287` | `+0.013260` | `+0.061161` |
| Weather | 23.94% | PatchTST (42.8%) | 4.18 | `+0.009038` | `+0.004537` | `+0.027491` |
| Electricity | 16.34% | iTransformer (60.7%) | 6.11 | `-0.005665` | `-0.006723` | `+0.002164` |

## Win fraction by expert

- **ETTh1**: PatchTST=35.5%, iTransformer=37.5%, TimesNet=26.9%
- **ETTh2**: DLinear=53.2%, PatchTST=7.3%, ModernTCN=39.5%
- **ETTm1**: DLinear=30.9%, PatchTST=31.7%, TimesNet=37.4%
- **Weather**: PatchTST=42.8%, iTransformer=28.0%, TimesNet=29.1%
- **Electricity**: PatchTST=23.1%, iTransformer=60.7%, TimesNet=16.2%

## Hard rule compliance

```text
TEST SET ACCESSED: NO
TEST CACHE LOADED: NO
TEST METRICS COMPUTED: NO
```
