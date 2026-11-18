Embedding ablation for Affinity-Weighted Window-Dependent Expert Choice

```text
TEST SET ACCESSED: NO
ROUTER_VAL ACCESSED: NO
UNTOUCHED DATA ACCESSED: NO
```

## OOF MAE by variant

| Dataset | E_full | E_noH | E_noV | E_noExpert | E_none |
|---|---:|---:|---:|---:|---:|
| ETTh1 | `0.351804` | `0.350493` | `0.352906` | `0.352012` | `0.353683` |
| ETTh2 | `0.284354` | `0.284739` | `0.284941` | `0.286267` | `0.285672` |
| ETTm1 | `0.253681` | `0.255513` | `0.255265` | `0.254532` | `0.253749` |
| Weather | `0.166304` | `0.167757` | `0.167497` | `0.166520` | `0.168787` |
| Electricity | `0.227905` | `0.228791` | `0.229250` | `0.228480` | `0.229829` |

## Classifications

### H embedding

- Removal hurts OOF on 4/5 datasets (need >=3)
- Block-24 support: 3/5
- Aggregate delta positive (net hurt): `True`
- **Label: SUPPORTED**

### V embedding

- Removal hurts OOF on 5/5 datasets (need >=3)
- Block-24 support: 3/5
- Aggregate delta positive (net hurt): `True`
- **Label: SUPPORTED**

F0_anchor (from the feature-ablation experiment) already includes static_gain[h,v,e] as an explicit scalar input alongside the V embedding for every variant here (feature groups are held fixed at the selected variant, embeddings are the only thing ablated in this experiment). The V-embedding removal test above (E_noV vs E_full, same static_gain scalar present in both) isolates whether the LEARNED variable identity vector adds anything beyond what static_gain[h,v,e] already encodes numerically.

### Expert embedding

- Removal hurts OOF on 5/5 datasets (need >=3)
- Block-24 support: 3/5
- Aggregate delta positive (net hurt): `True`
- **Label: SUPPORTED**

The scorer is SHARED across all K=3 experts (one set of weights, not one network per expert). The expert embedding is the only per-expert-identity signal available to that shared network (besides static_gain, which already varies by expert). E_noExpert tests whether the shared scorer can still distinguish heterogeneous experts' residual competence without a learned expert identity vector.

