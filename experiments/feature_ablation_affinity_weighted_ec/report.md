Feature-group ablation for Affinity-Weighted Window-Dependent Expert Choice

```text
TEST SET ACCESSED: NO
ROUTER_VAL ACCESSED: NO
UNTOUCHED DATA ACCESSED: NO
```

## OOF MAE by variant

| Dataset | F0_anchor | F1_cell | F2_local | F3_full | Full_NoCell | Full_NoPerVariable | Full_NoGlobal |
|---|---:|---:|---:|---:|---:|---:|---:|
| ETTh1 | `0.360185` | `0.353092` | `0.351804` | `0.351324` | `0.360423` | `0.354307` | `0.351804` |
| ETTh2 | `0.288718` | `0.286013` | `0.284354` | `0.286132` | `0.289744` | `0.286221` | `0.284354` |
| ETTm1 | `0.261997` | `0.255611` | `0.253681` | `0.254763` | `0.255992` | `0.253956` | `0.253681` |
| Weather | `0.185310` | `0.174185` | `0.166304` | `0.166782` | `0.180464` | `0.172095` | `0.166304` |
| Electricity | `0.243443` | `0.232565` | `0.227905` | `0.229061` | `0.243322` | `0.232349` | `0.227905` |

Best predeclared variant by mean OOF MAE across F0-F3: **F2_local**

## Group classifications

### cell

- Adding helps on 5/5 datasets (need >=3): `True`
- Removing hurts on 0/5 datasets (need >=3): `False`
- Block-24 dependence support: add=5/5, remove=0/5
- Independent add/remove evidence: `True`
- **Label: MIXED**

### local

- Adding helps on 5/5 datasets (need >=3): `True`
- Removing hurts on 1/5 datasets (need >=3): `False`
- Block-24 dependence support: add=4/5, remove=0/5
- Independent add/remove evidence: `True`
- **Label: MIXED**

### global

- Adding helps on 1/5 datasets (need >=3): `False`
- Removing hurts on 4/5 datasets (need >=3): `True`
- Block-24 dependence support: add=0/5, remove=4/5
- Independent add/remove evidence: `False` -- add and remove tests for this group are the SAME comparison (F3_full vs F2_local); only one independent piece of evidence exists.
- **Label: MIXED**

