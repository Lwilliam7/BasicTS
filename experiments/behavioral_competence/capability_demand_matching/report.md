# Natural Capability-Demand Matching

Validation-only study. No test cache, target, or metric was loaded.

Final classification: `CAPABILITY_SIGNAL_BUT_NO_MATCHING_GAIN`
ETTh2 integrity status: `ETTH2_INTEGRITY_RESOLVED`

## Competence MAE

| Dataset | Counted | Global | Passive | FAME-style | Demand+ID | Capability | Expert shuffle | Axis shuffle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ETTh1 | True | `0.036173` | `0.033263` | `0.037434` | `0.036173` | `0.035850` | `0.045754` | `0.036378` |
| ETTh2 | True | `0.023789` | `0.023971` | `0.026848` | `0.023789` | `0.024121` | `0.036248` | `0.029687` |
| ETTm1 | True | `0.032954` | `0.030820` | `0.035093` | `0.032954` | `0.033159` | `0.035076` | `0.035339` |
| Weather | True | `0.019614` | `0.017493` | `0.019140` | `0.019614` | `0.019518` | `0.023500` | `0.032378` |
| Electricity | True | `0.017378` | `0.016233` | `0.017445` | `0.017377` | `0.017006` | `0.034447` | `0.148623` |

## Routing Proxy MAE

| Dataset | Method | MAE | MSE | Temperature | Regret vs oracle single |
|---|---|---:|---:|---:|---:|
| ETTh1 | GlobalPrior | `0.366482` | `0.309393` | `0.050` | `0.022499` |
| ETTh1 | Passive | `0.366099` | `0.308994` | `0.100` | `0.022116` |
| ETTh1 | FAMEStyleDemand | `0.365038` | `0.307485` | `0.100` | `0.021054` |
| ETTh1 | DemandExpertID | `0.366482` | `0.309393` | `0.050` | `0.022499` |
| ETTh1 | CapabilityMatch | `0.366010` | `0.308694` | `0.100` | `0.022026` |
| ETTh1 | ExpertShuffledCapability | `0.367408` | `0.310753` | `1.000` | `0.023424` |
| ETTh1 | AxisShuffledCapability | `0.365895` | `0.308422` | `0.100` | `0.021912` |
| ETTh2 | GlobalPrior | `0.274939` | `0.165630` | `0.020` | `0.008456` |
| ETTh2 | Passive | `0.277618` | `0.168239` | `0.020` | `0.011135` |
| ETTh2 | FAMEStyleDemand | `0.278289` | `0.169623` | `0.050` | `0.011806` |
| ETTh2 | DemandExpertID | `0.274939` | `0.165630` | `0.020` | `0.008456` |
| ETTh2 | CapabilityMatch | `0.275095` | `0.165901` | `0.020` | `0.008612` |
| ETTh2 | ExpertShuffledCapability | `0.280908` | `0.171950` | `1.000` | `0.014425` |
| ETTh2 | AxisShuffledCapability | `0.279383` | `0.170331` | `0.200` | `0.012900` |
| ETTm1 | GlobalPrior | `0.249614` | `0.148473` | `0.050` | `0.022308` |
| ETTm1 | Passive | `0.248624` | `0.147615` | `0.100` | `0.021318` |
| ETTm1 | FAMEStyleDemand | `0.248493` | `0.147192` | `0.100` | `0.021187` |
| ETTm1 | DemandExpertID | `0.249614` | `0.148473` | `0.050` | `0.022308` |
| ETTm1 | CapabilityMatch | `0.249680` | `0.148596` | `0.050` | `0.022374` |
| ETTm1 | ExpertShuffledCapability | `0.248192` | `0.146699` | `1.000` | `0.020886` |
| ETTm1 | AxisShuffledCapability | `0.248214` | `0.146764` | `1.000` | `0.020908` |
| Weather | GlobalPrior | `0.159878` | `0.277377` | `0.050` | `0.009635` |
| Weather | Passive | `0.159374` | `0.282276` | `0.100` | `0.009131` |
| Weather | FAMEStyleDemand | `0.159554` | `0.276168` | `0.050` | `0.009311` |
| Weather | DemandExpertID | `0.159878` | `0.277377` | `0.050` | `0.009635` |
| Weather | CapabilityMatch | `0.159883` | `0.276821` | `0.050` | `0.009640` |
| Weather | ExpertShuffledCapability | `0.160359` | `0.278882` | `1.000` | `0.010116` |
| Weather | AxisShuffledCapability | `0.160058` | `0.279150` | `1.000` | `0.009815` |
| Electricity | GlobalPrior | `0.215335` | `0.121607` | `0.100` | `-0.002106` |
| Electricity | Passive | `0.217177` | `0.125469` | `0.050` | `-0.000263` |
| Electricity | FAMEStyleDemand | `0.215419` | `0.121675` | `0.100` | `-0.002021` |
| Electricity | DemandExpertID | `0.215335` | `0.121606` | `0.100` | `-0.002106` |
| Electricity | CapabilityMatch | `0.215322` | `0.121565` | `0.100` | `-0.002119` |
| Electricity | ExpertShuffledCapability | `0.214518` | `0.117762` | `1.000` | `-0.002923` |
| Electricity | AxisShuffledCapability | `0.215114` | `0.116616` | `1.000` | `-0.002326` |

## Interpretation

The target is relative competence `z[t,k] = expert_error[t,k] - mean_j expert_error[t,j]`; lower predicted values mean a better-matched expert. The primary `CapabilityMatch` score is the fixed equal average of a LOW/MED/HIGH regime profile and a quadratic Ridge capability curve for each semantic demand axis.

ETTh2 is counted only if the separate audit resolves the cache/runtime reproduction discrepancy. See `etth2_integrity_audit.json` for the normalized-history convention check.

## Integrity

- Test loaded: `False`.
- Demand fingerprints are computed from histories only.
- LOW/MED/HIGH bins, capability profiles, Ridge baselines, and routing temperatures are fit from legal router_train prefixes only.
- Router_val target corruption leaves features, predictions, and weights unchanged exactly.
- Checkpoint hashes are recorded before and after the run.
