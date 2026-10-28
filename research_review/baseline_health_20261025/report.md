# Four-port synthetic health audit

Status: HEALTH_FAILURE_OR_INCOMPLETE

Paper-table eligibility: **NOT CLEARED for every port**. Full-paper fidelity has not been established.

| Port | Software health | Control | MAE | Equal control | Oracle | Table eligible |
| --- | --- | --- | ---: | ---: | ---: | --- |
| fame | FAIL_OR_INCOMPLETE | predictable | 0.020000 | 2.420001 | 0.020000 | No |
| fame | FAIL_OR_INCOMPLETE | shuffled | 0.963011 | 2.404375 | 0.020000 | No |
| fame | FAIL_OR_INCOMPLETE | permuted | 0.022683 | 2.420001 | 0.020000 | No |

fame: Rolling-window frozen-pool adaptation; no published dataset/result reproduced.
Cross-case checks: `{"expert_name_column_permutation": false, "permutation_max_abs_difference": 0.18066756427288055}`

| timerouter | PASS | predictable | 0.020000 | 2.420001 | 0.020000 | No |
| timerouter | PASS | shuffled | 1.039954 | 2.404375 | 0.020000 | No |
| timerouter | PASS | permuted | 0.020000 | 2.420001 | 0.020000 | No |

timerouter: Port uses a retrained OVA head; the authors' head training code is not released.
Cross-case checks: `{"expert_name_column_permutation": true, "permutation_max_abs_difference": 0.0}`

| onenet | PASS | branch1 | 0.132602 | 1.020000 | 0.020000 | No |
| onenet | PASS | branch2 | 0.080158 | 0.980000 | 0.020000 | No |
| onenet | PASS | shuffled | 1.043538 | 1.000625 | 0.020000 | No |
| onenet | PASS | permuted | 0.132602 | 1.020000 | 0.020000 | No |
| onenet | PASS | poisoned | 17.602446 | — | — | No |

onenet: Two-branch OCP block with frozen experts and delayed feedback; full branch co-adaptation is absent. Online protocol differs from offline routers.
Cross-case checks: `{"expert_name_column_permutation": true, "permutation_max_abs_difference": 0.0, "future_target_poison_prefix_unchanged": true, "future_target_poison_prefix_max_difference": 0.0}`

| superlinear | PASS | predictable | 0.136117 | 2.420001 | 0.020000 | No |
| superlinear | PASS | shuffled | 1.008490 | 2.404375 | 0.020000 | No |
| superlinear | PASS | permuted | 0.136117 | 2.420001 | 0.020000 | No |

superlinear: Spectral combination gate over frozen experts; full frequency-expert training is absent.
Cross-case checks: `{"expert_name_column_permutation": true, "permutation_max_abs_difference": 0.0}`

## Interpretation

Failures remain failures; no thresholds or method settings were tuned after seeing outcomes.
OneNet is its causal OCP block on two frozen branches. It cannot be presented as a full OneNet reproduction or ranked as an offline frozen router.
Detailed diagnostics, source/environment manifests, per-window losses and OneNet causal traces accompany this report.
