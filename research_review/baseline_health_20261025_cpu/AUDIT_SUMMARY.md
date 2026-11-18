# Baseline health audit — 2026-10-25

**All four ports passed 14 known-answer software controls with CPU explicitly selected.**
**Their paper-comparison rows remain NOT CLEARED.** This is a local synthetic audit,
not a published-result reproduction or a server-environment validation.

| Port/control | MAE | Equal control | Oracle MAE | Finding |
| --- | ---: | ---: | ---: | --- |
| FAME, predictable | 0.021294 | 2.420001 | 0.020000 | Learns the planted winner |
| TimeRouter, predictable | 0.020000 | 2.420001 | 0.020000 | Learns the planted winner; no fallback on this case |
| Super-Linear gate, predictable | 0.136117 | 2.420001 | 0.020000 | Learns the planted winner |
| OneNet OCP, branch 1 wins | 0.132602 | 1.020000 | 0.020000 | Learns to favour branch 1 |
| OneNet OCP, branch 2 wins | 0.080158 | 0.980000 | 0.020000 | Learns to favour branch 2 |

These are different software controls, not a ranking of the methods. OneNet's
control uses its actual two branches; the other three use the five-expert fixture.

- All four pass the shuffled-winner guards and the named-column permutation check.
  Permutation differences are exactly zero in the explicit-CPU audit.
- OneNet makes 127 delayed updates per case. Every update was checked against its
  target-resolution time and actual input tensors. Poisoning targets from row 64
  onward leaves predictions through row 64 inclusive exactly unchanged.
- The deliberately poisoned case's full-period MAE is not a performance result;
  its purpose is the exact unchanged-prefix check.
- Pinned official source/hash checks pass for all four, for the components covered
  by their existing port checks. This does not establish parity of an entire paper method.
- Runtime loads were restricted to newly generated audit files. No research cache,
  expert checkpoint or locked test block was loaded.

## FAME execution discrepancy retained

The first audit intended to hide CUDA using an empty CUDA_VISIBLE_DEVICES mask.
That did not disable CUDA on this Windows host. FAME's upstream auto-device
selector chose CUDA; independent seed-7 fits gave differing losses despite
identical canonical fingerprints, loss matrices, forecast tensors and targets.
The first permutation check failed (maximum per-window difference 0.180668).
Those results and their executed source snapshot are retained in
`../baseline_health_20261025/`; diagnostic fits are retained separately.

The second audit explicitly sets FAME's configuration device to CPU and asserts
the actual device. It passes the original, unchanged permutation threshold
exactly. No grids, learning settings or thresholds were tuned. This clears only
the explicit-CPU software control; it does not fix or validate the historical
CUDA runs, whose exact reproducibility remains open.

## Paper-fidelity gate

- FAME: frozen-pool rolling-window adaptation; a published benchmark result was not reproduced.
- TimeRouter: existing custom retrained OVA head; original head training code is not available here.
- OneNet: only the OCP block over two frozen branches, with delayed feedback.
  Full branch co-adaptation is absent. It must not be labelled a full OneNet
  reproduction or ranked as an offline frozen router.
- Super-Linear: the spectral gate over frozen experts; full frequency-expert training is absent.

`paper_table_gate.json` explicitly keeps `table_eligible: false` for every port.
The software-health prerequisite is now supported locally. Faithful scope,
published-result reproduction and protocol-compatible comparisons are still
unresolved before any full-method paper row can be cleared.

## Artifacts

`results.json`, `report.md`, `manifest.json` and per-case folders retain the exact
environment, sources, upstream pins, losses, diagnostics, and OneNet causal trace.
The run used local Python 3.13.5, Torch 2.6.0+cu124, NumPy 2.3.3 and pandas 2.3.3;
it is not evidence of a run under the server's different package versions.
