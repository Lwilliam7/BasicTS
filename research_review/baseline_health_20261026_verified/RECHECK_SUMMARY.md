# Four-port baseline health recheck, 2026-10-26

All four ports pass the 14 known-answer CPU software cases. The runner and the
independent artifact verifier both exited 0. Every method retains
`paper_fidelity: NOT_ESTABLISHED` and `table_eligible: false`.

The fixture uses 96 history points, horizon 12, one channel, 384 training windows
and 128 evaluation windows. Independent fixture checks confirm that the planted
best expert has MAE 0.02000, histories agree with earlier realized targets, and
training/evaluation targets are disjoint. The offline shuffled control keeps
candidate forecasts exactly unchanged while removing the predictable winner.

| Port and positive control | MAE | Equal mix MAE | Best expert MAE |
| --- | ---: | ---: | ---: |
| FAME, predictable winner | 0.021294 | 2.420001 | 0.020000 |
| TimeRouter, predictable winner | 0.020000 | 2.420001 | 0.020000 |
| Super-Linear gate, predictable winner | 0.136117 | 2.420001 | 0.020000 |
| OneNet OCP, branch 1 wins | 0.132602 | 1.020000 | 0.020000 |
| OneNet OCP, branch 2 wins | 0.080158 | 0.980000 | 0.020000 |

OneNet uses two-branch online controls; the other three use a five-expert router
fixture. These values describe those software controls.

All four pass the shuffled-winner guards and expert-name/column permutations.
Each permutation produces exactly identical per-window losses. OneNet makes
127 delayed updates per case, each checked against target resolution and the
actual tensors. Poisoning labels from row 64 leaves predictions through row 64
exactly unchanged; later predictions change once poisoned feedback resolves
(maximum standardized prediction difference 1.954566).

One reporting bug was corrected in `reference_fixture.py`: FAME's supplementary
winner accuracy now comes from its primary top-2 evaluation, matching the
exported losses. The earlier helper took the last probability call, from the
secondary top-1 model, which selected a different tau on shuffled data. Its
shuffled accuracy changes from 53.90625% to 51.5625%. Fitting, selection,
thresholds and forecasts are unchanged. All 14 MAE/MSE arrays are bitwise
identical to both the preceding October 26 recheck and the October 25 explicit
CPU run. The independent verifier also checked 24 source hashes, 58 artifact
hashes, and correspondence between every exported loss array and the actual
port output. Earlier results and executed helper snapshots are preserved.

The paper-comparison gate remains closed for these concrete reasons:

- FAME uses the frozen-pool rolling-window adaptation; no published benchmark
  result is reproduced here. The earlier CUDA reproducibility failure remains
  unresolved; this evidence covers explicitly selected CPU execution.
- TimeRouter uses the existing retrained one-vs-all head over the frozen pool.
- OneNet covers the OCP block over two frozen branches; branch co-adaptation is
  absent. Its online feedback protocol differs from the offline routers.
- Super-Linear covers the spectral gate over frozen experts; frequency-expert
  training is absent.

The runs load newly generated synthetic audit files. No real research forecast
cache, expert checkpoint, or locked test block is loaded. This is a local run
with Python 3.13.5 and Torch 2.6.0+cu124 using CPU execution; it does not validate
a separate server environment.

Evidence: `report.md`, `results.json`, `manifest.json`, `status.json`,
`fixture_validation.json`, `artifact_verification.json`, `verify_artifacts.py`,
`source_snapshot/`, and the per-case logs, losses and OneNet causality traces.

Executed from the workspace root:

```powershell
python -B -u scripts/baseline_health_20261025/run_health.py run --repo-root C:\Data\Documents\Code\BasicTS --official-root research_review/reference_checks/baseline_health_20261025/third_party --output research_review/baseline_health_20261026_verified
python -B research_review/baseline_health_20261026_verified/verify_artifacts.py
```

The first command preserves existing output: a fresh repeat requires a new
output directory; `--resume` only reuses identical completed evidence.
