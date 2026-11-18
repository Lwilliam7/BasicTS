# FAME, TimeRouter, OneNet and Super-Linear: synthetic software audit

Read PROTOCOL.md before interpreting results. This CPU-only audit uses existing
port code and known-answer synthetic forecasts. No real research cache or test
split is loaded. Every result keeps `table_eligible: false` and
`paper_fidelity: NOT_ESTABLISHED` even when software controls pass.

The October 20 deployed suite is not modified. This package adds the missing
OneNet OCP audit in a separate process/output directory. It also checks the
other three ports, their expert-column mapping, and shuffled-winner controls.
OneNet has two directional known-branch tests, a shuffled test, an expert-column
test, a causal update trace and a future-label perturbation test.

The exact scope of each port is preserved. In particular, the OneNet entry is
the OCP combination block with frozen branches and delayed feedback; it is not
the complete method from the paper. Do not rank it as an offline frozen router.

## Server use

Extract the source archive into the existing BasicTS root, then inspect:

```bash
/mnt/nvme0/$USER/envs/basicts-ec/bin/python scripts/baseline_health_20261025/run_health.py plan
```

The four pinned official source trees must already be under `third_party/`.
The runner performs the ports' original upstream hash/source-equality checks.
It installs nothing and does not download sources or touch existing jobs.

To run (from the repository root):

```bash
nohup /mnt/nvme0/$USER/envs/basicts-ec/bin/python -u scripts/baseline_health_20261025/run_health.py run > logs/baseline_health_20261025.log 2>&1 < /dev/null &
```

Results: `experiments/baseline_health_20261025/{report.md,results.json,status.json}`.
Fourteen cases run sequentially in CPU worker processes. Runtime errors and
failed criteria stay visible. Exit 0 means all software cases/cross-case checks
passed; exit 2 means a failure or incomplete audit. Neither exit clears paper
fidelity or table eligibility. Existing output is preserved; identical resumes
require `--resume` and matching source/environment/artifact hashes.

## Local audit

The local full repository is separate from this workspace. Reproduce locally
with an explicit `--repo-root`, `--official-root` and new writable `--output`.
Public upstream file snapshots were fetched at the ports' pinned commits and
hash-checked. No direct server connection was used. Local Python/package
versions are recorded, so this does not assert a pass in another environment.

The initial local audit is preserved at `research_review/baseline_health_20261025`.
It uncovered FAME auto-selecting CUDA despite an empty visibility mask on this
Windows host. The corrected explicit-CPU audit uses the distinct directory
`research_review/baseline_health_20261025_cpu`; thresholds, grids and learning
settings were not changed. The initial failure is not overwritten.

The October 26 recheck also corrects FAME's supplementary winner-accuracy capture
to use the primary top-2 model rather than the secondary top-1 model. Its fresh
results are in `research_review/baseline_health_20261026_verified`; the preceding
recheck and source snapshot are preserved separately. See PROTOCOL.md.
