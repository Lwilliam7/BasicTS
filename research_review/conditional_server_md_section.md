## Conditional-combination comparison — prepared 2026-10-17

**Prepared locally, not uploaded or launched.** Transfer this single file to the
server's existing `scripts/` directory:

`scripts/compare_conditional_combinations.py`

It is standalone Python (no project-module imports). It reads the existing H=96
pilot and long-horizon forecast caches, DLinear scaler checkpoint, and the five
wide-top3 `_work/seed*.pt` results with their JSON metadata. It imports the existing
classical comparator results too. The server files are used directly; no manual
NPZ export and no new expert training are required. Before fitting, it checks
per-window equal-five/equal-core errors, training core-fold scores, primary means,
fold origins/purges and integrity metadata. Original primary input hashes were
not recorded; historical association is numerical/metadata-based, and this
limitation is disclosed in the output. Current input hashes protect later resumes.

Default scope: all seven datasets at H=96 and H=192, two concurrent CPU unit
processes, no GPU requirement. Add `--horizons 96 192 336 720` to plan/run for the
full grid, with a separate `--output` if the default run already started. Traffic
H=720 retains the original exclusion; missing Traffic H=336 seed-17 results stay
BLOCKED. Other ready units continue; the report cannot silently omit failed units.

```bash
cd /mnt/nvme0/$USER/BasicTS
P=/mnt/nvme0/$USER/envs/basicts-ec/bin/python
R=scripts/compare_conditional_combinations.py
mkdir -p logs
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
"$P" "$R" --server plan --jobs 2 --threads 8 --workers 4
"$P" "$R" --self-test --threads 2
nohup "$P" "$R" --server run --jobs 2 --threads 8 --workers 4 \
  > logs/conditional_combinations.log 2>&1 < /dev/null &
echo "queue PID: $!"
```

`$USER` is the server account owning the documented repo and environment. If the
required packages are absent, install with that Python before planning:
`"$P" -m pip install 'xgboost>=2.1' 'tsfeatures==0.4.5' threadpoolctl`.

Status: `"$P" "$R" --server status`. Per-unit logs:
`experiments/conditional_comparison_server/logs/<DS>_h<H>.log`. Restart the same run
command after a stopped run: it revalidates inputs and resumes saved configurations
and seed fits. A lock rejects simultaneous launches. Keep code/packages/settings
fixed within an output directory.

Results: `experiments/conditional_comparison_server/{report.md,summary.csv,summary.json}`
and `units/<DS>_h<H>/`. When the queue ends, run `"$P" "$R" --server pack` and copy
back `experiments/conditional_comparison_server.tar.gz`. With a custom output,
pass that same `--output` to status and pack.

Full instructions and adaptation limitations: `scripts/compare_conditional_combinations.md`.
Checks completed locally: planted routing, archive mismatch rejection, interrupted
resume, completed resume, changed-protocol refusal, duplicate lock, server path
selection, failed-worker reporting, and result packaging. This preparation does
not claim a server experiment has completed.
