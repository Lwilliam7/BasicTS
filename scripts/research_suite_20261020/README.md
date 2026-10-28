# Four server follow-ups

Prepared locally; no server upload or run has been performed by the assistant.
Scientific scope and limitations are in PROTOCOL.md. Original experiment outputs
are preserved; new granularity evidence is exploratory, post-result evidence.

Stop the earlier interactive `Waiting for an idle GPU...` loop with Ctrl+C if it
is still waiting. Do not stop an already-running scientific job. This avoids two
independent launchers racing to start the Traffic recovery.

Upload `research_review/research_suite_20261020.tar.gz` from the laptop to the
existing server BasicTS directory. Extract the archive there. It contains only
the new `scripts/research_suite_20261020/` and
`experiments/gate_granularity_exploratory_20261020/` source directories.

Then use the server's existing environment:

```bash
cd /mnt/nvme0/$USER/BasicTS
tar -xzf research_suite_20261020.tar.gz
bash scripts/research_suite_20261020/launch.sh --plan
```

Review missing packages/official clones before launch. No automatic installation
or cloning is performed. After those requirements are present:

```bash
bash scripts/research_suite_20261020/launch.sh
```

The suite starts independent CPU and GPU lanes. CPU health checks precede real
port extensions; failed health checks block that port's extension. Conditional
baselines run on CPU independently of those checks. The GPU lane resumes Traffic
first, then runs the exploratory anchor, canary and 70 granularity jobs serially
as cards become idle. It never kills other users' jobs. The suite's cooperative
GPU lock is not a reservation against unrelated workloads.

```bash
bash scripts/research_suite_20261020/launch.sh --status
tail -n 30 logs/research_suite_20261020.log
```

New results: `experiments/research_suite_20261020/`. Exploratory granularity:
`experiments/gate_granularity_exploratory_20261020/`. At the end a result archive
is written to `experiments/research_suite_20261020.tar.gz`. Reusable feature
caches are omitted. `report.md` and `status.json` keep failed, blocked and
unsupported work explicit; TimeRouter H=192 requires a new CV design and stays
UNSUPPORTED. PARTIAL is an expected overall label until those units are resolved.

To resume an interrupted suite, use the same launch command. Completed real-data
workers revalidate identities. Retain the same sources, packages and settings;
otherwise choose a new output directory explicitly with the Python runner.

Local verification is recorded separately in `LOCAL_VALIDATION.md`. The three
pinned official repositories and research forecast pools are on the server, so
passing local orchestration/component checks does not claim a passed server run.
