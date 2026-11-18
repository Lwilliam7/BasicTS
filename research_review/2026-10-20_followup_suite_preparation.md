# Four follow-ups requested — preparation, 2026-10-20

The user requested running all four outstanding items: matched conditional
baselines, Traffic H=336 seed-17 recovery, modern-port health checks/extensions,
and exploratory window/cell granularity comparisons.

Prepared `scripts/research_suite_20261020/` and the isolated derived runner in
`experiments/gate_granularity_exploratory_20261020/`. Protocol, limitations,
source provenance and local validation are retained in those directories.
The deployable source archive is `research_review/research_suite_20261020.tar.gz`.

Latest user-provided server status remains the original October 12 queue:
104 of 105 seed jobs done, Traffic H=336 seed 17 failed with GPU OOM, analysis
exit 0. All six GPUs were busy, about 486 GiB RAM available and 9.8 GiB disk free.
The user confirmed backing up the original result bundle, queue state and failed
job log (3.2 MB total). The idle-GPU retry guard printed "Nothing started."
An interactive waiting loop was subsequently provided; its launch has not been
confirmed. Stop that waiting loop before starting the combined suite.

Server interaction preference in MDFILES/server.md remains commands for the user,
one step per reply. No SSH/SCP or server command was executed by the assistant.
No result or pending item is marked completed by this preparation. The first
remaining step is user-run SCP of the source archive, followed by extraction and
the read-only server plan. TimeRouter H=192 is explicitly unsupported by its
current 96-point-history CV mapping. Granularity is exploratory, post-result.
