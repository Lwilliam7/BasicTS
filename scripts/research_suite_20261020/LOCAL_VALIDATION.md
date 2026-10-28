# Local validation — 2026-10-20

Prepared and tested on the laptop. No server upload, SSH connection, server
training, cache rebuild, test evaluation or server result is claimed here.

Passed:

- Eight orchestration/integrity checks: duplicate kernel lock, failed-worker
  reporting, independent-lane continuation, idle-GPU selection, changed-source
  resume refusal, failed-job/integrity detection even after analysis exit 0,
  AST equality of the derived runner's scientific functions, and the actual
  TimeRouter port's CV alignment on both synthetic cache roles.
- All eleven self-test groups of the isolated eight-arm granularity runner.
- Conditional baseline planted-signal self-test: MAE 0.005735 (FFORMA),
  4.23e-9 (error softmax), 0.021163 (MLP), against equal MAE 1.0.
- Actual Super-Linear port pipeline component check (pool A, k grid 2/3/4/5,
  seed 7, original epochs/patience): predictable MAE 0.136117 versus equal
  2.420000, winner accuracy 0.976563; shuffled MAE 1.008490 versus equal
  2.404375, winner accuracy 0.468750. The final synthetic candidate forecasts
  are independent of the winner, and the cache's CV alignment is preserved.
- Python source compilation.

Server-only checks still required:

- Official FAME, TimeRouter and Super-Linear pinned repositories are absent on
  the laptop. Thus no pinned-source health-pass claim is made for any port.
  The server worker checks official hashes before running those controls.
- Forecast/checkpoint/primary alignment on the actual server pools.
- New EXACT granularity anchor, canary and all real-data fits.
- Bash execution. The bundled Windows Bash binary failed to initialize
  (exit 0xC0000142); the shell launcher was inspected but not executed locally.

The source bundle contains the new scripts and protocol only. Results remain
pending until the user uploads, checks the server plan, and starts the suite.
