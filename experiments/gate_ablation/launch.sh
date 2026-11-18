#!/usr/bin/env bash
# Existing server environment; no installs, expert training or test access.
# Anchor must already have passed EXACT. --status starts nothing.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
P="${GATE_PYTHON:-$(dirname "$ROOT")/envs/basicts-ec/bin/python}"
R=experiments/gate_ablation/run_gate_ablation.py
[[ -x "$P" ]] || { echo "REFUSED: server Python not found"; exit 1; }
if [[ "${1:-}" == "--status" ]]; then
  exec "$P" "$R" --status
fi
[[ $# == 0 ]] || { echo "Usage: bash experiments/gate_ablation/launch.sh [--status]"; exit 1; }
command -v flock >/dev/null || { echo "REFUSED: flock is required"; exit 1; }
# Check the kernel lock before launch; Python also holds it for the full queue.
flock -n experiments/gate_ablation/.campaign.lock true || { echo "REFUSED: ablation is active"; exit 1; }
if pgrep -f '[r]un_gate_ablation.py --(queue|unit|anchor|analyze)' >/dev/null; then
  echo "REFUSED: an ablation process is still running"
  exit 1
fi
mkdir -p logs/gate_ablation
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
nohup "$P" -u "$R" --queue >> logs/gate_ablation/queue.log 2>&1 < /dev/null &
pid=$!
sleep 2
kill -0 "$pid" 2>/dev/null || { echo "REFUSED: queue exited; read logs/gate_ablation/queue.log"; exit 1; }
echo "Queue started (pid $pid). Use --status to check progress."
