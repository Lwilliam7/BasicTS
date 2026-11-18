#!/usr/bin/env bash
# Seed-7 pilot of the offline gate experiment (PREDECLARATION.md): the selftest, the 14 jobs, then the analysis.
# Existing server environment; no installs, no expert training, no test access. --status starts nothing.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
P="${GATE_PYTHON:-$(dirname "$ROOT")/envs/basicts-ec/bin/python}"
M=experiments.offline_bates_router.run_offline_bates_router
WORK=experiments/offline_bates_router/_work
LOG=logs/offline_bates_router/pilot.log
[[ -x "$P" ]] || { echo "REFUSED: server Python not found (set GATE_PYTHON)"; exit 1; }
if [[ "${1:-}" == "--status" ]]; then
  "$P" - "$WORK" <<'EOF'
import json, pathlib, sys
recs = sorted(pathlib.Path(sys.argv[1]).glob("*/seed7.json"))
ok = sum(bool(json.loads(r.read_text()).get("integrity", {}).get("ok")) for r in recs)
print(f"{ok} of 14 seed-7 jobs valid, {len(recs) - ok} invalid")
EOF
  tail -n 5 "$LOG" 2>/dev/null || true
  if [[ -f experiments/offline_bates_router/report.md ]]; then echo "Report: experiments/offline_bates_router/report.md"; fi
  exit 0
fi
[[ $# == 0 ]] || { echo "Usage: bash experiments/offline_bates_router/launch_pilot.sh [--status]"; exit 1; }
if pgrep -f '[r]un_gate_ablation.py --(queue|unit|anchor|analyze)' >/dev/null || pgrep -f '[r]un_offline_bates_router' >/dev/null; then
  echo "REFUSED: a gate-ablation or offline-experiment process is still running"
  exit 1
fi
# No runner is alive, so any job lock left behind by a killed run is stale.
find "$WORK" -name '*.lock' -delete 2>/dev/null || true
"$P" experiments/offline_bates_router/run_offline_bates_router.py --selftest
mkdir -p logs/offline_bates_router
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
nohup bash -c "\"$P\" -u -m $M --run all --seeds 7; \"$P\" -u -m $M --analyze" >> "$LOG" 2>&1 < /dev/null &
pid=$!
sleep 2
kill -0 "$pid" 2>/dev/null || { echo "REFUSED: pilot exited; read $LOG"; exit 1; }
echo "Pilot started (pid $pid): 14 jobs, then the analysis. Progress: bash experiments/offline_bates_router/launch_pilot.sh --status"
