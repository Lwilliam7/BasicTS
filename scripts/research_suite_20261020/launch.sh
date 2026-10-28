#!/usr/bin/env bash
# User launches this on the server. Existing environment; new results stay isolated.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
P="${RESEARCH_PYTHON:-$(dirname "$ROOT")/envs/basicts-ec/bin/python}"
R=scripts/research_suite_20261020/run_suite.py
[[ -x "$P" ]] || { echo "Server Python not found: $P"; exit 1; }
if [[ "${1:-}" == "--status" ]]; then
  exec "$P" "$R" status
fi
if [[ "${1:-}" == "--plan" ]]; then
  exec "$P" "$R" plan
fi
[[ $# == 0 ]] || { echo "Usage: launch.sh [--status|--plan]"; exit 1; }
mkdir -p logs
nohup "$P" -u "$R" run >> logs/research_suite_20261020.log 2>&1 < /dev/null &
pid=$!
sleep 2
if ! kill -0 "$pid" 2>/dev/null; then
  echo "Suite exited; see logs/research_suite_20261020.log"
  tail -n 20 logs/research_suite_20261020.log
  exit 1
fi
echo "Suite launched: PID $pid. CPU work starts first; GPU work waits for an idle card."
echo "Status: bash scripts/research_suite_20261020/launch.sh --status"
