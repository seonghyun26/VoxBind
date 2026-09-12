#!/bin/bash
# Launch the standalone qualitative-comparison viewer.
#
# Usage:
#   bash notebook/figures/run_fig_qual.sh
#   PORT=8877 HOST=0.0.0.0 bash notebook/figures/run_fig_qual.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP="${SCRIPT_DIR}/fig_qual_server.py"
VOXBIND_PYTHON="${VOXBIND_PYTHON:-/home/shpark/.conda/envs/voxbind/bin/python}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8766}"

if [ ! -x "${VOXBIND_PYTHON}" ]; then
  echo "VoxBind Python not found: ${VOXBIND_PYTHON}" >&2
  echo "Set VOXBIND_PYTHON to the Python executable used by the voxbind kernel." >&2
  exit 1
fi

port_in_use() { ss -ltn "sport = :$1" 2>/dev/null | rg -q 'LISTEN'; }
START_PORT="${PORT}"
for _ in $(seq 0 20); do
  port_in_use "${PORT}" || break
  PORT=$((PORT + 1))
done
if port_in_use "${PORT}"; then
  echo "No free port found in ${START_PORT}-${PORT}" >&2
  exit 1
fi
if [ "${PORT}" != "${START_PORT}" ]; then
  echo "Port ${START_PORT} busy; using ${PORT} instead" >&2
fi

exec "${VOXBIND_PYTHON}" "${APP}" --host "${HOST}" --port "${PORT}"
