#!/bin/bash
# Launch the VoxBind sampling-results browser (Streamlit app).
#
# Usage:
#   bash notebook/webapp/run.sh                # default: 0.0.0.0:8501
#   PORT=8600 bash notebook/webapp/run.sh      # override port
#   HOST=127.0.0.1 bash notebook/webapp/run.sh # bind to localhost only
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP="${HERE}/app.py"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8501}"
WATCHER="${WATCHER:-none}"

# If PORT is taken, roll forward to the next free port (up to +20).
port_in_use() { ss -ltn "sport = :$1" 2>/dev/null | grep -q LISTEN; }

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

exec streamlit run "${APP}" \
    --server.address "${HOST}" \
    --server.port "${PORT}" \
    --server.headless true \
    --server.fileWatcherType "${WATCHER}" \
    --browser.gatherUsageStats false
