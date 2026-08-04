#!/bin/bash
# Watch base/_casf/BindNet.json for the final 3-seed result (the detached retrain chain,
# PID 1204949, writes it via aggregate_casf_results.py). Rebuild results.html when it lands.
# Exits when BindNet CASF has >=3 seeds, or after ~4h.
cd /home/shpark/prj-denovo/VoxBind
PLOG=base/_cl_campaign/poller_bindnet_casf.log
: > "$PLOG"
for i in $(seq 1 48); do
  N=$(/home/shpark/.conda/envs/bindnet/bin/python - <<'PY' 2>/dev/null
import json
try:
    d=json.load(open('base/_casf/BindNet.json'))
    ps=d.get('leaky',{}).get('per_seed') or d.get('per_seed') or []
    print(len(ps))
except Exception:
    print(0)
PY
)
  python3 notebook/html/build_results.py >/dev/null 2>&1
  echo "[$(date +%F\ %H:%M:%S)] BindNet CASF per_seed=$N (iter $i)" >> "$PLOG"
  [ "${N:-0}" -ge 3 ] && { echo "[$(date +%F\ %H:%M:%S)] BindNet CASF FINAL (3 seeds) — table complete" >> "$PLOG"; break; }
  sleep 300
done
