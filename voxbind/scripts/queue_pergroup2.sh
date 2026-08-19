#!/usr/bin/env bash
# QUEUE: per_group + [7,4,2] + variable mask U[0.7,0.9] + v2.2 — launches on the 0-3
# pretraining lane once the current per_group [7,4,1,1] run finishes. Detached:
#   setsid bash scripts/queue_pergroup2.sh >/dev/null 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
LOG=log/queue_pergroup2.log
GUARD=exps/260813_cdg_100m_v22_g742_pergroup_varmask7090
alive() { pgrep -f "exp_name=$1" >/dev/null 2>&1; }

echo "[$(date '+%F %T')] queue poller up; waiting for 0-3 (per_group g7411 to finish)" >> "$LOG"
while true; do
  [ -e "$GUARD" ] && { echo "[$(date '+%F %T')] guard present, already launched — exit" >> "$LOG"; exit 0; }
  alive 260813_cdg_100m_v22_g7411_pergroup || break
  sleep 300
done
echo "[$(date '+%F %T')] lane 0-3 free -> launching per_group [7,4,2] varmask" >> "$LOG"
setsid bash scripts/run_pergroup.sh --gpus 0-3 --groups 7,4,2 --min 0.7 --max 0.9 --in_vocab >> "$LOG" 2>&1 < /dev/null &
sleep 20
if alive 260813_cdg_100m_v22_g742_pergroup_varmask7090; then
  echo "[$(date '+%F %T')] OK launched: $(pgrep -f 'exp_name=260813_cdg_100m_v22_g742_pergroup_varmask7090' | head -1)" >> "$LOG"
else
  echo "[$(date '+%F %T')] WARNING: launch not detected — check log/260813_cdg_100m_v22_g742_pergroup_varmask7090.log" >> "$LOG"
fi
