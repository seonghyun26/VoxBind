#!/usr/bin/env bash
# QUEUE: launch the per-channel-group masking run (per_group [7,4,1,1] v2.2) on the FIRST
# of the two training lanes that frees — {0-3 = champion-v2.2 mask075, 4-7 = g7411 varmask7090}.
# Start DETACHED so it survives session/harness cleanup:
#   setsid bash scripts/queue_pergroup.sh >/dev/null 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
LOG=log/queue_pergroup.log
GUARD=exps/260813_cdg_100m_v22_g7411_pergroup          # once this exists the run has launched
alive() { pgrep -f "exp_name=$1" >/dev/null 2>&1; }    # specific → never self-matches this poller

echo "[$(date '+%F %T')] queue poller up; waiting for a free lane {0-3 mask075 | 4-7 g7411_varmask7090}" >> "$LOG"
while true; do
  [ -e "$GUARD" ] && { echo "[$(date '+%F %T')] guard present, already launched — exit" >> "$LOG"; exit 0; }
  if   ! alive 260813_cdg_100m_v22_mask075;           then LANE=0-3; break
  elif ! alive 260813_cdg_100m_v22_g7411_varmask7090; then LANE=4-7; break
  fi
  sleep 300
done
echo "[$(date '+%F %T')] lane $LANE free -> launching per_group" >> "$LOG"
setsid bash scripts/run_pergroup.sh --gpus "$LANE" --in_vocab >> "$LOG" 2>&1 < /dev/null &
sleep 20
if alive 260813_cdg_100m_v22_g7411_pergroup; then
  echo "[$(date '+%F %T')] OK launched on $LANE: $(pgrep -f 'exp_name=260813_cdg_100m_v22_g7411_pergroup' | head -1)" >> "$LOG"
else
  echo "[$(date '+%F %T')] WARNING: launch not detected — check log/260813_cdg_100m_v22_g7411_pergroup.log" >> "$LOG"
fi
