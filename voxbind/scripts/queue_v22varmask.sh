#!/usr/bin/env bash
# QUEUE: launch the v2.2 varmask (R2MAE [7,4,2], dset.in_vocab_only) on the FIRST of the two
# training lanes that frees — {0-3 = varmask6090, 4-7 = v22_g7411}. Designed to be started
# DETACHED (setsid) so it survives session/harness cleanup (a run_in_background harness poller
# died silently once, leaving GPUs idle — this is the more robust replacement).
#
#   setsid bash scripts/queue_v22varmask.sh >/dev/null 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
LOG=log/queue_v22varmask.log
GUARD=exps/260813_cdg_100m_v22_varmask6090          # once this exists the run has launched
alive() { pgrep -f "exp_name=$1" >/dev/null 2>&1; }  # specific → never self-matches this poller

echo "[$(date '+%F %T')] queue poller up; waiting for lane {0-3 varmask | 4-7 v22_g7411}" >> "$LOG"
while true; do
  [ -e "$GUARD" ] && { echo "[$(date '+%F %T')] guard present, already launched — exit" >> "$LOG"; exit 0; }
  if   ! alive 260813_cdg_100m_v2_varmask6090; then LANE=0-3; break
  elif ! alive 260813_cdg_100m_v22_g7411;      then LANE=4-7; break
  fi
  sleep 300
done
echo "[$(date '+%F %T')] lane $LANE free -> launching v22 varmask" >> "$LOG"
setsid bash scripts/run_varmask.sh --gpus "$LANE" --in_vocab >> "$LOG" 2>&1 < /dev/null &
sleep 15
if alive 260813_cdg_100m_v22_varmask6090; then
  echo "[$(date '+%F %T')] OK launched on $LANE: $(pgrep -f 'exp_name=260813_cdg_100m_v22_varmask6090' | head -1)" >> "$LOG"
else
  echo "[$(date '+%F %T')] WARNING: launch not detected — check log/260813_cdg_100m_v22_varmask6090.log" >> "$LOG"
fi
