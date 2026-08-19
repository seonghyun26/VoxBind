#!/usr/bin/env bash
# QUEUE (per user 260815): two champion-recipe pretrainings on the 4-7 lane, sequentially,
# once GPU 4-7 frees (the per_group [7,4,1,1] probe on 4 + ensemble search on 5 + smokes finish):
#   run1 = champion recipe on CASF-clean v2.4          (260813_cdg_100m_v24_mask075)
#   run2 = champion recipe, NO gradmag, [7,4,1], v2.2  (260813_cd_100m_v22_g741_mask075)
# Detached:  setsid bash scripts/queue_2runs.sh >/dev/null 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
LOG=log/queue_2runs.log
CKPT() { echo "exps/$1/checkpoint_e0049.pth.tar"; }
gpu47_free() {   # sum of GPU 4-7 used-MiB under 6 GB → lane free
  local s; s=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n '5,8p' | awk '{s+=$1} END{print s+0}')
  [ "$s" -lt 6000 ]
}
busy() { pgrep -f "eval_cdg_encoder.sh|ensemble_search.py|rank_probe.py|smoke_260813" >/dev/null 2>&1; }

echo "[$(date '+%F %T')] queue up; waiting for GPU 4-7 free" >> "$LOG"
while busy || ! gpu47_free; do sleep 120; done

# ---- RUN 1: champion v2.4 ----
if [ ! -e "$(CKPT 260813_cdg_100m_v24_mask075)" ]; then
  echo "[$(date '+%F %T')] launching run1 champion-v2.4 on 4-7" >> "$LOG"
  setsid bash scripts/run_champion.sh --gpus 4-7 --v24 >> "$LOG" 2>&1 < /dev/null &
  sleep 90
  while [ ! -e "$(CKPT 260813_cdg_100m_v24_mask075)" ]; do sleep 300; done
  echo "[$(date '+%F %T')] run1 finished (e49)" >> "$LOG"
fi

# ---- RUN 2: no-gradmag [7,4,1] v2.2 ----
if [ ! -e "$(CKPT 260813_cd_100m_v22_g741_mask075)" ]; then
  echo "[$(date '+%F %T')] launching run2 cd-g741-v2.2 on 4-7" >> "$LOG"
  setsid bash scripts/run_champion.sh --gpus 4-7 --groups 7,4,1 --no_gradmag --in_vocab >> "$LOG" 2>&1 < /dev/null &
  sleep 90
  pgrep -f "exp_name=260813_cd_100m_v22_g741_mask075" >/dev/null \
    && echo "[$(date '+%F %T')] run2 launched" >> "$LOG" \
    || echo "[$(date '+%F %T')] WARN run2 not detected — check log/260813_cd_100m_v22_g741_mask075.log" >> "$LOG"
fi
echo "[$(date '+%F %T')] queue_2runs done" >> "$LOG"
