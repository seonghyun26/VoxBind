#!/bin/bash
# 260624_ar_cvit_queue.sh — autonomous ChannelViT C+D+G autoresearch worker-pool.
# Two workers (GPU 0-3, GPU 4-7). Each: wait for its OWN GPU set to free (so it
# dovetails behind the running batch-1), then loop: atomically pop the next trial
# from the shared QUEUE -> 40M ChannelViT pretrain (100ep) -> frozen probe on
# lp_edrscc_v2 -> repeat until the queue is empty. Designed to run unattended 12h+.
# rope3d is intentionally absent (crashes with channel_group patch-embed).
set -uf                                    # -f: keep hydra list overrides like [7,4,2] literal (no globbing)
cd /home/shpark/prj-denovo/VoxBind/voxbind || exit 1
PY=/home/shpark/.conda/envs/voxbind/bin
CFG=config_train_atomblob_density_gradmag_channelvit_mae_40m_plinder_otf_mask050
RES=dataset/data/pdbbind/results
LOG=log
QUEUE=$LOG/260624_ar_cvit_queue.txt
LOCK=$LOG/260624_ar_cvit_queue.lock

pop () {                                   # atomically pop first queue line -> stdout (empty if none)
  ( flock 9
    local line; line=$(head -n1 "$QUEUE" 2>/dev/null)
    [ -n "$line" ] && sed -i '1d' "$QUEUE"
    printf '%s' "$line"
  ) 9>"$LOCK"
}

gpus_idle () {                             # $1 = "0,1,2,3"; true if all <2GB used
  local busy; busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" \
                     | awk '$1>2000{c++} END{print c+0}')
  [ "${busy:-9}" -eq 0 ]
}

run_one () {                               # $1=GPUS $2=PORT $3=RID $4=EXP $5=overrides
  local GPUS=$1 PORT=$2 RID=$3 EXP=$4 OV=$5
  echo "###### [$EXP] PRETRAIN start $(date '+%m-%d %H:%M:%S') GPU=$GPUS :: $OV ######"
  CUDA_VISIBLE_DEVICES=$GPUS "$PY/torchrun" --nnodes=1 --nproc_per_node=4 \
      --rdzv-backend=c10d --rdzv-endpoint=localhost:"$PORT" --rdzv-id="$RID" \
      train_density.py --config-name=$CFG exp_name=$EXP bsz=8 accum_steps=4 $OV \
    && echo "[$EXP] PRETRAIN OK $(date '+%H:%M:%S')" || { echo "[$EXP] PRETRAIN FAILED $(date '+%H:%M:%S')"; return 1; }
  bash scripts/04_probe.sh --exp "$EXP" --condition atomblob_density_gradmag --tasks affinity \
      --split lp_edrscc_v2 --gpu "${GPUS%%,*}" --tag "$EXP" --num_workers 0 -- --require_density \
    && echo "[$EXP] PROBE OK $(date '+%H:%M:%S')" || echo "[$EXP] PROBE FAILED"
  echo "###### [$EXP] result ######"; tail -2 "$RES/probe_results_e99_v5_lp_edrscc_v2split_${EXP}.csv" 2>/dev/null
}

worker () {                                # $1=GPUS $2=PORT $3=RID
  local GPUS=$1 PORT=$2 RID=$3 i spec EXP OV
  echo "[$RID] waiting for GPU $GPUS to free (behind batch-1)..."
  for i in $(seq 1 720); do gpus_idle "$GPUS" && break; sleep 60; done
  echo "[$RID] GPU $GPUS free $(date '+%H:%M:%S') — pulling trials"
  while :; do
    spec=$(pop); [ -z "$spec" ] && { echo "[$RID] queue empty — worker done $(date '+%m-%d %H:%M:%S')"; break; }
    EXP="${spec%%|*}"; OV="${spec#*|}"
    run_one "$GPUS" "$PORT" "$RID" "$EXP" "$OV" || true
  done
}

echo "===== AR ChannelViT QUEUE worker-pool START $(date '+%m-%d %H:%M:%S') ====="
echo "queue: $(wc -l < "$QUEUE") trials | workers: qA GPU0-3, qB GPU4-7"
worker 0,1,2,3 29521 qA > "$LOG/260624_ar_cvit_qA.log" 2>&1 &
worker 4,5,6,7 29531 qB > "$LOG/260624_ar_cvit_qB.log" 2>&1 &
wait
echo "===== AR ChannelViT QUEUE worker-pool COMPLETE $(date '+%m-%d %H:%M:%S') ====="
