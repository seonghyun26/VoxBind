#!/bin/bash
# Re-probe Channel-ViT OTF. Its 100ep training finished (e99 ckpt intact) but the first frozen
# probe crashed with a transient "CUDA error: invalid argument" in the pin-memory thread (the
# frozen Channel-ViT probe succeeded earlier, so it's not a channel_group probe bug). Wait until
# the arch+coords chain has fully exited and GPU 0 is idle, then re-extract features + 3-seed probe
# with num_workers=0 (no pin-memory worker) on a clean GPU.
set -u
VOX=/home/shpark/prj-denovo/VoxBind/voxbind
PY=/home/shpark/.conda/envs/voxbind/bin
cd "$VOX" || exit 1
ts(){ date "+%F %T"; }
GATEGPU=0
EXP=260620_plinder_otf_channelvit_vit_mae_40m_pretrain
TAG=plinder_otf_channelvit
CSV=dataset/data/pdbbind/probe_results_e99_v5_${TAG}.csv
LOG=$VOX/log/${EXP}_reprobe.log
CKPT=exps/$EXP/checkpoint_e0099.pth.tar
mkdir -p "$VOX/log"

[ -f "$CKPT" ] || { echo "[$(ts)] [reprobe] ABORT: no e99 ckpt at $CKPT" | tee -a "$LOG"; exit 1; }

# Wait for the arch+coords chain to fully finish (so it isn't using the GPUs), then GPU idle.
echo "[$(ts)] [reprobe] waiting for arch+coords chain to exit ..." | tee -a "$LOG"
while pgrep -f run_plinder_otf_arch_coords_chain.sh >/dev/null 2>&1; do sleep 120; done
echo "[$(ts)] [reprobe] chain exited; waiting for GPU $GATEGPU idle (3×60s)..." | tee -a "$LOG"
free=0
while [ "$free" -lt 3 ]; do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GATEGPU 2>/dev/null)
  if [ -n "$m" ] && [ "$m" -lt 2000 ]; then free=$((free+1)); else free=0; fi
  sleep 60
done

echo "[$(ts)] [reprobe] RE-PROBE $EXP -> $CSV (num_workers=0)" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=$GATEGPU $PY/python dataset/01c_pdbbind_probe.py features \
  --condition atomblob_density_gradmag --voxel_version v5 --epoch 99 --num_workers 0 \
  --atom_source ligvdw --exp_dir "exps/$EXP" --tag "$TAG" >> "$LOG" 2>&1 \
&& CUDA_VISIBLE_DEVICES=$GATEGPU $PY/python dataset/01c_pdbbind_probe.py probe \
  --conditions atomblob_density_gradmag --voxel_version v5 --epoch 99 --seeds 3 \
  --feature_tag "$TAG" --exp_dir "exps/$EXP" --allow_stale_features \
  --out_csv "$CSV" >> "$LOG" 2>&1
rc=$?
[ -f "$CSV" ] && echo "[$(ts)] [reprobe] DONE -> $CSV (rc=$rc)" | tee -a "$LOG" \
              || echo "[$(ts)] [reprobe] STILL FAILED (rc=$rc) — inspect $LOG" | tee -a "$LOG"
