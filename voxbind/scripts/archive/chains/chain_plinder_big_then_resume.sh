#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════════
# chain_plinder_big_then_resume.sh
#   Stop the sig=1.0 denoiser (+ its keep-going watcher), run the ChannelViT-on-big-
#   PLINDER density-MAE pretrain on all 4 H200s at max util, then AUTO-RESUME the
#   denoiser (re-arm the keep-going watcher, which extends it +400 epochs).
# This is the user-approved GPU protocol: stop-denoiser -> pretrain -> resume.
#
# Launch (detached, own session so it survives the shell):
#   cd /home1/irteam/VoxBind/voxbind && \
#   setsid nohup bash scripts/archive/chains/chain_plinder_big_then_resume.sh \
#     > log/chain_plinder_big.log 2>&1 &
#
# Stop just this orchestrator (leaves whatever is running): pkill -9 -f '[c]hain_plinder_big_then_resume.sh'
set -uo pipefail

VOX=/home1/irteam/VoxBind/voxbind
PY=/opt/conda/envs/voxbind/bin
LOG=$VOX/log
CFG=config_train_atomblob_density_gradmag_channelvit_plinder_big
EXP=260620_plinder_big_channelvit_cdg_40m_pretrain
CROPS=$VOX/dataset/data/xray_crops_aligned_plinder_big
STATS=$CROPS/stats.json
DENOISER_CK=$VOX/exps/exp_sig1.0+prefetch_factor16_wjs.n_targets0/checkpoint.pth.tar
LOCK=$LOG/chain_plinder_big.lock

# no-C-compiler box: bf16 amp is fine, but kill dynamo just in case (memory: no cc)
export LD_LIBRARY_PATH=/opt/conda/envs/voxbind/lib:${LD_LIBRARY_PATH:-}
export TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=8

mkdir -p "$LOG"; cd "$VOX" || exit 1
ts(){ date "+%Y-%m-%d %H:%M:%S"; }
log(){ echo "[$(ts)] $*"; }

# ── single-instance lock ──────────────────────────────────────────────────────
exec 9>"$LOCK" || { log "cannot open lock $LOCK"; exit 1; }
if ! flock -n 9; then log "another chain_plinder_big is already running — exiting"; exit 1; fi

# ── PRE-FLIGHT: never stop the denoiser unless the dataset is actually ready ───
log "pre-flight: checking big PLINDER dataset is built ..."
if [ ! -f "$VOX/dataset/data/data_train_plinder_big.pt" ] || [ ! -f "$STATS" ] || [ ! -d "$CROPS/train" ]; then
  log "ABORT: dataset not built (need data_train_plinder_big.pt + $STATS + $CROPS/train). Denoiser untouched."
  exit 1
fi
NTRAIN=$("$PY/python" -c "import json;print(json.load(open('$STATS'))['n_train'])" 2>/dev/null)
if [ -z "${NTRAIN:-}" ] || [ "$NTRAIN" -lt 1100 ]; then
  log "ABORT: could not read a sane n_train from $STATS (got '${NTRAIN:-}'). Denoiser untouched."
  exit 1
fi
SUBSET_N=$(( NTRAIN - 100 ))     # val = last 100 of the train-crop pool (disjoint), per 03c
NCROPS=$(ls "$CROPS/train"/*.npy 2>/dev/null | wc -l)
log "pre-flight OK: n_train=$NTRAIN  -> subset_n=$SUBSET_N subset_val_n=100  (crops on disk: $NCROPS)"

# ── 1. stop the keep-going watcher FIRST (else it relaunches the denoiser) ─────
log "stopping chain_sig1.0_keepgoing watcher (if any) ..."
pkill -9 -f '[c]hain_sig1.0_keepgoing.sh' && log "  watcher killed" || log "  no watcher running"
sleep 3

# ── 2. stop the running denoiser (train_ddp.py + its torchrun) by signature ───
# bracket pattern -> pkill never matches its own argv; train_ddp.py != our pretrain
# (train_density_vit_mae.py), so this only touches the denoiser.
if pgrep -f '[t]rain_ddp.py' >/dev/null 2>&1; then
  log "stopping denoiser train_ddp.py ..."
  pkill -9 -f '[t]rain_ddp.py'; sleep 5
  pkill -9 -f '[t]orchrun --standalone --nproc_per_node=4 train_ddp.py' 2>/dev/null || true
else
  log "no denoiser train_ddp.py running"
fi

# ── 3. wait for GPU memory to drain before launching ──────────────────────────
log "waiting for GPU memory to drain ..."
for _ in $(seq 1 24); do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END{print s+0}')
  [ "${USED:-99999}" -lt 6000 ] && break
  sleep 10
done
log "GPU mem used before launch (MiB total across 4): ${USED:-n/a}"

# ── 4. run the ChannelViT-PLINDER-big pretrain (FOREGROUND; blocks ~3.5-4h) ────
RUNLOG="$LOG/${EXP}_$(date +%Y%m%d_%H%M%S).log"
UTILCSV="$LOG/gpu_util_plinder_big.csv"
log "LAUNCH pretrain: cfg=$CFG exp=$EXP subset_n=$SUBSET_N  (log: $RUNLOG)"

# background GPU-util sampler (mean across 4 GPUs every 20s) → for the report
: > "$UTILCSV"
( while true; do
    U=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1;n++} END{if(n)printf "%.1f",s/n}')
    echo "$(date +%s),${U:-0}" >> "$UTILCSV"; sleep 20
  done ) &
SAMPLER=$!

START=$SECONDS
CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY/torchrun" --standalone --nproc_per_node=4 \
  "$VOX/train_density_vit_mae.py" \
  --config-name="$CFG" \
  dset.data_dir="$VOX/dataset/data" \
  dset.subset_n="$SUBSET_N" \
  dset.subset_val_n=100 \
  exp_name="$EXP" \
  hydra.run.dir="$VOX/exps/$EXP" \
  > "$RUNLOG" 2>&1
RC=$?
DUR=$(( SECONDS - START ))
kill "$SAMPLER" 2>/dev/null || true
log "pretrain exit=$RC dur=${DUR}s  (see $RUNLOG)"

# ── 5. AUTO-RESUME the denoiser: re-arm the keep-going watcher ─────────────────
# It will find no train_ddp.py running, read the denoiser ckpt epoch, and relaunch
# +400 epochs (indefinite keep-going restored), whether the pretrain succeeded or not.
if [ -f "$DENOISER_CK" ]; then
  log "re-arming chain_sig1.0_keepgoing watcher to resume the denoiser ..."
  setsid nohup bash "$VOX/scripts/archive/chains/chain_sig1.0_keepgoing.sh" \
    > "$LOG/chain_sig1.0_keepgoing.log" 2>&1 &
  log "  watcher re-armed (it will resume sig=1.0 +400 from the latest ckpt)"
else
  log "WARN: denoiser ckpt $DENOISER_CK missing — NOT re-arming (manual resume needed)"
fi
log "done."
