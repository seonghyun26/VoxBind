#!/usr/bin/env bash
# Train the v2 ATOMBLOB-8ch C+D+G ChannelViT on the precomputed boxes, to completion.
# Gates on the atomblob build's ALIGNMENT verdict (won't train on misaligned boxes).
# wait build(aligned) → stop denoiser → train 100ep (boxes, ~85% util) → resume denoiser.
#   setsid nohup bash scripts/chain_atomblob8_train.sh > log/chain_atomblob8.log 2>&1 &
set -uo pipefail
VOX=/home1/irteam/VoxBind/voxbind
B=/opt/conda/envs/voxbind/bin
LOG=$VOX/log
CFG=config_train_atomblob8_density_gradmag_channelvit_mae_40m_plinder_v2_box
EXP=260629_plinder_v2_box_atomblob8_cdg_channelvit_full_pretrain
RD=$VOX/dataset/data/pretrain/xray_resample_plinder_v2
MANIFEST=$RD/train_manifest.npz
TUPLES=$VOX/dataset/data/pretrain/data_train_plinder_v2_atomblob8.pt
DENOISER_CK=$VOX/exps/exp_sig1.0+prefetch_factor16_wjs.n_targets0/checkpoint.pth.tar
BSZ=${BSZ:-32}; NWORK=${NWORK:-16}
LOCK=$LOG/chain_atomblob8.lock
export PATH=$B:${PATH}
export CXX=$B/x86_64-conda-linux-gnu-g++ CC=$B/x86_64-conda-linux-gnu-gcc
export LD_LIBRARY_PATH=/opt/conda/envs/voxbind/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$VOX" || exit 1
ts(){ date "+%Y-%m-%d %H:%M:%S"; }; log(){ echo "[$(ts)] $*"; }
exec 9>"$LOCK"; flock -n 9 || { log "another atomblob8 chain running — exit"; exit 1; }
resume_denoiser(){ [ -f "$DENOISER_CK" ] && { setsid nohup bash "$VOX/scripts/chain_sig1.0_keepgoing.sh" > "$LOG/chain_sig1.0_keepgoing.log" 2>&1 & log "denoiser resume re-armed."; }; }

DEADLINE=$(( $(date +%s) + 12*3600 ))
wait_for(){ local what="$1"; shift; log "waiting for $what ..."; until "$@"; do [ "$(date +%s)" -gt "$DEADLINE" ] && { log "ABORT: timeout on $what"; exit 1; }; sleep 30; done; log "  $what ready"; }

# ── 1) wait for the atomblob build to finish, then GATE on alignment ──────────
wait_for "atomblob8 build" bash -c '[ -f "'"$TUPLES"'" ] && ! pgrep -f "[0]4b_build_atomblob8.py" >/dev/null'
if ! grep -q "ALIGN] ✓" "$LOG/build_atomblob8.log" 2>/dev/null; then
  log "ABORT: atomblob8 build did NOT verify alignment (✓) — refusing to train (boxes would mispair). Denoiser left running."
  exit 1
fi
log "alignment ✓ — atomblob8 tuples reuse the box96 manifest."

# ── 2) subset_n + stop denoiser ──────────────────────────────────────────────
NPOS=$("$B/python" -c "import numpy as np;print(len(np.load('$MANIFEST',allow_pickle=True)['pdb_id']))")
SUBSET_N=$(( NPOS - 100 ))
log "subset_n=$SUBSET_N"
pkill -9 -f '[c]hain_sig1.0_keepgoing.sh' && log "watcher killed" || log "no watcher"
if pgrep -f '[t]rain_ddp.py' >/dev/null; then pkill -9 -f '[t]rain_ddp.py'; log "denoiser stopped"; sleep 5; fi
for _ in $(seq 1 40); do U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|awk '{s+=$1}END{print s+0}'); [ "${U:-99999}" -lt 6000 ] && break; sleep 5; done
log "GPU drained (${U:-?} MiB)"

# ── 3) train to completion (boxes) ───────────────────────────────────────────
RUNLOG="$LOG/${EXP}_$(date +%Y%m%d_%H%M%S).log"
log "LAUNCH atomblob8 train: 4-GPU bsz=$BSZ workers=$NWORK subset_n=$SUBSET_N → $RUNLOG"
CUDA_VISIBLE_DEVICES=0,1,2,3 "$B/torchrun" --standalone --nproc_per_node=4 \
  "$VOX/train_density.py" --config-name="$CFG" \
  bsz="$BSZ" accum_steps=1 num_workers="$NWORK" \
  dset.subset_n="$SUBSET_N" dset.subset_val_n=100 \
  exp_name="$EXP" hydra.run.dir="$VOX/exps/$EXP" > "$RUNLOG" 2>&1
RC=$?; log "atomblob8 train exit=$RC (ckpt: exps/$EXP)"

# ── 4) resume denoiser ───────────────────────────────────────────────────────
resume_denoiser
log "ATOMBLOB8 CHAIN DONE (rc=$RC)."
