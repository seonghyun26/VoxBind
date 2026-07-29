#!/usr/bin/env bash
# Box-training chain: wait for the 96³ precompute → stop denoiser → train on boxes →
# measure GPU util → KEEP (util ≥ THRESH, run to completion) or REVERT (kill, resume denoiser).
# Either way the denoiser is resumed at the end. Per user: "if util improved keep, else resume."
#   THRESH=78 setsid nohup bash scripts/archive/chains/chain_box_train_decide.sh > log/chain_box.log 2>&1 &
set -uo pipefail
VOX=/home1/irteam/VoxBind/voxbind
B=/opt/conda/envs/voxbind/bin
LOG=$VOX/log
CFG=config_train_roleblob_diverse_density_gradmag_channelvit_mae_40m_plinder_v2_box
EXP=260629_plinder_v2_box_roleblob_diverse_cdg_channelvit_full_pretrain
RD=$VOX/dataset/data/pretrain/xray_resample_plinder_v2
MANIFEST=$RD/train_manifest.npz
BOXMETA=$RD/box96_meta.json
DENOISER_CK=$VOX/exps/exp_sig1.0+prefetch_factor16_wjs.n_targets0/checkpoint.pth.tar
THRESH=${THRESH:-78}
BSZ=${BSZ:-32}; NWORK=${NWORK:-16}
LOCK=$LOG/chain_box.lock
export PATH=$B:${PATH}
export CXX=$B/x86_64-conda-linux-gnu-g++ CC=$B/x86_64-conda-linux-gnu-gcc
export LD_LIBRARY_PATH=/opt/conda/envs/voxbind/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$VOX" || exit 1
ts(){ date "+%Y-%m-%d %H:%M:%S"; }; log(){ echo "[$(ts)] $*"; }
exec 9>"$LOCK"; flock -n 9 || { log "another box chain running — exit"; exit 1; }

resume_denoiser(){
  if [ -f "$DENOISER_CK" ]; then
    setsid nohup bash "$VOX/scripts/archive/chains/chain_sig1.0_keepgoing.sh" > "$LOG/chain_sig1.0_keepgoing.log" 2>&1 &
    log "denoiser resume re-armed."
  fi
}

DEADLINE=$(( $(date +%s) + 12*3600 ))
wait_for(){ local what="$1"; shift; log "waiting for $what ..."; until "$@"; do [ "$(date +%s)" -gt "$DEADLINE" ] && { log "ABORT: timeout on $what"; exit 1; }; sleep 30; done; log "  $what ready"; }

# ── 1) wait for the 96³ precompute to finish ─────────────────────────────────
wait_for "box precompute" bash -c '[ -f "'"$BOXMETA"'" ] && ! pgrep -f "[0]5_make_boxes.py" >/dev/null'
NPOS=$("$B/python" -c "import numpy as np;print(len(np.load('$MANIFEST',allow_pickle=True)['pdb_id']))")
NBOX=$("$B/python" -c "import numpy as np,json;m=json.load(open('$BOXMETA'));mm=np.memmap('$RD/box96.dat',dtype=np.float16,mode='r',shape=(m['n'],m['g_box'],m['g_box'],m['g_box']));print(int((np.abs(mm[::997]).sum(axis=(1,2,3))>0).sum()))")
SUBSET_N=$(( NPOS - 100 ))
log "precompute done: manifest=$NPOS  sampled-nonzero≈$NBOX/$((NPOS/997+1))  subset_n=$SUBSET_N"

# ── 2) stop denoiser + watcher, drain GPUs ───────────────────────────────────
pkill -9 -f '[c]hain_sig1.0_keepgoing.sh' && log "watcher killed" || log "no watcher"
if pgrep -f '[t]rain_ddp.py' >/dev/null; then pkill -9 -f '[t]rain_ddp.py'; log "denoiser stopped"; sleep 5; fi
for _ in $(seq 1 40); do U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|awk '{s+=$1}END{print s+0}'); [ "${U:-99999}" -lt 6000 ] && break; sleep 5; done
log "GPU drained (${U:-?} MiB)"

# ── 3) launch box training (4-GPU) in the BACKGROUND so we can measure ────────
RUNLOG="$LOG/${EXP}_$(date +%Y%m%d_%H%M%S).log"
log "LAUNCH box train: 4-GPU bsz=$BSZ workers=$NWORK subset_n=$SUBSET_N → $RUNLOG"
CUDA_VISIBLE_DEVICES=0,1,2,3 "$B/torchrun" --standalone --nproc_per_node=4 \
  "$VOX/train_density.py" --config-name="$CFG" \
  bsz="$BSZ" accum_steps=1 num_workers="$NWORK" \
  dset.subset_n="$SUBSET_N" dset.subset_val_n=100 \
  exp_name="$EXP" hydra.run.dir="$VOX/exps/$EXP" > "$RUNLOG" 2>&1 &
TPID=$!
log "torchrun pid=$TPID"

# ── 4) wait for warm (epoch 0 done = compile over) then measure util 60s ──────
for i in $(seq 1 60); do
  kill -0 "$TPID" 2>/dev/null || { log "ABORT: training died during warmup (see $RUNLOG)"; resume_denoiser; exit 1; }
  grep -qiE "out of memory|Error executing job|Traceback" "$RUNLOG" 2>/dev/null && { log "ABORT: training error"; kill -9 "$TPID" 2>/dev/null; resume_denoiser; exit 1; }
  tr '\r' '\n' < "$RUNLOG" 2>/dev/null | grep -qE "epoch: 0 \(" && break
  sleep 15
done
log "warm; measuring util (60s, val_every=5 → clean mid-epoch)"
SUM=0; N=0
for s in $(seq 1 30); do
  for v in $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits); do SUM=$((SUM+v)); N=$((N+1)); done
  sleep 2
done
AVG=$(( SUM / (N>0?N:1) ))
ET=$(tr '\r' '\n' < "$RUNLOG" 2>/dev/null | grep -oE "epoch: 0 \([0-9.]+s\)" | head -1)
log "MEASURED util=${AVG}%  (threshold ${THRESH}%)  ${ET}"

# ── 5) decide: keep (run to completion) or revert (kill + resume denoiser) ────
if [ "$AVG" -ge "$THRESH" ]; then
  log "KEEP: util ${AVG}% ≥ ${THRESH}% — letting box training run to completion."
  wait "$TPID"; RC=$?
  log "box training finished (rc=$RC, ckpt: exps/$EXP)."
  resume_denoiser
  log "BOX CHAIN DONE (kept; util=${AVG}%)."
else
  log "REVERT: util ${AVG}% < ${THRESH}% — stopping box training, resuming denoiser."
  kill -9 "$TPID" 2>/dev/null; pkill -9 -f '[v]2_box' 2>/dev/null; sleep 5
  resume_denoiser
  log "BOX CHAIN DONE (reverted; util=${AVG}%)."
fi
