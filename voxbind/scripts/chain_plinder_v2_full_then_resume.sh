#!/usr/bin/env bash
# PLINDER v2 FULL pretrain, end-to-end, auto-chained. Sequence:
#   wait for 03_download to finish -> full 04_build -> derive subset_n from manifest ->
#   stop denoiser+watcher, drain GPUs -> probe max bsz (ladder, compile ON) ->
#   4-GPU OTF ChannelViT C+D+G pretrain (full v2 set) -> resume denoiser.
#
#   setsid nohup bash scripts/chain_plinder_v2_full_then_resume.sh > log/chain_plinder_v2_full.log 2>&1 &
set -uo pipefail
VOX=/home1/irteam/VoxBind/voxbind
B=/opt/conda/envs/voxbind/bin
LOG=$VOX/log
CFG=config_train_roleblob_diverse_density_gradmag_channelvit_mae_40m_plinder_v2_full
EXP=260627_plinder_v2_otf_roleblob_diverse_cdg_channelvit_full_pretrain
MANIFEST=$VOX/dataset/data/pretrain/xray_resample_plinder_v2/train_manifest.npz
DENOISER_CK=$VOX/exps/exp_sig1.0+prefetch_factor16_wjs.n_targets0/checkpoint.pth.tar
BSZ_LADDER="32 24 16"
NWORK=24
LOCK=$LOG/chain_plinder_v2_full.lock
# torch.compile env (compiler installed 2026-06-24)
export PATH=$B:${PATH}
export CXX=$B/x86_64-conda-linux-gnu-g++ CC=$B/x86_64-conda-linux-gnu-gcc
export LD_LIBRARY_PATH=/opt/conda/envs/voxbind/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=8
mkdir -p "$LOG"; cd "$VOX" || exit 1
ts(){ date "+%Y-%m-%d %H:%M:%S"; }; log(){ echo "[$(ts)] $*"; }
exec 9>"$LOCK"; flock -n 9 || { log "another v2 instance running — exit"; exit 1; }

DEADLINE=$(( $(date +%s) + 18*3600 ))
wait_for(){ local what="$1"; shift; log "waiting for $what ..."; until "$@"; do [ "$(date +%s)" -gt "$DEADLINE" ] && { log "ABORT: timeout on $what"; exit 1; }; sleep 60; done; log "  $what ready"; }

# ── 1) wait for the background download to finish ─────────────────────────────
wait_for "03_download to finish"  bash -c '! pgrep -f "[0]3_download.py" >/dev/null'
NCIF=$(find "$VOX/dataset/data/cif" -maxdepth 1 -name '*.cif' -printf . 2>/dev/null | wc -c)
NMAP=$(find "$VOX/dataset/data/ccp4" -maxdepth 1 -name '*.ccp4' -printf . 2>/dev/null | wc -c)
log "download done: cif=$NCIF ccp4=$NMAP"
if [ "$NCIF" -lt 30000 ]; then log "ABORT: only $NCIF cifs — download looks incomplete"; exit 1; fi

# ── 2) full build (no --limit) ───────────────────────────────────────────────
log "FULL BUILD (04_build.py) → ~32 min ..."
"$B/python" "$VOX/dataset/plinder/04_build.py" > "$LOG/plinder_v2_build.log" 2>&1
RC=$?; log "build exit=$RC"
[ "$RC" -eq 0 ] && [ -f "$MANIFEST" ] || { log "ABORT: build failed / no manifest"; exit 1; }

# ── 3) derive subset_n from the built manifest ───────────────────────────────
NPOS=$("$B/python" -c "import numpy as np;print(len(np.load('$MANIFEST',allow_pickle=True)['pdb_id']))")
SUBSET_N=$(( NPOS - 100 ))
log "manifest positions=$NPOS → subset_n=$SUBSET_N subset_val_n=100"
[ "$SUBSET_N" -gt 1000 ] || { log "ABORT: subset_n=$SUBSET_N too small"; exit 1; }

# ── 4) stop denoiser + watcher, drain GPUs ───────────────────────────────────
pkill -9 -f '[c]hain_sig1.0_keepgoing.sh' && log "watcher killed" || log "no watcher"
if pgrep -f '[t]rain_ddp.py' >/dev/null; then pkill -9 -f '[t]rain_ddp.py'; log "denoiser stopped"; sleep 5; fi
for _ in $(seq 1 30); do U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null|awk '{s+=$1}END{print s+0}'); [ "${U:-99999}" -lt 6000 ] && break; sleep 10; done
log "GPU drained (${U:-?} MiB)"

# ── 5) probe max safe bsz (1 GPU, compile ON = faithful memory, tiny subset) ──
BSZ=""
for cand in $BSZ_LADDER; do
  log "probe bsz=$cand (1-GPU, compile ON) ..."
  PLOG="$LOG/v2_bsz_probe_${cand}.log"
  CUDA_VISIBLE_DEVICES=0 timeout 600 "$B/python" -u "$VOX/train_density.py" --config-name="$CFG" \
    exp_name="_bszprobe_${cand}" hydra.run.dir="$VOX/exps/_bszprobe_${cand}" wandb=false \
    bsz="$cand" accum_steps=1 num_epochs=1 num_workers=6 prefetch_factor=4 \
    dset.subset_n=128 dset.subset_val_n=32 mae.val_every=999 mae.ckpt_every=999 \
    > "$PLOG" 2>&1
  prc=$?
  if [ "$prc" -eq 0 ] && ! grep -qiE "out of memory|OutOfMemoryError|CUDA error" "$PLOG"; then
    BSZ="$cand"; log "  bsz=$cand FITS → using it"; rm -rf "$VOX/exps/_bszprobe_${cand}"; break
  fi
  log "  bsz=$cand failed (rc=$prc / OOM) — trying smaller"; rm -rf "$VOX/exps/_bszprobe_${cand}"
done
[ -n "$BSZ" ] || { log "ABORT: no bsz in ladder fit"; BSZ=8; log "fallback bsz=8"; }

# ── 6) full 4-GPU OTF pretrain (compile ON) ──────────────────────────────────
RUNLOG="$LOG/${EXP}_$(date +%Y%m%d_%H%M%S).log"
log "LAUNCH v2 pretrain: 4-GPU bsz=$BSZ workers=$NWORK subset_n=$SUBSET_N → $RUNLOG"
CUDA_VISIBLE_DEVICES=0,1,2,3 "$B/torchrun" --standalone --nproc_per_node=4 \
  "$VOX/train_density.py" --config-name="$CFG" \
  bsz="$BSZ" accum_steps=1 num_workers="$NWORK" \
  dset.subset_n="$SUBSET_N" dset.subset_val_n=100 \
  exp_name="$EXP" hydra.run.dir="$VOX/exps/$EXP" > "$RUNLOG" 2>&1
RC=$?; log "v2 pretrain exit=$RC (ckpt: exps/$EXP)"

# ── 7) resume denoiser ───────────────────────────────────────────────────────
if [ -f "$DENOISER_CK" ]; then
  setsid nohup bash "$VOX/scripts/chain_sig1.0_keepgoing.sh" > "$LOG/chain_sig1.0_keepgoing.log" 2>&1 &
  log "denoiser resume re-armed."
fi
log "V2 FULL PRETRAIN CHAIN DONE."
