#!/usr/bin/env bash
# coords+density (atomblob_density, 12ch) on PLINDER _v2clean, 2 GPUs/run, bsz=64
# (eff-batch 64x2x1=128), the two archs IN PARALLEL: ViT on GPU 0,1 and ChannelViT on
# GPU 2,3 -> higher per-GPU util than 4-GPU-sequential. Stops the denoiser first, runs
# both, then re-arms the keepgoing watcher to resume it. The autoprobe_coordsdens.sh
# watcher (already running) probes both + rebuilds the HTML.
#   setsid nohup bash scripts/archive/chains/chain_plinder_coordsdens_2gpu_then_resume.sh > log/chain_plinder_coordsdens_2gpu.log 2>&1 &
set -uo pipefail
VOX=/home1/irteam/VoxBind/voxbind
PY=/opt/conda/envs/voxbind/bin
LOG=$VOX/log
CROPS=$VOX/dataset/data/xray_crops_aligned_plinder_v2clean
STATS=$CROPS/stats.json
DENOISER_CK=$VOX/exps/exp_sig1.0+prefetch_factor16_wjs.n_targets0/checkpoint.pth.tar
LOCK=$LOG/chain_plinder_coordsdens_2gpu.lock
export LD_LIBRARY_PATH=/opt/conda/envs/voxbind/lib:${LD_LIBRARY_PATH:-}
export TORCHDYNAMO_DISABLE=1 OMP_NUM_THREADS=6
mkdir -p "$LOG"; cd "$VOX" || exit 1
ts(){ date "+%Y-%m-%d %H:%M:%S"; }; log(){ echo "[$(ts)] $*"; }
exec 9>"$LOCK"; flock -n 9 || { log "another instance running — exit"; exit 1; }

# ── pre-flight ────────────────────────────────────────────────────────────────
for c in atomblob_density_vit_mae_40m atomblob_density_channelvit_40m; do
  [ -f "$VOX/configs/config_train_${c}_plinder.yaml" ] || { log "ABORT: missing config $c"; exit 1; }
done
[ -f "$VOX/dataset/data/data_train_plinder_v2clean.pt" ] && [ -f "$STATS" ] || { log "ABORT: _v2clean dataset missing"; exit 1; }
NTRAIN=$("$PY/python" -c "import json;print(json.load(open('$STATS'))['n_train'])")
[ "${NTRAIN:-0}" -ge 1100 ] || { log "ABORT: bad n_train=$NTRAIN"; exit 1; }
SUBSET_N=$(( NTRAIN - 100 ))
log "pre-flight OK: n_train=$NTRAIN subset_n=$SUBSET_N"

# ── stop watcher + denoiser, drain GPUs ───────────────────────────────────────
pkill -9 -f '[c]hain_sig1.0_keepgoing.sh' && log "watcher killed" || log "no watcher"
if pgrep -f '[t]rain_ddp.py' >/dev/null; then pkill -9 -f '[t]rain_ddp.py'; log "denoiser stopped"; sleep 5; else log "no denoiser"; fi
for _ in $(seq 1 30); do U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null|awk '{s+=$1}END{print s+0}'); [ "${U:-99999}" -lt 6000 ] && break; sleep 10; done
log "GPU mem before launch: ${U:-?}"

run_one(){ # gpus port rdzvid cfg exp
  CUDA_VISIBLE_DEVICES="$1" "$PY/torchrun" --nnodes=1 --nproc_per_node=2 \
    --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:"$2" --rdzv_id="$3" \
    "$VOX/train_density.py" --config-name="$4" \
    dset.data_dir="$VOX/dataset/data" dset.subset_n="$SUBSET_N" dset.subset_val_n=100 \
    exp_name="$5" hydra.run.dir="$VOX/exps/$5" > "$LOG/${5}_$(date +%Y%m%d_%H%M%S).log" 2>&1
}

log "LAUNCH ViT on GPU 0,1 (port 29511)"
run_one 0,1 29511 cdvit config_train_atomblob_density_vit_mae_40m_plinder atomblob_density_vit_mae_40m_plinder & PID1=$!
log "stagger 150s so ViT writes the shared val-voxel cache before ChannelViT starts"
sleep 150
log "LAUNCH ChannelViT on GPU 2,3 (port 29512)"
run_one 2,3 29512 cdcvit config_train_atomblob_density_channelvit_40m_plinder atomblob_density_channelvit_40m_plinder & PID2=$!

wait "$PID1"; RC1=$?; log "ViT exit=$RC1"
wait "$PID2"; RC2=$?; log "ChannelViT exit=$RC2"
log "both done (ViT rc=$RC1, ChannelViT rc=$RC2)"

# ── resume the denoiser ───────────────────────────────────────────────────────
if [ -f "$DENOISER_CK" ]; then
  log "re-arming keepgoing watcher to resume the denoiser ..."
  setsid nohup bash "$VOX/scripts/archive/chains/chain_sig1.0_keepgoing.sh" > "$LOG/chain_sig1.0_keepgoing.log" 2>&1 &
  log "  watcher re-armed."
else
  log "WARN: denoiser ckpt missing — NOT re-arming."
fi
log "done. (autoprobe_coordsdens.sh handles probes + results_h200.html)"
