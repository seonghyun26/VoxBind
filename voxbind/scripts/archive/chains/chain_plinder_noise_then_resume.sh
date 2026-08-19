#!/usr/bin/env bash
# Noise-density control experiment, auto-chained AFTER the coords+density campaign
# AND after its auto-probe (so we don't fight for GPU 0). Sequence: wait for both
# coords+density encoders (e0099) + the _v2clean noise crops + the coords+density
# scatter (= auto-probe done) -> stop denoiser -> 4-GPU noise-D+G pretrain
# (torch.compile ON) -> probe on lp_edrscc_v2 (+ scatter_noise_dg_vit.csv) -> resume denoiser.
#   setsid nohup bash scripts/archive/chains/chain_plinder_noise_then_resume.sh > log/chain_plinder_noise.log 2>&1 &
set -uo pipefail
VOX=/home1/irteam/VoxBind/voxbind
B=/opt/conda/envs/voxbind/bin
LOG=$VOX/log
STATS=$VOX/dataset/data/xray_crops_aligned_plinder_v2clean/stats.json
NOISE_TRAIN=$VOX/dataset/data/xray_crops_aligned_plinder_v2clean_noise/train
SCATTER_DIR=/home1/irteam/VoxBind/voxbind/dataset/data/pdbbind/results/scatter
DENOISER_CK=$VOX/exps/exp_sig1.0+prefetch_factor16_wjs.n_targets0/checkpoint.pth.tar
CFG=config_train_noise_density_gradmag_vit_mae_40m_plinder
EXP=noise_density_gradmag_vit_mae_40m_plinder
TAG=noise_dg_vit
LOCK=$LOG/chain_plinder_noise.lock
# torch.compile env (compiler installed 2026-06-24) — do NOT set TORCHDYNAMO_DISABLE
export PATH=$B:${PATH}
export CXX=$B/x86_64-conda-linux-gnu-g++ CC=$B/x86_64-conda-linux-gnu-gcc
export LD_LIBRARY_PATH=/opt/conda/envs/voxbind/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=8
mkdir -p "$LOG"; cd "$VOX" || exit 1
ts(){ date "+%Y-%m-%d %H:%M:%S"; }; log(){ echo "[$(ts)] $*"; }
exec 9>"$LOCK"; flock -n 9 || { log "another noise instance running — exit"; exit 1; }

DEADLINE=$(( $(date +%s) + 8*3600 ))
wait_for(){ local what="$1"; shift; log "waiting for $what ..."; until "$@"; do [ "$(date +%s)" -gt "$DEADLINE" ] && { log "ABORT: timeout on $what"; exit 1; }; sleep 60; done; log "  $what ready"; }

# ── prerequisites ─────────────────────────────────────────────────────────────
wait_for "coords+density ViT e0099"        test -f "exps/atomblob_density_vit_mae_40m_plinder/checkpoint_e0099.pth.tar"
wait_for "coords+density ChannelViT e0099"  test -f "exps/atomblob_density_channelvit_40m_plinder/checkpoint_e0099.pth.tar"
wait_for "noise crops (>=17000)"            bash -c '[ "$(ls '"$NOISE_TRAIN"'/*.npy 2>/dev/null|wc -l)" -ge 17000 ]'
# wait for the coords+density AUTO-PROBE to finish (its last scatter file) so it owns
# GPU 0 alone; only THEN do we grab all 4 GPUs for the noise run.
wait_for "coords+density auto-probe done"   test -f "$SCATTER_DIR/scatter_e99_v5_lp_edrscc_v2_abd_cvit_seed0.csv"
log "all prereqs ready. grace 30s."
sleep 30

NTRAIN=$("$B/python" -c "import json;print(json.load(open('$STATS'))['n_train'])")
SUBSET_N=$(( NTRAIN - 100 ))

# ── stop denoiser + watcher, drain GPUs ───────────────────────────────────────
pkill -9 -f '[c]hain_sig1.0_keepgoing.sh' && log "watcher killed" || log "no watcher"
if pgrep -f '[t]rain_ddp.py' >/dev/null; then pkill -9 -f '[t]rain_ddp.py'; log "denoiser stopped"; sleep 5; fi
for _ in $(seq 1 30); do U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null|awk '{s+=$1}END{print s+0}'); [ "${U:-99999}" -lt 6000 ] && break; sleep 10; done
log "GPU drained (${U:-?}); n_train=$NTRAIN subset_n=$SUBSET_N"

# ── 4-GPU noise pretrain (torch.compile ON) ───────────────────────────────────
RUNLOG="$LOG/${EXP}_$(date +%Y%m%d_%H%M%S).log"
log "LAUNCH noise pretrain (4-GPU, compile ON) → $RUNLOG"
CUDA_VISIBLE_DEVICES=0,1,2,3 "$B/torchrun" --standalone --nproc_per_node=4 \
  "$VOX/train_density.py" --config-name="$CFG" \
  dset.data_dir="$VOX/dataset/data" dset.subset_n="$SUBSET_N" dset.subset_val_n=100 \
  exp_name="$EXP" hydra.run.dir="$VOX/exps/$EXP" > "$RUNLOG" 2>&1
RC=$?; log "noise pretrain exit=$RC"

# ── probe on lp_edrscc_v2 (real pdbbind density; noise-pretrained encoder) + scatter ──
if [ "$RC" -eq 0 ] && [ -f "exps/$EXP/checkpoint_e0099.pth.tar" ]; then
  log "FEATURES + PROBE (density_gradmag, tag=$TAG) on lp_edrscc_v2 ..."
  CUDA_VISIBLE_DEVICES=0 "$B/python" -u dataset/01c_pdbbind_probe.py features \
    --condition density_gradmag --exp_dir "exps/$EXP" --epoch 99 --voxel_version v5 \
    --tag "$TAG" --device cuda --batch_size 24 > "$LOG/${TAG}_features.log" 2>&1
  CUDA_VISIBLE_DEVICES=0 "$B/python" -u dataset/01c_pdbbind_probe.py probe \
    --conditions density_gradmag --exp_dir "exps/$EXP" --epoch 99 --voxel_version v5 \
    --feature_tag "$TAG" --tag "$TAG" --split lp_edrscc_v2 --seeds 3 --device cuda > "$LOG/${TAG}_probe.log" 2>&1
  log "probe done → scatter_${TAG}.csv (in $SCATTER_DIR)"
else
  log "WARN: noise pretrain failed (rc=$RC) — skipping probe"
fi

# ── resume denoiser ───────────────────────────────────────────────────────────
if [ -f "$DENOISER_CK" ]; then
  setsid nohup bash "$VOX/scripts/archive/chains/chain_sig1.0_keepgoing.sh" > "$LOG/chain_sig1.0_keepgoing.log" 2>&1 &
  log "denoiser resume re-armed."
fi
log "NOISE EXPERIMENT DONE."
