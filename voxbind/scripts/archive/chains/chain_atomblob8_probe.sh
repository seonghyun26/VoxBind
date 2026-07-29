#!/usr/bin/env bash
# After the atomblob8 training finishes: build the 8-ch pdbbind atom cache (voxels_ligvdw8),
# then frozen-probe on lp_edrscc_v2, then resume the denoiser.
# wait e0099 → stop denoiser → 01b voxelize --other_channel (atoms only, reuse v5 density)
#            → features → probe (3 seeds) → resume denoiser.
#   setsid nohup bash scripts/archive/chains/chain_atomblob8_probe.sh > log/chain_atomblob8_probe.log 2>&1 &
set -uo pipefail
VOX=/home1/irteam/VoxBind/voxbind
B=/opt/conda/envs/voxbind/bin
LOG=$VOX/log
EXP=260629_plinder_v2_box_atomblob8_cdg_channelvit_full_pretrain
CKPT=$VOX/exps/$EXP/checkpoint_e0099.pth.tar
COND=atomblob_density_gradmag                     # → input_mode atomblob_density; cfg n_channels_ligand=8 → 14ch + voxels_ligvdw8
TAG=atomblob8_cdg_cvit
VOX8=$VOX/dataset/data/pdbbind/voxels_ligvdw8
DENOISER_CK=$VOX/exps/exp_sig1.0+prefetch_factor16_wjs.n_targets0/checkpoint.pth.tar
LOCK=$LOG/chain_atomblob8_probe.lock
export PATH=$B:${PATH}
export CXX=$B/x86_64-conda-linux-gnu-g++ CC=$B/x86_64-conda-linux-gnu-gcc
export LD_LIBRARY_PATH=/opt/conda/envs/voxbind/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=8
cd "$VOX" || exit 1
ts(){ date "+%Y-%m-%d %H:%M:%S"; }; log(){ echo "[$(ts)] $*"; }
exec 9>"$LOCK"; flock -n 9 || { log "another atomblob8 probe running — exit"; exit 1; }
resume_denoiser(){ [ -f "$DENOISER_CK" ] && { setsid nohup bash "$VOX/scripts/archive/chains/chain_sig1.0_keepgoing.sh" > "$LOG/chain_sig1.0_keepgoing.log" 2>&1 & log "denoiser resume re-armed."; }; }

DEADLINE=$(( $(date +%s) + 24*3600 ))
wait_for(){ local what="$1"; shift; log "waiting for $what ..."; until "$@"; do [ "$(date +%s)" -gt "$DEADLINE" ] && { log "ABORT: timeout on $what"; exit 1; }; sleep 60; done; log "  $what ready"; }

# ── 1) wait for training to finish (e0099 + train proc gone) ─────────────────
# NB: bracket-trick the pattern ([t]) so this check's own `bash -c` argv (which contains
# the literal pattern) is not matched by pgrep — else the wait self-matches forever.
wait_for "atomblob8 training (e0099)" bash -c '[ -f "'"$CKPT"'" ] && ! pgrep -f "[t]rain_density.py.*atomblob8" >/dev/null'
log "training done."

# ── 2) stop denoiser (the train chain resumed it), drain GPUs ────────────────
pkill -9 -f '[c]hain_sig1.0_keepgoing.sh' && log "watcher killed" || log "no watcher"
if pgrep -f '[t]rain_ddp.py' >/dev/null; then pkill -9 -f '[t]rain_ddp.py'; log "denoiser stopped"; sleep 5; fi
for _ in $(seq 1 40); do U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|awk '{s+=$1}END{print s+0}'); [ "${U:-99999}" -lt 6000 ] && break; sleep 5; done
log "GPU drained (${U:-?} MiB)"

# ── 3) build the 8-ch pdbbind atom cache (atoms only; density reused from v5) ─
if [ ! -f "$VOX8/metadata.json" ]; then
  log "VOXELIZE 8-ch atoms → $VOX8 (--other_channel --ligand_vdw --no_density) ..."
  CUDA_VISIBLE_DEVICES=0 "$B/python" -u dataset/legacy/01b_pdbbind_preprocess.py voxelize \
    --other_channel --ligand_vdw --no_density --device cuda \
    --out_dir "$VOX8" --overwrite > "$LOG/${TAG}_voxelize.log" 2>&1
  RCV=$?; log "voxelize exit=$RCV"
  [ "$RCV" -eq 0 ] || { log "VOXELIZE FAILED — see $LOG/${TAG}_voxelize.log"; resume_denoiser; exit 1; }
else
  log "voxels_ligvdw8 already present — skipping voxelize."
fi

# ── 4) features + probe (lp_edrscc_v2, 3 seeds) ──────────────────────────────
log "FEATURES (cond=$COND tag=$TAG) ..."
CUDA_VISIBLE_DEVICES=0 "$B/python" -u dataset/01c_pdbbind_probe.py features \
  --condition "$COND" --exp_dir "exps/$EXP" --epoch 99 --voxel_version v5 \
  --atom_source ligvdw --tag "$TAG" --device cuda --batch_size 24 > "$LOG/${TAG}_features.log" 2>&1
RC1=$?; log "features exit=$RC1"
if [ "$RC1" -eq 0 ]; then
  log "PROBE lp_edrscc_v2 (3 seeds) ..."
  CUDA_VISIBLE_DEVICES=0 "$B/python" -u dataset/01c_pdbbind_probe.py probe \
    --conditions "$COND" --exp_dir "exps/$EXP" --epoch 99 --voxel_version v5 \
    --feature_tag "$TAG" --tag "$TAG" --split lp_edrscc_v2 --seeds 3 --device cuda \
    > "$LOG/${TAG}_probe.log" 2>&1
  RC2=$?; log "probe exit=$RC2"
else
  log "FEATURES FAILED — see $LOG/${TAG}_features.log"; RC2=1
fi

# ── 5) resume denoiser ───────────────────────────────────────────────────────
resume_denoiser
log "ATOMBLOB8 PROBE CHAIN DONE (vox/feat/probe). Results: dataset/data/pdbbind/results/probe_results_e99_v5_lp_edrscc_v2split_${TAG}.csv"
