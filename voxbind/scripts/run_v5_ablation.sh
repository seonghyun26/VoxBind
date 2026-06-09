#!/bin/bash
# run_v5_ablation.sh — autonomous v5 modality-ablation pipeline.
#   STAGE0  gate on data builds (v5 crops + PDBbind 01a) -> run 01b (CPU)
#   STAGE1  stop sig=1.0 (record resume epoch)
#   STAGE2  pretrain a/b/c (4 GPU each, sequential, ~4h each)
#   STAGE3  probe: frozen features x3 (1 GPU each, parallel) -> probe compare
#   STAGE4  resume sig=1.0
# Safety: if anything fails AFTER sig=1.0 is stopped and before STAGE4, the EXIT
# trap auto-resumes sig=1.0 so 3 days of denoiser compute aren't left idle.
set -u
VOX=/home1/irteam/VoxBind/voxbind
PY=/opt/conda/envs/voxbind/bin
DATA=$VOX/dataset/data
LOG=$VOX/log/run_v5_ablation.log
SIG_EXP=exp_sig1.0+prefetch_factor16_wjs.n_targets0
export PATH=$PY:$PATH
cd "$VOX" || exit 1

ts(){ date "+%F %T"; }
log(){ echo "[$(ts)] $*" | tee -a "$LOG"; }
die(){ log "FATAL: $*"; exit 1; }
gpu_used(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1}END{print s+0}'; }
wait_free(){ local i; for i in $(seq 1 150); do [ "$(gpu_used)" -lt 4000 ] && return 0; sleep 3; done; return 1; }

SIG_STOPPED=0; STAGE4_DONE=0; RESUME_EPOCH=0
resume_sig(){ wait_free || true
  log "resuming sig=1.0 @ep${RESUME_EPOCH}"
  SIG=1.0 NUM_EPOCHS=1200 RESUME_EPOCH=$RESUME_EPOCH setsid nohup bash scripts/35_train_baseline_4gpu.sh \
      >> "$VOX/log/99_watch_pretrain_then_sig1.0.log" 2>&1 & }
cleanup(){ local rc=$?
  if [ "$SIG_STOPPED" = 1 ] && [ "$STAGE4_DONE" != 1 ]; then
    log "cleanup: exited rc=$rc with sig=1.0 stopped before STAGE4 -> auto-resuming sig=1.0"
    resume_sig
  fi; }
trap cleanup EXIT

log "================ run_v5_ablation START ================"

# ---------- STAGE 0: data ----------
log "STAGE0: waiting for v5 crops (00b)…"
while pgrep -f '0[0]b_density_preprocess' >/dev/null 2>&1; do sleep 30; done
[ -f "$DATA/xray_crops_aligned_v5/train_available.npy" ] && [ -f "$DATA/xray_crops_aligned_v5/stats.json" ] || die "v5 crops incomplete"
log "STAGE0: v5 crops ready"
log "STAGE0: waiting for PDBbind acquire (01a)…"
while pgrep -f '01[a]_pdbbind_acquire' >/dev/null 2>&1; do sleep 60; done
[ -f "$DATA/pdbbind/index.csv" ] || die "01a incomplete (no index.csv)"
NV5=$(find "$DATA/pdbbind/voxels_v5/density" -name '*.npy' 2>/dev/null | wc -l)
V5SRC=$($PY/python -c "import json,os;p='$DATA/pdbbind/voxels_v5/stats.json';print(json.load(open(p)).get('stats_source','pdbbind') if os.path.exists(p) else 'missing')" 2>/dev/null)
NAT=$(find "$DATA/pdbbind/voxels/atoms" -name '*.npy' 2>/dev/null | wc -l)
NVDW=$(find "$DATA/pdbbind/voxels_ligvdw/atoms" -name '*.npy' 2>/dev/null | wc -l)
if [ "$NAT" -lt 5000 ]; then
  log "STAGE0: 01b voxelize default atoms+density (CPU)…"
  $PY/python dataset/01b_pdbbind_preprocess.py voxelize --device cpu \
    --out_dir "$DATA/pdbbind/voxels" >> "$VOX/log/01b_pdbbind.log" 2>&1 || die "01b voxelize default"
else
  log "STAGE0: default PDBbind atoms ready (n=$NAT)"
fi
if [ "$NV5" -lt 4000 ] || [ "$V5SRC" != "reference" ]; then
  log "STAGE0: 01b poolnorm v2..v5 with CrossDocked v5 reference stats…"
  $PY/python dataset/01b_pdbbind_preprocess.py poolnorm \
    --v5_stats_source reference \
    --v5_reference_stats_json "$DATA/xray_crops_aligned_v5/stats.json" \
    >> "$VOX/log/01b_pdbbind.log" 2>&1 || die "01b poolnorm"
else
  log "STAGE0: PDBbind v5 density ready (n=$NV5, stats_source=$V5SRC)"
fi
if [ "$NVDW" -lt 5000 ]; then
  log "STAGE0: 01b voxelize ligand-vdW atoms only (CPU)…"
  $PY/python dataset/01b_pdbbind_preprocess.py voxelize --ligand_vdw --no_density --device cpu \
    --out_dir "$DATA/pdbbind/voxels_ligvdw" >> "$VOX/log/01b_pdbbind.log" 2>&1 || die "01b voxelize ligvdw"
else
  log "STAGE0: ligand-vdW PDBbind atoms ready (n=$NVDW)"
fi
[ -d "$DATA/pdbbind/voxels_v5/density" ] || die "voxels_v5 missing"
[ -d "$DATA/pdbbind/voxels_ligvdw/atoms" ] || die "voxels_ligvdw atoms missing"
log "STAGE0: PDBbind voxels ready (default atoms + ligand-vdW atoms + v5 density)"

# ---------- STAGE 1: stop sig=1.0 ----------
log "STAGE1: stopping sig=1.0 (watcher + scripts/35 + train_ddp)…"
pkill -TERM -f '9[9]_watch_n_launch' 2>/dev/null || true
pkill -TERM -f '3[5]_train_baseline' 2>/dev/null || true
pkill -TERM -f '[t]rain_ddp.py'      2>/dev/null || true
sleep 8
pkill -KILL -f '9[9]_watch_n_launch' 2>/dev/null || true
pkill -KILL -f '3[5]_train_baseline' 2>/dev/null || true
pkill -KILL -f '[t]rain_ddp.py'      2>/dev/null || true
wait_free || die "GPUs not free after stopping sig=1.0"
CKE=$($PY/python -c "import torch;print(int(torch.load('$VOX/exps/$SIG_EXP/checkpoint.pth.tar',map_location='cpu',weights_only=False)['epoch']))" 2>/dev/null)
RESUME_EPOCH=$(( ${CKE:-0} + 1 )); SIG_STOPPED=1
log "STAGE1: sig=1.0 stopped (ckpt epoch=${CKE}; resume target=${RESUME_EPOCH}); GPUs free"

# ---------- STAGE 2: pretrain a/b/c ----------
run_pre(){ local cfg=$1 exp=$2
  log "STAGE2: pretrain $exp …"
  CUDA_VISIBLE_DEVICES=0,1,2,3 $PY/torchrun --standalone --nproc_per_node=4 \
    "$VOX/train_density_vit_mae.py" --config-name="$cfg" \
    dset.data_dir="$DATA" exp_name="$exp" \
    output_dir="$VOX/exps/$exp" hydra.run.dir="$VOX/exps/$exp" \
    >> "$VOX/log/${exp}.log" 2>&1
  [ -f "$VOX/exps/$exp/checkpoint_e0099.pth.tar" ] || die "pretrain $exp produced no checkpoint_e0099"
  wait_free || die "GPUs not free after $exp"
  log "STAGE2: $exp DONE"; }
run_pre config_train_atomblob_vit_mae_40m_invfreq                    260606_atomblob_vit_mae_40m_invfreq_pretrain
run_pre config_train_atomblob_density_vit_mae_40m_invfreq_v5         260606_atomblob_density_vit_mae_40m_invfreq_v5_pretrain
run_pre config_train_atomblob_density_gradmag_vit_mae_40m_invfreq_v5 260606_atomblob_density_gradmag_vit_mae_40m_invfreq_v5_pretrain

# ---------- STAGE 3: probe ----------
log "STAGE3: frozen features (3x, 1 GPU each)…"
CUDA_VISIBLE_DEVICES=0 $PY/python dataset/01c_pdbbind_probe.py features --condition atomblob                 --voxel_version v5 --epoch 99 >> "$VOX/log/probe_features.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY/python dataset/01c_pdbbind_probe.py features --condition atomblob_density         --voxel_version v5 --epoch 99 >> "$VOX/log/probe_features.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 $PY/python dataset/01c_pdbbind_probe.py features --condition atomblob_density_gradmag --voxel_version v5 --epoch 99 >> "$VOX/log/probe_features.log" 2>&1 &
wait
log "STAGE3: features done; probe compare…"
CUDA_VISIBLE_DEVICES=0 $PY/python dataset/01c_pdbbind_probe.py probe \
  --conditions atomblob atomblob_density atomblob_density_gradmag \
  --voxel_version v5 --epoch 99 --seeds 3 >> "$VOX/log/probe_compare.log" 2>&1 || die "probe compare"
log "STAGE3: probe DONE -> dataset/data/pdbbind/probe_results_e99*.csv"

# ---------- STAGE 4: resume sig=1.0 ----------
log "STAGE4: resuming sig=1.0…"
resume_sig
sleep 25
STAGE4_DONE=1
log "STAGE4: sig=1.0 resume launched @ep${RESUME_EPOCH}"
log "================ run_v5_ablation COMPLETE ================"
