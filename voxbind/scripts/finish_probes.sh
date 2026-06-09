#!/bin/bash
# finish_probes.sh — complete the STAGE3 probe (gradmag features were broken by a
# missing voxels_ligvdw/atoms dir) then resume sig=1.0. The EXIT trap ALWAYS
# resumes sig=1.0, so it comes back even if a step fails.
set -u
VOX=/home1/irteam/VoxBind/voxbind; PY=/opt/conda/envs/voxbind/bin; DATA=$VOX/dataset/data
LOG=$VOX/log/finish_probes.log; SIG_EXP=exp_sig1.0+prefetch_factor16_wjs.n_targets0
export PATH=$PY:$PATH; cd "$VOX" || exit 1
ts(){ date "+%F %T"; }; log(){ echo "[$(ts)] $*" | tee -a "$LOG"; }
gpu_used(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|awk '{s+=$1}END{print s+0}'; }
wait_free(){ local i; for i in $(seq 1 150); do [ "$(gpu_used)" -lt 4000 ] && return 0; sleep 3; done; return 1; }
RESUME_EPOCH=378
resume_sig(){ wait_free||true; log "resume sig=1.0 @ep$RESUME_EPOCH"; SIG=1.0 NUM_EPOCHS=1200 RESUME_EPOCH=$RESUME_EPOCH \
  setsid nohup bash scripts/35_train_baseline_4gpu.sh >> "$VOX/log/99_watch_pretrain_then_sig1.0.log" 2>&1 & }
trap 'resume_sig' EXIT

log "=== finish_probes START ==="
# 1. ensure the real ligand-vdW PDBbind atom grid exists. Do not symlink this
# back to default atoms: v5 encoders were pretrained with ligand_radius<=0.
NVDW=$(find "$DATA/pdbbind/voxels_ligvdw/atoms" -name '*.npy' 2>/dev/null | wc -l)
if [ "$NVDW" -lt 5000 ]; then
  log "building ligand-vdW PDBbind atoms (CPU)…"
  $PY/python dataset/01b_pdbbind_preprocess.py voxelize --ligand_vdw --no_density --device cpu \
    --out_dir "$DATA/pdbbind/voxels_ligvdw" >> "$VOX/log/01b_pdbbind.log" 2>&1 || { log "FATAL: ligvdw voxelize"; exit 1; }
fi
log "voxels_ligvdw atoms ready ($(find "$DATA/pdbbind/voxels_ligvdw/atoms" -name '*.npy' 2>/dev/null | wc -l) npy)"
V5SRC=$($PY/python -c "import json,os;p='$DATA/pdbbind/voxels_v5/stats.json';print(json.load(open(p)).get('stats_source','pdbbind') if os.path.exists(p) else 'missing')" 2>/dev/null)
if [ "$V5SRC" != "reference" ]; then
  log "rebuilding PDBbind v5 density with CrossDocked reference stats…"
  $PY/python dataset/01b_pdbbind_preprocess.py poolnorm \
    --v5_stats_source reference \
    --v5_reference_stats_json "$DATA/xray_crops_aligned_v5/stats.json" \
    >> "$VOX/log/01b_pdbbind.log" 2>&1 || { log "FATAL: reference poolnorm"; exit 1; }
fi
# 2. stop the (trap-)resumed sig=1.0
log "stopping sig=1.0…"
pkill -TERM -f '9[9]_watch_n_launch' 2>/dev/null||true; pkill -TERM -f '3[5]_train_baseline' 2>/dev/null||true; pkill -TERM -f '[t]rain_ddp.py' 2>/dev/null||true
sleep 8; pkill -KILL -f '3[5]_train_baseline' 2>/dev/null||true; pkill -KILL -f '[t]rain_ddp.py' 2>/dev/null||true
wait_free || { log "FATAL: GPUs not free"; exit 1; }
CKE=$($PY/python -c "import torch;print(int(torch.load('$VOX/exps/$SIG_EXP/checkpoint.pth.tar',map_location='cpu',weights_only=False)['epoch']))" 2>/dev/null)
RESUME_EPOCH=$(( ${CKE:-377}+1 )); log "sig=1.0 stopped (ckpt=$CKE, resume=$RESUME_EPOCH); GPUs free"
# 3. corrected v5 features (GPU)
log "corrected v5 features (GPU0)…"
CUDA_VISIBLE_DEVICES=0 $PY/python dataset/01c_pdbbind_probe.py features --condition atomblob --voxel_version v5 --epoch 99 >> "$VOX/log/probe_features.log" 2>&1
CUDA_VISIBLE_DEVICES=0 $PY/python dataset/01c_pdbbind_probe.py features --condition atomblob_density --voxel_version v5 --epoch 99 >> "$VOX/log/probe_features.log" 2>&1
CUDA_VISIBLE_DEVICES=0 $PY/python dataset/01c_pdbbind_probe.py features --condition atomblob_density_gradmag --voxel_version v5 --epoch 99 >> "$VOX/log/probe_features.log" 2>&1
NF=$($PY/python -c "import torch;print(len(torch.load('$DATA/pdbbind/features/atomblob_density_gradmag_e99_v5.pt')))" 2>/dev/null)
log "gradmag features saved: ${NF:-0}"
[ "${NF:-0}" -ge 100 ] || { log "FATAL: gradmag features still empty"; exit 1; }
# 4. probe compare (all 3)
log "probe compare (atomblob | atomblob_density | atomblob_density_gradmag)…"
CUDA_VISIBLE_DEVICES=0 $PY/python dataset/01c_pdbbind_probe.py probe \
  --conditions atomblob atomblob_density atomblob_density_gradmag --voxel_version v5 --epoch 99 --seeds 3 \
  >> "$VOX/log/probe_compare.log" 2>&1 || { log "FATAL: probe compare failed"; exit 1; }
log "=== finish_probes COMPLETE — probe_results_e99*.csv written; trap resumes sig=1.0 ==="
