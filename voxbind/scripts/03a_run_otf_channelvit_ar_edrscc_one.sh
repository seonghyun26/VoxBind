#!/bin/bash
# Channel-ViT OTF autoresearch — ONE round, probed on the NEW ED+RSCC split (5817/1498/2813).
# Base = Channel-ViT OTF (C+D+G, invfreq, seed42, PLINDER OTF resample, patch_embed_mode=channel_group
# groups [7,4,1,1], learnable PE, bsz8 x accum4, expandable_segments). Each round changes ONE+ Hydra
# override on top, then frozen-probes on the ED+RSCC allowlist:
#   (1) extract features on the full ED pool, (2) filter to ed_rscc_pids.txt, (3) 3-seed probe.
# This mirrors run_pdbbind_edrscc_chain.sh exactly so AR rounds are apples-to-apples vs the new-split
# baselines (edrscc_CDG / edrscc_C). If the e99 ckpt already exists, training is SKIPPED (so this also
# re-probes the existing Channel-ViT base encoder on the new split).
#
# Usage:  bash run_otf_channelvit_ar_edrscc_one.sh <EXP_NAME> <TAG> "<extra hydra overrides>"
set -u
VOX=/home/shpark/prj-denovo/VoxBind/voxbind
PY=/home/shpark/.conda/envs/voxbind/bin
DATA=$VOX/dataset/data
ALLOW=$DATA/pdbbind/ed_rscc_pids.txt
FEAT=$DATA/pdbbind/features
cd "$VOX" || exit 1
ts(){ date "+%F %T"; }
GPUS=0,1,2,3; GATEGPU=0
COND=atomblob_density_gradmag

EXP=$1; TAG=$2; OVERRIDES=${3:-}
# $4 = the COMPLETE channel_groups override token (brackets INSIDE the var survive bash; literal
# brackets around a var get mangled in this env). Default is a static single-quoted token.
CG=${4:-}; [ -z "$CG" ] && CG='model.channel_groups=[7,4,1,1]'
LOG=$VOX/log/$EXP.log
CKPT=exps/$EXP/checkpoint_e0099.pth.tar
EXTAG=plinder_edpool_${TAG}; RSTAG=plinder_edrscc_${TAG}
INF=$FEAT/${COND}_e99_v5_${EXTAG}.pt
OUTF=$FEAT/${COND}_e99_v5_${RSTAG}.pt
CSV=$DATA/pdbbind/probe_results_e99_v5_${TAG}.csv
mkdir -p "$VOX/log"

# ── gate: GPU 0 idle (3x60s), so it can queue behind a running job ──
echo "[$(ts)] [CVAR-ed:$TAG] waiting for GPU $GATEGPU idle (3x60s)..." | tee -a "$LOG"
free=0
while [ "$free" -lt 3 ]; do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GATEGPU 2>/dev/null)
  if [ -n "$m" ] && [ "$m" -lt 2000 ]; then free=$((free+1)); else free=0; fi
  sleep 60
done

# ── train (skip if e99 ckpt already present → base re-probe) ──
if [ -f "$CKPT" ]; then
  echo "[$(ts)] [CVAR-ed:$TAG] e99 ckpt exists → SKIP train, re-probe only" | tee -a "$LOG"
else
  echo "[$(ts)] [CVAR-ed:$TAG] TRAIN $EXP  overrides='$OVERRIDES'  cg_token='$CG'" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=$GPUS PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY/torchrun --standalone --nproc_per_node=4 \
    train_density.py --config-name=config_train_atomblob_density_gradmag_vit_mae_40m_invfreq \
    dset.data_dir="$DATA" dset.crops_dir="" dset.resample_dir="$DATA/xray_resample_plinder" \
    dset.data_file=data_train_plinder.pt dset.subset_n=17430 dset.subset_val_n=100 dset.subset_xray_only=true \
    model.patch_embed_mode=channel_group "$CG" \
    seed=42 num_workers=8 num_epochs=100 bsz=8 accum_steps=4 \
    $OVERRIDES \
    wandb_tags="[pretrain,atomblob_density_gradmag,40m,invfreq,plinder,otf,channelvit,channel_group,autoresearch,edrscc,$TAG]" \
    exp_name="$EXP" output_dir="$VOX/exps/$EXP" >> "$LOG" 2>&1
  echo "[$(ts)] [CVAR-ed:$TAG] train exit=$?" | tee -a "$LOG"
  [ -f "$CKPT" ] || { echo "[$(ts)] [CVAR-ed:$TAG] ABORT: no e99" | tee -a "$LOG"; exit 1; }
fi

# ── (1) extract features on the full ED pool ──
echo "[$(ts)] [CVAR-ed:$TAG] extract features (ED pool) -> $INF" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=$GATEGPU $PY/python dataset/01c_pdbbind_probe.py features \
  --condition "$COND" --voxel_version v5 --epoch 99 --num_workers 0 --atom_source ligvdw \
  --exp_dir "exps/$EXP" --tag "$EXTAG" >> "$LOG" 2>&1
[ -f "$INF" ] || { echo "[$(ts)] [CVAR-ed:$TAG] ABORT: no features ($INF)" | tee -a "$LOG"; exit 1; }

# ── (2) filter to the ED+RSCC allowlist (mirror run_pdbbind_edrscc_chain.sh) ──
echo "[$(ts)] [CVAR-ed:$TAG] filter to RSCC allowlist ($(wc -l <$ALLOW) pids)" | tee -a "$LOG"
$PY/python -c "
import torch
allow=set(open('$ALLOW').read().split())
d=torch.load('$INF',weights_only=False)
d['features']={k:v for k,v in d['features'].items() if k in allow}
torch.save(d,'$OUTF'); print('rscc-filtered pids:',len(d['features']))" >> "$LOG" 2>&1

# ── (3) 3-seed frozen probe on the ED+RSCC split ──
echo "[$(ts)] [CVAR-ed:$TAG] probe (3 seeds) -> $CSV" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=$GATEGPU $PY/python dataset/01c_pdbbind_probe.py probe \
  --conditions "$COND" --voxel_version v5 --epoch 99 --seeds 3 \
  --feature_tag "$RSTAG" --exp_dir "exps/$EXP" --allow_stale_features \
  --out_csv "$CSV" >> "$LOG" 2>&1
[ -f "$CSV" ] && echo "[$(ts)] [CVAR-ed:$TAG] DONE -> $CSV" | tee -a "$LOG" \
              || echo "[$(ts)] [CVAR-ed:$TAG] PROBE FAILED — inspect $LOG" | tee -a "$LOG"
