#!/bin/bash
# 260622_ar_trial.sh — one Phase-3 autoresearch trial for the plain-ViT C+D+G encoder:
#   pretrain (4-GPU DDP, 100 ep) on the PLINDER-OTF mask050 base config, then frozen
#   affinity probe on lp_edrscc (3 seeds). One knob is varied via the trailing Hydra
#   overrides. Result CSV: dataset/data/pdbbind/results/probe_results_e99_v5_lp_edrsccsplit_<EXP>.csv
#
# Usage: bash 260622_ar_trial.sh <exp_name> <cuda_set> <probe_gpu> <hydra overrides...>
#   e.g. bash 260622_ar_trial.sh 260622_ar_vit_t1_atombias 0,1,2,3 0 mae.mask_strategy=atom_biased
set -u
VOX=/home/shpark/prj-denovo/VoxBind/voxbind
PY=/home/shpark/.conda/envs/voxbind/bin
cd "$VOX" || exit 1
EXP="$1"; CUDA="$2"; PGPU="$3"; shift 3
NPROC=$(awk -F, '{print NF}' <<< "$CUDA")   # GPU count derived from the CUDA set
BASE=config_train_atomblob_density_gradmag_vit_mae_40m_invfreq_plinder_otf_mask050
LOG="log/${EXP}.log"
ts(){ date "+%Y-%m-%d %H:%M:%S"; }

echo "[$(ts)] AR trial $EXP | cuda=$CUDA (nproc=$NPROC) | probe_gpu=$PGPU | overrides: $*" > "$LOG"

# ── pretrain ──────────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES="$CUDA" "$PY/torchrun" --standalone --nproc_per_node="$NPROC" train_density.py \
  --config-name="$BASE" exp_name="$EXP" \
  "wandb_tags=[pretrain,autoresearch,vit,cdg,plinder,otf,mask050]" "$@" >> "$LOG" 2>&1
rc=$?
echo "[$(ts)] pretrain exit=$rc" >> "$LOG"
if [ "$rc" -ne 0 ] || [ ! -f "exps/$EXP/checkpoint_e0099.pth.tar" ]; then
  echo "[$(ts)] TRIAL $EXP PRETRAIN FAILED (rc=$rc, ckpt missing?)" >> "$LOG"
  echo "AR_TRIAL_${EXP}_FAILED"
  exit 2
fi

# ── frozen probe (lp_edrscc, 3 seeds, robust num_workers=0) ────────────────────
bash scripts/04_probe.sh --exp "$EXP" --condition atomblob_density_gradmag --tasks affinity \
  --split lp_edrscc --atom_source ligvdw --seeds 3 --gpu "$PGPU" --tag "$EXP" \
  --epoch 99 --voxel v5 --num_workers 0 >> "$LOG" 2>&1
prc=$?
echo "[$(ts)] probe exit=$prc" >> "$LOG"
echo "[$(ts)] AR TRIAL $EXP DONE (probe rc=$prc)" >> "$LOG"
echo "AR_TRIAL_${EXP}_DONE"
