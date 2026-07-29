#!/bin/bash
# 260629_ar_cvit_g742_crossmodal.sh — DRAMATIC change #2: CROSS-MODAL masking (new code).
# Opt-in mae.modal_mask_prob (default 0 = no-op; uniform spatial mask + atom_biased/cluster
# all untouched). On top of the usual 50% spatial MAE mask, per-sample with prob p, drop the
# ENTIRE density+gradmag modality from the input (zero channels density_idx/gradmag_idx) so the
# encoder must PREDICT density from the atom channels alone — a cross-modal objective. Target is
# unchanged, so the recon loss scores the atoms→density mapping. Tests whether learning the
# physics map atoms↔density yields a better representation than spatial inpainting. p=0.5 on the
# [7,4,2] winner. GPU 0-3 (4-GPU eff-128). Probe lp_edrscc_v2.
set -uf
cd /home/shpark/prj-denovo/VoxBind/voxbind || exit 1
PY=/home/shpark/.conda/envs/voxbind/bin
CFG=config_train_atomblob_density_gradmag_channelvit_mae_40m_plinder_otf_mask050
RES=dataset/data/pdbbind/results
GPUS=0,1,2,3; PORT=29602; RID=arG742CM
EXP=260629_ar_cvit_g742_crossmodal

echo "===== AR [7,4,2] CROSS-MODAL MASK (p=0.5) START $(date '+%m-%d %H:%M:%S') GPU=$GPUS ====="
CUDA_VISIBLE_DEVICES=$GPUS "$PY/torchrun" --nnodes=1 --nproc_per_node=4 \
    --rdzv-backend=c10d --rdzv-endpoint=localhost:"$PORT" --rdzv-id="$RID" \
    train_density.py --config-name="$CFG" exp_name="$EXP" bsz=8 accum_steps=4 \
    model.channel_groups=[7,4,2] +mae.modal_mask_prob=0.5 \
  && echo "[$EXP] PRETRAIN OK $(date '+%H:%M:%S')" \
  || { echo "[$EXP] PRETRAIN FAILED $(date '+%H:%M:%S')"; exit 1; }
echo "###### [$EXP] PROBE start $(date '+%H:%M:%S') ######"
bash scripts/04_probe.sh --exp "$EXP" --condition atomblob_density_gradmag \
    --tasks affinity --split lp_edrscc_v2 --gpu "${GPUS%%,*}" --tag "$EXP" --num_workers 0 \
    -- --require_density \
  && echo "[$EXP] PROBE OK" || echo "[$EXP] PROBE FAILED"
echo "###### [$EXP] result ######"
tail -2 "$RES/probe_results_e99_v5_lp_edrscc_v2split_${EXP}.csv" 2>/dev/null
echo "===== AR [7,4,2] CROSS-MODAL COMPLETE $(date '+%m-%d %H:%M:%S') ====="
