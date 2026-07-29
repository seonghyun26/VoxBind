#!/bin/bash
# 260629_ar_cvit_g742_patch4.sh — DRAMATIC change #5: finer PATCHES (architectural, no new code).
# Masking algorithms are exhausted (all targeted masking hurts). New axis = spatial resolution: the
# whole campaign used patch_size 8 (2Å patches). Drop to patch_size=4 (1Å patches) so the encoder reads
# density gradients / pocket detail at 2× finer resolution → 16³=4096 patches/group, 12,288 ChannelViT
# tokens (8×). Tests whether the frozen-probe plateau is partly a resolution limit. [7,4,2] base; bsz2 ×
# accum16 × 4gpu = eff-batch 128; compile OFF (12K-token graph). GPU 4-7. Probe lp_edrscc_v2.
set -uf
cd /home/shpark/prj-denovo/VoxBind/voxbind || exit 1
PY=/home/shpark/.conda/envs/voxbind/bin
CFG=config_train_atomblob_density_gradmag_channelvit_mae_40m_plinder_otf_mask050
RES=dataset/data/pdbbind/results
GPUS=4,5,6,7; PORT=29606; RID=arG742P4
EXP=260629_ar_cvit_g742_patch4

echo "===== AR [7,4,2] PATCH=4 (1Å, 12K tokens) START $(date '+%m-%d %H:%M:%S') GPU=$GPUS ====="
CUDA_VISIBLE_DEVICES=$GPUS "$PY/torchrun" --nnodes=1 --nproc_per_node=4 \
    --rdzv-backend=c10d --rdzv-endpoint=localhost:"$PORT" --rdzv-id="$RID" \
    train_density.py --config-name="$CFG" exp_name="$EXP" bsz=2 accum_steps=16 compile.enabled=false \
    model.channel_groups=[7,4,2] model.patch_size=4 \
  && echo "[$EXP] PRETRAIN OK $(date '+%H:%M:%S')" \
  || { echo "[$EXP] PRETRAIN FAILED $(date '+%H:%M:%S')"; exit 1; }
echo "###### [$EXP] PROBE start $(date '+%H:%M:%S') ######"
bash scripts/04_probe.sh --exp "$EXP" --condition atomblob_density_gradmag \
    --tasks affinity --split lp_edrscc_v2 --gpu "${GPUS%%,*}" --tag "$EXP" --num_workers 0 \
    -- --require_density \
  && echo "[$EXP] PROBE OK" || echo "[$EXP] PROBE FAILED"
echo "###### [$EXP] result ######"
tail -2 "$RES/probe_results_e99_v5_lp_edrscc_v2split_${EXP}.csv" 2>/dev/null
echo "===== AR [7,4,2] PATCH=4 COMPLETE $(date '+%m-%d %H:%M:%S') ====="
