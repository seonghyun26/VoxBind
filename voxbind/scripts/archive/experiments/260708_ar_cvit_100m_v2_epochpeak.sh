#!/bin/bash
# 260708 H7 EPOCH-PEAK: the winner d640/L18/h10 @ mask0.75 peaks by e49 (0.644) and OVERSHOOTS by
# e99 (0.618). Where is the real peak — is it BELOW 50ep? Re-run the winner with mae.ckpt_every=10
# and probe e19/29/39/49/59 to map the frozen-probe-vs-epoch curve. Same seed+constant LR → faithfully
# reconstructs the winner's early epochs. If the peak is <50ep, the whole grid (all 100ep) was over-trained.
set -uf
cd /home/shpark/prj-denovo/VoxBind/voxbind || exit 1
PY=/home/shpark/.conda/envs/voxbind/bin
CFG=config_train_atomblob_density_gradmag_channelvit_mae_40m_plinder_otf_mask050
RES=dataset/data/pdbbind/results
GPUS=4,5,6,7; PORT=29591
EXP=260708_ar_cvit_100m_v2_epochpeak

echo "===== H7 epoch-peak (d640/L18/h10 @ mask0.75, ckpt_every=10) START $(date '+%m-%d %H:%M:%S') ====="
CUDA_VISIBLE_DEVICES=$GPUS "$PY/torchrun" --nnodes=1 --nproc_per_node=4 \
    --rdzv-backend=c10d --rdzv-endpoint=localhost:"$PORT" --rdzv-id=cvitEpochPeak \
    train_density.py --config-name="$CFG" exp_name="$EXP" bsz=4 accum_steps=8 \
    num_workers=6 prefetch_factor=2 num_epochs=60 mae.ckpt_every=10 \
    model.channel_groups=[7,4,2] model.dim=640 model.depth=18 model.heads=10 \
    mae.mask_ratio=0.75 \
    dset.data_file=pretrain/data_train_plinder_v2_perelem.pt \
    dset.resample_dir=dataset/data/pretrain/xray_resample_plinder_v2_perelem \
    dset.subset_n=112000 dset.subset_val_n=100 \
  && echo "[$EXP] PRETRAIN OK $(date '+%H:%M:%S')" \
  || { echo "[$EXP] PRETRAIN FAILED $(date '+%H:%M:%S')"; exit 1; }

for EP in 19 29 39 49 59; do
  CK="exps/$EXP/checkpoint_e$(printf '%04d' $EP).pth.tar"
  [ -f "$CK" ] || { echo "[$EXP] no ckpt e$EP, skip"; continue; }
  echo "###### [$EXP] PROBE e$EP $(date '+%H:%M:%S') ######"
  bash scripts/04_probe.sh --exp "$EXP" --condition atomblob_density_gradmag \
      --tasks affinity --split lp_edrscc_v2 --epoch $EP --gpu 4 \
      --tag "${EXP}_e${EP}" --num_workers 0 -- --require_density \
    && tail -1 "$RES/probe_results_e${EP}_v5_lp_edrscc_v2split_${EXP}_e${EP}.csv" 2>/dev/null
done
echo "===== H7 epoch-peak COMPLETE $(date '+%m-%d %H:%M:%S') ====="
