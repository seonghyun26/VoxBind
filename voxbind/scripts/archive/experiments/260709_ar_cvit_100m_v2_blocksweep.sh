#!/bin/bash
# 260709 BLOCK-SIZE sweep at the BEST recipe (d640/L18/h10 @ mask0.75, v2, 50ep = 0.644).
# Re-tests the mask GEOMETRY at the new recipe — block size was only swept at v1/mask0.50/40M
# (block4=1Å hurt 0.583, block8=2Å best, cluster/large hurt). Does the block-size optimum shift
# now that the mask RATIO is 0.75? mae.block_size in VOXELS (grid 64³ @0.25Å): 4→1Å, 8→2Å, 16→4Å.
# usage: blocksweep.sh <GPUS csv> <PORT> <BLOCK>
set -uf
cd /home/shpark/prj-denovo/VoxBind/voxbind || exit 1
GPUS="$1"; PORT="$2"; BLOCK="$3"
PY=/home/shpark/.conda/envs/voxbind/bin
CFG=config_train_atomblob_density_gradmag_channelvit_mae_40m_plinder_otf_mask050
RES=dataset/data/pdbbind/results
EPOCHS=50; PROBE_EP=49
EXP=260709_ar_cvit_100m_v2_block${BLOCK}_m075
NG=$(echo "$GPUS" | tr ',' '\n' | wc -l)

echo "===== BLOCK=$BLOCK · d640/L18/h10 @ mask0.75 START $(date '+%m-%d %H:%M:%S') GPU=$GPUS ====="
CUDA_VISIBLE_DEVICES=$GPUS "$PY/torchrun" --nnodes=1 --nproc_per_node="$NG" \
    --rdzv-backend=c10d --rdzv-endpoint=localhost:"$PORT" --rdzv-id=cvitBlk$BLOCK \
    train_density.py --config-name="$CFG" exp_name="$EXP" bsz=4 accum_steps=8 \
    num_workers=6 prefetch_factor=2 num_epochs=$EPOCHS \
    model.channel_groups=[7,4,2] model.dim=640 model.depth=18 model.heads=10 \
    mae.mask_ratio=0.75 mae.block_size=$BLOCK \
    dset.data_file=pretrain/data_train_plinder_v2_perelem.pt \
    dset.resample_dir=dataset/data/pretrain/xray_resample_plinder_v2_perelem \
    dset.subset_n=112000 dset.subset_val_n=100 \
  && echo "[$EXP] PRETRAIN OK $(date '+%H:%M:%S')" \
  || { echo "[$EXP] PRETRAIN FAILED $(date '+%H:%M:%S')"; exit 1; }

echo "###### [$EXP] PROBE start $(date '+%H:%M:%S') ######"
bash scripts/04_probe.sh --exp "$EXP" --condition atomblob_density_gradmag \
    --tasks affinity --split lp_edrscc_v2 --epoch $PROBE_EP --gpu "${GPUS%%,*}" \
    --tag "$EXP" --num_workers 0 -- --require_density \
  && echo "[$EXP] PROBE OK" || echo "[$EXP] PROBE FAILED"
echo "###### [$EXP] result ######"
tail -2 "$RES/probe_results_e${PROBE_EP}_v5_lp_edrscc_v2split_${EXP}.csv" 2>/dev/null
echo "===== BLOCK=$BLOCK COMPLETE $(date '+%m-%d %H:%M:%S') ====="
