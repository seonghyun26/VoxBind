#!/bin/bash
# 260715 HCS (channel-group dropout) on the 100M v2 recipe. HCS randomly drops whole channel
# GROUPS per forward (train-only, keep ≥1) — with groups [7,4,2] this includes dropping the entire
# 7-ch LIGAND group → encoder becomes robust to a ZERO ligand channel (downstream: ligand-masked
# generation / ligand-free inputs). Test whether HCS matches the no-HCS baseline (0.631±.010, 3-seed);
# if similar, HCS is the better default (same perf + missing-channel robustness). p=0.15 (the v1-best HCS).
# usage: hcs.sh <GPUS> <PORT> <SEED>
set -uf
cd /home/shpark/prj-denovo/VoxBind/voxbind || exit 1
PY=/home/shpark/.conda/envs/voxbind/bin
CFG=config_train_atomblob_density_gradmag_channelvit_mae_40m_plinder_otf_mask050
RES=dataset/data/pdbbind/results
GPUS="$1"; PORT="$2"; SEED="$3"
EXP=260715_ar_cvit_100m_v2_hcs015_s${SEED}

echo "===== HCS p=0.15 · 100M d640/L18/h10 @ mask0.75 · seed$SEED START $(date '+%m-%d %H:%M:%S') GPU=$GPUS ====="
CUDA_VISIBLE_DEVICES=$GPUS "$PY/torchrun" --nnodes=1 --nproc_per_node=4 \
    --rdzv-backend=c10d --rdzv-endpoint=localhost:"$PORT" --rdzv-id=cvitHCS015s$SEED \
    train_density.py --config-name="$CFG" exp_name="$EXP" bsz=4 accum_steps=8 \
    num_workers=6 prefetch_factor=2 num_epochs=50 seed=$SEED \
    model.channel_groups=[7,4,2] model.dim=640 model.depth=18 model.heads=10 \
    model.channel_group_dropout=0.15 mae.mask_ratio=0.75 \
    dset.data_file=pretrain/data_train_plinder_v2_perelem.pt \
    dset.resample_dir=dataset/data/pretrain/xray_resample_plinder_v2_perelem \
    dset.subset_n=112000 dset.subset_val_n=100 \
  && echo "[$EXP] PRETRAIN OK $(date '+%H:%M:%S')" \
  || { echo "[$EXP] PRETRAIN FAILED $(date '+%H:%M:%S')"; exit 1; }

echo "###### [$EXP] PROBE start $(date '+%H:%M:%S') ######"
bash scripts/04_probe.sh --exp "$EXP" --condition atomblob_density_gradmag \
    --tasks affinity --split lp_edrscc_v2 --epoch 49 --gpu "${GPUS%%,*}" --tag "$EXP" \
    --num_workers 0 -- --require_density \
  && echo "[$EXP] PROBE OK" || echo "[$EXP] PROBE FAILED"
echo "###### [$EXP] result ######"
tail -2 "$RES/probe_results_e49_v5_lp_edrscc_v2split_${EXP}.csv" 2>/dev/null
echo "===== HCS seed$SEED COMPLETE $(date '+%m-%d %H:%M:%S') ====="
