#!/bin/bash
# 260716 — the 0.656 recipe (40M ChannelViT [7,4,2] C+D+G, mask 0.5, lr 1e-4) on PLINDER
# v2 (112K, 6.4× v1) + HCS channel-group dropout p=0.15. Question: does the v1-best 0.656
# recipe hold / improve on the larger v2 corpus WITH the free missing-channel robustness of
# HCS p=0.15 (drops whole groups incl the 7-ch ligand group; downstream ligand-masked gen).
#   40M = config defaults dim512/L12/h8 (NO dim override — that's what makes it the 40M/0.656 arm).
#   mask 0.5 (config default) = the 0.656 recipe (NOT the 100M-v2 mask-0.75 arm).
#   50 epochs on v2 ≈ 320 v1-epoch-equiv of steps (v2 is 6.4× v1) — comparable to the v2 campaign.
#   compile OFF: HCS drops groups → variable token count → thrashes torch.compile (result-neutral).
#   bsz4×accum8×4gpu = eff-batch 128 (campaign-standard); nw6/pf2 to stay host-RAM-safe on v2.
# usage: 260716_ar_cvit_40m_v2_hcs015.sh <GPUS> <PORT>
set -uf
cd /home/shpark/prj-denovo/VoxBind/voxbind || exit 1
PY=/home/shpark/.conda/envs/voxbind/bin
CFG=config_train_atomblob_density_gradmag_channelvit_mae_40m_plinder_otf_mask050
RES=dataset/data/pdbbind/results
GPUS="$1"; PORT="$2"
EXP=260716_ar_cvit_40m_v2_hcs015

echo "===== HCS p=0.15 · 40M d512/L12/h8 @ mask0.50 · PLINDER v2 START $(date '+%m-%d %H:%M:%S') GPU=$GPUS ====="
CUDA_VISIBLE_DEVICES=$GPUS "$PY/torchrun" --nnodes=1 --nproc_per_node=4 \
    --rdzv-backend=c10d --rdzv-endpoint=localhost:"$PORT" --rdzv-id=cvit40mV2HCS \
    train_density.py --config-name="$CFG" exp_name="$EXP" bsz=4 accum_steps=8 \
    num_workers=6 prefetch_factor=2 num_epochs=50 \
    model.channel_groups=[7,4,2] model.channel_group_dropout=0.15 \
    compile.enabled=false \
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
echo "===== 40M v2 HCS0.15 COMPLETE $(date '+%m-%d %H:%M:%S') ====="
