#!/bin/bash
# 260703_ar_cvit_g742_v2_depth24.sh — DEPTH-vs-WIDTH capacity test on v2.
# The 98M model was pure WIDTH-scaling (dim512→768, depth stayed 12) and REGRESSED on v2
# (ρ 0.595). Hypothesis (user): stacking LAYERS may scale better than widening. So match the
# large-capacity budget via DEPTH instead: dim512/depth24/heads8 (~85M, "double the layers")
# on the same v2-112K, [7,4,2] 7ch, mask0.50, eff-batch128, 100ep. Compare vs width-98M 0.595
# and base-47M. bsz4/accum8 (depth24 activation ~14GB/24GB at bsz4, safe — width-98M used ~11GB).
# GPU 4-7, then frozen probe lp_edrscc_v2.
set -uf
cd /home/shpark/prj-denovo/VoxBind/voxbind || exit 1
PY=/home/shpark/.conda/envs/voxbind/bin
CFG=config_train_atomblob_density_gradmag_channelvit_mae_40m_plinder_otf_mask050
RES=dataset/data/pdbbind/results
GPUS=4,5,6,7
PORT=29572
RID=cvitG742v2d24
EXP=260703_ar_cvit_g742_v2_depth24

echo "===== depth24 (~85M) · PLINDER v2 START $(date '+%m-%d %H:%M:%S') GPU=$GPUS ====="
CUDA_VISIBLE_DEVICES=$GPUS "$PY/torchrun" --nnodes=1 --nproc_per_node=4 \
    --rdzv-backend=c10d --rdzv-endpoint=localhost:"$PORT" --rdzv-id="$RID" \
    train_density.py --config-name="$CFG" exp_name="$EXP" bsz=4 accum_steps=8 \
    model.channel_groups=[7,4,2] model.depth=24 \
    dset.data_file=pretrain/data_train_plinder_v2_perelem.pt \
    dset.resample_dir=dataset/data/pretrain/xray_resample_plinder_v2_perelem \
    dset.subset_n=112000 dset.subset_val_n=100 \
  && echo "[$EXP] PRETRAIN OK $(date '+%H:%M:%S')" \
  || { echo "[$EXP] PRETRAIN FAILED $(date '+%H:%M:%S')"; exit 1; }

echo "###### [$EXP] PROBE start $(date '+%H:%M:%S') ######"
bash scripts/04_probe.sh --exp "$EXP" --condition atomblob_density_gradmag \
    --tasks affinity --split lp_edrscc_v2 --gpu "${GPUS%%,*}" --tag "$EXP" --num_workers 0 \
    -- --require_density \
  && echo "[$EXP] PROBE OK" || echo "[$EXP] PROBE FAILED"
echo "###### [$EXP] result ######"
tail -2 "$RES/probe_results_e99_v5_lp_edrscc_v2split_${EXP}.csv" 2>/dev/null
echo "===== depth24 (~85M) · PLINDER v2 COMPLETE $(date '+%m-%d %H:%M:%S') ====="
