#!/bin/bash
# 260725 — disentangle SIZE vs MASK vs CORPUS behind "60M beat 100M?". The best-60M (v3/mask0.85
# 0.641) vs champion (100M/v2/mask0.75 0.644) differ in 3 axes. Fill the missing grid cells:
#   T1 = 60M · v2 · mask0.75  → direct SIZE control vs the champion (same corpus+mask). 60M>100M here = real.
#   T2 = 100M · v3 · mask0.85 → matched vs best-60M (same corpus+mask); does mask0.85 help 100M too?
# C+D+G ChannelViT [7,4,2] + HCS0.15, eff-batch128, 50ep, ema0.999, compile OFF. Parameterized.
# usage: 260729_ar_cvit_recipe_grid.sh <GPUS> <PORT> <DIM> <DEPTH> <HEADS> <DATA v2|v3> <MASK> <NAME>
set -uf
cd /home/shpark/prj-denovo/VoxBind/voxbind || exit 1
GPUS="$1"; PORT="$2"; DIM="$3"; DEPTH="$4"; HEADS="$5"; DATA="$6"; MASK="$7"; NAME="$8"; HCS="${9:-0.15}"
PY=/home/shpark/.conda/envs/voxbind/bin
CFG=config_train_atomblob_density_gradmag_channelvit_mae_40m_plinder_otf_mask050
RES=dataset/data/pdbbind/results
NG=$(echo "$GPUS" | tr ',' '\n' | wc -l)
if [ "$DATA" = "v2" ]; then DF=pretrain/data_train_plinder_v2_perelem.pt; RS=dataset/data/pretrain/xray_resample_plinder_v2_perelem; SUB=112000
else                        DF=pretrain/data_train_plinder_v3_perelem.pt; RS=dataset/data/pretrain/xray_resample_plinder_v3_perelem; SUB=70725; fi
EXP=260729_ar_cvit_${NAME}

echo "===== [$EXP] ${DIM}d/${DEPTH}L · $DATA · mask$MASK START $(date '+%m-%d %H:%M:%S') GPU=$GPUS ====="
CUDA_VISIBLE_DEVICES=$GPUS "$PY/torchrun" --nnodes=1 --nproc_per_node="$NG" \
    --rdzv-backend=c10d --rdzv-endpoint=localhost:"$PORT" --rdzv-id="cvit$NAME" \
    train_density.py --config-name="$CFG" exp_name="$EXP" bsz=4 accum_steps=8 \
    num_workers=6 prefetch_factor=2 num_epochs=50 \
    model.channel_groups=[7,4,2] model.dim=$DIM model.depth=$DEPTH model.heads=$HEADS \
    model.channel_group_dropout=$HCS mae.mask_ratio=$MASK mae.ema_decay=0.999 compile.enabled=false \
    dset.data_file=$DF dset.resample_dir=$RS dset.subset_n=$SUB dset.subset_val_n=100 \
  && echo "[$EXP] PRETRAIN OK $(date '+%H:%M:%S')" \
  || { echo "[$EXP] PRETRAIN FAILED $(date '+%H:%M:%S')"; exit 1; }

echo "###### [$EXP] PROBE start $(date '+%H:%M:%S') ######"
bash scripts/04_probe.sh --exp "$EXP" --condition atomblob_density_gradmag \
    --tasks affinity --split lp_edrscc_v2 --epoch 49 --gpu "${GPUS%%,*}" --tag "$EXP" \
    --num_workers 0 -- --require_density \
  && echo "[$EXP] PROBE OK" || echo "[$EXP] PROBE FAILED"
echo "###### [$EXP] result (r / rho / RMSE) ######"
tail -2 "$RES/probe_results_e49_v5_lp_edrscc_v2split_${EXP}.csv" 2>/dev/null
echo "===== [$EXP] COMPLETE $(date '+%m-%d %H:%M:%S') ====="
