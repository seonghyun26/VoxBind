#!/bin/bash
# 260728 — NEW pretraining configs on v3 (pose-diverse clean, 70,725), 100M ChannelViT [7,4,2].
# The v3 mask sweep (m0.75=0.632 .. m0.90=0.645 .. m0.95=0.644) was all 50ep / ema0.999 / const-LR.
# This launcher exposes the UNTESTED axes so we can probe past that plateau:
#   EPOCHS   — is v3 undertrained at 50 (clean+high-mask reconstruction is harder)?
#   EMA      — window scales non-invariantly with corpus size (v3 smaller/cleaner than v2)
#   SCHEDULE — warmup+cosine LR (opt-in in train_density.py, never run on v3)
# eff-batch = 4*8*NG (128 on 4 GPUs) to match the champion. HCS0.15, compile OFF, [7,4,2] fixed.
# usage: 260728_ar_cvit_v3_newcfg.sh <GPUS> <PORT> <MASK> <EPOCHS> <EMA> <SCHEDULE const|cosine> <WARMUP> <NAME>
set -uf
cd /home/shpark/prj-denovo/VoxBind/voxbind || exit 1
GPUS="$1"; PORT="$2"; MASK="$3"; EPOCHS="$4"; EMA="$5"; SCHED="$6"; WARMUP="$7"; NAME="$8"
PY=/home/shpark/.conda/envs/voxbind/bin
CFG=config_train_atomblob_density_gradmag_channelvit_mae_40m_plinder_otf_mask050
RES=dataset/data/pdbbind/results
NG=$(echo "$GPUS" | tr ',' '\n' | wc -l)
DF=pretrain/data_train_plinder_v3_perelem.pt
RS=dataset/data/pretrain/xray_resample_plinder_v3_perelem
SUB=70725
DIM=640; DEPTH=18; HEADS=10
EXP=260728_ar_cvit_${NAME}

echo "===== [$EXP] 100M v3 · mask$MASK · ${EPOCHS}ep · ema$EMA · sched=$SCHED(warmup$WARMUP) START $(date '+%m-%d %H:%M:%S') GPU=$GPUS ====="
CUDA_VISIBLE_DEVICES=$GPUS "$PY/torchrun" --nnodes=1 --nproc_per_node="$NG" \
    --rdzv-backend=c10d --rdzv-endpoint=localhost:"$PORT" --rdzv-id="v3$NAME" \
    train_density.py --config-name="$CFG" exp_name="$EXP" bsz=4 accum_steps=8 \
    num_workers=6 prefetch_factor=2 num_epochs="$EPOCHS" \
    model.channel_groups=[7,4,2] model.dim=$DIM model.depth=$DEPTH model.heads=$HEADS \
    model.channel_group_dropout=0.15 mae.mask_ratio="$MASK" mae.ema_decay="$EMA" compile.enabled=false \
    +lr_schedule="$SCHED" +lr_warmup_epochs="$WARMUP" +lr_min=1e-6 \
    dset.data_file=$DF dset.resample_dir=$RS dset.subset_n=$SUB dset.subset_val_n=100 \
  && echo "[$EXP] PRETRAIN OK $(date '+%H:%M:%S')" \
  || { echo "[$EXP] PRETRAIN FAILED $(date '+%H:%M:%S')"; exit 1; }

PEP=$((EPOCHS-1))
echo "###### [$EXP] PROBE start (e$PEP) $(date '+%H:%M:%S') ######"
bash scripts/04_probe.sh --exp "$EXP" --condition atomblob_density_gradmag \
    --tasks affinity --split lp_edrscc_v2 --epoch "$PEP" --gpu "${GPUS%%,*}" --tag "$EXP" \
    --num_workers 0 -- --require_density \
  && echo "[$EXP] PROBE OK" || echo "[$EXP] PROBE FAILED"
echo "###### [$EXP] result (r / rho / RMSE) ######"
tail -2 "$RES/probe_results_e${PEP}_v5_lp_edrscc_v2split_${EXP}.csv" 2>/dev/null
echo "===== [$EXP] COMPLETE $(date '+%m-%d %H:%M:%S') ====="
