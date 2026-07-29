#!/bin/bash
# 260705_ar_cvit_100m_v2_arch.sh — ROUND 2 of the brief autoresearch: ARCHITECTURE sweep at
# ~100M on PLINDER v2 (user idea: increase heads + adjust depth/width at fixed budget). Heads is
# FREE param-wise (qkv/proj are dim×dim regardless of head count), so we can test attention
# granularity + the depth/width split independently. Run at round-1's WINNING mask ratio.
# Reduced 50ep (probe e49); winner full-trained to 100ep after. num_workers6/prefetch2 = concurrent-OOM-safe.
# usage: arch.sh <GPUS csv> <PORT> <DIM> <DEPTH> <HEADS> <MASK>
set -uf
cd /home/shpark/prj-denovo/VoxBind/voxbind || exit 1
GPUS="$1"; PORT="$2"; DIM="$3"; DEPTH="$4"; HEADS="$5"; MASK="$6"; EPOCHS="${7:-50}"; SEED="${8:-42}"
MTAG=$(echo "$MASK" | tr -d '.')
PROBE_EP=$((EPOCHS-1))
PY=/home/shpark/.conda/envs/voxbind/bin
CFG=config_train_atomblob_density_gradmag_channelvit_mae_40m_plinder_otf_mask050
RES=dataset/data/pdbbind/results
# seed suffix only for non-default seeds so existing seed42 runs keep their names
SFX=""; [ "$SEED" != "42" ] && SFX="_s${SEED}"
RID=cvit100mv2d${DIM}L${DEPTH}h${HEADS}e${EPOCHS}s${SEED}
EXP=260705_ar_cvit_100m_v2_d${DIM}L${DEPTH}h${HEADS}_m${MTAG}_e${EPOCHS}${SFX}
NG=$(echo "$GPUS" | tr ',' '\n' | wc -l)

echo "===== ARCH d${DIM}/L${DEPTH}/h${HEADS} · v2 · mask=$MASK START $(date '+%m-%d %H:%M:%S') GPU=$GPUS ====="
CUDA_VISIBLE_DEVICES=$GPUS "$PY/torchrun" --nnodes=1 --nproc_per_node="$NG" \
    --rdzv-backend=c10d --rdzv-endpoint=localhost:"$PORT" --rdzv-id="$RID" \
    train_density.py --config-name="$CFG" exp_name="$EXP" bsz=4 accum_steps=8 \
    num_workers=6 prefetch_factor=2 num_epochs=$EPOCHS \
    model.channel_groups=[7,4,2] model.dim="$DIM" model.depth="$DEPTH" model.heads="$HEADS" \
    mae.mask_ratio=$MASK seed=$SEED \
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
echo "===== ARCH d${DIM}/L${DEPTH}/h${HEADS} · mask=$MASK COMPLETE $(date '+%m-%d %H:%M:%S') ====="
