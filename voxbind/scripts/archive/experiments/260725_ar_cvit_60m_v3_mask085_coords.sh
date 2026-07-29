#!/bin/bash
# 260725 — MATCHED COORDS-ONLY control for the best v3 recipe. Pairs with the C+D+G
# 60M·v3·mask0.85 run (ρ 0.641 / r 0.661). IDENTICAL model (dim512/depth18/heads8 ~60M),
# corpus (v3, 70,725), mask (0.85), 50ep, eff-batch128 — EXCEPT coords-only input:
# input_mode=atomblob, with_gradmag=false, n_in=11 (7 lig + 4 poc, NO density/gradmag).
# Gives the density gain (C+D+G − C) on the clean v3 corpus + small-model sweet spot.
# NB coords = plain ViT atomblob (no channel_groups / HCS — those are density-channel constructs),
# matching how the champion C-vs-C+D+G pair was measured (C+D+G ChannelViT vs C plain atomblob).
# usage: 260725_ar_cvit_60m_v3_mask085_coords.sh <GPUS csv> <PORT>
set -uf
cd /home/shpark/prj-denovo/VoxBind/voxbind || exit 1
GPUS="$1"; PORT="$2"
PY=/home/shpark/.conda/envs/voxbind/bin
CFG=config_train_atomblob_vit_mae_40m_invfreq_plinder_otf_mask050_coords
RES=dataset/data/pdbbind/results
EPOCHS=50; PROBE_EP=49; SUBSET=70725
NG=$(echo "$GPUS" | tr ',' '\n' | wc -l)
EXP=260725_ar_cvit_60m_v3_mask085_coords

echo "===== MATCHED COORDS (C) 60M·v3·mask0.85 START $(date '+%m-%d %H:%M:%S') GPU=$GPUS ====="
CUDA_VISIBLE_DEVICES=$GPUS "$PY/torchrun" --nnodes=1 --nproc_per_node="$NG" \
    --rdzv-backend=c10d --rdzv-endpoint=localhost:"$PORT" --rdzv-id=cvit60mV3coords \
    train_density.py --config-name="$CFG" exp_name="$EXP" bsz=4 accum_steps=8 \
    num_workers=6 prefetch_factor=2 num_epochs=$EPOCHS \
    model.dim=512 model.depth=18 model.heads=8 \
    mae.mask_ratio=0.85 compile.enabled=false \
    dset.data_file=pretrain/data_train_plinder_v3_perelem.pt \
    dset.resample_dir=dataset/data/pretrain/xray_resample_plinder_v3_perelem \
    dset.subset_n=$SUBSET dset.subset_val_n=100 \
  && echo "[$EXP] PRETRAIN OK $(date '+%H:%M:%S')" \
  || { echo "[$EXP] PRETRAIN FAILED $(date '+%H:%M:%S')"; exit 1; }

echo "###### [$EXP] PROBE start $(date '+%H:%M:%S') ######"
bash scripts/04_probe.sh --exp "$EXP" --condition atomblob \
    --tasks affinity --split lp_edrscc_v2 --epoch $PROBE_EP --gpu "${GPUS%%,*}" \
    --tag "$EXP" --num_workers 0 \
  && echo "[$EXP] PROBE OK" || echo "[$EXP] PROBE FAILED"
echo "###### [$EXP] result (coords-only C; r / rho / RMSE) ######"
tail -2 "$RES/probe_results_e${PROBE_EP}_v5_lp_edrscc_v2split_${EXP}.csv" 2>/dev/null
echo "===== MATCHED COORDS COMPLETE $(date '+%m-%d %H:%M:%S') ====="
