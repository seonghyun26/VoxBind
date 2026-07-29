#!/bin/bash
# 260718 — EMA-DECAY ablation on the clean v2.1 corpus. IDENTICAL to the v2.1 baseline
# (260718_ar_cvit_40m_v2p1_clean: 40M ChannelViT [7,4,2], mask0.5, 100ep) EXCEPT
# mae.ema_decay 0.999 -> 0.9999. RATIONALE: the frozen probe reads encoder_state_dict_ema,
# so the EMA weights ARE the evaluated features. decay 0.999 averages ~1000 opt-steps; on
# v2.1 (292 steps/ep) that is only ~3.4% of the 29.2K-step run, vs ~7 epochs' worth on v1 —
# i.e. the EMA window shrinks as a FRACTION of training as data grows. 0.9999 → ~10K-step
# window (~1/3 of training), the standard long-run EMA (MAE/DINO), restoring the averaging
# fraction. Tests whether an EMA that de-tunes with scale is part of the plateau.
# usage: 260718_ar_cvit_40m_v2p1_ema9999.sh <GPUS csv> <PORT> [SEED=42]
set -uf
cd /home/shpark/prj-denovo/VoxBind/voxbind || exit 1
GPUS="$1"; PORT="$2"; SEED="${3:-42}"
PY=/home/shpark/.conda/envs/voxbind/bin
CFG=config_train_atomblob_density_gradmag_channelvit_mae_40m_plinder_otf_mask050
RES=dataset/data/pdbbind/results
EPOCHS=100; PROBE_EP=99; SUBSET_N=37401
NG=$(echo "$GPUS" | tr ',' '\n' | wc -l)
EXP=260718_ar_cvit_40m_v2p1_ema9999
[ "$SEED" != "42" ] && EXP="${EXP}_s${SEED}"

echo "===== 40M v2.1 · mask0.5 · ema0.9999 · 100ep START $(date '+%m-%d %H:%M:%S') GPU=$GPUS ====="
CUDA_VISIBLE_DEVICES=$GPUS "$PY/torchrun" --nnodes=1 --nproc_per_node="$NG" \
    --rdzv-backend=c10d --rdzv-endpoint=localhost:"$PORT" --rdzv-id=cvit40mV2p1ema \
    train_density.py --config-name="$CFG" exp_name="$EXP" bsz=8 accum_steps=4 \
    num_workers=6 prefetch_factor=2 num_epochs=$EPOCHS seed=$SEED \
    model.channel_groups=[7,4,2] mae.ema_decay=0.9999 \
    dset.data_file=pretrain/data_train_plinder_v2p1_perelem.pt \
    dset.resample_dir=dataset/data/pretrain/xray_resample_plinder_v2p1_perelem \
    dset.subset_n=$SUBSET_N dset.subset_val_n=100 \
  && echo "[$EXP] PRETRAIN OK $(date '+%H:%M:%S')" \
  || { echo "[$EXP] PRETRAIN FAILED $(date '+%H:%M:%S')"; exit 1; }

echo "###### [$EXP] PROBE start $(date '+%H:%M:%S') ######"
bash scripts/04_probe.sh --exp "$EXP" --condition atomblob_density_gradmag \
    --tasks affinity --split lp_edrscc_v2 --epoch $PROBE_EP --gpu "${GPUS%%,*}" \
    --tag "$EXP" --num_workers 0 -- --require_density \
  && echo "[$EXP] PROBE OK" || echo "[$EXP] PROBE FAILED"
echo "###### [$EXP] result ######"
tail -2 "$RES/probe_results_e${PROBE_EP}_v5_lp_edrscc_v2split_${EXP}.csv" 2>/dev/null
echo "===== 40M v2.1 ema0.9999 COMPLETE $(date '+%m-%d %H:%M:%S') ====="
