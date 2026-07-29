#!/bin/bash
# 260718 — SAME RECIPE on the CLEAN v2.1 corpus. 40M ChannelViT [7,4,2] C+D+G, mask 0.5,
# 100ep — IDENTICAL to the 40M runs that scored v1-17K 0.637 and v2-112K 0.620 (no HCS,
# constant LR, eff-batch 128). Only the corpus changes: PLINDER v2.1 = dedup + res<=2.5 +
# ligand&pocket RSCC>=0.8 = 37,885 crops (~2.2× v1, CLEAN). Completes the controlled triple:
#   v1 clean-17K 0.637 · v2.1 clean-38K ??? · v2 noisy-112K 0.620
# → isolates DATA QUALITY: does clean ~2× scaling lift the plateau where noisy 6.5× did not?
# SUBSET_N is passed as $3 (exact train-position count from the build's manifest).
# usage: 260718_ar_cvit_40m_v2p1.sh <GPUS csv> <PORT> <SUBSET_N>
set -uf
cd /home/shpark/prj-denovo/VoxBind/voxbind || exit 1
GPUS="$1"; PORT="$2"; SUBSET_N="$3"; SEED="${4:-42}"
PY=/home/shpark/.conda/envs/voxbind/bin
CFG=config_train_atomblob_density_gradmag_channelvit_mae_40m_plinder_otf_mask050
RES=dataset/data/pdbbind/results
EPOCHS=100; PROBE_EP=99
NG=$(echo "$GPUS" | tr ',' '\n' | wc -l)
EXP=260718_ar_cvit_40m_v2p1_clean
[ "$SEED" != "42" ] && EXP="${EXP}_s${SEED}"

echo "===== 40M v2.1-CLEAN · mask0.5 · 100ep · subset=$SUBSET_N START $(date '+%m-%d %H:%M:%S') GPU=$GPUS ====="
CUDA_VISIBLE_DEVICES=$GPUS "$PY/torchrun" --nnodes=1 --nproc_per_node="$NG" \
    --rdzv-backend=c10d --rdzv-endpoint=localhost:"$PORT" --rdzv-id=cvit40mV2p1 \
    train_density.py --config-name="$CFG" exp_name="$EXP" bsz=8 accum_steps=4 \
    num_workers=6 prefetch_factor=2 num_epochs=$EPOCHS \
    model.channel_groups=[7,4,2] seed=$SEED \
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
echo "===== 40M v2.1-clean COMPLETE $(date '+%m-%d %H:%M:%S') ====="
