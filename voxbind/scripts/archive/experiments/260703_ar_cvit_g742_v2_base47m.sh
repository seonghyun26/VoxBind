#!/bin/bash
# 260703_ar_cvit_g742_v2_base47m.sh — DATA-SCALE CONTROL for the base model.
# The campaign-base [7,4,2] dim512 (~47M) got ρ 0.637 on v1-17K. dim768 (98M) on v2-112K
# REGRESSED to 0.595. Missing control = the SAME 47M base on v2-112K → isolates "does 6.5×
# data help the base model?" (the H200 0.624 is [8,4,2]/96³ — confounded). IDENTICAL recipe
# to the v1 base (dim512/depth12/heads8, [7,4,2], mask0.50, dens-wt0.1, eff-batch128, 100ep);
# only the dataset swaps v1-17K → v2 per-element 112K. GPU 0-3, then frozen probe lp_edrscc_v2.
set -uf
cd /home/shpark/prj-denovo/VoxBind/voxbind || exit 1
PY=/home/shpark/.conda/envs/voxbind/bin
CFG=config_train_atomblob_density_gradmag_channelvit_mae_40m_plinder_otf_mask050
RES=dataset/data/pdbbind/results
# 260704 RELAUNCH after host-RAM OOM: first attempt (bsz8/nw16/pf8 on GPU0-3, concurrent with
# the depth24 run on 4-7) hit the Linux OOM-killer at ep56 — 8 ranks × huge prefetch buffers
# (nw16×pf8) exhausted host RAM. Fixes: (1) bsz4/accum10 eff120 (≈128, within seed noise);
# (2) num_workers 16→6 + prefetch_factor 8→2 → ~10× less prefetch RAM so it COEXISTS with the
# depth24 run; (3) GPU 1,2,3 only (GPU0 has a 14GB leaked context from the SIGKILL'd rank, no
# root to reset). eff-batch 120 on 3 GPUs.
GPUS=1,2,3
PORT=29573
RID=cvitG742v2base
EXP=260703_ar_cvit_g742_v2_base47m

echo "===== 47M base · PLINDER v2 START $(date '+%m-%d %H:%M:%S') GPU=$GPUS ====="
CUDA_VISIBLE_DEVICES=$GPUS "$PY/torchrun" --nnodes=1 --nproc_per_node=3 \
    --rdzv-backend=c10d --rdzv-endpoint=localhost:"$PORT" --rdzv-id="$RID" \
    train_density.py --config-name="$CFG" exp_name="$EXP" bsz=4 accum_steps=10 \
    num_workers=6 prefetch_factor=2 \
    model.channel_groups=[7,4,2] \
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
echo "===== 47M base · PLINDER v2 COMPLETE $(date '+%m-%d %H:%M:%S') ====="
