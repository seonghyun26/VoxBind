#!/bin/bash
# 260629_ar_roleblob_v2_datascale.sh — DATA-SCALE test (predicts the [8,4,2]-on-v2 payoff).
# The whole campaign concluded the frozen-probe plateau (ρ≈0.63) is data/representation-bound, not
# pretext-bound (masking algos all flat/hurt; finetune hurt; capacity hurt). The one untested lever
# is DATA QUANTITY. The full v2 corpus (data_train_plinder_v2.pt = 113,874 entries / 112,733 positions,
# 6.5× the original 17.5K) is staged — but in ROLE-collapsed format (atoms_channel=0), so it runs with
# the roleblob rep, not element atomblob. Run the best role config [1,1,2] (=0.592 on 17K, 100ep) on the
# full v2 for 50 ep (~2× the original total samples). If 6.5× data lifts roleblob materially, the user's
# [8,4,2]-on-v2 (element rep, once fully built) is promising; if flat, data quantity isn't the bottleneck.
# GPU 0-3. resample reads the shared data/ccp4 (here). Probe e49.
set -uf
cd /home/shpark/prj-denovo/VoxBind/voxbind || exit 1
PY=/home/shpark/.conda/envs/voxbind/bin
CFG=config_train_roleblob_diverse_density_gradmag_channelvit_mae_40m_plinder_otf_mask050
RES=dataset/data/pdbbind/results
GPUS=0,1,2,3; PORT=29605; RID=arV2DS
EXP=260629_ar_roleblob_v2_g112_50ep

echo "===== AR roleblob [1,1,2] on FULL v2 (113K, 50ep) START $(date '+%m-%d %H:%M:%S') GPU=$GPUS ====="
CUDA_VISIBLE_DEVICES=$GPUS "$PY/torchrun" --nnodes=1 --nproc_per_node=4 \
    --rdzv-backend=c10d --rdzv-endpoint=localhost:"$PORT" --rdzv-id="$RID" \
    train_density.py --config-name="$CFG" exp_name="$EXP" bsz=8 accum_steps=4 num_epochs=50 \
    model.channel_groups=[1,1,2] \
    dset.data_file=pretrain/data_train_plinder_v2.pt \
    'dset.resample_dir=${oc.env:VOXBIND_DATA_ROOT,dataset/data}/pretrain/xray_resample_plinder_v2' \
    dset.subset_n=112633 \
  && echo "[$EXP] PRETRAIN OK $(date '+%H:%M:%S')" \
  || { echo "[$EXP] PRETRAIN FAILED $(date '+%H:%M:%S')"; exit 1; }
echo "###### [$EXP] PROBE start (e49) $(date '+%H:%M:%S') ######"
bash scripts/04_probe.sh --exp "$EXP" --condition roleblob_density_gradmag_channelvit \
    --tasks affinity --split lp_edrscc_v2 --epoch 49 --gpu "${GPUS%%,*}" --tag "$EXP" --num_workers 0 \
  && echo "[$EXP] PROBE OK" || echo "[$EXP] PROBE FAILED"
echo "###### [$EXP] result ######"
tail -2 "$RES/probe_results_e49_v5_lp_edrscc_v2split_${EXP}.csv" 2>/dev/null
echo "===== AR roleblob v2 datascale COMPLETE $(date '+%m-%d %H:%M:%S') ====="
