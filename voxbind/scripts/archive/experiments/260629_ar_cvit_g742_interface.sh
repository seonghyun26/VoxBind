#!/bin/bash
# 260629_ar_cvit_g742_interface.sh — DRAMATIC change #4: INTERFACE masking (mask the binding contact).
# New opt-in mask_strategy=interface (make_interface_mask in mae_ops.py; uniform default, others untouched).
# Scores each block by ligand×pocket co-occupancy (dilated, so adjacency counts) and masks the top blocks —
# i.e., the ligand-pocket CONTACT region, where binding affinity originates — then reconstructs it from the
# surroundings, forcing the encoder to learn interaction structure. Complementary to trial 3 (ligand masking).
# ratio 0.3 (contact + nearby), tau 0 (deterministic interface). [7,4,2] base. GPU 4-7 (parallel to trial 3 on 0-3).
set -uf
cd /home/shpark/prj-denovo/VoxBind/voxbind || exit 1
PY=/home/shpark/.conda/envs/voxbind/bin
CFG=config_train_atomblob_density_gradmag_channelvit_mae_40m_plinder_otf_mask050
RES=dataset/data/pdbbind/results
GPUS=4,5,6,7; PORT=29604; RID=arG742Iface
EXP=260629_ar_cvit_g742_interface

echo "===== AR [7,4,2] INTERFACE-MASK START $(date '+%m-%d %H:%M:%S') GPU=$GPUS ====="
CUDA_VISIBLE_DEVICES=$GPUS "$PY/torchrun" --nnodes=1 --nproc_per_node=4 \
    --rdzv-backend=c10d --rdzv-endpoint=localhost:"$PORT" --rdzv-id="$RID" \
    train_density.py --config-name="$CFG" exp_name="$EXP" bsz=8 accum_steps=4 \
    model.channel_groups=[7,4,2] mae.mask_strategy=interface mae.mask_ratio=0.3 mae.mask_atom_tau=0.0 \
  && echo "[$EXP] PRETRAIN OK $(date '+%H:%M:%S')" \
  || { echo "[$EXP] PRETRAIN FAILED $(date '+%H:%M:%S')"; exit 1; }
echo "###### [$EXP] PROBE start $(date '+%H:%M:%S') ######"
bash scripts/04_probe.sh --exp "$EXP" --condition atomblob_density_gradmag \
    --tasks affinity --split lp_edrscc_v2 --gpu "${GPUS%%,*}" --tag "$EXP" --num_workers 0 \
    -- --require_density \
  && echo "[$EXP] PROBE OK" || echo "[$EXP] PROBE FAILED"
echo "###### [$EXP] result ######"
tail -2 "$RES/probe_results_e99_v5_lp_edrscc_v2split_${EXP}.csv" 2>/dev/null
echo "===== AR [7,4,2] INTERFACE-MASK COMPLETE $(date '+%m-%d %H:%M:%S') ====="
