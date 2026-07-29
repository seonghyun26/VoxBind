#!/bin/bash
# 260629_ar_cvit_g742_ligand.sh — DRAMATIC change #3: LIGAND masking (pocket-conditioned ligand prediction).
# New opt-in mask_strategy=ligand (train_density.py; uniform default + atom_biased/cluster untouched). Masks the
# blocks most occupied by LIGAND atoms (reuses the atom-biased top-K, weighted by ligand mass only) so the encoder
# reconstructs the LIGAND region from the surrounding POCKET — directly mirrors VoxBind's de-novo ligand-placement
# task, and should learn binding-relevant features. mask_ratio 0.25 ≈ ligand footprint, tau 0.5 = strong ligand
# focus. On the [7,4,2] winner. GPU 0-3 (4-GPU eff-128). Probe lp_edrscc_v2.
set -uf
cd /home/shpark/prj-denovo/VoxBind/voxbind || exit 1
PY=/home/shpark/.conda/envs/voxbind/bin
CFG=config_train_atomblob_density_gradmag_channelvit_mae_40m_plinder_otf_mask050
RES=dataset/data/pdbbind/results
GPUS=0,1,2,3; PORT=29603; RID=arG742Lig
EXP=260629_ar_cvit_g742_ligand

echo "===== AR [7,4,2] LIGAND-MASK (pocket→ligand) START $(date '+%m-%d %H:%M:%S') GPU=$GPUS ====="
CUDA_VISIBLE_DEVICES=$GPUS "$PY/torchrun" --nnodes=1 --nproc_per_node=4 \
    --rdzv-backend=c10d --rdzv-endpoint=localhost:"$PORT" --rdzv-id="$RID" \
    train_density.py --config-name="$CFG" exp_name="$EXP" bsz=8 accum_steps=4 \
    model.channel_groups=[7,4,2] mae.mask_strategy=ligand mae.mask_ratio=0.25 mae.mask_atom_tau=0.5 \
  && echo "[$EXP] PRETRAIN OK $(date '+%H:%M:%S')" \
  || { echo "[$EXP] PRETRAIN FAILED $(date '+%H:%M:%S')"; exit 1; }
echo "###### [$EXP] PROBE start $(date '+%H:%M:%S') ######"
bash scripts/04_probe.sh --exp "$EXP" --condition atomblob_density_gradmag \
    --tasks affinity --split lp_edrscc_v2 --gpu "${GPUS%%,*}" --tag "$EXP" --num_workers 0 \
    -- --require_density \
  && echo "[$EXP] PROBE OK" || echo "[$EXP] PROBE FAILED"
echo "###### [$EXP] result ######"
tail -2 "$RES/probe_results_e99_v5_lp_edrscc_v2split_${EXP}.csv" 2>/dev/null
echo "===== AR [7,4,2] LIGAND-MASK COMPLETE $(date '+%m-%d %H:%M:%S') ====="
