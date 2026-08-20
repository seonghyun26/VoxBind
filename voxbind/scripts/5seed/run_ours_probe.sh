#!/usr/bin/env bash
# Re-probe OUR C / C+D+G / C+D+G+corr at 5 seeds on all 4 CL tiers (frozen features -> MLP head).
# CPU + capped threads so it does not contend with the GPU baseline campaign. Cached features reused.
set -u
cd /home/shpark/prj-denovo/VoxBind/voxbind || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
PY=/home/shpark/.conda/envs/voxbind/bin/python
RES=dataset/data/pdbbind/results
CDG=260705_ar_cvit_100m_v2_mask075
CCO=260723_ar_cvit_100m_v2_mask075_coords
for SPLIT in lp_edrscc_v2 lp_edrscc_v2_cl1 lp_edrscc_v2_cl12 lp_edrscc_v2_cl123; do
  echo "===== $SPLIT : C+D+G (mse) ====="
  nice -19 $PY dataset/01c_pdbbind_probe.py probe --conditions atomblob_density_gradmag \
    --epoch 49 --voxel_version v5 --split "$SPLIT" --feature_tag "$CDG" --exp_dir "exps/$CDG" \
    --allow_stale_features --seeds 5 --probe_loss mse --device cpu --no_wandb || echo "FAIL CDG $SPLIT"
  echo "===== $SPLIT : C (coords, mse) ====="
  nice -19 $PY dataset/01c_pdbbind_probe.py probe --conditions atomblob \
    --epoch 49 --voxel_version v5 --split "$SPLIT" --feature_tag "$CCO" --exp_dir "exps/$CCO" \
    --allow_stale_features --seeds 5 --probe_loss mse --device cpu --no_wandb || echo "FAIL C $SPLIT"
  echo "===== $SPLIT : C+D+G +corr ====="
  nice -19 $PY dataset/01c_pdbbind_probe.py probe --conditions atomblob_density_gradmag \
    --epoch 49 --voxel_version v5 --split "$SPLIT" --feature_tag "$CDG" --exp_dir "exps/$CDG" \
    --allow_stale_features --seeds 5 --probe_loss mse+corr --aux_weight 5 --device cpu --no_wandb \
    --out_csv "$RES/probe_results_e49_v5_${SPLIT}split_loss-mse-corr-w5.csv" || echo "FAIL corr $SPLIT"
done
echo "ALL OURS 5-SEED PROBE DONE"
