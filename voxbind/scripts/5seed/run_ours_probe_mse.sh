#!/usr/bin/env bash
# Re-run C / C+D+G at 5 seeds with EXPLICIT --out_csv = the exact generator-read filename.
set -u
cd /home/shpark/prj-denovo/VoxBind/voxbind || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
PY=/home/shpark/.conda/envs/voxbind/bin/python
RES=dataset/data/pdbbind/results
CDG=260705_ar_cvit_100m_v2_mask075
CCO=260723_ar_cvit_100m_v2_mask075_coords
for SPLIT in lp_edrscc_v2 lp_edrscc_v2_cl1 lp_edrscc_v2_cl12 lp_edrscc_v2_cl123; do
  echo "===== $SPLIT C+D+G mse ====="
  nice -19 $PY dataset/01c_pdbbind_probe.py probe --conditions atomblob_density_gradmag \
    --epoch 49 --voxel_version v5 --split "$SPLIT" --feature_tag "$CDG" --exp_dir "exps/$CDG" \
    --allow_stale_features --seeds 5 --probe_loss mse --device cpu --no_wandb \
    --out_csv "$RES/probe_results_e49_v5_${SPLIT}split_${CDG}.csv" || echo "FAIL CDG $SPLIT"
  echo "===== $SPLIT C mse ====="
  nice -19 $PY dataset/01c_pdbbind_probe.py probe --conditions atomblob \
    --epoch 49 --voxel_version v5 --split "$SPLIT" --feature_tag "$CCO" --exp_dir "exps/$CCO" \
    --allow_stale_features --seeds 5 --probe_loss mse --device cpu --no_wandb \
    --out_csv "$RES/probe_results_e49_v5_${SPLIT}split_${CCO}.csv" || echo "FAIL C $SPLIT"
done
echo "OURS MSE 5-SEED DONE"
