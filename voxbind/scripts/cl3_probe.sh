#!/usr/bin/env bash
# Probe 4 DIVERSE encoders (all ≥0.62 on the full test) on the CL3-filtered test =
# lp_edrscc_v2_cl123 (CL1+CL2+CL3 leakage removal, 733 test — the strictest leak-proof
# split). HEAD-ONLY: reuses cached features (encoder frozen), retrains the MLP head on the
# CL123 train partition. 5 seeds, both recipes. CPU capped (01c intrinsic + nice).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
PY=/home/shpark/.conda/envs/voxbind/bin/python
SPLIT=lp_edrscc_v2_cl123
CDG=atomblob_density_gradmag
CD=atomblob_density

ENCS=(
  "260705_ar_cvit_100m_v2_mask075:$CDG"                 # champion (CDG [7,4,2] v2)
  "260813_cdg_100m_v22_varmask6090:$CDG"                # R2MAE variable mask, v2.2
  "260813_cdg_100m_v22_g742_pergroup_varmask7090:$CDG"  # per_group [7,4,2] + variable
  "260813_cd_100m_v22_g741_mask075:$CD"                 # no-gradmag CD [7,4,1] (12-ch)
)

i=0
for pair in "${ENCS[@]}"; do
  run="${pair%%:*}"; cond="${pair##*:}"; gpu=$((i % 4)); i=$((i + 1))
  ( for loss in mse "mse+corr"; do
      suf=mse; aux=(); [ "$loss" = "mse+corr" ] && { suf=msecorr5; aux=(--aux_weight 5); }
      CUDA_VISIBLE_DEVICES=$gpu nice -n 19 "$PY" dataset/01c_pdbbind_probe.py probe \
        --conditions "$cond" --epoch 49 --voxel_version v5 --split "$SPLIT" \
        --feature_tag "$run" --exp_dir "exps/$run" --allow_stale_features --seeds 5 \
        --probe_loss "$loss" "${aux[@]}" --device cuda:0 --no_wandb --tag "${run}_${suf}"
    done ) > "log/cl3probe_${run}.log" 2>&1 &
  echo "launched $run (gpu $gpu, cond $cond)"
done
wait
echo "==== ALL CL3 PROBE DONE ===="
