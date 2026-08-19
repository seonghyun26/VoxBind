#!/usr/bin/env bash
# Re-probe all recent campaign encoders with 5 seeds (0..4) to match the baselines
# (which used 5 seeds). HEAD-ONLY: features are already cached, so this only retrains the
# MLP head (fast). Overwrites the 3-seed result CSVs (same tags) → they become 5-seed.
# CPU is capped (01c sets thread cap intrinsically; + nice here). Round-robins GPU 0-3.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
PY=/home/shpark/.conda/envs/voxbind/bin/python
CDG=atomblob_density_gradmag
CD=atomblob_density

# <run>:<condition>  — 11 recent encoders (champion + the 260813 campaign)
ENCS=(
  "260705_ar_cvit_100m_v2_mask075:$CDG"                 # champion (v2)
  "260813_cdg_100m_v22_mask075:$CDG"                    # v2.2 champion
  "260813_cdg_100m_v2_varmask6090:$CDG"                 # R2MAE v2
  "260813_cdg_100m_v22_varmask6090:$CDG"                # R2MAE v2.2
  "260813_cdg_100m_v2_g7411:$CDG"                       # [7,4,1,1] v2
  "260813_cdg_100m_v22_g7411:$CDG"                      # [7,4,1,1] v2.2
  "260813_cdg_100m_v22_g7411_varmask7090:$CDG"          # [7,4,1,1] + variable
  "260813_cdg_100m_v22_g7411_pergroup:$CDG"             # per_group [7,4,1,1]
  "260813_cdg_100m_v22_g742_pergroup_varmask7090:$CDG"  # per_group [7,4,2] + variable
  "260813_cdg_100m_v24_mask075:$CDG"                    # champion on v2.4
  "260813_cd_100m_v22_g741_mask075:$CD"                 # no-gradmag CD [7,4,1] (12-ch)
)

i=0
for pair in "${ENCS[@]}"; do
  run="${pair%%:*}"; cond="${pair##*:}"; gpu=$((i % 4)); i=$((i + 1))
  ( for loss in mse "mse+corr"; do
      suf=mse; aux=(); [ "$loss" = "mse+corr" ] && { suf=msecorr5; aux=(--aux_weight 5); }
      CUDA_VISIBLE_DEVICES=$gpu nice -n 19 "$PY" dataset/01c_pdbbind_probe.py probe \
        --conditions "$cond" --epoch 49 --voxel_version v5 --split lp_edrscc_v2 \
        --feature_tag "$run" --exp_dir "exps/$run" --allow_stale_features --seeds 5 \
        --probe_loss "$loss" "${aux[@]}" --device cuda:0 --no_wandb --tag "${run}_${suf}"
    done ) > "log/reprobe5_${run}.log" 2>&1 &
  echo "launched $run (gpu $gpu, cond $cond)"
done
wait
echo "==== ALL 5-SEED REPROBE DONE ===="
