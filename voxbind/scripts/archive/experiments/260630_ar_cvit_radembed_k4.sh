#!/bin/bash
# 260630_ar_cvit_radembed_k4.sh — DRAMATIC change #7: LEARNABLE RADIUS-CONDITIONED ATOM EMBEDDING.
# Replace the one-hot per-element atom channels with a learned k-dim feature φ(vdW-radius) per role:
# two small MLPs φ_lig, φ_poc map a scalar radius → ℝ^k, evaluated at the per-element vdW radii to
# embed the ligand/pocket occupancy into k channels each (groups [k,k,2] with density+gradmag). It's a
# learnable front-end INSIDE the encoder (so the frozen probe inherits it); the MAE still reconstructs
# the RAW 13-ch atomblob+D+G, so this is a controlled swap of ONLY the input atom encoding vs [7,4,2].
# Because φ sees the continuous radius and k=4<7, it's non-absorbable (a learned soft vocabulary) and
# recovers element separation that roleblob's size-encoding sums away. Caveat: lig S and P share vdW
# 1.8Å so radius cannot separate them (by construction). k=4. [7,4,2]-raw base; bsz8×accum4×4gpu =
# eff-batch 128; compile OFF. GPU 0-3. Probe lp_edrscc_v2. Report → notebook/html/260701/trial7.html
set -uf
cd /home/shpark/prj-denovo/VoxBind/voxbind || exit 1
PY=/home/shpark/.conda/envs/voxbind/bin
CFG=config_train_atomblob_density_gradmag_channelvit_mae_40m_plinder_otf_mask050
RES=dataset/data/pdbbind/results
GPUS=0,1,2,3; PORT=29615; RID=arRadEmb
EXP=260630_ar_cvit_radembed_k4

echo "===== AR radius-embed k=4 START $(date '+%m-%d %H:%M:%S') GPU=$GPUS ====="
CUDA_VISIBLE_DEVICES=$GPUS "$PY/torchrun" --nnodes=1 --nproc_per_node=4 \
    --rdzv-backend=c10d --rdzv-endpoint=localhost:"$PORT" --rdzv-id="$RID" \
    train_density.py --config-name="$CFG" exp_name="$EXP" bsz=8 accum_steps=4 compile.enabled=false \
    model.channel_groups=[7,4,2] +model.radius_embed_k=4 \
  && echo "[$EXP] PRETRAIN OK $(date '+%H:%M:%S')" \
  || { echo "[$EXP] PRETRAIN FAILED $(date '+%H:%M:%S')"; exit 1; }
echo "###### [$EXP] PROBE start $(date '+%H:%M:%S') ######"
bash scripts/04_probe.sh --exp "$EXP" --condition atomblob_density_gradmag \
    --tasks affinity --split lp_edrscc_v2 --gpu "${GPUS%%,*}" --tag "$EXP" --num_workers 0 \
    -- --require_density \
  && echo "[$EXP] PROBE OK" || echo "[$EXP] PROBE FAILED"
echo "###### [$EXP] result ######"
tail -2 "$RES/probe_results_e99_v5_lp_edrscc_v2split_${EXP}.csv" 2>/dev/null
echo "===== AR radius-embed k=4 COMPLETE $(date '+%m-%d %H:%M:%S') ====="
