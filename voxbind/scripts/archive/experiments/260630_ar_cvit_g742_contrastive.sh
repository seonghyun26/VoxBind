#!/bin/bash
# 260630_ar_cvit_g742_contrastive.sh — DRAMATIC change #6: CONTRASTIVE auxiliary (new algorithm).
# Every prior dramatic trial varied the MASKING (cluster/ligand/interface) or the pretext target
# (denoise) — all within reconstruction. This adds a genuinely DIFFERENT objective: a SimCLR-style
# InfoNCE contrastive loss alongside MAE. Two augmented views = two INDEPENDENT MAE maskings of the
# same complex; the encoder is pulled to map both maskings to the same pooled vector — directly
# shaping the representation the frozen probe consumes. Opt-in, non-destructive (λ=0 → pure MAE).
#   L = L_mae + λ·L_infonce,  λ=0.05  (balances MAE≈0.065 vs InfoNCE≈1.6 → contrastive on par, not dominant)
#   local in-batch negatives (bsz8 → 14/anchor), projector 2-layer MLP D→D→128, temp 0.2.
# View-a's feature is tapped FREE from the MAE pass (return_tokens); view-b costs +1 trunk forward
# (~1.7× encoder compute/step). [7,4,2] base; bsz8 × accum4 × 4gpu = eff-batch 128; compile OFF
# (data-dependent dual-forward branch). GPU 0-3. Probe lp_edrscc_v2. Report → notebook/html/260701/trial6.html
set -uf
cd /home/shpark/prj-denovo/VoxBind/voxbind || exit 1
PY=/home/shpark/.conda/envs/voxbind/bin
CFG=config_train_atomblob_density_gradmag_channelvit_mae_40m_plinder_otf_mask050
RES=dataset/data/pdbbind/results
GPUS=0,1,2,3; PORT=29613; RID=arG742CON
EXP=260630_ar_cvit_g742_contrastive

echo "===== AR [7,4,2] CONTRASTIVE (InfoNCE λ=0.05) START $(date '+%m-%d %H:%M:%S') GPU=$GPUS ====="
CUDA_VISIBLE_DEVICES=$GPUS "$PY/torchrun" --nnodes=1 --nproc_per_node=4 \
    --rdzv-backend=c10d --rdzv-endpoint=localhost:"$PORT" --rdzv-id="$RID" \
    train_density.py --config-name="$CFG" exp_name="$EXP" bsz=8 accum_steps=4 compile.enabled=false \
    model.channel_groups=[7,4,2] \
    +mae.contrastive_weight=0.05 +mae.contrastive_dim=128 +mae.contrastive_temp=0.2 \
  && echo "[$EXP] PRETRAIN OK $(date '+%H:%M:%S')" \
  || { echo "[$EXP] PRETRAIN FAILED $(date '+%H:%M:%S')"; exit 1; }
echo "###### [$EXP] PROBE start $(date '+%H:%M:%S') ######"
bash scripts/04_probe.sh --exp "$EXP" --condition atomblob_density_gradmag \
    --tasks affinity --split lp_edrscc_v2 --gpu "${GPUS%%,*}" --tag "$EXP" --num_workers 0 \
    -- --require_density \
  && echo "[$EXP] PROBE OK" || echo "[$EXP] PROBE FAILED"
echo "###### [$EXP] result ######"
tail -2 "$RES/probe_results_e99_v5_lp_edrscc_v2split_${EXP}.csv" 2>/dev/null
echo "===== AR [7,4,2] CONTRASTIVE COMPLETE $(date '+%m-%d %H:%M:%S') ====="
