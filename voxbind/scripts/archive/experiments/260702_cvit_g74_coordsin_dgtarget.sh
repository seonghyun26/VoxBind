#!/bin/bash
# 260702_cvit_g74_coordsin_dgtarget.sh — coords-only INPUT, density+gradmag as recon TARGET only.
# Same architecture/recipe as the ChannelViT [7,4,2] C+D+G winner (ρ 0.637), but the encoder is fed
# ONLY the 11 atom channels (ChannelViT groups [7,4]); density + gradmag are dropped from the input
# and kept ONLY as reconstruction targets. So a coords-only encoder must PREDICT the electron-density
# field (+‖∇ρ‖) at masked voxels from atoms alone — the always-on (p=1.0), genuine-[7,4] version of
# the trial-2 cross-modal experiment (which zeroed density for half the batch, groups stayed [7,4,2]).
#   n_in=11 (patch embed [7,4]) / n_recon=13 (atoms+density+gradmag). Masked-voxels-only recon loss
#   (standard MAE, directly comparable to the 0.637 base). Opt-in via +mae.density_input=false;
#   density_input=true (default) reproduces the base bit-for-bit.
# PLINDER v1 (data_train_plinder.pt + OTF xray_resample_plinder), 100 ep, bsz8×accum4 eff128,
# compile OFF, GPU 0-3. Probe = coords-only (atomblob) frozen encoder on lp_edrscc_v2 (Kd/Ki).
set -uf
cd /home/shpark/prj-denovo/VoxBind/voxbind || exit 1
PY=/home/shpark/.conda/envs/voxbind/bin
CFG=config_train_atomblob_density_gradmag_channelvit_mae_40m_plinder_otf_mask050
RES=dataset/data/pdbbind/results
GPUS=0,1,2,3; PORT=29631; RID=arG74CIDT
EXP=260702_cvit_g74_coordsin_dgtarget

echo "===== [7,4] coords-in / D+G-target START $(date '+%m-%d %H:%M:%S') GPU=$GPUS ====="
CUDA_VISIBLE_DEVICES=$GPUS "$PY/torchrun" --nnodes=1 --nproc_per_node=4 \
    --rdzv-backend=c10d --rdzv-endpoint=localhost:"$PORT" --rdzv-id="$RID" \
    train_density.py --config-name="$CFG" exp_name="$EXP" bsz=8 accum_steps=4 compile.enabled=false \
    'model.channel_groups=[7,4]' model.n_in_channels=11 +mae.density_input=false \
    'wandb_tags=[pretrain,channelvit,cdg,coordsin,dgtarget,plinder,otf,mask050,g74]' \
  && echo "[$EXP] PRETRAIN OK $(date '+%H:%M:%S')" \
  || { echo "[$EXP] PRETRAIN FAILED $(date '+%H:%M:%S')"; exit 1; }

echo "###### [$EXP] PROBE start $(date '+%H:%M:%S') ######"
# Encoder input is coords-only (density_input=false → atoms-only patch embed); the probe's
# infer_feature_spec reads that from cfg.yaml and feeds 11-ch atoms. Coords-only condition, no --require_density.
bash scripts/04_probe.sh --exp "$EXP" --condition atomblob \
    --tasks affinity --split lp_edrscc_v2 --gpu "${GPUS%%,*}" --tag "$EXP" --num_workers 0 \
  && echo "[$EXP] PROBE OK" || echo "[$EXP] PROBE FAILED"
echo "###### [$EXP] result ######"
tail -2 "$RES/probe_results_e99_v5_lp_edrscc_v2split_${EXP}.csv" 2>/dev/null
echo "===== [7,4] coords-in / D+G-target COMPLETE $(date '+%m-%d %H:%M:%S') ====="
