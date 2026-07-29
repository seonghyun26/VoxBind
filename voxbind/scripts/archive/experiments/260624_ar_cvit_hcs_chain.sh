#!/bin/bash
# 260624_ar_cvit_hcs_chain.sh — NEW component: ChannelViT Hierarchical Channel
# Sampling (HCS, Bao et al. 2023). The 260623/260624 autoresearch swept only generic
# ViT/MAE knobs (grouping, mask ratio, recon weight, lr, wd, block, head-depth, RoPE);
# none touched the ChannelViT-native training procedure. HCS per-forward drops whole
# channel-GROUPS (Bernoulli, ≥1 kept) so attention learns to fuse variable channel
# subsets and the MAE becomes a cross-modal completion task (infer density from atoms
# and vice-versa). Base = the c1 winner: groups [7,4,2], mask 0.50, dens-wt 0.1.
#
# Single lane on GPU 0-3 (4-GPU DDP, bsz8×accum4 → eff-batch 128, IDENTICAL to the
# rest of the campaign). compile OFF: HCS varies the token count per step, which would
# thrash torch.compile; compile is result-neutral here, only ~1.5× slower. Two trials
# run SEQUENTIALLY: p=0.15 then p=0.30. Per trial: 40M ChannelViT pretrain (100ep) →
# frozen affinity probe on lp_edrscc_v2 (Kd/Ki, n=1320). Best-so-far to beat: rho 0.637.
set -uf                       # -f: keep hydra list overrides like [7,4,2] literal
cd /home/shpark/prj-denovo/VoxBind/voxbind || exit 1
PY=/home/shpark/.conda/envs/voxbind/bin
CFG=config_train_atomblob_density_gradmag_channelvit_mae_40m_plinder_otf_mask050
RES=dataset/data/pdbbind/results
LOG=log
GPUS=0,1,2,3
PORT=29541
RID=cvitHCS

run_trial () {                # $1=EXP ; rest = extra hydra overrides
  local EXP=$1; shift
  echo "###### [$EXP] PRETRAIN start $(date '+%m-%d %H:%M:%S') GPU=$GPUS :: $* ######"
  CUDA_VISIBLE_DEVICES=$GPUS "$PY/torchrun" --nnodes=1 --nproc_per_node=4 \
      --rdzv-backend=c10d --rdzv-endpoint=localhost:"$PORT" --rdzv-id="$RID" \
      train_density.py --config-name="$CFG" exp_name="$EXP" bsz=8 accum_steps=4 \
      model.channel_groups=[7,4,2] compile.enabled=false "$@" \
    && echo "[$EXP] PRETRAIN OK $(date '+%H:%M:%S')" \
    || { echo "[$EXP] PRETRAIN FAILED $(date '+%H:%M:%S')"; return 1; }
  echo "###### [$EXP] PROBE start $(date '+%H:%M:%S') ######"
  bash scripts/04_probe.sh --exp "$EXP" --condition atomblob_density_gradmag \
      --tasks affinity --split lp_edrscc_v2 --gpu "${GPUS%%,*}" --tag "$EXP" --num_workers 0 \
      -- --require_density \
    && echo "[$EXP] PROBE OK" || echo "[$EXP] PROBE FAILED"
  echo "###### [$EXP] result ######"
  tail -2 "$RES/probe_results_e99_v5_lp_edrscc_v2split_${EXP}.csv" 2>/dev/null
}

echo "===== AR ChannelViT HCS chain START $(date '+%m-%d %H:%M:%S') (GPU $GPUS) ====="
run_trial 260624_ar_cvit_hcs015 model.channel_group_dropout=0.15
run_trial 260624_ar_cvit_hcs030 model.channel_group_dropout=0.30
echo "===== AR ChannelViT HCS chain COMPLETE $(date '+%m-%d %H:%M:%S') ====="
