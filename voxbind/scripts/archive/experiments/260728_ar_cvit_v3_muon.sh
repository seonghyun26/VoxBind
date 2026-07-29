#!/bin/bash
# 260728 — MUON optimizer pilot on v3, 100M ChannelViT [7,4,2]. Muon (Newton-Schulz
# orthogonalized momentum) on the transformer BLOCK matrices (~89M), AdamW aux on the
# rest (~11M: embeds/norms/biases/patch-embed/recon-head). Constant LR (muon group keeps
# muon_lr, aux keeps cfg.lr=1e-4). Direct comparison target: v3/mask0.90 AdamW probe = 0.645.
# usage: 260728_ar_cvit_v3_muon.sh <GPUS> <PORT> <DATA v2|v3> <MASK> <EPOCHS> <MUONLR> <NAME>
set -uf
cd /home/shpark/prj-denovo/VoxBind/voxbind || exit 1
GPUS="$1"; PORT="$2"; DATA="$3"; MASK="$4"; EPOCHS="$5"; MUONLR="$6"; NAME="$7"
PY=/home/shpark/.conda/envs/voxbind/bin
CFG=config_train_atomblob_density_gradmag_channelvit_mae_40m_plinder_otf_mask050
RES=dataset/data/pdbbind/results
NG=$(echo "$GPUS" | tr ',' '\n' | wc -l)
if [ "$DATA" = "v2" ]; then DF=pretrain/data_train_plinder_v2_perelem.pt; RS=dataset/data/pretrain/xray_resample_plinder_v2_perelem; SUB=112000
else                        DF=pretrain/data_train_plinder_v3_perelem.pt; RS=dataset/data/pretrain/xray_resample_plinder_v3_perelem; SUB=70725; fi
DIM=640; DEPTH=18; HEADS=10
EXP=260728_ar_cvit_${NAME}

echo "===== [$EXP] MUON 100M '"$DATA"' · mask$MASK · ${EPOCHS}ep · muon_lr=$MUONLR START $(date '+%m-%d %H:%M:%S') GPU=$GPUS ====="
CUDA_VISIBLE_DEVICES=$GPUS "$PY/torchrun" --nnodes=1 --nproc_per_node="$NG" \
    --rdzv-backend=c10d --rdzv-endpoint=localhost:"$PORT" --rdzv-id="muon$NAME" \
    train_density.py --config-name="$CFG" exp_name="$EXP" bsz=4 accum_steps=8 \
    num_workers=6 prefetch_factor=2 num_epochs="$EPOCHS" \
    model.channel_groups=[7,4,2] model.dim=$DIM model.depth=$DEPTH model.heads=$HEADS \
    model.channel_group_dropout=0.15 mae.mask_ratio="$MASK" mae.ema_decay=0.999 compile.enabled=false \
    +optimizer.type=muon +optimizer.muon_lr="$MUONLR" +optimizer.muon_momentum=0.95 \
    dset.data_file=$DF dset.resample_dir=$RS dset.subset_n=$SUB dset.subset_val_n=100 \
  && echo "[$EXP] PRETRAIN OK $(date '+%H:%M:%S')" \
  || { echo "[$EXP] PRETRAIN FAILED $(date '+%H:%M:%S')"; exit 1; }

PEP=$((EPOCHS-1))
echo "###### [$EXP] PROBE start (e$PEP) $(date '+%H:%M:%S') ######"
bash scripts/04_probe.sh --exp "$EXP" --condition atomblob_density_gradmag \
    --tasks affinity --split lp_edrscc_v2 --epoch "$PEP" --gpu "${GPUS%%,*}" --tag "$EXP" \
    --num_workers 0 -- --require_density \
  && echo "[$EXP] PROBE OK" || echo "[$EXP] PROBE FAILED"
echo "###### [$EXP] result (r / rho / RMSE) ######"
tail -2 "$RES/probe_results_e${PEP}_v5_lp_edrscc_v2split_${EXP}.csv" 2>/dev/null
echo "===== [$EXP] COMPLETE $(date '+%m-%d %H:%M:%S') ====="
