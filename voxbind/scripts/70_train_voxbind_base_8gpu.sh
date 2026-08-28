#!/usr/bin/env bash
# 70_train_voxbind_base_8gpu.sh
#   Vanilla VoxBind (ICML'24 recipe) trained from scratch on all 8 GPUs via DDP.
#
#   Config: stock configs/config_train.yaml defaults — model/voxbind (density-free,
#   111.6M params), dset/crossdocked, vox 64^3 @ 0.25A, smooth_sigma=0.9, lr=1e-5,
#   wd=1e-2, aug=True, 350 epochs.
#
#   bsz is PER-RANK under DDP, so bsz=8 x 8 ranks = effective batch 64, identical to
#   the single-process default (bsz=64 total). The optimizer math is unchanged.
#
#   wjs.n_targets=0 disables mid-training WJS sampling. Sampling fires at epochs
#   100/200/300/349 on rank 0 only while the other 7 ranks sit at the epoch barrier;
#   100 targets x 10 samples x 10.4k Langevin steps runs well past the 2h NCCL
#   watchdog timeout in train_ddp.py and would kill the run. Sample separately from
#   a checkpoint with sample.py instead.
#
#   Throughput knobs that do NOT change the math:
#     ddp_static_graph=true  DDP records the unused-param set once (the UNet3D time-embed
#                            MLP never gets grad with t=None) instead of re-traversing the
#                            autograd graph every step, as find_unused_parameters does.
#     num_workers=8          per-rank loader workers (8 x 8 ranks = 64; box has 384 cores).
#                            GPU util dipped to 75-82% at the default 4, i.e. input starved.
#
#   PYTHONPATH is set because this env's editable install still maps `voxbind` to
#   the deleted /home/shpark/prj-ligand/Voxbind checkout. `pip install -e .` from the
#   repo root would fix it permanently.
#
#   Env knobs: EXP_NAME, NUM_EPOCHS, BSZ (per-rank), NUM_WORKERS, RESUME (exp dir), MASTER_PORT.
set -uo pipefail
ROOT=/home/shpark/prj-denovo/Voxbind
cd "$ROOT/voxbind" || exit 1

EXP_NAME="${EXP_NAME:-260827_voxbind_base_8gpu}"
NUM_EPOCHS="${NUM_EPOCHS:-350}"
BSZ="${BSZ:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
RESUME="${RESUME:-}"
MASTER_PORT="${MASTER_PORT:-29531}"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TORCHDYNAMO_DISABLE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC=$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")

mkdir -p logs
LOG="logs/${EXP_NAME}.log"

RESUME_ARG=()
[ -n "$RESUME" ] && RESUME_ARG=("resume=$RESUME")

echo ">> exp=$EXP_NAME nproc=$NPROC bsz=$BSZ (effective $((BSZ * NPROC))) epochs=$NUM_EPOCHS workers=$NUM_WORKERS"
echo ">> log: $ROOT/voxbind/$LOG"

exec torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" train_ddp.py \
    bsz="$BSZ" \
    num_epochs="$NUM_EPOCHS" \
    num_workers="$NUM_WORKERS" \
    ddp_static_graph=true \
    wjs.n_targets=0 \
    exp_name="$EXP_NAME" \
    wandb_tags="[voxbind,baseline,crossdocked,sig0.9,8gpu,ddp]" \
    "${RESUME_ARG[@]}" \
    >> "$LOG" 2>&1
