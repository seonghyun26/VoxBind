#!/usr/bin/env bash
# 69_resume_frozen_v3_after_funcbind.sh
#   Wait for the four FuncBind MCP sampling chunks (cmp10/*) to finish, then resume
#   the frozen-encoder v3 fine-tune that was stopped to free the GPUs.
#
#   The run was stopped at epoch 212 (checkpoint written 2026-08-20 01:40:15) so the
#   four H200s could host vanilla-vs-finetuned FuncBind sampling. GPU work does not
#   overlap: this only fires once every chunk has written its exit_code.
#
#   train_ddp.py's resume path reloads cfg.yaml from the run dir wholesale and takes
#   start_epoch from the checkpoint, so the original ~40 CLI overrides must NOT be
#   repeated here — only wandb / wjs / num_epochs / resume_epoch survive from the
#   command line (see train_ddp.py, "resume?" block).
#
#   Lives in a script FILE on purpose: a chain loop typed into a shell puts the
#   literal "train_ddp" in that shell's own cmdline, which makes a later
#   `pkill -f train_ddp` kill the orchestrator too (see [[voxbind-train-ops]]).
set -uo pipefail

REPO=/home1/irteam/VoxBind
EXP="$REPO/voxbind/exps/voxbind_frozen_v3_100m_mask090_default_mlp3_h32_sig0.9"
CMP=/home1/irteam/funcbind/artifacts/reproduction/mcpp/cmp10
CHUNKS=(vanilla_a vanilla_b finetuned_a finetuned_b)
POLL=${POLL:-120}
TIMEOUT=${TIMEOUT:-43200}
LOG="$EXP/resume_chain.log"
say(){ echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

say "waiting for ${#CHUNKS[@]} sampling chunks under $CMP"
waited=0
while :; do
    done_n=0
    for c in "${CHUNKS[@]}"; do [ -f "$CMP/$c/exit_code" ] && done_n=$((done_n + 1)); done
    (( done_n >= ${#CHUNKS[@]} )) && { say "all $done_n chunks finished"; break; }
    if (( waited >= TIMEOUT )); then say "TIMEOUT after ${waited}s with $done_n/${#CHUNKS[@]} done — resuming anyway"; break; fi
    sleep "$POLL"; waited=$((waited + POLL))
done

for c in "${CHUNKS[@]}"; do
    rc=$(cat "$CMP/$c/exit_code" 2>/dev/null || echo "MISSING")
    [ "$rc" = "0" ] || say "WARNING: chunk $c exited $rc"
done

# A sampling rank that is still tearing down still holds its GPU memory; resuming
# into that races the trainer's allocator against a process that is about to exit.
say "waiting for sampling processes to release the GPUs"
for _ in $(seq 1 60); do
    pgrep -f "[s]ample_fb.py" >/dev/null || break
    sleep 10
done

say "resuming training from $EXP"
cd "$REPO/voxbind" || exit 1
export LD_LIBRARY_PATH=/opt/conda/envs/voxbind/lib
export TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
setsid nohup /opt/conda/envs/voxbind/bin/torchrun --standalone --nproc_per_node=4 \
    train_ddp.py \
    --config-name config_train_voxbind_frozenenc_channelvit_atomblob7_v2p1 \
    resume="$EXP" \
    wandb=true \
    num_epochs=350 \
    wjs.n_targets=0 \
    >>"$EXP/train_launch.log" 2>&1 &
say "relaunched pid $! -> $EXP/train_launch.log"
