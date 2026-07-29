#!/usr/bin/env bash
# Launch a 4-GPU ViT-MAE train in its OWN session (setsid) so the whole process tree can be
# killed via the recorded pgid file (avoids pkill/ps self-match on process names). Reusable for
# util-tuning bursts and the real runs. Writes exps/<EXP>.train.pgid = the session-leader pgid.
#   Usage:  _train_bg.sh <config_name> <exp_name> [extra hydra overrides ...]
#   Kill :  kill -TERM -- -"$(cat exps/<EXP>.train.pgid)"
set -u
ROOT=/home1/irteam/VoxBind/voxbind
PY=/opt/conda/envs/voxbind/bin
cd "$ROOT" || exit 1
export LD_LIBRARY_PATH=/opt/conda/envs/voxbind/lib:${LD_LIBRARY_PATH:-}
export TORCHDYNAMO_DISABLE=1
CFG=$1; EXP=$2; shift 2
LOG=$ROOT/log/${EXP}.log
PGF=$ROOT/exps/${EXP}.train.pgid
mkdir -p "$ROOT/exps/$EXP"
ARGS=( "$PY/torchrun" --standalone --nproc_per_node=4
  "$ROOT/train_density_vit_mae.py" --config-name="$CFG"
  exp_name="$EXP" output_dir="$ROOT/exps/$EXP" hydra.run.dir="$ROOT/exps/$EXP" "$@" )
# setsid (not already a pgroup leader when backgrounded) -> new session; $$ inside == leader pid == pgid.
# exec keeps that pid, so the pgid file points at the whole training tree's group.
setsid bash -c 'echo $$ > "$0"; exec "$@"' "$PGF" "${ARGS[@]}" > "$LOG" 2>&1 &
sleep 1
echo "launched exp=$EXP pgid=$(cat "$PGF" 2>/dev/null) log=$LOG"
