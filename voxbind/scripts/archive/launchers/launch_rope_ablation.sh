#!/usr/bin/env bash
# Launch one run of the {density-only,gradmag-only} x {learnable,rope3d} ablation.
# Usage: launch_rope_ablation.sh <config-name> <exp_name> <pos_encoding> <gpus> [bsz] [accum]
#   config-name : e.g. config_train_density_only_vit_mae_40m_v5
#   exp_name    : e.g. density_only_vit_mae_40m_v5_rope3d
#   pos_encoding: learnable | rope3d
#   gpus        : e.g. 0,1,2,3  (nproc = number of comma-separated ids)
# Kept in a separate file so the orchestration shell's cmdline never contains the
# train literal (avoids the pkill self-match gotcha; see memory voxbind-train-ops).
set -euo pipefail

CFG="$1"; EXP="$2"; POS="$3"; GPUS="$4"; BSZ="${5:-16}"; ACCUM="${6:-2}"
NPROC=$(awk -F, '{print NF}' <<<"$GPUS")

ROOT=/home1/irteam/VoxBind
PY=/opt/conda/envs/voxbind/bin
OUT="$ROOT/voxbind/exps/$EXP"
LOG="$ROOT/voxbind/log/${EXP}.log"
mkdir -p "$OUT" "$ROOT/voxbind/log"

cd "$ROOT/voxbind"
CUDA_VISIBLE_DEVICES="$GPUS" setsid nohup "$PY/torchrun" \
  --standalone --nproc_per_node="$NPROC" \
  "$ROOT/voxbind/train_density_vit_mae.py" \
  --config-name="$CFG" \
  exp_name="$EXP" \
  output_dir="$OUT" \
  hydra.run.dir="$OUT" \
  model.pos_encoding="$POS" \
  bsz="$BSZ" accum_steps="$ACCUM" \
  > "$LOG" 2>&1 &

echo "launched $EXP (pid $!) on GPUs $GPUS  ->  $LOG"
