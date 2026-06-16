#!/bin/bash
# sample_4gpu.sh — Run original VoxBind evaluation across 4 GPUs (0–3).
#
# Shards the pocket range across GPUs by launching 4 background sample.py
# processes, each covering 25 pockets.  Results land in non-overlapping
# target_XX/ subdirs of the same save_dir and merge automatically.
#
# ── Usage ─────────────────────────────────────────────────────────────────────
#   bash scripts/sample_4gpu.sh --exp EXP [options]
#
# Required:
#   --exp EXP       experiment dir under exps/ → pretrained_path=exps/EXP
#
# Optional:
#   --split S       val | test                       (default: test)
#   --targets N     total pockets to sample          (default: 100, must be div by 4)
#   --samples N     wjs.n_samples_per_pocket         (default: 10)
#   --out NAME      result sub-dir name              (default: res_<split>_4gpu)
#   --gpus G        comma-separated 4 GPU ids        (default: 0,1,2,3)
#   --dry-run       print commands and exit
#
# ── Example ───────────────────────────────────────────────────────────────────
#   bash scripts/sample_4gpu.sh --exp exp_sig0.9 --split test --targets 100
set -euo pipefail

VOX="$(cd "$(dirname "$0")/.." && pwd)"
PY=/home/shpark/miniforge3/envs/voxbind/bin/python
LOG=$VOX/log
ts(){ date "+%Y-%m-%d %H:%M:%S"; }
die(){ echo "sample_4gpu.sh: $*" >&2; exit 1; }

# ── Defaults ──────────────────────────────────────────────────────────────────
EXP=""; SPLIT=test; TARGETS=100; SAMPLES=10; OUT=""; GPUS="0,1,2,3"; DRYRUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --exp)      EXP="$2";    shift 2;;
        --split)    SPLIT="$2";  shift 2;;
        --targets)  TARGETS="$2"; shift 2;;
        --samples)  SAMPLES="$2"; shift 2;;
        --out)      OUT="$2";    shift 2;;
        --gpus)     GPUS="$2";   shift 2;;
        --dry-run)  DRYRUN=1;    shift;;
        -h|--help)  awk 'NR==1{next} /^#/{print;next} {exit}' "$0"; exit 0;;
        *)          die "unknown arg: $1";;
    esac
done

[[ -n "$EXP" ]] || die "--exp EXP is required"
[[ $((TARGETS % 4)) -eq 0 ]] || die "--targets must be divisible by 4 (got $TARGETS)"
[[ -z "$OUT" ]] && OUT="res_${SPLIT}_4gpu"

PRETRAINED=$VOX/exps/$EXP
SAVE_DIR=$VOX/exps/$EXP/samples/$OUT
PER_GPU=$((TARGETS / 4))

IFS=',' read -ra GPU_LIST <<< "$GPUS"
[[ ${#GPU_LIST[@]} -eq 4 ]] || die "--gpus must be exactly 4 comma-separated IDs (got: $GPUS)"

mkdir -p "$LOG" "$SAVE_DIR"
cd "$VOX"

echo "[$(ts)] sample_4gpu  exp=$EXP  split=$SPLIT  targets=$TARGETS  samples=$SAMPLES  gpus=$GPUS"
echo "          save_dir=$SAVE_DIR"
echo "          per-GPU: $PER_GPU pockets each"

PIDS=()
for i in 0 1 2 3; do
    GPU=${GPU_LIST[$i]}
    START=$((i * PER_GPU))
    END=$(( (i + 1) * PER_GPU - 1 ))
    LOGFILE=$LOG/sample_${EXP}_${OUT}_gpu${GPU}.log

    CMD=( "$PY" "$VOX/sample.py"
          pretrained_path="$PRETRAINED"
          wjs.split="$SPLIT"
          wjs.n_targets=$((TARGETS + 10))   # high enough to not cut early
          wjs.start="$START"
          wjs.end="$END"
          wjs.n_samples_per_pocket="$SAMPLES"
          out_dir="$OUT"
          save_dir="$SAVE_DIR" )

    echo "  GPU $GPU → pockets $START–$END  (log: $LOGFILE)"
    if [[ "$DRYRUN" -eq 1 ]]; then
        echo "    CUDA_VISIBLE_DEVICES=$GPU \\"
        printf '    %q ' "${CMD[@]}"; echo
    else
        CUDA_VISIBLE_DEVICES="$GPU" "${CMD[@]}" > "$LOGFILE" 2>&1 &
        PIDS+=($!)
    fi
done

[[ "$DRYRUN" -eq 1 ]] && exit 0

echo "[$(ts)] all 4 workers launched (PIDs: ${PIDS[*]})"
echo "  tail logs:  tail -f $LOG/sample_${EXP}_${OUT}_gpu*.log"

# wait for all and report per-GPU exit codes
FAILED=0
for i in "${!PIDS[@]}"; do
    GPU=${GPU_LIST[$i]}
    PID=${PIDS[$i]}
    if wait "$PID"; then
        echo "[$(ts)] GPU $GPU done (ok)"
    else
        echo "[$(ts)] GPU $GPU FAILED (exit $?)" >&2
        FAILED=1
    fi
done

if [[ "$FAILED" -eq 0 ]]; then
    echo "[$(ts)] all done  ->  $SAVE_DIR"
    echo "  pockets written: $(ls -d "$SAVE_DIR"/target_* 2>/dev/null | wc -l)"
else
    echo "[$(ts)] one or more GPUs failed — check logs in $LOG/" >&2
    exit 1
fi
