#!/bin/bash
# 02_sample.sh — unified VoxBind sampler (sample.py), with optional Vina dock-eval.
# Replaces the per-experiment sampling scripts 09/12/16/17/20/21/23/24/25/26/27/30:
# pick the experiment, split, pocket/sample counts and output sub-dir; everything
# else is derived. Pass --dock to chain the metrics.py Vina eval after sampling.
#
# ── Usage ─────────────────────────────────────────────────────────────────────
#   bash scripts/02_sample.sh --exp EXP [options]
#
# Required:
#   --exp EXP         experiment dir under exps/ → pretrained_path=exps/EXP
#
# Sampling:
#   --split S         val | test                       (default test)
#   --targets N       wjs.n_targets                     (default 100)
#   --samples N       wjs.n_samples_per_pocket          (default 10)
#   --range S:E       wjs.start:wjs.end (full-test sweep, e.g. 0:200); omit for default 0:100
#   --out NAME        result sub-dir → save_dir=exps/EXP/samples/NAME, out_dir=NAME
#                     (default res_<split>)
#   --gpu G           single GPU id                     (default 7)
#   --data VER        v1 pretrain/xray_crops_aligned [default] | vN pretrain/xray_crops_aligned_vN
#                     (N≥2 → normalize=false; match the model's training version)
#   --subset N        dset.subset_n                     (default 10000)
#   --no-density      dset.use_xray=false  (for no-density / reproduction models)
#
# Dock eval (optional Phase 2, CPU):
#   --dock            after sampling, run notebook/webapp/metrics.py --docking vina_dock
#   --dock-workers N  (default 8)
#   --dock-cpu N      (default 4)
#
# Misc:
#   --dry-run         print the resolved sample command and exit
#   -- ARG ...        pass remaining args to sample.py (Hydra) verbatim
#
# ── Reproductions (resolved sample-config verified identical) ─────────────────
#   16 = --exp 260521_voxbind_10k_density_mae_frozen --split val --targets 1 \
#        --out res_ep99_quick10 --gpu 7
#   23 = --exp 260523_voxbind_10k_density_vit_mae_frozen_e0199_snap --split val \
#        --targets 9 --out res_ep199_10pockets --gpu 7 --dock
#   26 = --exp 260523_voxbind_10k_density_vit_mae_frozen_e0379_snap --split test \
#        --targets 200 --range 0:200 --out res_ep379_fulltest --gpu 7
#   30 = --exp 260525_voxbind_10k_density_vit_electra_frozen --split test \
#        --targets 200 --range 0:200 --out res_electra_fulltest --gpu 6 --dock
set -u

VOX=/home/shpark/prj-denovo/VoxBind/voxbind
PY=/home/shpark/.conda/envs/voxbind/bin/python
DATA=$VOX/dataset/data
LOG=$VOX/log
METRICS=$VOX/../notebook/webapp/metrics.py
ts(){ date "+%Y-%m-%d %H:%M:%S"; }
die(){ echo "02_sample.sh: $*" >&2; exit 1; }

# ── Defaults ──────────────────────────────────────────────────────────────────
EXP=""; SPLIT=test; TARGETS=100; SAMPLES=10; RANGE=""; OUT=""; GPU=7
DATA_VER=v1; SUBSET=10000; USE_XRAY=true
DOCK=0; DOCK_WORKERS=8; DOCK_CPU=4; DRYRUN=0
PASSTHRU=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --exp)          EXP="$2"; shift 2;;
        --split)        SPLIT="$2"; shift 2;;
        --targets)      TARGETS="$2"; shift 2;;
        --samples)      SAMPLES="$2"; shift 2;;
        --range)        RANGE="$2"; shift 2;;
        --out)          OUT="$2"; shift 2;;
        --gpu)          GPU="$2"; shift 2;;
        --data)         DATA_VER="$2"; shift 2;;
        --subset)       SUBSET="$2"; shift 2;;
        --no-density)   USE_XRAY=false; shift;;
        --dock)         DOCK=1; shift;;
        --dock-workers) DOCK_WORKERS="$2"; shift 2;;
        --dock-cpu)     DOCK_CPU="$2"; shift 2;;
        --dry-run)      DRYRUN=1; shift;;
        --)             shift; PASSTHRU=("$@"); break;;
        -h|--help)      awk 'NR==1{next} /^#/{print;next} {exit}' "$0"; exit 0;;
        *)              die "unknown arg: $1  (see --help)";;
    esac
done

[[ -n "$EXP" ]] || die "--exp EXP is required"
[[ -z "$OUT" ]] && OUT="res_${SPLIT}"

# Data version: v1 = pretrain/xray_crops_aligned (normalize default true); vN (N≥2) =
# pre-normalised pretrain/xray_crops_aligned_vN → normalize=false. Sample with the SAME
# version the model was trained on. Generic so v4, v5, … need no edit.
if [[ "$DATA_VER" == v1 ]]; then
    CROPS=$DATA/pretrain/xray_crops_aligned;          NORMALIZE=""
elif [[ "$DATA_VER" =~ ^v[0-9]+$ ]]; then
    CROPS=$DATA/pretrain/xray_crops_aligned_$DATA_VER; NORMALIZE=false
else
    die "unknown --data '$DATA_VER' (expected v1, v2, v3, …)"
fi

SAVE_DIR=$VOX/exps/$EXP/samples/$OUT

# ── range → wjs.start / wjs.end (omit when not given → sample.py defaults) ─────
RANGE_ARGS=()
if [[ -n "$RANGE" ]]; then
    [[ "$RANGE" =~ ^[0-9]+:[0-9]+$ ]] || die "--range must be START:END integers, no spaces (e.g. 0:200)"
    RANGE_ARGS=( wjs.start="${RANGE%%:*}" wjs.end="${RANGE##*:}" )
fi

# ── Assemble sample command ───────────────────────────────────────────────────
CMD=( "$PY" "$VOX/sample.py"
      pretrained_path="$VOX/exps/$EXP"
      dset=crossdocked_xray
      dset.data_dir="$DATA"
      dset.crops_dir="$CROPS"
      dset.subset_n="$SUBSET" dset.subset_xray_only=true dset.subset_val_n=100
      dset.use_xray="$USE_XRAY"
      wjs.split="$SPLIT" wjs.n_targets="$TARGETS" )
[[ -n "$NORMALIZE" ]] && CMD+=( dset.normalize="$NORMALIZE" )
CMD+=( "${RANGE_ARGS[@]}" )
CMD+=( wjs.n_samples_per_pocket="$SAMPLES"
       out_dir="$OUT" save_dir="$SAVE_DIR" )
[[ ${#PASSTHRU[@]} -gt 0 ]] && CMD+=( "${PASSTHRU[@]}" )

LOGFILE=$LOG/sample_${EXP}_${OUT}.log

echo "[$(ts)] sample  exp=$EXP  split=$SPLIT  targets=$TARGETS  samples=$SAMPLES${RANGE:+  range=$RANGE}  gpu=$GPU  data=$DATA_VER"
echo "          save_dir=$SAVE_DIR${DOCK:+  (+dock eval)}"
if [[ "$DRYRUN" -eq 1 ]]; then
    echo "CUDA_VISIBLE_DEVICES=$GPU \\"
    printf '  %q' "${CMD[@]}"; echo
    [[ "$DOCK" -eq 1 ]] && echo "  # then: $PY $METRICS $SAVE_DIR --docking vina_dock --workers $DOCK_WORKERS --cpu $DOCK_CPU --skip-existing"
    exit 0
fi

mkdir -p "$LOG" "$SAVE_DIR"
cd "$VOX" || die "cd $VOX failed"
echo "[$(ts)] Phase 1: WJS sample -> log: $LOGFILE"
CUDA_VISIBLE_DEVICES="$GPU" "${CMD[@]}" > "$LOGFILE" 2>&1
RC=$?
echo "[$(ts)] WJS sample done (exit $RC)"
[[ $RC -ne 0 ]] && exit $RC

if [[ "$DOCK" -eq 1 ]]; then
    echo "[$(ts)] Phase 2: Vina dock eval (workers=$DOCK_WORKERS cpu=$DOCK_CPU)"
    "$PY" "$METRICS" "$SAVE_DIR" \
        --docking vina_dock --workers "$DOCK_WORKERS" --cpu "$DOCK_CPU" --skip-existing \
        >> "$LOGFILE" 2>&1
    DOCK_RC=$?
    echo "[$(ts)] dock eval done (exit $DOCK_RC)"
    [[ $DOCK_RC -ne 0 ]] && exit $DOCK_RC
fi
echo "[$(ts)] all done  ->  $SAVE_DIR"
