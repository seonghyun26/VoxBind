#!/bin/bash
# train_downstream.sh — unified launcher for downstream VoxBind training
# (train_ddp.py): trains the diffusion model, optionally with a (frozen)
# pre-trained density encoder. Replaces the per-variant scripts 11/14/15/19/22/29
# (and the per-arm calls inside the 01a/08 orchestrators).
#
# ── Usage ─────────────────────────────────────────────────────────────────────
#   bash scripts/train_downstream.sh --gpus GPUS [options]
#
# Required:
#   --gpus GPUS       comma list or dash range — "1-6" or "1,2,3,4,5"
#
# Model / encoder:
#   --density         model.with_density=true   [default]
#   --no-density      model.with_density=false  (baseline; ignores density input)
#   --encoder KIND    none | cnn | vit          [default none = train from scratch]
#                       cnn → model.density_encoder_blocks=4
#                       vit → model.density_encoder_type=vit
#   --encoder-ckpt P  model.density_pretrained_path=P   (required when --encoder != none)
#   --no-freeze       model.density_freeze=false  (default: freeze when an encoder is loaded)
#
# Data / schedule:
#   --subset N        dset.subset_n           (default 10000)
#   --data VER        v1 pretrain/xray_crops_aligned [default] | vN pretrain/xray_crops_aligned_vN
#                     (N≥2 → normalize=false; e.g. v2, v3, v4)
#   --epochs N        num_epochs              (default 100)
#   --bsz N           per-rank batch size     (default 4)
#   --accum N         accum_steps             (default 1)
#   --seed N          (default 42)
#
# Resume (per the 11/15/22 lesson, the FULL arg set must be re-passed on resume):
#   --resume          resume=<output_dir>   (continue this exp's checkpoint.pth.tar)
#   --resume-from P   resume=P              (explicit path)
#   --resume-epoch N  resume_epoch=N        (force the start epoch; codebase '+0' convention)
#
# Misc:
#   --name NAME       exp_name / output dir  (default: <yymmdd>_voxbind_<subset>_<auto>)
#   --tags a,b,c      extra wandb tags appended to the auto tags
#   --dry-run         print the resolved command and exit
#   -- ARG ...        pass remaining args to Hydra verbatim
#
# ── Reproductions (resolved-config verified identical) ────────────────────────
#   11 = --no-... --density --subset 10000 --epochs 100 --gpus 1-6 --resume \
#        --name 260519_voxbind_10k_density_aligned          (scratch, with density)
#   14 = --density --encoder cnn --encoder-ckpt <mae> --gpus 1-6 --epochs 100 ...
#   15 = (14) --resume --resume-epoch 100
#   19 = --density --encoder vit --encoder-ckpt <vit-mae> --gpus 1-6 --epochs 200 ...
#   22 = (19) --epochs 180 --resume --resume-epoch 200
#   29 = --density --encoder vit --encoder-ckpt <electra> --gpus 1-5 --epochs 200 ...
set -u

VOX="${VOXBIND_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY=/home/shpark/.conda/envs/voxbind/bin
DATA=$VOX/dataset/data
LOG=$VOX/log
ts(){ date "+%Y-%m-%d %H:%M:%S"; }
die(){ echo "train_downstream.sh: $*" >&2; exit 1; }

# ── Defaults ──────────────────────────────────────────────────────────────────
GPUS=""; WITH_DENSITY=true; ENCODER=none; ENC_CKPT=""; FREEZE=true
SUBSET=10000; DATA_VER=v1; EPOCHS=100; BSZ=4; ACCUM=1; SEED=42
RESUME=""; RESUME_EPOCH=""; NAME=""; EXTRA_TAGS=""; DRYRUN=0
PASSTHRU=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)         GPUS="$2"; shift 2;;
        --density)      WITH_DENSITY=true; shift;;
        --no-density)   WITH_DENSITY=false; shift;;
        --encoder)      ENCODER="$2"; shift 2;;
        --encoder-ckpt) ENC_CKPT="$2"; shift 2;;
        --no-freeze)    FREEZE=false; shift;;
        --subset)       SUBSET="$2"; shift 2;;
        --data)         DATA_VER="$2"; shift 2;;
        --epochs)       EPOCHS="$2"; shift 2;;
        --bsz)          BSZ="$2"; shift 2;;
        --accum)        ACCUM="$2"; shift 2;;
        --seed)         SEED="$2"; shift 2;;
        --resume)       RESUME="@output"; shift;;
        --resume-from)  RESUME="$2"; shift 2;;
        --resume-epoch) RESUME_EPOCH="$2"; shift 2;;
        --name)         NAME="$2"; shift 2;;
        --tags)         EXTRA_TAGS="$2"; shift 2;;
        --dry-run)      DRYRUN=1; shift;;
        --)             shift; PASSTHRU=("$@"); break;;
        -h|--help)      awk 'NR==1{next} /^#/{print;next} {exit}' "$0"; exit 0;;
        *)              die "unknown arg: $1  (see --help)";;
    esac
done

[[ -n "$GPUS" ]] || die "--gpus is required (e.g. 1-6)"
case "$ENCODER" in none|cnn|vit) ;; *) die "--encoder must be none|cnn|vit";; esac
[[ "$ENCODER" != none && -z "$ENC_CKPT" ]] && die "--encoder $ENCODER requires --encoder-ckpt PATH"
[[ "$WITH_DENSITY" == false && "$ENCODER" != none ]] && die "--encoder $ENCODER needs --density (an encoder is meaningless for a --no-density baseline)"

# ── GPU spec → CUDA list + nproc ──────────────────────────────────────────────
expand_gpus(){
    local spec="$1" out="" p lo hi i
    IFS=',' read -ra parts <<< "$spec"
    for p in "${parts[@]}"; do
        if [[ "$p" == *-* ]]; then lo=${p%-*}; hi=${p#*-}; for ((i=lo;i<=hi;i++)); do out+="${i},"; done
        else out+="${p},"; fi
    done
    echo "${out%,}"
}
CUDA_LIST=$(expand_gpus "$GPUS")
NPROC=$(awk -F, '{print NF}' <<< "$CUDA_LIST")
[[ "$NPROC" -ge 1 ]] || die "could not parse --gpus '$GPUS'"

# ── Data version → crops_dir / normalize ──────────────────────────────────────
# v1 = pretrain/xray_crops_aligned (normalize default true); vN (N≥2) = pre-normalised
# pretrain/xray_crops_aligned_vN → normalize=false. Generic so v4, v5, … need no edit.
if [[ "$DATA_VER" == v1 ]]; then
    CROPS=$DATA/pretrain/xray_crops_aligned;          NORMALIZE="";    DV_TAG=""
elif [[ "$DATA_VER" =~ ^v[0-9]+$ ]]; then
    CROPS=$DATA/pretrain/xray_crops_aligned_$DATA_VER; NORMALIZE=false; DV_TAG=$DATA_VER
else
    die "unknown --data '$DATA_VER' (expected v1, v2, v3, …)"
fi

# ── exp_name (auto unless --name) ─────────────────────────────────────────────
SUBSET_TAG=$(( SUBSET / 1000 ))k
if [[ -z "$NAME" ]]; then
    NAME="$(date +%y%m%d)_voxbind_${SUBSET_TAG}"
    [[ "$WITH_DENSITY" == true ]] && NAME="${NAME}_density" || NAME="${NAME}_baseline"
    [[ "$ENCODER" != none ]] && NAME="${NAME}_${ENCODER}"
    [[ "$ENCODER" != none && "$FREEZE" == true ]] && NAME="${NAME}_frozen"
    [[ -n "$DV_TAG" ]] && NAME="${NAME}_${DV_TAG}"
fi
EXP="$NAME"

# ── wandb tags ────────────────────────────────────────────────────────────────
TAGS="voxbind,${SUBSET_TAG}"
[[ "$WITH_DENSITY" == true ]] && TAGS="${TAGS},density" || TAGS="${TAGS},baseline"
if [[ "$ENCODER" != none ]]; then
    TAGS="${TAGS},${ENCODER}"
    [[ "$FREEZE" == true ]] && TAGS="${TAGS},frozen_encoder"
fi
[[ -n "$DV_TAG" ]] && TAGS="${TAGS},${DV_TAG}"
TAGS="${TAGS},xray"
[[ -n "$EXTRA_TAGS" ]] && TAGS="${TAGS},${EXTRA_TAGS}"

# ── use_xray: always true. The dataset is the x-ray subset regardless; only the
# MODEL (model.with_density) changes for a baseline, so the baseline still loads
# the (unused) density crops — matching the archived 08 baseline (cfg use_xray=true).
USE_XRAY=true

# ── Assemble command ──────────────────────────────────────────────────────────
CMD=( "$PY/torchrun" --standalone --nproc_per_node="$NPROC" train_ddp.py
      dset=crossdocked_xray
      dset.data_dir="$DATA"
      dset.crops_dir="$CROPS"
      dset.subset_n="$SUBSET" dset.subset_xray_only=true dset.subset_val_n=100
      dset.use_xray="$USE_XRAY"
      num_epochs="$EPOCHS" bsz="$BSZ" accum_steps="$ACCUM" seed="$SEED"
      wjs.n_targets=0
      model.with_density="$WITH_DENSITY" )
[[ -n "$NORMALIZE" ]] && CMD+=( dset.normalize="$NORMALIZE" )
case "$ENCODER" in
    cnn) CMD+=( model.density_encoder_blocks=4 model.density_pretrained_path="$ENC_CKPT" model.density_freeze="$FREEZE" );;
    vit) CMD+=( model.density_encoder_type=vit model.density_pretrained_path="$ENC_CKPT" model.density_freeze="$FREEZE" );;
esac
CMD+=( "wandb_tags=[${TAGS}]" exp_name="$EXP" output_dir="$VOX/exps/$EXP" )
if [[ -n "$RESUME" ]]; then
    [[ "$RESUME" == "@output" ]] && RESUME="$VOX/exps/$EXP"
    CMD+=( resume="$RESUME" )
    [[ -n "$RESUME_EPOCH" ]] && CMD+=( resume_epoch="$RESUME_EPOCH" )
fi
[[ ${#PASSTHRU[@]} -gt 0 ]] && CMD+=( "${PASSTHRU[@]}" )

# ── Report / run ──────────────────────────────────────────────────────────────
echo "[$(ts)] downstream  density=$WITH_DENSITY  encoder=$ENCODER  freeze=$FREEZE  subset=$SUBSET  data=$DATA_VER"
echo "          GPUs=$CUDA_LIST  nproc=$NPROC  bsz=$BSZ  accum=$ACCUM  epochs=$EPOCHS  eff_batch=$((BSZ*NPROC*ACCUM))"
echo "          exp=$EXP${RESUME:+  resume=$RESUME${RESUME_EPOCH:+ @ep$RESUME_EPOCH}}"
if [[ "$DRYRUN" -eq 1 ]]; then
    echo "CUDA_VISIBLE_DEVICES=$CUDA_LIST \\"
    printf '  %q' "${CMD[@]}"; echo
    exit 0
fi

mkdir -p "$LOG"
cd "$VOX" || die "cd $VOX failed"
echo "[$(ts)] launching $EXP  ->  log: $LOG/${EXP}.log"
CUDA_VISIBLE_DEVICES="$CUDA_LIST" "${CMD[@]}" >> "$LOG/${EXP}.log" 2>&1
echo "[$(ts)] $EXP done (exit $?)  ->  exps/$EXP"
