#!/bin/bash
# pretrain.sh — unified launcher for ViT-MAE encoder pre-training
# (train_density_vit_mae.py). Replaces the per-variant scripts 31–37 with one
# parameterized entry point: pick the input modality, data version, weighting
# recipe and GPU set; the config-name, channel count, use_xray flag, accum_steps,
# exp name and wandb tags are all derived.
#
# ── Usage ─────────────────────────────────────────────────────────────────────
#   bash scripts/pretrain.sh --mode MODE --gpus GPUS [options]
#
# Required:
#   --mode MODE     density | atomblob | atomblob_density | atomblob_merged_density
#   --gpus GPUS     comma list or dash range — "1,2,3,4,5" or "4-7" or "4,5"
#
# Options:
#   --data VER      v1  xray_crops_aligned     (per-crop ±3σ z-score)   [default]
#                   v2  xray_crops_aligned_v2  (pool z-score, no clip; normalize=false)
#                   v3  xray_crops_aligned_v3  (pool max-abs→[−1,1];    normalize=false)
#                   vN  xray_crops_aligned_vN  (any N≥2 → normalize=false; e.g. v4, v5)
#   --weighted      weighted recipe (atom_biased mask + inv_sqrt_freq
#                   [+ density downweight]). Required for atomblob_merged_density;
#                   not available for density (ignored with a warning).
#   --gradmag       append a gradient-magnitude ‖∇ρ‖ channel to the density input
#                   (density-bearing modes only). Sets with_gradmag=true, bumps
#                   n_in by 1, and makes gradmag an MAE reconstruction target
#                   (gradmag_reconstruct=true → recon width = n_in). Tags +gradmag;
#                   auto-name +_gradmag.
#   --epochs N      num_epochs              (default 100)
#   --bsz N         per-GPU batch size      (default 8)
#   --target N      target effective batch for accum auto-calc (default 80)
#   --accum N       force accum_steps (overrides the auto-calc)
#   --lr V          (default 1e-4)
#   --wd V          (default 5e-2)
#   --seed N        (default 42)
#   --name NAME     exp_name / output dir   (default: <yymmdd>_<auto>_pretrain)
#   --tags a,b,c    extra wandb tags appended to the auto tags
#   --dry-run       print the resolved command and exit (no training)
#   -- ARG ...      everything after a bare "--" is passed to Hydra verbatim
#                   (e.g. -- resume=exps/foo resume_epoch=100)
#
# ── Reproductions (core training args verified byte-identical) ────────────────
#   31 = --mode density                            --gpus 1,2,3,4,5
#   32 = --mode atomblob                           --gpus 1,2,3,4,5
#   33 = --mode atomblob_density                   --gpus 1,2,3,4,5
#   34 = --mode atomblob            --weighted     --gpus 1,2,3,4,5
#   35 = --mode atomblob_density    --weighted     --gpus 4,5
#   36 = --mode atomblob_merged_density --weighted --gpus 4,5,6,7
#   37 = --mode atomblob_merged_density --weighted --data v2 --gpus 4,5,6,7
set -u

VOX=/home/shpark/prj-denovo/VoxBind/voxbind
PY=/home/shpark/.conda/envs/voxbind/bin
DATA=$VOX/dataset/data
LOG=$VOX/log
ts(){ date "+%Y-%m-%d %H:%M:%S"; }
die(){ echo "pretrain.sh: $*" >&2; exit 1; }

# ── Defaults ──────────────────────────────────────────────────────────────────
MODE=""; GPUS=""; DATA_VER="v1"; WEIGHTED=0; GRADMAG=0
EPOCHS=100; BSZ=8; TARGET=80; ACCUM=""; LR=1e-4; WD=5e-2; SEED=42
NAME=""; EXTRA_TAGS=""; DRYRUN=0
PASSTHRU=()

# ── Arg parse ─────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)     MODE="$2"; shift 2;;
        --gpus)     GPUS="$2"; shift 2;;
        --data)     DATA_VER="$2"; shift 2;;
        --weighted) WEIGHTED=1; shift;;
        --gradmag)  GRADMAG=1; shift;;
        --epochs)   EPOCHS="$2"; shift 2;;
        --bsz)      BSZ="$2"; shift 2;;
        --target)   TARGET="$2"; shift 2;;
        --accum)    ACCUM="$2"; shift 2;;
        --lr)       LR="$2"; shift 2;;
        --wd)       WD="$2"; shift 2;;
        --seed)     SEED="$2"; shift 2;;
        --name)     NAME="$2"; shift 2;;
        --tags)     EXTRA_TAGS="$2"; shift 2;;
        --dry-run)  DRYRUN=1; shift;;
        --)         shift; PASSTHRU=("$@"); break;;
        -h|--help)  awk 'NR==1{next} /^#/{print;next} {exit}' "$0"; exit 0;;
        *)          die "unknown arg: $1  (see --help)";;
    esac
done

[[ -n "$MODE" ]] || die "--mode is required (density|atomblob|atomblob_density|atomblob_merged_density)"
[[ -n "$GPUS" ]] || die "--gpus is required (e.g. 1,2,3,4,5 or 4-7)"

# ── GPU spec → CUDA list + nproc ──────────────────────────────────────────────
expand_gpus(){
    local spec="$1" out="" p lo hi i
    IFS=',' read -ra parts <<< "$spec"
    for p in "${parts[@]}"; do
        if [[ "$p" == *-* ]]; then
            lo=${p%-*}; hi=${p#*-}
            for ((i=lo; i<=hi; i++)); do out+="${i},"; done
        else
            out+="${p},"
        fi
    done
    echo "${out%,}"
}
CUDA_LIST=$(expand_gpus "$GPUS")
NPROC=$(awk -F, '{print NF}' <<< "$CUDA_LIST")
[[ "$NPROC" -ge 1 ]] || die "could not parse --gpus '$GPUS'"

# ── accum_steps: ceil(target / (bsz*nproc)) unless forced ─────────────────────
PERSTEP=$((BSZ * NPROC))
if [[ -z "$ACCUM" ]]; then
    ACCUM=$(( (TARGET + PERSTEP - 1) / PERSTEP ))   # ceil division
fi
EFF=$((BSZ * NPROC * ACCUM))

# ── Mode → config / input_mode / use_xray / channels ──────────────────────────
# Shell-only: each (mode, weighted) maps to an existing config_train_*.yaml; the
# data version, channels, use_xray and compute knobs ride in as CLI overrides.
INPUT_MODE=""; USE_XRAY=""; N_IN=""; W_TAG=""
case "$MODE" in
    density)
        CFG=config_train_density_vit_mae_40m_xray
        USE_XRAY=true
        # this config has no top-level input_mode key (default 'density'); leave unset
        [[ "$WEIGHTED" -eq 1 ]] && echo "[warn] --weighted has no effect for --mode density (single channel); ignoring" >&2
        ;;
    atomblob)
        CFG=config_train_atomblob_vit_mae_40m
        INPUT_MODE=atomblob; USE_XRAY=false
        if [[ "$WEIGHTED" -eq 1 ]]; then CFG=config_train_atomblob_vit_mae_40m_weighted; W_TAG=weighted; fi
        ;;
    atomblob_density)
        CFG=config_train_atomblob_density_vit_mae_40m
        INPUT_MODE=atomblob_density; USE_XRAY=true
        if [[ "$WEIGHTED" -eq 1 ]]; then CFG=config_train_atomblob_density_vit_mae_40m_weighted; W_TAG=weighted; fi
        ;;
    atomblob_merged_density)
        [[ "$WEIGHTED" -eq 1 ]] || die "--mode atomblob_merged_density only has a weighted config; add --weighted"
        CFG=config_train_atomblob_merged_density_vit_mae_40m_weighted
        INPUT_MODE=atomblob_merged_density; USE_XRAY=true; N_IN=8; W_TAG=weighted
        ;;
    *)
        die "unknown --mode '$MODE'";;
esac

# ── Gradmag: append ‖∇ρ‖ channel (density-bearing modes only) ──────────────────
# Overrides n_in to include the trailing gradmag channel; with_gradmag and
# gradmag_reconstruct ride in as ++ overrides in the command assembly below.
# Per-mode n_in = (base atoms + density) + 1: density 1→2, atomblob_density
# 12→13, atomblob_merged_density 8→9.
if [[ "$GRADMAG" -eq 1 ]]; then
    case "$MODE" in
        density)                 N_IN=2;;
        atomblob_density)        N_IN=13;;
        atomblob_merged_density) N_IN=9;;
        *) die "--gradmag requires a density-bearing --mode (density|atomblob_density|atomblob_merged_density); got '$MODE'";;
    esac
fi

# ── Data version → crops_dir / normalize ──────────────────────────────────────
# v1 = xray_crops_aligned (per-crop ±3σ z-score; dset.normalize default true).
# vN (N≥2) = xray_crops_aligned_vN, pre-normalised by 00b_density_preprocess.py → normalize=false.
# Generic so new versions (v4, v5, …) need no script edit.
if [[ "$DATA_VER" == v1 ]]; then
    CROPS=$DATA/xray_crops_aligned;          NORMALIZE="";    DV_TAG=""
elif [[ "$DATA_VER" =~ ^v[0-9]+$ ]]; then
    CROPS=$DATA/xray_crops_aligned_$DATA_VER; NORMALIZE=false; DV_TAG=$DATA_VER
else
    die "unknown --data '$DATA_VER' (expected v1, v2, v3, …)"
fi

# ── exp_name (auto unless --name) ─────────────────────────────────────────────
if [[ -z "$NAME" ]]; then
    namebase="$MODE"
    [[ "$MODE" == density ]] && namebase="density_xray"
    NAME="$(date +%y%m%d)_${namebase}_vit_mae_40m"
    [[ -n "$W_TAG"  ]] && NAME="${NAME}_weighted"
    [[ -n "$DV_TAG" ]] && NAME="${NAME}_${DV_TAG}"
    [[ "$GRADMAG" -eq 1 ]] && NAME="${NAME}_gradmag"
    NAME="${NAME}_pretrain"
fi
EXP="$NAME"

# ── wandb tags ────────────────────────────────────────────────────────────────
TAGS="pretrain,${MODE},40m"
[[ -n "$W_TAG"  ]] && TAGS="${TAGS},weighted"
[[ -n "$DV_TAG" ]] && TAGS="${TAGS},${DV_TAG}"
[[ "$GRADMAG" -eq 1 ]] && TAGS="${TAGS},gradmag"
TAGS="${TAGS},crossdocked_xray"
[[ -n "$EXTRA_TAGS" ]] && TAGS="${TAGS},${EXTRA_TAGS}"

# ── Assemble command ──────────────────────────────────────────────────────────
CMD=( "$PY/torchrun" --standalone --nproc_per_node="$NPROC" train_density_vit_mae.py
      --config-name="$CFG"
      dset=crossdocked_xray
      dset.data_dir="$DATA"
      dset.crops_dir="$CROPS"
      dset.subset_xray_only=true
      dset.subset_n=78428
      dset.subset_val_n=100
      dset.use_xray="$USE_XRAY" )
[[ -n "$NORMALIZE"  ]] && CMD+=( dset.normalize="$NORMALIZE" )
[[ -n "$INPUT_MODE" ]] && CMD+=( input_mode="$INPUT_MODE" )
[[ -n "$N_IN"       ]] && CMD+=( ++model.n_in_channels="$N_IN" )
[[ "$GRADMAG" -eq 1 ]] && CMD+=( ++with_gradmag=true ++mae.gradmag_reconstruct=true )
CMD+=( num_epochs="$EPOCHS" bsz="$BSZ" accum_steps="$ACCUM"
       "wandb_tags=[${TAGS}]"
       lr="$LR" wd="$WD" seed="$SEED"
       exp_name="$EXP" output_dir="$VOX/exps/$EXP" )
[[ ${#PASSTHRU[@]} -gt 0 ]] && CMD+=( "${PASSTHRU[@]}" )

# ── Report / run ──────────────────────────────────────────────────────────────
echo "[$(ts)] pretrain  mode=$MODE  data=$DATA_VER  weighted=$WEIGHTED  gradmag=$GRADMAG  config=$CFG"
echo "          GPUs=$CUDA_LIST  nproc=$NPROC  bsz=$BSZ  accum=$ACCUM  eff_batch=$EFF"
echo "          exp=$EXP"
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
