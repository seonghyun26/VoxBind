#!/usr/bin/env bash
# 03_train.sh — train the density-conditioned VoxBind generative model.
#
# The headline model ("VoxBind + density") is a walk-jump voxel denoiser whose
# pocket branch is conditioned on a FROZEN electron-density encoder (CDG_v2).
# Producing it takes two trained pieces:
#
#   1. base denoiser  — vanilla VoxBind, the warm-start weights (MODE=base)
#   2. density fusion — base denoiser + frozen CDG_v2 encoder, token fusion (MODE=fusion)
#
# The frozen CDG_v2 encoder itself ships in the prepared copy (01); rebuilding it
# is optional and heavy (MODE=encoder, needs the PLINDER density corpus).
#
#   bash script/03_train.sh                       # MODE=fusion (default): density-conditioned model
#   MODE=base   bash script/03_train.sh           # train the vanilla base denoiser (warm start)
#   MODE=all    bash script/03_train.sh           # base, then fusion
#   MODE=encoder bash script/03_train.sh          # (advanced) re-pretrain the CDG_v2 encoder
#
# GPUs / batch (per-rank under DDP; effective batch = BSZ * n_gpus):
#   GPUS=0,1,2,3 BSZ=8 NUM_EPOCHS=100 bash script/03_train.sh
#
# Key overrides: MODE, GPUS, BSZ, NUM_EPOCHS, SIGMA, EXP_NAME, ENCODER,
#                WARM_START, CROPS_DIR, MASTER_PORT, WANDB.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

MODE="${MODE:-fusion}"
BSZ="${BSZ:-8}"                 # per-rank; effective = BSZ * n_gpus
SIGMA="${SIGMA:-0.9}"
WANDB="${WANDB:-false}"         # off by default for a fresh checkout; set true after `wandb login`
MASTER_PORT="${MASTER_PORT:-29531}"
CROPS_DIR="${CROPS_DIR:-$DATA_ROOT/pretrain/xray_crops_aligned_v5}"
ENCODER="${ENCODER:-$CODE_ROOT/model_zoo/CDG_v2/checkpoint_e0025.pth.tar}"
BASE_EXP="${BASE_EXP:-voxbind_base}"
FUSION_EXP="${FUSION_EXP:-voxbind_density_fusion}"

require_env "$VOXBIND_ENV"
export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TORCHDYNAMO_DISABLE=1
mkdir -p "$CODE_ROOT/logs"

torchrun_in_code() {  # torchrun_in_code <extra torchrun+train_ddp args...>
    ( cd "$CODE_ROOT" && conda_run "$VOXBIND_ENV" \
        torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" train_ddp.py "$@" )
}

# --------------------------------------------------------------------------
# MODE=base — vanilla VoxBind denoiser (ICML'24 recipe), from scratch.
# --------------------------------------------------------------------------
train_base() {
    require_data "data_train.pt" "run 02_preprocess_data.sh first"
    local exp="${EXP_NAME:-$BASE_EXP}" epochs="${NUM_EPOCHS:-350}"
    banner "MODE=base  exp=$exp  gpus=$GPUS (nproc=$NPROC)  bsz=$BSZ (eff $((BSZ*NPROC)))  epochs=$epochs"
    log "log: $CODE_ROOT/logs/${exp}.log"
    torchrun_in_code \
        bsz="$BSZ" num_epochs="$epochs" num_workers="${NUM_WORKERS:-8}" \
        ddp_static_graph=true wjs.n_targets=0 smooth_sigma="$SIGMA" \
        exp_name="$exp" wandb="$WANDB" \
        wandb_tags="[voxbind,base,crossdocked,sig${SIGMA}]" \
        >> "$CODE_ROOT/logs/${exp}.log" 2>&1 \
        || die "base training failed (see logs/${exp}.log)"
    log "base done -> $CODE_ROOT/exps/$exp/checkpoint.pth.tar"
}

# --------------------------------------------------------------------------
# MODE=fusion — density-conditioned model: base warm start + FROZEN CDG_v2.
# --------------------------------------------------------------------------
train_fusion() {
    local exp="${EXP_NAME:-$FUSION_EXP}" epochs="${NUM_EPOCHS:-100}"
    local warm="${WARM_START:-$CODE_ROOT/exps/$BASE_EXP/checkpoint.pth.tar}"

    [ -f "$ENCODER" ]        || die "frozen encoder not found: $ENCODER (rerun 01_download_data.sh)"
    [ -d "$CROPS_DIR/train" ] || die "density crops not found: $CROPS_DIR/train (rerun 01/02)"
    [ -f "$warm" ] || die "base warm-start checkpoint not found: $warm — run 'MODE=base bash script/03_train.sh' first (or set WARM_START)"

    # Encoder geometry MUST be read from the encoder's own cfg.yaml — model_zoo
    # entries are not interchangeable at fixed dims (channel layout differs by family).
    local enc_cfg; enc_cfg="$(dirname "$ENCODER")/cfg.yaml"
    [ -f "$enc_cfg" ] || die "missing encoder cfg $enc_cfg"
    eval "$(conda_run "$VOXBIND_ENV" python - "$enc_cfg" <<'PY'
import sys, yaml
m = (yaml.safe_load(open(sys.argv[1])) or {}); m = m.get("model", m)
g = [int(x) for x in (m.get("channel_groups") or [7, 4, 2])]
print(f'VIT_PATCH={int(m.get("patch_size",8))}')
print(f'VIT_DIM={int(m["dim"])}')
print(f'VIT_DEPTH={int(m["depth"])}')
print(f'VIT_HEADS={int(m["heads"])}')
print(f'VIT_MLP_RATIO={int(m.get("mlp_ratio",4))}')
print(f'VIT_DROPOUT={float(m.get("dropout",0.1))}')
print(f'VIT_NCH={int(m.get("n_in_channels",13))}')
print("VIT_GROUPS='[%s]'" % ",".join(str(x) for x in g))
PY
)"
    [ -n "${VIT_DIM:-}" ] || die "could not read geometry from $enc_cfg"

    # SUBSET_N = usable x-ray-available train rows minus a 100-row val holdout.
    local subset_val_n="${SUBSET_VAL_N:-100}"
    local subset_n="${SUBSET_N:-}"
    if [ -z "$subset_n" ]; then
        subset_n="$(conda_run "$VOXBIND_ENV" python - "$CROPS_DIR" "$subset_val_n" <<'PY'
import sys, numpy as np
n = int(np.load(f"{sys.argv[1]}/train_available.npy").sum())
print(max(n - int(sys.argv[2]), 1))
PY
)"
    fi
    [ -n "$subset_n" ] || die "could not derive SUBSET_N"

    banner "MODE=fusion  exp=$exp  gpus=$GPUS (nproc=$NPROC)  bsz=$BSZ (eff $((BSZ*NPROC)))  epochs=$epochs"
    log "encoder (frozen): $ENCODER  dim=$VIT_DIM depth=$VIT_DEPTH heads=$VIT_HEADS groups=$VIT_GROUPS"
    log "warm start:       $warm"
    log "crops:            $CROPS_DIR  subset=${subset_n}+${subset_val_n}"
    log "log:              $CODE_ROOT/logs/${exp}.log"

    torchrun_in_code \
        --config-name config_train_voxbind_fusion_champion_reference \
        wandb="$WANDB" num_workers="${NUM_WORKERS:-12}" prefetch_factor=8 \
        exp_name="$exp" num_epochs="$epochs" bsz="$BSZ" accum_steps=1 \
        lr="${LR:-1e-5}" wd="${WD:-1e-2}" smooth_sigma="$SIGMA" \
        ++ddp_static_graph=false \
        dset.crops_dir="$CROPS_DIR" dset.normalize=false \
        dset.pocket_radius=-1 dset.ligand_radius=0.5 \
        dset.use_xray=true dset.subset_xray_only=true \
        dset.subset_n="$subset_n" dset.subset_val_n="$subset_val_n" dset.cache_size=32 \
        model.with_density=true model.density_encoder_type=vit model.density_freeze=true \
        model.density_pretrained_path="$ENCODER" \
        model.density_vit.patch="$VIT_PATCH" model.density_vit.dim="$VIT_DIM" \
        model.density_vit.depth="$VIT_DEPTH" model.density_vit.heads="$VIT_HEADS" \
        model.density_vit.mlp_ratio="$VIT_MLP_RATIO" model.density_vit.dropout="$VIT_DROPOUT" \
        model.density_vit.n_in_channels="$VIT_NCH" \
        model.density_vit.patch_embed_mode=channel_group \
        model.density_vit.channel_groups="$VIT_GROUPS" \
        model.density_encoder_sees_ligand=true model.density_mask_ligand=false \
        model.fusion="${FUSION:-v4}" model.density_encoder_amp=true \
        ++model.density_cond_dropout="${COND_DROPOUT:-0}" \
        wjs.n_targets=0 \
        pretrained_path="$warm" \
        wandb_tags="[voxbind,fusion,${FUSION:-v4},cdg_v2,sig${SIGMA}]" \
        >> "$CODE_ROOT/logs/${exp}.log" 2>&1 \
        || die "fusion training failed (see logs/${exp}.log)"
    log "fusion done -> $CODE_ROOT/exps/$exp/checkpoint.pth.tar"
    log "sample it with:  EXP=$exp bash script/04_sample.sh"
}

# --------------------------------------------------------------------------
# MODE=encoder — (advanced) re-pretrain the CDG_v2 density encoder via MAE.
# Needs the PLINDER density corpus; the prepared copy already ships CDG_v2,
# so almost everyone skips this.
# --------------------------------------------------------------------------
train_encoder() {
    local cfg="${ENCODER_CONFIG:-config_train_cdg_mfodfc_channelvit_100m_v2_mask075}"
    local exp="${EXP_NAME:-cdg_encoder_repro}" epochs="${NUM_EPOCHS:-100}"
    banner "MODE=encoder (advanced)  config=$cfg  exp=$exp  gpus=$GPUS"
    log "requires the PLINDER density corpus (pretrain/data_train_plinder*.pt + xray_crops_aligned_v5)"
    log "log: $CODE_ROOT/logs/${exp}.log"
    ( cd "$CODE_ROOT" && conda_run "$VOXBIND_ENV" \
        torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" train_density.py \
        --config-name "$cfg" exp_name="$exp" num_epochs="$epochs" wandb="$WANDB" ) \
        >> "$CODE_ROOT/logs/${exp}.log" 2>&1 \
        || die "encoder pretrain failed (see logs/${exp}.log)"
    log "encoder done -> $CODE_ROOT/exps/$exp/  (point ENCODER=... at its checkpoint for MODE=fusion)"
}

case "$MODE" in
    base)    train_base ;;
    fusion)  train_fusion ;;
    all)     train_base; train_fusion ;;
    encoder) train_encoder ;;
    *) die "unknown MODE '$MODE' (use: fusion | base | all | encoder)" ;;
esac
