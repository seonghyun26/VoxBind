#!/usr/bin/env bash
set -u

VOX="${VOXBIND_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${VOXBIND_PY:-/home/shpark/.conda/envs/voxbind/bin}"

CFG="config_train_roleblob_diverse_density_gradmag_channelvit_mae_40m_plinder_otf_mask050"
EXP="${EXP:-260626_plinder_v2_roleblob_diverse_cdg_channelvit_mask050_pretrain}"
GPUS="${GPUS:-0,1,2,3}"
NPROC="${NPROC:-4}"
RDZV_PORT="${RDZV_PORT:-29566}"
SUBSET_N="${SUBSET_N:-112633}"
SUBSET_VAL_N="${SUBSET_VAL_N:-100}"

LOG="$VOX/log/${EXP}.log"
mkdir -p "$VOX/log"

ts() { date "+%Y-%m-%d %H:%M:%S"; }

{
    echo "[$(ts)] start: $EXP"
    echo "[$(ts)] gpus=$GPUS nproc=$NPROC rdzv_port=$RDZV_PORT subset_n=$SUBSET_N subset_val_n=$SUBSET_VAL_N"
    cd "$VOX" || exit 1

    CMD=(
        "$PY/torchrun"
        --nnodes=1
        --nproc_per_node="$NPROC"
        --rdzv-backend=c10d
        --rdzv-endpoint="localhost:${RDZV_PORT}"
        --rdzv-id=plinderv2div
        train_density.py
        --config-name="$CFG"
        "exp_name=$EXP"
        "dset.resample_dir=dataset/data/pretrain/xray_resample_plinder_v2"
        "dset.data_file=pretrain/data_train_plinder_v2.pt"
        "dset.subset_n=$SUBSET_N"
        "dset.subset_val_n=$SUBSET_VAL_N"
        "wandb_tags=[pretrain,roleblob,diverse,v2,density_gradmag,40m,plinder,otf,ligvdw,mask050,uniform,cdg,channelvit,rolesplit,archcompare]"
    )

    CUDA_VISIBLE_DEVICES="$GPUS" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${CMD[@]}"
    rc=$?
    echo "[$(ts)] exit: $EXP rc=$rc"
    exit "$rc"
} >> "$LOG" 2>&1
