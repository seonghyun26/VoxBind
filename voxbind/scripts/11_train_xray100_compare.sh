#!/bin/bash
# Train two VoxBind models from scratch on the same 100 training samples to
# isolate the effect of X-ray density conditioning. Uses DistributedDataParallel
# via torchrun (one process per GPU).
#
#   Arm A (with density)    : model.with_density=True  → exps/xray100_density/
#   Arm B (without density) : model.with_density=False → exps/xray100_nodensity/
#
# Both arms use:
#   - dset=crossdocked_xray with subset_n=100, subset_xray_only=True
#     (same 100 PDBs with a CCP4 map in both arms)
#   - DDP across NPROC_PER_NODE GPUs (default 4). cfg.bsz is PER-RANK in DDP,
#     so world batch = bsz * NPROC_PER_NODE.
#   - num_epochs=350, num_workers=4, seed=42
#   - wjs.n_targets=0 (skip WJS sampling — sampling pipeline isn't density-aware yet)
#
# Usage:
#   bash scripts/11_train_xray100_compare.sh [parallel|seq|with|without]
#     parallel (default) — both arms simultaneously, NPROC GPUs each
#     seq                — with-density first, then without (one at a time)
#     with               — only the with-density arm
#     without            — only the without-density arm
#
# Env vars:
#   GPUS_WITH        GPUs for the with-density arm     (default: 0,1,2,3)
#   GPUS_WITHOUT     GPUs for the without-density arm  (default: 4,5,6,7)
#   NPROC_PER_NODE   number of DDP ranks per arm       (default: 4)
#   EPOCHS           training epochs                   (default: 350)
#   BSZ              per-rank batch size               (default: 16)
#                    world batch = BSZ * NPROC_PER_NODE
#   SEED             random seed                       (default: 42)
#   MASTER_PORT_A    rendezvous port for arm A         (default: 29500)
#   MASTER_PORT_B    rendezvous port for arm B         (default: 29501)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOXBIND_DIR="$(dirname "$SCRIPT_DIR")"   # .../VoxBind/voxbind  (where train.py lives)
REPO_ROOT="$(dirname "$VOXBIND_DIR")"    # .../VoxBind            (where dataset/data/ lives)
cd "$VOXBIND_DIR"

MODE="${1:-parallel}"
GPUS_WITH="${GPUS_WITH:-0,1,2,3}"
GPUS_WITHOUT="${GPUS_WITHOUT:-4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
EPOCHS="${EPOCHS:-350}"
BSZ="${BSZ:-4}"
SEED="${SEED:-42}"
MASTER_PORT_A="${MASTER_PORT_A:-29500}"
MASTER_PORT_B="${MASTER_PORT_B:-29501}"

# Data lives at the repo root, not under voxbind/. Use absolute paths so
# both the prereq check and the dataset class agree regardless of CWD.
DATA_DIR="${REPO_ROOT}/dataset/data"
CROPS_DIR="${DATA_DIR}/xray_crops"
if [ ! -f "${DATA_DIR}/data_train.pt" ]; then
    echo "ERROR: missing ${DATA_DIR}/data_train.pt — run 01_preprocess.sh or pull from svr12"
    exit 1
fi
if [ ! -d "${CROPS_DIR}/train" ]; then
    echo "ERROR: missing ${CROPS_DIR}/train — run 07_process_xray.sh"
    exit 1
fi

COMMON_OVERRIDES=(
    "dset=crossdocked_xray"
    "dset.data_dir=${DATA_DIR}"
    "dset.crops_dir=${CROPS_DIR}"
    "dset.subset_n=100"
    "dset.subset_xray_only=true"
    "dset.use_xray=true"
    "num_epochs=${EPOCHS}"
    "bsz=${BSZ}"
    "seed=${SEED}"
    "wjs.n_targets=0"
)

run_arm() {
    local arm="$1"            # "density" or "nodensity"
    local with_density="$2"   # "true" or "false"
    local gpus="$3"
    local port="$4"
    local exp_name="xray100_${arm}"
    local out_dir="exps/${exp_name}"

    echo "==> [${arm}] with_density=${with_density}  GPUs=${gpus}  nproc=${NPROC_PER_NODE}  port=${port}  out=${out_dir}"
    CUDA_VISIBLE_DEVICES="$gpus" torchrun \
        --standalone \
        --nproc_per_node="$NPROC_PER_NODE" \
        --rdzv-endpoint="localhost:${port}" \
        train.py \
        "${COMMON_OVERRIDES[@]}" \
        "model.with_density=${with_density}" \
        "exp_name=${exp_name}" \
        "output_dir=${out_dir}"
}

case "$MODE" in
    parallel)
        echo "==> Launching both arms in parallel (DDP, ${NPROC_PER_NODE} GPUs each)"
        run_arm density   true  "$GPUS_WITH"    "$MASTER_PORT_A" > exps_xray100_density.log   2>&1 &
        PID_WITH=$!
        run_arm nodensity false "$GPUS_WITHOUT" "$MASTER_PORT_B" > exps_xray100_nodensity.log 2>&1 &
        PID_WITHOUT=$!
        echo "    with-density   : PID $PID_WITH    (log: exps_xray100_density.log)"
        echo "    without-density: PID $PID_WITHOUT (log: exps_xray100_nodensity.log)"
        wait $PID_WITH    && echo "==> with-density arm done"
        wait $PID_WITHOUT && echo "==> without-density arm done"
        ;;
    seq)
        run_arm density   true  "$GPUS_WITH" "$MASTER_PORT_A"
        run_arm nodensity false "$GPUS_WITH" "$MASTER_PORT_A"
        ;;
    with)
        run_arm density   true  "$GPUS_WITH" "$MASTER_PORT_A"
        ;;
    without)
        run_arm nodensity false "$GPUS_WITHOUT" "$MASTER_PORT_B"
        ;;
    *)
        echo "ERROR: unknown mode '${MODE}'. Use: parallel | seq | with | without"
        exit 1
        ;;
esac

echo "==> Both runs complete. Compare metrics in W&B (project=voxbind)."
