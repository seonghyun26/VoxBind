#!/bin/bash
# Fine-tune VoxBind on N test-set samples — two conditions:
#   (i)   no-density      : original voxel data only  → exps/poc_xray/finetune_test/
#   (ii)  density         : voxel + X-ray density     → exps/poc_xray/overfit_test/
#   (iii) random density  : voxel + random N(0,1)     → exps/poc_xray/random_test/
#
# Usage:
#   bash scripts/08c_density_overfit_test.sh [mode]
#
# Arguments (optional):
#   mode    ccp4 | crops  (default: crops)
#
# Environment variables (optional):
#   CUDA_VISIBLE_DEVICES   GPUs to use (default: 7)
#   N_SAMPLES              number of test samples to overfit on (default: 20)
#   ITERS                  training iterations per condition (default: 2000)
#   LR                     LR for density_encoder+density_proj (default: 1e-3)
#   LR_BACKBONE            LR for pretrained backbone (default: 1e-4)
#   NORMALIZE              0 to skip per-sample z-score normalization (default: 1)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

MODE="${1:-crops}"
GPUS="${CUDA_VISIBLE_DEVICES:-7}"
N_SAMPLES="${N_SAMPLES:-20}"
ITERS="${ITERS:-2000}"
LR="${LR:-1e-3}"
LR_BACKBONE="${LR_BACKBONE:-1e-4}"
NORMALIZE="${NORMALIZE:-1}"

PRETRAINED="${PROJECT_ROOT}/exps/exp_sig0.9"
DATA_DIR="${PROJECT_ROOT}/dataset/data"
CCP4_DIR="${PROJECT_ROOT}/dataset/data/ccp4"
CROPS_DIR="${PROJECT_ROOT}/dataset/data/xray_crops"
OUT_NODENSITY="${PROJECT_ROOT}/exps/poc_xray/finetune_test/finetune_ckpt.pth.tar"
OUT_DENSITY="${PROJECT_ROOT}/exps/poc_xray/overfit_test/overfit_ckpt.pth.tar"
OUT_RANDOM="${PROJECT_ROOT}/exps/poc_xray/random_test/overfit_ckpt.pth.tar"

cd "$PROJECT_ROOT"

if [ ! -f "${PRETRAINED}/checkpoint.pth.tar" ]; then
    echo "ERROR: No checkpoint found at ${PRETRAINED}/checkpoint.pth.tar"
    exit 1
fi

# ── Resolve data source ────────────────────────────────────────────────────────
case "$MODE" in
    ccp4)
        if [ ! -d "$CCP4_DIR" ] || [ -z "$(ls -A "$CCP4_DIR" 2>/dev/null)" ]; then
            echo "ERROR: No CCP4 maps found in ${CCP4_DIR}"
            echo "Run scripts/06_download_xray.sh first."
            exit 1
        fi
        DATA_FLAG="--ccp4_dir $CCP4_DIR"
        ;;
    crops)
        if [ ! -d "$CROPS_DIR/test" ] || [ -z "$(ls -A "$CROPS_DIR/test" 2>/dev/null)" ]; then
            echo "ERROR: No precomputed test crops found in ${CROPS_DIR}/test"
            echo "Run scripts/07_process_xray.sh test first."
            exit 1
        fi
        DATA_FLAG="--crops_dir $CROPS_DIR"
        ;;
    *)
        echo "ERROR: Unknown mode '${MODE}'. Choose: ccp4 | crops"
        exit 1
        ;;
esac

# ── Pre-check: find and print the N sample indices with X-ray density ──────────
echo "==> Scanning test split for ${N_SAMPLES} samples with X-ray density..."
python - <<EOF
import sys
sys.path.insert(0, "${PROJECT_ROOT}")
from voxbind.dataset.crossdocked_xray import DatasetCrossDockedXray

crops_dir = "${CROPS_DIR}" if "${MODE}" == "crops" else None
ccp4_dir  = "${CCP4_DIR}"  if "${MODE}" == "ccp4"  else "dataset/data/ccp4"

dset = DatasetCrossDockedXray(
    data_dir="${DATA_DIR}",
    ccp4_dir=ccp4_dir,
    crops_dir=crops_dir,
    split="test",
    aug=False,
    use_xray=True,
)

indices, pocket_ids = [], []
for i in range(min(2000, len(dset))):
    sample = dset[i]
    if sample["xray_available"].item():
        indices.append(i)
        pocket_ids.append(sample["pocket"]["id"][0] if isinstance(sample["pocket"]["id"], list) else sample["pocket"]["id"])
        if len(indices) == ${N_SAMPLES}:
            break

if len(indices) < ${N_SAMPLES}:
    print(f"ERROR: only found {len(indices)}/${N_SAMPLES} samples with X-ray density", flush=True)
    sys.exit(1)

print(f"Found {len(indices)} samples with X-ray density:", flush=True)
for idx, pid in zip(indices, pocket_ids):
    print(f"  [{idx:5d}]  {pid}", flush=True)
EOF

echo
echo "==> Configuration"
echo "    Pretrained   : ${PRETRAINED}"
echo "    Split        : test"
echo "    Data mode    : ${MODE}"
echo "    n_samples    : ${N_SAMPLES}"
echo "    iters        : ${ITERS}"
echo "    lr           : ${LR}  lr_backbone=${LR_BACKBONE}"
echo "    GPUs         : ${GPUS}"
echo

# ── (i) No-density fine-tune ───────────────────────────────────────────────────
echo "==> [1/3] Fine-tuning WITHOUT density (original voxel data only) ..."
echo "    Output: ${OUT_NODENSITY}"
echo
CUDA_VISIBLE_DEVICES="$GPUS" python scripts/poc_density_overfit.py \
    --pretrained_path "$PRETRAINED" \
    --data_dir        "$DATA_DIR" \
    $DATA_FLAG \
    --split           test \
    --n_samples       "$N_SAMPLES" \
    --normalize       "$NORMALIZE" \
    --iters           "$ITERS" \
    --lr              "$LR" \
    --lr_backbone     "$LR_BACKBONE" \
    --out             "$OUT_NODENSITY" \
    --no_density

echo
echo "==> [1/3] Done. Checkpoint: ${OUT_NODENSITY}"
echo

# ── (ii) Density-conditioned fine-tune ────────────────────────────────────────
echo "==> [2/3] Fine-tuning WITH X-ray density conditioning ..."
echo "    Output: ${OUT_DENSITY}"
echo
CUDA_VISIBLE_DEVICES="$GPUS" python scripts/poc_density_overfit.py \
    --pretrained_path "$PRETRAINED" \
    --data_dir        "$DATA_DIR" \
    $DATA_FLAG \
    --split           test \
    --n_samples       "$N_SAMPLES" \
    --normalize       "$NORMALIZE" \
    --iters           "$ITERS" \
    --lr              "$LR" \
    --lr_backbone     "$LR_BACKBONE" \
    --out             "$OUT_DENSITY"

echo
echo "==> [2/3] Done. Checkpoint: ${OUT_DENSITY}"
echo

# ── (iii) Random-density fine-tune (negative control) ────────────────────────
echo "==> [3/3] Fine-tuning WITH RANDOM density (negative control) ..."
echo "    Output: ${OUT_RANDOM}"
echo
CUDA_VISIBLE_DEVICES="$GPUS" python scripts/poc_density_overfit.py \
    --pretrained_path "$PRETRAINED" \
    --data_dir        "$DATA_DIR" \
    $DATA_FLAG \
    --split           test \
    --n_samples       "$N_SAMPLES" \
    --normalize       "$NORMALIZE" \
    --iters           "$ITERS" \
    --lr              "$LR" \
    --lr_backbone     "$LR_BACKBONE" \
    --out             "$OUT_RANDOM" \
    --random_density

echo
echo "==> [3/3] Done. Checkpoint: ${OUT_RANDOM}"
echo
echo "==> All three conditions complete."
echo "    (i)   no-density      : ${OUT_NODENSITY}"
echo "    (ii)  X-ray density   : ${OUT_DENSITY}"
echo "    (iii) random density  : ${OUT_RANDOM}"
echo
echo "==> Next: sample with"
echo "    bash scripts/09c_poc_compare_test.sh crops"
