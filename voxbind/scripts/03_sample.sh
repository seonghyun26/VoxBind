#!/bin/bash
# Sample molecules with VoxBind on the Crossdocked dataset.
# Run from the project root: bash scripts/03_sample.sh [OPTIONS]
#
# Usage:
#   bash scripts/03_sample.sh [sigma] [split] [n_samples] [n_targets]
#
# Arguments (all optional, positional):
#   sigma       smooth_sigma of the pretrained model (default: 0.9)
#   split       dataset split to sample from: val or test (default: test)
#   n_samples   number of samples per pocket (default: 10)
#   n_targets   number of targets to sample (default: 100)
#
# Examples:
#   bash scripts/03_sample.sh                        # sigma=0.9, test split, 10 samples, 100 targets
#   bash scripts/03_sample.sh 0.9 val 10 100         # validation split
#   bash scripts/03_sample.sh 1.0 test 20 100
#
# Generated molecules are saved in:
#   voxbind/exps/exp_sig<sigma>/samples/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

SIGMA="${1:-0.9}"
SPLIT="${2:-test}"
N_SAMPLES="${3:-4}"
N_TARGETS="${4:-10}"
PRETRAINED_PATH="exps/exp_sig${SIGMA}"
GPUS="${CUDA_VISIBLE_DEVICES:-4,5}"

cd "$PROJECT_ROOT"

if [ ! -d "$PRETRAINED_PATH" ]; then
    echo "ERROR: Pretrained checkpoint not found at voxbind/${PRETRAINED_PATH}"
    echo "Run 02_train.sh first, or set the correct sigma."
    exit 1
fi

echo "==> Sampling with VoxBind"
echo "    Pretrained path : ${PRETRAINED_PATH}"
echo "    Split           : ${SPLIT}"
echo "    Samples/pocket  : ${N_SAMPLES}"
echo "    Targets         : ${N_TARGETS}"

CUDA_VISIBLE_DEVICES="$GPUS" conda run -n voxbind python sample.py \
    pretrained_path="${PRETRAINED_PATH}" \
    wjs.split="${SPLIT}" \
    wjs.n_samples_per_pocket="${N_SAMPLES}" \
    wjs.n_targets="${N_TARGETS}"

echo "==> Sampling complete. Results saved in ${PRETRAINED_PATH}/samples/"
