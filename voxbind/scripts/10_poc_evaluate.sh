#!/bin/bash
# Evaluate VoxBind PoC samples using TargetDiff metrics.
# Run from the voxbind/ directory: bash scripts/10_poc_evaluate.sh [docking_mode]
#
# Usage:
#   bash scripts/10_poc_evaluate.sh [docking_mode]
#
# Arguments (optional):
#   docking_mode    none | vina_score | vina_dock  (default: none)
#                     none       — QED, SA, LogP, Lipinski only (fast)
#                     vina_score — + Vina score_only + minimize
#                     vina_dock  — + full Vina docking (slow)
#
# Environment variables (optional):
#   CUDA_VISIBLE_DEVICES   GPUs (default: 4,5)
#   COMPARE_DIR            compare output dir (default: exps/poc_xray/compare)
#   PROTEIN_ROOT           CrossDocked protein root for Vina docking
#                          (required for vina_score / vina_dock)
#
# Prerequisites:
#   bash scripts/09_poc_compare_samples.sh   (generates the SDF files)
#
# Outputs:
#   exps/poc_xray/compare/eval_summary.txt   human-readable table
#   exps/poc_xray/compare/eval_results.pt    raw per-molecule data

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

DOCKING_MODE="${1:-none}"
GPUS="${CUDA_VISIBLE_DEVICES:-7}"
COMPARE_DIR="${COMPARE_DIR:-${PROJECT_ROOT}/exps/poc_xray/compare}"
TARGETDIFF_ROOT="${PROJECT_ROOT}/../../targetdiff"
# Default to VoxBind's crossdocked_pocket10 (pocket-only PDBs, always available).
# For full receptor accuracy set PROTEIN_ROOT to crossdocked_v1.1_rmsd1.0/.
PROTEIN_ROOT="${PROTEIN_ROOT:-${PROJECT_ROOT}/dataset/data/crossdocked_pocket10}"

cd "$PROJECT_ROOT"

# ── Prerequisite checks ───────────────────────────────────────────────────────
if [ ! -d "$COMPARE_DIR" ]; then
    echo "ERROR: Compare directory not found at ${COMPARE_DIR}"
    echo "Run scripts/09_poc_compare_samples.sh first."
    exit 1
fi

if [ ! -f "${COMPARE_DIR}/baseline/samples.sdf" ] && \
   [ ! -f "${COMPARE_DIR}/xray_cond/samples.sdf" ]; then
    echo "ERROR: No samples.sdf found in ${COMPARE_DIR}/"
    echo "Run scripts/09_poc_compare_samples.sh first."
    exit 1
fi

if [ ! -d "$TARGETDIFF_ROOT" ]; then
    echo "ERROR: TargetDiff root not found at ${TARGETDIFF_ROOT}"
    echo "Set TARGETDIFF_ROOT or adjust the path in this script."
    exit 1
fi

# Auto-detect conditions (always baseline + xray_cond; add finetuned/random_density if present)
CONDITIONS="baseline xray_cond"
if [ -f "${COMPARE_DIR}/finetuned/samples.sdf" ]; then
    CONDITIONS="baseline finetuned xray_cond"
    echo "==> Three-way evaluation  (finetuned/ found)"
fi
if [ -f "${COMPARE_DIR}/random_density/samples.sdf" ]; then
    CONDITIONS="${CONDITIONS} random_density"
    echo "    (random_density/ found — adding to evaluation)"
fi
if [ -f "${COMPARE_DIR}/random_trained/samples.sdf" ]; then
    CONDITIONS="${CONDITIONS} random_trained"
    echo "    (random_trained/ found — adding to evaluation)"
fi

echo "==> Evaluating PoC samples  (docking_mode=${DOCKING_MODE})"
echo "    Compare dir    : ${COMPARE_DIR}"
echo "    Conditions     : ${CONDITIONS}"
echo "    TargetDiff     : ${TARGETDIFF_ROOT}"
echo "    GPUs           : ${GPUS}"
echo

case "$DOCKING_MODE" in
    none)
        CUDA_VISIBLE_DEVICES="$GPUS" python scripts/poc_evaluate.py \
            --compare_dir     "$COMPARE_DIR" \
            --targetdiff_root "$TARGETDIFF_ROOT" \
            --docking_mode    none \
            --conditions      $CONDITIONS
        ;;

    vina_score | vina_dock)
        if [ ! -d "$PROTEIN_ROOT" ]; then
            echo "ERROR: Protein root not found at ${PROTEIN_ROOT}"
            echo "Set PROTEIN_ROOT to the TargetDiff CrossDocked data directory."
            exit 1
        fi
        echo "    Protein root   : ${PROTEIN_ROOT}"
        echo
        CUDA_VISIBLE_DEVICES="$GPUS" python scripts/poc_evaluate.py \
            --compare_dir     "$COMPARE_DIR" \
            --targetdiff_root "$TARGETDIFF_ROOT" \
            --docking_mode    "$DOCKING_MODE" \
            --protein_root    "$PROTEIN_ROOT" \
            --conditions      $CONDITIONS
        ;;

    *)
        echo "ERROR: Unknown docking_mode '${DOCKING_MODE}'."
        echo "Choose: none | vina_score | vina_dock"
        exit 1
        ;;
esac

echo
echo "==> Evaluation complete."
echo "    Summary : ${COMPARE_DIR}/eval_summary.txt"
echo "    Raw     : ${COMPARE_DIR}/eval_results.pt"
