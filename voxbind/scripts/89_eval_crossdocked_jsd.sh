#!/usr/bin/env bash
# 89_eval_crossdocked_jsd.sh — the TargetDiff / VoxBind distribution metrics for every arm
# figures/draw.py draws: bond-distance, pair-distance and atom-type JSD against the 100
# CrossDocked test ligands, the ring-size table, and the VoxBind Fig. 10 histograms.
#
#   bash voxbind/scripts/89_eval_crossdocked_jsd.sh
#   /opt/conda/envs/voxbind/bin/python figures/draw.py jsd
#
# The arms and their roots are figures/_parts/00_core.py's ARMS, spelled out: the labels
# ARE the keys the figures colour by, so a label changed here must change there too. The
# molecules are the 79 electron-density pockets for every arm (TargetDiff and VoxBind hold
# 100; the extra 21 are left out so every arm stands on the same pockets). The reference
# stays the full 100 test ligands, because that is what the published numbers use --
# see the docstring of tools/eval_crossdocked_jsd.py for why it is not TargetDiff's
# shipped training-set histograms.
#
# --selfcheck re-scores TargetDiff over all 100 pockets and fails the run if it drifts
# from the published row, so the protocol claim is re-verified every time this runs.
# ~2 min on 16 workers; no GPU.
set -euo pipefail
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT"
export VOXBIND_BASEDRUG="${VOXBIND_BASEDRUG:-$(cd "$ROOT/.." && pwd)/base_drug}"
PY="${PY:-/opt/conda/envs/voxbind/bin/python}"
WORKERS="${WORKERS:-16}"
OUT="${OUT:-results/task2-drugdesign/_shared/260913_crossdocked_jsd/crossdocked_jsd.json}"
E=voxbind/exps

"$PY" voxbind/scripts/tools/eval_crossdocked_jsd.py \
    --run "AR=$E/baselines_pose/ar" \
    --run "Pocket2Mol=$E/baselines_pose/pocket2mol" \
    --run "DiffSBDD=$E/baselines_pose/diffsbdd" \
    --run "DecompDiff=$E/baselines_pose/decompdiff" \
    --run "FuncBind=$E/baselines_pose/funcbind" \
    --run "TargetDiff=$VOXBIND_BASEDRUG/eval/targetdiff" \
    --run "VoxBind=$E/_vanilla_ep923/samples/full_eval_ep923" \
    --run "CoDE=$E/voxbind_frozenenc_atomblob7_v2p1_sig0.9/samples/full_eval_ep350" \
    --workers "$WORKERS" --selfcheck --out "$OUT"
