#!/usr/bin/env bash
# 66_chain_urepa_two_arms.sh — run BOTH U-REPA loss compositions for 100 epochs each,
# back-to-back on the same 4 GPUs, from the same ep350 warm start and the same champion
# teacher. Everything except the alignment-loss composition is held fixed, so the two
# curves are directly comparable at matched epochs.
#
#   arm A (running when this chain starts) — measurement-driven
#     L_align = manifold only, relkl, centered, tau 0.1, split(w_intra=w_inter=1), lam 5.0
#     Rationale: at the bottleneck R²(teacher<-student)=0.069-0.076 and only 8/512 CCA
#     directions exceed 0.9, so the pointwise target is largely unreachable for THIS pair.
#
#   arm B — paper-faithful (arXiv:2503.18414)
#     L := L_denoise + lam(L_REPA + 3*L_ML), L_ML = ||sim(y*[i],y*[j]) - sim(h[i],h[j])||²_F
#     i.e. tokenwise REPA KEPT, manifold over SAMPLES, Frobenius (gram) form, NO temperature.
#     lam 40 so its contribution is ~2% of the denoise loss — matching arm A's 2.2%, since
#     otherwise the comparison would be "alignment on" vs "alignment ~off".
#
#   Deliberate deviations from the paper, and why:
#     * lr 1e-5 / wd 1e-2 (not 1e-4 / 0): inherited from the VoxBind baseline so that a
#       lam=0 control means "continue ep350 unchanged" and the curve isolates alignment.
#       The paper's 1e-4/wd0 is a from-scratch setting.
#     * center=true on the similarity matrices: the paper does not mention centering, but
#       measured on real champion tokens the UNcentered pooled cosines are 0.999 +- 0.001,
#       so an uncentered Frobenius loss mostly pushes the student toward all-similar.
#     * Stage-1 (U-Net lr=0) warmup: from this project's plan doc, not the paper.
#
# usage: 66_chain_urepa_two_arms.sh <arm_A_torchrun_pid>
#        ARM_A_EPOCHS/ARM_B_EPOCHS override the 100/100 default.
set -uo pipefail
ROOT=/home1/irteam/VoxBind/voxbind
A_PID="${1:?usage: 66_chain_urepa_two_arms.sh <arm_A_torchrun_pid>}"
A_DIR="$ROOT/exps/voxbind_urepa_champion_ep350_lam5.0"
A_EPOCHS="${ARM_A_EPOCHS:-100}"
B_EPOCHS="${ARM_B_EPOCHS:-100}"
cd "$ROOT" || exit 1

echo "[66_chain] arm A pid=$A_PID dir=$A_DIR — capping at $A_EPOCHS epochs"

# ── wait for arm A to reach the cap (epoch index A_EPOCHS-1) ────────────────────
while kill -0 "$A_PID" 2>/dev/null; do
  n=$(grep -cE ">> epoch: [0-9]+ \(" "$A_DIR"/train*.log 2>/dev/null || echo 0)
  [ "$n" -ge "$A_EPOCHS" ] && { echo "[66_chain] arm A reached $n epochs — stopping"; break; }
  sleep 120
done

if kill -0 "$A_PID" 2>/dev/null; then
  kill "$A_PID"
  for _ in $(seq 30); do kill -0 "$A_PID" 2>/dev/null || break; sleep 5; done
  kill -9 "$A_PID" 2>/dev/null
fi
# GPUs must be fully released before arm B allocates ~108 GB/card
for _ in $(seq 60); do
  [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0 ] && break
  sleep 5
done
echo "[66_chain] GPUs free; arm A final: $(tail -2 "$A_DIR"/train*.log | head -1)"

# ── arm B: paper-faithful composition ──────────────────────────────────────────
echo "[66_chain] launching arm B (paper-faithful) for $B_EPOCHS epochs"
EXP_NAME=voxbind_urepa_champion_ep350_paper \
LAM=40.0 MODE=gram TAU=1.0 SAMPLING=pool \
REPA_WEIGHT=1.0 ML_WEIGHT=3.0 \
NUM_EPOCHS="$B_EPOCHS" \
  bash "$ROOT/scripts/65_train_urepa_champion_4gpu.sh"
echo "[66_chain] CHAIN_DONE"
