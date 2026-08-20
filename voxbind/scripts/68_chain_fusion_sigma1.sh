#!/usr/bin/env bash
# 68_chain_fusion_sigma1.sh — when the sigma=0.9 from-scratch fusion run finishes its 350
# epochs, launch the identical run at sigma=1.0.
#
#   sigma 0.9 : exps/voxbind_fusion_champion_oracle_scratch      (vanilla peer: exp_sig0.9_v2)
#   sigma 1.0 : exps/voxbind_fusion_champion_oracle_scratch_sig1 (vanilla peer: exp_sig1.0_350ep)
#
# Everything else is held fixed (from scratch, 350 epochs, champion frozen, ligand+density
# fed to the encoder), so the pair isolates the walk-jump noise level. Both remain ORACLE
# runs — the encoder is given the real ligand — so neither belongs in the de-novo table.
#
# usage: 68_chain_fusion_sigma1.sh <sigma0.9_torchrun_pid>
set -uo pipefail
ROOT=/home1/irteam/VoxBind/voxbind
A_PID="${1:?usage: 68_chain_fusion_sigma1.sh <sigma0.9_torchrun_pid>}"
A_DIR="$ROOT/exps/voxbind_fusion_champion_oracle_scratch"
TARGET_EPOCHS="${TARGET_EPOCHS:-350}"
cd "$ROOT" || exit 1

echo "[68_chain] waiting for sigma0.9 (pid $A_PID) to reach $TARGET_EPOCHS epochs"
while kill -0 "$A_PID" 2>/dev/null; do
  n=$(grep -cE ">> epoch: [0-9]+ \(" "$A_DIR"/train*.log 2>/dev/null || echo 0)
  [ "$n" -ge "$TARGET_EPOCHS" ] && { echo "[68_chain] sigma0.9 reached $n epochs"; break; }
  sleep 300
done

# It may have exited on its own (350 epochs completed) — either way, make sure it is gone
# and the GPUs are released before the sigma=1.0 run allocates ~110 GB/card.
if kill -0 "$A_PID" 2>/dev/null; then
  kill "$A_PID"
  for _ in $(seq 36); do kill -0 "$A_PID" 2>/dev/null || break; sleep 5; done
  kill -9 "$A_PID" 2>/dev/null
fi
for _ in $(seq 60); do
  [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0 ] && break
  sleep 5
done
echo "[68_chain] sigma0.9 final: $(grep -E 'train\]' "$A_DIR"/train*.log | tail -1)"
echo "[68_chain] GPUs free; launching sigma=1.0"

SIGMA=1.0 WARM_START="" NUM_EPOCHS=350 \
EXP_NAME=voxbind_fusion_champion_oracle_scratch_sig1 \
  bash "$ROOT/scripts/67_train_fusion_champion_oracle_4gpu.sh"
echo "[68_chain] CHAIN_DONE"
