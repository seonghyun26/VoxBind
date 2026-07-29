#!/usr/bin/env bash
# TAKEOVER chain (replaces the tail of chain_scaling_then_sigma.sh):
#   1) wait for the (orphaned) 0.4M pretrain to finish,
#   2) run the scaling probe (4M + 0.4M vs the cached 40M) → scaling CSV,
#   3) train the GENERATIVE VoxBind with the FROZEN PLINDER encoder (260616_best, Path B),
#   4) ALWAYS resume the original sig=1.0 denoiser to ep1600 — whether the generative
#      run FINISHES or FAILS (per request).
#
# Launch (detached) — own session/pgid so the denoiser-group kill cannot hit it:
#   cd /home1/irteam/VoxBind/voxbind && \
#   setsid nohup bash scripts/archive/chains/chain_frozenenc_then_sigma.sh \
#     > log/chain_frozenenc_then_sigma.log 2>&1 &
set -u
ROOT=/home1/irteam/VoxBind/voxbind
PY=/opt/conda/envs/voxbind/bin
LOG=$ROOT/log
SDIR=$ROOT/exps/exp_sig1.0+prefetch_factor16_wjs.n_targets0
TARGET=1600                                            # denoiser final epoch

# scaling arms (probe) + generative arm
EXP_0P4M=atomblob_density_gradmag_vit_mae_0p4m_v5_pretrain
CK_0P4M=$ROOT/exps/$EXP_0P4M/checkpoint_e0099.pth.tar
COND_4M=atomblob_density_gradmag_4m
COND_0P4M=atomblob_density_gradmag_0p4m
BASE_COND=atomblob_density_gradmag
GEN_CFG=config_train_voxbind_frozenenc_plinder
GEN_EXP=voxbind_frozenenc_plinder_sig0.9

export LD_LIBRARY_PATH=/opt/conda/envs/voxbind/lib:${LD_LIBRARY_PATH:-}
export TORCHDYNAMO_DISABLE=1   # no C compiler on this box → neutralize torch.compile everywhere
ts(){ date "+%Y-%m-%d %H:%M:%S"; }
drain_gpus(){ for _ in $(seq 1 18); do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END{print s+0}')
    [ "${USED:-99999}" -lt 4000 ] && break; sleep 10; done
  echo "[$(ts)] GPU mem used (MiB total): ${USED:-n/a}"; }
mkdir -p "$LOG"; cd "$ROOT" || exit 1

# ── 1) wait for the orphaned 0.4M pretrain to finish (its e0099 ckpt to appear) ──
echo "[$(ts)] waiting for 0.4M pretrain e0099 ..."
while [ ! -f "$CK_0P4M" ]; do
  if ! pgrep -f '[t]rain_density_vit_mae.py.*0p4m' >/dev/null 2>&1; then
    echo "[$(ts)] WARN: 0.4M process gone and no e0099 ckpt yet — waiting 60s more then giving up on it"
    sleep 60
    [ -f "$CK_0P4M" ] || { echo "[$(ts)] 0.4M produced no e0099 — will SKIP its probe row"; break; }
  fi
  sleep 60
done
[ -f "$CK_0P4M" ] && echo "[$(ts)] 0.4M done."
sleep 30          # let final ckpt flush + NCCL teardown
drain_gpus

# ── 2) scaling probe: features for whichever arms have e0099, then one probe vs 40M ──
PROBE_CONDS=("$BASE_COND")
for pair in "atomblob_density_gradmag_vit_mae_4m_v5_pretrain:$COND_4M" \
            "$EXP_0P4M:$COND_0P4M"; do
  EXP="${pair%%:*}"; COND="${pair##*:}"
  if [ -f "$ROOT/exps/$EXP/checkpoint_e0099.pth.tar" ]; then
    echo "[$(ts)] extracting features for $COND ..."
    CUDA_VISIBLE_DEVICES=0 "$PY/python" dataset/01c_pdbbind_probe.py features \
      --condition "$COND" --voxel_version v5 --epoch 99 --device cuda --batch_size 32 \
      > "$LOG/probe_features_${EXP}.log" 2>&1
    echo "[$(ts)] features $COND exit=$?"; PROBE_CONDS+=("$COND")
  else
    echo "[$(ts)] WARN: $EXP e0099 missing — excluding $COND from probe"
  fi
done
if [ "${#PROBE_CONDS[@]}" -gt 1 ]; then
  echo "[$(ts)] probing scaling ladder: ${PROBE_CONDS[*]}"
  CUDA_VISIBLE_DEVICES="" "$PY/python" dataset/01c_pdbbind_probe.py probe \
    --conditions "${PROBE_CONDS[@]}" --voxel_version v5 --epoch 99 --seeds 3 \
    --device cpu --tag scaling --allow_stale_features > "$LOG/probe_scaling.log" 2>&1
  echo "[$(ts)] probe exit=$?  -> dataset/data/pdbbind/probe_results_e99_v5_scaling.csv"
fi

# ── 3) preempt ANY running denoiser (safety) + drain GPUs ───────────────────────
echo "[$(ts)] ensuring GPUs are free for the generative run ..."
mapfile -t SIG_PGIDS < <(ps -eo pgid,cmd | grep "[t]rain_ddp.py smooth_sigma=1.0" | awk '{print $1}' | sort -u)
for g in "${SIG_PGIDS[@]:-}"; do [ -n "$g" ] && kill -TERM -- -"$g" 2>/dev/null; done
sleep 10
for g in "${SIG_PGIDS[@]:-}"; do [ -n "$g" ] && kill -KILL -- -"$g" 2>/dev/null; done
drain_gpus

# ── 4) train the GENERATIVE VoxBind with the FROZEN encoder (non-fatal) ──────────
OUT=$ROOT/exps/$GEN_EXP
mkdir -p "$OUT"
echo "[$(ts)] START generative (frozen 260616_best encoder): $GEN_EXP (cfg=$GEN_CFG)"
CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY/torchrun" --standalone --nproc_per_node=4 \
  "$ROOT/train_ddp.py" --config-name="$GEN_CFG" \
  exp_name="$GEN_EXP" output_dir="$OUT" hydra.run.dir="$OUT" \
  > "$LOG/${GEN_EXP}.log" 2>&1
GEN_EXIT=$?
echo "[$(ts)] END generative exit=$GEN_EXIT (last $(grep -oE 'epoch: [0-9]+' "$LOG/${GEN_EXP}.log" | tail -1))"
[ "$GEN_EXIT" -ne 0 ] && echo "[$(ts)] NOTE: generative run FAILED (exit=$GEN_EXIT) — resuming denoiser anyway per request"

# ── 5) resume the original sig=1.0 denoiser to ep$TARGET (ALWAYS) ────────────────
drain_gpus
LASTEP=$("$PY/python" -c "import torch;print(torch.load('$SDIR/checkpoint.pth.tar',map_location='cpu',weights_only=False)['epoch'])" 2>/dev/null)
if [ -z "${LASTEP:-}" ]; then
  echo "[$(ts)] ERROR: cannot read denoiser ckpt epoch — resume manually"
else
  RES=$(( LASTEP + 1 )); REM=$(( TARGET - RES ))
  echo "[$(ts)] resuming sig=1.0 (ckpt ep $LASTEP -> resume $RES, +$REM -> $TARGET)"
  if [ "$REM" -gt 0 ]; then
    SIG=1.0 NUM_EPOCHS=$REM RESUME_EPOCH=$RES bash "$ROOT/scripts/archive/launchers/35_train_baseline_4gpu.sh" \
      > "$LOG/sig1.0_resume_after_frozenenc.log" 2>&1
    echo "[$(ts)] sig=1.0 launcher exit=$?"
  else
    echo "[$(ts)] sig=1.0 already at/past $TARGET (last $LASTEP) — not resuming"
  fi
fi
echo "[$(ts)] CHAIN COMPLETE"
