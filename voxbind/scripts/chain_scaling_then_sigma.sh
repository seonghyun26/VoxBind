#!/usr/bin/env bash
# ENCODER-SIZE SCALING LADDER (0.4M / 4M / 40M):
#   stop the sig=1.0 denoiser (+ any extend watcher), pretrain the 4M then the 0.4M
#   C+D+G arm (exact frozen-crop invfreq-v5 twins of the 40M, only the transformer
#   width/depth shrunk), frozen-probe BOTH vs the cached 40M baseline
#   (atomblob_density_gradmag, probe 0.595) into one scaling CSV, then RESUME the
#   denoiser to ep1600. Mirrors scripts/chain_resample_then_sigma.sh +
#   scripts/chain_sig1.0_extend_to_1600.sh (resume convention / 1600 target).
#
# Launch (detached) — own session/pgid so the denoiser-group kill cannot hit it:
#   cd /home1/irteam/VoxBind/voxbind && \
#   setsid nohup bash scripts/chain_scaling_then_sigma.sh \
#     > log/chain_scaling_then_sigma.log 2>&1 &
set -u
ROOT=/home1/irteam/VoxBind/voxbind
PY=/opt/conda/envs/voxbind/bin
LOG=$ROOT/log
# scaling arms: "exp_name:config_name:head_hidden_dim:probe_condition"
ARMS=(
  "atomblob_density_gradmag_vit_mae_4m_v5_pretrain:config_train_atomblob_density_gradmag_vit_mae_4m_v5:atomblob_density_gradmag_4m"
  "atomblob_density_gradmag_vit_mae_0p4m_v5_pretrain:config_train_atomblob_density_gradmag_vit_mae_0p4m_v5:atomblob_density_gradmag_0p4m"
)
BASE_COND=atomblob_density_gradmag                      # 40M frozen-crop invfreq v5 (probe 0.595, features cached)
SDIR=$ROOT/exps/exp_sig1.0+prefetch_factor16_wjs.n_targets0
TARGET=1600                                             # denoiser final epoch (matches extend-to-1600 watcher)
export LD_LIBRARY_PATH=/opt/conda/envs/voxbind/lib:${LD_LIBRARY_PATH:-}
export TORCHDYNAMO_DISABLE=1   # no C compiler on this box → neutralize ALL torch.compile (belt+braces w/ compile.enabled=false)
ts(){ date "+%Y-%m-%d %H:%M:%S"; }
mkdir -p "$LOG"; cd "$ROOT" || exit 1

# ── 1) stop the sig=1.0 denoiser + any extend watcher (kill ALL matching pgroups) ─
echo "[$(ts)] stopping sig=1.0 denoiser + extend watcher ..."
mapfile -t SIG_PGIDS < <(ps -eo pgid,cmd | grep "[t]rain_ddp.py smooth_sigma=1.0" | awk '{print $1}' | sort -u)
# exclude THIS script's own pgid so the watcher grep can't self-kill
SELF_PGID=$(ps -o pgid= -p $$ | tr -d ' ')
mapfile -t CHN_PGIDS < <(ps -eo pgid,cmd | grep "[c]hain_sig1.0_extend" | awk '{print $1}' | sort -u | grep -v "^${SELF_PGID}$")
ALL_PGIDS=("${SIG_PGIDS[@]:-}" "${CHN_PGIDS[@]:-}")
for g in "${ALL_PGIDS[@]}"; do
  [ -n "$g" ] && { echo "[$(ts)] kill -TERM pgroup $g"; kill -TERM -- -"$g" 2>/dev/null; }
done
sleep 15
for g in "${ALL_PGIDS[@]}"; do [ -n "$g" ] && kill -KILL -- -"$g" 2>/dev/null; done
# wait until the GPUs actually drain (NCCL teardown) before launching, up to ~3 min
for _ in $(seq 1 18); do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END{print s+0}')
  [ "${USED:-99999}" -lt 4000 ] && break
  sleep 10
done
echo "[$(ts)] GPU mem used (MiB total): ${USED:-n/a}"

# ── 2) pretrain each scaling arm (4 GPU, effective batch 128 = baseline) ───────
for arm in "${ARMS[@]}"; do
  EXP="${arm%%:*}"; rest="${arm#*:}"; CFG="${rest%%:*}"; COND="${rest##*:}"
  OUT=$ROOT/exps/$EXP
  if [ -f "$OUT/checkpoint_e0099.pth.tar" ]; then
    echo "[$(ts)] SKIP pretrain $EXP — e99 checkpoint already exists"
  else
    mkdir -p "$OUT"
    echo "[$(ts)] START pretrain $EXP (100 ep, bsz=32 accum=1 x4 = eff 128)"
    CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY/torchrun" --standalone --nproc_per_node=4 \
      "$ROOT/train_density_vit_mae.py" --config-name="$CFG" \
      exp_name="$EXP" output_dir="$OUT" hydra.run.dir="$OUT" \
      > "$LOG/${EXP}.log" 2>&1
    echo "[$(ts)] END pretrain $EXP exit=$? (last $(grep -oE 'epoch: [0-9]+' "$LOG/${EXP}.log" | tail -1))"
  fi
done

# ── 3) extract features for each arm, then ONE probe of all three vs the 40M baseline ─
PROBE_CONDS=("$BASE_COND")
for arm in "${ARMS[@]}"; do
  EXP="${arm%%:*}"; rest="${arm#*:}"; COND="${rest##*:}"
  OUT=$ROOT/exps/$EXP
  if [ -f "$OUT/checkpoint_e0099.pth.tar" ]; then
    echo "[$(ts)] extracting features for $COND (GPU)..."
    CUDA_VISIBLE_DEVICES=0 "$PY/python" dataset/01c_pdbbind_probe.py features \
      --condition "$COND" --voxel_version v5 --epoch 99 --device cuda --batch_size 32 \
      > "$LOG/probe_features_${EXP}.log" 2>&1
    echo "[$(ts)] features $COND exit=$?"
    PROBE_CONDS+=("$COND")
  else
    echo "[$(ts)] ERROR: $EXP e99 checkpoint missing — excluding $COND from the probe"
  fi
done

if [ "${#PROBE_CONDS[@]}" -gt 1 ]; then
  echo "[$(ts)] probing scaling ladder: ${PROBE_CONDS[*]}"
  CUDA_VISIBLE_DEVICES="" "$PY/python" dataset/01c_pdbbind_probe.py probe \
    --conditions "${PROBE_CONDS[@]}" \
    --voxel_version v5 --epoch 99 --seeds 3 --device cpu --tag scaling --allow_stale_features \
    > "$LOG/probe_scaling.log" 2>&1
  echo "[$(ts)] probe exit=$?  -> dataset/data/pdbbind/probe_results_e99_v5_scaling.csv"
else
  echo "[$(ts)] no new encoders probed — skipping scaling probe"
fi

# ── 4) resume sig=1.0 to ep$TARGET (resume epoch = checkpoint epoch + 1) ────────
LASTEP=$("$PY/python" -c "import torch;print(torch.load('$SDIR/checkpoint.pth.tar',map_location='cpu',weights_only=False)['epoch'])" 2>/dev/null)
if [ -z "${LASTEP:-}" ]; then
  echo "[$(ts)] ERROR: cannot read denoiser ckpt epoch (env wiped?) — NOT resuming; resume manually"
else
  RES=$(( LASTEP + 1 )); REM=$(( TARGET - RES ))
  echo "[$(ts)] resuming sig=1.0 (ckpt ep $LASTEP -> resume $RES, +$REM -> $TARGET)"
  if [ "$REM" -gt 0 ]; then
    SIG=1.0 NUM_EPOCHS=$REM RESUME_EPOCH=$RES bash "$ROOT/scripts/35_train_baseline_4gpu.sh" \
      > "$LOG/sig1.0_resume_after_scaling.log" 2>&1
    echo "[$(ts)] sig=1.0 launcher exit=$?"
  else
    echo "[$(ts)] sig=1.0 already at/past $TARGET (last $LASTEP) — not resuming"
  fi
fi
echo "[$(ts)] CHAIN COMPLETE"
