#!/usr/bin/env bash
# Wait for scripts/build_plinder_otf.sh to finish, VALIDATE the manifest + data index,
# then run the standard protocol around the voxbind denoiser:
#   stop the running sig=1.0 denoiser → run the 40M PLINDER otf pretrain → resume the
#   denoiser to ep1600 (whether the pretrain finishes OR fails).
# Aborts (leaving the denoiser running) if the build validation fails.
set -u
ROOT=/home1/irteam/VoxBind/voxbind
PY=/opt/conda/envs/voxbind/bin
LOG=$ROOT/log
CFG=config_train_atomblob_density_gradmag_vit_mae_40m_invfreq_plinder_otf
EXP=260617_plinder_otf_cdg_invfreq_p100_pretrain
RDIR=$ROOT/dataset/data/xray_resample_plinder
DATAPT=$ROOT/dataset/data/data_train_plinder.pt
SDIR=$ROOT/exps/exp_sig1.0+prefetch_factor16_wjs.n_targets0
TARGET=1600
export LD_LIBRARY_PATH=/opt/conda/envs/voxbind/lib:${LD_LIBRARY_PATH:-}
export TORCHDYNAMO_DISABLE=1
ts(){ date "+%Y-%m-%d %H:%M:%S"; }
drain_gpus(){ for _ in $(seq 1 18); do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END{print s+0}')
    [ "${USED:-99999}" -lt 4000 ] && break; sleep 10; done
  echo "[$(ts)] GPU mem used (MiB total): ${USED:-n/a}"; }
stop_denoiser(){
  echo "[$(ts)] stopping sig=1.0 denoiser ..."
  mapfile -t PG < <(ps -eo pgid,cmd | grep "[t]rain_ddp.py smooth_sigma=1.0" | awk '{print $1}' | sort -u)
  for g in "${PG[@]:-}"; do [ -n "$g" ] && { echo "[$(ts)] kill -TERM pgroup $g"; kill -TERM -- -"$g" 2>/dev/null; }; done
  sleep 12
  for g in "${PG[@]:-}"; do [ -n "$g" ] && kill -KILL -- -"$g" 2>/dev/null; done
  drain_gpus
}
resume_denoiser(){
  LASTEP=$("$PY/python" -c "import torch;print(torch.load('$SDIR/checkpoint.pth.tar',map_location='cpu',weights_only=False)['epoch'])" 2>/dev/null)
  if [ -z "${LASTEP:-}" ]; then echo "[$(ts)] ERROR: cannot read denoiser ckpt epoch — resume manually"; return; fi
  RES=$(( LASTEP + 1 )); REM=$(( TARGET - RES ))
  echo "[$(ts)] resuming sig=1.0 (ckpt ep $LASTEP -> resume $RES, +$REM -> $TARGET)"
  if [ "$REM" -gt 0 ]; then
    SIG=1.0 NUM_EPOCHS=$REM RESUME_EPOCH=$RES bash "$ROOT/scripts/35_train_baseline_4gpu.sh" \
      > "$LOG/sig1.0_resume_after_plinder.log" 2>&1
    echo "[$(ts)] sig=1.0 launcher exit=$?"
  else
    echo "[$(ts)] sig=1.0 already at/past $TARGET (last $LASTEP) — not resuming"
  fi
}
cd "$ROOT" || exit 1

# single-instance guard: if another watcher already holds the lock, exit (prevents
# a double stop-denoiser / double pretrain launch from an accidental relaunch).
exec 9>/tmp/plinder_otf_watcher.lock
if ! flock -n 9; then echo "[$(ts)] another watcher holds the lock — exiting"; exit 0; fi

echo "[$(ts)] waiting for build (BUILD COMPLETE in build_plinder_otf.log) ..."
for _ in $(seq 1 120); do
  grep -q 'BUILD COMPLETE' log/build_plinder_otf.log 2>/dev/null && break
  sleep 30
done
grep -q 'BUILD COMPLETE' log/build_plinder_otf.log 2>/dev/null || { echo "[$(ts)] build did not complete in time — ABORT (denoiser left running)"; exit 1; }
echo "[$(ts)] build complete. validating artifacts ..."

"$PY/python" - "$RDIR" "$DATAPT" << 'PY'
import sys, os, glob, random, numpy as np, torch
rdir, datapt = sys.argv[1], sys.argv[2]
VAL_SZ, MAX_LEN = 100, 30
npz = sorted(glob.glob(os.path.join(rdir, "*_manifest.npz")))
assert npz, f"no *_manifest.npz in {rdir}"
z = np.load(npz[0], allow_pickle=True)
assert {"pdb_id","centroid","ok"} <= set(z.files), f"manifest keys: {z.files}"
n_mani = len(z["pdb_id"]); n_ok = int(z["ok"].sum())
man_pids = [str(x) for x in z["pdb_id"]]
# Replicate the loader's train laydown (shuffle1234 -> drop VAL_SZ -> filter max_len) and
# require the manifest to align with the FILTERED data file (NOT the unfiltered tuples).
tuples = torch.load(datapt, weights_only=False)
order = list(tuples); random.Random(1234).shuffle(order)
order = order[: len(order) - VAL_SZ]
order = [t for t in order if t[1]["max_len"] <= MAX_LEN]
filt_pids = [str(p["id"]).split("/")[-1].split("_")[0].lower() for p,_l in order]
print(f"  manifest={os.path.basename(npz[0])} n={n_mani} ok={n_ok} ({100*n_ok/max(n_mani,1):.1f}%)  "
      f"tuples={len(tuples)} filtered_train={len(filt_pids)}")
assert len(filt_pids) == n_mani, f"MISALIGNED: filtered train {len(filt_pids)} != manifest {n_mani}"
assert filt_pids == man_pids, "MISALIGNED: pdb_id order differs (shuffle/filter mismatch)"
assert n_ok > 0.5*n_mani, f"too few ok: {n_ok}/{n_mani}"
assert os.path.exists(os.path.join(rdir, "resample.json")), "resample.json missing"
print("  VALIDATION OK (manifest aligns with filtered train order)")
PY
if [ $? -ne 0 ]; then echo "[$(ts)] VALIDATION FAILED — NOT touching the denoiser, NOT launching pretrain"; exit 1; fi

# ── protocol: stop denoiser → pretrain → resume denoiser ──
stop_denoiser
OUT=$ROOT/exps/$EXP
mkdir -p "$OUT"
echo "[$(ts)] START PLINDER otf pretrain $EXP (100 ep, bsz=32 x4 = eff 128)"
CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY/torchrun" --standalone --nproc_per_node=4 \
  "$ROOT/train_density_vit_mae.py" --config-name="$CFG" \
  exp_name="$EXP" output_dir="$OUT" hydra.run.dir="$OUT" \
  > "$LOG/${EXP}.log" 2>&1
GEN_EXIT=$?
echo "[$(ts)] END PLINDER pretrain exit=$GEN_EXIT (last $(grep -oE 'epoch: [0-9]+' "$LOG/${EXP}.log" | tail -1))"
[ "$GEN_EXIT" -ne 0 ] && echo "[$(ts)] NOTE: pretrain FAILED — resuming denoiser anyway per request"

resume_denoiser
echo "[$(ts)] CHAIN COMPLETE"
