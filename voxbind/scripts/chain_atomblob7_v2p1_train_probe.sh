#!/usr/bin/env bash
# v2.1 atomblob7: wait for v2.1 boxes → stop denoiser → train 100ep → probe lp_edrscc_v2 →
# resume denoiser. All steps inline (no cross-process pgrep wait → no self-match). atomblob7
# probes the EXISTING 7-ch voxels_ligvdw cache (no re-voxelize).
#   setsid nohup bash scripts/chain_atomblob7_v2p1_train_probe.sh > log/chain_atomblob7_v2p1.log 2>&1 &
set -uo pipefail
VOX=/home1/irteam/VoxBind/voxbind
B=/opt/conda/envs/voxbind/bin
LOG=$VOX/log
CFG=config_train_atomblob7_density_gradmag_channelvit_mae_40m_plinder_v2p1_box
EXP=260701_plinder_v2p1_box_atomblob7_cdg_channelvit_full_pretrain
RD=$VOX/dataset/data/pretrain/xray_resample_plinder_v2p1
MANIFEST=$RD/train_manifest.npz
BOXMETA=$RD/box96_meta.json
COND=atomblob_density_gradmag
TAG=atomblob7_v2p1_cdg_cvit
DENOISER_CK=$VOX/exps/exp_sig1.0+prefetch_factor16_wjs.n_targets0/checkpoint.pth.tar
BSZ=${BSZ:-32}; NWORK=${NWORK:-16}
LOCK=$LOG/chain_atomblob7_v2p1.lock
export PATH=$B:${PATH}
export CXX=$B/x86_64-conda-linux-gnu-g++ CC=$B/x86_64-conda-linux-gnu-gcc
export LD_LIBRARY_PATH=/opt/conda/envs/voxbind/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$VOX" || exit 1
ts(){ date "+%Y-%m-%d %H:%M:%S"; }; log(){ echo "[$(ts)] $*"; }
exec 9>"$LOCK"; flock -n 9 || { log "another atomblob7 chain running — exit"; exit 1; }
resume_denoiser(){ [ -f "$DENOISER_CK" ] && { setsid nohup bash "$VOX/scripts/chain_sig1.0_keepgoing.sh" > "$LOG/chain_sig1.0_keepgoing.log" 2>&1 & log "denoiser resume re-armed."; }; }

DEADLINE=$(( $(date +%s) + 14*3600 ))
wait_for(){ local what="$1"; shift; log "waiting for $what ..."; until "$@"; do [ "$(date +%s)" -gt "$DEADLINE" ] && { log "ABORT: timeout on $what"; exit 1; }; sleep 30; done; log "  $what ready"; }

# ── 1) wait for v2.1 boxes (meta written only at the end of 05_make_boxes) ────
wait_for "v2.1 boxes" test -f "$BOXMETA"
NPOS=$("$B/python" -c "import numpy as np;print(len(np.load('$MANIFEST',allow_pickle=True)['pdb_id']))")
SUBSET_N=$(( NPOS - 100 ))
log "v2.1 manifest positions=$NPOS → subset_n=$SUBSET_N"

# ── 2) stop denoiser, drain ──────────────────────────────────────────────────
pkill -9 -f '[c]hain_sig1.0_keepgoing.sh' && log "watcher killed" || log "no watcher"
if pgrep -f '[t]rain_ddp.py' >/dev/null; then pkill -9 -f '[t]rain_ddp.py'; log "denoiser stopped"; sleep 5; fi
for _ in $(seq 1 40); do U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|awk '{s+=$1}END{print s+0}'); [ "${U:-99999}" -lt 6000 ] && break; sleep 5; done
log "GPU drained (${U:-?} MiB)"

# ── 3) train to completion (inline) ──────────────────────────────────────────
RUNLOG="$LOG/${EXP}_$(date +%Y%m%d_%H%M%S).log"
log "LAUNCH atomblob7 train: 4-GPU bsz=$BSZ workers=$NWORK subset_n=$SUBSET_N → $RUNLOG"
CUDA_VISIBLE_DEVICES=0,1,2,3 "$B/torchrun" --standalone --nproc_per_node=4 \
  "$VOX/train_density.py" --config-name="$CFG" \
  bsz="$BSZ" accum_steps=1 num_workers="$NWORK" \
  dset.subset_n="$SUBSET_N" dset.subset_val_n=100 \
  exp_name="$EXP" hydra.run.dir="$VOX/exps/$EXP" > "$RUNLOG" 2>&1
RC=$?; log "atomblob7 train exit=$RC"
[ "$RC" -eq 0 ] || { log "TRAIN FAILED — see $RUNLOG"; resume_denoiser; exit 1; }

# ── 4) features + probe (lp_edrscc_v2, 3 seeds) — 7-ch uses EXISTING voxels_ligvdw ──
log "FEATURES (cond=$COND tag=$TAG) ..."
CUDA_VISIBLE_DEVICES=0 "$B/python" -u dataset/01c_pdbbind_probe.py features \
  --condition "$COND" --exp_dir "exps/$EXP" --epoch 99 --voxel_version v5 \
  --atom_source ligvdw --tag "$TAG" --device cuda --batch_size 24 > "$LOG/${TAG}_features.log" 2>&1
RC1=$?; log "features exit=$RC1"
if [ "$RC1" -eq 0 ]; then
  log "PROBE lp_edrscc_v2 (3 seeds) ..."
  CUDA_VISIBLE_DEVICES=0 "$B/python" -u dataset/01c_pdbbind_probe.py probe \
    --conditions "$COND" --exp_dir "exps/$EXP" --epoch 99 --voxel_version v5 \
    --feature_tag "$TAG" --tag "$TAG" --split lp_edrscc_v2 --seeds 3 --device cuda \
    > "$LOG/${TAG}_probe.log" 2>&1
  log "probe exit=$?"
else
  log "FEATURES FAILED — see $LOG/${TAG}_features.log"
fi

# ── 5) resume denoiser ───────────────────────────────────────────────────────
resume_denoiser
log "ATOMBLOB7 v2.1 CHAIN DONE. Results: dataset/data/pdbbind/results/probe_results_e99_v5_lp_edrscc_v2split_${TAG}.csv"
