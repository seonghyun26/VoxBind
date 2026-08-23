#!/usr/bin/env bash
# Wait for ONE pretraining checkpoint (checkpoint_e{EP:04d}.pth.tar) to appear, then probe it on
# the 6 campaign test sets — LP full + CL1/CL12/CL123 leak tiers + lba30/lba60 protein-ID splits —
# with 5 seeds, both the MSE and MSE+corr recipes. Features are extracted ONCE per checkpoint
# (all 6 splits reuse the cached feats via --allow_stale_features). Prints an aggregated table.
# Runs under bash (shebang) so $PY word-splits and arrays behave; safe from the zsh Bash-tool.
#   bash scripts/watch_probe_ckpts.sh <RUN_NAME> <EPOCH> [GPU=1]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
NAME="${1:?run name (exps/<NAME>)}"; EP="${2:?epoch}"; GPU_HINT="${3:-auto}"
# 4th arg = probe condition. CDG encoders → atomblob_density_gradmag (default);
# coords-only encoders (input_mode=atomblob, n_in=11) → atomblob.
COND_ARG="${4:-atomblob_density_gradmag}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
PYBIN=/home/shpark/.conda/envs/voxbind/bin/python

# Shared cluster: GPU occupancy shifts under us (other users + our own 4-7 training run).
# Pick, at probe time, a GPU with the most free memory among 0-3 (avoid the 4-7 training lane),
# requiring ≥18 GB free. Falls back to GPU_HINT if the query fails.
pick_gpu() {
  local best="" free_best=0
  while IFS=',' read -r idx total used; do
    idx=$(echo "$idx"|tr -d ' '); [ "$idx" -ge 4 ] && continue   # skip training lane
    local free=$(( ${total// /} - ${used// /} ))
    if [ "$free" -gt "$free_best" ]; then free_best=$free; best=$idx; fi
  done < <(nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits)
  if [ -n "$best" ] && [ "$free_best" -ge 18000 ]; then echo "$best"; else echo "${GPU_HINT/auto/0}"; fi
}
EP4=$(printf '%04d' "$EP")
CKPT="exps/$NAME/checkpoint_e${EP4}.pth.tar"
COND="$COND_ARG"
# 5-set panel: FULL + CL123 (leak-proof) + lba30 + lba60 (protein-ID<30/<60) via 01c,
# plus CASF-2016 clean-92 (honest: 0 exact overlap with lp_edrscc_v2 train/val AND with
# PLINDER-v2 pretraining) via test/probe_casf_100m_mask075.py.
SPLITS=(lp_edrscc_v2 lp_edrscc_v2_cl123 lba30 lba60)

echo "[watch] $NAME e$EP → waiting for $CKPT ..."
for _ in $(seq 1 400); do [ -f "$CKPT" ] && break; sleep 120; done
[ -f "$CKPT" ] || { echo "[watch] TIMEOUT — no $CKPT"; exit 1; }
sleep 15  # let the trainer finish flushing the file
GPU=$(pick_gpu)
echo "[extract] features e$EP on gpu $GPU (auto-picked; hint=$GPU_HINT)"
CUDA_VISIBLE_DEVICES="$GPU" nice -n 19 "$PYBIN" dataset/01c_pdbbind_probe.py features \
  --condition "$COND" --voxel_version v5 --epoch "$EP" --tag "$NAME" \
  --exp_dir "exps/$NAME" --device cuda:0 --num_workers 0 || { echo "extract FAILED"; exit 1; }

for split in "${SPLITS[@]}"; do
  for loss in mse "mse+corr"; do
    suf=mse; aux=(); [ "$loss" = "mse+corr" ] && { suf=msecorr5; aux=(--aux_weight 5); }
    CUDA_VISIBLE_DEVICES="$GPU" nice -n 19 "$PYBIN" dataset/01c_pdbbind_probe.py probe \
      --conditions "$COND" --epoch "$EP" --voxel_version v5 --split "$split" \
      --feature_tag "$NAME" --exp_dir "exps/$NAME" --allow_stale_features --seeds 5 \
      --probe_loss "$loss" "${aux[@]}" --device cuda:0 --no_wandb \
      --tag "${NAME}_e${EP}_${split}_${suf}" >/dev/null 2>&1 \
      && echo "  ok  $split $suf" || echo "  FAIL $split $suf"
  done
done

# CASF-2016: reuse the just-cached bundle (CASF ⊂ PDBbind → its features are already in it).
# Trains head on lp_edrscc_v2 train, predicts CASF; we read the clean-92 subset downstream.
CASF_FEAT="dataset/data/pdbbind/features/${COND}_e${EP}_v5_${NAME}.pt"
for loss in mse "mse+corr"; do
  CUDA_VISIBLE_DEVICES="$GPU" nice -n 19 "$PYBIN" test/probe_casf_100m_mask075.py \
    --feat "$CASF_FEAT" --model "${NAME}_e${EP}" --loss "$loss" >/dev/null 2>&1 \
    && echo "  ok  casf2016 $loss" || echo "  FAIL casf2016 $loss"
done

echo "===== e$EP RESULTS ($NAME) ====="
"$PYBIN" - "$NAME" "$EP" <<'PY'
import csv, glob, statistics as st, sys, os
import numpy as np
from scipy.stats import spearmanr, pearsonr
RES="dataset/data/pdbbind/results"; NAME, EP = sys.argv[1], sys.argv[2]
CASF_OUT="/home/shpark/prj-denovo/VoxBind/base/_casf"
SPL=[("FULL","lp_edrscc_v2"),("CL123","lp_edrscc_v2_cl123"),
     ("lba30","lba30"),("lba60","lba60")]
def agg(split, suf):
    corr = "loss-mse-corr-w5_" if suf=="msecorr5" else ""
    pat=f"{RES}/probe_results_e{EP}_v5_{split}split_{corr}{NAME}_e{EP}_{split}_{suf}.csv"
    fs=glob.glob(pat)
    if not fs: return None
    rows=list(csv.DictReader(open(fs[0])))
    g=lambda k:[float(r[k]) for r in rows if r.get(k) not in (None,"")]
    return st.mean(g("test_spearman")), st.mean(g("test_pearson")), st.mean(g("test_rmse")), len(rows)
def agg_casf(suf):
    # CASF clean-92 = homology-clean subset; read per-seed preds, filter to clean pids.
    try:
        clean=set(r['pid'].strip().lower() for r in csv.DictReader(open("splits/casf2016_clean.csv")))
    except FileNotFoundError:
        return None
    tag = f"{NAME}_e{EP}" if suf=="mse" else f"{NAME}_e{EP}_corr5"
    fs=sorted(glob.glob(f"{CASF_OUT}/{tag}_casf2016_preds_seed*.csv"))
    if not fs: return None
    rho,r,rm=[],[],[]
    for f in fs:
        rr=[x for x in csv.DictReader(open(f)) if x['pid'].strip().lower() in clean]
        if len(rr)<3: continue
        y=np.array([float(x['y']) for x in rr]); p=np.array([float(x['pred']) for x in rr])
        rho.append(spearmanr(y,p).correlation); r.append(pearsonr(y,p)[0])
        rm.append(float(np.sqrt(np.mean((y-p)**2))))
    if not rho: return None
    return st.mean(rho), st.mean(r), st.mean(rm), len(rr)
print(f"{'split':11} | {'mse ρ':>7} {'mse r':>7} {'mse RMSE':>8} | {'msecorr ρ':>9} {'mc r':>7} {'mc RMSE':>8}")
for lab,sp in SPL:
    a=agg(sp,"mse"); b=agg(sp,"msecorr5")
    af=f"{a[0]:.4f} {a[1]:.4f} {a[2]:8.4f}" if a else "   --      --       --  "
    bf=f"{b[0]:.4f} {b[1]:.4f} {b[2]:8.4f}" if b else "   --      --       --  "
    print(f"{lab:11} | {af} | {bf}")
a=agg_casf("mse"); b=agg_casf("msecorr5")
af=f"{a[0]:.4f} {a[1]:.4f} {a[2]:8.4f}" if a else "   --      --       --  "
bf=f"{b[0]:.4f} {b[1]:.4f} {b[2]:8.4f}" if b else "   --      --       --  "
print(f"{'CASFclean':11} | {af} | {bf}   (3-seed, clean-92)")
PY
echo "[watch] e$EP DONE"
