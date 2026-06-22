#!/bin/bash
# run_edrscc_chain.sh — retrain HBGSA (3.06M + ~40M) on the NEW PDBbind ED+RSCC split
# (LP ∩ ED ∩ ligand-RSCC>=0.8 ∩ pocket-RSCC>=0.8), via train.py --restrict_ids.
# Builds the H-bond cache for the full manifest first (skips the ~5k already done).
set -u
HB=/home/shpark/prj-denovo/VoxBind/hbgsa_baseline
PY=/home/shpark/.conda/envs/hbgsa/bin/python
ALLOW=/home/shpark/prj-denovo/VoxBind/voxbind/dataset/data/pdbbind/ed_rscc_pids.txt
cd "$HB/src" || exit 1
ts(){ date "+%F %T"; }

echo "[$(ts)] === HBGSA ED+RSCC retrain ==="
echo "[$(ts)] [1/3] H-bond cache build (full manifest, skips existing)"
$PY hbonds.py 2>&1 | tail -5

echo "[$(ts)] [2/3] train 3.06M (default config) on ED+RSCC restrict"
$PY train.py --restrict_ids "$ALLOW" --tag edrscc_3p06m --seeds 0,1,2 --epochs 150 2>&1 | tail -8

echo "[$(ts)] [3/3] train ~40M (width-scaled) on ED+RSCC restrict"
$PY train.py --restrict_ids "$ALLOW" --tag edrscc_40m --seeds 0,1,2 --epochs 150 \
    --emb_dim 416 --d_model 416 --gcn_hidden 416 --pocket_hidden 416 --head_hidden 416 \
    --conv_channels 512 2>&1 | tail -8

echo "[$(ts)] === HBGSA ED+RSCC retrain COMPLETE → results/hbgsa_results_edrscc_*.csv ==="
