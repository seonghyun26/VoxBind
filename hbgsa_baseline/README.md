# HBGSA baseline (standalone)

A from-scratch reimplementation of **HBGSA — Hydrogen Bond Graph with
Self-Attention** (arXiv 2604.23115) run on **our** LP-PDBbind `new_split`, as an
external supervised baseline for the VoxBind downstream affinity task.

This folder is **self-contained**: it imports nothing from `voxbind/` and reads
**only** the PDBbind data under `voxbind/dataset/data/pdbbind/` (structures +
`LP_PDBBind.csv` / `index.csv`). It runs in its own conda env (`hbgsa`).

## Why a separate baseline, and how to read the number

- HBGSA is a **fully-supervised, end-to-end** model trained on the affinity
  labels. The VoxBind downstream numbers are **frozen-encoder linear probes**.
  So HBGSA belongs in the table as an **external supervised reference**
  (≈ an upper bound for "how well can a dedicated supervised model do on this
  split"), *not* as a representation-quality probe. Compare accordingly.
- Same partition: it trains/val/tests on the LP-PDBbind `new_split` column —
  the exact partition all our probe/finetune experiments use.

## Method (full 4-branch)

The paper has **no public code** and omits several hyperparameters. We implement
all **four** branches, each → 128-d, concatenated → FC `512→128→64→1`:

| branch | paper name | encoder |
|---|---|---|
| **H-bond graph** (paper core) | `s_hb` | up to 20 H-bonds as nodes, 9-d feature `[protein-end xyz, ligand-end xyz, midpoint xyz]`, dynamic KNN (k=5) over midpoints, 2× `GCNConv` + residual, global max-pool |
| **protein sequence** | `v_seq` | per-residue physicochemical descriptors → 1D Transformer self-attention → masked mean |
| **binding pocket** | `v_pkt` | pocket residues (from `{pdb}_pocket.pdb`) → physicochemical descriptors → 1D conv (kernel 3) → masked max-pool |
| **SMILES** | `v_smi` | regex atom-level tokens → embedding → 1D Transformer self-attention → masked mean |

**Loss**: `SmoothL1 + 50·(1 − Pearson)` (paper hybrid loss, λ=50). The target pK
is standardized on the train split so the SmoothL1 term calibrates the absolute
scale (Pearson is affine-invariant); predictions are de-standardized before
metrics, so reported RMSE is in pK units.

### Deviations from the paper (documented)

1. **Pocket / seq / SMILES encoders** — paper uses dilated convs + self-attention
   for `v_seq`/`v_smi` and 1D conv (kernel 3) for `v_pkt`; we use plain
   self-attention for seq/SMILES and 1D conv for the pocket. All four branches
   are present (the pocket branch reads `{pdb}_pocket.pdb`).
2. **Sequence descriptors** — the paper says "40-dim physicochemical" but never
   publishes the set. We use a **validated** descriptor set covering the same
   categories (Kyte-Doolittle, Hopp-Woods, flexibility, surface-accessibility,
   Janin scales + charge / polarity / aromaticity / MW) rather than fabricating
   AAindex values; the branch projects it to the model width.
3. **H-bonds** — paper uses PyMOL `distance mode=2`; we use PyMOL for structure
   I/O + H-add, then apply the paper's stated criteria directly
   (d ≤ 3.5 Å, D–H···A angle ≥ 120°, polar N/O/S).
4. **Unstated hyperparameters** (optimizer, LR, epochs, batch) → sensible
   defaults: AdamW, lr 1e-3, batch 64, ≤150 epochs, early-stop on val Pearson.

## Layout

```
hbgsa_baseline/
  env/setup_env.sh     # builds the isolated `hbgsa` conda env
  src/
    config.py          # paths + hyperparameters
    manifest.py        # PDBbind manifest (join index.csv + LP_PDBBind.csv)
    hbonds.py          # PyMOL H-bond graph extraction + cache
    featurize.py       # sequence descriptors + SMILES tokenizer
    dataset.py         # torch Dataset + collate (manual KNN, no torch_cluster)
    model.py           # HBGSA 3-branch model + hybrid loss
    train.py           # train/eval on new_split → results CSV
  cache/hbonds/        # per-complex H-bond graphs (.npz)
  results/             # hbgsa_results_<tag>.csv + per-seed preds
  logs/
```

## Run

```bash
PY=/home/shpark/.conda/envs/hbgsa/bin/python

# 1. build env (one-time)
bash env/setup_env.sh

# 2. build H-bond cache (~5 min, single process; supports --shard/--nshards)
$PY src/hbonds.py

# 3. train + eval on the full new_split (3 seeds)
$PY src/train.py --tag full_newsplit --seeds 0,1,2 --epochs 150

#    or the CL1-clean subset:
$PY src/train.py --tag cl1_newsplit --seeds 0,1,2 --cl1_only
```

`train.py` reports test metrics (Pearson R, Spearman ρ, RMSE, MAE) on the **full**
test set and the nested **CL1-clean** subset from a single trained model. Pass
`--probe_pids <file>` (one pdb_id per line) to also score the exact complex set
of a specific VoxBind probe run for a pdb-for-pdb comparison.

## Data universe (complexes with structures, after dropping covalent + unresolved SMILES)

| split | full new_split | CL1-clean |
|---|---:|---:|
| train | 2931 | ~2275 |
| val   | 623  | ~552  |
| test  | 1433 | 1341  |

H-bond cache: 4987 complexes, 0 extraction errors, ~6% with no detected H-bond
(handled by a zero-node fallback).
