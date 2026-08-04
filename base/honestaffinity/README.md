# HonestAffinity-Pocket baseline

Reimplementation of **HonestAffinity-Pocket** (arXiv 2606.03422, *HonestAffinity:
Leak-Aware Evaluation of Protein and Pocket Priors*), the PLM-augmented headline variant,
evaluated on the frozen VoxBind `lp_edrscc_v2*` splits. No official code/checkpoints are
available (anonymous snapshot); this is implemented faithfully **from the paper spec**.

## Model (`src/model.py`)

Supervised protein+ligand affinity regressor, ~3.0M trainable params (ESM is frozen and
precomputed, not counted):

- **Protein branch (PLM-augmented):** frozen **ESM-2-650M** (`esm2_t33_650M_UR50D`)
  per-residue embeddings `[Lp,1280]` → `Linear(1280→256)`; a **learned binary
  pocket-position marker** `nn.Embedding(2,256)` is *added* to residues in the pocket;
  then a **multi-scale 1D-conv** encoder (parallel `Conv1d` kernels `{1,3,5,7}`, output
  channels `{32,32,64,128}`=256, concat → `Linear(256→256)` + residual) and a **single**
  `nn.TransformerEncoderLayer` (d=256, nhead=8→dk=16, ff=1024, batch_first). → `Zprot`.
- **Ligand branch (SMILES):** char-level **53-token learned vocab** → `Embedding(53,256)`;
  same multi-scale conv + residual + single Transformer layer → `Zsmi`.
- **Compatibility + head:** `S = (Zprot @ Zsmiᵀ)/√d` `[Lp,Ll]`; cross-attention pooling
  (see below) → 1024-d vector; MLP head `1024→256→64→1` with **PReLU** + **dropout 0.5**.

### Pooling (resolved ambiguity)
The spec offers cross-attention pooling as the "faithful choice". Implemented exactly as
described: `softmax(S, dim=lig)` gives each protein residue a ligand-context vector
(`prot_ctx`); `softmax(S, dim=prot)` gives each ligand token a protein-context vector
(`smi_ctx`). The fixed 1024-d pooled vector is
`[prot_mean, mean(prot_ctx), smi_mean, mean(smi_ctx)]` (each 256). All softmaxes and means
are padding-masked.

## Data prep (`src/data_prep.py`)

For each pid in a split we need (protein sequence, pocket-residue mask, ligand SMILES, pK):

- **Sequence + pocket mask are derived from the structure** for self-consistent indexing:
  `{pid}_protein.pdb` → ordered residue list → 1-letter sequence; `{pid}_pocket.pdb`
  (the upstream binding-site residues near the ligand, ~10 Å, same pocket the profsa /
  dsmbind baselines use) → the set of residue keys marked `pocket==1`. This guarantees the
  learned marker aligns perfectly with the ESM per-residue embedding.
- **SMILES** from `LP_PDBBind.csv` `smiles` (structure-SDF canonical SMILES fallback).
- **pK label** from `LP_PDBBind.csv` `value` (same source as every other baseline).
- **Long sequences** (>1000 residues, ~6% — dimers / large complexes) are cropped to a
  1000-residue window chosen to cover the most pocket residues, so ESM-2 (~1022 tokens)
  stays in budget; pocket residues outside the window (rare) are dropped from the marker.
- **ESM cache:** per-residue embeddings are computed once and cached as float16 `[L,1280]`
  keyed by sequence hash (`cache/esm/<hash>.pt`), so identical proteins across pids/splits
  share one file. All 4 splits share **4297 unique** sequences (4.3 GB total).

All 4 splits build **100% of pids** (0 failures).

## Env

Uses the existing **`dsmbind`** conda env
(`/home/shpark/.conda/envs/dsmbind/bin/python`) — it has `fair-esm` + `rdkit` + `torch`
1.13.1 (cuda) + `scipy`. No new env needed.

## Usage

```bash
PY=/home/shpark/.conda/envs/dsmbind/bin/python
cd base/honestaffinity/src

# 1) data prep + ESM cache (once per split; ESM cache is shared, so after v2 the CL
#    subsets need 0 new ESM computations)
CUDA_VISIBLE_DEVICES=<gpu> $PY data_prep.py --split lp_edrscc_v2 --gpu 0

# 2) train 3 seeds, full schedule, write results/results_<split>.json
CUDA_VISIBLE_DEVICES=<gpu> $PY train.py --split lp_edrscc_v2 --seeds 0 1 2 --gpu 0
```

### Full run (4 splits × 3 seeds) — launch command

Data prep + ESM cache are **already done for all 4 splits**. To launch the full training:

```bash
PY=/home/shpark/.conda/envs/dsmbind/bin/python
cd base/honestaffinity/src
GPU=<gpu>   # e.g. 4,5,6,7 — GPUs 0-3 are off-limits
for SPLIT in lp_edrscc_v2_cl123 lp_edrscc_v2_cl12 lp_edrscc_v2_cl1 lp_edrscc_v2; do
  CUDA_VISIBLE_DEVICES=$GPU $PY train.py --split $SPLIT --seeds 0 1 2 --gpu 0 \
    2>&1 | tee ../logs/train_$SPLIT.log
done
```

(Or run `bash run_full.sh <gpu>`.) Rough per-seed time on one GPU: cl123 ~10-15 min,
cl12/cl1 ~15-23 min, v2 ~20-30 min (80 epochs, early-stop patience 20). Whole 4×3 sweep
≈ 3-4 h on a single GPU.

## Outputs

`results/results_<split>.json`:
```json
{"pearson":{"mean":..,"std":..}, "spearman":{"mean":..,"std":..},
 "rmse":{"mean":..,"std":..}, "seeds":[0,1,2], "n_test":N, "per_seed":[...], "config":{...}}
```

## Files

- `src/data_prep.py`   — build per-pid records + cache ESM-2-650M embeddings
- `src/smiles_tokenizer.py` — fixed 53-token char-level SMILES vocab
- `src/model.py`       — HonestAffinity-Pocket network
- `src/dataset.py`     — torch Dataset + padded collate
- `src/train.py`       — 3-seed train / early-stop / test, writes results JSON
- `run_full.sh`        — launch all 4 splits × 3 seeds
- `cache/`             — `records_<split>.json` + `esm/<hash>.pt` (gitignored)
- `results/`           — `results_<split>.json` (committed)
