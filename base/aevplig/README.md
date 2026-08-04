# AEV-PLIG baseline (standalone)

Architecture-only reproduction of **AEV-PLIG** — Valsson, Warren, Magarkar,
Biggin, Deane, *"Narrowing the gap between machine learning scoring functions and
free energy perturbation using augmented data"*, **Communications Chemistry 2025**
([nature](https://www.nature.com/articles/s42004-025-01428-y),
[code](https://github.com/oxpig/AEV-PLIG)). Its original baseline is **PLIG**
(Moesser et al., bioRxiv 2022) — same GAT graph net, AEV-PLIG only swaps PLIG's
interaction-count node features for Atomic Environment Vectors.

Run on **our** frozen `lp_edrscc_v2` PDBbind split (Kd/Ki), as an external
supervised baseline for the VoxBind affinity task — the same role as
`hbgsa_baseline/`. Self-contained: reads only the local PDBbind data under
`voxbind/dataset/data/pdbbind/` + the frozen split in `voxbind/splits/`; runs in
its own conda env (`aevplig`).

## "Architecture only" (like HBGSA)

We take the **model + featurisation** and ignore the paper's headline contribution
(augmenting training with semi-synthetic BindingNet/BindingDB data). The paper's
hard-coded targetdiff structure paths don't exist on this server, so structure
resolution is repointed to the local repo tree.

## Method

- **Graph** — nodes = ligand heavy atoms; edges = covalent bonds (bidirectional),
  `edge_attr` = bond-type one-hot `[single, aromatic, double, triple]` (4-d).
- **Node features (367-d)** = 15 atom descriptors `[atom-symbol one-hot(10),
  #heavy-nbrs, #H, explicit-valence, aromatic, in-ring]` ++ **352-d radial AEV**
  (22 PDB atom types × 16 ANI-2x radial shifts, `Rc = 5.1 Å`). The **protein
  enters ONLY through these per-ligand-atom radial AEVs** (ligand atoms encoded
  as carbon; protein atoms typed by `data/PDB_Atom_Keys.csv`). No protein tower.
- **Net** `GATv2Net` (7.55M params) — 5× `GATv2Conv(256, heads=3, edge_dim=4)` +
  BatchNorm + LeakyReLU → `[global_max ‖ global_mean]` (1536) → MLP
  `1536→1024→512→256→1`.
- **Training** — MSE on StandardScaler-normalised pK; Adam lr 1.23e-4, batch 128,
  200 ep; best ckpt by 8-epoch rolling val-Pearson. 3 seeds {0,1,2}; ensemble =
  mean of per-seed predictions (paper uses 10).

## Result — `lp_edrscc_v2` (train/val/test = 3843/817/1319)

| size | | Test Pearson r | Test Spearman ρ | Test RMSE |
|---|---|---|---|---|
| **7.55M** (paper) | per-seed (mean ± std, n=3) | 0.522 ± 0.019 | 0.492 ± 0.019 | 1.617 ± 0.004 |
| **7.55M** (paper) | 3-model ensemble | **0.584** | **0.550** | **1.480** |
| **40.15M** (scaled) | per-seed (mean ± std, n=3) | 0.494 ± 0.020 | 0.472 ± 0.017 | 1.675 ± 0.053 |
| **40.15M** (scaled) | 3-model ensemble | 0.565 | 0.537 | 1.512 |

For context on this split: coords-only PLINDER ViT probe ρ 0.605, C+D+G ChannelViT
ρ 0.637 (best), HBGSA 3.06M ρ 0.546, EGNN ρ 0.533, EGNN+TargetDiff ρ 0.579.
AEV-PLIG single-model is the weakest structural baseline here; its 3-model
ensemble lands between HBGSA and EGNN+TargetDiff — still below the frozen density
probe. (8 of 5987 complexes dropped: unreadable mol2.)

**Parameter scaling 7.55M → 40M** (hidden 256→512, 5→8 GNN layers, MLP
`[1024,512,256]`→`[1536,640,256]`; `--hidden_dim 512 --head 3
--number_GNN_layers 8 --mlp_dims 1536 640 256`) **regresses** (per-seed ρ
0.492→0.472, ensemble ρ 0.550→0.537). Naive scaling hurts in this data-limited
regime (3.8k train) — same pattern as HBGSA-40M and the dim-768 ChannelViT encoder.

## Reproduce

```bash
PY=/home/shpark/.conda/envs/aevplig/bin/python
cd aevplig_baseline/src
$PY build_graphs.py                                  # -> graphs/aevplig_edrscc_graphs.pickle
CUDA_VISIBLE_DEVICES=<gpu> $PY train_edrscc.py --seeds 0 1 2 --epochs 200
#   -> results/aevplig_edrscc.json  + results/aevplig_edrscc_preds.csv
```

## Layout

```
src/
  model_defs.py     GATv2Net           (vendored, unchanged)
  helpers.py utils.py                  (vendored: metrics, PyG GraphDataset)
  torchani_mod/     modified AEVComputer (vendored)
  build_graphs.py   AEV graph builder  (adapted: local paths, lp_edrscc_v2)
  train_edrscc.py   split+label glue, 3-seed train + ensemble report
data/PDB_Atom_Keys.csv                 22 ANI-2x protein atom types
graphs/  results/  logs/  output/
```
