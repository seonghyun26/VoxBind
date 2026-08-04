# GET baseline (trained on split) — lp_edrscc_v2

Train **GET** (Generalist Equivariant Transformer; Kong et al., **ICML 2024**,
[arXiv 2306.01474](https://arxiv.org/abs/2306.01474);
[repo](https://github.com/THUNLP-MT/GET)) on our `lp_edrscc_v2` split as a supervised
structure-based affinity baseline. GET is an E(3)-equivariant transformer with bilevel
(block/atom) attention over protein+ligand atoms; its PDBBind benchmark task regresses pK.

## Method

Reuses GET's own pipeline. `_edrscc/src/make_data.py` converts each complex into GET's
block format with GET's own utilities (`pdb_to_list_blocks` on our `{pid}_protein.pdb`,
`mol2_to_blocks` on `{pid}_ligand.mol2`, 8 Å `blocks_interface` pocket, `blocks_to_data`),
labelled with `neglog_aff = pK` and bucketed by the frozen split → `datasets/edrscc/{train,valid,test}.pkl`
(3850 / 816 / 1320; 1 dropped, no interface). Trained PL-only (`task=PDBBind`, no
protein-protein `train_set2`) via the stock `scripts/train/train.sh`; evaluated with the
stock `inference.py` + `evaluate.py` on the best-validation checkpoint.

Config (`_edrscc/edrscc_get.json`): GET, hidden 64, 3 layers, 4 heads, n_rbf 32,
k_neighbors 9, atom-level (n_channel 1), **0.72M params**, lr 1e-4, 20 epochs, dynamic
batch (1500 vertices/GPU). 3 seeds {0,1,2}.

## Result — `lp_edrscc_v2` test (1320), 3-seed mean ± std

| | Pearson r | Spearman ρ | RMSE |
|---|---|---|---|
| **GET** | **0.596 ± 0.005** | **0.591 ± 0.005** | **1.428 ± 0.008** |

Per-seed: r 0.598 / 0.600 / 0.591. Strong — above HBGSA (ρ 0.546), EGNN (0.533) and
≈ EGNN+TargetDiff (0.579); the best supervised structure-based baseline on this split,
still below the frozen C+D+G density probe (ρ 0.637) and ≈ coords-only ViT (ρ 0.605).
Added as a Table-1 row + bar in `260701_meeting.html`.

## Reproduce

```bash
PY=/home/shpark/.conda/envs/get/bin/python
$PY _edrscc/src/make_data.py                                   # -> datasets/edrscc/*.pkl
GPU=<n> bash scripts/train/train.sh _edrscc/edrscc_get.json    # train
CUDA_VISIBLE_DEVICES=<n> python inference.py --test_set datasets/edrscc/test.pkl \
    --task PDBBind --ckpt <best.ckpt> --gpu 0 --save_path preds.jsonl
python evaluate.py --predictions preds.jsonl                   # Pearson/Spearman/RMSE
```

## Env notes (`get` conda env)

Conda env-file solve fails (biopython repo-priority); built a lean **pip** env instead:
python 3.9, **torch 1.13.1+cu117**, matched `torch-scatter/sparse/cluster ==…+pt113cu117`
wheels, atom3d, biopython, rdkit-pypi, numpy<1.24. **Do not install e3nn** — it pulls
torch 2.x and breaks the pt113 pyg wheels (e3nn is only needed for GET's MACE model, not
the GET model). The config's `test_set`/`out_dir` keys are for `evaluate.py`, not
`train.py` (the training config omits them).
