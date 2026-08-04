# ProFSA baseline (pretrained encoder + probe) — lp_edrscc_v2

Evaluate **ProFSA** (Gao et al., **ICLR 2024**, [arXiv 2310.07229](https://arxiv.org/abs/2310.07229);
[repo](https://github.com/bowen-gao/ProFSA)) on our `lp_edrscc_v2` split. ProFSA self-supervised
pretrains a Uni-Mol pocket encoder by aligning protein-fragment pockets to pretrained small-molecule
representations (fragment-surroundings alignment, ~5M pseudo-complexes).

Following the paper's LBA protocol (`experiment=lba30`): load the released ProFSA checkpoint as a
**frozen** encoder, then train a regression head (DrugCLIPReg) on the affinity labels — i.e. a
pretrained-encoder + probe, the same shape as VoxBind's own C / C+D+G probes.

## Method

`_edrscc/src/make_lmdb.py` builds our complexes into ProFSA's LBA LMDB schema (7 keys:
`atoms`, `coordinates(1,N,3)`, `pocket_atoms`, `pocket_coordinates(M,3)`, `pocket`, `smi`, `label`)
— ligand heavy atoms from SDF/mol2, pocket heavy atoms from `{pid}_pocket.pdb`; the loader removes H,
crops the pocket to the 256 nearest atoms, and normalizes coordinates. Then the stock `train.py
experiment=lba30` runs with our data + `last.ckpt` as frozen `pretrained_weights`, 50 epochs, best
checkpoint by `val/RMSE`, auto-test on test.

LMDBs: train 3850 / valid 817 / test 1320 (0 failures).

## Result — `lp_edrscc_v2` test (1320), single seed

| | Pearson r | Spearman ρ | RMSE |
|---|---|---|---|
| **ProFSA** | **0.626** | **0.599** | **1.508** |

Best-val RMSE 1.673 @ epoch 48. Strongest structural baseline on Spearman here — above GET
(ρ 0.591), HBGSA (0.546), EGNN+TargetDiff (0.579); ≈ coords-only ViT (ρ 0.605), below the frozen
C+D+G density probe (ρ 0.637). Single seed (paper's LBA protocol); the other table rows are 3-seed.

**Caveat:** ProFSA's encoder was self-supervised on PDB structures (pre-2021), so there may be mild
pocket overlap with our test complexes — though the affinity head is trained only on our train split.

## Reproduce

```bash
PY=/home/shpark/.conda/envs/profsa/bin/python
$PY _edrscc/src/make_lmdb.py                      # -> data/dataset/edrscc/{train,valid,test}.lmdb (+dicts)
P=$(pwd); CKPT=$P/data/log/train/profsa/profsa_release/checkpoints/last.ckpt
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=<gpu> $PY train.py experiment=lba30 logging=csv '~callbacks.rich' \
  dataset.dataset_cfg.train.data_dir=$P/data/dataset/edrscc model.cfg.data_dir=$P/data/dataset/edrscc \
  model.cfg.pretrained_weights=$CKPT model.cfg.dropout=0.5 optim.lr=0.0002 scheduler.num_warmup_steps=200 \
  trainer.devices=1 run_test=true +trainer.enable_progress_bar=false hydra.run.dir=$P/_edrscc/runs/lba30
#  test metrics in _edrscc/logs/full_train.log ; result -> results/profsa_edrscc.json
```

## Env notes (`profsa` conda env)

torch 1.13.1+cu117 + **unicore** wheel (`unicore-0.0.1+cu117torch1.13.1`), lightning 2.0.9,
pandas==1.5.3 (the LBA LMDBs were pickled with pandas 1.x; 2.x can't unpickle), transformers 4.30.2,
hydra-core 1.3.2, zstandard, einops, biotite/biopython, rdkit. The weights bundle (`profsa.tar.gz`,
~7 GB) is from Google Drive (gdown headless was access-denied — downloaded manually). Configs use
Docker mount paths (`/data`, `/log`) — overridden to absolute repo paths via Hydra. Remove the
`rich` callback (`~callbacks.rich`) — RichProgressBar crashes in a non-TTY shell.
