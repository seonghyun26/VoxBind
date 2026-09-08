# VoxBind — Electron-density voxel features for structure-based drug design

This repository grew out of [VoxBind](https://arxiv.org/abs/2405.03961) (a
pocket-conditional voxel-diffusion generative model) into a broader study of
**electron-density voxel representations** for structure-based drug design (SBDD).
The central question: does adding real X-ray **electron density** — on top of the
usual atom coordinates — give a protein-pocket encoder a better view of binding?

We voxelize each pocket–ligand complex into three channel groups and pretrain a
3D-ViT encoder with masked autoencoding (MAE), then evaluate it across three tasks.

- **C** — atom **c**oordinates (atom-type "blob" channels)
- **D** — electron **d**ensity (from experimental maps / MTZ → gemmi FFT)
- **G** — density **g**radient magnitude

The headline encoder is **`CDG-v2`** (C+D+G channels, ChannelViT, MAE-pretrained
on a PLINDER ligand-matched density corpus). "Ours" in the reports refers to it.

## Three tasks

| Task | Question | Where |
|---|---|---|
| **1 · Affinity** | Do density features improve binding-affinity prediction (frozen-encoder probe / fine-tune)? | `results/task1-affinity/`, `results/reports/results.html` |
| **2 · Drug design** | Does density-conditioning improve *de novo* ligand generation (VoxBind + density)? | `results/task2-drugdesign/`, `results/reports/results_drug_design.html` |
| **3 · Macrocycles** | Density-conditioned generation for macrocyclic peptides (FuncBind, pilot) | `results/task3-mcp/`, `results/reports/results_mcp.html` |

Canonical affinity benchmark: **`lp_edrscc_v2`** — LP-PDBBind ∩ electron-density-available
∩ (ligand & pocket RSCC ≥ 0.8), Kd/Ki only, 3850 / 817 / 1320 split (see `voxbind/splits/`).
Task 1 compares our encoders against ~20 baselines (ProFSA, GET, DSMBind, AEV-PLIG,
HBGSA, CheapNet, BindNet, IPDiff/IPNet, GeoSSL, Nesso, Boltz-2, DeepDTA, MolTrans, …).

## Repository layout

```
voxbind/            main code
├── train.py                  original VoxBind denoiser training (DDP)
├── train_density.py          density-encoder MAE pretraining (mae / denoise / chamae methods)
├── sample.py / sample_from_file.py   walk-jump ligand sampling
├── models/                   density_vit, density_cha_mae, voxbind, unet3d, urepa, …
├── configs/                  Hydra configs (dset/model/mae/experiment groups)
├── dataset/                  voxelization + density/gradmag/PLINDER build pipeline
├── test/                     probes, CASF eval, cliff analysis, benchmarks
├── splits/                   frozen, hash-verified train/val/test manifests
├── model_zoo/                pretrained encoders (CDG_v2, C_v2, champion, …; Dropbox-backed)
└── scripts/                  parameterized launchers (pretrain / downstream / sample / watch)

base/               self-contained external baselines (one folder per model; see base/README.md)
results/            per-task metrics + rendered HTML reports (Dropbox-backed; see results/README.md)
notebook/           analysis notebooks + HTML reports (notebook/html/, YYMMDD_ prefixed)
examples/           sample pockets (8UWP, 6AU3) for sample_from_file.py
figures/            paper figures
```

Large artifacts (model weights, raw voxel data, per-sample eval) live outside git
and sync to Dropbox — `voxbind/model_zoo/`, `results/`, and per-baseline outputs
each carry `dropbox_push.sh` / `dropbox_pull.sh`.

## Quickstart

### 1. Environment
```bash
mamba env create -f env.yaml
conda activate voxbind
pip install -e .
```

### 2. Data
The original CrossDocked pipeline for the generative task:
```bash
cd voxbind/dataset && python preprocess_crossdocked.py     # produces train_data.pt / test_data.pt
```
The density/affinity pipeline (electron-density crops, gradmag, PLINDER corpus)
is built by the numbered `dataset/00*_*.py` stages — see `voxbind/dataset/build/`
and the notes in `voxbind/EXPERIMENTS.md` / `notebook/html/experiments.html`.

### 3. Pretrain a density encoder (MAE)
```bash
cd voxbind
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_density.py \
  --config-name config_train_atomblob_density_gradmag_channelvit_mae_40m_plinder_otf_mask050
# or use scripts/03_pretrain.sh / scripts/pretrain.sh with CLI + Hydra passthrough
```

### 4. Evaluate on affinity (frozen-encoder probe)
```bash
cd voxbind
python -m test.<probe>  # see scripts/04_probe.sh; results land in results/task1-affinity/
```

### 5. Generative sampling (original VoxBind)
Train the denoiser and sample ligands for a pocket:
```bash
cd voxbind
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py smooth_sigma=0.9
python sample.py pretrained_path=exps/exp_sig0.9 wjs.split=val wjs.n_samples_per_pocket=10 wjs.n_targets=100
# de novo from a provided pocket:
python sample_from_file.py pretrained_path=exps/exp_sig0.9/ n_samples=20
```
See `configs/config_sample*.yaml` for all sampling options and `examples/` for the
8UWP / 6AU3 demo pockets.

## Upstream & citation

The generative core (walk-jump sampling, voxel denoiser) is from VoxBind:

```bibtex
@inproceedings{pinheiro2024voxbind,
  title     = {Structure-based drug design by denoising voxel grids},
  author    = {Pinheiro, Pedro O and Jamasb, Arian and Mahmood, Omar and Sresht, Vishnu and Saremi, Saeed},
  booktitle = {ICML},
  year      = {2024}
}
```

External baselines under `base/` retain their own upstream licenses and READMEs.

## License

Apache License 2.0 — see `LICENSE.txt`.
