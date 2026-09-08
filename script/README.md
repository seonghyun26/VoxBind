# VoxBind — shareable end-to-end pipeline

Self-contained scripts to run the **density-conditioned VoxBind** pipeline on a
fresh machine: environment → data → preprocessing → training → sampling →
evaluation. Written for a coworker who is starting from nothing (no data, no
weights, no envs).

The target model is **VoxBind + electron density**: a walk-jump voxel denoiser
whose pocket branch is conditioned on a *frozen* electron-density encoder
(**CDG_v2**). Evaluation reproduces the paper table — chemical/geometry metrics
+ AutoDock **Vina 1.2.2** docking + **PoseCheck** pose quality.

Each stage is a numbered script and runs independently:

| Step | Script | What it does | Runs on |
|------|--------|--------------|---------|
| 0 | `00_setup_env.sh` | Build the 3 conda envs + clone TargetDiff | CPU |
| 1 | `01_download_data.sh` | Pull the prepared data + weights copy | CPU/net |
| 2 | `02_preprocess_data.sh` | Verify / build `data_{train,test}.pt`, density crops | CPU |
| 3 | `03_train.sh` | Train base denoiser + density-fusion model | **GPU** |
| 4 | `04_sample.sh` | Walk-jump sample ligands (density-conditioned) | **GPU** |
| 5 | `05_evaluate.sh` | Chem + Vina docking + PoseCheck | CPU |

All shared config lives in `_common.sh` (paths, GPUs, env names, the data link).
Every knob is overridable from the environment — no script edits needed.

---

## 0. Prerequisites

- **GPU**: NVIDIA GPU(s), driver ≥ 520 (for CUDA 11.8). Multi-GPU is used
  automatically via `torchrun`; set `GPUS=0,1,2,3` to pick devices.
- **Disk**: ~150–250 GB (CrossDocked + density crops + weights + samples).
- **Conda**: Miniforge / Mamba / Micromamba on `PATH` (native install only).
- **Data link**: the prepared-copy source. Set **one** of these in `_common.sh`
  (or export before step 1):
  - `DATA_RCLONE_REMOTE` — an rclone remote path (Dropbox/Drive), **or**
  - `DATA_HTTP_URL` — a single `.tar` / `.tar.gz` / `.tar.zst` bundle URL.

> ⚠️ The placeholder `DATA_RCLONE_REMOTE` in `_common.sh` points at the lab
> Dropbox folder. **Replace it with the real share link** (or set `DATA_HTTP_URL`)
> before running `01`.

---

## Quickstart

### Option A — Docker (recommended, all-in-one)

The image bundles all three conda envs + TargetDiff. Data/weights stay outside
the image (mounted or pulled at runtime).

```bash
# build from the repo root
docker build -f script/Dockerfile -t voxbind:allinone .

# run with GPUs; mount the big dirs so they persist across containers
docker run --rm -it --gpus all --shm-size=16g \
  -v $PWD/voxbind/dataset/data:/workspace/VoxBind/voxbind/dataset/data \
  -v $PWD/voxbind/model_zoo:/workspace/VoxBind/voxbind/model_zoo \
  -v $PWD/voxbind/exps:/workspace/VoxBind/voxbind/exps \
  voxbind:allinone

# inside the container:
export DATA_RCLONE_REMOTE="dropbox:/…/VoxBind/share"   # or DATA_HTTP_URL=…
bash script/01_download_data.sh
bash script/02_preprocess_data.sh
GPUS=0,1,2,3 bash script/03_train.sh          # MODE=fusion (default)
EXP=voxbind_density_fusion GPUS=0,1,2,3 bash script/04_sample.sh
SAMPLE_DIR=voxbind/exps/voxbind_density_fusion/samples/samples_test100 \
  bash script/05_evaluate.sh
```

### Option B — native (bare metal / cluster)

```bash
bash script/00_setup_env.sh                    # builds voxbind, voxdock, moleval + clones TargetDiff
export DATA_RCLONE_REMOTE="dropbox:/…/VoxBind/share"
bash script/01_download_data.sh
bash script/02_preprocess_data.sh
GPUS=0,1,2,3 bash script/03_train.sh
EXP=voxbind_density_fusion GPUS=0,1,2,3 bash script/04_sample.sh
SAMPLE_DIR=voxbind/exps/voxbind_density_fusion/samples/samples_test100 \
  bash script/05_evaluate.sh
```

---

## Step details

### `00_setup_env.sh` — environments
Builds three conda envs and clones TargetDiff as a **sibling** of the repo.
Idempotent (skips what exists). `FORCE=1 … voxdock` rebuilds one env.

- `voxbind` — GPU pipeline (`env.yaml` + `pip install -e .`)
- `voxdock` — Vina **1.2.2** docking (python 3.8; pins reproduce the paper's
  absolute affinities — do not bump)
- `moleval` — PoseCheck **1.3.1** + PoseBusters + ProLIF (python 3.10)

### `01_download_data.sh` — prepared copy
Pulls the already-preprocessed data + weights so nothing is rebuilt from raw
maps. Sources: rclone remote (`DATA_RCLONE_REMOTE`) or one tar bundle
(`DATA_HTTP_URL`). Incremental — safe to re-run. It verifies the key artifacts
land in place. Expected layout under `voxbind/dataset/data/`:

```
crossdocked_pocket10/                      raw CrossDocked pockets + ligands
split_by_name.pt                           CrossDocked split
data_train.pt / data_test.pt               preprocessed tensors
pretrain/xray_crops_aligned_v5/{train,test}/…  + {train,test}_available.npy
voxbind/model_zoo/CDG_v2/                   frozen density encoder (+ cfg.yaml)
<sibling>/targetdiff/data/test_set/         full receptors (docking/pose eval)
```

### `02_preprocess_data.sh` — tensors + crops
With the prepared copy this is mostly a **verify** step. If `data_{train,test}.pt`
are missing it rebuilds them from CrossDocked (`preprocess_crossdocked.py`).
`MODE=crops` rebuilds the density corpus from raw 2Fo-Fc maps (heavy; only if
you are *not* using the prepared crops).

### `03_train.sh` — training (GPU)
`MODE` selects the piece:
- `fusion` *(default)* — density-conditioned model: base warm start + **frozen
  CDG_v2** encoder, token fusion. Needs a base checkpoint (see `base`) and the
  CDG_v2 encoder from step 1.
- `base` — vanilla VoxBind denoiser from scratch (the warm-start weights).
- `all` — `base` then `fusion`.
- `encoder` — *(advanced)* re-pretrain the CDG_v2 encoder via MAE; needs the
  PLINDER density corpus. Most users skip this — CDG_v2 ships in step 1.

Encoder geometry is read from `model_zoo/CDG_v2/cfg.yaml` automatically (model_zoo
entries are not interchangeable at fixed dims). Batch is **per-rank**: effective
batch = `BSZ × n_gpus`. Checkpoints land in `voxbind/exps/<EXP_NAME>/`.

### `04_sample.sh` — sampling (GPU)
Walk-jump samples ligands over the 100-pocket CrossDocked test split, sharded
across `GPUS`. Auto-detects density conditioning from the checkpoint and feeds
the x-ray crops. Density models sample only map-backed pockets (~79/100 for v5)
— a lower target count is expected. Output: `…/samples/<OUT>/target_XX/`.

### `05_evaluate.sh` — evaluation (CPU)
Three passes over a sampling dir: (1) chem/geometry, (2) Vina docking, (3)
PoseCheck/PoseBusters. Scores against the **full receptor** (`FULL_RECEPTOR_ROOT`),
not the crop. Refuses to run unless `voxdock` has vina 1.2.2. Writes per-target
`metrics.json` + a run-level `summary.json`.

---

## Environment variables (most useful)

**Global** (`_common.sh`): `REPO_ROOT`, `CODE_ROOT`, `DATA_ROOT`, `GPUS`,
`VOXBIND_ENV` / `VOXDOCK_ENV` / `MOLEVAL_ENV`, `DATA_RCLONE_REMOTE`,
`DATA_HTTP_URL`, `TARGETDIFF_ROOT`, `FULL_RECEPTOR_ROOT`, `CONDA_LAUNCHER`.

| Script | Vars |
|--------|------|
| `00` | `FORCE` (rebuild an env); positional targets: `voxbind voxdock moleval targetdiff` |
| `01` | `DATA_RCLONE_REMOTE` **or** `DATA_HTTP_URL` |
| `02` | `MODE=verify\|preprocess\|crops`, `FORCE`, `VERSION`, `WORKERS` |
| `03` | `MODE=fusion\|base\|all\|encoder`, `GPUS`, `BSZ`, `NUM_EPOCHS`, `SIGMA`, `EXP_NAME`, `ENCODER`, `WARM_START`, `WANDB` |
| `04` | `EXP` *(required)*, `OUT`, `SAMPLES`, `SPLIT`, `N_POCKETS`, `GPUS` |
| `05` | `SAMPLE_DIR` *(required)*, `DOCK=vina_dock\|vina_min\|vina_score\|none`, `POSE=posecheck\|posebusters\|all\|none`, `EXH`, `CPU`, `WORKERS`, `RECEPTOR_ROOT` |

---

## Notes & gotchas

- **Vina version is pinned to 1.2.2.** A different build shifts absolute
  affinities; `05` refuses to run otherwise. Rebuild with
  `FORCE=1 bash script/00_setup_env.sh voxdock`.
- **PoseCheck is pinned to 1.3.1.** Upstream silently redefined strain energy;
  1.3.1 is the definition all reported numbers use.
- **Full receptors** must be present for meaningful Vina scores and clash counts
  (a 10 Å crop can't clash with atoms it doesn't contain). They come from
  TargetDiff's `test_set` release (folded into the prepared copy).
- **TargetDiff is a sibling** of the VoxBind checkout — `notebook/webapp/data.py`
  resolves `VOXBIND.parent/targetdiff`. The Docker image places it correctly at
  `/workspace/targetdiff`.
- **Weights & Biases** logging is off by default (`WANDB=false`). Run
  `wandb login` and pass `WANDB=true` to enable it.
- These scripts wrap the same entrypoints as the internal
  `voxbind/scripts/70_`, `72_`, `73_`, `75_` launchers, but with portable paths
  and no box-specific assumptions.
