# SB handoff: density-conditioned MCP fine-tuning

The MCP workflow is in the `FuncBind` submodule, not the numbered VoxBind
small-molecule scripts in this directory. Use the committed submodule revision.

## Code and environment

- [SB shared repository](https://github.com/sbintuitions/kaist-sbint-ai4s)
- [FuncBind repository](https://github.com/seonghyun26/funcbind)
- [MCP instructions and Docker mounts](../FuncBind/scripts/README_mcp_density.md)
- [FuncBind Dockerfile](dockerfile-funcbind.sbint), extending the existing
  [VoxBind SB Dockerfile](dockerfile.sbint) image (`voxbind:allinone` by default).

```bash
# In your existing SB repository checkout:
git pull --ff-only
git submodule update --init --recursive FuncBind
bash FuncBind/scripts/0_env_setup.sh
```

The FuncBind image was rebuilt and tested locally as `voxbind-funcbind:sb`.
This is a local tag, NOT a published registry image: SB should build it using
the Dockerfile. No host `.repro-env` or source-code mount is needed. Checkpoints
and datasets are not part of Git or the image; use persistent volumes.

## Files to provide

| Asset | Download/source | Purpose |
|---|---|---|
| FuncBind `fb_unified/checkpoint.pth.tar` (61.68 GB) | [Pinned official download](https://huggingface.co/mkirchmeyer/funcbind/resolve/f42d3daeb6e7c1fa2b20096f9a147aa3b1f8814f/fb_unified/checkpoint.pth.tar) | Density-free pretrained starting model; `FB_MODEL_URL` |
| NF `nf_unified/model.pt` (1.45 GB) | [Pinned official download](https://huggingface.co/mkirchmeyer/funcbind/resolve/f42d3daeb6e7c1fa2b20096f9a147aa3b1f8814f/nf_unified/model.pt) | Neural-field encoder/decoder; `NF_MODEL_URL` |
| CDG v2 `checkpoint_e0025.pth.tar` (1.19 GB) | Dropbox: `/박성현/VoxBind/results/task1-affinity/CDG-v2/checkpoint_e0025.pth.tar`; file link delivered separately | Frozen density encoder; `CDG_MODEL_URL` |
| MCP splits + original structures | [Public dataset](https://huggingface.co/datasets/Willete3/mcpp-dataset/tree/main) | Automatically downloaded by step 1; original archive alone is ~32.7 GB compressed |
| X-ray coordinates + 2Fo-Fc maps | RCSB + PDBe, per target | Automatically downloaded and processed by step 1 |

The CDG upload was confirmed on 2026-09-15 (server modified 2026-09-14).
It is the actual checkpoint, not the task1-affinity representation files.
The shared file link is intentionally kept out of public Git repositories.

Copy [the URL/checksum template](../FuncBind/scripts/mcp_assets.env.example)
to `FuncBind/scripts/mcp_assets.env`, fill the CDG file URL supplied with this
handoff, and source it on the host. Use the Docker invocation in the MCP
instructions above so URL **and checksum** variables reach the container.
The three weights total ~64.3 GB; this excludes datasets, caches, and training
outputs. No Dropbox configuration is needed for publicly downloadable links.

Do **not** use an old density-conditioned MCP checkpoint as `FB_MODEL_URL`:
this experiment starts a fresh fine-tune from the density-free FuncBind model.
Leave `MCP_MODEL_URL` unset; it is only an optional generation-checkpoint
download, not a training-resume setting. Sampling later uses the checkpoint
produced by the full training run; the smoke run does not save that checkpoint.

## Run on SB

Target: one node, **8 x H100 80GB**, 32 CPU cores, preferably 256 GiB host RAM.
Keep at least 200 GiB free on the checkpoint volume **after** staging inputs;
retained old checkpoints/shards need more. Use writable persistent volumes.

Inside the container, in `/workspace/FuncBind`:

```bash
bash scripts/1_data_process.sh
SMOKE=1 N_SAMPLES=48 EFFECTIVE_BATCH=16 bash scripts/2_train.sh
# Inspect the smoke log and resolve any preflight/runtime errors first.
SMOKE=1 bash scripts/2_train.sh
bash scripts/2_train.sh
# After training has produced a checkpoint:
SMOKE=1 bash scripts/3_generate.sh
bash scripts/3_generate.sh
```

The regular training recipe uses default density fusion, frozen CDG v2 epoch 25,
`bf16-mixed`, batch 1/GPU and accumulation 95 (effective batch 760). It includes
ZeRO-1 Adam-state sharding, CPU EMA, activation checkpointing, retained DDP
gradient buffers and conservative loader/validation settings. Parameters and
optimizer moments remain FP32; EMA is preserved, not disabled.

Verified locally: 16 tests passed (4 optional-artifact tests skipped), two-process
Adam/reference/save-resume parity, and a two-RTX-3090 reduced-model MCP smoke
covering accumulation, real CDG v2 conditioning, checkpoint reload, and latent
sampling/NF decoding. **Full 5.14B model capacity on H100 and generation quality
are not yet validated.** Ask SB to report training/validation completion and
GPU peak memory before launching the long experiment; do not bypass preflight
just to force a launch.
