# VoxBind overview figures

These figures describe the current two-tower pre-training and pocket-adapter
integration path. The SVG files are the editable masters; PDF is intended for
paper inclusion and PNG for quick review.

## Outputs

- `pretraining_overview.{svg,pdf,png}` — protein-vdW density partitioning,
  independent pocket/ligand ChannelViT-MAE runs, the frozen-token affinity
  check, and the pocket-only transfer boundary.
- `voxbind_adapter_overview.{svg,pdf,png}` — frozen VoxBind coordinate path,
  frozen pretrained pocket tower, trainable zero-initialized spatial adapter,
  denoising objective, and walk-jump sampling.
- `single_cdg_encoder_overview.{svg,pdf,png}` — the earlier single-encoder
  pipeline in which ligand coordinates, pocket coordinates, density, and
  density gradient form one aligned 13-channel CDG tensor and are learned by a
  shared encoder.

Regenerate all formats with:

```bash
MPLCONFIGDIR=/tmp/voxbind-paper-mpl python figures/overview/make_paper_overviews.py
```

## Suggested captions

**Pre-training.** Aligned protein-ligand complexes and experimental electron
density are partitioned with a protein van-der-Waals occupancy mask. Separate
pocket and ligand ChannelViT encoders are pretrained by masked voxel
reconstruction. Their frozen spatial tokens are evaluated with a bidirectional
cross-attention affinity head; only the pre-fusion pocket encoder is transferred
to VoxBind.

**VoxBind integration.** The pretrained pocket ChannelViT and the original
VoxBind model are frozen. A trainable spatial adapter converts the pocket
encoder's group-pooled patch tokens into a full-resolution residual that is
added to the original ligand-plus-pocket coordinate representation. The adapter
output is zero-initialized, so the integrated model starts exactly at the frozen
VoxBind baseline.

**Single CDG encoder.** Ligand coordinates, pocket coordinates, experimental
electron density, and density gradient are aligned in one pocket-centered frame
and concatenated into a 13-channel CDG tensor. One masked-autoencoding encoder
learns their joint spatial representation. For VoxBind conditioning, the
original noisy-ligand and pocket inputs form the baseline feature stream, while
the frozen CDG encoder receives zeroed ligand channels together with pocket
coordinates and experimental ED. Their high-level feature fusion is shown as a
module under evaluation rather than committing to one detailed implementation.

## Code grounding

- Two-tower input construction and masking: `voxbind/train_density.py`
- Pocket/ligand checkpoint configurations: `voxbind/model_zoo/twotower_*/cfg.yaml`
- Frozen-token cross-attention probe: `voxbind/test/twotower_probe.py` and
  `voxbind/models/twotower_head.py`
- PocketAdapter and VoxBind fusion: `voxbind/models/voxbind.py`
- Checkpoint loading and freeze policy: `voxbind/models/__init__.py`
- Adapter training configuration: `voxbind/configs/config_train_voxbind_adapter_pocket.yaml`

The figures intentionally show the current `fusion: adapter` route. Older
`default` density-branch and `v3` fusion alternatives are not included.
