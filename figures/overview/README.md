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
- `single_cdg_pretraining_scheme.{svg,pdf,png}` — the single-encoder
  pre-training scheme in which ligand coordinates, pocket coordinates, density,
  and density gradient form one aligned 13-channel CDG tensor and are jointly
  reconstructed by a shared encoder.
- `single_cdg_downstream_tasks.{svg,pdf,png}` — two frozen-encoder downstream
  paths: binding-affinity probing with MSE plus Pearson-correlation loss, and
  high-level feature fusion with the VoxBind generation pipeline.

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

**Single-CDG pre-training.** Ligand coordinates, pocket coordinates,
experimental electron density, and density gradient are aligned in one
pocket-centered frame and concatenated into a 13-channel CDG tensor. A shared
masked-autoencoding encoder reconstructs all three modalities in the same frame
using a masked-only reconstruction objective.

**Frozen-encoder downstream tasks.** For binding-affinity probing, the frozen
CDG encoder is followed by global mean pooling and a trainable MLP probe. The
probe combines mean-squared error with a correlation loss defined as one minus
the Pearson correlation. For VoxBind conditioning, the original noisy-ligand
and pocket inputs form one frozen feature stream, while a frozen CDG encoder
receives zeroed ligand channels together with pocket coordinates and
experimental ED. A high-level feature-fusion interface is shown under
evaluation before conditioned denoising and walk-jump sampling.

## Code grounding

- Two-tower input construction and masking: `voxbind/train_density.py`
- Pocket/ligand checkpoint configurations: `voxbind/model_zoo/twotower_*/cfg.yaml`
- Frozen-token cross-attention probe: `voxbind/test/twotower_probe.py` and
  `voxbind/models/twotower_head.py`
- Single-CDG affinity probe and MSE/correlation objective:
  `voxbind/dataset/01c_pdbbind_probe.py`
- PocketAdapter and VoxBind fusion: `voxbind/models/voxbind.py`
- Checkpoint loading and freeze policy: `voxbind/models/__init__.py`
- Adapter training configuration: `voxbind/configs/config_train_voxbind_adapter_pocket.yaml`

The figures intentionally show the current `fusion: adapter` route. Older
`default` density-branch and `v3` fusion alternatives are not included.
