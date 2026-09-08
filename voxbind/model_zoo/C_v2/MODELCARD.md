# C_v2

**Coords-only (C) twin of the CDG v2 champion** — 100M ChannelViT MAE on atom channels ONLY
(no density, no gradmag). The bottom rung of the C / CD / CDG input-modality ladder, all on the
identical atom-biased v2 recipe, so density's marginal contribution is isolated.

Copied from source run `260822_coords_100m_v2_atombias`, **epoch 20** (atom_biased early-peaks;
6-set probe found e20 the peak: FULL 0.589 / CL3 0.594 / CL3-ID30 0.584 / CASF-nontrain 0.678).

## What it is
- Arch: ChannelViT `[7,4]`, dim 640 / depth 18 / heads 10, `patch_embed_mode=channel_group`, `n_in=11`.
- Inputs: atomblob only (7 lig + 4 poc element channels) — NO density, NO gradmag.
- Pretrain: PLINDER **v2** (112K), MAE, `mask_ratio=0.75`, **`mask_strategy=atom_biased`**
  (`mask_atom_tau=1.0`), `channel_weighting=inv_freq`, EMA 0.999. EXACT CDG v2 recipe minus density.
- Source run: `260822_coords_100m_v2_atombias`, **epoch 20** (= `checkpoint_e0020.pth.tar`).

## Modality ladder (all atom_biased v2 recipe, 6-set Spearman)
- **C_v2**  = `[7,4]`  n_in=11 (this)
- **CD_v2** = `[7,4,1]` n_in=12 (coords+density, `260903_cd_100m_v2_atombias`)
- **CDG_v2** = `[7,4,2]` n_in=13 (coords+density+gradmag, champion `260806_cdg_100m_v2_ep100` e25)
