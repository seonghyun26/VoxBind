# CD_v2

**Coords+Density (CD) middle rung** of the C / CD / CDG input-modality ladder — 100M ChannelViT
MAE on atom channels **+ X-ray density** but NO density-gradient (gradmag). Isolates density's
marginal value on the atom_biased v2 recipe (C→CD = +density; CD→CDG = +gradmag).

Source run `260903_cd_100m_v2_atombias`, **epoch 10** (atom_biased peaks EARLY; 6-set probe found
e10 the best-balanced: FULL 0.574 / CL3 0.573 / CL3-ID60 0.600 / CL3-ID30 0.589 / CASF-nontrain
0.685 / CASF-clean 0.685 — MEAN 0.618; e15/e20 raise FULL but drop CASF+novelty).

## What it is
- Arch: ChannelViT `[7,4,1]`, dim 640 / depth 18 / heads 10, `patch_embed_mode=channel_group`, `n_in=12`.
- Inputs: atomblob (7 lig + 4 poc) + X-ray density = 12 channels. NO gradmag.
- Pretrain: PLINDER **v2** (112K), MAE, `mask_ratio=0.75`, **`mask_strategy=atom_biased`**
  (`mask_atom_tau=1.0`), `channel_weighting=inv_freq`, `density_channel_weight=0.1`,
  `gradmag_reconstruct=false`, EMA 0.999. EXACT CDG v2 recipe minus gradmag.
- Source run: `260903_cd_100m_v2_atombias`, **epoch 10** (= `checkpoint_e0010.pth.tar`).

## Modality ladder (all atom_biased v2 recipe, 6-set Spearman at each peak)
| encoder | channels | FULL | CL3 | CL3-ID60 | CL3-ID30 | CASF-nt | CASF-cl |
|---|---|---|---|---|---|---|---|
| C_v2 (e20)  | [7,4] n11  | 0.589 | 0.594 | 0.591 | 0.584 | 0.678 | 0.674 |
| **CD_v2 (e10)** | [7,4,1] n12 | 0.574 | 0.573 | 0.600 | 0.589 | 0.685 | 0.685 |
| CDG_v2 (e25) | [7,4,2] n13 | 0.653 | 0.622 | 0.618 | 0.591 | 0.689 | 0.683 |

+D (density = CD−C): helps CASF (+0.007/+0.011) & CL3-ID60/30 (+0.009/+0.005), neutral/negative on FULL/CL3.
+G (gradmag = CDG−CD): DRIVES FULL/CL3 (+0.079/+0.049), small on CASF/novelty. The two density channels
cover complementary axes — density → holdout/novelty, gradmag → in-distribution.
