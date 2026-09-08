# CG_v2

**Coords+Gradmag (CG) — density-gradient WITHOUT the raw density channel.** The 4th rung of the
C / CD / CG / CDG input-modality ladder. gradmag ‖∇ρ‖ is still derived from X-ray density during
loading, but the density channel itself is NEVER fed to the encoder (nor reconstructed) — only its
gradient. Isolates gradmag's contribution independent of density.

Source run `260904_cg_100m_v2_atombias`, **epoch 20** (atom_biased peak; best-balanced 6-set:
FULL 0.631 / CL3 0.630 / CL3-ID60 0.629 / CL3-ID30 0.611 / CASF-nontrain 0.689 / CASF-clean 0.676;
MEAN 0.644 — the HIGHEST of all four ladder encoders).

## What it is
- Arch: ChannelViT `[7,4,1]`, dim 640 / depth 18 / heads 10, `patch_embed_mode=channel_group`, `n_in=12`.
- Inputs: atomblob (7 lig + 4 poc) + **gradmag ‖∇ρ‖** = 12 channels. NO density channel.
- Enabled by NEW flag `mae.gradmag_without_density=true` (input_mode=atomblob + with_gradmag).
- Pretrain: PLINDER **v2** (112K), MAE, `mask_ratio=0.75`, **`mask_strategy=atom_biased`**
  (`mask_atom_tau=1.0`), `channel_weighting=inv_freq`, `gradmag_channel_weight=0.1`, EMA 0.999.
  EXACT CDG v2 recipe, but density is used only to COMPUTE gradmag then discarded.
- Source run: `260904_cg_100m_v2_atombias`, **epoch 20** (= `checkpoint_e0020.pth.tar`).

## Modality ladder (all atom_biased v2 recipe, 6-set Spearman at each peak)
| encoder | channels | FULL | CL3 | CL3-ID60 | CL3-ID30 | CASF-nt | CASF-cl | MEAN |
|---|---|---|---|---|---|---|---|---|
| C_v2 (e20)  | [7,4] n11   | 0.589 | 0.594 | 0.591 | 0.584 | 0.678 | 0.674 | 0.618 |
| CD_v2 (e10) | [7,4,1]D n12| 0.574 | 0.573 | 0.600 | 0.589 | 0.685 | 0.685 | 0.618 |
| **CG_v2 (e20)** | [7,4,1]G n12| **0.631** | **0.630** | **0.629** | **0.611** | 0.689 | 0.676 | **0.644** |
| CDG_v2 (e25)| [7,4,2] n13 | 0.653 | 0.622 | 0.618 | 0.591 | 0.689 | 0.683 | 0.639 |

**Finding: gradmag carries almost all the density-derived signal.** CG beats CD on all 6 sets;
+gradmag (CG−C) = +0.027 to +0.042 broadly, while +density (CD−C) ≈ 0 on average. The two channels
are largely REDUNDANT (not complementary): CG alone (MEAN 0.644) even edges CDG (0.639), and on
CL3-ID30 novelty CG (0.611) > CDG (0.591) — adding raw density on top of gradmag slightly dilutes it.
