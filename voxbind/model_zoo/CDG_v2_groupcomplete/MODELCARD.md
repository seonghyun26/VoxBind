# CDG_v2_groupcomplete

**Cross-family masking variant of the CDG v2 encoder** — 100M ChannelViT `[7,4,2]`
(coords+density+gradmag), identical to the champion **CDG v2** in every way EXCEPT the MAE
masking strategy: `atom_biased` → **`group_complete`**. Kept as the group_complete data point
in the masking-strategy ablation.

> ⚠️ **HONEST STATUS (corrected 260908): this is NOT a win over the champion.** A 3-seed probe
> initially showed group_complete e20 beating the champion on the CASF holdout sets (+0.027–0.029),
> but a **10-seed re-probe REFUTED it** — the gap collapsed to CASF-nt **−0.003** / CASF-cl +0.011
> (within seed noise) and the 6-set MEAN is **−0.006 behind** the champion. The apparent win was
> small-n (CASF 124/92, per-seed std ~0.05) + few-seed noise. group_complete ≈ champion, very
> slightly worse. Copied from source run `260907_cdg_100m_v2_groupcomplete`, **epoch 20** (its peak).

## What it is
- Arch: ChannelViT `[7,4,2]`, dim 640 / depth 18 / heads 10, `patch_embed_mode=channel_group`, `n_in=13`. (Same as CDG_v2.)
- Inputs: atomblob (7 lig + 4 poc) + X-ray density + gradmag = 13 channels.
- Pretrain: PLINDER **v2** (112K), MAE, `channel_weighting=inv_freq`, EMA 0.999. (Same as CDG_v2.)
- **The ONLY change vs CDG_v2:** `mask_strategy=group_complete` (`group_complete_ratio=0.9`,
  `group_complete_light_ratio=0.15`). Each sample HARD-masks ONE semantic family (~0.9:
  ligand-atoms OR pocket-atoms OR density+gradmag) and lightly masks the other two (~0.15) →
  reconstruct the fully-hidden family from the two visible ones, cycled ligand↔pocket↔density.
- Source run: `260907_cdg_100m_v2_groupcomplete`, **epoch 20** (= `checkpoint_e0020.pth.tar`).

## Results (6-set casf-machinery probe, tta_6set base = champion CDG v2 e25)
| cohort | champ e25 (10-seed ρ) | **gc e20 (10-seed ρ)** | Δρ | ~~3-seed Δρ (noisy)~~ |
|---|---|---|---|---|
| FULL (1320) | 0.631 | 0.625 | −0.006 | −0.006 |
| CL3 (733) | 0.626 | 0.614 | −0.012 | −0.009 |
| CL3-ID60 (454) | 0.625 | 0.610 | −0.016 | −0.012 |
| CL3-ID30 (262, novelty) | 0.601 | 0.588 | −0.013 | −0.002 |
| CASF-nontrain (124) | 0.706 | 0.703 | **−0.003** | ~~+0.029~~ |
| CASF-clean (92) | 0.700 | 0.711 | +0.011 | ~~+0.027~~ |
| **6-set MEAN** | **0.648** | 0.642 | **−0.006** | ~~+0.004~~ |

**Lesson:** the CASF-nt "win" (+0.029 at 3 seeds → −0.003 at 10 seeds) was entirely inside the
probe-head seed variance on the small CASF sets. ALWAYS use ≥10 seeds when the holdout sets are
n<150. group_complete is a clean-masking variant that ties (slightly loses) the champion — the
symmetric whole-family masking neither helps nor clearly hurts on this corpus.

## Caveat (peak epoch)
Early-peak at **e20** then declines (e25 CASF-nt drops with ballooning seed variance = overtraining).
Reprobe with `--epoch 20`. Related: [[project_group_cross_channel_260907]].

## Load (VoxBind / probe)
Same as CDG_v2 (`patch_embed_mode=channel_group`, `channel_groups=[7,4,2]`), but `--epoch 20`.
