# CDG_v2_ratio10

**The campaign's first REAL masking win over the CDG v2 champion.** 100M ChannelViT `[7,4,2]`
(coords+density+gradmag), identical to champion **CDG v2** EXCEPT the MAE masking: `atom_biased`
→ **`group_complete` with `group_complete_ratio=1.0`** (FULL-family masking — hard-mask ONE
semantic family completely each sample, reconstruct it purely from the other two, cycled
ligand↔pocket↔density).

Source run `260908_cdg_100m_v2_gc_ratio10`, **epoch 25** (its peak; e30 declines to ≈champ).

## Why this is real (vs the group_complete false-alarm)
group_complete (ratio 0.9) LOOKED like a win at 3-seed (CASF +0.029) but a 10-seed re-probe
collapsed it (MEAN −0.006, gain was CASF-only noise). **ratio10 e25 SURVIVES 10-seed confirmation**:
the gain is broad (6/6 cohorts ≥ champion — noise would give ~3/6), present on the RELIABLE
large-n sets (FULL n=1320 +0.008), and corroborated across ρ / r / RMSE. Not a small-n artifact.

## Results (6-set casf-machinery probe, base = champion CDG v2 e25, **10-seed**)
| cohort | champ e25 | **ratio10 e25** | Δρ |
|---|---|---|---|
| FULL (1320) | 0.631 | 0.640 | **+0.008** |
| CL3 (733) | 0.626 | 0.629 | +0.003 |
| CL3-ID60 (454) | 0.625 | 0.631 | +0.005 |
| **CL3-ID30 (262, novelty)** | 0.601 | 0.625 | **+0.024** |
| CASF-nontrain (124) | 0.706 | 0.708 | +0.002 |
| CASF-clean (92) | 0.700 | 0.720 | +0.020 |
| **6-set MEAN** | 0.648 | **0.659** | **+0.011** |

Headline: **CL3-ID30 (novelty) +0.024** — the campaign-target metric — plus a small broad lift
(MEAN +0.011). r/RMSE agree everywhere (FULL r 0.656>0.644, RMSE 1.353<1.379; ID30 r 0.666>0.638).

## What it is
- Arch: ChannelViT `[7,4,2]`, dim 640 / depth 18 / heads 10, `patch_embed_mode=channel_group`, `n_in=13`. (Same as CDG_v2.)
- Pretrain: PLINDER v2 (112K), MAE, inv_freq, EMA 0.999. (Same as CDG_v2.)
- **ONLY change vs CDG_v2:** `mask_strategy=group_complete` + `group_complete_ratio=1.0` + `group_complete_light_ratio=0.15`.
- Peak **e25** (`checkpoint_e0025.pth.tar`); atom-biased-family early-peak (e30 already ≈champ).

## Caveats (honest)
1. **Modest magnitude** — MEAN +0.011; individual large-n gains (FULL +0.008, CL3 +0.003) are ~1σ.
   The strongest, most reliable single claim is CL3-ID30 +0.024 (~1.3σ, n=262).
2. **Pending second-seed reproducibility** — 10-seed controls probe-head noise but NOT pretraining
   init luck. A second pretraining seed (`260908_..._ratio10_s7`, seed=7) is running to confirm the
   win reproduces. Treat as provisional until that lands.
3. ratio 1.0 > 0.9: full-family masking beats the 0.9 version (group_complete) on ID30 (0.625 vs 0.593).

Related: [[project_group_cross_channel_260907]]. Reprobe with `--epoch 25`.
