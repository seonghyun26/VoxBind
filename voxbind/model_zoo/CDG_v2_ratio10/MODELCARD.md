# CDG_v2_ratio10

**Full-family masking variant (group_complete_ratio=1.0), seed 42.** 100M ChannelViT `[7,4,2]`,
identical to champion **CDG v2** EXCEPT MAE masking: `atom_biased` → **`group_complete` +
`group_complete_ratio=1.0`** (hard-mask ONE semantic family completely per sample, reconstruct
it from the other two, cycled ligand↔pocket↔density). Source `260908_cdg_100m_v2_gc_ratio10`,
epoch 25 (its peak).

> ⚠️ **HONEST STATUS (corrected 260908): NOT a robust win. The apparent win did NOT reproduce
> across pretraining seed.** At its OWN seed (42), ratio10 e25 cleanly beats the champion
> (same-seed comparison: MEAN +0.011 / CL3-ID30 +0.024 @10-seed, 6/6 cohorts positive, survives
> 10-seed probe-noise control). BUT a second pretraining seed (`..._ratio10_s7`, seed 7) lands
> **−0.022 MEAN vs champion, below on all 6 sets.** The two seeds STRADDLE the champion
> (seed42 MEAN 0.663, seed7 0.624, champion ≈0.647 @5-seed) — a 0.039 seed-to-seed swing, LARGER
> than any masking-recipe effect (~0.01-0.02). **Conclusion: masking-recipe differences are below
> the pretraining-init noise floor (±0.02-0.03); ratio10's win was seed-42 luck.** Kept as an
> ablation artifact, NOT a champion replacement.

## The key lesson
The 10-seed probe confirmed the win wasn't PROBE-head noise — but only the second PRETRAINING seed
(s7) revealed it was INIT noise. Masking variants (atom_biased / per_group / group_complete /
ratio10) all sit within ±0.02 of each other AND within the ~±0.02-0.03 pretraining-seed variance,
so NONE can be declared a robust winner from single-seed runs. Real gains must exceed the seed
floor → the bigger levers (ESM2 cross-modal, ProFSA-align, more data, architecture), not masking.

## Results (6-set casf-machinery probe, base = champion CDG v2 e25)
| cohort | ratio10 e25 (seed42) Δρ@10s | s7 e25 (seed7) Δρ@5s |
|---|---|---|
| FULL (1320) | +0.008 | −0.017 |
| CL3 (733) | +0.003 | −0.024 |
| CL3-ID60 (454) | +0.005 | −0.034 |
| CL3-ID30 (262, novelty) | +0.024 | −0.015 |
| CASF-nontrain (124) | +0.002 | −0.029 |
| CASF-clean (92) | +0.020 | −0.012 |
| **6-set MEAN** | **+0.011** | **−0.022** |

## What it is
- Arch: ChannelViT `[7,4,2]`, dim 640 / depth 18 / heads 10, `channel_group`, `n_in=13` (= CDG_v2).
- Pretrain: PLINDER v2 (112K), MAE, inv_freq, EMA 0.999, **seed 42**. ONLY change vs CDG_v2 = `mask_strategy=group_complete` + `group_complete_ratio=1.0`.
- Peak e25 (`checkpoint_e0025.pth.tar`). Reprobe with `--epoch 25`.

Related: [[project_group_cross_channel_260907]].
