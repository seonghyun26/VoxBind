# interface_curriculum_100m_v2_e20

**100M ChannelViT CDG (coords+density+gradmag) MAE encoder — interface-masked + easy→hard curriculum.**
The campaign's strongest all-around encoder: **beats the champion on all 5 affinity axes** and is the
first CDG encoder to decisively beat ProFSA on the honest CASF-2016 clean-92 (0.706 vs 0.676, 5-seed).

## What it is
- Arch: ChannelViT `[7,4,2]`, dim 640 / depth 18 / heads 10, `patch_embed_mode=channel_group`, `n_in=13`.
- Inputs: atomblob (7 lig + 4 poc) + X-ray density + gradmag = 13 channels.
- Pretrain: PLINDER **v2** (112K), MAE, **interface masking** (pocket–ligand contact region) with a
  **stepwise curriculum mask ratio 0.60→0.70→0.80→0.90** (4 stages, `mask_curriculum=true`,
  `mask_curriculum_steps=4`), `channel_weighting=inv_freq`, EMA 0.999.
- Source run: `260823_cdg_100m_v2_interface_curriculum_0609`, **epoch 20** (= the 0.70 stage; this is the
  sweet spot — later high-mask stages 0.80/0.90 DEGRADE it, see caveat).

## Results (frozen mean-pool MLP probe, test ρ; FULL/CL123/lba = 5-seed panel, CASF = 5-seed casf_5seed.py)
| axis | this (curr e20) | champion (mask075) | 0.75-e40 | Δ vs champion |
|---|---|---|---|---|
| FULL (leaky, 1320) | 0.649 | 0.648 | 0.648 | +0.001 |
| CL123 (leak-proof, 733) | 0.611 | 0.603 | **0.626** | +0.008 |
| lba30 (novel ID<30%) | **0.598** | 0.571 | — | +0.027 |
| lba60 (novel ID<60%) | **0.630** | 0.612 | — | +0.018 |
| **CASF-clean (honest, 92)** | **0.706±0.020** | 0.662 | 0.667 | **+0.044** (> ProFSA 0.676) |

Wins 4/5 axes outright (FULL/lba30/lba60/CASF-clean); on CL123 it beats champion but trails the plain
interface variants (0.75-e40 0.626, 0.85-e30 0.635). Best pick when **novel-protein generalization**
(what VoxBind needs for novel pockets) matters more than the CL123 structural-cluster split.

## Load (VoxBind / probe)
`load_encoder` reads **`encoder_state_dict_ema`** from the checkpoint (NOT `state_dict_ema`).
```bash
python dataset/01c_pdbbind_probe.py probe \
  --conditions atomblob_density_gradmag --epoch 20 --voxel_version v5 --split lp_edrscc_v2 \
  --exp_dir model_zoo/interface_curriculum_100m_v2_e20 --allow_stale_features --seeds 5
```
Self-contained: `cfg.yaml` + `checkpoint_e0020.pth.tar` (+ `.hydra/`, `train_density.log`).

## Caveat (honest)
The gain is concentrated at **e20 (the 0.70 curriculum stage)**; pushing the ramp to 0.80 (e30) / 0.90
(e40+) monotonically hurts CASF-clean and CL123. So the finding is *"interface + a 0.60→0.70 mask
warmup is the sweet spot; high masking hurts,"* not *"a full easy→hard schedule wins."* CL123 is the one
axis where the plain interface_100m_v2_e40 (0.626) is stronger — keep that one if CL123 is the priority.
