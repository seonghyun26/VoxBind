# interface_100m_v2_e40

**100M ChannelViT CDG (coords+density+gradmag) MAE encoder — interface-masked.**
The campaign's first pretext-alignment win: beats the champion on every leak-proof /
protein-novelty axis while tying on the leaky FULL benchmark and matching ProFSA on
honest CASF-2016 clean-92.

## What it is
- Arch: ChannelViT `[7,4,2]`, dim 640 / depth 18 / heads 10, `patch_embed_mode=channel_group`, `n_in=13`.
- Inputs: atomblob (7 lig + 4 poc) + X-ray density + gradmag = 13 channels.
- Pretrain: PLINDER **v2** (112K positions, `data_train_plinder_v2_perelem.pt`), MAE, **interface masking @ 0.75**
  (masks the pocket–ligand CONTACT region), `channel_weighting=inv_freq`, EMA 0.999.
- Source run: `260821_cdg_100m_v2_interface_mask075`, **epoch 40** (early-peak; declines by e49).

## Results (frozen mean-pool MLP probe, lp_edrscc_v2 Kd/Ki, test ρ, 5-seed, MSE+corr recipe)
| axis | this (interface e40) | champion (mask075) | Δ |
|---|---|---|---|
| FULL (leaky, 1320) | 0.648 | 0.648 | tie |
| **CL123 (leak-proof, 733)** | **0.626** | 0.603 | **+0.023** |
| **CASF-clean (honest, 92)** | **0.675** (= ProFSA 0.676) | 0.662 | **+0.013** |

## Load (VoxBind / probe)
`load_encoder` reads **`encoder_state_dict_ema`** from the checkpoint (NOT `state_dict_ema`).
```bash
python dataset/01c_pdbbind_probe.py probe \
  --conditions atomblob_density_gradmag --epoch 40 --voxel_version v5 --split lp_edrscc_v2 \
  --exp_dir model_zoo/interface_100m_v2_e40 --allow_stale_features --seeds 5
```
Self-contained: `cfg.yaml` + `checkpoint_e0040.pth.tar` (+ `.hydra/`, `train_density.log`).

## Notes
- For a stronger *leak-proof* variant, interface @ **0.85** (CL123 0.635) was training as of 260823 —
  swap in later once complete if leak-proof generalization is the priority.
- `v2.4` (CASF-homolog-decontaminated pretrain) confirmed this CASF-clean win is honest (0.675→0.680 held).
