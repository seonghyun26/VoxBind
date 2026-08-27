# CDG_v2

**The paper's headline "CDG v2" C+D+G encoder** — 100M ChannelViT (coords+density+gradmag) MAE,
atom-biased masking. Best-on-record affinity encoder in the campaign; the gold **CDG v2** row in
`results.html` Tables 1a/1b/1c.

Copied from `atombias_100m_v2_e25` (source run `260806_cdg_100m_v2_ep100`, **epoch 25**) and renamed to
the paper-facing name. Same weights, same `--epoch 25`.

## What it is
- Arch: ChannelViT `[7,4,2]`, dim 640 / depth 18 / heads 10, `patch_embed_mode=channel_group`, `n_in=13`.
- Inputs: atomblob (7 lig + 4 poc) + X-ray density + gradmag = 13 channels.
- Pretrain: PLINDER **v2** (112K), MAE, `mask_ratio=0.75`, **`mask_strategy=atom_biased`**
  (`mask_atom_tau=1.0`), `channel_weighting=inv_freq`, EMA 0.999.
- Source run: `260806_cdg_100m_v2_ep100`, **epoch 25** (= `checkpoint_e0025.pth.tar`).

## Results (frozen mean-pool MLP probe, `lp_edrscc_v2` Kd/Ki, 3850/817/1320)
| axis | ρ / r / RMSE |
|---|---|
| FULL (leaky, 1320) | **0.653 / 0.666 / 1.355** — best-on-record |
| CL3-leakproof | ρ **0.622** (> champion 0.60) |

Headline uses the **MSE-only** probe head (Pearson-aux dropped 260825 for a fair comparison vs the
mse-only baselines ProFSA / GeoSSL). This is the "ours" row across all cohorts in `results.html`.

## Caveat (honest)
**Early-peak**: this 100-epoch run peaks at **e25** then declines (atom_biased overtrains); a weight-soup
of later epochs does NOT beat e25. Always reprobe with `--epoch 25`.

## Load (VoxBind / probe)
`load_encoder` reads **`encoder_state_dict_ema`** from the checkpoint (NOT `state_dict_ema`).
```bash
python dataset/01c_pdbbind_probe.py probe \
  --conditions atomblob_density_gradmag --epoch 25 --voxel_version v5 --split lp_edrscc_v2 \
  --exp_dir model_zoo/CDG_v2 --allow_stale_features --seeds 3
```
Self-contained: `cfg.yaml` + `checkpoint_e0025.pth.tar` (+ `.hydra/`, `train_density.log`).
