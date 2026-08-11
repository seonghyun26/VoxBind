# VoxBind Model Zoo

Preserved "good" checkpoints for the density-pretrained affinity encoders, copied out of
`exps/` so they survive checkpoint cleanups. Open **`models.html`** for the summary table
(metrics, configs, and the train-on-test diagnostic).

Each subfolder holds the exact `cfg.yaml` (Hydra config the run trained under), `.hydra/`,
the training log, and `checkpoint_e0049.pth.tar` (probe reads `encoder_state_dict_ema`).

| folder | source exp | data | mask | test ρ / r / RMSE |
|---|---|---|---|---|
| `champion_100m_v2_mask075` | 260705_ar_cvit_100m_v2_mask075 | v2 (112K) | 0.75 | 0.644 / 0.660 / 1.349 |
| `pareto_100m_v3_mask095` | 260725_ar_cvit_100m_v3_m095 | v3 (71.7K) | 0.95 | 0.644 / 0.663 / 1.334 (Pareto-best) |
| `v3_100m_mask090` | 260725_ar_cvit_100m_v3_m090 | v3 (71.7K) | 0.90 | 0.645 / 0.660 / 1.383 |
| `efficient_60m_v3_mask085` | 260723_ar_cvit_60m_v3_mask085 | v3 (71.7K) | 0.85 | 0.641 / 0.661 / 1.372 (**65.6M — ties champion at ⅓ fewer params**) |
| `coords_100m_v2_mask075` | 260723_ar_cvit_100m_v2_mask075_coords | v2 (112K) | 0.75 | 0.596 / 0.632 / 1.381 (control) |

All ChannelViT `[7,4,2]`, 60-100M params, frozen mean-pool MLP probe on `lp_edrscc_v2` (Kd/Ki, 3850/817/1320, 3 seeds).

## Stable pointers (track the current best without renaming)

Version-agnostic symlinks point at whatever is currently best, so code/scripts can reference a
fixed path and a new champion is adopted by just **repointing the symlink** (relative → portable):

| pointer | → current target |
|---|---|
| `champion` | `champion_100m_v2_mask075` (C+D+G best encoder) |
| `coords` | `coords_100m_v2_mask075` (matched coords-only control) |

Use `--exp_dir model_zoo/champion` in the probe (each target dir is self-contained: `cfg.yaml` +
`checkpoint_e0049.pth.tar`). To promote a new champion: add its folder here, then
`ln -sfn <new_folder> champion`. The exps checkpoint that was cleaned up is restored via a relative
symlink back into this zoo (`exps/260705_ar_cvit_100m_v2_mask075/checkpoint_e0049.pth.tar`).
Best **probe-head recipe** on the champion encoder (head-only, 260806): MSE + Pearson-aux (λ=5) →
ρ 0.647 / r 0.665 / RMSE 1.325 (vs 0.644 / 0.660 / 1.349 mean-pool MSE).

## Two-tower encoders (for VoxBind pocket-adapter transfer)

Separate ~45M pocket / ligand ChannelViT-MAE encoders (260806, PLINDER v2 holo, protein_vdw mask +
mask-channel). Probed jointly via `test/twotower_probe.py` (frozen towers → cross-attention
interaction head, 3 seeds). The pocket encoder is the one that transfers to VoxBind.

| folder | input | test ρ / r / RMSE |
|---|---|---|
| `twotower_pocket_protein_vdw_mc` | pocket 4-atom + M_P⊙ρ + ‖∇ρ‖ + M_P (`[4,2,1]`) | joint two-tower: **0.587 / 0.599 / 1.474** |
| `twotower_ligdens_protein_vdw_mc` | ligand 7-atom + (1−M_P)⊙ρ + ‖∇ρ‖ + (1−M_P) (`[7,2,1]`) | (same probe) |

Below the single-encoder champion (0.644) — two-tower late-fusion handicap; see
`../../notebook/html/260806/voxbind_twotower_affinity_playbook.md` for the improvement ladder.

## Reprobe a kept checkpoint
```bash
python dataset/01c_pdbbind_probe.py probe \
  --conditions atomblob_density_gradmag --epoch 49 --voxel_version v5 --split lp_edrscc_v2 \
  --feature_tag 260725_ar_cvit_100m_v3_m095 --exp_dir model_zoo/pareto_100m_v3_mask095
```
Add `--leak_test append` for the train-on-test optimistic-ceiling diagnostic (writes a `_leaktest`-tagged CSV).
