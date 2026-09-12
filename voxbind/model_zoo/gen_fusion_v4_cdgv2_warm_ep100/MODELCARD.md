# gen_fusion_v4_cdgv2_warm_ep100

Token fusion v4 + frozen CDG_v2 encoder, warm-started. Generator checkpoint for the **Fusion v4 cdgv2 warm ep100** row of
`results/task2-drugdesign/Fusion-v4-cdgv2-warm-ep100` (samples + per-molecule metrics live there).

| | |
|---|---|
| source run | `voxbind/exps/260831_fusion_v4_cdgv2_8gpu` (svr12) |
| weights | `checkpoint.pth.tar`, 1.6 GiB, 2026-08-31 |
| sigma | 0.9 |
| samples | `results/task2-drugdesign/Fusion-v4-cdgv2-warm-ep100/samples/` -- 79 pockets x 10 molecules |
| frozen encoder | `model_zoo/CDG_v2/checkpoint_e0025.pth.tar` -- **required to load this checkpoint**; already in the zoo and on Dropbox |


## Recipe

`scripts/75_train_fusion_v4_cdgv2_8gpu.sh`, warm-started from `gen_voxbind_base_ep350`, 100 ep on the 78.5k x-ray subset, bsz 16/rank = 128. The frozen encoder's PATCH TOKENS feed the denoiser directly (`models/voxbind.py::_density_token_grid`) -- the representation the affinity probe was scored on.

## Why it exists

First density-conditioned arm. Came out ~0.5 kcal/mol WORSE on Vina dock than the vanilla baseline over the same 79 pockets while val miou rose, which is what motivated the two scratch arms: warm-starting confounds density conditioning, the subset restriction and 100 epochs of drift.

## Generation numbers

Vina in kcal/mol, mean over molecules; PB = PoseBusters dock-mode validity;
clash/strain are PoseCheck medians. `all` = every pocket this arm sampled,
`density79` = the 79 pockets with usable deposited density (the apples-to-apples
scope across arms). Source: `results/task2-drugdesign/Fusion-v4-cdgv2-warm-ep100/metrics.json`.

| scope | pockets | mols | vina score | vina dock | PB valid | clash med | strain med |
|---|---|---|---|---|---|---|---|
| all | 79 | 790 | -6.01 | -7.48 | 69.2 % | 7.0 | 204.7 |
| density79 | 79 | 790 | -6.01 | -7.48 | 69.2 % | 7.0 | 204.7 |

QED 0.448 - SA 0.603 - diversity 0.674 -
high-affinity 57.5 % - validity 1.000 -
docked 790/790 (0 failed).

Our arms draw 10 molecules per pocket against the published baselines' ~100, so
compare per-molecule means, not totals.

## Re-sample from it

```bash
cd voxbind
python sample.py pretrained_path=model_zoo/gen_fusion_v4_cdgv2_warm_ep100 wjs.split=test \
    wjs.n_samples_per_pocket=10 wjs.n_targets=79
```

`cfg.yaml` + `.hydra/` are this run's training config; `train_ddp.log` is its log.
Weights and configs are git-ignored here and travel via `dropbox_push.sh`.
