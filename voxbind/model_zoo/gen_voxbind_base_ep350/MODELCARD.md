# gen_voxbind_base_ep350

Vanilla VoxBind, the ICML'24 recipe retrained here. Generator checkpoint for the **VoxBind base (ours, ep350)** row of
`results/task2-drugdesign/VoxBind-base-ep350` (samples + per-molecule metrics live there).

| | |
|---|---|
| source run | `voxbind/exps/260827_voxbind_base_8gpu` (svr12) |
| weights | `checkpoint.pth.tar`, 1.2 GiB, 2026-08-30 |
| sigma | 0.9 |
| samples | `results/task2-drugdesign/VoxBind-base-ep350/samples/` -- 100 pockets x 10 molecules |


## Recipe

Stock `configs/config_train.yaml`: density-free `model/voxbind` (111.6M), crossdocked, 64^3 @ 0.25 A, sigma=0.9, lr 1e-5, wd 1e-2, aug, 350 ep, 8-GPU DDP (bsz 8/rank = 64 effective). Trained by `scripts/70_train_voxbind_base_8gpu.sh`.

## Why it exists

The density-free control every fusion arm is measured against -- and the only arm that covers all 100 test pockets.

## Generation numbers

Vina in kcal/mol, mean over molecules; PB = PoseBusters dock-mode validity;
clash/strain are PoseCheck medians. `all` = every pocket this arm sampled,
`density79` = the 79 pockets with usable deposited density (the apples-to-apples
scope across arms). Source: `results/task2-drugdesign/VoxBind-base-ep350/metrics.json`.

| scope | pockets | mols | vina score | vina dock | PB valid | clash med | strain med |
|---|---|---|---|---|---|---|---|
| all | 100 | 1000 | -6.26 | -7.65 | 60.3 % | 6.0 | 261.6 |
| density79 | 79 | 790 | -6.14 | -8.08 | 61.9 % | 6.0 | 258.2 |

QED 0.518 - SA 0.595 - diversity 0.726 -
high-affinity 60.6 % - validity 1.000 -
docked 1000/1000 (0 failed).

Our arms draw 10 molecules per pocket against the published baselines' ~100, so
compare per-molecule means, not totals.

## Re-sample from it

```bash
cd voxbind
python sample.py pretrained_path=model_zoo/gen_voxbind_base_ep350 wjs.split=test \
    wjs.n_samples_per_pocket=10 wjs.n_targets=79
```

`cfg.yaml` + `.hydra/` are this run's training config; `train_ddp.log` is its log.
Weights and configs are git-ignored here and travel via `dropbox_push.sh`.
