# gen_fusion_v4_cdgv2_scratch_ep350

Same fusion, trained from scratch. Generator checkpoint for the **Fusion v4 cdgv2 scratch ep350** row of
`results/task2-drugdesign/Fusion-v4-cdgv2-scratch-ep350` (samples + per-molecule metrics live there).

| | |
|---|---|
| source run | `voxbind/exps/260902_fusion_v4_cdgv2_scratch_8gpu` (svr12) |
| weights | `checkpoint.pth.tar`, 1.6 GiB, 2026-09-05 |
| sigma | 0.9 |
| samples | `results/task2-drugdesign/Fusion-v4-cdgv2-scratch-ep350/samples/` -- 79 pockets x 10 molecules |
| frozen encoder | `model_zoo/CDG_v2/checkpoint_e0025.pth.tar` -- **required to load this checkpoint**; already in the zoo and on Dropbox |


## Recipe

`scripts/80_chain_scratch_v4_full.sh` (WARM_START=""), 350 ep so it matches the from-scratch vanilla baseline; ~686 s/epoch, ~2.8 days on 8 GPUs.

## Why it exists

Removes the warm-start drift term: the density branch is learned jointly with the denoiser instead of bolted onto a converged model.

## Generation numbers

Vina in kcal/mol, mean over molecules; PB = PoseBusters dock-mode validity;
clash/strain are PoseCheck medians. `all` = every pocket this arm sampled,
`density79` = the 79 pockets with usable deposited density (the apples-to-apples
scope across arms). Source: `results/task2-drugdesign/Fusion-v4-cdgv2-scratch-ep350/metrics.json`.

| scope | pockets | mols | vina score | vina dock | PB valid | clash med | strain med |
|---|---|---|---|---|---|---|---|
| all | 79 | 790 | -5.91 | -7.41 | 68.5 % | 8.0 | 207.5 |
| density79 | 79 | 790 | -5.91 | -7.41 | 68.5 % | 8.0 | 207.5 |

QED 0.427 - SA 0.598 - diversity 0.617 -
high-affinity 60.6 % - validity 1.000 -
docked 790/790 (0 failed).

Our arms draw 10 molecules per pocket against the published baselines' ~100, so
compare per-molecule means, not totals.

## Re-sample from it

```bash
cd voxbind
python sample.py pretrained_path=model_zoo/gen_fusion_v4_cdgv2_scratch_ep350 wjs.split=test \
    wjs.n_samples_per_pocket=10 wjs.n_targets=79
```

`cfg.yaml` + `.hydra/` are this run's training config; `train_ddp.log` is its log.
Weights and configs are git-ignored here and travel via `dropbox_push.sh`.
