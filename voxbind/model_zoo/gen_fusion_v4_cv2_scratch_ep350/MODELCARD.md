# gen_fusion_v4_cv2_scratch_ep350

Coords-only control (C_v2) through the identical fusion. Generator checkpoint for the **Fusion v4 cv2 scratch ep350** row of
`results/task2-drugdesign/Fusion-v4-cv2-scratch-ep350` (samples + per-molecule metrics live there).

| | |
|---|---|
| source run | `voxbind/exps/260905_fusion_v4_cv2_scratch_8gpu` (svr12) |
| weights | `checkpoint.pth.tar`, 1.6 GiB, 2026-09-08 |
| sigma | 0.9 |
| samples | `results/task2-drugdesign/Fusion-v4-cv2-scratch-ep350/samples/` -- 79 pockets x 10 molecules |
| frozen encoder | `model_zoo/C_v2/checkpoint_e0020.pth.tar` -- **required to load this checkpoint**; already in the zoo and on Dropbox |


## Recipe

`scripts/81_chain_cv2_scratch_after_sampling.sh`. C_v2 = the same 100M ChannelViT trunk pretrained on the same PLINDER v2 data but on coords ONLY (input_mode=atomblob, n_in=11, groups [7,4]) -- no rho, no ||grad rho||. Same fusion, capacity, subset, schedule and seed as the arm above.

## Why it exists

The matched control for the density question: the gap against `gen_fusion_v4_cdgv2_scratch_ep350` is attributable to the DENSITY CHANNELS rather than to "some frozen encoder helps".

## Generation numbers

Vina in kcal/mol, mean over molecules; PB = PoseBusters dock-mode validity;
clash/strain are PoseCheck medians. `all` = every pocket this arm sampled,
`density79` = the 79 pockets with usable deposited density (the apples-to-apples
scope across arms). Source: `results/task2-drugdesign/Fusion-v4-cv2-scratch-ep350/metrics.json`.

| scope | pockets | mols | vina score | vina dock | PB valid | clash med | strain med |
|---|---|---|---|---|---|---|---|
| all | 79 | 788 | -5.84 | -7.34 | 68.3 % | 7.0 | 223.3 |
| density79 | 79 | 788 | -5.84 | -7.34 | 68.3 % | 7.0 | 223.3 |

QED 0.422 - SA 0.596 - diversity 0.574 -
high-affinity 58.1 % - validity 1.000 -
docked 788/788 (0 failed).

Our arms draw 10 molecules per pocket against the published baselines' ~100, so
compare per-molecule means, not totals.

## Re-sample from it

```bash
cd voxbind
python sample.py pretrained_path=model_zoo/gen_fusion_v4_cv2_scratch_ep350 wjs.split=test \
    wjs.n_samples_per_pocket=10 wjs.n_targets=79
```

`cfg.yaml` + `.hydra/` are this run's training config; `train_ddp.log` is its log.
Weights and configs are git-ignored here and travel via `dropbox_push.sh`.
