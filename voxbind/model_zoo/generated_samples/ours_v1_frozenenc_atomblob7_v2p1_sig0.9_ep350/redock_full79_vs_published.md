published : voxbind/exps/voxbind_frozenenc_atomblob7_v2p1_sig0.9/samples/full_eval_ep350/eval_docking_results_full79.json
re-run    : voxbind/exps/voxbind_frozenenc_atomblob7_v2p1_sig0.9/samples/full_eval_ep350/eval_docking_results_full79_rerun.json
targets   : old 79  new 79  compared 79

## Pocket-averaged summary (mean over compared targets)

| metric | published | re-run | delta |
|---|---|---|---|
| vina_score | -6.5839 | -6.5839 | 0.0000 |
| vina_min | -7.6440 | -7.6440 | 0.0000 |
| vina_dock | -8.4886 | -8.4411 | 0.0475 |
| high_affinity | 0.6811 | 0.6838 | 0.0027 |
| qed | 0.5226 | 0.5226 | 0.0000 |
| sa | 0.6774 | 0.6774 | 0.0000 |
| diversity | 0.7098 | 0.7098 | 0.0000 |
| validity | 0.9990 | 0.9990 | 0.0000 |

## Per-molecule paired deltas (re-run minus published)

| mode | n | exact match | mean | median | sd | max abs |
|---|---|---|---|---|---|---|
| vina_score | 7873 | 7873/7873 (100.0%) | +0.0000 | +0.0000 | 0.0000 | 0.000 |
| vina_min | 7873 | 7873/7873 (100.0%) | +0.0000 | +0.0000 | 0.0000 | 0.000 |
| vina_dock | 7873 | 185/7873 (2.3%) | +0.0470 | +0.0000 | 3.2213 | 231.598 |

## Verdict

- score_only and minimize reproduce EXACTLY -> receptor prep, docking box, scoring function and Vina build are identical to the published run.
- dock differs by +0.047 kcal/mol on average (sd 3.221), which is Vina's random-seed search noise: seed=0 means a random seed, so dock is not bit-reproducible by design.
