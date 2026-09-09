published : voxbind/exps/voxbind_frozenenc_atomblob7_v2p1_sig0.9/samples/full_eval_ep350/eval_docking_results.json
re-run    : voxbind/exps/voxbind_frozenenc_atomblob7_v2p1_sig0.9/samples/full_eval_ep350/eval_docking_results_rerun.json
targets   : old 79  new 79  compared 79

## Pocket-averaged summary (mean over compared targets)

| metric | published | re-run | delta |
|---|---|---|---|
| vina_score | -5.5134 | -5.5134 | 0.0000 |
| vina_min | -7.0898 | -7.0898 | 0.0000 |
| vina_dock | -8.2382 | -8.2562 | -0.0180 |
| high_affinity | 0.6752 | 0.6767 | 0.0016 |
| qed | 0.5226 | 0.5226 | 0.0000 |
| sa | 0.6774 | 0.6774 | 0.0000 |
| diversity | 0.7098 | 0.7098 | 0.0000 |
| validity | 0.9990 | 0.9990 | 0.0000 |

## Per-molecule paired deltas (re-run minus published)

| mode | n | exact match | mean | median | sd | max abs |
|---|---|---|---|---|---|---|
| vina_score | 7774 | 7774/7774 (100.0%) | +0.0000 | +0.0000 | 0.0000 | 0.000 |
| vina_min | 7774 | 7774/7774 (100.0%) | +0.0000 | +0.0000 | 0.0000 | 0.000 |
| vina_dock | 7774 | 174/7774 (2.2%) | -0.0181 | -0.0010 | 1.8791 | 82.930 |

## Verdict

- score_only and minimize reproduce EXACTLY -> receptor prep, docking box, scoring function and Vina build are identical to the published run.
- dock differs by -0.018 kcal/mol on average (sd 1.879), which is Vina's random-seed search noise: seed=0 means a random seed, so dock is not bit-reproducible by design.
