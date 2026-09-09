# fig-vina-3line — Vina per-pocket rank charts

Moved here from `260903/` on 2026-09-09; the builder, its CSVs and its figures now sit
together, as in the other `fig-*` folders.

## Build

    /opt/conda/envs/voxbind/bin/python \
        notebook/html/260910/fig-vina-3line/build_vina_3line.py

Reads `eval_docking_results_full79.json` from two sample roots (absolute paths at the top
of the script: `_vanilla_ep923` and `voxbind_frozenenc_atomblob7_v2p1_sig0.9`) and writes
everything beside itself.

## Outputs

| file | what |
|---|---|
| `vina_{dock,score,min}_3line_{mean,median}.{png,svg,pdf}` | 79 pockets ranked by the reference ligand's score; reference as a grey dashed line, VoxBind as sand triangles, ours as periwinkle circles |
| `vina_{...}_3line_pair.{png,svg,pdf}` | mean and median side by side, on one shared y-range |
| `vina_{...}_3line_{mean,median}.csv` | the plotted numbers |

The pair figure's width is **measured** per metric so that one of its panels is the same
shape as the standalone figure (the run prints the residual, ~0.003 in). Don't replace
that measurement with a constant: the furniture outside the plot area moves with the type
size and with how wide that metric's tick labels happen to be.

Style — no panel titles, warm near-black furniture (#514F52), dotted grid — is shared with
`fig-vina-per-atom`, `fig-posecheck` and `pose_common.py`.
