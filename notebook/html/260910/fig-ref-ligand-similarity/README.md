# fig-ref-ligand-similarity — how much do generated molecules look like the crystal ligand?

Moved here on 2026-09-09 from `notebook/html/` (the metric builder) and `260903/` (the
figure builder and the appendix). The whole chain — measure, tabulate, draw — is now in
this one folder.

## Chain

    cd notebook/html/260910/fig-ref-ligand-similarity
    /opt/conda/envs/voxbind/bin/python build_reference_similarity.py --novelty
    /opt/conda/envs/voxbind/bin/python build_similarity_figure.py
    /opt/conda/envs/voxbind/bin/python build_ligand_similarity_appendix.py

1. **`build_reference_similarity.py`** reads every method's `samples.sdf` (absolute sample
   roots at the top of the file) and writes `reference_similarity.{json,csv,tex}`,
   `_wrap.tex` and `_table.html`. `--novelty` adds the training-set columns and is the
   slow part; it caches its CrossDocked index under `voxbind/dataset/data/`.
2. **`build_similarity_figure.py`** only *draws* `reference_similarity.csv`, so the figure
   cannot drift from the table. `--all` also rebuilds the three rejected candidates
   (`similarity_{b_bars,c_scatter,d_table}`), which `similarity_variants.html` compares.
3. **`build_ligand_similarity_appendix.py`** assembles `ligand_similarity_appendix.txt`
   from those outputs — paper table, body paragraph, measured values, reproduction notes.

`reference_similarity_table.html` is read by `notebook/results/latex-task2.ipynb`.

## The two figures

| file | left panel | right panel |
|---|---|---|
| `similarity_a_dumbbell` | ECFP4 Tanimoto to the pocket's crystal ligand: mean (filled) and median (hollow) | scaffold match, % |
| `similarity_novelty` | novelty vs. the training set: molecule (filled) and scaffold (hollow) | SNN to the training set |

Same nine rows, same order, same geometry, so the two stack. **Their directions differ**:
on `_a_dumbbell` and on the SNN panel low is more novel, on the novelty panel high is.

## Reading the numbers

ECFP4 (Morgan r=2 / 2048-bit) puts two *unrelated* real ligands at ~0.09 — measured, see
the permutation control in the appendix — so the ~0.10 column means "as related as two
random drugs", not "our molecules are near-copies". Quote that floor with the table.

AR / Pocket2Mol / DiffSBDD / DecompDiff were sampled on the Blackwell box and only their
aggregates came back, so they have no novelty columns; `blackwell_similarity_request.md`
is the request that would fill them in.
