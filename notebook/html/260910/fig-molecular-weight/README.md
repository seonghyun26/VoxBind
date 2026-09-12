# Molecular weight — what each arm actually makes

How much of each arm's output sits at each molecular weight. Same 79 pockets, same arms,
same colours as every other 260910 figure — it shares `../pose_common.py` and
`../method_colors.py` with `../fig-posecheck`, `../fig-posebusters` and `../fig-consistency`.

**This is not a quality metric.** Nothing here says an arm is better. It is the size prior
that every quality metric in 260910 is read against: Vina score grows with molecule size,
strain grows with size, rigid-fragment RMSD grows with fragment size. "Which arm wins" only
means something once you know whether the arms are drawing from the same weight
distribution — and they are not.

**Naming.** The method is **CoDE** (`\textsc{CoDE}` in the .tex files; matplotlib has no
small caps without a TeX backend, so the figures carry the plain string). Vanilla VoxBind
is drawn as **VoxBind$_{\sigma=0.9}$** but stored as plain `VoxBind` — legend text that
differs from the data key lives in `../method_colors.py`'s `DISPLAY`, so every exported
CSV column and JSON key stays plain and joinable.

## Files

| file | what |
|---|---|
| `mw_distribution_{core,all}.*` | share of an arm's molecules per 20 Da bin |
| `mw_ecdf_{core,all}.*` | the same distributions, cumulative — the form to read a shift off |
| `mw_histogram.csv` | the plotted bins: arm, bin, count, share |
| `mw_summary.json` | n, mean, median, quartiles, tails; `p79` and `all_pockets` |

`_core` is VoxBind, CoDE and the crystal reference; `_all` adds TargetDiff and the five published baselines — nine series, so it drops the fill and reads as lines.
At nine series the **ECDF is the figure to read**; the histogram is there to show where the mass sits, and nine overlaid step curves cross a great deal.

**Each curve is normalised by its own total**, which is the only way 79 crystal ligands and
7,888 generated ones share an axis. Absolute counts are in `mw_histogram.csv`.

**Where the numbers come from.** RDKit `Descriptors.MolWt` (average mass, implicit
hydrogens included) on every molecule in `target_*/samples.sdf`.

It used to read the SMILES in each target's `metrics.json`, and that had to change to cover
the five baselines: their sample dirs are staged and complete, but their `metrics.json`
files appear one pocket at a time as the PoseBusters run finishes them. A size prior should
not depend on the progress of a scoring run — the molecules exist either way. **The two
sources agree exactly**: over every arm whose metrics are complete, SDF and recorded SMILES
give the same weight for all 23,048 molecules, which the run log re-checks on every build.

**Two smoothing/clipping choices, both reported rather than hidden.** The crystal reference
is 79 molecules over ~35 occupied bins — about two per bin — so its raw histogram is a
picket fence of 2.5 %-tall spikes; it is rolled over ±2 bins and *averaged*, keeping it on
the same per-bin scale as the models, exactly as `pose_common.size_distribution` rolls it
over ligand size. The ECDF is not rolled — 79 molecules are 79 honest steps. And the x axis
stops at 800 Da: molecules above it are **not** folded into the last bin (0.25 % / 0.93 % /
0.29 % / 1.27 % of each arm), so the plotted shares sum to just under 100.

## Numbers over the 79 pockets

Sorted by median, lightest first — which is also the order the ECDF draws them in:

| arm | molecules | median | mean | IQR | ≤200 Da | ≤300 Da | >500 Da |
|---|---|---|---|---|---|---|---|
| Pocket2Mol | 7,772 | 209.2 | 252.8 | 149–341 | 47.1 % | 69.4 % | 6.8 % |
| AR | 7,655 | 222.4 | 240.9 | 163–309 | 40.5 % | 72.8 % | 1.3 % |
| FuncBind | 7,895 | 266.3 | 278.6 | 176–365 | 32.5 % | 58.3 % | 4.9 % |
| DecompDiff | 6,427 | 281.3 | 303.5 | 186–414 | 28.4 % | 55.3 % | 11.5 % |
| DiffSBDD | 7,720 | 286.4 | 294.8 | 188–381 | 28.3 % | 54.3 % | 6.3 % |
| TargetDiff | 7,287 | 317.4 | 315.2 | 199–406 | 25.1 % | 45.2 % | 7.4 % |
| VoxBind | 7,888 | 336.4 | 336.3 | 230–418 | 19.7 % | 40.3 % | 9.9 % |
| CoDE | 7,873 | **354.3** | 348.4 | 238–441 | 18.9 % | 35.9 % | 12.9 % |
| *Reference ligand* | *79* | *320.3* | *332.2* | *201–451* | *25.3 %* | *48.1 %* | *17.7 %* |

Molecule counts match `../fig-posecheck`'s exactly for all eight arms, which is the cheapest
check that this folder and that one are reading the same samples.

## What to carry into the writeup

1. **CoDE is the heaviest arm in the section, and the weight ordering tracks the Vina
   ordering.** Its median is 18 Da above VoxBind, 37 above TargetDiff and 145 above
   Pocket2Mol; it puts 12.9 % of its output over 500 Da where AR puts 1.3 %. Vina score is
   close to additive in heavy atoms, so a systematically heavier arm scores better without
   binding better. **This is the confound behind the Vina ordering**, and with the five
   baselines drawn it is no longer a two-arm quibble: the span from Pocket2Mol to CoDE is
   145 Da of median, far more than any binding claim in this section rests on. Read the
   per-atom and ligand-efficiency views before ranking anything on Vina.
2. **Neither VoxBind arm matches the crystal ligands' shape, and the mismatch is in the
   tails, not the centre.** All four medians sit within 37 Da of each other (317–354), but
   the crystal set is the widest by far: 25.3 % at or below 200 Da *and* 17.7 % above 500,
   against CoDE's 18.9 % and 12.9 %. Crystal ligands are bimodal — real binders are
   either fragment-sized or large — and every model fills the middle instead. Read that off
   `mw_distribution_*`: the reference peaks at 200–220 Da and then runs *below* both VoxBind
   arms across 300–440 Da, which is exactly where they peak (VoxBind at 380–400, CoDE at
   340–360). Every baseline peaks lower still — TargetDiff at 160–180 Da, Pocket2Mol and AR
   at 120–180.
3. **Every published baseline is lighter than every VoxBind-family arm**, without
   exception: the five run 209–286 Da against VoxBind's 336 and CoDE's 354, and the gap is
   in the small end — Pocket2Mol puts 47 % of its output under 200 Da and AR 41 %, against
   19–20 % for the VoxBind arms. This cuts *against* the VoxBind arms on every quality
   metric in the section: strain, clashes and fragment RMSD all get harder as molecules
   grow, so the baselines are being scored on an easier size distribution and their
   deficits are understated, not inflated. TargetDiff is the closest baseline on weight
   (317 Da) and is also the one the quality metrics separate most clearly — that
   comparison is the least confounded of the seven.
4. **`all_pockets` and `p79` agree** for VoxBind (333.5 / 336.4) and TargetDiff, so the
   79-pocket restriction is not itself selecting for weight. CoDE and the five baselines
   only ever covered the 79, so their two blocks are identical by construction.
5. **TargetDiff is the only arm that emits disconnected molecules** — 511 of 7,798 over the
   79 pockets, 6.6 % — and they are dropped here, as `metrics.py` drops them before scoring
   anything else. Their weight is not a ligand's weight; it is the sum of two or more pieces
   that never bonded. Every other arm emits none.

## Rebuild

```bash
/opt/conda/envs/voxbind/bin/python \
    notebook/html/260910/fig-molecular-weight/build_molecular_weight.py
```

Reads only `metrics.json` under each arm's run root (`../pose_common.py` owns those paths).
Nothing is re-sampled, re-docked or re-scored, so it is seconds and is safe to re-run.
