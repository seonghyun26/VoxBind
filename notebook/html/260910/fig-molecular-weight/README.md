# Molecular weight — what each arm actually makes

How much of each arm's output sits at each molecular weight. Same 79 pockets, same arms,
same colours as every other 260910 figure — it shares `../pose_common.py` and
`../method_colors.py` with `../fig-posecheck`, `../fig-posebusters` and `../fig-consistency`.

**This is not a quality metric.** Nothing here says an arm is better. It is the size prior
that every quality metric in 260910 is read against: Vina score grows with molecule size,
strain grows with size, rigid-fragment RMSD grows with fragment size. "Which arm wins" only
means something once you know whether the arms are drawing from the same weight
distribution — and they are not.

## Files

| file | what |
|---|---|
| `mw_distribution_{core,all}.*` | share of an arm's molecules per 20 Da bin |
| `mw_ecdf_{core,all}.*` | the same distributions, cumulative — the form to read a shift off |
| `mw_histogram.csv` | the plotted bins: arm, bin, count, share |
| `mw_summary.json` | n, mean, median, quartiles, tails; `p79` and `all_pockets` |

`_core` is VoxBind, VoxBind + Ours and the crystal reference; `_all` adds TargetDiff.

**Each curve is normalised by its own total**, which is the only way 79 crystal ligands and
7,888 generated ones share an axis. Absolute counts are in `mw_histogram.csv`.

**Where the numbers come from.** RDKit `Descriptors.MolWt` (average mass, implicit
hydrogens included) on the SMILES each target's `metrics.json` already records — the same
string every other 260910 figure identifies a molecule by, so no sample-to-SDF index slip
is possible. All 23,127 SMILES parse; the run log prints the unparsed count each time and
it is 0 for every arm.

**Two smoothing/clipping choices, both reported rather than hidden.** The crystal reference
is 79 molecules over ~35 occupied bins — about two per bin — so its raw histogram is a
picket fence of 2.5 %-tall spikes; it is rolled over ±2 bins and *averaged*, keeping it on
the same per-bin scale as the models, exactly as `pose_common.size_distribution` rolls it
over ligand size. The ECDF is not rolled — 79 molecules are 79 honest steps. And the x axis
stops at 800 Da: molecules above it are **not** folded into the last bin (0.25 % / 0.93 % /
0.29 % / 1.27 % of each arm), so the plotted shares sum to just under 100.

## Numbers over the 79 pockets

| arm | molecules | median | mean | IQR | ≤200 Da | ≤300 Da | >500 Da |
|---|---|---|---|---|---|---|---|
| TargetDiff | 7,287 | 317.4 | 315.2 | 199–406 | 25.1 % | 45.2 % | 7.4 % |
| VoxBind | 7,888 | 336.4 | 336.3 | 230–418 | 19.7 % | 40.3 % | 9.9 % |
| VoxBind + Ours | 7,873 | **354.3** | 348.4 | 238–441 | 18.9 % | 35.9 % | 12.9 % |
| *Reference ligand* | *79* | *320.3* | *332.2* | *201–451* | *25.3 %* | *48.1 %* | *17.7 %* |

## What to carry into the writeup

1. **The arms are ordered by weight, and it is the same order as their Vina scores.**
   Ours v1's median is 18 Da above VoxBind's and 37 Da above TargetDiff's, and it puts
   12.9 % of its output above 500 Da against VoxBind's 9.9 % and TargetDiff's 7.4 %. Vina
   score is close to additive in heavy atoms, so a systematically heavier arm scores better
   without binding better. **This is the confound behind the Vina ordering** — the same one
   the per-atom and ligand-efficiency views exist to control for, now stated directly.
2. **Neither VoxBind arm matches the crystal ligands' shape, and the mismatch is in the
   tails, not the centre.** All four medians sit within 37 Da of each other (317–354), but
   the crystal set is the widest by far: 25.3 % at or below 200 Da *and* 17.7 % above 500,
   against Ours v1's 18.9 % and 12.9 %. Crystal ligands are bimodal — real binders are
   either fragment-sized or large — and every model fills the middle instead. Read that off
   `mw_distribution_*`: the reference peaks at 200–220 Da and then runs *below* both VoxBind
   arms across 300–440 Da, which is exactly where they peak (VoxBind at 380–400, Ours v1 at
   340–360). TargetDiff peaks lower still, at 160–180 Da.
3. **TargetDiff is the lightest arm**, which cuts the other way from every quality metric
   in this section: it is worst on strain, clashes and fragment RMSD *while* generating the
   smallest molecules, and all three of those metrics get harder as molecules grow. Its
   deficit is therefore understated by the pooled numbers, not inflated by them.
4. **`all_pockets` and `p79` agree** for VoxBind (333.5 / 336.4) and TargetDiff
   (322.8 / 317.4), so the 79-pocket restriction is not itself selecting for weight. Ours v1
   only ever sampled the 79, so its two blocks are identical by construction.

## Rebuild

```bash
/opt/conda/envs/voxbind/bin/python \
    notebook/html/260910/fig-molecular-weight/build_molecular_weight.py
```

Reads only `metrics.json` under each arm's run root (`../pose_common.py` owns those paths).
Nothing is re-sampled, re-docked or re-scored, so it is seconds and is safe to re-run.
