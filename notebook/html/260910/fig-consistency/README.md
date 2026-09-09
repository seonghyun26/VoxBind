# Rigid-fragment consistency — local geometry

The third structural-quality measure of the VoxBind paper (Fig. 11b), alongside strain
(`../fig-posecheck`) and PoseBusters validity (`../fig-posebusters`). Same 79 pockets, same
arms, same colours — this folder shares `../pose_common.py` and `../method_colors.py` with
those two, so an arm reads the same here as it does there.

## The metric

A rigid fragment has no internal degrees of freedom. A benzene ring, an amide, a fused
bicycle each has exactly one correct shape, so a force field has no reason to move it.
**If it moves, the generated geometry was wrong to begin with.**

1. MMFF-optimise the molecule (hydrogens added first — MMFF needs them);
2. cut every rotatable bond; the connected components left are the rigid fragments;
3. per fragment, RMSD between its heavy-atom coordinates before and after, after Kabsch
   superposition (with a reflection check — a mirror image is not a rigid motion);
4. report the median, resolved by fragment size. Ångström, lower is better.

Two choices decide whether the number means anything. **Superposition**: optimising the
rest of the molecule translates and rotates every fragment bodily, and without Kabsch that
drift would swamp the shape change this is trying to measure. **Heavy atoms only**: MMFF
needs hydrogens but RDKit places them by heuristic, so counting them would measure
`AddHs`, not the model.

### Why it is not the other two metrics

| | strain (Fig 6) | rigid fragments (Fig 11b) |
|---|---|---|
| force field | UFF | MMFF |
| quantity | energy difference (kcal mol⁻¹) | coordinate RMSD (Å) |
| scope | whole-molecule conformation | inside rigid fragments only |
| catches | a globally contorted conformer | wrong bond lengths, angles, ring shape |

They are complements by construction: strain relaxes under a 0.1 Å position constraint
*precisely so* that local imperfections wash out — which is exactly what this targets.
PoseBusters asks the same local question but only through pass/fail thresholds; this is
continuous, so it still separates two models that both pass everything.

## Files

| file | what |
|---|---|
| `rigid_violin_{core,all}.*` | **violins per exact fragment size** to 9, then 10+ — range, IQR, median, mean |
| `rigid_violin_paperbins_{core,all}.*` | the same over the paper's coarser bins, for comparison with 260827 |
| `rigid_per_size_median_{core,all}.*` | median RMSD against exact fragment size, over the fragment-size distribution |
| `rigid_per_size_mean_{core,all}.*` | the same, **mean** — a separate figure, as everywhere in 260910 |
| `rigid_ecdf_pair_{core,all}.*` | the RMSD distribution: all fragments, and the 7+ atom fragments |
| `rigid_fragment_by_bin.csv` | every glyph both violins draw: binning, n, min, q25, median, mean, q75, max |
| `rigid_fragment_per_size.{json,csv}` | the line figures' curves, per exact fragment size |
| `rigid_fragment_summary.json` | pooled numbers, `p79` and `all_pockets`, plus the paper's bins |

**One bin per exact fragment size, not the paper's pooled bins.** The models hold
567–10,952 fragments at every size from 2 to 9, so there is nothing to pool, and pooling
costs something real: the paper's `5–6` bin puts benzene — 6 atoms, 21 % of both VoxBind
arms' fragments, and the size where they come closest to crystal quality — in with the
thin and much worse 5-atom fragments, and the finding disappears. Resolved, the two sizes
are opposite ends of the axis: 5 atoms is a sharp local peak, worse than every size up to
12 or 13 (VoxBind 0.164, Ours v1 0.164, TargetDiff 0.180, against the crystal 0.066), while
6 atoms is among the best anywhere (0.029 / 0.035 / 0.085, against 0.027). Pooled into one
`5–6` bin they read as a single flat 0.033 / 0.040 / 0.111 and neither is visible.

`rigid_violin_paperbins_*` keeps the coarse grouping so the numbers stay directly
comparable to `../../260827/rigid_fragment_consistency.html`, which reported them that way.
It is not the figure to read.

Only the 79 crystal ligands are thin enough to want pooling — 9 to 15 fragments at sizes 4,
5, 8 and 9 — and that is handled where it belongs: **a violin with fewer than 20 values is
drawn as its glyphs alone, with no KDE body**, because a density fitted to nine points is a
shape invented from noise. The run log names every bin this touches on each build.

**Violins, not bars, because one number per bin is not the finding.** The distribution
inside a bin is wide and strongly right-skewed, so each violin carries the KDE plus the
full **min–max range**, the **IQR** box, the **median** (white dot) and the **mean** (dark
diamond). The mean sits above the median in every arm and every bin — that is the skew made
visible, and it is why the median is the number reported everywhere else in this folder.
The crystal reference is hatched grey, the violin analogue of the dashed line the other
figures use for it.

Two mechanics behind that figure. **The KDE is fitted in log space and the axis relabelled
with real values** — fragment RMSD spans four decades and is right-skewed, so a linear
violin is a spike on the floor, and matplotlib fits the density in data space, so setting a
log scale afterwards would warp the drawn shape rather than the fit. Same trick and same
reason as the clash violins in `../fig-posecheck`. **And the y axis is floored at 1e-4 Å**:
the 2-atom bin reaches 8.6e-08 — a bond MMFF did not move at all — seven decades below that
bin's median, and an axis reaching it would squash every violin into a line. Values below
the floor are drawn at it; that is 0.08–0.34 % of each arm (45 of VoxBind's 31,094, 29 of
Ours v1's 30,418, 21 of TargetDiff's 27,159, 1 of the crystal set's 294), all in the 2-atom
bin, and the run log prints the count on every build.

`_core` is VoxBind, VoxBind + Ours and the crystal reference; `_all` adds TargetDiff. Same
split as the Vina and pose figures, and it is part of the filename.

**The x axis counts atoms in the FRAGMENT, not in the ligand.** Every other size-resolved
figure in 260910 plots against ligand size; this metric has no per-molecule value, only
per-fragment ones, and fragment size is what drives it. Do not read a point here against a
point in `../fig-posecheck` at the same x.

Two details behind the curves, both in the builder's docstrings. **The crystal reference is
drawn per exact fragment size and stops at 10 atoms**, rather than windowed the way
pose_common windows it over ligand size. Windowing was tried and was actively misleading:
the fragment-size distribution spikes at 6 (benzene), where the 33 crystal fragments sit at
0.027 Å, and a ±1 window pools that spike with the thin, worse-scoring 5s and 7s, lifting
the plotted point to 0.041 — *above* both models, reversing the comparison the line exists
to make. 294 crystal fragments cannot support a per-size curve past ~10 atoms, so it stops
there and the pooled 7+ comparison is made in the ECDF figure instead. **And the x range
stops at the first gap** in the sizes clearing `MIN_N`, so no line crosses a dropped count:
sizes 25–27 are too thin and 28 is not, so 28 is dropped rather than joined to 24.

## Numbers over the 79 pockets

| arm | fragments | median (Å) | 2–6 atoms | 7+ atoms | share 7+ | MMFF failures |
|---|---|---|---|---|---|---|
| TargetDiff | 27,159 | 0.0533 | 0.0371 | 0.3244 | 28.7 % | 6.8 % |
| VoxBind | 31,094 | 0.0378 | 0.0266 | 0.0992 | 25.8 % | 3.9 % |
| VoxBind + Ours | 30,418 | 0.0467 | 0.0316 | 0.1149 | 29.5 % | 6.0 % |
| *Reference ligand* | *294* | *0.0402* | *0.0186* | *0.0671* | *29.9 %* | *0 %* |

**Read the stratified columns, not the pooled one.** VoxBind's pooled median (0.0378) comes
out *below* the crystal ligands' (0.0402) while being *above* them in both size strata —
a Simpson's reversal driven purely by fragment mix, since VoxBind draws the fewest large
fragments (25.8 % against the crystal set's 29.9 %) and large fragments score worst. This
is the same size confound the Vina numbers have, in a different metric.

Median RMSD by fragment size, the paper's bins:

| fragment size | TargetDiff | VoxBind | VoxBind + Ours | *Reference* |
|---|---|---|---|---|
| 2 | 0.0210 | 0.0138 | 0.0166 | *0.0058* |
| 3–4 | 0.0504 | 0.0470 | 0.0482 | *0.0491* |
| 5–6 | 0.1105 | 0.0329 | 0.0404 | *0.0324* |
| 7–9 | 0.1864 | 0.0782 | 0.0844 | *0.0576* |
| 10–13 | 0.3342 | 0.1085 | 0.1243 | *0.0609* |
| 14+ | 0.6422 | 0.3349 | 0.2915 | *0.2837* |

## What to carry into the writeup

1. **Local geometry is not where the fusion model loses.** Ours v1 trails VoxBind by
   0.009 Å pooled and by 0.001–0.016 Å in every size bin but the largest, where it is
   0.043 Å *better*. For scale, heavy-atom coordinate precision in a good crystal structure
   is ~0.1–0.2 Å, so every one of those gaps is inside the noise a crystallographer would
   ignore — while over the same molecules PoseCheck strain rises 62 → 82 kcal mol⁻¹ and
   clashes 4 → 5. The deficit those two report therefore lives in **global conformation and
   placement**, not in how the model draws a ring. The honest counter-argument: the
   *relative* gap here (+24 %) is not far off strain's (+32 %), and it is the absolute
   magnitude that makes it negligible.
2. **TargetDiff is the arm this separates.** It runs 2–3× above both VoxBind arms from 5
   atoms upward — 0.111 against 0.033/0.040 at 5–6 atoms, 0.334 against 0.109/0.124 at
   10–13, and 0.324 against 0.099/0.115 pooled over 7+ atom fragments — and its 7+ ECDF is
   displaced bodily to the right. That is the same ordering PoseCheck and PoseBusters give
   from different measurements, which is the reason to trust any of them. It is *not* the
   Vina ordering.
3. **Both VoxBind arms sit just above the crystal ligands at every fragment size, and the
   gap widens with size.** At the benzene spike (6 atoms, a fifth of all their fragments)
   they are within 0.002–0.009 Å of crystal — 0.029 / 0.035 against 0.027. Pooled over the
   7+ atom fragments they are 1.5–1.7× above it (0.099 / 0.115 against 0.067) while
   TargetDiff is 4.8× (0.324). Medium and large rigid systems — fused rings, extended
   conjugation — are where voxel-decoded geometry is least accurate, and both VoxBind arms
   share that limitation in the same mild form.
4. **Five-atom rigid fragments are a sharp local failure, for every arm including
   TargetDiff.** At 5 atoms all three sit at 0.16–0.18 Å — worse than anything up to 12–13
   atoms and 2.5× the crystal ligands' 0.066 — then drop by 5× at 6 atoms. Five-membered
   rings and small non-aromatic rigid systems are where voxel-decoded geometry is least
   accurate relative to size, and the effect is model-independent, which points at the
   representation rather than at any one model. This is only visible per exact size; the
   paper's binning hides it.
5. **The largest bin reverses.** At 14+ atoms Ours v1 (0.2915) beats VoxBind (0.3349) and
   nearly matches the reference (0.2837) — the same reversal strain and clashes show.
6. **The MMFF failure rate is itself a result, and it is the caveat.** Molecules MMFF
   cannot parameterise are silently excluded from every number above: 3.9 % for VoxBind,
   6.0 % for Ours v1, 6.8 % for TargetDiff. If those failures are enriched for bad geometry
   — plausible, since parameterisation fails on unusual valences and atom types — then
   every arm's number is optimistic and the two worse arms' more so. That ordering is the
   clearest local-chemistry signal in this analysis and it agrees with the rest.

## Provenance

Evaluation: `voxbind/exps/frozenenc_probes/eval_rigid_fragments.py`, CPU only, no receptor
needed, ~45 s per run at 12 workers. It writes `rigid_fragment_results.json` into each run
root; `--reference` scores the deposited ligand in each target directory instead and writes
`rigid_fragment_reference.json`. This folder only draws what that wrote.

```bash
V=/opt/conda/envs/voxdock/bin/python
E=voxbind/exps
$V $E/frozenenc_probes/eval_rigid_fragments.py $E/_vanilla_ep923/samples/full_eval_ep923 --workers 12
$V $E/frozenenc_probes/eval_rigid_fragments.py $E/voxbind_frozenenc_atomblob7_v2p1_sig0.9/samples/full_eval_ep350 --workers 12
$V $E/frozenenc_probes/eval_rigid_fragments.py /home1/irteam/base_drug/eval/targetdiff --workers 12
$V $E/frozenenc_probes/eval_rigid_fragments.py $E/voxbind_frozenenc_atomblob7_v2p1_sig0.9/samples/full_eval_ep350 --reference

/opt/conda/envs/voxbind/bin/python notebook/html/260910/fig-consistency/build_rigid_fragment.py
```

The three arms were scored on 2026-08-20 (`frozenenc_probes/logs/rigidfrag_*.log`,
`td_rigidfrag.log`); the reference on 2026-09-09 for this folder, reproducing the numbers
`../../260827/rigid_fragment_summary.json` already carried, fragment for fragment.

Two differences from `../../260827/rigid_fragment_consistency.html`, which reported the
same metric on the arms as they stood then:

* **TargetDiff is now on the same 79 pockets as everything else.** 260827 scored it over
  its own 100 and 92 pockets, which is not like-for-like against arms that only sampled the
  79. The p79 median (0.0533) is within 0.0002 of the 100-pocket one, so nothing moves — but
  the figures here are matched by construction, not by luck.
* Arms are named as in the rest of 260910: *VoxBind* is 260827's "Vanilla σ=0.9",
  *VoxBind + Ours* its "Holo ED fusion σ=0.9 (atomblob7 v2p1)".

`exps/samples_reference_receptor_ed_ep350` (Ours v2, 92 pockets) has been scored too and
its JSON is on disk; it is not one of the arms this section reports, matching the decision
in `../pose_common.py`. Re-adding it is one line there.
