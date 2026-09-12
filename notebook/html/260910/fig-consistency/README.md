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

**Naming.** The method is **CoDE** (`\textsc{CoDE}` in the .tex files; matplotlib has no
small caps without a TeX backend, so the figures carry the plain string). Vanilla VoxBind
is drawn as **VoxBind$_{\sigma=0.9}$** but stored as plain `VoxBind` — legend text that
differs from the data key lives in `../method_colors.py`'s `DISPLAY`, so every exported
CSV column and JSON key stays plain and joinable.

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

**One bin per exact fragment size, not the paper's pooled bins.** Every arm holds hundreds
to thousands of fragments at each size from 2 to 9, so there is nothing to pool, and pooling
costs something real: the paper's `5–6` bin puts five-membered rings in with benzene, and
those two sizes are opposite ends of this metric. Five atoms is a sharp local peak for
seven of the eight arms *and for the crystal ligands* (0.164 / 0.164 / 0.180 for VoxBind,
CoDE and TargetDiff, 0.066 for the reference); six atoms is among the best sizes anywhere
(0.029 / 0.035 / 0.084, reference 0.027). Pooled into one `5–6` bin they read as a single
flat number and neither is visible.

`rigid_violin_paperbins_*` keeps the coarse grouping so the numbers stay directly
comparable to `../../260827/rigid_fragment_consistency.html`, which reported them that way.
It is not the figure to read.

Only the 79 crystal ligands are thin enough to want pooling — 9 to 15 fragments at sizes 4,
5, 8 and 9, which is also why they get no size-standardised summary — and that is handled where it belongs: **a violin with fewer than 20 values is
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
reason as the clash violins in `../fig-posecheck`. **And the y axis is floored at 1e-3 Å**:
fragment RMSD reaches 8.6e-08 — a bond MMFF did not move at all — decades below any bin's
median, and an axis reaching it would squash every violin into a line. The floor sits where
the low tail stops carrying anything: 0.97 % of the 201,568 fragments across all nine series
fall below it, almost all 2-atom, and dropping to 1e-4 would cost a whole decade of panel
height to reach 0.09 % more. Values below the floor are drawn at it, and the run log prints
the share for every arm on each build (0.35–1.57 % of each model arm, 2.7 % of the 294
crystal fragments).

**There is no size-1 bin**, and that is by construction rather than for want of data: a
one-atom fragment has RMSD 0 after superposition no matter what the model did with it — a
point laid on a point is exactly coincident — so `eval_rigid_fragments.py` sets
`MIN_FRAG_ATOMS = 2` and never records them. Including them would add a spike of exact
zeros whose height measures how many terminal atoms an arm's rotatable bonds cut off, not
how good its geometry is. Two atoms is the smallest fragment that carries anything: a bond
length.

`_core` is VoxBind, CoDE and the crystal reference; `_all` adds TargetDiff and the five
published baselines — nine series. Same split as the Vina and pose figures, and it is part
of the filename.

**`rigid_violin_all` wraps to two rows.** Nine series over fourteen sizes is 126 violins,
and in one row at a width where a violin is still a shape rather than a coloured tick that
is a figure fifty inches across. So the figure adds a row — and **two is the cap**: a third
band makes the reader hunt for a size across three strips, and the extra width needed to
keep violins readable in two rows is the price of avoiding that. `core` (3 series × 14 =
42) stays one row; `all` splits 7 sizes over 7 at 18 in wide.

**Each row is cropped to the decades its own bins reach, and pays for them in height.**
RMSD grows with fragment size, so the 9+ atom row never comes within a decade of the 1e-3
floor the 2-atom fragments sit on — its smallest drawn fragment is 0.018 Å. A shared bottom
spends a third of that panel on a band holding nothing, so the row starts at **1e-2**
instead, and its panel is shortened by exactly the decade it gave up: **a decade is the same
physical height in both rows**, which is what keeps them comparable — not a shared range —
and is what stops a cropped row from silently stretching its violins. The top is shared and
never cropped. Nothing is hidden by this: the crop stops at the enclosing decade of the
row's own minimum, computed from the drawn bins, and the run log prints the floor each row
ended up with.

**The rows split the size bins, never the arms** — comparing two arms must never mean
looking at two panels. Both rows span the same number of slots even where the last is
empty, so a violin is exactly as wide in each. **Both keys sit along the bottom of the last
row**, methods right in a 3×3 block and the glyph key left: the violins climb rightward and
upward with fragment size, so that band is the emptiest strip in the figure, and splitting
the keys to opposite ends of it keeps each clear of the other and of the data. A tall
single-column method key would run up into the violins above it, and in the two-row figure
the glyph key goes in one row of four rather than 2×2, since the cropped row is shallower
and the figure is 18 in wide.

**Neither key may cover a whisker, and that strip is measured rather than assumed.** The
figure is laid out once, each key is asked how tall it actually came out, that height is
converted to decades through the panel's own scale, and the row is handed back exactly the
shortfall — per half of the row, since the two halves bottom out in different places. What
is added is blank axis *below the lowest tick*, so 1e-2 stays the bottom label of the lower
row and 1e-3 of the upper one. A `min–max` whisker hidden under the key that draws the
`min–max` glyph is the one failure this figure cannot afford.

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

`std med` re-weights every arm onto **one common fragment-size mix** — see below; it is the
column to rank on. Sorted by it.

| arm | fragments | pooled med | **std med** | mean | molecules | MMFF failures |
|---|---|---|---|---|---|---|
| VoxBind | 31,094 | 0.0378 | **0.0601** | 0.079 | 7,888 | 3.9 % |
| CoDE | 30,418 | 0.0467 | **0.0654** | 0.089 | 7,873 | 6.0 % |
| Pocket2Mol | 17,618 | 0.0962 | 0.0873 | 0.149 | 7,772 | 0.0 % |
| DecompDiff | 23,613 | 0.0431 | 0.1048 | 0.118 | 6,427 | 2.9 % |
| FuncBind | 23,006 | 0.0947 | 0.1194 | 0.134 | 7,895 | 12.2 % |
| TargetDiff | 24,917 | 0.0531 | 0.1221 | 0.154 | 7,287 | 6.0 % |
| DiffSBDD | 29,670 | 0.0726 | 0.1424 | 0.147 | 7,720 | 0.5 % |
| AR | 20,938 | 0.0954 | 0.1572 | 0.182 | 7,655 | 5.9 % |
| *Reference ligand* | *294* | *0.0402* | *n/a* | *0.067* | *79* | *0 %* |

Median RMSD (Å) per exact fragment size — the `rigid_violin_*` figures' own grouping.
The crystal reference holds 2–4 fragments at sizes 11–14, too few to plot or to read,
so those cells are blank in the figure and italic here:

| fragment size | AR | Pocket2Mol | DiffSBDD | DecompDiff | FuncBind | TargetDiff | VoxBind | CoDE | *Reference* |
|---|---|---|---|---|---|---|---|---|---|
| **2** | 0.024 | 0.031 | 0.025 | 0.016 | 0.038 | 0.021 | 0.014 | 0.017 | *0.006* |
| **3** | 0.071 | 0.059 | 0.066 | 0.041 | 0.071 | 0.048 | 0.045 | 0.046 | *0.043* |
| **4** | 0.141 | 0.100 | 0.130 | 0.075 | 0.120 | 0.080 | 0.091 | 0.094 | *0.079* |
| **5** | 0.158 | 0.119 | 0.169 | 0.103 | 0.184 | 0.180 | 0.164 | 0.164 | *0.066* |
| **6** | 0.185 | 0.094 | 0.130 | 0.044 | 0.129 | 0.084 | 0.029 | 0.035 | *0.027* |
| **7** | 0.257 | 0.097 | 0.194 | 0.124 | 0.159 | 0.128 | 0.063 | 0.070 | *0.058* |
| **8** | 0.322 | 0.107 | 0.266 | 0.233 | 0.213 | 0.226 | 0.091 | 0.107 | *0.050* |
| **9** | 0.332 | 0.143 | 0.302 | 0.281 | 0.238 | 0.276 | 0.103 | 0.112 | *0.064* |
| **10** | 0.298 | 0.148 | 0.307 | 0.261 | 0.211 | 0.271 | 0.087 | 0.101 | *0.056* |
| **11** | 0.404 | 0.151 | 0.342 | 0.343 | 0.251 | 0.295 | 0.127 | 0.129 | *0.066* |
| **12** | 0.437 | 0.169 | 0.411 | 0.383 | 0.284 | 0.381 | 0.156 | 0.179 | *0.119* |
| **13** | 0.470 | 0.199 | 0.472 | 0.413 | 0.307 | 0.423 | 0.172 | 0.180 | *0.124* |
| **14** | 0.355 | 0.223 | 0.539 | 0.449 | 0.292 | 0.471 | 0.167 | 0.204 | *0.470* |
| **15+** | 0.607 | 0.348 | 0.689 | 0.620 | 0.411 | 0.678 | 0.424 | 0.324 | *0.217* |

### The pooled median cannot be compared across arms

Every arm draws a different mix of fragment sizes, and small fragments score better, so the
pooled column measures the mix as much as the geometry. **DecompDiff is the proof, not a
hypothetical**: pooled it is second best of the eight (0.0431, below CoDE' 0.0467), and it
is *worse than both VoxBind arms at every fragment size from 5 atoms up* — 0.233 against
0.091/0.107 at 8 atoms. Standardised it lands sixth. Pocket2Mol moves the other way, from
worst pooled to best baseline.

`std med` is each arm's median at every fragment size, averaged over the pooled size mix of
all nine series. It is an average of medians, not a median — a median cannot be re-weighted.
An arm gets one only if the sizes it clears 25 fragments at cover ≥90 % of that mix; the
79 crystal ligands reach 65 %, at the four sizes crystal geometry is best at, so they get
`n/a` rather than a flattering number that would read as like-for-like.

## What to carry into the writeup

1. **Both VoxBind arms beat every published baseline on local geometry, and it is not
   close.** Size-standardised, VoxBind is 0.060 and CoDE 0.065 against 0.087–0.157 for the
   seven baselines — the nearest, Pocket2Mol, is 33 % above CoDE and the farthest, AR, is
   2.4×. Per size the separation opens at 6 atoms and holds all the way out: across sizes
   7–13 the VoxBind arms run 0.063–0.180 while the baselines run 0.097–0.472, and only
   Pocket2Mol is ever within 50 % of them.
2. **That ordering is the opposite of the size prior.** `../fig-molecular-weight` shows
   every baseline is *lighter* than both VoxBind arms (median 209–317 Da against 336/354),
   and this metric gets harder as fragments grow. The baselines are being scored on the
   easier end of the size axis and still lose, which makes the gap a floor on the real one.
3. **Local geometry is not where CoDE loses to VoxBind.** CoDE trails by 0.005 Å pooled and
   0.005 standardised, and by 0.001–0.016 Å in every paper bin but the largest, where it is
   0.043 Å *better*. Heavy-atom coordinate precision in a good crystal structure is
   ~0.1–0.2 Å, so every one of those gaps is inside the noise a crystallographer would
   ignore — while over the same molecules PoseCheck strain rises 62 → 82 kcal mol⁻¹ and
   clashes 4 → 5. The deficit those report lives in **global conformation and placement**,
   not in how the model draws a ring. Honest counter-argument: the *relative* gap here
   (+24 % pooled) is not far off strain's (+32 %); it is the absolute magnitude that makes
   it negligible.
4. **Five-atom rigid fragments are a sharp local peak for almost every arm — including the
   crystal ligands.** Seven of eight models put 5 atoms above both 4 and 6, and the
   reference's 5-atom median is 2.5× its 6-atom one (0.066 against 0.027). So this is **not
   a generative-model artifact**: five-membered rings have a soft pucker coordinate that
   MMFF re-optimises, where benzene has nothing to move. The models exaggerate it rather
   than create it — VoxBind's 5-to-6 ratio is 5.6× against the crystal set's 2.5× — and the
   paper's `5–6` bin hides all of it by pooling the two.
5. **The MMFF failure rate is a result in its own right, and it is this metric's caveat.**
   Molecules MMFF cannot parameterise are excluded from every number above: 0.0 % for
   Pocket2Mol, 0.5 % DiffSBDD, 2.9 % DecompDiff, 3.9 % VoxBind, 5.9 % AR, 6.0 % TargetDiff
   and CoDE, and **12.2 % for FuncBind** — an outlier worth its own look. If those failures
   are enriched for bad geometry, which is plausible since parameterisation fails on unusual
   valences and atom types, then every arm's number is optimistic and the worse arms' more
   so.
6. **The largest fragments reverse between the two VoxBind arms.** CoDE beats VoxBind at
   15+ atoms (0.324 against 0.424) and in the paper's 14+ bin (0.2915 against 0.3349),
   where it nearly matches the reference (0.2837) — the same reversal strain and clashes
   show. It is the one size range where the two arms are not within noise of each other.

## Provenance

Evaluation: `voxbind/exps/frozenenc_probes/eval_rigid_fragments.py`, CPU only, no receptor
needed. It writes `rigid_fragment_results.json` into each run root; `--reference` scores the
deposited ligand instead, into `rigid_fragment_reference.json`. This folder only draws what
that wrote.

**`--connected-only` is required, and the builder refuses to mix modes.** `metrics.py` drops
molecules that are not one connected component before scoring anything, so without the flag
this metric would cover a different molecule set than the strain, clash and PoseBusters
figures for the same run. It matters for exactly one arm — TargetDiff emits 511 disconnected
molecules of 7,798 over the 79 pockets (6.6 %), every other arm emits none — and the mode is
stamped into each output, so an arm scored one way and an arm scored the other cannot be
drawn on one axis by accident.

```bash
V=/opt/conda/envs/voxdock/bin/python
E=voxbind/exps
R=$E/frozenenc_probes/eval_rigid_fragments.py
for r in $E/baselines_pose/{ar,pocket2mol,diffsbdd,decompdiff,funcbind} \
         /home1/irteam/base_drug/eval/targetdiff \
         $E/_vanilla_ep923/samples/full_eval_ep923 \
         $E/voxbind_frozenenc_atomblob7_v2p1_sig0.9/samples/full_eval_ep350; do
  $V $R "$r" --connected-only --workers 4
done
$V $R $E/voxbind_frozenenc_atomblob7_v2p1_sig0.9/samples/full_eval_ep350 \
      --reference --connected-only

/opt/conda/envs/voxbind/bin/python notebook/html/260910/fig-consistency/build_rigid_fragment.py
```

Use few workers while `86_pose_eval_baselines.sh` is running: the container is capped at 32
cores whatever `nproc` says.

The five baselines are the staged sample dirs `voxbind/scripts/tools/stage_baseline_samples.py`
writes under `exps/baselines_pose/`; their molecule counts here (7,655 / 7,772 / 7,720 /
6,427 / 7,895) match `../fig-posecheck`'s exactly, which is the cheapest check that both
folders read the same samples. An arm with no results file, or one not scored over all 79
pockets, is left out of the figures and named in the run log rather than drawn as a partial
curve.

Two differences from `../../260827/rigid_fragment_consistency.html`, which reported this
metric on the arms as they stood then:

* **TargetDiff is now on the same 79 pockets as everything else**, and without its
  disconnected molecules. 260827 scored it over its own 100 and 92 pockets. Its median moves
  0.0533 → 0.0531, so nothing turns on it — but the figures here are matched by
  construction rather than by luck.
* Arms are named as in the rest of 260910: *VoxBind* is 260827's "Vanilla σ=0.9", *CoDE* its
  "Holo ED fusion σ=0.9 (atomblob7 v2p1)". VoxBind's and CoDE' numbers reproduce 260827
  fragment for fragment.

`exps/samples_reference_receptor_ed_ep350` (CoDE v2, 92 pockets) has been scored too and its
JSON is on disk; it is not one of the arms this section reports, matching the decision in
`../pose_common.py`.
