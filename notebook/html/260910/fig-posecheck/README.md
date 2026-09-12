# PoseCheck — strain and steric clashes

Everything in this folder measures PoseCheck. Its sibling `../fig-posebusters` measures
PoseBusters dock-mode validity. The two split by metric and share `../pose_common.py`
(arm list, pocket set, loader, house furniture) and `../method_colors.py` (one colour per
method), which is everything a disagreement between them would silently corrupt. **Each
folder carries the code that draws its own figures.**

## The seven builders

| builder | draws | runs here? |
|---|---|---|
| `build_posecheck_per_atom.py` | strain and clashes against ligand SIZE, three local arms | yes |
| `build_strain_per_rotbond.py` | strain against ROTATABLE BONDS, the same three arms | yes |
| `build_strain_rotbond_baselines.py` | the same axis with the **published baselines** added | yes |
| `build_strain_atom_all_methods.py` | the SIZE axis with the **published baselines** added | yes |
| `build_posecheck_all_by_atom_range.py` | all eight methods, binned: ECDF + violin | yes |
| `build_posecheck_baselines_by_atom_range.py` | the five published baselines alone → `_baselines_only/` | no — needs `prj-denovo/baselines` |
| `export_posecheck_json.py` | `posecheck_<Method>.json`, the five baselines per molecule | no — same |

## Size-resolved (`build_posecheck_per_atom.py`)

| file | what |
|---|---|
| `strain_per_atom_mean_{core,all}.*` | strain **mean** against heavy-atom count |
| `strain_per_atom_median_{core,all}.*` | strain **median** against heavy-atom count |
| `clash_per_atom_{mean,median}_{core,all}.*` | clashes, a figure per statistic |
| `strain_clash_ecdf_pair_{core,all}.*` | both distributions, pooled, on one shared y |
| `posecheck_per_atom.{json,csv}` | the plotted curves, per heavy-atom count |
| `posecheck_summary.json` | pooled numbers, `p79` and `all_pockets` |
| `posecheck_per_molecule_<arm>.json` | per-molecule `t`/`n`/`s`/`c`, for merging |

**Two variants of every figure.** `_core` draws VoxBind, CoDE and the crystal
reference — the comparison this section makes, and the same three series the Vina 3-line
figures carry. `_all` adds every baseline the figure has data for (here: TargetDiff). The
data exports always cover every arm regardless of which variant a figure draws.

**Careful with two meanings of "all" in this folder.** In `strain_per_atom_*_all` it is
the arm variant; in `strain_ecdf_all` / `clash_violin_all` it is the *size* bin (all sizes
pooled), from the binned builder below.

**Mean and median are separate figures**, which is what the 3-line figures do
(`vina_dock_3line_mean` / `_median`) and here it is not optional. Strain's mean is not a
location statistic: 6.6 % of molecules fail UFF relaxation and land between 1e4 and 1e13,
and one of those at a thin heavy-atom count moves that count's mean by four decades. Drawn
apart, the contrast is itself the argument for reporting the median — in the mean figure
the arms are tangled between 1e3 and 1e6 with no order, in the median figure they separate
cleanly. The mean figure is clipped to the bulk and the 13 (arm, size) points that run off
the top are **named in the run log**, not hidden.

Headline numbers over the 79 pockets — strain median / clash median:

| arm | heavy atoms | strain median | clash median | molecules |
|---|---|---|---|---|
| TargetDiff | 22.2 | 355.3 | 8.0 | 7,287 |
| VoxBind | 24.0 | 62.2 | 4.0 | 7,888 |
| CoDE | 24.9 | 81.9 | 5.0 | 7,873 |
| *Reference ligand* | *22.8* | *34.3* | *5.0* | *79* |

Strain medians at 15 / 25 / 35 heavy atoms: TargetDiff 141 / 577 / 1425, VoxBind
29 / 65 / 116, CoDE 29 / 90 / 169. So **the two VoxBind arms stay within a factor
of ~2-3 of the crystal ligands across the whole size range while TargetDiff runs 5-12×
above them**, with CoDE drifting slightly higher than vanilla as molecules grow — though
only up to ~34 heavy atoms: past that the two cross and CoDE is the lower of the pair, see
the all-methods size section below. That is
the same ordering PoseBusters gives from a different measurement, which is the main reason
to trust either — and it is *not* the Vina ordering, where CoDE wins. Pose quality and
affinity are separate axes.

One thing only the clash **mean** figure shows: above ~28 heavy atoms the crystal reference
ligands clash more than VoxBind and CoDE do (~10 against 6–8). Crystal poses are not a
ceiling on this metric.

## Torsion-resolved (`build_strain_per_rotbond.py`)

The VoxBind paper's Fig. 13 form: the same strain, the same three arms and the same 79
pockets as the section above, against the number of **rotatable bonds** instead of ligand
size. Nothing else changes — same loader, same statistics, same furniture, same colours —
so a point here and a point there are the same molecules grouped two ways.

| file | what |
|---|---|
| `strain_box_per_rotbond_{core,all}.*` | the **distribution** at each count, as boxes — **the figure to read** |
| `strain_per_rotbond_median_{core,all}.*` | the medians alone, as a line |
| `strain_per_rotbond_mean_{core,all}.*` | the same line, **mean** — kept so that claim can be checked |
| `strain_per_rotbond.{json,csv}` | the curves plus the `n` behind every point |

```bash
python build_strain_per_rotbond.py                # everything (default)
python build_strain_per_rotbond.py --kind box     # only the boxes
python build_strain_per_rotbond.py --kind line    # only the median and mean lines
python build_strain_per_rotbond.py --stat median  # lines: only the median one
```

The exports always carry both statistics regardless of `--stat`.

**The boxes are the primary figure and the lines are kept beside them.** A median line says
where an arm sits; it cannot say whether two arms a factor of 1.5 apart are actually
separated, and on a metric whose distribution spans four decades *inside a single bond
count* that is the question. The boxes answer it — and they show something the lines
cannot: **the arms' inter-quartile ranges overlap almost completely at every count**, so the
median ordering below is a shift of a wide distribution, not a separation of two narrow
ones. The lines stay because they are what a reader compares against the per-atom figure
above, and because the mean only exists there.

Three things decide whether the boxes are readable at all, and all three are in the builder:

* **Whiskers are the 5th and 95th percentiles and fliers are not drawn.** 4–7 % of
  molecules relax to 1e4–1e13, so Tukey whiskers with fliers would put single points nine
  decades above the boxes and squash every box in the figure into a line. The percentile is
  named on the y axis; the tail it leaves out is reported as `strain_gt_1e4`.
* **Strain is floored at 1e-2 before boxing**, the value `strain_clash_ecdf_pair` already
  floors it to. A rigid ligand can relax to ~0, and on a log axis a single 1e-11 at 0
  rotatable bonds pulled the panel down through fifteen decades and flattened every box in
  it. Values are *clipped, not dropped*, so the whisker rests on the floor and no count
  changes.
* **No mean marker.** `showmeans` was tried and cannot work: within one arm and one bond
  count the mean sits at 1e4–1e11 while the box sits near 1e2, so every marker lands far
  above its own box and drags the axis with it. The mean has its own line figure, where
  being unreadable at least reads as the finding it is.

**Why this axis is worth having.** Strain is conformational — the energy a pose carries
because its torsions are not where the force field would put them — so the number of
torsions a molecule *has* is the more direct explanatory variable and heavy-atom count is
its proxy. They are not interchangeable: a fused polycyclic and a long-chain ligand of
equal size have very different torsional freedom. Here the curves come out visibly
**smoother** than the per-atom ones, which is the point — strain tracks torsion count more
tightly than it tracks size.

Strain median at 0 / 4 / 8 / 12 rotatable bonds:

| arm | 0 | 4 | 8 | 12 | molecules scored | strain > 1e4 |
|---|---|---|---|---|---|---|
| TargetDiff | 77.0 | 384.5 | 800.9 | 1983.5 | 7,232 | 6.28 % |
| VoxBind | 11.6 | 57.0 | 154.3 | 301.5 | 7,854 | 4.34 % |
| CoDE | 12.9 | 81.1 | 270.8 | 304.5 | 7,826 | 6.56 % |
| *Reference ligand* | *12.9* | *24.1* | *56.2* | *—* | *79* | *0.00 %* |

"molecules scored" is `n_scored` in the JSON — molecules whose UFF relaxation converged, so
it is the sibling section's `n_strain` and not its `n_molecules`. The two exports agree on
it exactly (7,232 / 7,854 / 7,826 / 79), which is the cheapest check that both builders are
reading the same molecules.

Same ordering as the size-resolved view, and the same reading. Over 0–9 bonds, the range
the crystal ligands cover, **VoxBind runs 0.9–3.6× the reference and CoDE 1.0–4.8×,
while TargetDiff runs 6.0–19.4×**. Both VoxBind arms sit *at* crystal-ligand strain at 0
rotatable bonds (0.9× and 1.0×), where there is no torsional freedom left to get wrong, and
separate from it as torsions are added — the whole gap is in the torsions, which is the
argument for this axis over the size one.

Between the two VoxBind arms, CoDE is 23–76 % above vanilla at every count from 1 to 8
bonds, then within ±20 % of it from 9 to 13 with no consistent sign. **Do not read the 14-bond point**: it rests on 29
molecules for Ours against 56 for vanilla, the thinnest pair on the axis, and it is the
only place the curves diverge sharply.

**Rotatable bonds are counted from the SMILES `metrics.json` already records**, with
RDKit's default (strict) `CalcNumRotatableBonds` — amides, terminal bonds and ring bonds
excluded. It is a topological descriptor, so reading it off the recorded SMILES rather than
the pose gives the same integer with no risk of a sample-to-SDF index slip. No SMILES in
the three local arms failed to parse.

**The crystal reference window is ±1 here, not `pose_common`'s ±4.** One ligand per pocket
is too thin for a per-count curve — that is why it is windowed at all — but rotatable-bond
counts run 0–14 where ligand sizes run 5–45, so ±4 would span two thirds of the axis and
flatten the reference into a near-constant line. ±1 pools 12–29 ligands per point, and the
line **stops at 9** where the window falls under `MIN_REF` rather than being extended into
an invented value. Same narrowing, for the same reason, as `../fig-consistency`. The
exports carry both counts side by side — `n` is the ligands at that exact bond count,
`n_window` the pool the value was computed from — because for the reference they differ by
a factor of 2–4 and a single `n` column would misstate one row or the other.

**The mean panel is not clipped, where the per-atom one is.** That figure clips to 1e6
because its curves have a bulk below it and only a few thin heavy-atom counts spike out.
This axis has no such bulk: 15 bond counts pool 500–1,100 molecules each, so nearly every
point catches one of the conformers that relax to 1e8–1e13, and the same clip left the
curve as disconnected fragments with most of it off-panel. Drawn whole it spans nine
decades, sits 6–9 decades above the crystal ligands and carries no ordering at all — which
is the honest picture of what a mean does to this metric, and is itself the argument for
the median. The tail driving it is in the table above and in the JSON as `strain_gt_1e4`.

## Torsion-resolved, all methods (`build_strain_rotbond_baselines.py`)

The section above with the published baselines added: **8 methods + the crystal ligands**,
same 79 pockets, same axis.

| file | what |
|---|---|
| `strain_box_per_rotbond_all_methods.*` | the **distribution** at each bond count, all methods dodged on one axis — **the figure to read** |
| `rotbond_distribution_all_methods.*` | where each method puts its ligands on that axis, one panel per method, 3×3 |
| `strain_rotbond_all_methods_{median,mean}.*` | the medians (and means) alone, as lines, per exact bond count |
| `strain_rotbond_all_methods.{json,csv}` | the numbers behind all of them |

**One dodged axis, and the shares are a separate figure.** Small multiples — one panel per
method, the paper's Fig. 12/13 form — were built first and dropped: the comparison that
matters is between methods *at* a bond count, and that is the one a grid makes hardest.
Eight boxes dodged inside each count put it back. The two figures answer different
questions (the boxes compare methods at a count; the shares compare the counts a method
produces) and are read at different times, so they no longer share a canvas — nine share
panels do not fit under a box panel at any useful size.

**The axis stops at 12 rotatable bonds** (`X_MAX`). Past it every individual count is thin,
and stretching to the last molecule anyone made — 25, one VoxBind ligand — spent two thirds
of the width on a handful of boxes. The cut is **not** silent: every share panel prints its
own excluded percentage (`>12 bonds: N%` — 0.0 % for Pocket2Mol up to 5.0 % for VoxBind),
and raising `X_MAX` brings the tail back, splitting the axis over two rows above 14 counts.

**Each series is drawn over its own range, not the narrowest one.** Boxes appear wherever a
method has ≥10 molecules at a count (≥5 for the crystal ligands, whose panel is the one
place a box over six ligands is worth more than nothing); the median *lines* use the house
`MIN_N = 25`. So Pocket2Mol's line stops at 9 bonds while the other seven run to 12, and
the grey reference line stops at 9 as well — 79 crystal ligands cannot fill a window past
it. Being cut off is the honest thing for it to do.

**The y range of the box figure is pinned to 1e0–1e5** and the tails are allowed to leave
the top. FuncBind's 95th percentile at 6 rotatable bonds reaches ~1e11; autoscaling to it
stretched the panel over fourteen decades and pressed every box into the bottom fifth. How
much tail each method carries is reported as `strain_gt_1e4`, not drawn. The cost is that
the 5th-percentile whiskers of the low-strain methods at 0–2 bonds now rest on the floor.

**The share figure is a finding in its own right**: Pocket2Mol puts a third of its output at
0–1 rotatable bonds, while the VoxBind arms peak at 3–4 and carry a tail past 12 that only
DecompDiff matches.

Strain median by bond bin, over the 79 pockets:

| method | mols | atoms | 0 | 1–2 | 3–4 | 5–6 | 7–9 | 10+ | strain > 1e4 |
|---|---|---|---|---|---|---|---|---|---|
| AR | 7,654 | 17.2 | 130.3 | 238.4 | 426.7 | 628.7 | 921.9 | 1121.1 | 5.83 % |
| TargetDiff | 7,232 | 22.2 | 77.0 | 172.0 | 313.7 | 546.6 | 819.5 | 1496.8 | 6.28 % |
| DiffSBDD | 7,720 | 20.2 | 57.8 | 131.7 | 255.0 | 388.5 | 652.0 | 1174.7 | 1.24 % |
| FuncBind | 7,895 | 19.0 | 41.3 | 83.5 | 201.9 | 1623.2 | 1882.0 | 1427.2 | **12.36 %** |
| DecompDiff | 6,299 | 20.9 | 34.9 | 50.7 | 96.7 | 231.5 | 326.1 | 699.1 | 1.25 % |
| Pocket2Mol | 7,772 | 18.5 | 24.9 | 41.6 | 113.7 | 133.7 | 156.6 | 146.9 | **0.04 %** |
| **CoDE** | 7,826 | 24.9 | 12.9 | 33.8 | 70.1 | 109.3 | 205.1 | 428.7 | 6.56 % |
| **VoxBind** | 7,854 | 24.0 | **11.6** | **25.8** | **48.7** | **87.0** | **135.8** | 414.7 | 4.34 % |
| *Reference ligand* | *79* | *22.8* | *—* | *10.2* | *33.4* | *26.5* | *56.2* | *—* | *0.00 %* |

**Vanilla VoxBind is the lowest-strain generator in every bin from 0 to 7–9**, and both
VoxBind arms do it while generating the *largest* molecules in the table (24.0 / 24.9 heavy
atoms against 17.2–20.9) — the opposite of what the size confound would produce. The
ordering is **not** stable across the axis, though, and the two places it breaks are the
interesting ones:

* **Pocket2Mol overtakes at high torsion counts.** It is third in the low bins, passes
  CoDE at 7–9 (157 vs 205) and is the **best method of all at 10+** (147, against 415 /
  429 for the VoxBind arms). Per exact count it crosses below Ours at 8 bonds and below
  both arms at 9. Its tail is also in a different class: only **0.04 %** of its molecules
  relax past 1e4, against 1.2–1.5 % for DiffSBDD/DecompDiff, 4.3–6.6 % for the VoxBind
  arms and TargetDiff, and 12.4 % for FuncBind. It makes small molecules (18.5 atoms) and
  almost never produces a catastrophic conformer. Read this together with its Vina numbers
  before calling it a win.
* **DecompDiff beats Pocket2Mol in exactly one bin** (3–4: 97 vs 114) and nowhere else.
* **FuncBind breaks at five rotatable bonds.** Through 3–4 it is the third-lowest method in
  the table (41 / 84 / 202, under everything but the two VoxBind arms); at 5–6 it jumps to
  1,623 and stays there, making it the **worst** method of all at 5–6 and 7–9 — 2.6× and
  2.0× AR. Its tail says the same thing from the other end: **12.4 %** of its molecules
  relax past 1e4, twice the next worst. Whatever fails, fails on flexible ligands and fails
  hard, so read its pooled numbers as two populations rather than one.

Everything else holds throughout: DiffSBDD < TargetDiff < AR in every bin up to 7–9. **AR
is the worst of the published baselines over most of the axis, not TargetDiff** (FuncBind
aside, which is worst wherever it is broken) — and it is worst at *zero*
rotatable bonds (130.3 against TargetDiff's 77.0), where there is no torsional freedom at
all, so that is bond and angle geometry rather than conformation. Only at 10+ does
TargetDiff take last place (1497 vs AR's 1121).

**The `_core`/`_all` split does not apply to this family.** Every figure it writes draws
every method there is data for, so the filenames carry no variant suffix.

### Where the rotatable bonds come from, and why the join is safe

The baselines' strain has been in `posecheck_<Method>.json` all along, but those exports
carry no SMILES — and without one there is no rotatable-bond count. The molecules are now
in the results bundle (`results/task2-drugdesign/<M>/samples/meta/`, TargetDiff meta
format), and `export_posecheck_json.py` built those exports **from the same meta**, walking
each pocket in order and skipping `mol is None`. So `load_meta()` here reproduces that
loader exactly — base + `_part2` concatenated per pocket, and **not** `_gap.pt`, which
ships in the bundle but which that loader never read.

That is an argument, not evidence, so the builder **checks** it: for every pocket the
heavy-atom sequence recomputed from the meta must equal the export's `n` sequence position
by position, and a mismatch aborts rather than guesses. **The check runs over all 100
pockets, not only the 79 that are drawn** — the meta is already in memory and the export
already carries them, so the other 21 cost nothing and would otherwise be evidence nobody
looked at. It matches **100/100 for all five methods** (`pockets_join_checked` in the
JSON), and the 79 kept pockets hold exactly 7,655 / 7,772 / 7,720 / 6,427 / 7,895
molecules — the totals this README already records from
`baselines/_eval/summary_density79.json`.

### FuncBind is in, and how it got there

Until 2026-09-10 it was not: its strain export existed (`posecheck_FuncBind.json`, 9,992
molecules) but **no meta for it had reached this box**, and without one there is no torsion
count. Its meta arrived in the results bundle that day and joins cleanly — 100/100 pockets,
9,992 molecules, the export's own total.

**Do not reach for the shard SDFs if the meta ever goes missing again.** Those under
`funcbind/artifacts/reproduction/crossdocked/paper_run` were tried as a substitute and
**fail the same check**: 1 pocket of 100 matches, pocket 0 holds 99 molecules against the
export's 100, and the atom counts disagree from the first record. They are a different
sampling run from the one PoseCheck scored — the pocket mapping is not the problem,
`pocket_ligand_filename` agrees with the shard target dirs. Joining them would have
attached the wrong molecule's torsion count to every strain value, silently.

### Mixing the two scoring runs is safe for strain, and only for strain

The baselines were scored against the whole `*_rec.pdb` receptor; the local arms are read
from the pocket10 crop, the same source the section above uses. Strain is a property of the
ligand's own conformer — UFF relaxation under a position constraint — so receptor scope
cannot enter it, and measuring the same molecules both ways confirms it does not: median
|relative difference| **0.6–1.0 %** (the run-to-run noise `num_confs=50` already carries),
and pooled medians move under 1 % — 60.1 vs 62.6, 80.5 vs 82.9, 346.2 vs 350.4. **Do not
extend this to clashes or interactions**: those are receptor-dependent, and a crop cannot
see an atom it does not contain.

## Size-resolved, all methods (`build_strain_atom_all_methods.py`)

The size axis of `build_posecheck_per_atom.py` with the published baselines added:
**8 methods + the crystal ligands**, same 79 pockets, same metric. The torsion sibling
above and this one are the same molecules grouped two ways.

| file | what |
|---|---|
| `strain_per_atom_all_methods_median.*` | strain **median** against heavy-atom count, every method on one axis — **the figure to read** |
| `strain_per_atom_all_methods_mean.*` | the same, **mean**, clipped to 1e6 with the off-panel points named in the run log |
| `strain_per_atom_all_methods.{json,csv}` | the plotted curves plus the `n` behind every point |

**No bundle join is needed on this axis.** The rotatable-bond figure had to go back to
`results/task2-drugdesign` for SMILES; heavy atoms and strain are *both* already in
`posecheck_<Method>.json`, so the five baselines are read straight out of their exports
(p79 pockets only, by the export's own pocket index). The row counts land on the same
7,655 / 7,772 / 7,720 / 6,427 / 7,895 the bundle join produces, which is the cheapest
available check that the two builders are looking at the same molecules.

**The key sits in a strip under the panel, not inside it.** Nine series leave no free
corner: across the top the key covered exactly the high-strain curves the figure is about,
and in a corner it sat on the crystal-ligand line.

### DecompDiff puts exactly one molecule size in each pocket, and it is the crystal ligand's

Checked on every run and printed: **79/79** pockets hold a single heavy-atom count, and
**79/79** of those equal the reference ligand's. Its reference prior fixes the atom budget
per pocket, so on this axis DecompDiff is not a distribution but a comb over the 79
reference sizes, with exact zeros at 7, 24, 30, 34, 36, 39–41 and 43–44 atoms. Two things
follow, both live in the builder:

* **Every model curve pools a ±1-atom window** — the same rule for all eight methods — so
  DecompDiff's line does not break ten times inside the axis on sizes that were never going
  to exist. At 100–400 molecules a count it moves a median almost nowhere. The one hole it
  cannot fill (40 atoms, where no crystal ligand sits within ±1) is left as a break rather
  than widened away. **The window is clipped to the axis**, which matters at exactly one
  place: DecompDiff has nothing at 43 or 44 atoms and 125 molecules at 45+, so an unclipped
  window would have put a point at 44 computed entirely from molecules the same file
  reports as *beyond* the axis. Clipped, its line ends at 43 (and AR's at 43 too).
* **DecompDiff is size-matched to the reference by construction**, which is exactly the
  confound this figure controls for. Its pooled numbers elsewhere carry that advantage;
  here it is neutralised, so this is the fair place to read it — and it reads badly:
  232 kcal/mol at 20 heavy atoms against Pocket2Mol's 80 and VoxBind's 47.

The axis runs **5–44 heavy atoms**, each line drawn only where its own method has ≥25
molecules in the window — so AR's and DecompDiff's stop at 43. The exports carry `n` (that
exact count) and `n_window` (the pooled sample the value came from) as separate columns;
for DecompDiff's comb they differ by everything. The share past 44 is exported per method (0.1 % for AR up to 3.4 %
for CoDE, and 3.8 % of the crystal ligands). The reference line uses the house ±4-atom
window and ends at 36, where 79 ligands stop filling it.

Strain median at 10 / 15 / 20 / 25 / 30 / 35 / 40 heavy atoms:

| method | mols | mean atoms | 10 | 15 | 20 | 25 | 30 | 35 | 40 |
|---|---|---|---|---|---|---|---|---|---|
| AR | 7,654 | 17.2 | 125.2 | 267.3 | 494.4 | 874.3 | 1142.3 | 1859.9 | 2194.7 |
| TargetDiff | 7,232 | 22.2 | 45.9 | 150.3 | 312.5 | 500.6 | 700.9 | 1318.2 | 1442.3 |
| DiffSBDD | 7,720 | 20.2 | 68.2 | 172.9 | 314.9 | 514.0 | 728.4 | 1065.0 | 1410.1 |
| FuncBind | 7,895 | 19.0 | 38.8 | 121.5 | 217.5 | 488.3 | 687.3 | 1128.7 | 2082.8 |
| DecompDiff | 6,299 | 20.9 | 22.1 | 81.9 | 231.8 | 310.3 | 393.2 | 872.2 | — |
| Pocket2Mol | 7,772 | 18.5 | 15.3 | 32.3 | 80.1 | 102.6 | 195.1 | 183.3 | 332.1 |
| **CoDE** | 7,826 | 24.9 | 17.0 | 30.1 | 51.1 | 89.9 | 111.8 | **143.6** | **193.7** |
| **VoxBind** | 7,854 | 24.0 | **13.0** | **30.0** | **47.4** | **63.0** | **92.9** | 150.1 | 276.0 |
| *Reference ligand* | *79* | *22.8* | *9.1* | *12.9* | *29.4* | *42.2* | *54.0* | *55.9* | *—* |

**The size confound is what this figure removes, and the ordering survives it.** The two
VoxBind arms generate the largest molecules in the table (24.0 / 24.9 mean heavy atoms
against 17.2–22.2) and still sit lowest *at matched size* — over 7–36 atoms, where the
crystal ligands are drawn, VoxBind runs 1.3–3.0× and CoDE 1.5–2.7× above them, against
7–39× for AR and 2–24× for TargetDiff. Per-molecule aggregates would have flattered the
small-molecule methods; this axis does not.

**The two VoxBind arms cross at ~35 heavy atoms.** Up to 34 vanilla is lower (the drift the
size-resolved section above reports); from 35 on CoDE is, and the gap widens with size —
194 vs 276 at 40 atoms, 246 vs 602 at 44. It is not an artefact of the ±1 pooling: in the
unwindowed per-count series (`posecheck_per_atom.json`) CoDE is the lower of the two at
**every** drawn size from 36 up, 35 being the one count that goes the other way. The
largest ligands are where the two arms differ most, and there CoDE is ahead.

**Pocket2Mol's torsion-axis advantage does not appear here.** On the bond axis it is the
best method of all at 10+ rotatable bonds; on the size axis it sits above vanilla VoxBind at
every size from 6 to 40 — up to 2.1× at 30 atoms — and crosses below only at 41–44, where a
few dozen of its molecules are left. Against CoDE it is lower only below 13 atoms and again
past 41. The two axes ask different questions of the same molecules — Pocket2Mol's ligands
are both smaller *and* more rigid than the VoxBind arms' (a third of them at 0–1 rotatable
bonds), so the few it makes with 10+ rotatable bonds are a different population from
theirs, and a win on that axis is not a win on this one.

**FuncBind's break shows on this axis too, and later.** It is the third-lowest method up to
~20 heavy atoms, passes DiffSBDD for good at 35, passes AR at 41, and is the highest of the
eight from 41 on (3,061 at 42, 3,244 at 44) — the size axis puts its collapse a little
further out than the torsion axis does, but in the same direction. Its mean curve is over
the 1e6 clip at **22 of 40** counts, against at most 15 for any other method.

## All eight methods, binned (`build_posecheck_all_by_atom_range.py`)

`strain_ecdf_<bin>.*` and `clash_violin_<bin>.*` over `le15`, `16_20`, `21_25`, `26_30`,
`gt30` and `all` (all sizes), plus `posecheck_all_by_atom_range.{json,csv}`. Colours come
from `../method_colors.py`, so an arm reads the same here as in every other figure.

This family scores the local arms against the **whole receptor**
(`frozenenc_probes/posecheck_full/`), while `build_posecheck_per_atom.py` uses the pocket10
**crop** that the target `metrics.json` files carry. The two are not interchangeable — a
crop cannot show a clash with an atom it does not contain — so do not read a number from
one against a number from the other.

**And the crop, not the whole receptor, is what the literature reports** (checked against
both papers, 2026-09-10). PoseCheck's own pipeline is crop-scored: `get_ids_to_pockets`
reads element `[0]` of each CrossDocked split entry, and that element is
`..._pocket10.pdb` (element `[1]` is the ligand SDF). The PoseCheck paper says the same in
words — "the model is given a reduced PDB file containing only the atoms for a single
pocket" — without naming the radius, which the code settles. Two numeric anchors agree:

| | VoxBind paper, Fig. 7 | our crop | our whole receptor |
|---|---|---|---|
| TargetDiff clash mean | 10.8 | **10.72** | 11.01 |
| VoxBind σ=0.9 clash mean | 5.1 | **5.38** | 6.23 |

So `build_posecheck_per_atom.py`'s numbers sit on the published axis and this family's do
not. Two things follow. The "~10.8" that older scripts in this repo cite is the **VoxBind**
paper's figure, not PoseCheck's — the PoseCheck paper reports 9.08 for TargetDiff, from its
own samples, and that gap is a difference of sample set, not of scope. And the whole-
receptor numbers err upward, not downward: scoring against more protein finds more clashes,
so this family is not a conservative version of the crop, it is a different measurement.

## Two things to carry into the writeup

1. **Strain here is the posecheck 1.3.1 definition, not the VoxBind paper's.** Numbers are
   ~10× smaller than Fig. 6 of arXiv 2405.03961 and must never be placed beside it. Now
   confirmed from both papers: VoxBind's Fig. 6 reference median is 102.5, the identical
   value the PoseCheck paper gives for its CrossDocked baseline, so VoxBind carried the old
   definition. Ours is 34.3. Report
   the median: the distribution is heavy-tailed enough that the mean reports the UFF
   failure rate instead (vanilla: mean 2.26e8, median 62).
2. **Strain is not reproducible run-to-run**, even with `randomSeed=0` forced. The same
   crystal ligand scored under two run roots on adjacent days differs by median 0.09,
   p90 ~0.9, max 6.0 kcal mol⁻¹, with only ~25-30 % of pockets matching exactly. Pooled
   medians over ~7.8k molecules are stable; per-molecule strain comparisons are not exact.

---

# The five published baselines (`_baselines_only/`)

Sampled and scored on another box; only their exports made it across. Strain and clashes
both rise with molecule size and the methods draw different size mixes, so everything is
stratified by heavy-atom count. Figures cover the **79 electron-density pockets** (the
subset tables 1b/2b of `../../260903/baseline.html` use); the JSON exports carry all 100 so
either subset can be drawn without another round trip.

| file | what |
|---|---|
| `_baselines_only/strain_ecdf_<bin>.{png,svg}` | cumulative probability of UFF strain |
| `_baselines_only/clash_violin_<bin>.{png,svg}` | steric-clash distribution |
| `posecheck_baselines_by_atom_range.json` | the per-bin numbers behind them |
| `posecheck_<Method>.json` | **per-molecule** results, for merging elsewhere |

## Per-molecule export format

One file per method: `posecheck_AR.json`, `posecheck_Pocket2Mol.json`,
`posecheck_DiffSBDD.json`, `posecheck_DecompDiff.json`, `posecheck_FuncBind.json`.

```jsonc
{
  "method": "AR",
  "n_pockets": 100,
  "n_molecules": 9729,
  "n_strain": 9728,                 // molecules whose UFF relaxation converged
  "density79_pockets": [0, 2, ...], // the 79 subset, as pocket indices
  "pocket_ligand_filename": {"0": "BSD_ASPTE_1_130_0/2z3h_..._docked_3.sdf", ...},
  "molecules": [{"p": 0, "n": 17, "s": 123.456, "c": 4.0}, ...]
}
```

`p` pocket index 0–99, in `split_by_name.pt['test']` order — the same order VoxBind's
`target_<NN>` directories use (checked by canonical SMILES, 100/100).
`n` heavy-atom count · `s` UFF strain in kcal/mol · `c` steric clashes.

`s` is `null` where the relaxation failed. That is **not** a zero and must not be counted
as one — only DecompDiff has a meaningful number of these (177 of 8,207; 128 inside the
79 subset). `clashes` is never null in this data.

## Merging with TargetDiff / vanilla VoxBind / CoDE

Those three read from `/home1/irteam/...`, which is not on the box that produced the
five-baseline exports, so `_baselines_only/` shows the five plus the crystal reference.
On this box they merge: `build_posecheck_all_by_atom_range.py` draws all eight.
`build_posecheck_baselines_by_atom_range.py` already declares their roots and merges
them onto the same axes wherever the directories exist — running it there yields one
figure with all eight. To feed these five in from a `metrics.json`-shaped pipeline
instead:

```python
import json, numpy as np
EDGES = [0, 16, 21, 26, 31, 10**6]

def load(path, only79=True):
    """(bin index) -> {'strain': [...], 'clash': [...]}, same shape as pull()."""
    d = json.load(open(path))
    keep = set(d["density79_pockets"]) if only79 else None
    out = {b: {"strain": [], "clash": []} for b in range(len(EDGES) - 1)}
    for m in d["molecules"]:
        if keep is not None and m["p"] not in keep:
            continue
        b = min(np.searchsorted(EDGES, m["n"], side="right") - 1, len(EDGES) - 2)
        if m["s"] is not None:
            out[b]["strain"].append(m["s"])
        if m["c"] is not None:
            out[b]["clash"].append(m["c"])
    return out
```

Round-tripping the exports through that loader reproduces every number in
`posecheck_baselines_by_atom_range.json` exactly (0 mismatches over 5 methods × 5 bins ×
3 statistics).

## Provenance

Sampling and scoring: `prj-denovo/baselines` — 100 pockets × 100 ligands from each
model's official checkpoint, PoseCheck on the pose as generated, receptors protonated
with pdb2pqr against the full `*_rec.pdb`.

PoseCheck was written per (pocket, chunk-of-20) without the atom count, so counts are
read back from the meta the chunks were built from. Both the figure script and the
exporter re-check that alignment on every chunk — matching length **and**
`ligand_filename` — and abort rather than guess. Molecule totals agree with
`baselines/_eval/summary_density79.json` exactly: 7,655 / 7,772 / 7,720 / 6,427 / 7,895.

Rebuild with:

```bash
python build_posecheck_baselines_by_atom_range.py   # figures + per-bin JSON
python export_posecheck_json.py                     # per-molecule JSON
```
