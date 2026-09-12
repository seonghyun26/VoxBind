# Interaction fingerprints — what the pose does against the pocket

The fourth pose-quality measure of the 260910 section, and the VoxBind paper's Fig. 14
form — including its letter-value panels, see [Two panel forms](#two-panel-forms). Its
three siblings measure whether a pose is *physically plausible*; this one measures what it
actually *touches*. Same 79 pockets, same arms, same colours — this folder shares
`../pose_common.py` and `../method_colors.py` with them.

| folder | measures |
|---|---|
| `../fig-posecheck` | strain and steric clashes |
| `../fig-posebusters` | dock-mode validity, per check |
| `../fig-consistency` | rigid-fragment geometry |
| **here** | the protein–ligand interactions the pose makes |

## The metric

An interaction fingerprint records, per pose, which contacts it makes with the receptor.
PoseCheck runs ProLIF over the protonated pocket and the pose **as generated** — see
Harris et al., 2023 (the PoseCheck paper) for the definitions. Four types appear in this
data:

| type | what |
|---|---|
| `VdWContact` | van der Waals contact — much the most common, and mostly a size proxy |
| `Hydrophobic` | apolar contact |
| `HBAcceptor` | the ligand **accepts** a hydrogen bond from the protein |
| `HBDonor` | the ligand **donates** one to the protein |

**More is not better, and this is not a leaderboard.** A bigger ligand makes more of every
type, so an arm that generates larger molecules scores higher on all four without binding
better. The crystal ligands are the reference precisely because they are the distribution a
real binder draws from: the reading is which arm sits **closest to them**, not which sits
highest. The mean heavy-atom count is in every table below for exactly this reason, and the
per-molecule counts are exported so a size-matched comparison can be made.

## Files

| file | what |
|---|---|
| `interaction_contacts_{core,all}.*` | van der Waals and hydrophobic contacts, as violins |
| `interaction_hbonds_{core,all}.*` | H-bonds accepted and donated, as letter-value blocks |
| `interactions.{json,csv}` | mean / median / quartiles / share-nonzero, every method |
| `interactions_<Baseline>.json` | per-molecule fingerprints, recomputed (see below) |
| `interaction_pair_{contacts,hbonds}_core.*` | the same molecules generated vs. redocked |
| `interactions_pair.csv` | both series' means and the change between them |
| `interactions_<Arm>-{gen,docked}.json` | the paired generated/redocked series, core arms |
| `poses/<Arm>-{gen,docked}/*.sdf` | the poses those were scored from — Dropbox, not git |
| `poses/manifest.json` | which rows survived docking, per arm and pocket, and why not |

**Two figures of two panels, not one of four**, since 2026-09-10. The split is the one the
forms already make — contacts are violins, H-bonds are blocks — and it is a real one: under
one caption, a reader comparing arms on H-bonds had two violins in the same eyeful arguing
for a different reading of the same data at a different scale. They are separate
measurements, so they are separate figures.

Two earlier renames, both on the outputs only. The figures were `interaction_violin_*`
until 2026-09-09 — half the panels are no longer violins, so the stem no longer says so.
And `all` used to mean *our arms plus TargetDiff*, with a third figure called
`interaction_all_methods` carrying the baselines; two figures differing only in how many
baselines they drew was one too many, so `all` became what it says.

```bash
/opt/conda/envs/voxbind/bin/python build_interactions.py

# The baselines, once. ~9 min; the figure appears only after this has run.
/opt/conda/envs/voxbind/bin/python stage_baseline_poses.py <scratch>/baseline_poses
PATH=/opt/conda/envs/moleval/bin:$PATH /opt/conda/envs/moleval/bin/python \
    compute_baseline_interactions.py <scratch>/baseline_poses
```

`_core` is VoxBind, ours and the crystal reference — the comparison this section is
making; `_all` is every method that carries a fingerprint over all 79 pockets. Both the
part and the variant are in the FILENAME, never only in the content: two figures that
differ in what they draw must not be able to sit in a folder under one name. The exports
always cover every arm regardless of which variant a figure draws.

**No panel titles, and the type names the y axis instead.** That is the 260910 house style,
which this folder had to break while it was one 2x2 figure: all four panels plotted the same
quantity in the same unit, so the y axis could not carry the type as well. Two panels to a
figure, each y axis is free again.

**The key is a boxed legend beneath the panels, and there are no category tick labels.**
With the key there, printing the same eight names under each of two panels as well is the
same information three times — which, rotated to fit, cost a third of the figure's height.
The box is the house one `pose_common.legend()` draws: opaque white, square corners, a rule
in the axis pen. The swatch is drawn the way that figure's own marks are — a violin body
for the contacts, a block outlined in the axis pen for the letter values — because a legend
that does not look like the thing it names is a second key to learn.

**Both figures carry the same saturation.** The violin bodies have always been drawn at
`BODY_ALPHA` = 0.55; the blocks were solid, which made the H-bond figure read as a much
louder chart than its companion for no reason but its geometry. They now take the same
alpha, on the FACE only as a fourth channel — a patch-level alpha would take the outline
down with it, and the outline is what makes a block a block. The depth ramp was tightened
to match (`BOXEN_LIGHT` 0.66 → 0.46), since the alpha already lightens every tier and a
wide ramp on top of it washed the pale end out.

**The figures label our arm `\textsc{CoDE}` and the reference `Reference`.** Those are
display names, applied by `DISPLAY` in the builder at draw time; `pose_common.REF_LABEL`
and `ARMS` keep the canonical spellings the whole 260910 section shares, and renaming them
there would move every other figure in it. Our arm is written in its LaTeX form verbatim
because these figures are placed in the paper and typeset there — **it renders literally in
the PNG and the PDF**, and the SVG, which keeps live text (`svg.fonttype: none`), is the
copy that gets typeset. `method_colors.ALIASES` already resolves that spelling, along with
`Ours`, `CoDE`, `VoxBind + Ours` and `Ours v1`, so neither the colour nor the drug-design
row order depends on which one a builder passes.

## Two panel forms

The two **contact** panels are violins. The two **H-bond** panels are **letter-value
plots** — VoxBind's Fig. 14(a),(b), whose SVG measures as five nested boxes per column at
exactly halving widths (21.39, 10.69, 5.35, 2.67, 1.34 pt), each box one quantile depth
further out than the one inside it:

| depth | box width | spans |
|---|---|---|
| 1 | 0.78 | the quartiles, p25–p75 |
| 2 | 0.39 | the eighths, p12.5–p87.5 |
| 3 | 0.195 | p6.25–p93.75 |
| 4 | 0.098 | p3.1–p96.9 |
| 5 | 0.049 | p1.6–p98.4 |

**The width carries the depth, not the colour** — so each column stays one method's own
colour, tinted toward white as it goes out, and the eight arms remain a glance apart. Every
lower bound is 0, at every depth and for every arm but one, because at least a quarter of
every arm's molecules make none of that H-bond type. So the boxes sit on the axis and the
column reads as a stack of blocks narrowing upward. That shape *is* the distribution: a
pile on zero with a thin tail. (The exception: the crystal ligands' quartile box on
accepted H-bonds starts at **1**, not 0 — only 25.3 % of them accept none, so its p25 lands
on the first molecule that does. It is the one column here whose widest block does not
touch the axis.)

**Every cell is its own block, and the number sits in the middle of it.** Fig. 14 places
its y ticks the same way — its SVG puts the stubs at +0.5, +1.5, +2.5 … of a count off the
baseline, never on the integers. Three things make that work here:

- **The cells are numbered from 0**, because 0 is a real count on these panels and the
  commonest one. Cell *k* is the band [k, k+1] and its number sits at k+0.5, so the dotted
  rules — which stay on the integers, where the block edges are — **are the counts**, and a
  number names its own cell's *lower* edge. The topmost number in a column is therefore one
  less than the count that column reaches.
- **Every band is cut again at each count it crosses**, so a depth spanning three counts is
  three blocks and not one tall rectangle. Uncut, two arms with the same top edge looked
  different depending on where their quantiles happened to fall, and the column stopped
  being something you could count.
- **Every letter value is an observed count**, `method="nearest"` rather than numpy's
  default interpolation. No pose accepts 1.5 hydrogen bonds, but the default put the crystal
  ligands' quartiles at 0.5 and 2.5 purely because there are 79 of them — which cut their
  blocks half a count off the rules, landing a block edge exactly where a cell's number
  sits. Snapped, every edge falls on a count and every block is a whole cell. **The `q25` /
  `q75` in the exports are *not* snapped** — those stay the conventional interpolated
  quantiles, being a statistic rather than something drawn.

Labels stop at the highest count anyone reached rather than at the panel top, so the
caption's head-room is not numbered and the captioned and uncaptioned variants of a panel
carry the same axis. Every block is outlined in the axis pen, which is what makes them read
as blocks rather than as a colour field.

**The boxes are cut into non-overlapping bands, which Fig. 14's are not.** boxenplot draws
each depth as a full rectangle from its lower to its upper bound, so the narrow pale ones
are painted over the middle of the wide dark ones; a horizontal slice through a column then
runs dark at the flanks to pale at the centre. That gradient encodes nothing — the depth is
already in the width — and it costs the blocks their edges. Here each depth contributes only
the step it *adds* above and below the one inside it, so **one height of the column is one
width and one shade**. A band of zero height is dropped rather than drawn, which is why a
width can be skipped: where two letter values are equal, no height of the column belongs to
the shallower of them.

**Why not violins there.** These counts are small integers — 0–10 accepted, 0–6 donated. A
KDE has to smooth them into a continuum, which draws a body *between* counts no molecule
could have made and a tail below zero, and the arms then differ partly in the shape of an
artefact. Nested boxes are quantiles of the data and nothing else: no bandwidth, no
interpolation, and the zero-heavy left edge stays an edge. The contact types run to 30 and
12 per pose and do have a body a KDE can describe, so they keep theirs.

Two consequences worth knowing when reading a number off a panel. **The H-bond panels' top
is the p98.4 letter value, not the largest count anyone made** — the violin panels' top
still is, so the two forms do not share a y convention. And the letter-value panels keep
the violins' **white median dot** rather than boxenplot's median line, so the mark for "here
is the middle" does not change between panels of one figure; on the donor panel that dot
sits *on the axis* for six of the eight arms, which is the finding: only the crystal
ligands and TargetDiff donate a hydrogen bond in more than half their poses.

## Generated poses, and the redocked pair

Fig. 14 draws two series per method — the pose **as generated** (green) and the same
molecule **redocked** (orange). **The four figures above are the first one only.** The
second had to be produced from scratch, because nothing in the pipeline keeps a docked
pose. Verified down the chain rather than assumed:

- `notebook/webapp/metrics.py` reads the sample SDF into `valid_mols` and hands
  `valid_mols[i]` straight to `run_pose_eval`, which is what writes
  `metrics.json`'s `posecheck.interactions`.
- The docking path in the same function calls `run_vina_docking(..., deepcopy(mol), ...)`
  — Vina works on a **deep copy**, so it cannot write a redocked conformer back into the
  molecule PoseCheck later sees, even though it runs first.
- `run_vina_docking` returns `{"score_only", "minimize", "dock"}` — three floats. **The
  redocked pose itself is never returned or persisted**, only its affinity.

So the orange series cannot be drawn from what is stored. `dock_core_poses.py` produces it:
it re-docks the core three arms at the protocol's exhaustiveness 16 and stages **both**
poses per molecule, and `compute_baseline_interactions.py` fingerprints the staged layout
with the same verified interpreter that scores everything else here.

The baselines are the same: `stage_baseline_poses.py` writes the poses out of the results
bundle's meta `.pt` — the generated coordinates — and `compute_baseline_interactions.py`
fingerprints exactly those. That is also why the heavy-atom sequence check in
`stage_baseline_poses.py` is meaningful; a redocked pose would still pass it, but the
coordinates never go near Vina.

### What the paired run covers

79 pockets × 3 arms, 13 molecules per pocket per model arm (evenly spaced across generation
order, never a prefix) and the one crystal ligand for the reference. 2,133 picked, **2,105
paired**, 3 h 30 m on 16 workers. The 28 that did not pair:

| dropped | why |
|---|---|
| target_71, all three arms (27) | the pocket10 receptor has a chain gap at GLU 866 and `pdb2pqr30` refuses it, so nothing docks there. Deterministic, molecule-independent, and the same reason every other docking result in this project rests on 78 pockets rather than 79 |
| VoxBind target_96 row 8 (1) | the sample is `CNNO`, four heavy atoms; its extent gives Vina a box with a zero dimension |

**A failed dock removes its molecule from both series, not just the docked one.** Otherwise
the generated column would carry 28 molecules the docked column cannot have, and the
difference between the columns is supposed to be the docking alone. `manifest.json`
records every dropped row with its SMILES and the exception, and lists the surviving row
indices per pocket — which a drop makes discontiguous, so the scorer walks that list rather
than `range(n)`, keeping each molecule's heavy-atom count with the molecule.

### Reading the pair

`build_gen_vs_docked.py` draws it: two adjacent columns per method, the redocked one
hatched, and `interactions_pair.csv` carries the numbers.

**Colour stays the method's**, which is where this figure departs from Fig. 14 deliberately.
Fig. 14 gives each series its own ramp — green `#72b6a1`→`#eaf4f1` generated, salmon
`#e99675`→`#fcefea` redocked — and puts the methods on the x axis, which it can afford
because the series is the only other thing in its panel. Here colour means the method across
four folders of this section, so the pair takes the two channels Fig. 14 leaves free:
adjacency and a hatch. The method names come back to the x axis because the reason the
sibling figures dropped them does not apply here: three pairs have room for them upright
where eight rotated names cost a third of the figure's height. That leaves the legend free
to name the pose condition, which is what this figure is about — each channel gets its own
place, the method on the axis and the condition in the key.

**The crystal ligands are the control, and they behave.** They are already in a measured
pose, so redocking should barely move them, and it does not: van der Waals contacts change
by **0.000** per molecule and the four types together by **+0.04**. Against that floor the
model arms both gain — VoxBind **+1.10**, ours **+0.71** — and the gap between those two is
the readable quantity. A generated pose that already sits where Vina would put it has little
left to gain from redocking; one that is merely plausible-looking has more.

| | VdW | Hydrophobic | HBAcceptor | HBDonor | all four |
|---|---|---|---|---|---|
| Reference | +0.00 | +0.19 | −0.19 | +0.04 | **+0.04** |
| VoxBind | +0.53 | +0.23 | +0.17 | +0.17 | **+1.10** |
| `\textsc{CoDE}` | +0.31 | +0.12 | +0.15 | +0.13 | **+0.71** |

Mean per molecule, generated → redocked. **This is not a leaderboard either**, and for a
second reason on top of the size one: a smaller gain means the generated pose was already
docking-like, not that the pose is better. Read it beside the absolute counts, which are in
the same CSV.

That the two series really are one molecule in two poses was checked on all 2,105: element
list, formal charges, atom order and bond table identical for 2,088, and for the other 17
identical too once the ring is read rather than the Kekulé form it happens to be written in
— connectivity and flat SMILES match on every one. Stereo perception differs on 5, where
the docked geometry is too near-planar at one centre for RDKit to re-assign it from
coordinates; fingerprints do not read CIP labels. Centroids moved a median of 1.05 Å and no
pose came back unmoved.

## The method order

Every figure and every table here is in the row order of
`../../260827/table_drug_design.tex` — Reference, AR, Pocket2Mol, DiffSBDD, TargetDiff,
DecompDiff, VoxBind, CoDE — the table's `VoxBind + CDG` row. Not a ranking, and not the
order the data happens to load in: a reader moves between that table and these panels, and
re-finding a method in a different order each time is a cost paid on every glance. `ORDER`
in the builder is the one place it is written down, and it **raises** on a method it does
not name rather than quietly appending it — spellings resolve through
`method_colors.ALIASES` first, so the order survived our method's rename to CoDE on
2026-09-10 without touching a figure. The only exception is the per-heavy-atom table below,
which is sorted by value on purpose — it is the one table here that *is* a ranking.

## What it says

Mean interactions per pose, over the 79 pockets:

| method | heavy atoms | VdW | Hydrophobic | HB accepted | HB donated | all |
|---|---|---|---|---|---|---|
| *Reference ligand* | *22.8* | *8.40* | *1.57* | *1.96* | *0.92* | *12.86* |
| AR | 17.2 | 7.03 | 1.02 | 1.28 | 0.61 | 9.94 |
| Pocket2Mol | 18.5 | 7.07 | 1.73 | 1.01 | 0.39 | 10.20 |
| DiffSBDD | 20.2 | 8.87 | 1.64 | 1.28 | 0.54 | 12.34 |
| TargetDiff | 22.2 | 9.24 | 1.52 | 1.48 | 0.70 | 12.95 |
| DecompDiff | 20.9 | 8.36 | 1.43 | 1.49 | 0.63 | 11.91 |
| VoxBind | 24.0 | 8.25 | 2.10 | 0.95 | 0.49 | 11.79 |
| CoDE | 24.9 | 8.92 | 2.24 | 1.08 | 0.54 | 12.78 |

Share of molecules making the type **at all**:

| method | VdW | Hydrophobic | HB accepted | HB donated |
|---|---|---|---|---|
| *Reference ligand* | *100 %* | *49.4 %* | *74.7 %* | *55.7 %* |
| AR | 99.7 % | 47.4 % | 67.6 % | 46.0 % |
| Pocket2Mol | 99.9 % | 70.5 % | 59.5 % | 32.8 % |
| DiffSBDD | 100 % | 68.6 % | 66.5 % | 41.1 % |
| TargetDiff | 100 % | 66.3 % | 72.2 % | 50.1 % |
| DecompDiff | 100 % | 61.7 % | 69.6 % | 46.2 % |
| VoxBind | 100 % | 78.5 % | 53.1 % | 38.0 % |
| CoDE | 100 % | 77.7 % | 57.8 % | 40.1 % |

Molecule sizes range from 17.2 to 24.9 heavy atoms across these methods, so the raw counts
above are not comparable between them. **Per heavy atom** — the one table here sorted by
value rather than by the drug-design table's order, because it is the one table that is a
ranking (`interactions.json` carries the inputs):

| method | all / atom | HB acc / atom | HB don / atom | Hydrophobic / atom |
|---|---|---|---|---|
| DiffSBDD | 0.610 | 0.0632 | 0.0269 | 0.0812 |
| TargetDiff | 0.583 | 0.0667 | 0.0316 | 0.0686 |
| AR | 0.579 | 0.0747 | 0.0354 | 0.0594 |
| DecompDiff | 0.570 | 0.0713 | 0.0300 | 0.0686 |
| *Reference ligand* | *0.565* | *0.0862* | *0.0406* | *0.0690* |
| Pocket2Mol | 0.551 | 0.0548 | 0.0213 | 0.0933 |
| CoDE | 0.513 | 0.0434 | 0.0216 | 0.0897 |
| VoxBind | 0.491 | 0.0395 | 0.0205 | 0.0875 |

**1. The totals say nothing, and the breakdown says a lot.** Raw totals run 9.94–12.95 and
that spread just tracks molecule size; per atom every method sits between 0.491 and 0.610
against the reference's 0.565, and every method makes VdW contact in ~100 % of poses. Read
only the total and all eight columns look interchangeable. They are not.

**2. Every single method under-produces hydrogen bonds, and it is not close.** Per heavy
atom the crystal ligands accept 0.0862 and donate 0.0406; the *best* model on each is AR at
0.0747 and 0.0354, and the worst is vanilla VoxBind at 0.0395 and 0.0205 — under half the
crystal rate. **7 of 7 methods fall below the reference on both H-bond types**, with the
baselines included. This is a property of the field, not of any one model, and it is the
single clearest gap between generated and crystal poses in this folder.

**3. The deficit is paid in hydrophobic contact.** Per atom the reference sits at 0.0690;
Pocket2Mol (0.0933), CoDE (0.0897), VoxBind (0.0875) and DiffSBDD (0.0812) are
all above it, TargetDiff and DecompDiff land on it (0.0686), and only AR is below (0.0594).
The share-nonzero table is the sharper form: a crystal ligand makes some hydrophobic
contact 49.4 % of the time, and every model except AR does so more often — VoxBind 78.5 %.
**A directional bias, not a quality gap** — the models find the greasy part of the pocket
and under-use its polar contacts.

**4. Ours moves toward the crystal ligands on both H-bond types, and it survives the size
correction.** Raw, it is +0.13 accepted and +0.05 donated over vanilla, and +4.7 pp /
+2.1 pp on the share that makes any — but Ours also generates larger molecules (24.9 against
24.0 heavy atoms), so that could have been arithmetic. Per heavy atom it still leads:
0.0434 vs 0.0395 accepted, 0.0216 vs 0.0205 donated. Small, and in the direction of the
reference. It is *further* from the reference on hydrophobic contact, though — the two arms
are the second- and third-greasiest in the table.

**5. The methods that look best here look worst next door.** AR is closest to the crystal
ligands on H-bond density (0.0747 / 0.0354) and is the least greasy method in the table —
while being the **highest-strain** generator in `../fig-posecheck` at almost every
rotatable-bond count. DecompDiff is second-best on H-bonds and mid-pack on strain. The two
VoxBind arms are the reverse: lowest strain of any method, worst H-bond density of any
method. A pose can make the right *kind* of contact and still be a bad conformer, and the
other way round. That is the reason to read this figure beside its three siblings rather
than in place of them.

## Where the baselines' fingerprints come from

**As of 2026-09-10 the five published baselines are `pose_common.ARMS` entries**, staged by
`tools/stage_baseline_samples.py` under `exps/baselines_pose/` and scored by
`86_pose_eval_baselines.sh`. That scoring was still running when this figure was last built,
and what it has produced so far is PoseBusters, not PoseCheck — `pc.arms_for("ifp", data)`
returns only TargetDiff, VoxBind and CoDE. So **this folder still takes its baselines from
its own `interactions_<Method>.json`**, and the builder drops a baseline from that side
channel the moment the same name turns up as a fully scored arm, so no method can enter a
figure twice. When the run finishes, the recompute below becomes redundant and the two
sources should be reconciled against each other before it is deleted — they are both the
`pocket10` crop, but they are not the same code path.

`posecheck_<Method>.json` — the per-molecule baseline export the strain figures use —
carries **only strain and clashes**. The fingerprint was never exported, and the raw
per-chunk PoseCheck output stays on svr12. But the poses themselves are now in the results
bundle, so the fingerprint is **recomputed here** rather than fetched, in two steps because
no single environment can do both halves: `torch` (needed to open the meta `.pt`) is in
`voxbind`, and PoseCheck/ProLIF are in `moleval`.

1. `stage_baseline_poses.py` (voxbind) writes the p79 poses to SDF in the exact order
   `export_posecheck_json.py` walked them, checking the heavy-atom sequence against the
   export per pocket and aborting on any mismatch. It matched 100/100 pockets for all four
   methods: 7,655 / 7,772 / 7,720 / 6,427 molecules.
2. `compute_baseline_interactions.py` (moleval) scores them against the **same pocket10
   crop** the local arms use, and writes `interactions_<Method>.json`.

**Recomputing is only legitimate if it reproduces what is already stored, so that is
checked before anything else runs.** Re-scoring 298 local-arm molecules over three pockets
with the `moleval` interpreter reproduced their `metrics.json` fingerprints **exactly —
100.0 %, 298/298 on all four types**. The run aborts if it ever stops matching
(`--verify` runs the check alone). All 29,574 baseline molecules scored; none failed.

Two further checks, because a silent frame or identity error here would look like a result:
the baseline ligand centroids sit **1–5 Å** from their pocket centroid (same coordinate
frame), and the four interaction types recovered are exactly the four the local arms have.

**FuncBind is absent.** Its strain export exists (9,992 molecules) but no meta for it
reached this box, and its shard SDFs are a *different sampling run* — they fail the
heavy-atom check at 99 of 100 pockets. See `../fig-posecheck/README.md`.

## Three things behind the numbers

**A type a molecule does not make is absent from its fingerprint, not zero in it.** Every
scored molecule contributes a 0 to every type it did not make, and the violins are over all
scored molecules — not over the ones that happened to make that type. Counting only the
latter would turn *"Ours makes H-bond donors less often"* into *"Ours makes more of them
when it does"*, which is a different and much smaller claim. `counts()` in the builder is
the one line that decides this.

**The pocket is the `pocket10` crop**, the same source `../fig-posecheck/build_posecheck_per_atom.py`
uses — not the whole-receptor scoring in `frozenenc_probes/posecheck_full/`. That scoring
exists and covers all 79 pockets for all three arms, but it has `reference: null`, so it
cannot draw the crystal ligands, which are the whole point. The cost was **measured over
the 21,865 molecules the two scorings share**, not assumed: per molecule the crop finds
0.16–0.29 fewer VdW contacts (1.8–3.2 %) and is within 0.03 of the same count on all three
other types. A 10 Å crop around the crystal ligand contains essentially everything inside
ProLIF's cutoffs. Do *not*, however, read a number here against one from
`../fig-posecheck/build_posecheck_all_by_atom_range.py`, which **is** whole-receptor.

**Both remaining violins use one absolute KDE bandwidth (0.45 of a count), not
matplotlib's default.** Scott's rule scales bandwidth by n^(-1/5), and the crystal ligands
are 79 molecules against each arm's ~7,800 — so with the default the reference came out
visibly smoother than every arm beside it *for no reason but its sample size*, on a figure
whose whole job is comparing shapes. Their panel top is the largest count anyone actually
made rather than a percentile, because cutting at p99.5 sliced the KDE tails off mid-body
and left a row of hard vertical edges that read as real structure. Neither point applies to
the two H-bond panels any more — they have no bandwidth, and their top is a letter value.

**Both panels of both figures step the y axis by whole counts.** On the H-bond figure that
falls out of the cell axis; on the contacts figure it is a `MaxNLocator(integer=True)`,
because the default offered 0.0 / 2.5 / 5.0 on the hydrophobic panel as soon as the panels
grew to full height — a tick at two and a half contacts, beside a companion panel stepping
by whole ones.

## What rides Dropbox

```bash
bash dropbox_sync.sh status     # what is where
bash dropbox_sync.sh pull       # take what Dropbox has, never overwriting a local file
bash dropbox_sync.sh push       # send what this box has
```

Remote is `박성현/VoxBind/results/task2-drugdesign/_shared/260910_interaction`, and only the
things that cost real time to make again go: the baselines' fingerprints (~9 min), the
paired gen/docked JSON, and the staged poses (~2,100 Vina docks, 3 h 30 m). The figures,
`interactions.{json,csv}` and the per-worker `poses/_tmp/` scratch stay out — the first two
rebuild in seconds, and syncing a derived file only invites a stale copy to beat a fresh
build.

Pull is `rclone copy --ignore-existing`, so a file already here wins; delete it first if you
actually want the remote one. Both directions are `copy`, never `sync`, so neither end ever
deletes at the other.

**Checked before any of this was written, 2026-09-12: Dropbox had none of it.**
`_shared/260910_posecheck/posecheck_<M>.json` carries only `p`/`n`/`s`/`c` — pocket, heavy
atoms, strain, clashes — with no fingerprint, which is exactly why this folder recomputes
them; `_shared/posecheck_interactions_78.csv` has interaction *means* for the local arms
only. Nothing docked was staged anywhere. So the first run of the script was a push.
