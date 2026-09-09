# Interaction fingerprints — what the pose does against the pocket

The fourth pose-quality measure of the 260910 section, and the VoxBind paper's Fig. 14
form. Its three siblings measure whether a pose is *physically plausible*; this one measures
what it actually *touches*. Same 79 pockets, same arms, same colours — this folder shares
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
| `interaction_violin_{core,all}.*` | four panels, one per type; a violin per arm |
| `interactions.{json,csv}` | mean / median / quartiles / share-nonzero per arm and type |

```bash
/opt/conda/envs/voxbind/bin/python build_interactions.py
```

`_core` is VoxBind, VoxBind + Ours and the crystal reference; `_all` adds TargetDiff. Same
split as every other figure in 260910, and it is part of the filename. The exports always
cover every arm regardless of which variant a figure draws.

## What it says

Mean interactions per pose, over the 79 pockets:

| arm | heavy atoms | VdW | Hydrophobic | HB accepted | HB donated | all |
|---|---|---|---|---|---|---|
| TargetDiff | 22.2 | 9.24 | 1.52 | 1.48 | 0.70 | 12.95 |
| VoxBind | 24.0 | 8.25 | 2.10 | 0.95 | 0.49 | 11.79 |
| VoxBind + Ours | 24.9 | 8.92 | 2.24 | 1.08 | 0.54 | 12.78 |
| *Reference ligand* | *22.8* | *8.40* | *1.57* | *1.96* | *0.92* | *12.86* |

Share of molecules making the type **at all**:

| arm | VdW | Hydrophobic | HB accepted | HB donated |
|---|---|---|---|---|
| TargetDiff | 100 % | 66.3 % | 72.2 % | 50.1 % |
| VoxBind | 100 % | 78.5 % | 53.1 % | 38.0 % |
| VoxBind + Ours | 100 % | 77.7 % | 57.8 % | 40.1 % |
| *Reference ligand* | *100 %* | *49.4 %* | *74.7 %* | *55.7 %* |

**1. The totals say nothing, and the breakdown says a lot.** Every arm lands within 1.1 of
the crystal ligands' 12.86 total interactions, and every arm makes VdW contact with the
pocket in 100 % of poses. Read only the total and all four methods look interchangeable.
They are not — they distribute those interactions differently.

**2. The generated ligands trade hydrogen bonds for hydrophobic contact.** Both VoxBind
arms make 34–43 % *more* hydrophobic contacts than the crystal ligands (2.10 / 2.24 against
1.57) and roughly *half* the hydrogen bonds (0.95 / 1.08 against 1.96 accepted; 0.49 / 0.54
against 0.92 donated). The share-nonzero table is the sharper form of it: a crystal ligand
donates at least one H-bond 55.7 % of the time and VoxBind does 38.0 % of the time, while
VoxBind makes some hydrophobic contact in 78.5 % of poses against the crystal ligands'
49.4 %. **This is a directional bias, not a quality gap** — the models are finding the
greasy part of the pocket and under-using its polar contacts.

**3. Ours moves toward the crystal ligands on both H-bond types**, +0.13 accepted and +0.05
donated over vanilla, and +4.7 pp / +2.1 pp on the share that makes any. Small, and it is
**not clean of the size confound**: Ours also generates larger molecules (24.9 against 24.0
heavy atoms), which lifts every count. Ours is also slightly *further* from the reference on
hydrophobic contacts. Treat this as suggestive and check it size-matched from the exports
before it goes in a writeup.

**4. TargetDiff is closest to the crystal ligands on both H-bond types** (1.48 / 0.70,
against VoxBind's 0.95 / 0.49) — while being the worst arm on strain, on PoseBusters
validity and on clashes. A pose can make the right *kind* of contact and still be a
physically bad conformer. That is the reason to read this figure beside its three siblings
rather than in place of them.

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

**Every violin uses one absolute KDE bandwidth (0.45 of a count), not matplotlib's
default.** Scott's rule scales bandwidth by n^(-1/5), and the crystal ligands are 79
molecules against each arm's ~7,800 — so with the default the reference came out visibly
smoother than every arm beside it *for no reason but its sample size*, on a figure whose
whole job is comparing shapes. The panel top is the largest count anyone actually made
rather than a percentile, because cutting at p99.5 sliced the KDE tails off mid-body and
left a row of hard vertical edges that read as real structure.

The type is a **panel title** here, which the 260910 house style otherwise avoids. All four
panels plot the same quantity in the same unit, so the y axis cannot carry the type as well,
and the names are too long to sit rotated beside a half-height panel without being clipped.
Same exception, for the same reason, as the binned PoseCheck figures.
