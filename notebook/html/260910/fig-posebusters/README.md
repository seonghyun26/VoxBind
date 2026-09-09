# PoseBusters — dock-mode validity, three arms, 79 pockets

Everything in this folder measures PoseBusters. Its sibling `../fig-posecheck` measures
PoseCheck (strain and steric clashes). The two split by metric and share
`../pose_common.py` (arm list, pocket set, loader, house furniture) and
`../method_colors.py` (one colour per method), which is everything a disagreement between
them would silently corrupt. **Each folder carries the code that draws its own figures** —
here, `build_posebusters_figures.py`.

Built 2026-09-09 from the per-target `metrics.json` of the runs on this box. PoseBusters
had only ever run on 33/79 of Ours v1 and 53/100 of vanilla, and never on TargetDiff;
`voxbind/scripts/85_fill_pose_eval_4runs.sh` filled the remaining 276 pocket-dirs, so this
is the first build where every arm carries it on every pocket — and the first with the
per-check breakdown that says *why* a validity rate is what it is.

| figure label | also called | run root | pockets |
|---|---|---|---|
| TargetDiff | — | `/home1/irteam/base_drug/eval/targetdiff` | 100 |
| VoxBind | vanilla, σ=0.9 | `exps/_vanilla_ep923/samples/full_eval_ep923` | 100 |
| VoxBind + Ours | Ours v1 | `exps/voxbind_frozenenc_atomblob7_v2p1_sig0.9/samples/full_eval_ep350` | 79 |

**Ours v2** (`exps/samples_reference_receptor_ed_ep350`, 92 pockets) was dropped from this
section on 2026-09-09. It is still evaluated — both metrics are complete on it and
`85_fill_pose_eval_4runs.sh` still fills it — it is just not one of the arms reported here.
Re-adding it is one line in `../pose_common.py`'s `ARMS`.

**Everything plotted is the 79-pocket set** (`frozenenc_probes/p79_targets.json`), the
electron-density pockets all three arms cover — Ours v1 never sampled more. Each arm's own
full-coverage number is in `posebusters_summary.json` under `all_pockets`; it is **not**
comparable across arms and moves the headline by ≤1.3 points. target_71 is included: it is
unscoreable on the *docking* side only, both pose metrics work on it.

The five published baselines (AR / Pocket2Mol / DiffSBDD / DecompDiff / FuncBind) have no
PoseBusters at all — only their PoseCheck exports made it across from the other box, so
they appear in `../fig-posecheck` and cannot appear here without a run there.

## Headline — 79 pockets

| arm | heavy atoms | **PB-valid** | PB-valid, size-std | molecules |
|---|---|---|---|---|
| TargetDiff | 22.2 | **60.9 %** | 58.8 % | 7,287 |
| VoxBind | 24.0 | **69.3 %** | 69.8 % | 7,888 |
| VoxBind + Ours | 24.9 | **67.5 %** | 69.0 % | 7,873 |
| *Reference ligand* | *22.8* | *96.2 %* | *—* | *79* |

`size-std` is the direct-standardized rate: each arm's validity **within** every
1-heavy-atom stratum, re-weighted by one common size distribution (every arm pooled), so
what is left is validity at matched size. It is the number to compare arms with; the crude
rate is what the arm actually produced. Both are in `posebusters_summary.json`.

**Read the bins, not the pool.** Validity falls steeply with molecule size for every method
and the arms draw different size mixes, so a pooled rate partly reports the size mix — the
same confound the Vina numbers have.

| arm | ≤15 | 16–20 | 21–25 | 26–30 | >30 |
|---|---|---|---|---|---|
| TargetDiff | 85.1 % | 66.1 % | 58.6 % | 46.5 % | 37.0 % |
| VoxBind | 85.0 % | 76.3 % | 68.8 % | 62.2 % | 56.5 % |
| VoxBind + Ours | 83.2 % | 76.3 % | 67.7 % | 61.6 % | 56.1 % |

**Size explains more than half of the v1 ↔ vanilla gap, but not all of it.** Crude, vanilla
leads Ours v1 by 1.8 points; size-standardized the lead is **0.8** (69.8 % vs 69.0 %).
Vanilla is ahead in every bin — by 1.8 / 0.0 / 1.1 / 0.6 / 0.4 points from ≤15 to >30 — so
the remaining gap is small but real, not an artefact. Do not claim the two are tied on pose
validity; claim the gap is under one point once size is controlled.

**TargetDiff's deficit is entirely a large-molecule deficit.** In the ≤15 bin it is
actually the *best* arm (85.1 %, ahead of Ours v1 by 1.9 and level with vanilla), and it
only falls behind from 16–20 onward, ending 19.1 points below Ours v1 at >30. Standardized
it lands at 58.8 %, ~11 points below both VoxBind-family arms.

The crystal reference poses hold 92-100 % across every bin, so the size slope is a property
of the generated poses, not of the checks becoming unpassable for big ligands.

## Which checks fail (`pb_check_failures`)

| check | TargetDiff | VoxBind | VoxBind + Ours | Reference |
|---|---|---|---|---|
| non-aromatic ring non-flatness | 8.9 % | 21.1 % | **22.5 %** | 1.3 % |
| bond angles | **22.5 %** | 5.5 % | 5.9 % | 0.0 % |
| minimum distance to protein | **10.0 %** | 1.3 % | 1.6 % | 2.5 % |
| internal steric clash | 6.1 % | 4.1 % | 5.3 % | 0.0 % |
| bond lengths | 1.1 % | 2.9 % | 3.6 % | 0.0 % |
| internal energy | 1.8 % | 2.5 % | 2.6 % | 0.0 % |

The two families fail for different reasons, and the voxel methods' single dominant failure
is one check: **non-aromatic ring non-flatness**, 21-23 % of molecules against the crystal
ligands' 1.3 %. Puckering a saturated ring past the tolerance is what costs VoxBind-family
poses their validity, not protein contact — their `minimum distance to protein` failure
rate (1.3-1.6 %) is *below* the crystal ligands' own 2.5 %. TargetDiff inverts this: it
fails bond angles (22.5 %) and runs into the protein (10.0 %) instead. **Fixing ring
geometry is the single highest-value target for our arms**; it is worth roughly 20 points
of validity on its own.

A check no method fails above 0.5 % is left off the figure — it would be a row of white
space saying only that PoseBusters ran it. The full counts are in
`posebusters_check_failures.json`, and the `_core` figure re-ranks the rows by what its own
two arms fail, so its order differs from `_all`.

## Files

Figures follow the 260903 3-line house style (`../fig-vina-per-atom`): no panel titles,
warm near-black furniture, dotted rules, live text in the SVG and TrueType in the PDF.

**Every figure is written in two variants.** `_core` draws VoxBind, VoxBind + Ours and the
crystal reference — the comparison this section is making, and the same three series the
Vina 3-line figures carry. `_all` adds every baseline the figure has data for (here:
TargetDiff). The split is the one the Vina figures already make, not a subset picked after
seeing the numbers, and the data exports always cover every arm regardless of which variant
a figure draws.

| file | what |
|---|---|
| `build_posebusters_figures.py` | builds everything below |
| `pb_valid_per_atom_{core,all}.*` | **headline** — validity against ligand size, with each arm's size distribution underneath |
| `pb_check_failures_{core,all}.*` | per-check failure rate, all sizes pooled |
| `posebusters_summary.json` | coverage + pooled rates, `p79` and `all_pockets` |
| `posebusters_by_atom_range.{json,csv}` | the per-bin numbers |
| `posebusters_check_failures.json` | per-check failure counts and rates |
| `posebusters_per_molecule_<arm>.json` | per-molecule export (`t` target, `n` atoms, `v` valid, `f` failed checks) |

Each figure is written as `.png`, `.svg` and `.pdf`. The size-distribution strip is kept
only on the headline figure, where the size mix is the argument being made.

Unlike PoseCheck strain, **PoseBusters validity is deterministic** — its checks are
threshold tests on the pose as given, so a re-run reproduces it exactly.
