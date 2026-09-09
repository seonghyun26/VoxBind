# Pose quality — PoseCheck + PoseBusters, four arms, 79 pockets

Built 2026-09-09 by `./build_pose_figures.py` from the per-target `metrics.json` of the
four runs that live on this box. This is the first build where **all four arms carry both
metrics on every pocket**: PoseBusters had only ever run on 33/79 of Ours v1 and 53/100 of
vanilla, and never on TargetDiff or Ours v2. `voxbind/scripts/85_fill_pose_eval_4runs.sh`
filled the remaining 276 pocket-dirs.

| figure label | also called | run root | pockets |
|---|---|---|---|
| TargetDiff | — | `/home1/irteam/base_drug/eval/targetdiff` | 100 |
| VoxBind | vanilla, σ=0.9 | `exps/_vanilla_ep923/samples/full_eval_ep923` | 100 |
| VoxBind + Ours | Ours v1 | `exps/voxbind_frozenenc_atomblob7_v2p1_sig0.9/samples/full_eval_ep350` | 79 |
| Ours v2 | reference receptor_ed | `exps/samples_reference_receptor_ed_ep350` | 92 |

The figure labels are the ones the Vina figures use, so an arm reads the same across the
section; `pose_summary.json` is keyed by them too. The prose below keeps the shorter
"Ours v1" / "Ours v2".

**Everything plotted is the 79-pocket set** (`frozenenc_probes/p79_targets.json`), the
electron-density pockets all four arms cover — Ours v1 never sampled more than those. Each
arm's own full-coverage number is in `pose_summary.json` under `all_pockets`; it is **not**
comparable across arms and moves the headline by ≤1.3 points. target_71 is included: it is
unscoreable on the *docking* side only, both pose metrics work on it.

The five published baselines (AR / Pocket2Mol / DiffSBDD / DecompDiff / FuncBind) are not
here — they were sampled on the other box and only their PoseCheck exports made it across
(`../fig-posecheck/posecheck_*.json`). Adding PoseBusters for them needs a run
there.

## Headline — 79 pockets

| arm | heavy atoms | clash med | strain med | **PB-valid** | PB-valid, size-std | molecules |
|---|---|---|---|---|---|---|
| TargetDiff | 22.2 | 8.0 | 355.3 | **60.9 %** | 59.7 % | 7,287 |
| VoxBind | 24.0 | 4.0 | 62.2 | **69.3 %** | 70.4 % | 7,888 |
| VoxBind + Ours | 24.9 | 5.0 | 81.9 | **67.5 %** | 69.5 % | 7,873 |
| Ours v2 | 21.6 | 6.0 | 202.4 | **66.5 %** | 64.6 % | 7,896 |
| *Reference ligand* | *22.8* | *5.0* | *34.3* | *96.2 %* | *—* | *79* |

`size-std` is the direct-standardized rate: each arm's validity **within** every 1-heavy-atom
stratum, re-weighted by one common size distribution (all four arms pooled), so what is
left is pose quality at matched size. It is the number to compare arms with; the crude rate
is what the arm actually produced. Both are in `pose_summary.json`.

**Read the bins, not the pool.** Validity falls steeply with molecule size for every method
(85 % → 37-57 % from ≤15 to >30 heavy atoms) and the arms draw different size mixes, so a
pooled rate partly reports the size mix — the same confound the Vina numbers have.

| arm | ≤15 | 16–20 | 21–25 | 26–30 | >30 |
|---|---|---|---|---|---|
| TargetDiff | 85.1 % | 66.1 % | 58.6 % | 46.5 % | 37.0 % |
| VoxBind | 85.0 % | 76.3 % | 68.8 % | 62.2 % | 56.5 % |
| VoxBind + Ours | 83.2 % | 76.3 % | 67.7 % | 61.6 % | 56.1 % |
| Ours v2 | 81.0 % | 72.9 % | 64.4 % | 56.1 % | 46.8 % |

**Size explains about half of the v1 ↔ vanilla gap, not all of it.** Crude, vanilla leads
Ours v1 by 1.8 points; size-standardized the lead is **0.9** (70.4 % vs 69.5 %). Vanilla is
ahead in every bin — by 1.8 / 0.0 / 1.1 / 0.6 / 0.4 points from ≤15 to >30 — so the
remaining gap is small but real, not an artefact. Do not claim the two are tied on pose
validity; claim the gap is under one point once size is controlled.

The size correction moves **Ours v2 the other way**: it draws the smallest molecules of the
four (21.6 heavy atoms), which flatters its crude 66.5 % — standardized it drops to 64.6 %,
nearly 5 points below v1. Its deficit is larger than the pooled table suggests.

**TargetDiff's deficit is entirely a large-molecule deficit.** In the ≤15 bin it is
actually the *best* arm (85.1 %, ahead of Ours v1 by 1.9 and level with vanilla), and it
only falls behind from 16–20 onward, ending 19.1 points below Ours v1 at >30. Standardized
it lands at 59.7 %, ~10 points below both VoxBind-family arms.

The crystal reference poses hold 92-100 % across every bin, so the size slope is a property
of the generated poses, not of the checks becoming unpassable for big ligands.

## Which checks fail (`pb_check_failures`)

| check | TargetDiff | VoxBind | VoxBind + Ours | Ours v2 | Reference |
|---|---|---|---|---|---|
| non-aromatic ring non-flatness | 8.9 % | 21.1 % | 22.5 % | 20.5 % | 1.3 % |
| bond angles | **22.5 %** | 5.5 % | 5.9 % | 7.5 % | 0.0 % |
| minimum distance to protein | **10.0 %** | 1.3 % | 1.6 % | 2.1 % | 2.5 % |
| internal steric clash | 6.1 % | 4.1 % | 5.3 % | 4.8 % | 0.0 % |
| bond lengths | 1.1 % | 2.9 % | 3.6 % | 6.2 % | 0.0 % |
| internal energy | 1.8 % | 2.5 % | 2.6 % | 2.6 % | 0.0 % |

The two families fail for different reasons, and the voxel methods' single dominant failure
is one check: **non-aromatic ring non-flatness**, 20-23 % of molecules against the crystal
ligands' 1.3 %. Puckering a saturated ring past the tolerance is what costs VoxBind-family
poses their validity, not protein contact — their `minimum distance to protein` failure
rate (1.3-2.1 %) is *below* the crystal ligands' own 2.5 %. TargetDiff inverts this: it
fails bond angles (22.5 %) and runs into the protein (10.0 %) instead. **Fixing ring
geometry is the single highest-value target for our arms**; it is worth roughly 20 points
of validity on its own.

## PoseCheck says the same thing, at every size (`posecheck_per_atom`)

Strain and clashes split the four arms into the same two pairs, and the split holds at
every ligand size rather than appearing only in the pooled number. At 20 heavy atoms the
strain medians are roughly TargetDiff 350, Ours v2 200, VoxBind 50, VoxBind + Ours 55
against the crystal ligands' 30 — so **VoxBind and Ours v1 sit within a factor of ~2 of
crystal poses across the whole size range, while TargetDiff and Ours v2 run 5–10× above
them.** Clash medians say it again: 4–5 and 5–6 against TargetDiff's 8–15 and Ours v2's
6–11 at the large end.

That is the same ordering PoseBusters gives, arrived at from a different measurement, which
is the main reason to trust either. Note it is *not* the Vina ordering — Ours v1 wins on
affinity — so pose quality and affinity are separate axes here, and Ours v2 is the arm that
loses on both.

## Files

Figures follow the 260903 3-line house style (`../fig-vina-per-atom`): no panel titles,
warm near-black furniture, dotted rules, live text in the SVG and TrueType in the PDF.
Colour is the arm, and the house entries are kept exactly — VoxBind sand, VoxBind + Ours
periwinkle, reference grey — so an arm reads the same here as in the Vina figures.

| file | what |
|---|---|
| `build_pose_figures.py` | builds everything below, from the runs' `metrics.json` |
| `pb_valid_per_atom.{png,svg,pdf}` | **headline** — validity against ligand size, with each arm's size distribution underneath |
| `pb_check_failures.{png,svg,pdf}` | per-check failure rate, all sizes pooled |
| `posecheck_per_atom.{png,svg,pdf}` | strain and clash medians against ligand size |
| `strain_clash_ecdf_pair.{png,svg,pdf}` | the two PoseCheck distributions, pooled, on one shared y |
| `pose_summary.json` | coverage + pooled numbers, `p79` and `all_pockets` |
| `pose_by_atom_range.{json,csv}` | the per-bin numbers behind the figures |
| `pose_check_failures.json` | per-check failure counts and rates |
| `pose_per_molecule_<arm>.json` | per-molecule export (`t` target, `n` atoms, `s` strain, `c` clashes, `v` valid, `f` failed checks) — for merging elsewhere |

## Two things to carry into the writeup

1. **Strain here is the posecheck 1.3.1 definition, not the VoxBind paper's.** Numbers are
   ~10× smaller than Fig. 6 of arXiv 2405.03961 and must never be placed beside it. Report
   the median: the distribution is heavy-tailed enough that the mean reports the UFF
   failure rate instead (vanilla's mean is 2.26e8 against a median of 62).
2. **Strain is not reproducible run-to-run**, even with `randomSeed=0` forced. The same
   crystal ligand scored under two run roots on adjacent days differs by median 0.09,
   p90 ~0.9, max 6.0 kcal mol⁻¹, with only ~25-30 % of pockets matching exactly. Pooled
   medians over ~7.8k molecules are stable; per-molecule strain comparisons are not exact.
   PoseBusters validity has no such problem — its checks are deterministic.
