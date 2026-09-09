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

| arm | heavy atoms | **PB-valid** | PB-valid, size-std | molecules scored |
|---|---|---|---|---|
| TargetDiff | 22.2 | **60.8 %** | 58.7 % | 7,287 |
| VoxBind | 24.0 | **69.3 %** | 69.8 % | 7,888 |
| VoxBind + Ours | 24.9 | **67.5 %** | 68.9 % | 7,873 |
| *Reference ligand* | *22.8* | *96.2 %* | *—* | *79* |

"Molecules scored" is `n_posebusters`, the denominator every rate in the row is over. It
now equals `n_molecules` for every arm. It did not on 2026-09-09: two chunks of 20 in
TargetDiff (target_31, target_85) had hit the 1800 s per-chunk timeout and carried no
verdict. Re-scored at `POSE_CHUNK_SIZE=1` so a slow molecule costs only itself — both
pockets then finished in ~25 min each and TargetDiff went 7,247 → 7,287 scored, moving its
validity by 0.1 points.

`size-std` is the direct-standardized rate: each arm's validity **within** every
1-heavy-atom stratum, re-weighted by one common size distribution (every arm pooled), so
what is left is validity at matched size. It is the number to compare arms with; the crude
rate is what the arm actually produced. Both are in `posebusters_summary.json`.

**Read the bins, not the pool.** Validity falls steeply with molecule size for every method
and the arms draw different size mixes, so a pooled rate partly reports the size mix — the
same confound the Vina numbers have.

| arm | ≤15 | 16–20 | 21–25 | 26–30 | >30 |
|---|---|---|---|---|---|
| TargetDiff | 85.1 % | 66.0 % | 58.6 % | 46.6 % | 36.8 % |
| VoxBind | 85.0 % | 76.3 % | 68.8 % | 62.2 % | 56.5 % |
| VoxBind + Ours | 83.2 % | 76.3 % | 67.7 % | 61.6 % | 56.1 % |

**Size explains more than half of the v1 ↔ vanilla gap, but not all of it.** Crude, vanilla
leads Ours v1 by 1.8 points; size-standardized the lead is **0.9** (69.8 % vs 68.9 %).
Vanilla is ahead in every bin — by 1.8 / 0.0 / 1.1 / 0.6 / 0.4 points from ≤15 to >30 — so
the remaining gap is small but real, not an artefact. Do not claim the two are tied on pose
validity; claim the gap is under one point once size is controlled.

**TargetDiff's deficit is entirely a large-molecule deficit.** In the ≤15 bin it is
actually the *best* arm (85.1 %, ahead of Ours v1 by 1.9 and level with vanilla), and it
only falls behind from 16–20 onward, ending 19.3 points below Ours v1 at >30. Standardized
it lands at 58.7 %, ~10-11 points below both VoxBind-family arms.

The crystal reference poses hold 92-100 % across every bin, so the size slope is a property
of the generated poses, not of the checks becoming unpassable for big ligands.

## Which checks fail (`pb_check_failures`)

| check | TargetDiff | VoxBind | VoxBind + Ours | Reference |
|---|---|---|---|---|
| non-aromatic ring non-flatness | 8.9 % | 21.1 % | **22.5 %** | 1.3 % |
| bond angles | **22.5 %** | 5.5 % | 5.9 % | 0.0 % |
| minimum distance to protein | **10.1 %** | 1.3 % | 1.6 % | 2.5 % |
| internal steric clash | 6.1 % | 4.1 % | 5.3 % | 0.0 % |
| bond lengths | 1.1 % | 2.9 % | 3.6 % | 0.0 % |
| internal energy | 1.7 % | 2.5 % | 2.6 % | 0.0 % |

The two families fail for different reasons, and the voxel methods' single dominant failure
is one check: **non-aromatic ring non-flatness**, 21-23 % of molecules against the crystal
ligands' 1.3 %. Puckering a saturated ring past the tolerance is what costs VoxBind-family
poses their validity, not protein contact — their `minimum distance to protein` failure
rate (1.3-1.6 %) is *below* the crystal ligands' own 2.5 %. TargetDiff inverts this: it
fails bond angles (22.5 %) and runs into the protein (10.1 %) instead. **Fixing ring
geometry is the single highest-value target for our arms**; it is worth roughly 20 points
of validity on its own.

The bars are **counts**, and the arms hold different numbers of molecules, so each carries
its own *n* in the key. Rates are the fair cross-arm comparison and are in the table above
and in `posebusters_check_failures.json`. The crystal ligands are not drawn in the count
figure at all — with 79 of them against ~7,900, their worst row is 1 molecule, an invisible
tick beside a bar of 1,769.

A check no method fails above 0.5 % is left off the figure — it would be a row of white
space saying only that PoseBusters ran it. The full counts are in
`posebusters_check_failures.json`, and the `_core` figure re-ranks the rows by what its own
two arms fail, so its order differs from `_all`.

## SuCOS — the one check `gen` mode adds (`build_sucos.py`)

PoseBusters' `gen` config is `dock` plus **SuCOS**: shape overlap × pharmacophore-feature
overlap against a reference ligand, here each pocket's crystal ligand. 1.0 is a perfect
superposition. It needs `mol_true`, which is why our `dock` run does not have it.

**It is computed but deliberately NOT folded into `valid`.** Every other column asks *is
this pose physically possible*; SuCOS asks *does it sit where the crystal ligand sits*. A
de novo model is not trying to reproduce the crystal ligand, so a low SuCOS is a statement
about novelty and pocket occupancy, not a broken molecule. Running `config="gen"` would
merge the two silently — its chosen binary output is `sucos_within_threshold` at 0.4, which
enters the all-must-pass `valid`. What that would cost, if the two were independent:

| arm | dock `valid` | gen `valid` (upper bound) |
|---|---|---|
| TargetDiff | 60.8 % | 23.9 % |
| VoxBind | 69.3 % | 15.6 % |
| VoxBind + Ours | 67.5 % | 20.5 % |

`build_sucos.py` therefore calls `check_sucos(..., sucos_threshold=0.4)` — exactly what
`gen.yml` configures — and keeps the number separate. It also skips re-running the 20 dock
checks for one extra column: SuCOS needs only the two molecules, ~1.4 s per pocket.

| arm | mean | median | size-std mean | ≥ 0.4 |
|---|---|---|---|---|
| TargetDiff | 0.373 | 0.371 | 0.374 | 39.4 % |
| VoxBind | 0.336 | 0.328 | 0.337 | 22.5 % |
| VoxBind + Ours | 0.361 | 0.355 | 0.365 | 30.3 % |

**Three things worth carrying:**

1. **The ordering flips.** On validity, strain and clashes, TargetDiff is last; on SuCOS it
   is first. The two are asking different questions, and this is the cleanest evidence for
   that — a method can place molecules like the crystal ligand while building them badly.
2. **SuCOS is the one metric here that is NOT size-confounded.** It is nearly flat in
   heavy-atom count (~0.30–0.40 across the whole range), so standardizing barely moves it
   (0.373 → 0.374 for TargetDiff). Everything else in this section needs the size caveat;
   this does not.
3. **Density conditioning raises overlap with the crystal ligand, consistently.** Paired
   per pocket: Ours − VoxBind = **+0.026 mean SuCOS (95 % CI +0.019 … +0.032), higher in
   63 of 79 pockets**. Against TargetDiff it is −0.011 (CI −0.024 … +0.003, higher in
   31/79), i.e. not separable.

**The caveat that decides what (3) means.** Our conditioning channel is the *experimental
electron density of the holo crystal*, which contains the ligand's own density. So "closer
to the crystal ligand" may be the model reading the answer rather than learning to place
molecules well — the leakage question this project already carries elsewhere. The decisive
control is the apo-conditioned arm
(`exps/voxbind_frozenenc_atomblob7_cgd0p2_apo_sig0.9`): if the +0.026 survives with no
ligand density in the map, it is real. **That arm has no samples on this box yet**, so the
control has not been run and (3) should not be reported as a clean win until it has.

| file | what |
|---|---|
| `sucos_ecdf_{core,all}.*` | distribution, with the 0.4 gen threshold marked |
| `sucos_per_atom_{mean,median}_{core,all}.*` | SuCOS against ligand size |
| `sucos_summary.json` | per-arm statistics, size-standardized values, paired tests |
| `sucos_per_molecule_<arm>.json` | per-molecule values (cache; `CACHE=0` to recompute) |

`build_sucos.py` runs in the **`moleval`** env (posebusters lives there), unlike
`build_posebusters_figures.py` which runs in `voxbind`.

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
