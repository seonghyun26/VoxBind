# PoseBusters — dock-mode validity, eight methods, 79 pockets

Everything in this folder measures PoseBusters. Its sibling `../fig-posecheck` measures
PoseCheck (strain and steric clashes). The two split by metric and share
`../pose_common.py` (arm list, pocket set, loader, house furniture) and
`../method_colors.py` (one colour per method), which is everything a disagreement between
them would silently corrupt. **Each folder carries the code that draws its own figures** —
here, `build_posebusters_figures.py` and `build_sucos.py`.

PoseBusters had only ever run on 33/79 of CoDE and 53/100 of vanilla, and never on anything
else. Two fills closed that: `voxbind/scripts/85_fill_pose_eval_4runs.sh` for our own arms,
and — after the five published baselines' molecules turned up in
`results/task2-drugdesign/<M>/samples/meta/*.pt` as TargetDiff meta bundles —
`tools/stage_baseline_samples.py` + `voxbind/scripts/86_pose_eval_baselines.sh` for the
rest. **All eight methods now carry it on all 79 pockets, scored by the same metrics.py
against the same hard-linked `*_pocket10.pdb` crops.**

| method | source | molecules scored |
|---|---|---|
| AR, Pocket2Mol, DiffSBDD, DecompDiff, FuncBind | `exps/baselines_pose/<key>` (staged from the meta bundles) | 6,427 – 7,895 |
| TargetDiff | `/home1/irteam/base_drug/eval/targetdiff` | 7,287 |
| VoxBind (vanilla, σ=0.9) | `exps/_vanilla_ep923/samples/full_eval_ep923` | 7,888 |
| **CoDE** (`\textsc{CoDE}`) | `exps/voxbind_frozenenc_atomblob7_v2p1_sig0.9/samples/full_eval_ep350` | 7,873 |

Everything is the **79-pocket electron-density set** (`frozenenc_probes/p79_targets.json`),
the pockets every arm covers. `DecompDiff` is the `_ref_prior` variant, the one this
project's tables use everywhere — **it is given priors derived from the reference ligand**,
which matters for how its numbers read below.

## Headline — 79 pockets

| arm | heavy atoms | PB-valid | PB-valid, size-std | molecules scored |
|---|---|---|---|---|
| DecompDiff | 20.9 | 81.1 % | **82.5 %** | 6,427 |
| VoxBind | 24.0 | 69.3 % | **73.0 %** | 7,888 |
| Pocket2Mol | 18.5 | 73.6 % | **72.4 %** | 7,772 |
| **CoDE** | 24.9 | 67.5 % | **71.9 %** | 7,873 |
| TargetDiff | 22.2 | 60.8 % | **63.9 %** | 7,287 |
| AR | 17.2 | 69.9 % | **62.2 %** | 7,615 |
| DiffSBDD | 20.2 | 56.3 % | **55.8 %** | 7,700 |
| FuncBind | 19.0 | 50.1 % | **46.4 %** | 7,895 |
| *Reference ligand* | *22.8* | *96.2 %* | *—* | *79* |

`size-std` is each arm's validity **within** every 1-heavy-atom stratum, re-weighted by one
common size distribution (the eight arms pooled). It is the number to compare arms with;
the crude rate is what the arm actually produced.

**Size reorders the table, and by a lot.** The baselines draw much smaller molecules than we
do — 17.2 heavy atoms for AR against CoDE's 24.9 — and every check gets harder with size. AR
looks 2.4 points *ahead* of CoDE crude and lands 9.7 points *behind* it once size is
controlled. Read the size-standardized column, or the bins.

| arm | ≤15 | 16–20 | 21–25 | 26–30 | >30 |
|---|---|---|---|---|---|
| DecompDiff | 93.1 % | 82.7 % | 78.3 % | 74.9 % | 63.7 % |
| VoxBind | 85.0 % | 76.3 % | 68.8 % | 62.2 % | 56.5 % |
| Pocket2Mol | 83.2 % | 75.8 % | 75.6 % | 65.4 % | 41.2 % |
| CoDE | 83.2 % | 76.3 % | 67.7 % | 61.6 % | 56.1 % |
| TargetDiff | 85.1 % | 66.0 % | 58.6 % | 46.6 % | 36.8 % |
| AR | 84.4 % | 69.2 % | 57.8 % | 35.7 % | 32.8 % |
| DiffSBDD | 75.6 % | 60.7 % | 49.5 % | 38.8 % | 28.3 % |
| FuncBind | 71.3 % | 51.7 % | 34.2 % | 31.4 % | 14.1 % |

**DecompDiff leads at every size**, and the VoxBind pair is second in the large bins — the
only other arms still above 56 % past 30 heavy atoms, where Pocket2Mol falls to 41 % and AR
to 33 %. Pocket2Mol holds up well to 25 atoms and then drops off a cliff.

**CoDE against the model it modifies:** crude, vanilla leads by 1.8 points; size-standardized
by 1.1 (73.0 % vs 71.9 %), and it is ahead in every bin. The gap is real but about a point
once size is controlled — do not report the two as tied.

The crystal reference poses hold 92–100 % across every bin, so the size slope is a property
of the generated poses, not of the checks becoming unpassable for big ligands.

## Which checks fail (`pb_check_failures`)

| check | DecompDiff | VoxBind | Pocket2Mol | CoDE | TargetDiff | AR | DiffSBDD | FuncBind | Reference |
|---|---|---|---|---|---|---|---|---|---|
| non-aromatic ring non-flatness | 1.9 % | 21.1 % | 0.9 % | **22.5 %** | 8.9 % | 0.1 % | 1.2 % | **22.5 %** | 1.3 % |
| bond angles | 3.9 % | 5.5 % | 2.0 % | 5.9 % | **22.5 %** | 13.4 % | **21.2 %** | 14.5 % | 0.0 % |
| minimum distance to protein | 9.9 % | **1.3 %** | 16.8 % | **1.6 %** | 10.1 % | 3.9 % | **21.0 %** | 4.1 % | 2.5 % |
| internal steric clash | 3.2 % | 4.1 % | 0.8 % | 5.3 % | 6.1 % | 10.1 % | 6.5 % | 7.0 % | 0.0 % |
| bond lengths | 2.2 % | 2.9 % | 10.4 % | 3.6 % | 1.1 % | 2.0 % | 19.0 % | **23.1 %** | 0.0 % |
| internal energy | 0.7 % | 2.5 % | 0.2 % | 2.6 % | 1.7 % | 0.9 % | 1.4 % | 5.1 % | 0.0 % |
| aromatic ring flatness | 0.0 % | 0.0 % | 0.5 % | 0.0 % | 0.0 % | **8.5 %** | 0.1 % | 0.0 % | 0.0 % |
| volume overlap with protein | 0.3 % | 0.2 % | 0.6 % | 0.1 % | 0.4 % | 0.4 % | 5.6 % | 0.4 % | 0.0 % |

Three patterns, and none of them is "one method is worse":

**Our arms own the protein-contact checks.** `minimum distance to protein` is 1.3 % for
VoxBind and 1.6 % for CoDE — the best two of the eight, and *below the crystal ligands' own
2.5 %*. DiffSBDD fails it on 21 % of molecules and Pocket2Mol on 17 %. Whatever the voxel
models get wrong, it is not where they put the molecule relative to the protein.

**What they get wrong is one check: non-aromatic ring non-flatness**, 21–23 % against the
crystal ligands' 1.3 %. It is a family trait shared with FuncBind (22.5 %) and, mildly,
TargetDiff (8.9 %); the fragment and autoregressive methods barely fail it at all (AR 0.1 %,
Pocket2Mol 0.9 %, DecompDiff 1.9 %). **Fixing saturated-ring geometry is worth ~20 points of
validity to us and would put CoDE level with DecompDiff.**

**The atom-diffusion methods fail bond geometry instead** — TargetDiff 22.5 % and DiffSBDD
21.2 % on bond angles, DiffSBDD 19 % and FuncBind 23 % on bond lengths — while the voxel pair
sits at 5–6 % and 3–4 %. AR is the only method that meaningfully breaks aromatic ring
flatness (8.5 %).

The bars are **rates** — the share of each set's molecules that fail the check — which is
the fair cross-arm comparison, since the arms hold 6,427 to 7,895 molecules each (the exact
*n* and the raw counts for every arm and check are in `posebusters_check_failures.json`).

They were counts until 2026-09-13, and the switch is what let the **crystal ligands become an
ordinary bar**: on a count axis 79 of them against ~7,900 generated molecules put their worst
row at 2 molecules, an invisible tick beside a bar of 1,822, and they had to be drawn as a
rate-matched marker instead. The price of a rate is that it hides its denominator — the
reference's 2.5 % *is* those 2 molecules, and carries about ±1.8 points of binomial noise
against the arms' ±0.2 — so **the reference bar alone keeps its *n* in the key**. Read it as
"the crystal ligands essentially never fail this", not as a number with two decimals.

**The x axis is log — symlog, to be exact.** The rates that matter span 0.09 % to 23 %, and
on a linear axis everything under ~2 % (volume overlap, internal energy, aromatic ring
flatness, and both of the reference's own rows) was a stub against FuncBind's 23 %. A plain
log axis cannot draw a bar that starts at zero, and 13 of the 81 cells here are an exact zero
with 5 more at a single-digit molecule count — it would clip all of them to whatever floor
the axis was given, making *never fails this* and *fails it 5 times* the same picture. symlog
is linear below 0.1 % (~8 of an arm's ~7,900 molecules) and logarithmic above, so bars still
start at a true zero, a 1-molecule cell still looks like 1 molecule, and nothing is hidden or
invented. The cost is the usual one for bars on a log axis: bar *length* no longer encodes the
rate, only the position of its right end does, so read the ends and not the areas.

A check no method fails above 0.5 % is left off, and the `_core` figure re-ranks rows by what
its own two arms fail, so its order differs from `_all`.

**The key sits outside the axes, centred along the bottom, 3 × 3.** There is no empty corner inside:
the long bars fill the top and the right, and the in-axes box this used to carry reached far
enough left to bury the bottom rows — until 2026-09-13 `double bond flatness` looked empty in
`_all` while AR, DecompDiff and Pocket2Mol were failing it 3.1, 1.8 and 1.4 % of the time.
Three columns fit only because the bars are rates: an arm's own *n* no longer has to be in the
key for its bar to be readable. The bars run in the key's order from the top of each group
down.

**The check names are wrapped, not abbreviated.** `non-aromatic_ring_non-flatness` on one line
is 30 characters and was eating 2.9 in of a 7.6 in figure as a tick label. It cannot be safely
shortened: that check passes when a non-aromatic ring is sufficiently *non*-flat
(`check_nonflat: True`, threshold 0.1 Å), so failing it means **a saturated ring came out
planar** — every abbreviation either flips that sense or reads as the aromatic check sitting
two rows below it.

## SuCOS — the one check `gen` mode adds (`build_sucos.py`)

PoseBusters' `gen` config is `dock` plus **SuCOS**: shape overlap × pharmacophore-feature
overlap against a reference ligand, here each pocket's crystal ligand. It needs `mol_true`,
which is why our `dock` run does not have it.

**It is computed but deliberately NOT folded into `valid`.** Every other column asks *is this
pose physically possible*; SuCOS asks *does it sit where the crystal ligand sits*. A de novo
model is not trying to reproduce the crystal ligand, so a low SuCOS is a statement about
novelty and pocket occupancy, not a broken molecule. Running `config="gen"` would merge the
two silently — its chosen binary output is `sucos_within_threshold` at 0.4, which enters the
all-must-pass `valid`. `build_sucos.py` calls `check_sucos(..., sucos_threshold=0.4)`,
exactly what `gen.yml` configures, and keeps the number separate.

| arm | mean | median | size-std mean | ≥ 0.4 | gen `valid` would be |
|---|---|---|---|---|---|
| DecompDiff | 0.465 | 0.450 | **0.472** | 73.1 % | 59.2 % |
| FuncBind | 0.426 | 0.406 | **0.420** | 52.3 % | 26.2 % |
| **CoDE** | 0.361 | 0.355 | **0.372** | 30.3 % | 20.5 % |
| TargetDiff | 0.373 | 0.371 | **0.372** | 39.4 % | 23.9 % |
| AR | 0.363 | 0.350 | **0.359** | 34.8 % | 24.3 % |
| Pocket2Mol | 0.343 | 0.345 | **0.345** | 29.2 % | 21.5 % |
| VoxBind | 0.336 | 0.328 | **0.344** | 22.5 % | 15.6 % |
| DiffSBDD | 0.329 | 0.332 | **0.331** | 28.1 % | 15.8 % |

(The last column is an upper bound — `dock valid × within-threshold`, which assumes the two
are independent. It is there to show what folding SuCOS into validity would cost, not as a
number to report.)

**The ordering is not the validity ordering.** DiffSBDD is last on both, but Pocket2Mol is
3rd on validity and 6th here, and TargetDiff is 5th on validity and 3rd here. A method can
place molecules like the crystal ligand while building them badly, and the reverse.

**SuCOS is the one metric in this section that is not size-confounded** — nearly flat in
heavy-atom count, so standardizing moves it by ≤0.01 everywhere.

**Density conditioning raises overlap with the crystal ligand, consistently.** Paired per
pocket, from `sucos_summary.json`:

| | mean diff | 95 % CI | CoDE higher in |
|---|---|---|---|
| CoDE − VoxBind | **+0.026** | +0.019 … +0.032 | **63 / 79** |
| CoDE − TargetDiff | −0.011 | −0.024 … +0.003 | 31 / 79 |
| CoDE − DecompDiff | −0.098 | −0.110 … −0.086 | 3 / 79 |

**Two reasons not to over-read that table.** Ours: CoDE's conditioning channel is the
*experimental electron density of the holo crystal*, which contains the ligand's own density,
so "closer to the crystal ligand" may be the model reading the answer rather than learning to
place molecules well. The control is the apo-conditioned arm
(`exps/voxbind_frozenenc_atomblob7_cgd0p2_apo_sig0.9`), which has no samples on this box.
Theirs: **DecompDiff here is `_ref_prior`** — it is handed priors derived from the reference
ligand, so leading a reference-similarity metric by 0.10 is close to by construction. Neither
number is a clean win for the method it favours.

## Files

Figures follow the 260903 3-line house style (`../fig-vina-per-atom`): no panel titles, warm
near-black furniture, dotted rules, live text in the SVG and TrueType in the PDF.

**Every figure is written in two variants.** `_core` draws VoxBind, CoDE and the crystal
reference — the comparison this section is making. `_all` adds every method the figure has
data for; here that is all eight. The data exports always cover every arm regardless of which
variant a figure draws, and `pose_common.arms_for` keeps an arm out until all 79 of its
pockets are scored, so a run in progress cannot appear as a complete-looking curve.

One cost of eight arms: the `_all` per-atom figure spans **6–33 heavy atoms** rather than
`_core`'s 5–45, because the x range is the counts where *every drawn arm* holds at least 25
molecules, and the baselines stop producing large molecules sooner.

| file | what |
|---|---|
| `build_posebusters_figures.py` | validity figures + the exports below (`voxbind` env) |
| `build_sucos.py` | SuCOS figures + exports (**`moleval` env** — posebusters lives there) |
| `pb_valid_per_atom_{core,all}.*` | **headline** — validity against ligand size, with each arm's size distribution underneath |
| `pb_check_failures_{core,all}.*` | per-check failure rates |
| `sucos_ecdf_{core,all}.*` | SuCOS distribution, 0.4 gen threshold marked |
| `sucos_per_atom_{mean,median}_{core,all}.*` | SuCOS against ligand size |
| `posebusters_summary.json` | coverage + pooled rates, `p79` and `all_pockets` |
| `posebusters_by_atom_range.{json,csv}` | the per-bin numbers |
| `posebusters_check_failures.json` | per-check failure counts and rates |
| `posebusters_per_molecule_<arm>.json` | per-molecule export (`t`, `n`, `v`, `f`) |
| `sucos_summary.json` · `sucos_per_molecule_<arm>.json` | SuCOS numbers, paired tests, per-molecule cache |

Unlike PoseCheck strain, **PoseBusters validity is deterministic** — its checks are threshold
tests on the pose as given, so a re-run reproduces it exactly.
