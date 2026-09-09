# PoseCheck — strain and steric clashes

Everything in this folder measures PoseCheck. Its sibling `../fig-posebusters` measures
PoseBusters dock-mode validity. The two split by metric and share `../pose_common.py`
(arm list, pocket set, loader, house furniture) and `../method_colors.py` (one colour per
method), which is everything a disagreement between them would silently corrupt. **Each
folder carries the code that draws its own figures.**

## The five builders

| builder | draws | runs here? |
|---|---|---|
| `build_posecheck_per_atom.py` | strain and clashes against ligand SIZE, three local arms | yes |
| `build_strain_per_rotbond.py` | strain against ROTATABLE BONDS, the same three arms | yes |
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

**Two variants of every figure.** `_core` draws VoxBind, VoxBind + Ours and the crystal
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
| VoxBind + Ours | 24.9 | 81.9 | 5.0 | 7,873 |
| *Reference ligand* | *22.8* | *34.3* | *5.0* | *79* |

Strain medians at 15 / 25 / 35 heavy atoms: TargetDiff 141 / 577 / 1425, VoxBind
29 / 65 / 116, VoxBind + Ours 29 / 90 / 169. So **the two VoxBind arms stay within a factor
of ~2-3 of the crystal ligands across the whole size range while TargetDiff runs 5-12×
above them**, with Ours v1 drifting slightly higher than vanilla as molecules grow. That is
the same ordering PoseBusters gives from a different measurement, which is the main reason
to trust either — and it is *not* the Vina ordering, where Ours v1 wins. Pose quality and
affinity are separate axes.

One thing only the clash **mean** figure shows: above ~28 heavy atoms the crystal reference
ligands clash more than VoxBind and Ours v1 do (~10 against 6–8). Crystal poses are not a
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
| VoxBind + Ours | 12.9 | 81.1 | 270.8 | 304.5 | 7,826 | 6.56 % |
| *Reference ligand* | *12.9* | *24.1* | *56.2* | *—* | *79* | *0.00 %* |

"molecules scored" is `n_scored` in the JSON — molecules whose UFF relaxation converged, so
it is the sibling section's `n_strain` and not its `n_molecules`. The two exports agree on
it exactly (7,232 / 7,854 / 7,826 / 79), which is the cheapest check that both builders are
reading the same molecules.

Same ordering as the size-resolved view, and the same reading. Over 0–9 bonds, the range
the crystal ligands cover, **VoxBind runs 0.9–3.6× the reference and Ours v1 1.0–4.8×,
while TargetDiff runs 6.0–19.4×**. Both VoxBind arms sit *at* crystal-ligand strain at 0
rotatable bonds (0.9× and 1.0×), where there is no torsional freedom left to get wrong, and
separate from it as torsions are added — the whole gap is in the torsions, which is the
argument for this axis over the size one.

Between the two VoxBind arms, Ours v1 is 23–76 % above vanilla at every count from 1 to 8
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
an invented value. Same narrowing, for the same reason, as `../fig-consistency`. In the
exports the reference's `n` is that window's pool, not the exact-count one.

**The mean panel is not clipped, where the per-atom one is.** That figure clips to 1e6
because its curves have a bulk below it and only a few thin heavy-atom counts spike out.
This axis has no such bulk: 15 bond counts pool 500–1,100 molecules each, so nearly every
point catches one of the conformers that relax to 1e8–1e13, and the same clip left the
curve as disconnected fragments with most of it off-panel. Drawn whole it spans nine
decades, sits 6–9 decades above the crystal ligands and carries no ordering at all — which
is the honest picture of what a mean does to this metric, and is itself the argument for
the median. The tail driving it is in the table above and in the JSON as `strain_gt_1e4`.

## All eight methods, binned (`build_posecheck_all_by_atom_range.py`)

`strain_ecdf_<bin>.*` and `clash_violin_<bin>.*` over `le15`, `16_20`, `21_25`, `26_30`,
`gt30` and `all` (all sizes), plus `posecheck_all_by_atom_range.{json,csv}`. Colours come
from `../method_colors.py`, so an arm reads the same here as in every other figure.

This family scores the local arms against the **whole receptor**
(`frozenenc_probes/posecheck_full/`), while `build_posecheck_per_atom.py` uses the pocket10
**crop** that the target `metrics.json` files carry. The two are not interchangeable — a
crop cannot show a clash with an atom it does not contain — so do not read a number from
one against a number from the other.

## Two things to carry into the writeup

1. **Strain here is the posecheck 1.3.1 definition, not the VoxBind paper's.** Numbers are
   ~10× smaller than Fig. 6 of arXiv 2405.03961 and must never be placed beside it. Report
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

## Merging with TargetDiff / vanilla VoxBind / Ours v1

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
