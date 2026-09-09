# PoseCheck — strain and steric clashes

Everything in this folder measures PoseCheck. Its sibling `../fig-posebusters` measures
PoseBusters dock-mode validity. The two split by metric and share `../pose_common.py`
(arm list, pocket set, loader, house furniture) and `../method_colors.py` (one colour per
method), which is everything a disagreement between them would silently corrupt. **Each
folder carries the code that draws its own figures.**

## The four builders

| builder | draws | runs here? |
|---|---|---|
| `build_posecheck_per_atom.py` | strain and clashes against ligand SIZE, three local arms | yes |
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
