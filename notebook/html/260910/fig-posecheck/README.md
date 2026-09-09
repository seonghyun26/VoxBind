# PoseCheck by heavy-atom range — the five published baselines

Strain and clashes both rise with molecule size, and the methods draw different size
mixes, so a pooled number partly reports the size mix rather than pose quality. Everything
here is therefore stratified by the generated molecule's heavy-atom count, matching
`../build_posecheck_by_atom_range.py`.

Figures cover the **79 electron-density pockets** (the subset tables 1b/2b of
`../baseline.html` use). The JSON exports carry **all 100** so either subset can be drawn
without another round trip.

## What is here

| file | what |
|---|---|
| `strain_ecdf_<bin>.{png,svg}` | cumulative probability of UFF strain, one per bin |
| `clash_violin_<bin>.{png,svg}` | steric-clash distribution, one per bin |
| `posecheck_baselines_by_atom_range.json` | the per-bin numbers behind the figures |
| `posecheck_<Method>.json` | **per-molecule** results, for merging elsewhere |

Bins are `≤15`, `16–20`, `21–25`, `26–30`, `>30` (slugs `le15`, `16_20`, `21_25`,
`26_30`, `gt30`).

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

Those three read from `/home1/irteam/...`, which is not on the box that produced this
folder, so the figures here show the five baselines plus the crystal reference only.
`../build_posecheck_baselines_by_atom_range.py` already declares their roots and merges
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
python ../build_posecheck_baselines_by_atom_range.py   # figures + per-bin JSON
python ../export_posecheck_json.py                     # per-molecule JSON
```
