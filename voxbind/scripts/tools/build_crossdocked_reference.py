#!/usr/bin/env python3
"""build_crossdocked_reference.py — the CrossDocked-wide reference ligand set, for the
Vina-against-ligand-size figures.

    <repo>/voxbind/scripts/tools/build_crossdocked_reference.py
    -> results/task2-drugdesign/Reference/crossdocked_reference_dock.json

figures/draw.py's fig-vina-per-atom draws a grey reference curve behind the models. Its
default population is the 79 crystal ligands of the test pockets -- one per pocket, docked
under our baseline protocol -- which is thin above ~36 heavy atoms and stops the curve
there. This writes the whole-dataset alternative: every CrossDocked ligand the dataset
itself carries a Vina score for.

WHERE THE NUMBERS COME FROM, and why that matters.

  vina_dock   `targetdiff/data/affinity_info.pkl` -- 184,087 entries, one per CrossDocked
              pose, each {"rmsd", "pk", "vina"}. This is CROSSDOCKED2020'S OWN DOCKING,
              computed when the dataset was built. It is NOT our baseline protocol (whole
              *_rec.pdb receptor, exhaustiveness 32, our receptor files), and the two are
              not interchangeable per ligand: over the 79 test references they agree to
              -0.076 kcal/mol at the median but individual ligands differ by as much as
              9 kcal/mol (target_91: ours -5.74, dataset +3.39).

              A curve here is a median over thousands of ligands per heavy-atom count, so
              that per-ligand scatter averages out and the median offset is negligible
              against the ~8 kcal/mol the panel spans. A per-ligand comparison would not
              be. The figure labels this population "Reference ligand (CrossDocked, n=...)"
              precisely so it is never read as our re-dock.

  n_atoms     counted off each ligand's own molfile atom block, excluding hydrogens, from
              voxbind/dataset/data/crossdocked_pocket10/<pocket>/<ligand>.sdf.

WHICH LIGANDS. `--source split` (default) takes THE SET THE DATASET ACTUALLY USES:
`split_by_name.pt`'s train + test lists, 100,100 ligand files, of which 100,090 carry both
a Vina score and an SDF. That is the CrossDocked the models were trained and tested on, so
it is the population a "CrossDocked reference" should mean.

It is a mixture of pose kinds -- 52,482 `_docked_` and 47,618 `_min_` -- because the split
is. An earlier version of this script took `_min_0` only (89,973 ligands, "the minimised
crystal pose") on the theory that it was the same kind of object as the 79 test references.
That was wrong in both directions: it pulled in `_min_0` entries the dataset never uses and
dropped every `_docked_` pose that it does. `--source min0` still builds that set, and
`--source all` takes every scored entry in affinity_info (184,087), but neither is the
dataset.
"""
import argparse
import json
import os
import pickle
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
AFFINITY = REPO / "targetdiff" / "data" / "affinity_info.pkl"
# split_by_name.pt names the ligand FILES; the sibling checkout carries it.
SPLIT = Path(os.environ.get("VOXBIND_BASEDRUG", REPO.parent / "base_drug")) \
    / "targetdiff" / "data" / "split_by_name.pt"
SDF_ROOT = REPO / "voxbind" / "dataset" / "data" / "crossdocked_pocket10"
OUT = REPO / "results" / "task2-drugdesign" / "Reference" / "crossdocked_reference_dock.json"


def heavy_atoms(path):
    """Heavy-atom count from the molfile atom block. Line 4 is the counts line; each of the
    next `n` lines is an atom whose 4th field is its element symbol."""
    try:
        with open(path, "r", errors="replace") as fh:
            lines = fh.readlines()
    except OSError:
        return None
    if len(lines) < 4:
        return None
    try:
        n = int(lines[3][:3])
    except ValueError:
        return None
    h = sum(1 for ln in lines[4:4 + n]
            if len(ln.split()) >= 4 and ln.split()[3] != "H")
    return h or None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", choices=("split", "min0", "all"), default="split",
                    help="split = the dataset's own train+test ligands (default); "
                         "min0 = minimised crystal poses only; all = every scored entry")
    ap.add_argument("-o", "--out", default=str(OUT))
    args = ap.parse_args()

    for p in (AFFINITY, SDF_ROOT):
        if not p.exists():
            sys.exit(f"missing input: {p}")

    aff = pickle.load(open(AFFINITY, "rb"))
    if args.source == "split":
        if not SPLIT.exists():
            sys.exit(f"missing input: {SPLIT} (needed for --source split)")
        import torch
        sn = torch.load(SPLIT, weights_only=False)
        keys = [lig[:-4] for part in ("train", "test") for _, lig in sn[part]]
        selection = f"dataset split (train+test) from {SPLIT.name}"
    elif args.source == "min0":
        keys = [k for k in aff if k.endswith("_min_0")]
        selection = "minimised crystal pose (_min_0)"
    else:
        keys = list(aff)
        selection = "every scored entry"
    print(f"{len(aff):,} entries in {AFFINITY.name}; {len(keys):,} selected ({selection})")

    ligands, no_sdf, no_count, no_vina = [], 0, 0, 0
    for k in keys:
        v = (aff.get(k) or {}).get("vina")
        if v is None:
            no_vina += 1
            continue
        path = SDF_ROOT / f"{k}.sdf"
        if not path.exists():
            no_sdf += 1
            continue
        n = heavy_atoms(path)
        if not n:
            no_count += 1
            continue
        ligands.append({"n_atoms": n, "vina_dock": round(float(v), 4),
                        "pocket": k.split("/")[0], "ligand": k.split("/", 1)[1]})

    if not ligands:
        sys.exit("no ligands resolved — nothing written")

    vals = [l["vina_dock"] for l in ligands]
    sizes = [l["n_atoms"] for l in ligands]
    doc = {
        "protocol": {
            "engine": "CrossDocked2020 dataset docking (targetdiff affinity_info.pkl)",
            "dock_receptor_scope": "crossdocked pocket, as the dataset shipped it",
            "NOT": "our baseline protocol (whole *_rec.pdb, exhaustiveness 32). Median "
                   "agreement with it over the 79 test references is 0.08 kcal/mol, but "
                   "single ligands differ by up to ~9. Use for a pooled curve, never for "
                   "a per-ligand comparison.",
            "pose_selection": selection,
            "pose_kinds": {
                "docked": sum(1 for l in ligands if "_docked_" in l["ligand"]),
                "min": sum(1 for l in ligands if "_min_" in l["ligand"]),
            },
            "n_atoms": "heavy atoms, counted off each ligand's molfile atom block",
        },
        "source": {
            "affinity_info": str(AFFINITY.relative_to(REPO)),
            "sdf_root": str(SDF_ROOT.relative_to(REPO)),
            "built_by": str(Path(__file__).resolve().relative_to(REPO)),
        },
        "n_ligands": len(ligands),
        "n_pockets": len({l["pocket"] for l in ligands}),
        "dropped": {"no_sdf": no_sdf, "no_atom_count": no_count, "no_vina": no_vina},
        "summary": {
            "vina_dock_mean": round(st.mean(vals), 4),
            "vina_dock_median": round(st.median(vals), 4),
            "n_atoms_mean": round(st.mean(sizes), 2),
            "n_atoms_median": st.median(sizes),
            "n_atoms_min": min(sizes), "n_atoms_max": max(sizes),
        },
        "ligands": ligands,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh)
    print(f"{len(ligands):,} ligands over {doc['n_pockets']:,} pockets "
          f"-> {out.relative_to(REPO)}  ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"  dropped: {no_sdf} no sdf, {no_count} no atom count, {no_vina} no vina")
    print(f"  vina_dock median {doc['summary']['vina_dock_median']:+.3f}   "
          f"heavy atoms {doc['summary']['n_atoms_min']}–{doc['summary']['n_atoms_max']} "
          f"(median {doc['summary']['n_atoms_median']})")


if __name__ == "__main__":
    main()
