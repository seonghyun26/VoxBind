"""extract_misato_targets.py — MISATO-derived regression targets → pdb_id→value JSON.

MISATO ships HDF5 (QM ligand electronics + MD trajectories). voxbind's env has no
h5py, so — like extract_bfactors.py — this isolates the HDF5 read into a small
extraction step (run it with an h5py-equipped interpreter, e.g.
/home/shpark/.conda/envs/cellmaes/bin/python) that emits a compact JSON map the
probe loads with no h5py dependency. Keys are lowercased PDB IDs (MISATO stores
them uppercase) so they join LP_PDBBind splits directly.

    QM  (small, available now):
      /home/shpark/.conda/envs/cellmaes/bin/python dataset/extract_misato_targets.py \
          --source qm --misato_dir dataset/data/misato
      → dataset/data/misato/misato_qm.json
        per ligand: partial_charge (mean|q| over GFN2 atom charges) + EA/hardness/IP/…

    MD  (after the 133 GiB MD.hdf5 finishes):
      ... --source md ...
      → dataset/data/misato/misato_md.json
        per complex: adaptability, rmsf, pose_rmsd, interaction_energy (pocket-mean
        for the per-atom ones; per-complex mean over frames for the rest).

Pocket-mean for adaptability/rmsf uses MISATO's own atom indexing (frame-invariant
scalars, so no alignment to the voxel crop is needed).
"""
import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np

CHARGE_PROP = "gfn2_charge"          # MISATO's primary (GFN2-xTB) atomic partial charge
MOL_SCALARS = ["Electron_Affinity", "Hardness", "Ionization_Potential",
               "Electronegativity", "molecular_weight", "total_charge"]


def finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def extract_qm(qm_path: Path) -> dict:
    """Per-ligand QM targets. partial_charge = mean|q| over the ligand's GFN2 charges."""
    out = {}
    with h5py.File(qm_path, "r") as f:
        for pid in f.keys():
            g = f[pid]
            try:
                names = [s.decode() if isinstance(s, bytes) else s
                         for s in g["atom_properties/atom_properties_names"][:]]
                ci = names.index(CHARGE_PROP)
                vals = g["atom_properties/atom_properties_values"][:]
                q = vals[:, ci].astype(np.float64)
                rec = {"partial_charge": float(np.abs(q).mean())}
                for s in MOL_SCALARS:
                    key = f"mol_properties/{s}"
                    if key in g:
                        rec[s.lower()] = float(np.asarray(g[key][()]))
                out[pid.lower()] = rec
            except Exception as e:
                print(f"  [skip] {pid}: {e!r}")
    return out


def _pocket_atom_mask(g, cutoff: float = 6.0) -> np.ndarray:
    """Protein heavy atoms within `cutoff` Å of any ligand atom, on frame 0.

    MISATO MD groups store all atoms; the ligand is the last molecule
    (molecules_begin_atom_index[-1] : end). Returns a boolean mask over atoms.
    """
    coords0 = g["trajectory_coordinates"][0]                       # (n_atoms, 3)
    begin = np.asarray(g["molecules_begin_atom_index"][:]).astype(int)
    lig_start = begin[-1]
    lig = coords0[lig_start:]
    prot = coords0[:lig_start]
    # min distance of each protein atom to any ligand atom
    d = np.sqrt(((prot[:, None, :] - lig[None, :, :]) ** 2).sum(-1)).min(axis=1)
    mask = np.zeros(coords0.shape[0], dtype=bool)
    mask[:lig_start] = d <= cutoff
    return mask


def extract_md(md_path: Path) -> dict:
    """Per-complex MD targets: adaptability/rmsf (pocket-mean), pose_rmsd, interaction_energy."""
    out = {}
    with h5py.File(md_path, "r") as f:
        for pid in f.keys():
            g = f[pid]
            rec = {}
            try:
                if "frames_rmsd_ligand" in g:
                    rec["pose_rmsd"] = float(np.asarray(g["frames_rmsd_ligand"][:]).mean())
                if "frames_interaction_energy" in g:
                    rec["interaction_energy"] = float(np.asarray(g["frames_interaction_energy"][:]).mean())
                # per-atom flexibility → pocket-mean
                pocket = _pocket_atom_mask(g)
                if "atoms_adaptability" in g:
                    a = np.asarray(g["atoms_adaptability"][:])
                    rec["adaptability"] = float(a[pocket].mean()) if pocket.any() else float(a.mean())
                # RMSF from the trajectory (per-atom std of position about the mean)
                xyz = np.asarray(g["trajectory_coordinates"][:])      # (T, n_atoms, 3)
                rmsf = np.sqrt(((xyz - xyz.mean(0, keepdims=True)) ** 2).sum(-1).mean(0))  # (n_atoms,)
                rec["rmsf"] = float(rmsf[pocket].mean()) if pocket.any() else float(rmsf.mean())
                out[pid.lower()] = {k: v for k, v in rec.items() if finite(v)}
            except Exception as e:
                print(f"  [skip] {pid}: {e!r}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["qm", "md"], required=True)
    ap.add_argument("--misato_dir", type=Path,
                    default=Path(__file__).parent / "data" / "misato")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.source == "qm":
        src = args.misato_dir / "QM.hdf5"
        out = args.out or (args.misato_dir / "misato_qm.json")
        print(f"=== MISATO QM targets ===\n  in : {src}\n  out: {out}")
        rec = extract_qm(src)
        field = "partial_charge"
    else:
        src = args.misato_dir / "MD.hdf5"
        out = args.out or (args.misato_dir / "misato_md.json")
        print(f"=== MISATO MD targets ===\n  in : {src}\n  out: {out}")
        rec = extract_md(src)
        field = "rmsf"

    out.write_text(json.dumps(rec, indent=0))
    usable = [r[field] for r in rec.values() if field in r and finite(r[field])]
    print(f"  wrote {len(rec):,} records → {out}  ({out.stat().st_size/1e6:.1f} MB)")
    if usable:
        u = np.array(usable)
        print(f"  {field}: n={len(u):,} min={u.min():.3f} median={np.median(u):.3f} max={u.max():.3f}")


if __name__ == "__main__":
    main()
