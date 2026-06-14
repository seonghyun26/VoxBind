"""02b_misato_targets.py — MISATO HDF5 → compact pdb_id→value JSON targets.

Stage 02b. Isolates the HDF5 read (voxbind's env has no h5py) into a small
extraction so the probe loads plain JSON. Run with an h5py interpreter, e.g.
/home/shpark/.conda/envs/cellmaes/bin/python. Keys are lowercased PDB IDs (MISATO
stores them uppercase) so they join the LP/MISATO splits directly.

  qm  — per ligand from QM.hdf5 (fast):
          partial_charge = mean|gfn2_charge| over ligand atoms; plus mol-level
          electron_affinity / hardness / ionization_potential / electronegativity / …
        → misato_qm.json   (~19.4k complexes)

  md  — per complex from MD.hdf5 (parallel; reads the 100-frame trajectory):
          pose_rmsd          = mean of frames_rmsd_ligand          (Å)
          interaction_energy = mean of frames_interaction_energy   (kcal/mol)
          rmsf               = pocket-mean per-atom RMSF from trajectory_coordinates (Å)
        pocket = protein atoms within 6 Å of any ligand atom on frame 0; ligand =
        last molecule (molecules_begin_atom_index[-1]:). RMSF stands in for MISATO's
        "adaptability" (not stored in the base Zenodo MD.hdf5). Frame-invariant
        scalars → no alignment to the voxel crop needed.
        → misato_md.json   (over --pids, default pool_pids.txt)

    cellmaes/python dataset/02b_misato_targets.py qm
    cellmaes/python dataset/02b_misato_targets.py md --pids data/misato/pool_pids.txt --procs 8
"""
import argparse
import json
import math
import time
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
MISATO = HERE / "data" / "misato"

CHARGE_PROP = "gfn2_charge"
MOL_SCALARS = ["Electron_Affinity", "Hardness", "Ionization_Potential",
               "Electronegativity", "molecular_weight", "total_charge"]
POCKET_CUT = 6.0


def finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


# ── QM ─────────────────────────────────────────────────────────────────────────
def extract_qm(qm_path: Path) -> dict:
    out = {}
    with h5py.File(qm_path, "r") as f:
        for pid in f.keys():
            g = f[pid]
            try:
                names = [s.decode() if isinstance(s, bytes) else s
                         for s in g["atom_properties/atom_properties_names"][:]]
                ci = names.index(CHARGE_PROP)
                q = g["atom_properties/atom_properties_values"][:, ci].astype(np.float64)
                rec = {"partial_charge": float(np.abs(q).mean())}
                for s in MOL_SCALARS:
                    key = f"mol_properties/{s}"
                    if key in g:
                        rec[s.lower()] = float(np.asarray(g[key][()]))
                out[pid.lower()] = rec
            except Exception as e:
                print(f"  [skip] {pid}: {e!r}")
    return out


# ── MD (parallel) ────────────────────────────────────────────────────────────
_H5 = None
_MD_PATH = None


def _init():
    global _H5
    _H5 = h5py.File(_MD_PATH, "r")


def _one(pid_up):
    g = _H5.get(pid_up)
    if g is None:
        return pid_up.lower(), None
    try:
        xyz = g["trajectory_coordinates"][:]                 # (T, N, 3)
        ls = int(g["molecules_begin_atom_index"][-1])        # ligand start
        f0 = xyz[0]
        d = np.sqrt(((f0[:ls, None, :] - f0[None, ls:, :]) ** 2).sum(-1)).min(1)
        pocket = np.where(d <= POCKET_CUT)[0]
        if pocket.size == 0:
            pocket = np.arange(ls)
        rmsf = np.sqrt(((xyz - xyz.mean(0, keepdims=True)) ** 2).sum(-1).mean(0))  # (N,)
        return pid_up.lower(), {
            "pose_rmsd":          float(g["frames_rmsd_ligand"][:].mean()),
            "interaction_energy": float(g["frames_interaction_energy"][:].mean()),
            "rmsf":               float(rmsf[pocket].mean()),
        }
    except Exception as e:
        return pid_up.lower(), {"_err": repr(e)[:80]}


def extract_md(md_path: Path, pids_file: Path, procs: int) -> dict:
    global _MD_PATH
    _MD_PATH = str(md_path)
    pids = [p.strip().upper() for p in open(pids_file) if p.strip()]
    print(f"  pids: {len(pids)}  procs={procs}", flush=True)
    out, done, n_ok, t0 = {}, 0, 0, time.time()
    with Pool(procs, initializer=_init) as pool:
        for pid_l, rec in pool.imap_unordered(_one, pids, chunksize=8):
            done += 1
            if rec and "_err" not in rec:
                out[pid_l] = rec; n_ok += 1
            if done % 500 == 0:
                print(f"  {done}/{len(pids)}  ok={n_ok}  {done/(time.time()-t0):.0f}/s", flush=True)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pq = sub.add_parser("qm", help="QM.hdf5 → misato_qm.json")
    pq.add_argument("--misato_dir", type=Path, default=MISATO)
    pq.add_argument("--out", type=Path, default=None)
    pq.set_defaults(field="partial_charge")

    pm = sub.add_parser("md", help="MD.hdf5 → misato_md.json (parallel)")
    pm.add_argument("--misato_dir", type=Path, default=MISATO)
    pm.add_argument("--pids", type=Path, default=MISATO / "pool_pids.txt")
    pm.add_argument("--procs", type=int, default=8)
    pm.add_argument("--out", type=Path, default=None)
    pm.set_defaults(field="rmsf")

    args = p.parse_args()
    if args.cmd == "qm":
        src = args.misato_dir / "QM.hdf5"
        out = args.out or (args.misato_dir / "misato_qm.json")
        print(f"=== MISATO QM targets ===\n  in : {src}\n  out: {out}")
        rec = extract_qm(src)
    else:
        src = args.misato_dir / "MD.hdf5"
        out = args.out or (args.misato_dir / "misato_md.json")
        print(f"=== MISATO MD targets ===\n  in : {src}\n  out: {out}")
        rec = extract_md(src, args.pids, args.procs)

    out.write_text(json.dumps(rec, indent=0))
    usable = [r[args.field] for r in rec.values() if args.field in r and finite(r[args.field])]
    print(f"  wrote {len(rec):,} records → {out}  ({out.stat().st_size/1e6:.1f} MB)")
    if usable:
        u = np.array(usable)
        print(f"  {args.field}: n={len(u):,} min={u.min():.3f} median={np.median(u):.3f} max={u.max():.3f}")


if __name__ == "__main__":
    main()
