"""extract_misato_md_local.py — MD targets from the LOCAL MD.hdf5 (after full download).

For the probe-pool pids, per complex:
    pose_rmsd          mean of frames_rmsd_ligand          (ligand pose stability over MD)
    interaction_energy mean of frames_interaction_energy   (MM protein-ligand energy)
    rmsf               pocket-mean per-atom RMSF from trajectory_coordinates (flexibility)

Pocket = protein atoms within 6 Å of any ligand atom on frame 0; ligand = last molecule
(molecules_begin_atom_index[-1]:). RMSF = sqrt(mean_t |r_t - <r>|^2) per atom. NOTE: MISATO's
"adaptability" is its (RMSF-equivalent) flexibility measure, NOT stored in the Zenodo base
file — RMSF stands in for it here.

    /home/shpark/.conda/envs/cellmaes/bin/python -u dataset/extract_misato_md_local.py
  → dataset/data/misato/misato_md.json
"""
import json
import time
from multiprocessing import Pool

import h5py
import numpy as np

MD_PATH = "dataset/data/misato/MD.hdf5"
PIDS_FILE = "dataset/data/misato/pool_pids.txt"
OUT = "dataset/data/misato/misato_md.json"
POCKET_CUT = 6.0
_H5 = None


def _init():
    global _H5
    _H5 = h5py.File(MD_PATH, "r")


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


def main():
    pids = [p.strip().upper() for p in open(PIDS_FILE) if p.strip()]
    print(f"pids: {len(pids)}  (local MD.hdf5)", flush=True)
    out, done, n_ok, t0 = {}, 0, 0, time.time()
    with Pool(8, initializer=_init) as pool:
        for pid_l, rec in pool.imap_unordered(_one, pids, chunksize=8):
            done += 1
            if rec and "_err" not in rec:
                out[pid_l] = rec
                n_ok += 1
            if done % 500 == 0:
                el = time.time() - t0
                print(f"  {done}/{len(pids)}  ok={n_ok}  {done/el:.0f}/s", flush=True)
    json.dump(out, open(OUT, "w"), indent=0)
    print(f"wrote {len(out)} -> {OUT}  ({time.time()-t0:.0f}s)", flush=True)
    import statistics as st
    for f in ("pose_rmsd", "interaction_energy", "rmsf"):
        v = [r[f] for r in out.values() if f in r]
        if v:
            print(f"  {f}: n={len(v)} min={min(v):.2f} median={st.median(v):.2f} max={max(v):.2f}", flush=True)


if __name__ == "__main__":
    main()
