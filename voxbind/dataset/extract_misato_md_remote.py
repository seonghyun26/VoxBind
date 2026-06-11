"""extract_misato_md_remote.py — pull small MD targets straight from the REMOTE HDF5.

The full MD.hdf5 (123 GiB) is Zenodo-throttled to ~0.4 MB/s (~days). But pose_rmsd
(frames_rmsd_ligand) and interaction_energy (frames_interaction_energy) are tiny
per-frame arrays, so we read just those via HTTP byte-range (h5py over fsspec) for
the probe-pool pids only — no full download needed. Reads are latency-bound, so a
process Pool parallelises them: ~13 s/group sequential → ~16 workers ≈ <1 h for 3.6k.

64 KiB blocks are essential: 1 MiB blocks over-fetch the 16k-group link metadata
(150 s warmup); 64 KiB drops it to ~9 s and steady-state to ~7-23 s/group.

Run with an h5py+fsspec env, with aria2 paused to free the bandwidth:
    pkill -STOP -x aria2c
    /home/shpark/.conda/envs/cellmaes/bin/python -u dataset/extract_misato_md_remote.py
    pkill -CONT -x aria2c
  → dataset/data/misato/misato_md.json  {pid: {pose_rmsd, interaction_energy}}
"""
import argparse
import json
import time
from pathlib import Path
from multiprocessing import Pool

import fsspec
import h5py

URL = "https://zenodo.org/record/7711953/files/MD.hdf5"
BLOCK = 65536
_H5 = None


def _open():
    return h5py.File(fsspec.open(URL, block_size=BLOCK).open(), "r")


def _init():
    global _H5
    _H5 = _open()


def _read_one(pid_upper):
    """(pid_lower, pose_rmsd|None, interaction_energy|None). Reopens handle on transient errors."""
    global _H5
    for _ in range(3):
        try:
            g = _H5[pid_upper]
            pose = float(g["frames_rmsd_ligand"][:].mean()) if "frames_rmsd_ligand" in g else None
            ie = float(g["frames_interaction_energy"][:].mean()) if "frames_interaction_energy" in g else None
            return pid_upper.lower(), pose, ie
        except KeyError:
            return pid_upper.lower(), None, None          # complex not in MD subset
        except Exception:
            try:
                _H5 = _open()                             # reconnect on dropped stream
            except Exception:
                pass
            time.sleep(2)
    return pid_upper.lower(), None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pids_file", default="dataset/data/misato/pool_pids.txt")
    ap.add_argument("--out", default="dataset/data/misato/misato_md.json")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    pids = [p.strip().upper() for p in open(args.pids_file) if p.strip()]
    print(f"pids: {len(pids)}  workers: {args.workers}", flush=True)

    out, done, n_ok, t0 = {}, 0, 0, time.time()
    with Pool(args.workers, initializer=_init) as pool:
        for pid_l, pose, ie in pool.imap_unordered(_read_one, pids, chunksize=1):
            done += 1
            rec = {}
            if pose is not None:
                rec["pose_rmsd"] = pose
            if ie is not None:
                rec["interaction_energy"] = ie
            if rec:
                out[pid_l] = rec
                n_ok += 1
            if done % 100 == 0:
                el = time.time() - t0
                rate = done / el if el else 0
                eta = (len(pids) - done) / rate / 60 if rate else 0
                print(f"  {done}/{len(pids)}  ok={n_ok}  {rate:.1f}/s  ETA {eta:.0f}m", flush=True)

    Path(args.out).write_text(json.dumps(out, indent=0))
    import statistics as st
    print(f"wrote {len(out)} records -> {args.out}  ({time.time()-t0:.0f}s)", flush=True)
    for f in ("pose_rmsd", "interaction_energy"):
        v = [r[f] for r in out.values() if f in r]
        if v:
            print(f"  {f}: n={len(v)} min={min(v):.2f} median={st.median(v):.2f} max={max(v):.2f}", flush=True)


if __name__ == "__main__":
    main()
