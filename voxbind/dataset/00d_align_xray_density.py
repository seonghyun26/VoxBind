"""00d_align_xray_density.py — Frame-corrected X-ray density crops for CrossDocked2020

Replaces the 00a (download) + 00b (crop) pair with a single PDB-grouped pass that
fixes the coordinate-frame bug in the original pipeline.

The bug
-------
CrossDocked2020 superposes every receptor of a binding pocket onto a shared
reference frame, which differs from the deposited crystallographic frame the
PDBe-EDS CCP4 map lives in (often by a crystal symmetry operation). The original
_crop_density samples the map at CrossDocked-frame coordinates and therefore
crops the wrong region — receptor atoms land at ~+0.06 sigma density (noise)
instead of ~+1.4 sigma.

The fix
-------
Per structure, recover the rigid transform CrossDocked-frame -> deposited-frame
by residue-matched Kabsch against the deposited PDB, then crop the map along the
transformed grid: crossdocked_xray._crop_density(..., transform=(R, t)).

A crop is DROPPED (marked unavailable -> model falls back to zero density) when:
  * no PDBe-EDS map for the PDB ID            (reason: no_eds_map)
  * no deposited PDB on PDBe                  (reason: no_deposited_pdb)
  * too few residue-matched atoms             (reason: too_few_matched_atoms)
  * post-fit Kabsch RMSD above --max-rmsd     (reason: high_rmsd)
  * weak density at corrected atoms           (reason: weak_density)

Outputs (--out-dir, default dataset/data/xray_crops_aligned; the original
dataset/data/xray_crops is left untouched)
-----------------------------------------------------------------------
    {split}/{idx:06d}.npy     float16 (64,64,64) frame-corrected crop
    {split}_available.npy     bool (N,)  True where a trustworthy crop was written
    {split}_transforms.npz    R,t,rmsd,n_match,rot_deg,density,ok per sample
    {split}_stats.csv         human-readable per-sample log

Usage
-----
    cd voxbind
    python dataset/00d_align_xray_density.py --splits test           # validate
    python dataset/00d_align_xray_density.py --splits train test     # full run
"""

import argparse
import csv
import os
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import gemmi
import numpy as np
import requests
import scipy.ndimage
import torch

# make the `voxbind` package importable, then reuse the crop helpers
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from voxbind.dataset.crossdocked_xray import _crop_density, _load_grid

REPO    = Path(__file__).resolve().parents[2]
VOXDATA = REPO / "voxbind" / "dataset" / "data"

MAP_URL = "https://www.ebi.ac.uk/pdbe/entry-files/{pid}.ccp4"
PDB_URL = "https://www.ebi.ac.uk/pdbe/entry-files/download/pdb{pid}.ent"


# ── data loading (mirrors DatasetCrossDockedXray / 00b exactly) ─────────────────

def load_split(data_dir, split, max_len):
    if split in ("train", "val"):
        data = torch.load(os.path.join(data_dir, "data_train.pt"), weights_only=False)
        random.Random(1234).shuffle(data)
        val_sz = 100
        data = data[: len(data) - val_sz] if split == "train" else data[len(data) - val_sz :]
    else:
        data = torch.load(os.path.join(data_dir, "data_test.pt"), weights_only=False)
    return [(p, l) for p, l in data if l["max_len"] <= max_len]


def get_pdb_id(pocket_id):
    return pocket_id.split("/")[1].split("_")[0].lower()


def get_centroid(ligand_):
    mask = ligand_["atoms_channel"] < 7
    return ligand_["coords"][mask].float().mean(dim=0).numpy().astype(np.float64)


# ── geometry helpers ────────────────────────────────────────────────────────────

def kabsch(P, Q):
    """Rigid transform mapping P onto Q. Returns (R, t, rmsd, rot_deg) with
    P @ R.T + t ~= Q."""
    P0, Q0 = P.mean(0), Q.mean(0)
    U, S, Vt = np.linalg.svd((P - P0).T @ (Q - Q0))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = Q0 - P0 @ R.T
    rmsd = float(np.sqrt((((P @ R.T + t) - Q) ** 2).sum(1).mean()))
    rot = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))
    return R, t, rmsd, rot


def sample_map(arr, frac_T, nu, nv, nw, xyz):
    """Trilinear-sample the z-scored map at Cartesian-Angstrom points."""
    fr = np.asarray(xyz, float) @ frac_T
    idx = [(fr[:, 0] * nu) % nu, (fr[:, 1] * nv) % nv, (fr[:, 2] * nw) % nw]
    return scipy.ndimage.map_coordinates(arr, idx, order=1, mode="wrap")


def atoms_by_key(structure, per_chain):
    """Heavy protein atoms keyed by (residue number, atom name).

    per_chain=True  -> {chain_name: {key: xyz}}   (deposited structure)
    per_chain=False -> {key: xyz}                 (CrossDocked pocket10 file)

    No altloc filtering: gemmi reports '\\x00' (not '') for atoms with no
    alternate conformation, so an explicit altloc whitelist silently drops
    every such atom. Alt-conf atoms simply overwrite in the dict — harmless
    for the rigid Kabsch fit (the alternate position is ~0.5 A away).
    """
    out = defaultdict(dict) if per_chain else {}
    for chain in structure[0]:
        for res in chain:
            if res.is_water() or res.het_flag == "H":
                continue
            for a in res:
                if a.is_hydrogen():
                    continue
                key = (res.seqid.num, a.name)
                pos = [a.pos.x, a.pos.y, a.pos.z]
                if per_chain:
                    out[chain.name][key] = pos
                else:
                    out[key] = pos
    return out


# ── downloads ───────────────────────────────────────────────────────────────────

def download(url, dest, retries=3):
    """Resumable download; returns True if dest holds a non-empty file."""
    if dest.exists() and dest.stat().st_size > 0:
        return True
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=90, stream=True)
            if r.status_code == 404:
                return False
            r.raise_for_status()
            tmp = dest.with_suffix(dest.suffix + ".tmp")
            with tmp.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
            tmp.rename(dest)
            return True
        except requests.RequestException:
            time.sleep(2.0 * (attempt + 1))
    return False


# ── per-PDB worker ──────────────────────────────────────────────────────────────

def process_pdb(pdb_id, jobs, cfg):
    """Download map + deposited PDB once, then align & crop every sample of this
    PDB. jobs = list of (split, idx, pocket10_path, centroid). Returns a list of
    per-sample result dicts."""
    ccp4_dir, pdb_dir, out_dir = cfg["ccp4"], cfg["pdb"], cfg["out"]
    overwrite = cfg["overwrite"]

    def blank(split, idx):
        return dict(split=split, idx=idx, ok=False, reason="", n_match=0,
                    rmsd=-1.0, rot_deg=-1.0, density=-9.0,
                    R=np.eye(3), t=np.zeros(3))

    def fail_all(reason):
        out = []
        for split, idx, _, _ in jobs:
            r = blank(split, idx); r["reason"] = reason; out.append(r)
        return out

    try:
        ccp4 = ccp4_dir / f"{pdb_id}.ccp4"
        if not download(MAP_URL.format(pid=pdb_id), ccp4):
            return fail_all("no_eds_map")
        grid = _load_grid(ccp4)
        if grid is None:
            return fail_all("map_load_failed")
        arr, frac_T, nu, nv, nw = grid

        pdbf = pdb_dir / f"{pdb_id}.pdb"
        if not download(PDB_URL.format(pid=pdb_id), pdbf):
            return fail_all("no_deposited_pdb")
        try:
            dep = atoms_by_key(gemmi.read_structure(str(pdbf)), per_chain=True)
        except Exception:
            return fail_all("deposited_parse_failed")
        if not dep:
            return fail_all("deposited_empty")

        results = []
        for split, idx, p10_path, centroid in jobs:
            r = blank(split, idx)
            try:
                pk = atoms_by_key(gemmi.read_structure(p10_path), per_chain=False)
            except Exception:
                r["reason"] = "pocket10_parse_failed"; results.append(r); continue

            # match the receptor pocket against each deposited chain, keep the
            # lowest-RMSD fit (handles homo-oligomers / chain renaming)
            best = None
            for datoms in dep.values():
                keys = [k for k in pk if k in datoms]
                if len(keys) < cfg["min_atoms"]:
                    continue
                P = np.array([pk[k] for k in keys], float)
                Q = np.array([datoms[k] for k in keys], float)
                R, t, rmsd, rot = kabsch(P, Q)
                if best is None or rmsd < best["rmsd"]:
                    best = dict(R=R, t=t, rmsd=rmsd, rot=rot, n=len(keys), P=P)
            if best is None:
                r["reason"] = "too_few_matched_atoms"; results.append(r); continue

            r.update(n_match=best["n"], rmsd=best["rmsd"], rot_deg=best["rot"],
                     R=best["R"], t=best["t"])

            # Density at the corrected receptor atoms is the physical
            # correctness check; RMSD is only a looser structural sanity bound,
            # so density is gated first.
            dens = float(sample_map(arr, frac_T, nu, nv, nw,
                                    best["P"] @ best["R"].T + best["t"]).mean())
            r["density"] = dens
            if dens < cfg["min_density"]:
                r["reason"] = "weak_density"; results.append(r); continue
            if best["rmsd"] > cfg["max_rmsd"]:
                r["reason"] = "high_rmsd"; results.append(r); continue

            out_path = out_dir / split / f"{idx:06d}.npy"
            if overwrite or not out_path.exists():
                crop = _crop_density(arr, frac_T, nu, nv, nw,
                                     np.asarray(centroid, float),
                                     transform=(best["R"], best["t"]))
                np.save(str(out_path), crop.astype(np.float16))
            r["ok"] = True
            r["reason"] = "ok"
            results.append(r)
        return results
    except Exception as e:  # never let one PDB kill the pool
        return fail_all(f"error:{type(e).__name__}")


# ── per-split driver ────────────────────────────────────────────────────────────

def run_split(split, args, cfg):
    data = load_split(args.data_dir, split, args.max_len)
    n = len(data)
    (Path(args.out_dir) / split).mkdir(parents=True, exist_ok=True)

    jobs = defaultdict(list)
    for i, (pocket_, ligand_) in enumerate(data):
        p10 = str(Path(args.pocket_dir) / pocket_["id"])
        jobs[get_pdb_id(pocket_["id"])].append((split, i, p10, get_centroid(ligand_)))

    pdb_ids = list(jobs)
    if args.limit:
        pdb_ids = pdb_ids[: args.limit]
    print(f"\n[{split}] {n:,} samples | {len(jobs):,} unique PDB IDs"
          f"{f' (limited to {len(pdb_ids)})' if args.limit else ''}", flush=True)

    avail   = np.zeros(n, dtype=bool)
    Rs      = np.tile(np.eye(3), (n, 1, 1))
    ts      = np.zeros((n, 3))
    rmsd    = np.full(n, -1.0)
    rot     = np.full(n, -1.0)
    dens    = np.full(n, -9.0)
    nmatch  = np.zeros(n, dtype=np.int32)
    reasons = ["unprocessed"] * n

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_pdb, pid, jobs[pid], cfg): pid for pid in pdb_ids}
        done = 0
        for fut in as_completed(futs):
            for r in fut.result():
                i = r["idx"]
                avail[i]   = r["ok"]
                reasons[i] = r["reason"]
                rmsd[i]    = r["rmsd"]
                rot[i]     = r["rot_deg"]
                dens[i]    = r["density"]
                nmatch[i]  = r["n_match"]
                Rs[i]      = r["R"]
                ts[i]      = r["t"]
            done += 1
            if done % 250 == 0 or done == len(pdb_ids):
                el = time.time() - t0
                print(f"[{split}] {done:,}/{len(pdb_ids):,} PDBs | "
                      f"ok {int(avail.sum()):,}/{n:,} | {el:.0f}s", flush=True)

    out = Path(args.out_dir)
    np.save(out / f"{split}_available.npy", avail)
    np.savez(out / f"{split}_transforms.npz", R=Rs, t=ts, rmsd=rmsd,
             rot_deg=rot, density=dens, n_match=nmatch, ok=avail)
    with (out / f"{split}_stats.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "ok", "reason", "n_match", "rmsd", "rot_deg", "density"])
        for i in range(n):
            w.writerow([i, int(avail[i]), reasons[i], int(nmatch[i]),
                        f"{rmsd[i]:.3f}", f"{rot[i]:.2f}", f"{dens[i]:.3f}"])

    ok = int(avail.sum())
    print(f"[{split}] DONE: {ok:,}/{n:,} crops written ({100*ok/max(n,1):.1f}%)")
    for reason, c in Counter(reasons).most_common():
        print(f"    {reason:24s} {c:,}")
    al = dens[avail]
    if len(al):
        print(f"    aligned-crop density: mean {al.mean():+.2f} sigma  "
              f"min {al.min():+.2f}  median {np.median(al):+.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir",   default=str(VOXDATA))
    ap.add_argument("--ccp4-dir",   default=str(VOXDATA / "ccp4"))
    ap.add_argument("--pdb-dir",    default=str(VOXDATA / "pdb"))
    ap.add_argument("--pocket-dir", default=str(VOXDATA / "crossdocked_pocket10"))
    ap.add_argument("--out-dir",    default=str(VOXDATA / "xray_crops_aligned"))
    ap.add_argument("--splits", nargs="+", default=["train", "test"],
                    choices=["train", "val", "test"])
    ap.add_argument("--max-len", type=int, default=30,
                    help="ligand size filter; must match DatasetCrossDockedXray.max_len")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--min-atoms",   type=int,   default=30,
                    help="min residue-matched atoms for a valid Kabsch fit")
    ap.add_argument("--max-rmsd",    type=float, default=10.0,
                    help="loose backstop on post-fit Kabsch RMSD (A); the density "
                         "gate below is the primary correctness check")
    ap.add_argument("--min-density", type=float, default=0.5,
                    help="min mean density (sigma) at corrected receptor atoms — "
                         "the primary gate (correct crops sit at ~+2.3 sigma)")
    ap.add_argument("--overwrite", action="store_true",
                    help="recompute crops even if the .npy already exists")
    ap.add_argument("--limit", type=int, default=0,
                    help="process only the first N PDB IDs per split (debug)")
    args = ap.parse_args()

    for d in (args.ccp4_dir, args.pdb_dir, args.out_dir):
        Path(d).mkdir(parents=True, exist_ok=True)

    cfg = dict(ccp4=Path(args.ccp4_dir), pdb=Path(args.pdb_dir),
               out=Path(args.out_dir), overwrite=args.overwrite,
               min_atoms=args.min_atoms, max_rmsd=args.max_rmsd,
               min_density=args.min_density)

    print("=== 00d frame-corrected X-ray density crops ===")
    print(f"  data_dir   : {args.data_dir}")
    print(f"  out_dir    : {args.out_dir}")
    print(f"  gates      : min_atoms={args.min_atoms} max_rmsd={args.max_rmsd} "
          f"min_density={args.min_density}")
    print(f"  workers    : {args.workers}")

    for split in args.splits:
        run_split(split, args, cfg)
    print("\nDone.")


if __name__ == "__main__":
    main()
