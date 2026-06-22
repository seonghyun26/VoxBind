"""00b_density_preprocess.py — Frame-aligned X-ray density crops for CrossDocked2020.

Single entry point for every density-crop *version*. All versions share the
same frame-corrected (Kabsch-aligned) 64³ × 0.25 Å crop at the ligand centroid;
they differ only in the normalisation applied to the voxel values — that is the
`--version` if-branch:

    v1  pretrain/xray_crops_aligned      per-MAP z-score, ±3σ clip applied at LOAD time
                                (dset.normalize=true). The aligned crop itself.
    v2  pretrain/xray_crops_aligned_v2   pocket-pool z-score        x' = (x − μ)   / σ
    v3  pretrain/xray_crops_aligned_v3   pocket-pool symmetric      x' =  x        / max_abs   → [−1, +1]
    v4  pretrain/xray_crops_aligned_v4   pocket-pool clip+z-score   x' = (clip(x) − μ_c) / σ_c
    v5  pretrain/xray_crops_aligned_v5   pocket-pool arcsinh+z      x' = (arcsinh(x/s) − μ_a) / σ_a
    v6  pretrain/xray_crops_aligned_v6   v5 arcsinh+z, but availability restricted to LIGAND-MATCHED
                                poses (--native_filter). The receptor's 2Fo-Fc map only
                                matches the modeled ligand when that ligand is the receptor's
                                own crystal ligand; v6 keeps only those (default tt_min: native
                                ligand in its crystal pose). Must be generated alone.

v2/v3/v4/v5/v6 are stored pre-normalised → load them with dset.normalize=false.
v2/v3 (and v4's clip / v5's arcsinh stats) are derived from a SINGLE pocket pool, so when
several are requested in one run they share one cropping pass and one stat pass.

This file consolidates the former 00d (align + v1), 00e (v2/v3) and 00f (v4).

Pipeline (per split)
--------------------
  1. ALIGN   per-PDB residue-matched Kabsch CrossDocked-frame → crystal-frame.
             Cached as pretrain/xray_crops_aligned/{split}_transforms.npz; reused on
             subsequent runs unless --force_realign. Reads {pdb}.ccp4 + {pdb}.pdb
             produced by dataset/00a_density_download.py (no downloading here).
  2. CROP    per-PDB, using the transforms: v1 → z-scored crop written directly;
             v2/v3/v4 → raw crop written to scratch + pool stats accumulated.
  3. NORM    (only if v2/v3/v4 requested) compute pool stats, then write each
             requested version's normalised crops + stats.json.

A crop is DROPPED (available=False → model falls back to zero density) when the
PDB has no map / no deposited PDB, too few residue-matched atoms, post-fit RMSD
above --max_rmsd, or weak density at the corrected receptor atoms. v6 additionally
drops every pose whose modeled ligand is not the receptor's own crystal ligand
(see --native_filter), so its density genuinely corresponds to the ligand.

Usage
-----
    cd voxbind
    python dataset/00b_density_preprocess.py --version v3                 # default
    python dataset/00b_density_preprocess.py --version v1 v2 v3 v4        # everything
    python dataset/00b_density_preprocess.py --version v3 --splits test   # validate on test
    python dataset/00b_density_preprocess.py --version v4 --keep_raw      # retain raw scratch
    python dataset/00b_density_preprocess.py --version v6 --save_gradmag  # ligand-matched (tt_min)

Outputs (per requested version dir)
-----------------------------------
    {split}/{idx:06d}.npy        float16 (64,64,64)
    {split}_available.npy        bool (N,)  True where a trustworthy crop exists
    stats.json                   normalisation constants (v2/v3/v4)
  and, in pretrain/xray_crops_aligned/ (always — the alignment artifacts):
    {split}_transforms.npz       R,t,rmsd,rot_deg,density,n_match,ok per sample
    {split}_stats.csv            human-readable per-sample alignment log
"""

import argparse
import csv
import os
import random
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# gemmi BEFORE torch — see feedback_gemmi_torch_import_order memory.
import gemmi
import json

import numpy as np
import scipy.ndimage
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from voxbind.dataset.crossdocked_xray import _crop_density, _load_grid


# ── Config ─────────────────────────────────────────────────────────────────────

VOXDATA = Path(__file__).resolve().parent / "data"
GRID_DIM = 64
RESOLUTION = 0.25

# version → output directory name (relative to --out_root). v1 is the aligned
# dir that also holds the shared transforms/stats artifacts.
DIR_NAME = {
    "v1": "pretrain/xray_crops_aligned",
    "v2": "pretrain/xray_crops_aligned_v2",
    "v3": "pretrain/xray_crops_aligned_v3",
    "v4": "pretrain/xray_crops_aligned_v4",
    "v5": "pretrain/xray_crops_aligned_v5",
    "v6": "pretrain/xray_crops_aligned_v6",
}

# Versions stored pre-normalised from a pooled raw-crop pass (load with
# dset.normalize=false). v1 is per-map z-scored at load time and excluded here.
RAW_VERSIONS = ("v2", "v3", "v4", "v5", "v6")

# v4 clip thresholds (raw 2Fo-Fc units), from the 200-sample voxel-pool quantile
# estimate in notebook/260527 — P0.1% / P99.9%. Asymmetric: the positive tail is
# atom-peak signal, the negative tail is noise floor.
CLIP_LO = -0.578
CLIP_HI = +1.647

# v5 arcsinh soft-squash scale: x' = z-score(arcsinh(x / s)). arcsinh is ~linear
# for |x|<<s and ~log for |x|>>s, so the bright-atom tail compresses smoothly with
# NO clip and no pile-up spike (raw +30 → ~+9 at s=0.5). Mirrors 01b's v5.
V5_ARCSINH_SCALE = 0.5


# ── Sample list (mirrors DatasetCrossDockedXray's load order) ───────────────────

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


# ── Ligand-provenance / density-match filter (v6) ───────────────────────────────
# CrossDocked2020 sample names encode where the modeled ligand came from:
#   {receptor}_{chain}_rec_{ligand-PDB}_{resname}_lig_[itN_]*tt_{min|docked}_{rank}
#   e.g. 4xe6_X_rec_3fqc_55v_lig_tt_docked_4  → ligand 55v (from 3fqc) cross-docked into 4xe6
#        5acc_A_rec_5acc_ke9_lig_tt_min_0     → 5acc's own ligand, energy-minimised (native)
# The 2Fo-Fc map belongs to the RECEPTOR crystal, so its ligand-region density only
# corresponds to the modeled ligand when that ligand IS the receptor's crystal ligand.

_PROV_RE = re.compile(
    r"([0-9a-zA-Z]{4})_\w+?_rec_([0-9a-zA-Z]{4})_\w+?_lig_(?:it\d+_)*tt_(min|docked)")


def parse_provenance(pocket_id):
    """(receptor_pdb, ligand_src_pdb, pose_type) or None if the name doesn't parse."""
    m = _PROV_RE.match(pocket_id.split("/")[-1])
    if not m:
        return None
    return m.group(1).lower(), m.group(2).lower(), m.group(3)


def is_native_mask(samples, mode):
    """Boolean mask over `samples` — True where the receptor's crystal density matches
    the modeled ligand.

      mode="tt_min"   : ligand source PDB == receptor PDB AND pose is the energy-minimised
                        crystal pose → identity AND position match the density (strictest).
      mode="same_pdb" : ligand source PDB == receptor PDB (identity match; a redocked pose
                        may drift from the crystal position).

    Unparsable names (→ None) are treated as non-native.
    """
    out = np.zeros(len(samples), dtype=bool)
    for i, (pocket_, _ligand_) in enumerate(samples):
        prov = parse_provenance(pocket_["id"])
        if prov is None:
            continue
        rec, src, pose = prov
        if mode == "same_pdb":
            out[i] = src == rec
        else:  # tt_min
            out[i] = (src == rec) and (pose == "min")
    return out


# ── CCP4 raw load (NO per-map z-score; for v2/v3/v4 pool normalisation) ─────────

def load_raw_grid(ccp4_path):
    """(arr_raw, frac_T, nu, nv, nw) or None — RAW CCP4 voxel values."""
    try:
        m = gemmi.read_ccp4_map(str(ccp4_path))
        m.setup(float("nan"))
        grid = m.grid
        cell = grid.unit_cell
        arr = np.array(grid, dtype=np.float32)
        nu, nv, nw = arr.shape
        if not np.isfinite(arr).all():
            return None
        orth = np.array(cell.orth.mat.tolist())
        frac_M = np.linalg.inv(orth)
        return arr, frac_M.T, nu, nv, nw
    except Exception:
        return None


# ── Geometry / alignment helpers ────────────────────────────────────────────────

def kabsch(P, Q):
    """Rigid transform mapping P onto Q (P @ R.T + t ≈ Q). Returns (R, t, rmsd, rot_deg)."""
    P0, Q0 = P.mean(0), Q.mean(0)
    U, S, Vt = np.linalg.svd((P - P0).T @ (Q - Q0))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = Q0 - P0 @ R.T
    rmsd = float(np.sqrt((((P @ R.T + t) - Q) ** 2).sum(1).mean()))
    rot = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))
    return R, t, rmsd, rot


def sample_map(arr, frac_T, nu, nv, nw, xyz):
    """Trilinear-sample the (z-scored) map at Cartesian-Angstrom points."""
    fr = np.asarray(xyz, float) @ frac_T
    idx = [(fr[:, 0] * nu) % nu, (fr[:, 1] * nv) % nv, (fr[:, 2] * nw) % nw]
    return scipy.ndimage.map_coordinates(arr, idx, order=1, mode="wrap")


def kabsch_match(pk, dep, cfg, density_fn, criterion="density"):
    """Best chain-Kabsch fit mapping pocket10 atoms to the deposited frame.

    criterion="rmsd"    : lowest Kabsch RMSD (legacy chain selection).
    criterion="density" : highest mean density at the corrected atoms (default;
                          robust for interface/heterogeneous pockets where a
                          low-RMSD subset fit can still land pocket10 in noise).
    """
    best = None
    for chain, datoms in dep.items():
        keys = [k for k in pk if k in datoms]
        if len(keys) < cfg["min_atoms"]:
            continue
        P = np.array([pk[k] for k in keys], float)
        Q = np.array([datoms[k] for k in keys], float)
        R, t, rmsd, rot = kabsch(P, Q)
        cand = dict(R=R, t=t, rmsd=rmsd, rot=rot, n=len(keys), P=P,
                    chain=chain, density=None)
        if criterion == "density":
            cand["density"] = float(density_fn(P @ R.T + t))
            if best is None or cand["density"] > best["density"]:
                best = cand
        else:  # rmsd
            if best is None or cand["rmsd"] < best["rmsd"]:
                best = cand
    return best


def atoms_by_key(structure, per_chain):
    """Heavy protein atoms keyed by (residue number, atom name).

    per_chain=True  → {chain_name: {key: xyz}}   (deposited structure)
    per_chain=False → {key: xyz}                 (CrossDocked pocket10 file)
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


# ── Stage 1: alignment worker (transforms only; reads map+pdb from disk) ─────────

def align_pdb(pdb_id, jobs, cfg):
    """Kabsch frame transform for every sample of one PDB.

    jobs = [(split, idx, pocket10_path, centroid), ...].
    Returns a list of per-sample result dicts (no .npy written here)."""
    ccp4_dir, pdb_dir = cfg["ccp4"], cfg["pdb"]

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
        if not (ccp4.exists() and ccp4.stat().st_size > 0):
            return fail_all("no_eds_map")
        grid = _load_grid(ccp4)
        if grid is None:
            return fail_all("map_load_failed")
        arr, frac_T, nu, nv, nw = grid

        pdbf = pdb_dir / f"{pdb_id}.pdb"
        if not (pdbf.exists() and pdbf.stat().st_size > 0):
            return fail_all("no_deposited_pdb")
        try:
            dep = atoms_by_key(gemmi.read_structure(str(pdbf)), per_chain=True)
        except Exception:
            return fail_all("deposited_parse_failed")
        if not dep:
            return fail_all("deposited_empty")

        density_fn = lambda xyz: sample_map(arr, frac_T, nu, nv, nw, xyz).mean()
        results = []
        for split, idx, p10_path, centroid in jobs:
            r = blank(split, idx)
            try:
                pk = atoms_by_key(gemmi.read_structure(p10_path), per_chain=False)
            except Exception:
                r["reason"] = "pocket10_parse_failed"; results.append(r); continue

            best = kabsch_match(pk, dep, cfg, density_fn, criterion=cfg["chain_select"])
            if best is None:
                r["reason"] = "too_few_matched_atoms"; results.append(r); continue

            r.update(n_match=best["n"], rmsd=best["rmsd"], rot_deg=best["rot"],
                     R=best["R"], t=best["t"])

            dens = best["density"]
            if dens is None:
                dens = float(density_fn(best["P"] @ best["R"].T + best["t"]))
            r["density"] = dens
            if dens < cfg["min_density"]:
                r["reason"] = "weak_density"; results.append(r); continue
            if best["rmsd"] > cfg["max_rmsd"]:
                r["reason"] = "high_rmsd"; results.append(r); continue

            r["ok"] = True; r["reason"] = "ok"; results.append(r)
        return results
    except Exception as e:  # never let one PDB kill the pool
        return fail_all(f"error:{type(e).__name__}")


def run_alignment(split, samples, args, cfg, aligned_dir):
    """Compute + persist transforms for one split. Returns (R, t, ok) arrays."""
    n = len(samples)
    jobs = defaultdict(list)
    for i, (pocket_, ligand_) in enumerate(samples):
        p10 = str(Path(args.pocket_dir) / pocket_["id"])
        jobs[get_pdb_id(pocket_["id"])].append((split, i, p10, get_centroid(ligand_)))

    pdb_ids = list(jobs)
    if args.limit:
        pdb_ids = pdb_ids[: args.limit]
    print(f"\n[{split}] ALIGN  {n:,} samples | {len(jobs):,} unique PDBs"
          f"{f' (limited to {len(pdb_ids)})' if args.limit else ''}", flush=True)

    avail = np.zeros(n, dtype=bool)
    Rs = np.tile(np.eye(3), (n, 1, 1)); ts = np.zeros((n, 3))
    rmsd = np.full(n, -1.0); rot = np.full(n, -1.0); dens = np.full(n, -9.0)
    nmatch = np.zeros(n, dtype=np.int32); reasons = ["unprocessed"] * n

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(align_pdb, pid, jobs[pid], cfg): pid for pid in pdb_ids}
        done = 0
        for fut in as_completed(futs):
            for r in fut.result():
                i = r["idx"]
                avail[i] = r["ok"]; reasons[i] = r["reason"]
                rmsd[i] = r["rmsd"]; rot[i] = r["rot_deg"]; dens[i] = r["density"]
                nmatch[i] = r["n_match"]; Rs[i] = r["R"]; ts[i] = r["t"]
            done += 1
            if done % 250 == 0 or done == len(pdb_ids):
                print(f"[{split}] align {done:,}/{len(pdb_ids):,} | "
                      f"ok {int(avail.sum()):,}/{n:,} | {time.time() - t0:.0f}s", flush=True)

    aligned_dir.mkdir(parents=True, exist_ok=True)
    np.save(aligned_dir / f"{split}_available.npy", avail)
    np.savez(aligned_dir / f"{split}_transforms.npz", R=Rs, t=ts, rmsd=rmsd,
             rot_deg=rot, density=dens, n_match=nmatch, ok=avail)
    with (aligned_dir / f"{split}_stats.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "ok", "reason", "n_match", "rmsd", "rot_deg", "density"])
        for i in range(n):
            w.writerow([i, int(avail[i]), reasons[i], int(nmatch[i]),
                        f"{rmsd[i]:.3f}", f"{rot[i]:.2f}", f"{dens[i]:.3f}"])

    ok = int(avail.sum())
    print(f"[{split}] ALIGN done: {ok:,}/{n:,} crops trustworthy "
          f"({100 * ok / max(n, 1):.1f}%)")
    for reason, c in Counter(reasons).most_common():
        print(f"    {reason:24s} {c:,}")
    al = dens[avail]
    if len(al):
        print(f"    aligned-crop density: mean {al.mean():+.2f}σ  "
              f"min {al.min():+.2f}  median {np.median(al):+.2f}")
    return Rs, ts, avail


def ensure_transforms(split, samples, args, cfg, aligned_dir):
    """Load cached transforms if present (and shape-compatible), else compute."""
    tx_path = aligned_dir / f"{split}_transforms.npz"
    if tx_path.exists() and not args.force_realign:
        tx = np.load(tx_path)
        R, t, ok = tx["R"], tx["t"], tx["ok"]
        if len(ok) != len(samples):
            print(f"[error] cached transforms for {split} have {len(ok)} rows but the "
                  f"dataset has {len(samples)}. Re-run with --force_realign.")
            sys.exit(1)
        print(f"\n[{split}] ALIGN cached → {tx_path}  (ok {int(ok.sum()):,}/{len(ok):,})")
        return R, t, ok
    return run_alignment(split, samples, args, cfg, aligned_dir)


# ── Stage 2: crop worker (v1 z-scored crop direct; v2/v3/v4 raw crop + stats) ────

def crop_pdb(pdb_id, jobs, cfg):
    """Crop one PDB's available samples. jobs = [(split, idx, centroid, R, t), ...].

    Returns (partial_stats, n_written). Partial stats accumulate over this PDB's
    RAW crops in float64 (sum, sum_sq, n, min, max, plus clipped sum/sum_sq for v4)."""
    write_v1 = cfg["write_v1"]
    need_raw = cfg["need_raw"]; need_v4 = cfg["need_v4"]; need_v5 = cfg["need_v5"]
    aligned_dir, scratch_dir = cfg["aligned_dir"], cfg["scratch_dir"]
    overwrite = cfg["overwrite"]
    clip_lo, clip_hi = cfg["clip_lo"], cfg["clip_hi"]
    s5 = cfg["arcsinh_scale"]

    st = dict(sum=0.0, sum_sq=0.0, n=0, min=np.inf, max=-np.inf,
              csum=0.0, csum_sq=0.0, asum=0.0, asum_sq=0.0)
    written = 0

    ccp4 = cfg["ccp4"] / f"{pdb_id}.ccp4"
    zgrid = _load_grid(ccp4) if write_v1 else None
    rgrid = load_raw_grid(ccp4) if need_raw else None

    for split, idx, centroid, R, t in jobs:
        centroid = np.asarray(centroid, float)
        transform = (np.asarray(R, float), np.asarray(t, float))

        if write_v1 and zgrid is not None:
            arr_n, fT, nu, nv, nw = zgrid
            out = aligned_dir / split / f"{idx:06d}.npy"
            if overwrite or not out.exists():
                crop = _crop_density(arr_n, fT, nu, nv, nw, centroid,
                                     G=GRID_DIM, res=RESOLUTION, transform=transform)
                np.save(str(out), crop.astype(np.float16))

        if need_raw and rgrid is not None:
            arr, fT, nu, nv, nw = rgrid
            crop = _crop_density(arr, fT, nu, nv, nw, centroid,
                                 G=GRID_DIM, res=RESOLUTION, transform=transform)
            c64 = crop.astype(np.float64)
            st["sum"] += c64.sum(); st["sum_sq"] += (c64 * c64).sum(); st["n"] += c64.size
            st["min"] = min(st["min"], float(c64.min()))
            st["max"] = max(st["max"], float(c64.max()))
            if need_v4:
                cc = np.clip(c64, clip_lo, clip_hi)
                st["csum"] += cc.sum(); st["csum_sq"] += (cc * cc).sum()
            if need_v5:
                aa = np.arcsinh(c64 / s5)
                st["asum"] += aa.sum(); st["asum_sq"] += (aa * aa).sum()
            np.save(str(scratch_dir / split / f"{idx:06d}.npy"), crop.astype(np.float16))

        written += 1
    return st, written


def run_crop(split, samples, R, t, ok, cfg):
    """Per-PDB crop pass over available samples. Returns (pooled_stats, n_written)."""
    jobs = defaultdict(list)
    for i, (pocket_, ligand_) in enumerate(samples):
        if not ok[i]:
            continue
        jobs[get_pdb_id(pocket_["id"])].append(
            (split, i, get_centroid(ligand_), R[i], t[i]))

    pdb_ids = list(jobs)
    print(f"[{split}] CROP   {int(ok.sum()):,} available | {len(pdb_ids):,} PDBs", flush=True)

    partials, written = [], 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=cfg["workers"]) as ex:
        futs = {ex.submit(crop_pdb, pid, jobs[pid], cfg): pid for pid in pdb_ids}
        done = 0
        for fut in as_completed(futs):
            pid = futs[fut]
            stp, w = fut.result()
            partials.append((pid, stp)); written += w
            done += 1
            if done % 250 == 0 or done == len(pdb_ids):
                print(f"[{split}] crop {done:,}/{len(pdb_ids):,} | "
                      f"{written:,} crops | {time.time() - t0:.0f}s", flush=True)

    # Sum partials in sorted-PDB order so pool stats are independent of the
    # process completion order (float addition is not associative).
    total = dict(sum=0.0, sum_sq=0.0, n=0, min=np.inf, max=-np.inf,
                 csum=0.0, csum_sq=0.0, asum=0.0, asum_sq=0.0)
    for _, stp in sorted(partials, key=lambda x: x[0]):
        total["sum"] += stp["sum"]; total["sum_sq"] += stp["sum_sq"]; total["n"] += stp["n"]
        total["csum"] += stp["csum"]; total["csum_sq"] += stp["csum_sq"]
        total["asum"] += stp["asum"]; total["asum_sq"] += stp["asum_sq"]
        total["min"] = min(total["min"], stp["min"]); total["max"] = max(total["max"], stp["max"])
    print(f"[{split}] CROP done: {written:,} crops")
    return total, written


# ── Resample-manifest mode (prepare for on-the-fly train-time density aug) ──────

def run_resample(args, native_filter, aligned_dir, out_root):
    """Emit per-split manifests so density can be cropped from the full CCP4 map at the
    AUGMENTED pose at TRAIN time — no frozen crops written.

    Reuses the cached Kabsch alignment ({split}_transforms.npz, recomputed only if missing)
    and the normalization recipe from an existing precomputed version's stats.json
    (--resample_norm_version). Writes, under --resample_out, per split:
        {split}_manifest.npz   pdb_id, centroid, R, t, ok   (full-length N, positional —
                               row i aligns with crop index i / DatasetCrossDockedXray order)
        {split}_available.npy  bool (N,)
    plus a single resample.json carrying the grid + normalization recipe + ccp4 path so the
    train-time reader is self-describing.
    """
    norm_ver = args.resample_norm_version
    stats_path = out_root / DIR_NAME[norm_ver] / "stats.json"
    if not stats_path.exists():
        sys.exit(f"[error] resample needs the normalization recipe at {stats_path}. "
                 f"Run `00b … --version {norm_ver}` once first (it writes stats.json).")
    norm = json.loads(stats_path.read_text())

    ccp4 = Path(args.ccp4_dir)
    n_maps = len(list(ccp4.glob("*.ccp4"))) if ccp4.exists() else 0
    if n_maps == 0:
        sys.exit(f"[error] no .ccp4 maps in {ccp4}. Resample mode needs the FULL maps retained on "
                 f"disk (this corpus may have streamed+deleted them — re-acquire before resampling).")

    out_dir = out_root / args.resample_out
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(ccp4=ccp4, pdb=Path(args.pdb_dir),
               min_atoms=args.min_atoms, max_rmsd=args.max_rmsd, min_density=args.min_density,
               chain_select=args.chain_select, workers=args.workers)

    print("=== density RESAMPLE-manifest mode (on-the-fly aug prep) ===")
    print(f"  norm recipe  : {norm_ver}  ({norm.get('scheme')})")
    print(f"  ccp4_dir     : {ccp4}  ({n_maps:,} maps retained)")
    print(f"  out_dir      : {out_dir}")
    print(f"  splits       : {args.splits}  (max_len {args.max_len})")
    if native_filter != "none":
        print(f"  native_filter: {native_filter}  (availability restricted to ligand-matched poses)")

    for split in args.splits:
        samples = load_split(args.data_dir, split, args.max_len)
        R, t, ok = ensure_transforms(split, samples, args, cfg, aligned_dir)
        if native_filter != "none":
            nat = is_native_mask(samples, native_filter)
            ok = ok & nat
            print(f"[{split}] native_filter={native_filter}: {int(nat.sum()):,} matched | "
                  f"{int(ok.sum()):,} after ∩ density-available")
        pdb_ids = np.array([get_pdb_id(p["id"]) for p, _l in samples])
        centroids = (np.stack([get_centroid(l) for _p, l in samples]).astype(np.float32)
                     if samples else np.zeros((0, 3), np.float32))
        np.save(out_dir / f"{split}_available.npy", ok)
        np.savez(out_dir / f"{split}_manifest.npz",
                 pdb_id=pdb_ids, centroid=centroids,
                 R=R.astype(np.float32), t=t.astype(np.float32), ok=ok)
        print(f"[{split}] manifest: {int(ok.sum()):,} available / {len(samples):,} samples "
              f"→ {split}_manifest.npz")

    recipe = dict(
        kind="resample_manifest", grid_dim=GRID_DIM, resolution=RESOLUTION,
        ccp4_dir=str(ccp4.resolve()), ccp4_ext=".ccp4",
        frame="kabsch: transform=(R,t) maps crop(CrossDocked)-frame → map(deposited)-frame",
        norm_version=norm_ver,
        normalization={k: norm[k] for k in
                       ("scheme", "formula", "arcsinh_scale", "mu_a", "sigma_a",
                        "mu", "sigma", "max_abs", "clip_lo_raw", "clip_hi_raw",
                        "mu_clipped", "sigma_clipped") if k in norm},
        native_filter=native_filter,
        train_time_note=(
            "Per sample i where ok[i]: sample the RAW map (load_raw_grid, NOT z-scored _load_grid) at "
            "cart = (centroid[i] + R_aug.T @ (o - t_aug)) @ R[i].T + t[i], o = 64³ box offsets (Å); "
            "then apply `normalization` (v5: x' = (arcsinh(raw/arcsinh_scale) - mu_a)/sigma_a). "
            "Derive gradmag from x' on the fly. (R_aug,t_aug) = random per-sample aug; "
            "ideally fold R_aug.T@offsets + the resample into a GPU grid_sample at train time."),
    )
    (out_dir / "resample.json").write_text(json.dumps(recipe, indent=2))
    print(f"\nDone. Resample manifests + recipe → {out_dir}")


# ── Main ─────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--version", nargs="+", default=["v3"],
                   choices=["v1", "v2", "v3", "v4", "v5", "v6"],
                   help="Which crop version(s) to produce (the normalisation if-branch). "
                        "v6 = v5 arcsinh normalisation restricted to ligand-matched poses; "
                        "must be generated alone (see --native_filter).")
    p.add_argument("--native_filter", choices=["none", "tt_min", "same_pdb"], default="none",
                   help="Restrict availability to ligand-matched poses (density truly corresponds "
                        "to the modeled ligand). Auto-set to tt_min for --version v6; not allowed "
                        "for v1-v5. tt_min=native crystal pose (strict); same_pdb=native identity, "
                        "pose may be redocked.")
    p.add_argument("--data_dir", default=str(VOXDATA))
    p.add_argument("--ccp4_dir", default=str(VOXDATA / "ccp4"))
    p.add_argument("--pdb_dir", default=str(VOXDATA / "pdb"))
    p.add_argument("--pocket_dir", default=str(VOXDATA / "crossdocked_pocket10"))
    p.add_argument("--out_root", default=str(VOXDATA),
                   help="Parent dir where pretrain/xray_crops_aligned[/ _vN] are written")
    p.add_argument("--splits", nargs="+", default=["train", "test"],
                   choices=["train", "val", "test"])
    p.add_argument("--max_len", type=int, default=30,
                   help="Ligand size filter; must match DatasetCrossDockedXray.max_len")
    p.add_argument("--workers", type=int, default=16)
    # alignment gates
    p.add_argument("--chain_select", choices=["density", "rmsd"], default="density",
                   help="Kabsch chain selection: density=highest density at corrected "
                        "atoms (default); rmsd=lowest Kabsch RMSD (legacy)")
    p.add_argument("--min_atoms", type=int, default=30,
                   help="Min residue-matched atoms for a valid Kabsch fit")
    p.add_argument("--max_rmsd", type=float, default=10.0,
                   help="Loose backstop on post-fit Kabsch RMSD (Å)")
    p.add_argument("--min_density", type=float, default=0.5,
                   help="Min mean density (σ) at corrected atoms — the primary gate")
    # v4 clip
    p.add_argument("--clip_lo", type=float, default=CLIP_LO, help="v4 lower clip (raw units)")
    p.add_argument("--clip_hi", type=float, default=CLIP_HI, help="v4 upper clip (raw units)")
    # v5 arcsinh
    p.add_argument("--v5_arcsinh_scale", type=float, default=V5_ARCSINH_SCALE,
                   help="v5 arcsinh scale s in arcsinh(x/s) (smaller = harder tail squash)")
    # control
    p.add_argument("--force_realign", action="store_true",
                   help="Recompute Kabsch transforms even if cached")
    p.add_argument("--overwrite", action="store_true",
                   help="Rewrite v1 crops even if the .npy already exists")
    p.add_argument("--keep_raw", action="store_true",
                   help="Keep the raw-crop scratch dir (faster re-normalisation later)")
    p.add_argument("--save_gradmag", action="store_true",
                   help="Also save a versioned gradient-magnitude ‖∇ρ‖ channel next to each density "
                        "version: {dir}_gradmag/{split}/{idx}.npy = per-sample-z-scored ‖∇(density)‖, "
                        "plus a stats.json carrying the density version. v2–v5 only.")
    p.add_argument("--limit", type=int, default=0,
                   help="Process only the first N PDB IDs per split (debug)")
    # on-the-fly resample manifest (prepare for train-time density augmentation)
    p.add_argument("--mode", choices=["precompute", "resample"], default="precompute",
                   help="precompute=write frozen normalized crops (default). resample=write ONLY a "
                        "per-split manifest (pdb_id, centroid, Kabsch R/t, ok) + a normalization recipe, "
                        "so density is cropped from the full CCP4 map at the AUGMENTED pose at train "
                        "time (no crops written). Reuses cached transforms + an existing stats.json.")
    p.add_argument("--resample_norm_version", default="v5", choices=["v2", "v3", "v4", "v5", "v6"],
                   help="resample mode: which precomputed version's stats.json supplies the "
                        "normalization recipe baked into the manifest (default v5 = canonical scale).")
    p.add_argument("--resample_out", default="pretrain/xray_resample_v5",
                   help="resample mode: output dir name (under --out_root) for manifests + recipe.")
    return p.parse_args()


def main():
    args = parse_args()
    versions = list(dict.fromkeys(args.version))  # dedup, preserve order

    # v6 (ligand-matched subset) changes the `ok` mask + pool stats globally, so it
    # must be produced on its own to avoid corrupting v1-v5 outputs.
    native_filter = args.native_filter
    if args.mode == "precompute":
        if "v6" in versions:
            if versions != ["v6"]:
                sys.exit("[error] --version v6 (ligand-matched subset) must be generated alone: "
                         "run `--version v6` by itself.")
            if native_filter == "none":
                native_filter = "tt_min"   # v6 default
        elif native_filter != "none":
            sys.exit("[error] --native_filter only applies to --version v6.")
    # resample mode uses --native_filter as-is (decoupled from the v6 crop-version logic)

    need_raw = any(v in versions for v in RAW_VERSIONS)
    need_v4 = "v4" in versions
    need_v5 = ("v5" in versions) or ("v6" in versions)   # both use arcsinh normalisation

    out_root = Path(args.out_root)
    aligned_dir = out_root / DIR_NAME["v1"]

    # Resample-manifest mode: emit manifest + recipe (no crops) and stop.
    if args.mode == "resample":
        run_resample(args, native_filter, aligned_dir, out_root)
        return

    ver_dirs = {v: out_root / DIR_NAME[v] for v in versions}
    scratch_dir = out_root / ".raw_crops_scratch"

    print("=== density crop preprocessing ===")
    print(f"  versions     : {versions}")
    if native_filter != "none":
        print(f"  native_filter: {native_filter}  (availability restricted to ligand-matched poses)")
    print(f"  splits       : {args.splits}  (max_len {args.max_len})")
    print(f"  data_dir     : {args.data_dir}")
    print(f"  ccp4 / pdb   : {args.ccp4_dir}  |  {args.pdb_dir}")
    print(f"  out_root     : {out_root}")
    print(f"  chain_select : {args.chain_select}  | gates: min_atoms={args.min_atoms} "
          f"max_rmsd={args.max_rmsd} min_density={args.min_density}")
    for v in versions:
        print(f"   → {v:<3s} {ver_dirs[v]}")
    if need_raw:
        print(f"  scratch      : {scratch_dir}  (raw crops; "
              f"{'kept' if args.keep_raw else 'removed after norm'})")

    aligned_dir.mkdir(parents=True, exist_ok=True)
    for d in ver_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
        for s in args.splits:
            (d / s).mkdir(parents=True, exist_ok=True)
    if need_raw:
        for s in args.splits:
            (scratch_dir / s).mkdir(parents=True, exist_ok=True)

    cfg = dict(
        ccp4=Path(args.ccp4_dir), pdb=Path(args.pdb_dir),
        min_atoms=args.min_atoms, max_rmsd=args.max_rmsd, min_density=args.min_density,
        chain_select=args.chain_select,
        write_v1=("v1" in versions), need_raw=need_raw, need_v4=need_v4, need_v5=need_v5,
        aligned_dir=aligned_dir, scratch_dir=scratch_dir,
        overwrite=args.overwrite, workers=args.workers,
        clip_lo=args.clip_lo, clip_hi=args.clip_hi, arcsinh_scale=args.v5_arcsinh_scale,
    )

    per_split_stats, per_split_ok = {}, {}
    for split in args.splits:
        samples = load_split(args.data_dir, split, args.max_len)
        R, t, ok = ensure_transforms(split, samples, args, cfg, aligned_dir)
        if native_filter != "none":
            nat = is_native_mask(samples, native_filter)
            ok = ok & nat
            print(f"[{split}] native_filter={native_filter}: {int(nat.sum()):,} ligand-matched | "
                  f"{int(ok.sum()):,} after ∩ density-available ({100 * ok.mean():.1f}% of {len(samples):,})")
        for v in versions:
            np.save(ver_dirs[v] / f"{split}_available.npy", ok)
        stats, _ = run_crop(split, samples, R, t, ok, cfg)
        per_split_stats[split] = stats
        per_split_ok[split] = ok

    if not need_raw:
        print("\nDone (v1 only — per-map z-scored; load with dset.normalize=true).")
        return

    # ── Pool stats over all requested splits ────────────────────────────────────
    S = sum(s["sum"] for s in per_split_stats.values())
    SS = sum(s["sum_sq"] for s in per_split_stats.values())
    N = sum(s["n"] for s in per_split_stats.values())
    MIN = min(s["min"] for s in per_split_stats.values())
    MAX = max(s["max"] for s in per_split_stats.values())
    if N == 0:
        print("[error] no voxels accumulated; aborting."); sys.exit(1)

    mu = S / N
    sigma = float(np.sqrt(max(SS / N - mu * mu, 0.0)))
    max_abs = max(abs(MIN), abs(MAX))
    print(f"\n── pocket-pool stats over {args.splits} ──")
    print(f"  n_voxels {N:,} | raw [{MIN:+.4f}, {MAX:+.4f}] | "
          f"μ {mu:+.6f} | σ {sigma:.6f} | max_abs {max_abs:.6f}")

    mu_clip = sigma_clip = None
    if need_v4:
        CS = sum(s["csum"] for s in per_split_stats.values())
        CSS = sum(s["csum_sq"] for s in per_split_stats.values())
        mu_clip = CS / N
        sigma_clip = float(np.sqrt(max(CSS / N - mu_clip * mu_clip, 0.0)))
        print(f"  v4 clip [{args.clip_lo:+.3f}, {args.clip_hi:+.3f}] | "
              f"μ_clip {mu_clip:+.6f} | σ_clip {sigma_clip:.6f}")

    mu_a = sigma_a = None
    if need_v5:
        AS = sum(s["asum"] for s in per_split_stats.values())
        ASS = sum(s["asum_sq"] for s in per_split_stats.values())
        mu_a = AS / N
        sigma_a = float(np.sqrt(max(ASS / N - mu_a * mu_a, 0.0)))
        print(f"  v5 arcsinh s={args.v5_arcsinh_scale} | "
              f"μ_a {mu_a:+.6f} | σ_a {sigma_a:.6f}")

    def v2n(x): return (x - mu) / sigma
    def v3n(x): return x / max_abs
    def v4n(x): return (np.clip(x, args.clip_lo, args.clip_hi) - mu_clip) / sigma_clip
    def v5n(x): return (np.arcsinh(x / args.v5_arcsinh_scale) - mu_a) / sigma_a
    NORM = {"v2": v2n, "v3": v3n, "v4": v4n, "v5": v5n, "v6": v5n}  # v6 reuses v5 arcsinh

    # ── stats.json per version ──────────────────────────────────────────────────
    common = dict(splits=args.splits, n_voxels=N, raw_min=MIN, raw_max=MAX,
                  per_split={s: {"n_voxels": st["n"]} for s, st in per_split_stats.items()})
    if "v2" in versions:
        (ver_dirs["v2"] / "stats.json").write_text(json.dumps({
            "scheme": "pocket-pool z-score (CrossDocked v2)",
            "formula": "x' = (x - mu) / sigma", "mu": mu, "sigma": sigma, **common}, indent=2))
    if "v3" in versions:
        (ver_dirs["v3"] / "stats.json").write_text(json.dumps({
            "scheme": "pocket-pool symmetric max-abs (CrossDocked v3)",
            "formula": "x' = x / max_abs", "max_abs": max_abs, **common}, indent=2))
    if "v4" in versions:
        (ver_dirs["v4"] / "stats.json").write_text(json.dumps({
            "scheme": "pocket-pool clip + z-score (CrossDocked v4)",
            "formula": "x' = (clip(x, [lo, hi]) - mu_clip) / sigma_clip",
            "clip_lo_raw": args.clip_lo, "clip_hi_raw": args.clip_hi,
            "mu_clipped": mu_clip, "sigma_clipped": sigma_clip,
            "note": "clip stats computed directly from raw aligned crops (no v2 round-trip)",
            **common}, indent=2))
    if "v5" in versions:
        (ver_dirs["v5"] / "stats.json").write_text(json.dumps({
            "scheme": "pocket-pool arcsinh soft-squash + z-score (CrossDocked v5)",
            "formula": "x' = (arcsinh(x / s) - mu_a) / sigma_a",
            "arcsinh_scale": args.v5_arcsinh_scale,
            "mu_a": mu_a, "sigma_a": sigma_a,
            "note": "arcsinh stats computed directly from raw aligned crops",
            **common}, indent=2))
    if "v6" in versions:
        (ver_dirs["v6"] / "stats.json").write_text(json.dumps({
            "scheme": "pocket-pool arcsinh + z-score, ligand-matched subset (CrossDocked v6)",
            "formula": "x' = (arcsinh(x / s) - mu_a) / sigma_a",
            "arcsinh_scale": args.v5_arcsinh_scale,
            "mu_a": mu_a, "sigma_a": sigma_a,
            "native_filter": native_filter,
            "note": "Same arcsinh normalisation as v5, but availability (and pool stats) are "
                    "restricted to ligand-matched poses where the receptor's crystal density "
                    "corresponds to the modeled ligand (modeled ligand == receptor crystal ligand). "
                    f"native_filter={native_filter}.",
            **common}, indent=2))

    # ── Pass B: read raw scratch once, write each requested version ─────────────
    raw_versions = [v for v in versions if v in RAW_VERSIONS]
    # Optional: a versioned ‖∇ρ‖ channel written alongside each density version,
    # computed exactly as the training pipeline does it (per_sample_zscore of the
    # gradient magnitude of the FINAL normalized density), so it is a drop-in
    # gradmag source. ‖∇·‖ is rotation-invariant → still aligned under aug.
    gmag_dirs = {}
    if args.save_gradmag and raw_versions:
        from voxbind.models.mae_ops import gradient_magnitude3d, per_sample_zscore
        import torch
        gmag_dirs = {v: out_root / (DIR_NAME[v] + "_gradmag") for v in raw_versions}
        for v in raw_versions:
            for split in args.splits:
                (gmag_dirs[v] / split).mkdir(parents=True, exist_ok=True)

        def _gradmag(norm_np):
            g = gradient_magnitude3d(torch.from_numpy(norm_np.astype(np.float32))
                                     .view(1, 1, GRID_DIM, GRID_DIM, GRID_DIM))
            return per_sample_zscore(g).view(GRID_DIM, GRID_DIM, GRID_DIM).numpy().astype(np.float16)

    print()
    for split in args.splits:
        idxs = np.where(per_split_ok[split])[0]
        for i in tqdm(idxs, desc=f"normalize {split}", unit="file"):
            raw_path = scratch_dir / split / f"{i:06d}.npy"
            if not raw_path.exists():
                continue
            raw = np.load(raw_path).astype(np.float32)
            for v in raw_versions:
                norm = NORM[v](raw).astype(np.float16)
                np.save(str(ver_dirs[v] / split / f"{i:06d}.npy"), norm)
                if gmag_dirs:
                    np.save(str(gmag_dirs[v] / split / f"{i:06d}.npy"), _gradmag(norm))

    # versioned gradmag stats.json + availability (mirror the density version)
    if gmag_dirs:
        for v in raw_versions:
            for split in args.splits:
                src = ver_dirs[v] / f"{split}_available.npy"
                if src.exists():
                    (gmag_dirs[v] / f"{split}_available.npy").write_bytes(src.read_bytes())
            (gmag_dirs[v] / "stats.json").write_text(json.dumps({
                "scheme": f"gradient magnitude ‖∇ρ‖ of CrossDocked {v} density",
                "formula": f"x' = per_sample_zscore(‖∇(density_{v})‖)",
                "density_version": v, "gradmag": True, **common}, indent=2))
            print(f"  {v} gradmag → {gmag_dirs[v]}")

    if not args.keep_raw:
        shutil.rmtree(scratch_dir, ignore_errors=True)
        print(f"  removed scratch {scratch_dir}")
    else:
        print(f"  kept raw scratch {scratch_dir}")

    print("\nDone.")
    for v in versions:
        print(f"  {v} → {ver_dirs[v]}")


if __name__ == "__main__":
    main()
