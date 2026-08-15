#!/usr/bin/env python
"""Build canonical-v5 CrossDocked crops using receptor ED availability only.

The original aligned-v5 set gates every pose by local protein density. This
builder instead shares one receptor-frame transform per (receptor PDB, chain)
and keeps every cross-docked pose whose receptor has a readable ED map.

Existing trustworthy v5 crops are hard-linked into the new dataset. Only newly
recovered rows are cropped, so the overlay costs about 12k crops rather than
duplicating the full corpus.
"""

import argparse
import csv
import json
import os
import random
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Import gemmi before torch (required by this environment).
import gemmi
import numpy as np
import scipy.ndimage
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from voxbind.dataset.crossdocked_xray import _crop_density


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
GRID_DIM = 64
RESOLUTION = 0.25
KEY_RE = re.compile(r"^([0-9A-Za-z]{4})_([^_]+)_rec_")


def load_split(data_dir: Path, split: str, max_len: int):
    if split == "train":
        data = torch.load(data_dir / "data_train.pt", weights_only=False)
        random.Random(1234).shuffle(data)
        data = data[:-100]
    elif split == "test":
        data = torch.load(data_dir / "data_test.pt", weights_only=False)
    else:
        raise ValueError(split)
    return [(p, l) for p, l in data if l["max_len"] <= max_len]


def receptor_key(pocket_id: str):
    base = pocket_id.split("/")[-1]
    match = KEY_RE.match(base)
    if match is None:
        raise ValueError(f"cannot parse receptor PDB+chain from {pocket_id}")
    return match.group(1).lower(), match.group(2)


def ligand_centroid(ligand: dict):
    mask = ligand["atoms_channel"] < 7
    return ligand["coords"][mask].float().mean(dim=0).numpy().astype(np.float64)


def atoms_by_key(structure, per_chain: bool):
    out = defaultdict(dict) if per_chain else {}
    for chain in structure[0]:
        for residue in chain:
            if residue.is_water() or residue.het_flag == "H":
                continue
            for atom in residue:
                if atom.is_hydrogen():
                    continue
                key = (residue.seqid.num, atom.name)
                pos = [atom.pos.x, atom.pos.y, atom.pos.z]
                if per_chain:
                    out[chain.name][key] = pos
                else:
                    out[key] = pos
    return out


def kabsch(P: np.ndarray, Q: np.ndarray):
    P0, Q0 = P.mean(0), Q.mean(0)
    U, _S, Vt = np.linalg.svd((P - P0).T @ (Q - Q0))
    sign = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, sign]) @ U.T
    t = Q0 - P0 @ R.T
    rmsd = float(np.sqrt((((P @ R.T + t) - Q) ** 2).sum(1).mean()))
    return R, t, rmsd


def fallback_transform(
    pocket_paths,
    deposited_path: str,
    receptor_chain: str,
):
    """Recover one receptor transform in a CPU process."""
    deposited_path = Path(deposited_path)
    if not deposited_path.exists():
        return None
    try:
        deposited = atoms_by_key(
            gemmi.read_structure(str(deposited_path)), per_chain=True
        )
    except Exception:
        return None

    chain_candidates = (
        {receptor_chain: deposited[receptor_chain]}
        if receptor_chain in deposited
        else deposited
    )
    best = None
    # Receptor coordinates are repeated across cross-docked poses. A few pocket
    # files are enough to find the most complete residue/atom correspondence.
    for pocket_path in pocket_paths:
        try:
            pocket = atoms_by_key(
                gemmi.read_structure(str(pocket_path)), per_chain=False
            )
        except Exception:
            continue
        for chain_name, dep_atoms in chain_candidates.items():
            keys = [key for key in pocket if key in dep_atoms]
            if len(keys) < 3:
                continue
            P = np.asarray([pocket[key] for key in keys], dtype=np.float64)
            Q = np.asarray([dep_atoms[key] for key in keys], dtype=np.float64)
            R, t, rmsd = kabsch(P, Q)
            candidate = (len(keys), -rmsd, R, t, chain_name)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    if best is None:
        return None
    return dict(n_match=best[0], rmsd=-best[1], R=best[2], t=best[3], chain=best[4])


def load_raw_grid(path: Path):
    try:
        ccp4 = gemmi.read_ccp4_map(str(path))
        ccp4.setup(float("nan"))
        grid = ccp4.grid
        arr = np.array(grid, dtype=np.float32)
        if not np.isfinite(arr).all():
            return None
        frac_T = np.linalg.inv(np.array(grid.unit_cell.orth.mat.tolist())).T
        return arr, frac_T, *arr.shape
    except Exception:
        return None


def crop_receptor_jobs(pdb_id: str, jobs, cfg):
    grid = load_raw_grid(Path(cfg["ccp4_dir"]) / f"{pdb_id}.ccp4")
    if grid is None:
        return [], [job[0] for job in jobs], "map_load_failed"

    arr, frac_T, nu, nv, nw = grid
    out_dir = Path(cfg["out_dir"])
    scale = cfg["scale"]
    mu = cfg["mu"]
    sigma = cfg["sigma"]
    written, failed = [], []

    for idx, centroid, R, t in jobs:
        destination = out_dir / f"{idx:06d}.npy"
        if destination.exists():
            written.append(idx)
            continue
        try:
            crop = _crop_density(
                arr,
                frac_T,
                nu,
                nv,
                nw,
                np.asarray(centroid, dtype=np.float64),
                G=GRID_DIM,
                res=RESOLUTION,
                transform=(np.asarray(R), np.asarray(t)),
            )
            crop = (np.arcsinh(crop / scale) - mu) / sigma
            temporary = destination.with_suffix(".npy.tmp")
            with temporary.open("wb") as handle:
                np.save(handle, crop.astype(np.float16))
            os.replace(temporary, destination)
            written.append(idx)
        except Exception:
            failed.append(idx)
    return written, failed, "ok"


def choose_shared_transforms(
    samples,
    transforms,
    old_reasons,
    ccp4_dir: Path,
    pdb_dir: Path,
    pocket_dir: Path,
    fallback_workers: int,
):
    n = len(samples)
    R_shared = np.tile(np.eye(3, dtype=np.float64), (n, 1, 1))
    t_shared = np.zeros((n, 3), dtype=np.float64)
    available = np.zeros(n, dtype=bool)
    selected = np.full(n, -1, dtype=np.int64)
    source_kind = np.full(n, "unavailable", dtype="<U32")

    groups = defaultdict(list)
    for idx, (pocket, _ligand) in enumerate(samples):
        groups[receptor_key(pocket["id"])].append(idx)

    def assign(ids, picked, kind, R, t):
        group_counts[kind] += 1
        for idx in ids:
            available[idx] = True
            R_shared[idx] = R
            t_shared[idx] = t
            selected[idx] = picked
            source_kind[idx] = kind

    group_counts = Counter()
    fallback_jobs = []
    for (pdb_id, chain), ids in groups.items():
        map_path = ccp4_dir / f"{pdb_id}.ccp4"
        if not (map_path.exists() and map_path.stat().st_size > 0):
            group_counts["no_map"] += 1
            continue

        candidates = [
            idx
            for idx in ids
            if int(transforms["n_match"][idx]) > 0
            and np.isfinite(transforms["R"][idx]).all()
            and np.isfinite(transforms["t"][idx]).all()
        ]
        if candidates:
            def score(idx):
                rmsd = float(transforms["rmsd"][idx])
                return (
                    int(bool(transforms["ok"][idx])),
                    int(transforms["n_match"][idx]),
                    float(transforms["density"][idx]),
                    -rmsd if rmsd >= 0 else -1e9,
                )
            picked = max(candidates, key=score)
            kind = (
                "shared_trustworthy"
                if bool(transforms["ok"][picked])
                else "shared_ungated"
            )
            assign(
                ids,
                picked,
                kind,
                transforms["R"][picked],
                transforms["t"][picked],
            )
            continue

        pocket_paths = [
            str(pocket_dir / samples[idx][0]["id"])
            for idx in ids[: min(3, len(ids))]
        ]
        fallback_jobs.append(
            (pdb_id, chain, ids, pocket_paths, str(pdb_dir / f"{pdb_id}.pdb"))
        )

    if fallback_jobs:
        print(
            f"[transforms] recovering {len(fallback_jobs):,} receptor groups "
            f"with {fallback_workers} workers",
            flush=True,
        )
        with ProcessPoolExecutor(max_workers=fallback_workers) as executor:
            futures = {
                executor.submit(fallback_transform, paths, deposited, chain): (ids, pdb_id)
                for pdb_id, chain, ids, paths, deposited in fallback_jobs
            }
            for done, future in enumerate(as_completed(futures), 1):
                ids, _pdb_id = futures[future]
                recovered = future.result()
                if recovered is None:
                    group_counts["no_transform"] += 1
                else:
                    assign(
                        ids,
                        -2,
                        "fallback_kabsch",
                        recovered["R"],
                        recovered["t"],
                    )
                if done % 250 == 0 or done == len(futures):
                    print(
                        f"[transforms] fallback {done:,}/{len(futures):,}",
                        flush=True,
                    )

    return R_shared, t_shared, available, selected, source_kind, group_counts


def load_reasons(path: Path, n: int):
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != n:
        raise RuntimeError(f"{path} has {len(rows)} rows, expected {n}")
    return [row["reason"] for row in rows]


def link_existing(source_dir: Path, destination_dir: Path, old_ok, new_ok):
    linked = copied = already = 0
    for idx in np.flatnonzero(old_ok & new_ok):
        source = source_dir / f"{idx:06d}.npy"
        destination = destination_dir / f"{idx:06d}.npy"
        if destination.exists():
            already += 1
            continue
        if not source.exists():
            continue
        try:
            os.link(source, destination)
            linked += 1
        except OSError:
            shutil.copy2(source, destination)
            copied += 1
    return linked, copied, already


def build_split(split: str, args, norm):
    samples = load_split(args.data_dir, split, args.max_len)
    n = len(samples)
    tx_path = args.alignment_dir / f"{split}_transforms.npz"
    stats_path = args.alignment_dir / f"{split}_stats.csv"
    with np.load(tx_path) as archive:
        transforms = {key: archive[key] for key in archive.files}
    old_reasons = load_reasons(stats_path, n)
    old_ok = np.asarray(transforms["ok"], dtype=bool)

    R, t, available, selected, source_kind, group_counts = choose_shared_transforms(
        samples,
        transforms,
        old_reasons,
        args.ccp4_dir,
        args.pdb_dir,
        args.pocket_dir,
        args.workers,
    )
    print(
        f"[{split}] receptor-ED available {int(available.sum()):,}/{n:,}; "
        f"old gated {int(old_ok.sum()):,}; recovered {int((available & ~old_ok).sum()):,}"
    )
    print(f"[{split}] receptor groups: {dict(group_counts)}")
    if args.dry_run:
        return dict(
            n=n,
            available=int(available.sum()),
            old_available=int(old_ok.sum()),
            recovered=int((available & ~old_ok).sum()),
        )

    split_out = args.out_dir / split
    split_out.mkdir(parents=True, exist_ok=True)
    linked, copied, already = link_existing(
        args.source_crops / split, split_out, old_ok, available
    )
    print(
        f"[{split}] existing crops: hardlinked={linked:,} copied={copied:,} "
        f"already={already:,}"
    )

    jobs = defaultdict(list)
    for idx in np.flatnonzero(available):
        destination = split_out / f"{idx:06d}.npy"
        if destination.exists():
            continue
        pdb_id, _chain = receptor_key(samples[idx][0]["id"])
        jobs[pdb_id].append(
            (idx, ligand_centroid(samples[idx][1]), R[idx], t[idx])
        )

    cfg = dict(
        ccp4_dir=str(args.ccp4_dir),
        out_dir=str(split_out),
        scale=float(norm["arcsinh_scale"]),
        mu=float(norm["mu_a"]),
        sigma=float(norm["sigma_a"]),
    )
    failed = []
    written = 0
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(crop_receptor_jobs, pdb_id, group_jobs, cfg): pdb_id
            for pdb_id, group_jobs in jobs.items()
        }
        for done, future in enumerate(as_completed(futures), 1):
            made, bad, status = future.result()
            written += len(made)
            failed.extend(bad)
            if done % 250 == 0 or done == len(futures):
                print(
                    f"[{split}] crop {done:,}/{len(futures):,} PDBs | "
                    f"written {written:,} failed {len(failed):,} | "
                    f"{time.time() - started:.0f}s",
                    flush=True,
                )

    if failed:
        available[np.asarray(failed, dtype=int)] = False
        source_kind[np.asarray(failed, dtype=int)] = "crop_failed"

    np.save(args.out_dir / f"{split}_available.npy", available)
    np.savez(
        args.out_dir / f"{split}_receptor_transforms.npz",
        R=R.astype(np.float32),
        t=t.astype(np.float32),
        ok=available,
        selected_source_index=selected,
        source_kind=source_kind,
    )

    missing = [
        idx
        for idx in np.flatnonzero(available)
        if not (split_out / f"{idx:06d}.npy").exists()
    ]
    if missing:
        raise RuntimeError(f"[{split}] {len(missing)} available crops are missing")
    file_count = len(list(split_out.glob("[0-9][0-9][0-9][0-9][0-9][0-9].npy")))
    print(
        f"[{split}] complete: available={int(available.sum()):,}, "
        f"crop_files={file_count:,}, newly_written={written:,}"
    )
    return dict(
        n=n,
        available=int(available.sum()),
        old_available=int(old_ok.sum()),
        recovered=int((available & ~old_ok).sum()),
        crop_failed=len(failed),
        crop_files=file_count,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data_dir", type=Path, default=DATA)
    parser.add_argument(
        "--alignment_dir", type=Path, default=DATA / "xray_crops_aligned"
    )
    parser.add_argument(
        "--source_crops", type=Path, default=DATA / "xray_crops_aligned_v5"
    )
    parser.add_argument(
        "--out_dir", type=Path, default=DATA / "xray_crops_receptor_ed_v5"
    )
    parser.add_argument("--ccp4_dir", type=Path, default=DATA / "ccp4")
    parser.add_argument("--pdb_dir", type=Path, default=DATA / "pdb")
    parser.add_argument(
        "--pocket_dir", type=Path, default=DATA / "crossdocked_pocket10"
    )
    parser.add_argument(
        "--splits", nargs="+", choices=["train", "test"], default=["train", "test"]
    )
    parser.add_argument("--max_len", type=int, default=30)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    norm_path = args.source_crops / "stats.json"
    norm = json.loads(norm_path.read_text())
    required = ("arcsinh_scale", "mu_a", "sigma_a")
    if not all(key in norm for key in required):
        raise RuntimeError(f"{norm_path} is not a canonical v5 normalization recipe")

    print("=== CrossDocked receptor-ED v5 dataset ===")
    print(f"source={args.source_crops}")
    print(f"output={args.out_dir}")
    print(f"splits={args.splits} workers={args.workers} dry_run={args.dry_run}")

    if not args.dry_run:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for split in args.splits:
        summary[split] = build_split(split, args, norm)

    if not args.dry_run:
        metadata = dict(norm)
        metadata.update(
            {
                "dataset": "CrossDocked receptor-ED v5",
                "availability_gate": (
                    "readable receptor 2Fo-Fc map + receptor PDB/chain frame transform; "
                    "no per-pose density, ligand-source, or docking-pose gate"
                ),
                "transform_scope": "shared per receptor PDB+chain",
                "normalization_source": str(norm_path.resolve()),
                "storage": "existing trustworthy crops hard-linked; recovered crops materialized",
                "summary": summary,
            }
        )
        (args.out_dir / "stats.json").write_text(json.dumps(metadata, indent=2))
        (args.out_dir / ".complete").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
