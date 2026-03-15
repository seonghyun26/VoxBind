"""05_process_xray.py — Precompute 64³ X-ray density crops for CrossDocked2020

For every sample in data_train.pt / data_test.pt, crops the VoxBind-aligned
64³ × 0.25 Å density box from the corresponding CCP4 map and saves it as a
float16 numpy array.

Samples are processed in PDB-ID groups so each CCP4 file is read only once.
The output index matches DatasetCrossDockedXray exactly (same shuffle seed,
same size filter) so the dataset can load crops by sample index with no
additional lookup.

Run AFTER scripts/04_download_xray.py.

Usage
-----
    cd voxbind
    python scripts/05_process_xray.py \
        --data_dir dataset/data \
        --ccp4_dir dataset/data/ccp4 \
        --out_dir  dataset/data/xray_crops \
        --max_len  30

Outputs
-------
    dataset/data/xray_crops/
        train/{:06d}.npy     float16 (64, 64, 64) per training sample
        test/{:06d}.npy      float16 (64, 64, 64) per test sample
        train_available.npy  bool (N_train,)  True where CCP4 map was found
        test_available.npy   bool (N_test,)
"""

import argparse
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# reuse helpers from the dataset module
from voxbind.dataset.crossdocked_xray import _load_grid, _crop_density


# ── Data loading (mirrors DatasetCrossDockedXray.__init__) ─────────────────────

def load_split(data_dir: str, split: str, max_len: int):
    """Return the filtered sample list in the same order as the dataset class."""
    if split in ("train", "val"):
        data = torch.load(os.path.join(data_dir, "data_train.pt"), weights_only=False)
        random.Random(1234).shuffle(data)
        val_sz = 100
        data = data[: len(data) - val_sz] if split == "train" else data[len(data) - val_sz :]
    else:
        data = torch.load(os.path.join(data_dir, "data_test.pt"), weights_only=False)

    data = [(p, l) for p, l in data if l["max_len"] <= max_len]
    return data


def get_pdb_id(pocket_id: str) -> str:
    return pocket_id.split("/")[1].split("_")[0].lower()


def get_centroid(ligand_: dict) -> np.ndarray:
    """Ligand centroid in PDB Cartesian Å (same as DatasetCrossDockedXray)."""
    mask = ligand_["atoms_channel"] < 7
    return ligand_["coords"][mask].float().mean(dim=0).numpy()


# ── Processing ─────────────────────────────────────────────────────────────────

def process_split(
    data: list,
    ccp4_dir: Path,
    out_dir: Path,
    split: str,
    overwrite: bool,
) -> None:
    split_dir = out_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    n = len(data)
    available = np.zeros(n, dtype=bool)

    # ── Group sample indices by PDB ID ────────────────────────────────────────
    # Each CCP4 file is loaded once and used for all its samples.
    pdb_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, (pocket_, _) in enumerate(data):
        pdb_to_indices[get_pdb_id(pocket_["id"])].append(i)

    n_pdbs_found = sum(
        1 for pdb_id in pdb_to_indices
        if (ccp4_dir / f"{pdb_id}.map").exists()
    )
    print(f"\n[{split}]  {n:,} samples  |  "
          f"{len(pdb_to_indices):,} unique PDB IDs  |  "
          f"{n_pdbs_found:,} with CCP4 map")

    # ── Process PDB by PDB ────────────────────────────────────────────────────
    for pdb_id, indices in tqdm(pdb_to_indices.items(),
                                desc=f"{split}", unit="PDB"):
        ccp4_path = ccp4_dir / f"{pdb_id}.map"
        if not ccp4_path.exists():
            continue  # available[indices] stays False

        grid = _load_grid(ccp4_path)
        if grid is None:
            continue

        arr_norm, frac_T, nu, nv, nw = grid

        for i in indices:
            out_path = split_dir / f"{i:06d}.npy"

            if out_path.exists() and not overwrite:
                available[i] = True
                continue

            _, ligand_ = data[i]
            centroid = get_centroid(ligand_)
            crop = _crop_density(arr_norm, frac_T, nu, nv, nw, centroid)

            np.save(str(out_path), crop.astype(np.float16))
            available[i] = True

    # ── Save availability mask ─────────────────────────────────────────────────
    avail_path = out_dir / f"{split}_available.npy"
    np.save(str(avail_path), available)

    n_ok = available.sum()
    print(f"  Saved {n_ok:,} / {n:,} crops  "
          f"({100 * n_ok / n:.1f}%)  →  {split_dir}")
    print(f"  Availability mask  →  {avail_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Precompute 64³ X-ray density crops for CrossDocked2020",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir", default="dataset/data",
                   help="Directory containing data_train.pt / data_test.pt")
    p.add_argument("--ccp4_dir", default="dataset/data/ccp4",
                   help="Directory of downloaded .map CCP4 files")
    p.add_argument("--out_dir",  default="dataset/data/xray_crops",
                   help="Output directory for precomputed crops")
    p.add_argument("--splits",   nargs="+", default=["train", "test"],
                   choices=["train", "val", "test"],
                   help="Which splits to process")
    p.add_argument("--max_len",  type=int, default=30,
                   help="Discard ligands with max coordinate extent > max_len Å "
                        "(must match DatasetCrossDockedXray.max_len)")
    p.add_argument("--overwrite", action="store_true",
                   help="Recompute and overwrite existing crop files")
    return p.parse_args()


def main():
    args = parse_args()
    ccp4_dir = Path(args.ccp4_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== X-ray density crop preprocessing ===")
    print(f"  data_dir : {Path(args.data_dir).resolve()}")
    print(f"  ccp4_dir : {ccp4_dir.resolve()}")
    print(f"  out_dir  : {out_dir.resolve()}")
    print(f"  splits   : {args.splits}")
    print(f"  max_len  : {args.max_len} Å")
    print(f"  overwrite: {args.overwrite}")

    for split in args.splits:
        data = load_split(args.data_dir, split, args.max_len)
        process_split(data, ccp4_dir, out_dir, split, args.overwrite)

    print("\nDone.")


if __name__ == "__main__":
    main()
