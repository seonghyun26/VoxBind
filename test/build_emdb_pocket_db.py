"""
Build EMDB pocket-crop database
================================
For every (pocket, EMDB map) pair where a downloaded map exists:
  1. Load the pocket's center coordinates (ligand CoM, PDB coordinate frame)
  2. Load the EMDB .map.gz (MRC format) — each map is loaded ONCE for all pockets
  3. Z-score normalise the full-map density
  4. Crop a 64³ box at native EMDB resolution centred on the pocket
  5. Store in HDF5: /{split}/{pocket_key}/{emdb_num} → float32 (64,64,64)
     Attributes: voxel_size_angstrom, center_pdb[3], crop_origin_vox[3]

Usage
-----
    pip install h5py          # if not already installed
    conda run -n voxbind python test/build_emdb_pocket_db.py

The HDF5 database is written incrementally ("a" mode) so the script can be
interrupted and resumed — already-written entries are skipped.
"""

import gzip
import json
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import h5py
import mrcfile
import numpy as np
import torch
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).parent.parent
DATA_DIR     = REPO_ROOT / "voxbind" / "dataset" / "data"
NB_DATA_DIR  = REPO_ROOT / "notebook" / "data"
EMDB_DIR     = NB_DATA_DIR / "emdb"
MAPPING_FILE = NB_DATA_DIR / "crossdocked_pdb_to_emdb.json"
OUT_DIR      = Path(__file__).parent / "data"
OUT_DB       = OUT_DIR / "emdb_pocket_crops.h5"

CROP_DIM = 64   # voxels per axis at native EMDB resolution (~1 Å/voxel → ~64 Å box)
HALF     = CROP_DIM // 2

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Coordinate / IO helpers ───────────────────────────────────────────────────

def extract_pdb_id(pocket_id: str) -> str:
    """'crossdocked/1AJ6_rec_...' → '1AJ6'"""
    return pocket_id.split("/")[1].split("_")[0].upper()


def pocket_hdf5_key(pocket_id: str) -> str:
    """Replace '/' so the pocket_id can be used as a single HDF5 group name."""
    return pocket_id.replace("/", "__")


def emdb_num(emdb_id: str) -> str:
    """'EMD-1234' → '1234'"""
    return emdb_id.split("-")[-1]


def map_path(emdb_id: str) -> Path:
    return EMDB_DIR / f"emd_{emdb_num(emdb_id)}.map.gz"


def load_mrc(gz_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decompress and read an EMDB .map.gz.

    Returns
    -------
    data_xyz : np.ndarray, float32, shape (NX, NY, NZ)
        Density values with axes ordered (X, Y, Z) regardless of internal MRC
        axis convention (mapc/mapr/maps).
    vs : np.ndarray, float64, shape (3,)
        Voxel size in Angstroms along (X, Y, Z).
    origin : np.ndarray, float64, shape (3,)
        Physical coordinate (Å) of the (0,0,0) voxel corner in data_xyz.
    """
    with tempfile.NamedTemporaryFile(suffix=".map", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with gzip.open(gz_path, "rb") as gz, open(tmp_path, "wb") as f:
            shutil.copyfileobj(gz, f)
        with mrcfile.open(str(tmp_path), mode="r", permissive=True) as mrc:
            vs = np.array(
                [mrc.voxel_size.x, mrc.voxel_size.y, mrc.voxel_size.z],
                dtype=np.float64,
            )
            ox = float(mrc.header.origin.x)
            oy = float(mrc.header.origin.y)
            oz = float(mrc.header.origin.z)
            if ox == 0.0 and oy == 0.0 and oz == 0.0:
                # Fall back to nstart * voxel_size (common in older EMDB maps)
                ox = float(mrc.header.nxstart) * vs[0]
                oy = float(mrc.header.nystart) * vs[1]
                oz = float(mrc.header.nzstart) * vs[2]
            origin = np.array([ox, oy, oz], dtype=np.float64)

            # MRC stores (nsections, nrows, ncols), i.e. (maps-dim, mapr-dim, mapc-dim)
            mapc = int(mrc.header.mapc)  # physical axis (1=X,2=Y,3=Z) along columns
            mapr = int(mrc.header.mapr)  # along rows
            maps = int(mrc.header.maps)  # along sections

            data = mrc.data.astype(np.float32)  # (nz, ny, nx) in storage order
    finally:
        tmp_path.unlink(missing_ok=True)

    # Reorder axes so that output dim 0 = X, dim 1 = Y, dim 2 = Z
    # storage_order[i] is the physical axis (1/2/3) at data dimension i
    storage_order = [maps, mapr, mapc]
    perm = [storage_order.index(i + 1) for i in range(3)]
    data_xyz = np.transpose(data, perm)

    return data_xyz, vs, origin


def zscore_map(data: np.ndarray) -> np.ndarray:
    """Z-score normalise using full-map statistics."""
    mu  = data.mean()
    std = data.std()
    if std < 1e-10:
        return np.zeros_like(data)
    return ((data - mu) / std).astype(np.float32)


def crop_map(
    data_xyz: np.ndarray,
    vs: np.ndarray,
    origin: np.ndarray,
    center: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract a CROP_DIM³ sub-volume centred on `center` (Å, PDB frame).

    Returns
    -------
    crop : np.ndarray, float32, shape (CROP_DIM, CROP_DIM, CROP_DIM)
        Zero-padded if the region extends outside the map.
    crop_origin_vox : np.ndarray, int32, shape (3,)
        Voxel index (X, Y, Z) of the [0,0,0] corner of the crop within data_xyz.
    """
    frac  = (center - origin) / vs           # fractional voxel index of center
    ci    = np.round(frac).astype(int)       # nearest integer voxel
    lo    = ci - HALF                        # inclusive start corner
    hi    = lo + CROP_DIM                    # exclusive end corner

    shape = np.array(data_xyz.shape, dtype=int)  # (NX, NY, NZ)

    lo_c = np.clip(lo, 0, shape)
    hi_c = np.clip(hi, 0, shape)

    crop   = np.zeros((CROP_DIM, CROP_DIM, CROP_DIM), dtype=np.float32)
    dst_lo = lo_c - lo
    dst_hi = dst_lo + (hi_c - lo_c)

    crop[
        dst_lo[0]:dst_hi[0],
        dst_lo[1]:dst_hi[1],
        dst_lo[2]:dst_hi[2],
    ] = data_xyz[lo_c[0]:hi_c[0], lo_c[1]:hi_c[1], lo_c[2]:hi_c[2]]

    return crop, lo.astype(np.int32)


# ── Main ──────────────────────────────────────────────────────────────────────

def build_reverse_index(
    splits: dict[str, list],
    pdb_to_emdb: dict[str, list[str]],
) -> dict[str, list[tuple[str, str, np.ndarray]]]:
    """
    For every pocket that has at least one downloaded EMDB map, record
    which EMDB maps it needs.

    Returns
    -------
    emdb_to_pockets : dict  emdb_id → [(split, pocket_id, center_xyz), ...]
        Grouping by EMDB map lets us load each .map.gz exactly once.
    """
    emdb_to_pockets: dict[str, list] = defaultdict(list)

    for split_name, raw_data in splits.items():
        for pocket_, ligand_ in raw_data:
            pid      = extract_pdb_id(pocket_["id"])
            emdb_ids = [e for e in pdb_to_emdb.get(pid, []) if map_path(e).exists()]
            if not emdb_ids:
                continue

            mask   = ligand_["atoms_channel"] < 7
            center = ligand_["coords"][mask].mean(dim=0).numpy()  # PDB-frame CoM

            for emdb_id in emdb_ids:
                emdb_to_pockets[emdb_id].append((split_name, pocket_["id"], center))

    return emdb_to_pockets


def main() -> None:
    print("=" * 60)
    print("EMDB pocket-crop database builder")
    print("=" * 60)
    print(f"  Mapping  : {MAPPING_FILE}")
    print(f"  EMDB dir : {EMDB_DIR}")
    print(f"  Output   : {OUT_DB}")
    print(f"  Crop dim : {CROP_DIM}³ voxels at native EMDB resolution")

    with open(MAPPING_FILE) as f:
        pdb_to_emdb: dict[str, list[str]] = json.load(f)

    n_maps_on_disk = sum(1 for v in pdb_to_emdb.values() for e in v if map_path(e).exists())
    print(f"\n  PDB IDs in mapping : {len(pdb_to_emdb):,}")
    print(f"  EMDB maps on disk  : {n_maps_on_disk:,}")

    # ── Load raw .pt files ─────────────────────────────────────────────────────
    print("\n[Loading CrossDocked .pt files …]")
    splits: dict[str, list] = {}
    for split_name, fname in [("train", "data_train.pt"), ("test", "data_test.pt")]:
        pt_path = DATA_DIR / fname
        if not pt_path.exists():
            print(f"  [SKIP] {fname} not found")
            continue
        splits[split_name] = torch.load(pt_path, weights_only=False)
        print(f"  {fname}: {len(splits[split_name]):,} samples")

    # ── Build reverse index (emdb_id → pockets that need it) ──────────────────
    print("\n[Indexing pockets by EMDB map …]")
    emdb_to_pockets = build_reverse_index(splits, pdb_to_emdb)
    total_crops = sum(len(v) for v in emdb_to_pockets.values())
    print(f"  Unique EMDB maps needed : {len(emdb_to_pockets):,}")
    print(f"  Total (pocket, map) pairs : {total_crops:,}")

    # ── Process map by map ─────────────────────────────────────────────────────
    print(f"\n[Writing HDF5 database (mode='a', skips existing entries) …]")
    written      = 0
    already_done = 0
    map_errors   = 0

    with h5py.File(OUT_DB, "a") as h5:
        h5.attrs["description"] = (
            f"EMDB {CROP_DIM}³ crops (native resolution, z-scored) "
            "centred on CrossDocked pocket ligand CoM"
        )
        h5.attrs["crop_dim"] = CROP_DIM

        # Ensure split groups exist
        for split_name in splits:
            h5.require_group(split_name)

        for emdb_id, pocket_list in tqdm(
            emdb_to_pockets.items(),
            desc="EMDB maps",
            unit="map",
        ):
            # Skip if every crop for this map already exists
            ekey = emdb_num(emdb_id)
            pending = [
                (spl, pid, ctr)
                for spl, pid, ctr in pocket_list
                if ekey not in h5[spl].get(pocket_hdf5_key(pid), {})
            ]
            if not pending:
                already_done += len(pocket_list)
                continue

            # Load and normalise the map once
            try:
                data_xyz, vs, origin = load_mrc(map_path(emdb_id))
            except Exception as exc:
                tqdm.write(f"  [WARN] {emdb_id}: {exc}")
                map_errors += 1
                continue

            data_norm = zscore_map(data_xyz)

            for split_name, pocket_id, center in pending:
                pkey = pocket_hdf5_key(pocket_id)
                pgrp = h5[split_name].require_group(pkey)

                # Pocket-level metadata (written once)
                if "center_pdb" not in pgrp.attrs:
                    pgrp.attrs["center_pdb"] = center.astype(np.float32)
                    pgrp.attrs["pdb_id"]     = extract_pdb_id(pocket_id)
                    pgrp.attrs["pocket_id"]  = pocket_id

                if ekey in pgrp:
                    already_done += 1
                    continue

                crop, crop_origin = crop_map(data_norm, vs, origin, center)

                ds = pgrp.create_dataset(
                    ekey,
                    data=crop,
                    compression="gzip",
                    compression_opts=4,
                )
                ds.attrs["emdb_id"]             = emdb_id
                ds.attrs["voxel_size_angstrom"]  = vs.astype(np.float32)
                ds.attrs["crop_origin_vox"]      = crop_origin

                written += 1

            # flush periodically to avoid losing work on interruption
            if written % 5000 == 0 and written > 0:
                h5.flush()

    size_gb = OUT_DB.stat().st_size / 1e9
    print(f"\nDone.")
    print(f"  Written          : {written:,}")
    print(f"  Already cached   : {already_done:,}")
    print(f"  Map load errors  : {map_errors:,}")
    print(f"  DB size          : {size_gb:.2f} GB  →  {OUT_DB}")


if __name__ == "__main__":
    main()
