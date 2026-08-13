#!/usr/bin/env python3
"""Freeze official ATOM3D LBA ID30/ID60 assignments on our eligible pool.

The output membership is exactly::

    official ATOM3D LBA assignment ∩ lp_edrscc_v2

No sequences are re-clustered here.  The official split membership is read from
the released ATOM3D LMDBs mirrored under ``base/profsa/data/dataset``; our
quality/label filters are represented by the already-frozen ``lp_edrscc_v2``
manifest.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
import types
from collections import Counter
from pathlib import Path

import lmdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT / "voxbind" / "splits"
DEFAULT_ATOM3D_ROOT = ROOT / "base" / "profsa" / "data" / "dataset"

OFFICIAL_COUNTS = {
    30: {"train": 3507, "val": 466, "test": 490},
    60: {"train": 3563, "val": 448, "test": 452},
}
FILTERED_COUNTS = {
    30: {"train": 2078, "val": 266, "test": 340},
    60: {"train": 2150, "val": 273, "test": 261},
}


def _install_pandas_pickle_compat() -> None:
    """Provide aliases needed by the pandas version used to build old LMDBs."""
    module = types.ModuleType("pandas.core.indexes.numeric")
    for name in ("Int64Index", "UInt64Index", "Float64Index"):
        setattr(module, name, pd.Index)
    sys.modules.setdefault("pandas.core.indexes.numeric", module)


def _read_lmdb_ids(path: Path) -> list[str]:
    _install_pandas_pickle_compat()
    env = lmdb.open(
        str(path), subdir=False, readonly=True, lock=False, readahead=False,
        meminit=False, max_readers=256,
    )
    try:
        with env.begin() as txn:
            pids = []
            for _, raw in txn.cursor():
                item = pickle.loads(raw)
                pids.append(str(item["pocket"]).lower())
    finally:
        env.close()
    if len(pids) != len(set(pids)):
        raise RuntimeError(f"{path}: duplicate PDB IDs")
    return pids


def _read_eligible(path: Path) -> set[str]:
    with path.open(newline="") as handle:
        rows = csv.DictReader(handle)
        eligible = {str(row["pid"]).lower() for row in rows}
    if not eligible:
        raise RuntimeError(f"{path}: no eligible IDs")
    return eligible


def _official_assignment(root: Path, identity: int) -> dict[str, str]:
    assignment: dict[str, str] = {}
    counts = {}
    for split in ("train", "val", "test"):
        filename = "valid.lmdb" if split == "val" else f"{split}.lmdb"
        pids = _read_lmdb_ids(root / f"lba_identity_{identity}" / filename)
        counts[split] = len(pids)
        for pid in pids:
            if pid in assignment:
                raise RuntimeError(f"ID{identity}: {pid} occurs in multiple splits")
            assignment[pid] = split
    if counts != OFFICIAL_COUNTS[identity]:
        raise RuntimeError(
            f"ID{identity}: official LMDB counts changed: {counts} "
            f"!= {OFFICIAL_COUNTS[identity]}"
        )
    return assignment


def _write(path: Path, assignment: dict[str, str], eligible: set[str]) -> Counter:
    rows = sorted(
        ((pid, split) for pid, split in assignment.items() if pid in eligible),
        key=lambda row: (row[1], row[0]),
    )
    counts = Counter(split for _, split in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("pid", "split"))
        writer.writerows(rows)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atom3d-root", type=Path, default=DEFAULT_ATOM3D_ROOT)
    parser.add_argument("--eligible", type=Path, default=SPLIT_DIR / "lp_edrscc_v2.csv")
    parser.add_argument("--out-dir", type=Path, default=SPLIT_DIR)
    args = parser.parse_args()

    eligible = _read_eligible(args.eligible)
    official = {identity: _official_assignment(args.atom3d_root, identity) for identity in (30, 60)}
    if set(official[30]) != set(official[60]):
        raise RuntimeError("Official ID30 and ID60 releases do not contain the same pool")

    overlap = set(official[30]) & eligible
    if len(overlap) != 2684:
        raise RuntimeError(f"official ATOM3D ∩ lp_edrscc_v2 changed: {len(overlap)} != 2684")

    for identity in (30, 60):
        out = args.out_dir / f"atom3d_lba{identity}_edrscc_v2.csv"
        counts = _write(out, official[identity], eligible)
        actual = {split: counts[split] for split in ("train", "val", "test")}
        if actual != FILTERED_COUNTS[identity]:
            raise RuntimeError(
                f"ID{identity}: filtered counts changed: {actual} "
                f"!= {FILTERED_COUNTS[identity]}"
            )
        print(f"{out}: {actual} total={sum(actual.values())}")


if __name__ == "__main__":
    main()
