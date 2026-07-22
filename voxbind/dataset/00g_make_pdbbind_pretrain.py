#!/usr/bin/env python
"""00g_make_pdbbind_pretrain.py — PDBbind TRAIN-split density complexes as MAE pretrain data.

Repackages PDBbind experimental 2Fo-Fc density complexes into the exact layout
`DatasetCrossDockedXray` consumes, so they can be ConcatDataset'd onto the
CrossDocked pretraining set (combine experimental density across datasets).

Pool: LP_PDBBind `new_split == 'train'` ∩ has voxels_v5/density/{pid}.npy.
  LEAKAGE GUARD: the affinity probe tests on new_split=='test' (1310) and tunes
  on 'val' (580); both are EXCLUDED here (asserted), so pretraining never sees a
  probe-eval complex.

Outputs (mirror data_train.pt + xray_crops_aligned_v5/):
  dataset/data/pdbbind_pretrain/data_train.pt              list[(pocket,ligand)] CD schema
  dataset/data/pdbbind_pretrain/crops_v5/train/{i:06d}.npy density crop (symlink → voxels_v5/density/{pid})
  dataset/data/pdbbind_pretrain/crops_v5/train_available.npy  all-True (len = n_pids)
  dataset/data/pdbbind_pretrain/crops_v5/stats.json           symlink → voxels_v5/stats.json
  dataset/data/pdbbind_pretrain/pids_train.txt                index→pid manifest

The pocket/ligand atom parsing reuses 01b's parse_pocket_pdb / parse_ligand_sdf
(same code that built the probe voxels) so atom typing + centroid are identical.

    cd voxbind && /opt/conda/envs/voxbind/bin/python dataset/00g_make_pdbbind_pretrain.py
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
PDBB = HERE / "data" / "pdbbind"
STRUCT = PDBB / "structures" / "pbpp-2020"
DENS = PDBB / "voxels_v5" / "density"
STATS = PDBB / "voxels_v5" / "stats.json"
LP = PDBB / "raw" / "LP_PDBBind.csv"
OUT = HERE / "data" / "pdbbind_pretrain"
CROPS = OUT / "crops_v5"

# Reuse 01b's parsers (module name starts with a digit → load via importlib).
_spec = importlib.util.spec_from_file_location("_p01b", HERE / "01b_pdbbind_preprocess.py")
_p01b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_p01b)
parse_pocket_pdb = _p01b.parse_pocket_pdb
parse_ligand_sdf = _p01b.parse_ligand_sdf


def train_pids() -> list[str]:
    df = pd.read_csv(LP).rename(columns={"Unnamed: 0": "pdb_id"})
    df["pdb_id"] = df["pdb_id"].str.lower()
    have = {p.stem for p in DENS.glob("*.npy")}
    train = df[(df["new_split"] == "train") & (df["pdb_id"].isin(have))]["pdb_id"].tolist()
    # Leakage guard: NONE of the chosen pids may be in val/test.
    bad = set(df[df["new_split"].isin(["val", "test"])]["pdb_id"]) & set(train)
    assert not bad, f"LEAKAGE: {len(bad)} train pids also in val/test: {list(bad)[:5]}"
    return sorted(train)


def build_one(pid: str):
    """Return (pocket_dict, ligand_dict) in CrossDocked schema, or None on failure."""
    lig_sdf = STRUCT / pid / f"{pid}_ligand.sdf"
    poc_pdb = STRUCT / pid / f"{pid}_pocket.pdb"
    if not (lig_sdf.exists() and poc_pdb.exists()):
        return None
    try:
        lig_coords, lig_ch, _heavy_centroid, _unsup, _lig_rad = parse_ligand_sdf(lig_sdf)
        poc_coords, poc_ch, _unsupp = parse_pocket_pdb(poc_pdb)
    except Exception:
        return None
    if lig_coords.numel() == 0 or poc_coords.numel() == 0:
        return None
    # max_len = ligand bounding-box max side (matches DatasetCrossDockedXray._filter_by_size,
    # which drops ligands with max_len > 30 Å). PDBbind ligands are ~all < 30.
    max_len = float((lig_coords.max(0).values - lig_coords.min(0).values).max())
    pocket = {"id": f"pdbbind/{pid}_pocket.pdb",
              "coords": poc_coords.float(), "atoms_channel": poc_ch.long()}
    ligand = {"id": f"pdbbind/{pid}_ligand.sdf",
              "coords": lig_coords.float(), "atoms_channel": lig_ch.long(), "max_len": max_len}
    return pocket, ligand


def main():
    import random as _random
    pids = train_pids()
    print(f"[00g] PDBbind train ∩ density-available: {len(pids)} pids (val/test excluded)")
    (CROPS / "train").mkdir(parents=True, exist_ok=True)

    # Match DatasetCrossDockedXray._filter_by_size(max_len=30): pre-drop oversize
    # ligands HERE so the dataset's later re-indexing filter is a no-op (keeps the
    # position↔crop mapping intact).
    MAX_LEN = 30
    data, kept_pids, n_skip, n_big = [], [], 0, 0
    for pid in pids:
        rec = build_one(pid)
        if rec is None:
            n_skip += 1
            continue
        if rec[1]["max_len"] > MAX_LEN:
            n_big += 1
            continue
        data.append(rec)
        kept_pids.append(pid)

    # data_train.pt is saved in CANONICAL (sorted-pid) order — the dataset itself
    # applies `random.Random(1234).shuffle(data)` on load (crossdocked_xray.py:399).
    torch.save(data, OUT / "data_train.pt")
    np.save(CROPS / "train_available.npy", np.ones(len(data), dtype=bool))

    # CRITICAL: 00b lays crops in the POST-shuffle order (00b:111 shuffle + 00b:294
    # enumerate), and the dataset replays the same Random(1234) shuffle. So crop
    # {i}.npy must be the density of the complex that lands at position i AFTER the
    # shuffle. Replicate that permutation (it depends only on len + seed, not items)
    # so atoms↔density stay aligned. Getting this wrong = silent garbage pretraining.
    order = list(range(len(data)))
    _random.Random(1234).shuffle(order)          # order[i] = canonical idx now at pos i
    for i, orig in enumerate(order):
        link = CROPS / "train" / f"{i:06d}.npy"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(os.path.relpath(DENS / f"{kept_pids[orig]}.npy", link.parent))
    # stats.json symlink (loader only needs it when normalize=true; we run normalize=false)
    sp = CROPS / "stats.json"
    if sp.is_symlink() or sp.exists():
        sp.unlink()
    sp.symlink_to(os.path.relpath(STATS, CROPS))
    (OUT / "pids_train.txt").write_text("\n".join(f"{i}\t{p}" for i, p in enumerate(kept_pids)) + "\n")
    print(f"[00g] wrote {len(data)} complexes (skipped {n_skip} parse-fail, {n_big} max_len>{MAX_LEN}) → {OUT}")
    print(f"      data_train.pt, crops_v5/train/*.npy, train_available.npy, stats.json")


if __name__ == "__main__":
    main()
