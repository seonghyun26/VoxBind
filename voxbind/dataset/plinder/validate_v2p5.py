#!/usr/bin/env python3
"""Validate the frozen PLINDER-v2.5 CL3-test-ID30 loader contract."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
VOX = ROOT / "voxbind"
DATA = VOX / "dataset/data"
SPLIT = VOX / "splits/plinder/v2p5"
RESAMPLE = DATA / "pretrain/xray_resample_plinder_v2p5_perelem"

EXPECTED = {
    "manifest_positions": 112733,
    "available_positions": 96292,
    "excluded_positions": 16441,
    "removed_unique_pdb": 7765,
    "witness_rows": 7765,
    "selection_rows": 97423,
    "selection_unique_pdb": 34068,
    "available_removed_overlap": 0,
    "box_is_hardlink": True,
}
SUBSET_N = 96192
VAL_N = 100


def validate_metadata() -> dict:
    contract = json.loads((RESAMPLE / "loader_contract.json").read_text())
    with np.load(RESAMPLE / "train_manifest.npz", allow_pickle=False) as manifest:
        pdb_ids = pd.Series(manifest["pdb_id"].astype(str)).str.lower()
        available = manifest["ok"].astype(bool)
        assert np.array_equal(available, manifest["cl3test_id30_keep"].astype(bool))
    removed = set((SPLIT / "cl3test_id30_removed_pdbs.txt").read_text().split())
    witnesses = pd.read_csv(SPLIT / "cl3test_id30_matches.tsv", sep="\t")
    selection = pd.read_csv(SPLIT / "plinder_selected.csv")
    selection["entry_pdb_id"] = selection.entry_pdb_id.astype(str).str.lower()
    source_box = DATA / "pretrain/xray_resample_plinder_v2_perelem/box116.dat"
    target_box = RESAMPLE / "box116.dat"
    checks = {
        "manifest_positions": len(pdb_ids),
        "available_positions": int(available.sum()),
        "excluded_positions": int((~available).sum()),
        "removed_unique_pdb": len(removed),
        "witness_rows": len(witnesses),
        "selection_rows": len(selection),
        "selection_unique_pdb": int(selection.entry_pdb_id.nunique()),
        "available_removed_overlap": len(set(pdb_ids[available]) & removed),
        "box_is_hardlink": os.path.samefile(source_box, target_box),
    }
    if checks != EXPECTED:
        raise RuntimeError(f"v2.5 validation drift: expected {EXPECTED}, got {checks}")
    if contract["subset_n"] + contract["subset_val_n"] != checks["available_positions"]:
        raise RuntimeError("train/validation contract does not consume every retained position")
    return checks


def make_dataset(split: str):
    from voxbind.dataset.crossdocked_density_box import DatasetCrossDockedDensityBox

    return DatasetCrossDockedDensityBox(
        box_path=str(RESAMPLE / "box116.dat"),
        resample_dir=str(RESAMPLE),
        data_dir=str(DATA),
        data_file="pretrain/data_train_plinder_v2_perelem.pt",
        split=split,
        aug=False,
        ligand_radius=-1,
        pocket_radius=-1,
        n_lig_ch=8,
        n_poc_ch=4,
        max_len=30,
        subset_n=SUBSET_N,
        subset_xray_only=True,
        subset_val_n=VAL_N,
        return_gradmag=True,
        cache_size=1,
    )


def loader_smoke() -> dict:
    output = {}
    for split, expected_n in (("train", SUBSET_N), ("val", VAL_N)):
        dataset = make_dataset(split)
        if len(dataset) != expected_n:
            raise RuntimeError(f"{split} length {len(dataset)} != {expected_n}")
        sample = dataset[len(dataset) - 1]
        if not bool(sample["xray_available"]):
            raise RuntimeError(f"{split} smoke sample unexpectedly lacks density")
        output[split] = {
            "n": len(dataset),
            "density_shape": list(sample["xray_density"].shape),
            "gradmag_shape": list(sample["xray_gradmag"].shape),
            "ligand_id": str(sample["ligand"]["id"]),
        }
        del sample, dataset
        gc.collect()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loader-smoke", action="store_true")
    args = parser.parse_args()
    result = {"metadata": validate_metadata()}
    if args.loader_smoke:
        result["loader"] = loader_smoke()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
