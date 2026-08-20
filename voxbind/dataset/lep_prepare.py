"""lep_prepare.py — extract ATOM3D LEP (split-by-protein) LMDB → portable .pt.

Ligand Efficacy Prediction (LEP), the ATOM3D task used in the CheapNet paper
(OpenReview A1HhtITVEi, §Ligand Efficacy Prediction). Each example is a small
molecule presented in TWO protein conformations — an *active* state and an
*inactive* state — and the binary task is whether the molecule is an ACTIVATOR
(label 'A') of the protein's function or not ('I'). Metrics: AUROC / AUPRC.

This script only reads the LMDB and dumps a numpy-only payload so the heavy
featurisation (voxelise + frozen-encoder) can run in the `voxbind` env, which
lacks lmdb/atom3d. RUN THIS IN AN ENV WITH lmdb + pandas (e.g. `get`):

    conda run -n get python dataset/lep_prepare.py

Output: dataset/data/lep/lep_complexes.pt
    { split_name: [ {id, label(int), active{elem,xyz,chain}, inactive{...}} ... ] }

Ligand atoms are chain == 'L' (ATOM3D convention); everything else is protein.
Pocket selection (≤6 Å) is deferred to the featuriser so the raw payload stays
faithful to the source.
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent                       # voxbind/dataset
REPO = HERE.parent.parent                                    # VoxBind
# Local atom3d source (vendored in the CheapNet baseline) — no pip install needed.
sys.path.insert(0, str(REPO / "base" / "cheapnet" / "atom3d"))

from atom3d.datasets import LMDBDataset  # noqa: E402

DATA_DIR = HERE / "data" / "lep"
LMDB_ROOT = DATA_DIR / "raw" / "split-by-protein" / "data"
OUT_PT = DATA_DIR / "lep_complexes.pt"


def _atoms_payload(df):
    """DataFrame (ATOM3D 'atoms') → dict of numpy arrays (element, xyz, chain)."""
    return {
        "elem": df["element"].astype(str).str.strip().to_numpy(),
        "xyz": df[["x", "y", "z"]].to_numpy(dtype=np.float32),
        "chain": df["chain"].astype(str).to_numpy(),
    }


def main():
    if not LMDB_ROOT.exists():
        raise SystemExit(f"LMDB not found at {LMDB_ROOT} — download/extract the tarball first.")
    out = {}
    for split in ("train", "val", "test"):
        ds = LMDBDataset(str(LMDB_ROOT / split))
        rows = []
        for i in range(len(ds)):
            item = ds[i]
            active, inactive = item["atoms_active"], item["atoms_inactive"]
            label_raw = item["label"]
            rid = str(item.get("id", item.get("ensemble", i)))
            rows.append({
                "id": rid,
                "label": int(label_raw == "A"),          # activator = 1 (CheapNet convention)
                "label_raw": str(label_raw),
                "active": _atoms_payload(active),
                "inactive": _atoms_payload(inactive),
            })
        out[split] = rows
        n_pos = sum(r["label"] for r in rows)
        print(f"[{split}] n={len(rows)}  pos(activator)={n_pos}  neg={len(rows) - n_pos}")

    OUT_PT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, OUT_PT)
    print(f"wrote {OUT_PT}  ({sum(len(v) for v in out.values())} complexes)")


if __name__ == "__main__":
    main()
