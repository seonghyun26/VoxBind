"""make_casf_data.py — build a CASF-2016 test pkl for GET/EGNN inference.

Reads structures from the same PDBbind tree as make_data.py, but uses
voxbind/splits/casf2016_eval.csv (pid, in_v2train, pK) as the manifest.
ALL 214 entries are put into a single test.pkl regardless of in_v2train.
"""
import os
import sys
import json
import pickle
import argparse

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
GET_ROOT = os.path.dirname(os.path.dirname(HERE))         # base/get/ (parent of _casf_get -> base/get)
# Actually: _casf_get is inside base/get, so parent of HERE is base/get
GET_ROOT = os.path.dirname(HERE)                          # base/get/
REPO = os.path.dirname(os.path.dirname(GET_ROOT))         # VoxBind/
sys.path.insert(0, GET_ROOT)

from data.converter.pdb_to_list_blocks import pdb_to_list_blocks
from data.converter.mol2_to_blocks import mol2_to_blocks
from data.dataset import blocks_interface, blocks_to_data

CASF_CSV = os.path.join(REPO, "voxbind", "splits", "casf2016_eval.csv")
STRUCT_BASES = [
    os.path.join(REPO, "voxbind", "dataset", "data", "pdbbind", "structures", "pbpp-2020"),
    os.path.join(REPO, "voxbind", "dataset", "data", "pdbbind", "structures", "misato_qm_built"),
]


def resolve(pid):
    for base in STRUCT_BASES:
        d = os.path.join(base, pid)
        prot = os.path.join(d, f"{pid}_protein.pdb")
        mol2 = os.path.join(d, f"{pid}_ligand.mol2")
        if os.path.exists(prot) and os.path.exists(mol2):
            return prot, mol2
    return None, None


def process_one(pid, label, interface_dist_th, fragment):
    prot, mol2 = resolve(pid)
    if prot is None:
        return None, "no_structure"
    try:
        list_blocks1 = pdb_to_list_blocks(prot)
    except Exception as e:
        return None, f"protein_parse:{type(e).__name__}"
    try:
        blocks2 = mol2_to_blocks(mol2, fragment=fragment)
    except Exception as e:
        return None, f"ligand_parse:{type(e).__name__}"
    blocks1 = []
    for b in list_blocks1:
        blocks1.extend(b)
    blocks1, _ = blocks_interface(blocks1, blocks2, interface_dist_th)
    if len(blocks1) == 0:
        return None, "no_interface"
    try:
        data = blocks_to_data(blocks1, blocks2)
    except Exception as e:
        return None, f"blocks_to_data:{type(e).__name__}"
    for k in data:
        if isinstance(data[k], np.ndarray):
            data[k] = data[k].tolist()
    return {"id": pid, "affinity": {"neglog_aff": float(label)}, "data": data}, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=os.path.join(HERE, "datasets", "casf2016"))
    ap.add_argument("--casf_csv", default=CASF_CSV)
    ap.add_argument("--fragment", action="store_true")
    ap.add_argument("--interface_dist_th", type=float, default=8.0)
    args = ap.parse_args()

    casf = pd.read_csv(args.casf_csv)
    casf["pid"] = casf["pid"].astype(str).str.lower()

    os.makedirs(args.out_dir, exist_ok=True)
    items = []
    fails = {}

    for _, row in casf.iterrows():
        pid = row["pid"]
        pk = float(row["pK"])
        item, status = process_one(pid, pk, args.interface_dist_th, args.fragment)
        if item is None:
            fails[pid] = status
            print(f"  FAIL {pid}: {status}")
        else:
            items.append(item)

    out = os.path.join(args.out_dir, "test.pkl")
    with open(out, "wb") as f:
        pickle.dump(items, f)
    print(f"Built CASF-2016 pkl: {len(items)} ok, {len(fails)} fail -> {out}")

    with open(os.path.join(args.out_dir, "fails.json"), "w") as f:
        json.dump(fails, f, indent=2)

    # also save the manifest (pid, in_v2train, pK) for scoring
    casf.to_csv(os.path.join(args.out_dir, "manifest.csv"), index=False)
    # and a list of built pids in order (for alignment check)
    built_pids = [it["id"] for it in items]
    with open(os.path.join(args.out_dir, "built_pids.json"), "w") as f:
        json.dump(built_pids, f)
    print(f"saved manifest and built_pids list")


if __name__ == "__main__":
    main()
