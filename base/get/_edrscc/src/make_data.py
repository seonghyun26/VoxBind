"""make_data.py — convert lp_edrscc_v2 PDBbind complexes into GET's .pkl block format.

Mirrors GET's scripts/data_process/process_PDBbind_benchmark.py::process_one, but reads
structures from OUR local repo tree and assigns membership/labels from the frozen split
(voxbind/splits/lp_edrscc_v2.csv) + LP_PDBBind.csv pK. Emits train/valid/test.pkl lists of
{id, affinity:{neglog_aff: pK}, data: blocks_to_data(pocket_blocks, ligand_blocks)} that the
GET PDBBindBenchmark dataset loads directly.

Run inside the `get` conda env (needs GET's data utils + biopython + rdkit).
"""
import os
import sys
import json
import pickle
import argparse

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
GET_ROOT = os.path.dirname(os.path.dirname(HERE))          # base/get/
REPO = os.path.dirname(os.path.dirname(GET_ROOT))          # VoxBind/
sys.path.insert(0, GET_ROOT)

from data.converter.pdb_to_list_blocks import pdb_to_list_blocks      # noqa: E402
from data.converter.mol2_to_blocks import mol2_to_blocks              # noqa: E402
from data.dataset import blocks_interface, blocks_to_data            # noqa: E402

SPLIT = os.path.join(REPO, "voxbind", "splits", "lp_edrscc_v2.csv")
LP = os.path.join(REPO, "voxbind", "dataset", "data", "pdbbind", "raw", "LP_PDBBind.csv")
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
    ap.add_argument("--out_dir", default=os.path.join(GET_ROOT, "datasets", "edrscc"))
    ap.add_argument("--split_csv", default=SPLIT, help="frozen split manifest (pid,split)")
    ap.add_argument("--fragment", action="store_true", help="PS fragment representation (else atom-level)")
    ap.add_argument("--interface_dist_th", type=float, default=8.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    sp = pd.read_csv(args.split_csv); sp["pid"] = sp["pid"].astype(str).str.lower()
    lp = pd.read_csv(LP).rename(columns={"Unnamed: 0": "pid", "value": "pK"})
    lp["pid"] = lp["pid"].astype(str).str.lower()
    pk = dict(zip(lp["pid"], lp["pK"]))
    split_map = dict(zip(sp["pid"], sp["split"]))

    os.makedirs(args.out_dir, exist_ok=True)
    buckets = {"train": [], "valid": [], "test": []}
    name_map = {"train": "train", "val": "valid", "test": "test"}
    fails = {}
    pids = list(sp["pid"])
    if args.limit:
        pids = pids[:args.limit]

    for i, pid in enumerate(pids):
        if pid not in pk or not np.isfinite(pk[pid]):
            fails[pid] = "no_label"; continue
        item, status = process_one(pid, pk[pid], args.interface_dist_th, args.fragment)
        if item is None:
            fails[pid] = status; continue
        buckets[name_map[split_map[pid]]].append(item)
        if (i + 1) % 500 == 0:
            print(f"  processed {i+1}/{len(pids)}  ok so far: "
                  f"{sum(len(v) for v in buckets.values())}", flush=True)

    for name in ["train", "valid", "test"]:
        out = os.path.join(args.out_dir, f"{name}.pkl")
        with open(out, "wb") as f:
            pickle.dump(buckets[name], f)
        print(f"{name}: {len(buckets[name])} -> {out}")

    from collections import Counter
    print(f"failures: {len(fails)}  {dict(Counter(fails.values()))}")
    with open(os.path.join(args.out_dir, "make_data_fails.json"), "w") as f:
        json.dump(fails, f, indent=2)


if __name__ == "__main__":
    main()
