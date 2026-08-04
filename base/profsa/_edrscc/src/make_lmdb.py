"""make_lmdb.py — build ProFSA LBA-format LMDBs for the lp_edrscc_v2 split.

Each entry matches ProFSA's LBA schema exactly:
  atoms              : list[str]  ligand heavy-atom element symbols
  coordinates        : np.ndarray (1, N, 3) float  (single conformer)
  pocket_atoms       : list[str]  pocket heavy-atom element symbols
  pocket_coordinates : np.ndarray (M, 3) float
  pocket             : str  pdb id
  smi                : str  ligand SMILES
  label              : float  pK
The loader removes H, crops pocket to 256 nearest, and normalizes coords itself.

Pocket = our 10 A {pid}_pocket.pdb heavy atoms (the model crops to a 256-atom patch).
Writes train/valid/test.lmdb (subdir=False) + copies dict_mol.txt / dict_pkt.txt.
"""
import os
import sys
import pickle
import argparse

import numpy as np
import pandas as pd
import lmdb
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
PROFSA_ROOT = os.path.dirname(os.path.dirname(HERE))           # base/profsa/
REPO = os.path.dirname(os.path.dirname(PROFSA_ROOT))           # VoxBind/
SPLIT = os.path.join(REPO, "voxbind", "splits", "lp_edrscc_v2.csv")
LP = os.path.join(REPO, "voxbind", "dataset", "data", "pdbbind", "raw", "LP_PDBBind.csv")
STRUCT_BASES = [
    os.path.join(REPO, "voxbind", "dataset", "data", "pdbbind", "structures", "pbpp-2020"),
    os.path.join(REPO, "voxbind", "dataset", "data", "pdbbind", "structures", "misato_qm_built"),
]
REF_DICT_DIR = os.path.join(PROFSA_ROOT, "data", "dataset", "lba_identity_30")
# pocket atom elements the pocket dict supports (others dropped)
POCKET_ELEMS = {"C", "N", "O", "S", "H"}


def resolve(pid):
    for base in STRUCT_BASES:
        d = os.path.join(base, pid)
        pocket = os.path.join(d, f"{pid}_pocket.pdb")
        sdf = os.path.join(d, f"{pid}_ligand.sdf")
        mol2 = os.path.join(d, f"{pid}_ligand.mol2")
        if os.path.exists(pocket) and (os.path.exists(sdf) or os.path.exists(mol2)):
            return pocket, sdf, mol2
    return None, None, None


def load_ligand(sdf, mol2):
    if os.path.exists(sdf):
        m = next(iter(Chem.SDMolSupplier(sdf, sanitize=True, removeHs=True)), None)
        if m is None:
            m = next(iter(Chem.SDMolSupplier(sdf, sanitize=False, removeHs=True)), None)
            if m is not None:
                try:
                    Chem.SanitizeMol(m)
                except Exception:
                    pass
        if m is not None and m.GetNumConformers() > 0:
            return m
    if os.path.exists(mol2):
        m = Chem.MolFromMol2File(mol2, sanitize=True) or Chem.MolFromMol2File(mol2, sanitize=False)
        if m is not None and m.GetNumConformers() > 0:
            return m
    return None


def parse_pocket(pocket_pdb):
    """Heavy protein atoms from the pocket pdb -> (elements, coords[M,3])."""
    elems, coords = [], []
    with open(pocket_pdb) as f:
        for line in f:
            if line[:4] != "ATOM" and line[:6] != "HETATM":
                continue
            name = line[12:16].strip()
            if name.startswith("H") or (len(name) > 1 and name[0].isdigit() and name[1] == "H"):
                continue
            elem = line[76:78].strip() or name[0]
            elem = elem[0].upper() + elem[1:].lower() if len(elem) > 1 else elem.upper()
            if elem == "H" or elem not in POCKET_ELEMS:
                continue
            try:
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            except ValueError:
                continue
            elems.append(elem); coords.append([x, y, z])
    return elems, np.array(coords, dtype=np.float64)


def build_entry(pid, pK):
    pocket, sdf, mol2 = resolve(pid)
    if pocket is None:
        return None, "no_structure"
    mol = load_ligand(sdf, mol2)
    if mol is None:
        return None, "ligand_none"
    conf = mol.GetConformer()
    lig_atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    lig_coords = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
                           for i in range(mol.GetNumAtoms())], dtype=np.float64)
    if len(lig_atoms) == 0:
        return None, "empty_ligand"
    poc_atoms, poc_coords = parse_pocket(pocket)
    if len(poc_atoms) == 0:
        return None, "empty_pocket"
    try:
        smi = Chem.MolToSmiles(mol)
    except Exception:
        smi = ""
    entry = {
        "atoms": list(lig_atoms),
        "coordinates": lig_coords.reshape(1, -1, 3),
        "pocket_atoms": list(poc_atoms),
        "pocket_coordinates": poc_coords,
        "pocket": pid,
        "smi": smi,
        "label": float(pK),
    }
    return entry, "ok"


def write_lmdb(entries, path):
    if os.path.exists(path):
        os.remove(path)
    env = lmdb.open(path, subdir=False, map_size=int(50e9), lock=False)
    with env.begin(write=True) as txn:
        for i, e in enumerate(entries):
            txn.put(str(i).encode(), pickle.dumps(e, protocol=pickle.HIGHEST_PROTOCOL))
    env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=os.path.join(PROFSA_ROOT, "data", "dataset", "edrscc"))
    ap.add_argument("--split_csv", default=SPLIT, help="frozen split manifest (pid,split)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    sp = pd.read_csv(args.split_csv); sp["pid"] = sp["pid"].astype(str).str.lower()
    lp = pd.read_csv(LP).rename(columns={"Unnamed: 0": "pid", "value": "pK"})
    lp["pid"] = lp["pid"].astype(str).str.lower()
    pk = dict(zip(lp["pid"], lp["pK"]))
    split_map = dict(zip(sp["pid"], sp["split"]))

    os.makedirs(args.out_dir, exist_ok=True)
    # copy dictionaries
    for df in ("dict_mol.txt", "dict_pkt.txt"):
        src = os.path.join(REF_DICT_DIR, df)
        if os.path.exists(src):
            import shutil
            shutil.copy(src, os.path.join(args.out_dir, df))

    buckets = {"train": [], "valid": [], "test": []}
    name_map = {"train": "train", "val": "valid", "test": "test"}
    fails = {}
    pids = list(sp["pid"])
    if args.limit:
        pids = pids[:args.limit]

    for i, pid in enumerate(pids):
        if pid not in pk or not np.isfinite(pk[pid]):
            fails[pid] = "no_label"; continue
        e, status = build_entry(pid, pk[pid])
        if e is None:
            fails[pid] = status; continue
        buckets[name_map[split_map[pid]]].append(e)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(pids)} ok={sum(len(v) for v in buckets.values())}", flush=True)

    for name in ("train", "valid", "test"):
        write_lmdb(buckets[name], os.path.join(args.out_dir, f"{name}.lmdb"))
        print(f"{name}: {len(buckets[name])} -> {name}.lmdb")
    from collections import Counter
    print(f"failures {len(fails)}: {dict(Counter(fails.values()))}")


if __name__ == "__main__":
    main()
