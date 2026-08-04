"""make_casf_lmdb.py — build a single ProFSA LBA-format LMDB for the 214 CASF-2016 complexes.

Entry schema matches ProFSA's LBA schema exactly:
  atoms              : list[str]  ligand heavy-atom element symbols
  coordinates        : np.ndarray (1, N, 3) float  (single conformer)
  pocket_atoms       : list[str]  pocket heavy-atom element symbols
  pocket_coordinates : np.ndarray (M, 3) float
  pocket             : str  pdb id
  smi                : str  ligand SMILES
  label              : float  pK

Writes casf2016.lmdb to data/dataset/casf2016/.
Also copies dict_mol.txt / dict_pkt.txt from the edrscc dataset.
"""
import os
import sys
import pickle
import shutil

import numpy as np
import pandas as pd
import lmdb
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
PROFSA_ROOT = os.path.dirname(HERE)                             # base/profsa/
REPO = os.path.dirname(os.path.dirname(PROFSA_ROOT))           # VoxBind/
CASF_CSV = os.path.join(REPO, "voxbind", "splits", "casf2016_eval.csv")
STRUCT_BASES = [
    os.path.join(REPO, "voxbind", "dataset", "data", "pdbbind", "structures", "pbpp-2020"),
    os.path.join(REPO, "voxbind", "dataset", "data", "pdbbind", "structures", "misato_qm_built"),
]
REF_DICT_DIR = os.path.join(PROFSA_ROOT, "data", "dataset", "edrscc")
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
    if mol2 and os.path.exists(mol2):
        m = Chem.MolFromMol2File(mol2, sanitize=True) or Chem.MolFromMol2File(mol2, sanitize=False)
        if m is not None and m.GetNumConformers() > 0:
            return m
    return None


def parse_pocket(pocket_pdb):
    """Heavy protein atoms from pocket pdb -> (elements, coords[M,3])."""
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
            elems.append(elem)
            coords.append([x, y, z])
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
    lig_coords = np.array(
        [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
         for i in range(mol.GetNumAtoms())],
        dtype=np.float64,
    )
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
    env = lmdb.open(path, subdir=False, map_size=int(10e9), lock=False)
    with env.begin(write=True) as txn:
        for i, e in enumerate(entries):
            txn.put(str(i).encode(), pickle.dumps(e, protocol=pickle.HIGHEST_PROTOCOL))
    env.close()


def main():
    out_dir = os.path.join(PROFSA_ROOT, "data", "dataset", "casf2016")
    os.makedirs(out_dir, exist_ok=True)

    # Copy dictionaries from edrscc dataset
    for df in ("dict_mol.txt", "dict_pkt.txt"):
        src = os.path.join(REF_DICT_DIR, df)
        dst = os.path.join(out_dir, df)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
            print(f"Copied {df}")

    # Load CASF CSV
    casf = pd.read_csv(CASF_CSV)
    casf["pid"] = casf["pid"].astype(str).str.lower()
    print(f"CASF-2016: {len(casf)} complexes")

    entries = []
    order = []  # list of (pid, in_v2train, pK) in insertion order
    fails = {}

    for _, row in casf.iterrows():
        pid = row["pid"]
        pK = float(row["pK"])
        in_v2train = int(row["in_v2train"])
        e, status = build_entry(pid, pK)
        if e is None:
            fails[pid] = status
            print(f"FAIL {pid}: {status}")
        else:
            entries.append(e)
            order.append({"pid": pid, "in_v2train": in_v2train, "pK": pK, "idx": len(entries) - 1})

    lmdb_path = os.path.join(out_dir, "casf2016.lmdb")
    write_lmdb(entries, lmdb_path)
    print(f"\nWrote {len(entries)} entries -> {lmdb_path}")
    print(f"Failures: {fails}")

    # Save order CSV for downstream metric computation
    order_path = os.path.join(out_dir, "casf2016_order.csv")
    pd.DataFrame(order).to_csv(order_path, index=False)
    print(f"Order saved -> {order_path}")
    return out_dir


if __name__ == "__main__":
    main()
