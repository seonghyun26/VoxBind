"""
Compute train-test similarity for LP-PDBBind and Timesplit using LP-PDBBind methodology.
- Protein sim: difflib.SequenceMatcher ratio (proxy for NW)
- Ligand sim: Dice coefficient on Morgan fingerprints (radius=2, nBits=2048)
"""

import os
import sys
import pickle
import difflib
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

BASE = Path("/home/shpark/prj-ligand/targetdiff/data")
PDBBIND_BASE = BASE / "pdbbind_v2020/pocket_10_general/structures/pbpp-2020"

# ─── helpers ────────────────────────────────────────────────────────────────

def morgan_fp(smi):
    """Return Morgan fingerprint or None."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    except Exception:
        return None

def morgan_fp_from_sdf(path):
    """Return Morgan fingerprint from SDF file."""
    try:
        sup = Chem.SDMolSupplier(str(path), removeHs=True)
        for mol in sup:
            if mol is not None:
                return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    except Exception:
        pass
    return None

def seq_from_pdb(path):
    """Extract one-letter AA sequence from PDB ATOM records."""
    AA3 = {
        'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C',
        'GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
        'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
        'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
        'MSE':'M','HSD':'H','HSE':'H','HSP':'H',
    }
    seen = {}
    try:
        with open(path) as f:
            for line in f:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    res_name = line[17:20].strip()
                    chain = line[21]
                    res_seq = line[22:26].strip()
                    key = (chain, res_seq)
                    if key not in seen and res_name in AA3:
                        seen[key] = AA3[res_name]
    except Exception:
        pass
    return "".join(seen.values())

def protein_sim(a, b):
    """SequenceMatcher ratio."""
    return difflib.SequenceMatcher(None, a, b).ratio()

def ligand_sim(fp1, fp2):
    """Dice similarity."""
    return DataStructs.DiceSimilarity(fp1, fp2)

def nn_stats(test_items, train_items, sim_fn, desc=""):
    """
    For each test item, find max similarity to any train item.
    Returns array of per-test max similarities.
    """
    results = []
    n_test = len(test_items)
    for i, t in enumerate(test_items):
        if t is None:
            results.append(0.0)
            continue
        best = 0.0
        for tr in train_items:
            if tr is None:
                continue
            s = sim_fn(t, tr)
            if s > best:
                best = s
        results.append(best)
        if (i+1) % 100 == 0:
            print(f"  {desc}: {i+1}/{n_test}", flush=True)
    return np.array(results)

def report(name, prot_nn, lig_nn, prot_thresh=0.50, lig_thresh=0.99):
    above = np.sum((prot_nn > prot_thresh) & (lig_nn > lig_thresh))
    print(f"\n{'='*60}")
    print(f"Split: {name}")
    print(f"  Protein sim  — mean: {prot_nn.mean():.4f}  median: {np.median(prot_nn):.4f}  max: {prot_nn.max():.4f}")
    print(f"  Ligand sim   — mean: {lig_nn.mean():.4f}  median: {np.median(lig_nn):.4f}  max: {lig_nn.max():.4f}")
    print(f"  Pairs above threshold (prot>{prot_thresh} AND lig>{lig_thresh}): {above} / {len(prot_nn)}")

# ─── LP-PDBBind ─────────────────────────────────────────────────────────────

print("\n=== Loading LP-PDBBind ===")
lp_path = "/home/shpark/prj-ligand/Voxbind/voxbind/dataset/data/pdbbind/raw/LP_PDBBind.csv"
df = pd.read_csv(lp_path)
print(f"  Total rows: {len(df)}")
print(f"  Splits: {df['new_split'].value_counts().to_dict()}")

lp_train = df[df['new_split'] == 'train']
lp_test  = df[df['new_split'] == 'test']

# Protein sequences
lp_train_seqs = lp_train['seq'].fillna('').tolist()
lp_test_seqs  = lp_test['seq'].fillna('').tolist()

# Ligand fingerprints
print("  Computing LP train fingerprints...")
lp_train_fps = [morgan_fp(s) for s in lp_train['smiles'].fillna('')]
lp_test_fps  = [morgan_fp(s) for s in lp_test['smiles'].fillna('')]

lp_train_fps_valid = [fp for fp in lp_train_fps if fp is not None]
print(f"  Train fps valid: {len(lp_train_fps_valid)}/{len(lp_train_fps)}")

# Protein NN
print("  Computing LP protein NN similarities...")
lp_prot_nn = nn_stats(lp_test_seqs, lp_train_seqs, protein_sim, desc="LP-prot")

# Ligand NN
print("  Computing LP ligand NN similarities...")
lp_lig_nn = nn_stats(
    lp_test_fps, lp_train_fps_valid, ligand_sim, desc="LP-lig"
)

report("LP-PDBBind", lp_prot_nn, lp_lig_nn)

# ─── Timesplit ───────────────────────────────────────────────────────────────

print("\n\n=== Loading Timesplit ===")
split = torch.load(BASE / "pdbbind_v2020/pocket_10_combined/split.pt")
index = pickle.load(open(BASE / "pdbbind_v2020/pocket_10_combined/index.pkl", "rb"))

train_idx = split['train']
test_idx  = split['test']
print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")

def load_timesplit_entry(idx):
    """Returns (seq, fp) for a given index."""
    entry = index[idx]
    pocket_rel, ligand_rel = entry[0], entry[1]
    pocket_path = BASE / pocket_rel
    ligand_path = BASE / ligand_rel
    seq = seq_from_pdb(pocket_path)
    fp  = morgan_fp_from_sdf(ligand_path)
    return seq, fp

print("  Loading timesplit train entries...")
ts_train_seqs, ts_train_fps = [], []
for i, idx in enumerate(train_idx):
    seq, fp = load_timesplit_entry(idx)
    ts_train_seqs.append(seq)
    ts_train_fps.append(fp)
    if (i+1) % 2000 == 0:
        print(f"    loaded {i+1}/{len(train_idx)}", flush=True)

ts_train_fps_valid = [fp for fp in ts_train_fps if fp is not None]
print(f"  Train fps valid: {len(ts_train_fps_valid)}/{len(ts_train_fps)}")

print("  Loading timesplit test entries...")
ts_test_seqs, ts_test_fps = [], []
for i, idx in enumerate(test_idx):
    seq, fp = load_timesplit_entry(idx)
    ts_test_seqs.append(seq)
    ts_test_fps.append(fp)
    if (i+1) % 50 == 0:
        print(f"    loaded {i+1}/{len(test_idx)}", flush=True)

# Protein NN
print("  Computing Timesplit protein NN similarities...")
ts_prot_nn = nn_stats(ts_test_seqs, ts_train_seqs, protein_sim, desc="TS-prot")

# Ligand NN
print("  Computing Timesplit ligand NN similarities...")
ts_lig_nn = nn_stats(
    ts_test_fps, ts_train_fps_valid, ligand_sim, desc="TS-lig"
)

report("Timesplit", ts_prot_nn, ts_lig_nn)

# ─── Summary table ───────────────────────────────────────────────────────────
print("\n\n" + "="*60)
print("SUMMARY TABLE")
print("="*60)
print(f"{'Split':<15} {'Prot mean':<12} {'Prot med':<12} {'Prot max':<12} {'Lig mean':<12} {'Lig med':<12} {'Lig max':<12} {'Pairs>thresh'}")
for name, prot, lig in [("LP-PDBBind", lp_prot_nn, lp_lig_nn), ("Timesplit", ts_prot_nn, ts_lig_nn)]:
    above = np.sum((prot > 0.50) & (lig > 0.99))
    print(f"{name:<15} {prot.mean():<12.4f} {np.median(prot):<12.4f} {prot.max():<12.4f} {lig.mean():<12.4f} {np.median(lig):<12.4f} {lig.max():<12.4f} {above}/{len(prot)}")
