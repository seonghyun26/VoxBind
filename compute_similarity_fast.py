"""
Fast train-test similarity for LP-PDBBind and Timesplit.
- Protein sim: difflib.SequenceMatcher (proxy for NW), parallelized with multiprocessing
- Ligand sim: Dice on Morgan fps, bulk numpy matrix (fp as uint8 packed bits)
"""

import os, sys, pickle, difflib, time
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool
import functools

import torch
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

N_WORKERS = 64
BASE = Path("/home/shpark/prj-ligand/targetdiff/data")

# ─── fingerprint helpers ─────────────────────────────────────────────────────

def morgan_fp(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None: return None
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    except: return None

def morgan_fp_from_sdf(path):
    try:
        sup = Chem.SDMolSupplier(str(path), removeHs=True)
        for mol in sup:
            if mol is not None:
                return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    except: pass
    return None

def fp_to_np(fp):
    """Convert RDKit ExplicitBitVect to uint8 numpy array (2048 bits → 256 bytes)."""
    arr = np.zeros(2048, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def dice_matrix_max(test_fps_np, train_fps_np):
    """
    Compute per-test max Dice similarity against all train fps.
    Dice(a,b) = 2*|a & b| / (|a| + |b|)
    Uses numpy matrix multiply on uint8 arrays packed as float32.
    """
    # test_fps_np: (n_test, 2048), train_fps_np: (n_train, 2048)
    test  = test_fps_np.astype(np.float32)
    train = train_fps_np.astype(np.float32)
    # dot product gives intersection counts
    inter = test @ train.T                         # (n_test, n_train)
    test_counts  = test.sum(axis=1, keepdims=True)  # (n_test, 1)
    train_counts = train.sum(axis=1, keepdims=True) # (n_train, 1)
    denom = test_counts + train_counts.T             # (n_test, n_train)
    # avoid div-by-zero
    with np.errstate(divide='ignore', invalid='ignore'):
        dice = np.where(denom > 0, 2.0 * inter / denom, 0.0)
    return dice.max(axis=1)  # (n_test,)

# ─── protein similarity (parallel) ──────────────────────────────────────────

def _prot_nn_chunk(args):
    """Worker: compute max SequenceMatcher ratio for a chunk of test seqs vs all train."""
    test_chunk, train_seqs = args
    results = []
    for ts in test_chunk:
        if not ts:
            results.append(0.0)
            continue
        best = 0.0
        sm = difflib.SequenceMatcher(None, ts, '')
        for tr in train_seqs:
            if not tr: continue
            sm.set_seq2(tr)
            r = sm.ratio()
            if r > best: best = r
        results.append(best)
    return results

def protein_nn_parallel(test_seqs, train_seqs, n_workers=N_WORKERS, desc=""):
    chunk_size = max(1, len(test_seqs) // n_workers + 1)
    chunks = [test_seqs[i:i+chunk_size] for i in range(0, len(test_seqs), chunk_size)]
    args = [(chunk, train_seqs) for chunk in chunks]
    t0 = time.time()
    print(f"  [{desc}] {len(test_seqs)} test × {len(train_seqs)} train, {len(chunks)} chunks × {n_workers} workers", flush=True)
    with Pool(n_workers) as pool:
        results = pool.map(_prot_nn_chunk, args)
    elapsed = time.time() - t0
    print(f"  [{desc}] done in {elapsed:.1f}s", flush=True)
    return np.array([v for chunk in results for v in chunk])

# ─── seq from PDB ────────────────────────────────────────────────────────────

def seq_from_pdb(path):
    AA3 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
           'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
           'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
           'MSE':'M','HSD':'H','HSE':'H','HSP':'H'}
    seen = {}
    try:
        with open(path) as f:
            for line in f:
                if line.startswith(("ATOM","HETATM")):
                    rn = line[17:20].strip()
                    key = (line[21], line[22:26].strip())
                    if key not in seen and rn in AA3:
                        seen[key] = AA3[rn]
    except: pass
    return "".join(seen.values())

def report(name, prot_nn, lig_nn, prot_thresh=0.50, lig_thresh=0.99):
    above = int(np.sum((prot_nn > prot_thresh) & (lig_nn > lig_thresh)))
    print(f"\n{'='*60}")
    print(f"Split: {name}  (n_test={len(prot_nn)})")
    print(f"  Protein sim  — mean: {prot_nn.mean():.4f}  median: {np.median(prot_nn):.4f}  max: {prot_nn.max():.4f}")
    print(f"  Ligand sim   — mean: {lig_nn.mean():.4f}  median: {np.median(lig_nn):.4f}  max: {lig_nn.max():.4f}")
    print(f"  Pairs above threshold (prot>{prot_thresh} AND lig>{lig_thresh}): {above}/{len(prot_nn)}")

# ════════════════════════════════════════════════════════════════════════════
# LP-PDBBind
# ════════════════════════════════════════════════════════════════════════════
print("\n=== LP-PDBBind ===", flush=True)
lp = pd.read_csv("/home/shpark/prj-ligand/Voxbind/voxbind/dataset/data/pdbbind/raw/LP_PDBBind.csv")
lp_train = lp[lp['new_split'] == 'train']
lp_test  = lp[lp['new_split'] == 'test']
print(f"  train={len(lp_train)}, test={len(lp_test)}", flush=True)

lp_train_seqs = lp_train['seq'].fillna('').tolist()
lp_test_seqs  = lp_test['seq'].fillna('').tolist()

# Ligand fps
print("  Computing LP fingerprints...", flush=True)
lp_train_fps_rdkit = [morgan_fp(s) for s in lp_train['smiles'].fillna('')]
lp_test_fps_rdkit  = [morgan_fp(s) for s in lp_test['smiles'].fillna('')]

lp_train_fp_np = np.array([fp_to_np(fp) for fp in lp_train_fps_rdkit if fp is not None])
lp_test_fp_list = [(fp_to_np(fp) if fp is not None else None) for fp in lp_test_fps_rdkit]

# For test entries without a valid fp, assign lig_nn=0
lp_test_fp_valid_mask = np.array([x is not None for x in lp_test_fp_list])
lp_test_fp_np = np.array([x for x in lp_test_fp_list if x is not None])

print(f"  LP train fps valid: {len(lp_train_fp_np)}, test fps valid: {lp_test_fp_valid_mask.sum()}", flush=True)

# Ligand NN (bulk matrix)
print("  Computing LP ligand NN (matrix)...", flush=True)
t0 = time.time()
lp_lig_nn_valid = dice_matrix_max(lp_test_fp_np, lp_train_fp_np)
print(f"  LP ligand done in {time.time()-t0:.1f}s", flush=True)

lp_lig_nn = np.zeros(len(lp_test_fps_rdkit))
lp_lig_nn[lp_test_fp_valid_mask] = lp_lig_nn_valid

# Protein NN (parallel)
lp_prot_nn = protein_nn_parallel(lp_test_seqs, lp_train_seqs, n_workers=N_WORKERS, desc="LP-prot")

report("LP-PDBBind", lp_prot_nn, lp_lig_nn)

# ════════════════════════════════════════════════════════════════════════════
# Timesplit
# ════════════════════════════════════════════════════════════════════════════
print("\n\n=== Timesplit ===", flush=True)
split = torch.load(BASE / "pdbbind_v2020/pocket_10_combined/split.pt")
index = pickle.load(open(BASE / "pdbbind_v2020/pocket_10_combined/index.pkl", "rb"))

train_idx = split['train']
test_idx  = split['test']
print(f"  train={len(train_idx)}, test={len(test_idx)}", flush=True)

def load_entry(idx):
    entry = index[idx]
    pocket_path = BASE / entry[0]
    ligand_path = BASE / entry[1]
    seq = seq_from_pdb(pocket_path)
    fp  = morgan_fp_from_sdf(ligand_path)
    return seq, fp

print("  Loading timesplit train entries...", flush=True)
ts_train_seqs, ts_train_fps_rdkit = [], []
for i, idx in enumerate(train_idx):
    seq, fp = load_entry(idx)
    ts_train_seqs.append(seq)
    ts_train_fps_rdkit.append(fp)
    if (i+1) % 3000 == 0:
        print(f"    loaded {i+1}/{len(train_idx)}", flush=True)

print("  Loading timesplit test entries...", flush=True)
ts_test_seqs, ts_test_fps_rdkit = [], []
for i, idx in enumerate(test_idx):
    seq, fp = load_entry(idx)
    ts_test_seqs.append(seq)
    ts_test_fps_rdkit.append(fp)

# Ligand fps
ts_train_fp_np = np.array([fp_to_np(fp) for fp in ts_train_fps_rdkit if fp is not None])
ts_test_fp_list = [(fp_to_np(fp) if fp is not None else None) for fp in ts_test_fps_rdkit]
ts_test_fp_valid_mask = np.array([x is not None for x in ts_test_fp_list])
ts_test_fp_np = np.array([x for x in ts_test_fp_list if x is not None])

print(f"  TS train fps valid: {len(ts_train_fp_np)}, test fps valid: {ts_test_fp_valid_mask.sum()}", flush=True)

# Ligand NN
print("  Computing TS ligand NN (matrix)...", flush=True)
t0 = time.time()
ts_lig_nn_valid = dice_matrix_max(ts_test_fp_np, ts_train_fp_np)
print(f"  TS ligand done in {time.time()-t0:.1f}s", flush=True)

ts_lig_nn = np.zeros(len(ts_test_fps_rdkit))
ts_lig_nn[ts_test_fp_valid_mask] = ts_lig_nn_valid

# Protein NN (parallel)
ts_prot_nn = protein_nn_parallel(ts_test_seqs, ts_train_seqs, n_workers=N_WORKERS, desc="TS-prot")

report("Timesplit", ts_prot_nn, ts_lig_nn)

# ─── Summary table ───────────────────────────────────────────────────────────
print("\n\n" + "="*70)
print("SUMMARY TABLE")
print("="*70)
hdr = f"{'Split':<15} {'Prot mean':>10} {'Prot med':>10} {'Prot max':>10} {'Lig mean':>10} {'Lig med':>10} {'Lig max':>10} {'Pairs>thr':>12}"
print(hdr)
for name, prot, lig in [("LP-PDBBind", lp_prot_nn, lp_lig_nn), ("Timesplit", ts_prot_nn, ts_lig_nn)]:
    above = int(np.sum((prot > 0.50) & (lig > 0.99)))
    print(f"{name:<15} {prot.mean():>10.4f} {np.median(prot):>10.4f} {prot.max():>10.4f} "
          f"{lig.mean():>10.4f} {np.median(lig):>10.4f} {lig.max():>10.4f} {above:>5}/{len(prot)}")

# Save raw arrays for later analysis
np.savez("/home/shpark/prj-ligand/Voxbind/sim_results.npz",
         lp_prot_nn=lp_prot_nn, lp_lig_nn=lp_lig_nn,
         ts_prot_nn=ts_prot_nn, ts_lig_nn=ts_lig_nn)
print("\nSaved raw arrays to sim_results.npz")
