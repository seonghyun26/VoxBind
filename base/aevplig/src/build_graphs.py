"""build_graphs.py — build AEV-PLIG ligand graphs for the lp_edrscc_v2 PDBbind split.

Faithful port of the AEV-PLIG reference (oxpig/AEV-PLIG generate_pdbbind_graphs.py):
the ligand is turned into a heavy-atom graph whose node features are
[15 atom descriptors ++ 352-dim radial AEV] and whose edges are covalent bonds with
a 4-dim bond-type one-hot.  The protein enters ONLY through the per-ligand-atom radial
Atomic Environment Vectors (ANI-2x radial coefs, Rc=5.1 Å over 22 PDB atom types).

Only the architecture/featurisation is taken from the paper.  Membership comes from the
frozen voxbind split (voxbind/splits/lp_edrscc_v2.csv); structures are resolved from the
LOCAL repo tree (the paper's hard-coded targetdiff paths do not exist on this server).

Output: graphs/aevplig_edrscc_graphs.pickle  ->  {pid: (c_size, features, edge_index, edge_attr)}
"""
import os
import sys
import pickle
import argparse

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from rdkit import Chem, RDLogger

import torchani
import torchani_mod
import qcelemental as qcel

RDLogger.DisableLog("rdApp.*")

# ── repo paths ───────────────────────────────────────────────────────────────
HERE      = os.path.dirname(os.path.abspath(__file__))
BASE_DIR  = os.path.dirname(HERE)                                   # base/aevplig/
REPO_DIR  = os.path.dirname(os.path.dirname(BASE_DIR))              # VoxBind/ (base/aevplig → base → VoxBind)
SPLIT_CSV = os.path.join(REPO_DIR, "voxbind", "splits", "lp_edrscc_v2.csv")
PDBBIND   = os.path.join(REPO_DIR, "voxbind", "dataset", "data", "pdbbind")
STRUCT_BASES = [
    os.path.join(PDBBIND, "structures", "pbpp-2020"),
    os.path.join(PDBBIND, "structures", "misato_qm_built"),
]
ATOM_KEYS_CSV = os.path.join(BASE_DIR, "data", "PDB_Atom_Keys.csv")


def resolve_complex(pid):
    """Return (protein_pdb, mol2_or_None, sdf_or_None) from the first local base with the protein.
    Both ligand paths are returned so read_ligand can fall back mol2 → sdf."""
    for base in STRUCT_BASES:
        d = os.path.join(base, pid)
        prot = os.path.join(d, f"{pid}_protein.pdb")
        if not os.path.exists(prot):
            continue
        mol2 = os.path.join(d, f"{pid}_ligand.mol2")
        sdf = os.path.join(d, f"{pid}_ligand.sdf")
        return prot, (mol2 if os.path.exists(mol2) else None), (sdf if os.path.exists(sdf) else None)
    return None, None, None


def read_ligand(mol2, sdf):
    """RDKit mol with explicit Hs+coords. Robust 3-tier read (obabel-built holdout ligands often
    fail the strict mol2/sanitized-SDF readers): mol2 (reference reader) → SDF sanitized →
    SDF unsanitized + best-effort sanitize (skip kekulize). Recovers the obabel-valence cases."""
    mol = None
    if mol2 and os.path.exists(mol2):
        mol = Chem.MolFromMol2File(mol2)
    if mol is None and sdf and os.path.exists(sdf):
        mol = next(iter(Chem.SDMolSupplier(sdf, sanitize=True, removeHs=False)), None)
    if mol is None and sdf and os.path.exists(sdf):
        mol = next(iter(Chem.SDMolSupplier(sdf, sanitize=False, removeHs=False)), None)
        if mol is not None:
            try:
                mol.UpdatePropertyCache(strict=False)
                Chem.SanitizeMol(
                    mol,
                    sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
                    catchErrors=True,
                )
            except Exception:
                pass
    if mol is None:
        return None
    try:
        mol = Chem.AddHs(mol, addCoords=True)
    except Exception:
        pass  # keep the heavy-atom mol; AEV featurisation can still proceed
    return mol


# ─────────────────────────────────────────────────────────────────────────────
# Below: verbatim from oxpig/AEV-PLIG generate_pdbbind_graphs.py (featurisation).
# ─────────────────────────────────────────────────────────────────────────────
def LoadMolasDF(mol):
    m = mol
    atoms = []
    for atom in m.GetAtoms():
        if atom.GetSymbol() != "H":
            entry = [int(atom.GetIdx())]
            entry.append(str(atom.GetSymbol()))
            pos = m.GetConformer().GetAtomPosition(atom.GetIdx())
            entry.append(float("{0:.4f}".format(pos.x)))
            entry.append(float("{0:.4f}".format(pos.y)))
            entry.append(float("{0:.4f}".format(pos.z)))
            atoms.append(entry)
    df = pd.DataFrame(atoms)
    df.columns = ["ATOM_INDEX", "ATOM_TYPE", "X", "Y", "Z"]
    return df


def LoadPDBasDF(PDB, atom_keys):
    prot_atoms = []
    f = open(PDB)
    for i in f:
        if i[:4] == "ATOM":
            if (len(i[12:16].replace(" ", "")) < 4 and i[12:16].replace(" ", "")[0] != "H") or \
               (len(i[12:16].replace(" ", "")) == 4 and i[12:16].replace(" ", "")[1] != "H" and i[12:16].replace(" ", "")[0] != "H"):
                prot_atoms.append([int(i[6:11]),
                                   i[17:20] + "-" + i[12:16].replace(" ", ""),
                                   float(i[30:38]),
                                   float(i[38:46]),
                                   float(i[46:54])])
    f.close()
    df = pd.DataFrame(prot_atoms, columns=["ATOM_INDEX", "PDB_ATOM", "X", "Y", "Z"])
    df = df.merge(atom_keys, left_on='PDB_ATOM', right_on='PDB_ATOM')[
        ["ATOM_INDEX", "ATOM_TYPE", "X", "Y", "Z"]].sort_values(by="ATOM_INDEX").reset_index(drop=True)
    if list(df["ATOM_TYPE"].isna()).count(True) > 0:
        print("WARNING: Protein contains unsupported atom types. Only supported atom-type pairs are counted.")
    return df


def GetMolAEVs_extended(protein_path, mol, atom_keys, radial_coefs, atom_map):
    Target = LoadPDBasDF(protein_path, atom_keys)
    Ligand = LoadMolasDF(mol)

    RcR = radial_coefs[0]
    EtaR = radial_coefs[1]
    RsR = radial_coefs[2]

    RcA = 2.0
    Zeta = torch.tensor([1.0])
    TsA = torch.tensor([1.0])
    EtaA = torch.tensor([1.0])
    RsA = torch.tensor([1.0])

    distance_cutoff = RcR + 0.1
    for i in ["X", "Y", "Z"]:
        Target = Target[Target[i] < float(Ligand[i].max()) + distance_cutoff]
        Target = Target[Target[i] > float(Ligand[i].min()) - distance_cutoff]

    Target = Target.merge(atom_map, on='ATOM_TYPE', how='left')

    mol_len = torch.tensor(len(Ligand))
    atomicnums = np.append(np.ones(mol_len) * 6, Target["ATOM_NR"])
    atomicnums = torch.tensor(atomicnums, dtype=torch.int64)
    atomicnums = atomicnums.unsqueeze(0)

    coordinates = pd.concat([Ligand[['X', 'Y', 'Z']], Target[['X', 'Y', 'Z']]])
    coordinates = torch.tensor(coordinates.values)
    coordinates = coordinates.unsqueeze(0)

    atom_symbols = []
    for i in range(1, 23):
        atom_symbols.append(qcel.periodictable.to_symbol(i))

    AEVC = torchani_mod.AEVComputer(RcR, RcA, EtaR, RsR, EtaA, Zeta, RsA, TsA, len(atom_symbols))
    SC = torchani.SpeciesConverter(atom_symbols)
    sc = SC((atomicnums, coordinates))
    aev = AEVC.forward((sc.species, sc.coordinates), mol_len)

    n = len(atom_symbols)
    n_rad_sub = len(EtaR) * len(RsR)
    indices = list(np.arange(n * n_rad_sub))

    return Ligand, aev.aevs.squeeze(0)[:mol_len, indices]


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom, features=("atom_symbol", "num_heavy_atoms", "total_num_Hs",
                                  "explicit_valence", "is_aromatic", "is_in_ring")):
    feature_list = []
    if "atom_symbol" in features:
        feature_list.extend(one_of_k_encoding(atom.GetSymbol(),
                            ['F', 'N', 'Cl', 'O', 'Br', 'C', 'B', 'P', 'I', 'S']))
    if "num_heavy_atoms" in features:
        feature_list.append(len([x.GetSymbol() for x in atom.GetNeighbors() if x.GetSymbol() != "H"]))
    if "total_num_Hs" in features:
        feature_list.append(len([x.GetSymbol() for x in atom.GetNeighbors() if x.GetSymbol() == "H"]))
    if "explicit_valence" in features:
        feature_list.append(atom.GetExplicitValence())
    if "is_aromatic" in features:
        feature_list.append(1 if atom.GetIsAromatic() else 0)
    if "is_in_ring" in features:
        feature_list.append(1 if atom.IsInRing() else 0)
    return np.array(feature_list)


def mol_to_graph(mol, mol_df, aevs):
    features = []
    heavy_atom_index = []
    idx_to_idx = {}
    counter = 0
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "H":
            idx_to_idx[atom.GetIdx()] = counter
            aev_idx = mol_df[mol_df['ATOM_INDEX'] == atom.GetIdx()].index
            heavy_atom_index.append(atom.GetIdx())
            feature = np.append(atom_features(atom), aevs[aev_idx, :])
            features.append(feature)
            counter += 1

    edges = []
    for bond in mol.GetBonds():
        idx1 = bond.GetBeginAtomIdx()
        idx2 = bond.GetEndAtomIdx()
        if idx1 in heavy_atom_index and idx2 in heavy_atom_index:
            bond_type = one_of_k_encoding(bond.GetBondType(), [1, 12, 2, 3])
            bond_type = [float(b) for b in bond_type]
            edge1 = [idx_to_idx[idx1], idx_to_idx[idx2]]
            edge1.extend(bond_type)
            edge2 = [idx_to_idx[idx2], idx_to_idx[idx1]]
            edge2.extend(bond_type)
            edges.append(edge1)
            edges.append(edge2)

    df = pd.DataFrame(edges, columns=['atom1', 'atom2', 'single', 'aromatic', 'double', 'triple'])
    df = df.sort_values(by=['atom1', 'atom2'])
    edge_index = df[['atom1', 'atom2']].to_numpy().tolist()
    edge_attr = df[['single', 'aromatic', 'double', 'triple']].to_numpy().tolist()
    return len(mol_df), features, edge_index, edge_attr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(BASE_DIR, "graphs", "aevplig_edrscc_graphs.pickle"))
    ap.add_argument("--split", default=None,
                    help="split name under voxbind/splits (e.g. holdout2019_eval); default lp_edrscc_v2")
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N pids")
    args = ap.parse_args()

    split_csv = os.path.join(REPO_DIR, "voxbind", "splits", f"{args.split}.csv") if args.split else SPLIT_CSV
    sp = pd.read_csv(split_csv)
    pids = sp["pid"].astype(str).str.lower().tolist()
    if args.limit:
        pids = pids[:args.limit]
    print(f"split pids: {len(pids)}  (from {split_csv})")

    atom_keys = pd.read_csv(ATOM_KEYS_CSV, sep=",")
    atom_map = pd.DataFrame(pd.unique(atom_keys["ATOM_TYPE"]))
    atom_map[1] = list(np.arange(len(atom_map)) + 1)
    atom_map = atom_map.rename(columns={0: "ATOM_TYPE", 1: "ATOM_NR"})

    # ANI-2x radial coefficients
    RcR = 5.1
    EtaR = torch.tensor([19.7])
    RsR = torch.tensor([0.80, 1.07, 1.34, 1.61, 1.88, 2.14, 2.41, 2.68,
                        2.95, 3.22, 3.49, 3.76, 4.03, 4.29, 4.56, 4.83])
    radial_coefs = [RcR, EtaR, RsR]

    mol_graphs = {}
    failed_read = []
    failed_featurize = []

    for pid in tqdm(pids):
        prot, mol2, sdf = resolve_complex(pid)
        if prot is None:
            failed_read.append((pid, "no_structure"))
            continue
        mol = read_ligand(mol2, sdf)
        if mol is None:
            failed_read.append((pid, "rdkit_none"))
            continue
        try:
            mol_df, aevs = GetMolAEVs_extended(prot, mol, atom_keys, radial_coefs, atom_map)
            graph = mol_to_graph(mol, mol_df, aevs)
            if len(graph[1]) == 0 or len(graph[2]) == 0:
                failed_featurize.append((pid, "empty_graph"))
                continue
            mol_graphs[pid] = graph
        except Exception as e:
            failed_featurize.append((pid, f"{type(e).__name__}:{e}"))
            continue

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as h:
        pickle.dump(mol_graphs, h, protocol=pickle.HIGHEST_PROTOCOL)

    nfeat = len(next(iter(mol_graphs.values()))[1][0]) if mol_graphs else 0
    print(f"\nbuilt graphs: {len(mol_graphs)}  node_feature_dim={nfeat}")
    print(f"failed read: {len(failed_read)}  failed featurize: {len(failed_featurize)}")
    for pid, why in (failed_read + failed_featurize)[:25]:
        print(f"  {pid}: {why}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
