"""casf_eval.py — train AEV-PLIG on lp_edrscc_v2 TRAIN, evaluate on CASF-2016.

HonestAffinity canonical-vs-non-train protocol:
- Train set  : lp_edrscc_v2 train split
- Test set A  : all 214 CASF-2016 complexes  ("leaky")
- Test set B  : the 124 complexes NOT in lp_edrscc_v2 train ("nontrain")

First builds graphs for any CASF pids missing from the existing pickle,
then augments the pickle (in-memory only, not saved back) and trains 3 seeds.

Writes  base/_casf/AEV.json  with leaky/nontrain mean±std per metric.
"""
import os
import sys
import json
import time
import pickle
import random
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model_defs import GATv2Net
from utils import GraphDataset, init_weights
from helpers import rmse, pearson, spearman, get_num_parameters

BASE_DIR  = os.path.dirname(HERE)                                # aevplig/
REPO_DIR  = os.path.dirname(os.path.dirname(BASE_DIR))          # VoxBind/
SPLITS_DIR = os.path.join(REPO_DIR, "voxbind", "splits")
LP_CSV    = os.path.join(REPO_DIR, "voxbind", "dataset", "data", "pdbbind", "raw", "LP_PDBBind.csv")
GRAPHS    = os.path.join(BASE_DIR, "graphs", "aevplig_edrscc_graphs.pickle")
PROC_ROOT = os.path.join(BASE_DIR, "graphs", "pyg")
CASF_OUT  = os.path.join(os.path.dirname(BASE_DIR), "_casf")   # base/_casf/

PDBBIND   = os.path.join(REPO_DIR, "voxbind", "dataset", "data", "pdbbind")
STRUCT_BASES = [
    os.path.join(PDBBIND, "structures", "pbpp-2020"),
    os.path.join(PDBBIND, "structures", "misato_qm_built"),
]
ATOM_KEYS_CSV = os.path.join(BASE_DIR, "data", "PDB_Atom_Keys.csv")


# ── graph-building helpers (verbatim from build_graphs.py) ───────────────────
def resolve_complex(pid):
    for base in STRUCT_BASES:
        d = os.path.join(base, pid)
        prot = os.path.join(d, f"{pid}_protein.pdb")
        if not os.path.exists(prot):
            continue
        mol2 = os.path.join(d, f"{pid}_ligand.mol2")
        sdf  = os.path.join(d, f"{pid}_ligand.sdf")
        if os.path.exists(mol2):
            return prot, mol2, "mol2"
        if os.path.exists(sdf):
            return prot, sdf, "sdf"
    return None, None, None


def read_ligand(ligand_path, kind):
    from rdkit import Chem
    if kind == "mol2":
        mol = Chem.MolFromMol2File(ligand_path)
    else:
        mol = next(iter(Chem.SDMolSupplier(ligand_path, sanitize=True, removeHs=False)), None)
    if mol is None:
        return None
    try:
        mol = Chem.AddHs(mol, addCoords=True)
    except Exception:
        return None
    return mol


def LoadMolasDF(mol):
    atoms = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "H":
            entry = [int(atom.GetIdx()), str(atom.GetSymbol())]
            pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
            entry += [float(f"{pos.x:.4f}"), float(f"{pos.y:.4f}"), float(f"{pos.z:.4f}")]
            atoms.append(entry)
    df = pd.DataFrame(atoms, columns=["ATOM_INDEX", "ATOM_TYPE", "X", "Y", "Z"])
    return df


def LoadPDBasDF(PDB, atom_keys):
    prot_atoms = []
    with open(PDB) as f:
        for line in f:
            if line[:4] == "ATOM":
                name = line[12:16].replace(" ", "")
                if (len(name) < 4 and name[0] != "H") or \
                   (len(name) == 4 and name[1] != "H" and name[0] != "H"):
                    prot_atoms.append([int(line[6:11]),
                                       line[17:20] + "-" + name,
                                       float(line[30:38]),
                                       float(line[38:46]),
                                       float(line[46:54])])
    df = pd.DataFrame(prot_atoms, columns=["ATOM_INDEX", "PDB_ATOM", "X", "Y", "Z"])
    df = df.merge(atom_keys, on="PDB_ATOM")[
        ["ATOM_INDEX", "ATOM_TYPE", "X", "Y", "Z"]
    ].sort_values("ATOM_INDEX").reset_index(drop=True)
    return df


def GetMolAEVs_extended(protein_path, mol, atom_keys, radial_coefs, atom_map):
    import torchani
    import torchani_mod
    import qcelemental as qcel

    Target = LoadPDBasDF(protein_path, atom_keys)
    Ligand = LoadMolasDF(mol)

    RcR, EtaR, RsR = radial_coefs
    RcA = 2.0
    Zeta = torch.tensor([1.0]); TsA = torch.tensor([1.0])
    EtaA = torch.tensor([1.0]); RsA = torch.tensor([1.0])

    distance_cutoff = RcR + 0.1
    for ax in ["X", "Y", "Z"]:
        Target = Target[Target[ax] < float(Ligand[ax].max()) + distance_cutoff]
        Target = Target[Target[ax] > float(Ligand[ax].min()) - distance_cutoff]

    Target = Target.merge(atom_map, on="ATOM_TYPE", how="left")

    mol_len = torch.tensor(len(Ligand))
    atomicnums = np.append(np.ones(mol_len) * 6, Target["ATOM_NR"])
    atomicnums = torch.tensor(atomicnums, dtype=torch.int64).unsqueeze(0)

    coords = pd.concat([Ligand[["X", "Y", "Z"]], Target[["X", "Y", "Z"]]])
    coordinates = torch.tensor(coords.values).unsqueeze(0)

    atom_symbols = [qcel.periodictable.to_symbol(i) for i in range(1, 23)]
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
        raise Exception(f"input {x} not in allowable set {allowable_set}")
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom, features=("atom_symbol", "num_heavy_atoms", "total_num_Hs",
                                  "explicit_valence", "is_aromatic", "is_in_ring")):
    fl = []
    if "atom_symbol" in features:
        fl.extend(one_of_k_encoding(atom.GetSymbol(),
                  ['F', 'N', 'Cl', 'O', 'Br', 'C', 'B', 'P', 'I', 'S']))
    if "num_heavy_atoms" in features:
        fl.append(len([x for x in atom.GetNeighbors() if x.GetSymbol() != "H"]))
    if "total_num_Hs" in features:
        fl.append(len([x for x in atom.GetNeighbors() if x.GetSymbol() == "H"]))
    if "explicit_valence" in features:
        fl.append(atom.GetExplicitValence())
    if "is_aromatic" in features:
        fl.append(1 if atom.GetIsAromatic() else 0)
    if "is_in_ring" in features:
        fl.append(1 if atom.IsInRing() else 0)
    return np.array(fl)


def mol_to_graph(mol, mol_df, aevs):
    features, heavy_atom_index, idx_to_idx = [], [], {}
    counter = 0
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "H":
            idx_to_idx[atom.GetIdx()] = counter
            aev_idx = mol_df[mol_df["ATOM_INDEX"] == atom.GetIdx()].index
            heavy_atom_index.append(atom.GetIdx())
            feature = np.append(atom_features(atom), aevs[aev_idx, :])
            features.append(feature)
            counter += 1
    edges = []
    for bond in mol.GetBonds():
        i1, i2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i1 in heavy_atom_index and i2 in heavy_atom_index:
            bt = one_of_k_encoding(bond.GetBondType(), [1, 12, 2, 3])
            bt = [float(b) for b in bt]
            edges.append([idx_to_idx[i1], idx_to_idx[i2]] + bt)
            edges.append([idx_to_idx[i2], idx_to_idx[i1]] + bt)
    df = pd.DataFrame(edges, columns=["atom1", "atom2", "single", "aromatic", "double", "triple"])
    df = df.sort_values(["atom1", "atom2"])
    return len(mol_df), features, df[["atom1", "atom2"]].values.tolist(), \
           df[["single", "aromatic", "double", "triple"]].values.tolist()


def build_missing_graphs(missing_pids):
    """Build graphs for pids not in the pickle; returns dict {pid: graph}."""
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    atom_keys = pd.read_csv(ATOM_KEYS_CSV)
    atom_map = pd.DataFrame(pd.unique(atom_keys["ATOM_TYPE"]))
    atom_map[1] = list(np.arange(len(atom_map)) + 1)
    atom_map = atom_map.rename(columns={0: "ATOM_TYPE", 1: "ATOM_NR"})

    RcR = 5.1
    EtaR = torch.tensor([19.7])
    RsR  = torch.tensor([0.80, 1.07, 1.34, 1.61, 1.88, 2.14, 2.41, 2.68,
                         2.95, 3.22, 3.49, 3.76, 4.03, 4.29, 4.56, 4.83])
    radial_coefs = [RcR, EtaR, RsR]

    built, failed = {}, []
    for pid in missing_pids:
        prot, lig, kind = resolve_complex(pid)
        if prot is None:
            print(f"  MISSING STRUCTURE: {pid}")
            failed.append(pid)
            continue
        mol = read_ligand(lig, kind)
        if mol is None:
            print(f"  RDKit failed: {pid}")
            failed.append(pid)
            continue
        try:
            mol_df, aevs = GetMolAEVs_extended(prot, mol, atom_keys, radial_coefs, atom_map)
            graph = mol_to_graph(mol, mol_df, aevs)
            if len(graph[1]) == 0 or len(graph[2]) == 0:
                print(f"  empty graph: {pid}")
                failed.append(pid)
                continue
            built[pid] = graph
            print(f"  built graph: {pid}")
        except Exception as e:
            print(f"  featurize error {pid}: {e}")
            failed.append(pid)
    return built, failed


# ── training helpers ──────────────────────────────────────────────────────────
class Cfg:
    def __init__(self, hidden_dim=256, head=3, activation_function="leaky_relu",
                 number_GNN_layers=5, mlp_dims=(1024, 512, 256)):
        self.hidden_dim = hidden_dim
        self.head = head
        self.activation_function = activation_function
        self.number_GNN_layers = number_GNN_layers
        self.mlp_dims = list(mlp_dims)


def predict(model, device, loader, y_scaler):
    model.eval()
    preds, labels, pids_out = [], [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            preds.append(out.cpu())
            labels.append(data.y.view(-1, 1).cpu())
    P = torch.cat(preds).numpy().flatten().reshape(-1, 1)
    G = torch.cat(labels).numpy().flatten().reshape(-1, 1)
    return (y_scaler.inverse_transform(G).flatten(),
            y_scaler.inverse_transform(P).flatten())


def train_epoch(model, device, loader, optimizer, loss_fn):
    model.train()
    total = 0.0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = loss_fn(out, data.y.view(-1, 1).to(device))
        loss.backward()
        optimizer.step()
        total += loss.item() * len(data.y)
    return total / len(loader.dataset)


def run_seed(seed, args, train_data, valid_data, casf_data, y_scaler, device, ckpt_dir):
    random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)

    cfg = Cfg()
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False)
    casf_loader  = DataLoader(casf_data,  batch_size=args.batch_size, shuffle=False)

    model = GATv2Net(node_feature_dim=train_data.num_node_features,
                     edge_feature_dim=train_data.num_edge_features, config=cfg)
    model.apply(init_weights)
    model.to(device)

    loss_fn   = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)

    ckpt = os.path.join(ckpt_dir, f"aevplig_casf_seed{seed}.model")
    best_pc, pcs = -1.1, []
    for epoch in range(args.epochs):
        tr  = train_epoch(model, device, train_loader, optimizer, loss_fn)
        Gv, Pv = predict(model, device, valid_loader, y_scaler)
        cur = pearson(Gv, Pv)
        pcs.append(cur)
        avg = np.mean(pcs[max(epoch - 7, 0):epoch + 1])
        if avg > best_pc:
            torch.save(model.state_dict(), ckpt)
            best_pc = avg
        if epoch % 20 == 0 or epoch == args.epochs - 1:
            print(f"  seed {seed} ep {epoch:3d}  train_mse {tr:.4f}  "
                  f"val_pc {cur:.4f}  best_avg {best_pc:.4f}")

    model.load_state_dict(torch.load(ckpt, weights_only=True))
    Gc, Pc = predict(model, device, casf_loader, y_scaler)
    print(f"  -> seed {seed}: best_val_pc {best_pc:.4f}")
    return Gc, Pc, float(best_pc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds",      type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs",     type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr",         type=float, default=0.00012291937615434127)
    ap.add_argument("--device",     type=str, default="cuda")
    ap.add_argument("--casf_csv",   type=str, default=os.path.join(SPLITS_DIR, "casf2016_eval.csv"))
    ap.add_argument("--tag",        type=str, default="")
    args = ap.parse_args()

    os.makedirs(CASF_OUT, exist_ok=True)
    ckpt_dir = os.path.join(BASE_DIR, "output", "casf_ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(PROC_ROOT, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # ── load graphs ──────────────────────────────────────────────────────────
    print(f"loading graphs from {GRAPHS}")
    with open(GRAPHS, "rb") as h:
        graphs = pickle.load(h)
    print(f"  {len(graphs)} graphs loaded")

    # ── CASF-2016 manifest ────────────────────────────────────────────────────
    casf_df = pd.read_csv(args.casf_csv)
    casf_df["pid"] = casf_df["pid"].str.lower()
    casf_pids_all  = casf_df["pid"].tolist()
    casf_pk        = dict(zip(casf_df["pid"], casf_df["pK"]))
    nontrain_mask  = casf_df["in_v2train"] == 0

    # ── build missing graphs ──────────────────────────────────────────────────
    missing = [p for p in casf_pids_all if p not in graphs]
    n_reused = len(casf_pids_all) - len(missing)
    print(f"CASF graphs: {n_reused} reused from pickle, {len(missing)} to build")
    if missing:
        print(f"  Building: {missing}")
        new_graphs, failed = build_missing_graphs(missing)
        graphs.update(new_graphs)
        if failed:
            print(f"  WARNING: could not build graphs for {failed}")

    # ── filter CASF to those with graphs ─────────────────────────────────────
    casf_ids  = [p for p in casf_pids_all if p in graphs]
    casf_ys   = [casf_pk[p] for p in casf_ids]
    # track nontrain subset by pid
    nontrain_pids = set(casf_df[nontrain_mask]["pid"])
    print(f"CASF evaluable: {len(casf_ids)}/214 "
          f"(nontrain: {sum(1 for p in casf_ids if p in nontrain_pids)})")

    # ── lp_edrscc_v2 train/val ────────────────────────────────────────────────
    sp_csv = os.path.join(SPLITS_DIR, "lp_edrscc_v2.csv")
    sp = pd.read_csv(sp_csv)
    sp["pid"] = sp["pid"].str.lower()

    lp = pd.read_csv(LP_CSV).rename(columns={"Unnamed: 0": "pdb_id", "value": "pK"})
    lp["pdb_id"] = lp["pdb_id"].str.lower()
    pk_lp = dict(zip(lp["pdb_id"], lp["pK"]))

    def members(which):
        return [p for p in sp[sp["split"] == which]["pid"]
                if p in graphs and p in pk_lp and pd.notna(pk_lp[p])]

    train_ids = members("train")
    val_ids   = members("val")
    print(f"lp_edrscc_v2 train={len(train_ids)} val={len(val_ids)}")

    TAG = "casf_eval"

    def fresh(ids, name):
        proc = os.path.join(PROC_ROOT, TAG, "processed")
        pt = os.path.join(proc, f"{name}.pt")
        if os.path.exists(pt):
            os.remove(pt)

    fresh(train_ids, f"{TAG}_train")
    fresh(val_ids,   f"{TAG}_valid")
    fresh(casf_ids,  f"{TAG}_casf")

    root = os.path.join(PROC_ROOT, TAG)
    train_data = GraphDataset(root=root, dataset=f"{TAG}_train",
                              ids=train_ids, y=[pk_lp[p] for p in train_ids],
                              graphs_dict=graphs, y_scaler=None)
    valid_data = GraphDataset(root=root, dataset=f"{TAG}_valid",
                              ids=val_ids, y=[pk_lp[p] for p in val_ids],
                              graphs_dict=graphs, y_scaler=train_data.y_scaler)
    # CASF test uses same scaler as train
    casf_data  = GraphDataset(root=root, dataset=f"{TAG}_casf",
                              ids=casf_ids, y=casf_ys,
                              graphs_dict=graphs, y_scaler=train_data.y_scaler)

    # ── 3-seed training ───────────────────────────────────────────────────────
    all_preds = []
    truth_ref = None
    t0 = time.time()

    for seed in args.seeds:
        print(f"\n=== SEED {seed} ===")
        Gc, Pc, bvpc = run_seed(seed, args,
                                train_data, valid_data, casf_data,
                                train_data.y_scaler, device, ckpt_dir)
        all_preds.append(Pc)
        truth_ref = Gc

    # ── metrics ───────────────────────────────────────────────────────────────
    casf_pid_list = casf_ids  # same order as truth_ref / all_preds

    # build index masks
    leaky_mask    = np.array([True] * len(casf_pid_list))
    nontrain_idx  = np.array([p in nontrain_pids for p in casf_pid_list])

    def compute_metrics(G, P, mask):
        Gm, Pm = G[mask], P[mask]
        return {
            "pearson":  float(pearson(Gm, Pm)),
            "spearman": float(spearman(Gm, Pm)),
            "rmse":     float(rmse(Gm, Pm)),
            "n":        int(mask.sum()),
        }

    def ms_metrics(mask):
        stats_list = [compute_metrics(truth_ref, p, mask) for p in all_preds]
        result = {}
        for key in ("pearson", "spearman", "rmse"):
            vals = np.array([s[key] for s in stats_list])
            result[key] = {"mean": float(vals.mean()), "std": float(vals.std())}
        result["n"] = int(mask.sum())
        return result

    leaky_metrics    = ms_metrics(leaky_mask)
    nontrain_metrics = ms_metrics(nontrain_idx)

    print(f"\n{'='*60}")
    print(f"CASF-2016 LEAKY    (n={leaky_metrics['n']}):")
    print(f"  Pearson  {leaky_metrics['pearson']['mean']:.4f} ± {leaky_metrics['pearson']['std']:.4f}")
    print(f"  Spearman {leaky_metrics['spearman']['mean']:.4f} ± {leaky_metrics['spearman']['std']:.4f}")
    print(f"  RMSE     {leaky_metrics['rmse']['mean']:.4f} ± {leaky_metrics['rmse']['std']:.4f}")
    print(f"CASF-2016 NONTRAIN (n={nontrain_metrics['n']}):")
    print(f"  Pearson  {nontrain_metrics['pearson']['mean']:.4f} ± {nontrain_metrics['pearson']['std']:.4f}")
    print(f"  Spearman {nontrain_metrics['spearman']['mean']:.4f} ± {nontrain_metrics['spearman']['std']:.4f}")
    print(f"  RMSE     {nontrain_metrics['rmse']['mean']:.4f} ± {nontrain_metrics['rmse']['std']:.4f}")
    print(f"wall time: {(time.time()-t0)/60:.1f} min")

    result = {
        "model": "AEV-PLIG",
        "train": "lp_edrscc_v2 train",
        "seeds": args.seeds,
        "n_graphs_reused": n_reused,
        "n_graphs_built":  len(missing),
        "failed_builds":   [p for p in missing if p not in graphs],
        "leaky":    leaky_metrics,
        "nontrain": nontrain_metrics,
        "wall_s":   round(time.time() - t0, 1),
    }

    out_json = os.path.join(CASF_OUT, f"AEV{args.tag}.json")
    with open(out_json, "w") as h:
        json.dump(result, h, indent=2)
    print(f"\nsaved -> {out_json}")

    # ── per-pid predictions CSV ───────────────────────────────────────────────
    pred_df = pd.DataFrame({
        "pid": casf_pid_list,
        "pK_truth": truth_ref,
        "in_v2train": [0 if p in nontrain_pids else 1 for p in casf_pid_list],
    })
    for i, (seed, P) in enumerate(zip(args.seeds, all_preds)):
        pred_df[f"pred_seed{seed}"] = P
    pred_df["pred_ensemble"] = np.mean(np.stack(all_preds), axis=0)
    pred_df.to_csv(os.path.join(CASF_OUT, f"AEV{args.tag}_preds.csv"), index=False)
    print(f"saved -> {os.path.join(CASF_OUT, 'AEV_preds.csv')}")


if __name__ == "__main__":
    main()
