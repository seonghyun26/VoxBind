"""train_edrscc.py — train AEV-PLIG (GATv2Net) on the frozen lp_edrscc_v2 PDBbind split.

Architecture-only reproduction of AEV-PLIG (Valsson et al., Comms Chem 2025):
  - graphs built by build_graphs.py (15 atom desc ++ 352 radial-AEV node features,
    4-dim bond-type edges)
  - 5x GATv2Conv(hidden=256, heads=3) + BatchNorm + LeakyReLU -> max||mean pool
    -> MLP 1536->1024->512->256->1
  - MSE on StandardScaler-normalised pK, Adam lr 1.23e-4, batch 128, 200 ep,
    best ckpt by 8-epoch rolling val-Pearson  (all paper defaults)

Membership + bucket come from voxbind/splits/lp_edrscc_v2.csv (hash-stamped).
pK labels come from voxbind/dataset/data/pdbbind/raw/LP_PDBBind.csv.

Reports per-seed test Pearson/Spearman/RMSE (mean+/-std over seeds) AND the
seed-ensemble (mean of per-seed predictions), mirroring the paper's 10-model
ensemble at the chosen --seeds count.
"""
import os
import sys
import json
import time
import random
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model_defs import GATv2Net                         # noqa: E402
from utils import GraphDataset, init_weights            # noqa: E402
from helpers import rmse, pearson, spearman, get_num_parameters  # noqa: E402

BASE_DIR = os.path.dirname(HERE)
# VoxBind/ is two levels up from src/: src -> aevplig -> base -> VoxBind
REPO_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
SPLITS_DIR = os.path.join(REPO_DIR, "voxbind", "splits")
LP_CSV = os.path.join(REPO_DIR, "voxbind", "dataset", "data", "pdbbind", "raw", "LP_PDBBind.csv")
GRAPHS = os.path.join(BASE_DIR, "graphs", "aevplig_edrscc_graphs.pickle")
PROC_ROOT = os.path.join(BASE_DIR, "graphs", "pyg")     # InMemoryDataset root
RESULTS = os.path.join(BASE_DIR, "results")


def split_hash(pids):
    """sha256 over the lowercased pid set, matching voxbind.splits.content_hash."""
    import hashlib
    uniq = sorted({str(p).strip().lower() for p in pids})
    return hashlib.sha256("\n".join(uniq).encode()).hexdigest()


def load_split_and_labels(split_name):
    split_csv = os.path.join(SPLITS_DIR, f"{split_name}.csv")
    if not os.path.exists(split_csv):
        raise FileNotFoundError(f"split CSV not found: {split_csv}")
    sp = pd.read_csv(split_csv)
    sp["pid"] = sp["pid"].astype(str).str.lower()
    lp = pd.read_csv(LP_CSV).rename(columns={"Unnamed: 0": "pdb_id", "value": "pK"})
    lp["pdb_id"] = lp["pdb_id"].astype(str).str.lower()
    pk = dict(zip(lp["pdb_id"], lp["pK"]))
    return sp, pk


class Cfg:
    def __init__(self, hidden_dim, head, activation_function,
                 number_GNN_layers=5, mlp_dims=(1024, 512, 256)):
        self.hidden_dim = hidden_dim
        self.head = head
        self.activation_function = activation_function
        self.number_GNN_layers = number_GNN_layers
        self.mlp_dims = list(mlp_dims)


def predict(model, device, loader, y_scaler):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            preds.append(out.cpu())
            labels.append(data.y.view(-1, 1).cpu())
    P = torch.cat(preds, 0).numpy().flatten().reshape(-1, 1)
    G = torch.cat(labels, 0).numpy().flatten().reshape(-1, 1)
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


def run_seed(seed, args, data, device, ckpt_dir):
    random.seed(seed)
    torch.manual_seed(int(seed))
    np.random.seed(seed)

    train_data, valid_data, test_data, y_scaler = data
    cfg = Cfg(args.hidden_dim, args.head, args.activation_function,
              number_GNN_layers=args.number_GNN_layers, mlp_dims=args.mlp_dims)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    model = GATv2Net(node_feature_dim=train_data.num_node_features,
                     edge_feature_dim=train_data.num_edge_features, config=cfg)
    model.apply(init_weights)
    model.to(device)
    if seed == args.seeds[0]:
        print(f"  node_features={train_data.num_node_features} "
              f"edge_features={train_data.num_edge_features} params={get_num_parameters(model):,}")

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)

    ckpt = os.path.join(ckpt_dir, f"aevplig_edrscc_seed{seed}.model")
    best_pc, pcs = -1.1, []
    for epoch in range(args.epochs):
        tr = train_epoch(model, device, train_loader, optimizer, loss_fn)
        G, P = predict(model, device, valid_loader, y_scaler)
        cur = pearson(G, P)
        pcs.append(cur)
        avg = np.mean(pcs[max(epoch - 7, 0):epoch + 1])
        if avg > best_pc:
            torch.save(model.state_dict(), ckpt)
            best_pc = avg
        if epoch % 20 == 0 or epoch == args.epochs - 1:
            print(f"  seed {seed} ep {epoch:3d}  train_mse {tr:.4f}  val_pc {cur:.4f}  best_avg {best_pc:.4f}")

    model.load_state_dict(torch.load(ckpt))
    Gt, Pt = predict(model, device, test_loader, y_scaler)
    return {
        "seed": seed,
        "test_pearson": float(pearson(Gt, Pt)),
        "test_spearman": float(spearman(Gt, Pt)),
        "test_rmse": float(rmse(Gt, Pt)),
        "best_val_pc": float(best_pc),
    }, Gt, Pt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, default="lp_edrscc_v2",
                    help="name of split in voxbind/splits/<split>.csv; "
                         "default: lp_edrscc_v2 (backward-compatible)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--head", type=int, default=3)
    ap.add_argument("--number_GNN_layers", type=int, default=5)
    ap.add_argument("--mlp_dims", type=int, nargs="+", default=[1024, 512, 256])
    ap.add_argument("--lr", type=float, default=0.00012291937615434127)
    ap.add_argument("--activation_function", type=str, default="leaky_relu")
    ap.add_argument("--tag", type=str, default=None,
                    help="output tag; defaults to --split value")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()
    if args.tag is None:
        args.tag = args.split

    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(PROC_ROOT, exist_ok=True)
    ckpt_dir = os.path.join(BASE_DIR, "output", "trained_models")
    os.makedirs(ckpt_dir, exist_ok=True)

    import pickle
    with open(GRAPHS, "rb") as h:
        graphs = pickle.load(h)
    print(f"graphs loaded: {len(graphs)}")

    sp, pk = load_split_and_labels(args.split)
    # keep only complexes with a graph AND a pK label
    def members(which):
        ids = [p for p in sp[sp["split"] == which]["pid"] if p in graphs and p in pk and pd.notna(pk[p])]
        return ids
    train_ids = members("train")
    val_ids = members("val")
    test_ids = members("test")
    print(f"usable train/val/test: {len(train_ids)}/{len(val_ids)}/{len(test_ids)}")
    print(f"test split hash: {split_hash(test_ids)}")

    # fresh InMemoryDataset cache per run-tag (avoid stale .pt collisions)
    def build(ids, name):
        proc = os.path.join(PROC_ROOT, args.tag, "processed")
        pt = os.path.join(proc, f"{name}.pt")
        if os.path.exists(pt):
            os.remove(pt)
        return ids

    train_ids = build(train_ids, f"{args.tag}_train")
    val_ids = build(val_ids, f"{args.tag}_valid")
    test_ids = build(test_ids, f"{args.tag}_test")

    root = os.path.join(PROC_ROOT, args.tag)
    train_data = GraphDataset(root=root, dataset=f"{args.tag}_train",
                              ids=train_ids, y=[pk[p] for p in train_ids],
                              graphs_dict=graphs, y_scaler=None)
    valid_data = GraphDataset(root=root, dataset=f"{args.tag}_valid",
                              ids=val_ids, y=[pk[p] for p in val_ids],
                              graphs_dict=graphs, y_scaler=train_data.y_scaler)
    test_data = GraphDataset(root=root, dataset=f"{args.tag}_test",
                             ids=test_ids, y=[pk[p] for p in test_ids],
                             graphs_dict=graphs, y_scaler=train_data.y_scaler)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    data = (train_data, valid_data, test_data, train_data.y_scaler)

    per_seed, all_preds = [], []
    truth_ref = None
    t0 = time.time()
    for seed in args.seeds:
        res, Gt, Pt = run_seed(seed, args, data, device, ckpt_dir)
        per_seed.append(res)
        all_preds.append(Pt)
        truth_ref = Gt
        print(f"  -> seed {seed}: test_pc {res['test_pearson']:.4f} "
              f"test_sp {res['test_spearman']:.4f} test_rmse {res['test_rmse']:.4f}")

    ens_pred = np.mean(np.stack(all_preds, 0), axis=0)
    ensemble = {
        "test_pearson": float(pearson(truth_ref, ens_pred)),
        "test_spearman": float(spearman(truth_ref, ens_pred)),
        "test_rmse": float(rmse(truth_ref, ens_pred)),
    }

    def ms(key):
        v = np.array([r[key] for r in per_seed])
        return float(v.mean()), float(v.std())

    summary = {
        "model": "AEV-PLIG (GATv2Net, architecture-only)",
        "split": args.split,
        "n_train": len(train_ids), "n_val": len(val_ids), "n_test": len(test_ids),
        "test_split_hash": split_hash(test_ids),
        "seeds": list(args.seeds),
        "epochs": args.epochs, "lr": args.lr,
        "hidden_dim": args.hidden_dim, "head": args.head,
        "number_GNN_layers": args.number_GNN_layers, "mlp_dims": list(args.mlp_dims),
        "n_params": int(get_num_parameters(GATv2Net(
            node_feature_dim=train_data.num_node_features,
            edge_feature_dim=train_data.num_edge_features,
            config=Cfg(args.hidden_dim, args.head, args.activation_function,
                       number_GNN_layers=args.number_GNN_layers, mlp_dims=args.mlp_dims)))),
        "per_seed": per_seed,
        "per_seed_mean_std": {
            "test_pearson": ms("test_pearson"),
            "test_spearman": ms("test_spearman"),
            "test_rmse": ms("test_rmse"),
        },
        "ensemble": ensemble,
        "wall_clock_s": time.time() - t0,
    }

    out_json = os.path.join(RESULTS, f"aevplig_{args.tag}.json")
    with open(out_json, "w") as h:
        json.dump(summary, h, indent=2)

    # per-test-complex predictions
    df = pd.DataFrame({"pid": test_ids, "truth": truth_ref})
    for r, P in zip(per_seed, all_preds):
        df[f"pred_seed{r['seed']}"] = P
    df["pred_ensemble"] = ens_pred
    df.to_csv(os.path.join(RESULTS, f"aevplig_{args.tag}_preds.csv"), index=False)

    print(f"\n================ AEV-PLIG  {args.split} ================")
    pc_m, pc_s = ms("test_pearson")
    sp_m, sp_s = ms("test_spearman")
    rm_m, rm_s = ms("test_rmse")
    print(f"per-seed  test Pearson  {pc_m:.4f} +/- {pc_s:.4f}")
    print(f"per-seed  test Spearman {sp_m:.4f} +/- {sp_s:.4f}")
    print(f"per-seed  test RMSE     {rm_m:.4f} +/- {rm_s:.4f}")
    print(f"ensemble  test Pearson  {ensemble['test_pearson']:.4f}")
    print(f"ensemble  test Spearman {ensemble['test_spearman']:.4f}")
    print(f"ensemble  test RMSE     {ensemble['test_rmse']:.4f}")
    print(f"saved -> {out_json}")


if __name__ == "__main__":
    main()
