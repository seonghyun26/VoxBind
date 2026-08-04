"""probe_mlp.py — train an MLP head on FROZEN DSMBind-encoder features (3 seeds).

Mirrors the ProFSA / our-voxel probe protocol: frozen encoder features (from
extract_features.py) -> small MLP head regressing pK, early-stopped on val, 3 seeds.
Reports test Pearson / Spearman / RMSE mean±std + ensemble. Now that a head is trained,
RMSE is in pK units and directly comparable to the other baselines (unlike the zero-shot
energy correlation).

Run in any env with torch + scipy (the `dsmbind` env works).
"""
import os
import sys
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
DSM_ROOT = os.path.dirname(os.path.dirname(HERE))


class MLPHead(nn.Module):
    def __init__(self, d_in, hidden=256, p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_split(pt_path):
    d = torch.load(pt_path)
    feat, pK, sp = d["feat"], d["pK"], d["split"]
    buckets = {"train": [], "val": [], "test": []}
    for pid in feat:
        buckets[sp[pid]].append(pid)
    def stack(pids):
        X = np.stack([feat[p] for p in pids]).astype(np.float32)
        y = np.array([pK[p] for p in pids], dtype=np.float32)
        return X, y, pids
    return {k: stack(v) for k, v in buckets.items()}, d["dim"]


def metrics(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    return dict(pearson=float(pearsonr(y, yhat)[0]),
                spearman=float(spearmanr(y, yhat)[0]),
                rmse=float(np.sqrt(((y - yhat) ** 2).mean())),
                mae=float(np.abs(y - yhat).mean()))


def train_one(data, seed, dim, epochs=300, lr=1e-3, wd=1e-4, patience=40, device="cuda"):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr, _ = data["train"]; Xva, yva, _ = data["val"]; Xte, yte, te_pids = data["test"]
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    ym, ys = float(ytr.mean()), float(ytr.std() + 1e-6)
    def prep(X): return torch.tensor((X - mu) / sd, device=device)
    Xtr_t, Xva_t, Xte_t = prep(Xtr), prep(Xva), prep(Xte)
    ytr_t = torch.tensor((ytr - ym) / ys, device=device)

    model = MLPHead(dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.MSELoss()
    best_val, best_state, bad = 1e9, None, 0
    n = Xtr_t.size(0); bs = 256
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xva_t).cpu().numpy() * ys + ym
        vrmse = float(np.sqrt(((yva - pv) ** 2).mean()))
        if vrmse < best_val - 1e-4:
            best_val, best_state, bad = vrmse, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        pte = model(Xte_t).cpu().numpy() * ys + ym
    return metrics(yte, pte), dict(zip(te_pids, pte.tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="lp_edrscc_v2")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    pt = os.path.join(DSM_ROOT, "_edrscc", "features", f"{args.split}.pt")
    data, dim = load_split(pt)
    print(f"{args.split}: train {len(data['train'][1])} / val {len(data['val'][1])} "
          f"/ test {len(data['test'][1])} | feat dim {dim}")

    per_seed, preds = [], {}
    for s in range(args.seeds):
        m, p = train_one(data, seed=s, dim=dim, device=device)
        per_seed.append(m)
        for pid, v in p.items():
            preds.setdefault(pid, []).append(v)
        print(f"seed{s}: r={m['pearson']:.4f} rho={m['spearman']:.4f} "
              f"rmse={m['rmse']:.4f} mae={m['mae']:.4f}")

    _, yte, te_pids = data["test"]
    ens = np.array([np.mean(preds[p]) for p in te_pids])
    yt = np.array([data["test"][1][i] for i in range(len(te_pids))])
    em = metrics(yt, ens)
    def ms(key):
        v = [d[key] for d in per_seed]; return float(np.mean(v)), float(np.std(v))
    summary = {
        "model": "DSMBind (frozen encoder + MLP head probe, 3 seeds)",
        "split": args.split, "subset": "test", "n_test": len(te_pids),
        "feat_dim": dim,
        "mean_pearson": ms("pearson")[0], "std_pearson": ms("pearson")[1],
        "mean_spearman": ms("spearman")[0], "std_spearman": ms("spearman")[1],
        "mean_rmse": ms("rmse")[0], "std_rmse": ms("rmse")[1],
        "ensemble": em, "per_seed": per_seed,
    }
    out_dir = os.path.join(DSM_ROOT, "results"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"dsmbind_probe_{args.split}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n===== DSMBind MLP-probe  {args.split} (test, n={len(te_pids)}) =====")
    print(f"MEAN±STD ({args.seeds} seeds): "
          f"r={ms('pearson')[0]:.4f}±{ms('pearson')[1]:.4f}  "
          f"rho={ms('spearman')[0]:.4f}±{ms('spearman')[1]:.4f}  "
          f"rmse={ms('rmse')[0]:.4f}±{ms('rmse')[1]:.4f}")
    print(f"ENSEMBLE: r={em['pearson']:.4f} rho={em['spearman']:.4f} "
          f"rmse={em['rmse']:.4f} mae={em['mae']:.4f}")


if __name__ == "__main__":
    main()
