"""probe_casf.py — evaluate the frozen-encoder probes on CASF-2016 (leaky) + CASF-non-train.

Trains a small MLP head on the lp_edrscc_v2 TRAIN set (frozen encoder features), early-stops
on v2-val, then predicts the CASF-2016 eval set (voxbind/splits/casf2016_eval.csv, 214 pids).
Reports metrics on:
  - leaky   : all 214 CASF complexes (90 are in v2-train -> memorised -> inflated)
  - nontrain: the 124 CASF complexes NOT in v2-train (honest / OOD)
Both come from the SAME trained head (one prediction pass), 3 seeds. This mirrors the
HonestAffinity canonical-vs-non-train protocol with our density encoders.

    cd voxbind && CUDA_VISIBLE_DEVICES=6 python test/probe_casf.py
"""
import csv
import json
import os

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr

REPO = "/home/shpark/prj-denovo/VoxBind"
FD = f"{REPO}/voxbind/dataset/data/pdbbind/features"
OUT = f"{REPO}/base/_casf"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# frozen-encoder feature caches (01c bundle: {'features': pid->tensor}; dsmbind: {'feat': ...})
METHODS = {
    "C":     dict(path=f"{FD}/atomblob_e99_v5_otf_coords_mask050.pt", key="features"),
    "CDG":   dict(path=f"{FD}/atomblob_density_gradmag_e99_v5_260623_ar_cvit_c1_g742.pt", key="features"),
    "DSMBind": dict(path=f"{REPO}/base/dsmbind/_edrscc/features/lp_edrscc_v2.pt", key="feat"),
}
HP = dict(hidden=128, dropout=0.1, lr=1e-3, wd=1e-4, epochs=200, patience=30, bs=64, seeds=3)


class MLP(nn.Module):
    def __init__(self, d, h, p):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(p),
                                 nn.Linear(h, h // 2), nn.ReLU(), nn.Dropout(p),
                                 nn.Linear(h // 2, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_feats(path, key):
    d = torch.load(path, weights_only=False)
    feats = d.get(key) if isinstance(d, dict) else None
    if feats is None:
        feats = d.get("features", d.get("feat"))
    return {p: np.asarray(v.numpy() if hasattr(v, "numpy") else v, dtype=np.float32) for p, v in feats.items()}


def load_pK():
    import pandas as pd
    lp = pd.read_csv(f"{REPO}/voxbind/dataset/data/pdbbind/raw/LP_PDBBind.csv").rename(columns={"Unnamed: 0": "pid"})
    lp["pid"] = lp["pid"].astype(str).str.lower()
    return {p: float(v) for p, v in zip(lp["pid"], lp["value"]) if v == v}


def v2_split():
    m = {}
    for r in csv.DictReader(open(f"{REPO}/voxbind/splits/lp_edrscc_v2.csv")):
        m[r["pid"].lower()] = r["split"]
    return m


def casf_eval():
    rows = list(csv.DictReader(open(f"{REPO}/voxbind/splits/casf2016_eval.csv")))
    pids = [r["pid"].lower() for r in rows]
    nontrain = {r["pid"].lower() for r in rows if r["in_v2train"] == "0"}
    return pids, nontrain


def metrics(y, yhat, mask=None):
    y, yhat = np.asarray(y), np.asarray(yhat)
    if mask is not None:
        y, yhat = y[mask], yhat[mask]
    return dict(pearson=float(pearsonr(y, yhat)[0]), spearman=float(spearmanr(y, yhat)[0]),
                rmse=float(np.sqrt(((y - yhat) ** 2).mean())), n=int(len(y)))


def train_predict(feats, pK, v2, casf_pids, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    def arrs(pids):
        pids = [p for p in pids if p in feats and p in pK]
        X = np.stack([feats[p] for p in pids]).astype(np.float32)
        y = np.array([pK[p] for p in pids], dtype=np.float32)
        return X, y, pids
    tr = [p for p, s in v2.items() if s == "train"]
    va = [p for p, s in v2.items() if s == "val"]
    Xtr, ytr, _ = arrs(tr); Xva, yva, _ = arrs(va); Xte, yte, te_pids = arrs(casf_pids)
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    ym, ys = float(ytr.mean()), float(ytr.std() + 1e-6)
    def prep(X): return torch.tensor((X - mu) / sd, device=DEVICE)
    Xtr_t, Xva_t, Xte_t = prep(Xtr), prep(Xva), prep(Xte)
    ytr_t = torch.tensor((ytr - ym) / ys, device=DEVICE)
    model = MLP(Xtr.shape[1], HP["hidden"], HP["dropout"]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=HP["lr"], weight_decay=HP["wd"])
    lossf = nn.MSELoss(); best, best_state, bad = 1e9, None, 0
    n = Xtr_t.size(0)
    for ep in range(HP["epochs"]):
        model.train(); perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, HP["bs"]):
            idx = perm[i:i + HP["bs"]]; opt.zero_grad()
            lossf(model(Xtr_t[idx]), ytr_t[idx]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xva_t).cpu().numpy() * ys + ym
        vr = float(np.sqrt(((yva - pv) ** 2).mean()))
        if vr < best - 1e-4:
            best, best_state, bad = vr, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= HP["patience"]:
                break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        pte = model(Xte_t).cpu().numpy() * ys + ym
    return te_pids, yte, pte


def run_method(name, cfg, pK, v2, casf_pids, nontrain):
    feats = load_feats(cfg["path"], cfg["key"])
    per = {"leaky": [], "nontrain": []}
    for s in range(HP["seeds"]):
        te_pids, yte, pte = train_predict(feats, pK, v2, casf_pids, s)
        nt_mask = np.array([p in nontrain for p in te_pids])
        per["leaky"].append(metrics(yte, pte))
        per["nontrain"].append(metrics(yte, pte, mask=nt_mask))
    def agg(lst):
        out = {}
        for k in ("pearson", "spearman", "rmse"):
            v = [d[k] for d in lst]
            out[k] = {"mean": float(np.mean(v)), "std": float(np.std(v))}
        out["n"] = lst[0]["n"]
        return out
    res = {"model": name, "train": "lp_edrscc_v2 train (frozen-encoder + MLP head, 3 seeds)",
           "leaky": agg(per["leaky"]), "nontrain": agg(per["nontrain"])}
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f"{OUT}/{name}.json", "w"), indent=2)
    print(f"{name:8s} leaky(n={res['leaky']['n']}) rho={res['leaky']['spearman']['mean']:.3f} "
          f"r={res['leaky']['pearson']['mean']:.3f} | nontrain(n={res['nontrain']['n']}) "
          f"rho={res['nontrain']['spearman']['mean']:.3f} r={res['nontrain']['pearson']['mean']:.3f}")


def main():
    pK = load_pK(); v2 = v2_split(); casf_pids, nontrain = casf_eval()
    print(f"CASF eval: {len(casf_pids)} leaky | {len(nontrain)} non-train | device={DEVICE}\n")
    for name, cfg in METHODS.items():
        run_method(name, cfg, pK, v2, casf_pids, nontrain)


if __name__ == "__main__":
    main()
