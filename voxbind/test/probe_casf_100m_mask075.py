"""probe_casf_100m_mask075.py — CASF-2016 probe for the 100M ChannelViT mask075 encoder.

Trains MLP head on lp_edrscc_v2 TRAIN (epoch-49 features, already cached), early-stops on
v2-val, then predicts CASF-2016 eval set (214 pids, all present in the v2 feature bundle).

Outputs:
  base/_casf/CDG_100m_mask075.json  (same schema as CDG.json)

Usage:
    cd voxbind && CUDA_VISIBLE_DEVICES=4 python test/probe_casf_100m_mask075.py
"""
import argparse
import csv
import json
import os

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr


def _pearson_term(p, y):
    """1 - Pearson r over the batch (GenScore-style aux). Scale-free."""
    p = p - p.mean(); y = y - y.mean()
    return 1.0 - (p * y).sum() / (p.norm() * y.norm() + 1e-8)


def build_loss(spec, aux_weight):
    """spec: 'mse' (default) or 'mse+corr' (MSE + aux_weight*(1-Pearson))."""
    mse = nn.MSELoss()
    if spec == "mse":
        return lambda p, y: mse(p, y)
    if spec == "mse+corr":
        return lambda p, y: mse(p, y) + aux_weight * _pearson_term(p, y)
    raise ValueError(f"unknown --loss {spec!r} (use mse | mse+corr)")

REPO = "/home/shpark/prj-denovo/VoxBind"
FD   = f"{REPO}/voxbind/dataset/data/pdbbind/features"
OUT  = f"{REPO}/base/_casf"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FEAT_PATH = f"{FD}/atomblob_density_gradmag_e49_v5_260705_ar_cvit_100m_v2_mask075.pt"
HP = dict(hidden=128, dropout=0.1, lr=1e-3, wd=1e-4, epochs=200, patience=30, bs=64, seeds=3)


class MLP(nn.Module):
    def __init__(self, d, h, p):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, h), nn.ReLU(), nn.Dropout(p),
            nn.Linear(h, h // 2), nn.ReLU(), nn.Dropout(p),
            nn.Linear(h // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_feats(path):
    d = torch.load(path, weights_only=False)
    feats = d.get("features", d.get("feat"))
    return {p: np.asarray(v.numpy() if hasattr(v, "numpy") else v, dtype=np.float32)
            for p, v in feats.items()}


def load_pK():
    import pandas as pd
    lp = pd.read_csv(f"{REPO}/voxbind/dataset/data/pdbbind/raw/LP_PDBBind.csv").rename(
        columns={"Unnamed: 0": "pid"})
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


def train_predict(feats, pK, v2, casf_pids, seed, loss_fn=None):
    torch.manual_seed(seed); np.random.seed(seed)
    if loss_fn is None:
        loss_fn = build_loss("mse", 0.0)

    def arrs(pids):
        pids = [p for p in pids if p in feats and p in pK]
        X = np.stack([feats[p] for p in pids]).astype(np.float32)
        y = np.array([pK[p] for p in pids], dtype=np.float32)
        return X, y, pids

    tr = [p for p, s in v2.items() if s == "train"]
    va = [p for p, s in v2.items() if s == "val"]
    Xtr, ytr, _ = arrs(tr)
    Xva, yva, _ = arrs(va)
    Xte, yte, te_pids = arrs(casf_pids)

    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    ym, ys = float(ytr.mean()), float(ytr.std() + 1e-6)

    def prep(X):
        return torch.tensor((X - mu) / sd, device=DEVICE)

    Xtr_t, Xva_t, Xte_t = prep(Xtr), prep(Xva), prep(Xte)
    ytr_t = torch.tensor((ytr - ym) / ys, device=DEVICE)

    model = MLP(Xtr.shape[1], HP["hidden"], HP["dropout"]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=HP["lr"], weight_decay=HP["wd"])
    best, best_state, bad = 1e9, None, 0
    n = Xtr_t.size(0)

    for ep in range(HP["epochs"]):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, HP["bs"]):
            idx = perm[i:i + HP["bs"]]
            if idx.numel() < 4:
                continue
            opt.zero_grad()
            loss_fn(model(Xtr_t[idx]), ytr_t[idx]).backward()
            opt.step()
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

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pte = model(Xte_t).cpu().numpy() * ys + ym
    return te_pids, yte, pte


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", default="mse", help="probe-head loss: mse (default) | mse+corr")
    ap.add_argument("--aux_weight", type=float, default=5.0, help="Pearson-aux weight for mse+corr")
    ap.add_argument("--feat", default=FEAT_PATH, help="feature cache to probe (default = CDG champion)")
    ap.add_argument("--model", default="CDG_100m_mask075", help="model name for output json")
    ap.add_argument("--out", default=None, help="override output json path")
    args = ap.parse_args()
    loss_fn = build_loss(args.loss, args.aux_weight)
    tag = "" if args.loss == "mse" else f"_corr{args.aux_weight:g}"

    print(f"Loading features from: {args.feat}")
    print(f"loss = {args.loss}" + (f" (aux_weight {args.aux_weight:g})" if args.loss != "mse" else ""))
    feats = load_feats(args.feat)
    print(f"Loaded {len(feats):,} feature vectors (dim={next(iter(feats.values())).shape[0]})")

    pK = load_pK()
    v2 = v2_split()
    casf_pids, nontrain = casf_eval()

    # clean = CASF pids NOT in our lp_edrscc_v2 train OR val (nontrain only removes train,
    # so it still contains ~32 val-overlap complexes; clean removes both → truly held out).
    clean_set = {p for p in casf_pids if v2.get(p) not in ("train", "val")}
    print(f"CASF eval: {len(casf_pids)} leaky | {len(nontrain)} non-train | "
          f"{len(clean_set)} clean (not in v2 train/val) | device={DEVICE}\n")

    # Check coverage
    missing = [p for p in casf_pids if p not in feats]
    if missing:
        print(f"WARNING: {len(missing)} CASF pids not in feature bundle: {missing[:5]}")
    else:
        print(f"All {len(casf_pids)} CASF pids covered by feature bundle.")

    per = {"leaky": [], "nontrain": [], "clean": []}
    for s in range(HP["seeds"]):
        print(f"  seed {s}...")
        te_pids, yte, pte = train_predict(feats, pK, v2, casf_pids, s, loss_fn=loss_fn)
        pred_tag = f"{args.model}{tag}"
        pred_path = os.path.join(OUT, f"{pred_tag}_casf2016_preds_seed{s}.csv")
        with open(pred_path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["pid", "pred", "y"])
            writer.writerows((pid, float(pred), float(y))
                             for pid, pred, y in zip(te_pids, pte, yte))
        nt_mask = np.array([p in nontrain for p in te_pids])
        cl_mask = np.array([p in clean_set for p in te_pids])
        lm = metrics(yte, pte)
        nm = metrics(yte, pte, mask=nt_mask)
        cm = metrics(yte, pte, mask=cl_mask)
        per["leaky"].append(lm)
        per["nontrain"].append(nm)
        per["clean"].append(cm)
        print(f"    leaky    n={lm['n']} ρ={lm['spearman']:.3f} r={lm['pearson']:.3f} rmse={lm['rmse']:.3f}")
        print(f"    nontrain n={nm['n']} ρ={nm['spearman']:.3f} r={nm['pearson']:.3f} rmse={nm['rmse']:.3f}")
        print(f"    clean    n={cm['n']} ρ={cm['spearman']:.3f} r={cm['pearson']:.3f} rmse={cm['rmse']:.3f}")

    def agg(lst):
        out = {}
        for k in ("pearson", "spearman", "rmse"):
            v = [d[k] for d in lst]
            out[k] = {"mean": float(np.mean(v)), "std": float(np.std(v))}
        out["n"] = lst[0]["n"]
        return out

    res = {
        "model": f"{args.model}{tag}",
        "encoder": "260705_ar_cvit_100m_v2_mask075 (epoch 49)",
        "train": f"lp_edrscc_v2 train (frozen-encoder + MLP head, 3 seeds); loss={args.loss}"
                 + (f" aux_weight={args.aux_weight:g}" if args.loss != "mse" else ""),
        "leaky": agg(per["leaky"]),
        "nontrain": agg(per["nontrain"]),
        "clean": agg(per["clean"]),
    }
    os.makedirs(OUT, exist_ok=True)
    out_path = args.out or f"{OUT}/{args.model}{tag}.json"
    json.dump(res, open(out_path, "w"), indent=2)
    print(f"\nResult written to: {out_path}")
    print(f"leaky(n={res['leaky']['n']})   ρ={res['leaky']['spearman']['mean']:.3f}±{res['leaky']['spearman']['std']:.3f}  "
          f"r={res['leaky']['pearson']['mean']:.3f}  rmse={res['leaky']['rmse']['mean']:.3f}")
    print(f"nontrain(n={res['nontrain']['n']}) ρ={res['nontrain']['spearman']['mean']:.3f}±{res['nontrain']['spearman']['std']:.3f}  "
          f"r={res['nontrain']['pearson']['mean']:.3f}  rmse={res['nontrain']['rmse']['mean']:.3f}")


if __name__ == "__main__":
    main()
