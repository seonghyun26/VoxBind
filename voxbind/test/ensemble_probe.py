#!/usr/bin/env python
"""Encoder-ensemble affinity probe: concat frozen mean-pool features from several
CDG encoders → one MLP2 head. A cheap attempt to beat the single champion (0.644).
Reuses the EXACT head recipe from probe_loss_ablation.py so numbers are comparable.

  python test/ensemble_probe.py [split]      # default lp_edrscc_v2
"""
import csv, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr

REPO = "/home/shpark/prj-denovo/VoxBind"
FDIR = f"{REPO}/voxbind/dataset/data/pdbbind/features"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
HP = dict(hidden=128, dropout=0.1, lr=1e-3, wd=1e-4, max_epochs=300, patience=30, bs=128)
SEEDS = list(range(5))
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "lp_edrscc_v2"

TAGS = {
    "champion": "atomblob_density_gradmag_e49_v5_260705_ar_cvit_100m_v2_mask075.pt",
    "v3_m090":  "atomblob_density_gradmag_e49_v5_260725_ar_cvit_100m_v3_m090.pt",
    "v3_m095":  "atomblob_density_gradmag_e49_v5_260725_ar_cvit_100m_v3_m095.pt",
}
# which member-sets to evaluate
RUNS = [
    ("champion (single)", ["champion"]),
    ("v3_m090 (single)",  ["v3_m090"]),
    ("v3_m095 (single)",  ["v3_m095"]),
    ("ENSEMBLE x3",       ["champion", "v3_m090", "v3_m095"]),
]


class MLP2(nn.Module):
    def __init__(self, d, h=128, dr=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.SiLU(), nn.Dropout(dr), nn.Linear(h, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)


def pearson_loss(p, y):
    p = p - p.mean(); y = y - y.mean()
    return 1.0 - (p * y).sum() / (p.norm() * y.norm() + 1e-8)

LOSSES = {"mse": lambda p, y: F.mse_loss(p, y),
          "mse+corr@3": lambda p, y: F.mse_loss(p, y) + 3.0 * pearson_loss(p, y)}


def load_feats(fn):
    d = torch.load(f"{FDIR}/{fn}", weights_only=False)
    f = d.get("features", d.get("feat"))
    return {p.lower(): np.asarray(v.numpy() if hasattr(v, "numpy") else v, dtype=np.float32)
            for p, v in f.items()}


def load_pK():
    import pandas as pd
    lp = pd.read_csv(f"{REPO}/voxbind/dataset/data/pdbbind/raw/LP_PDBBind.csv").rename(
        columns={"Unnamed: 0": "pid"})
    lp["pid"] = lp["pid"].astype(str).str.lower()
    return {p: float(v) for p, v in zip(lp["pid"], lp["value"]) if v == v}


def split_map(split):
    sys.path.insert(0, REPO)
    from voxbind.splits import load_split
    sp = load_split(split)
    return {p: s for s in ("train", "val", "test") for p in sp[s]}


def train_one(data, loss_fn, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr = (torch.from_numpy(data["train"][k]).to(DEVICE) for k in ("X", "y"))
    Xva, yva = (torch.from_numpy(data["val"][k]).to(DEVICE) for k in ("X", "y"))
    Xte, yte = (torch.from_numpy(data["test"][k]).to(DEVICE) for k in ("X", "y"))
    model = MLP2(Xtr.shape[1], HP["hidden"], HP["dropout"]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=HP["lr"], weight_decay=HP["wd"])
    n = Xtr.shape[0]
    best_val, best_state, since = -np.inf, None, 0
    yva_np, yte_np = yva.cpu().numpy(), yte.cpu().numpy()
    for _ in range(HP["max_epochs"]):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for s in range(0, n, HP["bs"]):
            idx = perm[s:s + HP["bs"]]
            if idx.numel() < 4:
                continue
            opt.zero_grad(); loss_fn(model(Xtr[idx]), ytr[idx]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vs = spearmanr(model(Xva).cpu().numpy(), yva_np).statistic
        if vs > best_val:
            best_val, since = vs, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= HP["patience"]:
                break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        pte = model(Xte).cpu().numpy()
    return dict(val_rho=float(best_val),
                test_r=float(pearsonr(pte, yte_np).statistic),
                test_rho=float(spearmanr(pte, yte_np).statistic),
                test_rmse=float(np.sqrt(((pte - yte_np) ** 2).mean())))


def build(members, feat_cache, pK, sm):
    dicts = [feat_cache[m] for m in members]
    shared = set(sm) & set(pK)
    for d in dicts:
        shared &= set(d)
    data = {}
    for split in ("train", "val", "test"):
        pids = sorted(p for p in shared if sm[p] == split)
        X = np.concatenate([np.stack([d[p] for p in pids]) for d in dicts], axis=1).astype(np.float32)
        y = np.array([pK[p] for p in pids], dtype=np.float32)
        data[split] = {"X": X, "y": y}
    return data, data["train"]["X"].shape[1]


def main():
    print(f"device={DEVICE}  split={SPLIT}")
    feat_cache = {k: load_feats(v) for k, v in TAGS.items()}
    pK, sm = load_pK(), split_map(SPLIT)
    print(f"{'model':22}{'dim':>6}{'loss':>12}{'val ρ':>9}{'test r':>9}{'test ρ':>9}{'RMSE':>8}  (5 seeds)")
    for label, members in RUNS:
        data, dim = build(members, feat_cache, pK, sm)
        for lname, fn in LOSSES.items():
            rs = [train_one(data, fn, s) for s in SEEDS]
            m = {k: float(np.mean([r[k] for r in rs])) for k in ("val_rho", "test_r", "test_rho", "test_rmse")}
            print(f"{label:22}{dim:>6}{lname:>12}{m['val_rho']:>9.3f}{m['test_r']:>9.3f}"
                  f"{m['test_rho']:>9.3f}{m['test_rmse']:>8.3f}")


if __name__ == "__main__":
    main()
