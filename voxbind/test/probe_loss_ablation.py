"""probe_loss_ablation.py — does adding a ranking/correlation term to the frozen-probe
head loss actually raise test r/ρ over pure MSE? (checks meeting-260806 option B)

Everything is held fixed to the canonical C+D+G probe (100M v2 mask075 encoder,
epoch-49 cached features, lp_edrscc_v2 split, MLP2 head, hidden128/dropout0.1,
bs128/lr1e-3/wd1e-4, max_epochs300/patience30, early-stop on val Spearman) — the ONLY
thing that varies is the training loss:

  mse            : nn.MSELoss on raw pK          (baseline; must reproduce ρ≈0.644)
  mse+corr(λ)    : mse + λ·(1 − pearson(batch))  (push linear correlation)
  mse+rank(λ)    : mse + λ·RankNet(batch)        (push pairwise ordering)
  corr_only      : 1 − pearson(batch)            (scale-free → RMSE meaningless, ρ ceiling)

Usage:  cd voxbind && CUDA_VISIBLE_DEVICES=1 python test/probe_loss_ablation.py
"""
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr

REPO = "/home/shpark/prj-denovo/VoxBind"
FEAT = f"{REPO}/voxbind/dataset/data/pdbbind/features/atomblob_density_gradmag_e49_v5_260705_ar_cvit_100m_v2_mask075.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HP = dict(hidden=128, dropout=0.1, lr=1e-3, wd=1e-4, max_epochs=300, patience=30, bs=128)
SEEDS = list(range(5))


class MLP2(nn.Module):  # identical to 01c_pdbbind_probe.MLP2
    def __init__(self, d, hidden=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.SiLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def pearson_loss(pred, y):
    pred = pred - pred.mean()
    y = y - y.mean()
    num = (pred * y).sum()
    den = pred.norm() * y.norm() + 1e-8
    return 1.0 - num / den


def ranknet_loss(pred, y):
    dp = pred.unsqueeze(1) - pred.unsqueeze(0)          # (n,n) logits
    tgt = (y.unsqueeze(1) > y.unsqueeze(0)).float()
    mask = (y.unsqueeze(1) != y.unsqueeze(0)).float()
    loss = F.binary_cross_entropy_with_logits(dp, tgt, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp(min=1)


LOSSES = {
    "mse":            lambda p, y: F.mse_loss(p, y),
    "mse+corr@1":     lambda p, y: F.mse_loss(p, y) + 1.0 * pearson_loss(p, y),
    "mse+corr@3":     lambda p, y: F.mse_loss(p, y) + 3.0 * pearson_loss(p, y),
    "mse+rank@1":     lambda p, y: F.mse_loss(p, y) + 1.0 * ranknet_loss(p, y),
    "mse+rank@3":     lambda p, y: F.mse_loss(p, y) + 3.0 * ranknet_loss(p, y),
    "corr_only":      lambda p, y: pearson_loss(p, y),
}


def load_feats():
    d = torch.load(FEAT, weights_only=False)
    feats = d.get("features", d.get("feat"))
    return {p: np.asarray(v.numpy() if hasattr(v, "numpy") else v, dtype=np.float32)
            for p, v in feats.items()}


def load_pK():
    import pandas as pd
    lp = pd.read_csv(f"{REPO}/voxbind/dataset/data/pdbbind/raw/LP_PDBBind.csv").rename(
        columns={"Unnamed: 0": "pid"})
    lp["pid"] = lp["pid"].astype(str).str.lower()
    return {p: float(v) for p, v in zip(lp["pid"], lp["value"]) if v == v}


def split_map():
    m = {}
    for r in csv.DictReader(open(f"{REPO}/voxbind/splits/lp_edrscc_v2.csv")):
        m[r["pid"].lower()] = r["split"]
    return m


def arrs(pids, feats, pK):
    pids = [p for p in pids if p in feats and p in pK]
    X = np.stack([feats[p] for p in pids]).astype(np.float32)
    y = np.array([pK[p] for p in pids], dtype=np.float32)
    return X, y


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
            opt.zero_grad()
            loss_fn(model(Xtr[idx]), ytr[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xva).cpu().numpy()
        vs = spearmanr(pv, yva_np).statistic
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


def main():
    print(f"device={DEVICE}  feature={FEAT.split('/')[-1]}")
    feats, pK, sm = load_feats(), load_pK(), split_map()
    data = {}
    for split in ("train", "val", "test"):
        X, y = arrs([p for p in sm if sm[p] == split], feats, pK)
        data[split] = {"X": X, "y": y}
        print(f"  {split:5s} n={len(y)}")
    print(f"\n{'loss':<14}{'val ρ':>10}{'test r':>10}{'test ρ':>10}{'RMSE':>10}   (mean±std, 5 seeds)")
    base = None
    for name, fn in LOSSES.items():
        rs = [train_one(data, fn, s) for s in SEEDS]
        agg = {k: (float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs])))
               for k in ("val_rho", "test_r", "test_rho", "test_rmse")}
        if base is None:
            base = agg
        dr = agg["test_rho"][0] - base["test_rho"][0]
        dd = agg["test_r"][0] - base["test_r"][0]
        print(f"{name:<14}"
              f"{agg['val_rho'][0]:>7.3f}±{agg['val_rho'][1]:.2f}"
              f"{agg['test_r'][0]:>7.3f}±{agg['test_r'][1]:.2f}"
              f"{agg['test_rho'][0]:>7.3f}±{agg['test_rho'][1]:.2f}"
              f"{agg['test_rmse'][0]:>7.3f}±{agg['test_rmse'][1]:.2f}"
              f"    Δr {dd:+.3f}  Δρ {dr:+.3f}")


if __name__ == "__main__":
    main()
