#!/usr/bin/env python
"""STRICT mask-ratio-only ensemble probe.

Family {v3_m085, v3_m090, v3_m095}: identical v3 data, 100M dim640 [7,4,2],
UNIFORM masking — the ONLY difference is mask_ratio (0.85 / 0.90 / 0.95).
So any ensemble gain here is pure mask-ratio diversity (no data confound,
unlike the champion+v3 ensemble which mixed v2/v3).

Reports each single, every pair, and the triple. Same MLP2 head recipe as
ensemble_probe.py so numbers are comparable.  Run capped-CPU to not disturb
GPU pretraining:
  OMP_NUM_THREADS=2 nice -19 python -u test/ensemble_maskonly.py [split]
"""
import os, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr

torch.set_num_threads(int(os.environ.get("PROBE_THREADS", "2")))
REPO = "/home/shpark/prj-denovo/VoxBind"
FDIR = f"{REPO}/voxbind/dataset/data/pdbbind/features"
DEVICE = os.environ.get("PROBE_DEVICE", "cpu")
HP = dict(hidden=128, dropout=0.1, lr=1e-3, wd=1e-4, max_epochs=300, patience=30, bs=128)
SEEDS = list(range(5))
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "lp_edrscc_v2"

TAGS = {
    "m085": "atomblob_density_gradmag_e49_v5_260725_ar_cvit_100m_v3_m085.pt",
    "m090": "atomblob_density_gradmag_e49_v5_260725_ar_cvit_100m_v3_m090.pt",
    "m095": "atomblob_density_gradmag_e49_v5_260725_ar_cvit_100m_v3_m095.pt",
}
RUNS = [
    ("m085 (single)",        ["m085"]),
    ("m090 (single)",        ["m090"]),
    ("m095 (single)",        ["m095"]),
    ("ens m085+m090",        ["m085", "m090"]),
    ("ens m090+m095",        ["m090", "m095"]),
    ("ens m085+m095",        ["m085", "m095"]),
    ("ens m085+m090+m095",   ["m085", "m090", "m095"]),
]


class MLP2(nn.Module):
    def __init__(self, d, h=128, dr=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.SiLU(), nn.Dropout(dr), nn.Linear(h, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)


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


def train_one(data, seed):
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
            opt.zero_grad(); F.mse_loss(model(Xtr[idx]), ytr[idx]).backward(); opt.step()
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
    return data, data["train"]["X"].shape[1], len(data["test"]["y"])


def main():
    print(f"device={DEVICE}  split={SPLIT}  threads={torch.get_num_threads()}")
    feat_cache = {k: load_feats(v) for k, v in TAGS.items()}
    pK, sm = load_pK(), split_map(SPLIT)
    print(f"{'model':22}{'dim':>6}{'val ρ':>9}{'test r':>9}{'test ρ':>9}{'RMSE':>8}  (mse head, 5 seeds)")
    base = {}
    for label, members in RUNS:
        data, dim, nte = build(members, feat_cache, pK, sm)
        rs = [train_one(data, s) for s in SEEDS]
        m = {k: float(np.mean([r[k] for r in rs])) for k in ("val_rho", "test_r", "test_rho", "test_rmse")}
        sd = float(np.std([r["test_rho"] for r in rs]))
        base[label] = m
        print(f"{label:22}{dim:>6}{m['val_rho']:>9.3f}{m['test_r']:>9.3f}"
              f"{m['test_rho']:>9.3f}±{sd:.3f}{m['test_rmse']:>8.3f}")
    # verdict: best pair/triple vs best single
    singles = [base[k]["test_rho"] for k in base if "single" in k]
    ens = {k: v["test_rho"] for k, v in base.items() if "ens" in k}
    best_single = max(singles)
    best_ens_k = max(ens, key=ens.get)
    print(f"\nbest single test ρ = {best_single:.3f} | best ensemble = {best_ens_k} {ens[best_ens_k]:.3f} "
          f"| Δ = {ens[best_ens_k]-best_single:+.3f}")


if __name__ == "__main__":
    main()
