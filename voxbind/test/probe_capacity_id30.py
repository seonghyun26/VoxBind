"""probe_capacity_id30.py — does a BIGGER probe head (wider / deeper MLP) change
CL3-ID30 performance? Tests whether the current 2-layer/128 probe is under-capacity
for the novel-protein cohort, in the DEPLOYMENT setting (train on familiar v2-train,
early-stop on v2-val, test on FULL + CL3-ID30).

Sweep: hidden width h ∈ {64,128,256,512,1024} × depth L ∈ {1,2,3} hidden layers
(uniform width), 5 seeds. Reports FULL ρ and CL3-ID30 ρ per config, plus the current
tapered default (128→64) for reference.

If ID30 ρ is flat across capacity → the ceiling is the feature/probe-transfer, NOT probe
capacity (consistent with within_id30_ceiling.py). If ID30 rises with capacity → the
current head under-fits novel pockets.

Usage: cd voxbind && CUDA_VISIBLE_DEVICES=0 python test/probe_capacity_id30.py
"""
import csv
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr

REPO = "/home/shpark/prj-denovo/VoxBind"
FD = f"{REPO}/voxbind/dataset/data/pdbbind/features"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FEAT = f"{FD}/atomblob_density_gradmag_e25_v5_260806_cdg_100m_v2_ep100_e25.pt"
SEEDS = 5


class MLP(nn.Module):
    """Configurable probe: `L` hidden layers of width `h` (uniform), dropout p.
    tapered=True reproduces the canonical default (h → h//2)."""
    def __init__(self, d, h, L, p=0.1, tapered=False):
        super().__init__()
        layers, prev = [], d
        widths = ([h, h // 2] if tapered else [h] * L)
        for w in widths:
            layers += [nn.Linear(prev, w), nn.ReLU(), nn.Dropout(p)]
            prev = w
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

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


def ids(path):
    return set(l.strip().lower() for l in open(path) if l.strip())


def train_predict(feats, pK, v2, test_pids, seed, h, L, tapered):
    torch.manual_seed(seed); np.random.seed(seed)
    lossf = nn.MSELoss()

    def arrs(pids):
        pids = [p for p in pids if p in feats and p in pK]
        return (np.stack([feats[p] for p in pids]).astype(np.float32),
                np.array([pK[p] for p in pids], dtype=np.float32), pids)

    tr = [p for p, s in v2.items() if s == "train"]
    va = [p for p, s in v2.items() if s == "val"]
    Xtr, ytr, _ = arrs(tr); Xva, yva, _ = arrs(va); Xte, yte, te = arrs(test_pids)
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    ym, ys = float(ytr.mean()), float(ytr.std() + 1e-6)
    prep = lambda A: torch.tensor((A - mu) / sd, device=DEVICE)
    Xtr_t, Xva_t, Xte_t = prep(Xtr), prep(Xva), prep(Xte)
    ytr_t = torch.tensor((ytr - ym) / ys, device=DEVICE)
    model = MLP(Xtr.shape[1], h, L, tapered=tapered).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best, bstate, bad, bs = 1e9, None, 0, 64
    n = Xtr_t.size(0)
    for ep in range(200):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            if j.numel() < 4:
                continue
            opt.zero_grad(); lossf(model(Xtr_t[j]), ytr_t[j]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xva_t).cpu().numpy() * ys + ym
        vr = float(np.sqrt(((yva - pv) ** 2).mean()))
        if vr < best - 1e-4:
            best, bstate, bad = vr, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 30:
                break
    model.load_state_dict(bstate); model.eval()
    with torch.no_grad():
        pte = model(Xte_t).cpu().numpy() * ys + ym
    return dict(zip(te, pte)), dict(zip(te, yte))


def rho(pred, yt, keep):
    ps = [p for p in pred if p in keep]
    y = np.array([yt[p] for p in ps]); yh = np.array([pred[p] for p in ps])
    return spearmanr(y, yh).statistic


def main():
    feats, pK, v2 = load_feats(FEAT), load_pK(), v2_split()
    full = {p for p, s in v2.items() if s == "test"}
    id30 = ids(f"{REPO}/base/_casf/cl123_seqfilter_5seed_260818/cl123_test_novel30.txt")
    pool = full | id30
    n_params = lambda h, L, t: sum(p.numel() for p in MLP(640, h, L, tapered=t).parameters())
    print(f"device={DEVICE}  FULL={len(full)}  ID30={len(id30)}  seeds={SEEDS}\n")

    configs = [("default(128→64)", 128, 2, True)]
    for L in (1, 2, 3):
        for h in (64, 128, 256, 512, 1024):
            configs.append((f"L{L}×{h}", h, L, False))

    print(f"{'config':<16}{'params':>9}{'FULL ρ':>16}{'ID30 ρ':>16}")
    print("-" * 57)
    for name, h, L, tap in configs:
        fs, ds = [], []
        for s in range(SEEDS):
            pred, yt = train_predict(feats, pK, v2, pool, s, h, L, tap)
            fs.append(rho(pred, yt, full)); ds.append(rho(pred, yt, id30))
        fm, fsd = np.mean(fs), np.std(fs)
        dm, dsd = np.mean(ds), np.std(ds)
        print(f"{name:<16}{n_params(h, L, tap):>9,}{fm:>10.3f}±{fsd:.3f}{dm:>10.3f}±{dsd:.3f}")


if __name__ == "__main__":
    main()
