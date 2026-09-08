"""coral_da_probe.py — transductive domain-adaptation probe (CORAL) to recover the
CL3-ID30 novelty gap WITHOUT retraining the encoder or using any novel labels.

The ceiling diagnosis (within_id30_ceiling.py) showed the CDG v2 feature CAN rank novel
pockets to ~0.64 if the probe is trained on the novel-fold distribution, but only ~0.57
when trained on familiar folds — a covariate-shift gap. CORAL (Sun & Saenko 2016) aligns
the SECOND-order statistics (mean + covariance) of the labeled source (train) features to
the UNLABELED target (test) features, then trains the same SiLU-128 probe on the recolored
source. This tests how much of the gap is pure covariate shift (fixable probe-side) vs
conditional shift (needs the encoder to change).

Per-table protocol: CL3 cohorts train on cl123-train; FULL/CASF train on v2-train.
Reports plain vs CORAL ρ per cohort (5 seeds). Honest caveat: transductive (uses the
unlabeled test feature distribution) — a recognized DA setting, distinct from inductive.

Usage: cd voxbind && CUDA_VISIBLE_DEVICES=0 python test/coral_da_probe.py
"""
import csv
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from scipy.linalg import sqrtm

REPO = "/home/shpark/prj-denovo/VoxBind"
FD = f"{REPO}/voxbind/dataset/data/pdbbind/features"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FEAT = f"{FD}/atomblob_density_gradmag_e25_v5_260806_cdg_100m_v2_ep100_e25.pt"


class MLP(nn.Module):
    def __init__(self, d, h=128, p=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.SiLU(), nn.Dropout(p), nn.Linear(h, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_feats(path):
    d = torch.load(path, weights_only=False)
    f = d.get("features", d)
    return {p: np.asarray(v.numpy() if hasattr(v, "numpy") else v, dtype=np.float64)
            for p, v in f.items()}


def load_pK():
    import pandas as pd
    lp = pd.read_csv(f"{REPO}/voxbind/dataset/data/pdbbind/raw/LP_PDBBind.csv").rename(
        columns={"Unnamed: 0": "pid"})
    lp["pid"] = lp["pid"].astype(str).str.lower()
    return {p: float(v) for p, v in zip(lp["pid"], lp["value"]) if v == v}


def split_of(path):
    return {r["pid"].lower(): r["split"] for r in csv.DictReader(open(path))}


def ids(path):
    return set(l.strip().lower() for l in open(path) if l.strip())


def casf_eval():
    rows = list(csv.DictReader(open(f"{REPO}/voxbind/splits/casf2016_eval.csv")))
    return [r["pid"].lower() for r in rows], {r["pid"].lower() for r in rows if r["in_v2train"] == "0"}


def coral_recolor(Xs, Xt, eps=1.0):
    """Align source features Xs to target Xt: whiten by Cs^-1/2, recolor by Ct^1/2,
    shift mean to target. Returns recolored Xs (same rows/labels)."""
    mus, mut = Xs.mean(0), Xt.mean(0)
    d = Xs.shape[1]
    Cs = np.cov(Xs, rowvar=False) + eps * np.eye(d)
    Ct = np.cov(Xt, rowvar=False) + eps * np.eye(d)
    Ws = np.real(sqrtm(np.linalg.inv(Cs)))
    Wt = np.real(sqrtm(Ct))
    return (Xs - mus) @ Ws @ Wt + mut


def train_predict(Xtr, ytr, Xva, yva, Xte, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    ym, ys = float(ytr.mean()), float(ytr.std() + 1e-6)
    prep = lambda A: torch.tensor(((A - mu) / sd).astype(np.float32), device=DEVICE)
    Xtr_t, Xva_t, Xte_t = prep(Xtr), prep(Xva), prep(Xte)
    ytr_t = torch.tensor(((ytr - ym) / ys).astype(np.float32), device=DEVICE)
    model = MLP(Xtr.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.MSELoss()
    best, bstate, bad, n = 1e9, None, 0, Xtr_t.size(0)
    for ep in range(200):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, 64):
            j = perm[i:i + 64]
            if j.numel() < 4:
                continue
            opt.zero_grad(); lossf(model(Xtr_t[j]), ytr_t[j]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xva_t).cpu().numpy() * ys + ym
        vr = float(np.sqrt(((yva - pv) ** 2).mean()))
        if vr < best - 1e-4:
            best, bstate, bad = vr, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 30:
                break
    model.load_state_dict(bstate); model.eval()
    with torch.no_grad():
        return model(Xte_t).cpu().numpy() * ys + ym


def arrs(feats, pK, pids):
    pids = [p for p in pids if p in feats and p in pK]
    return (np.stack([feats[p] for p in pids]), np.array([pK[p] for p in pids]), pids)


def run_cohort(feats, pK, split, test_pids, keep, seeds):
    tr = [p for p, s in split.items() if s == "train"]
    va = [p for p, s in split.items() if s == "val"]
    Xtr, ytr, _ = arrs(feats, pK, tr)
    Xva, yva, _ = arrs(feats, pK, va)
    Xte, yte, te = arrs(feats, pK, [p for p in test_pids if p in keep])
    Xtr_c = coral_recolor(Xtr, Xte)                       # align train -> test distribution
    plain, coral = [], []
    for s in range(seeds):
        pp = train_predict(Xtr, ytr, Xva, yva, Xte, s)
        pc = train_predict(Xtr_c, ytr, Xva, yva, Xte, s)  # NOTE: val from same recolor? keep val plain
        plain.append(spearmanr(yte, pp).statistic)
        coral.append(spearmanr(yte, pc).statistic)
    return (np.mean(plain), np.std(plain)), (np.mean(coral), np.std(coral)), len(te)


def main():
    feats, pK = load_feats(FEAT), load_pK()
    v2 = split_of(f"{REPO}/voxbind/splits/lp_edrscc_v2.csv")
    cl = split_of(f"{REPO}/voxbind/splits/lp_edrscc_v2_cl123.csv")
    D = f"{REPO}/base/_casf/cl123_seqfilter_5seed_260818"
    casf_pids, nontrain = casf_eval()
    clean = {p for p in casf_pids if v2.get(p) not in ("train", "val")}
    full = {p for p, s in v2.items() if s == "test"}
    cl_test = [p for p, s in cl.items() if s == "test"]

    cohorts = [
        ("FULL",          v2, list(full),  full),
        ("CL3",           cl, cl_test,     ids(f"{D}/cl123_test.txt")),
        ("CL3-ID60",      cl, cl_test,     ids(f"{D}/cl123_test_novel60.txt")),
        ("CL3-ID30",      cl, cl_test,     ids(f"{D}/cl123_test_novel30.txt")),
        ("CASF-nontrain", v2, casf_pids,   nontrain),
        ("CASF-clean",    v2, casf_pids,   clean),
    ]
    print(f"CORAL domain-adaptation probe (CDG v2, SiLU-128, 5 seeds)  device={DEVICE}\n")
    print(f"{'cohort':<15}{'n':>5}{'plain ρ':>15}{'CORAL ρ':>15}{'Δ':>9}")
    print("-" * 59)
    for name, split, test_pids, keep in cohorts:
        (pm, ps), (cm, cs), n = run_cohort(feats, pK, split, test_pids, keep, 5)
        flag = "  <-- novelty" if name == "CL3-ID30" else ""
        print(f"{name:<15}{n:>5}{pm:>9.3f}±{ps:.3f}{cm:>9.3f}±{cs:.3f}{cm-pm:>+9.3f}{flag}")


if __name__ == "__main__":
    main()
