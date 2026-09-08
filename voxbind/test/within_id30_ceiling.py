"""within_id30_ceiling.py — is the CL3-ID30 weakness a probe-TRANSFER problem
(info IS in the feature) or a feature-INFORMATION problem (info NOT in the feature)?

Frozen CDG v2 features (already cached). Three probe arms, all evaluated on the SAME
262 CL3-ID30 novel-protein complexes (pooled K-fold predictions → ρ over 262):

  FULL-FAMILIAR  : train on ALL ~3850 familiar v2-train complexes → test ID30   (= the current 0.60)
  MATCHED-FAMIL. : train on N random familiar (N = within-CV train size ~210)    → test ID30 fold
  WITHIN-NOVEL   : K-fold WITHIN the 262 → train on ~210 novel, test held-out novel fold

The clean, sample-size-controlled comparison is WITHIN-NOVEL vs MATCHED-FAMILIAR
(identical training-set size; only the train DISTRIBUTION differs — novel vs familiar):
  • WITHIN ≫ MATCHED  → training on the novel distribution helps a lot → the feature HAS
    transferable novel-pocket info; the current probe just trained on the wrong distribution
    → CHEAP probe-side fix (domain adaptation / include-novel / recalibration).
  • WITHIN ≈ MATCHED  → training distribution doesn't matter → the feature itself is the
    ceiling → need BETTER features (retrain: alignment / ESM).

Usage: cd voxbind && CUDA_VISIBLE_DEVICES=0 python test/within_id30_ceiling.py
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
K = 5
SEEDS = 5


class MLP(nn.Module):
    def __init__(self, d, h=128, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, h), nn.ReLU(), nn.Dropout(p),
            nn.Linear(h, h // 2), nn.ReLU(), nn.Dropout(p),
            nn.Linear(h // 2, 1))

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
    return [l.strip().lower() for l in open(path) if l.strip()]


def train_mlp(Xtr, ytr, Xte, seed, epochs=150, patience=25):
    """Train MLP on (Xtr,ytr) with a carved 15% early-stop val, predict Xte. mse."""
    torch.manual_seed(seed); np.random.seed(seed)
    n = len(Xtr)
    idx = np.random.permutation(n)
    nv = max(8, int(0.15 * n))
    vi, ti = idx[:nv], idx[nv:]
    mu, sd = Xtr[ti].mean(0, keepdims=True), Xtr[ti].std(0, keepdims=True) + 1e-6
    ym, ys = float(ytr[ti].mean()), float(ytr[ti].std() + 1e-6)
    prep = lambda X: torch.tensor((X - mu) / sd, device=DEVICE)
    Xt, Xv, Xe = prep(Xtr[ti]), prep(Xtr[vi]), prep(Xte)
    yt = torch.tensor((ytr[ti] - ym) / ys, device=DEVICE)
    yv = ytr[vi]
    model = MLP(Xtr.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.MSELoss()
    best, bstate, bad, bs = 1e9, None, 0, 64
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(ti), device=DEVICE)
        for i in range(0, len(ti), bs):
            j = perm[i:i + bs]
            if j.numel() < 4:
                continue
            opt.zero_grad(); lossf(model(Xt[j]), yt[j]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xv).cpu().numpy() * ys + ym
        vr = float(np.sqrt(((yv - pv) ** 2).mean()))
        if vr < best - 1e-4:
            best, bstate, bad = vr, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(bstate); model.eval()
    with torch.no_grad():
        return model(Xe).cpu().numpy() * ys + ym


def kfold(pids, k, seed):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(pids))
    return [[pids[i] for i in f] for f in np.array_split(idx, k)]


def X(feats, ps): return np.stack([feats[p] for p in ps]).astype(np.float32)
def Y(pK, ps):    return np.array([pK[p] for p in ps], dtype=np.float32)


def run_seed(novel, familiar, feats, pK, seed):
    folds = kfold(novel, K, seed)
    rng = np.random.RandomState(seed + 1000)
    pw, pf = {}, {}
    for fi in range(K):
        te = folds[fi]
        tr_nov = [p for j, f in enumerate(folds) if j != fi for p in f]
        n = len(tr_nov)
        tr_fam = list(rng.choice(familiar, size=n, replace=False))
        Xte = X(feats, te)
        for p, v in zip(te, train_mlp(X(feats, tr_nov), Y(pK, tr_nov), Xte, seed)):
            pw[p] = v
        for p, v in zip(te, train_mlp(X(feats, tr_fam), Y(pK, tr_fam), Xte, seed)):
            pf[p] = v
    # full-familiar reference
    pfull = dict(zip(novel, train_mlp(X(feats, familiar), Y(pK, familiar), X(feats, novel), seed)))
    return pw, pf, pfull


def rho(preds, pK, pids):
    y = Y(pK, pids); yh = np.array([preds[p] for p in pids])
    return spearmanr(y, yh).statistic, pearsonr(y, yh)[0]


def main():
    feats, pK, v2 = load_feats(FEAT), load_pK(), v2_split()
    novel = [p for p in ids(f"{REPO}/base/_casf/cl123_seqfilter_5seed_260818/cl123_test_novel30.txt")
             if p in feats and p in pK]
    familiar = [p for p, s in v2.items() if s == "train" and p in feats and p in pK and p not in set(novel)]
    n_train = len(novel) - len(novel) // K
    print(f"device={DEVICE}  novel(ID30)={len(novel)}  familiar-pool={len(familiar)}  "
          f"K={K}  matched N_train≈{n_train}  seeds={SEEDS}\n")

    W, F, FULL = [], [], []
    for s in range(SEEDS):
        pw, pf, pfull = run_seed(novel, familiar, feats, pK, s)
        rw, rf, rfl = rho(pw, pK, novel), rho(pf, pK, novel), rho(pfull, pK, novel)
        W.append(rw); F.append(rf); FULL.append(rfl)
        print(f"  seed{s}:  WITHIN ρ={rw[0]:.3f}  MATCHED-FAM ρ={rf[0]:.3f}  FULL-FAM ρ={rfl[0]:.3f}")

    def agg(L, i=0):
        v = np.array([x[i] for x in L]); return v.mean(), v.std()
    (wm, ws), (fm, fs), (flm, fls) = agg(W), agg(F), agg(FULL)
    print(f"\n{'arm':<26}{'train N':>9}{'train dist':>12}{'ID30 ρ':>16}")
    print("-" * 63)
    print(f"{'FULL-FAMILIAR (current)':<26}{len(familiar):>9}{'familiar':>12}{flm:>10.3f}±{fls:.3f}")
    print(f"{'MATCHED-FAMILIAR':<26}{n_train:>9}{'familiar':>12}{fm:>10.3f}±{fs:.3f}")
    print(f"{'WITHIN-NOVEL (ceiling)':<26}{n_train:>9}{'novel':>12}{wm:>10.3f}±{ws:.3f}")
    d = wm - fm
    print(f"\nΔ(WITHIN − MATCHED, same N) = {d:+.3f}   "
          f"(vs seed-noise ±{max(ws, fs):.3f})")
    print("\ninterpretation:")
    if d > 3 * max(ws, fs, 1e-6):
        print("  → WITHIN ≫ MATCHED: feature HAS transferable novel-pocket info; the probe just")
        print("    trained on the wrong distribution → CHEAP probe-side fix (domain adaptation).")
    elif d > max(ws, fs):
        print("  → WITHIN > MATCHED (modest): partial transfer signal; probe-side helps somewhat.")
    else:
        print("  → WITHIN ≈ MATCHED: training distribution doesn't matter → the FEATURE is the")
        print("    ceiling → need better features (retrain: alignment / ESM), not a probe fix.")


if __name__ == "__main__":
    main()
