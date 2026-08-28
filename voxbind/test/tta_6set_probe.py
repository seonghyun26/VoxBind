"""tta_6set_probe.py — does test-time rotation averaging (TTA) help the CDG v2 encoder,
especially on the CL3-ID30 protein-novelty axis?

Compares two frozen-feature caches of the SAME encoder (260806_cdg_100m_v2_ep100 e25):
  --feat_base : single-orientation pooled features (K=1)
  --feat_tta  : pooled features averaged over the 24 proper cube rotations (K=24),
                produced by `01c_pdbbind_probe.py features --tta_rot 24`.

Both arms run the IDENTICAL probe (casf-machinery MLP, mse, 5 seeds): train on
lp_edrscc_v2 TRAIN, early-stop on v2 VAL, predict the union of all test cohorts, then
mask into the 6 canonical test sets. Absolute numbers are casf-machinery (≈0.02-0.03
below the 01c pipeline on FULL/CL3) — the apples-to-apples signal is the per-cohort
Δ(TTA − base), same pipeline both arms.

The 6 canonical sets (CASF-leaky EXCLUDED):
  FULL, CL3, CL3-ID60, CL3-ID30, CASF-nontrain, CASF-clean

Usage:
  cd voxbind && CUDA_VISIBLE_DEVICES=4 python test/tta_6set_probe.py \
    --feat_base dataset/data/pdbbind/features/atomblob_density_gradmag_e25_v5_260806_cdg_100m_v2_ep100_e25.pt \
    --feat_tta  dataset/data/pdbbind/features/atomblob_density_gradmag_e25_v5_260806_cdg_100m_v2_ep100_e25_tta24.pt
"""
import argparse
import csv
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr

REPO = "/home/shpark/prj-denovo/VoxBind"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HP = dict(hidden=128, dropout=0.1, lr=1e-3, wd=1e-4, epochs=200, patience=30, bs=64)


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


def ids(path):
    return set(l.strip().lower() for l in open(path) if l.strip())


def cohorts(v2):
    """The 6 canonical test sets as pid-sets (CASF-leaky excluded)."""
    D = f"{REPO}/base/_casf/cl123_seqfilter_5seed_260818"
    rows = list(csv.DictReader(open(f"{REPO}/voxbind/splits/casf2016_eval.csv")))
    casf = [r["pid"].lower() for r in rows]
    nontrain = {r["pid"].lower() for r in rows if r["in_v2train"] == "0"}
    clean = {p for p in casf if v2.get(p) not in ("train", "val")}
    full = {p for p, s in v2.items() if s == "test"}
    return {
        "FULL":          full,
        "CL3":           ids(f"{D}/cl123_test.txt"),
        "CL3-ID60":      ids(f"{D}/cl123_test_novel60.txt"),
        "CL3-ID30":      ids(f"{D}/cl123_test_novel30.txt"),
        "CASF-nontrain": nontrain,
        "CASF-clean":    clean,
    }


def train_predict(feats, pK, v2, test_pids, seed):
    """Train MLP on v2 train, early-stop on v2 val, predict `test_pids`. mse loss."""
    torch.manual_seed(seed); np.random.seed(seed)
    loss_fn = nn.MSELoss()

    def arrs(pids):
        pids = [p for p in pids if p in feats and p in pK]
        X = np.stack([feats[p] for p in pids]).astype(np.float32)
        y = np.array([pK[p] for p in pids], dtype=np.float32)
        return X, y, pids

    tr = [p for p, s in v2.items() if s == "train"]
    va = [p for p, s in v2.items() if s == "val"]
    Xtr, ytr, _ = arrs(tr)
    Xva, yva, _ = arrs(va)
    Xte, yte, te_pids = arrs(test_pids)

    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    ym, ys = float(ytr.mean()), float(ytr.std() + 1e-6)
    prep = lambda X: torch.tensor((X - mu) / sd, device=DEVICE)
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
    return dict(zip(te_pids, pte)), dict(zip(te_pids, yte))


def metrics(pred, ytrue, keep):
    pids = [p for p in pred if p in keep]
    if len(pids) < 3:
        return None
    y = np.array([ytrue[p] for p in pids])
    yh = np.array([pred[p] for p in pids])
    return (float(pearsonr(y, yh)[0]), float(spearmanr(y, yh)[0]),
            float(np.sqrt(((y - yh) ** 2).mean())), len(pids))


def run_arm(feats, pK, v2, coh, seeds):
    test_pool = set().union(*coh.values())
    per = {c: [] for c in coh}
    for s in range(seeds):
        pred, ytrue = train_predict(feats, pK, v2, test_pool, s)
        for c, keep in coh.items():
            m = metrics(pred, ytrue, keep)
            if m:
                per[c].append(m)
    agg = {}
    for c, lst in per.items():
        a = np.array(lst)  # (seeds, 4) = r, rho, rmse, n
        agg[c] = dict(r=(a[:, 0].mean(), a[:, 0].std()),
                      rho=(a[:, 1].mean(), a[:, 1].std()),
                      rmse=(a[:, 2].mean(), a[:, 2].std()), n=int(a[0, 3]))
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_base", required=True)
    ap.add_argument("--feat_tta", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    pK, v2 = load_pK(), v2_split()
    coh = cohorts(v2)
    print(f"device={DEVICE} seeds={args.seeds}")
    print("cohorts:", {c: len(s) for c, s in coh.items()}, "\n")

    print("loading base features...");  fb = load_feats(args.feat_base)
    print("loading tta  features...");  ft = load_feats(args.feat_tta)
    print(f"base n={len(fb):,}  tta n={len(ft):,}\n")

    base = run_arm(fb, pK, v2, coh, args.seeds)
    tta  = run_arm(ft, pK, v2, coh, args.seeds)

    order = ["FULL", "CL3", "CL3-ID60", "CL3-ID30", "CASF-nontrain", "CASF-clean"]
    print(f"\n{'cohort':<15}{'n':>5}  {'base ρ':>13}  {'TTA ρ':>13}  {'Δρ':>7}   "
          f"{'base r':>7} {'TTA r':>7}  {'base RMSE':>9} {'TTA RMSE':>9}")
    print("-" * 108)
    for c in order:
        b, t = base[c], tta[c]
        dr = t["rho"][0] - b["rho"][0]
        flag = "  <-- novelty" if c == "CL3-ID30" else ""
        print(f"{c:<15}{b['n']:>5}  {b['rho'][0]:>6.3f}±{b['rho'][1]:.3f}  "
              f"{t['rho'][0]:>6.3f}±{t['rho'][1]:.3f}  {dr:>+7.3f}   "
              f"{b['r'][0]:>7.3f} {t['r'][0]:>7.3f}  {b['rmse'][0]:>9.3f} {t['rmse'][0]:>9.3f}{flag}")


if __name__ == "__main__":
    main()
