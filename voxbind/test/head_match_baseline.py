"""head_match_baseline.py — probe a frozen-feature BASELINE (GeoSSL / ProFSA) with the
same SiLU-128 head we use for our CDG rows, vs its NATIVE head, on the 6 canonical test
sets with the CORRECT per-table training protocol:
    FULL, CASF-nontrain, CASF-clean  -> train on lp_edrscc_v2 TRAIN
    CL3, CL3-ID60, CL3-ID30          -> train on lp_edrscc_v2_cl123 TRAIN
Answers "is our SiLU-128 head an unfair edge?" — if native≈SiLU-128 for the baseline,
head choice is not the source of the ranking.

Feature cache: a torch .pt whose payload is (a) a plain {pid: vec} dict, or (b) a dict
with a 'features' key holding that dict. dims are inferred (GeoSSL 128, ProFSA 1024).

Usage:
  python test/head_match_baseline.py --feat <path> --name GeoSSL \
    --native 256:relu --seeds 5
"""
import argparse
import csv
import glob
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr

REPO = "/home/shpark/prj-denovo/VoxBind"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_ACTS = {"relu": nn.ReLU, "silu": nn.SiLU, "gelu": nn.GELU}


class MLP(nn.Module):
    """d -> hidden -> 1, single hidden layer, configurable width+activation."""
    def __init__(self, d, hidden, act, p=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), _ACTS[act](), nn.Dropout(p),
                                 nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_feats(path):
    d = torch.load(path, weights_only=False)
    feats = d["features"] if isinstance(d, dict) and "features" in d else d
    return {str(k).lower(): np.asarray(v.numpy() if hasattr(v, "numpy") else v, dtype=np.float32)
            for k, v in feats.items()}


def load_pK():
    import pandas as pd
    lp = pd.read_csv(f"{REPO}/voxbind/dataset/data/pdbbind/raw/LP_PDBBind.csv").rename(
        columns={"Unnamed: 0": "pid"})
    lp["pid"] = lp["pid"].astype(str).str.lower()
    return {p: float(v) for p, v in zip(lp["pid"], lp["value"]) if v == v}


def split_of(path):
    m = {}
    for r in csv.DictReader(open(path)):
        m[r["pid"].lower()] = r["split"]
    return m


def ids(path):
    return set(l.strip().lower() for l in open(path) if l.strip())


def casf_eval():
    rows = list(csv.DictReader(open(f"{REPO}/voxbind/splits/casf2016_eval.csv")))
    pids = [r["pid"].lower() for r in rows]
    nontrain = {r["pid"].lower() for r in rows if r["in_v2train"] == "0"}
    return pids, nontrain


def train_predict(feats, pK, split, train_keys, test_pids, seed, hidden, act):
    torch.manual_seed(seed); np.random.seed(seed)
    lossf = nn.MSELoss()

    def arrs(pids):
        pids = [p for p in pids if p in feats and p in pK]
        return (np.stack([feats[p] for p in pids]).astype(np.float32),
                np.array([pK[p] for p in pids], dtype=np.float32), pids)

    tr = [p for p, s in split.items() if s == "train"]
    va = [p for p, s in split.items() if s == "val"]
    Xtr, ytr, _ = arrs(tr); Xva, yva, _ = arrs(va); Xte, yte, te = arrs(test_pids)
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    ym, ys = float(ytr.mean()), float(ytr.std() + 1e-6)
    prep = lambda A: torch.tensor((A - mu) / sd, device=DEVICE)
    Xtr_t, Xva_t, Xte_t = prep(Xtr), prep(Xva), prep(Xte)
    ytr_t = torch.tensor((ytr - ym) / ys, device=DEVICE)
    model = MLP(Xtr.shape[1], hidden, act).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best, bstate, bad = 1e9, None, 0
    n = Xtr_t.size(0)
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
    if len(ps) < 3:
        return None
    y = np.array([yt[p] for p in ps]); yh = np.array([pred[p] for p in ps])
    return spearmanr(y, yh).statistic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--native", default="256:relu", help="native head 'hidden:act' (GeoSSL 256:relu)")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    feats, pK = load_feats(args.feat), load_pK()
    dim = len(next(iter(feats.values())))
    v2 = split_of(f"{REPO}/voxbind/splits/lp_edrscc_v2.csv")
    cl = split_of(f"{REPO}/voxbind/splits/lp_edrscc_v2_cl123.csv")
    D = f"{REPO}/base/_casf/cl123_seqfilter_5seed_260818"
    cl3 = {"CL3": ids(f"{D}/cl123_test.txt"), "CL3-ID60": ids(f"{D}/cl123_test_novel60.txt"),
           "CL3-ID30": ids(f"{D}/cl123_test_novel30.txt")}
    casf_pids, nontrain = casf_eval()
    clean = {p for p in casf_pids if v2.get(p) not in ("train", "val")}
    full = {p for p, s in v2.items() if s == "test"}
    nh, na = args.native.split(":")
    heads = {"native": (int(nh), na), "silu128": (128, "silu")}

    print(f"{args.name}: dim={dim}  device={DEVICE}  seeds={args.seeds}  "
          f"native={args.native}  vs  silu128 (d->128->1 SiLU)\n")

    # cohort -> (split, train_pool_predict_pids, mask)
    def run(hidden, act):
        out = {}
        for s in range(args.seeds):
            # v2-train arm: FULL + CASF
            pv2, yv2 = train_predict(feats, pK, v2, None, list(full) + casf_pids, s, hidden, act)
            # cl123-train arm: CL3 cohorts
            pcl, ycl = train_predict(feats, pK, cl, None,
                                     list(set().union(*cl3.values())), s, hidden, act)
            for name, keep in [("FULL", full), ("CASF-nontrain", nontrain), ("CASF-clean", clean)]:
                out.setdefault(name, []).append(rho(pv2, yv2, keep))
            for name, keep in cl3.items():
                out.setdefault(name, []).append(rho(pcl, ycl, keep))
        return {k: (np.mean([x for x in v if x is not None]),
                    np.std([x for x in v if x is not None])) for k, v in out.items()}

    res = {hn: run(h, a) for hn, (h, a) in heads.items()}
    order = ["FULL", "CL3", "CL3-ID60", "CL3-ID30", "CASF-nontrain", "CASF-clean"]
    print(f"{'cohort':<15}{'native ρ':>16}{'silu128 ρ':>16}{'Δ(silu-nat)':>14}")
    print("-" * 61)
    for c in order:
        nv, ns = res["native"][c][0], res["silu128"][c][0]
        print(f"{c:<15}{nv:>10.3f}±{res['native'][c][1]:.3f}{ns:>10.3f}±{res['silu128'][c][1]:.3f}{ns-nv:>+14.3f}")


if __name__ == "__main__":
    main()
