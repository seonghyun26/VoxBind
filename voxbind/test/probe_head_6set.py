"""probe_head_6set.py — 6-set (mse, 5-seed) probe of ONE feature cache with a
CONFIGURABLE MLP head, so we can compare the canonical default (128→64 tapered) vs
the smaller L1×64 head found best-for-ID30 in probe_capacity_id30.py.

Same 6 canonical test sets as tta_6set_probe.py (CASF-leaky EXCLUDED):
  FULL, CL3, CL3-ID60, CL3-ID30, CASF-nontrain, CASF-clean
Train on lp_edrscc_v2 TRAIN, early-stop on v2-VAL, predict the cohort union, mask.

Usage:
  cd voxbind && CUDA_VISIBLE_DEVICES=0 python test/probe_head_6set.py \
    --feat dataset/data/pdbbind/features/<cache>.pt --head default,l1x64 --seeds 5
"""
import argparse
import csv
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr

REPO = "/home/shpark/prj-denovo/VoxBind"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# identical HP to the canonical 01c probe + tta_6set_probe (hidden/lr/wd/bs/patience/epochs)
HP = dict(dropout=0.1, lr=1e-3, wd=1e-4, epochs=200, patience=30, bs=64)

HEADS = {                       # name -> (width, n_hidden_layers, tapered, act)
    "default":  (128, 2, True,  "relu"),   # casf-machinery 128→64 (NOT the results.html head)
    "l1x64":    (64, 1, False, "relu"),    # smallest, best-for-ID30 in the capacity sweep (ReLU)
    "l1x128":   (128, 1, False, "relu"),
    "l3x256":   (256, 3, False, "relu"),
    # ── FAITHFUL to results.html: MLP2 = SiLU, n_layers=2 → 640→hidden→1 ──
    "o1c128":   (128, 1, False, "silu"),   # = CURRENT results.html Table-1a head (--hidden 128)
    "o1c64":    (64, 1, False, "silu"),    # = proposed L1×64 in the ACTUAL 01c pipeline (--hidden 64)
    # ── activation-only ablation at the FIXED 640→128→1 head ──
    "act_relu": (128, 1, False, "relu"),
    "act_silu": (128, 1, False, "silu"),
    "act_gelu": (128, 1, False, "gelu"),
    # ── width / depth at SiLU (the winning activation) ──
    "silu64":    (64, 1, False, "silu"),
    "silu128":   (128, 1, False, "silu"),
    "silu256":   (256, 1, False, "silu"),
    "silu128d2": (128, 2, False, "silu"),
    # ── EXTENDED activation zoo at the FIXED 640→128→1 head ──
    "mish":      (128, 1, False, "mish"),
    "leaky":     (128, 1, False, "leaky_relu"),
    "elu":       (128, 1, False, "elu"),
    "selu":      (128, 1, False, "selu"),
    "tanh":      (128, 1, False, "tanh"),
    "softplus":  (128, 1, False, "softplus"),
    "prelu":     (128, 1, False, "prelu"),
    "hardswish": (128, 1, False, "hardswish"),
    # gated (GLU-family): first Linear → 2×h, split into (value, gate), value*act(gate)
    "swiglu":    (128, 1, False, "swiglu"),
    "geglu":     (128, 1, False, "geglu"),
    "reglu":     (128, 1, False, "reglu"),
    # ── HARDSWISH width / depth sweep (does MLP size matter with hardswish?) ──
    "hsw32":     (32, 1, False, "hardswish"),
    "hsw64":     (64, 1, False, "hardswish"),
    "hsw128":    (128, 1, False, "hardswish"),
    "hsw256":    (256, 1, False, "hardswish"),
    "hsw512":    (512, 1, False, "hardswish"),
    "hsw128d2":  (128, 2, False, "hardswish"),
    "hsw256d2":  (256, 2, False, "hardswish"),
    "hsw128d3":  (128, 3, False, "hardswish"),
    # ── plain Swish (β=1 ≡ SiLU) vs learnable-β Swish, at 640→128→1 ──
    "swish1":    (128, 1, False, "silu"),      # plain swish β=1 (== SiLU, anchor)
    "swishb":    (128, 1, False, "swish_b"),   # original Swish, learnable β
    # ── BIG heads (does ~ProFSA-scale capacity help us? ProFSA head = 1.74M) ──
    "big1m":     (1024, 2, False, "silu"),     # 640→1024→1024→1  ≈ 1.7M params
    "big2m":     (1152, 2, False, "silu"),     # 640→1152→1152→1  ≈ 2.07M params
    "big2m_d":   (512, 7, False, "silu"),      # 640→512×7→1       ≈ 2.0M params (deep)
}

class SwishBeta(nn.Module):
    """Original Swish x·sigmoid(β·x) with a LEARNABLE β (Ramachandran et al. 2017).
    β=1 reduces to SiLU; a trainable β lets the head interpolate toward linear (β→0)
    or ReLU-like (β→∞)."""
    def __init__(self, beta_init=1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


_ACTS = {"relu": nn.ReLU, "silu": nn.SiLU, "gelu": nn.GELU, "mish": nn.Mish,
         "leaky_relu": nn.LeakyReLU, "elu": nn.ELU, "selu": nn.SELU,
         "tanh": nn.Tanh, "softplus": nn.Softplus, "prelu": nn.PReLU,
         "hardswish": nn.Hardswish, "swish_b": SwishBeta}
_GATED = {"swiglu": nn.SiLU, "geglu": nn.GELU, "reglu": nn.ReLU}


class _GLUBlock(nn.Module):
    """GLU-family hidden block: Linear(d→2h) → split → value * gate_act(gate) → Dropout."""
    def __init__(self, d, h, gate_act, p):
        super().__init__()
        self.proj = nn.Linear(d, 2 * h)
        self.act = gate_act()
        self.drop = nn.Dropout(p)

    def forward(self, x):
        v, g = self.proj(x).chunk(2, dim=-1)
        return self.drop(v * self.act(g))


class MLP(nn.Module):
    def __init__(self, d, h, L, p=0.1, tapered=False, act="relu"):
        super().__init__()
        widths = ([h, h // 2] if tapered else [h] * L)
        layers, prev = [], d
        for w in widths:
            if act in _GATED:
                layers += [_GLUBlock(prev, w, _GATED[act], p)]
            else:
                layers += [nn.Linear(prev, w), _ACTS[act](), nn.Dropout(p)]
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


def cohorts(v2):
    D = f"{REPO}/base/_casf/cl123_seqfilter_5seed_260818"
    rows = list(csv.DictReader(open(f"{REPO}/voxbind/splits/casf2016_eval.csv")))
    casf = [r["pid"].lower() for r in rows]
    nontrain = {r["pid"].lower() for r in rows if r["in_v2train"] == "0"}
    clean = {p for p in casf if v2.get(p) not in ("train", "val")}
    full = {p for p, s in v2.items() if s == "test"}
    return {"FULL": full, "CL3": ids(f"{D}/cl123_test.txt"),
            "CL3-ID60": ids(f"{D}/cl123_test_novel60.txt"),
            "CL3-ID30": ids(f"{D}/cl123_test_novel30.txt"),
            "CASF-nontrain": nontrain, "CASF-clean": clean}


def train_predict(feats, pK, v2, test_pids, seed, h, L, tapered, act="relu"):
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
    model = MLP(Xtr.shape[1], h, L, HP["dropout"], tapered, act).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=HP["lr"], weight_decay=HP["wd"])
    best, bstate, bad = 1e9, None, 0
    n = Xtr_t.size(0)
    for ep in range(HP["epochs"]):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, HP["bs"]):
            j = perm[i:i + HP["bs"]]
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
            if bad >= HP["patience"]:
                break
    model.load_state_dict(bstate); model.eval()
    with torch.no_grad():
        pte = model(Xte_t).cpu().numpy() * ys + ym
    return dict(zip(te, pte)), dict(zip(te, yte))


def metrics(pred, yt, keep):
    ps = [p for p in pred if p in keep]
    if len(ps) < 3:
        return None
    y = np.array([yt[p] for p in ps]); yh = np.array([pred[p] for p in ps])
    return (spearmanr(y, yh).statistic, pearsonr(y, yh)[0],
            float(np.sqrt(((y - yh) ** 2).mean())), len(ps))


def run_head(feats, pK, v2, coh, seeds, h, L, tap, act="relu"):
    pool = set().union(*coh.values())
    per = {c: [] for c in coh}
    for s in range(seeds):
        pred, yt = train_predict(feats, pK, v2, pool, s, h, L, tap, act)
        for c, keep in coh.items():
            m = metrics(pred, yt, keep)
            if m:
                per[c].append(m)
    agg = {}
    for c, lst in per.items():
        a = np.array(lst)
        agg[c] = (a[:, 0].mean(), a[:, 0].std(), a[:, 2].mean(), int(a[0, 3]))  # rho_m, rho_sd, rmse_m, n
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", required=True)
    ap.add_argument("--head", default="default,l1x64", help="comma list of head names")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    feats, pK, v2 = load_feats(args.feat), load_pK(), v2_split()
    coh = cohorts(v2)
    order = ["FULL", "CL3", "CL3-ID60", "CL3-ID30", "CASF-nontrain", "CASF-clean"]
    print(f"cache={args.name or args.feat.split('/')[-1]}  device={DEVICE}  seeds={args.seeds}")
    heads = [hn.strip() for hn in args.head.split(",")]
    res = {hn: run_head(feats, pK, v2, coh, args.seeds, *HEADS[hn]) for hn in heads}

    # matrix: cohorts × heads (ρ mean), best per cohort marked *
    print(f"\n{'cohort':<15}{'n':>5}" + "".join(f"{hn:>14}" for hn in heads))
    print("-" * (20 + 14 * len(heads)))
    wins = {hn: 0 for hn in heads}
    for c in order:
        vals = {hn: res[hn][c][0] for hn in heads}
        best = max(vals, key=vals.get); wins[best] += 1
        line = f"{c:<15}{res[heads[0]][c][3]:>5}"
        for hn in heads:
            mark = "*" if hn == best else " "
            line += f"{vals[hn]:>12.3f}{mark} "
        print(line)
    # mean over the 6 cohorts + wins
    print("-" * (20 + 14 * len(heads)))
    meanline = f"{'MEAN(6)':<15}{'':>5}"
    means = {hn: np.mean([res[hn][c][0] for c in order]) for hn in heads}
    bestmean = max(means, key=means.get)
    for hn in heads:
        mark = "*" if hn == bestmean else " "
        meanline += f"{means[hn]:>12.3f}{mark} "
    print(meanline)
    print(f"{'#best-cohort':<15}{'':>5}" + "".join(f"{wins[hn]:>13} " for hn in heads))
    print(f"\nhead defs (width, n_hidden, tapered, act):")
    for hn in heads:
        print(f"  {hn:<10} {HEADS[hn]}")


if __name__ == "__main__":
    main()
