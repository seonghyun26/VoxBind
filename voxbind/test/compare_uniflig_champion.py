#!/usr/bin/env python
"""Fair champion(vdW-lig) vs uniflig(uniform-lig) comparison on the SAME complexes.

uniflig's uniform-ligand 'default' voxels only cover 779/1320 test (436/817 val),
so its raw probe (0.566) is NOT comparable to champion's full-set 0.644. Here BOTH
encoders are probed on the COMMON pid set (champion ∩ uniflig ∩ split) with the same
head recipe → isolates ligand radius (vdW vs uniform).  NOTE confound: champion used
uniform masking, uniflig used atom_biased — so Δ bundles ligand-radius + mask-strategy.

  OMP_NUM_THREADS=2 nice -19 python -u test/compare_uniflig_champion.py [split]
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
    "champion (vdW-lig)":   "atomblob_density_gradmag_e49_v5_260705_ar_cvit_100m_v2_mask075.pt",
    "uniflig (uniform-lig)": "atomblob_density_gradmag_e49_v5_260810_cdg_100m_v2_uniflig_vdwpoc.pt",
}


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


def main():
    feats = {k: load_feats(v) for k, v in TAGS.items()}
    pK, sm = load_pK(), split_map(SPLIT)
    # COMMON pid set across BOTH encoders (+ pK + split) so both probe identical complexes.
    common = set(sm) & set(pK)
    for d in feats.values():
        common &= set(d)
    n = {s: sum(1 for p in common if sm[p] == s) for s in ("train", "val", "test")}
    print(f"device={DEVICE} split={SPLIT}  COMMON complexes: "
          f"train={n['train']} val={n['val']} test={n['test']}  (both encoders, identical set)")
    print(f"{'encoder':24}{'val ρ':>9}{'test r':>9}{'test ρ':>9}{'RMSE':>9}  (mse head, 5 seeds)")
    res = {}
    for label, fdict in feats.items():
        data = {}
        for s in ("train", "val", "test"):
            pids = sorted(p for p in common if sm[p] == s)
            data[s] = {"X": np.stack([fdict[p] for p in pids]).astype(np.float32),
                       "y": np.array([pK[p] for p in pids], dtype=np.float32)}
        rs = [train_one(data, s) for s in SEEDS]
        m = {k: float(np.mean([r[k] for r in rs])) for k in ("val_rho", "test_r", "test_rho", "test_rmse")}
        sd = float(np.std([r["test_rho"] for r in rs]))
        res[label] = m
        print(f"{label:24}{m['val_rho']:>9.3f}{m['test_r']:>9.3f}{m['test_rho']:>9.3f}±{sd:.3f}{m['test_rmse']:>8.3f}")
    a = res["champion (vdW-lig)"]["test_rho"]; b = res["uniflig (uniform-lig)"]["test_rho"]
    print(f"\nligand vdW − uniform (matched {n['test']} test): Δtest ρ = {a-b:+.3f}  "
          f"(champion {a:.3f} vs uniflig {b:.3f})")
    print("NOTE: Δ bundles ligand-radius AND mask-strategy (champion=uniform-mask, uniflig=atom_biased).")


if __name__ == "__main__":
    main()
