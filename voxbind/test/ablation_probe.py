#!/usr/bin/env python
"""Uniform re-probe of the 260813 CDG-encoder campaign → one JSON for results.html §1.1.

Every entry is probed with the SAME frozen mean-pool → MLP2 head (mse, 5 seeds) on
lp_edrscc_v2 so val ρ / test r / test ρ / RMSE are directly comparable. Ensembles concat
member features; the mse+corr row swaps the head loss. Writes notebook/html/ablation_cdg.json.

  OMP_NUM_THREADS=2 nice -19 python -u test/ablation_probe.py
  PROBE_SPLIT=lp_edrscc_v2_cl123 PROBE_OUT=../notebook/html/ablation_cdg_cl123.json \
    python -u test/ablation_probe.py
"""
import os, sys, json
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr

torch.set_num_threads(int(os.environ.get("PROBE_THREADS", "2")))
REPO = "/home/shpark/prj-denovo/VoxBind"
FDIR = f"{REPO}/voxbind/dataset/data/pdbbind/features"
DEVICE = os.environ.get("PROBE_DEVICE", "cpu")
HP = dict(hidden=128, dropout=0.1, lr=1e-3, wd=1e-4, max_epochs=300, patience=30, bs=128)
SEEDS = list(range(int(os.environ.get("PROBE_SEEDS", "5"))))
SPLIT = os.environ.get("PROBE_SPLIT", "lp_edrscc_v2")
P = "atomblob_density_gradmag_e49_v5_"        # standard CDG feature prefix

# num, label, group, kind, members(feature files), head, what-it-tests, verdict
E = [
 (1,"Champion (C+D+G)","Baseline","single",[P+"260705_ar_cvit_100m_v2_mask075.pt"],"mse",
    "vdW-radii C+D+G, ChannelViT [7,4,2], uniform mask 0.75 — the reference.","ref"),
 (2,"Coords-only (C)","Modality","single",["atomblob_e49_v5_260723_ar_cvit_100m_v2_mask075_coords.pt"],"mse",
    "Drop density+gradmag channels — is electron density worth its capacity?","hurt"),
 (3,"+ mse+corr head","Probe head","head",[P+"260705_ar_cvit_100m_v2_mask075.pt"],"mse+corr",
    "Add a Pearson-corr aux term to the probe MLP loss (GenScore-style).","help"),
 (4,"Ensemble ×3","Ensemble","single",[P+"260705_ar_cvit_100m_v2_mask075.pt",
    P+"260725_ar_cvit_100m_v3_m090.pt",P+"260725_ar_cvit_100m_v3_m095.pt"],"mse",
    "Concat champion + v3_m090 + v3_m095 (mixes data v2/v3 AND mask 0.75/0.90/0.95).","help"),
 (5,"Ensemble: mask-only 085+095","Ensemble","single",[P+"260725_ar_cvit_100m_v3_m085.pt",
    P+"260725_ar_cvit_100m_v3_m095.pt"],"mse",
    "Concat two v3 encoders that differ ONLY in mask ratio (0.85/0.95) — pure mask diversity, no data confound.","help"),
 (6,"Longer 100 ep","Schedule","single",["atomblob_density_gradmag_e99_v5_260705_ar_cvit_100m_v2_mask075_e100.pt"],"mse",
    "Train 100 epochs instead of 50 — is the champion undertrained?","neutral"),
 (7,"Clean data v3","Data","single",[P+"260808_cdg_100m_v3_m075.pt"],"mse",
    "Dedup+resolution+pocket-RSCC-cleaned v3 corpus at champion mask 0.75.","neutral"),
 (8,"Uniform mask 0.50","Masking","single",[P+"260705_ar_cvit_100m_v2_mask050.pt"],"mse",
    "Lower the uniform mask ratio 0.75 → 0.50.","neutral"),
 (9,"Atom-biased mask 0.50","Masking","single",[P+"260810_cdg_100m_v2_atommask050.pt"],"mse",
    "Mask blocks near atoms (empty space wasteful), ratio 0.50 — user idea.","hurt"),
 (10,"Atom-biased mask 0.50 (46M)","Masking","single",[P+"260809_cdg_45m_v2_atommask050.pt"],"mse",
    "Same atom-biased idea at 46M (size control).","hurt"),
 (11,"3D RoPE","Pos-enc","single",[P+"260809_cdg_100m_v2_m075_rope3d.pt"],"mse",
    "Swap learnable position encoding → axial 3D RoPE.","hurt"),
 (12,"data2vec aux","Objective","single",[P+"260806_cdg_100m_v2_d2vaux05.pt"],"mse",
    "Add a data2vec latent-prediction aux objective on top of MAE.","hurt"),
 (13,"Uniform ligand (orig VoxBind)","Radius","single",[P+"260810_cdg_100m_v2_uniflig_vdwpoc.pt"],"mse",
    "Uniform-radius ligand (0.5) + vdW pocket = original VoxBind repr (matched 779; uniform-lig voxels cover 59%).","hurt"),
 (14,"All-uniform radius","Radius","single",[P+"260808_cdg_100m_v2_m075_uniformrad.pt"],"mse",
    "Uniform 0.5 radius for BOTH ligand and pocket — blobs lose steric size.","hurt"),
 (15,"R2MAE variable mask (v2)","Masking","single",[P+"260813_cdg_100m_v2_varmask6090.pt"],"mse",
    "Draw mask ratio r~U[0.6,0.9] per batch in ONE encoder — fold the ensemble's mask-diversity into a single model at 1x cost. Below champion: single-encoder mask-variety is NOT ensemble diversity (that needs separate weights).","hurt"),
 (16,"R2MAE variable mask (v2.2)","Masking","single",[P+"260813_cdg_100m_v22_varmask6090.pt"],"mse",
    "Same R2MAE on clean v2.2 (v2 minus 5,709 out-of-vocab-ligand complexes). +0.01 over the v2 varmask — data cleaning helps — but still below champion.","neutral"),
 (17,"Channel-sep [7,4,1,1] (v2.2)","Arch","single",[P+"260813_cdg_100m_v22_g7411.pt"],"mse",
    "Density &amp; gradmag as SEPARATE ChannelViT groups [7,4,1,1] vs grouped [7,4,2], on clean v2.2. +0.01 over dirty-v2 g7411 but grouped still wins — separating the physics channels hurts.","neutral"),
]


class MLP2(nn.Module):
    def __init__(s, d, h=128, dr=0.1):
        super().__init__(); s.net = nn.Sequential(nn.Linear(d, h), nn.SiLU(), nn.Dropout(dr), nn.Linear(h, 1))
    def forward(s, x): return s.net(x).squeeze(-1)

def pcorr_loss(p, y):
    p = p - p.mean(); y = y - y.mean()
    return 1.0 - (p * y).sum() / (p.norm() * y.norm() + 1e-8)

def load_feats(fn):
    d = torch.load(f"{FDIR}/{fn}", weights_only=False); f = d.get("features", d.get("feat"))
    return {p.lower(): np.asarray(v.numpy() if hasattr(v, "numpy") else v, dtype=np.float32) for p, v in f.items()}

def load_pK():
    import pandas as pd
    lp = pd.read_csv(f"{REPO}/voxbind/dataset/data/pdbbind/raw/LP_PDBBind.csv").rename(columns={"Unnamed: 0": "pid"})
    lp["pid"] = lp["pid"].astype(str).str.lower()
    return {p: float(v) for p, v in zip(lp["pid"], lp["value"]) if v == v}

def split_map():
    sys.path.insert(0, REPO); from voxbind.splits import load_split
    sp = load_split(SPLIT); return {p: s for s in ("train", "val", "test") for p in sp[s]}

def train_one(data, head, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr = (torch.from_numpy(data["train"][k]).to(DEVICE) for k in ("X", "y"))
    Xva, yva = (torch.from_numpy(data["val"][k]).to(DEVICE) for k in ("X", "y"))
    Xte, yte = (torch.from_numpy(data["test"][k]).to(DEVICE) for k in ("X", "y"))
    m = MLP2(Xtr.shape[1], HP["hidden"], HP["dropout"]).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=HP["lr"], weight_decay=HP["wd"]); n = Xtr.shape[0]
    lossf = (lambda p, y: F.mse_loss(p, y) + 3.0 * pcorr_loss(p, y)) if head == "mse+corr" else (lambda p, y: F.mse_loss(p, y))
    bv, bs, since = -np.inf, None, 0; yv, yt = yva.cpu().numpy(), yte.cpu().numpy()
    for _ in range(HP["max_epochs"]):
        m.train(); perm = torch.randperm(n, device=DEVICE)
        for s in range(0, n, HP["bs"]):
            idx = perm[s:s+HP["bs"]]
            if idx.numel() < 4: continue
            opt.zero_grad(); lossf(m(Xtr[idx]), ytr[idx]).backward(); opt.step()
        m.eval()
        with torch.no_grad(): vs = spearmanr(m(Xva).cpu().numpy(), yv).statistic
        if vs > bv: bv, since, bs = vs, 0, {k: v.detach().clone() for k, v in m.state_dict().items()}
        else:
            since += 1
            if since >= HP["patience"]: break
    m.load_state_dict(bs); m.eval()
    with torch.no_grad(): pt = m(Xte).cpu().numpy()
    return dict(val_rho=float(bv), test_r=float(pearsonr(pt, yt).statistic),
                test_rho=float(spearmanr(pt, yt).statistic), test_rmse=float(np.sqrt(((pt-yt)**2).mean())))

def build(members, pK, sm):
    dicts = [load_feats(f) for f in members]
    shared = set(sm) & set(pK)
    for d in dicts: shared &= set(d)
    out = {}
    for s in ("train", "val", "test"):
        pids = sorted(p for p in shared if sm[p] == s)
        out[s] = {"X": np.concatenate([np.stack([d[p] for p in pids]) for d in dicts], 1).astype(np.float32),
                  "y": np.array([pK[p] for p in pids], dtype=np.float32)}
    return out, len(out["test"]["y"])

def main():
    pK, sm = load_pK(), split_map(); rows = []
    for num, label, group, kind, members, head, what, verdict in E:
        try:
            data, nte = build(members, pK, sm)
            rs = [train_one(data, head, s) for s in SEEDS]
            metric_keys = ("val_rho", "test_r", "test_rho", "test_rmse")
            agg = {k: float(np.mean([r[k] for r in rs])) for k in metric_keys}
            agg.update({f"{k}_std": float(np.std([r[k] for r in rs])) for k in metric_keys})
            rows.append(dict(num=num, label=label, group=group, what=what, verdict=verdict, n_test=nte, **agg))
            print(f"#{num:2d} {label:32s} n={nte:4d}  "
                  f"r={agg['test_r']:.3f}±{agg['test_r_std']:.3f}  "
                  f"ρ={agg['test_rho']:.3f}±{agg['test_rho_std']:.3f}  "
                  f"RMSE={agg['test_rmse']:.3f}±{agg['test_rmse_std']:.3f}  "
                  f"valρ={agg['val_rho']:.3f}±{agg['val_rho_std']:.3f}")
        except Exception as ex:
            print(f"#{num:2d} {label:32s} FAILED: {ex}")
    outp = os.environ.get("PROBE_OUT", f"{REPO}/notebook/html/ablation_cdg.json")
    json.dump(rows, open(outp, "w"), indent=1); print("wrote", outp, f"({len(rows)} rows)")

if __name__ == "__main__":
    main()
