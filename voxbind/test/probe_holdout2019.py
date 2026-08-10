"""probe_holdout2019.py — evaluate the frozen probe (CDG / C) on the 2019 temporal holdout.

Extracts pooled features for the holdout complexes (voxels_v5 crops) with the champion / coords
encoder, reuses the cached lp_edrscc_v2 features for head training, then trains the MLP head on
v2 train (early-stop on v2 val) and predicts the holdout. All holdout complexes are 2019+ and NOT
in our train/val, so the whole set is a clean temporal holdout (no leaky/clean split).

Loss configurable (mse | mse+corr). Writes base/_casf/{model}_holdout2019.json.

Usage: cd voxbind && CUDA_VISIBLE_DEVICES=4 python test/probe_holdout2019.py \
         --exp_dir model_zoo/champion --cond atomblob_density_gradmag --model CDG --loss mse+corr --aux_weight 5
"""
import gemmi  # noqa: F401
import argparse, csv, importlib.util, json, os, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr

VOX = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VOX))
_spec = importlib.util.spec_from_file_location("probe01c", VOX / "dataset" / "01c_pdbbind_probe.py")
P = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(P)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FD = VOX / "dataset" / "data" / "pdbbind" / "features"


def pearson_term(p, y):
    p = p - p.mean(); y = y - y.mean()
    return 1.0 - (p * y).sum() / (p.norm() * y.norm() + 1e-8)


class MLP(nn.Module):
    def __init__(s, d, h=128, p=0.1):
        super().__init__(); s.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(p),
                                                  nn.Linear(h, h // 2), nn.ReLU(), nn.Dropout(p), nn.Linear(h // 2, 1))
    def forward(s, x): return s.net(x).squeeze(-1)


def extract_holdout_feats(exp_dir, cond, epoch, pids):
    cfg = OmegaConf.load(Path(exp_dir) / "cfg.yaml")
    spec = P.infer_feature_spec(cond, cfg, "auto")
    vox_dir = P.voxel_dir_for("v5"); atom_dir = P.atom_dir_for(vox_dir, spec.atom_source)
    dens_dir = vox_dir / "density"
    enc = P.load_encoder(Path(exp_dir), epoch, DEVICE, cfg=cfg); n_in = enc.n_in_channels
    G = int(cfg.vox.grid_dim); pe = 8; npatch = (G // pe) ** 3
    ds = P._ExtractDataset(pids, cond, n_in, atom_dir, dens_dir, spec.input_mode, spec.with_gradmag, None)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=4, collate_fn=P._extract_collate)
    feats = {}
    for bpids, x, errs in tqdm(loader, desc="extract holdout"):
        if x is None: continue
        x = x.to(DEVICE)
        with torch.no_grad():
            tok = P.encode_tokens(enc, x)
        B, D = tok.shape[0], tok.shape[2]; nG = tok.shape[1] // npatch
        g = tok.reshape(B, nG, npatch, D).mean(1).mean(1)   # pooled (B, D)
        for i, pid in enumerate(bpids): feats[pid] = g[i].float().cpu().numpy()
    return feats


def load_cache(cond, epoch, tag):
    d = torch.load(FD / f"{cond}_e{epoch}_v5_{tag}.pt", weights_only=False)
    f = d.get("features", d.get("feat"))
    return {p: np.asarray(v.numpy() if hasattr(v, "numpy") else v, dtype=np.float32) for p, v in f.items()}


def curcode_v2_feats(exp_dir, cond, epoch, v2_split):
    """Extract v2 train/val/test features with the SAME current-code path as the holdout
    (extract_holdout_feats), so train and test features are guaranteed consistent. The
    on-disk lp_edrscc_v2 caches were built pre-Jul13 density_vit and are numerically
    inconsistent (orthogonal) with current forward_features → mixing them collapses the
    head. Cached to disk keyed by exp_dir basename so re-runs are fast."""
    V = VOX / "dataset/data/pdbbind/voxels_v5"
    LA = VOX / "dataset/data/pdbbind/voxels_ligvdw/atoms"
    needs = cond.startswith("atomblob_density") or "gradmag" in cond
    def ready(p):
        if not (LA / f"{p}.npy").exists():
            return False
        if needs:
            return (V / "density" / f"{p}.npy").exists() and (V / "gradmag/density" / f"{p}.npy").exists()
        return True
    pids = [p for p in v2_split if ready(p)]
    ck = FD / f"{cond}_e{epoch}_v5_CURCODE_{Path(exp_dir).name}.pt"
    if ck.exists():
        d = torch.load(ck, weights_only=False)
        cached = d["features"]
        miss = [p for p in pids if p not in cached]
        if not miss:
            return {p: np.asarray(cached[p], np.float32) for p in pids}
    print(f"  [curcode] extracting {len(pids)} v2 features with current code → {ck.name}")
    feats = extract_holdout_feats(exp_dir, cond, epoch, pids)
    torch.save({"features": {p: feats[p] for p in feats}}, ck)
    return {p: np.asarray(feats[p], np.float32) for p in feats}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", required=True)
    ap.add_argument("--cond", required=True)
    ap.add_argument("--cache_tag", required=True, help="feature cache tag for v2 (e.g. 260705_ar_cvit_100m_v2_mask075)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--epoch", type=int, default=49)
    ap.add_argument("--loss", default="mse")
    ap.add_argument("--aux_weight", type=float, default=5.0)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()

    pK = {r["pid"].lower(): float(r["pK"]) for r in csv.DictReader(open(VOX / "splits" / "holdout2019_eval.csv"))}
    v2 = {r["pid"].lower(): r["split"] for r in csv.DictReader(open(VOX / "splits" / "lp_edrscc_v2.csv"))}
    lp = P.load_lp_index(P.LP_CSV); v2pk = {p: v for p, v in zip(lp["pdb_id"], lp["pK"]) if v == v}

    # holdout pids that are CDG-ready (have crops); extract their features.
    # atoms live in the ligvdw precompute (the champion's atom_source=ligvdw), density+gradmag in voxels_v5.
    V = VOX / "dataset/data/pdbbind/voxels_v5"
    LA = VOX / "dataset/data/pdbbind/voxels_ligvdw/atoms"
    ready = [p for p in pK if (LA / f"{p}.npy").exists() and (V / "density" / f"{p}.npy").exists()
             and (V / "gradmag/density" / f"{p}.npy").exists()]
    print(f"[{a.model}] holdout CDG-ready: {len(ready)}")
    hold_f = extract_holdout_feats(a.exp_dir, a.cond, a.epoch, ready)
    # v2 train/val/test features from the SAME current-code path (NOT the stale on-disk cache)
    cache = curcode_v2_feats(a.exp_dir, a.cond, a.epoch, v2)
    tr = [p for p in cache if v2.get(p) == "train" and p in v2pk]
    va = [p for p in cache if v2.get(p) == "val" and p in v2pk]
    te_v2 = [p for p in cache if v2.get(p) == "test" and p in v2pk]   # in-domain sanity
    te = [p for p in hold_f if p in pK]
    Xtr = np.stack([cache[p] for p in tr]); ytr = np.array([v2pk[p] for p in tr], np.float32)
    Xva = np.stack([cache[p] for p in va]); yva = np.array([v2pk[p] for p in va], np.float32)
    Xte = np.stack([hold_f[p] for p in te]); yte = np.array([pK[p] for p in te], np.float32)
    Xv2 = np.stack([cache[p] for p in te_v2]); yv2 = np.array([v2pk[p] for p in te_v2], np.float32)
    print(f"  train {len(tr)} val {len(va)} holdout-test {len(te)}  v2-test {len(te_v2)}  dim {Xtr.shape[1]}")

    mse = nn.MSELoss()
    def lossf(p, y): return mse(p, y) + (a.aux_weight * pearson_term(p, y) if a.loss == "mse+corr" else 0.0)
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    ym, ys = float(ytr.mean()), float(ytr.std() + 1e-6)
    def prep(X): return torch.tensor((X - mu) / sd, dtype=torch.float32, device=DEVICE)
    Xtr_t, Xva_t, Xte_t, Xv2_t = prep(Xtr), prep(Xva), prep(Xte), prep(Xv2)
    ytr_t = torch.tensor((ytr - ym) / ys, device=DEVICE)
    res = []; pt_all = []; v2rho = []
    for seed in range(a.seeds):
        torch.manual_seed(seed); np.random.seed(seed)
        m = MLP(Xtr.shape[1]).to(DEVICE); opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
        best, bstate, bad = -1e9, None, 0; n = Xtr_t.size(0)
        for ep in range(200):
            m.train(); perm = torch.randperm(n, device=DEVICE)
            for i in range(0, n, 64):
                idx = perm[i:i + 64]
                if idx.numel() < 4: continue
                opt.zero_grad(); lossf(m(Xtr_t[idx]), ytr_t[idx]).backward(); opt.step()
            m.eval()
            with torch.no_grad(): pv = m(Xva_t).cpu().numpy() * ys + ym
            vs = spearmanr(pv, yva).statistic
            if vs > best: best, bstate, bad = vs, {k: v.clone() for k, v in m.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 30: break
        m.load_state_dict(bstate); m.eval()
        with torch.no_grad():
            pt = m(Xte_t).cpu().numpy() * ys + ym
            pv2 = m(Xv2_t).cpu().numpy() * ys + ym
        pt_all.append(pt)
        v2rho.append(spearmanr(pv2, yv2).statistic)
        res.append((pearsonr(pt, yte)[0], spearmanr(pt, yte).statistic, np.sqrt(((pt - yte) ** 2).mean())))
    print(f"  [sanity] current-code v2-test ρ={np.mean(v2rho):.3f} (published ~0.62-0.64 → confirms encoder OK)")
    # dump seed-averaged per-complex predictions for the common-subset comparison
    tag_p = a.model + ("_corr5" if a.loss == "mse+corr" else "")
    with open(VOX.parent / "base" / "_casf" / f"{tag_p}_holdout2019_preds.csv", "w") as _f:
        _f.write("pid,pred,y\n")
        pm = np.mean(pt_all, axis=0)
        for _p, _pr, _y in zip(te, pm, yte):
            _f.write(f"{_p},{_pr},{_y}\n")
    # per-seed preds too → the common-set table can report mean±std (not just the ensemble point)
    for _s, _pt in enumerate(pt_all):
        with open(VOX.parent / "base" / "_casf" / f"{tag_p}_holdout2019_preds_seed{_s}.csv", "w") as _f:
            _f.write("pid,pred,y\n")
            for _p, _pr, _y in zip(te, _pt, yte):
                _f.write(f"{_p},{_pr},{_y}\n")
    a_ = np.array(res)
    out = {"model": a.model, "loss": a.loss, "n": len(te),
           "holdout": {"pearson": {"mean": float(a_[:, 0].mean()), "std": float(a_[:, 0].std())},
                       "spearman": {"mean": float(a_[:, 1].mean()), "std": float(a_[:, 1].std())},
                       "rmse": {"mean": float(a_[:, 2].mean()), "std": float(a_[:, 2].std())}, "n": len(te)}}
    tag = a.model + ("_corr5" if a.loss == "mse+corr" else "")
    json.dump(out, open(VOX.parent / "base" / "_casf" / f"{tag}_holdout2019.json", "w"), indent=2)
    print(f"  {a.model} [{a.loss}] holdout n={len(te)}  r={a_[:,0].mean():.3f}  ρ={a_[:,1].mean():.3f}  RMSE={a_[:,2].mean():.3f}")


if __name__ == "__main__":
    main()
