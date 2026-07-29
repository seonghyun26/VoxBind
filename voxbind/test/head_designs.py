"""head_designs.py — TerraBind-inspired probe HEAD designs on the frozen champion.

The frozen affinity probe currently mean-pools all patch tokens → one vector → MLP2.
This tests richer readouts that inject the ligand↔pocket INTERACTION inductive bias
(TerraBind's pair representation), operating on the FROZEN champion encoder
(100M v2 mask0.75). No re-pretraining — extract once, train many heads.

Extraction: for each complex, encode → per-patch tokens (groups pooled) → soft-pool
into g_mean / g_lig / g_poc using per-patch ligand vs pocket OCCUPANCY (from the input
voxels, channels 0..n_lig-1 = ligand, n_lig..n_lig+n_poc-1 = pocket).

Heads (train_one MLP2 on the assembled feature vector, 3 seeds, lp_edrscc_v2):
  mean      : g_mean (640)                          — BASELINE, must reproduce champion ρ≈0.644
  A_bilinear: [g_lig, g_poc, g_lig*g_poc, |g_lig-g_poc|] (2560) — interaction bias, ~0 extra structure
  A_concat  : [g_lig, g_poc] (1280)                 — split-pool control (does splitting alone help?)

Usage:  python test/head_designs.py [--extract] [--heads mean,A_bilinear,A_concat]
Cached tokens → test/head_feats_champion.pt so head sweeps skip re-extraction.
"""
import gemmi  # noqa: F401 — before torch
import argparse, importlib.util, sys
from pathlib import Path
import numpy as np, torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr

VOX = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VOX))
_spec = importlib.util.spec_from_file_location("probe01c", VOX / "dataset" / "01c_pdbbind_probe.py")
P = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(P)

DEVICE = "cuda:1"
EXP_DIR = VOX / "exps" / "260705_ar_cvit_100m_v2_mask075"
EPOCH, VOXV, COND = 49, "v5", "atomblob_density_gradmag"
CACHE = VOX / "test" / "head_feats_champion.pt"


def extract():
    cfg = OmegaConf.load(EXP_DIR / "cfg.yaml")
    spec = P.infer_feature_spec(COND, cfg, "auto")
    n_lig = int(cfg.model.get("n_channels_ligand", 7))
    n_poc = int(cfg.model.get("n_channels_pocket", 4))
    vox_dir = P.voxel_dir_for(VOXV); atom_dir = P.atom_dir_for(vox_dir, spec.atom_source)
    dens_dir = vox_dir / "density"
    enc = P.load_encoder(EXP_DIR, EPOCH, DEVICE, cfg=cfg)
    n_in = enc.n_in_channels
    smap, scheme = P.load_frozen_split_map("lp_edrscc_v2")
    pids = sorted(smap.keys())
    ds = P._ExtractDataset(pids, COND, n_in, atom_dir, dens_dir, spec.input_mode, spec.with_gradmag, None)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0, collate_fn=P._extract_collate)

    G = int(cfg.vox.grid_dim); pe = 8; npatch = (G // pe) ** 3
    g_mean, g_lig, g_poc = {}, {}, {}
    n_deg = 0
    for bpids, x, errs in tqdm(loader, desc="extract"):
        if x is None: continue
        x = x.to(DEVICE)
        B = x.shape[0]
        with torch.no_grad():
            tok = P.encode_tokens(enc, x)                       # (B, nG*npatch, D)
        nG = tok.shape[1] // npatch
        tok = tok.reshape(B, nG, npatch, tok.shape[2]).mean(1)  # (B, npatch, D) per-patch
        # per-patch occupancy in the SAME (i,j,k row-major) order as conv patches
        occ = x.reshape(B, n_in, G // pe, pe, G // pe, pe, G // pe, pe).sum(dim=(3, 5, 7))  # (B,C,g,g,g)
        occ = occ.reshape(B, n_in, npatch)
        lig_w = occ[:, :n_lig].sum(1)                            # (B, npatch)
        poc_w = occ[:, n_lig:n_lig + n_poc].sum(1)
        def wpool(w):
            w = w.clamp(min=0); s = w.sum(1, keepdim=True)
            fallback = s.squeeze(1) < 1e-6
            wn = torch.where(s > 1e-6, w / s.clamp(min=1e-6), torch.full_like(w, 1.0 / npatch))
            return (wn.unsqueeze(-1) * tok).sum(1), fallback     # (B,D)
        gm = tok.mean(1)
        gl, fbl = wpool(lig_w); gp, fbp = wpool(poc_w)
        n_deg += int(fbl.sum() + fbp.sum())
        for i, pid in enumerate(bpids):
            g_mean[pid] = gm[i].cpu().clone(); g_lig[pid] = gl[i].cpu().clone(); g_poc[pid] = gp[i].cpu().clone()
    print(f"extracted {len(g_mean)} complexes ({n_deg} degenerate lig/poc pools → mean fallback)")
    torch.save({"g_mean": g_mean, "g_lig": g_lig, "g_poc": g_poc, "scheme": scheme, "smap": smap}, CACHE)
    return torch.load(CACHE, weights_only=False)


def assemble(bundle, head):
    gm, gl, gp = bundle["g_mean"], bundle["g_lig"], bundle["g_poc"]
    def feat(pid):
        m, l, p = gm[pid].numpy(), gl[pid].numpy(), gp[pid].numpy()
        if head == "mean":       return m
        if head == "A_concat":   return np.concatenate([l, p])
        if head == "A_bilinear": return np.concatenate([l, p, l * p, np.abs(l - p)])
        raise ValueError(head)
    return feat


def run_head(bundle, head, lp_df, seeds=(0, 1, 2)):
    smap = bundle["smap"]; feat = assemble(bundle, head)
    pk = {p: v for p, v in zip(lp_df["pdb_id"], lp_df["pK"]) if v == v}
    data = {}
    for split in ("train", "val", "test"):
        pids = [p for p in smap if smap[p] == split and p in bundle["g_mean"] and p in pk]
        X = np.stack([feat(p) for p in pids]).astype(np.float32)
        y = np.array([pk[p] for p in pids], dtype=np.float32)
        data[split] = {"X": X, "y": y}
    res = []
    for s in seeds:
        m = P.train_one(data, seed=s, device=DEVICE, max_epochs=300, patience=30,
                        batch_size=128, lr=1e-3, weight_decay=1e-4, hidden=128, dropout=0.1)
        res.append(m)
    def agg(k): return float(np.mean([r[k] for r in res])), float(np.std([r[k] for r in res]))
    return dict(dim=data["train"]["X"].shape[1],
                val_rho=agg("best_val_spearman"), test_rho=agg("test_spearman"),
                test_r=agg("test_pearson"), test_rmse=agg("test_rmse"))


import torch.nn as nn


class EpinetHead(nn.Module):
    """TerraBind-style epinet: point estimate + epistemic residual over an index z~N(0,I).
    ŷ(g,z) = base(g) + <ε_learn(sg h, z), z> + prior_scale·<ε_prior(sg h, z), z>.
    prior net is frozen random init (epistemic prior); learnable net cancels its variance
    where data supports it → residual z-variance = calibrated epistemic uncertainty."""
    def __init__(self, in_dim, hidden=128, dz=8, prior_scale=1.0, dropout=0.1):
        super().__init__()
        self.dz = dz; self.prior_scale = prior_scale
        self.base = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(), nn.Dropout(dropout))
        self.base_out = nn.Linear(hidden, 1)
        def epi(): return nn.Sequential(nn.Linear(hidden + dz, 64), nn.SiLU(), nn.Linear(64, dz))
        self.learn = epi()
        self.prior = epi()
        for p in self.prior.parameters(): p.requires_grad_(False)  # frozen prior

    def forward(self, g, z):                              # g:(B,in) z:(B,dz)
        h = self.base(g)
        y0 = self.base_out(h).squeeze(-1)
        hd = h.detach()                                  # epinet doesn't move base features
        inp = torch.cat([hd, z], -1)
        el = (self.learn(inp) * z).sum(-1) / (self.dz ** 0.5)
        ep = (self.prior(inp) * z).sum(-1) / (self.dz ** 0.5)
        return y0 + el + self.prior_scale * ep


def train_epinet(data, seed, device, max_epochs=300, patience=30, K=200, dz=8, prior_scale=1.0):
    torch.manual_seed(seed); np.random.seed(seed)
    def T(s): return (torch.from_numpy(data[s]["X"]).to(device), torch.from_numpy(data[s]["y"]).to(device))
    Xtr, ytr = T("train"); Xva, yva = T("val"); Xte, yte = T("test")
    m = EpinetHead(Xtr.shape[1], dz=dz, prior_scale=prior_scale).to(device)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4)
    mse = nn.MSELoss()
    def ens(X):                                          # K-sample ensemble → (mean, std) preds
        with torch.no_grad():
            m.eval(); ps = []
            for _ in range(K):
                z = torch.randn(X.shape[0], dz, device=device)
                ps.append(m(X, z))
            P = torch.stack(ps); m.train()
            return P.mean(0), P.std(0)
    best_val, best = -1e9, None; since = 0
    n = Xtr.shape[0]
    for ep in range(max_epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, 128):
            idx = perm[i:i+128]; xb, yb = Xtr[idx], ytr[idx]
            z = torch.randn(xb.shape[0], dz, device=device)
            loss = mse(m(xb, z), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        vm, _ = ens(Xva)
        vr = spearmanr(vm.cpu().numpy(), yva.cpu().numpy()).correlation
        if vr > best_val: best_val, best, since = vr, {k: v.detach().clone() for k, v in m.state_dict().items()}, 0
        else: since += 1
        if since >= patience: break
    m.load_state_dict(best)
    mu, sd = ens(Xte)
    mu, sd, yt = mu.cpu().numpy(), sd.cpu().numpy(), yte.cpu().numpy()
    err = np.abs(mu - yt)
    # point-estimate metrics + uncertainty quality
    rmse = float(np.sqrt(((mu - yt) ** 2).mean()))
    unc_err_rho = float(spearmanr(sd, err).correlation)               # σ tracks error? (>0 good)
    order = np.argsort(sd)                                             # ascending σ = most confident first
    def sel_rmse(frac):
        k = max(2, int(len(order) * frac)); s = order[:k]
        return float(np.sqrt(((mu[s] - yt[s]) ** 2).mean()))
    return dict(val_rho=best_val, test_rho=float(spearmanr(mu, yt).correlation),
                test_r=float(pearsonr(mu, yt)[0]), rmse=rmse, unc_err_rho=unc_err_rho,
                rmse_50=sel_rmse(0.5), rmse_25=sel_rmse(0.25), rmse_all=sel_rmse(1.0),
                sd_mean=float(sd.mean()))


class SmallMLP(nn.Module):
    def __init__(self, d, hidden=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
    def forward(self, x): return self.net(x).squeeze(-1)


def _fit_mlp(Xtr, ytr, Xva, yva, seed, device, dropout=0.1, boot=False, max_epochs=300, patience=30):
    torch.manual_seed(seed); g = torch.Generator(device="cpu").manual_seed(seed)
    m = SmallMLP(Xtr.shape[1], dropout=dropout).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4); mse = nn.MSELoss()
    n = Xtr.shape[0]
    if boot:
        bi = torch.randint(0, n, (n,), generator=g).to(device); Xtr, ytr = Xtr[bi], ytr[bi]
    best, bv, since = None, -1e9, 0
    for ep in range(max_epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, 128):
            idx = perm[i:i+128]; loss = mse(m(Xtr[idx]), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad(): vr = spearmanr(m(Xva).cpu().numpy(), yva.cpu().numpy()).correlation
        m.train()
        if vr > bv: bv, best, since = vr, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else: since += 1
        if since >= patience: break
    m.load_state_dict(best); m.eval(); return m, bv


def _unc_metrics(mu, sd, yt):
    err = np.abs(mu - yt); order = np.argsort(sd)
    def sel(frac):
        k = max(2, int(len(order) * frac)); s = order[:k]
        return float(np.sqrt(((mu[s] - yt[s]) ** 2).mean()))
    return dict(test_rho=float(spearmanr(mu, yt).correlation), test_r=float(pearsonr(mu, yt)[0]),
                rmse=float(np.sqrt((err ** 2).mean())), unc_err_rho=float(spearmanr(sd, err).correlation),
                rmse_all=sel(1.0), rmse_50=sel(0.5), rmse_25=sel(0.25), sd_mean=float(sd.mean()))


def run_uncertainty(bundle, lp_df, methods=("ensemble", "mcdropout", "epinet")):
    smap = bundle["smap"]; pk = {p: v for p, v in zip(lp_df["pdb_id"], lp_df["pK"]) if v == v}
    def T(split):
        pids = [p for p in smap if smap[p] == split and p in bundle["g_mean"] and p in pk]
        X = torch.from_numpy(np.stack([bundle["g_mean"][p].numpy() for p in pids]).astype(np.float32)).to(DEVICE)
        y = np.array([pk[p] for p in pids], dtype=np.float32)
        return X, torch.from_numpy(y).to(DEVICE), y
    Xtr, ytr, _ = T("train"); Xva, yva, _ = T("val"); Xte, _, yte = T("test")

    def report(name, mu, sd):
        m = _unc_metrics(mu, sd, yte)
        print(f"\n=== {name} ===")
        print(f"  point : ρ {m['test_rho']:.3f}  r {m['test_r']:.3f}  RMSE {m['rmse']:.3f}")
        print(f"  σ↔err : corr(σ,|err|) = {m['unc_err_rho']:+.3f}   mean σ {m['sd_mean']:.3f} pK")
        print(f"  triage: RMSE all {m['rmse_all']:.3f} → conf-50% {m['rmse_50']:.3f} → conf-25% {m['rmse_25']:.3f}")
        return m

    if "ensemble" in methods:   # deep ensemble (bootstrap + seed diversity) — gold-standard epistemic
        preds = []
        for s in range(10):
            m, _ = _fit_mlp(Xtr, ytr, Xva, yva, seed=100 + s, device=DEVICE, boot=True)
            with torch.no_grad(): preds.append(m(Xte).cpu().numpy())
        P = np.stack(preds); report("Deep ensemble (10 members)", P.mean(0), P.std(0))
    if "mcdropout" in methods:  # MC-dropout: one net, dropout ON at inference
        m, _ = _fit_mlp(Xtr, ytr, Xva, yva, seed=0, device=DEVICE, dropout=0.2)
        m.train()  # keep dropout active
        with torch.no_grad():
            ps = [m(Xte).cpu().numpy() for _ in range(200)]
        P = np.stack(ps); report("MC-dropout (p=0.2, 200 samples)", P.mean(0), P.std(0))
    if "epinet" in methods:     # retuned epinet: harder-to-cancel prior
        rs = [train_epinet({"train": {"X": Xtr.cpu().numpy(), "y": ytr.cpu().numpy()},
                            "val": {"X": Xva.cpu().numpy(), "y": yva.cpu().numpy()},
                            "test": {"X": Xte.cpu().numpy(), "y": yte}}, s, DEVICE, dz=16, prior_scale=5.0)
              for s in (0, 1, 2)]
        print(f"\n=== Epinet v2 (dz16, prior×5, 3 seeds) ===")
        A = lambda k: np.mean([r[k] for r in rs])
        print(f"  point : ρ {A('test_rho'):.3f}  r {A('test_r'):.3f}  RMSE {A('rmse'):.3f}")
        print(f"  σ↔err : corr(σ,|err|) = {A('unc_err_rho'):+.3f}   mean σ {A('sd_mean'):.3f} pK")
        print(f"  triage: RMSE all {A('rmse_all'):.3f} → conf-50% {A('rmse_50'):.3f} → conf-25% {A('rmse_25'):.3f}")


def run_epinet(bundle, lp_df, seeds=(0, 1, 2)):
    smap = bundle["smap"]
    pk = {p: v for p, v in zip(lp_df["pdb_id"], lp_df["pK"]) if v == v}
    data = {}
    for split in ("train", "val", "test"):
        pids = [p for p in smap if smap[p] == split and p in bundle["g_mean"] and p in pk]
        data[split] = {"X": np.stack([bundle["g_mean"][p].numpy() for p in pids]).astype(np.float32),
                       "y": np.array([pk[p] for p in pids], dtype=np.float32)}
    rs = [train_epinet(data, s, DEVICE) for s in seeds]
    def agg(k): return float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs]))
    print("\n=== D · Epinet uncertainty head (mean±std over 3 seeds) ===")
    print(f"  point estimate : ρ {agg('test_rho')[0]:.3f}  r {agg('test_r')[0]:.3f}  RMSE {agg('rmse')[0]:.3f}   (val ρ {agg('val_rho')[0]:.3f})")
    print(f"  uncertainty    : corr(σ, |err|) = {agg('unc_err_rho')[0]:+.3f}  (>0 ⇒ σ tracks error)")
    print(f"  selective RMSE : all {agg('rmse_all')[0]:.3f} → conf-50% {agg('rmse_50')[0]:.3f} → conf-25% {agg('rmse_25')[0]:.3f}   (drop ⇒ σ is useful for triage)")
    print(f"  mean σ         : {agg('sd_mean')[0]:.3f} pK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--heads", default="mean,A_concat,A_bilinear")
    ap.add_argument("--epinet", action="store_true", help="run the epinet uncertainty head (D)")
    ap.add_argument("--uncertainty", action="store_true", help="compare ensemble / MC-dropout / epinet-v2")
    args = ap.parse_args()
    bundle = extract() if (args.extract or not CACHE.exists()) else torch.load(CACHE, weights_only=False)
    lp_df = P.load_lp_index(P.LP_CSV)
    if args.epinet:
        run_epinet(bundle, lp_df)
        return
    if args.uncertainty:
        run_uncertainty(bundle, lp_df)
        return
    print(f"\n{'head':<12}{'dim':>6}{'val ρ':>9}{'test ρ':>10}{'test r':>10}{'RMSE':>10}")
    for head in args.heads.split(","):
        r = run_head(bundle, head, lp_df)
        print(f"{head:<12}{r['dim']:>6}{r['val_rho'][0]:>9.3f}{r['test_rho'][0]:>10.3f}"
              f"{r['test_r'][0]:>10.3f}{r['test_rmse'][0]:>10.3f}   (±{r['test_rho'][1]:.3f} ρ)")


if __name__ == "__main__":
    main()
