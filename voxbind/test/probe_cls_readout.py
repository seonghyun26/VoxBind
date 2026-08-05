"""probe_cls_readout.py — does a CLS-token + positional-encoding + transformer readout on the
frozen champion's PER-PATCH tokens beat the mean-pool MLP head? (260806 idea check)

Everything frozen except the readout. Extract per-patch tokens once (champion 100M v2 mask075,
e49) for lp_edrscc_v2, then train two heads on the SAME tokens:
  mean : tokens.mean(patch) -> zscore -> MLP2(128)        (reproduces champion ρ≈0.644)
  cls  : proj -> +learned PE -> [CLS]+tokens -> Transformer(2L,4H) -> CLS -> Linear

3 seeds, early-stop on val Spearman, MSE loss. Reports test r/ρ/RMSE + val ρ.

Usage:  cd voxbind && CUDA_VISIBLE_DEVICES=4 python test/probe_cls_readout.py [--extract]
"""
import gemmi  # noqa: F401 — before torch
import argparse, importlib.util, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr

VOX = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VOX))
_spec = importlib.util.spec_from_file_location("probe01c", VOX / "dataset" / "01c_pdbbind_probe.py")
P = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(P)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EXP_DIR = VOX / "exps" / "260705_ar_cvit_100m_v2_mask075"
EPOCH, VOXV, COND = 49, "v5", "atomblob_density_gradmag"
CACHE = VOX / "test" / "cls_tokens_champion.pt"
SEEDS = (0, 1, 2)


def extract():
    cfg = OmegaConf.load(EXP_DIR / "cfg.yaml")
    spec = P.infer_feature_spec(COND, cfg, "auto")
    vox_dir = P.voxel_dir_for(VOXV); atom_dir = P.atom_dir_for(vox_dir, spec.atom_source)
    dens_dir = vox_dir / "density"
    enc = P.load_encoder(EXP_DIR, EPOCH, DEVICE, cfg=cfg)
    n_in = enc.n_in_channels
    smap, scheme = P.load_frozen_split_map("lp_edrscc_v2")
    pids = sorted(smap.keys())
    ds = P._ExtractDataset(pids, COND, n_in, atom_dir, dens_dir, spec.input_mode, spec.with_gradmag, None)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=4, collate_fn=P._extract_collate)
    G = int(cfg.vox.grid_dim); pe = 8; npatch = (G // pe) ** 3
    toks = {}
    for bpids, x, errs in tqdm(loader, desc="extract per-patch tokens"):
        if x is None:
            continue
        x = x.to(DEVICE)
        with torch.no_grad():
            tok = P.encode_tokens(enc, x)                       # (B, nG*npatch, D)
        B, D = tok.shape[0], tok.shape[2]
        nG = tok.shape[1] // npatch
        tok = tok.reshape(B, nG, npatch, D).mean(1)             # (B, npatch, D) per-patch
        for i, pid in enumerate(bpids):
            toks[pid] = tok[i].half().cpu().clone()
    print(f"extracted {len(toks)} complexes; per-patch shape {next(iter(toks.values())).shape}")
    torch.save({"toks": toks, "smap": smap}, CACHE)
    return {"toks": toks, "smap": smap}


# ── heads ────────────────────────────────────────────────────────────────────
class MeanMLP(nn.Module):
    def __init__(self, d, mu, sd, hidden=128, dropout=0.1):
        super().__init__()
        self.register_buffer("mu", mu); self.register_buffer("sd", sd)
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):                                       # x:(B,npatch,D)
        g = x.mean(1)
        g = (g - self.mu) / self.sd
        return self.net(g).squeeze(-1)


class CLSReadout(nn.Module):
    def __init__(self, d, npatch, d_model=256, heads=4, layers=2, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(d, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, npatch + 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02); nn.init.trunc_normal_(self.pos, std=0.02)
        enc = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=2 * d_model,
                                         dropout=dropout, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(enc, layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):                                       # x:(B,npatch,D)
        h = self.proj(x)
        cls = self.cls.expand(h.shape[0], -1, -1)
        h = torch.cat([cls, h], 1) + self.pos
        h = self.tf(h)
        return self.head(self.norm(h[:, 0])).squeeze(-1)


def arrays(bundle, split, pk):
    sm = bundle["smap"]; toks = bundle["toks"]
    pids = [p for p in sm if sm[p] == split and p in toks and p in pk]
    X = torch.stack([toks[p].float() for p in pids])           # (N, npatch, D)
    y = torch.tensor([pk[p] for p in pids], dtype=torch.float32)
    return X, y


def train_head(kind, data, seed, max_epochs=300, patience=30, bs=64):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr = data["train"]; Xva, yva = data["val"]; Xte, yte = data["test"]
    npatch, D = Xtr.shape[1], Xtr.shape[2]
    if kind == "mean":
        g = Xtr.mean(1); mu = g.mean(0); sd = g.std(0) + 1e-6
        model = MeanMLP(D, mu, sd).to(DEVICE)
    else:
        model = CLSReadout(D, npatch).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.MSELoss()
    Xtr_d, ytr_d = Xtr.to(DEVICE), ytr.to(DEVICE)
    Xva_d, Xte_d = Xva.to(DEVICE), Xte.to(DEVICE)
    yva_np, yte_np = yva.numpy(), yte.numpy()
    n = Xtr.shape[0]; best_val, best_state, bad = -1e9, None, 0
    for ep in range(max_epochs):
        model.train(); perm = torch.randperm(n, device=DEVICE)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            opt.zero_grad(); lossf(model(Xtr_d[idx]), ytr_d[idx]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.cat([model(Xva_d[i:i + 256]) for i in range(0, Xva_d.shape[0], 256)]).cpu().numpy()
        vs = spearmanr(pv, yva_np).statistic
        if vs > best_val:
            best_val, bad = vs, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        pt = torch.cat([model(Xte_d[i:i + 256]) for i in range(0, Xte_d.shape[0], 256)]).cpu().numpy()
    return dict(val=float(best_val), r=float(pearsonr(pt, yte_np)[0]),
                rho=float(spearmanr(pt, yte_np).statistic),
                rmse=float(np.sqrt(((pt - yte_np) ** 2).mean())),
                nparam=sum(p.numel() for p in model.parameters()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--extract", action="store_true"); a = ap.parse_args()
    bundle = extract() if (a.extract or not CACHE.exists()) else torch.load(CACHE, weights_only=False)
    lp = P.load_lp_index(P.LP_CSV); pk = {p: v for p, v in zip(lp["pdb_id"], lp["pK"]) if v == v}
    data = {s: arrays(bundle, s, pk) for s in ("train", "val", "test")}
    print(f"tokens: train {data['train'][0].shape} val {data['val'][0].shape[0]} test {data['test'][0].shape[0]}\n")
    print(f"{'head':<6}{'params':>10}{'val ρ':>9}{'test r':>10}{'test ρ':>10}{'RMSE':>10}   (3 seeds)")
    for kind in ("mean", "cls"):
        rs = [train_head(kind, data, s) for s in SEEDS]
        agg = {k: (np.mean([r[k] for r in rs]), np.std([r[k] for r in rs])) for k in ("val", "r", "rho", "rmse")}
        print(f"{kind:<6}{rs[0]['nparam']:>10,}"
              f"{agg['val'][0]:>7.3f}±{agg['val'][1]:.2f}"
              f"{agg['r'][0]:>7.3f}±{agg['r'][1]:.2f}"
              f"{agg['rho'][0]:>7.3f}±{agg['rho'][1]:.2f}"
              f"{agg['rmse'][0]:>7.3f}±{agg['rmse'][1]:.2f}")


if __name__ == "__main__":
    main()
