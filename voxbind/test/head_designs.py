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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--heads", default="mean,A_concat,A_bilinear")
    args = ap.parse_args()
    bundle = extract() if (args.extract or not CACHE.exists()) else torch.load(CACHE, weights_only=False)
    lp_df = P.load_lp_index(P.LP_CSV)
    print(f"\n{'head':<12}{'dim':>6}{'val ρ':>9}{'test ρ':>10}{'test r':>10}{'RMSE':>10}")
    for head in args.heads.split(","):
        r = run_head(bundle, head, lp_df)
        print(f"{head:<12}{r['dim']:>6}{r['val_rho'][0]:>9.3f}{r['test_rho'][0]:>10.3f}"
              f"{r['test_r'][0]:>10.3f}{r['test_rmse'][0]:>10.3f}   (±{r['test_rho'][1]:.3f} ρ)")


if __name__ == "__main__":
    main()
