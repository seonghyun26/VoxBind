"""recon_granularity.py — finer-grained C↔CDG reconstruction (260827 follow-up).

The main report reconstructs at the COMPLEX level (one 640-d mean-pooled vector per pocket:
CDG→C R²=0.94, C→CDG=0.60). This script adds two finer granularities:

  A. PATCH-TOKEN (local representation): the 512 spatial patch tokens (8^3 grid, group-pooled)
     of the C and CDG encoders, reconstructed per patch → does density add LOCAL structure?
  B. VOXEL (input level): at each voxel, predict CDG's extra input channels (electron density
     + |grad rho|) from C's coordinate-blob channels → is density inferable from coordinates?

Both are "between C and CDG", finer than the per-complex analysis. Run on GPU:
    CUDA_VISIBLE_DEVICES=2,3 python notebook/html/260827/recon_granularity.py
"""
import sys
import json
import base64
import importlib.util
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

REPO = Path("/home/shpark/prj-denovo/VoxBind")
VOX = REPO / "voxbind"
HERE = Path(__file__).resolve().parent
OUT_JSON = HERE / "repr_analysis" / "granularity.json"
dev = "cuda" if torch.cuda.is_available() else "cpu"
GP = 8

_spec = importlib.util.spec_from_file_location("p01c", str(VOX / "dataset" / "01c_pdbbind_probe.py"))
pr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pr)

# checkpoints live in model_zoo (the exps/ coords checkpoint was cleaned up); cfgs match the
# 260705 champion (CDG, n_in13 groups[7,4,2]) and 260723 coords twin (n_in11 groups[7,4]) exactly.
CDG = dict(exp=VOX / "model_zoo" / "champion", cond="atomblob_density_gradmag")
C = dict(exp=VOX / "model_zoo" / "coords", cond="atomblob")
EP = 49


def _dirs(cfg, cond):
    spec = pr.infer_feature_spec(cond, cfg, "auto")
    vox = pr.voxel_dir_for("v5")
    return spec, pr.atom_dir_for(vox, spec.atom_source), vox / "density"


def extract_spatial_tokens(enc_cfg, pids, bs=16):
    """(N, 512, D) group-pooled spatial patch tokens for one encoder."""
    cfg = OmegaConf.load(enc_cfg["exp"] / "cfg.yaml")
    spec, atom_dir, dens_dir = _dirs(cfg, enc_cfg["cond"])
    enc = pr.load_encoder(enc_cfg["exp"], EP, dev, cfg=cfg)
    nG = len(enc.channel_groups) if getattr(enc, "channel_groups", None) else 1
    D, n_in = enc.dim, spec.expected_channels
    out = np.zeros((len(pids), GP ** 3, D), np.float32)
    for i in range(0, len(pids), bs):
        chunk = pids[i:i + bs]
        xs = [pr.load_voxels_for(p, enc_cfg["cond"], n_in, atom_dir, dens_dir,
                                 spec.input_mode, spec.with_gradmag) for p in chunk]
        x = torch.stack(xs).to(dev)
        tk = pr.encode_tokens(enc, x)                              # (B, nG*512, D)
        tk = tk.reshape(len(chunk), nG, GP ** 3, D).mean(1)        # (B, 512, D)
        out[i:i + len(chunk)] = tk.cpu().numpy()
        if i % (bs * 20) == 0:
            print(f"    tokens {i}/{len(pids)}", flush=True)
    return out


def _train_mlp(Xtr, Ytr, Xte, hidden=512, epochs=40, bs=8192, lr=1e-3, wd=1e-5, seed=0):
    torch.manual_seed(seed)
    din, dout = Xtr.shape[1], Ytr.shape[1]
    net = nn.Sequential(nn.Linear(din, hidden), nn.GELU(), nn.Linear(hidden, dout)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=dev)
    Ytr_t = torch.tensor(Ytr, dtype=torch.float32, device=dev)
    n = len(Xtr)
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        for j in range(0, n, bs):
            b = perm[j:j + bs]
            opt.zero_grad()
            F.mse_loss(net(Xtr_t[b]), Ytr_t[b]).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        pred = net(torch.tensor(Xte, dtype=torch.float32, device=dev)).cpu().numpy()
    return pred


def r2_vw(y, p):
    ss_res = ((y - p) ** 2).sum(0)
    ss_tot = ((y - y.mean(0)) ** 2).sum(0)
    return float(1 - ss_res.sum() / ss_tot.sum())


def recon(Xs, Xt, groups, seed=0):
    """held-out reconstruction R^2, split by GROUP id (complex) to avoid leakage. std both."""
    rng = np.random.RandomState(seed)
    gids = np.unique(groups)
    te_g = set(rng.choice(gids, max(1, int(0.2 * len(gids))), replace=False))
    te = np.array([g in te_g for g in groups])
    tr = ~te
    # standardize on train
    mx, sx = Xs[tr].mean(0), Xs[tr].std(0) + 1e-6
    my, sy = Xt[tr].mean(0), Xt[tr].std(0) + 1e-6
    Xs_n, Xt_n = (Xs - mx) / sx, (Xt - my) / sy
    # subsample train rows for speed
    tri = np.where(tr)[0]
    if len(tri) > 200_000:
        tri = rng.choice(tri, 200_000, replace=False)
    pred = _train_mlp(Xs_n[tri], Xt_n[tri], Xs_n[te], seed=seed)
    return r2_vw(Xt_n[te], pred)


def main():
    torch.set_num_threads(4)
    sm, _ = pr.load_frozen_split_map("lp_edrscc_v2")
    test = set(p.lower() for p, s in sm.items() if s == "test")
    import pandas as pd
    av = pd.read_csv(pr.voxel_dir_for("v5") / "availability.csv")
    ok = set(av[av["has_atoms"] & av["has_density"]]["pdb_id"])
    pids = sorted(test & ok)
    print(f"complexes: {len(pids)}", flush=True)

    res = {}

    # ---------- A. patch-token (local representation) ----------
    print("[A] extracting spatial patch tokens (CDG, C) ...", flush=True)
    tok_cdg = extract_spatial_tokens(CDG, pids)         # (N,512,D)
    tok_c = extract_spatial_tokens(C, pids)             # (N,512,D)
    N, T, D = tok_cdg.shape
    gid = np.repeat(np.arange(N), T)                    # complex id per token row
    Xc = tok_c.reshape(N * T, D)
    Xcdg = tok_cdg.reshape(N * T, D)
    print("[A] patch-token reconstruction ...", flush=True)
    a_cdg2c = recon(Xcdg, Xc, gid)
    a_c2cdg = recon(Xc, Xcdg, gid)
    # complex-level control on the SAME features (mean over the 512 tokens)
    pc, pcdg = tok_c.mean(1), tok_cdg.mean(1)
    gpool = np.arange(N)
    cx_cdg2c = recon(pcdg, pc, gpool)
    cx_c2cdg = recon(pc, pcdg, gpool)
    res["patch_token"] = {"CDG->C": a_cdg2c, "C->CDG": a_c2cdg, "asym": a_cdg2c - a_c2cdg,
                          "n_tokens": int(N * T)}
    res["complex_control"] = {"CDG->C": cx_cdg2c, "C->CDG": cx_c2cdg, "asym": cx_cdg2c - cx_c2cdg}
    print(f"  patch-token : CDG->C {a_cdg2c:.3f} | C->CDG {a_c2cdg:.3f} | asym {a_cdg2c-a_c2cdg:+.3f}")
    print(f"  complex ctrl: CDG->C {cx_cdg2c:.3f} | C->CDG {cx_c2cdg:.3f} | asym {cx_cdg2c-cx_c2cdg:+.3f}")

    # ---------- B. voxel (input level) ----------
    print("[B] per-voxel coords(11ch) -> density+gradmag(2ch), occupied voxels ...", flush=True)
    res["voxel_input"] = voxel_recon(pids[:400])
    print(f"  voxel coords->density+gradmag  R2 {res['voxel_input']['coords->dg']:.3f}"
          f"  ({res['voxel_input']['n_voxels']} occupied voxels, {res['voxel_input']['n_complexes']} complexes)")

    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(res, indent=2))
    print(f"\nsaved -> {OUT_JSON}")
    print(json.dumps(res, indent=2))


def voxel_recon(pids, bs=8):
    """coords(11)->density+gradmag(2) per occupied voxel, split by complex."""
    cfg = OmegaConf.load(CDG["exp"] / "cfg.yaml")
    spec, atom_dir, dens_dir = _dirs(cfg, CDG["cond"])
    rng = np.random.RandomState(0)
    coords_rows, dg_rows, gids = [], [], []
    for k, p in enumerate(pids):
        x = pr.load_voxels_for(p, CDG["cond"], spec.expected_channels, atom_dir, dens_dir,
                               spec.input_mode, spec.with_gradmag)
        occ = x[:11].sum(0).reshape(-1).numpy() > 0.1
        idx = np.where(occ)[0]
        if len(idx) == 0:
            continue
        if len(idx) > 600:
            idx = rng.choice(idx, 600, replace=False)
        coords_rows.append(x[:11].reshape(11, -1).T.numpy()[idx])
        dg_rows.append(x[11:13].reshape(2, -1).T.numpy()[idx])
        gids.append(np.full(len(idx), k))
        if k % 100 == 0:
            print(f"    voxels {k}/{len(pids)}", flush=True)
    Xc = np.concatenate(coords_rows); Ydg = np.concatenate(dg_rows); g = np.concatenate(gids)
    r = recon(Xc, Ydg, g)
    return {"coords->dg": r, "n_voxels": int(len(Xc)), "n_complexes": len(pids)}


if __name__ == "__main__":
    main()
