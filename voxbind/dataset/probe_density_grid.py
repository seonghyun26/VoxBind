#!/usr/bin/env python
"""probe_density_grid.py — standalone PDBbind frozen-probe for the density×gradmag
grid. SELF-CONTAINED: imports only voxbind.models, NOT 01c_pdbbind_probe.py (which
gets reset by another agent). The MLP probe (MLP2 / train_one / build_dataset) is a
FAITHFUL copy of 01c's trusted recipe — Adam, mini-batch 64, RAW pK (no X/y
standardization), early-stop on val-ρ, restore best — so numbers match the 0.474/0.477
baselines and the whole 9-cell grid is one self-consistent table.

Each cell sets its density + (optional) gradmag SOURCE dir; density_dir=None → atoms-only.
  CUDA_VISIBLE_DEVICES=0 python dataset/probe_density_grid.py [--epoch 99] [--seeds 3] [--device cpu]
  → dataset/data/pdbbind/probe_results_e99_v5_gridcells.csv
"""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from voxbind.models.density_vit import DensityViT

HERE = Path(__file__).resolve().parent
PDBB = HERE / "data" / "pdbbind"
LP = PDBB / "raw" / "LP_PDBBind.csv"
ATOMS = PDBB / "voxels_ligvdw" / "atoms"        # 11ch (7 lig + 4 poc), element-vdW ligand
OUT_CSV = PDBB / "probe_results_e99_v5_gridcells.csv"
G = 64

_V5D = PDBB / "voxels_v5" / "density"
_V5G = PDBB / "voxels_v5" / "gradmag" / "density"            # real ‖∇ρ‖
_NSD = PDBB / "voxels_v5_noise" / "density"
_NSG = PDBB / "voxels_v5_noise" / "graddens" / "density"     # ‖∇(noise ρ)‖
# (condition, exp_dir, density_dir_or_None, gradmag_dir_or_None) — full 3×3 grid on one probe.
CELLS = [
    ("atomblob",                      "exps/260606_atomblob_vit_mae_40m_invfreq_pretrain", None, None),
    ("atomblob_density",              "exps/260606_atomblob_density_vit_mae_40m_invfreq_v5_pretrain", _V5D, None),
    ("atomblob_noisedensity",         "exps/atomblob_noisedensity_vit_mae_40m_v5", _NSD, None),
    ("atomblob_density_gradmag",      "exps/260606_atomblob_density_gradmag_vit_mae_40m_invfreq_v5_pretrain", _V5D, _V5G),
    ("atomblob_noisedensity_gradmag", "exps/atomblob_noisedensity_gradmag_vit_mae_40m_v5", _NSD, _NSG),
    ("atomblob_realgrad",             "exps/atomblob_realgrad_vit_mae_40m_v5",             _V5G, None),
    ("atomblob_noisegrad",            "exps/atomblob_noisegrad_vit_mae_40m_v5",            _NSG, None),
    ("atomblob_density_noisegrad",    "exps/atomblob_density_noisegrad_vit_mae_40m_v5",    _V5D, _NSG),
    ("atomblob_noisedensity_realgrad", "exps/atomblob_noisedensity_realgrad_vit_mae_40m_v5", _NSD, _V5G),
]


def load_encoder(exp_dir, epoch, device):
    cfg = OmegaConf.load(Path(exp_dir) / "cfg.yaml")
    m = cfg.model
    enc = DensityViT(
        grid_dim=cfg.vox.grid_dim, patch_size=m.patch_size, n_in_channels=m.n_in_channels,
        c_out=m.n_channels // 2, dim=m.dim, depth=m.depth, n_heads=m.heads,
        mlp_ratio=m.mlp_ratio, dropout=m.dropout, pos_encoding=m.get("pos_encoding", "learnable"),
    )
    ck = torch.load(Path(exp_dir) / f"checkpoint_e{epoch:04d}.pth.tar", map_location="cpu", weights_only=False)
    raw = ck["encoder_state_dict_ema"]
    sd = {k[len("encoder."):]: v for k, v in raw.items() if k.startswith("encoder.")}
    enc.load_state_dict(sd, strict=True)
    return enc.to(device).eval(), int(m.n_in_channels)


@torch.no_grad()
def extract_features(exp_dir, density_dir, gradmag_dir, epoch, pids, device, bs=32):
    enc, n_in = load_encoder(exp_dir, epoch, device)

    def build(pid):
        chans = [torch.from_numpy(np.load(ATOMS / f"{pid}.npy").astype("f4"))]   # (11,G,G,G)
        if density_dir is not None:
            chans.append(torch.from_numpy(np.load(Path(density_dir) / f"{pid}.npy").astype("f4")[None]))
        if gradmag_dir is not None:
            chans.append(torch.from_numpy(np.load(Path(gradmag_dir) / f"{pid}.npy").astype("f4")[None]))
        x = torch.cat(chans, 0)
        assert x.shape[0] == n_in, f"{pid}: built {x.shape[0]} ch, encoder wants {n_in}"
        return x

    feats = {}
    for s in range(0, len(pids), bs):
        xs, ok = [], []
        for pid in pids[s:s + bs]:
            try:
                xs.append(build(pid)); ok.append(pid)
            except Exception:
                pass
        if not xs:
            continue
        x = torch.stack(xs).to(device)
        z = enc.patch_embed(x).flatten(2).transpose(1, 2)
        if enc.pos_embed is not None:
            z = z + enc.pos_embed
        for blk in enc.blocks:
            z = blk(z, rope=getattr(enc, "rope", None))
        z = enc.norm(z).mean(dim=1).cpu()
        for pid, v in zip(ok, z):
            feats[pid] = v.contiguous().clone()
    return feats


# ── MLP probe — FAITHFUL copy of 01c train_one (Adam, mini-batch 64, raw pK, no
#    standardization, early-stop on val-ρ + restore best). Do not "improve" it. ──
class MLP2(nn.Module):
    def __init__(self, input_dim, hidden=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.SiLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_lp_index():
    df = pd.read_csv(LP).rename(columns={"Unnamed: 0": "pdb_id", "value": "pK"})
    df["pdb_id"] = df["pdb_id"].str.lower()
    return df


def build_dataset(features, lp_df, drop_covalent=True):
    df = lp_df.copy()
    if drop_covalent and "covalent" in df:
        df = df[~df["covalent"].astype(bool)]
    df = df[df["pdb_id"].isin(features.keys())]
    df = df[df["new_split"].isin(["train", "val", "test"])].dropna(subset=["pK"])
    out = {}
    for split in ("train", "val", "test"):
        sub = df[df["new_split"] == split]
        pids = sub["pdb_id"].tolist()
        X = np.stack([features[p].numpy() for p in pids]).astype(np.float32)
        y = sub["pK"].astype(np.float32).to_numpy()
        out[split] = {"X": X, "y": y, "pid": pids}
    return out


def train_one(data, seed, device, max_epochs=200, patience=30, batch_size=64,
              lr=1e-3, weight_decay=1e-4, hidden=128, dropout=0.1):
    torch.manual_seed(seed); np.random.seed(seed)
    def t(split):
        return (torch.from_numpy(data[split]["X"]).to(device),
                torch.from_numpy(data[split]["y"]).to(device))
    Xtr, ytr = t("train"); Xva, yva = t("val"); Xte, yte = t("test")
    model = MLP2(Xtr.shape[1], hidden=hidden, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse = nn.MSELoss()
    n = Xtr.shape[0]; best_val, best_state, bad = -np.inf, None, 0
    for ep in range(max_epochs):
        model.train(); perm = torch.randperm(n, device=device)
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            opt.zero_grad(); mse(model(Xtr[idx]), ytr[idx]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pva = model(Xva).cpu().numpy()
        vs = spearmanr(pva, yva.cpu().numpy()).correlation
        if vs > best_val:
            best_val, best_state, bad = vs, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        pte = model(Xte).cpu().numpy()
    yt = yte.cpu().numpy()
    return dict(best_val_spearman=best_val, test_spearman=spearmanr(pte, yt).correlation,
                test_pearson=pearsonr(pte, yt)[0], test_rmse=float(np.sqrt(((pte - yt) ** 2).mean())),
                test_mae=float(np.abs(pte - yt).mean()),
                n_train=n, n_val=Xva.shape[0], n_test=Xte.shape[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, default=99)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    lp = load_lp_index()
    # common pool = real-density-available ∩ LP, shared by ALL cells (identical test set).
    pool = sorted(set(lp["pdb_id"]) & {p.stem for p in _V5D.glob("*.npy")})
    print(f"common pool: {len(pool)} complexes | device={args.device}")
    rows = []
    for cond, exp_dir, ddir, gdir in CELLS:
        if not (Path(exp_dir) / f"checkpoint_e{args.epoch:04d}.pth.tar").exists():
            print(f"[skip] {cond}: no e{args.epoch} checkpoint"); continue
        print(f"[{cond}] density={'-' if ddir is None else Path(ddir).parent.name} "
              f"gradmag={'-' if gdir is None else Path(gdir).parent.name}", flush=True)
        feats = extract_features(exp_dir, ddir, gdir, args.epoch, pool, args.device)
        data = build_dataset(feats, lp, drop_covalent=True)
        res = [train_one(data, s, args.device) for s in range(args.seeds)]
        for s, r in enumerate(res):
            rows.append(dict(condition=cond, seed=s, **r))
        ts = np.array([r["test_spearman"] for r in res])
        print(f"    test_spearman {ts.mean():.4f} ± {ts.std():.4f}  (n_test={res[0]['n_test']})", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print("\n── Summary (test_spearman mean±std) ──")
    print(df.groupby("condition")["test_spearman"].agg(["mean", "std"]).round(4).to_string())
    print(f"[write] {OUT_CSV}")


if __name__ == "__main__":
    main()
