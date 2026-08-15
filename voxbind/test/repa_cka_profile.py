"""repa_cka_profile.py — B2 proper: CKA(CDG tokens, U-Net feature) at every U-Net depth.

The per-depth affinity probe answered "which depth carries the most binding information".
CKA answers the different question U-REPA actually needs: "which depth is most GEOMETRICALLY
similar to the teacher" — i.e. where an alignment loss has the least distance to close.
The two can disagree, so both are needed before fixing the alignment point.

Linear CKA on centered features:
    CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
It only needs second moments, so we stream and accumulate
    S_xx = sum x x^T,  S_yy = sum y y^T,  S_yx = sum y x^T,  s_x, s_y, n
and center at the end (S_xx - s_x s_x^T / n, etc). No feature ever hits disk.

Two row definitions, as the checklist asks:
  token  : rows = (pocket, patch) pairs. Every stage is avg-pooled to the 8^3 patch grid so
           it aligns 1:1 with the ViT's 8^3 tokens. This is the level the manifold loss runs on.
  pooled : rows = pockets (spatial mean). The global-summary view.

Conditions (student x teacher = 4 pairings):
  student  sig0.9 = ligand + N(0, 0.9) noise (the training condition the checklist specifies)
           noise  = ligand replaced by pure noise (the walk-jump chain init = generation)
  teacher  holo      = full 13ch
           atoms0_dm = ligand atoms zeroed + rho/|grad rho| blanked in the ligand footprint
                       (the apo-like condition that matches generation)

Run:
    python test/repa_cka_profile.py --n 64      # smoke
    python test/repa_cka_profile.py
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[2]
VOX = REPO / "voxbind"
sys.path.insert(0, str(REPO))

_spec = importlib.util.spec_from_file_location(
    "p01c", str(VOX / "dataset" / "01c_pdbbind_probe.py"))
pr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pr)

from voxbind.models import create_model  # noqa: E402

DEFAULT_UNET = VOX / "exps" / "exp_sig0.9+prefetch_factor16_wjs.n_targets0" / "checkpoint.pth.tar"
DEFAULT_CDG = VOX / "model_zoo" / "efficient_60m_v3_mask085"
COND = "atomblob_density_gradmag"
DEPTHS = ("L0", "L1", "L2", "L3", "BN")
STUDENTS = ("sig0.9", "noise")
TEACHERS = ("holo", "atoms0_dm")
GP = 8                                  # ViT patch grid edge (64 / patch 8)
MASK_THRESH, MASK_DILATE = 0.2, 2


class _Vox(Dataset):
    """13ch [7 lig | 4 poc | rho | |grad rho|] — the teacher layout; the student reads [:7] / [7:11]."""

    def __init__(self, pids, atom_dir, dens_dir):
        self.pids, self.atom_dir, self.dens_dir = list(pids), atom_dir, dens_dir

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, i):
        try:
            x = pr.load_voxels_for(self.pids[i], COND, 13, self.atom_dir, self.dens_dir,
                                   input_mode="atomblob_density", with_gradmag=True)
            return x, True
        except Exception:
            return torch.zeros(13, 64, 64, 64), False


def _collate(b):
    xs = [x for x, ok in b if ok]
    return (torch.stack(xs, 0) if xs else None)


class _CKA:
    """Streaming linear-CKA accumulator for one (X, Y) pair."""

    def __init__(self, p, q, device):
        self.sxx = torch.zeros(p, p, dtype=torch.float64, device=device)
        self.syy = torch.zeros(q, q, dtype=torch.float64, device=device)
        self.syx = torch.zeros(q, p, dtype=torch.float64, device=device)
        self.sx = torch.zeros(p, dtype=torch.float64, device=device)
        self.sy = torch.zeros(q, dtype=torch.float64, device=device)
        self.n = 0

    def add(self, X, Y):
        X = X.double(); Y = Y.double()
        self.sxx += X.T @ X
        self.syy += Y.T @ Y
        self.syx += Y.T @ X
        self.sx += X.sum(0)
        self.sy += Y.sum(0)
        self.n += X.shape[0]

    def value(self):
        n = self.n
        cxx = self.sxx - torch.outer(self.sx, self.sx) / n
        cyy = self.syy - torch.outer(self.sy, self.sy) / n
        cyx = self.syx - torch.outer(self.sy, self.sx) / n
        num = (cyx ** 2).sum()
        den = torch.linalg.matrix_norm(cxx) * torch.linalg.matrix_norm(cyy)
        return float(num / den) if den > 0 else float("nan")


def _ligand_mask(x):
    occ = x[:, :7].sum(1, keepdim=True)
    m = (occ > MASK_THRESH).float()
    k = 2 * MASK_DILATE + 1
    return F.max_pool3d(m, kernel_size=k, stride=1, padding=MASK_DILATE) > 0.5


@torch.no_grad()
def _unet_stages(model, ligand, pocket):
    """{depth: (B, C, 8, 8, 8)} — every stage avg-pooled onto the ViT patch grid."""
    unet = model.unet3d
    x = model.ligand_encoder(ligand) + model.pocket_encoder(pocket)
    x = unet.grid_projection(x)
    out, lvl, n = {}, 0, len(unet.down)
    for i, m in enumerate(unet.down):
        x = m(x)
        nxt = unet.down[i + 1] if i + 1 < n else None
        if (nxt is None or type(nxt).__name__ == "Downsample") \
                and type(m).__name__ != "Downsample":
            out[DEPTHS[lvl]] = F.adaptive_avg_pool3d(x, GP)
            lvl += 1
    out["BN"] = F.adaptive_avg_pool3d(unet.middle(x), GP)
    return out


def _tok(v):
    """(B, C, 8,8,8) -> (B, 512, C), raster order matching the ViT patch grid."""
    B, C = v.shape[0], v.shape[1]
    return v.permute(0, 2, 3, 4, 1).reshape(B, GP ** 3, C)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512, help="pockets to stream")
    ap.add_argument("--unet_ckpt", default=str(DEFAULT_UNET))
    ap.add_argument("--teacher_dir", default=str(DEFAULT_CDG))
    ap.add_argument("--teacher_epoch", type=int, default=49)
    ap.add_argument("--tag", default=None, help="suffix for the results CSV")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device
    UNET_CKPT, CDG_EXP = Path(args.unet_ckpt), Path(args.teacher_dir)
    CDG_EPOCH = args.teacher_epoch

    ck = torch.load(UNET_CKPT, map_location="cpu", weights_only=False)
    unet = create_model(ck["cfg"], device=dev)
    unet.load_state_dict(ck["state_dict_ema"], strict=True)
    unet.eval()
    sigma = float(ck["cfg"].smooth_sigma)

    cfg = OmegaConf.load(CDG_EXP / "cfg.yaml")
    spec = pr.infer_feature_spec(COND, cfg, "auto")
    vox_dir = pr.voxel_dir_for("v5")
    atom_dir = pr.atom_dir_for(vox_dir, spec.atom_source)
    enc = pr.load_encoder(CDG_EXP, CDG_EPOCH, dev, cfg=cfg)
    n_groups = len(enc.channel_groups)
    print(f"  student : U-Net ep{ck['epoch']} sigma={sigma}")
    print(f"  teacher : {CDG_EXP.name} dim={enc.dim} groups={enc.channel_groups}")

    avail_csv = vox_dir / "availability.csv"
    if not avail_csv.exists():
        avail_csv = pr.PDBBIND_DIR / "voxels" / "availability.csv"
    av = pd.read_csv(avail_csv)
    ok = set(av[av["has_atoms"] & av["has_density"]]["pdb_id"])
    sm, _ = pr.load_frozen_split_map("lp_edrscc_v2")
    pids = sorted(p for p in sm if p in ok)[:args.n]
    print(f"  pockets : {len(pids)}  (rows: token {len(pids)*GP**3:,} / pooled {len(pids)})\n")

    loader = DataLoader(_Vox(pids, atom_dir, vox_dir / "density"), batch_size=args.batch,
                        shuffle=False, num_workers=args.workers, collate_fn=_collate,
                        pin_memory=str(dev).startswith("cuda"))

    dims = {"L0": 32, "L1": 64, "L2": 128, "L3": 512, "BN": 512}
    acc = {(lvl, s, t, d): _CKA(dims[d], enc.dim, dev)
           for lvl in ("token", "pooled") for s in STUDENTS for t in TEACHERS for d in DEPTHS}

    seen = 0
    with torch.no_grad():
        for bi, x in enumerate(loader):
            if x is None:
                continue
            x = x.to(dev, non_blocking=True)
            lig, poc = x[:, :7], x[:, 7:11]
            m = _ligand_mask(x)
            eps = torch.randn(lig.shape, generator=torch.Generator(device="cpu")
                              .manual_seed(1234 + bi)).to(dev)

            tea = {}
            for t in TEACHERS:
                xi = x.clone()
                if t != "holo":
                    xi[:, :7] = 0.0
                    xi[:, 11:13] = xi[:, 11:13] * (~m).float()
                tk = pr.encode_tokens(enc, xi)                       # (B, nG*512, D)
                B, _, D = tk.shape
                tea[t] = tk.reshape(B, n_groups, GP ** 3, D).mean(1)  # (B, 512, D) group-pooled

            for s in STUDENTS:
                l_in = (lig + sigma * eps) if s == "sig0.9" else (sigma * eps)
                st = _unet_stages(unet, l_in, poc)
                for d in DEPTHS:
                    Xt = _tok(st[d])                                  # (B, 512, C)
                    for t in TEACHERS:
                        Yt = tea[t]
                        acc[("token", s, t, d)].add(Xt.reshape(-1, Xt.shape[-1]),
                                                    Yt.reshape(-1, Yt.shape[-1]))
                        acc[("pooled", s, t, d)].add(Xt.mean(1), Yt.mean(1))
            seen += x.shape[0]
            if bi % 10 == 0:
                print(f"    {seen}/{len(pids)}", flush=True)

    rows = [dict(level=lvl, student=s, teacher=t, depth=d, dim=dims[d],
                 cka=acc[(lvl, s, t, d)].value())
            for (lvl, s, t, d) in acc]
    df = pd.DataFrame(rows)
    tag = args.tag or f"{UNET_CKPT.parent.name[:24]}_{CDG_EXP.name}"
    csv = VOX / "test" / "results" / f"repa_cka_profile_{tag}.csv"
    csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv, index=False)

    for lvl in ("token", "pooled"):
        print(f"\n  === {lvl}-level linear CKA (n_pockets={seen}) ===")
        piv = (df[df.level == lvl]
               .assign(pair=lambda z: z.student + " -> " + z.teacher)
               .pivot(index="depth", columns="pair", values="cka")
               .reindex(list(DEPTHS)))
        print(piv.round(3).to_string())
    print(f"\n  -> {csv}")


if __name__ == "__main__":
    main()
