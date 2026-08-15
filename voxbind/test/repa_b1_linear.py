"""repa_b1_linear.py — B1: how far can a LINEAR map already get from student to teacher?

If a plain linear map already reaches the teacher (test R^2 ~ 1), U-REPA's projector MLP
absorbs the alignment loss in a few steps and the U-Net backbone never moves. If it reaches
nothing (R^2 ~ 0) the two coordinate systems are unrelated and only a relation-based
(manifold) loss can work. Mid-range is the regime the alignment is designed for.

Measured properly:
  * POCKET-level train/val/test split (60/20/20). Tokens from one pocket are strongly
    correlated, so a token-level split leaks and drives in-sample R^2 to 1.0 mechanically
    (effective n ~ #pockets, while p = 512 predictors).
  * Ridge, with lambda chosen on val and reported on test.
  * Both directions, plus CCA — R^2 asymmetry is confounded by differing predictor counts,
    CCA is dimension-symmetric and reports the shared-subspace size directly.

Everything runs from streamed second moments (no features stored), same accumulation trick
as repa_cka_profile.py.

Run:
    python test/repa_b1_linear.py --n 64      # smoke
    python test/repa_b1_linear.py
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
DIMS = {"L0": 32, "L1": 64, "L2": 128, "L3": 512, "BN": 512}
STUDENTS = ("sig0.9", "noise")
TEACHERS = ("holo", "atoms0_dm")
SPLITS = ("train", "val", "test")
GP = 8
MASK_THRESH, MASK_DILATE = 0.2, 2
LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


class _Vox(Dataset):
    def __init__(self, pids, atom_dir, dens_dir, split_of):
        self.pids, self.atom_dir, self.dens_dir = list(pids), atom_dir, dens_dir
        self.split_of = split_of

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, i):
        pid = self.pids[i]
        try:
            x = pr.load_voxels_for(pid, COND, 13, self.atom_dir, self.dens_dir,
                                   input_mode="atomblob_density", with_gradmag=True)
            return x, self.split_of[pid], True
        except Exception:
            return torch.zeros(13, 64, 64, 64), "train", False


def _collate(b):
    good = [(x, s) for x, s, ok in b if ok]
    if not good:
        return None, None
    xs, ss = zip(*good)
    return torch.stack(xs, 0), list(ss)


class _Mom:
    """Second moments of one feature stream: sum x x^T, sum x, n."""

    def __init__(self, p, dev):
        self.s2 = torch.zeros(p, p, dtype=torch.float64, device=dev)
        self.s1 = torch.zeros(p, dtype=torch.float64, device=dev)
        self.n = 0

    def add(self, X):
        X = X.double()
        self.s2 += X.T @ X
        self.s1 += X.sum(0)
        self.n += X.shape[0]


def _ligand_mask(x):
    occ = x[:, :7].sum(1, keepdim=True)
    k = 2 * MASK_DILATE + 1
    return F.max_pool3d((occ > MASK_THRESH).float(), k, 1, MASK_DILATE) > 0.5


@torch.no_grad()
def _unet_stages(model, ligand, pocket):
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
    B, C = v.shape[0], v.shape[1]
    return v.permute(0, 2, 3, 4, 1).reshape(B, GP ** 3, C)


def _center(s2, s1, n):
    return s2 - torch.outer(s1, s1) / n


def _cross_center(sxy, sx, sy, n):
    return sxy - torch.outer(sx, sy) / n


def ridge_r2(mx, my, sxy, dev):
    """Fit X->Y on train (ridge, lambda by val), return test R^2 and the chosen lambda."""
    p = mx["train"].s2.shape[0]
    ntr = mx["train"].n
    mux, muy = mx["train"].s1 / ntr, my["train"].s1 / ntr
    Cxx = _center(mx["train"].s2, mx["train"].s1, ntr)
    Cxy = _cross_center(sxy["train"], mx["train"].s1, my["train"].s1, ntr)
    scale = float(torch.diagonal(Cxx).mean())
    eye = torch.eye(p, dtype=torch.float64, device=dev)

    def rss_tss(split, W):
        n = mx[split].n
        sx, sy = mx[split].s1, my[split].s1
        # moments re-centered on the TRAIN means (the fitted intercept)
        Xc = mx[split].s2 - torch.outer(sx, mux) - torch.outer(mux, sx) + n * torch.outer(mux, mux)
        Xy = sxy[split] - torch.outer(sx, muy) - torch.outer(mux, sy) + n * torch.outer(mux, muy)
        Yc = my[split].s2 - torch.outer(sy, muy) - torch.outer(muy, sy) + n * torch.outer(muy, muy)
        rss = torch.trace(Yc) - 2 * torch.trace(W.T @ Xy) + torch.trace(W.T @ Xc @ W)
        muy_s = sy / n                                    # TSS about the split's own mean
        tss = torch.trace(my[split].s2 - n * torch.outer(muy_s, muy_s))
        return float(rss), float(tss)

    best, best_lam = -1e9, None
    for lam in LAMBDAS:
        W = torch.linalg.solve(Cxx + lam * scale * ntr * eye, Cxy)
        r, t = rss_tss("val", W)
        r2 = 1 - r / t
        if r2 > best:
            best, best_lam, bestW = r2, lam, W
    r, t = rss_tss("test", bestW)
    return 1 - r / t, best, best_lam


def cca_spectrum(mx, my, sxy, dev, eps=1e-6, k=10):
    """Canonical correlations from the train moments (dimension-symmetric shared subspace)."""
    n = mx["train"].n
    Cxx = _center(mx["train"].s2, mx["train"].s1, n)
    Cyy = _center(my["train"].s2, my["train"].s1, n)
    Cxy = _cross_center(sxy["train"], mx["train"].s1, my["train"].s1, n)

    def invsqrt(C):
        w, V = torch.linalg.eigh(C)
        w = torch.clamp(w, min=eps * float(w.max()))
        return V @ torch.diag(w.rsqrt()) @ V.T

    M = invsqrt(Cxx) @ Cxy @ invsqrt(Cyy)
    s = torch.linalg.svdvals(M).clamp(0, 1)
    return s[:k].cpu().numpy(), int((s > 0.9).sum()), int((s > 0.7).sum()), int((s > 0.5).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1024)
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
    nG, D = len(enc.channel_groups), enc.dim

    avail_csv = vox_dir / "availability.csv"
    if not avail_csv.exists():
        avail_csv = pr.PDBBIND_DIR / "voxels" / "availability.csv"
    av = pd.read_csv(avail_csv)
    ok = set(av[av["has_atoms"] & av["has_density"]]["pdb_id"])
    sm, _ = pr.load_frozen_split_map("lp_edrscc_v2")
    pids = sorted(p for p in sm if p in ok)[:args.n]
    rng = np.random.RandomState(0)
    perm = rng.permutation(len(pids))
    ntr, nva = int(0.6 * len(pids)), int(0.2 * len(pids))
    split_of = {}
    for j, i in enumerate(perm):
        split_of[pids[i]] = "train" if j < ntr else ("val" if j < ntr + nva else "test")
    print(f"  pockets {len(pids)}  -> train {ntr} / val {nva} / test {len(pids)-ntr-nva}"
          f"  (pocket-level split)\n")

    loader = DataLoader(_Vox(pids, atom_dir, vox_dir / "density", split_of),
                        batch_size=args.batch, shuffle=False, num_workers=args.workers,
                        collate_fn=_collate, pin_memory=str(dev).startswith("cuda"))

    S = {(s, d, sp): _Mom(DIMS[d], dev) for s in STUDENTS for d in DEPTHS for sp in SPLITS}
    T = {(t, sp): _Mom(D, dev) for t in TEACHERS for sp in SPLITS}
    X = {(s, d, t, sp): torch.zeros(DIMS[d], D, dtype=torch.float64, device=dev)
         for s in STUDENTS for d in DEPTHS for t in TEACHERS for sp in SPLITS}

    seen = 0
    with torch.no_grad():
        for bi, (x, sps) in enumerate(loader):
            if x is None:
                continue
            x = x.to(dev, non_blocking=True)
            lig, poc = x[:, :7], x[:, 7:11]
            m = _ligand_mask(x)
            eps_ = torch.randn(lig.shape, generator=torch.Generator(device="cpu")
                               .manual_seed(1234 + bi)).to(dev)
            tea = {}
            for t in TEACHERS:
                xi = x.clone()
                if t != "holo":
                    xi[:, :7] = 0.0
                    xi[:, 11:13] = xi[:, 11:13] * (~m).float()
                tk = pr.encode_tokens(enc, xi)
                tea[t] = tk.reshape(tk.shape[0], nG, GP ** 3, D).mean(1)      # (B,512,D)
            stu = {s: _unet_stages(unet, (lig + sigma * eps_) if s == "sig0.9"
                                  else (sigma * eps_), poc) for s in STUDENTS}
            for sp in SPLITS:
                sel = [i for i, v in enumerate(sps) if v == sp]
                if not sel:
                    continue
                idx = torch.tensor(sel, device=dev)
                for t in TEACHERS:
                    Yt = tea[t][idx].reshape(-1, D)
                    T[(t, sp)].add(Yt)
                    for s in STUDENTS:
                        for d in DEPTHS:
                            Xt = _tok(stu[s][d][idx]).reshape(-1, DIMS[d])
                            X[(s, d, t, sp)] += Xt.double().T @ Yt.double()
                for s in STUDENTS:
                    for d in DEPTHS:
                        S[(s, d, sp)].add(_tok(stu[s][d][idx]).reshape(-1, DIMS[d]))
            seen += x.shape[0]
            if bi % 10 == 0:
                print(f"    {seen}/{len(pids)}", flush=True)

    rows = []
    for s in STUDENTS:
        for t in TEACHERS:
            for d in DEPTHS:
                mx = {sp: S[(s, d, sp)] for sp in SPLITS}
                my = {sp: T[(t, sp)] for sp in SPLITS}
                fwd = {sp: X[(s, d, t, sp)] for sp in SPLITS}
                bwd = {sp: X[(s, d, t, sp)].T for sp in SPLITS}
                r2f, vf, lf = ridge_r2(mx, my, fwd, dev)          # teacher <- student
                r2b, vb, lb = ridge_r2(my, mx, bwd, dev)          # student <- teacher
                spec_, n9, n7, n5 = cca_spectrum(mx, my, fwd, dev)
                rows.append(dict(student=s, teacher=t, depth=d, dim=DIMS[d],
                                 r2_teacher_from_student=r2f, r2_student_from_teacher=r2b,
                                 cca_gt_0p9=n9, cca_gt_0p7=n7, cca_gt_0p5=n5,
                                 cca_top1=spec_[0], cca_top5=spec_[4] if len(spec_) > 4 else np.nan))
                print(f"  {s:<7}->{t:<10} {d:<3} dim{DIMS[d]:>4} | "
                      f"R2(T<-S) {r2f:.3f}  R2(S<-T) {r2b:.3f} | "
                      f"CCA >0.9:{n9:>3} >0.7:{n7:>3} >0.5:{n5:>3}  top1 {spec_[0]:.3f}",
                      flush=True)

    df = pd.DataFrame(rows)
    tag = args.tag or f"{UNET_CKPT.parent.name[:24]}_{CDG_EXP.name}"
    csv = VOX / "test" / "results" / f"repa_b1_linear_{tag}.csv"
    csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv, index=False)
    print(f"\n  -> {csv}")


if __name__ == "__main__":
    main()
