"""cdg_pocketonly_probe.py — the CDG teacher measured under the GENERATION condition.

The 0.644 / 0.641 headline is a HOLO number: the encoder sees 7 ligand atom channels and
a 2Fo-Fc density that contains the ligand's own electron density. VoxBind at sampling time
has neither. So the teacher/student comparison that actually governs a U-REPA transfer is
the pocket-only one, against the U-Net's ligand-free ("noise") bottleneck.

Three teacher conditions on the identical split/head:
  holo      : full 13ch [7 lig | 4 poc | rho | |grad rho|]        -> reproduces the zoo number
  atoms0    : ligand ATOM channels zeroed, density untouched      -> what the current
              frozen-enc generator actually feeds (density still carries the ligand)
  atoms0_dm : ligand atoms zeroed AND rho/|grad rho| blanked inside the dilated ligand
              footprint -> the honest apo-like condition (cf. model.density_mask_ligand,
              density_mask_threshold=0.2, density_mask_dilate=2)

Run:
    python test/cdg_pocketonly_probe.py --max 64      # smoke
    python test/cdg_pocketonly_probe.py
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

DEFAULT_EXP = VOX / "model_zoo" / "efficient_60m_v3_mask085"   # dim 512 == U-Net bottleneck width
COND = "atomblob_density_gradmag"
SPLIT = "lp_edrscc_v2"
FEAT_DIR = VOX / "dataset" / "data" / "pdbbind" / "features"
VARIANTS = ("holo", "atoms0", "atoms0_dm")
MASK_THRESH, MASK_DILATE = 0.2, 2
HP = dict(max_epochs=200, patience=30, batch_size=64, lr=1e-3, weight_decay=1e-4,
          hidden=128, dropout=0.1, head="scalar", soft_sigma=1.0)


class _Vox(Dataset):
    def __init__(self, pids, atom_dir, dens_dir):
        self.pids, self.atom_dir, self.dens_dir = list(pids), atom_dir, dens_dir

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, i):
        pid = self.pids[i]
        try:
            x = pr.load_voxels_for(pid, COND, 13, self.atom_dir, self.dens_dir,
                                   input_mode="atomblob_density", with_gradmag=True)
            return pid, x, ""
        except Exception as e:
            return pid, None, repr(e)[:140]


def _collate(b):
    good = [(p, x) for p, x, e in b if x is not None]
    errs = [(p, e) for p, x, e in b if x is None]
    if not good:
        return [], None, errs
    p, x = zip(*good)
    return list(p), torch.stack(x, 0), errs


def _ligand_mask(x):
    """(B,1,G,G,G) bool: dilated footprint of the ligand atom channels."""
    occ = x[:, :7].sum(1, keepdim=True)
    m = (occ > MASK_THRESH).float()
    k = 2 * MASK_DILATE + 1
    return F.max_pool3d(m, kernel_size=k, stride=1, padding=MASK_DILATE) > 0.5


def extract(args):
    dev = args.device
    exp, out = Path(args.exp_dir), Path(args.out)
    cfg = OmegaConf.load(exp / "cfg.yaml")
    spec = pr.infer_feature_spec(COND, cfg, "auto")
    vox_dir = pr.voxel_dir_for("v5")
    atom_dir = pr.atom_dir_for(vox_dir, spec.atom_source)
    dens_dir = vox_dir / "density"
    enc = pr.load_encoder(exp, args.epoch, dev, cfg=cfg)
    print(f"  encoder    : {exp.name} dim={enc.dim} n_in={enc.n_in_channels} "
          f"groups={getattr(enc, 'channel_groups', None)}")
    print(f"  atom_source: {spec.atom_source} ({atom_dir.name})   dens: {dens_dir.name}")

    split_map, scheme = pr.load_frozen_split_map(SPLIT)
    # availability.csv lives in v1 only; v2..v5 share its pid universe (see run_features)
    avail_csv = vox_dir / "availability.csv"
    if not avail_csv.exists():
        avail_csv = pr.PDBBIND_DIR / "voxels" / "availability.csv"
    avail = pd.read_csv(avail_csv)
    ok = set(avail[avail["has_atoms"] & avail["has_density"]]["pdb_id"])
    pids = sorted(p for p in split_map if p in ok)
    if args.max:
        pids = pids[:args.max]
    print(f"  split      : {scheme}  {len(pids):,} complexes with atoms+density\n")

    loader = DataLoader(_Vox(pids, atom_dir, dens_dir), batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, collate_fn=_collate,
                        pin_memory=str(dev).startswith("cuda"))
    feats = {v: {} for v in VARIANTS}
    seen, n_err = 0, 0
    with torch.no_grad():
        for bi, (bp, x, errs) in enumerate(loader):
            n_err += len(errs)
            if x is None:
                continue
            x = x.to(dev, non_blocking=True)
            m = _ligand_mask(x)
            for var in VARIANTS:
                xi = x.clone()
                if var != "holo":
                    xi[:, :7] = 0.0
                if var == "atoms0_dm":
                    xi[:, 11:13] = xi[:, 11:13] * (~m).float()
                v = pr.encode_tokens(enc, xi).mean(dim=1).float().cpu()
                for pid, vec in zip(bp, v):
                    feats[var][pid] = vec.clone()
            seen += len(bp)
            if bi % 25 == 0:
                print(f"    {seen}/{len(pids)}  err={n_err}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"exp": str(exp), "epoch": args.epoch, "split": scheme,
                "variants": list(VARIANTS), "features": feats}, out)
    print(f"\n  saved {seen:,} -> {out} ({out.stat().st_size/1e6:.1f} MB)")


def probe(args):
    blob = torch.load(Path(args.out), map_location="cpu", weights_only=False)
    lp_df = pr.load_lp_index(pr.LP_CSV)
    split_map, _ = pr.load_frozen_split_map(SPLIT)
    rows = []
    for var in blob["variants"]:
        data = pr.build_dataset(blob["features"][var], lp_df, drop_covalent=True,
                                cl1_only=False, split_map=split_map)
        # RAW by default: MLP2 has no input norm and every published model_zoo rho was
        # produced on raw features. Standardizing is a DIFFERENT protocol — it is a no-op
        # on U-Net stage features (0.547 vs 0.548) but costs the ViT teacher ~0.05
        # (champion holo 0.644 raw vs 0.590 z), so a z-scored number is not comparable
        # to the zoo table. Opt in with --zscore only for a like-for-like z comparison.
        if args.zscore:
            mu = data["train"]["X"].mean(0, keepdims=True)
            sd = data["train"]["X"].std(0, keepdims=True) + 1e-6
            for k in ("train", "val", "test"):
                data[k]["X"] = (data[k]["X"] - mu) / sd
        ms = [pr.train_one(data, seed=s, device=args.device, **HP) for s in range(args.seeds)]
        g = lambda k: np.array([m[k] for m in ms])
        rows.append(dict(variant=var, dim=data["train"]["X"].shape[1],
                         n_test=ms[0]["n_test"], rho=g("test_spearman").mean(),
                         rmse=g("test_rmse").mean()))
        r = rows[-1]
        print(f"  {r['variant']:<10} dim{r['dim']:>4}  rho {r['rho']:.3f}  rmse {r['rmse']:.3f}",
              flush=True)
    df = pd.DataFrame(rows)
    csv = VOX / "test" / "results" / f"cdg_pocketonly_probe_{Path(args.exp_dir).name}.csv"
    csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv, index=False)
    print(f"\n  matched student (U-Net BN, ligand-free) 0.474 raw | pocket-composition null 0.145 raw")
    print(f"  -> {csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", default=str(DEFAULT_EXP), help="frozen CDG teacher exp dir")
    ap.add_argument("--epoch", type=int, default=49)
    ap.add_argument("--out", default=None, help="feature cache (default: derived from exp_dir)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--probe_only", action="store_true")
    ap.add_argument("--zscore", action="store_true",
                    help="standardize features with train stats (NOT the published protocol)")
    args = ap.parse_args()
    if args.out is None:
        args.out = str(FEAT_DIR / f"cdg_pocketonly_{Path(args.exp_dir).name}.pt")
    if not args.probe_only:
        extract(args)
    probe(args)


if __name__ == "__main__":
    main()
