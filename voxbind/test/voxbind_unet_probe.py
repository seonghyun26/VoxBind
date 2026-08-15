"""voxbind_unet_probe.py — affinity probe on the VoxBind U-Net's own representation.

Answers A1's missing arm: how much binding-affinity signal does the *generator's*
U-Net carry, compared with the CDG teacher (0.644) and the coords-only control (0.596)?

The generator is a walk-jump denoiser, so "its representation" is depth- and
noise-dependent. We therefore extract, in one pass over PDBbind:

  depths   L0 (64^3, 32ch) · L1 (32^3, 64) · L2 (16^3, 128) · L3 (8^3, 512) · BN (middle, 8^3, 512)
  ligand   sig0.9 (training condition)  ·  clean (sigma=0, upper bound)
           ·  noise (ligand replaced by N(0,sigma) = walk-jump chain init, i.e. what
              actually conditions generation before any ligand exists)

Each feature is the spatial mean-pool of that stage, matching how the CDG probe
mean-pools its tokens. Probing then reuses 01c_pdbbind_probe's dataset/head/split
machinery verbatim so the numbers are comparable to the model_zoo table.

Run:
    python test/voxbind_unet_probe.py extract --max 64          # smoke
    python test/voxbind_unet_probe.py extract
    python test/voxbind_unet_probe.py probe
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[2]
VOX = REPO / "voxbind"
sys.path.insert(0, str(REPO))

_spec = importlib.util.spec_from_file_location(
    "p01c", str(VOX / "dataset" / "01c_pdbbind_probe.py"))
pr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pr)

from voxbind.models import create_model  # noqa: E402  (after sys.path insert)

CKPT = VOX / "exps" / "exp_sig0.9+prefetch_factor16_wjs.n_targets0" / "checkpoint.pth.tar"
ATOM_DIR = VOX / "dataset" / "data" / "pdbbind" / "voxels_v5" / "atoms"
OUT = VOX / "dataset" / "data" / "pdbbind" / "features" / "voxbind_unet_ep923.pt"
SPLIT = "lp_edrscc_v2"
VARIANTS = ("sig0.9", "clean", "noise")
DEPTHS = ("L0", "L1", "L2", "L3", "BN")


# ── extraction ────────────────────────────────────────────────────────────────

class _Atoms(Dataset):
    """(pid, (11,G,G,G) float32) — 7 ligand-element + 4 pocket-element channels."""

    def __init__(self, pids):
        self.pids = list(pids)

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, i):
        pid = self.pids[i]
        try:
            a = np.load(ATOM_DIR / f"{pid}.npy").astype(np.float32)
            return pid, torch.from_numpy(a), ""
        except Exception as e:                                    # logged, not fatal
            return pid, None, repr(e)[:160]


def _collate(batch):
    good = [(p, x) for p, x, e in batch if x is not None]
    errs = [(p, e) for p, x, e in batch if x is None]
    if not good:
        return [], None, errs
    pids, xs = zip(*good)
    return list(pids), torch.stack(xs, 0), errs


@torch.no_grad()
def _stage_features(model, ligand, pocket):
    """Run the U-Net down path once; return {depth: (B, C) mean-pooled}."""
    unet = model.unet3d
    x = model.ligand_encoder(ligand) + model.pocket_encoder(pocket)
    x = unet.grid_projection(x)
    out, lvl = {}, 0
    n = len(unet.down)
    for i, m in enumerate(unet.down):
        x = m(x)
        # end of a resolution level = last module before a Downsample (or before middle)
        nxt = unet.down[i + 1] if i + 1 < n else None
        if nxt is None or type(nxt).__name__ == "Downsample":
            if type(m).__name__ != "Downsample":
                out[DEPTHS[lvl]] = x.mean(dim=(2, 3, 4)).float().cpu()
                lvl += 1
    x = unet.middle(x)
    out["BN"] = x.mean(dim=(2, 3, 4)).float().cpu()
    return out


def extract(args):
    dev = args.device
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    print(f"  ckpt        : {CKPT.name}  (epoch {ckpt['epoch']})")
    print(f"  smooth_sigma: {cfg.smooth_sigma}")

    model = create_model(cfg, device=dev)
    missing, unexpected = model.load_state_dict(ckpt["state_dict_ema"], strict=True), None
    model.eval()
    sigma = float(cfg.smooth_sigma)

    split_map, scheme = pr.load_frozen_split_map(SPLIT)
    have = {p.stem for p in ATOM_DIR.glob("*.npy")}
    pids = sorted(p for p in split_map if p in have)
    if args.max:
        pids = pids[:args.max]
    print(f"  split       : {scheme}  ({len(split_map):,} pids, {len(pids):,} with atom voxels)")
    print(f"  device      : {dev}   batch {args.batch}")

    loader = DataLoader(_Atoms(pids), batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, collate_fn=_collate,
                        pin_memory=str(dev).startswith("cuda"))

    feats = {v: {d: {} for d in DEPTHS} for v in VARIANTS}
    n_err, seen = 0, 0
    for bi, (bpids, x, errs) in enumerate(loader):
        n_err += len(errs)
        if x is None:
            continue
        x = x.to(dev, non_blocking=True)
        lig, poc = x[:, :7], x[:, 7:11]
        # deterministic per-batch noise so reruns reproduce exactly
        g = torch.Generator(device="cpu").manual_seed(1234 + bi)
        eps = torch.randn(lig.shape, generator=g).to(dev)
        for var in VARIANTS:
            if var == "clean":
                l_in = lig
            elif var == "sig0.9":
                l_in = lig + sigma * eps
            else:                                  # walk-jump chain init: no ligand at all
                l_in = sigma * eps
            for d, v in _stage_features(model, l_in, poc).items():
                for pid, vec in zip(bpids, v):
                    feats[var][d][pid] = vec.clone()
        seen += len(bpids)
        if bi % 20 == 0:
            print(f"    {seen:>6}/{len(pids)}  err={n_err}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ckpt": str(CKPT), "epoch": ckpt["epoch"], "sigma": sigma,
                "split": scheme, "variants": list(VARIANTS), "depths": list(DEPTHS),
                "features": feats}, OUT)
    dims = {d: len(next(iter(feats["clean"][d].values()))) for d in DEPTHS}
    print(f"\n  saved {seen:,} complexes -> {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)")
    print(f"  dims: {dims}")


# ── probing ───────────────────────────────────────────────────────────────────

HP = dict(max_epochs=200, patience=30, batch_size=64, lr=1e-3, weight_decay=1e-4,
          hidden=128, dropout=0.1, head="scalar", soft_sigma=1.0)


def _zscore(data):
    """Standardize features with TRAIN-split statistics (the probe head has no input norm,
    and U-Net stage activations are not unit-scale like post-LayerNorm ViT tokens; without
    this a scale artifact is indistinguishable from missing signal)."""
    mu = data["train"]["X"].mean(0, keepdims=True)
    sd = data["train"]["X"].std(0, keepdims=True) + 1e-6
    out = {k: dict(v) for k, v in data.items()}
    for k in ("train", "val", "test"):
        out[k]["X"] = (data[k]["X"] - mu) / sd
    return out


def probe(args):
    blob = torch.load(OUT, map_location="cpu", weights_only=False)
    feats = blob["features"]
    lp_df = pr.load_lp_index(pr.LP_CSV)
    split_map, scheme = pr.load_frozen_split_map(SPLIT)
    dev = args.device
    print(f"  features: {OUT.name} (ep{blob['epoch']}, sigma={blob['sigma']}, split={scheme})\n")

    rows = []
    for var in blob["variants"]:
        for d in blob["depths"]:
            fd = {p: v for p, v in feats[var][d].items()}
            raw = pr.build_dataset(fd, lp_df, drop_covalent=True, cl1_only=False,
                                   split_map=split_map)
            # BN is the headline comparison vs CDG — run it raw as well, to show the
            # z-score is not what is carrying (or hiding) the result.
            norms = ["z", "raw"] if d == "BN" else ["z"]
            for norm in norms:
                data = _zscore(raw) if norm == "z" else raw
                ms = [pr.train_one(data, seed=s, device=dev, **HP) for s in range(args.seeds)]
                gv = lambda k: np.array([m[k] for m in ms])
                rows.append(dict(variant=var, depth=d, norm=norm,
                                 dim=data["train"]["X"].shape[1],
                                 n_train=ms[0]["n_train"], n_test=ms[0]["n_test"],
                                 rho=gv("test_spearman").mean(), rho_sd=gv("test_spearman").std(),
                                 rmse=gv("test_rmse").mean()))
                r = rows[-1]
                print(f"  {var:<8} {d:<3} {norm:<3} dim{r['dim']:>4}  "
                      f"rho {r['rho']:.3f}+-{r['rho_sd']:.3f}  rmse {r['rmse']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    csv = VOX / "test" / "results" / "voxbind_unet_probe.csv"
    csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv, index=False)
    piv = df[df.norm == "z"].pivot(index="depth", columns="variant", values="rho")
    print(f"\n{piv.reindex(list(DEPTHS)).round(3)}")
    print(f"\n  reference (model_zoo, same split): CDG champion 0.644 | "
          f"CDG efficient60m 0.641 | coords-only 0.596")
    print(f"  -> {csv}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pe = sub.add_parser("extract")
    pe.add_argument("--batch", type=int, default=8)
    pe.add_argument("--workers", type=int, default=4)
    pe.add_argument("--max", type=int, default=0)
    pe.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    pe.set_defaults(func=extract)
    pp = sub.add_parser("probe")
    pp.add_argument("--seeds", type=int, default=3)
    pp.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    pp.set_defaults(func=probe)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
