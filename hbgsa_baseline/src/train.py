"""train.py — train + evaluate HBGSA on the LP-PDBbind new_split.

Trains end-to-end on new_split==train, early-stops on val (Pearson R), and
reports test metrics on the full test set AND nested subsets (CL1-clean, and any
external pdb_id list via --probe_pids) so the number drops into the matching
VoxBind probe table. HBGSA is a fully-supervised baseline (not a frozen probe).

Example:
  python train.py --tag full_seed0 --seeds 0,1,2 --epochs 150
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

from config import RESULTS_DIR
from dataset import HBGSADataset, collate
from featurize import build_smiles_vocab
from manifest import build_manifest
from model import HBGSA, hybrid_loss


def set_seed(s: int):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def move(batch, device):
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device, non_blocking=True)
    return batch


@torch.no_grad()
def predict(model, loader, device, y_mean=0.0, y_std=1.0):
    """Returns (pids, raw-pK targets, de-standardized pK predictions)."""
    model.eval()
    pids, ys, ps = [], [], []
    for batch in loader:
        pid = batch["pdb_id"]
        batch = move(batch, device)
        pred = model(batch) * y_std + y_mean          # de-standardize → pK units
        pids += pid
        ys.append(batch["y"].cpu().numpy())
        ps.append(pred.cpu().numpy())
    return np.array(pids), np.concatenate(ys), np.concatenate(ps)


def metrics(y, p) -> dict:
    return {
        "pearson": float(pearsonr(y, p)[0]),
        "spearman": float(spearmanr(y, p)[0]),
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "mae": float(np.mean(np.abs(y - p))),
        "n": int(len(y)),
    }


def run_one_seed(manifest, vocab, seed, args, device):
    set_seed(seed)
    sub = {s: manifest[manifest.new_split == s] for s in ("train", "val", "test")}
    loaders = {
        s: DataLoader(HBGSADataset(df, vocab), batch_size=args.batch_size,
                      shuffle=(s == "train"), collate_fn=collate,
                      num_workers=args.num_workers, pin_memory=True, drop_last=(s == "train"))
        for s, df in sub.items()
    }
    # standardize target on the TRAIN split (correlation is affine-invariant; this
    # only conditions the SmoothL1 term so the absolute scale calibrates properly)
    y_mean = float(sub["train"].pK.mean())
    y_std = float(sub["train"].pK.std()) or 1.0

    model = HBGSA(
        smiles_vocab_size=len(vocab),
        seq_d_model=args.d_model, smi_d_model=args.d_model, emb_dim=args.emb_dim,
        n_layers=args.n_layers, n_heads=args.n_heads, gcn_hidden=args.gcn_hidden,
        pocket_hidden=args.pocket_hidden, head_hidden=args.head_hidden,
    ).to(device)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"  seed{seed} model params: {n_param:,} (~{n_param/1e6:.1f}M)", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=6)

    best_val, best_state, bad = -1e9, None, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        for batch in loaders["train"]:
            batch = move(batch, device)
            target = (batch["y"] - y_mean) / y_std
            loss, _, _ = hybrid_loss(model(batch), target)
            opt.zero_grad(); loss.backward(); opt.step()
        _, yv, pv = predict(model, loaders["val"], device, y_mean, y_std)
        val_s = float(spearmanr(yv, pv)[0])          # select on val Spearman (matches probe)
        val_r = float(pearsonr(yv, pv)[0])
        sched.step(val_s)
        if val_s > best_val:
            best_val, bad = val_s, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        print(f"  seed{seed} ep{ep:3d}  val_spearman={val_s:.4f}  val_pearson={val_r:.4f}  "
              f"best_val_ρ={best_val:.4f}  ({time.time()-t0:.1f}s)  lr={opt.param_groups[0]['lr']:.1e}",
              flush=True)
        if bad >= args.patience:
            print(f"  seed{seed} early-stop at ep{ep} (no val improve {args.patience} ep)", flush=True)
            break

    model.load_state_dict(best_state)
    pid, yt, pt = predict(model, loaders["test"], device, y_mean, y_std)
    _, yvb, pvb = predict(model, loaders["val"], device, y_mean, y_std)

    # val (best model) + full test + nested subsets
    out = {"val": metrics(yvb, pvb), "full": metrics(yt, pt)}
    cl1_pids = set(manifest[(manifest.new_split == "test") & (manifest.CL1 == True)].pdb_id)  # noqa: E712
    mask = np.array([p in cl1_pids for p in pid])
    if mask.any():
        out["cl1"] = metrics(yt[mask], pt[mask])
    if args.probe_pids:
        keep = set(l.strip().lower() for l in Path(args.probe_pids).read_text().split())
        m2 = np.array([p in keep for p in pid])
        if m2.any():
            out["probe"] = metrics(yt[m2], pt[m2])
    out["_best_val_spearman"] = best_val
    out["_preds"] = {"pdb_id": pid.tolist(), "y": yt.tolist(), "pred": pt.tolist()}
    return out


def main():
    ap = argparse.ArgumentParser(description="Train/eval HBGSA on LP-PDBbind new_split")
    ap.add_argument("--tag", required=True, help="run tag for the results CSV")
    ap.add_argument("--seeds", default="0", help="comma-separated seeds")
    ap.add_argument("--no_cl1", action="store_true", help="disable the default CL1-clean filter")
    ap.add_argument("--restrict_ids", default="", help="file of pdb_ids to restrict manifest to (matched split)")
    ap.add_argument("--keep_covalent", action="store_true")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    # model size knobs (defaults = the ~0.77M paper-scale model)
    ap.add_argument("--emb_dim",      type=int, default=128, help="per-branch embedding dim")
    ap.add_argument("--d_model",      type=int, default=128, help="seq/SMILES transformer width")
    ap.add_argument("--n_layers",     type=int, default=2,   help="transformer layers per attn branch")
    ap.add_argument("--n_heads",      type=int, default=4,   help="attention heads (must divide d_model)")
    ap.add_argument("--gcn_hidden",   type=int, default=128)
    ap.add_argument("--pocket_hidden",type=int, default=128)
    ap.add_argument("--head_hidden",  type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--probe_pids", default="", help="optional file of test pdb_ids to also score")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"=== HBGSA train/eval  (device={device})  tag={args.tag} ===")
    manifest = build_manifest(cl1_only=not args.no_cl1, drop_covalent=not args.keep_covalent)
    if args.restrict_ids:
        keep = {l.strip().lower() for l in Path(args.restrict_ids).read_text().split()}
        n0 = len(manifest)
        manifest = manifest[manifest.pdb_id.isin(keep)].reset_index(drop=True)
        print(f"  restrict_ids: kept {len(manifest)}/{n0} complexes "
              f"(matched to {Path(args.restrict_ids).name})")
    vocab = build_smiles_vocab(manifest[manifest.new_split == "train"].smiles)
    print(f"  smiles vocab: {len(vocab)}   "
          f"splits: " + " ".join(f"{s}={int((manifest.new_split==s).sum())}"
                                  for s in ("train", "val", "test")))

    seeds = [int(s) for s in args.seeds.split(",")]
    rows = []
    for seed in seeds:
        res = run_one_seed(manifest, vocab, seed, args, device)
        (RESULTS_DIR / f"preds_{args.tag}_seed{seed}.json").write_text(json.dumps(res.pop("_preds")))
        for subset, m in res.items():
            if subset.startswith("_"):
                continue
            rows.append({"tag": args.tag, "seed": seed, "subset": subset,
                         "best_val_spearman": round(res["_best_val_spearman"], 4), **m})
        print(f"  >> seed{seed}: full ρ={res['full']['spearman']:.4f} r={res['full']['pearson']:.4f} "
              f"rmse={res['full']['rmse']:.3f}  | cl1 ρ="
              f"{res.get('cl1', {}).get('spearman', float('nan')):.4f}", flush=True)

    df = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / f"hbgsa_results_{args.tag}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n  saved → {out_csv}")
    print("\n=== summary (mean ± std over seeds) ===")
    for subset in df.subset.unique():
        d = df[df.subset == subset]
        print(f"  {subset:6s} n={int(d.n.iloc[0]):5d}  "
              f"pearson={d.pearson.mean():.4f}±{d.pearson.std():.4f}  "
              f"spearman={d.spearman.mean():.4f}  rmse={d.rmse.mean():.3f}")


if __name__ == "__main__":
    main()
