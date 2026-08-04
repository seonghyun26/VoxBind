"""train.py — train HonestAffinity-Pocket on a frozen split, N seeds, report test
Pearson r / Spearman rho / RMSE.

TRAINING (per SPEC): MSE loss on pK; AdamW lr=1e-4 wd=1e-4; 80 epochs cosine annealing;
early-stop patience 20 on val (best-val checkpoint restored for test); batch size 32.

Writes results/results_{split}.json:
  {"pearson":{"mean","std"}, "spearman":{"mean","std"}, "rmse":{"mean","std"},
   "seeds":[...], "n_test":N, "per_seed":[...], ...}
"""
import os
import sys
import json
import math
import time
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
HA_ROOT = os.path.dirname(HERE)
CACHE_DIR = os.path.join(HA_ROOT, "cache")
RESULTS_DIR = os.path.join(HA_ROOT, "results")

from model import HonestAffinityPocket           # noqa: E402
from dataset import AffinityDataset, collate, load_records  # noqa: E402
from smiles_tokenizer import vocab_size            # noqa: E402


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def evaluate(model, loader, dev):
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for esm, prot_pad, pmark, smi, smi_pad, y in loader:
            esm = esm.to(dev); prot_pad = prot_pad.to(dev); pmark = pmark.to(dev)
            smi = smi.to(dev); smi_pad = smi_pad.to(dev)
            out = model(esm, prot_pad, pmark, smi, smi_pad)
            preds.append(out.cpu().numpy())
            ys.append(y.numpy())
    p = np.concatenate(preds); t = np.concatenate(ys)
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    pear = float(pearsonr(p, t)[0])
    spear = float(spearmanr(p, t)[0])
    return pear, spear, rmse, p, t


def train_one(seed, buckets, dev, epochs, patience, bs, lr, wd, num_workers, log_every):
    set_seed(seed)
    ds_tr = AffinityDataset(buckets["train"])
    ds_va = AffinityDataset(buckets["val"])
    ds_te = AffinityDataset(buckets["test"])
    g = torch.Generator(); g.manual_seed(seed)
    dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True, collate_fn=collate,
                       num_workers=num_workers, drop_last=False, generator=g,
                       pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=bs, shuffle=False, collate_fn=collate,
                       num_workers=num_workers, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=bs, shuffle=False, collate_fn=collate,
                       num_workers=num_workers, pin_memory=True)

    model = HonestAffinityPocket(vocab_size=vocab_size()).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.MSELoss()

    best_val = math.inf
    best_state = None
    bad = 0
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        tot = 0.0; n = 0
        for esm, prot_pad, pmark, smi, smi_pad, y in dl_tr:
            esm = esm.to(dev); prot_pad = prot_pad.to(dev); pmark = pmark.to(dev)
            smi = smi.to(dev); smi_pad = smi_pad.to(dev); y = y.to(dev)
            opt.zero_grad()
            out = model(esm, prot_pad, pmark, smi, smi_pad)
            loss = lossf(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * y.size(0); n += y.size(0)
        sched.step()
        vpear, vspear, vrmse, _, _ = evaluate(model, dl_va, dev)
        # early stop on val RMSE (val MSE proxy)
        vmse = vrmse ** 2
        improved = vmse < best_val - 1e-5
        if improved:
            best_val = vmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if log_every and (ep % log_every == 0 or ep == epochs - 1 or improved):
            print(f"  seed{seed} ep{ep:02d} train_mse={tot/max(n,1):.4f} "
                  f"val_rmse={vrmse:.4f} r={vpear:.4f} {'*' if improved else ''} "
                  f"({time.time()-t0:.1f}s)", flush=True)
        if bad >= patience:
            print(f"  seed{seed} early stop at ep{ep} (patience {patience})", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    tpear, tspear, trmse, p, t = evaluate(model, dl_te, dev)
    print(f"  seed{seed} TEST  r={tpear:.4f} rho={tspear:.4f} rmse={trmse:.4f}", flush=True)
    return {"seed": seed, "pearson": tpear, "spearman": tspear, "rmse": trmse,
            "n_test": len(t)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--tag", default="", help="suffix for results filename (e.g. _smoke)")
    args = ap.parse_args()

    dev = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    buckets = load_records(args.split, CACHE_DIR)
    print(f"[{args.split}] train={len(buckets['train'])} val={len(buckets['val'])} "
          f"test={len(buckets['test'])}  device={dev}")

    per_seed = []
    for s in args.seeds:
        print(f"=== seed {s} ===", flush=True)
        per_seed.append(train_one(
            s, buckets, dev, args.epochs, args.patience, args.bs,
            args.lr, args.wd, args.num_workers, args.log_every))

    def agg(key):
        vals = [r[key] for r in per_seed]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    out = {
        "model": "HonestAffinity-Pocket (arXiv 2606.03422, reimpl from spec)",
        "split": args.split,
        "pearson": agg("pearson"),
        "spearman": agg("spearman"),
        "rmse": agg("rmse"),
        "seeds": list(args.seeds),
        "n_test": per_seed[0]["n_test"],
        "per_seed": per_seed,
        "config": {"epochs": args.epochs, "patience": args.patience, "bs": args.bs,
                   "lr": args.lr, "wd": args.wd},
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"results_{args.split}{args.tag}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {path}")
    print(f"[{args.split}] test  r={out['pearson']['mean']:.4f}±{out['pearson']['std']:.4f}  "
          f"rho={out['spearman']['mean']:.4f}±{out['spearman']['std']:.4f}  "
          f"rmse={out['rmse']['mean']:.4f}±{out['rmse']['std']:.4f}")


if __name__ == "__main__":
    main()
