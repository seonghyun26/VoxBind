"""holdout_eval.py — Evaluate HonestAffinity on the 2019 temporal holdout as an external test set.

Same protocol as casf_eval.py, but the external set is the 2019 PDBbind holdout
(voxbind/splits/holdout2019_eval.csv):
  - Train on lp_edrscc_v2 TRAIN split (3 seeds, best-val checkpoint restored).
  - Build records for the holdout pids (pK from holdout2019_eval.csv, SMILES from the
    ligand SDF/mol2 — holdout complexes are NOT in LP_PDBBind.csv — structures from the
    standard struct dirs: pbpp-2020 / misato_qm_built).
  - Cache ESM-2-650M embeddings (hash-keyed cache shared with lp_edrscc_v2).
  - Per seed: predict the holdout and dump base/_casf/HonestAffinity_holdout2019_preds_seed{S}.csv
    (pid,pred,y) — the per-seed format common94_holdout.py's seed_csv() reads (→ mean±std, and
    the PLINDER-v2 / ED / common-set filtering is applied downstream by common94_holdout.py).

Usage: cd base/honestaffinity/src && CUDA_VISIBLE_DEVICES=<gpu> \
       /home/shpark/.conda/envs/dsmbind/bin/python holdout_eval.py --seeds 0 1 2
"""
import os
import sys
import csv
import json
import math
import time
import random
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr

# ── paths ──────────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
HA_ROOT = os.path.dirname(HERE)          # base/honestaffinity/
BASE_DIR = os.path.dirname(HA_ROOT)      # base/
REPO = os.path.dirname(BASE_DIR)         # VoxBind/

CACHE_DIR = os.path.join(HA_ROOT, "cache")
PREDS_DIR = os.path.join(BASE_DIR, "_casf")   # common94_holdout.py reads seed CSVs from here
SUMMARY_OUT = os.path.join(PREDS_DIR, "HonestAffinity_holdout2019.json")

LP_CSV = os.path.join(REPO, "voxbind", "dataset", "data", "pdbbind", "raw", "LP_PDBBind.csv")
HOLDOUT_CSV = os.path.join(REPO, "voxbind", "splits", "holdout2019_eval.csv")

MAX_LEN = 1000

# ── imports from data_prep (same helpers casf_eval uses) ───────────────────────
from data_prep import (
    resolve, parse_residues, pocket_keys, best_window, ligand_smiles,
    AA3TO1, cache_esm_embeddings,
)
from model import HonestAffinityPocket
from dataset import AffinityDataset, collate, load_records
from smiles_tokenizer import vocab_size


# ── holdout record builder ─────────────────────────────────────────────────────
def build_holdout_records(csv_path, gpu=0):
    """Per-pid records for the 2019 holdout. SMILES from the ligand structure (holdout is
    not in LP_PDBBind.csv); LP smiles used only as a fallback if a pid happens to overlap."""
    df = pd.read_csv(csv_path)
    df["pid"] = df["pid"].astype(str).str.lower()
    pk_map = dict(zip(df["pid"], df["pK"]))
    in_v2train = dict(zip(df["pid"], df.get("in_v2train", pd.Series([0] * len(df)))))

    lp = pd.read_csv(LP_CSV).rename(columns={"Unnamed: 0": "pid", "value": "pK_lp"})
    lp["pid"] = lp["pid"].astype(str).str.lower()
    smi_map = dict(zip(lp["pid"], lp["smiles"]))

    records, fails = [], {}
    for pid in df["pid"].tolist():
        prot_pdb, poc_pdb, sdf, mol2 = resolve(pid)
        if prot_pdb is None:
            fails[pid] = "no_structure"; continue
        residues = parse_residues(prot_pdb)
        if not residues:
            fails[pid] = "empty_protein"; continue
        pkeys = pocket_keys(poc_pdb)
        seq = "".join(AA3TO1.get(rn, "X") for _, rn in residues)
        mask = np.array([1 if k in pkeys else 0 for k, _ in residues], dtype=np.int8)
        if len(seq) > MAX_LEN:
            s, e = best_window(mask, MAX_LEN)
            seq = seq[s:e]; mask = mask[s:e]
        if mask.sum() == 0:
            fails[pid] = "empty_pocket"; continue

        smi = ligand_smiles(sdf, mol2) or smi_map.get(pid)
        if not isinstance(smi, str) or not smi:
            fails[pid] = "no_smiles"; continue

        pK = pk_map.get(pid)
        if pK is None or not np.isfinite(pK):
            fails[pid] = "no_label"; continue

        records.append({
            "pid": pid, "seq": seq, "pocket_mask": mask.tolist(),
            "smiles": smi, "pK": float(pK), "in_v2train": int(in_v2train.get(pid, 0)),
        })

    vc = dict(pd.Series(list(fails.values())).value_counts()) if fails else {}
    print(f"[holdout] built {len(records)}/{len(df)}  fails={vc}", flush=True)
    n_new = cache_esm_embeddings([r["seq"] for r in records], gpu=gpu)
    print(f"[holdout] ESM cache: {n_new} new sequences computed", flush=True)
    return records


# ── training (identical schedule to casf_eval) ─────────────────────────────────
def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def predict(model, records, dev, bs, num_workers):
    ds = AffinityDataset(records)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=collate,
                    num_workers=num_workers, pin_memory=True)
    model.eval()
    preds = []
    with torch.no_grad():
        for esm, prot_pad, pmark, smi, smi_pad, y in dl:
            esm = esm.to(dev); prot_pad = prot_pad.to(dev); pmark = pmark.to(dev)
            smi = smi.to(dev); smi_pad = smi_pad.to(dev)
            preds.append(model(esm, prot_pad, pmark, smi, smi_pad).cpu().numpy())
    return np.concatenate(preds)


def val_rmse(model, loader, dev):
    model.eval()
    ps, ys = [], []
    with torch.no_grad():
        for esm, prot_pad, pmark, smi, smi_pad, y in loader:
            esm = esm.to(dev); prot_pad = prot_pad.to(dev); pmark = pmark.to(dev)
            smi = smi.to(dev); smi_pad = smi_pad.to(dev)
            ps.append(model(esm, prot_pad, pmark, smi, smi_pad).cpu().numpy()); ys.append(y.numpy())
    p = np.concatenate(ps); t = np.concatenate(ys)
    return float(np.sqrt(np.mean((p - t) ** 2)))


def train_one_seed(seed, buckets, dev, epochs, patience, bs, lr, wd, num_workers, log_every):
    set_seed(seed)
    dl_tr = DataLoader(AffinityDataset(buckets["train"]), batch_size=bs, shuffle=True,
                       collate_fn=collate, num_workers=num_workers, pin_memory=True,
                       generator=torch.Generator().manual_seed(seed))
    dl_va = DataLoader(AffinityDataset(buckets["val"]), batch_size=bs, shuffle=False,
                       collate_fn=collate, num_workers=num_workers, pin_memory=True)

    model = HonestAffinityPocket(vocab_size=vocab_size()).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.MSELoss()

    best_val, best_state, bad = math.inf, None, 0
    for ep in range(epochs):
        model.train(); t0 = time.time(); tot = 0.0; n = 0
        for esm, prot_pad, pmark, smi, smi_pad, y in dl_tr:
            esm = esm.to(dev); prot_pad = prot_pad.to(dev); pmark = pmark.to(dev)
            smi = smi.to(dev); smi_pad = smi_pad.to(dev); y = y.to(dev)
            opt.zero_grad()
            loss = lossf(model(esm, prot_pad, pmark, smi, smi_pad), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * y.size(0); n += y.size(0)
        sched.step()
        vrmse = val_rmse(model, dl_va, dev); vmse = vrmse ** 2
        improved = vmse < best_val - 1e-5
        if improved:
            best_val = vmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if log_every and (ep % log_every == 0 or ep == epochs - 1 or improved):
            print(f"  seed{seed} ep{ep:02d} train_mse={tot/max(n,1):.4f} val_rmse={vrmse:.4f} "
                  f"{'*' if improved else ''} ({time.time()-t0:.1f}s)", flush=True)
        if bad >= patience:
            print(f"  seed{seed} early stop ep{ep}", flush=True); break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--log_every", type=int, default=5)
    args = ap.parse_args()

    dev = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}", flush=True)

    print("\n=== Step 1: build holdout records + ESM cache ===", flush=True)
    holdout = build_holdout_records(HOLDOUT_CSV, gpu=args.gpu)
    print(f"  {len(holdout)} holdout records ready", flush=True)

    print("\n=== Step 2: load lp_edrscc_v2 records ===", flush=True)
    buckets = load_records("lp_edrscc_v2", CACHE_DIR)
    print(f"  train={len(buckets['train'])} val={len(buckets['val'])} test={len(buckets['test'])}", flush=True)

    print("\n=== Step 3: train+predict (per seed) ===", flush=True)
    os.makedirs(PREDS_DIR, exist_ok=True)
    labels = np.array([r["pK"] for r in holdout])
    per_seed = []
    for s in args.seeds:
        print(f"\n--- seed {s} ---", flush=True)
        model = train_one_seed(s, buckets, dev, args.epochs, args.patience,
                               args.bs, args.lr, args.wd, args.num_workers, args.log_every)
        preds = predict(model, holdout, dev, args.bs, args.num_workers)
        out_csv = os.path.join(PREDS_DIR, f"HonestAffinity_holdout2019_preds_seed{s}.csv")
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["pid", "pred", "y"])
            for r, p in zip(holdout, preds):
                w.writerow([r["pid"], float(p), float(r["pK"])])
        r_all = float(pearsonr(preds, labels)[0])
        rho_all = float(spearmanr(preds, labels)[0])
        rmse_all = float(np.sqrt(np.mean((preds - labels) ** 2)))
        print(f"  seed{s} holdout(n={len(holdout)}) r={r_all:.4f} rho={rho_all:.4f} "
              f"rmse={rmse_all:.4f} → {os.path.basename(out_csv)}", flush=True)
        per_seed.append({"seed": s, "pearson": r_all, "spearman": rho_all,
                         "rmse": rmse_all, "n": len(holdout)})

    ms = lambda k: {"mean": float(np.mean([d[k] for d in per_seed])),
                    "std": float(np.std([d[k] for d in per_seed]))}
    summary = {"model": "HonestAffinity", "set": "holdout2019 (all-ED, pre-filter)",
               "seeds": list(args.seeds), "n": len(holdout),
               "pearson": ms("pearson"), "spearman": ms("spearman"), "rmse": ms("rmse"),
               "per_seed": per_seed}
    with open(SUMMARY_OUT, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {SUMMARY_OUT}")
    print(f"[holdout raw n={len(holdout)}] r={summary['pearson']['mean']:.4f}±{summary['pearson']['std']:.4f}  "
          f"rho={summary['spearman']['mean']:.4f}±{summary['spearman']['std']:.4f}  "
          f"rmse={summary['rmse']['mean']:.4f}±{summary['rmse']['std']:.4f}")
    print("NOTE: this is the raw-record score; the Table-3 number is common94_holdout.py's "
          "common-ED subset (PLINDER-v2 excluded, intersected across methods).")


if __name__ == "__main__":
    main()
