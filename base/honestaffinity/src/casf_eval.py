"""casf_eval.py — Evaluate HonestAffinity on CASF-2016 as an external test set.

Protocol:
  - Train on lp_edrscc_v2 TRAIN split (same as the main paper eval).
  - Build records for the 214 CASF-2016 pids (pK from casf2016_eval.csv,
    SMILES from LP_PDBBind.csv, structures from the standard struct dirs).
  - Cache ESM-2-650M embeddings (reuses the existing hash-keyed cache, so
    proteins that overlap with lp_edrscc_v2 don't get recomputed).
  - 3 seeds: train -> best-val checkpoint -> predict 214 CASF.
  - Metrics computed twice:
      "leaky"    over all 214 (includes the 90 in lp_edrscc_v2 train)
      "nontrain" over the 124 with in_v2train == 0

Output: base/_casf/HonestAffinity.json
"""
import os
import sys
import json
import math
import time
import random
import argparse
import hashlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from scipy.stats import pearsonr, spearmanr

# ── paths ──────────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
HA_ROOT = os.path.dirname(HERE)          # base/honestaffinity/
BASE_DIR = os.path.dirname(HA_ROOT)     # base/
REPO = os.path.dirname(BASE_DIR)        # VoxBind/

CACHE_DIR   = os.path.join(HA_ROOT, "cache")
ESM_DIR     = os.path.join(CACHE_DIR, "esm")
CASF_OUT    = os.path.join(BASE_DIR, "_casf", "HonestAffinity.json")
CKPT_DIR    = os.path.join(HA_ROOT, "cache", "ckpts_casf")

LP_CSV = os.path.join(REPO, "voxbind", "dataset", "data", "pdbbind", "raw", "LP_PDBBind.csv")
CASF_CSV = os.path.join(REPO, "voxbind", "splits", "casf2016_eval.csv")
V2_RECORDS = os.path.join(CACHE_DIR, "records_lp_edrscc_v2.json")

STRUCT_BASES = [
    os.path.join(REPO, "voxbind", "dataset", "data", "pdbbind", "structures", "pbpp-2020"),
    os.path.join(REPO, "voxbind", "dataset", "data", "pdbbind", "structures", "misato_qm_built"),
]

MAX_LEN = 1000

# ── imports from data_prep ─────────────────────────────────────────────────────
from data_prep import (
    resolve, parse_residues, pocket_keys, best_window, ligand_smiles,
    AA3TO1, seq_hash, esm_path, cache_esm_embeddings,
)
from model import HonestAffinityPocket
from dataset import AffinityDataset, collate, load_records
from smiles_tokenizer import vocab_size, encode, PAD_ID


# ── CASF record builder ────────────────────────────────────────────────────────
def build_casf_records(casf_csv, gpu=0):
    """Build per-pid records for the 214 CASF-2016 complexes.

    pK label comes from casf2016_eval.csv (the CASF power-of-10 pK column).
    SMILES from LP_PDBBind.csv (same source as all other baselines);
    fallback to structure ligand SDF/mol2 if absent.
    Structures from the standard struct dirs.
    """
    df = pd.read_csv(casf_csv)
    df["pid"] = df["pid"].astype(str).str.lower()
    pk_map = dict(zip(df["pid"], df["pK"]))
    in_v2train = dict(zip(df["pid"], df["in_v2train"]))

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

        smi = smi_map.get(pid)
        if not isinstance(smi, str) or not smi:
            smi = ligand_smiles(sdf, mol2)
        if not smi:
            fails[pid] = "no_smiles"; continue

        pK = pk_map.get(pid)
        if pK is None or not np.isfinite(pK):
            fails[pid] = "no_label"; continue

        records.append({
            "pid": pid,
            "seq": seq,
            "pocket_mask": mask.tolist(),
            "smiles": smi,
            "pK": float(pK),
            "in_v2train": int(in_v2train.get(pid, 0)),
        })

    print(f"[CASF] built {len(records)}/214  fails={dict(pd.Series(list(fails.values())).value_counts())}")
    if fails:
        print(f"  failed pids: {list(fails.keys())}")

    # cache ESM embeddings for new CASF sequences
    n_new = cache_esm_embeddings([r["seq"] for r in records], gpu=gpu)
    print(f"[CASF] ESM cache: {n_new} new sequences computed")

    return records


# ── per-seed training ──────────────────────────────────────────────────────────
def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def evaluate(model, loader, dev):
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for esm, prot_pad, pmark, smi, smi_pad, y in loader:
            esm = esm.to(dev); prot_pad = prot_pad.to(dev); pmark = pmark.to(dev)
            smi = smi.to(dev); smi_pad = smi_pad.to(dev)
            out = model(esm, prot_pad, pmark, smi, smi_pad)
            preds.append(out.cpu().numpy()); ys.append(y.numpy())
    p = np.concatenate(preds); t = np.concatenate(ys)
    r  = float(pearsonr(p, t)[0])
    rh = float(spearmanr(p, t)[0])
    rm = float(np.sqrt(np.mean((p - t) ** 2)))
    return r, rh, rm, p, t


def train_seed(seed, buckets, casf_records, dev, epochs, patience, bs, lr, wd,
               num_workers, log_every):
    set_seed(seed)
    ds_tr = AffinityDataset(buckets["train"])
    ds_va = AffinityDataset(buckets["val"])
    g = torch.Generator(); g.manual_seed(seed)
    dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True, collate_fn=collate,
                       num_workers=num_workers, drop_last=False, generator=g,
                       pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=bs, shuffle=False, collate_fn=collate,
                       num_workers=num_workers, pin_memory=True)

    model = HonestAffinityPocket(vocab_size=vocab_size()).to(dev)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.MSELoss()

    best_val = math.inf
    best_state = None
    bad = 0
    for ep in range(epochs):
        model.train()
        t0 = time.time(); tot = 0.0; n = 0
        for esm, prot_pad, pmark, smi, smi_pad, y in dl_tr:
            esm = esm.to(dev); prot_pad = prot_pad.to(dev); pmark = pmark.to(dev)
            smi = smi.to(dev); smi_pad = smi_pad.to(dev); y = y.to(dev)
            opt.zero_grad()
            out   = model(esm, prot_pad, pmark, smi, smi_pad)
            loss  = lossf(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * y.size(0); n += y.size(0)
        sched.step()
        _, _, vrmse, _, _ = evaluate(model, dl_va, dev)
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
                  f"val_rmse={vrmse:.4f} {'*' if improved else ''} "
                  f"({time.time()-t0:.1f}s)", flush=True)
        if bad >= patience:
            print(f"  seed{seed} early stop ep{ep}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ── predict on all 214 CASF records ────────────────────────────────────────
    ds_casf = AffinityDataset(casf_records)
    dl_casf = DataLoader(ds_casf, batch_size=bs, shuffle=False, collate_fn=collate,
                         num_workers=num_workers, pin_memory=True)
    _, _, _, preds, _ = evaluate(model, dl_casf, dev)
    labels = np.array([r["pK"] for r in casf_records])
    in_train_flags = np.array([r["in_v2train"] for r in casf_records])

    def metrics(mask):
        p, t = preds[mask], labels[mask]
        return {
            "pearson": float(pearsonr(p, t)[0]),
            "spearman": float(spearmanr(p, t)[0]),
            "rmse": float(np.sqrt(np.mean((p - t) ** 2))),
            "n": int(mask.sum()),
        }

    all_mask    = np.ones(len(casf_records), dtype=bool)
    non_mask    = in_train_flags == 0

    leaky   = metrics(all_mask)
    nontrain = metrics(non_mask)

    print(f"  seed{seed} CASF-leaky(214)   r={leaky['pearson']:.4f} "
          f"rho={leaky['spearman']:.4f} rmse={leaky['rmse']:.4f}", flush=True)
    print(f"  seed{seed} CASF-nontrain(124) r={nontrain['pearson']:.4f} "
          f"rho={nontrain['spearman']:.4f} rmse={nontrain['rmse']:.4f}", flush=True)

    return leaky, nontrain


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu",         type=int,   default=0)
    ap.add_argument("--seeds",       type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs",      type=int,   default=80)
    ap.add_argument("--patience",    type=int,   default=20)
    ap.add_argument("--bs",          type=int,   default=32)
    ap.add_argument("--lr",          type=float, default=1e-4)
    ap.add_argument("--wd",          type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int,   default=4)
    ap.add_argument("--log_every",   type=int,   default=5)
    ap.add_argument("--skip_esm",    action="store_true",
                    help="skip ESM caching (use when already cached)")
    args = ap.parse_args()

    dev = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}", flush=True)

    # ── step 1: build CASF records + ESM cache ──────────────────────────────
    print("\n=== Step 1: build CASF-2016 records ===", flush=True)
    if args.skip_esm:
        # rebuild records without recomputing ESM
        casf_records = build_casf_records.__wrapped__(CASF_CSV, gpu=args.gpu) \
            if hasattr(build_casf_records, "__wrapped__") else None

    casf_records = build_casf_records(CASF_CSV, gpu=args.gpu)
    print(f"  {len(casf_records)} CASF records ready", flush=True)

    # ── step 2: load lp_edrscc_v2 train+val ─────────────────────────────────
    print("\n=== Step 2: load lp_edrscc_v2 records ===", flush=True)
    buckets = load_records("lp_edrscc_v2", CACHE_DIR)
    print(f"  train={len(buckets['train'])} val={len(buckets['val'])} "
          f"test={len(buckets['test'])}", flush=True)

    # ── step 3: 3-seed train + CASF eval ─────────────────────────────────────
    print("\n=== Step 3: train+eval (3 seeds) ===", flush=True)
    all_leaky, all_nontrain = [], []
    for s in args.seeds:
        print(f"\n--- seed {s} ---", flush=True)
        leaky, nontrain = train_seed(
            s, buckets, casf_records, dev,
            args.epochs, args.patience, args.bs, args.lr, args.wd,
            args.num_workers, args.log_every,
        )
        all_leaky.append(leaky)
        all_nontrain.append(nontrain)

    # ── step 4: aggregate ────────────────────────────────────────────────────
    def agg_metric(rows, key):
        vals = [r[key] for r in rows]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    def agg_split(rows):
        return {
            "pearson":  agg_metric(rows, "pearson"),
            "spearman": agg_metric(rows, "spearman"),
            "rmse":     agg_metric(rows, "rmse"),
            "n":        rows[0]["n"],
            "per_seed": rows,
        }

    out = {
        "model":    "HonestAffinity",
        "train":    "lp_edrscc_v2 train",
        "seeds":    list(args.seeds),
        "leaky":    agg_split(all_leaky),
        "nontrain": agg_split(all_nontrain),
    }

    os.makedirs(os.path.dirname(CASF_OUT), exist_ok=True)
    with open(CASF_OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {CASF_OUT}")

    L = out["leaky"];  NT = out["nontrain"]
    print(f"\n[CASF-2016] leaky(n={L['n']})    "
          f"r={L['pearson']['mean']:.4f}±{L['pearson']['std']:.4f}  "
          f"rho={L['spearman']['mean']:.4f}±{L['spearman']['std']:.4f}  "
          f"rmse={L['rmse']['mean']:.4f}±{L['rmse']['std']:.4f}")
    print(f"[CASF-2016] nontrain(n={NT['n']}) "
          f"r={NT['pearson']['mean']:.4f}±{NT['pearson']['std']:.4f}  "
          f"rho={NT['spearman']['mean']:.4f}±{NT['spearman']['std']:.4f}  "
          f"rmse={NT['rmse']['mean']:.4f}±{NT['rmse']['std']:.4f}")


if __name__ == "__main__":
    main()
