"""holdout_eval.py — DSMBind zero-shot energy on the 2019 temporal holdout.

Same machinery as run_eval.py (pocket.pdb target + ESM-2-3B + pretrained drug all-atom energy),
but scores every holdout complex (voxbind/splits/holdout2019_eval.csv) instead of the lp test split.
score = -binding_energy; dumped per-pid so common94_holdout.py can rank it against pK.

Output: base/_casf/DSMBind_holdout2019_preds.csv  (pid, pred, y)
Usage:  cd base/dsmbind && CUDA_VISIBLE_DEVICES=<g> python _edrscc/src/holdout_eval.py --gpu 0
"""
import argparse, csv, os, sys
import numpy as np
import pandas as pd
import scipy.stats
import torch
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_eval import (build_entry, CKPT, REPO, DSM_ROOT,      # noqa: E402
                      DrugAllAtomEnergyModel, DrugDataset, load_esm_embedding)

HOLDOUT = os.path.join(REPO, "voxbind", "splits", "holdout2019_eval.csv")
OUT_CSV = os.path.join(os.path.dirname(os.path.dirname(DSM_ROOT)), "_casf", "DSMBind_holdout2019_preds.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    torch.cuda.set_device(a.gpu)

    ho = pd.read_csv(HOLDOUT); ho["pid"] = ho["pid"].astype(str).str.lower()
    pk = dict(zip(ho["pid"], ho["pK"]))
    pids = ho["pid"].tolist()
    if a.limit:
        pids = pids[:a.limit]
    print(f"holdout pids: {len(pids)}", flush=True)

    print(f"loading DSMBind drug model from {CKPT}", flush=True)
    model_ckpt, _, model_args = torch.load(CKPT, map_location="cpu")
    model = DrugAllAtomEnergyModel(model_args).cuda()
    model.load_state_dict(model_ckpt); model.eval()

    entries, fails = [], {}
    for pid in tqdm(pids, desc="build entries"):
        if pid not in pk or not np.isfinite(pk[pid]):
            fails[pid] = "no_label"; continue
        e, status = build_entry(pid, pk[pid])
        if e is None:
            fails[pid] = status
        else:
            entries.append(e)
    print(f"built {len(entries)} | fails {len(fails)}: "
          f"{dict(pd.Series(list(fails.values())).value_counts()) if fails else {}}", flush=True)

    print("computing ESM-2-3B pocket embeddings...", flush=True)
    embedding = load_esm_embedding(entries, ["target_seq"])
    proc = DrugDataset.process(entries, model_args.patch_size)

    rows = []
    with torch.no_grad():
        for entry in tqdm(proc, desc="predict energy"):
            try:
                binder, target = DrugDataset.make_bind_batch([entry], embedding, model_args)
                energy = model.predict(binder, target).item()
            except Exception as e:
                fails[entry["pdb"]] = f"predict:{type(e).__name__}"; continue
            rows.append((entry["pdb"], float(entry["affinity"]), -energy))   # pred = -energy

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["pid", "pred", "y"])
        for pid, y, pred in rows:
            w.writerow([pid, pred, y])
    df = pd.DataFrame(rows, columns=["pid", "y", "pred"])
    r = float(np.corrcoef(df["y"], df["pred"])[0, 1])
    rho = float(scipy.stats.spearmanr(df["y"], df["pred"])[0])
    print(f"\nwrote {OUT_CSV}  n={len(df)}  r={r:.4f} rho={rho:.4f}", flush=True)
    print("NOTE: zero-shot energy — correlation only (energy not in pK units); size-confounded per memory.")


if __name__ == "__main__":
    main()
