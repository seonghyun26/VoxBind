"""build_dta_splits.py — build (pid,seq,smiles,pK) train/val/test CSVs for DTA baselines
on the CL / ATOM3D-LBA / CleanSplit cohorts, by re-bucketing LP_PDBBind per each split CSV.
Writes base/dta/data/{split}_{train,val,test}.csv. v2 already built by make_data.py.
"""
import os, csv
import pandas as pd

REPO = "/home/shpark/prj-denovo/VoxBind"
SPL = f"{REPO}/voxbind/splits"
DATA = os.path.dirname(os.path.abspath(__file__)) + "/data"
LP = f"{REPO}/voxbind/dataset/data/pdbbind/raw/LP_PDBBind.csv"

lp = pd.read_csv(LP).rename(columns={"Unnamed: 0": "pid"})
lp["pid"] = lp["pid"].astype(str).str.lower()
info = {r.pid: (r.seq, r.smiles, r.value) for r in lp.itertuples()
        if isinstance(getattr(r, "seq", None), str) and isinstance(getattr(r, "smiles", None), str)
        and pd.notna(r.value)}

# split -> {bucket: source_csv#bucket}  (train/val/test each pull from a split CSV's bucket)
SPLITS = {
    "atom3d_lba60_edrscc_v2_v22clean": ("atom3d_lba60_edrscc_v2_v22clean", "atom3d_lba60_edrscc_v2_v22clean", "atom3d_lba60_edrscc_v2_v22clean"),
    "atom3d_lba30_edrscc_v2_v22clean": ("atom3d_lba30_edrscc_v2_v22clean", "atom3d_lba30_edrscc_v2_v22clean", "atom3d_lba30_edrscc_v2_v22clean"),
    "clean_ed_v1_indep": ("clean_ed_v1", "clean_ed_v1", "clean_ed_v1_indep"),   # train/val from full CleanSplit, test = indep-109
    "lp_edrscc_v2_cl1": ("lp_edrscc_v2_cl1",)*3,
    "lp_edrscc_v2_cl12": ("lp_edrscc_v2_cl12",)*3,
    "lp_edrscc_v2_cl123": ("lp_edrscc_v2_cl123",)*3,
}
BMAP = {"train": "train", "val": "val", "valid": "val", "test": "test"}

def pids_of(split_csv, bucket):
    df = pd.read_csv(f"{SPL}/{split_csv}.csv")
    return [str(p).lower() for p, s in zip(df["pid"], df["split"]) if BMAP.get(s) == bucket]

for name, (tr_csv, va_csv, te_csv) in SPLITS.items():
    for bucket, src in (("train", tr_csv), ("val", va_csv), ("test", te_csv)):
        rows = [(p, *info[p]) for p in pids_of(src, bucket) if p in info]
        out = f"{DATA}/{name}_{bucket}.csv"
        with open(out, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["pid", "seq", "smiles", "pK"]); w.writerows(rows)
    n = {b: sum(1 for _ in open(f"{DATA}/{name}_{b}.csv")) - 1 for b in ("train", "val", "test")}
    print(f"{name}: {n}")
print("done")
