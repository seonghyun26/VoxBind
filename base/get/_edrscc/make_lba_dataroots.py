"""make_lba_dataroots.py — re-bucket the built lp_edrscc_v2 GET .pkl blocks into the
LBA-style protein-identity splits (lba30/lba60). Those splits re-partition the SAME
complex pool as lp_edrscc_v2, so every pid is already built in edrscc/{train,valid,test}.pkl
— we only reassign each item to its new bucket. (Adapted from make_cl_dataroots.py.)

Usage (from base/get/):
    /home/shpark/.conda/envs/get/bin/python _edrscc/make_lba_dataroots.py
"""
import os
import pickle

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
GET_ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(GET_ROOT))
SPLITS_DIR = os.path.join(REPO, "voxbind", "splits")
V2_DIR = os.path.join(GET_ROOT, "datasets", "edrscc")
VARIANTS = ["lba30", "lba60"]
BUCKET_MAP = {"train": "train", "val": "valid", "test": "test"}


def main():
    print("Loading lp_edrscc_v2 built pkl blocks (pooled across buckets) ...")
    pool = {}
    for bucket in ("train", "valid", "test"):
        items = pickle.load(open(os.path.join(V2_DIR, f"{bucket}.pkl"), "rb"))
        for item in items:
            pool[item["id"].lower()] = item
    print(f"  pooled {len(pool)} built complexes")

    for split in VARIANTS:
        out_dir = os.path.join(GET_ROOT, "datasets", f"edrscc_{split}")
        os.makedirs(out_dir, exist_ok=True)
        df = pd.read_csv(os.path.join(SPLITS_DIR, f"{split}_edrscc.csv"))
        df["pid"] = df["pid"].astype(str).str.lower()
        df["bucket"] = df["split"].map(BUCKET_MAP)
        counts, dropped = {}, {}
        for bucket in ("train", "valid", "test"):
            want = df.loc[df["bucket"] == bucket, "pid"].tolist()
            items = [pool[p] for p in want if p in pool]
            counts[bucket] = len(items)
            dropped[bucket] = len(want) - len(items)
            with open(os.path.join(out_dir, f"{bucket}.pkl"), "wb") as f:
                pickle.dump(items, f)
        print(f"{split:8s}  train={counts['train']:4d}  valid={counts['valid']:4d}  "
              f"test={counts['test']:4d}  dropped(not-built)={dropped}")


if __name__ == "__main__":
    main()
