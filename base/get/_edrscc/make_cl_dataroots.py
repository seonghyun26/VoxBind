"""make_cl_dataroots.py — Subset the built lp_edrscc_v2 GET .pkl files to each CL split.

The lp_edrscc_v2_cl{1,12,123} splits are exact subsets of lp_edrscc_v2 with the same
per-pid split assignment and identical pocket structures — no need to re-run make_data.py.
Each CL variant gets its own datasets/edrscc_cl{1,12,123}/ directory with
{train,valid,test}.pkl lists subsetted from the already-built edrscc/ blocks.

Usage (from base/get/):
    /home/shpark/.conda/envs/get/bin/python _edrscc/make_cl_dataroots.py
"""
import os
import pickle

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
GET_ROOT = os.path.dirname(HERE)               # base/get/
REPO = os.path.dirname(os.path.dirname(GET_ROOT))   # VoxBind/
SPLITS_DIR = os.path.join(REPO, "voxbind", "splits")

V2_DIR = os.path.join(GET_ROOT, "datasets", "edrscc")
VARIANTS = ["lp_edrscc_v2_cl1", "lp_edrscc_v2_cl12", "lp_edrscc_v2_cl123"]
# voxbind split column value -> pkl bucket name
BUCKET_MAP = {"train": "train", "val": "valid", "test": "test"}


def main():
    # Load v2 built blocks, keyed by pid per bucket
    print("Loading lp_edrscc_v2 built pkl files ...")
    v2 = {}
    for bucket in ("train", "valid", "test"):
        path = os.path.join(V2_DIR, f"{bucket}.pkl")
        items = pickle.load(open(path, "rb"))
        v2[bucket] = {item["id"].lower(): item for item in items}
        print(f"  v2/{bucket}: {len(v2[bucket])} built")

    for split in VARIANTS:
        cl_suffix = split.replace("lp_edrscc_v2_", "")  # "cl1", "cl12", "cl123"
        out_dir = os.path.join(GET_ROOT, "datasets", f"edrscc_{cl_suffix}")
        os.makedirs(out_dir, exist_ok=True)

        df = pd.read_csv(os.path.join(SPLITS_DIR, f"{split}.csv"))
        df["pid"] = df["pid"].astype(str).str.lower()
        # map voxbind split names -> bucket names
        df["bucket"] = df["split"].map(BUCKET_MAP)

        counts, dropped = {}, {}
        for bucket in ("train", "valid", "test"):
            want_pids = df.loc[df["bucket"] == bucket, "pid"].tolist()
            items = [v2[bucket][p] for p in want_pids if p in v2[bucket]]
            n_dropped = len(want_pids) - len(items)
            counts[bucket] = len(items)
            dropped[bucket] = n_dropped
            out_path = os.path.join(out_dir, f"{bucket}.pkl")
            with open(out_path, "wb") as f:
                pickle.dump(items, f)

        print(
            f"{split:30s}  "
            f"train={counts['train']:4d}  "
            f"valid={counts['valid']:4d}  "
            f"test={counts['test']:4d}  "
            f"dropped(not-built)={dropped}"
        )


if __name__ == "__main__":
    main()
