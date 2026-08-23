"""casf_ours_best.py — compute the 5-seed CASF cohorts for the new headline
"Ours (best)" = v2_ep100_e25 (260806_cdg_100m_v2_ep100_e25) + mse+corr probe head, and MERGE
it into base/_casf/casf_table1c_ours_5seed.json (preserving the existing C/C+D+G/+corr keys).
Reuses the exact casf_table1c_ours machinery (train_predict on lp_edrscc_v2 train, mask cohorts)."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from casf_table1c_ours import load_feats_basename, load_ids  # noqa: E402
from probe_casf_100m_mask075 import (  # noqa: E402
    OUT, build_loss, casf_eval, load_pK, metrics, train_predict, v2_split)
import torch; torch.set_num_threads(4)  # noqa: E402

LABEL = "Ours (best)"
BASENAME = "atomblob_density_gradmag_e25_v5_260806_cdg_100m_v2_ep100_e25.pt"
LOSS = "mse+corr"
SEEDS = 5


def main():
    pK, v2 = load_pK(), v2_split()
    casf_pids, nontrain = casf_eval()
    clean_set = {p for p in casf_pids if v2.get(p) not in ("train", "val")}
    id60 = load_ids(f"{OUT}/casf_similarity/casf_clean_cl3_novel60.txt")
    id30 = load_ids(f"{OUT}/casf_similarity/casf_clean_cl3_novel30.txt")
    cohorts = {"leaky": set(casf_pids), "nontrain": nontrain, "clean": clean_set,
               "id60": id60, "id30": id30}
    loss_fn = build_loss(LOSS, 5.0)
    feats = load_feats_basename(BASENAME)
    per = {c: [] for c in cohorts}
    for s in range(SEEDS):
        te_pids, yte, pte = train_predict(feats, pK, v2, casf_pids, s, loss_fn=loss_fn)
        for c, ids in cohorts.items():
            mask = np.array([p in ids for p in te_pids])
            if mask.sum() >= 3:
                per[c].append(metrics(yte, pte, mask=mask))
    out = {}
    for c, lst in per.items():
        if not lst:
            continue
        out[c] = {k: [float(np.mean([d[k] for d in lst])), float(np.std([d[k] for d in lst]))]
                  for k in ("pearson", "spearman", "rmse")}
        out[c]["n"] = lst[0]["n"]

    path = f"{OUT}/casf_table1c_ours_5seed.json"
    d = json.load(open(path))
    d[LABEL] = out
    json.dump(d, open(path, "w"), indent=2)
    line = "  ".join(f"{c} ρ={out[c]['spearman'][0]:.3f}(n{out[c]['n']})" for c in cohorts if c in out)
    print(f"[{LABEL}] merged → {path}\n  {line}")


if __name__ == "__main__":
    main()
