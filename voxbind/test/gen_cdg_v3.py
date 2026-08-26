"""gen_cdg_v3.py — CDG v3 (interface+curriculum, e20) artifacts for results.html.
  (1) CL3 per-pid preds (cl123 train->test)  -> preds_ours/CDG_v3_cl123_seed{0-4}.csv   [Table 1b]
  (2) CASF cohorts (leaky/nontrain/clean/id60/id30) 5-seed r/rho/rmse -> MERGE 'CDG v3'
      into casf_table1c_ours_5seed.json (preserving existing entries)                    [Table 1c]
mse+corr, 5-seed, SAME probe_casf.train_predict machinery as the other 'ours' rows.
"""
import csv, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from probe_casf_100m_mask075 import (  # noqa: E402
    FD, OUT, REPO, build_loss, casf_eval, load_pK, metrics, train_predict, v2_split)
import torch  # noqa: E402
torch.set_num_threads(4)

FEAT = "atomblob_density_gradmag_e20_v5_260823_cdg_100m_v2_interface_curriculum_0609.pt"
LABEL = "CDG v3"
SEEDS = 5
LOSS = build_loss("mse+corr", 5.0)
PREDS_OUT = f"{REPO}/base/_casf/cl123_seqfilter_5seed_260818/preds_ours"
CASF_JSON = f"{OUT}/casf_table1c_ours_5seed.json"


def load_feats(bn):
    d = torch.load(os.path.join(FD, bn), weights_only=False)
    f = d.get("features", d.get("feat"))
    return {p.lower(): np.asarray(v.numpy() if hasattr(v, "numpy") else v, dtype=np.float32)
            for p, v in f.items()}


def load_ids(path):
    raw = open(path).read().replace(",", " ").split()
    return {x.strip().lower() for x in raw if x.strip()}


def cl123_split():
    return {r["pid"].lower(): r["split"]
            for r in csv.DictReader(open(f"{REPO}/voxbind/splits/lp_edrscc_v2_cl123.csv"))}


pK, v2 = load_pK(), v2_split()
feats = load_feats(FEAT)

# (1) CL3 per-pid preds
cl = cl123_split(); cl_test = [p for p, s in cl.items() if s == "test"]
for s in range(SEEDS):
    te, yte, pte = train_predict(feats, pK, cl, cl_test, s, loss_fn=LOSS)
    with open(f"{PREDS_OUT}/CDG_v3_cl123_seed{s}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["pid", "pred", "y"])
        w.writerows((p, float(pr), float(y)) for p, pr, y in zip(te, pte, yte))
print(f"(1) dumped CL3 preds -> {PREDS_OUT}/CDG_v3_cl123_seed*.csv (n_test={len(cl_test)})", flush=True)

# (2) CASF cohorts -> merge into casf_table1c_ours_5seed.json
casf_pids, nontrain = casf_eval()
clean_set = {p for p in casf_pids if v2.get(p) not in ("train", "val")}
id60 = load_ids(f"{OUT}/casf_similarity/casf_clean_cl3_novel60.txt")
id30 = load_ids(f"{OUT}/casf_similarity/casf_clean_cl3_novel30.txt")
cohorts = {"leaky": set(casf_pids), "nontrain": nontrain, "clean": clean_set, "id60": id60, "id30": id30}
per = {c: [] for c in cohorts}
for s in range(SEEDS):
    te, yte, pte = train_predict(feats, pK, v2, casf_pids, s, loss_fn=LOSS)
    for c, ids in cohorts.items():
        mask = np.array([p in ids for p in te])
        if mask.sum() >= 3:
            per[c].append(metrics(yte, pte, mask=mask))
entry = {}
for c, lst in per.items():
    if not lst:
        continue
    entry[c] = {k: [float(np.mean([d[k] for d in lst])), float(np.std([d[k] for d in lst]))]
                for k in ("pearson", "spearman", "rmse")}
    entry[c]["n"] = lst[0]["n"]
j = json.load(open(CASF_JSON)) if os.path.exists(CASF_JSON) else {}
j[LABEL] = entry
json.dump(j, open(CASF_JSON, "w"), indent=2)
print(f"(2) merged '{LABEL}' -> {CASF_JSON}: "
      f"leaky ρ={entry['leaky']['spearman'][0]:.3f} clean ρ={entry['clean']['spearman'][0]:.3f}", flush=True)
