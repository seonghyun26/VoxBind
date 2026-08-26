"""gen_cdg_mse.py — MSE-only (no Pearson aux) regen of CDG v2 & CDG v3 for Table 1b/1c.
  (1) re-dump cl123-test preds -> preds_ours/CDG_v{2,3}_cl123_seed*.csv   [Table 1b, overwrites mse+corr]
  (2) recompute CASF cohorts (leaky/nontrain/clean/id60/id30) -> MERGE into casf_table1c_ours_5seed.json
mse loss, 5-seed, same probe_casf.train_predict machinery."""
import csv, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from probe_casf_100m_mask075 import (
    FD, OUT, REPO, build_loss, casf_eval, load_pK, metrics, train_predict, v2_split)
import torch; torch.set_num_threads(2)

LOSS = build_loss("mse", 0.0)          # MSE only
SEEDS = 5
PREDS = f"{REPO}/base/_casf/cl123_seqfilter_5seed_260818/preds_ours"
CASF_JSON = f"{OUT}/casf_table1c_ours_5seed.json"
ENC = {  # label -> feature bundle
    "CDG v2": "atomblob_density_gradmag_e25_v5_260806_cdg_100m_v2_ep100_e25.pt",
    "CDG v3": "atomblob_density_gradmag_e20_v5_260823_cdg_100m_v2_interface_curriculum_0609.pt",
}
SAFE = {"CDG v2": "CDG_v2", "CDG v3": "CDG_v3"}


def load_feats(bn):
    d = torch.load(os.path.join(FD, bn), weights_only=False); f = d.get("features", d.get("feat"))
    return {p.lower(): np.asarray(v.numpy() if hasattr(v, "numpy") else v, dtype=np.float32) for p, v in f.items()}
def ids(p): return {x.strip().lower() for x in open(p).read().replace(",", " ").split() if x.strip()}
def cl123_split(): return {r["pid"].lower(): r["split"] for r in csv.DictReader(open(f"{REPO}/voxbind/splits/lp_edrscc_v2_cl123.csv"))}

pK, v2 = load_pK(), v2_split()
cp, nt = casf_eval(); clean = {p for p in cp if v2.get(p) not in ("train", "val")}
id60 = ids(f"{OUT}/casf_similarity/casf_clean_cl3_novel60.txt")
id30 = ids(f"{OUT}/casf_similarity/casf_clean_cl3_novel30.txt")
cohorts = {"leaky": set(cp), "nontrain": nt, "clean": clean, "id60": id60, "id30": id30}
cl = cl123_split(); cl_test = [p for p, s in cl.items() if s == "test"]
j = json.load(open(CASF_JSON)) if os.path.exists(CASF_JSON) else {}

for label, bn in ENC.items():
    feats = load_feats(bn)
    # (1) cl123 preds (mse)
    for s in range(SEEDS):
        te, y, p = train_predict(feats, pK, cl, cl_test, s, loss_fn=LOSS)
        with open(f"{PREDS}/{SAFE[label]}_cl123_seed{s}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["pid", "pred", "y"]); w.writerows((q, float(pr), float(yy)) for q, pr, yy in zip(te, p, y))
    # (2) CASF cohorts (mse)
    per = {c: [] for c in cohorts}
    for s in range(SEEDS):
        te, y, p = train_predict(feats, pK, v2, cp, s, loss_fn=LOSS)
        for c, keep in cohorts.items():
            m = np.array([q in keep for q in te])
            if m.sum() >= 3: per[c].append(metrics(y, p, mask=m))
    entry = {}
    for c, lst in per.items():
        if lst:
            entry[c] = {k: [float(np.mean([d[k] for d in lst])), float(np.std([d[k] for d in lst]))] for k in ("pearson", "spearman", "rmse")}
            entry[c]["n"] = lst[0]["n"]
    j[label] = entry
    print(f"[{label}] MSE: cl123 preds re-dumped; CASF clean ρ={entry['clean']['spearman'][0]:.3f} leaky ρ={entry['leaky']['spearman'][0]:.3f}", flush=True)

json.dump(j, open(CASF_JSON, "w"), indent=2)
print(f"wrote {CASF_JSON}", flush=True)
