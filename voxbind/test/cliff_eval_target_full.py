#!/usr/bin/env python3
"""Headline cliff table on the LARGE UniProt-target-grouped cliff set (~113 test-test pairs) —
the statistically meaningful version (vs the n=7 MoleculeACE-density set). Reuses the exact
UniProt grouping + consensus-cliff detection of cliff_eval_target.py, and scores ALL four models
(C, C+D+G, TargetDiff, HBGSA) with seed-averaged predictions on the pairs each can cover.
  sign_acc ↑ (tt pairs) · rmse_cliff ↓ (cliff mols) · roughslope ↓ · sanity ρ (bulk).
Writes cliff_eval_target_full.json.
"""
import sys, types, json, importlib.util
from collections import defaultdict
import numpy as np, csv
from scipy.stats import spearmanr
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

REPO = "/home/shpark/prj-denovo/VoxBind"; sys.path.insert(0, REPO)
_s = importlib.util.spec_from_file_location("p01c", f"{REPO}/voxbind/dataset/01c_pdbbind_probe.py")
pr = importlib.util.module_from_spec(_s); _s.loader.exec_module(pr)
def _lev(a, b):
    if a == b: return 0
    if not a: return len(b)
    if not b: return len(a)
    prev = list(range(len(b)+1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0]*len(b)
        for j, cb in enumerate(b, 1): cur[j] = min(prev[j]+1, cur[j-1]+1, prev[j-1]+(ca != cb))
        prev = cur
    return prev[len(b)]
sys.modules["Levenshtein"] = types.ModuleType("Levenshtein"); sys.modules["Levenshtein"].distance = _lev
sys.path.insert(0, f"{REPO}/notebook"); import moleculeace_cliffs as mace
mace.tqdm = lambda it, **k: it

# ── predictions: C, C+D+G (canonical, seed-avg) + TargetDiff + HBGSA (seed-avg) ──
CE = json.load(open(f"{REPO}/voxbind/dataset/data/pdbbind/cliff_eval_canonical.json"))
test_pids = CE["test_pids"]; SEEDS = CE["seeds"]
pK = {p: float(v) for p, v in CE["pK"].items()}
P = {m: {p: float(np.mean(CE["preds"][m][p])) for p in test_pids} for m in CE["models"]}
P["TargetDiff / EGNN"] = {r["pid"]: float(r["y_pred"])
                          for r in csv.DictReader(open(f"{REPO}/notebook/html/260625/scatter_egnn_targetdiff.csv"))}
hb = [json.load(open(f"{REPO}/base/hbgsa/results/preds_edrscc_40m_seed{s}.json")) for s in range(3)]
hbm = [dict(zip(h["pdb_id"], h["pred"])) for h in hb]
P["HBGSA"] = {p: float(np.mean([m[p] for m in hbm if p in m]))
              for p in set().union(*[set(m) for m in hbm])}
MODELS = ["C (ViT)", "C+D+G (ChannelViT)", "TargetDiff / EGNN", "HBGSA"]

# ── UniProt-grouped cliff detection (identical to cliff_eval_target.detect) ──
prim = json.load(open("/tmp/pdbmap/prim_uniprot.json"))
lp = pr.load_lp_index(pr.LP_CSV); meta = {r["pdb_id"]: r for r in lp.to_dict("records")}
sm, scheme = pr.load_frozen_split_map("lp_edrscc_v2")
import torch
feC = set(torch.load(pr.FEAT_DIR / "atomblob_e99_v5_260622_plinder_otf_coords_mask050_pretrain.pt", weights_only=False)["features"])
feD = set(torch.load(pr.FEAT_DIR / "atomblob_density_gradmag_e99_v5_260623_ar_cvit_c1_g742.pt", weights_only=False)["features"])
rows = [{"pid": p, "smiles": str(meta[p]["smiles"]),
         "uni": prim.get(p, "seq:" + str(meta[p]["seq"])), "pK": float(meta[p]["pK"]), "split": sm.get(p)}
        for p in (feC & feD)
        if p in meta and p in sm and np.isfinite(meta[p]["pK"]) and Chem.MolFromSmiles(str(meta[p]["smiles"]))]
groups = defaultdict(list)
for r in rows: groups[r["uni"]].append(r)
pairs, rough = [], {p: 0.0 for p in test_pids}
for v in groups.values():
    if len(v) < 2: continue
    S = mace.moleculeace_similarity([x["smiles"] for x in v], 0.9, hide=True)
    for i in range(len(v)):
        for j in range(i+1, len(v)):
            if S[i][j] and abs(v[i]["pK"] - v[j]["pK"]) > 1.0: pairs.append((v[i], v[j]))
        if v[i]["split"] == "test":
            nb = [abs(v[i]["pK"] - v[j]["pK"]) for j in range(len(v)) if j != i and S[i][j]]
            if nb: rough[v[i]["pid"]] = max(nb)
tt = [(a, b) for a, b in pairs if a["split"] == "test" and b["split"] == "test"]
test_cliff = sorted({x["pid"] for p in pairs for x in p if x["split"] == "test"})
has_nb = [p for p in test_pids if rough[p] > 0]
print(f"UniProt grouping: {len(pairs)} pairs | {len(test_cliff)} test-cliff mols | {len(tt)} test-test pairs\n")

# ── metrics per model on covered subset ─────────────────────────────────
res = {"scheme": scheme, "grouping": "uni", "n_pairs": len(pairs), "n_test_cliff": len(test_cliff),
       "n_tt": len(tt), "models": {}}
for m in MODELS:
    d = P[m]
    ttc = [(a, b) for a, b in tt if a["pid"] in d and b["pid"] in d]
    tcc = [p for p in test_cliff if p in d]
    nbc = [p for p in has_nb if p in d]
    tsc = [p for p in test_pids if p in d]
    sa = np.mean([np.sign(d[a["pid"]]-d[b["pid"]]) == np.sign(a["pK"]-b["pK"]) for a, b in ttc])
    rc = np.sqrt(np.mean([(d[p]-pK[p])**2 for p in tcc]))
    rho = float(spearmanr([d[p] for p in tsc], [pK[p] for p in tsc]).statistic)
    err = np.array([abs(d[p]-pK[p]) for p in nbc]); rg = np.array([rough[p] for p in nbc])
    rs = float(spearmanr(err, rg).statistic) if len(nbc) > 2 else float("nan")
    res["models"][m] = dict(sanity_rho=rho, sign_acc=float(sa), rmse_cliff=float(rc),
                            roughslope=rs, n_tt=len(ttc), n_cliff=len(tcc))

json.dump(res, open(f"{REPO}/voxbind/dataset/data/pdbbind/cliff_eval_target_full.json", "w"), indent=1)
print(f"{'model':22s}{'sanity ρ':>10}{'sign-acc':>10}{'RMSE_cliff':>12}{'roughslope':>12}   n_tt")
print("-"*78)
for m, r in res["models"].items():
    print(f"{m:22s}{r['sanity_rho']:>10.3f}{r['sign_acc']:>10.3f}{r['rmse_cliff']:>12.3f}"
          f"{r['roughslope']:>12.3f}   {r['n_tt']}/{len(tt)}")
