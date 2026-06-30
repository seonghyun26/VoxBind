"""Target-level (UniProt) cliff analysis — PRIMARY — vs exact-sequence grouping.

Motivation (260625): our cliff detector grouped analogs by EXACT protein `seq`, so
construct variants of the same target landed in different groups and real cliffs were
missed (e.g. Factor X 2j95/2j94, same UniProt P00742, span 5 exact-seq groups). The
expert MoleculeACE benchmark groups by TARGET. We adopt that: group the pool by SIFTS
UniProt accession (primary canonical grouping) and re-detect consensus cliffs
(Tanimoto/scaffold/Levenshtein >=0.9 AND |dpK|>1), REUSING the cached C(ViT) /
C+D+G(ChannelViT) predictions from cliff_eval_canonical.json (same encoders, split,
seeds) so only the cliff SET changes. Emits a comprehensive artifact for the report:
both groupings' summary metrics + the full per-pair detail for the PRIMARY (UniProt)
grouping (examples, test-test pairs, Design-B roughness). CPU only.

Inputs : cliff_eval_canonical.json (cached preds), /tmp/pdbmap/prim_uniprot.json (pid->UniProt).
Output : cliff_eval_target.json."""
import sys, types, json, importlib.util
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr, wilcoxon
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

# ── cached predictions (no retraining) ───────────────────────────────────────
CE = json.load(open(f"{REPO}/voxbind/dataset/data/pdbbind/cliff_eval_canonical.json"))
MODELS = CE["models"]; SEEDS = CE["seeds"]; test_pids = CE["test_pids"]
pK = {p: float(v) for p, v in CE["pK"].items()}
preds = {m: [np.array([CE["preds"][m][p][s] for p in test_pids]) for s in range(len(SEEDS))] for m in MODELS}
pred_avg = {m: np.stack(preds[m]).mean(0) for m in MODELS}
pidpred = {m: dict(zip(test_pids, pred_avg[m])) for m in MODELS}

# ── pool with both group keys ────────────────────────────────────────────────
prim = json.load(open("/tmp/pdbmap/prim_uniprot.json"))   # pid -> UniProt
lp = pr.load_lp_index(pr.LP_CSV); meta = {r["pdb_id"]: r for r in lp.to_dict("records")}
sm, scheme = pr.load_frozen_split_map("lp_edrscc_v2")
import torch
feC = set(torch.load(pr.FEAT_DIR / "atomblob_e99_v5_260622_plinder_otf_coords_mask050_pretrain.pt", weights_only=False)["features"])
feD = set(torch.load(pr.FEAT_DIR / "atomblob_density_gradmag_e99_v5_260623_ar_cvit_c1_g742.pt", weights_only=False)["features"])
shared = feC & feD
rows = [{"pid": p, "smiles": str(meta[p]["smiles"]), "seq": str(meta[p]["seq"]),
         "uni": prim.get(p, "seq:" + str(meta[p]["seq"])), "pK": float(meta[p]["pK"]),
         "split": sm.get(p)}
        for p in shared
        if p in meta and p in sm and np.isfinite(meta[p]["pK"]) and Chem.MolFromSmiles(str(meta[p]["smiles"]))]
print(f"split={scheme} pool={len(rows)} | UniProt-mapped={sum(not r['uni'].startswith('seq:') for r in rows)}")

def detect(group_key):
    """Return (pairs, roughness) for a grouping. pairs = list of (rowA,rowB);
    roughness[pid] = max |dpK| to a same-group consensus-similar neighbour (Design B)."""
    groups = defaultdict(list)
    for r in rows: groups[r[group_key]].append(r)
    pairs, rough = [], {p: 0.0 for p in test_pids}
    for v in groups.values():
        if len(v) < 2: continue
        S = mace.moleculeace_similarity([x["smiles"] for x in v], 0.9, hide=True)
        for i in range(len(v)):
            for j in range(i+1, len(v)):
                if S[i][j] and abs(v[i]["pK"] - v[j]["pK"]) > 1.0:
                    pairs.append((v[i], v[j]))
            if v[i]["split"] == "test":
                nb = [abs(v[i]["pK"] - v[j]["pK"]) for j in range(len(v)) if j != i and S[i][j]]
                if nb: rough[v[i]["pid"]] = max(nb)
    return pairs, rough

def rmse_on(pids, m, seed=None):
    arr = preds[m][seed] if seed is not None else pred_avg[m]
    d = dict(zip(test_pids, arr)); e = np.array([d[p] - pK[p] for p in pids])
    return np.sqrt((e ** 2).mean()) if len(pids) else float("nan")

def summarize(pairs):
    test_cliff = sorted({x["pid"] for p in pairs for x in p if x["split"] == "test"})
    tt = [(a, b) for a, b in pairs if a["split"] == "test" and b["split"] == "test"]
    noncliff = [p for p in test_pids if p not in set(test_cliff)]
    mt = {}
    for m in MODELS:
        rc = [rmse_on(test_cliff, m, s) for s in range(len(SEEDS))]
        rn = [rmse_on(noncliff, m, s) for s in range(len(SEEDS))]
        sa = []
        for s in range(len(SEEDS)):
            d = dict(zip(test_pids, preds[m][s]))
            sa.append(np.mean([np.sign(d[a["pid"]] - d[b["pid"]]) == np.sign(a["pK"] - b["pK"]) for a, b in tt]) if tt else float("nan"))
        mt[m] = dict(rmse_cliff=(float(np.mean(rc)), float(np.std(rc))),
                     rmse_noncliff=(float(np.mean(rn)), float(np.std(rn))),
                     sign_acc=(float(np.nanmean(sa)), float(np.nanstd(sa))))
    A, B = MODELS
    eA = np.array([abs(pidpred[A][p] - pK[p]) for p in test_cliff])
    eB = np.array([abs(pidpred[B][p] - pK[p]) for p in test_cliff])
    wp = float(wilcoxon(eA, eB).pvalue) if len(test_cliff) > 1 else float("nan")
    return dict(n_pairs=len(pairs), n_test_cliff=len(test_cliff), n_tt=len(tt),
                n_noncliff=len(noncliff), metrics=mt, wilcoxon_p=wp,
                aerr={A: float(eA.mean()), B: float(eB.mean())},
                test_cliff=test_cliff, tt=[(a["pid"], b["pid"]) for a, b in tt])

groupings = {}
detail = {}
for gk, name in [("seq", "exact-sequence"), ("uni", "UniProt target")]:
    pairs, rough = detect(gk)
    s = summarize(pairs); s["name"] = name; groupings[gk] = s
    print(f"### {name}: pairs={s['n_pairs']} test_cliff={s['n_test_cliff']} tt={s['n_tt']} "
          f"sign-acc {MODELS[0]}={s['metrics'][MODELS[0]]['sign_acc'][0]:.2f} "
          f"{MODELS[1]}={s['metrics'][MODELS[1]]['sign_acc'][0]:.2f} wilcoxon={s['wilcoxon_p']:.2f}")
    if gk == "uni":      # rich detail for the PRIMARY grouping
        seen, examples = set(), []
        for a, b in sorted(pairs, key=lambda ab: -abs(ab[0]["pK"] - ab[1]["pK"])):
            hi, lo = (a, b) if a["pK"] >= b["pK"] else (b, a)
            kk = (hi["pid"], lo["pid"])
            if kk in seen: continue
            seen.add(kk)
            examples.append(dict(hi_pid=hi["pid"], lo_pid=lo["pid"], dpK=abs(a["pK"] - b["pK"]),
                                 hi_pK=hi["pK"], lo_pK=lo["pK"], hi_split=hi["split"], lo_split=lo["split"],
                                 hi_smiles=hi["smiles"], lo_smiles=lo["smiles"],
                                 hi_uni=hi["uni"], lo_uni=lo["uni"],
                                 cross_construct=(hi["seq"] != lo["seq"])))
        has_nb = [p for p in test_pids if rough[p] > 0]
        roughslope = {}
        for m in MODELS:
            err = np.array([abs(pidpred[m][p] - pK[p]) for p in has_nb]); rg = np.array([rough[p] for p in has_nb])
            roughslope[m] = float(spearmanr(err, rg).statistic) if len(has_nb) > 2 else float("nan")
        detail = dict(examples=examples, roughslope=roughslope, n_has_nb=len(has_nb))

out = dict(scheme=scheme, models=MODELS, seeds=SEEDS, n_test=len(test_pids),
           primary="uni", groupings=groupings, primary_detail=detail,
           # mirror predictions+pK so the report is buildable from this one file
           pK={p: pK[p] for p in test_pids},
           preds={m: {test_pids[i]: [float(preds[m][s][i]) for s in range(len(SEEDS))]
                      for i in range(len(test_pids))} for m in MODELS})
json.dump(out, open(f"{REPO}/voxbind/dataset/data/pdbbind/cliff_eval_target.json", "w"))
print("\nsaved -> voxbind/dataset/data/pdbbind/cliff_eval_target.json")
