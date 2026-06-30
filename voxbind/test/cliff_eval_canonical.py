"""Activity-cliff evaluation for the 260625 canonical conditions:
  C (ViT)            = atomblob_e99_v5_260622_plinder_otf_coords_mask050_pretrain
  C+D+G (ChannelViT) = atomblob_density_gradmag_e99_v5_260623_ar_cvit_c1_g742  ([7,4,2])
on the canonical FROZEN split lp_edrscc_v2 (Kd/Ki-only, ED+RSCC, 3850/817/1320).

Mirrors voxbind/test/cliff_eval.py (the 260618 PLINDER C/C+D+G/noise-D analysis) but
re-keyed onto Table-1's headline encoders + the frozen split. Replicates 01c probe's
train_one scalar path to capture per-pid test predictions, then runs the MoleculeACE
consensus-cliff protocol (sim>=0.9 AND |dpK|>1) within exact-sequence target groups.
CPU, frozen cached features (no GPU). Writes cliff_eval_canonical.json for the report."""
import sys, types, importlib.util
from collections import defaultdict
import numpy as np, torch
torch.set_num_threads(2)          # be a good neighbour to the GPU training's CPU side
from scipy.stats import spearmanr, pearsonr, wilcoxon
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

REPO = "/home/shpark/prj-denovo/VoxBind"; sys.path.insert(0, REPO)
_s = importlib.util.spec_from_file_location("p01c", f"{REPO}/voxbind/dataset/01c_pdbbind_probe.py")
pr = importlib.util.module_from_spec(_s); _s.loader.exec_module(pr)

def _lev(a, b):
    if a == b: return 0
    lb = len(b)
    if not a: return lb
    if not lb: return len(a)
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0]*lb
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j]+1, cur[j-1]+1, prev[j-1]+(ca != cb))
        prev = cur
    return prev[lb]
sys.modules["Levenshtein"] = types.ModuleType("Levenshtein"); sys.modules["Levenshtein"].distance = _lev
sys.path.insert(0, f"{REPO}/notebook"); import moleculeace_cliffs as mace
mace.tqdm = lambda it, **k: it

SEEDS = [0, 1, 2]
SPLIT_FLAG = "lp_edrscc_v2"
MODELS = {"C (ViT)":            "atomblob_e99_v5_260622_plinder_otf_coords_mask050_pretrain.pt",
          "C+D+G (ChannelViT)": "atomblob_density_gradmag_e99_v5_260623_ar_cvit_c1_g742.pt"}
feats = {k: torch.load(pr.FEAT_DIR / v, weights_only=False)["features"] for k, v in MODELS.items()}
shared = set.intersection(*[set(f) for f in feats.values()])
lp = pr.load_lp_index(pr.LP_CSV)
split_map, scheme = pr.load_frozen_split_map(SPLIT_FLAG)
print(f"split = {scheme} | shared pids across {' / '.join(MODELS)} = {len(shared)}")

def train_predict(data, seed):   # scalar head, mirrors train_one (CPU)
    torch.manual_seed(seed); np.random.seed(seed)
    g = lambda s: (torch.from_numpy(data[s]["X"]), torch.from_numpy(data[s]["y"]))
    Xtr, ytr = g("train"); Xva, yva = g("val"); Xte, _ = g("test")
    model = pr.MLP2(Xtr.shape[1], hidden=128, dropout=0.1, out_dim=1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4); mse = torch.nn.MSELoss()
    n = Xtr.shape[0]; best_val, best_state, since = -1e9, None, 0
    for _ in range(200):
        model.train(); perm = torch.randperm(n)
        for s in range(0, n, 64):
            idx = perm[s:s+64]; opt.zero_grad(); mse(model(Xtr[idx]), ytr[idx]).backward(); opt.step()
        model.eval()
        with torch.no_grad(): vs = spearmanr(model(Xva).numpy(), yva.numpy()).statistic
        if vs > best_val: best_val, best_state, since = vs, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            since += 1
            if since >= 30: break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad(): return model(Xte).numpy()

# per-model, per-seed test predictions on the frozen lp_edrscc_v2 split
data0 = pr.build_dataset({p: feats["C (ViT)"][p] for p in shared}, lp, drop_covalent=True,
                         cl1_only=False, split_map=split_map)
test_pids = data0["test"]["pid"]; pK = dict(zip(test_pids, data0["test"]["y"].astype(float)))
preds = {m: [] for m in MODELS}            # preds[m][seed] -> np.array aligned to test_pids
sanity = {}
for m in MODELS:
    data = pr.build_dataset({p: feats[m][p] for p in shared}, lp, drop_covalent=True,
                            cl1_only=False, split_map=split_map)
    assert data["test"]["pid"] == test_pids
    for sd in SEEDS:
        preds[m].append(train_predict(data, sd))
    P = np.stack(preds[m]); yte = data0["test"]["y"]
    rho  = [spearmanr(P[i], yte).statistic for i in range(len(SEEDS))]
    r    = [pearsonr(P[i], yte).statistic for i in range(len(SEEDS))]
    rmse = [np.sqrt(((P[i]-yte)**2).mean()) for i in range(len(SEEDS))]
    sanity[m] = {"rho": (float(np.mean(rho)), float(np.std(rho))),
                 "r":   (float(np.mean(r)),   float(np.std(r))),
                 "rmse":(float(np.mean(rmse)),float(np.std(rmse))), "n": len(test_pids)}
    print(f"  [{m:20s}] overall test rho={np.mean(rho):.3f}±{np.std(rho):.3f}  "
          f"r={np.mean(r):.3f}  RMSE={np.mean(rmse):.3f}  (Table-1 sanity, n={len(test_pids)})")

pred_avg = {m: np.stack(preds[m]).mean(0) for m in MODELS}     # seed-averaged per-pid
pidpred = {m: dict(zip(test_pids, pred_avg[m])) for m in MODELS}

# ── consensus cliffs within target (exact seq) on the frozen split ────────────
meta = {r["pdb_id"]: r for r in lp.to_dict("records")}
rows = [{"pid": p, "smiles": str(meta[p]["smiles"]), "seq": str(meta[p]["seq"]),
         "pK": float(meta[p]["pK"]), "split": split_map.get(p)}
        for p in shared
        if p in meta and p in split_map and split_map[p] in ("train", "val", "test")
        and np.isfinite(meta[p]["pK"]) and Chem.MolFromSmiles(str(meta[p]["smiles"]))]
groups = defaultdict(list)
for r in rows: groups[r["seq"]].append(r)
pairs = []
for v in groups.values():
    if len(v) < 2: continue
    try: S = mace.moleculeace_similarity([x["smiles"] for x in v], 0.9, hide=True)
    except Exception: continue
    for i in range(len(v)):
        for j in range(i+1, len(v)):
            if S[i][j] == 1 and abs(v[i]["pK"]-v[j]["pK"]) > 1.0:
                pairs.append((v[i], v[j]))
test_cliff = sorted({x["pid"] for p in pairs for x in p if x["split"] == "test"})
tt_pairs = [(a, b) for a, b in pairs if a["split"] == "test" and b["split"] == "test"]
noncliff = [p for p in test_pids if p not in set(test_cliff)]
print(f"\nconsensus cliffs: {len(pairs)} pairs | TEST cliff mols={len(test_cliff)} | "
      f"test-test pairs={len(tt_pairs)} | non-cliff test mols={len(noncliff)}")

def rmse_on(pids, m, seed=None):
    arr = preds[m][seed] if seed is not None else pred_avg[m]
    d = dict(zip(test_pids, arr))
    e = np.array([d[p]-pK[p] for p in pids]); return np.sqrt((e**2).mean())

print("\n" + "="*78)
print(f"{'model':20s}{'RMSE_cliff (n=%d)'%len(test_cliff):>22}{'RMSE_noncliff':>18}{'sign-acc':>14}")
print("-"*78)
summary = {}
for m in MODELS:
    rc = [rmse_on(test_cliff, m, s) for s in range(len(SEEDS))]
    rn = [rmse_on(noncliff, m, s) for s in range(len(SEEDS))]
    sa = []
    for s in range(len(SEEDS)):
        d = dict(zip(test_pids, preds[m][s]))
        sa.append(np.mean([np.sign(d[a["pid"]]-d[b["pid"]]) == np.sign(a["pK"]-b["pK"]) for a, b in tt_pairs]))
    summary[m] = {"rmse_cliff": (float(np.mean(rc)), float(np.std(rc))),
                  "rmse_noncliff": (float(np.mean(rn)), float(np.std(rn))),
                  "sign_acc": (float(np.mean(sa)), float(np.std(sa)))}
    print(f"{m:20s}{np.mean(rc):>13.3f}±{np.std(rc):.3f}{np.mean(rn):>11.3f}±{np.std(rn):.3f}"
          f"{np.mean(sa):>10.2f}±{np.std(sa):.2f}")
print("="*78)

# paired significance on the cliff mols (seed-averaged abs error): C  vs  C+D+G
def aerr(m, pids): d = pidpred[m]; return np.array([abs(d[p]-pK[p]) for p in pids])
A, B = "C (ViT)", "C+D+G (ChannelViT)"
eA, eB = aerr(A, test_cliff), aerr(B, test_cliff)
wp = float(wilcoxon(eA, eB).pvalue) if len(test_cliff) else float("nan")
print(f"paired cliff |err|: {A}={eA.mean():.3f} vs {B}={eB.mean():.3f}  "
      f"(median {np.median(eA):.3f} vs {np.median(eB):.3f})  wilcoxon p={wp:.3f}")

# Design B: per-test-mol local SAR roughness = max dpK to same-target consensus neighbour
rough = {p: 0.0 for p in test_pids}
for v in groups.values():
    if len(v) < 2: continue
    try: S = mace.moleculeace_similarity([x["smiles"] for x in v], 0.9, hide=True)
    except Exception: continue
    for i in range(len(v)):
        if v[i]["split"] != "test": continue
        nb = [abs(v[i]["pK"]-v[j]["pK"]) for j in range(len(v)) if j != i and S[i][j] == 1]
        if nb: rough[v[i]["pid"]] = max(nb)
has_nb = [p for p in test_pids if rough[p] > 0]
print(f"\nDesign-B: {len(has_nb)} test mols have a same-target similar analog")
roughslope = {}
for m in MODELS:
    d = pidpred[m]; err = np.array([abs(d[p]-pK[p]) for p in has_nb]); rgh = np.array([rough[p] for p in has_nb])
    sp = float(spearmanr(err, rgh).statistic) if len(has_nb) > 2 else float("nan")
    roughslope[m] = sp
    print(f"  [{m:20s}] spearman(|err|, roughness) = {sp:+.3f}")

# cliff-pair detail table (top by |dpK|) for the report
cliff_examples = []
seen_pair = set()
for a, b in sorted(pairs, key=lambda ab: -abs(ab[0]["pK"]-ab[1]["pK"])):
    hi, lo = (a, b) if a["pK"] >= b["pK"] else (b, a)
    kk = (hi["pid"], lo["pid"])
    if kk in seen_pair: continue
    seen_pair.add(kk)
    cliff_examples.append({
        "hi_pid": hi["pid"], "lo_pid": lo["pid"], "dpK": abs(a["pK"]-b["pK"]),
        "hi_pK": hi["pK"], "lo_pK": lo["pK"], "hi_split": hi["split"], "lo_split": lo["split"],
        "hi_smiles": hi["smiles"], "lo_smiles": lo["smiles"],
    })

# ── persist artifact for the report (no retraining on rebuild) ───────────────
import json
artifact = {
    "scheme": scheme, "models": list(MODELS), "seeds": SEEDS, "n_test": len(test_pids),
    "sanity": sanity, "summary": summary,
    "n_cliff_pairs": len(pairs), "n_test_cliff": len(test_cliff),
    "n_tt_pairs": len(tt_pairs), "n_noncliff": len(noncliff), "n_has_nb": len(has_nb),
    "wilcoxon_p_cliff": wp, "cliff_aerr_mean": {A: float(eA.mean()), B: float(eB.mean())},
    "roughslope": roughslope,
    "test_pids": test_pids, "pK": {p: float(pK[p]) for p in test_pids},
    "preds": {m: {test_pids[i]: [float(preds[m][s][i]) for s in range(len(SEEDS))]
                  for i in range(len(test_pids))} for m in MODELS},
    "cliff": {"test_cliff": list(test_cliff),
              "tt_pairs":  [[a["pid"], b["pid"]] for a, b in tt_pairs],
              "noncliff":  list(noncliff),
              "roughness": {p: float(rough[p]) for p in test_pids}},
    "cliff_examples": cliff_examples,
}
outp = f"{REPO}/voxbind/dataset/data/pdbbind/cliff_eval_canonical.json"
json.dump(artifact, open(outp, "w"))
print("\nsaved artifact ->", outp)
