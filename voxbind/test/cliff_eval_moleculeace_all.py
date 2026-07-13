#!/usr/bin/env python3
"""MoleculeACE-density cliff eval WITHOUT the out-of-sample restriction — score every pair a
model can reach, not just the 7 both-in-test pairs. For the frozen probes (C, C+D+G) we train
the MLP head on the lp_edrscc_v2 train split (3 seeds) and predict on ALL cliff molecules that
have cached features (19/25 → 11/17 pairs); molecules in our TRAIN/VAL split are therefore
IN-SAMPLE (leaky, optimistic) — flagged. Baselines (TargetDiff, HBGSA) only ship predictions
for the test split → still 7 pairs. Label = MoleculeACE expert pKi.
Writes cliff_eval_moleculeace_all.json.
"""
import sys, types, importlib.util, json, csv
import numpy as np, torch
torch.set_num_threads(2)
from scipy.stats import spearmanr

REPO = "/home/shpark/prj-denovo/VoxBind"; sys.path.insert(0, REPO)
_s = importlib.util.spec_from_file_location("p01c", f"{REPO}/voxbind/dataset/01c_pdbbind_probe.py")
pr = importlib.util.module_from_spec(_s); _s.loader.exec_module(pr)

SEEDS = [0, 1, 2]
MODELS = {"C (ViT)": "atomblob_e99_v5_260622_plinder_otf_coords_mask050_pretrain.pt",
          "C+D+G (ChannelViT)": "atomblob_density_gradmag_e99_v5_260623_ar_cvit_c1_g742.pt"}
feats = {k: torch.load(pr.FEAT_DIR / v, weights_only=False)["features"] for k, v in MODELS.items()}
lp = pr.load_lp_index(pr.LP_CSV)
split_map, scheme = pr.load_frozen_split_map("lp_edrscc_v2")
TEST = {p for p, s in split_map.items() if s == "test"}

# ── MoleculeACE cliff pairs + expert pKi ────────────────────────────────
M = json.load(open(f"{REPO}/voxbind/dataset/data/moleculeace_density/manifest.json"))
pairs, pKi = [], {}
for tgt, rec in M.items():
    ms = rec.get("density_mols", [])
    for i, j in rec.get("pairs", []):
        a, b = ms[i], ms[j]; pairs.append((tgt, a["best"], b["best"]))
        pKi[a["best"]] = a["pKi"]; pKi[b["best"]] = b["pKi"]
cliff_mols = sorted(pKi)

# ── probe: train on lp_edrscc_v2 train, predict on cliff mols (raw features) ──
def probe_predict(model):
    ds = pr.build_dataset({p: feats[model][p] for p in feats[model]}, lp,
                          drop_covalent=True, cl1_only=False, split_map=split_map)
    have = [p for p in cliff_mols if p in feats[model]]
    Xc = torch.from_numpy(np.stack([feats[model][p].numpy() for p in have]).astype(np.float32))
    Xtr, ytr = torch.from_numpy(ds["train"]["X"]), torch.from_numpy(ds["train"]["y"])
    Xva, yva = torch.from_numpy(ds["val"]["X"]), torch.from_numpy(ds["val"]["y"])
    seed_preds = []
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        net = pr.MLP2(Xtr.shape[1], hidden=128, dropout=0.1, out_dim=1)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4); mse = torch.nn.MSELoss()
        n, best, bstate, since = Xtr.shape[0], -1e9, None, 0
        for _ in range(200):
            net.train(); perm = torch.randperm(n)
            for s in range(0, n, 64):
                idx = perm[s:s+64]; opt.zero_grad(); mse(net(Xtr[idx]), ytr[idx]).backward(); opt.step()
            net.eval()
            with torch.no_grad(): vs = spearmanr(net(Xva).numpy(), yva.numpy()).statistic
            if vs > best: best, bstate, since = vs, {k: v.clone() for k, v in net.state_dict().items()}, 0
            else:
                since += 1
                if since >= 30: break
        net.load_state_dict(bstate); net.eval()
        with torch.no_grad(): seed_preds.append(net(Xc).numpy().ravel())
    return dict(zip(have, np.stack(seed_preds).mean(0)))

preds = {m: probe_predict(m) for m in MODELS}
preds["TargetDiff / EGNN"] = {r["pid"]: float(r["y_pred"]) for r in csv.DictReader(
    open(f"{REPO}/notebook/html/260625/scatter_egnn_targetdiff.csv")) if r["pid"] in pKi}
hb_seed = [json.load(open(f"{REPO}/base/hbgsa/results/preds_edrscc_40m_seed{s}.json")) for s in range(3)]
preds["HBGSA"] = {p: float(np.mean([dict(zip(h["pdb_id"], h["pred"]))[p]
                 for h in hb_seed if p in set(h["pdb_id"])]))
                 for p in pKi if all(p in set(h["pdb_id"]) for h in hb_seed)}

# ── metrics per model on ALL pairs it can reach (no test filter) ─────────
def sanity(model):
    d = preds[model]; ts = [p for p in TEST if p in d]
    # sanity needs a full-test pred; probe dict only has cliff mols → skip for probes, report cliff-set only
    return float("nan")
res = {"n_pairs_total": len(pairs), "note": "no test-split filter; TRAIN/VAL members in-sample for probes",
       "models": {}}
for m in preds:
    d = preds[m]
    cov = [(t, a, b) for t, a, b in pairs if a in d and b in d]
    sa = np.mean([np.sign(d[a]-d[b]) == np.sign(pKi[a]-pKi[b]) for _, a, b in cov])
    mols = {}
    for _, a, b in cov: mols[a] = pKi[a]; mols[b] = pKi[b]
    rc = np.sqrt(np.mean([(d[p]-pKi[p])**2 for p in mols]))
    oos = [(t, a, b) for t, a, b in cov if a in TEST and b in TEST]
    res["models"][m] = {"sign_acc": float(sa), "rmse_cliff": float(rc),
                        "n_pairs": len(cov), "n_oos_pairs": len(oos),
                        "sign_correct": int(np.sum([np.sign(d[a]-d[b]) == np.sign(pKi[a]-pKi[b]) for _, a, b in cov]))}

json.dump(res, open(f"{REPO}/voxbind/dataset/data/pdbbind/cliff_eval_moleculeace_all.json", "w"), indent=1)
print(f"{'model':22s}{'sign-acc':>10}{'RMSE_cliff':>12}{'n_pairs':>9}{'(oos)':>7}")
print("-"*62)
for m, r in res["models"].items():
    print(f"{m:22s}{r['sign_acc']:>10.3f}{r['rmse_cliff']:>12.3f}{r['n_pairs']:>9}{r['n_oos_pairs']:>7}"
          f"   ({r['sign_correct']}/{r['n_pairs']})")
