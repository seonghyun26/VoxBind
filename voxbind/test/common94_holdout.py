"""common94_holdout.py — head-to-head on the COMMON complex subset of the 2019 holdout.

Every method is scored on the SAME complexes (the intersection of all methods' available
per-complex predictions, which is bounded by the ~94 CDG-ready voxel set). Uses seed-averaged
predictions per method. Removes the subset confound in the main Table A2 (94 vs 167 vs 704).
"""
import csv, glob, json, os, re
import numpy as np
from scipy.stats import pearsonr, spearmanr

REPO = "/home/shpark/prj-denovo/VoxBind"
PIC50 = 6.0
pK = {r["pid"].lower(): float(r["pK"]) for r in csv.DictReader(open(f"{REPO}/voxbind/splits/holdout2019_eval.csv"))}

# Consistency with CASF clean-92: exclude complexes the champion saw during SSL PRE-TRAINING
# (PLINDER-v2). The holdout is already ¬lp_edrscc_v2(train/val); 76 of 704 also appear in PLINDER-v2
# pretraining, so drop them → the eval set is unseen by our method end-to-end (pretrain + downstream).
_PV2 = set()
for _r in csv.DictReader(open(f"{REPO}/voxbind/splits/plinder/v2/plinder_selected.csv")):
    _m = re.match(r"([0-9a-zA-Z]{4})", str(_r["entry_pdb_id"]))
    if _m:
        _PV2.add(_m.group(1).lower())
_n_before = len(pK)
pK = {p: v for p, v in pK.items() if p not in _PV2}
print(f"holdout: {_n_before} → {len(pK)} after dropping PLINDER-v2 pretraining overlap ({_n_before - len(pK)})")


# ── per-seed loaders (return a LIST of per-seed pred dicts so Table 3 can report mean±std, like the
# CASF table). Single-pass methods (probe csv, PLAPT/Nesso zero-shot) return a 1-element list.
def seed_jsonl(pattern, id_key="id", valid_range=None):
    out = []
    for f in sorted(glob.glob(pattern)):
        d = {}
        for l in open(f):
            o = json.loads(l); k = str(o.get(id_key) or o.get("id") or o.get("pdbid")).lower(); v = float(o["pred"])
            if valid_range and not (valid_range[0] <= v <= valid_range[1]):
                continue
            d[k] = v
        if d:
            out.append(d)
    return out


def seed_csv(pattern, id_col="pid", pred_col="pred"):
    out = []
    for f in sorted(glob.glob(pattern)):
        d = {r[id_col].lower(): float(r[pred_col]) for r in csv.DictReader(open(f))}
        if d:
            out.append(d)
    return out


def probe_csv(path):
    if not os.path.exists(path):
        return []
    return [{r["pid"].lower(): float(r["pred"]) for r in csv.DictReader(open(path))}]


def hbgsa():
    out = []
    for f in sorted(glob.glob(f"{REPO}/base/hbgsa/results/preds_holdout2019_hbgsa_seed*.json")):
        o = json.load(open(f))
        out.append({p.lower(): float(v) for p, v in zip(o["pdb_id"], o["pred"])})
    return out


def aev():
    p = f"{REPO}/base/_casf/AEV_holdout2019_preds.csv"
    if not os.path.exists(p):
        return []
    rows = list(csv.DictReader(open(p)))
    cols = [c for c in rows[0] if c.startswith("pred_seed")] or ["pred_ensemble"]
    return [{r["pid"].lower(): float(r[c]) for r in rows if r.get(c) not in (None, "")} for c in cols]


def nesso():
    d = f"{REPO}/base/nesso/_holdout2019/outputs/predictions"
    out = {}
    for dd in os.listdir(d) if os.path.isdir(d) else []:
        af = f"{d}/{dd}/affinity.json"
        if os.path.exists(af):
            v = json.load(open(af)).get("affinity_pred_value")
            if v is not None and np.isfinite(v):
                out[dd.lower()] = -float(v)          # signed for pK-correlation (zero-shot, 1 "seed")
    return [out] if out else []


PREDS = {
    "C+D+G +corr": seed_csv(f"{REPO}/base/_casf/CDG_corr5_holdout2019_preds_seed*.csv") or probe_csv(f"{REPO}/base/_casf/CDG_corr5_holdout2019_preds.csv"),
    "C+D+G":       seed_csv(f"{REPO}/base/_casf/CDG_holdout2019_preds_seed*.csv") or probe_csv(f"{REPO}/base/_casf/CDG_holdout2019_preds.csv"),
    "C":           seed_csv(f"{REPO}/base/_casf/C_holdout2019_preds_seed*.csv") or probe_csv(f"{REPO}/base/_casf/C_holdout2019_preds.csv"),
    "GET":               seed_jsonl(f"{REPO}/base/get/_casf_get/preds/preds_GET_holdout2019_seed*.jsonl"),
    "EGNN":              seed_jsonl(f"{REPO}/base/get/_casf_get/preds/preds_EGNN_holdout2019_seed*.jsonl"),
    "EGNN + TargetDiff": seed_jsonl(f"{REPO}/base/get/_casf_get/preds/preds_EGNN_TD_holdout2019_seed*.jsonl"),
    "CheapNet":    seed_jsonl(f"{REPO}/base/cheapnet/_edrscc/data_holdout2019/preds_casf_seed*.jsonl", valid_range=(0, 14)),
    "ProFSA":      seed_csv(f"{REPO}/base/profsa/_casf/preds_holdout2019_seed*.csv"),
    "HBGSA":       hbgsa(),
    "AEV-PLIG":    aev(),
    "Nesso-1":     nesso(),
    # seq+SMILES DTA baselines (base/dta + base/plapt)
    "DeepDTA":     seed_csv(f"{REPO}/base/_casf/DeepDTA_holdout2019_preds_seed*.csv"),
    "MolTrans":    seed_csv(f"{REPO}/base/_casf/MolTrans_holdout2019_preds_seed*.csv"),
    "PLAPT":       seed_csv(f"{REPO}/base/_casf/PLAPT_holdout2019_preds.csv"),
}
PREDS = {k: v for k, v in PREDS.items() if v}


def avg_of(seedlist):
    per = {}
    for d in seedlist:
        for k, v in d.items():
            per.setdefault(k, []).append(v)
    return {k: float(np.mean(v)) for k, v in per.items()}


# ── ONE consolidated evaluation set: the ED-available holdout (electron-density crops exist,
# = the set our voxel methods can score) intersected across all FULL-COVERAGE methods. Methods
# whose coverage of the ED set is below MIN_COV (HBGSA — loader resolves only pbpp-2020; Nesso —
# slow ESM timed out) cannot join the identical-complex common, so they are reported separately.
ED_DIR = f"{REPO}/voxbind/dataset/data/pdbbind/voxels_v5/density"
ED_AVAIL = {p for p in pK if os.path.exists(f"{ED_DIR}/{p}.npy")}
MIN_COV = 400        # a method must cover ≥400 of the ED set to define/join the common


def metr_seeds(m, seedlist, common):
    """per-seed r/ρ/RMSE on the common set → {metric:{mean,std}} (matches the CASF table's ±)."""
    rs, rhos, rmses = [], [], []
    for d in seedlist:
        pids = [p for p in common if p in d]
        if len(pids) < 5:
            continue
        y = np.array([pK[p] for p in pids]); pc = np.array([d[p] for p in pids])
        rmse_pred = pc + PIC50 if m == "Nesso-1" else pc     # Nesso RMSE on pIC50 scale
        rs.append(float(pearsonr(y, pc)[0])); rhos.append(float(spearmanr(y, pc).statistic))
        rmses.append(float(np.sqrt(((rmse_pred - y) ** 2).mean())))
    ms = lambda a: {"mean": float(np.mean(a)), "std": float(np.std(a))}
    return {"r": ms(rs), "rho": ms(rhos), "rmse": ms(rmses)}


def main():
    avg = {m: avg_of(v) for m, v in PREDS.items()}
    cov = {m: set(avg[m]) & ED_AVAIL for m in PREDS}
    full = [m for m in PREDS if len(cov[m]) >= MIN_COV]        # define the common
    low = [m for m in PREDS if len(cov[m]) < MIN_COV]          # HBGSA, Nesso, …
    common = set(ED_AVAIL)
    for m in full:
        common &= cov[m]
    common = sorted(common)
    print(f"ED-available={len(ED_AVAIL)} | full-coverage methods={len(full)} | common n={len(common)}")
    print(f"low-coverage (reported separately): {[(m, len(cov[m])) for m in low]}\n")

    out = {}
    for m in full:
        res = metr_seeds(m, PREDS[m], common); res["n"] = len(common); out[m] = res
    for m in low:
        sub = sorted(set(common) & cov[m]) or sorted(cov[m])
        if len(sub) < 5:
            continue
        res = metr_seeds(m, PREDS[m], sub); res["n"] = len(sub); res["partial"] = True; out[m] = res

    order = sorted(out, key=lambda m: -out[m]["rho"]["mean"])
    print(f"{'method':<20}{'r':>9}{'ρ':>9}{'RMSE':>9}{'n':>6}{'seeds':>6}")
    for m in order:
        b = out[m]
        print(f"{m:<20}{b['r']['mean']:>9.3f}{b['rho']['mean']:>9.3f}{b['rmse']['mean']:>9.3f}"
              f"{b['n']:>6}{len(PREDS[m]):>6}{'  *partial' if b.get('partial') else ''}")
    json.dump(out, open(f"{REPO}/base/_casf/_holdout2019_common.json", "w"), indent=2)
    open(f"{REPO}/base/_casf/_holdout2019_common_pids.txt", "w").write("\n".join(common))
    print(f"\nwritten -> base/_casf/_holdout2019_common.json (common ED set n={len(common)})")


if __name__ == "__main__":
    main()
