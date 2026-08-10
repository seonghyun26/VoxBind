"""casf_clean_rescore.py — score saved CASF-2016 predictions on the CASF2016-clean subset.

CASF-2016 (214) overlaps our lp_edrscc_v2: train 90 / val 32 / test 87 / outside 5.
 - leaky    = all 214
 - nontrain = not in v2 TRAIN (124; still contains the 32 val-overlap)
 - clean    = CASF2016-clean = not in v2 train OR val (92; truly held out from the probe)

Re-aggregates existing per-complex predictions (NO re-training) for every baseline that saved
per-complex CASF preds: CheapNet, GET, EGNN, EGNN+TD, HBGSA, AEV-PLIG (3-seed) and Nesso-1
(zero-shot). CDG / C come from probe_casf_100m_mask075.py (--clean group).
NOTE: Nesso trained on all of PDBbind, so even 'clean' is leaked FOR NESSO (leaked lower bound).

Usage:  cd voxbind && python test/casf_clean_rescore.py
"""
import csv, glob, json, os
import numpy as np
from scipy.stats import pearsonr, spearmanr

REPO = "/home/shpark/prj-denovo/VoxBind"
PIC50_OFFSET = 6.0  # Nesso: pIC50 = -log10(IC50/M) = -raw + 6


def load_sets():
    casf = list(csv.DictReader(open(f"{REPO}/voxbind/splits/casf2016_eval.csv")))
    pK = {r["pid"].lower(): float(r["pK"]) for r in casf}
    nontrain = {r["pid"].lower() for r in casf if r["in_v2train"] == "0"}
    v2 = {r["pid"].lower(): r["split"] for r in csv.DictReader(open(f"{REPO}/voxbind/splits/lp_edrscc_v2.csv"))}
    clean = {p for p in pK if v2.get(p) not in ("train", "val")}
    return pK, nontrain, clean


def score_masks(preds, pK, nontrain, clean, rmse_pred=None):
    rmse_pred = rmse_pred if rmse_pred is not None else preds
    out = {}
    for name, keep in (("leaky", set(preds)), ("nontrain", nontrain), ("clean", clean)):
        pids = [p for p in preds if p in keep and p in pK]
        y = np.array([pK[p] for p in pids]); pc = np.array([preds[p] for p in pids])
        pr = np.array([rmse_pred[p] for p in pids])
        out[name] = dict(n=len(pids), pearson=float(pearsonr(y, pc)[0]),
                         spearman=float(spearmanr(y, pc).statistic),
                         rmse=float(np.sqrt(((pr - y) ** 2).mean())))
    return out


def agg(per):
    out = {}
    for k in per:
        out[k] = {mk: (float(np.mean([d[mk] for d in per[k]])), float(np.std([d[mk] for d in per[k]])))
                  for mk in ("pearson", "spearman", "rmse")}
        out[k]["n"] = per[k][0]["n"]
    return out


# ── per-seed prediction loaders → list of {pid: pred} (one dict per seed) ─────
def jsonl_seeds(pattern, pred_key="pred", id_key="id"):
    out = []
    for f in sorted(glob.glob(pattern)):
        out.append({json.loads(l)[id_key].lower(): float(json.loads(l)[pred_key]) for l in open(f)})
    return out


def hbgsa_seeds():
    out = []
    for f in sorted(glob.glob(f"{REPO}/base/hbgsa/results/preds_casf2016_hbgsa_3p06m_seed*.json")):
        o = json.load(open(f))
        out.append({p.lower(): float(v) for p, v in zip(o["pdb_id"], o["pred"])})
    return out


def aev_seeds():
    rows = list(csv.DictReader(open(f"{REPO}/base/_casf/AEV_preds.csv")))
    return [{r["pid"].lower(): float(r[s]) for r in rows} for s in ("pred_seed0", "pred_seed1", "pred_seed2")]


def csv_seeds(pattern, id_col="pid", pred_col="pred"):
    return [{r[id_col].lower(): float(r[pred_col]) for r in csv.DictReader(open(f))}
            for f in sorted(glob.glob(pattern))]


SEED_METHODS = {
    "CheapNet":          lambda: jsonl_seeds(f"{REPO}/base/cheapnet/_edrscc/preds_casf_seed*.jsonl"),
    "GET":               lambda: jsonl_seeds(f"{REPO}/base/get/_casf_get/preds/preds_GET_casf_seed*.jsonl"),
    "EGNN":              lambda: jsonl_seeds(f"{REPO}/base/get/_casf_get/preds/preds_EGNN_casf_seed*.jsonl"),
    "EGNN + TargetDiff": lambda: jsonl_seeds(f"{REPO}/base/get/_casf_get/preds/preds_EGNN_TD_casf_seed*.jsonl"),
    "HBGSA":             hbgsa_seeds,
    "AEV-PLIG":          aev_seeds,
    "ProFSA":            lambda: csv_seeds(f"{REPO}/base/profsa/_casf/preds/preds_seed*.csv"),
}


def nesso(pK, nontrain, clean):
    pred_dir = f"{REPO}/base/nesso/_edrscc/outputs/predictions"
    raw = {}
    for d in os.listdir(pred_dir):
        af = f"{pred_dir}/{d}/affinity.json"
        if os.path.exists(af):
            v = json.load(open(af)).get("affinity_pred_value")
            if v is not None and np.isfinite(v):
                raw[d.lower()] = float(v)
    signed = {p: -v for p, v in raw.items()}
    pic50 = {p: -v + PIC50_OFFSET for p, v in raw.items()}
    s = score_masks(signed, pK, nontrain, clean, rmse_pred=pic50)
    return {k: {mk: (s[k][mk], 0.0) for mk in ("pearson", "spearman", "rmse")} | {"n": s[k]["n"]} for k in s}


def main():
    pK, nontrain, clean = load_sets()
    print(f"masks: leaky {len(pK)} | nontrain {len(nontrain)} | clean(CASF2016-clean) {len(clean)}\n")
    results = {}
    for meth, loader in SEED_METHODS.items():
        seeds = loader()
        per = {k: [] for k in ("leaky", "nontrain", "clean")}
        for d in seeds:
            s = score_masks(d, pK, nontrain, clean)
            for k in per:
                per[k].append(s[k])
        results[meth] = agg(per)
    results["Nesso-1"] = nesso(pK, nontrain, clean)

    print(f"{'method':<20}{'set':<10}{'n':>5}   r          ρ       RMSE")
    print("-" * 62)
    for meth, res in results.items():
        for s in ("leaky", "nontrain", "clean"):
            b = res[s]
            print(f"{meth:<20}{s:<10}{b['n']:>5}   {b['pearson'][0]:.3f}±{b['pearson'][1]:.2f}"
                  f"   {b['spearman'][0]:.3f}   {b['rmse'][0]:.3f}")
        print()
    json.dump(results, open(f"{REPO}/base/_casf/_casf2016_clean_baselines.json", "w"), indent=2)
    print("written → base/_casf/_casf2016_clean_baselines.json")

    # merge the clean block INTO each method's {key}.json so the results table (load_casf) reads it
    KEY = {"CheapNet": "CheapNet", "GET": "GET", "EGNN": "EGNN", "EGNN + TargetDiff": "EGNN_TD",
           "HBGSA": "HBGSA", "AEV-PLIG": "AEV", "Nesso-1": "Nesso", "ProFSA": "ProFSA"}
    for meth, res in results.items():
        key = KEY.get(meth); jp = f"{REPO}/base/_casf/{key}.json" if key else None
        if not jp or not os.path.exists(jp):
            continue
        d = json.load(open(jp)); c = res["clean"]
        d["clean"] = {mk: {"mean": c[mk][0], "std": c[mk][1]} for mk in ("pearson", "spearman", "rmse")}
        d["clean"]["n"] = c["n"]
        json.dump(d, open(jp, "w"), indent=2)
        print(f"  merged clean-92 → {key}.json")


if __name__ == "__main__":
    main()
