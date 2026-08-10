"""compile_holdout2019.py — aggregate all method predictions on the 2019 temporal holdout.

The holdout is entirely clean (2019+, not in our train/val), so a single metric set per method.
Reads whatever predictions exist (skips missing), computes r/ρ/RMSE (3-seed mean±std where
applicable), and writes base/_casf/_holdout2019_summary.json + prints a leaderboard.
"""
import csv, glob, json, os
import numpy as np
from scipy.stats import pearsonr, spearmanr

REPO = "/home/shpark/prj-denovo/VoxBind"
PIC50 = 6.0
pK = {r["pid"].lower(): float(r["pK"]) for r in csv.DictReader(open(f"{REPO}/voxbind/splits/holdout2019_eval.csv"))}


def metr(preds, rmse_preds=None):
    rmse_preds = rmse_preds if rmse_preds is not None else preds
    pids = [p for p in preds if p in pK]
    if len(pids) < 2: return None
    y = np.array([pK[p] for p in pids]); pc = np.array([preds[p] for p in pids])
    pr = np.array([rmse_preds[p] for p in pids])
    return dict(n=len(pids), r=float(pearsonr(y, pc)[0]), rho=float(spearmanr(y, pc).statistic),
                rmse=float(np.sqrt(((pr - y) ** 2).mean())))


def agg(dicts):
    return {k: (float(np.mean([d[k] for d in dicts])), float(np.std([d[k] for d in dicts]))) for k in ("r", "rho", "rmse")} | {"n": dicts[0]["n"]}


def jsonl_seeds(pattern, pred_key="pred", id_key="id", valid_range=None):
    """valid_range=(lo,hi): drop predictions outside [lo,hi] as invalid model outputs
    (CheapNet's GATv2 blows up to pred>20 on auto-prepped OOD structures — those are
    not evaluable; the in-range subset is its valid domain)."""
    out = []
    for f in sorted(glob.glob(pattern)):
        d = {}
        for l in open(f):
            o = json.loads(l)
            key = o.get("id") or o.get("pdbid") or o.get("pid")
            v = float(o[pred_key])
            if valid_range and not (valid_range[0] <= v <= valid_range[1]):
                continue
            d[str(key).lower()] = v
        m = metr(d)
        if m: out.append(m)
    return out


def from_probe_json(path):
    if not os.path.exists(path): return None
    d = json.load(open(path))["holdout"]
    return {"r": (d["pearson"]["mean"], d["pearson"]["std"]), "rho": (d["spearman"]["mean"], d["spearman"]["std"]),
            "rmse": (d["rmse"]["mean"], d["rmse"]["std"]), "n": d["n"]}


def nesso():
    pd_ = f"{REPO}/base/nesso/_holdout2019/outputs/predictions"
    if not os.path.isdir(pd_): return None
    raw = {}
    for dd in os.listdir(pd_):
        af = f"{pd_}/{dd}/affinity.json"
        if os.path.exists(af):
            v = json.load(open(af)).get("affinity_pred_value")
            if v is not None and np.isfinite(v): raw[dd.lower()] = float(v)
    if not raw: return None
    signed = {p: -v for p, v in raw.items()}; pic = {p: -v + PIC50 for p, v in raw.items()}
    m = metr(signed, rmse_preds=pic)
    return {"r": (m["r"], 0.0), "rho": (m["rho"], 0.0), "rmse": (m["rmse"], 0.0), "n": m["n"]}


def aev():
    p = f"{REPO}/base/_casf/AEV_holdout2019_preds.csv"
    if not os.path.exists(p): return None
    rows = list(csv.DictReader(open(p)))
    cols = [c for c in rows[0] if c.startswith("pred_seed")]
    if not cols: cols = [c for c in rows[0] if "pred" in c][:1]
    ds = [metr({r["pid"].lower(): float(r[c]) for r in rows if r.get(c) not in (None, "")}) for c in cols]
    return agg(ds) if ds else None


def main():
    R = {}
    for m in ("GET", "EGNN", "EGNN_TD"):
        d = jsonl_seeds(f"{REPO}/base/get/_casf_get/preds/preds_{m}_holdout2019_seed*.jsonl")
        if d: R[{"EGNN_TD": "EGNN + TargetDiff"}.get(m, m)] = agg(d)
    cn = jsonl_seeds(f"{REPO}/base/cheapnet/_edrscc/data_holdout2019/preds_casf_seed*.jsonl", valid_range=(0, 14))
    if cn: R["CheapNet"] = agg(cn)
    # ProFSA: per-seed CSVs (pid, pred, label, in_v2train)
    pf = []
    for f in sorted(glob.glob(f"{REPO}/base/profsa/_casf/preds_holdout2019_seed*.csv")):
        m = metr({r["pid"].lower(): float(r["pred"]) for r in csv.DictReader(open(f))})
        if m: pf.append(m)
    if pf: R["ProFSA"] = agg(pf)
    for name, path in (("C+D+G", "CDG_holdout2019"), ("C+D+G +corr", "CDG_corr5_holdout2019"), ("C", "C_holdout2019")):
        r = from_probe_json(f"{REPO}/base/_casf/{path}.json")
        if r: R[name] = r
    for name, fn in (("AEV-PLIG", aev), ("Nesso-1", nesso)):
        r = fn()
        if r: R[name] = r
    # seq+SMILES DTA baselines — per-seed (DeepPurpose) or single-pass (PLAPT zero-shot) CSVs (pid,pred,y)
    def csv_seeds(pattern):
        out = []
        for f in sorted(glob.glob(pattern)):
            m = metr({r["pid"].lower(): float(r["pred"]) for r in csv.DictReader(open(f))})
            if m: out.append(m)
        return out
    for name, pat in (("DeepDTA", "base/_casf/DeepDTA_holdout2019_preds_seed*.csv"),
                      ("GraphDTA", "base/_casf/GraphDTA_holdout2019_preds_seed*.csv"),
                      ("MolTrans", "base/_casf/MolTrans_holdout2019_preds_seed*.csv"),
                      ("PLAPT", "base/_casf/PLAPT_holdout2019_preds.csv")):
        d = csv_seeds(f"{REPO}/{pat}")
        if d: R[name] = agg(d)
    # HBGSA: parse per-seed preds (pdb_id[]+pred[]) if present, else the summary json leaky block
    hb = sorted(glob.glob(f"{REPO}/base/hbgsa/results/preds_holdout2019_hbgsa_seed*.json"))
    hbd = []
    for f in hb:
        o = json.load(open(f))
        if isinstance(o, dict) and "pdb_id" in o and "pred" in o:
            m = metr({p.lower(): float(v) for p, v in zip(o["pdb_id"], o["pred"])})
            if m: hbd.append(m)
    if hbd:
        R["HBGSA"] = agg(hbd)
    elif os.path.exists(f"{REPO}/base/_casf/HBGSA_holdout2019.json"):
        b = json.load(open(f"{REPO}/base/_casf/HBGSA_holdout2019.json"))["leaky"]
        R["HBGSA"] = {"r": (b["pearson"]["mean"], b["pearson"].get("std", 0)), "rho": (b["spearman"]["mean"], b["spearman"].get("std", 0)),
                      "rmse": (b["rmse"]["mean"], b["rmse"].get("std", 0)), "n": b.get("n", 0)}
    # print leaderboard sorted by rho
    order = sorted(R, key=lambda m: -R[m]["rho"][0])
    print(f"\n{'method':<20}{'n':>5}{'r':>9}{'ρ':>9}{'RMSE':>9}   (2019 holdout, clean)")
    print("-" * 62)
    for m in order:
        b = R[m]
        print(f"{m:<20}{b['n']:>5}{b['r'][0]:>9.3f}{b['rho'][0]:>9.3f}{b['rmse'][0]:>9.3f}")
    json.dump(R, open(f"{REPO}/base/_casf/_holdout2019_summary.json", "w"), indent=2)
    print(f"\nwritten -> base/_casf/_holdout2019_summary.json  ({len(R)} methods)")


if __name__ == "__main__":
    main()
