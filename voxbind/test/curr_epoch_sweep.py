"""curr_epoch_sweep.py — which epoch of the interface+curriculum run is best?

For each saved epoch of 260823_cdg_100m_v2_interface_curriculum_0609, train the 5-seed
mse+corr head (SAME machinery as results.html: probe_casf.train_predict) and score the
results.html cohorts we care about:
  CL3-ID60 / CL3-ID30  (cl123 train->test, filter to protein-novelty pid lists)
  CASF leaky / nontrain / clean
FULL(LP) is read from the 01c watch panel (already results.html-consistent), not recomputed.
Prints a per-epoch table + the argmax epoch per cohort.  (CASF id60/id30 intentionally skipped.)
"""
import csv, os, sys, statistics as st
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from probe_casf_100m_mask075 import (  # noqa: E402
    FD, REPO, build_loss, load_pK, casf_eval, v2_split, train_predict)
import torch  # noqa: E402
torch.set_num_threads(4)

RUN = "260823_cdg_100m_v2_interface_curriculum_0609"
EPOCHS = [10, 20, 30, 40, 49]
SEEDS = 5
LOSS = build_loss("mse+corr", 5.0)
CLD = f"{REPO}/base/_casf/cl123_seqfilter_5seed_260818"
# FULL(LP) msecorr 5-seed from the 01c watch panel (results.html-consistent source)
FULL_PANEL = {10: 0.607, 20: 0.649, 30: 0.641, 40: 0.638, 49: 0.630}


def load_feats(ep):
    bn = f"atomblob_density_gradmag_e{ep}_v5_{RUN}.pt"
    d = torch.load(os.path.join(FD, bn), weights_only=False)
    f = d.get("features", d.get("feat"))
    return {p.lower(): np.asarray(v.numpy() if hasattr(v, "numpy") else v, dtype=np.float32)
            for p, v in f.items()}


def ids(path):
    return set(l.strip().lower() for l in open(path) if l.strip())


def cl123_split():
    return {r["pid"].lower(): r["split"]
            for r in csv.DictReader(open(f"{REPO}/voxbind/splits/lp_edrscc_v2_cl123.csv"))}


def rho_on(te, yte, pte, keep):
    xs = [(pte[i], yte[i]) for i, p in enumerate(te) if (keep is None or p in keep)]
    if len(xs) < 3:
        return None
    yh = np.array([a for a, _ in xs]); yy = np.array([b for _, b in xs])
    return spearmanr(yh, yy).correlation


def agg(vals):
    vals = [v for v in vals if v is not None]
    return (st.mean(vals), st.pstdev(vals) if len(vals) > 1 else 0.0) if vals else (float("nan"), 0.0)


def main():
    pK = load_pK()
    v2 = v2_split(); casf_pids, nontrain = casf_eval()
    clean_set = {p for p in casf_pids if v2.get(p) not in ("train", "val")}
    cl = cl123_split(); cl_test = [p for p, s in cl.items() if s == "test"]
    nov60 = ids(f"{CLD}/cl123_test_novel60.txt"); nov30 = ids(f"{CLD}/cl123_test_novel30.txt")
    print(f"cohorts: CL3-test={len(cl_test)} nov60={len(nov60)} nov30={len(nov30)} | "
          f"CASF leaky={len(casf_pids)} nontrain={len(nontrain)} clean={len(clean_set)} | seeds={SEEDS}\n", flush=True)

    COLS = ["FULL", "CL3-ID60", "CL3-ID30", "CASF-leaky", "CASF-nontrain", "CASF-clean"]
    table = {}
    for ep in EPOCHS:
        feats = load_feats(ep)
        buf = {c: [] for c in COLS if c != "FULL"}
        for s in range(SEEDS):
            # CASF: head on lp_edrscc_v2 train -> predict CASF
            te, yte, pte = train_predict(feats, pK, v2, casf_pids, s, loss_fn=LOSS)
            buf["CASF-leaky"].append(rho_on(te, yte, pte, None))
            buf["CASF-nontrain"].append(rho_on(te, yte, pte, nontrain))
            buf["CASF-clean"].append(rho_on(te, yte, pte, clean_set))
            # CL3: head on cl123 train -> predict cl123 test -> filter novelty
            tc, ytc, ptc = train_predict(feats, pK, cl, cl_test, s, loss_fn=LOSS)
            buf["CL3-ID60"].append(rho_on(tc, ytc, ptc, nov60))
            buf["CL3-ID30"].append(rho_on(tc, ytc, ptc, nov30))
        row = {"FULL": (FULL_PANEL.get(ep, float("nan")), 0.0)}
        for c in buf:
            row[c] = agg(buf[c])
        table[ep] = row
        print(f"e{ep:<3} " + "  ".join(f"{c}={row[c][0]:.3f}" for c in COLS), flush=True)

    print("\n=== best epoch per cohort ===")
    for c in COLS:
        best = max(EPOCHS, key=lambda e: (table[e][c][0] if table[e][c][0] == table[e][c][0] else -9))
        print(f"  {c:14} best=e{best}  ρ={table[c if False else best][c][0]:.3f}")
    # composite = mean of the 5 leak-proof/honest axes (exclude leaky)
    honest = ["CL3-ID60", "CL3-ID30", "CASF-nontrain", "CASF-clean"]
    print("\n=== composite (mean of CL3-ID60/30 + CASF-nontrain/clean) ===")
    for ep in EPOCHS:
        m = st.mean([table[ep][c][0] for c in honest])
        print(f"  e{ep:<3} composite={m:.3f}")


if __name__ == "__main__":
    main()
