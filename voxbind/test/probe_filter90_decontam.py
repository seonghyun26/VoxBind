"""probe_filter90_decontam.py — Nesso-style 90% protein-identity train decontamination.

Question: does the champion's test performance rely on train complexes whose PROTEIN is a near-
duplicate (>=90% sequence identity) of a test protein? We remove those train complexes and retrain
ONLY the probe head (frozen encoder features are cached → nothing re-pretrained), then compare ρ.

  full   = head trained on ALL train
  filt90 = head trained on train MINUS the 158 pids that are >=90% seq-id to any test protein
           (mmseqs easy-search test↔train, /tmp/seqsim/train_ge90_to_test.txt)

Test set is IDENTICAL in both (the 158 removed are all train pids). Cheap: MLP head only, 3 seeds.
Runs the matched champion C+D+G (d640L18h10, PLINDER-v2, mask0.75, e49) and its coords twin C.

Run:  cd voxbind && CUDA_VISIBLE_DEVICES=<g> python test/probe_filter90_decontam.py
"""
import csv, importlib.util, sys
import numpy as np
import torch

REPO = "/home/shpark/prj-denovo/VoxBind"
sys.path.insert(0, REPO)
_spec = importlib.util.spec_from_file_location("p01c", f"{REPO}/voxbind/dataset/01c_pdbbind_probe.py")
pr = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(pr)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HP = dict(max_epochs=200, patience=30, batch_size=64, lr=1e-3, weight_decay=1e-4,
          hidden=128, dropout=0.1, head="scalar", soft_sigma=1.0)
CDG_FEAT = "atomblob_density_gradmag_e49_v5_260705_ar_cvit_100m_v2_d640L18h10_m075_e50_s1.pt"
C_FEAT   = "atomblob_e49_v5_260723_ar_cvit_100m_v2_mask075_coords.pt"
FILTER   = "/tmp/seqsim/train_ge90_to_test.txt"
SPLIT_CSV = f"{REPO}/voxbind/splits/lp_edrscc_v2.csv"   # canonical champion split (Kd/Ki, test 1320)
SPLIT_MAP = {r["pid"].lower(): r["split"] for r in csv.DictReader(open(SPLIT_CSV))}


def feats(fname):
    return torch.load(pr.FEAT_DIR / fname, weights_only=False)["features"]


def run(feat_dict, lp_df, label):
    # split_map forces the canonical lp_edrscc_v2 membership (NOT LP_PDBBind new_split)
    data = pr.build_dataset(feat_dict, lp_df, drop_covalent=True, cl1_only=False, split_map=SPLIT_MAP)
    ms = [pr.train_one(data, seed=s, device=DEVICE, **HP) for s in range(3)]
    g = lambda k: np.array([m[k] for m in ms])
    return dict(label=label, n_test=ms[0]["n_test"], n_train=ms[0]["n_train"],
                rho=g("test_spearman"), rmse=g("test_rmse"))


def main():
    lp_df = pr.load_lp_index(pr.LP_CSV)
    filt = set(l.strip().lower() for l in open(FILTER) if l.strip())
    cdg, c = feats(CDG_FEAT), feats(C_FEAT)
    shared = set(cdg) & set(c)
    cdg = {p: cdg[p] for p in shared}; c = {p: c[p] for p in shared}
    n_filt_present = len(filt & shared)
    print(f"device={DEVICE} | shared pids {len(shared)} | 90%-to-test train pids present {n_filt_present}\n")

    out = {}
    for cond, fd in (("C+D+G", cdg), ("C", c)):
        out[(cond, "full")] = run(fd, lp_df, f"{cond} full")
        fd_filt = {p: v for p, v in fd.items() if p not in filt}       # drop >=90%-to-test train pids
        out[(cond, "filt90")] = run(fd_filt, lp_df, f"{cond} filt90")

    print(f"{'run':<16}{'n_tr':>6}{'n_te':>6}{'test_ρ':>16}{'test_RMSE':>16}")
    print("-" * 56)
    for cond in ("C+D+G", "C"):
        for v in ("full", "filt90"):
            r = out[(cond, v)]
            print(f"{r['label']:<16}{r['n_train']:>6}{r['n_test']:>6}"
                  f"{r['rho'].mean():>9.3f}±{r['rho'].std():.3f}{r['rmse'].mean():>10.3f}±{r['rmse'].std():.3f}")
        f, fl = out[(cond, "full")], out[(cond, "filt90")]
        print(f"{'  → Δ(filt-full)':<16}{'':>12}{fl['rho'].mean()-f['rho'].mean():>+9.3f}"
              f"{'':>7}{fl['rmse'].mean()-f['rmse'].mean():>+10.3f}\n")


if __name__ == "__main__":
    main()
