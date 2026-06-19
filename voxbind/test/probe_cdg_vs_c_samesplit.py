"""probe_cdg_vs_c_samesplit.py — clean C vs C+D+G on the IDENTICAL PLINDER split.

The chain probes evaluated CDG (n=839) and coords (n=922) on DIFFERENT pools (coords needs no
density voxel → larger), confounding the "CDG > C" read. Here we intersect the two feature caches
(CDG pids ⊆ coords pids) so both heads train/test on the SAME molecules, and report ρ + RMSE + MAE
(MAE just added to train_one). 3 seeds, same head hyperparams as the original probe.

Run:
    cd /home/shpark/prj-denovo/VoxBind/voxbind
    python test/probe_cdg_vs_c_samesplit.py
"""
import importlib.util
import sys

import numpy as np
import torch

REPO = "/home/shpark/prj-denovo/VoxBind"
sys.path.insert(0, REPO)

_spec = importlib.util.spec_from_file_location("p01c", f"{REPO}/voxbind/dataset/01c_pdbbind_probe.py")
pr = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(pr)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HP = dict(max_epochs=200, patience=30, batch_size=64, lr=1e-3, weight_decay=1e-4,
          hidden=128, dropout=0.1, head="scalar", soft_sigma=1.0)   # probe defaults
PAIRS = {
    "invfreq": ("atomblob_density_gradmag_e99_v5_plinder.pt",
                "atomblob_ligvdw_e99_v5_plinder.pt"),
    "uniform": ("atomblob_density_gradmag_e99_v5_plinder_uniform.pt",
                "atomblob_ligvdw_e99_v5_plinder_coords_uniform.pt"),
    # 260618 general-set expansion: same invfreq PLINDER encoders, features re-extracted
    # over the full LP split (refined+core+general, ~15k). See project_pdbbind_general_expansion.
    "invfreq_full": ("atomblob_density_gradmag_e99_v5_plinder_full.pt",
                     "atomblob_ligvdw_e99_v5_plinder_full.pt"),
}


def feats(fname):
    return torch.load(pr.FEAT_DIR / fname, weights_only=False)["features"]


def run(feat_dict, lp_df, label):
    data = pr.build_dataset(feat_dict, lp_df, drop_covalent=True, cl1_only=False)
    ms = [pr.train_one(data, seed=s, device=DEVICE, **HP) for s in range(3)]
    g = lambda k: np.array([m[k] for m in ms])
    return dict(label=label, n_test=ms[0]["n_test"], n_train=ms[0]["n_train"],
                rho=g("test_spearman"), rmse=g("test_rmse"), mae=g("test_mae"))


def main():
    lp_df = pr.load_lp_index(pr.LP_CSV)
    print(f"device={DEVICE}  | clean C-vs-CDG on the IDENTICAL PLINDER split (CDG ∩ coords pids)\n")
    out = {}
    for w, (cdg_f, c_f) in PAIRS.items():
        cdg, c = feats(cdg_f), feats(c_f)
        shared = set(cdg) & set(c)
        print(f"[{w}] CDG feats {len(cdg)} | coords feats {len(c)} | shared {len(shared)} "
              f"(CDG⊆coords: {set(cdg) <= set(c)})")
        cdg_r = {p: cdg[p] for p in shared}
        c_r = {p: c[p] for p in shared}
        out[(w, "CDG")] = run(cdg_r, lp_df, f"{w} CDG")
        out[(w, "C")] = run(c_r, lp_df, f"{w} coords")

    print(f"\n{'run':<16}{'n_tr':>6}{'n_te':>6}{'test_rho':>15}{'test_RMSE':>15}{'test_MAE':>15}")
    print("-" * 73)
    for w in PAIRS:
        for cond in ("CDG", "C"):
            r = out[(w, cond)]
            print(f"{r['label']:<16}{r['n_train']:>6}{r['n_test']:>6}"
                  f"{r['rho'].mean():>8.3f}±{r['rho'].std():.3f}"
                  f"{r['rmse'].mean():>9.3f}±{r['rmse'].std():.3f}"
                  f"{r['mae'].mean():>9.3f}±{r['mae'].std():.3f}")
        d = out[(w, "CDG")]; b = out[(w, "C")]
        print(f"{'  → density Δ':<16}{'':>6}{'(same '+str(d['n_test'])+')':>6}"
              f"{d['rho'].mean()-b['rho'].mean():>+8.3f}{'':>7}"
              f"{d['rmse'].mean()-b['rmse'].mean():>+9.3f}{'':>6}"
              f"{d['mae'].mean()-b['mae'].mean():>+9.3f}")
        print()


if __name__ == "__main__":
    main()
