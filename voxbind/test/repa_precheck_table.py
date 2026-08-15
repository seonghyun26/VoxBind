"""repa_precheck_table.py — one consolidated, protocol-matched table for the U-REPA pre-check.

Every row uses the PUBLISHED probe protocol: RAW (un-standardized) mean-pooled features,
2-layer MLP head, lp_edrscc_v2 frozen split, drop_covalent. Standardizing was checked and
rejected: it is a no-op on the U-Net stages (0.547 vs 0.548) but costs the CDG encoder
~0.05 rho (holo 0.642 raw vs 0.588 z) — post-LayerNorm ViT features carry signal in their
per-dimension scale. Raw also reproduces the model_zoo number exactly (holo 0.642 vs 0.641).

Reads the three feature caches produced by:
    test/voxbind_probe_null.py        (trivial baselines, no network)
    test/voxbind_unet_probe.py        (VoxBind ep923 U-Net, 5 depths x 3 ligand conditions)
    test/cdg_pocketonly_probe.py      (CDG teacher, 3 ligand conditions)

Run:
    python test/repa_precheck_table.py
"""
import contextlib
import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
VOX = REPO / "voxbind"
sys.path.insert(0, str(REPO))

_spec = importlib.util.spec_from_file_location(
    "p01c", str(VOX / "dataset" / "01c_pdbbind_probe.py"))
pr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pr)

FEAT = VOX / "dataset" / "data" / "pdbbind" / "features"
HP = dict(max_epochs=200, patience=30, batch_size=64, lr=1e-3, weight_decay=1e-4,
          hidden=128, dropout=0.1, head="scalar", soft_sigma=1.0)
SEEDS = 1
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# what information each row's input actually contains -> which null it must be judged against
LIG_ROWS = "lig+poc"      # ligand present  -> null atomsum11
POC_ROWS = "poc"          # ligand absent   -> null pocsum4


def probe(feats, lp, sm, label, family, info):
    with contextlib.redirect_stdout(io.StringIO()):          # silence split-availability spam
        d = pr.build_dataset(feats, lp, drop_covalent=True, cl1_only=False, split_map=sm)
        ms = [pr.train_one(d, seed=s, device=DEV, **HP) for s in range(SEEDS)]
    rho = np.array([m["test_spearman"] for m in ms])
    rmse = np.array([m["test_rmse"] for m in ms])
    r = dict(family=family, row=label, info=info, dim=d["train"]["X"].shape[1],
             n_test=ms[0]["n_test"], rho=rho.mean(), rmse=rmse.mean())
    print(f"  {family:<6} {label:<16} {info:<8} dim{r['dim']:>5}  "
          f"rho {r['rho']:.3f}  rmse {r['rmse']:.3f}", flush=True)
    return r


def main():
    lp = pr.load_lp_index(pr.LP_CSV)
    sm, scheme = pr.load_frozen_split_map("lp_edrscc_v2")
    print(f"  split {scheme} | RAW features | {SEEDS} seed(s)\n")
    rows = []

    # ── trivial baselines (no network) ────────────────────────────────────────
    s = torch.load(FEAT / "voxbind_null_atomsum.pt", map_location="cpu",
                   weights_only=False)["features"]
    rows += [
        probe(s, lp, sm, "atomsum11", "null", LIG_ROWS),
        probe({p: v[:7].clone() for p, v in s.items()}, lp, sm, "ligsum7", "null", LIG_ROWS),
        probe({p: v[:7].sum(0, keepdim=True) for p, v in s.items()}, lp, sm,
              "ligmass1", "null", LIG_ROWS),
        probe({p: v[7:11].clone() for p, v in s.items()}, lp, sm, "pocsum4", "null", POC_ROWS),
        probe({p: v[7:11].sum(0, keepdim=True) for p, v in s.items()}, lp, sm,
              "pocmass1", "null", POC_ROWS),
    ]

    # ── VoxBind U-Net (student) ───────────────────────────────────────────────
    u = torch.load(FEAT / "voxbind_unet_ep923.pt", map_location="cpu", weights_only=False)
    for var in u["variants"]:
        for d in u["depths"]:
            info = POC_ROWS if var == "noise" else LIG_ROWS
            rows.append(probe(u["features"][var][d], lp, sm, f"{var}/{d}", "unet", info))

    # ── CDG teacher ───────────────────────────────────────────────────────────
    c = torch.load(FEAT / "cdg_pocketonly_e60m.pt", map_location="cpu", weights_only=False)
    for var in c["variants"]:
        info = LIG_ROWS if var == "holo" else POC_ROWS
        rows.append(probe(c["features"][var], lp, sm, var, "cdg", info))

    df = pd.DataFrame(rows)
    # headroom above the information-matched trivial baseline
    base = {LIG_ROWS: df[(df.family == "null") & (df.row == "atomsum11")].rho.iloc[0],
            POC_ROWS: df[(df.family == "null") & (df.row == "pocsum4")].rho.iloc[0]}
    df["null"] = df["info"].map(base)     # df.info is DataFrame.info(), not the column
    df["above_null"] = (df.rho - df["null"]).round(3)
    csv = VOX / "test" / "results" / "repa_precheck_table.csv"
    csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv, index=False)

    print(f"\n  === ligand PRESENT (null atomsum11 = {base[LIG_ROWS]:.3f}) ===")
    print(df[df["info"] == LIG_ROWS][["family", "row", "rho", "above_null"]]
          .to_string(index=False))
    print(f"\n  === ligand ABSENT / generation condition (null pocsum4 = {base[POC_ROWS]:.3f}) ===")
    print(df[df["info"] == POC_ROWS][["family", "row", "rho", "above_null"]]
          .to_string(index=False))
    print(f"\n  -> {csv}")


if __name__ == "__main__":
    main()
