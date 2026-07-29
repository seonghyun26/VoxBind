"""tabpfn_probe.py — TabPFN / XGBoost readout on FROZEN encoder features.

Swaps the 2-layer MLP probe for stronger small-sample tabular learners on the exact
cached mean-pooled features, using the identical lp_edrscc_v2 frozen split. NO leak:
fit on the original train set only, select nothing on test, evaluate on val + test.

Phase 1 (default): raw 640-dim density features   → TabPFN vs XGBoost.
Phase 2 (--augment): concat cheap ORTHOGONAL columns the density rep can't contain —
  RDKit ligand physchem (MW, logP, HBD/HBA, TPSA, rot-bonds, rings, formal charge,
  heavy-atom count) + crystal resolution — then re-run. The leak test proved the
  density features alone cap ~0.69, so only NEW information can lift the ceiling.

Usage:
  python test/tabpfn_probe.py --features <feat.pt> --tag champion --device cuda:0
  python test/tabpfn_probe.py --features <feat.pt> --tag champion --augment --device cuda:0
"""
import argparse, importlib.util, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

HERE = Path(__file__).resolve().parent
PROBE_PY = HERE.parent / "dataset" / "01c_pdbbind_probe.py"


def load_probe_module():
    spec = importlib.util.spec_from_file_location("probe01c", PROBE_PY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["probe01c"] = mod
    spec.loader.exec_module(mod)
    return mod


def metrics(pred, y):
    pred, y = np.asarray(pred, float), np.asarray(y, float)
    return {
        "r":    float(pearsonr(pred, y).statistic),
        "rho":  float(spearmanr(pred, y).statistic),
        "rmse": float(np.sqrt(((pred - y) ** 2).mean())),
        "n":    int(len(y)),
    }


# ── phase-2 orthogonal descriptors ──────────────────────────────────────────
_RD_COLS = ["MolWt", "MolLogP", "NumHDonors", "NumHAcceptors", "TPSA",
            "NumRotatableBonds", "RingCount", "NumAromaticRings",
            "FractionCSP3", "HeavyAtomCount", "NumHeteroatoms"]


def rdkit_descriptors(smiles: str):
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
    m = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if m is None:
        return [np.nan] * len(_RD_COLS)
    return [
        Descriptors.MolWt(m), Descriptors.MolLogP(m),
        Lipinski.NumHDonors(m), Lipinski.NumHAcceptors(m),
        rdMolDescriptors.CalcTPSA(m), Lipinski.NumRotatableBonds(m),
        rdMolDescriptors.CalcNumRings(m), rdMolDescriptors.CalcNumAromaticRings(m),
        rdMolDescriptors.CalcFractionCSP3(m), m.GetNumHeavyAtoms(),
        rdMolDescriptors.CalcNumHeteroatoms(m),
    ]


def build_aug_map(lp_df):
    """pid -> extra feature vector [rdkit... , resolution]. NaN-safe."""
    idcol = "pdb_id" if "pdb_id" in lp_df.columns else "header"
    aug = {}
    for _, row in lp_df.iterrows():
        pid = str(row[idcol]).lower()
        desc = rdkit_descriptors(row.get("smiles"))
        res = row.get("resolution", np.nan)
        try:
            res = float(res)
        except (TypeError, ValueError):
            res = np.nan
        aug[pid] = np.asarray(desc + [res], dtype=np.float32)
    return aug, _RD_COLS + ["resolution"]


def attach_aug(data, aug_map, ncols):
    """Concat aug columns onto each split's X (median-impute NaN from TRAIN)."""
    # collect train aug rows to compute impute medians
    def rows(split):
        return np.stack([aug_map.get(p.lower(), np.full(ncols, np.nan, np.float32))
                         for p in data[split]["pid"]]).astype(np.float32)
    tr = rows("train")
    med = np.nanmedian(tr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    for split in ("train", "val", "test"):
        a = rows(split)
        nan = ~np.isfinite(a)
        a[nan] = np.take(med, np.where(nan)[1])
        data[split]["X"] = np.concatenate([data[split]["X"], a], axis=1).astype(np.float32)


# ── learners ────────────────────────────────────────────────────────────────
def run_tabpfn(data, device, seeds=(0, 1, 2)):
    from tabpfn import TabPFNRegressor
    Xtr, ytr = data["train"]["X"], data["train"]["y"]
    out = {}
    for split in ("val", "test"):
        preds = []
        for s in seeds:
            reg = TabPFNRegressor(device=device, random_state=s)
            reg.fit(Xtr, ytr)
            preds.append(reg.predict(data[split]["X"]))
        # average the per-seed predictions (TabPFN internal ensemble already; this
        # just averages the config permutations across random_state)
        p = np.mean(preds, axis=0)
        out[split] = metrics(p, data[split]["y"])
    return out


def run_xgb(data, seeds=(0, 1, 2)):
    from xgboost import XGBRegressor
    Xtr, ytr = data["train"]["X"], data["train"]["y"]
    out = {}
    for split in ("val", "test"):
        preds = []
        for s in seeds:
            reg = XGBRegressor(n_estimators=600, max_depth=5, learning_rate=0.03,
                               subsample=0.8, colsample_bytree=0.6, reg_lambda=2.0,
                               n_jobs=8, random_state=s)
            reg.fit(Xtr, ytr)
            preds.append(reg.predict(data[split]["X"]))
        out[split] = metrics(np.mean(preds, axis=0), data[split]["y"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="cached feature bundle .pt")
    ap.add_argument("--tag", default="enc", help="label for printout")
    ap.add_argument("--split", default="lp_edrscc_v2")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--augment", action="store_true",
                    help="phase 2: concat RDKit ligand descriptors + resolution")
    ap.add_argument("--learners", nargs="+", default=["tabpfn", "xgb"],
                    choices=["tabpfn", "xgb"])
    args = ap.parse_args()

    probe = load_probe_module()
    split_map, scheme = probe.load_frozen_split_map(args.split)
    lp_df = probe.load_lp_index(probe.LP_CSV)

    bundle = torch.load(args.features, weights_only=False)
    feats = bundle["features"]
    data = probe.build_dataset(feats, lp_df, drop_covalent=True, cl1_only=False,
                               target_map=None, split_map=split_map)
    dim = data["train"]["X"].shape[1]
    print(f"=== TabPFN/XGB readout · {args.tag} · split={args.split} ({scheme}) ===")
    print(f"  feature bundle : {Path(args.features).name}")
    print(f"  split sizes    : train={len(data['train']['pid']):,}  "
          f"val={len(data['val']['pid']):,}  test={len(data['test']['pid']):,}")
    print(f"  base dim       : {dim}")

    if args.augment:
        aug_map, cols = build_aug_map(lp_df)
        attach_aug(data, aug_map, len(cols))
        print(f"  + augment      : {len(cols)} cols {cols} -> dim {data['train']['X'].shape[1]}")

    for L in args.learners:
        fn = {"tabpfn": lambda: run_tabpfn(data, args.device),
              "xgb": lambda: run_xgb(data)}[L]
        res = fn()
        v, t = res["val"], res["test"]
        print(f"\n  [{L}]  {args.tag}{'  (+aug)' if args.augment else ''}")
        print(f"    val : r={v['r']:.4f}  rho={v['rho']:.4f}  rmse={v['rmse']:.4f}  (n={v['n']})")
        print(f"    test: r={t['r']:.4f}  rho={t['rho']:.4f}  rmse={t['rmse']:.4f}  (n={t['n']})")


if __name__ == "__main__":
    main()
