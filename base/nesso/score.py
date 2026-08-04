"""score.py — collect Nesso predictions and compute metrics for LP-edrscc splits + CASF.

Usage:
    python score.py [--out_dir _edrscc/outputs] [--sign {pos,neg,auto}]

Reads:
  - _edrscc/outputs/predictions/{pid}/affinity.json  (affinity_pred_value field)
  - LP_PDBBind.csv for pK labels
  - voxbind/splits/lp_edrscc_v2*.csv for split membership
  - voxbind/splits/casf2016_eval.csv for CASF

Writes:
  - _edrscc/predictions_v2.csv         (pid, pred, pK)
  - _edrscc/results_Nesso_<split>.json  for each split
  - base/_casf/Nesso.json              (CASF metrics)
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

HERE = Path(__file__).parent
REPO = HERE.parent.parent  # VoxBind/
LP_CSV = REPO / "voxbind/dataset/data/pdbbind/raw/LP_PDBBind.csv"
SPLITS_DIR = REPO / "voxbind/splits"
CASF_DIR = REPO / "base/_casf"

SIGN_NOTE = (
    "zero-shot cofolding; leaked (PDBbind in Nesso training); "
    "sign flipped: Nesso output = log10(IC50/uM) lower=stronger, "
    "negated to pred = -log10(IC50/uM); RMSE computed on pIC50 = pred + 6 "
    "(uM->M decade offset: pIC50 = -log10(IC50/M) = pred + 6) so it matches the pK scale; "
    "Pearson/Spearman are offset-invariant and unchanged by the +6; "
    "residual RMSE offset still expected (Nesso predicts IC50-like potency, labels are Kd/Ki)"
)

SIGN_NOTE_POS = (
    "zero-shot cofolding; leaked (PDBbind in Nesso training); "
    "Nesso output = log10(IC50/uM), positive correlation with pK directly; "
    "RMSE computed on pIC50 = pred + 6 to match pK scale"
)

# uM -> M decade offset: pIC50 = -log10(IC50/M) = -log10(IC50/uM) + 6 = pred + 6
PIC50_OFFSET = 6.0


def load_preds(out_dir: Path):
    """Return dict {pid: affinity_pred_value} from affinity.json files."""
    preds = {}
    pred_dir = out_dir / "predictions"
    if not pred_dir.exists():
        raise FileNotFoundError(f"No predictions dir at {pred_dir}")
    for pid_dir in pred_dir.iterdir():
        if not pid_dir.is_dir():
            continue
        aff_file = pid_dir / "affinity.json"
        if not aff_file.exists():
            continue
        with open(aff_file) as f:
            data = json.load(f)
        val = data.get("affinity_pred_value")
        if val is not None and np.isfinite(val):
            preds[pid_dir.name.lower()] = float(val)
    return preds


def compute_metrics(preds_arr, labels_arr):
    """Return dict with pearson r, spearman rho, rmse.

    preds_arr is the signed pred = -log10(IC50/uM). Pearson/Spearman are
    offset-invariant so they use preds_arr directly. RMSE is computed on the
    pIC50 scale (pred + PIC50_OFFSET) so it is comparable to the pK labels.
    """
    preds_arr = np.asarray(preds_arr, dtype=float)
    labels_arr = np.asarray(labels_arr, dtype=float)
    r, _ = stats.pearsonr(labels_arr, preds_arr)
    rho, _ = stats.spearmanr(labels_arr, preds_arr)
    pic50 = preds_arr + PIC50_OFFSET
    rmse = float(np.sqrt(np.mean((pic50 - labels_arr) ** 2)))
    return {"pearson": float(r), "spearman": float(rho), "rmse": rmse, "n": len(preds_arr)}


def result_json(metrics, note, split_key="v2"):
    return {
        "pearson": {"mean": round(metrics["pearson"], 4), "std": 0.0},
        "spearman": {"mean": round(metrics["spearman"], 4), "std": 0.0},
        "rmse": {"mean": round(metrics["rmse"], 4), "std": 0.0},
        "n_test": metrics["n"],
        "seeds": [0],
        "note": note,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="_edrscc/outputs",
                        help="Directory containing predictions/ subdirectory")
    parser.add_argument("--sign", default="auto", choices=["pos", "neg", "auto"],
                        help="pos: use raw value; neg: negate; auto: determine from r sign on v2 test")
    args = parser.parse_args()

    out_dir = HERE / args.out_dir

    # Load raw predictions
    print("[score] Loading predictions...")
    preds_raw = load_preds(out_dir)
    print(f"[score] Loaded {len(preds_raw)} predictions")

    # Load labels (LP_PDBBind has unnamed first column = pid)
    lp = pd.read_csv(LP_CSV, index_col=0)
    lp.index = lp.index.str.lower()
    pk_map = lp["value"].to_dict()
    smi_map = lp["smiles"].to_dict()

    # Load v2 test pids
    v2_df = pd.read_csv(SPLITS_DIR / "lp_edrscc_v2.csv")
    v2_test = set(v2_df[v2_df["split"] == "test"]["pid"].str.lower())

    # Determine sign convention from v2 test set
    v2_pids_ok = [(p, preds_raw[p], pk_map[p])
                  for p in v2_test
                  if p in preds_raw and p in pk_map and np.isfinite(pk_map[p])]
    if not v2_pids_ok:
        raise RuntimeError("No overlapping pids between predictions and v2 test set!")

    v2_pids_list, v2_pred_arr, v2_label_arr = zip(*v2_pids_ok)
    v2_pred_arr = np.array(v2_pred_arr)
    v2_label_arr = np.array(v2_label_arr)

    r_raw, _ = stats.pearsonr(v2_label_arr, v2_pred_arr)
    print(f"[score] v2 raw Pearson r = {r_raw:.4f} (before sign fix)")

    if args.sign == "auto":
        # If r < 0, flip sign so positive correlation is reported
        negate = (r_raw < 0)
        print(f"[score] auto sign: {'negating' if negate else 'keeping positive'}")
    elif args.sign == "neg":
        negate = True
    else:
        negate = False

    note = SIGN_NOTE if negate else SIGN_NOTE_POS

    def signed_pred(raw):
        return -raw if negate else raw

    # Build per-pid prediction series
    all_records = []
    for pid, raw_val in preds_raw.items():
        pk = pk_map.get(pid)
        if pk is not None and np.isfinite(pk):
            sp = signed_pred(raw_val)
            all_records.append({
                "pid": pid,
                "pred_raw": raw_val,
                "pred": sp,
                "pred_pIC50": sp + PIC50_OFFSET,
                "pK": float(pk),
            })
    preds_df = pd.DataFrame(all_records)
    print(f"[score] Records with label: {len(preds_df)}")

    # Save predictions for v2 test only (canonical CSV): raw signed pred + pIC50-shifted
    v2_df_out = preds_df[preds_df["pid"].isin(v2_test)].copy()
    v2_df_out = v2_df_out[["pid", "pred", "pred_pIC50", "pK"]]
    out_csv = HERE / "_edrscc/predictions_v2.csv"
    v2_df_out.to_csv(out_csv, index=False)
    print(f"[score] Saved {len(v2_df_out)} v2 predictions → {out_csv}")

    # --- LP-edrscc tier metrics ---
    splits_cfg = {
        "lp_edrscc_v2": SPLITS_DIR / "lp_edrscc_v2.csv",
        "lp_edrscc_v2_cl1": SPLITS_DIR / "lp_edrscc_v2_cl1.csv",
        "lp_edrscc_v2_cl12": SPLITS_DIR / "lp_edrscc_v2_cl12.csv",
        "lp_edrscc_v2_cl123": SPLITS_DIR / "lp_edrscc_v2_cl123.csv",
    }

    for split_name, split_csv in splits_cfg.items():
        if not split_csv.exists():
            print(f"[score] WARNING: {split_csv} not found, skipping")
            continue
        df = pd.read_csv(split_csv)
        test_pids = set(df[df["split"] == "test"]["pid"].str.lower())
        sub = preds_df[preds_df["pid"].isin(test_pids)]
        if len(sub) == 0:
            print(f"[score] WARNING: no predictions for {split_name}")
            continue

        preds_sub = sub["pred"].values
        labels_sub = sub["pK"].values
        m = compute_metrics(preds_sub, labels_sub)
        print(f"[score] {split_name}: n={m['n']}, pearson={m['pearson']:.4f}, "
              f"spearman={m['spearman']:.4f}, rmse={m['rmse']:.4f}")

        out_json = HERE / f"_edrscc/results_Nesso_{split_name}.json"
        with open(out_json, "w") as f:
            json.dump(result_json(m, note), f, indent=2)
        print(f"[score] Wrote {out_json}")

    # --- CASF metrics ---
    casf_csv = SPLITS_DIR / "casf2016_eval.csv"
    if casf_csv.exists():
        casf_df = pd.read_csv(casf_csv)
        # casf_df columns: pid, in_v2train, pK
        casf_df["pid"] = casf_df["pid"].str.lower()

        # Use pK from casf_df (its own column) not LP_PDBBind
        casf_preds = []
        for _, row in casf_df.iterrows():
            pid = row["pid"]
            pk = float(row["pK"])
            raw = preds_raw.get(pid)
            if raw is not None and np.isfinite(raw):
                casf_preds.append({
                    "pid": pid,
                    "pred": signed_pred(raw),
                    "pK": pk,
                    "in_v2train": int(row["in_v2train"]),
                })
        casf_preds_df = pd.DataFrame(casf_preds)
        print(f"[score] CASF: {len(casf_preds_df)}/214 predictions available")

        leaky_all = casf_preds_df
        nontrain = casf_preds_df[casf_preds_df["in_v2train"] == 0]

        def casf_metrics(sub):
            if len(sub) == 0:
                return {"pearson": {"mean": None, "std": 0.0},
                        "spearman": {"mean": None, "std": 0.0},
                        "rmse": {"mean": None, "std": 0.0},
                        "n": 0}
            m = compute_metrics(sub["pred"].values, sub["pK"].values)
            return {
                "pearson": {"mean": round(m["pearson"], 4), "std": 0.0},
                "spearman": {"mean": round(m["spearman"], 4), "std": 0.0},
                "rmse": {"mean": round(m["rmse"], 4), "std": 0.0},
                "n": m["n"],
            }

        casf_result = {
            "model": "Nesso-1",
            "note": note + "; CASF leaky/nontrain is w.r.t. our v2-train; Nesso itself trained on both",
            "leaky": casf_metrics(leaky_all),
            "nontrain": casf_metrics(nontrain),
        }

        casf_out = CASF_DIR / "Nesso.json"
        CASF_DIR.mkdir(exist_ok=True)
        with open(casf_out, "w") as f:
            json.dump(casf_result, f, indent=2)
        print(f"[score] Wrote CASF → {casf_out}")
        print(f"[score] CASF leaky (n={casf_result['leaky']['n']}): "
              f"pearson={casf_result['leaky']['pearson']['mean']}, "
              f"spearman={casf_result['leaky']['spearman']['mean']}, "
              f"rmse={casf_result['leaky']['rmse']['mean']}")
        print(f"[score] CASF nontrain (n={casf_result['nontrain']['n']}): "
              f"pearson={casf_result['nontrain']['pearson']['mean']}, "
              f"spearman={casf_result['nontrain']['spearman']['mean']}, "
              f"rmse={casf_result['nontrain']['rmse']['mean']}")

    print("[score] Done.")


if __name__ == "__main__":
    main()
