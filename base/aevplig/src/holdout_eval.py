"""holdout_eval.py — AEV-PLIG on the 2019 temporal holdout, PREDICT-ONLY.

Reuses the 3 lp_edrscc_v2-trained checkpoints (output/casf_ckpts/aevplig_casf_seed{S}.model);
no retraining. Loads the pre-built holdout graphs (aevplig_holdout2019_graphs.pickle, built by
build_graphs.py --split holdout2019_eval with the robust 3-tier ligand reader → 694/704), fits the
same StandardScaler on lp_edrscc_v2 train pK the checkpoints were trained with, and predicts.

Writes base/_casf/AEV_holdout2019_preds.csv (pid,pK_truth,in_v2train,pred_seed0/1/2,pred_ensemble)
— the exact columns common94_holdout.py's aev() reads — plus AEV_holdout2019.json.

Runs on CPU in seconds (GATv2 inference, ~700 graphs). Usage:
  cd base/aevplig/src && python holdout_eval.py [--device cpu] [--seeds 0 1 2]
"""
import os, sys, json, pickle, argparse
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model_defs import GATv2Net
from utils import GraphDataset, init_weights                       # noqa: F401
from helpers import rmse, pearson, spearman
from casf_eval import Cfg, predict, BASE_DIR, REPO_DIR, SPLITS_DIR, LP_CSV, PROC_ROOT, CASF_OUT

EDRSCC_GRAPHS  = os.path.join(BASE_DIR, "graphs", "aevplig_edrscc_graphs.pickle")
HOLDOUT_GRAPHS = os.path.join(BASE_DIR, "graphs", "aevplig_holdout2019_graphs.pickle")
CKPT_DIR       = os.path.join(BASE_DIR, "output", "casf_ckpts")
HOLDOUT_CSV    = os.path.join(SPLITS_DIR, "holdout2019_eval.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    print(f"device: {device}")

    # ── graphs: train/val from the edrscc pickle, holdout from the rebuilt pickle ──
    with open(EDRSCC_GRAPHS, "rb") as h:
        graphs = pickle.load(h)
    with open(HOLDOUT_GRAPHS, "rb") as h:
        graphs.update(pickle.load(h))
    print(f"graphs: {len(graphs)} total (edrscc + holdout)")

    # ── labels ──────────────────────────────────────────────────────────────────
    lp = pd.read_csv(LP_CSV).rename(columns={"Unnamed: 0": "pdb_id", "value": "pK"})
    lp["pdb_id"] = lp["pdb_id"].astype(str).str.lower()
    pk_lp = dict(zip(lp["pdb_id"], lp["pK"]))

    sp = pd.read_csv(os.path.join(SPLITS_DIR, "lp_edrscc_v2.csv"))
    sp["pid"] = sp["pid"].astype(str).str.lower()
    train_ids = [p for p in sp[sp["split"] == "train"]["pid"]
                 if p in graphs and p in pk_lp and pd.notna(pk_lp[p])]

    ho = pd.read_csv(HOLDOUT_CSV)
    ho["pid"] = ho["pid"].astype(str).str.lower()
    ho_pk = dict(zip(ho["pid"], ho["pK"]))
    holdout_ids = [p for p in ho["pid"] if p in graphs and p in ho_pk and pd.notna(ho_pk[p])]
    print(f"train={len(train_ids)}  holdout evaluable={len(holdout_ids)}/{len(ho)}")

    # ── datasets (train only to fit the SAME y_scaler the checkpoints used) ───────
    TAG = "holdout_eval"
    proc = os.path.join(PROC_ROOT, TAG, "processed")
    for nm in (f"{TAG}_train", f"{TAG}_holdout"):
        pt = os.path.join(proc, f"{nm}.pt")
        if os.path.exists(pt):
            os.remove(pt)
    root = os.path.join(PROC_ROOT, TAG)
    train_data = GraphDataset(root=root, dataset=f"{TAG}_train", ids=train_ids,
                              y=[pk_lp[p] for p in train_ids], graphs_dict=graphs, y_scaler=None)
    holdout_data = GraphDataset(root=root, dataset=f"{TAG}_holdout", ids=holdout_ids,
                                y=[ho_pk[p] for p in holdout_ids], graphs_dict=graphs,
                                y_scaler=train_data.y_scaler)
    loader = DataLoader(holdout_data, batch_size=args.batch_size, shuffle=False)

    # ── per-seed predict from the saved checkpoints (NO training) ─────────────────
    all_preds, truth_ref = [], None
    for seed in args.seeds:
        ckpt = os.path.join(CKPT_DIR, f"aevplig_casf_seed{seed}.model")
        if not os.path.exists(ckpt):
            print(f"  seed{seed}: MISSING checkpoint {ckpt} — skipping"); continue
        model = GATv2Net(node_feature_dim=train_data.num_node_features,
                         edge_feature_dim=train_data.num_edge_features, config=Cfg()).to(device)
        model.load_state_dict(torch.load(ckpt, weights_only=True, map_location=device))
        G, P = predict(model, device, loader, train_data.y_scaler)
        truth_ref = G
        all_preds.append(P)
        print(f"  seed{seed}: r={pearson(G, P):.4f} rho={spearman(G, P):.4f} rmse={rmse(G, P):.4f} (ckpt reused)")

    # ── write preds CSV (common94 aev() reads pred_seed* cols) + json ─────────────
    os.makedirs(CASF_OUT, exist_ok=True)
    pred_df = pd.DataFrame({"pid": holdout_ids, "pK_truth": truth_ref,
                            "in_v2train": [0] * len(holdout_ids)})
    for seed, P in zip(args.seeds[:len(all_preds)], all_preds):
        pred_df[f"pred_seed{seed}"] = P
    pred_df["pred_ensemble"] = np.mean(np.stack(all_preds), axis=0)
    csv_out = os.path.join(CASF_OUT, "AEV_holdout2019_preds.csv")
    pred_df.to_csv(csv_out, index=False)

    def ms(metric):
        vals = np.array([metric(truth_ref, P) for P in all_preds])
        return {"mean": float(vals.mean()), "std": float(vals.std())}
    summary = {"model": "AEV-PLIG", "set": "holdout2019 (predict-only, ckpt reused)",
               "seeds": args.seeds, "n": len(holdout_ids),
               "pearson": ms(pearson), "spearman": ms(spearman), "rmse": ms(rmse)}
    json.dump(summary, open(os.path.join(CASF_OUT, "AEV_holdout2019.json"), "w"), indent=2)
    print(f"\nsaved -> {csv_out}  (n={len(holdout_ids)}, was 443)")
    print(f"[AEV holdout raw n={len(holdout_ids)}] r={summary['pearson']['mean']:.4f}±{summary['pearson']['std']:.4f}  "
          f"rho={summary['spearman']['mean']:.4f}±{summary['spearman']['std']:.4f}  "
          f"rmse={summary['rmse']['mean']:.4f}±{summary['rmse']['std']:.4f}")
    print("NOTE: Table-3 number = common94_holdout.py common-ED subset (PLINDER-v2 excluded, intersected).")


if __name__ == "__main__":
    main()
